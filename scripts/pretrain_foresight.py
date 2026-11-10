"""AUGUR-I foresight pretraining (LOG-005): the treatment that precedes fine-tuning.

Trains only the foresight pathway of SmolVLA, with no action loss:
  LoRA on the VLM text layers (same placement as scripts/train_augur.py), the K foresight
  queries and the prediction head. The vision encoder, the VLM base weights and the action
  expert are untouched. Pretraining is the same for the -TR and -TT modes of a target; the
  TT conditioning projection is created and trained only in fine-tuning.

Data (each step draws a fixed number of samples from each source):
  hr  human + robot: SSv2 frames (registered 50k-clip subset, 2 fps) + LIBERO main camera
  r   robot only:    LIBERO only, same batch size and number of steps
Inputs
  LIBERO  both cameras, state, task text -- exactly the fine-tuning inputs.
  SSv2    the frame as the main camera; the wrist camera is masked out (padding mask
          False, so its tokens are dropped from attention); state = 0 in normalised space
          (the dataset mean, as with state dropout); the SSv2 label as the task text.
  Base augmentation (LOG-003 section 5) on the inputs only; targets come from the cache of
  un-augmented frames. Only samples whose true 1-second future exists are drawn.
Targets (scripts/augur_foresight.py, cache from scripts/foresight_cache.py)
  t3  future embedding -> VLM text layers with EMA LoRA (tau 0.996) -> quadrant pool -> LN
  t5  RAFT flow tokens standardised with flow_stats.json
Validation: 1 in 50 LIBERO episodes and 1 in 50 SSv2 clips are excluded from training;
  a fixed set of up to 512 samples per source is scored every --val_every steps (loss and
  foresight error, i.e. cosine distance), no augmentation.
Output  <out>/{config.json, log.csv, step_XXXXXX.pt, final.pt}; a checkpoint holds the
  online LoRA tensors, the EMA LoRA tensors and the Foresight state dict.

Usage
  python scripts/pretrain_foresight.py --target t3 --data hr --out runs/pretrain/t3_hr_s0 \
      --cache ~/augur/data/foresight_cache --frames ~/augur/data/ssv2_frames
  smoke test: add --steps 30 --val_every 15 --save_every 30 --batch 8 --workers 2
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--target", choices=["t3", "t5"], required=True)
p.add_argument("--data", choices=["hr", "r"], required=True)
p.add_argument("--out", required=True)
p.add_argument("--cache", default="~/augur/data/foresight_cache")
p.add_argument("--frames", default="~/augur/data/ssv2_frames")
p.add_argument("--subset", default="env/ssv2_subset_50k.json")
p.add_argument("--repo_id", default="lerobot/libero")
p.add_argument("--k", type=int, default=4)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch", type=int, default=64, help="total per step; hr splits it half/half")
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--min_lr", type=float, default=2.5e-6)
p.add_argument("--grad_clip", type=float, default=10.0)
p.add_argument("--ema_tau", type=float, default=0.996)
p.add_argument("--val_every", type=int, default=1000)
p.add_argument("--save_every", type=int, default=5000)
p.add_argument("--n_val", type=int, default=512)
p.add_argument("--workers", type=int, default=6)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

os.environ.setdefault("AUGUR_AUG", "base")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cache = Path(args.cache).expanduser()
frames_dir = Path(args.frames).expanduser()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if dev == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

import augur_foresight as AF  # noqa: E402
import train_augur as TA  # noqa: E402  (LoRA, augmentation; importing sets the bf16 autocast default)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402
import lerobot.policies.smolvla.modeling_smolvla as M  # noqa: E402

CAM, CAM2, STATE = "observation.images.image", "observation.images.image2", "observation.state"


# ----------------------------------------------------------------------------- data
class CacheReader:
    """Lazy per-worker memmaps of one source's cache."""

    def __init__(self, src):
        self.src, self._emb, self._flow = src, None, None
        self.future = np.load(cache / f"{src}_future.npy")
        self.valid = np.load(cache / f"{src}_valid.npy")

    def target(self, row):
        if args.target == "t3":
            if self._emb is None:
                self._emb = np.load(cache / f"{self.src}_emb.npy", mmap_mode="r")
            return torch.from_numpy(np.array(self._emb[self.future[row]]))  # (64, D) fp16
        if self._flow is None:
            self._flow = np.load(cache / f"{self.src}_flow.npy", mmap_mode="r")
        return torch.from_numpy(np.array(self._flow[row]))  # (4, 32) fp16


class LiberoFrames(Dataset):
    def __init__(self, ds, rows, reader):
        self.ds, self.rows, self.reader = ds, rows, reader

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        r = int(self.rows[j])
        it = self.ds[r]
        assert int(it["index"]) == r
        return {CAM: it[CAM], CAM2: it[CAM2], STATE: it[STATE], "task": it["task"],
                "tgt": self.reader.target(r), "src": 0}


class SSv2Frames(Dataset):
    def __init__(self, rows, clip_ids, offsets, labels, reader):
        self.rows, self.clip_ids, self.offsets, self.labels, self.reader = rows, clip_ids, offsets, labels, reader

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        from PIL import Image

        r = int(self.rows[j])
        ci = int(np.searchsorted(self.offsets, r, side="right") - 1)
        cid, k = str(self.clip_ids[ci]), r - int(self.offsets[ci])
        im = np.asarray(Image.open(frames_dir / cid[-2:] / f"{cid}_{k:02d}.jpg").convert("RGB"))
        img = torch.from_numpy(im.copy()).permute(2, 0, 1).float() / 255.0
        return {CAM: img, "task": self.labels[cid], "tgt": self.reader.target(r), "src": 1}


def forever(dl):
    while True:
        for b in dl:
            yield b


ds = LeRobotDataset(args.repo_id)
lib_reader = CacheReader("libero")
assert len(lib_reader.future) == len(ds), "LIBERO cache does not match the dataset"
n_filled = int((lib_reader.future >= 0).sum())
if n_filled < len(ds):
    print(f"WARNING: LIBERO cache filled for {n_filled}/{len(ds)} frames (partial cache: smoke tests only)", flush=True)
_eps = ds.meta.episodes
_fr, _to = np.asarray(_eps["dataset_from_index"]), np.asarray(_eps["dataset_to_index"])
ep_of_row = np.repeat(np.arange(len(_fr)), _to - _fr)
assert len(ep_of_row) == len(ds)
lib_ok = lib_reader.valid
lib_train = np.flatnonzero(lib_ok & (ep_of_row % 50 != 0))
lib_val_all = np.flatnonzero(lib_ok & (ep_of_row % 50 == 0))

n_h = args.batch // 2 if args.data == "hr" else 0
n_r = args.batch - n_h
loaders = {"libero": forever(DataLoader(LiberoFrames(ds, lib_train, lib_reader), batch_size=n_r, shuffle=True,
                                        drop_last=True, num_workers=args.workers, persistent_workers=args.workers > 0))}
rng = np.random.default_rng(args.seed)
val_sets = {"libero": LiberoFrames(ds, np.sort(rng.choice(lib_val_all, min(args.n_val, len(lib_val_all)), replace=False)), lib_reader)}

# SSv2 is always loaded for validation (the robot-only model is also scored on human video)
ss_reader = CacheReader("ssv2")
rows_npz = np.load(cache / "ssv2_rows.npz")
clip_ids, offsets = rows_npz["clip_ids"], rows_npz["offsets"]
labels = {c["id"]: c["label"] for c in json.load(open(args.subset))["clips"]}
clip_of_row = np.repeat(np.arange(len(clip_ids)), np.diff(offsets))
ss_ok = ss_reader.valid
ss_train = np.flatnonzero(ss_ok & (clip_of_row % 50 != 0))
ss_val_all = np.flatnonzero(ss_ok & (clip_of_row % 50 == 0))
val_sets["ssv2"] = SSv2Frames(np.sort(rng.choice(ss_val_all, min(args.n_val, len(ss_val_all)), replace=False)),
                              clip_ids, offsets, labels, ss_reader)
if n_h:
    loaders["ssv2"] = forever(DataLoader(SSv2Frames(ss_train, clip_ids, offsets, labels, ss_reader), batch_size=n_h,
                                         shuffle=True, drop_last=True, num_workers=max(1, args.workers // 2),
                                         persistent_workers=True))
print(f"libero train {len(lib_train)} val {len(val_sets['libero'])} | ssv2 train {len(ss_train)} "
      f"val {len(val_sets['ssv2'])} | per step: ssv2 {n_h} + libero {n_r}", flush=True)

# ----------------------------------------------------------------------------- model
cfg = SmolVLAConfig(load_vlm_weights=True, device=dev)
policy = make_policy(cfg, ds_meta=ds.meta)
TA.inject_lora(policy)
fs = AF.install(policy, args.target, "tr", k=args.k)
fs.ema_tau = args.ema_tau
if args.target == "t5":
    st = json.load(open(cache / "flow_stats.json"))
    fs.tgt_mean.copy_(torch.tensor(st["mean"]))
    fs.tgt_std.copy_(torch.tensor(st["std"]))
pre, _ = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds.meta.stats)
m = policy.model
vwe = m.vlm_with_expert
for n_, p_ in policy.named_parameters():
    p_.requires_grad = (".lora_" in n_) or n_.startswith("model.augur.")
params = [p_ for p_ in policy.parameters() if p_.requires_grad]
print(f"trainable {sum(p_.numel() for p_ in params):,} params "
      f"(LoRA {sum(p_.numel() for n_, p_ in policy.named_parameters() if '.lora_' in n_):,})", flush=True)
policy.train()
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(1.0, t)))


def to_model_batch(raw):
    """raw collated batch (one source) -> preprocessed dict with the keys the model needs."""
    b = len(raw["task"])
    ss = CAM2 not in raw
    x = {CAM: raw[CAM].float(), "task": list(raw["task"])}
    if ss:
        x[CAM2] = torch.zeros_like(x[CAM])
        x[STATE] = torch.zeros(b, ds.meta.features[STATE]["shape"][0])
    else:
        x[CAM2] = raw[CAM2].float()
        x[STATE] = raw[STATE].float()
    x["action"] = torch.zeros(b, ds.meta.features["action"]["shape"][0])
    y = pre(x)
    if ss:
        y[STATE] = torch.zeros_like(y[STATE])  # normalised space: the dataset mean
    y[f"{CAM2}_padding_mask"] = torch.full((b,), not ss, dtype=torch.bool, device=dev)
    return {k: y[k] for k in (CAM, CAM2, f"{CAM2}_padding_mask", STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)}


def concat(parts):
    if len(parts) == 1:
        return parts[0]
    lt = max(q[OBS_LANGUAGE_TOKENS].shape[1] for q in parts)
    for q in parts:  # pad_language_to = "longest": pad each part to the common length (masked)
        d = lt - q[OBS_LANGUAGE_TOKENS].shape[1]
        if d:
            q[OBS_LANGUAGE_TOKENS] = F.pad(q[OBS_LANGUAGE_TOKENS], (0, d))
            q[OBS_LANGUAGE_ATTENTION_MASK] = F.pad(q[OBS_LANGUAGE_ATTENTION_MASK], (0, d))
    return {k: torch.cat([q[k] for q in parts]) for k in parts[0]}


def predict(batch):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    pe, pp, pa = m.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state)
    att2d = M.make_att_2d_masks(pp, pa)
    pos = torch.cumsum(pp, dim=1) - 1
    # prefix-only passes must use the cache-filling path: with use_cache=False the cross-attention
    # layers expect expert tokens. (Same call as augur_foresight.prefix_only; gradients still flow.)
    outs, _ = vwe.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                          inputs_embeds=[pe, None], use_cache=True)
    return fs.predict(outs[0][:, fs._q0: fs._q0 + args.k])


def make_target(tgt):
    tgt = tgt.to(dev)
    if args.target == "t3":
        with torch.no_grad():
            return AF.t3_target(policy, tgt).float()
    return AF.standardise(fs, tgt)


def autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")


@torch.no_grad()
def validate():
    policy.eval()
    res = {}
    for src, vset in val_sets.items():
        dl = DataLoader(vset, batch_size=32, shuffle=False, num_workers=2)
        losses, errs = [], []
        for raw in dl:
            batch = to_model_batch(raw)
            with autocast():
                pred = predict(batch).float()
                tgt = make_target(raw["tgt"])
            losses.append(((pred - tgt) ** 2).mean(dim=(1, 2)).cpu())
            errs.append(AF.foresight_error(pred, tgt).cpu())
        res[f"val_{src}_loss"] = float(torch.cat(losses).mean())
        res[f"val_{src}_ferr"] = float(torch.cat(errs).mean())
    policy.train()
    return res


def save(name, step):
    sd = {
        "lora": {n_: p_.detach().float().cpu() for n_, p_ in policy.named_parameters() if ".lora_" in n_},
        "ema": {n_: t.detach().float().cpu() for n_, t in fs.ema.items()},
        "augur": {n_: t.detach().cpu() for n_, t in fs.state_dict().items()},
        "step": step, "args": vars(args),
    }
    torch.save(sd, out / name)


json.dump({**vars(args), "aug": os.environ["AUGUR_AUG"], "lora_r": TA.LORA_R, "lora_alpha": TA.LORA_ALPHA,
           "lora_targets": TA.LORA_TARGETS, "n_libero_train": int(len(lib_train)), "n_ssv2_train": int(len(ss_train))},
          open(out / "config.json", "w"), indent=1)
fs.ema_init(m)
log_f = open(out / "log.csv", "w", newline="")
log = csv.writer(log_f)
log.writerow(["step", "lr", "loss", "loss_libero", "loss_ssv2", "grad_norm", "s_per_step",
              "val_libero_loss", "val_libero_ferr", "val_ssv2_loss", "val_ssv2_ferr"])
print("step 0 validation:", validate(), flush=True)
t0, acc = time.time(), []
for step in range(args.steps):
    raws = [next(loaders[s]) for s in (["ssv2"] if n_h else []) + ["libero"]]
    batch = TA.augment_batch(concat([to_model_batch(r) for r in raws]))
    tgt_raw = torch.cat([r["tgt"] for r in raws])
    src = torch.cat([r["src"] for r in raws]).to(dev)
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    with autocast():
        tgt = make_target(tgt_raw)
        pred = predict(batch).float()
    per = ((pred - tgt) ** 2).mean(dim=(1, 2))
    loss = per.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
    opt.step()
    if args.target == "t3":
        fs.ema_update(m)
    acc.append((float(loss), float(per[src == 0].mean()), float(per[src == 1].mean()) if n_h else float("nan"), gn))
    s1 = step + 1
    if s1 % 50 == 0 or s1 == args.steps:
        a = np.array(acc)
        acc = []
        sps = (time.time() - t0) / s1
        row = [s1, lr_at(step), *np.nanmean(a, 0).round(5)[:3], round(float(a[:, 3].mean()), 3), round(sps, 3)]
        v = validate() if (s1 % args.val_every == 0 or s1 == args.steps) else {}
        row += [round(v[k], 5) if k in v else "" for k in ("val_libero_loss", "val_libero_ferr", "val_ssv2_loss", "val_ssv2_ferr")]
        log.writerow(row)
        log_f.flush()
        eta = sps * (args.steps - s1) / 3600
        print(f"step {s1}/{args.steps} loss {row[2]:.4f} (libero {row[3]:.4f} ssv2 {row[4]:.4f}) gn {row[5]} "
              f"{sps:.2f}s/step eta {eta:.1f}h" + (f" | {v}" if v else ""), flush=True)
    if s1 % args.save_every == 0 and s1 != args.steps:
        save(f"step_{s1:06d}.pt", s1)
save("final.pt", args.steps)
print("done", out / "final.pt", flush=True)
