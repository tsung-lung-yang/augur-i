"""AUGUR-I training entry point (T0 and, later, the foresight conditions).

Runs LeRobot's lerobot-train unchanged except for three hooks:
  1. LoRA on the VLM text layers (attention projections) of SmolVLA. The vision
     encoder stays frozen; the action expert trains as in the stock recipe.
     Checkpoints are saved with LoRA merged into the base weights, so a saved
     model is a plain SmolVLA that lerobot-eval / scripts/eval_mixed.py can load.
  2. Base augmentation from LOG-003 section 5, applied on GPU to every training
     batch after LeRobot's preprocessor: random shift (replicate-pad then crop),
     colour jitter (brightness, contrast, saturation, hue, in that fixed order),
     each camera sampled independently; state dropout sets the normalised state
     vector of a sample to 0 (the dataset mean) with probability p.
  3. Optional foresight (LOG-005; AUGUR_TARGET != none): installs scripts/augur_foresight.py,
     optionally initialised from a foresight pretraining checkpoint, and adds
     AUGUR_LAMBDA * foresight loss to the action loss. Targets come from the cache of
     scripts/foresight_cache.py, looked up by the dataset row index of each sample (the
     preprocessor is wrapped to carry it through). For t3 the EMA LoRA is updated after
     every optimiser step. Saved checkpoints stay plain merged SmolVLA weights plus
     augur.safetensors (foresight module) and augur_lora.safetensors (online/EMA LoRA);
     scripts/eval_mixed.py installs the module when augur_train.json names a target.
  Data, optimiser, LR schedule and checkpointing are LeRobot's.

Environment variables (defaults are the registered settings):
  AUGUR_AUG          base | strong | none      (LOG-003 sections 5 and 6)
  AUGUR_LORA_R       LoRA rank, default 16
  AUGUR_LORA_ALPHA   LoRA alpha, default 2 * rank
  AUGUR_LORA_TARGETS comma list, default q_proj,k_proj,v_proj,o_proj
  AUGUR_TARGET       none | t3 | t5                (default none = T0)
  AUGUR_MODE         tr | tt                       (default tr)
  AUGUR_K            foresight tokens, default 4
  AUGUR_LAMBDA       foresight loss weight, default 0.1
  AUGUR_INIT         foresight pretraining checkpoint (final.pt), default none
  AUGUR_CACHE        foresight cache directory, default ~/augur/data/foresight_cache
All other arguments are lerobot-train arguments, e.g.
  python scripts/train_augur.py --policy.type=smolvla --policy.load_vlm_weights=true \
    --policy.push_to_hub=false --policy.use_amp=true --dataset.repo_id=lerobot/libero \
    --dataset.video_backend=torchcodec --batch_size=32 --steps=60000 --save_freq=5000 \
    --log_freq=100 --num_workers=8 --wandb.enable=false --seed=0 --output_dir=runs/train/t0_curve_s0
Resuming from a checkpoint is not supported (saved weights are merged).
"""
import json
import logging
import math
import os
import sys
import types

# LOG-007: in TT runs the CUDA caching allocator's reserved memory creeps to the 12 GB limit and
# the Windows driver then pages GPU memory to system RAM (2-3x slower). expandable_segments made it
# worse under WSL (T5-TT slow from the first steps), so it is not used; instead the cache is
# released every AUGUR_EMPTY_CACHE steps (default 500). Neither changes any computation.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

AUG_PRESETS = {
    "base": dict(pad=12, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, state_p=0.3),
    "strong": dict(pad=24, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, state_p=0.5),
    "none": None,
}
AUG_NAME = os.environ.get("AUGUR_AUG", "base")
AUG = AUG_PRESETS[AUG_NAME]
LORA_R = int(os.environ.get("AUGUR_LORA_R", "16"))
LORA_ALPHA = float(os.environ.get("AUGUR_LORA_ALPHA", str(2 * LORA_R)))
LORA_TARGETS = os.environ.get("AUGUR_LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj").split(",")
FS_TARGET = os.environ.get("AUGUR_TARGET", "none")
FS_MODE = os.environ.get("AUGUR_MODE", "tr")
FS_K = int(os.environ.get("AUGUR_K", "4"))
FS_LAMBDA = float(os.environ.get("AUGUR_LAMBDA", "0.1"))
FS_INIT = os.environ.get("AUGUR_INIT", "")
FS_CACHE = os.path.expanduser(os.environ.get("AUGUR_CACHE", "~/augur/data/foresight_cache"))
assert FS_TARGET in ("none", "t3", "t5") and FS_MODE in ("tr", "tt")
_FS = {}  # runtime state of the foresight module: fs, reader arrays


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Linear):
    """nn.Linear that keeps the original `weight`/`bias` names and adds a low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return out + delta * self.scale

    @torch.no_grad()
    def merged_weight(self):
        w = self.weight.float() + self.scale * (self.lora_B @ self.lora_A)
        return w.to(self.weight.dtype).contiguous()


def inject_lora(policy) -> int:
    text = policy.model.vlm_with_expert.get_vlm_model().text_model
    n = 0
    for layer in text.layers:
        attn = layer.self_attn
        for name in LORA_TARGETS:
            base = getattr(attn, name)
            if isinstance(base, LoRALinear):
                continue
            setattr(attn, name, LoRALinear(base, LORA_R, LORA_ALPHA))
            n += 1
    return n


def _save_merged(self, save_directory, state_dict=None):
    """Save a plain SmolVLA state dict: LoRA merged into weight, lora_* tensors dropped."""
    sd = {k: v for k, v in self.state_dict().items()
          if ".lora_A" not in k and ".lora_B" not in k and not k.startswith("model.augur.")}
    for name, m in self.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{name}.weight"] = m.merged_weight()
    sd = {k: v.detach().contiguous() for k, v in sd.items()}
    type(self)._save_pretrained(self, save_directory, state_dict=sd)
    info = {"aug": AUG_NAME, "aug_params": AUG, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_targets": LORA_TARGETS, "lora_merged": True, "target": "none"}
    if "fs" in _FS:
        import augur_foresight as AF

        info = {**info, **AF.save_augur(_FS["fs"], self.model, save_directory,
                                        {"lambda": FS_LAMBDA, "init": FS_INIT or None, "cache": FS_CACHE})}
    with open(os.path.join(str(save_directory), "augur_train.json"), "w") as f:
        json.dump(info, f, indent=1)


# ----------------------------------------------------------------------------- augmentation
def _shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    n, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    oy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = oy[:, None] + torch.arange(h, device=x.device)[None, :]
    cols = ox[:, None] + torch.arange(w, device=x.device)[None, :]
    idx = torch.arange(n, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[idx, rows[:, :, None], cols[:, None, :]]
    return out.permute(0, 3, 1, 2).contiguous()


def _uniform(n, mag, device):
    return 1.0 + (torch.rand(n, device=device) * 2 - 1) * mag


def _gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _jitter(x: torch.Tensor, a: dict) -> torch.Tensor:
    n = x.shape[0]
    f = _uniform(n, a["brightness"], x.device).view(n, 1, 1, 1)
    x = (x * f).clamp(0, 1)
    f = _uniform(n, a["contrast"], x.device).view(n, 1, 1, 1)
    m = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * f + m).clamp(0, 1)
    f = _uniform(n, a["saturation"], x.device).view(n, 1, 1, 1)
    g = _gray(x)
    x = ((x - g) * f + g).clamp(0, 1)
    hs = (torch.rand(n) * 2 - 1) * a["hue"]
    x = torch.stack([TF.adjust_hue(x[i], float(hs[i])) for i in range(n)])
    return x


def augment_batch(batch: dict) -> dict:
    if AUG is None:
        return batch
    for k in list(batch.keys()):
        if not k.startswith("observation.images.") or k.endswith("_padding_mask"):
            continue
        v = batch[k]
        if not torch.is_tensor(v) or v.ndim not in (4, 5):
            continue
        shape = v.shape
        x = v.reshape(-1, *shape[-3:]).float()
        x = _jitter(_shift(x, AUG["pad"]), AUG)
        batch[k] = x.reshape(shape).to(v.dtype)
    s = batch.get("observation.state")
    if torch.is_tensor(s) and AUG["state_p"] > 0:
        drop = torch.rand(s.shape[0], device=s.device) < AUG["state_p"]
        s = s.clone()
        s[drop] = 0
        batch["observation.state"] = s
    return batch


# ----------------------------------------------------------------------------- foresight
def setup_foresight(policy):
    """Install the foresight module, load the pretraining checkpoint, open the target cache."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import augur_foresight as AF

    fs = AF.install(policy, FS_TARGET, FS_MODE, k=FS_K)
    dev = next(policy.parameters()).device
    if FS_TARGET == "t5":
        st = json.load(open(os.path.join(FS_CACHE, "flow_stats.json")))["libero"]
        fs.tgt_mean.copy_(torch.tensor(st["mean"]))
        fs.tgt_std.copy_(torch.tensor(st["std"]))
    msg = "no pretraining"
    if FS_INIT:
        ck = torch.load(FS_INIT, map_location="cpu", weights_only=False)
        named = dict(policy.named_parameters())
        missing = [n for n in named if ".lora_" in n and n not in ck["lora"]]
        assert not missing, f"pretraining checkpoint lacks LoRA tensors: {missing[:3]}"
        with torch.no_grad():
            for n, t in ck["lora"].items():
                named[n].copy_(t)
        res = fs.load_state_dict(ck["augur"], strict=False)
        allowed = {"cond_proj.weight", "cond_proj.bias", "cond_pos"}
        assert set(res.missing_keys) <= allowed and not res.unexpected_keys, res
        fs.ema = {n: t.to(dev) for n, t in ck["ema"].items()} if ck["ema"] else {}
        assert ck["args"]["target"] == FS_TARGET and ck["args"]["k"] == FS_K, "pretraining target/K mismatch"
        msg = f"init {FS_INIT} (step {ck['step']}, data {ck['args']['data']})"
    if not fs.ema:
        fs.ema_init(policy.model)
    rd = {"fut": np.load(os.path.join(FS_CACHE, "libero_future.npy")),
          "valid": np.load(os.path.join(FS_CACHE, "libero_valid.npy"))}
    assert (rd["fut"] >= 0).all(), "LIBERO foresight cache incomplete"
    rd["arr"] = np.load(os.path.join(FS_CACHE, "libero_emb.npy" if FS_TARGET == "t3" else "libero_flow.npy"), mmap_mode="r")
    _FS.update(fs=fs, AF=AF, reader=rd, dev=dev)
    orig_forward = policy.forward

    def forward_with_foresight(self, batch, noise=None, time=None, reduction="mean"):
        loss, d = orig_forward(batch, noise=noise, time=time, reduction=reduction)
        if not self.training or "augur_index" not in batch:
            return loss, d
        tgt, valid = foresight_targets(batch["augur_index"])
        fl = AF.foresight_loss(fs, tgt, valid)
        with torch.no_grad():
            fe = AF.foresight_error(fs.last_pred, tgt)
            d["foresight_loss"] = float(fl)
            d["foresight_err"] = float(fe[valid].mean()) if bool(valid.any()) else float("nan")
        return loss + FS_LAMBDA * fl, d

    policy.forward = types.MethodType(forward_with_foresight, policy)
    return fs, msg


def foresight_targets(index):
    """Dataset row indices (B,) -> (target (B, K, d) float32 on device, valid (B,) bool)."""
    fs, AF, rd, dev = _FS["fs"], _FS["AF"], _FS["reader"], _FS["dev"]
    rows = index.detach().cpu().long().numpy().reshape(-1)
    valid = torch.from_numpy(rd["valid"][rows]).to(dev)
    if FS_TARGET == "t3":
        emb = torch.from_numpy(np.array(rd["arr"][rd["fut"][rows]])).to(dev)
        tgt = AF.t3_target(_FS["policy_ref"], emb).float()
    else:
        tgt = AF.standardise(fs, torch.from_numpy(np.array(rd["arr"][rows])).to(dev))
    return tgt, valid


# ----------------------------------------------------------------------------- hooks
import lerobot.scripts.lerobot_train as T  # noqa: E402

_make_policy = T.make_policy
_update_policy = T.update_policy


def make_policy_with_lora(*args, **kwargs):
    policy = _make_policy(*args, **kwargs)
    n = inject_lora(policy)
    policy._save_pretrained = _save_merged.__get__(policy)
    if FS_TARGET != "none":
        fs, msg = setup_foresight(policy)
        _FS["policy_ref"] = policy
        n_fs = sum(p.numel() for p in fs.parameters())
        print(f"[augur] foresight {FS_TARGET}-{FS_MODE} K={FS_K} lambda={FS_LAMBDA} ({n_fs:,} params); {msg}", flush=True)
        logging.info(f"[augur] foresight {FS_TARGET}-{FS_MODE} K={FS_K} lambda={FS_LAMBDA}; {msg}")
    n_lora = sum(p.numel() for name, p in policy.named_parameters() if "lora_" in name)
    logging.info(f"[augur] LoRA r={LORA_R} alpha={LORA_ALPHA} on {n} VLM projections ({n_lora:,} params); aug={AUG_NAME}")
    print(f"[augur] LoRA r={LORA_R} alpha={LORA_ALPHA} on {n} VLM projections ({n_lora:,} params); aug={AUG_NAME}", flush=True)
    return policy


_EMPTY_EVERY = int(os.environ.get("AUGUR_EMPTY_CACHE", "500"))
_STEP = [0]


def update_policy_with_aug(train_metrics, policy, batch, *args, **kwargs):
    out = _update_policy(train_metrics, policy, augment_batch(batch), *args, **kwargs)
    if FS_TARGET == "t3":
        _FS["fs"].ema_update(_FS["policy_ref"].model)
    _STEP[0] += 1
    if _EMPTY_EVERY and _STEP[0] % _EMPTY_EVERY == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()  # LOG-007: release fragmented cached blocks
    return out


class _KeepIndex:
    """Wraps LeRobot's preprocessor so each sample's dataset row index survives as augur_index."""

    def __init__(self, pre):
        self._pre = pre

    def __call__(self, batch, *a, **k):
        idx = batch.get("index") if isinstance(batch, dict) else None
        out = self._pre(batch, *a, **k)
        if idx is not None and isinstance(out, dict):
            out["augur_index"] = idx
        return out

    def __getattr__(self, name):
        return getattr(self._pre, name)


_make_pre_post = T.make_pre_post_processors


def make_pre_post_keep_index(*args, **kwargs):
    pre, post = _make_pre_post(*args, **kwargs)
    return (_KeepIndex(pre) if FS_TARGET != "none" else pre), post


T.make_pre_post_processors = make_pre_post_keep_index


T.make_policy = make_policy_with_lora

# LeRobot's --policy.use_amp uses torch's default CUDA autocast dtype (float16). The registered
# precision is bfloat16 (LOG-002, LOG-004), so set the default before LeRobot reads it.
import torch as _torch  # noqa: E402

_AMP = {"bf16": _torch.bfloat16, "fp16": _torch.float16}[os.environ.get("AUGUR_AMP_DTYPE", "bf16")]
if _torch.cuda.is_available():
    _torch.set_autocast_dtype("cuda", _AMP)

T.update_policy = update_policy_with_aug

if __name__ == "__main__":
    T.main()
