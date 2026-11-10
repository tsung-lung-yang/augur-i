"""Precompute foresight targets (LOG-005): image embeddings (t3) and optical flow (t5).

For every frame of the main camera:
  emb   frozen SigLIP + connector embedding, (64, D) float16 -- the vision tower is frozen and
        has no LoRA, so one cache serves every condition. The t3 target is computed at train
        time from emb[future] through the VLM text layers with EMA LoRA.
  flow  RAFT-small flow from the frame to its 1-second future, average-pooled to 8x8 and
        divided by the image size, arranged as 4 quadrant tokens of 4x4x2 = 32 values.
  future, valid   index of the 1-second future (clamped to the last frame) and whether the
        true future exists.
Sources
  libero  lerobot/libero, main camera observation.images.image, 10 fps -> future = +10 frames
  ssv2    the registered SSv2 subset (env/ssv2_subset_50k.json) extracted at 2 fps by
          scripts/ssv2_prepare.py -> future = +2 frames
Output  <out>/<source>_{emb,flow,future,valid}.npy (+ ssv2_rows.npz, flow_stats.json)
Resumable per episode / clip block (<source>_done.npy).

Usage
  python scripts/foresight_cache.py --source libero --out ~/augur/data/foresight_cache
  python scripts/foresight_cache.py --source ssv2   --out ~/augur/data/foresight_cache \
      --frames ~/augur/data/ssv2_frames
  python scripts/foresight_cache.py --stats --out ~/augur/data/foresight_cache
  add --limit N (episodes or clips) for a smoke test into a scratch --out.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--source", choices=["libero", "ssv2"])
p.add_argument("--stats", action="store_true", help="compute flow mean/std over valid libero+ssv2 flows")
p.add_argument("--out", required=True)
p.add_argument("--frames", default="~/augur/data/ssv2_frames")
p.add_argument("--subset", default="env/ssv2_subset_50k.json")
p.add_argument("--repo_id", default="lerobot/libero")
p.add_argument("--camera", default="observation.images.image")
p.add_argument("--batch", type=int, default=32)
p.add_argument("--workers", type=int, default=6)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--no_flow", action="store_true")
p.add_argument("--no_emb", action="store_true")
args = p.parse_args()
out = Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ stats only
if args.stats:
    vals = []
    for src in ("libero", "ssv2"):
        f, v = out / f"{src}_flow.npy", out / f"{src}_valid.npy"
        if f.exists():
            fl, va = np.load(f, mmap_mode="r"), np.load(v)
            idx = np.flatnonzero(va)
            idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), 200000)).astype(int)]
            vals.append(np.asarray(fl[idx], dtype=np.float32).reshape(len(idx), -1, 32))
    x = np.concatenate(vals).reshape(-1, 32)
    stats = {"mean": x.mean(0).tolist(), "std": (x.std(0) + 1e-6).tolist(), "n": int(len(x))}
    json.dump(stats, open(out / "flow_stats.json", "w"))
    print("flow stats over", stats["n"], "tokens; mean |mean|", float(np.abs(x.mean(0)).mean()), "mean std", float(x.std(0).mean()))
    sys.exit(0)


# ------------------------------------------------------------------ models
def build_embedder():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = SmolVLAConfig(load_vlm_weights=True, device=dev)
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    pol = SmolVLAPolicy(cfg).to(dev).eval()

    @torch.no_grad()
    def embed(img01):  # (B,3,256,256) float in [0,1]
        imgs, _ = pol.prepare_images({"observation.images.image": img01.to(dev)})
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            e = pol.model.vlm_with_expert.embed_image(imgs[0])
        return e.float().cpu().numpy().astype(np.float16)

    d = pol.model.vlm_with_expert.config.text_config.hidden_size
    return embed, d


def build_flow():
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    w = Raft_Small_Weights.DEFAULT
    net = raft_small(weights=w).to(dev).eval()
    tf = w.transforms()

    @torch.no_grad()
    def flow(a01, b01):  # (B,3,256,256) in [0,1] -> (B,4,32) float16
        a, b = tf(a01.to(dev), b01.to(dev))
        fl = net(a, b)[-1]  # (B,2,H,W) pixels
        h = fl.shape[-1]
        g = F.adaptive_avg_pool2d(fl, 8) / h  # (B,2,8,8), fraction of image size
        g = g.permute(0, 2, 3, 1)  # (B,8,8,2)
        q = torch.stack([g[:, :4, :4], g[:, :4, 4:], g[:, 4:, :4], g[:, 4:, 4:]], 1)  # (B,4,4,4,2)
        return q.reshape(q.shape[0], 4, 32).float().cpu().numpy().astype(np.float16)

    return flow


def open_arrays(src, n, d):
    def mm(name, shape, dtype, fill=None):
        f = out / f"{src}_{name}.npy"
        if f.exists():
            return np.load(f, mmap_mode="r+")
        a = np.lib.format.open_memmap(f, mode="w+", dtype=dtype, shape=shape)
        if fill is not None:
            a[:] = fill
        return a

    return (None if args.no_emb else mm("emb", (n, 64, d), np.float16),
            None if args.no_flow else mm("flow", (n, 4, 32), np.float16),
            mm("future", (n,), np.int64, -1), mm("valid", (n,), np.bool_, False))


def process_block(frames, rows, fut_local, valid_local, embed, flow, arrs):
    """frames: (T,3,256,256) float [0,1] of one episode/clip; rows: global row ids (T,)."""
    emb_a, flow_a, fut_a, val_a = arrs
    for s in range(0, len(frames), args.batch):
        sl = slice(s, s + args.batch)
        if emb_a is not None:
            emb_a[rows[sl]] = embed(frames[sl])
        if flow_a is not None:
            flow_a[rows[sl]] = flow(frames[sl], frames[fut_local[sl]])
    fut_a[rows] = rows[fut_local]
    val_a[rows] = valid_local


# ------------------------------------------------------------------ sources
def run_libero(embed, flow, d):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id)
    n, fps = len(ds), ds.fps
    horizon = int(round(fps * 1.0))
    print(f"libero: {n} frames, fps {fps}, horizon {horizon} frames, camera {args.camera}", flush=True)
    arrs = open_arrays("libero", n, d)
    n_eps = ds.num_episodes
    done_f = out / "libero_done.npy"
    done = np.load(done_f) if done_f.exists() else np.zeros(n_eps, bool)
    keys = [args.camera, "episode_index", "frame_index", "index"]

    class Frames(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            it = ds[i]
            return {k: it[k] for k in keys}

    dl = torch.utils.data.DataLoader(Frames(), batch_size=64, shuffle=False, num_workers=args.workers)
    buf, cur, n_done, t0 = [], None, 0, time.time()

    def flush(items):
        nonlocal n_done
        ep = int(items[0]["episode_index"])
        if not done[ep]:
            fr = torch.stack([x[args.camera] for x in items])
            rows = np.array([int(x["index"]) for x in items])
            fi = np.array([int(x["frame_index"]) for x in items])
            assert np.all(np.diff(fi) == 1), f"episode {ep} frames not contiguous"
            t = len(items)
            loc = np.arange(t)
            fut = np.minimum(loc + horizon, t - 1)
            process_block(fr, rows, fut, loc + horizon < t, embed, flow, arrs)
            done[ep] = True
        n_done += 1
        if n_done % 50 == 0:
            np.save(done_f, done)
            print(f"  {n_done}/{n_eps} episodes, {(time.time() - t0) / 60:.1f} min", flush=True)

    for batch in dl:
        for j in range(len(batch["index"])):
            item = {k: batch[k][j] for k in keys}
            ep = int(item["episode_index"])
            if cur is not None and ep != cur:
                flush(buf)
                buf = []
                if args.limit and n_done >= args.limit:
                    break
            cur = ep
            buf.append(item)
        if args.limit and n_done >= args.limit:
            break
    else:
        if buf:
            flush(buf)
    np.save(done_f, done)
    print(f"libero done: {int(done.sum())}/{n_eps} episodes", flush=True)


def run_ssv2(embed, flow, d):
    from PIL import Image

    frames_dir = Path(args.frames).expanduser()
    sub = json.load(open(args.subset))["clips"]
    man = {}
    for r in csv.DictReader(open(frames_dir / "manifest.csv")):  # retried clips may appear twice
        man[r["id"]] = max(man.get(r["id"], 0), int(r["n_frames"] or 0))
    clips = [c["id"] for c in sub if man.get(c["id"], 0) > 0]
    if args.limit:
        clips = clips[: args.limit]
    counts = np.array([man[c] for c in clips])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n = int(offsets[-1])
    np.savez(out / "ssv2_rows.npz", clip_ids=np.array(clips), offsets=offsets)
    print(f"ssv2: {len(clips)} clips, {n} frames (2 fps, horizon 2 frames)", flush=True)
    arrs = open_arrays("ssv2", n, d)
    done_f = out / "ssv2_done.npy"
    done = np.load(done_f) if done_f.exists() else np.zeros(len(clips), bool)

    class Clips(torch.utils.data.Dataset):
        def __len__(self):
            return len(clips)

        def __getitem__(self, i):
            c = clips[i]
            ims = [np.asarray(Image.open(frames_dir / c[-2:] / f"{c}_{k:02d}.jpg").convert("RGB")) for k in range(counts[i])]
            return i, torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float() / 255.0

    dl = torch.utils.data.DataLoader(Clips(), batch_size=None, shuffle=False, num_workers=args.workers)
    t0 = time.time()
    pend, prow, pfut, pval, pidx = [], [], [], [], []

    def flush_pending():
        if not pend:
            return
        fr = torch.cat(pend)
        rows = np.concatenate(prow)
        fut_global = np.concatenate(pfut)
        # futures are local to the concatenated block
        process_block(fr, rows, fut_global, np.concatenate(pval), embed, flow, arrs)
        for i in pidx:
            done[i] = True
        pend.clear(); prow.clear(); pfut.clear(); pval.clear(); pidx.clear()

    base = 0
    for k, (i, fr) in enumerate(dl):
        if done[i]:
            continue
        t = fr.shape[0]
        loc = np.arange(t)
        pend.append(fr)
        prow.append(np.arange(offsets[i], offsets[i] + t))
        pfut.append(base + np.minimum(loc + 2, t - 1))
        pval.append(loc + 2 < t)
        pidx.append(i)
        base += t
        if base >= 256:
            flush_pending()
            base = 0
        if (k + 1) % 2000 == 0:
            np.save(done_f, done)
            el = time.time() - t0
            print(f"  {k + 1}/{len(clips)} clips, {el / 60:.1f} min, eta {el / (k + 1) * (len(clips) - k - 1) / 60:.0f} min", flush=True)
    flush_pending()
    np.save(done_f, done)
    print(f"ssv2 done: {int(done.sum())}/{len(clips)} clips", flush=True)


if __name__ == "__main__":
    embed, d = (None, None) if args.no_emb else build_embedder()
    if d is None:
        d = 960
    flow = None if args.no_flow else build_flow()
    print(f"device {dev}, emb {'off' if embed is None else d}, flow {'off' if flow is None else 'raft_small'}", flush=True)
    {"libero": run_libero, "ssv2": run_ssv2}[args.source](embed, flow, d)
