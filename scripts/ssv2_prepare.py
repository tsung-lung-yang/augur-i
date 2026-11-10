"""Select the registered SSv2 subset and extract frames for foresight pretraining.

Step 1 (subset): from the SSv2 train split, draw --n clip ids with random.Random(--seed)
and write env/ssv2_subset_<n>.json (id, label, template). Deterministic; registered in LOG-005.
Step 2 (frames): for each clip, decode the .webm, take frames at --fps (nearest decoded
frame to k / fps seconds), centre-crop to a square, resize to --size, save JPEG
(quality 90) as <out>/<id[-2:]>/<id>_<k:02d>.jpg. A foresight pair is (k, k + fps) = 1 s apart.
Writes <out>/manifest.csv (id, n_frames, src_fps, duration). Resumable: clips whose
frames already exist are skipped. Runs with low CPU priority.

Usage:
  python scripts/ssv2_prepare.py --root ~/augur/data/ssv2 --out ~/augur/data/ssv2_frames --workers 8
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", required=True, help="folder with 20bn-something-something-v2/ and labels/")
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=50000)
p.add_argument("--seed", type=int, default=20261109)
p.add_argument("--fps", type=float, default=2.0)
p.add_argument("--size", type=int, default=256)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--subset_only", action="store_true")
p.add_argument("--limit", type=int, default=None, help="extract only the first N clips (smoke test)")
args = p.parse_args()

root, out = Path(args.root).expanduser(), Path(args.out).expanduser()


def find(*cands):
    for c in cands:
        for q in root.rglob(c):
            return q
    sys.exit(f"cannot find any of {cands} under {root}")


# ---------------------------------------------------------------- step 1: subset
train_json = find("something-something-v2-train.json", "train.json")
video_dir = find("20bn-something-something-v2")
train = json.load(open(train_json))
train = sorted(train, key=lambda r: int(r["id"]))
rng = random.Random(args.seed)
subset = sorted(rng.sample(train, args.n), key=lambda r: int(r["id"]))
os.makedirs("env", exist_ok=True)
sub_path = Path("env") / f"ssv2_subset_{args.n // 1000}k.json"
json.dump(
    {"seed": args.seed, "n": args.n, "source": str(train_json.name), "n_train": len(train),
     "clips": [{"id": r["id"], "label": r["label"], "template": r["template"]} for r in subset]},
    open(sub_path, "w"), indent=0,
)
print(f"train clips: {len(train)}; subset {len(subset)} -> {sub_path}", flush=True)
if args.subset_only:
    sys.exit(0)

# ---------------------------------------------------------------- step 2: frames
def extract(cid):
    import av
    from PIL import Image

    d = out / cid[-2:]
    src = video_dir / f"{cid}.webm"
    try:
        with av.open(str(src)) as c:
            s = c.streams.video[0]
            frames, times = [], []
            for f in c.decode(s):
                times.append(float(f.time) if f.time is not None else len(times) / float(s.average_rate or 12))
                frames.append(f)
            if not frames:
                return cid, 0, 0.0, 0.0, "no frames"
            src_fps = float(s.average_rate or 0)
            dur = times[-1]
            n_out = int(dur * args.fps) + 1
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n_out):
                t = k / args.fps
                j = min(range(len(times)), key=lambda i: abs(times[i] - t))
                img = frames[j].to_image()
                w, h = img.size
                m = min(w, h)
                img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
                img = img.resize((args.size, args.size), Image.BICUBIC)
                img.save(d / f"{cid}_{k:02d}.jpg", quality=90)
            return cid, n_out, src_fps, dur, ""
    except Exception as e:  # noqa: BLE001
        return cid, 0, 0.0, 0.0, repr(e)[:200]


def already(cid):
    return (out / cid[-2:] / f"{cid}_00.jpg").exists()


if __name__ == "__main__":
    try:
        os.nice(19)
    except OSError:
        pass
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"
    done = set()
    if man_path.exists():
        done = {r["id"] for r in csv.DictReader(open(man_path)) if int(r["n_frames"]) > 0}
    todo = [r["id"] for r in subset if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"to extract: {len(todo)} (already done {len(done)})", flush=True)
    new = not man_path.exists()
    t0, n_ok, n_err = time.time(), 0, 0
    with open(man_path, "a", newline="") as f, Pool(args.workers) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "n_frames", "src_fps", "duration", "error"])
        for i, (cid, n, fps, dur, err) in enumerate(pool.imap_unordered(extract, todo, chunksize=16), 1):
            w.writerow([cid, n, round(fps, 3), round(dur, 3), err])
            n_ok += n > 0
            n_err += n == 0
            if i % 1000 == 0 or i == len(todo):
                f.flush()
                el = time.time() - t0
                print(f"{i}/{len(todo)} done, {n_err} errors, {el / 60:.1f} min, "
                      f"eta {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
    print(f"finished: {n_ok} ok, {n_err} errors", flush=True)
