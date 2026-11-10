"""Evaluate checkpoints on the dev split of vanilla LIBERO (LOG-002 section 5, step 2).

For each checkpoint: 12 dev tasks (LOG-001 split) x 10 initial states = 120
episodes, run through scripts/eval_mixed.py in the `augur` env with the
registered eval settings (256 px, batch 10, n_action_steps 5, bf16).
Writes <out>/curve.csv with one row per (checkpoint, suite) and prints the curve.

Usage:
  python scripts/eval_dev_vanilla.py --run runs/train/t0_curve_s0 --out runs/eval/t0_curve_s0
  (evaluates every runs/train/.../checkpoints/<step>/pretrained_model, skipping ones already done)
  python scripts/eval_dev_vanilla.py --ckpt <path/to/pretrained_model> --out <dir>
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from make_split import make_split  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", help="training output dir containing checkpoints/<step>/pretrained_model")
p.add_argument("--ckpt", help="single pretrained_model dir")
p.add_argument("--out", required=True)
p.add_argument("--n_init", type=int, default=10)
p.add_argument("--batch", type=int, default=10)
p.add_argument("--n_action_steps", type=int, default=5)
p.add_argument("--obs", type=int, default=256)
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--suites", default=None, help="comma-separated subset (smoke tests only)")
args = p.parse_args()

if args.run:
    ck = sorted(Path(args.run, "checkpoints").glob("*/pretrained_model"), key=lambda q: q.parent.name)
    ck = [c for c in ck if c.parent.name.isdigit()]
elif args.ckpt:
    ck = [Path(args.ckpt)]
else:
    sys.exit("give --run or --ckpt")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
curve_path = out / "curve.csv"
done = set()
if curve_path.exists():
    done = {(r["ckpt"], r["suite"]) for r in csv.DictReader(open(curve_path))}
split = make_split()
if args.suites:
    split = {k: v for k, v in split.items() if k in args.suites.split(",")}
here = Path(__file__).resolve().parent
new_file = not curve_path.exists()

with open(curve_path, "a", newline="") as fcsv:
    w = csv.writer(fcsv)
    if new_file:
        w.writerow(["ckpt", "step", "suite", "successes", "episodes", "rate"])
    for c in ck:
        step = c.parent.name if c.parent.name.isdigit() else "single"
        tot_s = tot_n = 0
        for suite, sp in split.items():
            key = (str(c), suite)
            if key in done:
                continue
            spec = [[t, e] for t in sp["dev"] for e in range(args.n_init)]
            while len(spec) % args.batch:
                spec.append(spec[-1])
            sdir = out / f"{step}_{suite}"
            sdir.mkdir(exist_ok=True)
            json.dump(spec, open(sdir / "spec.json", "w"))
            env = dict(os.environ, PYTHONUNBUFFERED="1", AUGUR_MIXED_SPEC=str(sdir / "spec.json"), AUGUR_EVAL_VIDEOS="0")
            cmd = [
                sys.executable, str(here / "eval_mixed.py"),
                f"--policy.path={c}", "--env.type=libero", f"--env.task={suite}",
                f"--eval.batch_size={args.batch}", f"--eval.n_episodes={args.batch}",
                f"--policy.n_action_steps={args.n_action_steps}", "--policy.use_amp=true",
                f"--env.observation_height={args.obs}", f"--env.observation_width={args.obs}",
                f"--seed={args.seed}", f"--output_dir={sdir / 'lerobot'}",
            ]
            with open(sdir / "log.txt", "w") as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                print(f"[{step}] {suite}: FAILED (see {sdir / 'log.txt'})", flush=True)
                continue
            info = json.load(open(sdir / "lerobot" / "eval_info.json"))
            succ = [s for t in sorted(info["per_task"], key=lambda t: int(t["task_id"])) for s in t["metrics"]["successes"]]
            succ = succ[: len(sp["dev"]) * args.n_init]
            s = sum(1 for v in succ if v == v and v)
            w.writerow([str(c), step, suite, s, len(succ), round(s / len(succ), 4)])
            fcsv.flush()
            tot_s, tot_n = tot_s + s, tot_n + len(succ)
            print(f"[{step}] {suite}: {s}/{len(succ)}", flush=True)
        if tot_n:
            print(f"[{step}] dev vanilla total: {tot_s}/{tot_n} = {100 * tot_s / tot_n:.1f}%", flush=True)
