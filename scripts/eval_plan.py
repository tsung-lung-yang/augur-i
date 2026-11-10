"""Run a registered LIBERO-Plus or LIBERO-PRO evaluation plan for one checkpoint.

LIBERO-PRO: --plan env/eval_plan_pro.csv --env_type libero
  --suites libero_spatial_swap,libero_object_swap,libero_goal_swap,libero_10_swap
  (run in the augur-pro env).

Reads env/eval_plan_plus.csv (scripts/make_eval_plan.py), keeps the rows of one
phase, and for each suite runs scripts/eval_mixed.py with batch-size chunks of
different variants. Writes one row per episode to <out>/results.csv and prints
success by axis.

Example (official checkpoint, pilot plan, first 20 episodes of one suite):
  python scripts/eval_plan.py --policy HuggingFaceVLA/smolvla_libero \
      --phase pilot --suites libero_spatial --limit 20 --out runs/eval/smoke_plus
Settings fixed by LOG-002 are the defaults (batch 10, n_action_steps 5, bf16).
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True)
p.add_argument("--phase", choices=["pilot", "main"], required=True)
p.add_argument("--out", required=True)
p.add_argument("--plan", default="env/eval_plan_plus.csv")
p.add_argument("--env_type", default="libero_plus")
p.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
p.add_argument("--categories", default=None, help="comma-separated subset of axes")
p.add_argument("--limit", type=int, default=None, help="max episodes per suite (smoke tests)")
p.add_argument("--batch", type=int, default=10)
p.add_argument("--n_action_steps", type=int, default=5)
p.add_argument("--no_amp", action="store_true")
p.add_argument("--videos", action="store_true")
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--obs", type=int, default=256, help="camera height = width in px (LIBERO-Plus noise assumes 256)")
args, extra = p.parse_known_args()

out = Path(args.out)
if out.exists():
    sys.exit(f"{out} exists; choose a new --out")
out.mkdir(parents=True)
json.dump(vars(args) | {"extra": extra}, open(out / "args.json", "w"), indent=1)

cats = set(args.categories.split(",")) if args.categories else None
plan = [r for r in csv.DictReader(open(args.plan)) if r["phase"] == args.phase and (cats is None or r["category"] in cats)]
here = Path(__file__).resolve().parent
results, t0 = [], time.time()

for suite in args.suites.split(","):
    rows = [r for r in plan if r["suite"] == suite]
    rows.sort(key=lambda r: (int(r["orig_task"]), r["category"], int(r["task_ids_index"]), int(r["rep"])))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        continue
    spec = [[int(r["task_ids_index"]), int(r["rep"])] for r in rows]
    n_real = len(spec)
    while len(spec) % args.batch:  # pad the last chunk; padded episodes are discarded
        spec.append([spec[-1][0], spec[-1][1] + 1])
    sdir = out / suite
    sdir.mkdir()
    json.dump(spec, open(sdir / "spec.json", "w"))
    env = dict(os.environ, PYTHONUNBUFFERED="1", AUGUR_MIXED_SPEC=str(sdir / "spec.json"), AUGUR_EVAL_VIDEOS="1" if args.videos else "0")
    cmd = [
        sys.executable, str(here / "eval_mixed.py"),
        f"--policy.path={args.policy}", f"--env.type={args.env_type}", f"--env.task={suite}",
        f"--eval.batch_size={args.batch}", f"--eval.n_episodes={args.batch}",
        f"--policy.n_action_steps={args.n_action_steps}", f"--policy.use_amp={str(not args.no_amp).lower()}",
        f"--seed={args.seed}", f"--env.observation_height={args.obs}", f"--env.observation_width={args.obs}",
        f"--output_dir={sdir / 'lerobot'}", *extra,
    ]
    ts = time.time()
    print(f"\n== {suite}: {n_real} episodes ({len(spec) - n_real} padding), {len(spec) // args.batch} chunks", flush=True)
    with open(sdir / "lerobot.log", "w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        sys.exit(f"eval failed for {suite} (exit {rc}); see {sdir / 'lerobot.log'}")
    info = json.load(open(sdir / "lerobot" / "eval_info.json"))
    succ = {}
    for t in info["per_task"]:
        for slot, s in enumerate(t["metrics"]["successes"]):
            succ[int(t["task_id"]) * args.batch + slot] = "error" if s is None or s != s else int(bool(s))
    n_err = 0
    for k, r in enumerate(rows):
        results.append({**r, "success": succ[k], "chunk": k // args.batch, "slot": k % args.batch})
        n_err += succ[k] == "error"
    if n_err:
        print(f"   WARNING: {n_err} episodes errored (see CHUNK FAILED in {sdir / 'lerobot.log'})", flush=True)
    dt = time.time() - ts
    print(f"   done in {dt / 60:.1f} min, {dt / len(spec):.1f} s per episode (incl. padding)", flush=True)

with open(out / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

by, err = collections.defaultdict(list), collections.Counter()
for r in results:
    if r["success"] == "error":
        err[r["category"]] += 1
    else:
        by[r["category"]].append(r["success"])
print(f"\nwrote {out / 'results.csv'}: {len(results)} episodes in {(time.time() - t0) / 60:.1f} min")
for c in sorted(set(by) | set(err)):
    v = by[c]
    rate = f"{100 * sum(v) / len(v):5.1f}%" if v else "  n/a"
    print(f"  {c:24s} {sum(v):4d}/{len(v):<4d} {rate}   errors: {err[c]}")
