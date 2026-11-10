"""Write the pilot job queue (LOG-001 section 6, LOG-005) for scripts/run_queue.py.

Order (everything on one GPU, one job at a time):
  0. wait until T0 curve training has written its last checkpoint
  1. foresight cache: LIBERO, SSv2, per-source flow statistics
  (wait for the T0 learning-curve evaluation to finish)
  2. smoke: 100 steps of T3-TT fine-tuning without pretraining, then an 8-episode
     LIBERO-Plus evaluation of it -- exercises the real LeRobot training loop, saving and
     evaluation with a foresight module before hours of compute are spent
     then a 200-step pretraining timing run (sets P)
  -- T0 seed 0 evaluation on the pilot plans = T0 floor gate; the queue stops unless it passes
  3. pretraining: t3 human+robot, t3 robot-only, t5 human+robot (seed 0, P steps)
  4. seed 0 of every condition, then seed 1: fine-tune S steps, then evaluate on the pilot
     plans (LIBERO-Plus dev, 7 axes, augur-plus env; LIBERO-PRO Position dev, augur-pro env)
     T0 seed 0 is the T0 learning-curve checkpoint at step S (LOG-004 section 4), evaluated
     once, before the floor gate.
Conditions: T0, T3-TR, T3-TT, T3-R (T3-TR pretrained on robot data only), T5-TR.
Usage
  python scripts/make_pilot_queue.py --phase_a --out runs/queue/phase_a.jsonl   (before S is known)
  python scripts/make_pilot_queue.py --S 30000 --P 20000
  python scripts/run_queue.py runs/queue/pilot.jsonl --dry_run
"""
import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--S", type=int, default=None, help="fine-tuning steps (LOG-005)")
p.add_argument("--P", type=int, default=None, help="pretraining steps (LOG-005)")
p.add_argument("--phase_a", action="store_true",
               help="only the jobs that need neither S nor P: cache, wait for the T0 curve, smoke, timing")
p.add_argument("--seeds", default="0,1")
p.add_argument("--cache", default="~/augur/data/foresight_cache")
p.add_argument("--frames", default="~/augur/data/ssv2_frames")
p.add_argument("--t0_curve", default="runs/train/t0_curve_s0")
p.add_argument("--out", default="runs/queue/pilot.jsonl")
args = p.parse_args()
assert args.phase_a or (args.S and args.P), "give --S and --P (or --phase_a)"
C = os.path.expanduser(args.cache)
F = os.path.expanduser(args.frames)
S = args.S

TRAIN = ("python scripts/train_augur.py --policy.type=smolvla --policy.load_vlm_weights=true "
         "--policy.push_to_hub=false --policy.use_amp=true --dataset.repo_id=lerobot/libero "
         "--dataset.video_backend=torchcodec --batch_size=32 --log_freq=100 --num_workers=8 "
         "--wandb.enable=false")
EVAL_ENV = {"HF_HUB_OFFLINE": "1", "MUJOCO_GL": "egl", "GALLIUM_DRIVER": "d3d12"}
PRO_SUITES = "libero_spatial_swap,libero_object_swap,libero_goal_swap,libero_10_swap"
COND = {  # name: (target, mode, pretraining run or None)
    "t0": ("none", "tr", None),
    "t3_tr": ("t3", "tr", "t3_hr"),
    "t3_tt": ("t3", "tt", "t3_hr"),
    "t3_r": ("t3", "tr", "t3_r"),
    "t5_tr": ("t5", "tr", "t5_hr"),
}
PRETRAIN = {"t3_hr": ("t3", "hr"), "t3_r": ("t3", "r"), "t5_hr": ("t5", "hr")}
jobs = []


def ckpt(run, step):
    return f"{run}/checkpoints/{step:06d}/pretrained_model"


def train_job(name, out, steps, seed, target, mode, init):
    env = {"AUGUR_TARGET": target, "AUGUR_MODE": mode, "AUGUR_CACHE": C}
    if init:
        env["AUGUR_INIT"] = init
    jobs.append({"name": name, "conda": "augur", "env": env, "clean": out,
                 "cmd": f"env -u HF_HUB_OFFLINE {TRAIN} --steps={steps} --save_freq={steps} --seed={seed} --output_dir={out}",
                 "done": f"{ckpt(out, steps)}/model.safetensors"})


def eval_jobs(name, policy, out, limit=None):
    lim = f" --limit {limit} --suites libero_spatial" if limit else ""
    jobs.append({"name": f"{name}_plus", "conda": "augur-plus", "env": EVAL_ENV, "clean": f"{out}/plus",
                 "cmd": f"python scripts/eval_plan.py --policy {policy} --phase pilot --batch 8 --out {out}/plus{lim}",
                 "done": f"{out}/plus/results.csv"})
    if limit:
        return
    jobs.append({"name": f"{name}_pro", "conda": "augur-pro", "env": EVAL_ENV, "clean": f"{out}/pro",
                 "cmd": (f"python scripts/eval_plan.py --policy {policy} --phase pilot --batch 10 "
                         f"--plan env/eval_plan_pro.csv --env_type libero --suites {PRO_SUITES} --out {out}/pro"),
                 "done": f"{out}/pro/results.csv"})


# 0. wait for T0 curve training to finish (the cache needs the GPU memory it uses)
T0_LAST = f"{ckpt(args.t0_curve, 60000)}/model.safetensors"
jobs.append({"name": "wait_t0_train", "conda": "augur", "done": T0_LAST,
             "cmd": f"while [ ! -f {T0_LAST} ]; do sleep 60; done"})
# 1. cache
jobs += [
    {"name": "cache_libero", "conda": "augur", "done": f"{C}/libero.complete",
     "cmd": f"env -u HF_HUB_OFFLINE python scripts/foresight_cache.py --source libero --out {C} --batch 32 --workers 6"},
    {"name": "cache_ssv2", "conda": "augur", "done": f"{C}/ssv2.complete",
     "cmd": f"env -u HF_HUB_OFFLINE python scripts/foresight_cache.py --source ssv2 --out {C} --frames {F} --batch 32 --workers 6"},
    {"name": "cache_stats", "conda": "augur", "done": f"{C}/flow_stats.json",
     "cmd": f"python scripts/foresight_cache.py --stats --out {C}"},
]
# wait until the T0 learning-curve evaluation (auto-started after T0 training) has finished, so
# no training job shares the GPU with it
# curve.csv is written row by row, so wait for the evaluation process to exit and then require
# all 12 checkpoints x 4 suites (fails, and stops the queue, if the evaluation died early)
jobs.append({"name": "wait_t0_curve", "conda": "augur", "done": "runs/eval/t0_curve_s0/curve.complete",
             "cmd": ("while pgrep -f eval_dev_vanilla.py > /dev/null; do sleep 60; done; "
                     "test $(tail -n +2 runs/eval/t0_curve_s0/curve.csv | wc -l) -eq 48 && "
                     "touch runs/eval/t0_curve_s0/curve.complete")})
# 2. smoke
train_job("smoke_ft_t3_tt", "runs/train/smoke_ft_t3_tt", 100, 0, "t3", "tt", None)
eval_jobs("smoke_eval_t3_tt", ckpt("runs/train/smoke_ft_t3_tt", 100), "runs/eval/smoke_ft_t3_tt", limit=8)
# pretraining speed at the registered batch on a free GPU (sets P, LOG-005)
jobs.append({"name": "pretrain_timing", "conda": "augur", "clean": "runs/pretrain/timing_t3_hr",
             "done": "runs/pretrain/timing_t3_hr/final.pt",
             "cmd": (f"env -u HF_HUB_OFFLINE python scripts/pretrain_foresight.py --target t3 --data hr "
                     f"--out runs/pretrain/timing_t3_hr --steps 200 --val_every 200 --save_every 1000 --n_val 64 "
                     f"--cache {C} --frames {F}")})
# T0 floor gate (LOG-003 section 6) before any non-T0 training
if not args.phase_a:
    eval_jobs("eval_t0_s0", ckpt(args.t0_curve, S), "runs/eval/pilot/t0_s0")
    jobs.append({"name": "floor_gate", "conda": "augur", "done": "runs/eval/pilot/t0_s0/floor_gate_pass.json",
                 "cmd": ("python scripts/floor_gate.py --plus runs/eval/pilot/t0_s0/plus/results.csv "
                         "--pro runs/eval/pilot/t0_s0/pro/results.csv --out runs/eval/pilot/t0_s0")})
# 3. pretraining
for name, (target, data) in ({} if args.phase_a else PRETRAIN).items():
    out = f"runs/pretrain/{name}_s0"
    jobs.append({"name": f"pretrain_{name}", "conda": "augur", "clean": out, "done": f"{out}/final.pt",
                 "cmd": (f"env -u HF_HUB_OFFLINE python scripts/pretrain_foresight.py --target {target} --data {data} "
                         f"--out {out} --steps {args.P} --cache {C} --frames {F} --seed 0")})
# 4. conditions, seed by seed
for seed in ([] if args.phase_a else [int(s) for s in args.seeds.split(",")]):
    for cond, (target, mode, pre) in COND.items():
        run = f"runs/train/pilot/{cond}_s{seed}"
        if cond == "t0" and seed == 0:
            continue  # evaluated above (floor gate)
        else:
            init = f"runs/pretrain/{pre}_s0/final.pt" if pre else None
            train_job(f"train_{cond}_s{seed}", run, S, seed, target, mode, init)
            policy = ckpt(run, S)
        eval_jobs(f"eval_{cond}_s{seed}", policy, f"runs/eval/pilot/{cond}_s{seed}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(f"wrote {args.out}: {len(jobs)} jobs (S={S}, P={args.P}, seeds {args.seeds}{', phase A only' if args.phase_a else ''})")
for j in jobs:
    print(f"  {j['name']:24s} {j['conda']:10s} done: {j['done']}")
