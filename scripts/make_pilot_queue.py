"""Write the pilot job queue (LOG-001 section 6, LOG-005) for scripts/run_queue.py.

One job at a time: an evaluation's simulator workers take about 2.2 GB of RAM each (about
18 GB at batch 8-10), so nothing else may run beside an evaluation (LOG-005).
Order:
  0. wait until T0 curve training and the T0 learning-curve evaluation have both finished
  1. T0 floor gate (LOG-003 section 6): T0 seed 0 = the curve checkpoint at step S,
     evaluated on the pilot plans (LIBERO-Plus dev, 7 axes; LIBERO-PRO Position dev);
     the queue stops here unless the gate passes
  2. foresight cache: LIBERO, SSv2, per-source flow statistics
  3. smoke: 100 steps of T3-TT fine-tuning without pretraining plus an 8-episode LIBERO-Plus
     evaluation (exercises the real training loop, saving and loading with a foresight
     module); then a 200-step pretraining timing run; set_P applies the LOG-005 rule
     (P = 20,000 if 20,000 x seconds-per-step <= 8 h, else 10,000) -> runs/pretrain/P.txt
  4. pretraining: t3 human+robot, t3 robot-only, t5 human+robot (seed 0, P steps)
  5. seed 0 of every condition, then seed 1: fine-tune S steps, then evaluate on the pilot
     plans. T0 seed 0 is the curve checkpoint (already evaluated in step 1).
Conditions: T0, T3-TR, T3-TT, T3-R (T3-TR pretrained on robot data only), T5-TR,
and T5-TT with --t5_tt.
Usage
  python scripts/make_pilot_queue.py --S 25000 [--t5_tt]
  python scripts/run_queue.py runs/queue/pilot.jsonl --dry_run
"""
import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--S", type=int, required=True, help="fine-tuning steps (LOG-005)")
p.add_argument("--P", type=int, default=None,
               help="pretraining steps; default: set by the LOG-005 rule from the timing run (set_P job)")
p.add_argument("--t5_tt", action="store_true", help="add the T5-TT condition (LOG-005)")
p.add_argument("--seeds", default="0,1")
p.add_argument("--cache", default="~/augur/data/foresight_cache")
p.add_argument("--frames", default="~/augur/data/ssv2_frames")
p.add_argument("--t0_curve", default="runs/train/t0_curve_s0")
p.add_argument("--out", default="runs/queue/pilot.jsonl")
args = p.parse_args()
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
if args.t5_tt:
    COND["t5_tt"] = ("t5", "tt", "t5_hr")
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
                 "cmd": (f"python scripts/eval_plan.py --policy {policy} --phase pilot --batch 8 "
                         f"--plan env/eval_plan_pro.csv --env_type libero --suites {PRO_SUITES} --out {out}/pro"),
                 "done": f"{out}/pro/results.csv"})


# 0. wait for T0 training and the T0 curve evaluation
T0_LAST = f"{ckpt(args.t0_curve, 60000)}/model.safetensors"
jobs.append({"name": "wait_t0_train", "conda": "augur", "done": T0_LAST,
             "cmd": f"while [ ! -f {T0_LAST} ]; do sleep 60; done"})
# curve.csv is written row by row: wait for the evaluation process to exit, then require all
# 12 checkpoints x 4 suites (fails, and stops the queue, if the evaluation died early)
jobs.append({"name": "wait_t0_curve", "conda": "augur", "done": "runs/eval/t0_curve_s0/curve.complete",
             "cmd": ("while pgrep -f '[e]val_dev_vanilla.py' > /dev/null; do sleep 60; done; "  # [e]: no self-match
                     "test $(tail -n +2 runs/eval/t0_curve_s0/curve.csv | wc -l) -eq 48 && "
                     "touch runs/eval/t0_curve_s0/curve.complete")})
# 1. T0 floor gate
eval_jobs("eval_t0_s0", ckpt(args.t0_curve, S), "runs/eval/pilot/t0_s0")
jobs.append({"name": "floor_gate", "conda": "augur", "done": "runs/eval/pilot/t0_s0/floor_gate_pass.json",
             "cmd": ("python scripts/floor_gate.py --plus runs/eval/pilot/t0_s0/plus/results.csv "
                     "--pro runs/eval/pilot/t0_s0/pro/results.csv --out runs/eval/pilot/t0_s0")})
# 2. cache
jobs += [
    {"name": "cache_libero", "conda": "augur", "done": f"{C}/libero.complete",
     "cmd": f"env -u HF_HUB_OFFLINE python scripts/foresight_cache.py --source libero --out {C} --batch 32 --workers 6"},
    {"name": "cache_ssv2", "conda": "augur", "done": f"{C}/ssv2.complete",
     "cmd": f"env -u HF_HUB_OFFLINE python scripts/foresight_cache.py --source ssv2 --out {C} --frames {F} --batch 32 --workers 6"},
    {"name": "cache_stats", "conda": "augur", "done": f"{C}/flow_stats.json",
     "cmd": f"python scripts/foresight_cache.py --stats --out {C}"},
]
# 3. smoke and timing
train_job("smoke_ft_t3_tt", "runs/train/smoke_ft_t3_tt", 100, 0, "t3", "tt", None)
eval_jobs("smoke_eval_t3_tt", ckpt("runs/train/smoke_ft_t3_tt", 100), "runs/eval/smoke_ft_t3_tt", limit=8)
jobs.append({"name": "pretrain_timing", "conda": "augur", "clean": "runs/pretrain/timing_t3_hr",
             "done": "runs/pretrain/timing_t3_hr/final.pt",
             "cmd": (f"env -u HF_HUB_OFFLINE python scripts/pretrain_foresight.py --target t3 --data hr "
                     f"--out runs/pretrain/timing_t3_hr --steps 200 --val_every 200 --save_every 1000 --n_val 64 "
                     f"--cache {C} --frames {F}")})
# LOG-005 rule: t = seconds per step of the timing run (batch 64, free GPU);
# P = 20,000 if 20,000 * t <= 8 h, else 10,000. Written to runs/pretrain/P.txt.
SET_P = ("python -c \"import csv; r=list(csv.DictReader(open('runs/pretrain/timing_t3_hr/log.csv')))[-1]; "
         "t=float(r['s_per_step']); P=20000 if 20000*t<=8*3600 else 10000; "
         "open('runs/pretrain/P.txt','w').write(str(P)); print('t', t, 's/step -> P', P)\"")
if args.P:
    SET_P = f"echo {args.P} > runs/pretrain/P.txt"
jobs.append({"name": "set_P", "conda": "augur", "done": "runs/pretrain/P.txt", "cmd": SET_P})
if True:
    # 4. pretraining
    for name, (target, data) in PRETRAIN.items():
        out = f"runs/pretrain/{name}_s0"
        jobs.append({"name": f"pretrain_{name}", "conda": "augur", "clean": out, "done": f"{out}/final.pt",
                     "cmd": (f"env -u HF_HUB_OFFLINE python scripts/pretrain_foresight.py --target {target} "
                             f"--data {data} --out {out} --steps $(cat runs/pretrain/P.txt) --cache {C} "
                             f"--frames {F} --seed 0")})
    # 5. conditions, seed by seed
    for seed in [int(s) for s in args.seeds.split(",")]:
        for cond, (target, mode, pre) in COND.items():
            if cond == "t0" and seed == 0:
                continue  # the curve checkpoint, evaluated in step 1
            run = f"runs/train/pilot/{cond}_s{seed}"
            init = f"runs/pretrain/{pre}_s0/final.pt" if pre else None
            train_job(f"train_{cond}_s{seed}", run, S, seed, target, mode, init)
            eval_jobs(f"eval_{cond}_s{seed}", ckpt(run, S), f"runs/eval/pilot/{cond}_s{seed}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print(f"wrote {args.out}: {len(jobs)} jobs (S={S}, P={args.P or 'by rule from timing'}, seeds {args.seeds}, "
      f"conditions {', '.join(COND)})")
for j in jobs:
    print(f"  {j['name']:24s} {j['conda']:10s} done: {j['done']}")
