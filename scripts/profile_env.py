"""AUGUR-I week-1: time the LIBERO simulator alone (no policy).

Steps libero_spatial task 1 (dev split) with a constant no-op action and
reports per-step time with cameras on, and physics-only time with cameras off.
Usage: MUJOCO_GL=egl python scripts/profile_env.py [--steps 100] [--size 256]
"""
import argparse
import os
import time

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100)
p.add_argument("--size", type=int, default=256)
args = p.parse_args()

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(1)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
init_states = suite.get_task_init_states(1)
action = np.array([0.0] * 6 + [-1.0])

print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("task      =", task.name)


def run(use_camera):
    kw = dict(bddl_file_name=bddl, camera_heights=args.size, camera_widths=args.size)
    t0 = time.perf_counter()
    if use_camera:
        env = OffScreenRenderEnv(**kw)
    else:
        env = ControlEnv(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, **kw)
    env.seed(0)
    env.reset()
    env.set_init_state(init_states[0])
    t_make = time.perf_counter() - t0
    for _ in range(5):  # warm-up
        env.step(action)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(action)
    dt = (time.perf_counter() - t0) / args.steps
    env.close()
    return t_make, dt


for cam in (True, False):
    try:
        t_make, dt = run(cam)
    except Exception as e:  # noqa: BLE001
        print("cameras" if cam else "physics only", "FAILED:", repr(e)[:300])
        continue
    label = f"cameras {args.size}px" if cam else "physics only"
    print(f"{label:16s}: make {t_make:5.1f} s | {dt * 1000:7.1f} ms/step | {1 / dt:6.1f} steps/s")
