"""LOG-001: deterministic dev / held-out task split for AUGUR-I.

Indices refer to the official task order of each LIBERO suite
(libero.benchmark.get_benchmark(suite)().get_task_names()).
LIBERO-Plus and LIBERO-PRO perturb these same tasks, so one split covers all.
Run with no arguments; output must match log/001_preregistration.md.
"""
import json
import random

SEED = 20261109
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
N_TASKS = 10
N_DEV = 3  # about one third of each suite

def make_split():
    split = {}
    for suite in SUITES:
        rng = random.Random(f"{SEED}-{suite}")
        dev = sorted(rng.sample(range(N_TASKS), N_DEV))
        held = [i for i in range(N_TASKS) if i not in dev]
        split[suite] = {"dev": dev, "heldout": held}
    return split

if __name__ == "__main__":
    print(json.dumps({"seed": SEED, "split": make_split()}, indent=2))
