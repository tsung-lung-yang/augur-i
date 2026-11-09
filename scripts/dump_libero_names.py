"""Dump the official task names of the 4 LIBERO suites (run in the `augur` env).

Writes env/libero_task_names.json: {suite: [name_0, ..., name_9]} in the
official task order used by scripts/make_split.py.
Usage: python scripts/dump_libero_names.py
"""
import json
import os

from libero.libero import benchmark

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
d = benchmark.get_benchmark_dict()
names = {s: list(d[s]().get_task_names()) for s in SUITES}
assert all(len(v) == 10 for v in names.values()), {k: len(v) for k, v in names.items()}
os.makedirs("env", exist_ok=True)
with open("env/libero_task_names.json", "w") as f:
    json.dump(names, f, indent=1)
for s, v in names.items():
    print(s)
    for i, n in enumerate(v):
        print(f"  {i}: {n}")
