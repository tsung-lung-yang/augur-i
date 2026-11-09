"""Index LIBERO-Plus variants by (suite, original task, perturbation axis).

Run in the `augur-plus` env after scripts/dump_libero_names.py was run in `augur`.
Writes env/libero_plus_index.csv with one row per variant:
  suite, task_ids_index, cls_id, name, category, difficulty, orig_task, split
where task_ids_index is the 0-based position in the LIBERO-Plus benchmark task
list (what `lerobot-eval --env.task_ids` takes) and split is dev/heldout from
LOG-001.
Usage: python scripts/plus_index.py
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_split import make_split  # noqa: E402

from libero.libero import benchmark, get_libero_path  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
vanilla = json.load(open("env/libero_task_names.json"))
cls_path = os.path.join(get_libero_path("benchmark_root"), "benchmark", "task_classification.json")
cls = json.load(open(cls_path))
split = make_split()
bench = benchmark.get_benchmark_dict()

rows, problems = [], collections.Counter()
for suite in SUITES:
    names = list(bench[suite]().get_task_names())
    pos = {n: i for i, n in enumerate(names)}
    entries = cls[suite]
    print(f"\n== {suite}: {len(names)} benchmark tasks, {len(entries)} classified entries")
    # how cls 'id' relates to the benchmark list position
    rel = collections.Counter()
    for e in entries[:200]:
        i = pos.get(e["name"])
        rel["name not in benchmark" if i is None else f"position = id {i - e['id']:+d}"] += 1
    print("   id vs position (first 200):", dict(rel))
    orig_sorted = sorted(enumerate(vanilla[suite]), key=lambda t: -len(t[1]))
    for e in entries:
        i = pos.get(e["name"])
        if i is None:
            problems[f"{suite}: name not in benchmark"] += 1
            continue
        orig = next((k for k, n in orig_sorted if e["name"].startswith(n)), None)
        if orig is None:
            problems[f"{suite}: no original-task prefix"] += 1
            continue
        sp = "dev" if orig in split[suite]["dev"] else "heldout"
        rows.append([suite, i, e["id"], e["name"], e["category"], e.get("difficulty_level"), orig, sp])

with open("env/libero_plus_index.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["suite", "task_ids_index", "cls_id", "name", "category", "difficulty", "orig_task", "split"])
    w.writerows(rows)
print(f"\nwrote env/libero_plus_index.csv: {len(rows)} rows")
print("problems:", dict(problems) or "none")

cats = sorted({r[4] for r in rows})
print("\ncategories:", cats)
cnt = collections.Counter((r[0], r[6], r[4]) for r in rows)
print("\nvariants per (original task, category): min / median / max, and cells with < 3 variants")
for c in cats:
    vals = sorted(cnt[(s, t, c)] for s in SUITES for t in range(10))
    low = [(s, t) for s in SUITES for t in range(10) if cnt[(s, t, c)] < 3]
    print(f"  {c:28s} {vals[0]:4d} / {vals[len(vals) // 2]:4d} / {vals[-1]:4d}   <3: {len(low)} {low[:6]}")
