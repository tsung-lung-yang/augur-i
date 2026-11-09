"""Registered LIBERO-Plus evaluation plan: which variants each cell uses.

A cell is (suite, original task, perturbation axis). For every cell we draw
k distinct variants without replacement (k=5 for the pilot on the dev split,
k=3 for the main matrix on the held-out split), one episode each. If a cell
has fewer than k variants, all are used and the remainder is drawn again
from the same cell (marked rep > 0). Seeds are derived from SEED and the cell
name, so the plan does not depend on row order.

Input : env/libero_plus_index.csv  (scripts/plus_index.py)
Output: env/eval_plan_plus.csv     columns: phase, suite, orig_task, split,
        category, task_ids_index, rep
Usage : python scripts/make_eval_plan.py
"""
import collections
import csv
import random

SEED = 20261109
PHASES = {"pilot": ("dev", 5), "main": ("heldout", 3)}

cells = collections.defaultdict(list)
with open("env/libero_plus_index.csv") as f:
    for r in csv.DictReader(f):
        cells[(r["suite"], int(r["orig_task"]), r["split"], r["category"])].append(int(r["task_ids_index"]))

rows = []
for phase, (want_split, k) in PHASES.items():
    for (suite, task, split, cat), ids in sorted(cells.items()):
        if split != want_split:
            continue
        rng = random.Random(f"{SEED}-{phase}-{suite}-{task}-{cat}")
        pool = sorted(ids)
        picks = rng.sample(pool, min(k, len(pool)))
        reps = [0] * len(picks)
        while len(picks) < k:
            picks.append(rng.choice(pool))
            reps.append(1)
        for i, rep in zip(picks, reps):
            rows.append([phase, suite, task, split, cat, i, rep])

with open("env/eval_plan_plus.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase", "suite", "orig_task", "split", "category", "task_ids_index", "rep"])
    w.writerows(rows)

summary = collections.Counter((r[0], r[4]) for r in rows)
reused = collections.Counter(r[0] for r in rows if r[6])
print(f"wrote env/eval_plan_plus.csv: {len(rows)} rows (pilot + main, per checkpoint)")
for phase in PHASES:
    tot = sum(v for (p, _), v in summary.items() if p == phase)
    print(f"  {phase}: {tot} episodes, reused-variant episodes: {reused[phase]}")
    for (p, c), v in sorted(summary.items()):
        if p == phase:
            print(f"     {c:24s} {v}")
