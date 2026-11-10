"""Registered LIBERO-PRO evaluation plan (LOG-005): the Position axis.

A cell is (suite, original task). LIBERO-PRO's Position suite (libero_<suite>_swap) has one
perturbed layout per task and a file of init states for it; for every cell we draw k distinct
init states without replacement (k=5 for the pilot on the dev split, k=3 for the main matrix
on the held-out split), one episode each, mirroring the LIBERO-Plus plan. Tasks whose layout
is unchanged by the released files (perturbed = 0 in the index) are excluded and listed.
Seeds are derived from SEED and the cell name.

Input : env/libero_pro_index.csv (scripts/libero_pro_setup.py)
Output: env/eval_plan_pro.csv  columns as env/eval_plan_plus.csv: phase, suite, orig_task,
        split, category, task_ids_index, rep  (rep = init state index; task_ids_index =
        the task's position in the swap suite, which equals the base order)
Usage : python scripts/make_eval_plan_pro.py
"""
import collections
import csv
import random

SEED = 20261109
PHASES = {"pilot": ("dev", 5), "main": ("heldout", 3)}
CATEGORY = {"swap": "pro_position"}

idx = list(csv.DictReader(open("env/libero_pro_index.csv")))
rows, excluded = [], []
for phase, (want, k) in PHASES.items():
    for r in idx:
        if r["split"] != want:
            continue
        if r["perturbed"] != "1":
            excluded.append((phase, r["suite"], r["orig_task"]))
            continue
        rng = random.Random(f"{SEED}-{phase}-{r['suite']}-{r['orig_task']}")
        n = int(r["n_init"])
        for init_id in sorted(rng.sample(range(n), min(k, n))):
            cat = CATEGORY[r["suite"].rsplit("_", 1)[1]]
            rows.append([phase, r["suite"], r["orig_task"], r["split"], cat, r["orig_task"], init_id])

with open("env/eval_plan_pro.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase", "suite", "orig_task", "split", "category", "task_ids_index", "rep"])
    w.writerows(rows)
c = collections.Counter((r[0], r[1]) for r in rows)
print(f"wrote env/eval_plan_pro.csv: {len(rows)} rows")
for (ph, s), v in sorted(c.items()):
    print(f"  {ph:5s} {s:22s} {v}")
if excluded:
    print("excluded (layout unchanged):", excluded)
