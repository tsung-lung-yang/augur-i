"""T0 floor gate (LOG-003 section 6, LOG-005).

Reads the per-episode results of T0 (seed 0, S steps) on the pilot plans and applies the rule:
  pass  iff  mean success over the appearance axes   (LIBERO-Plus camera, light, background, noise) >= 10%
        and  mean success over the geometric axes    (LIBERO-Plus robot initial state, object layout;
                                                      LIBERO-PRO Position)                        >= 10%
"Mean" is the unweighted mean of the per-axis success rates (LOG-005). Language is reported only.
Episodes marked "error" are excluded from the rates and counted.
Writes <out>/floor_gate.json always, and <out>/floor_gate_pass.json only when the gate passes
(the pilot queue stops otherwise). Exit code 0 = pass, 2 = fail.
Usage
  python scripts/floor_gate.py --plus runs/eval/pilot/t0_s0/plus/results.csv \
      --pro runs/eval/pilot/t0_s0/pro/results.csv --out runs/eval/pilot/t0_s0
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--plus", required=True)
p.add_argument("--pro", default=None)
p.add_argument("--out", required=True)
p.add_argument("--threshold", type=float, default=0.10)
args = p.parse_args()

GROUPS = {  # keyword in the (lower-cased) category name -> group
    "camera": "appearance", "light": "appearance", "background": "appearance", "noise": "appearance",
    "robot": "geometric", "layout": "geometric", "position": "geometric",
    "language": "report_only",
}


def group_of(cat):
    hits = {g for k, g in GROUPS.items() if k in cat.lower()}
    if len(hits) != 1:
        sys.exit(f"cannot classify axis {cat!r} (matches {hits or 'nothing'})")
    return hits.pop()


rows = list(csv.DictReader(open(args.plus)))
if args.pro:
    rows += list(csv.DictReader(open(args.pro)))
by, err = collections.defaultdict(list), collections.Counter()
for r in rows:
    if r["success"] == "error":
        err[r["category"]] += 1
    else:
        by[r["category"]].append(int(r["success"]))
axes = {}
for cat in sorted(set(by) | set(err)):
    v = by[cat]
    axes[cat] = {"group": group_of(cat), "n": len(v), "success": sum(v), "rate": (sum(v) / len(v)) if v else None,
                 "errors": err[cat]}
groups = {}
for g in ("appearance", "geometric", "report_only"):
    rates = [a["rate"] for a in axes.values() if a["group"] == g and a["rate"] is not None]
    groups[g] = sum(rates) / len(rates) if rates else None
missing = [g for g in ("appearance", "geometric") if groups[g] is None]
passed = not missing and groups["appearance"] >= args.threshold and groups["geometric"] >= args.threshold
res = {"pass": passed, "threshold": args.threshold, "groups": groups, "axes": axes,
       "sources": {"plus": args.plus, "pro": args.pro}, "missing_groups": missing}
out = Path(args.out)
json.dump(res, open(out / "floor_gate.json", "w"), indent=1)
for cat, a in axes.items():
    rate = f"{100 * a['rate']:5.1f}%" if a["rate"] is not None else "  n/a"
    print(f"  {cat:28s} {a['group']:11s} {a['success']:3d}/{a['n']:<3d} {rate}  errors {a['errors']}")
for g, v in groups.items():
    print(f"  mean {g:11s} {'n/a' if v is None else f'{100 * v:.1f}%'}")
print("FLOOR GATE:", "PASS" if passed else "FAIL")
if passed:
    json.dump(res, open(out / "floor_gate_pass.json", "w"), indent=1)
sys.exit(0 if passed else 2)
