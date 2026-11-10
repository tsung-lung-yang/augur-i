"""Set up and index LIBERO-PRO for evaluation (LOG-005). Run in the `augur-pro` env.

1. Copies the released perturbation suites (bddl_files / init_files from the Hugging Face
   dataset zhouxueyang/LIBERO-Pro) into the LIBERO-PRO checkout, without overwriting the
   original LIBERO suites that ship with the repository.
2. Writes a separate LIBERO config (default ~/.libero_pro/config.yaml) pointing at the
   checkout, so the hf-libero and LIBERO-Plus installs are untouched.
3. Imports LIBERO-PRO's benchmark and indexes the four Position suites (libero_<suite>_swap):
   task order versus the base suite, number of init states, and which objects moved (the
   (On obj region) lines that differ from the base task's bddl :init section).
   Output: env/libero_pro_index.csv; the run fails if a task order differs from hf-libero's.
Usage
  python scripts/libero_pro_setup.py --pro_root ~/augur/code/LIBERO-PRO --hf_dir ~/augur/data/libero_pro_hf
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--pro_root", default="~/augur/code/LIBERO-PRO")
p.add_argument("--hf_dir", default="~/augur/data/libero_pro_hf")
p.add_argument("--config_dir", default="~/.libero_pro")
p.add_argument("--axes", default="swap", help="comma list of suffixes to index (swap = Position)")
args = p.parse_args()
root = Path(args.pro_root).expanduser().resolve()
hf = Path(args.hf_dir).expanduser().resolve()
lib = root / "libero" / "libero"
assert (lib / "benchmark" / "__init__.py").exists(), f"not a LIBERO-PRO checkout: {root}"
BASES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# ---------------------------------------------------------------- 1. copy released files
for kind in ("bddl_files", "init_files"):
    srcs = [d for d in hf.rglob(kind) if d.is_dir()]
    assert srcs, f"no {kind} directory under {hf}"
    n_new, n_skip = 0, 0
    for src in srcs:
        for suite in sorted(x for x in src.iterdir() if x.is_dir()):
            dst = lib / kind / suite.name
            if dst.exists():
                n_skip += 1
                continue
            shutil.copytree(suite, dst)
            n_new += 1
    print(f"{kind}: copied {n_new} suites, kept {n_skip} existing ({', '.join(str(s.relative_to(hf)) for s in srcs)})")

# ---------------------------------------------------------------- 2. config
cfg_dir = Path(args.config_dir).expanduser()
cfg_dir.mkdir(parents=True, exist_ok=True)
paths = {"benchmark_root": str(lib), "bddl_files": str(lib / "bddl_files"), "init_states": str(lib / "init_files"),
         "datasets": str(lib.parent / "datasets"), "assets": str(lib / "assets")}
with open(cfg_dir / "config.yaml", "w") as f:
    for k, v in paths.items():
        f.write(f"{k}: {v}\n")
print(f"wrote {cfg_dir / 'config.yaml'}")

# ---------------------------------------------------------------- 3. import and index
os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)
sys.path.insert(0, str(root))
import torch  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402

import libero.libero as LL  # noqa: E402

assert Path(LL.__file__).resolve().is_relative_to(root), f"imported libero from {LL.__file__}, not {root}"
print("libero imported from", Path(LL.__file__).parent)
bench = benchmark.get_benchmark_dict()

split = None
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from make_split import make_split  # noqa: E402

split = make_split()
names_json = here.parent / "env" / "libero_task_names.json"
hf_names = json.load(open(names_json)) if names_json.exists() else None


def init_ons(bddl):
    txt = open(bddl).read()
    m = re.search(r"\(:init(.*?)\(:goal", txt, re.S)
    body = m.group(1) if m else ""
    return set(re.findall(r"\(On\s+(\S+)\s+(\S+)\s*\)", body))


rows, problems = [], []
for b in BASES:
    base = bench[b]()
    base_names = [t.name for t in base.tasks]
    if hf_names is not None:
        ref = hf_names[b] if isinstance(hf_names, dict) else None
        ref = [r["name"] if isinstance(r, dict) else r for r in ref] if ref else None
        if ref is not None and ref != base_names:
            problems.append(f"{b}: LIBERO-PRO base task order differs from hf-libero")
    for ax in args.axes.split(","):
        name = f"{b}_{ax}"
        if name not in bench:
            problems.append(f"{name}: not registered")
            continue
        s = bench[name]()
        if [t.name for t in s.tasks] != base_names:
            problems.append(f"{name}: task order differs from {b}")
        for i, t in enumerate(s.tasks):
            bf = Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
            inf = Path(get_libero_path("init_states")) / t.problem_folder / t.init_states_file
            if not bf.exists() or not inf.exists():
                problems.append(f"{name}[{i}]: missing {'bddl' if not bf.exists() else 'init'} file")
                continue
            n_init = len(torch.load(inf, weights_only=False))
            a = init_ons(Path(get_libero_path("bddl_files")) / base.tasks[i].problem_folder / base.tasks[i].bddl_file)
            c = init_ons(bf)
            moved = sorted(o for o, r in c - a)
            sp = "dev" if i in split[b]["dev"] else "heldout"
            rows.append({"suite": name, "base_suite": b, "orig_task": i, "split": sp, "task": t.name,
                         "n_init": n_init, "perturbed": int(bool(moved)), "moved": ";".join(moved)})

out = here.parent / "env" / "libero_pro_index.csv"
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}: {len(rows)} tasks")
for r in rows:
    print(f"  {r['suite']:22s} {r['orig_task']} {r['split']:7s} init {r['n_init']:3d}  moved: {r['moved'] or '-- (unchanged)'}")
if hf_names is None:
    print("note: env/libero_task_names.json not found; base order not cross-checked against hf-libero")
if problems:
    print("\nPROBLEMS:")
    for x in problems:
        print("  ", x)
    sys.exit(1)
print("\nOK")
