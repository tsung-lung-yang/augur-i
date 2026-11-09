"""AUGUR-I week-1: environment inventory (no training, no evaluation).

Prints what is installed so the feasibility scripts can target the right
versions. Safe to run in any conda env; every check is independent.
Usage: python scripts/env_probe.py
"""
import importlib
import os
import platform
import shutil
import subprocess
import sys


def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e!r}"


def section(name):
    print(f"\n=== {name} ===")


section("system")
print("python      :", sys.version.split()[0], sys.executable)
print("platform    :", platform.platform())
print("conda env   :", os.environ.get("CONDA_DEFAULT_ENV"))
print("MUJOCO_GL   :", os.environ.get("MUJOCO_GL"))
print("PYOPENGL    :", os.environ.get("PYOPENGL_PLATFORM"))
print("HF_HOME     :", os.environ.get("HF_HOME"))
print("cpu count   :", os.cpu_count())
print("ram         :", sh("free -g | awk '/Mem:/{print $2\" GB total, \"$7\" GB available\"}'"))

section("gpu")
print(sh("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader"))

section("torch")
try:
    import torch
    print("torch       :", torch.__version__, "cuda build", torch.version.cuda)
    print("cuda avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print("device      :", torch.cuda.get_device_name(0))
        print("mem free    : %.1f / %.1f GB" % (free / 1e9, total / 1e9))
        print("bf16        :", torch.cuda.is_bf16_supported())
except Exception as e:  # noqa: BLE001
    print("torch       : NOT AVAILABLE", repr(e))

section("packages")
PKGS = [
    "lerobot", "transformers", "peft", "accelerate", "datasets", "huggingface_hub",
    "timm", "flash_attn", "libero", "robosuite", "mujoco", "gymnasium",
    "decord", "av", "cv2", "einops", "numpy", "wandb",
]
for p in PKGS:
    try:
        m = importlib.import_module(p)
        print(f"{p:16s}: {getattr(m, '__version__', '?'):12s} {os.path.dirname(getattr(m, '__file__', '') or '')}")
    except Exception as e:  # noqa: BLE001
        print(f"{p:16s}: -            ({type(e).__name__})")

section("smolvla policy module")
for mod in ("lerobot.policies.smolvla.modeling_smolvla",
            "lerobot.common.policies.smolvla.modeling_smolvla"):
    try:
        importlib.import_module(mod)
        print("importable  :", mod)
        break
    except Exception as e:  # noqa: BLE001
        print("not found   :", mod, f"({type(e).__name__})")

section("libero benchmark")
try:
    from libero.libero import benchmark
    d = benchmark.get_benchmark_dict()
    print("suites      :", sorted(d.keys()))
    names = d["libero_spatial"]().get_task_names()
    print("spatial[0]  :", names[0])
except Exception as e:  # noqa: BLE001
    print("libero      : not usable", repr(e)[:200])

section("mujoco egl render (subprocess, 30 s timeout)")
code = ("import os; os.environ.setdefault('MUJOCO_GL','egl'); import mujoco; "
        "ctx=mujoco.GLContext(64,64); ctx.make_current(); print('egl ok')")
print(sh(f'{sys.executable} -I -c "{code}"', timeout=30))

section("hugging face cache (model folders)")
hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
if os.path.isdir(hub):
    for d in sorted(os.listdir(hub)):
        if d.startswith(("models--", "datasets--")):
            print(" ", d)
else:
    print("no cache at", hub)

section("disk")
for path in (os.path.expanduser("~"), "/mnt/c", "/mnt/d", "/mnt/e"):
    if os.path.exists(path):
        u = shutil.disk_usage(path)
        print(f"{path:10s}: {u.free / 1e9:7.0f} GB free of {u.total / 1e9:7.0f} GB")

section("conda envs")
print(sh("conda env list"))
