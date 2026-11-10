"""Run a queue of jobs one after another, unattended and resumable (LOG-005).

The queue is a JSON-lines file (scripts/make_pilot_queue.py writes one). Each job:
  name      unique name; log goes to <queue_dir>/logs/<name>.log
  conda     conda env to activate (augur, augur-plus, augur-pro)
  cmd       shell command, run from the repo root
  env       extra environment variables (dict)
  done      path that exists only when the job finished (e.g. final.pt, results.csv)
  clean     optional path of a partial output to move aside before (re)running
A job whose `done` path exists is skipped, so re-running the queue resumes where it stopped.
The queue stops at the first failing job (status.csv shows which); fix, then run again.
Usage
  python scripts/run_queue.py runs/queue/pilot.jsonl --dry_run     # show what would run
  nohup python scripts/run_queue.py runs/queue/pilot.jsonl > runs/queue/runner.log 2>&1 &
  add --only <prefix> to run a subset, --stop_after <name> to stop after a job.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("queue")
p.add_argument("--dry_run", action="store_true")
p.add_argument("--only", default=None)
p.add_argument("--stop_after", default=None)
args = p.parse_args()

repo = Path(__file__).resolve().parent.parent
qfile = Path(args.queue)
qdir = qfile.parent
(qdir / "logs").mkdir(parents=True, exist_ok=True)
jobs = [json.loads(line) for line in open(qfile) if line.strip()]
names = [j["name"] for j in jobs]
assert len(names) == len(set(names)), "duplicate job names"

lock = qdir / "runner.lock"
if not args.dry_run:
    if lock.exists():
        pid = lock.read_text().strip()
        if pid and Path(f"/proc/{pid}").exists():
            sys.exit(f"another runner is active (pid {pid}); remove {lock} if that is wrong")
    lock.write_text(str(os.getpid()))

status_f = qdir / "status.csv"
new = not status_f.exists()
status = open(status_f, "a", newline="")
sw = csv.writer(status)
if new:
    sw.writerow(["name", "start", "end", "hours", "rc"])
    status.flush()


def stamp(t=None):
    return time.strftime("%m-%d %H:%M", time.localtime(t or time.time()))


try:
    conda_sh = None
    if not args.dry_run:
        exe = os.environ.get("CONDA_EXE")
        base = str(Path(exe).resolve().parent.parent) if exe else \
            subprocess.run(["conda", "info", "--base"], capture_output=True, text=True).stdout.strip()
        conda_sh = base + "/etc/profile.d/conda.sh"
        assert Path(conda_sh).exists(), f"cannot find {conda_sh}"
    for j in jobs:
        if args.only and not j["name"].startswith(args.only):
            continue
        done = repo / j["done"]
        if done.exists():
            print(f"[{stamp()}] skip  {j['name']} (done)", flush=True)
            continue
        envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in j.get("env", {}).items())
        line = f"source {conda_sh} && conda activate {j['conda']} && cd {shlex.quote(str(repo))} && {envs} {j['cmd']}"
        if args.dry_run:
            print(f"[todo] {j['name']}  ({j['conda']})\n       {envs} {j['cmd']}", flush=True)
            continue
        if j.get("clean"):
            c = repo / j["clean"]
            if c.exists():
                dst = c.with_name(c.name + f".partial_{time.strftime('%m%d_%H%M')}")
                c.rename(dst)
                print(f"[{stamp()}] moved partial output {c} -> {dst.name}", flush=True)
        print(f"[{stamp()}] start {j['name']}", flush=True)
        t0 = time.time()
        with open(qdir / "logs" / f"{j['name']}.log", "a") as log:
            log.write(f"\n===== {stamp()} {line}\n")
            log.flush()
            rc = subprocess.run(["bash", "-c", line], stdout=log, stderr=subprocess.STDOUT).returncode
        hours = (time.time() - t0) / 3600
        ok = rc == 0 and done.exists()
        sw.writerow([j["name"], stamp(t0), stamp(), f"{hours:.2f}", rc if rc else ("ok" if ok else "no-output")])
        status.flush()
        print(f"[{stamp()}] {'done ' if ok else 'FAIL '} {j['name']} in {hours:.2f} h (rc {rc})", flush=True)
        if not ok:
            sys.exit(f"stopped: {j['name']} failed; see {qdir / 'logs' / (j['name'] + '.log')}")
        if args.stop_after and j["name"] == args.stop_after:
            print("stop_after reached", flush=True)
            break
    print(f"[{stamp()}] queue finished", flush=True)
finally:
    if not args.dry_run and lock.exists() and lock.read_text().strip() == str(os.getpid()):
        lock.unlink()
