"""Configuration, output folders and small helpers shared by every experiment.

All experiment settings live in CFG and are fixed before a run. Environment variables (QSAW_*) only choose
folders, switch parts on or off, or shrink a run for testing.
"""
import os, re, gc, glob, time, json, heapq, shutil, zipfile, tarfile, pickle, zlib, warnings, ipaddress
import tracemalloc, subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.linalg as sla
from scipy import stats
warnings.filterwarnings("ignore")

CFG = dict(
    OUT=os.environ.get("QSAW_OUT", "/kaggle/working/qsaw_final"),
    RUN_MULVAL=os.environ.get("QSAW_RUN_MULVAL", "1") == "1",
    RUN_FLOW=os.environ.get("QSAW_RUN_FLOW", "1") == "1",
    # ---- Part 1 (protocol fixed before the run; seeds 0-1 and 100-107 were development only)
    SIZES=[25, 50, 75, 100], N_GRAPHS=10, SEED0=200, MAX_TRIES=24, MULVAL_TIMEOUT=2400,
    GT_SAMPLES=20000, GT_SEED=12345, BUDGETS=[10, 30, None],          # None = unlimited attacker
    OMEGAS=[round(0.1 * i, 1) for i in range(11)], MAIN_OMEGA=0.5, DT=0.05,
    NOISE=[0.25, 0.5], MISSING=0.2, MC_BASE=[100, 1000],
    # ---- Part 2 (flow datasets)
    FLOW_ROOT=os.environ.get("QSAW_FLOW_ROOT",
                             "/kaggle/input/datasets/ernie55ernie/improved-cicids2017-and-csecicids2018"),
    FLOW_DATASETS=["CIC2017", "CSE2018"], FLOW_SIZES={50: "5min", 200: "5min", 500: "15min"},
    FLOW_MAXWIN=int(os.environ.get("QSAW_FLOW_MAXWIN", 20)),
)
for k in ("SIZES", "N_GRAPHS", "SEED0", "GT_SAMPLES"):          # test overrides only
    if os.environ.get(f"QSAW_{k}"):
        CFG[k] = json.loads(os.environ[f"QSAW_{k}"])
OUT = CFG["OUT"]; os.makedirs(OUT, exist_ok=True)
TAG = f"{zlib.crc32(json.dumps({k: CFG[k] for k in CFG if k not in ('OUT', 'RUN_MULVAL', 'RUN_FLOW')}, sort_keys=True, default=str).encode()):08x}"
CACHE = f"{OUT}/cache_{TAG}"; FIG = f"{OUT}/figures"
os.makedirs(CACHE, exist_ok=True); os.makedirs(FIG, exist_ok=True)
REPORT = f"{OUT}/RESULTS.md"            # each experiment script calls set_report() to start its own report file
DT = CFG["DT"]; NEG = -1e300
NCPU = os.cpu_count() or 1


def set_report(path, header=None):
    """Point say()/table() at a new markdown report; write a fresh header if given."""
    global REPORT
    REPORT = path; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if header is not None:
        open(REPORT, "w").write(header)
    return REPORT


def bkey(b): return "Binf" if b is None else f"B{b}"


GT_KEYS = [bkey(b) for b in CFG["BUDGETS"]]      # attacker names: B10, B30, Binf (unlimited)


def say(*a):
    s = " ".join(str(x) for x in a); print(s)
    with open(REPORT, "a") as f:
        f.write(s + "\n")


def table(df, title=None, digits=4):
    if title: say(f"\n### {title}")
    say("```\n" + df.round(digits).to_string() + "\n```")


def save(obj, path):
    pickle.dump(obj, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)


try:
    import torch
    GPU = torch.cuda.is_available()
except Exception:
    torch = None; GPU = False
USE_TORCH = (torch is not None) and (GPU or os.environ.get("QSAW_FORCE_TORCH") == "1")
DEV = "cuda" if GPU else "cpu"
mem_gb = int(open("/proc/meminfo").read().split()[1]) / 2**20 if os.path.exists("/proc/meminfo") else float("nan")
say(f"[env] GPU: {torch.cuda.get_device_name(0) if GPU else 'NONE'} | quantum batch backend: "
    f"{'torch-' + DEV if USE_TORCH else 'numpy (slow)'} | CPU cores: {NCPU} | RAM: {mem_gb:.0f} GB | cache: {CACHE}")
if not GPU:
    say("[warn] No GPU: the quantum walks will run on CPU and the notebook may take many hours. "
        "Settings -> Accelerator -> GPU.")


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]
