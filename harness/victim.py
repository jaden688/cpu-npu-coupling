"""Victim: a pinned, single-threaded CPU compute loop whose per-iteration time
is a proxy for effective CPU frequency. This is the fast (kHz-samplable)
observer for the coupling experiment. Writes one row per rep: (unix_ns, dur_ns).

A native (C/Rust) loop is the precision upgrade; a single-thread numpy matmul is
a portable v1 that still resolves DVFS-scale effects (floor ~0.45% CV on this
machine).
"""
import os
# Force single-threaded BLAS BEFORE importing numpy so timing reflects one core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import gc
import csv
import time
import argparse
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


def _prio_map():
    return {
        "normal": psutil.NORMAL_PRIORITY_CLASS,
        "above": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
        "high": psutil.HIGH_PRIORITY_CLASS,
    }


def pin(core, priority="above"):
    """Pin to one core for clean timing. NOTE: keep priority at 'above' (not
    'high') — a HIGH-priority busy loop starves the typeperf telemetry
    collector, blanking the Energy Meter rails. The dedicated core already gives
    clean timing; robust/winsorized stats absorb the rare preemption."""
    if psutil is None:
        print("[victim] psutil missing; not pinning/prioritizing", file=sys.stderr)
        return
    p = psutil.Process()
    try:
        p.cpu_affinity([core])
    except Exception as e:
        print(f"[victim] cpu_affinity failed: {e}", file=sys.stderr)
    try:
        if os.name == "nt":
            p.nice(_prio_map().get(priority, psutil.ABOVE_NORMAL_PRIORITY_CLASS))
    except Exception as e:
        print(f"[victim] priority set failed: {e}", file=sys.stderr)
    try:
        print(f"[victim] affinity={p.cpu_affinity()} priority={priority}",
              file=sys.stderr)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=int, default=4)
    ap.add_argument("--matrix", type=int, default=128)
    ap.add_argument("--inner", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--priority", default="above",
                    choices=["normal", "above", "high"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pin(args.core, args.priority)
    n = args.matrix
    a = np.random.rand(n, n).astype(np.float64)
    b = np.random.rand(n, n).astype(np.float64)

    # warmup
    for _ in range(5):
        c = a
        for _ in range(args.inner):
            c = c @ b

    stop_time = time.time() + args.seconds if args.seconds else None
    inner = args.inner
    ts_buf, du_buf = [], []          # buffer in RAM; no I/O inside the loop
    ts_append, du_append = ts_buf.append, du_buf.append

    gc.collect()
    gc.disable()                     # deterministic refcount frees only
    i = 0
    while True:
        t0 = time.perf_counter_ns()
        c = a
        for _ in range(inner):
            c = c @ b
        t1 = time.perf_counter_ns()
        du_append(t1 - t0)
        ts_append(time.time_ns())
        if c[0, 0] == -12345.0:      # keep result live, never true
            print(c[0, 0])
        i += 1
        if args.reps and i >= args.reps:
            break
        if (i & 0x3FF) == 0 and stop_time and time.time() >= stop_time:
            break
    gc.enable()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unix_ns", "dur_ns"])
        for tsv, duv in zip(ts_buf, du_buf):
            w.writerow([tsv, duv])
    print(f"[victim] wrote {i} reps to {args.out}")


if __name__ == "__main__":
    main()
