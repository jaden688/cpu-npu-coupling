"""Analysis: align victim timing + power rails + load events, then quantify
coupling via (1) phase-folded on/off comparison, (2) lock-in demodulation at the
carrier, (3) a circular-shift permutation null for a p-value.

Baseline runs (no events) report the idle jitter floor and rail rest levels.

Usage:
  py analysis\\lockin.py --latest
  py analysis\\lockin.py --run data\\runs\\<id>
"""
import os
import sys
import csv
import glob
import json
import argparse
import datetime as dt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newest_run():
    runs = sorted(glob.glob(os.path.join(ROOT, "data", "runs", "*")))
    if not runs:
        sys.exit("no runs found")
    return runs[-1]


def load_victim(path):
    t, d = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            t.append(int(row[0]) / 1e9)
            d.append(int(row[1]) / 1e6)   # ms
    return np.array(t), np.array(d)


def load_power(path):
    """Parse typeperf CSV -> {rail_name: (unix_times, values)}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.reader(f))
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return {}
    header = rows[0]
    cols = {i: header[i].split("\\")[-2] + "\\" + header[i].split("\\")[-1]
            if header[i].count("\\") >= 2 else header[i]
            for i in range(1, len(header))}
    times, data = [], {i: [] for i in cols}
    for row in rows[1:]:
        try:
            ts = dt.datetime.strptime(row[0].strip('"'),
                                      "%m/%d/%Y %H:%M:%S.%f").timestamp()
        except Exception:
            continue
        times.append(ts)
        for i in cols:
            try:
                data[i].append(float(row[i]))
            except (ValueError, IndexError):
                data[i].append(np.nan)
    times = np.array(times)
    return {cols[i]: (times, np.array(v)) for i, v in data.items()}


def load_events(path):
    if not os.path.exists(path):
        return []
    ev = []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                ev.append((int(row[0]) / 1e9, int(row[1])))
    return ev


def carrier_from_events(events):
    ons = [t for t, s in events if s == 1]
    if len(ons) < 2:
        return None, None
    period = np.median(np.diff(ons))
    return 1.0 / period, ons[0]


def lockin(t, x, f, t0):
    x = x - np.nanmean(x)
    ph = 2 * np.pi * f * (t - t0)
    I = np.nanmean(x * np.cos(ph))
    Q = np.nanmean(x * np.sin(ph))
    return 2 * np.hypot(I, Q)


def perm_null(t, x, f, t0, n=500):
    amps = np.empty(n)
    for i in range(n):
        shift = np.random.uniform(0, (t[-1] - t[0]))
        amps[i] = lockin(t, x, f, t0 + shift)
    return amps


def on_off_split(t, x, events):
    on = np.zeros(len(t), bool)
    state = 0
    idx = 0
    ev = sorted(events)
    for i, ti in enumerate(t):
        while idx < len(ev) and ev[idx][0] <= ti:
            state = ev[idx][1]
            idx += 1
        on[i] = state == 1
    return x[on], x[~on]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()
    run_dir = args.run or (newest_run() if args.latest else newest_run())

    tv, dv = load_victim(os.path.join(run_dir, "victim.csv"))
    power = load_power(os.path.join(run_dir, "power.csv"))
    events = load_events(os.path.join(run_dir, "events.csv"))

    out = {"run": run_dir, "victim_reps": int(len(dv))}
    print(f"\n=== {os.path.basename(run_dir)} ===")
    print(f"victim reps: {len(dv)}")
    if len(dv):
        cv = 100 * np.std(dv) / np.mean(dv)
        med = float(np.median(dv))
        mad = float(np.median(np.abs(dv - med)))
        rcv = 100 * 1.4826 * mad / med if med else float("nan")
        out["victim_mean_ms"] = float(np.mean(dv))
        out["victim_median_ms"] = med
        out["victim_cv_pct"] = float(cv)
        out["victim_robust_cv_pct"] = float(rcv)
        print(f"victim median={med:.3f} ms  robustCV={rcv:.3f}%   "
              f"(mean={np.mean(dv):.3f}, plainCV={cv:.1f}%, "
              f"p01={np.percentile(dv, 1):.3f}, p99={np.percentile(dv, 99):.3f})")

    for name, (pt, pv) in power.items():
        if np.all(np.isnan(pv)):
            continue
        print(f"  rail {name:38s} mean={np.nanmean(pv):9.1f}  "
              f"sd={np.nanstd(pv):8.1f}")

    if events and len(dv) > 10:
        f, t0 = carrier_from_events(events)
        if f:
            print(f"\ncarrier f = {f:.4f} Hz  (period {1/f:.1f} s)")
            lo, hi = np.percentile(dv, [1, 99])      # winsorize sparse outliers
            dvw = np.clip(dv, lo, hi)
            amp = lockin(tv, dvw, f, t0)
            null = perm_null(tv, dvw, f, t0)
            p = float(np.mean(null >= amp))
            on, off = on_off_split(tv, dvw, events)
            delta = float(np.mean(on) - np.mean(off)) if len(on) and len(off) else float("nan")
            out.update({"carrier_hz": float(f), "victim_lockin_ms": float(amp),
                        "victim_perm_p": p, "victim_on_minus_off_ms": delta})
            print(f"victim lock-in amp = {amp:.4f} ms   perm-p = {p:.3f}")
            print(f"victim on-off delta = {delta:+.4f} ms "
                  f"({100*delta/np.mean(dv):+.2f}% of mean)")
            verdict = "SIGNAL (p<0.05)" if p < 0.05 else "no signal at floor"
            print(f"VERDICT: {verdict}")
            out["verdict"] = verdict
    else:
        print("\n(baseline / no events) -> jitter floor + rail rest levels only")

    try:
        _plot(run_dir, tv, dv, power, events)
        print(f"plot -> {os.path.join(run_dir, 'analysis.png')}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"results -> {os.path.join(run_dir, 'results.json')}")


def _plot(run_dir, tv, dv, power, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t0 = tv[0] if len(tv) else 0
    ax[0].plot(tv - t0, dv, lw=0.6)
    ax[0].set_ylabel("victim rep time (ms)")
    ax[0].set_title(os.path.basename(run_dir))
    import re
    for name, (pt, pv) in power.items():
        if "Power" in name and not np.all(np.isnan(pv)):
            m = re.search(r"\((.*?)\)", name)
            ax[1].plot(pt - t0, pv, label=m.group(1) if m else name)
    ax[1].set_ylabel("power (mW)")
    ax[1].set_xlabel("time (s)")
    if ax[1].get_legend_handles_labels()[1]:
        ax[1].legend(fontsize=6, ncol=3)
    for t, s in events:
        if s == 1:
            ax[0].axvspan(t - t0, t - t0, color="k", alpha=0.05)
    for a in ax:
        prev = None
        for t, s in events:
            if s == 1:
                prev = t - t0
            elif s == 0 and prev is not None:
                a.axvspan(prev, t - t0, color="orange", alpha=0.10)
                prev = None
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "analysis.png"), dpi=110)


if __name__ == "__main__":
    main()
