"""Orchestrator: runs a reproducible coupling experiment.

Starts telemetry (typeperf) + the pinned victim as subprocesses, drives the
square-wave accelerator load in-process, and writes a run manifest. Use
--baseline for an idle capture (Experiment 1) that needs numpy only (no onnx).
"""
import os
import sys
import json
import time
import argparse
import platform
import subprocess
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def ort_providers():
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception as e:
        return [f"onnxruntime-unavailable: {e}"]


def load_config(args):
    cfg = {
        "run_name": "baseline" if args.baseline else "run",
        "device": "dml", "victim_core": 4, "victim_priority": "above",
        "victim_matrix": 128, "victim_inner": 8, "load_dim": 1024, "load_depth": 8,
        "cycles": 20, "on_s": 15, "off_s": 15, "sham": False,
        "telemetry_interval_s": 1,
    }
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    if args.seconds:
        cfg["seconds"] = args.seconds
    if args.sham:
        cfg["sham"] = True
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--baseline", action="store_true",
                    help="idle capture, no accelerator load (no onnx needed)")
    ap.add_argument("--seconds", type=float,
                    help="baseline duration; overrides cycle math")
    ap.add_argument("--sham", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args)

    if args.baseline:
        duration = cfg.get("seconds", 30)
    else:
        duration = cfg["cycles"] * (cfg["on_s"] + cfg["off_s"])

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ROOT, "data", "runs", f"{ts}_{cfg['run_name']}")
    os.makedirs(run_dir, exist_ok=True)
    victim_csv = os.path.join(run_dir, "victim.csv")
    power_csv = os.path.join(run_dir, "power.csv")
    events_csv = os.path.join(run_dir, "events.csv")
    model_path = os.path.join(run_dir, "_load.onnx")

    manifest = {
        "timestamp": ts, "config": cfg, "baseline": args.baseline,
        "duration_s": duration, "git_rev": git_rev(),
        "ort_providers_available": ort_providers(),
        "machine": {"node": platform.node(), "processor": platform.processor(),
                    "python": sys.version.split()[0]},
    }

    # 1) telemetry
    import telemetry
    tele = telemetry.start(power_csv, cfg["telemetry_interval_s"])
    print(f"[run] telemetry -> {power_csv}")

    # 2) victim (separate pinned process)
    victim_proc = subprocess.Popen([
        sys.executable, os.path.join(HERE, "victim.py"),
        "--core", str(cfg["victim_core"]),
        "--priority", str(cfg.get("victim_priority", "above")),
        "--matrix", str(cfg["victim_matrix"]),
        "--inner", str(cfg["victim_inner"]),
        "--seconds", str(duration + 1),
        "--out", victim_csv,
    ])
    print(f"[run] victim pinned to core {cfg['victim_core']} -> {victim_csv}")
    time.sleep(1.0)  # let observers settle

    # 3) stimulus
    if args.baseline:
        print(f"[run] BASELINE: idle for {duration}s")
        with open(events_csv, "w") as f:
            f.write("unix_ns,state\n")
        time.sleep(duration)
    else:
        import gpu_load
        if not os.path.exists(model_path):
            gpu_load.build_model(model_path, cfg["load_dim"], cfg["load_depth"])
        sess, providers = gpu_load.make_session(model_path, cfg["device"])
        manifest["ort_providers_used"] = providers
        print(f"[run] load device={cfg['device']} providers={providers} "
              f"sham={cfg['sham']}")
        inp = sess.get_inputs()[0].name
        gpu_load.run(sess, inp, cfg["load_dim"], cfg["cycles"],
                     cfg["on_s"], cfg["off_s"], events_csv, cfg["sham"])

    # 4) teardown
    victim_proc.wait(timeout=duration + 30)
    tele.terminate()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[run] done -> {run_dir}")
    print(f"[run] analyze: py analysis\\lockin.py --run \"{run_dir}\"")


if __name__ == "__main__":
    main()
