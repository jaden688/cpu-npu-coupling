# CPU↔NPU Coupling Lab (Phase 0)

A standalone research bench for characterizing **undocumented electrical and
computational coupling between the CPU and the NPU** on a modern consumer SoC.
Software-only, no oscilloscope required. Spun out of the JL Engine repo to keep
it a clean, independent project.

> **Framing.** We do *not* assume coupling exists. The job of this bench is to
> detect it, quantify it, and — just as importantly — *falsify* it. A negative
> result here is a real upper bound, not "we couldn't see anything."

## Device under test (measured, not assumed)

| Fact | Value | How we know |
|---|---|---|
| SoC | AMD Ryzen AI 9 365 (Strix Point, Zen 5) | `Win32_Processor` |
| iGPU | Radeon 880M (RDNA 3.5) | `Win32_Processor` |
| NPU | XDNA 2, ~50 TOPS | vendor spec |
| Logical CPUs | 20 | `Win32_ComputerSystem` |
| CPU base / boost | 2000 MHz → ~3240 MHz effective (162%) | `Processor Information` |
| Victim timing floor | **CV 0.45%**, boost swing ~1.9% | pinned loop, this repo |
| ORT providers present | `DmlExecutionProvider`, `CPUExecutionProvider` | `onnxruntime` |
| **NPU execution path** | **NOT installed** (no VitisAI/Ryzen AI EP) | see Two Tracks |

### Power rails exposed software-only (`typeperf`, ~1 Hz)

Discovered on this machine — no HWiNFO, no kernel driver, no admin:

```
Apu Power, CPU Power, GPU Power, NPU Power, Socket Power, System Power,
RAPL_Package0_PKG, RAPL_Package0_Core{0..13}_CORE
```

The **NPU Power** rail is the prize: a direct, labeled ground-truth for NPU
activity (0.0 W at idle). It doubles as a built-in falsifier — if it stays 0,
the NPU never ran and any "coupling" is spurious.

## The critical caveat: DirectML is NOT the NPU

`DmlExecutionProvider` on this box routes to the **Radeon 880M iGPU**, not the
XDNA 2 NPU. Running "NPU inference" through ONNX Runtime today would silently
measure *GPU* inference. Hence:

### Two tracks

- **Track A — today, zero installs beyond `onnx`.** Drive the **iGPU** as a
  square-wave load via DML and run the full harness end-to-end. The iGPU shares
  the same PDN, thermal envelope, and fabric as the NPU, so it's a legitimate
  coupling target *and* the correct shakedown of the whole methodology.
- **Track B — after installing the AMD Ryzen AI SDK.** Swap the load generator
  to a true XDNA 2 workload (Ryzen AI / VitisAI EP + INT8-quantized model). The
  NPU Power rail will light up, giving direct source ground-truth. GPU-vs-NPU
  coupling comparison is a first-class result.

## Layers

| Layer | File | Role |
|---|---|---|
| Victim (fast observer) | `harness/victim.py` | Pinned CPU compute loop; per-rep timing = frequency proxy (kHz-samplable) |
| Telemetry (slow observer) | `harness/telemetry.py` | `typeperf` logger for named power rails + effective freq + thermal (~1 Hz) |
| Stimulus | `harness/gpu_load.py` | Square-wave accelerator load (DML now, NPU in Track B); emits event log |
| Orchestration | `harness/run_experiment.py` | Ties it together, writes a reproducible run manifest |
| Analysis | `analysis/lockin.py` | Alignment, lock-in demod, phase-folded on/off test, permutation null, plots |

## Quickstart

```powershell
# 1. (recommended) isolated env
py -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # onnxruntime, onnx, numpy, matplotlib, psutil

# 2. idle baseline (Experiment 1) — needs numpy only, no onnx
py harness\run_experiment.py --baseline --seconds 30

# 3. Track A: CPU<->GPU square-wave coupling run
py harness\run_experiment.py --config config.example.json

# 4. analyze the newest run
py analysis\lockin.py --latest
```

Each run lands in `data/runs/<timestamp>_<name>/` with `manifest.json`,
`victim.csv`, `power.csv`, `events.csv`, and analysis outputs.

## Safety & disclosure

This characterizes a potential **hardware side channel** (one accelerator's
data-dependent activity leaking into another's observable power/timing). If a
run confirms an exploitable channel in shipping silicon, treat it as a
vulnerability: **coordinated disclosure to the vendor**, not a public exploit.
Keep findings disclosure-ready; hold weaponizable detail.

See `docs/PROTOCOL.md` for the experiment runbook and falsification checklist.
