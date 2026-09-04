# Experiment Protocol & Falsification Runbook

## The real null hypothesis

Not "there is no coupling." It is:

> Any CPU-side change observed during accelerator load is explained by
> **(a)** the driver/runtime doing CPU-side work (submission, DMA, interrupts,
> pre/post-processing) plus **(b)** the shared power-budget DVFS policy
> (Hertzbleed-class) — *not* exotic electrical/substrate coupling.

You must subtract (a) and (b) before claiming anything electrical.

## Controls (run every campaign)

1. **Sham control** — identical schedule, no actual compute (`--sham` / baseline).
   The analysis pipeline MUST return null. If it doesn't, the pipeline
   manufactures signal.
2. **NPU-Power falsifier (Track B)** — the `NPU Power` rail must track the
   square wave. If it stays ~0, the accelerator never ran; discard the run.
3. **Frequency-lock test** — pin CPU frequency and repeat:
   ```powershell
   powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99
   powercfg /setactive SCHEME_CURRENT      # 99% max disables boost
   ```
   If the timing channel **vanishes** when frequency is locked → it was DVFS
   policy (b). A residual that **survives** frequency-lock is the genuinely
   interesting electrical/contention signal. Restore with `PROCTHROTTLEMAX 100`.
4. **Scarce-headroom mode** — Settings → Power slider → "Best power efficiency"
   (or run on battery) so accelerator load actually competes for budget. On AC
   with full boost, a small model won't move CPU frequency at all.
5. **Amplitude scaling** — signal should grow with load intensity.
6. **Carrier discrimination** — signal appears at carrier `f`. Energy only at
   `2f` = a mixing/measurement artifact, not coupling.

## Experiment ladder

| # | Name | Load | Key readout |
|---|---|---|---|
| 1 | Idle baseline | none | victim jitter floor, rail rest levels |
| 2 | Continuous load | steady | steady-state rail deltas, freq shift |
| 3 | Square-wave load | modulated | lock-in amplitude at `f` (headline) |
| 4 | CPU timing vs load | modulated | victim demod + phase-folded on/off |
| 5 | Package power | modulated | CPU/APU/PKG rail lock-in |
| 6 | CPU vs GPU vs NPU | per device | comparative coupling (Track B) |
| 7 | Provider audit | — | confirm which device actually executed |

## Detection floor

Single-shot victim CV ≈ **0.45%**. Lock-in over `N` square-wave cycles lowers
the floor ~`sqrt(N)`: 100 cycles → ~0.045% effective-frequency sensitivity.
Choose `on_s + off_s >= 20 s` when you want the ~1 Hz power rails to resolve the
carrier too (Nyquist); use short periods for victim-only fast runs.

## Reproducibility

Every run writes `manifest.json` (config, git rev, machine info, actual ORT
providers used, timestamps). Raw CSVs are content for the data lake. Re-running
the same config against the same manifest is the replication unit.
