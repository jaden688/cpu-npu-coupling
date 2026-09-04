"""Telemetry: log named power rails + effective CPU frequency via the built-in
Windows `typeperf` (no driver, no admin). ~1 Hz update rate. This is the slow,
directly-labeled corroborating observer. The NPU Power rail is ground-truth for
NPU activity.
"""
import subprocess

# Rails confirmed present on the Ryzen AI 9 365 via `typeperf -q "Energy Meter"`.
RAILS = [
    r"\Energy Meter(Apu Power)\Power",
    r"\Energy Meter(CPU Power)\Power",
    r"\Energy Meter(GPU Power)\Power",
    r"\Energy Meter(NPU Power)\Power",
    r"\Energy Meter(Socket Power)\Power",
    r"\Energy Meter(System Power)\Power",
    r"\Energy Meter(RAPL_Package0_PKG)\Power",
    r"\Processor Information(_Total)\% Processor Performance",
    r"\Processor Information(_Total)\Processor Frequency",
    # NOTE: the Thermal Zone counter has a backslash in its instance name
    # (\_tz.thrm) which corrupts the whole typeperf query (all Energy Meter
    # columns go blank). Omitted. If you need die temp, log it in a separate
    # typeperf process with a wildcard: \Thermal Zone Information(*)\Temperature
]


def start(out_csv, interval_s=1):
    """Launch typeperf as a subprocess writing CSV. Returns the Popen handle;
    terminate it to stop logging."""
    args = ["typeperf"] + RAILS + [
        "-si", str(interval_s), "-f", "CSV", "-o", out_csv, "-y",
    ]
    return subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


if __name__ == "__main__":
    import sys
    p = start(sys.argv[1] if len(sys.argv) > 1 else "power.csv")
    print("logging... Ctrl+C to stop")
    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
