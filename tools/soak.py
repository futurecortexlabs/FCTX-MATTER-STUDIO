"""Run the studio unattended for a long time and watch what it costs.

An exhibition leaves the application running for a day.  A leak of a few
megabytes per minute never shows in a test that lasts thirty seconds and
takes the whole installation down at four in the afternoon.  This runs the
real application headless in the configuration a kiosk would use -- camera
wanted but absent, so the retry thread fires every few seconds; attract mode
on, so the demonstration cycles through presets and rebuilds the scene over
and over -- and samples the process every ``--every`` seconds:

* working set of the process (RSS), through ``psapi``;
* dedicated GPU memory of the process, through the Windows performance
  counter ``GPU Process Memory``, since ``nvidia-smi`` cannot attribute
  memory per process under WDDM;
* rolling frame, physics, tracking and render times.

Samples go to a CSV and the summary at the end fits a line through each
memory series and reports the slope per hour, which is the number that
matters: a flat line is a run that can go all day.

    uv run python tools/soak.py --minutes 45
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.wintypes
import dataclasses
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fctx.config import preset  # noqa: E402


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_mb() -> float:
    """Resident memory of this process, in MiB (Windows only; -1 elsewhere)."""
    if sys.platform != "win32":
        return -1.0
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    query = kernel32.K32GetProcessMemoryInfo
    query.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                      ctypes.wintypes.DWORD]
    query.restype = ctypes.wintypes.BOOL
    if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return -1.0
    return counters.WorkingSetSize / (1 << 20)


_GPU_COUNTER_CMD = (
    "(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage' "
    "-ErrorAction Stop).CounterSamples | Measure-Object -Property CookedValue -Sum "
    "| Select-Object -ExpandProperty Sum"
)


def gpu_dedicated_mb(pid: int) -> float:
    """Dedicated GPU memory of ``pid`` in MiB, or -1 if the counter is unavailable."""
    if sys.platform != "win32":
        return -1.0
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _GPU_COUNTER_CMD.format(pid=pid)],
            capture_output=True, text=True, timeout=20, check=False)
        text = out.stdout.strip()
        return float(text) / (1 << 20) if text else -1.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return -1.0


@dataclasses.dataclass
class Sample:
    minutes: float
    frame: int
    rss_mb: float
    gpu_mb: float
    frame_ms: float
    physics_ms: float
    tracking_ms: float
    render_ms: float
    source: str


def _slope_per_hour(samples: list[Sample], key: str) -> float | None:
    pts = [(s.minutes / 60.0, getattr(s, key)) for s in samples if getattr(s, key) >= 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    var = sum((x - mx) ** 2 for x, _ in pts)
    if var == 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / var


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--every", type=float, default=30.0, help="seconds between samples")
    ap.add_argument("--idle-demo", type=float, default=5.0,
                    help="seconds of nobody before the demonstration starts")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "captures" / "soak.csv")
    ap.add_argument("--preset", default="cloth")
    ap.add_argument("--mode", choices=("kiosk", "sim", "demo", "retry"), default="kiosk",
                    help="kiosk: camera wanted+absent, attract mode (the real thing); "
                         "sim: synthetic hand only; demo: the demonstration cycling "
                         "with no camera retries; retry: camera retries, no demo")
    ap.add_argument("--tracemalloc", action="store_true",
                    help="snapshot Python allocations at 1 min and at the end")
    args = ap.parse_args(argv)

    from fctx.app import MatterStudio

    cfg = preset(args.preset)
    wants_camera = args.mode in ("kiosk", "retry")
    cfg = dataclasses.replace(
        cfg, headless=True, lockstep=False, max_frames=0,
        idle_demo=args.idle_demo if args.mode == "kiosk" else 0.0, resilient=True,
        log_file=args.out.with_suffix(".log"),
        tracking=dataclasses.replace(cfg.tracking,
                                     source="camera" if wants_camera else "synthetic"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import os
    pid = os.getpid()
    samples: list[Sample] = []
    started = time.perf_counter()
    next_sample = started
    deadline = started + args.minutes * 60.0

    if args.tracemalloc:
        import tracemalloc
        tracemalloc.start(10)
    baseline = None

    app = MatterStudio(cfg)
    if args.mode == "demo":
        app._start_demo()
    print(f"soak: {args.mode} for {args.minutes:g} min, sampling every {args.every:g} s, "
          f"pid {pid}, source: {app.source_note}")

    def on_frame(frame_no: int, _virtual: float) -> None:
        nonlocal next_sample
        now = time.perf_counter()
        nonlocal baseline
        if args.tracemalloc and baseline is None and now - started >= 60.0:
            baseline = tracemalloc.take_snapshot()
        if now >= next_sample:
            next_sample = now + args.every
            s = Sample(
                minutes=(now - started) / 60.0, frame=frame_no,
                rss_mb=working_set_mb(), gpu_mb=gpu_dedicated_mb(pid),
                frame_ms=app.perf.frame.mean, physics_ms=app.physics_ms.mean,
                tracking_ms=app.tracking_ms.mean, render_ms=app.render_ms.mean,
                source=app.source_note)
            samples.append(s)
            print(f"soak: {s.minutes:6.1f} min  frame {s.frame:8d}  "
                  f"rss {s.rss_mb:7.1f} MiB  gpu {s.gpu_mb:7.1f} MiB  "
                  f"frame {s.frame_ms:5.2f} ms  phys {s.physics_ms:5.2f}  "
                  f"render {s.render_ms:5.2f}  [{s.source}]", flush=True)
        if now >= deadline:
            app.window.request_close()

    app.on_frame = on_frame
    try:
        app.run()
        if args.tracemalloc and baseline is not None:
            final = tracemalloc.take_snapshot()
            print("\nsoak: Python allocations grown since minute 1 (top 12):")
            for stat in final.compare_to(baseline, "lineno")[:12]:
                print(f"  {stat.size_diff / (1 << 10):+9.1f} KiB  "
                      f"{stat.count_diff:+8d} blocks  {stat.traceback[0]}")
            cur, peak = tracemalloc.get_traced_memory()
            print(f"  traced now {cur / (1 << 20):.1f} MiB, peak {peak / (1 << 20):.1f} MiB")
    finally:
        app.close()

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f.name for f in dataclasses.fields(Sample)])
        for s in samples:
            writer.writerow(dataclasses.astuple(s))

    if len(samples) < 3:
        print("soak: too few samples for a verdict")
        return 1
    # The first minute is start-up: kernels compiling, pools warming.  Fit the
    # steady state, not the ramp.
    steady = [s for s in samples if s.minutes >= 1.0] or samples
    rss = _slope_per_hour(steady, "rss_mb")
    gpu = _slope_per_hour(steady, "gpu_mb")
    first, last = steady[0], steady[-1]
    print(f"\nsoak: {samples[-1].minutes:.1f} min, {samples[-1].frame:,} frames, "
          f"{len(samples)} samples -> {args.out}")
    print(f"  rss  {first.rss_mb:7.1f} -> {last.rss_mb:7.1f} MiB   "
          f"slope {rss:+.1f} MiB/h" if rss is not None else "  rss  n/a")
    print(f"  gpu  {first.gpu_mb:7.1f} -> {last.gpu_mb:7.1f} MiB   "
          f"slope {gpu:+.1f} MiB/h" if gpu is not None else "  gpu  n/a")
    print(f"  frame {first.frame_ms:.2f} -> {last.frame_ms:.2f} ms, "
          f"physics {first.physics_ms:.2f} -> {last.physics_ms:.2f} ms")
    verdict_ok = (rss is None or rss < 50.0) and (gpu is None or gpu < 50.0)
    print("  verdict: " + ("flat -- fit to run all day" if verdict_ok
                           else "GROWING -- investigate before an exhibition"))
    return 0 if verdict_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
