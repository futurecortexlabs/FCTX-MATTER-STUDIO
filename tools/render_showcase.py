"""Render the choreographed demonstration to a video file.

    .venv/Scripts/python.exe tools/render_showcase.py                # docs/showcase.mp4
    .venv/Scripts/python.exe tools/render_showcase.py --size 1920x1080 --fps 60
    .venv/Scripts/python.exe tools/render_showcase.py --out take.mp4 --quick

Everything runs headless in lockstep, so the same command produces the same
frames on any machine that can run the app at all; only the wall-clock time
differs.  Encoding is OpenCV's, so no extra dependency: H.264 in an MP4 where
the platform provides it, MPEG-4 part 2 otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fctx.config import preset  # noqa: E402
from fctx.demo import Choreography  # noqa: E402


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    import cv2

    for fourcc in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                 fps, size)
        if writer.isOpened():
            return writer, fourcc
        writer.release()
    raise SystemExit(f"could not open a video encoder for {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "showcase.mp4")
    ap.add_argument("--size", default="1600x900", metavar="WxH")
    ap.add_argument("--fps", type=float, default=60.0,
                    help="video frame rate; the physics runs at its own 90 Hz")
    ap.add_argument("--quick", action="store_true",
                    help="960x540, bloom and SSAO off, for checking the timing")
    ap.add_argument("--start", type=float, default=0.0, metavar="SECONDS")
    ap.add_argument("--end", type=float, default=None, metavar="SECONDS")
    args = ap.parse_args(argv)

    w, h = (int(v) for v in args.size.lower().split("x"))
    if args.quick:
        w, h = 960, 540

    cfg = preset("cloth")
    render = dataclasses.replace(cfg.render, width=w, height=h,
                                 bloom=not args.quick,
                                 ssao_samples=0 if args.quick else cfg.render.ssao_samples,
                                 show_webcam=False)
    # The video frame rate and the physics rate are decoupled: the app steps
    # once per frame in lockstep, so we run it at the physics rate and keep
    # every (rate / fps)-th frame.  90 / 60 is not an integer, so frames are
    # picked by accumulated time rather than by stride.
    cfg = dataclasses.replace(
        cfg, render=render, headless=True, lockstep=True, demo=True,
        tracking=dataclasses.replace(cfg.tracking, source="synthetic"))

    from fctx.app import MatterStudio

    choreography = Choreography()
    end = args.end if args.end is not None else choreography.duration + 1.5
    physics_dt = 1.0 / cfg.solver.rate_hz
    video_dt = 1.0 / args.fps
    total_frames = int(round(end / physics_dt))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    app = MatterStudio(cfg)
    writer = None
    written = 0
    next_capture = args.start
    started = time.perf_counter()
    try:
        writer, fourcc = open_writer(args.out, args.fps, (w, h))
        print(f"encoding {w}x{h} @ {args.fps:g} fps with {fourcc} -> {args.out}")

        def on_frame(frame_no: int, sim_time: float) -> None:
            nonlocal written, next_capture
            if sim_time + 1e-6 < next_capture:
                return
            next_capture += video_dt
            rgb = app.window.read_pixels()
            writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
            written += 1
            if written % 60 == 0:
                elapsed = time.perf_counter() - started
                print(f"\r  {sim_time:5.1f} s  {written:5d} frames  "
                      f"({elapsed:5.1f} s wall)", end="", flush=True)

        app.on_frame = on_frame
        app.cfg = dataclasses.replace(app.cfg, max_frames=total_frames)
        app.run()
    finally:
        app.close()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - started
    print(f"\nwrote {written} frames ({written / args.fps:.1f} s of video) "
          f"in {elapsed:.1f} s -> {args.out} ({args.out.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
