"""Measure a real hand on a real camera, so the stage can be set for it.

    uv run python tools/calibrate.py                # camera 0
    uv run python tools/calibrate.py --camera 1 --seconds 30

Hold your hand open in front of the camera and move it around the volume you
want to work in: near, far, left, right, up, down.  When the time is up (or
on Ctrl-C) this prints the statistics the tracking configuration is built
from, and the ``[tracking]`` block to paste into a config file:

* ``reference_hand_span`` -- the apparent size of an open hand at the
  distance you want to be the middle of the stage (the median span seen);
* ``depth_scale`` -- how far the span swings between the nearest and the
  farthest position, so those land at the ends of ``z_range``;
* the fraction of the image the hand actually covered, which says whether
  the camera is too close or too far for the whole stage to be reachable.

Nothing here touches the simulation; it is the camera path alone.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fctx.config import TrackingConfig  # noqa: E402
from fctx.hands.projection import hand_span_image  # noqa: E402
from fctx.hands.sources import CameraUnavailable, create_source  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--size", default="1280x720", metavar="WxH")
    args = ap.parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))

    cfg = TrackingConfig(source="camera", camera_index=args.camera,
                         camera_width=w, camera_height=h, max_hands=1)
    try:
        source = create_source(cfg)
        source.start()
    except CameraUnavailable as exc:
        print(f"calibrate: {exc}")
        return 2

    spans: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    frames = seen = 0
    started = time.perf_counter()
    print(f"calibrating on camera {args.camera} for {args.seconds:.0f} s -- "
          "move an open hand through the whole volume you want to use.")
    try:
        while time.perf_counter() - started < args.seconds:
            frame = source.poll()
            if frame is None:
                time.sleep(0.005)
                continue
            frames += 1
            if not frame.hands:
                continue
            seen += 1
            hand = frame.hands[0]
            spans.append(hand_span_image(hand.image, cfg))
            wrist = hand.image[0]
            xs.append(float(wrist[0]))
            ys.append(float(wrist[1]))
            if seen % 30 == 0:
                print(f"\r  {seen:5d} hand frames  span now {spans[-1]:.3f}  "
                      f"wrist ({xs[-1]:.2f}, {ys[-1]:.2f})", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    print()

    if seen < 30:
        print(f"only {seen} frames with a hand in {frames} captured; "
              "nothing to calibrate from.  Check lighting and framing.")
        return 1

    s = np.asarray(spans)
    x = np.asarray(xs)
    y = np.asarray(ys)
    p = np.percentile
    span_mid = float(np.median(s))
    span_near, span_far = float(p(s, 95)), float(p(s, 5))
    # depth_scale maps the span ratio onto metres; choose it so the 5th and
    # 95th percentiles of what was seen land on the ends of the default
    # z_range, which is 0.8 m deep.
    ratio_far = span_mid / max(span_far, 1e-6)
    ratio_near = span_mid / max(span_near, 1e-6)
    depth_scale = 0.8 / max(ratio_far - ratio_near, 1e-6)

    print(f"hand seen in {seen} of {frames} frames ({100.0 * seen / frames:.0f}%)")
    print(f"apparent span   median {span_mid:.3f}   near {span_near:.3f}   far {span_far:.3f}")
    print(f"wrist x         {p(x, 5):.2f} .. {p(x, 95):.2f}  of the image width")
    print(f"wrist y         {p(y, 5):.2f} .. {p(y, 95):.2f}  of the image height")
    covered = (p(x, 95) - p(x, 5)) * (p(y, 95) - p(y, 5))
    if covered < 0.15:
        print("the hand covered under 15% of the image: the camera is far away "
              "or the volume is small.  Move the camera closer or widen the "
              "stage in the config.")
    print()
    print("paste into your config file:")
    print()
    print("[tracking]")
    print(f"camera_index = {args.camera}")
    print(f"reference_hand_span = {span_mid:.3f}")
    print(f"depth_scale = {min(max(depth_scale, 0.15), 2.0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
