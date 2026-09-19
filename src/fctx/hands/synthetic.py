"""A procedural hand, so the application is never blocked on hardware.

The synthetic hand is not a stub.  It builds a real 21-landmark skeleton with
plausible bone lengths in the same metric frame MediaPipe uses, then projects
it into normalised image coordinates and hands both halves to the tracker.
Everything downstream -- the depth estimate, the in-plane alignment, the
One-Euro filters, the gesture classifier -- runs exactly as it does on camera
input.  A synthetic source that skipped the projection would test the tracker
against its own output and prove nothing.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from ..config import TrackingConfig
from ..core.types import NUM_LANDMARKS, Handedness, HandFrame, L, TrackerFrame
from .projection import hand_span_metric, stage_to_image
from .sources import HandSource

__all__ = ["SyntheticSource", "build_metric_hand"]

# Anthropometry, metres, for an average adult hand.  The frame is MediaPipe's
# metric one: +x right, +y down, +z away from the camera.  The canonical pose
# is a right hand, palm toward the camera, fingers up.
_MCP = np.array(
    [
        [-0.021, -0.092, -0.004],   # index
        [-0.003, -0.090, -0.002],   # middle
        [+0.015, -0.085, -0.001],   # ring
        [+0.032, -0.077, +0.003],   # pinky
    ],
    dtype=np.float64,
)
_PHALANX = np.array(
    [
        [0.040, 0.024, 0.019],
        [0.045, 0.027, 0.020],
        [0.041, 0.026, 0.019],
        [0.031, 0.019, 0.017],
    ],
    dtype=np.float64,
)
#: Sideways fan of each finger when the hand is open, as a slope against the
#: finger direction.  Fingers converge as they close, which is why this is
#: scaled down by the curl.
_SPLAY = (-0.10, -0.02, 0.06, 0.16)

#: Flexion added at the MCP, PIP and DIP joints at full curl, radians.  These
#: are cumulative down the chain, so the tip ends up folded into the palm.
_JOINT_FLEX = (1.40, 1.70, 1.00)

_THUMB_CMC = np.array([-0.029, -0.020, -0.007], dtype=np.float64)
_THUMB_REST_DIRS = np.array(
    [
        [-0.62, -0.72, -0.31],
        [-0.48, -0.80, -0.36],
        [-0.38, -0.86, -0.34],
    ],
    dtype=np.float64,
)
_THUMB_LENGTHS = (0.037, 0.031, 0.026)

#: How far the thumb tip travels toward the middle phalanges at full curl.
#: All the way would drive it through the fingers it is meant to lie across.
_THUMB_FOLD = 0.70

#: A pinch is not just a thumb movement: the index has to come to meet it.
#: With a straight index the thumb is 60 mm short of the fingertip and no
#: amount of thumb rotation closes that gap.
_PINCH_INDEX_CURL = 0.62

#: Apparent hand span, in normalised image units, at the far and near ends of
#: the usable depth range.  Chosen so the resulting world z stays comfortably
#: inside TrackingConfig.z_range for the default stage.
_SPAN_FAR = 0.135
_SPAN_NEAR = 0.430

#: Seconds of unchanged manual input before the autopilot takes the hand
#: back, and how long it takes to fade in once it does.  The application
#: pushes the cursor state in every frame whether the cursor moved or not,
#: so "manual control" has to mean a *changed* value rather than a call --
#: otherwise the autopilot could never run inside the real app, and a demo
#: with nobody at the mouse would show a hand standing perfectly still.
_AUTO_IDLE = 1.5
_AUTO_FADE = 0.8

_PALM_AWAY = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_EPS = 1.0e-12


def _mix(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _unit_interval(who: str, value: float) -> float:
    """Clamp a control input to [0, 1], rejecting a non-finite one.

    ``np.clip`` propagates NaN rather than clamping it, so clipping alone
    would store a NaN control value, build a hand full of NaN landmarks out
    of it, and only fail two subsystems later inside ``project_hand``.
    """
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{who} needs a finite value in [0, 1], got {value}")
    return min(max(v, 0.0), 1.0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < _EPS:
        raise ValueError("cannot normalise a zero-length vector in the hand rig")
    return v / n


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def _fabrik(
    root: np.ndarray,
    lengths: tuple[float, ...],
    rest: np.ndarray,
    target: np.ndarray,
    iterations: int = 8,
) -> np.ndarray:
    """Solve a chain to reach ``target``, keeping every bone length exact.

    Seeded from the rest pose so the solution keeps the thumb's natural arc
    instead of snapping to whichever of the many valid configurations the
    solver happens to find first.
    """
    pts = rest.copy()
    pts[0] = root
    total = float(sum(lengths))
    delta = target - root
    dist = float(np.linalg.norm(delta))
    if dist > total * 0.999:
        direction = delta / max(dist, _EPS)
        for i, length in enumerate(lengths):
            pts[i + 1] = pts[i] + length * direction
        return pts
    for _ in range(iterations):
        pts[-1] = target
        for i in range(len(lengths) - 1, -1, -1):
            d = pts[i] - pts[i + 1]
            n = float(np.linalg.norm(d))
            pts[i] = pts[i + 1] + (d / n if n > _EPS else _PALM_AWAY) * lengths[i]
        pts[0] = root
        for i in range(len(lengths)):
            d = pts[i + 1] - pts[i]
            n = float(np.linalg.norm(d))
            pts[i + 1] = pts[i] + (d / n if n > _EPS else -_PALM_AWAY) * lengths[i]
    return pts


def _finger(index: int, curl: float) -> np.ndarray:
    """Four points (MCP, PIP, DIP, TIP) of one finger, metres."""
    splay = _SPLAY[index] * (1.0 - 0.6 * curl)
    direction = _unit(np.array([splay, -1.0, -0.06], dtype=np.float64))
    # Flexion happens about the knuckle line, which is perpendicular to both
    # the finger and the palm normal; deriving it rather than hard-coding +x
    # is what makes splayed fingers converge correctly as they close.
    axis = _unit(np.cross(direction, _PALM_AWAY))

    out = np.empty((4, 3), dtype=np.float64)
    out[0] = _MCP[index]
    angle = 0.0
    for j in range(3):
        angle += _JOINT_FLEX[j] * curl
        out[j + 1] = out[j] + _PHALANX[index, j] * _rotate(direction, axis, angle)
    return out


def _thumb(
    index_tip: np.ndarray,
    fold_point: np.ndarray,
    pinch: float,
    curl: float,
) -> np.ndarray:
    """Four points (CMC, MCP, IP, TIP) of the thumb, metres.

    The thumb has two jobs and they are not the same motion: opposing the
    index tip for a pinch, and folding across the middle phalanges for a
    fist.  Driving it from the pinch alone leaves the thumb sticking straight
    out of a closed fist.
    """
    rest = np.empty((4, 3), dtype=np.float64)
    rest[0] = _THUMB_CMC
    for j in range(3):
        rest[j + 1] = rest[j] + _THUMB_LENGTHS[j] * _unit(_THUMB_REST_DIRS[j])
    target = rest[3] + (index_tip - rest[3]) * float(np.clip(pinch, 0.0, 1.0))
    target = target + (fold_point - target) * (_THUMB_FOLD * float(np.clip(curl, 0.0, 1.0)))
    return _fabrik(_THUMB_CMC, _THUMB_LENGTHS, rest, target)


def build_metric_hand(pinch: float, curl: float) -> np.ndarray:
    """A right hand as ``(21, 3)`` float32 metres in MediaPipe's world frame.

    The origin is the hand's geometric centre, matching what
    ``hand_world_landmarks`` reports, so this array can be dropped straight
    into a :class:`~fctx.core.types.HandFrame`.
    """
    pinch = _unit_interval("build_metric_hand(pinch=...)", pinch)
    curl = _unit_interval("build_metric_hand(curl=...)", curl)

    fingers = [_finger(i, max(curl, _PINCH_INDEX_CURL * pinch) if i == 0 else curl)
               for i in range(4)]
    thumb = _thumb(fingers[0][3], fingers[1][1], pinch, curl)

    pts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
    pts[int(L.THUMB_CMC):int(L.THUMB_TIP) + 1] = thumb
    for i, chain in enumerate(fingers):
        base = int(L.INDEX_MCP) + 4 * i
        pts[base:base + 4] = chain
    pts -= pts.mean(axis=0)
    return pts.astype(np.float32)


class SyntheticSource(HandSource):
    """A fully articulated procedural hand under manual or automatic control.

    ``realtime=False`` turns the internal clock into a virtual one that steps
    by exactly one frame per :meth:`poll`, which is what makes a 300-frame
    test finish in milliseconds and produce bit-identical output every run.
    """

    is_synthetic = True

    def __init__(
        self,
        cfg: TrackingConfig,
        auto: bool = True,
        fps: float = 60.0,
        realtime: bool = True,
    ) -> None:
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.cfg = cfg
        self.auto = bool(auto)
        self.fps = float(fps)
        self.realtime = bool(realtime)

        self._started = False
        self._origin = 0.0
        self._clock = 0.0
        self._next = 0.0
        self._reset_pose()

    def _reset_pose(self) -> None:
        """Return the rig to the pose it is constructed in.

        Without this a source that is stopped and started again resumes from
        wherever the last run left the hand, so two runs of the same script
        produce different landmarks -- and a recording made to reproduce a
        bug replays against a different input than the one that caused it.
        """
        self._pointer = (0.5, 0.5)
        self._span = math.sqrt(_SPAN_FAR * _SPAN_NEAR)
        self._pinch = 0.0
        self._curl = 0.0
        self._auto_blend = 0.0
        self._last_manual = self._manual_state()
        self._manual_at = 0.0
        self._counter = itertools.count()

    def _manual_state(self) -> tuple[float, float, float, float, float]:
        return (*self._pointer, self._span, self._pinch, self._curl)

    # -- manual control ---------------------------------------------------

    def set_pointer(self, nx: float, ny: float) -> None:
        """Place the wrist, in normalised stage coordinates.

        ``(0, 0)`` is the top-left of the interaction volume and ``(1, 1)``
        the bottom-right, in *world* terms -- so moving right always moves the
        hand right, whether or not the stage is mirrored.
        """
        self._pointer = (_unit_interval("set_pointer(nx=...)", nx),
                         _unit_interval("set_pointer(ny=...)", ny))

    def set_depth(self, z01: float) -> None:
        """0 pushes the hand to the back of the stage, 1 brings it forward."""
        t = _unit_interval("set_depth", z01)
        # Geometric, not linear: apparent size goes as 1/distance, so a
        # linear sweep of the span would crawl at the back of the stage and
        # jump at the front.
        self.set_span(_SPAN_FAR * (_SPAN_NEAR / _SPAN_FAR) ** t)

    def set_span(self, span: float) -> None:
        """Set the apparent hand size directly, in normalised image units.

        This is the quantity the depth estimator actually reads, so it is
        also the honest way to place the hand at a known apparent distance.
        """
        value = float(span)
        if not math.isfinite(value):
            raise ValueError(f"apparent hand span must be finite, got {span}")
        if not 0.001 <= value <= 4.0:
            raise ValueError(
                f"apparent hand span {value} is outside the plausible range "
                "[0.001, 4.0] of normalised image units")
        self._span = value

    @property
    def span(self) -> float:
        return self._span

    def set_pinch(self, t01: float) -> None:
        self._pinch = _unit_interval("set_pinch", t01)

    def set_curl(self, t01: float) -> None:
        self._curl = _unit_interval("set_curl", t01)

    def autopilot_targets(self, t: float) -> tuple[float, float, float, float, float]:
        """The path, as ``(nx, ny, depth, pinch, curl)`` at time ``t``.

        The three periods share no small common multiple, so the path does
        not visibly repeat for minutes and a demo left running unattended
        does not look looped.
        """
        phase = math.fmod(t, 5.0)
        ramp = 0.35
        if phase < ramp:
            g = phase / ramp
        elif phase < ramp + 1.6:
            g = 1.0
        elif phase < 2.0 * ramp + 1.6:
            g = 1.0 - (phase - ramp - 1.6) / ramp
        else:
            g = 0.0
        return (
            0.5 + 0.34 * math.sin(2.0 * math.pi * t / 7.3),
            0.5 + 0.24 * math.sin(2.0 * math.pi * t / 4.7 + 1.1),
            0.5 + 0.34 * math.sin(2.0 * math.pi * t / 11.0),
            g * g * (3.0 - 2.0 * g),
            0.10 + 0.08 * math.sin(2.0 * math.pi * t / 3.1),
        )

    def autopilot(self, t: float, blend: float = 1.0) -> None:
        """Move the hand onto the autopilot path at time ``t``.

        ``blend`` mixes between the current manual pose and the path, which
        is how :meth:`poll` fades control over instead of cutting: a cut
        teleports the hand from wherever the cursor left it to wherever the
        path happens to be, and the tracker turns that jump into a velocity
        spike the solver would faithfully act on.
        """
        nx, ny, depth, pinch, curl = self.autopilot_targets(t)
        b = float(np.clip(blend, 0.0, 1.0))
        span_now = self._span
        self.set_depth(depth)
        span_target = self._span

        self.set_pointer(_mix(self._pointer[0], nx, b),
                         _mix(self._pointer[1], ny, b))
        self.set_span(_mix(span_now, span_target, b))
        self.set_pinch(_mix(self._pinch, pinch, b))
        self.set_curl(_mix(self._curl, curl, b))

    # -- HandSource -------------------------------------------------------

    def start(self) -> None:
        # Idempotent, as the HandSource contract requires: re-entering start()
        # on a running source would rewind the autopilot clock and teleport
        # the hand back to the beginning of its path mid-demo.
        if self._started:
            return
        self._reset_pose()
        self._origin = time.perf_counter()
        self._clock = self._origin
        self._next = self._origin
        # Never touched means already idle: the autopilot starts immediately
        # on a source nobody is steering.
        self._manual_at = -math.inf
        self._started = True

    def close(self) -> None:
        self._started = False

    @property
    def describe(self) -> str:
        mode = "autopilot" if self.auto else "manual"
        return f"synthetic hand ({mode}, {self.fps:.0f} fps)"

    def poll(self) -> TrackerFrame | None:
        if not self._started:
            raise RuntimeError("SyntheticSource.poll() before start()")
        if self.realtime:
            now = time.perf_counter()
            if now < self._next:
                return None
            # Re-base rather than accumulate: after a stall, catching up by
            # emitting a burst of frames would hand the tracker a pile of
            # sub-millisecond dt values and spike every velocity.
            self._next = now + 1.0 / self.fps
            self._clock = now
        else:
            self._clock += 1.0 / self.fps
            now = self._clock

        self._arbitrate(now)
        return self._build(now)

    def _arbitrate(self, now: float) -> None:
        """Hand control to the autopilot once the manual inputs go quiet."""
        state = self._manual_state()
        if state != self._last_manual:
            self._last_manual = state
            self._manual_at = now
            self._auto_blend = 0.0
        if not self.auto:
            return
        if now - self._manual_at < _AUTO_IDLE:
            return
        self._auto_blend = min(
            1.0, self._auto_blend + (1.0 / self.fps) / _AUTO_FADE)
        self.autopilot(now - self._origin, self._auto_blend)
        # The autopilot writes through the same setters a user would, so its
        # own output must not be mistaken for someone taking the wheel.
        self._last_manual = self._manual_state()

    # -- frame construction -----------------------------------------------

    def _build(self, now: float) -> TrackerFrame:
        cfg = self.cfg
        metric = build_metric_hand(self._pinch, self._curl).astype(np.float64)
        local = metric - metric[int(L.WRIST)]

        span_metric = hand_span_metric(metric)
        if span_metric <= 0.0:
            raise ValueError("synthetic hand collapsed to a point")
        scale = self._span / span_metric

        tx = (self._pointer[0] - 0.5) * 2.0 * cfg.stage_half_width
        ty = cfg.stage_center_y - (self._pointer[1] - 0.5) * 2.0 * cfg.stage_half_height
        u_w, v_w = stage_to_image(tx, ty, cfg)

        # Normalised image units are anisotropic (x is divided by the frame
        # width, y by its height); the stage's aspect is the correction, and
        # it has to be applied here too or the projection will measure a
        # different span than the one this frame was built to have.
        aspect = cfg.stage_half_width / cfg.stage_half_height

        image = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
        image[:, 0] = u_w + scale * local[:, 0]
        image[:, 1] = v_w + scale * aspect * local[:, 1]
        image[:, 2] = scale * local[:, 2]

        hand = HandFrame(
            image=image,
            world=np.ascontiguousarray(metric, dtype=np.float32),
            handedness=Handedness.RIGHT,
            score=1.0,
        )
        return TrackerFrame(hands=[hand], timestamp=now, index=next(self._counter))
