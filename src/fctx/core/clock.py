"""Fixed-timestep clock and rolling performance statistics.

Physics runs at a fixed rate and rendering runs as fast as the display will
take it.  Those two rates are decoupled by an accumulator, which is the only
way to get a simulation whose behaviour does not change when the frame rate
does -- and this application's whole premise is that the material behaves the
same way every time you touch it.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class FixedTimestep:
    """Accumulate real time and hand it out in fixed-size physics steps.

    ``max_steps`` bounds how many steps a single frame may run.  Without it, a
    frame that stalls (a shader compile, a garbage collection, the window being
    dragged) hands the accumulator a huge delta, the next frame tries to catch
    up with dozens of steps, that frame takes even longer, and the application
    never recovers.  Dropping the backlog instead means the simulation runs
    slightly slow for a moment, which nobody notices.
    """

    rate_hz: float = 90.0
    max_steps: int = 5
    #: Clamp on a single real-time delta, seconds.
    max_delta: float = 0.10

    _accumulator: float = field(default=0.0, init=False)
    _last: float = field(default=0.0, init=False)
    _started: bool = field(default=False, init=False)
    #: Steps that were dropped because the backlog exceeded max_steps.
    dropped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.rate_hz > 0.0:
            raise ValueError(f"rate_hz must be positive, got {self.rate_hz}")

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz

    def reset(self) -> None:
        self._accumulator = 0.0
        self._started = False
        self.dropped = 0

    def tick(self, now: float | None = None) -> int:
        """Advance the clock and return how many physics steps to run."""
        now = time.perf_counter() if now is None else now
        if not self._started:
            self._last = now
            self._started = True
            return 0

        # Clamped at both ends.  A backward jump -- a caller passing its own
        # clock, or a platform timer that is not quite monotonic -- gave a
        # negative step count, which `if steps` reads as truthy and
        # `virtual_time += steps * dt` runs backwards, while `range(steps)`
        # quietly steps nothing.
        delta = min(max(now - self._last, 0.0), self.max_delta)
        self._last = now
        self._accumulator += delta

        dt = self.dt
        steps = int(self._accumulator / dt)
        if steps > self.max_steps:
            self.dropped += steps - self.max_steps
            steps = self.max_steps
            self._accumulator = 0.0
        else:
            self._accumulator -= steps * dt
        return steps

    @property
    def alpha(self) -> float:
        """Fraction of the way into the next physics step, for interpolation."""
        return self._accumulator / self.dt


class Rolling:
    """A rolling mean over the last ``n`` samples."""

    __slots__ = ("_values", "_sum")

    def __init__(self, n: int = 90) -> None:
        self._values: deque[float] = deque(maxlen=n)
        self._sum = 0.0

    def push(self, value: float) -> None:
        if len(self._values) == self._values.maxlen:
            self._sum -= self._values[0]
        self._values.append(value)
        self._sum += value

    @property
    def mean(self) -> float:
        return self._sum / len(self._values) if self._values else 0.0

    @property
    def last(self) -> float:
        return self._values[-1] if self._values else 0.0

    def percentile(self, q: float) -> float:
        if not self._values:
            return 0.0
        ordered = sorted(self._values)
        idx = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
        return ordered[idx]

    def clear(self) -> None:
        self._values.clear()
        self._sum = 0.0


class Stopwatch:
    """Context manager that records an elapsed time in milliseconds."""

    __slots__ = ("_target", "_start", "ms")

    def __init__(self, target: Rolling | None = None) -> None:
        self._target = target
        self._start = 0.0
        self.ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
        if self._target is not None:
            self._target.push(self.ms)


@dataclass(slots=True)
class PerfMonitor:
    """Rolling timings for every stage of the frame."""

    window: int = 90
    frame: Rolling = field(init=False)
    physics: Rolling = field(init=False)
    render: Rolling = field(init=False)
    tracking: Rolling = field(init=False)
    _frame_count: int = field(default=0, init=False)
    _tracking_count: int = field(default=0, init=False)
    _last_frame: float = field(default=0.0, init=False)
    _last_tracking: float = field(default=0.0, init=False)
    _tracking_fps: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.frame = Rolling(self.window)
        self.physics = Rolling(self.window)
        self.render = Rolling(self.window)
        self.tracking = Rolling(self.window)

    def begin_frame(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if self._last_frame:
            self.frame.push((now - self._last_frame) * 1000.0)
        self._last_frame = now
        self._frame_count += 1

    def note_tracking_frame(self, now: float | None = None) -> None:
        """Record that a *new* tracker result arrived, for the tracking rate."""
        now = time.perf_counter() if now is None else now
        if self._last_tracking:
            dt = now - self._last_tracking
            if dt > 1e-6:
                # Tracking runs slower and burstier than rendering, so a long
                # window here would lag reality; a light EMA reads better.
                inst = 1.0 / dt
                self._tracking_fps = (0.85 * self._tracking_fps + 0.15 * inst
                                      if self._tracking_fps else inst)
        self._last_tracking = now
        self._tracking_count += 1

    @property
    def fps(self) -> float:
        mean = self.frame.mean
        return 1000.0 / mean if mean > 1e-6 else 0.0

    @property
    def tracking_fps(self) -> float:
        return self._tracking_fps

    @property
    def frame_count(self) -> int:
        return self._frame_count
