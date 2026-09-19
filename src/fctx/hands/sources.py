"""The source interface: something that produces :class:`TrackerFrame`.

Four implementations exist -- live camera, video file, procedural synthetic
hand, and recorded playback -- and the rest of the application cannot tell
them apart.  That is what lets the whole pipeline be developed and tested on
a machine with no camera attached while the camera path stays the real one.
"""

from __future__ import annotations

import abc
import itertools
import time

from ..config import TrackingConfig
from ..core.types import TrackerFrame

__all__ = ["HandSource", "NullSource", "CameraUnavailable", "create_source"]


class CameraUnavailable(RuntimeError):
    """No capture device could be opened.  Carries what was tried and why."""


class HandSource(abc.ABC):
    """A producer of tracker frames.

    ``poll`` must never block the render loop.  Whatever work a source does --
    a USB read, a neural network -- happens on its own thread, and ``poll``
    returns the newest finished result or ``None``.  A source that blocked for
    even one camera frame would cap the whole application at the camera's
    frame rate, which is half the physics rate.
    """

    #: True only for the procedural hand.  The HUD asks every source for this
    #: so that it knows whether to print the mouse-steering help; declaring it
    #: here rather than on one subclass is what stops a caller's ``getattr``
    #: default from quietly answering for a source that forgot to say.
    is_synthetic: bool = False

    @abc.abstractmethod
    def start(self) -> None:
        """Acquire devices and start any worker threads.  Idempotent."""

    @abc.abstractmethod
    def poll(self) -> TrackerFrame | None:
        """Return the newest completed frame, or ``None`` if there is none."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release everything.  Must be safe to call twice, and after a
        failed :meth:`start`."""

    @property
    @abc.abstractmethod
    def describe(self) -> str:
        """One line for the HUD saying where the hands are coming from."""

    def __enter__(self) -> HandSource:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullSource(HandSource):
    """Produces empty frames forever.  The scene runs, nothing touches it."""

    def __init__(self, fps: float = 60.0) -> None:
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._period = 1.0 / fps
        self._counter = itertools.count()
        self._next = 0.0
        self._started = False

    def start(self) -> None:
        self._next = time.perf_counter()
        self._started = True

    def poll(self) -> TrackerFrame | None:
        if not self._started:
            raise RuntimeError("NullSource.poll() before start()")
        now = time.perf_counter()
        if now < self._next:
            return None
        self._next = now + self._period
        return TrackerFrame(hands=[], timestamp=now, index=next(self._counter))

    def close(self) -> None:
        self._started = False

    @property
    def describe(self) -> str:
        return "null (no hands)"


def create_source(cfg: TrackingConfig) -> HandSource:
    """Build the source named by ``cfg.source``.

    MediaPipe and OpenCV are imported lazily so that the synthetic and replay
    paths -- the ones the tests use -- do not pay a multi-second import for
    libraries they never touch.
    """
    name = cfg.source.strip().lower()
    if name == "synthetic":
        from .synthetic import SyntheticSource

        return SyntheticSource(cfg)
    if name == "replay":
        from .recording import ReplaySource

        if cfg.replay_path is None:
            raise ValueError("tracking.source is 'replay' but replay_path is None")
        return ReplaySource(cfg.replay_path, cfg)
    if name == "null":
        return NullSource()
    if name == "camera":
        from .mediapipe_source import CameraSource

        return CameraSource(cfg)
    if name == "video":
        from .mediapipe_source import VideoSource

        if cfg.video_path is None:
            raise ValueError("tracking.source is 'video' but video_path is None")
        return VideoSource(cfg.video_path, cfg)
    raise ValueError(
        f"unknown tracking source {cfg.source!r}; expected one of "
        "'camera', 'video', 'synthetic', 'replay', 'null'")
