"""The live paths: a webcam and a video file, both through MediaPipe.

Two rules shape this whole module.

First, ``poll`` may never block.  OpenCV's ``read`` blocks until the next
sensor exposure and MediaPipe's inference takes several milliseconds; doing
either on the render thread pins the whole application to the camera's frame
rate, which is well under the physics rate.  So capture lives on its own
daemon thread, inference runs in ``LIVE_STREAM`` mode with a callback, and
``poll`` only ever reads a finished result out of a one-slot mailbox.

Second, a late frame is worse than no frame.  The queue holds exactly one
image and a new capture overwrites it, because a hand position from 80 ms ago
is not a smaller error than a missing one -- it is a wrong one that the
solver will happily integrate.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from ..config import HAND_MODEL_PATH, TrackingConfig
from ..core.types import NUM_LANDMARKS, Handedness, HandFrame, TrackerFrame
from .sources import CameraUnavailable, HandSource

__all__ = ["CameraSource", "VideoSource", "CameraUnavailable"]

#: Capture backends to try, in order.  MSMF is the modern Windows path and
#: the only one that reliably delivers 60 fps at 720p; DSHOW is the fallback
#: for cameras with no Media Foundation driver; CAP_ANY lets OpenCV pick,
#: which is what works on Linux and macOS.
_BACKENDS: tuple[tuple[str, int], ...] = (
    ("CAP_MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
    ("CAP_DSHOW", getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)),
    ("CAP_ANY", cv2.CAP_ANY),
)

#: Seconds to wait for the capture thread to notice it has been stopped.
_JOIN_TIMEOUT = 2.0

#: Preview frames held while their inference is in flight.  MediaPipe silently
#: skips frames submitted while it is busy and never calls back for them, so
#: the only thing that evicts an entry is a *later* result arriving; until the
#: first one does -- a cold model can take several hundred milliseconds -- this
#: map would grow by a full 1280x720x3 image on every poll.
_MAX_PENDING = 8


def _resolve_model(model_path: str | Path | None) -> Path:
    path = Path(model_path) if model_path is not None else HAND_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"hand landmarker model not found at {path}. Run "
            "'python tools/download_models.py' to fetch it.")
    return path


def _flip_handedness(side: Handedness) -> Handedness:
    if side == Handedness.LEFT:
        return Handedness.RIGHT
    if side == Handedness.RIGHT:
        return Handedness.LEFT
    return side


class _MediaPipeBase(HandSource):
    """Shared landmarker setup and result conversion."""

    def __init__(self, cfg: TrackingConfig, model_path: str | Path | None) -> None:
        self.cfg = cfg
        self._model_path = _resolve_model(model_path)
        self._landmarker: vision.HandLandmarker | None = None
        self._last_ts_ms = -1

    def _make_options(self, **extra: object) -> vision.HandLandmarkerOptions:
        return vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self._model_path)),
            num_hands=self.cfg.max_hands,
            min_hand_detection_confidence=self.cfg.min_detection_confidence,
            min_hand_presence_confidence=self.cfg.min_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            **extra,  # type: ignore[arg-type]
        )

    def _next_timestamp_ms(self, seconds: float) -> int:
        """Strictly increasing millisecond stamps.

        MediaPipe rejects a timestamp that is not ahead of the previous one,
        and two captures inside the same millisecond are perfectly normal at
        60 fps with a jittery clock, so equal stamps have to be nudged rather
        than passed through.
        """
        ts = int(seconds * 1000.0)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def _convert(self, result: object) -> list[HandFrame]:
        landmarks = getattr(result, "hand_landmarks", None) or []
        world = getattr(result, "hand_world_landmarks", None) or []
        handed = getattr(result, "handedness", None) or []

        hands: list[HandFrame] = []
        for i, lm in enumerate(landmarks):
            if i >= len(world):
                break
            image_arr = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
            world_arr = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
            if len(lm) != NUM_LANDMARKS or len(world[i]) != NUM_LANDMARKS:
                raise ValueError(
                    f"MediaPipe returned {len(lm)} image and {len(world[i])} "
                    f"world landmarks; this pipeline is built for "
                    f"{NUM_LANDMARKS}")
            for j, p in enumerate(lm):
                image_arr[j] = (p.x, p.y, p.z)
            for j, p in enumerate(world[i]):
                world_arr[j] = (p.x, p.y, p.z)

            side = Handedness.UNKNOWN
            score = 1.0
            if i < len(handed) and handed[i]:
                entry = handed[i][0]
                name = str(entry.category_name)
                side = (Handedness.LEFT if name == "Left"
                        else Handedness.RIGHT if name == "Right"
                        else Handedness.UNKNOWN)
                score = float(entry.score)
                # MediaPipe names the hand as if the image were already a
                # selfie-view mirror.  When the stage mirrors as well, the
                # label describes the other hand than the one on screen.
                if self.cfg.mirror:
                    side = _flip_handedness(side)

            hands.append(HandFrame(image=image_arr, world=world_arr,
                                   handedness=side, score=score))
        return hands

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


class _CaptureThread(threading.Thread):
    """Reads the camera as fast as it will go, keeping only the newest frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        super().__init__(name="fctx-capture", daemon=True)
        self._capture = capture
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, float] | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                ok, bgr = self._capture.read()
                if not ok:
                    # A transient read failure is normal while a camera
                    # renegotiates its format; only a persistent one matters,
                    # and the poll side reports that as a stale mailbox.
                    time.sleep(0.005)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = (rgb, time.perf_counter())
        except BaseException as exc:  # surfaced on the next poll
            self._error = exc
        finally:
            # The device is released here rather than by close(), because
            # close() cannot afford to wait indefinitely for the join: a USB
            # stall can hold read() for seconds, and releasing a capture that
            # another thread is still reading from faults inside the driver.
            with self._lock:
                self._frame = None
            self._capture.release()

    def take(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

    def stop(self) -> None:
        self._stop.set()

    @property
    def error(self) -> BaseException | None:
        return self._error


def open_capture(cfg: TrackingConfig) -> cv2.VideoCapture:
    """Open the configured camera, trying each backend in turn."""
    tried: list[str] = []
    for name, backend in _BACKENDS:
        capture = cv2.VideoCapture(cfg.camera_index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
            capture.set(cv2.CAP_PROP_FPS, cfg.camera_fps)
            # One frame of driver buffering; anything more shows up directly
            # as tracking latency because OpenCV hands out the oldest frame.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = capture.read()
            if ok:
                return capture
            capture.release()
            tried.append(f"{name} (opened but delivered no frames)")
            continue
        capture.release()
        tried.append(f"{name} (would not open)")
    raise CameraUnavailable(
        f"could not open camera index {cfg.camera_index}. Tried: "
        + "; ".join(tried)
        + ". Check that a camera is plugged in, that no other application "
        "is holding it, and that camera access is enabled for desktop apps "
        "in the operating system's privacy settings.")


class CameraSource(_MediaPipeBase):
    """Live webcam through MediaPipe's ``LIVE_STREAM`` mode."""

    def __init__(
        self,
        cfg: TrackingConfig,
        model_path: str | Path | None = None,
    ) -> None:
        super().__init__(cfg, model_path)
        self._capture: cv2.VideoCapture | None = None
        self._thread: _CaptureThread | None = None
        self._pending: dict[int, np.ndarray] = {}
        self._pending_lock = threading.Lock()
        self._result: TrackerFrame | None = None
        self._result_lock = threading.Lock()
        self._counter = 0
        self._origin = 0.0

    @property
    def describe(self) -> str:
        return (f"camera {self.cfg.camera_index} "
                f"({self.cfg.camera_width}x{self.cfg.camera_height} @ "
                f"{self.cfg.camera_fps} fps)")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._capture = open_capture(self.cfg)
        self._origin = time.perf_counter()
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(
                self._make_options(
                    running_mode=vision.RunningMode.LIVE_STREAM,
                    result_callback=self._on_result))
        except BaseException:
            self._capture.release()
            self._capture = None
            raise
        self._thread = _CaptureThread(self._capture)
        self._thread.start()

    def _on_result(self, result: object, output_image: object, ts_ms: int) -> None:
        with self._pending_lock:
            preview = self._pending.pop(ts_ms, None)
            # Any stamp older than the one that just came back will never be
            # claimed: MediaPipe delivers in order and would have returned it
            # first.  Dropping them here is what keeps this dict bounded.
            for stale in [k for k in self._pending if k < ts_ms]:
                del self._pending[stale]
        frame = TrackerFrame(
            hands=self._convert(result),
            timestamp=self._origin + ts_ms / 1000.0,
            index=self._counter,
            preview=preview,
        )
        self._counter += 1
        with self._result_lock:
            self._result = frame

    def _bound_pending(self) -> None:
        """Forget the oldest previews whose inference never came back."""
        with self._pending_lock:
            while len(self._pending) > _MAX_PENDING:
                del self._pending[min(self._pending)]

    def poll(self) -> TrackerFrame | None:
        if self._thread is None or self._landmarker is None:
            raise RuntimeError("CameraSource.poll() before start()")
        error = self._thread.error
        if error is not None:
            raise CameraUnavailable(
                f"the capture thread for camera {self.cfg.camera_index} died: "
                f"{error!r}")

        taken = self._thread.take()
        if taken is not None:
            rgb, captured_at = taken
            ts_ms = self._next_timestamp_ms(captured_at - self._origin)
            with self._pending_lock:
                self._pending[ts_ms] = rgb
            self._bound_pending()
            self._landmarker.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

        with self._result_lock:
            frame, self._result = self._result, None
        return frame

    def close(self) -> None:
        if self._thread is not None:
            self._thread.stop()
            self._thread.join(_JOIN_TIMEOUT)
            self._thread = None
            # Ownership of the device passed to the thread when it started;
            # it releases the capture on its way out, even if the join above
            # gave up waiting for it.
            self._capture = None
        super().close()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        with self._pending_lock:
            self._pending.clear()


class VideoSource(_MediaPipeBase):
    """A video file, paced to its own frame rate.

    Uses ``RunningMode.VIDEO`` rather than ``LIVE_STREAM``: inference is
    synchronous, which is what makes playback deterministic, and the reader
    thread absorbs the cost so ``poll`` still returns immediately.
    """

    def __init__(
        self,
        path: str | Path,
        cfg: TrackingConfig,
        model_path: str | Path | None = None,
        loop: bool | None = None,
    ) -> None:
        super().__init__(cfg, model_path)
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no video file at {self.path}")
        self.loop = cfg.replay_loop if loop is None else bool(loop)
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._result: TrackerFrame | None = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._counter = 0
        self._fps = 30.0
        self._origin = 0.0

    @property
    def describe(self) -> str:
        return f"video {self.path.name} ({self._fps:.1f} fps)"

    def start(self) -> None:
        if self._thread is not None:
            return
        capture = cv2.VideoCapture(str(self.path), cv2.CAP_ANY)
        if not capture.isOpened():
            capture.release()
            raise CameraUnavailable(
                f"OpenCV could not open the video file {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        # A container with no frame-rate metadata reports 0; pacing on that
        # would divide by zero and play the whole file in one frame.
        self._fps = fps if 1.0 <= fps <= 480.0 else 30.0
        self._capture = capture
        self._origin = time.perf_counter()
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(
                self._make_options(running_mode=vision.RunningMode.VIDEO))
        except BaseException:
            capture.release()
            self._capture = None
            raise
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="fctx-video", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Bound to locals up front: close() clears both attributes as soon as
        # it has asked this thread to stop, and reading them through self
        # would turn a join that timed out into an AttributeError here.
        capture = self._capture
        landmarker = self._landmarker
        assert capture is not None and landmarker is not None
        try:
            period = 1.0 / self._fps
            due = time.perf_counter()
            while not self._stop.is_set():
                ok, bgr = capture.read()
                if not ok:
                    if not self.loop:
                        return
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                now = time.perf_counter()
                ts_ms = self._next_timestamp_ms(now - self._origin)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
                frame = TrackerFrame(
                    hands=self._convert(result),
                    timestamp=now,
                    index=self._counter,
                    preview=rgb,
                )
                self._counter += 1
                with self._lock:
                    self._result = frame
                due += period
                delay = due - time.perf_counter()
                if delay > 0.0:
                    self._stop.wait(delay)
                else:
                    due = time.perf_counter()
        except BaseException as exc:
            self._error = exc
        finally:
            capture.release()

    def poll(self) -> TrackerFrame | None:
        if self._thread is None:
            raise RuntimeError("VideoSource.poll() before start()")
        if self._error is not None:
            raise RuntimeError(
                f"the decode thread for {self.path} died: {self._error!r}")
        with self._lock:
            frame, self._result = self._result, None
        return frame

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(_JOIN_TIMEOUT)
            self._thread = None
            # The decode thread owns the file handle and releases it itself,
            # for the same reason the camera's capture thread does.
            self._capture = None
        super().close()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
