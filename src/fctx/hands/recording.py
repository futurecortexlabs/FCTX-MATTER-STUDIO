"""Capture and replay of tracker frames: the ``.fhr`` format.

A recording is what makes a tracking bug reproducible.  Hand input is
unrepeatable by nature -- you cannot wave your hand the same way twice -- so
without this, every change to the filter or the projection is evaluated
against a different input than the last one was.

The container is a compressed ``.npz``: one stacked array per landmark field
plus a JSON header.  Nothing here imports OpenCV or MediaPipe, so a recording
made on the demo machine replays anywhere.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import TrackingConfig
from ..core.types import NUM_LANDMARKS, Handedness, HandFrame, TrackerFrame
from .sources import HandSource

__all__ = ["Recorder", "ReplaySource", "FHR_MAGIC", "FHR_VERSION"]

FHR_MAGIC = "fctx-hand-recording"
FHR_VERSION = 1

#: Preview frames dominate the file size -- 1280x720x3 is 2.6 MB per frame
#: uncompressed, against 500 bytes of landmarks -- so they are off by default
#: and downscaled when they are on.
DEFAULT_PREVIEW_WIDTH = 160


def _subsample(image: np.ndarray, width: int) -> np.ndarray:
    """Downscale by integer striding.

    Deliberately not an interpolating resize: this module stays free of
    OpenCV so that a recording can be replayed on a machine that has no
    camera stack installed at all, and the preview is a thumbnail in the
    corner of the HUD, not something anyone measures.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"preview must be (H, W, 3) RGB, got shape {image.shape}")
    step = max(1, int(image.shape[1] // max(width, 1)))
    return np.ascontiguousarray(image[::step, ::step], dtype=np.uint8)


class Recorder:
    """Accumulate tracker frames and write them out on :meth:`close`.

    Buffering in memory rather than appending to disk is deliberate: a write
    on the frame thread would add unpredictable latency to exactly the code
    path whose latency is being recorded.
    """

    def __init__(
        self,
        path: str | Path,
        max_hands: int = 2,
        store_preview: bool = False,
        preview_width: int = DEFAULT_PREVIEW_WIDTH,
        source: str = "unknown",
    ) -> None:
        if max_hands < 1:
            raise ValueError(f"max_hands must be at least 1, got {max_hands}")
        self.path = Path(path)
        self.max_hands = int(max_hands)
        self.store_preview = bool(store_preview)
        self.preview_width = int(preview_width)
        self.source = str(source)

        self._image: list[np.ndarray] = []
        self._world: list[np.ndarray] = []
        self._count: list[int] = []
        self._handedness: list[np.ndarray] = []
        self._score: list[np.ndarray] = []
        self._timestamp: list[float] = []
        self._index: list[int] = []
        self._preview: list[np.ndarray] = []
        self._closed = False

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        return len(self._timestamp)

    #: Alias used by the HUD, which shows a live frame counter while
    #: recording.
    count = frame_count

    def add(self, frame: TrackerFrame) -> None:
        if self._closed:
            raise RuntimeError(f"Recorder for {self.path} is already closed")
        n = self.max_hands
        image = np.zeros((n, NUM_LANDMARKS, 3), dtype=np.float32)
        world = np.zeros((n, NUM_LANDMARKS, 3), dtype=np.float32)
        handedness = np.zeros(n, dtype=np.int8)
        score = np.zeros(n, dtype=np.float32)
        for i, hand in enumerate(frame.hands[:n]):
            image[i] = hand.image
            world[i] = hand.world
            handedness[i] = int(hand.handedness)
            score[i] = hand.score

        self._image.append(image)
        self._world.append(world)
        self._count.append(min(len(frame.hands), n))
        self._handedness.append(handedness)
        self._score.append(score)
        self._timestamp.append(float(frame.timestamp))
        self._index.append(int(frame.index))
        if self.store_preview:
            if frame.preview is None:
                raise ValueError(
                    f"Recorder was asked to store previews but frame "
                    f"{frame.index} has none")
            self._preview.append(_subsample(frame.preview, self.preview_width))

    def close(self) -> int:
        """Write the file and return how many frames it holds.

        A recorder that never saw a frame writes nothing and reports zero.
        Raising instead would turn the ordinary case of starting and stopping
        the recorder inside one frame into a crash, and would make ``__exit__``
        replace whatever exception was already unwinding the ``with`` block.
        An empty ``.fhr`` would be worse still: :class:`ReplaySource` has no
        frame to hand out and nothing to pace against.
        """
        if self._closed:
            return len(self._timestamp)
        self._closed = True
        if not self._timestamp:
            return 0

        stamps = np.asarray(self._timestamp, dtype=np.float64)
        span = float(stamps[-1] - stamps[0])
        fps = (len(stamps) - 1) / span if span > 0.0 else 0.0
        header = {
            "magic": FHR_MAGIC,
            "version": FHR_VERSION,
            "frames": len(stamps),
            "max_hands": self.max_hands,
            "fps": fps,
            "duration": span,
            "source": self.source,
            "has_preview": bool(self._preview),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        arrays: dict[str, np.ndarray] = {
            "header": np.array(json.dumps(header)),
            "image": np.stack(self._image),
            "world": np.stack(self._world),
            "hand_count": np.asarray(self._count, dtype=np.int32),
            "handedness": np.stack(self._handedness),
            "score": np.stack(self._score),
            "timestamp": stamps,
            "index": np.asarray(self._index, dtype=np.int64),
        }
        if self._preview:
            if len({p.shape for p in self._preview}) != 1:
                raise ValueError(
                    "preview frames changed size mid-recording; a recording "
                    "stores one fixed-size stack")
            arrays["preview"] = np.stack(self._preview)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written through a file object because savez silently appends '.npz'
        # to a path that does not already end in it, and the recording the
        # caller asked for must be at the path the caller named.
        with self.path.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        return len(stamps)


class ReplaySource(HandSource):
    """Play a ``.fhr`` file back through the normal source interface.

    ``realtime=False`` returns exactly one frame per :meth:`poll` with no
    pacing, which is what a test needs in order to reproduce a run frame for
    frame; the default paces playback against the wall clock and drops stale
    frames, exactly as a camera does.
    """

    def __init__(
        self,
        path: str | Path,
        cfg: TrackingConfig,
        realtime: bool = True,
    ) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.realtime = bool(realtime)
        if not self.path.exists():
            raise FileNotFoundError(f"no recording at {self.path}")

        with np.load(self.path, allow_pickle=False) as data:
            try:
                header = json.loads(str(data["header"]))
            except KeyError as exc:
                raise ValueError(
                    f"{self.path} is not an .fhr recording (no header)") from exc
            if header.get("magic") != FHR_MAGIC:
                raise ValueError(
                    f"{self.path} is not an .fhr recording "
                    f"(magic {header.get('magic')!r})")
            if header.get("version") != FHR_VERSION:
                raise ValueError(
                    f"{self.path} is .fhr version {header.get('version')}, "
                    f"this build reads version {FHR_VERSION}")
            self.header = header
            self._image = data["image"]
            self._world = data["world"]
            self._count = data["hand_count"]
            self._handedness = data["handedness"]
            self._score = data["score"]
            self._time = data["timestamp"] - data["timestamp"][0]
            self._preview = data["preview"] if "preview" in data.files else None

        self._frames = int(self._time.shape[0])
        self._cursor = 0
        self._loops = 0
        self._origin = 0.0
        self._counter = 0
        self._started = False

    @property
    def frame_count(self) -> int:
        return self._frames

    @property
    def duration(self) -> float:
        return float(self._time[-1])

    @property
    def describe(self) -> str:
        return (f"replay {self.path.name} "
                f"({self._frames} frames, {self.header.get('fps', 0.0):.1f} fps)")

    def start(self) -> None:
        self._origin = time.perf_counter()
        self._cursor = 0
        self._loops = 0
        self._counter = 0
        self._started = True

    def close(self) -> None:
        self._started = False

    def poll(self) -> TrackerFrame | None:
        if not self._started:
            raise RuntimeError("ReplaySource.poll() before start()")
        if self._cursor >= self._frames:
            if not self.cfg.replay_loop:
                return None
            self._cursor = 0
            self._loops += 1
            self._origin = time.perf_counter()

        if not self.realtime:
            return self._frame_at(self._cursor, post_increment=True)

        elapsed = time.perf_counter() - self._origin
        if self._time[self._cursor] > elapsed:
            return None
        # Skip everything the clock has already passed: a recording must not
        # slow down to wait for a consumer that fell behind, or a replay of a
        # 60 Hz capture would drift further from real time every frame.
        nxt = self._cursor
        while nxt + 1 < self._frames and self._time[nxt + 1] <= elapsed:
            nxt += 1
        self._cursor = nxt
        return self._frame_at(nxt, post_increment=True)

    def _frame_at(self, i: int, post_increment: bool) -> TrackerFrame:
        hands: list[HandFrame] = []
        for h in range(int(self._count[i])):
            hands.append(HandFrame(
                image=np.ascontiguousarray(self._image[i, h], dtype=np.float32),
                world=np.ascontiguousarray(self._world[i, h], dtype=np.float32),
                handedness=Handedness(int(self._handedness[i, h])),
                score=float(self._score[i, h]),
            ))
        preview = None
        if self._preview is not None:
            preview = np.ascontiguousarray(self._preview[i])
        frame = TrackerFrame(
            hands=hands,
            timestamp=self._origin + float(self._time[i]),
            index=self._counter,
            preview=preview,
        )
        if post_increment:
            self._cursor = i + 1
            self._counter += 1
        return frame
