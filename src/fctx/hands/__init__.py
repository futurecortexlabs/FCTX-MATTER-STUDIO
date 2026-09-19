"""Camera pixels to :class:`~fctx.core.types.HandPose`.

Nothing in this package imports ``warp`` or ``moderngl``: hand tracking is a
CPU problem and keeping it that way is what lets the whole subsystem be
tested on a machine with neither a GPU nor a camera.

``CameraSource`` and ``VideoSource`` are resolved lazily, because importing
them drags in OpenCV and MediaPipe -- several seconds and a few hundred
megabytes that the synthetic and replay paths never need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .filters import ExponentialFilter, OneEuroFilter, VelocityEstimator
from .gestures import (
    classify_gesture,
    finger_curls,
    hand_scale,
    overall_curl,
    palm_normal,
    pinch_frame,
    pinch_strength,
)
from .projection import (
    estimate_depth,
    hand_span_image,
    hand_span_metric,
    image_to_stage,
    project_hand,
    stage_to_image,
    world_chirality,
    world_to_image_preview,
)
from .recording import Recorder, ReplaySource
from .sources import CameraUnavailable, HandSource, NullSource, create_source
from .synthetic import SyntheticSource, build_metric_hand
from .tracker import HandTracker

if TYPE_CHECKING:
    from .mediapipe_source import CameraSource, VideoSource

__all__ = [
    "CameraSource",
    "CameraUnavailable",
    "ExponentialFilter",
    "HandSource",
    "HandTracker",
    "NullSource",
    "OneEuroFilter",
    "Recorder",
    "ReplaySource",
    "SyntheticSource",
    "VelocityEstimator",
    "VideoSource",
    "build_metric_hand",
    "classify_gesture",
    "create_source",
    "estimate_depth",
    "finger_curls",
    "hand_scale",
    "hand_span_image",
    "hand_span_metric",
    "image_to_stage",
    "overall_curl",
    "palm_normal",
    "pinch_frame",
    "pinch_strength",
    "project_hand",
    "stage_to_image",
    "world_chirality",
    "world_to_image_preview",
]

_LAZY = {"CameraSource", "VideoSource"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import mediapipe_source

        return getattr(mediapipe_source, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
