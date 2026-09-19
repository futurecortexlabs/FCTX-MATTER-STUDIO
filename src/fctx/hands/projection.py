"""Image space plus a metric hand shape becomes world-space joints.

MediaPipe gives two incomplete halves of the answer.  ``image`` landmarks know
*where* the hand is but are normalised, unitless and have no usable absolute
depth.  ``world`` landmarks are honest metres with correct bone lengths but
their origin is the hand's own centre, so they know the shape and nothing
about position.

This module glues them: estimate the wrist's world position from the image
landmarks, then hang the metric hand off that point.  The result keeps real
bone lengths, which matters because those bones become capsule colliders --
a hand that stretches as it moves away from the camera would push the cloth
with a different sized finger every frame.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import TrackingConfig
from ..core.types import NUM_LANDMARKS, Handedness, HandFrame, L

__all__ = [
    "project_hand",
    "world_to_image_preview",
    "stage_to_image",
    "image_to_stage",
    "estimate_depth",
    "hand_span_image",
    "hand_span_metric",
    "world_chirality",
    "apparent_span",
]

#: Pairs whose length does not change when the fingers move.  Measuring the
#: apparent size from the fingertips instead would make the depth estimate
#: lurch every time the user closes their hand.
_RIGID_PAIRS: tuple[tuple[int, int], ...] = (
    (int(L.WRIST), int(L.INDEX_MCP)),
    (int(L.WRIST), int(L.MIDDLE_MCP)),
    (int(L.WRIST), int(L.PINKY_MCP)),
    (int(L.INDEX_MCP), int(L.PINKY_MCP)),
)

#: An open hand spans roughly twice its palm frame.  The span is measured on
#: the palm for stability and scaled up so that ``reference_hand_span`` can be
#: read as what it says it is: the apparent size of the whole hand.
_PALM_TO_SPAN = 2.0

#: Landmarks that carry MediaPipe's relative-depth cue.  Using only the palm
#: keeps a waving finger out of the depth estimate.
_PALM_LANDMARKS: tuple[int, ...] = (
    int(L.WRIST), int(L.INDEX_MCP), int(L.MIDDLE_MCP),
    int(L.RING_MCP), int(L.PINKY_MCP),
)

#: Below this the hand is a degenerate point and the size cue is meaningless.
_MIN_SPAN = 1.0e-4

_PAIR_A = np.array([p[0] for p in _RIGID_PAIRS], dtype=np.intp)
_PAIR_B = np.array([p[1] for p in _RIGID_PAIRS], dtype=np.intp)


def _aspect(cfg: TrackingConfig) -> float:
    """Vertical-to-horizontal scale of normalised image units.

    MediaPipe divides x by the frame width and y by the frame height, so a
    square object is *not* square in normalised units.  The stage's own aspect
    is the only camera aspect this module is told about, and the two are
    matched by construction, so it is the right correction: without it the
    measured span, and therefore the depth, would depend on which way the
    hand is pointing.
    """
    if cfg.stage_half_width <= 0.0 or cfg.stage_half_height <= 0.0:
        raise ValueError(
            "stage_half_width and stage_half_height must be positive, got "
            f"{cfg.stage_half_width} and {cfg.stage_half_height}")
    return cfg.stage_half_height / cfg.stage_half_width


def hand_span_image(image: np.ndarray, cfg: TrackingConfig) -> float:
    """Apparent size of the hand in normalised horizontal image units."""
    uv = np.asarray(image, dtype=np.float64)[:, :2]
    d = uv[_PAIR_A] - uv[_PAIR_B]
    d[:, 1] *= _aspect(cfg)
    return float(_PALM_TO_SPAN * np.mean(np.hypot(d[:, 0], d[:, 1])))


def hand_span_metric(points: np.ndarray) -> float:
    """The counterpart of :func:`hand_span_image`, in metres.

    Measured in x and y only, because that is what a camera sees: a hand
    turned edge-on really does present a smaller span, and including z here
    would make the image and metric measures disagree by the foreshortening,
    which is exactly the quantity the two are used to convert between.
    """
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    d = xy[_PAIR_A] - xy[_PAIR_B]
    return float(_PALM_TO_SPAN * np.mean(np.hypot(d[:, 0], d[:, 1])))


def apparent_span(z: float, cfg: TrackingConfig) -> float | None:
    """Invert the depth estimate: what span puts a hand at world ``z``?

    Only the size cue is inverted -- MediaPipe's relative-z contribution is
    a per-frame property of the hand's pose and is not recoverable from the
    world joints -- so this is approximate, and it returns ``None`` rather
    than a wrong number when the configuration makes it meaningless.
    """
    blend = min(max(cfg.depth_size_blend, 0.0), 1.0)
    if blend <= 0.0 or cfg.depth_scale == 0.0:
        return None
    denom = 1.0 - (z - cfg.stage_center_z) / (blend * cfg.depth_scale)
    if denom <= 1.0e-4:
        return None
    return cfg.reference_hand_span / denom


def estimate_depth(image: np.ndarray, cfg: TrackingConfig) -> float:
    """World z of the hand, blended from two cues and clamped to ``z_range``.

    Apparent size is the only cue that is absolute -- a pinhole camera makes
    distance proportional to ``1 / apparent size`` -- but it is noisy, because
    it is a difference of landmarks that each wobble.  MediaPipe's z is smooth
    but it is measured relative to the wrist, so on its own it cannot say how
    far away the hand is; it contributes the lean of the palm and otherwise
    pulls the estimate toward the middle of the stage.  Trusting either alone
    gives you drift or chatter; ``depth_size_blend`` picks the mix.
    """
    span = max(hand_span_image(image, cfg), _MIN_SPAN)
    if cfg.reference_hand_span <= 0.0:
        raise ValueError(
            f"reference_hand_span must be positive, got {cfg.reference_hand_span}")
    ratio = cfg.reference_hand_span / span
    z_size = cfg.stage_center_z - cfg.depth_scale * (ratio - 1.0)

    palm_z = float(np.mean(np.asarray(image, dtype=np.float64)[_PALM_LANDMARKS, 2]))
    z_mp = cfg.stage_center_z - cfg.depth_scale * palm_z

    blend = min(max(cfg.depth_size_blend, 0.0), 1.0)
    z = (1.0 - blend) * z_mp + blend * z_size

    lo, hi = cfg.z_range
    if lo > hi:
        raise ValueError(f"z_range must be (lo, hi) with lo <= hi, got {cfg.z_range}")
    return float(min(max(z, lo), hi))


def image_to_stage(u: float, v: float, cfg: TrackingConfig) -> tuple[float, float]:
    """Normalised image position to world x, y on the stage plane.

    The mapping is deliberately depth-independent: the camera's field of view
    is stretched over a fixed box so that the user can always reach the whole
    stage, however far back they stand.  Depth is handled separately by
    :func:`estimate_depth`.
    """
    uu = (1.0 - u) if cfg.mirror else u
    x = (uu - 0.5) * 2.0 * cfg.stage_half_width
    y = cfg.stage_center_y - (v - 0.5) * 2.0 * cfg.stage_half_height
    return float(x), float(y)


def stage_to_image(x: float, y: float, cfg: TrackingConfig) -> tuple[float, float]:
    """Inverse of :func:`image_to_stage`."""
    uu = x / (2.0 * cfg.stage_half_width) + 0.5
    u = (1.0 - uu) if cfg.mirror else uu
    v = 0.5 - (y - cfg.stage_center_y) / (2.0 * cfg.stage_half_height)
    return float(u), float(v)


def world_chirality(handedness: Handedness, cfg: TrackingConfig) -> float:
    """Sign that turns ``(index_mcp - wrist) x (pinky_mcp - wrist)`` outward.

    A right hand seen palm-on has its index MCP to the *left* of its pinky MCP
    in world x, so that cross product comes out pointing into the palm and the
    base sign is negative.  Mirroring the view mirrors the geometry as well: a
    right hand on a mirrored stage has the winding of a left hand even though
    the handedness label still, correctly, says right.  Deriving the palm
    normal from the label alone would point it backwards on exactly half the
    configurations.
    """
    sign = 1.0 if handedness == Handedness.LEFT else -1.0
    return sign * (-1.0 if cfg.mirror else 1.0)


def _inplane_rotation(stage_xy: np.ndarray, local_xy: np.ndarray) -> float:
    """Angle about world +Z that best rotates ``local_xy`` onto ``stage_xy``.

    Closed-form 2D Kabsch over all 21 landmarks.  Using a single bone instead
    would let one badly-tracked landmark spin the whole hand; using none at
    all would trust MediaPipe's world-landmark frame to agree with the image,
    which it does only approximately and which fails loudly when it does not
    (the hand points backwards).
    """
    cross = float(np.sum(local_xy[:, 0] * stage_xy[:, 1]
                         - local_xy[:, 1] * stage_xy[:, 0]))
    dot = float(np.sum(local_xy[:, 0] * stage_xy[:, 0]
                       + local_xy[:, 1] * stage_xy[:, 1]))
    if math.hypot(cross, dot) < 1.0e-12:
        return 0.0
    return math.atan2(cross, dot)


def project_hand(frame: HandFrame, cfg: TrackingConfig) -> np.ndarray:
    """Lift one tracked hand into world space.

    Returns ``(21, 3)`` float32 metres in the simulation frame: +X right,
    +Y up, +Z toward the viewer.
    """
    image = np.asarray(frame.image, dtype=np.float64)
    world = np.asarray(frame.world, dtype=np.float64)
    if image.shape != (NUM_LANDMARKS, 3) or world.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"project_hand needs (21, 3) landmarks, got {image.shape} and "
            f"{world.shape}")
    if not np.isfinite(image).all():
        raise ValueError("project_hand received non-finite image landmarks")
    if not np.isfinite(world).all():
        raise ValueError("project_hand received non-finite world landmarks")

    wrist = int(L.WRIST)
    wx, wy = image_to_stage(float(image[wrist, 0]), float(image[wrist, 1]), cfg)
    wz = estimate_depth(image, cfg)

    # MediaPipe's metric frame is x right, y down, z away from the camera.
    # The simulation wants x right, y up, z toward the viewer, and a mirrored
    # stage flips x as well.
    flip_x = -1.0 if cfg.mirror else 1.0
    local = world - world[wrist]
    local = np.column_stack((flip_x * local[:, 0], -local[:, 1], -local[:, 2]))

    stage = np.empty((NUM_LANDMARKS, 2), dtype=np.float64)
    uu = (1.0 - image[:, 0]) if cfg.mirror else image[:, 0]
    stage[:, 0] = (uu - 0.5) * 2.0 * cfg.stage_half_width
    stage[:, 1] = cfg.stage_center_y - (image[:, 1] - 0.5) * 2.0 * cfg.stage_half_height
    stage -= stage[wrist]

    theta = _inplane_rotation(stage, local[:, :2])
    c, s = math.cos(theta), math.sin(theta)
    rx = c * local[:, 0] - s * local[:, 1]
    ry = s * local[:, 0] + c * local[:, 1]

    out = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
    out[:, 0] = rx + wx
    out[:, 1] = ry + wy
    out[:, 2] = local[:, 2] + wz
    return out


def world_to_image_preview(
    joints: np.ndarray,
    cfg: TrackingConfig,
    width: int = 1,
    height: int = 1,
) -> np.ndarray:
    """World joints back to preview pixels, for drawing the landmark overlay.

    With the default ``width``/``height`` of 1 the result is in normalised
    image units.

    The wrist goes through the stage map, but the rest of the hand cannot:
    the stage map is depth-independent by design, so a hand drawn at stage
    scale would be visibly the wrong size against the camera picture it is
    overlaid on.  The skeleton is therefore scaled about the wrist by the
    apparent size that this hand's depth implies.  Only x and y come back;
    depth was estimated, not measured, so there is no z to invert.
    """
    xyz = np.asarray(joints, dtype=np.float64)
    if xyz.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"world_to_image_preview needs (21, 3), got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("world_to_image_preview received non-finite joints")

    wrist = xyz[int(L.WRIST)]
    u_w, v_w = stage_to_image(float(wrist[0]), float(wrist[1]), cfg)

    scale = 1.0 / (2.0 * cfg.stage_half_width)
    span = apparent_span(float(wrist[2]), cfg)
    if span is not None:
        metric = hand_span_metric(xyz - wrist)
        if metric > _MIN_SPAN:
            scale = span / metric

    rel = xyz - wrist
    flip_x = -1.0 if cfg.mirror else 1.0
    u = u_w + flip_x * scale * rel[:, 0]
    v = v_w - scale * (cfg.stage_half_width / cfg.stage_half_height) * rel[:, 1]
    return np.column_stack((u * float(width), v * float(height))).astype(np.float32)
