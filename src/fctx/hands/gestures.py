"""Reading intent out of 21 world-space joints.

Everything here is scale-normalised by the hand's own palm.  An absolute
threshold in metres would read "pinch" the moment the user stepped back from
the camera, because a hand projected smaller has smaller gaps between every
pair of landmarks.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.types import Gesture, Handedness, L

__all__ = [
    "hand_scale",
    "pinch_strength",
    "finger_curls",
    "overall_curl",
    "palm_normal",
    "pinch_frame",
    "classify_gesture",
    "mat_to_quat",
    "chirality_for",
]

#: Thumb-to-index gap, in units of palm length, at which the pinch reads as
#: fully closed and as fully open.  0.28 is a fingertip's width apart; beyond
#: 1.05 palm lengths the hand is simply open.
_PINCH_CLOSED = 0.28
_PINCH_OPEN = 1.05

#: Finger chains, base to tip.  The thumb is measured from the CMC because its
#: MCP is where its useful articulation starts.
_FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (int(L.THUMB_CMC), int(L.THUMB_MCP), int(L.THUMB_IP), int(L.THUMB_TIP)),
    (int(L.INDEX_MCP), int(L.INDEX_PIP), int(L.INDEX_DIP), int(L.INDEX_TIP)),
    (int(L.MIDDLE_MCP), int(L.MIDDLE_PIP), int(L.MIDDLE_DIP), int(L.MIDDLE_TIP)),
    (int(L.RING_MCP), int(L.RING_PIP), int(L.RING_DIP), int(L.RING_TIP)),
    (int(L.PINKY_MCP), int(L.PINKY_PIP), int(L.PINKY_DIP), int(L.PINKY_TIP)),
)

#: Tip-to-base distance over total chain length when the finger is as curled
#: as it goes.  The thumb barely folds, so it needs its own floor or it would
#: never report more than half curled.
_CURL_FLOOR: tuple[float, ...] = (0.62, 0.42, 0.40, 0.40, 0.42)

#: No finger may be less folded than this for the hand to count as a fist.
_FIST_MIN_FINGER = 0.55

_CHAIN_IDX = np.array(_FINGER_CHAINS, dtype=np.intp)

_EPS = 1.0e-9


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return np.zeros(3, dtype=np.float64)
    return v / n


def hand_scale(joints: np.ndarray) -> float:
    """Palm length, metres.  The unit every other measure here is divided by."""
    xyz = np.asarray(joints, dtype=np.float64)
    d = float(np.linalg.norm(xyz[int(L.MIDDLE_MCP)] - xyz[int(L.WRIST)]))
    # A collapsed hand would otherwise turn every ratio into an infinity and
    # take the whole frame's gesture state with it.
    return max(d, 1.0e-4)


def pinch_strength(joints: np.ndarray) -> float:
    """0 when thumb and index are wide apart, 1 when they touch."""
    xyz = np.asarray(joints, dtype=np.float64)
    gap = float(np.linalg.norm(xyz[int(L.THUMB_TIP)] - xyz[int(L.INDEX_TIP)]))
    ratio = gap / hand_scale(xyz)
    return _smoothstep((_PINCH_OPEN - ratio) / (_PINCH_OPEN - _PINCH_CLOSED))


def finger_curls(joints: np.ndarray) -> np.ndarray:
    """``(5,)`` float32 curl per finger, thumb first, 0 straight to 1 folded.

    Measured as how far short the tip falls of its own fully-extended reach,
    which needs no joint angles and degrades gracefully when one landmark in
    the chain is badly placed.
    """
    xyz = np.asarray(joints, dtype=np.float64)
    pts = xyz[_CHAIN_IDX]
    seg = np.linalg.norm(pts[:, 1:] - pts[:, :-1], axis=2).sum(axis=1)
    span = np.linalg.norm(pts[:, 3] - pts[:, 0], axis=1)
    ratio = span / np.maximum(seg, _EPS)
    floor = np.asarray(_CURL_FLOOR, dtype=np.float64)
    curl = (1.0 - ratio) / (1.0 - floor)
    return np.clip(curl, 0.0, 1.0).astype(np.float32)


def overall_curl(curls: np.ndarray) -> float:
    """How closed the hand is, 0 flat to 1 fist.

    The thumb is excluded on purpose: it folds across the palm during a pinch,
    and counting it would make every pinch look like the start of a fist.
    """
    return float(np.mean(np.asarray(curls, dtype=np.float64)[1:]))


def palm_normal(joints: np.ndarray, chirality: float) -> np.ndarray:
    """Unit outward normal of the palm.

    ``chirality`` comes from :func:`fctx.hands.projection.world_chirality`; it
    is +1 when the winding wrist -> index MCP -> pinky MCP is right-handed
    about the outward normal and -1 when the stage has mirrored the hand.
    """
    xyz = np.asarray(joints, dtype=np.float64)
    wrist = xyz[int(L.WRIST)]
    n = np.cross(xyz[int(L.INDEX_MCP)] - wrist, xyz[int(L.PINKY_MCP)] - wrist)
    n = _normalize(n) * (1.0 if chirality >= 0.0 else -1.0)
    if float(np.dot(n, n)) < 0.5:
        # Wrist and both MCPs collinear: no plane exists.  Facing the viewer
        # is the least surprising answer and keeps the vector unit length.
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return n.astype(np.float32)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns are the basis vectors) to ``(x, y, z, w)``."""
    m = np.asarray(m, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = (
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = (
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = (
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        )
    out = np.array(q, dtype=np.float64)
    n = float(np.linalg.norm(out))
    if n < _EPS:
        raise ValueError(f"mat_to_quat produced a degenerate quaternion from {m!r}")
    return (out / n).astype(np.float32)


def pinch_frame(
    joints: np.ndarray,
    chirality: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position, quaternion)`` of the frame a pinch grabs with.

    The position is the midpoint of the two fingertips.  The orientation has
    +X along the pinch opening and +Y pointing back down the arm, so that
    twisting the wrist twists whatever is held: a grab that only tracked
    position would let the user carry a body but never turn it over.
    """
    xyz = np.asarray(joints, dtype=np.float64)
    thumb = xyz[int(L.THUMB_TIP)]
    index = xyz[int(L.INDEX_TIP)]
    position = (thumb + index) * 0.5

    x_axis = _normalize(index - thumb)
    if float(np.dot(x_axis, x_axis)) < 0.5:
        # Fingertips coincident at full pinch: the opening direction is gone,
        # so fall back to the palm plane, which is still well defined.
        x_axis = _normalize(xyz[int(L.INDEX_MCP)] - xyz[int(L.THUMB_MCP)])
    if float(np.dot(x_axis, x_axis)) < 0.5:
        x_axis = np.array([1.0, 0.0, 0.0])

    reach = position - xyz[int(L.WRIST)]
    y_axis = _normalize(reach - float(np.dot(reach, x_axis)) * x_axis)
    if float(np.dot(y_axis, y_axis)) < 0.5:
        n = palm_normal(xyz, chirality).astype(np.float64)
        y_axis = _normalize(np.cross(n, x_axis))
    if float(np.dot(y_axis, y_axis)) < 0.5:
        y_axis = _normalize(np.cross(x_axis, np.array([0.0, 0.0, 1.0])))
    if float(np.dot(y_axis, y_axis)) < 0.5:
        y_axis = _normalize(np.cross(x_axis, np.array([0.0, 1.0, 0.0])))

    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = np.cross(z_axis, x_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    return position.astype(np.float32), mat_to_quat(basis)


def classify_gesture(
    pinch: float,
    curls: np.ndarray,
    pinch_threshold: float = 0.62,
    fist_threshold: float = 0.70,
) -> Gesture:
    """Label the hand.

    The fist is tested first, and it has to be: in a closed fist the thumb
    comes to rest beside the index finger, so a pinch measured from fingertip
    distance alone reads a fist as a pinch.  Requiring *every* finger to be
    folded is what separates them -- a real pinch keeps the index partly
    extended to meet the thumb.
    """
    c = np.asarray(curls, dtype=np.float64)
    fingers = c[1:]
    if (float(np.min(fingers)) >= _FIST_MIN_FINGER
            and float(np.mean(fingers)) >= fist_threshold):
        return Gesture.FIST
    if pinch >= pinch_threshold:
        return Gesture.PINCH
    if fingers[0] <= 0.30 and float(np.mean(fingers[1:])) >= 0.55:
        return Gesture.POINT
    return Gesture.OPEN


def chirality_for(handedness: Handedness, mirrored: bool) -> float:
    """Convenience wrapper mirroring :func:`projection.world_chirality`."""
    sign = 1.0 if handedness == Handedness.LEFT else -1.0
    return sign * (-1.0 if mirrored else 1.0)
