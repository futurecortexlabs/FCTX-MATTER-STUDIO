"""Pseudo-haptics: make hardness felt with nothing but the picture.

There is no force feedback in a webcam.  What there is, is the hand on the
screen -- and the brain reads resistance from how that hand moves relative
to where it feels its own hand to be.  Slow the drawn hand down as it
presses into something and the something feels stiff; let it sink and it
feels soft.  This is the control/display-ratio manipulation of the
pseudo-haptics literature (Lécuyer et al., 2000; the stiffness variant is
studied in e.g. Argelaguet et al., 2013) and it is measurable: the study
module interleaves trials with it on and off so an installation can find
out whether it helps its visitors tell materials apart, rather than
assuming it does.

The mechanism, per hand:

1. Contact starts when the solver reports the *displayed* hand's capsules
   touching a body (:meth:`fctx.solver.solver.XPBDSolver.hand_contacts`).
   The displayed palm position there is the anchor, and the contact patch's
   surface normal (oriented toward the hand) is the push-back direction.
2. While in contact, the real hand's depth past the anchor along that
   normal is ``d``; the displayed hand goes only ``g * d`` deep, where the
   gain ``g`` falls from ``soft_gain`` at hardness 0 to ``hard_gain`` at
   hardness 1 geometrically, like every other stiffness on the dial.  The
   hand is translated rigidly by ``(1 - g) * d`` along the normal; sideways
   motion passes through untouched, so sliding over a surface still works.
3. Contact ends when the real hand has backed out past the anchor and the
   capsules no longer touch; the offset then relaxes to zero over
   ``relax_time`` rather than snapping.

The displayed hand is what the solver collides with and what grabs, so the
matter and the picture agree: a hard sample under a held-back hand is
indented less, which is also what it would do.  A hand that is holding
something is left alone (its offset relaxes away), because a grab is
anchored to where the hand is and pseudo-haptics is about pressing.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .config import HapticsConfig
from .core.types import HandPose

__all__ = ["PseudoHaptics", "HandContact", "palm_centre"]

#: Wrist and the four finger MCP joints: a point that moves with the hand as
#: a whole and not with the fingers curling.
_PALM = (0, 5, 9, 13, 17)


def palm_centre(pose: HandPose) -> np.ndarray:
    return np.asarray(pose.joints, np.float64)[list(_PALM)].mean(axis=0)


@dataclass(slots=True)
class HandContact:
    """What one hand is touching this frame, per body."""

    #: (bodies,) particles within reach of the hand's capsules.
    count: np.ndarray
    #: (bodies, 3) summed surface normals of those particles.
    normal: np.ndarray
    #: (bodies, 3) summed positions of those particles.
    position: np.ndarray

    def strongest(self, min_count: int) -> int | None:
        if self.count.size == 0:
            return None
        i = int(np.argmax(self.count))
        return i if int(self.count[i]) >= min_count else None


@dataclass(slots=True)
class _Track:
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    anchor: np.ndarray | None = None
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    body: int = -1
    lost: int = 0
    depth_real: float = 0.0
    depth_shown: float = 0.0


class PseudoHaptics:
    """Turn real hand poses into displayed ones, per the contact state."""

    def __init__(self, cfg: HapticsConfig) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self._tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def gain(self, hardness: float) -> float:
        """Displayed depth per unit of real depth at ``hardness``."""
        h = min(max(float(hardness), 0.0), 1.0)
        lo, hi = float(self.cfg.soft_gain), float(self.cfg.hard_gain)
        return float(lo * (hi / lo) ** h)

    def offset(self, track_id: int) -> np.ndarray:
        t = self._tracks.get(track_id)
        return np.zeros(3) if t is None else t.offset.copy()

    def depths(self, track_id: int) -> tuple[float, float]:
        """(real, displayed) press depth of a hand in contact, metres."""
        t = self._tracks.get(track_id)
        return (0.0, 0.0) if t is None else (t.depth_real, t.depth_shown)

    def touching(self, track_id: int) -> int:
        """The body a hand is pressing, or -1."""
        t = self._tracks.get(track_id)
        return -1 if t is None or t.anchor is None else t.body

    def _relax(self, t: _Track, dt: float, tau: float) -> None:
        k = 1.0 - math.exp(-dt / max(tau, 1e-4))
        t.offset = t.offset * (1.0 - k)
        if float(np.linalg.norm(t.offset)) < 1e-5:
            t.offset = np.zeros(3)
        t.depth_real = t.depth_shown = 0.0

    def update(self, poses: list[HandPose], contacts: dict[int, HandContact],
               hardness: list[float], held: set[int], dt: float) -> list[HandPose]:
        """Return the poses to display (and to simulate) this frame.

        ``contacts`` is keyed by track id and describes the *previous*
        frame's displayed hands, which is all there is: the solver reports
        contact after it has stepped.  ``hardness`` is per body.
        """
        cfg = self.cfg
        live = {p.track_id for p in poses}
        for gone in [k for k in self._tracks if k not in live]:
            del self._tracks[gone]
        out: list[HandPose] = []
        for pose in poses:
            t = self._tracks.setdefault(pose.track_id, _Track())
            real = palm_centre(pose)
            if not self.enabled or pose.track_id in held:
                t.anchor = None
                self._relax(t, dt, cfg.hold_relax_time if pose.track_id in held
                            else cfg.relax_time)
            else:
                self._press(t, real, contacts.get(pose.track_id), hardness, dt)
            out.append(_translate(pose, t.offset) if t.offset.any() else pose)
        return out

    def _press(self, t: _Track, real: np.ndarray, contact: HandContact | None,
               hardness: list[float], dt: float) -> None:
        cfg = self.cfg
        body = contact.strongest(cfg.min_contact) if contact is not None else None
        shown = real + t.offset
        if body is not None:
            count = max(int(contact.count[body]), 1)
            centroid = contact.position[body] / count
            n = np.asarray(contact.normal[body], np.float64)
            to_hand = shown - centroid
            if float(np.dot(n, to_hand)) < 0.0:
                n = -n
            length = float(np.linalg.norm(n))
            if length < 1e-6:
                length = float(np.linalg.norm(to_hand))
                n = to_hand if length > 1e-6 else np.array([0.0, 1.0, 0.0])
                length = max(float(np.linalg.norm(n)), 1e-9)
            n = n / length
            if t.anchor is None or body != t.body:
                t.anchor = shown.copy()
                t.normal = n
                t.body = body
            else:
                # Follow a curved surface as the hand slides, but not the
                # frame-to-frame noise of a small contact patch.
                m = t.normal * (1.0 - cfg.normal_follow) + n * cfg.normal_follow
                t.normal = m / max(float(np.linalg.norm(m)), 1e-9)
            t.lost = 0
        elif t.anchor is not None:
            t.lost += 1
            depth = float(np.dot(t.anchor - real, t.normal))
            if t.lost > cfg.release_frames and depth <= 0.0:
                t.anchor = None
            elif t.lost > cfg.release_frames * 6:
                # Slid off the side, or the matter moved away under the hand:
                # nothing is resisting any more, so neither should the picture.
                t.anchor = None
        if t.anchor is None:
            self._relax(t, dt, cfg.relax_time)
            return
        h = hardness[t.body] if 0 <= t.body < len(hardness) else 0.5
        depth_real = max(0.0, float(np.dot(t.anchor - real, t.normal)))
        depth_shown = depth_real * self.gain(h)
        push = min(depth_real - depth_shown, float(cfg.max_offset))
        target = t.normal * push
        k = 1.0 - math.exp(-dt / max(cfg.smoothing, 1e-4))
        t.offset = t.offset + (target - t.offset) * k
        t.depth_real = depth_real
        t.depth_shown = depth_real - float(np.dot(t.offset, t.normal))


def _translate(pose: HandPose, offset: np.ndarray) -> HandPose:
    o = np.asarray(offset, np.float32)
    return dataclasses.replace(
        pose,
        joints=np.asarray(pose.joints, np.float32) + o,
        pinch_point=np.asarray(pose.pinch_point, np.float32) + o,
    )
