"""Raw tracker frames in, usable :class:`HandPose` objects out.

This is where the noisy, unordered, occasionally-missing output of a hand
detector becomes something a physics solver can be driven by: smoothed,
metric, velocity-bearing, and above all *stable* -- the same physical hand
keeps the same slot from the moment it appears until it leaves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import GrabConfig, TrackingConfig
from ..core.types import (
    NUM_LANDMARKS,
    Gesture,
    Handedness,
    HandFrame,
    HandPose,
    L,
    TrackerFrame,
)
from .filters import MAX_DT, ExponentialFilter, OneEuroFilter, VelocityEstimator
from .gestures import (
    classify_gesture,
    finger_curls,
    overall_curl,
    palm_normal,
    pinch_frame,
    pinch_strength,
)
from .projection import project_hand, world_chirality

__all__ = ["HandTracker"]

#: How far a wrist may jump between two frames and still be the same hand.
#: A hand crossing the stage at 3 m/s moves 100 mm between 30 Hz detections;
#: 0.35 m survives a couple of dropped frames without ever being wide enough
#: to swap two hands that are on opposite sides of the stage.
_MATCH_RADIUS = 0.35

#: Speed limit on a joint, m/s.  A single mis-detected frame can otherwise
#: produce a metres-per-millisecond velocity, and that velocity is handed
#: straight to the matter on release.
_MAX_JOINT_SPEED = 8.0

#: Time constant for bleeding off velocity while a hand is being coasted.
#: Coasting at the full last-seen velocity for the whole coast window sends
#: the ghost hand flying off the stage.
_COAST_DAMP_TAU = 0.10

#: Cutoffs, Hz, for the scalar gesture signals.  Lower than the landmark
#: filters because these cross a threshold that starts and stops a grab, and
#: a single frame of chatter there drops whatever the user is holding.
_PINCH_CUTOFF = 5.0
_CURL_CUTOFF = 4.0


def _same_hemisphere(previous: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Keep successive pinch quaternions on one side of the double cover.

    ``q`` and ``-q`` name the same rotation, and :func:`gestures.mat_to_quat`
    chooses its branch from the matrix trace, so a wrist turning smoothly
    through that boundary emits a quaternion that jumps to its antipode -- 27
    times over a minute of the demo path.  Rotating a vector by it is immune
    (the sandwich product is quadratic in ``q``), but anything that
    interpolates between two frames or takes their difference reads the jump
    as a half turn, and twisting a held body is exactly such a difference.
    """
    if float(np.dot(previous, q)) < 0.0:
        return np.ascontiguousarray(-q, dtype=np.float32)
    return q


@dataclass(slots=True)
class _Track:
    """One physical hand, persisting across frames."""

    slot: int
    handedness: Handedness
    pos_filter: OneEuroFilter
    velocity: VelocityEstimator
    pinch_filter: ExponentialFilter
    curl_filter: ExponentialFilter
    joints: np.ndarray
    velocities: np.ndarray
    last_seen: float
    last_update: float
    score: float = 1.0
    confidence: float = 1.0
    pinch: float = 0.0
    curls: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32))
    curl: float = 0.0
    pinching: bool = False
    pinch_since: float | None = None
    gesture: Gesture = Gesture.OPEN
    palm: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))
    pinch_point: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))
    pinch_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))
    pinch_rotation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


class HandTracker:
    """Turn :class:`TrackerFrame` into stable, smoothed :class:`HandPose`.

    Slots are small integers in ``[0, max_hands)`` and they are the contract
    with the solver, which indexes per-hand device arrays by slot.  A slot
    stays reserved for a short while after its hand disappears so that a
    momentary detection dropout does not hand the same physical hand a
    different slot -- which would look to the solver like one hand vanishing
    mid-grab and another appearing.
    """

    def __init__(self, cfg: TrackingConfig, grab: GrabConfig | None = None) -> None:
        if cfg.max_hands < 1:
            raise ValueError(f"max_hands must be at least 1, got {cfg.max_hands}")
        self.cfg = cfg
        self.grab = grab if grab is not None else GrabConfig()
        self._tracks: dict[int, _Track] = {}

    # -- public -----------------------------------------------------------

    def reset(self) -> None:
        self._tracks.clear()

    @property
    def active_slots(self) -> list[int]:
        return sorted(self._tracks)

    def update(self, frame: TrackerFrame | None, now: float) -> list[HandPose]:
        """Advance every track to ``now`` and return the visible hands."""
        if not math.isfinite(now):
            raise ValueError(f"HandTracker.update needs a finite time, got {now}")

        hands: list[HandFrame] = []
        joints: list[np.ndarray] = []
        if frame is not None and frame.hands:
            for hand in frame.hands[: self.cfg.max_hands]:
                hands.append(hand)
                joints.append(project_hand(hand, self.cfg))

        assigned = self._match(hands, joints)
        seen: set[int] = set()
        for det, slot in assigned.items():
            self._advance(self._tracks[slot], hands[det], joints[det], now)
            seen.add(slot)
        for det in range(len(hands)):
            if det in assigned:
                continue
            track = self._spawn(hands[det], joints[det], now)
            if track is not None:
                seen.add(track.slot)

        poses: list[HandPose] = []
        for slot in sorted(self._tracks):
            track = self._tracks[slot]
            if slot in seen:
                poses.append(self._pose(track))
                continue
            state = self._coast(track, now)
            if state == "drop":
                del self._tracks[slot]
            elif state == "visible":
                poses.append(self._pose(track))
        return poses

    # -- association ------------------------------------------------------

    def _match(
        self,
        hands: list[HandFrame],
        joints: list[np.ndarray],
    ) -> dict[int, int]:
        """Greedy nearest-wrist assignment of detections to existing tracks.

        MediaPipe's output order is not stable between frames, so pairing by
        list index silently swaps the two hands' identities the moment the
        detector reorders them -- and with it whatever each hand was holding.
        """
        wrist = int(L.WRIST)
        candidates: list[tuple[float, int, int]] = []
        for det, j in enumerate(joints):
            hand_side = hands[det].handedness
            for slot, track in self._tracks.items():
                if (hand_side != Handedness.UNKNOWN
                        and track.handedness != Handedness.UNKNOWN
                        and hand_side != track.handedness):
                    continue
                d = float(np.linalg.norm(j[wrist] - track.joints[wrist]))
                if d > _MATCH_RADIUS:
                    continue
                candidates.append((d, det, slot))

        candidates.sort()
        assigned: dict[int, int] = {}
        taken: set[int] = set()
        for _, det, slot in candidates:
            if det in assigned or slot in taken:
                continue
            assigned[det] = slot
            taken.add(slot)
        return assigned

    def _free_slot(self) -> int | None:
        for slot in range(self.cfg.max_hands):
            if slot not in self._tracks:
                return slot
        return None

    def _spawn(
        self,
        hand: HandFrame,
        joints: np.ndarray,
        now: float,
    ) -> _Track | None:
        slot = self._free_slot()
        if slot is None:
            # Every slot is held by a track that is still inside its lost
            # timeout.  Dropping the detection is correct: stealing a slot
            # would rip a hand off whatever it is currently holding.
            return None
        cfg = self.cfg
        track = _Track(
            slot=slot,
            handedness=hand.handedness,
            pos_filter=OneEuroFilter(
                cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff),
            velocity=VelocityEstimator(),
            pinch_filter=ExponentialFilter(_PINCH_CUTOFF),
            curl_filter=ExponentialFilter(_CURL_CUTOFF),
            joints=np.array(joints, dtype=np.float32, copy=True),
            velocities=np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
            last_seen=now,
            last_update=now,
        )
        self._tracks[slot] = track
        self._advance(track, hand, joints, now, first=True)
        return track

    # -- per-track update -------------------------------------------------

    def _advance(
        self,
        track: _Track,
        hand: HandFrame,
        joints: np.ndarray,
        now: float,
        first: bool = False,
    ) -> None:
        gap = now - track.last_seen
        if not first and gap > self.cfg.coast_time:
            # The hand was out for longer than we were willing to predict, so
            # the filter state describes a pose that is no longer true and
            # the velocity estimator would read the jump back as a throw.
            track.pos_filter.reset()
            track.velocity.reset()
            first = True
        dt = 0.0 if first else min(max(now - track.last_update, 0.0), MAX_DT)

        smoothed = track.pos_filter.filter(joints, dt)
        velocity = track.velocity.update(smoothed, dt)
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        over = speed > _MAX_JOINT_SPEED
        if bool(np.any(over)):
            velocity = np.where(
                over, velocity * (_MAX_JOINT_SPEED / np.maximum(speed, 1e-9)),
                velocity).astype(np.float32)

        track.joints = np.ascontiguousarray(smoothed, dtype=np.float32)
        track.velocities = np.ascontiguousarray(velocity, dtype=np.float32)
        if hand.handedness != Handedness.UNKNOWN:
            track.handedness = hand.handedness
        track.score = float(min(max(hand.score, 0.0), 1.0))
        track.confidence = track.score
        track.last_seen = now
        track.last_update = now

        track.pinch = track.pinch_filter.filter(pinch_strength(track.joints), dt)
        track.curls = finger_curls(track.joints)
        track.curl = track.curl_filter.filter(overall_curl(track.curls), dt)
        self._update_grip(track, now)

        chirality = world_chirality(track.handedness, self.cfg)
        track.palm = palm_normal(track.joints, chirality)
        track.pinch_point, rotation = pinch_frame(track.joints, chirality)
        track.pinch_rotation = _same_hemisphere(track.pinch_rotation, rotation)
        track.pinch_velocity = (
            (track.velocities[int(L.THUMB_TIP)] + track.velocities[int(L.INDEX_TIP)])
            * np.float32(0.5))
        track.gesture = classify_gesture(
            track.pinch, track.curls, self.grab.start_threshold)

    def _update_grip(self, track: _Track, now: float) -> None:
        """Hysteresis plus a hold time on the pinch that starts a grab.

        One threshold would make a pinch held right at the boundary latch on
        and off every few frames; the hold time stops a fast open-and-close
        wave from snatching whatever it passes over.
        """
        grab = self.grab
        if track.pinching:
            if track.pinch <= grab.release_threshold:
                track.pinching = False
                track.pinch_since = None
            return
        if track.pinch >= grab.start_threshold:
            if track.pinch_since is None:
                track.pinch_since = now
            elif now - track.pinch_since >= grab.hold_time:
                track.pinching = True
        else:
            track.pinch_since = None

    def _coast(self, track: _Track, now: float) -> str:
        """Predict a missing hand forward.  Returns visible / hidden / drop."""
        age = now - track.last_seen
        if age > self.cfg.lost_timeout:
            return "drop"
        if age > self.cfg.coast_time:
            # Past the coast window the prediction is worthless, but the slot
            # stays reserved until lost_timeout so the hand gets it back.
            track.last_update = now
            track.confidence = 0.0
            return "hidden"

        dt = min(max(now - track.last_update, 0.0), MAX_DT)
        track.joints = np.ascontiguousarray(
            track.velocity.extrapolate(track.joints, dt), dtype=np.float32)
        track.velocity.decay(math.exp(-dt / _COAST_DAMP_TAU))
        track.velocities = np.ascontiguousarray(
            track.velocities * math.exp(-dt / _COAST_DAMP_TAU), dtype=np.float32)
        track.last_update = now

        fade = 1.0 - (age / self.cfg.coast_time if self.cfg.coast_time > 0.0 else 1.0)
        track.confidence = float(min(max(track.score * fade, 0.0), 1.0))
        track.pinch_point = np.ascontiguousarray(
            track.pinch_point + track.pinch_velocity * dt, dtype=np.float32)
        return "visible"

    # -- output -----------------------------------------------------------

    def _pose(self, track: _Track) -> HandPose:
        return HandPose(
            joints=track.joints,
            velocities=track.velocities,
            handedness=track.handedness,
            pinch=float(track.pinch),
            pinching=bool(track.pinching),
            curl=float(track.curl),
            gesture=track.gesture,
            palm_normal=track.palm,
            pinch_point=track.pinch_point,
            pinch_velocity=track.pinch_velocity,
            pinch_rotation=track.pinch_rotation,
            confidence=float(track.confidence),
            track_id=track.slot,
        )
