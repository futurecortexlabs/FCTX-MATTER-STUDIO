"""When a pinch becomes a grip, and when it lets go.

This is the layer between "the tracker says the fingers are 18 mm apart" and
"the solver is now holding 340 particles".  It is small, but it is the part
people actually feel: a grab that triggers late feels unresponsive, and a grab
that triggers eagerly makes the matter leap into your hand uninvited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GrabConfig
from .core.types import HandPose


def _slot_map_by_track_id(poses: list[HandPose], n_slots: int
                          ) -> dict[int, HandPose]:
    """The same rule ``XPBDSolver.slot_map`` uses, for callers without one."""
    out: dict[int, HandPose] = {}
    spare: list[HandPose] = []
    for pose in poses[:n_slots]:
        slot = int(pose.track_id)
        if 0 <= slot < n_slots and slot not in out:
            out[slot] = pose
        else:
            spare.append(pose)
    free = (s for s in range(n_slots) if s not in out)
    for pose, slot in zip(spare, free):
        out[slot] = pose
    return out


@dataclass(slots=True)
class GripState:
    """One hand's grip, tracked across frames."""

    slot: int
    held: bool = False
    #: How long the pinch has been above the start threshold, seconds.
    closing_for: float = 0.0
    #: Particles currently attached.
    count: int = 0
    #: Seconds since the grip was taken; used to fade in the attachment.
    age: float = 0.0
    #: Set for one frame when the grip is taken or released.
    just_grabbed: bool = False
    just_released: bool = False
    #: Release velocity, for the HUD.
    release_speed: float = 0.0
    _track_id: int = -1

    @property
    def track_id(self) -> int:
        """The hand this grip belongs to (-1 while the slot is empty)."""
        return self._track_id


class GripManager:
    """Decide, per hand slot, when to call ``begin_grab`` / ``end_grab``.

    Hysteresis is the whole job.  MediaPipe's fingertip landmarks move by a
    couple of millimetres frame to frame even when a hand is perfectly still,
    which is enough to cross a single threshold several times a second.  With
    one threshold the object is dropped and re-snatched continuously and the
    interaction feels broken; with a separate, lower release threshold plus a
    short hold time before committing, it feels like picking something up.
    """

    def __init__(self, cfg: GrabConfig, max_hands: int) -> None:
        self.cfg = cfg
        self.grips: list[GripState] = [GripState(slot=i) for i in range(max_hands)]

    def reset(self) -> None:
        for g in self.grips:
            g.held = False
            g.closing_for = 0.0
            g.count = 0
            g.age = 0.0
            g.just_grabbed = False
            g.just_released = False
            g.release_speed = 0.0
            g._track_id = -1

    def update(self, poses: list[HandPose], dt: float, solver: object
               ) -> list[GripState]:
        """Drive ``solver.begin_grab`` / ``solver.end_grab`` for every slot.

        ``solver`` is duck-typed so this module stays importable without a GPU.
        """
        cfg = self.cfg
        # The solver decides which slot a hand occupies, and this has to agree
        # with it exactly -- a grab is a conversation about a slot number.
        # Two plausible rules exist (the pose's track id, or its position in
        # this list) and they diverge the moment a lower-numbered hand leaves,
        # because HandTracker compacts its output: the survivor keeps its id
        # and changes position.  Disagreeing there leaves the manager holding
        # a grab the solver has already dropped, the HUD reporting particles
        # nobody is holding, and that hand unable to grab again for the rest
        # of the session.  So there is one authority, and it is the solver;
        # asking it is not a nicety.  The fallback is for the duck-typed
        # stubs the tests drive this with.
        slot_map = getattr(solver, "slot_map", None)
        if callable(slot_map):
            by_slot: dict[int, HandPose] = slot_map(poses)
        else:
            by_slot = _slot_map_by_track_id(poses, len(self.grips))

        for grip in self.grips:
            grip.just_grabbed = False
            grip.just_released = False
            pose = by_slot.get(grip.slot)

            if pose is None:
                # The hand vanished.  Drop whatever it was holding rather than
                # leaving particles pinned to a ghost position.
                if grip.held:
                    solver.end_grab(grip.slot, None)  # type: ignore[attr-defined]
                    grip.held = False
                    grip.count = 0
                    grip.just_released = True
                    grip.release_speed = 0.0
                grip.closing_for = 0.0
                grip._track_id = -1
                continue

            if pose.track_id != grip._track_id and grip.held:
                # A different hand took this slot; do not hand the grip over.
                solver.end_grab(grip.slot, pose)  # type: ignore[attr-defined]
                grip.held = False
                grip.count = 0
                grip.just_released = True
            grip._track_id = pose.track_id

            if grip.held:
                grip.age += dt
                if pose.pinch < cfg.release_threshold or pose.confidence < 0.15:
                    solver.end_grab(grip.slot, pose)  # type: ignore[attr-defined]
                    grip.held = False
                    grip.count = 0
                    grip.age = 0.0
                    grip.just_released = True
                    grip.release_speed = float(
                        np.linalg.norm(pose.pinch_velocity))
                    grip.closing_for = 0.0
            else:
                if pose.pinch >= cfg.start_threshold and pose.confidence > 0.4:
                    grip.closing_for += dt
                    if grip.closing_for >= cfg.hold_time:
                        n = int(solver.begin_grab(grip.slot, pose))  # type: ignore[attr-defined]
                        grip.count = n
                        grip.age = 0.0
                        # Closing on empty air must not latch: if nothing was
                        # in reach, stay unheld so the next frame can try again
                        # as the hand moves into the matter.  Retrying every
                        # frame costs a whole-scene device readback inside
                        # begin_grab (measured +0.29 ms/frame on cloth and
                        # +0.45 ms on 24k grains), and it is still the right
                        # trade: the alternative is a retry interval, which
                        # delays the grab by exactly that interval at the one
                        # moment the user is reaching for the matter.
                        grip.held = n > 0
                        grip.just_grabbed = grip.held
                        if not grip.held:
                            grip.closing_for = cfg.hold_time
                else:
                    grip.closing_for = 0.0

        return self.grips

    @property
    def total_held(self) -> int:
        return sum(g.count for g in self.grips if g.held)

    @property
    def any_held(self) -> bool:
        return any(g.held for g in self.grips)


@dataclass(slots=True)
class Highlight:
    """A short-lived on-screen annotation, e.g. 'GRABBED 312 particles'."""

    text: str
    ttl: float
    age: float = 0.0

    @property
    def alpha(self) -> float:
        # Hold at full opacity for the first 60% of the life, then fade, so
        # the message is readable in a screen recording played at speed.
        t = self.age / max(self.ttl, 1e-6)
        return 1.0 if t < 0.6 else max(0.0, 1.0 - (t - 0.6) / 0.4)


class Notifier:
    """A tiny queue of transient HUD messages."""

    def __init__(self, limit: int = 4) -> None:
        self.items: list[Highlight] = []
        self.limit = limit

    def post(self, text: str, ttl: float = 1.8) -> None:
        self.items.append(Highlight(text=text, ttl=ttl))
        if len(self.items) > self.limit:
            del self.items[0: len(self.items) - self.limit]

    def update(self, dt: float) -> list[Highlight]:
        for item in self.items:
            item.age += dt
        self.items = [i for i in self.items if i.age < i.ttl]
        return self.items
