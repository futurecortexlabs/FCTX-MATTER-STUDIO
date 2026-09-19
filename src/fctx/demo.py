"""A choreographed demonstration: grab, lift, sweep the dial, let go.

The application's autopilot wanders; this walks. It exists so the one thing
the project is about -- the same body changing material *while it is being
held* -- can be shown on demand, in a window with the ``D`` key or into a
video file with ``tools/render_showcase.py``, identically every time.

The choreography is a list of keyframes in the synthetic hand's own control
space (pointer, depth, pinch, curl) plus the dial and the preset, and it is
interpolated with a smoothstep so nothing ever jumps.  Keyframes are in
seconds of simulation time, which under lockstep is frames divided by the
physics rate, so a rendered video and a live run agree to the frame.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Key:
    t: float
    #: Synthetic hand controls.  pointer is normalised stage space, origin
    #: top-left; depth 0 is the back of the stage and 1 the front.
    nx: float
    ny: float
    depth: float
    pinch: float
    curl: float
    #: Where the dial is being dragged to.
    hardness: float
    #: Preset to be on at this moment.  A change is applied once, when the
    #: timeline crosses the keyframe.
    preset: str


@dataclass(frozen=True, slots=True)
class Cue:
    nx: float
    ny: float
    depth: float
    pinch: float
    curl: float
    hardness: float
    preset: str
    #: A caption for the HUD, chosen from the last keyframe with one.
    caption: str


# Measured on the shipped stage: at pointer (0.5, 0.5) the pinch point sits at
# y = 0.39, and depth maps to world z as 0.00 -> -0.06, 0.13 -> 0.00,
# 0.25 -> +0.06, 0.50 -> +0.15, 1.00 -> +0.27.  The cloth hangs in the plane
# z = 0, which is why every grab below happens at depth 0.13.
_ON_SHEET = 0.13
_IN_FRONT = 0.55

#: Where the cloth's lower-middle is, in pointer space.
_SHEET_X, _SHEET_Y = 0.50, 0.62

CLOTH_ACT: tuple[tuple[Key, str], ...] = (
    # enter from the right, in front of the sheet, hand open
    (Key(0.0, 0.95, 0.55, _IN_FRONT, 0.0, 0.1, 0.30, "cloth"), "a sheet of cotton, hardness 30%"),
    (Key(1.6, _SHEET_X + 0.04, _SHEET_Y, _IN_FRONT, 0.0, 0.1, 0.30, "cloth"), ""),
    # move onto the sheet and close the pinch
    (Key(2.4, _SHEET_X, _SHEET_Y, _ON_SHEET, 0.0, 0.1, 0.30, "cloth"), ""),
    (Key(2.9, _SHEET_X, _SHEET_Y, _ON_SHEET, 1.0, 0.1, 0.30, "cloth"), "pinch"),
    # lift it, and bring it a little towards the viewer
    (Key(4.6, _SHEET_X - 0.06, 0.30, 0.30, 1.0, 0.1, 0.30, "cloth"), "lift"),
    # the showpiece: hold still, and walk the dial end to end
    (Key(5.2, _SHEET_X - 0.06, 0.30, 0.30, 1.0, 0.1, 0.30, "cloth"), ""),
    (Key(6.2, _SHEET_X - 0.06, 0.30, 0.30, 1.0, 0.1, 0.02, "cloth"), "same grip -- softer"),
    (Key(8.0, _SHEET_X - 0.06, 0.30, 0.30, 1.0, 0.1, 0.02, "cloth"), ""),
    (Key(11.5, _SHEET_X - 0.06, 0.30, 0.30, 1.0, 0.1, 0.97, "cloth"), "same grip -- harder"),
    (Key(13.0, _SHEET_X - 0.06, 0.30, 0.30, 1.0, 0.1, 0.97, "cloth"), ""),
    # swing it, so the stiffness shows in how it moves
    (Key(14.2, _SHEET_X + 0.12, 0.34, 0.30, 1.0, 0.1, 0.97, "cloth"), "stiff: it swings as a plate"),
    (Key(15.4, _SHEET_X - 0.14, 0.34, 0.30, 1.0, 0.1, 0.97, "cloth"), ""),
    (Key(17.0, _SHEET_X - 0.02, 0.32, 0.30, 1.0, 0.1, 0.35, "cloth"), "back to cotton"),
    (Key(18.0, _SHEET_X + 0.10, 0.34, 0.30, 1.0, 0.1, 0.35, "cloth"), ""),
    (Key(19.0, _SHEET_X - 0.10, 0.34, 0.30, 1.0, 0.1, 0.35, "cloth"), "soft: it swings as cloth"),
    # let go, and leave
    (Key(19.8, _SHEET_X - 0.04, 0.34, 0.30, 0.0, 0.1, 0.35, "cloth"), "release"),
    (Key(21.5, 0.96, 0.40, _IN_FRONT, 0.0, 0.1, 0.35, "cloth"), ""),
    (Key(23.0, 0.96, 0.40, _IN_FRONT, 0.0, 0.1, 0.35, "cloth"), ""),
)

# The soft body rests on the floor centred at (0, 0.13, 0) with its top at
# y = 0.26.  The pinch point sits about 5 cm above the wrist pointer, so a
# pointer at ny = 0.78, depth 0.13 puts the pinch at (0.02, 0.25, 0.00):
# on the cap.  The act grabs *first*, while the ball is still where it was
# built -- any press beforehand nudges it a few centimetres, and a blind
# choreography then closes on air.
_BALL_X, _BALL_GRAB_Y = 0.50, 0.78
_OVER_BALL = _ON_SHEET
#: Held height: high enough to clear the floor by a hand's width, low enough
#: that a rigid drop lands within a bounce of where it started.
_HOLD_Y = 0.42

SOFT_ACT: tuple[tuple[Key, str], ...] = (
    (Key(0.0, 0.92, 0.25, _IN_FRONT, 0.0, 0.2, 0.40, "soft"), "a soft body, hardness 40%"),
    (Key(1.4, _BALL_X, 0.40, _OVER_BALL, 0.0, 0.2, 0.40, "soft"), ""),
    # down onto the cap, close the pinch, lift
    (Key(2.4, _BALL_X, _BALL_GRAB_Y, _OVER_BALL, 0.0, 0.2, 0.40, "soft"), ""),
    (Key(2.9, _BALL_X, _BALL_GRAB_Y, _OVER_BALL, 1.0, 0.2, 0.40, "soft"), "grab"),
    (Key(4.6, _BALL_X + 0.02, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.40, "soft"), "lift"),
    # the showpiece again: held still, jelly to rigid and back
    (Key(5.4, _BALL_X + 0.02, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.40, "soft"), ""),
    (Key(7.4, _BALL_X + 0.02, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.03, "soft"), "held: to jelly"),
    (Key(9.0, _BALL_X + 0.02, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.03, "soft"), ""),
    (Key(12.0, _BALL_X + 0.02, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.96, "soft"), "held: to rigid"),
    (Key(13.2, _BALL_X + 0.02, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.96, "soft"), ""),
    # swing it, then drop it rigid: it bounces
    (Key(14.2, _BALL_X + 0.14, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.96, "soft"), "swing"),
    (Key(15.4, _BALL_X - 0.10, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.96, "soft"), ""),
    (Key(16.2, _BALL_X, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.96, "soft"), ""),
    # a second of stillness before letting go, so the pendulum has stopped
    # and the ball drops straight rather than being thrown across the stage
    (Key(17.8, _BALL_X, _HOLD_Y, _OVER_BALL + 0.08, 1.0, 0.2, 0.96, "soft"), ""),
    (Key(18.3, _BALL_X, _HOLD_Y, _OVER_BALL + 0.08, 0.0, 0.2, 0.96, "soft"), "drop it rigid: it bounces"),
    (Key(20.2, _BALL_X, _HOLD_Y, _OVER_BALL + 0.08, 0.0, 0.2, 0.96, "soft"), ""),
    # and once it has settled, melt it where it lies
    (Key(22.7, _BALL_X, _HOLD_Y, _OVER_BALL + 0.08, 0.0, 0.2, 0.02, "soft"), "melt it where it lies"),
    (Key(25.2, 0.94, 0.28, _IN_FRONT, 0.0, 0.2, 0.02, "soft"), ""),
    (Key(26.7, 0.94, 0.28, _IN_FRONT, 0.0, 0.2, 0.02, "soft"), ""),
)


def _smoothstep(t: float) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


class Choreography:
    """Evaluate a keyframed demonstration at a time.

    Acts are concatenated with a gap of ``pause`` seconds between them, during
    which the last cue of the previous act holds -- the hand is off stage by
    then, so the matter gets a moment to settle in shot before the preset
    changes underneath it.
    """

    def __init__(self, acts: tuple[tuple[tuple[Key, str], ...], ...] = (CLOTH_ACT, SOFT_ACT),
                 pause: float = 1.0) -> None:
        keys: list[Key] = []
        captions: list[str] = []
        offset = 0.0
        for act in acts:
            for key, caption in act:
                keys.append(Key(key.t + offset, key.nx, key.ny, key.depth,
                                key.pinch, key.curl, key.hardness, key.preset))
                captions.append(caption)
            offset = keys[-1].t + pause
        if len(keys) < 2:
            raise ValueError("a choreography needs at least two keyframes")
        for a, b in zip(keys, keys[1:]):
            if b.t < a.t:
                raise ValueError(f"keyframes run backwards at t={b.t}")
        self._keys = keys
        self._captions = captions
        self._times = [k.t for k in keys]

    @property
    def duration(self) -> float:
        return self._times[-1]

    def cue(self, t: float) -> Cue:
        keys = self._keys
        if t <= keys[0].t:
            k = keys[0]
            return Cue(k.nx, k.ny, k.depth, k.pinch, k.curl, k.hardness,
                       k.preset, self._caption_at(0))
        if t >= keys[-1].t:
            k = keys[-1]
            return Cue(k.nx, k.ny, k.depth, k.pinch, k.curl, k.hardness,
                       k.preset, self._caption_at(len(keys) - 1))
        i = bisect.bisect_right(self._times, t) - 1
        a, b = keys[i], keys[i + 1]
        span = b.t - a.t
        u = _smoothstep((t - a.t) / span) if span > 1e-9 else 1.0

        def mix(p: float, q: float) -> float:
            return p + (q - p) * u

        # The preset is whatever the keyframe we have *reached* says, not the
        # one we are heading for: it changes on crossing, never in between.
        return Cue(mix(a.nx, b.nx), mix(a.ny, b.ny), mix(a.depth, b.depth),
                   mix(a.pinch, b.pinch), mix(a.curl, b.curl),
                   mix(a.hardness, b.hardness), a.preset, self._caption_at(i))

    def _caption_at(self, i: int) -> str:
        while i >= 0:
            if self._captions[i]:
                return self._captions[i]
            i -= 1
        return ""
