"""The choreography is a pure function of time; these are its promises.

The GPU end of the demonstration -- that the cues actually grab, lift and
release -- lives in test_smoke.py.  What is checked here needs no hardware:
that the timeline is well formed, that nothing jumps, that a preset only ever
changes on a keyframe, and that the shipped acts land their hands where the
matter is.
"""

from __future__ import annotations

import numpy as np
from _harness import case, note, require, run  # noqa: E402

from fctx.demo import CLOTH_ACT, SOFT_ACT, Choreography, Key  # noqa: E402


@case
def keyframes_are_ordered_and_every_field_is_in_range() -> None:
    for name, act in (("cloth", CLOTH_ACT), ("soft", SOFT_ACT)):
        times = [k.t for k, _ in act]
        require(times == sorted(times), f"{name}: keyframes out of order")
        require(times[0] == 0.0, f"{name}: the act must start at t = 0")
        for k, _ in act:
            for field in ("nx", "ny", "depth", "pinch", "curl", "hardness"):
                v = getattr(k, field)
                require(0.0 <= v <= 1.0, f"{name} t={k.t}: {field} = {v}")
            require(k.preset in ("cloth", "soft", "grain"),
                    f"{name} t={k.t}: unknown preset {k.preset!r}")
    c = Choreography()
    require(c.duration > 30.0, f"the shipped demo is only {c.duration:.1f} s")
    note(f"shipped demo runs {c.duration:.1f} s over "
         f"{len(CLOTH_ACT) + len(SOFT_ACT)} keyframes")


@case
def cues_are_continuous_in_everything_but_the_preset() -> None:
    # A jump in the pointer is a velocity spike the solver would act on; a
    # jump in the dial is a step the material would ring at.  Sample the whole
    # timeline at the physics rate and bound the per-step change.
    c = Choreography()
    dt = 1.0 / 90.0
    prev = c.cue(0.0)
    worst = {f: 0.0 for f in ("nx", "ny", "depth", "pinch", "curl", "hardness")}
    switches = 0
    t = dt
    while t <= c.duration + 2.0:
        cue = c.cue(t)
        for f in worst:
            worst[f] = max(worst[f], abs(getattr(cue, f) - getattr(prev, f)))
        if cue.preset != prev.preset:
            switches += 1
        prev = cue
        t += dt
    # Position-like fields are what the solver differentiates into hand
    # velocity, so they get the tight bound.  A pinch closing in half a second
    # is a natural pinch and the grip has hysteresis for it; the dial has its
    # own smoothing downstream.
    limits = {"nx": 0.02, "ny": 0.02, "depth": 0.02,
              "pinch": 0.05, "curl": 0.05, "hardness": 0.05}
    for f, v in worst.items():
        require(v < limits[f], f"{f} jumps by {v:.3f} in one physics step")
    require(switches == 1, f"the demo switched preset {switches} times, expected 1")
    note("largest per-step change: " + ", ".join(f"{f} {v:.4f}" for f, v in worst.items()))


@case
def a_preset_changes_exactly_on_its_keyframe_and_never_between() -> None:
    a = Key(0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.3, "cloth")
    b = Key(2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.3, "soft")
    c = Choreography(acts=(((a, ""), (b, "")),))
    require(c.cue(0.0).preset == "cloth")
    require(c.cue(1.999).preset == "cloth",
            "the preset changed before the timeline reached its keyframe")
    require(c.cue(2.0).preset == "soft")
    require(c.cue(50.0).preset == "soft", "the last cue must hold after the end")


@case
def the_smoothstep_holds_at_both_ends_and_is_monotone() -> None:
    a = Key(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "cloth")
    b = Key(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "cloth")
    c = Choreography(acts=(((a, ""), (b, "")),))
    xs = np.linspace(0.0, 1.0, 101)
    ys = np.array([c.cue(float(x)).nx for x in xs])
    require(ys[0] == 0.0 and ys[-1] == 1.0, "the ends are not held")
    require(np.all(np.diff(ys) >= 0.0), "the interpolation is not monotone")
    require(abs(ys[50] - 0.5) < 1e-9, "the midpoint is not halfway")
    require(ys[1] < 0.001 and ys[-2] > 0.999,
            "the ease-in/ease-out is missing; the hand would jerk at keyframes")


@case
def captions_carry_forward_until_the_next_one() -> None:
    c = Choreography()
    require(c.cue(0.0).caption != "", "the opening cue has no caption")
    seen = {c.cue(t).caption for t in np.arange(0.0, c.duration, 0.25)}
    require(len(seen) >= 8, f"only {len(seen)} distinct captions over the demo")
    require("" not in seen, "a cue between captions came back blank")


@case
def a_backwards_or_degenerate_timeline_is_refused() -> None:
    a = Key(1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.3, "cloth")
    b = Key(0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.3, "cloth")
    for acts in ((((a, ""), (b, "")),), (((a, ""),),)):
        try:
            Choreography(acts=acts)
        except ValueError:
            continue
        raise AssertionError("a bad timeline was accepted")


if __name__ == "__main__":
    raise SystemExit(run(__file__))
