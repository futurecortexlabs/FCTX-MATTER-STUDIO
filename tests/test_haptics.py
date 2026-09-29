"""Pseudo-haptics: the drawn hand is held back by what it presses, per hardness.

Driven with hand-built poses and contact reports the way the application
drives it: contact describes the previous frame's displayed hand.
"""

from __future__ import annotations

import numpy as np
from _harness import case, note, require, run

from fctx.config import HapticsConfig
from fctx.core.types import HandPose
from fctx.haptics import HandContact, PseudoHaptics, palm_centre

SURFACE_Y = 0.30   # top of a flat sample


def hand(y: float, x: float = 0.0, track_id: int = 0, pinching: bool = False) -> HandPose:
    joints = np.zeros((21, 3), np.float32)
    joints[:, 0] = x + np.linspace(-0.03, 0.03, 21)
    joints[:, 1] = y
    pinch = np.array([x, y, 0.0], np.float32)
    return HandPose(joints=joints, velocities=np.zeros((21, 3), np.float32),
                    track_id=track_id, pinching=pinching, pinch_point=pinch)


def contact_for(shown: HandPose, bodies: int = 2, body: int = 0) -> dict[int, HandContact]:
    """A flat surface at SURFACE_Y under ``body``: touching once the drawn palm reaches it."""
    y = float(palm_centre(shown)[1])
    count = np.zeros(bodies)
    normal = np.zeros((bodies, 3))
    pos = np.zeros((bodies, 3))
    if y <= SURFACE_Y + 0.004:
        count[body] = 40
        normal[body] = (0.0, 40.0, 0.0)
        pos[body] = (0.0, SURFACE_Y * 40, 0.0)
    return {shown.track_id: HandContact(count, normal, pos)}


def press(ph: PseudoHaptics, depth: float, hardness: list[float], body: int = 0,
          frames: int = 60, dt: float = 1 / 60) -> tuple[float, float]:
    """Lower a hand from above the surface to ``depth`` below it; return (real, drawn) y."""
    contacts: dict[int, HandContact] = {}
    shown = None
    for k in range(frames):
        y = SURFACE_Y + 0.05 - (0.05 + depth) * min(1.0, k / (frames * 0.6))
        shown = ph.update([hand(y)], contacts, hardness, set(), dt)[0]
        contacts = contact_for(shown, body=body)
    return y, float(palm_centre(shown)[1])


@case
def the_gain_falls_geometrically_with_hardness() -> None:
    ph = PseudoHaptics(HapticsConfig(soft_gain=0.9, hard_gain=0.1))
    g = [ph.gain(h) for h in (0.0, 0.25, 0.5, 0.75, 1.0)]
    require(abs(g[0] - 0.9) < 1e-9 and abs(g[-1] - 0.1) < 1e-9, f"endpoints {g}")
    require(all(a > b for a, b in zip(g, g[1:])), f"not monotone: {g}")
    require(abs(g[2] - (0.9 * 0.1) ** 0.5) < 1e-9, "not geometric")


@case
def a_hard_surface_holds_the_drawn_hand_back_and_a_soft_one_lets_it_sink() -> None:
    depth = 0.04
    shown_depth = {}
    for h in (0.0, 0.5, 1.0):
        ph = PseudoHaptics(HapticsConfig())
        real_y, drawn_y = press(ph, depth, [h, h])
        shown_depth[h] = SURFACE_Y - drawn_y
        expect = depth * ph.gain(h)
        require(abs(shown_depth[h] - expect) < 0.004,
                f"hardness {h}: drawn depth {shown_depth[h]:.4f}, expected {expect:.4f}")
    require(shown_depth[0.0] > shown_depth[0.5] > shown_depth[1.0],
            f"drawn depth not ordered by hardness: {shown_depth}")
    note("4 cm real press shows as " + ", ".join(
        f"{v * 1000:.1f} mm at h={k}" for k, v in shown_depth.items()))


@case
def only_the_pressed_body_sets_the_resistance() -> None:
    ph = PseudoHaptics(HapticsConfig())
    _, soft_y = press(ph, 0.04, [0.0, 1.0], body=0)
    ph = PseudoHaptics(HapticsConfig())
    _, hard_y = press(ph, 0.04, [0.0, 1.0], body=1)
    require(soft_y < hard_y - 0.015,
            f"pressing the soft sample ({soft_y:.3f}) vs the hard one ({hard_y:.3f})")


@case
def sideways_motion_passes_through_while_pressing() -> None:
    ph = PseudoHaptics(HapticsConfig())
    press(ph, 0.03, [1.0, 1.0])
    contacts = contact_for(ph.update([hand(SURFACE_Y - 0.03)], {}, [1.0, 1.0], set(), 1 / 60)[0])
    for k in range(30):
        shown = ph.update([hand(SURFACE_Y - 0.03, x=0.002 * k)], contacts, [1.0, 1.0],
                          set(), 1 / 60)[0]
        contacts = contact_for(shown)
    dx = float(palm_centre(shown)[0]) - float(palm_centre(hand(0.0, x=0.002 * 29))[0])
    require(abs(dx) < 1e-4, f"the drawn hand lagged sideways by {dx * 1000:.2f} mm")


@case
def backing_out_releases_and_the_offset_relaxes_away() -> None:
    ph = PseudoHaptics(HapticsConfig())
    press(ph, 0.05, [1.0, 1.0])
    require(float(np.linalg.norm(ph.offset(0))) > 0.03, "no offset built up")
    contacts: dict[int, HandContact] = {}
    for _ in range(60):
        shown = ph.update([hand(SURFACE_Y + 0.08)], contacts, [1.0, 1.0], set(), 1 / 60)[0]
        contacts = contact_for(shown)
    require(float(np.linalg.norm(ph.offset(0))) < 1e-4, f"offset left: {ph.offset(0)}")
    require(ph.touching(0) == -1, "still marked as pressing")
    require(abs(float(palm_centre(shown)[1]) - (SURFACE_Y + 0.08)) < 1e-4,
            "the drawn hand did not return to the real one")


@case
def a_hand_that_holds_something_is_left_alone() -> None:
    ph = PseudoHaptics(HapticsConfig(hold_relax_time=0.1))
    press(ph, 0.05, [1.0, 1.0])
    for _ in range(60):
        shown = ph.update([hand(SURFACE_Y - 0.05)], contact_for(hand(SURFACE_Y)),
                          [1.0, 1.0], {0}, 1 / 60)[0]
    require(float(np.linalg.norm(ph.offset(0))) < 1e-4, "a holding hand kept its offset")
    require(abs(float(palm_centre(shown)[1]) - (SURFACE_Y - 0.05)) < 1e-4)


@case
def disabled_means_the_drawn_hand_is_the_real_one() -> None:
    ph = PseudoHaptics(HapticsConfig(enabled=False))
    real_y, drawn_y = press(ph, 0.05, [1.0, 1.0])
    require(abs(real_y - drawn_y) < 1e-6, "a disabled proxy moved the hand")


@case
def the_push_never_exceeds_the_limit_and_whole_hand_moves_rigidly() -> None:
    ph = PseudoHaptics(HapticsConfig(max_offset=0.02, hard_gain=0.01))
    press(ph, 0.10, [1.0, 1.0])
    off = ph.offset(0)
    require(float(np.linalg.norm(off)) <= 0.02 + 1e-6, f"offset {off} over the limit")
    real = hand(SURFACE_Y - 0.10)
    shown = ph.update([real], contact_for(hand(SURFACE_Y)), [1.0, 1.0], set(), 1 / 60)[0]
    d = np.asarray(shown.joints) - np.asarray(real.joints)
    require(np.allclose(d, d[0], atol=1e-6), "joints did not move together")
    require(np.allclose(np.asarray(shown.pinch_point) - np.asarray(real.pinch_point), d[0],
                        atol=1e-6), "the pinch point did not move with the hand")


@case
def hands_that_leave_are_forgotten() -> None:
    ph = PseudoHaptics(HapticsConfig())
    press(ph, 0.03, [1.0, 1.0])
    ph.update([], {}, [1.0, 1.0], set(), 1 / 60)
    require(ph.offset(0).tolist() == [0.0, 0.0, 0.0], "a vanished hand kept its state")


if __name__ == "__main__":
    raise SystemExit(run(__file__))
