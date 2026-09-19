"""Image space to world space: the map the whole illusion rests on.

If this is wrong, nothing downstream can be right -- the hand will be in the
wrong place, the wrong size, or pointing backwards, and no amount of solver
tuning fixes that.  Every case here is driven through the same
:class:`SyntheticSource` the application uses, so these are end-to-end
checks of the real path, not of a mock.
"""

from __future__ import annotations

import dataclasses
import math
import sys

import numpy as np
from _harness import case, note, run

from fctx.config import TrackingConfig
from fctx.core.types import HAND_BONES, NUM_LANDMARKS, Handedness, HandFrame, L
from fctx.hands.projection import (
    estimate_depth,
    hand_span_image,
    hand_span_metric,
    image_to_stage,
    project_hand,
    stage_to_image,
    world_chirality,
    world_to_image_preview,
)
from fctx.hands.synthetic import SyntheticSource, build_metric_hand

CFG = TrackingConfig()
BONES = np.asarray(HAND_BONES, dtype=np.intp)


def _hand(
    cfg: TrackingConfig = CFG,
    nx: float = 0.5,
    ny: float = 0.5,
    span: float | None = None,
    depth: float = 0.5,
    pinch: float = 0.0,
    curl: float = 0.0,
) -> HandFrame:
    src = SyntheticSource(cfg, auto=False, realtime=False)
    src.start()
    src.set_pointer(nx, ny)
    if span is None:
        src.set_depth(depth)
    else:
        src.set_span(span)
    src.set_pinch(pinch)
    src.set_curl(curl)
    frame = src.poll()
    assert frame is not None and frame.hands
    return frame.hands[0]


def _wrist(**kwargs: object) -> np.ndarray:
    return project_hand(_hand(**kwargs), kwargs.get("cfg", CFG))[int(L.WRIST)]  # type: ignore[arg-type]


@case
def test_monotonic_in_x() -> None:
    xs = [float(_wrist(nx=t)[0]) for t in np.linspace(0.0, 1.0, 21)]
    assert all(b > a for a, b in zip(xs, xs[1:])), xs
    assert xs[0] < -0.9 * CFG.stage_half_width < 0.0 < 0.9 * CFG.stage_half_width < xs[-1]


@case
def test_monotonic_in_y() -> None:
    ys = [float(_wrist(ny=t)[1]) for t in np.linspace(0.0, 1.0, 21)]
    assert all(b < a for a, b in zip(ys, ys[1:])), (
        "stage y=0 is the top of the volume, so world y must fall")
    assert ys[-1] < CFG.stage_center_y < ys[0]


@case
def test_monotonic_in_depth() -> None:
    zs = [float(_wrist(depth=t)[2]) for t in np.linspace(0.0, 1.0, 21)]
    assert all(b > a for a, b in zip(zs, zs[1:])), zs
    lo, hi = CFG.z_range
    assert lo < zs[0] and zs[-1] < hi, "the usable depth sweep must not clamp"


@case
def test_bigger_apparent_hand_is_nearer() -> None:
    near = estimate_depth(_hand(span=0.40).image, CFG)
    far = estimate_depth(_hand(span=0.15).image, CFG)
    assert near > far, f"span 0.40 gave z={near}, span 0.15 gave z={far}"


@case
def test_mirror_flips_x_and_nothing_else() -> None:
    mirrored = dataclasses.replace(CFG, mirror=True)
    plain = dataclasses.replace(CFG, mirror=False)
    a = project_hand(_hand(cfg=mirrored, nx=0.8), mirrored)
    b = project_hand(_hand(cfg=plain, nx=0.8), plain)
    # set_pointer is in stage coordinates, so the wrist lands in the same
    # world place either way; what mirroring changes is the hand's chirality.
    assert abs(float(a[int(L.WRIST), 0]) - float(b[int(L.WRIST), 0])) < 1e-5
    assert np.allclose(a[int(L.WRIST)], b[int(L.WRIST)], atol=1e-5)
    thumb_a = float(a[int(L.THUMB_TIP), 0] - a[int(L.WRIST), 0])
    thumb_b = float(b[int(L.THUMB_TIP), 0] - b[int(L.WRIST), 0])
    assert thumb_a * thumb_b < 0.0, (
        f"the thumb must swap sides when the stage mirrors: {thumb_a} / {thumb_b}")
    assert abs(thumb_a + thumb_b) < 1e-5


@case
def test_mirror_flips_the_image_mapping() -> None:
    mirrored = dataclasses.replace(CFG, mirror=True)
    plain = dataclasses.replace(CFG, mirror=False)
    assert image_to_stage(0.9, 0.5, plain)[0] > image_to_stage(0.1, 0.5, plain)[0]
    assert image_to_stage(0.9, 0.5, mirrored)[0] < image_to_stage(0.1, 0.5, mirrored)[0]


@case
def test_reference_span_lands_at_the_stage_centre() -> None:
    z = estimate_depth(_hand(span=CFG.reference_hand_span).image, CFG)
    note(f"reference span {CFG.reference_hand_span} -> z = {z:.4f} m "
         f"(stage centre {CFG.stage_center_z})")
    assert abs(z - CFG.stage_center_z) < 0.01, (
        f"a hand at the reference span landed at z={z:.4f}, not "
        f"{CFG.stage_center_z}")


@case
def test_depth_clamps_to_z_range() -> None:
    lo, hi = CFG.z_range
    tiny = _hand(span=0.01)
    assert estimate_depth(tiny.image, CFG) == lo

    leaning = _hand()
    forward = leaning.image.copy()
    forward[:, 2] -= 4.0
    assert estimate_depth(forward, CFG) == hi
    backward = leaning.image.copy()
    backward[:, 2] += 4.0
    assert estimate_depth(backward, CFG) == lo

    for image in (tiny.image, forward, backward):
        frame = HandFrame(image=np.ascontiguousarray(image, np.float32),
                          world=leaning.world, handedness=Handedness.RIGHT)
        joints = project_hand(frame, CFG)
        wrist_z = float(joints[int(L.WRIST), 2])
        assert lo - 1e-6 <= wrist_z <= hi + 1e-6, wrist_z
        # The rest of the hand hangs off the clamped wrist, so it may sit a
        # hand's thickness outside the range -- but no further.
        assert float(joints[:, 2].min()) > lo - 0.12
        assert float(joints[:, 2].max()) < hi + 0.12


@case
def test_output_is_always_finite() -> None:
    rng = np.random.default_rng(4)
    for _ in range(200):
        hand = _hand(nx=float(rng.random()), ny=float(rng.random()),
                     depth=float(rng.random()), pinch=float(rng.random()),
                     curl=float(rng.random()))
        joints = project_hand(hand, CFG)
        assert joints.shape == (NUM_LANDMARKS, 3)
        assert joints.dtype == np.float32
        assert np.isfinite(joints).all()


@case
def test_degenerate_landmarks_do_not_explode() -> None:
    collapsed = HandFrame(
        image=np.zeros((NUM_LANDMARKS, 3), np.float32),
        world=np.zeros((NUM_LANDMARKS, 3), np.float32),
        handedness=Handedness.RIGHT)
    joints = project_hand(collapsed, CFG)
    assert np.isfinite(joints).all()
    assert hand_span_image(collapsed.image, CFG) == 0.0


@case
def test_non_finite_input_is_rejected() -> None:
    hand = _hand()
    for field in ("image", "world"):
        bad = np.array(getattr(hand, field), copy=True)
        bad[3, 1] = np.nan
        broken = HandFrame(
            image=bad if field == "image" else hand.image,
            world=bad if field == "world" else hand.world,
            handedness=Handedness.RIGHT)
        try:
            project_hand(broken, CFG)
        except ValueError:
            continue
        raise AssertionError(f"non-finite {field} landmarks were accepted")


@case
def test_bone_lengths_survive_projection() -> None:
    """Projection may move and turn the hand.  It may not resize it."""
    for depth in (0.0, 0.25, 0.5, 0.75, 1.0):
        for pinch, curl in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.6, 0.6)):
            hand = _hand(nx=0.3, ny=0.7, depth=depth, pinch=pinch, curl=curl)
            metric = np.asarray(hand.world, dtype=np.float64)
            world = project_hand(hand, CFG).astype(np.float64)
            want = np.linalg.norm(metric[BONES[:, 0]] - metric[BONES[:, 1]], axis=1)
            got = np.linalg.norm(world[BONES[:, 0]] - world[BONES[:, 1]], axis=1)
            err = float(np.max(np.abs(got - want)))
            assert err < 1e-6, f"bone length changed by {err * 1e3:.4f} mm"
            assert 0.05 < float(np.sum(want)) < 1.0


@case
def test_pairwise_distances_survive_projection() -> None:
    hand = _hand(nx=0.2, ny=0.2, depth=0.8, pinch=0.4, curl=0.3)
    metric = np.asarray(hand.world, dtype=np.float64)
    world = project_hand(hand, CFG).astype(np.float64)
    dm = np.linalg.norm(metric[:, None, :] - metric[None, :, :], axis=2)
    dw = np.linalg.norm(world[:, None, :] - world[None, :, :], axis=2)
    assert float(np.max(np.abs(dm - dw))) < 1e-6


@case
def test_projected_hand_points_the_right_way() -> None:
    """The metric hand must be turned to agree with the image, not ignored."""
    hand = _hand()
    world = project_hand(hand, CFG).astype(np.float64)
    wrist = world[int(L.WRIST)]
    assert float(world[int(L.MIDDLE_TIP), 1]) > float(wrist[1]) + 0.10, (
        "an upright hand must have its fingers above its wrist")

    # Rotate the image landmarks a quarter turn about the wrist and the world
    # hand must follow; a projection that only placed the wrist would not.
    image = np.array(hand.image, dtype=np.float64, copy=True)
    aspect = CFG.stage_half_width / CFG.stage_half_height
    rel = image[:, :2] - image[int(L.WRIST), :2]
    rel[:, 1] /= aspect
    ang = math.pi / 2.0
    rot = np.array([[math.cos(ang), -math.sin(ang)],
                    [math.sin(ang), math.cos(ang)]])
    rel = rel @ rot.T
    rel[:, 1] *= aspect
    image[:, :2] = image[int(L.WRIST), :2] + rel
    turned = project_hand(
        HandFrame(image=np.ascontiguousarray(image, np.float32),
                  world=hand.world, handedness=Handedness.RIGHT), CFG)
    finger = turned[int(L.MIDDLE_TIP)] - turned[int(L.WRIST)]
    assert abs(float(finger[0])) > 3.0 * abs(float(finger[1])), (
        f"the rotated hand still points up: {finger}")


@case
def test_span_measures_agree() -> None:
    """The image span and the metric span must be the same measure."""
    hand = _hand(span=0.30)
    assert abs(hand_span_image(hand.image, CFG) - 0.30) < 1e-4
    metric = hand_span_metric(hand.world)
    assert 0.12 < metric < 0.25, f"an adult hand spans {metric:.3f} m?"


@case
def test_span_is_pose_invariant() -> None:
    open_span = hand_span_image(_hand(span=0.26, curl=0.0).image, CFG)
    fist_span = hand_span_image(_hand(span=0.26, curl=1.0).image, CFG)
    pinch_span = hand_span_image(_hand(span=0.26, pinch=1.0).image, CFG)
    assert abs(open_span - fist_span) < 1e-4, "closing the hand changed its depth"
    assert abs(open_span - pinch_span) < 1e-4


@case
def test_stage_and_image_are_inverses() -> None:
    rng = np.random.default_rng(11)
    for cfg in (CFG, dataclasses.replace(CFG, mirror=False)):
        for _ in range(100):
            u, v = float(rng.random()), float(rng.random())
            x, y = image_to_stage(u, v, cfg)
            u2, v2 = stage_to_image(x, y, cfg)
            assert abs(u - u2) < 1e-9 and abs(v - v2) < 1e-9


@case
def test_preview_round_trips_in_pixels() -> None:
    hand = _hand(nx=0.32, ny=0.71)
    joints = project_hand(hand, CFG)
    px = world_to_image_preview(joints, CFG, 1280, 720)
    assert px.shape == (NUM_LANDMARKS, 2) and px.dtype == np.float32
    expected = np.column_stack((hand.image[:, 0] * 1280.0, hand.image[:, 1] * 720.0))
    err = float(np.max(np.abs(px - expected)))
    note(f"landmark overlay error at 1280x720: {err:.2f} px")
    # Not exact, and cannot be: the depth estimate mixes in MediaPipe's
    # relative z, which is a property of the frame rather than of the world
    # joints, so inverting it recovers the apparent scale to within the
    # weight of that term.  A couple of pixels is invisible in a 1280-wide
    # overlay; anything larger means the wrist or the scale is wrong.
    assert err < 3.0, f"the overlay is {err:.1f} px off the landmarks"
    assert float(np.max(np.abs(px[int(L.WRIST)] - expected[int(L.WRIST)]))) < 0.01


@case
def test_chirality_tracks_the_geometry() -> None:
    mirrored = dataclasses.replace(CFG, mirror=True)
    plain = dataclasses.replace(CFG, mirror=False)
    assert world_chirality(Handedness.RIGHT, plain) == -1.0
    assert world_chirality(Handedness.LEFT, plain) == 1.0
    assert world_chirality(Handedness.RIGHT, mirrored) == 1.0
    assert world_chirality(Handedness.LEFT, mirrored) == -1.0

    from fctx.hands.gestures import palm_normal

    for cfg in (plain, mirrored):
        world = project_hand(_hand(cfg=cfg), cfg)
        n = palm_normal(world, world_chirality(Handedness.RIGHT, cfg))
        assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-5
        assert float(n[2]) > 0.8, (
            f"the canonical hand faces the viewer; got a palm normal of {n}")


@case
def test_metric_hand_is_anatomically_plausible() -> None:
    hand = build_metric_hand(0.0, 0.0)
    assert hand.shape == (NUM_LANDMARKS, 3) and hand.dtype == np.float32
    lengths = np.linalg.norm(hand[BONES[:, 0]] - hand[BONES[:, 1]], axis=1)
    assert float(lengths.min()) > 0.005, "a bone shorter than 5 mm"
    assert float(lengths.max()) < 0.12, "a bone longer than 120 mm"
    assert float(np.abs(np.mean(hand, axis=0)).max()) < 1e-6
    # Bone lengths are rigid: curling and pinching move joints, never resize.
    for pinch, curl in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)):
        posed = build_metric_hand(pinch, curl).astype(np.float64)
        other = np.linalg.norm(posed[BONES[:, 0]] - posed[BONES[:, 1]], axis=1)
        assert float(np.max(np.abs(other - lengths))) < 1e-5, (pinch, curl)


if __name__ == "__main__":
    sys.exit(run(__file__))
