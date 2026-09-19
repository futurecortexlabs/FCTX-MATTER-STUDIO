"""BodyData.validate is the guard rail every other subsystem trusts.

A wrong dtype or a stray non-contiguous array does not fail loudly in Warp; it
fails as garbage on the GPU three layers away. These checks make sure the
guard rail actually catches things rather than just existing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from _harness import case, note, require, run  # noqa: E402

from fctx.core.types import (  # noqa: E402
    BONE_RADII,
    FLAG_PINNED,
    FLAG_SURFACE,
    HAND_BONES,
    NUM_LANDMARKS,
    BodyData,
    Gesture,
    Handedness,
    HandFrame,
    HandPose,
    L,
    MatterKind,
    TrackerFrame,
)


def _quad() -> BodyData:
    """The smallest well-formed body: two triangles sharing an edge."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    return BodyData(
        kind=MatterKind.CLOTH,
        name="quad",
        positions=pos,
        inv_mass=np.ones(4, np.float32),
        flags=np.full(4, FLAG_SURFACE, np.uint32),
        dist_idx=np.array([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3]], np.int32),
        dist_rest=np.array([1, 1, 1, 1, np.sqrt(2.0)], np.float32),
        dist_kind=np.zeros(5, np.int32),
        dist_color=np.array([0, 0, 1, 1, 2], np.int32),
        bend_idx=np.array([[0, 3, 1, 2]], np.int32),
        bend_rest=np.zeros(1, np.float32),
        bend_color=np.zeros(1, np.int32),
        tet_idx=np.zeros((0, 4), np.int32),
        tet_dm_inv=np.zeros((0, 3, 3), np.float32),
        tet_rest_volume=np.zeros(0, np.float32),
        tet_color=np.zeros(0, np.int32),
        tri_idx=np.array([[0, 1, 3], [0, 3, 2]], np.int32),
        uv=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32),
        double_sided=True,
    )


def _expect_rejection(mutate, reason: str) -> None:
    body = _quad()
    mutate(body)
    try:
        body.validate()
    except ValueError:
        return
    raise AssertionError(f"validate() accepted a body with {reason}")


@case
def a_well_formed_body_passes() -> None:
    body = _quad()
    body.validate()
    require(body.num_particles == 4)
    require(body.num_dist == 5)
    require(body.num_bend == 1)
    require(body.num_tets == 0)
    require(body.num_tris == 2)


@case
def wrong_dtypes_are_rejected() -> None:
    _expect_rejection(
        lambda b: setattr(b, "positions", b.positions.astype(np.float64)),
        "float64 positions")
    _expect_rejection(
        lambda b: setattr(b, "dist_idx", b.dist_idx.astype(np.int64)),
        "int64 indices")
    _expect_rejection(
        lambda b: setattr(b, "flags", b.flags.astype(np.int32)),
        "int32 flags")
    _expect_rejection(
        lambda b: setattr(b, "uv", b.uv.astype(np.float16)),
        "float16 uv")


@case
def wrong_shapes_are_rejected() -> None:
    _expect_rejection(
        lambda b: setattr(b, "inv_mass", np.ones(3, np.float32)),
        "an inv_mass array shorter than the particle count")
    _expect_rejection(
        lambda b: setattr(b, "uv", np.zeros((4, 3), np.float32)),
        "three-component uv")
    _expect_rejection(
        lambda b: setattr(b, "bend_idx", np.zeros((1, 3), np.int32)),
        "a three-vertex bending constraint")


@case
def non_contiguous_arrays_are_rejected() -> None:
    # This is the failure that would otherwise reach the GPU silently: a view
    # with a stride copies the wrong bytes when Warp uploads it.
    def strided(b: BodyData) -> None:
        wide = np.zeros((4, 6), np.float32)
        b.positions = wide[:, :3]

    _expect_rejection(strided, "a strided view for positions")


@case
def out_of_range_indices_are_rejected() -> None:
    _expect_rejection(
        lambda b: setattr(b, "dist_idx",
                          np.array([[0, 99]] * 5, np.int32)),
        "a distance index past the end of the particle array")
    _expect_rejection(
        lambda b: setattr(b, "tri_idx", np.array([[0, 1, -1]], np.int32)),
        "a negative triangle index")


@case
def non_finite_positions_and_negative_mass_are_rejected() -> None:
    def nan_pos(b: BodyData) -> None:
        p = b.positions.copy()
        p[2, 1] = np.nan
        b.positions = p

    _expect_rejection(nan_pos, "a NaN position")
    _expect_rejection(
        lambda b: setattr(b, "inv_mass",
                          np.array([1, 1, -1, 1], np.float32)),
        "a negative inverse mass")


@case
def degenerate_tetrahedra_are_rejected() -> None:
    def bad_tet(b: BodyData) -> None:
        b.tet_idx = np.array([[0, 1, 2, 3]], np.int32)
        b.tet_dm_inv = np.zeros((1, 3, 3), np.float32)
        b.tet_rest_volume = np.array([0.0], np.float32)
        b.tet_color = np.zeros(1, np.int32)

    _expect_rejection(bad_tet, "a zero-volume tetrahedron")


@case
def the_hand_skeleton_is_consistent() -> None:
    require(len(HAND_BONES) == len(BONE_RADII),
            "every bone needs a collider radius")
    seen = set()
    for a, b in HAND_BONES:
        require(0 <= a < NUM_LANDMARKS and 0 <= b < NUM_LANDMARKS,
                f"bone ({a}, {b}) references a landmark that does not exist")
        key = (min(a, b), max(a, b))
        require(key not in seen, f"bone {key} is listed twice")
        seen.add(key)
    require(all(r > 0.0 for r in BONE_RADII), "a bone has a non-positive radius")
    # Every landmark except the wrist must hang off something, or part of the
    # hand would have no collider at all.
    covered = {i for bone in HAND_BONES for i in bone}
    missing = set(range(NUM_LANDMARKS)) - covered
    require(not missing, f"landmarks with no bone attached: {sorted(missing)}")
    note(f"{len(HAND_BONES)} capsules covering {len(covered)} landmarks, "
         f"radii {min(BONE_RADII) * 1000:.1f}-{max(BONE_RADII) * 1000:.1f} mm")


@case
def hand_pose_derives_capsules_that_match_the_skeleton() -> None:
    joints = np.random.default_rng(7).normal(size=(21, 3)).astype(np.float32)
    pose = HandPose(joints=joints, velocities=np.zeros_like(joints),
                    handedness=Handedness.RIGHT, gesture=Gesture.OPEN)
    a, b, r = pose.bone_segments()
    require(a.shape == b.shape == (len(HAND_BONES), 3), f"got {a.shape}")
    require(r.shape == (len(HAND_BONES),))
    require(a.dtype == np.float32 and b.dtype == np.float32)
    require(a.flags["C_CONTIGUOUS"] and b.flags["C_CONTIGUOUS"],
            "capsule arrays must be contiguous before they reach Warp")
    require(np.allclose(a[0], joints[L.WRIST]),
            "the first capsule should start at the wrist")
    va, vb = pose.bone_velocities()
    require(va.shape == vb.shape == (len(HAND_BONES), 3))


@case
def hand_frame_rejects_malformed_landmarks() -> None:
    good = np.zeros((21, 3), np.float32)
    HandFrame(image=good, world=good.copy())
    for bad in (np.zeros((20, 3), np.float32), np.zeros((21, 2), np.float32)):
        try:
            HandFrame(image=bad, world=good.copy())
        except ValueError:
            continue
        raise AssertionError(f"HandFrame accepted landmarks of shape {bad.shape}")


@case
def tracker_frame_reports_emptiness() -> None:
    require(not TrackerFrame().ok, "an empty frame should not report ok")
    z = np.zeros((21, 3), np.float32)
    require(TrackerFrame(hands=[HandFrame(image=z, world=z.copy())]).ok)


@case
def flags_are_distinct_bits() -> None:
    from fctx.core.types import FLAG_GRABBED, FLAG_SELF_COLLIDE

    bits = [FLAG_PINNED, FLAG_GRABBED, FLAG_SURFACE, FLAG_SELF_COLLIDE]
    require(len(set(bits)) == len(bits), "duplicate flag value")
    for bit in bits:
        require(bit & (bit - 1) == 0, f"{bit} is not a single bit")


@case
def body_data_is_mutable_enough_to_be_offset_by_the_solver() -> None:
    # SolverState shifts local indices into a global buffer, so the arrays must
    # not be read-only and dataclasses.replace must work on the type.
    body = _quad()
    shifted = dataclasses.replace(body, dist_idx=body.dist_idx + 100)
    require(shifted.dist_idx.dtype == np.int32,
            "adding an int shifted the dtype; the solver relies on int32")
    require(body.dist_idx.max() < 100, "the original must not be mutated")


if __name__ == "__main__":
    raise SystemExit(run(__file__))
