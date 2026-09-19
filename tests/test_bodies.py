"""Geometry: the claims the solver and the renderer are allowed to assume.

A body that passes here can be uploaded to the GPU without further checking:
shapes and dtypes are exact, tetrahedra are positively oriented, soft surfaces
are closed and outward-facing, and the masses are the ones the material asked
for.
"""

from __future__ import annotations

import sys

import numpy as np
from _harness import case, note, require, run

from fctx.bodies import build_cloth, build_granular, build_scene, build_soft_body
from fctx.bodies.cloth import dihedral_angle
from fctx.bodies.softbody import (
    _make_well_composed,
    sdf_box,
    sdf_sphere,
    sdf_torus,
)
from fctx.config import PRESETS, SceneConfig, SolverConfig, preset
from fctx.core.material import (
    CLOTH_MATERIAL,
    GRAIN_MATERIAL,
    SOFT_MATERIAL,
    areal_particle_mass,
)
from fctx.core.types import (
    FLAG_PINNED,
    FLAG_SELF_COLLIDE,
    FLAG_SURFACE,
    ConstraintKind,
    MatterKind,
)

# Resolution 7, 8 and 17 are here because an unrepaired torus is non-manifold
# at exactly those lattices: the hole pinches into a diagonal-only voxel pair.
_SOFT_CASES = [(shape, res) for shape in ("sphere", "box", "torus")
               for res in (4, 6, 7, 8, 9, 13, 14, 17)]


def _closed_manifold(tri: np.ndarray) -> None:
    directed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    undirected, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                   return_counts=True)
    if not np.all(counts == 2):
        bad = int((counts != 2).sum())
        raise AssertionError(
            f"{bad} of {undirected.shape[0]} edges are not shared by exactly "
            f"two triangles; the surface is not a closed manifold")
    _, dup = np.unique(directed, axis=0, return_counts=True)
    require(np.all(dup == 1),
            "a directed edge appears twice: the surface is not consistently wound")


def _min_separation(pos: np.ndarray, cell: float) -> float:
    """Distance between the closest pair, via a uniform grid.

    Brute force is 5.8e8 pairs at 24000 grains; bucketing by a cell wider than
    the answer we are looking for keeps every candidate pair inside the 27
    neighbouring cells.
    """
    keys = np.floor(pos / cell).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, keys.tolist())):
        buckets.setdefault(key, []).append(index)

    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    best = np.inf
    for (kx, ky, kz), members in buckets.items():
        near: list[int] = []
        for dx, dy, dz in offsets:
            near.extend(buckets.get((kx + dx, ky + dy, kz + dz), ()))
        mine = np.asarray(members)
        theirs = np.asarray(near)
        d = np.linalg.norm(pos[mine][:, None, :] - pos[theirs][None, :, :], axis=2)
        d[mine[:, None] == theirs[None, :]] = np.inf
        best = min(best, float(d.min()))
    return best


def _enclosed_volume(tri: np.ndarray, pos: np.ndarray) -> float:
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


@case
def the_subsystem_stays_pure_numpy() -> None:
    # The contract is that a scene can be built and inspected on a machine
    # with no GPU and no camera, which is also what lets these tests run at
    # all.  Importing warp here would only fail on the machine that has none.
    for name in PRESETS:
        build_scene(preset(name).scene)
    for forbidden in ("warp", "moderngl", "glfw", "mediapipe", "cv2", "OpenGL"):
        require(forbidden not in sys.modules,
                f"fctx.bodies pulled in {forbidden}")


@case
def every_preset_validates() -> None:
    for name in PRESETS:
        scene = preset(name).scene
        bodies = build_scene(scene)
        require(len(bodies) == 1, f"preset {name} built {len(bodies)} bodies")
        body = bodies[0]
        body.validate()
        require(body.kind is MatterKind(scene.kind),
                f"preset {name} built the wrong kind of matter")
        note(f"{name:8s} P={body.num_particles:6d} D={body.num_dist:6d} "
             f"B={body.num_bend:6d} T={body.num_tets:6d} F={body.num_tris:6d} "
             f"r={body.particle_radius * 1000:.2f} mm")


@case
def arrays_are_contiguous_and_typed() -> None:
    # validate() already checks this, but only for the arrays it knows about;
    # a non-contiguous array reaching wp.array() copies silently on some paths
    # and raises on others, so it is worth failing here instead.
    for name in PRESETS:
        body = build_scene(preset(name).scene)[0]
        for field in ("positions", "inv_mass", "flags", "dist_idx", "dist_rest",
                      "dist_kind", "dist_color", "bend_idx", "bend_rest",
                      "bend_color", "tet_idx", "tet_dm_inv", "tet_rest_volume",
                      "tet_color", "tri_idx", "uv"):
            arr = getattr(body, field)
            require(arr.flags["C_CONTIGUOUS"], f"{name}.{field} is not contiguous")


@case
def indices_stay_in_range() -> None:
    for name in PRESETS:
        body = build_scene(preset(name).scene)[0]
        for field in ("dist_idx", "bend_idx", "tet_idx", "tri_idx"):
            arr = getattr(body, field)
            if arr.size == 0:
                continue
            require(arr.min() >= 0 and arr.max() < body.num_particles,
                    f"{name}.{field} references a particle that does not exist")
        if body.num_bend:
            rows = np.sort(body.bend_idx, axis=1)
            require(np.all(np.diff(rows, axis=1) > 0),
                    f"{name}: a bend constraint reuses a particle")
        if body.num_tets:
            rows = np.sort(body.tet_idx, axis=1)
            require(np.all(np.diff(rows, axis=1) > 0),
                    f"{name}: a tetrahedron reuses a vertex")


@case
def masses_are_finite_and_positive() -> None:
    for name in PRESETS:
        body = build_scene(preset(name).scene)[0]
        require(np.isfinite(body.inv_mass).all(), f"{name}: non-finite inverse mass")
        pinned = (body.flags & FLAG_PINNED) != 0
        require(np.all(body.inv_mass[pinned] == 0.0),
                f"{name}: a pinned particle has a finite mass")
        free = body.inv_mass[~pinned]
        require(np.all(free > 0.0),
                f"{name}: {int((free <= 0).sum())} unpinned particles weigh nothing")
        note(f"{name:8s} mass {1.0 / free.max() * 1000:.4f}..."
             f"{1.0 / free.min() * 1000:.4f} g per particle")


# --------------------------------------------------------------------------
# cloth
# --------------------------------------------------------------------------


@case
def cloth_rest_lengths_match_the_grid() -> None:
    for resolution, size in ((8, 0.6), (33, 0.6), (72, 0.6), (88, 0.72)):
        scene = SceneConfig(cloth_resolution=resolution, cloth_size=size)
        body = build_cloth(scene, CLOTH_MATERIAL)
        spacing = size / (resolution - 1)
        rest = body.dist_rest.astype(np.float64)
        stretch = rest[body.dist_kind == int(ConstraintKind.STRETCH)]
        shear = rest[body.dist_kind == int(ConstraintKind.SHEAR)]
        require(np.abs(stretch - spacing).max() < 1e-6,
                f"cloth-{resolution} stretch rest length is off the grid spacing "
                f"by {np.abs(stretch - spacing).max():.3g} m")
        require(np.abs(shear - spacing * np.sqrt(2.0)).max() < 1e-6,
                f"cloth-{resolution} shear rest length is not the cell diagonal")
        note(f"cloth-{resolution}: spacing {spacing * 1000:.4f} mm, "
             f"max error {np.abs(stretch - spacing).max():.2e} m")


@case
def cloth_bend_rest_angle_is_flat() -> None:
    body = build_cloth(SceneConfig(cloth_resolution=24), CLOTH_MATERIAL)
    require(np.abs(body.bend_rest).max() == 0.0,
            "a flat sheet must start at a zero dihedral rest angle")
    # The same function on a right-angle fold must read a right angle, or the
    # rest angle above is zero for the wrong reason.
    folded = dihedral_angle(
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    require(abs(float(folded) - np.pi / 2.0) < 1e-12,
            f"dihedral_angle read {float(folded):.6f} rad for a square fold")


@case
def cloth_total_mass_matches_areal_density() -> None:
    for resolution, size in ((8, 0.6), (33, 0.6), (72, 0.6), (88, 0.72)):
        scene = SceneConfig(cloth_resolution=resolution, cloth_size=size,
                            cloth_pinned="none")
        body = build_cloth(scene, CLOTH_MATERIAL)
        total = float((1.0 / body.inv_mass.astype(np.float64)).sum())
        expected = CLOTH_MATERIAL.areal_density * size * size
        error = abs(total - expected) / expected
        require(error < 0.01,
                f"cloth-{resolution} weighs {total:.5f} kg, expected "
                f"{expected:.5f} kg ({error * 100:.2f}% out)")
        note(f"cloth-{resolution}: {total * 1000:.2f} g vs "
             f"{expected * 1000:.2f} g ({error * 100:.3f}%)")


@case
def cloth_pinning_modes_pin_what_they_say() -> None:
    n = 12
    expected = {"none": 0, "top_corners": 2, "top_edge": n, "corners": 4}
    for mode, count in expected.items():
        body = build_cloth(
            SceneConfig(cloth_resolution=n, cloth_pinned=mode), CLOTH_MATERIAL)
        pinned = np.flatnonzero((body.flags & FLAG_PINNED) != 0)
        require(pinned.size == count,
                f"cloth_pinned={mode!r} pinned {pinned.size} particles, "
                f"expected {count}")
        require(np.all(body.inv_mass[pinned] == 0.0),
                f"cloth_pinned={mode!r}: a pinned particle can still move")
        if mode in ("top_corners", "top_edge"):
            top = body.positions[:, 1].max()
            require(np.allclose(body.positions[pinned, 1], top),
                    f"cloth_pinned={mode!r} pinned something off the top edge")
    try:
        build_cloth(SceneConfig(cloth_pinned="middle"), CLOTH_MATERIAL)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown pinning mode must raise")


@case
def cloth_surface_is_wound_towards_the_viewer() -> None:
    body = build_cloth(SceneConfig(cloth_resolution=16), CLOTH_MATERIAL)
    require(body.double_sided, "a sheet of cloth must be lit from both sides")
    pos = body.positions.astype(np.float64)
    normal = np.cross(pos[body.tri_idx[:, 1]] - pos[body.tri_idx[:, 0]],
                      pos[body.tri_idx[:, 2]] - pos[body.tri_idx[:, 0]])
    require(np.all(normal[:, 2] > 0.0),
            "every cloth triangle must face +Z in the rest configuration")
    require(body.num_tris == 2 * 15 * 15, "cloth must emit two triangles per cell")


@case
def cloth_is_perturbed_but_planar_enough() -> None:
    body = build_cloth(SceneConfig(cloth_resolution=32), CLOTH_MATERIAL)
    z = body.positions[:, 2]
    require(np.abs(z).max() > 0.0, "the sheet needs a symmetry-breaking nudge")
    require(np.abs(z).max() < 1e-3,
            f"the nudge is {np.abs(z).max() * 1000:.3f} mm, big enough to see")
    again = build_cloth(SceneConfig(cloth_resolution=32), CLOTH_MATERIAL)
    require(np.array_equal(body.positions, again.positions),
            "the scene must be reproducible frame for frame")


@case
def the_cloth_nudge_never_outgrows_the_grid() -> None:
    # An absolute nudge is a third of a per cent of a 0.6 m sheet's cell and
    # ten cells wide on a 2 mm one.  Once it passes the spacing the sheet is
    # not a perturbed plane any more, and since the rest angles are measured on
    # the ideal grid it would start loaded with bending energy it did not earn.
    for resolution, size in ((3, 0.6), (32, 0.6), (72, 0.6), (88, 0.72),
                             (100, 0.6), (110, 0.6), (40, 0.25)):
        body = build_cloth(SceneConfig(cloth_resolution=resolution,
                                       cloth_size=size), CLOTH_MATERIAL)
        spacing = size / (resolution - 1)
        nudge = float(np.abs(body.positions[:, 2]).max())
        require(0.0 < nudge <= 0.031 * spacing,
                f"cloth-{resolution} at {size} m nudged by {nudge * 1000:.4f} mm "
                f"against a {spacing * 1000:.4f} mm spacing")
        # The nudge must not disturb what the solver treats as the rest state.
        require(float(np.abs(body.bend_rest).max()) == 0.0,
                f"cloth-{resolution} at {size} m has a non-flat rest angle")
        rest = body.dist_rest.astype(np.float64)
        stretch = rest[body.dist_kind == int(ConstraintKind.STRETCH)]
        require(float(np.abs(stretch - spacing).max()) < 1e-5 * spacing,
                f"cloth-{resolution} at {size} m baked the nudge into its rest "
                f"lengths")


@case
def a_body_under_the_mass_floor_refuses_to_build() -> None:
    # core.material floors every particle mass so a degenerate element cannot
    # hand the solver an infinite inverse mass.  Under that floor the body
    # stops weighing what its density says, and the whole premise that the
    # hardness dial runs on real material constants goes with it.  The one
    # thing that must not happen is building anyway.
    cases = (
        ("cloth", lambda: build_cloth(
            SceneConfig(cloth_resolution=72, cloth_size=0.05), CLOTH_MATERIAL)),
        ("cloth", lambda: build_cloth(
            SceneConfig(cloth_resolution=300, cloth_size=0.6), CLOTH_MATERIAL)),
        ("soft", lambda: build_soft_body(
            SceneConfig(soft_shape="sphere", soft_resolution=13,
                        soft_size=0.01), SOFT_MATERIAL)),
        ("grain", lambda: build_granular(
            SceneConfig(grain_count=64, grain_radius=1e-4,
                        grain_extent=0.02), GRAIN_MATERIAL)),
    )
    for label, build in cases:
        try:
            build()
        except ValueError as exc:
            require("floor" in str(exc),
                    f"{label}: message does not explain the floor: {exc}")
        else:
            raise AssertionError(
                f"{label} built a body whose particles are under the mass floor")

    # Everything shipped must stay clear of the floor with room to spare, or
    # this check turns into a tripwire on the presets themselves.
    for name in PRESETS:
        body = build_scene(preset(name).scene)[0]
        free = body.inv_mass[body.inv_mass > 0]
        lightest = 1.0 / float(free.max())
        floor = 1e-7 if body.kind is MatterKind.GRAIN else 1e-6
        require(lightest > 2.0 * floor,
                f"{name}: lightest particle is {lightest / floor:.2f}x the "
                f"floor, too close to trip on")
        note(f"{name:8s} lightest particle {lightest / floor:.2f}x the floor")


@case
def cloth_uv_spans_the_unit_square() -> None:
    body = build_cloth(SceneConfig(cloth_resolution=20), CLOTH_MATERIAL)
    require(np.isclose(body.uv.min(), 0.0) and np.isclose(body.uv.max(), 1.0),
            "cloth uv must span [0, 1]^2 exactly")
    require(np.all((body.flags & (FLAG_SURFACE | FLAG_SELF_COLLIDE))
                   == (FLAG_SURFACE | FLAG_SELF_COLLIDE)),
            "every cloth particle is on the surface and self-collides")


# --------------------------------------------------------------------------
# soft bodies
# --------------------------------------------------------------------------


@case
def tetrahedra_have_positive_volume() -> None:
    for shape, resolution in _SOFT_CASES:
        body = build_soft_body(
            SceneConfig(soft_shape=shape, soft_resolution=resolution), SOFT_MATERIAL)
        pos = body.positions.astype(np.float64)
        edges = pos[body.tet_idx[:, 1:]] - pos[body.tet_idx[:, [0]]]
        dm = np.swapaxes(edges, 1, 2)
        volume = np.linalg.det(dm) / 6.0
        require(np.all(volume > 0.0),
                f"{shape}-{resolution}: {int((volume <= 0).sum())} inverted tets")
        stored = body.tet_rest_volume.astype(np.float64)
        require(np.allclose(volume, stored, rtol=1e-4, atol=1e-12),
                f"{shape}-{resolution}: tet_rest_volume disagrees with the geometry")


@case
def dm_inverse_uses_the_column_convention() -> None:
    # F = Ds @ Dm_inv must be the identity in the rest pose.  With Dm built
    # from rows instead of columns this returns Dm^T @ Dm^-1, which is still
    # orthogonal-looking for an axis-aligned tet and only goes wrong under
    # shear -- exactly the failure that is impossible to see by eye.
    for shape, resolution in (("sphere", 9), ("box", 6), ("torus", 13)):
        body = build_soft_body(
            SceneConfig(soft_shape=shape, soft_resolution=resolution), SOFT_MATERIAL)
        pos = body.positions.astype(np.float64)
        ds = np.swapaxes(pos[body.tet_idx[:, 1:]] - pos[body.tet_idx[:, [0]]], 1, 2)
        f = ds @ body.tet_dm_inv.astype(np.float64)
        error = np.abs(f - np.eye(3)).max()
        require(error < 1e-4,
                f"{shape}-{resolution}: rest deformation gradient is off the "
                f"identity by {error:.3g}")
        note(f"{shape}-{resolution}: max |F - I| = {error:.2e}")


@case
def soft_surface_is_a_closed_manifold() -> None:
    for shape, resolution in _SOFT_CASES:
        body = build_soft_body(
            SceneConfig(soft_shape=shape, soft_resolution=resolution), SOFT_MATERIAL)
        _closed_manifold(body.tri_idx)


@case
def soft_surface_faces_outward() -> None:
    for shape, resolution in _SOFT_CASES:
        scene = SceneConfig(soft_shape=shape, soft_resolution=resolution)
        body = build_soft_body(scene, SOFT_MATERIAL)
        pos = body.positions.astype(np.float64)
        enclosed = _enclosed_volume(body.tri_idx, pos)
        require(enclosed > 0.0,
                f"{shape}-{resolution}: the surface encloses {enclosed:.4g} m^3, "
                f"so the triangles are wound inward")
        tet_volume = float(body.tet_rest_volume.astype(np.float64).sum())
        error = abs(enclosed - tet_volume) / tet_volume
        require(error < 1e-3,
                f"{shape}-{resolution}: the surface encloses {enclosed:.6g} but "
                f"the tetrahedra fill {tet_volume:.6g}")
        note(f"{shape}-{resolution}: volume {tet_volume * 1e6:.1f} cm^3, "
             f"surface {body.num_tris} tris")


@case
def soft_shapes_have_the_right_size_and_place() -> None:
    for shape in ("sphere", "box", "torus"):
        scene = SceneConfig(soft_shape=shape, soft_resolution=14,
                            soft_size=0.26, soft_height=0.34)
        body = build_soft_body(scene, SOFT_MATERIAL)
        pos = body.positions.astype(np.float64)
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        widest = float((hi - lo).max())
        require(abs(widest - 0.26) < 0.26 * 0.12,
                f"{shape} is {widest:.3f} m across, asked for 0.260 m")
        centre = 0.5 * (lo + hi)
        require(abs(centre[0]) < 1e-6 and abs(centre[2]) < 1e-6,
                f"{shape} is not centred on the stage")
        require(abs(centre[1] - 0.34) < 1e-6,
                f"{shape} sits at y={centre[1]:.4f}, asked for 0.34")
    try:
        build_soft_body(SceneConfig(soft_shape="dodecahedron"), SOFT_MATERIAL)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown soft_shape must raise")


@case
def soft_flags_mark_only_the_shell() -> None:
    body = build_soft_body(
        SceneConfig(soft_shape="sphere", soft_resolution=13), SOFT_MATERIAL)
    surface = np.unique(body.tri_idx)
    marked = np.flatnonzero((body.flags & FLAG_SURFACE) != 0)
    require(np.array_equal(surface, marked),
            "FLAG_SURFACE must mark exactly the particles the surface uses")
    require(np.array_equal(
        np.flatnonzero((body.flags & FLAG_SELF_COLLIDE) != 0), marked),
        "interior particles must not take part in self-collision")
    require(marked.size < body.num_particles,
            "a solid body must have interior particles at this resolution")
    note(f"sphere-13: {marked.size} surface of {body.num_particles} particles")


@case
def soft_mass_matches_its_density() -> None:
    for shape in ("sphere", "box", "torus"):
        body = build_soft_body(
            SceneConfig(soft_shape=shape, soft_resolution=13), SOFT_MATERIAL)
        total = float((1.0 / body.inv_mass.astype(np.float64)).sum())
        volume = float(body.tet_rest_volume.astype(np.float64).sum())
        expected = SOFT_MATERIAL.density * volume
        error = abs(total - expected) / expected
        require(error < 1e-3,
                f"{shape} weighs {total:.4f} kg but its {volume:.6f} m^3 at "
                f"{SOFT_MATERIAL.density} kg/m^3 should be {expected:.4f} kg")
        note(f"{shape}: {total * 1000:.1f} g")


@case
def voxel_repair_closes_critical_configurations() -> None:
    solid = np.zeros((3, 3, 3), dtype=bool)
    solid[0, 0, 0] = solid[1, 1, 0] = True          # 2-D diagonal pair
    solid[1, 1, 2] = solid[2, 2, 2] = True
    repaired = _make_well_composed(solid)
    require(repaired.sum() > solid.sum(), "a diagonal-only pair must be bridged")
    _closed_manifold(build_soft_body(
        SceneConfig(soft_shape="torus", soft_resolution=17), SOFT_MATERIAL).tri_idx)

    corner = np.zeros((2, 2, 2), dtype=bool)
    corner[0, 0, 0] = corner[1, 1, 1] = True        # 3-D antipodal pair
    require(_make_well_composed(corner).sum() == 8,
            "an antipodal-only pair must be closed by completing the block")
    require(_no_critical_configuration(repaired) == 0,
            "the repair left a critical configuration behind")


def _no_critical_configuration(solid: np.ndarray) -> int:
    """Count 2-D diagonal-only squares left in a voxel set."""
    bad = 0
    for axis_u, axis_v in ((0, 1), (0, 2), (1, 2)):
        def shift(du: int, dv: int) -> tuple[slice, ...]:
            sl = [slice(None)] * 3
            sl[axis_u] = slice(du, du + solid.shape[axis_u] - 1)
            sl[axis_v] = slice(dv, dv + solid.shape[axis_v] - 1)
            return tuple(sl)
        aa, bb, ab, ba = shift(0, 0), shift(1, 1), shift(0, 1), shift(1, 0)
        bad += int((solid[aa] & solid[bb] & ~solid[ab] & ~solid[ba]).sum())
        bad += int((solid[ab] & solid[ba] & ~solid[aa] & ~solid[bb]).sum())
    return bad


@case
def the_repair_respects_the_lattice_mirrors() -> None:
    # The cheap repair -- name one of the two empty cells of a critical square
    # -- is not equivariant under the reflection that swaps them, so a shape
    # symmetric about the origin came out lopsided: the torus's centre of mass
    # sat 10.8 mm off axis at resolution 4 and 1.6 mm off at 8.  A centre of
    # mass off the stage centre is a body that leans and rolls for no reason
    # the viewer can see.
    for shape, resolution in _SOFT_CASES + [("torus", res) for res in (15, 29, 35)]:
        body = build_soft_body(
            SceneConfig(soft_shape=shape, soft_resolution=resolution), SOFT_MATERIAL)
        pos = body.positions.astype(np.float64)
        mass = 1.0 / body.inv_mass.astype(np.float64)
        com = (pos * mass[:, None]).sum(axis=0) / mass.sum()
        off = max(abs(com[0]), abs(com[2]))
        require(off < 1e-9,
                f"{shape}-{resolution}: centre of mass is {off * 1000:.4f} mm "
                f"off the vertical axis")
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        centre = 0.5 * (lo + hi)
        require(abs(centre[0]) < 1e-9 and abs(centre[2]) < 1e-9,
                f"{shape}-{resolution}: bounding box is not centred")


@case
def the_torus_keeps_its_hole() -> None:
    # Completing a critical square costs voxels, and the cheapest place to
    # spend them is the narrow part of the ring.  Euler characteristic 0 is
    # what says the repair widened the tube instead of plugging the hole.
    for resolution in (5, 6, 7, 8, 9, 13, 14, 17, 20):
        body = build_soft_body(
            SceneConfig(soft_shape="torus", soft_resolution=resolution),
            SOFT_MATERIAL)
        tri = body.tri_idx
        verts = int(np.unique(tri).size)
        edges = int(np.unique(np.sort(np.concatenate(
            [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1),
            axis=0).shape[0])
        chi = verts - edges + tri.shape[0]
        require(chi == 0,
                f"torus-{resolution} has Euler characteristic {chi}, so it is "
                f"genus {(2 - chi) // 2}, not a torus")


@case
def soft_uv_is_not_degenerate() -> None:
    body = build_soft_body(
        SceneConfig(soft_shape="sphere", soft_resolution=13), SOFT_MATERIAL)
    require(np.isfinite(body.uv).all(), "soft-body uv contains non-finite values")
    require(body.uv.min() >= 0.0 and body.uv.max() <= 1.0,
            "soft-body uv must stay inside the unit square")
    require(np.ptp(body.uv[:, 0]) > 0.9 and np.ptp(body.uv[:, 1]) > 0.9,
            "soft-body uv collapsed to a line")


# --------------------------------------------------------------------------
# granular
# --------------------------------------------------------------------------


@case
def grains_fill_their_box_without_overlapping() -> None:
    for count in (1000, 8000, 24000):
        scene = SceneConfig(grain_count=count)
        body = build_granular(scene, GRAIN_MATERIAL)
        require(body.num_particles == count,
                f"asked for {count} grains, got {body.num_particles}")
        pos = body.positions.astype(np.float64)
        offset = np.abs(pos - np.array([0.0, scene.grain_height, 0.0]))
        require(offset.max() + scene.grain_radius <= scene.grain_extent + 1e-6,
                "a grain escaped the box")
        require(np.all((body.flags & FLAG_SELF_COLLIDE) != 0),
                "every grain must self-collide")
        require(np.all(body.flags & FLAG_SURFACE == 0),
                "grains are drawn as sprites, not as a surface")
        require(body.num_tris == 0 and body.num_dist == 0 and body.num_tets == 0,
                "granular matter carries no constraints and no surface")

        # Grains that start interpenetrating make the first substep resolve
        # the whole pack at once, which reads as an explosion.
        closest = _min_separation(pos, 4.0 * scene.grain_radius)
        require(closest >= 2.0 * scene.grain_radius,
                f"grain-{count}: closest pair is {closest * 1000:.3f} mm apart, "
                f"less than the {scene.grain_radius * 2000:.3f} mm diameter")
        note(f"grain-{count}: block "
             f"{(pos.max(axis=0) - pos.min(axis=0)).round(3).tolist()} m, "
             f"closest pair {closest * 1000:.3f} mm")


@case
def grains_start_outside_the_solvers_contact_threshold() -> None:
    # Not overlapping is not enough.  The solver calls two particles a contact
    # at (r_i + r_j) * (1 + collision_margin) and starts pushing, so a pack
    # that merely avoids overlap still hands the first substep the whole pile
    # to resolve at once.  That is not theoretical: a slack of 1.12 against a
    # margin of 0.12 put 66792 of the 24000-grain pile's pairs in contact on
    # frame 0 and the pile left the stage at the velocity clamp.
    margin = SolverConfig().collision_margin
    for count in (1000, 8000, 24000):
        scene = SceneConfig(grain_count=count)
        body = build_granular(scene, GRAIN_MATERIAL)
        pos = body.positions.astype(np.float64)
        threshold = 2.0 * body.particle_radius * (1.0 + margin)
        closest = _min_separation(pos, 2.0 * threshold)
        require(closest > threshold,
                f"grain-{count}: closest pair is {closest * 1000:.3f} mm apart "
                f"but the solver starts separating at {threshold * 1000:.3f} mm")
        note(f"grain-{count}: closest pair {closest * 1000:.3f} mm, "
             f"contact threshold {threshold * 1000:.3f} mm "
             f"({100 * (closest / threshold - 1):.1f}% clear)")
    # The constant is derived from the margin; this is the arithmetic that
    # derivation has to satisfy, spelled out so a change to either number
    # fails here instead of on screen.
    from fctx.bodies.granular import _JITTER_FRACTION, _SPACING_SLACK
    require(_SPACING_SLACK - np.sqrt(3.0) * _JITTER_FRACTION > 1.0 + margin,
            f"a slack of {_SPACING_SLACK:.4f} with {_JITTER_FRACTION} jitter "
            f"does not clear a collision margin of {margin}")


@case
def grain_packing_is_reproducible() -> None:
    a = build_granular(SceneConfig(grain_count=5000), GRAIN_MATERIAL)
    b = build_granular(SceneConfig(grain_count=5000), GRAIN_MATERIAL)
    require(np.array_equal(a.positions, b.positions),
            "the grain packing must come from a seeded RNG")


@case
def an_impossible_box_fails_loudly() -> None:
    try:
        build_granular(
            SceneConfig(grain_count=1_000_000, grain_extent=0.05), GRAIN_MATERIAL)
    except ValueError as exc:
        require("box" in str(exc), f"unhelpful error message: {exc}")
    else:
        raise AssertionError("over-filling the box must raise")


@case
def the_surface_fit_never_leaves_a_sliver_in_the_rest_pose() -> None:
    """A flattened rest tetrahedron is one that inverts under contact load.

    The backoff used to measure only volume, which cannot tell a small
    tetrahedron from a flat one, so the skin elements it drove down to the
    volume floor came out at two thirds of the lattice's shape quality -- and
    22 of the shipped sphere's 12765 tetrahedra were permanently inverted
    after a three-second settle on the floor.  Measured here at rest, where
    it is cheap and needs no GPU.
    """
    from fctx.bodies.softbody import (
        _lattice_axis,
        _tet_quality,
        _tetrahedra,
        _voxelise,
    )

    # The builder's floor is 0.80 and the discrete backoff overshoots it, so
    # the measured worst is 0.84 or better on every shipped shape.  The
    # literal here rather than the constant: this has to fail if the floor is
    # ever lowered back to where the volume check alone was leaving 0.67.
    worst_allowed = 0.78

    for shape, res in (("sphere", 13), ("sphere", 17), ("torus", 14),
                       ("box", 12)):
        scene = SceneConfig(soft_shape=shape, soft_resolution=res)
        body = build_soft_body(scene, SOFT_MATERIAL)
        built = _tet_quality(np.asarray(body.positions, np.float64),
                             np.asarray(body.tet_idx, np.int64))
        solid, cell, dims = _voxelise(shape, res)
        coords, lattice_tets = _tetrahedra(solid)
        lattice = _lattice_axis(2 * coords, dims, cell)
        raw = _tet_quality(lattice, lattice_tets)
        ratio = float(built.min()) / float(raw.min())
        note(f"{shape}-{res}: worst rest shape quality {built.min():.4f} "
             f"against the lattice's {raw.min():.4f} ({ratio:.3f} of it)")
        require(ratio >= worst_allowed,
                f"{shape}-{res}: the fit left a tetrahedron at {ratio:.3f} of "
                f"the lattice's worst shape quality, under the "
                f"{worst_allowed} the backoff is supposed to hold")


@case
def fitting_a_box_leaves_its_lattice_exactly_alone() -> None:
    """A box's lattice vertices are already on its isosurface.

    Every resolution, every vertex, zero travel -- so the fourteen backoff
    passes are pure startup cost, and the early-out that skips them has to
    reproduce the lattice exactly, not nearly.
    """
    from fctx.bodies.softbody import (
        _fit_to_surface,
        _lattice_axis,
        _surface,
        _tetrahedra,
        _voxelise,
    )

    for res in (2, 5, 12, 20):
        solid, cell, dims = _voxelise("box", res)
        coords, tets = _tetrahedra(solid)
        lattice = _lattice_axis(2 * coords, dims, cell)
        boundary = np.unique(_surface(tets, lattice))
        fitted, skin = _fit_to_surface(lattice, boundary, tets, "box", cell)
        require(np.array_equal(fitted, lattice),
                f"box-{res}: the fit moved a lattice that is already on the "
                f"isosurface by up to {np.abs(fitted - lattice).max():.3g} m")
        require(np.array_equal(skin, lattice),
                f"box-{res}: the render skin was pulled off a lattice that "
                f"was already exactly on the isosurface")
    note("box lattices at resolutions 2, 5, 12 and 20 come back bit-identical, "
         "skin included")


@case
def cloth_mass_helper_agrees_with_the_material() -> None:
    # If this drifts, the 1% total-mass check above is measuring the builder
    # against itself instead of against core.material.
    scene = SceneConfig(cloth_resolution=20, cloth_size=0.5, cloth_pinned="none")
    body = build_cloth(scene, CLOTH_MATERIAL)
    spacing = scene.cloth_size / (scene.cloth_resolution - 1)
    interior = areal_particle_mass(CLOTH_MATERIAL, spacing * spacing)
    heaviest = 1.0 / float(body.inv_mass.min())
    require(abs(heaviest - interior) < 1e-12,
            f"the heaviest particle weighs {heaviest:.6g} kg, the material says "
            f"{interior:.6g} kg")


if __name__ == "__main__":
    sys.exit(run(__file__))


def skin_positions(body, pos: np.ndarray) -> np.ndarray:
    """The drawn position of every particle, exactly as kernels.apply_skin
    computes it: the plain sum over every (element, corner) slot, with
    unbound particles drawn where they are."""
    out = pos.copy()
    tet = body.skin_tet
    bary = body.skin_bary.astype(np.float64)
    used = tet >= 0
    safe = np.where(used, tet, 0)
    corners = pos[body.tet_idx[safe]]                    # (P, K, 4, 3)
    contrib = (corners * bary[..., None]).sum(axis=(1, 2))
    bound = used.any(axis=1)
    out[bound] = contrib[bound]
    return out


@case
def the_render_skin_is_bound_to_the_real_isosurface() -> None:
    """The drawn surface must be the shape, not the lattice under it.

    A voxelised body cannot be both smooth and stable with one set of points:
    pulling the skin all the way onto the isosurface is what makes it look
    like a sphere, and the boundary tetrahedra are what invert when it is then
    squeezed.  So the skin is bound to the lattice barycentrically instead.
    This checks the binding reconstructs the isosurface exactly at rest --
    which is the only place it can be checked without a solver.
    """
    for shape, resolution in (("sphere", 9), ("sphere", 21), ("torus", 13),
                              ("box", 11)):
        scene = SceneConfig(soft_shape=shape, soft_resolution=resolution,
                            soft_size=0.26, soft_height=0.34)
        body = build_soft_body(scene, SOFT_MATERIAL)
        body.validate()
        require(body.skin_tet is not None and body.skin_bary is not None,
                f"{shape}: no render skin was emitted")

        pos = body.positions.astype(np.float64)
        bound = (body.skin_tet >= 0).any(axis=1)
        surface = np.unique(body.tri_idx)
        missing = np.setdiff1d(surface, np.flatnonzero(bound))
        require(missing.size == 0,
                f"{shape}: {missing.size} drawn vertices have no skin binding")
        require(not bound[np.setdiff1d(np.arange(body.num_particles), surface)].any(),
                f"{shape}: an interior particle was given a skin binding")

        skin = skin_positions(body, pos)[bound]
        require(np.isfinite(skin).all(), f"{shape}: skin is not finite")

        # Every skinned vertex must sit on the isosurface.  The shapes live in
        # a [-1, 1] box scaled by soft_size / 2, so the SDF reads in those
        # units.
        scale = 0.5 * float(scene.soft_size)
        centre = np.array([0.0, float(scene.soft_height), 0.0])
        sdf = {"sphere": sdf_sphere, "box": sdf_box, "torus": sdf_torus}[shape]
        residual = np.abs(sdf((skin - centre) / scale)) * scale
        lattice = np.abs(sdf((pos[bound] - centre) / scale)) * scale
        require(residual.max() < 0.004,
                f"{shape}-{resolution}: a drawn vertex sits "
                f"{residual.max() * 1000:.2f} mm off the surface")
        require(residual.mean() < 0.2 * max(lattice.mean(), 1e-9),
                f"{shape}-{resolution}: the skin is no closer to the surface "
                f"than the lattice it was supposed to improve on")
        note(f"{shape}-{resolution}: lattice {lattice.mean() * 1000:5.2f} mm "
             f"off the surface, skin {residual.mean() * 1000:5.3f} mm "
             f"(worst {residual.max() * 1000:.3f})")


@case
def the_render_skin_follows_the_body_rigidly() -> None:
    """An affine move of the lattice must move the skin the same way.

    This is the property the whole binding rests on: the weights are
    barycentric, so anything affine -- a translation, a rotation, a uniform
    stretch -- has to carry the skin exactly, with no drift and no scaling of
    its own.  If it did not, a body that merely fell would visibly shed its
    surface.
    """
    scene = SceneConfig(soft_shape="sphere", soft_resolution=13,
                        soft_size=0.26, soft_height=0.34)
    body = build_soft_body(scene, SOFT_MATERIAL)
    pos = body.positions.astype(np.float64)
    bound = (body.skin_tet >= 0).any(axis=1)

    def skin_of(p: np.ndarray) -> np.ndarray:
        return skin_positions(body, p)[bound]

    rest = skin_of(pos)
    angle = 0.7
    rot = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(angle), 0.0, np.cos(angle)]])
    shear = np.array([[1.0, 0.25, 0.0], [0.0, 1.1, 0.0], [0.15, 0.0, 0.9]])
    for name, matrix, offset in (("translate", np.eye(3), np.array([0.4, -0.2, 0.7])),
                                 ("rotate", rot, np.zeros(3)),
                                 ("stretch", np.diag([1.3, 0.8, 1.3]), np.zeros(3)),
                                 ("shear", shear, np.array([0.1, 0.1, 0.1]))):
        moved = pos @ matrix.T + offset
        expected = rest @ matrix.T + offset
        err = np.abs(skin_of(moved) - expected).max()
        require(err < 1e-9,
                f"under a {name} the skin drifted by {err:.3e} m")
    note("translation, rotation, stretch and shear all carry the skin exactly")
