"""Cloth: a regular grid of particles with stretch, shear and dihedral bend.

Pure numpy.  Nothing here knows that a GPU exists.
"""

from __future__ import annotations

import numpy as np

from ..config import SceneConfig
from ..core.material import MaterialParams, areal_particle_mass
from ..core.types import (
    FLAG_PINNED,
    FLAG_SELF_COLLIDE,
    FLAG_SURFACE,
    BodyData,
    ConstraintKind,
    MatterKind,
)
from .coloring import greedy_color

__all__ = ["build_cloth", "dihedral_angle", "PINNING_MODES"]

PINNING_MODES = ("none", "top_corners", "two_points", "top_edge", "corners")

#: Where "two_points" puts its pins along the top edge, as a fraction of the
#: width from the left.  Pinning the actual corners looks wrong: the top edge
#: is inextensible, so pins exactly one width apart hold it dead straight and
#: the sheet hangs as a smooth shell with no folds at all.  Bringing the pins
#: in leaves slack, the top edge buckles, and the fabric breaks into the
#: diagonal folds people recognise as cloth.
TWO_POINT_PINS = (0.22, 0.78)

#: A perfectly planar sheet is a degenerate rest state for the dihedral
#: gradient (1/sqrt(1-d^2) with d == 1) and, worse, has no broken symmetry for
#: the first fold to follow, so it hangs like a board until something disturbs
#: it.  A quarter of a millimetre of noise is invisible, costs no energy
#: because the rest quantities are measured on the ideal flat grid, and gets
#: the sheet folding the way fabric does on the first frame.
_PERTURB_METRES = 2.5e-4
#: ...and never more than this fraction of the grid spacing, whichever is
#: smaller.  A quarter of a millimetre is a third of a per cent of a 0.6 m
#: sheet's cell but ten cells wide on a 2 mm one, where it stops being a nudge
#: and becomes a crumpled ball whose rest angles -- measured on the ideal flat
#: grid -- describe nothing like its actual shape, so it starts loaded with
#: bending energy and snaps straight on the first frame.  Three per cent of the
#: spacing is a rest-angle error of about 0.06 rad, which is what the shipped
#: resolutions already had.
_PERTURB_SPACING_FRACTION = 0.03
_PERTURB_SEED = 0x_C10C_5EED


def dihedral_angle(
    e0: np.ndarray, e1: np.ndarray, w0: np.ndarray, w1: np.ndarray
) -> np.ndarray:
    """Angle between the two triangles hinged on edge ``(e0, e1)``.

    ``w0`` and ``w1`` are the wing vertices.  All four are ``(..., 3)``.  The
    normals follow section 6.3 of ARCHITECTURE.md exactly -- n1 from the
    ``w0`` triangle, n2 from the ``w1`` triangle with the edge reversed -- so
    that the value returned here is the same quantity the solver's kernel
    subtracts its rest angle from.  Get the winding wrong and the sheet folds
    towards the wrong side of flat.
    """
    n1 = np.cross(e0 - w0, e1 - w0)
    n2 = np.cross(e1 - w1, e0 - w1)
    l1 = np.linalg.norm(n1, axis=-1)
    l2 = np.linalg.norm(n2, axis=-1)
    if np.any(l1 <= 0.0) or np.any(l2 <= 0.0):
        raise ValueError("degenerate bending triangle: zero-area wing")
    d = np.einsum("...i,...i->...", n1, n2) / (l1 * l2)
    return np.arccos(np.clip(d, -1.0, 1.0))


def _pinned_mask(mode: str, n: int) -> np.ndarray:
    """Boolean ``(n, n)`` mask indexed ``[row, column]`` = ``[y, x]``."""
    mask = np.zeros((n, n), dtype=bool)
    if mode == "none":
        return mask
    if mode == "top_corners":
        mask[n - 1, 0] = True
        mask[n - 1, n - 1] = True
        return mask
    if mode == "two_points":
        for fraction in TWO_POINT_PINS:
            mask[n - 1, min(int(round(fraction * (n - 1))), n - 1)] = True
        return mask
    if mode == "top_edge":
        mask[n - 1, :] = True
        return mask
    if mode == "corners":
        mask[0, 0] = mask[0, n - 1] = mask[n - 1, 0] = mask[n - 1, n - 1] = True
        return mask
    raise ValueError(
        f"unknown cloth_pinned {mode!r}; expected one of {PINNING_MODES}")


def build_cloth(scene: SceneConfig, mp: MaterialParams) -> BodyData:
    """Build the cloth described by ``scene``, with ``mp``'s areal density."""
    n = int(scene.cloth_resolution)
    if n < 3:
        raise ValueError(
            f"cloth_resolution must be at least 3 to have an interior edge "
            f"to bend across, got {n}")
    size = float(scene.cloth_size)
    if not size > 0.0:
        raise ValueError(f"cloth_size must be positive, got {size}")

    spacing = size / (n - 1)
    cell_area = spacing * spacing
    num_particles = n * n

    # [row, column] == [y, x]; row 0 is the bottom edge, row n-1 the top.
    pid = np.arange(num_particles, dtype=np.int64).reshape(n, n)

    axis = np.linspace(-0.5 * size, 0.5 * size, n, dtype=np.float64)
    flat = np.empty((num_particles, 3), dtype=np.float64)
    flat[:, 0] = np.tile(axis, n)
    flat[:, 1] = np.repeat(axis, n) + float(scene.cloth_height)
    flat[:, 2] = 0.0

    # --- distance constraints -------------------------------------------
    stretch = np.concatenate([
        np.stack([pid[:, :-1].ravel(), pid[:, 1:].ravel()], axis=1),
        np.stack([pid[:-1, :].ravel(), pid[1:, :].ravel()], axis=1),
    ])
    shear = np.concatenate([
        np.stack([pid[:-1, :-1].ravel(), pid[1:, 1:].ravel()], axis=1),
        np.stack([pid[:-1, 1:].ravel(), pid[1:, :-1].ravel()], axis=1),
    ])
    dist_idx = np.concatenate([stretch, shear])
    dist_kind = np.concatenate([
        np.full(stretch.shape[0], int(ConstraintKind.STRETCH)),
        np.full(shear.shape[0], int(ConstraintKind.SHEAR)),
    ])
    # Measured on the ideal grid, not on the perturbed positions: the rest
    # state of this sheet is flat, and a rest length that bakes in the
    # symmetry-breaking noise would leave the cloth permanently pre-stressed.
    dist_rest = np.linalg.norm(flat[dist_idx[:, 0]] - flat[dist_idx[:, 1]], axis=1)

    # --- dihedral bending -------------------------------------------------
    # Creases run along both grid axes.  Each hinge is an axis-aligned edge
    # with the neighbouring particles on either side as wings, which resists
    # the sheet folding about that axis.
    bend_v = np.stack([
        pid[:-1, 1:-1].ravel(), pid[1:, 1:-1].ravel(),
        pid[:-1, :-2].ravel(), pid[:-1, 2:].ravel(),
    ], axis=1)
    bend_h = np.stack([
        pid[1:-1, :-1].ravel(), pid[1:-1, 1:].ravel(),
        pid[:-2, :-1].ravel(), pid[2:, :-1].ravel(),
    ], axis=1)
    bend_idx = np.concatenate([bend_v, bend_h])
    bend_rest = dihedral_angle(
        flat[bend_idx[:, 0]], flat[bend_idx[:, 1]],
        flat[bend_idx[:, 2]], flat[bend_idx[:, 3]],
    )

    # --- mass --------------------------------------------------------------
    # Each particle owns a quarter of every cell it corners, so edge particles
    # own half a cell and corners a quarter.  Giving every particle a whole
    # cell instead would overstate the sheet's mass by (n/(n-1))^2, nearly 3%
    # at n = 72, and make the hem as heavy as the middle.
    share = np.full(n, 1.0)
    share[0] = share[-1] = 0.5
    owned_area = np.outer(share, share).ravel() * cell_area
    # areal_particle_mass floors its result so a degenerate cell cannot produce
    # an infinite inverse mass.  Past that floor the sheet quietly stops
    # weighing what its areal density says: the four corners go over first (at
    # cloth_resolution 128 on a 0.6 m sheet), and by resolution 242 the whole
    # sheet is more than 1% heavy.  Refusing beats shipping a sheet that falls
    # at the wrong speed for no visible reason.
    floor = areal_particle_mass(mp, 0.0)
    if areal_particle_mass(mp, float(owned_area.min())) <= floor:
        raise ValueError(
            f"cloth {n}x{n} at cloth_size={size} m owns only "
            f"{owned_area.min():.3g} m^2 per corner particle, which at "
            f"{mp.areal_density} kg/m^2 is at or under the {floor:.3g} kg floor "
            f"in areal_particle_mass; the sheet would come out heavier than its "
            f"areal density. Lower cloth_resolution or raise cloth_size.")
    mass = np.empty(num_particles, dtype=np.float64)
    for value in np.unique(owned_area):
        mass[owned_area == value] = areal_particle_mass(mp, float(value))

    pinned = _pinned_mask(str(scene.cloth_pinned), n).ravel()
    inv_mass = np.where(pinned, 0.0, 1.0 / mass)

    flags = np.full(num_particles, FLAG_SURFACE | FLAG_SELF_COLLIDE, dtype=np.uint32)
    flags[pinned] |= np.uint32(FLAG_PINNED)

    # --- render surface ----------------------------------------------------
    a = pid[:-1, :-1].ravel()
    b = pid[:-1, 1:].ravel()
    c = pid[1:, 1:].ravel()
    d = pid[1:, :-1].ravel()
    # Counter-clockwise seen from +Z, for both triangles of every cell.
    tri_idx = np.concatenate([
        np.stack([a, b, c], axis=1),
        np.stack([a, c, d], axis=1),
    ])

    uv = np.empty((num_particles, 2), dtype=np.float64)
    unit = np.linspace(0.0, 1.0, n, dtype=np.float64)
    uv[:, 0] = np.tile(unit, n)
    uv[:, 1] = np.repeat(unit, n)

    positions = flat.copy()
    nudge = min(_PERTURB_METRES, _PERTURB_SPACING_FRACTION * spacing)
    rng = np.random.default_rng(_PERTURB_SEED)
    positions[:, 2] += rng.uniform(-nudge, nudge, num_particles)

    body = BodyData(
        kind=MatterKind.CLOTH,
        name=f"cloth-{n}x{n}",
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        inv_mass=np.ascontiguousarray(inv_mass, dtype=np.float32),
        flags=np.ascontiguousarray(flags, dtype=np.uint32),
        dist_idx=np.ascontiguousarray(dist_idx, dtype=np.int32),
        dist_rest=np.ascontiguousarray(dist_rest, dtype=np.float32),
        dist_kind=np.ascontiguousarray(dist_kind, dtype=np.int32),
        dist_color=greedy_color(dist_idx.astype(np.int32), num_particles),
        bend_idx=np.ascontiguousarray(bend_idx, dtype=np.int32),
        bend_rest=np.ascontiguousarray(bend_rest, dtype=np.float32),
        bend_color=greedy_color(bend_idx.astype(np.int32), num_particles),
        tet_idx=np.zeros((0, 4), dtype=np.int32),
        tet_dm_inv=np.zeros((0, 3, 3), dtype=np.float32),
        tet_rest_volume=np.zeros(0, dtype=np.float32),
        tet_color=np.zeros(0, dtype=np.int32),
        tri_idx=np.ascontiguousarray(tri_idx, dtype=np.int32),
        uv=np.ascontiguousarray(uv, dtype=np.float32),
        double_sided=True,
        # Just over half the grid spacing, so a sheet folded back on itself
        # separates by about one layer thickness instead of passing through.
        particle_radius=0.55 * spacing,
    )
    body.validate()
    return body
