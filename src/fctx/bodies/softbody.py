"""Soft bodies: voxelise an implicit shape, then tetrahedralise the voxels.

Pure numpy.  The output feeds the stable Neo-Hookean tetrahedron constraints
described in section 6.4 of ARCHITECTURE.md.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..config import SceneConfig
from ..core.material import MaterialParams, volumetric_particle_mass
from ..core.types import (
    FLAG_SELF_COLLIDE,
    FLAG_SURFACE,
    BodyData,
    ConstraintKind,
    MatterKind,
)
from .coloring import greedy_color

__all__ = ["build_soft_body", "SHAPES", "sdf_sphere", "sdf_box", "sdf_torus"]

SHAPES = ("sphere", "box", "torus")

#: Torus radii in the shape's own [-1, 1] box, chosen so the outer diameter is
#: exactly 2 and the shape shares the sphere's and the box's scale.
_TORUS_MAJOR = 0.70
_TORUS_MINOR = 0.30

# The two complementary 5-tetrahedron splits of a cube, as (dx, dy, dz)
# corner offsets.  A single split orientation would cut each shared face along
# a diagonal that the neighbouring cell disagrees with, so the two cells would
# not share faces and the mesh would be full of slits; alternating the split by
# cell parity makes every internal face match.  _TET_ODD is _TET_EVEN mirrored
# through x -> 1-x, which is what guarantees the match on all three axes.
_TET_EVEN = (
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 1, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 0, 0), (1, 1, 1), (0, 0, 1)),
    ((0, 1, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
)
_TET_ODD = (
    ((1, 0, 0), (0, 0, 0), (1, 1, 0), (1, 0, 1)),
    ((0, 1, 0), (0, 0, 0), (1, 1, 0), (0, 1, 1)),
    ((0, 0, 1), (0, 0, 0), (1, 0, 1), (0, 1, 1)),
    ((1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
)

_MAX_MANIFOLD_PASSES = 64


def sdf_sphere(p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p, axis=-1) - 1.0


def sdf_box(p: np.ndarray) -> np.ndarray:
    q = np.abs(p) - 1.0
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


def sdf_torus(p: np.ndarray) -> np.ndarray:
    """Torus about the +Y axis: a donut lying flat, as gravity expects."""
    radial = np.linalg.norm(p[..., [0, 2]], axis=-1) - _TORUS_MAJOR
    return np.hypot(radial, p[..., 1]) - _TORUS_MINOR


_SDFS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], tuple[float, float, float]]] = {
    "sphere": (sdf_sphere, (2.0, 2.0, 2.0)),
    "box": (sdf_box, (2.0, 2.0, 2.0)),
    "torus": (sdf_torus, (2.0, 2.0 * _TORUS_MINOR, 2.0)),
}


def _make_well_composed(solid: np.ndarray) -> np.ndarray:
    """Add cells until the voxel set has a manifold boundary.

    A voxel set whose boundary is a closed 2-manifold is exactly a
    *well-composed* set: no 2x2 square in any axis plane may hold only a
    diagonal pair, and no 2x2x2 block may hold only an antipodal pair.  Either
    configuration leaves a boundary edge shared by four triangles instead of
    two, which makes normals, self-collision and any downstream half-edge walk
    nonsense.

    Every repair completes the offending block rather than bridging it with
    the one or two cells that would technically suffice.  The cheap repair has
    to name one of the empty cells, and the symmetries that fix a critical
    configuration -- reflecting a square across the diagonal it is missing,
    rotating a cube about the diagonal joining the two corner-touching cells --
    permute those candidates, so naming one makes the output depend on the
    order the axes happen to be in.  A torus is symmetric about the origin, so
    that showed up directly as a body whose centre of mass sat 10.8 mm off
    axis at soft_resolution=4 and 1.6 mm off at 8.  Completing the block costs
    at most a few per cent more voxels and preserves the hole: the repaired
    torus still has Euler characteristic 0 at every resolution that resolves
    one at all.
    """
    solid = solid.copy()
    for _ in range(_MAX_MANIFOLD_PASSES):
        # One pass reads a frozen snapshot and applies every repair at the end,
        # so the result cannot depend on which plane was visited first either.
        snapshot = solid.copy()
        fill = np.zeros_like(solid)

        for axis_u, axis_v in ((0, 1), (0, 2), (1, 2)):
            def shift(du: int, dv: int) -> tuple[slice, ...]:
                sl = [slice(None)] * 3
                sl[axis_u] = slice(du, du + solid.shape[axis_u] - 1)
                sl[axis_v] = slice(dv, dv + solid.shape[axis_v] - 1)
                return tuple(sl)

            aa, bb = shift(0, 0), shift(1, 1)
            ab, ba = shift(0, 1), shift(1, 0)
            main = snapshot[aa] & snapshot[bb] & ~snapshot[ab] & ~snapshot[ba]
            anti = snapshot[ab] & snapshot[ba] & ~snapshot[aa] & ~snapshot[bb]
            fill[ab] |= main
            fill[ba] |= main
            fill[aa] |= anti
            fill[bb] |= anti

        block = {}
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    block[(dx, dy, dz)] = (
                        slice(dx, dx + solid.shape[0] - 1),
                        slice(dy, dy + solid.shape[1] - 1),
                        slice(dz, dz + solid.shape[2] - 1),
                    )
        antipodal = (
            ((0, 0, 0), (1, 1, 1)),
            ((1, 0, 0), (0, 1, 1)),
            ((0, 1, 0), (1, 0, 1)),
            ((0, 0, 1), (1, 1, 0)),
        )
        for lo, hi in antipodal:
            others = np.zeros_like(snapshot[block[lo]])
            for key, where in block.items():
                if key not in (lo, hi):
                    others |= snapshot[where]
            critical = snapshot[block[lo]] & snapshot[block[hi]] & ~others
            if critical.any():
                for where in block.values():
                    fill[where] |= critical

        if not fill.any():
            return solid
        solid |= fill
    raise RuntimeError(
        "voxel manifold repair did not converge; the lattice is pathological")


def _lattice_axis(steps: np.ndarray, dim: int | np.ndarray,
                  cell: float) -> np.ndarray:
    """Lattice coordinates about the origin, exactly antisymmetric.

    ``steps`` counts half-cells: ``2i + 1`` addresses the centre of cell ``i``
    and ``2j`` addresses vertex ``j``.  Both arguments are integers, and
    ``dim`` broadcasts, so one call converts an ``(N, 3)`` block of vertex
    indices against the three lattice dimensions.  Written the obvious way, as
    ``origin + (i + 0.5) * cell`` with ``origin = -0.5 * dim * cell``, the two
    ends of the lattice round differently, and a cell centre that lands within
    an ULP of the surface is then inside on one side of the origin and outside
    on the other.  That costs the torus two of its 542 cells at
    soft_resolution=15.  Halving a float is exact and IEEE multiplication is
    sign-symmetric, so scaling an exact integer keeps index ``i`` and its
    mirror exact negatives; every SDF here is even in each coordinate, so the
    voxel set comes out exactly symmetric.
    """
    return (steps - dim) * (0.5 * cell)


def _voxelise(shape: str, resolution: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Return ``(solid, cell, dims)`` for ``shape`` at ``resolution``."""
    try:
        sdf, extent = _SDFS[shape]
    except KeyError:
        raise ValueError(
            f"unknown soft_shape {shape!r}; expected one of {SHAPES}") from None
    if resolution < 2:
        raise ValueError(f"soft_resolution must be at least 2, got {resolution}")

    # Cubic cells sized off the longest axis, so a torus does not end up with
    # four squashed cells through its thickness.
    cell = max(extent) / resolution
    dims = np.array([max(1, int(np.ceil(e / cell - 1e-9))) for e in extent])

    centres = np.stack(np.meshgrid(
        *(_lattice_axis(2 * np.arange(d) + 1, d, cell) for d in dims),
        indexing="ij",
    ), axis=-1)
    solid = sdf(centres) < 0.0
    if not solid.any():
        raise ValueError(
            f"{shape} at soft_resolution={resolution} voxelises to nothing; "
            f"raise the resolution")
    return _make_well_composed(solid), cell, dims


def _tetrahedra(solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertex_grid_coords, tets)`` for a solid voxel mask."""
    nx, ny, nz = solid.shape
    cells = np.argwhere(solid)
    used = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                used[cells[:, 0] + dx, cells[:, 1] + dy, cells[:, 2] + dz] = True

    vertex_id = np.full(used.shape, -1, dtype=np.int64)
    coords = np.argwhere(used)
    vertex_id[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(coords.shape[0])

    parity = (cells.sum(axis=1) & 1).astype(bool)
    tets = np.empty((cells.shape[0], 5, 4), dtype=np.int64)
    for split, mask in ((_TET_EVEN, ~parity), (_TET_ODD, parity)):
        block = cells[mask]
        if block.size == 0:
            continue
        for t, corners in enumerate(split):
            for v, (dx, dy, dz) in enumerate(corners):
                tets[mask, t, v] = vertex_id[
                    block[:, 0] + dx, block[:, 1] + dy, block[:, 2] + dz]
    return coords, tets.reshape(-1, 4)


def _surface(tets: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Outward-wound triangles of the faces referenced by exactly one tet."""
    faces = np.concatenate([
        tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]], tets[:, [1, 2, 3]],
    ])
    apex = np.concatenate([tets[:, 3], tets[:, 2], tets[:, 1], tets[:, 0]])

    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = counts[np.ravel(inverse)] == 1
    tri = faces[boundary]
    apex = apex[boundary]

    # Wind each triangle so its normal points away from the tet it came from.
    # Relying on a fixed face ordering instead would silently invert half the
    # surface whenever the positive-volume fix swapped a tet's vertices.
    p0, p1, p2 = positions[tri[:, 0]], positions[tri[:, 1]], positions[tri[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    inward = np.einsum("ij,ij->i", normal, positions[apex] - p0) > 0.0
    tri[inward] = tri[inward][:, [0, 2, 1]]
    return tri


#: Outer bound on how far a boundary vertex may travel, in lattice cells.
#: Purely a guard against a Newton step running away on a bad gradient; the
#: quality backoff below is what normally decides the distance.
_FIT_LIMIT = 0.75
_FIT_ITERATIONS = 4
#: A tetrahedron may not be squashed below this fraction of its lattice
#: volume.  Rounding the skin costs tet quality, and a sliver at rest is a
#: sliver that inverts the moment the body is squeezed: fitting the sphere
#: with a flat 0.45-cell clamp and no check at all left 26 of its 5945
#: tetrahedra inverted after a three-second settle at hardness 0.
_FIT_MIN_VOLUME_RATIO = 0.40
#: ...and it may not lose this fraction of its lattice *shape* either, which
#: is the check that actually matters.  Volume alone cannot tell a small
#: tetrahedron from a flat one, and flat is what inverts: with only the
#: volume floor the shipped sphere ends a three-second settle at hardness 0
#: with 22 of its 12765 tetrahedra inverted -- permanently, still inverted at
#: ten seconds -- every one of them in the ground contact patch, every one of
#: them a skin element the backoff had driven down to the floor.  Dropping
#: the ground away entirely gives 0, so it is contact load on flattened rest
#: elements, not the fit on its own.  Raising the volume floor instead
#: happens to fix the shipped resolution and is not monotone in the floor
#: (0.50 gives 0, 0.60 gives 6), because the backoff ladder is discrete and
#: what protects an element is how far past the floor it overshot.  The
#: shape floor is monotone over the same sweep and costs the same roundness.
#: Measured at hardness 0, three seconds, inverted tetrahedra by resolution
#: 8/13/17/24: volume floor alone 2/4/22/44, with this 0/2/0/2, no fit at all
#: 0/0/0/0.  The price is the rest skin scatter the fit exists to remove:
#: 4.37 mm standard deviation against 2.89 mm for the volume floor alone and
#: 6.60 mm unfitted, so the fit still removes a third of the staircase.
_FIT_MIN_QUALITY_RATIO = 0.88
_FIT_BACKOFF = 0.55
_FIT_BACKOFF_PASSES = 14
#: Laplacian relaxation of the interior after the skin has been pulled in.
_RELAX_ITERATIONS = 24
_RELAX_RATE = 0.55


def _tet_volumes(positions: np.ndarray, tets: np.ndarray) -> np.ndarray:
    edges = positions[tets[:, 1:]] - positions[tets[:, [0]]]
    return np.linalg.det(np.swapaxes(edges, 1, 2)) / 6.0


def _tet_quality(positions: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """``12 (3V)^(2/3) / sum(edge^2)``: 1 for a regular tetrahedron, 0 flat.

    The standard radius-ratio-like shape measure.  It is what distinguishes
    a tetrahedron that is merely small -- which simulates perfectly well --
    from one that has been flattened into a sliver, which is what inverts
    under load.  Scale-free, so the same threshold means the same thing at
    every resolution.
    """
    v = np.abs(_tet_volumes(positions, tets))
    edge_sq = np.zeros(tets.shape[0], dtype=np.float64)
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        d = positions[tets[:, a]] - positions[tets[:, b]]
        edge_sq += np.einsum("ij,ij->i", d, d)
    return 12.0 * np.cbrt((3.0 * v) ** 2) / np.maximum(edge_sq, 1e-300)


def _relax_interior(positions: np.ndarray, tets: np.ndarray,
                    frozen: np.ndarray) -> np.ndarray:
    """Laplacian-smooth the interior vertices with the skin held in place.

    Projecting the skin onto the isosurface squashes the one layer of
    tetrahedra directly behind it, because that layer has to absorb the whole
    correction on its own.  Letting the interior slide spreads the distortion
    through the body, where there is room for it.

    It does not buy extra roundness -- the limit there is the boundary
    tetrahedra that have all four vertices on the skin, which no amount of
    interior motion can help -- but it more than quadruples the worst rest
    volume in the mesh (5.4e-7 against 1.2e-7 m^3 on the shipped sphere), and
    that margin is what the body spends when a hand squeezes it.
    """
    count = positions.shape[0]
    if count == 0 or tets.shape[0] == 0:
        return positions
    pairs = np.concatenate([
        tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
        tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
    ])
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    degree = np.bincount(src, minlength=count).astype(np.float64)
    degree[degree == 0.0] = 1.0

    movable = np.ones(count, dtype=bool)
    movable[frozen] = False
    out = positions.copy()
    for _ in range(_RELAX_ITERATIONS):
        total = np.zeros_like(out)
        np.add.at(total, src, out[dst])
        target = total / degree[:, None]
        step = (target - out) * _RELAX_RATE
        out[movable] += step[movable]
    return out


def _project(points: np.ndarray, sdf, eps: float, steps: int) -> np.ndarray:
    """Newton-step ``points`` onto the ``sdf`` zero set."""
    p = points
    for _ in range(steps):
        d = sdf(p)
        grad = np.empty_like(p)
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = eps
            grad[:, axis] = (sdf(p + step) - sdf(p - step)) / (2.0 * eps)
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        # A vertex on a gradient singularity -- the axis of the torus, the
        # centre of a box -- has no surface direction to move along.
        p = p - np.divide(grad, np.maximum(norm, 1e-9)) * d[:, None]
    return p


#: Elements each skin vertex is blended over.  A lattice vertex touches up
#: to twenty tetrahedra; the eight most interior are enough that no single
#: element's shear shows through as a crease.
SKIN_BLEND = 8


def _skin_binding(lattice: np.ndarray, tets: np.ndarray, boundary: np.ndarray,
                  target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bind each skin vertex's ideal position to its incident tetrahedra.

    Returns ``(skin_tet, skin_bary)`` of shapes ``(P, K)`` and ``(P, K, 4)``:
    up to ``K = SKIN_BLEND`` elements per particle, each carrying barycentric
    weights already scaled by ``1/count`` so that the drawn position is the
    plain sum over every ``(element, corner)`` pair.  Unused slots hold
    ``-1`` and zero weights; interior particles have no slots at all and are
    drawn where they are.

    One element per vertex places the vertex exactly and follows that
    element's deformation exactly -- and shows the seam where the neighbouring
    element deforms differently, as an orange-peel crease across every
    boundary of a squeezed body.  Averaging the affine maps of the vertex's
    incident elements is linear blend skinning: still exact at rest (each map
    reproduces the ideal point on its own, so their mean does too) and
    continuous in the deformation gradient across element boundaries, which
    is what removes the crease.  The elements are ranked by how interior the
    ideal point is to them, so the extrapolated maps -- the ones with the
    longest lever arm on element noise -- are the first dropped.
    """
    count = lattice.shape[0]
    K = SKIN_BLEND
    skin_tet = np.full((count, K), -1, np.int32)
    skin_bary = np.zeros((count, K, 4), np.float32)
    if boundary.size == 0 or tets.shape[0] == 0:
        return skin_tet, skin_bary

    dm = np.swapaxes(lattice[tets[:, 1:]] - lattice[tets[:, [0]]], 1, 2)
    dm_inv = np.linalg.inv(dm)

    on_skin = np.zeros(count, bool)
    on_skin[boundary] = True
    vert = tets.reshape(-1)
    tet_of = np.repeat(np.arange(tets.shape[0]), 4)
    keep = on_skin[vert]
    vert, tet_of = vert[keep], tet_of[keep]

    rel = target[vert] - lattice[tets[tet_of, 0]]
    uvw = np.einsum("nij,nj->ni", dm_inv[tet_of], rel)
    bary = np.concatenate([1.0 - uvw.sum(axis=1, keepdims=True), uvw], axis=1)
    score = bary.min(axis=1)

    order = np.lexsort((-score, vert))
    vert, tet_of, bary = vert[order], tet_of[order], bary[order]
    # Rank inside each vertex's run: 0 for its most interior element, then
    # 1, and so on.  Everything ranked K or later is dropped.
    new_run = np.r_[True, vert[1:] != vert[:-1]]
    starts = np.flatnonzero(new_run)
    run_id = np.cumsum(new_run) - 1
    rank = np.arange(vert.size) - starts[run_id]
    keep = rank < K
    vert, tet_of, bary, rank = vert[keep], tet_of[keep], bary[keep], rank[keep]
    per_vertex = np.bincount(vert, minlength=count).astype(np.float64)

    skin_tet[vert, rank] = tet_of.astype(np.int32)
    skin_bary[vert, rank] = (bary / per_vertex[vert][:, None]).astype(np.float32)
    return skin_tet, skin_bary


def _boundary_edges(tets: np.ndarray, slot: np.ndarray, count: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tet edges with both ends on the skin, in boundary-slot indices."""
    pairs = np.concatenate([
        tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
        tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
    ])
    a, b = slot[pairs[:, 0]], slot[pairs[:, 1]]
    keep = (a >= 0) & (b >= 0)
    src = np.concatenate([a[keep], b[keep]])
    dst = np.concatenate([b[keep], a[keep]])
    degree = np.bincount(src, minlength=count).astype(np.float64)
    degree[degree == 0.0] = 1.0
    return src, dst, degree


def _fit_to_surface(lattice: np.ndarray, boundary: np.ndarray,
                    tets: np.ndarray, shape: str,
                    cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Pull the boundary vertices of the voxel lattice onto the real surface.

    A lattice that is only voxelised has a staircase for a skin, and at the
    resolutions this runs at -- 13 cells across a 0.26 m body -- it does not
    read as a sphere, it reads as Minecraft.  Smoothing the shading normals
    does not help, because the facets are genuinely flat.  So the boundary
    vertices are moved onto the isosurface with a few Newton steps along the
    gradient, *before* ``Dm_inv`` and the rest volumes are computed.  The rest
    pose therefore stays self-consistent: the body is not a distorted cube, it
    is an undistorted sphere.

    Each vertex then backs off until every tetrahedron touching it still has a
    decent share of its lattice volume *and* of its lattice shape.  A single
    global distance clamp cannot do this job -- the displacement a vertex can
    afford depends on which cells happen to sit behind it, so one clamp is
    either too timid on the flats or too greedy in the corners.
    """
    out = lattice.copy()
    if boundary.size == 0 or tets.shape[0] == 0:
        return out, out.copy()

    sdf, _ = _SDFS[shape]
    origin = lattice[boundary]
    eps = cell * 0.05
    p = _project(origin.copy(), sdf, eps, _FIT_ITERATIONS)

    offset = p - origin
    # A box's lattice vertices are already exactly on its isosurface at every
    # resolution, so the whole fit is fourteen relaxation passes that arrive
    # back where they started -- 77 ms of startup at the shipped resolution
    # and 3.6 s at 40.  Returning the lattice is also more exact than
    # returning a copy that has been through the float arithmetic.
    if not np.any(np.abs(offset) > 1e-9 * cell):
        return out, out.copy()
    dist = np.linalg.norm(offset, axis=1, keepdims=True)
    offset *= np.minimum(1.0, (_FIT_LIMIT * cell) / np.maximum(dist, 1e-12))

    # Where the skin would be if the tetrahedra could take it. The physics
    # lattice will only get part of the way; the render skin is bound to this.
    ideal = lattice.copy()
    ideal[boundary] = origin + offset

    # Map each boundary vertex to its slot, so a bad tetrahedron can name the
    # vertices that have to give way.
    slot = np.full(lattice.shape[0], -1, dtype=np.int64)
    slot[boundary] = np.arange(boundary.size)

    src, dst, degree = _boundary_edges(tets, slot, boundary.size)
    volume_floor = _FIT_MIN_VOLUME_RATIO * np.abs(_tet_volumes(lattice, tets))
    quality_floor = _FIT_MIN_QUALITY_RATIO * _tet_quality(lattice, tets)
    scale = np.ones(boundary.size)
    for _ in range(_FIT_BACKOFF_PASSES):
        moved = lattice.copy()
        moved[boundary] = origin + offset * scale[:, None]
        candidate = _relax_interior(moved, tets, boundary)
        bad = ((np.abs(_tet_volumes(candidate, tets)) < volume_floor)
               | (_tet_quality(candidate, tets) < quality_floor))
        if not bad.any():
            return candidate, ideal
        touched = slot[np.unique(tets[bad])]
        scale[touched[touched >= 0]] *= _FIT_BACKOFF
        # Spread the backoff over the skin.  Held back on its own, a vertex
        # stays at its staircase position while its neighbours travel to the
        # surface, and the dimple that leaves is deep enough to self-shadow:
        # it shows up as a scatter of black notches all over an otherwise
        # clean body.  Smoothing the scale field costs a little roundness and
        # turns those notches into a gentle undulation.  The min keeps it a
        # backoff: smoothing may only ever hold a vertex back further.
        if src.size:
            total = np.zeros_like(scale)
            np.add.at(total, src, scale[dst])
            scale = np.minimum(scale, 0.5 * scale + 0.5 * total / degree)
    # Every boundary vertex ran out of room.  A staircase that simulates is
    # better than a sphere that inverts, so give up on moving the lattice --
    # but the render skin is not load bearing, so it still goes all the way.
    return lattice, ideal


def build_soft_body(scene: SceneConfig, mp: MaterialParams) -> BodyData:
    """Build the tetrahedralised soft body described by ``scene``."""
    shape = str(scene.soft_shape)
    resolution = int(scene.soft_resolution)
    size = float(scene.soft_size)
    if not size > 0.0:
        raise ValueError(f"soft_size must be positive, got {size}")

    solid, cell, dims = _voxelise(shape, resolution)
    coords, tets = _tetrahedra(solid)

    # The shape lives in a box whose longest axis is 2 units across.
    scale = 0.5 * size
    centre = np.array([0.0, float(scene.soft_height), 0.0], dtype=np.float64)
    lattice = _lattice_axis(2 * coords, dims, cell)
    surface_vertices = np.unique(_surface(tets, lattice))
    lattice, skin_target = _fit_to_surface(
        lattice, surface_vertices, tets, shape, cell)
    # Re-centre on what was actually built.  The voxel lattice is exactly
    # symmetric, but the fit's quality backoff is not: it reads tetrahedron
    # volumes, and the parity-alternating split gives a cell and its mirror
    # different tetrahedra.  The drift is a tenth of a millimetre, far too
    # small to see and quite large enough to make a torus roll.
    recentre = 0.5 * (lattice.min(axis=0) + lattice.max(axis=0))
    lattice -= recentre
    skin_target -= recentre
    positions = lattice * scale + centre
    num_particles = positions.shape[0]

    # Dm has x1-x0, x2-x0, x3-x0 as COLUMNS (ARCHITECTURE.md 6.4), so that
    # F = Ds @ Dm_inv.  Stacking them as rows instead transposes Dm_inv, which
    # leaves volume and uniform scaling right and every shear response wrong --
    # the body then looks plausible until you twist it.
    edges = positions[tets[:, 1:]] - positions[tets[:, [0]]]
    dm = np.swapaxes(edges, 1, 2)
    det = np.linalg.det(dm)

    flipped = det < 0.0
    if flipped.any():
        tets[flipped] = tets[flipped][:, [0, 2, 1, 3]]
        edges = positions[tets[:, 1:]] - positions[tets[:, [0]]]
        dm = np.swapaxes(edges, 1, 2)
        det = np.linalg.det(dm)
    if not np.all(det > 0.0):
        raise RuntimeError(
            f"{int((det <= 0.0).sum())} tetrahedra are degenerate after the "
            f"orientation fix; the lattice is broken")

    dm_inv = np.linalg.inv(dm)
    rest_volume = det / 6.0

    # After the winding fix, not before: flipping a tetrahedron swaps two of
    # its vertices, and barycentric weights bound to the old order then name
    # the wrong corners.  That is a silent failure -- the skin still moves
    # with the body, just wrongly, and it looks like a bad fit rather than a
    # bad index.
    skin_tet, skin_bary = _skin_binding(lattice, tets, surface_vertices,
                                        skin_target)

    # --- distance constraints on every unique tet edge ---------------------
    pairs = np.concatenate([
        tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
        tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
    ])
    dist_idx = np.unique(np.sort(pairs, axis=1), axis=0)
    dist_rest = np.linalg.norm(
        positions[dist_idx[:, 0]] - positions[dist_idx[:, 1]], axis=1)

    # --- mass ---------------------------------------------------------------
    owned = np.bincount(
        tets.ravel(), weights=np.repeat(rest_volume * 0.25, 4),
        minlength=num_particles)
    if not np.all(owned > 0.0):
        raise RuntimeError("a soft-body particle belongs to no tetrahedron")
    # volumetric_particle_mass floors the result so a degenerate element cannot
    # produce an infinite inverse mass.  Below that floor the body stops
    # weighing what its density says -- a 0.02 m sphere comes out 23% heavy and
    # a 0.005 m one 46 times heavy -- and the hardness dial's claim to run on
    # real material constants goes with it.  Silently is the one way this must
    # not fail.
    floor = volumetric_particle_mass(mp, 0.0)
    lightest = volumetric_particle_mass(mp, float(owned.min()))
    if lightest <= floor:
        raise ValueError(
            f"{shape} at soft_size={size} m, soft_resolution={resolution} gives "
            f"its smallest particle {owned.min():.3g} m^3, which at "
            f"{mp.density} kg/m^3 is at or under the {floor:.3g} kg floor in "
            f"volumetric_particle_mass; the body would come out heavier than "
            f"its density. Raise soft_size or lower soft_resolution.")
    mass = np.array([volumetric_particle_mass(mp, float(v)) for v in owned])

    tri_idx = _surface(tets, positions)

    flags = np.zeros(num_particles, dtype=np.uint32)
    # Interior particles never reach another body's surface before the shell
    # does, so including them in the broad phase is pure cost.
    flags[np.unique(tri_idx)] = FLAG_SURFACE | FLAG_SELF_COLLIDE

    offset = positions - centre
    radius = np.maximum(np.linalg.norm(offset, axis=1), 1e-9)
    uv = np.stack([
        np.arctan2(offset[:, 2], offset[:, 0]) / (2.0 * np.pi) + 0.5,
        np.arccos(np.clip(offset[:, 1] / radius, -1.0, 1.0)) / np.pi,
    ], axis=1)

    body = BodyData(
        kind=MatterKind.SOFT,
        name=f"soft-{shape}-{resolution}",
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        inv_mass=np.ascontiguousarray(1.0 / mass, dtype=np.float32),
        flags=np.ascontiguousarray(flags, dtype=np.uint32),
        dist_idx=np.ascontiguousarray(dist_idx, dtype=np.int32),
        dist_rest=np.ascontiguousarray(dist_rest, dtype=np.float32),
        dist_kind=np.full(dist_idx.shape[0], int(ConstraintKind.STRETCH), np.int32),
        dist_color=greedy_color(dist_idx.astype(np.int32), num_particles),
        bend_idx=np.zeros((0, 4), dtype=np.int32),
        bend_rest=np.zeros(0, dtype=np.float32),
        bend_color=np.zeros(0, dtype=np.int32),
        tet_idx=np.ascontiguousarray(tets, dtype=np.int32),
        tet_dm_inv=np.ascontiguousarray(dm_inv, dtype=np.float32),
        tet_rest_volume=np.ascontiguousarray(rest_volume, dtype=np.float32),
        tet_color=greedy_color(tets.astype(np.int32), num_particles),
        tri_idx=np.ascontiguousarray(tri_idx, dtype=np.int32),
        uv=np.ascontiguousarray(uv, dtype=np.float32),
        double_sided=False,
        particle_radius=0.5 * cell * scale,
        skin_tet=np.ascontiguousarray(skin_tet, dtype=np.int32),
        skin_bary=np.ascontiguousarray(skin_bary, dtype=np.float32),
    )
    body.validate()
    return body
