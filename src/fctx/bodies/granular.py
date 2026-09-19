"""Granular matter: a jittered close packing of unconstrained grains.

There are no constraints at all.  Everything a pile of sand does -- the angle
of repose, the way it parts around a hand and slumps back -- comes out of
particle-particle contact and friction in the solver.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import SceneConfig, SolverConfig
from ..core.material import MaterialParams, grain_particle_mass
from ..core.types import FLAG_SELF_COLLIDE, BodyData, MatterKind

__all__ = ["build_granular"]

#: Jitter as a fraction of the grain radius.  A perfect lattice drains like a
#: crystal: it shears along its own glide planes instead of flowing.
_JITTER_FRACTION = 0.06
#: Clearance left between the closest possible pair and the distance at which
#: the solver first calls them a contact, as a fraction of the diameter.
_CONTACT_CLEARANCE = 0.03
#: Lattice spacing as a multiple of the grain diameter.
#:
#: Geometric non-overlap is not the bar.  The solver starts pushing two
#: particles apart at ``(r_i + r_j) * (1 + collision_margin)``, so the pack has
#: to begin outside *that*, not merely outside contact.  Two neighbours can
#: each be displaced by the jitter along all three axes at once, so the gap
#: closes by ``2*sqrt(3)`` jitters in the worst case, not by two, which puts
#: the minimum separation at ``2r * (slack - sqrt(3) * jitter)``.
#:
#: Sized by hand this went wrong quietly: a slack of 1.12 against a margin of
#: 0.12 put the nominal spacing exactly on the contact threshold, so 66792 of
#: the 24000-grain pile's pairs were already in contact on frame 0, the first
#: substep tried to resolve all of them at once, and the pile left the stage at
#: the 12 m/s velocity clamp.  Deriving it keeps the two numbers in step.
_SPACING_SLACK = (1.0 + SolverConfig().collision_margin
                  + math.sqrt(3.0) * _JITTER_FRACTION + _CONTACT_CLEARANCE)
_SEED = 0x6A5EED


def build_granular(scene: SceneConfig, mp: MaterialParams) -> BodyData:
    """Build the grain pile described by ``scene``."""
    count = int(scene.grain_count)
    radius = float(scene.grain_radius)
    extent = float(scene.grain_extent)
    if count < 1:
        raise ValueError(f"grain_count must be positive, got {count}")
    if not radius > 0.0:
        raise ValueError(f"grain_radius must be positive, got {radius}")
    if not extent > radius:
        raise ValueError(
            f"grain_extent {extent} leaves no room for a grain of radius {radius}")
    # grain_particle_mass floors its result so a zero radius cannot produce an
    # infinite inverse mass.  Under that floor every grain weighs the same
    # whatever its radius, so the pile stops responding to grain_radius at all
    # and a 0.1 mm grain comes out 27 times heavier than its density allows.
    mass = grain_particle_mass(mp, radius)
    floor = grain_particle_mass(mp, 0.0)
    if mass <= floor:
        raise ValueError(
            f"a grain of radius {radius} m at {mp.density} kg/m^3 weighs at or "
            f"under the {floor:.3g} kg floor in grain_particle_mass, so the "
            f"pile would come out heavier than its density. Raise grain_radius.")

    d = 2.0 * radius * _SPACING_SLACK
    row_dz = d * math.sqrt(3.0) / 2.0       # triangular rows within a layer
    layer_dy = d * math.sqrt(2.0 / 3.0)     # close-packed layer separation
    hollow_dz = row_dz / 3.0                # offset that drops a layer into
    hollow_dx = d / 2.0                     # the hollows of the one below

    # Centres must stay a radius plus a jitter inside the box, and the row and
    # layer offsets eat into the usable span before the lattice even starts.
    jitter = _JITTER_FRACTION * radius
    margin = 2.0 * (extent - radius - jitter)
    usable_x = margin - hollow_dx * 2.0
    usable_z = margin - hollow_dz
    nx = int(usable_x // d) + 1
    nz = int(usable_z // row_dz) + 1
    if usable_x < 0.0 or usable_z < 0.0 or nx < 1 or nz < 1:
        raise ValueError(
            f"a box of half-extent {extent} holds no grain lattice at radius "
            f"{radius}")

    per_layer = nx * nz
    layers = -(-count // per_layer)
    height = (layers - 1) * layer_dy + 2.0 * (radius + jitter)
    if height > 2.0 * extent:
        raise ValueError(
            f"{count} grains of radius {radius} need {layers} layers "
            f"({height:.3f} m) but the box is only {2.0 * extent:.3f} m tall")

    ix, iz, iy = np.meshgrid(
        np.arange(nx), np.arange(nz), np.arange(layers), indexing="ij")
    ix, iz, iy = ix.ravel(), iz.ravel(), iy.ravel()

    x = ix * d + (iz & 1) * hollow_dx + (iy & 1) * hollow_dx
    z = iz * row_dz + (iy & 1) * hollow_dz
    y = iy * layer_dy
    sites = np.stack([x, y, z], axis=1).astype(np.float64)

    rng = np.random.default_rng(_SEED)
    # Full layers from the bottom up, then a seeded scatter across the top one
    # so the pile has a broken surface instead of a raster-scan ledge.
    surplus = per_layer * layers - count
    if surplus > 0:
        top = np.flatnonzero(iy == layers - 1)
        drop = rng.choice(top, size=surplus, replace=False)
        keep = np.ones(sites.shape[0], dtype=bool)
        keep[drop] = False
        sites = sites[keep]

    sites += rng.uniform(-jitter, jitter, sites.shape)
    sites -= 0.5 * (sites.min(axis=0) + sites.max(axis=0))
    sites[:, 1] += float(scene.grain_height)

    overflow = np.abs(sites - np.array([0.0, scene.grain_height, 0.0])).max() + radius
    if overflow > extent + 1e-9:
        raise RuntimeError(
            f"grain packing escaped its box by {overflow - extent:.4g} m")

    num_particles = sites.shape[0]
    if num_particles != count:
        raise RuntimeError(
            f"asked for {count} grains, packed {num_particles}")

    span = 2.0 * extent
    uv = np.stack([
        (sites[:, 0] + extent) / span,
        (sites[:, 2] + extent) / span,
    ], axis=1)

    body = BodyData(
        kind=MatterKind.GRAIN,
        name=f"grain-{count}",
        positions=np.ascontiguousarray(sites, dtype=np.float32),
        inv_mass=np.full(num_particles, 1.0 / mass, dtype=np.float32),
        flags=np.full(num_particles, FLAG_SELF_COLLIDE, dtype=np.uint32),
        dist_idx=np.zeros((0, 2), dtype=np.int32),
        dist_rest=np.zeros(0, dtype=np.float32),
        dist_kind=np.zeros(0, dtype=np.int32),
        dist_color=np.zeros(0, dtype=np.int32),
        bend_idx=np.zeros((0, 4), dtype=np.int32),
        bend_rest=np.zeros(0, dtype=np.float32),
        bend_color=np.zeros(0, dtype=np.int32),
        tet_idx=np.zeros((0, 4), dtype=np.int32),
        tet_dm_inv=np.zeros((0, 3, 3), dtype=np.float32),
        tet_rest_volume=np.zeros(0, dtype=np.float32),
        tet_color=np.zeros(0, dtype=np.int32),
        tri_idx=np.zeros((0, 3), dtype=np.int32),
        uv=np.ascontiguousarray(uv, dtype=np.float32),
        double_sided=False,
        particle_radius=radius,
    )
    body.validate()
    return body
