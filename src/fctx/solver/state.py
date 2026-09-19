"""Device-side storage for the whole scene.

One ``SolverState`` owns every GPU array listed in ARCHITECTURE.md section 5.
All bodies share a single global particle buffer -- cloth, soft bodies and
grains are the same particles with different constraints on them -- which is
what lets the renderer bind one VBO pair for the entire scene and what lets a
grab attach to anything without asking what it is made of.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..config import AppConfig
from ..core.types import BodyData
from . import kernels as K

if TYPE_CHECKING:
    from ..core.material import Material

__all__ = ["SolverState"]

#: How far outside ``SolverConfig.world_extent`` a particle may stray before
#: the solver calls it diverged.  See :attr:`SolverState.contain_limit`.
CONTAINMENT_FACTOR = 4.0

#: Fraction of its shortest bonded edge that a particle's contact sphere may
#: reach.  See :meth:`SolverState._collision_radii`.
BONDED_EDGE_FRACTION = 0.25


def _sort_by_color(color: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return a permutation grouping ``color`` into runs, plus the runs.

    ``kind="stable"`` matters: it keeps constraints in the order the builders
    emitted them inside each colour, so two runs of the same scene produce
    byte-identical buffers and a regression in the physics cannot hide behind
    a reshuffle.
    """
    order = np.argsort(color, kind="stable").astype(np.int64)
    if order.size == 0:
        return order, []
    sorted_color = color[order]
    starts = np.flatnonzero(np.r_[True, sorted_color[1:] != sorted_color[:-1]])
    ends = np.r_[starts[1:], sorted_color.size]
    return order, [(int(s), int(e - s)) for s, e in zip(starts, ends)]


class SolverState:
    """Every device array the solver reads or writes.

    Parameters
    ----------
    bodies:
        Bodies in render order.  Their local particle indices are offset into
        one global buffer; ``body_particle_range`` maps back.
    cfg:
        Capacities for hands and grabs come from here, as do the tunables the
        kernels read out of the parameter arrays.
    device:
        A Warp device string, e.g. ``"cuda:0"``.
    """

    def __init__(self, bodies: list[BodyData], cfg: AppConfig, device: str) -> None:
        if not bodies:
            raise ValueError("SolverState needs at least one body")
        for body in bodies:
            body.validate()

        self.cfg = cfg
        self.device = device
        self.bodies = list(bodies)

        counts = np.array([b.num_particles for b in bodies], dtype=np.int64)
        self._particle_offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        num_particles = int(self._particle_offsets[-1])
        num_bodies = len(bodies)

        pos = np.concatenate([b.positions for b in bodies], axis=0).astype(np.float32)
        inv_mass = np.concatenate([b.inv_mass for b in bodies]).astype(np.float32)
        flags = np.concatenate([b.flags for b in bodies]).astype(np.uint32)
        body_of = np.concatenate(
            [np.full(b.num_particles, i, np.int32) for i, b in enumerate(bodies)])
        radius = np.concatenate(
            [np.full(b.num_particles, b.particle_radius, np.float32)
             for b in bodies])
        uv = np.concatenate([b.uv for b in bodies], axis=0).astype(np.float32)

        dist_idx, dist_rest, dist_kind, dist_body, dist_color = self._gather_dist(bodies)
        bend_idx, bend_rest, bend_body, bend_color = self._gather_bend(bodies)
        tet_idx, tet_dm, tet_vol, tet_body, tet_color = self._gather_tet(bodies)
        tri_idx, self._tri_offsets = self._gather_tris(bodies)

        d_order, self.dist_batches = _sort_by_color(dist_color)
        b_order, self.bend_batches = _sort_by_color(bend_color)
        t_order, self.tet_batches = _sort_by_color(tet_color)

        dist_idx = dist_idx[d_order]
        dist_rest = dist_rest[d_order]
        dist_kind = dist_kind[d_order]
        dist_body = dist_body[d_order]
        bend_idx = bend_idx[b_order]
        bend_rest = bend_rest[b_order]
        bend_body = bend_body[b_order]
        tet_idx = tet_idx[t_order]
        tet_dm = tet_dm[t_order]
        tet_vol = tet_vol[t_order]
        tet_body = tet_body[t_order]

        skin_tet, skin_bary = self._gather_skin(bodies, t_order)

        self.num_dist = int(dist_idx.shape[0])
        self.num_bend = int(bend_idx.shape[0])
        self.num_tet = int(tet_idx.shape[0])
        self.num_tri = int(tri_idx.shape[0])
        self._num_particles = num_particles
        self._num_bodies = num_bodies
        self._tet_wsum_host = self._tet_rest_wsum(tet_idx, tet_dm, inv_mass)
        self._tet_vol_host = tet_vol.astype(np.float64)
        self._tet_body_host = tet_body.astype(np.int64)

        self.max_hands = int(cfg.tracking.max_hands)
        if self.max_hands < 1:
            raise ValueError("TrackingConfig.max_hands must be at least 1")
        self.bones_per_hand = 21
        self.capsule_capacity = self.max_hands * self.bones_per_hand
        self.grab_per_hand = int(cfg.grab.max_particles)
        self.grab_capacity = self.max_hands * self.grab_per_hand

        collide_radius = self._collision_radii(radius, dist_idx, dist_rest)
        self.grid_radius = float(
            max(2.0 * collide_radius.max() * (1.0 + cfg.solver.collision_margin),
                1.0e-4))

        #: Distance from the origin past which a particle is treated as
        #: diverged.  The stage is under a metre across, so any multiple of
        #: ``world_extent`` is already unreachable by honest motion; the point
        #: of the bound is to stay far under the coordinate at which the hash
        #: grid's cell index overflows, which is ``2^31 * grid_radius`` and
        #: which the grid does not check.
        self.contain_limit = float(
            max(cfg.solver.world_extent, 1.0) * CONTAINMENT_FACTOR)
        grid_overflow = 2.0 ** 31 * self.grid_radius
        if self.contain_limit * 8.0 >= grid_overflow:
            raise ValueError(
                f"containment limit {self.contain_limit:g} m is too close to "
                f"the hash grid's index overflow at {grid_overflow:g} m; "
                f"raise SolverConfig.collision_margin or lower world_extent")

        # Only the flags need a host copy for reset: x_rest and w_rest are
        # already on the device and reset copies them there.
        self._rest_flags = flags.copy()
        self._triangles = tri_idx
        self.uv = uv

        d = device
        self.x = wp.array(pos, dtype=wp.vec3, device=d)
        self.x_prev = wp.array(pos, dtype=wp.vec3, device=d)
        self.x_rest = wp.array(pos, dtype=wp.vec3, device=d)
        self.v = wp.zeros(num_particles, dtype=wp.vec3, device=d)
        self.w = wp.array(inv_mass, dtype=float, device=d)
        self.w_rest = wp.array(inv_mass, dtype=float, device=d)
        self.flags = wp.array(flags, dtype=wp.uint32, device=d)
        self.body = wp.array(body_of, dtype=wp.int32, device=d)
        self.radius = wp.array(radius, dtype=float, device=d)
        self.collide_radius = wp.array(collide_radius, dtype=float, device=d)
        self.normal = wp.zeros(num_particles, dtype=wp.vec3, device=d)
        self.contact_n = wp.zeros(num_particles, dtype=wp.vec3, device=d)
        self.contact_vn = wp.zeros(num_particles, dtype=float, device=d)
        self.contact_dx = wp.zeros(num_particles, dtype=wp.vec3, device=d)

        self.dist_idx = self._vec_array(dist_idx, wp.vec2i, 2, np.int32)
        self.dist_rest = self._num_array(dist_rest, float, np.float32)
        self.dist_kind = self._num_array(dist_kind, wp.int32, np.int32)
        self.dist_body = self._num_array(dist_body, wp.int32, np.int32)
        self.dist_lambda = wp.zeros(max(self.num_dist, 1), dtype=float, device=d)

        self.bend_idx = self._vec_array(bend_idx, wp.vec4i, 4, np.int32)
        self.bend_rest = self._num_array(bend_rest, float, np.float32)
        self.bend_body = self._num_array(bend_body, wp.int32, np.int32)
        self.bend_lambda = wp.zeros(max(self.num_bend, 1), dtype=float, device=d)

        self.tet_idx = self._vec_array(tet_idx, wp.vec4i, 4, np.int32)
        self.tet_dm_inv = wp.array(
            tet_dm if self.num_tet else np.zeros((1, 3, 3), np.float32),
            dtype=wp.mat33, device=d)
        self.tet_volume = self._num_array(tet_vol, float, np.float32)
        self.tet_body = self._num_array(tet_body, wp.int32, np.int32)
        self.tet_wsum_rest = self._num_array(
            self._tet_wsum_host, float, np.float32)
        self.tet_lambda_d = wp.zeros(max(self.num_tet, 1), dtype=float, device=d)
        self.tet_lambda_h = wp.zeros(max(self.num_tet, 1), dtype=float, device=d)

        self.tri_idx = self._vec_array(tri_idx, wp.vec3i, 3, np.int32)

        # The drawn surface of a soft body is not the physics lattice; see
        # BodyData.skin_tet.  When nothing in the scene uses one, x_skin is
        # x itself and the skinning kernel never runs.
        self.has_skin = bool((skin_tet >= 0).any())
        self.skin_tet = self._num_array(skin_tet, wp.int32, np.int32)
        self.skin_bary = self._vec_array(skin_bary, wp.vec4, 4, np.float32)
        self.x_skin = (wp.zeros(num_particles, dtype=wp.vec3, device=d)
                       if self.has_skin else self.x)

        self.mat_stretch = wp.zeros(num_bodies, dtype=float, device=d)
        self.mat_shear = wp.zeros(num_bodies, dtype=float, device=d)
        self.mat_bend = wp.zeros(num_bodies, dtype=float, device=d)
        self.mat_dev = wp.zeros(num_bodies, dtype=float, device=d)
        self.mat_hyd = wp.zeros(num_bodies, dtype=float, device=d)
        self.mat_grab = wp.zeros(num_bodies, dtype=float, device=d)
        self.mat_damping = wp.zeros(num_bodies, dtype=float, device=d)
        self.mat_friction = wp.zeros(num_bodies, dtype=float, device=d)
        self.mat_restitution = wp.zeros(num_bodies, dtype=float, device=d)

        c = self.capsule_capacity
        self.cap_a = wp.zeros(c, dtype=wp.vec3, device=d)
        self.cap_b = wp.zeros(c, dtype=wp.vec3, device=d)
        # Where the same capsules were at the start of this frame.  Hands
        # arrive once per rendered frame but the solver takes twelve substeps
        # per frame, so a capsule held still for the whole frame and then
        # teleported to its new place hands the matter a correction the size
        # of a whole frame's hand motion inside one substep, which finalize
        # reads as a velocity twelve times too large.  Sweeping the capsule
        # between the two instead costs two arrays of 42 vectors.
        self.cap_a_prev = wp.zeros(c, dtype=wp.vec3, device=d)
        self.cap_b_prev = wp.zeros(c, dtype=wp.vec3, device=d)
        #: ...and the radius they had, for the same reason: a collider fading
        #: in grows by millimetres per frame, and a millimetre applied inside
        #: one substep is a metre per second handed to whatever it grew into,
        #: which is the detonation the fade exists to prevent.
        self.cap_r_prev = wp.zeros(c, dtype=float, device=d)
        self.cap_r = wp.zeros(c, dtype=float, device=d)
        self.cap_va = wp.zeros(c, dtype=wp.vec3, device=d)
        self.cap_vb = wp.zeros(c, dtype=wp.vec3, device=d)
        self.cap_count = wp.zeros(1, dtype=wp.int32, device=d)

        g = self.grab_capacity
        self.grab_particle = wp.array(np.full(g, -1, np.int32), dtype=wp.int32, device=d)
        self.grab_local = wp.zeros(g, dtype=wp.vec3, device=d)
        self.grab_hand = wp.array(np.full(g, -1, np.int32), dtype=wp.int32, device=d)
        self.grab_lambda = wp.zeros(g, dtype=float, device=d)
        self.grab_count = wp.zeros(1, dtype=wp.int32, device=d)

        hcap = self.max_hands
        self.hand_pos = wp.zeros(hcap, dtype=wp.vec3, device=d)
        self.hand_rot = wp.array(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (hcap, 1)),
            dtype=wp.quat, device=d)
        self.hand_vel = wp.zeros(hcap, dtype=wp.vec3, device=d)
        self.hand_active = wp.zeros(hcap, dtype=wp.int32, device=d)

        self.contact_count = wp.zeros(1, dtype=wp.int32, device=d)
        self.body_bad = wp.zeros(num_bodies, dtype=wp.int32, device=d)

        self.params = wp.zeros(K.NUM_PARAMS, dtype=float, device=d)
        self.vparams = wp.zeros(K.NUM_VPARAMS, dtype=wp.vec3, device=d)
        self.upload_params()

        self.grid = wp.HashGrid(cfg.solver.grid_dim, cfg.solver.grid_dim,
                                cfg.solver.grid_dim, device=d)

        self._render_positions: wp.array | None = None
        self._render_normals: wp.array | None = None

    # -- assembly helpers --------------------------------------------------

    def _vec_array(self, data: np.ndarray, dtype: type, width: int,
                   np_dtype: type) -> wp.array:
        if data.shape[0] == 0:
            data = np.zeros((1, width), np_dtype)
        return wp.array(np.ascontiguousarray(data, np_dtype), dtype=dtype,
                        device=self.device)

    def _num_array(self, data: np.ndarray, dtype: object, np_dtype: type) -> wp.array:
        if data.shape[0] == 0:
            data = np.zeros(1, np_dtype)
        return wp.array(np.ascontiguousarray(data, np_dtype), dtype=dtype,
                        device=self.device)

    def _gather_dist(self, bodies: list[BodyData]) -> tuple[np.ndarray, ...]:
        idx, rest, kind, owner, color = [], [], [], [], []
        for i, b in enumerate(bodies):
            if b.num_dist == 0:
                continue
            idx.append(b.dist_idx + int(self._particle_offsets[i]))
            rest.append(b.dist_rest)
            kind.append(b.dist_kind)
            owner.append(np.full(b.num_dist, i, np.int32))
            color.append(b.dist_color)
        if not idx:
            return (np.zeros((0, 2), np.int32), np.zeros(0, np.float32),
                    np.zeros(0, np.int32), np.zeros(0, np.int32),
                    np.zeros(0, np.int32))
        return (np.concatenate(idx).astype(np.int32),
                np.concatenate(rest).astype(np.float32),
                np.concatenate(kind).astype(np.int32),
                np.concatenate(owner).astype(np.int32),
                np.concatenate(color).astype(np.int32))

    def _gather_bend(self, bodies: list[BodyData]) -> tuple[np.ndarray, ...]:
        idx, rest, owner, color = [], [], [], []
        for i, b in enumerate(bodies):
            if b.num_bend == 0:
                continue
            idx.append(b.bend_idx + int(self._particle_offsets[i]))
            rest.append(b.bend_rest)
            owner.append(np.full(b.num_bend, i, np.int32))
            color.append(b.bend_color)
        if not idx:
            return (np.zeros((0, 4), np.int32), np.zeros(0, np.float32),
                    np.zeros(0, np.int32), np.zeros(0, np.int32))
        return (np.concatenate(idx).astype(np.int32),
                np.concatenate(rest).astype(np.float32),
                np.concatenate(owner).astype(np.int32),
                np.concatenate(color).astype(np.int32))

    def _gather_tet(self, bodies: list[BodyData]) -> tuple[np.ndarray, ...]:
        idx, dm, vol, owner, color = [], [], [], [], []
        for i, b in enumerate(bodies):
            if b.num_tets == 0:
                continue
            idx.append(b.tet_idx + int(self._particle_offsets[i]))
            dm.append(b.tet_dm_inv)
            vol.append(b.tet_rest_volume)
            owner.append(np.full(b.num_tets, i, np.int32))
            color.append(b.tet_color)
        if not idx:
            return (np.zeros((0, 4), np.int32), np.zeros((0, 3, 3), np.float32),
                    np.zeros(0, np.float32), np.zeros(0, np.int32),
                    np.zeros(0, np.int32))
        return (np.concatenate(idx).astype(np.int32),
                np.concatenate(dm).astype(np.float32),
                np.concatenate(vol).astype(np.float32),
                np.concatenate(owner).astype(np.int32),
                np.concatenate(color).astype(np.int32))

    def _gather_skin(self, bodies: Sequence[BodyData], t_order: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate the per-body render-skin bindings into scene indices.

        Two offsets have to be applied and both are easy to forget: the
        body's tetrahedron offset, and the colour sort that reorders every
        tetrahedron afterwards.  A skin index that misses either one binds a
        vertex to an unrelated element somewhere else in the scene, which
        looks like a torn mesh rather than a bad index.
        """
        counts = [b.num_tets for b in bodies]
        tet_offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        tet_rank = np.empty(int(tet_offsets[-1]), np.int64)
        tet_rank[t_order] = np.arange(t_order.size)

        widths = [b.skin_tet.shape[1] for b in bodies if b.skin_tet is not None]
        k = max(widths) if widths else 1
        tets, barys = [], []
        for i, b in enumerate(bodies):
            mapped = np.full((b.num_particles, k), -1, np.int64)
            bary = np.zeros((b.num_particles, k, 4), np.float32)
            if b.skin_tet is not None and b.skin_bary is not None:
                w = b.skin_tet.shape[1]
                local = b.skin_tet.astype(np.int64)
                bound = local >= 0
                sub = mapped[:, :w]
                sub[bound] = tet_rank[local[bound] + int(tet_offsets[i])]
                bary[:, :w] = b.skin_bary
            tets.append(mapped.astype(np.int32))
            barys.append(bary)
        self.skin_blend = k
        return (np.ascontiguousarray(np.concatenate(tets).reshape(-1), np.int32),
                np.ascontiguousarray(np.concatenate(barys).reshape(-1, 4),
                                     np.float32))

    def _gather_tris(self, bodies: list[BodyData]) -> tuple[np.ndarray, np.ndarray]:
        tris, offsets = [], [0]
        for i, b in enumerate(bodies):
            if b.num_tris:
                tris.append(b.tri_idx + int(self._particle_offsets[i]))
            offsets.append(offsets[-1] + b.num_tris)
        joined = (np.concatenate(tris).astype(np.int32) if tris
                  else np.zeros((0, 3), np.int32))
        return joined, np.asarray(offsets, np.int64)

    def _collision_radii(self, radius: np.ndarray, dist_idx: np.ndarray,
                         dist_rest: np.ndarray) -> np.ndarray:
        """Cap each particle's contact radius against its shortest bonded edge.

        A cloth particle's render radius is larger than half the grid spacing,
        so used directly it would put every structural neighbour in permanent
        penetration and the contact solver would spend the whole frame fighting
        the distance constraints.

        The fraction has to leave real headroom, not just clear the rest pose.
        Sizing the contact sphere to exactly touch a bonded neighbour at rest
        -- half an edge, less the collision margin -- means any compression at
        all, including the compression a body puts on itself under gravity,
        turns every bond into a standing contact; those contacts and the
        distance constraints then push against each other every substep and
        the body gains energy.  At a quarter of an edge the margin cancels
        out -- ``(r_i + r_j) * (1 + margin)`` is exactly half the edge -- so a
        bond has to shorten by 50% before it registers, which the distance
        constraint it shares will never allow, while a genuine fold brings
        two cells that are not bonded to each other within half a cell.
        """
        crad = radius.copy()
        if dist_idx.shape[0]:
            shortest = np.full(radius.shape[0], np.inf, np.float64)
            np.minimum.at(shortest, dist_idx[:, 0], dist_rest)
            np.minimum.at(shortest, dist_idx[:, 1], dist_rest)
            bonded = np.isfinite(shortest)
            cap = (BONDED_EDGE_FRACTION * shortest[bonded]
                   / (1.0 + self.cfg.solver.collision_margin))
            crad[bonded] = np.minimum(crad[bonded], cap.astype(np.float32))
        return np.maximum(crad, 1.0e-5).astype(np.float32)

    def _tet_rest_wsum(self, tet_idx: np.ndarray, tet_dm: np.ndarray,
                       inv_mass: np.ndarray) -> np.ndarray:
        """``sum_i w_i |grad_i C_H|^2`` for each tetrahedron at ``F = I``.

        This is the mass term the XPBD compliance has to dominate for the
        element's projection to land on a real force rather than on a hard
        position projection; :func:`fctx.solver.kernels.resolvable_softening`
        divides both compliances by whatever it takes to get there.

        At the rest pose ``grad C_H = Dm^-T``, so the per-vertex gradients are
        the rows of ``Dm^-1`` and the fourth is minus their sum.  The
        deviatoric gradients are the same thing over ``sqrt(3)``, which is why
        one array serves both constraints.

        Inverse masses are the *rest* ones: a grab raises w on a small patch
        for as long as it is held, and recomputing this per frame to track that
        would make the material a function of where the hand is.
        """
        if tet_idx.shape[0] == 0:
            return np.zeros(0, np.float32)
        # Rows of Dm^-1, in float64: Dm^-1 is already the inverse of a matrix
        # whose entries are ~1e-2, so its own entries are ~1e2 and squaring
        # them in float32 throws away more than the result can afford.
        g = np.asarray(tet_dm, np.float64)            # (T, 3, 3)
        g0 = -g.sum(axis=1)                           # (T, 3)
        w = np.asarray(inv_mass, np.float64)[tet_idx]  # (T, 4)
        wsum = w[:, 0] * np.einsum("ti,ti->t", g0, g0)
        for j in range(3):
            wsum += w[:, j + 1] * np.einsum("ti,ti->t", g[:, j], g[:, j])
        return wsum.astype(np.float32)

    # -- runtime -----------------------------------------------------------

    def upload_params(self) -> None:
        """Push the scalar tunables from ``cfg`` into the device arrays.

        These live on the device rather than in kernel arguments so that the
        wind toggle and the HUD can change them while a captured CUDA graph is
        replaying; a scalar argument would have been frozen at capture time.
        """
        s = self.cfg.solver
        p = np.zeros(K.NUM_PARAMS, np.float32)
        p[K.PARAM_AIR_DRAG] = s.air_drag
        p[K.PARAM_GROUND_Y] = s.ground_y
        p[K.PARAM_GROUND_FRICTION] = s.ground_friction
        p[K.PARAM_COLLISION_MARGIN] = s.collision_margin
        p[K.PARAM_MAX_CORRECTION_RATIO] = s.max_correction_ratio
        p[K.PARAM_MAX_VELOCITY] = s.max_velocity
        p[K.PARAM_WIND_TURBULENCE] = s.wind_turbulence
        # Simulation time is advanced on the device from inside the captured
        # graph, so it has to survive a parameter rewrite rather than be reset.
        p[K.PARAM_TIME] = float(self.params.numpy()[K.PARAM_TIME])
        p[K.PARAM_SELF_COLLIDE] = 1.0 if s.self_collision else 0.0
        p[K.PARAM_GRAB_ROTATE] = 1.0 if self.cfg.grab.rotate_with_pinch else 0.0
        p[K.PARAM_BASIN_RADIUS] = s.basin_radius
        p[K.PARAM_BASIN_HEIGHT] = s.basin_height
        self.params.assign(p)

        vp = np.zeros((K.NUM_VPARAMS, 3), np.float32)
        vp[K.VPARAM_GRAVITY] = s.gravity
        vp[K.VPARAM_WIND] = s.wind
        self.vparams.assign(vp)

    def set_wind(self, wind: tuple[float, float, float], turbulence: float) -> None:
        vp = self.vparams.numpy()
        vp[K.VPARAM_WIND] = np.asarray(wind, np.float32)
        self.vparams.assign(vp)
        p = self.params.numpy()
        p[K.PARAM_WIND_TURBULENCE] = float(turbulence)
        self.params.assign(p)

    def upload_materials(self, materials: list[Material]) -> None:
        """Rewrite every per-body compliance.

        This is the hardness dial.  It touches nine small device arrays and
        nothing else -- no constraint is rebuilt, no buffer is reallocated and
        any captured graph stays valid, which is what makes the dial movable
        mid-grab.
        """
        if len(materials) != self._num_bodies:
            raise ValueError(
                f"expected {self._num_bodies} materials, got {len(materials)}")
        n = self._num_bodies
        stretch = np.empty(n, np.float32)
        shear = np.empty(n, np.float32)
        bend = np.empty(n, np.float32)
        dev = np.empty(n, np.float32)
        hyd = np.empty(n, np.float32)
        grab = np.empty(n, np.float32)
        damping = np.empty(n, np.float32)
        friction = np.empty(n, np.float32)
        restitution = np.empty(n, np.float32)
        for i, m in enumerate(materials):
            stretch[i] = m.stretch_compliance
            shear[i] = m.shear_compliance
            bend[i] = m.bend_compliance
            dev[i] = m.deviatoric_compliance
            hyd[i] = m.hydrostatic_compliance
            grab[i] = m.grab_compliance
            damping[i] = m.damping
            friction[i] = m.friction
            restitution[i] = m.restitution
        for arr in (stretch, shear, bend, dev, hyd, grab):
            if not np.isfinite(arr).all() or (arr <= 0.0).any():
                raise ValueError("material compliances must be positive and finite")
        self.mat_stretch.assign(stretch)
        self.mat_shear.assign(shear)
        self.mat_bend.assign(bend)
        self.mat_dev.assign(dev)
        self.mat_hyd.assign(hyd)
        self.mat_grab.assign(grab)
        self.mat_damping.assign(damping)
        self.mat_friction.assign(friction)
        self.mat_restitution.assign(restitution)

    def reset(self) -> None:
        """Return every particle to its rest state and drop all grabs."""
        wp.copy(self.x, self.x_rest)
        wp.copy(self.x_prev, self.x_rest)
        wp.copy(self.w, self.w_rest)
        self.v.zero_()
        self.flags.assign(self._rest_flags)
        self.normal.zero_()
        self.contact_n.zero_()
        self.contact_vn.zero_()
        self.contact_dx.zero_()
        self.dist_lambda.zero_()
        self.bend_lambda.zero_()
        self.tet_lambda_d.zero_()
        self.tet_lambda_h.zero_()
        self.grab_lambda.zero_()
        self.grab_particle.assign(np.full(self.grab_capacity, -1, np.int32))
        self.grab_hand.assign(np.full(self.grab_capacity, -1, np.int32))
        self.grab_local.zero_()
        self.grab_count.zero_()
        self.hand_active.zero_()
        self.contact_count.zero_()
        self.body_bad.zero_()

    def snapshot(self) -> dict[str, np.ndarray]:
        """Host copy of the state a test or a recording needs."""
        return {
            "x": self.x.numpy().copy(),
            "x_prev": self.x_prev.numpy().copy(),
            "v": self.v.numpy().copy(),
            "w": self.w.numpy().copy(),
            "flags": self.flags.numpy().copy(),
            "normal": self.normal.numpy().copy(),
            "body": self.body.numpy().copy(),
            "radius": self.radius.numpy().copy(),
            "grab_particle": self.grab_particle.numpy().copy(),
        }

    def restore(self, snap: dict[str, np.ndarray]) -> None:
        """Load a ``snapshot`` back onto the device.

        Comparing the CUDA-graph path against the plain-launch path is only
        meaningful if both start from exactly the same bytes, and that is the
        only thing this exists for.
        """
        self.x.assign(np.ascontiguousarray(snap["x"], np.float32))
        self.x_prev.assign(np.ascontiguousarray(snap["x_prev"], np.float32))
        self.v.assign(np.ascontiguousarray(snap["v"], np.float32))
        self.w.assign(np.ascontiguousarray(snap["w"], np.float32))
        self.flags.assign(np.ascontiguousarray(snap["flags"], np.uint32))

    # -- interface for the renderer ---------------------------------------

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def num_constraints(self) -> int:
        return self.num_dist + self.num_bend + self.num_tet

    @property
    def particle_offsets(self) -> np.ndarray:
        """``(num_bodies + 1,)`` prefix sums into the global particle buffer."""
        return self._particle_offsets

    @property
    def triangles(self) -> np.ndarray:
        """``(F, 3)`` int32 surface triangles, already in global indices."""
        return self._triangles

    @property
    def triangle_offsets(self) -> np.ndarray:
        return self._tri_offsets

    def tet_stiffness_ceiling(self, dt: float) -> np.ndarray:
        """``(num_bodies, 2)`` of the largest ``(mu, lambda)`` in Pa that the
        tetrahedra of each body can actually carry at this frame length.

        Above these, :func:`fctx.solver.kernels.resolvable_softening` scales the
        element down and the extra stiffness the dial asked for is delivered by
        the tet edge distance constraints instead of by the Neo-Hookean pair.
        That is a real departure from the material the dial names, so it is
        exposed rather than applied in silence: a HUD or a test can compare
        these against ``Material.lame_mu`` and ``Material.lame_lambda`` and say
        where the dial stops being literal.  ``inf`` for a body with no
        tetrahedra.
        """
        if dt <= 0.0:
            raise ValueError(f"tet_stiffness_ceiling needs a positive dt, got {dt}")
        out = np.full((self._num_bodies, 2), np.inf, np.float64)
        if self._tet_wsum_host.size == 0:
            return out
        h = dt / max(1, int(self.cfg.solver.substeps))
        margin = float(K.TET_STIFFNESS_MARGIN)
        denom = (margin * self._tet_vol_host * h * h
                 * np.maximum(self._tet_wsum_host.astype(np.float64), 1e-300))
        mu_max = np.full(self._num_bodies, np.inf, np.float64)
        np.minimum.at(mu_max, self._tet_body_host, 3.0 / denom)
        out[:, 0] = mu_max
        out[:, 1] = mu_max / 3.0
        return out

    def body_particle_range(self, index: int) -> tuple[int, int]:
        """Half-open ``[start, end)`` slice of the global buffer for a body."""
        if not 0 <= index < self._num_bodies:
            raise IndexError(f"body {index} out of range for {self._num_bodies} bodies")
        return (int(self._particle_offsets[index]),
                int(self._particle_offsets[index + 1]))

    def body_triangles(self, index: int) -> np.ndarray:
        """That body's surface triangles, in global particle indices."""
        if not 0 <= index < self._num_bodies:
            raise IndexError(f"body {index} out of range for {self._num_bodies} bodies")
        start = int(self._tri_offsets[index])
        end = int(self._tri_offsets[index + 1])
        return self._triangles[start:end]

    def attach_render_buffers(self, positions: wp.array | None,
                              normals: wp.array | None) -> None:
        """Point the solver at mapped GL buffers for ``sync_render``.

        The renderer maps its VBOs through ``wp.RegisteredGLBuffer`` and hands
        the mapped arrays here; ``sync_render`` then blits device to device,
        so positions never travel through the host.
        """
        for name, arr in (("positions", positions), ("normals", normals)):
            if arr is not None and arr.shape[0] < self._num_particles:
                raise ValueError(
                    f"render {name} buffer holds {arr.shape[0]} elements, "
                    f"need {self._num_particles}")
        self._render_positions = positions
        self._render_normals = normals

    def sync_render(self) -> None:
        if self._render_positions is not None:
            wp.copy(self._render_positions, self.x_skin,
                    count=self._num_particles)
        if self._render_normals is not None:
            wp.copy(self._render_normals, self.normal, count=self._num_particles)
