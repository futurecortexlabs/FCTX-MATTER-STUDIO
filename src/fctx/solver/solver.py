"""The XPBD stepper.

``XPBDSolver`` owns the substep loop, the hand colliders and the grab
attachments.  It is the only thing in the project that writes GPU physics
state; everything else reads it.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..config import AppConfig
from ..core.types import FLAG_GRABBED, HandPose
from . import kernels as K
from .state import SolverState

__all__ = ["XPBDSolver"]


def _palm_quat(joints: np.ndarray) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of the palm frame: wrist, index MCP, pinky MCP.

    The palm is the rigid part of a hand.  The pinch frame is built from the
    thumb and index tips and turns whenever the fingers move, which is
    exactly the motion a held body must not follow.
    """
    w = np.asarray(joints[0], np.float64)
    i = np.asarray(joints[5], np.float64)
    k = np.asarray(joints[17], np.float64)
    x = i - w
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    x /= nx
    z = np.cross(x, k - w)
    nz = np.linalg.norm(z)
    if nz < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    z /= nz
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                      (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                      0.25 * s, (m[1, 0] - m[0, 1]) / s])
    return (q / np.linalg.norm(q)).astype(np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` for (x, y, z, w) quaternions."""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], np.float32)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float32)


def _quat_conjugate_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of unit quaternion ``q`` (x, y, z, w)."""
    u = -np.asarray(q[:3], np.float64)
    s = float(q[3])
    v = np.asarray(v, np.float64)
    cross1 = np.cross(u, v)
    return (v + 2.0 * (s * cross1 + np.cross(u, cross1))).astype(np.float32)


class XPBDSolver:
    """Advance a :class:`SolverState` by fixed substeps.

    Parameters
    ----------
    state:
        The scene to advance.  The solver keeps a reference, never a copy.
    cfg:
        Substep count, iteration count and the CUDA-graph switch come from
        ``cfg.solver``; grab behaviour from ``cfg.grab``.
    """

    #: Seconds a newly tracked hand takes to reach its full collider radius.
    COLLIDER_FADE_IN = 0.12

    #: ...or metres of travel, whichever finishes the fade first.  The fade
    #: exists because a hand appears *wherever the person's hand is*, which
    #: can be inside the matter; a hand that has since moved most of its own
    #: width is no longer materialising inside anything, it is approaching
    #: from outside, which is the case a full-radius collider handles
    #: correctly.  Time alone protects only the case it was measured on:
    #: measured on a 70-grain slab swept by a palm, a hand that appears 30 mm
    #: away and sweeps at 1.2 m/s -- an ordinary swipe -- reaches the matter
    #: at 9% of its radius and leaks three times as many grains through
    #: itself as a grown collider does.
    COLLIDER_FADE_TRAVEL = 0.05

    def __init__(self, state: SolverState, cfg: AppConfig) -> None:
        self.state = state
        self.cfg = cfg
        self.device = state.device
        self.substeps = max(1, int(cfg.solver.substeps))
        self.iterations = max(1, int(cfg.solver.iterations))

        # Graph capture is a CUDA feature; the CPU path silently ignores the
        # config flag rather than failing, so tests can run anywhere.
        self.use_graph = (bool(cfg.solver.use_cuda_graph)
                          and wp.get_device(self.device).is_cuda)
        self._graph: object | None = None
        self._graph_h = 0.0

        self._step_index = 0
        self._contacts = 0
        #: Collider fade progress per slot, 0 at first sight and 1 once grown.
        self._cap_fade = [0.0] * self.state.max_hands
        self._grabbed = 0
        self._resets = 0

        h = state.max_hands
        self._grab_counts = [0] * h
        self._prev_cap_a: np.ndarray | None = None
        self._prev_cap_b: np.ndarray | None = None
        self._prev_hand_pos = np.zeros((h, 3), np.float32)
        self._prev_wrist = np.zeros((h, 3), np.float32)
        self._prev_palm = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], np.float32), (h, 1))
        self._prev_hand_rot = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], np.float32), (h, 1))
        self._prev_present = [False] * h
        #: When each slot's pinch first closed, or None.  None rather than a
        #: sentinel number because ``now`` is the caller's clock and zero is a
        #: perfectly ordinary reading on it.
        self._pinch_since: list[float | None] = [None] * h

        # Scratch buffers for begin_grab, sized once at capacity so that a
        # grab never allocates while the render loop is running.
        cap = state.grab_per_hand
        self._sel_dev = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._local_dev = wp.zeros(cap, dtype=wp.vec3, device=self.device)

    # ------------------------------------------------------------------
    # hands
    # ------------------------------------------------------------------

    def slot_map(self, poses: list[HandPose]) -> dict[int, HandPose]:
        """Place each pose in the hand slot its track id names.

        ``HandTracker`` documents ``track_id`` as the slot -- a small integer
        in ``[0, max_hands)`` that a physical hand keeps from the moment it
        appears until it leaves -- and it returns a *compacted* list, so when
        one of two hands leaves the survivor keeps its id and changes list
        position.  Keying this class's per-hand state by list position instead
        made that survivor inherit the vanished hand's capsule history (a full
        velocity-clamp friction impulse out of a hand that never moved), its
        collider fade (a continuously tracked hand collapsing to 2.4% of its
        radius whenever its partner reappeared) and its grab, while
        ``interaction.GripManager``, which does key by track id, went on
        driving the other slot -- so the solver released a grab the grip
        manager still believed in and that hand could never grab again.

        A caller that builds poses by hand and leaves ``track_id`` at its
        default gets the old behaviour: an id that is out of range, or one
        already claimed this frame, falls back to the next free slot.
        """
        n_slots = self.state.max_hands
        out: dict[int, HandPose] = {}
        spare: list[HandPose] = []
        for pose in poses[:n_slots]:
            slot = int(pose.track_id)
            if 0 <= slot < n_slots and slot not in out:
                out[slot] = pose
            else:
                spare.append(pose)
        free = (s for s in range(n_slots) if s not in out)
        for pose, slot in zip(spare, free):
            out[slot] = pose
        return out

    def set_hands(self, poses: list[HandPose], dt: float) -> None:
        """Upload hand capsules and pinch frames for this frame.

        ``dt`` is used to finite-difference the capsule endpoints rather than
        trusting ``HandPose.velocities``: friction reads the capsule's motion
        relative to the matter, and if that motion disagrees with the capsule
        positions the solver actually sees, the mismatch pumps energy into the
        cloth every frame.
        """
        if dt <= 0.0:
            raise ValueError(f"set_hands needs a positive dt, got {dt}")
        st = self.state
        n_slots = st.max_hands
        by_slot = self.slot_map(poses)
        bones = st.bones_per_hand

        cap_a = np.zeros((st.capsule_capacity, 3), np.float32)
        cap_b = np.zeros((st.capsule_capacity, 3), np.float32)
        prev_a = np.zeros_like(cap_a)
        prev_b = np.zeros_like(cap_b)
        cap_r = np.zeros(st.capsule_capacity, np.float32)
        prev_r = np.zeros_like(cap_r)
        cap_va = np.zeros_like(cap_a)
        cap_vb = np.zeros_like(cap_b)
        hand_pos = self._prev_hand_pos.copy()
        hand_vel = np.zeros((n_slots, 3), np.float32)
        hand_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (n_slots, 1))

        for slot, pose in by_slot.items():
            a, b, r = pose.bone_segments()
            if a.shape[0] != bones:
                raise ValueError(
                    f"hand {slot} produced {a.shape[0]} capsules, expected {bones}")
            lo = slot * bones
            hi = lo + bones
            cap_a[lo:hi] = a
            cap_b[lo:hi] = b

            # Differencing only against a frame where this same slot was also
            # tracked.  Differencing against a slot that was empty would treat
            # the jump from the origin as motion, and the throw-on-release
            # would then fire whatever the hand was holding across the stage
            # at the velocity clamp.
            tracked = self._prev_present[slot] and self._prev_cap_a is not None
            travel = 0.0
            if tracked:
                # The substep sweep starts from where this slot's capsules
                # actually were.  A slot that was empty has nowhere to sweep
                # from, so it starts where it is -- sweeping from the origin
                # would drag a 0.3 m segment through the scene.
                prev_a[lo:hi] = self._prev_cap_a[lo:hi]
                prev_b[lo:hi] = self._prev_cap_b[lo:hi]
                step_a = a - self._prev_cap_a[lo:hi]
                cap_va[lo:hi] = step_a / dt
                cap_vb[lo:hi] = (b - self._prev_cap_b[lo:hi]) / dt
                hand_vel[slot] = (pose.pinch_point
                                  - self._prev_hand_pos[slot]) / dt
                # Mean, not max: one landmark flickering is noise, and letting
                # noise finish the fade is exactly what the fade is for.
                travel = float(np.linalg.norm(step_a, axis=1).mean())
            else:
                prev_a[lo:hi] = a
                prev_b[lo:hi] = b
                va, vb = pose.bone_velocities()
                cap_va[lo:hi] = va
                cap_vb[lo:hi] = vb
                hand_vel[slot] = pose.pinch_velocity

            # Grow the colliders in whenever a slot starts being tracked.  A
            # hand appears wherever the person's hand happens to be, and that
            # can be in the middle of the matter: the granular scene puts the
            # interaction volume inside the pile, so at full radius on frame
            # one, twenty-one capsules materialise inside twenty-four thousand
            # grains and throw them five metres -- measured, 5.2 m and 35
            # grains off the stage against 0.30 m and none with the fade in
            # place.  Fading the radius lets the matter part
            # instead of detonate, and once grown it costs nothing.  The fade
            # advances on travel as well as on time so that it protects the
            # case it was written for without opening a hole in a hand that is
            # merely moving: see COLLIDER_FADE_TRAVEL.
            was = self._cap_fade[slot]
            self._cap_fade[slot] = min(
                1.0, was + dt / self.COLLIDER_FADE_IN
                + travel / self.COLLIDER_FADE_TRAVEL)
            fade = self._cap_fade[slot]
            cap_r[lo:hi] = r * (fade * fade * (3.0 - 2.0 * fade))
            if tracked:
                prev_r[lo:hi] = r * (was * was * (3.0 - 2.0 * was))
            else:
                prev_r[lo:hi] = cap_r[lo:hi]
            wrist = np.asarray(pose.joints[0], np.float32)
            palm = _palm_quat(pose.joints)
            holding = tracked and self._grab_counts[slot] > 0
            if holding:
                # The anchor is taken at the pinch point -- the midpoint of
                # the thumb and index tips -- but it must not *follow* it.
                # Opening the fingers moves that midpoint several centimetres
                # while the wrist stays put, and it starts moving before the
                # pinch value has dropped at all; the held matter was dragged
                # along with it and then thrown with its velocity, so a soft
                # ball let go by a motionless hand left at 0.8 m/s sideways.
                # Once something is held it is the hand's rigid motion that
                # should carry it, and the wrist is the joint that carries
                # that.  The twist is still the fingers' while they are
                # firmly closed, and freezes as they open.
                shift = wrist - self._prev_wrist[slot]
                hand_pos[slot] = self._prev_hand_pos[slot] + shift
                hand_vel[slot] = shift / dt
                # The twist, likewise, is the palm's: the anchor rotation
                # advances by the palm frame's change since last frame.  The
                # pinch frame turns with the fingers, and with a few hundred
                # particles held deep in a ball that turn is a whip -- 5 m/s
                # from fingers merely opening.
                delta = _quat_mul(palm, _quat_conj(self._prev_palm[slot]))
                rot = _quat_mul(delta, self._prev_hand_rot[slot])
                hand_rot[slot] = rot / max(float(np.linalg.norm(rot)), 1e-9)
            else:
                hand_pos[slot] = pose.pinch_point
                hand_rot[slot] = pose.pinch_rotation
            self._prev_wrist[slot] = wrist
            self._prev_palm[slot] = palm
            self._prev_hand_rot[slot] = hand_rot[slot]

        # A tracking dropout can still teleport a landmark within one tracked
        # slot; letting that become a capsule velocity would fire matter off
        # the stage.
        vmax = self.cfg.solver.max_velocity
        for arr in (cap_va, cap_vb, hand_vel):
            np.clip(arr, -vmax, vmax, out=arr)

        st.cap_a.assign(cap_a)
        st.cap_b.assign(cap_b)
        st.cap_a_prev.assign(prev_a)
        st.cap_b_prev.assign(prev_b)
        st.cap_r.assign(cap_r)
        st.cap_r_prev.assign(prev_r)
        st.cap_va.assign(cap_va)
        st.cap_vb.assign(cap_vb)
        # Slots are addressed individually, so an occupied slot 1 with an
        # empty slot 0 is an ordinary state; the count covers the highest slot
        # in use and the kernel skips the zero-radius capsules in between.
        live = (max(by_slot) + 1) * bones if by_slot else 0
        st.cap_count.assign(np.array([live], np.int32))
        st.hand_pos.assign(hand_pos)
        st.hand_rot.assign(hand_rot)
        st.hand_vel.assign(hand_vel)

        self._prev_cap_a = cap_a
        self._prev_cap_b = cap_b
        self._prev_hand_pos = hand_pos
        self._prev_present = [slot in by_slot for slot in range(n_slots)]

        # A hand that vanished cannot keep holding anything; releasing here
        # rather than waiting for the app keeps a lost track from pinning a
        # body to the last place the hand was seen.  Its pinch timer goes too,
        # or the slot's next hand inherits a hold time that elapsed while
        # nothing was there and grabs on its first frame.
        for slot in range(n_slots):
            if slot in by_slot:
                continue
            self._pinch_since[slot] = None
            self._cap_fade[slot] = 0.0
            if self._grab_counts[slot]:
                self._release_slot(slot, throw=False)

    # ------------------------------------------------------------------
    # grabs
    # ------------------------------------------------------------------

    def begin_grab(self, hand_slot: int, pose: HandPose) -> int:
        """Attach the particles near ``pose.pinch_point`` to that hand.

        Returns the number of particles attached.
        """
        st = self.state
        self._check_slot(hand_slot)
        if self._grab_counts[hand_slot]:
            return self._grab_counts[hand_slot]

        x = st.x.numpy()
        w = st.w.numpy()
        flags = st.flags.numpy()
        pinch = np.asarray(pose.pinch_point, np.float32)

        d2 = np.einsum("ij,ij->i", x - pinch, x - pinch)
        radius = float(self.cfg.grab.radius)
        eligible = (d2 <= radius * radius) & (w > 0.0) & ((flags & FLAG_GRABBED) == 0)
        candidates = np.flatnonzero(eligible)
        if candidates.size == 0:
            return 0
        limit = st.grab_per_hand
        if candidates.size > limit:
            # Keep the closest: an arbitrary truncation would drop the
            # particles under the fingertips and hold the far edge instead.
            order = np.argsort(d2[candidates], kind="stable")[:limit]
            candidates = candidates[np.sort(order)]

        local = np.empty((candidates.size, 3), np.float32)
        offsets = x[candidates] - pinch
        if self.cfg.grab.rotate_with_pinch:
            q = np.asarray(pose.pinch_rotation, np.float32)
            for i in range(candidates.size):
                local[i] = _quat_conjugate_rotate(q, offsets[i])
        else:
            local[:] = offsets

        count = int(candidates.size)
        cap = st.grab_per_hand
        sel_host = np.full(cap, -1, np.int32)
        sel_host[:count] = candidates
        local_host = np.zeros((cap, 3), np.float32)
        local_host[:count] = local
        self._sel_dev.assign(sel_host)
        self._local_dev.assign(local_host)

        scale = float(self.cfg.grab.mass_scale)
        if scale <= 0.0:
            raise ValueError("GrabConfig.mass_scale must be positive")
        wp.launch(
            K.apply_grab, dim=count,
            inputs=[self._sel_dev, self._local_dev, st.grab_particle, st.grab_local,
                    st.grab_hand, st.w, st.w_rest, st.flags,
                    hand_slot, hand_slot * st.grab_per_hand, 1.0 / scale],
            device=self.device)

        self._grab_counts[hand_slot] = count
        self._refresh_hand_active()
        self._grabbed = sum(self._grab_counts)
        st.grab_count.assign(np.array([self._grabbed], np.int32))
        return count

    def end_grab(self, hand_slot: int, pose: HandPose | None) -> None:
        """Release that hand's grab, handing the matter the hand's velocity.

        ``pose`` may be ``None``, and that is not the same event: it is how
        ``interaction.GripManager`` reports that the tracker lost the hand
        rather than that the person opened it.  There is no trustworthy hand
        velocity behind a lost track, so the matter is dropped where it is
        instead of being thrown.
        """
        self._check_slot(hand_slot)
        if not self._grab_counts[hand_slot]:
            return
        if pose is None:
            self._release_slot(hand_slot, throw=False)
            return
        st = self.state
        pinch_v = np.asarray(pose.pinch_velocity, np.float32)
        if float(np.dot(pinch_v, pinch_v)) > 1.0e-8:
            # The tracker's own pinch velocity is filtered and already leads
            # the finite difference slightly, which is what makes a flick feel
            # like it landed.  Fall back to the differenced value only when the
            # source does not supply one.
            vel = st.hand_vel.numpy()
            vel[hand_slot] = np.clip(pinch_v, -self.cfg.solver.max_velocity,
                                     self.cfg.solver.max_velocity)
            st.hand_vel.assign(vel)
        self._release_slot(hand_slot)

    def _release_slot(self, hand_slot: int, throw: bool = True) -> None:
        st = self.state
        wp.launch(
            K.release_grab, dim=st.grab_capacity,
            inputs=[st.grab_particle, st.grab_hand, st.grab_lambda, st.w, st.w_rest,
                    st.v, st.flags, st.hand_vel, hand_slot,
                    float(self.cfg.grab.throw_gain), 0 if throw else 1],
            device=self.device)
        self._grab_counts[hand_slot] = 0
        self._refresh_hand_active()
        self._grabbed = sum(self._grab_counts)
        st.grab_count.assign(np.array([self._grabbed], np.int32))

    def update_grabs(self, poses: list[HandPose], now: float) -> None:
        """Start and stop grabs from pinch strength, with hysteresis.

        ARCHITECTURE.md 7.6: one threshold makes a pinch held near the
        boundary drop and re-take the object several times a second, which
        looks like the physics failing rather than the tracking failing.
        """
        g = self.cfg.grab
        by_slot = self.slot_map(poses)
        for slot in range(self.state.max_hands):
            pose = by_slot.get(slot)
            if pose is None:
                self._pinch_since[slot] = None
                continue
            if self._grab_counts[slot]:
                if pose.pinch < g.release_threshold:
                    self.end_grab(slot, pose)
                continue
            if pose.pinch >= g.start_threshold:
                since = self._pinch_since[slot]
                if since is None:
                    self._pinch_since[slot] = now
                elif now - since >= g.hold_time:
                    self.begin_grab(slot, pose)
                    self._pinch_since[slot] = None
            else:
                self._pinch_since[slot] = None

    def _refresh_hand_active(self) -> None:
        active = np.array([1 if c else 0 for c in self._grab_counts], np.int32)
        self.state.hand_active.assign(active)

    def _check_slot(self, hand_slot: int) -> None:
        if not 0 <= hand_slot < self.state.max_hands:
            raise IndexError(
                f"hand slot {hand_slot} outside [0, {self.state.max_hands})")

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds."""
        if dt <= 0.0:
            raise ValueError(f"step needs a positive dt, got {dt}")
        st = self.state
        h = dt / self.substeps

        # Before anything reads the positions, and in particular before the
        # hash grid is built from them, put any particle that has left the
        # world back at rest.  ARCHITECTURE.md 7 rule 8 asks for this every
        # step, and it has to be: the grid build is not defence against a
        # coordinate it cannot bucket, it is the thing that crashes on one.
        wp.launch(K.contain_particles, dim=st.num_particles,
                  inputs=[st.x, st.x_prev, st.x_rest, st.v, st.body, st.body_bad,
                          st.contain_limit],
                  device=self.device)

        # The hash grid cannot be built from inside a graph capture, so it is
        # rebuilt here, once, against the positions at the start of the step,
        # and the substeps then query a grid that is up to one step out of
        # date.  That is a real limit, not something the collision margin
        # covers: a particle at the velocity clamp travels 0.13 m in a step
        # against a cell of about 1.5 cm, so a fast pile finds some of its
        # contacts a step late and already deep.  It is why the contact
        # correction is bounded by the contact sphere in
        # `solve_particle_contacts` -- a late contact must not be allowed to
        # pay off its whole debt in one substep.
        if self.cfg.solver.self_collision:
            st.grid.build(points=st.x, radius=st.grid_radius)

        if self.use_graph:
            if self._graph is None or self._graph_h != h:
                self._capture(h)
            wp.capture_launch(self._graph)
        else:
            self._record(h)

        self._step_index += 1
        interval = int(self.cfg.solver.sanity_interval)
        if interval > 0 and self._step_index % interval == 0:
            self._sanity_sweep()

    def _capture(self, h: float) -> None:
        # Compiling or loading a kernel inside a capture is not allowed, and
        # the failure surfaces much later as a corrupt graph, so force the
        # whole module resident first.
        wp.load_module(K, device=self.device)
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self._record(h)
        self._graph = capture.graph
        self._graph_h = h

    def _record(self, h: float) -> None:
        """Emit the substep loop, either for capture or for direct launch."""
        st = self.state
        dev = self.device
        p = st.num_particles

        lam_dim = max(st.num_dist, st.num_bend, st.num_tet, st.grab_capacity, 1)
        for sub in range(self.substeps):
            # How far through the frame this substep ends.  The loop is
            # unrolled here, so this is a different literal in each launch and
            # the same literal every frame -- which is what makes it safe to
            # bake into a captured graph.
            frac = float(sub + 1) / float(self.substeps)
            # Cleared per substep, so stats() reports contacts in flight at the
            # end of the step rather than a sum over substeps that would read
            # like twelve times as much matter is touching.
            wp.launch(K.clear_counters, dim=1, inputs=[st.contact_count],
                      device=dev)
            wp.launch(
                K.reset_lambdas, dim=lam_dim,
                inputs=[st.dist_lambda, st.bend_lambda, st.tet_lambda_d,
                        st.tet_lambda_h, st.grab_lambda,
                        st.num_dist, st.num_bend, st.num_tet, st.grab_capacity],
                device=dev)
            wp.launch(
                K.integrate, dim=p,
                inputs=[st.x, st.x_prev, st.v, st.w, st.contact_n, st.contact_vn,
                        st.params, st.vparams, h],
                device=dev)

            for _ in range(self.iterations):
                for offset, count in st.dist_batches:
                    wp.launch(
                        K.solve_distance, dim=count,
                        inputs=[st.x, st.w, st.radius, st.dist_idx, st.dist_rest,
                                st.dist_kind, st.dist_body, st.dist_lambda,
                                st.mat_stretch, st.mat_shear, st.params, offset, h],
                        device=dev)
                for offset, count in st.bend_batches:
                    wp.launch(
                        K.solve_bending, dim=count,
                        inputs=[st.x, st.w, st.radius, st.bend_idx, st.bend_rest,
                                st.bend_body, st.bend_lambda, st.mat_bend,
                                st.params, offset, h],
                        device=dev)
                for offset, count in st.tet_batches:
                    wp.launch(
                        K.solve_tet_deviatoric, dim=count,
                        inputs=[st.x, st.w, st.radius, st.tet_idx, st.tet_dm_inv,
                                st.tet_volume, st.tet_wsum_rest, st.tet_body,
                                st.tet_lambda_d, st.mat_dev, st.mat_hyd,
                                st.params, offset, h],
                        device=dev)
                    wp.launch(
                        K.solve_tet_hydrostatic, dim=count,
                        inputs=[st.x, st.w, st.radius, st.tet_idx, st.tet_dm_inv,
                                st.tet_volume, st.tet_wsum_rest, st.tet_body,
                                st.tet_lambda_h, st.mat_dev, st.mat_hyd,
                                st.params, offset, h],
                        device=dev)

                wp.launch(
                    K.solve_grab, dim=st.grab_capacity,
                    inputs=[st.x, st.w, st.radius, st.body, st.grab_particle,
                            st.grab_local, st.grab_hand, st.grab_lambda,
                            st.hand_pos, st.hand_rot, st.hand_active,
                            st.mat_grab, st.params, h],
                    device=dev)
                wp.launch(
                    K.solve_capsules, dim=p,
                    inputs=[st.x, st.x_prev, st.v, st.w, st.flags, st.radius,
                            st.body,
                            st.contact_n, st.contact_vn, st.cap_a, st.cap_b,
                            st.cap_a_prev, st.cap_b_prev,
                            st.cap_r, st.cap_r_prev,
                            st.cap_va, st.cap_vb, st.cap_count,
                            st.mat_friction, st.mat_restitution,
                            st.contact_count, st.params, frac, h],
                    device=dev)
                wp.launch(
                    K.solve_ground, dim=p,
                    inputs=[st.x, st.x_prev, st.v, st.w, st.radius, st.body,
                            st.contact_n, st.contact_vn, st.mat_friction,
                            st.mat_restitution, st.contact_count, st.params, h],
                    device=dev)
                if self.cfg.solver.self_collision:
                    wp.launch(
                        K.solve_particle_contacts, dim=p,
                        inputs=[st.x, st.x_prev, st.contact_dx, st.w,
                                st.radius, st.collide_radius, st.flags, st.body,
                                st.mat_friction, st.contact_count,
                                st.grid.id, st.params, st.grid_radius],
                        device=dev)
                    wp.launch(K.apply_corrections, dim=p,
                              inputs=[st.x, st.contact_dx], device=dev)

            wp.launch(
                K.finalize, dim=p,
                inputs=[st.x, st.x_prev, st.v, st.w, st.body, st.contact_n,
                        st.contact_vn, st.mat_damping, st.params, h],
                device=dev)
            wp.launch(K.advance_time, dim=1, inputs=[st.params, h], device=dev)

    def compute_normals(self) -> None:
        """Rebuild the drawn surface and its smooth shading normals.

        The normals come from ``x_skin``, not ``x``: shading the lattice and
        drawing the skin would light a sphere as though it were the staircase
        underneath it.
        """
        st = self.state
        if st.has_skin:
            wp.launch(K.apply_skin, dim=st.num_particles,
                      inputs=[st.x, st.tet_idx, st.skin_tet, st.skin_bary,
                              st.skin_blend, st.x_skin], device=self.device)
        wp.launch(K.zero_vec3, dim=st.num_particles, inputs=[st.normal],
                  device=self.device)
        if st.num_tri:
            wp.launch(K.accumulate_normals, dim=st.num_tri,
                      inputs=[st.x_skin, st.tri_idx, st.normal],
                      device=self.device)
        wp.launch(K.normalize_normals, dim=st.num_particles, inputs=[st.normal],
                  device=self.device)

    def _sanity_sweep(self) -> int:
        """Reset whole bodies that ``contain_particles`` caught since the last
        sweep, and return how many.

        The per-step containment only rescues the particles that actually left
        the world; the rest of that body is still mid-explosion.  This is the
        part that needs a host round trip -- to release the grabs, which is
        host state -- so it stays on the ``sanity_interval`` cadence.
        """
        st = self.state
        bad = st.body_bad.numpy()
        count = int(bad.sum())
        if count == 0:
            return 0
        # Resetting only the offending body keeps one exploded soft body from
        # taking the cloth next to it with it.
        wp.launch(K.reset_flagged, dim=st.num_particles,
                  inputs=[st.x, st.x_prev, st.x_rest, st.v, st.w, st.w_rest,
                          st.flags, st.body, st.body_bad], device=self.device)
        wp.launch(K.clear_int, dim=st.num_bodies, inputs=[st.body_bad],
                  device=self.device)
        # A grab whose particles have just been teleported home is holding
        # nothing meaningful, and its stored local offsets are now wrong.
        # Releasing without a throw: there is no hand motion behind this.
        for slot in range(st.max_hands):
            if self._grab_counts[slot]:
                self._release_slot(slot, throw=False)
        self._resets += count
        return count

    def reset(self) -> None:
        self.state.reset()
        self._grab_counts = [0] * self.state.max_hands
        self._pinch_since = [None] * self.state.max_hands
        self._prev_present = [False] * self.state.max_hands
        self._prev_cap_a = None
        self._prev_cap_b = None
        self._prev_hand_pos = np.zeros((self.state.max_hands, 3), np.float32)
        self._prev_wrist = np.zeros((self.state.max_hands, 3), np.float32)
        self._prev_palm = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32),
                                  (self.state.max_hands, 1))
        self._prev_hand_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32),
                                      (self.state.max_hands, 1))
        self._cap_fade = [0.0] * self.state.max_hands
        self._grabbed = 0
        self._step_index = 0
        self._resets = 0

    def stats(self) -> dict[str, int]:
        st = self.state
        self._contacts = int(st.contact_count.numpy()[0])
        return {
            "particles": st.num_particles,
            "bodies": st.num_bodies,
            "constraints": st.num_constraints,
            "distance": st.num_dist,
            "bending": st.num_bend,
            "tetrahedra": st.num_tet,
            "triangles": st.num_tri,
            "substeps": self.substeps,
            "iterations": self.iterations,
            "contacts": self._contacts,
            "grabbed": self._grabbed,
            "steps": self._step_index,
            "resets": self._resets,
            "graph": 1 if self.use_graph else 0,
        }
