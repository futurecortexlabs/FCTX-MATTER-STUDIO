"""Shared data types for FCTX MATTER STUDIO.

This module is the interface contract between the four subsystems:

    hands/   produces  HandFrame -> HandPose
    bodies/  produces  BodyData  (CPU, numpy)
    solver/  consumes  BodyData + HandPose, produces GPU particle state
    render/  consumes  GPU particle state + HandPose

Nothing here imports warp, OpenGL or mediapipe: it must stay importable on a
machine with no GPU so that tests and tooling can run anywhere.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# hand landmark topology (MediaPipe Hands, 21 points)
# --------------------------------------------------------------------------

NUM_LANDMARKS = 21


class L(enum.IntEnum):
    """Named indices into a 21-landmark hand."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


#: Bones of the hand skeleton, as (parent, child) landmark index pairs.
#: These become the capsule colliders that push the simulated matter around.
HAND_BONES: tuple[tuple[int, int], ...] = (
    (L.WRIST, L.THUMB_CMC),
    (L.THUMB_CMC, L.THUMB_MCP),
    (L.THUMB_MCP, L.THUMB_IP),
    (L.THUMB_IP, L.THUMB_TIP),
    (L.WRIST, L.INDEX_MCP),
    (L.INDEX_MCP, L.INDEX_PIP),
    (L.INDEX_PIP, L.INDEX_DIP),
    (L.INDEX_DIP, L.INDEX_TIP),
    (L.INDEX_MCP, L.MIDDLE_MCP),
    (L.MIDDLE_MCP, L.MIDDLE_PIP),
    (L.MIDDLE_PIP, L.MIDDLE_DIP),
    (L.MIDDLE_DIP, L.MIDDLE_TIP),
    (L.MIDDLE_MCP, L.RING_MCP),
    (L.RING_MCP, L.RING_PIP),
    (L.RING_PIP, L.RING_DIP),
    (L.RING_DIP, L.RING_TIP),
    (L.RING_MCP, L.PINKY_MCP),
    (L.PINKY_MCP, L.PINKY_PIP),
    (L.PINKY_PIP, L.PINKY_DIP),
    (L.PINKY_DIP, L.PINKY_TIP),
    (L.WRIST, L.PINKY_MCP),
)

#: Collider radius (metres) for the capsule spanning each bone in HAND_BONES.
#: Fingers taper towards the tips; the palm bones are fat.
BONE_RADII: tuple[float, ...] = (
    0.017, 0.014, 0.012, 0.011,      # thumb
    0.018, 0.012, 0.010, 0.0095,     # index
    0.018, 0.012, 0.010, 0.0095,     # middle, plus the index->middle palm link
    0.018, 0.011, 0.0095, 0.009,     # ring, plus the middle->ring palm link
    0.017, 0.011, 0.009, 0.0085,     # pinky, plus the ring->pinky palm link
    0.019,                           # wrist -> pinky mcp (outer palm edge)
)

assert len(BONE_RADII) == len(HAND_BONES)

#: Landmarks whose triangle defines the palm plane (used for the palm normal).
PALM_TRIANGLE = (int(L.WRIST), int(L.INDEX_MCP), int(L.PINKY_MCP))

#: Landmarks used to measure how open the hand is.
FINGER_TIPS = (int(L.THUMB_TIP), int(L.INDEX_TIP), int(L.MIDDLE_TIP),
               int(L.RING_TIP), int(L.PINKY_TIP))
FINGER_MCPS = (int(L.THUMB_CMC), int(L.INDEX_MCP), int(L.MIDDLE_MCP),
               int(L.RING_MCP), int(L.PINKY_MCP))


class Handedness(enum.IntEnum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2


# --------------------------------------------------------------------------
# raw tracker output
# --------------------------------------------------------------------------


@dataclass(slots=True)
class HandFrame:
    """One hand as reported by a tracker, before smoothing or projection.

    ``image`` landmarks are normalised to the camera frame: x and y in [0, 1]
    with the origin at the top-left, z a *relative* depth on roughly the same
    scale as x (negative = closer to the camera than the wrist).

    ``world`` landmarks are metres, right-handed, origin at the hand's
    geometric centre.  They carry the hand's shape but not its position.
    """

    image: np.ndarray  # (21, 3) float32
    world: np.ndarray  # (21, 3) float32
    handedness: Handedness = Handedness.UNKNOWN
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.image.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"image landmarks must be (21, 3), got {self.image.shape}")
        if self.world.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"world landmarks must be (21, 3), got {self.world.shape}")


@dataclass(slots=True)
class TrackerFrame:
    """Everything a tracker produces for a single camera frame."""

    hands: list[HandFrame] = field(default_factory=list)
    #: Monotonic seconds, from the same clock as ``time.perf_counter``.
    timestamp: float = 0.0
    #: Frame index, monotonically increasing, never reused.
    index: int = 0
    #: RGB uint8 preview image (H, W, 3), or None when the source has none.
    preview: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return len(self.hands) > 0


# --------------------------------------------------------------------------
# processed hand, in world (simulation) space
# --------------------------------------------------------------------------


class Gesture(enum.IntEnum):
    OPEN = 0
    PINCH = 1
    FIST = 2
    POINT = 3


@dataclass(slots=True)
class HandPose:
    """A tracked hand lifted into simulation space, ready to drive physics.

    All positions are metres in the simulation's world frame: +X right,
    +Y up, +Z towards the viewer.
    """

    #: (21, 3) float32 smoothed joint positions in world space.
    joints: np.ndarray
    #: (21, 3) float32 joint velocities, m/s.
    velocities: np.ndarray
    handedness: Handedness = Handedness.UNKNOWN
    #: 0 = thumb and index wide apart, 1 = touching.
    pinch: float = 0.0
    #: True while the pinch is closed enough to hold matter.
    pinching: bool = False
    #: 0 = flat hand, 1 = closed fist.
    curl: float = 0.0
    gesture: Gesture = Gesture.OPEN
    #: Outward normal of the palm, unit length.
    palm_normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], np.float32))
    #: Midpoint of thumb tip and index tip: where a pinch grabs.
    pinch_point: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    pinch_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    #: Unit quaternion (x, y, z, w) of the pinch frame, for twisting a held body.
    pinch_rotation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], np.float32))
    #: Tracking quality in [0, 1]; drops while the hand is predicted, not seen.
    confidence: float = 1.0
    #: Stable id, so a two-handed scene can tell the hands apart across frames.
    track_id: int = 0

    def bone_segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(a, b, radius)`` arrays describing this hand's capsules."""
        idx = np.asarray(HAND_BONES, dtype=np.int32)
        return (
            np.ascontiguousarray(self.joints[idx[:, 0]], dtype=np.float32),
            np.ascontiguousarray(self.joints[idx[:, 1]], dtype=np.float32),
            np.asarray(BONE_RADII, dtype=np.float32),
        )

    def bone_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(va, vb)``: the velocity of each capsule's two endpoints."""
        idx = np.asarray(HAND_BONES, dtype=np.int32)
        return (
            np.ascontiguousarray(self.velocities[idx[:, 0]], dtype=np.float32),
            np.ascontiguousarray(self.velocities[idx[:, 1]], dtype=np.float32),
        )


# --------------------------------------------------------------------------
# matter
# --------------------------------------------------------------------------


class MatterKind(enum.IntEnum):
    CLOTH = 0
    SOFT = 1
    GRAIN = 2


class ConstraintKind(enum.IntEnum):
    """Constraint groups.  A body may use several; the solver runs them in
    this order within every substep."""

    STRETCH = 0   # structural distance constraint (cloth warp/weft, grain bonds)
    SHEAR = 1     # diagonal distance constraint (cloth)
    BEND = 2      # dihedral angle between two triangles (cloth)
    TETRA = 3     # stable Neo-Hookean tetrahedron (soft body)


# Particle flag bits, stored as one uint32 per particle.
FLAG_PINNED = 1 << 0        # infinite mass in the rest configuration
FLAG_GRABBED = 1 << 1       # currently held by a hand
FLAG_SURFACE = 1 << 2       # lies on the render surface
FLAG_SELF_COLLIDE = 1 << 3  # takes part in particle-particle collision


@dataclass(slots=True)
class BodyData:
    """CPU-side description of one body, produced by :mod:`fctx.bodies`.

    Index spaces are *local*: particle indices in the constraint arrays run
    from ``0`` to ``num_particles - 1``.  :class:`fctx.solver.state.SolverState`
    offsets them when several bodies share one GPU buffer.
    """

    kind: MatterKind
    name: str

    # --- particles -------------------------------------------------------
    positions: np.ndarray            # (P, 3) float32
    inv_mass: np.ndarray             # (P,)   float32, 0 == pinned
    flags: np.ndarray                # (P,)   uint32

    # --- distance constraints (stretch / shear) --------------------------
    dist_idx: np.ndarray             # (D, 2) int32
    dist_rest: np.ndarray            # (D,)   float32
    dist_kind: np.ndarray            # (D,)   int32, a ConstraintKind
    dist_color: np.ndarray           # (D,)   int32, graph colour

    # --- dihedral bending constraints ------------------------------------
    bend_idx: np.ndarray             # (B, 4) int32: (edge0, edge1, wing0, wing1)
    bend_rest: np.ndarray            # (B,)   float32, rest dihedral angle (rad)
    bend_color: np.ndarray           # (B,)   int32

    # --- tetrahedra (stable Neo-Hookean) ---------------------------------
    tet_idx: np.ndarray              # (T, 4) int32
    tet_dm_inv: np.ndarray           # (T, 3, 3) float32, inverse rest shape matrix
    tet_rest_volume: np.ndarray      # (T,)   float32
    tet_color: np.ndarray            # (T,)   int32

    # --- render surface ---------------------------------------------------
    tri_idx: np.ndarray              # (F, 3) int32, surface triangles
    uv: np.ndarray                   # (P, 2) float32
    #: True when the surface must be lit from both sides (a sheet of cloth).
    double_sided: bool = False

    # --- embedded render skin ---------------------------------------------
    #: Optional (P, K) int32 / (P, K, 4) float32 pair binding each particle's
    #: *rendered* position to up to K tetrahedra, as barycentric weights in
    #: each one's rest pose, pre-scaled so the drawn point is the plain sum
    #: over every slot.  A slot of ``-1`` is unused; a particle with no slots
    #: is drawn where it actually is.
    #:
    #: A voxelised soft body cannot be both smooth and stable: pulling its
    #: skin onto the real isosurface is what makes it look like a sphere, and
    #: the boundary tetrahedra -- the ones with all four vertices on the skin
    #: and nowhere to give -- are what invert when it is then squeezed.  The
    #: way out is to stop asking one set of points to do both jobs.  Physics
    #: keeps the lattice, which is well conditioned; rendering gets the
    #: isosurface, carried along by the lattice's own deformation.  The
    #: mapping is affine per element, so the drawn surface follows every
    #: stretch, twist and dent exactly, at the cost of one kernel over the
    #: particles.
    skin_tet: np.ndarray | None = None
    skin_bary: np.ndarray | None = None

    #: Particle radius, used for collision and for point-sprite rendering.
    particle_radius: float = 0.01

    @property
    def num_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_dist(self) -> int:
        return int(self.dist_idx.shape[0])

    @property
    def num_bend(self) -> int:
        return int(self.bend_idx.shape[0])

    @property
    def num_tets(self) -> int:
        return int(self.tet_idx.shape[0])

    @property
    def num_tris(self) -> int:
        return int(self.tri_idx.shape[0])

    def validate(self) -> None:
        """Raise ``ValueError`` if this body is internally inconsistent."""
        p = self.num_particles
        checks = [
            ("positions", self.positions, (p, 3), np.float32),
            ("inv_mass", self.inv_mass, (p,), np.float32),
            ("flags", self.flags, (p,), np.uint32),
            ("dist_idx", self.dist_idx, (self.num_dist, 2), np.int32),
            ("dist_rest", self.dist_rest, (self.num_dist,), np.float32),
            ("dist_kind", self.dist_kind, (self.num_dist,), np.int32),
            ("dist_color", self.dist_color, (self.num_dist,), np.int32),
            ("bend_idx", self.bend_idx, (self.num_bend, 4), np.int32),
            ("bend_rest", self.bend_rest, (self.num_bend,), np.float32),
            ("bend_color", self.bend_color, (self.num_bend,), np.int32),
            ("tet_idx", self.tet_idx, (self.num_tets, 4), np.int32),
            ("tet_dm_inv", self.tet_dm_inv, (self.num_tets, 3, 3), np.float32),
            ("tet_rest_volume", self.tet_rest_volume, (self.num_tets,), np.float32),
            ("tet_color", self.tet_color, (self.num_tets,), np.int32),
            ("tri_idx", self.tri_idx, (self.num_tris, 3), np.int32),
            ("uv", self.uv, (p, 2), np.float32),
        ]
        for name, arr, shape, dtype in checks:
            if arr.shape != shape:
                raise ValueError(
                    f"{self.name}.{name}: expected shape {shape}, got {arr.shape}")
            if arr.dtype != dtype:
                raise ValueError(
                    f"{self.name}.{name}: expected dtype {dtype}, got {arr.dtype}")
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(f"{self.name}.{name}: array must be C-contiguous")
        for name, idx in (("dist_idx", self.dist_idx), ("bend_idx", self.bend_idx),
                          ("tet_idx", self.tet_idx), ("tri_idx", self.tri_idx)):
            if idx.size and (idx.min() < 0 or idx.max() >= p):
                raise ValueError(f"{self.name}.{name}: index outside [0, {p})")
        if not np.isfinite(self.positions).all():
            raise ValueError(f"{self.name}.positions contains non-finite values")
        if (self.inv_mass < 0).any():
            raise ValueError(f"{self.name}.inv_mass must be non-negative")
        if self.num_tets and (self.tet_rest_volume <= 0).any():
            raise ValueError(f"{self.name}.tet_rest_volume must be strictly positive")
        if (self.skin_tet is None) != (self.skin_bary is None):
            raise ValueError(
                f"{self.name}: skin_tet and skin_bary go together or not at all")
        if self.skin_tet is not None and self.skin_bary is not None:
            if self.skin_tet.ndim != 2 or self.skin_tet.shape[0] != p:
                raise ValueError(
                    f"{self.name}.skin_tet: expected ({p}, K), got "
                    f"{self.skin_tet.shape}")
            k = int(self.skin_tet.shape[1])
            for name, arr, shape, dtype in (
                    ("skin_tet", self.skin_tet, (p, k), np.int32),
                    ("skin_bary", self.skin_bary, (p, k, 4), np.float32)):
                if arr.shape != shape or arr.dtype != dtype:
                    raise ValueError(
                        f"{self.name}.{name}: expected {shape} {dtype}, "
                        f"got {arr.shape} {arr.dtype}")
                if not arr.flags["C_CONTIGUOUS"]:
                    raise ValueError(f"{self.name}.{name} must be C-contiguous")
            used = self.skin_tet >= 0
            if used.any() and int(self.skin_tet[used].max()) >= self.num_tets:
                raise ValueError(
                    f"{self.name}.skin_tet references a tetrahedron past "
                    f"the end of tet_idx")
            if not np.isfinite(self.skin_bary).all():
                raise ValueError(f"{self.name}.skin_bary is not finite")
            if (np.abs(self.skin_bary[~used]) > 0).any():
                raise ValueError(
                    f"{self.name}.skin_bary carries weight in an unused slot")
            # Over all of a bound particle's slots the weights must sum to
            # one, or the skin scales with the element instead of following
            # it.
            bound = used.any(axis=1)
            sums = self.skin_bary[bound].sum(axis=(1, 2))
            if sums.size and not np.allclose(sums, 1.0, atol=1e-4):
                raise ValueError(
                    f"{self.name}.skin_bary rows do not sum to 1 "
                    f"(worst {np.abs(sums - 1.0).max():.2e})")


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FrameStats:
    """Per-frame timings, in milliseconds unless stated otherwise."""

    frame_ms: float = 0.0
    physics_ms: float = 0.0
    render_ms: float = 0.0
    tracking_ms: float = 0.0
    fps: float = 0.0
    tracking_fps: float = 0.0
    substeps: int = 0
    particles: int = 0
    constraints: int = 0
    contacts: int = 0
    grabbed: int = 0
    hands: int = 0
