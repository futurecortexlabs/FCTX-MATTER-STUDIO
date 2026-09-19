"""CPU-side mesh helpers.

Everything here is small, static geometry built once at startup: the capsule
shell the hand skeleton is instanced from, a unit quad for the 2D overlay, and
the floor plane.  Simulated matter never passes through this module -- its
vertices live on the GPU and are written by the solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Mesh",
    "CapsuleShell",
    "uv_sphere",
    "capsule_shell",
    "unit_quad",
    "grid_plane",
]


@dataclass(frozen=True, slots=True)
class Mesh:
    positions: np.ndarray   # (V, 3) float32
    normals: np.ndarray     # (V, 3) float32
    uvs: np.ndarray         # (V, 2) float32
    indices: np.ndarray     # (F, 3) int32

    def __post_init__(self) -> None:
        v = self.positions.shape[0]
        if self.normals.shape != (v, 3) or self.uvs.shape != (v, 2):
            raise ValueError("Mesh attribute arrays disagree on vertex count")
        if self.indices.size and int(self.indices.max()) >= v:
            raise ValueError("Mesh index out of range")

    @property
    def num_vertices(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.indices.shape[0])


@dataclass(frozen=True, slots=True)
class CapsuleShell:
    """A unit sphere split so its two halves can be pulled apart into a capsule.

    ``side`` is 0 for vertices belonging to the first endpoint and 1 for the
    second.  The equator ring is present twice, once with each side value, so
    the strip between them stretches into the cylindrical waist as the two
    endpoints separate.  ``directions`` are unit vectors in the capsule's local
    frame, which doubles as the surface normal.
    """

    directions: np.ndarray  # (V, 3) float32, |d| == 1
    side: np.ndarray        # (V,)   float32, 0.0 or 1.0
    indices: np.ndarray     # (F, 3) int32

    @property
    def num_vertices(self) -> int:
        return int(self.directions.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.indices.shape[0])


def uv_sphere(rings: int = 16, segments: int = 24, radius: float = 1.0) -> Mesh:
    """A latitude/longitude sphere with duplicated seam vertices."""
    if rings < 2 or segments < 3:
        raise ValueError(f"uv_sphere: rings={rings} segments={segments} too small")

    lat = np.linspace(0.0, math.pi, rings + 1)
    lon = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    la, lo = np.meshgrid(lat, lon, indexing="ij")

    dirs = np.stack([
        np.sin(la) * np.sin(lo),
        np.cos(la),
        np.sin(la) * np.cos(lo),
    ], axis=-1).reshape(-1, 3)

    uvs = np.stack([lo / (2.0 * math.pi), 1.0 - la / math.pi],
                   axis=-1).reshape(-1, 2)

    tris: list[tuple[int, int, int]] = []
    stride = segments + 1
    for i in range(rings):
        for j in range(segments):
            a = i * stride + j
            b = a + stride
            tris.append((a, b, a + 1))
            tris.append((a + 1, b, b + 1))

    return Mesh(
        positions=np.ascontiguousarray(dirs * radius, dtype=np.float32),
        normals=np.ascontiguousarray(dirs, dtype=np.float32),
        uvs=np.ascontiguousarray(uvs, dtype=np.float32),
        indices=np.asarray(tris, dtype=np.int32).reshape(-1, 3),
    )


def capsule_shell(rings: int = 12, segments: int = 20) -> CapsuleShell:
    """Build the instanced capsule shell described by :class:`CapsuleShell`."""
    if rings < 2 or rings % 2 != 0:
        raise ValueError(f"capsule_shell: rings must be even and >= 2, got {rings}")
    if segments < 3:
        raise ValueError(f"capsule_shell: segments must be >= 3, got {segments}")

    half = rings // 2
    # Latitudes from the +Y pole down to the equator, then equator to -Y pole.
    # The equator angle appears in both halves, which is what creates the waist.
    upper = np.linspace(0.0, math.pi * 0.5, half + 1)
    lower = np.linspace(math.pi * 0.5, math.pi, half + 1)
    lat = np.concatenate([upper, lower])
    side = np.concatenate([np.ones(half + 1), np.zeros(half + 1)])

    lon = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    dirs = np.stack([
        np.sin(la) * np.sin(lo),
        np.cos(la),
        np.sin(la) * np.cos(lo),
    ], axis=-1).reshape(-1, 3)
    side_flat = np.repeat(side[:, None], segments + 1, axis=1).reshape(-1)

    tris: list[tuple[int, int, int]] = []
    stride = segments + 1
    rows = lat.shape[0]
    for i in range(rows - 1):
        for j in range(segments):
            a = i * stride + j
            b = a + stride
            tris.append((a, b, a + 1))
            tris.append((a + 1, b, b + 1))

    return CapsuleShell(
        directions=np.ascontiguousarray(dirs, dtype=np.float32),
        side=np.ascontiguousarray(side_flat, dtype=np.float32),
        indices=np.asarray(tris, dtype=np.int32).reshape(-1, 3),
    )


def unit_quad() -> np.ndarray:
    """Four corners of the unit square, as a triangle strip: (0,0) (1,0) (0,1) (1,1)."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    dtype=np.float32)


def grid_plane(size: float, subdivisions: int = 1, height: float = 0.0,
               center: tuple[float, float] = (0.0, 0.0)) -> Mesh:
    """A horizontal plane of ``size`` metres, centred on ``center`` at ``height``."""
    if size <= 0.0:
        raise ValueError(f"grid_plane: size must be positive, got {size}")
    n = max(1, int(subdivisions))
    t = np.linspace(-0.5, 0.5, n + 1)
    gx, gz = np.meshgrid(t * size + center[0], t * size + center[1], indexing="ij")
    pos = np.stack([gx, np.full_like(gx, height), gz], axis=-1).reshape(-1, 3)
    uv = np.stack(np.meshgrid(t + 0.5, t + 0.5, indexing="ij"),
                  axis=-1).reshape(-1, 2)

    tris: list[tuple[int, int, int]] = []
    stride = n + 1
    for i in range(n):
        for j in range(n):
            a = i * stride + j
            b = a + stride
            # Wound counter-clockwise seen from +Y so the plane faces up.
            tris.append((a, a + 1, b))
            tris.append((a + 1, b + 1, b))

    normals = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32),
                      (pos.shape[0], 1))
    return Mesh(
        positions=np.ascontiguousarray(pos, dtype=np.float32),
        normals=np.ascontiguousarray(normals, dtype=np.float32),
        uvs=np.ascontiguousarray(uv, dtype=np.float32),
        indices=np.asarray(tris, dtype=np.int32).reshape(-1, 3),
    )
