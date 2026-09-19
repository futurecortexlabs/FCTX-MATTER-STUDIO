"""GPU buffers shared between the solver and the renderer.

There is exactly one position VBO and one normal VBO for the whole scene,
covering every particle of every body.  Bodies do not own vertex data; they
own an index buffer of *global* particle indices into those two VBOs.  That
is what lets the solver write the entire scene's vertices in one
device-to-device copy with no per-body bookkeeping and no host round trip.

The copy goes through :class:`warp.RegisteredGLBuffer`.  When CUDA-OpenGL
interop is unavailable -- no CUDA device, a headless CI box, a laptop running
the GL context on a different GPU from the CUDA one -- the same API falls back
to staging through host memory.  The path in use is visible as
:attr:`ParticleBuffers.interop`, because the two have very different frame
costs and a silent downgrade would look like a physics regression.
"""

from __future__ import annotations

import gc
from typing import Any

import moderngl
import numpy as np

__all__ = ["ParticleBuffers", "InteropError"]


class InteropError(RuntimeError):
    """Raised when a CUDA-OpenGL mapping is used incorrectly."""


def _as_numpy(array: Any, name: str, count: int) -> np.ndarray:
    """Coerce a warp or numpy (N, 3) buffer to a contiguous float32 host array."""
    if isinstance(array, np.ndarray):
        host = array
    elif hasattr(array, "numpy"):
        host = array.numpy()
    else:
        raise TypeError(f"{name}: expected a numpy or warp array, got {type(array)!r}")
    host = np.ascontiguousarray(host, dtype=np.float32).reshape(-1, 3)
    if host.shape[0] < count:
        raise ValueError(
            f"{name}: holds {host.shape[0]} vectors but the scene has {count} "
            "particles")
    return host[:count]


class ParticleBuffers:
    """The scene-wide position and normal VBOs, plus their Warp mapping."""

    def __init__(
        self,
        ctx: moderngl.Context,
        num_particles: int,
        device: str = "cuda:0",
        *,
        prefer_interop: bool = True,
    ) -> None:
        if num_particles <= 0:
            raise ValueError(
                f"ParticleBuffers: num_particles must be positive, got {num_particles}")
        self.ctx = ctx
        self.num_particles = int(num_particles)
        self.device = device
        self._released = False
        self._mapped: tuple[Any, Any] | None = None

        nbytes = self.num_particles * 3 * 4
        self.positions = ctx.buffer(reserve=nbytes, dynamic=True)
        self.normals = ctx.buffer(reserve=nbytes, dynamic=True)
        # Start from a defined state: an unwritten VBO holds whatever the
        # driver last had in that memory, and a frame drawn before the first
        # solver step would otherwise scatter triangles across the scene.
        zeros = np.zeros((self.num_particles, 3), dtype=np.float32)
        up = np.tile(np.array([0.0, 1.0, 0.0], np.float32), (self.num_particles, 1))
        self.positions.write(zeros)
        self.normals.write(up)

        self._reg_pos: Any = None
        self._reg_nrm: Any = None
        self._host_pos: Any = None
        self._host_nrm: Any = None
        self.interop = False
        self.interop_error: str | None = None

        if prefer_interop:
            self._try_register()
        if not self.interop:
            self._make_host_staging()

    # -- setup ------------------------------------------------------------

    def _try_register(self) -> None:
        try:
            import warp as wp
        except Exception as exc:
            self.interop_error = f"warp unavailable: {exc}"
            return
        try:
            wp.init()
            dev = wp.get_device(self.device)
            if not dev.is_cuda:
                self.interop_error = f"device {self.device!r} is not a CUDA device"
                return
            # fallback_to_copy would silently paper over a failed registration;
            # the host path below is the one this module wants to control.
            flags = wp.RegisteredGLBuffer.NONE
            self._reg_pos = wp.RegisteredGLBuffer(
                self.positions.glo, dev, flags, fallback_to_copy=False)
            self._reg_nrm = wp.RegisteredGLBuffer(
                self.normals.glo, dev, flags, fallback_to_copy=False)
        except Exception as exc:
            self._reg_pos = None
            self._reg_nrm = None
            self.interop_error = f"{type(exc).__name__}: {exc}"
            return
        self.interop = True

    def _make_host_staging(self) -> None:
        try:
            import warp as wp

            dev = wp.get_device(self.device)
        except Exception:
            self._host_pos = np.zeros((self.num_particles, 3), dtype=np.float32)
            self._host_nrm = np.zeros((self.num_particles, 3), dtype=np.float32)
            return
        self._host_pos = wp.zeros(self.num_particles, dtype=wp.vec3, device=dev)
        self._host_nrm = wp.zeros(self.num_particles, dtype=wp.vec3, device=dev)

    # -- per-frame --------------------------------------------------------

    def map_for_warp(self) -> tuple[Any, Any]:
        """Return ``(positions, normals)`` as writable ``warp`` arrays.

        On the interop path these alias GL memory directly and must be released
        with :meth:`unmap` before any draw call reads the buffers -- the CUDA
        driver owns them until then and GL is free to return stale data.
        """
        if self._released:
            raise InteropError("ParticleBuffers.map_for_warp after release()")
        if self._mapped is not None:
            raise InteropError("ParticleBuffers.map_for_warp called while mapped")
        if self.interop:
            import warp as wp

            pos = self._reg_pos.map(dtype=wp.vec3, shape=(self.num_particles,))
            nrm = self._reg_nrm.map(dtype=wp.vec3, shape=(self.num_particles,))
        else:
            pos, nrm = self._host_pos, self._host_nrm
        self._mapped = (pos, nrm)
        return pos, nrm

    def unmap(self) -> None:
        """Hand the buffers back to OpenGL."""
        if self._mapped is None:
            raise InteropError("ParticleBuffers.unmap called while not mapped")
        self._mapped = None
        if self.interop:
            self._reg_pos.unmap()
            self._reg_nrm.unmap()
        else:
            self.positions.write(
                _as_numpy(self._host_pos, "positions", self.num_particles))
            self.normals.write(
                _as_numpy(self._host_nrm, "normals", self.num_particles))

    def update_from(self, x: Any, normal: Any) -> None:
        """Copy solver state into the VBOs, using whichever path is active.

        ``x`` and ``normal`` may be warp arrays (device to device) or numpy
        arrays (host upload); the renderer must not care which, because the
        fake solver state used by the tests supplies the latter.
        """
        if self._mapped is not None:
            raise InteropError("ParticleBuffers.update_from called while mapped")

        x_is_warp = not isinstance(x, np.ndarray) and hasattr(x, "numpy")
        n_is_warp = not isinstance(normal, np.ndarray) and hasattr(normal, "numpy")

        if self.interop and x_is_warp and n_is_warp:
            import warp as wp

            dst_pos, dst_nrm = self.map_for_warp()
            try:
                wp.copy(dst_pos, x, count=self.num_particles)
                wp.copy(dst_nrm, normal, count=self.num_particles)
            finally:
                self.unmap()
            return

        self.positions.write(_as_numpy(x, "positions", self.num_particles))
        self.normals.write(_as_numpy(normal, "normals", self.num_particles))

    # -- teardown ---------------------------------------------------------

    def release(self) -> None:
        """Unregister from CUDA and free the GL objects, in that order.

        ``RegisteredGLBuffer`` only unregisters in ``__del__``, so the
        references are dropped and a collection is forced here.  Leaving it to
        interpreter shutdown means the GL context is already gone by then and
        the CUDA driver raises "invalid OpenGL or DirectX context".
        """
        if self._released:
            return
        self._released = True
        if self._mapped is not None:
            try:
                self.unmap()
            except Exception:
                self._mapped = None
        self._reg_pos = None
        self._reg_nrm = None
        gc.collect()
        self.positions.release()
        self.normals.release()

    def describe(self) -> str:
        if self.interop:
            return f"CUDA-GL interop on {self.device}"
        reason = self.interop_error or "disabled"
        return f"host staging ({reason})"
