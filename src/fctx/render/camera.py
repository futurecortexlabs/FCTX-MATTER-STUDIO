"""The virtual camera and the shadow-map light frustum.

Matrices are plain ``numpy`` 4x4 arrays in the usual mathematical convention,
so ``M @ v`` transforms a column vector.  GLSL wants column-major memory, so
:func:`mat_bytes` transposes on the way to the GPU and nothing else in the
renderer has to think about storage order.  ``pyglm`` would do the same job
faster, but a wrong matrix is the single most common rendering bug and a numpy
array can be printed, sliced and compared in a debugger.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import CameraConfig

__all__ = [
    "OrbitCamera",
    "look_at",
    "perspective",
    "orthographic",
    "mat_bytes",
]

#: Where the matter lives, from ARCHITECTURE.md section 3.
STAGE_CENTER = (0.0, 0.30, 0.10)
STAGE_RADIUS = 0.78


def mat_bytes(m: np.ndarray) -> bytes:
    """Pack a 4x4 numpy matrix for a GLSL ``mat4`` uniform."""
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("look_at: eye and target coincide")
    f = forward / n
    up = np.asarray(up, dtype=np.float64)
    s = np.cross(f, up)
    sn = np.linalg.norm(s)
    if sn < 1e-9:
        # Looking straight along the up axis: any side vector is valid, and
        # picking one is far better than emitting a matrix full of NaN.
        s = np.cross(f, np.array([1.0, 0.0, 0.0]))
        sn = np.linalg.norm(s)
        if sn < 1e-9:
            s = np.cross(f, np.array([0.0, 0.0, 1.0]))
            sn = np.linalg.norm(s)
    s = s / sn
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -np.array([s @ eye, u @ eye, -f @ eye])
    return m


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    if near <= 0.0 or far <= near:
        raise ValueError(f"perspective: bad depth range near={near} far={far}")
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(half_w: float, half_h: float, near: float, far: float) -> np.ndarray:
    if half_w <= 0.0 or half_h <= 0.0:
        raise ValueError(
            f"orthographic: extents must be positive, got {half_w}x{half_h}")
    if far <= near:
        raise ValueError(f"orthographic: bad depth range near={near} far={far}")
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1.0 / half_w
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


class OrbitCamera:
    """Look-at camera on a sphere around the stage.

    The scene is designed to be viewed head-on, so the orbit is clamped well
    short of the poles: looking at a sheet of cloth exactly edge-on or from
    directly underneath tells the viewer nothing and makes the shadow frustum
    degenerate.
    """

    MIN_PITCH = math.radians(-62.0)
    MAX_PITCH = math.radians(78.0)

    def __init__(self, cfg: CameraConfig, aspect: float = 16.0 / 9.0,
                 height: float | None = None) -> None:
        # Two call shapes: (cfg, aspect) and (cfg, width, height).  The second
        # is what the application has to hand -- it knows the window size, not
        # its ratio -- and making it compute the division at every call site is
        # how one of them ends up dividing by a zero-height minimised window.
        self.cfg = cfg
        self.aspect = (float(aspect) if height is None
                       else float(aspect) / max(float(height), 1.0))
        self.target = np.asarray(cfg.target, dtype=np.float64).copy()
        self.up = np.asarray(cfg.up, dtype=np.float64).copy()
        self.fov_y = float(cfg.fov_y)
        self.near = float(cfg.near)
        self.far = float(cfg.far)

        #: Key light direction: the direction light travels, so shading uses -l.
        self.light_dir = np.array([-0.42, -0.82, -0.38], dtype=np.float64)
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.stage_center = np.asarray(STAGE_CENTER, dtype=np.float64)
        self.stage_radius = STAGE_RADIUS

        self.distance = 1.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.reset()

    # -- state ------------------------------------------------------------

    def reset(self) -> None:
        """Return to the framing declared in :class:`CameraConfig`."""
        offset = np.asarray(self.cfg.position, dtype=np.float64) - self.target
        self.distance = float(max(np.linalg.norm(offset), 1e-3))
        self.pitch = float(math.asin(np.clip(offset[1] / self.distance, -1.0, 1.0)))
        self.yaw = float(math.atan2(offset[0], offset[2]))
        self.target = np.asarray(self.cfg.target, dtype=np.float64).copy()

    @property
    def position(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return self.target + self.distance * np.array([
            cp * math.sin(self.yaw),
            math.sin(self.pitch),
            cp * math.cos(self.yaw),
        ])

    def orbit(self, dx_pixels: float, dy_pixels: float) -> None:
        if not self.cfg.allow_orbit:
            return
        speed = self.cfg.orbit_speed * 0.01
        self.yaw -= dx_pixels * speed
        self.pitch = float(np.clip(self.pitch + dy_pixels * speed,
                                   self.MIN_PITCH, self.MAX_PITCH))

    def zoom(self, ticks: float) -> None:
        # Multiplicative so one wheel click covers the same visual fraction of
        # the distance whether you are close in or right back.
        self.distance = float(np.clip(
            self.distance * math.exp(-ticks * self.cfg.zoom_speed), 0.18, 12.0))

    def frame_stage(self, center: np.ndarray, radius: float) -> None:
        """Point at ``center`` and back off far enough to contain ``radius``."""
        self.target = np.asarray(center, dtype=np.float64).copy()
        self.stage_center = self.target.copy()
        self.stage_radius = float(max(radius, 1e-3))
        half_fov = math.tan(math.radians(self.fov_y) * 0.5)
        vertical = self.stage_radius / half_fov
        horizontal = self.stage_radius / (half_fov * max(self.aspect, 1e-6))
        self.distance = float(max(vertical, horizontal) * 1.18)

    # -- matrices ---------------------------------------------------------

    def view(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    def projection(self) -> np.ndarray:
        return perspective(self.fov_y, self.aspect, self.near, self.far)

    def view_proj(self) -> np.ndarray:
        return self.projection() @ self.view()

    #: Distance of the light's eye from the stage centre, in shadow extents.
    LIGHT_DISTANCE_SCALE = 2.2
    SHADOW_FAR_SCALE = 4.8
    #: Near plane of the light frustum, as a fraction of the shadow extent.
    #: An absolute near plane does not survive ``frame_stage`` being given a
    #: small stage: the far plane shrinks with the extent, eventually falls
    #: behind a fixed near, and ``orthographic`` then rejects the inverted
    #: range.  Scaling it keeps the whole light frustum similar at every stage
    #: size, so the depth precision per shadow texel does not change either.
    SHADOW_NEAR_SCALE = 0.019

    @property
    def shadow_extent(self) -> float:
        """Half-width, in metres, of the orthographic shadow frustum."""
        return max(self.stage_radius, 1e-4) * 1.35

    def light_view(self) -> np.ndarray:
        eye = self.stage_center - self.light_dir * (
            self.shadow_extent * self.LIGHT_DISTANCE_SCALE)
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(self.light_dir @ up)) > 0.98:
            up = np.array([0.0, 0.0, 1.0])
        return look_at(eye, self.stage_center, up)

    @property
    def shadow_near(self) -> float:
        return self.shadow_extent * self.SHADOW_NEAR_SCALE

    def light_projection(self) -> np.ndarray:
        radius = self.shadow_extent
        return orthographic(radius, radius, self.shadow_near,
                            radius * self.SHADOW_FAR_SCALE)

    @property
    def shadow_depth_range(self) -> float:
        """Metres spanned by the shadow map's 0..1 depth, for the depth bias."""
        return self.shadow_extent * self.SHADOW_FAR_SCALE - self.shadow_near

    def light_view_proj(self) -> np.ndarray:
        """Orthographic view-projection covering the stage, from the key light.

        The frustum is fitted to the stage sphere rather than to the visible
        geometry: a bound that changes size as the cloth swings makes the
        shadow texel size change with it, and the shimmer that produces is far
        more objectionable than the resolution a tight fit would buy.
        """
        return self.light_projection() @ self.light_view()

    def shadow_world_texel(self, shadow_size: int) -> float:
        """World-space width of one shadow-map texel, for the normal offset."""
        if shadow_size <= 0:
            return 0.0
        return (self.shadow_extent * 2.0) / float(shadow_size)
