"""Configuration for FCTX MATTER STUDIO.

Everything tunable lives here as frozen dataclasses with sane defaults, so a
scene can be reproduced exactly from a single :class:`AppConfig`.  The CLI in
:mod:`fctx.__main__` builds one of these and nothing else reads ``sys.argv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .core.types import MatterKind

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "assets"
MODEL_DIR = ASSET_DIR / "models"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

RECORDING_DIR = REPO_ROOT / "recordings"


# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """XPBD solver settings.

    ``substeps`` is the single most important number here.  XPBD converges far
    better with many small steps and one constraint iteration each than with
    one big step and many iterations (Macklin et al., "Small Steps in Physics
    Simulation", 2019), so the default leans hard on substeps.
    """

    #: Physics updates per second.  The render loop runs free of this.
    rate_hz: float = 90.0
    #: Substeps per physics update.
    substeps: int = 12
    #: Constraint iterations inside each substep.
    iterations: int = 1
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    #: Ground plane height, metres.
    ground_y: float = 0.0
    ground_friction: float = 0.55
    #: Radius of a cylindrical basin holding the matter in, metres.  0 leaves
    #: the floor open.  Only the granular scene needs one: position-based
    #: friction gives an angle of repose around 7 degrees instead of the 30-odd
    #: a real granular material has, so an unconfined pile spreads into a sheet
    #: and slides off the stage instead of staying somewhere you can stir it.
    basin_radius: float = 0.0
    #: How high the basin wall stands above the ground plane.  Above it the
    #: matter is free, so a hand can still lift a fistful straight out.
    basin_height: float = 0.0
    #: Particle-particle collision.  Cloth self-collision is the expensive one.
    self_collision: bool = True
    #: Extra separation kept between colliding particles, as a fraction of
    #: their radius.  A little slack keeps the contact solver from chattering.
    collision_margin: float = 0.12
    #: Hash grid resolution per axis; the grid covers ``world_extent``.
    grid_dim: int = 64
    world_extent: float = 3.0
    #: Speed limit, m/s.  Purely a safety net against a tracking glitch
    #: launching the whole body out of the scene.
    max_velocity: float = 12.0
    #: Positional correction limit per substep, in multiples of the particle
    #: radius.  Bounds how violently one substep can move anything.
    max_correction_ratio: float = 3.0
    #: Capture the substep loop into a CUDA graph.  Cuts tens of microseconds
    #: of launch overhead per kernel, which matters at 12 substeps x ~10
    #: kernels x 90 Hz.  Disable to get readable per-kernel profiles.
    use_cuda_graph: bool = True
    #: Run a finite-value check every N physics steps and reset any body that
    #: has gone non-finite.  0 disables the check.
    sanity_interval: int = 30
    #: Air drag, 1/s, applied to velocity.  Gives cloth its float.
    air_drag: float = 0.12
    #: Wind, m/s.  Off by default; the scene can turn it on.
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wind_turbulence: float = 0.0


@dataclass(frozen=True, slots=True)
class GrabConfig:
    """How a pinch takes hold of matter."""

    #: Particles within this distance of the pinch point get attached.
    radius: float = 0.075
    #: Hard cap on attached particles per hand, to bound the kernel cost.
    max_particles: int = 4096
    #: Pinch strength above which a grab starts, and below which it ends.
    #: The gap is deliberate hysteresis: without it a pinch held right at the
    #: threshold flickers and the object is dropped every few frames.
    start_threshold: float = 0.62
    release_threshold: float = 0.42
    #: Seconds the pinch must stay closed before the grab commits.  Stops a
    #: fast open-close gesture from snatching something by accident.
    hold_time: float = 0.045
    #: Fraction of the hand's velocity handed to the matter on release, so a
    #: flick actually throws.  Above 1 exaggerates for showmanship.
    throw_gain: float = 1.15
    #: Grabbed particles keep this fraction of their own mass; dropping it
    #: makes the held region follow the hand more tightly.
    mass_scale: float = 0.35
    #: Let the grab rotate the held region with the pinch frame.
    rotate_with_pinch: bool = True


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Hand tracking and how image space becomes world space."""

    source: str = "camera"          # camera | synthetic | replay | video
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 60
    #: Mirror the camera image.  On by default: people expect their hand to
    #: move right when they move it right.
    mirror: bool = True
    max_hands: int = 2
    min_detection_confidence: float = 0.55
    min_presence_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    #: Run MediaPipe on the GPU delegate when the model pack supports it.
    use_gpu_delegate: bool = False

    # -- One Euro filter, per landmark -------------------------------------
    #: Lower min_cutoff = smoother but laggier.  These values were tuned so a
    #: hand held still does not visibly jitter, while a fast swipe still lands
    #: within about one frame.
    filter_min_cutoff: float = 1.7
    #: The published 0.007-0.05 figures for beta assume pixels per second.
    #: These landmarks are metres, three orders of magnitude smaller, so a
    #: beta in that range never engages and the filter degenerates into a
    #: fixed 1.7 Hz low-pass: 100 mm of lag at a brisk 1.2 m/s. At 5.0 the
    #: same motion lags 14 mm, for about 9% more jitter at the filter output
    #: on a still hand -- 7x the lag bought back for a sixth of the noise.
    filter_beta: float = 5.0
    filter_d_cutoff: float = 1.2
    #: Drop a hand after this long without a detection.
    lost_timeout: float = 0.35
    #: Keep predicting a lost hand forward for this long before it disappears,
    #: which hides the one- or two-frame dropouts MediaPipe has under motion.
    coast_time: float = 0.12

    # -- image space -> world space ----------------------------------------
    #: Half-width of the interaction volume, metres.  The camera's field of
    #: view is mapped onto a box this wide at the reference depth.
    stage_half_width: float = 0.42
    stage_half_height: float = 0.26
    #: World-space z of a hand at the reference distance from the camera.
    stage_center_z: float = 0.10
    #: Metres of world z per unit of MediaPipe's relative depth.  MediaPipe's
    #: z is only loosely metric, so this is a feel parameter, not a calibration.
    depth_scale: float = 0.55
    #: Blend between depth from MediaPipe's z and depth from apparent hand
    #: size.  Hand size is noisier but does not drift; z is smooth but biased.
    depth_size_blend: float = 0.45
    #: Reference apparent hand span (normalised image units) at stage_center_z.
    reference_hand_span: float = 0.26
    #: Vertical offset of the interaction volume, metres.
    stage_center_y: float = 0.30
    #: Clamp world z to this range so a bad depth estimate cannot push the
    #: hand behind the camera or through the back wall.
    z_range: tuple[float, float] = (-0.32, 0.48)

    # -- sources other than the live camera --------------------------------
    replay_path: Path | None = None
    video_path: Path | None = None
    record_path: Path | None = None
    #: Loop a replay when it reaches the end.
    replay_loop: bool = True


@dataclass(frozen=True, slots=True)
class RenderConfig:
    width: int = 1600
    height: int = 900
    fullscreen: bool = False
    vsync: bool = True
    msaa: int = 4
    #: Shadow map resolution; 0 disables shadows.
    shadow_size: int = 2048
    shadow_softness: float = 1.4
    bloom: bool = True
    bloom_strength: float = 0.055
    bloom_threshold: float = 1.05
    #: Screen-space ambient occlusion sample count; 0 disables it.
    ssao_samples: int = 16
    ssao_radius: float = 0.09
    ssao_strength: float = 0.85
    exposure: float = 1.05
    #: ACES filmic tonemap.  Off falls back to Reinhard.
    aces: bool = True
    vignette: float = 0.30
    chromatic_aberration: float = 0.0016
    #: Draw the webcam feed with the landmark overlay in a corner.
    show_webcam: bool = True
    webcam_scale: float = 0.22
    show_hud: bool = True
    #: Draw the hand skeleton in 3D.
    show_hands: bool = True
    #: Tint hands while they are gripping.
    grab_highlight: bool = True
    background_top: tuple[float, float, float] = (0.045, 0.052, 0.075)
    background_bottom: tuple[float, float, float] = (0.012, 0.014, 0.022)
    #: Title bar text.
    title: str = "FCTX MATTER STUDIO"


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """The virtual camera looking at the stage."""

    position: tuple[float, float, float] = (0.0, 0.42, 1.18)
    target: tuple[float, float, float] = (0.0, 0.30, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_y: float = 42.0
    near: float = 0.02
    far: float = 24.0
    #: Let the mouse orbit the camera.  The scene is designed to be looked at
    #: head-on, but being able to swing round sells that it is really 3D.
    allow_orbit: bool = True
    orbit_speed: float = 0.32
    zoom_speed: float = 0.12


@dataclass(frozen=True, slots=True)
class SceneConfig:
    """Which body is on the stage and how finely it is discretised."""

    kind: MatterKind = MatterKind.CLOTH
    #: Starting position on the hardness dial.
    hardness: float = 0.35

    # cloth
    cloth_resolution: int = 72        # particles per side
    cloth_size: float = 0.60          # metres per side
    #: none | top_corners | two_points | top_edge | corners
    cloth_pinned: str = "two_points"
    #: Centre height of the sheet.  It has to sit near the hand's reach --
    #: the wrist spans stage_center_y +- stage_half_height, 0.04 to 0.56 m,
    #: and the fingers add about 0.18 m above that -- or most of the cloth is
    #: simply unreachable, which is the whole point of the app.  The default
    #: sheet spans 0.06 to 0.66 m, so its top strip is fingers-only; once it
    #: has settled, 98% of it is touchable.
    cloth_height: float = 0.36

    # soft body
    soft_shape: str = "sphere"        # sphere | box | torus
    #: Lattice cells along the longest axis.  The skin is a fitted voxel
    #: lattice, so this is what sets the facet size the eye reads: at 17 the
    #: shipped 0.26 m sphere has 15 mm facets and a 4.4 mm radial scatter, at
    #: 21 it has 12 mm facets and 3.6 mm, for 1.9x the tetrahedra and about a
    #: millisecond more per step -- which the 11.1 ms budget has.
    soft_resolution: int = 21
    soft_size: float = 0.26
    soft_height: float = 0.32

    # granular
    grain_count: int = 24000
    grain_radius: float = 0.0068
    grain_extent: float = 0.24
    grain_height: float = 0.40


@dataclass(frozen=True, slots=True)
class AppConfig:
    solver: SolverConfig = field(default_factory=SolverConfig)
    grab: GrabConfig = field(default_factory=GrabConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    device: str = "cuda:0"
    #: Exit after this many frames.  Used by the smoke tests; 0 means run
    #: until the window is closed.
    max_frames: int = 0
    #: Render into an offscreen buffer and never open a window.
    headless: bool = False
    #: Advance one physics step per frame off a virtual clock instead of
    #: following the wall clock.  A headless run finishes as fast as the GPU
    #: allows, so on the wall clock 600 frames is two seconds of simulation on
    #: one machine and six on another; in lockstep it is always 600/rate_hz
    #: seconds, which is what makes a captured frame or a smoke test
    #: reproducible.  ``--headless`` and ``--benchmark`` turn it on (see
    #: ``config_from_args``); a caller building an AppConfig directly gets
    #: this default and has to ask for it, which is why every smoke test
    #: passes it explicitly.
    lockstep: bool = False
    #: Start with the choreographed demonstration running (the ``D`` key).
    demo: bool = False
    #: Attract mode: after this many seconds without a hand in front of the
    #: camera, run the demonstration on the synthetic hand until someone
    #: shows up.  0 disables it.  Only meaningful with a camera source.
    idle_demo: float = 0.0
    #: Keep running through a frame that raises: log it, reset the scene, go
    #: on.  An exhibition must not die at the first bad frame; a developer
    #: wants the traceback.  Off means raise.
    resilient: bool = False
    #: Restart hygiene for an installation: after this many hours of uptime,
    #: exit cleanly at the next moment nobody is interacting, so that a
    #: supervisor loop (``run_kiosk.bat``) starts a fresh process.  0 never.
    max_uptime: float = 0.0
    #: Append a log of what happened -- source changes, resets, errors -- to
    #: this file.
    log_file: Path | None = None
    #: Write a PNG of frame ``screenshot_frame`` here, then keep going.
    screenshot: Path | None = None
    screenshot_frame: int = 60
    verbose: bool = False
    #: Print a per-kernel profile every N seconds.  0 disables it.
    profile_interval: float = 0.0

    def with_scene(self, **kwargs: object) -> AppConfig:
        return replace(self, scene=replace(self.scene, **kwargs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------


def preset(name: str) -> AppConfig:
    """Return a ready-made configuration.

    ``fctx-matter-studio --preset <name>`` uses these; they are also what the
    number keys 1-5 switch between at runtime.
    """
    base = AppConfig()
    if name == "cloth":
        return replace(base, scene=replace(
            base.scene, kind=MatterKind.CLOTH, cloth_resolution=72,
            cloth_pinned="two_points", hardness=0.30))
    if name == "banner":
        return replace(
            base,
            scene=replace(base.scene, kind=MatterKind.CLOTH, cloth_resolution=88,
                          cloth_size=0.66, cloth_height=0.40,
                          cloth_pinned="top_edge", hardness=0.22),
            solver=replace(base.solver, wind=(0.35, 0.0, -0.15),
                           wind_turbulence=0.55))
    if name == "drape":
        return replace(base, scene=replace(
            base.scene, kind=MatterKind.CLOTH, cloth_resolution=80,
            cloth_pinned="none", cloth_height=0.46, hardness=0.28))
    if name == "soft":
        return replace(base, scene=replace(
            base.scene, kind=MatterKind.SOFT, soft_shape="sphere",
            soft_resolution=21, hardness=0.40))
    if name == "cube":
        return replace(base, scene=replace(
            base.scene, kind=MatterKind.SOFT, soft_shape="box",
            soft_resolution=12, hardness=0.55))
    if name == "torus":
        return replace(base, scene=replace(
            base.scene, kind=MatterKind.SOFT, soft_shape="torus",
            soft_resolution=14, hardness=0.35))
    if name == "grain":
        return replace(
            base,
            scene=replace(base.scene, kind=MatterKind.GRAIN, hardness=0.5,
                          grain_count=24000, grain_radius=0.0048,
                          grain_extent=0.21, grain_height=0.13),
            solver=replace(base.solver, basin_radius=0.30, basin_height=0.13),
            # The matter in this scene lives on the floor rather than at chest
            # height, so the interaction volume has to come down with it or the
            # hand can only ever reach the very top of the pile.
            tracking=replace(base.tracking, stage_center_y=0.19,
                             stage_half_height=0.20))
    raise ValueError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")


PRESETS = ("cloth", "banner", "drape", "soft", "cube", "torus", "grain")

#: What the number keys cycle through at runtime.
PRESET_KEYS = ("cloth", "banner", "soft", "cube", "grain")

#: Solver and tracking fields that belong to the session rather than to the
#: scene.  The CLI and the hardware set these, so switching preset at runtime
#: must leave them alone; everything else in those two configs is the
#: preset's to decide.
SESSION_SOLVER_FIELDS = (
    "rate_hz", "substeps", "iterations", "self_collision", "use_cuda_graph",
    "grid_dim", "sanity_interval",
)
SESSION_TRACKING_FIELDS = (
    "source", "camera_index", "camera_width", "camera_height", "camera_fps",
    "mirror", "max_hands", "use_gpu_delegate", "min_detection_confidence",
    "min_presence_confidence", "min_tracking_confidence",
    "replay_path", "video_path", "record_path", "replay_loop",
)


def switch_preset(current: AppConfig, name: str) -> AppConfig:
    """Config for pressing a number key: ``name``'s scene on this session.

    A preset is a whole scene, not just its geometry.  ``grain`` is the one
    that proves it: its basin is a ``SolverConfig`` field and its lowered
    interaction volume is a ``TrackingConfig`` field, and carrying only
    ``scene`` across left the granular pile with no container -- the exact
    failure ``SolverConfig.basin_radius`` exists to prevent -- with the hand
    still reaching at chest height above it.  Carrying the *new* preset's
    values wholesale matters just as much in reverse: a field the incoming
    preset leaves at its default has to go back to that default, or the
    basin stays standing as an invisible cylinder in the cloth scene that
    replaces it.

    What survives is what the preset has no business knowing: the hardness
    the user has dialled in, and the session's own settings (which camera,
    how many substeps, whether a CUDA graph is captured).
    """
    new = preset(name)
    keep_solver = {f: getattr(current.solver, f) for f in SESSION_SOLVER_FIELDS}
    keep_track = {f: getattr(current.tracking, f) for f in SESSION_TRACKING_FIELDS}
    return replace(
        current,
        scene=replace(new.scene, hardness=current.scene.hardness),
        solver=replace(new.solver, **keep_solver),
        tracking=replace(new.tracking, **keep_track),
    )


def env_device(default: str = "cuda:0") -> str:
    """Honour ``FCTX_DEVICE`` so CI can force the CPU path."""
    return os.environ.get("FCTX_DEVICE", default)
