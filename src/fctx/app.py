"""The application: camera in, matter out.

One frame is: drain input, pull the newest tracked hands, re-evaluate the
material from the hardness dial, run the physics at its own fixed rate, draw.
The dial is re-uploaded every single frame, which is the entire reason you can
turn jelly into rubber while your fingers are still holding it.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from .config import RECORDING_DIR, AppConfig, switch_preset
from .core.clock import FixedTimestep, PerfMonitor, Rolling, Stopwatch
from .core.material import DEFAULT_MATERIALS, Material, evaluate
from .core.types import FrameStats, HandPose, MatterKind
from .demo import Choreography
from .interaction import GripManager, Notifier
from .ui.controls import SYNTHETIC_HELP_LINES, Controls, ControlState


class MatterStudio:
    """Owns every subsystem and the frame loop that drives them."""

    #: What the wind toggle uses on a preset that ships with still air.
    DEFAULT_WIND = (0.32, 0.02, -0.14)
    DEFAULT_TURBULENCE = 0.6
    #: Frames between device-synchronising reads of the solver counters.
    STATS_INTERVAL = 6
    #: How far towards the eye, and how far sideways as a fraction of the
    #: hand's reach, a free body is assumed to be carried when framing.
    CARRY_DEPTH = 0.25
    CARRY_WIDTH = 0.7
    #: Seconds between attempts to open the camera while it is missing.
    CAMERA_RETRY = 3.0
    #: Frames the camera must show a hand for before attract mode yields.
    ATTRACT_WAKE_FRAMES = 3

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._closed = False
        self._wind_on = cfg.solver.wind != (0.0, 0.0, 0.0)
        self._solver_stats: dict[str, int] = {}
        try:
            self._build(cfg)
        except BaseException:
            # Anything that fails after the window exists -- an unwritable
            # --record path is the easy one to reach -- leaves CUDA-registered
            # GL buffers alive with nobody holding a reference, and the
            # process then tears the GL context down underneath the CUDA
            # driver.  close() is written to survive a half-built object:
            # every step is tried separately and an attribute that was never
            # assigned is just another exception it swallows.
            self.close()
            raise

    def _build(self, cfg: AppConfig) -> None:
        import warp as wp

        wp.config.quiet = not cfg.verbose
        wp.init()
        self.wp = wp
        self.device = self._resolve_device(cfg.device)

        from .bodies import build_scene
        from .hands.sources import CameraUnavailable, create_source
        from .hands.tracker import HandTracker
        from .render.camera import OrbitCamera
        from .render.context import Window
        from .render.pipeline import Renderer
        from .solver.solver import XPBDSolver
        from .solver.state import SolverState

        self._create_source = create_source
        self._CameraUnavailable = CameraUnavailable

        self.bodies = build_scene(cfg.scene)
        for body in self.bodies:
            body.validate()
        self.state = SolverState(self.bodies, cfg, self.device)
        self.solver = XPBDSolver(self.state, cfg)

        self.window = Window(cfg.render, headless=cfg.headless)
        self.camera = OrbitCamera(cfg.camera, cfg.render.width, cfg.render.height)
        # The renderer takes the window, not its bare context: a screenshot
        # requested after the buffers have swapped needs the window to ask for
        # a re-composite, and only the window knows its own framebuffer size.
        self.renderer = Renderer(self.window, self.state, self.bodies, cfg)

        self._frame_camera()

        self.tracker = HandTracker(cfg.tracking, cfg.grab)
        self.source, self.source_note = self._open_source()

        self.controls = Controls(ControlState(
            hardness_target=cfg.scene.hardness,
            hardness=cfg.scene.hardness,
            show_hud=cfg.render.show_hud,
            show_webcam=cfg.render.show_webcam,
            show_hands=cfg.render.show_hands,
            # Every toggle has to start where the config already is, or the
            # first _apply_commands reads the dataclass default as if the user
            # had just pressed the key: --preset banner shipped with wind and
            # was told "WIND OFF" over a dead-still banner on frame one.
            wind=self._wind_on,
            # GLFW only reports a size when it *changes*, so a window that is
            # never resized keeps whatever this says -- and the synthetic
            # hand's pointer is the cursor divided by it.  On a 1280x720
            # window the default put the screen centre at 0.4, 0.4.  Window
            # coordinates, not the framebuffer: that is what the cursor
            # callback and the RESIZE event both speak in.
            _window_size=tuple(getattr(self.window, "_window_size",
                                       self.window.size)),
        ))
        self.grips = GripManager(cfg.grab, cfg.tracking.max_hands)
        self.notify = Notifier()
        self.clock = FixedTimestep(cfg.solver.rate_hz)
        self.perf = PerfMonitor()
        self._tet_ceiling = self.state.tet_stiffness_ceiling(self.clock.dt)
        #: The choreographed demonstration, when it is running.
        self.demo: Choreography | None = None
        self._demo_started = 0.0
        self._demo_caption = ""
        self._preset_name: str | None = None
        #: Optional hook called after every frame is drawn and before it is
        #: presented, with (frame_no, simulation_time).  The showcase encoder
        #: reads the framebuffer from it.
        self.on_frame = None
        self.physics_ms = Rolling(90)
        self.render_ms = Rolling(90)
        self.tracking_ms = Rolling(90)

        self.recorder = None
        if cfg.tracking.record_path is not None:
            self._start_recording(cfg.tracking.record_path)

        self.materials: list[Material] = self._evaluate_materials(
            cfg.scene.hardness)
        self.state.upload_materials(self.materials)

        self._poses: list[HandPose] = []
        self._preview: np.ndarray | None = None
        self._last_frame_index = -1
        self._frame_no = 0
        self._virtual_time = 0.0
        self._polls_due = 0
        self._last_synth_input: tuple | None = None
        if cfg.lockstep:
            # A source that paces itself against the wall clock would emit a
            # handful of frames across a whole headless run; switching it to
            # its virtual clock makes one poll mean one tracker frame.
            if hasattr(self.source, "realtime"):
                self.source.realtime = False
        self._wall_start = time.perf_counter()
        self._last_profile = self._wall_start
        self._pending_screenshot: Path | None = None

        # Last, once the clocks it reads exist.
        if cfg.demo:
            self._start_demo()

        if cfg.verbose:
            print(self.describe())

    # -- setup helpers -----------------------------------------------------

    def _resolve_device(self, requested: str) -> str:
        wp = self.wp
        if requested.startswith("cuda") and wp.get_cuda_device_count() == 0:
            print("fctx: no CUDA device visible, falling back to the CPU "
                  "solver -- expect single-digit frame rates.")
            return "cpu"
        return requested

    def _open_source(self) -> tuple[object, str]:
        """Open the configured hand source, falling back to synthetic.

        The live source (a camera or a video) and the synthetic fallback are
        kept as two objects, because an installation needs both at once: the
        fallback drives the matter while the camera is unplugged, and it
        drives the attract-mode demonstration while the camera sees nobody.
        Either way the camera keeps being polled, so the moment a hand shows
        up it takes over.
        """
        cfg = self.cfg.tracking
        self.live_source = None
        self._camera_wanted = cfg.source == "camera"
        self._camera_seen = False   # ever had a live camera: "lost" vs "none"
        self._retry_thread: threading.Thread | None = None
        self._retry_result: object | None = None
        self._retry_at = 0.0
        self._hand_last_seen = time.perf_counter()
        self._started_at = time.perf_counter()
        self._attract = False
        self._wake_frames = 0
        self._synthetic = None
        try:
            source = self._create_source(cfg)
            source.start()
            if cfg.source in ("camera", "video"):
                self.live_source = source
                self._camera_seen = True
                return source, source.describe
            return source, source.describe
        except self._CameraUnavailable as exc:
            print(f"fctx: {exc}")
            print("fctx: falling back to the synthetic hand -- "
                  "move the mouse to steer it, click or press space to pinch."
                  + ("  Retrying the camera in the background."
                     if self._camera_wanted else ""))
        except FileNotFoundError as exc:
            print(f"fctx: {exc}")
            print("fctx: falling back to the synthetic hand.")
        source = self._synthetic_source()
        self._retry_at = time.perf_counter() + self.CAMERA_RETRY
        return source, source.describe + " (camera unavailable)"

    def _synthetic_source(self):
        if self._synthetic is None:
            self._synthetic = self._create_source(
                replace(self.cfg.tracking, source="synthetic"))
            self._synthetic.start()
            if self.cfg.lockstep and hasattr(self._synthetic, "realtime"):
                self._synthetic.realtime = False
        return self._synthetic

    # -- camera hot-plug -----------------------------------------------------

    def _camera_lost(self, exc: BaseException) -> None:
        """The live camera died mid-session; carry on without it."""
        print(f"fctx: camera lost: {exc}")
        try:
            if self.live_source is not None:
                self.live_source.close()
        except Exception:  # noqa: BLE001 -- the device is already gone
            pass
        self.live_source = None
        self._preview = None
        if self.source is not self._synthetic:
            self.source = self._synthetic_source()
        self.source_note = self.source.describe + (
            "  (attract mode, camera lost)" if self._attract else " (camera lost)")
        self._retry_at = time.perf_counter() + self.CAMERA_RETRY
        self.notify.post("CAMERA LOST -- synthetic hand until it returns", 3.0)

    def _poll_camera_retry(self) -> None:
        """Try to reopen the camera, off the frame loop, every few seconds.

        Probing a missing camera takes a third of a second or more, which on
        the frame loop is a visible hitch every retry.  The attempt runs on a
        thread and hands the started source back; the source's own capture
        thread and MediaPipe callbacks do not care which thread created them.
        """
        if not self._camera_wanted or self.live_source is not None:
            return
        if self._retry_thread is not None:
            if self._retry_thread.is_alive():
                return
            result, self._retry_result, self._retry_thread = self._retry_result, None, None
            if result is not None:
                self.live_source = result
                self._camera_seen = True
                self._hand_last_seen = time.perf_counter()
                self.tracker = type(self.tracker)(self.cfg.tracking, self.cfg.grab)
                if self._attract:
                    # Nobody has arrived, so the demonstration keeps the stage;
                    # the returned camera is what will notice when someone does.
                    self.source_note = result.describe + "  (attract mode)"
                else:
                    self.source = result
                    self.source_note = result.describe
                self.notify.post("CAMERA BACK", 2.5)
                print(f"fctx: camera back: {result.describe}")
            else:
                self._retry_at = time.perf_counter() + self.CAMERA_RETRY
            return
        if time.perf_counter() < self._retry_at:
            return

        def attempt() -> None:
            try:
                source = self._create_source(self.cfg.tracking)
                source.start()
                self._retry_result = source
            except Exception:  # noqa: BLE001 -- still absent; try again later
                self._retry_result = None

        self._retry_thread = threading.Thread(
            target=attempt, name="fctx-camera-retry", daemon=True)
        self._retry_thread.start()

    # -- attract mode ---------------------------------------------------------

    def _poll_attract(self, live_hands: int, now: float) -> None:
        """Run the demonstration on the synthetic hand while nobody is there.

        ``idle_demo`` seconds without a hand in front of the camera starts the
        choreography on the synthetic hand; the camera keeps being watched,
        and the first person to hold a hand up gets the stage back within a
        few frames.  The camera-lost fallback uses the same switch, so an
        unplugged camera in an exhibition plays the demo rather than showing
        a frozen sheet.
        """
        idle = float(self.cfg.idle_demo)
        if idle <= 0.0 or not self._camera_wanted:
            # Without a camera to watch there is nobody to wake it up, and a
            # synthetic-only run would lose its hand to the demonstration.
            return
        if live_hands > 0:
            self._hand_last_seen = now
            self._wake_frames += 1
        else:
            self._wake_frames = 0

        if not self._attract:
            if now - self._hand_last_seen >= idle:
                self._attract = True
                self.source = self._synthetic_source()
                camera = (self.live_source.describe if self.live_source is not None
                          else "camera lost" if self._camera_seen else "no camera")
                self.source_note = camera + "  (attract mode)"
                if self.demo is None:
                    self._start_demo()
        elif self._wake_frames >= self.ATTRACT_WAKE_FRAMES:
            self._attract = False
            self._wake_frames = 0
            if self.demo is not None:
                self._stop_demo()
            self.source = self.live_source
            self.source_note = self.live_source.describe
            self.tracker = type(self.tracker)(self.cfg.tracking, self.cfg.grab)
            self.notify.post("WELCOME", 1.5)

    def _frame_camera(self) -> None:
        """Point the camera at the matter, but keep the hand's reach on screen.

        Framing the body alone looks tighter, and then the top of a raised hand
        leaves the frame -- which reads as the tracking having lost it. The
        vertical span therefore covers both, and at 16:9 the horizontal field
        that buys is already wider than the hand can travel sideways.
        """
        track = self.cfg.tracking
        basin = float(self.cfg.solver.basin_radius)
        lo = np.array([np.inf, track.stage_center_y - track.stage_half_height,
                       np.inf])
        hi = np.array([-np.inf, track.stage_center_y + track.stage_half_height,
                       -np.inf])
        for body in self.bodies:
            lo = np.minimum(lo, body.positions.min(axis=0))
            hi = np.maximum(hi, body.positions.max(axis=0))
            if not (body.inv_mass == 0.0).any():
                # Nothing holds this body up, so it is going to end up on the
                # floor. Framing its starting bounds would leave the settled
                # pile off the bottom of the screen a second after launch.
                lo[1] = min(lo[1], self.cfg.solver.ground_y)
                # And it is going to be carried.  A hand can pick a free body
                # up and put it down anywhere it reaches, and what it puts
                # down *nearer the eye* projects further down the screen than
                # its height says: the autopilot set a cube down 0.32 m
                # forward and its front bottom corner rendered a quarter of a
                # frame below the edge.  Cover the near half of the reach and
                # most of its width; behind and above take care of themselves.
                hi[2] = max(hi[2], track.stage_center_z + self.CARRY_DEPTH)
                span = track.stage_half_width * self.CARRY_WIDTH
                lo[0] = min(lo[0], -span)
                hi[0] = max(hi[0], span)
                if basin > 0.0:
                    # Same argument sideways: a free body in a basin spreads
                    # until it meets the wall, and the granular pile starts
                    # packed well inside it.
                    lo[0] = min(lo[0], -basin)
                    lo[2] = min(lo[2], -basin)
                    hi[0] = max(hi[0], basin)
                    hi[2] = max(hi[2], basin)
        centre = (lo + hi) * 0.5
        # frame_stage treats the radius as the vertical half-extent, so a wide
        # flat body is framed by its height and the aspect ratio covers the
        # width; a tall one is framed by its height directly.
        # The 6% is for the pitch: the camera looks slightly down, so a point
        # at the top of the stage projects higher than its half-extent alone
        # predicts, and framing it exactly puts it a percent outside the frame.
        radius = 1.06 * max(
            float(hi[1] - lo[1]) * 0.5,
            float(hi[0] - lo[0]) * 0.5 / max(self.camera.aspect, 1e-3),
            0.05)
        self.camera.frame_stage(centre, max(radius, self._depth_radius(lo, hi)))

    def _depth_radius(self, lo: np.ndarray, hi: np.ndarray) -> float:
        """Smallest framing radius that also contains the scene's *near* face.

        The half-extent rule above measures the stage as if it were flat.  A
        deep scene is not: the granular basin is 0.6 m across, so its near rim
        sits 0.35 m from a camera 0.65 m from the centre, and a grain there
        projects almost half again as far down the screen as its height alone
        predicts -- with the shipped basin, 8.5% of the settled pile renders
        off the bottom edge.  Solving the containment for each corner of the
        bounding box costs nothing and, unlike padding the radius, adds only
        what the depth actually needs: every preset but ``grain`` is already
        far enough back and comes out of here unchanged.
        """
        cam = self.camera
        # Rows of the view matrix are the camera basis; row 2 points from the
        # target back towards the eye, so a positive component is nearer.
        basis = np.asarray(cam.view(), dtype=np.float64)[:3, :3]
        tan_half = math.tan(math.radians(cam.fov_y) * 0.5)
        aspect = max(cam.aspect, 1e-3)
        half = (hi - lo) * 0.5
        needed = 0.0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    x_c, y_c, near = basis @ (half * (sx, sy, sz))
                    needed = max(needed, near + max(abs(y_c) / tan_half,
                                                    abs(x_c) / (tan_half * aspect)))
        # frame_stage backs off to 1.18 * radius / (tan * min(1, aspect)); go
        # back through that so the number handed over means the same thing it
        # does on the other branch.
        return needed * tan_half * min(1.0, aspect) / 1.18

    def _evaluate_materials(self, hardness: float) -> list[Material]:
        return [evaluate(DEFAULT_MATERIALS[b.kind], hardness) for b in self.bodies]

    def _start_recording(self, path: Path) -> None:
        from .hands.recording import Recorder

        path = path if path.is_absolute() else RECORDING_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.recorder = Recorder(path)
        self.notify.post(f"RECORDING -> {path.name}", 2.5)
        print(f"fctx: recording hands to {path}")

    def _stop_recording(self) -> None:
        if self.recorder is None:
            return
        path = self.recorder.path
        frames = self.recorder.close()
        self.recorder = None
        if frames:
            self.notify.post(f"SAVED {frames} frames", 2.5)
            print(f"fctx: wrote {frames} frames to {path}")
        else:
            # Recorder.close() writes nothing for an empty take, so claiming
            # a file exists would send the user looking for one that does not.
            self.notify.post("RECORDING CANCELLED (no hands seen)", 2.5)

    # -- the frame loop ----------------------------------------------------

    def run(self) -> str | None:
        cfg = self.cfg
        # An exact == against the frame counter wrote nothing, and said
        # nothing, whenever the run was too short to reach it -- a capture
        # script could not tell that it had asked for frame 180 of a 30-frame
        # run.  Frame 0 is the natural way to ask for the first frame and
        # could never match either.  Clamping into range means the file
        # always appears; the frame it shows is the closest one that exists.
        shot_frame = max(1, cfg.screenshot_frame)
        if cfg.max_frames:
            shot_frame = min(shot_frame, cfg.max_frames)
        if cfg.screenshot is not None and shot_frame != cfg.screenshot_frame:
            print(f"fctx: --screenshot-frame {cfg.screenshot_frame} is outside "
                  f"this run; capturing frame {shot_frame} instead.")
        while not self.window.should_close:
            self.perf.begin_frame()
            frame_dt = (self.clock.dt if cfg.lockstep
                        else max(self.perf.frame.last / 1000.0, 1e-4))

            ctrl = self.controls.handle(self.window.poll_events())
            ctrl = self.controls.update(frame_dt)
            if ctrl.quit_requested:
                break
            if ctrl.demo_toggle_requested:
                self._stop_demo() if self.demo else self._start_demo()
            if self.demo is not None:
                self._apply_demo(ctrl)
            self._apply_commands(ctrl)

            try:
                with Stopwatch(self.tracking_ms):
                    self._update_tracking(ctrl, frame_dt)

                self.materials = self._with_effective_young(
                    self._evaluate_materials(ctrl.hardness))
                self.state.upload_materials(self.materials)

                with Stopwatch(self.physics_ms):
                    self._update_physics(ctrl, frame_dt)
            except Exception as exc:  # noqa: BLE001 -- see _survive
                if not self._survive(exc):
                    raise
                continue
            # "Consecutive" means a good frame clears the count; a bad frame
            # every few minutes is a nuisance to log, not a reason to quit.
            self._failures = 0

            with Stopwatch() as drawn:
                self._draw(ctrl)
            if self.on_frame is not None:
                self.on_frame(self._frame_no, self._virtual_time)
            # Prefer the GPU's own measurement. With vsync on, the driver
            # blocks inside a GL call once the queue is full, and a wall-clock
            # timer charges that wait to rendering -- which turns a 1 ms frame
            # into a 15 ms one and makes the readout useless for tuning.
            gpu = getattr(self.renderer, "gpu_ms", 0.0)
            self.render_ms.push(gpu if gpu > 0.0 else drawn.ms)

            # Save before the swap: after it the back buffer is undefined and
            # the renderer has to re-composite the whole frame to recover it.
            if cfg.screenshot and self._frame_no + 1 >= shot_frame:
                self._save_screenshot(cfg.screenshot)
                cfg = self.cfg = replace(self.cfg, screenshot=None)
            if self._pending_screenshot is not None:
                self._save_screenshot(self._pending_screenshot)
                self._pending_screenshot = None

            self.window.swap()
            self._frame_no += 1

            if cfg.profile_interval:
                self._maybe_profile()
            if cfg.max_frames and self._frame_no >= cfg.max_frames:
                break
            if self._restart_due():
                break

        return self._benchmark_report() if self.cfg.headless else None

    def _apply_commands(self, ctrl: ControlState) -> None:
        if ctrl.reset_requested:
            # XPBDSolver.reset already resets the state; calling both wiped
            # every device array twice for one key press.
            self.solver.reset()
            self.grips.reset()
            self.notify.post("RESET")
        if ctrl.preset_requested:
            self._switch_preset(ctrl.preset_requested)
        if ctrl.fullscreen_requested:
            self.window.toggle_fullscreen()
        if ctrl.screenshot_requested:
            self._pending_screenshot = self._next_capture_path()
        if ctrl.record_toggle_requested:
            if self.recorder is None:
                self._start_recording(self._next_recording_path())
            else:
                self._stop_recording()
        self.camera.orbit(*ctrl.orbit_delta)
        self.camera.zoom(ctrl.zoom_delta)
        self.renderer.set_wireframe(ctrl.wireframe)
        self._apply_wind(ctrl.wind)

    # -- the choreographed demonstration -----------------------------------

    def _now(self) -> float:
        return self._virtual_time if self.cfg.lockstep else time.perf_counter()

    def _start_demo(self) -> None:
        self.demo = Choreography()
        self._demo_started = self._now()
        self.controls.state.demo = True
        self.controls.state.auto_sweep = False
        # The synthetic hand hands itself to its autopilot after a second and
        # a half of unchanged input.  Holding still is the whole point of the
        # showpiece -- the dial moves while the hand does not -- so a cue
        # that stays put would be read as idleness and the hand would wander
        # off with the cloth still attached.
        if hasattr(self.source, "auto"):
            self._source_auto = bool(self.source.auto)
            self.source.auto = False
        self.notify.post("DEMO  press D to stop", 2.5)

    def _stop_demo(self) -> None:
        self.demo = None
        self._demo_caption = ""
        self.controls.state.demo = False
        if hasattr(self.source, "auto"):
            self.source.auto = getattr(self, "_source_auto", True)
        self.notify.post("DEMO OFF")

    def _apply_demo(self, ctrl: ControlState) -> None:
        """Drive the hand and the dial from the choreography.

        The cue overwrites the same control fields the mouse and keyboard
        write, so everything downstream -- the synthetic source, the dial
        smoothing, the HUD -- behaves exactly as it does for a person.  On a
        camera source only the dial follows; the hand is the person's.
        """
        assert self.demo is not None
        t = self._now() - self._demo_started
        if t > self.demo.duration + 1.5:
            # Loop: put the matter back and start again, so a demo left
            # running in a window keeps showing the point.
            self.solver.reset()
            self.grips.reset()
            self._demo_started = self._now()
            t = 0.0
        cue = self.demo.cue(t)
        ctrl.pointer = (cue.nx, cue.ny)
        ctrl.pointer_depth = cue.depth
        ctrl.synth_pinch = cue.pinch
        ctrl.synth_curl = cue.curl
        ctrl.hardness_target = cue.hardness
        self._demo_caption = cue.caption
        current = self._preset_name or self._guess_preset_name()
        if cue.preset != current:
            ctrl.preset_requested = cue.preset

    def _guess_preset_name(self) -> str:
        kind = self.bodies[0].kind if self.bodies else MatterKind.CLOTH
        return {MatterKind.CLOTH: "cloth", MatterKind.SOFT: "soft",
                MatterKind.GRAIN: "grain"}[kind]

    def _with_effective_young(self, materials: list[Material]) -> list[Material]:
        """Tell the HUD what the tetrahedra are really delivering.

        Above the resolvable ceiling the solver softens the element pair and
        the edge constraints carry the rest, so the number on the dial is
        what was asked for, not what is simulated.  Quoting it unqualified
        would be the one dishonest readout on the screen.
        """
        out = []
        for i, mat in enumerate(materials):
            if mat.params.kind is MatterKind.SOFT and i < len(self._tet_ceiling):
                mu_cap = float(self._tet_ceiling[i, 0])
                if np.isfinite(mu_cap) and mu_cap < mat.lame_mu:
                    mat = replace(mat, young_effective=mat.young * mu_cap / mat.lame_mu)
            out.append(mat)
        return out

    def _apply_wind(self, on: bool) -> None:
        """Turn wind on or off without rebuilding anything.

        A preset that ships with no wind still needs something to switch on,
        so the toggle falls back to a default breeze rather than doing nothing.
        """
        if on == self._wind_on:
            return
        self._wind_on = on
        base = self.cfg.solver.wind
        turbulence = self.cfg.solver.wind_turbulence
        if on and base == (0.0, 0.0, 0.0):
            base, turbulence = self.DEFAULT_WIND, self.DEFAULT_TURBULENCE
        self.state.set_wind(base if on else (0.0, 0.0, 0.0),
                            turbulence if on else 0.0)
        self.notify.post("WIND ON" if on else "WIND OFF")

    def _switch_preset(self, name: str) -> None:
        from .bodies import build_scene
        from .hands.tracker import HandTracker
        from .solver.solver import XPBDSolver
        from .solver.state import SolverState

        cfg = switch_preset(
            replace(self.cfg, scene=replace(
                self.cfg.scene, hardness=self.controls.state.hardness_target)),
            name)
        self.grips.reset()
        self.bodies = build_scene(cfg.scene)
        for body in self.bodies:
            body.validate()
        self.state = SolverState(self.bodies, cfg, self.device)
        self.solver = XPBDSolver(self.state, cfg)
        self.renderer.rebind(self.state, self.bodies)
        self.cfg = cfg
        self._preset_name = name
        self._tet_ceiling = self.state.tet_stiffness_ceiling(self.clock.dt)
        # The tracker holds its own copy of the tracking config, and that is
        # what image-to-world projection reads: a preset that moves the
        # interaction volume does not reach the hand until this is rebuilt.
        # Its live tracks are discarded with it, which is right -- they were
        # projected into the old volume.
        self.tracker = HandTracker(cfg.tracking, cfg.grab)
        self._poses = []
        self._frame_camera()
        self._wind_on = cfg.solver.wind != (0.0, 0.0, 0.0)
        # The toggle follows the preset, for the same reason it is seeded from
        # the config at startup: leaving it where the previous scene left it
        # made the next _apply_commands switch the new preset's wind straight
        # back off, and post "WIND OFF" over a banner the user just chose.
        self.controls.state.wind = self._wind_on
        self.materials = self._evaluate_materials(self.controls.state.hardness)
        self.state.upload_materials(self.materials)
        self.notify.post(f"{name.upper()}  "
                         f"{self.state.num_particles:,} particles")

    def _update_tracking(self, ctrl: ControlState, dt: float) -> None:
        source = self.source
        if hasattr(source, "set_pointer"):
            # Only push when something actually moved. The synthetic source
            # hands over to its autopilot after a second and a half of
            # unchanged manual input, and re-sending the same cursor position
            # every frame reads as continuous input -- which suppresses the
            # autopilot forever and leaves the hand frozen mid-stage.
            wanted = (ctrl.pointer, ctrl.pointer_depth,
                      ctrl.synth_pinch, ctrl.synth_curl)
            if wanted != self._last_synth_input:
                self._last_synth_input = wanted
                source.set_pointer(*ctrl.pointer)
                source.set_depth(ctrl.pointer_depth)
                source.set_pinch(ctrl.synth_pinch)
                source.set_curl(ctrl.synth_curl)

        # The live camera is polled whether or not it is driving the matter:
        # in attract mode, and while a lost camera is being retried, it is
        # what decides when a person has arrived.
        live_frame = None
        if self.live_source is not None and self.live_source is not source:
            try:
                live_frame = self.live_source.poll()
            except self._CameraUnavailable as exc:
                self._camera_lost(exc)
        self._poll_camera_retry()
        source = self.source

        try:
            if self.cfg.lockstep:
                # Poll on the virtual clock at the source's own rate, so the
                # hand does not move 1.5x too fast just because physics runs
                # at 90 Hz and the tracker claims 60.
                rate = float(getattr(source, "fps", self.cfg.tracking.camera_fps) or 60)
                due = int(self._virtual_time * rate) + 1
                frame = source.poll() if due > self._polls_due else None
                self._polls_due = max(self._polls_due, due)
            else:
                frame = source.poll()
        except self._CameraUnavailable as exc:
            self._camera_lost(exc)
            frame = None

        if source is self.live_source and frame is not None:
            live_frame = frame
        if live_frame is not None and live_frame.preview is not None:
            self._preview = live_frame.preview
        self._poll_attract(len(live_frame.hands) if live_frame is not None else 0,
                           time.perf_counter())

        if frame is not None and frame.index != self._last_frame_index:
            self._last_frame_index = frame.index
            self.perf.note_tracking_frame()
            if frame.preview is not None:
                self._preview = frame.preview
            if self.recorder is not None:
                self.recorder.add(frame)
        else:
            frame = None

        now = self._virtual_time if self.cfg.lockstep else time.perf_counter()
        self._poses = self.tracker.update(frame, now)

    def _update_physics(self, ctrl: ControlState, frame_dt: float) -> None:
        dt = self.clock.dt
        # Capsules are uploaded once per *frame*, so the elapsed time between
        # two uploads is the frame delta, not the physics step.  Handing
        # set_hands the physics dt scales every capsule and pinch velocity by
        # rate_hz / frame_hz -- a constant 1.5x over-report on a 60 Hz display
        # at the shipped 90 Hz physics, which drags the cloth 9% further
        # sideways for the same physical hand motion -- and it measures the
        # collider fade-in in frames rather than seconds.  Clamped the same
        # way the physics clock clamps a stalled frame: set_hands also ages
        # the collider fade by this value, and a one-second hitch that lands
        # on the frame a hand first appears would otherwise skip the fade
        # entirely and materialise a full-radius hand inside the matter.
        self.solver.set_hands(self._poses, min(frame_dt, self.clock.max_delta))
        self.grips.update(self._poses, frame_dt, self.solver)

        steps = 1 if self.cfg.lockstep else self.clock.tick()
        if ctrl.paused:
            steps = 1 if ctrl.step_once else 0
        self._virtual_time += steps * dt
        for _ in range(steps):
            self.solver.step(dt)
        if steps or not ctrl.paused:
            self.solver.compute_normals()

        # stats() reads a device counter, which costs a synchronisation. The
        # HUD does not need it every frame, and at 90 Hz nobody can read a
        # number that changes faster than this anyway.
        if self._frame_no % self.STATS_INTERVAL == 0:
            self._solver_stats = self.solver.stats()

    def _draw(self, ctrl: ControlState) -> None:
        stats = FrameStats(
            frame_ms=self.perf.frame.mean,
            physics_ms=self.physics_ms.mean,
            render_ms=self.render_ms.mean,
            tracking_ms=self.tracking_ms.mean,
            fps=self.perf.fps,
            tracking_fps=self.perf.tracking_fps,
            substeps=self.cfg.solver.substeps,
            particles=self.state.num_particles,
            constraints=self.state.num_constraints,
            contacts=self._solver_stats.get("contacts", 0),
            grabbed=self.grips.total_held,
            hands=len(self._poses),
        )
        self.renderer.draw(
            camera=self.camera,
            poses=self._poses if ctrl.show_hands else [],
            materials=self.materials,
            preview=self._preview if ctrl.show_webcam else None,
            stats=stats,
            hud_lines=self._hud_lines(ctrl),
            hardness=ctrl.hardness,
            notifications=self.notify.update(self.perf.frame.last / 1000.0),
            show_hud=ctrl.show_hud,
            paused=ctrl.paused,
        )

    #: Consecutive failed frames before a resilient run gives up.
    MAX_CONSECUTIVE_FAILURES = 30

    #: With no attract mode configured, this long without a hand counts as
    #: "nobody there" for a scheduled restart.
    RESTART_IDLE = 10.0

    def _restart_due(self, now: float | None = None) -> bool:
        """Is it time for the scheduled restart, and is nobody watching?

        A long-running process is best restarted on a schedule -- drivers,
        allocators and the odd library all creep -- but never in front of a
        person: the exit waits until no hand has been seen for as long as
        attract mode waits (or :attr:`RESTART_IDLE` when there is none).
        Without a camera nobody can be there, so it goes at once.
        """
        hours = float(self.cfg.max_uptime)
        if hours <= 0.0:
            return False
        now = time.perf_counter() if now is None else now
        uptime = now - self._started_at
        if uptime < hours * 3600.0:
            return False
        if self.live_source is not None:
            idle = float(self.cfg.idle_demo) or self.RESTART_IDLE
            if now - self._hand_last_seen < idle:
                return False
        self.log(f"scheduled restart after {uptime / 3600.0:.2f} h of uptime")
        return True

    def _survive(self, exc: BaseException) -> bool:
        """Decide whether a frame that raised ends the run.

        In an exhibition the answer is no: log it with its traceback, put the
        matter back, and carry on.  Thirty failures in a row is not a bad
        frame, it is a broken installation, and then the answer is yes so
        that a supervisor script can restart the process.
        """
        if not self.cfg.resilient:
            return False
        self._failures = getattr(self, "_failures", 0) + 1
        self.log(f"frame {self._frame_no} raised {exc!r}", exc=exc)
        if self._failures > self.MAX_CONSECUTIVE_FAILURES:
            self.log("too many consecutive failures; giving up")
            return False
        try:
            self.solver.reset()
            self.grips.reset()
            self.notify.post("RECOVERED FROM AN ERROR -- scene reset", 3.0)
        except Exception as inner:  # noqa: BLE001
            self.log(f"reset after a failure also raised {inner!r}", exc=inner)
        self._frame_no += 1
        return True

    def log(self, message: str, exc: BaseException | None = None) -> None:
        """Print, and append to the log file when there is one."""
        print(f"fctx: {message}")
        path = self.cfg.log_file
        if path is None:
            return
        try:
            with Path(path).open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")
                if exc is not None:
                    import traceback

                    fh.write("".join(traceback.format_exception(exc)))
        except OSError:
            pass

    def _hud_lines(self, ctrl: ControlState) -> list[str]:
        # The renderer already draws a control strip along the bottom of the
        # frame, so repeating HELP_LINES here would just be a second copy
        # covering the matter. Only what the strip cannot say goes in.
        lines = [self.source_note]
        if getattr(self.source, "is_synthetic", False):
            lines.extend(SYNTHETIC_HELP_LINES)
        if self.demo is not None:
            lines.append("DEMO" + (f":  {self._demo_caption}" if self._demo_caption else "")
                         + "   (D to stop)")
        elif ctrl.auto_sweep:
            lines.append("AUTO-SWEEP ON -- the dial is moving on its own")
        if self.recorder is not None:
            lines.append(f"REC {self.recorder.count} frames")
        return lines

    # -- output ------------------------------------------------------------

    def _next_capture_path(self) -> Path:
        directory = Path("captures")
        directory.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return directory / f"fctx-{stamp}.png"

    def _next_recording_path(self) -> Path:
        RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return RECORDING_DIR / f"take-{stamp}.fhr"

    def _save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.window.save_png(path)
        self.notify.post(f"SAVED {path.name}", 2.0)
        print(f"fctx: wrote {path}")

    def _maybe_profile(self) -> None:
        now = time.perf_counter()
        if now - self._last_profile < self.cfg.profile_interval:
            return
        self._last_profile = now
        print(f"[{self._frame_no:6d}] "
              f"{self.perf.fps:6.1f} fps | "
              f"phys {self.physics_ms.mean:6.2f} ms "
              f"(p95 {self.physics_ms.percentile(0.95):6.2f}) | "
              f"draw {self.render_ms.mean:6.2f} ms | "
              f"track {self.tracking_ms.mean:5.2f} ms "
              f"@ {self.perf.tracking_fps:4.1f} Hz | "
              f"contacts {self._solver_stats.get('contacts', 0):6d} | "
              f"held {self.grips.total_held:4d}")

    def describe(self) -> str:
        s = self.state
        body_desc = ", ".join(
            f"{b.name} ({b.num_particles:,}p)" for b in self.bodies)
        return (
            f"FCTX MATTER STUDIO\n"
            f"  device      {self.device}\n"
            f"  scene       {body_desc}\n"
            f"  particles   {s.num_particles:,}\n"
            f"  distance    {s.num_dist:,} in {len(s.dist_batches)} colours\n"
            f"  bending     {s.num_bend:,} in {len(s.bend_batches)} colours\n"
            f"  tetrahedra  {s.num_tet:,} in {len(s.tet_batches)} colours\n"
            f"  solver      {self.cfg.solver.rate_hz:.0f} Hz x "
            f"{self.cfg.solver.substeps} substeps"
            f"{' (cuda graph)' if self.cfg.solver.use_cuda_graph else ''}\n"
            f"  hands       {self.source_note}"
        )

    def _benchmark_report(self) -> str:
        elapsed = time.perf_counter() - self._wall_start
        return (
            "\n"
            "benchmark\n"
            f"  frames        {self._frame_no}\n"
            f"  wall clock    {elapsed:.2f} s\n"
            f"  mean fps      {self._frame_no / max(elapsed, 1e-6):.1f}\n"
            f"  frame  mean   {self.perf.frame.mean:.2f} ms "
            f"(p95 {self.perf.frame.percentile(0.95):.2f})\n"
            f"  physics mean  {self.physics_ms.mean:.2f} ms "
            f"(p95 {self.physics_ms.percentile(0.95):.2f})\n"
            f"  render  mean  {self.render_ms.mean:.2f} ms "
            f"(p95 {self.render_ms.percentile(0.95):.2f})\n"
            f"  tracking mean {self.tracking_ms.mean:.2f} ms\n"
            f"  particles     {self.state.num_particles:,}\n"
            f"  constraints   {self.state.num_constraints:,}\n"
            # In lockstep the clock is never ticked -- one physics step per
            # frame, off a virtual clock -- so a zero here would be a
            # structural zero, not a measurement.  --benchmark --realtime is
            # the run that can actually drop a step.
            f"  dropped steps {'n/a (lockstep)' if self.cfg.lockstep else self.clock.dropped}\n"
        )

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Order matters: the renderer holds CUDA-registered GL buffers, and
        # unregistering them after the GL context is gone raises
        # "invalid OpenGL or DirectX context" out of the CUDA driver.
        for shutdown in (self._stop_recording,
                         lambda: self.source.close(),
                         lambda: self.live_source.close() if self.live_source is not None else None,
                         lambda: self._synthetic.close() if self._synthetic is not None else None,
                         lambda: self.renderer.release(),
                         lambda: self.window.close()):
            try:
                shutdown()
            except Exception as exc:  # noqa: BLE001
                if self.cfg.verbose:
                    print(f"fctx: during shutdown: {exc!r}")


def run(cfg: AppConfig) -> str | None:
    app = MatterStudio(cfg)
    try:
        return app.run()
    finally:
        app.close()
