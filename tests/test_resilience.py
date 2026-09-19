"""What an installation needs: the camera can go, and nobody may be there.

These drive the real application headless with a scripted stand-in for the
camera, because the failures they guard against are in the wiring between
the source, the tracker, the demo and the frame loop -- not inside any of
them.  A GPU and an OpenGL context are required.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from _harness import case, note, require, run

from fctx.config import preset
from fctx.core.types import HandFrame, TrackerFrame
from fctx.hands.sources import CameraUnavailable, HandSource


class ScriptedCamera(HandSource):
    """A 'camera' that shows a hand when told to and dies when told to."""

    is_synthetic = False

    def __init__(self) -> None:
        self.show_hand = False
        self.die = False
        self.started = False
        self.closed = False
        self._index = 0
        self.fps = 60.0

    @property
    def describe(self) -> str:
        return "scripted camera"

    def start(self) -> None:
        self.started = True

    def poll(self) -> TrackerFrame | None:
        if self.die:
            raise CameraUnavailable("the scripted camera was unplugged")
        self._index += 1
        hands = []
        if self.show_hand:
            image = np.zeros((21, 3), np.float32)
            image[:, 0] = 0.5 + np.linspace(-0.08, 0.08, 21)
            image[:, 1] = 0.5 + np.linspace(-0.06, 0.06, 21)
            world = np.zeros((21, 3), np.float32)
            world[:, 0] = np.linspace(-0.08, 0.08, 21)
            world[:, 1] = np.linspace(-0.06, 0.06, 21)
            hands.append(HandFrame(image=image, world=world, score=0.95))
        return TrackerFrame(hands=hands, timestamp=time.perf_counter(),
                            index=self._index, preview=None)

    def close(self) -> None:
        self.closed = True


def _app(scripted: ScriptedCamera, idle_demo: float = 0.0, retry: float = 0.05):
    """Build the real application with the scripted camera standing in.

    ``_build`` imports ``create_source`` from ``fctx.hands.sources`` at call
    time, so swapping the module attribute for the duration of construction
    is enough; the app keeps the swapped function for its own retries, and
    every non-camera request still goes to the real factory.
    """
    import fctx.hands.sources as sources
    from fctx.app import MatterStudio

    cfg = preset("cloth")
    cfg = dataclasses.replace(
        cfg, headless=True, lockstep=False, max_frames=0, idle_demo=idle_demo,
        render=dataclasses.replace(cfg.render, width=480, height=270,
                                   bloom=False, ssao_samples=0),
        tracking=dataclasses.replace(cfg.tracking, source="camera"))

    real_create = sources.create_source

    def create(cfg_tracking):  # noqa: ANN001
        if cfg_tracking.source == "camera":
            if scripted.die:
                raise CameraUnavailable("scripted camera absent")
            scripted.started = True
            return scripted
        return real_create(cfg_tracking)

    class Studio(MatterStudio):
        CAMERA_RETRY = retry

    sources.create_source = create
    try:
        app = Studio(cfg)
    finally:
        sources.create_source = real_create
    return app


def _frames(app, n: int) -> None:
    dt = app.clock.dt
    for _ in range(n):
        ctrl = app.controls.update(dt)
        if app.demo is not None:
            app._apply_demo(ctrl)
        app._apply_commands(ctrl)
        app._update_tracking(ctrl, dt)
        app.materials = app._with_effective_young(app._evaluate_materials(ctrl.hardness))
        app.state.upload_materials(app.materials)
        app._update_physics(ctrl, dt)


@case
def a_camera_that_dies_mid_session_is_replaced_and_then_taken_back() -> None:
    cam = ScriptedCamera()
    cam.show_hand = True
    app = _app(cam, retry=0.05)
    try:
        require(app.source is cam and app.live_source is cam,
                "the scripted camera was not the live source")
        _frames(app, 10)
        require(len(app._poses) == 1, "the camera's hand did not reach the tracker")

        cam.die = True
        _frames(app, 3)
        require(app.live_source is None, "a dead camera is still the live source")
        require(getattr(app.source, "is_synthetic", False),
                "the matter was not handed to the synthetic hand")
        require(cam.closed, "the dead camera was not closed")
        note("camera died: synthetic hand took over within 3 frames")

        # It comes back: the retry thread must pick it up and hand it over.
        cam.die = False
        cam.started = False
        cam.closed = False
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and app.live_source is not cam:
            _frames(app, 1)
            time.sleep(0.01)
        require(app.live_source is cam and app.source is cam,
                "the camera came back but was not taken back")
        _frames(app, 10)
        require(len(app._poses) == 1, "the returned camera's hand is not tracked")
        note("camera returned: taken back automatically")
    finally:
        app.close()


@case
def attract_mode_starts_when_nobody_is_there_and_stops_when_someone_is() -> None:
    cam = ScriptedCamera()
    cam.show_hand = False
    app = _app(cam, idle_demo=0.3)
    try:
        require(app.demo is None and not app._attract, "attract mode began at once")
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline and not app._attract:
            _frames(app, 1)
            time.sleep(0.005)
        require(app._attract, "attract mode never started with nobody there")
        require(app.demo is not None, "attract mode did not start the demonstration")
        require(getattr(app.source, "is_synthetic", False),
                "attract mode is not driving the synthetic hand")
        require(app.live_source is cam, "the camera was dropped during attract mode")
        note("attract mode after 0.3 s idle: demo running on the synthetic hand")

        cam.show_hand = True
        _frames(app, app.ATTRACT_WAKE_FRAMES + 2)
        require(not app._attract, "a hand in front of the camera did not end attract mode")
        require(app.demo is None, "the demonstration kept running with a person present")
        require(app.source is cam, "the stage was not handed back to the camera")
        note("a hand appeared: back to the camera within "
             f"{app.ATTRACT_WAKE_FRAMES + 2} frames")
    finally:
        app.close()


@case
def an_unplugged_camera_in_an_empty_room_still_plays_the_demonstration() -> None:
    """The exhibition case: camera dies, nobody is there, camera comes back."""
    cam = ScriptedCamera()
    cam.show_hand = False
    app = _app(cam, idle_demo=0.3, retry=0.05)
    try:
        _frames(app, 2)
        cam.die = True
        _frames(app, 3)
        require(app.live_source is None and not app._attract,
                "the camera did not die, or attract mode began early")

        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline and not app._attract:
            _frames(app, 1)
            time.sleep(0.005)
        require(app._attract and app.demo is not None,
                "no camera and nobody there, yet the demonstration did not start")
        note("camera lost with nobody there: the demonstration started anyway")

        # The camera returns to an empty room: the demo must keep the stage.
        cam.die = False
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and app.live_source is not cam:
            _frames(app, 1)
            time.sleep(0.01)
        require(app.live_source is cam, "the camera was not taken back")
        _frames(app, 5)
        require(app._attract and app.demo is not None,
                "the returned camera stopped the demonstration with nobody there")
        require(getattr(app.source, "is_synthetic", False),
                "the returned camera took the stage from the demonstration")
        note("camera back, still nobody: the demonstration kept the stage")

        cam.show_hand = True
        _frames(app, app.ATTRACT_WAKE_FRAMES + 2)
        require(not app._attract and app.demo is None and app.source is cam,
                "a hand in front of the returned camera did not end attract mode")
        note("a hand appeared: the camera has the stage again")
    finally:
        app.close()


@case
def a_synthetic_only_run_never_loses_its_hand_to_attract_mode() -> None:
    from fctx.app import MatterStudio

    cfg = preset("cloth")
    cfg = dataclasses.replace(
        cfg, headless=True, lockstep=False, max_frames=0, idle_demo=0.05,
        render=dataclasses.replace(cfg.render, width=480, height=270,
                                   bloom=False, ssao_samples=0),
        tracking=dataclasses.replace(cfg.tracking, source="synthetic"))
    app = MatterStudio(cfg)
    try:
        time.sleep(0.1)
        _frames(app, 5)
        require(not app._attract and app.demo is None,
                "attract mode took over a run that has no camera to wake it")
        note("no camera wanted: idle_demo is ignored")
    finally:
        app.close()


@case
def a_scheduled_restart_waits_until_nobody_is_there() -> None:
    cam = ScriptedCamera()
    cam.show_hand = True
    app = _app(cam, idle_demo=0.2)
    try:
        _frames(app, 5)
        require(not app._restart_due(), "a run with max_uptime 0 wanted to restart")
        app.cfg = dataclasses.replace(app.cfg, max_uptime=1.0)
        require(not app._restart_due(), "a young process wanted to restart")
        app._started_at -= 2 * 3600.0
        _frames(app, 2)
        require(not app._restart_due(), "restart with a hand in front of the camera")
        cam.show_hand = False
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline and not app._restart_due():
            _frames(app, 1)
            time.sleep(0.01)
        require(app._restart_due(), "nobody there for longer than idle_demo, yet no restart")
        note("restart is due only once the last hand is idle_demo old")

        # No camera at all: nobody can be there, so it goes at once.
        cam.die = True
        _frames(app, 3)
        require(app.live_source is None and app._restart_due(),
                "without a camera the restart still waited")
    finally:
        app.close()


@case
def a_resilient_run_survives_a_frame_that_raises() -> None:
    cam = ScriptedCamera()
    cam.show_hand = True
    app = _app(cam)
    app.cfg = dataclasses.replace(app.cfg, resilient=True)
    try:
        boom = RuntimeError("scripted failure")
        require(app._survive(boom), "a resilient run gave up on the first failure")
        require(app._failures == 1)
        for _ in range(app.MAX_CONSECUTIVE_FAILURES):
            app._survive(boom)
        require(not app._survive(boom),
                "a run failing every frame forever was not allowed to stop")
        app.cfg = dataclasses.replace(app.cfg, resilient=False)
        require(not app._survive(boom), "a non-resilient run swallowed an error")
        note(f"recovers up to {app.MAX_CONSECUTIVE_FAILURES} times, then stops")
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(run(__file__))
