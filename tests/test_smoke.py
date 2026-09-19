"""End to end: the whole application, headless, on the synthetic hand.

Every other test file exercises one subsystem against fakes.  This one runs
the real thing -- geometry, GPU solver, tracking, renderer, HUD and the frame
loop -- because the failures that actually stop the app from starting live in
the wiring between them, not inside any of them.

Needs a GPU and an OpenGL context.  It takes about a minute.
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

import numpy as np
from _harness import case, note, require, run  # noqa: E402

from fctx.config import PRESET_KEYS, AppConfig, preset  # noqa: E402
from fctx.core.types import MatterKind  # noqa: E402

CAPTURES = Path(__file__).resolve().parents[1] / "captures"


def _headless(name: str = "cloth", **overrides: object) -> AppConfig:
    cfg = preset(name)
    render = dataclasses.replace(cfg.render, width=1280, height=720)
    return dataclasses.replace(
        cfg, render=render, headless=True, lockstep=True,
        tracking=dataclasses.replace(cfg.tracking, source="synthetic"),
        **overrides)  # type: ignore[arg-type]


def _run(cfg: AppConfig):
    from fctx.app import MatterStudio

    app = MatterStudio(cfg)
    try:
        app.run()
        return app, app.state.snapshot()
    finally:
        app.close()


@case
def the_application_starts_runs_and_shuts_down_cleanly() -> None:
    cfg = _headless(max_frames=180)
    start = time.perf_counter()
    app, snap = _run(cfg)
    elapsed = time.perf_counter() - start
    require(app._frame_no == 180, f"ran {app._frame_no} frames, expected 180")
    require(np.isfinite(snap["x"]).all(), "positions went non-finite")
    note(f"180 frames in {elapsed:.2f} s, "
         f"{app.perf.fps:.0f} fps, physics {app.physics_ms.mean:.2f} ms, "
         f"render {app.render_ms.mean:.2f} ms")


@case
def lockstep_makes_a_headless_run_reproducible() -> None:
    # Without this the synthetic hand and the physics both chase the wall
    # clock, so the same command produces a different frame on every machine
    # and a captured image can never be compared against anything.
    runs = [_run(_headless(max_frames=90))[1]["x"] for _ in range(2)]
    delta = float(np.abs(runs[0] - runs[1]).max())
    require(delta == 0.0,
            f"two identical headless runs diverged by {delta:.3e} m")
    note("two 90-frame runs are bit-identical")


@case
def every_preset_runs() -> None:
    for name in PRESET_KEYS:
        app, snap = _run(_headless(name, max_frames=60))
        require(np.isfinite(snap["x"]).all(), f"{name} went non-finite")
        pos = snap["x"]
        require(np.abs(pos).max() < 3.0,
                f"{name} threw matter {np.abs(pos).max():.2f} m from the origin")
        note(f"{name:7s} {app.state.num_particles:6,} particles  "
             f"{app.physics_ms.mean:5.2f} ms physics  "
             f"{app.render_ms.mean:5.2f} ms render")


@case
def the_hardness_dial_reaches_the_solver_every_frame() -> None:
    from fctx.app import MatterStudio

    cfg = _headless(max_frames=0)
    app = MatterStudio(cfg)
    try:
        seen = []
        for target in (0.0, 1.0, 0.5):
            app.controls.state.hardness_target = target
            app.controls.state.hardness = target
            for _ in range(20):
                app._update_physics(app.controls.state, 1 / 90)
            app.materials = app._evaluate_materials(target)
            app.state.upload_materials(app.materials)
            seen.append(float(app.state.mat_stretch.numpy()[0]))
        require(seen[0] > seen[1],
                f"compliance did not fall as hardness rose: {seen}")
        require(seen[1] < seen[2] < seen[0],
                f"the middle of the dial is not between its ends: {seen}")
        note("stretch compliance at hardness 0 / 1 / 0.5: "
             + ", ".join(f"{v:.3e}" for v in seen))
    finally:
        app.close()


@case
def a_screenshot_comes_out_looking_like_a_rendered_scene() -> None:
    CAPTURES.mkdir(exist_ok=True)
    path = CAPTURES / "test_smoke.png"
    path.unlink(missing_ok=True)
    cfg = _headless(max_frames=150, screenshot=path, screenshot_frame=140)
    _run(cfg)
    require(path.exists(), "no screenshot was written")

    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    require(img.shape[:2] == (720, 1280), f"screenshot is {img.shape}")
    mean, std = float(img.mean()), float(img.std())
    require(8.0 < mean < 240.0, f"screenshot mean brightness is {mean:.1f}")
    require(std > 12.0,
            f"screenshot has almost no contrast (std {std:.1f}); a frame that "
            "renders nothing still passes a brightness check")
    # The matter is the only strongly coloured thing on a blue-grey stage.
    spread = float(np.abs(img[..., 0] - img[..., 2]).max())
    require(spread > 25.0,
            f"nothing on screen has any colour (max R-B spread {spread:.0f})")
    note(f"{path.name}: mean {mean:.1f}, std {std:.1f}, R-B spread {spread:.0f}")


@case
def a_screenshot_frame_outside_the_run_still_produces_a_file() -> None:
    # An exact == against the frame counter wrote nothing and said nothing
    # when the run was too short to reach it, and exited 0, so a capture
    # script could not tell it had failed.  Frame 0 could never match at all.
    CAPTURES.mkdir(exist_ok=True)
    for frame_no, frames in ((900, 12), (0, 8)):
        path = CAPTURES / f"test_smoke_clamp_{frame_no}.png"
        path.unlink(missing_ok=True)
        _run(_headless(max_frames=frames, screenshot=path,
                       screenshot_frame=frame_no))
        require(path.exists(),
                f"--screenshot-frame {frame_no} of a {frames}-frame run "
                "wrote nothing")
        path.unlink(missing_ok=True)
    note("frame 900 of a 12-frame run and frame 0 both land on a real frame")


@case
def a_failed_start_still_tears_down_what_it_had_built() -> None:
    # Anything that fails after the renderer exists leaves CUDA-registered
    # GL buffers alive with nobody holding them, and the process then pulls
    # the GL context out from under the CUDA driver.  An unwritable --record
    # path is the reachable way in: it runs after the renderer is up.
    from fctx.app import MatterStudio

    seen: list[bool] = []
    real_close = MatterStudio.close

    def spy(self: object) -> None:
        seen.append(True)
        real_close(self)  # type: ignore[arg-type]

    cfg = _headless(max_frames=2)
    cfg = dataclasses.replace(cfg, tracking=dataclasses.replace(
        cfg.tracking, record_path=Path("Z:/no-such-volume/take.fhr")))
    MatterStudio.close = spy  # type: ignore[method-assign]
    try:
        MatterStudio(cfg)
    except Exception:
        pass
    else:
        raise AssertionError("an unwritable --record path started anyway")
    finally:
        MatterStudio.close = real_close  # type: ignore[method-assign]
    require(seen, "a construction that failed after the renderer never closed")


@case
def the_hand_colliders_are_told_how_long_the_frame_took() -> None:
    # Capsules are uploaded once per frame, so the elapsed time between two
    # uploads is the frame delta.  Handing set_hands the physics dt scaled
    # every capsule and pinch velocity by rate_hz / frame_hz -- 1.5x on a
    # 60 Hz display at the shipped 90 Hz physics -- and measured the collider
    # fade-in in frames instead of seconds.
    from fctx.app import MatterStudio

    app = MatterStudio(_headless(max_frames=0))
    try:
        seen: list[float] = []
        real = app.solver.set_hands
        app.solver.set_hands = lambda poses, dt: (seen.append(dt),
                                                  real(poses, dt))[1]
        for frame_dt in (1 / 60, 1 / 30):
            app._update_physics(app.controls.state, frame_dt)
            require(abs(seen[-1] - frame_dt) < 1e-9,
                    f"a {frame_dt * 1000:.1f} ms frame told the colliders "
                    f"{seen[-1] * 1000:.1f} ms")
        # ...but not an unbounded one: the same value ages the collider fade,
        # and a one-second hitch on the frame a hand appears would skip it.
        app._update_physics(app.controls.state, 4.0)
        require(seen[-1] <= app.clock.max_delta + 1e-9,
                f"a 4 s stall aged the collider fade by {seen[-1]:.2f} s")
        note(f"frame deltas reach set_hands unchanged up to "
             f"{app.clock.max_delta * 1000:.0f} ms, then clamp")
    finally:
        app.close()


@case
def a_benchmark_does_not_report_a_step_count_it_never_measured() -> None:
    # --benchmark implies lockstep, and lockstep never ticks the clock, so
    # "dropped steps 0" was a structural zero rather than a measurement.
    from fctx.app import MatterStudio

    app = MatterStudio(_headless(max_frames=12))
    try:
        report = app.run() or ""
    finally:
        app.close()
    require("lockstep" in report.split("dropped steps")[-1],
            f"the benchmark report claims a dropped-step count:\n{report}")


@case
def the_pointer_is_seeded_from_the_window_the_app_opened() -> None:
    # --source synthetic is the documented no-camera path and its pointer is
    # the cursor divided by this.  glfw only reports a size when it changes,
    # so a window that is never dragged keeps whatever it was seeded with:
    # the 1600x900 default put the centre of a 1280x720 window at 0.4, 0.4.
    from fctx.app import MatterStudio
    from fctx.ui.controls import EventKind, InputEvent

    app = MatterStudio(_headless(max_frames=0))
    try:
        width, height = app.window._window_size
        require(tuple(app.controls.state._window_size) == (width, height),
                f"the controls think the window is "
                f"{app.controls.state._window_size}, it is {(width, height)}")
        app.controls.handle([InputEvent(kind=EventKind.CURSOR,
                                        x=width * 0.5, y=height * 0.5)])
        px, py = app.controls.state.pointer
        require(abs(px - 0.5) < 1e-3 and abs(py - 0.5) < 1e-3,
                f"the centre of the {width}x{height} window the app opened "
                f"maps to {(px, py)}")
        note(f"window {width}x{height}, centre -> ({px:.3f}, {py:.3f})")
    finally:
        app.close()


@case
def a_preset_that_ships_with_wind_keeps_it() -> None:
    # Every ControlState toggle has to start where the config already is, or
    # the first _apply_commands reads the dataclass default as a key press:
    # --preset banner showed "WIND OFF" over a dead-still banner on frame one.
    from fctx.app import MatterStudio
    from fctx.solver import kernels as K

    def wind_on_device(app: object) -> float:
        vector = app.state.vparams.numpy()[K.VPARAM_WIND]  # type: ignore[attr-defined]
        return float(np.linalg.norm(vector))

    banner_wind = float(np.linalg.norm(preset("banner").solver.wind))
    require(banner_wind > 0.0, "the banner preset no longer ships with wind")

    app, _ = _run(_headless("banner", max_frames=8))
    require(wind_on_device(app) > 0.0,
            "--preset banner turned its own wind off during the run")

    app = MatterStudio(_headless("cloth", max_frames=0))
    try:
        app._switch_preset("banner")
        for _ in range(6):
            app._apply_commands(app.controls.update(1 / 90))
        require(wind_on_device(app) > 0.0,
                "pressing the banner key turned the banner's wind off again")
    finally:
        app.close()
    note(f"banner wind {banner_wind:.2f} m/s survives both routes in")


@case
def a_number_key_gives_the_same_scene_as_the_command_line() -> None:
    # A preset is a scene, a solver and an interaction volume.  Carrying only
    # `scene` across a runtime switch gave the granular preset no basin --
    # the failure ARCHITECTURE 4.1 says the basin exists to prevent -- and
    # left the hand reaching at chest height above a pile on the floor.
    from fctx.app import MatterStudio

    app = MatterStudio(_headless("cloth", max_frames=0))
    try:
        for name in ("grain", "cloth"):
            app._switch_preset(name)
            want = preset(name)
            require(app.cfg.solver.basin_radius == want.solver.basin_radius,
                    f"{name}: basin_radius is {app.cfg.solver.basin_radius}, "
                    f"--preset {name} gives {want.solver.basin_radius}")
            require(app.cfg.tracking.stage_center_y
                    == want.tracking.stage_center_y,
                    f"{name}: the interaction volume did not move")
            # The projection reads the tracker's own copy, not app.cfg.
            require(app.tracker.cfg.stage_center_y
                    == want.tracking.stage_center_y,
                    f"{name}: the tracker kept the old interaction volume")
            for _ in range(20):
                app._update_physics(app.controls.state, 1 / 90)
            require(np.isfinite(app.state.snapshot()["x"]).all(),
                    f"{name} went non-finite after the switch")
        note("grain arrives with its basin and gives it back on the way out")
    finally:
        app.close()


@case
def switching_preset_at_runtime_rebuilds_everything() -> None:
    from fctx.app import MatterStudio

    app = MatterStudio(_headless("cloth", max_frames=0))
    try:
        before = app.state.num_particles
        app._switch_preset("soft")
        require(app.bodies[0].kind is MatterKind.SOFT,
                "preset switch did not change the matter")
        after = app.state.num_particles
        for _ in range(30):
            app._update_physics(app.controls.state, 1 / 90)
        snap = app.state.snapshot()
        require(np.isfinite(snap["x"]).all(),
                "the scene went non-finite after a preset switch")
        app._switch_preset("cloth")
        require(app.state.num_particles == before,
                "switching back did not restore the original scene")
        note(f"cloth {before:,} -> soft {after:,} -> cloth "
             f"{app.state.num_particles:,} particles, all finite")
    finally:
        app.close()


@case
def the_camera_keeps_the_matter_and_the_hand_in_frame() -> None:
    # Checked at rest *and* once the matter has settled.  Rest positions
    # alone missed the granular pile: the basin is 0.6 m deep, its near rim
    # sits much closer to the eye than the centre does, and 8.5% of the
    # settled grains rendered off the bottom edge while the rest state still
    # had 7% of margin in hand.
    from fctx.app import MatterStudio

    for name in PRESET_KEYS:
        app = MatterStudio(_headless(name, max_frames=360))
        try:
            track = app.cfg.tracking
            reach = np.array([
                [0.0, track.stage_center_y - track.stage_half_height, 0.0],
                [0.0, track.stage_center_y + track.stage_half_height, 0.0],
            ], np.float64)

            def worst_ndc(blocks: list[np.ndarray]) -> tuple[float, float]:
                view_proj = app.camera.view_proj()
                wx = wy = 0.0
                for block in blocks:
                    homogeneous = np.concatenate(
                        [block, np.ones((len(block), 1))], axis=1)
                    clip = homogeneous @ view_proj.T
                    ndc = clip[:, :2] / np.maximum(np.abs(clip[:, 3:4]), 1e-9)
                    wx = max(wx, float(np.abs(ndc[:, 0]).max()))
                    wy = max(wy, float(np.abs(ndc[:, 1]).max()))
                return wx, wy

            rest_x, rest_y = worst_ndc([b.positions for b in app.bodies] + [reach])
            app.run()
            settled_x, settled_y = worst_ndc(
                [app.state.x.numpy().astype(np.float64), reach])
            for label, worst in (("at rest, vertically", rest_y),
                                 ("at rest, horizontally", rest_x),
                                 ("once settled, vertically", settled_y),
                                 ("once settled, horizontally", settled_x)):
                require(worst <= 1.0,
                        f"{name}: {label}, something sits {worst:.2f} of the "
                        "way past the edge of the frame")
            note(f"{name:7s} fills {rest_y * 100:4.0f}% of the frame height at "
                 f"rest, {settled_y * 100:4.0f}% once settled "
                 f"({settled_x * 100:4.0f}% of its width)")
        finally:
            app.close()


@case
def a_missing_camera_falls_back_instead_of_failing() -> None:
    from fctx.app import MatterStudio

    # The machine this is most likely to run on has no webcam plugged in, and
    # refusing to start would make the whole application untestable there.
    cfg = dataclasses.replace(
        _headless(max_frames=20),
        tracking=dataclasses.replace(preset("cloth").tracking,
                                     source="camera", camera_index=93))
    app = MatterStudio(cfg)
    try:
        app.run()
        require(getattr(app.source, "is_synthetic", False),
                f"fell back to {app.source_note!r}, not the synthetic hand")
        note(f"camera 93 unavailable -> {app.source_note}")
    finally:
        app.close()


@case
def recording_a_session_and_replaying_it_gives_the_same_hands() -> None:
    from fctx.app import MatterStudio
    from fctx.hands.sources import create_source
    from fctx.hands.tracker import HandTracker

    out = Path(os.environ.get("TEMP", ".")) / "fctx-smoke-take.fhr"
    out.unlink(missing_ok=True)
    cfg = dataclasses.replace(
        _headless(max_frames=120),
        tracking=dataclasses.replace(
            _headless().tracking, record_path=out))
    app = MatterStudio(cfg)
    try:
        app.run()
    finally:
        app.close()
    require(out.exists(), "no recording was written")

    replay_cfg = dataclasses.replace(
        _headless().tracking, source="replay", replay_path=out,
        replay_loop=False)
    source = create_source(replay_cfg)
    source.realtime = False
    source.start()
    tracker = HandTracker(replay_cfg)
    now, seen = 0.0, 0
    try:
        for _ in range(200):
            frame = source.poll()
            if frame is None:
                break
            now += 1.0 / 60.0
            seen += len(tracker.update(frame, now))
    finally:
        source.close()
    require(seen > 50, f"replay produced only {seen} hand poses")
    note(f"recorded {out.stat().st_size:,} B, replayed {seen} poses")
    out.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(run(__file__))
