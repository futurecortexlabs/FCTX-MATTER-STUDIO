"""Command line entry point for FCTX MATTER STUDIO."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from .config import (
    HAND_MODEL_PATH,
    PRESETS,
    RECORDING_DIR,
    AppConfig,
    env_device,
    preset,
)
from .core.types import MatterKind


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fctx-matter-studio",
        description="Manipulate simulated matter with your bare hands. "
                    "GPU XPBD physics on NVIDIA Warp, hand tracking on MediaPipe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
  fctx-matter-studio                          live camera, a sheet of cloth
  fctx-matter-studio --preset soft            a soft body instead
  fctx-matter-studio --source synthetic       no camera: steer with the mouse
  fctx-matter-studio --record take01.fhr      save the hand motion
  fctx-matter-studio --source replay --replay take01.fhr
  fctx-matter-studio --benchmark              headless timing run
""")

    g = p.add_argument_group("configuration")
    g.add_argument("--config", type=Path, metavar="FILE",
                   help="a TOML file of settings; command-line flags override it")
    g.add_argument("--dump-config", action="store_true",
                   help="print the effective settings as TOML and exit")
    g.add_argument("--kiosk", action="store_true",
                   help="exhibition mode: fullscreen, resilient to bad frames, "
                        "attract-mode demo after 20 s idle, camera hot-plug")
    g.add_argument("--idle-demo", type=float, default=None, metavar="SECONDS",
                   help="run the demonstration when no hand has been seen for "
                        "this long (0 disables)")
    g.add_argument("--resilient", action="store_true",
                   help="log a frame that raises, reset the scene and continue")
    g.add_argument("--log-file", type=Path, metavar="FILE")

    g = p.add_argument_group("scene")
    g.add_argument("--preset", choices=PRESETS, default=None,
                   help="starting scene (default: cloth, or the config file's)")
    g.add_argument("--hardness", type=float, metavar="0..1",
                   help="starting position on the hardness dial")
    g.add_argument("--resolution", type=int, metavar="N",
                   help="cloth particles per side, or soft-body lattice cells")
    g.add_argument("--grains", type=int, metavar="N",
                   help="granular particle count")

    g = p.add_argument_group("tracking")
    g.add_argument("--source", choices=("camera", "synthetic", "replay", "video"),
                   help="where hands come from (default: camera, "
                        "falling back to synthetic if none is available)")
    g.add_argument("--camera", type=int, default=None, metavar="INDEX",
                   help="camera device index")
    g.add_argument("--camera-size", type=str, default=None, metavar="WxH",
                   help="requested capture resolution, e.g. 1280x720")
    g.add_argument("--no-mirror", action="store_true",
                   help="do not mirror the camera image")
    g.add_argument("--hands", type=int, choices=(1, 2), default=None,
                   help="how many hands to track")
    g.add_argument("--replay", type=Path, metavar="FILE",
                   help="a .fhr recording to play back")
    g.add_argument("--video", type=Path, metavar="FILE",
                   help="a video file to track instead of a camera")
    g.add_argument("--record", type=Path, metavar="FILE",
                   help="record the tracked hands to this .fhr file")

    g = p.add_argument_group("solver")
    g.add_argument("--substeps", type=int, metavar="N")
    g.add_argument("--rate", type=float, metavar="HZ", help="physics rate")
    g.add_argument("--no-self-collision", action="store_true")
    g.add_argument("--no-cuda-graph", action="store_true",
                   help="launch kernels individually; slower but profilable")
    g.add_argument("--device", default=None, metavar="DEV",
                   help="warp device, e.g. cuda:0 or cpu")

    g = p.add_argument_group("display")
    g.add_argument("--size", type=str, default=None, metavar="WxH",
                   help="window size, e.g. 1920x1080")
    g.add_argument("--fullscreen", action="store_true")
    g.add_argument("--no-vsync", action="store_true")
    g.add_argument("--msaa", type=int, choices=(0, 2, 4, 8), default=None)
    g.add_argument("--no-bloom", action="store_true")
    g.add_argument("--no-ssao", action="store_true")
    g.add_argument("--no-shadows", action="store_true")
    g.add_argument("--no-webcam-view", action="store_true")
    g.add_argument("--no-hud", action="store_true")

    g = p.add_argument_group("diagnostics")
    g.add_argument("--headless", action="store_true",
                   help="render offscreen, open no window")
    g.add_argument("--frames", type=int, default=0, metavar="N",
                   help="exit after N frames")
    g.add_argument("--screenshot", type=Path, metavar="FILE")
    g.add_argument("--screenshot-frame", type=int, default=60, metavar="N")
    g.add_argument("--benchmark", action="store_true",
                   help="headless timing run, no window, prints a report")
    g.add_argument("--demo", action="store_true",
                   help="run the choreographed demonstration: grab, lift, "
                        "sweep the dial, let go (the D key at runtime)")
    g.add_argument("--realtime", action="store_true",
                   help="follow the wall clock even when headless, instead of "
                        "one physics step per frame")
    g.add_argument("--profile", type=float, default=0.0, metavar="SECONDS",
                   help="print a stage breakdown every N seconds")
    g.add_argument("-v", "--verbose", action="store_true")
    g.add_argument("--check", action="store_true",
                   help="verify the environment and exit")
    return p


def _recording_path(path: Path, existing: bool = False) -> Path:
    """Where a bare .fhr name lives: alongside everything else --record wrote.

    ``existing`` keeps a relative path that really is there working, so this
    only ever adds a place to look.
    """
    if path.is_absolute() or (existing and path.exists()):
        return path
    return RECORDING_DIR / path


def _parse_size(text: str, what: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--{what}: expected WxH, got {text!r}") from exc


def config_from_args(args: argparse.Namespace) -> AppConfig:
    rep = dataclasses.replace
    cfg = preset(args.preset or "cloth")
    if args.config is not None:
        from .settings import load_config

        cfg = load_config(args.config, base=cfg)
        if args.preset is not None:
            # The flag wins over the file's preset, but the file's other
            # sections still apply on top of it.
            cfg = load_config(args.config, base=preset(args.preset),
                              ignore_preset=True)

    scene = cfg.scene
    if args.hardness is not None:
        scene = rep(scene, hardness=min(max(args.hardness, 0.0), 1.0))
    if args.resolution is not None:
        if scene.kind is MatterKind.CLOTH:
            scene = rep(scene, cloth_resolution=max(args.resolution, 4))
        elif scene.kind is MatterKind.SOFT:
            scene = rep(scene, soft_resolution=max(args.resolution, 3))
    if args.grains is not None:
        scene = rep(scene, grain_count=max(args.grains, 1))

    track = cfg.tracking
    if args.source:
        track = rep(track, source=args.source)
    if args.camera is not None:
        track = rep(track, camera_index=args.camera)
    if args.camera_size:
        w, h = _parse_size(args.camera_size, "camera-size")
        track = rep(track, camera_width=w, camera_height=h)
    if args.no_mirror:
        track = rep(track, mirror=False)
    if args.hands is not None:
        track = rep(track, max_hands=args.hands)
    if args.replay:
        # Resolved the same way --record resolves, so the round trip the
        # README prints works as printed: a bare name written by --record
        # lands in RECORDING_DIR, and --replay has to look there for it or it
        # falls back to the synthetic hand and exits 0, which reads as a run
        # that worked.
        track = rep(track, replay_path=_recording_path(args.replay, True),
                    source="replay")
    if args.video:
        track = rep(track, video_path=args.video, source="video")
    if args.record:
        track = rep(track, record_path=_recording_path(args.record))

    solver = cfg.solver
    if args.substeps is not None:
        solver = rep(solver, substeps=max(args.substeps, 1))
    if args.rate is not None:
        solver = rep(solver, rate_hz=max(args.rate, 15.0))
    if args.no_self_collision:
        solver = rep(solver, self_collision=False)
    if args.no_cuda_graph:
        solver = rep(solver, use_cuda_graph=False)

    render = cfg.render
    if args.size:
        w, h = _parse_size(args.size, "size")
        render = rep(render, width=w, height=h)
    if args.fullscreen:
        render = rep(render, fullscreen=True)
    if args.no_vsync:
        render = rep(render, vsync=False)
    if args.msaa is not None:
        render = rep(render, msaa=args.msaa)
    if args.no_bloom:
        render = rep(render, bloom=False)
    if args.no_ssao:
        render = rep(render, ssao_samples=0)
    if args.no_shadows:
        render = rep(render, shadow_size=0)
    if args.no_webcam_view:
        render = rep(render, show_webcam=False)
    if args.no_hud:
        render = rep(render, show_hud=False)

    headless = args.headless or args.benchmark
    frames = args.frames
    if args.benchmark and not frames:
        frames = 900

    idle_demo = cfg.idle_demo
    resilient = cfg.resilient
    if args.kiosk:
        render = rep(render, fullscreen=True)
        resilient = True
        if idle_demo <= 0.0:
            idle_demo = 20.0
    if args.idle_demo is not None:
        idle_demo = max(0.0, args.idle_demo)
    if args.resilient:
        resilient = True

    return rep(
        cfg,
        scene=scene,
        tracking=track,
        solver=solver,
        render=render,
        device=args.device or env_device(),
        headless=headless,
        lockstep=headless and not args.realtime,
        demo=args.demo or cfg.demo,
        idle_demo=idle_demo,
        resilient=resilient,
        log_file=args.log_file or cfg.log_file,
        max_frames=frames,
        screenshot=args.screenshot,
        screenshot_frame=args.screenshot_frame,
        verbose=args.verbose,
        profile_interval=args.profile,
    )


def _silence_opencv() -> None:
    """Quieten OpenCV's backend chatter, on whichever OpenCV is installed.

    Nothing here is allowed to fail: a noisy probe is a cosmetic problem, a
    probe that does not run is the diagnostic being useless.
    """
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return
    for setter in (getattr(getattr(getattr(cv2, "utils", None), "logging", None),
                           "setLogLevel", None),
                   getattr(cv2, "setLogLevel", None)):
        if setter is None:
            continue
        try:
            setter(0)
            return
        except Exception:  # noqa: BLE001
            continue


def check_environment(verbose: bool = True,
                      config: Path | None = None) -> int:
    """Report on the GPU, OpenGL, the model pack, the camera and the config file."""
    ok = True

    def line(tag: str, text: str) -> None:
        print(f"  {tag:<5} {text}")

    print("FCTX MATTER STUDIO -- environment check\n")

    try:
        import warp as wp
        wp.init()
        devices = wp.get_devices()
        cuda = [d for d in devices if d.is_cuda]
        if cuda:
            d = cuda[0]
            line("[ok]", f"warp {wp.config.version} on {d.name} (sm_{d.arch}, "
                         f"{d.total_memory // (1 << 30)} GiB)")
        else:
            ok = False
            line("[!!]", f"warp {wp.config.version} found no CUDA device; "
                         "physics will run on the CPU and will be slow")
    except Exception as exc:  # noqa: BLE001
        ok = False
        line("[!!]", f"warp unavailable: {exc}")

    try:
        import glfw
        import moderngl
        if not glfw.init():
            raise RuntimeError("glfw.init() returned false")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        win = glfw.create_window(64, 64, "probe", None, None)
        if not win:
            raise RuntimeError("could not create an OpenGL 4.3 core context")
        glfw.make_context_current(win)
        ctx = moderngl.create_context()
        line("[ok]", f"opengl {ctx.info['GL_VERSION']} on "
                     f"{ctx.info['GL_RENDERER']}")
        ctx.release()
        glfw.destroy_window(win)
        glfw.terminate()
    except Exception as exc:  # noqa: BLE001
        ok = False
        line("[!!]", f"opengl unavailable: {exc}")

    if HAND_MODEL_PATH.exists():
        line("[ok]", f"hand model {HAND_MODEL_PATH.name} "
                     f"({HAND_MODEL_PATH.stat().st_size:,} B)")
    else:
        ok = False
        line("[!!]", "hand model missing -- run: "
                     "python tools/download_models.py")

    try:
        import cv2
        # OpenCV logs a warning per failed backend probe, which makes a
        # perfectly normal "no camera plugged in" look like a crash.  The
        # call moved in OpenCV 5 (the pinned version), and reaching for the
        # 4.x spelling threw before a single VideoCapture was tried -- the
        # probe this whole command exists for never ran at all.
        _silence_opencv()
        found = []
        for idx in range(3):
            for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
                cap = cv2.VideoCapture(idx, api)
                opened = cap.isOpened()
                cap.release()
                if opened:
                    found.append(idx)
                    break
        if found:
            line("[ok]", f"camera index {found} available")
        else:
            line("[--]", "no camera found; --source synthetic still works")
    except Exception as exc:  # noqa: BLE001
        line("[--]", f"camera probe failed: {exc}")

    if config is not None:
        # The one thing an operator can fix at the venue: say so before showtime.
        from .settings import ConfigError, load_config

        try:
            cfg = load_config(config)
        except ConfigError as exc:
            ok = False
            line("[!!]", f"config: {exc}")
        else:
            line("[ok]", f"config {config} ({cfg.scene.kind.name.lower()}, "
                         f"camera {cfg.tracking.camera_index}, "
                         f"source {cfg.tracking.source})")
    print("\n" + ("ready." if ok else "not ready -- fix the [!!] lines above."))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return check_environment(args.verbose, config=args.config)

    try:
        cfg = config_from_args(args)
    except Exception as exc:  # noqa: BLE001 -- a bad config file is a user error
        print(f"fctx-matter-studio: {exc}", file=sys.stderr)
        return 2
    if args.dump_config:
        from .settings import dump_config

        sys.stdout.write(dump_config(cfg, args.preset))
        return 0

    from .app import MatterStudio  # imported late: pulls in warp and OpenGL

    try:
        app = MatterStudio(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"fctx-matter-studio: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        print("\nrun with --check to diagnose the environment, or --verbose "
              "for the full traceback.", file=sys.stderr)
        return 2

    try:
        report = app.run()
    finally:
        app.close()

    if args.benchmark and report is not None:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
