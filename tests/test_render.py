"""Render tests.  No pytest: plain asserts and a __main__ runner.

Everything here runs against a hand-built scene and a duck-typed solver state,
because ``fctx.solver`` and ``fctx.hands`` are written in parallel with this
module and must not be able to break the graphics tests.  If they happen to
import cleanly they are exercised as a bonus at the end.

    .venv/Scripts/python.exe tests/test_render.py
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import moderngl  # noqa: E402

from fctx.config import AppConfig, RenderConfig  # noqa: E402
from fctx.core.material import DEFAULT_MATERIALS, evaluate  # noqa: E402
from fctx.core.types import (  # noqa: E402
    FLAG_SELF_COLLIDE,
    FLAG_SURFACE,
    BodyData,
    FrameStats,
    Gesture,
    Handedness,
    HandPose,
    MatterKind,
)
from fctx.render import (  # noqa: E402
    InteropError,
    OrbitCamera,
    ParticleBuffers,
    Renderer,
    ShaderLibrary,
    TextAtlas,
    Window,
)

CAPTURES = REPO_ROOT / "captures"
WIDTH, HEIGHT = 1280, 720


# --------------------------------------------------------------------------
# a scene, built without fctx.bodies
# --------------------------------------------------------------------------


def _empty_constraints(p: int) -> dict[str, np.ndarray]:
    return dict(
        dist_idx=np.zeros((0, 2), np.int32),
        dist_rest=np.zeros(0, np.float32),
        dist_kind=np.zeros(0, np.int32),
        dist_color=np.zeros(0, np.int32),
        bend_idx=np.zeros((0, 4), np.int32),
        bend_rest=np.zeros(0, np.float32),
        bend_color=np.zeros(0, np.int32),
        tet_idx=np.zeros((0, 4), np.int32),
        tet_dm_inv=np.zeros((0, 3, 3), np.float32),
        tet_rest_volume=np.zeros(0, np.float32),
        tet_color=np.zeros(0, np.int32),
    )


def _grid_triangles(n: int) -> np.ndarray:
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + n
            tris.append((a, b, a + 1))
            tris.append((a + 1, b, b + 1))
    return np.asarray(tris, np.int32).reshape(-1, 3)


def fake_cloth(n: int = 40, size: float = 0.46) -> BodyData:
    """A sagging sheet: curved so shading, folds and shadows all have something
    to bite on, which a perfectly flat quad would not give."""
    t = np.linspace(-0.5, 0.5, n)
    gx, gy = np.meshgrid(t * size, t * size, indexing="ij")
    sag = -0.09 * np.cos(gx / size * math.pi) * np.cos(gy / size * math.pi)
    pos = np.stack([gx - 0.26, gy + 0.44, sag + 0.06], axis=-1).reshape(-1, 3)

    p = pos.shape[0]
    uv = np.stack(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n),
                              indexing="ij"), axis=-1).reshape(-1, 2)
    return BodyData(
        kind=MatterKind.CLOTH,
        name="fake_cloth",
        positions=np.ascontiguousarray(pos, np.float32),
        inv_mass=np.ones(p, np.float32),
        flags=np.full(p, FLAG_SURFACE, np.uint32),
        tri_idx=_grid_triangles(n),
        uv=np.ascontiguousarray(uv, np.float32),
        double_sided=True,
        particle_radius=0.006,
        **_empty_constraints(p),
    )


def fake_soft(rings: int = 22, segments: int = 32, radius: float = 0.115) -> BodyData:
    from fctx.render.geometry import uv_sphere

    mesh = uv_sphere(rings, segments, radius)
    pos = mesh.positions + np.array([0.13, 0.20, 0.04], np.float32)
    p = pos.shape[0]
    return BodyData(
        kind=MatterKind.SOFT,
        name="fake_soft",
        positions=np.ascontiguousarray(pos, np.float32),
        inv_mass=np.ones(p, np.float32),
        flags=np.full(p, FLAG_SURFACE, np.uint32),
        tri_idx=mesh.indices,
        uv=mesh.uvs,
        double_sided=False,
        particle_radius=0.010,
        **_empty_constraints(p),
    )


def fake_grain(count: int = 1400, radius: float = 0.0075) -> BodyData:
    rng = np.random.default_rng(20260919)
    n = int(np.ceil(count ** (1.0 / 3.0)))
    lattice = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"),
                       axis=-1).reshape(-1, 3)[:count].astype(np.float32)
    pos = lattice * (radius * 2.05) + rng.normal(0.0, radius * 0.18,
                                                 (count, 3)).astype(np.float32)
    pos -= pos.mean(axis=0)
    pos += np.array([0.36, 0.13, 0.03], np.float32)
    return BodyData(
        kind=MatterKind.GRAIN,
        name="fake_grain",
        positions=np.ascontiguousarray(pos, np.float32),
        inv_mass=np.ones(count, np.float32),
        flags=np.full(count, FLAG_SELF_COLLIDE, np.uint32),
        tri_idx=np.zeros((0, 3), np.int32),
        uv=np.zeros((count, 2), np.float32),
        double_sided=False,
        particle_radius=radius,
        **_empty_constraints(count),
    )


def vertex_normals(positions: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.zeros_like(positions)
    if tris.size:
        a, b, c = (positions[tris[:, i]] for i in range(3))
        face = np.cross(b - a, c - a)
        for i in range(3):
            np.add.at(n, tris[:, i], face)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    flat = length[:, 0] < 1e-12
    n[flat] = np.array([0.0, 1.0, 0.0], np.float32)
    length[flat] = 1.0
    return np.ascontiguousarray(n / length, np.float32)


def build_scene() -> tuple[list[BodyData], np.ndarray, np.ndarray]:
    cloth, soft, grain = fake_cloth(), fake_soft(), fake_grain()
    bodies = [cloth, soft, grain]
    # The sphere's normals are analytic.  Averaging its faces instead would
    # give the two copies of the duplicated longitude seam different normals
    # and put a false crease with a doubled highlight down the middle.
    sphere_centre = np.array([0.13, 0.20, 0.04], np.float32)
    radial = soft.positions - sphere_centre
    soft_normals = radial / np.linalg.norm(radial, axis=1, keepdims=True)
    normals = np.concatenate([
        vertex_normals(cloth.positions, cloth.tri_idx),
        soft_normals.astype(np.float32),
        vertex_normals(grain.positions, grain.tri_idx),
    ]).astype(np.float32)
    positions = np.concatenate([b.positions for b in bodies]).astype(np.float32)
    return bodies, np.ascontiguousarray(positions), np.ascontiguousarray(normals)


def fake_state(positions: np.ndarray, normals: np.ndarray, device: str,
               *, use_warp: bool) -> object:
    if use_warp:
        import warp as wp

        wp.init()
        dev = wp.get_device(device)
        return SimpleNamespace(
            x=wp.array(positions, dtype=wp.vec3, device=dev),
            normal=wp.array(normals, dtype=wp.vec3, device=dev),
            num_particles=positions.shape[0],
        )
    return SimpleNamespace(x=positions.copy(), normal=normals.copy(),
                           num_particles=positions.shape[0])


def fake_pose(phase: float, pinching: bool) -> HandPose:
    """A plausible open hand, built from the MediaPipe topology by hand."""
    joints = np.zeros((21, 3), np.float32)
    wrist = np.array([0.02 + 0.06 * math.sin(phase), 0.16, 0.24], np.float32)
    joints[0] = wrist
    spread = np.linspace(-0.055, 0.055, 5)
    for finger in range(5):
        mcp = 1 + finger * 4
        base = wrist + np.array([spread[finger], 0.045, -0.005], np.float32)
        for seg in range(4):
            reach = 0.030 * (seg + 1) * (0.72 if finger == 0 else 1.0)
            bend = 0.9 if pinching else 0.15
            joints[mcp + seg] = base + np.array([
                spread[finger] * 0.35 * seg,
                reach * math.cos(bend * seg * 0.5),
                -reach * math.sin(bend * seg * 0.5) * 0.6,
            ], np.float32)
    pinch_point = (joints[4] + joints[8]) * 0.5
    return HandPose(
        joints=np.ascontiguousarray(joints),
        velocities=np.zeros((21, 3), np.float32),
        handedness=Handedness.RIGHT,
        pinch=0.95 if pinching else 0.12,
        pinching=pinching,
        curl=0.7 if pinching else 0.1,
        gesture=Gesture.PINCH if pinching else Gesture.OPEN,
        pinch_point=np.ascontiguousarray(pinch_point),
        confidence=0.95,
        track_id=1,
    )


def fake_preview(w: int = 320, h: int = 180) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([
        60 + 90 * np.sin(x / 34.0),
        40 + 70 * np.cos(y / 27.0),
        70 + 60 * np.sin((x + y) / 45.0),
    ], axis=-1)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def fake_stats() -> FrameStats:
    return FrameStats(frame_ms=11.4, physics_ms=5.2, render_ms=3.8,
                      tracking_ms=2.1, fps=87.6, tracking_fps=59.4,
                      substeps=12, particles=5184, constraints=20448,
                      contacts=1734, grabbed=96, hands=1)


def base_config(**render_kwargs: object) -> AppConfig:
    render = RenderConfig(width=WIDTH, height=HEIGHT, vsync=False,
                          title="FCTX MATTER STUDIO -- test")
    if render_kwargs:
        render = replace(render, **render_kwargs)  # type: ignore[arg-type]
    return replace(AppConfig(), render=render, headless=True)


def numpy_state(positions: np.ndarray, normals: np.ndarray) -> object:
    return SimpleNamespace(x=positions.copy(), normal=normals.copy(),
                           num_particles=positions.shape[0])


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}{('  -- ' + detail) if detail else ''}")


def test_shaders_compile(window: Window) -> None:
    """Every program in the pipeline, reported individually by name."""
    lib = ShaderLibrary(window.ctx)
    specs = [
        ("depth", "depth_only.vert", "depth_only.frag", None),
        ("matter", "matter.vert", "matter.frag", None),
        ("floor", "floor.vert", "floor.frag", None),
        ("background", "fullscreen.vert", "background.frag", None),
        ("hand", "hand.vert", "hand.frag", None),
        ("hand_depth", "hand.vert", "depth_only.frag", None),
        ("particle", "particle.vert", "particle.frag", None),
        ("particle_depth", "particle.vert", "particle_depth.frag", None),
        ("ssao", "fullscreen.vert", "ssao.frag", None),
        ("ssao_blur", "fullscreen.vert", "ssao_blur.frag", None),
        ("resolve_ms", "fullscreen.vert", "resolve.frag", {"SAMPLE_COUNT": 4}),
        ("resolve_1x", "fullscreen.vert", "resolve.frag", {"SAMPLE_COUNT": 1}),
        ("bloom_down", "fullscreen.vert", "bloom_down.frag", None),
        ("bloom_up", "fullscreen.vert", "bloom_up.frag", None),
        ("composite", "fullscreen.vert", "composite.frag", None),
        ("overlay", "overlay.vert", "overlay.frag", None),
    ]
    failures = 0
    for name, vert, frag, defines in specs:
        try:
            lib.program(name, vertex=vert, fragment=frag, defines=defines)
            record(f"shader:{name}", True)
        except Exception as exc:
            failures += 1
            record(f"shader:{name}", False, str(exc).splitlines()[0])
            print(str(exc))
    lib.release()
    assert failures == 0, f"{failures} shader(s) failed to build"


def test_include_cycle_detected(window: Window) -> None:
    lib = ShaderLibrary(window.ctx)
    src = lib.source("matter.frag")
    ok = "#include" not in src and "d_ggx" in src and "POISSON16" in src
    record("shader includes expand", ok,
           "" if ok else "matter.frag did not splice its includes")
    assert ok


def test_text_atlas(window: Window) -> None:
    atlas = TextAtlas(window.ctx, px=44)
    sample = "FCTX 100% hard\nSOFT GEL"
    expected = sum(1 for ch in sample if ch not in " \n\t")
    got = TextAtlas.count_quads(sample)
    record("text atlas bake", atlas.cell_w > 2 and atlas.cell_h > 4,
           f"{Path(atlas.source).name} cell {atlas.cell_w}x{atlas.cell_h} "
           f"atlas {atlas.width}x{atlas.height}")
    record("text quad count", got == expected == 19, f"{got} quads")
    assert got == expected == 19
    assert atlas.advance(40.0) > 4.0
    atlas.release()


def test_interop(window: Window) -> None:
    buffers = ParticleBuffers(window.ctx, 4096, AppConfig().device)
    record("interop path", True, buffers.describe())
    for _ in range(3):
        pos, nrm = buffers.map_for_warp()
        assert pos is not None and nrm is not None
        buffers.unmap()
    try:
        buffers.unmap()
    except Exception as exc:
        record("unmap while unmapped raises", True, type(exc).__name__)
    else:
        record("unmap while unmapped raises", False, "no exception")
        raise AssertionError("ParticleBuffers.unmap must reject a stray unmap")

    host = np.random.default_rng(1).random((4096, 3)).astype(np.float32)
    buffers.update_from(host, host)
    buffers.release()
    buffers.release()  # idempotent
    record("interop map/unmap round trip", True, "3 cycles, released cleanly")


def _draw_once(renderer: Renderer, camera: OrbitCamera, bodies: list[BodyData],
               *, poses: list[HandPose], preview: np.ndarray | None) -> None:
    materials = [evaluate(DEFAULT_MATERIALS[b.kind], h)
                 for b, h in zip(bodies, (0.28, 0.62, 0.45))]
    renderer.draw(camera, poses, materials, preview, fake_stats(),
                  ["source     synthetic", "device     cuda:0"])


def test_headless_frame(window: Window, bodies: list[BodyData],
                        state: object) -> float:
    cfg = base_config()
    renderer = Renderer(window, state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    poses = [fake_pose(0.0, False), fake_pose(2.1, True)]
    preview = fake_preview()

    _draw_once(renderer, camera, bodies, poses=poses, preview=preview)
    window.swap()
    err = window.ctx.error
    record("frame renders without GL error", err == "GL_NO_ERROR", err)
    assert err == "GL_NO_ERROR"

    CAPTURES.mkdir(parents=True, exist_ok=True)
    out = window.save_png(CAPTURES / "test_render.png")
    img = window.read_pixels().astype(np.float32) / 255.0
    mean = float(img.mean())
    var = float(img.var())
    ok = 0.02 < mean < 0.92 and var > 1e-4
    record("frame is neither black nor white", ok,
           f"mean={mean:.4f} var={var:.5f} -> {out}")
    assert ok, f"degenerate frame: mean={mean} var={var}"

    # Warm up, then time a steady-state frame including the GPU finishing.
    for _ in range(10):
        _draw_once(renderer, camera, bodies, poses=poses, preview=preview)
    window.ctx.finish()
    frames = 60
    start = time.perf_counter()
    for _ in range(frames):
        camera.orbit(0.6, 0.0)
        _draw_once(renderer, camera, bodies, poses=poses, preview=preview)
    window.ctx.finish()
    ms = (time.perf_counter() - start) * 1000.0 / frames
    record("steady-state frame time", True, f"{ms:.2f} ms/frame at {WIDTH}x{HEIGHT}")

    renderer.release()
    return ms


def test_toggles(window: Window, bodies: list[BodyData], state: object) -> None:
    camera = OrbitCamera(base_config().camera, window.aspect)
    poses = [fake_pose(1.0, True)]
    preview = fake_preview()

    variants: list[tuple[str, AppConfig, dict[str, bool]]] = [
        ("bloom off", base_config(bloom=False), {}),
        ("ssao off (config)", base_config(ssao_samples=0), {}),
        ("shadows off (config)", base_config(shadow_size=0), {}),
        ("no msaa", base_config(msaa=1), {}),
        ("msaa 8x", base_config(msaa=8), {}),
        ("no post", base_config(bloom=False, ssao_samples=0, shadow_size=0,
                                vignette=0.0, chromatic_aberration=0.0,
                                aces=False), {}),
        ("hud off", base_config(), {"show_hud": False}),
        ("webcam off", base_config(), {"show_webcam": False}),
        ("hands off", base_config(), {"show_hands": False}),
        ("wireframe", base_config(), {"wireframe": True}),
        ("bloom off at runtime", base_config(), {"bloom_enabled": False}),
        ("ssao off at runtime", base_config(), {"ssao_enabled": False}),
        ("shadows off at runtime", base_config(), {"shadows_enabled": False}),
        ("everything off", base_config(), {
            "show_hud": False, "show_webcam": False, "show_hands": False,
            "bloom_enabled": False, "ssao_enabled": False,
            "shadows_enabled": False}),
    ]
    for label, cfg, toggles in variants:
        renderer = Renderer(window, state, bodies, cfg)
        for key, value in toggles.items():
            setattr(renderer, key, value)
        use_preview = renderer.show_webcam
        _draw_once(renderer, camera, bodies, poses=poses,
                   preview=preview if use_preview else None)
        window.ctx.finish()
        err = window.ctx.error
        renderer.release()
        record(f"toggle: {label}", err == "GL_NO_ERROR", err)
        assert err == "GL_NO_ERROR", f"{label}: {err}"


def test_no_hands_and_no_preview(window: Window, bodies: list[BodyData],
                                 state: object) -> None:
    cfg = base_config()
    renderer = Renderer(window, state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    _draw_once(renderer, camera, bodies, poses=[], preview=None)
    window.ctx.finish()
    err = window.ctx.error
    renderer.release()
    record("empty hand list and no preview", err == "GL_NO_ERROR", err)
    assert err == "GL_NO_ERROR"


def test_material_mismatch_is_loud(window: Window, bodies: list[BodyData],
                                   state: object) -> None:
    cfg = base_config()
    renderer = Renderer(window, state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    try:
        renderer.draw(camera, [], [], None, fake_stats(), [])
    except ValueError as exc:
        record("material count mismatch raises", True, str(exc).split(";")[0])
    else:
        record("material count mismatch raises", False, "no exception")
        raise AssertionError("Renderer.draw accepted the wrong material count")
    finally:
        renderer.release()


def test_host_fallback_path(window: Window, bodies: list[BodyData],
                            positions: np.ndarray, normals: np.ndarray) -> None:
    """The renderer must work when the state holds plain numpy arrays."""
    cfg = base_config()
    numpy_state = fake_state(positions, normals, cfg.device, use_warp=False)
    renderer = Renderer(window, numpy_state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    _draw_once(renderer, camera, bodies, poses=[fake_pose(0.4, False)],
               preview=fake_preview())
    window.ctx.finish()
    err = window.ctx.error
    img = window.read_pixels()
    renderer.release()
    ok = err == "GL_NO_ERROR" and float(img.mean()) > 5.0
    record("numpy (host) state path", ok, f"{err} mean={img.mean():.1f}")
    assert ok


def test_resize(window: Window, bodies: list[BodyData], state: object) -> None:
    """Resizing must rebuild every size-dependent target without leaking."""
    import glfw

    cfg = base_config()
    renderer = Renderer(window, state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    sizes = [(960, 540), (1600, 900), (WIDTH, HEIGHT)]
    seen = []
    try:
        for w, h in sizes:
            glfw.set_window_size(window.handle, w, h)
            window.poll()
            _draw_once(renderer, camera, bodies, poses=[fake_pose(0.3, False)],
                       preview=fake_preview())
            window.ctx.finish()
            err = window.ctx.error
            assert err == "GL_NO_ERROR", f"{w}x{h}: {err}"
            seen.append(f"{window.size[0]}x{window.size[1]}")
        # Random sizes, including a one-pixel-high window: every bloom level,
        # the half-resolution AO buffer and the multisample target are rebuilt
        # each time, and a rounding error in any of them makes the framebuffer
        # incomplete rather than merely ugly.
        rng = np.random.default_rng(20260919)
        for _ in range(10):
            w, h = int(rng.integers(140, 900)), int(rng.integers(1, 700))
            glfw.set_window_size(window.handle, w, h)
            window.poll()
            _draw_once(renderer, camera, bodies, poses=[], preview=None)
            window.ctx.finish()
            assert window.ctx.error == "GL_NO_ERROR", window.size
        seen.append("10 random sizes")
    finally:
        renderer.release()
        glfw.set_window_size(window.handle, WIDTH, HEIGHT)
        window.poll()
    record("resize rebuilds render targets", True, " -> ".join(seen))


def test_windowed_context(bodies: list[BodyData], positions: np.ndarray,
                          normals: np.ndarray) -> None:
    """The shipping path draws into the default framebuffer, not an FBO."""
    cfg = replace(base_config(), headless=False)
    win = Window(cfg)
    try:
        state = fake_state(positions, normals, cfg.device, use_warp=False)
        renderer = Renderer(win, state, bodies, cfg)
        camera = OrbitCamera(cfg.camera, win.aspect)
        _draw_once(renderer, camera, bodies, poses=[fake_pose(0.0, True)],
                   preview=fake_preview())
        err = win.ctx.error
        mean = float(win.read_pixels().mean())
        win.swap()
        renderer.release()
    finally:
        win.close()
    ok = err == "GL_NO_ERROR" and mean > 5.0
    record("visible window renders to ctx.screen", ok, f"{err} mean={mean:.1f}")
    assert ok


def test_bare_context(window: Window, bodies: list[BodyData],
                      state: object) -> None:
    """Renderer must accept a raw moderngl.Context, not only a Window."""
    cfg = base_config()
    renderer = Renderer(window.ctx, state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    try:
        _draw_once(renderer, camera, bodies, poses=[fake_pose(0.9, False)],
                   preview=None)
        window.ctx.finish()
        err = window.ctx.error
    finally:
        renderer.release()
    record("bare moderngl context", err == "GL_NO_ERROR", err)
    assert err == "GL_NO_ERROR"


def test_camera_math() -> None:
    from fctx.render.camera import look_at, orthographic, perspective

    cfg = base_config().camera
    cam = OrbitCamera(cfg, 16 / 9)
    assert np.allclose(cam.position, cfg.position, atol=1e-6), cam.position

    proj = perspective(cfg.fov_y, 16 / 9, cfg.near, cfg.far)
    near_pt = proj @ np.array([0.0, 0.0, -cfg.near, 1.0])
    far_pt = proj @ np.array([0.0, 0.0, -cfg.far, 1.0])
    assert abs(near_pt[2] / near_pt[3] + 1.0) < 1e-6
    assert abs(far_pt[2] / far_pt[3] - 1.0) < 1e-6

    view = look_at(np.array([0.0, 0.0, 2.0]), np.zeros(3), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(view @ np.array([0.0, 0.0, 0.0, 1.0]),
                       [0.0, 0.0, -2.0, 1.0], atol=1e-9)

    ortho = orthographic(2.0, 2.0, 0.1, 10.0)
    assert abs((ortho @ np.array([2.0, 2.0, -0.1, 1.0]))[0] - 1.0) < 1e-9

    before = cam.distance
    cam.zoom(1.0)
    cam.orbit(50.0, 20.0)
    assert cam.distance < before
    assert OrbitCamera.MIN_PITCH <= cam.pitch <= OrbitCamera.MAX_PITCH

    lvp = cam.light_view_proj()
    centre = lvp @ np.append(cam.stage_center, 1.0)
    centre = centre[:3] / centre[3]
    assert np.all(np.abs(centre) <= 1.0 + 1e-6), centre
    record("camera and projection matrices", True,
           f"shadow extent {cam.shadow_extent:.3f} m")


def test_geometry() -> None:
    from fctx.render.geometry import capsule_shell, grid_plane, uv_sphere

    shell = capsule_shell(rings=12, segments=20)
    lengths = np.linalg.norm(shell.directions, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-6)
    assert set(np.unique(shell.side).tolist()) == {0.0, 1.0}
    # The equator must exist on both halves or the capsule waist has a hole.
    equator = np.isclose(shell.directions[:, 1], 0.0, atol=1e-6)
    assert set(np.unique(shell.side[equator]).tolist()) == {0.0, 1.0}
    assert int(shell.indices.max()) < shell.num_vertices

    sphere = uv_sphere(8, 12, 0.5)
    assert np.allclose(np.linalg.norm(sphere.positions, axis=1), 0.5, atol=1e-6)

    plane = grid_plane(2.0, 2, height=0.25)
    assert np.allclose(plane.positions[:, 1], 0.25)
    assert plane.num_triangles == 8
    record("geometry helpers", True,
           f"capsule {shell.num_vertices} verts / {shell.num_triangles} tris")


def test_optional_real_modules(window: Window) -> None:
    """Exercise the sibling subsystems only if they already import."""
    notes = []
    try:
        from fctx.bodies import build_scene as real_build_scene

        cfg = base_config()
        real_bodies = real_build_scene(cfg.scene)
        for body in real_bodies:
            body.validate()
        notes.append(f"bodies ok ({len(real_bodies)})")
    except Exception as exc:
        notes.append(f"bodies skipped ({type(exc).__name__}: {exc})")
        real_bodies = None

    if real_bodies:
        try:
            from fctx.solver.state import SolverState

            cfg = base_config()
            state = SolverState(real_bodies, cfg, cfg.device)
            renderer = Renderer(window, state, real_bodies, cfg)
            camera = OrbitCamera(cfg.camera, window.aspect)
            materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4)
                         for b in real_bodies]
            renderer.draw(camera, [], materials, None, fake_stats(), [])
            window.ctx.finish()
            err = window.ctx.error
            renderer.release()
            notes.append(f"solver ok ({err})")
        except Exception as exc:
            # The message, not just the type: a sibling subsystem breaking is
            # only useful information if it says what broke.
            notes.append(f"solver skipped ({type(exc).__name__}: {exc})")
    record("optional real subsystems", True, "; ".join(notes))


# --------------------------------------------------------------------------
# limits: nothing below may raise or produce a non-finite frame
# --------------------------------------------------------------------------


def _render_once(window: Window, cfg: AppConfig, bodies: list[BodyData],
                 state: object, *, hardness: float = 0.4,
                 poses: Sequence[HandPose] = (),
                 preview: np.ndarray | None = None,
                 toggles: dict[str, object] | None = None,
                 hardness_override: float | None = None,
                 **draw_kwargs: object) -> np.ndarray:
    """Build a renderer, draw one frame, tear it down, return the pixels."""
    if hardness_override is not None:
        draw_kwargs["hardness"] = hardness_override
    renderer = Renderer(window, state, bodies, cfg)
    for key, value in (toggles or {}).items():
        setattr(renderer, key, value)
    camera = OrbitCamera(cfg.camera, window.aspect)
    try:
        materials = [evaluate(DEFAULT_MATERIALS[b.kind], hardness)
                     for b in bodies]
        renderer.draw(camera, list(poses), materials, preview, fake_stats(),
                      ["source     test"], **draw_kwargs)  # type: ignore[arg-type]
        window.ctx.finish()
        err = window.ctx.error
        img = window.read_pixels()
    finally:
        renderer.release()
    assert err == "GL_NO_ERROR", err
    assert np.isfinite(img).all()
    return img


def test_hardness_extremes(window: Window, bodies: list[BodyData],
                           state: object) -> None:
    """Both ends of the dial, and the eleven points between, must render."""
    cfg = base_config(show_hud=False, show_webcam=False)
    frames = {}
    for h in (0.0, 0.5, 1.0):
        frames[h] = _render_once(window, cfg, bodies, state, hardness=h
                                 ).astype(np.int16)
        mean = float(frames[h].mean())
        assert 4.0 < mean < 250.0, f"hardness {h}: degenerate frame mean={mean}"
    spread = float(np.abs(frames[0.0] - frames[1.0]).max())
    changed = float((np.abs(frames[0.0] - frames[1.0]).max(axis=2) > 6).mean())
    ok = spread > 40.0 and changed > 0.05
    record("hardness 0.0 vs 1.0 look different", ok,
           f"max delta {spread:.0f}/255 over {changed * 100:.1f}% of the frame")
    assert ok, "the hardness dial changed nothing on screen"

    for h in np.linspace(0.0, 1.0, 11):
        img = _render_once(window, cfg, bodies, state, hardness=float(h))
        assert np.isfinite(img).all() and 4.0 < img.mean() < 250.0, h
    record("hardness sweep 0..1 in 11 steps", True, "all frames well formed")


def test_hand_extremes(window: Window, bodies: list[BodyData],
                       state: object) -> None:
    from fctx.config import TrackingConfig

    cfg = base_config()
    preview = fake_preview()
    cases: list[tuple[str, AppConfig, list[HandPose]]] = [
        ("zero hands", cfg, []),
        ("one hand", cfg, [fake_pose(0.0, False)]),
        ("max hands", cfg, [fake_pose(0.0, False), fake_pose(2.0, True)]),
        ("more poses than max_hands", cfg,
         [fake_pose(i * 0.4, i % 2 == 0) for i in range(5)]),
        ("max_hands=0", replace(cfg, tracking=replace(TrackingConfig(),
                                                      max_hands=0)),
         [fake_pose(0.0, True)]),
        ("max_hands=8, eight hands",
         replace(cfg, tracking=replace(TrackingConfig(), max_hands=8)),
         [fake_pose(i * 0.3, i % 3 == 0) for i in range(8)]),
    ]
    for label, variant, poses in cases:
        _render_once(window, variant, bodies, state, poses=poses,
                     preview=preview)
        record(f"hands: {label}", True, f"{len(poses)} pose(s)")

    nan = fake_pose(0.0, True)
    nan.joints[3] = np.nan
    nan.joints[7] = np.inf
    nan.confidence = float("nan")
    nan.pinch_point[:] = np.nan
    _render_once(window, cfg, bodies, state, poses=[nan], preview=preview)
    record("hands: NaN joints do not crash or leak into the frame", True)

    flat = fake_pose(0.0, False)
    flat.joints[:] = 0.0
    _render_once(window, cfg, bodies, state, poses=[flat], preview=preview)
    record("hands: every joint coincident", True, "zero-length capsules")


def test_config_extremes(window: Window, bodies: list[BodyData],
                         state: object) -> None:
    variants = [
        ("ssao_samples=1", base_config(ssao_samples=1)),
        ("ssao_samples=256", base_config(ssao_samples=256)),
        ("ssao_radius=0", base_config(ssao_radius=0.0)),
        ("ssao_strength=0", base_config(ssao_strength=0.0)),
        ("shadow_size=1", base_config(shadow_size=1)),
        ("shadow_size=8192", base_config(shadow_size=8192)),
        ("shadow_softness=0", base_config(shadow_softness=0.0)),
        ("msaa=0", base_config(msaa=0)),
        ("msaa=3", base_config(msaa=3)),
        ("msaa=32", base_config(msaa=32)),
        ("exposure=0", base_config(exposure=0.0)),
        ("bloom_threshold=0", base_config(bloom_threshold=0.0)),
        ("vignette=1", base_config(vignette=1.0)),
        ("aberration=0.05", base_config(chromatic_aberration=0.05)),
        ("webcam_scale=0", base_config(webcam_scale=0.0)),
        ("webcam_scale=10", base_config(webcam_scale=10.0)),
    ]
    for label, cfg in variants:
        _render_once(window, cfg, bodies, state, preview=fake_preview(),
                     poses=[fake_pose(0.0, True)])
        record(f"config: {label}", True)


def test_extreme_sizes(window: Window, bodies: list[BodyData],
                       state: object) -> None:
    """A bare context renders at the configured size, however absurd.

    The window manager clamps a real window to a minimum width, so the only
    way to prove the pipeline survives a one-pixel target -- where every bloom
    level and the half-resolution AO buffer all collapse to 1x1 -- is to drive
    it through a context whose size comes from the config.
    """
    seen = []
    for w, h in [(1, 1), (1, 720), (720, 1), (3, 5), (17, 9), (64, 36),
                 (4096, 64)]:
        cfg = base_config(width=w, height=h)
        renderer = Renderer(window.ctx, state, bodies, cfg)
        # A context that belongs to a window is resolved back to it, and the
        # window manager will not make a window this small.  Detaching it puts
        # the renderer on the bare-context path, where the size comes from the
        # config and nothing clamps it.
        renderer.window = None
        camera = OrbitCamera(cfg.camera, w, h)
        try:
            materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4)
                         for b in bodies]
            renderer.draw(camera, [fake_pose(0.0, True)], materials,
                          fake_preview(), fake_stats(), ["x"])
            window.ctx.finish()
            err = window.ctx.error
        finally:
            renderer.release()
            window.remove_teardown(renderer)
        assert err == "GL_NO_ERROR", f"{w}x{h}: {err}"
        seen.append(f"{w}x{h}")
    record("extreme render sizes", True, " ".join(seen))


def test_two_windows(bodies: list[BodyData], positions: np.ndarray,
                     normals: np.ndarray) -> None:
    """Closing one window must leave the other one's context usable.

    ``glfwDestroyWindow`` unbinds the context it destroyed, and with nothing
    made current afterwards every later GL call on the survivor fails.
    """
    cfg = base_config(width=320, height=240)
    first = Window(cfg)
    second = Window(cfg)
    try:
        renderer = Renderer(first, numpy_state(positions, normals), bodies,
                            cfg)
        second.close()
        first.make_current()
        camera = OrbitCamera(cfg.camera, first.aspect)
        renderer.draw(camera, [], [evaluate(DEFAULT_MATERIALS[b.kind], 0.4)
                                   for b in bodies], None, fake_stats(), [])
        first.ctx.finish()
        err = first.ctx.error
        img = first.read_pixels()
        renderer.release()
    finally:
        first.close()
    ok = err == "GL_NO_ERROR" and img.shape == (240, 320, 3) and img.mean() > 4.0
    record("closing a second window leaves the first usable", ok,
           f"{err} mean={img.mean():.1f}")
    assert ok

    # Closing in the order they were opened: the second window must not try
    # to hand the context back to the first, which no longer exists.
    import warnings

    oldest, newest = Window(cfg), Window(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oldest.close()
        newest.close()
    glfw_warnings = [str(w.message) for w in caught if "GLFW" in str(w.message)]
    record("closing windows oldest-first is quiet", not glfw_warnings,
           "; ".join(glfw_warnings)[:70] or "no GLFW error")
    assert not glfw_warnings


def test_state_array_abuse(window: Window, bodies: list[BodyData],
                           positions: np.ndarray, normals: np.ndarray) -> None:
    cfg = base_config()
    p = positions.shape[0]

    _render_once(window, cfg, bodies,
                 SimpleNamespace(x=positions.astype(np.float64),
                                 normal=normals.astype(np.float64),
                                 num_particles=p))
    record("state: float64 arrays are converted", True)

    interleaved = np.zeros((p, 6), np.float32)
    interleaved[:, :3] = positions
    interleaved[:, 3:] = normals
    assert not interleaved[:, :3].flags["C_CONTIGUOUS"]
    _render_once(window, cfg, bodies,
                 SimpleNamespace(x=interleaved[:, :3], normal=interleaved[:, 3:],
                                 num_particles=p))
    record("state: non-contiguous views are made contiguous", True)

    poisoned = positions.copy()
    poisoned[5] = np.nan
    poisoned[9] = np.inf
    img = _render_once(window, cfg, bodies,
                       SimpleNamespace(x=poisoned, normal=normals.copy(),
                                       num_particles=p))
    record("state: NaN/Inf positions still produce a frame", True,
           f"mean={img.mean():.1f}")

    _render_once(window, cfg, bodies,
                 SimpleNamespace(x=positions.copy(),
                                 normal=np.zeros_like(normals),
                                 num_particles=p))
    record("state: zero-length normals do not divide by zero", True)

    for label, state, exc in (
        ("shorter than the scene",
         SimpleNamespace(x=positions[:10], normal=normals[:10],
                         num_particles=p), ValueError),
        ("missing x", SimpleNamespace(normal=normals, num_particles=p),
         AttributeError),
        ("x is not an array",
         SimpleNamespace(x=[1, 2, 3], normal=normals, num_particles=p),
         (TypeError, ValueError)),
    ):
        renderer = Renderer(window, state, bodies, cfg)
        camera = OrbitCamera(cfg.camera, window.aspect)
        materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4) for b in bodies]
        try:
            renderer.draw(camera, [], materials, None, fake_stats(), [])
        except exc as raised:
            record(f"state: {label} raises", True, type(raised).__name__)
        else:
            raise AssertionError(f"state {label} was accepted silently")
        finally:
            renderer.release()


def test_preview_shapes(window: Window, bodies: list[BodyData],
                        state: object) -> None:
    cfg = base_config()
    cases = [
        ("1x1", np.full((1, 1, 3), 128, np.uint8)),
        ("rgba", np.full((37, 53, 4), 90, np.uint8)),
        # 13 * 3 bytes is not a multiple of four, so a texture uploaded with
        # the default GL row alignment would shear.
        ("row length not 4-aligned", np.full((11, 13, 3), 200, np.uint8)),
        ("non-contiguous", fake_preview()[:, ::-1]),
        ("tall", np.full((400, 40, 3), 60, np.uint8)),
    ]
    for label, preview in cases:
        _render_once(window, cfg, bodies, state, preview=preview,
                     poses=[fake_pose(0.0, False)])
        record(f"preview: {label}", True, f"{preview.shape}")

    renderer = Renderer(window, state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4) for b in bodies]
    try:
        for shape in ((20, 30, 3), (60, 40, 3), (20, 30, 3)):
            renderer.draw(camera, [], materials,
                          np.zeros(shape, np.uint8), fake_stats(), [])
        window.ctx.finish()
        err = window.ctx.error
        try:
            renderer.draw(camera, [], materials, np.zeros((4, 4), np.uint8),
                          fake_stats(), [])
        except ValueError as exc:
            detail = str(exc)[:48]
        else:
            raise AssertionError("a 2D preview was accepted")
    finally:
        renderer.release()
    assert err == "GL_NO_ERROR", err
    record("preview: shape changes between frames", True,
           f"texture rebuilt; bad shape raises ({detail})")


def test_empty_body(window: Window) -> None:
    """A body with no particles must not take the scene down with it."""
    cloth = fake_cloth(6)
    empty = BodyData(
        kind=MatterKind.SOFT, name="empty",
        positions=np.zeros((0, 3), np.float32),
        inv_mass=np.zeros(0, np.float32), flags=np.zeros(0, np.uint32),
        tri_idx=np.zeros((0, 3), np.int32), uv=np.zeros((0, 2), np.float32),
        **_empty_constraints(0))
    empty.validate()
    grain = fake_grain(64)
    bodies = [empty, cloth, grain]
    positions = np.concatenate([b.positions for b in bodies]).astype(np.float32)
    normals = np.concatenate([
        np.zeros((0, 3), np.float32),
        vertex_normals(cloth.positions, cloth.tri_idx),
        np.zeros((grain.num_particles, 3), np.float32),
    ]).astype(np.float32)
    state = numpy_state(positions, normals)

    cfg = base_config()
    renderer = Renderer(window, state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    try:
        # The empty body still owns a slot in ``materials``; if the renderer
        # paired materials with draws by zipping, the cloth would be shaded
        # with the grain's colour and nothing would say so.
        materials = [evaluate(DEFAULT_MATERIALS[MatterKind.SOFT], 0.9),
                     evaluate(DEFAULT_MATERIALS[MatterKind.CLOTH], 0.0),
                     evaluate(DEFAULT_MATERIALS[MatterKind.GRAIN], 0.5)]
        renderer.draw(camera, [], materials, None, fake_stats(), [])
        window.ctx.finish()
        err = window.ctx.error
        indices = [d.index for d in renderer.draws]
    finally:
        renderer.release()
    ok = err == "GL_NO_ERROR" and indices == [1, 2]
    record("body with zero particles", ok,
           f"{err}; draws keep their material index {indices}")
    assert ok


def test_lifecycle(window: Window, bodies: list[BodyData],
                   state: object) -> None:
    cfg = base_config()
    camera = OrbitCamera(cfg.camera, window.aspect)
    materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4) for b in bodies]

    renderer = Renderer(window, state, bodies, cfg)
    renderer.release()
    renderer.release()
    for name, call in (("draw", lambda: renderer.draw(
                            camera, [], materials, None, fake_stats(), [])),
                       ("repaint", lambda: renderer.repaint(
                            window.framebuffer)),
                       ("rebind", lambda: renderer.rebind(state, bodies))):
        try:
            call()
        except RuntimeError as exc:
            record(f"{name} after release raises", True, str(exc)[:46])
        else:
            raise AssertionError(f"{name} after release was accepted")

    fresh = Renderer(window, state, bodies, cfg)
    try:
        fresh.repaint(window.framebuffer)
    except RuntimeError as exc:
        record("repaint before the first draw raises", True, str(exc)[:46])
    else:
        raise AssertionError("repaint before draw was accepted")
    finally:
        fresh.release()

    renderer = Renderer(window, state, bodies, cfg)
    try:
        renderer.draw(camera, [], materials, None, fake_stats(),
                      [f"line {i:04d} " + "x" * 70 for i in range(400)])
    except RuntimeError as exc:
        record("overlay overflow fails loudly", True, str(exc)[:56])
    else:
        raise AssertionError("the overlay batch overflowed silently")
    finally:
        renderer.release()


def test_rebind(window: Window, bodies: list[BodyData], positions: np.ndarray,
                normals: np.ndarray) -> None:
    """Switching preset swaps the scene without rebuilding the renderer."""
    cfg = base_config()
    renderer = Renderer(window, numpy_state(positions, normals), bodies, cfg)
    camera = OrbitCamera(cfg.camera, window.aspect)
    try:
        materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4) for b in bodies]
        renderer.draw(camera, [], materials, None, fake_stats(), [])

        cloth = fake_cloth(24)
        new_state = numpy_state(
            cloth.positions,
            vertex_normals(cloth.positions, cloth.tri_idx))
        renderer.rebind(new_state, [cloth])
        renderer.draw(camera, [], [evaluate(DEFAULT_MATERIALS[cloth.kind], 0.8)],
                      None, fake_stats(), [])
        window.ctx.finish()
        err = window.ctx.error
        img = window.read_pixels()
        counts = (renderer.num_particles, len(renderer.draws))

        renderer.rebind(numpy_state(positions, normals), bodies)
        renderer.draw(camera, [], materials, None, fake_stats(), [])
        window.ctx.finish()
        err2 = window.ctx.error
    finally:
        renderer.release()
    ok = err == "GL_NO_ERROR" and err2 == "GL_NO_ERROR" and img.mean() > 4.0
    record("rebind to a new scene and back", ok,
           f"{counts[0]} particles / {counts[1]} draw(s) mid-way; {err2}")
    assert ok


def test_visible_window_resize(bodies: list[BodyData], positions: np.ndarray,
                               normals: np.ndarray) -> None:
    """Enlarging a real window must not leave a stale band down its edge.

    ``moderngl`` measures the default framebuffer once, at context creation,
    so a resize leaves the composite drawing into the original rectangle and
    ``read_pixels`` reading a region that no longer exists.
    """
    import glfw

    cfg = replace(base_config(width=760, height=540), headless=False)
    win = Window(cfg)
    try:
        renderer = Renderer(win, numpy_state(positions, normals), bodies, cfg)
        camera = OrbitCamera(cfg.camera, win.aspect)
        materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4) for b in bodies]
        renderer.draw(camera, [], materials, None, fake_stats(), [])

        glfw.set_window_size(win.handle, 1180, 380)
        for _ in range(4):
            win.poll()
        assert win.size == (1180, 380), win.size
        renderer.draw(camera, [], materials, None, fake_stats(), [])
        win.ctx.finish()
        err = win.ctx.error
        img = win.read_pixels()
        after_read = win.ctx.error
        renderer.release()
    finally:
        win.close()

    rows = img.mean(axis=(1, 2))
    cols = img.mean(axis=(0, 2))
    dead = int((rows < 1.0).sum()) + int((cols < 1.0).sum())
    ok = (err == "GL_NO_ERROR" and after_read == "GL_NO_ERROR"
          and img.shape == (380, 1180, 3) and dead == 0)
    record("visible window resize covers the whole frame", ok,
           f"{img.shape} dead rows+cols={dead} err={err}/{after_read}")
    assert ok


def test_warp_state_on_another_device(window: Window, bodies: list[BodyData],
                                      positions: np.ndarray,
                                      normals: np.ndarray) -> None:
    """A solver running on the CPU must still feed the GPU's vertex buffers.

    ``FCTX_DEVICE=cpu`` is the documented way to take the solver off the GPU,
    and the renderer's interop path then has a host-side source array on one
    side of the copy and CUDA-mapped GL memory on the other.
    """
    import warp as wp

    wp.init()
    cfg = base_config()
    state = SimpleNamespace(
        x=wp.array(positions, dtype=wp.vec3, device="cpu"),
        normal=wp.array(normals, dtype=wp.vec3, device="cpu"),
        num_particles=positions.shape[0])
    img = _render_once(window, cfg, bodies, state)
    record("warp state on a different device from the GL context", True,
           f"mean={img.mean():.1f}")


def test_screenshot_after_swap(bodies: list[BodyData], positions: np.ndarray,
                               normals: np.ndarray) -> None:
    """A screenshot must be the frame the user saw, either side of the swap.

    After ``glfwSwapBuffers`` the back buffer is undefined, and on this driver
    it reads back solid black, so the renderer repaints the frame instead.
    """
    cfg = replace(base_config(width=620, height=420), headless=False)
    win = Window(cfg)
    try:
        renderer = Renderer(win, numpy_state(positions, normals), bodies, cfg)
        camera = OrbitCamera(cfg.camera, win.aspect)
        materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.62) for b in bodies]
        renderer.draw(camera, [fake_pose(0.0, True)], materials,
                      fake_preview(), fake_stats(), ["source     test"])
        before = win.read_pixels().astype(np.int16)
        win.swap()
        after = win.read_pixels().astype(np.int16)
        err = win.ctx.error

        # A resize between the swap and the screenshot leaves the repaint
        # drawing a frame composed for the old extent into a buffer of the
        # new one; it must still produce an image of the right shape.
        import glfw

        glfw.set_window_size(win.handle, 900, 300)
        for _ in range(4):
            win.poll()
        resized = win.read_pixels()
        resized_err = win.ctx.error
        renderer.release()
    finally:
        win.close()
    delta = float(np.abs(before - after).mean())
    ok = (err == "GL_NO_ERROR" and resized_err == "GL_NO_ERROR"
          and after.mean() > 4.0 and delta < 1.0
          and resized.shape == (300, 900, 3) and resized.mean() > 4.0)
    record("screenshot after swap matches the presented frame", ok,
           f"mean {before.mean():.1f} -> {after.mean():.1f}, "
           f"mean|delta|={delta:.3f}; after resize {resized.shape}")
    assert ok


def test_effects_are_visible(window: Window, bodies: list[BodyData],
                             state: object) -> None:
    """Each pass has to change the picture, or it is dead code on the GPU."""
    plain = base_config(show_hud=False, show_webcam=False)
    reference = _render_once(window, plain, bodies, state).astype(np.int16)

    def delta(img: np.ndarray, threshold: int) -> tuple[int, float]:
        d = np.abs(img.astype(np.int16) - reference).max(axis=2)
        return int(d.max()), float((d > threshold).mean())

    checks = [
        ("shadows", _render_once(window, plain, bodies, state,
                                 toggles={"shadows_enabled": False}), 6, 0.01),
        ("ssao", _render_once(window, plain, bodies, state,
                              toggles={"ssao_enabled": False}), 3, 0.002),
        ("wireframe", _render_once(window, plain, bodies, state,
                                   toggles={"wireframe": True}), 6, 0.02),
        ("hands", _render_once(window, plain, bodies, state,
                               poses=[fake_pose(0.0, True)]), 6, 0.005),
        ("hud", _render_once(window, base_config(show_webcam=False), bodies,
                             state), 6, 0.05),
        ("webcam inset", _render_once(window, base_config(show_hud=False),
                                      bodies, state,
                                      preview=fake_preview()), 6, 0.01),
    ]
    for label, img, threshold, min_fraction in checks:
        peak, fraction = delta(img, threshold)
        ok = fraction >= min_fraction
        record(f"visible effect: {label}", ok,
               f"peak {peak}/255 over {fraction * 100:.2f}% of the frame")
        assert ok, f"{label} changed nothing on screen"

    bright = base_config(show_hud=False, show_webcam=False, bloom_threshold=0.0,
                         bloom_strength=0.5)
    with_bloom = _render_once(window, bright, bodies, state)
    without = _render_once(window, bright, bodies, state,
                           toggles={"bloom_enabled": False})
    d = np.abs(with_bloom.astype(np.int16) - without.astype(np.int16))
    ok = float(d.max()) > 6.0
    record("visible effect: bloom", ok, f"peak {d.max():.0f}/255")
    assert ok


def test_shader_library_errors(window: Window) -> None:
    import tempfile

    from fctx.render import ShaderError

    root = Path(tempfile.mkdtemp(prefix="fctx-shader-"))
    (root / "a.glsl").write_text('#version 430 core\n#include "b.glsl"\n')
    (root / "b.glsl").write_text('#include "a.glsl"\n')
    (root / "no_version.frag").write_text("void main() {}\n")
    (root / "ok.vert").write_text(
        "#version 430 core\nvoid main() { gl_Position = vec4(0.0); }\n")
    (root / "broken.frag").write_text(
        "#version 430 core\nout vec4 f;\nvoid main() { f = nonsense(); }\n")

    lib = ShaderLibrary(window.ctx, root=root)
    cases = [
        ("circular include", lambda: lib.source("a.glsl")),
        ("missing file", lambda: lib.source("absent.frag")),
        ("missing #version", lambda: lib.source("no_version.frag")),
        ("compile failure", lambda: lib.program(
            "broken", vertex="ok.vert", fragment="broken.frag")),
    ]
    for label, call in cases:
        try:
            call()
        except ShaderError as exc:
            record(f"shader library: {label} raises", True,
                   str(exc).splitlines()[0][:58])
        else:
            raise AssertionError(f"{label} was accepted")
    lib.release()

    # The real library must also splice a missing include loudly rather than
    # emitting source with the directive left in it.
    real = ShaderLibrary(window.ctx)
    src = real.source("particle.frag")
    assert "#include" not in src and src.count("#version") == 1
    record("shader library: one #version survives expansion", True,
           f"{len(src.splitlines())} lines")
    real.release()


def test_particle_buffers_contract(window: Window) -> None:
    ctx = window.ctx
    try:
        ParticleBuffers(ctx, 0, AppConfig().device)
    except ValueError as exc:
        record("ParticleBuffers: zero particles raises", True, str(exc)[:46])
    else:
        raise AssertionError("zero particles accepted")

    host = ParticleBuffers(ctx, 64, AppConfig().device, prefer_interop=False)
    assert not host.interop
    data = np.arange(64 * 3, dtype=np.float32).reshape(64, 3)
    host.update_from(data, data)
    back = np.frombuffer(host.positions.read(), np.float32).reshape(64, 3)
    assert np.array_equal(back, data), "host staging lost the data"
    host.map_for_warp()
    host.unmap()
    host.release()
    record("ParticleBuffers: host staging round trip", True, host.describe())

    buf = ParticleBuffers(ctx, 32, AppConfig().device)
    buf.map_for_warp()
    for label, call in (("double map", buf.map_for_warp),
                        ("update while mapped",
                         lambda: buf.update_from(data[:32], data[:32]))):
        try:
            call()
        except InteropError:
            record(f"ParticleBuffers: {label} raises", True)
        else:
            raise AssertionError(f"{label} accepted")
    # release() has to cope with a buffer still handed to CUDA, because that
    # is the state an exception mid-frame leaves it in.
    buf.release()
    record("ParticleBuffers: release while mapped", True, "no exception")


def test_renderer_contract(window: Window, bodies: list[BodyData],
                           positions: np.ndarray, normals: np.ndarray) -> None:
    cfg = base_config()
    state = numpy_state(positions, normals)
    p = positions.shape[0]

    for label, args, exc in (
        ("no bodies", (state, []), ValueError),
        ("body_offsets of the wrong length",
         (SimpleNamespace(x=positions, normal=normals, num_particles=p,
                          body_offsets=[0, 1]), bodies), ValueError),
        ("state smaller than the bodies",
         (SimpleNamespace(x=positions, normal=normals, num_particles=10),
          bodies), ValueError),
    ):
        try:
            Renderer(window, args[0], args[1], cfg)
        except exc as raised:
            record(f"renderer rejects {label}", True, str(raised)[:54])
        else:
            raise AssertionError(f"{label} accepted")

    offsets, running = [], 0
    for body in bodies:
        offsets.append(running)
        running += body.num_particles
    published = SimpleNamespace(x=positions, normal=normals,
                                num_particles=running,
                                body_offsets=np.asarray(offsets, np.int32))
    renderer = Renderer(window, published, bodies, cfg)
    try:
        camera = OrbitCamera(cfg.camera, window.aspect)
        renderer.draw(camera, [], [evaluate(DEFAULT_MATERIALS[b.kind], 0.4)
                                   for b in bodies], None, fake_stats(), [])
        window.ctx.finish()
        err = window.ctx.error
        used = [d.offset for d in renderer.draws]
    finally:
        renderer.release()
    assert err == "GL_NO_ERROR", err
    record("renderer honours state.body_offsets", used == offsets,
           f"{used}")
    assert used == offsets


def test_camera_limits() -> None:
    from fctx.render.camera import look_at, orthographic

    cfg = base_config().camera

    cam = OrbitCamera(cfg, 1920, 1080)
    assert abs(cam.aspect - 16 / 9) < 1e-9, cam.aspect
    assert np.isfinite(OrbitCamera(cfg, 1920, 0).projection()).all()
    assert np.isfinite(OrbitCamera(cfg, 0.0).projection()).all()

    cam = OrbitCamera(cfg, 16 / 9)
    cam.light_dir = np.array([0.0, -1.0, 0.0])
    assert np.isfinite(cam.light_view_proj()).all(), "light straight down"

    # frame_stage on a tiny object used to invert the light frustum: the far
    # plane scales with the stage while the near plane did not.
    for radius in (1e-3, 1e-2, 0.08, 4.0):
        cam.frame_stage(np.array([0.0, 0.3, 0.0]), radius)
        vp = cam.light_view_proj()
        assert np.isfinite(vp).all(), radius
        centre = vp @ np.append(cam.stage_center, 1.0)
        centre = centre[:3] / centre[3]
        assert np.all(np.abs(centre) <= 1.0 + 1e-6), (radius, centre)
        assert cam.shadow_depth_range > 0.0

    cam = OrbitCamera(cfg, 16 / 9)
    for _ in range(200):
        cam.zoom(10.0)
    assert cam.distance >= 0.18
    for _ in range(200):
        cam.zoom(-10.0)
    assert cam.distance <= 12.0
    for _ in range(200):
        cam.orbit(1e5, 1e5)
    assert np.isfinite(cam.position).all() and np.isfinite(cam.view_proj()).all()

    # Eye on the up axis, and eye exactly at the target: both must give a
    # usable matrix rather than one full of NaN.
    assert np.isfinite(OrbitCamera(replace(cfg, position=(0.0, 1.3, 0.0)), 1.0)
                       .view()).all()
    assert np.isfinite(OrbitCamera(replace(cfg, position=cfg.target), 1.0)
                       .view()).all()
    try:
        look_at(np.zeros(3), np.zeros(3), np.array([0.0, 1.0, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("look_at accepted a degenerate eye/target")
    try:
        orthographic(0.0, 1.0, 0.1, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("orthographic accepted a zero extent")
    record("camera limits", True, "tiny stage, poles, saturation, bad input")


def test_input_translation() -> None:
    """Raw glfw events must arrive as the names fctx.ui.controls matches on."""
    import glfw

    from fctx.render.context import InputEvent as RawEvent
    from fctx.ui.controls import Controls, EventKind

    cfg = base_config()
    win = Window(cfg)
    try:
        win.events.extend([
            RawEvent("key", key=glfw.KEY_R, action=glfw.PRESS),
            RawEvent("key", key=glfw.KEY_RIGHT_BRACKET, action=glfw.PRESS),
            RawEvent("key", key=glfw.KEY_RIGHT_BRACKET, action=glfw.REPEAT),
            RawEvent("key", key=glfw.KEY_RIGHT_BRACKET, action=glfw.RELEASE),
            RawEvent("key", key=glfw.KEY_F11, action=glfw.PRESS),
            RawEvent("key", key=glfw.KEY_2, action=glfw.PRESS),
            RawEvent("key", key=glfw.KEY_ESCAPE, action=glfw.PRESS),
            RawEvent("mouse", button=glfw.MOUSE_BUTTON_RIGHT,
                     action=glfw.PRESS, x=10.0, y=20.0),
            RawEvent("mouse", button=glfw.MOUSE_BUTTON_RIGHT,
                     action=glfw.RELEASE, x=40.0, y=60.0),
            RawEvent("scroll", dy=2.0),
            RawEvent("cursor", x=30.0, y=40.0),
            RawEvent("resize", x=640, y=480),
            RawEvent("close"),
        ])
        events = win.poll_events()
    finally:
        win.close()

    names = [(e.kind, e.name) for e in events]
    expected_keys = [("r", EventKind.KEY_DOWN), ("]", EventKind.KEY_DOWN),
                     ("]", EventKind.KEY_UP), ("f11", EventKind.KEY_DOWN),
                     ("2", EventKind.KEY_DOWN), ("escape", EventKind.KEY_DOWN)]
    for name, kind in expected_keys:
        assert (kind, name) in names, f"{kind} {name!r} missing from {names}"
    # The OS auto-repeat is dropped: Controls ramps a held key itself, and
    # feeding it the repeat as well makes the dial jump at the repeat rate.
    assert sum(1 for k, n in names if n == "]" and k is EventKind.KEY_DOWN) == 1
    assert (EventKind.CLOSE, "") in names
    assert any(e.kind is EventKind.RESIZE for e in events)
    assert any(e.kind is EventKind.SCROLL and e.y == 2.0 for e in events)
    assert any(e.kind is EventKind.MOUSE_DOWN and e.name == "right"
               for e in events)

    state = Controls().handle(events)
    ok = (state.quit_requested and state.reset_requested
          and state.fullscreen_requested and state.preset_requested is not None
          and state.hardness_target > 0.35)
    record("glfw events translate into control events", ok,
           f"{len(events)} events -> preset={state.preset_requested} "
           f"hardness={state.hardness_target:.2f}")
    assert ok


def test_hud_badges(window: Window, bodies: list[BodyData],
                    state: object) -> None:
    """Notifications and the paused marker draw with the HUD hidden too."""
    cfg = base_config(show_webcam=False)
    notes = [SimpleNamespace(text="GRABBED 312 PARTICLES", alpha=1.0),
             SimpleNamespace(text="CUBE  4,096 particles", alpha=0.35),
             SimpleNamespace(text="faded out", alpha=0.0)]
    plain = _render_once(window, cfg, bodies, state,
                         toggles={"show_hud": False},
                         show_hud=False).astype(np.int16)
    badged = _render_once(window, cfg, bodies, state,
                          toggles={"show_hud": False}, show_hud=False,
                          notifications=notes, paused=True).astype(np.int16)
    changed = float((np.abs(plain - badged).max(axis=2) > 6).mean())
    ok = changed > 0.002
    record("notifications and paused badge draw without the HUD", ok,
           f"{changed * 100:.2f}% of the frame")
    assert ok

    # The dial follows the control value, not any one body's material.
    left = _render_once(window, base_config(show_webcam=False), bodies, state,
                        hardness=0.5, hardness_override=0.0)
    right = _render_once(window, base_config(show_webcam=False), bodies, state,
                         hardness=0.5, hardness_override=1.0)
    moved = float((np.abs(left.astype(np.int16)
                          - right.astype(np.int16)).max(axis=2) > 6).mean())
    record("dial knob follows the hardness override", moved > 0.0005,
           f"{moved * 100:.2f}% of the frame")
    assert moved > 0.0005


def test_unicode_labels(window: Window, bodies: list[BodyData],
                        state: object) -> None:
    """Text outside the ASCII atlas draws through the label cache, bounded."""
    cfg = base_config(show_webcam=False)
    plain = _render_once(window, cfg, bodies, state,
                         toggles={"show_hud": False}, show_hud=False).astype(np.int16)
    notes = [SimpleNamespace(text="親指と人差し指でつまんでみてください", alpha=1.0),
             SimpleNamespace(text="~ シリコーンゴム", alpha=0.8)]
    labelled = _render_once(window, cfg, bodies, state,
                            toggles={"show_hud": False}, show_hud=False,
                            notifications=notes).astype(np.int16)
    err = window.ctx.error
    record("unicode labels draw without GL error", err == "GL_NO_ERROR", err)
    assert err == "GL_NO_ERROR"
    changed = float((np.abs(plain - labelled).max(axis=2) > 6).mean())
    ok = changed > 0.002
    record("a Japanese prompt changes the frame", ok, f"{changed * 100:.2f}% of the frame")
    assert ok

    # The cache is keyed by string and size, reused, and bounded.
    from fctx.render.text import LabelCache
    cache = LabelCache(window.ctx, capacity=8)
    tex, w, h = cache.get("つまむ", 24.0)
    again = cache.get("つまむ", 24.0)
    record("label cache reuses a string", again[0] is tex, f"{w}x{h}, font {cache.source}")
    assert again[0] is tex and w > 4 and h > 4
    for i in range(20):
        cache.get(f"ラベル{i}", 24.0)
    record("label cache is bounded", len(cache) <= 8, f"{len(cache)} entries after 21 strings")
    assert len(cache) <= 8
    cache.release()
    assert len(cache) == 0


def _grain_slab(radius: float = 0.0048, nx: int = 26, ny: int = 3,
                nz: int = 26, centre=(0.0, 0.30, 0.0)) -> BodyData:
    """A dense raft of grains at the shipped grain radius.

    Deterministic: the contact-shadow test measures a luminance difference of
    a fraction of a level, so the two renders it compares have to be of
    exactly the same arrangement of grains.
    """
    rng = np.random.default_rng(7)
    grid = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                indexing="ij"), axis=-1).reshape(-1, 3)
    pos = grid.astype(np.float32) * (radius * 2.02)
    pos += rng.normal(0.0, radius * 0.06, pos.shape).astype(np.float32)
    pos -= pos.mean(axis=0)
    pos += np.asarray(centre, np.float32)
    n = pos.shape[0]
    return BodyData(
        kind=MatterKind.GRAIN,
        name="grain_slab",
        positions=np.ascontiguousarray(pos, np.float32),
        inv_mass=np.ones(n, np.float32),
        flags=np.full(n, FLAG_SELF_COLLIDE, np.uint32),
        tri_idx=np.zeros((0, 3), np.int32),
        uv=np.zeros((n, 2), np.float32),
        double_sided=False,
        particle_radius=radius,
        **_empty_constraints(n),
    )


def test_grain_contact_shadow(window: Window) -> None:
    """Grains must keep the shadows they cast on each other.

    The slope bias in ``shadow.glsl`` has a branch that multiplies the offset
    by up to 43x.  It is written for a closed body drawn back-face only, whose
    shadow map holds its *far* surface and which therefore cannot detach a
    contact shadow.  Grains are point impostors drawn with culling off, so the
    map holds their near surface: given that branch, the lookup lands several
    grains away and the pile stops reading as touching spheres.

    Rendered here against a copy of the shader tree that asks for the
    closed-body bias, which is what the code did before.
    """
    import shutil
    import tempfile

    from fctx.render import pipeline as pipeline_mod
    from fctx.render import shaders as shaders_mod

    body = _grain_slab()
    state = numpy_state(body.positions, np.zeros_like(body.positions))
    cfg = base_config(show_hud=False, show_webcam=False)

    tmp = Path(tempfile.mkdtemp(prefix="fctx_shadow_"))
    try:
        shutil.copytree(shaders_mod.SHADER_DIR, tmp / "shaders")
        patched = tmp / "shaders" / "particle.frag"
        text = patched.read_text(encoding="utf-8")
        patched.write_text(text.replace("SHADOW_NEAR", "SHADOW_CLOSED"),
                           encoding="utf-8")

        def render(root: Path | None) -> np.ndarray:
            # Renderer builds its own ShaderLibrary, so the variant tree has
            # to be swapped in at the module the pipeline looks it up on.
            original = pipeline_mod.ShaderLibrary
            if root is not None:
                class _Variant(original):  # type: ignore[valid-type,misc]
                    def __init__(self, ctx, *a, **k):
                        super().__init__(ctx, root)

                pipeline_mod.ShaderLibrary = _Variant
            try:
                renderer = Renderer(window, state, [body], cfg)
                renderer.show_hud = False
                camera = OrbitCamera(cfg.camera, WIDTH / HEIGHT)
                # The bias is in shadow texels, so the framing decides how big
                # it is in metres.  This is what the grain preset frames to.
                camera.frame_stage(np.array([0.0, 0.30, 0.0]), 0.212)
                material = evaluate(DEFAULT_MATERIALS[body.kind], 0.5)
                try:
                    renderer.draw(camera, [], [material], None, fake_stats(),
                                  [], show_hud=False)
                    window.ctx.finish()
                    assert window.ctx.error == "GL_NO_ERROR"
                    return window.read_pixels()
                finally:
                    renderer.release()
            finally:
                pipeline_mod.ShaderLibrary = original

        shipped = render(None)
        closed_bias = render(tmp / "shaders")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    def luminance(img: np.ndarray) -> np.ndarray:
        f = img[:, :, :3].astype(np.float32)
        return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]

    lit, old = luminance(shipped), luminance(closed_bias)
    pile = lit > 90.0  # the pale grains; the floor is far darker
    assert pile.sum() > 20000, f"only {pile.sum()} pile pixels; scene mislaid"
    brightened = float(((old - lit)[pile] > 20.0).mean())
    mean_drop = float(old[pile].mean() - lit[pile].mean())
    ok = brightened > 0.003 and mean_drop > 0.15
    record("grains keep their contact shadows", ok,
           f"the closed-body bias brightens {brightened * 100:.2f}% of the "
           f"pile by >20 levels, mean {lit[pile].mean():.2f} -> "
           f"{old[pile].mean():.2f}")
    assert ok


def test_timer_owner_survives_another_release(window: Window,
                                              bodies: list[BodyData],
                                              state: object) -> None:
    """Releasing one renderer must not hand away another's timer query.

    Only one GL_TIME_ELAPSED query may be in flight per context.  If release()
    clears the ownership slot it does not hold, the next renderer opens a
    second query on top of a running one and the GL_INVALID_OPERATION that
    produces is latched onto whatever GL call is checked next.
    """
    from fctx.render.pipeline import _TIMER_OWNERS, _timer_busy

    cfg = base_config(width=320, height=180)
    holder = Renderer(window, state, bodies, cfg)
    other = Renderer(window, state, bodies, cfg)
    ctx = window.ctx
    while ctx.error != "GL_NO_ERROR":
        pass
    if holder._timer is None:
        record("timer ownership survives another release", True,
               "no GL_TIME_ELAPSED on this driver; nothing to own")
        holder.release()
        other.release()
        return

    holder._timer.__enter__()
    holder._timer_open = True
    _TIMER_OWNERS[ctx] = holder
    try:
        other.release()
        still_owned = _TIMER_OWNERS.get(ctx) is holder and _timer_busy(ctx)
    finally:
        holder._timer.__exit__(None, None, None)
        holder._timer_open = False
        holder.release()
    err = ctx.error
    ok = still_owned and err == "GL_NO_ERROR"
    record("timer ownership survives another release", ok,
           f"owner kept={still_owned} {err}")
    assert ok

    # And the number the HUD prints as 'draw' has to actually arrive.  It is
    # read one frame late, so the second draw is the first that can report.
    renderer = Renderer(window, state, bodies, cfg)
    camera = OrbitCamera(cfg.camera, 16 / 9)
    materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4) for b in bodies]
    try:
        for _ in range(3):
            renderer.draw(camera, [], materials, None, fake_stats(), ["x"])
        window.ctx.finish()
        gpu_ms = renderer.gpu_ms
    finally:
        renderer.release()
    ok = math.isfinite(gpu_ms) and gpu_ms > 0.0
    record("the GPU timer reports a frame time", ok, f"{gpu_ms:.3f} ms")
    assert ok


class _RecordingBatch:
    """Stands in for :class:`OverlayBatch` and keeps what was asked for.

    The HUD is pure layout arithmetic, so what it emits is worth checking
    directly: a pixel comparison cannot say whether two strings were drawn on
    top of each other, only that the result changed.
    """

    def __init__(self, atlas: TextAtlas) -> None:
        self.atlas = atlas
        self.quads: list[tuple[float, float, float, float]] = []
        self.texts: list[tuple[str, float, float, float, float]] = []

    def rect(self, x, y, w, h, color, *, radius=0.0, rotation=0.0,
             softness=1.0) -> None:
        self.quads.append((x, y, w, h))

    def gradient(self, x, y, w, h, left, right, *, radius=0.0) -> None:
        self.quads.append((x, y, w, h))

    def image(self, x, y, w, h, **kwargs) -> None:
        self.quads.append((x, y, w, h))

    def circle(self, cx, cy, r, color) -> None:
        self.quads.append((cx - r, cy - r, 2.0 * r, 2.0 * r))

    def line(self, x0, y0, x1, y1, width, color) -> None:
        self.quads.append((min(x0, x1), min(y0, y1),
                           abs(x1 - x0), max(abs(y1 - y0), width)))

    def measure(self, s, size_px) -> tuple[float, float]:
        return self.atlas.measure(s, size_px)

    def text(self, s, x, y, size_px, color, *, align="left") -> float:
        w = len(s) * self.atlas.advance(size_px)
        x0 = x - (w * 0.5 if align == "center" else w if align == "right" else 0.0)
        self.quads.append((x0, y, w, size_px))
        self.texts.append((s, x0, y, w, size_px))
        return w


def test_hud_layout_fits_the_window(window: Window) -> None:
    """Nothing the HUD draws may leave the window or land on another string.

    The panels are sized from the window *height* and the space they have from
    its *width*, so a window narrower than about two thirds of its height used
    to push the stats panel off the right edge and draw the material's state
    word straight through its modulus.
    """
    from fctx.render.hud import CONTROL_HINT, Hud

    atlas = TextAtlas(window.ctx)
    hud = Hud()
    # The longest line the shipped application ever puts in the panel.
    lines = ["synthetic hand (autopilot, 60 fps)",
             "SYNTHETIC HAND: move the mouse to steer, left-click or SPACE to pinch",
             "Q / E push the hand away / pull it closer,  C curls the fingers"]
    materials = [evaluate(DEFAULT_MATERIALS[MatterKind.SOFT], 0.4)]

    sizes = [(3840, 2160), (2560, 1080), (1920, 1080), (1600, 900),
             (1366, 768), (1280, 720), (1024, 768), (800, 600), (640, 360),
             (600, 1080), (480, 270), (400, 900), (320, 180)]
    worst_over = 0.0
    worst_at = ""
    collisions: list[str] = []
    for w, h in sizes:
        # Under 270 px of height the two panels together are taller than the
        # window: the type stops shrinking at scale 0.6 on purpose, since a
        # HUD nobody can read is not worth laying out.  Those windows still
        # have to keep everything inside the frame, they just cannot also
        # keep the panels out of each other.
        crowded = h < 270
        batch = _RecordingBatch(atlas)
        hud.build(batch, (w, h), materials, fake_stats(), list(lines),
                  hardness=0.4)
        # Badges are meant to sit on top of whatever is behind them, so they
        # are held to the window bounds but not to the collision rule.
        badges = _RecordingBatch(atlas)
        hud.badges(badges, (w, h),
                   [SimpleNamespace(text="BANNER  7,744 particles", alpha=1.0)],
                   paused=True)
        for x, y, qw, qh in batch.quads + badges.quads:
            # 2 px of slack: the panel edge is a stroke drawn 1.5 px outside.
            over = max(-x, -y, (x + qw) - w, (y + qh) - h) - 2.0
            if over > worst_over:
                worst_over, worst_at = over, f"{w}x{h}"
        if crowded:
            continue
        for i, (s0, x0, y0, w0, h0) in enumerate(batch.texts):
            for s1, x1, y1, w1, h1 in batch.texts[i + 1:]:
                rows_meet = y0 < y1 + h1 * 0.7 and y1 < y0 + h0 * 0.7
                cols_meet = x0 < x1 + w1 and x1 < x0 + w0
                if rows_meet and cols_meet:
                    collisions.append(f"{w}x{h}: {s0!r} over {s1!r}")
    ok = worst_over <= 0.0 and not collisions
    record("HUD fits every window shape", ok,
           f"worst overflow {worst_over:.1f} px"
           + (f" at {worst_at}" if worst_at else "")
           + (f"; {collisions[0]}" if collisions else ""))
    assert ok, collisions[:3]

    # The strip is the only place a viewer of a recording can learn the keys,
    # so it has to name every one the documentation does.
    missing = [k for k in ("WHEEL", "CTRL+WHEEL", "RIGHT-DRAG", "1-5", "R ", "D ",
                           "P ", ". step", "H ", "W ", "K ", "G ", "F ", "A ",
                           "F9", "F11", "F12", "ESC")
               if k not in CONTROL_HINT]
    record("the control strip names every documented key", not missing,
           f"missing {missing}" if missing else f"{len(CONTROL_HINT)} chars")
    assert not missing


def test_no_leak(window: Window, bodies: list[BodyData],
                 state: object) -> None:
    """Building and releasing a renderer must not accumulate GL objects."""
    import gc

    cfg = base_config(width=512, height=288)
    camera = OrbitCamera(cfg.camera, 16 / 9)
    materials = [evaluate(DEFAULT_MATERIALS[b.kind], 0.4) for b in bodies]

    def cycle() -> None:
        renderer = Renderer(window, state, bodies, cfg)
        renderer.draw(camera, [fake_pose(0.0, True)], materials,
                      fake_preview(), fake_stats(), ["x"])
        renderer.release()

    kinds = (moderngl.Texture, moderngl.Framebuffer, moderngl.Buffer,
             moderngl.VertexArray, moderngl.Program)

    def live() -> tuple[int, int]:
        """Count GL objects still holding a driver handle, and dead wrappers.

        ``release()`` swaps a moderngl object's implementation for an
        ``InvalidObject`` rather than deleting the Python wrapper, so counting
        wrappers measures nothing.  The dead count matters too: it is how a
        renderer that stays referenced after release -- by the window's
        teardown list, say -- shows up.
        """
        gc.collect()
        alive = dead = 0
        for obj in gc.get_objects():
            if isinstance(obj, kinds):
                if type(obj.mglo).__name__ == "InvalidObject":
                    dead += 1
                else:
                    alive += 1
        return alive, dead

    for _ in range(3):
        cycle()
    before, dead_before = live()
    for _ in range(10):
        cycle()
    after, dead_after = live()
    ok = after <= before and dead_after <= dead_before + 4
    record("renderer create/release does not leak", ok,
           f"{before} -> {after} live GL objects, "
           f"{dead_before} -> {dead_after} released wrappers still referenced")
    assert ok


# --------------------------------------------------------------------------


def main() -> int:
    print(f"fctx.render tests  ({WIDTH}x{HEIGHT} headless)")
    bodies, positions, normals = build_scene()
    print(f"  scene: {len(bodies)} bodies, {positions.shape[0]} particles")

    cfg = base_config()
    window = Window(cfg)
    print(f"  GL: {window.ctx.info['GL_VERSION']} on "
          f"{window.ctx.info['GL_RENDERER']}")

    timing: dict[str, float] = {}
    failed = 0
    try:
        use_warp = True
        try:
            import warp as wp

            wp.init()
            use_warp = wp.get_device(cfg.device).is_cuda
        except Exception as exc:
            print(f"  warp unavailable ({exc}); using the numpy state path")
            use_warp = False
        state = fake_state(positions, normals, cfg.device, use_warp=use_warp)

        steps = [
            ("shaders", lambda: test_shaders_compile(window)),
            ("includes", lambda: test_include_cycle_detected(window)),
            ("camera", test_camera_math),
            ("geometry", test_geometry),
            ("text", lambda: test_text_atlas(window)),
            ("interop", lambda: test_interop(window)),
            ("frame", lambda: timing.__setitem__(
                "frame_ms", test_headless_frame(window, bodies, state))),
            ("toggles", lambda: test_toggles(window, bodies, state)),
            ("empty", lambda: test_no_hands_and_no_preview(window, bodies, state)),
            ("mismatch", lambda: test_material_mismatch_is_loud(
                window, bodies, state)),
            ("host path", lambda: test_host_fallback_path(
                window, bodies, positions, normals)),
            ("bare ctx", lambda: test_bare_context(window, bodies, state)),
            ("resize", lambda: test_resize(window, bodies, state)),
            ("windowed", lambda: test_windowed_context(
                bodies, positions, normals)),
            ("window resize", lambda: test_visible_window_resize(
                bodies, positions, normals)),
            ("post-swap shot", lambda: test_screenshot_after_swap(
                bodies, positions, normals)),
            ("cpu warp state", lambda: test_warp_state_on_another_device(
                window, bodies, positions, normals)),
            ("hardness", lambda: test_hardness_extremes(window, bodies, state)),
            ("hands", lambda: test_hand_extremes(window, bodies, state)),
            ("configs", lambda: test_config_extremes(window, bodies, state)),
            ("sizes", lambda: test_extreme_sizes(window, bodies, state)),
            ("two windows", lambda: test_two_windows(
                bodies, positions, normals)),
            ("state arrays", lambda: test_state_array_abuse(
                window, bodies, positions, normals)),
            ("previews", lambda: test_preview_shapes(window, bodies, state)),
            ("empty body", lambda: test_empty_body(window)),
            ("lifecycle", lambda: test_lifecycle(window, bodies, state)),
            ("rebind", lambda: test_rebind(window, bodies, positions, normals)),
            ("effects", lambda: test_effects_are_visible(
                window, bodies, state)),
            ("shader errors", lambda: test_shader_library_errors(window)),
            ("buffers", lambda: test_particle_buffers_contract(window)),
            ("contract", lambda: test_renderer_contract(
                window, bodies, positions, normals)),
            ("camera limits", test_camera_limits),
            ("input", test_input_translation),
            ("badges", lambda: test_hud_badges(window, bodies, state)),
            ("labels", lambda: test_unicode_labels(window, bodies, state)),
            ("hud layout", lambda: test_hud_layout_fits_the_window(window)),
            ("grain shadow", lambda: test_grain_contact_shadow(window)),
            ("timer owner", lambda: test_timer_owner_survives_another_release(
                window, bodies, state)),
            ("leak", lambda: test_no_leak(window, bodies, state)),
            ("optional", lambda: test_optional_real_modules(window)),
        ]
        for label, fn in steps:
            try:
                fn()
            except Exception:
                failed += 1
                print(f"  [FAIL] {label} raised:")
                traceback.print_exc()
    finally:
        window.close()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    failed += sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{passed}/{total} checks passed; {failed} failure(s)")
    # The shape tools/run_tests.py greps for when it summarises a file.
    print(f"--- {passed}/{total} passed")
    frame_ms = timing.get("frame_ms")
    if frame_ms:
        print(f"headless frame time: {frame_ms:.2f} ms "
              f"({1000.0 / frame_ms:.0f} fps) at {WIDTH}x{HEIGHT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
