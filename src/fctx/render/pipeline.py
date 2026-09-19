"""The frame graph: shadow, depth, SSAO, matter, bloom, composite, overlay.

The renderer reads ``SolverState`` and never writes to it.  It is also written
so that a *duck-typed* state works: anything exposing ``x`` and ``normal`` as
either warp arrays or numpy arrays will render, which is what lets the graphics
be developed and tested while the solver is still being written.

Pass order, and why it is that order:

1. **Shadow**, depth only, front faces culled.  Rendering back faces means the
   stored depth is the far side of the object, so the bias needed to stop
   self-shadowing no longer has to grow with the surface slope, and the
   contact shadow stays attached to the object.
2. **Depth prepass** at full resolution, non-multisampled.  SSAO needs a depth
   texture it can sample, and this is a forward renderer with no G-buffer.
3. **SSAO** at half resolution plus a blur, sampled by the main pass in screen
   space so the occlusion lands on the ambient term only.
4. **Main pass** into an f16 multisampled target: backdrop, floor, matter,
   hands, grains.
5. **Resolve** through a shader rather than a blit, so the multisample average
   can be luminance-weighted.
6. **Bloom**, six progressive levels down and back up.
7. **Composite**: exposure, ACES, vignette, chromatic aberration, dither.
8. **Overlay**: webcam inset, landmark skeleton, HUD and the hardness dial.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import moderngl
import numpy as np

from ..config import AppConfig
from ..core.material import Material
from ..core.types import HAND_BONES, BodyData, FrameStats, HandPose, MatterKind
from .buffers import ParticleBuffers
from .camera import OrbitCamera, mat_bytes
from .context import Window, window_for_context
from .geometry import capsule_shell, grid_plane
from .hud import Hud
from .shaders import ShaderLibrary
from .text import OverlayBatch, TextAtlas

__all__ = ["Renderer", "Lighting", "BLOOM_LEVELS"]

BLOOM_LEVELS = 6
_JOINT_RADIUS = 0.011
_HAND_COLOR = (0.38, 0.86, 1.00)
_HAND_GRAB_COLOR = (1.00, 0.72, 0.26)


@dataclass(frozen=True, slots=True)
class Lighting:
    """Scene lighting in linear radiance units, pre-exposure."""

    key_color: tuple[float, float, float] = (4.30, 4.07, 3.75)
    #: Travel direction of the rim light.  It sits behind, left and above the
    #: stage so its contribution lands on the silhouette and, through the
    #: translucency term, on whatever the key light is not reaching.
    rim_dir: tuple[float, float, float] = (0.62, -0.34, 0.62)
    rim_color: tuple[float, float, float] = (0.44, 0.82, 1.62)
    #: Kept low on purpose.  A generous hemisphere ambient hides every shading
    #: error, and it also flattens the body into a pastel silhouette -- the
    #: opposite of what a demo about how matter *feels* needs.
    sky_color: tuple[float, float, float] = (0.112, 0.142, 0.208)
    ground_color: tuple[float, float, float] = (0.030, 0.028, 0.036)
    #: How much brighter the ambient probe is for specular than for diffuse.
    #: It stands in for a studio softbox the scene does not actually have, and
    #: it is what keeps the metallic end of the dial from going black.
    spec_gain: float = 6.5
    floor_color: tuple[float, float, float] = (0.052, 0.058, 0.074)
    floor_grid: tuple[float, float, float] = (0.115, 0.150, 0.205)
    floor_size: float = 9.0
    floor_fade: float = 1.55
    floor_grid_spacing: float = 0.05


@dataclass
class _BodyDraw:
    body: BodyData
    #: Index into the caller's ``bodies`` / ``materials`` lists.  Bodies with
    #: no particles produce no draw, so position in ``draws`` is not position
    #: in ``bodies`` and the material cannot be paired off by zipping.
    index: int
    offset: int
    as_points: bool
    ibo: moderngl.Buffer
    count: int
    vao_main: moderngl.VertexArray
    vao_depth: moderngl.VertexArray
    vao_shadow: moderngl.VertexArray


@dataclass
class _Targets:
    """Every size-dependent GPU resource, rebuilt on resize."""

    size: tuple[int, int]
    objects: list[Any] = field(default_factory=list)

    def track(self, obj: Any) -> Any:
        self.objects.append(obj)
        return obj

    def release(self) -> None:
        for obj in reversed(self.objects):
            obj.release()
        self.objects.clear()


#: One GL_TIME_ELAPSED query may be open per context at a time; this records
#: which renderer currently owns it.  Keyed weakly so a dead context does not
#: keep a renderer alive.
_TIMER_OWNERS: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _timer_busy(ctx: Any) -> bool:
    return _TIMER_OWNERS.get(ctx) is not None


def _set_timer_owner(ctx: Any, owner: Any) -> None:
    if owner is None:
        _TIMER_OWNERS.pop(ctx, None)
    else:
        _TIMER_OWNERS[ctx] = owner


def _clear_timer_owner(ctx: Any, owner: Any) -> None:
    """Give up the slot only if ``owner`` is the one holding it.

    A renderer being released has no idea whether another renderer on the
    same context is mid-query.  Clearing the slot regardless tells the next
    renderer the context is free, it calls glBeginQuery on top of a query
    that is still running, and the GL_INVALID_OPERATION that produces is
    latched -- so it gets blamed on whatever GL call is checked next.
    """
    if _TIMER_OWNERS.get(ctx) is owner:
        _TIMER_OWNERS.pop(ctx, None)


class Renderer:
    """Draws one frame of FCTX MATTER STUDIO.

    ``ctx`` may be a :class:`~fctx.render.context.Window` or a
    :class:`moderngl.Context`.  A context that belongs to a Window is resolved
    back to it, because the two are the same request: without that, a caller
    passing ``window.ctx`` would get a renderer that composites into the
    default framebuffer while the window reads its offscreen one, and that
    shows up as a screenshot full of black rather than as an error.  The
    renderer then registers itself for teardown, so closing the window
    releases it in the right order.  With a context that no Window owns, the
    caller owns :meth:`release` and must call it before the context dies or
    the CUDA interop registration outlives its buffers.
    """

    def __init__(
        self,
        ctx: Window | moderngl.Context,
        state: Any,
        bodies: Sequence[BodyData],
        cfg: AppConfig,
    ) -> None:
        if isinstance(ctx, Window):
            self.window: Window | None = ctx
            self.ctx = ctx.ctx
        else:
            self.ctx = ctx
            self.window = window_for_context(ctx)
        # Creating a window makes its context current, so a second window
        # opened after the first silently redirects every GL call that
        # follows.  Binding here -- and on every entry point that touches GL
        # -- means this renderer's objects are created in, and destroyed
        # from, the context they belong to; getting it wrong does not fail,
        # it segfaults when the other window is destroyed.
        self._bind()
        self.state = state
        self.bodies = list(bodies)
        self.cfg = cfg
        self.rcfg = cfg.render
        self.lighting = Lighting()
        self._released = False

        if not self.bodies:
            raise ValueError("Renderer needs at least one body to draw")

        # Runtime toggles start from the config and are then owned by the
        # application: the keyboard flips these, not the frozen config.
        self.show_hud = bool(self.rcfg.show_hud)
        self.show_webcam = bool(self.rcfg.show_webcam)
        self.show_hands = bool(self.rcfg.show_hands)
        self.bloom_enabled = bool(self.rcfg.bloom)
        self.ssao_enabled = self.rcfg.ssao_samples > 0
        self.shadows_enabled = self.rcfg.shadow_size > 0
        self.wireframe = False

        self.samples = self._pick_samples(self.rcfg.msaa)
        # Without this the driver ignores gl_PointSize entirely and every grain
        # rasterises as a single pixel, which looks like the solver has lost
        # the granular body rather than like a render bug.
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.library = ShaderLibrary(self.ctx)
        self._build_programs()
        self._build_particle_buffers()
        self._build_bodies()
        self._build_static_geometry()
        self._build_hand_instances()
        self._build_shadow_target()
        self._build_overlay()

        self._targets: _Targets | None = None
        self._size = (0, 0)
        self._preview_tex: moderngl.Texture | None = None
        self._preview_shape: tuple[int, int] | None = None
        # What repaint() needs to reproduce the last frame without re-lighting.
        self._capture_target: moderngl.Framebuffer | None = None
        self._last_bloom: moderngl.Texture | None = None

        # A GL timer query, read one frame late.  With vsync on, the driver
        # blocks inside whichever GL call fills the queue, so a wall-clock
        # timer around draw() reports the wait as render cost: 14.9 ms for a
        # frame the GPU finishes in 1.1.  Reading the query at the start of
        # the *next* frame gets the real GPU time with no stall, because by
        # then the result is certainly ready.
        self.gpu_ms = 0.0
        try:
            self._timer = self.ctx.query(time=True)
        except Exception:  # noqa: BLE001  -- no GL_TIME_ELAPSED on this driver
            self._timer = None
        self._timer_pending = False
        self._timer_open = False
        self._last_preview_tex: moderngl.Texture | None = None
        self._last_overlay_count: int | None = None
        self._ensure_targets(self._current_size())

        if self.window is not None:
            self.window.add_teardown(self)
            self.window.set_frame_source(self)

    # -- construction -----------------------------------------------------

    def _bind(self) -> None:
        if self.window is not None:
            self.window.make_current()

    def _pick_samples(self, requested: int) -> int:
        limit = int(self.ctx.info.get("GL_MAX_SAMPLES", 1) or 1)
        if requested <= 1:
            return 1
        # Round down to a power of two the driver will actually give us; an
        # unsupported sample count makes framebuffer completion fail at the
        # first resize rather than here, which is much harder to trace.
        n = 1 << int(math.floor(math.log2(min(int(requested), limit))))
        return max(1, n)

    def _build_programs(self) -> None:
        lib = self.library
        self.prog_depth = lib.program("depth", vertex="depth_only.vert",
                                      fragment="depth_only.frag")
        self.prog_matter = lib.program("matter", vertex="matter.vert",
                                       fragment="matter.frag")
        self.prog_floor = lib.program("floor", vertex="floor.vert",
                                      fragment="floor.frag")
        self.prog_background = lib.program("background", vertex="fullscreen.vert",
                                           fragment="background.frag")
        self.prog_hand = lib.program("hand", vertex="hand.vert",
                                     fragment="hand.frag")
        self.prog_hand_depth = lib.program("hand_depth", vertex="hand.vert",
                                           fragment="depth_only.frag")
        self.prog_particle = lib.program("particle", vertex="particle.vert",
                                         fragment="particle.frag")
        self.prog_particle_depth = lib.program("particle_depth",
                                               vertex="particle.vert",
                                               fragment="particle_depth.frag")
        self.prog_ssao = lib.program("ssao", vertex="fullscreen.vert",
                                     fragment="ssao.frag")
        self.prog_ssao_blur = lib.program("ssao_blur", vertex="fullscreen.vert",
                                          fragment="ssao_blur.frag")
        self.prog_resolve = lib.program("resolve", vertex="fullscreen.vert",
                                        fragment="resolve.frag",
                                        defines={"SAMPLE_COUNT": self.samples})
        self.prog_bloom_down = lib.program("bloom_down", vertex="fullscreen.vert",
                                           fragment="bloom_down.frag")
        self.prog_bloom_up = lib.program("bloom_up", vertex="fullscreen.vert",
                                         fragment="bloom_up.frag")
        self.prog_composite = lib.program("composite", vertex="fullscreen.vert",
                                          fragment="composite.frag")
        # Fullscreen passes build their vertices from gl_VertexID, so their
        # vertex arrays carry no attributes at all -- one per program.
        self._fs_vaos: dict[int, moderngl.VertexArray] = {}

    def _fullscreen(self, program: moderngl.Program) -> moderngl.VertexArray:
        vao = self._fs_vaos.get(program.glo)
        if vao is None:
            vao = self.ctx.vertex_array(program, [])
            self._fs_vaos[program.glo] = vao
        return vao

    def _build_particle_buffers(self) -> None:
        counts = [b.num_particles for b in self.bodies]
        self.body_offsets: list[int] = []
        running = 0
        for c in counts:
            self.body_offsets.append(running)
            running += c
        # Prefer the solver's own layout when it publishes one; falling back to
        # a cumulative sum assumes bodies were concatenated in list order,
        # which is how SolverState assembles them.
        published = getattr(self.state, "body_offsets", None)
        if published is not None:
            offsets = [int(v) for v in published]
            if len(offsets) != len(self.bodies):
                raise ValueError(
                    f"state.body_offsets has {len(offsets)} entries but there "
                    f"are {len(self.bodies)} bodies")
            self.body_offsets = offsets
        self.num_particles = int(
            getattr(self.state, "num_particles", None) or running)
        if self.num_particles < running:
            raise ValueError(
                f"solver state holds {self.num_particles} particles, the "
                f"bodies need {running}")

        self.particles = ParticleBuffers(
            self.ctx, self.num_particles, self.cfg.device,
            prefer_interop=True)

    def _build_bodies(self) -> None:
        self.draws: list[_BodyDraw] = []
        for index, (body, offset) in enumerate(
                zip(self.bodies, self.body_offsets)):
            # An index buffer of length zero is rejected by the GL binding, so
            # a body with nothing in it is dropped rather than crashing the
            # whole scene; BodyData.validate() considers an empty body legal.
            if body.num_particles == 0:
                continue
            as_points = body.kind is MatterKind.GRAIN or body.num_tris == 0
            if as_points:
                idx = np.arange(offset, offset + body.num_particles,
                                dtype=np.int32)
                count = body.num_particles
                main_prog = self.prog_particle
                depth_prog = self.prog_particle_depth
            else:
                idx = (body.tri_idx.astype(np.int32) + offset).reshape(-1)
                count = idx.size
                main_prog = self.prog_matter
                depth_prog = self.prog_depth
            ibo = self.ctx.buffer(np.ascontiguousarray(idx).tobytes())

            pos, nrm = self.particles.positions, self.particles.normals
            if as_points:
                content: list[Any] = [(pos, "3f", "in_pos")]
            else:
                content = [(pos, "3f", "in_pos"), (nrm, "3f", "in_nrm")]

            self.draws.append(_BodyDraw(
                body=body,
                index=index,
                offset=offset,
                as_points=as_points,
                ibo=ibo,
                count=count,
                vao_main=self.ctx.vertex_array(
                    main_prog, content, index_buffer=ibo, index_element_size=4),
                vao_depth=self.ctx.vertex_array(
                    depth_prog, [(pos, "3f", "in_pos")], index_buffer=ibo,
                    index_element_size=4),
                vao_shadow=self.ctx.vertex_array(
                    depth_prog, [(pos, "3f", "in_pos")], index_buffer=ibo,
                    index_element_size=4),
            ))

    def _build_static_geometry(self) -> None:
        plane = grid_plane(self.lighting.floor_size, subdivisions=1,
                           height=self.cfg.solver.ground_y)
        self._floor_vbo = self.ctx.buffer(plane.positions.tobytes())
        self._floor_ibo = self.ctx.buffer(plane.indices.tobytes())
        self._floor_vao = self.ctx.vertex_array(
            self.prog_floor, [(self._floor_vbo, "3f", "in_pos")],
            index_buffer=self._floor_ibo, index_element_size=4)
        self._floor_depth_vao = self.ctx.vertex_array(
            self.prog_depth, [(self._floor_vbo, "3f", "in_pos")],
            index_buffer=self._floor_ibo, index_element_size=4)

    def _build_hand_instances(self) -> None:
        shell = capsule_shell(rings=12, segments=20)
        self._capsule_dir = self.ctx.buffer(shell.directions.tobytes())
        self._capsule_side = self.ctx.buffer(shell.side.tobytes())
        self._capsule_ibo = self.ctx.buffer(shell.indices.tobytes())
        self._capsule_count = shell.num_triangles * 3

        # 21 bones plus one sphere per joint, for every hand slot the tracker
        # can report.  Capacity is fixed so the buffer is never reallocated
        # mid-session.
        self._hand_capacity = max(1, self.cfg.tracking.max_hands) * (
            len(HAND_BONES) + 21)
        self._hand_vbo = self.ctx.buffer(
            reserve=self._hand_capacity * 11 * 4, dynamic=True)
        self._hand_count = 0

        self._hand_vao = self._capsule_vao(self.prog_hand)
        self._hand_shadow_vao = self._capsule_vao(self.prog_hand_depth)

    def _capsule_vao(self, program: moderngl.Program) -> moderngl.VertexArray:
        """Bind the capsule buffers to ``program``, padding inactive attributes.

        The depth-only variant never reads the instance colour, so the linker
        drops it and moderngl cannot find a location for it.  Replacing the
        dropped attributes with padding of the same width keeps one interleaved
        instance buffer serving both programs.
        """
        pad = {"3f": "3x4", "1f": "1x4", "4f": "4x4"}
        instance_fields = [("3f", "i_a"), ("3f", "i_b"), ("1f", "i_r"),
                           ("4f", "i_color")]
        parts: list[str] = []
        names: list[str] = []
        for fmt, name in instance_fields:
            if program.get(name, None) is None:
                parts.append(pad[fmt])
            else:
                parts.append(fmt)
                names.append(name)
        content: list[Any] = []
        for buf, fmt, name in ((self._capsule_dir, "3f", "in_dir"),
                               (self._capsule_side, "1f", "in_side")):
            if program.get(name, None) is not None:
                content.append((buf, fmt, name))
        content.append((self._hand_vbo, " ".join(parts) + "/i", *names))
        return self.ctx.vertex_array(
            program, content, index_buffer=self._capsule_ibo,
            index_element_size=4)

    def _build_shadow_target(self) -> None:
        size = int(self.rcfg.shadow_size)
        self._shadow_size = size
        if size <= 0:
            # A one-texel dummy keeps the sampler bound and the shader branch
            # free of an undefined texture unit when shadows are off.
            self._shadow_tex = self.ctx.depth_texture((1, 1))
            self._shadow_fbo = self.ctx.framebuffer(
                depth_attachment=self._shadow_tex)
            self._shadow_fbo.clear(depth=1.0)
        else:
            self._shadow_tex = self.ctx.depth_texture((size, size))
            self._shadow_fbo = self.ctx.framebuffer(
                depth_attachment=self._shadow_tex)
        self._shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._shadow_tex.repeat_x = False
        self._shadow_tex.repeat_y = False
        self._shadow_tex.compare_func = ""

    def _build_overlay(self) -> None:
        self.atlas = TextAtlas(self.ctx, px=44)
        self.overlay = OverlayBatch(self.ctx, self.library, self.atlas)
        self.hud = Hud()

    # -- size-dependent resources -----------------------------------------

    def _current_size(self) -> tuple[int, int]:
        if self.window is not None:
            return self.window.size
        return (int(self.rcfg.width), int(self.rcfg.height))

    def _ensure_targets(self, size: tuple[int, int]) -> None:
        size = (max(1, int(size[0])), max(1, int(size[1])))
        if self._targets is not None and self._size == size:
            return
        if self._targets is not None:
            self._targets.release()
        t = _Targets(size)
        w, h = size
        ctx = self.ctx
        # GL has no such thing as a one-sample multisample texture: asking for
        # samples=1 still produces a TEXTURE_2D_MULTISAMPLE, which then cannot
        # be bound to the plain sampler2D the single-sample resolve declares.
        s = self.samples if self.samples > 1 else 0

        self.scene_color = t.track(ctx.texture(size, 4, dtype="f2", samples=s))
        self.scene_depth = t.track(ctx.depth_texture(size, samples=s))
        self.scene_fbo = t.track(ctx.framebuffer([self.scene_color],
                                                 self.scene_depth))
        if s == 0:
            self.scene_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.scene_color.repeat_x = False
            self.scene_color.repeat_y = False

        self.resolved = t.track(ctx.texture(size, 4, dtype="f2"))
        self.resolved.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.resolved.repeat_x = False
        self.resolved.repeat_y = False
        self.resolve_fbo = t.track(ctx.framebuffer([self.resolved]))

        self.prepass_depth = t.track(ctx.depth_texture(size))
        self.prepass_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.prepass_depth.repeat_x = False
        self.prepass_depth.repeat_y = False
        self.prepass_depth.compare_func = ""
        self.prepass_fbo = t.track(ctx.framebuffer(
            depth_attachment=self.prepass_depth))

        ao_size = (max(1, w // 2), max(1, h // 2))
        self.ao_size = ao_size
        self.ao_tex = t.track(ctx.texture(ao_size, 1, dtype="f1"))
        self.ao_blur = t.track(ctx.texture(ao_size, 1, dtype="f1"))
        for tex in (self.ao_tex, self.ao_blur):
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
        self.ao_fbo = t.track(ctx.framebuffer([self.ao_tex]))
        self.ao_blur_fbo = t.track(ctx.framebuffer([self.ao_blur]))
        self.ao_blur_fbo.clear(1.0, 1.0, 1.0, 1.0)

        self.bloom_tex: list[moderngl.Texture] = []
        self.bloom_fbo: list[moderngl.Framebuffer] = []
        for i in range(BLOOM_LEVELS):
            bw = max(1, w >> (i + 1))
            bh = max(1, h >> (i + 1))
            tex = t.track(ctx.texture((bw, bh), 4, dtype="f2"))
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            self.bloom_tex.append(tex)
            self.bloom_fbo.append(t.track(ctx.framebuffer([tex])))

        self._black = t.track(ctx.texture((1, 1), 4, b"\x00" * 8, dtype="f2"))
        self._white = t.track(ctx.texture((1, 1), 1, b"\xff"))

        self._targets = t
        self._size = size

    # -- public -----------------------------------------------------------

    def set_wireframe(self, on: bool) -> None:
        self.wireframe = bool(on)

    def draw(
        self,
        camera: OrbitCamera,
        poses: list[HandPose],
        materials: list[Material],
        preview: np.ndarray | None,
        stats: FrameStats,
        hud_lines: list[str],
        *,
        hardness: float | None = None,
        notifications: Sequence[Any] = (),
        show_hud: bool | None = None,
        paused: bool = False,
    ) -> None:
        """Render one frame.

        ``hardness`` overrides where the dial's knob sits.  The materials have
        a hardness of their own, but with several bodies on stage they can
        differ, and it is the control value -- not any one body's -- that the
        gauge is reporting.  ``notifications`` are transient messages; anything
        with ``text`` and ``alpha`` attributes will do, so this module does not
        have to depend on the interaction layer to draw them.
        """
        if self._released:
            raise RuntimeError("Renderer.draw after release()")
        self._bind()
        if len(materials) != len(self.bodies):
            raise ValueError(
                f"draw() got {len(materials)} materials for "
                f"{len(self.bodies)} bodies")

        size = self._current_size()
        self._ensure_targets(size)
        camera.aspect = size[0] / max(1, size[1])

        # OpenGL allows exactly one GL_TIME_ELAPSED query in flight per
        # context, so a second renderer sharing the context must not open one
        # -- glBeginQuery would raise GL_INVALID_OPERATION and poison the
        # error state for whatever ran next.
        if self._timer is not None and not _timer_busy(self.ctx):
            if self._timer_pending:
                self.gpu_ms = self._timer.elapsed / 1.0e6
            self._timer.__enter__()
            self._timer_open = True
            _set_timer_owner(self.ctx, self)

        try:
            self._draw_scene(camera, poses, materials, preview, stats,
                             hud_lines, hardness, notifications, show_hud,
                             paused, size)
        finally:
            if self._timer_open:
                self._timer.__exit__(None, None, None)
                self._timer_open = False
                self._timer_pending = True
                _set_timer_owner(self.ctx, None)

    def _draw_scene(self, camera, poses, materials, preview, stats, hud_lines,
                    hardness, notifications, show_hud, paused, size) -> None:
        self._upload_particles()
        self._update_hands(poses)

        use_shadow = self.shadows_enabled and self._shadow_size > 0
        use_ao = self.ssao_enabled and self.rcfg.ssao_samples > 0

        if use_shadow:
            self._shadow_pass(camera)
        if use_ao:
            self._depth_prepass(camera)
            self._ssao_pass(camera)

        self._main_pass(camera, materials, use_shadow, use_ao)
        self._resolve_pass()
        self._last_bloom = self._bloom_pass() if self.bloom_enabled else None
        self._composite_pass(self._last_bloom)
        self._overlay_pass(poses, materials, preview, stats, hud_lines,
                           hardness, notifications,
                           self.show_hud if show_hud is None else bool(show_hud),
                           paused)

    def repaint(self, target: moderngl.Framebuffer) -> None:
        """Redraw the frame :meth:`draw` last produced, into ``target``.

        Only the composite and the overlay are re-run; the lit scene is still
        sitting in the resolve and bloom targets, so this costs two fullscreen
        passes rather than a whole frame.  It exists so a screenshot taken
        after the buffers have been swapped is still the frame the user saw.
        """
        if self._released:
            raise RuntimeError("Renderer.repaint after release()")
        self._bind()
        if self._targets is None or self._last_overlay_count is None:
            raise RuntimeError("Renderer.repaint before the first draw()")
        self._capture_target = target
        try:
            self._composite_pass(self._last_bloom)
            self._use_output()
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA,
                                   moderngl.ONE_MINUS_SRC_ALPHA)
            self.overlay.draw(target.size, self._last_preview_tex)
            self.ctx.disable(moderngl.BLEND)
        finally:
            self._capture_target = None

    # -- passes -----------------------------------------------------------

    def _upload_particles(self) -> None:
        # The drawn position, not the physics one: a soft body's skin is
        # bound to its lattice barycentrically and lives in x_skin.  A state
        # without one (the test fakes, a scene of pure cloth) aliases x.
        x = getattr(self.state, "x_skin", None)
        if x is None:
            x = getattr(self.state, "x", None)
        normal = getattr(self.state, "normal", None)
        if x is None or normal is None:
            raise AttributeError(
                "solver state must expose 'x' and 'normal' particle arrays")
        self.particles.update_from(x, normal)

    def _update_hands(self, poses: list[HandPose]) -> None:
        if not self.show_hands or not poses:
            self._hand_count = 0
            return
        rows: list[np.ndarray] = []
        highlight = self.rcfg.grab_highlight
        for pose in poses:
            a, b, radius = pose.bone_segments()
            grabbing = bool(pose.pinching) and highlight
            base = _HAND_GRAB_COLOR if grabbing else _HAND_COLOR
            alpha = float(np.clip(pose.confidence, 0.25, 1.0))
            colour = np.array([*base, alpha], dtype=np.float32)

            bones = np.zeros((a.shape[0], 11), dtype=np.float32)
            bones[:, 0:3] = a
            bones[:, 3:6] = b
            bones[:, 6] = radius
            bones[:, 7:11] = colour
            rows.append(bones)

            joints = np.zeros((pose.joints.shape[0], 11), dtype=np.float32)
            joints[:, 0:3] = pose.joints
            joints[:, 3:6] = pose.joints
            joints[:, 6] = _JOINT_RADIUS
            joints[:, 7:11] = colour
            # Fingertips read as the business end of the hand, so they get a
            # brighter core than the knuckles.
            joints[[4, 8, 12, 16, 20], 6] = _JOINT_RADIUS * 1.45
            rows.append(joints)

        data = np.concatenate(rows, axis=0)[: self._hand_capacity]
        self._hand_count = int(data.shape[0])
        self._hand_vbo.write(np.ascontiguousarray(data).tobytes())

    def _draw_matter_depth(self, program: moderngl.Program, view: np.ndarray,
                           proj: np.ndarray, point_scale: float,
                           ortho: bool, attr: str, *, cull: bool = False) -> None:
        for draw in self.draws:
            vao = getattr(draw, attr)
            if cull:
                # A sheet of cloth has no interior, so front-face culling
                # would drop exactly the faces the light can see and the
                # cloth would cast no shadow at all.
                if draw.body.double_sided or draw.as_points:
                    self.ctx.disable(moderngl.CULL_FACE)
                else:
                    self.ctx.enable(moderngl.CULL_FACE)
                    self.ctx.cull_face = "front"
            if draw.as_points:
                self.prog_particle_depth["u_view"].write(mat_bytes(view))
                self.prog_particle_depth["u_proj"].write(mat_bytes(proj))
                self.prog_particle_depth["u_radius"].value = \
                    float(draw.body.particle_radius)
                self.prog_particle_depth["u_point_scale"].value = point_scale
                self.prog_particle_depth["u_ortho"].value = 1 if ortho else 0
                vao.render(moderngl.POINTS, vertices=draw.count)
            else:
                program["u_view_proj"].write(mat_bytes(proj @ view))
                vao.render(moderngl.TRIANGLES, vertices=draw.count)

    def _shadow_pass(self, camera: OrbitCamera) -> None:
        light_view = camera.light_view()
        light_proj = camera.light_projection()
        self._shadow_fbo.use()
        self._shadow_fbo.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Closed bodies are rendered back-face only: storing the far surface
        # instead of the near one removes acne without a slope bias big enough
        # to detach the contact shadow.
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "front"

        # Point sprites under an orthographic projection keep a constant pixel
        # size: one world metre spans shadow_size / (2 * extent) texels.
        point_scale = self._shadow_size / (2.0 * camera.shadow_extent)
        self._draw_matter_depth(self.prog_depth, light_view, light_proj,
                                point_scale, True, "vao_shadow", cull=True)

        if self._hand_count:
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.cull_face = "front"
            self.prog_hand_depth["u_view_proj"].write(
                mat_bytes(light_proj @ light_view))
            self._hand_shadow_vao.render(moderngl.TRIANGLES,
                                         vertices=self._capsule_count,
                                         instances=self._hand_count)
        self.ctx.cull_face = "back"
        self.ctx.disable(moderngl.CULL_FACE)

    def _depth_prepass(self, camera: OrbitCamera) -> None:
        view, proj = camera.view(), camera.projection()
        view_proj = proj @ view
        self.prepass_fbo.use()
        self.prepass_fbo.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._draw_matter_depth(self.prog_depth, view, proj,
                                self._point_scale(camera), False, "vao_depth")
        # The floor and the hands belong in here too.  Occlusion where a body
        # meets the ground is the single most useful thing SSAO contributes,
        # and it cannot see a contact whose other half is not in the depth
        # buffer.
        self.prog_depth["u_view_proj"].write(mat_bytes(view_proj))
        self._floor_depth_vao.render(moderngl.TRIANGLES)
        if self._hand_count:
            self.prog_hand_depth["u_view_proj"].write(mat_bytes(view_proj))
            self._hand_shadow_vao.render(moderngl.TRIANGLES,
                                         vertices=self._capsule_count,
                                         instances=self._hand_count)

    def _point_scale(self, camera: OrbitCamera) -> float:
        # Pixels per unit of world size at one metre: the standard point-sprite
        # scale factor, so gl_PointSize = 2 r * scale / distance.
        half_fov = math.radians(camera.fov_y) * 0.5
        return self._size[1] / (2.0 * math.tan(half_fov))

    def _ssao_pass(self, camera: OrbitCamera) -> None:
        proj = camera.projection()
        inv_proj = np.linalg.inv(proj)
        self.ao_fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.prepass_depth.use(0)
        p = self.prog_ssao
        p["u_depth"].value = 0
        p["u_proj"].write(mat_bytes(proj))
        p["u_inv_proj"].write(mat_bytes(inv_proj))
        p["u_noise_scale"].value = (1.0, 1.0)
        p["u_radius"].value = float(self.rcfg.ssao_radius)
        p["u_strength"].value = float(self.rcfg.ssao_strength)
        p["u_bias"].value = 0.012
        p["u_samples"].value = int(self.rcfg.ssao_samples)
        self._fullscreen(p).render(moderngl.TRIANGLES, vertices=3)

        self.ao_blur_fbo.use()
        self.ao_tex.use(0)
        b = self.prog_ssao_blur
        b["u_ao"].value = 0
        b["u_texel"].value = (1.0 / self.ao_size[0], 1.0 / self.ao_size[1])
        self._fullscreen(b).render(moderngl.TRIANGLES, vertices=3)

    def _main_pass(self, camera: OrbitCamera, materials: list[Material],
                   use_shadow: bool, use_ao: bool) -> None:
        ctx = self.ctx
        view, proj = camera.view(), camera.projection()
        view_proj = proj @ view
        light_vp = camera.light_view_proj()
        cam_pos = tuple(float(v) for v in camera.position)
        light = self.lighting

        self.scene_fbo.use()
        self.scene_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        bg = self.prog_background
        bg["u_top"].value = tuple(float(v) for v in self.rcfg.background_top)
        bg["u_bottom"].value = tuple(float(v) for v in self.rcfg.background_bottom)
        bg["u_glow"].value = (0.055, 0.085, 0.130)
        bg["u_glow_center"].value = (0.5, 0.60)
        bg["u_glow_radius"].value = 0.46
        self._fullscreen(bg).render(moderngl.TRIANGLES, vertices=3)

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        shadow_unit, ao_unit = 0, 1
        self._shadow_tex.use(shadow_unit)
        (self.ao_blur if use_ao else self._white).use(ao_unit)

        def shadow_uniforms(prog: moderngl.Program) -> None:
            prog["u_shadow_map"].value = shadow_unit
            prog["u_light_vp"].write(mat_bytes(light_vp))
            prog["u_shadow_texel"].value = camera.shadow_world_texel(
                self._shadow_size)
            prog["u_shadow_texel_uv"].value = 1.0 / max(self._shadow_size, 1)
            prog["u_shadow_depth_range"].value = camera.shadow_depth_range
            prog["u_shadow_softness"].value = float(self.rcfg.shadow_softness)
            prog["u_use_shadow"].value = 1 if use_shadow else 0

        f = self.prog_floor
        f["u_view_proj"].write(mat_bytes(view_proj))
        f["u_center"].value = tuple(float(v) for v in camera.stage_center)
        f["u_fade_radius"].value = light.floor_fade
        f["u_grid_spacing"].value = light.floor_grid_spacing
        f["u_base_color"].value = light.floor_color
        f["u_grid_color"].value = light.floor_grid
        f["u_key_color"].value = light.key_color
        f["u_key_dir"].value = tuple(float(v) for v in camera.light_dir)
        f["u_sky_color"].value = light.sky_color
        shadow_uniforms(f)
        f["u_ao_tex"].value = ao_unit
        f["u_resolution"].value = (float(self._size[0]), float(self._size[1]))
        f["u_use_ao"].value = 1 if use_ao else 0
        self._floor_vao.render(moderngl.TRIANGLES)

        ctx.disable(moderngl.BLEND)
        if self.wireframe:
            ctx.wireframe = True

        point_scale = self._point_scale(camera)
        inv_view = np.linalg.inv(view)
        for draw in self.draws:
            material = materials[draw.index]
            if draw.as_points:
                p = self.prog_particle
                p["u_view"].write(mat_bytes(view))
                p["u_proj"].write(mat_bytes(proj))
                p["u_inv_view"].write(mat_bytes(inv_view))
                p["u_radius"].value = float(draw.body.particle_radius)
                p["u_point_scale"].value = point_scale
                p["u_ortho"].value = 0
            else:
                p = self.prog_matter
                p["u_view_proj"].write(mat_bytes(view_proj))
                p["u_double_sided"].value = 1 if draw.body.double_sided else 0
                p["u_emissive"].value = 0.0
                p["u_ao_tex"].value = ao_unit
                p["u_resolution"].value = (float(self._size[0]),
                                           float(self._size[1]))
                p["u_use_ao"].value = 1 if use_ao else 0

            p["u_cam_pos"].value = cam_pos
            p["u_base_color"].value = tuple(float(v) for v in material.color)
            p["u_roughness"].value = float(material.roughness)
            p["u_metallic"].value = float(material.metallic)
            p["u_translucency"].value = float(material.translucency)
            p["u_key_dir"].value = tuple(float(v) for v in camera.light_dir)
            p["u_key_color"].value = light.key_color
            p["u_rim_dir"].value = light.rim_dir
            p["u_rim_color"].value = light.rim_color
            p["u_sky_color"].value = light.sky_color
            p["u_ground_color"].value = light.ground_color
            p["u_spec_gain"].value = light.spec_gain
            shadow_uniforms(p)

            if draw.as_points:
                draw.vao_main.render(moderngl.POINTS, vertices=draw.count)
            else:
                draw.vao_main.render(moderngl.TRIANGLES, vertices=draw.count)

        if self.wireframe:
            ctx.wireframe = False

        if self._hand_count:
            h = self.prog_hand
            h["u_view_proj"].write(mat_bytes(view_proj))
            h["u_cam_pos"].value = cam_pos
            h["u_key_color"].value = light.key_color
            h["u_key_dir"].value = tuple(float(v) for v in camera.light_dir)
            h["u_emissive"].value = 0.85
            self._hand_vao.render(moderngl.TRIANGLES,
                                  vertices=self._capsule_count,
                                  instances=self._hand_count)

    def _resolve_pass(self) -> None:
        self.resolve_fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.scene_color.use(0)
        p = self.prog_resolve
        p["u_src"].value = 0
        if self.samples > 1:
            p["u_resolution"].value = (float(self._size[0]),
                                       float(self._size[1]))
        self._fullscreen(p).render(moderngl.TRIANGLES, vertices=3)

    def _bloom_pass(self) -> moderngl.Texture:
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        down = self.prog_bloom_down
        down["u_src"].value = 0
        down["u_threshold"].value = float(self.rcfg.bloom_threshold)
        down["u_knee"].value = max(0.08, float(self.rcfg.bloom_threshold) * 0.5)

        src_tex: moderngl.Texture = self.resolved
        src_size = self._size
        for i in range(BLOOM_LEVELS):
            self.bloom_fbo[i].use()
            src_tex.use(0)
            down["u_texel"].value = (1.0 / src_size[0], 1.0 / src_size[1])
            down["u_prefilter"].value = 1 if i == 0 else 0
            self._fullscreen(down).render(moderngl.TRIANGLES, vertices=3)
            src_tex = self.bloom_tex[i]
            src_size = src_tex.size

        up = self.prog_bloom_up
        up["u_src"].value = 0
        up["u_scatter"].value = 1.0
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(BLOOM_LEVELS - 1, 0, -1):
            self.bloom_fbo[i - 1].use()
            self.bloom_tex[i].use(0)
            size = self.bloom_tex[i].size
            up["u_texel"].value = (1.0 / size[0], 1.0 / size[1])
            self._fullscreen(up).render(moderngl.TRIANGLES, vertices=3)
        ctx.disable(moderngl.BLEND)
        return self.bloom_tex[0]

    def _output(self) -> moderngl.Framebuffer:
        if self._capture_target is not None:
            return self._capture_target
        if self.window is not None:
            return self.window.framebuffer
        return self.ctx.screen

    def _use_output(self) -> moderngl.Framebuffer:
        out = self._output()
        out.use()
        # Binding a framebuffer restores the viewport moderngl recorded for
        # it, and for ctx.screen that was measured once at context creation.
        # Setting it explicitly is what keeps the composite covering the whole
        # window after a resize.  A capture target may have been sized after
        # the frame was composed, so it gets its own extent rather than the
        # frame's.
        size = out.size if self._capture_target is not None else self._size
        out.viewport = (0, 0, size[0], size[1])
        if self.window is not None and self._capture_target is None:
            self.window.mark_frame_drawn()
        return out

    def _composite_pass(self, bloom: moderngl.Texture | None) -> None:
        self._use_output()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.resolved.use(0)
        (bloom or self._black).use(1)
        c = self.prog_composite
        c["u_scene"].value = 0
        c["u_bloom"].value = 1
        c["u_exposure"].value = float(self.rcfg.exposure)
        c["u_bloom_strength"].value = (
            float(self.rcfg.bloom_strength) if bloom is not None else 0.0)
        c["u_vignette"].value = float(self.rcfg.vignette)
        c["u_aberration"].value = float(self.rcfg.chromatic_aberration)
        c["u_aces"].value = 1 if self.rcfg.aces else 0
        self._fullscreen(c).render(moderngl.TRIANGLES, vertices=3)

    # -- overlay ----------------------------------------------------------

    def _preview_texture(self, preview: np.ndarray | None) -> moderngl.Texture | None:
        if preview is None:
            return None
        if preview.ndim != 3 or preview.shape[2] not in (3, 4):
            raise ValueError(
                f"preview must be (H, W, 3) or (H, W, 4) uint8, got {preview.shape}")
        h, w = int(preview.shape[0]), int(preview.shape[1])
        comps = int(preview.shape[2])
        if self._preview_shape != (w, h) or self._preview_tex is None:
            if self._preview_tex is not None:
                self._preview_tex.release()
            self._preview_tex = self.ctx.texture((w, h), comps, dtype="f1")
            self._preview_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._preview_tex.repeat_x = False
            self._preview_tex.repeat_y = False
            self._preview_shape = (w, h)
        self._preview_tex.write(
            np.ascontiguousarray(preview, dtype=np.uint8).tobytes())
        return self._preview_tex

    def _image_uv(self, world: np.ndarray) -> np.ndarray:
        """Project world-space joints back into normalised camera-image space.

        This inverts the image-to-world mapping documented for
        :class:`TrackingConfig`.  It is only ever used to draw the skeleton on
        the webcam inset, where being a few pixels off is invisible, and it
        keeps the renderer from needing the tracker's internals.
        """
        t = self.cfg.tracking
        u = 0.5 + world[:, 0] / (2.0 * max(t.stage_half_width, 1e-6))
        v = 0.5 - (world[:, 1] - t.stage_center_y) / (
            2.0 * max(t.stage_half_height, 1e-6))
        return np.clip(np.stack([u, v], axis=1), 0.0, 1.0)

    def _overlay_pass(self, poses: list[HandPose], materials: list[Material],
                      preview: np.ndarray | None, stats: FrameStats,
                      hud_lines: list[str], hardness: float | None,
                      notifications: Sequence[Any], show_hud: bool,
                      paused: bool) -> None:
        batch = self.overlay
        batch.clear()
        width, height = self._size

        tex = self._preview_texture(preview) if self.show_webcam else None
        if tex is not None:
            scale = float(np.clip(self.rcfg.webcam_scale, 0.06, 0.6))
            iw = width * scale
            ih = iw * tex.size[1] / max(1, tex.size[0])
            margin = 22.0 * max(height, 540) / 900.0
            ix, iy = width - iw - margin, margin
            batch.rect(ix - 2.0, iy - 2.0, iw + 4.0, ih + 4.0,
                       (0.32, 0.72, 0.95, 0.35), radius=10.0)
            batch.image(ix, iy, iw, ih, radius=8.0)
            self._draw_skeleton(batch, poses, ix, iy, iw, ih)

        if show_hud:
            self.hud.build(batch, (width, height), materials, stats, hud_lines,
                           hardness=hardness)
        self.hud.badges(batch, (width, height), notifications, paused)

        self._last_preview_tex = tex
        self._last_overlay_count = len(batch)

        self._use_output()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        batch.draw((width, height), tex)
        self.ctx.disable(moderngl.BLEND)

    def _draw_skeleton(self, batch: OverlayBatch, poses: list[HandPose],
                       ix: float, iy: float, iw: float, ih: float) -> None:
        line_w = max(1.5, iw * 0.006)
        for pose in poses:
            uv = self._image_uv(np.asarray(pose.joints, dtype=np.float32))
            px = ix + uv[:, 0] * iw
            py = iy + uv[:, 1] * ih
            grabbing = bool(pose.pinching)
            colour = ((1.0, 0.72, 0.26, 0.95) if grabbing
                      else (0.38, 0.86, 1.0, 0.85))
            for a, b in HAND_BONES:
                batch.line(px[a], py[a], px[b], py[b], line_w, colour)
            for i in range(px.shape[0]):
                batch.circle(px[i], py[i], line_w * 1.1, colour)
            pinch = self._image_uv(
                np.asarray(pose.pinch_point, dtype=np.float32).reshape(1, 3))[0]
            batch.circle(ix + pinch[0] * iw, iy + pinch[1] * ih,
                         line_w * 2.6 * (1.0 + pose.pinch),
                         (1.0, 0.72, 0.26, 0.45 + 0.5 * pose.pinch))

    # -- teardown ---------------------------------------------------------

    def release_scene(self) -> None:
        """Free everything tied to the current bodies and solver state.

        Switching preset rebuilds the whole simulation, which means new
        particle buffers.  Those carry the CUDA-GL registration, so they have
        to be unregistered before the solver that wrote through them goes
        away -- not left to the garbage collector, which would get to them
        after the replacement has already registered a buffer of its own.
        """
        for draw in self.draws:
            draw.vao_main.release()
            draw.vao_depth.release()
            draw.vao_shadow.release()
            draw.ibo.release()
        self.draws.clear()
        self.particles.release()
        self._last_bloom = None
        self._last_overlay_count = None

    def rebind(self, state: Any, bodies: Sequence[BodyData]) -> None:
        """Point the renderer at a new scene, keeping programs and targets."""
        if self._released:
            raise RuntimeError("Renderer.rebind after release()")
        self._bind()
        if not bodies:
            raise ValueError("Renderer.rebind needs at least one body")
        self.release_scene()
        self.state = state
        self.bodies = list(bodies)
        self._build_particle_buffers()
        self._build_bodies()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._bind()
        if self._timer is not None:
            # Leaving a query open would block the next renderer on this
            # context from ever opening one, and releasing the object while
            # the query is still running is undefined.
            if self._timer_open:
                self._timer.__exit__(None, None, None)
                self._timer_open = False
            _clear_timer_owner(self.ctx, self)
            self._timer = None
        if self.window is not None:
            if self.window.frame_source is self:
                self.window.set_frame_source(None)
            self.window.remove_teardown(self)
        # Interop first: the CUDA registration must go while the GL context is
        # still alive and still owns the buffers it refers to.
        self.release_scene()
        if self._preview_tex is not None:
            self._preview_tex.release()
            self._preview_tex = None
        if self._targets is not None:
            self._targets.release()
            self._targets = None
        self._hand_vao.release()
        self._hand_shadow_vao.release()
        self._hand_vbo.release()
        self._capsule_dir.release()
        self._capsule_side.release()
        self._capsule_ibo.release()
        self._floor_vao.release()
        self._floor_depth_vao.release()
        self._floor_vbo.release()
        self._floor_ibo.release()
        self._shadow_fbo.release()
        self._shadow_tex.release()
        self.overlay.release()
        self.atlas.release()
        for vao in self._fs_vaos.values():
            vao.release()
        self._fs_vaos.clear()
        self.library.release()
