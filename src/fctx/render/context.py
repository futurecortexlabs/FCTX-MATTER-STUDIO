"""Window, OpenGL context and input plumbing.

Two modes share one class.  A visible window swaps a real framebuffer; a
headless window is created hidden and renders into an offscreen framebuffer
instead.  Hidden-but-real is deliberate: a fully windowless EGL/OSMesa context
would not exercise the same driver path as the shipped application, and the
smoke test is only worth running if it renders through the code people see.

No application logic lives here.  Input arrives as a queue of
:class:`InputEvent` values that ``app.py`` drains once per frame; this module
never decides what a key means.  It does name keys, though: ``poll_events``
hands back :class:`fctx.ui.controls.InputEvent` values, because ``fctx.ui`` is
deliberately free of glfw and the translation has to live on whichever side of
that boundary owns the window.
"""

from __future__ import annotations

import ctypes
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import glfw
import moderngl
import numpy as np

from ..config import AppConfig, RenderConfig
from ..ui.controls import EventKind
from ..ui.controls import InputEvent as ControlEvent

__all__ = ["InputEvent", "Window", "GLError", "FrameSource", "check_gl_error"]


class GLError(RuntimeError):
    """Raised when ``glGetError`` reports a problem."""


_GL_ERRORS = {
    0x0500: "GL_INVALID_ENUM",
    0x0501: "GL_INVALID_VALUE",
    0x0502: "GL_INVALID_OPERATION",
    0x0503: "GL_STACK_OVERFLOW",
    0x0504: "GL_STACK_UNDERFLOW",
    0x0505: "GL_OUT_OF_MEMORY",
    0x0506: "GL_INVALID_FRAMEBUFFER_OPERATION",
}


def check_gl_error(ctx: moderngl.Context, where: str) -> None:
    """Raise :class:`GLError` if the driver has an error pending."""
    code = ctx.error
    if code and code != "GL_NO_ERROR":
        raise GLError(f"{where}: {code}")


@dataclass(frozen=True, slots=True)
class InputEvent:
    """One user input, in the order the OS delivered it.

    ``kind`` is one of ``key``, ``char``, ``mouse``, ``scroll``, ``cursor``,
    ``resize`` or ``close``.  The remaining fields carry whatever that kind
    needs; unused ones stay at zero.
    """

    kind: str
    key: int = 0
    scancode: int = 0
    action: int = 0
    mods: int = 0
    button: int = 0
    x: float = 0.0
    y: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    text: str = ""


class FrameSource(Protocol):
    """Something that can redraw the frame it last produced.

    :meth:`Window.read_pixels` needs one on a visible window: after
    ``glfwSwapBuffers`` the back buffer's contents are undefined by
    specification, and on this driver they come back solid black, so a
    screenshot taken after the swap would silently be an empty image.
    """

    def repaint(self, target: moderngl.Framebuffer) -> None: ...


_BUTTON_NAMES = {
    glfw.MOUSE_BUTTON_LEFT: "left",
    glfw.MOUSE_BUTTON_RIGHT: "right",
    glfw.MOUSE_BUTTON_MIDDLE: "middle",
}

#: Keys that ``fctx.ui.controls`` matches by name rather than by character.
_NAMED_KEYS = {
    glfw.KEY_ESCAPE: "escape", glfw.KEY_SPACE: "space", glfw.KEY_ENTER: "enter",
    glfw.KEY_TAB: "tab", glfw.KEY_BACKSPACE: "backspace",
    glfw.KEY_LEFT: "left", glfw.KEY_RIGHT: "right",
    glfw.KEY_UP: "up", glfw.KEY_DOWN: "down",
    glfw.KEY_LEFT_CONTROL: "ctrl", glfw.KEY_RIGHT_CONTROL: "ctrl",
    glfw.KEY_LEFT_SHIFT: "shift", glfw.KEY_RIGHT_SHIFT: "shift",
    glfw.KEY_LEFT_ALT: "alt", glfw.KEY_RIGHT_ALT: "alt",
    glfw.KEY_PERIOD: ".", glfw.KEY_COMMA: ",", glfw.KEY_MINUS: "-",
    glfw.KEY_EQUAL: "=", glfw.KEY_LEFT_BRACKET: "[",
    glfw.KEY_RIGHT_BRACKET: "]", glfw.KEY_SLASH: "/",
}
for _i in range(1, 13):
    _NAMED_KEYS[getattr(glfw, f"KEY_F{_i}")] = f"f{_i}"
for _i in range(10):
    _NAMED_KEYS[getattr(glfw, f"KEY_{_i}")] = str(_i)
    _NAMED_KEYS[getattr(glfw, f"KEY_KP_{_i}")] = str(_i)
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _NAMED_KEYS[getattr(glfw, f"KEY_{_c}")] = _c.lower()


def _key_name(key: int, scancode: int) -> str | None:
    name = _NAMED_KEYS.get(key)
    if name is not None:
        return name
    # A key this table does not list may still be a printable character on a
    # non-US layout; glfw can tell us which one.
    try:
        printable = glfw.get_key_name(key, scancode)
    except Exception:
        return None
    return printable.lower() if printable else None


def _mods(mods: int) -> dict[str, bool]:
    return {
        "shift": bool(mods & glfw.MOD_SHIFT),
        "ctrl": bool(mods & glfw.MOD_CONTROL),
        "alt": bool(mods & glfw.MOD_ALT),
    }


_glfw_users = 0

#: Addresses of the GLFW window handles that are still alive.  A GLFW handle
#: is a ctypes pointer and two pointers to the same window do not compare
#: equal, so identity has to be tested on the address.
_LIVE_HANDLES: set[int] = set()


def _handle_addr(handle: object) -> int:
    return ctypes.cast(handle, ctypes.c_void_p).value or 0

#: Live windows, keyed by the id of the moderngl context each one owns.  A
#: caller that hands the renderer ``window.ctx`` instead of the window means
#: the same thing by it, and without this the renderer would take the bare
#: context path: composite into the default framebuffer even when the window
#: is headless and has an offscreen one, and never follow a resize.  The
#: values are weak so that a window nobody closed is still collectable.
_WINDOWS_BY_CONTEXT: weakref.WeakValueDictionary[int, Window] = (
    weakref.WeakValueDictionary())


def window_for_context(ctx: moderngl.Context) -> Window | None:
    """Return the :class:`Window` that owns ``ctx``, if one does."""
    return _WINDOWS_BY_CONTEXT.get(id(ctx))


def _acquire_glfw() -> None:
    global _glfw_users
    if _glfw_users == 0:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed; no usable window system")
    _glfw_users += 1


def _release_glfw() -> None:
    global _glfw_users
    _glfw_users = max(0, _glfw_users - 1)
    if _glfw_users == 0:
        glfw.terminate()


class Window:
    """A GLFW window plus its ModernGL context, or a hidden headless twin."""

    def __init__(self, cfg: AppConfig | RenderConfig, *,
                 headless: bool | None = None) -> None:
        # The application has an AppConfig; a test that only cares about the
        # framebuffer has a RenderConfig and a headless flag.  Accepting both
        # keeps the caller from having to synthesise the half it does not own.
        if isinstance(cfg, RenderConfig):
            self.render_cfg = cfg
            self.cfg: AppConfig | None = None
        else:
            self.render_cfg = cfg.render
            self.cfg = cfg
        self.headless = bool(headless if headless is not None
                             else (self.cfg is not None and self.cfg.headless))
        self.events: deque[InputEvent] = deque(maxlen=4096)
        self._teardown: list[object] = []
        self._closed = False
        self._cursor: tuple[float, float] | None = None
        self._offscreen: moderngl.Framebuffer | None = None
        self._offscreen_tex: moderngl.Texture | None = None
        self._windowed_rect: tuple[int, int, int, int] | None = None
        self._frame_source: FrameSource | None = None
        self._capture_fbo: moderngl.Framebuffer | None = None
        self._capture_tex: moderngl.Texture | None = None
        #: False once swap() has handed the back buffer to the driver and
        #: nothing has drawn into it since.
        self._back_buffer_valid = False

        _acquire_glfw()
        try:
            self._create(self.render_cfg)
        except Exception:
            _release_glfw()
            raise

    # -- construction -----------------------------------------------------

    def _create(self, r: RenderConfig) -> None:
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE if self.headless else glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.SRGB_CAPABLE, glfw.FALSE)
        # Everything is rendered into an f16 offscreen target and tonemapped by
        # hand, so a multisampled default framebuffer would only cost memory.
        glfw.window_hint(glfw.SAMPLES, 0)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        # A second window must not steal the current context from the first
        # one for good: destroying it would then leave *no* context current
        # and every later GL call on the survivor fails.
        self._previous_context = glfw.get_current_context()

        monitor = None
        width, height = int(r.width), int(r.height)
        if r.fullscreen and not self.headless:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            width, height = mode.size.width, mode.size.height

        self.handle = glfw.create_window(width, height, r.title, monitor, None)
        if not self.handle:
            raise RuntimeError(
                "failed to create an OpenGL 4.3 core window; "
                "the driver may not support it")

        _LIVE_HANDLES.add(_handle_addr(self.handle))
        glfw.make_context_current(self.handle)
        glfw.swap_interval(1 if (r.vsync and not self.headless) else 0)

        # gc_mode is left at moderngl's default (no automatic collection).
        # Every GL object this package creates is released explicitly, and
        # automatic collection would otherwise run release() from whatever
        # thread CPython happened to collect on, with no context current.
        self.ctx = moderngl.create_context(require=430)
        fb_w, fb_h = glfw.get_framebuffer_size(self.handle)
        self._size = (max(1, fb_w), max(1, fb_h))
        self._window_size = self._size
        _WINDOWS_BY_CONTEXT[id(self.ctx)] = self
        self._sync_screen()

        if self.headless:
            self._make_offscreen(self._size)

        self._install_callbacks()

    def _sync_screen(self) -> None:
        """Point the default framebuffer's viewport at the whole window.

        ``moderngl`` measures ``ctx.screen`` once, when the context is
        created, and never notices a resize.  Left alone it keeps drawing the
        composite into the original rectangle, so enlarging the window leaves
        a stale band down two of its edges.
        """
        self.ctx.screen.viewport = (0, 0, self._size[0], self._size[1])
        self.ctx.screen.scissor = None

    def _make_offscreen(self, size: tuple[int, int]) -> None:
        if self._offscreen is not None:
            self._offscreen.release()
            assert self._offscreen_tex is not None
            self._offscreen_tex.release()
        self._offscreen_tex = self.ctx.texture(size, 4, dtype="f1")
        self._offscreen_tex.repeat_x = False
        self._offscreen_tex.repeat_y = False
        self._offscreen = self.ctx.framebuffer(
            color_attachments=[self._offscreen_tex])

    def _install_callbacks(self) -> None:
        def on_key(_w: object, key: int, scancode: int, action: int, mods: int) -> None:
            self.events.append(InputEvent("key", key=key, scancode=scancode,
                                          action=action, mods=mods))

        def on_char(_w: object, codepoint: int) -> None:
            self.events.append(InputEvent("char", text=chr(codepoint)))

        def on_button(_w: object, button: int, action: int, mods: int) -> None:
            x, y = glfw.get_cursor_pos(self.handle)
            self.events.append(InputEvent("mouse", button=button, action=action,
                                          mods=mods, x=x, y=y))

        def on_scroll(_w: object, dx: float, dy: float) -> None:
            self.events.append(InputEvent("scroll", dx=dx, dy=dy))

        def on_cursor(_w: object, x: float, y: float) -> None:
            prev = self._cursor
            self._cursor = (x, y)
            dx = 0.0 if prev is None else x - prev[0]
            dy = 0.0 if prev is None else y - prev[1]
            self.events.append(InputEvent("cursor", x=x, y=y, dx=dx, dy=dy))

        def on_resize(_w: object, width: int, height: int) -> None:
            size = (max(1, width), max(1, height))
            if size == self._size:
                return
            self._size = size
            self._sync_screen()
            self._release_capture()
            if self.headless:
                self._make_offscreen(size)
            self.events.append(InputEvent("resize", x=size[0], y=size[1]))

        def on_window_resize(_w: object, width: int, height: int) -> None:
            # Cursor positions arrive in window coordinates, which are not
            # framebuffer pixels on a scaled display, so the size the pointer
            # is normalised against has to come from this callback and not
            # from the framebuffer one.
            self._window_size = (max(1, width), max(1, height))

        def on_close(_w: object) -> None:
            self.events.append(InputEvent("close"))

        glfw.set_key_callback(self.handle, on_key)
        glfw.set_char_callback(self.handle, on_char)
        glfw.set_mouse_button_callback(self.handle, on_button)
        glfw.set_scroll_callback(self.handle, on_scroll)
        glfw.set_cursor_pos_callback(self.handle, on_cursor)
        glfw.set_framebuffer_size_callback(self.handle, on_resize)
        glfw.set_window_size_callback(self.handle, on_window_resize)
        glfw.set_window_close_callback(self.handle, on_close)
        self._window_size = tuple(glfw.get_window_size(self.handle))  # type: ignore[assignment]

    # -- per-frame --------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def aspect(self) -> float:
        return self._size[0] / max(1, self._size[1])

    @property
    def framebuffer(self) -> moderngl.Framebuffer:
        """The framebuffer the final composite must be written into."""
        return self._offscreen if self._offscreen is not None else self.ctx.screen

    @property
    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.handle))

    def request_close(self) -> None:
        glfw.set_window_should_close(self.handle, True)

    def poll(self) -> list[InputEvent]:
        """Pump the OS queue and return everything that arrived since last call."""
        glfw.poll_events()
        drained = list(self.events)
        self.events.clear()
        return drained

    def poll_events(self) -> list[ControlEvent]:
        """Pump the OS queue and return events in :mod:`fctx.ui.controls` terms.

        ``fctx.ui`` is deliberately free of glfw, so the translation from key
        codes to the names it matches on has to happen on this side of the
        boundary.  ``poll`` returns the same events untranslated, for anything
        that needs the raw codes.
        """
        out: list[ControlEvent] = []
        for ev in self.poll():
            if ev.kind == "key":
                name = _key_name(ev.key, ev.scancode)
                if name is None or ev.action == glfw.REPEAT:
                    # Repeats are dropped: Controls already ramps a held key
                    # from its own key-down set, and feeding it the OS repeat
                    # as well makes the dial jump at the auto-repeat rate.
                    continue
                kind = (EventKind.KEY_DOWN if ev.action == glfw.PRESS
                        else EventKind.KEY_UP)
                out.append(ControlEvent(kind, name, **_mods(ev.mods)))
            elif ev.kind == "mouse":
                name = _BUTTON_NAMES.get(ev.button)
                if name is None:
                    continue
                kind = (EventKind.MOUSE_DOWN if ev.action == glfw.PRESS
                        else EventKind.MOUSE_UP)
                out.append(ControlEvent(kind, name, x=ev.x, y=ev.y,
                                        **_mods(ev.mods)))
            elif ev.kind == "cursor":
                out.append(ControlEvent(EventKind.CURSOR, x=ev.x, y=ev.y))
            elif ev.kind == "scroll":
                out.append(ControlEvent(
                    EventKind.SCROLL, x=ev.dx, y=ev.dy,
                    ctrl=self.key_down(glfw.KEY_LEFT_CONTROL)
                    or self.key_down(glfw.KEY_RIGHT_CONTROL)))
            elif ev.kind == "resize":
                # In window coordinates, to match the cursor events that get
                # normalised against this size.
                w, h = self._window_size
                out.append(ControlEvent(EventKind.RESIZE, x=float(w), y=float(h)))
            elif ev.kind == "close":
                out.append(ControlEvent(EventKind.CLOSE))
        return out

    def mouse_down(self, button: int) -> bool:
        return glfw.get_mouse_button(self.handle, button) == glfw.PRESS

    def key_down(self, key: int) -> bool:
        return glfw.get_key(self.handle, key) == glfw.PRESS

    def swap(self) -> None:
        self._back_buffer_valid = False
        if not self.headless:
            glfw.swap_buffers(self.handle)
        else:
            # Nothing is presented, but the driver still needs a point at which
            # the frame's commands are known to be complete before read_pixels.
            self.ctx.finish()

    def toggle_fullscreen(self) -> None:
        if self.headless:
            return
        if glfw.get_window_monitor(self.handle):
            assert self._windowed_rect is not None
            x, y, w, h = self._windowed_rect
            glfw.set_window_monitor(self.handle, None, x, y, w, h, 0)
            self._windowed_rect = None
        else:
            x, y = glfw.get_window_pos(self.handle)
            w, h = glfw.get_window_size(self.handle)
            self._windowed_rect = (x, y, w, h)
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(self.handle, monitor, 0, 0,
                                    mode.size.width, mode.size.height,
                                    mode.refresh_rate)

    # -- output -----------------------------------------------------------

    @property
    def frame_source(self) -> FrameSource | None:
        return self._frame_source

    def set_frame_source(self, source: FrameSource | None) -> None:
        """Register who can redraw the last frame, for a post-swap screenshot."""
        self._frame_source = source

    def mark_frame_drawn(self) -> None:
        """Record that fresh pixels have been written to the output buffer."""
        self._back_buffer_valid = True

    def _ensure_capture(self) -> moderngl.Framebuffer:
        if self._capture_fbo is None:
            self._capture_tex = self.ctx.texture(self._size, 4, dtype="f1")
            self._capture_fbo = self.ctx.framebuffer([self._capture_tex])
        return self._capture_fbo

    def _release_capture(self) -> None:
        if self._capture_fbo is not None:
            self._capture_fbo.release()
            assert self._capture_tex is not None
            self._capture_tex.release()
            self._capture_fbo = None
            self._capture_tex = None

    def read_pixels(self, *, components: int = 3) -> np.ndarray:
        """Return the last drawn frame as ``(H, W, C)`` uint8, top-down.

        Reading a visible window after :meth:`swap` cannot come from the back
        buffer: the specification leaves its contents undefined once it has
        been presented, and this driver returns solid black.  When that has
        happened the frame source registered by the renderer redraws the frame
        it last composited into an offscreen buffer instead, so the caller gets
        the same image whichever side of the swap it asks on.
        """
        fbo = self.framebuffer
        if fbo is self.ctx.screen and not self._back_buffer_valid:
            if self._frame_source is None:
                raise RuntimeError(
                    "read_pixels() on a visible window after swap(), with no "
                    "frame source registered: the presented back buffer is "
                    "undefined.  Read before swapping, or construct the "
                    "Renderer with this Window so it can repaint the frame.")
            fbo = self._ensure_capture()
            self._frame_source.repaint(fbo)

        width, height = self._size
        # moderngl measures ctx.screen once and the viewport is the only part
        # of it that a resize updates, so the extent has to be passed in.
        raw = fbo.read(viewport=(0, 0, width, height), components=components,
                       dtype="f1")
        if fbo is self.ctx.screen:
            # Reading the default framebuffer leaves exactly one
            # GL_INVALID_OPERATION latched (moderngl selects a colour
            # attachment, which FBO 0 does not have).  The pixels are correct;
            # draining it here stops the next check_gl_error from blaming
            # whatever happens to run after the screenshot.
            _ = self.ctx.error
        img = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, components)
        # GL's origin is bottom-left and every image format expects top-left.
        return np.ascontiguousarray(img[::-1])

    def save_png(self, path: str | Path) -> Path:
        from PIL import Image

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.read_pixels()).save(out)
        return out

    # -- teardown ---------------------------------------------------------

    def add_teardown(self, obj: object) -> None:
        """Register something whose ``release()`` must run before the context dies.

        CUDA-OpenGL interop registrations are the reason this exists: unmapping
        or unregistering a buffer after the GL context has gone raises
        "invalid OpenGL or DirectX context" from the CUDA driver, which surfaces
        as a noisy crash at interpreter exit rather than where the mistake was.
        """
        self._teardown.append(obj)

    def remove_teardown(self, obj: object) -> None:
        """Forget something that has already released itself.

        Without this the list grows by one renderer every time the scene is
        rebuilt, and each dead entry keeps the whole object graph it owned --
        every program, buffer and texture wrapper -- alive for the life of
        the window.
        """
        try:
            self._teardown.remove(obj)
        except ValueError:
            pass

    def make_current(self) -> None:
        """Bind this window's GL context to the calling thread.

        Rebinding a context that is already current costs about 60 us in this
        driver, which is real money against a 1.5 ms frame, so the no-op case
        is skipped.
        """
        if _handle_addr(glfw.get_current_context()) != _handle_addr(self.handle):
            glfw.make_context_current(self.handle)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        glfw.make_context_current(self.handle)
        for obj in reversed(self._teardown):
            release = getattr(obj, "release", None)
            if release is not None:
                release()
        self._teardown.clear()
        self._frame_source = None
        _WINDOWS_BY_CONTEXT.pop(id(self.ctx), None)
        self._release_capture()
        if self._offscreen is not None:
            self._offscreen.release()
            assert self._offscreen_tex is not None
            self._offscreen_tex.release()
            self._offscreen = None
            self._offscreen_tex = None
        # glfwDestroyWindow deletes the GL context itself.  Calling moderngl's
        # Context.release() as well deletes the driver context a second time,
        # and GLFW then reports "failed to clear current context" from a
        # handle that is already gone.
        glfw.destroy_window(self.handle)
        _LIVE_HANDLES.discard(_handle_addr(self.handle))
        # Only hand the context back to a window that is still there.  Closing
        # two windows in the order they were opened would otherwise make a
        # destroyed context current, which GLFW reports as "WGL: Failed to
        # make context current" and which leaves the thread with no context
        # at all.
        if (self._previous_context
                and _handle_addr(self._previous_context) in _LIVE_HANDLES):
            glfw.make_context_current(self._previous_context)
        _release_glfw()

    def __enter__(self) -> Window:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
