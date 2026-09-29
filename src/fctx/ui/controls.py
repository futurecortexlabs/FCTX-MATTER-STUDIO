"""Input handling and the hardness dial.

Kept free of glfw and OpenGL so it can be driven from a test with a synthetic
event list.  :mod:`fctx.app` translates the window's raw callbacks into the
:class:`InputEvent` values this module consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config import PRESET_KEYS


class EventKind(Enum):
    KEY_DOWN = auto()
    KEY_UP = auto()
    MOUSE_DOWN = auto()
    MOUSE_UP = auto()
    CURSOR = auto()
    SCROLL = auto()
    RESIZE = auto()
    CLOSE = auto()


@dataclass(slots=True, frozen=True)
class InputEvent:
    kind: EventKind
    #: For key events, the lowercase character or a name like "escape", "f11",
    #: "left", "space".  For mouse events, "left" / "right" / "middle".
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    shift: bool = False
    ctrl: bool = False
    alt: bool = False


# Keys that nudge the dial, and by how much per press.
_HARDNESS_KEYS = {"]": +0.05, "[": -0.05, "=": +0.05, "-": -0.05,
                  "up": +0.05, "down": -0.05}


@dataclass(slots=True)
class ControlState:
    """Everything the rest of the application reads from the user's input."""

    #: Where the dial is being dragged to, 0..1.
    hardness_target: float = 0.35
    #: Where the material actually is.  Follows the target with a time
    #: constant so that a scroll wheel -- which arrives as a burst of discrete
    #: clicks -- produces a continuous change in the material rather than a
    #: staircase.  The whole point of the demo is that hardness is a
    #: continuum, so it must not move in visible steps.
    hardness: float = 0.35

    paused: bool = False
    step_once: bool = False
    wireframe: bool = False
    wind: bool = False
    show_hud: bool = True
    show_webcam: bool = True
    show_hands: bool = True
    #: Sweep the dial back and forth automatically, for demos and recordings.
    auto_sweep: bool = False
    #: Run the choreographed demonstration (fctx.demo).  Only meaningful on
    #: the synthetic hand; on a camera it still drives the dial.
    demo: bool = False
    demo_toggle_requested: bool = False
    #: +1 / -1 for one frame: step the dial to the next / previous
    #: catalogue material (M / N).
    material_step: int = 0
    #: 0 / 1 for one frame: a staff answer for the study (Z left, X right).
    study_key: int | None = None
    _sweep_phase: float = 0.0

    preset_index: int = 0
    #: Set for one frame when the user asks for something.
    reset_requested: bool = False
    preset_requested: str | None = None
    screenshot_requested: bool = False
    fullscreen_requested: bool = False
    record_toggle_requested: bool = False
    quit_requested: bool = False

    # -- camera ------------------------------------------------------------
    orbit_delta: tuple[float, float] = (0.0, 0.0)
    zoom_delta: float = 0.0

    # -- pointer, used to drive the synthetic hand -------------------------
    #: Cursor in normalised window coordinates, origin top-left.
    pointer: tuple[float, float] = (0.5, 0.5)
    #: Synthetic hand depth, 0 = far, 1 = near.
    pointer_depth: float = 0.5
    #: Synthetic pinch, 0..1.
    synth_pinch: float = 0.0
    synth_curl: float = 0.0

    # -- internal ----------------------------------------------------------
    _orbit_active: bool = False
    _last_cursor: tuple[float, float] = (0.5, 0.5)
    _keys_down: set[str] = field(default_factory=set)
    _window_size: tuple[int, int] = (1600, 900)

    @property
    def ctrl_held(self) -> bool:
        return "ctrl" in self._keys_down

    def clear_oneshots(self) -> None:
        self.reset_requested = False
        self.preset_requested = None
        self.screenshot_requested = False
        self.fullscreen_requested = False
        self.record_toggle_requested = False
        self.demo_toggle_requested = False
        self.material_step = 0
        self.study_key = None
        self.step_once = False
        self.orbit_delta = (0.0, 0.0)
        self.zoom_delta = 0.0


class Controls:
    """Turn input events into a :class:`ControlState`."""

    #: Seconds for the material to travel roughly 63% of the way to the target.
    HARDNESS_TIME_CONSTANT = 0.11
    #: Full sweep period in seconds when auto_sweep is on.
    SWEEP_PERIOD = 7.0

    def __init__(self, state: ControlState | None = None) -> None:
        self.state = state or ControlState()

    # -- event handling ----------------------------------------------------

    def handle(self, events: list[InputEvent]) -> ControlState:
        s = self.state
        s.clear_oneshots()
        for ev in events:
            if ev.kind is EventKind.CLOSE:
                s.quit_requested = True
            elif ev.kind is EventKind.RESIZE:
                s._window_size = (max(int(ev.x), 1), max(int(ev.y), 1))
            elif ev.kind is EventKind.KEY_DOWN:
                self._key_down(ev)
            elif ev.kind is EventKind.KEY_UP:
                s._keys_down.discard(ev.name)
                if ev.name == "space":
                    s.synth_pinch = 0.0
            elif ev.kind is EventKind.MOUSE_DOWN:
                self._mouse_down(ev)
            elif ev.kind is EventKind.MOUSE_UP:
                if ev.name == "right":
                    s._orbit_active = False
                elif ev.name == "left":
                    s.synth_pinch = 0.0
            elif ev.kind is EventKind.CURSOR:
                self._cursor(ev)
            elif ev.kind is EventKind.SCROLL:
                self._scroll(ev)
        return s

    def _key_down(self, ev: InputEvent) -> None:
        s = self.state
        s._keys_down.add(ev.name)
        name = ev.name

        if name in _HARDNESS_KEYS:
            s.hardness_target = _clamp01(s.hardness_target + _HARDNESS_KEYS[name])
            s.auto_sweep = False
            return
        if name in "12345" and len(name) == 1:
            idx = int(name) - 1
            if idx < len(PRESET_KEYS):
                s.preset_index = idx
                s.preset_requested = PRESET_KEYS[idx]
            return

        match name:
            case "escape":
                s.quit_requested = True
            case "r":
                s.reset_requested = True
            case "p":
                s.paused = not s.paused
            case ".":
                s.step_once = True
                s.paused = True
            case "h":
                s.show_hud = not s.show_hud
            case "w":
                s.show_webcam = not s.show_webcam
            case "k":
                s.show_hands = not s.show_hands
            case "g":
                s.wireframe = not s.wireframe
            case "f":
                s.wind = not s.wind
            case "d":
                s.demo_toggle_requested = True
            case "z":
                s.study_key = 0
            case "x":
                s.study_key = 1
            case "m":
                s.material_step = 1
            case "n":
                s.material_step = -1
            case "a":
                s.auto_sweep = not s.auto_sweep
                s._sweep_phase = math.asin(
                    max(-1.0, min(1.0, s.hardness_target * 2.0 - 1.0)))
            case "f11":
                s.fullscreen_requested = True
            case "f12":
                s.screenshot_requested = True
            case "f9":
                s.record_toggle_requested = True
            case "space":
                s.synth_pinch = 1.0
            case "0":
                s.hardness_target = 0.0
                s.auto_sweep = False

    def _mouse_down(self, ev: InputEvent) -> None:
        s = self.state
        if ev.name == "right":
            s._orbit_active = True
            s._last_cursor = (ev.x, ev.y)
        elif ev.name == "left":
            s.synth_pinch = 1.0

    def _cursor(self, ev: InputEvent) -> None:
        s = self.state
        w, h = s._window_size
        nx = _clamp01(ev.x / w)
        ny = _clamp01(ev.y / h)
        if s._orbit_active:
            s.orbit_delta = (s.orbit_delta[0] + (ev.x - s._last_cursor[0]),
                             s.orbit_delta[1] + (ev.y - s._last_cursor[1]))
        else:
            s.pointer = (nx, ny)
        s._last_cursor = (ev.x, ev.y)

    def _scroll(self, ev: InputEvent) -> None:
        s = self.state
        if ev.ctrl or s.ctrl_held:
            s.zoom_delta += ev.y
        else:
            # The wheel is the primary way to work the dial during a demo,
            # so one click is a perceptible but not jarring step.
            s.hardness_target = _clamp01(s.hardness_target + 0.04 * ev.y)
            s.auto_sweep = False

    # -- per-frame update --------------------------------------------------

    def update(self, dt: float) -> ControlState:
        """Advance smoothing and held-key repeat.  Call once per frame."""
        s = self.state
        keys = s._keys_down

        if s.auto_sweep:
            s._sweep_phase += 2.0 * math.pi * dt / self.SWEEP_PERIOD
            s.hardness_target = 0.5 + 0.5 * math.sin(s._sweep_phase)

        # Holding a bracket key ramps continuously; tapping it steps.
        ramp = 0.0
        if "]" in keys or "=" in keys or "up" in keys:
            ramp += 0.9
        if "[" in keys or "-" in keys or "down" in keys:
            ramp -= 0.9
        if ramp:
            s.hardness_target = _clamp01(s.hardness_target + ramp * dt)
            s.auto_sweep = False

        # Exponential approach, framerate independent.
        k = 1.0 - math.exp(-dt / max(self.HARDNESS_TIME_CONSTANT, 1e-4))
        s.hardness += (s.hardness_target - s.hardness) * k
        if abs(s.hardness_target - s.hardness) < 2e-4:
            s.hardness = s.hardness_target

        # Synthetic-hand depth on the keyboard, so the mouse stays free for
        # the x/y position.
        depth = 0.0
        if "q" in keys:
            depth -= 1.0
        if "e" in keys:
            depth += 1.0
        if depth:
            s.pointer_depth = _clamp01(s.pointer_depth + depth * dt * 0.8)
        s.synth_curl = 1.0 if "c" in keys else 0.0
        return s


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# The general control list lives in one place, `render.hud.CONTROL_HINT`,
# which is the one the user actually sees.  A second copy here had already
# drifted out of agreement with it and was imported by nothing.
SYNTHETIC_HELP_LINES: tuple[str, ...] = (
    "SYNTHETIC HAND: move the mouse to steer, left-click or SPACE to pinch",
    "Q / E push the hand away / pull it closer,  C curls the fingers",
)
