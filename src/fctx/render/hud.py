"""The on-screen readouts and the hardness dial.

The dial is the part of the interface the demo is actually about, so it is
sized for a screen recording rather than for an engineer sitting a foot from
the monitor: the state word is set at roughly a fortieth of the window height,
the gauge runs most of the window's width, and the colour swatch repeats the
material colour so the same information survives a compressed video where
small text will not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.material import Material
from ..core.types import FrameStats
from .text import RGBA, OverlayBatch

__all__ = ["HudTheme", "Hud", "CONTROL_HINT"]

#: The on-screen control strip.  It has to carry every key the README and
#: ARCHITECTURE section 8 document, or the demo has a key nobody watching a
#: recording can discover -- but it also has to stay one line: _hint() shrinks
#: the font to make it fit, so each extra character costs every other one
#: legibility.  At 180 characters it still draws at its full 14 px on any 16:9
#: window; the separators are two spaces for that reason, not three.
CONTROL_HINT = (
    "WHEEL/[ ] hardness  M/N material  D demo  A sweep  F wind  1-5 preset  R reset  P pause  . step  "
    "RIGHT-DRAG orbit  CTRL+WHEEL zoom  H hud  W webcam  K hands  G wire  "
    "F9 rec  F12 shot  F11 full  ESC quit"
)


def _srgb(c: tuple[float, float, float], alpha: float = 1.0) -> RGBA:
    """Linear scene colour to the display space the overlay is drawn in.

    The overlay is composited after the tonemap and the gamma encode, so a
    ``Material.color`` pasted in raw would come out noticeably darker than the
    body it is supposed to be labelling.
    """
    g = 1.0 / 2.2
    return (max(c[0], 0.0) ** g, max(c[1], 0.0) ** g, max(c[2], 0.0) ** g, alpha)


@dataclass(frozen=True, slots=True)
class HudTheme:
    panel: RGBA = (0.035, 0.045, 0.065, 0.82)
    panel_edge: RGBA = (0.32, 0.72, 0.95, 0.30)
    text: RGBA = (0.93, 0.96, 1.00, 1.0)
    text_dim: RGBA = (0.56, 0.64, 0.75, 1.0)
    accent: RGBA = (0.32, 0.80, 1.00, 1.0)
    warn: RGBA = (1.00, 0.62, 0.30, 1.0)
    track: RGBA = (0.10, 0.12, 0.16, 0.95)
    knob: RGBA = (1.00, 1.00, 1.00, 1.0)


class Hud:
    """Emits the whole 2D interface into an :class:`OverlayBatch`."""

    def __init__(self, theme: HudTheme | None = None) -> None:
        self.theme = theme or HudTheme()

    # -- public -----------------------------------------------------------

    def build(
        self,
        batch: OverlayBatch,
        size: tuple[int, int],
        materials: list[Material],
        stats: FrameStats,
        hud_lines: list[str],
        *,
        hardness: float | None = None,
    ) -> None:
        """Lay the interface out for a window of ``size``.

        ``materials[0]`` drives the dial: with several bodies on stage the
        dial shows the one the scene was built around.  A ``hud_lines`` entry
        that starts with a space is drawn in the dim colour, which is how a
        caller marks a continuation or a less important line.
        """
        width, height = size
        scale = max(height, 540) / 900.0
        self._stats_panel(batch, width, height, scale, stats, hud_lines)
        if materials:
            knob = (materials[0].hardness if hardness is None
                    else min(max(float(hardness), 0.0), 1.0))
            self._dial(batch, width, height, scale, materials[0], knob)
        self._hint(batch, width, height, scale, bool(materials))

    def panels(self, batch: OverlayBatch, size: tuple[int, int],
               panels: Sequence[Any]) -> None:
        """Draw response targets: a labelled box that fills as it is chosen.

        Each item needs ``text``, ``x`` (centre) and ``y`` (top) as fractions
        of the frame, ``w`` as a fraction of the width, ``progress`` in
        [0, 1] and ``active``.
        """
        width, height = size
        scale = max(height, 540) / 900.0
        accent = self.theme.accent
        for p in panels:
            w = float(p.w) * width
            h = 74.0 * scale
            x = float(p.x) * width - w * 0.5
            y = float(p.y) * height
            active = bool(p.active)
            batch.rect(x, y, w, h, (0.03, 0.05, 0.08, 0.80 if active else 0.55),
                       radius=14.0 * scale)
            if active:
                batch.rect(x - 2.0, y - 2.0, w + 4.0, h + 4.0,
                           (accent[0], accent[1], accent[2], 0.35), radius=16.0 * scale)
                batch.rect(x, y, w, h, (0.03, 0.05, 0.08, 0.85), radius=14.0 * scale)
            size_px = 28.0 * scale
            batch.text(str(p.text), x + w * 0.5, y + h * 0.5 - size_px * 0.62, size_px,
                       (1.0, 1.0, 1.0, 1.0 if active else 0.75), align="center")
            bar = 7.0 * scale
            prog = min(max(float(p.progress), 0.0), 1.0)
            batch.rect(x + 12.0 * scale, y + h - bar - 9.0 * scale,
                       w - 24.0 * scale, bar, (1.0, 1.0, 1.0, 0.12), radius=bar * 0.5)
            if prog > 0.0:
                batch.rect(x + 12.0 * scale, y + h - bar - 9.0 * scale,
                           (w - 24.0 * scale) * prog, bar,
                           (accent[0], accent[1], accent[2], 1.0), radius=bar * 0.5)

    def badges(self, batch: OverlayBatch, size: tuple[int, int],
               notifications: Sequence[Any] = (), paused: bool = False) -> None:
        """Draw transient messages and the paused marker.

        These are deliberately outside :meth:`build`: a notification must stay
        visible while the HUD is hidden, because hiding the HUD for a clean
        recording is exactly when the user still needs to see that the preset
        changed or the screenshot landed.
        """
        width, height = size
        scale = max(height, 540) / 900.0
        if paused:
            self._paused(batch, width, scale)
        y = 34.0 * scale + (22.0 * scale if paused else 0.0)
        for item in notifications:
            text = str(getattr(item, "text", item))
            alpha = float(getattr(item, "alpha", 1.0))
            if alpha <= 0.0:
                continue
            size_px = 22.0 * scale
            w = batch.measure(text, size_px)[0]
            pad = 14.0 * scale
            batch.rect(width * 0.5 - w * 0.5 - pad, y - pad * 0.5,
                       w + pad * 2.0, size_px + pad,
                       (0.03, 0.05, 0.08, 0.72 * alpha),
                       radius=(size_px + pad) * 0.5)
            batch.text(text, width * 0.5, y, size_px,
                       (self.theme.accent[0], self.theme.accent[1],
                        self.theme.accent[2], alpha), align="center")
            y += size_px + pad * 1.6

    def _paused(self, batch: OverlayBatch, width: int, scale: float) -> None:
        size_px = 20.0 * scale
        text = "PAUSED"
        w = len(text) * batch.atlas.advance(size_px)
        pad = 12.0 * scale
        batch.rect(width * 0.5 - w * 0.5 - pad, 8.0 * scale, w + pad * 2.0,
                   size_px + pad, (0.42, 0.10, 0.12, 0.85),
                   radius=(size_px + pad) * 0.5)
        batch.text(text, width * 0.5, 8.0 * scale + pad * 0.5, size_px,
                   self.theme.text, align="center")

    # -- pieces -----------------------------------------------------------

    def _panel(self, batch: OverlayBatch, x: float, y: float, w: float, h: float,
               scale: float) -> None:
        r = 14.0 * scale
        batch.rect(x - 1.5, y - 1.5, w + 3.0, h + 3.0, self.theme.panel_edge,
                   radius=r + 1.5)
        batch.rect(x, y, w, h, self.theme.panel, radius=r)

    def _stats_panel(self, batch: OverlayBatch, width: int, height: int,
                     scale: float, stats: FrameStats,
                     hud_lines: list[str]) -> None:
        body = self._stat_lines(stats) + list(hud_lines)
        # The panel sits on top of the matter, so every row it does not need
        # is a row of the simulation nobody can see. 14 px still reads in a
        # 1080p screen recording; the dial is what has to be legible from
        # across a room, not the telemetry.
        size = 14.0 * scale
        title_size = 17.0 * scale
        pad = 13.0 * scale
        line_h = size * 1.26

        longest = max((len(s) for s in body), default=0)
        pw = max(len("FCTX MATTER STUDIO") * batch.atlas.advance(title_size),
                 longest * batch.atlas.advance(size)) + pad * 2.0
        x, y = 22.0 * scale, 22.0 * scale

        # The panel is as wide as its longest line, and the longest line is
        # supplied by the caller: with the synthetic source's help text in it
        # the panel wants 600 px, which runs off the right edge of a window
        # narrower than about 0.66 of its height.  Shrink the type to fit
        # rather than clip it -- a stat line cut off mid-word reads as a
        # rendering fault, and these lines are the ones that say why the hand
        # is not being tracked.
        avail = width - x * 2.0
        if pw > avail > pad * 2.0:
            shrink = (avail - pad * 2.0) / (pw - pad * 2.0)
            size *= shrink
            title_size *= shrink
            line_h = size * 1.26
            pw = avail
        ph = pad * 2.0 + title_size * 1.9 + line_h * len(body)
        self._panel(batch, x, y, pw, ph, scale)

        batch.text("FCTX MATTER STUDIO", x + pad, y + pad, title_size,
                   self.theme.accent)
        batch.rect(x + pad, y + pad + title_size * 1.35, pw - pad * 2.0,
                   max(1.0, scale), self.theme.panel_edge)

        cursor = y + pad + title_size * 1.9
        for line in body:
            colour = self.theme.text_dim if line.startswith(" ") else self.theme.text
            batch.text(line, x + pad, cursor, size, colour)
            cursor += line_h

    @staticmethod
    def _stat_lines(stats: FrameStats) -> list[str]:
        return [
            f"{stats.fps:6.0f} fps   {stats.frame_ms:5.2f} ms/frame",
            f"phys {stats.physics_ms:5.2f}  draw {stats.render_ms:5.2f}  "
            f"track {stats.tracking_ms:5.2f} ms",
            f"{stats.particles:,} particles x {stats.substeps} substeps",
            f"{stats.constraints:,} constraints  {stats.contacts:,} contacts",
            f"{stats.hands} hand(s)   {stats.grabbed} held",
        ]

    def _dial(self, batch: OverlayBatch, width: int, height: int,
              scale: float, material: Material, knob_t: float) -> None:
        t = self.theme
        pad = 17.0 * scale
        # A window narrower than the panel's own padding would give the panel
        # a negative width, and every rounded-rect corner radius derived from
        # it then inverts.  There is nothing readable to draw at that size.
        pw = min(width - 72.0 * scale, 1180.0 * scale)
        if pw <= pad * 2.0:
            return
        ph = 128.0 * scale
        x = (width - pw) * 0.5
        y = height - ph - 22.0 * scale
        self._panel(batch, x, y, pw, ph, scale)

        # --- identity row -------------------------------------------------
        swatch = 44.0 * scale
        sx, sy = x + pad, y + pad
        batch.rect(sx - 2.0, sy - 2.0, swatch + 4.0, swatch + 4.0,
                   t.panel_edge, radius=10.0 * scale)
        batch.rect(sx, sy, swatch, swatch, _srgb(material.color),
                   radius=8.0 * scale)

        text_x = sx + swatch + 18.0 * scale
        pct = f"{knob_t * 100.0:.0f}%"
        right = x + pw - pad

        # Left and right of this row are laid out independently -- the state
        # word grows rightwards from the swatch, the modulus grows leftwards
        # from the panel edge -- and the two are sized off the window height
        # while the space between them comes from the window width.  On a tall
        # narrow window they are drawn on top of each other.  One shared
        # factor keeps their relative sizes, which is what makes the state
        # word read as the headline.
        label_size, name_size = 14.0 * scale, 29.0 * scale
        pct_size, desc_size = 36.0 * scale, 15.0 * scale
        span = max(right - text_x - 14.0 * scale, 1.0)
        pairs = ((material.hardness_name, name_size, material.describe(), desc_size),
                 (material.label.upper(), label_size, pct, pct_size))
        fit = 1.0
        for left_s, left_px, right_s, right_px in pairs:
            need = (len(left_s) * batch.atlas.advance(left_px)
                    + len(right_s) * batch.atlas.advance(right_px))
            if need > span:
                fit = min(fit, span / need)
        label_size *= fit
        name_size *= fit
        pct_size *= fit
        desc_size *= fit

        batch.text(material.label.upper(), text_x, sy, label_size, t.text_dim)
        batch.text(material.hardness_name, text_x, sy + 16.0 * scale,
                   name_size, t.text)
        batch.text(pct, right, sy - 3.0 * scale, pct_size, t.accent,
                   align="right")
        batch.text(material.describe(), right, sy + 34.0 * scale,
                   desc_size, t.text_dim, align="right")

        # --- gauge ----------------------------------------------------------
        gx = x + pad
        gw = pw - pad * 2.0
        gh = 16.0 * scale
        gy = y + pad + 58.0 * scale
        radius = gh * 0.5

        batch.rect(gx, gy, gw, gh, t.track, radius=radius)

        # The knob travels between the centres of the track's two rounded
        # ends, not between its outer edges, so at 0% and 100% it still sits
        # on the track instead of hanging off it.
        knob_x = gx + radius + (gw - gh) * knob_t
        fill = max(knob_x - gx, gh)
        soft = _srgb(material.params.color_soft)
        hard = _srgb(material.params.color_hard)
        # The gradient spans the *filled* length, so the colour under the knob
        # is the colour the body actually has right now.
        batch.gradient(gx, gy, fill, gh, soft, hard, radius=radius)

        for i in range(5):
            tick_x = gx + radius + (gw - gh) * (i / 4.0)
            batch.rect(tick_x - 1.0 * scale, gy + gh + 8.0 * scale,
                       2.0 * scale, 5.0 * scale, t.panel_edge)

        knob_r = gh * 0.92
        batch.circle(knob_x, gy + gh * 0.5, knob_r * 1.35,
                     (t.accent[0], t.accent[1], t.accent[2], 0.28))
        batch.circle(knob_x, gy + gh * 0.5, knob_r, t.knob)
        batch.circle(knob_x, gy + gh * 0.5, knob_r * 0.52, _srgb(material.color))

        label_y = gy + gh + 16.0 * scale
        batch.text("SOFT", gx, label_y, 13.0 * scale, t.text_dim)
        batch.text("HARD", gx + gw, label_y, 13.0 * scale, t.text_dim,
                   align="right")
        batch.text("HARDNESS", gx + gw * 0.5, label_y, 13.0 * scale, t.accent,
                   align="center")

    def _hint(self, batch: OverlayBatch, width: int, height: int,
              scale: float, has_dial: bool) -> None:
        size = 14.0 * scale
        # The hint sits over the stage, so it needs its own backing or it
        # becomes unreadable the moment a pale body drifts underneath it.
        text_w = min(len(CONTROL_HINT) * batch.atlas.advance(size),
                     width - 48.0 * scale)
        if text_w <= 0.0:
            return
        if text_w < len(CONTROL_HINT) * batch.atlas.advance(size):
            size = text_w / max(len(CONTROL_HINT), 1) / batch.atlas.aspect
        y = height - (246.0 if has_dial else 46.0) * scale
        pad_x, pad_y = 16.0 * scale, 6.0 * scale
        batch.rect(width * 0.5 - text_w * 0.5 - pad_x, y - pad_y,
                   text_w + pad_x * 2.0, size + pad_y * 2.0,
                   (0.02, 0.03, 0.045, 0.62), radius=(size + pad_y * 2.0) * 0.5)
        batch.text(CONTROL_HINT, width * 0.5, y, size, self.theme.text_dim,
                   align="center")
