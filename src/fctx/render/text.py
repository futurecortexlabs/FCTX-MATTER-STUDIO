"""Bitmap font baking and the single-batch 2D overlay.

The font atlas is rendered at startup with PIL from a monospace face already
present on the machine.  Shipping a font file would mean carrying a licence
and a binary asset for the sake of a HUD, and pulling in a text-shaping
dependency would mean a second rasteriser in the process; a monospace atlas is
enough because everything drawn here is either a readout or a label.

Everything 2D goes through :class:`OverlayBatch`: panels, the hardness dial,
the webcam inset, the landmark skeleton and every glyph are instances of the
same unit quad in the same buffer, drawn in one call in append order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np

from .geometry import unit_quad
from .shaders import ShaderLibrary

__all__ = ["TextAtlas", "OverlayBatch", "LabelCache", "FONT_CANDIDATES",
           "LABEL_FONT_CANDIDATES", "RGBA"]

#: Tried in order.  Consolas and Cascadia Mono ship with Windows; DejaVu comes
#: with most Linux distributions and with matplotlib.
FONT_CANDIDATES: tuple[str, ...] = (
    "consola.ttf",
    "CascadiaMono.ttf",
    "CascadiaCode.ttf",
    "DejaVuSansMono.ttf",
    "lucon.ttf",
    "cour.ttf",
)

#: For text the ASCII atlas cannot draw -- a Japanese prompt, a venue's own
#: material names.  Yu Gothic, Meiryo and BIZ UD ship with Windows; Noto is
#: what a Linux box is likely to have.
LABEL_FONT_CANDIDATES: tuple[str, ...] = (
    "YuGothM.ttc",
    "meiryo.ttc",
    "BIZ-UDGothicR.ttc",
    "msgothic.ttc",
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJKjp-Regular.otf",
    "NotoSansJP-Regular.ttf",
    "DejaVuSans.ttf",
)

FIRST_CHAR = 32
LAST_CHAR = 126
ATLAS_COLS = 16
_GLYPH_PAD = 2

MODE_RECT = 0.0
MODE_IMAGE = 1.0
MODE_GLYPH = 2.0
MODE_GRADIENT = 3.0

RGBA = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _Baked:
    image: np.ndarray
    cell_w: int
    cell_h: int
    source: str


def _bake(px: int) -> _Baked:
    from PIL import Image, ImageDraw, ImageFont

    font = None
    source = ""
    for name in FONT_CANDIDATES:
        for candidate in (Path("C:/Windows/Fonts") / name, Path(name)):
            try:
                font = ImageFont.truetype(str(candidate), px)
                source = str(candidate)
                break
            except OSError:
                continue
        if font is not None:
            break
    if font is None:
        try:
            font = ImageFont.load_default(size=px)
        except TypeError:
            font = ImageFont.load_default()
        source = "PIL default"

    try:
        ascent, descent = font.getmetrics()
    except AttributeError:
        ascent, descent = px, max(1, px // 4)

    advance = max(
        font.getlength(chr(c)) for c in range(FIRST_CHAR, LAST_CHAR + 1))
    cell_w = int(np.ceil(advance)) + 2 * _GLYPH_PAD
    cell_h = int(ascent + descent) + 2 * _GLYPH_PAD
    if cell_w <= 0 or cell_h <= 0:
        raise RuntimeError(
            f"font {source!r} reported a degenerate cell {cell_w}x{cell_h}")

    count = LAST_CHAR - FIRST_CHAR + 1
    rows = (count + ATLAS_COLS - 1) // ATLAS_COLS
    img = Image.new("L", (ATLAS_COLS * cell_w, rows * cell_h), 0)
    draw = ImageDraw.Draw(img)
    for i in range(count):
        col, row = i % ATLAS_COLS, i // ATLAS_COLS
        draw.text((col * cell_w + _GLYPH_PAD, row * cell_h + _GLYPH_PAD),
                  chr(FIRST_CHAR + i), font=font, fill=255)

    return _Baked(np.asarray(img, dtype=np.uint8), cell_w, cell_h, source)


class TextAtlas:
    """A monospace ASCII atlas on the GPU, plus the metrics to lay it out."""

    def __init__(self, ctx: moderngl.Context, px: int = 40) -> None:
        if px < 8:
            raise ValueError(f"TextAtlas: px must be at least 8, got {px}")
        baked = _bake(px)
        self.ctx = ctx
        self.cell_w = baked.cell_w
        self.cell_h = baked.cell_h
        self.source = baked.source
        self.rows = baked.image.shape[0] // baked.cell_h
        self.width = baked.image.shape[1]
        self.height = baked.image.shape[0]

        self.texture = ctx.texture((self.width, self.height), 1,
                                   np.ascontiguousarray(baked.image).tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = False
        self.texture.repeat_y = False

    @property
    def aspect(self) -> float:
        """Advance width divided by line height."""
        return self.cell_w / self.cell_h

    def advance(self, size_px: float) -> float:
        return size_px * self.aspect

    def measure(self, text: str, size_px: float) -> tuple[float, float]:
        longest = max((len(line) for line in text.split("\n")), default=0)
        lines = text.count("\n") + 1
        return longest * self.advance(size_px), lines * size_px

    def glyph_uv(self, ch: str) -> tuple[float, float, float, float]:
        code = ord(ch)
        if code < FIRST_CHAR or code > LAST_CHAR:
            code = ord("?")
        i = code - FIRST_CHAR
        col, row = i % ATLAS_COLS, i // ATLAS_COLS
        u0 = col * self.cell_w / self.width
        v0 = row * self.cell_h / self.height
        return u0, v0, u0 + self.cell_w / self.width, v0 + self.cell_h / self.height

    @staticmethod
    def count_quads(text: str) -> int:
        """Glyphs that actually produce geometry: spaces and newlines do not."""
        return sum(1 for ch in text if ch not in " \n\t")

    def release(self) -> None:
        self.texture.release()


def _label_font(px: int):
    from PIL import ImageFont

    for name in LABEL_FONT_CANDIDATES + FONT_CANDIDATES:
        for candidate in (Path("C:/Windows/Fonts") / name, Path(name)):
            try:
                return ImageFont.truetype(str(candidate), px), str(candidate)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=px), "PIL default"
    except TypeError:
        return ImageFont.load_default(), "PIL default"


class LabelCache:
    """Whole strings rasterised on demand, for text outside the atlas.

    The atlas is monospace ASCII, which is right for readouts and wrong for
    a prompt in the visitor's language.  A label is one PIL rendering of one
    string at one size, kept as a small RGBA texture (white, with the
    coverage in alpha, so the tint colours it) and reused frame after frame;
    the cache is bounded so a stream of one-off strings cannot grow it.
    """

    def __init__(self, ctx: moderngl.Context, capacity: int = 64) -> None:
        self.ctx = ctx
        self.capacity = int(capacity)
        self._fonts: dict[int, object] = {}
        self._items: dict[tuple[str, int], tuple[moderngl.Texture, int, int]] = {}
        self._order: list[tuple[str, int]] = []
        self.source = ""

    def _font(self, px: int):
        font = self._fonts.get(px)
        if font is None:
            font, self.source = _label_font(px)
            self._fonts[px] = font
        return font

    def get(self, text: str, size_px: float) -> tuple[moderngl.Texture, int, int]:
        """The texture for ``text`` at ``size_px`` line height, and its size."""
        from PIL import Image, ImageDraw

        px = max(8, int(round(size_px * 0.92)))
        key = (text, px)
        hit = self._items.get(key)
        if hit is not None:
            self._order.remove(key)
            self._order.append(key)
            return hit
        font = self._font(px)
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=font)
        pad = max(2, px // 8)
        w = max(1, int(right - left) + 2 * pad)
        h = max(1, int(bottom) + 2 * pad)
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).multiline_text((pad - left, pad), text, font=font, fill=255)
        alpha = np.asarray(img, dtype=np.uint8)
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = alpha
        tex = self.ctx.texture((w, h), 4, np.ascontiguousarray(rgba).tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        if len(self._order) >= self.capacity:
            old = self._order.pop(0)
            self._items.pop(old)[0].release()
        self._items[key] = (tex, w, h)
        self._order.append(key)
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    def release(self) -> None:
        for tex, _, _ in self._items.values():
            tex.release()
        self._items.clear()
        self._order.clear()


class OverlayBatch:
    """Accumulates 2D instances and draws them all in one call.

    Text the atlas cannot draw goes through :attr:`labels` when one is
    attached: each such string becomes one textured quad drawn after the
    main batch, in append order, so a prompt in Japanese layers exactly
    where its ASCII equivalent would.
    """

    FLOATS_PER_INSTANCE = 20

    def __init__(
        self,
        ctx: moderngl.Context,
        library: ShaderLibrary,
        atlas: TextAtlas,
        capacity: int = 16384,
    ) -> None:
        self.ctx = ctx
        self.atlas = atlas
        self.capacity = int(capacity)
        self._data = np.zeros((self.capacity, self.FLOATS_PER_INSTANCE),
                              dtype=np.float32)
        self._count = 0

        self.program = library.program("overlay", vertex="overlay.vert",
                                       fragment="overlay.frag")
        self._corner_vbo = ctx.buffer(unit_quad().tobytes())
        self._instance_vbo = ctx.buffer(
            reserve=self.capacity * self.FLOATS_PER_INSTANCE * 4, dynamic=True)
        self._vao = ctx.vertex_array(
            self.program,
            [
                (self._corner_vbo, "2f", "in_corner"),
                (self._instance_vbo, "4f 4f 4f 4f 4f/i",
                 "i_rect", "i_uv", "i_color", "i_color2", "i_params"),
            ],
        )
        self._blank = ctx.texture((1, 1), 4, b"\xff\xff\xff\xff")
        #: Optional :class:`LabelCache` for non-ASCII text.
        self.labels: LabelCache | None = None
        self._label_draws: list[tuple[moderngl.Texture,
                                      tuple[float, float, float, float], RGBA]] = []
        self._label_row = np.zeros((1, self.FLOATS_PER_INSTANCE), dtype=np.float32)

    # -- accumulation -----------------------------------------------------

    def clear(self) -> None:
        self._count = 0
        self._label_draws.clear()

    def measure(self, s: str, size_px: float) -> tuple[float, float]:
        """Width and height ``s`` would occupy, whichever path draws it."""
        if self.labels is not None and not s.isascii():
            _, w, h = self.labels.get(s, size_px)
            return float(w), float(h)
        return self.atlas.measure(s, size_px)

    def __len__(self) -> int:
        return self._count

    def _push(self, rect: Sequence[float], uv: Sequence[float],
              color: Sequence[float], color2: Sequence[float],
              params: Sequence[float]) -> None:
        if self._count >= self.capacity:
            raise RuntimeError(
                f"OverlayBatch is full at {self.capacity} instances; raise the "
                "capacity rather than dropping HUD elements silently")
        row = self._data[self._count]
        row[0:4] = rect
        row[4:8] = uv
        row[8:12] = color
        row[12:16] = color2
        row[16:20] = params
        self._count += 1

    def rect(self, x: float, y: float, w: float, h: float, color: RGBA,
             *, radius: float = 0.0, rotation: float = 0.0,
             softness: float = 1.0) -> None:
        self._push((x, y, w, h), (0.0, 0.0, 1.0, 1.0), color, color,
                   (radius, MODE_RECT, rotation, softness))

    def gradient(self, x: float, y: float, w: float, h: float,
                 left: RGBA, right: RGBA, *, radius: float = 0.0) -> None:
        self._push((x, y, w, h), (0.0, 0.0, 1.0, 1.0), left, right,
                   (radius, MODE_GRADIENT, 0.0, 1.0))

    def image(self, x: float, y: float, w: float, h: float, *,
              uv: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
              tint: RGBA = (1.0, 1.0, 1.0, 1.0), radius: float = 0.0) -> None:
        self._push((x, y, w, h), uv, tint, tint, (radius, MODE_IMAGE, 0.0, 1.0))

    def circle(self, cx: float, cy: float, r: float, color: RGBA) -> None:
        self.rect(cx - r, cy - r, 2.0 * r, 2.0 * r, color, radius=r)

    def line(self, x0: float, y0: float, x1: float, y1: float,
             width: float, color: RGBA) -> None:
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        if length < 1e-4:
            return
        angle = float(np.arctan2(dy, dx))
        # Built as a rotated rect centred on the segment; the vertex shader
        # rotates about the rect centre, so the rect is authored axis-aligned.
        self._push(((x0 + x1) * 0.5 - length * 0.5, (y0 + y1) * 0.5 - width * 0.5,
                    length, width),
                   (0.0, 0.0, 1.0, 1.0), color, color,
                   (width * 0.5, MODE_RECT, angle, 1.0))

    def text(self, s: str, x: float, y: float, size_px: float, color: RGBA,
             *, align: str = "left") -> float:
        """Draw ``s`` with its top-left at ``(x, y)``.  Returns the advance width."""
        if self.labels is not None and not s.isascii():
            tex, w, h = self.labels.get(s, size_px)
            if align == "center":
                x -= w * 0.5
            elif align == "right":
                x -= w
            self._label_draws.append((tex, (x, y - (h - size_px) * 0.5, float(w), float(h)),
                                      tuple(color)))
            return float(w)
        adv = self.atlas.advance(size_px)
        pen_y = y
        width = 0.0
        for line in s.split("\n"):
            line_w = len(line) * adv
            width = max(width, line_w)
            if align == "center":
                pen_x = x - line_w * 0.5
            elif align == "right":
                pen_x = x - line_w
            else:
                pen_x = x
            for ch in line:
                if ch not in " \t":
                    self._push((pen_x, pen_y, adv, size_px),
                               self.atlas.glyph_uv(ch), color, color,
                               (0.0, MODE_GLYPH, 0.0, 1.0))
                pen_x += adv
            pen_y += size_px
        return width

    # -- drawing ----------------------------------------------------------

    def draw(self, resolution: tuple[int, int],
             image: moderngl.Texture | None = None) -> None:
        if self._count == 0 and not self._label_draws:
            return
        self.program["u_resolution"].value = (float(resolution[0]),
                                              float(resolution[1]))
        self.atlas.texture.use(0)
        self.program["u_font"].value = 0
        self.program["u_image"].value = 1
        if self._count:
            self._instance_vbo.write(
                np.ascontiguousarray(self._data[: self._count]).tobytes())
            (image or self._blank).use(1)
            self._vao.render(moderngl.TRIANGLE_STRIP, vertices=4,
                             instances=self._count)
        for tex, rect, color in self._label_draws:
            row = self._label_row[0]
            row[0:4] = rect
            row[4:8] = (0.0, 0.0, 1.0, 1.0)
            row[8:12] = color
            row[12:16] = color
            row[16:20] = (0.0, MODE_IMAGE, 0.0, 1.0)
            self._instance_vbo.write(self._label_row.tobytes())
            tex.use(1)
            self._vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=1)

    def release(self) -> None:
        if self.labels is not None:
            self.labels.release()
            self.labels = None
        self._vao.release()
        self._corner_vbo.release()
        self._instance_vbo.release()
        self._blank.release()
