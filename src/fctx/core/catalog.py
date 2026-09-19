"""A catalogue of named materials, so the dial can say what it is showing.

The hardness dial is a continuum, which is the point; but a visitor reads a
name before a number, and a venue that sells foam or fabric wants *its*
names on the screen.  A catalogue entry pins one real-world material to one
point on the dial for one kind of matter:

* soft bodies -- a Young's modulus in pascals.  The built-in values are
  literature order-of-magnitude figures (gelatin dessert a few kPa, silicone
  rubber about a megapascal, tyre tread ten).
* cloth and grains -- the solver's stretch stiffness in N/m.  These are not
  textile or soil measurements; they are the stiffness at which the shipped
  sheet or pile *reads* as denim or gravel, and are labelled as such.

A venue supplies its own file (``[exhibit] catalog = "materials.toml"``):

    [[material]]
    name = "Our 40 kg/m3 foam"
    kind = "soft"
    value = 4.5e4        # Pa
    note = "seat cushion grade"

and the HUD shows the nearest entry to wherever the dial is, with the ``M``
and ``N`` keys stepping the dial from one entry to the next.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .material import Material, MaterialParams, log_lerp
from .types import MatterKind

__all__ = ["CatalogEntry", "CatalogError", "DEFAULT_CATALOG", "load_catalog",
           "for_kind", "nearest", "hardness_for", "headline", "step"]


class CatalogError(ValueError):
    """A catalogue file said something unusable."""


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    name: str
    kind: MatterKind
    #: Young's modulus (Pa) for soft bodies; stretch stiffness (N/m) otherwise.
    value: float
    note: str = ""


DEFAULT_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry("GELATIN DESSERT", MatterKind.SOFT, 8.0e3, "wobbles, tears easily"),
    CatalogEntry("SOFT SILICONE GEL", MatterKind.SOFT, 3.0e4, "cushion gel"),
    CatalogEntry("SOFT TISSUE", MatterKind.SOFT, 1.0e5, "skin, muscle at rest"),
    CatalogEntry("MEMORY FOAM", MatterKind.SOFT, 3.0e5, "slow to recover"),
    CatalogEntry("SILICONE RUBBER", MatterKind.SOFT, 1.0e6, "kitchen spatula"),
    CatalogEntry("NATURAL RUBBER", MatterKind.SOFT, 2.5e6, "rubber band"),
    CatalogEntry("TYRE TREAD", MatterKind.SOFT, 1.0e7, "hard rubber"),

    CatalogEntry("SILK CHIFFON", MatterKind.CLOTH, 2.0e2, "sheer, floats"),
    CatalogEntry("COTTON JERSEY", MatterKind.CLOTH, 8.0e2, "t-shirt"),
    CatalogEntry("COTTON POPLIN", MatterKind.CLOTH, 3.0e3, "shirt"),
    CatalogEntry("DENIM", MatterKind.CLOTH, 1.0e4, "jeans"),
    CatalogEntry("CANVAS", MatterKind.CLOTH, 3.0e4, "tote bag"),
    CatalogEntry("TARPAULIN", MatterKind.CLOTH, 9.0e4, "truck cover"),

    CatalogEntry("DRY SAND", MatterKind.GRAIN, 1.5e3, "runs through the fingers"),
    CatalogEntry("DAMP SAND", MatterKind.GRAIN, 6.0e3, "holds a shape briefly"),
    CatalogEntry("FINE GRAVEL", MatterKind.GRAIN, 2.0e4, "pea gravel"),
    CatalogEntry("COARSE GRAVEL", MatterKind.GRAIN, 6.0e4, "railway ballast"),
    CatalogEntry("PACKED EARTH", MatterKind.GRAIN, 1.8e5, "hard ground"),
)


def load_catalog(path: str | Path) -> tuple[CatalogEntry, ...]:
    """Read a ``[[material]]`` file; every entry must be complete and typed."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CatalogError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CatalogError(f"{path}: {exc}") from exc
    rows = data.get("material")
    if not isinstance(rows, list) or not rows:
        raise CatalogError(f"{path}: expected one or more [[material]] tables")
    entries: list[CatalogEntry] = []
    for i, row in enumerate(rows, 1):
        where = f"{path} [[material]] #{i}"
        if not isinstance(row, dict):
            raise CatalogError(f"{where}: not a table")
        unknown = set(row) - {"name", "kind", "value", "young", "stretch", "note"}
        if unknown:
            raise CatalogError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise CatalogError(f"{where}: 'name' must be a non-empty string")
        kind_text = row.get("kind")
        try:
            kind = MatterKind[str(kind_text).upper()]
        except KeyError:
            raise CatalogError(
                f"{where}: kind {kind_text!r} is not one of "
                f"{', '.join(k.name.lower() for k in MatterKind)}") from None
        value = row.get("value", row.get("young", row.get("stretch")))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise CatalogError(f"{where}: 'value' must be a positive number "
                               "(Pa for soft, N/m for cloth and grain)")
        note = row.get("note", "")
        if not isinstance(note, str):
            raise CatalogError(f"{where}: 'note' must be a string")
        entries.append(CatalogEntry(name.strip(), kind, float(value), note))
    return tuple(entries)


def for_kind(entries: tuple[CatalogEntry, ...] | list[CatalogEntry],
             kind: MatterKind) -> list[CatalogEntry]:
    """The entries of one kind, softest first."""
    return sorted((e for e in entries if e.kind is kind), key=lambda e: e.value)


def headline(material: Material) -> float:
    """The one number the catalogue compares against, for this kind."""
    if material.params.kind is MatterKind.SOFT:
        return float(material.young)
    return float(material.stretch_k)


def _range(params: MaterialParams) -> tuple[float, float]:
    if params.kind is MatterKind.SOFT:
        return params.young_soft, params.young_hard
    return params.stretch_soft, params.stretch_hard


def hardness_for(params: MaterialParams, value: float) -> float:
    """Where on the dial ``params`` reaches ``value`` (clamped to [0, 1]).

    The inverse of :func:`fctx.core.material.log_lerp`, so a catalogue entry
    outside the dial's range lands on the nearest end rather than raising.
    """
    lo, hi = _range(params)
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return float(math.log(value / lo) / math.log(hi / lo))


def nearest(entries: tuple[CatalogEntry, ...] | list[CatalogEntry],
            material: Material) -> CatalogEntry | None:
    """The entry closest to ``material`` on a logarithmic scale, or None."""
    value = headline(material)
    best: CatalogEntry | None = None
    best_d = math.inf
    for e in for_kind(entries, material.params.kind):
        d = abs(math.log(e.value / value))
        if d < best_d:
            best, best_d = e, d
    return best


def step(entries: tuple[CatalogEntry, ...] | list[CatalogEntry],
         material: Material, direction: int) -> CatalogEntry | None:
    """The next (``+1``) or previous (``-1``) entry from where the dial is.

    "Next" means the softest entry strictly stiffer than the current value,
    so pressing the key repeatedly walks the catalogue end to end without
    getting stuck on the entry the dial is already sitting on.
    """
    value = headline(material)
    ordered = for_kind(entries, material.params.kind)
    if not ordered:
        return None
    tol = 1.02  # within 2% counts as "already there"
    if direction > 0:
        for e in ordered:
            if e.value > value * tol:
                return e
        return ordered[-1]
    for e in reversed(ordered):
        if e.value < value / tol:
            return e
    return ordered[0]


def check_range(entries: tuple[CatalogEntry, ...] | list[CatalogEntry],
                params: MaterialParams) -> list[str]:
    """Names of entries the dial cannot actually reach for ``params``."""
    lo, hi = _range(params)
    return [e.name for e in entries
            if e.kind is params.kind and not (lo * 0.999 <= e.value <= hi * 1.001)]


def _unused(_: float = log_lerp(1.0, 2.0, 0.0)) -> None:  # keeps the import honest
    """``log_lerp`` is the forward map :func:`hardness_for` inverts."""
