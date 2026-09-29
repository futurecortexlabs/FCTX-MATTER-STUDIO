"""Configuration files: a TOML document that maps onto :class:`AppConfig`.

An installation -- a booth, a classroom, a lab bench -- has its own camera,
its own lighting and its own idea of where the stage is, and none of that
belongs on a command line typed fresh every morning.  A file does:

    preset = "cloth"

    [tracking]
    camera_index = 1
    reference_hand_span = 0.24
    stage_half_width = 0.40

    [render]
    fullscreen = true
    show_hud = false

Every section is one of the dataclasses hanging off :class:`AppConfig`, every
key is one of its fields, and anything else is an error that names the key.
A typo that silently did nothing is exactly the failure a config file exists
to prevent.  ``--dump-config`` writes the effective configuration back out in
this format, so the way to start a file is to dump the defaults and delete
what you do not mean to change.
"""

from __future__ import annotations

import dataclasses
import enum
import tomllib
from pathlib import Path
from typing import Any

from .config import PRESETS, AppConfig, preset
from .core.types import MatterKind

__all__ = ["load_config", "apply_toml", "dump_config", "ConfigError"]


class ConfigError(ValueError):
    """A configuration file said something the application cannot use."""


#: Sections of the file, in the order they are dumped.
_SECTIONS = ("scene", "solver", "grab", "tracking", "render", "camera", "exhibit",
             "haptics")


def load_config(path: str | Path, base: AppConfig | None = None,
                ignore_preset: bool = False) -> AppConfig:
    """Read ``path`` and return the configuration it describes.

    The file may name a ``preset`` to start from; otherwise it starts from
    ``base`` (or the defaults).  Sections then override field by field.
    ``ignore_preset`` keeps ``base`` even when the file names a preset, which
    is how a command-line ``--preset`` wins over the file while the file's
    other sections still apply.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return apply_toml(data, base, where=str(path), ignore_preset=ignore_preset)


def apply_toml(data: dict[str, Any], base: AppConfig | None = None,
               where: str = "config", ignore_preset: bool = False) -> AppConfig:
    """Apply a parsed TOML mapping on top of ``base``."""
    data = dict(data)
    name = data.pop("preset", None)
    if name is not None and ignore_preset:
        name = None
    if name is not None:
        if name not in PRESETS:
            raise ConfigError(
                f"{where}: preset {name!r} is not one of {', '.join(PRESETS)}")
        cfg = preset(str(name))
    else:
        cfg = base if base is not None else AppConfig()

    top_fields = {f.name: f for f in dataclasses.fields(AppConfig)}
    overrides: dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTIONS:
            if not isinstance(value, dict):
                raise ConfigError(f"{where}: [{key}] must be a table")
            overrides[key] = _apply_section(getattr(cfg, key), value,
                                            f"{where} [{key}]")
        elif key in top_fields:
            overrides[key] = _coerce(top_fields[key], value, f"{where} {key}")
        else:
            raise ConfigError(
                f"{where}: unknown key {key!r}; sections are "
                f"{', '.join(_SECTIONS)} and top-level keys are "
                f"{', '.join(sorted(top_fields))}")
    return dataclasses.replace(cfg, **overrides)


def _apply_section(section: Any, values: dict[str, Any], where: str) -> Any:
    fields = {f.name: f for f in dataclasses.fields(section)}
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key not in fields:
            near = [n for n in fields if n.startswith(key[:3])]
            hint = f" (did you mean {', '.join(near)}?)" if near else ""
            raise ConfigError(f"{where}: unknown key {key!r}{hint}")
        out[key] = _coerce(fields[key], value, f"{where}.{key}")
    return dataclasses.replace(section, **out)


def _coerce(field: dataclasses.Field, value: Any, where: str) -> Any:
    """Turn a TOML value into the field's declared type, or explain why not."""
    hint = field.type if not isinstance(field.type, str) else field.type
    text = str(hint)
    default = field.default

    if isinstance(default, enum.Enum) or "MatterKind" in text:
        if isinstance(value, str):
            try:
                return MatterKind[value.upper()]
            except KeyError:
                raise ConfigError(
                    f"{where}: {value!r} is not one of "
                    f"{', '.join(k.name.lower() for k in MatterKind)}") from None
        if isinstance(value, int):
            return MatterKind(value)
        raise ConfigError(f"{where}: expected a matter kind, got {value!r}")
    if "Path" in text:
        if value is None or value == "":
            return None
        return Path(str(value))
    if text.startswith("tuple") or isinstance(default, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        if isinstance(default, tuple) and len(default) not in (0, len(value)):
            raise ConfigError(
                f"{where}: expected {len(default)} values, got {len(value)}")
        return tuple(float(v) for v in value)
    if text == "bool" or isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value
    if text == "int" or isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        if isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{where}: expected a whole number, got {value!r}")
        return int(value)
    if text == "float" or isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if text == "str" or isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if "None" in text and value is None:
        return None
    return value


def dump_config(cfg: AppConfig, preset_name: str | None = None) -> str:
    """Render ``cfg`` as a TOML document :func:`load_config` accepts."""
    lines = [
        "# FCTX MATTER STUDIO configuration.",
        "# Every key below is optional; delete what you do not mean to change.",
        "",
    ]
    if preset_name:
        lines.append(f'preset = "{preset_name}"')
    for f in dataclasses.fields(AppConfig):
        if f.name in _SECTIONS:
            continue
        value = getattr(cfg, f.name)
        if value is None:
            continue
        lines.append(f"{f.name} = {_toml(value)}")
    for name in _SECTIONS:
        section = getattr(cfg, name)
        lines.extend(["", f"[{name}]"])
        for f in dataclasses.fields(section):
            value = getattr(section, f.name)
            if value is None:
                continue
            lines.append(f"{f.name} = {_toml(value)}")
    return "\n".join(lines) + "\n"


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, enum.Enum):
        return f'"{value.name.lower()}"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, Path):
        return '"' + str(value).replace("\\", "/") + '"'
    return '"' + str(value).replace('"', '\\"') + '"'

