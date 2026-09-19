"""GLSL loading, ``#include`` expansion and program construction.

The shaders live next to this module as real ``.glsl`` files rather than as
Python string literals so that an editor can syntax-check them and a compile
error can point at a file and a line.  GLSL has no include directive of its
own, so this module implements one: a flat textual splice with cycle
detection, which is all the pipeline needs to share the PBR and shadow code
between five different fragment shaders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import moderngl

__all__ = ["ShaderError", "ShaderLibrary", "SHADER_DIR"]

SHADER_DIR = Path(__file__).resolve().parent / "shaders"

_INCLUDE_RE = re.compile(r'^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$', re.MULTILINE)
_VERSION_RE = re.compile(r'^[ \t]*#version[^\n]*\n', re.MULTILINE)
#: NVIDIA reports errors as ``0(123) : error C0000: ...``; mesa as ``0:123(4):``.
_ERR_LINE_RE = re.compile(r"\b0[(:](\d+)[)(:]")


class ShaderError(RuntimeError):
    """Raised when a shader fails to load, compile or link."""


@dataclass
class ShaderLibrary:
    """Loads GLSL from ``root`` and builds :class:`moderngl.Program` objects."""

    ctx: moderngl.Context
    root: Path = SHADER_DIR
    #: Every program built through this library, keyed by its reported name.
    programs: dict[str, moderngl.Program] = field(default_factory=dict)
    _source_cache: dict[Path, str] = field(default_factory=dict, repr=False)

    def path(self, name: str) -> Path:
        p = self.root / name
        if not p.is_file():
            raise ShaderError(f"shader file not found: {p}")
        return p

    def _read(self, path: Path) -> str:
        cached = self._source_cache.get(path)
        if cached is None:
            cached = path.read_text(encoding="utf-8")
            self._source_cache[path] = cached
        return cached

    def source(self, name: str, *, defines: dict[str, object] | None = None) -> str:
        """Return the fully preprocessed source of ``name``."""
        text = self._expand(self.path(name), [])
        match = _VERSION_RE.search(text)
        if match is None:
            raise ShaderError(f"{name}: missing a #version directive")
        if defines:
            lines = "".join(f"#define {k} {v}\n" for k, v in defines.items())
            text = text[: match.end()] + lines + text[match.end():]
        return text

    def _expand(self, path: Path, stack: list[Path]) -> str:
        if path in stack:
            chain = " -> ".join(p.name for p in (*stack, path))
            raise ShaderError(f"circular #include: {chain}")
        stack = [*stack, path]
        text = self._read(path)

        def splice(match: re.Match[str]) -> str:
            child = (path.parent / match.group(1)).resolve()
            if not child.is_file():
                raise ShaderError(
                    f"{path.name}: #include \"{match.group(1)}\" does not exist")
            body = self._expand(child, stack)
            # An included file must not carry its own #version: GLSL only
            # accepts one and only as the first token of the unit.
            return _VERSION_RE.sub("", body).rstrip()

        return _INCLUDE_RE.sub(splice, text)

    def program(
        self,
        name: str,
        *,
        vertex: str,
        fragment: str,
        geometry: str | None = None,
        defines: dict[str, object] | None = None,
    ) -> moderngl.Program:
        """Compile and link a program, remembering it under ``name``."""
        stages = {
            "vertex_shader": self.source(vertex, defines=defines),
            "fragment_shader": self.source(fragment, defines=defines),
        }
        origins = {"vertex_shader": vertex, "fragment_shader": fragment}
        if geometry is not None:
            stages["geometry_shader"] = self.source(geometry, defines=defines)
            origins["geometry_shader"] = geometry
        try:
            prog = self.ctx.program(**stages)
        except Exception as exc:  # moderngl.Error and driver-specific subclasses
            raise ShaderError(self._explain(name, stages, origins, exc)) from exc
        self.programs[name] = prog
        return prog

    @staticmethod
    def _explain(
        name: str,
        stages: dict[str, str],
        origins: dict[str, str],
        exc: Exception,
    ) -> str:
        message = str(exc)
        files = ", ".join(f"{k.split('_')[0]}={v}" for k, v in origins.items())
        out = [f"program {name!r} failed ({files})", message.rstrip()]

        # Drivers report a line number in the *preprocessed* source, which is
        # not the line number in any file on disk, so quote the neighbourhood
        # instead of sending the reader on a hunt.
        hit = _ERR_LINE_RE.search(message)
        if hit is not None:
            lineno = int(hit.group(1))
            stage = next(
                (k for k in stages if k.split("_")[0] in message.lower()),
                "fragment_shader" if "fragment_shader" in stages else "vertex_shader",
            )
            lines = stages[stage].splitlines()
            lo = max(0, lineno - 6)
            hi = min(len(lines), lineno + 5)
            out.append(f"--- preprocessed {stage} lines {lo + 1}..{hi} ---")
            for i in range(lo, hi):
                mark = ">>" if i + 1 == lineno else "  "
                out.append(f"{mark} {i + 1:5d} | {lines[i]}")
        return "\n".join(out)

    def release(self) -> None:
        for prog in self.programs.values():
            prog.release()
        self.programs.clear()
