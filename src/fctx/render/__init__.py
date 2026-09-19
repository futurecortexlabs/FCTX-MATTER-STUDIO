"""Rendering for FCTX MATTER STUDIO.

The renderer is a reader.  It takes the solver's particle buffers and the
tracker's hand poses and turns them into pixels; it never writes physics
state.  Import this package only when you actually intend to open a GL
context -- it pulls in ``moderngl`` and ``glfw``.
"""

from __future__ import annotations

from .buffers import InteropError, ParticleBuffers
from .camera import OrbitCamera, look_at, mat_bytes, orthographic, perspective
from .context import FrameSource, GLError, InputEvent, Window, check_gl_error
from .geometry import CapsuleShell, Mesh, capsule_shell, grid_plane, uv_sphere
from .hud import CONTROL_HINT, Hud, HudTheme
from .pipeline import BLOOM_LEVELS, Lighting, Renderer
from .shaders import SHADER_DIR, ShaderError, ShaderLibrary
from .text import OverlayBatch, TextAtlas

__all__ = [
    "BLOOM_LEVELS",
    "CONTROL_HINT",
    "CapsuleShell",
    "FrameSource",
    "GLError",
    "Hud",
    "HudTheme",
    "InputEvent",
    "InteropError",
    "Lighting",
    "Mesh",
    "OrbitCamera",
    "OverlayBatch",
    "ParticleBuffers",
    "Renderer",
    "SHADER_DIR",
    "ShaderError",
    "ShaderLibrary",
    "TextAtlas",
    "Window",
    "capsule_shell",
    "check_gl_error",
    "grid_plane",
    "look_at",
    "mat_bytes",
    "orthographic",
    "perspective",
    "uv_sphere",
]
