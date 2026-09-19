"""FCTX MATTER STUDIO -- manipulate simulated matter with your bare hands.

    from fctx import AppConfig, preset, run
    run(preset("cloth"))

Importing this package is cheap: it pulls in numpy and nothing else.  Warp,
OpenGL and MediaPipe are only loaded when :func:`run` builds the application.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import PRESETS, AppConfig, preset
from .core.material import Material, MaterialParams, evaluate
from .core.types import BodyData, HandPose, MatterKind

__all__ = [
    "__version__",
    "AppConfig",
    "BodyData",
    "HandPose",
    "Material",
    "MaterialParams",
    "MatterKind",
    "PRESETS",
    "evaluate",
    "preset",
    "run",
]


def run(config: AppConfig | None = None) -> str | None:
    """Open the studio and run until the window is closed."""
    from .app import run as _run

    return _run(config or AppConfig())
