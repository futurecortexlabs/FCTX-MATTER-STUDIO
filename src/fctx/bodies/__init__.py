"""Geometry generation: the only source of :class:`~fctx.core.types.BodyData`.

Pure numpy by contract -- no ``warp``, no ``moderngl``, no ``mediapipe`` -- so
the whole scene can be built, inspected and tested on a machine with no GPU
and no camera.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..config import SceneConfig
from ..core.material import DEFAULT_MATERIALS, MaterialParams
from ..core.types import BodyData, MatterKind
from . import coloring
from .cloth import build_cloth
from .coloring import colour_batches, greedy_color, sort_by_color
from .granular import build_granular
from .softbody import build_soft_body

__all__ = [
    "build_scene",
    "build_samples",
    "build_cloth",
    "build_soft_body",
    "build_granular",
    "greedy_color",
    "colour_batches",
    "sort_by_color",
    "coloring",
]

_BUILDERS = {
    MatterKind.CLOTH: build_cloth,
    MatterKind.SOFT: build_soft_body,
    MatterKind.GRAIN: build_granular,
}


def build_scene(scene: SceneConfig) -> list[BodyData]:
    """Build every body ``scene`` asks for, ready to hand to the solver.

    The list exists because ``SolverState`` packs several bodies into one set
    of GPU buffers; today a scene holds exactly one.
    """
    kind = MatterKind(scene.kind)
    try:
        builder = _BUILDERS[kind]
    except KeyError:
        raise ValueError(f"no builder for matter kind {kind!r}") from None
    params: MaterialParams = DEFAULT_MATERIALS[kind]
    return [builder(scene, params)]


def build_samples(scene: SceneConfig, count: int = 2, spacing: float = 0.34
                  ) -> list[BodyData]:
    """``count`` identical bodies of ``scene.kind`` side by side along x.

    For comparing materials: every sample is the same geometry, built once
    and translated, so the only thing that can differ between them is the
    material the solver is given per body -- which is what a blind
    comparison needs.  Sample 0 is the leftmost.
    """
    if count < 1:
        raise ValueError(f"build_samples needs at least one sample, got {count}")
    template = build_scene(scene)
    if len(template) != 1:
        raise ValueError("build_samples expects a scene of exactly one body")
    base = template[0]
    out: list[BodyData] = []
    for i in range(count):
        x = (i - 0.5 * (count - 1)) * float(spacing)
        pos = np.array(base.positions, dtype=np.float32, copy=True)
        pos[:, 0] += np.float32(x)
        body = dataclasses.replace(base, positions=pos, name=f"{base.name}_{i}")
        body.validate()
        out.append(body)
    return out
