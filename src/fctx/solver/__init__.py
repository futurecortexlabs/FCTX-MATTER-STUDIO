"""GPU XPBD physics: the only module that writes simulation state.

    state  = SolverState(bodies, cfg, device="cuda:0")
    state.upload_materials([evaluate(CLOTH_MATERIAL, 0.3)])
    solver = XPBDSolver(state, cfg)
    solver.set_hands(poses, dt)
    solver.step(dt)
    solver.compute_normals()

Importing this pulls in ``warp``; ``fctx.bodies`` and ``fctx.core`` deliberately
do not, so the geometry and material tests still run on a machine with no GPU.
"""

from __future__ import annotations

from . import kernels
from .solver import XPBDSolver
from .state import SolverState

__all__ = ["SolverState", "XPBDSolver", "kernels"]
