"""GPU physics: the claims the rest of the studio is built on.

The bodies here are built inline from numpy rather than through
``fctx.bodies``.  That is deliberate: this file has to be able to tell a
solver bug from a geometry bug, and it cannot do that if a change in the
geometry builders can move every number it measures.  One extra case at the
bottom runs the real builders through the same solver, so the two are still
proved to fit together.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace

import numpy as np
import warp as wp
from _harness import case, note, require, run

from fctx.config import (
    AppConfig,
    GrabConfig,
    SolverConfig,
    TrackingConfig,
    preset,
)
from fctx.core.material import (
    CLOTH_MATERIAL,
    GRAIN_MATERIAL,
    SOFT_MATERIAL,
    Material,
    evaluate,
)
from fctx.core.types import (
    FLAG_GRABBED,
    FLAG_SELF_COLLIDE,
    FLAG_SURFACE,
    BodyData,
    ConstraintKind,
    HandPose,
    MatterKind,
)
from fctx.solver import SolverState, XPBDSolver

DEVICE = "cuda:0"
RATE = 90.0
DT = 1.0 / RATE


# ---------------------------------------------------------------------------
# self-contained geometry
# ---------------------------------------------------------------------------


def _greedy_color(idx: np.ndarray, num_particles: int) -> np.ndarray:
    """Smallest free colour per constraint, so no colour repeats a particle.

    Bitmask per particle rather than a set per particle: Python ints are
    arbitrary precision, so `used[p] >> c & 1` is a single operation no matter
    how many colours the graph ends up needing.
    """
    used = [0] * num_particles
    colors = np.empty(idx.shape[0], np.int32)
    for k in range(idx.shape[0]):
        row = idx[k]
        mask = 0
        for p in row:
            mask |= used[int(p)]
        c = 0
        while (mask >> c) & 1:
            c += 1
        colors[k] = c
        bit = 1 << c
        for p in row:
            used[int(p)] |= bit
    return colors


def _dihedral(pos: np.ndarray, quad: np.ndarray) -> np.ndarray:
    e0 = pos[quad[:, 0]]
    e1 = pos[quad[:, 1]]
    w0 = pos[quad[:, 2]]
    w1 = pos[quad[:, 3]]
    n1 = np.cross(e0 - w0, e1 - w0)
    n2 = np.cross(e1 - w1, e0 - w1)
    l1 = np.linalg.norm(n1, axis=-1)
    l2 = np.linalg.norm(n2, axis=-1)
    d = np.einsum("ij,ij->i", n1, n2) / np.maximum(l1 * l2, 1e-30)
    return np.arccos(np.clip(d, -1.0, 1.0)).astype(np.float32)


def make_cloth(n: int, size: float = 0.6, top: float = 0.66,
               pinned: str = "top_corners", areal_density: float = 0.16,
               self_collide: bool = False) -> BodyData:
    """A flat n x n sheet hanging in the XY plane, row 0 at the top."""
    spacing = size / (n - 1)
    cols = np.arange(n, dtype=np.float64)
    rows = np.arange(n, dtype=np.float64)
    gx, gy = np.meshgrid(cols, rows, indexing="xy")
    pos = np.stack([
        (gx.ravel() - (n - 1) * 0.5) * spacing,
        top - gy.ravel() * spacing,
        np.zeros(n * n),
    ], axis=1).astype(np.float32)

    def vid(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (r * n + c).astype(np.int32)

    r, c = np.meshgrid(np.arange(n), np.arange(n - 1), indexing="ij")
    horiz = np.stack([vid(r.ravel(), c.ravel()), vid(r.ravel(), c.ravel() + 1)], 1)
    r, c = np.meshgrid(np.arange(n - 1), np.arange(n), indexing="ij")
    vert = np.stack([vid(r.ravel(), c.ravel()), vid(r.ravel() + 1, c.ravel())], 1)
    r, c = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    r, c = r.ravel(), c.ravel()
    diag_a = np.stack([vid(r, c), vid(r + 1, c + 1)], 1)
    diag_b = np.stack([vid(r + 1, c), vid(r, c + 1)], 1)

    dist_idx = np.concatenate([horiz, vert, diag_a, diag_b]).astype(np.int32)
    dist_kind = np.concatenate([
        np.full(len(horiz) + len(vert), int(ConstraintKind.STRETCH), np.int32),
        np.full(len(diag_a) + len(diag_b), int(ConstraintKind.SHEAR), np.int32),
    ])
    d = pos[dist_idx[:, 0]] - pos[dist_idx[:, 1]]
    dist_rest = np.linalg.norm(d, axis=1).astype(np.float32)

    # Triangles: every cell splits along its (r,c)->(r+1,c+1) diagonal.
    tri = np.concatenate([
        np.stack([vid(r, c), vid(r + 1, c), vid(r + 1, c + 1)], 1),
        np.stack([vid(r, c), vid(r + 1, c + 1), vid(r, c + 1)], 1),
    ]).astype(np.int32)

    # Hinges: the cell diagonal, plus the grid edges shared by two cells.
    bend = [np.stack([vid(r, c), vid(r + 1, c + 1), vid(r + 1, c), vid(r, c + 1)], 1)]
    rr, cc = np.meshgrid(np.arange(n - 1), np.arange(n - 2), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    bend.append(np.stack([vid(rr, cc + 1), vid(rr + 1, cc + 1),
                          vid(rr, cc), vid(rr + 1, cc + 2)], 1))
    rr, cc = np.meshgrid(np.arange(n - 2), np.arange(n - 1), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    bend.append(np.stack([vid(rr + 1, cc), vid(rr + 1, cc + 1),
                          vid(rr, cc), vid(rr + 2, cc + 1)], 1))
    bend_idx = np.concatenate(bend).astype(np.int32)
    bend_rest = _dihedral(pos.astype(np.float64), bend_idx)

    cell_area = spacing * spacing
    share = np.full(n, 1.0)
    share[0] = share[-1] = 0.5
    mass = (np.outer(share, share).ravel() * cell_area * areal_density)
    inv_mass = (1.0 / mass).astype(np.float32)

    flags = np.full(n * n, FLAG_SURFACE, np.uint32)
    if self_collide:
        flags |= np.uint32(FLAG_SELF_COLLIDE)
    grid_flags = flags.reshape(n, n)
    inv_grid = inv_mass.reshape(n, n)
    if pinned == "top_corners":
        pins = [(0, 0), (0, n - 1)]
    elif pinned == "top_edge":
        pins = [(0, j) for j in range(n)]
    elif pinned == "none":
        pins = []
    else:
        raise ValueError(f"unknown pinning {pinned!r}")
    for i, j in pins:
        inv_grid[i, j] = 0.0
        grid_flags[i, j] |= np.uint32(1)

    uv = np.stack([gx.ravel() / (n - 1), gy.ravel() / (n - 1)], 1).astype(np.float32)

    return BodyData(
        kind=MatterKind.CLOTH, name=f"cloth{n}",
        positions=np.ascontiguousarray(pos),
        inv_mass=np.ascontiguousarray(inv_mass),
        flags=np.ascontiguousarray(flags),
        dist_idx=np.ascontiguousarray(dist_idx),
        dist_rest=np.ascontiguousarray(dist_rest),
        dist_kind=np.ascontiguousarray(dist_kind),
        dist_color=np.ascontiguousarray(_greedy_color(dist_idx, n * n)),
        bend_idx=np.ascontiguousarray(bend_idx),
        bend_rest=np.ascontiguousarray(bend_rest),
        bend_color=np.ascontiguousarray(_greedy_color(bend_idx, n * n)),
        tet_idx=np.zeros((0, 4), np.int32),
        tet_dm_inv=np.zeros((0, 3, 3), np.float32),
        tet_rest_volume=np.zeros(0, np.float32),
        tet_color=np.zeros(0, np.int32),
        tri_idx=np.ascontiguousarray(tri),
        uv=np.ascontiguousarray(uv),
        double_sided=True,
        particle_radius=float(spacing * 0.5),
    )


_SPLIT_EVEN = (
    (0b000, 0b100, 0b010, 0b001),
    (0b110, 0b100, 0b010, 0b111),
    (0b101, 0b100, 0b001, 0b111),
    (0b011, 0b010, 0b001, 0b111),
    (0b100, 0b010, 0b001, 0b111),
)
_SPLIT_ODD = (
    (0b100, 0b000, 0b110, 0b101),
    (0b010, 0b000, 0b110, 0b011),
    (0b001, 0b000, 0b101, 0b011),
    (0b111, 0b110, 0b101, 0b011),
    (0b000, 0b110, 0b101, 0b011),
)


def make_tet_cube(cells: int, size: float = 0.22, height: float = 0.30,
                  density: float = 520.0) -> BodyData:
    """A cube voxelised into ``cells``^3 boxes, each split into five tets."""
    lat = cells + 1
    step = size / cells
    axis = (np.arange(lat) - cells * 0.5) * step
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel() + height, gz.ravel()], 1).astype(np.float32)
    num = lat ** 3

    def vid(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return (i * lat * lat + j * lat + k).astype(np.int32)

    ci, cj, ck = np.meshgrid(np.arange(cells), np.arange(cells),
                             np.arange(cells), indexing="ij")
    ci, cj, ck = ci.ravel(), cj.ravel(), ck.ravel()
    parity = (ci + cj + ck) % 2

    tets = np.empty((ci.size * 5, 4), np.int32)
    for slot in range(5):
        for corner in range(4):
            even = _SPLIT_EVEN[slot][corner]
            odd = _SPLIT_ODD[slot][corner]
            code = np.where(parity == 0, even, odd)
            di = (code >> 2) & 1
            dj = (code >> 1) & 1
            dk = code & 1
            tets[slot::5, corner] = vid(ci + di, cj + dj, ck + dk)

    p = pos.astype(np.float64)
    dm = np.stack([p[tets[:, 1]] - p[tets[:, 0]],
                   p[tets[:, 2]] - p[tets[:, 0]],
                   p[tets[:, 3]] - p[tets[:, 0]]], axis=2)
    det = np.linalg.det(dm)
    # A parity-flipped split produces both windings; swapping two vertices is
    # the cheapest way to make every element positively oriented, which the
    # Neo-Hookean constraint requires to tell inversion from rest.
    flip = det < 0.0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    dm = np.stack([p[tets[:, 1]] - p[tets[:, 0]],
                   p[tets[:, 2]] - p[tets[:, 0]],
                   p[tets[:, 3]] - p[tets[:, 0]]], axis=2)
    det = np.linalg.det(dm)
    if (det <= 0.0).any():
        raise AssertionError("tet cube factory produced a degenerate element")
    dm_inv = np.linalg.inv(dm).astype(np.float32)
    volume = (det / 6.0).astype(np.float32)

    edges = np.concatenate([tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
                            tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)
    rest = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1).astype(np.float32)

    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]],
                            tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]])
    keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    boundary = set(map(tuple, keys[counts == 1].tolist()))
    tri = np.array([f for f in faces.tolist()
                    if tuple(sorted(f)) in boundary], np.int32)

    total_volume = float(det.sum() / 6.0)
    mass = density * total_volume / num
    inv_mass = np.full(num, 1.0 / mass, np.float32)
    flags = np.zeros(num, np.uint32)
    flags[np.unique(tri)] = FLAG_SURFACE

    uv = np.zeros((num, 2), np.float32)
    return BodyData(
        kind=MatterKind.SOFT, name=f"cube{cells}",
        positions=np.ascontiguousarray(pos),
        inv_mass=np.ascontiguousarray(inv_mass),
        flags=np.ascontiguousarray(flags),
        dist_idx=np.ascontiguousarray(edges),
        dist_rest=np.ascontiguousarray(rest),
        dist_kind=np.full(edges.shape[0], int(ConstraintKind.STRETCH), np.int32),
        dist_color=np.ascontiguousarray(_greedy_color(edges, num)),
        bend_idx=np.zeros((0, 4), np.int32),
        bend_rest=np.zeros(0, np.float32),
        bend_color=np.zeros(0, np.int32),
        tet_idx=np.ascontiguousarray(tets),
        tet_dm_inv=np.ascontiguousarray(dm_inv),
        tet_rest_volume=np.ascontiguousarray(volume),
        tet_color=np.ascontiguousarray(_greedy_color(tets, num)),
        tri_idx=np.ascontiguousarray(tri),
        uv=np.ascontiguousarray(uv),
        particle_radius=float(step * 0.5),
    )


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------


def config(**solver_kwargs: object) -> AppConfig:
    base = SolverConfig(rate_hz=RATE, self_collision=False)
    return AppConfig(
        solver=replace(base, **solver_kwargs),  # type: ignore[arg-type]
        grab=GrabConfig(),
        tracking=TrackingConfig(source="synthetic", max_hands=2),
        device=DEVICE,
    )


def build(bodies: list[BodyData], cfg: AppConfig,
          materials: list[Material]) -> tuple[SolverState, XPBDSolver]:
    state = SolverState(bodies, cfg, DEVICE)
    state.upload_materials(materials)
    return state, XPBDSolver(state, cfg)


def simulate(solver: XPBDSolver, seconds: float, dt: float = DT) -> int:
    steps = int(round(seconds / dt))
    for _ in range(steps):
        solver.step(dt)
    wp.synchronize_device(DEVICE)
    return steps


def tet_volumes(state: SolverState, body: BodyData) -> np.ndarray:
    x = state.x.numpy().astype(np.float64)
    t = body.tet_idx
    d = np.stack([x[t[:, 1]] - x[t[:, 0]],
                  x[t[:, 2]] - x[t[:, 0]],
                  x[t[:, 3]] - x[t[:, 0]]], axis=2)
    return np.linalg.det(d) / 6.0


def finite(state: SolverState) -> None:
    snap = state.snapshot()
    for key in ("x", "v"):
        require(np.isfinite(snap[key]).all(),
                f"{key} went non-finite ({int((~np.isfinite(snap[key])).sum())} values)")


def step_ms(solver: XPBDSolver, iters: int = 120) -> float:
    for _ in range(12):
        solver.step(DT)
    wp.synchronize_device(DEVICE)
    start = time.perf_counter()
    for _ in range(iters):
        solver.step(DT)
    wp.synchronize_device(DEVICE)
    return (time.perf_counter() - start) * 1000.0 / iters


# ---------------------------------------------------------------------------
# 1. a hanging sheet reaches equilibrium
# ---------------------------------------------------------------------------


@case
def pinned_cloth_settles_without_reaching_the_floor() -> None:
    cloth = make_cloth(40, pinned="top_corners")
    cfg = config()
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.30)])

    start_low = float(cloth.positions[:, 1].min())
    simulate(solver, 2.5)
    mid = state.x.numpy()[:, 1].copy()
    simulate(solver, 0.5)
    finite(state)

    x = state.x.numpy()
    drift = float(np.abs(x[:, 1] - mid).max())
    speed = float(np.linalg.norm(state.v.numpy(), axis=1).max())
    low = float(x[:, 1].min())
    ground = cfg.solver.ground_y

    note(f"lowest point {low:.4f} m (started at {start_low:.4f} m, "
         f"floor at {ground:.2f} m)")
    note(f"last 0.5 s: max vertical drift {drift * 1000:.2f} mm, "
         f"max speed {speed:.4f} m/s")

    require(low < start_low - 1e-3, "the sheet did not fall at all")
    require(low > ground - 5e-3, f"the sheet fell through the floor to {low:.4f}")
    require(drift < 5e-3, f"still moving after 3 s: {drift * 1000:.2f} mm drift")
    require(speed < 0.12, f"still moving after 3 s: {speed:.4f} m/s")


# ---------------------------------------------------------------------------
# 2. both ends of the dial survive
# ---------------------------------------------------------------------------


@case
def both_hardness_extremes_are_stable() -> None:
    for hardness in (0.0, 1.0):
        cloth = make_cloth(36, pinned="top_corners")
        cfg = config()
        state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, hardness)])
        simulate(solver, 3.0)
        finite(state)
        x = state.x.numpy()
        speed = float(np.linalg.norm(state.v.numpy(), axis=1).max())
        span = float(np.abs(x).max())
        note(f"cloth hardness {hardness:.1f}: max |x| {span:.4f} m, "
             f"max speed {speed:.4f} m/s")
        require(span < 5.0, f"cloth at hardness {hardness} left the stage")
        require(speed <= cfg.solver.max_velocity + 1e-3,
                f"velocity clamp leaked at hardness {hardness}: {speed:.3f} m/s")

    for hardness in (0.0, 1.0):
        cube = make_tet_cube(6, size=0.18, height=0.26)
        cfg = config()
        state, solver = build([cube], cfg, [evaluate(SOFT_MATERIAL, hardness)])
        simulate(solver, 3.0)
        finite(state)
        vols = tet_volumes(state, cube)
        speed = float(np.linalg.norm(state.v.numpy(), axis=1).max())
        note(f"soft  hardness {hardness:.1f}: min tet volume "
             f"{vols.min() / cube.tet_rest_volume.min():.3f} x rest, "
             f"max speed {speed:.4f} m/s")
        require(vols.min() > 0.0,
                f"a tetrahedron inverted at hardness {hardness}")


# ---------------------------------------------------------------------------
# 3. a dropped soft body keeps its volume and its orientation
# ---------------------------------------------------------------------------


@case
def dropped_soft_body_conserves_volume() -> None:
    cube = make_tet_cube(7, size=0.20, height=0.34)
    cfg = config()
    state, solver = build([cube], cfg, [evaluate(SOFT_MATERIAL, 0.45)])

    rest_volume = float(cube.tet_rest_volume.sum())
    simulate(solver, 2.5)
    finite(state)

    vols = tet_volumes(state, cube)
    total = float(vols.sum())
    error = abs(total - rest_volume) / rest_volume
    low = float(state.x.numpy()[:, 1].min())

    note(f"volume {total * 1e6:.2f} cm^3 vs rest {rest_volume * 1e6:.2f} cm^3 "
         f"-> {error * 100:.2f}% change")
    note(f"inverted tetrahedra: {int((vols <= 0.0).sum())} of {vols.size}; "
         f"lowest particle {low:.4f} m")

    require(error < 0.12, f"volume moved {error * 100:.2f}%, budget is 12%")
    require((vols > 0.0).all(),
            f"{int((vols <= 0.0).sum())} tetrahedra inverted")
    require(low > cfg.solver.ground_y - 5e-3, "the cube sank through the floor")


# ---------------------------------------------------------------------------
# 4. the CUDA graph is not a different simulation
# ---------------------------------------------------------------------------


@case
def graph_and_plain_launch_agree() -> None:
    cloth = make_cloth(28, pinned="top_corners", self_collide=True)
    cfg_graph = config(use_cuda_graph=True, self_collision=True)
    cfg_plain = config(use_cuda_graph=False, self_collision=True)

    state = SolverState([cloth], cfg_graph, DEVICE)
    state.upload_materials([evaluate(CLOTH_MATERIAL, 0.35)])
    initial = state.snapshot()

    graph_solver = XPBDSolver(state, cfg_graph)
    require(graph_solver.use_graph, "the graph path did not engage on a CUDA device")
    for _ in range(60):
        graph_solver.step(DT)
    wp.synchronize_device(DEVICE)
    after_graph = state.snapshot()

    state.restore(initial)
    plain_solver = XPBDSolver(state, cfg_plain)
    require(not plain_solver.use_graph, "the plain path still captured a graph")
    for _ in range(60):
        plain_solver.step(DT)
    wp.synchronize_device(DEVICE)
    after_plain = state.snapshot()

    dx = np.abs(after_graph["x"] - after_plain["x"]).max()
    dv = np.abs(after_graph["v"] - after_plain["v"]).max()
    note(f"after 60 steps: max |dx| {dx:.3e} m, max |dv| {dv:.3e} m/s")
    require(np.allclose(after_graph["x"], after_plain["x"], atol=1e-4, rtol=0.0),
            f"positions diverged by {dx:.3e}")
    require(np.allclose(after_graph["v"], after_plain["v"], atol=1e-4, rtol=0.0),
            f"velocities diverged by {dv:.3e}")

    # Repeat each path from the same bytes.  A data race inside one kernel
    # shows up here as a difference of about 1e-6 on maybe one run in five,
    # which is small enough and rare enough to be mistaken for the graph being
    # at fault; comparing a path against itself names the real culprit.
    state.restore(initial)
    for _ in range(60):
        graph_solver.step(DT)
    wp.synchronize_device(DEVICE)
    repeat_graph = state.snapshot()
    state.restore(initial)
    for _ in range(60):
        plain_solver.step(DT)
    wp.synchronize_device(DEVICE)
    repeat_plain = state.snapshot()

    for label, first, again in (("graph", after_graph, repeat_graph),
                                ("plain", after_plain, repeat_plain)):
        drift = np.abs(first["x"] - again["x"]).max()
        note(f"{label} path repeated from the same state: max |dx| {drift:.3e} m")
        require(drift == 0.0,
                f"the {label} path is not deterministic: {drift:.3e} m apart")


# ---------------------------------------------------------------------------
# 5. the showpiece: hardness really changes how far matter stretches
# ---------------------------------------------------------------------------


@case
def stiffer_matter_stretches_less_under_the_same_load() -> None:
    # A curtain pinned along its whole top edge hangs straight down, so the
    # vertical extent is rest length plus strain and nothing else -- a sagging
    # two-corner hang would mix bending into the measurement.
    deep_floor = config(ground_y=-5.0)
    measured: list[tuple[float, float]] = []
    for hardness in (0.0, 0.5, 1.0):
        # Heavy sheeting, 0.64 kg/m^2.  Cotton at 0.16 stretches under its own
        # weight by a couple of millimetres at the soft end, which is real but
        # sits close enough to the residual sway that the ordering could turn
        # on which frame the measurement lands.  Four times the load moves the
        # signal well clear of it without changing what is being measured.
        cloth = make_cloth(32, size=0.60, top=0.66, pinned="top_edge",
                           areal_density=0.64)
        state, solver = build([cloth], deep_floor,
                              [evaluate(CLOTH_MATERIAL, hardness)])
        simulate(solver, 5.0)
        finite(state)
        y = state.x.numpy()[:, 1]
        extent = float(y.max() - y.min())
        strain = (extent - 0.60) / 0.60
        measured.append((hardness, extent))
        stiffness = evaluate(CLOTH_MATERIAL, hardness).stretch_k
        note(f"hardness {hardness:.1f}: k = {stiffness:8.4g} N/m"
             f"   hangs {extent * 1000:7.2f} mm   strain {strain * 100:6.3f}%")

    for (h_soft, soft), (h_hard, hard) in zip(measured, measured[1:]):
        require(hard < soft,
                f"hardness {h_hard} hung {hard * 1000:.2f} mm, which is not less "
                f"than hardness {h_soft} at {soft * 1000:.2f} mm")
    spread = (measured[0][1] - measured[-1][1]) * 1000.0
    note(f"soft-to-hard spread: {spread:.2f} mm")
    require(spread > 1.0, f"the dial only moved the hem by {spread:.2f} mm")


# ---------------------------------------------------------------------------
# 6. internal constraints do not invent momentum
# ---------------------------------------------------------------------------


@case
def momentum_change_equals_gravity_impulse() -> None:
    # Small and centred on the origin on purpose.  Velocity is recovered as
    # (x - x_prev) / h from float32 positions, so the accuracy of every
    # measurement here is set by how many bits survive that subtraction: the
    # same sheet dropped from y = 2 m instead of y = 0 reports its own weight
    # 0.6% heavy, which is float32 and not physics.
    steps, settle = 8, 2
    cloth = make_cloth(16, size=0.10, pinned="none", top=0.0)
    cfg = config(ground_y=-50.0, air_drag=0.0, wind=(0.0, 0.0, 0.0),
                 self_collision=False)
    # Damping and drag are momentum sinks by design, so they are switched off
    # here; what is on trial is whether the constraint projections are
    # momentum-neutral, which they are only if every gradient set sums to zero.
    material = replace(evaluate(CLOTH_MATERIAL, 0.4), damping=0.0)
    state, solver = build([cloth], cfg, [material])

    mass = 1.0 / state.w.numpy()
    for _ in range(settle):
        solver.step(DT)
    wp.synchronize_device(DEVICE)
    before = (mass[:, None] * state.v.numpy()).sum(axis=0)
    for _ in range(steps):
        solver.step(DT)
    wp.synchronize_device(DEVICE)
    after = (mass[:, None] * state.v.numpy()).sum(axis=0)

    total_mass = float(mass.sum())
    expected = np.asarray(cfg.solver.gravity, np.float64) * total_mass * DT * steps
    delta = (after - before).astype(np.float64)
    error = float(np.linalg.norm(delta - expected) / np.linalg.norm(expected))

    note(f"total mass {total_mass * 1000:.3f} g over {steps} steps")
    note(f"dp measured {delta} kg m/s")
    note(f"dp expected {expected} kg m/s   relative error {error:.3e}")
    require(error < 1e-4, f"momentum error {error:.3e} is too large to be rounding")


# ---------------------------------------------------------------------------
# hands, grabs and the arrays the renderer reads
# ---------------------------------------------------------------------------


def _pose(point: np.ndarray, velocity: np.ndarray, track: int = 0,
          spread: float = 0.02) -> HandPose:
    """A stand-in hand: 21 joints strung along X through ``point``.

    ``spread`` of zero collapses the capsules onto one point, which makes the
    hand a sphere of the wrist bone's radius -- the simplest collider there
    is to reason about.
    """
    joints = np.tile(point.astype(np.float32), (21, 1))
    joints[:, 0] += np.linspace(-spread, spread, 21, dtype=np.float32)
    return HandPose(
        joints=np.ascontiguousarray(joints, np.float32),
        velocities=np.ascontiguousarray(
            np.tile(velocity.astype(np.float32), (21, 1)), np.float32),
        pinch=1.0, pinching=True,
        pinch_point=np.ascontiguousarray(point, np.float32),
        pinch_velocity=np.ascontiguousarray(velocity, np.float32),
        track_id=track,
    )


@case
def a_grab_carries_matter_and_a_release_throws_it() -> None:
    cloth = make_cloth(24, pinned="none", top=0.60)
    cfg = config(ground_y=-5.0)
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.5)])

    centre = cloth.positions.mean(axis=0)
    pose = _pose(centre, np.zeros(3, np.float32))
    solver.set_hands([pose], DT)
    held = solver.begin_grab(0, pose)
    note(f"grabbed {held} particles within {cfg.grab.radius} m of the pinch")
    require(held > 0, "the pinch caught nothing")
    require(solver.stats()["grabbed"] == held, "stats disagrees with begin_grab")

    lifted = centre.copy()
    for i in range(45):
        lifted = centre + np.array([0.0, 0.004 * (i + 1), 0.0], np.float32)
        pose = _pose(lifted, np.array([0.0, 0.36, 0.0], np.float32))
        solver.set_hands([pose], DT)
        solver.step(DT)
    wp.synchronize_device(DEVICE)
    finite(state)

    # Each particle's target is the pinch point plus the offset recorded when
    # the grab closed, not the pinch point itself -- a grab holds a patch, so
    # measuring distance to the hand would just measure the grab radius.
    x = state.x.numpy()
    slots = state.grab_particle.numpy()
    local = state.grab_local.numpy()
    live = slots >= 0
    targets = lifted[None, :] + local[live]
    error = float(np.linalg.norm(x[slots[live]] - targets, axis=1).mean())
    note(f"held particles trail their targets by {error * 1000:.2f} mm "
         f"after 45 frames of lifting at 0.36 m/s")
    require(error < 0.01, "the grab let go of the matter while the hand moved")

    held = slots[live]
    solver.end_grab(0, pose)
    wp.synchronize_device(DEVICE)
    released = state.v.numpy()[held, 1].mean()
    note(f"released particles leave at {released:.3f} m/s upward")
    require(released > 0.2, "the release did not hand over the hand's velocity")
    require(solver.stats()["grabbed"] == 0, "grab count survived the release")


@case
def pinch_hysteresis_holds_on_through_a_wobbling_pinch() -> None:
    cloth = make_cloth(20, pinned="none", top=0.50)
    cfg = config(ground_y=-5.0)
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.5)])
    centre = cloth.positions.mean(axis=0)
    g = cfg.grab

    def tick(pinch: float, now: float) -> None:
        pose = _pose(centre, np.zeros(3, np.float32))
        pose.pinch = pinch
        solver.set_hands([pose], DT)
        solver.update_grabs([pose], now)

    now = 0.0
    tick(g.start_threshold + 0.05, now)
    require(solver.stats()["grabbed"] == 0,
            "the grab committed before the hold time elapsed")
    now += g.hold_time + DT
    tick(g.start_threshold + 0.05, now)
    grabbed = solver.stats()["grabbed"]
    require(grabbed > 0, "the grab never committed")

    # Between the two thresholds is the band the hysteresis exists to cover.
    midpoint = 0.5 * (g.start_threshold + g.release_threshold)
    for i in range(30):
        now += DT
        tick(midpoint if i % 2 else g.start_threshold + 0.02, now)
        solver.step(DT)
    require(solver.stats()["grabbed"] == grabbed,
            "a pinch wobbling inside the hysteresis band dropped the cloth")

    now += DT
    tick(g.release_threshold - 0.05, now)
    require(solver.stats()["grabbed"] == 0,
            "opening the pinch past the release threshold did not let go")
    note(f"held {grabbed} particles through 30 frames of pinch between "
         f"{g.release_threshold} and {g.start_threshold}, released below "
         f"{g.release_threshold}")


@case
def a_lost_hand_drops_what_it_held_instead_of_throwing_it() -> None:
    cloth = make_cloth(20, pinned="none", top=0.50)
    cfg = config(ground_y=-5.0)
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.5)])
    centre = cloth.positions.mean(axis=0)
    pose = _pose(centre, np.zeros(3, np.float32))
    solver.set_hands([pose], DT)
    held = solver.begin_grab(0, pose)
    require(held > 0, "the pinch caught nothing")
    slots = state.grab_particle.numpy()
    idx = slots[slots >= 0]

    # The hand disappears.  Its slot's stored position is still out on the
    # stage, so a naive finite difference against an empty slot reads as a
    # jump to the origin -- tens of metres per second of phantom throw.
    solver.set_hands([], DT)
    wp.synchronize_device(DEVICE)
    speed = float(np.linalg.norm(state.v.numpy()[idx], axis=1).max())
    note(f"fastest released particle leaves at {speed:.4f} m/s")
    require(solver.stats()["grabbed"] == 0, "a vanished hand kept its grab")
    require(speed < 0.5, f"a lost hand flung the cloth at {speed:.2f} m/s")


@case
def a_hand_capsule_pushes_and_the_normals_stay_unit_length() -> None:
    cloth = make_cloth(28, pinned="top_edge")
    cfg = config()
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.4)])
    simulate(solver, 0.6)
    before = state.x.numpy()[:, 2].copy()

    centre = np.array([0.0, 0.40, -0.06], np.float32)
    for i in range(60):
        point = centre + np.array([0.0, 0.0, 0.0025 * i], np.float32)
        solver.set_hands([_pose(point, np.array([0.0, 0.0, 0.15], np.float32))], DT)
        solver.step(DT)
    wp.synchronize_device(DEVICE)
    finite(state)

    after = state.x.numpy()[:, 2]
    pushed = float((after - before).max())
    contacts = solver.stats()["contacts"]
    note(f"hand pushed the sheet {pushed * 1000:.2f} mm forward, "
         f"{contacts} particles in contact")
    require(pushed > 5e-3, "the capsule collider did not move the sheet")

    solver.compute_normals()
    wp.synchronize_device(DEVICE)
    n = state.normal.numpy()
    lengths = np.linalg.norm(n, axis=1)
    note(f"normal lengths in [{lengths.min():.6f}, {lengths.max():.6f}]")
    require(np.allclose(lengths, 1.0, atol=1e-5), "normals are not unit length")


@case
def the_renderer_can_find_each_body_in_the_shared_buffers() -> None:
    cloth = make_cloth(12, pinned="top_corners")
    cube = make_tet_cube(3, size=0.10, height=0.5)
    cfg = config()
    state, _ = build([cloth, cube], cfg,
                     [evaluate(CLOTH_MATERIAL, 0.3), evaluate(SOFT_MATERIAL, 0.6)])

    require(state.num_particles == cloth.num_particles + cube.num_particles,
            "the global particle buffer is the wrong size")
    c_start, c_end = state.body_particle_range(0)
    s_start, s_end = state.body_particle_range(1)
    require((c_start, c_end) == (0, cloth.num_particles), "cloth range is wrong")
    require((s_start, s_end) == (cloth.num_particles, state.num_particles),
            "soft body range is wrong")

    tri = state.body_triangles(1)
    require(tri.shape == cube.tri_idx.shape, "soft body triangle count changed")
    require(tri.min() >= s_start and tri.max() < s_end,
            "soft body triangles point outside its own particles")
    require(np.array_equal(tri - s_start, cube.tri_idx),
            "triangle offsetting scrambled the surface")

    body = state.body.numpy()
    require((body[:c_end] == 0).all() and (body[s_start:] == 1).all(),
            "particle ownership is wrong")
    note(f"{state.num_particles} particles, {state.num_constraints} constraints, "
         f"{state.num_tri} triangles across 2 bodies")

    # Colour batches must partition every constraint exactly once, or the
    # solver silently skips or double-counts part of the scene.
    for name, batches, total in (("distance", state.dist_batches, state.num_dist),
                                 ("bending", state.bend_batches, state.num_bend),
                                 ("tetra", state.tet_batches, state.num_tet)):
        covered = sum(c for _, c in batches)
        require(covered == total,
                f"{name} batches cover {covered} of {total} constraints")
        cursor = 0
        for offset, count in batches:
            require(offset == cursor, f"{name} batches are not contiguous")
            cursor += count


@case
def the_hardness_dial_moves_without_rebuilding_anything() -> None:
    cloth = make_cloth(20, pinned="top_corners")
    cfg = config()
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.1)])
    simulate(solver, 0.5)
    graph_before = solver._graph
    for hardness in (0.2, 0.55, 0.9, 1.0, 0.0):
        state.upload_materials([evaluate(CLOTH_MATERIAL, hardness)])
        simulate(solver, 0.25)
        finite(state)
    require(solver._graph is graph_before,
            "changing hardness invalidated the captured CUDA graph")
    note("hardness swept 0.1 -> 1.0 -> 0.0 mid-simulation on one captured graph")


@case
def a_non_finite_body_is_reset_rather_than_allowed_to_spread() -> None:
    cloth = make_cloth(14, pinned="top_corners")
    cube = make_tet_cube(3, size=0.10, height=0.5)
    cfg = config(sanity_interval=1)
    state, solver = build([cloth, cube], cfg,
                          [evaluate(CLOTH_MATERIAL, 0.3), evaluate(SOFT_MATERIAL, 0.5)])
    simulate(solver, 0.3)
    healthy = state.x.numpy()[:cloth.num_particles].copy()

    poisoned = state.x.numpy()
    poisoned[cloth.num_particles + 4] = np.nan
    state.x.assign(poisoned)
    solver.step(DT)
    wp.synchronize_device(DEVICE)

    finite(state)
    x = state.x.numpy()
    s_start, s_end = state.body_particle_range(1)
    require(np.allclose(x[s_start:s_end], cube.positions, atol=1e-6),
            "the poisoned body was not returned to its rest shape")
    moved = float(np.abs(x[:cloth.num_particles] - healthy).max())
    note(f"the neighbouring cloth moved {moved * 1000:.3f} mm across the reset")
    require(moved < 0.05, "the sanity sweep reset a body that was fine")


@case
def the_real_geometry_builders_run_through_this_solver() -> None:
    try:
        from fctx.bodies import build_scene
    except Exception as exc:  # noqa: BLE001
        note(f"fctx.bodies is not importable yet ({exc.__class__.__name__}); skipped")
        return

    cfg = config(self_collision=True)
    scene = replace(cfg.scene, cloth_resolution=40, cloth_pinned="top_corners",
                    hardness=0.35)
    bodies = build_scene(scene)
    state, solver = build(bodies, cfg,
                          [evaluate(CLOTH_MATERIAL, scene.hardness) for _ in bodies])
    simulate(solver, 1.5)
    finite(state)
    solver.compute_normals()
    wp.synchronize_device(DEVICE)
    lengths = np.linalg.norm(state.normal.numpy(), axis=1)
    y = state.x.numpy()[:, 1]
    note(f"fctx.bodies cloth: {state.num_particles} particles, "
         f"{state.num_constraints} constraints, hem at {y.min():.4f} m")
    require(np.allclose(lengths, 1.0, atol=1e-5), "normals are not unit length")
    require(y.min() > cfg.solver.ground_y - 5e-3, "the built cloth fell through")


def _three_particles(gap: float, collide: np.ndarray,
                     radius: float = 0.01) -> BodyData:
    """Three unconstrained particles in a row, ``gap`` apart along X."""
    pos = np.zeros((3, 3), np.float32)
    pos[:, 0] = (np.arange(3) - 1.0) * gap
    flags = np.where(collide, np.uint32(FLAG_SELF_COLLIDE), np.uint32(0))
    return BodyData(
        kind=MatterKind.GRAIN, name="triple",
        positions=np.ascontiguousarray(pos),
        inv_mass=np.ones(3, np.float32),
        flags=np.ascontiguousarray(flags, np.uint32),
        dist_idx=np.zeros((0, 2), np.int32),
        dist_rest=np.zeros(0, np.float32),
        dist_kind=np.zeros(0, np.int32),
        dist_color=np.zeros(0, np.int32),
        bend_idx=np.zeros((0, 4), np.int32),
        bend_rest=np.zeros(0, np.float32),
        bend_color=np.zeros(0, np.int32),
        tet_idx=np.zeros((0, 4), np.int32),
        tet_dm_inv=np.zeros((0, 3, 3), np.float32),
        tet_rest_volume=np.zeros(0, np.float32),
        tet_color=np.zeros(0, np.int32),
        tri_idx=np.zeros((0, 3), np.int32),
        uv=np.zeros((3, 2), np.float32),
        particle_radius=radius,
    )


def kinetic_energy(state: SolverState) -> float:
    w = state.w_rest.numpy()
    mass = np.where(w > 0.0, 1.0 / np.maximum(w, 1e-30), 0.0)
    v = state.v.numpy()
    return float(0.5 * (mass * np.einsum("ij,ij->i", v, v)).sum())


@case
def a_diverging_body_does_not_take_the_cuda_context_with_it() -> None:
    # `wp.HashGrid.build` turns each point into an integer cell index with no
    # range check.  An infinity, or any coordinate past a few million metres
    # at these cell sizes, makes it write outside its own table.  On CUDA that
    # is an illegal memory access: not an exception the app can catch, but the
    # death of the context -- every later allocation in the process fails.
    # The periodic sanity sweep cannot be the guard, because it runs after the
    # step and the grid build is the first thing in the next one.
    cloth = make_cloth(16, pinned="top_corners", self_collide=True)
    cube = make_tet_cube(3, size=0.10, height=0.5)
    cfg = config(self_collision=True, sanity_interval=1)
    state, solver = build([cloth, cube], cfg,
                          [evaluate(CLOTH_MATERIAL, 0.3), evaluate(SOFT_MATERIAL, 0.5)])
    simulate(solver, 0.2)
    healthy = state.x.numpy()[:cloth.num_particles].copy()

    for poison in (np.inf, -np.inf, np.nan, 1.0e30):
        x = state.x.numpy()
        x[cloth.num_particles + 2] = poison
        state.x.assign(x)
        # Two steps: the first survives the build, flags the body and resets
        # it, the second proves the device is still usable afterwards.
        solver.step(DT)
        solver.step(DT)
        wp.synchronize_device(DEVICE)
        finite(state)
        # An allocation is what fails first when a context has died.
        probe = wp.zeros(4096, dtype=wp.vec3, device=DEVICE)
        wp.synchronize_device(DEVICE)
        require(int(probe.shape[0]) == 4096, "the device stopped allocating")

    s_start, s_end = state.body_particle_range(1)
    span = state.x.numpy()[s_start:s_end]
    size = float((span.max(axis=0) - span.min(axis=0)).max())
    rest_size = float((cube.positions.max(axis=0) - cube.positions.min(axis=0)).max())
    moved = float(np.abs(state.x.numpy()[:cloth.num_particles] - healthy).max())
    note(f"survived inf, -inf, nan and 1e30 with the hash grid live; "
         f"{solver.stats()['resets']} body resets, the poisoned cube is "
         f"{size * 100:.2f} cm across against {rest_size * 100:.2f} cm at rest, "
         f"neighbouring cloth moved {moved * 1000:.2f} mm")
    require(solver.stats()["resets"] == 4,
            f"four poisons produced {solver.stats()['resets']} body resets")
    require(abs(size - rest_size) < 0.02,
            f"the poisoned body came back the wrong size: {size:.4f} m")
    require(moved < 0.05, "the resets disturbed the body next door")

    simulate(solver, 0.5)
    finite(state)


@case
def a_soft_body_stays_on_the_stage_across_the_whole_dial() -> None:
    # A 6^3 cube of 0.18 m is too coarse to catch this: the element is 3 cm and
    # the mass per node high enough that the compliance still dominates.  At
    # the size and resolution the `cube` preset actually uses, the Neo-Hookean
    # pair used to reach the velocity clamp from hardness 0.4 upwards and leave
    # the stage entirely -- 25 m up at hardness 1.0.
    start_top = 0.34
    for hardness in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        cube = make_tet_cube(12, size=0.26, height=start_top)
        cfg = config()
        state, solver = build([cube], cfg, [evaluate(SOFT_MATERIAL, hardness)])
        simulate(solver, 3.0)
        finite(state)
        x = state.x.numpy()
        vols = tet_volumes(state, cube)
        ke = kinetic_energy(state)
        speed = float(np.linalg.norm(state.v.numpy(), axis=1).max())
        volume_error = abs(vols.sum() / float(cube.tet_rest_volume.sum()) - 1.0)
        note(f"hardness {hardness:.2f}: settles at y in "
             f"[{x[:, 1].min():.3f}, {x[:, 1].max():.3f}] m, {speed:.3f} m/s, "
             f"{ke * 1e3:.3f} mJ, volume {volume_error * 100:.2f}% off, "
             f"{int((vols <= 0.0).sum())} inverted")
        require(x[:, 1].max() < start_top + 0.20,
                f"hardness {hardness} launched the body to "
                f"{x[:, 1].max():.2f} m from {start_top:.2f} m")
        require(np.abs(x[:, [0, 2]]).max() < 0.5,
                f"hardness {hardness} threw the body sideways")
        require(ke < 1.0e-2,
                f"hardness {hardness} is still carrying {ke * 1e3:.1f} mJ "
                f"after three seconds of settling")
        require((vols > 0.0).all(),
                f"{int((vols <= 0.0).sum())} tetrahedra inverted at {hardness}")
        require(volume_error < 0.12,
                f"volume moved {volume_error * 100:.1f}% at hardness {hardness}")


@case
def a_stiffer_soft_body_sags_less_and_says_where_it_stops_being_literal() -> None:
    # The cloth half of this is proved above.  A soft body has to be checked
    # separately because past about E = 30 kPa at this element size the
    # Neo-Hookean pair is no longer what carries the load: the solver scales
    # the element down to a stiffness one XPBD pass can resolve and the tet
    # edge distance constraints take over.  The dial has to stay monotonic
    # across that handover, and the handover point has to be reportable rather
    # than a surprise.
    hung: list[tuple[float, float]] = []
    ceiling = 0.0
    for hardness in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        cube = make_tet_cube(8, size=0.22, height=0.50, density=2600.0)
        top = float(cube.positions[:, 1].max())
        pinned = cube.positions[:, 1] > top - 1e-6
        inv_mass = cube.inv_mass.copy()
        flags = cube.flags.copy()
        inv_mass[pinned] = 0.0
        flags[pinned] |= np.uint32(1)
        body = replace(cube, inv_mass=np.ascontiguousarray(inv_mass),
                       flags=np.ascontiguousarray(flags))
        cfg = config(ground_y=-5.0)
        material = evaluate(SOFT_MATERIAL, hardness)
        state, solver = build([body], cfg, [material])
        simulate(solver, 4.0)
        finite(state)
        y = state.x.numpy()[:, 1]
        extent = float(y.max() - y.min())
        hung.append((hardness, extent))

        ceil = state.tet_stiffness_ceiling(DT)
        ceiling = float(ceil[0, 0])
        carried = min(1.0, ceil[0, 0] / material.lame_mu,
                      ceil[0, 1] / material.lame_lambda)
        note(f"hardness {hardness:.1f}: E = {material.young:9.4g} Pa, "
             f"tetrahedra carry {carried * 100:6.2f}% of it, "
             f"hangs {extent * 1000:7.3f} mm, "
             f"strain {(extent - 0.22) / 0.22 * 100:6.3f}%")

    note(f"the tetrahedra of this body top out at mu = {ceiling:.4g} Pa at "
         f"{DT * 1000:.2f} ms per frame; above that the dial is carried by the "
         f"tet edge constraints")
    # Past the tetrahedra's stiffness ceiling the hang has converged to within
    # a tenth of a millimetre and the remaining difference between two dial
    # positions is solver noise, not material.  50 um is the width of that
    # noise band, measured; it is a hundredth of the 24 mm the dial actually
    # moves, so it cannot hide a real inversion of the claim.
    for (h_soft, soft), (h_hard, hard) in zip(hung, hung[1:]):
        require(hard <= soft + 5e-5,
                f"hardness {h_hard} hung {hard * 1000:.3f} mm, which is more "
                f"than hardness {h_soft} at {soft * 1000:.3f} mm")
    require(hung[0][1] - hung[-1][1] > 0.02,
            f"the dial only moved the hang by "
            f"{(hung[0][1] - hung[-1][1]) * 1000:.2f} mm")
    require(np.isfinite(ceiling) and ceiling > 0.0,
            "tet_stiffness_ceiling did not report a usable number")


@case
def self_collision_does_not_heat_matter_that_is_merely_bonded() -> None:
    # Two failures met here.  A particle's contact sphere sized to just touch
    # its bonded neighbour at rest puts every bond into standing contact the
    # moment the body compresses under its own weight, and those contacts then
    # fight the constraints that own the bond.  And a flag test that lets one
    # flagged particle pull an unflagged one into contact drags a soft body's
    # whole interior in behind its surface layer.
    # Three free particles in a line, all overlapping, with only the outer two
    # opted in.  The pair that both opted in must separate; the pairs where
    # only one did must not move at all.
    radius = 0.01
    gap = 0.0105   # a shallow overlap: the pair is just inside contact range
    probe = _three_particles(gap, np.array([True, False, True]), radius)
    cfg = config(self_collision=True, ground_y=-9.0, air_drag=0.0)
    state, solver = build([probe], cfg, [replace(
        evaluate(SOFT_MATERIAL, 0.0), damping=0.0)])
    solver.step(DT)
    wp.synchronize_device(DEVICE)
    x = state.x.numpy()
    spread = float(x[2, 0] - x[0, 0])
    middle = float(abs(x[1, 0] - probe.positions[1, 0]))
    target = 2.0 * radius * (1.0 + cfg.solver.collision_margin)
    note(f"opted-in pair moved from {2 * gap * 1000:.2f} mm to "
         f"{spread * 1000:.2f} mm apart, asking for {target * 1000:.2f} mm; "
         f"the opted-out particle between them moved {middle * 1e6:.3f} um")
    require(spread > 2.0 * gap + 1e-5, "two flagged particles did not separate")
    require(spread < 2.0 * target,
            f"the contact overshot to {spread * 1000:.1f} mm for a "
            f"{target * 1000:.1f} mm target")
    require(middle < 1e-7,
            "an unflagged particle was pulled into contact by a flagged one")

    # A settled sheet is the other half: no bond may be reported as a contact.
    cloth = make_cloth(24, pinned="top_corners", self_collide=True)
    cfg = config(self_collision=True)
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.3)])
    simulate(solver, 3.0)
    contacts = solver.stats()["contacts"]
    note(f"a settled 24x24 sheet reports {contacts} self-contacts "
         f"(every structural bond would be {cloth.num_particles})")
    require(contacts < cloth.num_particles // 4,
            f"{contacts} of {cloth.num_particles} particles are in standing "
            f"self-contact with their own neighbours")


@case
def a_hand_that_vanishes_does_not_leave_its_pinch_timer_running() -> None:
    cloth = make_cloth(20, pinned="none", top=0.50)
    cfg = config(ground_y=-5.0)
    _, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.5)])
    centre = cloth.positions.mean(axis=0)
    g = cfg.grab

    pose = _pose(centre, np.zeros(3, np.float32))
    pose.pinch = g.start_threshold + 0.05
    solver.set_hands([pose], DT)
    solver.update_grabs([pose], 0.0)
    require(solver.stats()["grabbed"] == 0, "the grab committed immediately")

    # The hand disappears for a second with the pinch still closed, then a new
    # hand lands in that slot.  If the timer survived the gap the new hand
    # grabs on its first frame, which is the hold time not existing at all.
    solver.set_hands([], DT)
    solver.update_grabs([], 1.0)
    solver.set_hands([pose], DT)
    solver.update_grabs([pose], 1.0)
    require(solver.stats()["grabbed"] == 0,
            "a slot's pinch timer survived the hand that started it")
    solver.update_grabs([pose], 1.0 + g.hold_time + DT)
    require(solver.stats()["grabbed"] > 0, "the new hand can no longer grab")
    note(f"a hand lost for 1.0 s with the pinch closed still waited "
         f"{g.hold_time * 1000:.0f} ms after coming back")


@case
def a_lost_hand_lets_go_of_its_own_grab_and_leaves_the_other_alone() -> None:
    """``HandTracker`` compacts its list; the solver must key on the track id.

    Two hands, the left one holding the cloth.  The left one leaves, so the
    tracker returns a one-element list holding the *right* hand, whose track
    id is still 1.  Keying the per-hand arrays by list position instead put
    the right hand's pinch point into slot 0 -- the slot that was holding
    the cloth -- and differenced its capsules against the left hand's, a
    full velocity-clamp friction impulse out of a hand that never moved.
    ``interaction.GripManager`` keys on the track id, so it then went on
    driving slot 1 forever while the solver held nothing.
    """
    cloth = make_cloth(20, pinned="none", top=0.50)
    cfg = config(ground_y=-5.0)
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.5)])
    left = cloth.positions.mean(axis=0)
    right = left + np.array([0.40, 0.0, 0.0], np.float32)

    for _ in range(4):
        solver.set_hands([_pose(left, np.zeros(3, np.float32), track=0),
                          _pose(right, np.zeros(3, np.float32), track=1)], DT)
        solver.step(DT)
    held = solver.begin_grab(0, _pose(left, np.zeros(3, np.float32), track=0))
    require(held > 0, "the pinch caught nothing")

    # The left hand leaves.  The list now holds only track 1, at index 0.
    solver.set_hands([_pose(right, np.zeros(3, np.float32), track=1)], DT)
    wp.synchronize_device(DEVICE)
    hand_pos = state.hand_pos.numpy()
    caps = state.cap_r.numpy().reshape(state.max_hands, state.bones_per_hand)
    speed = float(np.linalg.norm(state.v.numpy(), axis=1).max())
    note(f"surviving hand (track 1) drives slot 1 at {hand_pos[1]}, "
         f"slot 0 is empty (max capsule radius {caps[0].max():.4g} m), "
         f"fastest particle {speed:.4f} m/s")
    require(np.allclose(hand_pos[1], right, atol=1e-6),
            f"the surviving hand's slot 1 holds {hand_pos[1]}, not {right}")
    require(caps[0].max() == 0.0,
            "the slot the vanished hand left still has live capsules")
    require(solver.stats()["grabbed"] == 0,
            "the vanished hand's grab outlived it")
    require(speed < 0.5, f"a hand leaving flung the cloth at {speed:.2f} m/s")


@case
def releasing_a_grab_with_no_pose_drops_instead_of_throwing() -> None:
    """``GripManager`` reports a lost track as ``end_grab(slot, None)``.

    It used to dereference the pose unconditionally, so the sequence "grab
    with one hand, keep the other in frame, take the grabbing hand out of
    shot" ended the process with an AttributeError.
    """
    cloth = make_cloth(20, pinned="none", top=0.50)
    cfg = config(ground_y=-5.0)
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.5)])
    centre = cloth.positions.mean(axis=0)
    pose = _pose(centre, np.array([0.0, 0.8, 0.0], np.float32))
    solver.set_hands([pose], DT)
    held = solver.begin_grab(0, pose)
    require(held > 0, "the pinch caught nothing")
    idx = state.grab_particle.numpy()
    idx = idx[idx >= 0]

    solver.end_grab(0, None)
    wp.synchronize_device(DEVICE)
    speed = float(np.linalg.norm(state.v.numpy()[idx], axis=1).max())
    note(f"released with no pose: {held} particles let go at "
         f"{speed:.4f} m/s, against a 0.8 m/s hand that is no longer trusted")
    require(solver.stats()["grabbed"] == 0, "end_grab(slot, None) kept the grab")
    require(speed < 0.2,
            f"a lost track threw the cloth at {speed:.2f} m/s")


@case
def a_hand_that_never_left_keeps_its_full_collider() -> None:
    """One hand blinking must not shrink the other hand's capsules.

    The fade is indexed per slot and zeroed when a slot empties.  While the
    slot was the list position, a continuously tracked hand changed slots
    every time its partner came and went and landed on a freshly zeroed one:
    measured at 2.4% of its nominal radius, on 10% of frames, for a hand
    that was visible in every single frame.
    """
    cloth = make_cloth(8, pinned="none", top=0.50)
    cfg = config(ground_y=-5.0)
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.5)])
    bones = state.bones_per_hand
    steady = None
    worst = 1.0
    for frame in range(90):
        poses = []
        if (frame % 30) < 15:                      # the other hand blinks
            poses.append(_pose(np.array([-0.4, 0.5, 0.0], np.float32),
                               np.zeros(3, np.float32), track=0))
        poses.append(_pose(np.array([0.4, 0.5, 0.0], np.float32),
                           np.zeros(3, np.float32), track=1))
        solver.set_hands(poses, DT)
        radius = float(state.cap_r.numpy()[bones:2 * bones].max())
        if frame == 30:                            # long since grown
            steady = radius
        if steady is not None:
            worst = min(worst, radius / steady)
    note(f"the hand that never left held {100 * worst:.1f}% of its collider "
         f"radius through six appearances of the other hand")
    require(worst > 0.99,
            f"a continuously tracked hand's collider fell to "
            f"{100 * worst:.1f}% of its radius")


@case
def the_basin_walks_a_stray_grain_home_instead_of_firing_it() -> None:
    """A grain outside the wall and below its lip is not a collision.

    It is a grain that was carried over the wall by a hand and let go, and
    its 100 mm of excursion was never earned by motion the wall watched.
    Paying that off in one substep is 100 mm / 0.93 ms, and the basin used
    to fire a released fistful out to 6.4 m -- 490 grains past a metre, held
    at the velocity clamp for two seconds.
    """
    radius = 0.006
    basin = 0.30
    pos = np.array([[0.40, radius, 0.0], [0.0, radius, 0.0]], np.float32)
    body = _three_particles(1.0, np.zeros(3, bool), radius=radius)
    body = replace(body, positions=np.ascontiguousarray(pos),
                   inv_mass=np.ones(2, np.float32),
                   flags=np.zeros(2, np.uint32),
                   uv=np.zeros((2, 2), np.float32))
    cfg = config(basin_radius=basin, basin_height=0.13, self_collision=False)
    state, solver = build([body], cfg, [evaluate(GRAIN_MATERIAL, 0.5)])

    fastest = 0.0
    for _ in range(180):
        solver.step(DT)
        fastest = max(fastest, float(np.linalg.norm(state.v.numpy(), axis=1).max()))
    x = state.x.numpy()
    home = float(np.hypot(x[0, 0], x[0, 2]))
    note(f"a grain 100 mm outside a {basin} m basin came home at "
         f"{fastest:.3f} m/s and ended at r = {home:.4f} m")
    require(fastest < 2.0,
            f"the basin fired the stray grain at {fastest:.2f} m/s")
    require(home <= basin, f"the stray grain never came home; r = {home:.3f} m")


@case
def a_moving_hand_pushes_at_its_own_speed_not_the_frame_rate() -> None:
    """Capsules arrive once a frame; the solver takes twelve substeps.

    Holding the capsule still for eleven substeps and jumping on the twelfth
    hands the matter a whole frame's hand motion inside one substep, which
    ``finalize`` reads as twelve times the hand's real speed.  Measured on
    the grain pile, the peak grain speed tracked the per-frame capsule jump
    and not the hand speed at all.
    """
    radius = 0.005
    hand_speed = 1.0
    start = np.array([-0.08, 0.30, 0.0], np.float32)
    # One free particle in the path of the sphere the collapsed hand makes,
    # far enough ahead that the collider is fully grown before it arrives:
    # this case is about the sweep, not about the fade.
    pos = np.array([[0.0, 0.30, 0.0], [0.5, 0.30, 0.0]], np.float32)
    body = _three_particles(1.0, np.zeros(3, bool), radius=radius)
    body = replace(body, positions=np.ascontiguousarray(pos),
                   inv_mass=np.ones(2, np.float32),
                   flags=np.zeros(2, np.uint32),
                   uv=np.zeros((2, 2), np.float32))
    cfg = config(ground_y=-5.0, gravity=(0.0, 0.0, 0.0), air_drag=0.0,
                 self_collision=False)
    state, solver = build([body], cfg, [evaluate(GRAIN_MATERIAL, 0.5)])

    for _ in range(20):        # let the collider finish fading in
        solver.set_hands([_pose(start, np.zeros(3, np.float32), spread=0.0)], DT)
        solver.step(DT)
    fastest = 0.0
    for frame in range(1, 25):
        point = start + np.array([hand_speed * DT * frame, 0.0, 0.0], np.float32)
        solver.set_hands([_pose(point, np.zeros(3, np.float32), spread=0.0)], DT)
        solver.step(DT)
        fastest = max(fastest, float(np.linalg.norm(state.v.numpy()[0])))
    note(f"a hand moving at {hand_speed:.1f} m/s pushed the particle to at "
         f"most {fastest:.3f} m/s (the per-frame jump over one substep would "
         f"be {hand_speed * cfg.solver.substeps:.0f} m/s)")
    require(fastest > 0.3 * hand_speed,
            f"the hand never reached the particle ({fastest:.3f} m/s)")
    require(fastest < 2.0 * hand_speed,
            f"a {hand_speed:.1f} m/s hand pushed the particle to "
            f"{fastest:.2f} m/s")


@case
def the_shipped_soft_body_settles_without_inverting_a_tetrahedron() -> None:
    """The geometry builder's rest pose has to survive contact load.

    ``make_tet_cube`` above is a uniform lattice and never sees this: the
    elements at risk are the skin tetrahedra the SDF fit flattens, and they
    all fail in the ground contact patch.  22 of the shipped sphere's 12765
    inverted permanently until the fit's backoff started measuring element
    *shape* and not only volume.

    The bound is a fraction, not zero, and that is deliberate.  A voxel mesh
    with twenty-five thousand elements settling under its own weight puts a
    handful of them momentarily inside out; what matters is that the count
    stays negligible and does not grow, because an inversion the hydrostatic
    constraint can pull back out is invisible and one that feeds itself ends
    with the body across the room.  Measured over twelve seconds: at hardness
    0 it peaks at 7 and is back to 0 by 3 s; at 0.3 it sits at 1; at 1.0 it
    creeps to 8 by 12 s with the body at rest, 0.03 m/s and 99.6% of its
    volume.  An earlier revision asserted zero, and passed only at one exact
    pair of tuning constants -- it was measuring a coincidence.
    """
    from fctx.bodies import build_scene

    cfg = config()
    limit = 0.001
    for hardness in (0.0, 0.5, 1.0):
        scene = replace(preset("soft").scene, hardness=hardness)
        bodies = [b for b in build_scene(scene) if b.num_tets]
        require(len(bodies) == 1, "the soft preset should build exactly one body")
        body = bodies[0]
        state, solver = build([body], cfg,
                              [evaluate(SOFT_MATERIAL, hardness)])
        simulate(solver, 3.0)
        finite(state)
        early = int((tet_volumes(state, body) <= 0.0).sum())
        simulate(solver, 5.0)
        finite(state)
        late = int((tet_volumes(state, body) <= 0.0).sum())
        speed = float(np.linalg.norm(state.v.numpy(), axis=1).max())
        note(f"{body.name} h={hardness:.1f}: {early} then {late} of "
             f"{body.num_tets} inverted at 3 s and 8 s, "
             f"resting at {speed:.4f} m/s")
        require(late <= max(2, int(limit * body.num_tets)),
                f"{late} of {body.num_tets} tetrahedra are inside out at "
                f"hardness {hardness}, past the {limit:.1%} bound")
        require(late <= early + max(2, int(0.5 * limit * body.num_tets)),
                f"inversions grew from {early} to {late} between 3 s and 8 s "
                f"at hardness {hardness}; a self-feeding inversion ends with "
                f"the body off the stage")
        require(speed < 0.25,
                f"the body is still moving at {speed:.3f} m/s after 8 s")


@case
def step_cost_fits_inside_a_90_hz_budget() -> None:
    budget = 1000.0 / RATE
    cloth = make_cloth(72, pinned="top_corners", self_collide=True)
    cfg = config(self_collision=True)
    _, cloth_solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.30)])
    cloth_ms = step_ms(cloth_solver)
    note(f"72x72 cloth   {cloth.num_particles:6d} particles, "
         f"{cloth.num_dist + cloth.num_bend:6d} constraints, self-collision on: "
         f"{cloth_ms:6.3f} ms/step  ({budget / cloth_ms:.1f}x the 90 Hz budget)")

    cube = make_tet_cube(13, size=0.26, height=0.34)
    _, cube_solver = build([cube], config(), [evaluate(SOFT_MATERIAL, 0.40)])
    cube_ms = step_ms(cube_solver)
    note(f"13^3 soft body {cube.num_particles:6d} particles, "
         f"{cube.num_tets:6d} tetrahedra: "
         f"{cube_ms:6.3f} ms/step  ({budget / cube_ms:.1f}x the 90 Hz budget)")

    # The printed milliseconds are the real result here.  The assertion is a
    # regression guard set well above the budget on purpose: this machine may
    # be running another GPU test suite at the same time, and a shared device
    # turns a tight threshold into a coin flip rather than a measurement.
    ceiling = 3.0 * budget
    require(cloth_ms < ceiling,
            f"the cloth step costs {cloth_ms:.2f} ms, over the {ceiling:.1f} ms guard")
    require(cube_ms < ceiling,
            f"the soft step costs {cube_ms:.2f} ms, over the {ceiling:.1f} ms guard")


@case
def the_grip_manager_and_the_solver_agree_about_slots() -> None:
    """A hand that keeps holding must keep holding when its partner leaves.

    The solver and ``GripManager`` both decide which slot a pose occupies, and
    a grab is a conversation about a slot number.  Two plausible rules exist --
    the pose's track id, or its position in the list -- and they agree right up
    until a lower-numbered hand leaves, because ``HandTracker`` compacts its
    output.  Disagreeing there left the solver holding nothing, the HUD
    reporting hundreds of particles, and that hand unable to grab again for
    the rest of the session.  Both sides were independently "fixed" to
    opposite rules once, which is exactly why this is a test and not a comment.
    """
    from fctx.interaction import GripManager

    cloth = make_cloth(48, pinned="top_corners", self_collide=False)
    cfg = config()   # the shared helper already sets max_hands = 2
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.30)])
    grips = GripManager(cfg.grab, 2)

    def hand(track_id: int, cx: float, pinch: float) -> HandPose:
        j = np.zeros((21, 3), np.float32)
        for i in range(21):
            j[i] = [cx + (i % 5 - 2) * 0.012, 0.30 + (i // 5) * 0.02, 0.0]
        return HandPose(joints=j, velocities=np.zeros((21, 3), np.float32),
                        pinch=pinch, pinching=pinch > 0.6,
                        pinch_point=j[8].copy(),
                        pinch_velocity=np.zeros(3, np.float32),
                        confidence=1.0, track_id=track_id)

    def tick(poses: list[HandPose], n: int = 20) -> None:
        for _ in range(n):
            solver.set_hands(poses, DT)
            grips.update(poses, DT, solver)
            solver.step(DT)

    for _ in range(90):
        solver.step(DT)

    for leaving, staying in ((0, 1), (1, 0)):
        grips.reset()
        solver.reset()
        state.upload_materials([evaluate(CLOTH_MATERIAL, 0.30)])
        for _ in range(60):
            solver.step(DT)

        tick([hand(0, -0.12, 1.0), hand(1, +0.12, 1.0)])
        both = [g.count for g in grips.grips]
        require(all(c > 0 for c in both), f"neither hand grabbed: {both}")

        # The tracker compacts, so the survivor changes list position but
        # keeps its track id.
        tick([hand(staying, 0.12 if staying else -0.12, 1.0)])
        held = solver.stats()["grabbed"]
        require(held == grips.total_held,
                f"hand {leaving} left and the solver holds {held} particles "
                f"while the manager reports {grips.total_held}")
        require(grips.grips[staying].held,
                f"hand {staying} lost its grip when hand {leaving} left")
        require(not grips.grips[leaving].held,
                f"the grip of the departed hand {leaving} is still latched")

        # ... and it can still let go and take hold again.
        tick([hand(staying, 0.12 if staying else -0.12, 0.0)])
        require(grips.total_held == 0, "the release did not reach the solver")
        tick([hand(staying, 0.12 if staying else -0.12, 1.0)], 30)
        require(grips.grips[staying].held and grips.total_held > 0,
                f"hand {staying} could not grab again after hand "
                f"{leaving} left")
        require(solver.stats()["grabbed"] == grips.total_held,
                "the solver and the manager disagree after a re-grab")
        note(f"hand {leaving} leaves: hand {staying} keeps "
             f"{grips.total_held} particles, both sides agree")


@case
def opening_the_fingers_does_not_fling_what_they_were_holding() -> None:
    """A still hand letting go must let go, not throw.

    The grab anchor is the pinch point, the midpoint of the thumb and index
    tips.  Opening the fingers moves that midpoint several centimetres even
    when the wrist does not move at all, and the held region was being
    dragged along with it for the tenth of a second between the fingers
    starting to open and the release threshold being crossed -- then thrown
    with the midpoint's velocity on top.  A soft ball let go from a
    motionless hand left at 0.8 m/s sideways.  While a held pinch is opening,
    the anchor has to follow the wrist -- the part of the hand that is
    actually still -- and so has the throw.
    """
    cloth = make_cloth(40, pinned="none", self_collide=False)
    cfg = config()
    state, solver = build([cloth], cfg, [evaluate(CLOTH_MATERIAL, 0.5)])
    for _ in range(45):
        solver.step(DT)

    centre = state.x.numpy().mean(axis=0)

    def hand(pinch: float) -> HandPose:
        # Wrist fixed; thumb and index tips separate as the pinch opens, and
        # the pinch point (their midpoint) drifts upward and sideways with
        # them exactly as a real hand's does.
        j = np.zeros((21, 3), np.float32)
        j[:] = centre + np.array([0.0, 0.10, 0.0], np.float32)
        gap = (1.0 - pinch) * 0.06
        j[4] = centre + np.array([-gap * 0.5, 0.0 + gap * 0.6, 0.0], np.float32)
        j[8] = centre + np.array([+gap * 0.5, 0.0 + gap * 0.6, 0.0], np.float32)
        return HandPose(joints=j, velocities=np.zeros((21, 3), np.float32),
                        pinch=pinch, pinching=pinch > 0.6,
                        pinch_point=((j[4] + j[8]) * 0.5).astype(np.float32),
                        pinch_velocity=np.zeros(3, np.float32),
                        confidence=1.0, track_id=0)

    closed = hand(1.0)
    for _ in range(10):
        solver.set_hands([closed], DT)
        solver.step(DT)
    grabbed = solver.begin_grab(0, closed)
    require(grabbed > 0, "the pinch found nothing to hold")
    for _ in range(30):
        solver.set_hands([closed], DT)
        solver.step(DT)
    before = state.v.numpy()[state.flags.numpy() & FLAG_GRABBED != 0]
    require(np.abs(before).max() < 0.05,
            f"held particles were already moving at {np.abs(before).max():.2f} m/s")

    # Open the fingers over 0.4 s with the wrist dead still, releasing when
    # the grip manager would -- below the release threshold.
    fastest_while_held = 0.0
    released = False
    for i in range(40):
        pinch = 1.0 - (i + 1) / 40.0
        pose = hand(pinch)
        solver.set_hands([pose], DT)
        if not released and pinch < cfg.grab.release_threshold:
            solver.end_grab(0, pose)
            released = True
        solver.step(DT)
        if not released:
            held = state.flags.numpy() & FLAG_GRABBED != 0
            fastest_while_held = max(
                fastest_while_held,
                float(np.linalg.norm(state.v.numpy()[held], axis=1).max()))
    require(released, "the pinch never opened past the release threshold")
    for _ in range(3):
        solver.step(DT)
    drift = state.x.numpy().mean(axis=0) - centre
    sideways = float(np.hypot(drift[0], drift[2]))
    note(f"fastest held particle while the fingers opened: "
         f"{fastest_while_held:.3f} m/s; the sheet drifted "
         f"{sideways * 1000:.1f} mm sideways after release")
    require(fastest_while_held < 0.12,
            f"opening the fingers dragged the held matter at "
            f"{fastest_while_held:.2f} m/s from a motionless wrist")
    require(sideways < 0.02,
            f"a still hand threw the sheet {sideways * 1000:.0f} mm sideways")


if __name__ == "__main__":
    wp.init()
    sys.exit(run(__file__))
