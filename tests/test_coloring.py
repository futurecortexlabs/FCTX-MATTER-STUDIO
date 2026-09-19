"""Graph colouring: the claim that makes parallel constraint projection legal.

If any colour ever contains two constraints that share a particle, the solver
races on that particle's position and the simulation stops being reproducible.
Everything here exists to prove that never happens.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from _harness import case, note, require, run

from fctx.bodies import build_cloth, build_granular, build_soft_body
from fctx.bodies.coloring import (
    colour_batches,
    colour_permutation,
    greedy_color,
    num_colors,
    sort_by_color,
    verify_coloring,
)
from fctx.config import SceneConfig
from fctx.core.material import CLOTH_MATERIAL, GRAIN_MATERIAL, SOFT_MATERIAL


def _check_body(label: str, body) -> None:
    for kind, idx, colors in (
        ("dist", body.dist_idx, body.dist_color),
        ("bend", body.bend_idx, body.bend_color),
        ("tet", body.tet_idx, body.tet_color),
    ):
        if idx.shape[0] == 0:
            continue
        count = verify_coloring(idx, colors)
        valence = np.bincount(idx.ravel(), minlength=body.num_particles).max()
        require(count >= valence,
                f"{label} {kind}: {count} colours cannot cover a particle "
                f"shared by {valence} constraints")
        batches = colour_batches(colors)
        require(sum(n for _, n in batches) == idx.shape[0],
                f"{label} {kind}: batches do not cover every constraint")
        note(f"{label:22s} {kind:5s} N={idx.shape[0]:6d} "
             f"colours={count:3d} (lower bound {valence})")


@case
def empty_input_is_handled() -> None:
    colors = greedy_color(np.zeros((0, 2), np.int32), 16)
    require(colors.shape == (0,), "empty colouring must be shape (0,)")
    require(colors.dtype == np.int32, "empty colouring must be int32")
    require(colour_batches(colors) == [], "empty colouring has no batches")
    require(num_colors(colors) == 0, "empty colouring uses no colours")


@case
def clique_needs_one_colour_each() -> None:
    # Every constraint touches particle 0, so nothing may share a colour.
    idx = np.stack([np.zeros(24, np.int32), np.arange(1, 25, dtype=np.int32)], axis=1)
    colors = greedy_color(idx, 25)
    require(sorted(colors.tolist()) == list(range(24)),
            "a star graph must use exactly one colour per edge")


@case
def disjoint_constraints_share_one_colour() -> None:
    idx = np.arange(64, dtype=np.int32).reshape(32, 2)
    colors = greedy_color(idx, 64)
    require(num_colors(colors) == 1,
            "constraints that share nothing must all fit in colour 0")


@case
def random_graphs_are_conflict_free() -> None:
    rng = np.random.default_rng(7)
    for particles, n, k in ((50, 400, 2), (200, 1500, 4), (3000, 20000, 4)):
        idx = np.empty((n, k), np.int32)
        for row in range(n):
            idx[row] = rng.choice(particles, size=k, replace=False)
        verify_coloring(idx, greedy_color(idx, particles))


@case
def colours_are_contiguous_from_zero() -> None:
    rng = np.random.default_rng(11)
    idx = rng.integers(0, 400, size=(3000, 4)).astype(np.int32)
    colors = greedy_color(idx, 400)
    present = np.unique(colors)
    require(present[0] == 0 and present.size == num_colors(colors),
            "greedy colouring must not leave an unused colour id behind")


@case
def batches_index_the_sorted_order() -> None:
    body = build_cloth(SceneConfig(cloth_resolution=16), CLOTH_MATERIAL)
    perm, batches = sort_by_color(body.dist_color)
    require(np.array_equal(perm, colour_permutation(body.dist_color)),
            "sort_by_color must agree with colour_permutation")
    ordered = body.dist_color[perm]
    require(np.all(np.diff(ordered) >= 0), "permutation must sort by colour")
    for colour, (offset, count) in enumerate(batches):
        chunk = ordered[offset:offset + count]
        require(chunk.size == count and np.all(chunk == colour),
                f"batch {colour} does not hold exactly its own colour")


def _naive_color_count(idx: np.ndarray, num_particles: int) -> int:
    used = [0] * num_particles
    top = 0
    for row in idx.tolist():
        mask = 0
        for p in row:
            mask |= used[p]
        bit = (~mask) & (mask + 1)
        top = max(top, bit.bit_length())
        for p in row:
            used[p] |= bit
    return top


@case
def ordering_never_loses_to_the_build_order() -> None:
    # Degree-first ordering is the win on an irregular mesh and a one-colour
    # loss on a perfectly regular grid, which is why the builder runs both
    # sweeps.  This pins that down in both directions.
    cloth = build_cloth(SceneConfig(cloth_resolution=48), CLOTH_MATERIAL)
    soft = build_soft_body(
        SceneConfig(soft_shape="sphere", soft_resolution=13), SOFT_MATERIAL)
    for label, idx, colors in (
        ("cloth-48 dist", cloth.dist_idx, cloth.dist_color),
        ("cloth-48 bend", cloth.bend_idx, cloth.bend_color),
        ("sphere-13 tet", soft.tet_idx, soft.tet_color),
    ):
        particles = cloth.num_particles if label.startswith("cloth") else soft.num_particles
        naive = _naive_color_count(idx, particles)
        chosen = num_colors(colors)
        note(f"{label}: chosen {chosen}, build order alone {naive}")
        require(chosen <= naive,
                f"{label} used more colours ({chosen}) than the build order "
                f"would have ({naive})")
    require(num_colors(soft.tet_color) < _naive_color_count(
        soft.tet_idx, soft.num_particles),
        "degree ordering must beat the build order on a tetrahedral lattice")


@case
def cloth_stays_well_under_twenty_colours() -> None:
    for resolution in (3, 8, 24, 48, 72, 88):
        body = build_cloth(SceneConfig(cloth_resolution=resolution), CLOTH_MATERIAL)
        _check_body(f"cloth-{resolution}", body)
        require(num_colors(body.dist_color) < 20,
                f"cloth-{resolution} distance colouring exploded to "
                f"{num_colors(body.dist_color)} colours")
        require(num_colors(body.bend_color) < 20,
                f"cloth-{resolution} bend colouring exploded to "
                f"{num_colors(body.bend_color)} colours")


@case
def soft_bodies_are_conflict_free() -> None:
    for shape in ("sphere", "box", "torus"):
        for resolution in (4, 6, 9, 13):
            scene = SceneConfig(soft_shape=shape, soft_resolution=resolution)
            _check_body(f"{shape}-{resolution}", build_soft_body(scene, SOFT_MATERIAL))


@case
def granular_has_nothing_to_colour() -> None:
    body = build_granular(SceneConfig(grain_count=4000), GRAIN_MATERIAL)
    _check_body("grain-4000", body)
    require(body.dist_color.size == 0 and body.bend_color.size == 0
            and body.tet_color.size == 0,
            "granular matter must carry no constraints at all")


@case
def colouring_cost_is_linear() -> None:
    # O(N^2) would be ~1e10 operations at the larger size and never return.
    rng = np.random.default_rng(3)
    timings = []
    for n in (20_000, 160_000):
        idx = rng.integers(0, n // 4, size=(n, 4)).astype(np.int32)
        start = time.perf_counter()
        greedy_color(idx, n // 4)
        timings.append(time.perf_counter() - start)
    growth = timings[1] / max(timings[0], 1e-9)
    note(f"8x the constraints cost {growth:.1f}x the time")
    require(growth < 24.0, f"colouring scaled super-linearly ({growth:.1f}x for 8x)")


if __name__ == "__main__":
    sys.exit(run(__file__))
