"""Graph colouring for parallel constraint projection.

The solver projects constraints Gauss-Seidel style, one CUDA kernel launch per
colour.  Two constraints that share a particle must land in different colours
or their position writes race: the result is wrong *and* non-deterministic,
which is the worst kind of physics bug because it only shows up as occasional
jitter.  Everything in this module exists to make that impossible.

Fewer colours is directly fewer kernel launches per substep, and at 12 substeps
x 90 Hz the launch overhead is real, so the ordering heuristic matters.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "greedy_color",
    "colour_batches",
    "color_batches",
    "colour_permutation",
    "color_permutation",
    "sort_by_color",
    "num_colors",
    "verify_coloring",
]


def _greedy_pass(
    rows: list[list[int]], num_particles: int, order: list[int]
) -> tuple[list[int], int]:
    """One greedy sweep in the given visiting order; returns the colour count."""
    used: list[int] = [0] * num_particles
    colors: list[int] = [0] * len(rows)
    top = 0
    for c in order:
        row = rows[c]
        mask = 0
        for p in row:
            mask |= used[p]
        # Lowest clear bit of mask: mask+1 carries through the low run of ones,
        # and AND-ing with ~mask isolates exactly the first zero.
        bit = (~mask) & (mask + 1)
        length = bit.bit_length()
        colors[c] = length - 1
        if length > top:
            top = length
        for p in row:
            used[p] |= bit
    return colors, top


def greedy_color(indices: np.ndarray, num_particles: int) -> np.ndarray:
    """Colour constraints so no colour contains a particle twice.

    ``indices`` is ``(N, K)`` int32: the ``K`` particles each of the ``N``
    constraints touches (K = 2 for distance, 4 for bending and tetrahedra).
    Returns ``(N,)`` int32 colours, contiguous from 0.

    The primary ordering is largest-degree-first, which is what pays off on the
    irregular graphs: the crowded interior gets the palette while it is still
    empty and the sparse shell mops up afterwards.  Measured on the shipped
    tetrahedra, degree order against build order: sphere 38 colours to 39,
    box 37 to 39, torus 35 to 37 -- one or two launches per substep, on the
    largest constraint set in the scene.  A cloth grid is the
    opposite case -- it is so regular that its build order is already an
    optimal sweep, and degree ordering costs it one extra colour -- so both
    sweeps run and the better one wins.  Colouring happens once per scene
    build; a colour is a kernel launch on every substep of every frame
    thereafter, so the second sweep is close to free.
    """
    idx = np.ascontiguousarray(indices)
    if idx.ndim != 2:
        raise ValueError(f"indices must be 2-D (N, K), got shape {idx.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"indices must be an integer array, got {idx.dtype}")
    n, k = idx.shape
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    if k == 0:
        raise ValueError("indices must reference at least one particle per constraint")
    if num_particles <= 0:
        raise ValueError(f"num_particles must be positive, got {num_particles}")
    lo = int(idx.min())
    hi = int(idx.max())
    if lo < 0 or hi >= num_particles:
        raise ValueError(
            f"constraint particle index range [{lo}, {hi}] escapes "
            f"[0, {num_particles})")

    # Exact constraint degree would need the particle -> constraint adjacency,
    # which is O(N*K) memory and an extra pass; the sum of the valences of a
    # constraint's own particles is the same ordering to within duplicates and
    # costs one bincount.
    valence = np.bincount(idx.ravel(), minlength=num_particles)
    weight = valence[idx].sum(axis=1)

    rows = idx.tolist()
    by_degree, degree_count = _greedy_pass(
        rows, num_particles, np.argsort(-weight, kind="stable").tolist())
    as_built, built_count = _greedy_pass(rows, num_particles, list(range(n)))

    best = by_degree if degree_count < built_count else as_built
    return np.ascontiguousarray(best, dtype=np.int32)


def num_colors(colors: np.ndarray) -> int:
    """Number of distinct colour slots, i.e. kernel launches per iteration."""
    if colors.size == 0:
        return 0
    return int(colors.max()) + 1


def colour_permutation(colors: np.ndarray) -> np.ndarray:
    """Indices that sort constraints by colour, stably.

    Stability keeps the within-colour order equal to the build order, so a
    body's constraint arrays stay in a mesh-coherent layout and the memory
    access pattern inside a launch does not degrade to random.
    """
    if colors.size == 0:
        return np.zeros(0, dtype=np.int32)
    return np.ascontiguousarray(np.argsort(colors, kind="stable"), dtype=np.int32)


def colour_batches(colors: np.ndarray) -> list[tuple[int, int]]:
    """``(offset, count)`` per colour, into the colour-sorted constraint array.

    One entry per colour id up to the maximum, including any colour that no
    constraint uses, so ``batches[c]`` always means colour ``c``.
    """
    if colors.size == 0:
        return []
    if colors.min() < 0:
        raise ValueError("colours must be non-negative")
    counts = np.bincount(colors, minlength=num_colors(colors))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return [(int(offsets[i]), int(counts[i])) for i in range(counts.size)]


def sort_by_color(colors: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return ``(permutation, batches)`` together, which is what assembly needs."""
    return colour_permutation(colors), colour_batches(colors)


#: American spellings, because the solver and the renderer disagree about this.
color_permutation = colour_permutation
color_batches = colour_batches


def verify_coloring(indices: np.ndarray, colors: np.ndarray) -> int:
    """Raise ``ValueError`` if any colour touches a particle twice.

    Returns the colour count so callers can report it.
    """
    idx = np.asarray(indices)
    col = np.asarray(colors)
    if idx.shape[0] != col.shape[0]:
        raise ValueError(
            f"{idx.shape[0]} constraints but {col.shape[0]} colours")
    if idx.size == 0:
        return 0
    for c in range(num_colors(col)):
        members = idx[col == c]
        if members.size == 0:
            continue
        flat = members.ravel()
        if np.unique(flat).size != flat.size:
            dup = np.bincount(flat)
            worst = int(dup.argmax())
            raise ValueError(
                f"colour {c} uses particle {worst} {int(dup[worst])} times; "
                f"those constraints would race in one kernel launch")
    return num_colors(col)
