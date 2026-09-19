"""The hardness dial is the product. These checks are its contract."""

from __future__ import annotations

import math

from _harness import approx, case, note, require, run  # noqa: E402

from fctx.core.material import (  # noqa: E402
    DEFAULT_MATERIALS,
    areal_particle_mass,
    evaluate,
    grain_particle_mass,
    log_lerp,
    volumetric_particle_mass,
)
from fctx.core.types import MatterKind  # noqa: E402

SAMPLES = [i / 64.0 for i in range(65)]


@case
def endpoints_match_the_declared_constants() -> None:
    for kind, params in DEFAULT_MATERIALS.items():
        soft = evaluate(params, 0.0)
        hard = evaluate(params, 1.0)
        require(approx(soft.young, params.young_soft, 1e-6),
                f"{kind.name}: young at 0 is {soft.young}, "
                f"expected {params.young_soft}")
        require(approx(hard.young, params.young_hard, 1e-6),
                f"{kind.name}: young at 1 is {hard.young}")
        require(approx(soft.stretch_k, params.stretch_soft, 1e-6),
                f"{kind.name}: stretch at 0 is {soft.stretch_k}")
        require(approx(hard.stretch_k, params.stretch_hard, 1e-6),
                f"{kind.name}: stretch at 1 is {hard.stretch_k}")


@case
def everything_is_finite_and_positive_across_the_whole_dial() -> None:
    fields = ("young", "lame_mu", "lame_lambda", "stretch_k", "shear_k",
              "bend_k", "stretch_compliance", "shear_compliance",
              "bend_compliance", "deviatoric_compliance",
              "hydrostatic_compliance", "grab_compliance", "damping")
    for kind, params in DEFAULT_MATERIALS.items():
        for h in SAMPLES:
            m = evaluate(params, h)
            for f in fields:
                v = getattr(m, f)
                require(math.isfinite(v) and v > 0.0,
                        f"{kind.name} h={h}: {f} = {v}")
            require(0.0 <= m.friction <= 1.0, f"{kind.name} friction {m.friction}")
            require(0.0 <= m.restitution <= 1.0,
                    f"{kind.name} restitution {m.restitution}")
            require(m.poisson < 0.5,
                    f"{kind.name} poisson {m.poisson} would make lambda infinite")


@case
def stiffness_rises_and_compliance_falls_monotonically() -> None:
    for kind, params in DEFAULT_MATERIALS.items():
        prev = evaluate(params, 0.0)
        for h in SAMPLES[1:]:
            cur = evaluate(params, h)
            require(cur.young >= prev.young,
                    f"{kind.name}: young went down at h={h}")
            require(cur.stretch_k >= prev.stretch_k,
                    f"{kind.name}: stretch went down at h={h}")
            require(cur.stretch_compliance <= prev.stretch_compliance,
                    f"{kind.name}: compliance went up at h={h}")
            require(cur.deviatoric_compliance <= prev.deviatoric_compliance,
                    f"{kind.name}: deviatoric compliance went up at h={h}")
            require(cur.damping <= prev.damping,
                    f"{kind.name}: damping went up at h={h}")
            prev = cur


@case
def the_dial_spans_enough_decades_to_feel_like_a_different_material() -> None:
    # If the ends are not far apart the demo has nothing to show, so this is a
    # product requirement expressed as a number.
    for kind, params in DEFAULT_MATERIALS.items():
        soft, hard = evaluate(params, 0.0), evaluate(params, 1.0)
        decades = math.log10(hard.young / soft.young)
        require(decades >= 3.0,
                f"{kind.name}: only {decades:.1f} decades of Young's modulus")
        note(f"{kind.name:<6} {decades:.1f} decades of E, "
             f"{math.log10(soft.stretch_compliance / hard.stretch_compliance):.1f}"
             f" decades of stretch compliance")


@case
def lame_parameters_follow_from_young_and_poisson() -> None:
    for params in DEFAULT_MATERIALS.values():
        for h in (0.0, 0.25, 0.5, 0.75, 1.0):
            m = evaluate(params, h)
            mu = m.young / (2.0 * (1.0 + m.poisson))
            lam = (m.young * m.poisson
                   / ((1.0 + m.poisson) * (1.0 - 2.0 * m.poisson)))
            require(approx(m.lame_mu, mu, 1e-9), f"mu mismatch at h={h}")
            require(approx(m.lame_lambda, lam, 1e-9), f"lambda mismatch at h={h}")


@case
def compliance_is_exactly_the_reciprocal_of_stiffness() -> None:
    for params in DEFAULT_MATERIALS.values():
        for h in SAMPLES:
            m = evaluate(params, h)
            require(approx(m.stretch_compliance * m.stretch_k, 1.0, 1e-9),
                    "stretch compliance is not 1/k")
            require(approx(m.bend_compliance * m.bend_k, 1.0, 1e-9),
                    "bend compliance is not 1/k")
            require(approx(m.deviatoric_compliance * m.lame_mu, 1.0, 1e-9),
                    "deviatoric compliance is not 1/mu")
            require(approx(m.hydrostatic_compliance * m.lame_lambda, 1.0, 1e-9),
                    "hydrostatic compliance is not 1/lambda")


@case
def the_dial_clamps_instead_of_extrapolating() -> None:
    params = DEFAULT_MATERIALS[MatterKind.SOFT]
    for bad, expect in ((-5.0, 0.0), (17.0, 1.0), (float("-inf"), 0.0)):
        m = evaluate(params, bad)
        require(approx(m.hardness, expect, 1e-9),
                f"hardness {bad} clamped to {m.hardness}, expected {expect}")


@case
def log_lerp_is_geometric_and_hits_its_endpoints() -> None:
    require(approx(log_lerp(2.0, 2000.0, 0.0), 2.0))
    require(approx(log_lerp(2.0, 2000.0, 1.0), 2000.0))
    mid = log_lerp(2.0, 2000.0, 0.5)
    require(approx(mid, math.sqrt(2.0 * 2000.0), 1e-9),
            f"midpoint {mid} is not the geometric mean")


@case
def names_and_descriptions_stay_short_enough_for_the_hud() -> None:
    seen = set()
    for params in DEFAULT_MATERIALS.values():
        for h in SAMPLES:
            m = evaluate(params, h)
            require(len(m.hardness_name) <= 14,
                    f"hardness name too long: {m.hardness_name!r}")
            require(len(m.describe()) <= 40,
                    f"describe() too long: {m.describe()!r}")
            seen.add(m.hardness_name)
    require(len(seen) >= 5, f"only {len(seen)} distinct names on the dial")
    note(f"names on the dial: {sorted(seen)}")


@case
def particle_masses_are_positive_and_scale_correctly() -> None:
    cloth = DEFAULT_MATERIALS[MatterKind.CLOTH]
    soft = DEFAULT_MATERIALS[MatterKind.SOFT]
    grain = DEFAULT_MATERIALS[MatterKind.GRAIN]
    require(areal_particle_mass(cloth, 0.0) > 0.0, "zero area gave zero mass")
    require(approx(areal_particle_mass(cloth, 2.0),
                   2.0 * areal_particle_mass(cloth, 1.0), 1e-6),
            "areal mass is not linear in area")
    require(volumetric_particle_mass(soft, 1e-6) > 0.0, "tiny volume gave no mass")
    m1 = grain_particle_mass(grain, 0.005)
    m2 = grain_particle_mass(grain, 0.010)
    require(approx(m2 / m1, 8.0, 1e-6),
            f"grain mass should scale with r^3, got ratio {m2 / m1}")


@case
def colour_moves_continuously_with_no_jump_anywhere() -> None:
    for params in DEFAULT_MATERIALS.values():
        prev = evaluate(params, 0.0)
        for h in SAMPLES[1:]:
            cur = evaluate(params, h)
            jump = max(abs(a - b) for a, b in zip(cur.color, prev.color))
            require(jump < 0.05,
                    f"colour jumped by {jump:.3f} at h={h}; the dial must "
                    "look continuous, not stepped")
            prev = cur


if __name__ == "__main__":
    raise SystemExit(run(__file__))
