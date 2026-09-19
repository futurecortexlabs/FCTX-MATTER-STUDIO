"""The hardness dial: one scalar, every material property.

The centrepiece of FCTX MATTER STUDIO is being able to slide a body from
jelly to near-rigid *while your hand is holding it*, and have every downstream
behaviour follow: how far it stretches, how it folds, how it wobbles once you
let go, how it collides with everything else, and how it looks.

That is only honest if a single physical parameter drives all of it, so this
module maps ``hardness`` in [0, 1] onto real material constants:

* cloth  -> a stretch stiffness in N/m and a bending stiffness in N*m/rad
* soft   -> a Young's modulus in Pa and a Poisson ratio
* grain  -> contact stiffness and friction

and then converts those into the XPBD *compliance* values the solver wants.
Compliance is the inverse of stiffness, and XPBD divides it by dt^2 internally,
which is exactly why the feel of the material does not change when the solver
runs more or fewer substeps.  Sliding the dial changes the material; it never
changes the numerics.

References
----------
Macklin, Muller & Chentanez, "XPBD: Position-Based Simulation of Compliant
Constrained Dynamics" (2016) -- the compliance formulation.

Macklin & Muller, "A Constraint-based Formulation of Stable Neo-Hookean
Materials" (2021) -- the two-constraint tetrahedron used for soft bodies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .types import MatterKind

__all__ = [
    "MaterialParams",
    "Material",
    "CLOTH_MATERIAL",
    "SOFT_MATERIAL",
    "GRAIN_MATERIAL",
    "DEFAULT_MATERIALS",
    "log_lerp",
]


def log_lerp(lo: float, hi: float, t: float) -> float:
    """Interpolate geometrically from ``lo`` to ``hi``.

    Stiffness is perceived logarithmically -- 1 kPa to 2 kPa is a far bigger
    change than 4 MPa to 4.001 MPa -- so the dial has to move in decades for
    it to feel linear under the hand.
    """
    t = min(max(t, 0.0), 1.0)
    return float(lo * (hi / lo) ** t)


def _lerp(lo: float, hi: float, t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return float(lo + (hi - lo) * t)


@dataclass(frozen=True, slots=True)
class MaterialParams:
    """The endpoints of the hardness dial for one kind of matter.

    Every ``*_soft`` / ``*_hard`` pair is the value at ``hardness = 0`` and at
    ``hardness = 1``.
    """

    kind: MatterKind
    label: str

    # -- cloth ------------------------------------------------------------
    #: Stretch stiffness of one structural edge, N/m.
    stretch_soft: float = 2.0e2
    stretch_hard: float = 5.0e5
    #: Shear edges are a fraction of the structural stiffness; real fabric
    #: gives way to shear long before it gives way to a straight pull.
    shear_ratio: float = 0.25
    #: Bending stiffness, N*m/rad.  Silk is ~1e-5, starched canvas ~1e-1.
    bend_soft: float = 2.0e-6
    bend_hard: float = 8.0e-2

    # -- soft body --------------------------------------------------------
    #: Young's modulus, Pa.  Gelatin ~1e3, rubber eraser ~5e6.
    young_soft: float = 1.5e3
    young_hard: float = 8.0e6
    #: Poisson ratio.  Stays below 0.5 or the Lame lambda blows up; a soft
    #: body reads as more "liquid inside" when it resists volume change more.
    poisson_soft: float = 0.30
    poisson_hard: float = 0.45

    # -- shared -----------------------------------------------------------
    #: Density, kg/m^3 (cloth uses kg/m^2 areal density instead).
    density: float = 420.0
    #: Areal density for cloth, kg/m^2.  Cotton sheeting is about 0.15.
    areal_density: float = 0.18
    #: Velocity damping per second, as a fraction retained.  Softer matter
    #: sheds energy faster -- that is most of what "soft" looks like in motion.
    damping_soft: float = 3.2
    damping_hard: float = 0.35
    #: Coulomb friction against hands, ground and other matter.
    friction_soft: float = 0.85
    friction_hard: float = 0.30
    #: Restitution.  Hard matter keeps a little bounce; jelly keeps none.
    restitution_soft: float = 0.0
    restitution_hard: float = 0.25

    # -- grab -------------------------------------------------------------
    #: Compliance of the attachment holding a grabbed particle to the hand.
    #: Soft matter oozes out of a pinch; hard matter is held firmly.
    grab_compliance_soft: float = 1.2e-5
    grab_compliance_hard: float = 2.0e-7

    # -- appearance -------------------------------------------------------
    #: Linear sRGB base colour at hardness 0 and 1.
    color_soft: tuple[float, float, float] = (0.98, 0.42, 0.52)
    color_hard: tuple[float, float, float] = (0.36, 0.62, 0.95)
    roughness_soft: float = 0.72
    roughness_hard: float = 0.22
    #: How much light bleeds through the body; jelly glows, steel does not.
    translucency_soft: float = 0.85
    translucency_hard: float = 0.04
    metallic_hard: float = 0.55


CLOTH_MATERIAL = MaterialParams(
    kind=MatterKind.CLOTH,
    label="Cloth",
    stretch_soft=1.2e2,
    stretch_hard=9.0e4,
    shear_ratio=0.22,
    bend_soft=1.0e-6,
    bend_hard=6.0e-2,
    areal_density=0.16,
    damping_soft=2.4,
    damping_hard=0.30,
    friction_soft=0.90,
    friction_hard=0.40,
    restitution_soft=0.0,
    restitution_hard=0.05,
    grab_compliance_soft=6.0e-6,
    grab_compliance_hard=1.0e-7,
    color_soft=(0.96, 0.38, 0.45),
    color_hard=(0.42, 0.66, 0.98),
    roughness_soft=0.86,
    roughness_hard=0.34,
    translucency_soft=0.55,
    translucency_hard=0.06,
    metallic_hard=0.25,
)

SOFT_MATERIAL = MaterialParams(
    kind=MatterKind.SOFT,
    label="Soft body",
    # 6 kPa is the softest this can honestly go.  A body of density 520 puts
    # rho*g*h = 1.3 kPa of gravitational stress at the base of the shipped
    # 0.26 m shape, so below a few kPa the material cannot hold itself up: at
    # 1.2 kPa, measured over a 3 s settle at hardness 0, 568 of the 12,765
    # tetrahedra invert and it loses a tenth of its volume.  The cliff is
    # sharp and it sits at 4 kPa -- 3.5 kPa leaves 437 inverted, 4 kPa leaves
    # 22 -- for every size from 0.20 to 0.30 m, so 6 kPa is a 1.5x margin
    # over it rather than the edge of it.  Volume then holds to within 5%
    # (0.981 at 0.26 m, 0.959 at 0.44 m) and it still squashes visibly on
    # landing.  The couple of dozen that stay inverted above the cliff are
    # not a material failure and no stiffness up to 12 MPa removes them:
    # they are the SDF-projected skin slivers ARCHITECTURE 10 describes, a
    # tetrahedron with all four vertices on the skin having nowhere to give.
    young_soft=6.0e3,
    young_hard=1.2e7,
    poisson_soft=0.28,
    poisson_hard=0.46,
    density=520.0,
    damping_soft=3.6,
    damping_hard=0.30,
    friction_soft=0.80,
    friction_hard=0.28,
    restitution_soft=0.0,
    restitution_hard=0.35,
    grab_compliance_soft=1.6e-5,
    grab_compliance_hard=2.0e-7,
    color_soft=(1.00, 0.46, 0.40),
    color_hard=(0.50, 0.70, 1.00),
    roughness_soft=0.62,
    roughness_hard=0.16,
    translucency_soft=0.95,
    translucency_hard=0.03,
    metallic_hard=0.70,
)

GRAIN_MATERIAL = MaterialParams(
    kind=MatterKind.GRAIN,
    label="Granular",
    stretch_soft=1.0e3,
    stretch_hard=2.0e5,
    density=1400.0,
    damping_soft=1.8,
    damping_hard=0.55,
    friction_soft=0.95,
    friction_hard=0.35,
    restitution_soft=0.0,
    restitution_hard=0.45,
    grab_compliance_soft=4.0e-5,
    grab_compliance_hard=5.0e-6,
    color_soft=(1.00, 0.62, 0.30),
    color_hard=(0.62, 0.78, 1.00),
    roughness_soft=0.90,
    roughness_hard=0.20,
    translucency_soft=0.30,
    translucency_hard=0.02,
    metallic_hard=0.40,
)

#: Upper bounds of each band on the dial.  The last name has no edge: it
#: covers everything above the final entry.
_HARDNESS_EDGES = (0.12, 0.30, 0.50, 0.70, 0.88)

_HARDNESS_NAMES: dict[MatterKind, tuple[str, ...]] = {
    MatterKind.CLOTH: ("GOSSAMER", "CHIFFON", "COTTON", "DENIM", "CANVAS",
                       "SHEET METAL"),
    MatterKind.SOFT: ("LIQUID JELLY", "SOFT GEL", "RUBBER", "FIRM", "STIFF",
                      "NEAR RIGID"),
    MatterKind.GRAIN: ("DRY SAND", "LOOSE SAND", "GRAVEL", "PACKED",
                       "COMPACT", "FUSED"),
}

DEFAULT_MATERIALS: dict[MatterKind, MaterialParams] = {
    MatterKind.CLOTH: CLOTH_MATERIAL,
    MatterKind.SOFT: SOFT_MATERIAL,
    MatterKind.GRAIN: GRAIN_MATERIAL,
}


@dataclass(frozen=True, slots=True)
class Material:
    """A :class:`MaterialParams` evaluated at one hardness, in solver units.

    The ``*_compliance`` fields are XPBD compliances: ``alpha = 1 / k``.  The
    solver divides them by ``dt^2`` per substep, which is what makes the
    behaviour independent of the substep count.
    """

    hardness: float
    params: MaterialParams

    # physical constants at this hardness
    young: float          # Pa
    poisson: float
    lame_mu: float        # Pa
    lame_lambda: float    # Pa
    stretch_k: float      # N/m
    shear_k: float        # N/m
    bend_k: float         # N*m/rad

    # XPBD compliances (per unit of the relevant measure)
    stretch_compliance: float   # m/N
    shear_compliance: float     # m/N
    bend_compliance: float      # rad/(N*m)
    #: Deviatoric and hydrostatic compliance are divided by each tetrahedron's
    #: rest volume inside the kernel, so these are the per-unit-volume values.
    deviatoric_compliance: float
    hydrostatic_compliance: float
    grab_compliance: float

    # contact and motion
    damping: float
    friction: float
    restitution: float

    # appearance
    color: tuple[float, float, float]
    roughness: float
    metallic: float
    translucency: float

    #: What the tetrahedra can actually deliver, when it is less than
    #: ``young``.  The discretisation caps the stiffness one element resolves
    #: per substep; above it the dial is carried by edge constraints and the
    #: body is stiffer than this number but not as stiff as ``young``.  The
    #: solver reports the ceiling and the app fills this in, so the HUD can
    #: say so instead of quoting a modulus that is not being simulated.
    young_effective: float | None = None

    @property
    def label(self) -> str:
        return self.params.label

    @property
    def hardness_name(self) -> str:
        """A short word for the current point on the dial, for the HUD.

        Named per kind rather than generically: "LIQUID JELLY" is the right
        word for a soft body and a meaningless one for a sheet of fabric, and
        a viewer reads the label before they read the modulus.
        """
        names = _HARDNESS_NAMES[self.params.kind]
        for edge, name in zip(_HARDNESS_EDGES, names):
            if self.hardness < edge:
                return name
        return names[-1]

    def describe(self) -> str:
        if self.params.kind is MatterKind.SOFT:
            eff = self.young_effective
            if eff is not None and eff < 0.97 * self.young:
                return (f"E {_eng(self.young)}Pa (tets {_eng(eff)}Pa)  "
                        f"nu {self.poisson:.2f}")
            return f"E = {_eng(self.young)}Pa   nu = {self.poisson:.2f}"
        if self.params.kind is MatterKind.CLOTH:
            return f"k = {_eng(self.stretch_k)}N/m   kb = {_eng(self.bend_k)}Nm"
        return f"k = {_eng(self.stretch_k)}N/m   mu = {self.friction:.2f}"


def _eng(x: float) -> str:
    """Format a number with an SI prefix, e.g. 1.2M, 340k, 5.0m."""
    if x == 0.0:
        return "0 "
    sign = "-" if x < 0 else ""
    x = abs(x)
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, " "),
                          (1e-3, "m"), (1e-6, "u")):
        if x >= limit:
            return f"{sign}{x / limit:.3g}{suffix}"
    return f"{sign}{x / 1e-9:.3g}n"


def evaluate(params: MaterialParams, hardness: float) -> Material:
    """Evaluate ``params`` at ``hardness`` in [0, 1]."""
    h = min(max(float(hardness), 0.0), 1.0)

    young = log_lerp(params.young_soft, params.young_hard, h)
    poisson = _lerp(params.poisson_soft, params.poisson_hard, h)
    # Lame parameters.  poisson is clamped below 0.5 so the denominator of
    # lambda can never reach zero.
    poisson = min(poisson, 0.49)
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    stretch_k = log_lerp(params.stretch_soft, params.stretch_hard, h)
    shear_k = stretch_k * params.shear_ratio
    bend_k = log_lerp(params.bend_soft, params.bend_hard, h)

    return Material(
        hardness=h,
        params=params,
        young=young,
        poisson=poisson,
        lame_mu=mu,
        lame_lambda=lam,
        stretch_k=stretch_k,
        shear_k=shear_k,
        bend_k=bend_k,
        stretch_compliance=1.0 / stretch_k,
        shear_compliance=1.0 / shear_k,
        bend_compliance=1.0 / bend_k,
        deviatoric_compliance=1.0 / mu,
        hydrostatic_compliance=1.0 / lam,
        grab_compliance=log_lerp(params.grab_compliance_soft,
                                 params.grab_compliance_hard, h),
        damping=log_lerp(params.damping_soft, params.damping_hard, h),
        friction=_lerp(params.friction_soft, params.friction_hard, h),
        restitution=_lerp(params.restitution_soft, params.restitution_hard, h),
        color=tuple(_lerp(a, b, h) for a, b in  # type: ignore[arg-type]
                    zip(params.color_soft, params.color_hard)),
        roughness=_lerp(params.roughness_soft, params.roughness_hard, h),
        metallic=params.metallic_hard * _smoothstep(0.55, 1.0, h),
        translucency=log_lerp(params.translucency_soft,
                              params.translucency_hard, h),
    )


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(max((x - edge0) / max(edge1 - edge0, 1e-9), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def with_overrides(params: MaterialParams, **kwargs: float) -> MaterialParams:
    """Return a copy of ``params`` with named endpoints replaced."""
    return replace(params, **kwargs)


def areal_particle_mass(params: MaterialParams, cell_area: float) -> float:
    """Mass of one cloth particle owning ``cell_area`` square metres."""
    return max(params.areal_density * cell_area, 1e-6)


def volumetric_particle_mass(params: MaterialParams, volume: float) -> float:
    """Mass of one soft-body particle owning ``volume`` cubic metres."""
    return max(params.density * volume, 1e-6)


def grain_particle_mass(params: MaterialParams, radius: float) -> float:
    """Mass of one grain of the given radius, at ~62% packing efficiency."""
    return max(params.density * 0.62 * (4.0 / 3.0) * math.pi * radius ** 3, 1e-7)
