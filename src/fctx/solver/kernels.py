"""Warp kernels for the XPBD solver.

Every kernel here is written so that the whole substep loop can be replayed
from a CUDA graph.  That imposes one rule on the signatures: a scalar argument
is baked into the graph at capture time and would silently freeze at its first
value, so the only scalars any of them take are ones that never change between
frames -- the substep length ``h``, which the fixed-timestep clock guarantees
is constant, the constant ``offset`` of a colour batch, and ``solve_capsules``'
``frac``, which is a different literal per unrolled substep and the same
literal every frame.  Hardness, hands, grabs, wind and the ground all arrive
through device arrays.

The XPBD update implemented throughout is

    a~  = alpha / h^2
    dl  = (-C - a~ * lambda) / (sum_i w_i |grad_i C|^2 + a~)
    dx_i = w_i * grad_i C * dl

with ``alpha`` a real compliance in SI units, never a stiffness multiplier, so
the resulting behaviour does not move when the substep count moves.
"""

from __future__ import annotations

import warp as wp

# Indices into the scalar parameter array.  These live in a device array rather
# than in kernel arguments because the HUD, the wind toggle and the preset
# switcher all change them while a captured graph is running.
PARAM_AIR_DRAG = wp.constant(0)
PARAM_GROUND_Y = wp.constant(1)
PARAM_GROUND_FRICTION = wp.constant(2)
PARAM_COLLISION_MARGIN = wp.constant(3)
PARAM_MAX_CORRECTION_RATIO = wp.constant(4)
PARAM_MAX_VELOCITY = wp.constant(5)
PARAM_WIND_TURBULENCE = wp.constant(6)
PARAM_TIME = wp.constant(7)
PARAM_SELF_COLLIDE = wp.constant(8)
PARAM_GRAB_ROTATE = wp.constant(9)
#: Radius of the cylindrical basin the matter sits in, metres.
#: Zero means an open floor.
PARAM_BASIN_RADIUS = wp.constant(10)
#: Height of the basin wall above the ground plane.  Above it a
#: particle is free, which is what lets a hand lift matter out.
PARAM_BASIN_HEIGHT = wp.constant(11)
NUM_PARAMS = 12

# Indices into the vector parameter array.
VPARAM_GRAVITY = wp.constant(0)
VPARAM_WIND = wp.constant(1)
NUM_VPARAMS = 2

KIND_SHEAR = wp.constant(1)

FLAG_GRABBED_BIT = wp.constant(wp.uint32(2))
FLAG_SELF_COLLIDE_BIT = wp.constant(wp.uint32(8))

#: How far the XPBD compliance term of a tetrahedron must dominate its mass
#: term.  See :func:`resolvable_softening`.
TET_STIFFNESS_MARGIN = wp.constant(2.0)

_EPS = wp.constant(1.0e-9)
_UP = wp.constant(wp.vec3(0.0, 1.0, 0.0))

#: Successive over-relaxation for the averaged particle contact solve, at the
#: dense end.  Averaging n simultaneous contacts is what stops a pile from
#: exploding, but it also under-solves each one by a factor of n, and the
#: over-relaxation buys some of that back.  It is faded in with the contact
#: count: a particle touching exactly one neighbour has nothing averaged away,
#: so over-relaxing it is pure overshoot -- it separates past the contact,
#: finalize turns the excess into velocity, and the pair flies apart at twice
#: the distance it asked for.
_CONTACT_SOR = wp.constant(1.45)

#: How much faster than the capsule itself a capsule contact may move a
#: particle in one substep, and the fraction of a particle radius per substep
#: a stationary capsule may push at.  At 0.10 a cloth particle (4.6 mm) leaves
#: a still hand at up to 0.5 m/s and is clear within a frame; at 1.0 it would
#: leave at 5 m/s, which is the ejection this bound exists to stop.
_CAPSULE_PUSH_SLACK = wp.constant(2.0)
_CAPSULE_SETTLE = wp.constant(0.10)

#: How fast the basin wall may claw back an excursion it did not just watch
#: happen, m/s.  See :func:`solve_ground`.  About the speed of the hand that
#: put the matter there, which is the rule: the wall may never move matter
#: faster than whatever displaced it.  It is also, at 0.93 mm per substep,
#: far more than the tenth of a millimetre the settled pile's own contacts
#: push a wall grain out by -- so for everything the pile does on its own the
#: bound never binds and the wall behaves exactly as it did without it.
_BASIN_RECOVERY_SPEED = wp.constant(1.0)


@wp.func
def limit_correction(d: wp.vec3, max_len: float) -> wp.vec3:
    """Shorten ``d`` to ``max_len`` without changing its direction."""
    l2 = wp.dot(d, d)
    if l2 > max_len * max_len and l2 > 1.0e-24:
        return d * (max_len / wp.sqrt(l2))
    return d


@wp.func
def column(m: wp.mat33, j: int) -> wp.vec3:
    return wp.vec3(m[0, j], m[1, j], m[2, j])


@wp.func
def turbulent_air(wind: wp.vec3, strength: float, p: wp.vec3, t: float) -> wp.vec3:
    """Wind plus a divergence-ignoring swirl.

    A curl-noise field would be physically nicer, but this runs inside the
    integrate kernel for every particle at 1080 substeps a second; three sine
    products cost almost nothing and the eye only reads "the banner is alive".
    The amplitude is proportional to the wind speed so that turning the wind
    off also turns the swirl off.
    """
    if strength <= 0.0:
        return wind
    amp = strength * wp.length(wind)
    if amp <= 0.0:
        return wind
    sx = wp.sin(2.3 * p[1] + 1.7 * t) * wp.cos(1.9 * p[2] - 1.1 * t)
    sy = wp.sin(3.1 * p[0] - 1.3 * t) * wp.cos(2.2 * p[2] + 0.7 * t)
    sz = wp.cos(2.7 * p[0] + 2.1 * t) * wp.sin(1.5 * p[1] + 0.9 * t)
    return wind + wp.vec3(sx, 0.5 * sy, sz) * amp


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------


@wp.kernel
def reset_lambdas(dist_lambda: wp.array(dtype=float),
                  bend_lambda: wp.array(dtype=float),
                  tet_lambda_d: wp.array(dtype=float),
                  tet_lambda_h: wp.array(dtype=float),
                  grab_lambda: wp.array(dtype=float),
                  n_dist: int, n_bend: int, n_tet: int, n_grab: int) -> None:
    """Zero every XPBD multiplier.

    This runs at the start of each *substep*.  Carrying multipliers across a
    substep boundary leaves a residual lambda that opposes the next solve, and
    the material then behaves stiffer than the compliance asked for -- by a
    factor that depends on the substep count, which is exactly the coupling
    compliance exists to remove.
    """
    t = wp.tid()
    if t < n_dist:
        dist_lambda[t] = 0.0
    if t < n_bend:
        bend_lambda[t] = 0.0
    if t < n_tet:
        tet_lambda_d[t] = 0.0
        tet_lambda_h[t] = 0.0
    if t < n_grab:
        grab_lambda[t] = 0.0


@wp.kernel
def clear_counters(contact_count: wp.array(dtype=wp.int32)) -> None:
    contact_count[0] = 0


@wp.kernel
def advance_time(params: wp.array(dtype=float), h: float) -> None:
    params[PARAM_TIME] = params[PARAM_TIME] + h


# ---------------------------------------------------------------------------
# 1. integration
# ---------------------------------------------------------------------------


@wp.kernel
def integrate(x: wp.array(dtype=wp.vec3),
              x_prev: wp.array(dtype=wp.vec3),
              v: wp.array(dtype=wp.vec3),
              w: wp.array(dtype=float),
              contact_n: wp.array(dtype=wp.vec3),
              contact_vn: wp.array(dtype=float),
              params: wp.array(dtype=float),
              vparams: wp.array(dtype=wp.vec3),
              h: float) -> None:
    t = wp.tid()
    p = x[t]
    x_prev[t] = p
    contact_n[t] = wp.vec3(0.0, 0.0, 0.0)
    contact_vn[t] = 0.0
    if w[t] <= 0.0:
        v[t] = wp.vec3(0.0, 0.0, 0.0)
        return

    vel = v[t] + vparams[VPARAM_GRAVITY] * h
    air = turbulent_air(vparams[VPARAM_WIND], params[PARAM_WIND_TURBULENCE],
                        p, params[PARAM_TIME])
    # Drag towards the air velocity as an exponential relaxation rather than a
    # multiply: a naive (1 - drag*h) goes unstable once drag*h exceeds 1 and,
    # worse, makes the terminal velocity depend on the substep count.
    decay = wp.exp(-params[PARAM_AIR_DRAG] * h)
    vel = air + (vel - air) * decay

    v[t] = vel
    x[t] = p + vel * h


# ---------------------------------------------------------------------------
# 2. distance constraints
# ---------------------------------------------------------------------------


@wp.kernel
def solve_distance(x: wp.array(dtype=wp.vec3),
                   w: wp.array(dtype=float),
                   radius: wp.array(dtype=float),
                   idx: wp.array(dtype=wp.vec2i),
                   rest: wp.array(dtype=float),
                   kind: wp.array(dtype=wp.int32),
                   body: wp.array(dtype=wp.int32),
                   lam: wp.array(dtype=float),
                   mat_stretch: wp.array(dtype=float),
                   mat_shear: wp.array(dtype=float),
                   params: wp.array(dtype=float),
                   offset: int,
                   h: float) -> None:
    t = wp.tid() + offset
    e = idx[t]
    i = e[0]
    j = e[1]
    wi = w[i]
    wj = w[j]
    wsum = wi + wj
    if wsum <= 0.0:
        return

    d = x[i] - x[j]
    length = wp.length(d)
    if length < _EPS:
        return
    n = d / length

    b = body[t]
    alpha = mat_stretch[b]
    if kind[t] == KIND_SHEAR:
        alpha = mat_shear[b]
    a_tilde = alpha / (h * h)

    c = length - rest[t]
    dlam = (-c - a_tilde * lam[t]) / (wsum + a_tilde)
    lam[t] = lam[t] + dlam

    ratio = params[PARAM_MAX_CORRECTION_RATIO]
    if wi > 0.0:
        x[i] = x[i] + limit_correction(n * (wi * dlam), ratio * radius[i])
    if wj > 0.0:
        x[j] = x[j] - limit_correction(n * (wj * dlam), ratio * radius[j])


# ---------------------------------------------------------------------------
# 3. dihedral bending
# ---------------------------------------------------------------------------


@wp.kernel
def solve_bending(x: wp.array(dtype=wp.vec3),
                  w: wp.array(dtype=float),
                  radius: wp.array(dtype=float),
                  idx: wp.array(dtype=wp.vec4i),
                  rest: wp.array(dtype=float),
                  body: wp.array(dtype=wp.int32),
                  lam: wp.array(dtype=float),
                  mat_bend: wp.array(dtype=float),
                  params: wp.array(dtype=float),
                  offset: int,
                  h: float) -> None:
    t = wp.tid() + offset
    e = idx[t]
    # (edge0, edge1, wing0, wing1): the two triangles are (w0, e0, e1) and
    # (w1, e1, e0), which is the winding ARCHITECTURE.md 6.3 assumes.
    i2 = e[0]
    i3 = e[1]
    i0 = e[2]
    i1 = e[3]

    w0 = w[i0]
    w1 = w[i1]
    w2 = w[i2]
    w3 = w[i3]
    if w0 + w1 + w2 + w3 <= 0.0:
        return

    p0 = x[i0]
    p1 = x[i1]
    p2 = x[i2]
    p3 = x[i3]

    n1 = wp.cross(p2 - p0, p3 - p0)
    n2 = wp.cross(p3 - p1, p2 - p1)
    l1 = wp.length(n1)
    l2 = wp.length(n2)
    # A degenerate (zero-area) triangle has no defined normal; its dihedral
    # gradient is the 0/0 that turns a folded sheet into NaN in one substep.
    if l1 < 1.0e-12 or l2 < 1.0e-12:
        return
    u1 = n1 / l1
    u2 = n2 / l2

    d = wp.clamp(wp.dot(u1, u2), -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    denom = wp.sqrt(1.0 - d * d)

    # dd/dn1 and dd/dn2, from d(n_hat)/dn = (I - n_hat n_hat^T)/|n|.
    a = (u2 - u1 * d) / l1
    b = (u1 - u2 * d) / l2

    # dn1/dx and dn2/dx are cross products; contracting them with a and b gives
    # the gradients of d directly, with no 1/sin factor of their own.
    q0 = wp.cross(p2 - p3, a)
    q1 = wp.cross(p3 - p2, b)
    q2 = wp.cross(p3 - p0, a) + wp.cross(p1 - p3, b)
    q3 = wp.cross(p0 - p2, a) + wp.cross(p2 - p1, b)

    scale = -1.0 / denom
    g0 = q0 * scale
    g1 = q1 * scale
    g2 = q2 * scale
    g3 = q3 * scale

    wsum = (w0 * wp.dot(g0, g0) + w1 * wp.dot(g1, g1)
            + w2 * wp.dot(g2, g2) + w3 * wp.dot(g3, g3))
    if wsum <= 0.0:
        return

    c = wp.acos(d) - rest[t]
    a_tilde = mat_bend[body[t]] / (h * h)
    dlam = (-c - a_tilde * lam[t]) / (wsum + a_tilde)
    lam[t] = lam[t] + dlam

    ratio = params[PARAM_MAX_CORRECTION_RATIO]
    if w0 > 0.0:
        x[i0] = p0 + limit_correction(g0 * (w0 * dlam), ratio * radius[i0])
    if w1 > 0.0:
        x[i1] = p1 + limit_correction(g1 * (w1 * dlam), ratio * radius[i1])
    if w2 > 0.0:
        x[i2] = p2 + limit_correction(g2 * (w2 * dlam), ratio * radius[i2])
    if w3 > 0.0:
        x[i3] = p3 + limit_correction(g3 * (w3 * dlam), ratio * radius[i3])


# ---------------------------------------------------------------------------
# 4. stable Neo-Hookean tetrahedra
# ---------------------------------------------------------------------------


@wp.func
def resolvable_softening(a_dev: float, a_hyd: float, wsum_rest: float) -> float:
    """Fraction of the asked-for element stiffness this substep can deliver.

    An XPBD projection lands on the true compliant force, ``lambda = -C/a~``,
    only in one pass if ``a~`` dominates the mass term ``sum_i w_i |grad_i C|^2``.
    Below that it collapses into a hard PBD projection of ``C`` to zero.  For
    the hydrostatic constraint that is harmless -- ``det(F) = gamma`` is a pose
    an element can actually reach.  For the deviatoric one it is ruinous:
    ``C_D = sqrt(tr(F^T F))`` is zero only at ``F = 0``, an element collapsed to
    a point, so every substep asks for an infeasible projection, the edge
    distance constraints undo it, and the pair pumps energy until the body
    leaves the stage.  Measured on the ``cube`` preset that begins at hardness
    0.4 and by 0.6 the body is at the velocity clamp.

    The crossover is physics, not tuning: ``a~_D / wsum ~ 5 rho L^2 / (4 mu h^2)``,
    so for a 2 cm element of 520 kg/m^3 at h = 0.93 ms it sits at mu ~ 36 kPa.
    Asking for more is asking for a stiffness this discretisation cannot carry.

    So return the scale ``k <= 1`` that both compliances are divided by to bring
    them back inside what one pass resolves.  Scaling *both* is what makes this
    safe rather than merely different: the rest pose is force-free only because
    the deviatoric and hydrostatic rest forces cancel, and scaling ``mu`` and
    ``lambda`` together scales both multipliers by ``k`` and leaves the
    cancellation -- and ``gamma = 1 + mu/lambda`` -- exactly intact.  The result
    is an honest softer material (Young's modulus scaled by ``k``, Poisson ratio
    untouched), not a truncated correction, so it cannot inject energy.  The
    stiffness the dial promised at the hard end is then carried by the tet edge
    distance constraints that ARCHITECTURE.md 4.1 mandates, which project onto a
    rest length -- a target an element can actually reach.

    ``wsum_rest`` is the hydrostatic mass term at ``F = I``; the deviatoric one
    is exactly a third of it there, because ``grad C_D`` and ``grad C_H`` differ
    by the factor ``sqrt(3)`` when ``F = I``.
    """
    if wsum_rest <= 0.0:
        return 1.0
    need_d = TET_STIFFNESS_MARGIN * wsum_rest / 3.0
    need_h = TET_STIFFNESS_MARGIN * wsum_rest
    k = float(1.0)
    if a_dev < need_d:
        k = wp.min(k, a_dev / need_d)
    if a_hyd < need_h:
        k = wp.min(k, a_hyd / need_h)
    return wp.max(k, 1.0e-6)


@wp.func
def deformation_gradient(x: wp.array(dtype=wp.vec3),
                         i0: int, i1: int, i2: int, i3: int,
                         dm_inv: wp.mat33) -> wp.mat33:
    d1 = x[i1] - x[i0]
    d2 = x[i2] - x[i0]
    d3 = x[i3] - x[i0]
    ds = wp.mat33(d1[0], d2[0], d3[0],
                  d1[1], d2[1], d3[1],
                  d1[2], d2[2], d3[2])
    return ds * dm_inv


@wp.kernel
def solve_tet_deviatoric(x: wp.array(dtype=wp.vec3),
                         w: wp.array(dtype=float),
                         radius: wp.array(dtype=float),
                         idx: wp.array(dtype=wp.vec4i),
                         dm_inv: wp.array(dtype=wp.mat33),
                         volume: wp.array(dtype=float),
                         wsum_rest: wp.array(dtype=float),
                         body: wp.array(dtype=wp.int32),
                         lam: wp.array(dtype=float),
                         mat_dev: wp.array(dtype=float),
                         mat_hyd: wp.array(dtype=float),
                         params: wp.array(dtype=float),
                         offset: int,
                         h: float) -> None:
    t = wp.tid() + offset
    e = idx[t]
    i0 = e[0]
    i1 = e[1]
    i2 = e[2]
    i3 = e[3]
    w0 = w[i0]
    w1 = w[i1]
    w2 = w[i2]
    w3 = w[i3]
    if w0 + w1 + w2 + w3 <= 0.0:
        return

    dmi = dm_inv[t]
    f = deformation_gradient(x, i0, i1, i2, i3, dmi)

    i_c = wp.ddot(f, f)
    c = wp.sqrt(i_c)
    # A fully collapsed element has F = 0; the gradient F/|F| is then 0/0.
    # Bailing out leaves the companion hydrostatic constraint (whose gradient
    # stays finite) to push the element back open.
    if c < _EPS:
        return

    grad_f = f * (1.0 / c)
    g = grad_f * wp.transpose(dmi)
    g1 = column(g, 0)
    g2 = column(g, 1)
    g3 = column(g, 2)
    g0 = -(g1 + g2 + g3)

    wsum = (w0 * wp.dot(g0, g0) + w1 * wp.dot(g1, g1)
            + w2 * wp.dot(g2, g2) + w3 * wp.dot(g3, g3))
    if wsum <= 0.0:
        return

    b = body[t]
    inv_vh2 = 1.0 / (volume[t] * h * h)
    a_tilde = mat_dev[b] * inv_vh2 / resolvable_softening(
        mat_dev[b] * inv_vh2, mat_hyd[b] * inv_vh2, wsum_rest[t])
    dlam = (-c - a_tilde * lam[t]) / (wsum + a_tilde)
    lam[t] = lam[t] + dlam

    ratio = params[PARAM_MAX_CORRECTION_RATIO]
    if w0 > 0.0:
        x[i0] = x[i0] + limit_correction(g0 * (w0 * dlam), ratio * radius[i0])
    if w1 > 0.0:
        x[i1] = x[i1] + limit_correction(g1 * (w1 * dlam), ratio * radius[i1])
    if w2 > 0.0:
        x[i2] = x[i2] + limit_correction(g2 * (w2 * dlam), ratio * radius[i2])
    if w3 > 0.0:
        x[i3] = x[i3] + limit_correction(g3 * (w3 * dlam), ratio * radius[i3])


@wp.kernel
def solve_tet_hydrostatic(x: wp.array(dtype=wp.vec3),
                          w: wp.array(dtype=float),
                          radius: wp.array(dtype=float),
                          idx: wp.array(dtype=wp.vec4i),
                          dm_inv: wp.array(dtype=wp.mat33),
                          volume: wp.array(dtype=float),
                          wsum_rest: wp.array(dtype=float),
                          body: wp.array(dtype=wp.int32),
                          lam: wp.array(dtype=float),
                          mat_dev: wp.array(dtype=float),
                          mat_hyd: wp.array(dtype=float),
                          params: wp.array(dtype=float),
                          offset: int,
                          h: float) -> None:
    t = wp.tid() + offset
    e = idx[t]
    i0 = e[0]
    i1 = e[1]
    i2 = e[2]
    i3 = e[3]
    w0 = w[i0]
    w1 = w[i1]
    w2 = w[i2]
    w3 = w[i3]
    if w0 + w1 + w2 + w3 <= 0.0:
        return

    dmi = dm_inv[t]
    f = deformation_gradient(x, i0, i1, i2, i3, dmi)
    f0 = column(f, 0)
    f1 = column(f, 1)
    f2 = column(f, 2)

    c1 = wp.cross(f1, f2)
    c2 = wp.cross(f2, f0)
    c3 = wp.cross(f0, f1)
    grad_f = wp.mat33(c1[0], c2[0], c3[0],
                      c1[1], c2[1], c3[1],
                      c1[2], c2[2], c3[2])
    g = grad_f * wp.transpose(dmi)
    g1 = column(g, 0)
    g2 = column(g, 1)
    g3 = column(g, 2)
    g0 = -(g1 + g2 + g3)

    wsum = (w0 * wp.dot(g0, g0) + w1 * wp.dot(g1, g1)
            + w2 * wp.dot(g2, g2) + w3 * wp.dot(g3, g3))
    if wsum <= 0.0:
        return

    b = body[t]
    # gamma = 1 + mu/lambda.  mat_dev is 1/mu and mat_hyd is 1/lambda, so the
    # ratio inverts.  This offset is the only reason the rest pose is
    # force-free: it is what cancels the deviatoric term's non-zero rest
    # force.  With gamma = 1 instead, a cube settles at 0.91 to 0.96 of its
    # rest volume depending on hardness, measured, against 0.997 with it.
    # resolvable_softening scales mu and lambda by the same factor, which
    # leaves this ratio -- and therefore the cancellation -- untouched.
    gamma = 1.0 + mat_hyd[b] / mat_dev[b]
    c = wp.determinant(f) - gamma

    inv_vh2 = 1.0 / (volume[t] * h * h)
    a_tilde = mat_hyd[b] * inv_vh2 / resolvable_softening(
        mat_dev[b] * inv_vh2, mat_hyd[b] * inv_vh2, wsum_rest[t])
    dlam = (-c - a_tilde * lam[t]) / (wsum + a_tilde)
    lam[t] = lam[t] + dlam

    ratio = params[PARAM_MAX_CORRECTION_RATIO]
    if w0 > 0.0:
        x[i0] = x[i0] + limit_correction(g0 * (w0 * dlam), ratio * radius[i0])
    if w1 > 0.0:
        x[i1] = x[i1] + limit_correction(g1 * (w1 * dlam), ratio * radius[i1])
    if w2 > 0.0:
        x[i2] = x[i2] + limit_correction(g2 * (w2 * dlam), ratio * radius[i2])
    if w3 > 0.0:
        x[i3] = x[i3] + limit_correction(g3 * (w3 * dlam), ratio * radius[i3])


# ---------------------------------------------------------------------------
# 5. grab attachment
# ---------------------------------------------------------------------------


@wp.kernel
def solve_grab(x: wp.array(dtype=wp.vec3),
               w: wp.array(dtype=float),
               radius: wp.array(dtype=float),
               body: wp.array(dtype=wp.int32),
               grab_particle: wp.array(dtype=wp.int32),
               grab_local: wp.array(dtype=wp.vec3),
               grab_hand: wp.array(dtype=wp.int32),
               grab_lambda: wp.array(dtype=float),
               hand_pos: wp.array(dtype=wp.vec3),
               hand_rot: wp.array(dtype=wp.quat),
               hand_active: wp.array(dtype=wp.int32),
               mat_grab: wp.array(dtype=float),
               params: wp.array(dtype=float),
               h: float) -> None:
    t = wp.tid()
    p = grab_particle[t]
    if p < 0:
        return
    slot = grab_hand[t]
    if slot < 0:
        return
    if hand_active[slot] == 0:
        return

    wi = w[p]
    if wi <= 0.0:
        return

    local = grab_local[t]
    if params[PARAM_GRAB_ROTATE] <= 0.0:
        target = hand_pos[slot] + local
    else:
        target = hand_pos[slot] + wp.quat_rotate(hand_rot[slot], local)

    d = x[p] - target
    c = wp.length(d)
    if c < _EPS:
        return
    n = d / c

    a_tilde = mat_grab[body[p]] / (h * h)
    dlam = (-c - a_tilde * grab_lambda[t]) / (wi + a_tilde)
    grab_lambda[t] = grab_lambda[t] + dlam

    # A compliant pull, not an assignment.  Assigning the target position lets
    # a hand that jumps 20 cm between frames teleport the held particles
    # through the floor and through whatever else is in the way; the compliant
    # form leaves the collision constraints a chance to win.
    x[p] = x[p] + limit_correction(n * (wi * dlam),
                                   params[PARAM_MAX_CORRECTION_RATIO] * radius[p])


# ---------------------------------------------------------------------------
# 6. collision
# ---------------------------------------------------------------------------


@wp.kernel
def solve_capsules(x: wp.array(dtype=wp.vec3),
                   x_prev: wp.array(dtype=wp.vec3),
                   v: wp.array(dtype=wp.vec3),
                   w: wp.array(dtype=float),
                   flags: wp.array(dtype=wp.uint32),
                   radius: wp.array(dtype=float),
                   body: wp.array(dtype=wp.int32),
                   contact_n: wp.array(dtype=wp.vec3),
                   contact_vn: wp.array(dtype=float),
                   cap_a: wp.array(dtype=wp.vec3),
                   cap_b: wp.array(dtype=wp.vec3),
                   cap_a_prev: wp.array(dtype=wp.vec3),
                   cap_b_prev: wp.array(dtype=wp.vec3),
                   cap_r: wp.array(dtype=float),
                   cap_r_prev: wp.array(dtype=float),
                   cap_va: wp.array(dtype=wp.vec3),
                   cap_vb: wp.array(dtype=wp.vec3),
                   cap_count: wp.array(dtype=wp.int32),
                   mat_friction: wp.array(dtype=float),
                   mat_restitution: wp.array(dtype=float),
                   contact_total: wp.array(dtype=wp.int32),
                   params: wp.array(dtype=float),
                   frac: float,
                   h: float) -> None:
    """Push particles out of the hand capsules and rub them along.

    ``frac`` is how far through the frame this substep ends, so the capsule
    is swept from where it was when the frame began to where the tracker says
    it is now.  It is a compile-time constant per unrolled substep -- the
    substep count never changes at runtime -- so a captured CUDA graph is
    safe with it.  Without the sweep the capsule stands still for eleven
    substeps and jumps on the twelfth: a hand crossing the stage at 1.2 m/s
    moves 13 mm between frames, and paying that whole 13 mm off in one 0.93
    ms substep is 14 m/s of separation out of a hand that is moving at 1.2.
    Measured on the grain pile: peak grain speed tracked the per-frame
    capsule jump, not the hand speed -- 2.5 m/s at 3.3 mm/frame, 10.7 m/s at
    13.3 mm/frame, and the same 1.2 m/s hand at half the frame length gave
    half the peak.
    """
    t = wp.tid()
    wi = w[t]
    if wi <= 0.0:
        return
    # Held matter is positioned by its grab; the fingers holding it must not
    # also collide with it.  The grab reaches every particle within its
    # radius of the pinch point, and the fingertip capsules sit exactly
    # there, so some of what is held starts *inside* a capsule.  Opening the
    # fingers then sweeps those capsules through the held patch, and a
    # particle pushed out of a moving capsule leaves at the correction clamp
    # -- three radii per substep, 15 m/s -- not at the finger's speed.
    # Measured: a ball let go by a perfectly still wrist left at 0.9 m/s.
    if (flags[t] & FLAG_GRABBED_BIT) != wp.uint32(0):
        return

    n_caps = cap_count[0]
    if n_caps <= 0:
        return

    bi = body[t]
    mu = mat_friction[bi]
    restitution = mat_restitution[bi]
    ri = radius[t]
    max_corr = params[PARAM_MAX_CORRECTION_RATIO] * ri
    pos = x[t]
    prev = x_prev[t]
    vel = v[t]
    hit = int(0)
    best_n = wp.vec3(0.0, 0.0, 0.0)
    best_vn = float(0.0)

    c = int(0)
    for c in range(n_caps):
        rc = cap_r_prev[c] + (cap_r[c] - cap_r_prev[c]) * frac
        if rc <= 0.0:
            continue
        a = cap_a_prev[c] + (cap_a[c] - cap_a_prev[c]) * frac
        b = cap_b_prev[c] + (cap_b[c] - cap_b_prev[c]) * frac
        ab = b - a
        dd = wp.dot(ab, ab)
        s = float(0.0)
        if dd > 1.0e-12:
            s = wp.clamp(wp.dot(pos - a, ab) / dd, 0.0, 1.0)
        q = a + ab * s
        delta = pos - q
        dist = wp.length(delta)
        total = rc + ri
        if dist >= total:
            continue

        nrm = _UP
        if dist > _EPS:
            nrm = delta / dist
        depth = total - dist

        vq = cap_va[c] * (1.0 - s) + cap_vb[c] * s
        correction = nrm * depth

        # Tangential Coulomb friction against the capsule's own motion at the
        # contact point.  Without it a hand can only push along its surface
        # normal, so sliding a palm sideways under a sheet does nothing and
        # half of the interaction the project promises disappears.
        rel = (pos - prev) - vq * h
        tangent = rel - nrm * wp.dot(rel, nrm)
        t_len = wp.length(tangent)
        if t_len > _EPS:
            slide = wp.min(t_len, mu * depth)
            correction = correction - tangent * (slide / t_len)

        # A speed limit, not a distance clamp.  A particle that starts a
        # substep well inside a capsule -- matter caught between closing
        # fingers, the neighbour of a held patch -- used to be pushed to the
        # surface in one go, up to three radii per substep, and finalize read
        # that as velocity: fingers opening at 0.3 m/s ejected free particles
        # at 2.9.  A capsule can hand out its own speed, with slack for the
        # sweep, plus a small settling rate so a *still* hand with matter
        # resting inside it still clears itself over a frame or two.
        allowed = wp.min(max_corr,
                         wp.max(wp.length(vq) * h * _CAPSULE_PUSH_SLACK,
                                _CAPSULE_SETTLE * ri))
        pos = pos + limit_correction(correction, allowed)

        approach = wp.dot(vel - vq, nrm)
        if approach < 0.0:
            target = wp.dot(vq, nrm) - restitution * approach
            if hit == 0 or target > best_vn:
                best_n = nrm
                best_vn = target
            hit = int(1)

    x[t] = pos
    if hit != 0:
        # Several bones can touch one particle; keep the one that demands the
        # most separation so a finger pressed into a palm still throws the
        # particle clear instead of averaging itself away.
        contact_n[t] = best_n
        contact_vn[t] = best_vn
        wp.atomic_add(contact_total, 0, 1)


@wp.kernel
def solve_ground(x: wp.array(dtype=wp.vec3),
                 x_prev: wp.array(dtype=wp.vec3),
                 v: wp.array(dtype=wp.vec3),
                 w: wp.array(dtype=float),
                 radius: wp.array(dtype=float),
                 body: wp.array(dtype=wp.int32),
                 contact_n: wp.array(dtype=wp.vec3),
                 contact_vn: wp.array(dtype=float),
                 mat_friction: wp.array(dtype=float),
                 mat_restitution: wp.array(dtype=float),
                 contact_total: wp.array(dtype=wp.int32),
                 params: wp.array(dtype=float),
                 h: float) -> None:
    t = wp.tid()
    if w[t] <= 0.0:
        return

    pos = x[t]
    ri = radius[t]
    bi = body[t]

    # The basin wall.  Granular matter on an open floor does not make a heap
    # you can stir: with position-based friction the angle of repose comes out
    # near 7 degrees instead of a real material's 30-something, so a pile
    # spreads into a sheet and slides off the stage.  A shallow cylinder holds
    # it in reach without pretending the friction model is better than it is,
    # and because the wall stops at basin_height a hand can still lift matter
    # straight out of it.
    basin = params[PARAM_BASIN_RADIUS]
    if basin > 0.0 and pos[1] < params[PARAM_GROUND_Y] + params[PARAM_BASIN_HEIGHT]:
        radial = wp.vec2(pos[0], pos[2])
        dist = wp.length(radial)
        limit = basin - ri
        if dist > limit and dist > _EPS:
            inward = -radial / dist
            # The wall may always cancel the outward motion of this substep in
            # full: undoing a displacement cannot inject energy, it leaves the
            # radial velocity at zero, and that is what makes a wall a wall
            # however fast a hand ploughs into it.  What it may *not* do is
            # repay an older excursion at the same rate.  A grain carried over
            # the lip and dropped below it is 100 mm outside through no motion
            # of its own; snapping that back costs one substep, finalize reads
            # the displacement as 100 mm / 0.93 ms and the basin fires it
            # across the room at the velocity clamp -- measured at 6.4 m and
            # 490 grains past a metre, held there for two seconds.  A clamp
            # in radii does not help: one radius is still 5 m/s.  So the debt
            # is walked home at a fixed speed instead, which for anything the
            # settled pile does on its own is larger than the excursion and
            # therefore no change at all.  ``went_out`` is signed, not a
            # magnitude, and that is what makes the bound a speed limit
            # rather than an acceleration: a particle already drifting in at
            # the recovery speed asks for no further push.
            prev = x_prev[t]
            went_out = dist - wp.length(wp.vec2(prev[0], prev[2]))
            push = wp.min(dist - limit, went_out + _BASIN_RECOVERY_SPEED * h)
            if push > 0.0:
                x[t] = pos + wp.vec3(inward[0] * push, 0.0, inward[1] * push)
                pos = x[t]
            wp.atomic_add(contact_total, 0, 1)

    floor = params[PARAM_GROUND_Y] + ri
    depth = floor - pos[1]
    if depth <= 0.0:
        return

    # Geometric mean is the usual way to combine two Coulomb coefficients: it
    # keeps a frictionless surface frictionless whatever the other material is.
    mu = wp.sqrt(wp.max(params[PARAM_GROUND_FRICTION] * mat_friction[bi], 0.0))

    correction = wp.vec3(0.0, depth, 0.0)
    rel = pos - x_prev[t]
    tangent = wp.vec3(rel[0], 0.0, rel[2])
    t_len = wp.length(tangent)
    if t_len > _EPS:
        slide = wp.min(t_len, mu * depth)
        correction = correction - tangent * (slide / t_len)

    # One substep may not lift a particle further than its own radius.
    # max_correction_ratio is 3.0, i.e. three radii, and under a settling pile
    # the particles at the bottom are pushed deep enough to ask for all of it
    # -- for the grain preset's 4.8 mm radius at h under a millisecond that is
    # 15 m/s of implied velocity out of the floor, and a granular pour lands
    # by firing its bottom layer across the stage.
    x[t] = pos + limit_correction(
        correction, wp.min(params[PARAM_MAX_CORRECTION_RATIO] * ri, ri))

    approach = v[t][1]
    if approach < 0.0:
        target = -mat_restitution[bi] * approach
        if target > contact_vn[t]:
            contact_n[t] = _UP
            contact_vn[t] = target
    wp.atomic_add(contact_total, 0, 1)


@wp.kernel
def solve_particle_contacts(x: wp.array(dtype=wp.vec3),
                            x_prev: wp.array(dtype=wp.vec3),
                            dx: wp.array(dtype=wp.vec3),
                            w: wp.array(dtype=float),
                            radius: wp.array(dtype=float),
                            collide_radius: wp.array(dtype=float),
                            flags: wp.array(dtype=wp.uint32),
                            body: wp.array(dtype=wp.int32),
                            mat_friction: wp.array(dtype=float),
                            contact_total: wp.array(dtype=wp.int32),
                            grid: wp.uint64,
                            params: wp.array(dtype=float),
                            query_radius: float) -> None:
    """Stage every particle's self-collision correction, applying none of it.

    This kernel reads its neighbours' positions, so it must not write anybody's
    position, not even its own: a thread that has already moved would be seen
    in its new place by a neighbour still being solved, and the result would
    depend on which warp happened to run first.  That is a silent race -- it
    stays under 1e-6 for a minute, which is exactly long enough to be blamed on
    the CUDA graph -- so the correction is staged here and applied by
    :func:`apply_corrections` once every thread has read what it needs.
    """
    t = wp.tid()
    dx[t] = wp.vec3(0.0, 0.0, 0.0)
    if params[PARAM_SELF_COLLIDE] <= 0.0:
        return
    wi = w[t]
    if wi <= 0.0:
        return

    flag_i = flags[t]
    pos = x[t]
    disp = pos - x_prev[t]
    ri = collide_radius[t]
    margin = 1.0 + params[PARAM_COLLISION_MARGIN]
    mu = mat_friction[body[t]]

    total = wp.vec3(0.0, 0.0, 0.0)
    hit = int(0)

    query = wp.hash_grid_query(grid, pos, query_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == t:
            continue
        # Both ends have to opt in.  With an OR here a soft body's interior
        # particles -- which its builder deliberately leaves unflagged -- get
        # dragged into contact by the surface layer one lattice cell away,
        # where they sit permanently touching and fight the tetrahedra all
        # frame.  Measured on the `soft` preset at hardness 0: OR ends a
        # three-second settle at 2.1 J and pegged at the velocity clamp, AND
        # ends it at 3e-5 J and still falling (2.5e-7 J by four seconds).
        if ((flag_i & flags[j]) & FLAG_SELF_COLLIDE_BIT) == wp.uint32(0):
            continue
        wj = w[j]
        wsum = wi + wj
        if wsum <= 0.0:
            continue

        delta = pos - x[j]
        dist = wp.length(delta)
        rest = (ri + collide_radius[j]) * margin
        if dist >= rest:
            continue

        nrm = _UP
        if dist > _EPS:
            nrm = delta / dist
        depth = rest - dist
        # Each thread applies only its own mass-weighted half; the neighbour's
        # thread applies the mirror image.  Writing both sides here would race
        # and would also make the result depend on thread scheduling.
        share = wi / wsum
        correction = nrm * (depth * share)

        rel = disp - (x[j] - x_prev[j])
        tangent = rel - nrm * wp.dot(rel, nrm)
        t_len = wp.length(tangent)
        if t_len > _EPS:
            slide = wp.min(t_len, mu * depth)
            correction = correction - tangent * (slide / t_len) * share

        total = total + correction
        hit += 1

    if hit != 0:
        # Average, do not sum.  Each neighbour independently asks to be pushed
        # clear, and inside a dense pile a grain has a dozen of them asking at
        # once; adding those up moves it about a dozen times further than any
        # single contact justified, which is why a settling pile used to fling
        # grains out of the box and why more substeps made it worse rather
        # than better.  Averaging makes this a Jacobi iteration, which
        # converges over the substep loop instead of overshooting
        # (Macklin et al., "Unified Particle Physics", 2014).
        n = float(hit)
        sor = 1.0 + (_CONTACT_SOR - 1.0) * (n - 1.0) / n
        total = total * (sor / n)
        # One substep of contact response may not move a particle further than
        # its own contact sphere.  max_correction_ratio alone is three times
        # the *render* radius, which for the grain preset is 14 mm -- at h
        # under a millisecond that is a 15 m/s separation the contact never
        # earned, and since the grid is rebuilt once a step rather than once a
        # substep, deep penetrations that justify no such speed do turn up.
        dx[t] = limit_correction(
            total, wp.min(params[PARAM_MAX_CORRECTION_RATIO] * radius[t],
                          collide_radius[t]))
        wp.atomic_add(contact_total, 0, 1)


@wp.kernel
def apply_corrections(x: wp.array(dtype=wp.vec3),
                      dx: wp.array(dtype=wp.vec3)) -> None:
    t = wp.tid()
    x[t] = x[t] + dx[t]


# ---------------------------------------------------------------------------
# 7. finalize
# ---------------------------------------------------------------------------


@wp.kernel
def finalize(x: wp.array(dtype=wp.vec3),
             x_prev: wp.array(dtype=wp.vec3),
             v: wp.array(dtype=wp.vec3),
             w: wp.array(dtype=float),
             body: wp.array(dtype=wp.int32),
             contact_n: wp.array(dtype=wp.vec3),
             contact_vn: wp.array(dtype=float),
             mat_damping: wp.array(dtype=float),
             params: wp.array(dtype=float),
             h: float) -> None:
    t = wp.tid()
    if w[t] <= 0.0:
        v[t] = wp.vec3(0.0, 0.0, 0.0)
        return

    vel = (x[t] - x_prev[t]) / h

    nrm = contact_n[t]
    if wp.dot(nrm, nrm) > 0.5:
        vn = wp.dot(vel, nrm)
        target = contact_vn[t]
        if target > vn:
            vel = vel + nrm * (target - vn)

    # Exponential decay, not a per-substep multiply.  A multiply of the form
    # v *= (1 - damping) compounds once per substep, so raising the substep
    # count would quietly make every material softer-looking -- the one thing
    # the compliance formulation exists to prevent.
    vel = vel * wp.exp(-mat_damping[body[t]] * h)

    speed = wp.length(vel)
    vmax = params[PARAM_MAX_VELOCITY]
    if speed > vmax and speed > _EPS:
        vel = vel * (vmax / speed)

    v[t] = vel


# ---------------------------------------------------------------------------
# shading normals
# ---------------------------------------------------------------------------


@wp.kernel
def zero_vec3(a: wp.array(dtype=wp.vec3)) -> None:
    a[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def accumulate_normals(x: wp.array(dtype=wp.vec3),
                       tri: wp.array(dtype=wp.vec3i),
                       normal: wp.array(dtype=wp.vec3)) -> None:
    t = wp.tid()
    f = tri[t]
    i0 = f[0]
    i1 = f[1]
    i2 = f[2]
    # The unnormalised cross product has twice the triangle's area as its
    # length, so summing it is already an area-weighted average; normalising
    # per triangle first would let a sliver face count as much as the slab
    # next to it and put visible creases in a smooth surface.
    n = wp.cross(x[i1] - x[i0], x[i2] - x[i0])
    wp.atomic_add(normal, i0, n)
    wp.atomic_add(normal, i1, n)
    wp.atomic_add(normal, i2, n)


@wp.kernel
def normalize_normals(normal: wp.array(dtype=wp.vec3)) -> None:
    t = wp.tid()
    n = normal[t]
    length = wp.length(n)
    if length > 1.0e-12:
        normal[t] = n / length
    else:
        normal[t] = _UP


# ---------------------------------------------------------------------------
# grab bookkeeping
# ---------------------------------------------------------------------------


@wp.kernel
def apply_grab(selected: wp.array(dtype=wp.int32),
               local: wp.array(dtype=wp.vec3),
               grab_particle: wp.array(dtype=wp.int32),
               grab_local: wp.array(dtype=wp.vec3),
               grab_hand: wp.array(dtype=wp.int32),
               w: wp.array(dtype=float),
               w_rest: wp.array(dtype=float),
               flags: wp.array(dtype=wp.uint32),
               slot: int,
               base: int,
               inv_mass_scale: float) -> None:
    t = wp.tid()
    p = selected[t]
    grab_particle[base + t] = p
    grab_local[base + t] = local[t]
    grab_hand[base + t] = slot
    # Lighter while held, so the compliant attachment wins against the body's
    # own constraints and the held region tracks the hand instead of lagging.
    w[p] = w_rest[p] * inv_mass_scale
    flags[p] = flags[p] | FLAG_GRABBED_BIT


@wp.kernel
def release_grab(grab_particle: wp.array(dtype=wp.int32),
                 grab_hand: wp.array(dtype=wp.int32),
                 grab_lambda: wp.array(dtype=float),
                 w: wp.array(dtype=float),
                 w_rest: wp.array(dtype=float),
                 v: wp.array(dtype=wp.vec3),
                 flags: wp.array(dtype=wp.uint32),
                 hand_vel: wp.array(dtype=wp.vec3),
                 slot: int,
                 throw_gain: float,
                 keep_velocity: int) -> None:
    t = wp.tid()
    if grab_hand[t] != slot:
        return
    p = grab_particle[t]
    if p < 0:
        return
    w[p] = w_rest[p]
    flags[p] = flags[p] & (~FLAG_GRABBED_BIT)
    # A deliberate release hands the hand's velocity over, so a flick throws.
    # A release forced by the tracker losing the hand must not: there is no
    # trustworthy hand velocity to hand over, and stamping one on would either
    # freeze the matter mid-fall or fling it.
    if keep_velocity == 0:
        v[p] = hand_vel[slot] * throw_gain
    grab_particle[t] = -1
    grab_hand[t] = -1
    grab_lambda[t] = 0.0


@wp.kernel
def count_grabs(grab_particle: wp.array(dtype=wp.int32),
                grab_count: wp.array(dtype=wp.int32)) -> None:
    t = wp.tid()
    if grab_particle[t] >= 0:
        wp.atomic_add(grab_count, 0, 1)


# ---------------------------------------------------------------------------
# sanity sweep
# ---------------------------------------------------------------------------


@wp.kernel
def contain_particles(x: wp.array(dtype=wp.vec3),
                      x_prev: wp.array(dtype=wp.vec3),
                      x_rest: wp.array(dtype=wp.vec3),
                      v: wp.array(dtype=wp.vec3),
                      body: wp.array(dtype=wp.int32),
                      bad: wp.array(dtype=wp.int32),
                      limit: float) -> None:
    """Snap any particle that has left the world back to rest, and flag it.

    This has to run before the hash grid is rebuilt, and it has to run every
    step, because ``wp.HashGrid.build`` divides each point by the cell size and
    casts to an integer: hand it an infinity, or merely a coordinate past about
    1e7 m at these cell sizes, and the build writes outside its cell table.
    That is an illegal memory access, which on CUDA is not an exception the app
    can catch -- the context dies and every later allocation in the process
    fails.  Verified: a single +inf among a thousand sane points is enough.

    The periodic per-body sweep of ARCHITECTURE.md 7.8 cannot cover this on its
    own.  It runs after the step and only every ``sanity_interval`` steps, so a
    body that diverges is fed to the next ``build`` long before the sweep looks
    at it.  This is the same check moved in front of the grid and made
    unconditional; it writes ``bad`` for the sweep to act on and costs one
    read-mostly pass over the particles.
    """
    t = wp.tid()
    p = x[t]
    u = v[t]
    s = p[0] + p[1] + p[2] + u[0] + u[1] + u[2]
    # s != s catches NaN; the magnitude test catches an infinity and also a
    # body that is on its way to one, which is worth resetting just as early.
    if s == s and wp.abs(p[0]) <= limit and wp.abs(p[1]) <= limit \
            and wp.abs(p[2]) <= limit and wp.abs(s) <= limit * 8.0:
        return
    wp.atomic_max(bad, body[t], 1)
    r = x_rest[t]
    x[t] = r
    x_prev[t] = r
    v[t] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def reset_flagged(x: wp.array(dtype=wp.vec3),
                  x_prev: wp.array(dtype=wp.vec3),
                  x_rest: wp.array(dtype=wp.vec3),
                  v: wp.array(dtype=wp.vec3),
                  w: wp.array(dtype=float),
                  w_rest: wp.array(dtype=float),
                  flags: wp.array(dtype=wp.uint32),
                  body: wp.array(dtype=wp.int32),
                  bad: wp.array(dtype=wp.int32)) -> None:
    t = wp.tid()
    if bad[body[t]] == 0:
        return
    r = x_rest[t]
    x[t] = r
    x_prev[t] = r
    v[t] = wp.vec3(0.0, 0.0, 0.0)
    w[t] = w_rest[t]
    flags[t] = flags[t] & (~FLAG_GRABBED_BIT)


@wp.kernel
def clear_int(a: wp.array(dtype=wp.int32)) -> None:
    a[wp.tid()] = 0


@wp.kernel
def apply_skin(x: wp.array(dtype=wp.vec3),
               tet_idx: wp.array(dtype=wp.vec4i),
               skin_tet: wp.array(dtype=wp.int32),
               skin_bary: wp.array(dtype=wp.vec4),
               blend: int,
               out: wp.array(dtype=wp.vec3)) -> None:
    """Place each particle's *drawn* position by blending its bound elements.

    Physics and rendering want different points.  A voxelised soft body has to
    keep its lattice to stay well conditioned, and has to show the isosurface
    to look like the shape it is meant to be; binding the second to the first
    with barycentric weights gets both.  Each map is affine per element, so
    the drawn skin follows every stretch, shear and rotation of the elements
    it sits in.  Blending over several keeps the deformation gradient
    continuous across element boundaries -- one element alone shows its seam
    as a crease -- and the weights were pre-scaled at build time, so this is
    a plain sum.  Slots are read in order and stop at the first -1.
    """
    t = wp.tid()
    base = t * blend
    if skin_tet[base] < 0:
        out[t] = x[t]
        return
    acc = wp.vec3(0.0, 0.0, 0.0)
    k = int(0)
    while k < blend:
        e = skin_tet[base + k]
        if e < 0:
            break
        v = tet_idx[e]
        w = skin_bary[base + k]
        acc = acc + (x[v[0]] * w[0] + x[v[1]] * w[1]
                     + x[v[2]] * w[2] + x[v[3]] * w[3])
        k += 1
    out[t] = acc
