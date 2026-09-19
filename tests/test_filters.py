"""The One-Euro filter must smooth without lagging.

Both halves of that are load-bearing.  Too little smoothing and the hand
capsules vibrate, which injects energy into the cloth and reads to a viewer
as unstable physics.  Too much and the matter answers a hand that is no
longer there, which reads as the whole thing being fake.
"""

from __future__ import annotations

import sys

import numpy as np
from _harness import case, note, run

from fctx.config import TrackingConfig
from fctx.hands.filters import ExponentialFilter, OneEuroFilter, VelocityEstimator

CFG = TrackingConfig()
DT = 1.0 / 60.0


def _fresh() -> OneEuroFilter:
    return OneEuroFilter(CFG.filter_min_cutoff, CFG.filter_beta, CFG.filter_d_cutoff)


def _landmarks(value: float = 0.0) -> np.ndarray:
    return np.full((21, 3), value, dtype=np.float32)


@case
def test_first_sample_passes_through() -> None:
    f = _fresh()
    x = np.random.default_rng(0).normal(size=(21, 3)).astype(np.float32)
    out = f.filter(x, DT)
    assert np.allclose(out, x), "the first sample has nothing to blend with"
    assert f.initialized


@case
def test_converges_to_a_constant() -> None:
    f = _fresh()
    target = _landmarks(0.25)
    for _ in range(300):
        out = f.filter(target, DT)
    assert np.max(np.abs(out - target)) < 1e-5, np.max(np.abs(out - target))


def _noise_ratio(f: OneEuroFilter, sigma: float = 0.002, n: int = 900) -> float:
    rng = np.random.default_rng(12345)
    noisy, clean = [], []
    for _ in range(n):
        sample = _landmarks(0.1) + rng.normal(scale=sigma, size=(21, 3))
        noisy.append(sample)
        clean.append(f.filter(sample.astype(np.float32), DT))
    burn = 150
    noise_in = float(np.std(np.stack(noisy)[burn:] - 0.1))
    noise_out = float(np.std(np.stack(clean)[burn:] - 0.1))
    return noise_in / max(noise_out, 1e-12)


@case
def test_suppresses_noise_while_still() -> None:
    """A still hand must come out still, and by the amount theory predicts.

    A first-order low-pass has a closed-form noise gain of
    ``sqrt(a / (2 - a))``, so the measured suppression is checked against
    that rather than against a round number: this catches an implementation
    bug (wrong alpha, derivative fed back into the position path) instead of
    merely restating how the cutoff happens to be tuned today.
    """
    from fctx.hands.filters import smoothing_factor

    ratio = _noise_ratio(_fresh())
    alpha = float(smoothing_factor(DT, CFG.filter_min_cutoff))
    expected = 1.0 / np.sqrt(alpha / (2.0 - alpha))
    note(f"one-euro noise suppression: {ratio:.2f}x measured, "
         f"{expected:.2f}x predicted at {CFG.filter_min_cutoff} Hz / "
         f"{1 / DT:.0f} Hz")
    assert ratio > 3.0, f"only suppressed noise {ratio:.2f}x"
    assert abs(ratio - expected) / expected < 0.15, (
        f"measured {ratio:.2f}x against a predicted {expected:.2f}x")


@case
def test_lower_cutoff_suppresses_more() -> None:
    gentle = _noise_ratio(OneEuroFilter(0.6, CFG.filter_beta, CFG.filter_d_cutoff))
    sharp = _noise_ratio(OneEuroFilter(6.0, CFG.filter_beta, CFG.filter_d_cutoff))
    assert gentle > 2.0 * sharp, (
        f"cutoff barely matters: {gentle:.2f}x at 0.6 Hz vs {sharp:.2f}x at 6 Hz")


def _ramp_lag(f: OneEuroFilter, speed: float) -> float:
    lag = []
    for i in range(500):
        truth = speed * i * DT
        out = f.filter(_landmarks(truth), DT)
        if i > 150:
            lag.append(truth - float(out[0, 0]))
    return float(np.mean(lag))


@case
def test_tracks_a_ramp_with_bounded_lag() -> None:
    """Lag on a constant-velocity input settles; it does not accumulate."""
    speed = 0.8  # m/s, a brisk hand movement
    f = _fresh()
    early, late = [], []
    for i in range(700):
        truth = speed * i * DT
        out = f.filter(_landmarks(truth), DT)
        if 150 <= i < 250:
            early.append(truth - float(out[0, 0]))
        elif i >= 600:
            late.append(truth - float(out[0, 0]))
    assert 0.0 < float(np.mean(early)) < 0.15
    assert abs(float(np.mean(late)) - float(np.mean(early))) < 1e-3, (
        "lag is still growing, so the filter is integrating rather than tracking")


@case
def test_beta_buys_back_lag_on_motion() -> None:
    """The adaptive term is the reason to use One-Euro at all.

    With ``beta = 0`` this degenerates to a fixed low-pass and the lag is
    whatever ``min_cutoff`` dictates; raising beta must measurably shorten it
    while leaving the still-hand behaviour alone.
    """
    speed = 1.2
    fixed = _ramp_lag(OneEuroFilter(CFG.filter_min_cutoff, 0.0,
                                    CFG.filter_d_cutoff), speed)
    adaptive = _ramp_lag(OneEuroFilter(CFG.filter_min_cutoff, 6.0,
                                       CFG.filter_d_cutoff), speed)
    note(f"lag at {speed} m/s: {fixed * 1e3:.1f} mm with beta=0, "
         f"{adaptive * 1e3:.1f} mm with beta=6, "
         f"{_ramp_lag(_fresh(), speed) * 1e3:.1f} mm with the configured "
         f"beta={CFG.filter_beta}")
    assert adaptive < 0.35 * fixed, (
        f"beta hardly helped: {adaptive * 1e3:.1f} mm against "
        f"{fixed * 1e3:.1f} mm")
    assert _noise_ratio(OneEuroFilter(CFG.filter_min_cutoff, 6.0,
                                      CFG.filter_d_cutoff)) > 2.5


@case
def test_step_passes_faster_than_noise() -> None:
    """The adaptive cutoff is the whole point: a real movement must get
    through in fewer frames than a noise burst of the same amplitude."""
    step = 0.05
    fast = _fresh()
    for _ in range(60):
        fast.filter(_landmarks(0.0), DT)
    for _ in range(6):
        out_step = fast.filter(_landmarks(step), DT)

    rng = np.random.default_rng(7)
    slow = _fresh()
    for _ in range(60):
        slow.filter(_landmarks(0.0), DT)
    burst = []
    for _ in range(6):
        sign = 1.0 if rng.random() < 0.5 else -1.0
        burst.append(slow.filter(_landmarks(sign * step), DT))
    travelled = float(out_step[0, 0]) / step
    wandered = float(np.max(np.abs(np.stack(burst)))) / step
    assert travelled > wandered, (
        f"step reached {travelled:.2f} of its target while noise reached "
        f"{wandered:.2f}")


@case
def test_survives_dt_jitter_and_zero() -> None:
    rng = np.random.default_rng(99)
    f = _fresh()
    target = _landmarks(0.3)
    out = f.filter(target, DT)
    for i in range(500):
        dt = 0.0 if i % 17 == 0 else float(rng.uniform(0.001, 0.09))
        out = f.filter(target, dt)
        assert np.isfinite(out).all(), f"non-finite output at step {i}"
    assert np.max(np.abs(out - target)) < 1e-4


@case
def test_zero_dt_is_a_no_op() -> None:
    f = _fresh()
    f.filter(_landmarks(0.0), DT)
    a = f.filter(_landmarks(0.0), DT)
    b = f.filter(_landmarks(1.0), 0.0)
    assert np.array_equal(a, b), "a zero-length step must not advance the filter"


@case
def test_rejects_bad_input() -> None:
    f = _fresh()
    for bad in (np.full((21, 3), np.nan, np.float32),
                np.full((21, 3), np.inf, np.float32)):
        try:
            f.filter(bad, DT)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite input was accepted")
    try:
        f.filter(_landmarks(0.0), -0.01)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative dt was accepted")
    f.filter(_landmarks(0.0), DT)
    try:
        f.filter(np.zeros((5, 3), np.float32), DT)
    except ValueError:
        pass
    else:
        raise AssertionError("a shape change was accepted")


@case
def test_a_non_finite_dt_is_rejected_rather_than_clamped() -> None:
    """``min(nan, MAX_DT)`` is ``nan``.

    Clamping alone would therefore let a NaN time step through into the
    smoothing alpha and from there into the filter state, where it is
    permanent: every later output of that filter is NaN, and the traceback
    lands in whichever consumer first checks, frames after the cause.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        euro = _fresh()
        euro.filter(_landmarks(0.0), DT)
        exponential = ExponentialFilter(5.0)
        exponential.filter(0.0, DT)
        velocity = VelocityEstimator()
        velocity.update(_landmarks(0.0), DT)
        attempts = (
            ("OneEuroFilter.filter", lambda: euro.filter(_landmarks(1.0), bad)),
            ("ExponentialFilter.filter", lambda: exponential.filter(1.0, bad)),
            ("VelocityEstimator.update", lambda: velocity.update(_landmarks(1.0), bad)),
            ("VelocityEstimator.extrapolate",
             lambda: velocity.extrapolate(_landmarks(1.0), bad)),
            ("VelocityEstimator.decay", lambda: velocity.decay(bad)),
        )
        for name, call in attempts:
            try:
                call()
            except ValueError:
                continue
            raise AssertionError(f"{name} accepted dt={bad}")
        assert np.isfinite(euro.value).all(), "the rejected step still poisoned the state"
        assert np.isfinite(euro.filter(_landmarks(1.0), DT)).all()
        assert np.isfinite(velocity.update(_landmarks(1.0), DT)).all()

    negative = VelocityEstimator()
    negative.update(_landmarks(0.0), DT)
    try:
        negative.decay(-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative decay factor was accepted")


@case
def test_reset_forgets_everything() -> None:
    f = _fresh()
    for _ in range(50):
        f.filter(_landmarks(1.0), DT)
    f.reset()
    assert not f.initialized
    out = f.filter(_landmarks(-1.0), DT)
    assert np.allclose(out, -1.0)


@case
def test_exponential_filter_scalar() -> None:
    e = ExponentialFilter(5.0)
    assert e.filter(0.5, DT) == 0.5
    for _ in range(200):
        y = e.filter(0.9, DT)
    assert abs(y - 0.9) < 1e-5
    assert e.filter(0.0, 0.0) == y, "zero dt must hold the value"
    rng = np.random.default_rng(3)
    noisy = [e.filter(0.9 + rng.normal(scale=0.05), DT) for _ in range(400)]
    assert np.std(noisy[100:]) < 0.05


@case
def test_velocity_estimator_matches_a_known_speed() -> None:
    v = VelocityEstimator()
    speed = 1.25
    for i in range(400):
        out = v.update(_landmarks(speed * i * DT), DT)
    assert abs(float(out[0, 0]) - speed) < 0.02, float(out[0, 0])
    assert abs(float(out[0, 1]) - speed) < 0.02


@case
def test_velocity_estimator_is_quiet_when_still() -> None:
    rng = np.random.default_rng(21)
    raw = VelocityEstimator()
    filtered = VelocityEstimator()
    f = _fresh()
    for _ in range(400):
        sample = (_landmarks(0.2) + rng.normal(scale=0.002, size=(21, 3))
                  ).astype(np.float32)
        raw_v = raw.update(sample, DT)
        filtered_v = filtered.update(f.filter(sample, DT), DT)
    raw_speed = float(np.linalg.norm(raw_v, axis=1).max())
    smooth_speed = float(np.linalg.norm(filtered_v, axis=1).max())
    assert smooth_speed < raw_speed, (
        f"differentiating filtered positions ({smooth_speed:.4f} m/s) must beat "
        f"differentiating raw ones ({raw_speed:.4f} m/s)")
    assert smooth_speed < 0.05


@case
def test_velocity_estimator_extrapolates() -> None:
    v = VelocityEstimator()
    for i in range(300):
        v.update(_landmarks(2.0 * i * DT), DT)
    ahead = v.extrapolate(_landmarks(1.0), 0.1)
    assert abs(float(ahead[0, 0]) - 1.2) < 0.02
    v.decay(0.0)
    assert np.allclose(v.extrapolate(_landmarks(1.0), 0.1), 1.0)


if __name__ == "__main__":
    sys.exit(run(__file__))
