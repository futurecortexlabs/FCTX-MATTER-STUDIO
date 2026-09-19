"""Temporal smoothing for hand tracking.

A capsule collider that jitters by 3 mm at 60 Hz injects a 0.18 m/s velocity
impulse into the cloth every frame.  The physics is fine; the *input* is not.
Everything in this module exists to stop landmark noise from reaching the
solver, without adding the lag that a plain low-pass filter would.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["OneEuroFilter", "ExponentialFilter", "VelocityEstimator", "smoothing_factor"]

#: Longest gap we will treat as a real time step.  A stall (window drag, a slow
#: first MediaPipe inference) would otherwise make every filter alpha ~1.0 and
#: let a full frame of noise through at exactly the moment the user is looking.
MAX_DT = 0.25


def smoothing_factor(dt: float, cutoff: float | np.ndarray) -> float | np.ndarray:
    """Exponential-smoothing alpha for a low-pass at ``cutoff`` Hz over ``dt``."""
    tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)


def _check_dt(who: str, dt: float) -> float:
    """Validate a time step.

    A non-finite ``dt`` has to be rejected rather than clamped: ``min(nan, x)``
    returns ``nan``, so it would flow into the alpha, into the filter state,
    and from then on every output of this filter is ``nan`` -- with the
    traceback surfacing somewhere else entirely, frames later.
    """
    step = float(dt)
    if not math.isfinite(step):
        raise ValueError(f"{who} needs a finite dt, got {dt}")
    if step < 0.0:
        raise ValueError(f"{who} needs a non-negative dt, got {dt}")
    return step


class OneEuroFilter:
    """One-Euro filter (Casiez, Roussel & Vogel, 2012) over a whole array.

    The cutoff frequency rises with the estimated speed of each component, so
    the filter is aggressive while the hand is still and nearly transparent
    while it moves.  A fixed low-pass cannot do both: tuned to kill the jitter
    it adds visible lag, tuned for responsiveness it does not smooth.

    The entire ``(21, 3)`` landmark array is filtered in one vectorised pass.
    Looping over 21 landmarks in Python costs more than the filter maths does.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_hat", "_dx_hat")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        if min_cutoff <= 0.0 or d_cutoff <= 0.0:
            raise ValueError(
                f"cutoffs must be positive, got min_cutoff={min_cutoff}, "
                f"d_cutoff={d_cutoff}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_hat: np.ndarray | None = None
        self._dx_hat: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._x_hat is not None

    @property
    def value(self) -> np.ndarray | None:
        """The current estimate, or ``None`` before the first sample."""
        return None if self._x_hat is None else self._x_hat.copy()

    def reset(self) -> None:
        self._x_hat = None
        self._dx_hat = None

    def filter(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Return the smoothed value of ``x`` after ``dt`` seconds."""
        arr = np.asarray(x, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("OneEuroFilter received non-finite input")
        dt = _check_dt("OneEuroFilter", dt)

        if self._x_hat is None:
            self._x_hat = arr.copy()
            self._dx_hat = np.zeros_like(arr)
            return self._x_hat.astype(np.float32)

        if self._x_hat.shape != arr.shape:
            raise ValueError(
                f"OneEuroFilter was primed with shape {self._x_hat.shape} "
                f"and got {arr.shape}")

        # dt == 0 means two samples carry the same timestamp; no time has
        # passed, so there is nothing to integrate and dividing would blow up.
        if dt == 0.0:
            return self._x_hat.astype(np.float32)
        step = min(dt, MAX_DT)

        assert self._dx_hat is not None
        # The derivative is taken against the previous *filtered* value, as in
        # the reference implementation: differentiating the raw signal would
        # feed the noise straight into the adaptive cutoff.
        dx = (arr - self._x_hat) / step
        a_d = smoothing_factor(step, self.d_cutoff)
        self._dx_hat += a_d * (dx - self._dx_hat)

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx_hat)
        alpha = smoothing_factor(step, cutoff)
        self._x_hat += alpha * (arr - self._x_hat)
        return self._x_hat.astype(np.float32)


class ExponentialFilter:
    """Scalar low-pass with a frequency cutoff, for pinch and curl.

    These are already derived from many landmarks, so they are far less noisy
    than a single coordinate and do not need the adaptive machinery; what they
    do need is not to chatter across the grab threshold.
    """

    __slots__ = ("cutoff", "_y")

    def __init__(self, cutoff: float) -> None:
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.cutoff = float(cutoff)
        self._y: float | None = None

    @property
    def initialized(self) -> bool:
        return self._y is not None

    def reset(self) -> None:
        self._y = None

    def filter(self, x: float, dt: float) -> float:
        value = float(x)
        if not math.isfinite(value):
            raise ValueError("ExponentialFilter received non-finite input")
        dt = _check_dt("ExponentialFilter", dt)
        if self._y is None:
            self._y = value
            return value
        if dt == 0.0:
            return self._y
        alpha = float(smoothing_factor(min(dt, MAX_DT), self.cutoff))
        self._y += alpha * (value - self._y)
        return self._y


class VelocityEstimator:
    """Differentiate an already-filtered position signal.

    Differentiating the *raw* landmarks would amplify by ``1/dt`` exactly the
    noise the One-Euro filter just removed, and that velocity is what the
    solver hands to the matter on release: a 1 mm blip at 90 Hz becomes a
    0.09 m/s throw.  So this consumes filtered positions, and still applies a
    light low-pass of its own because differencing has unity gain at DC but
    grows linearly with frequency.
    """

    __slots__ = ("cutoff", "_prev", "_v")

    def __init__(self, cutoff: float = 6.0) -> None:
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.cutoff = float(cutoff)
        self._prev: np.ndarray | None = None
        self._v: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._prev is not None

    def reset(self) -> None:
        self._prev = None
        self._v = None

    def update(self, x: np.ndarray, dt: float) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("VelocityEstimator received non-finite input")
        dt = _check_dt("VelocityEstimator", dt)
        if self._prev is None:
            self._prev = arr.copy()
            self._v = np.zeros_like(arr)
            return self._v.astype(np.float32)
        assert self._v is not None
        if dt == 0.0:
            return self._v.astype(np.float32)
        step = min(dt, MAX_DT)
        raw = (arr - self._prev) / step
        alpha = float(smoothing_factor(step, self.cutoff))
        self._v += alpha * (raw - self._v)
        self._prev = arr.copy()
        return self._v.astype(np.float32)

    def extrapolate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Advance ``x`` by the current velocity, for coasting a lost hand."""
        step = _check_dt("VelocityEstimator.extrapolate", dt)
        if self._v is None:
            return np.asarray(x, dtype=np.float32).copy()
        return (np.asarray(x, dtype=np.float64) + self._v * step).astype(np.float32)

    def decay(self, factor: float) -> None:
        """Bleed off the stored velocity while a hand is being coasted."""
        f = float(factor)
        if not math.isfinite(f) or f < 0.0:
            raise ValueError(f"decay factor must be finite and non-negative, got {factor}")
        if self._v is not None:
            self._v *= f
