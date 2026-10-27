"""
Shared FitzHugh-Nagumo model definitions.

Both solvers (forward-Euler finite differences and lattice Boltzmann)
import the reaction terms and parameters from here, so any difference
in their results comes from the diffusion solver alone.

Model:
    dv/dt = D * laplacian(v) + v - v^3/3 - w + I_stim
    dw/dt = epsilon * (v + a - b*w)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FHNParams:
    """Classic FitzHugh form (rest near v = -1.2). Used in the first study."""
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08
    D: float = 1.0          # diffusion coefficient

    def reaction(self, v, w, stimulus):
        return reaction(v, w, stimulus, self)

    def rest(self):
        return resting_state(self)


@dataclass(frozen=True)
class FentonFHNParams:
    """
    Modified FitzHugh-Nagumo model from Fenton's group, with rest at u = 0
    and the upstroke toward ~1:

        du/dt = D lap(u) + mu*u*(1 - u)*(u - alpha) - v*u + I_stim
        dv/dt = eps*((beta - u)*(u - gamma) - delta*v - theta)

    Values: Rheaume, ..., Cherry & Fenton, Computing in Cardiology 2023,
    Eq. 2 (CinC2023-424). Symbols matched to their Eq. 1 by position;
    theta = -0.095 follows from "-theta" in Eq. 1 vs "+0.095" in Eq. 2.

    The paper gives D and dt only as a x9 rescaling of unstated base values,
    so D = 1 here is a working (dimensionless) choice, not theirs.

    Note: with these values the rest state is very weakly unstable
    (growth rate ~0.010 per time unit), so an isolated cell re-fires after
    ~7000 time units. Irrelevant for single-wave convergence runs.

    Standard FHN (Fenton, personal communication): replace
    (beta - u)(u - gamma) with u and set mu = delta = 1, theta = 0, i.e.

        du/dt = D lap(u) + u*(1 - u)*(u - alpha) - v*u + I_stim
        dv/dt = eps*(u - v)

    Use FentonFHNParams.standard(). alpha and eps are kept at the
    modified-model values (0.203, 0.005) unless given. The -v*u coupling
    is kept as instructed; it holds u >= 0 (no undershoot below rest).
    """
    mu: float = 1.475
    alpha: float = 0.203
    beta: float = 1.63
    gamma: float = 0.325
    delta: float = 1.403
    theta: float = -0.095
    epsilon: float = 0.005
    D: float = 1.0
    recovery: str = "quadratic"   # "quadratic": (beta-u)(u-gamma); "linear": u

    @classmethod
    def standard(cls, alpha=0.203, epsilon=0.005, D=1.0):
        """Fenton's standard FHN: linear recovery, mu = delta = 1, theta = 0."""
        return cls(mu=1.0, alpha=alpha, delta=1.0, theta=0.0, epsilon=epsilon,
                   D=D, recovery="linear")

    def _g(self, u):
        if self.recovery == "linear":
            return u
        return (self.beta - u) * (u - self.gamma)

    def reaction(self, u, v, stimulus):
        du = self.mu * u * (1.0 - u) * (u - self.alpha) - v * u + stimulus
        dv = self.epsilon * (self._g(u) - self.delta * v - self.theta)
        return du, dv

    def rest(self):
        """(u*, v*) with u* = 0 and v* from dv/dt = 0."""
        return 0.0, (self._g(0.0) - self.theta) / self.delta


def reaction(v, w, stimulus, p: FHNParams):
    """Local FHN reaction terms (no diffusion). Works on scalars or arrays."""
    dv = v - v ** 3 / 3.0 - w + stimulus
    dw = p.epsilon * (v + p.a - p.b * w)
    return dv, dw


def resting_state(p: FHNParams, iters: int = 50):
    """
    Resting state (v*, w*): intersection of the two nullclines,
        w = v - v^3/3   and   w = (v + a) / b.
    Solved with Newton's method. For a=0.7, b=0.8: v* ~ -1.1994, w* ~ -0.6243.
    """
    v = -1.2
    for _ in range(iters):
        f = v - v ** 3 / 3.0 - (v + p.a) / p.b
        df = 1.0 - v ** 2 - 1.0 / p.b
        v -= f / df
    return v, (v + p.a) / p.b


def euler_stability_limit(dx: float, D: float, ndim: int) -> float:
    """Largest stable forward-Euler time step for the diffusion term: dx^2 / (2 d D)."""
    return dx ** 2 / (2.0 * ndim * D)


def activation_times(v_history, times, threshold=0.0):
    """
    First time each node's v crosses `threshold` upward.
    v_history: array (n_times, n_nodes). Returns NaN where never activated.
    """
    v_history = np.asarray(v_history)
    above = v_history >= threshold
    first = np.argmax(above, axis=0)
    t_act = np.asarray(times)[first].astype(float)
    t_act[~above.any(axis=0)] = np.nan
    return t_act
