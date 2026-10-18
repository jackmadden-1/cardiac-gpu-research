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
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08
    D: float = 1.0          # diffusion coefficient


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
