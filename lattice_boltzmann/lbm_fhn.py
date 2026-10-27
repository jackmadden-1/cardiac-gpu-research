"""
Lattice Boltzmann solver for the FitzHugh-Nagumo reaction-diffusion system.

Idea
----
Instead of storing v directly, each node stores a few "populations" f_i,
one per lattice direction. The voltage-like variable is their sum:

    v = sum_i f_i

Each time step has two parts:

1. Collide (purely local, perfect for GPUs):
       f_i* = f_i - (f_i - f_i^eq) / tau + dt * w_i * R(v, w)
   with equilibrium f_i^eq = w_i * v and R the FHN reaction term.

2. Stream (move each population one node along its direction):
       f_i(x + c_i dx, t + dt) = f_i*(x, t)

The recovery variable w has no diffusion, so it is updated with a plain
forward-Euler step at each node, exactly as in the FD solver.

Lattices
--------
D1Q3 (1D): directions {0, +x, -x}, weights {2/3, 1/6, 1/6}
D2Q5 (2D): directions {0, +x, -x, +y, -y}, weights {1/3, 1/6, 1/6, 1/6, 1/6}
Both have lattice sound speed c_s^2 = 1/3.

Diffusion coefficient (from a Chapman-Enskog expansion):
    D = c_s^2 * (tau - 1/2) * dx^2 / dt
so for given D, dx, dt:
    tau = 1/2 + D * dt / (c_s^2 * dx^2)
Stability needs tau > 1/2. Accuracy is best for tau roughly in (0.5, 1.5].

Boundaries
----------
Zero-flux via halfway bounce-back: a population that would leave the
domain is reflected back into the opposite direction at the same node.
The effective wall sits half a cell outside the edge nodes, matching the
ghost-node boundary in euler_fd.py.

References: Rapaka et al., MICCAI 2012 (LBM-EP);
Campos et al., J. Comput. Appl. Math. 2016; Krueger et al., The Lattice
Boltzmann Method (Springer, 2017), ch. 8 (advection-diffusion).
"""

import numpy as np

from fhn_model import FHNParams

CS2 = 1.0 / 3.0
W_D1Q3 = np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
W_D2Q5 = np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])


def tau_from(D, dx, dt):
    tau = 0.5 + D * dt / (CS2 * dx ** 2)
    return tau


def init_populations(v0, ndim):
    """Start every population at equilibrium for the initial voltage field."""
    wts = W_D1Q3 if ndim == 1 else W_D2Q5
    shape = (len(wts),) + (1,) * v0.ndim
    return wts.reshape(shape) * v0[None, ...]


def density(f):
    """v = sum of populations."""
    return f.sum(axis=0)


def _collide(f, w, stimulus, dt, tau, p: FHNParams, wts):
    v = f.sum(axis=0)
    rv, rw = p.reaction(v, w, stimulus)
    shape = (len(wts),) + (1,) * v.ndim
    wk = wts.reshape(shape)
    f_star = f - (f - wk * v) / tau + dt * wk * rv
    w_new = w + dt * rw
    return f_star, w_new


def step_1d(f, w, stimulus, dt, tau, p: FHNParams):
    """One D1Q3 collide + stream step. f has shape (3, N)."""
    s, w_new = _collide(f, w, stimulus, dt, tau, p, W_D1Q3)
    out = np.empty_like(s)
    out[0] = s[0]
    # +x population moves right; at the left wall it comes from the bounced -x
    out[1, 1:] = s[1, :-1]
    out[1, 0] = s[2, 0]
    # -x population moves left; at the right wall it comes from the bounced +x
    out[2, :-1] = s[2, 1:]
    out[2, -1] = s[1, -1]
    return out, w_new


def stream_2d(s):
    """D2Q5 streaming with halfway bounce-back. s has shape (5, Ny, Nx)."""
    out = np.empty_like(s)
    out[0] = s[0]
    # +x
    out[1, :, 1:] = s[1, :, :-1]
    out[1, :, 0] = s[2, :, 0]
    # -x
    out[2, :, :-1] = s[2, :, 1:]
    out[2, :, -1] = s[1, :, -1]
    # +y
    out[3, 1:, :] = s[3, :-1, :]
    out[3, 0, :] = s[4, 0, :]
    # -y
    out[4, :-1, :] = s[4, 1:, :]
    out[4, -1, :] = s[3, -1, :]
    return out


def step_2d(f, w, stimulus, dt, tau, p: FHNParams):
    """One D2Q5 collide + stream step. f has shape (5, Ny, Nx); axis -1 is x."""
    s, w_new = _collide(f, w, stimulus, dt, tau, p, W_D2Q5)
    return stream_2d(s), w_new
