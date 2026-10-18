"""
Forward-Euler / finite-difference solver for the FitzHugh-Nagumo
reaction-diffusion system, in 1D and 2D.

Spatial discretisation:
    Cell-centred grid with the standard 3-point (1D) or 5-point (2D)
    Laplacian. Zero-flux (Neumann) boundaries are imposed with ghost
    nodes that copy the edge value, i.e. the wall sits half a cell
    outside the first/last node. This is the same boundary that
    "halfway bounce-back" gives in the lattice Boltzmann solver, so
    the two methods solve exactly the same problem.

Time stepping:
    Forward Euler for both v and w.
    Stability (diffusion part): dt <= dx^2 / (2 * ndim * D).
"""

import numpy as np

from fhn_model import FHNParams, reaction


def laplacian(v, dx):
    """Laplacian with zero-flux ghost nodes. Works for 1D or 2D arrays."""
    vp = np.pad(v, 1, mode="edge")
    if v.ndim == 1:
        return (vp[2:] - 2.0 * v + vp[:-2]) / dx ** 2
    return (vp[2:, 1:-1] + vp[:-2, 1:-1] + vp[1:-1, 2:] + vp[1:-1, :-2] - 4.0 * v) / dx ** 2


def step(v, w, stimulus, dt, dx, p: FHNParams):
    """Advance (v, w) by one forward-Euler step. Returns new arrays."""
    rv, rw = reaction(v, w, stimulus, p)
    v_new = v + dt * (p.D * laplacian(v, dx) + rv)
    w_new = w + dt * rw
    return v_new, w_new
