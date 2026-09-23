"""
One-dimensional FitzHugh-Nagumo propagation simulation.

Purpose:
    Extend the single-cell FitzHugh-Nagumo model into a one-dimensional
    line of excitable tissue.

    Each position in the line has its own excitation variable v and
    recovery variable w.

    Neighboring positions are coupled through a diffusion term, allowing
    excitation to propagate from one part of the tissue to another.

Numerical methods:
    - Forward Euler integration in time
    - Finite differences for the spatial diffusion term

Model equation:
    dv/dt = D * d²v/dx² + v - v³/3 - w + I

    dw/dt = epsilon * (v + a - b*w)

The important difference from the single-cell model is the term:

    D * d²v/dx²

This represents spatial diffusion and allows an excitation wave to move
through the simulated tissue.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FitzHugh-Nagumo reaction model
# ============================================================

def fitzhugh_nagumo(v, w, stimulus, a=0.7, b=0.8, epsilon=0.08):
    """
    Compute the local FitzHugh-Nagumo reaction terms.

    Parameters:
        v:
            Excitation / voltage-like state.

        w:
            Recovery state.

        stimulus:
            External stimulus applied to the tissue.

        a, b, epsilon:
            FitzHugh-Nagumo model parameters.

    Returns:
        reaction_v:
            Local rate of change of excitation, excluding diffusion.

        dw_dt:
            Rate of change of the recovery variable.
    """

    # Local excitation dynamics.
    reaction_v = v - (v ** 3) / 3 - w + stimulus

    # Local recovery dynamics.
    dw_dt = epsilon * (v + a - b * w)

    return reaction_v, dw_dt


# ============================================================
# Simulation parameters
# ============================================================

# Number of spatial locations in the simulated line of tissue.
num_cells = 200

# Distance between neighboring spatial grid points.
#
# This is Δx in the finite-difference approximation.
dx = 1.0

# Time-step size used by Forward Euler.
#
# This is Δt.
dt = 0.01

# Total simulated time.
total_time = 100.0

# Diffusion coefficient.
#
# This controls how strongly excitation spreads between neighboring
# positions in the tissue.
D = 1.0

# Create the simulation time points.
times = np.arange(0, total_time, dt)


# ============================================================
# Initial conditions
# ============================================================

# Every spatial location begins in the same resting state.
#
# v and w are now arrays instead of single numbers.
#
# v[i] and w[i] describe the state at spatial location i.
v = np.full(num_cells, -1.0)
w = np.full(num_cells, -0.5)


# ============================================================
# Result storage
# ============================================================

# Save selected snapshots rather than every single time step.
#
# Each saved snapshot contains the excitation state across the
# entire tissue line at one moment in time.
snapshots = []

snapshot_times = []


# ============================================================
# Time integration
# ============================================================

for t in times:

    # --------------------------------------------------------
    # External stimulus
    # --------------------------------------------------------

    # Create an array containing the stimulus applied at every
    # spatial location.
    stimulus = np.zeros(num_cells)

    # Stimulate only the first few cells near the left side of
    # the tissue between t = 10 and t = 11.
    #
    # The goal is to trigger an excitation wave that then
    # propagates through the rest of the tissue.
    if 10 <= t <= 11:
        stimulus[0:5] = 1.0


    # --------------------------------------------------------
    # Spatial diffusion
    # --------------------------------------------------------

    # Create an array to store the second spatial derivative
    # of v, also called the Laplacian in one dimension.
    laplacian_v = np.zeros(num_cells)

    # For interior grid points, approximate:
    #
    #              d²v
    #              ---
    #              dx²
    #
    # using the central finite-difference formula:
    #
    #     v[i+1] - 2*v[i] + v[i-1]
    #     -------------------------
    #                 dx²
    #
    # This measures how different each point is from its
    # neighboring points.
    laplacian_v[1:-1] = (
        v[2:]
        - 2 * v[1:-1]
        + v[:-2]
    ) / dx**2


    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------

    # Use zero-flux / no-flux boundary conditions.
    #
    # This means excitation is not allowed to diffuse out of
    # either end of the simulated tissue.
    #
    # We approximate this by treating the boundary value as if
    # it had an identical neighbor outside the domain.
    laplacian_v[0] = (
        v[1] - v[0]
    ) / dx**2

    laplacian_v[-1] = (
        v[-2] - v[-1]
    ) / dx**2


    # --------------------------------------------------------
    # FitzHugh-Nagumo reaction
    # --------------------------------------------------------

    # Compute the local reaction terms at every spatial point.
    reaction_v, dw_dt = fitzhugh_nagumo(
        v,
        w,
        stimulus
    )


    # --------------------------------------------------------
    # Reaction-diffusion equation
    # --------------------------------------------------------

    # The excitation variable now changes for two reasons:
    #
    # 1. Local FitzHugh-Nagumo reaction dynamics
    # 2. Diffusion from neighboring spatial locations
    #
    # Full equation:
    #
    #     dv/dt = D * d²v/dx² + reaction
    dv_dt = D * laplacian_v + reaction_v


    # --------------------------------------------------------
    # Forward Euler update
    # --------------------------------------------------------

    # Advance the entire tissue state by one time step.
    #
    # Because v and w are NumPy arrays, this updates all
    # spatial locations at once.
    v = v + dt * dv_dt
    w = w + dt * dw_dt


    # --------------------------------------------------------
    # Save occasional snapshots
    # --------------------------------------------------------

    # Save approximately every 5 units of simulated time.
    #
    # We do not need to store every single time step just to
    # visualize how the wave moves through the tissue.
    if int(t / dt) % int(5.0 / dt) == 0:
        snapshots.append(v.copy())
        snapshot_times.append(t)


# ============================================================
# Visualization
# ============================================================

# Plot several snapshots of excitation across the tissue.
#
# The x-axis represents position along the tissue.
# The y-axis represents the excitation / voltage-like variable.
plt.figure(figsize=(10, 6))

for i in range(0, len(snapshots), max(1, len(snapshots) // 10)):
    plt.plot(
        snapshots[i],
        label=f"t = {snapshot_times[i]:.1f}"
    )

plt.xlabel("Position along tissue")
plt.ylabel("Excitation variable v")
plt.title("1D FitzHugh-Nagumo Wave Propagation")
plt.legend()

plt.show()