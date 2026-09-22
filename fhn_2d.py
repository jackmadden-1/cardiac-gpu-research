"""
Single-cell FitzHugh-Nagumo simulation.

Purpose:
    Simulate the excitation and recovery dynamics of one excitable cell.

Numerical method:
    Forward Euler integration.

Model variables:
    v - fast excitation / voltage-like variable
    w - slower recovery variable

Model equations:
    dv/dt = v - v^3 / 3 - w + I
    dw/dt = epsilon * (v + a - b*w)

This serves as the baseline model before adding spatial propagation
between multiple cells.
"""

import numpy as np
import matplotlib.pyplot as plt


def fitzhugh_nagumo(v, w, stimulus, a=0.7, b=0.8, epsilon=0.08):
    """
    Compute the rate of change of the FitzHugh-Nagumo state variables.

    Parameters:
        v:
            Excitation / voltage-like variable.

        w:
            Recovery variable.

        stimulus:
            External stimulus applied to the cell.
            This is the I term in the FitzHugh-Nagumo equation.

        a:
            Parameter that shifts the recovery dynamics.

        b:
            Parameter controlling the influence of w in the recovery equation.

        epsilon:
            Controls how quickly the recovery variable changes.
            A small epsilon makes w change more slowly than v.

    Returns:
        dv_dt:
            Rate of change of the excitation variable.

        dw_dt:
            Rate of change of the recovery variable.
    """

    # Fast excitation equation.
    # This controls how the voltage-like variable v changes over time.
    dv_dt = v - (v ** 3) / 3 - w + stimulus

    # Slow recovery equation.
    # The recovery variable w changes more slowly and helps return
    # the cell toward its resting state after excitation.
    dw_dt = epsilon * (v + a - b * w)

    return dv_dt, dw_dt


# ============================================================
# Simulation parameters
# ============================================================

# Time-step size used by Forward Euler.
# Each iteration of the simulation advances time by dt.
#
# Smaller values generally improve numerical accuracy,
# but require more calculations and therefore increase runtime.
dt = 0.01

# Total amount of simulated time.
total_time = 100.0

# Create all time points used in the simulation:
# 0.00, 0.01, 0.02, ..., 99.99
times = np.arange(0, total_time, dt)


# ============================================================
# Initial conditions
# ============================================================

# Initial state of the simulated cell before stimulation.
#
# v represents the fast excitation / voltage-like state.
# w represents the slower recovery state.
v = -1.0
w = -0.5


# ============================================================
# Result storage
# ============================================================

# Store the value of v and w after every time step.
# These histories will be plotted after the simulation finishes.
v_history = []
w_history = []


# ============================================================
# Time integration
# ============================================================

# Step through the simulation one small time interval at a time.
for t in times:

    # Apply a temporary external stimulus between t = 10 and t = 11.
    #
    # The stimulus pushes the cell away from its resting state
    # and is intended to trigger an excitation response.
    if 10 <= t <= 11:
        stimulus = 1.0
    else:
        stimulus = 0.0

    # Evaluate the FitzHugh-Nagumo differential equations
    # using the cell's current state.
    #
    # dv_dt tells us how quickly v is currently changing.
    # dw_dt tells us how quickly w is currently changing.
    dv_dt, dw_dt = fitzhugh_nagumo(v, w, stimulus)

    # Forward Euler update.
    #
    # General form:
    #
    #     next_value = current_value + dt * rate_of_change
    #
    # We use the derivative at the current time step to estimate
    # where the system will be one small time step later.
    v = v + dt * dv_dt
    w = w + dt * dw_dt

    # Save the updated state so that we can inspect and plot
    # the complete response after the simulation finishes.
    v_history.append(v)
    w_history.append(w)


# ============================================================
# Visualization
# ============================================================

# Plot excitation and recovery as functions of time.
plt.plot(times, v_history, label="v - excitation")
plt.plot(times, w_history, label="w - recovery")

plt.xlabel("Time")
plt.ylabel("State")
plt.title("FitzHugh-Nagumo Single Cell")
plt.legend()

plt.show()