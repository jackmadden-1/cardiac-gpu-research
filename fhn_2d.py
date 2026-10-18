"""
Two-dimensional FitzHugh-Nagumo simulation: spiral (re-entrant) wave.

Purpose:
    Extend the 1D model to a 2D sheet of tissue and create a spiral
    wave, the 2D analogue of a re-entrant arrhythmia.

    The same script runs either solver so the two can be compared by eye:
        METHOD = "euler"  -> forward Euler + 5-point finite differences
        METHOD = "lbm"    -> D2Q5 lattice Boltzmann

How the spiral is started ("broken wave" initial condition):
    A vertical strip of excited tissue is placed in the lower half of the
    sheet only, with a strip of refractory tissue (high w) just to its
    left. The wave can only travel right, and its free upper end curls
    around into a rotating spiral.

Run from the project root:
    python fhn_2d.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import euler_fd
from fhn_model import FHNParams, euler_stability_limit, resting_state
from lattice_boltzmann import lbm_fhn

# ============================================================
# Settings
# ============================================================

METHOD = "euler"        # "euler" or "lbm"
N = 200                 # grid is N x N nodes
dx = 1.0
dt = 0.05
total_time = 600.0
snapshot_every = 100.0  # time units between saved frames

p = FHNParams()

if METHOD == "euler":
    limit = euler_stability_limit(dx, p.D, ndim=2)
    assert dt <= limit, f"dt={dt} exceeds the Euler stability limit {limit:.4f}"
else:
    tau = lbm_fhn.tau_from(p.D, dx, dt)
    print(f"LBM relaxation time tau = {tau:.3f}")

# ============================================================
# Initial conditions: rest everywhere + broken wave
# ============================================================

v_rest, w_rest = resting_state(p)
v = np.full((N, N), v_rest)
w = np.full((N, N), w_rest)

Y, X = np.mgrid[0:N, 0:N]
lower_half = Y < N // 2
v[lower_half & (X >= 95) & (X < 105)] = 1.5   # excited strip
w[lower_half & (X >= 80) & (X < 95)] = 1.0    # refractory strip behind it

no_stimulus = np.zeros_like(v)

if METHOD == "lbm":
    f = lbm_fhn.init_populations(v, ndim=2)

# ============================================================
# Time integration
# ============================================================

n_steps = int(round(total_time / dt))
save_every = int(round(snapshot_every / dt))
snapshots, snapshot_times = [], []

start = time.perf_counter()
for n in range(n_steps + 1):
    if n % save_every == 0:
        snapshots.append(v.copy())
        snapshot_times.append(n * dt)
    if n == n_steps:
        break

    if METHOD == "euler":
        v, w = euler_fd.step(v, w, no_stimulus, dt, dx, p)
    else:
        f, w = lbm_fhn.step_2d(f, w, no_stimulus, dt, tau, p)
        v = lbm_fhn.density(f)

elapsed = time.perf_counter() - start
updates = N * N * n_steps / elapsed
print(f"{METHOD}: {n_steps} steps on {N}x{N} in {elapsed:.1f} s "
      f"({updates / 1e6:.2f} M node-updates/s)")

# ============================================================
# Visualisation
# ============================================================

fig, axes = plt.subplots(1, len(snapshots), figsize=(2.6 * len(snapshots), 2.9))
for ax, snap, t in zip(axes, snapshots, snapshot_times):
    ax.imshow(snap, origin="lower", cmap="inferno", vmin=-2.0, vmax=2.0)
    ax.set_title(f"t = {t:.0f}")
    ax.axis("off")
fig.suptitle(f"2D FitzHugh-Nagumo spiral wave ({METHOD}, dx={dx}, dt={dt})")
fig.tight_layout()
fig.savefig(f"results/fhn_2d_spiral_{METHOD}.png", dpi=120)
plt.show()
