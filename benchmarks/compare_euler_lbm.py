"""
Benchmark: forward-Euler finite differences vs lattice Boltzmann
for the FitzHugh-Nagumo model (CPU / NumPy reference implementation).

Three experiments, each written to results/ as CSV + PNG:

1. Grid convergence (1D plane wave)
   Both methods at dx = 1, 0.5, 0.25 with the same dt = r * dx^2.
   Compared with a fine-grid FD reference on:
     - conduction velocity (CV)
     - RMS error of the activation-time map

2. Time-step sweep (1D, dx = 0.5)
   Increase r = dt / dx^2 past the Euler stability limit (r = 0.5 in 1D)
   to see where each method breaks down or loses accuracy.

3. Throughput (2D)
   Node-updates per second for both methods on N x N grids.
   NumPy numbers are a CPU baseline only; the GPU numbers come later
   from the WebGPU code in webgpu/.

Run from the project root:
    python benchmarks/compare_euler_lbm.py          (full, a few minutes)
    python benchmarks/compare_euler_lbm.py --quick  (smoke test)
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import euler_fd  # noqa: E402
from fhn_model import FHNParams, resting_state  # noqa: E402
from lattice_boltzmann import lbm_fhn  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

P = FHNParams()
V_REST, W_REST = resting_state(P)

LENGTH = 200.0        # cable length
T_END = 300.0         # long enough for the wave to cross the cable
STIM_WIDTH = 5.0      # stimulate x < 5
STIM_TIME = 2.0       # for t < 2
STIM_AMP = 1.0
X_A, X_B = 50.0, 150.0  # CV measured between these points


# ------------------------------------------------------------
# 1D run returning the activation-time map
# ------------------------------------------------------------

def run_1d(method, dx, dt, t_end=T_END):
    n = int(round(LENGTH / dx))
    x = (np.arange(n) + 0.5) * dx
    v = np.full(n, V_REST)
    w = np.full(n, W_REST)
    t_act = np.full(n, np.nan)
    stim_mask = x < STIM_WIDTH
    stim = np.zeros(n)

    if method == "lbm":
        tau = lbm_fhn.tau_from(P.D, dx, dt)
        f = lbm_fhn.init_populations(v, ndim=1)

    n_steps = int(round(t_end / dt))
    start = time.perf_counter()
    for k in range(n_steps):
        t = k * dt
        stim[:] = 0.0
        if t < STIM_TIME:
            stim[stim_mask] = STIM_AMP

        if method == "fd":
            v, w = euler_fd.step(v, w, stim, dt, dx, P)
        else:
            f, w = lbm_fhn.step_1d(f, w, stim, dt, tau, P)
            v = lbm_fhn.density(f)

        if not np.all(np.isfinite(v)) or np.abs(v).max() > 1e3:
            return x, None, time.perf_counter() - start  # blew up

        newly = np.isnan(t_act) & (v >= 0.0)
        t_act[newly] = t + dt

    return x, t_act, time.perf_counter() - start


def conduction_velocity(x, t_act):
    if t_act is None:
        return np.nan
    ta = np.interp(X_A, x, t_act)
    tb = np.interp(X_B, x, t_act)
    return (X_B - X_A) / (tb - ta)


def rms_error(x, t_act, x_ref, t_ref):
    if t_act is None:
        return np.nan
    ref_on_grid = np.interp(x, x_ref, t_ref)
    mask = (x > X_A) & (x < X_B) & np.isfinite(t_act) & np.isfinite(ref_on_grid)
    if not mask.any():
        return np.nan  # wave never reached the measurement window
    return float(np.sqrt(np.mean((t_act[mask] - ref_on_grid[mask]) ** 2)))


def write_csv(name, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


# ------------------------------------------------------------
# Experiment 1: grid convergence
# ------------------------------------------------------------

def experiment_convergence(quick):
    print("\n[1] Grid convergence (1D)")
    ratio = 0.2  # dt = 0.2 dx^2, safely below the 1D Euler limit of 0.5
    ref_dx = 0.25 if quick else 0.0625
    x_ref, t_ref, _ = run_1d("fd", ref_dx, ratio * ref_dx ** 2)
    cv_ref = conduction_velocity(x_ref, t_ref)
    print(f"  reference FD dx={ref_dx}: CV = {cv_ref:.4f}")

    dxs = [1.0, 0.5] if quick else [1.0, 0.5, 0.25, 0.125]
    rows = []
    for dx in dxs:
        dt = ratio * dx ** 2
        for method in ("fd", "lbm"):
            x, ta, secs = run_1d(method, dx, dt)
            cv = conduction_velocity(x, ta)
            row = dict(method=method, dx=dx, dt=dt,
                       tau=lbm_fhn.tau_from(P.D, dx, dt) if method == "lbm" else "",
                       cv=cv, cv_rel_err=abs(cv - cv_ref) / cv_ref,
                       t_act_rms_err=rms_error(x, ta, x_ref, t_ref),
                       seconds=secs)
            rows.append(row)
            print(f"  {method:3s} dx={dx:<6} CV={cv:.4f} "
                  f"rel.err={row['cv_rel_err']:.2e} rms(t_act)={row['t_act_rms_err']:.3f} "
                  f"{secs:.2f}s")
    write_csv("convergence_1d.csv", rows)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for method, color in (("fd", "#1f77b4"), ("lbm", "#d62728")):
        r = [row for row in rows if row["method"] == method]
        ax[0].loglog([q["dx"] for q in r], [q["cv_rel_err"] for q in r], "o-",
                     color=color, label=method.upper())
        ax[1].loglog([q["dx"] for q in r], [q["t_act_rms_err"] for q in r], "o-",
                     color=color, label=method.upper())
    dx_line = np.array(dxs)
    for a in ax:
        a.set_xlabel("dx")
        a.set_xticks(dxs, [str(d) for d in dxs])
        a.minorticks_off()
        a.grid(True, which="both", alpha=0.3)
    first_fd = rows[0]["cv_rel_err"]
    ax[0].loglog(dx_line, first_fd * (dx_line / dxs[0]) ** 2, "k--", lw=0.8, label="O(dx^2)")
    ax[0].set_ylabel("relative CV error")
    ax[1].set_ylabel("RMS activation-time error")
    ax[0].legend()
    fig.suptitle(f"Grid convergence, dt = {ratio} dx^2 (reference FD dx = {ref_dx})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "convergence_1d.png"), dpi=130)
    return x_ref, t_ref


# ------------------------------------------------------------
# Experiment 2: time-step sweep
# ------------------------------------------------------------

def experiment_dt_sweep(quick, x_ref, t_ref):
    print("\n[2] Time-step sweep (1D, dx = 0.5)")
    dx = 0.5
    ratios = [0.1, 0.4, 0.6, 1.0] if quick else [0.1, 0.2, 0.4, 0.5, 0.6, 1.0, 2.0, 4.0]
    rows = []
    for r in ratios:
        dt = r * dx ** 2
        for method in ("fd", "lbm"):
            x, ta, secs = run_1d(method, dx, dt)
            cv = conduction_velocity(x, ta)
            row = dict(method=method, dt_over_dx2=r, dt=dt,
                       tau=lbm_fhn.tau_from(P.D, dx, dt) if method == "lbm" else "",
                       stable=ta is not None, cv=cv,
                       t_act_rms_err=rms_error(x, ta, x_ref, t_ref), seconds=secs)
            rows.append(row)
            if ta is None:
                status = "UNSTABLE (blew up)"
            elif not np.isfinite(cv):
                status = "stable, but the wave did not propagate"
            else:
                status = f"CV={cv:.4f} rms={row['t_act_rms_err']:.3f}"
            print(f"  {method:3s} dt/dx^2={r:<4} {status}")
    write_csv("dt_sweep_1d.csv", rows)


# ------------------------------------------------------------
# Experiment 3: 2D throughput
# ------------------------------------------------------------

def experiment_throughput(quick):
    print("\n[3] 2D throughput (NumPy, CPU)")
    sizes = [128, 256] if quick else [128, 256, 512, 1024]
    n_steps = 50 if quick else 200
    dx, dt = 1.0, 0.05
    rows = []
    for n in sizes:
        v0 = np.full((n, n), V_REST)
        w0 = np.full((n, n), W_REST)
        z = np.zeros_like(v0)
        for method in ("fd", "lbm"):
            v, w = v0.copy(), w0.copy()
            if method == "lbm":
                f = lbm_fhn.init_populations(v, ndim=2)
                tau = lbm_fhn.tau_from(P.D, dx, dt)
            start = time.perf_counter()
            for _ in range(n_steps):
                if method == "fd":
                    v, w = euler_fd.step(v, w, z, dt, dx, P)
                else:
                    f, w = lbm_fhn.step_2d(f, w, z, dt, tau, P)
            secs = time.perf_counter() - start
            mnups = n * n * n_steps / secs / 1e6
            # bytes stored per node (float64): FD = v, w ; LBM = 5 populations + w
            bytes_per_node = 8 * (2 if method == "fd" else 6)
            rows.append(dict(method=method, N=n, steps=n_steps, seconds=secs,
                             mnups=mnups, state_bytes_per_node=bytes_per_node))
            print(f"  {method:3s} N={n:<5} {mnups:7.2f} M node-updates/s")
    write_csv("throughput_2d_numpy.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small, fast run")
    args = parser.parse_args()

    x_ref, t_ref = experiment_convergence(args.quick)
    experiment_dt_sweep(args.quick, x_ref, t_ref)
    experiment_throughput(args.quick)
    print("\nDone.")
