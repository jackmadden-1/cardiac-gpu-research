"""
Relaxation-time sweep: find the tau that makes the LBM solver most accurate.

For a fixed grid spacing dx, the LBM relaxation time and the time step are
tied together by

    tau = 1/2 + 3 D dt / dx^2      (D2Q5 / D1Q3, c_s^2 = 1/3)

so choosing tau is the same as choosing dt = (tau - 1/2) dx^2 / (3 D).
This script sweeps tau at several dx on the 1D cable from
compare_euler_lbm.py and records, against a fine-grid Euler reference:

  - conduction-velocity (CV) error (signed, so you can see where it crosses 0)
  - RMS activation-time error

Euler/FD at dt = 0.1, 0.2, 0.4 dx^2 is included for comparison.

Writes results/tau_sweep_1d.csv and results/tau_sweep_1d.png.

Run from the project root:
    python benchmarks/tau_sweep.py           (full, ~3 minutes)
    python benchmarks/tau_sweep.py --quick   (coarse grids only)
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare_euler_lbm import (RESULTS, conduction_velocity, rms_error,  # noqa: E402
                               run_1d)

TAUS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.25, 1.5]
FD_RATIOS = [0.1, 0.2, 0.4]


def zero_crossing(taus, errs):
    """Linear interpolation of the tau where the signed CV error crosses 0."""
    for (t0, e0), (t1, e1) in zip(zip(taus, errs), zip(taus[1:], errs[1:])):
        if np.isfinite(e0) and np.isfinite(e1) and e0 >= 0 > e1:
            return t0 + (t1 - t0) * e0 / (e0 - e1)
    return np.nan


def main(quick):
    ref_dx = 0.25 if quick else 0.0625
    x_ref, t_ref, _ = run_1d("fd", ref_dx, 0.2 * ref_dx ** 2)
    cv_ref = conduction_velocity(x_ref, t_ref)
    print(f"reference FD dx={ref_dx}: CV = {cv_ref:.4f}")

    dxs = [1.0, 0.5] if quick else [1.0, 0.5, 0.25]
    rows = []
    for dx in dxs:
        for tau in TAUS:
            dt = (tau - 0.5) * dx ** 2 / 3.0
            x, ta, secs = run_1d("lbm", dx, dt)
            cv = conduction_velocity(x, ta)
            rows.append(dict(method="lbm", dx=dx, tau=tau, dt=dt, cv=cv,
                             cv_err=(cv - cv_ref) / cv_ref,
                             t_act_rms_err=rms_error(x, ta, x_ref, t_ref), seconds=secs))
        for r in FD_RATIOS:
            dt = r * dx ** 2
            x, ta, secs = run_1d("fd", dx, dt)
            cv = conduction_velocity(x, ta)
            rows.append(dict(method="fd", dx=dx, tau="", dt=dt, cv=cv,
                             cv_err=(cv - cv_ref) / cv_ref,
                             t_act_rms_err=rms_error(x, ta, x_ref, t_ref), seconds=secs))
        lbm = [q for q in rows if q["method"] == "lbm" and q["dx"] == dx]
        best = min(lbm, key=lambda q: q["t_act_rms_err"])
        fd02 = next(q for q in rows if q["method"] == "fd" and q["dx"] == dx
                    and abs(q["dt"] - 0.2 * dx ** 2) < 1e-12)
        tau0 = zero_crossing([q["tau"] for q in lbm], [q["cv_err"] for q in lbm])
        print(f"dx={dx:<5} CV error crosses 0 at tau ~ {tau0:.3f}; "
              f"best RMS {best['t_act_rms_err']:.3f} at tau={best['tau']} "
              f"vs Euler {fd02['t_act_rms_err']:.3f} at dt=0.2dx^2")

    path = os.path.join(RESULTS, "tau_sweep_1d.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")

    colors = {1.0: "#2a78d6", 0.5: "#eb6834", 0.25: "#1baf7a"}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for dx in dxs:
        lbm = [q for q in rows if q["method"] == "lbm" and q["dx"] == dx]
        t = [q["tau"] for q in lbm]
        ax[0].plot(t, [100 * q["cv_err"] for q in lbm], "o-", color=colors[dx], label=f"LBM dx={dx:g}")
        ax[1].semilogy(t, [q["t_act_rms_err"] for q in lbm], "o-", color=colors[dx])
        fd02 = next(q for q in rows if q["method"] == "fd" and q["dx"] == dx
                    and abs(q["dt"] - 0.2 * dx ** 2) < 1e-12)
        ax[1].axhline(fd02["t_act_rms_err"], color=colors[dx], ls=":")
    ax[0].axhline(0, color="#888", lw=0.7)
    ax[0].set_xlabel("tau"); ax[0].set_ylabel("CV error (%)"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("tau"); ax[1].set_ylabel("RMS activation-time error")
    ax[1].set_title("dotted: Euler at dt = 0.2 dx^2", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "tau_sweep_1d.png"), dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
