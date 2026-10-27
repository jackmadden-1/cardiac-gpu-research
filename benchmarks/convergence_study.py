"""
Task 1: convergence of forward Euler (FD) and lattice Boltzmann (LBM)
for different dt, dx and tau, and the cost of reaching a converged value.

Model
-----
--model modified (default): the modified FitzHugh-Nagumo model from
  Fenton's group (Rheaume et al., CinC 2023), rest at u = 0, peak ~1.19
  in tissue, APD ~300.
--model standard: Fenton's standard FHN (linear recovery, mu = delta = 1,
  theta = 0; FentonFHNParams.standard()), peak ~0.88, APD ~75.
Each model has its own run length and APD threshold (MODELS below).

Setup (1D cable, length L = 100, zero-flux ends)
------------------------------------------------
- Stimulus I = 1 for x < 2 during t < 1, then run to t = 400.
- Activation time at every node: upward crossing of u = 0.5,
  linearly interpolated inside the time step.
- CV: (75 - 25) / (t_act(75) - t_act(25)).
- APD at x = 50: time between upward and downward crossings of
  u = U_APD (about 10% of the tissue peak), both interpolated.

Sweeps
------
- Euler: dx in DXS, and r = D dt / dx^2 in R_EULER (stable for r <= 0.5).
- LBM:   same dx, tau in TAUS; dt = (tau - 1/2) dx^2 / (3 D).
  So both methods cover the same (dx, dt) plane.

Reference
---------
Euler at dx = 0.125 and 0.0625 (r = 0.05), Richardson-extrapolated
assuming O(dx^2) error at fixed r (time error O(dt) = O(dx^2) there).
The finest run alone is also reported as a check.

Cost
----
- node-updates = nodes x steps (hardware-independent)
- NumPy wall time (CPU; LBM moves ~3x the data per node, so this
  is a CPU number, not the GPU comparison)
For each error tolerance, the cheapest run of each method that meets it
(for CV, APD, or both) is reported.

Outputs (prefix task1_ for modified, task1_standard_ for standard):
         <prefix>runs.csv, <prefix>summary.csv,
         <prefix>convergence.png, <prefix>cost.png

Run from the project root:
    python benchmarks/convergence_study.py           (full, ~10-20 min)
    python benchmarks/convergence_study.py --quick   (smoke test, ~1 min;
                                                      writes *_quick_* files)
    python benchmarks/convergence_study.py --model standard
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
from fhn_model import FentonFHNParams  # noqa: E402
from lattice_boltzmann import lbm_fhn  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

LENGTH = 100.0
T_END = 400.0    # overwritten per model in __main__
STIM_X, STIM_T, STIM_AMP = 2.0, 1.0, 1.0
X_A, X_B, X_APD = 25.0, 75.0, 50.0
U_ACT = 0.5
U_APD = 0.12     # overwritten per model in __main__

DXS = [1.0, 0.5, 0.25, 0.125]
R_EULER = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
TAUS = [0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 1.0, 1.1, 1.25, 1.5, 2.0]
PREFIX = "task1_"
MODELS = {
    # name: () -> (params, run length, APD threshold, output prefix)
    "modified": lambda: (FentonFHNParams(), 400.0, 0.12, "task1_"),
    "standard": lambda: (FentonFHNParams.standard(), 300.0, 0.09, "task1_standard_"),
}
TOLERANCES = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


def run_cable(method, dx, dt, p, t_end=None):
    """One 1D run. Returns dict with cv, apd, cost and timing (or stable=False)."""
    t_end = T_END if t_end is None else t_end
    n = int(round(LENGTH / dx))
    x = (np.arange(n) + 0.5) * dx
    u_rest, v_rest = p.rest()
    u = np.full(n, u_rest)
    v = np.full(n, v_rest)
    stim = np.zeros(n)
    stim_mask = x < STIM_X
    t_act = np.full(n, np.nan)
    i_apd = int(np.argmin(np.abs(x - X_APD)))
    apd_up = apd_down = np.nan

    if method == "lbm":
        tau = lbm_fhn.tau_from(p.D, dx, dt)
        f = lbm_fhn.init_populations(u, ndim=1)

    n_steps = int(round(t_end / dt))
    start = time.perf_counter()
    for k in range(n_steps):
        t = k * dt
        stim[:] = 0.0
        if t < STIM_T:
            stim[stim_mask] = STIM_AMP
        u_prev = u
        if method == "fd":
            u, v = euler_fd.step(u, v, stim, dt, dx, p)
        else:
            f, v = lbm_fhn.step_1d(f, v, stim, dt, tau, p)
            u = lbm_fhn.density(f)

        if k % 50 == 0 and (not np.all(np.isfinite(u)) or np.abs(u).max() > 1e3):
            return dict(stable=False, seconds=time.perf_counter() - start,
                        nodes=n, steps=k + 1)

        # interpolated activation crossings
        up = np.isnan(t_act) & (u_prev < U_ACT) & (u >= U_ACT)
        if up.any():
            frac = (U_ACT - u_prev[up]) / (u[up] - u_prev[up])
            t_act[up] = t + frac * dt
        # APD crossings at the recording node
        a, b = u_prev[i_apd], u[i_apd]
        if np.isnan(apd_up) and a < U_APD <= b:
            apd_up = t + (U_APD - a) / (b - a) * dt
        elif not np.isnan(apd_up) and np.isnan(apd_down) and a >= U_APD > b:
            apd_down = t + (a - U_APD) / (a - b) * dt

    secs = time.perf_counter() - start
    ta = np.interp(X_A, x, t_act)
    tb = np.interp(X_B, x, t_act)
    cv = (X_B - X_A) / (tb - ta) if np.isfinite(ta) and np.isfinite(tb) else np.nan
    apd = apd_down - apd_up
    return dict(stable=True, cv=cv, apd=apd, seconds=secs, nodes=n, steps=n_steps)


def reference(p, quick):
    """Richardson-extrapolated Euler reference for CV and APD."""
    r = 0.05
    if quick:
        pair = (0.5, 0.25)
    else:
        pair = (0.125, 0.0625)
    runs = [run_cable("fd", dx, r * dx ** 2 / p.D, p) for dx in pair]
    coarse, fine = runs
    ext = {k: (4 * fine[k] - coarse[k]) / 3 for k in ("cv", "apd")}
    print(f"reference (Euler, r = {r}): dx={pair[0]}: CV={coarse['cv']:.5f} APD={coarse['apd']:.3f}; "
          f"dx={pair[1]}: CV={fine['cv']:.5f} APD={fine['apd']:.3f}")
    print(f"  Richardson extrapolation: CV={ext['cv']:.5f} APD={ext['apd']:.3f}")
    return ext, dict(dx_coarse=pair[0], dx_fine=pair[1], cv_fine=fine["cv"],
                     apd_fine=fine["apd"], cv_coarse=coarse["cv"], apd_coarse=coarse["apd"])


def sweep(p, ref, quick):
    dxs = [1.0, 0.5] if quick else DXS
    r_list = [0.1, 0.4] if quick else R_EULER
    taus = [0.7, 0.9, 1.1] if quick else TAUS
    rows = []
    for dx in dxs:
        jobs = [("fd", r * dx ** 2 / p.D, "") for r in r_list]
        jobs += [("lbm", (tau - 0.5) * dx ** 2 / (3 * p.D), tau) for tau in taus]
        for method, dt, tau in jobs:
            res = run_cable(method, dx, dt, p)
            row = dict(method=method, dx=dx, dt=dt, r=p.D * dt / dx ** 2, tau=tau,
                       stable=res["stable"], nodes=res["nodes"], steps=res["steps"],
                       node_updates=res["nodes"] * res["steps"], seconds=res["seconds"])
            if res["stable"]:
                row.update(cv=res["cv"], apd=res["apd"],
                           cv_err=(res["cv"] - ref["cv"]) / ref["cv"],
                           apd_err=(res["apd"] - ref["apd"]) / ref["apd"])
            else:
                row.update(cv=np.nan, apd=np.nan, cv_err=np.nan, apd_err=np.nan)
            rows.append(row)
            tag = f"tau={tau}" if method == "lbm" else f"r={row['r']:.3f}"
            if res["stable"]:
                print(f"  {method:3s} dx={dx:<6} {tag:<10} dt={dt:.5f} "
                      f"CV err={row['cv_err']:+.2e} APD err={row['apd_err']:+.2e} {res['seconds']:.2f}s")
            else:
                print(f"  {method:3s} dx={dx:<6} {tag:<10} dt={dt:.5f} UNSTABLE")
    return rows


def cheapest(rows, tol, which):
    """Cheapest run per method meeting |error| <= tol for CV, APD, or both."""
    out = {}
    for method in ("fd", "lbm"):
        ok = []
        for q in rows:
            if q["method"] != method or not q["stable"] or not np.isfinite(q["cv_err"]):
                continue
            errs = {"cv": abs(q["cv_err"]), "apd": abs(q["apd_err"])}
            e = max(errs.values()) if which == "both" else errs[which]
            if np.isfinite(e) and e <= tol:
                ok.append(q)
        out[method] = min(ok, key=lambda q: q["node_updates"]) if ok else None
    return out


def write_csv(name, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def summarize(rows):
    print("\nCheapest run meeting each tolerance (node-updates; NumPy seconds):")
    out = []
    for which in ("cv", "apd", "both"):
        for tol in TOLERANCES:
            best = cheapest(rows, tol, which)
            line = dict(metric=which, tolerance=tol)
            for m in ("fd", "lbm"):
                q = best[m]
                line[f"{m}_dx"] = q["dx"] if q else ""
                line[f"{m}_dt"] = q["dt"] if q else ""
                line[f"{m}_tau"] = q["tau"] if q else ""
                line[f"{m}_node_updates"] = q["node_updates"] if q else ""
                line[f"{m}_seconds"] = q["seconds"] if q else ""
            fd_c, lb_c = line["fd_node_updates"], line["lbm_node_updates"]
            line["fd_over_lbm_cost"] = fd_c / lb_c if fd_c and lb_c else ""
            out.append(line)
            def fmt(m):
                q = best[m]
                if not q:
                    return "none in sweep"
                extra = f", tau={q['tau']}" if m == "lbm" else ""
                return f"dx={q['dx']}, dt={q['dt']:.4g}{extra}: {q['node_updates']:.2e} ({q['seconds']:.2f}s)"
            print(f"  {which:4s} {tol:>6.1%}  Euler {fmt('fd'):<48} LBM {fmt('lbm')}")
    write_csv(PREFIX + "summary.csv", out)
    return out


def plot(rows, ref, model):
    dxs = sorted({q["dx"] for q in rows}, reverse=True)
    shades_fd = plt.cm.Blues(np.linspace(0.45, 0.95, len(dxs)))
    shades_lb = plt.cm.Oranges(np.linspace(0.45, 0.95, len(dxs)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, dx in enumerate(dxs):
        for m, col, mk in (("fd", shades_fd[i], "o"), ("lbm", shades_lb[i], "s")):
            r = sorted([q for q in rows if q["method"] == m and q["dx"] == dx and q["stable"]
                        and np.isfinite(q["cv_err"])], key=lambda q: q["dt"])
            if not r:
                continue
            lab = f"{'Euler' if m == 'fd' else 'LBM'} dx={dx:g}"
            ax[0].loglog([q["dt"] for q in r], [max(abs(q["cv_err"]), 1e-7) for q in r],
                         marker=mk, ms=3.5, lw=1.1, color=col, label=lab)
            ax[1].loglog([q["dt"] for q in r], [max(abs(q["apd_err"]), 1e-7) for q in r],
                         marker=mk, ms=3.5, lw=1.1, color=col)
    ax[0].set_ylabel("|relative CV error|")
    ax[1].set_ylabel("|relative APD error|")
    for a in ax:
        a.set_xlabel("dt")
        a.grid(True, which="both", alpha=0.25)
    ax[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(f"Task 1: error vs dt at each dx ({model} model; reference CV={ref['cv']:.4f}, APD={ref['apd']:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, PREFIX + "convergence.png"), dpi=140)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m, col, mk, lab in (("fd", "#2a78d6", "o", "Euler"), ("lbm", "#eb6834", "s", "LBM")):
        r = [q for q in rows if q["method"] == m and q["stable"] and np.isfinite(q["cv_err"])]
        ax[0].loglog([q["node_updates"] for q in r], [max(abs(q["cv_err"]), 1e-7) for q in r],
                     mk, ms=4, color=col, alpha=0.8, label=lab)
        ax[1].loglog([q["node_updates"] for q in r], [max(abs(q["apd_err"]), 1e-7) for q in r],
                     mk, ms=4, color=col, alpha=0.8)
    ax[0].set_ylabel("|relative CV error|")
    ax[1].set_ylabel("|relative APD error|")
    for a in ax:
        a.set_xlabel("cost (node-updates)")
        a.grid(True, which="both", alpha=0.25)
    ax[0].legend(frameon=False)
    fig.suptitle("Task 1: accuracy vs cost (lower-left is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, PREFIX + "cost.png"), dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", choices=sorted(MODELS), default="modified")
    args = ap.parse_args()
    p, T_END, U_APD, PREFIX = MODELS[args.model]()
    if args.quick:
        PREFIX += "quick_"   # never overwrite full-run results with a smoke test
    ref, ref_info = reference(p, args.quick)
    rows = sweep(p, ref, args.quick)
    for q in rows:
        q["model"] = args.model
    write_csv(PREFIX + "runs.csv", rows)
    summarize(rows)
    plot(rows, ref, args.model)
    print("done")
