# cardiac-gpu-research

GPU-accelerated simulation of cardiac electrical propagation with the
two-variable FitzHugh-Nagumo (FHN) model. The project compares a
forward-Euler / finite-difference (FD) solver against a lattice Boltzmann
method (LBM) solver, with WebGPU as the target compute platform.

## Layout

| Path | What it is |
| --- | --- |
| `fhn_single_cell.py` | Single-cell FHN, forward Euler (your original script) |
| `fhn_1d.py` | 1D cable, forward Euler + FD (your original script) |
| `fhn_model.py` | Shared FHN parameters, reaction terms, resting state, Euler stability limit |
| `euler_fd.py` | FD solver (1D/2D), zero-flux ghost-node boundaries |
| `lattice_boltzmann/lbm_fhn.py` | LBM solver: D1Q3 and D2Q5, BGK collision, bounce-back walls |
| `fhn_2d.py` | 2D spiral-wave demo; set `METHOD = "euler"` or `"lbm"` |
| `benchmarks/compare_euler_lbm.py` | Convergence, time-step sweep and CPU throughput, writes to `results/` |
| `benchmarks/check_webgpu.py` | Runs the WebGPU kernels headless and checks them against NumPy |
| `webgpu/index.html` | Interactive WebGPU demo (both methods, click to stimulate) |
| `webgpu/shaders.js` | WGSL compute + render shaders |
| `webgpu/test.html` | Compute-only page used by `check_webgpu.py` |

## Quick start

```bash
pip install -r requirements.txt
python fhn_2d.py                                   # spiral wave, saves results/fhn_2d_spiral_<method>.png
python benchmarks/compare_euler_lbm.py --quick     # ~10 s smoke test
python benchmarks/compare_euler_lbm.py             # full run, a few minutes

# WebGPU demo (needs a local server, not file://)
cd webgpu && python -m http.server 8000           # then open http://localhost:8000
```

Optional, to check the GPU kernels against NumPy:

```bash
pip install playwright && playwright install chromium
python benchmarks/check_webgpu.py --n 200 --steps 1000
```

## Both solvers solve the same problem

* Same reaction terms (`fhn_model.reaction`) and parameters (a = 0.7, b = 0.8, epsilon = 0.08, D = 1).
* Same cell-centred grid, with the zero-flux wall half a cell outside the edge nodes
  (FD: ghost node copies the edge value; LBM: halfway bounce-back).
* Simulations start from the true resting state (v = -1.1994, w = -0.6243).
  The original scripts start from (-1, -0.5), which causes a small start-up transient.

## Key relations

* Euler stability for diffusion: dt <= dx^2 / (2 d D), so dx^2/(2D) in 1D and dx^2/(4D) in 2D.
* LBM diffusion: D = c_s^2 (tau - 1/2) dx^2 / dt with c_s^2 = 1/3, so tau = 1/2 + 3 D dt / dx^2.

## First results (CPU, NumPy)

* Both methods converge at second order in dx toward the same conduction velocity
  (about 0.812 in model units). At the same dx and dt = 0.2 dx^2, LBM's
  activation-time error is about 15-20% lower than FD's.
* LBM stays numerically stable beyond the Euler limit, but its accuracy degrades
  quickly as tau grows (BGK collision). Above dt/dx^2 of about 1, the wave stops
  propagating. TRT or MRT collision operators are the obvious next step.
* The WebGPU kernels reproduce the NumPy results to float32 precision.
