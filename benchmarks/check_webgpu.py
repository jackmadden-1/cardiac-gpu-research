"""
Check the WebGPU kernels against the NumPy solvers.

Serves webgpu/ on a local port, opens webgpu/test.html in headless Chromium
(Playwright), reads back the v field after N steps for both the FD and the
LBM shader, and compares it with euler_fd.py / lattice_boltzmann/lbm_fhn.py
run from the same initial condition.

The GPU runs in float32 and NumPy in float64, so expect small differences
(around 1e-4 or less), not exact equality.

Needs:  pip install playwright   (then: playwright install chromium)
Run from the project root:
    python benchmarks/check_webgpu.py --n 64 --steps 2000
"""

import argparse
import functools
import http.server
import os
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import euler_fd  # noqa: E402
from fhn_model import FHNParams, resting_state  # noqa: E402
from lattice_boltzmann import lbm_fhn  # noqa: E402

CHROME_ARGS = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-vulkan=swiftshader",
    "--use-webgpu-adapter=swiftshader",
    "--disable-vulkan-surface",
    "--ignore-gpu-blocklist",
]


def numpy_reference(n, steps, dt):
    p = FHNParams()
    vr, wr = resting_state(p)
    v = np.full((n, n), vr)
    w = np.full((n, n), wr)
    s = n / 200
    Y, X = np.mgrid[0:n, 0:n]
    lower = Y < n / 2
    v[lower & (X >= 95 * s) & (X < 105 * s)] = 1.5
    w[lower & (X >= 80 * s) & (X < 95 * s)] = 1.0
    z = np.zeros_like(v)

    vf, wf = v.copy(), w.copy()
    for _ in range(steps):
        vf, wf = euler_fd.step(vf, wf, z, dt, 1.0, p)

    # The WebGPU kernel stores post-collision populations and streams first
    # ("pull"), so its first step streams the initial equilibrium once before
    # colliding, and the v it reports after K steps is the sum of the populations
    # it pulled in step K (before that step's collision). That equals the NumPy
    # state after K - 1 steps started from the streamed equilibrium.
    f = lbm_fhn.stream_2d(lbm_fhn.init_populations(v, ndim=2))
    wl = w.copy()
    tau = lbm_fhn.tau_from(p.D, 1.0, dt)
    for _ in range(steps - 1):
        f, wl = lbm_fhn.step_2d(f, wl, z, dt, tau, p)
    return vf, lbm_fhn.density(f)


def gpu_result(n, steps, dt):
    from playwright.sync_api import sync_playwright

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=os.path.join(ROOT, "webgpu"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}/test.html?n={n}&steps={steps}&dt={dt}"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=CHROME_ARGS)
            page = browser.new_page()
            page.goto(url)
            page.wait_for_function("window.RESULT !== undefined", timeout=600_000)
            res = page.evaluate("window.RESULT")
            browser.close()
    finally:
        httpd.shutdown()
    if "error" in res:
        raise RuntimeError(res["error"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()

    res = gpu_result(a.n, a.steps, a.dt)
    v_fd, v_lbm = numpy_reference(a.n, a.steps, a.dt)
    for name, gpu, ref in (("FD ", res["fd"], v_fd), ("LBM", res["lbm"], v_lbm)):
        gpu = np.asarray(gpu).reshape(a.n, a.n)
        err = np.abs(gpu - ref)
        print(f"{name}: max |GPU - NumPy| = {err.max():.2e}, mean = {err.mean():.2e}, "
              f"v range GPU [{gpu.min():.3f}, {gpu.max():.3f}]")
    print(f"GPU wall time (software adapter here, not meaningful for speed): "
          f"FD {res['fd_ms']:.0f} ms, LBM {res['lbm_ms']:.0f} ms")
