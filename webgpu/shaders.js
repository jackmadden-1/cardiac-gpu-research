// WGSL compute + render shaders for the FHN WebGPU starter.
// Imported by index.html (interactive) and test.html (numerical check).

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------
export const COMMON = /* wgsl */`
struct Params {
  nx: u32, ny: u32, stimOn: u32, pad0: u32,
  dt: f32, dx: f32, D: f32, a: f32,
  b: f32, eps: f32, tau: f32, stimX: f32,
  stimY: f32, stimR: f32, stimAmp: f32, pad1: f32,
};
@group(0) @binding(0) var<uniform> P: Params;

fn stimulus(x: u32, y: u32) -> f32 {
  if (P.stimOn == 0u) { return 0.0; }
  let d = vec2f(f32(x), f32(y)) - vec2f(P.stimX, P.stimY);
  return select(0.0, P.stimAmp, dot(d, d) < P.stimR * P.stimR);
}
`;

// Forward Euler + 5-point Laplacian. Zero-flux boundary = clamp to the edge
// (ghost node copies the edge value), same as euler_fd.py.
export const FD_WGSL = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage, read> src: array<vec2f>;       // (v, w)
@group(0) @binding(2) var<storage, read_write> dst: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> vis: array<f32>;

fn at(x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, i32(P.nx) - 1);
  let cy = clamp(y, 0, i32(P.ny) - 1);
  return src[u32(cy) * P.nx + u32(cx)].x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let i = g.y * P.nx + g.x;
  let s = src[i];
  let v = s.x; let w = s.y;
  let x = i32(g.x); let y = i32(g.y);
  let lap = (at(x + 1, y) + at(x - 1, y) + at(x, y + 1) + at(x, y - 1) - 4.0 * v) / (P.dx * P.dx);
  let rv = v - v * v * v / 3.0 - w + stimulus(g.x, g.y);
  let rw = P.eps * (v + P.a - P.b * w);
  let vn = v + P.dt * (P.D * lap + rv);
  dst[i] = vec2f(vn, w + P.dt * rw);
  vis[i] = vn;
}
`;

// D2Q5 LBM, "pull" streaming fused with collision.
// Stored values are post-collision populations, SoA layout f[k * N + i].
// Directions: 0 rest, 1 +x, 2 -x, 3 +y, 4 -y. Halfway bounce-back at walls.
export const LBM_WGSL = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> vis: array<f32>;
@group(0) @binding(4) var<storage, read_write> wbuf: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let N = P.nx * P.ny;
  let i = g.y * P.nx + g.x;

  // --- stream (pull) ---
  let f0 = src[i];
  let f1 = select(src[N + i - 1u],       src[2u * N + i], g.x == 0u);          // +x from left
  let f2 = select(src[2u * N + i + 1u],  src[N + i],      g.x == P.nx - 1u);   // -x from right
  let f3 = select(src[3u * N + i - P.nx], src[4u * N + i], g.y == 0u);         // +y from below
  let f4 = select(src[4u * N + i + P.nx], src[3u * N + i], g.y == P.ny - 1u);  // -y from above
  // note: select(f, t, cond) evaluates both branches; out-of-range reads at the
  // edges are clamped by WGSL robustness rules and their value is discarded.

  // --- collide ---
  let v = f0 + f1 + f2 + f3 + f4;
  let w = wbuf[i];
  let rv = v - v * v * v / 3.0 - w + stimulus(g.x, g.y);
  let rw = P.eps * (v + P.a - P.b * w);
  let om = 1.0 / P.tau;
  let w0 = 1.0 / 3.0; let wi = 1.0 / 6.0;
  dst[i]          = f0 - om * (f0 - w0 * v) + P.dt * w0 * rv;
  dst[N + i]      = f1 - om * (f1 - wi * v) + P.dt * wi * rv;
  dst[2u * N + i] = f2 - om * (f2 - wi * v) + P.dt * wi * rv;
  dst[3u * N + i] = f3 - om * (f3 - wi * v) + P.dt * wi * rv;
  dst[4u * N + i] = f4 - om * (f4 - wi * v) + P.dt * wi * rv;
  wbuf[i] = w + P.dt * rw;
  vis[i] = v;
}
`;

export const RENDER_WGSL = COMMON + /* wgsl */`
@group(0) @binding(1) var<storage, read> vis: array<f32>;
struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) k: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4f(p[k], 0.0, 1.0);
  o.uv = 0.5 * (p[k] + 1.0);
  return o;
}
fn inferno(t: f32) -> vec3f {
  let c0 = vec3f(0.0002, 0.0016, -0.0194); let c1 = vec3f(0.1065, 0.5640, 3.9327);
  let c2 = vec3f(11.6024, -3.9728, -15.9424); let c3 = vec3f(-41.7040, 17.4364, 44.3541);
  let c4 = vec3f(77.1629, -33.4023, -81.8073); let c5 = vec3f(-71.3194, 32.6261, 73.2095);
  let c6 = vec3f(25.1311, -12.2426, -23.0703);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let x = min(u32(in.uv.x * f32(P.nx)), P.nx - 1u);
  let y = min(u32(in.uv.y * f32(P.ny)), P.ny - 1u);
  let v = vis[y * P.nx + x];
  let t = clamp((v + 2.0) / 4.0, 0.0, 1.0);
  return vec4f(clamp(inferno(t), vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

