// scripts/lib/pvnp-phase1-mlp-core.mjs
//
// JS forward-pass for the small_mlp policy trained by
// training/pvnp/train_mlp_policy.py. Loads the exported JSON weights and
// runs (in_dim → hidden → hidden → out_dim) with ReLU and tanh output.

import { readFile } from "node:fs/promises";

const PROBE_OFFSET_R = 0.04;
const GOAL_CENTER = { x: 0.925, y: 0.5 };

function relu(x) { return x > 0 ? x : 0; }

function matmulAdd(W, b, x) {
  // W: out × in (array of arrays); b: out; x: in.
  const out = new Float32Array(b.length);
  for (let o = 0; o < W.length; o += 1) {
    let s = b[o];
    const row = W[o];
    for (let i = 0; i < row.length; i += 1) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

export async function loadMlpWeights(jsonPath) {
  const text = await readFile(jsonPath, "utf8");
  const obj = JSON.parse(text);
  const layers = obj.layers.map((L) => ({
    name: L.name,
    W: L.weight, // [out][in]
    b: L.bias,
  }));
  return { meta: obj.meta, layers };
}

// Featurize observation matching training/pvnp/train_mlp_policy.py featurize().
function featurize(obs) {
  const byOffset = new Map();
  for (const p of obs.probes) {
    byOffset.set(`${Math.round(p.dx / PROBE_OFFSET_R)},${Math.round(p.dy / PROBE_OFFSET_R)}`, p.value);
  }
  const center = byOffset.get("0,0") ?? 0;
  const plusX = byOffset.get("1,0") ?? center;
  const minusX = byOffset.get("-1,0") ?? center;
  const plusY = byOffset.get("0,1") ?? center;
  const minusY = byOffset.get("0,-1") ?? center;
  const goalDx = GOAL_CENTER.x - obs.pos.x;
  const goalDy = GOAL_CENTER.y - obs.pos.y;
  const norm = Math.hypot(goalDx, goalDy) || 1;
  return new Float32Array([
    obs.pos.x, obs.pos.y, center, plusX, minusX, plusY, minusY,
    goalDx / norm,
  ]);
}

// Returns a step function compatible with the policies-core convention:
//   stepFn(obs, env) → { dx, dy } scaled to env.max_action_step
export function makeMlpStepFn(loadedWeights) {
  const layers = loadedWeights.layers;
  return function mlpStep(obs, env) {
    let x = featurize(obs);
    // fc1 → relu
    x = matmulAdd(layers[0].W, layers[0].b, x);
    for (let i = 0; i < x.length; i += 1) x[i] = relu(x[i]);
    // fc2 → relu
    x = matmulAdd(layers[1].W, layers[1].b, x);
    for (let i = 0; i < x.length; i += 1) x[i] = relu(x[i]);
    // fc3 → tanh
    x = matmulAdd(layers[2].W, layers[2].b, x);
    for (let i = 0; i < x.length; i += 1) x[i] = Math.tanh(x[i]);
    // Renormalize to unit step then scale.
    const dx = x[0];
    const dy = x[1];
    const n = Math.hypot(dx, dy) || 1;
    return { dx: (dx / n) * env.max_action_step, dy: (dy / n) * env.max_action_step };
  };
}
