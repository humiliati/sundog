// scripts/lib/pvnp-phase1-policies-core.mjs
//
// Hand-coded local-probe policies for SUNDOG_V_P_V_NP Phase 1.
//
// A policy receives an Observation each step:
//   { pos: {x,y}, probes: [{dx,dy,value}], step: t }
// and returns an action:
//   { dx, dy }   // capped to env.max_action_step
//
// Policies do NOT have access to env.hidden_state. They only see local
// probe samples (which the env provides through a sensor layer).
//
// Local-probe convention: each step the env returns 5 probe samples at
// offsets (0,0) and (±r,0), (0,±r) for r = 0.04. Each value is the noisy
// reading of the underlying signed-distance-to-basin field.

import { makeRng, gaussian, uniform } from "./pvnp-phase1-rng.mjs";

export const POLICY_CLASSES = Object.freeze({
  HC_SAFE: "hc_safe_seeker",
  HC_DECOY: "hc_decoy_seeker",
  MLP_SMALL: "small_mlp",
});

export const GOAL_CENTER = Object.freeze({ x: 0.925, y: 0.5 });

function normalize(dx, dy) {
  const n = Math.hypot(dx, dy);
  if (n < 1e-12) return { dx: 0, dy: 0 };
  return { dx: dx / n, dy: dy / n };
}

function clampStep({ dx, dy }, max) {
  const n = Math.hypot(dx, dy);
  if (n <= max) return { dx, dy };
  return { dx: (dx / n) * max, dy: (dy / n) * max };
}

// Estimate field gradient via finite differences from probe samples.
// Probe layout: [center, +x, -x, +y, -y] (we tolerate other orders if marked).
function estimateGradient(probes) {
  // Default: search by offset signs.
  const center = probes.find((p) => p.dx === 0 && p.dy === 0)?.value ?? probes[0].value;
  const plusX = probes.find((p) => p.dx > 0 && p.dy === 0)?.value ?? center;
  const minusX = probes.find((p) => p.dx < 0 && p.dy === 0)?.value ?? center;
  const plusY = probes.find((p) => p.dx === 0 && p.dy > 0)?.value ?? center;
  const minusY = probes.find((p) => p.dx === 0 && p.dy < 0)?.value ?? center;
  const r = Math.abs(probes.find((p) => p.dx !== 0 || p.dy !== 0)?.dx ?? 0.04);
  return {
    gx: (plusX - minusX) / (2 * r),
    gy: (plusY - minusY) / (2 * r),
    center,
  };
}

// Hand-coded safe-seeking policy.
//
// Strategy: gradient-ascend on the signed-distance field (which points away
// from the basin) blended with a goal-seeking direction. The blend weight
// grows when the local probe value (≈ distance to basin) is small.
export function hcSafeSeekerStep(obs, env) {
  const goalDx = GOAL_CENTER.x - obs.pos.x;
  const goalDy = GOAL_CENTER.y - obs.pos.y;
  const goalDir = normalize(goalDx, goalDy);

  const grad = estimateGradient(obs.probes);
  const safetyDir = normalize(grad.gx, grad.gy);

  // Field-magnitude based blend: low value → bigger safety weight.
  const danger = Math.max(0, 0.18 - grad.center);
  const safetyWeight = Math.min(0.85, 0.15 + 4.0 * danger);
  const goalWeight = 1 - safetyWeight;

  const dir = normalize(
    goalWeight * goalDir.dx + safetyWeight * safetyDir.dx,
    goalWeight * goalDir.dy + safetyWeight * safetyDir.dy,
  );
  return clampStep({ dx: dir.dx * env.max_action_step, dy: dir.dy * env.max_action_step }, env.max_action_step);
}

// Hand-coded decoy / unsafe-seeking policy.
//
// Strategy: greedy goal-seeker with no safety steering. Often blunders into
// basins, giving the verifier negative cases to discriminate. Adds a small
// deterministic perturbation so trajectories aren't strictly straight lines.
export function hcDecoySeekerStep(obs, env) {
  const goalDx = GOAL_CENTER.x - obs.pos.x;
  const goalDy = GOAL_CENTER.y - obs.pos.y;
  const goalDir = normalize(goalDx, goalDy);
  // Small step-modulated perturbation in y so different envs see different paths.
  const wobble = 0.10 * Math.sin(obs.step * 0.19);
  const dir = normalize(goalDir.dx, goalDir.dy + wobble);
  return clampStep({ dx: dir.dx * env.max_action_step, dy: dir.dy * env.max_action_step }, env.max_action_step);
}

// Policy lookup by class name. Returns a step function.
export function policyStepFnByClass(policyClass) {
  switch (policyClass) {
    case POLICY_CLASSES.HC_SAFE: return hcSafeSeekerStep;
    case POLICY_CLASSES.HC_DECOY: return hcDecoySeekerStep;
    default:
      throw new Error(`No step fn for policy class ${policyClass} (MLP loads via separate path)`);
  }
}

// Deterministic start position inside the start region, keyed off env id.
export function sampleStartPosition(envId) {
  const rng = makeRng(`${envId}::start`);
  return {
    x: uniform(rng, 0.05, 0.10),
    y: uniform(rng, 0.30, 0.70),
  };
}
