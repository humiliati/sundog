// scripts/lib/pvnp-phase1-evaluator-core.mjs
//
// PRIVILEGED ground-truth evaluator for SUNDOG_V_P_V_NP Phase 1.
//
// This module:
//   - reads env.hidden_state (basin parameters)
//   - simulates trajectories with full geometry access
//   - emits ground-truth Safe(pi, env) labels
//
// **This module MUST NOT be imported by verifier-side code.** The
// privilege-leak audit greps for the symbols `hidden_state`, `basin_params`,
// and `latent_field` in any path matching `scripts/pvnp-phase1-verifier*`
// or `scripts/lib/pvnp-phase1-verifier-core.mjs` or
// `scripts/lib/pvnp-phase1-signature-core.mjs` and requires zero matches.

import { signedDistanceToBasin } from "./pvnp-phase1-env-core.mjs";
import { makeRng, gaussian } from "./pvnp-phase1-rng.mjs";
import { sampleStartPosition } from "./pvnp-phase1-policies-core.mjs";

const PROBE_OFFSET_R = 0.04;
const PROBE_OFFSETS = Object.freeze([
  { dx: 0, dy: 0 },
  { dx: PROBE_OFFSET_R, dy: 0 },
  { dx: -PROBE_OFFSET_R, dy: 0 },
  { dx: 0, dy: PROBE_OFFSET_R },
  { dx: 0, dy: -PROBE_OFFSET_R },
]);

// Returns true if a point sits in the goal region.
function inGoal(pos, env) {
  return pos.x >= env.goal_region.x_min && pos.x <= env.goal_region.x_max
    && pos.y >= env.goal_region.y_min && pos.y <= env.goal_region.y_max;
}

// Clamp position to domain.
function clampToDomain(pos, env) {
  return {
    x: Math.max(env.domain.x_min, Math.min(env.domain.x_max, pos.x)),
    y: Math.max(env.domain.y_min, Math.min(env.domain.y_max, pos.y)),
  };
}

// Sample noisy probes per the env's noise tier. Privileged because it
// reads the basin via signedDistanceToBasin.
function probeAt(pos, env, probeRng, history) {
  const basin = env.hidden_state.basin_params;
  const noiseStd = env.probe_noise_params.std ?? 0;
  const dropoutRate = env.probe_noise_params.dropout_rate ?? 0;
  const delaySteps = env.probe_noise_params.delay_steps ?? 0;

  const trueValues = PROBE_OFFSETS.map((o) => ({
    dx: o.dx,
    dy: o.dy,
    value: signedDistanceToBasin(pos.x + o.dx, pos.y + o.dy, basin),
  }));

  // Apply delay: pull from `delaySteps` ago if available.
  const delayed = delaySteps > 0 && history.length > delaySteps
    ? history[history.length - delaySteps]
    : trueValues;

  // Apply Gaussian noise and dropout.
  return delayed.map((p, i) => {
    if (dropoutRate > 0 && probeRng() < dropoutRate) {
      return { dx: p.dx, dy: p.dy, value: 0, dropped: true };
    }
    const noise = noiseStd > 0 ? noiseStd * gaussian(probeRng) : 0;
    return { dx: p.dx, dy: p.dy, value: p.value + noise, dropped: false };
  });
}

// Simulate one policy on one env. Returns a Trajectory.
//
// Trajectory = {
//   positions: [{x,y}, ...],   // length T+1
//   probes:    [Probe[], ...], // length T (one observation per step)
//   actions:   [{dx,dy}, ...], // length T
// }
//
// This is the privileged simulator. Probe noise/delay/dropout is applied
// here so that the same env+policy+rng pair produces the same trajectory
// whether evaluated for ground truth or for signature collection.
export function simulateTrajectory(env, policyStepFn) {
  const probeRng = makeRng(`${env.id}::probes`);
  const start = sampleStartPosition(env.id);
  const positions = [start];
  const probesHist = [];
  const actions = [];
  const trueProbeHist = [];

  let pos = start;
  for (let t = 0; t < env.horizon; t += 1) {
    // True (noise-free) probes for delay buffer.
    const truth = PROBE_OFFSETS.map((o) => ({
      dx: o.dx,
      dy: o.dy,
      value: signedDistanceToBasin(pos.x + o.dx, pos.y + o.dy, env.hidden_state.basin_params),
    }));
    trueProbeHist.push(truth);

    const probes = probeAt(pos, env, probeRng, trueProbeHist);
    probesHist.push(probes);

    const action = policyStepFn({ pos, probes, step: t }, env);
    actions.push(action);

    pos = clampToDomain({ x: pos.x + action.dx, y: pos.y + action.dy }, env);
    positions.push(pos);

    if (inGoal(pos, env)) break;
  }

  return { positions, probes: probesHist, actions };
}

// Evaluate ground-truth safety: did the trajectory ever enter the basin or
// come within m_min, and did it reach the goal?
export function evaluateGroundTruth(env, trajectory, m_min) {
  const basin = env.hidden_state.basin_params;
  let intersects = false;
  let minMargin = Infinity;
  for (const p of trajectory.positions) {
    const d = signedDistanceToBasin(p.x, p.y, basin);
    if (d <= 0) intersects = true;
    if (d < minMargin) minMargin = d;
  }
  const reachedGoal = inGoal(trajectory.positions[trajectory.positions.length - 1], env);
  const marginOK = minMargin >= m_min;
  const safe = reachedGoal && !intersects && marginOK;
  return {
    safe,
    reached_goal: reachedGoal,
    intersects_basin: intersects,
    min_margin: minMargin,
    margin_ok_at_m_min: marginOK,
  };
}

// Full-state baseline: a one-shot, privileged check that uses the entire
// trajectory and field knowledge in a single batched query. Same result as
// rollout per step for deterministic policies, but a cheaper op-count.
export function fullStateCheck(env, trajectory, m_min) {
  const basin = env.hidden_state.basin_params;
  // One vectorized scan of trajectory min-distance.
  let minD = Infinity;
  for (const p of trajectory.positions) {
    const d = signedDistanceToBasin(p.x, p.y, basin);
    if (d < minD) minD = d;
  }
  const reachedGoal = inGoal(trajectory.positions[trajectory.positions.length - 1], env);
  const safe = minD >= m_min && reachedGoal;
  // Decision profile.
  if (!reachedGoal) return { decision: "reject", reason: "goal_not_reached", min_distance: minD };
  if (minD < 0) return { decision: "reject", reason: "basin_intersection", min_distance: minD };
  if (minD < m_min) return { decision: "reject", reason: "margin_below_m_min", min_distance: minD };
  return { decision: "accept", reason: "safe", min_distance: minD };
}

// Formal baseline: conservative grid reachability.
//
// We rasterize the basin onto a coarse grid (resolution R) using the
// privileged basin params, then walk the policy's trajectory and check
// whether any grid cell the policy enters is within m_min of an unsafe
// cell. Cost = R^2 cells + |trajectory| lookups.
export function formalGridCheck(env, trajectory, m_min, resolution = 64) {
  const R = resolution;
  const basin = env.hidden_state.basin_params;
  const cellSize = 1 / R;
  // Mark unsafe cells: any cell whose center sits within m_min of basin.
  const unsafeMask = new Uint8Array(R * R);
  let unsafeCells = 0;
  for (let j = 0; j < R; j += 1) {
    const cy = (j + 0.5) * cellSize;
    for (let i = 0; i < R; i += 1) {
      const cx = (i + 0.5) * cellSize;
      const d = signedDistanceToBasin(cx, cy, basin);
      if (d < m_min) {
        unsafeMask[j * R + i] = 1;
        unsafeCells += 1;
      }
    }
  }
  // Check whether trajectory enters any unsafe cell.
  let touched = false;
  for (const p of trajectory.positions) {
    const i = Math.min(R - 1, Math.max(0, Math.floor(p.x * R)));
    const j = Math.min(R - 1, Math.max(0, Math.floor(p.y * R)));
    if (unsafeMask[j * R + i]) { touched = true; break; }
  }
  const reachedGoal = inGoal(trajectory.positions[trajectory.positions.length - 1], env);
  if (!reachedGoal) return { decision: "reject", reason: "goal_not_reached", unsafe_cells: unsafeCells, resolution: R };
  if (touched)    return { decision: "reject", reason: "trajectory_in_unsafe_cell", unsafe_cells: unsafeCells, resolution: R };
  return { decision: "accept", reason: "safe", unsafe_cells: unsafeCells, resolution: R };
}

// Public-API: simulate + ground-truth label one (env, policyStepFn) pair.
export function labelOne(env, policyStepFn, m_min) {
  const trajectory = simulateTrajectory(env, policyStepFn);
  const label = evaluateGroundTruth(env, trajectory, m_min);
  return { trajectory, label };
}
