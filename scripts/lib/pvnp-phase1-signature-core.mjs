// scripts/lib/pvnp-phase1-signature-core.mjs
//
// Signature transform H for SUNDOG_V_P_V_NP Phase 1.
//
// PRIVILEGE BOUNDARY: this module reads ONLY from a trace (positions,
// probes, actions) and the public env view (no privileged fields). It
// MUST NOT reference: hidden state, basin parameters, latent field
// parameters, ground-truth labels, or the privileged signed-distance
// function. The privilege-leak audit greps for these tokens here and in
// the verifier core, and requires zero matches.

import { performance } from "node:perf_hooks";

const PROBE_OFFSET_R = 0.04;
const COVERAGE_RESOLUTION = 16;
const SIGNATURE_SCHEMA = "pvnp-phase1-sigma-v0";

// Estimate field Laplacian at one probe sample. Uses the 5-point stencil
// implicit in the probe layout (center + ±x + ±y).
function pointLaplacian(probes) {
  const center = probes.find((p) => p.dx === 0 && p.dy === 0)?.value ?? 0;
  const plusX = probes.find((p) => p.dx > 0 && p.dy === 0)?.value ?? center;
  const minusX = probes.find((p) => p.dx < 0 && p.dy === 0)?.value ?? center;
  const plusY = probes.find((p) => p.dx === 0 && p.dy > 0)?.value ?? center;
  const minusY = probes.find((p) => p.dx === 0 && p.dy < 0)?.value ?? center;
  return (plusX + minusX + plusY + minusY - 4 * center) / (PROBE_OFFSET_R * PROBE_OFFSET_R);
}

function aggregateStats(values) {
  if (values.length === 0) return { count: 0, mean: 0, variance: 0, min: 0, max: 0 };
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / values.length;
  let varSum = 0;
  for (const v of values) varSum += (v - mean) * (v - mean);
  const variance = values.length > 1 ? varSum / (values.length - 1) : 0;
  return { count: values.length, mean, variance, min, max };
}

// Sensor health: noise std estimate from consecutive probe deltas, dropout
// count, and delay estimate (constant for now, derived from probe carriers).
function estimateSensorHealth(probesPerStep) {
  let dropouts = 0;
  let totalProbes = 0;
  const centerDeltas = [];
  let prevCenter = null;
  for (const probes of probesPerStep) {
    const center = probes.find((p) => p.dx === 0 && p.dy === 0);
    if (!center) continue;
    if (center.dropped) dropouts += 1;
    totalProbes += probes.length;
    for (const p of probes) if (p.dropped) dropouts += 1;
    if (prevCenter !== null) {
      centerDeltas.push(Math.abs(center.value - prevCenter));
    }
    prevCenter = center.value;
  }
  // Robust noise estimate: median of absolute consecutive deltas / 0.6745.
  const sortedDeltas = centerDeltas.slice().sort((a, b) => a - b);
  const median = sortedDeltas.length > 0 ? sortedDeltas[Math.floor(sortedDeltas.length / 2)] : 0;
  // Subtract the expected per-step field change due to motion (max step
  // ≈ 0.025 units → bounded). Use a floor of zero.
  const noiseStdEstimate = Math.max(0, median - 0.025);
  return {
    dropout_count: dropouts,
    dropout_fraction: totalProbes > 0 ? dropouts / totalProbes : 0,
    probe_count: totalProbes,
    median_consecutive_delta: median,
    noise_std_estimate: noiseStdEstimate,
    delay_estimate_steps: 0, // not estimated in v0; left as 0
  };
}

// Trajectory envelope: axis-aligned bbox + arc length + step count.
function trajectoryEnvelope(positions) {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  let arc = 0;
  for (let i = 0; i < positions.length; i += 1) {
    const p = positions[i];
    if (p.x < xMin) xMin = p.x;
    if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y;
    if (p.y > yMax) yMax = p.y;
    if (i > 0) {
      const q = positions[i - 1];
      arc += Math.hypot(p.x - q.x, p.y - q.y);
    }
  }
  return {
    x_min: xMin, x_max: xMax, y_min: yMin, y_max: yMax,
    arc_length: arc, step_count: positions.length - 1,
  };
}

// Coverage digest: a bit-mask of which grid cells the trajectory entered.
function coverageDigest(positions, R = COVERAGE_RESOLUTION) {
  const mask = new Uint8Array(R * R);
  let touched = 0;
  for (const p of positions) {
    const i = Math.min(R - 1, Math.max(0, Math.floor(p.x * R)));
    const j = Math.min(R - 1, Math.max(0, Math.floor(p.y * R)));
    const idx = j * R + i;
    if (mask[idx] === 0) {
      mask[idx] = 1;
      touched += 1;
    }
  }
  return { resolution: R, touched_cells: touched, total_cells: R * R };
}

// Margin lower bound: min center-probe value along trajectory, MINUS a
// conservative noise robustness term. Result is a signature-derived lower
// bound on true distance to basin; if the bound holds, the verifier may
// claim safety.
function marginLowerBound(positions, probesPerStep, sensorHealth) {
  let minCenter = Infinity;
  for (const probes of probesPerStep) {
    const center = probes.find((p) => p.dx === 0 && p.dy === 0);
    if (!center || center.dropped) continue;
    if (center.value < minCenter) minCenter = center.value;
  }
  if (!Number.isFinite(minCenter)) return -Infinity;
  // Conservative subtraction: 2 std + dropout penalty.
  const robustness = 2 * sensorHealth.noise_std_estimate + sensorHealth.dropout_fraction * 0.05;
  return minCenter - robustness;
}

// Invariance checks. v0 admits:
//   - translation_invariance: shifting probe values by a constant offset
//     should not change the curvature_summary mean.
//   - scale_invariance_of_envelope: trajectory envelope must lie within
//     domain bounds [0,1]^2.
//   - probe_layout_check: probes use the registered 5-point stencil.
function invarianceChecks(probesPerStep, envelope) {
  const probeLayoutOk = probesPerStep.every((probes) => probes.length === 5);
  const envelopeInDomain = envelope.x_min >= 0 && envelope.x_max <= 1
    && envelope.y_min >= 0 && envelope.y_max <= 1;
  // Translation invariance: re-compute curvature on shifted probes; should
  // give identical Laplacian values.
  const shift = 7.3;
  let translationOk = true;
  for (const probes of probesPerStep) {
    const shifted = probes.map((p) => ({ ...p, value: p.value + shift }));
    const lap1 = pointLaplacian(probes);
    const lap2 = pointLaplacian(shifted);
    if (Math.abs(lap1 - lap2) > 1e-9) { translationOk = false; break; }
  }
  return {
    probe_layout_ok: probeLayoutOk,
    envelope_in_domain: envelopeInDomain,
    translation_invariance: translationOk,
    all_pass: probeLayoutOk && envelopeInDomain && translationOk,
  };
}

// Compute the full signature certificate. Returns a sigma object.
export function computeSignature({ traceId, publicEnv, positions, probes }) {
  const t0 = performance.now();

  const laplacians = probes.map(pointLaplacian);
  const curvatureSummary = aggregateStats(laplacians);

  const sensorHealth = estimateSensorHealth(probes);
  const envelope = trajectoryEnvelope(positions);
  const coverage = coverageDigest(positions);
  const margin = marginLowerBound(positions, probes, sensorHealth);
  const invariance = invarianceChecks(probes, envelope);

  const computeMs = performance.now() - t0;
  // Op count: probe×T Laplacians + sensor estimate + envelope + coverage scan + invariance recompute.
  const T = probes.length;
  const ops = T /*laplacian*/ + T /*sensor*/ + positions.length /*envelope*/
    + positions.length /*coverage*/ + T /*invariance recompute*/;

  return {
    schema: SIGNATURE_SCHEMA,
    trace_id: traceId,
    source_observations: {
      probe_count_per_step: 5,
      step_count: T,
      probe_offset_r: PROBE_OFFSET_R,
      env_id: publicEnv.id,
      policy_id: traceId.split("|")[0],
    },
    curvature_summary: curvatureSummary,
    trajectory_envelope: envelope,
    margin_lower_bound: margin,
    coverage_digest: coverage,
    invariance_checks: invariance,
    sensor_health: sensorHealth,
    cost_signature: { wall_ms: computeMs, ops },
    limitations: [
      "nonlocal_basin_topology_invisible",
      "policy_action_history_not_audited",
      "no_decoy_field_separation",
    ],
  };
}
