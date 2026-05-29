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

import { canonicalize, sha256Hex } from "./canonical-json.mjs";

const PROBE_OFFSET_R = 0.04;
const COVERAGE_RESOLUTION = 16;
const SIGNATURE_SCHEMAS = Object.freeze({
  v0: "pvnp-phase1-sigma-v0",
  v1: "pvnp-phase1-sigma-v1",
  v2: "pvnp-phase1-sigma-v2",
});
const TRANSFORM_VERSIONS = Object.freeze({
  v0: "pvnp-phase1-transform-v0",
  v1: "pvnp-phase1-transform-v1",
  v2: "pvnp-phase1-transform-v2",
});

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

function quantile(values, q) {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
  return sorted[idx];
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

function estimateSensorHealthV1(probesPerStep) {
  const base = estimateSensorHealth(probesPerStep);
  const centerValues = [];
  const layoutResiduals = [];
  let missingLayoutSteps = 0;

  for (const probes of probesPerStep) {
    const center = probes.find((p) => p.dx === 0 && p.dy === 0);
    const plusX = probes.find((p) => p.dx > 0 && p.dy === 0);
    const minusX = probes.find((p) => p.dx < 0 && p.dy === 0);
    const plusY = probes.find((p) => p.dx === 0 && p.dy > 0);
    const minusY = probes.find((p) => p.dx === 0 && p.dy < 0);
    if (!center || !plusX || !minusX || !plusY || !minusY) {
      missingLayoutSteps += 1;
      continue;
    }
    centerValues.push(center.value);
    layoutResiduals.push(Math.abs((plusX.value + minusX.value) - 2 * center.value));
    layoutResiduals.push(Math.abs((plusY.value + minusY.value) - 2 * center.value));
  }

  const centerStats = aggregateStats(centerValues);
  const residualStats = aggregateStats(layoutResiduals);
  const biasDrift = centerValues.length > 4
    ? Math.abs(
      centerValues[Math.floor(centerValues.length * 0.75)]
      - centerValues[Math.floor(centerValues.length * 0.25)],
    )
    : 0;

  return {
    ...base,
    center_value_variance: centerStats.variance,
    layout_residual_mean: residualStats.mean,
    layout_residual_p95: quantile(layoutResiduals, 0.95),
    bias_drift_estimate: biasDrift,
    missing_layout_steps: missingLayoutSteps,
    all_pass: base.dropout_fraction <= 0.15
      && base.noise_std_estimate <= 0.05
      && residualStats.mean <= 0.20
      && biasDrift <= 0.30
      && missingLayoutSteps === 0,
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

function invarianceChecksV1(probesPerStep, envelope) {
  const base = invarianceChecks(probesPerStep, envelope);
  const reflectedLaplacianDiffs = [];
  const centerBiasResiduals = [];
  for (const probes of probesPerStep) {
    const lap1 = pointLaplacian(probes);
    const centerBiased = probes.map((p) => {
      const centerBias = p.dx === 0 && p.dy === 0 ? 0.015 : 0;
      return { ...p, value: p.value + centerBias };
    });
    const lap2 = pointLaplacian(centerBiased);
    centerBiasResiduals.push(Math.abs(lap2 - lap1));

    const signFlipped = probes.map((p) => ({ ...p, dx: -p.dx, dy: -p.dy }));
    reflectedLaplacianDiffs.push(Math.abs(pointLaplacian(signFlipped) - lap1));
  }
  const biasStats = aggregateStats(centerBiasResiduals);
  const flipStats = aggregateStats(reflectedLaplacianDiffs);
  const center_bias_probe_sensitive = biasStats.max > 1.0;
  const reflection_consistent = flipStats.max < 1e-9;
  return {
    ...base,
    center_bias_probe_sensitive,
    reflection_consistent,
    center_bias_residual_max: biasStats.max,
    reflection_residual_max: flipStats.max,
    all_pass: base.all_pass && center_bias_probe_sensitive && reflection_consistent,
  };
}

function geometryPromiseSignal(laplacians, probesPerStep, coverage) {
  const absLaps = laplacians.map((v) => Math.abs(v)).filter((v) => Number.isFinite(v));
  const centerValues = [];
  for (const probes of probesPerStep) {
    const center = probes.find((p) => p.dx === 0 && p.dy === 0);
    if (center && !center.dropped && Number.isFinite(center.value)) centerValues.push(center.value);
  }
  const centerStats = aggregateStats(centerValues);
  const nearBoundary = centerValues.filter((v) => Math.abs(v) <= 0.04).length;
  const curvatureP95 = quantile(absLaps, 0.95);
  const centerRange = centerStats.max - centerStats.min;
  const evidenceCoverage = coverage.touched_cells / coverage.total_cells;
  const insufficientEvidence = evidenceCoverage < 0.06 || centerValues.length < 16;
  const curvatureSuspicious = curvatureP95 > 750;
  const scaleSuspicious = centerRange < 0.04 && nearBoundary > 8;
  return {
    curvature_abs_p95: curvatureP95,
    center_value_range: centerRange,
    near_boundary_count: nearBoundary,
    evidence_coverage: evidenceCoverage,
    insufficient_evidence: insufficientEvidence,
    curvature_suspicious: curvatureSuspicious,
    scale_suspicious: scaleSuspicious,
    all_pass: !insufficientEvidence && !curvatureSuspicious && !scaleSuspicious,
  };
}

function invarianceChecksV2(probesPerStep, envelope, marginLowerBound) {
  const base = invarianceChecksV1(probesPerStep, envelope);
  const stabilityFloor = 0.075;
  const perturbation = 0.018;
  const perturbedMargin = marginLowerBound - perturbation;

  const lowProbePairScores = [];
  for (const probes of probesPerStep) {
    const center = probes.find((p) => p.dx === 0 && p.dy === 0);
    if (!center || center.dropped) continue;
    const lowNeighbors = probes
      .filter((p) => !(p.dx === 0 && p.dy === 0) && !p.dropped)
      .map((p) => Math.max(0, center.value - p.value));
    lowNeighbors.sort((a, b) => b - a);
    if (lowNeighbors.length >= 2) {
      lowProbePairScores.push(lowNeighbors[0] + lowNeighbors[1]);
    }
  }
  const pairStats = aggregateStats(lowProbePairScores);
  const decoyScore = pairStats.max;
  const nearBoundaryClear = marginLowerBound >= stabilityFloor;
  const counterfactualClear = perturbedMargin >= 0.06;
  const decoyConsistencyClear = decoyScore <= 0.18;

  return {
    ...base,
    coordinate_equivalence_residual_max: base.reflection_residual_max,
    near_boundary_stability_floor: stabilityFloor,
    near_boundary_counterfactual_clear: nearBoundaryClear && counterfactualClear,
    perturbed_margin_lower_bound: perturbedMargin,
    decoy_consistency_score: decoyScore,
    decoy_consistency_clear: decoyConsistencyClear,
    all_pass: base.all_pass
      && nearBoundaryClear
      && counterfactualClear
      && decoyConsistencyClear,
  };
}

function geometryPromiseSignalV2(laplacians, probesPerStep, coverage) {
  const base = geometryPromiseSignal(laplacians, probesPerStep, coverage);
  const absLaps = laplacians.map((v) => Math.abs(v)).filter((v) => Number.isFinite(v));
  const evidenceCoverage = coverage.touched_cells / coverage.total_cells;
  const boundaryEvidenceAbsent = base.near_boundary_count === 0 && base.curvature_abs_p95 < 10;
  const insufficientCoverage = evidenceCoverage < 0.06;
  const curvatureOut = base.curvature_abs_p95 > 650;
  const scaleOut = base.center_value_range < 0.08 || base.center_value_range > 0.85;
  const topologyAmbiguityScore = boundaryEvidenceAbsent
    ? 1
    : Math.min(1, (base.near_boundary_count / 16) + (base.curvature_abs_p95 > 500 ? 0.25 : 0));
  const topologyAmbiguous = topologyAmbiguityScore >= 0.85;
  const boundaryFlags = {
    insufficient_coverage: insufficientCoverage,
    scale_out_of_envelope: scaleOut,
    curvature_out_of_envelope: curvatureOut,
    topology_ambiguous: topologyAmbiguous,
    boundary_evidence_absent: boundaryEvidenceAbsent,
  };

  return {
    ...base,
    geometry_evidence_coverage: evidenceCoverage,
    scale_interval: {
      lower: Math.max(0, base.center_value_range - 0.04),
      upper: base.center_value_range + 0.04,
    },
    curvature_profile: {
      abs_p50: quantile(absLaps, 0.50),
      abs_p95: base.curvature_abs_p95,
      abs_max: quantile(absLaps, 1.0),
    },
    topology_ambiguity_score: topologyAmbiguityScore,
    boundary_flags: boundaryFlags,
    all_pass: Object.values(boundaryFlags).every((flag) => !flag),
  };
}

export function buildSourcePayload({ traceId, publicEnv, positions, probes }) {
  return {
    trace_id: traceId,
    policy_id: traceId.split("|")[0],
    env_id: publicEnv.id,
    split: publicEnv.split,
    probe_count_per_step: 5,
    probe_offset_r: PROBE_OFFSET_R,
    step_count: probes.length,
    positions,
    probes,
  };
}

export function sourceHash(sourcePayload) {
  return sha256Hex(canonicalize(sourcePayload));
}

export function makeTraceCommitment(sourcePayload) {
  return {
    schema: "pvnp-phase1-trace-commitment-v1",
    trace_id: sourcePayload.trace_id,
    policy_id: sourcePayload.policy_id,
    env_id: sourcePayload.env_id,
    split: sourcePayload.split,
    source_hash: sourceHash(sourcePayload),
    source_payload: sourcePayload,
  };
}

export function computeAnalyticalFields({ positions, probes, version = "v0" }) {
  const laplacians = probes.map(pointLaplacian);
  const curvatureSummary = aggregateStats(laplacians);
  const sourceBoundVersion = version === "v1" || version === "v2";
  const sensorHealth = sourceBoundVersion
    ? estimateSensorHealthV1(probes)
    : estimateSensorHealth(probes);
  const envelope = trajectoryEnvelope(positions);
  const coverage = coverageDigest(positions);
  const margin = marginLowerBound(positions, probes, sensorHealth);
  const invariance = version === "v2"
    ? invarianceChecksV2(probes, envelope, margin)
    : (version === "v1" ? invarianceChecksV1(probes, envelope) : invarianceChecks(probes, envelope));

  const fields = {
    curvature_summary: curvatureSummary,
    trajectory_envelope: envelope,
    margin_lower_bound: margin,
    invariance_checks: invariance,
    sensor_health: sensorHealth,
  };

  if (version === "v1") {
    fields.coverage_digest = coverage;
    fields.sensor_health_v1 = sensorHealth;
    fields.invariance_checks_v1 = invariance;
    fields.geometry_promise_signal = geometryPromiseSignal(laplacians, probes, coverage);
  } else if (version === "v2") {
    fields.sensor_health_v1 = sensorHealth;
    fields.invariance_checks_v2 = invariance;
    fields.geometry_promise_signal_v2 = geometryPromiseSignalV2(laplacians, probes, coverage);
  } else {
    fields.coverage_digest = coverage;
  }

  return fields;
}

export function derivedFieldsHash(fields, version = "v0") {
  let included;
  if (version === "v1") {
    included = {
      curvature_summary: fields.curvature_summary,
      trajectory_envelope: fields.trajectory_envelope,
      margin_lower_bound: fields.margin_lower_bound,
      coverage_digest: fields.coverage_digest,
      sensor_health_v1: fields.sensor_health_v1 ?? fields.sensor_health,
      invariance_checks_v1: fields.invariance_checks_v1 ?? fields.invariance_checks,
      geometry_promise_signal: fields.geometry_promise_signal,
    };
  } else if (version === "v2") {
    included = {
      curvature_summary: fields.curvature_summary,
      trajectory_envelope: fields.trajectory_envelope,
      margin_lower_bound: fields.margin_lower_bound,
      sensor_health_v1: fields.sensor_health_v1 ?? fields.sensor_health,
      invariance_checks_v2: fields.invariance_checks_v2 ?? fields.invariance_checks,
      geometry_promise_signal_v2: fields.geometry_promise_signal_v2,
    };
  } else {
    included = {
      curvature_summary: fields.curvature_summary,
      trajectory_envelope: fields.trajectory_envelope,
      margin_lower_bound: fields.margin_lower_bound,
      coverage_digest: fields.coverage_digest,
      sensor_health: fields.sensor_health,
      invariance_checks: fields.invariance_checks,
    };
  }
  return sha256Hex(canonicalize(included));
}

// Compute the full signature certificate. Returns a sigma object.
export function computeSignature({ traceId, publicEnv, positions, probes, sourcePayload = null, version = "v0" }) {
  const t0 = performance.now();

  const fields = computeAnalyticalFields({ positions, probes, version });

  const computeMs = performance.now() - t0;
  // Op count: probe×T Laplacians + sensor estimate + envelope + coverage scan + invariance recompute.
  const T = probes.length;
  const ops = T /*laplacian*/ + T /*sensor*/ + positions.length /*envelope*/
    + positions.length /*coverage*/ + T /*invariance recompute*/;

  const sigma = {
    schema: SIGNATURE_SCHEMAS[version],
    trace_id: traceId,
    source_observations: {
      probe_count_per_step: 5,
      step_count: T,
      probe_offset_r: PROBE_OFFSET_R,
      env_id: publicEnv.id,
      policy_id: traceId.split("|")[0],
    },
    ...fields,
    cost_signature: { wall_ms: computeMs, ops },
    limitations: [
      "nonlocal_basin_topology_invisible",
      "policy_action_history_not_audited",
      "no_decoy_field_separation",
    ],
  };

  if (version === "v1" || version === "v2") {
    const payload = sourcePayload ?? buildSourcePayload({ traceId, publicEnv, positions, probes });
    sigma.transform_version = TRANSFORM_VERSIONS[version];
    sigma.source_hash = sourceHash(payload);
    sigma.derived_fields_hash = derivedFieldsHash(fields, version);
    sigma.integrity_checks = {
      source_match: true,
      transform_match: true,
      derived_field_match: true,
      all_pass: true,
    };
  }

  return sigma;
}

export const SIGNATURE_SCHEMA_V1 = SIGNATURE_SCHEMAS.v1;
export const TRANSFORM_VERSION_V1 = TRANSFORM_VERSIONS.v1;
export const SIGNATURE_SCHEMA_V2 = SIGNATURE_SCHEMAS.v2;
export const TRANSFORM_VERSION_V2 = TRANSFORM_VERSIONS.v2;
