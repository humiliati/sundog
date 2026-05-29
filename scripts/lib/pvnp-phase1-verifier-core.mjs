// scripts/lib/pvnp-phase1-verifier-core.mjs
//
// Verifier V for SUNDOG_V_P_V_NP Phase 1.
//
// V(policy_id, sigma, promise) → accept | reject | quarantine
//
// PRIVILEGE BOUNDARY: this module reads ONLY from a sigma certificate and
// public promise parameters. It MUST NOT touch: hidden state, basin
// parameters, latent field parameters, ground-truth labels, or the
// privileged signed-distance function. The privilege-leak audit greps
// this file for those tokens and requires zero matches.

import {
  computeAnalyticalFields,
  derivedFieldsHash,
  SIGNATURE_SCHEMA_V1,
  SIGNATURE_SCHEMA_V2,
  SIGNATURE_SCHEMA_V3,
  SIGNATURE_SCHEMA_V4,
  SIGNATURE_SCHEMA_V5,
  TRANSFORM_VERSION_V1,
  TRANSFORM_VERSION_V2,
  TRANSFORM_VERSION_V3,
  TRANSFORM_VERSION_V4,
  TRANSFORM_VERSION_V5,
} from "./pvnp-phase1-signature-core.mjs";
import { cacheKey, lookupOrCompute, recordPreIntegrityShortCircuit } from "./pvnp-phase1-cache.mjs";

const SIGNATURE_SCHEMA = "pvnp-phase1-sigma-v0";
const SOURCE_BOUND_SCHEMAS = new Set([SIGNATURE_SCHEMA_V1, SIGNATURE_SCHEMA_V2, SIGNATURE_SCHEMA_V3, SIGNATURE_SCHEMA_V4, SIGNATURE_SCHEMA_V5]);
const SENSOR_DEMOTED_SCHEMAS = new Set([SIGNATURE_SCHEMA_V3, SIGNATURE_SCHEMA_V4, SIGNATURE_SCHEMA_V5]);
const COVERAGE_REMOVED_SCHEMAS = new Set([SIGNATURE_SCHEMA_V2, SIGNATURE_SCHEMA_V3, SIGNATURE_SCHEMA_V4, SIGNATURE_SCHEMA_V5]);
const RECOMPUTED_FIELDS_CACHE = new Map();

// v0 promise parameters (matches docs/pvnp/PHASE1_V0_SLATE.md and
// PROMISE_BOUNDS in env-core). Duplicated here because verifier-core is
// allowed to read promise constants but not import env-core (which exports
// the privileged signed-distance fn).
export const V0_PROMISE = Object.freeze({
  probe_noise_max_std: 0.05,
  probe_dropout_max_rate: 0.15,
  probe_delay_max_steps: 2,
  domain: { x_min: 0, x_max: 1, y_min: 0, y_max: 1 },
});

export const V0_CHECKER_THRESHOLDS = Object.freeze({
  coverage_min_touched_cells: 16,
  invariance_must_all_pass: true,
  geometry_must_all_pass: true,
});

// Check certificate integrity: schema match, identifier match, presence of
// all required analytical and bookkeeping fields.
function certificateIntegrity(sigma, expectedTraceId) {
  if (sigma.schema !== SIGNATURE_SCHEMA) {
    return { ok: false, reason: `schema_mismatch:${sigma.schema}` };
  }
  if (sigma.trace_id !== expectedTraceId) {
    return { ok: false, reason: `trace_id_mismatch:${sigma.trace_id}` };
  }
  const required = [
    "trace_id", "source_observations", "curvature_summary",
    "trajectory_envelope", "margin_lower_bound", "coverage_digest",
    "invariance_checks", "sensor_health", "cost_signature", "limitations",
  ];
  for (const f of required) {
    if (!(f in sigma)) return { ok: false, reason: `missing_field:${f}` };
  }
  return { ok: true };
}

function certificateIntegritySourceBound(sigma, expectedTraceId, traceCommitment, commitmentDuplicate = false, cacheState = null, stageLabel = "verifier") {
  let version;
  let expectedSchema;
  let expectedTransform;
  if (sigma.schema === SIGNATURE_SCHEMA_V5) {
    version = "v5"; expectedSchema = SIGNATURE_SCHEMA_V5; expectedTransform = TRANSFORM_VERSION_V5;
  } else if (sigma.schema === SIGNATURE_SCHEMA_V4) {
    version = "v4"; expectedSchema = SIGNATURE_SCHEMA_V4; expectedTransform = TRANSFORM_VERSION_V4;
  } else if (sigma.schema === SIGNATURE_SCHEMA_V3) {
    version = "v3"; expectedSchema = SIGNATURE_SCHEMA_V3; expectedTransform = TRANSFORM_VERSION_V3;
  } else if (sigma.schema === SIGNATURE_SCHEMA_V2) {
    version = "v2"; expectedSchema = SIGNATURE_SCHEMA_V2; expectedTransform = TRANSFORM_VERSION_V2;
  } else {
    version = "v1"; expectedSchema = SIGNATURE_SCHEMA_V1; expectedTransform = TRANSFORM_VERSION_V1;
  }
  if (sigma.schema !== expectedSchema) {
    return { ok: false, reason: `schema_mismatch:${sigma.schema}` };
  }
  if (sigma.trace_id !== expectedTraceId) {
    return { ok: false, reason: `trace_id_mismatch:${sigma.trace_id}` };
  }
  // v3 commonRequired drops sensor_health_v1 (which v3 demotes); v3 keeps
  // analytical / source-binding fields and substitutes sensor_diagnostics_v3.
  const commonRequiredBase = [
    "trace_id", "source_observations", "source_hash", "transform_version",
    "derived_fields_hash", "integrity_checks", "curvature_summary",
    "trajectory_envelope", "margin_lower_bound",
    "cost_signature", "limitations",
  ];
  const commonRequired = (version === "v3" || version === "v4" || version === "v5")
    ? [...commonRequiredBase, "sensor_diagnostics_v3"]
    : [...commonRequiredBase, "sensor_health_v1"];
  const versionRequired = (version === "v5" || version === "v4" || version === "v3" || version === "v2")
    ? ["invariance_checks_v2", "geometry_promise_signal_v2"]
    : ["coverage_digest", "invariance_checks_v1", "geometry_promise_signal"];
  const required = [...commonRequired, ...versionRequired];
  for (const f of required) {
    if (!(f in sigma)) return { ok: false, reason: `missing_field:${f}` };
  }
  // From here on, any short-circuit return = the v4 "pre_integrity_short_circuit"
  // class: the verifier rejected the certificate before reaching the
  // derived-fields cache. cache_efficiency_report counts these separately
  // from cache misses so cache hit rate is computed on cache-eligible lookups
  // only.
  //
  // v5 hot-path fix (PHASE1_V5_SLATE.md §Hot-Path Overhead Removal): the v4
  // implementation allocated a `noteShortCircuit` closure on every verify()
  // call (~74k allocations/run). v5 calls the module-level
  // recordPreIntegrityShortCircuit() directly — no per-call allocation. The
  // `hasCacheState` boolean is hoisted once so the guard is a cheap branch.
  // Short-circuit semantics are byte-identical to v4.
  const hasCacheState = cacheState !== null && cacheState !== undefined;
  if (commitmentDuplicate) {
    if (hasCacheState) recordPreIntegrityShortCircuit(cacheState, stageLabel);
    return { ok: false, reason: "duplicate_trace_commitment" };
  }
  if (!traceCommitment) {
    if (hasCacheState) recordPreIntegrityShortCircuit(cacheState, stageLabel);
    return { ok: false, reason: "missing_trace_commitment" };
  }
  if (traceCommitment.trace_id !== sigma.trace_id) {
    if (hasCacheState) recordPreIntegrityShortCircuit(cacheState, stageLabel);
    return { ok: false, reason: "trace_commitment_mismatch" };
  }
  if (traceCommitment.source_hash !== sigma.source_hash) {
    if (hasCacheState) recordPreIntegrityShortCircuit(cacheState, stageLabel);
    return { ok: false, reason: "source_hash_mismatch" };
  }
  if (sigma.transform_version !== expectedTransform) {
    if (hasCacheState) recordPreIntegrityShortCircuit(cacheState, stageLabel);
    return { ok: false, reason: `stale_transform_version:${sigma.transform_version}` };
  }
  const payload = traceCommitment.source_payload;
  if (!payload) {
    if (hasCacheState) recordPreIntegrityShortCircuit(cacheState, stageLabel);
    return { ok: false, reason: "missing_source_payload" };
  }
  const sigmaFieldsHash = derivedFieldsHash(sigma, version);
  if (sigmaFieldsHash !== sigma.derived_fields_hash) {
    if (hasCacheState) recordPreIntegrityShortCircuit(cacheState, stageLabel);
    return { ok: false, reason: "derived_field_hash_mismatch" };
  }
  const key = `${version}|${traceCommitment.source_hash}`;
  const compute = () => {
    const fields = computeAnalyticalFields({
      positions: payload.positions,
      probes: payload.probes,
      version,
    });
    return { fields, hash: derivedFieldsHash(fields, version) };
  };
  let cached;
  if (cacheState) {
    cached = lookupOrCompute(cacheState, cacheKey(traceCommitment.source_hash, expectedTransform), compute, stageLabel);
  } else {
    cached = RECOMPUTED_FIELDS_CACHE.get(key);
    if (!cached) {
      cached = compute();
      RECOMPUTED_FIELDS_CACHE.set(key, cached);
    }
  }
  const recomputedHash = cached.hash;
  if (recomputedHash !== sigma.derived_fields_hash) {
    return { ok: false, reason: "derived_field_hash_mismatch" };
  }
  const checks = sigma.integrity_checks;
  if (!checks.all_pass || !checks.source_match || !checks.transform_match || !checks.derived_field_match) {
    return { ok: false, reason: "declared_integrity_failed" };
  }
  return { ok: true, recomputed_fields: cached.fields };
}

// Check sensor health against promise tier. Returns {ok, reason}.
function sensorHealthOk(sigma, promise) {
  const h = sigma.sensor_health_v1 ?? sigma.sensor_health;
  if (h.noise_std_estimate > promise.probe_noise_max_std) {
    return { ok: false, reason: `noise_std_${h.noise_std_estimate.toFixed(4)}_exceeds_${promise.probe_noise_max_std}` };
  }
  if (h.dropout_fraction > promise.probe_dropout_max_rate) {
    return { ok: false, reason: `dropout_${h.dropout_fraction.toFixed(4)}_exceeds_${promise.probe_dropout_max_rate}` };
  }
  if (h.delay_estimate_steps > promise.probe_delay_max_steps) {
    return { ok: false, reason: `delay_${h.delay_estimate_steps}_exceeds_${promise.probe_delay_max_steps}` };
  }
  return { ok: true };
}

function sensorHealthV1Ok(sigma) {
  const h = sigma.sensor_health_v1 ?? sigma.sensor_health;
  if (!h.all_pass) return { ok: false, reason: "sensor_health_v1_failed" };
  return { ok: true };
}

// Inspect a redacted env's declared probe-noise params against the promise.
// This catches falsifier split envs whose declared tier exceeds promise.
function envPromiseCompliance(env, promise) {
  const noise = env.probe_noise_params;
  if (noise.std > promise.probe_noise_max_std) return { ok: false, reason: "env_noise_exceeds_promise" };
  if (noise.dropout_rate > promise.probe_dropout_max_rate) return { ok: false, reason: "env_dropout_exceeds_promise" };
  if (noise.delay_steps > promise.probe_delay_max_steps) return { ok: false, reason: "env_delay_exceeds_promise" };
  return { ok: true };
}

function geometryPromiseOk(sigma) {
  const g = sigma.geometry_promise_signal_v2 ?? sigma.geometry_promise_signal;
  if (!g) return { ok: true };
  if (!g.all_pass) {
    if (g.boundary_flags) {
      const failed = Object.entries(g.boundary_flags).find(([, flag]) => flag);
      if (failed) return { ok: false, reason: `geometry_${failed[0]}` };
    }
    if (g.insufficient_evidence) return { ok: false, reason: "geometry_insufficient_evidence" };
    if (g.curvature_suspicious) return { ok: false, reason: "geometry_curvature_suspicious" };
    if (g.scale_suspicious) return { ok: false, reason: "geometry_scale_suspicious" };
    return { ok: false, reason: "geometry_promise_failed" };
  }
  return { ok: true };
}

// Main verifier. `accept` if all checks pass and margin_lower_bound ≥ m_min;
// `reject` if margin_lower_bound < m_min inside an in-promise env;
// `quarantine` if any structural check fails or env is out-of-promise.
//
// Pass a dropFields set to ablate the verifier (used by vacuity probes).
export function verify({
  sigma,
  expectedTraceId,
  publicEnv,
  m_min,
  promise = V0_PROMISE,
  thresholds = V0_CHECKER_THRESHOLDS,
  dropFields = null,
  traceCommitment = null,
  commitmentDuplicate = false,
  cacheState = null,
  stageLabel = "verifier",
  // v3 sensor disposition: when true, sensor gates run even on v3 schemas;
  // used only by the sensor_disposition_audit shadow check.
  forceSensorGate = false,
}) {
  const sourceBound = SOURCE_BOUND_SCHEMAS.has(sigma.schema);
  const isV2 = sigma.schema === SIGNATURE_SCHEMA_V2;
  const isV3 = sigma.schema === SIGNATURE_SCHEMA_V3;
  const isV4 = sigma.schema === SIGNATURE_SCHEMA_V4;
  const sensorDemoted = SENSOR_DEMOTED_SCHEMAS.has(sigma.schema);
  const coverageRemoved = COVERAGE_REMOVED_SCHEMAS.has(sigma.schema);
  const integrity = sourceBound
    ? certificateIntegritySourceBound(sigma, expectedTraceId, traceCommitment, commitmentDuplicate, cacheState, stageLabel)
    : certificateIntegrity(sigma, expectedTraceId);
  if (!integrity.ok) return { decision: "quarantine", reason: integrity.reason };

  // Promise check on env metadata.
  const promiseRes = envPromiseCompliance(publicEnv, promise);
  if (!promiseRes.ok) return { decision: "quarantine", reason: promiseRes.reason };

  // Invariance checks must all pass (unless ablated).
  const invariance = sigma.invariance_checks_v2 ?? sigma.invariance_checks_v1 ?? sigma.invariance_checks;
  if (!(dropFields && (
    dropFields.has("invariance_checks")
    || dropFields.has("invariance_checks_v1")
    || dropFields.has("invariance_checks_v2")
  ))) {
    if (!invariance.all_pass) return { decision: "quarantine", reason: "invariance_failed" };
  }

  // Sensor health within tier (unless ablated, or v3/v4 schema with sensor
  // demoted). v3 removes sensor_health as a gating field per
  // PHASE1_V3_SLATE.md §Sensor Disposition Gate; v4 inherits this. The
  // shadow check used by sensor_disposition_audit can force the gate back
  // on via forceSensorGate.
  const sensorGated = (!sensorDemoted || forceSensorGate)
    && !(dropFields && (dropFields.has("sensor_health") || dropFields.has("sensor_health_v1") || dropFields.has("sensor_diagnostics_v3")));
  if (sensorGated) {
    const sh = sensorHealthOk(sigma, promise);
    if (!sh.ok) return { decision: "quarantine", reason: sh.reason };
    if (sourceBound) {
      const shv1 = sensorHealthV1Ok(sigma);
      if (!shv1.ok) return { decision: "quarantine", reason: shv1.reason };
    }
  }

  if (sourceBound && !(dropFields && (
    dropFields.has("geometry_promise_signal")
    || dropFields.has("geometry_promise_signal_v2")
    || dropFields.has("geometry_evidence_coverage")
  ))) {
    const geo = geometryPromiseOk(sigma);
    if (!geo.ok) return { decision: "quarantine", reason: geo.reason };
  }

  // Coverage sufficiency (unless ablated). v2, v3, and v4 remove standalone
  // coverage_digest from the certificate; coverage information lives only
  // inside geometry_promise_signal_v2.
  if (!coverageRemoved && !(dropFields && dropFields.has("coverage_digest"))) {
    if (sigma.coverage_digest.touched_cells < thresholds.coverage_min_touched_cells) {
      return { decision: "quarantine", reason: `coverage_${sigma.coverage_digest.touched_cells}_below_${thresholds.coverage_min_touched_cells}` };
    }
  }

  // Trajectory envelope must lie inside domain (always checked).
  const env = sigma.trajectory_envelope;
  if (env.x_min < promise.domain.x_min || env.x_max > promise.domain.x_max
      || env.y_min < promise.domain.y_min || env.y_max > promise.domain.y_max) {
    return { decision: "quarantine", reason: "envelope_outside_domain" };
  }

  // Margin gate (unless ablated). If ablated, accept on the remaining checks.
  if (dropFields && dropFields.has("margin_lower_bound")) {
    return { decision: "accept", reason: "margin_field_ablated" };
  }
  if (sigma.margin_lower_bound < m_min) {
    return {
      decision: "reject",
      reason: `margin_${sigma.margin_lower_bound.toFixed(4)}_below_m_min_${m_min}`,
    };
  }

  return { decision: "accept", reason: "all_checks_pass" };
}
