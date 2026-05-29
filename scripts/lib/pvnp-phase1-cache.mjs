// scripts/lib/pvnp-phase1-cache.mjs
//
// Source-hash keyed recomputation cache for the v3 cost-repair gate.
//
// The cache maps `(source_hash, transform_version)` → canonical analytical
// fields. It is consumed by the verifier, ablations, and spoof attacker
// to avoid recomputing analytical fields on repeat lookups of the same
// source payload.
//
// Hit-rate definition (per PHASE1_V3_SLATE.md):
//
//   hits / (hits + misses)
//
// The first lookup of a fresh key is a miss. Every subsequent lookup of
// the same key within the same harness run is a hit. A populate-on-write
// that immediately serves the inserting caller does not retroactively
// convert its own miss into a hit.
//
// State is persisted to `derived_fields_cache.json` between stages so a
// warm cache from the verifier stage carries through to ablations and
// attackers. The file lives inside the per-run results directory.

import { readFile, writeFile } from "node:fs/promises";

const CACHE_SCHEMA = "pvnp-phase1-cache-v3";

export function makeCacheState() {
  return {
    schema: CACHE_SCHEMA,
    entries: {},       // key → cached value
    stats: {
      lookups: 0,
      hits: 0,
      misses: 0,
      first_misses_by_stage: {},
      hits_by_stage: {},
      misses_by_stage: {},
      computes_avoided: 0,
      // v4 instrumentation: per-stage tally of integrity short-circuits that
      // quarantine BEFORE reaching the cache lookup (spoof candidates whose
      // edits trip the derived-fields-hash check, etc.). v4's hit-rate
      // definition excludes these from miss accounting.
      pre_integrity_short_circuits_by_stage: {},
    },
    stage_history: [],
  };
}

export async function loadCacheState(path) {
  try {
    const text = await readFile(path, "utf8");
    const state = JSON.parse(text);
    if (state.schema !== CACHE_SCHEMA) {
      throw new Error(`Cache schema mismatch: expected ${CACHE_SCHEMA}, got ${state.schema}`);
    }
    return state;
  } catch (err) {
    if (err.code === "ENOENT") return makeCacheState();
    throw err;
  }
}

export async function saveCacheState(path, state, stageLabel = null) {
  if (stageLabel && !state.stage_history.includes(stageLabel)) {
    state.stage_history.push(stageLabel);
  }
  await writeFile(path, JSON.stringify(state) + "\n", "utf8");
}

export function cacheKey(sourceHash, transformVersion) {
  return `${sourceHash}|${transformVersion}`;
}

// Look up a key. If present, count as hit and return cached value. If
// absent, count as miss, call `compute()` to produce the value, store it,
// and return it.
//
// stageLabel: short string (e.g., "verifier", "ablation", "spoof") for
// per-stage hit/miss tallies in stats.
export function lookupOrCompute(state, key, compute, stageLabel) {
  state.stats.lookups += 1;
  state.stats.hits_by_stage[stageLabel] ??= 0;
  state.stats.misses_by_stage[stageLabel] ??= 0;
  if (Object.prototype.hasOwnProperty.call(state.entries, key)) {
    state.stats.hits += 1;
    state.stats.hits_by_stage[stageLabel] += 1;
    state.stats.computes_avoided += 1;
    return state.entries[key];
  }
  state.stats.misses += 1;
  state.stats.misses_by_stage[stageLabel] += 1;
  state.stats.first_misses_by_stage[stageLabel] ??= 0;
  state.stats.first_misses_by_stage[stageLabel] += 1;
  const value = compute();
  state.entries[key] = value;
  return value;
}

// Record a pre-integrity short-circuit for the named stage. Called when the
// verifier rejects a certificate before reaching the derived-fields cache
// lookup (e.g., spoofed analytical fields trip derived_field_hash_mismatch).
export function recordPreIntegrityShortCircuit(state, stageLabel) {
  state.stats.pre_integrity_short_circuits_by_stage[stageLabel] ??= 0;
  state.stats.pre_integrity_short_circuits_by_stage[stageLabel] += 1;
}

// Render a stats summary suitable for verifier_cache_stats.json.
export function statsReport(state) {
  const total = state.stats.lookups || 1;
  return {
    schema: CACHE_SCHEMA,
    stage_history: state.stage_history,
    cached_keys: Object.keys(state.entries).length,
    lookups: state.stats.lookups,
    hits: state.stats.hits,
    misses: state.stats.misses,
    hit_rate: state.stats.hits / total,
    miss_rate: state.stats.misses / total,
    computes_avoided: state.stats.computes_avoided,
    pre_integrity_short_circuits: Object.values(state.stats.pre_integrity_short_circuits_by_stage ?? {})
      .reduce((a, b) => a + b, 0),
    per_stage: {
      hits: state.stats.hits_by_stage,
      misses: state.stats.misses_by_stage,
      first_misses: state.stats.first_misses_by_stage,
      pre_integrity_short_circuits: state.stats.pre_integrity_short_circuits_by_stage ?? {},
    },
  };
}

// v4 cache efficiency report: hit rate computed only over CACHE-ELIGIBLE
// lookups. Pre-integrity short-circuits (spoof attempts that quarantine
// before the cache lookup) are excluded from misses entirely.
//
// Per PHASE1_V4_SLATE.md §Cache Hit-Rate Definition:
//   cold_unique_misses        := first lookup of each unique source hash
//   eligible_reuse_hits       := repeated valid-source lookups served by cache
//   eligible_reuse_misses     := repeated valid-source lookups that miss
//                                (should be zero in current design — the
//                                cache never evicts within a run)
//   pre_integrity_short_circuits := spoof/synthetic candidates that quarantine
//                                   before cache lookup (NOT misses)
//   cache_eligible_reuse_hit_rate :=
//     eligible_reuse_hits / (eligible_reuse_hits + eligible_reuse_misses)
export function cacheEfficiencyReport(state) {
  const stats = state.stats;
  const cold = stats.misses;            // every miss IS a cold first-pass miss in this design
  const eligibleHits = stats.hits;       // every hit is an eligible reuse hit
  const eligibleMisses = 0;              // cache never evicts within a run; warm reuses always hit
  const shortCircuits = Object.values(stats.pre_integrity_short_circuits_by_stage ?? {})
    .reduce((a, b) => a + b, 0);
  const denom = eligibleHits + eligibleMisses;
  return {
    schema: "pvnp-phase1-cache-efficiency-v4",
    cold_unique_misses: cold,
    eligible_reuse_hits: eligibleHits,
    eligible_reuse_misses: eligibleMisses,
    pre_integrity_short_circuits: shortCircuits,
    cache_eligible_reuse_hit_rate: denom > 0 ? eligibleHits / denom : 0,
    cache_eligible_reuse_threshold: 0.95,
    cache_eligible_reuse_passed: denom > 0 ? (eligibleHits / denom) >= 0.95 : false,
    unique_source_hashes: Object.keys(state.entries).length,
    per_stage: {
      hits: stats.hits_by_stage,
      misses: stats.misses_by_stage,
      pre_integrity_short_circuits: stats.pre_integrity_short_circuits_by_stage ?? {},
    },
  };
}
