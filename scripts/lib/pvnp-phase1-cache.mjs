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
    per_stage: {
      hits: state.stats.hits_by_stage,
      misses: state.stats.misses_by_stage,
      first_misses: state.stats.first_misses_by_stage,
    },
  };
}
