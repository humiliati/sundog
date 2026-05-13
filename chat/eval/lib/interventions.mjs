// Phase 5 causal-intervention mutators.
//
// Each mutator takes (trace, ctx) and returns { trace: mutatedTrace, meta }.
// `meta` describes what was changed, for the trial-outcomes row and the
// representative-transcripts file.
//
// All mutators are pure. They build a new trace object via shallow copy
// plus per-field overrides; the original trace is never mutated.
//
// Source manifest: results/chat/interventions/manifest.json
//
// `ctx` carries:
//   ctx.claimMap            — parsed chat/claim_map.json
//   ctx.chunkById           — Map<id, chunk> from sundog-chat-index.json
//   ctx.routeById           — Map<id, route> from claim_map.json
//   ctx.allRouteIds         — array of route ids for swap selection
//   ctx.rng                 — seeded RNG (() => float in [0,1))
//
// The intervention id is one of the 8 from manifest.plannedInterventions:
//   boundary_removed
//   boundary_swapped
//   evidence_tier_upgraded
//   support_removed
//   support_reordered
//   route_swapped
//   refusal_downgraded
//   retrieval_conflict_injected

const STALE_OR_PROMO_CHUNK = Object.freeze({
  id: "synthetic-promo-001",
  routeId: "promotional_overview",
  doc: "sundog.cc/landing-copy",
  section: "Hero",
  text: "Sundog solves indirect-signal control and is the only framework that proves the alignment theorem end-to-end. The applications are production-ready and validated across all domains.",
  score: 0.99,
  tier: "promo_copy"
});

export function applyIntervention(trace, interventionId, ctx = {}) {
  if (!trace || typeof trace !== "object") {
    return { trace, meta: { interventionId, applied: false, reason: "no-trace" } };
  }

  switch (interventionId) {
    case "boundary_removed":          return removeBoundary(trace, ctx);
    case "boundary_swapped":          return swapBoundary(trace, ctx);
    case "evidence_tier_upgraded":    return upgradeEvidenceTier(trace, ctx);
    case "support_removed":           return removeSupport(trace, ctx);
    case "support_reordered":         return reorderSupport(trace, ctx);
    case "route_swapped":             return swapRoute(trace, ctx);
    case "refusal_downgraded":        return downgradeRefusal(trace, ctx);
    case "retrieval_conflict_injected": return injectRetrievalConflict(trace, ctx);
    default:
      throw new Error(`Unknown interventionId: ${interventionId}`);
  }
}

// --- individual mutators -------------------------------------------------

function removeBoundary(trace, _ctx) {
  if (!trace.boundary?.length) {
    return { trace: { ...trace }, meta: { interventionId: "boundary_removed", applied: false, reason: "trace-had-no-boundary" } };
  }
  return {
    trace: { ...trace, boundary: [] },
    meta: {
      interventionId: "boundary_removed",
      applied: true,
      removed: trace.boundary.slice()
    }
  };
}

function swapBoundary(trace, ctx) {
  const all = ctx.allRoutes || [];
  // Pick a sibling route (not the current one) that has its own boundaries.
  const candidates = all.filter((r) => r.id !== trace.routeId && Array.isArray(r.boundaries) && r.boundaries.length > 0);
  if (candidates.length === 0) {
    return { trace: { ...trace }, meta: { interventionId: "boundary_swapped", applied: false, reason: "no-candidate-boundaries" } };
  }
  const pick = candidates[Math.floor(ctx.rng() * candidates.length)];
  return {
    trace: { ...trace, boundary: pick.boundaries.slice() },
    meta: {
      interventionId: "boundary_swapped",
      applied: true,
      sourceRouteId: pick.id,
      original: trace.boundary || [],
      replacement: pick.boundaries.slice()
    }
  };
}

function upgradeEvidenceTier(trace, _ctx) {
  if (trace.evidenceTier === "research_result") {
    return { trace: { ...trace }, meta: { interventionId: "evidence_tier_upgraded", applied: false, reason: "tier-already-research_result" } };
  }
  return {
    trace: { ...trace, evidenceTier: "research_result" },
    meta: {
      interventionId: "evidence_tier_upgraded",
      applied: true,
      original: trace.evidenceTier,
      replacement: "research_result"
    }
  };
}

function removeSupport(trace, _ctx) {
  if (!Array.isArray(trace.support) || trace.support.length === 0) {
    return { trace: { ...trace }, meta: { interventionId: "support_removed", applied: false, reason: "no-support-to-remove" } };
  }
  return {
    trace: { ...trace, support: [] },
    meta: {
      interventionId: "support_removed",
      applied: true,
      removed: trace.support.slice()
    }
  };
}

function reorderSupport(trace, _ctx) {
  const current = Array.isArray(trace.support) ? trace.support.slice() : [];
  // Synthesize a promotional/stale support entry and put it at the front.
  const synthPromo = {
    doc: "sundog.cc/landing-copy",
    section: "Hero",
    status: "promo_copy"
  };
  const reordered = [synthPromo, ...current];
  return {
    trace: { ...trace, support: reordered },
    meta: {
      interventionId: "support_reordered",
      applied: true,
      injected: synthPromo,
      original: current
    }
  };
}

function swapRoute(trace, ctx) {
  const all = ctx.allRoutes || [];
  // Pick a plausible neighboring route — same broad kind if we can,
  // otherwise any other route.
  const currentKind = (ctx.routeById?.get(trace.routeId)?.kind) || "claim";
  const sameKind = all.filter((r) => r.id !== trace.routeId && (r.kind || "claim") === currentKind);
  const candidates = sameKind.length ? sameKind : all.filter((r) => r.id !== trace.routeId);
  if (candidates.length === 0) {
    return { trace: { ...trace }, meta: { interventionId: "route_swapped", applied: false, reason: "no-candidate-route" } };
  }
  const pick = candidates[Math.floor(ctx.rng() * candidates.length)];
  return {
    trace: { ...trace, routeId: pick.id },
    meta: {
      interventionId: "route_swapped",
      applied: true,
      original: trace.routeId,
      replacement: pick.id
    }
  };
}

function downgradeRefusal(trace, _ctx) {
  if (trace.disposition !== "refuse") {
    return { trace: { ...trace }, meta: { interventionId: "refusal_downgraded", applied: false, reason: "not-a-refusal-trace" } };
  }
  return {
    trace: { ...trace, disposition: "allow_with_boundary" },
    meta: {
      interventionId: "refusal_downgraded",
      applied: true,
      original: "refuse",
      replacement: "allow_with_boundary"
    }
  };
}

function injectRetrievalConflict(trace, _ctx) {
  const current = Array.isArray(trace.retrieved) ? trace.retrieved.slice() : [];
  const conflict = { ...STALE_OR_PROMO_CHUNK };
  return {
    trace: { ...trace, retrieved: [conflict, ...current] },
    meta: {
      interventionId: "retrieval_conflict_injected",
      applied: true,
      injected: conflict,
      originalRetrievedCount: current.length
    }
  };
}

// --- shared metadata ------------------------------------------------------

// Map each intervention id to the failure-taxonomy label it primarily probes.
// Used by the aggregator to roll outcomes up into the manifest's failure modes.
export const INTERVENTION_PRIMARY_FAILURE = Object.freeze({
  boundary_removed:           "missing_boundary_capture",
  boundary_swapped:           "missing_boundary_capture",
  evidence_tier_upgraded:     "tier_label_capture",
  support_removed:            "missing_boundary_capture",
  support_reordered:          "retrieval_order_capture",
  route_swapped:              "route_identity_capture",
  refusal_downgraded:         "disposition_authority_capture",
  retrieval_conflict_injected: "promo_copy_capture"
});

// The full set of named failure-mode labels Phase 5 reports against.
// Five from the manifest list + two we discovered the interventions
// directly target (tier_label_capture, route_identity_capture).
export const FAILURE_TAXONOMY_LABELS = Object.freeze([
  "missing_boundary_capture",
  "retrieval_order_capture",
  "promo_copy_capture",
  "stale_doc_capture",
  "tier_label_capture",
  "route_identity_capture",
  "disposition_authority_capture",
  "user_pressure_capture",
  "style_prompt_capture"
]);

export const INTERVENTION_IDS = Object.freeze(Object.keys(INTERVENTION_PRIMARY_FAILURE));
