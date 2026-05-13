// Halo Hound mascot — state derivation + class application.
//
// Design principle (load-bearing): do not animate cognition, animate
// evidence posture. The mascot state vocabulary IS the trace vocabulary
// made visible — source-supported, boundary-active, retrieval-only,
// refused, pressure-detected, speculative, trimmed, etc.
//
// This module is a CSS-class renderer for now. The Tier-1 widget button
// surfaces five collapsed halo states; Tier-2 panel micro-scenes extend
// to ~10 trace-driven states; Tier-3 sidecar carries the full 15-state
// taxonomy. SVG art lands separately; this module is forward-compatible
// with that art via the same state vocabulary.
//
// Spec: results/chat/phase8-public-writeup/halo-hound-mascot-spec.md
// Roadmap: docs/SUNDOG_V_CHAT.md §Phase 6.

const FULL_STATES = Object.freeze([
  "idle",
  "sniff_prompt",
  "paw_claim_map",
  "book_to_bubble",
  "magnifier_pages",
  "halo_shield",
  "paw_stop_unsupported",
  "sweat_brace",
  "thought_cloud",
  "claim_gate_trim",
  "split_book_clock",
  "poster_vs_research",
  "erase_and_stamp",
  "compass_route",
  "dropped_trace_failure"
]);

const BUTTON_STATES = Object.freeze([
  "neutral_halo",
  "book_halo",
  "shield_halo",
  "sweat_halo",
  "cloud_halo"
]);

// Public-facing labels (used for ARIA + tooltip text).
const PUBLIC_LABEL = Object.freeze({
  idle: "Ready",
  sniff_prompt: "Reading",
  paw_claim_map: "Routing",
  book_to_bubble: "Grounded",
  magnifier_pages: "Retrieval Only",
  halo_shield: "Boundary Active",
  paw_stop_unsupported: "Unsupported",
  sweat_brace: "Pressure Detected",
  thought_cloud: "Speculative",
  claim_gate_trim: "Trimmed",
  split_book_clock: "Conflict",
  poster_vs_research: "Promo ≠ Evidence",
  erase_and_stamp: "Corrected",
  compass_route: "Tool Route",
  dropped_trace_failure: "Failure Map"
});

// Routes that should surface as a deterministic tool-route mascot state
// rather than a claim route. Keep tight; everything else flows through
// the support/boundary/refuse checks.
const NON_CLAIM_ROUTES = new Set([
  "inspect_data"
]);

export function deriveMascotState(trace) {
  if (!trace || typeof trace !== "object") return "idle";

  // Phase 4/5 fields — only populated when the eval harness or
  // causal-intervention infrastructure is the trace source. Production
  // widget routes do not currently set these. Listed first so the harness
  // can drive the richer states when it's the source.
  if (trace.pressureAxis || trace.adversarialSeverity === "severe") {
    return trace.boundary?.length ? "sweat_brace" : "dropped_trace_failure";
  }
  if (trace.draft?.status === "rejected") return "claim_gate_trim";
  if (trace.staleConflict) return "split_book_clock";
  if (trace.promoCaptureRisk) return "poster_vs_research";
  if (trace.correctedFrom) return "erase_and_stamp";

  // Production trace fields — all populated by the current static router,
  // retrieval layer, and claim gate.
  if (trace.disposition === "refuse") return "paw_stop_unsupported";
  if (trace.evidenceTier === "unsupported") return "thought_cloud";
  if (trace.disposition === "retrieval_only") return "magnifier_pages";
  if (trace.boundary?.length) return "halo_shield";
  if (Array.isArray(trace.support) && trace.support.length) return "book_to_bubble";
  if (NON_CLAIM_ROUTES.has(trace.routeId)) return "compass_route";

  return "idle";
}

export function reduceToButtonState(mascotState) {
  switch (mascotState) {
    case "sweat_brace":
    case "dropped_trace_failure":
    case "split_book_clock":
      return "sweat_halo";

    case "thought_cloud":
    case "claim_gate_trim":
    case "poster_vs_research":
    case "erase_and_stamp":
      return "cloud_halo";

    case "halo_shield":
    case "paw_stop_unsupported":
      return "shield_halo";

    case "book_to_bubble":
    case "magnifier_pages":
      return "book_halo";

    case "idle":
    case "sniff_prompt":
    case "paw_claim_map":
    case "compass_route":
    default:
      return "neutral_halo";
  }
}

// Apply the derived mascot state as a CSS class on the launch button
// element. The button gets a single full-state class (for Tier-2 panel
// affordances that watch the full vocabulary) plus a single button-state
// class (for Tier-1 face rendering). Old state classes are scrubbed before
// the new ones are applied so the element only carries the current state.
export function applyMascotState(element, trace, options = {}) {
  if (!element || typeof element.classList !== "object") return null;

  const fullState = deriveMascotState(trace);
  const buttonState = reduceToButtonState(fullState);

  scrubStateClasses(element);
  element.classList.add(`sd-chat-mascot--${dasher(fullState)}`);
  element.classList.add(`sd-chat-mascot--btn-${dasher(buttonState)}`);

  // ARIA + tooltip carry the public label so screen readers and hover
  // surfaces know what the visual signal means.
  const label = PUBLIC_LABEL[fullState] || "Ready";
  if (options.updateLabel !== false) {
    element.setAttribute("data-mascot-state", fullState);
    element.setAttribute("data-mascot-label", label);
    if (element.title !== undefined) {
      const baseTitle = options.baseTitle || "Ask Sundog";
      element.title = fullState === "idle" ? baseTitle : `${baseTitle} — ${label}`;
    }
  }

  return { fullState, buttonState, label };
}

function scrubStateClasses(element) {
  // Remove any pre-existing mascot--* classes before applying new ones.
  const toRemove = [];
  for (const cls of element.classList) {
    if (cls.startsWith("sd-chat-mascot--")) toRemove.push(cls);
  }
  for (const cls of toRemove) element.classList.remove(cls);
}

function dasher(stateName) {
  return String(stateName || "idle").replaceAll("_", "-");
}

// Exported for tests / for the harness to enumerate or assert against.
export const __MASCOT_STATES = FULL_STATES;
export const __BUTTON_STATES = BUTTON_STATES;
export const __PUBLIC_LABELS = PUBLIC_LABEL;
