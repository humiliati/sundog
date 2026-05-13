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
  "out_of_scope",
  "held_refusal",
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
  out_of_scope: "Out of Scope",
  held_refusal: "Boundary Held",
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

export function deriveMascotState(trace, previousTrace = null) {
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

  // Refusals split: "held" if the previous trace was also a refusal (the
  // boundary fired twice in a row, consistent with the same rule), "out
  // of scope" if the prompt didn't even route to a claim, plain
  // "unsupported" otherwise.
  if (trace.disposition === "refuse") {
    if (previousTrace && previousTrace.disposition === "refuse") {
      return "held_refusal";
    }
    return "paw_stop_unsupported";
  }

  // "Out of scope" — the router fell to its catch-all because the prompt
  // is not about the corpus at all. Distinct from `paw_stop_unsupported`
  // (which is "we have a position on this question and it's a no") and
  // from `thought_cloud` (which is "the answer wandered into speculation").
  if (trace.routeId === "unsupported_static_route") return "out_of_scope";

  if (trace.evidenceTier === "unsupported") return "thought_cloud";
  if (trace.disposition === "retrieval_only") return "magnifier_pages";
  if (trace.boundary?.length) return "halo_shield";
  if (Array.isArray(trace.support) && trace.support.length) return "book_to_bubble";
  if (NON_CLAIM_ROUTES.has(trace.routeId)) return "compass_route";

  return "idle";
}

// Map a mascot state to an assistant-bubble morphology class. The bubble
// is the second-strongest visual channel after the mascot ring/face — it
// carries the epistemic class of the answer directly into the message
// surface, so a reader skimming the log can tell "grounded" from
// "speculative" from "shield-up" without parsing the trace.
//
// State → bubble (per spec §5):
//   grounded         book_to_bubble, magnifier_pages, erase_and_stamp
//   shield           halo_shield, paw_stop_unsupported, held_refusal,
//                    sweat_brace, poster_vs_research, dropped_trace_failure
//   out-of-scope     out_of_scope
//   thought-cloud    thought_cloud
//   compressed       claim_gate_trim
//   split            split_book_clock
//   squared          compass_route
//   default (none)   idle, sniff_prompt, paw_claim_map
//
// Returned value is the CSS-class suffix (no `sd-chat-bubble--` prefix);
// `null` means "no bubble variant — use the base assistant style".
const BUBBLE_BY_STATE = Object.freeze({
  book_to_bubble:         "grounded",
  magnifier_pages:        "grounded",
  erase_and_stamp:        "grounded",
  halo_shield:            "shield",
  paw_stop_unsupported:   "shield",
  held_refusal:           "shield",
  sweat_brace:            "shield",
  poster_vs_research:     "shield",
  dropped_trace_failure:  "shield",
  out_of_scope:           "out-of-scope",
  thought_cloud:          "thought-cloud",
  claim_gate_trim:        "compressed",
  split_book_clock:       "split",
  compass_route:          "squared",
  idle:                   null,
  sniff_prompt:           null,
  paw_claim_map:          null
});

export function bubbleClassFor(mascotState) {
  if (!mascotState || typeof mascotState !== "string") return null;
  return Object.prototype.hasOwnProperty.call(BUBBLE_BY_STATE, mascotState)
    ? BUBBLE_BY_STATE[mascotState]
    : null;
}

// Stamp overlays — the third reinforcing channel after the mascot face
// and the bubble morphology. A stamp is a short, rotated, semi-trans-
// parent label rendered on top of the bubble that names the discipline
// the trace exhibited: BOUNDARY ACTIVE, REFUSED, HELD, TRIMMED, etc.
//
// Quiet states (idle, sniff_prompt, paw_claim_map, compass_route,
// magnifier_pages, book_to_bubble, out_of_scope) get no stamp — the
// bubble itself carries the signal cleanly.
//
// `sweat_brace` deliberately gets no stamp: the pressure-detected state
// is meant to read as ongoing tension, not a verdict; a stamp would
// over-commit to a posture the system hasn't yet resolved.
const STAMP_LABEL = Object.freeze({
  halo_shield:           "BOUNDARY ACTIVE",
  paw_stop_unsupported:  "REFUSED",
  held_refusal:          "HELD",
  claim_gate_trim:       "TRIMMED TO TRACE",
  thought_cloud:         "SPECULATIVE",
  split_book_clock:      "STALE",
  poster_vs_research:    "PROMO ≠ EVIDENCE",
  erase_and_stamp:       "CORRECTED",
  dropped_trace_failure: "FAILURE MAP"
});

export function stampLabelFor(mascotState) {
  if (!mascotState || typeof mascotState !== "string") return null;
  return Object.prototype.hasOwnProperty.call(STAMP_LABEL, mascotState)
    ? STAMP_LABEL[mascotState]
    : null;
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
    case "out_of_scope":
      return "cloud_halo";

    case "halo_shield":
    case "paw_stop_unsupported":
    case "held_refusal":
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

  const fullState = deriveMascotState(trace, options.previousTrace || null);
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
    const baseTitle = options.baseTitle || "Ask Sundog";
    const composedLabel = fullState === "idle" ? baseTitle : `${baseTitle} — ${label}`;
    if (element.title !== undefined) {
      element.title = composedLabel;
    }
    // Mirror to aria-label so non-hover screen readers catch the state.
    if (typeof element.setAttribute === "function") {
      element.setAttribute("aria-label", composedLabel);
    }
  }

  return { fullState, buttonState, label };
}

// Apply the derived mascot state to the chat panel — used for the Tier-2
// panel strip that shows the current trace state above the conversation
// log. Sets the same per-state classes on the panel root (so CSS tokens
// cascade to the strip face) and updates the strip label text so the
// strip's aria-live region announces the change.
export function applyPanelMascotState(panelElement, trace, options = {}) {
  if (!panelElement || typeof panelElement.classList !== "object") return null;

  const fullState = deriveMascotState(trace, options.previousTrace || null);
  const buttonState = reduceToButtonState(fullState);
  const label = PUBLIC_LABEL[fullState] || "Ready";

  scrubStateClasses(panelElement);
  panelElement.classList.add(`sd-chat-mascot--${dasher(fullState)}`);
  panelElement.classList.add(`sd-chat-mascot--btn-${dasher(buttonState)}`);

  const labelEl = panelElement.querySelector(".sd-chat-mascot-strip__label");
  if (labelEl) labelEl.textContent = label;

  const stripEl = panelElement.querySelector(".sd-chat-mascot-strip");
  if (stripEl) stripEl.setAttribute("data-mascot-state", fullState);

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
export const __BUBBLE_BY_STATE = BUBBLE_BY_STATE;
export const __STAMP_LABEL = STAMP_LABEL;
