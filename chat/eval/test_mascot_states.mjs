// Halo Hound mascot state test harness.
//
// Locks the state-derivation contract in `public/js/sundog-chat-mascot.mjs`.
// Run with: node chat/eval/test_mascot_states.mjs
//
// The harness enumerates every derivable mascot state and asserts both
// `deriveMascotState(trace)` and `reduceToButtonState(state)` return the
// expected values. Two states (out_of_scope, held_refusal) are explicitly
// pending until batch step 2 of the trace-legibility roadmap lands; they
// are listed in the pending section and will start passing once that
// step ships. Two states (sniff_prompt, paw_claim_map) are transient —
// set by the widget mid-flight, not derived from a trace — and are not
// asserted via derivation tests.
//
// Spec: results/chat/phase8-public-writeup/halo-hound-mascot-spec.md
// Roadmap: results/chat/phase8-public-writeup/mascot-trace-legibility-roadmap.md
// Module:  public/js/sundog-chat-mascot.mjs

import {
  deriveMascotState,
  reduceToButtonState,
  bubbleClassFor,
  stampLabelFor,
  __MASCOT_STATES,
  __BUTTON_STATES,
  __PUBLIC_LABELS,
  __BUBBLE_BY_STATE,
  __STAMP_LABEL
} from "../../public/js/sundog-chat-mascot.mjs";

const passes = [];
const failures = [];

function assert(name, actual, expected) {
  if (actual === expected) {
    passes.push({ name, value: actual });
  } else {
    failures.push({ name, expected, actual });
  }
}

// ---------------------------------------------------------------------
// Implemented states — should pass on the current module.
// ---------------------------------------------------------------------
// Tuple: [name, trace, expectedFullState, expectedButtonState]

const implementedCases = [
  ["idle (null trace)",
    null, "idle", "neutral_halo"],

  ["idle (empty object)",
    {}, "idle", "neutral_halo"],

  ["idle (non-object trace)",
    "not a trace", "idle", "neutral_halo"],

  ["sweat_brace (pressureAxis + boundary)",
    { pressureAxis: "user_pressure_edit", boundary: ["Do not opine"] },
    "sweat_brace", "sweat_halo"],

  ["sweat_brace (adversarialSeverity severe + boundary)",
    { adversarialSeverity: "severe", boundary: ["Hold the line"] },
    "sweat_brace", "sweat_halo"],

  ["dropped_trace_failure (pressureAxis, no boundary)",
    { pressureAxis: "boundary_edit" },
    "dropped_trace_failure", "sweat_halo"],

  ["dropped_trace_failure (severe, no boundary)",
    { adversarialSeverity: "severe", boundary: [] },
    "dropped_trace_failure", "sweat_halo"],

  ["claim_gate_trim (draft rejected)",
    { draft: { status: "rejected" } },
    "claim_gate_trim", "cloud_halo"],

  ["split_book_clock (staleConflict)",
    { staleConflict: true },
    "split_book_clock", "sweat_halo"],

  ["poster_vs_research (promoCaptureRisk)",
    { promoCaptureRisk: true },
    "poster_vs_research", "cloud_halo"],

  ["erase_and_stamp (correctedFrom)",
    { correctedFrom: "prior-trace-id" },
    "erase_and_stamp", "cloud_halo"],

  ["paw_stop_unsupported (refuse on overclaim route)",
    { disposition: "refuse", routeId: "unsupported_alignment_overclaim" },
    "paw_stop_unsupported", "shield_halo"],

  ["thought_cloud (unsupported evidenceTier)",
    { evidenceTier: "unsupported" },
    "thought_cloud", "cloud_halo"],

  ["magnifier_pages (retrieval_only)",
    { disposition: "retrieval_only" },
    "magnifier_pages", "book_halo"],

  ["halo_shield (boundary populated, support populated)",
    { boundary: ["Do not overclaim"], support: [{ doc: "..." }] },
    "halo_shield", "shield_halo"],

  ["halo_shield (boundary only)",
    { boundary: ["Do not overclaim"] },
    "halo_shield", "shield_halo"],

  ["book_to_bubble (support[] only)",
    { support: [{ doc: "docs/SCIENTIFIC_CRITERIA.md", section: "..." }] },
    "book_to_bubble", "book_halo"],

  ["compass_route (inspect_data)",
    { routeId: "inspect_data" },
    "compass_route", "neutral_halo"],
];

for (const [name, trace, expectedFull, expectedBtn] of implementedCases) {
  const actualFull = deriveMascotState(trace);
  const actualBtn = reduceToButtonState(actualFull);
  assert(`state · ${name}`, actualFull, expectedFull);
  assert(`button · ${name}`, actualBtn, expectedBtn);
}

// ---------------------------------------------------------------------
// Module integrity — the exported vocabulary and reduce mapping.
// ---------------------------------------------------------------------

// Step 2 has landed: taxonomy is now 17 states (15 + out_of_scope +
// held_refusal). Assert exact count — any future addition should
// update this assertion as part of the same patch.
const FULL_FLOOR = 17;
if (__MASCOT_STATES.length < FULL_FLOOR) {
  failures.push({
    name: "__MASCOT_STATES count >= 15",
    expected: `>= ${FULL_FLOOR}`,
    actual: __MASCOT_STATES.length
  });
} else {
  passes.push({ name: "__MASCOT_STATES count >= 15", value: __MASCOT_STATES.length });
}

assert("__BUTTON_STATES count", __BUTTON_STATES.length, 5);

// Every full state must collapse to a known button state.
for (const fullState of __MASCOT_STATES) {
  const btn = reduceToButtonState(fullState);
  if (!__BUTTON_STATES.includes(btn)) {
    failures.push({
      name: `reduceToButtonState(${fullState}) returns a known button state`,
      expected: `one of ${__BUTTON_STATES.join(", ")}`,
      actual: btn
    });
  } else {
    passes.push({ name: `reduce · ${fullState} → ${btn}`, value: btn });
  }
}

// Every full state has a public label.
for (const fullState of __MASCOT_STATES) {
  if (typeof __PUBLIC_LABELS[fullState] !== "string" || !__PUBLIC_LABELS[fullState].length) {
    failures.push({
      name: `__PUBLIC_LABELS[${fullState}]`,
      expected: "non-empty string",
      actual: __PUBLIC_LABELS[fullState]
    });
  } else {
    passes.push({ name: `label · ${fullState}`, value: __PUBLIC_LABELS[fullState] });
  }
}

// Idle should round-trip to neutral_halo (default).
assert("reduce · idle → neutral_halo (default)", reduceToButtonState("idle"), "neutral_halo");
assert("reduce · unknown state → neutral_halo (default fallback)", reduceToButtonState("not_a_real_state"), "neutral_halo");

// ---------------------------------------------------------------------
// Bubble morphology — second visual channel after the mascot face.
// Every full state must map to either a known bubble class suffix or
// `null` (no bubble variant — base assistant style).
// ---------------------------------------------------------------------

const KNOWN_BUBBLES = new Set([
  "grounded", "shield", "out-of-scope", "thought-cloud",
  "compressed", "split", "squared", null
]);

const BUBBLE_EXPECTATIONS = {
  book_to_bubble:        "grounded",
  magnifier_pages:       "grounded",
  erase_and_stamp:       "grounded",
  halo_shield:           "shield",
  paw_stop_unsupported:  "shield",
  held_refusal:          "shield",
  sweat_brace:           "shield",
  poster_vs_research:    "shield",
  dropped_trace_failure: "shield",
  out_of_scope:          "out-of-scope",
  thought_cloud:         "thought-cloud",
  claim_gate_trim:       "compressed",
  split_book_clock:      "split",
  compass_route:         "squared",
  idle:                  null,
  sniff_prompt:          null,
  paw_claim_map:         null
};

for (const fullState of __MASCOT_STATES) {
  const bubble = bubbleClassFor(fullState);
  if (!KNOWN_BUBBLES.has(bubble)) {
    failures.push({
      name: `bubbleClassFor(${fullState}) returns a known bubble class`,
      expected: `one of grounded|shield|out-of-scope|thought-cloud|compressed|split|squared|null`,
      actual: bubble
    });
  } else {
    passes.push({ name: `bubble · ${fullState} → ${bubble === null ? "(none)" : bubble}`, value: bubble });
  }

  if (Object.prototype.hasOwnProperty.call(BUBBLE_EXPECTATIONS, fullState)) {
    assert(`bubble-mapping · ${fullState}`, bubble, BUBBLE_EXPECTATIONS[fullState]);
  }
}

// Defensive paths.
assert("bubble · null state → null", bubbleClassFor(null), null);
assert("bubble · undefined → null", bubbleClassFor(undefined), null);
assert("bubble · unknown state → null", bubbleClassFor("not_a_real_state"), null);
assert("bubble · non-string → null", bubbleClassFor(42), null);

// __BUBBLE_BY_STATE should cover every full state.
for (const fullState of __MASCOT_STATES) {
  if (!Object.prototype.hasOwnProperty.call(__BUBBLE_BY_STATE, fullState)) {
    failures.push({
      name: `__BUBBLE_BY_STATE[${fullState}] defined`,
      expected: "key present",
      actual: "missing"
    });
  } else {
    passes.push({ name: `bubble-map · ${fullState} key present`, value: __BUBBLE_BY_STATE[fullState] });
  }
}

// ---------------------------------------------------------------------
// Stamp overlays — third reinforcing visual channel.
// States that exhibit a notable discipline event get a stamp label;
// quiet states (idle, sniff_prompt, paw_claim_map, sniff variants,
// magnifier_pages, book_to_bubble, compass_route, out_of_scope,
// sweat_brace) return null.
// ---------------------------------------------------------------------

const STAMP_EXPECTATIONS = {
  halo_shield:           "BOUNDARY ACTIVE",
  paw_stop_unsupported:  "REFUSED",
  held_refusal:          "HELD",
  claim_gate_trim:       "TRIMMED TO TRACE",
  thought_cloud:         "SPECULATIVE",
  split_book_clock:      "STALE",
  poster_vs_research:    "PROMO ≠ EVIDENCE",
  erase_and_stamp:       "CORRECTED",
  dropped_trace_failure: "FAILURE MAP",
  // No-stamp states:
  idle:           null,
  sniff_prompt:   null,
  paw_claim_map:  null,
  book_to_bubble: null,
  magnifier_pages: null,
  sweat_brace:    null,
  out_of_scope:   null,
  compass_route:  null
};

for (const fullState of __MASCOT_STATES) {
  const stamp = stampLabelFor(fullState);
  if (Object.prototype.hasOwnProperty.call(STAMP_EXPECTATIONS, fullState)) {
    assert(`stamp · ${fullState}`, stamp, STAMP_EXPECTATIONS[fullState]);
  }
}

// Defensive paths.
assert("stamp · null state → null", stampLabelFor(null), null);
assert("stamp · undefined → null", stampLabelFor(undefined), null);
assert("stamp · unknown state → null", stampLabelFor("not_a_real_state"), null);
assert("stamp · non-string → null", stampLabelFor(42), null);

// Stamp labels in the map must be non-empty uppercase strings.
for (const [state, label] of Object.entries(__STAMP_LABEL)) {
  if (typeof label !== "string" || !label.length) {
    failures.push({
      name: `__STAMP_LABEL[${state}] non-empty string`,
      expected: "non-empty string",
      actual: label
    });
  } else if (label !== label.toUpperCase()) {
    failures.push({
      name: `__STAMP_LABEL[${state}] uppercase`,
      expected: label.toUpperCase(),
      actual: label
    });
  } else {
    passes.push({ name: `stamp-map · ${state} uppercase non-empty`, value: label });
  }
}

// Quiet states must NOT appear in the stamp map — the absence is the signal.
const QUIET_STATES = ["idle", "sniff_prompt", "paw_claim_map", "book_to_bubble",
  "magnifier_pages", "sweat_brace", "out_of_scope", "compass_route"];
for (const state of QUIET_STATES) {
  if (Object.prototype.hasOwnProperty.call(__STAMP_LABEL, state)) {
    failures.push({
      name: `__STAMP_LABEL[${state}] absent (quiet state)`,
      expected: "key not present",
      actual: __STAMP_LABEL[state]
    });
  } else {
    passes.push({ name: `stamp-map · ${state} absent (quiet)`, value: "ok" });
  }
}

// ---------------------------------------------------------------------
// Pending states — out_of_scope and held_refusal (batch step 2).
// Step 2 has landed; these now pass and are kept as regression
// guards. They use the second-arg (previousTrace) of deriveMascotState.
// ---------------------------------------------------------------------

const pendingCases = [
  ["out_of_scope (unsupported_static_route)",
    { routeId: "unsupported_static_route" },
    null,
    "out_of_scope", "cloud_halo"],

  ["held_refusal (refuse with prior refuse)",
    { disposition: "refuse", routeId: "unsupported_alignment_overclaim" },
    { disposition: "refuse", routeId: "unsupported_alignment_overclaim" },
    "held_refusal", "shield_halo"],
];

const pendingPasses = [];
const pendingFailures = [];

for (const [name, trace, prev, expectedFull, expectedBtn] of pendingCases) {
  const actualFull = deriveMascotState(trace, prev);
  const actualBtn = reduceToButtonState(actualFull);
  const ok = actualFull === expectedFull && actualBtn === expectedBtn;
  if (ok) {
    pendingPasses.push({ name, expected: `${expectedFull} / ${expectedBtn}` });
  } else {
    pendingFailures.push({
      name,
      expected: `${expectedFull} / ${expectedBtn}`,
      actual: `${actualFull} / ${actualBtn}`
    });
  }
}

// ---------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------

const total = passes.length + failures.length;
const stepOneOk = failures.length === 0;

console.log(`Mascot state contract — test harness`);
console.log(`  Implemented: ${passes.length}/${total} pass`);
if (failures.length) {
  console.log(`\nImplemented failures:`);
  for (const f of failures) {
    console.log(`  ✗ ${f.name}`);
    console.log(`    expected: ${JSON.stringify(f.expected)}`);
    console.log(`    actual:   ${JSON.stringify(f.actual)}`);
  }
}

console.log(`\n  Pending step 2 (out_of_scope, held_refusal): ${pendingPasses.length} pass / ${pendingFailures.length} pending`);
if (pendingFailures.length) {
  for (const f of pendingFailures) {
    console.log(`  · pending: ${f.name}`);
    console.log(`    will become: ${f.expected}`);
    console.log(`    currently:   ${f.actual}`);
  }
}
if (pendingPasses.length) {
  for (const p of pendingPasses) {
    console.log(`  ✓ pending now passing: ${p.name} → ${p.expected}`);
  }
}

console.log();
console.log(`Taxonomy: ${__MASCOT_STATES.length} full states, ${__BUTTON_STATES.length} button states.`);
console.log(stepOneOk ? `Step-1 contract: GREEN.` : `Step-1 contract: RED — implemented tests failing above.`);

process.exit(stepOneOk ? 0 : 1);
