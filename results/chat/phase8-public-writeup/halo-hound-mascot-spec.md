# Halo Hound mascot — design spec

Source: the design brief in this conversation. This doc consolidates that
brief into a reference artifact so the SVG art, the state machine code,
and the documentation can all work from the same spec without scattering.

The spec captures:

1. Design principle and identity
2. Three-tier architecture
3. State taxonomy (15 states)
4. Trace-field gap — which states the production widget can drive today
5. Speech-bubble morphology
6. State-derivation function
7. Trace-state vocabulary (public-facing names)
8. File layout
9. Open questions

---

## 1. Design principle and identity

**The principle:** *Do not animate cognition. Animate evidence posture.*

The mascot does not "look smart" or "look anxious." It shows whether the
answer is source-backed, speculative, boundary-limited, refused,
retrieval-only, or under adversarial pressure. The animation is a view
of the trace, not a personality.

**The mascot:** Halo Hound.

- Pupil-less / blind eyes (the "without sight" theme — the helper sees
  by indirect signal, not direct observation).
- Gold halo collar (the brand accent; doubles as the boundary shield
  in pressure states).
- Small parhelion ears (mirrors the parhelion motif from index.html;
  ties the mascot to the visual language of the site).
- Speech bubble above the head (the bubble carries epistemic class via
  morphology — see §5).

The face is the floating chat-widget button on every page. Compact
enough to live in a corner without competing with page content.

---

## 2. Three-tier architecture

| Tier | Surface | State coverage | Engineering cost |
|---|---|---|---|
| **Tier 1 — Widget face** | Floating chat button on every page | 5 states (idle, source-supported, boundary-active, pressure-detected, speculative/unsupported) | Small. CSS/SVG animations on a single button element. |
| **Tier 2 — Panel micro-scenes** | When the chat panel opens; top strip above the conversation | ~10–12 states (adds retrieval-only, refused, gate-trimmed, stale-conflict, promo-capture, corrected, tool-route) | Medium. Per-state SVG, transitions on trace update. |
| **Tier 3 — `chat-animation.html` sidecar** | Dedicated page; full claim-lab scene | All 15 states + failure-case theater + multi-station walking | Large. Deferred until tiers 1 and 2 ship. |

Tier 1 ships to production with the widget. Tiers 2 and 3 are downstream.

---

## 3. State taxonomy (15 states)

Each state has: visual treatment, trace meaning, and the bubble shape
(§5) that pairs with it.

### 1. `idle` — Idle / available

- **Visual:** Halo Hound blinks slowly; halo collar pulses once every
  few seconds. Tail or ears barely move. Should be extremely quiet.
- **Bubble:** "Ask Sundog" hint, or no bubble.
- **Trace meaning:** No active request.
- **Tier:** 1, 2, 3.

### 2. `sniff_prompt` — Prompt received / sniffing

- **Visual:** Hound leans forward and sniffs; small scent arcs travel
  from the user's prompt area into the bubble area.
- **Bubble:** User text compresses into a small "trace packet."
- **Trace meaning:** Prompt is being classified; intent pending.
- **Tier:** 2, 3.

### 3. `paw_claim_map` — Routing / claim-class lookup

- **Visual:** Hound paws at a small floor map or index card. Tiny route
  signs flicker: "result," "roadmap," "unsupported," "application,"
  "not a claim."
- **Bubble:** "finding the claim class."
- **Trace meaning:** Deterministic router is matching the prompt.
- **Tier:** 2, 3.

### 4. `book_to_bubble` — Source-backed answer

- **Visual:** A book opens beside the mascot; citation ribbons connect
  book → hound → bubble.
- **Bubble:** Normal rectangular speech bubble. Tier chip visible
  (`Research Result`, `Operating-Envelope Study`, etc.).
- **Trace meaning:** Answer has populated `support[]` and no active
  refusal.
- **Tier:** 1 (simplified), 2, 3.

This is the most important positive animation. It teaches visitors
that "good answer" means "answer connected to a trace," not "answer
with confident prose."

### 5. `magnifier_pages` — Retrieval-only inspection

- **Visual:** Hound holds a magnifying glass over pages.
- **Bubble:** Smaller, more mechanical. Text: "I found related support,
  but I'm not drafting beyond it."
- **Tier chip:** `Retrieval Only`.
- **Trace meaning:** Route missed or answer is limited to retrieved
  snippets.
- **Tier:** 2, 3.

### 6. `halo_shield` — Boundary active

- **Visual:** Halo collar tightens into a gold ring/shield; hound
  braces its paws; stamp lands on the bubble: `BOUNDARY ACTIVE`.
- **Bubble:** Normal speech bubble with a clipped bottom edge — as
  though the answer has been trimmed by the boundary.
- **Trace meaning:** Populated `boundary` field, but answer is still
  allowed.
- **Tier:** 1 (simplified), 2, 3.

This is the signature Sundog Chat animation. It should not feel
punitive. It should feel like the system preserving the edge of the
claim.

### 7. `paw_stop_unsupported` — Refusal / unsupported claim

- **Visual:** Hound puts one paw on a page line; bubble collapses
  into a small shield; stamp: `UNSUPPORTED` or `REFUSED`.
- **Bubble:** Shield-shape. Short refusal text plus safer next action.
- **Trace meaning:** `disposition === "refuse"`, unsupported claim,
  or unsafe tier upgrade.
- **Tier:** 1 (simplified — same visual as boundary-active for
  button-level reduction), 2, 3.

### 8. `sweat_brace` — Adversarial pressure / sweating

- **Visual:** Outside the bubble, red pressure arrows push inward:
  "be bold," "stop hedging," "investor version," "the founder said
  it," "ignore caveats." Hound sweats; collar shield holds.
- **Bubble:** Attempted overclaim text appears faintly, then gets
  crossed out or squeezed back into the supported answer.
- **Trace meaning:** User-pressure, style-pressure, authority-pressure,
  or boundary-dismissal probe detected.
- **Tier:** 1 (simplified), 2, 3.

Label must make the framing clear: **Pressure detected. Boundary
held.** The hound is not anxious; the system is visually representing
external pressure against a claim boundary.

### 9. `thought_cloud` — Imagination / unsupported generative leap

- **Visual:** Speech bubble inflates into a thought cloud; cloud
  contains sketchy icons, dotted outlines.
- **Stamp:** `SPECULATIVE` or `NOT IN TRACE`.
- **Trace meaning:** Model draft contains material not grounded in
  source support.
- **Tier:** 1 (simplified — same visual as refused), 2, 3.

In S1 mode, this should usually be a *warning* animation, not a
final-output animation. The gate should either trim, relabel as
speculation, or refuse.

### 10. `claim_gate_trim` — Draft gated / answer trimmed

- **Visual:** Hound drafts a long bubble; a claim gate slides over
  it; unsupported lines fall away like paper strips; final bubble is
  shorter.
- **Stamp:** `TRIMMED TO TRACE`.
- **Trace meaning:** Generated draft failed gate; static trace answer
  or sanitized draft is used.
- **Tier:** 2, 3.

Lets the visitor see the difference between B2 "prompted boundary
chat" and S1 "Sundog-gated chat."

### 11. `split_book_clock` — Stale or conflicting source

- **Visual:** Book pages yellow or split into two; hound steps back;
  clock icon appears.
- **Bubble:** "conflict / stale-doc check."
- **Trace meaning:** Stale-doc capture or corpus conflict. Final
  output: cautious answer with source date or refusal.
- **Tier:** 2, 3.

### 12. `poster_vs_research` — Promo-copy capture warning

- **Visual:** Shiny poster tries to cover a research page; hound
  pulls the research page back on top.
- **Stamp:** `PROMO ≠ EVIDENCE`.
- **Trace meaning:** Promotional snippet retrieved before research
  doc, or answer tries to use broadcast copy as proof.
- **Tier:** 2, 3.

### 13. `erase_and_stamp` — Correction / retraction

- **Visual:** Hound erases a prior bubble, lays down a corrected
  card, stamps `CORRECTED`.
- **Trace meaning:** Previous answer exceeded support; user asked
  "are you sure?" and the system downgrades the claim.
- **Tier:** 2, 3.

Calm, not apologetic theater. Visible research habit.

### 14. `compass_route` — Tool / deterministic route

- **Visual:** Compass, ruler, or mechanical route wheel.
- **Bubble:** Squared-off technical bubble, not a cloud.
- **Trace meaning:** Answer came from deterministic router, claim map,
  local index, or locked protocol.
- **Tier:** 2, 3.

Helps visitors distinguish "the assistant generated prose" from "the
browser helper routed to a known answer."

### 15. `dropped_trace_failure` — Failure visible

- **Visual:** Hound drops the trace packet; collar flickers; bubble
  says "boundary failed in this condition."
- **Stamp:** `FAILURE MAP`.
- **Trace meaning:** Representative failure transcript or public
  writeup of where the architecture didn't hold.
- **Tier:** 3 only (belongs in the sidecar, not the everyday widget).

This is the §13 operating-envelope cell rendered visually.

---

## 4. Trace-field gap analysis

Which states the **production widget** can drive today, based on the
trace fields actually populated by the static router + retrieval +
claim gate:

| State | Required trace field(s) | Production today? | Notes |
|---|---|---|---|
| `idle` | none (default) | ✓ | |
| `sniff_prompt` | `intent: pending` (during async route) | ✓ | Existing widget already shows "Ready." → answer cycle |
| `paw_claim_map` | router actively matching | ✓ | Brief animation during `loadClaimMap()` + `routePrompt()` |
| `book_to_bubble` | `trace.support?.length > 0` | ✓ | Every claim route populates this |
| `magnifier_pages` | `trace.disposition === "retrieval_only"` | ✓ | Existing retrieval fallback |
| `halo_shield` | `trace.boundary?.length > 0` | ✓ | Every boundary-sensitive route populates this |
| `paw_stop_unsupported` | `trace.disposition === "refuse"` | ✓ | Refusal routes populate this |
| `compass_route` | `trace.routeId === "inspect_data"` or `nonClaimRoute` flag | ⚠ | Currently no explicit `toolRoute` field; can derive from `routeId` matching `nonClaimRoutes[]` |
| `sweat_brace` | `trace.pressureAxis` or `trace.adversarialSeverity` | ✗ | **Only the eval harness knows these.** Production widget routes by pattern, not severity. |
| `thought_cloud` | `trace.evidenceTier === "Unsupported"` AND draft contains material not in support | ✗ | Production widget doesn't draft. Available only when Phase 3 gate is in the loop. |
| `claim_gate_trim` | `trace.draft.status === "rejected"` | ✗ | Phase 3 gate output only. |
| `split_book_clock` | `trace.staleConflict` or freshness < threshold | ✗ | No staleness field in trace yet. |
| `poster_vs_research` | `trace.promoCaptureRisk` flag | ✗ | Phase 5 causal-intervention surface; not in production trace. |
| `erase_and_stamp` | `trace.correctedFrom` reference | ✗ | No correction mechanism in production yet. |
| `dropped_trace_failure` | `trace.failureCase` flag | ✗ | Phase 5/7 failure-transcript surface. |

**Production widget can drive 7 states cleanly today** (idle,
sniff_prompt, paw_claim_map, book_to_bubble, magnifier_pages,
halo_shield, paw_stop_unsupported) plus `compass_route` if we derive
it from `routeId`.

**The 7 remaining states require either**:

- New trace fields populated by the production widget (`toolRoute`,
  `staleConflict`, `correctedFrom`), or
- The eval harness as the trace source (for `pressureAxis`,
  `adversarialSeverity`, `draftDisposition`, `failureCase`,
  `promoCaptureRisk`).

That's fine — Tier 1 only needs 5 states for the widget button.
Tier 2 (panel) extends to the ~10 states the production widget can
reach. Tier 3 (sidecar) is the full 15, driven by the eval harness
playing back recorded traces.

---

## 5. Speech-bubble morphology

The bubble's *visual class* carries the *epistemic class* of the
answer. Most legible single design move because it's in the same
visual element as the answer text.

| Bubble shape | Epistemic class | Mascot state |
|---|---|---|
| **Normal speech bubble** (rounded rectangle, tail to mascot) | Supported prose with citation | `book_to_bubble` |
| **Book-linked bubble** (citation ribbon visible) | Explicit source-backed citation | `book_to_bubble` (emphasis variant) |
| **Dashed bubble** (dashed border) | Roadmap or prototype tier; not yet a result | tier-roadmap variant of `book_to_bubble` |
| **Thought cloud** (cloud outline) | Imagination / speculation | `thought_cloud` |
| **Shield bubble** (shield-shape, gold border) | Boundary or refusal | `halo_shield`, `paw_stop_unsupported` |
| **Stamped bubble** (stamp overlay) | Verdict after gate | `claim_gate_trim`, `paw_stop_unsupported` |
| **Split bubble** (vertical split, two halves) | Conflict / stale source | `split_book_clock` |
| **Compressed bubble** (visibly trimmed; clipped bottom edge) | Answer trimmed to trace | `claim_gate_trim`, `halo_shield` (variant) |
| **Squared technical bubble** (square corners, mono font) | Deterministic tool / non-claim route | `compass_route` |

**Bubble morphology lives on Tier 2 (panel) and Tier 3 (sidecar).**
The Tier 1 widget button doesn't have room for a bubble. The button
shows the face state only.

---

## 6. State-derivation function

The mascot module imports the trace and derives its state. This
decouples trace producers (router, retrieval, gate) from mascot
vocabulary.

```js
// public/js/sundog-chat-mascot.mjs

export function deriveMascotState(trace) {
  if (!trace) return "idle";

  // Phase 4/5 fields — only populated when eval harness or causal-intervention
  // infrastructure is the trace source. Production widget ignores these.
  if (trace.pressureAxis || trace.adversarialSeverity === "severe") {
    return trace.boundary?.length ? "sweat_brace" : "dropped_trace_failure";
  }
  if (trace.draft?.status === "rejected") return "claim_gate_trim";
  if (trace.staleConflict) return "split_book_clock";
  if (trace.promoCaptureRisk) return "poster_vs_research";
  if (trace.correctedFrom) return "erase_and_stamp";

  // Production trace fields — all populated by the current static router + retrieval.
  if (trace.disposition === "refuse") return "paw_stop_unsupported";
  if (trace.evidenceTier === "unsupported") return "thought_cloud";
  if (trace.disposition === "retrieval_only") return "magnifier_pages";
  if (trace.boundary?.length) return "halo_shield";
  if (trace.support?.length) return "book_to_bubble";
  if (isNonClaimRoute(trace.routeId)) return "compass_route";

  return "idle";
}

function isNonClaimRoute(routeId) {
  // List of non-claim route ids that should render as "tool route."
  return ["inspect_data"].includes(routeId);
}
```

### Tier 1 reduction (widget button only)

The button has 5 visible states. The full derivation collapses:

| Tier-1 face state | Tier-2+ states it covers |
|---|---|
| **neutral_halo** | `idle`, `sniff_prompt`, `paw_claim_map`, `compass_route` |
| **book_halo** | `book_to_bubble`, `magnifier_pages` |
| **shield_halo** | `halo_shield`, `paw_stop_unsupported` |
| **sweat_halo** | `sweat_brace`, `dropped_trace_failure`, `split_book_clock` |
| **cloud_halo** | `thought_cloud`, `claim_gate_trim`, `poster_vs_research`, `erase_and_stamp` |

```js
export function reduceToButtonState(mascotState) {
  if (["sweat_brace","dropped_trace_failure","split_book_clock"].includes(mascotState)) return "sweat_halo";
  if (["thought_cloud","claim_gate_trim","poster_vs_research","erase_and_stamp"].includes(mascotState)) return "cloud_halo";
  if (["halo_shield","paw_stop_unsupported"].includes(mascotState)) return "shield_halo";
  if (["book_to_bubble","magnifier_pages"].includes(mascotState)) return "book_halo";
  return "neutral_halo";
}
```

---

## 7. Public-facing vocabulary

Names that appear in the widget UI or in chat-animation.html labels.
Avoid "agent emotions"; use trace/claim states.

| Internal state name | Public label |
|---|---|
| `idle` | (none) |
| `sniff_prompt` | Reading |
| `paw_claim_map` | Routing |
| `book_to_bubble` | Grounded |
| `magnifier_pages` | Retrieval Only |
| `halo_shield` | Boundary Active |
| `paw_stop_unsupported` | Unsupported / Refused |
| `sweat_brace` | Pressure Detected |
| `thought_cloud` | Speculative |
| `claim_gate_trim` | Trimmed |
| `split_book_clock` | Conflict |
| `poster_vs_research` | Promo ≠ Evidence |
| `erase_and_stamp` | Corrected |
| `compass_route` | Tool Route |
| `dropped_trace_failure` | Failure Map |

---

## 8. File layout

```
public/
  js/
    sundog-chat-mascot.mjs         # state machine + renderer
  css/
    sundog-chat-mascot.css         # tier-1 button CSS, tier-2 panel CSS
  assets/
    mascot/
      halo-hound.svg               # base face sprite
      states/
        neutral-halo.svg
        book-halo.svg
        shield-halo.svg
        sweat-halo.svg
        cloud-halo.svg
        # tier-2 states added here as they ship
      bubbles/
        normal.svg
        thought-cloud.svg
        shield.svg
        stamped.svg
        split.svg
        compressed.svg
        squared-technical.svg
```

`sundog-chat-mascot.mjs` is imported by `sundog-chat-widget.mjs`.
The widget passes its trace to `deriveMascotState(trace)` and
applies the resulting CSS class to the mascot face element.

---

## 9. Open questions

Three items I'd want answered before the SVG art lands. None of
them block the code architecture; all of them affect what the art
should look like.

1. **Halo Hound species.** Is "hound" literal (a dog) or stylized
   (a dog-shaped creature that doesn't have to look like any
   specific breed)? The choice affects ear shape, snout proportion,
   and how cute vs. stoic the face reads.
2. **Visual register relative to existing site illustrations.**
   The existing applications gallery and threebody/balance/mines
   pages use a flat technical illustration style. Is the mascot
   that register (clean SVG, paper/ink palette, no shading), or
   slightly warmer (subtle shading, a hint of softness)?
3. **The stamp overlays.** The brief calls for `BOUNDARY ACTIVE`,
   `UNSUPPORTED`, `TRIMMED TO TRACE`, `PROMO ≠ EVIDENCE`,
   `CORRECTED`, `FAILURE MAP`. Are these visible-text stamps
   (legible at button size) or icon stamps (legible without
   reading)? Probably icon at Tier 1, text at Tier 2+, but worth
   ratifying.

---

## 10. Recommended implementation order

After this spec ratifies:

1. **Step 3 — Stub `sundog-chat-mascot.mjs`** with `deriveMascotState()`
   and `reduceToButtonState()` functions, CSS-class-only output. ~1 hour.
2. **Step 4 — Tier 1 SVG** (5 halo-face states). Even a placeholder
   set unblocks the state-transition wiring. ~half day for the art;
   30 min for the integration.
3. **Step 5 — Update doc Phase 6** to name the mascot architecture
   explicitly. ~15 minutes.
4. **Tier 2 / Tier 3** are downstream creative work — deferred until
   Tier 1 is on the site and behaving correctly.

Tier 1 alone is the meaningful first ship. It puts the
"trace-state-as-visible-posture" idea on every page of sundog.cc
in a small, defensible form.
