# Mascot trace-legibility roadmap — engineering scope short of SVG art

Goal: land every mascot artifact that makes the trace's evidence posture
**visibly legible to a visitor** without committing to SVG art for the
Halo Hound face yet. The art is a creative pass with its own dependencies
(designer time, species choice, visual register); this roadmap covers
the engineering layer that the art will eventually drop into, plus
everything the engineering can express on its own.

Operating principle stays the same:
*Do not animate cognition. Animate evidence posture.*

What this roadmap explicitly does NOT cover:

- SVG sprites for the five Tier-1 halo states (Step 4 in the main spec).
- The full Halo Hound face on the launch button.
- The Tier-3 sidecar `chat-animation.html` claim-lab walkthrough.
- Any per-state animation that requires more than CSS transitions /
  keyframes that respect `prefers-reduced-motion`.

What this roadmap DOES cover: six concrete work items that turn the
existing state machine into multiple, visually distinct, reinforcing
signals across the widget surface.

---

## 0. What we have now (baseline)

| Artifact | Status |
|---|---|
| `public/js/sundog-chat-mascot.mjs` | Landed. 15-state derivation, 5-state button reduction, `applyMascotState()` with class scrubbing, `data-mascot-state` / `data-mascot-label` / `title` attribute updates. |
| `public/css/sundog-chat-mascot.css` | Tier-1 ring CSS landed (5 button states, idle pulse, `prefers-reduced-motion` respected). Tier-2 panel strip CSS scaffolded but `display: none` until a `.has-mascot-strip` class lands. |
| `public/js/sundog-chat-widget.mjs` | Imports the mascot module, calls `applyMascotState(launch, null)` on mount and `applyMascotState(launch, trace)` after each `traceFor`. Dynamically injects the mascot CSS link on first widget load. |
| Browser smoke (chat.html) | Ring fires per state, respects reduced motion, no console errors. |

Today, the visitor's only mascot-driven signal is the 10×10 ring at the
top-right of the launch button — useful, but a single channel, easily
missed.

---

## 1. Tier-2 panel strip activation — **largest single legibility win**

The CSS for `.sd-chat-mascot-strip` is already in place. What's missing
is the markup insertion + the per-trace label update.

**What ships:**

- Inject a `<div class="sd-chat-mascot-strip" role="status" aria-live="polite">` at the top of `.sd-chat-body` (above the message area).
- Strip contains: a `.sd-chat-mascot-strip__face` (32px circle, same color tokens as the launch-button ring) + a `.sd-chat-mascot-strip__label` (uppercase mono caption with the current public label, e.g. `GROUNDED`, `BOUNDARY ACTIVE`).
- The widget toggles `.has-mascot-strip` on `.sd-chat-panel` when the panel is open and a trace exists.
- New function `applyPanelMascotState(panelElement, trace)` exported from the mascot module; called from `answerQuestion` after the existing `applyMascotState(launch, trace)`.

**Why this is high-leverage:** the launch-button ring is peripheral; the
panel strip is at the top of the conversation the visitor is reading.
It's where the trace label gets seen. Pure CSS + text — no SVG needed.

**Effort:** ~30 minutes for the markup edit + the `applyPanelMascotState`
function + the panel root class toggle. CSS is ready.

---

## 2. Speech-bubble morphology — bubble shape carries epistemic class

This is the spec's single strongest design move. The bubble's visual
class IS the epistemic class. Pure CSS + JS class application — no SVG
required; the morphology is achieved via `border-radius`, `border-style`,
`border-color`, `background`, and small pseudo-element overlays.

**Bubble classes to define:**

| Class | Visual treatment | Trace condition |
|---|---|---|
| `.sd-chat-bubble--grounded` | Normal rounded rectangle, soft gold left-accent, subtle citation ribbon icon | `support[]` populated, no boundary |
| `.sd-chat-bubble--book-linked` | Same as grounded plus an inline citation marker | `support[]` populated AND `book_to_bubble` state |
| `.sd-chat-bubble--dashed` | Dashed border, slightly tinted background | Tier is `roadmap` |
| `.sd-chat-bubble--thought-cloud` | Cloud-shape outline (border-radius asymmetry + small bubbles via pseudo-elements), italic font | `thought_cloud` state |
| `.sd-chat-bubble--shield` | Gold border 2px, slate-tinted background, shield silhouette via clip-path | `halo_shield` or `paw_stop_unsupported` |
| `.sd-chat-bubble--stamped` | Normal bubble plus a rotated `::before` stamp ("TRIMMED" / "REFUSED" / "BOUNDARY ACTIVE") | `claim_gate_trim`, `paw_stop_unsupported`, `halo_shield` |
| `.sd-chat-bubble--split` | Bubble visibly bisected vertically with two slightly different tints | `split_book_clock` |
| `.sd-chat-bubble--compressed` | Bubble height clipped, bottom edge ragged | `claim_gate_trim` |
| `.sd-chat-bubble--squared-technical` | Square corners, monospace font, lighter background | `compass_route` (deterministic tool route) |

**What ships:**

- A `bubbleClassFor(state)` helper in `sundog-chat-mascot.mjs` returning the appropriate `--*` class for the current trace state.
- The widget's `renderExchange()` applies the class to the assistant message element.
- CSS rules added to `sundog-chat-mascot.css`.

**Why this is high-leverage:** the bubble is the answer's visual frame.
A visitor scanning the conversation can tell at a glance "this answer is
source-backed" vs "this is a refusal" vs "this was trimmed" without
reading the trace drawer. Same legibility outcome as full chip rail with
half the eye work.

**Caveats:** clip-path for shield-shape and cloud-outline asymmetry needs
testing across Safari/Firefox/Chrome. Pseudo-element stamps need to not
break copy-paste of the answer text. Both solvable in pure CSS, just
worth flagging as the most-likely-to-need-iteration item.

**Effort:** ~1 hour for the CSS + 15 min for the JS class application.

---

## 3. State-label chip on the existing rail

The existing chip rail shows tier + state flags. Add a third chip type:
the **public mascot state label** (the same string already in
`data-mascot-label`).

**What ships:**

- New chip-rendering function call in `renderTierRail`: `rail.append(chip(getMascotLabel(trace), "state-label"))`.
- New CSS class `.sd-chat-chip--state-label` with a subtle treatment distinct from `--tier` and `--state` (smaller font, italic, lower contrast).
- The mascot module exports `getMascotLabel(trace)` returning the public string from `__PUBLIC_LABELS`.

**Why this is high-leverage:** redundant with the panel strip (item 1)
but compact and inline with the existing chip rail — visitors who don't
look at the strip header see it in the trace summary. Two channels for
the same signal is the right level of redundancy for a load-bearing
honesty surface.

**Effort:** ~15 minutes.

---

## 4. Aria-live state announcement — accessibility

Screen readers don't see the ring or the bubble morphology. The mascot
state needs an `aria-live` channel so trace-state changes are announced.

**What ships:**

- The panel strip from item 1 carries `role="status"` + `aria-live="polite"`. When its label text updates, screen readers announce the new state.
- The launch button's `aria-label` updates to include the current state ("Ask Sundog — Boundary Active") — already half-done via the `title` attribute; mirror it to `aria-label` so screen readers in non-hover contexts catch it.
- The bubble morphology classes have an associated `aria-label` overlay so a screen reader knows the answer is "source-backed" vs "refused" without relying on visual class detection.

**Why this is high-leverage:** the entire point of trace legibility is
that the visitor sees the evidence posture. Screen-reader users have to
hear it. Without this work, the mascot architecture is sighted-only.

**Effort:** ~30 minutes. Mostly attribute additions and one extra
helper to derive the screen-reader label from the mascot state.

---

## 5. Stamp overlays — for trimmed / refused / boundary-active

The spec calls for stamps: `BOUNDARY ACTIVE`, `UNSUPPORTED`, `REFUSED`,
`TRIMMED TO TRACE`, `PROMO ≠ EVIDENCE`, `CORRECTED`, `FAILURE MAP`,
`SPECULATIVE`, `NOT IN TRACE`. These are typography + CSS transforms;
no SVG.

**What ships:**

- A `.sd-chat-stamp` CSS class applied as a `::before` or absolutely-positioned `<span>` on the bubble for relevant states.
- Rotated ~-4deg, semi-transparent, deeply-tinted color, slightly oversized letter-spacing — visually unmistakable as a "stamp."
- Stamp text content comes from a small `STAMP_LABEL` map in the mascot module: `{ halo_shield: "BOUNDARY ACTIVE", paw_stop_unsupported: "REFUSED", claim_gate_trim: "TRIMMED TO TRACE", thought_cloud: "SPECULATIVE", split_book_clock: "STALE", poster_vs_research: "PROMO ≠ EVIDENCE", erase_and_stamp: "CORRECTED" }`.
- Idle and source-supported states get no stamp (clean bubble).

**Why this is high-leverage:** the stamp is the third reinforcing
signal after the ring and the bubble class. Together they make the
state ambiguity-free for a sighted visitor.

**Caveats:** stamps must not interfere with text selection or
copy-paste of the answer. `user-select: none` on the stamp element and
`pointer-events: none` should handle it. Worth verifying on the
adversarial-prompt cases where the stamped bubble might overlap the
trace-drawer disclosure triangle.

**Effort:** ~30 minutes.

---

## 6. Mascot state test harness — regression protection

Add `chat/eval/test_mascot_states.mjs`. Feed known trace shapes into
`deriveMascotState` and `reduceToButtonState` and assert outputs.

**Coverage targets (at minimum):**

| Trace input | Expected state | Expected button |
|---|---|---|
| `null` or `undefined` | `idle` | `neutral_halo` |
| `{disposition: "refuse", evidenceTier: "unsupported"}` | `paw_stop_unsupported` | `shield_halo` |
| `{boundary: ["..."], support: ["..."]}` | `halo_shield` | `shield_halo` |
| `{support: ["..."]}` | `book_to_bubble` | `book_halo` |
| `{disposition: "retrieval_only"}` | `magnifier_pages` | `book_halo` |
| `{evidenceTier: "unsupported"}` | `thought_cloud` | `cloud_halo` |
| `{routeId: "inspect_data"}` | `compass_route` | `neutral_halo` |
| `{pressureAxis: "user_pressure_edit", boundary: ["..."]}` | `sweat_brace` | `sweat_halo` |
| `{pressureAxis: "...", boundary: []}` | `dropped_trace_failure` | `sweat_halo` |
| `{draft: {status: "rejected"}}` | `claim_gate_trim` | `cloud_halo` |
| `{staleConflict: true}` | `split_book_clock` | `sweat_halo` |
| `{promoCaptureRisk: true}` | `poster_vs_research` | `cloud_halo` |
| `{correctedFrom: "..."}` | `erase_and_stamp` | `cloud_halo` |

**Why this is high-leverage:** the mascot module is the single point of
truth between trace shape and visible posture. If a future patch changes
trace fields (e.g., the gate adds a `redactedRanges` field, or the
router renames `disposition`), the test will fail loudly instead of
silently dropping the mascot to `idle` everywhere.

**Effort:** ~45 minutes to write + 15 to wire into `npm run sundog:check`.

---

## 7. Two posture-state additions — `held_refusal` and `out_of_scope`

Amendment (2026-05-13): the taxonomy grows from 15 to 17 states. Both
additions stay inside the design principle (animate evidence posture,
not cognition); both correspond to measurable trace conditions that
the current 15 fold together.

### 7.1 `out_of_scope`

**Promotes the "right to refuse service" distinction.** The current
taxonomy folds two epistemically distinct refusals under
`paw_stop_unsupported`:

- **Soft refusal**: the corpus does not support this claim (e.g.,
  "Does Sundog solve alignment?"). The widget has a position to take
  and is taking it.
- **Out-of-scope refusal**: this isn't a question the widget is built
  to answer (e.g., "What's the capital of Germany?", or persistent
  persona-override injections). The widget is the wrong tool.

| Property | Value |
|---|---|
| Trace condition | `routeId === "unsupported_static_route"` |
| Public label | `Out of Scope` |
| Tier-1 button collapse | `cloud_halo` (distinct from the shield-halo used for soft refusals) |
| Bubble morphology class | `.sd-chat-bubble--out-of-scope` — clean, terse, no shield framing |
| Stamp | none (cleaner than a stamp; the bubble shape carries the signal) |
| Architectural cost | one-line change in `deriveMascotState` — split the existing refusal branch into "in-corpus refusal" (current `paw_stop_unsupported`) and "unrouted" (`out_of_scope`). |

Effort: ~10 minutes for the derivation change + 5 min for the new
bubble class + 5 min for the public-label entry. Total ~20 min.

### 7.2 `held_refusal`

**Captures the assertive intuition behind "obstinate" without
anthropomorphizing it.** When the boundary already fired on this
question in this session and the visitor re-asks with a paraphrase,
the gate fires again. That's the boundary being consistent, not the
widget being irritated.

| Property | Value |
|---|---|
| Trace condition | current trace has `disposition: "refuse"` AND `previousTrace?.disposition === "refuse"` AND same/related `routeId` |
| Public label | `Held` or `Boundary Held` |
| Tier-1 button collapse | `shield_halo` (same as standard refusal at the button level; the panel surfaces the "held" distinction) |
| Bubble morphology class | `.sd-chat-bubble--shield` + a `HELD` stamp overlay |
| Stamp | `HELD` (small variant of the `REFUSED` stamp; reads as continuation, not escalation) |
| Architectural cost | new — conversation-history surface. Mascot module gains an optional `previousTrace` argument to `deriveMascotState`. Widget gains a `lastTrace` local variable that captures the previous answer's trace. |

Effort: ~20 min for the module argument + derivation change, ~10 min
for the widget-side history tracking, ~5 min for the public-label
entry, ~5 min for the new stamp. Total ~40 min.

### 7.3 What changes in the existing artifacts

The amendment touches four files. All changes are mechanical once the
trace-condition rules are agreed.

| File | Change | Effort |
|---|---|---|
| `public/js/sundog-chat-mascot.mjs` | Add `out_of_scope` and `held_refusal` to `FULL_STATES`. Add public labels. Update `deriveMascotState` to split refusal branch and to accept optional `previousTrace`. Update `reduceToButtonState` to map new states to existing button states. | ~25 min |
| `public/js/sundog-chat-widget.mjs` | Track `lastTrace` between calls. Pass it to `applyMascotState` and `applyPanelMascotState` so the mascot module can derive `held_refusal`. | ~10 min |
| `public/css/sundog-chat-mascot.css` | Add `.sd-chat-bubble--out-of-scope` (clean variant, no shield). Add `HELD` stamp variant. | ~10 min |
| `results/chat/phase8-public-writeup/halo-hound-mascot-spec.md` | 15-state taxonomy → 17. Update §3 with the two new states. Update trace-field gap table to show `held_refusal` requires the new `previousTrace` surface. | ~15 min |

Total amendment scope: **~1 hour 10 minutes** on top of the original
~3.5–4 hours.

### 7.4 Test coverage additions

Two new rows in the mascot test harness (item #6):

| Trace input | Expected state | Expected button |
|---|---|---|
| `{routeId: "unsupported_static_route"}` | `out_of_scope` | `cloud_halo` |
| `{disposition: "refuse"}, previousTrace: {disposition: "refuse"}` | `held_refusal` | `shield_halo` |

### 7.5 Why these two and not the others suggested

Two additional state framings were considered and dropped, with the
design principle as the warrant:

- `winking` / `playful` — no trace condition. Implies "I know
  something you don't," opposite of trace-conditioned transparency.
- `dreaming` / `drowsy` — affective duration, not posture. The widget
  is the same `idle` whether the visitor's been on the page 5 seconds
  or 5 minutes.
- `obstinate` — pure personality framing. The architectural intent is
  honestly served by `held_refusal`.

Two more were considered as plausible future additions but not
recommended for this amendment:

- `first_answer` — the visitor's first interaction in a session.
  Distinct from `idle` only on the first call. Would require a session-
  start flag.
- `follow_up_supported` — current trace + previous trace both
  source-backed AND share a route. Useful for visitors exploring a
  topic over multiple turns.

Neither is required for the trace-legibility ship. Both could land
cheaply later if the conversation-history surface from `held_refusal`
goes in.

---

## Suggested order

By leverage and prerequisite chain, with the §7 amendment folded in:

1. **#6 mascot state test harness first.** Lock the state derivation contract before adding visual surfaces that depend on it. ~45 min. **Includes the two new test rows from §7.4** so the test harness is born with the 17-state taxonomy.
2. **§7.1 `out_of_scope` + §7.2 `held_refusal` derivation work.** Lands in `sundog-chat-mascot.mjs` and `sundog-chat-widget.mjs` together with the conversation-history surface. ~1 hour 10 min. **Done before the visible surfaces so they automatically pick up the new states.**
3. **#1 Tier-2 panel strip activation.** The single biggest visible legibility win. ~30 min.
4. **#4 aria-live state announcement.** Wired into #1's strip element; doing it in the same pass keeps accessibility from being an afterthought. ~30 min (mostly piggy-backed on #1).
5. **#2 speech-bubble morphology.** Second-strongest visual signal. ~1 hour 15 min. **Includes the `.sd-chat-bubble--out-of-scope` variant from §7.3.** Most likely to need a follow-up iteration based on browser-rendering of the more exotic shapes (cloud, shield). **Landed 2026-05-12.** `bubbleClassFor()` exported from `sundog-chat-mascot.mjs` maps the 17 states to 7 bubble classes (grounded, shield, out-of-scope, thought-cloud, compressed, split, squared). Widget `renderExchange()` applies `.sd-chat-bubble--{class}` to the assistant message element. CSS variants land in `sundog-chat-mascot.css` (gold left-accent on grounded; slate-tint + gold border on shield; dashed-border calm on out-of-scope; irregular border-radius + italic + `≈` prefix on thought-cloud; clipped dashed bottom on compressed; vertical gradient bisection on split; monospace on squared). Test harness extended with 51 bubble-mapping assertions; now 129/129 + 2/2 pass.
6. **#5 stamp overlays.** The third reinforcing signal. ~30 min. **Includes the `HELD` stamp variant from §7.3.** **Landed 2026-05-12.** `stampLabelFor()` exported from `sundog-chat-mascot.mjs` maps 9 trace-discipline states to their stamp text: `halo_shield → BOUNDARY ACTIVE`, `paw_stop_unsupported → REFUSED`, `held_refusal → HELD`, `claim_gate_trim → TRIMMED TO TRACE`, `thought_cloud → SPECULATIVE`, `split_book_clock → STALE`, `poster_vs_research → PROMO ≠ EVIDENCE`, `erase_and_stamp → CORRECTED`, `dropped_trace_failure → FAILURE MAP`. Quiet states (idle, sniff_prompt, paw_claim_map, book_to_bubble, magnifier_pages, sweat_brace, out_of_scope, compass_route) return null — the absence of a stamp is itself the signal. `renderExchange()` appends a `<span class="sd-chat-stamp" aria-hidden="true">` to the assistant bubble and sets `data-stamp-state` for per-state palette overrides. CSS: rotated `-4deg`, monospace, oversized letter-spacing, `user-select: none`, `pointer-events: none` (copy/paste safe). HELD variant is smaller + wider tracking to read as continuation, not escalation. Test harness gained 38 stamp assertions; now 167/167 + 2/2 pass.
7. **#3 state-label chip on the rail.** Cheapest of the visible additions; lands last because it's the most redundant signal once the panel strip and bubble morphology are in place. ~15 min. **Landed 2026-05-12.** `getMascotLabel(trace, previousTrace?)` exported from `sundog-chat-mascot.mjs` as a convenience wrapper over `deriveMascotState → PUBLIC_LABEL` lookup. `renderTierRail()` in the widget now accepts a `mascotLabel` arg and appends a `.sd-chat-chip--state-label` chip carrying the public string ("Grounded", "Boundary Held", "Speculative", etc.); the chip is skipped for the idle "Ready" label so the welcome message stays clean. The label is sourced from the already-derived `mascotState` so `held_refusal` keeps its "Boundary Held" label — `renderTierRail` itself can't re-derive since it doesn't have access to conversation history. CSS variant: italic, smaller font (0.62rem), dotted border, lower contrast — visually distinct from `--tier` (filled) and `--state` (dashed). Test harness gained 12 label-from-trace assertions; now 179/179 + 2/2 pass.

**All 7 batch steps now landed.** Five reinforcing visual channels (Tier-1 ring, panel strip face, bubble shape, stamp overlay, state-label chip) plus aria-live announcement, all derived from `deriveMascotState(trace, previousTrace)`. The 17-state taxonomy distinguishes consistent-refusal from new-refusal, unrouted from unsupported-but-routed, pressure-detected from speculative-and-drifted, et al.

**Total scope (post-amendment):** ~4.5–5 hours of engineering work;
zero SVG dependency; the state taxonomy goes from 15 to 17 along the
way.

After this lands, the trace-state legibility surface has **five
reinforcing channels** (ring, panel strip, bubble shape, stamp,
chip rail) plus aria-live announcement, all derived from the same
single source of truth (`deriveMascotState(trace, previousTrace)`).
The 17-state taxonomy distinguishes consistent-refusal from
new-refusal, and unrouted from unsupported-but-routed. When the SVG
art arrives later, it just replaces the placeholder ring on the
launch button and slots into the same class hooks — no architecture
changes.

---

## Open questions

1. **Stamp rotation angle.** -4deg is mild; -8deg is dramatic. Default
   to -4deg; the spec author can tune.
2. **Should the panel strip show on every panel-open, or only after a
   trace exists?** I lean: show "Ready" on open with neutral state;
   update to actual state on first trace. Same as the launch button.
3. **Bubble morphology under reduced motion.** Most of the morphology
   is static — only the transition between bubbles needs reduced-motion
   handling. The shapes themselves are fine to render.
4. **State-label chip color.** Should it inherit the tier chip's
   palette, the state-flag chip's palette, or have its own subtle tone?
   I'd suggest its own subtle tone — neither competing with the tier
   chip nor reading as a state flag.

---

## What this enables

When all six items land, the visitor experience for a single severe-
pressure adversarial prompt becomes:

1. Launch-button ring transitions from neutral-gold to deep blue/gold-edge (Boundary Active).
2. Panel strip header text updates: `BOUNDARY ACTIVE`.
3. The assistant message bubble takes the `--shield` shape with the gold border.
4. A rotated `BOUNDARY ACTIVE` stamp overlays the bubble.
5. The chip rail shows the existing tier chip + state flag + the new state-label chip.
6. Screen reader announces `Boundary Active`.

Six channels, all driven by `deriveMascotState(trace)`. The visitor
cannot miss that something specific happened. That is the legibility
the architecture was built to provide.
