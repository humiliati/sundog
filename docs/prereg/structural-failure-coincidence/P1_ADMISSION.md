# Structural Failure Coincidence — P1 Admission Review

Pre-registration: [`README.md`](README.md) (frozen 2026-05-15 PT)
P0 deliverable under review: [`BOUNDARY_MAP.md`](BOUNDARY_MAP.md)
(frozen 2026-05-15 PT)
Existing falsifiability surface cross-referenced:
[`../../debunked.md`](../../debunked.md)
Roadmap: [`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) ▸ Candidate 13
Filed: **2026-05-15 (PT)**. Author: maintainer. Status: append-only
below the **Amendments** rule; the body above it is the frozen P1
record.

P1 is the independent admission review the roadmap gates P2 on. It does
**not** run an agent. It (a) ratifies the frozen boundary map against the
prereg's *First Falsifier Before Agents* requirements, (b) rules the two
recorded coded-vs-stated reconciliations in writing, (c) cross-references
the new structural-failure falsifier against the existing falsifiability
surface (`debunked.md`) so the two surfaces communicate and a
prioritization exists, and (d) renders an admission verdict. Frozen
P0 bodies are not edited here; review findings are carried as amendments
with timestamped justification (prereg discipline).

## A. Locus ratification (L1–L5)

Required fields per locus: eligible regime · abstain/switch/fail regime ·
exact source receipt · traceable-agent prediction · mere-correlate
prediction.

| locus | 5 fields present | receipt verified (file checked 2026-05-15) | admit? |
| --- | --- | --- | --- |
| L1 Parhelion offset | yes | `HALO_PHENOMENA_ACCOUNTING.md` §A; `PHASE10_OPTICAL_AUDIT_HANDOFF.md`; `parhelion-geometry.mjs` `phase3.daggerOffset` | **ADMIT** |
| L2 CZA cutoff | yes | `parhelion-geometry.mjs` `czaVisibleAtAltitude` (coded `h≤32°`) | **ADMIT** (reconciliation §B-1) |
| L3 Tangent→circumscribed merge | yes | `parhelion-geometry.mjs` `TANGENT_ARC_CIRCUMSCRIBED_H=29` + `tangentArcLocus` null-guard; `PASS_C7_OUTPUT.txt`; Tape AH Ch6 p62 | **ADMIT** |
| L4 Supralateral gate | yes | `PHASE10_OPTICAL_AUDIT_HANDOFF.md` line 54 | **ADMIT** (reconciliation §B-2) |
| L5 Rendered ≠ anchored | partial — see finding | `HALO_PHENOMENA_ACCOUNTING.md` §A honesty ratchet | **ADMIT, RE-SCOPED** |

### P1 finding — L5 is an admissibility rule, not a behavioral locus

L1–L4 are *behavioral failure-coincidence loci*: each names an input
regime at which a route-traceable agent must measurably degrade, abstain,
or switch handles, and at which a mere correlate measurably will not.
That is a falsifiable behavioral prediction.

L5 ("rendered ≠ anchored") is structurally different. It does not name an
input regime at which agent behavior bifurcates; it constrains **what the
experimenter is permitted to count as inverse evidence** when scoring
L1–L4. Left mis-filed as a behavioral locus it is a soft spot: a correlate
could be argued "admissible" by pointing at a drawn-but-unanchored
primitive — a *definitional* escape, exactly the kind the prereg exists
to close. Re-scoping it removes that escape.

**Ruling:** L5 is admitted **as the evidence-admissibility rule governing
the scoring of L1–L4**, not as a standalone structural-failure-coincidence
test. Operationally: in P2, only anchored closed-form rows may be counted
as inverse evidence for L1–L4; the presence of any rendered-optional /
named-only / not-modeled / hardcoded-placeholder primitive is **never**
admissible as route evidence. This strengthens the falsifier; it does not
weaken it. The frozen P0 body is unchanged; this interpretation is
carried by amendment.

## B. Reconciliation rulings (in writing)

**B-1 — L2 CZA cutoff.** Coded guard `czaVisibleAtAltitude` ⇒ `h ≤ 32°`
(`parhelion-geometry.mjs`). Literature/prereg: ~32.2° (Tape AH Ch6 p63) /
32.196° (earlier accounting matrix). **Ruling: operative boundary =
`h ≤ 32°` (the coded guard), as the harness will enforce exactly that.**
The ~0.2° coded-vs-literature spread is recorded, not averaged, and is
inside any plausible measurement tolerance; it does not move L2's
regime classification. P2 scores L2 against `h = 32°`.

**B-2 — L4 supralateral.** Receipt figure: angular distance from the sun
varies ~**0.5° across h = 0–22°** (`PHASE10_OPTICAL_AUDIT_HANDOFF.md`
line 54); prereg states ~0.3° over a narrower tested span. **Ruling:
operative figure = ~0.5°/h=0–22° (the audit receipt).** Either figure
yields the same classification: L4 is a *permanent fail* row (no eligible
regime at any altitude); the structural-discrimination gate is failed
everywhere. No reclassification.

Neither reconciliation changes a regime. Both are now ruled and frozen
into the P1 record.

## C. Cross-reference vs the existing falsifiability surface

The user's concern is correct to raise and the answer is **not "fully
orthogonal."** `debunked.md` carries three failure candidates; their
relation to the structural-failure-coincidence falsifier:

| existing avenue (`debunked.md`) | what it stresses | relation to BOUNDARY_MAP | must the surfaces communicate? |
| --- | --- | --- | --- |
| **Pushable Occluder / Occlusion Path** (best candidate) | Can the paradigm *act to make the signal readable* (two-stage act-then-align), not just read it? | **Orthogonal axis** — action-prerequisite vs route-traceability. Neither gates the other. | Low-coupling. Keep as distinct rail cards; do **not** conflate a Pushable-Occluder "Boundary Found" with a traceability/opaque-correlate result. Its own roadmaps (`HIGHLIGHTS_RAIL_ROADMAP.md`, `PUSHABLE_OCCLUDER_ROADMAP.md`). |
| **Occluded Code: Score Aliasing** | Two hidden codes cast the same alignment shadow ⇒ no controller can honestly distinguish them. | **Adjacent on the identifiability axis** — the *engineered worst case* of L1–L4's *natural* identifiability singularities. | Medium. Fold in as an optional **P2 synthetic stress case**: inject an aliased cause-pair; a route-traceable agent must abstain exactly as the L-type prediction says. Does not gate P0/P1. |
| **Sundog vs Mesa-Optimization: Proxy Collapse** ("most important scientific failure") | Signature-trained controller becomes indistinguishable from a reward-trained one above a capacity/selection threshold. | **Same target, two surfaces.** Candidate 13's falsification target is explicitly *"Mode (2): signature-is-reward-in-costume, in its traceability form."* `debunked.md` itself flags Proxy Collapse as the most important failure but *under-instrumented* ("less immediately visual, belongs later"). **BOUNDARY_MAP is the pre-agent, falsifiable instrument for it.** | **High — must converge.** A P2 fail on steerability or boundary-coincidence *is* a Proxy-Collapse confirmation. Report once, mirrored on both surfaces, in shared vocabulary. |

### Prioritization (given winning = B or D; brainstorm: the traceability null is the most-feared and most-clarifying)

1. **Highest — this program (BOUNDARY_MAP, P0✓ → P1 → P2).** Pre-agent,
   cheapest, fastest B-or-D, and it operationalizes the very failure
   (`Proxy Collapse`) `debunked.md` named most important and least
   instrumented. Priority 1; nothing below blocks it.
2. **Parallel / independent — Pushable Occluder.** Orthogonal axis,
   product/rail-facing, own roadmaps; runs on its own cadence, neither
   blocks nor is blocked by P1/P2.
3. **Folded — Occluded Code: Score Aliasing.** Optional P2 synthetic
   stress case under the identifiability framing; not separately
   prioritized, not gating.
4. **Convergent reporting — Proxy Collapse.** Not a separate experiment;
   it is the deep reading of a BOUNDARY_MAP P2 fail. One result, two
   surfaces.

### Verdict-vocabulary bridge (so the surfaces communicate)

Prereg Outcome Branching ⟷ `debunked.md` rail vocabulary:

| prereg outcome | rail verdict | also surfaces as |
| --- | --- | --- |
| Cannot write boundary map | UNTESTED / halt | — (passed; not applicable) |
| Fails to converge in eligible regimes | STALLED or BUSTED (D path) | convergence null |
| Converges but not steerable / crosses failure boundaries undamaged | STALLED (needs new controller class) or BOUNDARY FOUND (clarifies it was a correlate) | **Proxy-Collapse confirmation** |
| Converges, steerable, boundary-coincident | OPERATING ENVELOPE / CONFIRMED (the specific traceability claim) | apparatus / B path — never "theorem" |

This table is the concrete "surfaces communicate" deliverable: any P2
result has one rail card *and* one prereg-outcome row *and*, where it
implicates Proxy Collapse, one mesa-surface note — same words, same
verdict.

## D. P1 admission verdict

**PASS, with the L5 re-scoping in §A and the two rulings in §B.**

- L1–L4 admitted as behavioral structural-failure-coincidence loci.
- L5 admitted as the evidence-admissibility rule governing L1–L4 scoring.
- L2/L4 operative boundaries fixed (`h≤32°`; supralateral permanent-fail).
- The falsifier is confirmed *informed by* the existing surface: it is
  the rigorous instrument for `debunked.md`'s most-important
  (`Proxy Collapse`) avenue, orthogonal to Pushable Occluder, and
  superset-adjacent to Score Aliasing. Prioritization and a shared
  verdict vocabulary are now fixed.

**Gate result:** P2 (agent run) is **unblocked in principle but not
started.** It remains gated on building the agent + matched-baseline
harness and scoring the four quantities separately (Admission Rule).
Public copy stays under the prereg Public-Language Constraint until
structural-failure-coincidence has actually passed in P2 — no theorem /
universal-proof language anywhere, including the rail.

---

## Amendments

Append-only. Each amendment: timestamp (date + zone), author, one-line
justification. The body above is the frozen P1 record.

*(no amendments yet)*
