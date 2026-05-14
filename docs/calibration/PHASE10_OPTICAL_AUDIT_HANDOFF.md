# Phase 10 Optical Specialist Audit Handoff

> ## ⛔ DO NOT SEND — SUPERSEDED BY SYNTHETIC AUDIT (2026-05-13)
>
> **Status:** This handoff document is **stale**. It still presents the
> original Phase 10 single-handle verdict as send-ready, but a synthetic
> three-persona optical audit
> ([`PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md))
> returned three load-bearing findings that materially weaken the verdict's
> evidentiary chain:
>
> 1. **Atlas formula bug at `scripts/overlay_calibrate.py:381`** — CZA
>    apex hardcoded as `sun_y − R46`, geometrically correct only at
>    h ≈ 22°. The CZA-route "residual gate" failure listed in §1's table
>    is partly an artifact of this bug, not a route failure. A real
>    halo specialist will find this within fifteen minutes of opening
>    the atlas; sending this handoff before the fix lands forfeits
>    credibility on the rest of the audit ask.
> 2. **Parhelion-offset circular dependency on six of nine anchored
>    photos** — `sec(h) − 1` lever is below 2 % of `R22` on five low-h
>    photos, and on three of those (p20, p25, p26) the 22° halo arc is
>    not visible in the photograph at all so `R22` is parhelion-derived,
>    not ring-fit. The "passes residual gate at ~0 px on every eligible
>    photo" claim restates honestly as "passes on three photos (p2, p7,
>    p13) where the test has meaningful discrimination."
> 3. **Tangent-arc detection-gate misclassification at p7** — h = 59.4°
>    is the circumscribed-halo regime per atoptics.co.uk and
>    dewbow.co.uk; testing column-peak detection of an "upper tangent
>    arc" at this h is a literature-level primitive misclassification.
>
> **Action:** the campaign to address these findings is filed in
> [`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md). The
> roadmap's re-audit gate (§5) lists which passes must land before any
> handoff (this one or its successor) goes to a real specialist. This
> document will be rewritten — verdict table, audit-ask language, and
> §3 "what would change the verdict" all included — once the re-audit
> clears.
>
> **Until then: do not send this document, in whole or in part, to an
> external optical specialist.** The verdict table in §1 below, the
> framing in §2, and the deliverable shape in §5 all reflect the
> pre-audit state of the Phase 10 closeout and will mislead.

> **Audit ask in one line:** before we bake the Phase 10 "single image-recoverable inverse handle" verdict into program-level public framing, we want an atmospheric-optics specialist to push on whether the verdict is structurally sound or an artifact of dataset / detection choices.

## 1. Why we're asking

The Sundog program has a halo-overlay atlas calibrated to a small set of photographs of parhelion / 22° halo / 46° halo / CZA / tangent / supralateral / parhelic-circle structure. The atlas claims that all primitives derive forward from a single parameter `h` (sun altitude). Phase 10 of the geometry roadmap is the empirical test of whether the atlas inverts — i.e., whether `signature → h` is recoverable from photographs without the per-photo hand-anchor that currently bootstraps each fit.

We tested four candidate inverse routes against a calibration set of ~7 photos (plus ~13 triage candidates not all anchored). The verdict landed 2026-05-13:

| route | gate outcome | failure layer |
| --- | --- | --- |
| Parhelion offset → h | **promoted** (calibrated core) | none — passes residual gate at ~0 px on every eligible photo |
| CZA apex → h | fails | residual gate (p2 y = −19.3 px, p27 y = +21 px; both exceed `0.04 * R22`, opposite signs) |
| Supralateral position → h | fails | coverage gate (only p2 eligible on the committed set; p27 candidate but unmeasured) |
| Tangent-arc curvature → h | fails | detection gate (column-peak protocol fails on all four eligible photos with three distinct per-altitude degeneracy modes) |

The single-handle finding is now the load-bearing atlas-side receipt for the program's public framing in
[`../SUNDOG_V_GRAVITY.md`](../SUNDOG_V_GRAVITY.md): **the atlas is rich in forward generation (`h → all primitives`) and supports one image-recoverable inverse handle (parhelion offset).** This claim is parallel to a mesa-side in-vitro result and we want it audited before it goes further into public-facing materials.

## 2. What we want pushed on

We are not asking for validation. We want pushback at any of the following layers:

### 2.1 Are these the right four routes?

We tested parhelion-offset, CZA-apex, supralateral-position, and tangent-arc curvature. Atmospheric-optics literature lists more candidate features:

- 46° halo radius vs 22° halo radius — currently both treated as anchor-derived, not inverse routes
- Parhelic-circle visible extent — currently treated as visibility primitive only
- Sun pillar length (when present)
- Helic / sub-parhelic features
- Lower tangent arc curvature (mirror of upper-tangent test)
- Polarization signatures (out of scope of our photo-only protocol)

Question: are there `signature → h` inverse routes we are not even attempting that we should be? Conversely, is one of the four we tested not actually expected to invert cleanly under any protocol (i.e., were we testing a route that atmospheric optics knows is degenerate by construction)?

### 2.2 Are the documented failure modes interpreted correctly?

For each non-promoted route we recorded a specific failure-mode reading. We want each one stress-tested:

- **CZA-apex opposite-sign residuals (p2 −19.3 px, p27 +21 px).** Our reading: route-reliability gap, not stable atlas bias. Alternative readings worth surfacing: (i) one of the two photos has anchor error; (ii) CZA apex has known photo-environment sensitivity (humidity, ice-crystal habit, viewing-azimuth offset) that our model ignores; (iii) the +21 / −19 split is within expected real-atmosphere variability and shouldn't count as a failure.
- **Tangent-curvature detection-degeneracy across four photos** — three distinct failure modes by altitude (compression at h = 18.6°, halo-edge dominance at h = 59.4°, chromatic-haze at h = 6.83°, CZA / sun-bloom merge at h = 0.5°). Our reading: column-peak detection is the wrong instrument; the arc is real and visible but its brightness/chromatic spine is not a per-column peak. Alternative readings: (i) the upper tangent arc as a feature is genuinely chromatic-only, not a detectable curvature peak, so any inverse route on it is ill-posed; (ii) some altitudes (e.g., near-horizon) really do swap upper-tangent for lower-tangent and the route is regime-bounded; (iii) the failure is detector-fixable.
- **Supralateral coverage gate.** Only p2 of seven photos has supralateral visibility under our criteria. Question: is supralateral visibility intrinsically rare (selection of cold-cirrus + appropriate-h photos), making this route never reach coverage in any realistic dataset, or is our calibration set just unlucky?

### 2.3 Anchor-capture protocol soundness

Our per-photo anchor protocol records:

- `sun_px` via saturation-centroid + warm-index + crosshair-crop confirmation
- `R22_px` via saturation-ring fit (not visual ring guess)
- `parhelion_left_px` / `parhelion_right_px` via peak detection
- `parhelic_belt_y_px` via belt-region inspection
- CZA apex via observed-apex sampling (only on eligible photos)

Questions: (i) do these protocols match atmospheric-optics literature norms for halo geometry measurement, (ii) are there well-established alternative methods that would reduce anchor noise materially, (iii) would controlled-optics conditions (calibrated camera, known lens distortion, flat-field correction) change which routes pass the residual gate?

The third question is currently filed as Phase 10 Open Question #3 (lens-optics test). The audit's view on whether that question matters or is a distraction is itself useful.

### 2.4 Forward-generation claim

We claim `h → all primitives` runs cleanly forward through the atlas: parhelion offset, 22° halo radius, 46° halo radius, CZA visibility window, tangent positions, supralateral position, parhelic circle, infralateral arcs. The forward direction is not under audit in the same way the inverse direction is — but we want a sanity check that we are not over-claiming forward richness either.

### 2.5 The three-gate taxonomy

We introduced a residual / coverage / detection gate vocabulary to distinguish three structurally different failure modes. The taxonomy is now also being used on the mesa side (see
[`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md) Geometry Phase 10 closeout section).

Question: does this match standard atmospheric-optics measurement-stack vocabulary, or are we reinventing a taxonomy that already exists under different names? If the field already has language for these distinctions, we should adopt it.

### 2.6 The public-framing claim

The public framing claim derived from Phase 10 is:

> The atlas is rich in forward generation (`h → all primitives`) and supports one image-recoverable inverse handle (parhelion offset). The other three candidate inverse routes fail at three structurally different layers of the measurement stack.

Is this defensible to atmospheric-optics readers? Where it isn't, what would the right hedged phrasing be?

## 3. What would change the verdict

We are explicitly trying to surface counterexamples. The audit is most useful if it ends in any of:

1. **A fifth route we did not consider** that does invert cleanly on the current photo set. (Verdict shifts from "one handle" to "two-or-more handles.")
2. **A reinterpretation of one of the three failed routes** that flips it to "would work under protocol X." (Verdict shifts from "single handle" to "single handle plus N protocol-conditional handles.")
3. **A reinterpretation of the parhelion-offset success** as photo-set-lucky rather than structurally clean. (Verdict shifts from "single handle survives" to "no handle survives a fair audit.")
4. **A pointer to literature that says forward generation is itself overclaimed** in some specific primitive class.
5. **No counterexamples found; the verdict is structurally sound under known atmospheric-optics frameworks.** (Verdict baked.)

Any of the five outcomes is useful. We want the audit to be permitted to be brutal.

## 4. Materials provided

### 4.1 Photo set and overlays

`docs/calibration/` and `docs/calibration/overlays/` contain:

- The full image set (p1 through p31; not all anchored — anchored set is p2, p7, p13, p18, p19 (cropped CZA receipt), p20 (cropped CZA receipt), p22, p25, p26, p27, p30; the FF run added p22 / p25 / p26 / p30 on top of the original p2 / p7 / p13 / p27 set).
- Rich-vocabulary overlays for p2 / p7 / p13 (`p2.rich-vocabulary.overlay.png` etc.).
- Per-photo anchor JSONs at `p<NN>-anchor.json` for committed anchors.

### 4.2 Canonical project docs (read in this order)

1. **`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`** — Phase 10 source-of-truth: anchor summary, Per-Inversion-Route Residual Table, Phase 10 Promotion Verdict, Single-handle closeout subsection, Tangent-Curvature v3.8 Receipt. This is the document the audit should treat as authoritative for Phase 10 state.
2. **`docs/calibration/PHASE10_EXPANSION_TRIAGE.md`** — candidate photo inventory and triage calls (which photos are CZA-eligible, which are cropped, etc.).
3. **`docs/SUNDOG_V_GEOMETRY.md`** Phase 10 + Phase 11 sections — overall geometry roadmap, Phase 10 status block carries the single-handle verdict.
4. **`docs/MESA_CROSSOVER_NOTE.md`** — Geometry Phase 10 closeout section + three-gate taxonomy. Explains how the atlas-side result is being read against the mesa-side in-vitro result.
5. **`docs/SUNDOG_V_GRAVITY.md`** Forward/inverse asymmetry receipt subsection — the public-framing language under audit.
6. **`docs/calibration/PHASE10_BELT_Y_SPEC.md`** + **`docs/calibration/PHASE10_BELT_Y_RESULTS.md`** — Open Question #1 closed-out cycle (parhelic-belt-y replication; FF1 falsified, photo-specific, no primitive change). Useful as an example of the pre-registration / decision-rule discipline applied to a sub-question. Not load-bearing for the audit's main question.

### 4.3 What the audit can ignore

The mesa-side documents (`SUNDOG_V_MESA.md`, `PHASE6_V*_RESULTS.md`) are not under audit. They are a parallel in-vitro result that the atlas-side single-handle finding is being compared against in the public-framing language. The audit does not need to evaluate the mesa-side; it only needs to flag if the comparison is being overclaimed in the gravity-language doc.

## 5. Deliverable shape

A short memo (2-4 pages) addressing the questions in §2 with verdicts of the form:

- "Sound" / "Sound with caveat: ..." / "Pushback: ..." / "Out of my area"

Plus, where the answer is pushback, a pointer to literature or a counterproposal we can test.

We do not need a paper or a comprehensive review. We need the kind of memo a domain specialist would send a non-specialist team that has done careful work in a neighboring area and is about to publish. Brutally short is fine; brutally honest is better.

## 6. Out of scope for this audit

- Mesa-side in-vitro results (`SUNDOG_V_MESA.md`)
- The Sundog brand / UI / app surfaces
- The perception roadmap (`SUNDOG_V_PERCEPTION.md`) — that's a separate downstream roadmap, not part of the Phase 10 verdict
- Detailed redesign of the anchor-capture pipeline — pointers to better methods are useful; we will implement them ourselves
- Implementation of any proposed alternative detector for the tangent-arc-curvature route — pointing us at the right method is enough

## 7. Contact for clarifications

This document and the canonical docs in §4.2 are intended to be self-contained, but the program team is available for clarifying questions. Address questions to the geometry-side lead by including a pointer to the section of this handoff or the linked canonical docs.
