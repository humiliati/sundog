# Phase 10 Attack Roadmap

> Consolidated post-audit attack plan for the Geometry Phase 10 single-handle
> verdict. The synthetic optical audit
> ([`docs/calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md))
> returned §3 outcome (3) — *parhelion-offset reinterpreted as
> photo-set-lucky / fit-driven* — plus partial hits on outcomes (2) and (4).
> Three load-bearing findings survived persona disagreement and the
> consolidator's verification gate. This roadmap turns those findings into
> a sequenced campaign of attack passes, each with entry/exit criteria, so
> the work can be re-audited cleanly before any real-specialist outreach.
>
> **Status as of filing (2026-05-13):** the Phase 10 single-handle verdict
> in [`docs/SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md#phase-10---rich-display-overlay-tuning)
> is *not falsified* but its evidentiary chain is *substantially weaker than
> the closeout claims*. Public-framing surfaces leaning on the verdict
> (`SUNDOG_V_GRAVITY.md`, the homepage `#elevator-pitch` section,
> `MESA_CROSSOVER_NOTE.md`) are hedge-required pending re-audit.

## 1. The Three Load-Bearing Findings

The synthetic memo's §2 lists fourteen verified items; three of them carry
the campaign and gate everything downstream. The full list is the
authoritative source — what follows is the campaign-level summary.

### Finding A — Atlas formula bug at `scripts/overlay_calibrate.py:381–384`

CZA apex is hardcoded as `sun_y − R46`. That expression is geometrically
correct only at h ≈ 22°. At p2 (h = 18.6°) the literature CZA position
sits ~19 px above the hardcoded prediction — which exactly matches the
−19.3 px residual the team filed as a *route-reliability* failure in the
Phase 10 closeout. Compounding bug: `WB_R46 = 440` versus the literature
ratio of 2.091 · `WB_R22` ≈ 460, so even at h = 22° the atlas's R46 in
pixel space is ~4 % too small.

**Net implication:** the CZA-route "failure" is partly atlas bug, not
route failure. The single-handle verdict rests on a *three-route
failure* count that includes a route the atlas is not testing fairly.
([Memo §2 items 1, 2, 3](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md#2-verified-findings-that-should-change-the-phase-10-verdict))

### Finding B — Parhelion-offset has a circular dependency on six of nine anchored photos

The inversion `h = arccos(R22 / offset)` has discrimination
`sec(h) − 1 = 0.71 %` at h = 6.83° and 5.52 % at h = 18.6°. On the five
low-h photos (p13, p20, p22, p25, p26) the geometric lever is at or
below typical anchor noise; on three of those (p20, p25, p26) the 22°
halo arc is not visible in the photograph at all, so R22 is not from a
ring fit. p27 admits the circularity in its own anchor JSON
(`offset_L = offset_R = R22 = 219` with `_meta.note` flagging it).
p26's right side encodes a literal geometric impossibility
(`R22 / offset = 323 / 322 > 1`, `arccos` undefined) that the residual
table accepted with no flag.

**Net implication:** the closeout's *"passes residual gate at ~0 px on
every eligible photo"* restates honestly as *"passes on three photos
(p2, p7, p13) where the test has meaningful discrimination and an
independently fittable 22° halo."* The verdict survives, but the
hedged version is much harder to attack.
([Memo §2 items 4–7](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md#2-verified-findings-that-should-change-the-phase-10-verdict))

### Finding C — Tangent-arc detection-gate is misclassified for at least one photo

p7 is at h = 59.4°. Per `atoptics.co.uk/blog/tangent-arcs/` and
`dewbow.co.uk/haloes/utan1.html`, above h ≈ 29° the upper and lower
tangent arcs join into the *circumscribed halo*. Testing column-peak
detection of an "upper tangent arc" at h = 59.4° is a literature-level
primitive misclassification, not a detection failure. Two of the other
three "detection-degenerate" calls (p2, p13) are protocol artifacts
under any literature-standard wing-based or Lab b\*-channel detector,
not feature-level negatives.

**Net implication:** the tangent-arc *route* may be recoverable on the
current photo set under a literature-standard detector. The current
column-peak result is a tooling-conditional negative, not a class-level
negative, and at least one photo (p7) does not belong in the tangent
eligibility set at all.
([Memo §2 items 9, 10, 11](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md#2-verified-findings-that-should-change-the-phase-10-verdict))

## 2. Cross-Cutting Downstream Impact

The Phase 10 single-handle verdict has been ratcheted into multiple
public-framing surfaces that now over-state what the audit will
support. Each surface needs an explicit hedge or retraction in lockstep
with the §3 attack passes.

| Surface | Current language | Required treatment |
| --- | --- | --- |
| [`docs/SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md#phase-10---rich-display-overlay-tuning) Phase 10 closeout | "passes residual gate at ~0 px on every eligible photo"; three-route failure by residual / coverage / detection | Replace per memo §4.1–4.3; add the audit memo as a substantiation pointer; flag the closeout as "post-audit hedged 2026-05-13" |
| [`docs/MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md#geometry-phase-10-closeout-2026-05-13) Phase 10 closeout subsection | "three independent failure layers"; forward-rich/inverse-narrow asymmetry | Walk back the "three independent" framing — the audit shows two of the three failures have non-atlas explanations; preserve the field-shape pattern claim with hedged wording per memo §6 |
| [`docs/SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md) public-framing sentence | "rich in forward generation … one image-recoverable inverse handle" | Hedge-required; lean retract-required for the *forward-richness* half until the parhelic-circle rule, the CZA position bug, and the infralateral / 46° halo visibility caveats land. Memo §6 has the defensible compressed framing. |
| `index.html#elevator-pitch` (homepage v1 pitch) | "first empirical hint that field-not-reward … is describing a real category of object" | Per [`SUNDOG_V_CHAT.md` §16.2](SUNDOG_V_CHAT.md#162-known-integrity-gap-v1-pitch-2026-05-13), this is now logged as audit-driven retraction (not "decide whether to ratchet a route"). Pitch text needs a hedge or paragraph pull until the re-audit lands. |
| `chat/claim_map.json` routes `mesa_roadmap_status`, `framework_pattern`, `unsupported_alignment_overclaim` | Currently treats mesa subspace and field-not-reward as roadmap / unsupported | Do not ratchet. The audit weakens — not strengthens — the case for promoting these routes. Hold at current tier until re-audit. |

## 3. Pass-by-Pass Attack Plan

Each finding gets a sequence of named passes. Passes within a finding are
ordered; passes across findings are independent unless an explicit
dependency is called out.

### Finding A — Pass A1: Formula fix (the smoking gun)

**Goal.** Land the literature CZA formula in the atlas so any rerun is
testing the route fairly. Single-edit fix.

**Touch.**

- `scripts/overlay_calibrate.py:381–384`: replace
  `anchored = WB_SUN[1] - WB_R46` with the literature formula
  `sun_y − (90° − h − arcsin(√(n² − cos²h))) × (R22/22°)` (n = 1.31).
- Same file: replace the `WB_R46 = 440` constant with
  `WB_R46 = round(2.091 * WB_R22)` (≈ 460), or — preferable — derive
  `R46_px = r22_obs × (46/22)` from `r22_obs` directly per overlay so
  the constant is no longer load-bearing.

**Entry criteria.** Audit memo accepted; this roadmap filed.

**Exit criteria.**

- Formula edit committed.
- The two affected pixel-position tests, if any, regenerated against
  the new formula and the diff inspected by hand on at least p2 and p27.
- Smoke run: on p2 the predicted CZA apex y collapses from ~19 px below
  the literature locus to within ~5 px of it (memo §2 predicts ~0.7 px;
  treat as illustrative — the *direction* is what matters).

**Out of scope for this pass.** Re-classifying p27's chromatic arc
(that is Pass A2). Re-running the residual gate (that is Pass A3).
Updating `MESA_CROSSOVER_NOTE.md` (that is the §2 cross-cutting work).

### Finding A — Pass A2: p27 primitive re-classification

**Goal.** Re-label the p27 chromatic arc currently anchored as CZA
apex. At h = 0.5° the literature CZA apex is ~57° above the sun and
off-frame; the visible arc at y = 142 sits at ~42° above the sun and is
the 46° halo top / supralateral merger.

**Touch.**

- `docs/calibration/anchors/p27-anchor.json`: re-anchor the chromatic
  arc as `supralateral / 46° halo top` rather than `cza_apex`. Preserve
  the existing CZA fields under a `_disputed` sub-key so the change is
  reversible.
- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: note the
  re-classification with the audit memo as the rationale pointer.

**Entry criteria.** Pass A1 landed (so the residual table is being
read from a fixed atlas).

**Exit criteria.**

- p27's CZA-apex residual is removed from the CZA-route eligibility
  count (it was a misidentified primitive, not a CZA measurement).
- p27 is added to the supralateral-route eligibility set as a
  *candidate* (still pending coverage check).
- The "merges with CZA" framing for p27 in the v3.8 receipt is
  retracted per memo §4.4.

**Out of scope.** Whether p27 plus p2 is enough to reopen the
supralateral coverage gate (Pass A3 question). Whether the
re-classified p27 changes the parhelion-route count (it does not —
parhelion is unaffected by CZA primitive ID).

### Finding A — Pass A3: CZA-route re-verdict

**Goal.** Re-run the CZA residual gate against the fixed formula and
re-classified p27. Decide whether the route is dead, alive, or pending
more anchored photos.

**Touch.**

- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: regenerate the
  CZA-route residual table with the new formula and the re-classified
  p27. If only p2 remains eligible, declare a coverage-gate failure
  honestly rather than residual-gate.
- Phase 10 closeout in `SUNDOG_V_GEOMETRY.md`: rewrite the CZA-route
  row of the table per memo §4.2.

**Entry criteria.** Passes A1 and A2 landed.

**Exit criteria.**

- CZA-route verdict is re-derived and re-stated in the doc with the
  failure layer named correctly (residual / coverage / detection).
- The "three independent failure layers" framing is either preserved
  (with the formula bug call-out) or retracted (if the new count is
  two-failures-plus-tooling-question).

### Finding B — Pass B1: Add lever and R22-source columns to the residual table

**Goal.** Make the parhelion-route eligibility honest at the schema
level, so future runs cannot accidentally promote a row that fails the
audit's two real eligibility tests.

**Touch.**

- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: add two columns to
  the parhelion-route residual table: `sec(h) − 1` (geometric lever)
  and `R22-source` (`ring-fit` / `parhelion-derived` /
  `inferred-other`).
- Anchor JSON schema (across `docs/calibration/anchors/*.json`): add an
  `r22_source` field declaring whether `R22` was independently fit
  from a visible ring or inferred. p20, p25, p26, p27 are
  `parhelion-derived` per the audit; verify the others by image
  inspection.
- Add a `geometric_validity` flag for rows where `R22 / offset > 1`
  (p26 right side is the known offender).

**Entry criteria.** None — this pass is independent of A1–A3 and can
run in parallel.

**Exit criteria.**

- Every anchor JSON has an `r22_source` field.
- The residual table has both new columns and the geometric-validity
  flag on p26 right.
- The audit memo is cited as the rationale in the doc header.

### Finding B — Pass B2: Parhelion-route re-verdict on the eligible subset

**Goal.** Re-derive the parhelion promotion verdict against the now-honest
eligibility set. The audit predicts the verdict survives but with
materially weaker language.

**Touch.**

- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: under "Phase 10
  Promotion Verdict", restate the parhelion-offset row per memo §4.1.
- `SUNDOG_V_GEOMETRY.md` Phase 10 closeout: replace the *"passes
  residual gate at ~0 px on every eligible photo"* table cell with the
  hedged language from memo §6.

**Entry criteria.** Pass B1 landed.

**Exit criteria.**

- The promotion verdict is restated with the eligibility hedge
  visible.
- The "every eligible photo" language is gone from the geometry doc,
  the crossover note, the gravity ledger, and the elevator pitch.
- If the promotion still survives on the eligible subset (the audit
  predicts it does), the verdict is preserved and the doc gets a
  "post-audit hedged 2026-05-13" stamp. If it does not survive (audit
  did not predict this, but it's a possible outcome), the verdict is
  retracted and Phase 10 reopens.

### Finding C — Pass C1: Drop p7 from the tangent-arc eligibility set

**Goal.** Remove the literature-level primitive misclassification.

**Touch.**

- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: drop p7 from the
  tangent-arc eligibility table; record the reason as
  *circumscribed-halo regime per atoptics.co.uk* with the citation.
- Open a separate placeholder for a *circumscribed-halo ellipticity →
  h* test if the team wants an inverse handle at high h. This is a
  Phase 10 backlog item, not a Phase 10 deliverable.

**Entry criteria.** None — independent of A and B.

**Exit criteria.**

- p7 is gone from the tangent-arc residual table.
- A note in the closeout flags that the original "detection-degenerate
  on all four eligible photos" count was on a list with a primitive
  misclassification at the top.

### Finding C — Pass C2: Wing-based / chromaticity-based tangent detector spike

**Goal.** Build the literature-standard detector and re-test p2 and
p13 (the two photos the audit flags as protocol-artifact failures
under column-peak). This is the "tooling vs. physics" question.

**Touch.**

- New detector code under `scripts/`: Lab b\*-channel ridge filter on a
  22°-halo-subtracted residual image, or wing-azimuth-offset radial
  profiles.
- `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`: append a "Pass C2
  results" subsection with the new detector's verdicts on p2, p13, and
  any other photos in the tangent eligibility set after Pass C1.

**Entry criteria.** Pass C1 landed.

**Exit criteria.**

- The wing-based detector exists, is committed, and has been run
  against the tangent eligibility set.
- The receipt language is updated per memo §4.8: cap the detection
  claim at *"column-peak-on-sun-meridian-in-intensity-image fails on
  all four photos; a literature-standard wing-based or Lab b\*-channel
  ridge detector recovers / does not recover the route on
  {photos}."* The route-level verdict follows the empirical result of
  the new detector.

### Finding C — Pass C3: Tangent-arc route re-verdict

**Goal.** Decide whether the tangent-arc route is *class-level
negative*, *tooling-conditional negative*, or *recovered* after Pass
C2.

**Touch.**

- `SUNDOG_V_GEOMETRY.md` Phase 10 closeout: rewrite the tangent-arc
  row of the table to reflect the C2 outcome.
- If the route is recovered, the single-handle verdict becomes a
  two-handle verdict and the *forward-rich / inverse-narrow* framing
  needs a corresponding edit in `MESA_CROSSOVER_NOTE.md` and
  `SUNDOG_V_GRAVITY.md`.

**Entry criteria.** Pass C2 landed.

**Exit criteria.** Tangent-arc verdict is one of the three named
outcomes above, with the supporting evidence in the residual table
and the language ratcheted in (or out of) the public-framing
surfaces accordingly.

## 4. Sequencing and Dependencies

The passes form three independent threads that converge at the
re-audit gate. Within each thread the order is fixed; across threads
the work can run in parallel.

```
Thread A (CZA + atlas):  A1 → A2 → A3
Thread B (parhelion):    B1 → B2
Thread C (tangent):      C1 → C2 → C3
                         ↓    ↓    ↓
                         Re-audit gate (§5)
                         ↓
                         Real-specialist handoff
                         ↓
                         Public-framing ratchet (§2)
```

Cross-thread coupling worth naming explicitly:

- **B1 has no upstream dependency** but its output (the
  geometric-validity flag, the lever column) makes A3 and C3 easier
  to write honestly. Recommend running B1 first if there is any pass
  contention.
- **A2 may unlock supralateral**. If the re-classified p27 plus p2
  gives two eligible supralateral measurements, the supralateral
  coverage gate may reopen — though memo §2 item 12 notes the route
  has structurally weak h-discrimination (~0.5° change across h =
  0–22°) and is unlikely to be a useful inverse handle in principle.
  Treat any reopening as a Phase 10 backlog item, not a blocker.
- **None of the passes block the public-framing hedges** in §2.
  Those should land as soon as this roadmap is filed, with the
  hedged language carrying a "pending re-audit" stamp. Waiting for
  the re-audit to land before hedging the public copy keeps a known
  integrity gap open longer than necessary.

## 5. Re-Audit Gate (Before Real-Specialist Outreach)

Per memo §8: do not send the existing handoff to a real specialist
before the §3 passes land. The smoking-gun formula bug at
`overlay_calibrate.py:381` is a fifteen-minute find for any halo
specialist; if they land on it before the team does, the rest of the
audit credibility takes damage.

**Re-audit entry criteria.** All passes labelled "Required for
re-audit" below are committed:

- **A1** (formula fix) — required.
- **A2** (p27 re-classification) — required.
- **A3** (CZA-route re-verdict) — required.
- **B1** (lever and R22-source columns) — required.
- **B2** (parhelion-route re-verdict) — required.
- **C1** (drop p7 from tangent eligibility) — required.
- **C2** (new tangent detector) — *recommended but not required*. If
  C2 has not landed, the re-audit explicitly lists tangent-detector
  rebuild as Open Question for the specialist, per memo §4.8.
- **C3** (tangent-arc re-verdict) — gated on C2; same disposition.

**Re-audit shape.** Re-run the synthetic three-persona protocol
against the post-pass state of the docs and code. The consolidator's
verification gate is mandatory; treat any persona over-statement as
disqualifying for that finding (see [§7](#7-meta-treat-the-verify-gate-as-load-bearing)
on the verify gate).

**Specialist handoff entry criteria.** Re-audit completes with no new
load-bearing findings, or with new findings that are bounded to known
open questions in `PHASE10_OPTICAL_AUDIT_HANDOFF.md`. The handoff doc
itself is updated to reflect the post-audit state of the verdict
language.

## 6. Public-Framing Hedge Sequencing

Independent of the §3 passes, the surfaces in §2 should be hedged
*now* and re-ratcheted (or further retracted) after the re-audit. The
order minimizes window-of-inconsistency between coupled surfaces.

1. **`docs/SUNDOG_V_GEOMETRY.md` Phase 10 closeout.** Add a
   "Post-audit hedge 2026-05-13" sub-block at the top of the closeout
   that names the audit memo, the three load-bearing findings, and
   the fact that the closeout language below is pending re-derivation
   per this roadmap.
2. **`docs/MESA_CROSSOVER_NOTE.md` Phase 10 closeout subsection.** Add
   the same hedge stamp; walk back "three independent failure
   layers" to the audit-survived language ("three structurally
   different failure layers, two of which have non-atlas
   explanations").
3. **`docs/SUNDOG_V_GRAVITY.md` public-framing sentence.** Hedge the
   forward-richness half per memo §6.
4. **`index.html#elevator-pitch` v1 pitch.** Either pull the
   third paragraph (the "first empirical hint" sentence) until the
   re-audit, or add a hedging clause that names the audit-driven
   retraction. Bump `data-version` to v1.1, `data-revised` to today,
   and update the visible stamp.
5. **`docs/SUNDOG_V_CHAT.md` §16.2.** Update the third bullet
   (mesa↔geometry crossover) to flip the resolution path from
   "decide whether to ratchet a route" to "audit-driven retraction
   pending re-audit; do not ratchet `claim_map.json` routes."

## 7. Meta: Treat the Verify Gate as Load-Bearing

The synthetic audit's consolidator verification step caught one
Persona 3 over-statement (the p22 halo claim) and several uncited
Persona invocations. Both would have leaked into the handoff if the
audit had been sent without the verify gate. That is not a "courtesy
step"; it is the load-bearing one.

Going forward, the protocol is:

- A persona-style audit produces draft findings.
- The consolidator independently re-verifies each load-bearing
  finding against source files, photographs, and atmospheric-optics
  literature.
- Findings that fail verification go into the consolidated memo's
  *§3 Unverified / contradicted* bin, not the §2 verified bin.
- The consolidated memo is the input to action; the persona drafts
  are appendices.
- The same protocol applies to the post-pass re-audit before
  specialist handoff.

## 8. References

- [`docs/calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md)
  — the verified findings, the §3 dropped claims, the §4 doc-change
  recommendations, the §5 protocol fixes, the §6 public-framing
  treatment, the §7 persona disagreements, and the §8 confidence on
  next-step. **This roadmap is the action plan for that memo's §4 and
  §5; the memo is authoritative on findings.**
- [`docs/calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-1-tangent-arc.md`](calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-1-tangent-arc.md)
- [`docs/calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-2-cza-supralateral.md`](calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-2-cza-supralateral.md)
- [`docs/calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-3-parhelion-forward.md`](calibration/PHASE10_OPTICAL_AUDIT_DRAFTS/persona-3-parhelion-forward.md)
- [`docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md`](calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md)
  — the existing handoff doc; do not send to a specialist before the
  §5 re-audit gate.
- [`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md)
  — the canonical residual tables and per-photo notes; primary edit
  surface for Passes A3, B1, B2, C1, C2, C3.
- [`docs/SUNDOG_V_GEOMETRY.md` Phase 10](SUNDOG_V_GEOMETRY.md#phase-10---rich-display-overlay-tuning)
  — the Phase 10 section; receives the post-audit hedge per §6.
- [`docs/SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md) — mesa roadmap; the
  field-not-reward thesis it stages now depends on this roadmap's
  outcome via the crossover note.
- [`docs/MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md) — the
  crossover note; receives the hedge per §6.
- [`docs/SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md) — gravity ledger;
  the public-framing sentence is hedge-required.
- [`docs/SUNDOG_V_CHAT.md` §16](SUNDOG_V_CHAT.md#16-coupled-public-copy-surfaces-integrity-coordination)
  — coupled-public-copy integrity protocol; §16.2 third item is
  updated by §6 of this roadmap.
- `scripts/overlay_calibrate.py:381–384` — the load-bearing edit for
  Pass A1.
