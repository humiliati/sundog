# Structural Failure Boundary Map (P0 deliverable)

Pre-registration: [`README.md`](README.md) (frozen 2026-05-15 PT)
Roadmap: [`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) ▸ Candidate 13 ▸ Roadmap
Frozen: **2026-05-15 (PT)**. Append-only below the **Amendments** rule;
the body above it is the P0 artifact and is frozen at write time.

This is the P0 deliverable the prereg's *First Falsifier Before Agents*
section requires to exist before any agent is admitted. It is the
falsifier: a frozen map of where the documented closed-form inverse is
ill-posed, what a *traceable* agent must do there, and what a *mere
correlate* would do instead — written entirely from existing geometry
receipts, **no agent, no render, no training**.

Each locus carries the five fields the prereg mandates:
**eligible regime · abstain/switch/fail regime · exact source receipt ·
traceable-agent prediction · mere-correlate prediction.** A row with no
crisply-citable receipt is marked **BLOCKED** and halts the program
(fix = more geometry specification, not more training).

Reconciliation policy (prereg P1): where the project's *coded guard*,
the *literature value*, and the *prereg-stated* number differ, the
**coded guard is the operative boundary** (it is what any harness will
actually enforce); the others are recorded for honesty, not silently
averaged.

---

## L1 — Parhelion offset route (the sole promoted inverse handle)

- **Inverse.** `offset = R22 / cos(h)` — `phase3.daggerOffset(h)` in
  `public/js/parhelion-geometry.mjs` (≈ line 1060).
- **Eligible regime.** The strict eligible photo set **p2, p7, p13**
  only — the sole promoted hidden-state route after the Phase 10 audit.
- **Abstain / fail regime.**
  - Low-h, low-leverage: `sec(h) − 1` below ~2% of R22 ⇒
    anchor-noise-bounded, not an independent handle.
  - Parhelion-derived-R22 photos ⇒ **tautological** (R22 came from the
    parhelion; recovering h from it is circular), not independent
    evidence.
  - p26 right side ⇒ **geometrically invalid**, excluded.
- **Exact source receipt.**
  `docs/calibration/HALO_PHENOMENA_ACCOUNTING.md` §A "Sundog / parhelion"
  (sole promoted inverse handle; strict eligible set p2/p7/p13);
  `docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md` (the optical audit
  that promoted it and bounded eligibility);
  `public/js/parhelion-geometry.mjs` `phase3.daggerOffset`.
- **Traceable-agent prediction.** Succeeds on the strict eligible set;
  reports *low leverage* or *ineligible* on the low-h / tautological /
  invalid rows and does **not** count them as independent inverse
  evidence.
- **Mere-correlate prediction.** Emits smooth, confident `h` estimates
  across low-leverage, tautological, and invalid rows because image
  style, crop, metadata, or halo prominence co-varies with `h`.

## L2 — CZA visibility cutoff

- **Inverse.** Circumzenithal-arc route, gated by
  `czaVisibleAtAltitude(h)` in `public/js/parhelion-geometry.mjs`
  (≈ lines 965–968).
- **Eligible regime.** Sun altitude **h ≤ 32°** (the coded guard:
  `czaVisibleAtAltitude` returns true only for `altitudeDeg ≤ 32`).
- **Abstain / switch / fail regime.** Above the cutoff the CZA exits the
  visible hemisphere; a CZA-dependent route must fail, abstain, or switch
  handles.
- **Exact source receipt.** **Coded guard (operative):**
  `public/js/parhelion-geometry.mjs` `czaVisibleAtAltitude` ⇒ `h ≤ 32°`.
  **Literature (recorded, not operative):** Tape *Atmospheric Halos*
  Ch. 6 p63 "about 32°"; the earlier accounting matrix cited the
  closed-form cutoff h = 32.196°; the prereg states "about 32.2 deg".
  Operative boundary = **32°** per the coded guard; the ~0.2° spread
  between coded/literature is itself inside the tolerance band and is
  recorded here rather than averaged.
- **Traceable-agent prediction.** A CZA-dependent route fails / abstains
  / switches at the cutoff; it does **not** preserve a CZA-apex inverse
  past disappearance.
- **Mere-correlate prediction.** Continues to report altitude through
  the cutoff because other image features still carry distributional
  information about `h`.

## L3 — Tangent arc → circumscribed-halo merge

- **Inverse.** `tangentArcLocus(h)` upper-tangent handle in
  `public/js/parhelion-geometry.mjs`.
- **Eligible regime.** **h < 29°** (separate upper-tangent handle
  exists).
- **Abstain / switch / fail regime.** **h ≥ 29°**: `tangentArcLocus`
  returns `null` — the upper/lower tangent arcs have merged into the
  circumscribed-halo regime; no separate tangent-curvature handle.
- **Exact source receipt.** `public/js/parhelion-geometry.mjs`
  `const TANGENT_ARC_CIRCUMSCRIBED_H = 29` (≈ line 813) and the
  `tangentArcLocus` `null`-return guard (≈ line 830); cross-validated by
  `docs/calibration/PASS_C7_OUTPUT.txt` and Tape *Atmospheric Halos*
  Ch. 6 p62 ("at a sun elevation of 29° the two halos merge … the value
  29° is theoretical"). Coded boundary and literature agree at 29° here.
- **Traceable-agent prediction.** A tangent-dependent route degrades,
  abstains, or switches at the merge; it does **not** claim continuous
  tangent-curvature recovery through the singularity.
- **Mere-correlate prediction.** Maintains a continuous tangent-like
  estimate through the merge — a learned texture/shape correlate, not
  the canonical tangent handle.

## L4 — Supralateral route (structural-discrimination gate)

- **Inverse.** Supralateral angular position → `h` (a *candidate* handle
  that the audit rejected).
- **Eligible regime.** **None.** This row is a permanent *fail* row, not
  a windowed one: the handle fails the structural-discrimination gate at
  every altitude.
- **Abstain / fail regime.** All `h`. Receipt figure: supralateral
  angular distance from the sun varies only **~0.5° across h = 0–22°**,
  below the typical ~5–10 px visual-edge measurement noise even at
  perfect coverage.
- **Exact source receipt.**
  `docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md` line 54
  ("Supralateral position → h | **fails structural-discrimination gate**
  | … varies only ~0.5° across h = 0–22°, below the typical 5–10 px
  visual-edge measurement noise"). The prereg's "~0.3 deg" is the
  narrower tested-low-altitude-span figure; the operative receipt value
  is **~0.5° over h = 0–22°** per the handoff. Either way the row is a
  hard fail.
- **Traceable-agent prediction.** Refuses to promote supralateral
  position as a useful inverse handle under the documented apparatus.
- **Mere-correlate prediction.** Treats supralateral brightness, crop
  position, or co-occurring arcs as a usable altitude channel.

## L5 — Rendered ≠ anchored

- **Inverse.** None directly — this is the *evidence-admissibility*
  boundary that governs every other row.
- **Eligible regime.** Only **anchored closed-form** rows (the §A core
  with a project receipt) may count as inverse evidence.
- **Abstain / fail regime.** `rendered-optional`, `named-only`,
  `not-modeled`, and the hardcoded atlas-placeholder primitives
  (supralateral / suncave-Parry / Parry-supralateral / infralateral
  draw code, the parhelic-circle empirical smile, the sun-pillar vesica)
  may support display / vocabulary / future hypotheses but **never**
  traceability.
- **Exact source receipt.**
  `docs/calibration/HALO_PHENOMENA_ACCOUNTING.md` §A "Honesty ratchet —
  rendered ≠ anchored", plus the Status Vocabulary table.
- **Traceable-agent prediction.** Counts only anchored closed-form rows
  as inverse evidence; treats drawn-but-unanchored primitives as
  non-evidence.
- **Mere-correlate prediction.** Uses the presence of any drawn or named
  primitive as evidence that an inverse is available.

---

## P0 completion gate

Every locus L1–L5 is written crisply from a **named, verified source
receipt** (file + section/line checked 2026-05-15). No row is BLOCKED.
Two coded-vs-stated reconciliations are recorded, not hidden: L2 CZA
(coded `h ≤ 32°` operative; literature/prereg ~32.2°/32.196°) and L4
supralateral (receipt ~0.5°/0–22°; prereg ~0.3° narrower span). Neither
changes a regime classification.

**Verdict: P0 gate PASSES — the boundary map is freezable.** The program
is admitted to **P1 (falsifier admission review)**. This is a genuine,
non-trivial outcome and the payoff of the Phase 14/15 receipt discipline:
the inverse's singularities were already documented precisely enough that
the falsifier could be written *before* any agent — exactly the condition
the prereg required, and the condition that distinguishes a testable
apparatus from a theorem posture.

P2 (agent run) remains **blocked** until P1 signs off. No public copy may
use theorem / universal-proof language; the prereg Public-Language
Constraint is in force.

---

## Amendments

Append-only. Each amendment: timestamp (date + zone), author, one-line
justification. The body above is frozen at P0 write time.

**2026-05-15 (PT) — maintainer.** Context integration: cross-reference against
[`docs/Debunked.md`](../../Debunked.md) completed. The BOUNDARY_MAP path
(halo-geometry altitude inference, traceability test) is **orthogonal** to the
Pushable Occluder path documented in docs/Debunked.md (photometric controller,
PATH domain). No locus in L1–L5 depends on any assumption challenged or
invalidated in Debunked.md; the two documents test different failure modes in
different experimental domains. Full cross-reference analysis, unresolved-risk
inventory, and prioritization recommendation:
[`DEBUNKED_CROSSREF.md`](DEBUNKED_CROSSREF.md).
Verdict: treat this path as **primary** — advance to P1 (falsifier admission
review) using DEBUNKED_CROSSREF.md as the context-integration step.
