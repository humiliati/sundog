# Chat-v2 H1/V3-0b — Ambiguity-Slice Admission Receipt

> 2026-07-01. Run of `H1_V3_0B_AMBIGUITY_SLICE_SPEC.md` (chess arm; CPU, no model).
> **Non-promotional. No model run, no GPU, no R2 claim.**
> Script: `scripts/chatv2_h1_v3_0b_slice_admission.py`; log + witness JSONL in
> `results/chatv2/h1_v3/`.

## Verdict: `F3-V3b/input` — the strong negative, as pre-registered

**1 / 29 axes admitted (24 needed); kill causes = {probe: 28}.** The surface suite still
reads 28 of 29 axes *on their ambiguity slices*. Exactly the branch the spec named the
strong negative: **ambiguous fibers are not uniform — natural move statistics pin state
even where the bag formally underdetermines it.**

## The run

- **Marker scan (floor-count only):** ply 24/32/40 → 300/342/**366** axes meeting the
  120-instance slice floor → **ply 40 chosen**, 2,113 instances, co-ambiguity median 22 /
  p75 24 (matches the feasibility scan).
- **Balance on the slice:** 366 floor axes → **29 balanced** — the conditional
  distributions are themselves skewed (squares that change ≥2× still mostly end one way),
  so bank formation barely cleared 24. Cap 48 not binding.
- **Liveness:** `e2e4`-present read at **1.000** on slice instances — apparatus live.
- **Probes (UCI, registered):** slice probe-max range **0.59–0.90, bulk 0.61–0.72** —
  versus **0.94–0.996** on the whole distribution (V3-0). The slice removed *most* of the
  readability; the residual still exceeds the frozen 0.60 ceiling for 28/29 axes. Top
  reader is almost always LR-counts on the UCI bag. The two outliers at ~0.89 (`occ.f1`,
  `w.f1`) are castling-square axes — `e1g1` is itself a bag token. The one admit:
  **`b.a7`** (rook-pawn square, probe-max 0.590, 1 witness pair).
- **Leak controls (first 10 axes, SAN/sanitized):** 0.58–0.85 — comparable to UCI on
  slices; no condition is materially cleaner here.
- **Witness search — disclosure:** 5/29 axes covered (11 pairs), using **41 s of the 600 s
  budget** — an implementation shortfall (single pass over instances rather than
  budget-exhausting loop), plus a substantive finding: **legal adjacent same-color swaps
  mostly commute** (that is *why* they are legal), so undirected local search rarely flips
  the final board — instance-level witness coverage on a 200-sample: **0/200**. This did
  **not** affect the branch: every kill was a probe kill (no axis with probe-max ≤ 0.60
  lacked a witness). A budget-honest, interaction-directed search would raise coverage but
  cannot change an `/input` verdict.

## Reading — what the strong negative establishes

1. **The diagnosis was right and the redesign bit hard:** conditioning on the ambiguity
   slice collapsed surface readability from ~0.95+ to ~0.65. The V3-0 failure really was
   distributional redundancy, not a broken gate.
2. **But ambiguity ≠ unpredictability.** On formally-undetermined fibers, chess move
   statistics still recover state at 0.61–0.72. The fiber is non-degenerate (witness pairs
   exist) yet far from uniform. The absolute-undecodability gate (≤ 0.60 on natural data,
   bank scale) has now failed **six families**: count-parity, agreement, relations,
   code-vars, chess-board (whole distribution), chess-board (ambiguity slice).
3. **The through-line the lane should not lose:** H2 — the lane's one real positive —
   *also* did not clear an absolute ceiling: its hard-slice count baseline was **0.770**.
   H2 cleared a **relative margin** (residual 0.931 ≫ counts 0.770). R1's own licensed
   metric was relative too (`objective_excess` = gen − twin). The absolute
   input-undecodability requirement appears **unattainable at bank scale on natural
   data**; every positive this lane has ever produced was a *crossover* (model ≫ surface
   at matched evaluation), not an absolute floor.

## Disposition

Per the spec: **no rescue; V3-0.5 and V3-1 do not run; no GPU/H200;
`PROMOTE_GATE.md` unchanged.** Onward options recorded at filing:

1. **Bank R1 and freeze** — now strongly indicated. The R2 boundary is measured at six
   families, three levels, two slice designs, with a machine-checked theory anchor
   (`SurfaceBag`); the honest closing statement writes itself.
2. **A crossover-form V3 prereg** (explicitly a *gate redefinition*, not a rescue of this
   one): replace absolute surface-undecodability with the relative criterion the lane's
   positives actually satisfied — model-carry ≥ surface + margin (e.g., +0.15), evaluated
   per axis on its slice, with the surface probe suite as the matched baseline. Grounded in
   R1's `objective_excess` precedent and H2's crossover. Would need owner sign-off on the
   claim language ("the model reads state *better than* the surface allows," not "the
   state is surface-invisible") and a fresh prereg before any model run.
3. High-entropy synthetic-adjacent corpora (shuffled game records etc.) stay parked —
   they would surrender the "real task" half of the R2 claim.

**Owner decision after this receipt:** option 2 is taken. The fresh prereg is
`H1_V3_0C_CROSSOVER_SPEC.md`, with margins frozen at +0.15 over `surface_max` and
+0.15 over the same-architecture random-init floor, validation-only layer choice,
and claim language locked to "reads state better than the registered surface
statistic allows."

Cross-refs: `H1_V3_0B_AMBIGUITY_SLICE_SPEC.md` (the prereg), `H1_V3_0_DATA_ADMISSION_RECEIPT.md`
(the diagnosis this tested), `H1_V3_0C_CROSSOVER_SPEC.md` (the owner-approved crossover
redefinition), `R2_INTERSECTION_HYPOTHESES.md` (H2 crossover; H5 σ-bridge), `PROMOTE_GATE.md`
(unchanged).
