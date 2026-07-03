# AT-1 — Palinstrophy Boundary-Layer Receipt

> 2026-07-02. Rung 1 = owner-run `lock_disc_g200` + additive AT-1 export (~45 min;
> harness verdict reproduced the banked `PDE-C1-DISC-INCONCLUSIVE` — regression clean).
> Rung 2 = `scripts/at1_boundary_layer.py` per the frozen
> `AT1_PALINSTROPHY_BOUNDARY_LAYER_SPEC.md`. Artifacts: `results/proof/at1-g200/`
> (`at1-samples.npz`, `at1_boundary_layer.json`). **Non-promotional; the C1 separation
> and the banked discriminator verdicts are untouched (per `AT1_HARNESS_SIGNOFF.md`,
> no reinterpretation — this receipt runs the §12 "separate probe, not pursued there").**

## Verdict: `AT1_INCONCLUSIVE_MIXED` (frozen table) — and BOTH named hypotheses are refuted

| check | value |
| --- | --- |
| liveness (E_low unbanded a_mm) | **−0.0008** ≤ 0.005 ✓ |
| reproduction (palinstrophy unbanded) | **a_mm 0.1949** (banked 0.195), damp 0.300, R² 1.0000 — exact |
| excision curve a_mm(β), all bands powered | 0.195 → 0.191 → 0.183 → 0.171 → **0.141** (β = 0→0.20) |
| sibling (lookahead-**mean** palinstrophy, unbanded) | **a_mm 0.0002** (POSITIVE), damp 0.300, R² 1.0000 |

- `AT1_BOUNDARY_LAYER_ARTIFACT` does **not** fire: banded a_mm never approaches the
  POSITIVE line — the disagreement is *not* a thin margin layer.
- `AT1_TWO_POLE_CONFIRMED` does **not** fire: the burst-robust sibling is perfectly
  control-sufficient — the disagreement does *not* replicate off the max statistic.

## The mechanism (post-read diagnostics, non-gate-bearing; verdict unchanged)

**Threshold-on-near-atom degeneracy of the quantile-calibrated lookahead-max on a
near-periodic cell.**

1. The lookahead-max distribution carries a **near-atom of mass ≈ 0.504 within ±1×10⁻⁶
   of 6.193857** — the near-periodic G=200 cell (AT-6: E_low autocorr ≈ 261 steps,
   quasi-regular) catches the *same recurring burst peak* in almost every 5-t.u. window,
   reproduced to six decimals.
2. The portable-quantile calibration (q = 0.70) **lands e_max inside the atom**
   (mass 0.296 at 6.193858 just above; 0.187 at 6.193857 at/below): for half the samples,
   the action label is decided by the **6th decimal** of the recurring peak.
3. Consequences, all now forced: R² = 1.0000 (the value is near-constant + excursions —
   trivially predictable); a_mm = 0.195 (an ε-ball of radius 0.176 holding ~1,000
   neighbors straddles a 10⁻⁶ split — **ball-straddle fraction = 1.000**, every ball);
   the excision curve decays only in proportion to samples removed (the "boundary layer"
   has mass ≈ 0.5, larger than every registered band — the frozen grid was built for a
   thin layer and correctly refused to call this one);
   and the **banked G=300 flip is mechanistically explained** — aperiodic dynamics have
   no recurring peak, no atom, so the threshold lands in continuum and the objective
   reads POSITIVE. The sibling confirms by contrast: lookahead-mean has no repeated
   mass (max 0.001), healthy margins (median |margin| 0.116), a_mm 0.0002.

**Typed:** the banked "predictable-but-control-insufficient" anomaly is an
**instrument degeneracy** (calibrated threshold ∩ value atom), not a second decision
pole of the cell. The cell is decision-clean for palinstrophy once the functional is
burst-robust. F2's two-pole candidate is hereby typed and closed at G=200. Note the
symmetry with AT-6: **both artifacts trace to the same physical cause — G=200's
near-periodicity** (there: no effective noise floor; here: an atomic max distribution).

## Consequences (for the slate; banked receipts untouched)

1. **A new registration check for every future quantile-calibrated objective**
   (binds AT-2's spec-freeze and AT-4's label family): **threshold-atom clearance** —
   the calibrated threshold must not land within a registered numerical width of a
   value cluster carrying more than a registered mass (here: 10⁻⁶ width held 0.50 mass).
   The existing power gate (damp ∈ [0.20, 0.40]) does not catch this — damp was a clean
   0.300 while the label was degenerate.
2. **AT-4's palinstrophy row should use the mean-form functional** (clean margins,
   POSITIVE, R² = 1.0) — the max-form is degenerate at G=200.
3. **Rung 3 does not trigger** (no TWO_POLE). AT-1 is complete.
4. The discriminator's banked verdicts stand as filed; its §12 anomaly now has a
   measured mechanism, recorded here, in the probe it itself deferred.

Cross-refs: `AT1_PALINSTROPHY_BOUNDARY_LAYER_SPEC.md` (frozen prereg),
`AT1_HARNESS_SIGNOFF.md` (scope), `NSE_ATTRACTOR_TAIL_HYPOTHESES.md` AT-1/F2,
`AT6_CHARFUN_TYPING_RECEIPT.md` (the near-periodicity sibling finding),
`../proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md` §12 (the banked anomaly, untouched).
