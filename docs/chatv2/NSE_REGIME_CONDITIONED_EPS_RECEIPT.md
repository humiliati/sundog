# Regime-Conditioned ε_K (Approach B) — Receipt

> 2026-07-07. Build + R0 regression of `NSE_REGIME_CONDITIONED_EPS_SCOPE.md`.
> Non-promotional. **The regression gate did its job: it caught that B's
> anti-inflation guard, as frozen, rejects the certified anchor witness — before
> any G=675 read.** No G=675 compute was spent.

## Build + synthetic validation (passed)

Additive `--adjudicator twin-state-relative`; frozen constants
`RELATIVE_CONSISTENCY_TOL = 0.03`, `RELATIVE_COVERED_FLOOR = 0.10`,
`RELATIVE_MIN_TERCILE_PAIRS = 30`. Synthetic self-test 3/3: reduction-on-compact,
inflation-detection (guard fires on planted high-E false fibers), consistent
positive.

## R0 regression: FAIL → `NSE-H3-APPARATUS-REJECTED-B`

| cell | frozen | relative | f_rel | \|rel−froz\| disagree | scale-consistency |
| --- | --- | --- | --- | --- | --- |
| G=200 | `TWIN_STATE_CERTIFIED` | `TWIN_STATE_RELATIVE_CERTIFIED` | 1.0000 | 0.00000 | **False** (spread 0.063 > 0.03) |
| G=300 | `TWIN_STATE_CERTIFIED` | `TWIN_STATE_RELATIVE_CERTIFIED` | 1.0000 | 0.00000 | **False** (spread 0.052 > 0.03) |

**Reduction is flawless** — on both compact cells the relative read is
bit-identical to frozen (pair sets 693,774 / 942,834 == frozen, disagree
0.03678 / 0.03819 == frozen, f_rel = 1.0). B reduces to frozen exactly where the
energy is compact, as designed.

**But the scale-consistency guard fires on the certified anchor.** Tercile
disagree (n ≈ 437k / 566k per tercile — not sampling noise):

- G=200: [0.0453, 0.0000, 0.0629], overall 0.0368
- G=300: [0.0568, 0.0046, 0.0514], overall 0.0382

## What this means (the diagnosis)

1. **The guard is mis-specified, not B-in-principle-wrong.** On the anchor the
   relative radius *equals* frozen (reduction exact ⇒ zero inflation possible),
   yet the guard fires. So it is flagging something that is **not** inflation —
   a false positive of the guard.
2. **A true positive's witness has real, non-monotonic energy structure.** The
   certified anchor's disagree is U-shaped in `E_pair` (~0.05 at the energy
   extremes, ~0.00 in the middle) with a natural spread of ~0.05–0.06 — **above**
   the frozen 0.03 tolerance. The raw max−min spread cannot distinguish this
   benign U-shape from the *monotonic rise* that genuine inflation would produce.
3. **The regression gate worked exactly as intended.** §3 condition 4 ("the guard
   must not spuriously fire on the compact cells") caught the defect on the
   control, so no G=675 read was taken and no false positive was risked. This is
   the no-rescue design paying off: the apparatus is rejected as frozen, not
   quietly re-tuned to pass.

Per the frozen spec, R0 failure ⇒ **`NSE-H3-APPARATUS-REJECTED-B`, final for this
registration.** The `0.03` tolerance and the raw-spread formulation are **not**
changed here — that would be the gate-widening the lane forbids.

## H3 forcing axis — status after A and B

- **A (absolute ε_K):** `NSE-H3-COVERAGE-SLIVER` — no ε_K-dense fibers; the wall
  is a real geometric fact.
- **B (relative ε_K):** `NSE-H3-APPARATUS-REJECTED-B` — the relative-resolution
  route cannot be read without a valid anti-inflation guard, and the guard as
  specified rejects the control. The relative route is **blocked**, not answered.

## Owner fork (staged, not spent)

1. **Accept `APPARATUS-REJECTED-B` as final.** The regime-conditioned mechanism
   is closed; H3's forcing axis rests on A's real wall plus B's blocked route.
2. **Commission a v2 guard (new registration).** The diagnosis is specific: the
   guard must test for the *inflation signature* — a **monotonic** rise of
   disagree with energy, or a high-tercile **excess over the anchor's measured
   natural pattern** — with its threshold **calibrated from the anchor** (the
   control's ~0.06 spread and U-shape as the null model), not an arbitrary
   constant. That is a redesign + fresh pre-registration + its own R0, honestly
   calibrated-from-control rather than tuned-to-pass. Owner-gated.

Cross-refs: `NSE_REGIME_CONDITIONED_EPS_SCOPE.md`,
`NSE_COVERAGE_ADAPTIVE_APPARATUS_RECEIPT.md` (Approach A null),
`NSE_H3_ADMISSION_RECEIPT.md`,
`results/proof/c1-relative-reg-g{200,300}/` (this regression).
