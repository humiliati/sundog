# NSE-H3 Mid-Energy Sub-Regime — Receipt

> 2026-07-07. Read of the frozen `NSE_H3_MIDE_SUBREGIME_SPEC.md`. Post-processing
> on the banked `fallback_v7_g675_kf3` samples (no re-integration). Non-promotional;
> scoped to the mid-energy sub-attractor, proxy-relative, no C1 promotion, no-publish.

## Verdict: `NSE-H3-FORCING-GENERAL-MIDE` — a clean, gate-clearing positive

The frozen regime-2 certificate **clears all its registered gates on the
mid-energy sub-attractor** of the matched-Re (k_f=3, G=675) cell:

| gate | value | pass |
| --- | --- | --- |
| within-band candidate coverage | **0.6863** | ≥ s_pos 0.50 ✓ |
| state-insufficiency | `TWIN_STATE_CERTIFIED`, 82,547 witness pairs, witness-sample frac 0.686 | ✓ |
| control-sufficiency (paired) | `PAIRED_FIBER_CONSTANCY_POSITIVE`, disagree **0.0218** | ≤ delta_action 0.10 ✓ |

Central energy tercile: n = 66,666 / 200,000; frozen ε_K = 0.0589 (unchanged);
δ_H = 0.0158.

## Why this is a genuine test, not gate-shopping

1. **Regression clean.** The identical restriction machinery on the near-mono-energy
   anchors reproduces the banked certification: G=200 and G=300 both
   `TWIN_STATE_CERTIFIED`, coverage 1.000, paired POSITIVE, on their central
   terciles (16,666 each). The machinery does not manufacture a positive.
2. **Band fixed by rule.** The central energy tercile `[q⅓, q⅔]` — a symmetric,
   principled sub-regime — not a coverage-derived cut. Every verdict gate is the
   frozen protocol.
3. **The `UNDERCOVERED` null was live.** Restricting to a self-contained
   sub-sample could have dropped within-band coverage below 0.50 (neighbours
   outside the band are excluded). It did **not**: within-band coverage 0.6863
   ≈ the dig-in's cross-band 0.690 — **the mode band is self-coherent** (mid-E
   samples' Φ_K-near neighbours are themselves mid-E). That coherence is a real,
   falsifiable finding, not an assumption.
4. **Control-sufficiency is tight** — disagree 0.0218, below the anchor's 0.033 and
   the dig-in's dense-core 0.0249.

## Claim boundary (the fences stay load-bearing)

`FORCING-GENERAL-MIDE` says: **on the mid-energy sub-attractor — the typical-energy,
well-sampled bulk (~⅔ of samples, 68.6% covered) — the regime-2 witness of the
matched-Re forcing move clears the standard gates: state-insufficient *and*
control-sufficient.** It does **not** claim the full G=675 attractor (which stays
coverage-limited — the energy tails are too sparse; frozen DEFER / A SLIVER stand
as the full-support reads); it is disclosed that the central energy tercile is also
the densest region in signature space, so the scope is explicitly the mode band, not
the tails. One sub-regime, one forcing-axis move (k_f=2→3 at matched Re_f), finite,
sampled-support, proxy-relative. No universality, no cross-k_f generality, no C1
promotion, no infinite-dimensional statement.

## What it settles for the H3 forcing axis

The forcing axis — which read `INCONCLUSIVE_COVERAGE` (frozen) / `SLIVER` (A) /
fenced-positive (B) at full support — yields its **first genuinely clean,
gate-clearing positive once restricted to the coherent mode-band sub-attractor.**
The regime-2 witness *does* generalize to the matched-Re forcing geometry; the
full-attractor obstruction was geometric (coverage on a wider attractor's sparse
tails), not a failure of either witness half. The dig-in's reading — "structure
present, coverage-limited" — is now confirmed by a gate-clearing certificate on the
sub-regime where coverage is not limiting.

## Ledger note

This is a **scoped** positive and does not change the registered full-support
verdicts (`NSE-H3-INCONCLUSIVE_COVERAGE`, `-COVERAGE-SLIVER`,
`-FORCING-GENERAL-RELATIVE`-fenced). It stands beside them as the mode-band read.

Cross-refs: `NSE_H3_MIDE_SUBREGIME_SPEC.md`, `NSE_H3_G675_STRUCTURE_DIGIN.md`,
`NSE_H3_ADMISSION_RECEIPT.md`, `PDE_C1_TWIN_STATE_CERTIFICATE.md`,
`results/proof/c1-h3-kf3-g675-adaptive/mide_subregime_manifest.json`,
`results/proof/c1-relative-reg-g{200,300}/` (regression).
