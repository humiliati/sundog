# AUGURY G3 — Full-Run Result

> 2026-07-07. Adjudication of the binding G3 run against
> [`AUGURY_G1_PREREG.md`](AUGURY_G1_PREREG.md) §6–§7. Internal (`DOCS_NO_PUBLISH`
> until G5). Verdict tokens are §7's; public language is §9's.

## Verdict: `AUGURY_MARGIN_CONFIRMED`

Both pre-registered tests pass (`g3_result.json`, sha256 `bed09869…`):

- **DM (score):** market mean strike-Brier is **0.0127 lower** than the NBM comparator over
  the primary band; block-bootstrap se 0.0027, **DM stat −4.73, one-sided p ≈ 0** (H1: market
  lower). n = **3,488 valid station-days** (floor 500; `SAMPLE_COLLAPSE` not triggered).
- **Encompassing (kill test):** pooled logistic, `1{high>θ} ~ logit(F_NBM) + logit(F_mkt) +
  station FE`. **β_mkt = 0.725, 95% day-block CI [0.670, 0.776] — entirely > 0**, so the market
  survives. β_NBM = 0.340 (> 0): NBM is *also* non-redundant. Neither rung encompasses the
  other → the market is a **non-redundant member of the forecaster pantheon**, exactly the
  spine's sharpened claim.

## The crossover (exploratory, §3 — the mechanistic corroboration)

The market's edge grows monotonically toward short lead — the information-access signature the
spine predicted (mean strike-Brier, market − NBM; negative = market better):

| band offset | −12h | −9h | −6h | −5h | −4h | −3h | −2h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| diff | −0.010 | −0.013 | −0.017 | −0.023 | −0.024 | −0.034 | **−0.043** |

The market wins ~4× more at −2h than at −12h. Mechanism (fenced, Amendment A.1): NBM issues no
*new* same-day MaxT after ~12Z, so at short lead the market's real-time observations ("already
X° at 2pm bounds today's high") beat the operational blend's last-issued MaxT. This is
**declared-exploratory** (not adjudicating) but it is the shape the lane was designed to find.

## Robustness

- **Broad-based across the pantheon:** all 7 stations market ≤ NBM (KMDW −0.034, KNYC −0.018,
  KLAX −0.014, KDEN −0.008, KMIA −0.006, KPHL −0.006, KAUS +0.0002 ≈ neutral).
- **Not a single-era artifact:** excluding the densest era (2026-H1) the pooled diff stays
  negative (−0.0067, n = 2,300) at ~half the full magnitude (−0.0127, n = 3,488). Carried by
  the liquid 2025–2026 eras; 2022–2024 are sparse/noisy (era-cliff, as G2 flagged).
- **Reliability:** both rungs are reasonably calibrated (market p~0.9 → 0.956 realized; NBM →
  0.948), so the CRPS win is **not** a "NBM-Gaussian is broken" artifact — see caveat.

## Load-bearing caveats (the result is real; the interpretation is bounded)

1. **The NBM comparator is the mean+spread Gaussian, not NBM's native percentile product**
   (Amendment A.1 — qmd/NBP percentiles did not exist for the window). The reliability diagram
   shows this Gaussian is decently calibrated, and the **encompassing survival (β_mkt CI ≫ 0)
   is robust to the shape** in a way the raw CRPS win is not — so the encompassing result is the
   claim to lean on.
2. **The short-lead win is against the operational MaxT product as issued** (NBM stops
   refreshing today's MaxT after ~12Z). This is the honest, prereg-compliant comparison at
   matched *availability* cutoffs; it is a statement about operational products, not "the market
   beats the best conceivable model." An obs-augmented model-side nowcast is pre-named
   exploratory, never the primary rung.
3. Effect size is modest (~5–10% relative strike-Brier at mid-band) and era-dependent; the
   headline is **statistical non-redundancy + the horizon-localized crossover**, not a large
   uniform CRPS gap.

## What this opens (per the spine, owner-gated)

`AUGURY_MARGIN_CONFIRMED` unlocks **G4** (the allelopathy determining-shadow-set read — the
market's membership by horizon *is* the encompassing, re-posed on the first real-functional
substrate) and **G5** (`augury.html`, information-claim vs tradable-edge fenced; the
encompassing-survival is the substance, the crossover map the visual). Both remain owner-gated;
nothing public until G5.

Artifacts: `results/augury/g3-full-run/` — `g3_result.json` (verdict, sha `bed09869…`),
`scores.jsonl` (84,923 strike rows), `nbm_scalars.jsonl` (10,990 issues), `score_exclusions.json`,
`city_settlement_audit.json` (7/7 cities 20/20).
