# AUGURY G4 — Pre-Registration (the determining-shadow-set read)

Parent / boundary documents:

- [`../../SUNDOG_V_AUGURY.md`](../../SUNDOG_V_AUGURY.md) (spine; §Allelopathy nod, gate G4)
- [`AUGURY_G1_PREREG.md`](AUGURY_G1_PREREG.md) (universe, cutoffs, band, scoring, exclusions —
  all inherited unchanged; G4 adds forecaster rungs, not new constants)
- [`AUGURY_G3_RESULT.md`](AUGURY_G3_RESULT.md) (`AUGURY_MARGIN_CONFIRMED` — the gate that
  unlocked G4; the 2-way market-vs-NBM encompassing G4 generalizes to the full ladder)

Filed: **2026-07-07 (PT)**. Status: **DRAFT — EXECUTION NOT ADMITTED.** Gated on owner
admission + a tooling freeze-marker (as G2/G3 were). `DOCS_NO_PUBLISH` until G5.

## §0 — The question

G3 showed the market is non-redundant against NBM alone. G4 re-poses it as the spine's
allelopathy read: **treat the forecaster pantheon as shadows of the temperature body; find the
minimal subset of forecasters that determines (encompasses) the outcome; and report whether the
market is in that determining set, at which horizons.** The market's membership-by-horizon *is*
the encompassing question generalized — on the first real, de-confounded, hard-settled
functional (weather), not the synthetic functional that capped the chatv2/allelopathy arc.

## §1 — The pantheon (rungs = shadows), all as P(high > θ) at matched availability cutoffs

| rung | role | source | distribution → F_k(θ) |
| --- | --- | --- | --- |
| GEFS | physics ensemble | `noaa-gefs-pds` `geavg`+`gespr` TMAX (0.25°, since 2020) | civil-day max = max of the 6-h TMAX blocks covering 12Z–00Z; mean μ, spread σ → Normal(μ,σ) survival |
| NBM | statistics (post-processed blend) | G3 `nbm_scalars.jsonl` (reused, no re-pull) | Normal(μ,σ) survival (as G3) |
| MOS | human/statistical point | IEM `NBS`/`MEX` MaxT (text; cheap) | **deterministic point** → margin covariate `(MaxT − θ)`; enters as a rung but never a distribution (fenced) |
| market | aggregate | Kalshi candles (reused, no re-pull) | PAV-monotonized midpoint exceedance (as G3) |

- **ECMWF-ENS (physics+AI): DEFERRED to an exploratory sub-amendment** (open-data enfo ships
  all-members files — a much larger pull, and its window starts 2023-02, shrinking the sample).
  The **core determining set is {GEFS, NBM, MOS, market}**; ECMWF is a later add if the core
  result warrants it. AIFS stays dropped (G1 flag).
- GEFS civil-day-max window handling is a **named approximation** (max-of-block-means for μ;
  the max-contributing block's σ for the spread) — fenced as such; it makes GEFS a coarse
  Gaussian rung, adequate for a determining-set read, not a calibrated GEFS CRPS claim.

## §2 — Universe, cutoffs, scoring (inherited)

Stations, primary window (2022-01-01 → 2026-06-30), the diurnal band (11 cutoffs), matched
**availability** cutoffs, validity/liquidity exclusions, strike-set Brier — **all exactly as
G1/G3**. G4 scores the identical (station, day, band-cutoff, valid strike) rows produced by G3's
`score` stage; it only appends the GEFS and MOS forecaster columns per row. Sample floor 500
valid station-days (inherited).

## §3 — The determining-set read (the G4 machinery)

Per (station-day, band cutoff, valid strike) row, with `z = 1{high > θ}`:

1. **Joint encompassing (full ladder), ridge-regularized** (G1's collinearity mandate — GEFS
   and NBM are near-duplicate physics-derived rungs):
   `z ~ logit(F_GEFS) + logit(F_NBM) + (MaxT_MOS − θ) + logit(F_mkt) + station FE`,
   L2 penalty on the forecaster coefficients (λ chosen by one pre-registered rule: minimize
   day-blocked CV deviance over a fixed λ grid; the grid + rule frozen in the tooling amendment).
   Probabilities clipped [0.01, 0.99].
2. **A rung is "in the determining set"** iff its coefficient's **95% day-block-bootstrap CI
   excludes 0** in the joint model (adds information beyond all other rungs). This is the
   encompassing test, generalized to k rungs.
3. **Minimal determining set** = backward elimination: drop the rung whose removal least
   degrades day-blocked CV deviance until every remaining rung's CI excludes 0. Report the set.
4. **Market membership BY HORIZON** — the headline: run steps 1–2 **separately per
   pre-registered horizon band**: **SHORT = band offsets {−5,−4,−3,−2}** (near t_peak, where the
   G3 crossover peaked) and **LONG = {−12,−11,−10,−9,−8}** (early band). Report β_market CI in
   each. The crossover predicts market ∈ set at SHORT, possibly out at LONG.

## §4 — Verdicts (precedence order; tokens final)

| verdict | condition |
| --- | --- |
| `AUGURY_G4_COLLAPSE` | < 500 joint-complete valid station-days (a rung's data missing too often) — no adjudication |
| `AUGURY_G4_MARKET_MEMBER` | market coefficient CI > 0 in the full-ladder joint model at **≥ 1** pre-registered horizon band — the market is in the minimal determining set; non-redundancy holds against the **whole pantheon**, not just NBM |
| `AUGURY_G4_MARKET_ENCOMPASSED` | market CI ⊄ (0,∞) at **every** horizon once the full ladder is present — the physics+stats+human rungs jointly encompass the market (its G3 win was NBM-alone weakness) |
| `AUGURY_G4_GAP` | mixed / underpowered |

Reported alongside (descriptive, non-adjudicating): the named minimal determining set overall
and per horizon; every rung's coefficient + CI by horizon; how the set changes SHORT vs LONG
(the determine/resist-by-horizon map — the causal-access **aggregate row** grounding).

## §5 — Scope / fences

- The market-vs-full-pantheon **information** claim (raw midpoints) stays distinct from tradable
  edge (G1 fence). GEFS/MOS window-alignment approximations are named (§1). MOS is a point
  covariate, never a distribution. ECMWF deferred. This is a determining-set read, not a
  recalibration of any single model.
- Tooling reuses the G3 primitives (candles, NBM scalars, CLI, scoring, day-block bootstrap,
  the boustrophedon GRIB decode for GEFS) at their frozen hashes; a G4 freeze-marker amendment
  will record the new runner (`augury_g4.py`), the λ grid + rule, and the exact command before
  any binding run. Results → `results/augury/g4-run/` (gitignored).

## §6 — Estimated cost

GEFS pull ≈ the NBM order of magnitude (2 products × 2 daily-max blocks per issue, 0.25°
fields; issue-cached, resumable) — **~2–3 h**. MOS via IEM text — cheap. Score + adjudicate ≤
1 h. No ECMWF in the core. **~3–4 h wall, fully resumable.**
