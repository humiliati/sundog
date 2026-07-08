# AUGURY G4 — Pre-Registration (the determining-shadow-set read)

Parent / boundary documents:

- [`../../SUNDOG_V_AUGURY.md`](../../SUNDOG_V_AUGURY.md) (spine; §Allelopathy nod, gate G4)
- [`AUGURY_G1_PREREG.md`](AUGURY_G1_PREREG.md) (universe, cutoffs, band, scoring, exclusions —
  all inherited unchanged; G4 adds forecaster rungs, not new constants)
- [`AUGURY_G3_RESULT.md`](AUGURY_G3_RESULT.md) (`AUGURY_MARGIN_CONFIRMED` — the gate that
  unlocked G4; the 2-way market-vs-NBM encompassing G4 generalizes to the full ladder)

Filed: **2026-07-07 (PT)**. Status: **DRAFT — EXECUTION NOT ADMITTED.** Gated on owner
admission + a tooling freeze-marker (as G2/G3 were). `DOCS_NO_PUBLISH` until G5.
**Owner chose the full pantheon incl. ECMWF-ENS (2026-07-07).** Scope below updated: ECMWF is a
core rung; the full-ladder window is 2023-02-01 → 2026-06-30 (ECMWF archive start). Point
extraction verified 7/7 vs `find_nearest` on all three GRIB rungs (NBM Lambert+boustrophedon
reused from G3; GEFS + ECMWF are regular 0.25° lat/lon — trivial indexing, no boustrophedon).

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

| ECMWF-ENS | physics+AI | `ecmwf-forecasts` enfo `mx2t3` (0.25°, since 2023-01-18) | per-member civil-day max = max of the 3-h `mx2t3` blocks covering 12Z–00Z; ensemble μ, σ over the member subsample → Normal(μ,σ) survival |

- **ECMWF is a core rung** (owner choice). Because open-data enfo ships **members only** (50
  perturbed + control; no ensemble mean/spread product), μ,σ are computed from a **fixed
  member subsample**: control + perturbed members {1..20} = **21 members** (a named
  approximation — 21 members estimate ensemble μ/σ adequately for a determining-set read; full
  50 would ~2.4× the cost for negligible μ/σ change). Blocks: the `mx2t3` 3-h blocks covering
  the CLI civil day (ending 15/18/21/00 UTC). AIFS stays dropped (G1 flag).
- GEFS civil-day-max window handling is a **named approximation** (max-of-block-means for μ;
  the max-contributing block's σ for the spread) — fenced as such; it makes GEFS a coarse
  Gaussian rung, adequate for a determining-set read, not a calibrated GEFS CRPS claim. ECMWF
  uses the same block-max composition per member before the ensemble μ/σ.

## §2 — Universe, cutoffs, scoring (inherited)

Stations, the diurnal band (11 cutoffs), matched **availability** cutoffs, validity/liquidity
exclusions, strike-set Brier — **all exactly as G1/G3**. Window: the full-ladder run is
**2023-02-01 → 2026-06-30** (ECMWF archive start; G1's exploratory-ladder window). G4 scores the
subset of G3's (station, day, band-cutoff, valid strike) rows falling in that window and appends
the GEFS, ECMWF, and MOS forecaster columns per row. Sample floor 500 valid station-days.

## §3 — The determining-set read (the G4 machinery)

Per (station-day, band cutoff, valid strike) row, with `z = 1{high > θ}`:

1. **Joint encompassing (full ladder), ridge-regularized** (G1's collinearity mandate — GEFS,
   NBM, ECMWF are near-duplicate physics-derived rungs):
   `z ~ logit(F_GEFS) + logit(F_NBM) + logit(F_ECMWF) + (MaxT_MOS − θ) + logit(F_mkt) +
   station FE`, L2 penalty on the forecaster coefficients (λ chosen by one pre-registered rule:
   minimize day-blocked CV deviance over a fixed λ grid; the grid + rule frozen in the tooling
   amendment). Probabilities clipped [0.01, 0.99].
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

## §6 — Estimated cost (full pantheon)

- **GEFS** ≈ NBM order (geavg+gespr × 2 daily-max blocks per issue, 0.25° fields;
  issue-cached, resumable) — **~2–3 h**.
- **ECMWF-ENS** is the driver: 21 members × ~3–4 `mx2t3` blocks × ~3,000 distinct issues over
  2023-02→2026-06 at 0.64 MB/field ≈ **~120 GB, ~13 h** (bandwidth-bound; per-issue scalar
  cache, fully resumable). The 21-member subsample (vs 50) already trims this ~2.4×.
- **MOS** via IEM text — cheap. **Score + adjudicate** ≤ 1 h.
- **Total ~16–18 h wall, overnight, fully resumable.** Cheaper fallback if that is too much:
  swap ECMWF-ENS for **ECMWF-HRES** (the deterministic IFS `oper` run) as a point covariate
  like MOS — ~1–2 h, but loses the ensemble distribution (a determinstic ECMWF rung, not the
  physics+AI *ensemble* the owner chose).
