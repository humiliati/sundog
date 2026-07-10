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

- **ECMWF is a core rung** (owner choice). Because open-data enfo ships **members only** (no
  ensemble mean/spread product), μ,σ are computed from a **fixed member subsample**: perturbed
  members **{1..21} = 21 members** (the `mx2t3` product carries no control record —
  perturbed-only; a named approximation — 21 members estimate ensemble μ/σ adequately for a
  determining-set read; the full 50 would ~2.4× the cost for negligible μ/σ change). Blocks:
  the `mx2t3` 3-h blocks covering the CLI civil day (ending 15/18/21/00 UTC). Rate-limited S3
  (503 Slow Down) → 503-aware backoff + gentle pacing. AIFS stays dropped (G1 flag).
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

## Amendment G4-α — ECMWF S3 throttle wall + execution plan (2026-07-08)

Append-only. During the build, ECMWF's `ecmwf-forecasts` bucket (eu-central-1) proved to
**503 "Slow Down"–throttle sustained access** to the point of infeasibility: after this
session's recon+test bursts, a single ENS issue (84 fields) took ~500s and *failed*, and even
one **HRES** issue (4 fields) took ~470s and failed. The throttle is a burst penalty that
clears with cooldown, but a sustained ~3,000-issue pull re-triggers it. Point extraction itself
is verified correct 7/7 on GEFS and ECMWF (regular_ll) — this is purely a rate-limit wall, not
a data or code problem.

**Execution plan (forced by the wall):**
1. Pull the throttle-free rungs now: **GEFS** (`geavg`+`gespr` TMAX, regular_ll 7/7) and **MOS**
   (IEM text). These + reused **NBM** scalars + **market** candles = the **core-4 ladder**
   {GEFS, NBM, MOS, market} — a legitimate physics/statistics/human/aggregate determining-set.
2. Let ECMWF cool (no access) while 1 runs, then attempt a **gentle ECMWF-HRES** pull
   (deterministic `oper` mx2t3, 4 fields/issue, 0.25 s pace, 503-backoff). HRES enters as a
   **deterministic margin covariate** (like MOS), fenced as the ENS-throttle fallback — the
   flagship deterministic IFS, not the ensemble the owner picked (ENS was walled by S3, not by
   choice).
3. The runner is **rung-optional**: `score`/`adjudicate` include each of ECMWF and MOS only if
   its scalar cache is present/dense enough, else drop it. Either way a `AUGURY_G4_*` verdict
   lands. ECMWF-ENS remains a possible future sub-amendment from an un-throttled context.

**MOS rung DROPPED (data unavailability, not choice):** IEM's NBS (NBM-MOS) archive via the API
is essentially empty at 0/6/12/18Z runtimes historically — coverage 0% (2023), 0% (2024),
0% (2025), 24% (2026). MAV (short GFS-MOS) returns HTTP 422; MEX (extended GFS-MOS) has a deep
archive but is days-3+ only (no same-day `n_x`). No reliable historical same-day point-MOS
source was found. The "human/statistical-point" rung is therefore dropped and fenced as
unavailable; the reliable ladder is **{GEFS (physics ensemble), NBM (statistical blend),
market (aggregate)}** + ECMWF-HRES if the bucket cools. This still tests the core question:
is the market non-redundant given the physics ensemble + the operational blend (+ deterministic
ECMWF)?

## Amendment G4-β — ECMWF-ENS sub-amendment (2026-07-09)

Append-only; the pre-named ENS upgrade (owner-directed). Upgrades the ECMWF rung from the
HRES-deterministic margin (the throttle-forced fallback of Amdt G4-α) to the **ensemble
distribution the owner originally chose**: `F_ECMWF-ENS(θ) = Normal(μ, σ)` survival, μ/σ from
the **21-member perturbed subsample** (pf 1..21; `mx2t3` carries no control record), per-member
civil-day max over the four 3-h blocks ending 15/18/21/00 UTC — exactly the §1 construction.

**Scope: a SENSITIVITY of the G4 ECMWF rung, not a fresh adjudication.** The G4 verdict
(`AUGURY_G4_MARKET_MEMBER`) and the HARUSPEX verdicts stand on their own records; this
sub-amendment asks only (i) does the ECMWF rung's membership strengthen (tighter CI) when it
enters as a true distribution, and (ii) does any qualitative G4 conclusion change (expected: no —
pre-registered expectation: ENS strengthens ECMWF's coefficient, market membership unchanged).

**Pins (frozen with the runner):**
- **Issue set = the cached HRES issue set** (`results/augury/g4-run/ecmwf_scalars.jsonl` keys,
  ~1,250 day|cycle pairs) — the same cycles the availability join already selected. Named
  approximation: enfo (ENS) availability is taken as the oper (HRES) availability for cycle
  selection; the enfo `LastModified` is **recorded per issue** and the fraction where enfo
  published materially later (> 30 min) is reported as an honesty diagnostic.
- **Pacing discipline (the throttle lesson):** a **probe stage precedes the pull** — ~20 issues
  at 1-way then 2-way concurrency, recording 503-rate and throughput; the pull launches at the
  highest non-throttling pace. Sequential + 0.25 s gap + long 503 backoff is the floor
  (≈ 30–36 h, resumable per-issue; every decoded issue flushes to the scalar cache).
- **Rescore:** a new composing runner (`augury_ens.py`, read-only over the frozen `augury_g4`
  primitives — same discipline as HARUSPEX) rebuilds the G4 score rows with
  `f_ecmwf_ens` (logit rung, replacing the HRES margin) and re-runs the G4 adjudication;
  results → `results/augury/g4-ens-run/` (gitignored). Verdict tokens unchanged (§4).

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
