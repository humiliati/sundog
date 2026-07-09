# SUNDOG_V_AUGURY

*Lane spine. Do the gamblers beat the bureau — and does the crowd's price carry weather
information the models don't? Frame + gated build plan toward a public page. Nothing pulled,
nothing scored; the lit-pass memo (G0) gates everything.*

*Name: AUGURY = forecasting from signs; a weather market is a modern augur reading tomorrow's
temperature off prices, and Roll (1984) showed the augur beats the bureau at the margin. Rename
trivially to `SUNDOG_V_WEATHER` / `_KALSHI` if the codename reads too cute — it's a filename and a
header, not a claim.*

Status: **G3 RUN COMPLETE 2026-07-07 → `AUGURY_MARGIN_CONFIRMED`**
([`prereg/augury/AUGURY_G3_RESULT.md`](prereg/augury/AUGURY_G3_RESULT.md)). Market is a non-redundant
pantheon member (encompassing β_mkt CI [0.670,0.776]>0; DM p≈0) with the predicted horizon-localized
crossover (edge −0.010 at −12h → −0.043 at −2h). Caveat: NBM comparator = mean+spread Gaussian (native
percentiles unavailable) — encompassing survival is the robust core. G0/G1/G2/G3 all in
[`AUGURY_G1_PREREG.md`](prereg/augury/AUGURY_G1_PREREG.md) (Amendments A–C.6).

**G4 COMPLETE 2026-07-08 → `AUGURY_G4_MARKET_MEMBER`**
([`prereg/augury/AUGURY_G4_RESULT.md`](prereg/augury/AUGURY_G4_RESULT.md)). Determining-shadow-set read on
the full ladder {GEFS, NBM, ECMWF-HRES, market} (MOS dropped — no historical source; ECMWF-ENS throttle-
walled → HRES, Amdt G4-α). **Minimal determining set = {NBM, ECMWF-HRES, market}; GEFS screened off** (raw
physics ensemble redundant given NBM). Market in the set at both horizons (stronger short); ECMWF in only at
long lead — market & ECMWF at opposite ends of the horizon. **G5 (augury.html) now the remaining gate,
owner-gated; DOCS_NO_PUBLISH until G5.**

---

## The claim

The discriminator-passing one-sentence version (the hook):

> Kalshi's implied temperature distributions, snapshotted at a fixed information cutoff, achieve
> lower CRPS than the National Blend of Models at the same cutoff, for a **pre-registered**
> lead-time band.

That is weak as stated — a one-band CRPS win is exactly what a lossy echo of NBM produces by luck.
The lane's real object is the sharpened, portfolio-native form (the substance under the hook):

> **The market is a non-redundant member of the forecaster pantheon: its probability survives an
> encompassing regression against the full public model ladder, and its marginal information is
> horizon-localized to the short lead where it depends on real-time observations the model cycle
> does not.**

The crossover is an *information-access* result, not a curiosity: the market wins exactly and only
where it depends on information (real-time METAR) the stale model cycle does not carry — the
determine/resist invariant on real money.

## Precedent + the prior-art gap

- **Roll (1984), "Orange Juice and Weather" (AER):** FCOJ futures prices predicted subsequent errors
  in NWS Florida temperature forecasts — the market carried weather information beyond the bureau.
- **The gap (verify in G0):** public Kalshi analyses show *bucket calibration* (e.g. 90–100%-priced
  markets resolve YES ~98.6%). Nobody has published **CRPS-vs-NBM at matched cutoffs**, the
  horizon-crossover, or an encompassing kill-test. That gap is the contribution.

## Null vs live

- **Null:** market ≈ lossy compression of the best public ensemble, minus fees. Arb bots reprice
  within seconds of every NWS cycle → the market is a monolith-shadow of NBM; encompassing
  coefficient on the market = 0.
- **Live (horizon-dependent):** models refresh on 6-hour cycles; the market reprices continuously
  against real-time METAR. So the market should **lose at long lead** (day 3 — echoing stale
  ensembles) and **win in the final hours**, when "it's already 51°F at 2pm" bounds the daily high
  in ways the latest model post-processing can't fully capture at the same wall-clock cutoff. **The
  crossover is the finding.**

## The pantheon (the forecaster gods: physics → statistics → human → market, plus AI)

| rung | forecaster | source |
| --- | --- | --- |
| physics | GEFS ensemble | AWS open data via Herbie |
| statistics | **NBM** (operational post-processed blend) | `noaa-nbm` on AWS |
| human | NDFD / MOS point forecast | IEM archives |
| physics + AI | ECMWF ENS + AIFS | ECMWF open data |
| aggregate | **Kalshi market** | Kalshi API 1-min candles (yes_bid/yes_ask OHLC) |

The pantheon question is not "market vs NBM" — it is **what is the minimal sufficient forecaster
set, is the market in it, and at which horizons.** Primary encompassing = market vs NBM (the best
single public tool; clean, low collinearity). Secondary = full-ladder joint encompassing, reporting
the market coefficient **by horizon** — and handling GEFS/NBM/ECMWF collinearity explicitly
(ridge / orthogonalize; do not naively OLS a set of near-duplicate ensembles).

## Build spine (exact params deferred to the G1 pre-reg)

- **Market side:** Kalshi historical 1-min candles → bid-ask midpoints at fixed cutoffs (06/12/18Z)
  → stack the strike ladder into an implied CDF → isotonic-monotonize.
- **Ground truth:** NWS Daily Climate Report (CLI) for the named station, scraped via IEM.
  **Station match is non-negotiable.**
- **Model ladder:** the table above.
- **Scoring:** CRPS + Brier evaluated **at the exact strikes the market trades** (fairest
  comparison); reliability diagrams; Diebold-Mariano with **block bootstrap over station-days**
  (errors correlate spatially). `properscoring` / `scoringrules`.
- **Kill test:** encompassing regression — outcome on {NBM prob, market prob} (primary) and the full
  ladder (secondary). Market coefficient 0 → thesis dead, reported honestly.

## The crux + pre-registered failure boundaries (tight, so we can move on if there's no juice)

1. **Same wall-clock cutoff for every rung — including the freshest model cycle available at that
   minute** (NBM updates hourly now). Otherwise the final-hours "win" measures *staleness*, not
   skill. This is the interpretability crux of the entire crossover.
2. **Pre-register ONE primary band** — the mechanistic final-12h / daily-high-not-yet-in — defined
   against the **diurnal cycle**, not raw hours (the high collapses mid-afternoon; "final 12h"
   differs June vs December). "≥1 band" over many slices is a fishing hazard the DM bootstrap won't
   fully absorb; the dominance-by-horizon map is **declared-exploratory**.
3. **Liquidity exclusion:** spread > ~4¢ or daily volume < threshold ⇒ the "distribution" is
   fiction. Exclude, and **report the exclusion rate**.
4. **Fees:** a flat per-contract tax (~20% at 5¢ prices) mechanically distorts tails. Score **raw
   midpoints** for the information claim; keep "tradable edge" as a separate, weaker claim.

## Allelopathy nod (gated secondary — the recoverability instrument's first real functional)

The determining-shadow-set read (allelopathy lane; [`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md)'s
sibling instrument) asks "the minimal set of shadows that determines the body." Here the body is the
temperature outcome and the shadows are the forecasters, so the minimal determining set + the
market's membership by horizon **is** the encompassing question, re-posed. Its distinctive value:
the chatv2/allelopathy arc kept dying at the R1.5 ceiling — *"the functional is ours, synthetic."*
Here the functional is the **weather** (not ours), the shadows are genuinely de-confounded
(independent forecasting systems), and the ground truth is hard settlement — the **first
real-functional substrate** for the read. The concept ports; the machinery is the
forecast-encompassing / CRPS version, **not** the neural probe. Runs only if G3 clears (no
interesting substrate otherwise). Grounds the causal-access umbrella's **aggregate** row
([`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md)) on something that isn't a toy.

## Gates → deliverable ladder (toward a class-A page)

- **G0 — Lit-pass memo (REQUIRED FIRST).** `AUGURY_LIT_PASS_MEMO.md`: (a) confirm the prior-art gap;
  (b) Roll 1984 + prediction-market / weather-derivative forecasting follow-ups; (c) a
  **data-availability table** — Kalshi candle depth, IEM CLI/NDFD windows, Herbie/GEFS/NBM/ECMWF-open
  coverage — verifying all rungs + settlement are pullable for ≥ N station-days; (d) scoring tooling
  confirmed. **Gate:** proceed only if the gap is real AND the rungs are actually pullable.
- **G1 — Pre-registration.** The runnable pre-reg: cutoffs, the primary band, stations, exclusions,
  the encompassing design (primary + collinearity-handled secondary), the kill criterion, verdict
  tokens. Self-consistency gate (numbers reconciled across sections; gate = exact criterion).
- **G2 — Single-station pilot.** Build the pipeline on one high-liquidity station; verify the
  implied-CDF construction, the CLI settlement match, and the scoring harness; reproduce the known
  bucket calibration (~98.6%) as a sanity check. Verify-before-scale.
- **G3 — Full run + adjudication.** All stations/days → CRPS/Brier/reliability/DM/encompassing +
  the dominance-by-horizon map. Verdicts:
  - `AUGURY_MARGIN_CONFIRMED` — pre-registered band shows market CRPS < NBM **and** the market
    coefficient survives encompassing.
  - `AUGURY_ENCOMPASSED` — market adds nothing beyond the models (coefficient 0). Dead, honest
    negative; a clean result, not a failure.
  - `AUGURY_GAP` — muddy (e.g. CRPS win that doesn't survive encompassing). Inconclusive.
- **G4 — Allelopathy secondary** (only if G3 = CONFIRMED): the determining-shadow-set read.
- **G5 — Class-A page** (only if G3 = CONFIRMED): `augury.html`, honest-fenced (information claim vs
  tradable-edge kept separate; the encompassing-survival is the substance, the crossover map is the
  visual). LinkedIn post derived from the page. If `ENCOMPASSED`: no page, or a short clean-negative
  note.

## Scope / fences

- The **information claim** (raw midpoints) is distinct from a **tradable-edge claim** (fees +
  liquidity) — kept separate; the second is weaker and secondary.
- Tier: real-data, real-functional, externally-validated. **Not** a trading system, not investment
  advice, no live-money claim.
- What kills it: encompassing coefficient 0; or the "win" evaporating once the model side is
  snapshotted at the freshest same-cutoff cycle (staleness artifact); or liquidity exclusion eating
  the sample.
- Publication posture: this spec stays internal until G5; the page is the only intended public
  surface (`DOCS_NO_PUBLISH` per owner until then).

## Cross-links

- Pantheon / encompassing / dominance-by-horizon: the mesa Competence-Dominance Lemma
  ([`SUNDOG_V_TAUROCTONY.md`](SUNDOG_V_TAUROCTONY.md)).
- Determine/resist + minimal sufficient set: the sufficient-statistic-order slate and the
  Shadow-Invertibility law (lossy shadow determines discrete/topological, resists continuous).
- Aggregate-channel grounding: [`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md).
- **NSE circle-back (separate axis, NOT fused):** the C1/C2 recoverability / determining-modes leads
  ([`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md)) share the determining-set machinery but
  attach on their own axis (vortex reconnection → determining-mode jumps). Revisit after that lane's
  own freshen-up / cascade-specifics gate; do not fuse into AUGURY.
