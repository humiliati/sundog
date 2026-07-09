# SUNDOG_V_AUGURY_II (HARUSPEX) — G0 Lit-Pass Memo

> 2026-07-09. G0 pass against [`SUNDOG_V_AUGURY_II.md`](SUNDOG_V_AUGURY_II.md): (a) confirm the
> provenance-decomposition gap, (b) precedents for each piece, (c) a **data-availability table
> with live receipts** for the two new dependencies (real-time obs for H1's nowcast; the
> final-cycle model for H2), (d) confirm the reused Augury apparatus covers H1/H2/H3. Gate
> (verbatim from the spine): *proceed only if the gap is real AND the obs + final-cycle data are
> pullable.*

## (a) The gap: **REAL**

The Augury result (market is a non-redundant forecaster, edge in the final hours) is public and
now echoed by practitioners; the **provenance** of that edge has never been decomposed.

- Practitioner claims assert the conclusion without the mechanism: IBKR ("prediction markets
  might already be the best source for today's weather forecast") and wethr note that *"traders
  analyze all models — GFS, Euro, NAM — in real-time."* That sentence is precisely the
  **anticipation channel (H2)** stated as a feature, never tested as a confound.
- No published work localizes a weather market's information to a causal-access channel
  (measure / anticipation / aggregate) against a **matched-cutoff** model ladder, nor tests
  **access-as-sufficient-statistic** (H3) across a de-confounded forecaster pantheon. Each
  *piece* has precedent (below); the **joint decomposition on weather markets does not exist.**

## (b) Precedents — one per hypothesis, method-borrowable

- **Encompassing markets vs other forecasters (H1/H2 method):** Gürkaynak & Wolfers (2005),
  *Macroeconomic Derivatives* — the market-based forecast **encompasses** survey forecasts, and
  survey behavioral anomalies are **absent** in the market. Exactly our encompassing kill-test,
  but macro/survey, no weather, no channel decomposition.
- **Obs-nowcast baseline (H1 rung):** persistence and climatology are the two canonical forecast
  baselines (WeatherBench, arXiv 2002.00469); daily min/max temperature is **nowcastable from
  sub-hourly obs** hours ahead (Nowcasting daily minimum air temperature, PMC4735264). The
  documented diurnal-persistence shape — error rising through ~12 h then falling as the cycle
  returns — is directly the H1 nowcast's remaining-rise term. The rung is well-grounded, not ad
  hoc.
- **Market-leads-forecast (H2 motivation):** the prediction-market literature finds prices tend
  to **lead** official/expert forecasts (Gürkaynak-Wolfers; general price-discovery results). H2
  sharpens "leads" into a falsifiable fork: is the lead *mere anticipation* (later absorbed by the
  models — FRONTRUNNER) or *independent* (never absorbed through settlement — INDEPENDENT)?

## (c) Data-availability — the two NEW dependencies, receipted (2026-07-09)

| dependency | for | source | coverage | receipt |
| --- | --- | --- | --- | --- |
| real-time surface obs (temp) | H1 nowcast rung | IEM ASOS `cgi-bin/request/asos.py` (`data=tmpf`, UTC) | all 7 Augury stations; ≥ 2023-02 → present | NYC hourly at `:51` (2025-07-04 **and** 2023-02-01 = 36 °F — window start); MDW/LAX 1-min; MIA hourly; **DEN/PHL/AUS return data individually** (1-min, some `M`) |
| final-cycle model MaxT | H2 comparator | reused Augury NBM decode | already cached | the freshest same-day NBM TMAX cycle = the ~12Z issuance (Augury Amdt A.1: NBM issues no *new* same-day MaxT after ~12Z) — **already in `results/augury/g3-full-run/nbm_scalars.jsonl`** |

**Caveats found (carried to G1/G2, none gate-blocking):**

1. **IEM ASOS rate-limits bursts** ("Too many requests… slow down") — the DEN/AUS/PHL initial
   zeros were throttle, not missing data. G2 puller needs gentle pacing (as the ECMWF path did).
2. **Report frequency + missing values vary** (hourly METAR at `:51` vs 1-minute ASOS with `M`
   gaps). The nowcast needs only the **latest valid (non-`M`) ob ≤ cutoff** — robust to frequency;
   filter `M`.
3. **H2 same-day-TMAX cadence:** because NBM's last same-day MaxT is ~12Z, the "final cycle"
   comparator for *afternoon* cutoffs is close to Augury's matched cycle; **H2's discriminating
   power is at long lead** (does the dawn market beat the eventual noon model?). Alternatively
   compose a fresher daily-max from the latest hourly NBM 2 m temp — a G1 design choice, more
   pull. Either is pullable.

## (d) Apparatus coverage: **CONFIRMED**

Reused unchanged (verified in Augury): Kalshi candle puller + implied-CDF + strike Brier +
exclusions (`augury_pilot.py`); matched-availability-cutoff selection; NBM/GEFS/ECMWF GRIB decode
(7/7); the ridge-logistic encompassing rig + day-block bootstrap + by-horizon (`augury_g4.py`);
CLI ground truth; frozen G1 constants. New per hypothesis, each small: **H1** = ASOS puller +
nowcast rung `F_obs(θ)`; **H2** = final-cycle comparator (selection over cached NBM); **H3** =
access-feature schema + two-level meta-regression over a city×horizon×era membership panel.

## Gate verdict: **G0 = PASS → proceed to G1 pre-registration**

The gap is real (provenance decomposition of a weather market against a matched-cutoff ladder is
unpublished; the precedents cover each piece separately), and both new dependencies are pullable
with same-day receipts (ASOS obs for all 7 stations back to the window start; final-cycle NBM
already cached). Carry the three caveats into G1/G2. H3 stays flagged **exploratory-strength**
until the membership-panel cell count is fixed (degrees-of-freedom discipline, per the spine).

## Sources

- Gürkaynak & Wolfers (2005), Macroeconomic Derivatives / market-encompasses-survey.
- IBKR, "Prediction markets might already be the best source for today's weather forecast."
- WeatherBench (arXiv:2002.00469) — persistence/climatology baselines.
- Nowcasting daily minimum air temperature — PMC4735264.
- IEM ASOS request service: <https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py>
