# SUNDOG_V_AUGURY — G0 Lit-Pass Memo

> 2026-07-05. G0 pass against [`SUNDOG_V_AUGURY.md`](SUNDOG_V_AUGURY.md): (a) confirm the
> prior-art gap, (b) Roll-1984 lineage, (c) data-availability table with **live pull receipts**
> for every pantheon rung + settlement, (d) scoring tooling. Gate criterion (verbatim from the
> spine): *proceed only if the gap is real AND the rungs are actually pullable.*

## (a) The prior-art gap: **REAL**

Everything public on Kalshi weather markets is **market-vs-reality**; nothing is
**market-vs-model at matched cutoffs**.

- **CalibShi** (Zerve gallery): 8,494 settled `KXHIGHNY` markets; bucket calibration
  (90–100%-priced resolve YES ~98.6%, 0–10% resolve ~1.2%); ECE 0.01624 → 0.00109 after
  isotonic recalibration. This is the spine's "known bucket calibration" — and it has **no
  model side at all**. (Also fixes the G2 sanity-check target: reproduce ~98.6%.)
- **CW Data Solutions**, "Calibration and Skill of the Kalshi Prediction Markets": Brier Skill
  Score vs base rate over 8,476 contracts at 6-week/7-day/24-hour snapshots (BSS ~0.25–0.62).
  Direct read confirms: **"no comparison whatsoever to numerical weather models, CRPS, or
  other meteorological benchmarks."**
- **Practitioner layer** (WeatherEdgeFinder, OutcomeEdge, Lychee guide; open-source
  `polymarket-kalshi-weather-bot` trading on 31-member GFS ensembles): traders *do* compare
  models to prices for edge — but as live dashboards/bots, not a published skill comparison.
  The Lychee guide's color ("traders execute within seconds of model-cycle updates") is
  direct support for the **null mechanism** (market = arb-bot monolith-shadow of the blend),
  which is what makes the encompassing kill-test meaningful rather than rhetorical.
- **Journalism poses our exact question and leaves it open**: Bloomberg 2026-04-13
  ("Prediction markets like Kalshi, Polymarket could improve weather forecasting"),
  Insurance Journal 2026-04-15 ("Weather prediction markets are booming, but can they
  improve forecasts?").
- **Nearest methodological siblings** (method exists; domain differs):
  - arXiv 2605.11220 (May 2026): prediction markets vs the FluSight ensemble on Brier / log
    / CRPS — markets **underperform** the ensemble on every rule. The template for our
    scoring design, and a live prior that `AUGURY_ENCOMPASSED` is a real outcome, not a straw
    null.
  - Fed FEDS 2026-010, "Kalshi and the Rise of Macro Markets": Kalshi MAE ≈ professional
    forecasters at ~150-day horizon (fed funds). Macro, MAE, no matched cutoffs.

**Not found anywhere** (six search angles + direct reads): CRPS-vs-NBM at matched wall-clock
cutoffs; an encompassing regression with the market as a candidate pantheon member; a
horizon-crossover / dominance-by-horizon map. **The contribution stands as specified.**

## (b) Lineage: Roll 1984 → weather derivatives → today

- **Roll 1984 (AER)**: FCOJ futures returns predict *subsequent errors* in NWS central-Florida
  temperature forecasts — the market carries weather information beyond the bureau. Plus the
  famous residual puzzle (weather explains little of FCOJ variance).
- **Boudoukh–Richardson–Shen–Whitelaw 2007 (JFE)**: with the nonlinearity done right
  (freezing threshold), fundamentals explain ~50% of return variation on freeze days —
  the market-reads-weather link is real where theory says it should bind.
- **"What explains the orange juice puzzle" (IRFA 2015)**: sentiment carries the non-weather
  variance — the lineage's caution that price ≠ pure information.
- **Campbell & Diebold 2005 (JASA)**: density forecasting of daily temperature for weather
  derivatives; beyond ~8–10 days every point forecast reverts to climatology — pins our
  long-horizon expectation (market and models should *both* collapse to climatology at day 3+;
  the interesting axis is the short end).
- **Jewson 2003 (Met. Apps)**; CME temperature-futures pricing (JBF 2010): forecasts used *in
  pricing* and risk premia — the derivatives literature scores returns and prices, never the
  market as a **probabilistic forecaster** against the operational model ladder.

## (c) Data-availability table — **every rung pullable, each with a live receipt (2026-07-05)**

| rung | source | coverage | receipt (pulled today) |
| --- | --- | --- | --- |
| market (aggregate) | Kalshi REST, **no auth**; live + `/historical/` split at rolling cutoff (2026-05-07 at probe) | NYC launch bracketed **Jul–Dec 2021** (`HIGHNY-21JUL04` absent, `HIGHNY-21DEC01` present) → present | **1,641** 1-min candles, `KXHIGHNY-26JUL04-T98` (live ep); **289** 1-min candles, `HIGHNY-22JUL04-T86` (historical ep). Both carry `yes_bid`/`yes_ask` OHLC + volume + OI — exactly the midpoint-CDF inputs |
| settlement | NWS CLI via IEM `GET /json/cli.py?station=KNYC&year=YYYY` | verified 2020→present | KNYC 2020-01-01 daily report JSON (high/low/times + raw-product link). Kalshi rules text names the same product: "Central Park … NWS Climatological Report (Daily)" — **station match confirmed at the source** |
| statistics (NBM) | `s3://noaa-nbm-grib2-pds` (grib2), `noaa-nbm-pds` (COG); hourly cycles | **2020-05-18** → present (earliest prefix listed) | 2020-05-18/00z master files listed; 2026-07-01/12z core files listed **with `.idx` sidecars**. 2020-era files have **no `.idx`** (G2 flag below) |
| statistics, point path | IEM MOS archive, `model=NBS` (NBM hourly station bulletin) | verified KNYC **≥ 2020-09-01** | NBS bulletins returned for 2020-09-01, 2021, 2022, 2023, 2024 runtimes |
| physics (GEFS) | `s3://noaa-gefs-pds`, 4 cycles/day | **2017-01-01** → present (earliest prefix listed) | earliest prefixes `gefs.20170101/…` listed. Note: registry blurb says 21 members / 16 days — GEFSv12 is 31 members since Sept 2020; blurb is stale |
| human / MOS | IEM MOS archive, `model=MEX` / `GFS` | verified KNYC 2022 (IEM documents much deeper) | MEX + GFS MOS bulletins returned for 2022-07-03T12Z. NDFD grids not probed — MOS covers the rung; NDFD is an optional G2 add |
| physics+AI (ECMWF) | AWS `ecmwf-forecasts` — **accumulates** (unlike ECMWF portal's rolling ~12 runs) | **2023-01-18** → present (earliest prefix listed); 0.4°-beta early, 0.25° from Feb 2024; AIFS operational 2025-02-25 | earliest prefix `20230118/00z/0p4-beta/` listed |

**Series seam (G1 must pin):** NYC traded as `HIGHNY` through at least mid-2023
(`HIGHNY-23JUL04` exists) and as `KXHIGHNY` by mid-2025 (`KXHIGHNY-25JUL04` exists;
`KXHIGHNY-24JUL04` does **not**) — the KX rename splits the archive. **Pull both series and
verify the seam date at G2.**

**Station roster:** 286 climate/weather series enumerated. Legacy-depth cities ≈ 8 (NYC,
Chicago, Denver, LA, Houston, Austin, Miami, Philadelphia) + a newer `KXHIGHT*` cohort
(Boston, Phoenix, Las Vegas, Dallas, San Antonio, Minneapolis, OKC, New Orleans, SFO, …).

**Window arithmetic (reconciled):**

- Primary encompassing (market vs NBM): binding start = market launch ≈ **Dec 2021** →
  ~4.5 years ≈ **~1,650 NYC station-days** pre-exclusion; NBM (2020-05), GEFS (2017-01),
  CLI, MOS all cover it.
- Full ladder incl. ECMWF: binding start = **2023-01-18** → ~3.5 years; × ≥8 legacy cities
  → **>10,000 candidate station-days**.
- Either window clears any plausible G1 `N`.

## (d) Scoring tooling: **CONFIRMED**

- `properscoring` (`crps_ensemble` smoke test = 0.5714 on a toy 7-member ensemble) and
  `scoringrules 0.11.0` (has `crps_quantile` — CDF-at-the-traded-strikes scoring) both
  installed and ran clean in a scratch venv on Python 3.14 / Windows.
- DM test + block bootstrap = numpy/statsmodels-standard; no exotic dependency.

## Flags carried to G1 / G2 (found in this pass; none gate-blocking)

1. **Early-era liquidity:** the 2022 receipt (289 candles over 2 days, wide 14/19¢ book in
   the sample) shows thin early books — the liquidity-exclusion rate will be era-dependent.
   Pre-register **era-stratified exclusion reporting** in G1.
2. **GRIB2 on Windows:** Herbie/cfgrib needs eccodes — the one fragile dependency.
   Mitigations, in order: IEM NBS/MOS bulletins for all point rungs (verified above); COG
   bucket (`noaa-nbm-pds`); WSL/conda for GRIB decode. Decide at G2, not before.
3. **2020-era NBM `.idx` absence:** byte-range subsetting unavailable at the archive's start;
   moot if the market window (≥ Dec 2021) binds — verify the `.idx` era boundary at G2 only
   if pre-2022 NBM is ever needed.
4. **AIFS rung is thin** (operational Feb 2025): keep ECMWF-ENS as the ladder rung; treat
   AIFS as declared-exploratory or drop at G1.
5. **Historical cutoff rolls forward** (2026-05-07 at probe): the puller must route each
   market to live vs `/historical/` endpoints by settle date at pull time.

## Gate verdict: **G0 = PASS → proceed to G1 pre-registration**

The gap is real (nothing published on CRPS-vs-NBM at matched cutoffs, encompassing, or the
crossover), and every rung + settlement is pullable with a same-day receipt. The flu-market
paper (arXiv 2605.11220) is the closest methodological precedent and should be cited in G1 as
the template — and as evidence the kill branch is live.

## Sources

- CalibShi (Zerve): <https://www.zerve.ai/gallery/85cce830-f612-4b23-8b78-34d7da65a2c6>
- CW Data Solutions: <https://www.cwdatasolutions.com/post/calibration-and-skill-of-the-kalshi-prediction-markets>
- Lychee guide: <https://lycheedata.com/guides/kalshi-weather-prediction-markets-analysis>
- Trading bot: <https://github.com/suislanchez/polymarket-kalshi-weather-bot>
- Bloomberg 2026-04-13: <https://www.bloomberg.com/news/newsletters/2026-04-13/prediction-markets-like-kalshi-polymarket-could-improve-weather-forecasting>
- Insurance Journal 2026-04-15: <https://www.insurancejournal.com/news/national/2026/04/15/866011.htm>
- Flu-markets paper: <https://arxiv.org/abs/2605.11220>
- Fed FEDS 2026-010: <https://www.federalreserve.gov/econres/feds/files/2026010pap.pdf>
- Roll 1984: <https://www.anderson.ucla.edu/documents/areas/fac/finance/1984-6.pdf>
- BRSW 2007: <https://www.sciencedirect.com/science/article/abs/pii/S0304405X06001450>
- OJ puzzle / sentiment: <https://www.sciencedirect.com/science/article/abs/pii/S138641811500066X>
- Campbell & Diebold 2005: <https://www.tandfonline.com/doi/abs/10.1198/016214504000001051>
- Jewson 2003: <https://rmets.onlinelibrary.wiley.com/doi/abs/10.1017/S1350482703001099>
- CME temp futures pricing: <https://www.sciencedirect.com/science/article/abs/pii/S0378426609003306>
- Kalshi API docs (historical): <https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks>
- NBM on AWS: <https://registry.opendata.aws/noaa-nbm/> · GEFS: <https://registry.opendata.aws/noaa-gefs/> · ECMWF: <https://registry.opendata.aws/ecmwf-forecasts/>
- IEM CLI/MOS services: <https://mesonet.agron.iastate.edu/json/cli.py> · <https://mesonet.agron.iastate.edu/api/1/docs>
