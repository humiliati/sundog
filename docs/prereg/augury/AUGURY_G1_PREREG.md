# AUGURY G1 — Pre-Registration (market-vs-pantheon, matched cutoffs)

Parent / boundary documents:

- [`SUNDOG_V_AUGURY.md`](../../SUNDOG_V_AUGURY.md) (lane spine; gate ladder G0→G5)
- [`AUGURY_LIT_PASS_MEMO.md`](../../AUGURY_LIT_PASS_MEMO.md) (G0, PASSED 2026-07-05 — gap + rung receipts;
  all availability claims below carry that memo's receipts)
- Method template (cited, not copied): arXiv 2605.11220 (markets vs FluSight ensemble on proper scores —
  evidence the kill branch is live)

Filed: **2026-07-06 (PT)**

Status: **PREREG FILED; EXECUTION NOT ADMITTED.** No pull, no score. G2 (single-station pilot) requires a
tooling freeze-marker amendment to this file (runner, settlement audit, exact command) before any scoring
run; G3 (full run) requires a second freeze marker after G2's artifacts land. Constants in §1–§7 are frozen
with this file; any change after G2 begins reading market data = a named amendment, and any change after G3
begins = a protocol violation, reported as such. `DOCS_NO_PUBLISH` until G5 per the spine.

## §0 — Claims under test

- **Hook (weak form):** Kalshi implied temperature distributions, snapshotted at matched information
  cutoffs, achieve lower CRPS than NBM at the same cutoffs, in ONE pre-registered lead-time band (§3).
- **Real object (the substance):** the market is a non-redundant pantheon member — its probability
  survives an encompassing regression against NBM (primary) and the ladder (exploratory), with the margin
  horizon-localized to the short lead where the market carries real-time observations the model cycle
  does not.
- **Null:** market = lossy compression of the blend; encompassing coefficient 0. An honest, clean result.

## §1 — Universe (frozen)

**Primary stations (7) and settlement mapping** (mapping read from Kalshi `rules_primary` text 2026-07-05/06;
IEM CLI coverage verified 366/366 days for 2024 for every station):

| city | series lineage (legacy → current) | CLI station | LST offset |
| --- | --- | --- | --- |
| New York | `HIGHNY` → `KXHIGHNY` | KNYC (Central Park) | UTC−5 |
| Chicago | legacy TBD-at-G2 → `KXHIGHCHI` | KMDW (Midway) | UTC−6 |
| Denver | legacy TBD-at-G2 → `KXHIGHDEN` | KDEN | UTC−7 |
| Los Angeles | legacy TBD-at-G2 → `KXHIGHLAX` | KLAX (LA Airport) | UTC−8 |
| Austin | `HIGHAUS` → `KXHIGHAUS` | KAUS (Bergstrom) | UTC−6 |
| Miami | `HIGHMIA` → `KXHIGHMIA` | KMIA (Miami Intl) | UTC−5 |
| Philadelphia | legacy TBD-at-G2 → `KXHIGHPHIL` | KPHL (Phila Intl) | UTC−5 |

- The KX-rename seam (G0 flag): each city's full lineage (legacy ticker, seam date) is charted at G2 from
  the API (**seam chart = required G2 artifact**); both series are pulled. Station mapping is re-verified
  **per era from settled markets' rules text**; if any era names a different station, that era scores
  against *its* station (no exclusion, no averaging across stations).
- **Excluded from primary, declared exploratory:** Houston (triple-split lineage `KXHIGHOU` /
  `KXHOUHIGH` / `KXHIGHTHOU`) and the entire newer `KXHIGHT*` cohort (thin history).
- **Event semantics** come from machine-readable `cap_strike` / `floor_strike` / strike-type fields,
  never parsed prose. G2 must pass a **settlement audit**: 50 randomly sampled settled markets, API
  `result` vs CLI-derived outcome, required match 50/50.

**Windows (frozen calendars, civil local event days):**

- Primary (market vs NBM): **2022-01-01 → 2026-06-30** = 1,642 event-days/station; 7 stations ⇒ ceiling
  11,494 station-days before launch dates and exclusions.
- Exploratory full ladder (adds GEFS/ECMWF/MOS): **2023-02-01 → 2026-06-30** = 1,246 event-days/station
  (ECMWF AWS archive begins 2023-01-18).
- AIFS: **dropped** (operational 2025-02 only; G0 flag #4 discharged here).

## §2 — Matched-information cutoffs (the interpretability crux, frozen)

- Snapshot grid: every hour on the hour UTC across each market's life.
- At cutoff T, every rung uses the latest issue whose **availability timestamp ≤ T**:
  S3 `LastModified` for GRIB rungs; IEM product timestamp for bulletins; candle `end_period_ts` for the
  market. **Nominal cycle time is never used as availability.** This single rule is what makes a
  final-hours margin an information result rather than a staleness artifact.
- Market state at T: the last 1-min candle at-or-before T per strike; a strike is quotable at T only
  under §5's validity rules.

## §3 — The ONE primary band (diurnal-defined, frozen rule)

- `t_peak(station, month)` = climatological median local-standard-time of the daily high, computed from
  IEM CLI `high_time` over **2015-01-01 → 2019-12-31** (disjoint, pre-market). Computed once at G2,
  frozen as a table artifact **before any market data is scored**; the rule (not the numbers) is frozen
  here. CLI local times are converted to LST (fixed offsets, §1) during ingest.
- **Primary band** B(station, day) = the 11 hourly cutoffs T with
  `t_peak − 12h ≤ T ≤ t_peak − 2h` (LST), i.e. the mechanistic "final-12h, high-not-yet-in" window.
- The full hourly dominance-by-horizon map over all cutoffs is **declared exploratory** (spine crux #2).

## §4 — Forecast objects (frozen constructions)

- **Market CDF:** per strike θ, p = bid-ask midpoint of the YES side mapped to `P(high > θ)` via the
  strike fields; exceedance curve over the day's traded threshold set S(s,d); isotonic monotonization
  (pool-adjacent-violators, order-preserving); **no tail extension** — scoring never leaves S(s,d).
- **NBM (primary comparator):** `qmd` percentile product (verified on AWS with `.idx` from 2022-01-01),
  nearest land grid point to the station, percentile function linearly interpolated → `F_NBM(θ)` at the
  same S(s,d). Known caveat, frozen handling: NBM MaxT valid windows may not cover the full CLI civil
  day; **no adjustment, no exclusion** — days whose CLI high-time falls outside the MaxT window are
  counted and reported as a diagnostic (exclusion would bias toward the model).
- **GEFS / ECMWF-ENS (exploratory ladder):** member-wise civil-day max of 2 m temperature at nearest
  grid point → empirical CDF at S(s,d). Raw, uncalibrated — enters only the ridge encompassing (§6).
- **MOS/human rung (exploratory ladder):** MEX/NBS deterministic MaxT as a margin covariate
  `(MaxT − θ)`; never a primary object.

## §5 — Validity + liquidity exclusions (frozen constants)

A strike is **valid at T** iff: two-sided book, spread ≤ 4¢, quote age ≤ 60 min.
A (station-day, cutoff) is **valid** iff ≥ 4 valid strikes AND cumulative event volume by T ≥ 250
contracts. Exclusion rates are reported **era-stratified (2022 / 2023 / 2024 / 2025 / 2026-H1)** — the
G0 receipts show thin 2022 books; era asymmetry is expected and must be visible, not smoothed.

## §6 — Scoring + inference (frozen)

- **Primary score:** strike-set mean Brier at S(s,d) — the uniform-weight discrete threshold
  decomposition of CRPS at exactly the strikes the market trades; identical strike set and identical CLI
  outcome for both rungs. Δθ-weighted variant = sensitivity diagnostic only.
- **Primary test 1 (score):** per station-day, mean score differential (market − NBM) over B(s,d);
  Diebold–Mariano via **circular moving-block bootstrap over calendar days** (all stations share a
  day's block — spatial correlation respected), block length 7 days, 10,000 resamples, one-sided
  (market < NBM), α = 0.05.
- **Primary test 2 (kill test):** pooled logistic encompassing at valid strikes in B:
  `1{high > θ} ~ logit(p_NBM) + logit(p_mkt) + station fixed effects`, probabilities clipped to
  [0.01, 0.99]; 95% CI for β_mkt from the same day-block bootstrap. Survives iff the CI is entirely > 0.
- **Exploratory (declared, never adjudicating):** dominance-by-horizon map; full-ladder **ridge**
  encompassing by horizon (GEFS/NBM/ECMWF collinearity handled by penalty, never naive OLS); reliability
  diagrams; era-stratified everything; tradable-edge after fees (separate, weaker claim — spine fence);
  MaxT-window mismatch diagnostic; CalibShi-style bucket calibration.

## §7 — Verdicts (precedence in table order; tokens final)

| verdict | condition |
| --- | --- |
| `AUGURY_SAMPLE_COLLAPSE` | valid primary-band station-days < 500 after exclusions — no adjudication; lane re-poses or dies honestly (spine kill #3) |
| `AUGURY_MARGIN_CONFIRMED` | DM one-sided p < 0.05 **AND** β_mkt 95% CI > 0 |
| `AUGURY_ENCOMPASSED` | neither holds — market adds nothing; clean negative, reported as such |
| `AUGURY_GAP` | exactly one of the two holds — muddy; no page |

No gate, constant, band, station set, or window may be retuned after G2 first reads market data. The
exploratory map may not be promoted to a claim regardless of what it shows (fishing fence, spine crux #2).

## §8 — G2 pilot (single station = NYC), execution gating

G2 runs on KNYC only and must land these artifacts before a G3 freeze marker may be filed:

1. Series-seam chart (all 7 cities: legacy ticker, seam date, per-era station from rules text).
2. Settlement audit 50/50 (§1).
3. `t_peak` table from 2015–2019 CLI, frozen (§3).
4. Implied-CDF construction receipts (isotonic before/after examples; monotonicity violations counted).
5. NBM qmd decode at KNYC point, with the availability-timestamp join (§2) demonstrated.
6. Bucket-calibration sanity: NYC 90–100%-priced markets resolve YES in [97%, 100%] (CalibShi anchor
   ~98.6%); miss ⇒ construction bug until proven otherwise.
7. Era-stratified exclusion-rate table (§5) + MaxT-window mismatch frequency (§4).

Reserved paths (freeze-marker amendments will record hashes + exact commands):
runner `docs/prereg/augury/augury_pilot.py`; wrapper + npm `augury:g2:pilot`, `augury:g3:run`;
results `results/augury/g2-pilot-knyc/`, `results/augury/g3-full-run/` (gitignored); constants mirror
`docs/prereg/augury/g1_constants.yaml` (sha256 recorded at the G2 freeze; this document is canonical).

## §9 — Public language

Before any receipt: "A pre-registration for scoring weather prediction markets against the operational
model ladder at matched information cutoffs is filed. Nothing has been pulled or scored."
After receipts: only §7 verdict language. The information claim (raw midpoints) and the tradable-edge
claim (fees, liquidity) are never merged; no trading-system or investment-advice framing anywhere.
G4 (allelopathy determining-shadow-set read) and G5 (public page) remain gated on `AUGURY_MARGIN_CONFIRMED`
exactly as the spine specifies.

## §10 — Self-consistency ledger (per the pre-reg discipline)


- Windows: 2022-01-01→2026-06-30 = 365+365+366+365+181 = **1,642** days; 2023-02-01→2026-06-30 =
  334+366+365+181 = **1,246** days. Station-day ceiling 7 × 1,642 = **11,494**.
- Band = `t_peak−12h … t_peak−2h` hourly = **11 cutoffs** (§3 range and count agree).
- Constants appear once each: spread ≤ 4¢; age ≤ 60 min; ≥ 4 strikes; ≥ 250 contracts; clip [0.01,
  0.99]; block L = 7; B = 10,000; α = 0.05; sample floor 500; audit 50/50; calibration band [97%, 100%];
  `t_peak` source years 2015–2019.
- Verdict tokens here = spine tokens plus `AUGURY_SAMPLE_COLLAPSE` (names the spine's third kill
  condition, previously implicit).

---

## Amendment A — Tooling Freeze Marker + §4-NBM Comparator Respecification (2026-07-06 PT)

Append-only. Discharges §8's execution gating for the KNYC pilot. **No market price data has
been read for scoring; no constant in §1–§7 changed.** One construction in §4 is respecified
here because the product it named does not exist for the primary window — receipts below.

### A.1 §4-NBM respecification (forced by product availability, receipted)

- **qmd is dead for the primary window:** the 2022-era qmd files carry APCP percentiles ONLY
  (idx inventory of `blend.20220101/12/qmd/...f036.co.grib2.idx`: 316 records, all APCP).
  TMP percentiles appear in later eras; TMAX percentiles appear in NO era checked (f006–f050
  sampled, 2026-07-01 t12z).
- **NBP station bulletin is dead for the primary window:** TXNMN/TXNSD/TXNP1–P9 rows exist in
  the V5.0-era bulletin (2026-07-01) but are ABSENT for KNYC in 2022-01-01 (V4.0: wind/precip
  only), 2023-06-01, 2024-06-01, and 2025-06-01 bulletins.
- **Bound comparator (era-uniform 2022→2026, receipted at both ends):** NBM `core` **TMAX
  blend mean + "ens std dev"** (2 m, 12-hour max window ending 00Z UTC of event-day+1),
  nearest grid point to the station; `F_NBM(θ) = P(Normal(μ, σ²) > θ)`. Both records verified
  present with `.idx` in 2022-01-01 t12z f036 and 2026-07-01 t12z f036 / t08z f016.
- **Issuance-cadence fence:** cycles after the window opens (13Z+ for a 12Z-opening window)
  carry no same-day TMAX record — the freshest same-day MaxT issue at an afternoon cutoff is
  the ~12Z cycle (verified: smoke join at 2026-06-15 16Z selected t12z, available 13:08:35Z).
  This is the operational product's real cadence, honestly joined under §2 (availability
  timestamps). Interpretation fence: a final-hours market margin is a claim about the market
  vs the operational NBM MaxT product **as issued**; an obs-augmented model-side forecaster is
  a pre-named exploratory rung, never the primary. The per-era cycle/window availability map
  is a pilot artifact (`nbm_cycle_map.json`).
- The Gaussian shape is a named assumption of the comparator construction (the operational
  product ships mean + spread, not quantiles, uniformly across the window). Reliability
  diagnostics (§6) apply to it identically.

### A.2 Tooling (frozen with these hashes)

| file | sha256 |
| --- | --- |
| `docs/prereg/augury/augury_pilot.py` | `529583c9aa4a6b08cdb4bc76aa83225b5e885f25be1d8a78fdbcdb250be08e46` |
| `docs/prereg/augury/g1_constants.yaml` | `b5d8dc14dcf793f526903b472d31e05d11a38675fd26bc56c06975f2609e553e` |
| `scripts/augury-pilot.mjs` | `8354f69aaab8c7344488e760bf119d10cba3ac5662fb6f2ded5f04eebfc5fe85` |

Deterministic implementation choices are frozen in the runner's module docstring (quote =
last two-sided 1-min candle within the age window; exceedance from ladder bins + PAV;
strike-semantics discovery seed 20260705 → bind → audit seed 20260706 disjoint; availability
= max(S3 LastModified of .grib2, .idx); CLI local→LST DST rule; t_peak median rule). npm:
`augury:g2:selftest`, `augury:g2:smoke`, `augury:g2:pilot`. Python env: `.venv-augury/`
(3.12, `pip install numpy eccodes`; gitignored). Results path `results/augury/g2-pilot-knyc/`
(gitignored). Cache is append-only under `results/augury/g2-pilot-knyc/cache/`.

### A.3 Verification receipts (pre-admission, non-binding)

- **selftest 33/33** (PAV, exceedance builder, Brier, CLI time parse + DST, strike-semantics
  variants, band construction 11 cutoffs 08–18Z for a 15-LST peak, constants-vs-prereg
  cross-check).
- **smoke**: Kalshi candles 650 (settled `KXHIGHNY-26JUL05-T91`); historical cutoff endpoint
  live (2026-05-07, rolls); IEM CLI KNYC 2015 = 365 days; NBM matched join + byte-range
  eccodes decode at KNYC point → **μ = 77.81 °F, σ = 2.65 °F** (2026-06-15, plausible).
- **t_peak table computed and FROZEN before any scoring**:
  `sha256(tpeak_table.json) = 963b39738784c2b39ab4be65a9c15a155e8305eb1cff1734bd5c33f1724948b2`
  (the pilot stage prints the file's sha at start; it must match this value).

### A.4 Admitted command (exact, unchanged)

```
npm run augury:g2:pilot        # = node scripts/augury-pilot.mjs all --admitted
```

Runs: tpeak (deterministic re-derivation from immutable+cached CLI; sha must equal A.3) →
seam chart (7 cities) → nbmscan (cycle map) → the binding KNYC pilot (settlement audit
50/50, CDF receipts, NBM join demo, bucket calibration vs [97, 100]%, era-stratified
exclusion + MaxT-mismatch tables → `pilot_summary.json`). Estimated ~10–11k Kalshi candle
calls (~45–75 min), resumable via cache. G3 requires a further freeze marker after these
artifacts land.

---

## Amendment B — Pilot Execution, Tooling Fixes, G2 Adjudication (2026-07-06 PT)

Append-only. The owner executed A.4 (first binding run); the pilot surfaced tooling defects,
which were fixed and the pilot re-run (candle cache reused; markets/CLI/S3 inputs unchanged
and immutable). **No §1–§7 constant changed at any point.** Final runner
sha256 `e2c7b780c1eaba8e0919d284d2f2d162be545ee97c274327fc3293302ac92c4e`; wrapper
`de55310b7d7c0db39d7a67503bd9379704d09943ade1a0b2fcc70f16aa032faf`; selftest 44/44.

### B.1 Tooling defects found by run 1 and fixed (fix log)

1. **Legacy strike schema:** pre-~2024 market objects carry NO `floor_strike`/`cap_strike`/
   `strike_type` anywhere (listing or detail endpoint) — bounds now normalize to
   half-integer thresholds from fields when present, else from the rules text (three
   phrasings: "is [strictly ]greater than X°" / "is less than X°" / "is between X-Y°|X° and
   Y°"). Modern semantics verified against settlements: `between` inclusive of both integer
   ends, `greater`/`less` strict (witness: T90 with CLI high exactly 90 settled NO). The
   half-integer normalization dissolves Amendment A's inclusive/exclusive variant machinery.
2. **Two-sided book test was too loose:** `bid=0 / ask<1` books passed. Now strict interior:
   `0 < bid ≤ ask < 1`.
3. **Quote persistence:** 1-min candles are change-events, not continuous minutes. Standing
   book at cutoff = last candle's close; if older than the 60-min window it is admitted only
   with a **persistence witness** — the next candle *opens* on the identical book. (§5's
   60-min constant unchanged; this is its honest reading over event-sparse candles.)
4. **`expiration_value` is an empty string** on legacy markets — treated as absent, reported
   as its own category (it is never wrong where present: 39 match / 61 absent / 0 mismatch).
5. Wrapper now defaults to `.venv-augury` Python (run 1's NBM joins all failed on a bare
   `python` without eccodes); seam station regex now spans commas ("Central Park, New York").
6. Kalshi's historical DB returns the whole series lineage under the CURRENT ticker
   (`KXHIGHNY` pagination includes `HIGHNY-*` markets; querying `HIGHNY` returns nothing).

### B.2 Final pilot receipts (run 3; `pilot_summary.json` sha256 `facae7e0…d804e`)

| §8 artifact | outcome |
| --- | --- |
| 1. seam chart | **LANDED.** NYC+CHI 2021-09→; AUS+MIA 2023-06→; DEN+PHL 2024-12→; LAX 2025-02→; KX-seam 2024-10/11 for all legacy cities; station text consistent per city every year (KNYC = "Central Park, New York" throughout) |
| 2. settlement audit | **PASS 50/50** (validation sample also 50/50; expiration-vs-CLI 39 match / 61 absent / 0 mismatch) |
| 3. t_peak table | **FROZEN, unchanged** (`963b3973…4948b2`) |
| 4. CDF receipts | **LANDED**, 6/12 OK; 6 = TOO_FEW_VALID at the mid-band cutoff (consistent with §5 exclusions below, not a construction failure; isotonic violations 0 on OK days) |
| 5. NBM join demo | **PASS 9/9** across 2022/2025/2026 (μ 85.4/32.1/77.8 °F, σ 0.9–2.8 °F, availability lags 4–299 min, afternoon cutoffs correctly join the ~12Z issue) |
| 6. bucket calibration | **BAND MISSED, POWER-LIMITED:** top bucket 96.88% (62/64), 95% CI [92.6, 100] **contains the 98.6% anchor**; full decile curve monotone (3.5%→96.9%). No construction bug found (audit 50/50 independently validates the outcome mapping); n is small because the pinned snapshot (first valid band cutoff, strict two-sided) is far more restrictive than CalibShi's whole-history sample |
| 7. exclusion + mismatch tables | **LANDED.** Era structure is the pilot's main empirical finding (B.3) |

### B.3 Empirical findings for G3 planning (reported, not retuned)

- **Liquidity era-structure (NYC, primary band):** valid cutoff rates 2022 7.9% / 2023 0.6%
  / 2024 1.3% / 2025 31.8% / 2026-H1 61.2%. Failure decomposition: one-sided books
  (`fail_book`) + spread > 4¢ dominate; staleness ≈ 0 once persistence is witnessed. 2022
  additionally lists only ~3.4 strikes/day, so the ≥4-valid-strikes rule binds mechanically
  there (6/day from 2023 on).
- **Projected primary-band sample:** NYC ≈ 262 valid station-days; with the other six
  cities' launch dates (B.2 item 1) the pooled projection is ~1,300–1,800 — comfortably
  above the §7 floor of 500, but **dominated by 2025–2026**. Era-stratified reporting (§5)
  will carry that visibly.
- **MaxT-window mismatch:** CLI daily high falls outside the 12Z→00Z window on ~15% of NYC
  days (winter-skewed) — larger than assumed; stays a diagnostic per §4, now with a number.
- ~15% of legacy audit rows have absent `expiration_value`; settlement verification for
  those rests on the (passing) result-vs-CLI mapping.

### B.4 G2 adjudication

Artifacts 1–5 and 7 land clean; artifact 6 misses its pinned band at 62/64 with the anchor
inside the CI and no bug found under diagnosis. Per §8 ("miss ⇒ construction bug until
proven otherwise") the proof-otherwise is on record (B.2). **The decision to accept that
reading and admit G3 spec-work (a G3 freeze-marker amendment) is the owner's**; the runner
and constants stay frozen at the hashes above meanwhile.

---

## Amendment C — G3 Freeze Marker (2026-07-06 PT)

Append-only. **Owner accepted B.4 and admitted G3 on 2026-07-06** ("squared away admit
G3"). This amendment freezes the G3 tooling and stages the binding full run. No §1–§7
constant changed; the pilot runner is imported unmodified at its Amendment-B hash
(`e2c7b780…`, re-verified at filing).

### C.1 G3 tooling (frozen with these hashes)

| file | sha256 |
| --- | --- |
| `docs/prereg/augury/augury_g3.py` | `cf57dcbdc71eb65c64eadbdff79bf7274e8968ca71fa2c96c026e913b16d52f7` |
| `scripts/augury-g3.mjs` | `8c30c31c30c3d91a38a1de14d71aa548d4a804e23111c58c0c092dc70cb8fe15` |

npm: `augury:g3:selftest` / `augury:g3:plan` / `augury:g3:run`. Results path
`results/augury/g3-full-run/` (gitignored); resumable via the pilot's HTTP cache + a G3
NBM scalar cache (`nbm_scalars.jsonl`). G3 selftest 6/6 (IRLS coefficient recovery, MBB
determinism, logit clip, Φ tail, scoring identity).

### C.2 Implementation pins (§6/§7 operationalized; frozen in the runner docstring)

- **Station-day counts as valid** iff ≥ 1 of its 11 band cutoffs is valid (§5); scores
  average over valid cutoffs; the §7 floor (500) counts these station-days; the
  valid-cutoffs-per-day distribution is reported.
- **DM**: pooled mean of station-day differentials; circular moving-block bootstrap over
  calendar days (block 7, B = 10,000, seed 20260707; a resampled day carries all its
  stations); DM stat = mean/se_boot; one-sided p = Φ(stat).
- **Encompassing**: rows = (station-day, band cutoff, valid strike);
  `1{high>θ} ~ logit(F_NBM) + logit(F_mkt) + station FE + const`, clip [0.01, 0.99];
  IRLS (ridge 1e-8); β_mkt 95% percentile CI over **2,000** day-block bootstrap refits
  (seed 20260708; reduced from 10,000 for refit cost — pinned here, pre-run).
- **Per-city settlement audit** inside the run: n = 20/city, seed 20260707, gate 20/20
  per city; a failing city ABORTS the run before scoring (§1 station-match discipline).
- **NBM**: issue union across all stations' band joins; parallel S3 fetch (6 threads,
  unthrottled — AWS open data), serial eccodes decode (thread-safety), one decode per
  issue serving all 7 stations (coordinates pinned in `STATION_LL`).
- Verdict computed exactly per §7 precedence into `g3_result.json`; exploratory outputs
  (dominance-by-offset, era/station tables, reliability bins) are declared-exploratory.

### C.3 Plan receipts (dry enumeration, 2026-07-06)

| station | markets | event-days | window |
| --- | --- | --- | --- |
| KNYC | 8,908 | 1,641 | 2022-01-01 → 2026-06-30 |
| KMDW | 8,917 | 1,641 | 2022-01-01 → 2026-06-30 |
| KAUS | 6,855 | 1,143 | 2023-05-11 → |
| KMIA | 6,872 | 1,146 | 2023-05-11 → |
| KDEN | 3,528 | 588 | 2024-11-20 → |
| KLAX | 3,252 | 542 | 2025-01-05 → |
| KPHL | 3,528 | 588 | 2024-11-20 → |

Totals: 41,860 markets, 7,289 station-days pre-exclusion; NYC-liquidity-profile projection
≈ 1,300–1,700 valid station-days (floor 500). Estimated binding-run cost: ~33k new Kalshi
candle pulls (~2–3 h, throttled), NBM ~16–21k issue decodes ≈ 75–95 GB transfer (~2–4 h,
scalar-cached so re-runs are free), score+adjudicate ≤ 1 h CPU. **~5–7 h wall, fully
resumable.**

### C.4 Admitted command (exact, unchanged)

```
npm run augury:g3:run        # = node scripts/augury-g3.mjs all --admitted
```

Stages market → audit → nbm → score → adjudicate; emits `g3_result.json` with the §7
verdict. Adjudication language after the run is restricted to §7 tokens + §9.

### C.5 Audit-sampling clarification (2026-07-06, post first firing)

The first firing of C.4 aborted correctly at the audit gate: the KMIA sample drew
`KXHIGHMIA-24NOV05-B82.5`, whose event day (2024-11-05) has **no CLI row on IEM** — an
isolated archive gap (KMIA coverage is 360–365 days/yr; exactly that one day missing in
Nov 2024), not a semantics or station-match failure. Clarification (no §1–§7 change): the
audit samples only from CLI-covered markets — a missing CLI day is a ground-truth
**coverage gap**, unscoreable and already skipped by scoring, now counted and reported per
city (`cli_gap_days`). Re-run: **all 7 cities 20/20**; gap days across the full universe =
12 of 7,289 station-days (NYC 2, MIA 8, LAX 2, others 0; ≈0.16%). Updated runner sha256:
`66a73aceed1fa46830d7bc98e440b4e177265bdd908a1265ba113427653d2fe3` (supersedes C.1's;
wrapper unchanged). The C.4 command is unchanged and remains admitted.

---

## Amendment C.6 — NBM decode-path fix + G3 execution + adjudication (2026-07-07 PT)

Append-only. The first binding `nbm` decode crashed (OOM / hard exit); the fix below made the
decode fast and stable, the full run then completed and adjudicated. **No §1–§7 constant
changed;** the NBM extracted values are provably identical to the original `find_nearest`
reference (7/7, both eras). Final runner sha256:
`86e86ff039b9c7b2f1e00fc0b2f708ddd6a871e72b30cfaa52b11cbe84689775` (supersedes C.5; pilot
unchanged at `e2c7b780…`).

### C.6.1 The decode fix (root cause + resolution, receipted)

- **Root cause:** `codes_grib_find_nearest` rebuilds a spatial KD-tree **per GRIB message**
  (~2.3 s each) — the entire time cost (~97 h projected) *and* the OOM source. `ThreadPoolExecutor.map`
  over the whole todo also eagerly buffered every fetched message → memory flood.
- **Resolution:** extract each station's value by **grid index**, computed **once per grid
  geometry** via pyproj Lambert projection **+ the boustrophedon odd-row flip** (NBM core.co has
  `alternativeRowScanning=1` — odd rows stored right-to-left; this was the reason naïve index /
  `find_nearest.index` / `get_array('latitudes')` all disagreed with `find_nearest` for exactly
  the odd-row stations). Validated **7/7 against `find_nearest` in both eras** (v4-2022
  `core.f036`, v5-2026 `core.f012`; identical grid geometry). The runner re-validates 7/7 at
  first decode and aborts on any disagreement. Decode ~2.3 s → ~0.4 s/message, no KD-tree, no
  memory growth. Fetch: combined the adjacent mean+`ens std dev` records into one range request
  per issue; unthrottled S3 GET (AWS open data) at 12-way concurrency; per-record flush for
  crash-safe resume. New dep: `pyproj` 3.7.2 (`.venv-augury`).
- **Run:** `nbm` 10,990 issues in ~2.05 h (bandwidth-bound, ~49 GB), `score` 84,923 strike rows
  in 300 s, `adjudicate` 160 s. Fully resumable throughout.

### C.6.2 Verdict — `AUGURY_MARGIN_CONFIRMED`

Both pre-registered tests pass (detail + caveats in
[`AUGURY_G3_RESULT.md`](AUGURY_G3_RESULT.md); `g3_result.json` sha256 `bed09869…`):

- DM one-sided **p ≈ 0** (market strike-Brier 0.0127 lower; stat −4.73), n = **3,488** valid
  station-days.
- Encompassing **β_mkt = 0.725, 95% CI [0.670, 0.776] > 0** (survives); β_NBM = 0.340 > 0 (NBM
  also non-redundant) → market is a non-redundant pantheon member.
- Exploratory crossover present: market edge grows −0.010 (−12h) → −0.043 (−2h); broad across
  all 7 stations; holds excluding the densest era.

Binding caveats (AUGURY_G3_RESULT §"caveats"): the comparator is the NBM mean+spread **Gaussian**
(native percentiles unavailable, Amdt A.1) — the **encompassing survival is the robust core**;
the short-lead win is against the operational MaxT **as issued**. `MARGIN_CONFIRMED` unlocks G4
(allelopathy) + G5 (page), both owner-gated.
