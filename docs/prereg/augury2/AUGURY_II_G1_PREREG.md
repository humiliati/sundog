# AUGURY II (HARUSPEX) G1 — Pre-Registration

Parent / boundary documents:

- [`SUNDOG_V_AUGURY_II.md`](../../SUNDOG_V_AUGURY_II.md) (spine; H1/H2/H3, gate ladder)
- [`AUGURY_II_LIT_PASS_MEMO.md`](../../AUGURY_II_LIT_PASS_MEMO.md) (G0, PASSED 2026-07-09 — gap +
  data receipts; availability claims below carry that memo's receipts)
- [`../augury/AUGURY_G1_PREREG.md`](../augury/AUGURY_G1_PREREG.md) (Augury G1 — the universe,
  cutoffs, band, exclusions, scoring, and the encompassing rig are **inherited unchanged**; this
  pre-reg adds three forecaster-provenance tests, not new universe constants)

Filed: **2026-07-09 (PT)**.

Status: **PREREG FILED; EXECUTION NOT ADMITTED.** Gated on a G2 tooling freeze-marker (ASOS
puller + nowcast rung; final-cycle comparator; access panel) recording hashes + the exact
command, as Augury G2/G3 were. Constants below freeze with this file; any change after G2 reads
data = a named amendment. `DOCS_NO_PUBLISH` until the lane's G4 surface.

## §0 — Claims under test (the provenance bracket)

The market is a non-redundant forecaster (Augury). **Where does its information come from?**

- **H1 (measure vs aggregate):** does the market's short-lead edge survive against a real-time
  **observation nowcast**? Survives ⇒ aggregate-access; screened ⇒ measure-access.
- **H2 (independence vs anticipation):** does the dawn market beat the model stack's **final
  same-day word**, or merely anticipate it? Survives ⇒ independent; vanishes ⇒ front-runner.
- **H3 (access as sufficient statistic):** across the pantheon × horizon × era, is
  determining-set membership predicted by a forecaster's **causal-access signature** better than
  by its **identity**? Pinned but **exploratory-strength** (thin rung count — §4).

## §1 — Universe (inherited from Augury G1/G4, frozen)

Stations = the 7 (KNYC, KMDW, KDEN, KLAX, KAUS, KMIA, KPHL). Window = **2023-02-01 → 2026-06-30**
(the Augury full-ladder window; ASOS verified back to the start). Diurnal band = 11 hourly
cutoffs `t_peak−12h … t_peak−2h` from the frozen `t_peak` table. Matched-**availability** cutoffs,
liquidity/validity exclusions, strike-set Brier, PAV implied-CDF — **exactly as Augury**. Horizon
sets: **SHORT = {−5,−4,−3,−2}**, **LONG = {−12,−11,−10,−9,−8}** (the middle offsets −7,−6 enter
pooled fits only). Eras = **2023, 2024, 2025, 2026H1** (4). Inference: ridge-logistic IRLS +
station FE + circular day-block bootstrap (L = 7 d, B = 10,000; encompassing CI refits = 2,000),
probabilities clipped [0.01, 0.99], ridge λ by the Augury day-blocked-CV rule — **all inherited**.
Sample floor = **500 valid station-days** per adjudicated test.

## §2 — H1: the observation-nowcast rung (measure vs aggregate)

**The rung `F_obs(θ)` (frozen construction).** At cutoff T: `x_T` = the latest **valid (non-`M`)**
IEM ASOS temperature ob at-or-before T for the station. The settling daily high is modeled as a
**persistence + diurnal-climatology nowcast**:

`F_obs(θ) = P(Normal(x_T + Δ(o, m, s), σ(o, m, s)) > θ)`

where `o` = band offset (−12…−2), `m` = calendar month, `s` = station, and `Δ, σ` = the **mean
and SD of (CLI_high − x_T)** computed from the **disjoint pre-market climatology 2015-01-01 →
2019-12-31** (same source period as `t_peak`; no leakage), stratified by `(o, m, s)`, requiring
≥ 20 samples per cell (else back off m → season). A named approximation (Gaussian residual),
fenced; it is the canonical persistence+climatology baseline, not ad hoc.

**H1 design.** Two encompassing regressions, both **by horizon**, on the Augury scored rows
augmented with `F_obs`:

- **H1-minimal:** `z ~ logit(F_obs) + logit(F_mkt) + station FE`.
- **H1-full:** `z ~ logit(F_obs) + logit(F_gefs) + logit(F_nbm) + ecmwf_margin + logit(F_mkt) +
  station FE` (ridge on the forecaster block).

**Primary H1 test (SHORT horizon):** β_mkt 95% day-block CI in **H1-full at SHORT**.

| verdict | condition |
| --- | --- |
| `HARUSPEX_H1_AGGREGATE` | β_mkt CI entirely > 0 (market survives with F_obs present) |
| `HARUSPEX_H1_MEASURE` | β_mkt CI includes or lies ≤ 0 (F_obs screens the market) |
| `HARUSPEX_H1_GAP` | H1-minimal and H1-full disagree, or < 500 valid station-days |

Report β_obs (sanity: the nowcast should itself be non-redundant at SHORT) and the LONG-horizon
fits (descriptive).

## §3 — H2: the final-cycle comparator (independence vs anticipation)

**The comparator `F_final(θ)` (frozen).** For each (station, event-day) the **freshest same-day
NBM MaxT issuance** = the latest NBM cycle carrying a valid `D 12Z → D+1 00Z` TMAX; per Augury
Amdt A.1 this is the ≈ **12Z cycle** (available ≈ 13Z). `F_final = Normal(μ, σ)` survival from
that cycle's core TMAX mean + ens-std-dev at the station point (reused Augury decode; the 12Z
scalars are already cached). 

**Future-relative restriction (frozen).** H2 uses only (station-day, cutoff) pairs where the
cutoff **precedes** the final cycle's availability: `T < avail(F_final)`. Because
`avail(F_final) ≈ 13Z`, this retains the **LONG-lead / morning** cutoffs and drops afternoon
cutoffs (where the final cycle is not "future"). **H2 is therefore a LONG-horizon test by
construction** (the G0-flagged structure). Report the retained (station-day, cutoff) fraction.

**H2 design (LONG horizon, retained pairs):** `z ~ logit(F_final) + logit(F_mkt) + station FE`
(H2-minimal); and `z ~ logit(F_final) + logit(F_nbm@T) + logit(F_mkt) + station FE` (H2-isolated,
adds the T-matched NBM to separate "beats the future model" from "beats the current model").

**Primary H2 test:** β_mkt 95% day-block CI in **H2-minimal**.

| verdict | condition |
| --- | --- |
| `HARUSPEX_H2_INDEPENDENT` | β_mkt CI entirely > 0 (dawn market beats the model's final word) |
| `HARUSPEX_H2_FRONTRUNNER` | β_mkt CI includes or lies ≤ 0 (market edge absorbed by F_final) |
| `HARUSPEX_H2_GAP` | H2-minimal and H2-isolated disagree, or < 500 valid station-days |

## §4 — H3: access as a sufficient statistic (pinned, exploratory-strength)

**Rungs (5):** GEFS, NBM, ECMWF-HRES, obs-nowcast (F_obs), market. **Access features (frozen,
per rung):**

| rung | update_cadence_h | ingests_rt_obs | aggregates_agents | is_ensemble | is_market |
| --- | --- | --- | --- | --- | --- |
| GEFS | 6 | 0 | 0 | 1 | 0 |
| NBM | 1 | 0 | 0 | 1 | 0 |
| ECMWF-HRES | 12 | 0 | 0 | 0 | 0 |
| obs-nowcast | 1 | 1 | 0 | 0 | 0 |
| market | 0.02 | 1 | 1 | 0 | 1 |

**Membership panel.** Run the 5-rung Augury encompassing per **(city × horizon × era)** cell =
**7 × 2 × 4 = 56 cells**; record each rung's shrunk coefficient and its **in-set indicator**
(CI > 0). Panel = **56 × 5 = 280 rows**. Cells below the valid-station-day floor are dropped and
counted.

**Meta-regression.** `in_set ~ access_features × horizon`, logistic, **clustered by rung** (the
280 rows are not independent). The load-bearing content is the **cell-varying** signal
(access × horizon interactions — e.g., does `ingests_rt_obs` predict membership specifically at
SHORT), since rung-constant features are otherwise absorbed by identity. **Sufficiency test:** LR
test of adding the **rung-identity × horizon** block; access is sufficient iff (i) access ×
horizon features are jointly significant AND (ii) the identity × horizon block does **not**
significantly improve fit (p > 0.05).

| verdict | condition |
| --- | --- |
| `HARUSPEX_H3_ACCESS_SUFFICIENT` | access×horizon jointly significant AND identity×horizon LR p > 0.05 |
| `HARUSPEX_H3_IDENTITY` | identity×horizon block significantly improves fit (p ≤ 0.05) |
| `HARUSPEX_H3_GAP` | access not jointly significant / underpowered panel |

**DoF fence (binding).** 5 rungs is a thin identity space; access and identity features are
near-collinear at the rung level. H3 is pinned as a **directional meta-analysis**, NOT a proof of
law: an `ACCESS_SUFFICIENT` verdict is reported as *"directionally consistent on a 5-rung,
56-cell panel,"* never as a demonstrated sufficient statistic. Panel-fattening (more markets,
ECMWF-ENS, added cities) is the pre-named path to strength and is out of this pre-reg's scope.

## §5 — Scope / fences

- **Information claim only** (raw midpoints), inherited from Augury; no tradable-edge, no
  investment advice. `F_obs` (Gaussian remaining-rise) and `F_final` (freshest-issuance) are
  named approximations; the robust backbone is the encompassing survival, not any rung's
  calibrated CRPS.
- **Provenance reading (combines H1+H2, not a new token):** `H1_AGGREGATE ∧ H2_INDEPENDENT` ⇒
  genuinely independent aggregated information; `H1_MEASURE ∧ H2_FRONTRUNNER` ⇒ fast obs-reader
  anticipating the models; mixed cells are their own honest channel assignment. Reported as a
  2×2, not adjudicated to a single winner.
- H3 exploratory-strength per §4; do not promote on this panel.

## §6 — G2 pilot + execution gating

G2 builds and freeze-marks before any adjudication: (1) IEM ASOS puller (gentle pace — the memo's
rate-limit caveat; filter `M`; latest-valid-ob ≤ cutoff) + the `F_obs` rung, validated on KNYC
against a hand-checked day; (2) the `F_final` selector over cached NBM; (3) the access-panel
builder. Required G2 artifacts: the frozen 2015–2019 `(o,m,s)` nowcast-climatology table (built
before any 2023+ scoring); the retained-pair fraction for H2; a 5-rung pilot cell (KNYC × SHORT ×
2026H1) reproducing the Augury coefficients with F_obs added. Reserved: runner
`docs/prereg/augury2/haruspex.py`; results `results/augury2/` (gitignored); reuse `.venv-augury`.

## §7 — Self-consistency ledger

- Horizons: LONG = {−12,−11,−10,−9,−8} = **5** offsets; SHORT = {−5,−4,−3,−2} = **4**; band =
  −12…−2 = **11** total; middle {−7,−6} = 2 (pooled-only). 5+4+2 = 11 ✓.
- H3 panel: cities 7 × horizons 2 × eras 4 = **56** cells; × 5 rungs = **280** rows ✓.
- Eras in window: 2023 (Feb–Dec), 2024, 2025, 2026H1 = **4** ✓.
- H1 primary = SHORT (obs bites near the peak); H2 primary = LONG (final cycle is future only
  for morning cutoffs, `avail(F_final) ≈ 13Z`) — the two provenance tests are pinned to
  **opposite horizons by construction**, consistent with the Augury crossover.
- Constants appear once each: nowcast climatology years 2015–2019; ≥ 20 samples/cell; block
  L = 7; B = 10,000; enc refits 2,000; clip [0.01, 0.99]; sample floor 500; H3 LR α = 0.05.
- Verdict tokens: H1 {AGGREGATE, MEASURE, GAP}; H2 {INDEPENDENT, FRONTRUNNER, GAP}; H3
  {ACCESS_SUFFICIENT, IDENTITY, GAP} — all pre-named in the spine.

---

## Amendment G2-A — Tooling Freeze Marker + pilot (2026-07-09 PT)

Append-only. Discharges §6 for the HARUSPEX primitives. **No §1–§4 constant changed.** The
Augury runners are imported **read-only** and unchanged (`augury_pilot` `e2c7b780…`, `augury_g3`
`86e86ff0…`, `augury_g4` `58e09b05…`).

### Tooling (frozen)

| file / artifact | sha256 |
| --- | --- |
| `docs/prereg/augury2/haruspex.py` | `2ea29c2d4345ddaf53fbc9a7785f2e2572139c5e0f8d8ca5e2b30fd2c4379407` |
| `results/augury2/nowcast_climo.json` (frozen 2015–2019 table) | content `f6da8709…` (runner-computed) / file `da9e4ffb…` |

Env `.venv-augury` (numpy, eccodes, pyproj). Results `results/augury2/` (gitignored). ASOS
cache per-(station, year-range) under `results/augury2/asos_cache/`. Deterministic choices frozen
in the module docstring (per-station-year ASOS fetch with `M`-filter + throttle backoff;
`obs_at` = latest valid ob ≤ cutoff, ≤ 6 h stale; `F_obs` Normal(x_T+Δ, σ) with season back-off;
`F_final` = `latest_issue(day, day 20Z)`; generic `encompass` = ridge-IRLS + day-block bootstrap,
λ by 5-fold day-blocked CV).

### Verification receipts (pre-admission)

- **selftest 9/9** (nowcast survival math, access-table shape, design matrix logit+margin+FE,
  encompass recovers a planted signal).
- **ASOS smoke 7/7** stations return plausible obs (2025-07-04 18Z: NYC 81, LAX 70 coastal,
  MIA 83 °F).
- **Climatology frozen + physical:** 132 (offset,month) cells/station (11 × 12, all ≥ 20
  samples); KNYC-July remaining-rise peaks at −8h (+12.9 °F, coldest hour) and tightens toward
  the peak (−2h +4.1 °F, σ 4.1→2.0) — the diurnal signature.
- **Pilot (KNYC × SHORT × 2026H1, 1,376 rows, 100% valid obs):**
  reproduction — the 4-rung no-obs **market coef 0.744, CI [0.601, 0.928]** matches Augury's
  ~0.7 at short lead; H1 pilot signal — with `F_obs` added the **market survives (0.669, CI
  [0.514, 0.873])** and the obs rung is itself non-redundant (0.432, CI [0.262, 0.617]) → leans
  `HARUSPEX_H1_AGGREGATE` (pilot only; not the adjudicated verdict).
- **H2 retained-pair fraction (200-day sample):** long-lead offsets −12…−9 retain **≈1.0**,
  short-lead −4…−2 retain **0.0** (transition at −7/−6) — **confirms H2 is a LONG-horizon test
  by construction** (§3), exactly as pinned.

### G2 → G3 boundary

The **primitives** (ASOS+`F_obs`, `F_final`, `encompass`, the access table, the frozen climo)
are the G2 deliverable and are frozen above. The **G3 adjudication stages** (`h1`/`h2`/`h3`
full-run over all stations × horizons + the access-panel meta-regression) compose these frozen
primitives and will be added under a **G3 freeze marker** before the binding run — no primitive
changes, only composition + the exact staged command. Reserved results:
`results/augury2/{h1,h2,h3}-run/`.

---

## Amendment G3-A — Adjudication stages + run (2026-07-09 PT)

Append-only. The `h1`/`h2`/`h3` stages were added (composition of the frozen G2 primitives + a
chi-square tail for the H3 LR test; one bug fixed pre-verdict — a mid-string `f_` in
`ecmwf_margin` mislabeled that column, caught by the H3 rung loop, corrected, all three re-run
clean). **No §1–§4 constant, and no G2 primitive, changed.** Final runner sha256
`080ea4f2629578e1985dad3438110db92bff7359280b890d1e8c7fa9709f7f9f`.

### Verdicts (detail in [`AUGURY_II_G3_RESULT.md`](AUGURY_II_G3_RESULT.md))

- **H1 = `HARUSPEX_H1_AGGREGATE`** — SHORT, 5-rung: market 0.724 CI [0.659, 0.787] survives the
  obs nowcast (obs itself 0.181). `h1_result.json` `2543d073…`.
- **H2 = `HARUSPEX_H2_INDEPENDENT`** — LONG, 43,070 retained rows: market 0.662 CI [0.593, 0.727]
  beats the day's final NBM cycle (F_final 0.408). `h2_result.json` `988a4477…`.
- **H3 = `HARUSPEX_H3_ACCESS_SUFFICIENT`** *(exploratory-strength)* — 28/56 cells × 5 = 140 panel
  rows; access LR 126.6 (p≈0), identity LR 0.0 (p=1.0); pattern obs→SHORT, ECMWF→LONG,
  market→always, GEFS→never. **DoF-fenced: directional, not a law** (5-rung collinearity inflates
  the identity leg). `h3_result.json` `354d6ca1…`.

**Provenance bracket = AGGREGATE ∧ INDEPENDENT** (the strong corner): genuinely independent,
aggregated information.

### Exact command (unchanged, admitted)

```
./.venv-augury/Scripts/python docs/prereg/augury2/haruspex.py all --admitted
```

(= `h1`, `h2`, `h3` in sequence; caches: ASOS per-station-year, NBM availability. G4 = umbrella
fold + optional page section.)
