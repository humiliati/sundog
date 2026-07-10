# SUNDOG_V_AUGURY_II

*Lane spine. The Augury rig proved the market is a non-redundant forecaster. This lane asks the
next question and only that one: **where does the market's information come from** — and is a
forecaster's causal-access signature a sufficient statistic for its non-redundancy? Frame +
gated plan; nothing pulled, nothing scored. The lit-pass memo (G0) gates everything.*

*Name: HARUSPEX — the augur read the sky for the fact; the haruspex cuts deeper to ask where
the knowing comes from. Rename trivially to `SUNDOG_V_AUGURY_PROVENANCE` if the codename reads
too cute — it is a filename and a header, not a claim.*

Status: **G3 ADJUDICATED 2026-07-09** ([`prereg/augury2/AUGURY_II_G3_RESULT.md`](prereg/augury2/AUGURY_II_G3_RESULT.md)).
**H1 = `HARUSPEX_H1_AGGREGATE`** (market 0.724 survives the obs nowcast); **H2 = `HARUSPEX_H2_INDEPENDENT`**
(market 0.662 beats the final NBM cycle); **H3 = `HARUSPEX_H3_ACCESS_SUFFICIENT`** *(exploratory-strength,
DoF-fenced)*. **Provenance bracket = AGGREGATE ∧ INDEPENDENT** — the market holds genuinely independent,
aggregated information (not a thermometer-reader, not a model-anticipator). G0–G3 in
[`AUGURY_II_G1_PREREG.md`](prereg/augury2/AUGURY_II_G1_PREREG.md) (Amdts G2-A, G3-A). **G4 (umbrella fold +
optional augury.html section) is the remaining gate.**

---

## The through-line

Augury established the *fact*: the market survives an encompassing kill-test against the model
stack and is a non-redundant member of the minimal determining set {NBM, ECMWF, market}, with
its edge horizon-localized to the final hours. This lane asks the *provenance*: the market's
extra information could be (a) **measure-access** — it reads real-time observations the stale
model cycle has not ingested; (b) **anticipation** — it front-runs where the models will
themselves converge; or (c) **aggregate-access** — it pools many traders into information no
single instrument holds. These are not rhetorical: they are the causal-access channels of
[`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md), and Augury grounded that umbrella's
**aggregate row** *by assumption*. H1 and H2 localize the market's information to a channel; H3
asks whether the channel/access signature is a **sufficient statistic** for non-redundancy
across the whole pantheon. This is the causal-access thesis tested on a real, de-confounded,
hard-settled functional — the substrate the synthetic arcs never had.

## The apparatus (reused, already built + verified in Augury)

- Kalshi candle puller (no auth), implied-CDF via PAV isotonic on the strike ladder, strike-set
  Brier scoring, liquidity/validity exclusions — `docs/prereg/augury/augury_pilot.py`.
- Matched-**availability**-cutoff selection (latest issue ≤ cutoff by S3 `LastModified` / candle
  ts; never nominal cycle time).
- GRIB point extraction 7/7 vs `find_nearest`: NBM (Lambert + boustrophedon), GEFS + ECMWF
  (regular 0.25°); scalar caches under `results/augury/`.
- The encompassing rig: ridge logistic IRLS + day-block bootstrap CIs + backward selection +
  by-horizon (SHORT/LONG) — `docs/prereg/augury/augury_g4.py`.
- CLI ground truth via IEM; the frozen G1 constants (stations, band, cutoffs, exclusions).

Each hypothesis below reuses this rig and adds one new, small piece.

## H1 — Measure-access vs aggregate-access (the obs-nowcast discriminator)

**Question:** does the market's short-lead edge survive an encompassing regression that includes
a **real-time-observation nowcast** — latest METAR/ASOS temperature carried to the daily peak by
diurnal climatology?

- **Null / channel = MEASURE:** the obs-nowcast **screens** the market at short lead (market coef
  CI includes 0 once the nowcast rung is in). The market's edge *is* fast measurement access — it
  reads the thermometer the model cycle has not yet ingested. Not trivial, but it relocates the
  causal-access row from **aggregate** to **measure**.
- **Live / channel = AGGREGATE:** the market **survives** even with the nowcast present → it holds
  information beyond the raw current observation; the aggregate-access assignment stands.
- **Verdict tokens:** `HARUSPEX_H1_AGGREGATE` (survives) / `HARUSPEX_H1_MEASURE` (screened) /
  `HARUSPEX_H1_GAP` (mixed / underpowered).
- **Not a lane-killer, a channel-assignment.** Either outcome is a positive result for the
  causal-access taxonomy; the point is *which* channel, not pass/fail.
- **Tooling — new:** an IEM ASOS/METAR puller at matched cutoffs; the nowcast as a rung
  `F_obs(θ) = P(high > θ | latest ob + climatological remaining-rise)`, a **named-approximation**
  Normal(ob + Δ_clim(hour,season), σ_resid) survival with the remaining-rise mean/spread fit from
  disjoint pre-window climatology. **Reused:** matched-cutoff selection, scoring, encompassing.
- **Cost:** low. IEM ASOS is a simple API; one new rung; rerun the encompassing.

## H2 — Independence vs anticipation (beat the model's final word)

**Question:** does the market at cutoff T beat even the **day's freshest eventual model cycle**
(the latest NBM issuance the day ever produces, well after T), not just the T-matched cycle?

- **Null / ANTICIPATION:** the market's edge over the T-matched model **vanishes** once the day's
  final model cycle is in the regression → the market was front-running where the models would
  themselves converge; its "extra info" is a lead on the models, later absorbed by them.
- **Live / INDEPENDENCE:** the market at T **survives encompassing against the final model
  cycle** → its information is *never* captured by the model stack through settlement; it is
  genuinely independent (obs- or crowd-driven), not model-anticipation.
- **Verdict tokens:** `HARUSPEX_H2_INDEPENDENT` / `HARUSPEX_H2_FRONTRUNNER` / `HARUSPEX_H2_GAP`.
- **Tooling — new:** a "final-cycle" comparator = the freshest NBM MaxT issuance for each event
  day (a modest extra pull of the late-day cycles). **Reused:** the entire encompassing rig with
  the comparator swapped from matched-cutoff to final-cycle; NBM decode + most scalars cached.
- **Cost:** low. No new machinery — a different comparator into the same regression.

**H1 + H2 together bracket provenance.** Not-just-measurement (H1 = AGGREGATE) *and*
not-just-anticipation (H2 = INDEPENDENT) ⇒ the market holds genuinely independent aggregated
information. Any other combination is itself a clean channel assignment. This two-sided bracket
is the tight, defensible core; H3 is the synthesis.

## H3 — Access as a sufficient statistic (the umbrella payload)

**Question:** across the full pantheon, cities, and horizons, is a forecaster's
**determining-set membership predicted by its causal-access signature better than by its
identity** — i.e., does "access type" screen off "which model"?

- **Access features (per rung, pre-registered):** update-cadence (hours), ingests-real-time-obs
  {0,1}, aggregates-agents {0,1}, is-market {0,1}, native spatial resolution, deterministic-vs-
  ensemble {0,1}. Identity = rung dummies.
- **Design:** build a **membership panel** — run the G4 encompassing per (city × horizon × era)
  cell to get each rung's coefficient/CI; then a **meta-regression** of the membership signal
  (in-set indicator, or shrunk coefficient) on access features, with a likelihood-ratio test for
  whether adding identity dummies improves fit.
- **Null / IDENTITY:** access features do **not** predict membership; identity / idiosyncratic
  skill dominates (adding identity significantly improves the meta-fit) → the causal-access
  sufficiency claim is **not supported** on this functional. A clean negative for the umbrella.
- **Live / ACCESS-SUFFICIENT:** access features predict membership and identity adds nothing
  significant → access is a sufficient statistic for non-redundancy here; the umbrella's aggregate
  row is grounded as a *law*, not an assumption. H1's channel assignment enters as one access
  feature (the market's `ingests-real-time-obs` bit is set by H1's outcome).
- **Verdict tokens:** `HARUSPEX_H3_ACCESS_SUFFICIENT` / `HARUSPEX_H3_IDENTITY` /
  `HARUSPEX_H3_GAP`.
- **Tooling — new:** the access-feature schema + the two-level meta-regression. **Reused:** the
  G4 encompassing as the first stage, run across the cell grid to build the panel.
- **Cost:** moderate — the panel needs the G4 ladder over many cells (mostly cached inputs); the
  meta-model is light; the spec care (avoiding a degrees-of-freedom fishing hazard with few
  rungs) is the hard part and is the reason H3 is staged after H1/H2.

## Precedent + prior-art gap

- **Roll (1984)** showed markets carry weather information beyond the bureau; **provenance** —
  measure vs anticipation vs aggregate — was never decomposed. The market-microstructure
  literature studies price-discovery lead-lag (front-running) in isolation; the weather-nowcast
  literature studies obs-persistence baselines in isolation. **Gap:** nobody has jointly localized
  a weather market's information to a causal-access channel against a matched-cutoff model ladder,
  nor tested access-as-sufficient-statistic across a de-confounded forecaster pantheon.

## Scope / fences

- **Information claim only** (raw midpoints), inherited from Augury; no tradable-edge, no
  investment advice. The obs-nowcast and final-cycle comparators are **named approximations**
  (Gaussian remaining-rise; freshest-issuance definition), fenced as such; the robust backbone
  stays the encompassing survival.
- H3's meta-regression is **exploratory-strength** with the current ~4–5 rungs; its sufficiency
  claim is only as strong as the rung count and city×horizon cell grid allow. Pre-register the
  cell grid and the LR test; report the effective degrees of freedom honestly. Do **not** promote
  H3 to a "law" on a thin panel.
- **What kills the slate's force (not its execution):** if H1 = MEASURE *and* H2 = FRONTRUNNER,
  the market reduces to a fast obs-reader that anticipates the models — still a real result, but
  the aggregate-access story the umbrella wanted is gone; H3 then tests sufficiency over a
  narrower channel set. Reported honestly either way.

## Gates → deliverable ladder

- **G0 — Lit-pass memo (REQUIRED FIRST).** `AUGURY_II_LIT_PASS_MEMO.md`: (a) confirm the
  provenance-decomposition gap; (b) price-discovery / lead-lag + weather-nowcast-baseline
  precedents; (c) **data-availability** for IEM ASOS/METAR at matched cutoffs and the final-cycle
  NBM pull for ≥ N station-days; (d) confirm the reused apparatus covers H1/H2/H3 as specified.
  **Gate:** proceed only if the gap is real AND the obs + final-cycle data are pullable.
- **G1 — Pre-registration.** One doc, three sections: the obs-nowcast construction (H1), the
  final-cycle comparator (H2), the access schema + meta-regression + cell grid (H3); each with its
  verdict tokens and kill criterion; self-consistency gate (numbers reconciled, gates = exact
  criteria).
- **G2 — Tooling + pilot.** Build the ASOS puller + nowcast rung and the final-cycle comparator;
  verify on the Augury KNYC cell; freeze-marker amendment with hashes + exact command.
- **G3 — Run + adjudicate.** H1 and H2 across the Augury station set; then the H3 membership panel
  + meta-regression. Emit the three verdict tokens.
- **G4 — Umbrella fold + surface.** Fold the channel assignments into
  [`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md) (the aggregate row, now assigned not
  assumed). If the result is clean and public-worthy: a "where does the edge come from" section
  appended to the deployed `augury.html`, honest-fenced; else an internal note.

## Cross-links

- Parent apparatus + result: [`SUNDOG_V_AUGURY.md`](SUNDOG_V_AUGURY.md) (deployed).
- Payload / channel taxonomy: [`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md) (aggregate
  row).
- Sufficient-statistic order / determine-resist: the suffstat-order slate and the
  Shadow-Invertibility law (H3 ports σ-as-schema to a real functional; H1/H2 are the channel
  filtration).
- Optional Augury breadth (not in this slate): precipitation, extremes, horizon-extension —
  parked as an à-la-carte menu, conceptually secondary to provenance.
