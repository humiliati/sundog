# A1 pre-registration — Atlas phase boundaries vs Taivaanvahti citizen halo statistics

> **2026-06-12. FROZEN BEFORE THE LEG-0 REGISTERED RUN; binding for Leg 1.** Slate entry: S3-A1
> (`internal/slates/HYP_SLATE_3_EXTERNAL_ANCHORS_2026-06-11.md`, post-fix Fa3/N2.5/Fe2.5/I2.5; all ten
> refuter fixes incorporated below). NOT public-eligible. Standing discipline: pre-registered KILL — **a
> clean null is a SUCCESS**; forward-generate only; deterministic seeded runs + frozen tests; cheap
> headless first leg; name the nearest prior, state the delta. **NOTHING touches live data until Leg 0
> passes for BOTH P1 and P2** (P1 may proceed alone iff K0-P2 alone fires, stated in §6).

## 0. External anchor + sole precedent

Ursa Taivaanvahti/Skywarden (open APIs, open-science invitation — citizenscience.eu/project/524, quoted
verbatim in the slate entry; clause-1 satisfied in INVITATION-FORM, declared). Sole precedent: **McDowell
1979, JOSA 69(8):1119–1122** — CZA frequency vs solar elevation, 1894–1931 hand records, one arc, small N,
post-hoc shape fit. Delta: first database-scale, multi-boundary, PRE-REGISTERED, blind-estimator test of
DERIVED wall positions (zero free parameters in the predictions). Live Anubis anti-bot block on
taivaanvahti.fi confirmed twice (2026-06-11) — the Leg-1 access ladder is §7.

## 1. The predictions under test (all derived, zero free parameters)

- **CZA TIR wall (P1):** arccos(√(n²−1)) = **32.196°** (n=1.31; `atlas_forward_sweep.cza_wall()`),
  chromatic + sun-disk smear band **31.0–32.9°** (recomputed 2026-06-11 from `atlas_bifurcation_set.py`).
- **UTA+LTA merge (P2):** `atlas_caustic_map.merge_elevation()` = **29.71°** — an A₃-class metamorphosis,
  so P2 uses a logistic-MIDPOINT statistic ONLY (no edge estimator).
- **CHA-on complement (P3):** 57.804° > max Finnish solar elevation 53.62° ⇒ the Finnish CHA category is
  physics-EMPTY — a misclassification-floor estimator plus the hardened empty-side falsifier K1d (§6).

## 2. Leg-0 apparatus (all pinned)

**(a) Solar elevation** — new `solar_elevation(lat_deg, lon_deg, utc_datetime)` (NOAA/Meeus low-precision,
~40 lines; GEOMETRIC elevation, no refraction — refraction at h≈30° is ~0.03°, negligible vs the ±3.7°
time smear; stated). **Validation pins (exact identities + almanac ranges, all in the frozen test):**
(i) noon-elevation identity |max-elevation − (90° − lat + decl)| < 0.05° at Helsinki & Rovaniemi on 4
pinned dates; (ii) declination at the June-2025 solstice instant ∈ [23.42°, 23.45°]; |decl| < 0.5° on
2025-03-20; (iii) equation-of-time extremes: minimum ∈ [−15.5, −13.5] min in mid-Feb, maximum ∈ [15.5,
17.5] min in early Nov; (iv) max Finnish solar elevation (Hanko 59.81°N, June solstice) = 53.6° ± 0.3°. **[Amendment A1,
pre-run, 2026-06-12: anchor city corrected Helsinki→Hanko — Helsinki's solstice noon is 53.27°; the
53.62° figure is southernmost Finland. Identity unchanged; apparatus-tier fix logged before any run.]**

**(b) The frozen flux curve** — `cza_flux(e)` from `atlas_orientation_boundaries.py` (seeded raytracer
wrapper, n_orient=9000, seed=44 — UNTOUCHED), evaluated ONCE on the pinned grid **{3..28 step 1} ∪
{28.5..35.5 step 0.5} ∪ {36..40 step 1}** (46 calls, ~21 min) and cached (`scripts/_cache_a1_cza_flux.npz`;
the cache is a pure speedup — deterministic regeneration if absent). Continuous φ(h) = PCHIP interpolation,
clamped ≥0, φ≡0 outside [3, 35.5] support. **Window-validity probe (pre-freeze, recorded):** 12-point
sweep gave 19@5° / 781@10 / 1697@15 / 2398@20 / 2592@22 / 2618@28 / 1865@30 / 1090@31 / 355@32 / 91@33 /
0@34 — plateau 22–28°, collapse through the wall carrying the smear-band slope; 27.8 s/call timing.

**(c) Injection mechanism (PINNED, the one mechanism):** blind wall injection at **{28.0, 32.196, 36.0}°
by h-TRANSLATION of the frozen curve**: φ_w(h) = φ(h − (w − 32.196)). Shape-preserving; preserves the
§0.2 smear-band slope. (n-shift regeneration explicitly NOT used.)

**(d) Synthetic report generator (forward; report-level).** A report = (municipality, date, local time,
class, label). Sampling: municipality from the pinned table below; date uniform 2012-01-01..2025-12-31;
local clock time uniform 06:00–20:00 weighted by the pinned effort model **w_eff = exp(−((hour−13)/4)²) ×
(1.4 if weekend else 1.0)** (local = UTC+2 fixed; the +3 summer offset is absorbed by the time smear and
the ratio construction; stated). TRUE h = solar_elevation at the true instant; reports with h < 2° are
rejected (twilight floor). Class intensities at true h: **ring** ∝ 1 (the no-wall effort/availability
carrier — availability cancels in the ratio by construction); **CZA** ∝ κ·(φ_w(h)/max φ)^γ with γ swept
(never fitted) and κ set so realized E[N_CZA] hits the cell's target; **contaminant** = ring-distributed
reports mislabeled CZA at rate ε of the CZA-labeled total. RECORDED h (what the pipeline sees) = solar
elevation at true instant + U(−30, +30) min — ±3.7° worst-case at Finnish latitudes, the dominant noise.
**Pinned municipality table (lat, lon, weight):** Helsinki 60.17/24.94/.20, Espoo 60.21/24.66/.09,
Tampere 61.50/23.76/.08, Vantaa 60.29/25.04/.08, Oulu 65.01/25.47/.07, Turku 60.45/22.27/.07, Jyväskylä
62.24/25.75/.05, Lahti 60.98/25.66/.04, Kuopio 62.89/27.68/.04, Pori 61.49/21.80/.03, Joensuu
62.60/29.76/.03, Rovaniemi 66.50/25.73/.03, rest-of-Finland 63.00/26.00/.19. (A stress model, not a
demographic claim — the ratio estimator must be robust to it, which is what Leg 0 tests.)

**(e) Nuisance cells (P1).** γ ∈ {0.5, 1, 2} × ε ∈ {1, 5, 10}% at N_CZA=1000, N_ring=3×N_CZA — 9 cells ×
3 injection positions × **200 seeded replicates**. PLUS: **habit-mix climatology drift arm** — λ_CZA
multiplier (1 ± 0.30·(h−h_mid)/(Δh/2)) (linear ±30% plate-fraction trend across h∈[2,45], both signs, at
γ=1/ε=5%) — blind recovery must hold **±1.0°** under it; ring-ratio robustness rows N_ring/N_CZA ∈ {1, 6}
at γ=1/ε=5%; power tiers N_CZA ∈ {300, 3000} at γ=1/ε=5%.

**(f) Frozen estimator (P1; takes NO Atlas constant as input).** Per-1°-bin counts over h∈[5,45]:
ratio model p(h) = a·logistic((h₀−h)/w); binomial MLE in (a, h₀, w), bounds a∈(.01,.99), h₀∈[10,45],
w∈[0.15,6], 3 pinned starts (h₀∈{20,28,36}); detection statistic LR = 2(LL_fit − LL_flat) vs the
constant-ratio null. Secondary robustness column: isotonic-regression max-drop changepoint. **Constant
bias correction rule (blindness-preserving):** b̂ = pooled median(ĥ₀ − w_inj) across the three injected
positions, PERMITTED iff max pairwise |Δ median bias| ≤ 0.4° across positions; else forbidden and
recovery is scored uncorrected. The exposure-weighting (sun-minutes-per-bin) normalization is exercised
as a CONSISTENCY column on the γ=1/ε=5 cell (agreement within 0.5° reported; not a Leg-0 kill).

**(g) Wall-free controls — two DISJOINT seed batches.** Batch A (200 replicates, seed branch 1000+i):
calibrates the frozen detection threshold = 95th percentile of control LR. Batch B (200 FRESH replicates,
seed branch 5000+i): scores specificity against that threshold — **K0b fires on batch B only.**

**(h) McDowell-shape calibration check (ABORT, not a result):** the TRUE-wall (32.196°), γ=1, ε=0,
no-drift synthetic CZA elevation histogram must have its mode in **[18°, 29°]** (McDowell's observed ~22°
peak, qualitatively) and < 2% of unsmeared reports above 33°. Failure ⇒ ABORT/redesign the generator.

**(i) Leg 0-P2 (the metamorphosis midpoint pipeline, gated exactly as P1).** Two-class generator:
column-display reports distributed like ring; label = circumscribed with P(circ|h) = ε_m/2 + (1−ε_m)·
logistic((h−m)/w_t), w_t = 1.5° pinned (robustness rows w_t ∈ {0.8, 2.5}), ε_m = 5% label noise;
**blind midpoint injections m ∈ {26.7, 29.71, 32.7}°**; N_reports = 2000/replicate, 200 replicates per
(m, w_t) cell; estimator = 3-param binomial MLE (m, w, ε_mix), recovery = median |m̂ − m| ≤ **1.5°** per
cell; **no-transition control** (P(circ) = const 0.5): batch-A/-B threshold-and-specificity exactly as
(g), spec ≥95%.

**(j) Seeds.** BASE_SEED = 20260612; replicate rng = SeedSequence([BASE, leg_id, cell_idx, pos_idx, rep]).
All runs CPU, numpy/scipy only.

## 3. Leg-0 kill gates (any firing ⇒ NO live pull; banked as the lane's resolution-floor null = SUCCESS)

- **K0a** — median blind-injection recovery error > 1.0° **at N_CZA=1000, per nuisance cell (worst cell
  over the γ×ε grid, drift and ring-ratio rows included)**, 200 replicates each.
- **K0b** — batch-B wall-free specificity < 95%.
- **K0c** — detection power < 80% at N_CZA=1000 (LR > frozen threshold); the 300/3000 tiers are REPORTED
  (300 expected to fail power — that is the resolution floor statement, not a kill).
- **K0-P2** — midpoint recovery median > 1.5° in any (m, w_t) cell, or the no-transition control
  specificity < 95% ⇒ P2 never goes live (P1 may proceed alone; stated here).
- **ABORT-McD** — §2(h) fails ⇒ generator redesign (apparatus, never a result).

## 4. Leg-1 protocol (GATED: requires Leg-0 pass + owner sign-off for any contact; binding now)

1. **Read-only schema inspection:** taxonomy granularity (does the DB distinguish CZA / 22°-ring /
   UTA / LTA / circumscribed / CHA?); timestamp semantics (event time vs report time — if only report
   time exists, the measured smear replaces ±30 min and Leg 0 is RE-RUN with the measured smear: a NAMED
   re-gate, no silent widening); **guide-inspection step:** if Taivaanvahti's classification guidance
   states a numeric elevation rule for the circumscribed/tangent label, **P2 is VOID (named)** or
   rescoped to blind photo-morphology arbitration (closed-loop vs gapped-arc, blind to timestamp).
2. **Pull ladder (Anubis-aware):** documented csv/json open-interface endpoints (may be challenge-exempt)
   → owner-gated browser-session pull → admin contact under their own open-science invitation. Solar
   reports only, 2011–2025, Finland.
3. **Photo arbitration:** above-wall CZA tail, capped at **200 photos**; "tail too large to arbitrate" =
   a NAMED partial. Tail statistic denominator pinned: **N(photo-confirmed CZA at h>35°) / N(all CZA
   reports accepted into the primary statistic)** ≤ 5%.
4. **Verdicts:** **P1** edge (after double normalization + arbitration + the §2(f) bias rule): PASS within
   ±1.5° of 32.196° (consistent with the 31.0–32.9° derived band); **MARGINAL** in (1.5°, 3.0°] — a NAMED
   bounded-partial outcome, bootstrap CI (2000 report-level draws) reported against BOTH thresholds, **no
   outreach on a marginal**; **K1a** beyond 3.0° = falsification receipt (candidate causes pre-listed:
   timestamp semantics, selection-model misspecification, §0.2 smear underestimate). **K1b** tail > 5%.
   **P2** midpoint within ±3° of 29.71° with correct-signed slope; K1c outside/wrong-signed; VOID per the
   guide-inspection or missing label. **P3/K1d (hardened):** a Finnish CHA fires the empty-side falsifier
   ONLY on the pre-registered discrimination battery — horizon-parallelism vs upward curvature toward
   46°-halo tangency, azimuthal extent, color order, computed sun elevation < 53.6°, co-occurring display
   context — PLUS expert-grade independent arbitration OR ≥2 independent displays; a single ambiguous
   photo routes to the misclassification floor, never the falsifier.

## 5. Outputs + frozen test

`scripts/atlas_report_edge_pipeline.py` (Leg 0 registered run: gates → cells → kill verdicts; exit 0 =
both P1 and P2 Leg-0 PASS). Frozen test `scripts/test_atlas_report_edge_pipeline.py` pins: solar-position
identity/almanac checks, 3 cached flux-curve values, McDowell-shape mode, per-cell recovery medians
(worst cell), batch-B specificity, power at the three tiers, P2 recovery medians + specificity, and the
Leg-0 verdict token. Result doc `docs/atlas/A1_TAIVAANVAHTI_LEG0_RESULT.md` (Leg 1, if unlocked, gets its
own section with the live numbers; any amendment logged here).

## 6. Adjudication order

Apparatus build checks → ABORT-McD → K0a → K0b → K0c → K0-P2 → (Leg-0 verdict). Leg 1 only after owner
sign-off; live verdict space exactly §4.4, no dead zones (pass / marginal / kill fully partition).

## 7. Priors / citations

McDowell 1979 (sole precedent, one arc); ATLAS_PHASE7_PHASE_DIAGRAM.md + atlas_bifurcation_set.py +
atlas_orientation_boundaries.py (the derived walls, banked forward-tier); Bruus et al. EPSC-DPS 2025
(database scale, search-snippet tier); NOAA/Meeus solar position (standard low-precision algorithm).
