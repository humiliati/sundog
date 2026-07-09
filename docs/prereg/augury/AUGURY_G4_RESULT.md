# AUGURY G4 — Determining-Shadow-Set Result

> 2026-07-08. Adjudication of the G4 run against
> [`AUGURY_G4_PREREG.md`](AUGURY_G4_PREREG.md) §3–§4. Internal (`DOCS_NO_PUBLISH` until G5).

## Verdict: `AUGURY_G4_MARKET_MEMBER`

Final ladder: **{GEFS, NBM, ECMWF-HRES, market}** (4-rung). MOS dropped (NBS archive
unavailable, Amdt G4-α). n = **3,258 station-days**, 82,702 strike rows, window 2023-02-01 →
2026-06-30, ridge λ\* = 10 (day-blocked CV). `g4_result.json` sha256 `07857e44…` (4-rung) supersedes the
core-3 `d049eccf…`. (Core-3 {GEFS,NBM,market} adjudicated first as a robustness pass — same
verdict, {NBM,market} set, GEFS screened.)

## The determining set (pooled)

| rung | pooled coef | 95% CI (day-block) | in minimal set? |
| --- | --- | --- | --- |
| market | **0.702** | [0.647, 0.751] | **yes** |
| ECMWF-HRES | **0.614** | [0.123, 1.176] | **yes** |
| NBM | **0.373** | [0.319, 0.441] | **yes** |
| GEFS | −0.073 | [−0.111, −0.036] | no (screened off) |

- **Minimal determining set = {NBM, ECMWF-HRES, market}.** The market survives the full-ladder
  encompassing against the operational blend (NBM), a second independent model (ECMWF-HRES),
  *and* the raw physics ensemble (GEFS) jointly.
- **GEFS is the one screened off** (coefficient negative). NBM is a post-processed blend that
  already ingests GEFS-type physics, so the raw GEFS ensemble adds nothing beyond it — the
  determining-set machinery correctly drops the redundant rung. This is the determine/resist
  invariant on the pantheon: the aggregate (market) is non-redundant; the redundant physics
  ensemble collapses out.
- **ECMWF-HRES is non-redundant** (CI > 0, though wide — it enters as a deterministic margin
  covariate, structurally less informative than a probability). My pre-run expectation that
  ECMWF would be screened like GEFS was **wrong**: ECMWF's IFS carries information the
  US-centric NBM blend does not fully absorb.

## Market membership by horizon

| horizon | market coef | 95% CI | in set? | ECMWF coef | ECMWF CI | ECMWF in set? |
| --- | --- | --- | --- | --- | --- | --- |
| SHORT {−5..−2} | 0.772 | [0.709, 0.834] | **yes** | 0.449 | [−0.039, 0.971] | no (CI ∋ 0) |
| LONG {−12..−8} | 0.679 | [0.614, 0.739] | **yes** | 0.606 | [0.099, 1.178] | yes |

- **Market is in the set at both horizons**, stronger at short lead — the same direction as the
  G3 crossover (its edge concentrates near t_peak, where real-time observations bite).
- **ECMWF-HRES flips the other way**: in the set at LONG lead, *out* at SHORT lead. Complementary
  mechanism — at day-3 lead the model differences matter and ECMWF adds; in the final hours the
  market's real-time obs dominate and the day-ahead deterministic model is no longer
  distinguishing. The two non-model members of the set (market, ECMWF) carry information at
  *opposite* ends of the horizon. GEFS stays out at both.

## Reading

G3 showed the market beats NBM 2-way. G4 hardens it: with a physics ensemble (GEFS), the
operational blend (NBM), *and* a second independent model (ECMWF-HRES) all in the regression,
the market remains a non-redundant member of the minimal determining set — and it is **GEFS**,
not the market, that proves redundant. The market's information is not a lossy echo of the model
stack; it carries a component the blended model stack does not. The horizon split is the
sharpest part: **market and ECMWF occupy opposite ends** — the market is decisive in the final
hours (real-time obs), ECMWF at long lead (model skill), NBM throughout, GEFS nowhere. That is
the determine/resist invariant resolved by horizon on real money. The allelopathy
determining-shadow-set read realized on a real, de-confounded, hard-settled functional (weather)
— the substrate the synthetic chatv2 arc never had. Grounds the causal-access **aggregate row**
on a non-toy.

## Caveats / fences

- **MOS rung absent** (data unavailability, not choice — Amdt G4-α): no reliable historical
  same-day point-MOS source (NBS ~empty historically, MAV 422, MEX days-3+). The "human"
  pantheon rung is missing; the ladder is physics-ensemble + statistical-blend + aggregate.
- **ECMWF rung = HRES deterministic** (ENS was throttle-walled, Amdt G4-α), entering as a margin
  covariate — hence the wide CI. It landed non-redundant, contrary to the pre-run expectation.
  ECMWF-ENS (a distribution, not a point) remains a future sub-amendment from an un-throttled
  context; it would only strengthen the ECMWF rung, not change the market's membership.
- GEFS civil-day-max is the fenced max-of-6h-blocks Gaussian (Amdt spec §1). The determining-set
  claim rests on the encompassing coefficients (robust), not on any single rung's calibrated CRPS.

## What this opens

Both G3 and G4 now support the market as a non-redundant pantheon member. **G5** (`augury.html`,
owner-gated, `DOCS_NO_PUBLISH`) can present: the G3 crossover as the visual, the G4
determining-set ({NBM, market}; GEFS screened) as the substance, information-claim vs
tradable-edge fenced. Artifacts: `results/augury/g4-run/` (`g4_result.json`, `scores_g4.jsonl`,
`gefs_scalars.jsonl`, `mos_scalars.jsonl`).
