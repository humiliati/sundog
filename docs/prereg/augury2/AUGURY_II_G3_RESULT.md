# AUGURY II (HARUSPEX) G3 — Provenance Result

> 2026-07-09. Adjudication of the three provenance hypotheses against
> [`AUGURY_II_G1_PREREG.md`](AUGURY_II_G1_PREREG.md) §2–§4. Internal (`DOCS_NO_PUBLISH` until the
> lane's G4 surface). Verdict tokens are the pre-registered ones.

## Verdicts

| hyp | token | one line |
| --- | --- | --- |
| H1 | **`HARUSPEX_H1_AGGREGATE`** | the market's short-lead edge survives a real-time obs nowcast |
| H2 | **`HARUSPEX_H2_INDEPENDENT`** | the dawn market beats the model stack's final same-day word |
| H3 | **`HARUSPEX_H3_ACCESS_SUFFICIENT`** *(exploratory-strength — fenced)* | access predicts membership; identity adds nothing on this panel |

## The provenance bracket (H1 × H2)

The two localizers land in the **strong corner**:

> **AGGREGATE ∧ INDEPENDENT** ⇒ the market holds **genuinely independent, aggregated information**
> — not a fast thermometer-reader (it beats the obs nowcast), not a model-anticipator (it beats
> the models' final word). Its causal-access row is **aggregate**, confirmed, not assumed.

## H1 — measure vs aggregate (`AGGREGATE`)

82,702 rows, 100% with valid ASOS obs. Primary test = β_mkt CI in the full 5-rung regression
(with `F_obs`) at SHORT lead:

- **market 0.724, CI [0.659, 0.787]** — survives, clearly > 0 (H1-minimal agrees).
- obs-nowcast 0.181, CI [0.115, 0.259] — itself non-redundant but modest: at matched cutoffs the
  operational models already carry most of what the raw ob provides. The market's edge is beyond
  both. The market is **not** a repackaged current observation.

## H2 — independence vs anticipation (`INDEPENDENT`)

43,070 retained LONG-lead rows (cutoffs preceding the day's final NBM cycle; 44 dropped for an
uncached final scalar). Primary test = β_mkt CI in H2-minimal against `F_final` (the freshest
same-day NBM, ≈12Z, *later* than these dawn cutoffs):

- **market 0.662, CI [0.593, 0.727]** — survives even the model's final word.
- F_final 0.408, CI [0.364, 0.462] — non-redundant. The dawn market's information is **never
  absorbed by the model stack through settlement**; it is independent, not anticipation.

## H3 — access as a sufficient statistic (`ACCESS_SUFFICIENT`, exploratory-strength)

Membership panel = **28 of 56 cells** cleared the 40-station-day floor (2023-partial / thin
cells dropped) × 5 rungs = **140 panel rows**. Meta-regression `in_set ~ access × horizon`:

- **access features jointly predict membership:** LR stat **126.6**, df 10, **p ≈ 0**.
- **identity adds nothing beyond access:** LR stat **0.0**, df 8, **p = 1.0**.

Both pre-registered conditions met ⇒ the token is `ACCESS_SUFFICIENT`. The panel pattern is
mechanistically clean — membership tracks access, by horizon:

| rung | in-set (all) | SHORT | LONG | access story |
| --- | --- | --- | --- | --- |
| market | **1.00** | 1.00 | 1.00 | aggregate + rt-obs → always in |
| NBM | 0.93 | 0.93 | 0.93 | the operational blend → nearly always |
| ECMWF | 0.32 | 0.21 | **0.43** | model skill → **more at long lead** |
| obs-nowcast | 0.25 | **0.50** | 0.00 | rt-obs → **only at short lead** |
| GEFS | 0.00 | 0.00 | 0.00 | raw physics ensemble → screened everywhere |

**DoF fence (binding, per §4).** With only 5 rungs the access and identity spaces are
near-collinear, so the `identity adds nothing` leg (p = 1.0, stat exactly 0) is **partly
mechanical** — access can reconstruct identity by construction. The load-bearing, non-artifactual
content is (i) access features *strongly* predict membership (LR 126.6) and (ii) the **pattern**:
`obs` earns membership at SHORT, `ECMWF` at LONG, `market` always, `GEFS` never — an access-by-
horizon structure, not idiosyncratic per-model skill. Reported as **directionally consistent on a
thin 28-cell panel, NOT a demonstrated law.** Panel-fattening (more markets, ECMWF-ENS, more
cities) is the pre-named path to real strength.

## Reading

Augury said the market is non-redundant. HARUSPEX says **where that comes from**: the market's
information is genuinely independent and aggregated (survives both the obs nowcast and the models'
final word), and — directionally — a forecaster's determining-set membership is governed by its
**causal-access signature**, with the two non-blend members (obs, ECMWF) earning their place at
opposite ends of the horizon and the market earning it everywhere. This grounds the causal-access
umbrella's **aggregate row** as an assignment reached by test, not assumption, on a real,
de-confounded, hard-settled functional.

## Caveats / fences

- Information claim only (raw midpoints); no tradable-edge, no investment advice (inherited).
- `F_obs` (Gaussian remaining-rise, 2015–2019 climatology) and `F_final` (freshest-issuance) are
  named approximations; the encompassing survival is the robust backbone.
- H3 is **exploratory-strength** (§4 DoF fence): access strongly predicts membership and the
  pattern is clean, but the sufficiency-over-identity leg is inflated by the 5-rung collinearity.
  Do not promote to a law on this panel.
- H2 is a LONG-horizon test by construction (final cycle is future only for morning cutoffs).

## What this opens

The provenance bracket is publishable as the "where does the edge come from" follow-on. **G4**
folds the channel assignment into [`SUNDOG_V_CAUSAL_ACCESS.md`](../../SUNDOG_V_CAUSAL_ACCESS.md)
(aggregate row = assigned, not assumed) and — optionally — appends a section to the deployed
`augury.html`. Artifacts: `results/augury2/{h1,h2,h3}-run/*.json`, the frozen
`nowcast_climo.json`.
