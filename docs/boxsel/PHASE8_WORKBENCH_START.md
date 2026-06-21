# BoxSEL Phase 8 - Workbench Start

**Date:** 2026-06-21  
**Status:** `STARTED_REVIEW_ONLY`

Phase 8 turns the passed Phase-7b receipt into an inspectable workbench. This is an internal
review surface, not a public promotion and not a real-KG/product claim.

## Boundary

The workbench is allowed to show:

```text
sampled interval
box-attainable interval where Phase 4 proved it
query-pressure interval where Phase 7b generated it
exact oracle interval
trace flags and detector action
```

It is not allowed to claim:

```text
real-KG transfer
calibration guarantee
Ask Sundog product behavior
pressure response as exact inference
box-attainable endpoints for families where only pressure traces are present
```

## Artifacts

```text
boxsel.html
public/data/boxsel-phase8-workbench.json
scripts/boxsel_phase8_workbench_data.py
scripts/test_boxsel_phase8_workbench.py
```

The page is registered in `site-pages.json` as review-only and marked:

```text
noindex, nofollow
```

The SEO roadmap row is Class D. No public inbound path has been added.

## Workbench Surface

The first slice includes:

```text
case rail over all 28 Phase-7b cases
run summary strip
interval bars for exact, box/pressure, and sampled layers
query-pressure slider
dimension slider for the Helly box closed form
restart-reveal slider for ordinary endpoint movement
loss-tolerance slider for loss controls
trace quantity panel
receipt links
```

Layer naming is deliberately conservative:

- Helly rows show `Box I_box^1` or `Box I_box^n` because Phase 4 proved the closed form.
- Stable PMP and support-floor rows show `Pressure trace`, not `I_box`, because Phase 7b only
  generated query-pressure evidence for those families.

## Source

The workbench data is generated from the locked Phase-7b result:

```text
results/boxsel/phase7b_false_closure_run/manifest.json
docs/boxsel/PHASE7B_FALSE_CLOSURE_RUN.md
```

Summary carried into the page:

```text
PASS_PREREG_GATE
detector accepted false closures : 0 / 16
baseline accepted false closures : 16 / 16
stable-PMP pressure warnings     : 8 / 8
```

## Verification

```text
python scripts/boxsel_phase8_workbench_data.py
python scripts/test_boxsel_phase8_workbench.py
```

Result at start:

```text
17/17 checks pass, exit 0.
npm run build passes; boxsel.html is excluded from sitemap by noindex.
dist link check passes; public-copy integrity reports 0 FAIL, 1 WARN
(pre-existing docs/chatv2/LANE_CHARTER.md route warning).
```

---

*Sundog Research Lab - BoxSEL Phase-8 workbench start. Internal review surface; toy micro-SEL boundary applies.*
