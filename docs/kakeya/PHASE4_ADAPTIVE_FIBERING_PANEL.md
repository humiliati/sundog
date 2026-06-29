# Kakeya Phase 4 - Adaptive-Fibering Ambiguity Panel (H-K4)

- Artifact id: `KAK-PHASE4-ADAPTIVE-FIBERING-PANEL`
- Date: 2026-06-29
- Status: internal measurement receipt. **Falsifier `ADAPTIVE_FIBERING_NO_SIGNAL` FIRED** (a successful null).
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md) (H-K4)
- Script: [`../../scripts/kakeya-adaptive-fibering-panel.mjs`](../../scripts/kakeya-adaptive-fibering-panel.mjs)
- Output: [`../../results/kakeya/adaptive-fibering-panel/manifest.json`](../../results/kakeya/adaptive-fibering-panel/manifest.json)

## Verdict

**The falsifier fired: the naive adaptive-fibering gap is a finite-grid density artifact,
not a clean body-resistance metric.** For a body `K` in `F_q^2` we compared a fixed-direction
fiber description (cover `K` with full lines in the single best direction) against an adaptive
one (cover with full lines in *any* direction), with `gap = adaptive_covered -
fixed_best_covered`. The gap is positive and grows with the number of line-directions in `K`
- but **random same-size control bodies reproduce the gap once `K` fills ~half the grid**, so
the statistic mostly measures body density in a small field, exactly the
`ADAPTIVE_FIBERING_NO_SIGNAL` artifact clause.

This is a report-only null. It does not touch Euclidean Kakeya, optimal finite-field sets, or
any public claim; it tells us the body-resistance bridge **remains prose** in this naive form.

## Command Run

```powershell
node scripts/kakeya-adaptive-fibering-panel.mjs
```

(200 random control bodies per structured size; deterministic seed; `q in {5, 7}`.)

```text
KAK_ADAPTIVE_FIBERING q=5 maxStructGap=12 worstControlMean=6.8 anyControlGap=true signal=true structureDriven=false | gaps[k1:0 k2:4 k3:8 k4:12 k5:1 k6:1]
KAK_ADAPTIVE_FIBERING q=7 maxStructGap=30 worstControlMean=17.43 anyControlGap=true signal=true structureDriven=false | gaps[k1:0 k2:6 k3:12 k4:18 k5:24 k6:30 k7:1 k8:1]
KAK_ADAPTIVE_FIBERING_PANEL qs=5,7 falsifier=fired
```

## Evidence: the gap is real but contaminated by density

Structured bodies are `k`-direction line unions (one full line in each of `k` distinct
directions). `control_mean` / `control_max` are over 200 random bodies of the **same size**.

### q = 7 (49 points)

| body | size (% grid) | struct gap | control mean | control max | clean? |
| --- | --- | ---: | ---: | ---: | --- |
| k2 union | 13 (27%) | 6 | 0 | 0 | yes - structure only |
| k3 union | 19 (39%) | 12 | 0 | 0 | yes - structure only |
| k4 union | 25 (51%) | 18 | 0.3 | 6 | leaking |
| k5 union | 31 (63%) | 24 | 4.1 | 16 | contaminated |
| k6 union | 37 (76%) | 30 | 17.4 | 25 | contaminated |

### q = 5 (25 points)

| body | size (% grid) | struct gap | control mean | control max | clean? |
| --- | --- | ---: | ---: | ---: | --- |
| k2 union | 9 (36%) | 4 | 0 | 0 | yes - structure only |
| k3 union | 13 (52%) | 8 | 0.5 | 4 | leaking |
| k4 union | 17 (68%) | 12 | 6.4 | 12 | control == structure |

At `q=5, k4` the random control's *max* gap (`12`) equals the structured gap, and at
`q=7, k6` the control mean (`17.4`) is most of the structured gap (`30`). Once a body fills
roughly half of `F_q^2`, random point sets already contain full lines in several directions,
so the adaptive-over-fixed advantage is not about adaptive fibering - it is about density.

## The recoverable (sparse-regime) signal

In the **sparse regime** (`|K| <~ q^2 / 2`) the control gap is exactly `0` while the
structured gap is positive and grows linearly with the number of directions (`q=7`: 6, 12 at
k2, k3). There the gap *is* structure-driven. So a valid metric would be the **excess over a
size-matched random control** (`gap - control_mean`), reported only in the sparse regime - not
the raw gap. That refinement is the next move if H-K4 is pursued; as specified, the raw-gap
metric fires the falsifier.

## Interpretation Boundary

Supports only:

> The raw fixed-vs-adaptive fiber gap in the finite-field workbench is dominated by body
> density: random same-size bodies reproduce it for bodies filling >~ half the grid. A
> structure-driven adaptive-fibering signal survives only in the sparse regime, and only as
> excess over a size-matched control.

It does not support any Euclidean-Kakeya, maximal-function, or incidence-geometry statement,
and does not license public copy. H-K4's body-resistance bridge is not yet a clean workbench
metric; the sparse-regime / excess-over-control refinement is the path, not the raw gap.
