# H8-RF Result — double-descent removal in the RF closed form: a REAL fold-pair merge, germ-indeterminate (K2)

> **2026-06-12.** Slate-3 entry S3-A2-H8RF (`internal/slates/HYP_SLATE_3_EXTERNAL_ANCHORS_2026-06-11.md`);
> pre-registration `docs/atlas/H8_RF_CUSP_PREREG.md` (frozen 2026-06-11, before the first full-grid run).
> The owed substrate upgrade named by the banked isotropic null (`H8_DOUBLE_DESCENT_CUSP_RESULT.md`).
> NOT public-eligible. Attribution: Mei–Montanari (arXiv:1908.05355, the closed form); Nakkiran et al.
> (arXiv:2003.01897, regularization removal); Hastie et al. 2019; Thom/Whitney; the Atlas jet classifier.

## Headline verdict — **K2 (registered): finite merge of unresolved germ class**

Two findings, reported in their registered order:

1. **The structural question that killed the isotropic leg comes out the OTHER way here — K1 does NOT
   fire.** In the Mei–Montanari random-features closed form, the regularization-removal of model-wise
   double descent is a **genuine finite-interior fold-pair annihilation**: a bracketed interior max+min
   pair (the classical-regime min + the interpolation-side max) annihilates 2→0 at finite
   **(ψ₁\*, λ̄_c) = (3.8328, 0.064727)** on the primary slice (ψ₂=10, τ²=0.8), with no boundary exit, no
   escape (peak-ψ₁ 6.07→4.05 across the approach window), count exactly 2 throughout. The same holds on
   all five locus slices. The isotropic model's removal was a peak-escape to γ→∞; the RF model's removal
   is a real merge at a finite point — the mechanism-class difference the substrate upgrade existed to test.
2. **But the A3 germ label is INDETERMINATE under the pre-registered criterion.** The jet classifier reads
   exactly **1 cusp, corank-1, grid-stably (K4: ng-doubling + all 8 edge perturbations PASS)** on every
   slice — yet the |c₃| locus adjudication finds min shared-normalization ratio **0.475** (τ²=0.4 slice),
   inside the pre-registered germ-indeterminate band [0.25, 0.5), unchanged by the mandated ng-doubling
   escalation (0.475 → 0.475). The prereg's band rule is explicit ("no softer verdict may be reported from
   the band"), so the registered verdict is **K2** and the A3 claim is **dead as stated**.

## Updated germ table (the deliverable either way)

| substrate | removal mechanism | germ verdict |
|---|---|---|
| isotropic ridge (banked 2026-06-08) | single peak escapes/flattens to γ→∞ — **no fold pair** | NULL: not a catastrophe (peak-escape class) |
| Mei–Montanari RF (this run) | **finite-interior fold-pair annihilation** at (ψ₁\*, λ̄_c) | **K2: corank-1 merge of unresolved germ class** (A3-consistent on 4/5 slices; band-hit at the locus low end) |

## Gates (all passed before any verdict)

- **MC gate (ABORT-A-APPARATUS): 10/10 cells < 12%** (worst 10.66%, ψ₁=25/d=60 finite-size cell) —
  validating the Eq.-9 ζ pin over the paper's own intro-Eq.-4 variant (the named transcription
  discrepancy; the gate arbitrated as pre-registered).
- **Reference invariants: 4/4** (Fig-3 bump-at-small-λ̄ / monotone-at-large-λ̄ / R(ψ₁→0)→F₁²; Fig-1
  apparatus-tier max).
- **In-context controls (ABORT-B): 6/6** — Morin A4 dives (0.224 < 0.25), column A3 bounded (|c₃|=114),
  synthetic D4 fires corank-2 (0.0036), A4 chart corank-1, the banked isotropic chart re-run through THIS
  pipeline reproduces caustic-with-0-cusps, and the quartic ½-law control lands at slope 0.4953
  (R²=0.99998) → tol_cal=0.05.

## Registered measurements

| τ² | λ̄_c | ψ₁\* | #cusps | corank | shared-norm \|c₃\| | ratio to locus median |
|----|------|------|--------|--------|-----------------|----------------------|
| 0.4 | 0.038507 | 4.5723 | 1 | 1 | 59.72 | **0.475 ← band** |
| 0.8 (primary) | 0.064727 | 3.8328 | 1 | 1 | 95.31 | 0.758 |
| 1.2 | 0.087629 | 3.3896 | 1 | 1 | 125.70 | 1.000 |
| 1.6 | 0.108810 | 3.0709 | 1 | 1 | 153.05 | 1.218 |
| 2.4 | 0.148681 | 2.6153 | 1 | 1 | 197.94 | 1.575 |

Slice-validity pre-pins honored: τ² ≤ 0.2 excluded-with-report (bump dies below the λ̄=0.02 pole-recast
floor at high SNR — recorded in the prereg's probe); all five pinned locus slices valid; fixed-point
residuals ≤ 1.6e-14 everywhere.

## Post-verdict diagnostics (DESCRIPTIVE ONLY — `scripts/h8_rf_diag_postverdict.py`; verdict unaffected)

- **(D1) The scaling battery K3 was never adjudicated** (the lattice stops at K2). Measured descriptively:
  fold-pair separation slope **0.5258, R² = 0.99963, 12/12 points** — within the earned tol_cal of the A3
  normal-form ½-law.
- **(D2) The |c₃| trend is normalization-dominated, not germ-dominated:** under per-slice
  SELF-normalization the locus ordering **reverses** (τ²=0.4 becomes the LARGEST at 133.65; τ²=2.4 the
  smallest at 57.79; min ratio 0.725, all ≥ 0.5). The shared-normalization monotone trend (59.7→197.9)
  that produced the band hit tracks each slice's risk scale, consistent with the known gauge
  non-invariance of pointwise |c₃| (the HS14 ticket's subject: c₃ → det(Dψ)·μ²·c₃).

## Honest boundaries

- The K2 verdict is the registered outcome and stands. The band rule fired at 0.475 — 5% below threshold —
  on a criterion that compares |c₃| across charts of different physical scale under one shared
  normalization; the diagnostics suggest (but do not establish) that this is an instrument-gauge artifact
  rather than germ structure. **Adjudicating that requires a NEW pre-registration, not a reinterpretation.**
- Single-slice corank-1 + the descriptive ½-law are A3-*consistent* but were pre-registered as
  insufficient alone; that demotion was deliberate (locus-level |c₃| evidence) and is honored.
- The anchor remains ADJACENT (Timaeus board is LLC-centric; no LLC computed here).
- Prior-art zero verified at title+abstract tier only; full-text re-verification owed before any external
  priority claim.

## Named follow-up (owed, new prereg required: H8-RF-v2)

A gauge-aware germ criterion, pre-registered from the start: per-slice self-normalized |c₃| (or the
HS14 trend-exponent rule β from |c₃|(h) ~ (h−h\*)^β), with the K3 battery promoted to co-primary, and the
shared-vs-self normalization choice itself calibrated on the Morin A4 locus before first run. Natural
bundle: the HS14 jet-classifier gauge-fidelity ticket (same instrument, same gap, parked T3 on the
2026-06-10 slate). If v2 confirms A3, the germ table closes; if the band recurs under a gauge-fixed
criterion, "unresolved germ class" hardens into a real boundary of the instrument.

## Run engineering notes

- One apparatus bug found and fixed PRE-VERDICT (in the controls gate, before any kill was evaluated):
  `bisect_drop`'s termination test was sign-incorrect for negative control values (the quartic tol_cal
  control's h<0), causing an infinite loop; fixed with an `abs()` termination bound. The registered run
  completed end-to-end after the fix; no verdict-bearing code path changed.
- A first launch (2026-06-11 evening) died with the host session mid-controls-gate; the run is fully
  deterministic and was restarted from scratch — no scientific impact.

## Files

- `docs/atlas/H8_RF_CUSP_PREREG.md` — the frozen pre-registration (grids, gates, kill lattice, probe record).
- `scripts/double_descent_cusp_rf.py` — closed form + MC gate + controls + the registered adjudication.
- `scripts/h8_rf_diag_postverdict.py` — the descriptive post-verdict diagnostics (labeled, non-amending).
- `scripts/test_double_descent_cusp_rf.py` — frozen test locking every number above + the K2 verdict.
