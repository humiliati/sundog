# v0.19 liao2021 Floquet Spectral-Gap Reliability Mechanism Form

## Result (2026-06-03)

**Verdict: `spectral_gap_mechanism_partial`.** The first-principles mechanism for the
*reliability itself* is confirmed strongly; the per-cell bridge to AUC via the chosen
aggregation is not.

```text
reproduce-v0.18 gate:  max_abs_diff 0.00e+00  -> PASS (re-measurement reproduces v0.18
                       vf/score/frame_spread exactly; spectrum data trustworthy)
H1 (primary, label-blind):  Spearman(re_gap, frame_spread) = -0.836, p = 1.0e-5  -> PASS
                       (past the |0.45| bar); small spectral gap drives high frame-fragility
falsifier guard:       large-gap-high-spread fraction = 0.0052 (15/2880; GAP_CLEAR
                       0.0008368)  -> clean (<= 5%)
H2 (secondary):        per-cell Spearman(median gap_reliability, AUC_cell) = 0.063,
                       p = 0.40  -> NOT confirmed
```

**H1 is the headline and it lands hard: frame-fragility IS spectral near-degeneracy.**
The argmax-of-`Re(lambda)` selection flips under reparameterization exactly where the top
Floquet spectrum clusters, and that drives the four-frame `frame_spread` (rho -0.836,
2880 reproduce-gated orbits). The v0.18 phenomenological reliability map has a
first-principles cause at the orbit level: the spectral gap that makes the selection
well-posed.

**Why H2 fails -- aggregation mismatch, not mechanism refutation.** v0.18 predicted
per-cell AUC from the cell's frame-spread TAIL (`frame_p90`, rho 0.60). v0.19's H2 used
the cell's MEDIAN `gap_reliability` -- a central measure that washes out the fragile
minority that actually depresses AUC. The chain small-gap -> high-spread (H1, confirmed)
-> AUC-degradation (v0.18, confirmed) is intact, but the specific per-cell MEDIAN-gap
summary does not bridge it. A tail-based cell gap measure (cell min `re_gap`, or fraction
of near-degenerate orbits) would likely bridge it -- but that is a new locked statistic,
not a re-read, so v0.19 honestly stops at `partial`.

**Cross-substrate export (chatv2 handoff):** supported at the FRAGILITY level -- an
argmax-selected shadow's reparameterization-instability is its selection spectral gap,
computable label-blind a-priori (Floquet Re-gap / residual-stream eigengap / control-basis
gap). NOT yet supported as a full per-region transfer-strength predictor via this
aggregation.

**Independent verification.** A standalone hand-rolled rank/Pearson Spearman + falsifier
recompute from `per_row_spectrum.csv` reproduced H1 rho -0.836220, GAP_CLEAR 0.000836786,
falsifier 0.0052 (15/2880) BIT-FOR-BIT. Receipt + verifier:
`results/isotrophy/k-facet-v19-spectral-gap-reliability/` (`manifest.json`,
`gap_frame_spread_test.json`, `reproduce_v18_check.json`, `per_row_spectrum.csv`,
`_independent_check.py`).

**Bounded, unchanged:** Tier-2 / Li-Liao lineage / stable-support / within-cell /
tail-resolved score / not theorem-facing. H1 is the load-bearing first-principles result;
H2/H3 are downstream/exploratory. This explains the reliability mechanism (altitude 2,
gap -> fragility); it does NOT explain why vf -> stability (altitude 1, deferred).

**Provenance note:** the v0.19 chapter (runner + this form + npm aliases) was orphaned by
the 2026-06-03 leak-remediation history rewrite (committed at the now-dangling `88241ec9`);
recovered to the working tree from that commit 2026-06-03. The 2880-row measurement
survived (gitignored results).

---

Status: **OPERATOR LOCK 2026-06-03; VERDICT LANDED `spectral_gap_mechanism_partial` 2026-06-03.**
Reviewed for self-consistency: the sign is
confirmed (H1 negative on `re_gap` vs `frame_spread`; H2 positive on
`gap_reliability = log10(re_gap)` vs `AUC_cell`); `re_gap` / `selected_group_re_width` /
`selection_sv_gap` map onto `select_gamma_1`'s `Re >= max_real - 1e-6` group + cascade
SVD; falsifier is procedure-pinned; the reproduce-v0.18 gate is binding. Runner
implemented after lock; no v0.19 measurement has run and no v0.19 statistic computed at
lock time. This is the first **first-principles mechanism** chapter of the isotrophy
program: it does not collect more external-transfer evidence (the v0.11->v0.18 arc
is closed/consolidated). It asks *why* the reliability map exists.

## Frame

v0.18 registered that label-blind frame reliability (`-log10 frame_p90`) predicts per-cell
transfer AUC (Spearman rho 0.5975, p 0.00523, reversal guard clean). That is a
**phenomenological** reliability map: it says reliable cells transfer better, but not why
reliability varies. v0.19 proposes and tests a first-principles cause.

**The candidate mechanism is nearly forced by how the shadow is built.** `select_gamma_1`
picks the Floquet eigenvector of largest `Re(lambda)`. A v0.13a/b frame perturbation
re-integrates a *rotated copy* of the orbit; its monodromy is a similarity conjugate, so
the **eigenvalue spectrum is identical** and only the numerical argmax can move. The argmax
can move only when two distinct modes have near-equal `Re(lambda)` -- a small **Re-part
spectral gap** (near-degeneracy). Therefore:

```text
frame-fragility (argmax flips under reparameterization)  <=>  small Re-part spectral gap
the selected-direction shadow is reliable exactly when the selection is well-posed,
which is exactly when the top spectrum is gapped.
```

This is substrate-independent: it is the general law for any argmax / eigenvector /
top-component-selected shadow (chatv2 residual-stream directions, Mesa control bases, NSE
diagnostic projections). Confirming it would upgrade v0.18's reliability map into a
**spectral-gap map** with a portable predictor.

## Primary Question

> Does the Re-part Floquet spectral gap of the selected direction predict its
> frame-fragility (and hence the per-cell transfer AUC)? Is the v0.18 reliability map a
> spectral-gap map?

## Integrity Caveat

```text
theory-derived hypothesis (NOT discovered on data):
  The spectral-gap mechanism is derived from the argmax-conditioning argument above, not
  fit to any prior sample. The primary statistic (gap -> frame_spread) is label-blind on
  both sides and was never measured in v0.11-v0.18.

mechanism on the discovery rows is appropriate:
  A mechanism chapter EXPLAINS a found result on the rows where it was found. Unlike a
  confirmation chapter, it does not require a fresh outcome sample. The load-bearing
  primary (gap -> frame_spread) involves neither the stability label nor AUC, so it
  carries no discovery-sample circularity even on v0.18 rows.

feature + grid frozen:
  score = median(vf_0, vf_37, vf_90, vf_211); frozen v0.7/v0.13a D5; v0.14 8x8 grid
  construction; seed 20260523. The ONLY new object is the per-orbit monodromy spectrum
  dump and the spectral-gap statistic derived from it.
```

Forbidden:

```text
changing the score, frame set, D5 gates, grid, or seed
using the stability label or AUC in the primary gap->frame_spread predictor
reading the spectral gap off the selected direction's own vf (the gap is a spectrum
  property, computed before any velocity/position split)
lowering the locked correlation bars after seeing the spectrum
promoting to Tier-3, theorem-facing, or a controller claim
reopening or revising any v0.11-v0.18 verdict (this chapter explains, it does not amend)
```

## Sample -- Resolved Lock Default

```text
LOCK DEFAULT: re-measure the exact v0.18 sample (2880 rows, the 18 8x8 cells).
  - The mechanism explains the v0.18 reliability map on the rows that produced it.
  - The re-measurement reproduces v0.18's vf / score / frame_spread BIT-FOR-BIT
    (deterministic same-orbit re-run) as a built-in validation, then ADDS the spectrum.
  - No fresh draw, so no holdout exhaustion.

REJECTED ALTERNATIVE: fresh sixth-holdout draw.
  - A fresh 8x8 quintuple holdout is razor-thin: smallest eligible stable pool = 84
    (only 4 above the 80 draw) -- abort-prone. A fresh draw would need 10x10 (23 cells,
    smallest pool 112, 3680 rows) for headroom.
  - Cleaner for the per-cell gap->AUC test, but the primary gap->frame_spread test gains
    nothing (it is already label-blind), and it costs ~30% more compute on an exhausting
    stable pool.
```

Rationale: the verdict-bearing primary is label-blind, so re-measuring the v0.18 rows
is both clean and the most direct explanation of the map under study; the data limit
makes a fresh 8x8 draw fragile.

## New Measurement (spectrum dump)

Re-run the frozen four-frame ensemble; additionally persist, per orbit, at the **identity
frame** monodromy:

```text
floquet_eigs_real, floquet_eigs_imag   full 18-multiplier spectrum, not top-k
max_abs_lambda                         spectral radius -> elliptic vs hyperbolic type
selected_re, selected_im               the select_gamma_1 eigenvalue
re_gap                                 Re(selected) - Re(nearest competing DISTINCT mode),
                                       using the existing select_gamma_1 degeneracy
                                       tolerance to exclude the conjugate partner / the
                                       selected near-degenerate group
degenerate_eigenvalue_count            (the v0.7/v14 count, as a gap cross-check)
eigenspace_dim_used                    (cross-check)
selected_group_re_width                max-min Re(lambda) inside the selected group
selection_sv_gap                       report-only: cascade-step singular gap inside the
                                       selected group, when eigenspace_dim_used > 1
frame_spread                           reproduced from the four frames (must match v0.18)
```

The spectrum is frame-invariant, so it is computed once at the identity frame; the four
frames are still run to reproduce `frame_spread`.

## Spectral-Gap Definition (locked)

```text
re_gap = Re(lambda_selected) - max{ Re(lambda_j) : mode_j not in the selected
         near-degenerate group, at the select_gamma_1 degeneracy tolerance }
gap_reliability = log10(max(re_gap, 1e-9))   (large gap -> larger value -> reliable)
gap_unreliability = -gap_reliability         (small gap -> larger value -> fragile)
```

The conjugate partner and any modes within the selection's degeneracy tolerance are
excluded from the competitor, because they are handled by `select_gamma_1` as one
selected group. This makes the primary test specifically about the Re-part gap to the
next competing group. `degenerate_eigenvalue_count`, `selected_group_re_width`, and
`selection_sv_gap` are reported as cross-checks so any within-group cascade fragility is
visible instead of being silently folded into H1.

## Hypotheses (locked directions + bars)

```text
H1 (PRIMARY, verdict-bearing, label-blind):
  per-orbit, a SMALL Re-part gap drives HIGH frame_spread.
  statistic: Spearman rho(re_gap, frame_spread) over all success orbits.
  bar: rho <= -0.45 AND one-sided permutation p <= 0.01
       (100000 label-free shuffles of the gap/spread pairing, seed 20260523;
       p tail = shuffled rho <= observed rho).

H2 (SECONDARY, downstream): the v0.18 reliability map is a spectral-gap map.
  statistic: per-cell Spearman(cell median gap_reliability, AUC_cell) over the cells.
  bar (report + soft): rho >= 0.45 AND one-sided p <= 0.05 (100000 cell-level
  shuffles, seed 20260523; p tail = shuffled rho >= observed rho; mirrors v0.18).
  On v0.18 rows this is partly mediated by H1 + the known frame_spread->AUC link; it
  is confirmatory, not independent, and is labeled as such.

H3 (EXPLORATORY, report-only -- altitude 1, the deeper why):
  spectral TYPE and selected-eigenvector geometry vs vf and stability:
    - is unstable ~ a separated hyperbolic mode (max_abs_lambda > 1+eps), position-heavy
      selected vector (low vf)?
    - is stable ~ clustered elliptic modes (all |lambda| ~ 1), velocity-heavy (high vf)?
  Reported as a descriptive panel (per-orbit type, gap, vf, label). It cannot set the
  verdict; it scopes a possible dynamical "why vf->stability" for a later chapter.
```

## Reproduce-v0.18 Validation (gate)

Before any H1/H2 interpretation, assert the re-measurement reproduces v0.18 on the shared
columns:

```text
re-measured vf_0/37/90/211, score, frame_spread == v0.18 per_row_sample values,
max abs diff <= 1e-9 over all 2880 orbits.
```

If this fails, verdict is `blocked_by_nonreproduction` -- the spectrum dump perturbed the
measurement and nothing is interpreted.

## Falsification Guard

The mechanism must be falsifiable:

```text
GAP_CLEAR = 66.7th percentile of re_gap over successful v0.18 re-measure rows
            (procedure-pinned, label-blind)
SPREAD_HIGH = 0.10

large_gap_high_spread_orbit = re_gap >= GAP_CLEAR AND frame_spread >= SPREAD_HIGH
```

A non-trivial population of large-gap-yet-fragile orbits would mean the argmax flips for
reasons other than near-degeneracy, falsifying H1's mechanism even if the pooled rho is
negative. Report the count and fraction; pre-register that `> 5%` of success orbits in
this box demotes H1 to `mechanism_partial`.

## Verdict Tree

```text
blocked_by_receipt
  v0.18 receipts / sample frame / spectrum dump unavailable or inconsistent

blocked_by_nonreproduction
  re-measured frame_spread/vf/score do not match v0.18 within 1e-9

spectral_gap_explains_reliability
  H1 passes (rho <= -0.45, p <= 0.01) AND large-gap-high-spread fraction <= 5%
  AND H2 confirmatory (rho >= 0.45, p <= 0.05)

spectral_gap_mechanism_partial
  H1 passes OR is borderline, but H2 weak OR large-gap-high-spread fraction > 5%

spectral_gap_mechanism_not_supported
  H1 fails (rho > -0.45 OR p > 0.01)
```

## Required Outputs

```text
results/isotrophy/k-facet-v19-spectral-gap-reliability/
  manifest.json
  per_row_spectrum.csv         per-orbit: eigs, re_gap, gap_reliability,
                               max_abs_lambda, vf, frame_spread, degenerate_count,
                               eigenspace_dim, selected_group_re_width,
                               selection_sv_gap, stability, mass_cell
  reproduce_v18_check.json     bit-for-bit validation vs v0.18
  gap_frame_spread_test.json   H1 Spearman + permutation null + falsification-guard counts
  per_cell_gap_reliability.csv H2 per-cell gap_reliability vs AUC_cell
  spectral_type_panel.csv      H3 elliptic/hyperbolic + vf + label (report-only)
  operator_commands.md
```

`manifest.json` must include verdict, the H1 rho + p + N, `GAP_CLEAR`, `SPREAD_HIGH`,
the large-gap-high-spread fraction, the H2 per-cell rho + p, the reproduce-v18 max abs
diff, gitCommit, target_sha256, sample provenance (`v0.18_remeasure`), and the claim
boundary.

## Claim Boundary

If `spectral_gap_explains_reliability`:

```text
The v0.18 frame-reliability map is, first-principles, a Floquet Re-part spectral-gap map:
the selected-direction shadow is fragile exactly where the top spectrum is near-degenerate,
and that fragility is what depresses per-cell transfer AUC. This is a mechanism for the
RELIABILITY, computed label-blind from the monodromy spectrum.
```

Bounded, unchanged: Tier-2 / Li-Liao lineage / stable-support / within-cell / tail-resolved
score / not coarse-zone / not full-catalog / not Tier-3 / not theorem-facing. H1 is the
load-bearing first-principles result; H2 is confirmatory on the discovery rows; H3 is an
exploratory scope for a later "why vf->stability" dynamical chapter, not a claim.

This chapter does NOT explain *why velocity-fraction tracks stability* (altitude 1) -- it
explains *why the transfer's reliability varies* (altitude 2). The two are distinct.

## Cross-Substrate Export

If H1 holds, the exportable principle for CROSS_SUBSTRATE_NOTES / chatv2 is concrete:

```text
an argmax / top-component / eigenvector-selected shadow transfers in proportion to the
spectral gap that makes its selection well-posed; compute that gap and you have a
label-blind, a-priori reliability predictor -- in three-body (Floquet Re-gap), in an LLM
residual stream (top-direction eigengap), in Mesa (control-basis gap), in NSE (diagnostic
projection gap).
```

This is the chatv2-facing payoff: a portable reliability predictor, not a re-run.

## Resolved Decisions Before Lock

```text
1. Sample: re-measure the exact v0.18 rows; fresh 10x10 is rejected for this mechanism
   chapter because H1 is label-blind and the fresh pool is needlessly exhausting.
2. Spectrum persistence: write the full 18 multipliers; top-k is rejected because full
   spectrum is cheap and needed for honest H3 elliptic/hyperbolic typing.
3. H1 bar: keep rho <= -0.45 and p <= 0.01; stricter -0.5 is rejected because this is
   the first first-principles mechanism test and -0.45 already mirrors the v0.18
   reliability magnitude.
4. Falsification box: GAP_CLEAR = success-row 66.7th percentile of re_gap; SPREAD_HIGH
   = 0.10; large-gap-high-spread fraction > 5% demotes to partial.
5. H3 stays strictly report-only; altitude-1 "why vf->stability" deserves its own
   chapter.
```

## Lock-In Statement

Committed as a draft before any v0.19 runner exists. Any change to the spectral-gap
definition, the hypotheses, the bars, the falsification guard, the reproduce-v0.18 gate,
the verdict tree, or the claim boundary after execution is a re-registration. The runner
reuses the frozen v0.16/v0.18 ensemble + D5 + grid symbols unchanged and adds only the
identity-frame spectrum dump, the `re_gap` statistic, and the gap/frame_spread/AUC tests.
