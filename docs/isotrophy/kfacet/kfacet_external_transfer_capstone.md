# K_facet External Transfer Capstone

Status: **CONSOLIDATED 2026-06-03; UPDATED THROUGH v0.19.** This is a paper-side
synthesis of the v0.11 -> v0.19 external-transfer and reliability-mechanism arc.
It adds no new statistic, no new sample, and no new claim beyond the locked chapter
receipts.

## One-Sentence Claim

The Floquet velocity-fraction projection stratifies three-body stability after
conditioning on mass structure; read as a continuous tail-resolved score, it
transfers to the Li/Liao 2021 external catalog at Tier-2, replicates on fresh
rows, and its per-cell heterogeneity is explained by a registered label-blind
frame-reliability map whose per-orbit cause is Floquet spectral near-degeneracy.

## What Is Being Claimed

The body is the per-orbit Floquet/monodromy geometry of a three-body periodic
orbit. The shadow is a low-dimensional velocity-fraction readout: how much of a
selected Floquet direction lives in velocity coordinates rather than position
coordinates. The mature finding is conditional:

```text
velocity-fraction ranks stable above unstable rows inside the right mass strata;
it does not work as a mass-marginal predictor.
```

On the external Li/Liao 2021 target, the coarse v0.11 zone form saturates and
does not transfer at the registered magnitude floor. The continuous four-frame
ensemble-median score does transfer. The reliability of that four-frame read
then explains the otherwise puzzling per-cell heterogeneity.

## Evidence Ladder

| chapter | role | verdict / receipt |
|---|---|---|
| v0.10a | registers the ordered trend missed by chi-squared | Jonckheere-Terpstra exact p = 0.0073 |
| v0.10b | tests mass-marginal prediction | held-out pooled AUC = 0.4125, fails; within-bin signal remains |
| v0.11 | locks the conditional form | AUC_cond = 0.6783, exact p = 2.046e-7 |
| v0.12 | first external attempt | supp-A blocked by D5 attrition, not falsified |
| v0.13 | target search | no viable Tier-3 target; Tier-2 Li/Liao 2021 remains |
| v0.13a/b | leakage and frame audit | leakage 0.0; coarse zone frame-stable enough to test |
| v0.14 | sampled transfer | undecidable coverage: only 7/16 cells host both classes |
| v0.15 | stable-support case-control transfer | directional_weak: AUC_cond = 0.5125, p ~3e-4 |
| v0.16 | continuous tail-resolved transfer | passes_clean: AUC_cond = 0.6469866, p = 1e-5 |
| v0.17 | fresh replication + heterogeneity | AUC_cond = 0.646875, p = 1e-5; heterogeneity rho = 1.0 |
| v0.18 | reliability mechanism | AUC_cond = 0.620208, p = 1e-5; reliability rho = 0.5975, p = 0.00523 |
| v0.19 | first-principles reliability mechanism | H1 pass: Spearman(re_gap, frame_spread) = -0.836, p = 1e-5; H2 median-gap bridge misses |

The arc is complete because the result now has all four pieces: an internal
conditional positive, an external transfer, a fresh replication, and a
pre-registered mechanism for the replicated heterogeneity.

## What Failed

These failures are part of the result, not footnotes:

- The mass-marginal predictor failed. v0.10b showed a zone-only score cannot
  globally rank stability across mass bins because mass-bin base rates dominate
  the pooled ordering.
- The same-paper supp-A transfer was blocked by numerical attrition. v0.12 did
  not falsify velocity-fraction; it showed the frozen D5 measurement cannot
  support that target.
- The Tier-3 landscape was negative. v0.13 found independent catalogs were
  equal-mass, restricted-substrate, or too small for the registered transfer.
- The coarse v0.11 zone form did not transfer at magnitude. v0.15 showed the
  direction remained significant, but the target population was almost entirely
  zone-2, so the binning discarded the useful resolution.
- The result is not full-catalog prevalence. The external tests use stable-support
  outcome-balanced sampling and within-cell AUC.

## Mechanism

v0.16 and v0.17 produced a clean but heterogeneous external transfer: some mass
cells had strong in-direction AUC, while others dipped below 0.5. The v0.17
anatomy suggested an instrument-reliability map. v0.18 tested that explanation
fresh on an 8 x 8 grid:

```text
sample:          18 cells, 2880 rows, v14+v15+v16+v17 quadruple holdout
pooled transfer: AUC_cond = 0.620208, p = 1e-5
reliability:     Spearman rho(-log10 frame_p90, AUC_cell) = 0.5975, p = 0.00523
reversal guard:  0 stable-decisive-negative cells
```

The decisive negative cells were the frame-fragile cells. The only frame-stable
sub-0.5 cell did not clear the reverse-significance bar (`p_reverse = 0.0968`).
That matters because the reversal guard was capable of falsifying the mechanism:
a frame-stable decisive-negative cell would have shown a real competing local
law. None appeared.

v0.19 then asked what frame-fragility *is*. The answer is strong at the orbit
level: small Floquet Re-part spectral gap predicts high four-frame spread
(`rho = -0.836`, `p = 1e-5`) on the exact v0.18 rows, with the reproduce gate
bit-for-bit clean and the large-gap/high-spread falsifier nearly empty
(15/2880). The locked verdict is still `spectral_gap_mechanism_partial` because
the direct H2 bridge used a cell **median** gap summary and missed
(`rho = 0.063`): the v0.18 AUC relationship lives in the frame-spread tail
(`frame_p90`), and a median washes that tail out. So the honest synthesis is:

```text
spectral gap -> frame fragility (v0.19 H1, confirmed)
frame fragility tail -> per-cell AUC reliability map (v0.18, confirmed)
median spectral gap -> per-cell AUC (v0.19 H2, not confirmed)
```

The mechanism is therefore answered at altitude 2: reliability varies because
argmax-selected Floquet shadows become fragile near spectral degeneracy. The
specific median aggregation did not carry that mechanism to AUC.

## Claim Boundary

Allowed:

```text
The tail-resolved velocity-fraction projection transfers from supplementary-B to
Li/Liao 2021 at Tier-2 under stable-support, within-cell evaluation; the transfer
replicates on fresh rows; and per-cell transfer strength is explained by the
frame reliability of the selected Floquet direction.
```

Not allowed:

```text
Tier-3 independent validation
full-catalog prevalence
mass-marginal prediction
coarse-zone transfer
theorem-facing K_facet promotion
controller evidence
claiming velocity-fraction is frame-invariant geometry
```

The most compact public phrasing is:

> Velocity-fraction is a real conditional stability shadow, externally confirmed
> at Tier-2 when read continuously; its transfer heterogeneity is not noise, but
> a frame-reliability map, and that map is driven at the orbit level by spectral
> near-degeneracy of the selected Floquet direction.

## Sundog Lesson

This is one of the cleanest body/shadow examples in the repository. The shadow
works only after the held-fixed coordinate is named, only after the target can
host the comparison, and only when the feature's granularity matches the target
distribution. v0.18 and v0.19 add the reliability lessons:

```text
for eigenvector/argmax-selected shadows, per-region transfer strength can track
instrument reliability, and instrument reliability can be predicted label-blind
from the spectral gap that makes the selection well-posed.
```

That lesson is exportable. It should be tested in new threads, not by stretching
this lane:

- Mesa: do local mechanism probes become weaker in regions where the selected
  direction / feature basis is unstable under admissible reparameterizations?
- Navier-Stokes / control shadows: does transfer strength vary with the
  reliability of the diagnostic projection across flow regimes?

## Disposition

The evidence-collection lane is closed as a self-contained Tier-2 result. Further
in-lane work has low marginal value unless a genuinely independent Tier-3 catalog
appears. The next high-value move is communication: paper-side writeup, figures,
and a short external-review packet that foregrounds the nulls as guardrails rather
than as weaknesses.

Primary receipts:

- `docs/isotrophy/kfacet/kfacet_v11_m3_conditional_vf_rank_form.md`
- `docs/isotrophy/kfacet/kfacet_v16_liao2021_tail_resolved_transfer_form.md`
- `docs/isotrophy/kfacet/kfacet_v17_liao2021_heterogeneity_scope.md`
- `docs/isotrophy/kfacet/kfacet_v18_liao2021_reliability_auc_form.md`
- `docs/isotrophy/kfacet/kfacet_v19_spectral_gap_reliability_form.md`
- `results/isotrophy/k-facet-v18-liao2021-reliability-auc/manifest.json`
- `results/isotrophy/k-facet-v19-spectral-gap-reliability/manifest.json`
