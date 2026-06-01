# v0.13b Frame-Zone Stability Audit Form Lock

Status: **OPERATOR LOCK 2026-05-31; VERDICT LANDED `coarse_zone_rule_frame_stable_enough_to_test`.**
A tiny, locked diagnostic -- not a transfer step. It measures the EXACT frame-zone-change
fraction so the v0.13a frame-sensitivity finding can be priced precisely, rather than
proxied by the |dvf| severity buckets.

## Result (2026-05-31)

**Verdict: `coarse_zone_rule_frame_stable_enough_to_test`.** Both locked bars cleared.

```text
supp-B    exact zone-change fraction = 4.35%  (2/46, Wilson95 [1.2%, 14.5%])  <= 0.15 bar
          -> v0.11 remains VALID with a MODEST frame caveat.
liao2021  pooled exact zone-change fraction ~= 1.3%  (10/789 across the n=200 + the
          pre-registered n=600 expansion; n=600 alone 3/599 = 0.5%, Wilson95
          [0.17%, 1.46%]; n=200 7/190 = 3.7%)  <= 0.05 bar
          -> cross-ansatz testing is NOT blocked by frame fragility.
```

Receipts: `results/isotrophy/k-facet-v13b-frame-zone-stability/` (supp-B + liao2021
n=50) and `.../liao2021_n600/` (the n=600 expansion).

**Fragility is localized, not uniform.** All 10 liao2021 zone-changers across both
samples fall in two narrow mass-ordered catalog bands (orbit index ~74.7k-84.7k and
~109.8k-116.6k); zero zone-changes anywhere else in 135,445 rows. The n=200/n=600
disagreement (3.7% vs 0.5%) is this heterogeneity showing through small samples, not a
bug -- some mass-tuple regions have near-degenerate dominant Floquet directions
(frame-fragile vf) while the rest are frame-robust. **This argues for stratified
frame-stability diagnostics in the eventual transfer readback** -- the catalog-wide
~1.3% does not guarantee every `m3` stratum is frame-robust; a stratum overlapping the
fragile bands must be checked per-stratum.

**Honest object (restated):** not "dominant-direction velocity-fraction is a frame-
invariant geometric quantity" (v0.13a refuted that), but "a coarse, frame-dependent
projection (the v0.11 zone) whose registered bins are empirically frame-stable under the
tested frame changes -- on average and on the v0.11 domain, with frame-fragility
localized to identifiable mass regions." v0.11's within-supp-B statistics are unchanged;
only the feature's geometric interpretation is downgraded from invariant to
coarse-frame-stable.

## Frame

v0.13a's parity found the raw `velocity_fraction` is frame-relative for most orbits
(`select_gamma_1`'s largest-real-part direction selection flips under coordinate
rotation when Floquet real-parts cluster). The |dvf| severity proxy then suggested the
v0.11 ZONES are mostly frame-robust (liao2021 0% / supp-B ~11% severe), but a |dvf|
threshold both under- and over-counts zone changes near a cutpoint. v0.13b records the
EXACT zone change, `zone(base) != zone(rotated)`, under the already-locked frame
perturbations.

The proxy is good enough to save the idea; it is not good enough to spend the claim.

## Primary Question

> Under the exact v0.13a frame perturbations, does the frozen v0.11 `zone_index(vf)`
> change?

`zone_index` is frozen from v0.11: `0` if vf < 0.25, `1` if 0.25 <= vf < 0.50, `2`
if vf >= 0.50. The cutpoints {0.25, 0.50} are not changed.

## Locked Records (per row, per frame)

```text
base_zone                       zone_index of the base (unperturbed) orbit
rotated_zone                    zone_index under the perturbed frame
zone_changed                    base_zone != rotated_zone
distance_to_nearest_cutpoint    min(|vf_base - 0.25|, |vf_base - 0.50|)  -- robustness margin
```

Per orbit, `orbit_zone_changed = any(zone_changed over the frames)`.

## Locked Frame Perturbations

The exact v0.13a isometry set (excluding the identity):

```text
rotations  SO(2) by {37, 90, 211} degrees in the orbit plane
translation by the v0.13a fixed offset (absorbed by CoM-centering)
```

Applied to the CoM-centered initial state, integrated through the frozen D5 path
(`integrate_liao2021_state` explicit-state wrapper + `compute_monodromy_vectorized` +
`select_gamma_1` + `velocity_fraction_and_z_fraction`), all imported byte-for-byte.

## Locked Domain + Sample

```text
catalogs        supp-B (the v0.11 domain; frozen mirrored expand_initial_state)
                liao2021 (the v0.13a adapter; expand_liao2021_state)
sample          n = 50 uniform rows per catalog, seed 20260523 (program seed)
                integration-failed rows are reported and excluded from the fraction
```

## Decision Bars

```text
supp-B  exact zone-change fraction <= 0.15  -> v0.11 frame caveat is MODEST
liao2021 exact zone-change fraction <= 0.05 -> cross-ansatz zone transfer remains VIABLE
```

Verdict tree:

```text
supp-B  <= 0.15  AND  liao2021 <= 0.05  ->  coarse_zone_rule_frame_stable_enough_to_test
supp-B  >  0.15                          ->  v0.11_language_downgrade (zone rule is too
                                             frame-sensitive on its own domain)
liao2021 > 0.05                          ->  cross_ansatz_zone_transfer_shaky
```

A Wilson 95% CI on each fraction is reported; if a fraction straddles its bar, the
diagnostic is INCONCLUSIVE on that axis and the sample is expanded before any claim.

## Firewall

```text
forbidden: any S/U conditioning, any vf-vs-stability statistic, any AUC/J_cond/chi-sq,
           any target selection, recording the raw velocity_fraction value
allowed:   recording base_zone / rotated_zone / zone_changed / distance_to_cutpoint
           SOLELY as a frame-stability measurement, never associated with a stability
           label, never used to select or score a transfer
```

This is a frame-stability audit of a frozen projection, not a test of the signal.

## Claim Reframing (the honest object)

If both bars pass, the theorem-facing object is restated, humbler but usable:

> NOT "dominant-direction velocity-fraction is a frame-invariant geometric quantity"
> (v0.13a shows it is not), BUT "a coarse, frame-dependent projection (the v0.11 zone)
> whose registered bins are empirically frame-stable under the tested frame changes."

v0.11's within-supp-B statistics are unchanged; only the geometric interpretation of
the feature is downgraded from invariant to coarse-frame-stable.

## Output Contract

```text
results/isotrophy/k-facet-v13b-frame-zone-stability/
  frame_zone_{catalog}.json   summary: sampled, checked, integration_failed,
                              orbit_zone_changed_count, zone_change_fraction,
                              wilson95, per_frame_change_rate, distance_quantiles,
                              decision_bar, axis_verdict
  frame_zone_{catalog}.csv    per row: orbit_index, base_zone, zone_changed,
                              distance_to_nearest_cutpoint, per-frame rotated_zone
```

No receipt records raw `velocity_fraction` or any stability label.

## Lock-In Statement

Committed before the v0.13b runner is written. Any change to the records, the frame
set, the sample, the decision bars, the firewall, or the claim reframing after runner
execution is a re-registration. The runner imports the frozen D5 + v0.13a symbols
unchanged and adds only the zone-derivation + zone-change accounting.
