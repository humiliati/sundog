# Mesa Phase 6b - Large-Tier Activation Patching Result Note

This document records the Phase 6b result note for the Large-tier
activation-patching battery defined in
[`PHASE6B_SPEC.md`](PHASE6B_SPEC.md). Phase 6b is a Large sibling of
Phase 6 v1 (Medium `net.7` localization of the basin-attractor
circuit). The v1 22-cell Medium result is unchanged.

Status: Phase 6b v1.1 **complete with pre-registered falsification**.
GG6b-loc-D called (destructive-transfer falsifier at all 5 layers in
both directions); GG6b-mech not called (per §6 of the spec, mech is
not evaluated at destructive layers); GG6b-shape deferred (no locus
to decompose). The v1 single-layer locus finding **does not
generalize** to the Large field-coupled / under-budget cliff pair.

## 1. Summary

Phase 6b ran the Phase 6 v1 axis-b activation-patching protocol on
the Large cliff pair (mixed_0_99 recovery vs mixed_0_97 trough,
`--value-coef 0.25`, 10M env-steps, `seed=10000`), sweeping the 5
post-Tanh hidden layers (`net.1`, `net.3`, `net.5`, `net.7`, `net.9`)
at 64 seeds per cell, both `clean` and `intervened` conditions, both
patch directions, and producing per-layer per-direction aggregate
metrics. The smoke-v11 (n=8 at net.9) caught a v1 metric pathology
(`old_basin_pref` gap ~1.0 made the v1 metric unbounded); the
v1.1 amendment introduced an alignment-normalized parallel metric
and a conjunctive P4 rule (mean ∧ median ∧ ratio_of_means all ≥ 0.8)
plus a destructive-transfer falsifier (`|patched_align −
target_align| / |gap_align| > 0.5`).

The full-v11 sweep (2026-05-18, ~13.5 min wall-clock) yields one
substantive finding:

**At every swept layer at Large, activation injection between
mixed_0_99 and mixed_0_97 drives the recipient policy's terminal
alignment to ~0 (range 0.01–0.09) — below even the recipient's own
un-patched baseline.** The patching is *actively destructive of
navigation* rather than transferring behavior between sides. This
fires the destructive-transfer falsifier at every (layer, direction)
cell and routes the verdict to GG6b-loc-D.

The destruction is not a metric artifact — it reads off raw
`mean_patched_alignment`, a physical per-seed quantity that cannot
explode. The verdict therefore stands independent of the conjunctive
P4 reading of the normalized `patch_success_align` metric (which
would also yield "no layer clears" at the conjunctive rule, but
that's secondary).

## 2. Envelope

`clean` rows, all 5 swept layers, both patch directions. n=64
per cell. `align_gap = 0.4267` (= `mean_protected_alignment 0.912 −
mean_collapsed_alignment 0.486`). Rescue target = protected
alignment 0.912; break target = collapsed alignment 0.486.

| layer | dir | mean ps_align | median | ratio | patched_align | vs target / gap | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| net.1 | rescue (P→C) | 1.442 | -0.482 | -1.043 | 0.041 | 2.043 | **DESTRUCTIVE** |
| net.1 | break (C→P) | -0.162 | 1.407 | 1.983 | 0.066 | 0.983 | **DESTRUCTIVE** |
| net.3 | rescue (P→C) | 0.928 | -0.453 | -1.106 | 0.014 | 2.106 | **DESTRUCTIVE** |
| net.3 | break (C→P) | 0.065 | 1.485 | 2.070 | 0.029 | 1.070 | **DESTRUCTIVE** |
| net.5 | rescue (P→C) | 0.974 | -0.457 | -1.076 | 0.027 | 2.076 | **DESTRUCTIVE** |
| net.5 | break (C→P) | -0.026 | 1.285 | 1.924 | 0.091 | 0.924 | **DESTRUCTIVE** |
| net.7 | rescue (P→C) | 0.862 | -0.457 | -1.004 | 0.057 | 2.004 | **DESTRUCTIVE** |
| net.7 | break (C→P) | 0.076 | 1.464 | 2.103 | 0.015 | 1.103 | **DESTRUCTIVE** |
| net.9 | rescue (P→C) | 0.779 | -0.485 | -1.056 | 0.035 | 2.057 | **DESTRUCTIVE** |
| net.9 | break (C→P) | 0.058 | 1.460 | 2.103 | 0.015 | 1.103 | **DESTRUCTIVE** |

`patched_align` is `mean_patched_alignment`; `vs target / gap` is
`|mean_patched_alignment − target_align| / |align_gap|` (the
destructive-transfer test from spec §6: > 0.5 fires the
falsifier). Every cell exceeds 0.5; ten of ten gap-delta values lie
in the range 0.92–2.11.

The `intervened` rows are bit-identical to `clean` rows at all five
layers (the v1.1 clean/intervened bit-identity bug, see §6 below).
Verdicts read off `clean` rows only, per v1.1 acceptance criteria;
the destructive finding is condition-independent because it's a
raw alignment quantity.

## 3. Artifacts

Per-sweep outputs:

`results/mesa/phase6b-large-cliff-pair/full-v11/`

Files:

- `manifest.json` — `cliff_pair: large-v3`, both checkpoints, 64
  seeds × 5 layers × 2 conditions per cell.
- `axis-b-patch-smoke.csv` — per-(layer, condition, seed) raw
  rollout metrics with the v1.1 alignment columns.
- `axis-b-patch-smoke-aggregate.csv` — per-(layer, condition,
  direction) aggregates with both metric column families.

Earlier sweep artifacts preserved as methodological receipts:

- `results/mesa/phase6b-large-cliff-pair/smoke/` — v1 smoke (8 seeds
  net.9, basin-pref metric), captured the metric pathology that
  drove the v1.1 amendment.
- `results/mesa/phase6b-large-cliff-pair/full/` — v1 full sweep
  (64 seeds × 5 layers, basin-pref metric only), no verdict called.
  See [`PHASE6B_SPEC.md`](PHASE6B_SPEC.md) §6.A.
- `results/mesa/phase6b-large-cliff-pair/smoke-v11/` — v1.1 smoke
  (8 seeds net.9, both metrics), surfaced the mean-vs-median
  disagreement and drove the spec §6 P4-conjunctive-rule
  refinement.

## 4. GG Verdicts (pre-registered)

### GG6b-loc — **GG6b-loc-D called (Phase 6 v1 protocol doesn't generalize)**

The spec required, for any of GG6b-loc-{A, B, C}, that *some* layer
clear the conjunctive P4 rule (mean ∧ median ∧ ratio_of_means
all ≥ 0.8) in at least one direction without firing the
destructive-transfer falsifier. The full-v11 sweep produces:

- **Conjunctive P4 clears:** zero cells. No (layer, direction)
  satisfies mean ≥ 0.8 ∧ median ≥ 0.8 ∧ ratio_of_means ≥ 0.8.
  Per-statistic clears: mean ≥ 0.8 in **4 cells** (rescue rows for
  net.1 1.442, net.3 0.928, net.5 0.974, net.7 0.862; net.9 rescue
  0.779 misses; all 5 break-row means are near zero), median ≥ 0.8
  in **5 cells** (all break rows; break-direction medians
  1.285–1.485), ratio_of_means ≥ 0.8 in **5 cells** (all break
  rows; ratios 1.924–2.103). The mean and median/ratio statistics
  agree on opposite directions — mean clears on rescue rows where
  median/ratio do not, median/ratio clear on break rows where mean
  does not — and the three statistics agree on the conjunction at
  zero cells.
- **Destructive-transfer falsifier:** fires at *all 10 cells* (5
  layers × 2 directions). Gap-delta range 0.92–2.11, all well above
  the 0.5 threshold.

Per spec §6, destructive cells contribute to GG6b-loc-D, not
A/B/C. With every cell destructive, **GG6b-loc-D is unambiguously
called**: the Phase 6 v1 activation-patching mechanism does not
transfer between the Large recovery and trough policies. The v1
single-layer locus finding does not generalize.

### GG6b-mech — **not called**

Per spec §6, "GG6b-mech is not called at a layer that fired
GG6b-loc-D." Every swept layer fired GG6b-loc-D, so the
rescue-vs-break asymmetry question never gets a layer to evaluate
at. GG6b-mech-{A, B, C} are all undischarged.

The smoke-v11 mean-vs-median direction debate (mean 5.42 looking
like rescue clears, median 0.05 saying no transfer) is fully
resolved by the full-sweep destructive-transfer reading: both
directions destroy the recipient's navigation rather than
transferring behavior. The asymmetry the mean alone hinted at was
small-n outlier noise on top of a destructive substrate.

### GG6b-shape — **deferred**

Per spec §6 GG6b-shape-C: "GG6b-loc-A/B did not find a locus, so
the substrate-shape question doesn't apply at this layer." There
is no locus at any swept layer, so the 5D PCA decomposition
question (does the Medium net.7 entangled-subspace finding scale
to Large?) is structurally undefined for this cliff pair under
this protocol. Phase 6b v2 (or a different cliff-pair selection)
picks up the substrate-shape question if a non-destructive layer
ever surfaces.

## 5. Why GG6b-loc-D Is Robust (Not Another Metric Artifact)

The smoke-v11 audit raised the question: is the v1.1 metric
genuinely sound at Large, or is it still producing artifacts? The
full-sweep verdict rests on a chain of three independent readings,
the first of which is metric-free:

1. **Raw `mean_patched_alignment` (physical, not normalized).**
   Across all 10 cells, the patched policy terminates at alignment
   0.01–0.09 — *below* either reference (collapsed 0.486 / protected
   0.912). This is a per-seed mean of a physical quantity; it cannot
   be exploded by a small denominator. Patching at every swept layer
   drives the recipient policy into "alignment near zero" territory,
   which is roughly random-walk terminal performance.
2. **Destructive-transfer falsifier.** Independent of any
   normalization, `|patched_align − target_align| / |gap_align| > 0.5`
   fires at all 10 cells with gap-deltas 0.92–2.11. The patched
   policies are 1–2 gaps away from their intended target on either
   side. This is the spec §6 falsifier, fired across the entire
   layer sweep.
3. **Conjunctive P4 disagreement.** The mean, median, and
   ratio_of_means of `patch_success_align` disagree systematically:
   rescue rows have positive means (0.78–1.44) but negative medians
   (-0.45 to -0.49) and negative ratios (-1.00 to -1.11); break
   rows have near-zero means (-0.16 to 0.08) but positive medians
   (1.29–1.49) and positive ratios (1.92–2.10). No cell satisfies
   the conjunction in either direction. This is the v1 P4 discipline
   inherited verbatim and applied to the v1.1 metric.

Three independent readings, three confirmations of the same
finding: **the v1 activation-patching protocol does not transfer
behavior between the Large mixed_0_99 and mixed_0_97 policies at
any layer in the actor's MLP**. The verdict is robust against any
one metric being still-broken because it doesn't rest on one
metric.

## 6. v3-Consistency Interpretation

GG6b-loc-D fits cleanly into the v3 finding shape and sharpens it.
The v3 receipt
([`PHASE7_V3_RESULTS.md`](PHASE7_V3_RESULTS.md) §5) established:

- The Large U-trough cells (mixed_0_95, mixed_0_97) are
  `field-coupled, under-budget`, **not** the Medium `reward-coupled`
  collapse class.
- mixed_0_99 (the recovered side at Large) is `field-coupled` with
  basin-attractor avoidance (GG4-A).
- Neither side of the Large cliff pair internalizes the basin
  (`old_basin_pref` 0.46 / 1.47, both well below the Medium
  collapse floor 5.56).

The Phase 6 v1 mechanism question — "where in the network does the
basin attractor live?" — is well-posed at Medium **because** Medium
L-Mixed-0.97 has internalized the basin attractor. The cliff
between L-Mixed-0.95 (basin-resisting) and L-Mixed-0.97
(basin-internalized) is a real circuit-level boundary, and v1's
patching transfers it cleanly at net.7.

At Large, *neither side* has internalized a basin attractor. The
recovery-vs-trough cliff at Large is something else — both sides are
field-coupled, both have low basin-pref, the difference is *navigation
competence under high reward weight*, not basin internalization. So
the Phase 6 v1 protocol asks the wrong mechanistic question for
Large: there is no basin-attractor circuit to transfer because
neither side has one. Cross-policy activation injection just breaks
the recipient because the policies have different internal
representations and no shared mesa-trap substrate to swap.

**The Phase 6b verdict is therefore consistent with the v3 verdict
and tightens it.** v3 said the Large cliff is not a collapse cliff;
6b says the v1 mechanism-level test that *would have* probed a
collapse cliff finds no collapse-cliff-circuit at any layer. The
two results agree at substrate level.

## 7. What Phase 6b v1.1 Says, Plainly

The earned reading:

> At Large capacity (actor parameter_count 4,207,618 ≈ ~4.2M; see
> [`PHASE6B_SPEC.md`](PHASE6B_SPEC.md) §4) in the tested shadow-field
> navigation family, the v3 cliff between λ=0.99 recovery and
> λ=0.97 trough is **not localized to a single MLP layer** under
> the Phase 6 v1 activation-patching protocol. At every swept
> layer (net.1, net.3, net.5, net.7, net.9), cross-policy
> activation injection between the recovery and trough policies
> drives the recipient's terminal alignment to ~0 — actively
> destroying navigation rather than transferring behavior. This is
> consistent with the v3 finding that the Large cliff is not a
> collapse-class boundary: neither side internalizes a
> basin-attractor, so there is no transferable basin circuit to
> probe.

Hard scope:

- *In-vitro*, mesa shadow-field family only, single-seed Large
  policies (seed=10000), PPO-specific `vc=0.25` knob.
- Phase 6 v1 activation-patching protocol; alternative protocols
  (e.g. multi-layer co-injection, attention-style key/value editing
  for non-MLP architectures) are not tested by GG6b-loc-D and could
  in principle find different mechanism shapes.
- The "destructive-transfer" reading describes what happens to the
  *patched* policy's terminal alignment; it does not claim the
  unpatched policies are themselves destructive or unstable.
- *Not* a claim that Large policies lack mechanistic structure;
  it's a claim that v1's specific cross-policy-injection test does
  not surface that structure between this pair.

The result is **publishable as a pre-registered falsification**.
GG6b-loc-D was written into the spec precisely to catch the
"protocol doesn't generalize" outcome cleanly, and the verdict
discharges it.

## 8. Open Edges

- **clean / intervened bit-identity bug.** Identified in
  [`PHASE6B_SPEC.md`](PHASE6B_SPEC.md) §6.A on the first sweep
  (2026-05-18, v1 metric), persists at v1.1 — `intervened` rows
  are bit-identical to `clean` rows at all five layers across both
  metrics. v1.1 verdicts read `clean` rows only per acceptance
  criteria. The bug doesn't change the destructive-transfer reading
  (which is condition-independent) but should be investigated
  before any future v2 sweep that wants intervention as a
  discriminator. The intervention list is built in
  `run_patched_rollout` and forwarded to the bridge `make`
  request; the bridge-side `make` handler at the Large path is the
  likely inspection point.
- **Alternative cliff-pair selection.** Phase 6b v1.1 falsifies the
  v1 protocol for the mixed_0_99 / mixed_0_97 pair specifically. A
  v2 attempt might select a pair with a genuine basin-pref gap
  (e.g. signature_terminal_large vs reward_phase3_seg3_large, gap
  ~6.7) and ask the v1 question of *that* pair — though as the v3
  GG5 receipt established, reward_phase3 seg3 is
  `bootstrap-collapse` (degenerate fixed trajectory, not a
  competent policy), so cross-policy injection introduces a
  competence-axis confound. The "right" cliff pair for a v1-analog
  question at Large may not exist within the v3 envelope.
- **Multi-layer co-injection.** v1.1 patches one layer at a time;
  v1's Medium result also localized to a single layer. A v2
  protocol that co-injects activations at multiple layers (e.g.
  net.7 + net.9 simultaneously) might surface a multi-layer
  distributed mechanism that the single-layer sweep misses. Not in
  6b scope.
- **GG6b-shape (5D entangled subspace) at Large.** Structurally
  undefined for this pair under this protocol; deferred to v2 or
  to a non-destructive cliff pair if one is identified.

## 9. Cross-References

- **Spec source:** [`PHASE6B_SPEC.md`](PHASE6B_SPEC.md) v1 (initial
  spec) → v1.1 (metric reframe + conjunctive P4 + destructive-
  transfer falsifier).
- **v1 reference receipt:** [`PHASE6_RESULTS.md`](PHASE6_RESULTS.md)
  v1 (Medium net.7 localization, P→C 0.894/0.944/0.899, C→P
  0.934/0.860/0.854) — the v1 conjunctive P4 reading carried
  forward to v1.1.
- **v3 driver:** [`PHASE7_V3_RESULTS.md`](PHASE7_V3_RESULTS.md) §5
  (Large cliff is field-coupled-under-budget, not collapse) and §7
  (observation-pathway hint that motivated GG6b-mech; not called).
- **Harness:** `training/mesa/phase6_probes.py` — `axis-b-smoke`
  subcommand with `--cliff-pair large-v3` flag. The v1.1 amendment
  added `terminal_alignment` to `PatchRollout`, the parallel
  alignment-normalized aggregate columns, and conjunctive console
  output.

## 10. Versioning

- **v1.1 (2026-05-18)** — initial Phase 6b result note. Full-v11
  sweep at 64 seeds × 5 layers × 2 conditions, ~13.5 min wall-clock.
  GG6b-loc-D called (destructive-transfer at every cell, 10 of 10);
  GG6b-mech not called (per spec §6, mech is not evaluated at
  destructive layers); GG6b-shape deferred (no locus to decompose).
  Verdict robustness rests on three independent readings: raw
  `mean_patched_alignment` (physical), destructive-transfer
  falsifier (independent of normalization), conjunctive P4 rule
  on the v1.1 alignment-normalized metric. v3-consistency analysis
  in §6 frames GG6b-loc-D as a v3-confirming finding rather than a
  v3-falsifying one: the Phase 6 v1 protocol asks the wrong
  mechanistic question for a pair where neither side internalizes
  a basin-attractor. clean / intervened bit-identity bug carried
  forward as an open edge; does not change this verdict.
