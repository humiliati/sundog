# Mesa Phase 6 v3.4 — Substrate-Restricted Ablation and Jaccard Bootstrap

This document is the implementation-grade spec for Phase 6 v3.4 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 6 v3.3 (see
[`PHASE6_V33_RESULTS.md`](PHASE6_V33_RESULTS.md)) closed the
single-neuron-criticality question (AA1 falsified: no neuron has
ablation cost ≥ 0.3) and surfaced two positive findings:

- **AA2 confirmed strongly on P→C (Jaccard 0.049 vs v3.2 L2-rank),
  partially on C→P (0.362).** Ablation-rank and L2-rank measure
  different things, especially in the basin-inducing direction.
- **AA3 confirmed strongly (Jaccard 0.049 between P→C and C→P
  top-32).** The basin-inducing and basin-resisting sub-circuits
  occupy nearly-disjoint neuron substrates at `net.7`.

The AA3 finding is the load-bearing positive result of v3.3. v3.4
asks two questions about it that v3.3 did not test directly:

1. **Is the disjoint substrate functional?** The Jaccard 0.05 is an
   *anatomical* statement about ranking-set overlap. The functional
   statement — *ablating the P→C critical neurons breaks the P→C
   patch more than it breaks the C→P patch* — has not been directly
   tested.
2. **Is the Jaccard 0.05 statistically reliable, or seed-noise?** 8
   seeds × 256 neurons leaves room for sampling variation in the
   per-direction ranking. A bootstrap CI tells us whether 0.05 is a
   stable structural number or a noisy point estimate.

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6 v3.4.

## 1. Decision Lock

Phase 6 v3.4 starts with six pinned calls:

- **Two axes: P and Q.** Axis P (substrate-restricted ablation,
  functional dissociation test) and Axis Q (Jaccard bootstrap,
  statistical reliability test). Cross-policy substrate-generalization
  (apply the cliff-pair critical sets to J1/J2) is Path B and is
  deferred to v3.5 even if v3.4 confirms; deployment-time monitoring
  (Path C) is deferred to v3.6+ or to Phase 7 v2.
- **The cliff pair is the central artifact.** L-Mixed-M-λ=0.95 vs
  λ=0.97 again. v3.4 reuses the v3.3 critical-neuron sets directly
  from `results/mesa/phase6-v3-3-ablation/full/critical-top-32-{pc,cp}.csv`.
- **No new PPO training runs.** v3.4 is pure validation; no new
  policies, no new PCA basis, no new SAE training.
- **Ablation is set-to-zero, matching v3.3.** Apply the v3.2 axis-M
  patch hook with the v3 K=5 PCA delta, then force `h_new[j] = 0`
  for every j in the critical-neuron set being tested. Symmetric to
  v3.3's single-neuron ablation, just extended to a set.
- **Seed slate is 16 seeds for Axis P, 1000 resamples for Axis Q.**
  16 seeds doubles v3.3's per-direction baseline so the per-direction
  patch_success drop is measurable with reasonable noise. Q bootstraps
  on v3.3's existing 8-seed ablation table; new compute is in Axis P only.
- **Smoke gate before Axis P full run: cliff-pair K=5 baseline
  reproduces.** Run the v3 K=5 unablated patch on 16 seeds first;
  median P→C must land within 0.05 of the v3 published value (0.922)
  and median C→P within 0.05 of 0.830. This catches harness drift
  before v3.4 attributes a real effect to ablation. ~3 min.

Total v3.4 compute: 0 new PPO runs, ~30 LOC harness extension,
~10-15 minutes wall-clock for both axes combined.

## 2. Scope

Phase 6 v3.4 owns:

- One new subcommand `axis-p-substrate-ablation` in
  `training/mesa/phase6_v2_sae.py` — runs the v3 K=5 patch with a
  set of neurons zero-ablated, accepting the critical-neuron set from
  a CSV path argument.
- One new subcommand `axis-q-jaccard-bootstrap` — pure offline analysis
  on the v3.3 ablation table; resamples seeds, recomputes per-direction
  top-32 ablation-rank sets, computes Jaccard per resample, reports CI.
- Harness extension: extend the existing `zero_ablate_neuron: int | None`
  kwarg on `run_subspace_injected_rollout` to also accept a
  `zero_ablate_neuron_set: list[int] | None` for set-level ablation.
- v3.4 result note with functional dissociation outcome, Jaccard CI,
  and the BB1/BB2 prediction classifications.

Phase 6 v3.4 does **not** own:

- New PPO training runs.
- Recomputation of the PCA basis (uses v3 K=5 cached artifact).
- Recomputation of the v3.3 critical-neuron sets (uses v3.3 output as
  ground truth).
- Cross-policy ablation on J1/J2 (Path B, deferred to v3.5).
- Deployment-time monitoring probes (Path C, deferred to v3.6+).
- Integrated gradients or causal scrubbing (deferred to v3.7+ if
  v3.4 fails to validate).

## 3. Axes

### 3.1 Axis P — Substrate-restricted ablation

The functional-dissociation test. The v3.3 anatomical finding is that
the top-32 ablation-rank sets for P→C and C→P share Jaccard ≈ 0.05.
Axis P asks: *does ablating the P→C set break P→C patch_success more
than it breaks C→P patch_success — and symmetrically?*

**Protocol:** for each (mask_source, seed) cell:

1. Load critical-neuron set `S` from v3.3 output (P→C top-32 or
   C→P top-32).
2. Run the v3 K=5 patch with all neurons in `S` zero-ablated at the
   injection hook. Use the same hook variant as v3.3 but with a set
   instead of a single index:
   ```python
   delta = Q @ (target_c - current_c)
   h_new = h + delta
   h_new[S] = 0                                    # set-level ablation
   return h_new
   ```
3. Measure patch_success in *both* directions: P→C and C→P. The
   single ablation source yields two patch_success readings.

Compare against:
- v3 K=5 unablated baseline (P→C 0.922, C→P 0.830).
- v3.3 single-neuron ablation max (P→C 0.040, C→P 0.096).
- v3.2 L2-rank top-32 baseline (P→C 0.00, C→P 0.00).

Four readings per seed × two mask sources (P→C-critical, C→P-critical)
× two patch directions (P→C, C→P) = 16 patch_success numbers per seed.
With 16 seeds: 256 patch_success values feeding 8 medians (mask_source
× direction × stat).

### 3.2 Axis Q — Jaccard bootstrap

The statistical-reliability test. v3.3 reported a single Jaccard
number (0.049) between the P→C and C→P top-32 sets, computed once
from the 8-seed ablation table. Axis Q resamples seeds with
replacement and reports a 95% CI on the Jaccard.

**Protocol:**

1. Load v3.3's `ablation-table.csv` (8 seeds × 256 neurons × 2
   directions).
2. For each resample r ∈ {1..1000}:
   - Draw 8 seeds with replacement from the original 8.
   - For each direction, compute mean ablation cost per neuron across
     the resampled seeds, sort descending, take top-32.
   - Compute Jaccard between the resampled P→C top-32 and C→P top-32.
3. Report 2.5 / 50 / 97.5 percentiles of the Jaccard distribution.

**Reference Jaccard values:**

| value | meaning |
| ---: | --- |
| 0.000 | perfectly disjoint sets |
| 0.067 | chance-level — random 32-of-256 vs random 32-of-256 (E[intersection] = 32²/256 = 4; Jaccard = 4/60) |
| 0.200 | "modestly overlapping" |
| 0.400 | "substantially overlapping" |
| 1.000 | identical sets |

The observed v3.3 value of 0.049 is *below* chance-level (0.067) by
a small margin. Axis Q's question is whether that gap is reliable —
i.e., is 0.049 a statistical signature of *anti-correlation* between
the two rankings (one direction's critical neurons are *less* likely
to be the other direction's than chance would predict), or could it
plausibly be just chance with bootstrap noise?

## 4. Pre-Registered Predictions

Two load-bearing predictions, plus one optional gated prediction.

### 4.1 (BB1) Substrate-restricted ablation dissociates functionally

For both mask sources, the patch_success drop on the same-direction
patch is at least 0.15 larger than the drop on the cross-direction
patch:

```
drop(P→C-critical, P→C)  −  drop(P→C-critical, C→P)  ≥  0.15
drop(C→P-critical, C→P)  −  drop(C→P-critical, P→C)  ≥  0.15
```

where `drop(mask, dir) = baseline_PS(dir) − ablated_PS(mask, dir)`.

**Falsifier:** either dissociation is < 0.05 (no functional separation),
or one dissociation is > 0.15 but the other is < 0.05 (one direction
dissociates, the other doesn't). The latter is a "partial dissociation"
result that's still publishable but should be flagged as asymmetric in
the v3.4 result note.

Why 0.15 as the threshold: the v3 K=5 baseline gap between directions
is small (P→C 0.922 vs C→P 0.830, gap 0.09), and v3.3 single-neuron
ablation costs maxed at 0.10. A 0.15 dissociation in set-level
ablation is meaningfully larger than either single-neuron noise floor.

### 4.2 (BB2) Jaccard 0.049 is statistically reliable

The 95% bootstrap CI on the cliff-pair P→C vs C→P top-32 Jaccard has
its upper bound ≤ 0.20 ("modestly overlapping").

**Falsifier:** 95% CI upper bound > 0.30. The 0.049 point estimate is
not stable; the disjoint-substrate claim should be softened to
"ablation-rank-different" rather than "anatomically separable."

**Sub-prediction (BB2-strong):** 95% CI upper bound ≤ 0.10. The
disjoint-substrate finding is *below chance-level* with statistical
confidence — the two directions' critical neurons are systematically
*anti-correlated*, not just non-overlapping. This would be a stronger
structural claim than v3.3 made.

### 4.3 (BB3, optional) v3.3 vs v3.2 L2-rank Jaccard is also stable

Bootstrap the Jaccard between v3.3 P→C critical top-32 and v3.2
L2-rank top-32. v3.3 reported 0.049; Axis Q can bootstrap this too
for free (same resampling apparatus, different paired sets).
Prediction: 95% CI upper bound ≤ 0.20. Falsifier: upper bound > 0.30.

This is a free secondary diagnostic, not a gating prediction.

## 5. Cliff-Pair Manifest (no new policies)

| policy_id | label | tier | role |
| --- | --- | --- | --- |
| `mixed_lambda_0_95_medium_v4` | L-Mixed-M-λ=0.95 | Medium | protected |
| `mixed_lambda_0_97_medium_v4` | L-Mixed-M-λ=0.97 | Medium | collapsed |

## 6. Metrics

### 6.1 Axis P (substrate-restricted ablation)

For each (mask_source, direction, seed):
- `baseline_patch_success_{direction}` — v3 K=5 unablated.
- `ablated_patch_success_{mask}_{direction}` — with `S` ablated.
- `ablation_drop_{mask}_{direction} = baseline - ablated`.

Per-mask aggregates across 16 seeds: mean / median / 25th/75th percentile.

Headline diagnostic per mask source: **`dissociation = same_dir_drop −
cross_dir_drop`**. The v3.4 result note headline is two numbers — the
P→C-mask dissociation and the C→P-mask dissociation.

### 6.2 Axis Q (Jaccard bootstrap)

Per resample r:
- `top32_pc_resample[r]` (set of 32 neuron indices).
- `top32_cp_resample[r]`.
- `jaccard_pc_vs_cp_resample[r]`.
- `jaccard_pc_vs_l2_resample[r]` (for BB3).

Aggregates: 2.5 / 25 / 50 / 75 / 97.5 percentiles per Jaccard variant.

## 7. Harness Extension

`training/mesa/phase6_v2_sae.py` gains two subcommands:

```python
# Axis P
def axis_p_substrate_ablation(args: argparse.Namespace) -> None:
    """Run the v3 K=5 patch with a set of critical neurons zero-ablated."""
    # 1. Load critical-neuron set from --neuron-mask-source CSV
    # 2. Load v3 K=5 PCA basis (use load_or_build_cliff_pca_basis)
    # 3. For each seed, run the 4-forward battery with set-level ablation
    # 4. Compute baseline (no ablation) + ablated + dissociation per direction
    # 5. Write per-seed CSV + aggregate CSV + dissociation summary

# Axis Q
def axis_q_jaccard_bootstrap(args: argparse.Namespace) -> None:
    """Bootstrap the Jaccard on v3.3 ablation rankings. Pure offline."""
    # 1. Load v3.3 ablation-table.csv
    # 2. Group by (seed, direction, neuron), compute per-direction mean costs
    # 3. For r in 1..1000: resample seeds, recompute top-32 per direction, Jaccard
    # 4. Write bootstrap-jaccard.csv + bootstrap-summary.json
```

Required harness extension on `run_subspace_injected_rollout`: extend
the existing `zero_ablate_neuron: int | None` kwarg with
`zero_ablate_neuron_set: list[int] | None`. Mutually exclusive with the
single-neuron variant. ~10 LOC.

CLI:

```bash
# Smoke gate: K=5 baseline reproduces on 16 seeds
python -m training.mesa.phase6_v2_sae axis-h-pca \
    --num-components 5 --seeds 16 --layer net.7 \
    --out results/mesa/phase6-v3-4/baseline-k5

# Axis P: substrate-restricted ablation, P→C critical mask
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
    --seeds 16 --layer net.7 \
    --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-pc.csv \
    --out results/mesa/phase6-v3-4/axis-p-pc-mask

# Axis P: substrate-restricted ablation, C→P critical mask
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
    --seeds 16 --layer net.7 \
    --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-cp.csv \
    --out results/mesa/phase6-v3-4/axis-p-cp-mask

# Axis Q: Jaccard bootstrap, offline analysis
python -m training.mesa.phase6_v2_sae axis-q-jaccard-bootstrap \
    --ablation-table results/mesa/phase6-v3-3-ablation/full/ablation-table.csv \
    --l2-rank-source results/mesa/phase6-v3-2-neuron-mediation/axis-m-cliff-pair/top-32/neuron-ids-top-32.csv \
    --resamples 1000 \
    --out results/mesa/phase6-v3-4/axis-q-bootstrap
```

Total v3.4 harness extension: ~80 LOC (two subcommands, the
`zero_ablate_neuron_set` kwarg extension, the CSV loader for
critical-neuron sets, the Jaccard computation utility).

## 8. Outputs

```
results/mesa/phase6-v3-4/
  baseline-k5/                        # smoke gate output
    axis-h-pca-patch.csv
    axis-h-pca-patch-aggregate.csv
  axis-p-pc-mask/
    substrate-ablation.csv            # (seed, direction, baseline, ablated, drop)
    substrate-ablation-aggregate.csv
    dissociation-summary.json         # {pc_dir_drop, cp_dir_drop, dissociation}
  axis-p-cp-mask/
    substrate-ablation.csv
    substrate-ablation-aggregate.csv
    dissociation-summary.json
  axis-q-bootstrap/
    bootstrap-jaccard.csv             # 1000 rows: r, jaccard_pc_vs_cp, jaccard_pc_vs_l2
    bootstrap-summary.json            # CI percentiles
  reports/
    summary.json                       # BB1, BB2, BB3 outcomes
    v3-3-vs-v3-4-comparison.csv       # how v3.4 numbers map onto v3.3 critical-set framing
```

## 9. Execution Order

Recommended sequencing for Phase 6 v3.4:

1. **Harness extensions land first.** Extend `run_subspace_injected_rollout`
   with `zero_ablate_neuron_set` kwarg; add `axis-p-substrate-ablation`
   and `axis-q-jaccard-bootstrap` subcommands. Smoke-compile.
2. **Smoke gate (Axis-h-pca K=5 at 16 seeds).** Verify K=5 baseline
   reproduces (within 0.05 of v3 published values). ~3 min.
3. **Axis Q bootstrap.** Pure offline, runs in ~10 sec. Surfaces
   Jaccard CI immediately. Lets us see BB2 outcome before committing
   compute to Axis P.
4. **Axis P, P→C-critical mask.** 16 seeds × 4 forwards × 1 mask =
   64 forwards ≈ ~2 min.
5. **Axis P, C→P-critical mask.** Same compute.
6. **Aggregate + classify BB1, BB2, BB3.**
7. **v3.4 result note** at `docs/mesa/PHASE6_V34_RESULTS.md`.

Total wall-clock: ~10-15 minutes.

## 10. Exit Criterion

Phase 6 v3.4 complete when:

- `axis-p-substrate-ablation` and `axis-q-jaccard-bootstrap`
  subcommands land in `phase6_v2_sae.py`.
- Smoke gate passes (K=5 baseline reproduces).
- Axis P runs for both mask sources at 16 seeds; dissociation
  computed per mask.
- Axis Q bootstrap runs for 1000 resamples; CIs computed for both
  Jaccard variants.
- BB1, BB2 classified as confirmed or falsified (BB3 reported
  alongside but not gating).
- v3.4 result note (`PHASE6_V34_RESULTS.md`) is written.

## 11. Cross-References

- **Phase 6 v3.3 spec / results:** the AA3 disjoint-substrate finding
  v3.4 validates. [`PHASE6_V33_SPEC.md`](PHASE6_V33_SPEC.md),
  [`PHASE6_V33_RESULTS.md`](PHASE6_V33_RESULTS.md).
- **Phase 6 v3.2 spec / results:** the L2-rank Axis M data v3.4 uses
  for BB3. [`PHASE6_V32_SPEC.md`](PHASE6_V32_SPEC.md),
  [`PHASE6_V32_RESULTS.md`](PHASE6_V32_RESULTS.md).
- **Phase 6 v3.1 Axis L:** the seed-bootstrap precedent v3.4 Axis Q
  follows. [`PHASE6_V31_SPEC.md`](PHASE6_V31_SPEC.md) §3.4.
- **Phase 6 v3 (v2/v3):** the K=5 PCA basis v3.4 reuses unchanged.
  [`PHASE6_V2_SPEC.md`](PHASE6_V2_SPEC.md),
  [`PHASE6_V2_RESULTS.md`](PHASE6_V2_RESULTS.md).
- **Crossover note:** [`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md)
  carries the field-shape framing v3.4 contributes to.
- **Roadmap:** [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md).

## 12. What v3.5+ Inherits

If BB1 + BB2 both confirm:

- **The disjoint-substrate finding ratchets from anatomical to
  functional + statistical.** Cascade language in PROMO_HIGHLIGHTS,
  claims-and-scope, SUNDOG_V_GRAVITY, mesa.html upgrades from "nearly
  disjoint neuron substrates" to "**functionally and statistically
  separable** basin-inducing vs basin-resisting circuits at net.7."
- **v3.5 routes to Path B (cross-policy substrate generalization).**
  Run zero-ablation on J1 and J2 (the v3.1 generalization pairs).
  Compute Jaccard between cliff-pair P→C critical set and J1/J2 P→C
  critical sets. If high Jaccard → controller-family-wide basin-inducing
  substrate. Compute ~140 min.
- **v3.6 routes to Path C (deployment monitoring) if v3.5 confirms.**
  Train a probe at net.7 on the family-wide critical-neuron set;
  evaluate as a deployment-time basin-collapse predictor across the
  Phase 5 zoo.

If BB1 falsifies (no functional dissociation):

- The v3.3 anatomical claim is methodological noise rather than
  structural fact. The disjoint-substrate language must be withdrawn
  from the cascade. mesa-trap mechanistic anchor stays at "5D
  entangled subspace, non-decomposable to neurons by either L2-rank,
  single-neuron ablation, or critical-set ablation."
- v3.5 routes to integrated gradients or causal scrubbing as
  originally pre-named in PHASE6_V32_RESULTS §7.

If BB2 falsifies (Jaccard CI is wide):

- The 0.049 point estimate is unreliable; the v3.3 disjoint-substrate
  claim is unrobust. Public-claim language softens; v3.5 routes to
  increased seed count on v3.3-style ablation to tighten the
  Jaccard estimate.

## 13. Versioning

- **v3.4 (2026-05-13)** — initial pin. Two axes: Axis P (substrate-
  restricted ablation, functional dissociation test) and Axis Q
  (Jaccard bootstrap, statistical reliability test). Two pre-registered
  predictions BB1-BB2 plus one optional BB3. Cliff pair only; J1/J2
  generalization deferred to v3.5 Path B; deployment monitoring
  deferred to v3.6 Path C. Smoke gate: K=5 baseline reproduces at 16
  seeds. Compute: ~10-15 min, 0 new PPO runs, ~80 LOC harness
  extension.
