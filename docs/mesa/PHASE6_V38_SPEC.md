# Mesa Phase 6 v3.8 - Per-PC Neuron Decomposition and Signed Effect Map

This document is the implementation-grade spec for Phase 6 v3.8 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). v3.7 closed the
three-layer cross-policy substrate table: P->C/basin induction is shared
at the 5D `net.7` control-surface level but pair-specific at top-32
neuron identity; C->P/basin resistance generalizes at both neuron
identity and function, with the J2 caveat that the transferred cliff
mask beats J2's own mask.

v3.8 pivots from "which mask transfers?" to "how is the 5D subspace
internally organized?" It is also the first Phase 6 spec explicitly
written as a mesa-to-geometry crossover deliverable for
[`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md).

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6 v3.8.

## 1. Decision Lock

Phase 6 v3.8 starts with six pinned calls:

- **Primary axis: Axis T, per-PC zero-ablation decomposition.** For
  each cliff-pair PCA component `PC_i` where `i in {1, 2, 3, 4, 5}`,
  restrict the patch delta to that single component and run the v3.3
  neuron-by-neuron zero-ablation battery. Output is a `5 x 256`
  ablation-cost matrix per direction.
- **Free-rider axis: Axis U, signed direction-of-effect map.** Reanalyze
  existing zero-ablation CSVs from the cliff pair plus held-out J1/J2
  to classify each neuron by whether ablation hurts, helps, or is near
  neutral for P->C and C->P patching.
- **Cliff pair only for Axis T v1.** Held-out-pair per-PC decomposition
  is deferred. The cliff pair is the source of the canonical PCA basis
  and the cleanest transfer surface for geometry.
- **Both directions for Axis T.** Per-PC rankings are computed for
  P->C and C->P. Directional asymmetry is a load-bearing outcome.
- **No new PPO training.** All work uses existing Medium policies,
  cached cliff-pair PCA basis, and existing policy loaders.
- **Crossover section required.** The result note must explicitly state
  which `MESA_CROSSOVER_NOTE.md` findings are sharpened and which
  Phase 10/11 geometry deliverables inherit the method.

Total v3.8 compute: 0 new PPO runs. Axis T: ~50 min wall-clock, ~80 LOC
harness extension. Axis U: ~5 min, ~30 LOC offline analysis.

## 2. Purpose

Phase 6 has established three layers:

1. The behavioral cliff localizes to `net.7`.
2. The full-layer patch compresses to an entangled 5D PCA subspace.
3. Direction-specific top-32 neuron substrates are functional, but
   cross-policy transfer differs by direction.

v3.8 asks whether the five principal components share the same neuron
substrate or partition into component-specific substrates.

The geometry-facing analogy is direct: if the mesa 5D subspace has
per-component neuron partitions, then geometry should expect inversion
routes like parhelion offset, CZA apex, tangent curvature, and
supralateral position to have distinct residual fingerprints. If the
same neurons recur across PCs, geometry should expect shared residual
sources across primitives and avoid over-interpreting per-primitive
independence.

## 3. Scope

Phase 6 v3.8 owns:

- Axis T per-PC zero-ablation on the cliff pair:
  - `PC1`, `PC2`, `PC3`, `PC4`, `PC5`
  - P->C and C->P directions
  - 8 seeds, `net.7`, horizon 200
- Axis T matrix outputs:
  - per-PC per-neuron ablation costs
  - per-PC top-32 rankings
  - PC-to-PC Jaccard overlap matrix
  - PC concentration / partition metrics
- Axis U signed-effect analysis:
  - cliff pair, J1, J2 zero-ablation tables
  - per-neuron sign class per direction
  - P->C/C->P opposition table
- Result note at `docs/mesa/PHASE6_V38_RESULTS.md`.
- Explicit crossover contribution section in the result note.

Phase 6 v3.8 does **not** own:

- New training runs.
- New policy pairs.
- Small-tier work.
- Pair-specific PCA basis derivation for J1/J2.
- Integrated gradients or causal scrubbing.
- Geometry implementation changes. v3.8 informs geometry; it does not
  edit the geometry docs unless a later roadmap task requests that
  cascade.

## 4. Axis T - Per-PC Zero-Ablation Decomposition

### 4.1 Protocol

Load the cached cliff-pair PCA basis:

`results/mesa/phase6-v3-1-validation/pca-basis/cliff-pca-net7-seed10000-n64-h200-k5.npz`

For each `pc_index in {1, 2, 3, 4, 5}`:

1. Construct `Q_pc = Q_full[:, pc_index:pc_index+1]`.
2. Run the existing v3.3 zero-ablation battery with `Q_np = Q_pc`.
3. Run both directions:
   - P->C: protected activations injected into collapsed policy.
   - C->P: collapsed activations injected into protected policy.
4. Aggregate per-neuron ablation cost exactly as v3.3 does.
5. Emit one result directory per PC.

Important interpretation guardrail:

Single-PC patch_success is expected to be partial or weak. Axis T is not
asking whether any single PC reproduces the full K=5 patch. v3.1 already
showed that no proper sub-subspace reproduces the full effect. Axis T
asks which neurons mediate each component's partial directional effect.

### 4.2 Outputs

```
results/mesa/phase6-v3-8/
  axis-t-per-pc/
    pc1/
      ablation-table.csv
      ablation-aggregate.csv
      critical-top-32-pc.csv
      critical-top-32-cp.csv
      summary.json
    pc2/
      ...
    pc3/
      ...
    pc4/
      ...
    pc5/
      ...
    reports/
      pc-neuron-ablation-matrix.csv
      pc-top32-jaccard.csv
      pc-partition-summary.json
```

`pc-neuron-ablation-matrix.csv` rows:

| pc | direction | neuron_idx | mean_ablation_cost | median_ablation_cost | sign |
| --- | --- | ---: | ---: | ---: | --- |

`pc-top32-jaccard.csv` rows:

| direction | pc_a | pc_b | jaccard_top32 |
| --- | ---: | ---: | ---: |

### 4.3 Partition Metrics

For each direction:

- `mean_offdiag_jaccard`: average Jaccard among the five PC top-32 sets.
- `max_offdiag_jaccard`: largest PC-to-PC Jaccard.
- `unique_top32_neuron_count`: size of union over five PC top-32 sets.
- `shared_core_count`: number of neurons appearing in top-32 for at
  least three PCs.

Interpretation:

- Low off-diagonal Jaccard + high union count = PCs partition into
  component-specific anatomical substrates.
- High off-diagonal Jaccard + high shared-core count = PCs share a
  common neuron substrate.
- Mixed pattern = some components partition, some share.

## 5. Axis U - Signed Direction-of-Effect Map

Axis U is an offline analysis of existing CSVs, not a new ablation run.

Inputs:

- `results/mesa/phase6-v3-3-ablation/full/ablation-table.csv`
- `results/mesa/phase6-v3-5/axis-n-j1/ablation-table.csv`
- `results/mesa/phase6-v3-5/axis-n-j2/ablation-table.csv`

For each `(pair, direction, neuron_idx)`, compute mean ablation cost.
Classify:

- `hurts_patch`: mean cost `>= +0.01`
- `helps_patch`: mean cost `<= -0.01`
- `neutral`: otherwise

Then compute opposition classes per pair:

| class | definition |
| --- | --- |
| P_hurts_C_helps | hurts P->C, helps C->P |
| C_hurts_P_helps | hurts C->P, helps P->C |
| both_hurt | hurts both directions |
| both_help | helps both directions |
| directional_neutral | all other sign combinations |

The threshold `0.01` is deliberately small but nonzero: enough to avoid
floating noise while preserving weak directional effects. v3.8 may
report sensitivity at `0.02` as a robustness appendix. EE3/EE4
classification is keyed to the preregistered `0.01` threshold; the
`0.02` pass is non-gating and exists to distinguish strong signed
structure from threshold-fragile signed structure.

### 5.1 Outputs

```
results/mesa/phase6-v3-8/
  axis-u-signed-effects/
    signed-neuron-effects.csv
    directional-opposition-summary.csv
    pair-sign-overlap.csv
    summary.json
  axis-u-signed-effects-threshold-0-02/   # non-gating sensitivity
    ...
```

## 6. What This Contributes to MESA_CROSSOVER_NOTE.md

v3.8 directly sharpens three crossover findings.

### 6.1 Finding #2 - Variance is not mechanism

Axis T separates variance components from causal neuron mediation. PC1
may explain the most activation-diff variance, but v3.1 showed it has
near-zero patch effect alone. Per-PC neuron rankings test whether the
high-variance component nevertheless has an anatomical substrate, or
whether low-variance PCs own the causal machinery.

Geometry inheritance:

- Phase 10 per-primitive residual reporting should not rank primitives
  by visual salience or variance explained.
- The mesa output becomes the methodological blueprint for reporting
  residuals by primitive route rather than total overlay fit.

### 6.2 Finding #5 - No linear attribution

Axis T is not a linear attribution score. It reports component-restricted
causal sensitivity and explicitly keeps the v3.1 guardrail that no
single PC is expected to add up to the full K=5 mechanism.

Geometry inheritance:

- Phase 11 metric review should reject "arc X contributes N% of fit"
  language.
- A safer analogue is "when primitive route X is held fixed/removed,
  residual class Y changes by Z."

### 6.3 Finding #3 - Directional asymmetry is structural

Axis U turns directional asymmetry into a signed map: which neurons help
one inversion direction while hurting the other?

Geometry inheritance:

- The geometry `do not promote` list should record not just primitives
  that fail to help inversion, but primitives/routes that actively
  worsen another inversion route's residual profile.

## 7. Pre-Registered Predictions

### 7.1 (EE1) PC neuron substrates are partially partitioned

For at least one direction, `mean_offdiag_jaccard <= 0.20` across the
five PC top-32 sets.

**Falsifier:** `mean_offdiag_jaccard >= 0.40` in both directions. The
PCs share a common neuron substrate; per-PC anatomical partitioning is
not present.

### 7.2 (EE2) PC1 remains variance-heavy but mechanism-light

PC1 has lower max mean ablation cost than at least three of PCs 2-5 in
the P->C direction.

**Falsifier:** PC1 has the highest max mean ablation cost in P->C. The
old "PC1 is only offset / mechanism-light" language would need another
revision.

### 7.3 (EE3) Directional opposition is nontrivial

Axis U finds at least 16 neurons in either `P_hurts_C_helps` or
`C_hurts_P_helps` on the cliff pair at sign threshold `0.01`.

**Falsifier:** fewer than 8 opposition-class neurons on the cliff pair.
The negative cross-drops seen in v3.4/v3.6/v3.7 are set-level effects
without stable per-neuron signed structure.

### 7.4 (EE4) C->P signed structure is more stable across pairs

Pair-sign overlap for C->P `hurts_patch` neurons exceeds P->C
`hurts_patch` overlap between cliff and both held-out pairs.

**Falsifier:** P->C sign overlap equals or exceeds C->P sign overlap
for both held-out pairs. That would weaken the v3.5-v3.7 story that
C->P is the more family-wide neuron substrate.

## 8. Harness Extension

Add two subcommands to `training/mesa/phase6_v2_sae.py`.

### 8.1 `axis-t-per-pc-zero-ablation`

Suggested CLI:

```bash
python -m training.mesa.phase6_v2_sae axis-t-per-pc-zero-ablation \
  --seeds 8 --direction both --pcs 1 2 3 4 5 \
  --out results/mesa/phase6-v3-8/axis-t-per-pc
```

Implementation shape:

1. Load `Q_full` with `load_or_build_cliff_pca_basis(..., num_components=5)`.
2. For each requested PC, call `run_zero_ablation_battery` with the
   one-column `Q_pc`.
3. Write per-PC v3.3-style outputs under `pc{i}/`.
4. Aggregate all per-PC `ablation-aggregate.csv` files into the matrix
   and Jaccard reports.

No environment or policy-loader changes are needed.

### 8.2 `axis-u-signed-effects`

Suggested CLI:

```bash
python -m training.mesa.phase6_v2_sae axis-u-signed-effects \
  --tables cliff=results/mesa/phase6-v3-3-ablation/full/ablation-table.csv \
           J1=results/mesa/phase6-v3-5/axis-n-j1/ablation-table.csv \
           J2=results/mesa/phase6-v3-5/axis-n-j2/ablation-table.csv \
  --threshold 0.01 \
  --out results/mesa/phase6-v3-8/axis-u-signed-effects
```

Implementation shape:

1. Load each ablation table.
2. Aggregate mean ablation cost by `(pair, direction, neuron_idx)`.
3. Apply sign classification.
4. Emit sign overlap and opposition summaries.

## 9. Execution Order

Recommended sequence:

1. Implement `axis-u-signed-effects` first (~30 LOC).
2. Run Axis U on existing CSVs plus `0.02` sensitivity (~5 min including
   inspection).
3. Implement `axis-t-per-pc-zero-ablation` (~80 LOC).
4. Run Axis T on cliff pair, 5 PCs x both directions x 8 seeds
   (~50 min).
5. Generate `pc-neuron-ablation-matrix.csv`,
   `pc-top32-jaccard.csv`, and `pc-partition-summary.json`.
6. Classify EE1-EE4.
7. Write `PHASE6_V38_RESULTS.md`.
8. If results are clean, update the Phase 6 status block in
   `SUNDOG_V_MESA.md` and cite the crossover contribution.

## 10. Exit Criterion

Phase 6 v3.8 is complete when:

- Axis U signed-effect outputs land and EE3/EE4 are classified.
- Axis T per-PC zero-ablation outputs land for all five PCs.
- PC partition metrics land.
- EE1-EE4 are classified.
- `PHASE6_V38_RESULTS.md` is written with a dedicated crossover section.

## 11. Cross-References

- **Crossover framework:** [`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md)
- **Phase 6 v3.1:** entangled 5D PCA subspace and PC1/PCs-2-5
  decomposition result.
- **Phase 6 v3.4:** direction-specific top-32 substrates on the cliff
  pair.
- **Phase 6 v3.5:** cross-policy substrate identity map.
- **Phase 6 v3.6:** cliff C->P mask functional transfer.
- **Phase 6 v3.7:** pair-specific own-mask functional closeout.
- **Geometry Phase 10/11:** `../calibration/RICH_DISPLAY_OVERLAY_NOTES.md`
  and `../SUNDOG_V_GEOMETRY.md`.

## 12. What v3.9+ Inherits

If EE1 confirms (PCs partially partition):

- v3.9 candidate: held-out pair per-PC decomposition for J1/J2 or a
  pair-specific PCA basis, to see whether P->C pair specificity lives at
  PC-level partition or only at neuron identity.

If EE1 falsifies (PCs share common substrate):

- v3.9 candidate: integrated gradients or causal scrubbing. Per-PC
  partition is the wrong anatomical axis; the five directions use a
  common substrate.

If EE3/EE4 confirm:

- The directional-asymmetry story gains a signed-neuron table suitable
  for the geometry "do not promote" analogy.

If EE3/EE4 falsify:

- Directional asymmetry remains set-level and subspace-level, not
  per-neuron signed. Keep geometry transfer at the level of route
  residuals, not primitive sign classes.

## 13. Versioning

- **v3.8 (2026-05-13)** — initial pin. Option A chosen as the spine:
  per-PC neuron decomposition (Axis T). Option B included as a free
  offline rider: signed direction-of-effect analysis (Axis U). Crossover
  section made mandatory so the mesa result feeds Geometry Phase 10/11
  structurally rather than incidentally.
