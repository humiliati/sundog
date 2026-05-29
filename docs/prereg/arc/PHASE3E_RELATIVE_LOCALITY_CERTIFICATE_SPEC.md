# Phase 3E -- Relative Locality Certificate

Parent specs:

- [`PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md`](PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md)
- [`PHASE3E_PROGRAM_SKETCH_ORACLE_V2_SPEC.md`](PHASE3E_PROGRAM_SKETCH_ORACLE_V2_SPEC.md)
- [`PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md`](PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md)

Filed: **2026-05-29 (PT)**

Status: **SPEC FILED; EXECUTION HOLD**. This file freezes a rank-based
certificate lane for the expanded 108-task ARC register. It admits no execution
until runner tooling, npm wiring, result ignore path, leak-check coverage, a
smoke fingerprint, and a freeze-marker amendment are committed together.

## Purpose

The absolute-radius Phase 3E certificates found no registered
`signature_palette_context` collision and no near pairs:

- 36-task v2: `phase3e_v2_deferred_sparse_fibers`, minimum cross-task distance
  `0.207`, 0 near pairs;
- 108-task expansion: `phase3e_v2_expanded_oracle_regression`, minimum
  cross-task distance `0.196`, 0 near pairs.

The 108-task expansion also showed that the v2 oracle's anti-solver-leakage
gate is not register-size robust (`core_sketch_exact_lookup_fraction = 0.351`
against the frozen 0.20 threshold), even though vacuity and prior-laundering
still pass.

This spec asks a different geometry question: when absolute
`epsilon_primary = 0.05` fibers are empty, does the rank order induced by
`signature_palette_context` still place behaviorally similar registered contexts
nearer than controls?

This is a locality certificate, not a solver and not a sufficiency proof.

## Core Question

Does `signature_palette_context` induce statistically stronger cross-task
program-sketch locality among nearest neighbors than:

1. `signature_only_context`;
2. `metadata_only_context`;
3. `raw_grid_context`;
4. random cross-task neighbors;
5. prior-stratified random cross-task neighbors;
6. task-label or sketch-label permutations?

If yes, the representation has usable rank-local geometry despite empty
absolute-epsilon fibers. If no, the absolute-sparsity result is strengthened:
there is neither fixed-radius locality nor rank-local sketch coherence in the
registered expanded universe.

## Frozen Inputs

Primary register:

`docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv`

Diagnostic register:

`docs/prereg/arc/P0_TASK_REGISTER.csv`

Primary split:

`splitMode = sha256_expansion`

Diagnostic split:

`splitMode = frozen_v2`

The primary adjudication universe is inherited from Phase 3E:

```text
U_primary = test_lodo union pttest
```

The full diagnostic universe is:

```text
U_all = validation_lodo union validation_pttest union test_lodo union pttest
```

The 108-task expanded register is primary because it is the first register where
the absolute-epsilon sparsity claim was stress-tested at scale.

## Frozen Representations And Labels

The following are inherited unchanged:

- `signature_palette_context`, primary representation;
- `signature_only_context`, strict quotient diagnostic;
- `metadata_only_context`, nuisance control;
- `raw_grid_context`, raw-grid diagnostic;
- `d_context_*` distance functions;
- `program_sketch_v2` facet vocabulary;
- the two-stage no-target context fingerprint barrier.

The fixed absolute thresholds remain recorded but are not used for primary
branching:

```text
epsilon_primary = 0.05
epsilon_strict = 0.025
epsilon_loose = 0.10
```

No threshold retuning is admitted. The relative test is explicitly rank-based.

## Oracle Caveat

The expanded absolute certificate already established that
`program_sketch_v2` fails the original anti-solver-leakage gate at 108 tasks
(`core_sketch_exact_lookup_fraction = 0.351 > 0.20`). This spec does not retune
that gate and does not erase that verdict.

For this relative-locality lane:

- syntactic leakage remains fatal;
- anti-vacuity and anti-prior-laundering must still pass;
- `core_sketch_exact_lookup_fraction` and `unique_core_sketch_fraction` are
  reported as diagnostics and caveats, not as primary fail gates, because this
  certificate never uses exact output hashes to train, select, or solve;
- any positive branch must be phrased as **rank-local sketch coherence under a
  leakage-caveated oracle**, not as a solver license or sufficiency proof.

If syntactic leakage is detected, or if vacuity/prior-laundering fails on the
primary expanded universe, the branch is `phase3e_relative_oracle_invalid`.

## Neighbor Sets

For every context `c in U_primary`, compute cross-task nearest neighbors under
each representation arm.

Same-task neighbors are excluded from primary metrics. They are reported only as
diagnostics.

Rank cuts:

```text
k in {1, 3, 5, 10}
```

The primary rank cut is:

```text
k_primary = 5
```

Rationale: k=1 is too noisy, k=10 may average over the sparse tail, and k=5 is
large enough to estimate local sketch coherence while remaining local in rank
space.

## Sketch Similarity

For two contexts `a` and `b`, compute per-facet Jaccard similarity after
dropping `none` and `unknown`.

Let:

```text
facet_sim_f(a,b) = |labels_f(a) intersect labels_f(b)| /
                   |labels_f(a) union labels_f(b)|
```

If both contexts have no concrete labels for a facet, omit that facet from the
mean for that pair rather than scoring it as similar.

The pairwise sketch similarity is:

```text
sketch_sim(a,b) = mean_f facet_sim_f(a,b)
```

The pairwise sketch distance is:

```text
sketch_dist(a,b) = 1 - sketch_sim(a,b)
```

Diagnostics must also report:

- same-primary-prior rate;
- same-branch-family rate, where the branch family is the sorted set of
  non-`none`, non-`unknown` `rule_scope` labels;
- hard-incompatibility rate using the v2 incompatibility hard-pair rules;
- per-prior sketch similarity.

No target exact hashes, raw outputs, exact output coordinates, or per-cell masks
may enter the primary locality metric.

## Controls And Nulls

The primary arm is `signature_palette_context`.

Required controls:

1. `signature_only_context` nearest neighbors;
2. `metadata_only_context` nearest neighbors;
3. `raw_grid_context` nearest neighbors;
4. uniform random cross-task neighbors matched to each context and k;
5. prior-stratified random cross-task neighbors, drawing neighbors from the same
   primary-prior distribution observed in the `signature_palette_context` kNN
   set;
6. task-label permutation null: shuffle sketch records across task IDs within
   each primary prior, then recompute `sketch_sim` for the frozen neighbor graph;
7. global sketch permutation null: shuffle sketch records across all
   `U_primary` contexts, then recompute `sketch_sim`.

Random and permutation controls use:

```text
seed_slate = {20260529, 20260530, 20260531, 20260601, 20260602}
n_permutations = 1000
```

If runtime exceeds the repo's ten-minute rule, the first receipt must use a
capped smoke and stage the full command for the operator.

## Metrics

For each arm and k:

- `mean_neighbor_sketch_sim`;
- `median_neighbor_sketch_sim`;
- `mean_neighbor_sketch_dist`;
- `hard_incompatibility_rate`;
- `same_primary_prior_rate`;
- `same_rule_scope_family_rate`;
- per-prior mean sketch similarity;
- bootstrap 95% confidence interval over contexts.

Primary effect:

```text
delta_palette_vs_metadata =
  mean_neighbor_sketch_sim(signature_palette_context, k=5)
  - mean_neighbor_sketch_sim(metadata_only_context, k=5)
```

Primary null p-value:

```text
p_palette_vs_prior_stratified_random =
  permutation p-value for the k=5 signature_palette_context mean sketch_sim
  against the prior-stratified random control.
```

Secondary effects:

- `delta_palette_vs_signature_only`;
- `delta_palette_vs_raw_grid`;
- `delta_palette_vs_uniform_random`;
- `palette_hard_incompatibility_rate`.

## Branch Criteria

Branch precedence is table order.

| branch | condition | interpretation |
| --- | --- | --- |
| `phase3e_relative_oracle_invalid` | Syntactic leakage occurs, anti-vacuity fails, or anti-prior-laundering fails on the expanded primary universe. | The label oracle is not usable even for relative locality. |
| `phase3e_relative_locality_positive` | At k=5, `delta_palette_vs_metadata >= 0.10`, `delta_palette_vs_uniform_random >= 0.15`, `p_palette_vs_prior_stratified_random <= 0.01`, `palette_hard_incompatibility_rate <= 0.10`, and the same sign holds for k=3 and k=10. | `signature_palette_context` has rank-local sketch coherence beyond controls. This licenses only a future relative-selector spec, not sufficiency. |
| `phase3e_relative_locality_metadata_only` | `signature_palette_context` is positive against random controls, but `delta_palette_vs_metadata < 0.05` or metadata is statistically indistinguishable from palette. | Any locality is explained by coarse metadata, not the full signature. |
| `phase3e_relative_locality_negative` | `signature_palette_context` is not significantly above prior-stratified random at k=5, or has `palette_hard_incompatibility_rate > 0.25`. | No usable rank-local sketch geometry under this representation. |
| `phase3e_relative_locality_inconclusive` | None of the above, or positive at k=5 but not directionally stable across k=3 and k=10. | The rank-local signal is too weak or unstable to interpret. |

The absolute-epsilon receipts remain binding regardless of this branch. A
relative-locality positive does not retroactively populate fixed-radius fibers.

## Required Artifacts

Binding output path:

`results/arc/phase3e-relative-locality-certificate/`

Required files:

- `manifest.json`;
- `split.csv`;
- `context_fingerprints_no_targets.jsonl`;
- `context_fingerprints_no_targets.sha256`;
- `program_sketch_v2.jsonl`;
- `oracle_caveat_audit.csv`;
- `neighbor_graphs.csv`;
- `neighbor_similarity_by_context.csv`;
- `relative_locality_summary.csv`;
- `per_prior_relative_locality.csv`;
- `permutation_nulls.csv`;
- `bootstrap_intervals.csv`;
- `phase3e_relative_locality_certificate_receipt.json`;
- `branch_adjudication.md`;
- `commands.md`;
- `hashes.json`.

The manifest must record:

- this spec hash;
- parent spec hashes;
- expanded register hash;
- split mode;
- target-output barrier hash;
- all k values;
- all random seeds;
- permutation count;
- branch;
- source code hashes.

## Reserved Implementation Names

These names are reserved but not executable by this spec alone:

- Python runner:
  `docs/prereg/arc/phase3e_relative_locality_certificate.py`;
- Node wrapper:
  `scripts/arc-phase3e-relative-locality-certificate.mjs`;
- npm script:
  `arc:phase3e:relative-locality-certificate`;
- receipt path:
  `results/arc/phase3e-relative-locality-certificate/`.

The freeze-marker amendment must add runner tooling, wrapper, npm wiring,
result ignore path, leak-check coverage, a smoke fingerprint, and exact command
receipts before execution is admitted.

## Public Language

Allowed before a binding receipt:

> "Phase 3E has filed a relative-locality certificate spec. It keeps the
> absolute-epsilon results intact and asks whether rank-nearest neighbors under
> `signature_palette_context` are more program-sketch-coherent than controls. No
> receipt exists yet."

Allowed if `phase3e_relative_locality_positive` is filed:

> "The relative-locality certificate found rank-local program-sketch coherence
> for `signature_palette_context` beyond controls. This does not prove
> sufficiency and does not populate the fixed-radius fibers."

Allowed if `phase3e_relative_locality_metadata_only` is filed:

> "The relative-locality certificate found that any rank-local coherence is
> explained by coarse metadata rather than the full signature representation."

Allowed if `phase3e_relative_locality_negative` is filed:

> "The relative-locality certificate found no usable rank-local sketch geometry
> under `signature_palette_context` on the expanded register."

Forbidden:

- claiming a relative-locality positive proves Blackwell sufficiency;
- claiming a relative-locality positive licenses a Branch E solver without a
  new Branch E solver spec;
- claiming a relative-locality result changes the absolute-epsilon findings;
- retuning k, permutation count, control definitions, branch thresholds, or
  sketch similarity after seeing outputs;
- using target exact hashes or raw outputs in the locality metric.
