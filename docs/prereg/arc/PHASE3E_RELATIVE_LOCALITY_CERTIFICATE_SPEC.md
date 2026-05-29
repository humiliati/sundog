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

---

## Amendment A — Freeze Marker (2026-05-29 PT)

Append-only. Records runner tooling, leak-check coverage, frozen knobs, smoke
fingerprint, the ten-minute-rule decision, and the exact binding command required
by §"Reserved Implementation Names" before execution is admitted.

### Tooling added

- Python runner: `docs/prereg/arc/phase3e_relative_locality_certificate.py`,
  cp-seeded from `phase3e_program_sketch_oracle_v2.py`. It reuses unchanged the
  frozen context identities, `d_context_*` distances, `program_sketch_v2` facets,
  the two-stage no-target barrier, `load_tasks` + `--split-mode {frozen_v2,
  sha256_expansion}`, and the v2 incompatibility hard-pair rules; it replaces the
  v2 absolute-fiber gate/branch section with the rank-based locality metrics,
  controls, nulls, and five-branch adjudication. `runnerSha256` is recorded in
  every receipt manifest.
- Node wrapper: `scripts/arc-phase3e-relative-locality-certificate.mjs`
  (honors `SUNDOG_PYTHON`).
- npm: `arc:phase3e:relative-locality-certificate`.
- Result ignore path: `results/arc/phase3e-relative-locality-certificate/`.
- Leak-check: `npm run arc:phase0:leak-check` passes (0 fail / 0 warn; 22 scanned
  ARC scripts; the new wrapper carries no held-split literal).

### Frozen knobs (mirrored into the runner constants)

- arms: `signature_palette_context` (primary), `signature_only_context`,
  `metadata_only_context`, `raw_grid_context`.
- `K_VALUES=[1,3,5,10]`, `K_PRIMARY=5`; `SEED_SLATE=[20260529,20260530,20260531,
  20260601,20260602]`; `N_PERMUTATIONS=1000`; `N_BOOTSTRAP=2000`.
- `sketch_sim` = mean per-facet Jaccard over the nine facets, dropping
  `none`/`unknown`, omitting a facet when both contexts lack a concrete label.
- branch thresholds: `DELTA_META_MIN=0.10`, `DELTA_RANDOM_MIN=0.15`,
  `P_PRIMARY_MAX=0.01`, `HARD_INCOMPAT_POS_MAX=0.10`, `HARD_INCOMPAT_NEG_MIN=0.25`,
  `DELTA_META_ONLY_MAX=0.05`.
- Oracle caveat: syntactic leakage fatal; vacuity + prior-laundering are gates
  (same definitions as v2 — a laundering violation is a non-vacuous context with
  fewer than two extra facets, fraction taken over non-vacuous primary contexts,
  threshold 0.10); `core_sketch_exact_lookup_fraction` and
  `unique_core_sketch_fraction` are reported as diagnostics, NOT fail gates, per
  §"Oracle Caveat" (the binding 108-task leakage regression is preserved, not
  retuned).

### Smoke fingerprint

- Dry-run: empty receipt with the 16-artifact stub set.
- Capped smoke (`--register` expanded `--split-mode sha256_expansion
  --limit-tasks 36 --permutations 20 --bootstrap 200`): `U_primary=84`, ~57 s,
  all 16 artifacts emit, branch `phase3e_relative_locality_inconclusive`,
  `oracle_invalid=False` (syntactic clean, vacuity 0.0, laundering 0.0). Re-run
  is byte-identical on effects/branch/caveat → deterministic. **This is a 36-task
  subset with 20 permutations, not the binding result.**

### Ten-minute rule

The binding run uses the full 108-task expanded register (`U_primary ~336`) with
1000 permutations × 5 seeds. Neighbor graphs are `O(U_primary² × 4 arms)` with
bipartite conditioning-pair matching; extrapolating from the 84-context / 20-perm
smoke (~57 s) gives an estimated **~20-25 min** full wall, exceeding the repo
ten-minute rule. The binding run is therefore executed as a background job
(deterministic; pinned to this freeze-marker commit), per §"Controls And Nulls".

### Exact binding command

```powershell
node scripts/arc-phase3e-relative-locality-certificate.mjs `
  --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv `
  --split-mode sha256_expansion `
  --out results/arc/phase3e-relative-locality-certificate
```

(`SUNDOG_PYTHON` = the Python 3.12 interpreter; the certificate is CPU-only.)

---

## Amendment B — Binding Verdict (2026-05-29 PT)

Append-only. The relative-locality certificate was run on the expanded 108-task
register, pinned to freeze-marker commit `f7850f0` (`runnerSha256 3DD2C97A…`,
`splitMode=sha256_expansion`, 5-seed slate × 1000 permutations, `U_primary=336`,
`U_all=491`, ~56m35s background wall — heavier than the ~20-25 min freeze-marker
estimate; the bipartite conditioning-pair matching over 336²×4 arms plus the null
slate dominated). `gitDirty=true` reflects only concurrent parallel-lane tracked
edits, not any ARC-lane change; the binding inputs are pinned by `gitCommit`,
`runnerSha256`, and the matching `registerHash`. Receipt:
`results/arc/phase3e-relative-locality-certificate/`.

**Branch: `phase3e_relative_locality_negative`.**

### k=5 neighbor program-sketch coherence

| arm | mean sketch_sim @k5 | hard-incompat @k5 |
| --- | --- | --- |
| `signature_palette_context` | 0.4197 | **0.2845** |
| `signature_only_context` | 0.4060 | 0.2958 |
| `metadata_only_context` | 0.4139 | 0.2262 |
| `raw_grid_context` | **0.4561** | 0.2446 |

### Effects (k=5 unless noted)

- `delta_palette_vs_metadata` = **+0.0058** (k3 +0.0058, k10 −0.0022 → sign flips)
- `delta_palette_vs_uniform_random` = **+0.0720** (threshold for positive ≥ 0.15)
- `delta_palette_vs_signature_only` = +0.0138
- `delta_palette_vs_raw_grid` = **−0.0364** (palette is *below* raw_grid)
- `p_palette_vs_prior_stratified_random` = **0.0002** (palette is significantly
  above prior-stratified random)
- `palette_hard_incompatibility_rate` = **0.2845** ( > 0.25 negative trigger)
- `stable_sign_k3_k10` = **False**

### Why negative (and the honest nuance)

The negative branch fires on `palette_hard_incompatibility_rate 0.2845 > 0.25`:
more than a quarter of `signature_palette_context`'s rank-nearest neighbor pairs
are behaviorally **hard-incompatible** under the v2 rules. So even the
nearest-in-rank registered contexts are not behaviorally coherent.

The nuance, recorded honestly: palette's neighbor sketch similarity **is**
statistically above the prior-stratified random control (`p = 0.0002`), so a weak
rank-local signal exists. But it is **not usable**: it is essentially tied with
the coarse `metadata_only` control (Δ +0.006, and the sign flips negative by
k=10), it is **below `raw_grid`** (Δ −0.036 — the raw-grid arm is both more
sketch-coherent and less incompatible), it falls far short of the random-control
margin (Δ +0.072 < 0.15), and it is dominated by the 28% hard-incompatibility
rate. A statistically detectable but small, metadata-explained,
raw-grid-dominated, incompatibility-laden signal is exactly the
"no usable rank-local sketch geometry" the negative branch names.

### Oracle caveat (diagnostic, not a gate here)

`oracle_invalid = False`: syntactic leakage clean, vacuity 0.00, prior-laundering
0.00 on the expanded primary universe. The binding 108-task leakage regression is
carried as a diagnostic and **not retuned away**:
`core_sketch_exact_lookup_fraction = 0.351`, `unique_core_sketch_fraction =
0.551` (identical to the expanded absolute certificate). Per §"Oracle Caveat"
these are caveats, not fail gates, because this certificate never used target
exact hashes to train, select, or score — the locality metric is pure facet
Jaccard.

### What this establishes / does not

**Establishes**: the relative (rank-based) test **strengthens** the
absolute-sparsity result. On the expanded register `signature_palette_context`
has neither fixed-radius fibers (absolute: 0 near pairs, min distance 0.196) nor
**usable** rank-local program-sketch coherence (relative: weak signal, below
metadata-tie and raw_grid, 28% hard-incompatible). Notably `raw_grid_context`
edges `signature_palette_context` on both rank-local metrics — an anti-privilege
signal for the signature representation.

**Does not**: change any absolute-epsilon finding (they remain binding); prove or
disprove Blackwell sufficiency; license or block a Branch E solver; or claim
`raw_grid` is "sufficient" (it merely has marginally better rank-local geometry
on this metric). No threshold, k, permutation count, control, or sketch-similarity
definition was retuned after seeing outputs.

### Path forward

The Phase 3E locality-certificate program has now returned, in sequence:
`deferred_label_vacuity` → `deferred_sparse_fibers` (36-task) →
`expanded_oracle_regression` (108-task absolute) → **`relative_locality_negative`
(108-task rank-based)**. Across absolute and relative geometry, on 36 and 108
registered tasks, `signature_palette_context` shows no certifiable usable fiber or
rank-local locality. The remaining admissible direction is **Branch E on
capability grounds** — a solver justified by what it can do, not by certified
fiber geometry — under its own pre-registered spec, arena gate, and verdict
discipline; or to record the certificate program's negative as the durable
finding. The absolute-epsilon and oracle-regression receipts remain binding.
