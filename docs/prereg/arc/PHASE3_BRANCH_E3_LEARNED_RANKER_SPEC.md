# Branch E3 -- Learned Ranker Over Frozen Program Candidates

Parent / boundary specs:

- [`PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md`](PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md)
- [`PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md`](PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md)
- [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)
- [`PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md`](PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md)
- [`PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md`](PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md)

Drafted: **2026-05-29 (PT)**

Status: **DRAFT ONLY; EXECUTION NOT ADMITTED; TOOLING NOT FROZEN**.

This file is a proposed different-solver-class spec. It records the next
admissible Branch E direction after v2, but does not itself admit a run. A later
freeze-marker amendment must add runner, wrapper, npm wiring, leak-check receipt,
smoke fingerprint, timing estimate, and exact binding command before execution.

## Boundary Citation

Branch E v1 returned `branch_e_capability_demonstrated`: deterministic program
search over a frozen primitive library solved 2 distinct held-out tasks on both
`test_lodo` and `pttest`.

Branch E v2 returned `branch_e_v2_capability_replicated`: admitting the deferred
deterministic mask families, morphology cross-product, and depth-3 composition
retained the same two gated solves (`be94b721`, `f25fbde4`) but solved 0 new
gated tasks. It also dropped validation from 2 to 1 solved task per lane because
the larger train-consistent program set can crowd the correct candidate out of
the frozen top-2 family/simplicity ranking.

Therefore E3 does **not** add primitives. It targets the observed selector
bottleneck directly: can a learned ranker over the **same frozen Branch E v2
candidate generator** choose better top-2 attempts than the deterministic
family/simplicity ranker?

Unchanged caveats: this is held-out public-training capability work only. It is
not a Blackwell-sufficiency proof, ARC solve, public-evaluation result, or
Kaggle claim. Phase 6 remains the only public-evaluation gate.

## Core Question

Given the frozen Branch E v2 candidate generator, can a ranker trained only on
non-gated public-training candidate examples improve held-out top-2 selection on
the expanded 108-task registered universe, especially by repairing the v2
top-2-crowding failure?

## Frozen Inputs

```text
primary_register = docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv
primary_split    = sha256_expansion
U_primary        = test_lodo union pttest
U_validation     = validation_lodo union validation_pttest
U_all            = U_validation union U_primary
```

Training data for the ranker:

```text
ranker_aux_pool = ARC-AGI-2 public-training tasks
                  minus P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv task ids
                  minus any evaluation-blind or public-evaluation ids
```

The ranker may use public-training targets from `ranker_aux_pool` to label
candidate programs after the no-target candidate fingerprint barrier. It may use
expanded-register validation lanes only for early stopping, seed/model
selection, and threshold-free diagnostics. It may not use `U_primary` targets for
training, early stopping, feature selection, hyperparameter tuning, candidate
generation, or ranker selection.

## Frozen Candidate Generator

Candidate generation is inherited byte-for-byte from Branch E v2:

```text
generator = docs/prereg/arc/phase3_branch_e_v2_program_search.py
max_composition_depth = 3
candidate_budget_per_instance = 20000
attempts = 2
```

Allowed candidate families are exactly those frozen in Branch E v2:

- v1 structural transforms;
- v1 candidate-combinator candidates;
- Branch D deterministic color-rule bank;
- v2 deterministic mask families;
- v2 deterministic morphology cross-product;
- depth-3 structural composition.

Forbidden in E3:

- adding, removing, or reparameterizing primitives;
- increasing candidate budget after seeing receipts;
- using `signature_palette` geometry or certificate-neighborhood distances;
- using public-evaluation grids;
- using held-out `U_primary` targets before final scoring;
- any learned mask, learned primitive, or learned program proposer.

E3 changes only the selector from the deterministic v2 family/simplicity ranking
to a learned candidate ranker. The v2 deterministic selector remains the primary
control.

## No-Target Barrier

For every split (`ranker_aux_pool`, `U_validation`, `U_primary`):

1. Build all train-consistent candidates from conditioning pairs only.
2. Write `candidate_fingerprints_no_targets.jsonl` containing instance identity,
   conditioning identity, candidate program id, candidate output hash, candidate
   feature vector hash, v2 deterministic rank, and candidate count.
3. Hash the barrier file.
4. Only after the barrier hash is written, read query targets to label candidates
   for ranker training/validation/scoring.

The binding manifest must record the no-target barrier hashes separately for
auxiliary training, validation, and `U_primary`.

## Ranker Feature Schema

Feature schema: `arc-p3-branch-e3-ranker-feature-v1`.

All features are deterministic and target-free:

1. Program syntax features:
   `depth`, operation count, structural prefix length, terminal family, mask
   family, morphology op, color-rule family, candidate-combinator flag, and the
   frozen v2 deterministic rank.
2. Program-id hashed features:
   feature hashing over operation names, operation bigrams, mask family,
   morphology, color family, and normalized parameter tokens into
   `program_hash_dim = 512`.
3. Candidate-grid features:
   output height/width, area, palette size, nonzero count, density, component
   count, bounding-box area, symmetry flags, row/column periodicity flags, and
   shape/palette relation to the query input.
4. Conditioning-summary features:
   number of conditioning pairs, input/output shape-ratio summaries,
   palette-change summaries, edit-density summaries, and per-task primary prior
   one-hot as metadata.
5. Candidate-vs-input relation:
   normalized raw Hamming distance when shapes match, shape relation token,
   palette overlap, nonzero overlap after nearest-neighbor projection, and
   candidate output complexity deltas.

Forbidden features:

- target-output features for the scored instance;
- exact target hashes, output serializations, or target-derived residuals;
- `signature_palette` distances, fiber-neighborhood indices, or certificate
  oracle labels;
- task id as a feature;
- ranker features hand-added after inspecting `U_primary` labels.

## Ranker Model

Model family: pointwise binary classifier used as a ranker.

```text
input = ranker_feature_v1 vector
model = MLP( input_dim -> 192 -> 96 -> 1 )
activation = ReLU
dropout = 0.05
loss = weighted binary cross entropy
positive label = candidate grid exactly equals query target
negative label = train-consistent candidate grid does not equal query target
optimizer = AdamW
lr = 1.0e-3
betas = (0.9, 0.99)
eps = 1.0e-8
weight_decay = 1.0e-4
batch_size = 2048
max_epochs = 40
early_stop_patience = 6
grad_clip_norm = 1.0
seed_slate = 20260529, 20260530, 20260531, 20260601, 20260602
```

Class weights are computed from the ranker auxiliary training candidate labels
only and clipped to `[1.0, 100.0]`. Validation chooses seed/epoch by:

1. higher validation distinct-task `grid_exact_any` across `U_validation`;
2. higher validation exact-instance rate;
3. lower validation binary cross-entropy;
4. lower seed.

The final selected ranker scores every candidate in `U_primary`. Top-2 attempts
are the top-2 distinct candidate grids by:

1. higher learned score;
2. lower frozen v2 deterministic rank;
3. lower program description length;
4. lexicographic program id.

## Controls

The binding run must score the same candidate bank under:

1. `v2_deterministic_selector`: frozen Branch E v2 family/simplicity ranking.
2. `learned_ranker`: the primary E3 learned selector.
3. `metadata_only_ranker`: same architecture, but limited to conditioning-summary
   features and candidate-grid scalar features; no program-id hash and no program
   family tokens.
4. `label_shuffle_ranker`: same architecture and features as primary, but with
   auxiliary training labels deterministically shuffled by seed. Diagnostic only.
5. `oracle_candidate_ceiling`: non-adjudicating target-read diagnostic: whether
   any admitted candidate equals the target.

The branch decision uses only `learned_ranker` on `U_primary`, compared against
the pre-registered v1/v2 solved set. Controls are reported to characterize
whether any lift is ranker-specific or merely metadata/base-rate leakage.

## Branches

Frozen v1/v2 gated solved set:

```text
S_v1v2 = {be94b721, f25fbde4}
```

Branch precedence is table order.

| branch | condition on `learned_ranker` over `U_primary` | interpretation |
| --- | --- | --- |
| `branch_e3_ranker_material_lift` | `test_lodo` solves at least 4 distinct tasks, `pttest` solves at least 4 distinct tasks, and each lane solves at least 2 tasks outside `S_v1v2`. | Learned ranking materially lifts the bounded deterministic-search baseline. |
| `branch_e3_ranker_selector_lift_below_material` | At least one gated lane solves at least 1 task outside `S_v1v2`, but the material-lift condition is not met. | Learned ranking repairs at least one top-2 crowding case, but not enough for material lift. |
| `branch_e3_ranker_replicated` | At least 2 distinct tasks solved on both `test_lodo` and `pttest`, and zero new tasks outside `S_v1v2` on both lanes. | Learned ranking reproduces v1/v2 capability but does not improve it. |
| `branch_e3_ranker_regression` | The ranker solves fewer than 2 distinct tasks on either gated lane while the v2 deterministic selector retains 2 on both lanes. | Learned ranking hurts top-2 selection. |
| `branch_e3_ranker_partial` | At least one exact `U_primary` instance is solved, but none of the above branches applies. | Non-zero capability below the replicated baseline. |
| `branch_e3_ranker_floor` | Zero exact `U_primary` instances are solved. | Learned ranking floors. |

No branch gate may be retuned after reading `U_primary` labels.

## Quarantine Taxonomy

Per instance, report exactly one primary quarantine label:

- `candidate_coverage_failure`: no admitted candidate equals target.
- `v2_crowding_repaired`: correct candidate exists, v2 top-2 misses, learned
  ranker top-2 hits.
- `learned_ranker_miss`: correct candidate exists, learned ranker top-2 misses.
- `v2_ranker_already_solved`: v2 and learned ranker both solve.
- `learned_ranker_regression`: v2 solves, learned ranker misses.
- `metadata_only_matches`: learned ranker hit also hit by metadata-only ranker.
- `budget_exhausted_candidate_unknown`: candidate budget exhausted and no correct
  candidate observed.
- `no_admitted_programs`: no train-consistent candidate was admitted.

Secondary diagnostics may include candidate count, correct-candidate rank under
v2, correct-candidate rank under learned ranker, score margin, and whether the
winning program uses a v2-only family or depth-3 composition.

## Required Artifacts

Binding output path:

```text
results/arc/phase3-branch-e3-learned-ranker/
```

Required files:

- `manifest.json`
- `split.csv`
- `aux_task_list.csv`
- `candidate_fingerprints_no_targets_aux.jsonl`
- `candidate_fingerprints_no_targets_validation.jsonl`
- `candidate_fingerprints_no_targets_u_primary.jsonl`
- corresponding `.sha256` files for all three barrier files
- `ranker_feature_schema.json`
- `training_summary.csv`
- `learning_curves.csv`
- `seed_selection.csv`
- `scores_by_instance.jsonl`
- `solutions_by_instance.csv`
- `capability_summary.csv`
- `selector_comparison.csv`
- `candidate_ceiling.csv`
- `per_prior_capability.csv`
- `quarantine_by_instance.csv`
- `family_usage.csv`
- `phase3_branch_e3_learned_ranker_receipt.json`
- `branch_adjudication.md`
- `commands.md`
- `hashes.json`

The manifest must record parent spec hashes, parent runner hashes, register hash,
auxiliary pool hash, feature schema hash, candidate barrier hashes, model
hyperparameters, seed slate, selected seed/epoch, branch, source-code hashes, and
whether any candidate budget was exhausted.

## Reserved Implementation Names

- Python runner: `docs/prereg/arc/phase3_branch_e3_learned_ranker.py`
- Node wrapper: `scripts/arc-phase3-branch-e3-learned-ranker.mjs`
- npm script: `arc:phase3:branch-e3-learned-ranker`
- receipt path: `results/arc/phase3-branch-e3-learned-ranker/`

## Ten-Minute Rule

The freeze-marker amendment must include at least:

1. `py_compile` receipt;
2. dry-run artifact receipt;
3. capped candidate-generation smoke on the known v1/v2 solved tasks;
4. capped ranker-training smoke on a small auxiliary subset;
5. measured seconds per candidate instance and estimated full binding wall time.

If estimated wall time exceeds the repository ten-minute rule, the binding run
must be staged for the operator with exact PowerShell command, expected wall
time, resume-safety notes, and read-back paths.

## Public Language

Allowed before a binding receipt:

> "Branch E3 has a draft learned-ranker spec. It keeps the Branch E v2 candidate
> generator frozen and tests whether a supervised selector can beat the top-2
> crowding observed in v2. No tooling freeze marker or binding receipt exists
> yet."

Allowed if `branch_e3_ranker_material_lift`:

> "A learned ranker over the frozen Branch E v2 candidate set materially lifted
> the held-out public-training exact-match rate. This remains a capability
> result, not a Blackwell-sufficiency proof, ARC solve, public-evaluation result,
> or Kaggle claim."

Allowed if `branch_e3_ranker_selector_lift_below_material`:

> "The learned ranker repaired at least one top-2 crowding case but did not clear
> the pre-registered material-lift gate."

Allowed if `branch_e3_ranker_replicated`:

> "The learned ranker replicated the bounded v1/v2 program-search capability but
> did not improve it under the frozen gate."

Forbidden:

- claiming E3 proves Blackwell sufficiency;
- claiming any public-evaluation or Kaggle result;
- adding primitives under the learned-ranker spec;
- using `signature_palette` geometry as a selector;
- tuning ranker features, branch gates, model size, seed slate, or candidate
  budget after reading `U_primary` labels.

---

## Amendment A — Tooling Freeze Marker (BUILT; binding run PAUSED, 2026-05-29 PT)

Append-only. This admits + tracks the E3 tooling (per the §"Ten-Minute Rule"
freeze-marker requirement) and records that the binding run is **paused by
operator decision** — the measured cost is too high to launch now. No binding
receipt exists; no branch is adjudicated.

### Tooling added

- Python runner `docs/prereg/arc/phase3_branch_e3_learned_ranker.py`. It imports
  the frozen Branch E v2 generator and replicates only its enumeration **driver**
  (`enumerate_admitted`), calling v2's frozen family functions
  (`structural_programs`, `color_edit_programs_v2`, `combinator_programs`,
  `_consistent`, `_transform_pairs`, `_compose`) to return the FULL admitted
  candidate list. The selector is the new learned MLP ranker. v2 is unmodified.
- Wrapper `scripts/arc-phase3-branch-e3-learned-ranker.mjs`; npm
  `arc:phase3:branch-e3-learned-ranker[:shard|:merge]`; `.gitignore`
  `results/arc/phase3-branch-e3-learned-ranker/`.
- Two modes: `--shard-aux --shard-index i --shard-count N` (deterministic chunk of
  the sorted aux pool → candidate fingerprints + labels) and `--merge` (canonical
  shard merge → train + validate + score `U_primary` under five controls →
  adjudicate). The merge canonically sorts merged aux records by
  `(task_id, instance_id)` so the result is byte-equivalent regardless of shard
  count.
- Frozen knobs mirrored: `INPUT_DIM = 591` target-free features
  (`arc-p3-branch-e3-ranker-feature-v1`, `program_hash_dim = 512`); RankerMLP
  `591→192→96→1` weighted-BCE (`pos_weight` clipped `[1,100]`, batch 2048, 40
  epochs, patience 6); 5-seed slate; controls `v2_deterministic_selector`,
  `learned_ranker`, `metadata_only_ranker`, `label_shuffle_ranker`,
  `oracle_candidate_ceiling`; three split barriers (aux/validation/U_primary).

### Verification (smoke)

- `py_compile` clean; `npm run arc:phase0:leak-check` 0 fail / 0 warn (25 ARC
  scripts; the wrapper carries no held-split literal).
- Dry-run emits the artifact stub set.
- **Byte-faithfulness**: on all nine `be94b721` + `f25fbde4` instances
  (`test_lodo` + `pttest`), `enumerate_admitted` produces the identical
  `n_admitted` as `v2.solve_instance` (706 / 236 / 116…), and the
  `v2_deterministic_selector` control reproduces every v2 gated solve →
  generation is byte-for-byte v2.

### Measured cost + ten-minute-rule decision

A single-shard timing probe (12 aux tasks → 49 held-out instances) took
**38m45s = ~47.5 s/instance** — the intrinsic frozen-v2 enumeration cost over
uncurated auxiliary grids (most instances admit 0 train-consistent candidates yet
still pay the full enumeration). Mean ~60 candidates/instance; 1086 positive
labels from 12 tasks. The full ~892-task auxiliary pool (~3.6k instances)
extrapolates to **~48 h** of candidate generation. Memory is fine: ~220k training
rows, ~0.5 GB.

Per §"Ten-Minute Rule" this exceeds inline execution by orders of magnitude, and
**the operator chose to pause** rather than commit ~48 h now. The lane is
recorded in `docs/TODO.md` (ARC Branch E3, `compute-blocked`, paused) with the
two resume paths: (a) stage + run the full sharded slate, or (b) first amend the
spec to bound the aux pool / aux budget for a ~3-4× faster turnaround. No binding
run, no verdict, no public-language claim beyond the pre-receipt statement.

---

## Amendment B — Bounded-Aux Two-Stage Screening (PROPOSED; execution gated, 2026-06-30 PT)

Append-only. This admits **resume-path-(b)**: a faster screening of E3 by bounding the *auxiliary
training pool only*. It **admits no binding run** — a later freeze-marker sub-amendment must add the
smokes + exact commands below before execution. It changes **no** existing E3 gate, control, feature,
model, seed, or budget, and **no** prior verdict. Motivation: the ~48 h cost is almost entirely aux
candidate generation (~892 tasks / ~3.6 k instances at ~47.5 s); U_primary (336) + validation are a
small fixed floor. The only *faithful* speed knob is the aux pool size — the v2 generator,
U_validation, and the **full** U_primary scoring + gate stay byte-identical.

### The deterministic bound (reuses the existing shard mechanism — no generator change)
The Amendment-A runner already shards the **sorted** aux pool: `--shard-aux --shard-index i
--shard-count N`. Task ids are ARC sha256 hashes, so the sort order is uniform and a contiguous
shard is a uniform subsample. The bound is therefore declared as *which shards of `N=8` are
generated*:

```text
shard_count        = 8                     # frozen; the sorted-hash-id pool cut into 8 uniform chunks
stage_1 (1/8 aux)  = shards {0}
stage_2 (1/4 aux)  = shards {0, 1}         # reuses shard 0 from stage 1
full    (resume-a) = shards {0..7}         # reuses shards 0,1 from stages 1-2
```

No shard set is chosen after reading any `U_primary` label; `shard_count=8` and the stage ladder are
fixed by this amendment. The `--merge` step canonically sorts the present aux records, so a
subset-of-shards merge is a well-defined deterministic 1/8 or 1/4 aux pool.

### Frozen (everything except the aux pool size)
v2 generator (byte-faithful `enumerate_admitted`), `arc-p3-branch-e3-ranker-feature-v1` (591-dim),
the `591→192→96→1` weighted-BCE model + all hyperparameters, the 5-seed slate, U_validation, the
**full** U_primary candidate generation + scoring, all three no-target barriers, and the five controls
(`v2_deterministic_selector`, `learned_ranker`, `metadata_only_ranker`, `label_shuffle_ranker`,
`oracle_candidate_ceiling`). The §"Branches" table (`material_lift` / `selector_lift_below_material` /
`replicated` / `regression` / `partial` / `floor`) is **unchanged**; only its *interpretation* gets the
screening rider below.

### The asymmetric screening verdict (why a bounded run is sound, not just weaker)
Subsampling the aux pool can only **reduce** the ranker's training data, so it can only **hurt** the
learned ranker — never help it. Therefore, at each stage, adjudicate the normal branch on the **full**
U_primary and apply:

- **CONCLUSIVE (stop):** branch ∈ {`branch_e3_ranker_material_lift`, `branch_e3_ranker_selector_lift_below_material`}.
  A data-starved ranker that still lifts is a **lower bound** on the full ranker (the full aux run can
  only do ≥ as well), so the capability is established without spending more compute.
- **SCREEN ONLY (escalate):** branch ∈ {`branch_e3_ranker_replicated`, `branch_e3_ranker_regression`,
  `branch_e3_ranker_partial`, `branch_e3_ranker_floor`}. This is **inconclusive at the bounded
  aux_fraction** — it may be data-starvation, not genuine selection-hardness. Escalate 1/8 → 1/4 →
  full, **reusing** already-generated shards. **No `floor`/`regression` claim is binding until the FULL
  aux run (shards {0..7}) yields a non-positive.**

### Data-starvation guard (mandatory diagnostics per stage)
Each stage must report: positive-label count (vs the Amendment-A full-pool estimate ≈ 1086 positives /
12 aux tasks ⇒ ~80 k full), the `learning_curves.csv`, the seed-selection table, and the validation
distinct-task solves. If a stage's branch is non-positive **and** its positive-label count is far below
the full-pool estimate, escalation is **mandatory** (not optional) — a starved screen cannot ground a
floor.

### Freeze-marker requirements before any binding run (per §"Ten-Minute Rule")
A later sub-amendment must record: (1) `py_compile` (done in Amendment A); (2) leak-check
(`arc:phase0:leak-check`); (3) a smoke proving `--merge` over a **single** aux shard is byte-faithful
(the `v2_deterministic_selector` control must still reproduce the v1/v2 gated solves — the
Amendment-A byte-faithfulness check, restricted to one shard); (4) confirmation that U_primary +
validation candidate generation is **cached/computed once** and *not* regenerated per escalation stage
(if the runner regenerates it each `--merge`, add a `--score-only`/cached-barrier path so escalation
pays only the new aux shard); (5) measured shard-0 wall time → per-stage estimates; (6) the exact
staged commands. Expected per-stage wall (subject to (5)): stage-1 ≈ shard-0 aux (~6 h) + the fixed
U_primary/validation floor (~4 h) ≈ **~10 h**; stage-2 adds shard-1 (~6 h) ≈ **~16 h cumulative**; full
adds shards 2–7. All staged for the operator (each exceeds the ten-minute rule).

### Staged commands (illustrative — bind exact forms in the freeze-marker sub-amendment)
```powershell
# Stage 1 (1/8 aux): generate shard 0 of 8, then merge over the present shard(s).
node scripts/arc-phase3-branch-e3-learned-ranker.mjs --shard-aux --shard-index 0 --shard-count 8 `
  --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv --split-mode sha256_expansion `
  --out results/arc/phase3-branch-e3-learned-ranker
node scripts/arc-phase3-branch-e3-learned-ranker.mjs --merge `
  --out results/arc/phase3-branch-e3-learned-ranker            # adjudicates on FULL U_primary
# If Stage 1 is CONCLUSIVE (lift) -> stop. If SCREEN-only -> Stage 2: add shard 1, re-merge {0,1}.
# If Stage 2 SCREEN-only -> full: add shards 2..7, re-merge {0..7} (= resume-path-a).
```

### Public language (additions)
- bounded **lift**: "A learned ranker over a 1/8 (resp. 1/4) deterministic subsample of the auxiliary
  pool materially lifted held-out public-training selection — a **lower bound** on the full-pool ranker.
  Capability result only; not a Blackwell-sufficiency proof, ARC solve, public-eval, or Kaggle claim."
- bounded **non-positive**: "A 1/8 (resp. 1/4) aux screening run did **not** lift; this is
  **inconclusive** pending the full aux run — NOT an E3 floor." A binding floor requires shards {0..7}.
- Forbidden (unchanged) + : claiming a bounded non-positive is an E3 floor; choosing shards after
  reading U_primary labels; retuning anything but the declared shard ladder.

**No binding run, no verdict by this amendment.** Execution remains gated on the freeze-marker
sub-amendment above and is staged for the operator.
