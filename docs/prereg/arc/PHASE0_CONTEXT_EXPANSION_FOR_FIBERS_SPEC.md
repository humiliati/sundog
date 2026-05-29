# Phase 0 -- Context Expansion For Phase 3E Fibers

Roadmap: [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)

Parent specs:

- [`PHASE0_TASK_SUBSET_SPEC.md`](PHASE0_TASK_SUBSET_SPEC.md)
- [`PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md`](PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md)
- [`PHASE3E_PROGRAM_SKETCH_ORACLE_V2_SPEC.md`](PHASE3E_PROGRAM_SKETCH_ORACLE_V2_SPEC.md)

Filed: **2026-05-29 (PT)**

Status: **SPEC FILED; EXECUTION HOLD**. This file freezes a Phase 0 register
expansion protocol for the sole purpose of making the Phase 3E fiber geometry
measurable. It admits no register change, no certificate rerun, and no Branch E
solver until candidate tooling, expanded-register receipt, leak-check coverage,
and a freeze-marker amendment are filed.

## Purpose

The Phase 3E v2 binding receipt returned
`phase3e_v2_deferred_sparse_fibers`. It repaired the v1 oracle defect:
`program_sketch_v2` is non-vacuous across all six registered priors and passes
anti-prior-laundering plus anti-solver-leakage gates. The remaining obstruction
is geometric: on the 36-task registered universe, there are zero cross-task
neighbors within the frozen `epsilon_primary = 0.05`, and the minimum cross-task
context distance is `0.207`.

This spec attacks only that remaining obstruction. It expands the registered
public-training context universe so the frozen Phase 3E v2 certificate can test
whether `signature_palette` fibers become populated without changing the
oracle, context identity, distance function, kNN rule, or thresholds.

## Core Question

Does a larger, still pre-registered public-training ARC context universe contain
cross-task `signature_palette_context` neighbors within the frozen Phase 3E
radius, and if so are their `program_sketch_v2` labels locally compatible?

The question is not whether the expanded register is easier for a solver. It is
whether the finite context geometry becomes measurable.

## Frozen From Phase 3E v2

The following are inherited without modification:

- `program_sketch_v2` facet vocabulary and gate thresholds;
- `signature_palette_context`, `signature_only_context`,
  `metadata_only_context`, and `raw_grid_context` identities;
- `d_context_signature_palette` and all component distances;
- `epsilon_primary = 0.05`;
- `epsilon_strict = 0.025`;
- `epsilon_loose = 0.10`;
- kNN locality with `k = 3`;
- cross-task-neighbor requirement for primary locality;
- two-stage target-output barrier for public-training test contexts;
- prohibition on ARC public-evaluation inspection, Kaggle private outputs, and
  non-registered tasks in certificate adjudication.

Any change to these items voids this expansion as a Phase 3E continuation and
requires a new certificate spec.

## Expansion Target

The expansion target is:

```text
expanded_register_target = 108 included public-training tasks
per_prior_target = 18 included tasks per registered prior
new_tasks_required = 72 beyond the existing 36-task register
```

The six priors remain:

- `objectness`
- `counting`
- `symmetry`
- `spatial_transform`
- `local_completion`
- `color_role`

No prior may borrow surplus tasks from another prior. If any prior cannot reach
18 included tasks under the inclusion/exclusion rules below, the expansion
receipt must file HOLD rather than rebalance after seeing fiber distances.

Rationale: the current minimum cross-task distance is `0.207`, about 4.1x the
primary radius. Tripling the task register is the smallest expansion large
enough to give the nearest-neighbor distribution a meaningful chance to
contract while keeping manual inspection bounded.

## Selection Discipline

Selection must happen before any expanded-register Phase 3E v2 rerun.

Allowed before the expansion register is frozen:

- public-training metadata inventory;
- public-training manual grid inspection for prior assignment;
- non-Sundog task descriptions and exclusion notes;
- input-only or inventory-level coarse clustering used only to order the
  inspection queue.

Forbidden before the expansion register is frozen:

- running `phase3e_program_sketch_oracle_v2.py` on candidate tasks;
- computing `d_context_signature_palette` between candidates;
- computing target-output hashes, exact-output collision groups, or
  `program_sketch_v2` labels for candidates;
- using any Phase 3 decoder, Branch D bank score, solver success, or failure
  result to include/exclude a task;
- inspecting ARC public-evaluation grids or any Kaggle/private data.

After the expansion register is frozen, the Phase 3E v2 runner may read the
registered public-training outputs under the inherited no-target barrier.

## Candidate Ordering

Candidate ordering is mechanical so the expansion is not post-hoc nearest-
neighbor hunting.

1. Start from the public-training inventory used by Phase 0.
2. Remove the 36 already included tasks.
3. Remove invalid tasks and tasks with fewer than two training pairs.
4. Partition remaining tasks by coarse inventory hints for the six priors. A
   task may enter multiple candidate queues.
5. Within each prior queue, sort by:
   - descending number of matching coarse prior hints;
   - ascending absolute difference from that prior's current median grid area;
   - ascending `inventory_row_hash`;
   - ascending `task_id`.
6. Inspect tasks in sorted order until the prior reaches 18 included tasks or
   the queue is exhausted.

The exact candidate-ordering script must emit the ordered queue before manual
inspection begins. Manual inspection may reject a candidate only by a registered
exclusion reason, not by any certificate-distance result.

## Inclusion Criteria

A new task may enter the expansion register if all are true:

1. It is a public-training ARC-AGI-2 task not already included in
   `P0_TASK_REGISTER.csv`.
2. It has at least two training pairs and at least one public-training test
   query.
3. It can be assigned a primary prior from the six registered priors without
   relying on solver behavior.
4. Its rule can be described as an input-output transformation type rather than
   a task-specific lookup.
5. It preserves the Phase 0 evaluation discipline: manual inspection is allowed
   only because the task is public training, and no public-evaluation contents
   are inspected.
6. It has a predicted boundary from the existing Phase 0 vocabulary
   (`non-local information`, `capacity pressure`, `gauge-breaking ambiguity`,
   `full-state-only dependency`) or a named append-only boundary.

## Exclusion Criteria

Exclude a candidate if any are true:

1. The task requires external world knowledge, text/code/table lookup, or a
   non-ARC convention.
2. The rule cannot be assigned to a registered primary prior before looking at
   solver or certificate output.
3. The output appears to require unconstrained full-grid reconstruction rather
   than a compact transformation description.
4. The task is a near-duplicate of an already registered task by task ID,
   source duplication, or obvious literal copy of all train/test inputs.
5. The candidate was selected because of Phase 3E distance, collision, sketch,
   or solver-result information.
6. Public-evaluation or Kaggle/private data influenced inclusion.

Every manually inspected but rejected candidate must appear in the expansion
register with `status=exclude` and an exclusion reason.

## Register Artifacts

The expansion must not overwrite the original 36-task register. It creates:

- `docs/prereg/arc/P0_CONTEXT_EXPANSION_REGISTER.csv`
- `docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv`

`P0_CONTEXT_EXPANSION_REGISTER.csv` contains only new inspected candidates.
`P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv` is the concatenation of the original
36 included rows plus the expansion rows with `status=include`.

Both files use the Phase 0 register schema, with two additional required
columns:

| column | meaning |
| --- | --- |
| `expansion_batch` | Fixed string `fiber_context_expansion_v1`. |
| `selection_order_rank` | Rank in the pre-inspection candidate queue for the assigned primary prior. |

## Context Universe For The Expanded Certificate

The expanded Phase 3E v2 certificate uses the same context definitions:

```text
validation_lodo: each validation task training pair held out once
validation_pttest: each validation task public-training test query
test_lodo: each test task training pair held out once
pttest: each test task public-training test query
U_primary = test_lodo union pttest
U_all = validation_lodo union validation_pttest union test_lodo union pttest
```

The validation/test task partition is deterministic after the expanded register
is frozen:

1. For each prior, sort included task IDs by SHA-256 of
   `fiber_context_expansion_v1|task_id`.
2. Assign the first one third, rounded down but at least three tasks, to
   validation.
3. Assign the remaining tasks to test.

Same-task neighbors are still excluded from primary kNN locality. Finer LODO
increases diagnostic context count but cannot by itself satisfy the cross-task
fiber-locality gate.

## Measurement Plan

After the expanded register is frozen and admitted, run the unchanged Phase 3E
v2 certificate runner with the expanded register:

```powershell
node scripts/arc-phase3e-program-sketch-oracle-v2.mjs `
  --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv `
  --out results/arc/phase3e-program-sketch-oracle-v2-expanded
```

If the expanded pass remains under the repo's ten-minute rule, it may run
inline after the freeze-marker amendment. If timing exceeds ten minutes, stage
the exact PowerShell command, wall-clock estimate, resume-safety notes, and
branch-decision consequences instead.

## Expansion Branches

The Phase 0 expansion receipt branches before any certificate rerun:

| branch | condition | interpretation |
| --- | --- | --- |
| `phase0_fiber_expansion_admit` | 108-task balanced expanded register filed, leak checks pass, no selection-discipline violation. | The expanded Phase 3E v2 rerun is admitted. |
| `phase0_fiber_expansion_hold_insufficient_tasks` | Any prior cannot reach 18 included tasks under the rules. | The expansion target is not met; do not substitute other priors. |
| `phase0_fiber_expansion_hold_selection_audit` | Candidate ordering, exclusion logs, or register hashes are incomplete. | Repair paperwork before rerun. |
| `phase0_fiber_expansion_void_leak` | Public-evaluation/Kaggle/private data or post-hoc Phase 3E distance/sketch/solver information influenced selection. | Expansion void; no rerun admitted. |

## Expanded Certificate Branches

If `phase0_fiber_expansion_admit` is filed, the expanded Phase 3E v2 certificate
uses these branch names:

| branch | condition | interpretation |
| --- | --- | --- |
| `phase3e_v2_expanded_oracle_regression` | Any v2 oracle gate fails on the expanded `U_primary`. | The oracle was adequate on 36 tasks but not on the expanded register. |
| `phase3e_v2_expanded_exact_fiber_collision` | Exact or representation-level collision appears under unchanged identity. | Decoder-independent finite-context insufficiency witness. |
| `phase3e_v2_expanded_near_fiber_incompatibility` | At least one cross-task near pair within `epsilon_primary` has incompatible v2 sketches. | Locality obstruction for smooth Branch E selection. |
| `phase3e_v2_expanded_fiber_locality_positive` | No collision/incompatibility, at least 50% of primary contexts have fidelity-passing cross-task k=3 neighborhoods, and mean local incompatibility rate <= 0.10. | Certificate licenses a later Branch E selector test. Does not prove sufficiency. |
| `phase3e_v2_expanded_deferred_sparse_fibers` | Oracle gates pass, no collision/incompatibility, but fewer than 50% of primary contexts have fidelity-passing neighborhoods. | Even the expanded register is still too sparse at the frozen radius. |

## Required Outputs

Expansion receipt path:

`results/arc/phase0-context-expansion-for-fibers/`

Required expansion artifacts:

- `candidate_queue.csv`;
- `candidate_queue.sha256`;
- `manual_inspection_log.csv`;
- `P0_CONTEXT_EXPANSION_REGISTER.csv`;
- `P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv`;
- `prior_counts.csv`;
- `selection_audit.md`;
- `phase0_context_expansion_receipt.json`;
- `branch_adjudication.md`;
- `commands.md`;
- `hashes.json`.

Expanded certificate receipt path, if admitted:

`results/arc/phase3e-program-sketch-oracle-v2-expanded/`

It must contain the same artifact family as the v2 certificate plus an expanded
register hash and a comparison table against the 36-task v2 receipt:

- `U_primary`, `U_all`;
- min / median / p10 cross-task distance;
- number of near pairs at `epsilon_strict`, `epsilon_primary`, and
  `epsilon_loose`;
- fidelity-pass fraction;
- oracle gate margins;
- branch.

## Reserved Implementation Names

These names are reserved but not executable by this spec alone:

- candidate-ordering script:
  `scripts/arc-phase0-context-expansion-for-fibers.mjs`;
- npm script: `arc:phase0:context-expansion-for-fibers`;
- expansion receipt path: `results/arc/phase0-context-expansion-for-fibers/`;
- expanded certificate receipt path:
  `results/arc/phase3e-program-sketch-oracle-v2-expanded/`.

The freeze-marker amendment must add the candidate-ordering script, npm wiring,
result ignore paths, leak-check coverage, a small smoke fingerprint, and the
exact manual-inspection protocol before any expansion receipt is admitted.

## Public Language

Allowed before a binding expansion receipt:

> "Phase 0 has filed a context-expansion spec for Phase 3E fibers. It keeps the
> v2 oracle and all Phase 3E geometry thresholds frozen. No expanded register or
> rerun exists yet."

Allowed if `phase0_fiber_expansion_admit` is filed:

> "A balanced expanded public-training register has been filed for the Phase 3E
> fiber certificate. The expanded certificate rerun is admitted, but not yet
> adjudicated."

Allowed if `phase3e_v2_expanded_deferred_sparse_fibers` is filed:

> "Even after the registered context expansion, the Phase 3E v2 certificate
> remained too sparse at the frozen signature radius to certify fiber locality."

Forbidden:

- claiming the expansion proves signature sufficiency;
- claiming the expansion proves Branch E should work before an expanded
  certificate receipt;
- selecting expansion tasks by Phase 3E distances, sketches, solver outputs, or
  public-evaluation/Kaggle/private data;
- retuning `epsilon_primary`, kNN `k`, or `program_sketch_v2` after seeing the
  expanded distances;
- replacing a failed balanced expansion with surplus tasks from easier priors.

---

## Amendment A — Freeze Marker (2026-05-29 PT)

Append-only. This amendment records the tooling, the two frozen mechanical
interpretations, the runner partition resolution, the smoke fingerprint, and the
exact manual-inspection protocol required by §"Reserved Implementation Names"
before any expansion receipt is admitted. No register row exists yet; this only
unblocks Phase B inspection.

### Tooling added

- Candidate-ordering script: `scripts/arc-phase0-context-expansion-for-fibers.mjs`
  (Node, input-inventory only). Reads the frozen Phase 0 inventory
  (`results/arc/phase0-inventory/tasks.csv`) + `P0_TASK_REGISTER.csv`; never
  computes a Phase 3E distance, sketch, target-output hash, output-collision
  group, or solver result; never reads the held-out split.
- npm script: `arc:phase0:context-expansion-for-fibers`.
- Result ignore paths added to `.gitignore`:
  `results/arc/phase0-context-expansion-for-fibers/` and
  `results/arc/phase3e-program-sketch-oracle-v2-expanded/`.
- Inspection-support renderer: `scripts/arc_phase0_expansion_inspect.py`
  (read-only; renders public-training grids in queue order; computes no
  certificate distance/sketch/output-collision/solver result).
- Leak-check coverage: `npm run arc:phase0:leak-check` passes (0 fail / 0 warn);
  the new `.mjs` is among the 21 scanned non-allowlisted ARC scripts and carries
  no held-split path/data literal, so it is not allowlisted.

### Frozen mechanical interpretations (§"Candidate Ordering")

Two phrases in §"Candidate Ordering" are given a fixed, reproducible reading.
Because the candidate queue is emitted and SHA-256'd before any inspection, and
inspection rejects only by a registered exclusion reason, these readings fix a
deterministic inspection order without biasing the outcome:

1. "descending number of matching coarse prior hints" = descending count of the
   task's `prior_hints` tokens in the inventory (how many of the six priors the
   input-only inventory heuristic fired). Emitted as `matching_hint_count`.
2. "that prior's current median grid area" = the median of `max_area` over the
   originally registered 36-task set whose `primary_prior` equals that prior; a
   candidate is keyed by `abs(max_area − that median)`. Per-prior medians:
   objectness 122, counting 298, symmetry 81, spatial_transform 158,
   local_completion 168.5, color_role 210.

Full sort key per prior queue: (1) desc `matching_hint_count`, (2) asc
`abs_area_diff_from_prior_median`, (3) asc `inventory_row_hash`, (4) asc
`task_id`.

### Runner partition resolution (the §"Measurement Plan" runner)

§"Context Universe For The Expanded Certificate" defines a new validation/test
partition (SHA-256 of `fiber_context_expansion_v1|task_id`). The frozen Phase 3E
v2 runner assigned the val/test split from a hardcoded 36-task table, so it
cannot consume the expanded register unchanged. Resolution (geometry untouched):
`phase3e_program_sketch_oracle_v2.py` gains `--split-mode {frozen_v2,
sha256_expansion}`:

- `frozen_v2` (default) = the original hardcoded table. Verified to reproduce the
  committed v2 binding receipt's substantive fields byte-for-byte (registerHash,
  dataDirHash, contextUniverse, fiber, gates, branch all MATCH). Only
  `runnerSha256` advances `6163C208…` → `9D456143…` (file edited) and a new
  `splitMode` manifest key appears; the v2 receipt at `6163C208` remains the
  canonical v2 record.
- `sha256_expansion` = implements §"Context Universe" exactly (per `primary_prior`
  group, sort by SHA-256(`fiber_context_expansion_v1|`+task_id); first
  `max(3, floor(n/3))` → validation, rest → test; no train split). Confirmed
  deterministic (two runs byte-identical).

All Phase 3E geometry — identity, distance, `k=3`, `epsilon_*`, oracle, gates,
target barrier — is unchanged; only the val/test partition assignment is
generalized, which the spec itself froze.

### Smoke fingerprint

- `candidate_queue.sha256` =
  `43ea45506bc51c84f0d6603ddc2bb4d2c6c195a208bf2abf48728e4a1d11d82c` (re-run
  identical). Eligible pool 964 tasks (training, not registered, ≥2 train pairs,
  ≥1 test query); 4009 queue rows across 6 priors (a task may enter multiple
  queues). Pre-flight depth per prior (all ≥ 18 target): objectness 953, counting
  934, symmetry 311, spatial_transform 310, local_completion 558, color_role
  943 → no `phase0_fiber_expansion_hold_insufficient_tasks`.
- `sha256_expansion` on the current 36-task register yields contextUniverse
  `u_primary=73` (vs `frozen_v2`'s 25), 18 validation + 18 test tasks —
  confirming the partition path densifies the primary universe as intended (a
  smoke only; the binding expansion run uses the 108-task register).
- New runner `runnerSha256` =
  `9D456143F60490D8BF6A461610A870FB8733C46C1E0589DABE9C51A1B6603EE6`.

### Manual-inspection protocol (agent-performed, logged)

Per the approved lane decision, the maintainer's agent performs the inspection in
a single logged pass; the maintainer ratifies via the receipt +
`selection_audit.md`.

1. Order. Walk the six priors in the fixed order [objectness, counting, symmetry,
   spatial_transform, local_completion, color_role]. Within a prior, walk its
   `candidate_queue.csv` rows by ascending `selection_order_rank`.
2. Per candidate (not already decided). Render its public-training grids
   (`arc_phase0_expansion_inspect.py`) and decide: primary prior (one of the
   six), assigned from the input→output transformation type and never from
   solver/certificate behavior; secondary priors (subset of the six);
   `status ∈ {include, exclude}` with a registered reason; predicted_boundary
   keyed to the assigned prior by the existing register convention
   (symmetry→gauge-breaking ambiguity; spatial_transform→capacity pressure;
   counting→capacity pressure; local_completion→non-local information;
   color_role→gauge-breaking ambiguity; objectness→full-state-only dependency).
   An included task counts toward its assigned primary prior, regardless of which
   queue surfaced it.
3. Registered exclusion reasons (§"Exclusion Criteria"):
   `requires_external_knowledge`, `no_assignable_primary_prior`,
   `unconstrained_full_grid_reconstruction`, `near_duplicate_of_registered`,
   `selected_by_certificate_or_solver_info` and `held_split_or_kaggle_influence`
   (the last two must never fire — listed for completeness/audit).
4. Stop a prior at 18 includes. Do not rebalance across priors. If any prior's
   queue is exhausted before 18 includes → file
   `phase0_fiber_expansion_hold_insufficient_tasks`.
5. Log every inspected task (include or exclude) in `manual_inspection_log.csv`
   with its decision + reason.

---

## Amendment B — Expansion Receipt (2026-05-29 PT)

Append-only. Records the executed Phase B inspection and Phase C register
assembly. **Branch: `phase0_fiber_expansion_admit`.**

### Inspection (Phase B)

The maintainer's agent walked the frozen `candidate_queue.csv` in prior order,
classified ~150 distinct public-training tasks from their grids (train pairs
only; test outputs stayed under the renderer barrier), and assigned each a
primary prior. **0 hard-exclusions** (every walked task was a genuine
input→output transformation assignable to one of the six priors) and **0
discipline tripwires** (no certificate/solver/distance/held-split/Kaggle
information entered any decision). The coordinator spot-verified a sample across
all six priors (confident calls e.g. `3af2c5a8` symmetry; borderline calls e.g.
`af24b4cc`, `5614dbcf`, `73182012`, `e5c44e8f`, `3ee1011a` — all defensible
best-fit assignments) and confirmed the "3×3→6×6 four-fold" symmetry cluster
(`62c24649`, `67e8384a`, `46442a0e`, `0c786b71`) are genuinely distinct tasks
(criterion 4), not literal duplicates.

### Binding-selection correction (transparency)

The inspection agent was given an over-broad new-include target (18 new per
prior) rather than the spec's 12 new per prior (§"Expansion Target":
`new_tasks_required = 72`, the original register supplying the other 6 per
prior). The binding selection therefore takes the **12 lowest
`selection_order_rank` includes per prior** (= 72 new). Because the candidate
queue rank order was frozen and SHA-256'd before any inspection (Amendment A),
and the agent included in ascending-rank order, this is identical to what a
correct stop-at-12 walk over the frozen queue produces; no certificate/output/
solver information influenced which tasks were selected. The remaining 36
inspected includes are recorded in `P0_CONTEXT_EXPANSION_REGISTER.csv` with
`status=exclude`, `exclusion_reason=over_inspection_surplus_beyond_12_new_per_prior`
(over-inspection beyond the pre-registered quota, not a quality rejection). The
full 108-include inspection trail is preserved in `manual_inspection_log.csv`.

### Registers (Phase C)

- `docs/prereg/arc/P0_CONTEXT_EXPANSION_REGISTER.csv` — 108 inspected new
  candidates (72 `include` + 36 surplus `exclude`); SHA-256 `0a2e8ac3…`.
- `docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv` — the binding
  expanded register: 36 original includes + 72 new includes = **108 tasks,
  exactly 18 per prior (6 original + 12 new)**; SHA-256 `7c56141c…`. The
  original `P0_TASK_REGISTER.csv` is not modified.
- Frozen inputs: `candidate_queue.csv` SHA-256 `43ea4550…`;
  `manual_inspection_log.csv` SHA-256 `8a386029…`.

### Partition (for the expanded certificate)

The `sha256_expansion` rule (per `primary_prior` group, sort by
SHA-256(`fiber_context_expansion_v1|`+task_id), first `max(3, floor(n/3))` →
validation) over the 18-per-prior register yields **36 validation + 72 test
tasks** (6 validation / 12 test per prior). Same-task neighbors remain excluded
from primary kNN.

### Adjudication

All six priors reached 18 includes under the inclusion/exclusion rules with no
cross-prior rebalancing → **`phase0_fiber_expansion_admit`**. The expanded
Phase 3E v2 certificate rerun is admitted (run the unchanged-geometry runner with
`--split-mode sha256_expansion` against
`P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv`). Receipt artifacts:
`results/arc/phase0-context-expansion-for-fibers/`
(`phase0_context_expansion_receipt.json`, `prior_counts.csv`,
`branch_adjudication.md`, `commands.md`, `hashes.json`).

---

## Amendment C — Expanded Certificate Verdict (2026-05-29 PT)

Append-only. The expanded Phase 3E v2 certificate was run on the 108-task
register, pinned to freeze-marker commit `a0f1a4b`
(`runnerSha256 9D456143…`, `gitDirty=false`, `splitMode=sha256_expansion`,
~5m47s, deterministic). Receipt:
`results/arc/phase3e-program-sketch-oracle-v2-expanded/`.

**Branch: `phase3e_v2_expanded_oracle_regression`** (the runner's internal branch
`phase3e_oracle_v2_solver_leakage`; the §"Expanded Certificate Branches" name is
`…oracle_regression`: a v2 oracle gate fails on the expanded `U_primary`).

### What happened

The context universe densified massively: `U_primary` **25 → 336**, `U_all`
**49 → 491** (validation_lodo 118, validation_pttest 37, test_lodo 259,
pttest 77). But the v2 oracle's **anti-solver-leakage gate failed**:

| gate | 36-task v2 | 108-task expanded | threshold | expanded result |
| --- | --- | --- | --- | --- |
| anti-vacuity `vacuous_fraction` | 0.00 | **0.00** | ≤ 0.20 | PASS |
| anti-prior-laundering `violation_fraction` | 0.00 | **0.00** | ≤ 0.10 | PASS |
| leakage `unique_core_sketch_fraction` | 0.36 | **0.55** | ≤ 0.60 | pass |
| leakage `core_sketch_exact_lookup_fraction` | 0.04 | **0.351** | ≤ 0.20 | **FAIL** |

`core_sketch_exact_lookup_fraction` is the fraction of primary contexts whose
`core_sketch_tuple` group maps to a single target output (a lookup keyed only on
the core sketch would reproduce the output). At 36 tasks it was 0.04; at 108
tasks it is 0.351. As the registered universe diversifies, the framing-agnostic
nine-facet core sketch becomes specific enough that, on a third of primary
contexts, the sketch alone determines the output — which the anti-solver-leakage
gate (designed to reject a sketch acting as a covert lookup/solver) correctly
fires on. **The v2 oracle's leakage certification does not transfer from 36 to
108 tasks.**

### Fiber geometry (computed, but not adjudicated — gate failed first)

| metric | 36-task v2 | 108-task expanded |
| --- | --- | --- |
| min cross-task `d_context_signature_palette` | 0.207 | **0.196** |
| near pairs within `epsilon_primary=0.05` | 0 | **0** |
| exact / representation collisions | 0 | **0** |
| near-fiber incompatibilities | 0 | **0** |
| fidelity-pass fraction | 0.0 | **0.0** |

Even setting the gate aside, **tripling the register contracted the
nearest-neighbor distance by only ~5%** (0.207 → 0.196), nowhere near the ~4×
contraction needed to reach `epsilon_primary`. The fibers remain sparse; had the
oracle gate held, the branch would have been `…expanded_deferred_sparse_fibers`.

### What this does and does not establish

**Does**: (1) falsify the implicit assumption that the gate-clean v2 oracle
remains gate-clean on a larger register — the anti-solver-leakage threshold is
register-size/diversity-sensitive; (2) show the chosen expansion (3× the tasks,
balanced, manually inspected) is insufficient to populate `signature_palette`
fibers at the frozen radius (min distance barely moved).

**Does not**: prove or disprove `signature_palette` sufficiency; license or block
a Branch E solver; certify any fiber locality or incompatibility (the oracle gate
fails before locality is adjudicable). The 36-task v2 receipt
(`phase3e_v2_deferred_sparse_fibers`) and the seven full-grid floors stand
unchanged. No threshold was retuned after seeing the expanded output (doing so
would be a post-hoc p-hack; the gate fired as pre-registered).

### Path forward (each needs its own pre-registered spec)

- A **register-size/diversity-robust oracle (v3)** whose anti-solver-leakage
  metric is invariant to universe size (e.g., normalize exact-lookup against the
  per-task LODO base rate, or measure leakage on held-pairs rather than the full
  primary set), so a larger register can be adjudicated for fiber locality.
- Accept the **sparsity result as durable**: even a manually-curated 3× balanced
  expansion contracts min cross-task distance only 0.207 → 0.196, so populating
  `signature_palette` fibers at `epsilon_primary=0.05` likely needs a different
  locality test or a far larger / differently-constructed universe — a finding in
  its own right.
- Proceed to **Branch E on capability grounds** (unchanged from the 36-task
  reading: the certificate found no collision to block it and cannot certify
  locality to motivate a smooth selector).

**Public-language constraint**: the permitted addition is the factual
oracle-regression + still-sparse statement above; sufficiency, Branch-E-license,
collision-found, and locality-positive statements remain forbidden (none
obtained).
