# Lattice-Deduction Phase 1 - I5 Rollout Contract

> 2026-06-03. Draft contract for the last build-gate-runnable piece of
> `scripts/lattice_ldt_model.py`: the iterative deduction rollout. This is
> execution-side scaffolding, not a B-layer result and not a build-gate pass.
> It freezes the intended semantics before the full build-gate run so a miss
> cannot quietly become post-hoc solver engineering.

## 0. Why This Contract Exists

The LDT reimplementation now has:

- a parameter-faithful architecture skeleton (`798,346` params at full config);
- real Sudoku-Extreme loading and parser verification;
- CUDA smoke coverage;
- residual-stream capture at `(iteration, layer)` grains for later B1/B2/B3.

The last inferred implementation detail is **I5**:

```text
forward -> threshold-eliminate -> update lattice -> repeat,
with deterministic backtracking when the lattice conflicts or stalls.
```

The current cheap `solve_rate` is one-shot argmax. It is useful for smoke only.
The build-gate target must use the iterative lattice-deduction rollout below.

## 1. Scope Boundary

The rollout is an **evaluation / inference policy** for the build-gate model. It
is not:

- a new learned component;
- a Sudoku constraint-programming solver hidden under the model;
- a B2/B1/B3 measurement;
- an opportunity to tune thresholds after seeing full build-gate failures.

The answer key may be used only for evaluation counters (`exact`, false
elimination audit, parser checks). It may not be used to block eliminations,
choose branches, repair lattices, or stop early except by final scoring.

## 2. State Representation

Each rollout state is a candidate lattice:

```text
lattice: bool[81, 9]
```

Conventions:

- `lattice[i, d] = 1` means digit `d+1` remains a candidate in cell `i`.
- Clue cells start as singletons and are locked.
- Non-clue cells start with all candidates open unless the dataset encodes a
  stricter initial lattice later by explicit amendment.
- A solved lattice has exactly one open candidate in every cell.
- An empty cell has zero candidates and is a conflict.

The rollout may inspect candidate counts and clue locks. It may not run a
separate Sudoku propagation engine. Terminal validity checking is allowed only
to decide whether a complete singleton grid is a valid Sudoku; it is not used as
an internal pruning rule unless a later amendment admits it.

## 3. Model Output Semantics

The current model head is trained with BCE toward the solution one-hot. Therefore
the canonical interpretation is:

```text
keep_prob[i, d] = sigmoid(candidate_logit[i, d])
drop_conf[i, d] = 1 - keep_prob[i, d]
```

The rollout commits an elimination when:

```text
drop_conf[i, d] >= theta_drop
```

Default:

```text
theta_drop = 0.5
theta_cls = 0.6
```

The implementation must record these names explicitly in the manifest. If a
future runner changes the head to emit direct elimination logits, the contract
must be amended before any verdict-bearing run; it must not silently reuse the
same threshold name with opposite semantics.

## 4. Deduction Step

At each node, the model receives the current lattice and returns candidate logits
plus the conflict logit.

One deduction step:

1. Compute `p_conflict = sigmoid(conflict_logit)`.
2. If any cell is empty, mark node `empty_cell_conflict`.
3. If `p_conflict > theta_cls`, mark node `model_conflict`.
4. Otherwise compute proposed eliminations for open, non-clue candidates using
   `drop_conf >= theta_drop`.
5. Apply eliminations simultaneously with the last-candidate guard below.
6. If at least one elimination is committed, continue from the updated lattice.
7. If no elimination is committed and the lattice is solved, terminal-score it.
8. If no elimination is committed and the lattice is unsolved, branch.

### Last-Candidate Guard

The rollout must never create an empty cell by committing a batch of
eliminations. For each cell:

- proposed eliminations are sorted by `drop_conf` descending, tie by digit
  ascending;
- candidates are removed in that order only while at least one candidate remains;
- any proposed removal that would empty the cell is suppressed and counted as
  `blocked_last_candidate`.

This guard uses only the current lattice, not the answer key. A false elimination
that removes the true solution digit is still allowed to happen; it is counted
afterward by comparing the committed removals to the answer.

## 5. Branching / Backtracking

Branching is admitted only when the model stalls:

```text
no committed eliminations AND not solved AND not conflicted
```

Primary branch policy:

1. Choose the unsolved non-clue cell with the fewest open candidates greater
   than one.
2. Tie by row-major cell index.
3. Order that cell's candidate branches by latest `keep_prob` descending.
4. Tie by digit ascending.
5. Each branch sets the chosen cell to a singleton and leaves every other cell's
   candidates unchanged.

The branch stack is deterministic depth-first search. To preserve the chosen
order with a LIFO stack, push branches in reverse order.

The search may use a hash of the lattice to avoid revisiting identical states.
It may not use a CP solver, exact Sudoku enumerator, or the known solution to
choose branches.

## 6. Conflict Handling

If a node is marked conflicted:

- increment the appropriate conflict counter;
- discard the node;
- pop the next branch from the DFS stack;
- if no branch remains, return `unsolved_conflict_exhausted`.

Conflict sources are reported separately:

- `empty_cell_conflict`;
- `model_conflict`;
- `terminal_invalid_grid`;
- `node_cap_exceeded`;
- `step_cap_exceeded`.

Terminal invalidity means all cells are singleton but the resulting grid violates
row/column/box validity. This can reject a complete prediction, but it does not
permit earlier rule-based pruning.

## 7. Caps and Stop Reasons

The build-gate rollout must be bounded. Proposed initial caps, frozen before the
first full build-gate run:

```text
max_deduction_steps_per_node = 64
max_search_nodes_per_puzzle = 4096
max_wall_s_per_puzzle = optional, diagnostic only
```

Stop reasons:

| stop reason | meaning |
| --- | --- |
| `solved_exact` | complete valid grid equals the answer |
| `solved_valid_wrong` | complete valid grid, wrong answer |
| `terminal_invalid_grid` | complete singleton grid violates Sudoku rules |
| `unsolved_no_branches` | stalled and no legal branch exists |
| `unsolved_conflict_exhausted` | all branches conflicted |
| `node_cap_exceeded` | DFS node cap hit |
| `step_cap_exceeded` | per-node deduction cap hit |

Only `solved_exact` counts toward build-gate accuracy.

## 8. Build-Gate Metrics

Primary build-gate metric:

```text
rollout_exact_rate = mean(stop_reason == solved_exact)
```

Build-gate branch:

```text
build_gate_pass    iff rollout_exact_rate >= 0.999
build_gate_partial iff 0 < rollout_exact_rate < 0.999
build_gate_fail    iff rollout_exact_rate == 0
```

Diagnostics, always reported:

- `rollout_valid_rate`;
- `one_shot_exact_rate` (smoke/diagnostic only);
- `avg_committed_eliminations`;
- `false_elimination_rate_answer_key_audit`;
- `avg_nodes_expanded`;
- `node_cap_fraction`;
- `step_cap_fraction`;
- `avg_branches_created`;
- `model_conflict_count`;
- `empty_cell_conflict_count`;
- `blocked_last_candidate_count`;
- `stalled_branch_count`.

The false-elimination audit is a build-gate diagnostic only. The B2 fiber
false-elimination metric is stricter and lives downstream of a build-gate pass.

## 9. Receipt Contract

`results/lattice/build-gate-sudoku-extreme/manifest.json` must include:

```json
{
  "rolloutContract": "docs/lattice/PHASE1_I5_ROLLOUT_CONTRACT.md",
  "rolloutVersion": "phase1_i5_v1",
  "thetaDrop": 0.5,
  "thetaCls": 0.6,
  "logitSemantics": "candidate_keep_logit",
  "dropConfFormula": "1 - sigmoid(candidate_logit)",
  "maxDeductionStepsPerNode": 64,
  "maxSearchNodesPerPuzzle": 4096,
  "rolloutExactRate": null,
  "oneShotExactRate": null,
  "stopReasonCounts": {},
  "diagnostics": {}
}
```

The implementation should also write a per-puzzle JSONL or CSV file with:

- puzzle index;
- source if available;
- clue count;
- stop reason;
- exact/valid booleans;
- nodes expanded;
- branches created;
- committed eliminations;
- false eliminations against the answer key;
- conflict counts;
- optional compact final grid string.

## 10. Smoke Before Full Build-Gate

After implementing this contract, run only a bounded smoke inline:

```powershell
python scripts/lattice_ldt_model.py --mode smoke --data-dir docs/lattice/Soduko-Extreme --out results/lattice/_i5_rollout_smoke
```

The smoke must verify:

- rollout function executes on synthetic and real loaded puzzles;
- no clue cell is changed;
- no committed update creates an empty cell;
- stop reasons are populated;
- manifest records rollout metadata.

The full build-gate is expected to exceed the repo's inline-run budget and should
be staged as an operator/GPU-window command after smoke.

## 11. Amendment Rules

After this contract is implemented, changes to the following require an explicit
amendment before a full build-gate run:

- `theta_drop` / `theta_cls`;
- output-logit semantics;
- branching cell choice;
- candidate branch order;
- node or step caps;
- admission of any Sudoku rule propagation beyond terminal validity checking;
- use of terminal validity for internal pruning;
- pass/partial/fail thresholds.

If the first full build-gate returns `build_gate_partial`, do not tune I5 in place.
File an amendment naming which inference [I2-I5] is being changed and why.
