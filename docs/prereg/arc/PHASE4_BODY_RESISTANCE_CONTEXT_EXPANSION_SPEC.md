# ARC Phase 4 v2 -- Body-Resistance Context Expansion

Parent / boundary specs:

- [`PHASE4_BODY_RESISTANCE_SPEC.md`](PHASE4_BODY_RESISTANCE_SPEC.md)
- [`PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md`](PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md)
- [`PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md`](PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md)
- [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)
- [`../../CROSS_SUBSTRATE_NOTES.md`](../../CROSS_SUBSTRATE_NOTES.md)

Drafted: **2026-06-01 (PT)**

Status: **DRAFT ONLY; EXECUTION NOT ADMITTED; TOOLING NOT FROZEN**.

This is the disciplined reopen of Phase 4 v1's `arc_body_inconclusive` verdict.
It does not lower or reinterpret the v1 thresholds. It asks whether the same
body-resistance estimator clears the unchanged high-dimensional bar when the
context universe is expanded by a spectrum-blind rule.

## Boundary Citation

Phase 4 v1 ran the C1/Mesa participation-ratio + `FVE(body|shadow)` estimators on
the 108-task expanded register (`U_all = 491`) and returned
`arc_body_inconclusive`.

The v1 measurement was directionally non-marginal:

| measure | v1 value |
| --- | ---: |
| body participation ratio | **9.15** |
| top-1 energy fraction | 0.287 |
| energy ranks 90 / 95 / 99% | 96 / 159 / 293 |
| matched-dim held-out `FVE(body|top-28 PCA)` | **0.659** |
| top-200 PCA held-out FVE | 0.789 |

The frozen high-dimensional gate did not fire because the PR clause required
`PR_HIGH_MIN = 20.0`, and `9.15 < 20.0`. The matched-dim FVE clause did pass
(`0.659 <= 0.90`), but the gate is an AND. V1 therefore remains inconclusive.

The permitted inference is narrow: the body spectrum is broad and
reconstruction-resistant, but the 491-context register may under-populate the
population spectrum. A larger context universe is justified only as a
sample-size test of the unchanged v1 bar.

## Core Question

If the ARC public-training context universe is expanded by a deterministic,
spectrum-blind rule, does the raw-grid body participation ratio clear the
unchanged `PR_HIGH_MIN = 20.0` bar while the matched-dim FVE clause remains below
the unchanged `FVE_RECON_CEILING = 0.90`?

## Frozen Thresholds Carried From v1

These values are inherited unchanged from `PHASE4_BODY_RESISTANCE_SPEC.md`
Amendment 1:

```text
PR_HIGH_MIN              = 20.0
PR_MARGINAL_MAX          = 5.0
FVE_RECON_CEILING        = 0.90
FVE_MARGINAL_MIN         = 0.95
PR_BOUND_SATURATION_MAX  = 0.90
RIDGE_LAMBDA             = 1.0
SHADOW_DIM_K             = 28
heldout split            = sha256(instance_id) % 10 < 3
ENERGY_LEVELS            = [0.90, 0.95, 0.99]
PCA_K_GRID               = [1, 2, 5, 10, 28, 50, 100, 200]
```

Forbidden: lowering `PR_HIGH_MIN` to the observed v1 value, changing the FVE
ceilings, changing the held-out split, changing the PCA grid after seeing the
expanded spectrum, or adding a new branch keyed to the v1 outcome.

## Spectrum-Blind Context Expansion

Primary expanded context universe:

```text
register_v2 = all ARC-AGI-2 public-training task ids present under --data-dir
              minus public-evaluation/private ids
              minus files that fail JSON/grid validation
```

No manual task inspection, no prior balancing, no task selection by visual
complexity, no selection by body spectrum, no selection by candidate-solver
outcome, and no exclusion because a task looks low- or high-dimensional.

Context construction:

1. `train_lodo`: for every public-training task, hold out each train pair once
   and condition on the remaining train pairs.
2. `pttest`: for every public-training task with public-training test outputs,
   include each public-training test query as a target-known public-training
   context.
3. `U_all_expanded = train_lodo union pttest`.

The context identity and body encoding are inherited from v1:

```text
body = raw_grid_onehot(query_grid)     # 9900-dim
metadata_shadow = metadata_vector(query_grid)   # 28-dim
signature_palette = feature_vector(query_grid, signature_palette arm)
matched_dim PCA cut = top-28 body PCs
```

The public-evaluation split remains forbidden. This is still training-split-only.

## Sample-Size Diagnostics

The binding run must report, without changing the branch decision:

- total task count;
- context count by lane (`train_lodo`, `pttest`);
- PR sample bound `min(n_contexts - 1, body_dim)`;
- `PR / sample_bound`;
- top-1 energy fraction;
- 90/95/99% energy ranks;
- matched-dim FVE curve at the frozen `PCA_K_GRID`;
- downsample stability at frozen context prefixes.

Downsample stability:

```text
prefix_sizes = [491, 750, 1000, 1500, 2000, 3000, all]
prefix_order = lexicographic task id, then lane, then instance id
```

If a prefix size exceeds the available context count, skip it and log the skip.
Prefix diagnostics are descriptive only. They cannot change the branch.

## Branches

Branch precedence is table order.

| branch | condition | interpretation |
| --- | --- | --- |
| `arc_body_high_dim_expanded` | `PR / sample_bound <= 0.90`, body PR `>= 20.0`, and matched-dim held-out `FVE(body|top-28 PCA) <= 0.90`. | The expanded context universe clears the unchanged v1 high-dimensional body gate. |
| `arc_body_marginal_expanded` | body PR `<= 5.0` or matched-dim held-out `FVE(body|top-28 PCA) >= 0.95`. | The larger universe shows ARC's body collapses to the marginal band after all. |
| `arc_body_sample_saturated_expanded` | `PR / sample_bound > 0.90`. | The run is sample-bound and cannot adjudicate the high-dimensional claim. |
| `arc_body_inconclusive_expanded` | none of the above. | The larger universe still does not adjudicate the body-resistance threshold. |

The branch is based only on the full `U_all_expanded` run. Prefix diagnostics do
not create alternative thresholds.

## Required Artifacts

Binding output path:

```text
results/arc/phase4-body-resistance-context-expanded/
```

Required files:

- `manifest.json`
- `task_inventory.csv`
- `split.csv`
- `body_spectrum.csv`
- `shadow_fve.csv`
- `matched_dim_fve_curve.csv`
- `prefix_dimensionality.csv`
- `per_lane_dimensionality.csv`
- `phase4_body_resistance_context_expansion_receipt.json`
- `branch_adjudication.md`
- `commands.md`
- `hashes.json`

The manifest must record parent spec hashes, runner hash, register/task-inventory
hash, data directory hash, context count, frozen thresholds, branch, and whether
any task was excluded by JSON/grid validation.

## Reserved Implementation Names

- Python runner: `docs/prereg/arc/phase4_body_resistance_context_expansion.py`
- Node wrapper: `scripts/arc-phase4-body-resistance-context-expansion.mjs`
- npm script: `arc:phase4:body-resistance:context-expansion`
- receipt path: `results/arc/phase4-body-resistance-context-expanded/`

## Ten-Minute Rule

This run may be larger than Phase 4 v1. The freeze-marker amendment must include:

1. `py_compile` receipt;
2. dry-run artifact receipt;
3. capped context inventory smoke;
4. capped spectral smoke on a small prefix;
5. measured seconds per context and estimated full binding wall time;
6. exact staged PowerShell binding command if the estimate exceeds the
   repository ten-minute rule.

## Public Language

Allowed before a binding receipt:

> "ARC Phase 4 v2 has a draft context-expansion spec. It keeps the v1
> body-resistance thresholds unchanged and asks whether a larger, spectrum-blind
> public-training context universe lets the raw-grid body PR clear the frozen
> high-dimensional bar. No tooling freeze marker or binding receipt exists yet."

Allowed if `arc_body_high_dim_expanded`:

> "With a larger spectrum-blind public-training context universe, ARC's raw-grid
> body cleared the unchanged high-dimensional body gate. This is a read-off
> body-dimensionality result, not a control witness, Blackwell-sufficiency proof,
> ARC solve, public-evaluation result, or Kaggle claim."

Allowed if `arc_body_inconclusive_expanded`:

> "The larger context universe still did not adjudicate the frozen
> body-resistance threshold. The v1 inconclusive result remains, with updated
> sample-size diagnostics."

Forbidden:

- claiming v1 was high-dimensional by lowering the PR bar;
- claiming the expanded run proves a control-sufficient shadow;
- reading `signature_palette` reconstruction FVE as the verdict;
- any public-evaluation / Kaggle claim;
- excluding tasks after seeing the expanded spectrum;
- retuning thresholds after seeing the expanded spectrum.

---

## Amendment 1 — Freeze Marker (2026-06-01 PT)

Status: **EXECUTION ADMITTED; TOOLING FROZEN.** No threshold is changed; the
estimators are carried from v1 by *importing* its runner (hash pinned below).

### Frozen tooling

| component | path | sha256 (16) |
| --- | --- | --- |
| runner | `docs/prereg/arc/phase4_body_resistance_context_expansion.py` | `c831cc3867668f25` |
| wrapper | `scripts/arc-phase4-body-resistance-context-expansion.mjs` | `470e5f88f9d8cd2d` |
| v1 runner (imported — estimators + thresholds) | `docs/prereg/arc/phase4_body_resistance.py` | `5ca151e11f654ac4` |
| representations (imported) | `docs/prereg/arc/phase3d_mask_target_v3.py` | `9f6b4ba7931a08f0` |

The v1 runner hash `5ca151e1…` is **identical** to the value frozen in
`PHASE4_BODY_RESISTANCE_SPEC.md` Amendment 1, so `PR_HIGH_MIN=20.0`,
`FVE_RECON_CEILING=0.90`, `FVE_MARGINAL_MIN=0.95`, `PR_MARGINAL_MAX=5.0`,
`PR_BOUND_SATURATION_MAX=0.90`, `RIDGE_LAMBDA=1.0`, `SHADOW_DIM_K=28`, the held-out
rule, `ENERGY_LEVELS`, and `PCA_K_GRID` are **byte-for-byte the v1 constants** — the
runner reads them off the imported v1 module, it does not redefine them. npm:
`arc:phase4:body-resistance:context-expansion`. Result path (gitignored):
`results/arc/phase4-body-resistance-context-expanded/`.

### What is new vs v1 (only these)

- **All-training loader** (`load_all_training_tasks`): every `*.json` under
  `--data-dir/training` (1000 tasks), minus JSON/grid-invalid files; no register, no
  spectrum/solver/complexity selection. Two lanes — `train_lodo` (each train pair held
  out once) and `pttest` (each public-training test query); body = the held-out /
  test **input** grid only (no targets). Canonical order `(task_id, lane, instance_id)`.
- **Prefix downsample stability** + the 4-branch `*_expanded` adjudication (table-order
  precedence) per the spec.
- The matched-dim FVE curve is folded into **one** train-body SVD (the v2 train set is
  ~9× larger); verified **byte-identical** to calling v1's `fve_pca` per k (max abs
  diff `0.0` on a 200×9900 fixture).

### Leak receipt + smoke fingerprints

- `npm run arc:phase0:leak-check`: **0 fail / 0 warn**; register still 36 training /
  0 eval-blind / 0 eval (the v2 runner does not touch the register); 27 ARC scripts
  scanned (incl. the new wrapper); no Kaggle scaffolding. Training-split only
  (`assert_training_data_dir`; reads `…/training` only; eval forbidden).
- `py_compile`: clean; module imports with no side effects (v1 constants surfaced).
- Dry-run: 12-artifact stub. Inventory smoke: **1000 valid tasks, 0 excluded, 4308
  contexts** (`train_lodo 3232` / `pttest 1076`). Capped spectral smoke (120 tasks /
  497 contexts): `arc_body_inconclusive_expanded`, PR 10.81, in **20 s**.

### Timing → inline

Full binding spectral run over all **4308** contexts measured at **≈ 180 s** (≈ 0.04
s/context; ~11 SVDs of ≤ 4308×9900 + one 4124-dim ridge), **well under** the
repository ten-minute rule. The binding run therefore executes **inline** on a clean
ARC subtree — no staged operator command required.

### Staged binding command (for reference / re-run)

```powershell
cd C:\Users\hughe\Dev\sundog
node scripts/arc-phase4-body-resistance-context-expansion.mjs `
  --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
  --out results/arc/phase4-body-resistance-context-expanded
```

(The runner refuses a dirty worktree unless `--allow-dirty`, so the freeze marker is
committed before the binding run; the manifest `gitDirty` flag reflects only
concurrent non-ARC sibling lanes.)
