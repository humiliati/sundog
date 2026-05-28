# Phase 0 Baselines and Admission Receipt

Filed: **2026-05-28 (PT)**

Roadmap: [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)

Phase 0 spec: [`PHASE0_TASK_SUBSET_SPEC.md`](PHASE0_TASK_SUBSET_SPEC.md)

Status: **PARTIAL ADMIT -- SUBSET CLEAN; ARC OPERATOR DESIGN HOLD**.

## Commands

Inventory:

```powershell
node scripts/arc-phase0-inventory.mjs --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --out results/arc/phase0-inventory
```

Draft register:

```powershell
node scripts/arc-phase0-draft-register.mjs --inventory results/arc/phase0-inventory/tasks.csv --out docs/prereg/arc/P0_TASK_REGISTER.draft.csv
```

Baselines:

```powershell
npm run arc:phase0:baselines -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase0-baselines
```

## Inventory Receipt

Local artifact: `results/arc/phase0-inventory/manifest.json`

| field | value |
| --- | --- |
| data source | `$env:USERPROFILE\Datasets\ARC-AGI-2\data` |
| generated at | `2026-05-28T02:09:32.799Z` |
| source fingerprint | `b96e08db6488e5ea436f9eaea0d270cbfab1f0483795c91799e03a263db406d7` |
| public training tasks | `1000` |
| public evaluation tasks | `120` |
| public evaluation test-output metadata | omitted |

Leak-control verdict: **PASS**. The mechanical inventory saw the public
evaluation split but did not emit evaluation test-output metadata.

## Register Receipt

Local artifact: [`P0_TASK_REGISTER.csv`](P0_TASK_REGISTER.csv)

The registered subset contains **36 public-training tasks**, all marked
`include`, stratified evenly across the six Phase 0 priors:

| primary prior | count |
| --- | ---: |
| symmetry | 6 |
| spatial_transform | 6 |
| counting | 6 |
| local_completion | 6 |
| color_role | 6 |
| objectness | 6 |

Integrity checks:

- all 36 rows are from the public training split;
- all 36 `inventory_row_hash` values resolve into
  `results/arc/phase0-inventory/tasks.csv`;
- no duplicate task IDs are present;
- all rows are marked manually inspected.

Caveat: the final register still carries draft-provenance notes from
`arc-phase0-draft-register.mjs`. Treat the primary-prior labels as a
metadata-stratified starting taxonomy, not as settled semantic ground truth.

## Baseline Receipt

Local artifact: `results/arc/phase0-baselines/summary.csv`

| baseline | exact tasks | exact rate | mean pixel accuracy |
| --- | ---: | ---: | ---: |
| random_valid | 0 / 36 | 0.0000 | 0.1530 |
| identity_copy | 0 / 36 | 0.0000 | 0.5585 |
| dsl_lite_v0 | 0 / 36 | 0.0000 | 0.5585 |

The subset clears the **too-easy** guard: `dsl_lite_v0` solves far below the
70% ceiling.

It triggers the registered **zero-floor caveat**: all cheap baselines solve
0% exact. Pixel accuracy is nonzero, especially for identity/copy, but exact
ARC task match is the primary score. This means the subset may be hard,
misclassified, or simply outside the frozen cheap baseline family.

## Verdict

**PARTIAL ADMIT.**

Admitted:

- Phase 0 inventory is complete.
- Phase 0 register meets the 36-task target and six-prior stratification target.
- Public-evaluation leak control held.
- Baseline numbers are filed.

Held:

- Full Phase 1 ARC operator design is **not** admitted yet, because the
  preregistered zero-floor caveat fired.

Allowed next work:

- Phase 0 hold review: inspect the 36 registered tasks and either confirm that
  the zero cheap-baseline floor is acceptable for this hard subset, or rebalance
  the register so at least one cheap baseline lands a nonzero exact floor.
- Improve or add non-Sundog cheap baselines only by append-only amendment before
  looking at any Sundog operator result.
- Synthetic-grid warmup definitions that do not touch registered ARC task
  performance.

Forbidden until a later **ADMIT**:

- implementing or tuning the ARC shadow-projection operator against the
  registered ARC subset;
- training a signature decoder;
- scoring Sundog-specific features on the registered subset;
- inspecting or tuning against public-evaluation task grids.
