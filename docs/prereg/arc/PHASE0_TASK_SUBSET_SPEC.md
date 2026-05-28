# Phase 0 -- ARC-AGI Task Subset Specification

Roadmap: [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)

Filed: **2026-05-28 (PT)**

Status: **OPEN -- NO OPERATOR DESIGN ADMITTED**.

This is the first ARC-AGI preregistration artifact. Its job is to make the task
subset, exclusion criteria, evaluation handling, and baseline slate crisp before
any Sundog-specific shadow operator is designed.

## Claim Under Test

The Phase 0 claim is not that Sundog can solve ARC. The claim is narrower:

> There exists a registered, non-trivial ARC-AGI-2 task subset whose rules are
> naturally describable by core-knowledge priors -- objectness, counting,
> symmetry, spatial transformation, local completion, and color-role mapping --
> and whose boundaries can test whether a shadow-projected signature is
> sufficient without full grid reconstruction.

If this subset cannot be registered without leaking evaluation answers or
hand-picking post-hoc successes, the ARC coupling is blocked before operator
design.

## Data Source and Split Policy

Authoritative data source: the public
[arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) repository.

The local working copy should live outside this repo unless the maintainer
explicitly decides to vendor a small fixture. Suggested path:
`$env:USERPROFILE\Datasets\ARC-AGI-2`.

Split handling:

| split | allowed use in Phase 0 | restrictions |
| --- | --- | --- |
| public training | Inventory, manual inspection, taxonomy, development subset, cheap baselines. | Allowed to inspect; still record every inclusion/exclusion decision. |
| public evaluation | Locked adjudication subset only. | No manual grid inspection while designing the operator. Any selection must be by preregistered metadata/hash rule, and scripts must not emit test outputs. |
| Kaggle private / semi-private | Phase 6 submission only. | No access in Phase 0. Submission notebook must obey Kaggle no-internet evaluation. |

## Inventory Command

The Phase 0 inventory is mechanical and should complete in seconds:

```powershell
node scripts/arc-phase0-inventory.mjs --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --out results/arc/phase0-inventory
```

The script emits metadata only: task IDs, split counts, pair counts, dimensions,
palette sizes, simple shape-change flags, density ranges, connected-component
counts, and coarse prior hints. It must not emit public-evaluation test output
grids unless explicitly run with the evaluation-output override for a final,
post-freeze audit.

## Inclusion Criteria

A task may enter the registered development subset if all are true:

1. It is a valid ARC-AGI-2 JSON task with at least two training pairs and at
   least one test input.
2. Its apparent rule can be assigned to at least one core-knowledge prior:
   objectness, counting, symmetry, spatial transformation, local completion, or
   color-role mapping.
3. The expected transformation can be described without task-specific language
   labels, external world knowledge, or memorized lookup tables.
4. The output grid can be evaluated by exact match, and any auxiliary metric is
   recorded as diagnostic only.
5. The task has an explicit predicted failure boundary for the future signature
   audit: non-local information, capacity pressure, gauge-breaking ambiguity, or
   full-state-only dependency.

Target size: **36 public-training tasks**, stratified across the six priors
above. Minimum viable size: **18 tasks**, with at least three priors represented
by four or more tasks each. If the minimum cannot be met from public training
without relaxing the criteria, Phase 0 fails and the ARC lane retreats to
synthetic grid warmups.

## Exclusion Criteria

Exclude tasks from the Phase 0 development subset if any are true:

1. The rule appears to require long sequential program search rather than a
   compact abstraction.
2. The answer depends on a global readout that no local signature could
   plausibly retain.
3. The output information content is so high that exact reconstruction would be
   the real task.
4. The task is text-like, code-like, table-lookup-like, or otherwise outside
   ARC core-knowledge priors.
5. The task becomes classifiable only after inspecting multiple candidate
   solver failures.
6. Public-evaluation task contents were manually inspected before selection.

Excluded tasks are not discarded silently. Each excluded public-training task
that was manually considered gets a row in the task register with the exclusion
reason.

## Task Register Schema

The Phase 0 task register lives at:
`docs/prereg/arc/P0_TASK_REGISTER.csv`.

Required columns:

| column | meaning |
| --- | --- |
| `task_id` | ARC task filename stem. |
| `split` | `training` or `evaluation-blind`. |
| `status` | `include`, `exclude`, or `hold`. |
| `primary_prior` | One of the six registered priors. |
| `secondary_priors` | Semicolon-separated optional priors. |
| `inclusion_basis` | Short human-readable rule description for included tasks. |
| `exclusion_reason` | Required when `status=exclude`. |
| `predicted_boundary` | Non-local, capacity, gauge-breaking, full-state-only, or other named boundary. |
| `inventory_row_hash` | Hash of the metadata row used during selection. |
| `manual_inspection` | `yes` for public-training inspection; `no` for evaluation-blind. |
| `notes` | Free text; no solver results. |

## Baseline Slate

Phase 0 must record baseline numbers before Phase 1 begins. The first slate is:

1. **Random valid grid baseline.** Uses observed training output shapes and
   colors; reports exact-match and pixel-correctness diagnostics.
2. **Identity/copy baseline.** Emits the test input, plus a same-shape color
   remapping attempt when a one-to-one training color map is obvious.
3. **DSL-lite brute baseline.** Enumerates a frozen small set of transforms:
   rotate, reflect, translate/crop/pad, recolor, connected-component copy,
   fill, and compose up to a preregistered depth.
4. **Tiny learned reference.** Optional if it completes under the repo's
   ten-minute rule; otherwise stage the exact PowerShell command and estimate
   instead of running it.

Subset quality guard: if the DSL-lite brute baseline solves more than **70%**
of the development subset, the subset is too easy for the sufficiency question
and must be rebalanced before Phase 1. If all cheap baselines solve **0%**, the
subset may be too hard or misclassified; file a hold review before operator
design.

## Exit Criteria

Phase 0 exits only when all artifacts below exist:

1. `results/arc/phase0-inventory/manifest.json`
2. `results/arc/phase0-inventory/tasks.csv`
3. `docs/prereg/arc/P0_TASK_REGISTER.csv`
4. `docs/prereg/arc/P0_BASELINES.md`
5. An admission note appended to this file stating **ADMIT**, **PARTIAL ADMIT**,
   or **HOLD** for Phase 1.

Until then, the following are forbidden:

- implementing the ARC shadow-projection operator;
- training a signature decoder;
- inspecting public-evaluation task grids manually;
- tuning on public-evaluation scores;
- preparing a Kaggle submission notebook.

## Outcome Branching

| outcome | interpretation | next branch |
| --- | --- | --- |
| Cannot meet minimum subset size | The ARC-AGI-2 public-training subset does not cleanly expose the claimed core-knowledge surface under these criteria. | Retreat to synthetic grid warmups; no Phase 1 ARC operator. |
| Subset exists but cheap baselines solve >70% | The selected surface is too easy / too DSL-native. | Rebalance before operator design. |
| Subset exists but cheap baselines solve 0% | The selected surface may be too hard or misclassified. | HOLD; audit taxonomy before operator design. |
| Subset, blind policy, and baselines are filed | Phase 1 can define the discrete-grid shadow domain. | ADMIT Phase 1. |

---

## Amendments

Append-only. Each amendment must carry a timestamp, author, justification, and
verdict impact.

**2026-05-28 (PT) -- Codex.** Phase 0 inventory/register/baseline receipt
filed at [`P0_BASELINES.md`](P0_BASELINES.md). Inventory result:
`1120` public tasks (`1000` training, `120` evaluation), with public-evaluation
test-output metadata omitted. Register result: `36` included public-training
tasks, balanced `6` per registered prior, no duplicate task IDs, all
`inventory_row_hash` values resolving into the inventory CSV. Baseline result:
`random_valid`, `identity_copy`, and `dsl_lite_v0` all scored `0/36` exact;
diagnostic mean pixel accuracy was `0.1530`, `0.5585`, and `0.5585`
respectively. Verdict impact: **PARTIAL ADMIT -- subset clean, operator design
HOLD**. The subset clears the >70% too-easy guard but fires the zero-floor
caveat, so the next admitted work is a Phase 0 hold review / taxonomy rebalance
or synthetic-grid warmup only. No ARC shadow-projection operator, signature
decoder, or Sundog-specific feature scoring is admitted until a later amendment
records **ADMIT**.
