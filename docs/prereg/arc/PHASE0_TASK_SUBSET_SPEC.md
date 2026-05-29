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

**2026-05-28 (PT) -- Claude (Opus 4.7).** Phase 0 baseline expansion +
verdict upgrade. Following the prior PARTIAL ADMIT amendment's preregistered
"Allowed next work" path ("Improve or add non-Sundog cheap baselines only by
append-only amendment before looking at any Sundog operator result"), three
baselines were added to `scripts/arc-phase0-baselines.mjs`, keeping
`random_valid`, `identity_copy`, and `dsl_lite_v0` byte-frozen:

- `dsl_lite_v1`: `tile`, `translate`, `palette_permute` (depth 1).
- `dsl_lite_v2`: `pad`, `fill_enclosed`, `component_copy_largest`, plus
  depth-2 composition over the union of v0, v1, and v2 structural transforms
  with one final color-map fit per composed candidate.
- `tiny_learned_v0`: per-task nearest-neighbour over train pairs by padded
  pixel Hamming distance.

Combined primitive coverage now exhausts the spec Baseline Slate item 3
DSL-lite primitive list (rotate, reflect, translate/crop/pad, recolor
(palette permute), connected-component copy, fill) at composition depths 1
and 2. Optional tiny learned reference is included per Baseline Slate item 4.

Result on the registered 36-task subset: all five non-random baselines score
`0/36` exact (mean pixel accuracy `0.5585` for v0/v1/v2/identity fallback;
`0.3993` for tiny_learned). Implementation correctness verified by synthetic
sanity check at `tests/arc-baselines/out/summary.csv` (v1 solves a 2x2 tile
task; v2 solves both a 2x2 tile and a pad-with-zero border task), confirming
the registered-subset zero floor is a property of the subset, not a fitter
bug.

Verdict impact: **ADMIT** (Phase 1 -- ARC grid representation as shadow
domain -- admitted). The preregistered "Subset, blind policy, and baselines
are filed" outcome row from the Outcome Branching table now governs, with
the zero-floor caveat resolved by exhaustive baseline coverage rather than
by a nonzero solve. Phase 1 inherits `0/36` exact as the hard preregistered
floor that any Sundog-specific result must clear.

Public-evaluation discipline (no manual grid inspection, no Kaggle prep)
remains in force until Phase 6.

**2026-05-28 (PT) -- Claude (Opus 4.7).** Discipline hardening (no verdict
change). Following the ADMIT amendment above, the operational tooling that
enforces the public-evaluation and Kaggle restrictions was hardened so that
the discipline rules are checked by code, not only documented:

- `scripts/arc-phase0-inventory.mjs` now requires *two* flags to emit
  evaluation test outputs: `--include-evaluation-test-output` AND
  `--authorize-evaluation-leak`. The `--out` path must end in
  `_PRIVILEGED_AUDIT`; the manifest emits `privilegedAudit: true` and a
  banner `evaluationPolicy`. This formalises the "evaluation-output
  override for a final post-freeze audit" provision in the Data Source and
  Split Policy section above into a runtime gate.
- `scripts/arc-phase0-leak-check.mjs` (new) audits five invariants on every
  invocation: default inventory manifest is non-privileged; register has no
  `split=evaluation` rows and every `split=evaluation-blind` row carries
  `manual_inspection=no`; baseline predictions are a subset of registered
  training task IDs; no Kaggle scaffolding (kaggle.json, non-empty .ipynb)
  exists under `scripts/`, `docs/`, `tests/`, `notebooks/`; non-allowlisted
  ARC scripts (anything other than the inventory and leak-check themselves)
  contain no `evaluation` literal.
- `.githooks/pre-commit` runs the leak check before every commit, installed
  by `scripts/setup-githooks.mjs` via the npm `prepare` script.
- `.github/workflows/arc-discipline.yml` runs the leak check on push and PR
  for changes touching ARC paths.
- [`EVAL_BLIND_SELECTION.md`](EVAL_BLIND_SELECTION.md) is filed as a stub
  pattern for the first Phase 1+ amendment that adds evaluation-blind
  register rows, so the rule shape is preregistered before any concrete
  selection.

Verdict impact: none. The ADMIT verdict above continues to govern. This
amendment is a tooling tightening, not a scope or floor change. Future
amendments may add or modify these tools, but the verdict logic and
preregistered constraints in the body above remain unchanged.

**2026-05-29 (PT) -- Codex.** Context-expansion-for-fibers spec filed:
[`PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md`](PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md).
This is an append-only Phase 0 register-amendment path motivated by the Phase
3E v2 receipt `phase3e_v2_deferred_sparse_fibers`: the v2 oracle defect is
repaired, but the 36-task registered context universe has zero cross-task
neighbors within `epsilon_primary = 0.05` and minimum cross-task distance
`0.207`. The new spec freezes a balanced expansion target of 108 included
public-training tasks (18 per prior), mechanical pre-inspection candidate
ordering, no use of Phase 3E distances/sketches/solver outputs during
selection, and unchanged Phase 3E v2 oracle/geometry thresholds.

Verdict impact: **no register change and no execution admission**. The
original 36-task Phase 0 register remains binding. A future expansion receipt
must file `P0_CONTEXT_EXPANSION_REGISTER.csv`,
`P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv`, audit artifacts, leak-check
coverage, and a freeze-marker amendment before any expanded Phase 3E v2
certificate rerun is admitted.

**2026-05-29 (PT) -- Jeffery Hughes Jr.** Phase 0 fiber context expansion
admitted and expanded certificate filed. The expansion was executed under the
spec above:

- commit `f9a1dd9`: freeze marker, candidate-ordering script, `--split-mode`,
  inspection renderer, Amendment A;
- commit `a0f1a4b`: inspection + register assembly,
  **`phase0_fiber_expansion_admit`**;
- commit `fa079b9`: expanded Phase 3E v2 certificate,
  **`phase3e_v2_expanded_oracle_regression`**.

The binding expanded register contains 108 public-training tasks, exactly 18 per
prior (36 original + 72 new), selected from the frozen queue with 0
hard-exclusions, 0 discipline tripwires, and no cross-prior rebalancing. The
expanded certificate did not densify the fibers enough to adjudicate locality:
`U_primary` increased 25 -> 336, but min cross-task distance moved only
0.207 -> 0.196, with 0 near pairs and 0% fidelity at `epsilon_primary = 0.05`.

The certificate branch is nevertheless `phase3e_v2_expanded_oracle_regression`
because the anti-solver-leakage gate fails on the expanded register:
`core_sketch_exact_lookup_fraction = 0.351 > 0.20`. Anti-vacuity and
anti-prior-laundering still pass. Verdict impact: the expanded register is
filed, but it does not promote or block signature sufficiency or Branch E. The
36-task Phase 3E v2 receipt and the seven full-grid floors stand unchanged.
