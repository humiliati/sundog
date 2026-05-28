# Evaluation-Blind Selection Pattern (Stub)

Roadmap: [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)
Phase 0 spec: [`PHASE0_TASK_SUBSET_SPEC.md`](PHASE0_TASK_SUBSET_SPEC.md)
Baselines receipt: [`P0_BASELINES.md`](P0_BASELINES.md)

Status: **STUB**. No evaluation-blind rows are admitted yet. This document
exists so the first Phase 1+ amendment that adds them has a frozen pattern
to instantiate, rather than inventing one ad hoc.

## When this pattern is used

Phase 0 registered subset selection drew only from the public training
split, with full manual inspection allowed. Public evaluation tasks were
*not* selected from, and their test outputs were omitted from every emitted
artifact.

When a later phase needs to declare *which* evaluation tasks a Sundog
operator or signature decoder will be scored against -- so that the result
is preregistered rather than picked after the fact -- those tasks enter the
register as `split=evaluation-blind` rows. They are never opened, displayed,
or otherwise inspected before scoring.

## Hard rules

1. **No manual inspection of evaluation grids.** Every `evaluation-blind`
   row must carry `manual_inspection=no`. The leak-check
   (`npm run arc:phase0:leak-check`) fails the build if this is violated.

2. **Selection by preregistered metadata/hash rule only.** The inclusion
   criterion must be expressible as a deterministic predicate over the
   inventory-row metadata in
   `results/arc/phase0-inventory/tasks.csv` (or its `_PRIVILEGED_AUDIT`
   counterpart). Examples of admissible predicates:
   - `task_id` matches a literal allowlist whose membership rule is itself
     metadata-derived;
   - `inventory_row_hash` falls inside a preregistered hash window;
   - structural metadata satisfies a Boolean predicate (e.g.,
     `symmetry_hints != "" AND max_nonzero_components <= 6`).

3. **No look-then-decide.** The rule must be filed *before* the evaluation
   inventory metadata is even read against the predicate. Once filed, the
   rows that fall out of the predicate are the registered set; no
   post-filter is allowed.

4. **No baseline runs on evaluation-blind rows.** The baseline runner is
   hardcoded to load from `<data-dir>/training/` only. Evaluation-blind
   rows are reserved for the Sundog-specific operator/decoder evaluation in
   Phase 3+ and the falsification battery in Phase 5.

## Register-row template

| column | value for evaluation-blind |
| --- | --- |
| `task_id` | resolved from the predicate against the inventory |
| `split` | `evaluation-blind` |
| `status` | `include` (or `hold` while the predicate is under review) |
| `primary_prior` | preregistered prior the selection is aimed at |
| `secondary_priors` | preregistered, derived from inventory metadata only |
| `inclusion_basis` | the exact predicate text, machine-readable preferred |
| `exclusion_reason` | empty |
| `predicted_boundary` | preregistered, derived from inventory metadata only |
| `inventory_row_hash` | from `results/arc/phase0-inventory/tasks.csv` |
| `manual_inspection` | **must be `no`** |
| `notes` | "evaluation-blind; not inspected; selected by `<rule-name>`" |

## Selection-rule discipline

The recommended pattern for adding evaluation-blind rows:

1. File a new doc `docs/prereg/arc/PHASE<n>_EVAL_BLIND_SELECTION.md` that
   states the predicate in plain language, the expected resulting count,
   and the audit hash (sha256) of the inventory metadata bytes used.
2. Add a helper script `scripts/arc-phase<n>-eval-blind-select.mjs` that
   reads the inventory, applies the predicate, and emits the proposed rows
   to `docs/prereg/arc/P<n>_EVAL_BLIND_REGISTER.draft.csv` for review.
3. After human review (without opening eval task files), the draft rows
   are merged into the binding `P0_TASK_REGISTER.csv` with
   `manual_inspection=no`.
4. Re-run `npm run arc:phase0:leak-check`. It must pass with the new
   rows reported in the `evaluation-blind` count.

## What violates the pattern

Examples that the leak-check will catch:

- Adding a row with `split=evaluation` (anything other than `training` or
  `evaluation-blind` fails).
- Adding an `evaluation-blind` row with `manual_inspection=yes` (fails).
- Running the baseline runner against an evaluation-blind row: refused at
  runtime because the runner filters to `split=training`.
- Emitting evaluation test-output metadata into the default inventory
  output path: refused at runtime by the inventory's double-flag override
  (`--include-evaluation-test-output` requires `--authorize-evaluation-leak`,
  and the output path must end in `_PRIVILEGED_AUDIT`).

Examples that the leak-check does *not* catch (discipline only):

- Opening an evaluation JSON in an editor.
- Sharing the contents of an evaluation grid in chat or notes.
- Reading the official ARC-AGI-2 GitHub `evaluation/` folder via the web
  rather than the local clone.

Those remain procedural rules; no tooling enforces them.
