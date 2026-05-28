# Phase 3B — Compact-Signal Subset (Branch B Diagnostic Lane)

Filed: **2026-05-28 (PT)** · Author: Jeffery Hughes Jr.

Status: **PRE-REGISTERED — RUN NOT YET COMMENCED**

Parent spec: [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)
Companion reflection: [`PHASE3_5_REFLECTION.md`](PHASE3_5_REFLECTION.md) (Branch B)

This artifact pre-registers a **Branch B diagnostic lane** under the
PHASE3_5_REFLECTION framing. The purpose is to find the smallest
modification to the registered Phase 3 V2 lane that gives the existing
deterministic-low-capacity decoder a realistic chance of producing a
non-zero exact-match signal on the full-grid control. If the decoder
can clear a minimal exact-match floor on this narrowed task
distribution, a meaningful arena exists in which to compare
`signature_palette` against the full-grid control. If the decoder
still cannot clear the minimal floor, we exit Branch B and update
PHASE3_5_REFLECTION accordingly.

This is a **diagnostic** lane: it does not adjudicate sufficiency by
itself. It either opens or closes the comparison arena.

## 1. Pre-Registered Compact-Signal Subset

Source: Phase 2 projection-measurement receipt
`results/arc/phase2-projections/task-summary.csv`. The Phase 2
classification is **already filed** (see `PHASE2_PROJECTION_SPEC.md`)
and assigned each of the 36 registered tasks one of three signal
labels: `compact`, `mixed`, `dispersed`. **The 7 tasks labeled
`compact`** are listed below. The subset is taken verbatim from that
Phase 2 receipt with no re-classification.

| task_id | primary_prior | mean_train_pair_residual | signature_collision_residual |
| --- | --- | ---: | ---: |
| `009d5c81` | counting | 0.210032 | 0.000000 |
| `00d62c1b` | symmetry | 0.218116 | 0.000000 |
| `11dc524f` | objectness | 0.104620 | 0.000000 |
| `1acc24af` | objectness | 0.246205 | 0.000000 |
| `1b60fb0c` | objectness | 0.234362 | 0.000000 |
| `2601afb7` | color_role | 0.086294 | 0.142857 |
| `292dd178` | color_role | 0.214508 | 0.000000 |

Prior distribution: counting=1, symmetry=1, objectness=3, color_role=2.
**Not represented in the compact slice**: `spatial_transform`,
`local_completion`. The Branch B floor (§3) cannot speak to those
priors; the subset's discrimination is by construction restricted to
the 4 represented priors.

## 2. Pre-Registered Compact-Subset Split (Adjustment)

All 7 compact tasks are in the `train` role of the original
`EXPECTED_SPLIT` defined in `phase3_decoder.py`. Literal preservation
of "same train/validation/test splits" is therefore impossible — the
parent spec anticipated this case and admitted "Possibly the exact
train/val/test split ratios if the compact subset is too small
(pre-register any adjustment)."

The internal 4 / 1 / 2 split below is pre-registered and frozen for
the Branch B lane. It uses the same per-prior heldout-leaves-one-prior-
intact discipline as the parent: every test task is the sole compact
representative of its prior or the spare from an over-represented
prior, never the only training task for its prior.

| subset_split | task_id | primary_prior | rationale |
| --- | --- | --- | --- |
| `train` | `009d5c81` | counting | sole compact counting task |
| `train` | `00d62c1b` | symmetry | sole compact symmetry task |
| `train` | `1b60fb0c` | objectness | objectness train representative |
| `train` | `292dd178` | color_role | color_role train representative |
| `validation` | `1acc24af` | objectness | objectness held out for val-pool exact-match signal |
| `test` | `11dc524f` | objectness | objectness held out for `test_lodo` / `pttest` |
| `test` | `2601afb7` | color_role | color_role held out for `test_lodo` / `pttest` |

Frozen split file: `docs/prereg/arc/PHASE3B_COMPACT_SPLIT.csv`.
Its SHA-256 will be recorded in the binding receipt's manifest under
`subsetSpecHash`.

LODO structure on the new split (k = train pairs per task; from
`P0_TASK_REGISTER.csv`):

- `train_lodo` lane: 4 tasks × (k-1) instances each, drawn from the 4
  train tasks' k-pair training pools.
- `train_pttest` lane: 4 tasks × 1 instance each (each task's `test`
  pair held as a held-out probe during training).
- `validation_lodo` lane: 1 task × (k-1) instances = 1 task's LODO
  expansion (≈ 3–4 instances).
- `validation_pttest` lane: 1 task × 1 instance.
- `test_lodo` lane: 2 tasks × (k-1) instances each (≈ 4–6 instances).
- `test_pttest` lane (labelled `pttest` in the runner): 2 tasks × 1
  instance each = 2 instances.

The held-out lanes are intentionally small — that is the price of
restricting to 7 tasks. The minimal floor in §3 is calibrated to this
sparsity.

## 3. Pre-Registered Minimal Floor

For Branch B only, the held-out exact-match floor relaxes from the
parent V2 spec (`pttest_exact_tasks ≥ 2 AND test_lodo_exact_tasks ≥ 2`)
to the minimal-non-zero version:

> **`pttest_exact_tasks ≥ 1 AND test_lodo_exact_tasks ≥ 1`** where a
> "success" remains a task with `grid_exact_any_rate > 0.010` on the
> named held-out lane for arm `raw_grid_lowcap`.

Floor rationale: with only 2 test tasks contributing to `pttest` and
≈2 contributing to `test_lodo`, the V2 floor of "≥2" would require
**100% of held-out tasks** to clear the per-task exact-match
threshold. The relaxed minimal floor requires **at least one
exact-match task on each held-out lane**, which is the smallest
integer floor that still demonstrates the decoder lane can learn
something exact on held-out data when given full information.

Gate decisions for Branch B (named differently from the parent V2
gate to avoid namespace collision):

- `compact_full_grid_control_pass` — floor cleared. A signature
  comparison arena exists.
- `compact_full_grid_control_floor` — floor not cleared. The compact
  slice is still too hard for this decoder family; Branch B closes
  with a bounded-failure receipt.

## 4. Branch Criteria (Scoped To This Subset)

| branch | definition | next step |
| --- | --- | --- |
| **A — Clean Structural Zero** | `signature_palette` arm achieves exact-match performance statistically indistinguishable from (or better than) the `raw_grid_lowcap` full-grid control on the compact subset, conditional on the gate passing for the full-grid control. | File compact-Branch-A binding receipt; consider scaling back to the full 36-task spec with the same learner. |
| **B — Named Quarantine** | Full-grid control clears the minimal floor, but `signature_palette` shows a clear, attributable gap (e.g., palette recovery drops while shape recovery stays comparable). The gap must be named (palette / shape / count / boundary). | File compact-Branch-B binding receipt; attribute the gap; revisit PHASE3_5_REFLECTION Branch B narrowed-support claims with the named quarantine as the boundary condition. |
| **C — Bounded Failure** | Full-grid control fails to clear the minimal floor on the compact subset. | File compact-Branch-C bounded-failure receipt; PHASE3_5_REFLECTION Branch B is closed in the deterministic-low-capacity-learner family; surface Branch A (stochastic per-task) or Branch D (different framing) as the only remaining reopen paths. |

Branches A and B require **two** binding receipts (`raw_grid_lowcap`
then `signature_palette`). Branch C closes after the first receipt.
The first receipt is the full-grid control: if it floors, the
signature receipt does not run.

## 5. What Stays Frozen

To keep the comparison clean and consistent with the parent V2
discipline:

- **Decoder architecture**: identical small transformer
  (`d_model=192`, `heads=6`, `layers=4`, `feedforward_dim=768`,
  `dropout=0.1`); `MODEL_SPEC_V2` in `phase3_decoder_v2.py` is the
  pinned spec.
- **Hyperparameters**: AdamW (`lr=2e-4`, `betas=(0.9, 0.95)`,
  `eps=1e-8`, `weight_decay=0.01`), constant LR schedule, batch_size
  24, max_epochs 120, early_stop_patience 20, dropout 0.1.
- **Loss**: `cell_ce + 0.25 * height_ce + 0.25 * width_ce`.
- **Weight init**: xavier_uniform on linear, zero bias, embedding
  normal(0, 0.02), LayerNorm unit. Pinned `MAX_DEMOS = 5` token cap.
- **Seeds**: same slate `[20260528, 20260529, 20260530]`. Primary
  seed for the first compact tranche: **20260529** (matches the V2
  binding-receipt selection). The other two seeds are admitted but
  optional — they run only if the single-seed tranche clears the
  floor OR the operator elects multi-seed verification.
- **Selection rule**: identical to V2 — `-validation_metric,
  +validation_loss, +seed` lexicographic, recorded in
  `validation_candidates.json`.
- **Signature extraction**: `signature_palette` is the registered
  Sundog representation; no re-derivation in Branch B.
- **Metrics, evaluation code, aggregation**: identical to V2; only
  the gate adjudication threshold is overridden (§3).
- **Leak-check + shard-equivalence rules**: identical; the
  pre-commit hook and CI gate apply unchanged. The shard+merge
  protocol is admitted but unused for the single-seed first tranche.
- **Aux pool**: identical to V2 — all ARC public-training tasks
  except (a) the 36 registered Phase 0 tasks and (b) the 29
  `MAX_DEMOS = 5` token-cap exclusions (the same 29 listed in the V2
  manifest). The aux pool is **not narrowed** in Branch B; only the
  held-out task distribution is.

## 6. What Changes

- **Held-out task distribution**: 36 → 7 (the compact-signal slice).
- **Internal split**: pre-registered 4 / 1 / 2 as in §2.
- **Floor**: ≥2 / ≥2 → ≥1 / ≥1 (§3).
- **Gate names**: `compact_full_grid_control_*` rather than
  `full_grid_control_*` to keep the V2 receipt namespace clean.
- **Receipt output directories**:
  `results/arc/phase3-rawgrid-compact-7-shard-<seed>/` per shard;
  `results/arc/phase3-rawgrid-compact-7/` for the (optional) merge.
- **Pre-registration record**: this artifact + `PHASE3B_COMPACT_SPLIT.csv`.
- **Possibly**: re-run the other two seeds for verification (§5).

## 7. Wall-Time Estimate

The compact-7 subset training pool is 4 tasks × ~3–5 LODO instances
≈ 15 instances vs. V2's 24 train-task × ~5 instances ≈ 120
registered-task instances. But the **aux pool is unchanged** at ~971
tasks × ~5 instances each ≈ 4885 aux instances, which dominates the
per-epoch wall. Per-epoch GPU time scales ~linearly with total
instance count, so:

| protocol | wall estimate |
| --- | ---: |
| compact-7 single-seed (`20260529`, GPU) | ~22–25 min (≈ V2 shard) |
| compact-7 single-seed (`20260529`, CPU) | ~3.8 h |
| compact-7 3-seed sharded (GPU) | ~25 min wall, 3× compute |
| compact-7 3-seed sharded (CPU) | ~10 h |

The single-seed GPU tranche is the admitted first-cut posture. The
other two seeds are admitted as optional follow-on verification
(launched the same way V2 was, but only if the first tranche
warrants it).

## 8. Stop / Decision Rule

After the single-seed (`20260529`) tranche completes:

| outcome | decision |
| --- | --- |
| `compact_full_grid_control_floor` (both lanes 0 exact) | File Branch C; close Branch B; surface stochastic per-task (PHASE3_5_REFLECTION Branch A) or different framing (Branch D) as remaining options. **Do not run additional seeds.** **Do not run `signature_palette`.** |
| `compact_full_grid_control_pass` (≥1 on each held-out lane) | Decide whether to run multi-seed verification (3-seed shard+merge with the same selection rule) before launching `signature_palette`. Default: launch `signature_palette` immediately for the comparison arena; multi-seed verification only if the single-seed signal is borderline (e.g., exactly 1 exact match on one lane, 1 on the other, no margin). |
| Partial — one lane clears, the other does not | This is a "weak partial pass". File a verdict amendment naming the asymmetry; do not promote to compact-Branch-A or B yet; consider whether the asymmetric pass warrants the signature comparison or whether it points to a structural floor of its own. |

## 9. Discipline Tooling

The existing pre-commit + CI leak-check (`arc:phase0:leak-check`)
applies unchanged. The `subset-spec` plumbing in
`phase3_decoder_v2.py` (admitted by the parent-spec amendment, §
"Branch B – Compact Subset Path") writes `subsetSpecHash` and
`subsetSplit` into the manifest so receipt freeze-marker discipline
holds across compact subset runs.

## 10. Public-Language Constraint

Permitted (in addition to the parent README list):

- "A compact-signal subset of 7 Phase 2 tasks is pre-registered as a
  Branch B diagnostic lane for Phase 3."
- "The compact-7 lane will either reopen a signature-vs-full-grid
  comparison arena or file a third bounded-failure receipt closing
  Branch B in the deterministic-low-capacity-learner family."

Forbidden:

- "The compact-7 receipt vindicates Sundog" — the lane is a
  diagnostic, not a sufficiency adjudication.
- Any signature-vs-full-grid comparison claim that precedes a
  passing full-grid control on the compact subset.
- Any spatial_transform or local_completion claim from the compact
  receipt — those priors are not represented in the compact slice.

## Append-Only Amendments

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Compact-7 Single-Seed Binding Receipt: `compact_full_grid_control_floor`

The single-seed compact-7 tranche (seed `20260529`, GPU, full 120-epoch
budget) completed under freeze-marker commit
`50EAEBBFA00D146397BEA4C81FD460DEE3DED5D8`. The shard ran clean
(`gitDirty=false`), merged into a binding receipt with the same commit
(`mergeGitDirty=false`, `mergeAllowDirty=false`), and the
shard-equivalence guarantee holds: `scores.csv` and `per_task.csv` are
byte-identical between the merge output and the input shard
(`cmp` exit 0 on both).

Binding receipt: `results/arc/phase3-rawgrid-compact-7/`.
Shard: `results/arc/phase3-rawgrid-compact-7-shard-20260529/`.

**No prior verdict changes**: this is the first binding compact
receipt and it supersedes nothing. The V1 and V2 full-grid-control
receipts remain Branch C bounded failures as filed in the parent
spec.

#### Validation And Wall-Clock

| metric | value |
| --- | ---: |
| best_epoch | 55 |
| validation_loss | `0.7627738118171692` |
| validation_metric | `0.0` |
| elapsed_seconds | `2272.83` (37.9 min) |

`best_epoch=55` is well above V2's range (12–15) and the val_loss
drops to less than half of V2's (`0.7628` vs V2's `1.6249`); the
decoder kept improving for longer on the compact distribution and
reached a much tighter loss floor. The single-seed selection rule
trivially returns seed `20260529`.

#### Gate Adjudication

Pre-registered minimal floor (§3): `pttest_exact_tasks >= 1 AND
test_lodo_exact_tasks >= 1` for arm `raw_grid_lowcap`.

Observed (selected seed `20260529`, from
`results/arc/phase3-rawgrid-compact-7/scores.csv`):

| lane | instance_count | grid_exact_any | shape_exact_slot1 | palette_exact_slot1 | pixel_best_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pttest` | 2 | `0.000` | `1.000` | `0.000` | `0.877` |
| `test_lodo` | 6 | `0.000` | `1.000` | `0.000` | `0.860` |
| `validation_lodo` | 4 | `0.000` | `1.000` | `0.000` | `0.788` |
| `validation_pttest` | 1 | `0.000` | `1.000` | `0.000` | `0.757` |

`pttest_exact_tasks = 0`, `test_lodo_exact_tasks = 0`.
**Gate decision: `compact_full_grid_control_floor`**.

Manifest: `gateDecision = {"gate":
"compact_full_grid_control_floor", "pttest_exact_tasks": 0,
"test_lodo_exact_tasks": 0, "reason": "raw_grid_lowcap did not clear
the compact-subset minimal floor on both held-out lanes"}`.
Adjudication narrative:
`results/arc/phase3-rawgrid-compact-7/branch_adjudication.md`.

#### Named Failure Mode: Dominant-Color Mode Collapse

The compact-7 failure is **qualitatively distinct** from V1 / V2 and
deserves an explicit name:

- **`shape_exact_slot1 = 1.000` on every lane** — the decoder
  predicts the correct output dimensions for every held-out
  instance.
- **`palette_exact_slot1 = 0.000` on every lane** — the decoder
  never produces the correct palette set on slot 1.
- **`pixel_best_mean` in the 0.76–0.95 band** — much higher than
  V2's 0.23–0.34, but obtained by predicting the dominant
  background color almost everywhere.

Direct per-instance audit of all 13 held-out predictions in
`results/arc/phase3-rawgrid-compact-7/residuals.jsonl`:

| task | target colors (count) | slot-1 predicted colors (count) | dominant-color collapse? |
| --- | --- | --- | --- |
| `pttest:11dc524f:0` | `{7,5,2}` (3) | `{7}` (1) | yes — full collapse to background `7` |
| `pttest:2601afb7:0` | `{0,2,6,7,8,9}` (6) | `{7}` (1) | yes — full collapse to background `7` |
| `test_lodo:11dc524f:{0,1,2}` | `{7,5,2}` (3) | `{7}` (1) on all 3 | yes (×3) |
| `test_lodo:2601afb7:{0,1,2}` | various 6-color sets | `{7}` / `{5,7}` / `{2,7}` | yes — at most 2 colors out of 6 |
| `validation_lodo:1acc24af:{0..3}` | task-specific | `{0,1}` ×3, `{0}` ×1 | yes — at most 2 colors |
| `validation_pttest:1acc24af:0` | task-specific | `{0,1}` | yes — at most 2 colors |

Pattern: 13 of 13 held-out instances use **at most 2 colors** in the
slot-1 prediction, regardless of target palette size, and every
shape is correct. The cross-entropy loss favors the dominant-color
solution because background pixels swamp object pixels, and the
decoder has learned the marginal cell distribution rather than any
multi-color structural rule.

The name **"dominant-color mode collapse"** is filed for this
failure mode. It is structurally distinct from:

- V1 / V2's character on the 36-task receipts ("decoder fails to
  learn anything coherent"; shape ≈ 0.5–0.7, pixel ≈ 0.25–0.34,
  palette ≈ 0).
- A capacity-pressure failure (where the decoder would produce
  noisy multi-color output).
- A gauge-permutation failure (where palette set would match but
  colors would be permuted).

The compact-7 decoder is **more confident, more degenerate** than
the V2 decoder.

#### Branch Closure (Per §8 Stop Rule)

The pre-registered stop rule at §8 reads:

> `compact_full_grid_control_floor` (both lanes 0 exact) → File
> Branch C; close Branch B; surface stochastic per-task
> (PHASE3_5_REFLECTION Branch A) or different framing (Branch D) as
> remaining options. **Do not run additional seeds.** **Do not run
> `signature_palette`.**

Applied: Branch B (compact-subset diagnostic) **closes** as a
bounded failure in the deterministic-low-capacity-learner family.
The other two seeds (`20260528`, `20260530`) and the
`signature_palette` arm both stand down. No new shard or merge run
is admitted in this lane.

#### What This Verdict Does And Does Not Entail

**It does**:

- Confirm that narrowing the task distribution to the Phase 2
  compact-signal slice does **not** open a comparison arena for
  signature representations: the full-grid control still floors at
  zero exact matches.
- Name the qualitatively different failure mode the compact lane
  exhibits (dominant-color mode collapse) and distinguish it from
  the V1 / V2 failure character.
- Provide a direct, per-instance audit of the failure mode
  (residuals.jsonl) for any future learner that wishes to claim it
  has fixed the mode-collapse problem on this exact subset.

**It does not**:

- Adjudicate the shadow-projection sufficiency hypothesis. Three
  full-grid control receipts (V1, V2, compact-7) now agree on the
  floor; no comparison against `signature_palette` is licensed
  until a different family of learner clears it.
- Support any Branch A or Branch B narrowed-support claim from
  PHASE3_5_REFLECTION at the receipt level.
- Speak to the `spatial_transform` or `local_completion` priors —
  those are not represented in the compact-signal slice (§1).
- Generalise to a "compact tasks are unsolvable" claim. The
  deterministic-low-capacity decoder family fails on this slice
  with this failure mode; a stochastic per-task learner (Branch A)
  or a different framing (Branch D) might or might not pass — that
  is the next question, not a settled one.

#### Public-Language Update (Additive)

Permitted:

- "The Phase 3 Branch B compact-subset diagnostic lane filed a
  `compact_full_grid_control_floor` receipt: zero exact matches on
  the 7-task compact slice, with the decoder collapsing to the
  dominant background color on every held-out instance
  (dominant-color mode collapse)."
- "All three filed Phase 3 full-grid controls (V1, V2, compact-7)
  now floor at zero exact matches; the comparison arena for
  signature representations remains closed in the
  deterministic-low-capacity-learner family."

Forbidden:

- "Compact tasks are unsolvable" — the failure is of this learner
  family on this slice, not of the tasks.
- Any narrowed-support sufficiency claim from the compact receipt.
- Any signature-vs-full-grid comparison claim from the compact
  receipt.
- Any spatial_transform or local_completion claim from the compact
  receipt (those priors are absent from the slice).

#### Frozen By This Verdict

- The compact-7 binding receipt, manifest, hashes, and residuals
  are frozen at the paths above.
- The compact-7 subset spec
  (`docs/prereg/arc/PHASE3B_COMPACT_SPLIT.csv`, sha256
  `5759038A94DB...`) is frozen as the Branch B subset definition.
- The named failure mode "dominant-color mode collapse" is filed
  here as the per-instance characterisation of the compact-7
  bounded failure.
