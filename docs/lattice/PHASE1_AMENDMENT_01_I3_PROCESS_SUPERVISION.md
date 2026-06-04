# Phase-1 Amendment 01 — `[I3′]` Process Supervision (CP-imitation recipe)

> 2026-06-03. **PRE-REGISTRATION ONLY — no v3 run until owner-approved.** Amends the
> Phase-1 build-gate **training recipe** in response to the `build_gate_partial` verdict
> ([`receipts/2026-06-03_BUILD_GATE_V1V2.md`](receipts/2026-06-03_BUILD_GATE_V1V2.md)).
> Scope: the `[I3]` training objective (and a minor aligned `[I4′]`). **The I5 rollout
> contract, caps, thresholds, model architecture, data regime, and the gate command +
> threshold are UNCHANGED** — so the v3 verdict stays directly comparable to v1/v2. Per
> [`PHASE1_I5_ROLLOUT_CONTRACT.md`](PHASE1_I5_ROLLOUT_CONTRACT.md): a build-gate miss is
> resolved by amendment, not in-place tuning. This is that amendment.

## 0. The finding this answers

v1 (flat lr) was optimization-unstable; v2 (lr warmup+cosine decay) trained **stably**
(loss → 1.2e-5) but hit a **generalization ceiling**: it fit the 50k augmented pool yet
stayed **unsound on held-out test** — `false_elim 18.78`/puzzle, binding
`rollout_exact 0.324` (= one-shot; the rollout's search budget went unused because the
model dead-ends at the root). Root cause: **`[I3]` supervises the *endpoint*, not the
*process*.** Per-cell BCE toward the *solution one-hot* trains "keep only the solution
digit, eliminate everything else in one shot" — which rewards predicting answers
(memorizable) and gives no signal for *sound* iterative narrowing.

## 1. Thesis — supervise the sound deduction *process*, not the solution *endpoint*

Train the LDT to **imitate a classical constraint-propagation (CP) solver's sound
narrowing** over the lattice states it actually encounters. The five candidate ideas
collapse into one design:

| idea | role |
| --- | --- |
| classical enumerator → imitate sound narrowing | **the spine** (`[I6]` data-gen) |
| train on intermediate lattice states | run the enumerator *with search*; log every node's `(state → narrowing)` |
| deduction-step target (replace solution one-hot) | per-`(cell,digit)` label = "soundly eliminable from *this* state" |
| supervise only sound eliminations | the CP target is sound by definition |
| false-elimination penalty / mask | falls out free (true digit always *keep*); held in reserve |

**The fix is by construction (§2.2):** a sound CP step eliminates candidate `d` from cell
`c` *only* when `d` is provably impossible there — so the **true solution digit is never
an elimination target**. The model is never trained to eliminate a true digit →
`false_elim → 0`, and what it learns is the *narrowing algorithm*, which generalizes.

## 2. The amended objective `[I3′]`

### 2.1 What changes (and only this)
- **Old `[I3]`:** `elim_head` keep-probability supervised toward `one_hot(solution)` —
  keep the solution digit, eliminate all others, from the **initial puzzle** state.
- **New `[I3′]`:** `elim_head` keep-probability supervised toward the **sound candidate
  set**, on **intermediate lattice states**. Per `(cell c, digit d)` that is *currently a
  candidate* in input state `L`:
  - target `keep[c,d] = 1` if `d` survives sound CP to fixpoint from `L` (a sound
    candidate to keep — **includes the true digit and any not-yet-decidable digit**);
  - target `keep[c,d] = 0` iff sound CP eliminates `d` from `c` (provably impossible).
  Loss is **masked to currently-candidate positions** (the model decides keep/eliminate
  among live candidates only).

### 2.2 Soundness invariant (the load-bearing property)
For every logged state, `keep[c, solution[c]] = 1` — CP is sound, so it never eliminates
the true digit. Therefore the objective **cannot reward false elimination**. Verified in
§6 as a hard gate (answer-key used for *audit only*, never for the target — §3.3).

## 3. Data generation `[I6]` — CP + DFS trajectory dataset

### 3.1 The classical solver (answer-key-free)
A deterministic Sudoku reasoner using **only sound rules**: peer-elimination (remove `d`
from `c` when a peer of `c` is solved to `d`), **naked singles** (a cell with one
candidate is placed), **hidden singles** (a digit with one legal position in a unit is
placed); run to fixpoint. When CP stalls (Sudoku-Extreme puzzles are chosen to stall
basic CP), **DFS**: branch on the most-constrained unsolved cell (ties by digit order),
recurse, **backtrack on contradiction**. This is a *complete* solver (DFS is complete),
so basic CP + search suffices to reach the gate — no advanced human techniques needed.
**It reads only the puzzle, never the solution.**

### 3.2 State logging (the mid-search coverage v2 lacked)
At **every search node**, log `(L_t, fixpoint_narrowing(L_t))` where `L_t` is the
candidate-set lattice at that node and the target is CP-to-fixpoint from `L_t`. This
covers the **initial puzzle AND every mid-DFS hypothesis state** — exactly the states the
I5 rollout queries the model on (the rollout calls the model once per node, expects
fixpoint narrowing, then branches). The 1000 training puzzles → many `(state, narrowing)`
pairs each. Symmetry augmentation (the v2 `aug_factor` family: digit-permute + band /
within-band row+col permute + transpose) is applied to the **puzzles before trajectory
generation**, so augmented states are covered too — but the recipe's power now comes from
the **sound-process target**, not the augmentation (the model learns the position/label-
equivariant narrowing *function*, which generalizes off the base puzzles).

### 3.3 Leakage guard (forecloses "secretly solved")
The narrowing target is **pure CP from the puzzle — no answer key.** The solution is used
*only* for (a) build-gate scoring (the I5 rollout's answer-key audit, unchanged) and (b)
the §6 soundness audit. No solution hash / residual / one-hot enters the target or the
features. This closes the PROMOTE_GATE cross-decode / `LATTICE-REIMPL-LAUNDERING` failure
mode at the data-gen layer: the model imitates genuine deduction, not an answer-informed
shortcut.

## 4. The loss (deep supervision)

- **Per-`(cell,digit)` BCE** of `elim_head` keep-logit against `keep[c,d]` (§2.1), masked
  to currently-candidate positions.
- **Deep supervision:** apply `elim_head` + this loss at **each of the 16 recurrence
  iterations**, every iteration targeting the **sound fixpoint** from the input state.
  The model learns to *converge* to the sound narrowing across its iterations (early
  iterations approximate, later refine) — exercising the weight-shared recurrence `[I1]`
  as intended. *(Registered fallback if convergence is poor: a pass-curriculum, iter-`t`
  → CP state after `t` passes. Primary = all-iters-→-fixpoint.)*
- **`[I4′]` conflict head:** target = 1 on states where CP narrows some cell to **empty**
  (a real contradiction, which the DFS naturally produces at wrong branches). This
  replaces the synthetic corrupt-to-⊥ of `[I4]` with the *actual* CP contradiction signal
  the rollout needs to detect dead-ends — an aligned consequence of the same trajectory
  data, not an independent change.
- **False-elim penalty (reserve):** none by default — the sound target already forbids
  false elimination. If v3 still shows `false_elim`, add an asymmetric weight on
  false-positive eliminations of the true digit (a v3.1 micro-amendment, pre-registered
  then).
- **lr schedule:** retained from v2 (warmup 1000 + cosine decay to 0) — the optimization
  fix is kept; only the objective changes.

## 5. What stays FROZEN (the amendment boundary)

`[I1]` weight-shared 16-iter recurrence · `[I2]` input re-injection · the model
(798,346 params, d=128, 4 layers/heads, FFN×4, CLS head) · `[I5]` rollout **verbatim**
([`PHASE1_I5_ROLLOUT_CONTRACT.md`](PHASE1_I5_ROLLOUT_CONTRACT.md)) · caps
`maxSearchNodesPerPuzzle=4096`, `maxDeductionStepsPerNode=64` · `θ_drop=0.5`, `θ_cls=0.6`
· data regime (Sudoku-Extreme `n_train=1000`) · **the gate command + threshold**
(`rollout_exact_rate ≥ 0.999`). The amendment changes the *training target and its data*,
nothing the verdict is read through.

## 6. Gate + audits (UNCHANGED verdict, new soundness check)

- **Build-gate verdict — identical command:**
  `--mode build-gate --stage eval --resume latest --max-eval 1000` (full frozen caps).
  `build_gate_pass` iff `rollout_exact_rate ≥ 0.999`; else `build_gate_partial` / `_fail`.
- **CP-target soundness audit (new, hard):** over all logged states,
  `keep[c, solution[c]] == 1` for every solved-cell target — i.e. the target *never* marks
  a true digit eliminable. A single violation aborts the run (a CP-solver bug, not a
  result). Answer-key used here only.

## 7. Pre-registered predictions + falsifiers (interpret v3 honestly)

- **Success:** `rollout_exact_rate ≥ 0.999` → `build_gate_pass` → R0/R1 precondition met,
  **B2 unblocks**. Expected diagnostic: `false_elim → ~0` (vs v2's 18.78); the rollout
  begins **using** its search budget (`avg_nodes ≫ 1`).
- **Falsifier A — `false_elim` still high:** the model cannot imitate sound CP from the
  lattice → an **architecture/capacity** wall, not an objective one → v4 = the paper's
  exact architecture / a larger model. *(Not "tune the recipe again.")*
- **Falsifier B — sound but `rollout_exact < 0.999`:** the model imitates CP yet the
  rollout misses the bar → a CP+DFS **coverage** gap (audit the branch policy / raise CP
  power), not a soundness failure.
- **Falsifier C — low train loss, unsound on test:** still memorizing → the trajectory
  **state coverage** is too narrow (widen mid-search sampling); records that imitation
  alone did not generalize.
- Honest ceiling: a model trained to imitate CP **is** a learned CP-like reasoner — that
  is the point. The build-gate only asks for a sound solver; whether the *learned
  representation* realizes certified regime-2 is the **B2/B1/B3** question, gated behind
  this pass and measured there — not claimed here.

## 8. Implementation plan (post-approval, before any binding run)

1. **CP+DFS solver module** (`scripts/lib/` or inline): sound rules + search +
   per-node state logging; deterministic.
2. **Trajectory dataset builder**: `(L_t, fixpoint_narrowing(L_t), conflict_flag)` over
   the 1000 (+ augmented) puzzles, on-device tensors (mirror the v2 vectorized pool).
3. **`[I3′]` `compute_loss`** + per-iteration `elim_head` in the forward (deep
   supervision); `--recipe {solution_onehot|cp_imitation}` flag — **default
   `cp_imitation`** for v3, `solution_onehot` retained for v1/v2 reproducibility.
4. **Smokes (Ten-Minute Rule):** CP-solver soundness on N puzzles (true digit never
   eliminated; solves the known puzzles); trajectory determinism; deep-supervision loss
   decreases on a tiny set; `--mode smoke` still green; **the eval command byte-identical**
   to v2.
5. **Freeze-marker commit** (this amendment + the impl), then the pre-registered v3 run
   (H100 per [`BUILD_GATE_REMOTE_RUNBOOK.md`](BUILD_GATE_REMOTE_RUNBOOK.md)) → binding
   eval → receipt `receipts/2026-06-03+_build_gate_v3.md`.

## 9. The pre-registered v3 commands

```bash
# TRAIN (new recipe; lr-decay retained from v2). Step budget revisited at impl
# (CP-imitation may converge faster; pre-register the exact number in the freeze commit).
python scripts/lattice_ldt_model.py --mode build-gate --stage train \
  --recipe cp_imitation --data-dir docs/lattice/Soduko-Extreme --out $OUT --seed 0 \
  --aug-factor 50 --compile --warmup-steps 1000 --max-steps <pre-registered> \
  --batch 64 --checkpoint-every 1000 --log-every 200 --eval-every 1000 --max-eval 1000

# EVAL = the gate, UNCHANGED from v1/v2:
python scripts/lattice_ldt_model.py --mode build-gate --stage eval \
  --data-dir docs/lattice/Soduko-Extreme --out $OUT --resume latest --max-eval 1000
```

## 10. Cross-references

- [`receipts/2026-06-03_BUILD_GATE_V1V2.md`](receipts/2026-06-03_BUILD_GATE_V1V2.md) — the
  `build_gate_partial` finding this answers.
- [`PHASE1_I5_ROLLOUT_CONTRACT.md`](PHASE1_I5_ROLLOUT_CONTRACT.md) — the frozen rollout/caps
  this amendment does **not** touch.
- [`PROMOTE_GATE.md`](PROMOTE_GATE.md) — `build_gate_pass` is the R1 precondition; the
  cross-decode / Target-A fences the §3.3 leakage guard serves.
- [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md) — Phase-1 status; B2/B1/B3 gating.
