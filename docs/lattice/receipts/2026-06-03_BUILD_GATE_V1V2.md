# Build-Gate Receipt — LDT Sudoku-Extreme reimplementation (v1 + v2)

> 2026-06-03. R0-precondition receipt. **Verdict: `build_gate_partial` — NOT
> `build_gate_pass`.** Per [`../PROMOTE_GATE.md`](../PROMOTE_GATE.md) §2 (R1), no
> B-layer number (B2 twin-state / B1 `d_dec` / B3 soundness-frontier) may be read off
> this model. The lane stays **pre-R0 on every substrate leg**; this receipt records
> only the build-gate measurement and the recipe finding it produced.

## 1. Verdict (binding eval)

| field | value |
| --- | --- |
| `branch` | **`build_gate_partial`** (pass requires `rollout_exact_rate ≥ 0.999`) |
| `rollout_exact_rate` | **0.324** (324 / 1000 test puzzles) |
| `one_shot_exact_rate` | **0.324** — identical to rollout |
| `avg_nodes_expanded` | 1.03 · `node_cap_fraction` 0.0 |
| `false_elimination_rate_answer_key_audit` | **18.78 / puzzle** |
| `unsolved_conflict_exhausted` | 676 / 1000 |

**The I5 rollout adds nothing over one-shot** (`one_shot == rollout`, `avg_nodes ≈ 1`,
search budget untouched): the trained model is **unsound** — it eliminates ~19 true
solution digits per puzzle, so 676/1000 puzzles dead-end immediately and no amount of
the frozen 4096-node / 64-step search budget can recover an eliminated true digit.

## 2. What ran (provenance)

- **Model:** `LatticeDeductionTransformer`, **798,346 params** — d=128, 4 layers,
  4 heads, FFN×4, **L=16 weight-shared recurrence** ([I1], param-budget-forced),
  input re-injected per iteration [I2], learned 2D positional, CLS conflict head
  (θ_cls=0.6, λ_cls=0.1).
- **Data:** Sudoku-Extreme, `n_train=1000` (HRM/TRM small-data regime), public
  head-slices (train 100k rows / test 10k rows — result-identical to the full 718 MB
  for this config); symmetry-augmented pool **aug_factor=50 → 50,000** (digit-permute
  + band / within-band row+col permute + transpose).
- **Recipe inferences:** [I3] per-cell BCE → solution one-hot; [I4] conflict target
  via corrupt-to-⊥; [I5] rollout per [`../PHASE1_I5_ROLLOUT_CONTRACT.md`](../PHASE1_I5_ROLLOUT_CONTRACT.md).
- **Binding eval caps (frozen, manifest-confirmed):** `maxSearchNodesPerPuzzle=4096`,
  `maxDeductionStepsPerNode=64`, `thetaDrop=0.5`, `thetaCls=0.6`, `max-eval=1000`.
- **Env:** Hyperbolic **H100 80GB (CC 9.0)**, torch `2.5.1+cu121`, `--compile`
  (reduce-overhead / CUDA graphs), ~37 ms/step.
- **Git provenance:** box `gitCommit 345CF311C330572CDEF945621A90907013276B77`
  ( = `e3be47c4` + lr warmup/cosine-decay patch ), `gitDirty false`; runner
  `scripts/lattice_ldt_model.py` **sha256 `599838069b53a0c6a995c5390efca14851381c49c8b52e1486da1a2022ca7d54`**;
  seed 0.

## 3. The v1 → v2 iteration (two findings, in order)

**v1 — flat `lr=1e-3` → OPTIMIZATION-unstable.** The model oscillated: diag one-shot
climbed to ~0.28 then **crashed to 0.0** (step 66–68k) and partly recovered; `false_elim`
bounced 16–26/puzzle; `conflict_loss` spiked. Best ≈ `false_elim 16.9` @ step 35k, then
*regressed*. Constant high lr kept jumping the model out of good basins → could not reach
the gate.

**v2 — lr warmup (1000) + cosine decay to 0 over 100k → STABLE.** Loss converged
smoothly to `1.2e-5`, lr → 0, no crashes. **The lr schedule fixed v1's instability.**
But `false_elim` plateaued at **~16–19 on the held-out test set**; final binding
`rollout_exact 0.324`.

→ lr-decay solved the *optimization* problem and thereby **revealed the real wall:
generalization.**

## 4. Diagnosis — a generalization ceiling, not an optimization one

The model fits the augmented training pool to loss `~1e-5` yet stays unsound on
held-out test (`false_elim ~19`). The likely mechanism, in two parts:

1. **Symmetry augmentation adds no logical diversity.** Digit-permute / band / transpose
   produce puzzles that are **logically isomorphic** to the 1000 base puzzles — they
   impose label/position invariance but introduce **no new deduction structure**. The
   model can memorize the 1000 logical structures (→ train loss ≈ 0) without learning
   the general algorithm → unsound on different test puzzles.
2. **The `[I3]` objective likely permits this.** Per-cell BCE on the *final solution*
   one-hot rewards predicting answers, not performing iterative deduction. HRM / TRM /
   LDT-class models on 1K Sudoku-Extreme typically use **deep supervision** (loss at
   each reasoning iteration) and/or **ACT halting** to force algorithm-learning — not
   captured by the `[I3]` inference.

## 5. Next hypothesis (v3 — gated on owner decision, NOT auto-pursued)

Revisit **`[I3]`**: apply `elim_head` + loss at **each of the 16 recurrence iterations**
(deep supervision), and/or extract the LDT paper's **actual** training objective before
another run. This is a *recipe* change, not a compute change — v2 already converged
(lr → 0); more H100 hours on the same objective will not move `false_elim`.

## 6. Gate status (honest)

- `build_gate_pass`: **NO.** Per [`../PROMOTE_GATE.md`](../PROMOTE_GATE.md) §2, **B2
  (twin-state), B1 (`d_dec`/`k_control`), and B3 (soundness frontier) remain GATED.**
  No fiber-size, body-dimension, or control-sufficiency number is licensed off this
  model.
- The build-gate **functioned as designed** — PROMOTE_GATE §0/§4 pre-registered
  "soundness is measured and can fail." The lane made **no** R0+ substrate claim on an
  unsound model; the discipline held.

## 7. Artifacts (gitignored under `results/lattice/`)

- `build-gate-sudoku-extreme-v2/` — `manifest.json`, `train_log.jsonl`,
  `checkpoint_latest.pt` (step 100k).
- `build-gate-sudoku-extreme-v2_eval/` — `manifest.json` (binding verdict),
  `rollout_per_puzzle.jsonl`.
- Runner: `scripts/lattice_ldt_model.py` @ sha256 `599838069b53…` (the lr-decay v2;
  canonical-repo reconciliation pending — see note below).

> **Provenance note.** The binding run executed on the H100 at box commit `345cf311`
> ( `e3be47c4` + lr-decay ). The canonical repo's matching commit is to be reconciled
> when this receipt lands (the lr-decay patch + sha256 `599838…` fully determine the
> code; reproducible as `e3be47c4` + the documented patch).
