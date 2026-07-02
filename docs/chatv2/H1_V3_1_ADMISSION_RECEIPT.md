# Chat-v2 H1/V3-1 — Model-Admission Receipt (gate rung)

> 2026-07-01. Run of the V3-1 gate under `H1_V3_0C_CROSSOVER_SPEC.md` +
> `H1_V3_A1_ORDER_SHUFFLE_CONTROL_SPEC.md`, **full bank**, on the local GTX-1080 (fp16,
> torch 2.12.1+cu126 — sm_61 still shipped, no downgrade needed; fp16 hidden states verified
> finite). Script: `scripts/chatv2_h1_v3_1_admission.py` (reuses the owner's V3-0.5 battery
> verbatim; resume-safe extraction). Artifact:
> `results/chatv2/h1_v3/v3_1_admission_Qwen2.5-1.5B.json`. **Non-promotional; no R2 claim.**

## Verdict: `F2-V3c/carry` — 0 / 29 axes cross; all 29 surface-blocked

Qwen2.5-1.5B, 2,110 slice instances (3 of the manifest's 2,113 gids replay-diverged),
layers {9, 19, 28} validation-chosen, batch 16.

| margin | min | median | max | threshold |
| --- | --- | --- | --- | --- |
| `vs_surface` | −0.273 | **−0.123** | **0.000** (`b.a7`) | ≥ +0.15 |
| `vs_randinit` | −0.161 | **+0.017** | +0.106 | ≥ +0.15 |
| `shuffle_drop` | −0.114 | **+0.015** | +0.125 | ≥ 0.10 |

- **The model never reaches the surface baseline** — the crossover's first margin fails on
  every axis before the others matter (blockers: surface 29, floor 0, shuffle 0).
- **The smoke's d7 whisper was noise, as flagged:** at full n (1,249), `occ.d7`/`b.d7`
  regressed from +0.10/+0.13 to vs_randinit +0.005/+0.024, shuffle_drop +0.067/+0.043.
- **Scale bought almost nothing:** vs GPT-2-small (V3-0.5), 124M → 1.5B moved `orig` by a
  few points and nudged vs_randinit slightly positive, while a crossing axis needs
  `orig` ≈ 0.76–0.90 — a +0.2–0.3 jump. Same null signature, warmer by ε.

## Disposition — the V3 ladder is complete

Per the fork calls (scope §0.5): **the H200 required a live local carry signal; there is
none. The H200 is definitively not boarded for this lane.** Escalating to 7B-class models
was pre-excluded by the campaign ("don't begin at 7B"), and a chess-specialist model would
change the claim itself (no longer a *general* pretrained LM). V3-2 is never reached;
`PROMOTE_GATE.md` R2 stays **NOT STARTED**.

**The terminal recommendation is now unambiguous: bank R1 and freeze.** The R2 boundary is
the most thoroughly measured negative in the portfolio:

- **six label families** killed at the data level (count-parity, agreement, FewRel
  relations, code-vars, chess whole-distribution, chess ambiguity-slice);
- **two slice designs** (natural distribution; count-ambiguous fibers) and **two gate
  forms** (absolute ceiling; matched-baseline crossover);
- **two models × the full A1 battery** (GPT-2-small: zero carry, orig ≈ random-init;
  Qwen2.5-1.5B: zero order-sensitive carry, all surface-blocked) — with the apparatus
  proving itself on known-answer controls at every rung (liveness 1.000, null shuffle
  signature exactly where it should be);
- **machine-checked theory anchors**: `SurfaceBag.lean` (bag determines depth, resists
  stack-top; `[propext, Quot.sound]`) and the owner's `SurfaceBagGraded.lean`
  (`stackTop_resists_every_window`: σ_surface = ∞ at every window order);
- and the lane's one real positive intact and correctly scoped: **H2** — a pretrained LM
  carries an order-dependent state (bracket stack-top) that defeats the order-blind
  statistic, at low dimension, on the ambiguity slice.

The honest closing statement: *the intersection {model-computed ∧ surface-beating ∧
high-dimensional} exists at low dimension (H2) but has resisted every bank-scale
construction on natural data, through six families, three gate designs, and two model
scales.* That sentence, with R1's licensed toy result, is the lane's bankable content.

Cross-refs: `H1_V3_0_5_GPT2_CALIBRATION_RECEIPT.md` (the anchor this confirms),
`H1_V3_0C_BANK_RECEIPT.md` (frozen baselines), `H1_V3_STATEBANK_SCOPE.md` (fork calls),
`PHASE1_R1_COMPLETION.md` (R1, the result to bank), `PROMOTE_GATE.md` (unchanged).
