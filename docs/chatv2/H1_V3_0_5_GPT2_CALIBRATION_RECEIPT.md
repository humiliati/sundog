# Chat-v2 H1/V3-0.5 — GPT-2 Calibration Receipt (A1, non-gate)

> 2026-07-01. Run of the owner-built `scripts/chatv2_h1_v3_gpt2_calibration.py` under
> `H1_V3_A1_ORDER_SHUFFLE_CONTROL_SPEC.md`, on the frozen 29-axis bank
> (`v3_0c_bank_manifest.json`, ply 40). **Non-gate, non-promotional; no R2 claim; per the
> parent scope, GPT-2 is not required to pass.** Verdict as printed:
> `H1-V3-A1-GPT2-CALIBRATED`. Artifact: `results/chatv2/h1_v3/v3_0_5_gpt2_calibration.json`.

## Result: 0 / 29 axes cross — and the margins say why, cleanly

| margin (per axis, held-out) | min | median | max | threshold |
| --- | --- | --- | --- | --- |
| `vs_randinit` (pretrained − random-init floor) | −0.245 | **−0.026** | **+0.034** | ≥ +0.15 |
| `vs_surface` (pretrained − frozen `surface_max`) | −0.265 | **−0.118** | −0.021 | ≥ +0.15 |
| `shuffle_drop` (original − order-shuffled) | −0.052 | **0.034** | 0.146 | ≥ 0.10 |

Layers {4, 8, 12}, validation-chosen; thresholds 0.15 / 0.15 / 0.10 as frozen.

Three independent reads, all pointing the same way:

1. **Zero pretrained carry.** `orig ≈ randinit` on essentially every axis (median −0.026,
   best +0.034): everything GPT-2-small reads of ply-40 board state is explained by the
   **random-features kernel** — pretraining adds nothing here.
2. **Model below the surface suite everywhere** (median −0.118): no axis is even close to
   the crossover.
3. **No order-dependent encoding.** Shuffle drops are small (median 0.034; only 5/29 reach
   0.10, and those co-occur with `orig ≈ randinit`, i.e., kernel noise, not state).

## What the calibration bought (its actual job)

- **The weakest-general-LM anchor:** GPT-2-small carries **no chess board state at ply 40**.
  H2's stack-top positive does *not* extend from bracket state to board state at this scale
  — bracket tracking is shallow-window; 40-ply board state is not.
- **Apparatus validated end-to-end:** extraction, random-init floor, and the A1 shuffle
  control all behave sanely (floors ≈ orig, drops ≈ 0 when nothing is encoded — exactly the
  null signature the control should show on a model with no state).
- **The V3-1 bar, quantified:** with `surface_max` bulk 0.61–0.72 (outliers 0.89), a
  crossing axis needs `acc_model` ≈ **0.76–0.87**. That is a demanding, frozen bar.

## Expectation-setting for V3-1 (honest prior)

General 1.5B models are not strong chess players; `F2-V3c/carry` is a live outcome and
would honestly close the crossover route at 1.5B scale. Per the fork calls: local
CPU-lite / GTX-1080 first; **H200 only if the local route shows a live carry signal.**

**Staged V3-1 (owner-run):** Qwen/Qwen2.5-1.5B (primary; OLMo-2-1B alt), same bank
manifest, same three margins, layers {≈L/3, ≈2L/3, L} validation-chosen, CPU-lite
subsample first (~600 instances/axis cap). The calibration script generalizes — the staged
form is the same battery with the model swapped, e.g.:
`python scripts/chatv2_h1_v3_gpt2_calibration.py --model Qwen/Qwen2.5-1.5B --max-inst 600`
(model flag per the owner's script conventions; multi-hour CPU job → owner terminal, not
agent-launched).

Cross-refs: `H1_V3_A1_ORDER_SHUFFLE_CONTROL_SPEC.md` (the adopted amendment),
`H1_V3_0C_BANK_RECEIPT.md` (frozen bank + baselines), `H1_V3_STATEBANK_SCOPE.md` §8
(this rung's charter), `PROMOTE_GATE.md` (unchanged).
