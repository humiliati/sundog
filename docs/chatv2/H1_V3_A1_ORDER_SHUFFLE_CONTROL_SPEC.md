# Chat-v2 H1/V3 A1 - Order-Shuffle Control Amendment

> 2026-07-01, owner-adopted amendment. This adopts the A1 proposal from
> `H1_V3_0C_BANK_RECEIPT.md`. It amends `H1_V3_0C_CROSSOVER_SPEC.md` after
> `F3-V3c/witness`: witness pairs become reported evidence where available,
> not a binding data gate; the anti-bag-function loophole is moved to a
> model-time order-shuffle control. Non-promotional. Nothing here alters
> `PROMOTE_GATE.md`.

## 1. Why A1 is adopted

V3-0c showed two facts at once:

- **the bank exists:** 29 balanced, floored chess-state axes at ply 40, liveness
  1.000, frozen `surface_max` baselines in
  `results/chatv2/h1_v3/v3_0c_bank_manifest.json`;
- **the witness certificate does not scale:** random-directed search covered
  10/29 axes, exhaustive systematic search covered 7/29, and no marker reached
  24 witnessed axes.

So `F3-V3c/witness` is a certificate-availability result, not a bank-formation
result. A1 keeps the bank and moves the relevant loophole test to the model
representation itself.

## 2. Data gate under A1

The A1 data gate admits on:

- slice floor `>= 120`;
- balance `[0.40, 0.60]`;
- frozen matched surface baselines, including `surface_max`;
- liveness control `e2e4`-present `>= 0.95`;
- bank size `>= 24`.

The existing V3-0c manifest satisfies this gate: 29 axes, ply 40, liveness
1.000. Therefore the A1 data verdict is:

**`H1-V3-A1-DATA-ADMIT`** by the already filed V3-0c manifest.

Witnesses remain reportable evidence where available, but they are no longer
required per axis.

## 3. Model-time order-shuffle control

For every candidate crossing axis, evaluate three held-out accuracies on the
same slice and split:

- `acc_orig`: pretrained-model residual readout on the original UCI sequence;
- `acc_randinit`: same-architecture random-init residual readout on the original
  UCI sequence;
- `acc_shuffle`: pretrained-model residual readout on a deterministic
  same-bag, order-shuffled UCI sequence, against the original labels.

The shuffle is fixed and deterministic. Default construction:

- split the 40-ply UCI prefix into even-ply and odd-ply subsequences;
- shuffle each subsequence with a seed derived from `(global_seed, gid)`;
- re-interleave even/odd plies, preserving side-to-move parity and the exact
  token multiset.

The shuffled sequence is not required to be legal chess. The control asks
whether the model's representation carries order-dependent state rather than a
bag-readable label. If the label is effectively a bag function in the model's
features, shuffled accuracy should survive. If the signal depends on real order,
shuffled accuracy should drop.

## 4. V3-0.5 GPT-2 calibration

This rung is non-gate and CPU-bounded. It validates extraction and anchors
expectations.

Deliverable:

`docs/chatv2/H1_V3_0_5_GPT2_CALIBRATION_RECEIPT.md`

Script:

`scripts/chatv2_h1_v3_gpt2_calibration.py`

For each axis, report:

- `surface_max`;
- best validation-selected layer for `acc_orig`;
- held-out `acc_orig`;
- held-out `acc_randinit`;
- held-out `acc_shuffle`;
- margins: `acc_orig - surface_max`, `acc_orig - acc_randinit`,
  `acc_orig - acc_shuffle`.

GPT-2 does not gate the campaign. A weak or null GPT-2 result is expected and
does not falsify V3-1.

## 5. V3-1 1B gate under A1

V3-1 uses the same bank, splits, shuffle construction, and layer-selection
discipline. A 1B-scale model admits iff at least 20 axes satisfy all:

- `acc_orig >= surface_max + 0.15`;
- `acc_orig >= acc_randinit + 0.15`;
- `acc_orig - acc_shuffle >= 0.10`.

Layer selection:

- layer set `{~L/3, ~2L/3, L}`;
- layer choice is made on validation only;
- held-out test numbers are reported once.

For the floor and shuffle controls, use the best validation layer within the
same layer set for that control, making both controls conservative.

## 6. Branches

| branch | meaning |
| --- | --- |
| `H1-V3-A1-DATA-ADMIT` | existing 29-axis V3-0c bank admitted under A1 |
| `H1-V3-A1-GPT2-CALIBRATED` | GPT-2 calibration filed; non-gate |
| `H1-V3-1-A1-ADMIT` | 1B model crosses all three A1 margins on at least 20 axes |
| `F2-V3a1/carry` | model does not beat the frozen surface baseline |
| `F4-V3a1/floor` | random-init floor explains the apparent carry |
| `F5-V3a1/shuffle` | order-shuffle accuracy survives; signal is bag-like or not order-state |
| `F6-V3a1/compute` | extraction/compute cannot run the registered measurement cleanly |

## 7. Claim language

An eventual positive licenses only:

> "The model reads game state better than the registered surface statistic
> allows, and that readout is order-sensitive under the A1 shuffle control."

It does **not** license "surface-invisible state," "world model," "regime-2 for
AI," or any R3 language. Whether union-form `d_dec >= 20` satisfies
`PROMOTE_GATE.md` remains an explicit V3-2 owner decision.

## 8. Immediate next step

Build/run V3-0.5 GPT-2 calibration against the existing manifest, then file
`H1_V3_0_5_GPT2_CALIBRATION_RECEIPT.md`.

### Staged local commands

Smoke already run once on 2026-07-01:

```powershell
python scripts/chatv2_h1_v3_gpt2_calibration.py --max-axes 1 --batch-size 16
```

Observed wall clock: about 125 s for one axis / 130 unique slice instances / three
feature extractions. The full run touches about 2,113 unique instances, so estimate
25-40 min on local CPU depending on batch size and cache state. Per the repo's
time rule, run it owner-side:

```powershell
python scripts/chatv2_h1_v3_gpt2_calibration.py --batch-size 24
```

Readback path:

`results/chatv2/h1_v3/v3_0_5_gpt2_calibration.json`

Branch after readback:

- always file `H1_V3_0_5_GPT2_CALIBRATION_RECEIPT.md` because this rung is
  non-gate;
- if many axes cross or nearly cross the A1 margins, stage V3-1 on the local
  GTX-1080 before any H200 rental;
- if GPT-2 is null, record the null as expectation-setting, not a campaign stop.

Cross-refs: `H1_V3_0C_BANK_RECEIPT.md` (A1 proposal), `H1_V3_0C_CROSSOVER_SPEC.md`
(superseded witness gate), `H1_V3_STATEBANK_SCOPE.md` (parent), `R2_INTERSECTION_HYPOTHESES.md`
(H1 status), `PROMOTE_GATE.md` (unchanged).
