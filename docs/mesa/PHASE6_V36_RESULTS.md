# Mesa Phase 6 v3.6 - C->P Mask Functional Transfer Result Note

This document records the Phase 6 v3.6 micro-result following
[`PHASE6_V35_RESULTS.md`](PHASE6_V35_RESULTS.md). v3.5 unexpectedly
found that C->P critical-neuron identity generalized strongly across
held-out Medium pairs. v3.6 asks the cheapest next question: does the
cliff-pair C->P top-32 mask transfer functionally to J1 and J2?

Status: complete. No new PPO training. No new harness code.

## 1. Summary

The answer is yes.

Using the cliff-pair C->P top-32 mask from
`results/mesa/phase6-v3-3-ablation/full/critical-top-32-cp.csv`,
Axis P was rerun on the two v3.5 held-out pairs:

- J1: L-Sig-Terminal-M vs L-Reward-M
- J2: L-Mixed-M-lambda=0.9 vs L-Mixed-M-lambda=0.99

Both pairs show same-direction functional transfer:

| pair | C->P median drop | P->C median drop | dissociation |
| --- | ---: | ---: | ---: |
| J1 | `+0.102` | `-0.049` | `+0.151` |
| J2 | `+0.517` | `+0.009` | `+0.508` |

The v3.6 read:

> The cliff-pair C->P substrate is not merely identity-stable across
> Medium held-out pairs; it transfers functionally. A mask learned on
> the cliff pair preferentially disrupts C->P patching on both J1 and
> J2, with especially strong transfer on J2.

This confirms the v3.5 surprise and sharpens the revised mechanism:
C->P/rescue-side substrate identity is controller-family-wide at Medium
tier in both ranking and function. P->C/basin-inducing behavior remains
subspace-level or pair-specific at simple top-32 neuron identity.

## 2. Artifacts

Outputs:

`results/mesa/phase6-v3-6/`

Key files:

- `axis-p-cpmask-on-j1/substrate-ablation.csv`
- `axis-p-cpmask-on-j1/substrate-ablation-aggregate.csv`
- `axis-p-cpmask-on-j1/dissociation-summary.json`
- `axis-p-cpmask-on-j2/substrate-ablation.csv`
- `axis-p-cpmask-on-j2/substrate-ablation-aggregate.csv`
- `axis-p-cpmask-on-j2/dissociation-summary.json`
- `logs/axis_p_cpmask_j1.log`
- `logs/axis_p_cpmask_j2.log`

## 3. Commands

```bash
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 \
  --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-cp.csv \
  --mask-direction C_to_P --pair J1 \
  --out results/mesa/phase6-v3-6/axis-p-cpmask-on-j1

python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 \
  --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-cp.csv \
  --mask-direction C_to_P --pair J2 \
  --out results/mesa/phase6-v3-6/axis-p-cpmask-on-j2
```

## 4. Detailed Results

J1:

| direction | median baseline PS | median ablated PS | median drop |
| --- | ---: | ---: | ---: |
| P->C | `+0.946` | `+0.945` | `-0.049` |
| C->P | `+0.080` | `-0.005` | `+0.102` |

J2:

| direction | median baseline PS | median ablated PS | median drop |
| --- | ---: | ---: | ---: |
| P->C | `+0.992` | `+1.075` | `+0.009` |
| C->P | `+0.323` | `-0.231` | `+0.517` |

The J2 result is particularly strong. It combines v3.5's high C->P
Jaccard (`0.684`) with a large same-direction functional drop (`+0.517`)
and low cross-direction drop (`+0.009`).

## 5. Interpretation

v3.6 splits the v3.5 reversal into two claims:

1. **C->P substrate transfer is real.** The C->P top-32 set identified
   on the cliff pair transfers in both identity and function to J1/J2.
2. **P->C transfer remains unresolved at neuron identity.** v3.5 found
   weak/mixed P->C Jaccard transfer and the exploratory cliff-P->C mask
   only weakly transferred to J1 while staying ambiguous on J2.

So the updated Phase 6 cascade is:

- v1: the cliff localizes to `net.7`.
- v3/v3.1: the behavior-level control surface compresses to an
  entangled 5D subspace.
- v3.4: within the cliff pair, P->C and C->P top-32 substrates are
  functionally separable.
- v3.5/v3.6: across Medium held-out pairs, **C->P substrate identity
  and function generalize**, while P->C appears more pair-specific at
  simple top-32 neuron identity despite strong 5D-basis behavioral
  transfer.

Natural next branch: pair-specific P->C masks on J1/J2. If each pair's
own P->C mask functionally dissociates its P->C patch while the cliff
P->C mask does not transfer broadly, then the clean story is:

> basin induction is shared at the 5D subspace/control-surface level but
> pair-specific at top-32 neuron identity; basin resistance is shared at
> both levels across Medium policies.

