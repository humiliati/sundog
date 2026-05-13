# Mesa Phase 6 v3.7 - Pair-Specific Own-Mask Functional Ablation Result Note

This document records the Phase 6 v3.7 result for
[`PHASE6_V37_SPEC.md`](PHASE6_V37_SPEC.md). v3.7 closes the
three-layer cross-policy substrate table by asking whether each
held-out pair's own v3.5 top-32 critical-neuron masks functionally
dissociate the same pair's patches.

Status: complete. All four Axis S runs landed. No new PPO training and
no new harness code.

## 1. Summary

v3.7 mostly confirms the pair-specific-substrate story:

1. **DD1 confirmed.** J1 own P->C mask dissociates J1 P->C:
   dissociation `+0.128`, clearing the `>= 0.10` threshold.
2. **DD2 confirmed.** J2 own P->C mask dissociates J2 P->C:
   dissociation `+0.253`, clearing the `>= 0.10` threshold.
3. **DD3 confirmed.** J1 own C->P mask dissociates J1 C->P:
   dissociation `+0.208`, exceeding the v3.6 cliff-mask-on-J1
   reference (`+0.151`).
4. **DD4 mixed, not falsified.** J2 own C->P mask dissociates J2 C->P
   strongly (`+0.392`) and clears the falsifier floor (`< 0.30`), but
   does **not** match the v3.6 cliff-mask-on-J2 reference (`+0.508`).

The v3.7 synthesis:

> P->C/basin-inducing substrates are real within each held-out pair:
> own masks functionally dissociate their own P->C patches. Their
> neuron identity is pair-specific rather than controller-family-wide.
> C->P/basin-resisting substrates remain broadly shared, but J2's own
> C->P mask underperforming the cliff C->P mask means "own mask always
> beats transferred mask" is not earned.

## 2. Artifacts

Outputs:

`results/mesa/phase6-v3-7/`

Key files:

- `axis-s-j1-pc-own-mask/substrate-ablation.csv`
- `axis-s-j1-pc-own-mask/substrate-ablation-aggregate.csv`
- `axis-s-j1-pc-own-mask/dissociation-summary.json`
- `axis-s-j1-cp-own-mask/substrate-ablation.csv`
- `axis-s-j1-cp-own-mask/substrate-ablation-aggregate.csv`
- `axis-s-j1-cp-own-mask/dissociation-summary.json`
- `axis-s-j2-pc-own-mask/substrate-ablation.csv`
- `axis-s-j2-pc-own-mask/substrate-ablation-aggregate.csv`
- `axis-s-j2-pc-own-mask/dissociation-summary.json`
- `axis-s-j2-cp-own-mask/substrate-ablation.csv`
- `axis-s-j2-cp-own-mask/substrate-ablation-aggregate.csv`
- `axis-s-j2-cp-own-mask/dissociation-summary.json`
- `reports/three-layer-closeout.csv`
- `reports/summary.json`

## 3. Commands

```bash
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 --pair J1 \
  --neuron-mask-source results/mesa/phase6-v3-5/axis-n-j1/critical-top-32-pc.csv \
  --mask-direction P_to_C \
  --out results/mesa/phase6-v3-7/axis-s-j1-pc-own-mask

python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 --pair J1 \
  --neuron-mask-source results/mesa/phase6-v3-5/axis-n-j1/critical-top-32-cp.csv \
  --mask-direction C_to_P \
  --out results/mesa/phase6-v3-7/axis-s-j1-cp-own-mask

python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 --pair J2 \
  --neuron-mask-source results/mesa/phase6-v3-5/axis-n-j2/critical-top-32-pc.csv \
  --mask-direction P_to_C \
  --out results/mesa/phase6-v3-7/axis-s-j2-pc-own-mask

python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 --pair J2 \
  --neuron-mask-source results/mesa/phase6-v3-5/axis-n-j2/critical-top-32-cp.csv \
  --mask-direction C_to_P \
  --out results/mesa/phase6-v3-7/axis-s-j2-cp-own-mask
```

## 4. Axis S Results

| pair | mask | same-direction drop | cross-direction drop | dissociation | status |
| --- | --- | ---: | ---: | ---: | --- |
| J1 | own P->C | `+0.081` | `-0.047` | `+0.128` | DD1 confirmed |
| J1 | own C->P | `+0.196` | `-0.012` | `+0.208` | DD3 confirmed |
| J2 | own P->C | `+0.327` | `+0.075` | `+0.253` | DD2 confirmed |
| J2 | own C->P | `+0.478` | `+0.087` | `+0.392` | DD4 mixed |

DD4 is the only wrinkle. It is a real C->P dissociation, and far above
the `< 0.30` falsifier, but it underperforms the v3.6 cliff C->P mask
on J2 (`+0.508`). So the "own mask should match or exceed transferred
mask" sanity prediction is not fully earned on J2.

## 5. Three-Layer Closeout

The closeout table at
`results/mesa/phase6-v3-7/reports/three-layer-closeout.csv` combines:

- v3.1 behavior-level 5D-basis patch transfer,
- v3.5 neuron-identity Jaccard against the cliff pair,
- v3.5/v3.6 cliff-mask functional transfer,
- v3.7 own-mask functional dissociation.

Headline rows:

| pair | direction | v3.1 behavior PS | v3.5 identity Jaccard | transferred-mask dissociation | own-mask dissociation |
| --- | --- | ---: | ---: | ---: | ---: |
| J1 | P->C | `0.941` | `0.255` | `+0.113` | `+0.128` |
| J1 | C->P | `0.162` | `0.422` | `+0.151` | `+0.208` |
| J2 | P->C | `1.001` | `0.067` | `+0.096` | `+0.253` |
| J2 | C->P | `0.631` | `0.684` | `+0.508` | `+0.392` |

## 6. Interpretation

v3.7 closes the open P->C question from v3.5/v3.6. The cliff-pair
P->C mask did not transfer cleanly across pairs, and P->C neuron
identity Jaccard was weak, but each held-out pair's own P->C mask
works on that pair. That means the P->C substrate is not anatomical
noise; it is pair-specific.

The updated Phase 6 synthesis:

- **P->C / basin induction:** shared at the 5D `net.7` control-surface
  level, pair-specific at top-32 neuron identity, functionally real
  within each held-out pair.
- **C->P / basin resistance:** shared at neuron identity and functional
  transfer across Medium held-out pairs, with J2 showing the minor
  caveat that the cliff-derived mask beats J2's own mask.

This is a cleaner and more interesting split than the original v3.5
hypothesis. Basin induction appears to be a common geometric operation
implemented through pair-specific neuron substrates; basin resistance
has a more stable family-wide neuron substrate.

## 7. Next Branch

Natural v3.8: per-PC neuron rankings. The question is now internal
structure inside the 5D control surface:

- Do P->C pair-specific neurons cluster by principal component?
- Does C->P's family-wide substrate span all five PCs or concentrate in
  the same PCs across pairs?
- Is the J2 C->P "transferred mask beats own mask" wrinkle explained by
  one dominant PC or by seed noise in the v3.5 axis-N ranking?

No new PPO training is needed for that branch.

