# Mesa Phase 6 v3.4 - Substrate-Restricted Ablation and Jaccard Bootstrap Result Note

This document records the Phase 6 v3.4 result for
[`PHASE6_V34_SPEC.md`](PHASE6_V34_SPEC.md). v3.4 tested whether the
near-disjoint P->C and C->P top-32 zero-ablation rankings from v3.3
are functionally meaningful and statistically stable.

Status: Axis P and Axis Q **complete** on the cliff pair. BB1 and BB2
confirmed. BB3 confirmed for the P->C-vs-L2 comparison.

## 1. Summary

Phase 6 v3.4 gives the first neuron-substrate validation that survives
past the v3.2/v3.3 negative decomposition checks:

1. **BB1 confirmed.** Set-level ablation of each direction's critical
   top-32 neurons functionally dissociates the two patch directions.
   The P->C-critical mask has median dissociation `+0.174`, clearing
   the `>= 0.15` threshold. The C->P-critical mask has median
   dissociation `+0.662`, far above threshold.
2. **BB2 confirmed, with the strong-threshold caveat.** Bootstrap 95%
   CI for P->C vs C->P top-32 Jaccard is `[0.016, 0.085]`, with median
   `0.049`. This clears the BB2 upper-bound threshold (`<= 0.20`) and
   the spec's BB2-strong numeric threshold (`<= 0.10`). It should be
   described as robust near-disjointness; the CI still overlaps the
   chance-level reference of `0.067`, so the stronger "below chance"
   wording should remain soft.
3. **BB3 confirmed for P->C vs L2-rank.** Bootstrap 95% CI for P->C
   critical top-32 vs v3.2 L2 top-32 is `[0.032, 0.123]`, confirming
   that P->C ablation-rank and L2-rank stably disagree. C->P vs L2 is
   much higher (`[0.280, 0.391]`) and was not the BB3 headline.

The clean v3.4 headline:

> The P->C and C->P zero-ablation substrates at `net.7` are
> functionally and statistically separable: each direction's top-32
> substrate preferentially disrupts its own patch direction, and the
> P->C/C->P ranking overlap is robustly low under seed bootstrap.

## 2. Artifacts

Harness:

`training/mesa/phase6_v2_sae.py`

Outputs:

`results/mesa/phase6-v3-4/`

Key files:

- `baseline-k5/axis-h-pca-patch-aggregate.csv`
- `axis-p-pc-mask/substrate-ablation.csv`
- `axis-p-pc-mask/substrate-ablation-aggregate.csv`
- `axis-p-pc-mask/dissociation-summary.json`
- `axis-p-cp-mask/substrate-ablation.csv`
- `axis-p-cp-mask/substrate-ablation-aggregate.csv`
- `axis-p-cp-mask/dissociation-summary.json`
- `axis-q-bootstrap/bootstrap-jaccard.csv`
- `axis-q-bootstrap/bootstrap-summary.json`
- `reports/summary.json`
- `reports/v3-3-vs-v3-4-comparison.csv`

## 3. Smoke Gate Caveat

The original v3.4 smoke gate compared a 16-seed K=5 run to the
published 64-seed v3 medians (`0.922` P->C, `0.830` C->P), requiring
agreement within `0.05`. The 16-seed smoke output was lower:

| run | P->C median | C->P median |
| --- | ---: | ---: |
| v3 published, 64 seeds | `0.922` | `0.830` |
| original v3 artifact, first 16 seeds | `0.804` | `0.712` |
| v3.4 smoke, 16 seeds | `0.779` | `0.719` |

So the literal gate failed, but for a seed-slice calibration reason:
the original v3 first-16 slice would also fail the full-64 threshold.
The v3.4 smoke was close to the original first-16 slice, especially in
C->P, and was therefore not treated as harness drift. Axis P proceeded
with this caveat recorded.

## 4. Axis P - Functional Dissociation

Commands:

```bash
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 \
  --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-pc.csv \
  --mask-direction P_to_C \
  --out results/mesa/phase6-v3-4/axis-p-pc-mask

python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 \
  --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-cp.csv \
  --mask-direction C_to_P \
  --out results/mesa/phase6-v3-4/axis-p-cp-mask
```

Median patch drops:

| mask source | P->C drop | C->P drop | same-minus-cross dissociation |
| --- | ---: | ---: | ---: |
| P->C critical top-32 | `+0.077` | `-0.097` | `+0.174` |
| C->P critical top-32 | `-0.083` | `+0.579` | `+0.662` |

BB1 confirmed. Both directions clear the `>= 0.15` dissociation
threshold. The cross-direction drops are negative: ablating the
opposite direction's substrate slightly improves the patch. That is
stronger than "does not affect"; it is a functional opposition signal.

## 5. Axis Q - Jaccard Bootstrap

Command:

```bash
python -m training.mesa.phase6_v2_sae axis-q-jaccard-bootstrap \
  --ablation-table results/mesa/phase6-v3-3-ablation/full/ablation-table.csv \
  --l2-rank-source results/mesa/phase6-v3-2-neuron-mediation/axis-m-cliff-pair/top-32/neuron-ids-top-32.csv \
  --resamples 1000 \
  --out results/mesa/phase6-v3-4/axis-q-bootstrap
```

Bootstrap summary:

| comparison | median | 95% CI |
| --- | ---: | --- |
| P->C top-32 vs C->P top-32 | `0.049` | `[0.016, 0.085]` |
| P->C top-32 vs v3.2 L2 top-32 | `0.049` | `[0.032, 0.123]` |
| C->P top-32 vs v3.2 L2 top-32 | `0.333` | `[0.280, 0.391]` |

BB2 confirmed: the P->C/C->P Jaccard upper bound is below `0.20`.
BB3 confirmed for the P->C-vs-L2 comparison: its upper bound is also
below `0.20`.

The C->P-vs-L2 comparison lands much higher. This is not a BB3
falsifier because BB3 was pre-registered for P->C, but it is useful
mechanistic asymmetry: the basin-resisting critical set overlaps the
aggregate-L2 substrate more than the basin-inducing set does.

## 6. Prediction Outcomes

| prediction | outcome | read |
| --- | --- | --- |
| BB1 | confirmed | both same-direction substrate ablations dissociate functionally |
| BB2 | confirmed | P->C/C->P Jaccard is robustly low |
| BB2-strong | numeric threshold passed | do not overclaim below-chance anti-correlation |
| BB3 | confirmed for P->C-vs-L2 | P->C ablation-rank and L2-rank stably disagree |

## 7. Interpretation

v3.4 upgrades v3.3's anatomical disjoint-substrate finding into a
functional and statistical one. The v3.3 single-neuron costs were too
small to identify individually necessary neurons, but the top-32
sets behave coherently as substrates:

- The P->C substrate preferentially disrupts the P->C patch.
- The C->P substrate preferentially disrupts the C->P patch.
- The two substrates stay near-disjoint under seed bootstrap.

The mechanistic statement can now ratchet:

> The entangled 5D basin-attractor operation at `net.7` is implemented
> through direction-specific neuron substrates. Individual neurons are
> weak, but the P->C and C->P top-32 substrates are functionally and
> statistically separable.

This does not retract the v3.2/v3.3 negative results. The correct
reading is not "a few neurons explain the circuit." It is "single
neurons are weak, simple L2 top-k sufficiency fails, but
ablation-ranked neuron substrates reveal direction-specific functional
organization."

## 8. Next

Phase 6 v3.5 should route to Path B from the spec: cross-policy
substrate generalization. The natural test is whether the cliff-pair
P->C substrate transfers to J1/J2 P->C collapse cells, and whether
C->P remains more policy-specific.

If v3.5 is too expensive, a cheaper bridge is to rerun Axis P with
32 or 64 seeds on the cliff pair only, tightening the functional
dissociation intervals before cross-policy work.

## 9. Versioning

- **v1 (2026-05-13)** - initial v3.4 result note. Records the smoke
  gate calibration caveat, Axis P functional dissociation, Axis Q
  Jaccard bootstrap, and BB1/BB2/BB3 classifications.
