# Mesa Phase 6 v3.3 - Zero-Ablation Attribution Result Note

This document records the Phase 6 v3.3 result for
[`PHASE6_V33_SPEC.md`](PHASE6_V33_SPEC.md). v3.3 followed the v3.2
negative result by testing the dual question: if top-k neurons are not
sufficient to deliver the v3 K=5 patch, are any individual `net.7`
neurons necessary for that patch?

Status: Axis N **complete** on the cliff pair. The smoke gate passed
weakly, but the full 8-seed battery falsified AA1. The optional Axis M
critical-neuron-mask rerun was not gated.

## 1. Summary

Phase 6 v3.3 gives one main negative result and two secondary structure
signals:

1. **AA1 falsified.** No single neuron has substantial ablation cost in
   the full 8-seed battery. The largest P->C mean ablation cost is
   `0.040` at neuron `194`; the largest C->P mean ablation cost is
   `0.096` at neuron `100`. Both are below the AA1 falsifier threshold
   of `0.1`, far below the pre-registered substantial threshold of
   `0.3`.
2. **AA2 confirmed as a weak-ranking fact.** The P->C ablation-ranked
   top-32 set barely overlaps the v3.2 aggregate-L2 top-32 set
   (`Jaccard = 0.049`). C->P also stays below the `<= 0.4` prediction
   line (`Jaccard = 0.362`). L2-rank and zero-ablation-rank are not the
   same lens.
3. **AA3 confirmed.** P->C and C->P ablation-ranked top-32 sets barely
   overlap (`Jaccard = 0.049`). The v3.1 directional asymmetry has a
   neuron-ranking signature, even though no individual neuron is a
   strong necessary mediator.
4. **AA4 not run.** The critical-neuron Axis M rerun was gated on Axis N
   surfacing at least one neuron with ablation cost `>= 0.1`. The full
   battery did not clear that gate.

The clean v3.3 headline:

> The entangled 5D basin-attractor subspace at `net.7` is not
> decomposed by aggregate-L2 top-k sufficiency (v3.2) or by
> single-neuron zero-ablation necessity (v3.3). Direction-specific
> attribution rankings exist, but no single neuron is individually
> load-bearing enough to mediate the patch.

## 2. Artifacts

Harness:

`training/mesa/phase6_v2_sae.py`

Smoke output:

`results/mesa/phase6-v3-3-ablation/smoke/`

Full output:

`results/mesa/phase6-v3-3-ablation/full/`

Key files:

- `ablation-table.csv`
- `ablation-aggregate.csv`
- `critical-top-32-pc.csv`
- `critical-top-32-cp.csv`
- `critical-set-summary.csv`
- `jaccard-comparison.json`
- `summary.json`
- `manifest.json`

PCA basis loaded from:

`results/mesa/phase6-v3-1-validation/pca-basis/cliff-pca-net7-seed10000-n64-h200-k5.*`

## 3. Smoke Gate

Smoke command:

```bash
python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
  --seeds 4 --direction P_to_C \
  --out results/mesa/phase6-v3-3-ablation/smoke
```

Smoke result:

| metric | value | gate |
| --- | ---: | ---: |
| max P->C mean ablation cost | `0.057` | `>= 0.05` |
| top neuron | `242` | |
| smoke gate | pass | |

The smoke pass was weak and did not survive the full 8-seed battery.

## 4. Full Battery

Full command:

```bash
python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
  --seeds 8 --direction both \
  --out results/mesa/phase6-v3-3-ablation/full
```

Headline values:

| direction | max mean ablation cost | top neuron | mean baseline patch_success |
| --- | ---: | ---: | ---: |
| P->C | `0.040` | `194` | `0.765` |
| C->P | `0.096` | `100` | `0.516` |

Top P->C neurons:

| rank | neuron | mean ablation cost |
| ---: | ---: | ---: |
| 1 | `194` | `0.040` |
| 2 | `242` | `0.035` |
| 3 | `215` | `0.029` |
| 4 | `16` | `0.025` |
| 5 | `93` | `0.024` |

Top C->P neurons:

| rank | neuron | mean ablation cost |
| ---: | ---: | ---: |
| 1 | `100` | `0.096` |
| 2 | `85` | `0.081` |
| 3 | `146` | `0.076` |
| 4 | `149` | `0.073` |
| 5 | `217` | `0.070` |

Negative mean ablation costs are common: `302` of the aggregate
direction-neuron rows are negative. This means zeroing many neurons
slightly improves patch_success rather than degrading it, reinforcing
the read that individual-neuron zeroing is a noisy and weak necessity
probe for this circuit.

## 5. Critical-Set Structure

Top-32 Jaccards:

| comparison | Jaccard |
| --- | ---: |
| P->C Axis N top-32 vs v3.2 L2 top-32 | `0.049` |
| C->P Axis N top-32 vs v3.2 L2 top-32 | `0.362` |
| P->C Axis N top-32 vs C->P Axis N top-32 | `0.049` |

The ranked sets are meaningfully different even though individual
costs are weak. That preserves one mechanistic lesson from v3.1: the
basin-inducing and basin-resisting directions do not look like the
same operation under this probe.

Positive-cost concentration:

| direction | top-1 | top-4 | top-8 | top-16 | top-32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P->C | `0.047` | `0.150` | `0.253` | `0.402` | `0.616` |
| C->P | `0.061` | `0.206` | `0.368` | `0.584` | `0.830` |

C->P has stronger weak-neuron concentration than P->C in the
zero-ablation ranking, but neither direction clears the single-neuron
necessity threshold.

## 6. Prediction Outcomes

| prediction | outcome | read |
| --- | --- | --- |
| AA1 | falsified | no neuron reaches `0.1`, let alone `0.3` |
| AA2 | confirmed | ablation-rank and L2-rank disagree |
| AA3 | confirmed | P->C and C->P ranked sets differ |
| AA4 | not run | critical-mask rerun not gated |

The AA1 falsifier is the load-bearing result. AA2 and AA3 remain useful
as descriptive ranking facts, but they do not upgrade the mechanistic
anchor because the underlying ablation costs are too small.

## 7. Interpretation

v3.3 strengthens the negative decomposition story:

- v3 localized the cliff to an entangled 5D subspace at `net.7`.
- v3.2 showed that delivering that subspace shift through aggregate-L2
  top-32 neurons is not sufficient.
- v3.3 shows that removing any single neuron after the patch is not
  enough to break the patch strongly.

The circuit is therefore not decomposed by either a simple sufficiency
mask or a simple single-neuron necessity test. The best current
mechanistic statement remains:

> The basin-attractor mechanism is an entangled 5D activation-space
> operation at the actor's final hidden layer, implemented through a
> distributed neuron substrate whose direction-specific rankings are
> visible but whose individual neurons are not strongly necessary.

## 8. Next

Phase 6 v3.4 should choose between:

- **Pair-wise / multi-neuron ablation.** Natural continuation of AA1's
  falsifier branch: if no single neuron is costly, test whether pairs
  or small groups have super-additive ablation cost.
- **Set-to-mean ablation.** Checks whether set-to-zero pushed
  activations off distribution and washed out necessity signals.
- **Integrated gradients on `net.7 -> action`.** Moves away from the
  patch apparatus and asks which neurons receive action-credit along
  the activation path.
- **Causal scrubbing.** Best once v3.4 has a concrete circuit
  hypothesis to test.

Given v3.3, the strongest next move is pair-wise ablation on the union
of the weak top-ranked P->C and C->P sets, rather than all
`256 choose 2` pairs.

## 9. Versioning

- **v1 (2026-05-12)** - initial v3.3 result note. Records smoke pass,
  full-battery AA1 falsification, AA2/AA3 ranking facts, and AA4
  not-gated status.
