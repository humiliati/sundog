# Mesa Phase 6 v3.5 - Cross-Policy Substrate Generalization Result Note

This document records the Phase 6 v3.5 result for
[`PHASE6_V35_SPEC.md`](PHASE6_V35_SPEC.md). v3.5 tested whether the
v3.4 cliff-pair P->C and C->P top-32 neuron substrates generalize in
identity to two held-out Medium policy pairs.

Status: Axis N on J1 and J2 **complete**. Axis R **complete**. Optional
Axis P with the cliff-pair P->C mask was also run on J1 and J2, even
though Axis R did not strictly gate it.

## 1. Summary

Phase 6 v3.5 is a useful reversal of the v3.5 prior:

1. **Smoke gate passed.** J1 P->C 4-seed smoke reached max mean
   ablation cost `+0.099` at neuron `85`, clearing the `>= 0.03`
   proceed gate.
2. **CC1 mixed.** Basin-inducing substrate identity did not cleanly
   generalize. J1 P->C vs cliff P->C Jaccard was near but below the
   confirmation line (`0.255`, CI `[0.032, 0.422]`); J2 was chance-like
   (`0.067`, CI `[0.016, 0.208]`). This does not confirm CC1 and does
   not fully meet the "both pairs <= 0.15" falsifier, so the recorded
   status is mixed.
3. **CC2 falsified strongly.** Basin-resisting substrate identity
   generalized instead of staying policy-specific. J1 C->P vs cliff
   C->P Jaccard was `0.422` (CI `[0.255, 0.641]`); J2 was `0.684`
   (CI `[0.422, 0.829]`). Both exceed the `>= 0.30` falsifier.
4. **CC3 mixed.** Three off-diagonal medians stayed near chance
   (`0.085`, `0.085`, `0.049`), but J2 P->C vs cliff C->P was elevated
   at `0.185`, above the chance-window ceiling (`0.117`) but below the
   hard `> 0.20` falsifier.
5. **CC4 was not gated, but exploratory Axis P is informative.** The
   cliff-pair P->C mask weakly transferred to J1 (`+0.113`
   dissociation), but J2 was ambiguous (`+0.096` dissociation, with
   same-direction median drop `-0.009`). This is not a clean functional
   transfer result.

The revised v3.5 mechanistic statement:

> The cliff-pair 5D PCA basis remains a behavior-level control surface,
> but neuron-substrate identity does not transfer in the originally
> predicted direction. Basin-inducing P->C critical neurons are
> heterogeneous across held-out pairs; basin-resisting C->P critical
> neurons are the ones with strong cross-policy identity overlap. The
> v3.1 behavioral transfer asymmetry is therefore not explained by
> simple top-32 neuron identity transfer.

## 2. Artifacts

Harness:

`training/mesa/phase6_v2_sae.py`

Outputs:

`results/mesa/phase6-v3-5/`

Key files:

- `smoke-j1-pc/summary.json`
- `axis-n-j1/ablation-table.csv`
- `axis-n-j1/critical-top-32-pc.csv`
- `axis-n-j1/critical-top-32-cp.csv`
- `axis-n-j1/summary.json`
- `axis-n-j2/ablation-table.csv`
- `axis-n-j2/critical-top-32-pc.csv`
- `axis-n-j2/critical-top-32-cp.csv`
- `axis-n-j2/summary.json`
- `axis-r-generalization/jaccard-matrix.csv`
- `axis-r-generalization/cc-predictions-summary.json`
- `axis-p-cliffmask-on-j1/substrate-ablation-aggregate.csv`
- `axis-p-cliffmask-on-j1/dissociation-summary.json`
- `axis-p-cliffmask-on-j2/substrate-ablation-aggregate.csv`
- `axis-p-cliffmask-on-j2/dissociation-summary.json`

## 3. Commands

Smoke:

```bash
python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
  --seeds 4 --direction P_to_C --pair J1 --smoke-threshold 0.03 \
  --out results/mesa/phase6-v3-5/smoke-j1-pc
```

Full Axis N:

```bash
python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
  --seeds 8 --direction both --pair J1 \
  --out results/mesa/phase6-v3-5/axis-n-j1

python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
  --seeds 8 --direction both --pair J2 \
  --out results/mesa/phase6-v3-5/axis-n-j2
```

Axis R:

```bash
python -m training.mesa.phase6_v2_sae axis-r-substrate-generalization \
  --cliff-pc results/mesa/phase6-v3-3-ablation/full/critical-top-32-pc.csv \
  --cliff-cp results/mesa/phase6-v3-3-ablation/full/critical-top-32-cp.csv \
  --cliff-table results/mesa/phase6-v3-3-ablation/full/ablation-table.csv \
  --pair-tables results/mesa/phase6-v3-5/axis-n-j1/ablation-table.csv \
                results/mesa/phase6-v3-5/axis-n-j2/ablation-table.csv \
  --resamples 1000 \
  --out results/mesa/phase6-v3-5/axis-r-generalization
```

Exploratory Axis P:

```bash
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 \
  --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-pc.csv \
  --mask-direction P_to_C --pair J1 \
  --out results/mesa/phase6-v3-5/axis-p-cliffmask-on-j1

python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
  --seeds 16 --layer net.7 \
  --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-pc.csv \
  --mask-direction P_to_C --pair J2 \
  --out results/mesa/phase6-v3-5/axis-p-cliffmask-on-j2
```

## 4. Smoke Gate

| run | direction | max mean ablation cost | top neuron | gate |
| --- | --- | ---: | ---: | --- |
| J1 smoke, 4 seeds | P->C | `+0.099` | `85` | pass |

The smoke gate cleared the `>= 0.03` threshold. Full Axis N proceeded.

## 5. Axis N - Held-Out Critical Sets

Max mean ablation costs:

| pair | P->C max | C->P max |
| --- | ---: | ---: |
| J1: L-Sig-Terminal-M vs L-Reward-M | `+0.114` | `+0.068` |
| J2: L-Mixed-M-lambda=0.9 vs lambda=0.99 | `+0.081` | `+0.374` |

J2's C->P direction is the standout: its strongest single-neuron
zero-ablation cost is much larger than any other held-out direction.
That anticipates the Axis R result: the C->P/resisting-side substrate
is more stable and more anatomically concentrated than the original
v3.5 hypothesis expected.

## 6. Axis R - Jaccard Matrix

Median Jaccard with 1000-resample 95% CIs:

| pair | direction | vs cliff P->C | vs cliff C->P |
| --- | --- | ---: | ---: |
| J1 | P->C | `0.255` `[0.032, 0.422]` | `0.085` `[0.032, 0.164]` |
| J1 | C->P | `0.085` `[0.016, 0.164]` | `0.422` `[0.255, 0.641]` |
| J2 | P->C | `0.067` `[0.016, 0.208]` | `0.185` `[0.123, 0.280]` |
| J2 | C->P | `0.049` `[0.016, 0.123]` | `0.684` `[0.422, 0.829]` |

Chance reference for top-32-of-256 Jaccard is approximately `0.067`.

Prediction statuses:

| prediction | status | read |
| --- | --- | --- |
| CC1: P->C substrate generalizes | mixed | J1 partial (`0.255`), J2 chance-like (`0.067`) |
| CC2: C->P substrate does not generalize | falsified | C->P generalizes strongly on both J1 (`0.422`) and J2 (`0.684`) |
| CC3: off-diagonals near chance | mixed | three near chance, J2 P->C vs cliff C->P elevated (`0.185`) |
| CC4: optional P->C functional transfer | not gated | Axis R did not meet the CC1 gate |

## 7. Exploratory Axis P - Cliff P->C Mask on J1/J2

Although CC4 was not formally gated, the optional runs landed:

| pair | P->C baseline median | P->C ablated median | P->C drop | C->P drop | dissociation |
| --- | ---: | ---: | ---: | ---: | ---: |
| J1 | `0.946` | `0.776` | `+0.079` | `-0.035` | `+0.113` |
| J2 | `0.992` | `1.036` | `-0.009` | `-0.105` | `+0.096` |

J1 weakly clears the `>= 0.10` CC4-style dissociation mark. J2 does
not cleanly clear it, and its same-direction drop is negative; its
apparent dissociation is driven by a stronger cross-direction negative
drop. This should be treated as exploratory evidence that the cliff
P->C mask has some functional transfer to J1 but not a robust
family-wide transfer result.

## 8. Interpretation

v3.5 falsifies the cleanest substrate-generalization story that v3.4
suggested.

v3.1 showed that the cliff-pair 5D PCA basis patches held-out pairs
asymmetrically: P->C transfer is strong, C->P rescue is weaker. v3.5
asked whether top-32 neuron-substrate identity explains that behavioral
asymmetry. It does not.

The observed structure is almost the inverse:

- P->C behavior transfers well at the 5D-basis level, but P->C top-32
  neuron identity is heterogeneous across held-out pairs.
- C->P behavior transfers weakly at v3.1's basis-patch level, but C->P
  top-32 neuron identity generalizes strongly across held-out pairs.
- Off-direction overlap is mostly low, so direction labels remain
  meaningful, but J2 has a non-trivial P->C/C->P cross-coupling.

The most conservative reading:

> "Basin induction" is shared at the subspace/control-surface level but
> not at simple top-neuron identity. "Basin resistance" has a stable
> neuron-substrate identity across Medium policies, but that identity
> alone does not guarantee symmetric behavioral rescue under the cliff
> PCA patch.

This is a useful split. It separates three layers that earlier v3.x
language sometimes blurred:

1. **Subspace-level behavioral control:** the 5D cliff basis can move
   held-out policies.
2. **Neuron-identity substrate:** which neurons are necessary under
   zero-ablation rankings.
3. **Functional mask transfer:** whether a critical-neuron set learned
   on the cliff pair disrupts held-out patching.

v3.5 says these layers do not collapse to one simple story. The
gravity-claim mechanistic anchor remains the 5D `net.7` control
surface, but the neuron-substrate generalization story is now more
qualified.

## 9. Roadmap Implication

Update the Phase 6 cascade wording:

- Keep: *the Phase 5 cliff is a final-hidden, 5D activation-space
  decision surface.*
- Keep: *within the cliff pair, P->C and C->P top-32 substrates are
  direction-specific and functionally dissociable.*
- Revise: *the basin-inducing substrate is controller-family-wide while
  basin-resisting is policy-specific.*
- Replace with: *P->C behavioral transfer is subspace-level rather than
  simple neuron-identity-level; C->P neuron identity is unexpectedly
  stable across Medium held-out pairs.*

Natural v3.6 branches:

1. **Functional C->P mask transfer.** Run Axis P on J1/J2 using the
   cliff-pair C->P mask. v3.5 only tested the cliff P->C mask; the
   unexpectedly high C->P Jaccards make the C->P functional-transfer
   run the cleanest next check.
2. **Per-PC substrate maps.** Split top-32 rankings by PC contribution
   or per-PC ablation to see whether J2's P->C/C->P cross-coupling is
   a shared PC direction rather than a shared neuron set.
3. **Pair-specific P->C mask transfer.** Use J1/J2's own P->C top-32
   masks in Axis P. If own-mask P->C dissociation is strong while
   cliff-mask transfer is weak, P->C is pair-specific at neuron
   identity but real within pair.

The immediate result-note headline:

> v3.5 decouples subspace transfer from neuron-substrate identity:
> P->C transfers behaviorally through the 5D basis but not through a
> stable top-32 neuron set, while C->P unexpectedly shows the stable
> cross-policy neuron identity.

