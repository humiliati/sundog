# Mesa Phase 6 v3.8 - Per-PC Decomposition and Signed Effect Map Result Note

This document records the Phase 6 v3.8 result for
[`PHASE6_V38_SPEC.md`](PHASE6_V38_SPEC.md). v3.8 is the first Phase 6
spec written as an explicit mesa-to-geometry crossover deliverable for
[`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md). It carries
two axes: Axis T (per-PC zero-ablation decomposition over PC1..PC5,
cliff pair only) and Axis U (offline signed direction-of-effect map
from existing v3.3/v3.5 ablation CSVs).

Status: complete. Both axes landed. No new PPO training; ~30 LOC for
Axis U and a per-PC harness wrapper for Axis T.

## 1. Summary

The four pre-registered predictions split into one falsified, one
mixed, and two confirmed - but the surprising result is that **both
axes converge on the same directional asymmetry along independent
measures**. Per-component anatomy (Axis T) and cross-pair sign overlap
(Axis U) agree: P->C basin induction is more component-partitioned
and pair-specific, while C->P basin resistance is more component-shared
and family-wide.

1. **EE1 mixed.** Per-PC neuron substrates are not cleanly partitioned.
   - P->C mean off-diag PC top-32 Jaccard: `0.322`.
   - C->P mean off-diag PC top-32 Jaccard: `0.430`.
   - Confirm threshold `<= 0.20`; falsifier `>= 0.40` in both
     directions. P->C lands between the thresholds; C->P individually
     crosses the falsifier line, but the full EE1 falsifier required
     both directions to cross it.
2. **EE2 falsified.** PC1 is not mechanism-light at the per-neuron
   level. PC1 has the highest P->C max mean ablation cost (`0.0148`),
   ahead of PC2 (`0.0113`), PC4 (`0.0059`), PC5 (`0.0060`), and PC3
   (`0.0029`). v3.1's "no proper sub-subspace reproduces the full
   effect" guardrail stands - PC1's single-PC cost is still in floor
   territory and the K=5 entangled-mechanism result is intact - but the
   stronger "PC1 is offset only" framing is no longer earned.
3. **EE3 confirmed at threshold 0.01, mixed at 0.02.** Cliff pair
   directional opposition is `9` neurons in `P_hurts_C_helps` and `22`
   neurons in `C_hurts_P_helps`, totalling `31` opposition-class
   neurons (confirm threshold `>= 16`, falsifier `< 8`). At the wider
   0.02 threshold the counts drop to `1` and `13` for a total of `14`,
   which is between the confirm and falsifier floors.
4. **EE4 confirmed.** C->P signed structure is more stable across
   held-out pairs than P->C in both threshold regimes.
   - 0.01: J1 P->C `0.182` vs C->P `0.371`; J2 P->C `0.068` vs C->P
     `0.636`.
   - 0.02: J1 P->C `0.091` vs C->P `0.500`; J2 P->C `0.107` vs C->P
     `0.610`. Robust under threshold sensitivity.

The v3.8 synthesis:

> Basin induction (P->C) recruits more component-partitioned neuron
> substrates and pair-specific neuron identity. Basin resistance (C->P)
> recruits more component-shared neuron substrates and family-shared neuron
> identity. The directional asymmetry that v3.5-v3.7 saw at the
> identity layer is now visible at the per-component anatomy layer
> too, on the same cliff pair, with consistent direction. EE1's mixed
> outcome is not noise; it is the per-PC fingerprint of the same
> asymmetry that EE4 sees across pairs.

## 2. Artifacts

Outputs:

`results/mesa/phase6-v3-8/`

Key files:

- `axis-u-signed-effects/signed-neuron-effects.csv`
- `axis-u-signed-effects/directional-opposition-summary.csv`
- `axis-u-signed-effects/pair-sign-overlap.csv`
- `axis-u-signed-effects/summary.json`
- `axis-u-signed-effects-threshold-0-02/summary.json`
- `axis-u-signed-effects-threshold-0-02/directional-opposition-summary.csv`
- `axis-u-signed-effects-threshold-0-02/pair-sign-overlap.csv`
- `axis-t-per-pc/pc1/`, `pc2/`, `pc3/`, `pc4/`, `pc5/` - each holds
  `ablation-table.csv`, `ablation-aggregate.csv`,
  `critical-top-32-pc.csv`, `critical-top-32-cp.csv`,
  `critical-set-summary.csv`, `jaccard-comparison.json`,
  `manifest.json`, `summary.json`
- `axis-t-per-pc/reports/pc-neuron-ablation-matrix.csv`
- `axis-t-per-pc/reports/pc-top32-jaccard.csv`
- `axis-t-per-pc/reports/pc-partition-metrics.csv`
- `axis-t-per-pc/reports/pc-partition-summary.json`
- `logs/axis_t_per_pc.log`

## 3. Commands

```bash
# Axis U - signed direction-of-effect map (offline, ~5 min)
python -m training.mesa.phase6_v2_sae axis-u-signed-effects \
  --tables \
    cliff=results/mesa/phase6-v3-3-ablation/full/ablation-table.csv \
    J1=results/mesa/phase6-v3-5/axis-n-j1/ablation-table.csv \
    J2=results/mesa/phase6-v3-5/axis-n-j2/ablation-table.csv \
  --threshold 0.01 \
  --out results/mesa/phase6-v3-8/axis-u-signed-effects

# Axis U robustness appendix at threshold 0.02
python -m training.mesa.phase6_v2_sae axis-u-signed-effects \
  --tables \
    cliff=results/mesa/phase6-v3-3-ablation/full/ablation-table.csv \
    J1=results/mesa/phase6-v3-5/axis-n-j1/ablation-table.csv \
    J2=results/mesa/phase6-v3-5/axis-n-j2/ablation-table.csv \
  --threshold 0.02 \
  --out results/mesa/phase6-v3-8/axis-u-signed-effects-threshold-0-02

# Axis T - per-PC zero-ablation, cliff pair, 5 PCs x both directions x 8 seeds
python -m training.mesa.phase6_v2_sae axis-t-per-pc-zero-ablation \
  --pcs 1 2 3 4 5 --direction both --seeds 8 --horizon 200 \
  --layer net.7 \
  --out results/mesa/phase6-v3-8/axis-t-per-pc
```

## 4. Axis U Results

### 4.1 Cliff-pair directional opposition (threshold 0.01)

| class | count | neurons (top of list) |
| --- | ---: | --- |
| C_hurts_P_helps | 22 | 10, 33, 39, 42, 43, 44, 54, 65, 69, 85, ... |
| P_hurts_C_helps | 9 | 6, 11, 16, 58, 63, 77, 96, 178, 215 |
| both_hurt | 3 | 93, 101, 194 |
| both_help | 6 | 45, 79, 86, 98, 102, 202 |
| directional_neutral | 216 | - |

Opposition-class total `31` on the cliff pair clears EE3's confirm
threshold of `16`. The asymmetry inside the opposition - `22` favoring
basin-resistance over basin-induction versus `9` favoring the reverse -
is itself informative: at the per-neuron sign level, more of the cliff
machinery actively trades against basin induction than against basin
resistance.

### 4.2 Cross-pair sign-overlap stability (EE4)

`hurts_patch` neuron Jaccard between cliff and each held-out pair:

| pair | direction | threshold 0.01 | threshold 0.02 |
| --- | --- | ---: | ---: |
| J1 | P->C | `0.182` | `0.091` |
| J1 | C->P | `0.371` | `0.500` |
| J2 | P->C | `0.068` | `0.107` |
| J2 | C->P | `0.636` | `0.610` |

C->P sign overlap exceeds P->C sign overlap for every held-out pair
under both thresholds. EE4 is robust to threshold choice. The J2 C->P
value `0.636` (0.01) / `0.610` (0.02) is the strongest cross-pair sign
overlap in the table by a wide margin.

### 4.3 Threshold sensitivity for EE3

Tightening from 0.01 to 0.02 collapses the cliff-pair opposition
counts from `9`/`22` to `1`/`13`. The `C_hurts_P_helps` class survives
the tighter threshold (`13` is still close to the confirm floor of
`16`), but the `P_hurts_C_helps` class essentially evaporates. So:

- The "neurons that actively resist basin induction" class is real and
  threshold-robust.
- The "neurons that actively resist basin resistance" class is
  threshold-fragile and may be a mix of weakly signed effects plus the
  v3.4/v3.6/v3.7 set-level negative-cross-drop effect.

## 5. Axis T Results

### 5.1 PC top-32 Jaccard matrix (cliff pair, both directions)

| direction | PC pair | Jaccard | intersection | union |
| --- | --- | ---: | ---: | ---: |
| P->C | 1-2 | `0.032` | 2 | 62 |
| P->C | 1-3 | `0.524` | 22 | 42 |
| P->C | 1-4 | `0.306` | 15 | 49 |
| P->C | 1-5 | `0.255` | 13 | 51 |
| P->C | 2-3 | `0.085` | 5 | 59 |
| P->C | 2-4 | `0.103` | 6 | 58 |
| P->C | 2-5 | `0.123` | 7 | 57 |
| P->C | 3-4 | `0.488` | 21 | 43 |
| P->C | 3-5 | `0.422` | 19 | 45 |
| P->C | 4-5 | `0.882` | 30 | 34 |
| C->P | 1-2 | `0.422` | 19 | 45 |
| C->P | 1-3 | `0.422` | 19 | 45 |
| C->P | 1-4 | `0.391` | 18 | 46 |
| C->P | 1-5 | `0.362` | 17 | 47 |
| C->P | 2-3 | `0.391` | 18 | 46 |
| C->P | 2-4 | `0.306` | 15 | 49 |
| C->P | 2-5 | `0.333` | 16 | 48 |
| C->P | 3-4 | `0.391` | 18 | 46 |
| C->P | 3-5 | `0.455` | 20 | 44 |
| C->P | 4-5 | `0.829` | 29 | 35 |

### 5.2 PC partition metrics

| direction | mean off-diag Jaccard | max off-diag | min off-diag | unique top-32 union | shared-core (>=3 PCs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| P->C | `0.322` | `0.882` (PC4-5) | `0.032` (PC1-2) | 80 | 22 |
| C->P | `0.430` | `0.829` (PC4-5) | `0.306` (PC2-4) | 64 | 28 |

Three substructure features in the P->C direction worth flagging:

- **PC1 sits closer to PC3/PC4/PC5 than to PC2.** PC1-PC3 Jaccard
  `0.524`, PC1-PC4 `0.306`, PC1-PC5 `0.255` versus PC1-PC2 `0.032`.
  PC2 is the most isolated component in the P->C anatomy.
- **PC4 and PC5 share 30 of 34 top-32 neurons.** This is a sub-share
  inside an otherwise more-partitioned direction.
- **PC3/PC4/PC5 form a tight cluster.** Pairwise Jaccards `0.422` to
  `0.882`. The mean off-diag of `0.322` averages over a partitioned
  PC2-vs-rest structure plus a PC3/4/5 sub-share.

The C->P direction is flatter: the off-diagonal Jaccards cluster
between `0.306` and `0.455` except for the PC4-PC5 sub-share at
`0.829`. PC4 and PC5 appear to share substrate in both directions.

### 5.3 Per-PC P->C max mean ablation cost (EE2)

| PC | P->C max mean cost | rank |
| ---: | ---: | ---: |
| 1 | `0.01483` | 1 |
| 2 | `0.01126` | 2 |
| 5 | `0.00602` | 3 |
| 4 | `0.00591` | 4 |
| 3 | `0.00285` | 5 |

EE2 required PC1 to fall below at least three of PCs 2-5 in P->C max
cost. It fell below none.

Two interpretive guardrails matter here:

- The absolute costs are tiny. PC1's `0.0148` is well inside floor
  territory compared to the v3.3 full-K=5 maxima. This is the
  per-component sensitivity profile, not a reproduction of the patch
  effect. v3.1's "no proper sub-subspace reproduces the full effect"
  result is untouched.
- The PC1 substrate has 32 neurons in its P->C top-32 list, including
  many that overlap PC3/4/5. The PC1-PC3 Jaccard of `0.524` plus the
  cost profile suggests PC1 carries a small but anatomically grounded
  share of the P->C signal, possibly via the same neurons that PC3-5
  also recruit. Cleanly disentangling that is a v3.9 question.

## 6. Cross-Axis Convergence

The most consequential result in v3.8 is that **the same directional
partition shows up on both axes by independent measures**:

| direction | Axis T per-PC mean off-diag Jaccard | Axis U cross-pair sign overlap (J1, J2 @ 0.01) | Axis U cross-pair sign overlap (J1, J2 @ 0.02) |
| --- | ---: | ---: | ---: |
| P->C | `0.322` (more partitioned) | `0.182`, `0.068` (weaker transfer) | `0.091`, `0.107` |
| C->P | `0.430` (more shared) | `0.371`, `0.636` (stronger transfer) | `0.500`, `0.610` |

The directional ordering is the same on every column, under both
thresholds. Two independent ablation experiments on different evidence
- per-component anatomy on the cliff pair, sign overlap across
held-out pairs - arrive at the same partition.

This sharpens the v3.7 synthesis. The earlier phrasing was "shared at
the 5D `net.7` control-surface level, pair-specific at top-32 neuron
identity, functionally real within each held-out pair." v3.8 adds an
internal-structure layer: P->C and C->P also differ in how their
respective top-32 substrates are organized across the five PCs.

Updated v3.8 synthesis:

- **P->C / basin induction:** shared at the 5D control-surface level,
  more partitioned across PCs (with a PC4/PC5 sub-share), pair-specific
  at neuron identity, functionally real within each pair.
- **C->P / basin resistance:** shared at the control-surface level,
  shared across PCs at neuron substrate, family-wide at neuron
  identity, functionally transferable across pairs.

Per-PC anatomy and cross-pair identity are now two convergent slices
of the same directional asymmetry, not independent hypotheses.

## 7. Crossover Contribution to MESA_CROSSOVER_NOTE.md

v3.8 sharpens three findings in
[`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md).

### 7.1 Finding #2 - Variance is not mechanism

Sharpened in a specific way. PC1 is variance-heavy and was previously
described as patch-effect-negligible at v3.1, which read as
"PC1 is offset only." v3.8 shows PC1 has the **highest** per-neuron
P->C max mean ablation cost, while still being inadequate to reproduce
the K=5 patch on its own. The lesson is now finer-grained:

> Variance rank, single-component patch effect, and per-neuron
> ablation sensitivity are three different measures. A component can
> rank top by variance and per-neuron sensitivity yet still fail to
> reproduce the multi-component patch effect on its own.

**Geometry inheritance:** primitive-route residual reporting in
Phase 10/11 must not collapse these three measures. A primitive that
"explains a lot of variance" or "shows the strongest single-primitive
sensitivity" is not automatically the load-bearing one for the full
inversion result.

### 7.2 Finding #3 - Directional asymmetry is structural

Substantially sharpened. v3.5-v3.7 made the asymmetry a behavioral and
identity-level fact. v3.8 turns it into a four-way convergent result:

| layer | P->C signature | C->P signature |
| --- | --- | --- |
| 5D subspace transfer (v3.1) | high for J1/J2 | low for J1, mid for J2 |
| cliff-mask functional transfer (v3.6) | weak | strong on J1/J2 |
| own-mask functional dissociation (v3.7) | works per-pair | works per-pair, J2 underperforms cliff |
| per-PC anatomy (v3.8 Axis T) | partitioned (Jaccard 0.322) | shared (Jaccard 0.430) |
| cross-pair signed identity (v3.8 Axis U) | weak transfer | strong transfer |

The "do not promote" geometry analogy can now be expressed at two
levels: routes whose residual structure is more component-partitioned
(P->C analogue) versus routes whose residual structure is shared
across components and primitives (C->P analogue).

**Geometry inheritance:** the geometry `do not promote` list should
record not just per-primitive failures, but per-component partition
patterns - whether a residual route's signature is component-shared
across primitives or component-partitioned.

### 7.3 Finding #5 - No linear attribution

Reinforced. Axis T is per-component zero-ablation, not linear
attribution. The per-PC max costs sum to far less than the full K=5
patch effect, consistent with v3.1's irreducibility claim. The
result note explicitly disallows "PC X contributes N% of the patch
effect" language for v3.8.

**Geometry inheritance:** Phase 11 metric review continues to reject
"arc X contributes N% of fit" language. v3.8 adds a per-component
analogue: even per-component zero-ablation sensitivity does not add
up to the multi-component patch effect.

## 8. Interpretation

v3.8 was scoped to ask whether the 5D `net.7` control surface
partitions into per-component neuron substrates. The answer is:

- **Partially, and direction-dependently.** P->C is more partitioned;
  C->P is more shared. Both directions show a strong PC4/PC5
  sub-share.
- **The same partition appears in cross-pair sign overlap.** P->C
  identity transfers weakly across pairs; C->P identity transfers
  strongly. Two independent measures converge on the same direction
  story.
- **PC1 is anatomically nontrivial but mechanistically inadequate
  alone.** The "mechanism-light" phrasing was too strong. v3.1's
  irreducibility claim survives unaltered.

The directional asymmetry is now a multi-layer convergent finding,
not a single-measurement quirk. It is the part of the v3.x stack with
the most natural carryover into geometry Phase 10/11 because it
provides a structural template: routes can be partitioned per-component
and per-pair, or shared per-component and per-family, and these two
slices co-vary by direction.

## 9. Next Branch

Two candidate v3.9 directions, gated by what we want to invest:

- **Held-out pair per-PC decomposition.** Run Axis T on J1 and J2 with
  pair-specific PCA bases (or the cliff basis under integrity caveat)
  to test whether P->C pair-specific identity lives at the PC level or
  only at neuron identity. Spec was deferred from v3.8.
- **PC4/PC5 sub-share investigation.** The `0.882` (P->C) and `0.829`
  (C->P) Jaccards between PC4 and PC5 are a localized substructure
  finding. Worth deciding whether this is a basis artifact (PC4 and
  PC5 capture similar residual variance after PC1-3) or a
  substantive mechanism property.

Either branch needs no new PPO training.

The v3.8 cascade has landed in `SUNDOG_V_MESA.md`,
`MESA_CROSSOVER_NOTE.md`, `RICH_DISPLAY_OVERLAY_NOTES.md`,
`SUNDOG_V_GRAVITY.md`, `claims-and-scope.md`, `PROMO_HIGHLIGHTS.md`,
and `mesa.html`. Project memory can mirror the same summary line if the
next session needs the v3.8 result preloaded.
