# Mesa Phase 6 v3.5 — Cross-Policy Substrate Generalization (Path B)

This document is the implementation-grade spec for Phase 6 v3.5 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 6 v3.4 (see
[`PHASE6_V34_RESULTS.md`](PHASE6_V34_RESULTS.md)) confirmed both
load-bearing predictions:

- **BB1 confirmed:** the v3.3 P→C and C→P top-32 ablation-rank
  substrates dissociate functionally — each substrate preferentially
  disrupts its own patch direction (P→C drop 0.077 + C→P improvement
  0.097 = 0.174 dissociation; C→P drop 0.579 + P→C improvement 0.083
  = 0.662 dissociation). The basin-resisting substrate is ~4× more
  anatomically tight than basin-inducing.
- **BB2 confirmed:** the cross-direction Jaccard 0.049 is
  statistically stable (95% CI [0.016, 0.085]). Robust
  near-disjointness, though the CI still brackets chance-level
  (0.067) so the stronger "anti-correlated" reading remains unearned.
- **BB3 confirmed for P→C-vs-L2:** ablation-rank and L2-rank stably
  disagree in the basin-inducing direction (Jaccard CI [0.032,
  0.123]); the basin-resisting substrate overlaps L2-rank more
  substantially (Jaccard 0.333, CI [0.280, 0.391]) — retroactively
  explaining v3.2's asymmetric failure.

v3.4 validated the disjoint-substrate finding on the cliff pair.
v3.5 takes Path B: **does the substrate identity generalize across
the Medium controller family?** v3.1's patch_success generalization
showed asymmetric directional transfer (basin-inducing transfers,
basin-resisting doesn't); v3.5 asks whether the underlying
*neuron substrates* transfer with the same asymmetric structure.

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6 v3.5.

## 1. Decision Lock

Phase 6 v3.5 starts with six pinned calls:

- **One axis: Axis R.** Cross-policy zero-ablation and substrate
  generalization. Within-substrate refinement (per-PC neuron
  rankings, multi-mask intersections), Small-tier cross-tier
  adapter, deployment-monitoring probe (Path C), and integrated
  gradients are all deferred to v3.6+.
- **Two held-out pairs: J1 and J2.** J1 = L-Sig-Terminal-M ↔
  L-Reward-M-λ=1.0 (canonical signature-vs-reward at Medium).
  J2 = L-Mixed-M-λ=0.9 ↔ L-Mixed-M-λ=0.99 (within-L-Mixed at a
  different cliff). v3.1 Axis J already established that the
  cliff-pair PCA basis patches J1/J2 with the same asymmetric shape
  (P→C strong, C→P weak); v3.5 asks the substrate analog.
- **Compute v3.3-style zero-ablation rankings on J1 and J2.** Run
  the existing `axis-n-zero-ablation` subcommand with `--pair J1`
  and `--pair J2` (the harness already accepts these from v3.3).
  256 neurons × 8 seeds × 4 forwards × 2 directions per pair = ~70
  min per pair = ~140 min total.
- **Headline comparison: cliff-pair critical sets vs J1/J2 critical
  sets, per direction.** For each (cliff-pair direction, held-out
  pair, held-out direction) cell, compute Jaccard between cliff-pair
  top-32 and held-out top-32. Two main numbers per held-out pair:
  cliff P→C ↔ J P→C, and cliff C→P ↔ J C→P. Two cross-coupling
  numbers per pair: cliff P→C ↔ J C→P, cliff C→P ↔ J P→C
  (sanity-check that cross-direction overlap stays low).
- **Bootstrap every Jaccard comparison.** 1000 resamples on the
  ablation-table CSVs, same as v3.4 Axis Q. CIs reported alongside
  point estimates.
- **Smoke gate: J1 P→C smoke at 4 seeds.** If max ablation cost in
  the J1 P→C direction is < 0.03 (below v3.3's cliff-pair P→C max
  of 0.040), the v3.3 methodology may not surface meaningful
  critical neurons in held-out pairs at all, and v3.5 should pivot
  to substrate-restricted ablation directly (carrying the
  cliff-pair substrate forward as the mask). Otherwise proceed to
  full battery.

Total v3.5 compute: 0 new PPO runs, ~30 LOC harness extension
(--pair-driven aggregation wrapper, the Jaccard-pair-comparison CSV
emitter), ~140 min wall-clock for both J pairs.

## 2. Scope

Phase 6 v3.5 owns:

- Reuse of existing `axis-n-zero-ablation` subcommand with `--pair
  J1` and `--pair J2`.
- New analysis subcommand `axis-r-substrate-generalization` that
  takes the cliff-pair v3.3 critical-set CSVs plus the new J1/J2
  ablation-table CSVs and emits Jaccard comparison tables with
  bootstrap CIs.
- *Optional, gated on Axis R surfacing high-Jaccard generalization
  in the P→C direction:* re-run v3.4 Axis P (substrate-restricted
  ablation) on J1/J2 using the **cliff-pair** P→C top-32 mask.
  Tests whether the cliff-pair-derived substrate functionally
  dissociates J1/J2 patches even though it was learned on the
  cliff pair.
- v3.5 result note recording all Jaccard CIs, the CC1/CC2/CC3
  prediction classifications, and (if optional re-run happens) the
  functional-transfer result.

Phase 6 v3.5 does **not** own:

- New PPO training runs.
- Small-tier ablation (J3 still dimension-blocked).
- Deployment-monitoring probe (Path C, deferred to v3.6+).
- Per-PC neuron rankings (v3.3 used aggregate cost; per-PC ranking
  remains a v3.6 candidate).
- Cross-architecture transfer (deferred indefinitely).
- Phase 7 v2 envelope cross-product.

## 3. Axis R — Cross-Policy Substrate Generalization

The single axis of v3.5.

### 3.1 J1/J2 ablation rankings

Reuse v3.3's `axis-n-zero-ablation` with `--pair J1` and `--pair J2`.
For each pair:

- Cache clean baselines (J protected + J collapsed) once per seed.
- Compute baseline patch_success in both directions (no ablation)
  per seed.
- For each neuron j ∈ {0..255}, run the v3 K=5 patch with neuron j
  zeroed; compute ablated patch_success per direction.
- Aggregate per (direction, neuron) across 8 seeds; rank neurons by
  mean ablation cost descending; emit top-32 per direction as the
  critical set.

Output: per-pair critical-top-32-pc.csv and critical-top-32-cp.csv,
mirroring v3.3's cliff-pair output schema.

### 3.2 Jaccard comparison matrix

For each (cliff-pair direction, held-out pair, held-out direction)
cell:

```
J_cliff,pair,direction =
    |cliff_top32_direction ∩ pair_top32_direction|
  / |cliff_top32_direction ∪ pair_top32_direction|
```

Reported as a 2×2 matrix per held-out pair:

| | held P→C | held C→P |
| --- | ---: | ---: |
| cliff P→C | J(cliff_pc, pair_pc) | J(cliff_pc, pair_cp) |
| cliff C→P | J(cliff_cp, pair_pc) | J(cliff_cp, pair_cp) |

Diagonal entries (cliff P→C ↔ held P→C; cliff C→P ↔ held C→P) carry
the load-bearing generalization claims. Off-diagonal entries should
stay low (~chance level) if the directional substrate identity is
well-defined.

Bootstrap each Jaccard via 1000-resample seed-level over the
ablation-table.

### 3.3 Optional follow-up: cliff-pair-mask substrate-restricted ablation on J1/J2

Gated on CC1 confirming (diagonal Jaccard ≥ 0.30 in at least one
direction for at least one pair). If gated:

- Take the v3.3 cliff-pair P→C top-32 critical set.
- Run v3.4 Axis P (substrate-restricted ablation) on J1 and J2
  using this *cliff-pair-derived* mask instead of a fresh
  per-pair critical set.
- Measure dissociation: does ablating the cliff-pair P→C substrate
  on J1/J2 preferentially disrupt J1/J2 P→C patching?

If the cliff-pair substrate **functionally transfers** to J1/J2 P→C
even when the J1/J2 critical neurons differ, that's a stronger claim
than substrate identity (Jaccard): it says the cliff-pair-identified
*neurons themselves* are the basin-inducing infrastructure across the
Medium controller family, regardless of policy-specific tuning.

## 4. Pre-Registered Predictions

Three load-bearing predictions plus one optional gated prediction.

### 4.1 (CC1) Basin-inducing substrate generalizes

For both held-out pairs, cliff-pair P→C top-32 ∩ held-pair P→C
top-32 has Jaccard ≥ 0.30 (median across 1000 resamples).

**Falsifier:** Jaccard ≤ 0.15 for both held-out pairs. The
basin-inducing substrate is cliff-pair-specific in identity, even
though v3.1 showed the cliff-pair *basis* patches both held-out
pairs in the P→C direction.

**Strong-confirmation flag (CC1-strong):** Jaccard ≥ 0.45 for both
held-out pairs. The cliff-pair basin-inducing neurons *are* the
controller-family-wide basin-inducing neurons.

### 4.2 (CC2) Basin-resisting substrate does NOT generalize

For both held-out pairs, cliff-pair C→P top-32 ∩ held-pair C→P
top-32 has Jaccard ≤ 0.15 (median across 1000 resamples).

**Falsifier:** Jaccard ≥ 0.30 for either pair. The basin-resisting
substrate is also family-wide; the v3.1 cross-policy generalization
asymmetry (basin-resisting fails to transfer behaviorally) is *not*
explained by substrate identity, and there's something else
mediating the v3.1 asymmetry.

Confirmation here corroborates the v3.4 synthesis: basin-resisting
is policy-specific in both behavior (v3.1) and substrate (v3.5).

### 4.3 (CC3) Off-diagonal Jaccards stay near chance

Both cross-direction comparisons (cliff P→C ↔ held C→P, and cliff
C→P ↔ held P→C) have Jaccard within ±0.05 of chance-level (0.067).

**Falsifier:** Either off-diagonal Jaccard > 0.20. The
directional-substrate identity itself is fuzzy across policies —
what we called the "P→C substrate" in one policy doesn't correspond
to the "P→C substrate" in another. Would suggest the cliff-pair
substrate framing is structurally specific to the cliff pair and
doesn't compose across the controller family at all.

CC3 is the sanity-check prediction: it should confirm trivially if
CC1 + CC2 confirm.

### 4.4 (CC4, optional) Cliff-pair substrate transfers functionally to J1/J2

Gated on CC1 confirming for at least one held-out pair. Run v3.4
Axis P on J1/J2 with the cliff-pair P→C top-32 mask. Prediction:
the cliff-pair-derived mask achieves dissociation ≥ 0.10 on at
least one J1/J2 P→C patch (i.e., it preferentially disrupts J1/J2
P→C patching even though the mask was learned on the cliff pair).

**Falsifier:** dissociation < 0.05 in both J1 and J2. The
generalization is at the neuron-identity level but not at the
functional level — interesting but weaker than full transfer.

CC4 confirmation would be the strongest possible v3.5 outcome:
**the cliff-pair-identified basin-inducing neurons are the
controller-family-wide basin-inducing infrastructure, both in
ranking identity (CC1) and in functional behavior (CC4)**.

## 5. Held-Out Pair Manifest

J1 (canonical signature-vs-reward at Medium):

| role | policy_id |
| --- | --- |
| protected | `signature_terminal_medium` |
| collapsed | `reward_lambda_1_0_medium_anchor` |

J2 (within-L-Mixed family, different λ-cliff):

| role | policy_id |
| --- | --- |
| protected | `mixed_lambda_0_9_medium_v3` |
| collapsed | `mixed_lambda_0_99_medium_v4` |

J3 (Small-tier cross-family) remains dimension-blocked per v3.1.1
amendment.

## 6. Metrics

### 6.1 Per-pair critical-set extraction

Reuses v3.3 metrics (per-(direction, neuron, seed) ablation cost,
per-(direction, neuron) aggregate cost, top-k by aggregate cost).

### 6.2 Jaccard comparison matrix

For each of {cliff, J1, J2} × each of {P→C, C→P}:

| pair | direction | Jaccard vs cliff_PC | Jaccard vs cliff_CP |
| --- | --- | ---: | ---: |
| J1 | P→C | (load-bearing CC1) | (off-diagonal CC3) |
| J1 | C→P | (off-diagonal CC3) | (load-bearing CC2) |
| J2 | P→C | (load-bearing CC1) | (off-diagonal CC3) |
| J2 | C→P | (off-diagonal CC3) | (load-bearing CC2) |

Each cell reports median and 95% CI from 1000 resamples.

### 6.3 Optional CC4 dissociation

If gated:

| pair | direction | baseline_PS | ablated_PS | drop |
| --- | --- | ---: | ---: | ---: |
| J1 | P→C | | | |
| J1 | C→P | | | |
| J2 | P→C | | | |
| J2 | C→P | | | |

Per pair, compute cliff-mask dissociation = same-direction-drop −
cross-direction-drop, paralleling v3.4 Axis P.

## 7. Harness Extension

`training/mesa/phase6_v2_sae.py` adds one new analysis subcommand and
reuses two existing ones:

```bash
# Reuse: v3.3 axis-n on J1
python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
    --seeds 8 --direction both --pair J1 \
    --out results/mesa/phase6-v3-5/axis-n-j1

# Reuse: v3.3 axis-n on J2
python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
    --seeds 8 --direction both --pair J2 \
    --out results/mesa/phase6-v3-5/axis-n-j2

# New: axis-R analysis (Jaccard comparison + bootstrap)
python -m training.mesa.phase6_v2_sae axis-r-substrate-generalization \
    --cliff-pc results/mesa/phase6-v3-3-ablation/full/critical-top-32-pc.csv \
    --cliff-cp results/mesa/phase6-v3-3-ablation/full/critical-top-32-cp.csv \
    --cliff-table results/mesa/phase6-v3-3-ablation/full/ablation-table.csv \
    --pair-tables results/mesa/phase6-v3-5/axis-n-j1/ablation-table.csv \
                  results/mesa/phase6-v3-5/axis-n-j2/ablation-table.csv \
    --resamples 1000 \
    --out results/mesa/phase6-v3-5/axis-r-generalization

# Optional CC4: v3.4 axis-P with cliff-pair P→C mask, on J1/J2
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
    --seeds 16 --layer net.7 \
    --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32-pc.csv \
    --mask-direction P_to_C \
    --pair J1 \
    --out results/mesa/phase6-v3-5/axis-p-cliffmask-on-j1
# (and symmetric for J2)
```

Implementation notes:
- The `axis-r-substrate-generalization` subcommand is pure offline
  analysis on existing CSVs: ~80 LOC (loader, Jaccard computation,
  bootstrap loop, output writer).
- The v3.4 `axis-p-substrate-ablation` already accepts `--pair J1`
  / `--pair J2` via the v3.4 harness extension; CC4 reuses it
  without modification.
- v3.3's `axis-n-zero-ablation` already accepts `--pair J1` / `--pair
  J2` via the v3.3 harness; no new code is needed for the
  per-pair ablation runs.

Total v3.5 harness extension: ~80 LOC (one new analysis subcommand).

## 8. Outputs

```
results/mesa/phase6-v3-5/
  axis-n-j1/                          # reuses v3.3 axis-n output schema
    ablation-table.csv
    critical-top-32-pc.csv
    critical-top-32-cp.csv
    manifest.json
  axis-n-j2/                          # same schema
    ...
  axis-r-generalization/
    jaccard-matrix.csv                 # rows: (pair, direction); cols: vs cliff_pc, vs cliff_cp; with CI
    jaccard-bootstrap-cliff-vs-j1-pc.csv
    jaccard-bootstrap-cliff-vs-j1-cp.csv
    jaccard-bootstrap-cliff-vs-j2-pc.csv
    jaccard-bootstrap-cliff-vs-j2-cp.csv
    cc-predictions-summary.json
  axis-p-cliffmask-on-j1/             # optional, gated
    substrate-ablation.csv
    substrate-ablation-aggregate.csv
    dissociation-summary.json
  axis-p-cliffmask-on-j2/             # optional, gated
    ...
  reports/
    summary.json                       # CC1, CC2, CC3, CC4 outcomes
    v3-1-vs-v3-5-comparison.csv       # v3.1 patch_success generalization vs v3.5 substrate-Jaccard
```

## 9. Execution Order

Recommended sequencing for Phase 6 v3.5:

1. **J1 P→C smoke gate (4 seeds).** ~5-10 min. Verify max
   ablation cost ≥ 0.03 so the v3.3 methodology surfaces something
   on held-out pairs.
2. **Axis N on J1 full battery (8 seeds, both directions).** ~70 min.
3. **Axis N on J2 full battery.** ~70 min.
4. **Axis R analysis (Jaccard comparison + bootstrap).** ~30 sec.
5. **Classify CC1, CC2, CC3.**
6. **Gate on CC1 for optional CC4.** If CC1 confirmed in either J,
   run v3.4 Axis P with cliff-pair P→C mask on the confirming
   held-out pair. ~10 min per pair.
7. **v3.5 result note** at `docs/mesa/PHASE6_V35_RESULTS.md`.

Total wall-clock: ~150-170 minutes.

## 10. Exit Criterion

Phase 6 v3.5 complete when:

- New `axis-r-substrate-generalization` subcommand lands and emits
  the Jaccard comparison matrix.
- J1 P→C smoke gate result is recorded.
- If smoke passes: full axis-N batteries complete for both J1 and
  J2, both directions, 8 seeds each.
- Axis R Jaccard matrix is computed with 1000-resample CIs.
- CC1, CC2, CC3 classified.
- If CC1 confirms: optional Axis P re-run with cliff-pair mask
  completes; CC4 classified.
- v3.5 result note (`PHASE6_V35_RESULTS.md`) is written.

## 11. Cross-References

- **Phase 6 v3.4 spec / results:** the substrate-dissociation
  finding v3.5 generalizes.
  [`PHASE6_V34_SPEC.md`](PHASE6_V34_SPEC.md),
  [`PHASE6_V34_RESULTS.md`](PHASE6_V34_RESULTS.md).
- **Phase 6 v3.3 spec / results:** the original critical-neuron
  ranking machinery v3.5 reuses.
  [`PHASE6_V33_SPEC.md`](PHASE6_V33_SPEC.md),
  [`PHASE6_V33_RESULTS.md`](PHASE6_V33_RESULTS.md).
- **Phase 6 v3.1 Axis J:** the patch_success generalization
  precedent v3.5 mirrors at neuron-substrate level.
  [`PHASE6_V31_SPEC.md`](PHASE6_V31_SPEC.md) §3.2.
- **Phase 7 v1 envelope:** the policy-zoo manifest J1/J2 are drawn
  from. [`PHASE7_RESULTS.md`](PHASE7_RESULTS.md).
- **Crossover note:** the field-shape framing v3.5 contributes to.
  [`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md).
- **Roadmap:** [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md).

## 12. What v3.6+ Inherits

If CC1 + CC2 + CC3 all confirm (and CC4 if gated):

- **The gravity-claim mechanistic anchor ratchets to controller-
  family-wide neuron identity.** "5D entangled subspace at net.7,
  basin-inducing and basin-resisting substrates direction-specific"
  → "the basin-inducing substrate is a **named subset of net.7
  neurons shared across the Medium controller family**; the
  basin-resisting substrate is **named per-policy** but consistently
  anatomically tight."
- **v3.6 routes to Path C (deployment monitoring):** train a probe
  at net.7 using the cliff-pair P→C critical-neuron set; evaluate
  AUROC for predicting basin-collapse on the Phase 5 zoo.
- **Cascade pass:** PROMO_HIGHLIGHTS, claims-and-scope,
  SUNDOG_V_GRAVITY, mesa.html get one more update — "named neuron
  subset" replaces the current "direction-specific substrate"
  language for the basin-inducing claim specifically.

If CC1 falsifies (basin-inducing substrate is cliff-pair-specific
in neuron identity):

- The v3.1 patch_success generalization (P→C transfers) and the
  v3.5 substrate identity (P→C doesn't transfer) disagree. The
  cliff-pair PCA *basis* spans the family-wide basin-inducing
  *direction*, but the family-wide direction is implemented by
  different neurons in different policies. That's still a
  publishable structural claim and routes v3.6 to direction-only
  generalization analysis.

If CC2 falsifies (basin-resisting substrate generalizes too):

- The v3.1 patch_success asymmetry is genuinely surprising — at
  the substrate level, both directions transfer — and v3.6 needs
  to investigate why the *behavioral* C→P patch fails to transfer
  even when the substrate identity does.

## 13. Versioning

- **v3.5 (2026-05-13)** — initial pin. One axis: Axis R (cross-
  policy substrate generalization via Jaccard between cliff-pair
  critical sets and J1/J2 critical sets). Three pre-registered
  predictions CC1-CC3 plus optional gated CC4. J3 still dimension-
  blocked. Smoke gate: J1 P→C max ablation cost ≥ 0.03 at 4 seeds.
  Compute: ~150-170 min, 0 new PPO runs, ~80 LOC harness extension
  (one new analysis subcommand). Ratchet sentence to earn: *the
  cliff-pair-identified basin-inducing neurons are the controller-
  family-wide basin-inducing infrastructure at Medium tier; the
  basin-resisting neurons are per-policy.*
