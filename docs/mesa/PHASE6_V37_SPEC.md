# Mesa Phase 6 v3.7 — Pair-Specific Own-Mask Functional Ablation

This document is the implementation-grade spec for Phase 6 v3.7 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 6 v3.6 (see
[`PHASE6_V36_RESULTS.md`](PHASE6_V36_RESULTS.md)) confirmed that the
cliff-pair C→P top-32 mask transfers functionally to J1 and J2 (J1
dissociation +0.151, J2 +0.508). v3.5 exploratory cliff-pair P→C mask
transfer was weak (J1 +0.113, J2 ambiguous +0.096). v3.5 also found
that pair-specific neuron-identity rankings disagree across the
controller family in the P→C direction (Jaccard 0.255/0.067) but
agree in the C→P direction (Jaccard 0.422/0.684).

v3.7 asks the closing question for the three-layer cross-policy
table: *do each held-out pair's own P→C and C→P top-32 critical
neuron sets functionally dissociate their own patches?* If yes for
P→C, then basin induction has a real anatomically-grounded substrate
within each policy — the substrate is just pair-specific in identity.
If no for P→C, then the P→C "circuit" is anatomically distributed
even within a single policy and the v3.5 P→C rankings are partially
noise.

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6 v3.7.

## 1. Decision Lock

Phase 6 v3.7 starts with five pinned calls:

- **One axis: Axis S.** Pair-specific own-mask functional ablation —
  use each held-out pair's own v3.5-derived top-32 critical-neuron
  sets in v3.4 Axis P, with the patch run on the same pair the mask
  was learned from. Per-PC neuron rankings, integrated gradients, and
  Small-tier work are deferred to v3.8+.
- **Two held-out pairs: J1 and J2.** Same as v3.5/v3.6 (J1 =
  L-Sig-Terminal-M ↔ L-Reward-M; J2 = L-Mixed-M-λ=0.9 ↔ λ=0.99).
- **Four conditions total.** (pair × direction): J1-P→C-own-mask,
  J1-C→P-own-mask, J2-P→C-own-mask, J2-C→P-own-mask. The two C→P
  conditions act as sanity-checks against the v3.6 cliff-mask result
  — own-mask should match or exceed cliff-mask dissociation.
- **No new harness code.** v3.4's `axis-p-substrate-ablation`
  subcommand already accepts `--pair J1` / `--pair J2` and
  `--neuron-mask-source <path>`. v3.7 simply points it at the v3.5
  axis-N output CSVs.
- **No smoke gate.** The full battery is ~5-10 min total; cheaper to
  just run all four conditions than to design a gate.

Total v3.7 compute: 0 new PPO runs, 0 LOC harness extension, ~5-10
minutes wall-clock.

## 2. Scope

Phase 6 v3.7 owns:

- Four runs of `axis-p-substrate-ablation` (J1/J2 × P→C/C→P own-masks).
- v3.7 result note recording the four dissociation values, comparing
  against v3.4 cliff-pair-mask within-pair baselines and v3.6 cliff-
  pair-mask cross-pair transfer.
- Three-layer cross-policy table closeout: with v3.5 (identity), v3.6
  (cliff-pair mask functional transfer), and v3.7 (own-mask
  functional ablation) all in hand, the table is closed for the
  cliff pair + J1 + J2.

Phase 6 v3.7 does **not** own:

- New PPO training runs.
- Per-PC neuron rankings (deferred to v3.8 if v3.7 closes cleanly).
- Pair-specific PCA basis derivation (deferred to v3.8+).
- Integrated gradients or causal scrubbing (deferred to v3.9+).
- Small-tier work (still dimension-blocked).
- Cross-architecture transfer (deferred indefinitely).
- Phase 7 v2 envelope cross-product.

## 3. Axis S — Pair-Specific Own-Mask Functional Ablation

For each (pair, direction) cell:

1. Load critical-neuron set from
   `results/mesa/phase6-v3-5/axis-n-{pair}/critical-top-32-{direction}.csv`.
2. Run `axis-p-substrate-ablation` with `--pair {pair}` and that
   mask, at 16 seeds.
3. Compute dissociation = same-direction-drop − cross-direction-drop.

Compare against three reference points per cell:

- **v3 K=5 baseline patch_success on the pair** (for `baseline_PS`).
- **v3.4 within-cliff-pair dissociation** for the same direction
  (P→C 0.174, C→P 0.662).
- **v3.6 cliff-pair-mask transfer dissociation** on the same held-out
  pair (J1: P→C n/a, C→P +0.151; J2: P→C n/a, C→P +0.508).

The within-pair own-mask should dissociate at least as well as the
cross-pair cliff-mask if the underlying substrate is real.

## 4. Pre-Registered Predictions

Four load-bearing predictions, two of which (DD3, DD4) are
sanity-checks.

### 4.1 (DD1) J1 own-P→C mask dissociates J1 P→C

Using J1's own P→C top-32 mask on J1 yields P→C dissociation ≥ 0.10.

**Falsifier:** dissociation < 0.05. The v3.5 P→C ablation rankings on
J1 are noise-driven; J1's P→C "circuit" is anatomically distributed
even within the pair. Would push v3.8 toward per-PC rankings or
integrated gradients.

### 4.2 (DD2) J2 own-P→C mask dissociates J2 P→C

Symmetric for J2. Dissociation ≥ 0.10.

**Falsifier:** dissociation < 0.05. Same v3.8 routing as DD1.

### 4.3 (DD3, sanity) J1 own-C→P mask dissociates J1 C→P

Using J1's own C→P top-32 mask on J1 yields C→P dissociation ≥
v3.6 cliff-mask-on-J1 (≥ 0.151). Should match or exceed the cliff
mask since it's tuned to the policy directly.

**Falsifier:** dissociation < 0.10. Would mean v3.5 ablation
rankings produce noisy critical sets even where the v3.6 cliff-mask
transfer succeeded — a methodological surprise that would deflate
the v3.6 finding.

### 4.4 (DD4, sanity) J2 own-C→P mask dissociates J2 C→P

Symmetric for J2. Dissociation ≥ v3.6 cliff-mask-on-J2 (≥ 0.508).

**Falsifier:** dissociation < 0.30. Same surprise routing as DD3.

## 5. Pair Manifest (no new policies)

| pair | role | policy_id |
| --- | --- | --- |
| J1 | protected | `signature_terminal_medium` |
| J1 | collapsed | `reward_lambda_1_0_medium_anchor` |
| J2 | protected | `mixed_lambda_0_9_medium_v3` |
| J2 | collapsed | `mixed_lambda_0_99_medium_v4` |

## 6. Metrics

For each (pair, direction) own-mask run, mirroring v3.4 / v3.6
output:

- `baseline_patch_success_{direction}` — v3 K=5 unablated.
- `ablated_patch_success_{direction}` — same-direction with own-mask
  applied.
- `cross_direction_drop` — drop in the other direction's patch
  under the same own-mask.
- `dissociation = same_dir_drop − cross_dir_drop`.

Cross-cuts:

- **Three-layer table closeout** — a single CSV pulling together
  v3.1 behavioral transfer, v3.5 neuron identity transfer, v3.6
  cliff-mask functional transfer, and v3.7 own-mask functional
  ablation per pair and direction. Headline diagnostic for whether
  the v3.7 closeout aligns with the v3.6 picture.

## 7. Commands

```bash
# J1 own-P→C mask on J1
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
    --seeds 16 --layer net.7 --pair J1 \
    --neuron-mask-source results/mesa/phase6-v3-5/axis-n-j1/critical-top-32-pc.csv \
    --mask-direction P_to_C \
    --out results/mesa/phase6-v3-7/axis-s-j1-pc-own-mask

# J1 own-C→P mask on J1
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
    --seeds 16 --layer net.7 --pair J1 \
    --neuron-mask-source results/mesa/phase6-v3-5/axis-n-j1/critical-top-32-cp.csv \
    --mask-direction C_to_P \
    --out results/mesa/phase6-v3-7/axis-s-j1-cp-own-mask

# J2 own-P→C mask on J2
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
    --seeds 16 --layer net.7 --pair J2 \
    --neuron-mask-source results/mesa/phase6-v3-5/axis-n-j2/critical-top-32-pc.csv \
    --mask-direction P_to_C \
    --out results/mesa/phase6-v3-7/axis-s-j2-pc-own-mask

# J2 own-C→P mask on J2
python -m training.mesa.phase6_v2_sae axis-p-substrate-ablation \
    --seeds 16 --layer net.7 --pair J2 \
    --neuron-mask-source results/mesa/phase6-v3-5/axis-n-j2/critical-top-32-cp.csv \
    --mask-direction C_to_P \
    --out results/mesa/phase6-v3-7/axis-s-j2-cp-own-mask
```

## 8. Outputs

```
results/mesa/phase6-v3-7/
  axis-s-j1-pc-own-mask/
    substrate-ablation.csv
    substrate-ablation-aggregate.csv
    dissociation-summary.json
  axis-s-j1-cp-own-mask/
    ...
  axis-s-j2-pc-own-mask/
    ...
  axis-s-j2-cp-own-mask/
    ...
  reports/
    three-layer-closeout.csv          # v3.1 behavior + v3.5 identity + v3.6 cliff-mask + v3.7 own-mask
    summary.json                       # DD1-DD4 outcomes
```

## 9. Execution Order

Four-step sequential run:

1. J1 own-P→C mask (~2 min).
2. J1 own-C→P mask (~2 min).
3. J2 own-P→C mask (~2 min).
4. J2 own-C→P mask (~2 min).
5. Aggregate + classify DD1-DD4 (~30 sec).
6. Write three-layer-closeout CSV (~30 sec).
7. v3.7 result note (`docs/mesa/PHASE6_V37_RESULTS.md`).

Total wall-clock: ~10 min.

## 10. Exit Criterion

Phase 6 v3.7 complete when:

- All four `axis-p-substrate-ablation` runs land per the §7 commands.
- DD1-DD4 classified as confirmed or falsified.
- Three-layer cross-policy table written to `three-layer-closeout.csv`.
- v3.7 result note (`PHASE6_V37_RESULTS.md`) is written.

## 11. Cross-References

- **Phase 6 v3.6:** the cliff-pair C→P mask transfer result v3.7
  closes out with own-mask runs.
  [`PHASE6_V36_RESULTS.md`](PHASE6_V36_RESULTS.md).
- **Phase 6 v3.5:** the cross-policy ablation-rank tables v3.7
  reuses. [`PHASE6_V35_SPEC.md`](PHASE6_V35_SPEC.md),
  [`PHASE6_V35_RESULTS.md`](PHASE6_V35_RESULTS.md).
- **Phase 6 v3.4:** the within-pair functional dissociation
  baseline. [`PHASE6_V34_SPEC.md`](PHASE6_V34_SPEC.md),
  [`PHASE6_V34_RESULTS.md`](PHASE6_V34_RESULTS.md).
- **Phase 6 v3.1 Axis J:** the v3.1 behavioral transfer baseline
  (P→C strong, C→P weak) that the three-layer table inherits.
- **Crossover note:** [`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md).
- **Roadmap:** [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md).

## 12. What v3.8+ Inherits

If DD1 + DD2 + DD3 + DD4 all confirm:

- **The three-layer cross-policy table closes cleanly.** Cascade
  language can pin the synthesis: *basin induction has anatomically-
  grounded substrate within each policy (own-mask works in own pair,
  v3.7 DD1/DD2) but the substrate identity is pair-specific (cliff-
  mask doesn't transfer, v3.5 Jaccard low); basin resistance has
  family-wide substrate identity AND function (v3.5 + v3.6) and
  also functions cleanly within each pair (v3.7 DD3/DD4 sanity).*
- **v3.8 candidate:** per-PC neuron rankings — break out
  contribution by which principal component each neuron mediates.
  If P→C critical neurons within a pair localize to specific PCs
  while C→P critical neurons spread across all 5 PCs, that gives
  finer-grained anatomical structure inside the 5D subspace.

If DD1 OR DD2 falsifies (own-mask P→C doesn't dissociate):

- The P→C ablation rankings are anatomically empty even within a
  single policy. The 5D basin-inducing direction has no localizable
  per-policy substrate — it really is a *purely direction-shaped*
  phenomenon. This would tighten the public framing: P→C is
  geometric all the way down, not anatomical at any scale.
- v3.8 candidate: per-step P→C contribution probing (across steps
  within an episode) to look for a temporal-localization signal
  instead of a spatial one.

If DD3 OR DD4 falsifies (own-mask C→P doesn't dissociate or
underperforms cliff-mask):

- v3.6 cliff-mask C→P transfer was getting credit for something
  other than the substrate. Would deflate the "C→P shared at both
  levels" claim. v3.8 candidate: full v3.5 axis-N rerun on cliff
  pair with higher seed count to tighten the ablation rankings.

## 13. Versioning

- **v3.7 (2026-05-13)** — initial pin. One axis: Axis S (pair-
  specific own-mask functional ablation). Four pre-registered
  predictions DD1-DD4 (DD1/DD2 load-bearing, DD3/DD4 sanity-check).
  Cliff pair + J1 + J2 close out the three-layer cross-policy table.
  No new harness code, no smoke gate, ~10 min total compute.
