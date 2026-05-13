# Mesa Phase 6 v3.3 — Zero-Ablation Attribution at net.7

This document is the implementation-grade spec for Phase 6 v3.3 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 6 v3.2 (see
[`PHASE6_V32_RESULTS.md`](PHASE6_V32_RESULTS.md)) failed Z1 cleanly:
top-32 neuron-restricted patching delivered ~0% of the v3 K=5 patch
effect despite capturing 33.6% of aggregate L2 across PCs 1-5. The
v3.2 falsifier branch routes v3.3 to non-linear neuron-attribution
methods.

v3.3 asks the dual question: *instead of "which neuron subset is
sufficient to deliver the patch" (v3.2's question, which failed), ask
"which neurons are necessary — whose ablation breaks the patch."* The
two questions can have very different answers in non-linear systems.

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6 v3.3.

## 1. Decision Lock

Phase 6 v3.3 starts with six pinned calls:

- **One axis: Axis N.** Per-neuron zero-ablation attribution at
  `net.7` during the v3 K=5 patch battery. Integrated gradients
  (Axis O) and causal scrubbing (Axis P), pre-named in
  PHASE6_V32_RESULTS §7, are deferred to v3.4+ if Axis N does not
  resolve the mechanism.
- **The cliff pair is the central artifact.** L-Mixed-M-λ=0.95 vs
  λ=0.97, matched seed slate. v3.3 reuses the cached v3.1
  reconstruction of the v3 K=5 PCA basis at
  `results/mesa/phase6-v3-1-validation/pca-basis/cliff-pca-net7-seed10000-n64-h200-k5.*`.
- **Ablation is set-to-zero, not set-to-mean.** For each neuron j,
  the hook overwrites `h_new[j] = 0` after the v3 K=5 patch delta has
  been added. Set-to-mean (replace with the dataset mean activation
  of neuron j) is the v3.4 alternative if zero-ablation produces
  uninformative results because zeroing pushes activations off the
  policy's training distribution.
- **Seed slate is 8 seeds for the full sweep.** 256 neurons × 8 seeds
  × 4 forwards = ~8K forwards is the unit cost; smaller seed slates
  trade noise for speed. v3.3's per-neuron signal is averaged across
  seeds, so 8 is a reasonable noise floor; v3.4 can scale to 32 or
  64 if needed for tightness.
- **Smoke gate: top-1 ablation cost must clear 0.05.** Run 4 seeds
  on all 256 neurons in the P→C direction only (~10 min). If the
  most-ablation-costly neuron drops patch_success by less than 0.05,
  the mechanism is so distributed that single-neuron ablation will
  not surface critical neurons; route v3.4 to multi-neuron causal
  scrubbing. Otherwise, proceed to full battery.
- **Both directions are run.** Unlike v3.2 (one-sided P→C smoke),
  v3.3 reports critical-neuron sets for both P→C and C→P because the
  v3.1 directional-asymmetry finding is now load-bearing: if the
  critical neurons differ across direction, that's the neuron-level
  signature of the basin-inducing-vs-basin-resisting asymmetry.

Total v3.3 compute: 0 new PPO runs, ~50 LOC harness extension,
~60-90 minutes wall-clock for the cliff-pair full battery (8 seeds ×
256 neurons × 2 directions × 2 forwards each, with cached clean
baselines).

## 2. Scope

Phase 6 v3.3 owns:

- One new subcommand `axis-n-zero-ablation` in
  `training/mesa/phase6_v2_sae.py`.
- A new `run_zero_ablation_battery` helper that, for each neuron
  j ∈ {0..255}, reruns the v3 K=5 patch with neuron j pinned at zero
  during the injection hook.
- Per-neuron ablation-cost computation (the drop in patch_success
  when that neuron is zeroed, averaged across seeds).
- Critical-neuron set extraction (top-k by ablation cost, separately
  for each direction).
- Jaccard-overlap comparison vs the v3.2 L2-rank top-k set.
- *Optional, gated on Axis N surfacing critical neurons:* re-run
  v3.2 Axis M with the critical-neuron mask in place of the L2-rank
  mask, testing whether ablation-rank fares better than L2-rank for
  top-k patching.
- v3.3 result note with the per-neuron ablation table, critical-set
  overlap analysis, and (if Axis M re-run happens) the comparison
  between ablation-rank and L2-rank top-k patching.

Phase 6 v3.3 does **not** own:

- New PPO training runs.
- Recomputation of the PCA basis (uses the cached v3.1 artifact).
- Integrated gradients on `net.7` → action logits (deferred to v3.4
  as Axis O if Axis N is uninformative).
- Causal scrubbing on hypothesized circuit models (deferred to v3.4
  as Axis P).
- Set-to-mean ablation (deferred to v3.4 alternative if zero-ablation
  produces off-distribution artifacts).
- Pair-wise ablation (n=2-neuron joint ablation) — deferred to v3.4
  if single-neuron ablation finds no critical neurons.
- Small-tier policies (deferred until v3.5+ supplies a Small-tier
  PCA basis).
- Cross-architecture transfer (deferred indefinitely).
- Phase 7 v2 envelope cross-product.

## 3. Axis N — Zero-Ablation Attribution

The single axis of v3.3.

### 3.1 Ablation protocol

For each (seed, direction, neuron j) cell:

1. **Cache baselines once per seed.** Run cache_A (clean protected,
   recording the per-step subspace coords `Q^T h`) and cache_B (clean
   collapsed, recording). These are independent of j and reused
   across all 256 neuron ablations.
2. **Run baseline patch (no ablation).** Compute `baseline_patch_PS`
   from running cache_C (P→C patch, no ablation) and cache_D (C→P
   patch, no ablation). This is the v3 K=5 patch effect.
3. **For each neuron j ∈ {0..255}, run ablated patch.** Run
   cache_C_j (P→C patch with neuron j zeroed during the hook) and
   cache_D_j (C→P patch with neuron j zeroed). Compute
   `ablated_patch_PS_j`.

The ablation hook modifies the v3 injection:

```python
# v3 axis-h injection (reference):
delta = Q @ (target_c - current_c)              # (256,)
h_new = h + delta

# v3.3 Axis N ablation:
delta = Q @ (target_c - current_c)
h_new = h + delta
h_new[j] = 0                                    # zero-out the ablated neuron
return h_new
```

Note: the ablation happens *after* the patch delta is applied. This
isolates the contribution of neuron j to the patched activation,
holding all other neurons at their patched values. Alternative
orderings (zero before patching, zero during baseline rollouts) are
v3.4 candidates if the v3.3 setup produces ambiguous results.

### 3.2 Scoring

Per-neuron ablation cost is the drop in patch_success when that
neuron is zeroed:

```
ablation_cost_PC[j] = baseline_patch_PS_PC - ablated_patch_PS_PC[j]
ablation_cost_CP[j] = baseline_patch_PS_CP - ablated_patch_PS_CP[j]
```

Higher ablation_cost = neuron is more *necessary* for the patch.
Negative values are possible (zeroing the neuron *improves*
patch_success) and would be a real finding worth flagging if they
appear.

### 3.3 Critical-neuron set extraction

For each direction, the critical-neuron set is the top-k neurons by
ablation_cost. v3.3 reports k ∈ {1, 4, 8, 16, 32} for completeness;
the headline comparison number is **top-32** (matches v3.2 Axis M
top-k granularity for Jaccard comparison).

Diagnostics per critical set:
- **Aggregate ablation cost** captured by top-k: `Σ_{j ∈ top-k}
  ablation_cost[j]`. If a small k captures most of the total cost,
  the mechanism is concentrated at the neuron level even though
  v3.2 said it's not concentrated at the L2-rank level.
- **Jaccard overlap with v3.2 L2-rank top-32.** Low Jaccard (≤ 0.4)
  = the two rankings disagree substantially; v3.2's failure was
  partly a ranking choice. High Jaccard (≥ 0.6) = the rankings
  agree; v3.2's failure was about *threshold* (linear-additive
  top-k doesn't work at any rank), not about *which neurons* were
  picked.
- **Jaccard overlap between P→C and C→P critical sets.** Low
  Jaccard = the basin-inducing and basin-resisting mechanisms have
  different critical neurons, generalizing v3.1's directional
  asymmetry to neuron-level. High Jaccard = same critical neurons
  mediate both directions; the asymmetry is in *how* those neurons
  are used, not in *which* neurons.

### 3.4 Optional follow-up: Axis M re-run with critical-neuron mask

Gated on the Axis N smoke surfacing critical neurons (at least one
neuron with ablation_cost ≥ 0.1) and the critical-set extraction
completing for both directions. If gated:

- Take the top-32 critical-neuron set for each direction separately.
- Re-run v3.2 Axis M with that direction's critical-neuron mask in
  place of the L2-rank mask.
- Compare patch_success against:
  - v3.2 L2-rank top-32 (which delivered ~0% effect).
  - v3 K=5 unrestricted baseline (P→C 0.922, C→P 0.830).

If the critical-neuron mask delivers ≥ 0.4 patch_success in at least
one direction, **v3.3 ratchets the gravity-claim mechanistic anchor**
from "5D entangled subspace, non-decomposable to neurons" to "5D
entangled subspace mediated by an identifiable critical-neuron
subset." This is the strongest possible v3.3 outcome.

If the critical-neuron mask still delivers ~0% effect even though
ablation cost was non-trivial, that's a Z3-style "criticality is real
but non-additive" reading — neurons are necessary individually but
patching only them isn't sufficient. Routes v3.4 to integrated
gradients (which doesn't require the patching apparatus) or causal
scrubbing.

## 4. Pre-Registered Predictions

Four load-bearing predictions, each with explicit falsifier.

### 4.1 (AA1) At least one neuron has substantial ablation cost

For at least one neuron j and at least one direction, ablation_cost
≥ 0.3. (Roughly: zeroing that neuron drops patch_success by 33% of
the v3 K=5 baseline.)

**Falsifier:** all 256 neurons have ablation_cost < 0.1 in both
directions. Mechanism is genuinely distributed at the single-neuron
level; v3.4 routes to pair-wise or multi-neuron ablation.

### 4.2 (AA2) Critical-neuron set disagrees with v3.2 L2-rank

Jaccard overlap between Axis N top-32 (P→C critical) and Axis M
L2-rank top-32 is ≤ 0.4.

**Falsifier:** Jaccard ≥ 0.6. v3.2 picked the right neurons but the
linear-additive mask was the wrong methodology — would constrain v3.4
to non-linear methods even when the neuron substrate is correctly
identified.

### 4.3 (AA3) Critical-neuron sets differ across direction

Jaccard overlap between P→C critical top-32 and C→P critical top-32
is ≤ 0.4.

**Falsifier:** Jaccard ≥ 0.6. The same neurons mediate both
directions; the v3.1 directional asymmetry has no neuron-level
signature, and "becoming protected" vs "becoming collapsed" must
differ in *how* the same neurons are used (e.g., sign of activation,
interaction with other layers) rather than *which* neurons participate.

### 4.4 (AA4) Critical-neuron mask outperforms L2-rank in re-run

Gated on Axis N surfacing critical neurons. The follow-up Axis M
re-run with the critical-neuron mask achieves median patch_success
≥ 0.3 in at least one direction (substantially above L2-rank's ~0%).

**Falsifier:** critical-neuron mask still delivers median ≤ 0.1.
Necessity (ablation cost) and sufficiency (patch effect) are
decoupled in this circuit — individually necessary neurons are not
collectively sufficient. Routes v3.4 to integrated gradients or
causal scrubbing on a structured circuit hypothesis.

## 5. Policy Manifest (cliff pair only)

| policy_id | label | tier | role |
| --- | --- | --- | --- |
| `mixed_lambda_0_95_medium_v4` | L-Mixed-M-λ=0.95 | Medium | protected side |
| `mixed_lambda_0_97_medium_v4` | L-Mixed-M-λ=0.97 | Medium | collapsed side |

v3.3 does not extend to J1/J2 in v3.3 — generalization across the
controller family is deferred to v3.4+ if Axis N + optional Axis M
re-run produce a clean ratchet on the cliff pair.

## 6. Metrics

For each (seed, direction, neuron j):
- `baseline_patch_success_{direction}` — v3 K=5 unablated patch
  effect for this seed and direction (constant across j within a
  seed).
- `ablated_patch_success_{direction}_{j}` — patch effect with neuron
  j zeroed.
- `ablation_cost_{direction}_{j} = baseline - ablated`.

Aggregates per (direction, neuron j):
- mean / median across 8 seeds.
- 25th/75th percentile for distributional shape.

Cross-cuts:
- `critical_top_k_{direction}` for k ∈ {1, 4, 8, 16, 32}.
- `aggregate_cost_captured_{direction}_{k}` = Σ_{j ∈ top-k} ablation_cost[j].
- `jaccard_axis_n_vs_axis_m_l2_rank_{direction}` (Axis N's top-32 vs
  Axis M's L2-rank top-32).
- `jaccard_pc_vs_cp_{k}` (P→C top-k vs C→P top-k).
- `negative_ablation_cost_neurons` (any neurons where ablation
  *improved* patch_success — a flagged surprise).

## 7. Harness Extension

`training/mesa/phase6_v2_sae.py` gains one new subcommand,
`axis-n-zero-ablation`, and a new helper
`run_zero_ablation_battery`. Sketch:

```python
def run_zero_ablation_battery(
    *,
    Q_np: np.ndarray,
    seed_start: int,
    seeds: int,
    horizon: int,
    layer: str,
    out_dir: Path,
    protected_spec: PolicySpec = CLIFF_PROTECTED,
    collapsed_spec: PolicySpec = CLIFF_COLLAPSED,
    direction: str = "both",                # "P→C", "C→P", or "both"
) -> None:
    """For each neuron j ∈ {0..d_in}, rerun the v3 K=5 patch with neuron j
    zeroed during the injection hook. Score by ablation_cost (drop in
    patch_success).
    """
    Q = torch.tensor(Q_np, dtype=torch.float32)
    d_in = Q.shape[0]
    ...
    for offset in range(seeds):
        seed = seed_start + offset
        # Cache baselines once per seed.
        cache_A = run_subspace_recording_rollout(... Q=Q)         # no ablation
        cache_B = run_subspace_recording_rollout(... Q=Q)         # no ablation
        # Baseline patch (no ablation).
        cache_C = run_subspace_injected_rollout(... Q=Q, target_coords=cache_A.projections)
        cache_D = run_subspace_injected_rollout(... Q=Q, target_coords=cache_B.projections)
        baseline_ps_pc = safe_patch_success(cache_A.obp, cache_B.obp, cache_C.obp, ...)
        baseline_ps_cp = safe_patch_success(cache_A.obp, cache_B.obp, cache_D.obp, ...)
        # Per-neuron ablated rollouts.
        for j in range(d_in):
            cache_C_j = run_subspace_injected_rollout(
                ... Q=Q, target_coords=cache_A.projections,
                zero_ablate_neuron=j,
            )
            cache_D_j = run_subspace_injected_rollout(
                ... Q=Q, target_coords=cache_B.projections,
                zero_ablate_neuron=j,
            )
            ablated_ps_pc = safe_patch_success(cache_A.obp, cache_B.obp, cache_C_j.obp, ...)
            ablated_ps_cp = safe_patch_success(cache_A.obp, cache_B.obp, cache_D_j.obp, ...)
            # Record ablation_cost = baseline - ablated
            ...
```

Required harness change in `run_subspace_injected_rollout`: add an
optional `zero_ablate_neuron: int | None = None` kwarg. The injection
hook computes the v3 K=5 delta normally, then forces
`h_new[zero_ablate_neuron] = 0` before returning. Matches the
existing pattern (v3.2's `neuron_mask` kwarg was for sufficiency
masking; v3.3's `zero_ablate_neuron` is for necessity ablation).

CLI:

```bash
# Smoke gate: 4 seeds, P→C only
python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
    --seeds 4 --direction P_to_C \
    --out results/mesa/phase6-v3-3-ablation/smoke

# Full battery: 8 seeds, both directions
python -m training.mesa.phase6_v2_sae axis-n-zero-ablation \
    --seeds 8 --direction both \
    --out results/mesa/phase6-v3-3-ablation/full

# Optional Axis M re-run with critical-neuron mask (gated)
python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 32 --seeds 64 --pair cliff \
    --neuron-mask-source results/mesa/phase6-v3-3-ablation/full/critical-top-32.csv \
    --out results/mesa/phase6-v3-3-ablation/axis-m-critical-rerun
```

The `--neuron-mask-source` flag for Axis M is a small extension: when
provided, load the neuron IDs from the CSV instead of computing from
the L2 ranking. ~10 LOC.

Total harness extension for v3.3: ~50 LOC (new ablation hook variant,
new battery function, new CLI subcommand, plus the Axis M
`--neuron-mask-source` flag).

## 8. Outputs

```
results/mesa/phase6-v3-3-ablation/
  smoke/
    ablation-table.csv             # (seed, direction, neuron, ablation_cost)
    smoke-summary.json             # AA1 smoke outcome
  full/
    ablation-table.csv             # full 8 × 256 × 2 = 4096 rows
    ablation-aggregate.csv         # (direction, neuron) → mean/median/quartiles
    critical-top-32-pc.csv         # P→C critical-neuron set
    critical-top-32-cp.csv         # C→P critical-neuron set
    jaccard-comparison.json        # vs v3.2 L2-rank, plus PC vs CP
    manifest.json
  axis-m-critical-rerun/           # optional, gated
    pc-top-32-mask/
      patch.csv
      patch-aggregate.csv
    cp-top-32-mask/
      patch.csv
      patch-aggregate.csv
    v3-vs-v3-3-critical-comparison.csv
  reports/
    summary.json                   # AA1-AA4 outcomes
```

## 9. Execution Order

Recommended sequencing for Phase 6 v3.3:

1. **Harness extensions land first.** Add `zero_ablate_neuron` kwarg
   to `run_subspace_injected_rollout`, write `run_zero_ablation_battery`,
   add `axis-n-zero-ablation` CLI, add Axis M `--neuron-mask-source`
   flag. Smoke-compile.
2. **Axis N smoke (4 seeds, P→C, 256 neurons).** ~10-15 min.
3. **Gate check.** Verify AA1 smoke gate: max ablation_cost ≥ 0.05
   somewhere. If fails, stop. Record negative result. If passes,
   proceed.
4. **Axis N full battery (8 seeds, both directions, 256 neurons).**
   ~60-90 min.
5. **Critical-neuron extraction + Jaccard analysis.** ~1 min.
6. **AA2 + AA3 classified.** Confirm or falsify each.
7. **Optional Axis M re-run with critical-neuron mask** — gated on
   Axis N producing a critical-neuron set with at least one neuron
   at ablation_cost ≥ 0.1. ~10 min if gated.
8. **AA4 classified** (only if Axis M re-run happened).
9. **v3.3 result note** at `docs/mesa/PHASE6_V33_RESULTS.md`.

Total wall-clock: ~75-120 minutes including the optional follow-up.

## 10. Exit Criterion

Phase 6 v3.3 complete when:

- `axis-n-zero-ablation` subcommand lands in
  `training/mesa/phase6_v2_sae.py` and produces all required output
  CSVs.
- Axis N smoke gate result is recorded (pass or fail).
- If smoke passes: full battery completes for 8 seeds × 256 neurons
  × both directions.
- AA1-AA3 predictions are classified.
- If Axis N surfaces critical neurons (at least one ablation_cost
  ≥ 0.1): Axis M re-run with critical-neuron mask completes; AA4
  classified.
- v3.3 result note (`PHASE6_V33_RESULTS.md`) is written.

## 11. Cross-References

- **Phase 6 v3.2 spec / results:** the negative top-k mediation
  result that routes v3.3 to non-linear attribution.
  [`PHASE6_V32_SPEC.md`](PHASE6_V32_SPEC.md),
  [`PHASE6_V32_RESULTS.md`](PHASE6_V32_RESULTS.md).
- **Phase 6 v3.1 spec / results:** the entanglement and directional-
  asymmetry findings v3.3 builds on.
  [`PHASE6_V31_SPEC.md`](PHASE6_V31_SPEC.md),
  [`PHASE6_V31_RESULTS.md`](PHASE6_V31_RESULTS.md).
- **Phase 6 v2 / v3 spec / results:** the 5D PCA basis v3.3 reuses
  unchanged. [`PHASE6_V2_SPEC.md`](PHASE6_V2_SPEC.md),
  [`PHASE6_V2_RESULTS.md`](PHASE6_V2_RESULTS.md).
- **Phase 6 v1 spec / results:** the layer-level locus.
  [`PHASE6_SPEC.md`](PHASE6_SPEC.md),
  [`PHASE6_RESULTS.md`](PHASE6_RESULTS.md).
- **Roadmap:** [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md).

## 12. What v3.4 and Phase 7 v2 Inherit

If AA1-AA4 all confirm (critical neurons exist, disagree with L2-rank,
differ across direction, and the critical-neuron mask outperforms
L2-rank in re-run):

- **The gravity-claim mechanistic anchor ratchets.** "5D entangled
  subspace, non-decomposable to neurons by L2-rank" upgrades to "5D
  entangled subspace mediated by an identifiable critical-neuron
  subset whose membership differs across basin-inducing and basin-
  resisting directions." Cascades into PROMO_HIGHLIGHTS,
  claims-and-scope, SUNDOG_V_MESA, mesa.html. mesa.html §The Locus
  gains a critical-neuron sub-panel.
- **v3.4 work routes to:**
  - Generalization across J1/J2: does the cliff-pair critical-neuron
    set transfer to held-out collapse cells?
  - Pair-wise or triple-wise ablation: find synergistic neuron
    groups whose joint ablation produces super-additive cost.
  - Adversarial neuron editing: surgically modify the critical
    neurons in a collapsed policy and check whether the basin
    attractor is removed.

If AA1 falsifies (no neuron has substantial individual ablation cost):

- The basin-attractor circuit is not just non-decomposable to a few
  specific neurons; it's also not surfaced by single-neuron
  attribution. v3.4 routes to pair-wise ablation (n=2 joint neuron
  zeroing) or integrated gradients.
- The gravity-claim anchor stays at "5D entangled subspace,
  non-decomposable to neurons by either L2-rank (v3.2) or single-
  neuron ablation (v3.3)." That's a real claim about the circuit's
  structural properties even though it's negative.

If AA4 falsifies (critical neurons exist but their mask doesn't
deliver patch effect):

- Necessity and sufficiency are decoupled in this circuit. v3.4
  routes to integrated gradients (which doesn't require the patching
  apparatus and gives per-neuron credit along the activation-space
  integration path) or causal scrubbing (test specific circuit
  hypotheses).

## 13. Versioning

- **v3.3 (2026-05-12)** — initial pin. One axis: Axis N (zero-ablation
  attribution). Four pre-registered predictions AA1-AA4. Cliff pair
  only; J1/J2 generalization deferred to v3.4. Smoke gate: top-1
  ablation cost ≥ 0.05 in P→C with 4 seeds. Full battery: 8 seeds,
  256 neurons, both directions, ~60-90 min wall-clock. Optional Axis M
  re-run with critical-neuron mask gated on AA1 surfacing at least one
  neuron at ablation_cost ≥ 0.1.
