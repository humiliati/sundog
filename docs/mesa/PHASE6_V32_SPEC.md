# Mesa Phase 6 v3.2 — Top-k Neuron Mediation Inside the Entangled 5D Subspace

This document is the implementation-grade spec for Phase 6 v3.2 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 6 v2/v3 (see
[`PHASE6_V2_RESULTS.md`](PHASE6_V2_RESULTS.md)) localized the basin
attractor at `net.7` to a 5-dimensional subspace. Phase 6 v3.1 (see
[`PHASE6_V31_RESULTS.md`](PHASE6_V31_RESULTS.md)) refined three load-
bearing things:

1. The 5-dim subspace is **entangled** across PCs 1-5 — neither PC1
   alone nor PCs 2-5 alone reproduce the full patch effect; all five
   together are required.
2. The cliff-pair PCA basis **generalizes asymmetrically**: basin-
   *inducing* (protected → collapsed) at cliff-pair quality on J1 and
   J2; basin-*resisting* (collapsed → protected) is policy-specific
   (J1 collapses to 0.16, J2 partial at 0.63).
3. PCs 1-5 are **moderately concentrated** across net.7 neurons —
   top-32 L2-concentration ranges 0.377-0.663 per PC. Not
   single-neuron, not fully distributed.

v3.2 asks the natural next question: *if the mechanism is moderately
concentrated in neuron-space, can the 5D patch effect be approximated
by patching only the top-k contributing net.7 neurons — and does the
basin-inducing / basin-resisting asymmetry carry into neuron-level
mediation?*

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6 v3.2.

## 1. Decision Lock

Phase 6 v3.2 starts with six pinned calls:

- **One axis: Axis M.** Top-k neuron-restricted projection of the v3
  K=5 PCA patch. SAE per-policy work, CKA basis-similarity, Small-tier
  basis derivation, and per-attention-head granularity (n/a here, MLP)
  are deferred to v3.3+.
- **The cliff pair is the central artifact (again).** L-Mixed-M-λ=0.95
  vs λ=0.97, matched seed slate 10000-10063. v3.2 reuses the cached
  v3.1 reconstruction of the v3 K=5 PCA basis at
  `results/mesa/phase6-v3-1-validation/pca-basis/cliff-pca-net7-seed10000-n64-h200-k5.*`.
- **Neuron ranking is by aggregate L2 across PCs 1-5.** For each
  neuron j ∈ {0..255}, compute `score[j] = Σ_{i=1..5} v_ij²` where
  v_i is the i-th principal-component vector. Rank descending; top-k
  is the first k entries. This is the "shared mechanism participant"
  ordering — neurons heavily involved across multiple PCs rank higher.
- **k sweep is {8, 16, 32, 64, 256}.** k=256 is the unrestricted v3
  K=5 baseline (sanity check that v3.2 plumbing equals v3 result when
  no neurons are masked out). k ∈ {8, 16, 32, 64} characterize the
  recovery curve.
- **Patch protocol is "compute full delta, then mask non-top-k
  entries to zero before applying."** This isolates the contribution
  of top-k neurons to the v3 K=5 patch effect, holding the PC basis
  fixed. The alternative ("restrict the PC basis itself to top-k
  rows") would conflate basis-restriction with neuron-restriction;
  v3.2 keeps the basis intact and asks "of the delta v3 would have
  applied, how much can we deliver through k neurons?"
- **Smoke gate before the full battery.** Run k=32 with 8 seeds first.
  If top-32 P→C median patch_success ≥ 0.4 of v3 K=5 baseline (i.e.,
  ≥ 0.37 absolute on P→C), proceed to the full sweep. Below that:
  declare top-k mediation infeasible for this circuit and route v3.3
  to non-linear neuron-mediation methods (e.g., zero-ablation
  attribution or integrated gradients on net.7 → action).

Total v3.2 compute: 0 new PPO runs, ~30 LOC harness extension, ~30-50
minutes wall-clock for the cliff-pair sweep + ~30 min for optional
J1/J2 follow-up.

## 2. Scope

Phase 6 v3.2 owns:

- One new subcommand `axis-m-neuron-mediation` in
  `training/mesa/phase6_v2_sae.py`.
- A new `run_neuron_restricted_patch_battery` helper paralleling
  `run_subspace_patch_battery` from v3.
- Neuron ranking computation (aggregate L2 across PCs 1-5).
- Smoke gate result (top-32 cliff-pair pass/fail at 8 seeds).
- Full k-sweep battery on the cliff pair (k ∈ {8, 16, 32, 64, 256},
  both directions, 64 seeds).
- *Optional, gated on cliff-pair top-k mediation working:* apply the
  same neuron mask to J1 and J2 from v3.1 Axis J to test whether the
  basin-inducing / basin-resisting asymmetry carries through.
- v3.2 result note with the recovery curve, asymmetry diagnostics,
  and the neuron-ID list for the top-32 cliff-pair neurons.

Phase 6 v3.2 does **not** own:

- New PPO training runs.
- Recomputation of the PCA basis during smoke/full sweeps (use the
  canonical cached n=64, h=200, K=5 cliff-pair PCA artifact).
- Per-PC neuron ranking (v3.2 uses the *aggregate* L2 ranking across
  PCs 1-5; per-PC neuron rankings are a v3.3 candidate if the
  aggregate is too coarse).
- Non-linear neuron-mediation methods (zero-ablation attribution,
  integrated gradients) — deferred to v3.3 only if Axis M's smoke
  gate fails.
- Small-tier policies (deferred until v3.3 supplies a Small-tier
  PCA basis).
- Cross-architecture transfer (deferred indefinitely).
- Phase 7 v2 envelope cross-product.

## 3. Axis M — Top-k neuron mediation

The single axis of v3.2.

### 3.1 Neuron ranking

For each neuron j ∈ {0..255} at `net.7`, compute the aggregate L2
contribution across the v3 K=5 PCA basis:

```
score[j] = Σ_{i=1..5} v_ij²
```

where `v_i ∈ R^256` is the i-th principal-component vector (column i
of `Q_cliff ∈ R^(256 × 5)`). Rank neurons descending by `score`. The
top-k mask is `top_k_mask[j] = 1` if j ∈ top-k by score, else 0.

This is the "shared mechanism participant" ordering: neurons heavily
involved across multiple PCs rank higher than neurons concentrated in
a single PC. The alternative — per-PC top-k masks, then take the
union — biases toward neurons that appear in any one PC even if their
contribution to the other four is small. v3.2 prefers the aggregate
ranking; per-PC ranking is a v3.3 candidate if needed.

### 3.2 Patch protocol

For each (seed, direction) cell, the recording phase runs identically
to v3 axis-h (record per-step subspace coordinates `c = Q_cliff^T h`).
The injection phase differs:

```python
# v3 axis-h injection (reference):
delta_full = Q_cliff @ (target_c - current_c)   # (256,)
h_new = h + delta_full

# v3.2 axis-M injection:
delta_full = Q_cliff @ (target_c - current_c)   # (256,)
delta_masked = delta_full * top_k_mask          # zero out non-top-k entries
h_new = h + delta_masked
```

The mask is applied AFTER the full subspace delta is computed, so the
PC basis itself is unchanged. We are asking: "of the activation change
v3 would have applied to all 256 neurons, what fraction of behavior
do we recover by delivering only the change at the top-k neurons?"

### 3.3 k sweep

| k | as fraction of 256 | rationale |
| ---: | ---: | --- |
| 8 | 3.1% | top-of-rank stress test |
| 16 | 6.3% | |
| 32 | 12.5% | matches v3.1 Axis K top-32 L2-concentration metric |
| 64 | 25% | |
| 256 | 100% | sanity check; should equal v3 K=5 baseline |

### 3.4 Smoke gate

Before running the full sweep:

- Run k=32 with 8 seeds (`--seeds 8 --top-k 32`). ~3 minutes.
- Gate condition: **P→C median patch_success ≥ 0.37** (i.e., ≥ 40% of
  the v3 K=5 P→C baseline of 0.922).
- If passes: proceed to the full 64-seed k-sweep.
- If fails: stop. Top-k mediation with this neuron ranking is not
  feasible. Record the negative result and route v3.3 to non-linear
  attribution methods.

The smoke gate is *one-sided* on P→C only because v3.1 Axis J showed
basin-inducing generalizes more cleanly than basin-resisting; if the
mechanism is concentrated anywhere, it's most likely concentrated in
the P→C direction.

### 3.5 Optional follow-up: J1 and J2 with cliff-pair neuron mask

Gated on the cliff-pair sweep showing meaningful top-k mediation
(let's say top-32 recovers ≥ 60% of v3 K=5 P→C effect). If so:

- Reuse the cliff-pair top-k neuron mask.
- Run patch battery on J1 (`signature_terminal_medium` ↔
  `reward_lambda_1_0_medium_anchor`) and J2 (`mixed_lambda_0_9_medium_v3`
  ↔ `mixed_lambda_0_99_medium_v4`).
- Tests whether the cliff-pair-derived top-k neurons are the same
  neurons that mediate the basin-inducing subspace in held-out pairs.

Strong v3.2 outcome on the optional follow-up would be:
- Top-32 cliff-pair neurons mediate ≥ 60% of P→C patch effect on J1
  and J2 — the *basin-inducing subspace is mediated by a shared set
  of neurons across the controller family*.
- Top-32 cliff-pair neurons mediate ≤ 30% of C→P patch effect on J1
  and J2 — *consistent with v3.1's finding that basin-resisting is
  policy-specific*.

## 4. Pre-Registered Predictions

Three load-bearing predictions, each with explicit falsifier.

### 4.1 (Z1) Top-32 cliff-pair neurons mediate substantial P→C patch effect

Top-32 P→C median patch_success ≥ 0.5 (i.e., ≥ 54% of v3 K=5 P→C
baseline of 0.922).

**Falsifier:** top-32 P→C median < 0.2. Mechanism is non-linearly
distributed; routes v3.3 to non-linear attribution methods (zero-
ablation, integrated gradients).

### 4.2 (Z2) Basin-inducing / basin-resisting asymmetry carries into neuron-level mediation

At every k ∈ {8, 16, 32, 64}, top-k P→C recovery (fraction of v3 K=5
P→C effect) > top-k C→P recovery (fraction of v3 K=5 C→P effect).

**Falsifier:** at any k, C→P recovery exceeds P→C recovery by more
than 0.1. Would mean basin-resisting is *more* concentrated in
neuron-space than basin-inducing, which would invert v3.1's directional-
generalization reading and be a surprise worth investigating.

### 4.3 (Z3) Recovery curve saturates by k = 64

Top-64 P→C median patch_success is within 0.1 of top-256 (v3 K=5
baseline) P→C median. Analogous to v3's K=5 saturation in the PC
subspace.

**Falsifier:** top-64 still lags top-256 by more than 0.2. Mechanism
is distributed across more than a quarter of net.7's neurons; would
push v3.3 toward k > 64 and put pressure on the "moderately
concentrated" reading from v3.1 Axis K.

## 5. Cliff-Pair Manifest (no new policies)

| policy_id | label | tier | role |
| --- | --- | --- | --- |
| `mixed_lambda_0_95_medium_v4` | L-Mixed-M-λ=0.95 | Medium | protected side, cliff pair |
| `mixed_lambda_0_97_medium_v4` | L-Mixed-M-λ=0.97 | Medium | collapsed side, cliff pair |

For optional J-follow-up (gated):

| policy_id | label | tier | role |
| --- | --- | --- | --- |
| `signature_terminal_medium` | L-Sig-Terminal-M | Medium | J1 protected side |
| `reward_lambda_1_0_medium_anchor` | L-Reward-M λ=1.0 | Medium | J1 collapsed side |
| `mixed_lambda_0_9_medium_v3` | L-Mixed-M λ=0.9 | Medium | J2 protected side |
| `mixed_lambda_0_99_medium_v4` | L-Mixed-M λ=0.99 | Medium | J2 collapsed side |

## 6. Metrics

For each (cliff pair × k × direction × seed):
- `patch_success_mean / median / ratio_of_means` — same as v3 / v3.1.
- `fraction_of_v3_baseline` = patch_success / v3 K=5 baseline in that direction.

Per-k aggregates:
- mean / median / ratio across 64 seeds in both directions.
- `recovery_fraction_P→C` and `recovery_fraction_C→P`.

Per-run diagnostics:
- `top_k_neuron_indices` (the actual neuron IDs that were patched).
- `top_k_score_total` (sum of `score[j]` for j ∈ top_k) — the fraction
  of total PC L2 captured by the mask. Useful for "did patching X% of
  L2 deliver X% of mechanism" linearity question.

## 7. Harness Extension

Add to `training/mesa/phase6_v2_sae.py`:

```python
def compute_neuron_ranking(Q_cliff: np.ndarray) -> np.ndarray:
    """Return neuron indices sorted descending by aggregate L2
    contribution across all columns of Q_cliff."""
    scores = (Q_cliff ** 2).sum(axis=1)  # (d_in,) — aggregate L2 per neuron
    return np.argsort(-scores)  # descending


def run_neuron_restricted_patch_battery(
    Q_np: np.ndarray,
    neuron_mask: np.ndarray,    # (d_in,) {0, 1}, 1 = patch this neuron
    label: str,
    seed_start: int, seeds: int, horizon: int,
    layer: str, out_dir: Path,
    manifest_extra: dict[str, Any],
    protected_spec: PolicySpec = CLIFF_PROTECTED,
    collapsed_spec: PolicySpec = CLIFF_COLLAPSED,
) -> None:
    """Identical to run_subspace_patch_battery, but the injection hook
    masks the post-projection delta to top-k neurons before applying."""
    ...


def axis_m_neuron_mediation(args: argparse.Namespace) -> None:
    """Load cliff-pair PCA basis, compute neuron ranking, run patch
    battery with top-k mask."""
    # Load Q_cliff from the cached canonical n=64 cliff-pair PCA basis
    # Compute neuron ranking via compute_neuron_ranking(Q_cliff)
    # Build top_k_mask from args.top_k
    # Run run_neuron_restricted_patch_battery
```

CLI:

```bash
# Cliff pair, k sweep
python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 32 --seeds 8     # smoke

python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 8 --seeds 64
python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 16 --seeds 64
python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 32 --seeds 64
python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 64 --seeds 64
python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 256 --seeds 64    # sanity, should equal v3 K=5

# Optional J-follow-up (gated)
python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 32 --seeds 64 --pair J1
python -m training.mesa.phase6_v2_sae axis-m-neuron-mediation \
    --top-k 32 --seeds 64 --pair J2
```

Total harness extension: ~80 LOC (neuron ranking helper, the
restricted patch battery wrapping `run_subspace_patch_battery`, CLI
subcommand, output schema).

## 8. Outputs

```
results/mesa/phase6-v3-2-neuron-mediation/
  manifest.json
  axis-m-cliff-pair/
    top-8/
      patch.csv
      patch-aggregate.csv
    top-16/
    top-32/
    top-64/
    top-256/
    recovery-curve.csv         # k → P→C recovery, C→P recovery
    neuron-ids-top-32.csv      # the actual neuron IDs in top-32
  axis-m-J1/                   # optional
    top-32/
  axis-m-J2/                   # optional
    top-32/
  reports/
    summary.json               # Z1-Z3 outcomes
    asymmetry-table.csv        # k × direction × patch_success
```

## 9. Execution Order

Recommended sequencing:

1. **Harness extension lands first.** `compute_neuron_ranking`,
   `run_neuron_restricted_patch_battery`, `axis_m_neuron_mediation`
   added to `phase6_v2_sae.py`. Smoke-compile.
2. **Neuron ranking dry-run.** Load Q_cliff, compute ranking, print
   top-8 / top-32 neuron IDs to console. Cross-check against v3.1
   Axis K's per-PC top-8 lists — should overlap.
3. **k=32 smoke at 8 seeds.** ~3 min. If P→C median patch_success
   ≥ 0.37, proceed; else stop.
4. **Full cliff-pair k-sweep.** 5 runs at 64 seeds each (k = 8, 16,
   32, 64, 256). ~25-40 min total.
5. **Recovery-curve plot + report.** Cross-policy aggregator emits
   `recovery-curve.csv` and a brief summary.
6. **Gate on optional J-follow-up.** If top-32 P→C recovers ≥ 60% of
   v3 K=5 baseline on cliff pair, run J1 and J2 with the cliff-pair
   top-32 mask. Each ~7 min.
7. **Phase 6 v3.2 result note** at `docs/mesa/PHASE6_V32_RESULTS.md`.

Total wall-clock: ~45-90 minutes depending on whether J-follow-up
runs.

## 10. Exit Criterion

Phase 6 v3.2 complete when:

- `axis-m-neuron-mediation` lands in the harness with all required
  CLI args.
- Neuron ranking is computed and the top-32 neuron IDs are recorded.
- Smoke gate result is recorded (pass or fail).
- If smoke passes: full k-sweep (k ∈ {8, 16, 32, 64, 256}) on cliff
  pair × both directions × 64 seeds completes.
- Z1-Z3 predictions are classified as confirmed or falsified.
- If cliff-pair top-32 P→C recovers ≥ 60% of v3 K=5 baseline:
  optional J1/J2 runs complete with the cliff-pair top-32 mask.
- v3.2 result note (`PHASE6_V32_RESULTS.md`) is written with the
  recovery curve and asymmetry diagnostics as the headline.

## 11. Cross-References

- **Phase 6 v3.1 spec / results:** the entanglement and asymmetric-
  generalization findings v3.2 builds on.
  [`PHASE6_V31_SPEC.md`](PHASE6_V31_SPEC.md), `PHASE6_V31_RESULTS.md`.
- **Phase 6 v2/v3 spec / results:** the 5-dim PCA basis v3.2 reuses
  unchanged. [`PHASE6_V2_SPEC.md`](PHASE6_V2_SPEC.md),
  [`PHASE6_V2_RESULTS.md`](PHASE6_V2_RESULTS.md).
- **Phase 6 v1 spec / results:** the layer-level localization.
  [`PHASE6_SPEC.md`](PHASE6_SPEC.md),
  [`PHASE6_RESULTS.md`](PHASE6_RESULTS.md).
- **Roadmap:** [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md).

## 12. What v3.3 and Phase 7 v2 Inherit

If Z1-Z3 all confirm and the J-follow-up shows shared-neuron mediation
on basin-inducing but not basin-resisting:

- **The gravity claim's mechanistic anchor ratchets a third time.**
  "5-dim subspace at net.7" → "5-dim subspace at net.7 mediated by a
  named subset of top-k neurons; basin-inducing is shared across the
  family, basin-resisting is policy-specific." Cascade into
  PROMO_HIGHLIGHTS, claims-and-scope, SUNDOG_V_MESA, mesa.html.
- **v3.3 work routes to:**
  - Small-tier PCA basis derivation (enables J3 cross-tier from
    v3.1).
  - Per-policy basin-resisting subspace computation (PCA on
    diffs *within* a single protected policy under perturbation
    rather than across policies).
  - Adversarial neuron-ablation: can the basin attractor be removed
    by *editing* the cliff-pair top-k neurons in a collapsed policy,
    not just patching them at inference?
- **mesa.html §The Locus gains a neuron-grid sub-panel** showing the
  top-32 cliff-pair neurons and their per-PC contributions.

If Z1 falsifies (top-32 cliff-pair P→C median < 0.2):

- v3.2 records the negative result: the 5D subspace is mediated by
  a non-linearly-distributed neuron pattern that linear top-k
  ablation can't reproduce.
- v3.3 routes to non-linear attribution methods: zero-ablation
  attribution scores per neuron, integrated gradients on net.7 →
  action logits, or causal scrubbing.

## 13. Versioning

- **v3.2 (2026-05-12)** — initial pin. One axis (Axis M, top-k
  neuron mediation). Three pre-registered predictions Z1-Z3. Cliff
  pair as central artifact; J1/J2 follow-up gated on cliff-pair
  result. Neuron ranking is aggregate L2 across PCs 1-5; alternative
  rankings (per-PC, signed contribution) deferred to v3.3.
