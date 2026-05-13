# Mesa Phase 6 v2 — Direction-based Mechanistic Probing

This document is the implementation-grade spec for Phase 6 v2 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 6 v1 localized the
cliff causally to the actor's final hidden activation (`net.7`); the
v1 Axis A linear-probe maps failed to dissociate the cliff pair under
two target designs (endpoint-shaped behavior target, ΔR² depth
profile), leaving open the question of *what specific structure* in
the 256-dim `net.7` activation space carries the basin attractor.

Phase 6 v2 asks whether the cliff is a single direction inside `net.7`
or a distributed property of that layer, and whether direction-level
patching tightens P4 from single-layer to single-direction
localization. The methodology is sparse-autoencoder (SAE) feature
dictionary extraction at `net.7`, paired with direction-based
activation patching using the SAE feature most correlated with the
Phase 4 basin-attraction readout.

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6 v2.

## 1. Decision Lock

Phase 6 v2 starts with seven pinned calls:

- **Two axes only.** Axis D: SAE feature dictionary at `net.7` trained
  jointly across the cliff pair. Axis E: direction-based patching
  using the SAE feature most correlated with basin attraction. CKA /
  RSA between the cliff pair at all layers, per-neuron causal
  mediation, and cross-architecture probes are deferred to v3.
- **The cliff pair is the central artifact.** L-Mixed-M-`λ=0.95` vs
  L-Mixed-M-`λ=0.97`, same matched-seed slate (10000-10063) from v1.
  Other Phase 5 zoo policies appear as held-out evaluation only, not
  as SAE training data.
- **No new PPO training runs.** Phase 6 v2 reuses the Phase 5 zoo
  policy.json files as-is. The only new training is the SAE itself.
- **No new environment changes.** The shadow-field env, probe
  affordances, and intervention battery from Phases 3-4 are
  unchanged. Phase 6 v2 introduces only a new Python harness
  (`training/mesa/phase6_v2_sae.py`) on top of the existing
  `phase6_probes.py` activation-collection scaffold.
- **SAE architecture pinned to top-k.** Input dimension 256 (the
  `net.7` hidden width), expansion factor 4 (1024 features), top-k
  sparsity with `k=32` (≈3% activation density). L1-sparsity SAE is
  the v2.1 fallback if top-k training is unstable.
- **Single SAE shared across the cliff pair.** A single autoencoder
  is trained on the union of `net.7` activations from L-Mixed-M-λ=0.95
  and L-Mixed-M-λ=0.97. This gives a shared feature basis in which the
  two policies' activations can be directly compared. Separate
  per-policy SAEs are deferred to v3 (would answer "is the basin
  feature *unique* to the collapsed policy" but adds a methodological
  layer v2 doesn't need).
- **Direction-patching uses the SAE feature with highest correlation
  to the per-episode `basin_pref_intervened` outcome.** Multiple-feature
  patching (top-k features by correlation) is the v2.1 escalation if
  single-feature patching fails P4 threshold.

Total v2 compute: 0 new PPO training runs, 1 new Python harness, ~30-60
minutes for SAE training + direction-patching battery on the cliff
pair.

## 2. Scope

Phase 6 v2 owns:

- New harness `training/mesa/phase6_v2_sae.py` (SAE training on
  pre-collected activations + direction-patching runner on the cliff
  pair).
- Activation collection extension: ensure `phase6_probes.py` saves
  `net.7` activations to disk for the cliff pair across the matched
  seed slate (v1 may not have persisted activations; if not, re-run
  collection step before SAE training).
- SAE training over the joint cliff-pair `net.7` activations.
- Feature labeling: for each SAE feature, compute its correlation
  with per-episode `basin_pref_intervened` outcome (from the v1 Phase 4
  intervention battery).
- Direction-based patching at `net.7` using the top-correlated feature,
  in both directions (protected → collapsed and reverse), full
  matched-seed slate.
- Phase 6 v2 aggregate report with: SAE quality metrics, feature
  labeling table, direction-patch-success table, single-direction
  vs single-layer comparison against v1 baseline.

Phase 6 v2 does **not** own:

- New PPO training runs.
- SAE training on policies outside the cliff pair (deferred to
  v3 — natural extension if the basin-direction story holds).
- Per-policy SAEs (deferred to v3).
- CKA / RSA between cliff pair across all layers (deferred to v3 —
  the orthogonal "is the cliff pair similar everywhere except
  net.7" question).
- Per-neuron causal mediation (deferred to v3 — finer-grained
  granularity than directions; only worth running if direction-based
  patching fails P4).
- Cross-architecture probes (deferred indefinitely).
- Phase 7 v2 cross-product expansion.

## 3. Axes

### 3.1 Axis D — SAE feature dictionary at net.7

The descriptive axis. Train a single sparse autoencoder on the
joint cliff-pair `net.7` activations, extract a feature dictionary, and
label each feature by its correlation with the Phase 4
basin-attraction outcome.

**Activation source:**

For each policy in the cliff pair (P_protected = L-Mixed-M-λ=0.95,
P_collapsed = L-Mixed-M-λ=0.97):

1. Run 64 episodes on the matched seed slate (10000-10063) under
   nominal Phase 0 configuration. Record `net.7` activations at every
   step.
2. Compute per-episode `basin_pref_intervened(s)` by running a paired
   episode under live x_false intervention (Phase 4 protocol).

Total activations: ~32K per policy × 2 policies = ~64K activation
vectors of dimension 256. Concatenate.

**SAE architecture:**

```
encoder: Linear(256, 1024)
activation: TopK(k=32, dim=-1)      # zero out all but top 32 features
decoder: Linear(1024, 256)          # no bias on decoder per common SAE practice
```

Loss: mean-squared reconstruction error on the activation. No
auxiliary loss for top-k SAEs (sparsity is enforced architecturally).
L1-sparsity variant (`α · ||z||_1` auxiliary term with α=1e-3) is the
v2.1 fallback if top-k training is unstable.

**Training:**

- Optimizer: Adam with `lr=1e-3`, `betas=(0.9, 0.999)`.
- Batch size: 512 activation vectors.
- Training steps: 10000 (≈80 epochs over the 64K activation set).
- Weight initialization: Kaiming uniform on both layers.
- Tied weights: not used in v2 (encoder and decoder are independent;
  tied-weight variants are v3 work).

**Feature labeling protocol:**

For each SAE feature `f ∈ {0, ..., 1023}`:

1. For every (policy, seed, step) triple, compute the feature
   activation `z_f[policy, seed, step]`.
2. Aggregate per (policy, seed) to a scalar by max-over-step:
   `Z_f[policy, seed] = max_t z_f[policy, seed, t]`. (Max chosen over
   mean because basin-attraction is a per-episode binary-ish event;
   mean smooths it out.)
3. Compute Pearson correlation between `Z_f` and the per-episode
   `basin_pref_intervened` outcome across all (policy, seed) pairs.
4. Rank features by |correlation|. Report the top 10 features with
   their correlation magnitudes and signs.

**Quality metrics (SAE health checks):**

- **Reconstruction R²** on a held-out 20% of activation vectors. Pass
  threshold: R² > 0.8. Below: top-k may be too sparse; fall back to
  v2.1 L1-sparsity SAE.
- **Dead-feature rate**: percentage of features that never activate
  on the training set. Pass threshold: <30% dead. Above: expansion
  factor is too high; fall back to 2x (512 features).
- **Active-feature rate**: mean fraction of features that pass the
  top-k threshold per token. Should equal `k / n_features = 32 / 1024
  ≈ 3.1%` by construction; deviation indicates a broken top-k
  implementation.

**Tier coverage:** Medium only (the cliff pair is Medium-tier).

### 3.2 Axis E — Direction-based patching

The causal axis. Take the SAE feature with the highest |correlation|
to `basin_pref_intervened` (call it `f*`) and patch only that direction
between the cliff pair.

**Direction definition:**

`f*` corresponds to a row of the SAE encoder weight matrix
`W_enc[f*, :] ∈ R^256` and a column of the decoder
`W_dec[:, f*] ∈ R^256`. The encoder row is the *detection* direction
("what does the policy look like when this feature is active");
the decoder column is the *write* direction ("what does this feature
contribute to the layer's output").

For patching, use the decoder direction (the "write" direction) — this
is the direction the policy *adds to net.7* when the feature is
active. Patching the decoder direction therefore directly substitutes
the protected/collapsed policy's basin-attribution contribution.

**Patch protocol:**

For each seed `s` and direction (P→C / C→P):

1. **Forward A — clean protected.** Run `P_protected` on seed `s`
   with live x_false intervention (Phase 4 protocol). Record full
   `net.7` activation `h_A[t]` for every step `t`. Compute the
   scalar projection onto the basin-direction:
   `α_A[t] = ⟨h_A[t], W_dec[:, f*]⟩ / ||W_dec[:, f*]||²`.
2. **Forward B — clean collapsed.** Same with `P_collapsed`. Compute
   `α_B[t]`.
3. **Forward C — direction-patched (protected → collapsed).** Run
   `P_collapsed` on seed `s`. At each step `t`, compute
   `α_C[t] = ⟨h_C[t], W_dec[:, f*]⟩ / ||W_dec[:, f*]||²` and
   substitute the protected policy's projection:
   `h_C_patched[t] = h_C[t] + (α_A[t] - α_C[t]) · W_dec[:, f*] /
   ||W_dec[:, f*]||`. Continue forward with `h_C_patched[t]` into
   the policy head. Record `old_basin_pref_C`.
4. **Forward D — direction-patched (collapsed → protected).** Reverse.

Note: this is *direction substitution along a fixed basis vector*, not
full layer activation swap. All other 255 dimensions of `net.7` are
left untouched. If the v1 layer-level patch_success was 0.89 / 0.93
and the direction-level patch_success is comparable, the basin
attractor is the single direction `W_dec[:, f*]`. If direction-level
patch_success drops substantially, the basin attractor is distributed
across multiple features.

**Patch-success metric:**

Identical to v1:

```
patch_success_AC = (obp_B − obp_C_dir) / (obp_B − obp_A)
patch_success_DB = (obp_A − obp_D_dir) / (obp_A − obp_B)
```

Report mean, median, and ratio-of-means as in v1 (heavy-tail
discipline carried forward).

**Tier coverage:** Medium cliff pair only.

## 4. Pre-Registered Predictions

Four load-bearing predictions, all checkable from Phase 6 v2 harness
outputs.

### 4.1 (V1) SAE training is healthy

Reconstruction R² > 0.8 on held-out activations, dead-feature rate
< 30%, active-feature rate within ±0.005 of `32/1024 = 0.03125`. If
the v2 top-k SAE fails these health checks, fall back to v2.1
L1-sparsity SAE and re-run Axis D.

**Falsifier:** R² ≤ 0.7 even after v2.1 fallback. Would suggest
`net.7` activations are not well-described by sparse linear features
in a 1024-dim dictionary; would route v3 work to dictionary-learning
methods that don't assume sparsity (e.g., dense PCA + per-component
labeling).

### 4.2 (V2) A basin-attraction feature exists

Among the 1024 SAE features, at least one feature has |correlation|
with `basin_pref_intervened` ≥ 0.5. The top-correlated feature should
visibly distinguish the cliff pair: high mean activation on
L-Mixed-M-λ=0.97 (collapsed) seeds, low mean activation on
L-Mixed-M-λ=0.95 (protected) seeds, or vice versa.

**Falsifier:** the top feature's |correlation| < 0.3. Would suggest
the basin attractor is not linearly decodable as a single SAE
direction; route v3 to multi-feature representations or nonlinear
feature labeling.

### 4.3 (V3) Direction patching achieves patch_success > 0.5

In at least one direction (P→C or C→P), direction-based patching at
`net.7` using only `W_dec[:, f*]` achieves median patch_success > 0.5.
This is the load-bearing v2 prediction: it says the cliff is at least
*partially* localizable to a single direction.

**Falsifier:** both directions show median patch_success < 0.3. Would
mean the basin attractor lives in a multi-feature subspace at `net.7`;
v3 would extend to multi-direction patching (top-k correlated
features).

### 4.4 (V4) Direction patching is competitive with layer patching

The direction-patch median patch_success is within 0.2 of the v1
layer-patch median (0.944 P→C, 0.860 C→P). This is the *strong* v2
prediction: it says the cliff is *fully* localizable to a single
direction, not just partially.

**Falsifier:** direction-patch median lags layer-patch median by more
than 0.3 in both directions. Would mean the basin attractor is a
direction *and* an interaction effect (the direction explains some of
the cliff, the rest is distributed across features). This is a real
intermediate finding worth reporting and would route v3 to the
"direction + residual" framing.

## 5. Policy Zoo Manifest

Phase 6 v2 reuses the Phase 5 v4 cliff pair only:

| policy_id | label | tier | lambda | phase6_v1_annotation |
| --- | --- | --- | ---: | --- |
| `mixed_lambda_0_95_medium_v4` | L-Mixed-M-λ=0.95 | Medium | 0.95 | net7_localized:protected_side |
| `mixed_lambda_0_97_medium_v4` | L-Mixed-M-λ=0.97 | Medium | 0.97 | net7_localized:collapsed_side |

Other Phase 5 zoo policies are read-only references; they may be used
to evaluate held-out generalization of the SAE feature dictionary
(e.g., does feature `f*` activate strongly on L-Reward-M, which is
also collapsed? Does it stay quiet on L-Sig-Terminal-M, which is
held?), but they are not part of the SAE training set and not part of
the Axis E patch battery.

## 6. Metrics

### 6.1 Axis D — SAE feature dictionary

- `reconstruction_r2_test`: SAE held-out R² (≥ 0.8 target).
- `dead_feature_rate`: fraction of features with zero activations on
  the training set (< 30% target).
- `active_feature_rate`: mean fraction of features above the top-k
  threshold per token (≈ 0.0312 target by construction).
- `feature_correlation[f]`: Pearson correlation between feature `f`'s
  max-over-step activation and per-episode `basin_pref_intervened`,
  computed across all (policy, seed) pairs in the cliff pair.
- `top10_basin_features`: the 10 features with highest |correlation|,
  reported with their correlation magnitude, sign, mean activation
  per policy, and a short ad-hoc description (computed from feature
  examples — e.g., "fires strongly on seeds with terminal position
  near (-2.5, -2.5)").

### 6.2 Axis E — Direction patching

For each (seed, direction) cell:

- `obp_A_protected_clean`, `obp_B_collapsed_clean`: baseline values
  (carried over from v1 Phase 4 intervention battery on the cliff
  pair).
- `obp_C_direction_patched_pc`, `obp_D_direction_patched_cp`: post-
  direction-patch readout.
- `patch_success_AC_dir`, `patch_success_DB_dir`: normalized as in v1.

Aggregate across seeds: mean / median / ratio-of-means
patch_success per direction, with seed-bootstrap 95% CI.

### 6.3 Phase 6 v2 cross-comparison

The headline table:

| metric | v1 layer-patch (net.7) | v2 direction-patch (f*) |
| --- | --- | --- |
| P→C mean | 0.894 | tbd |
| P→C median | 0.944 | tbd |
| P→C ratio-of-means | 0.899 | tbd |
| C→P mean | 0.934 | tbd |
| C→P median | 0.860 | tbd |
| C→P ratio-of-means | 0.854 | tbd |

A small gap (≤ 0.2 by V4) ratifies single-direction localization. A
large gap motivates v3 multi-direction work.

## 7. Harness — `training/mesa/phase6_v2_sae.py`

A new Python harness. Sketch of structure:

```python
# training/mesa/phase6_v2_sae.py
import argparse, json, pathlib
import torch
import torch.nn as nn
import numpy as np

from .phase6_probes import (
    collect_net7_activations,           # reuse v1 helper
    run_with_basin_intervention,        # reuse v1 Phase 4 protocol
)

class TopKSAE(nn.Module):
    def __init__(self, d_in, n_features, k):
        super().__init__()
        self.encoder = nn.Linear(d_in, n_features)
        self.decoder = nn.Linear(n_features, d_in, bias=False)
        self.k = k

    def forward(self, x):
        z = self.encoder(x)
        # Top-k along feature dim
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        recon = self.decoder(z_sparse)
        return recon, z_sparse

def train_sae(activations, n_features=1024, k=32, steps=10000):
    """Train top-k SAE on stacked net.7 activations from cliff pair."""
    sae = TopKSAE(d_in=activations.shape[1], n_features=n_features, k=k)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    # 80/20 train/test split
    n = activations.shape[0]
    idx = torch.randperm(n)
    train, test = activations[idx[:int(0.8*n)]], activations[idx[int(0.8*n):]]
    for step in range(steps):
        batch_idx = torch.randint(0, len(train), (512,))
        x = train[batch_idx]
        recon, z = sae(x)
        loss = ((recon - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                recon_test, _ = sae(test)
                r2 = 1 - ((recon_test - test) ** 2).sum() / ((test - test.mean(0)) ** 2).sum()
                print(f"step {step}: train_loss={loss.item():.4f} test_r2={r2.item():.4f}")
    return sae

def label_features(sae, activations_per_policy_seed, basin_pref_per_seed):
    """For each feature, compute correlation with basin_pref_intervened."""
    feature_acts = []  # [n_features, n_episodes]: max-over-step activation
    for (policy, seed), activations in activations_per_policy_seed.items():
        with torch.no_grad():
            _, z = sae(activations)  # [n_steps, n_features]
            max_act = z.max(dim=0).values  # [n_features]
        feature_acts.append(max_act.numpy())
    feature_acts = np.stack(feature_acts, axis=1)  # [n_features, n_episodes]
    targets = np.array([basin_pref_per_seed[(p, s)] for (p, s) in keys])
    correlations = np.array([
        np.corrcoef(feature_acts[f], targets)[0, 1]
        for f in range(sae.encoder.out_features)
    ])
    return correlations

def direction_patch(p_protected, p_collapsed, sae, feature_idx, seed,
                    x_false_intervention=(2.5, 2.5)):
    """Patch only the SAE feature_idx direction at net.7 between policies."""
    direction = sae.decoder.weight[:, feature_idx]  # [d_in]
    direction_unit = direction / direction.norm()

    # Forward A: clean protected, recording per-step net.7 projection onto direction
    cache_A = run_with_recording_and_projection(p_protected, "net.7", direction_unit, seed)
    # Forward B: clean collapsed, recording
    cache_B = run_with_recording_and_projection(p_collapsed, "net.7", direction_unit, seed)
    # Forward C: collapsed with protected's projection injected
    readout_C = run_with_direction_injection(
        p_collapsed, "net.7", direction_unit, cache_A.projections, seed,
        x_false=x_false_intervention,
    )
    # Forward D: protected with collapsed's projection injected
    readout_D = run_with_direction_injection(
        p_protected, "net.7", direction_unit, cache_B.projections, seed,
        x_false=x_false_intervention,
    )
    return {
        "seed": seed, "feature_idx": feature_idx,
        "obp_A": cache_A.obp, "obp_B": cache_B.obp,
        "obp_C_dir": readout_C.obp, "obp_D_dir": readout_D.obp,
    }
```

Key implementation notes:

- **Activation collection reuses v1.** If `phase6_probes.py` doesn't
  already persist `net.7` activations to disk for the cliff pair,
  this is the first thing to fix. The v2 harness assumes activations
  are loadable from a known cache path.
- **Direction injection via PyTorch hooks.** The hook modifies `net.7`'s
  forward output to: `h_new = h_current + (α_target - α_current) ·
  direction_unit` where `α_current = ⟨h_current, direction_unit⟩`
  and `α_target` is the recorded projection from the other policy's
  forward pass on the same seed.
- **Seed determinism**: numpy random state seeded from the env's
  splitmix64 stream, not from Python's global. Run-to-run
  reproducibility is required for direction-patch metrics to match.
- **Compute budget**: SAE training ~15 min on CPU with 64K
  activations. Direction-patch battery ~5 min for the cliff pair × 64
  seeds × 4 forward passes × 2 directions. Total v2 wall-clock ≈
  30-60 min.

## 8. Outputs

```
results/mesa/phase6-v2-direction/
  manifest.json
  sae-config.json                       # architecture + training hyperparams
  sae-weights.pt                        # trained SAE state dict
  axis-d-sae-quality.json               # R², dead-rate, active-rate
  axis-d-feature-correlations.csv       # per-feature corr with basin_pref
  axis-d-top10-basin-features.csv       # ranked top features with descriptions
  axis-e-direction-patch.csv            # (seed, direction) → patch_success
  axis-e-direction-patch-aggregate.csv  # (direction) → mean + 95% CI
  reports/
    v1-vs-v2-comparison.csv             # layer-patch vs direction-patch table
    feature-fingerprint-protected.png   # optional: feature activation heatmap
    feature-fingerprint-collapsed.png   # optional
```

## 9. Execution Order

Recommended sequencing for Phase 6 v2:

1. **Verify activation cache exists.** Check that `net.7` activations
   for L-Mixed-M-λ=0.95 and λ=0.97 across the matched seed slate are
   persisted from v1. If not, run a collection pass first (~3 min per
   policy).
2. **SAE smoke train.** 1000-step quick train to verify the loss
   decreases and reconstruction R² is non-trivial. ~2 min.
3. **Axis D full train.** 10000-step SAE training. ~15 min.
4. **Axis D health-check pass.** Verify reconstruction R² > 0.8,
   dead-rate < 30%. If fails, fall back to v2.1 L1-sparsity SAE.
5. **Axis D feature labeling.** Compute correlations, rank features,
   identify `f*`. ~1 min.
6. **Axis E direction-patch smoke.** 8 seeds, single direction. ~1
   min. Verify patch mechanics work and `f*`'s decoder direction
   produces non-trivial behavior shift.
7. **Axis E direction-patch full battery.** 64 seeds × both
   directions. ~5 min.
8. **Comparison table generation.** v1 vs v2 patch success.
9. **Phase 6 v2 result note** at `docs/mesa/PHASE6_V2_RESULTS.md`,
   following the PHASE6_RESULTS template.

Total wall-clock: ~30-60 min v2 first pass.

## 10. Exit Criterion

Phase 6 v2 complete when:

- `phase6_v2_sae.py` lands and produces all eight CSVs / JSONs above.
- SAE health checks pass (R² > 0.8, dead-rate < 30%) on v2 or v2.1.
- Axis D feature-correlation CSV identifies a top basin-attraction
  feature `f*`.
- Axis E direction-patch CSV covers full matched seed slate × both
  directions.
- The v1-vs-v2 comparison table is generated and inspected for
  P4-tightening (V4 prediction).
- At least two of the four pre-registered predictions in §4 are
  confirmed or falsified.
- Phase 6 v2 result note (`docs/mesa/PHASE6_V2_RESULTS.md`) is
  written with the headline finding.

## 11. Cross-References

- **Phase 0 spec**: env spec, signature definition, false-basin
  configuration. [`PHASE0_SPEC.md`](PHASE0_SPEC.md).
- **Phase 4 spec / results**: basin-position intervention, the
  `basin_pref_intervened` outcome v2 correlates SAE features against.
  [`PHASE4_SPEC.md`](PHASE4_SPEC.md), [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md).
- **Phase 5 spec / results**: L-Mixed protection curve, cliff
  localization at `λ ≈ 0.953`.
  [`PHASE5_RESULTS.md`](PHASE5_RESULTS.md).
- **Phase 6 v1 spec / results**: layer-level activation patching,
  the `net.7` localization v2 is sharpening.
  [`PHASE6_SPEC.md`](PHASE6_SPEC.md), [`PHASE6_RESULTS.md`](PHASE6_RESULTS.md).
- **Phase 7 v1 results**: operating envelope with `net.7` annotation
  on the cliff pair. [`PHASE7_RESULTS.md`](PHASE7_RESULTS.md).
- **Roadmap**: [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md).

## 12. What Phase 7 v2 and Phase 8 v2 Inherit

If Phase 6 v2 lands a single-direction localization (V3 and V4
confirmed), three downstream consumers benefit:

- **Phase 7 v2 envelope:** the `phase6_annotation` column in
  `policies-inventory.csv` upgrades from `net7_localized:*` to
  `net7_feature_f<index>:*`. The mechanistic locus claim in the
  envelope map sharpens.
- **Phase 8 v2 public artifact:** `mesa.html` gains a "fingerprint"
  surface in §The Locus showing the basin-direction activation
  heatmap across the protected and collapsed policies. This is the
  visual story for "the cliff is a specific direction in activation
  space," which is mechanistically tighter than the v1 "the cliff is
  at this layer."
- **v3 generalization:** the basin-direction can be probed on
  held-out Phase 5 zoo policies. If `f*` activates strongly on
  L-Reward-M (also collapsed) and stays quiet on L-Sig-Terminal-M
  (held), the direction is *not* specific to the cliff pair — it's
  the basin-attraction feature of the controller family. That would
  ratchet the gravity claim further: not only does the cliff pair
  share a single mechanistic locus, but the *whole collapse pocket*
  shares the same locus.

If Phase 6 v2 falsifies the single-direction story (V3 falsifier),
v3 work is sequenced as: per-neuron causal mediation at `net.7` to
test "the cliff is a few specific neurons," then CKA/RSA between the
cliff pair at all layers to test "the cliff is entirely at net.7,"
then multi-feature direction patching to test "the cliff is a small
subspace at net.7." Each step adds methodological discipline at the
cost of more complex tooling.

## 13. Versioning

- **v2 (2026-05-12)** — initial pin. Two axes (SAE feature
  dictionary at net.7, direction-based patching with f*). Four
  pre-registered predictions (V1-V4). Cliff pair only; other zoo
  policies held out. Top-k SAE pinned; L1-sparsity is the v2.1
  fallback. Single SAE shared across cliff pair; per-policy SAEs
  deferred to v3. CKA/RSA, per-neuron mediation, multi-feature
  patching all deferred to v3.
