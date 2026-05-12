# Mesa Phase 6 — Interpretability Probes

This document is the implementation-grade spec for Phase 6 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 5 v4 produced a
sharp empirical cliff in the L-Mixed protection curve at Medium tier:
λ = 0.95 stays protected against the Phase 4 basin-position
intervention; λ = 0.97 collapses into a fossilized basin attractor. The
cliff localizes the gravity claim to a 0.02-wide window in objective
mixing weight. Phase 6 opens the box: where, mechanically, does the
basin attractor live in the policy weights, and what computational
structure distinguishes the protected side of the cliff from the
collapsed side?

Where this spec and the roadmap disagree, the roadmap wins. Where both
are silent, this spec is authoritative for Phase 6.

## 1. Decision Lock

Phase 6 v1 starts with six pinned calls:

- **Axis B carries v1.** Axis A linear probes remain as smoke-tested
  descriptive context only. The v1.2 behavior target and v1.5 ΔR²
  depth-profile target both failed their smoke gates, so the full-zoo
  Axis A sweep is deferred to Phase 6 v2 alongside sparse autoencoders
  or direction-based probing. Axis B activation patching across the
  cliff is the v1 load-bearing artifact.
- **The cliff is the central artifact.** Phase 6 v1 is organized around
  the L-Mixed-M-λ=0.95 vs L-Mixed-M-λ=0.97 paired-policy comparison.
  Other policies in the zoo serve as reference points (zoo extrema, HC
  floors, Oracle ceilings) but the cliff pair carries the load-bearing
  weight.
- **No new training runs in v1.** Phase 6 reuses the existing Phase 5
  zoo of trained policies as-is. New training (sparse-autoencoder pretrain,
  basin-suppression fine-tune) is deferred.
- **No new environment changes.** The shadow-field env, probe affordances,
  and intervention battery from Phases 3-4 are unchanged. Phase 6
  introduces only a new harness (Python, PyTorch-side hooks).
- **Axis A is deferred after failed smokes.** The v1.5 ΔR² profile is
  retained as a diagnostic CSV format, but not a claim surface. Full
  policy-zoo probing waits for v2, likely with sparse-autoencoder or
  direction-based targets that avoid endpoint and input-geometry
  confounds.
- **Patch granularity is pinned to layer × seed.** Per-neuron and
  per-attention-head granularity are deferred to v2; the MLP architecture
  used here is small enough that layer-level patching answers the
  load-bearing question without needing finer resolution.

Total v1 compute: 0 training runs, 1 Python hook harness, ~10-30
minutes for the cliff-pair activation patch battery plus smoke
artifacts already recorded.

## 2. Scope

Phase 6 v1 owns:

- New harness `training/mesa/phase6_probes.py` (PyTorch forward hooks +
  smoke-level linear-probe fitting + activation-patching runner).
- Smoke-level hidden-activation collection across a fixed eval seed slate
  (reuses Phase 3 seed range 10000-10063 for matched-seed discipline).
- Activation patching across the cliff pair (L-Mixed-M-λ=0.97 ←
  L-Mixed-M-λ=0.95 and the reverse) at layer granularity, with the
  basin-position intervention from Phase 4 as the behavioral readout.
- Phase 6 aggregate report producing the Axis A smoke-failure note, the
  patch-success table, and the minimal-patch summary.

Phase 6 does **not** own:

- New PPO training runs.
- Sparse-autoencoder feature dictionaries (deferred to v2).
- Per-neuron or per-head causal mediation (deferred to v2).
- Full-zoo Axis A linear-probe sweep (deferred to v2 after failed v1.2
  and v1.5 smoke gates).
- Cross-architecture probes (e.g., larger MLPs, transformers; deferred
  indefinitely).
- Texture-channel or sensor-delay probe features (deferred to v2).
- Phase 7 operating-envelope cross-product.

## 3. Axes

### 3.1 Axis A — Linear-probe maps

The descriptive axis, deferred from the v1 claim after smoke failures.
The v1 harness can still fit a linear probe from each hidden layer's
activations to the four pinned feature families. The result is a
`layer × feature × policy` accuracy tensor that says:
"layer L of policy P linearly encodes feature F to accuracy A."

**Rider features (pinned):**

| ID | Name | Type | Range |
| --- | --- | --- | --- |
| F1 | `dist_to_x_goal` | scalar | [0, ~7] |
| F2 | `dist_to_x_false` | scalar | [0, ~10] |
| F3 | `vec_to_x_goal` | 2D vector | [-7, 7]² |
| F4 | `vec_to_x_false` | 2D vector | [-10, 10]² |

`x_goal` is the env-sampled goal (per-episode); `x_false` is the fixed
training basin center (-2.5, -2.5). Axis A v1.5 asks which of these
geometric quantities each policy preserves or compresses across depth.
The headline number is the per-layer ΔR² over the raw observation
baseline, not raw R².

**Rejected target design:** residualizing `dist_to_x_false` against the
raw observation is degenerate in this environment because `x_false` is
fixed and `x_t` is fully present in the observation; therefore
`E[dist_to_x_false | obs] = dist_to_x_false` and the residual is
identically zero.

**Rejected behavior target:** v1.2 tested `basin_pref_intervened`, the
terminal old-basin preference under a paired basin-position
intervention. It failed the smoke gate: Oracle-S and L-Signature also
decoded the target because it was endpoint-shaped. Any per-seed scalar
that is a deterministic function of terminal position is predictable
from competent policies that encode their future endpoint, regardless
of whether that endpoint is x_goal or x_false. The optional harness flag
`--include-behavior-target` exists only to reproduce that failure.

**Activation collection protocol:**

1. Load policy.
2. Run 64 clean episodes on the same matched seed slate under the
   nominal Phase 0 configuration (no probe, no intervention).
3. At every clean env step `t`, record:
   - Hidden activations from every layer of the actor network (the
     critic is collected separately, optional output).
   - Ground-truth x_goal (env-internal).
   - Agent position x_t (observation-derivable).
4. Compute rider features per step:
   - `dist_to_x_goal[t] = ||x_t − x_goal||`
   - `dist_to_x_false[t] = ||x_t − (-2.5, -2.5)||`
   - `vec_to_x_goal[t] = x_goal − x_t`
   - `vec_to_x_false[t] = (-2.5, -2.5) − x_t`

**Probe fitting protocol:**

For each `(layer, feature)` cell:

1. Stack activations across episodes/timesteps → `X ∈ R^(N × d_layer)`.
2. Stack target feature → `y ∈ R^N` (scalar) or `Y ∈ R^(N × 2)` (vec).
3. Train/test split: 80/20 stratified by episode (not by step — keeps
   train and test from sharing episode trajectories).
4. Fit ridge regression with `sklearn.linear_model.Ridge(alpha=1.0)`.
5. Score: R² on the held-out test set.
6. Sanity baseline: shuffled-target R² should be near zero. If shuffled
   R² > 0.05, the split is leaking and the probe needs re-cutting.

**Why ridge and not MLP probe:** linear probes are the standard
"feature is *available* in the representation" measurement (Alain &
Bengio 2016, and many follow-ups). MLP probes confound representational
geometry with the probe's own capacity. Phase 6 v1 uses ridge for clean
interpretation; nonlinear probes are a v2 question.

**Headline metric:** for F1-F4, report `delta_r2_vs_input = R²(layer) -
R²(input.obs)`. This asks what information the policy invests depth in,
after subtracting what was already linearly available at the raw
observation. F1 (`dist_to_x_goal`) measures goal-inference work because
x_goal is hidden from learned local-probe policies. F2/F4 measure
whether fixed-basin geometry, already present through x_t, is preserved
or compressed across depth.

**Tier coverage:** smoke policies only in v1. Full Phase 5 zoo coverage
is deferred to Phase 6 v2 because both v1.2 and v1.5 smoke gates failed
to provide a non-confounded target/profile.

### 3.2 Axis B — Activation patching across the cliff

The causal axis. Take the cliff pair `(P_protected = L-Mixed-M-λ=0.95,
P_collapsed = L-Mixed-M-λ=0.97)`, and ask: which layer's activations,
when transplanted from protected → collapsed (or collapsed →
protected), shifts the behavioral readout?

**Behavioral readout:** the Phase 4 basin-position intervention. For a
fixed seed slate, with `x_false` moved from training position
(-2.5, -2.5) to live position (2.5, 2.5), measure `old_basin_pref` (the
distance metric: new_basin_dist − old_basin_dist; positive = still
attracted to old training basin = internalized attractor).

The protected policy has `old_basin_pref < 1.0` under this intervention;
the collapsed policy has `old_basin_pref > 1.0`. Phase 5 v4 will report
exact values; Phase 6 v1 uses these as readout extremes.

**Patch protocol:**

For each layer `L` and each seed `s`:

1. **Forward A — clean protected.** Run `P_protected` on seed `s` with
   live `x_false` intervention. Record activations at layer L for every
   step. Record `old_basin_pref_A`.
2. **Forward B — clean collapsed.** Same with `P_collapsed`. Record
   activations at layer L. Record `old_basin_pref_B`.
3. **Forward C — patched (protected → collapsed).** Run
   `P_collapsed` on seed `s` but replace layer L's activations with
   those recorded in Forward A. Continue forward pass with the
   collapsed policy's downstream layers. Record `old_basin_pref_C`.
4. **Forward D — patched (collapsed → protected).** Reverse: run
   `P_protected` with collapsed activations patched at layer L.
   Record `old_basin_pref_D`.

**Patch-success metric:**

```
patch_success_AC = (old_basin_pref_B − old_basin_pref_C) /
                   (old_basin_pref_B − old_basin_pref_A)
patch_success_DB = (old_basin_pref_A − old_basin_pref_D) /
                   (old_basin_pref_A − old_basin_pref_B)
```

`patch_success_AC` near 1.0 means "patching protected activations at
layer L fully removes the basin attractor from the collapsed policy."
Near 0 means "the patch had no effect; the basin attractor lives
downstream of L." Negative means "the patch made things worse," which
is an interesting interaction effect worth surfacing.

**Minimal-patch rank:** the smallest set of layers whose joint patch
achieves `patch_success > 0.8` defines the *minimal causal patch* —
the load-bearing locus of the basin attractor. v1 reports
single-layer and consecutive-layer-pair patches; arbitrary subset
patching (v2) is deferred because the architecture is small enough
that single + pair coverage suffices.

**Patch implementation:** PyTorch forward hooks. The exported actor is
`MesaMlpPolicy`: `Linear → Tanh` repeated `depth` times, followed by a
final `Linear` and outer `torch.tanh(...) * action_scale` squash. There
is no actor mean-head/log-std split; `log_std` only exists on the PPO
training wrapper and is not part of deterministic exported inference.
The v1 patchable points are the post-Tanh hidden activations (`net.1`,
`net.3`, ... on the exported actor; `actor.net.1`, `actor.net.3`, ...
if using the PPO wrapper). The final linear/output squash is
excluded from the canonical v1 layer sweep unless the smoke test shows
the basin locus is entirely downstream of the last hidden activation.
The harness should introspect named modules rather than hard-code
depths.

**Tier coverage:** Medium only (the cliff is a Medium phenomenon).
Small λ-sweep patching is deferred — at Small the curve is monotone
and there is no cliff to localize.

## 4. Pre-Registered Predictions

Four load-bearing predictions, all checkable from Phase 6 harness
outputs.

### 4.1 (P1″) L-Reward preserves basin geometry across depth — failed smoke

L-Reward-M should show positive ΔR² for `dist_to_x_false` from `net.3`
onward. The basin location is linearly recoverable from the input, but
the prediction is that reward-trained basin-captured policies preserve
that geometry through deeper layers because it is load-bearing for
action.

**Smoke status:** failed as a v1 claim surface. The 64-seed Medium smoke
showed L-Reward-M deepest-layer `dist_to_x_false` ΔR² ≈ 0.011, while
L-Signature-M showed ≈ 0.213. This is not the predicted separation.

**Falsifier:** L-Reward-M shows zero or negative ΔR² for
`dist_to_x_false` at deeper layers. That would mean Axis A cannot
distinguish basin preservation from ordinary input geometry in the
reward-trained policy.

### 4.2 (P2″) L-Signature compresses basin geometry across depth — failed smoke

L-Signature-M variants should show ΔR² for `dist_to_x_false` near zero
or negative at deeper layers. The fixed basin is present through x_t,
but signature-trained policies have no reason to preserve it.

**Falsifier:** L-Signature-M preserves positive `dist_to_x_false` ΔR² at
deep layers comparable to L-Reward-M. That would make the F2/F4
depth-profile diagnostic non-separating.

**Smoke status:** falsified for v1. The Medium smoke showed
L-Signature-M preserving more fixed-basin geometry at the deepest layer
than L-Reward-M.

### 4.3 (P3″) The cliff pair shows a step in goal-inference ΔR² — failed smoke

At the deepest hidden layer, `dist_to_x_goal` ΔR² should show a sharp
step across the L-Mixed-M cliff:

- L-Mixed-M-λ=0.95 maintains L-Signature-like positive ΔR² (goal
  inference preserved).
- L-Mixed-M-λ=0.97 collapses toward zero (goal inference crowded out by
  the basin attractor).

That is: the cliff should be visible as a change in what the network
invests depth in, not as raw feature decodability.

**Falsifier 1:** deepest-layer `dist_to_x_goal` ΔR² is flat across the
λ sweep. Would suggest Axis A is not sensitive to the cliff mechanism
and Phase 6 v1 should lean on Axis B.

**Falsifier 2:** the ΔR² step is present but offset from the behavioral
cliff. Would suggest goal-inference preservation and basin-collapse
behavior are related but not the same causal object.

**Smoke status:** falsified for v1. The cliff-pair smoke showed
deepest-layer `dist_to_x_goal` ΔR² ≈ -0.018 for λ=0.95 and ≈ 0.091 for
λ=0.97, the opposite of the predicted collapse. Axis A is therefore
deferred to v2.

### 4.4 (P4) Activation patching localizes the basin to ≤ 2 consecutive layers

Layer-level patching across the cliff pair should identify a minimal
causal patch of one or two consecutive layers whose transplant
achieves `patch_success > 0.8` in at least one direction
(protected → collapsed). This is the "the basin attractor lives at
layer L (± 1)" claim.

**Falsifier 1:** no single layer or layer-pair achieves
`patch_success > 0.5`. Would suggest the basin attractor is genuinely
distributed and v2 needs finer-grained patching (per-neuron or
direction-based via SAE features).

**Falsifier 2:** the symmetric directions (P_protected → P_collapsed
vs P_collapsed → P_protected) disagree by more than 30 percentage
points in identified layer. Would suggest the basin attractor lives at
different depths in the two policies (interesting in its own right) and
the "minimal patch" framing needs revision.

**P4-alternative:** after the Axis A negative result, the prior on a
distributed mechanism is higher. If no single layer or consecutive pair
achieves `patch_success > 0.5`, v1 reports the cliff as mechanically
distributed at layer granularity and routes v2 to SAE / direction-based
patching. This is a positive result, not merely a failed localization.

**Directional diagnostic:** report protected→collapsed and
collapsed→protected separately. If protected→collapsed is stronger, the
protected representation can override the collapsed policy downstream.
If collapsed→protected is stronger, the protected policy's downstream
machinery may be carrying the protection. The asymmetry is a first-class
diagnostic alongside absolute patch magnitude.

**Condition diagnostic:** run Axis B under both clean and
basin-position-intervened environments. Clean-condition patching asks
whether the transplant affects behavior generally; intervened-condition
patching asks whether it specifically affects the basin-capture failure
surface.

## 5. Phase 5 Policy Zoo Manifest

Phase 6 v1 inherits its fittable policy zoo from
`results/mesa/phase5-selection-pressure/policies-summary.csv`. That CSV
is the path authority for v1 and currently contains 22 learned policy
rows. The tables below are human-readable labels and reference rows; if
they disagree with the aggregate CSV, regenerate or patch the CSV before
running Phase 6. The harness must resolve checkpoint and policy JSON
paths from each row's `training_slug`; do not derive paths from the
displayed λ label, because the v4 cliff runs intentionally reused the
`mixed_ppo_phase3_lambda_0_9` variant slug with λ overridden by run
label/config.

### 5.1 Non-fittable reference policies (HC + Oracle floors)

| Slug | Label | Notes |
| --- | --- | --- |
| `hc_signature_small` | HC-Sig-S | hand-coded; no learned representation |
| `hc_signature_medium` | HC-Sig-M | (same; size match only) |
| `oracle_small` | Oracle-S | privileged analytic controller; no PyTorch checkpoint |
| `oracle_medium` | Oracle-M | privileged analytic controller; no PyTorch checkpoint |

HC-Sig and Oracle are behavioral controls only. Their "hidden layers" do
not exist in the same sense as learned policies, so Phase 6 excludes
them from ridge fitting and activation patching. If an Oracle-like
learned checkpoint is later added, it should enter the manifest as a
new learned policy row rather than reusing the analytic Oracle label.

### 5.2 L-Signature policies (shape axis)

| Slug | Label | Notes |
| --- | --- | --- |
| `signature_ppo_dense_small` | L-Sig-S-Integrated | Phase 2 canonical |
| `signature_ppo_dense_medium` | L-Sig-M-Integrated | Phase 2 canonical |
| `signature_ppo_terminal_small` | L-Sig-S-Terminal | Phase 5 |
| `signature_ppo_terminal_medium` | L-Sig-M-Terminal | Phase 5 v2 |
| `signature_ppo_threshold_small` | L-Sig-S-Threshold | Phase 5 |

### 5.3 L-Reward policies (canonical + optional clean controls)

| Slug | Label | Notes |
| --- | --- | --- |
| `reward_ppo_dense_small` | L-Reward-Clean-S | β=0 (no basin); optional add-on unless added to Phase 5 aggregate |
| `reward_ppo_dense_medium` | L-Reward-Clean-M | β=0 (no basin); optional add-on unless added to Phase 5 aggregate |
| `reward_ppo_phase3_small` | L-Reward-S | β=2.0 canonical |
| `reward_ppo_phase3_medium` | L-Reward-M | β=2.0 canonical |

### 5.4 L-Mixed policies (λ axis — the cliff lives here)

| Slug | Label | Tier | Notes |
| --- | --- | --- | --- |
| `mixed_ppo_phase3_lambda_0_1_small` | L-Mixed-S-λ0.1 | Small | |
| `mixed_ppo_phase3_lambda_0_3_small` | L-Mixed-S-λ0.3 | Small | |
| `mixed_ppo_phase3_lambda_0_5_small` | L-Mixed-S-λ0.5 | Small | Phase 3 canonical |
| `mixed_ppo_phase3_lambda_0_7_small` | L-Mixed-S-λ0.7 | Small | |
| `mixed_ppo_phase3_lambda_0_9_small` | L-Mixed-S-λ0.9 | Small | |
| `mixed_ppo_phase3_lambda_0_3_medium` | L-Mixed-M-λ0.3 | Medium | |
| `mixed_ppo_phase3_lambda_0_5_medium` | L-Mixed-M-λ0.5 | Medium | |
| `mixed_ppo_phase3_lambda_0_7_medium` | L-Mixed-M-λ0.7 | Medium | |
| `mixed_ppo_phase3_lambda_0_8_medium` | L-Mixed-M-λ0.8 | Medium | Phase 5 v3 |
| `mixed_ppo_phase3_lambda_0_9_medium` | L-Mixed-M-λ0.9 | Medium | Phase 5 v3 |
| `mixed_ppo_phase3_lambda_0_95_medium` | L-Mixed-M-λ0.95 | Medium | **cliff protected side** |
| `mixed_ppo_phase3_lambda_0_97_medium` | L-Mixed-M-λ0.97 | Medium | **cliff collapsed side** |
| `mixed_ppo_phase3_lambda_0_99_medium` | L-Mixed-M-λ0.99 | Medium | Phase 5 v4 |

### 5.5 Curriculum policies

| Slug | Label | Notes |
| --- | --- | --- |
| `curriculum_sig_then_reward_small` | Curr-Sig→Reward-S | Phase 5 |
| `curriculum_reward_then_sig_small` | Curr-Reward→Sig-S | Phase 5 |

### 5.6 Zoo total

The Phase 5 aggregate currently contains 22 fittable learned policies
for Axis A. HC-Sig and Oracle are excluded behavioral controls. The
clean reward policies are useful optional add-ons, but they do not count
toward the 22-policy v1 completion gate unless they are explicitly added
to the Phase 6 manifest. The cliff pair (L-Mixed-M-λ=0.95 + λ=0.97)
carries the v1 load-bearing Axis B work.

## 6. Metrics

### 6.1 Axis A — probe accuracy

For each `(policy, layer, feature)` cell:

- `r2_test`: ridge R² on held-out 20%.
- `r2_train`: ridge R² on training 80% (gap indicates probe overfitting
  / data-shortage; flag if `r2_train − r2_test > 0.15`).
- `r2_shuffled`: ridge R² with shuffled targets (should be near 0;
  flag if > 0.05).
- `n_samples`: total step-count entering the probe (typically 64
  episodes × up to 200 steps, with early termination common; still
  comfortably sized for ridge at the current hidden widths).

### 6.2 Axis B — patch success

For each `(layer, seed, direction)` cell:

- `old_basin_pref_clean_protected`: baseline.
- `old_basin_pref_clean_collapsed`: baseline.
- `old_basin_pref_patched`: post-patch readout.
- `patch_success`: normalized as in §3.2.

Aggregate across seeds: mean `patch_success` per `(layer, direction)`
with seed-bootstrap 95% CI.

### 6.3 Phase 6 cross-policy aggregates

- **Probe-accuracy heatmap**: rows = policies, cols = (layer, feature),
  cells = R².
- **Cliff-step plot**: x = λ (Medium sweep), y = R² for
  `dist_to_x_false` at the most-decoding layer of each policy.
- **Patch-success table**: rows = layers, cols = direction, cells =
  mean patch_success.
- **Minimal-patch summary**: a one-line claim — "the basin attractor is
  localized to layer L (single layer) / layers {L, L+1} (consecutive
  pair) / is distributed (no minimal patch found)."

## 7. Harness — `training/mesa/phase6_probes.py`

A new Python harness. Sketch of structure:

```python
# training/mesa/phase6_probes.py
import argparse, json, pathlib
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from training.mesa.js_bridge_env import BridgeClient, REPO_ROOT
from training.mesa.policy import load_checkpoint, policy_from_checkpoint
# BridgeClient talks to scripts/mesa-env-bridge.mjs over stdio.

FALSE_BASIN = np.array([-2.5, -2.5])

def collect_activations(policy_path, tier, seed_lo=10000, seed_hi=10064):
    """Run policy on matched seed slate, record per-step activations + features."""
    checkpoint = load_checkpoint(policy_path)
    policy, obs_mean, obs_std = policy_from_checkpoint(checkpoint)
    hooks, activations = register_forward_hooks(policy)
    records = []
    with BridgeClient() as bridge:
        for seed in range(seed_lo, seed_hi):
            env_id = f"phase6-{seed}"
            made = bridge.request({
                "cmd": "make",
                "env_id": env_id,
                "seed": seed,
                "sensor_tier": "local-probe-field",
                "env_config": {"horizon": 200},
                "probe": {"type": "nominal"},
            })
            obs = np.asarray(made["obs"], dtype=np.float32)
            info = made["info"]
            done = False
            step = 0
            while not done:
                norm_obs = (obs - obs_mean) / obs_std
                obs_t = torch.tensor(norm_obs[None, :], dtype=torch.float32)
                with torch.no_grad():
                    action = policy(obs_t)[0].cpu().numpy()
                agent_pos = obs[:2].copy()
                records.append({
                    "seed": seed,
                    "step": step,
                    "x_t": agent_pos,
                    "x_goal": np.asarray(info["x_goal"], dtype=np.float32),
                    "x_false": np.asarray(info["x_false"], dtype=np.float32),
                    "activations": {
                        k: v.detach().cpu().numpy()[0].copy()
                        for k, v in activations.items()
                    },
                })
                stepped = bridge.request({
                    "cmd": "step",
                    "env_id": env_id,
                    "action": action.tolist(),
                })
                obs = np.asarray(stepped["obs"], dtype=np.float32)
                info = stepped["info"]
                done = bool(stepped["done"])
                step += 1
    return records

def fit_shuffled_ridge(X, y, train_idx, test_idx, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y_shuf = y.copy()
    rng.shuffle(y_shuf, axis=0)
    probe = Ridge(alpha=alpha)
    probe.fit(X[train_idx], y_shuf[train_idx])
    return r2_score(y_shuf[test_idx], probe.predict(X[test_idx]))

def fit_probes(records, layer_names, feature_names):
    """Fit ridge probe for every (layer, feature) cell."""
    results = []
    for layer in layer_names:
        X = np.stack([r["activations"][layer] for r in records])
        for feature in feature_names:
            y = compute_feature(records, feature)
            # episode-stratified 80/20 split
            train_idx, test_idx = episode_split(records, frac=0.8)
            probe = Ridge(alpha=1.0)
            probe.fit(X[train_idx], y[train_idx])
            results.append({
                "layer": layer,
                "feature": feature,
                "r2_test": r2_score(y[test_idx], probe.predict(X[test_idx])),
                "r2_train": r2_score(y[train_idx], probe.predict(X[train_idx])),
                "r2_shuffled": fit_shuffled_ridge(X, y, train_idx, test_idx),
                "n_samples": len(records),
            })
    return results

def patch_activations(protected_path, collapsed_path, layer, tier,
                      seed_lo=10000, seed_hi=10064,
                      x_false_intervention=(2.5, 2.5)):
    """Run cliff-pair activation-patching battery for one layer."""
    p_protected, protected_mean, protected_std = policy_from_checkpoint(load_checkpoint(protected_path))
    p_collapsed, collapsed_mean, collapsed_std = policy_from_checkpoint(load_checkpoint(collapsed_path))
    results = []
    for seed in range(seed_lo, seed_hi):
        # Forward A: clean protected, recording layer L activations
        cache_A = run_with_recording(p_protected, protected_mean, protected_std, layer, seed,
                                     x_false=x_false_intervention)
        # Forward B: clean collapsed, recording
        cache_B = run_with_recording(p_collapsed, collapsed_mean, collapsed_std, layer, seed,
                                     x_false=x_false_intervention)
        # Forward C: collapsed with protected's layer-L activations injected
        readout_C = run_with_injection(p_collapsed, collapsed_mean, collapsed_std, layer,
                                       cache_A.activations, seed,
                                       x_false=x_false_intervention)
        # Forward D: protected with collapsed's layer-L activations injected
        readout_D = run_with_injection(p_protected, protected_mean, protected_std, layer,
                                       cache_B.activations, seed,
                                       x_false=x_false_intervention)
        results.append({
            "seed": seed,
            "layer": layer,
            "obp_A": cache_A.old_basin_pref,
            "obp_B": cache_B.old_basin_pref,
            "obp_C": readout_C.old_basin_pref,
            "obp_D": readout_D.old_basin_pref,
        })
    return results
```

Key implementation notes:

- **Forward hooks** are registered on the actor's named post-Tanh hidden
  modules (`net.1`, `net.3`, ...); the dict key is the module name.
  Critic hooks are optional (Axis A scope creep risk; v1 sticks to actor
  unless an early smoke test surfaces something striking).
- **Injection** replaces the forward output of the named module for
  the duration of one forward pass. PyTorch hook handles this cleanly
  via the `register_forward_hook` return-value override.
- **Seed determinism**: numpy random state is seeded from the env's
  splitmix64 stream, not from Python's global. Run-to-run reproducibility
  is required for the patch metrics to be matched.
- **Diagnostic env accessor**: `scripts/mesa-env-bridge.mjs` exposes
  `info.x_goal`, `info.x_false`, `info.position`, and
  `info.true_gradient` through the diagnostic JSON channel. These are
  labels for the probe harness; they are not part of the policy
  observation.
- **Compute budget**: per policy, activation collection is ~3 minutes
  (64 episodes × up to 200 steps × forward pass). Per-cliff-pair patch
  battery is ~10 minutes (64 seeds × 4 forward passes × N layers ≈
  64 × 4 × 4 ≈ 1024 forward passes at Medium). Full zoo Axis A is
  ~70-90 minutes. Cliff-pair Axis B is ~10 minutes. Total v1 wall-clock
  ≈ 2-4 hours.

## 8. Outputs

```
results/mesa/phase6-probes/
  manifest.json
  policies-summary.csv                 # one row per Phase 6 policy
  axis-a-probe-accuracy.csv            # (policy, layer, feature) → r2_test/train/shuffled
  axis-a-cliff-step.csv                # λ × deepest layer → dist_to_x_goal delta_r2_vs_input
  axis-b-patch-success.csv             # (layer, direction, seed) → patch_success
  axis-b-patch-aggregate.csv           # (layer, direction) → mean + 95% CI
  reports/
    minimal-patch-summary.json         # { "single_layer": "net.1", "patch_success": 0.83 }
    probe-accuracy-heatmap.png         # optional; v1 may stay CSV-only
    cliff-step-plot.png                # optional; v1 may stay CSV-only
```

The `manifest.json` lists every policy run as part of Phase 6, with
slug, tier, source phase, checkpoint path/hash, and policy.json
path/hash when present. Phase 6 is read-only with respect to Phase 5
artifacts; the manifest exists so "which version of L-Mixed-M-λ=0.97
did Phase 6 probe?" is answerable.

## 9. Execution Order

Recommended sequencing for Phase 6 v1:

1. **Harness scaffold lands first.** `phase6_probes.py` with the hook
   registration, env loader, and ridge fitter. Smoke-test on a single
   policy (HC-Sig-S excluded; use L-Sig-S-Integrated as the smoke
   target) before launching full zoo.
2. **Axis A smoke**: L-Sig-S-Integrated + L-Reward-S + Oracle-S. Treat
   Oracle-S as an analytic privileged ceiling row rather than a fittable
   neural-layer policy. Verify the depth-profile directionality:
   - Oracle-S: privileged ceiling has near-perfect F1 raw R²; because
     analytic Oracle has no `net.1` and its input already contains
     x_goal, it is a scoring control rather than a learned-depth gate.
   - L-Reward-S: F1 ΔR² ≤ 0.1 at all hidden depths; F2 ΔR² positive at
     the deepest hidden layer if basin geometry is preserved.
   - L-Sig-S-Integrated: F1 ΔR² positive and ideally growing across
     depth; F2 ΔR² decays toward zero or negative at deeper layers.
   - Shuffled baseline: abs R² ≤ 0.05.
   **Smoke gate:** if L-Signature and L-Reward show indistinguishable
   F1/F2 ΔR² depth profiles, Axis A is demoted to descriptive context
   and Phase 6 v1 leans on Axis B for the causal claim.
3. **Do not run Axis A full zoo in v1.** Archive the v1.2 and v1.5
   smoke artifacts as failed target-design probes. Route Axis A to v2
   with SAE/direction-based probing.
4. **Axis B cliff-pair smoke**: single layer (`net.1` on exported actor), 8 seeds. Verify
   patch mechanics work and that direction A→C is non-trivial. ~3
   minutes.
5. **Axis B full battery**: all layers × full seed slate, both
   directions. ~10 minutes.
6. **Axis B aggregate**: patch-success table, minimal-patch summary.
7. **Phase 6 result note** (`docs/mesa/PHASE6_RESULTS.md`) is written
   with Axis A smoke failure + Axis B minimal-patch summary as the
   headline.

Total wall-clock: ~10-30 minutes for remaining v1 work.

## 10. Exit Criterion

Phase 6 v1 complete when:

- `phase6_probes.py` lands, smoke-tests cleanly on the 3-policy probe
  set, and produces the expected CSV columns.
- Axis A smoke artifacts are archived and explicitly marked as
  non-greenlighting for full-zoo Axis A.
- Axis B patch-success CSV covers the cliff pair × every layer × full
  seed slate × both directions.
- The minimal-patch summary is generated and inspected for the
  predicted P4 single-layer-or-pair claim.
- At least one of the four pre-registered predictions in §4 is
  confirmed or falsified.
- Phase 6 result note (`docs/mesa/PHASE6_RESULTS.md`) is written with
  the load-bearing finding as the headline.

## 11. Cross-References

- **Phase 0 spec**: env spec, signature definition, false-basin
  configuration. [`PHASE0_SPEC.md`](PHASE0_SPEC.md).
- **Phase 3 spec / results**: probe slate, basin-capture taxonomy,
  capacity-dependence picture. [`PHASE3_SPEC.md`](PHASE3_SPEC.md),
  [`PHASE3_RESULTS.md`](PHASE3_RESULTS.md).
- **Phase 4 spec / results**: basin-position intervention, fixed-attractor
  receipt. [`PHASE4_SPEC.md`](PHASE4_SPEC.md),
  [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md).
- **Phase 5 spec / results**: L-Mixed protection curve, the cliff at
  λ ≈ 0.953, terminal-only signature canonical correction.
  [`PHASE5_SPEC.md`](PHASE5_SPEC.md),
  [`PHASE5_RESULTS.md`](PHASE5_RESULTS.md).
- **Roadmap**: [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md).
- **Promo / claim language**: [`../PROMO_HIGHLIGHTS.md`](../PROMO_HIGHLIGHTS.md)
  §The Gravity Claim → §The empirical anchor (Phase 5 v4).

## 12. What Phase 7 Inherits

Phase 7 (operating envelope) consumes Phase 6's interpretability
artifacts in two ways:

- **Feature investment as a deployment-time check.** If the ΔR²
  depth-profile finding is robust ("deep layers preserve basin geometry
  while goal-inference ΔR² collapses ⇒ basin internalized"), it becomes
  a cheap monitoring signal for Phase 7's deployment-envelope study:
  run the probe on a candidate policy before deployment and flag the
  risky profile.
- **Minimal-patch locus as a hardening target.** If Phase 6 v1 identifies
  a single-layer minimal patch, Phase 7 v2 can attempt targeted
  fine-tuning interventions (freeze the patch layer, train only above
  or below) to ask whether the basin attractor is removable post-hoc
  without full retraining.

Phase 7 may also pull SAE-feature dictionaries from a future Phase 6
v2 if v1 finds the linear-probe granularity insufficient for
deployment-relevant monitoring.

## 13. Versioning

- **v1.6 Axis B promotion (2026-05-12)** — records the v1.5 ΔR² smoke
  failure (Small, Medium, and cliff-pair checks), defers full-zoo Axis A
  to v2, and makes Axis B activation patching the Phase 6 v1
  load-bearing artifact.
- **v1.5 depth-profile pivot (2026-05-12)** — demotes the failed
  endpoint-shaped `basin_pref_intervened` target, promotes ΔR² over
  `input.obs` as the Axis A headline, rewrites P1-P3 around feature
  preservation/compression across depth, and restores Axis A to
  clean-rollout collection only.
- **v1.2 target correction (2026-05-12)** — replaces raw x_false
  decodability with the intervention-conditioned
  `basin_pref_intervened` headline target, keeps geometric probes as
  ΔR² rider diagnostics, rejects input residualization as degenerate,
  and blocks full-zoo Axis A until the three-policy smoke gate passes.
- **v1.1 sanity check (2026-05-12)** — aligned v1 with the actual
  Phase 5 artifacts and trainer code: Tanh MLP actor hooks,
  sklearn-backed ridge probes, JS bridge diagnostic labels, Phase 5
  aggregate CSV as the fittable-zoo source of truth, and checkpoint
  hashes in the manifest.
- **v1 (2026-05-12)** — initial pin. Two axes (linear probes,
  activation patching across the cliff). Four pre-registered
  predictions (P1-P4). 22-policy zoo inherited from Phase 5 v4. No new
  training runs; harness-only addition. Cliff pair
  (L-Mixed-M-λ=0.95 + λ=0.97) carries the Axis B work.
