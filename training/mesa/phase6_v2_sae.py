"""Phase 6 v2 — Direction-based mechanistic probing.

Phase 6 v1 localized the cliff causally to `net.7` of the actor MLP. v2 asks
which direction inside that 256-dim activation space carries the basin
attractor, by:

- Axis D: training a single top-k sparse autoencoder on the joint net.7
  activations of the cliff pair (L-Mixed-M-λ=0.95 vs λ=0.97), labeling each
  SAE feature by its correlation with per-episode `basin_pref_intervened`.
- Axis E: direction-based activation patching that substitutes only the
  projection along the top-correlated SAE feature's decoder column, leaving
  the other 255 dimensions of net.7 untouched. Tightens v1 P4 from
  single-layer to single-direction if it clears the threshold.

Spec: docs/mesa/PHASE6_V2_SPEC.md (v2, 2026-05-12).

The harness reuses v1 helpers from training.mesa.phase6_probes for env
stepping, policy loading, basin-position interventions, and patch-success
metrics — only the SAE training and direction-injection hook are new.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from training.mesa.js_bridge_env import BridgeClient, REPO_ROOT
from training.mesa.phase6_probes import (
    CHECKPOINT_DIR,
    CLIFF_COLLAPSED,
    CLIFF_PROTECTED,
    PolicySpec,
    basin_position_intervention,
    collect_learned,
    get_module,
    learned_action,
    load_learned_policy,
    mean_finite,
    median_finite,
    old_basin_pref,
    ratio_of_means,
    safe_patch_success,
    write_csv,
)


PHASE6_V2_OUT = REPO_ROOT / "results" / "mesa" / "phase6-v2-direction"
PHASE6_V31_OUT = REPO_ROOT / "results" / "mesa" / "phase6-v3-1-validation"
PHASE6_V32_OUT = REPO_ROOT / "results" / "mesa" / "phase6-v3-2-neuron-mediation"
PHASE6_V33_OUT = REPO_ROOT / "results" / "mesa" / "phase6-v3-3-ablation"
PHASE6_V34_OUT = REPO_ROOT / "results" / "mesa" / "phase6-v3-4"


V31_POLICY_SPECS: dict[str, PolicySpec] = {
    "signature_terminal_medium": PolicySpec(
        policy_id="signature_terminal_medium",
        label="L-Sig-Terminal-M",
        kind="learned",
        checkpoint=CHECKPOINT_DIR / "signature_ppo_terminal_medium_seed_0_medium_phase5_terminal_10m.pt",
        sensor_tier="local-probe-field",
    ),
    "reward_lambda_1_0_medium_anchor": PolicySpec(
        policy_id="reward_lambda_1_0_medium_anchor",
        label="L-Reward-M lambda=1.0 anchor",
        kind="learned",
        checkpoint=CHECKPOINT_DIR / "reward_ppo_phase3_medium_seed_0_medium_phase3_canonical_10m.pt",
        sensor_tier="local-probe-field",
    ),
    "mixed_lambda_0_9_medium_v3": PolicySpec(
        policy_id="mixed_lambda_0_9_medium_v3",
        label="L-Mixed-M lambda=0.9 v3",
        kind="learned",
        checkpoint=CHECKPOINT_DIR / "mixed_ppo_phase3_lambda_0_9_medium_seed_0_medium_phase5_v3_lambda_0_9_10m.pt",
        sensor_tier="local-probe-field",
    ),
    "mixed_lambda_0_99_medium_v4": PolicySpec(
        policy_id="mixed_lambda_0_99_medium_v4",
        label="L-Mixed-M lambda=0.99 v4",
        kind="learned",
        checkpoint=CHECKPOINT_DIR / "mixed_ppo_phase3_lambda_0_9_medium_seed_0_medium_phase5_v4_lambda_0_99_10m.pt",
        sensor_tier="local-probe-field",
    ),
    "signature_terminal": PolicySpec(
        policy_id="signature_terminal",
        label="L-Sig-Terminal-S",
        kind="learned",
        checkpoint=CHECKPOINT_DIR / "signature_ppo_terminal_small_seed_0_phase5.pt",
        sensor_tier="local-probe-field",
    ),
    "curriculum_reward_then_terminal_sig_v3": PolicySpec(
        policy_id="curriculum_reward_then_terminal_sig_v3",
        label="Curriculum reward-then-terminal-sig v3",
        kind="learned",
        checkpoint=CHECKPOINT_DIR / "curriculum_reward_then_terminal_sig_small_seed_0_phase5_v3_reward_pre_terminal_sig_ft_500k.pt",
        sensor_tier="local-probe-field",
    ),
}

V31_GENERALIZATION_PAIRS: dict[str, tuple[PolicySpec, PolicySpec]] = {
    "J1": (
        V31_POLICY_SPECS["signature_terminal_medium"],
        V31_POLICY_SPECS["reward_lambda_1_0_medium_anchor"],
    ),
    "J2": (
        V31_POLICY_SPECS["mixed_lambda_0_9_medium_v3"],
        V31_POLICY_SPECS["mixed_lambda_0_99_medium_v4"],
    ),
    "J3": (
        V31_POLICY_SPECS["signature_terminal"],
        V31_POLICY_SPECS["curriculum_reward_then_terminal_sig_v3"],
    ),
}


# ============================================================
# Top-k Sparse Autoencoder
# ============================================================


class TopKSAE(nn.Module):
    """Top-k sparse autoencoder.

    Encoder is a Linear layer mapping d_in -> n_features. Top-k operation
    zeroes out all but the k highest pre-activations per token. Decoder is a
    bias-less Linear mapping n_features -> d_in. No sparsity penalty in the
    loss; sparsity is enforced architecturally via top-k.
    """

    def __init__(self, d_in: int, n_features: int, k: int) -> None:
        super().__init__()
        if k > n_features:
            raise ValueError(f"k={k} must be <= n_features={n_features}")
        self.encoder = nn.Linear(d_in, n_features)
        self.decoder = nn.Linear(n_features, d_in, bias=False)
        self.k = k
        # Kaiming uniform on both layers (Linear default), then normalize
        # decoder columns to unit norm so direction interpretation is clean.
        with torch.no_grad():
            self.decoder.weight.data = self.decoder.weight.data / (
                self.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def topk_sparse(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch, n_features)
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pre = self.encode(x)
        z_sparse = self.topk_sparse(z_pre)
        recon = self.decoder(z_sparse)
        return recon, z_sparse


def train_sae(
    activations: np.ndarray,
    *,
    n_features: int,
    k: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int = 0,
    log_every: int = 500,
) -> tuple[TopKSAE, dict[str, Any]]:
    """Train a top-k SAE on stacked activations. Returns (sae, health_metrics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_in = activations.shape[1]
    sae = TopKSAE(d_in=d_in, n_features=n_features, k=k)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    # Stable train/test split by row index (rows are activation vectors
    # already shuffled across episodes — splitting by row index is fine for SAE
    # training because we are not making per-episode probe claims).
    n = activations.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    activations_t = torch.tensor(activations, dtype=torch.float32)
    train = activations_t[train_idx]
    test = activations_t[test_idx]

    # Track which features have ever been active in the top-k of any token.
    n_features = sae.encoder.out_features
    ever_active = torch.zeros(n_features, dtype=torch.bool)

    losses: list[float] = []
    for step in range(steps):
        batch_indices = torch.randint(0, len(train), (batch_size,))
        x = train[batch_indices]
        recon, z = sae(x)
        loss = ((recon - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            ever_active |= (z != 0).any(dim=0)
        losses.append(float(loss.item()))
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                recon_test, _ = sae(test)
                ss_res = ((recon_test - test) ** 2).sum().item()
                ss_tot = ((test - test.mean(dim=0)) ** 2).sum().item()
                r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            print(
                f"  sae step {step:5d}: train_loss={loss.item():.5f} "
                f"test_r2={r2:.4f} active_features={int(ever_active.sum())}/{n_features}",
                flush=True,
            )

    # Final health metrics
    sae.eval()
    with torch.no_grad():
        recon_test, z_test = sae(test)
        ss_res = ((recon_test - test) ** 2).sum().item()
        ss_tot = ((test - test.mean(dim=0)) ** 2).sum().item()
        reconstruction_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        active_per_token = (z_test != 0).float().mean(dim=1).mean().item()

    dead_feature_rate = 1.0 - float(ever_active.sum() / n_features)
    health = {
        "reconstruction_r2_test": float(reconstruction_r2),
        "dead_feature_rate": float(dead_feature_rate),
        "active_feature_rate": float(active_per_token),
        "target_active_rate": float(k / n_features),
        "final_train_loss": losses[-1],
        "n_features": int(n_features),
        "k": int(k),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "training_steps": int(steps),
    }
    return sae, health


# ============================================================
# Activation collection (delegate to v1 helper)
# ============================================================


@dataclass
class CliffActivations:
    policy_id: str
    label: str
    seeds: np.ndarray             # shape (n_steps,) — episode seed per row
    net7: np.ndarray              # shape (n_steps, 256)
    basin_pref_per_seed: dict[int, float]


def collect_cliff_activations(
    spec: PolicySpec,
    *,
    seed_start: int,
    seeds: int,
    horizon: int,
) -> CliffActivations:
    """Run the v1 collector with intervened basin_pref targets enabled."""
    collection = collect_learned(
        spec,
        seed_start=seed_start,
        seeds=seeds,
        horizon=horizon,
        include_behavior_target=True,
    )
    if "net.7" not in collection.layers:
        available = ", ".join(sorted(collection.layers))
        raise KeyError(
            f"net.7 activations not found for {spec.label}; available: {available}"
        )
    # Per-episode target: take the value at the first occurrence of each seed
    # (basin_pref_targets is broadcast across steps in v1's collector).
    basin_pref_per_seed: dict[int, float] = {}
    seen: set[int] = set()
    for seed, target in zip(collection.seeds.tolist(), collection.basin_pref_targets.tolist()):
        if seed not in seen:
            basin_pref_per_seed[int(seed)] = float(target)
            seen.add(int(seed))
    return CliffActivations(
        policy_id=spec.policy_id,
        label=spec.label,
        seeds=collection.seeds.copy(),
        net7=collection.layers["net.7"].copy(),
        basin_pref_per_seed=basin_pref_per_seed,
    )


# ============================================================
# Feature labeling
# ============================================================


def compute_feature_correlations(
    sae: TopKSAE,
    cliff_acts: list[CliffActivations],
) -> tuple[np.ndarray, list[tuple[str, int, float, float]]]:
    """For each feature f, compute Pearson correlation between max-over-step
    activation and per-episode basin_pref_intervened, across all (policy,seed)
    pairs in the cliff pair.

    Returns (correlations, per_episode_max_acts) where:
      correlations: shape (n_features,) of corrcoef values
      per_episode_max_acts: list of (policy_id, seed, max_act_vector, target)
    """
    sae.eval()
    feature_max_acts_per_episode: list[np.ndarray] = []
    targets: list[float] = []
    rows: list[tuple[str, int, float, float]] = []

    for ca in cliff_acts:
        # Encode all activations for this policy
        with torch.no_grad():
            x = torch.tensor(ca.net7, dtype=torch.float32)
            _, z = sae(x)  # (n_steps, n_features)
            z_np = z.numpy()
        # Group rows by seed; max over steps within each episode
        unique_seeds = np.unique(ca.seeds)
        for seed in unique_seeds:
            mask = ca.seeds == seed
            max_act = z_np[mask].max(axis=0)  # (n_features,)
            target = ca.basin_pref_per_seed.get(int(seed), float("nan"))
            if not math.isfinite(target):
                # No intervention target for this seed — skip from correlation
                continue
            feature_max_acts_per_episode.append(max_act)
            targets.append(target)
            rows.append((ca.policy_id, int(seed), float(max_act.sum()), target))

    feature_matrix = np.stack(feature_max_acts_per_episode, axis=1)  # (n_features, n_episodes)
    targets_arr = np.array(targets, dtype=np.float64)
    n_features = feature_matrix.shape[0]
    correlations = np.zeros(n_features, dtype=np.float64)
    target_std = targets_arr.std()
    for f in range(n_features):
        feat_vals = feature_matrix[f].astype(np.float64)
        feat_std = feat_vals.std()
        if feat_std < 1e-9 or target_std < 1e-9:
            correlations[f] = 0.0
            continue
        correlations[f] = float(np.corrcoef(feat_vals, targets_arr)[0, 1])
    return correlations, rows


def per_policy_feature_means(
    sae: TopKSAE,
    cliff_acts: list[CliffActivations],
    feature_indices: list[int],
) -> dict[str, dict[int, float]]:
    """For a small set of features, compute mean activation per policy.
    Used in the top-10 report to show how features partition the cliff pair.
    """
    sae.eval()
    result: dict[str, dict[int, float]] = {}
    for ca in cliff_acts:
        with torch.no_grad():
            x = torch.tensor(ca.net7, dtype=torch.float32)
            _, z = sae(x)
            z_np = z.numpy()
        feature_means: dict[int, float] = {}
        for f in feature_indices:
            feature_means[int(f)] = float(z_np[:, f].mean())
        result[ca.policy_id] = feature_means
    return result


# ============================================================
# Direction-based patching
# ============================================================


@dataclass
class DirectionRolloutCache:
    old_basin_pref: float
    terminal_position: np.ndarray
    terminal_outcome: str
    steps: int
    projections: list[float]  # per-step ⟨h, direction_unit⟩


def run_direction_recording_rollout(
    client: BridgeClient,
    *,
    policy: torch.nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seed: int,
    horizon: int,
    layer: str,
    env_id: str,
    direction_unit: torch.Tensor,
) -> DirectionRolloutCache:
    """Run policy under live x_false intervention, recording the per-step
    projection of `layer`'s output onto `direction_unit`. No modification."""
    made = client.request(
        {
            "cmd": "make",
            "env_id": env_id,
            "seed": seed,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": horizon},
            "interventions": [basin_position_intervention()],
        }
    )
    obs = np.asarray(made["obs"], dtype=np.float32)
    info = made["info"]
    terminal_position = np.asarray(info["position"], dtype=np.float32)
    terminal_outcome = "not_done"
    projections: list[float] = []
    step_index = 0

    module = get_module(policy, layer)

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        # output shape: (1, d_in); direction_unit shape: (d_in,)
        proj = float((output.squeeze(0) * direction_unit).sum().item())
        projections.append(proj)

    handle = module.register_forward_hook(hook)
    try:
        for _ in range(horizon + 1):
            action = learned_action(policy, obs, obs_mean, obs_std)
            response = client.request({"cmd": "step", "env_id": env_id, "action": action.tolist()})
            obs = np.asarray(response["obs"], dtype=np.float32)
            info = response["info"]
            terminal_position = np.asarray(info["position"], dtype=np.float32)
            step_index += 1
            if response["done"]:
                terminal_outcome = str(info.get("terminal_outcome") or "done")
                break
    finally:
        handle.remove()

    return DirectionRolloutCache(
        old_basin_pref=old_basin_pref(terminal_position),
        terminal_position=terminal_position,
        terminal_outcome=terminal_outcome,
        steps=step_index,
        projections=projections,
    )


def run_direction_injected_rollout(
    client: BridgeClient,
    *,
    policy: torch.nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seed: int,
    horizon: int,
    layer: str,
    env_id: str,
    direction_unit: torch.Tensor,
    target_projections: list[float],
) -> DirectionRolloutCache:
    """Run policy under live x_false intervention, substituting only the
    projection of `layer`'s output along `direction_unit` with the
    corresponding entry from `target_projections` at each step. The remaining
    255 dimensions of net.7 are left untouched."""
    if not target_projections:
        raise RuntimeError("cannot inject empty projection cache")
    made = client.request(
        {
            "cmd": "make",
            "env_id": env_id,
            "seed": seed,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": horizon},
            "interventions": [basin_position_intervention()],
        }
    )
    obs = np.asarray(made["obs"], dtype=np.float32)
    info = made["info"]
    terminal_position = np.asarray(info["position"], dtype=np.float32)
    terminal_outcome = "not_done"
    realized_projections: list[float] = []
    step_index = 0

    module = get_module(policy, layer)

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        idx = min(step_index, len(target_projections) - 1)
        alpha_target = target_projections[idx]
        # output shape: (1, d_in)
        h = output.squeeze(0)
        alpha_current = float((h * direction_unit).sum().item())
        delta = (alpha_target - alpha_current) * direction_unit
        h_new = h + delta
        realized_projections.append(float((h_new * direction_unit).sum().item()))
        return h_new.unsqueeze(0)

    handle = module.register_forward_hook(hook)
    try:
        for _ in range(horizon + 1):
            action = learned_action(policy, obs, obs_mean, obs_std)
            response = client.request({"cmd": "step", "env_id": env_id, "action": action.tolist()})
            obs = np.asarray(response["obs"], dtype=np.float32)
            info = response["info"]
            terminal_position = np.asarray(info["position"], dtype=np.float32)
            step_index += 1
            if response["done"]:
                terminal_outcome = str(info.get("terminal_outcome") or "done")
                break
    finally:
        handle.remove()

    return DirectionRolloutCache(
        old_basin_pref=old_basin_pref(terminal_position),
        terminal_position=terminal_position,
        terminal_outcome=terminal_outcome,
        steps=step_index,
        projections=realized_projections,
    )


# ============================================================
# Pipelines
# ============================================================


def axis_d_train_sae(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("phase6 v2 axis-D: collecting cliff-pair net.7 activations", flush=True)
    protected_acts = collect_cliff_activations(
        CLIFF_PROTECTED,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
    )
    collapsed_acts = collect_cliff_activations(
        CLIFF_COLLAPSED,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
    )

    # Stack net.7 activations from both policies for joint SAE training.
    joint_activations = np.concatenate([protected_acts.net7, collapsed_acts.net7], axis=0)
    print(
        f"  joint activation tensor: shape={joint_activations.shape} "
        f"(protected={protected_acts.net7.shape[0]} rows, "
        f"collapsed={collapsed_acts.net7.shape[0]} rows)",
        flush=True,
    )

    print("phase6 v2 axis-D: training top-k SAE", flush=True)
    sae, health = train_sae(
        joint_activations,
        n_features=args.n_features,
        k=args.k,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.sae_seed,
    )

    # Health-check pass
    health_path = out_dir / "axis-d-sae-quality.json"
    health_path.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"  sae health: r2={health['reconstruction_r2_test']:.4f} "
        f"dead_rate={health['dead_feature_rate']:.3f} "
        f"active_rate={health['active_feature_rate']:.4f} (target {health['target_active_rate']:.4f})",
        flush=True,
    )

    # Save SAE weights
    sae_path = out_dir / "sae-weights.pt"
    torch.save(sae.state_dict(), sae_path)

    # Save SAE config for reproducibility
    config_path = out_dir / "sae-config.json"
    config_path.write_text(
        json.dumps(
            {
                "d_in": int(joint_activations.shape[1]),
                "n_features": int(args.n_features),
                "k": int(args.k),
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "seed_start": int(args.seed_start),
                "seeds": int(args.seeds),
                "horizon": int(args.horizon),
                "sae_seed": int(args.sae_seed),
                "policies": [protected_acts.policy_id, collapsed_acts.policy_id],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("phase6 v2 axis-D: labeling features by basin_pref_intervened correlation", flush=True)
    correlations, _per_episode_rows = compute_feature_correlations(
        sae, [protected_acts, collapsed_acts]
    )

    abs_corr = np.abs(correlations)
    order = np.argsort(-abs_corr)
    top10_indices = order[:10].tolist()

    # Per-policy mean activation on the top features (for the top10 report)
    per_policy_means = per_policy_feature_means(
        sae, [protected_acts, collapsed_acts], top10_indices
    )

    # Write full correlation table
    corr_rows = [
        {"feature_idx": int(f), "correlation": float(correlations[f]), "abs_correlation": float(abs_corr[f])}
        for f in range(len(correlations))
    ]
    corr_rows.sort(key=lambda r: -r["abs_correlation"])
    write_csv(out_dir / "axis-d-feature-correlations.csv", corr_rows)

    # Top-10 report with per-policy mean activations
    top10_rows: list[dict[str, Any]] = []
    for rank, f in enumerate(top10_indices):
        row: dict[str, Any] = {
            "rank": rank + 1,
            "feature_idx": int(f),
            "correlation": float(correlations[f]),
            "abs_correlation": float(abs_corr[f]),
            "sign": "positive" if correlations[f] > 0 else "negative",
        }
        for policy_id, means in per_policy_means.items():
            row[f"mean_activation_{policy_id}"] = means.get(int(f), 0.0)
        # Diff (collapsed - protected) makes the basin-attribution sign explicit
        row["mean_activation_diff_collapsed_minus_protected"] = (
            per_policy_means[collapsed_acts.policy_id].get(int(f), 0.0)
            - per_policy_means[protected_acts.policy_id].get(int(f), 0.0)
        )
        top10_rows.append(row)
    write_csv(out_dir / "axis-d-top10-basin-features.csv", top10_rows)

    # Manifest
    manifest = {
        "phase": "phase6-v2-axis-d-sae",
        "protected": {
            "policy_id": protected_acts.policy_id,
            "label": protected_acts.label,
            "n_rows": int(protected_acts.net7.shape[0]),
            "n_seeds_with_target": int(len(protected_acts.basin_pref_per_seed)),
        },
        "collapsed": {
            "policy_id": collapsed_acts.policy_id,
            "label": collapsed_acts.label,
            "n_rows": int(collapsed_acts.net7.shape[0]),
            "n_seeds_with_target": int(len(collapsed_acts.basin_pref_per_seed)),
        },
        "sae": {
            "n_features": int(args.n_features),
            "k": int(args.k),
            "weights_path": str(sae_path.relative_to(REPO_ROOT)),
            "config_path": str(config_path.relative_to(REPO_ROOT)),
        },
        "health_metrics": health,
        "top_feature": {
            "feature_idx": int(top10_indices[0]),
            "correlation": float(correlations[top10_indices[0]]),
            "abs_correlation": float(abs_corr[top10_indices[0]]),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        "phase6 v2 axis-D: top feature is "
        f"f={top10_indices[0]} corr={correlations[top10_indices[0]]:+.4f}",
        flush=True,
    )
    print(
        f"phase6 v2 axis-D: wrote axis-d-feature-correlations.csv, "
        f"axis-d-top10-basin-features.csv, sae-weights.pt to {out_dir.relative_to(REPO_ROOT)}",
        flush=True,
    )


def axis_e_direction_patch(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE and pick the basin-direction
    sae_dir = Path(args.sae_dir)
    if not sae_dir.is_absolute():
        sae_dir = REPO_ROOT / sae_dir
    config = json.loads((sae_dir / "sae-config.json").read_text(encoding="utf-8"))
    sae = TopKSAE(
        d_in=int(config["d_in"]),
        n_features=int(config["n_features"]),
        k=int(config["k"]),
    )
    sae.load_state_dict(torch.load(sae_dir / "sae-weights.pt", weights_only=True))
    sae.eval()

    if args.feature_idx is not None:
        feature_idx = int(args.feature_idx)
    else:
        top10_rows = list(csv.DictReader((sae_dir / "axis-d-top10-basin-features.csv").open(encoding="utf-8")))
        feature_idx = int(top10_rows[0]["feature_idx"])

    # Decoder column = basin-direction
    direction = sae.decoder.weight.data[:, feature_idx].detach().clone()
    norm = float(direction.norm())
    if norm < 1e-12:
        raise RuntimeError(
            f"decoder column for feature {feature_idx} has near-zero norm; cannot define direction"
        )
    direction_unit = direction / norm
    print(
        f"phase6 v2 axis-E: feature_idx={feature_idx} "
        f"direction_norm={norm:.4f}, using unit vector",
        flush=True,
    )

    protected_policy, protected_mean, protected_std = load_learned_policy(CLIFF_PROTECTED)
    collapsed_policy, collapsed_mean, collapsed_std = load_learned_policy(CLIFF_COLLAPSED)

    rows: list[dict[str, Any]] = []

    with BridgeClient() as client:
        for offset in range(args.seeds):
            seed = args.seed_start + offset
            prefix = f"phase6-v2-axis-e-{seed}"

            # Forward A: clean protected, record projections
            cache_A = run_direction_recording_rollout(
                client,
                policy=protected_policy,
                obs_mean=protected_mean,
                obs_std=protected_std,
                seed=seed,
                horizon=args.horizon,
                layer=args.layer,
                env_id=f"{prefix}-A",
                direction_unit=direction_unit,
            )
            # Forward B: clean collapsed, record projections
            cache_B = run_direction_recording_rollout(
                client,
                policy=collapsed_policy,
                obs_mean=collapsed_mean,
                obs_std=collapsed_std,
                seed=seed,
                horizon=args.horizon,
                layer=args.layer,
                env_id=f"{prefix}-B",
                direction_unit=direction_unit,
            )
            # Forward C: collapsed with protected's projections injected
            cache_C = run_direction_injected_rollout(
                client,
                policy=collapsed_policy,
                obs_mean=collapsed_mean,
                obs_std=collapsed_std,
                seed=seed,
                horizon=args.horizon,
                layer=args.layer,
                env_id=f"{prefix}-C",
                direction_unit=direction_unit,
                target_projections=cache_A.projections,
            )
            # Forward D: protected with collapsed's projections injected
            cache_D = run_direction_injected_rollout(
                client,
                policy=protected_policy,
                obs_mean=protected_mean,
                obs_std=protected_std,
                seed=seed,
                horizon=args.horizon,
                layer=args.layer,
                env_id=f"{prefix}-D",
                direction_unit=direction_unit,
                target_projections=cache_B.projections,
            )

            success_pc = safe_patch_success(
                cache_A.old_basin_pref,
                cache_B.old_basin_pref,
                cache_C.old_basin_pref,
                direction="protected_to_collapsed",
            )
            success_cp = safe_patch_success(
                cache_A.old_basin_pref,
                cache_B.old_basin_pref,
                cache_D.old_basin_pref,
                direction="collapsed_to_protected",
            )
            rows.append(
                {
                    "seed": seed,
                    "feature_idx": feature_idx,
                    "layer": args.layer,
                    "protected_old_basin_pref": cache_A.old_basin_pref,
                    "collapsed_old_basin_pref": cache_B.old_basin_pref,
                    "patched_protected_to_collapsed_old_basin_pref": cache_C.old_basin_pref,
                    "patched_collapsed_to_protected_old_basin_pref": cache_D.old_basin_pref,
                    "patch_success_protected_to_collapsed": success_pc,
                    "patch_success_collapsed_to_protected": success_cp,
                    "baseline_gap_collapsed_minus_protected": cache_B.old_basin_pref - cache_A.old_basin_pref,
                    "mean_projection_protected": float(np.mean(cache_A.projections)) if cache_A.projections else float("nan"),
                    "mean_projection_collapsed": float(np.mean(cache_B.projections)) if cache_B.projections else float("nan"),
                    "protected_steps": cache_A.steps,
                    "collapsed_steps": cache_B.steps,
                }
            )
        client.request({"cmd": "close"})

    # Aggregate
    protected_values = [float(r["protected_old_basin_pref"]) for r in rows]
    collapsed_values = [float(r["collapsed_old_basin_pref"]) for r in rows]
    patched_pc = [float(r["patched_protected_to_collapsed_old_basin_pref"]) for r in rows]
    patched_cp = [float(r["patched_collapsed_to_protected_old_basin_pref"]) for r in rows]
    success_pc_values = [float(r["patch_success_protected_to_collapsed"]) for r in rows]
    success_cp_values = [float(r["patch_success_collapsed_to_protected"]) for r in rows]

    aggregate_rows = [
        {
            "direction": "protected_to_collapsed",
            "feature_idx": feature_idx,
            "layer": args.layer,
            "mean_patch_success": mean_finite(success_pc_values),
            "median_patch_success": median_finite(success_pc_values),
            "patch_success_ratio_of_means": ratio_of_means(
                protected_values,
                collapsed_values,
                patched_pc,
                direction="protected_to_collapsed",
            ),
            "mean_protected_old_basin_pref": mean_finite(protected_values),
            "mean_collapsed_old_basin_pref": mean_finite(collapsed_values),
            "mean_patched_old_basin_pref": mean_finite(patched_pc),
            "n": len(rows),
        },
        {
            "direction": "collapsed_to_protected",
            "feature_idx": feature_idx,
            "layer": args.layer,
            "mean_patch_success": mean_finite(success_cp_values),
            "median_patch_success": median_finite(success_cp_values),
            "patch_success_ratio_of_means": ratio_of_means(
                protected_values,
                collapsed_values,
                patched_cp,
                direction="collapsed_to_protected",
            ),
            "mean_protected_old_basin_pref": mean_finite(protected_values),
            "mean_collapsed_old_basin_pref": mean_finite(collapsed_values),
            "mean_patched_old_basin_pref": mean_finite(patched_cp),
            "n": len(rows),
        },
    ]

    write_csv(out_dir / "axis-e-direction-patch.csv", rows)
    write_csv(out_dir / "axis-e-direction-patch-aggregate.csv", aggregate_rows)

    # v1 vs v2 comparison
    v1_layer_patch_baselines = {
        # From PHASE6_RESULTS.md §4 net.7 row
        "protected_to_collapsed": {"mean": 0.894, "median": 0.944, "ratio": 0.899},
        "collapsed_to_protected": {"mean": 0.934, "median": 0.860, "ratio": 0.854},
    }
    comparison_rows = []
    for agg in aggregate_rows:
        baseline = v1_layer_patch_baselines[agg["direction"]]
        comparison_rows.append(
            {
                "direction": agg["direction"],
                "feature_idx": feature_idx,
                "v1_layer_patch_mean": baseline["mean"],
                "v2_direction_patch_mean": agg["mean_patch_success"],
                "v1_layer_patch_median": baseline["median"],
                "v2_direction_patch_median": agg["median_patch_success"],
                "v1_layer_patch_ratio": baseline["ratio"],
                "v2_direction_patch_ratio": agg["patch_success_ratio_of_means"],
                "v2_minus_v1_mean": agg["mean_patch_success"] - baseline["mean"],
                "v2_minus_v1_median": agg["median_patch_success"] - baseline["median"],
            }
        )
    write_csv(out_dir / "v1-vs-v2-comparison.csv", comparison_rows)

    manifest = {
        "phase": "phase6-v2-axis-e-direction-patch",
        "feature_idx": int(feature_idx),
        "direction_norm": float(norm),
        "sae_dir": str(sae_dir.relative_to(REPO_ROOT)),
        "layer": args.layer,
        "seed_start": int(args.seed_start),
        "seeds": int(args.seeds),
        "horizon": int(args.horizon),
        "v1_comparison_baseline": v1_layer_patch_baselines,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"phase6 v2 axis-E: wrote results to {out_dir.relative_to(REPO_ROOT)}", flush=True)
    for agg in aggregate_rows:
        print(
            f"  {agg['direction']}: "
            f"mean={agg['mean_patch_success']:.3f} "
            f"median={agg['median_patch_success']:.3f} "
            f"ratio={agg['patch_success_ratio_of_means']:.3f}",
            flush=True,
        )
    print("phase6 v2 axis-E: v1 vs v2 comparison:", flush=True)
    for cmp in comparison_rows:
        print(
            f"  {cmp['direction']}: "
            f"v1_median={cmp['v1_layer_patch_median']:.3f} -> "
            f"v2_median={cmp['v2_direction_patch_median']:.3f} "
            f"(delta={cmp['v2_minus_v1_median']:+.3f})",
            flush=True,
        )


# ============================================================
# v3 subspace patching (Axes F, G, H)
# ============================================================


V1_LAYER_PATCH_BASELINES = {
    # From PHASE6_RESULTS.md §4 net.7 row
    "protected_to_collapsed": {"mean": 0.894, "median": 0.944, "ratio": 0.899},
    "collapsed_to_protected": {"mean": 0.934, "median": 0.860, "ratio": 0.854},
}


def build_orthonormal_subspace(directions: np.ndarray) -> tuple[np.ndarray, int]:
    """Given a (d_in, K) matrix whose columns are direction vectors, return an
    orthonormal basis Q (d_in, K_eff) spanning the same subspace, where
    K_eff <= K if the input was rank-deficient. Uses QR decomposition.

    Returns (Q, K_eff). Q columns are orthonormal; Q @ Q.T projects onto the
    subspace.
    """
    if directions.ndim != 2:
        raise ValueError(f"expected (d_in, K) matrix; got shape {directions.shape}")
    Q, R = np.linalg.qr(directions)
    diag = np.abs(np.diag(R))
    rank = int((diag > 1e-8 * diag.max() if diag.size else False).sum() if diag.size else 0)
    if rank == 0:
        raise RuntimeError("input directions matrix is degenerate")
    Q_eff = Q[:, :rank].astype(np.float32)
    return Q_eff, int(rank)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def activation_dim(policy: torch.nn.Module, obs_mean: np.ndarray, layer: str) -> int:
    captured: list[np.ndarray] = []
    module = get_module(policy, layer)

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured.append(output.detach().cpu().numpy().copy())

    handle = module.register_forward_hook(hook)
    try:
        obs = torch.tensor(obs_mean[None, :], dtype=torch.float32)
        with torch.no_grad():
            _ = policy(obs)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"layer {layer!r} did not emit an activation")
    return int(captured[0].shape[-1])


def validate_subspace_compatible(
    *,
    Q_np: np.ndarray,
    layer: str,
    protected_policy: torch.nn.Module,
    protected_mean: np.ndarray,
    protected_spec: PolicySpec,
    collapsed_policy: torch.nn.Module,
    collapsed_mean: np.ndarray,
    collapsed_spec: PolicySpec,
) -> None:
    q_dim = int(Q_np.shape[0])
    protected_dim = activation_dim(protected_policy, protected_mean, layer)
    collapsed_dim = activation_dim(collapsed_policy, collapsed_mean, layer)
    if protected_dim != q_dim or collapsed_dim != q_dim:
        raise ValueError(
            "subspace dimension mismatch: "
            f"Q has dim {q_dim}, {protected_spec.policy_id} {layer} has dim {protected_dim}, "
            f"{collapsed_spec.policy_id} {layer} has dim {collapsed_dim}. "
            "This usually means a Medium cliff-pair basis is being applied to a Small-tier policy."
        )


def compute_cliff_pca_basis(
    *,
    seed_start: int,
    seeds: int,
    horizon: int,
    num_components: int,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    print("phase6 v3.1: collecting matched-seed cliff-pair net.7 activations for PCA basis", flush=True)
    protected_acts = collect_cliff_activations(
        CLIFF_PROTECTED,
        seed_start=seed_start,
        seeds=seeds,
        horizon=horizon,
    )
    collapsed_acts = collect_cliff_activations(
        CLIFF_COLLAPSED,
        seed_start=seed_start,
        seeds=seeds,
        horizon=horizon,
    )

    diffs: list[np.ndarray] = []
    protected_by_seed: dict[int, list[np.ndarray]] = {}
    collapsed_by_seed: dict[int, list[np.ndarray]] = {}
    for seed, h in zip(protected_acts.seeds.tolist(), protected_acts.net7):
        protected_by_seed.setdefault(int(seed), []).append(h)
    for seed, h in zip(collapsed_acts.seeds.tolist(), collapsed_acts.net7):
        collapsed_by_seed.setdefault(int(seed), []).append(h)
    for seed in protected_by_seed.keys() & collapsed_by_seed.keys():
        prot_arr = np.stack(protected_by_seed[seed])
        coll_arr = np.stack(collapsed_by_seed[seed])
        n = min(prot_arr.shape[0], coll_arr.shape[0])
        if n:
            diffs.append(coll_arr[:n] - prot_arr[:n])
    diff_matrix = np.concatenate(diffs, axis=0).astype(np.float32)
    diff_centered = diff_matrix - diff_matrix.mean(axis=0, keepdims=True)
    _U, S, Vt = np.linalg.svd(diff_centered, full_matrices=False)
    k_use = min(int(num_components), Vt.shape[0])
    Q_np = Vt[:k_use, :].T.astype(np.float32)
    Q_np, k_eff = build_orthonormal_subspace(Q_np)

    total_var = float((S ** 2).sum())
    captured = float((S[:k_use] ** 2).sum()) / max(total_var, 1e-12)
    var_rows = []
    cumulative_var = 0.0
    for i, s in enumerate(S):
        cumulative_var += float(s) ** 2
        var_rows.append({
            "rank": i + 1,
            "singular_value": float(s),
            "variance": float(s) ** 2,
            "cumulative_variance_fraction": cumulative_var / max(total_var, 1e-12),
        })
    metadata = {
        "source": "recomputed_from_axis_h_matched_activation_collection",
        "seed_start": int(seed_start),
        "seeds": int(seeds),
        "horizon": int(horizon),
        "layer": "net.7",
        "num_components_requested": int(num_components),
        "k_used": int(k_use),
        "k_eff": int(k_eff),
        "diff_matrix_shape": list(diff_matrix.shape),
        "total_variance": total_var,
        "variance_captured_top_K": captured,
    }
    return Q_np, metadata, var_rows[:min(64, len(var_rows))]


def load_or_build_cliff_pca_basis(
    *,
    out_root: Path,
    seed_start: int,
    seeds: int,
    horizon: int,
    num_components: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    basis_dir = out_root / "pca-basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cliff-pca-net7-seed{seed_start}-n{seeds}-h{horizon}-k{num_components}"
    basis_path = basis_dir / f"{stem}.npz"
    manifest_path = basis_dir / f"{stem}.manifest.json"
    variance_path = basis_dir / f"{stem}.variance.csv"
    if basis_path.exists() and manifest_path.exists():
        data = np.load(basis_path)
        Q_np = data["Q"].astype(np.float32)
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"phase6 v3.1: loaded cached PCA basis from {basis_path.relative_to(REPO_ROOT)}", flush=True)
        return Q_np, metadata

    Q_np, metadata, var_rows = compute_cliff_pca_basis(
        seed_start=seed_start,
        seeds=seeds,
        horizon=horizon,
        num_components=num_components,
    )
    np.savez_compressed(basis_path, Q=Q_np)
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(variance_path, var_rows)
    print(
        f"phase6 v3.1: cached PCA basis at {basis_path.relative_to(REPO_ROOT)} "
        f"(K_eff={metadata['k_eff']}, variance={metadata['variance_captured_top_K']:.3%})",
        flush=True,
    )
    return Q_np, metadata


def run_subspace_recording_rollout(
    client: BridgeClient,
    *,
    policy: torch.nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seed: int,
    horizon: int,
    layer: str,
    env_id: str,
    Q: torch.Tensor,
) -> DirectionRolloutCache:
    """Run policy under live x_false intervention, recording the K-dim coordinate
    of `layer`'s output in the orthonormal subspace `Q` per step. No
    modification."""
    made = client.request(
        {
            "cmd": "make",
            "env_id": env_id,
            "seed": seed,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": horizon},
            "interventions": [basin_position_intervention()],
        }
    )
    obs = np.asarray(made["obs"], dtype=np.float32)
    info = made["info"]
    terminal_position = np.asarray(info["position"], dtype=np.float32)
    terminal_outcome = "not_done"
    coords: list[np.ndarray] = []
    step_index = 0

    module = get_module(policy, layer)

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        # output shape: (1, d_in)
        h = output.squeeze(0)
        c = (Q.T @ h).detach().cpu().numpy().copy()  # (K_eff,)
        coords.append(c)

    handle = module.register_forward_hook(hook)
    try:
        for _ in range(horizon + 1):
            action = learned_action(policy, obs, obs_mean, obs_std)
            response = client.request({"cmd": "step", "env_id": env_id, "action": action.tolist()})
            obs = np.asarray(response["obs"], dtype=np.float32)
            info = response["info"]
            terminal_position = np.asarray(info["position"], dtype=np.float32)
            step_index += 1
            if response["done"]:
                terminal_outcome = str(info.get("terminal_outcome") or "done")
                break
    finally:
        handle.remove()

    return DirectionRolloutCache(
        old_basin_pref=old_basin_pref(terminal_position),
        terminal_position=terminal_position,
        terminal_outcome=terminal_outcome,
        steps=step_index,
        # Store coords as a list of per-step K_eff vectors (each np.ndarray).
        # We reuse the `projections` field of DirectionRolloutCache; the type
        # here is list[np.ndarray] rather than list[float].
        projections=coords,  # type: ignore[arg-type]
    )


def run_subspace_injected_rollout(
    client: BridgeClient,
    *,
    policy: torch.nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seed: int,
    horizon: int,
    layer: str,
    env_id: str,
    Q: torch.Tensor,
    target_coords: list[np.ndarray],
    neuron_mask: torch.Tensor | None = None,
    zero_ablate_neuron: int | None = None,
    zero_ablate_neuron_set: list[int] | None = None,
) -> DirectionRolloutCache:
    """Run policy under live x_false intervention, substituting only the
    projection of `layer`'s output onto the subspace `Q` with `target_coords`
    at each step. The orthogonal complement (255+ dims of net.7) is preserved.

    If `neuron_mask` is provided (a {0, 1} mask over `d_in`), the subspace
    delta is multiplied by the mask before being added to `h`. This restricts
    the patch to the masked neurons — used by Phase 6 v3.2 Axis M to test
    top-k neuron mediation of the v3 PCA subspace.
    """
    if not target_coords:
        raise RuntimeError("cannot inject empty target_coords cache")
    if zero_ablate_neuron is not None and zero_ablate_neuron_set is not None:
        raise ValueError("zero_ablate_neuron and zero_ablate_neuron_set are mutually exclusive")
    made = client.request(
        {
            "cmd": "make",
            "env_id": env_id,
            "seed": seed,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": horizon},
            "interventions": [basin_position_intervention()],
        }
    )
    obs = np.asarray(made["obs"], dtype=np.float32)
    info = made["info"]
    terminal_position = np.asarray(info["position"], dtype=np.float32)
    terminal_outcome = "not_done"
    realized_coords: list[np.ndarray] = []
    step_index = 0

    module = get_module(policy, layer)

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        idx = min(step_index, len(target_coords) - 1)
        target_c = torch.tensor(target_coords[idx], dtype=output.dtype, device=output.device)
        h = output.squeeze(0)
        current_c = Q.T @ h  # (K_eff,)
        delta = Q @ (target_c - current_c)  # (d_in,)
        if neuron_mask is not None:
            delta = delta * neuron_mask
        h_new = h + delta
        if zero_ablate_neuron is not None:
            h_new = h_new.clone()
            h_new[int(zero_ablate_neuron)] = 0.0
        if zero_ablate_neuron_set is not None:
            h_new = h_new.clone()
            h_new[torch.tensor(zero_ablate_neuron_set, dtype=torch.long, device=h_new.device)] = 0.0
        realized_coords.append((Q.T @ h_new).detach().cpu().numpy().copy())
        return h_new.unsqueeze(0)

    handle = module.register_forward_hook(hook)
    try:
        for _ in range(horizon + 1):
            action = learned_action(policy, obs, obs_mean, obs_std)
            response = client.request({"cmd": "step", "env_id": env_id, "action": action.tolist()})
            obs = np.asarray(response["obs"], dtype=np.float32)
            info = response["info"]
            terminal_position = np.asarray(info["position"], dtype=np.float32)
            step_index += 1
            if response["done"]:
                terminal_outcome = str(info.get("terminal_outcome") or "done")
                break
    finally:
        handle.remove()

    return DirectionRolloutCache(
        old_basin_pref=old_basin_pref(terminal_position),
        terminal_position=terminal_position,
        terminal_outcome=terminal_outcome,
        steps=step_index,
        projections=realized_coords,  # type: ignore[arg-type]
    )


def run_subspace_patch_battery(
    *,
    Q_np: np.ndarray,
    label: str,
    seed_start: int,
    seeds: int,
    horizon: int,
    layer: str,
    out_dir: Path,
    manifest_extra: dict[str, Any],
    protected_spec: PolicySpec = CLIFF_PROTECTED,
    collapsed_spec: PolicySpec = CLIFF_COLLAPSED,
    neuron_mask_np: np.ndarray | None = None,
) -> None:
    """Run the 4-forward direction-patch battery using subspace Q across the
    matched seed slate. Writes per-seed CSV, aggregate CSV, and v1-comparison
    CSV under out_dir.

    If `neuron_mask_np` is provided (a {0, 1} mask over `d_in`), the subspace
    delta is restricted to the masked neurons before being applied. Phase 6
    v3.2 Axis M uses this to test top-k neuron mediation of the v3 PCA
    subspace.
    """
    Q = torch.tensor(Q_np, dtype=torch.float32)
    K_eff = int(Q_np.shape[1])
    neuron_mask = (
        torch.tensor(neuron_mask_np.astype(np.float32))
        if neuron_mask_np is not None
        else None
    )
    protected_policy, protected_mean, protected_std = load_learned_policy(protected_spec)
    collapsed_policy, collapsed_mean, collapsed_std = load_learned_policy(collapsed_spec)
    validate_subspace_compatible(
        Q_np=Q_np,
        layer=layer,
        protected_policy=protected_policy,
        protected_mean=protected_mean,
        protected_spec=protected_spec,
        collapsed_policy=collapsed_policy,
        collapsed_mean=collapsed_mean,
        collapsed_spec=collapsed_spec,
    )

    rows: list[dict[str, Any]] = []
    with BridgeClient() as client:
        for offset in range(seeds):
            seed = seed_start + offset
            prefix = f"phase6-subspace-{label}-{seed}"

            cache_A = run_subspace_recording_rollout(
                client,
                policy=protected_policy,
                obs_mean=protected_mean,
                obs_std=protected_std,
                seed=seed,
                horizon=horizon,
                layer=layer,
                env_id=f"{prefix}-A",
                Q=Q,
            )
            cache_B = run_subspace_recording_rollout(
                client,
                policy=collapsed_policy,
                obs_mean=collapsed_mean,
                obs_std=collapsed_std,
                seed=seed,
                horizon=horizon,
                layer=layer,
                env_id=f"{prefix}-B",
                Q=Q,
            )
            cache_C = run_subspace_injected_rollout(
                client,
                policy=collapsed_policy,
                obs_mean=collapsed_mean,
                obs_std=collapsed_std,
                seed=seed,
                horizon=horizon,
                layer=layer,
                env_id=f"{prefix}-C",
                Q=Q,
                target_coords=cache_A.projections,  # type: ignore[arg-type]
                neuron_mask=neuron_mask,
            )
            cache_D = run_subspace_injected_rollout(
                client,
                policy=protected_policy,
                obs_mean=protected_mean,
                obs_std=protected_std,
                seed=seed,
                horizon=horizon,
                layer=layer,
                env_id=f"{prefix}-D",
                Q=Q,
                target_coords=cache_B.projections,  # type: ignore[arg-type]
                neuron_mask=neuron_mask,
            )

            success_pc = safe_patch_success(
                cache_A.old_basin_pref,
                cache_B.old_basin_pref,
                cache_C.old_basin_pref,
                direction="protected_to_collapsed",
            )
            success_cp = safe_patch_success(
                cache_A.old_basin_pref,
                cache_B.old_basin_pref,
                cache_D.old_basin_pref,
                direction="collapsed_to_protected",
            )
            rows.append(
                {
                    "seed": seed,
                    "k_eff": K_eff,
                    "layer": layer,
                    "protected_old_basin_pref": cache_A.old_basin_pref,
                    "collapsed_old_basin_pref": cache_B.old_basin_pref,
                    "patched_protected_to_collapsed_old_basin_pref": cache_C.old_basin_pref,
                    "patched_collapsed_to_protected_old_basin_pref": cache_D.old_basin_pref,
                    "patch_success_protected_to_collapsed": success_pc,
                    "patch_success_collapsed_to_protected": success_cp,
                    "baseline_gap_collapsed_minus_protected": cache_B.old_basin_pref - cache_A.old_basin_pref,
                }
            )
        client.request({"cmd": "close"})

    # Aggregate
    protected_values = [float(r["protected_old_basin_pref"]) for r in rows]
    collapsed_values = [float(r["collapsed_old_basin_pref"]) for r in rows]
    patched_pc = [float(r["patched_protected_to_collapsed_old_basin_pref"]) for r in rows]
    patched_cp = [float(r["patched_collapsed_to_protected_old_basin_pref"]) for r in rows]
    success_pc_values = [float(r["patch_success_protected_to_collapsed"]) for r in rows]
    success_cp_values = [float(r["patch_success_collapsed_to_protected"]) for r in rows]

    aggregate_rows = [
        {
            "direction": "protected_to_collapsed",
            "k_eff": K_eff,
            "layer": layer,
            "mean_patch_success": mean_finite(success_pc_values),
            "median_patch_success": median_finite(success_pc_values),
            "patch_success_ratio_of_means": ratio_of_means(
                protected_values, collapsed_values, patched_pc,
                direction="protected_to_collapsed",
            ),
            "mean_protected_old_basin_pref": mean_finite(protected_values),
            "mean_collapsed_old_basin_pref": mean_finite(collapsed_values),
            "mean_patched_old_basin_pref": mean_finite(patched_pc),
            "n": len(rows),
        },
        {
            "direction": "collapsed_to_protected",
            "k_eff": K_eff,
            "layer": layer,
            "mean_patch_success": mean_finite(success_cp_values),
            "median_patch_success": median_finite(success_cp_values),
            "patch_success_ratio_of_means": ratio_of_means(
                protected_values, collapsed_values, patched_cp,
                direction="collapsed_to_protected",
            ),
            "mean_protected_old_basin_pref": mean_finite(protected_values),
            "mean_collapsed_old_basin_pref": mean_finite(collapsed_values),
            "mean_patched_old_basin_pref": mean_finite(patched_cp),
            "n": len(rows),
        },
    ]
    write_csv(out_dir / f"{label}-patch.csv", rows)
    write_csv(out_dir / f"{label}-patch-aggregate.csv", aggregate_rows)

    # v1 comparison
    comparison_rows = []
    for agg in aggregate_rows:
        baseline = V1_LAYER_PATCH_BASELINES[agg["direction"]]
        comparison_rows.append({
            "direction": agg["direction"],
            "k_eff": K_eff,
            "v1_layer_patch_mean": baseline["mean"],
            "v3_subspace_patch_mean": agg["mean_patch_success"],
            "v1_layer_patch_median": baseline["median"],
            "v3_subspace_patch_median": agg["median_patch_success"],
            "v1_layer_patch_ratio": baseline["ratio"],
            "v3_subspace_patch_ratio": agg["patch_success_ratio_of_means"],
            "v3_minus_v1_mean": agg["mean_patch_success"] - baseline["mean"],
            "v3_minus_v1_median": agg["median_patch_success"] - baseline["median"],
        })
    write_csv(out_dir / "v1-vs-v3-comparison.csv", comparison_rows)

    manifest = {
        "phase": f"phase6-v3-{label}",
        "k_eff": K_eff,
        "layer": layer,
        "seed_start": int(seed_start),
        "seeds": int(seeds),
        "horizon": int(horizon),
        "protected_policy_id": protected_spec.policy_id,
        "protected_label": protected_spec.label,
        "collapsed_policy_id": collapsed_spec.policy_id,
        "collapsed_label": collapsed_spec.label,
        "v1_comparison_baseline": V1_LAYER_PATCH_BASELINES,
        **manifest_extra,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"phase6 v3 {label}: K_eff={K_eff}; wrote results to {out_dir.relative_to(REPO_ROOT)}", flush=True)
    for agg in aggregate_rows:
        print(
            f"  {agg['direction']}: "
            f"mean={agg['mean_patch_success']:+.3f} "
            f"median={agg['median_patch_success']:+.3f} "
            f"ratio={agg['patch_success_ratio_of_means']:+.3f}",
            flush=True,
        )
    print(f"phase6 v3 {label}: v1 vs v3 comparison:", flush=True)
    for cmp in comparison_rows:
        print(
            f"  {cmp['direction']}: "
            f"v1_median={cmp['v1_layer_patch_median']:.3f} -> "
            f"v3_median={cmp['v3_subspace_patch_median']:+.3f} "
            f"(delta={cmp['v3_minus_v1_median']:+.3f})",
            flush=True,
        )


# --- Axis F: multi-feature SAE subspace patching ---

def axis_f_multifeature_patch(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE
    sae_dir = Path(args.sae_dir)
    if not sae_dir.is_absolute():
        sae_dir = REPO_ROOT / sae_dir
    config = json.loads((sae_dir / "sae-config.json").read_text(encoding="utf-8"))
    sae = TopKSAE(
        d_in=int(config["d_in"]),
        n_features=int(config["n_features"]),
        k=int(config["k"]),
    )
    sae.load_state_dict(torch.load(sae_dir / "sae-weights.pt", weights_only=True))
    sae.eval()

    # Load correlations and pick top-K
    corr_rows = list(csv.DictReader((sae_dir / "axis-d-feature-correlations.csv").open(encoding="utf-8")))
    corr_rows.sort(key=lambda r: -abs(float(r["correlation"])))
    K = int(args.num_features)
    top_feature_indices = [int(r["feature_idx"]) for r in corr_rows[:K]]

    # Stack decoder columns and QR-orthogonalize
    directions = sae.decoder.weight.data[:, top_feature_indices].numpy().astype(np.float32)
    Q_np, K_eff = build_orthonormal_subspace(directions)
    print(
        f"phase6 v3 axis-F: top {K} SAE features -> K_eff={K_eff} after orthogonalization "
        f"(top correlations: {[round(float(r['correlation']), 3) for r in corr_rows[:min(K, 5)]]})",
        flush=True,
    )

    manifest_extra = {
        "top_feature_indices": top_feature_indices,
        "top_feature_correlations": [float(corr_rows[i]["correlation"]) for i in range(min(K, len(corr_rows)))],
        "K_requested": K,
        "sae_dir": str(sae_dir.relative_to(REPO_ROOT)),
    }
    run_subspace_patch_battery(
        Q_np=Q_np,
        label="axis-f-multifeature",
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        layer=args.layer,
        out_dir=out_dir,
        manifest_extra=manifest_extra,
    )


# --- Axis G: empirical between-policy mean-difference direction ---

def axis_g_mean_diff_patch(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("phase6 v3 axis-G: collecting cliff-pair net.7 activations for mean-diff direction", flush=True)
    protected_acts = collect_cliff_activations(
        CLIFF_PROTECTED,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
    )
    collapsed_acts = collect_cliff_activations(
        CLIFF_COLLAPSED,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
    )
    mean_protected = protected_acts.net7.mean(axis=0)
    mean_collapsed = collapsed_acts.net7.mean(axis=0)
    delta = mean_collapsed - mean_protected  # (d_in,)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        raise RuntimeError("between-policy mean-diff direction has near-zero norm")
    Q_np = (delta / norm).astype(np.float32).reshape(-1, 1)  # (d_in, 1)

    print(
        f"phase6 v3 axis-G: ||mean_collapsed - mean_protected|| = {norm:.4f} "
        f"(d_in={Q_np.shape[0]}, K=1)",
        flush=True,
    )

    manifest_extra = {
        "delta_norm": norm,
        "mean_protected_norm": float(np.linalg.norm(mean_protected)),
        "mean_collapsed_norm": float(np.linalg.norm(mean_collapsed)),
    }
    run_subspace_patch_battery(
        Q_np=Q_np,
        label="axis-g-mean-diff",
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        layer=args.layer,
        out_dir=out_dir,
        manifest_extra=manifest_extra,
    )


# --- Axis H: PCA on per-step diffs ---

def axis_h_pca_patch(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("phase6 v3 axis-H: collecting matched-seed cliff-pair net.7 activations", flush=True)
    protected_acts = collect_cliff_activations(
        CLIFF_PROTECTED,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
    )
    collapsed_acts = collect_cliff_activations(
        CLIFF_COLLAPSED,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
    )

    # Per-step diff aligned by (seed, step_index_within_seed). Both policies
    # ran on the same matched-seed slate; trajectories diverge by step but the
    # step indices align. Group by seed and pair up by min(len, len).
    diffs: list[np.ndarray] = []
    protected_by_seed: dict[int, list[np.ndarray]] = {}
    collapsed_by_seed: dict[int, list[np.ndarray]] = {}
    for seed, h in zip(protected_acts.seeds.tolist(), protected_acts.net7):
        protected_by_seed.setdefault(int(seed), []).append(h)
    for seed, h in zip(collapsed_acts.seeds.tolist(), collapsed_acts.net7):
        collapsed_by_seed.setdefault(int(seed), []).append(h)
    for seed in protected_by_seed.keys() & collapsed_by_seed.keys():
        prot_arr = np.stack(protected_by_seed[seed])
        coll_arr = np.stack(collapsed_by_seed[seed])
        n = min(prot_arr.shape[0], coll_arr.shape[0])
        if n == 0:
            continue
        diffs.append(coll_arr[:n] - prot_arr[:n])
    diff_matrix = np.concatenate(diffs, axis=0).astype(np.float32)  # (N_steps, d_in)

    # PCA via SVD on centered per-step diffs
    diff_centered = diff_matrix - diff_matrix.mean(axis=0, keepdims=True)
    # full_matrices=False so we get (N, min(N, d_in)) singular vectors
    U, S, Vt = np.linalg.svd(diff_centered, full_matrices=False)
    # Vt rows are principal directions in d_in space
    K = int(args.num_components)
    K_use = min(K, Vt.shape[0])
    Q_np = Vt[:K_use, :].T.astype(np.float32)  # (d_in, K_use)
    Q_np, K_eff = build_orthonormal_subspace(Q_np)

    # Variance-captured diagnostics
    total_var = float((S ** 2).sum())
    captured = float((S[:K_use] ** 2).sum()) / max(total_var, 1e-12)
    print(
        f"phase6 v3 axis-H: per-step diff matrix shape={diff_matrix.shape}; "
        f"using top-{K_use} PCA components capturing {captured:.3%} of total variance "
        f"(K_eff={K_eff})",
        flush=True,
    )

    # Write the variance-explained curve as a small CSV
    var_rows = []
    cumulative_var = 0.0
    for i, s in enumerate(S):
        cumulative_var += float(s) ** 2
        var_rows.append({
            "rank": i + 1,
            "singular_value": float(s),
            "variance": float(s) ** 2,
            "cumulative_variance_fraction": cumulative_var / max(total_var, 1e-12),
        })
    write_csv(out_dir / "axis-h-pca-variance.csv", var_rows[:min(64, len(var_rows))])

    manifest_extra = {
        "diff_matrix_shape": list(diff_matrix.shape),
        "K_requested": K,
        "K_used": K_use,
        "variance_captured_top_K": captured,
        "total_variance": total_var,
    }
    run_subspace_patch_battery(
        Q_np=Q_np,
        label="axis-h-pca",
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        layer=args.layer,
        out_dir=out_dir,
        manifest_extra=manifest_extra,
    )


# ============================================================
# v3.1 validation axes (I, J, K, L)
# ============================================================


def axis_i_pc_mech(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    Q_full, basis_metadata = load_or_build_cliff_pca_basis(
        out_root=PHASE6_V31_OUT,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        num_components=5,
    )
    if Q_full.shape[1] < 5:
        raise RuntimeError(f"expected at least 5 PCA columns, got {Q_full.shape[1]}")
    Q_pc2_to_5, k_eff = build_orthonormal_subspace(Q_full[:, 1:5])
    print(f"phase6 v3.1 axis-I: PC2-5 basis K_eff={k_eff}", flush=True)

    run_subspace_patch_battery(
        Q_np=Q_pc2_to_5,
        label="pc-mech",
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        layer=args.layer,
        out_dir=out_dir,
        manifest_extra={
            "axis": "I",
            "basis": "cliff_pair_pca_pc2_to_5",
            "basis_metadata": basis_metadata,
        },
    )

    baseline_path = PHASE6_V2_OUT / "axis-h-pca-k5" / "axis-h-pca-patch-aggregate.csv"
    current_path = out_dir / "pc-mech-patch-aggregate.csv"
    baseline_rows = {row["direction"]: row for row in read_csv_rows(baseline_path)}
    current_rows = read_csv_rows(current_path)
    comparison = []
    for row in current_rows:
        direction = row["direction"]
        baseline = baseline_rows[direction]
        comparison.append({
            "direction": direction,
            "baseline_k5_mean": float(baseline["mean_patch_success"]),
            "pc2_to_5_mean": float(row["mean_patch_success"]),
            "mean_delta": float(row["mean_patch_success"]) - float(baseline["mean_patch_success"]),
            "baseline_k5_median": float(baseline["median_patch_success"]),
            "pc2_to_5_median": float(row["median_patch_success"]),
            "median_delta": float(row["median_patch_success"]) - float(baseline["median_patch_success"]),
            "baseline_k5_ratio": float(baseline["patch_success_ratio_of_means"]),
            "pc2_to_5_ratio": float(row["patch_success_ratio_of_means"]),
            "ratio_delta": float(row["patch_success_ratio_of_means"]) - float(baseline["patch_success_ratio_of_means"]),
        })
    write_csv(out_dir / "v3-vs-v3-1-comparison.csv", comparison)


def axis_j_generalization(args: argparse.Namespace) -> None:
    pair_key = args.pair.upper()
    if pair_key not in V31_GENERALIZATION_PAIRS:
        raise ValueError(f"unknown pair {args.pair!r}; expected one of {sorted(V31_GENERALIZATION_PAIRS)}")
    if pair_key == "J3":
        raise ValueError(
            "J3 is dimension-blocked in Phase 6 v3.1: the Medium cliff-pair PCA basis is 256D, "
            "while the Small-tier J3 policies expose a 64D final hidden activation. "
            "Run J3 only after adding a Small-tier basis or cross-tier adapter in v3.2."
        )
    protected_spec, collapsed_spec = V31_GENERALIZATION_PAIRS[pair_key]

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / pair_key.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    Q_full, basis_metadata = load_or_build_cliff_pca_basis(
        out_root=PHASE6_V31_OUT,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        num_components=5,
    )
    print(
        f"phase6 v3.1 axis-J {pair_key}: {protected_spec.policy_id} -> {collapsed_spec.policy_id}",
        flush=True,
    )
    run_subspace_patch_battery(
        Q_np=Q_full,
        label=pair_key.lower(),
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        layer=args.layer,
        out_dir=out_dir,
        manifest_extra={
            "axis": "J",
            "pair": pair_key,
            "basis": "cliff_pair_pca_pc1_to_5",
            "basis_metadata": basis_metadata,
        },
        protected_spec=protected_spec,
        collapsed_spec=collapsed_spec,
    )

    summary_rows = []
    for pair in sorted(p.name for p in out_root.iterdir() if p.is_dir() and (p / f"{p.name}-patch-aggregate.csv").exists()):
        for row in read_csv_rows(out_root / pair / f"{pair}-patch-aggregate.csv"):
            summary_rows.append({
                "pair": pair.upper(),
                "direction": row["direction"],
                "k_eff": row["k_eff"],
                "layer": row["layer"],
                "mean_patch_success": row["mean_patch_success"],
                "median_patch_success": row["median_patch_success"],
                "patch_success_ratio_of_means": row["patch_success_ratio_of_means"],
                "mean_protected_old_basin_pref": row["mean_protected_old_basin_pref"],
                "mean_collapsed_old_basin_pref": row["mean_collapsed_old_basin_pref"],
                "mean_patched_old_basin_pref": row["mean_patched_old_basin_pref"],
                "n": row["n"],
            })
    write_csv(out_root / "generalization-summary.csv", summary_rows)


def axis_k_decompose(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    Q_full, basis_metadata = load_or_build_cliff_pca_basis(
        out_root=PHASE6_V31_OUT,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        num_components=5,
    )

    sparsity_rows = []
    top8_rows = []
    top32_sets: dict[int, set[int]] = {}
    for pc_index in range(Q_full.shape[1]):
        v = Q_full[:, pc_index].astype(np.float64)
        energy = v ** 2
        total_energy = float(energy.sum())
        order = np.argsort(-np.abs(v))
        top8 = order[:8]
        top16 = order[:16]
        top32 = order[:32]
        top32_sets[pc_index + 1] = set(int(i) for i in top32.tolist())
        sparsity_rows.append({
            "pc": pc_index + 1,
            "l2_concentration_top8": float(energy[top8].sum() / max(total_energy, 1e-12)),
            "l2_concentration_top16": float(energy[top16].sum() / max(total_energy, 1e-12)),
            "l2_concentration_top32": float(energy[top32].sum() / max(total_energy, 1e-12)),
            "participation_ratio": float((total_energy ** 2) / max(float((energy ** 2).sum()), 1e-12)),
            "max_abs_weight": float(np.max(np.abs(v))),
            "basis_metadata_variance_captured_top_K": basis_metadata.get("variance_captured_top_K", ""),
        })
        top8_rows.append({
            "pc": pc_index + 1,
            "top8_neuron_indices": ";".join(str(int(i)) for i in top8.tolist()),
            "top8_abs_weights": ";".join(f"{abs(float(v[i])):.8g}" for i in top8.tolist()),
        })

    overlap_rows = []
    pcs = sorted(top32_sets)
    for i, pc_a in enumerate(pcs):
        for pc_b in pcs[i + 1:]:
            a = top32_sets[pc_a]
            b = top32_sets[pc_b]
            overlap_rows.append({
                "pc_a": pc_a,
                "pc_b": pc_b,
                "top32_jaccard": len(a & b) / max(len(a | b), 1),
                "top32_overlap_count": len(a & b),
            })

    write_csv(out_dir / "pc-sparsity-table.csv", sparsity_rows)
    write_csv(out_dir / "pc-top8-neurons.csv", top8_rows)
    write_csv(out_dir / "pc-overlap-jaccard.csv", overlap_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps({"axis": "K", "basis_metadata": basis_metadata}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"phase6 v3.1 axis-K: wrote results to {out_dir.relative_to(REPO_ROOT)}", flush=True)


def bootstrap_gap_for_k(*, k: int, resamples: int, seed: int, out_dir: Path) -> dict[str, Any]:
    source = PHASE6_V2_OUT / f"axis-h-pca-k{k}" / "axis-h-pca-patch.csv"
    rows = read_csv_rows(source)
    paired = []
    for row in rows:
        pc = float(row["patch_success_protected_to_collapsed"])
        cp = float(row["patch_success_collapsed_to_protected"])
        if math.isfinite(pc) and math.isfinite(cp):
            paired.append((pc, cp))
    if not paired:
        raise RuntimeError(f"no finite paired patch-success rows in {source}")
    arr = np.asarray(paired, dtype=np.float64)
    observed_gap = float(np.median(arr[:, 0]) - np.median(arr[:, 1]))
    rng = np.random.default_rng(seed + k * 1009)
    boot_rows = []
    gaps = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, arr.shape[0], size=arr.shape[0])
        sample = arr[idx]
        gap = float(np.median(sample[:, 0]) - np.median(sample[:, 1]))
        gaps[i] = gap
        boot_rows.append({"resample": i, "median_gap": gap})
    write_csv(out_dir / f"bootstrap-gap-k{k}.csv", boot_rows)
    return {
        "k": k,
        "source": str(source.relative_to(REPO_ROOT)),
        "n": int(arr.shape[0]),
        "observed_median_protected_to_collapsed": float(np.median(arr[:, 0])),
        "observed_median_collapsed_to_protected": float(np.median(arr[:, 1])),
        "observed_median_gap": observed_gap,
        "bootstrap_median_gap_mean": float(gaps.mean()),
        "bootstrap_median_gap_lo_95": float(np.percentile(gaps, 2.5)),
        "bootstrap_median_gap_hi_95": float(np.percentile(gaps, 97.5)),
        "resamples": int(resamples),
    }


def axis_l_bootstrap(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(k) for k in args.k]
    summaries = [
        bootstrap_gap_for_k(k=k, resamples=args.resamples, seed=args.bootstrap_seed, out_dir=out_dir)
        for k in ks
    ]
    (out_dir / "bootstrap-summary.json").write_text(
        json.dumps({"axis": "L", "summaries": summaries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"phase6 v3.1 axis-L: wrote bootstrap summary to {out_dir.relative_to(REPO_ROOT)}", flush=True)
    for summary in summaries:
        print(
            f"  K={summary['k']}: observed_gap={summary['observed_median_gap']:+.3f}, "
            f"95% CI [{summary['bootstrap_median_gap_lo_95']:+.3f}, {summary['bootstrap_median_gap_hi_95']:+.3f}]",
            flush=True,
        )


# ============================================================
# v3.2 Axis M — Top-k neuron mediation
# ============================================================


def compute_neuron_ranking(Q_cliff: np.ndarray) -> np.ndarray:
    """Rank neurons at the patched layer by aggregate L2 contribution across
    the columns of `Q_cliff` (the v3 PCA basis). Returns neuron indices sorted
    descending by `score[j] = Σ_i v_ij²`. The top-k entries are the neurons
    most heavily involved across the basin-attractor subspace.
    """
    if Q_cliff.ndim != 2:
        raise ValueError(f"expected (d_in, K) basis; got {Q_cliff.shape}")
    scores = (Q_cliff ** 2).sum(axis=1)  # (d_in,)
    return np.argsort(-scores)


def axis_m_neuron_mediation(args: argparse.Namespace) -> None:
    """Phase 6 v3.2 Axis M: restrict the v3 K=5 PCA patch delta to top-k
    neurons at net.7, parameterized by --top-k. Optional --pair routes the
    same neuron mask to the v3.1 J1/J2 held-out pairs."""
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    pair_key = "cliff" if args.pair in (None, "cliff") else args.pair.upper()
    if pair_key not in {"cliff", "J1", "J2"}:
        raise ValueError(f"unknown pair {args.pair!r}; expected one of {{cliff, J1, J2}}")
    if pair_key == "cliff":
        protected_spec, collapsed_spec = CLIFF_PROTECTED, CLIFF_COLLAPSED
    else:
        protected_spec, collapsed_spec = V31_GENERALIZATION_PAIRS[pair_key]

    # Load (or build) the cliff-pair PCA basis (the v3 K=5 artifact).
    Q_full, basis_metadata = load_or_build_cliff_pca_basis(
        out_root=PHASE6_V31_OUT,
        seed_start=args.basis_seed_start,
        seeds=args.basis_seeds,
        horizon=args.basis_horizon,
        num_components=5,
    )
    d_in = int(Q_full.shape[0])
    top_k = int(args.top_k)
    if top_k <= 0 or top_k > d_in:
        raise ValueError(f"--top-k must be in [1, {d_in}], got {top_k}")

    ranking = compute_neuron_ranking(Q_full)
    if args.neuron_mask_source:
        mask_source = Path(args.neuron_mask_source)
        if not mask_source.is_absolute():
            mask_source = REPO_ROOT / mask_source
        source_rows = read_csv_rows(mask_source)
        if not source_rows or "neuron_idx" not in source_rows[0]:
            raise ValueError(f"{mask_source} must contain a neuron_idx column")
        top_k_indices = [int(row["neuron_idx"]) for row in source_rows[:top_k]]
        if len(top_k_indices) < top_k:
            raise ValueError(
                f"{mask_source} contains {len(top_k_indices)} neuron ids, fewer than --top-k {top_k}"
            )
        ranking_for_output = np.asarray(top_k_indices, dtype=int)
        mask_source_label = str(mask_source.relative_to(REPO_ROOT)) if mask_source.is_relative_to(REPO_ROOT) else str(mask_source)
    else:
        top_k_indices = ranking[:top_k].tolist()
        ranking_for_output = ranking[:top_k]
        mask_source_label = "aggregate_l2"

    neuron_mask_np = np.zeros(d_in, dtype=np.float32)
    neuron_mask_np[np.asarray(top_k_indices, dtype=int)] = 1.0

    # Per-PC contribution captured by the mask, as a diagnostic.
    captured_l2_per_pc = []
    total_l2_per_pc = []
    for i in range(Q_full.shape[1]):
        v = Q_full[:, i]
        total = float((v ** 2).sum())
        captured = float((v[np.asarray(top_k_indices, dtype=int)] ** 2).sum())
        captured_l2_per_pc.append(captured)
        total_l2_per_pc.append(total)
    captured_total = float(sum(captured_l2_per_pc))
    total_total = float(sum(total_l2_per_pc))
    captured_fraction = captured_total / max(total_total, 1e-12)

    label = f"top-{top_k}"
    pair_dir = "axis-m-cliff-pair" if pair_key == "cliff" else f"axis-m-{pair_key}"
    out_dir = out_root / pair_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"phase6 v3.2 axis-M ({pair_key} pair, top-{top_k}): "
        f"d_in={d_in}, captured_fraction={captured_fraction:.4f}",
        flush=True,
    )
    print(
        "  top-8 neurons: " + ",".join(str(int(idx)) for idx in ranking_for_output[:8]),
        flush=True,
    )
    print(
        "  top-32 neurons: " + ",".join(str(int(idx)) for idx in ranking_for_output[:32]),
        flush=True,
    )
    # Write the neuron-id list and the per-PC capture diagnostic up front.
    neuron_rows = [
        {
            "rank": rank + 1,
            "neuron_idx": int(idx),
            "aggregate_l2_score": float((Q_full[int(idx), :] ** 2).sum()),
        }
        for rank, idx in enumerate(top_k_indices)
    ]
    write_csv(out_dir / f"neuron-ids-top-{top_k}.csv", neuron_rows)
    pc_capture_rows = [
        {
            "pc_index": i + 1,
            "captured_l2": captured_l2_per_pc[i],
            "total_l2": total_l2_per_pc[i],
            "capture_fraction": captured_l2_per_pc[i] / max(total_l2_per_pc[i], 1e-12),
        }
        for i in range(Q_full.shape[1])
    ]
    write_csv(out_dir / "pc-l2-capture.csv", pc_capture_rows)

    run_subspace_patch_battery(
        Q_np=Q_full,
        label=label,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        layer=args.layer,
        out_dir=out_dir,
        manifest_extra={
            "axis": "M",
            "pair": pair_key,
            "top_k": top_k,
            "d_in": d_in,
            "captured_l2_fraction": captured_fraction,
            "top_k_neuron_indices": top_k_indices,
            "basis": "cliff_pair_pca_pc1_to_5",
            "basis_metadata": basis_metadata,
            "basis_seed_start": int(args.basis_seed_start),
            "basis_seeds": int(args.basis_seeds),
            "basis_horizon": int(args.basis_horizon),
            "neuron_mask_source": mask_source_label,
        },
        protected_spec=protected_spec,
        collapsed_spec=collapsed_spec,
        neuron_mask_np=neuron_mask_np,
    )


# ============================================================
# v3.3 Axis N — zero-ablation attribution
# ============================================================


def percentile_finite(values: list[float], q: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float("nan") if not finite else float(np.percentile(finite, q))


def jaccard(a: set[int], b: set[int]) -> float:
    return float("nan") if not (a or b) else len(a & b) / max(len(a | b), 1)


def direction_keys(direction_arg: str) -> list[str]:
    direction = direction_arg.strip()
    if direction == "both":
        return ["protected_to_collapsed", "collapsed_to_protected"]
    if direction == "P_to_C":
        return ["protected_to_collapsed"]
    if direction == "C_to_P":
        return ["collapsed_to_protected"]
    raise ValueError(f"unknown direction {direction_arg!r}; expected P_to_C, C_to_P, or both")


def run_zero_ablation_battery(
    *,
    Q_np: np.ndarray,
    seed_start: int,
    seeds: int,
    horizon: int,
    layer: str,
    out_dir: Path,
    direction: str,
    manifest_extra: dict[str, Any],
    protected_spec: PolicySpec = CLIFF_PROTECTED,
    collapsed_spec: PolicySpec = CLIFF_COLLAPSED,
) -> None:
    """Run Phase 6 v3.3 Axis N zero-ablation attribution.

    For each seed, record the protected/collapsed subspace-coordinate caches
    once, run the unablated K=5 patch once per requested direction, then rerun
    the patch with each individual post-patch activation coordinate set to
    zero. Ablation cost is baseline patch_success minus ablated patch_success.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    Q = torch.tensor(Q_np, dtype=torch.float32)
    K_eff = int(Q_np.shape[1])
    d_in = int(Q_np.shape[0])
    directions = direction_keys(direction)

    protected_policy, protected_mean, protected_std = load_learned_policy(protected_spec)
    collapsed_policy, collapsed_mean, collapsed_std = load_learned_policy(collapsed_spec)
    validate_subspace_compatible(
        Q_np=Q_np,
        layer=layer,
        protected_policy=protected_policy,
        protected_mean=protected_mean,
        protected_spec=protected_spec,
        collapsed_policy=collapsed_policy,
        collapsed_mean=collapsed_mean,
        collapsed_spec=collapsed_spec,
    )

    rows: list[dict[str, Any]] = []
    with BridgeClient() as client:
        for offset in range(seeds):
            seed = seed_start + offset
            prefix = f"phase6-zero-ablation-{seed}"
            print(f"phase6 v3.3 axis-N: seed {seed} ({offset + 1}/{seeds})", flush=True)

            cache_A = run_subspace_recording_rollout(
                client,
                policy=protected_policy,
                obs_mean=protected_mean,
                obs_std=protected_std,
                seed=seed,
                horizon=horizon,
                layer=layer,
                env_id=f"{prefix}-A",
                Q=Q,
            )
            cache_B = run_subspace_recording_rollout(
                client,
                policy=collapsed_policy,
                obs_mean=collapsed_mean,
                obs_std=collapsed_std,
                seed=seed,
                horizon=horizon,
                layer=layer,
                env_id=f"{prefix}-B",
                Q=Q,
            )

            baseline_ps: dict[str, float] = {}
            if "protected_to_collapsed" in directions:
                cache_C = run_subspace_injected_rollout(
                    client,
                    policy=collapsed_policy,
                    obs_mean=collapsed_mean,
                    obs_std=collapsed_std,
                    seed=seed,
                    horizon=horizon,
                    layer=layer,
                    env_id=f"{prefix}-C-baseline",
                    Q=Q,
                    target_coords=cache_A.projections,  # type: ignore[arg-type]
                )
                baseline_ps["protected_to_collapsed"] = safe_patch_success(
                    cache_A.old_basin_pref,
                    cache_B.old_basin_pref,
                    cache_C.old_basin_pref,
                    direction="protected_to_collapsed",
                )
            if "collapsed_to_protected" in directions:
                cache_D = run_subspace_injected_rollout(
                    client,
                    policy=protected_policy,
                    obs_mean=protected_mean,
                    obs_std=protected_std,
                    seed=seed,
                    horizon=horizon,
                    layer=layer,
                    env_id=f"{prefix}-D-baseline",
                    Q=Q,
                    target_coords=cache_B.projections,  # type: ignore[arg-type]
                )
                baseline_ps["collapsed_to_protected"] = safe_patch_success(
                    cache_A.old_basin_pref,
                    cache_B.old_basin_pref,
                    cache_D.old_basin_pref,
                    direction="collapsed_to_protected",
                )

            for neuron_idx in range(d_in):
                if neuron_idx % 32 == 0:
                    print(f"  zero-ablate neuron {neuron_idx}/{d_in}", flush=True)

                if "protected_to_collapsed" in directions:
                    cache_C_j = run_subspace_injected_rollout(
                        client,
                        policy=collapsed_policy,
                        obs_mean=collapsed_mean,
                        obs_std=collapsed_std,
                        seed=seed,
                        horizon=horizon,
                        layer=layer,
                        env_id=f"{prefix}-C-neuron-{neuron_idx}",
                        Q=Q,
                        target_coords=cache_A.projections,  # type: ignore[arg-type]
                        zero_ablate_neuron=neuron_idx,
                    )
                    ablated_ps = safe_patch_success(
                        cache_A.old_basin_pref,
                        cache_B.old_basin_pref,
                        cache_C_j.old_basin_pref,
                        direction="protected_to_collapsed",
                    )
                    rows.append({
                        "seed": seed,
                        "direction": "protected_to_collapsed",
                        "neuron_idx": neuron_idx,
                        "baseline_patch_success": baseline_ps["protected_to_collapsed"],
                        "ablated_patch_success": ablated_ps,
                        "ablation_cost": baseline_ps["protected_to_collapsed"] - ablated_ps,
                        "protected_old_basin_pref": cache_A.old_basin_pref,
                        "collapsed_old_basin_pref": cache_B.old_basin_pref,
                        "ablated_old_basin_pref": cache_C_j.old_basin_pref,
                    })

                if "collapsed_to_protected" in directions:
                    cache_D_j = run_subspace_injected_rollout(
                        client,
                        policy=protected_policy,
                        obs_mean=protected_mean,
                        obs_std=protected_std,
                        seed=seed,
                        horizon=horizon,
                        layer=layer,
                        env_id=f"{prefix}-D-neuron-{neuron_idx}",
                        Q=Q,
                        target_coords=cache_B.projections,  # type: ignore[arg-type]
                        zero_ablate_neuron=neuron_idx,
                    )
                    ablated_ps = safe_patch_success(
                        cache_A.old_basin_pref,
                        cache_B.old_basin_pref,
                        cache_D_j.old_basin_pref,
                        direction="collapsed_to_protected",
                    )
                    rows.append({
                        "seed": seed,
                        "direction": "collapsed_to_protected",
                        "neuron_idx": neuron_idx,
                        "baseline_patch_success": baseline_ps["collapsed_to_protected"],
                        "ablated_patch_success": ablated_ps,
                        "ablation_cost": baseline_ps["collapsed_to_protected"] - ablated_ps,
                        "protected_old_basin_pref": cache_A.old_basin_pref,
                        "collapsed_old_basin_pref": cache_B.old_basin_pref,
                        "ablated_old_basin_pref": cache_D_j.old_basin_pref,
                    })
        client.request({"cmd": "close"})

    write_csv(out_dir / "ablation-table.csv", rows)

    aggregate_rows: list[dict[str, Any]] = []
    critical_sets: dict[str, dict[int, set[int]]] = {}
    critical_top32_paths: dict[str, str] = {}
    for dir_key in directions:
        dir_agg: list[dict[str, Any]] = []
        for neuron_idx in range(d_in):
            neuron_rows = [
                row for row in rows
                if row["direction"] == dir_key and int(row["neuron_idx"]) == neuron_idx
            ]
            costs = [float(row["ablation_cost"]) for row in neuron_rows]
            baselines = [float(row["baseline_patch_success"]) for row in neuron_rows]
            ablated = [float(row["ablated_patch_success"]) for row in neuron_rows]
            dir_agg.append({
                "direction": dir_key,
                "neuron_idx": neuron_idx,
                "mean_ablation_cost": mean_finite(costs),
                "median_ablation_cost": median_finite(costs),
                "q25_ablation_cost": percentile_finite(costs, 25),
                "q75_ablation_cost": percentile_finite(costs, 75),
                "mean_baseline_patch_success": mean_finite(baselines),
                "mean_ablated_patch_success": mean_finite(ablated),
                "n": len(neuron_rows),
            })
        dir_agg.sort(key=lambda row: float(row["mean_ablation_cost"]), reverse=True)
        aggregate_rows.extend(dir_agg)

        critical_sets[dir_key] = {}
        for k in (1, 4, 8, 16, 32):
            top = {int(row["neuron_idx"]) for row in dir_agg[:min(k, len(dir_agg))]}
            critical_sets[dir_key][k] = top
        top32_rows = [
            {"rank": rank + 1, **row}
            for rank, row in enumerate(dir_agg[:32])
        ]
        suffix = "pc" if dir_key == "protected_to_collapsed" else "cp"
        path = out_dir / f"critical-top-32-{suffix}.csv"
        write_csv(path, top32_rows)
        critical_top32_paths[dir_key] = str(path.relative_to(REPO_ROOT))

    write_csv(out_dir / "ablation-aggregate.csv", aggregate_rows)

    l2_top32 = set(int(idx) for idx in compute_neuron_ranking(Q_np)[:32])
    k_rows: list[dict[str, Any]] = []
    for dir_key in directions:
        dir_rows = [row for row in aggregate_rows if row["direction"] == dir_key]
        total_positive_cost = sum(max(float(row["mean_ablation_cost"]), 0.0) for row in dir_rows)
        for k in (1, 4, 8, 16, 32):
            top = critical_sets[dir_key][k]
            top_cost = sum(
                max(float(row["mean_ablation_cost"]), 0.0)
                for row in dir_rows
                if int(row["neuron_idx"]) in top
            )
            k_rows.append({
                "direction": dir_key,
                "k": k,
                "aggregate_positive_cost_captured": top_cost,
                "fraction_positive_cost_captured": (
                    float("nan") if total_positive_cost <= 1e-12 else top_cost / total_positive_cost
                ),
                "jaccard_axis_n_vs_axis_m_l2_rank": jaccard(top, l2_top32) if k == 32 else "",
            })
    if len(directions) == 2:
        for k in (1, 4, 8, 16, 32):
            k_rows.append({
                "direction": "protected_to_collapsed_vs_collapsed_to_protected",
                "k": k,
                "aggregate_positive_cost_captured": "",
                "fraction_positive_cost_captured": "",
                "jaccard_axis_n_vs_axis_m_l2_rank": jaccard(
                    critical_sets["protected_to_collapsed"][k],
                    critical_sets["collapsed_to_protected"][k],
                ),
            })
    write_csv(out_dir / "critical-set-summary.csv", k_rows)

    max_by_direction: dict[str, float] = {}
    for dir_key in directions:
        dir_rows = [row for row in aggregate_rows if row["direction"] == dir_key]
        max_by_direction[dir_key] = max(float(row["mean_ablation_cost"]) for row in dir_rows)

    pc_top32 = critical_sets.get("protected_to_collapsed", {}).get(32, set())
    cp_top32 = critical_sets.get("collapsed_to_protected", {}).get(32, set())
    jaccard_summary = {
        "axis_m_l2_top32": sorted(l2_top32),
        "axis_n_top32_protected_to_collapsed": sorted(pc_top32),
        "axis_n_top32_collapsed_to_protected": sorted(cp_top32),
        "jaccard_pc_vs_axis_m_l2": jaccard(pc_top32, l2_top32) if pc_top32 else None,
        "jaccard_cp_vs_axis_m_l2": jaccard(cp_top32, l2_top32) if cp_top32 else None,
        "jaccard_pc_vs_cp": jaccard(pc_top32, cp_top32) if pc_top32 and cp_top32 else None,
    }
    (out_dir / "jaccard-comparison.json").write_text(
        json.dumps(jaccard_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    negative_rows = [
        row for row in aggregate_rows
        if math.isfinite(float(row["mean_ablation_cost"])) and float(row["mean_ablation_cost"]) < 0
    ]
    summary = {
        "axis": "N",
        "seed_start": int(seed_start),
        "seeds": int(seeds),
        "horizon": int(horizon),
        "layer": layer,
        "k_eff": K_eff,
        "d_in": d_in,
        "directions": directions,
        "max_mean_ablation_cost_by_direction": max_by_direction,
        "smoke_gate_threshold": 0.05,
        "smoke_gate_pass": max_by_direction.get("protected_to_collapsed", float("-inf")) >= 0.05,
        "aa1_substantial_threshold": 0.3,
        "aa1_falsifier_threshold": 0.1,
        "critical_top32_paths": critical_top32_paths,
        "jaccard": jaccard_summary,
        "negative_mean_ablation_cost_count": len(negative_rows),
        "manifest_extra": manifest_extra,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if directions == ["protected_to_collapsed"]:
        (out_dir / "smoke-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "phase": "phase6-v3.3-axis-n-zero-ablation",
        "protected_policy_id": protected_spec.policy_id,
        "protected_label": protected_spec.label,
        "collapsed_policy_id": collapsed_spec.policy_id,
        "collapsed_label": collapsed_spec.label,
        **summary,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"phase6 v3.3 axis-N: wrote results to {out_dir.relative_to(REPO_ROOT)}", flush=True)
    for dir_key, max_cost in max_by_direction.items():
        best = next(row for row in aggregate_rows if row["direction"] == dir_key)
        print(
            f"  {dir_key}: max_mean_ablation_cost={max_cost:+.3f} "
            f"(neuron {best['neuron_idx']})",
            flush=True,
        )


def axis_n_zero_ablation(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    Q_full, basis_metadata = load_or_build_cliff_pca_basis(
        out_root=PHASE6_V31_OUT,
        seed_start=args.basis_seed_start,
        seeds=args.basis_seeds,
        horizon=args.basis_horizon,
        num_components=5,
    )
    run_zero_ablation_battery(
        Q_np=Q_full,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        layer=args.layer,
        out_dir=out_dir,
        direction=args.direction,
        manifest_extra={
            "basis": "cliff_pair_pca_pc1_to_5",
            "basis_metadata": basis_metadata,
            "basis_seed_start": int(args.basis_seed_start),
            "basis_seeds": int(args.basis_seeds),
            "basis_horizon": int(args.basis_horizon),
        },
    )


# ============================================================
# v3.4 Axis P/Q — substrate ablation and Jaccard bootstrap
# ============================================================


def load_neuron_indices_csv(path: Path, *, limit: int | None = None) -> list[int]:
    rows = read_csv_rows(path)
    if not rows or "neuron_idx" not in rows[0]:
        raise ValueError(f"{path} must contain a neuron_idx column")
    indices = [int(row["neuron_idx"]) for row in rows]
    return indices[:limit] if limit is not None else indices


def infer_mask_direction(path: Path, explicit: str | None = None) -> str | None:
    if explicit:
        return {
            "P_to_C": "protected_to_collapsed",
            "C_to_P": "collapsed_to_protected",
            "unknown": None,
        }[explicit]
    name = path.name.lower()
    if "pc" in name:
        return "protected_to_collapsed"
    if "cp" in name:
        return "collapsed_to_protected"
    return None


def run_substrate_ablation_battery(
    *,
    Q_np: np.ndarray,
    neuron_indices: list[int],
    seed_start: int,
    seeds: int,
    horizon: int,
    layer: str,
    out_dir: Path,
    mask_source: Path,
    mask_direction: str | None,
    manifest_extra: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Q = torch.tensor(Q_np, dtype=torch.float32)
    protected_policy, protected_mean, protected_std = load_learned_policy(CLIFF_PROTECTED)
    collapsed_policy, collapsed_mean, collapsed_std = load_learned_policy(CLIFF_COLLAPSED)
    validate_subspace_compatible(
        Q_np=Q_np,
        layer=layer,
        protected_policy=protected_policy,
        protected_mean=protected_mean,
        protected_spec=CLIFF_PROTECTED,
        collapsed_policy=collapsed_policy,
        collapsed_mean=collapsed_mean,
        collapsed_spec=CLIFF_COLLAPSED,
    )

    rows: list[dict[str, Any]] = []
    with BridgeClient() as client:
        for offset in range(seeds):
            seed = seed_start + offset
            prefix = f"phase6-substrate-ablation-{seed}"
            print(f"phase6 v3.4 axis-P: seed {seed} ({offset + 1}/{seeds})", flush=True)

            cache_A = run_subspace_recording_rollout(
                client,
                policy=protected_policy,
                obs_mean=protected_mean,
                obs_std=protected_std,
                seed=seed,
                horizon=horizon,
                layer=layer,
                env_id=f"{prefix}-A",
                Q=Q,
            )
            cache_B = run_subspace_recording_rollout(
                client,
                policy=collapsed_policy,
                obs_mean=collapsed_mean,
                obs_std=collapsed_std,
                seed=seed,
                horizon=horizon,
                layer=layer,
                env_id=f"{prefix}-B",
                Q=Q,
            )

            for direction in ("protected_to_collapsed", "collapsed_to_protected"):
                if direction == "protected_to_collapsed":
                    policy = collapsed_policy
                    obs_mean = collapsed_mean
                    obs_std = collapsed_std
                    target_coords = cache_A.projections
                else:
                    policy = protected_policy
                    obs_mean = protected_mean
                    obs_std = protected_std
                    target_coords = cache_B.projections

                baseline = run_subspace_injected_rollout(
                    client,
                    policy=policy,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    seed=seed,
                    horizon=horizon,
                    layer=layer,
                    env_id=f"{prefix}-{direction}-baseline",
                    Q=Q,
                    target_coords=target_coords,  # type: ignore[arg-type]
                )
                ablated = run_subspace_injected_rollout(
                    client,
                    policy=policy,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    seed=seed,
                    horizon=horizon,
                    layer=layer,
                    env_id=f"{prefix}-{direction}-ablated",
                    Q=Q,
                    target_coords=target_coords,  # type: ignore[arg-type]
                    zero_ablate_neuron_set=neuron_indices,
                )
                baseline_ps = safe_patch_success(
                    cache_A.old_basin_pref,
                    cache_B.old_basin_pref,
                    baseline.old_basin_pref,
                    direction=direction,
                )
                ablated_ps = safe_patch_success(
                    cache_A.old_basin_pref,
                    cache_B.old_basin_pref,
                    ablated.old_basin_pref,
                    direction=direction,
                )
                rows.append({
                    "seed": seed,
                    "mask_source": str(mask_source.relative_to(REPO_ROOT)) if mask_source.is_relative_to(REPO_ROOT) else str(mask_source),
                    "mask_direction": mask_direction or "unknown",
                    "direction": direction,
                    "baseline_patch_success": baseline_ps,
                    "ablated_patch_success": ablated_ps,
                    "ablation_drop": baseline_ps - ablated_ps,
                    "protected_old_basin_pref": cache_A.old_basin_pref,
                    "collapsed_old_basin_pref": cache_B.old_basin_pref,
                    "baseline_old_basin_pref": baseline.old_basin_pref,
                    "ablated_old_basin_pref": ablated.old_basin_pref,
                    "mask_size": len(neuron_indices),
                    "mask_neuron_indices": ";".join(str(idx) for idx in neuron_indices),
                })
        client.request({"cmd": "close"})

    write_csv(out_dir / "substrate-ablation.csv", rows)
    aggregate_rows: list[dict[str, Any]] = []
    for direction in ("protected_to_collapsed", "collapsed_to_protected"):
        dir_rows = [row for row in rows if row["direction"] == direction]
        baseline_values = [float(row["baseline_patch_success"]) for row in dir_rows]
        ablated_values = [float(row["ablated_patch_success"]) for row in dir_rows]
        drop_values = [float(row["ablation_drop"]) for row in dir_rows]
        aggregate_rows.append({
            "mask_direction": mask_direction or "unknown",
            "direction": direction,
            "mean_baseline_patch_success": mean_finite(baseline_values),
            "median_baseline_patch_success": median_finite(baseline_values),
            "mean_ablated_patch_success": mean_finite(ablated_values),
            "median_ablated_patch_success": median_finite(ablated_values),
            "mean_ablation_drop": mean_finite(drop_values),
            "median_ablation_drop": median_finite(drop_values),
            "q25_ablation_drop": percentile_finite(drop_values, 25),
            "q75_ablation_drop": percentile_finite(drop_values, 75),
            "n": len(dir_rows),
        })
    write_csv(out_dir / "substrate-ablation-aggregate.csv", aggregate_rows)

    same_drop = None
    cross_drop = None
    if mask_direction:
        same = next(row for row in aggregate_rows if row["direction"] == mask_direction)
        cross = next(row for row in aggregate_rows if row["direction"] != mask_direction)
        same_drop = float(same["median_ablation_drop"])
        cross_drop = float(cross["median_ablation_drop"])
    summary = {
        "axis": "P",
        "mask_source": str(mask_source.relative_to(REPO_ROOT)) if mask_source.is_relative_to(REPO_ROOT) else str(mask_source),
        "mask_direction": mask_direction,
        "mask_size": len(neuron_indices),
        "mask_neuron_indices": neuron_indices,
        "seed_start": int(seed_start),
        "seeds": int(seeds),
        "horizon": int(horizon),
        "layer": layer,
        "same_direction_median_drop": same_drop,
        "cross_direction_median_drop": cross_drop,
        "dissociation_median": None if same_drop is None or cross_drop is None else same_drop - cross_drop,
        "manifest_extra": manifest_extra,
    }
    (out_dir / "dissociation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"phase6 v3.4 axis-P: wrote results to {out_dir.relative_to(REPO_ROOT)}", flush=True)
    for row in aggregate_rows:
        print(
            f"  {row['direction']}: median_baseline={row['median_baseline_patch_success']:+.3f} "
            f"median_ablated={row['median_ablated_patch_success']:+.3f} "
            f"median_drop={row['median_ablation_drop']:+.3f}",
            flush=True,
        )
    if summary["dissociation_median"] is not None:
        print(f"  dissociation_median={summary['dissociation_median']:+.3f}", flush=True)


def axis_p_substrate_ablation(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    mask_source = Path(args.neuron_mask_source)
    if not mask_source.is_absolute():
        mask_source = REPO_ROOT / mask_source
    neuron_indices = load_neuron_indices_csv(mask_source, limit=args.top_k)
    mask_direction = infer_mask_direction(mask_source, args.mask_direction)
    Q_full, basis_metadata = load_or_build_cliff_pca_basis(
        out_root=PHASE6_V31_OUT,
        seed_start=args.basis_seed_start,
        seeds=args.basis_seeds,
        horizon=args.basis_horizon,
        num_components=5,
    )
    run_substrate_ablation_battery(
        Q_np=Q_full,
        neuron_indices=neuron_indices,
        seed_start=args.seed_start,
        seeds=args.seeds,
        horizon=args.horizon,
        layer=args.layer,
        out_dir=out_dir,
        mask_source=mask_source,
        mask_direction=mask_direction,
        manifest_extra={
            "basis": "cliff_pair_pca_pc1_to_5",
            "basis_metadata": basis_metadata,
            "basis_seed_start": int(args.basis_seed_start),
            "basis_seeds": int(args.basis_seeds),
            "basis_horizon": int(args.basis_horizon),
        },
    )


def axis_q_jaccard_bootstrap(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.ablation_table)
    if not table_path.is_absolute():
        table_path = REPO_ROOT / table_path
    l2_path = Path(args.l2_rank_source)
    if not l2_path.is_absolute():
        l2_path = REPO_ROOT / l2_path

    rows = read_csv_rows(table_path)
    l2_top32 = set(load_neuron_indices_csv(l2_path, limit=32))
    seeds = sorted({int(row["seed"]) for row in rows})
    directions = ["protected_to_collapsed", "collapsed_to_protected"]
    costs: dict[tuple[str, int, int], float] = {}
    neurons = sorted({int(row["neuron_idx"]) for row in rows})
    for row in rows:
        costs[(row["direction"], int(row["seed"]), int(row["neuron_idx"]))] = float(row["ablation_cost"])

    def top32_for(direction: str, sample: np.ndarray) -> set[int]:
        means = []
        for neuron_idx in neurons:
            values = [costs[(direction, int(seed), neuron_idx)] for seed in sample]
            means.append((float(np.mean(values)), neuron_idx))
        means.sort(reverse=True)
        return {neuron_idx for _mean, neuron_idx in means[:32]}

    rng = np.random.default_rng(int(args.bootstrap_seed))
    boot_rows: list[dict[str, Any]] = []
    for r in range(int(args.resamples)):
        sample = rng.choice(seeds, size=len(seeds), replace=True)
        pc = top32_for("protected_to_collapsed", sample)
        cp = top32_for("collapsed_to_protected", sample)
        boot_rows.append({
            "resample": r,
            "sample_seeds": ";".join(str(int(seed)) for seed in sample.tolist()),
            "jaccard_pc_vs_cp": jaccard(pc, cp),
            "jaccard_pc_vs_l2": jaccard(pc, l2_top32),
            "jaccard_cp_vs_l2": jaccard(cp, l2_top32),
        })
    write_csv(out_dir / "bootstrap-jaccard.csv", boot_rows)

    def summarize_key(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in boot_rows]
        return {
            "p2_5": percentile_finite(values, 2.5),
            "p25": percentile_finite(values, 25),
            "median": percentile_finite(values, 50),
            "p75": percentile_finite(values, 75),
            "p97_5": percentile_finite(values, 97.5),
        }

    observed_pc = top32_for("protected_to_collapsed", np.asarray(seeds))
    observed_cp = top32_for("collapsed_to_protected", np.asarray(seeds))
    summary = {
        "axis": "Q",
        "ablation_table": str(table_path.relative_to(REPO_ROOT)) if table_path.is_relative_to(REPO_ROOT) else str(table_path),
        "l2_rank_source": str(l2_path.relative_to(REPO_ROOT)) if l2_path.is_relative_to(REPO_ROOT) else str(l2_path),
        "resamples": int(args.resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "seed_count": len(seeds),
        "observed": {
            "jaccard_pc_vs_cp": jaccard(observed_pc, observed_cp),
            "jaccard_pc_vs_l2": jaccard(observed_pc, l2_top32),
            "jaccard_cp_vs_l2": jaccard(observed_cp, l2_top32),
        },
        "ci": {
            "jaccard_pc_vs_cp": summarize_key("jaccard_pc_vs_cp"),
            "jaccard_pc_vs_l2": summarize_key("jaccard_pc_vs_l2"),
            "jaccard_cp_vs_l2": summarize_key("jaccard_cp_vs_l2"),
        },
    }
    (out_dir / "bootstrap-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"phase6 v3.4 axis-Q: wrote results to {out_dir.relative_to(REPO_ROOT)}", flush=True)
    for key, vals in summary["ci"].items():
        print(
            f"  {key}: median={vals['median']:.3f} "
            f"95% CI [{vals['p2_5']:.3f}, {vals['p97_5']:.3f}]",
            flush=True,
        )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 v2 - direction-based mechanistic probing")
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser(
        "axis-d-sae",
        help="Train top-k SAE on cliff-pair net.7 activations and label features by basin correlation",
    )
    train.add_argument("--out", default=str(PHASE6_V2_OUT / "axis-d-sae"))
    train.add_argument("--seed-start", type=int, default=10000)
    train.add_argument("--seeds", type=int, default=64)
    train.add_argument("--horizon", type=int, default=200)
    train.add_argument("--n-features", type=int, default=1024)
    train.add_argument("--k", type=int, default=32)
    train.add_argument("--steps", type=int, default=10000)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--sae-seed", type=int, default=0)

    patch = sub.add_parser(
        "axis-e-patch",
        help="Direction-based patching using the top SAE basin feature",
    )
    patch.add_argument("--out", default=str(PHASE6_V2_OUT / "axis-e-patch"))
    patch.add_argument("--sae-dir", default=str(PHASE6_V2_OUT / "axis-d-sae"))
    patch.add_argument("--feature-idx", type=int, default=None,
                       help="override feature index; default = read top from axis-d-top10-basin-features.csv")
    patch.add_argument("--seed-start", type=int, default=10000)
    patch.add_argument("--seeds", type=int, default=64)
    patch.add_argument("--horizon", type=int, default=200)
    patch.add_argument("--layer", default="net.7")

    multif = sub.add_parser(
        "axis-f-multifeature",
        help="Subspace patching using top-K basin-correlated SAE features (orthogonalized)",
    )
    multif.add_argument("--out", default=str(PHASE6_V2_OUT / "axis-f-multifeature"))
    multif.add_argument("--sae-dir", default=str(PHASE6_V2_OUT / "axis-d-sae"))
    multif.add_argument("--num-features", type=int, default=10,
                        help="number of top-correlated SAE features to span")
    multif.add_argument("--seed-start", type=int, default=10000)
    multif.add_argument("--seeds", type=int, default=64)
    multif.add_argument("--horizon", type=int, default=200)
    multif.add_argument("--layer", default="net.7")

    meandiff = sub.add_parser(
        "axis-g-mean-diff",
        help="K=1 direction patch along empirical between-policy mean-diff at net.7",
    )
    meandiff.add_argument("--out", default=str(PHASE6_V2_OUT / "axis-g-mean-diff"))
    meandiff.add_argument("--seed-start", type=int, default=10000)
    meandiff.add_argument("--seeds", type=int, default=64)
    meandiff.add_argument("--horizon", type=int, default=200)
    meandiff.add_argument("--layer", default="net.7")

    pca = sub.add_parser(
        "axis-h-pca",
        help="Subspace patching using top-K principal directions of per-step cliff-pair diffs",
    )
    pca.add_argument("--out", default=str(PHASE6_V2_OUT / "axis-h-pca"))
    pca.add_argument("--num-components", type=int, default=10,
                     help="number of top PCA components of the per-step diff matrix to span")
    pca.add_argument("--seed-start", type=int, default=10000)
    pca.add_argument("--seeds", type=int, default=64)
    pca.add_argument("--horizon", type=int, default=200)
    pca.add_argument("--layer", default="net.7")

    pci = sub.add_parser(
        "axis-i-pc-mech",
        help="Phase 6 v3.1 Axis I: patch using PCA PCs 2-5 only, skipping PC1",
    )
    pci.add_argument("--out", default=str(PHASE6_V31_OUT / "axis-i-pc-mech"))
    pci.add_argument("--seed-start", type=int, default=10000)
    pci.add_argument("--seeds", type=int, default=64)
    pci.add_argument("--horizon", type=int, default=200)
    pci.add_argument("--layer", default="net.7")

    gen = sub.add_parser(
        "axis-j-generalization",
        help="Phase 6 v3.1 Axis J: apply cliff-pair PCA basis to held-out policy pairs",
    )
    gen.add_argument("--out", default=str(PHASE6_V31_OUT / "axis-j-generalization"))
    gen.add_argument("--pair", choices=sorted(V31_GENERALIZATION_PAIRS), required=True)
    gen.add_argument("--seed-start", type=int, default=10000)
    gen.add_argument("--seeds", type=int, default=64)
    gen.add_argument("--horizon", type=int, default=200)
    gen.add_argument("--layer", default="net.7")

    decomp = sub.add_parser(
        "axis-k-decompose",
        help="Phase 6 v3.1 Axis K: decompose cliff-pair PCA directions by neuron concentration",
    )
    decomp.add_argument("--out", default=str(PHASE6_V31_OUT / "axis-k-decompose"))
    decomp.add_argument("--seed-start", type=int, default=10000)
    decomp.add_argument("--seeds", type=int, default=64)
    decomp.add_argument("--horizon", type=int, default=200)

    boot = sub.add_parser(
        "axis-l-bootstrap",
        help="Phase 6 v3.1 Axis L: bootstrap the directional-asymmetry median gap",
    )
    boot.add_argument("--out", default=str(PHASE6_V31_OUT / "axis-l-bootstrap"))
    boot.add_argument("--k", type=int, nargs="+", default=[3, 5], choices=[3, 5])
    boot.add_argument("--resamples", type=int, default=1000)
    boot.add_argument("--bootstrap-seed", type=int, default=0)

    nmed = sub.add_parser(
        "axis-m-neuron-mediation",
        help="Phase 6 v3.2 Axis M: top-k neuron-restricted projection of the v3 PCA patch",
    )
    nmed.add_argument("--out", default=str(PHASE6_V32_OUT))
    nmed.add_argument("--top-k", type=int, required=True,
                      help="number of top-ranked neurons (by aggregate L2 across PCs 1-5) to include in the patch mask")
    nmed.add_argument("--neuron-mask-source", default=None,
                      help="optional CSV with neuron_idx column; first --top-k ids replace the aggregate-L2 ranking")
    nmed.add_argument("--pair", default=None, choices=["cliff", "J1", "J2"],
                      help="policy pair to patch; default is the cliff pair (L-Mixed-M-lambda=0.95 vs lambda=0.97)")
    nmed.add_argument("--seed-start", type=int, default=10000)
    nmed.add_argument("--seeds", type=int, default=64)
    nmed.add_argument("--horizon", type=int, default=200)
    nmed.add_argument("--basis-seed-start", type=int, default=10000,
                      help="seed start for the cached cliff-pair PCA basis; default is canonical v3.1 basis")
    nmed.add_argument("--basis-seeds", type=int, default=64,
                      help="seed count for the cached cliff-pair PCA basis; keep at 64 for smoke/full runs")
    nmed.add_argument("--basis-horizon", type=int, default=200,
                      help="horizon for the cached cliff-pair PCA basis")
    nmed.add_argument("--layer", default="net.7")

    ablate = sub.add_parser(
        "axis-n-zero-ablation",
        help="Phase 6 v3.3 Axis N: per-neuron zero-ablation attribution during the v3 PCA patch",
    )
    ablate.add_argument("--out", default=str(PHASE6_V33_OUT / "smoke"))
    ablate.add_argument("--direction", default="P_to_C", choices=["P_to_C", "C_to_P", "both"],
                        help="patch direction to test; smoke uses P_to_C, full battery uses both")
    ablate.add_argument("--seed-start", type=int, default=10000)
    ablate.add_argument("--seeds", type=int, default=4)
    ablate.add_argument("--horizon", type=int, default=200)
    ablate.add_argument("--basis-seed-start", type=int, default=10000,
                        help="seed start for the cached cliff-pair PCA basis; default is canonical v3.1 basis")
    ablate.add_argument("--basis-seeds", type=int, default=64,
                        help="seed count for the cached cliff-pair PCA basis; keep at 64 for smoke/full runs")
    ablate.add_argument("--basis-horizon", type=int, default=200,
                        help="horizon for the cached cliff-pair PCA basis")
    ablate.add_argument("--layer", default="net.7")

    substrate = sub.add_parser(
        "axis-p-substrate-ablation",
        help="Phase 6 v3.4 Axis P: set-level ablation of v3.3 critical-neuron substrates",
    )
    substrate.add_argument("--out", default=str(PHASE6_V34_OUT / "axis-p-substrate-ablation"))
    substrate.add_argument("--neuron-mask-source", required=True,
                           help="CSV containing a neuron_idx column, typically v3.3 critical-top-32-pc/cp.csv")
    substrate.add_argument("--mask-direction", default=None, choices=["P_to_C", "C_to_P", "unknown"],
                           help="optional explicit same-direction label for the mask; inferred from filename by default")
    substrate.add_argument("--top-k", type=int, default=32)
    substrate.add_argument("--seed-start", type=int, default=10000)
    substrate.add_argument("--seeds", type=int, default=16)
    substrate.add_argument("--horizon", type=int, default=200)
    substrate.add_argument("--basis-seed-start", type=int, default=10000)
    substrate.add_argument("--basis-seeds", type=int, default=64)
    substrate.add_argument("--basis-horizon", type=int, default=200)
    substrate.add_argument("--layer", default="net.7")

    jboot = sub.add_parser(
        "axis-q-jaccard-bootstrap",
        help="Phase 6 v3.4 Axis Q: bootstrap Jaccard overlap of v3.3 ablation rankings",
    )
    jboot.add_argument("--out", default=str(PHASE6_V34_OUT / "axis-q-bootstrap"))
    jboot.add_argument("--ablation-table", required=True)
    jboot.add_argument("--l2-rank-source", required=True)
    jboot.add_argument("--resamples", type=int, default=1000)
    jboot.add_argument("--bootstrap-seed", type=int, default=0)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    return args


def main() -> None:
    args = parse_args()
    if args.command == "axis-d-sae":
        axis_d_train_sae(args)
    elif args.command == "axis-e-patch":
        axis_e_direction_patch(args)
    elif args.command == "axis-f-multifeature":
        axis_f_multifeature_patch(args)
    elif args.command == "axis-g-mean-diff":
        axis_g_mean_diff_patch(args)
    elif args.command == "axis-h-pca":
        axis_h_pca_patch(args)
    elif args.command == "axis-i-pc-mech":
        axis_i_pc_mech(args)
    elif args.command == "axis-j-generalization":
        axis_j_generalization(args)
    elif args.command == "axis-k-decompose":
        axis_k_decompose(args)
    elif args.command == "axis-l-bootstrap":
        axis_l_bootstrap(args)
    elif args.command == "axis-m-neuron-mediation":
        axis_m_neuron_mediation(args)
    elif args.command == "axis-n-zero-ablation":
        axis_n_zero_ablation(args)
    elif args.command == "axis-p-substrate-ablation":
        axis_p_substrate_ablation(args)
    elif args.command == "axis-q-jaccard-bootstrap":
        axis_q_jaccard_bootstrap(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
