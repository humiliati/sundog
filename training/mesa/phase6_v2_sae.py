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
            f"(Δ={cmp['v2_minus_v1_median']:+.3f})",
            flush=True,
        )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 v2 — direction-based mechanistic probing")
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
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
