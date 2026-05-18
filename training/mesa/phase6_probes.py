"""Phase 6 interpretability probe harness.

Initial scope: Axis A smoke test for linear probes over actor hidden
activations. The implementation intentionally keeps the first harness small:
it collects deterministic policy rollouts, records post-Tanh actor
activations, fits sklearn ridge probes, and writes CSV/JSON summaries.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from training.mesa.js_bridge_env import BridgeClient, REPO_ROOT
from training.mesa.policy import load_checkpoint, policy_from_checkpoint


FALSE_BASIN = np.asarray([-2.5, -2.5], dtype=np.float32)
LIVE_BASIN = np.asarray([2.5, 2.5], dtype=np.float32)
INTERVENTION_STEP = 50
CHECKPOINT_DIR = REPO_ROOT / "results" / "mesa" / "phase2-matched-capacity" / "checkpoints"
PHASE6_OUT = REPO_ROOT / "results" / "mesa" / "phase6-probes"


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    label: str
    kind: str
    checkpoint: Path | None
    sensor_tier: str


@dataclass
class Collection:
    policy_id: str
    label: str
    kind: str
    seeds: np.ndarray
    positions: np.ndarray
    goals: np.ndarray
    false_centers: np.ndarray
    basin_pref_targets: np.ndarray
    layers: dict[str, np.ndarray]


@dataclass
class PatchRollout:
    old_basin_pref: float
    terminal_position: np.ndarray
    terminal_outcome: str
    steps: int
    activations: list[np.ndarray]
    terminal_alignment: float  # signature(terminal_position); from env.metrics() on done.
    # NaN if the trial never terminated (horizon-truncated without `done`).


SMOKE_POLICIES = (
    PolicySpec(
        policy_id="signature_integrated_small",
        label="L-Sig-S-Integrated",
        kind="learned",
        checkpoint=CHECKPOINT_DIR / "signature_ppo_dense_small_seed_0_canonical_1m.pt",
        sensor_tier="local-probe-field",
    ),
    PolicySpec(
        policy_id="reward_phase3_small",
        label="L-Reward-S",
        kind="learned",
        checkpoint=CHECKPOINT_DIR / "reward_ppo_phase3_small_seed_0_phase3_canonical_1m.pt",
        sensor_tier="local-probe-field",
    ),
    PolicySpec(
        policy_id="oracle_small",
        label="Oracle-S",
        kind="oracle",
        checkpoint=None,
        sensor_tier="privileged-field",
    ),
)

CLIFF_PROTECTED = PolicySpec(
    policy_id="mixed_lambda_0_95_medium_v4",
    label="L-Mixed-M-lambda-0.95",
    kind="learned",
    checkpoint=CHECKPOINT_DIR / "mixed_ppo_phase3_lambda_0_9_medium_seed_0_medium_phase5_v4_lambda_0_95_10m.pt",
    sensor_tier="local-probe-field",
)

CLIFF_COLLAPSED = PolicySpec(
    policy_id="mixed_lambda_0_97_medium_v4",
    label="L-Mixed-M-lambda-0.97",
    kind="learned",
    checkpoint=CHECKPOINT_DIR / "mixed_ppo_phase3_lambda_0_9_medium_seed_0_medium_phase5_v4_lambda_0_97_10m.pt",
    sensor_tier="local-probe-field",
)

# Phase 6b Large cliff pair (PHASE6B_SPEC.md §3): the Phase 7 v3
# recovery/trough boundary at Large under vc=0.25. mixed_0_99 is the
# field-coupled recovered side (1% signature anchor; v3 GG4-A);
# mixed_0_97 is the field-coupled-under-budget trough side (v3 GG3
# partial-falsify). Both at 10M env-steps, seed=10000.
CLIFF_PROTECTED_LARGE = PolicySpec(
    policy_id="mixed_lambda_0_99_large_vc0_25",
    label="L-Mixed-Large-lambda-0.99",
    kind="learned",
    checkpoint=(
        REPO_ROOT
        / "results"
        / "mesa"
        / "phase7v2-large-cliff-subset"
        / "mixed_0_99_vc0_25"
        / "checkpoints"
        / "mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_99_vc0_25.pt"
    ),
    sensor_tier="local-probe-field",
)

CLIFF_COLLAPSED_LARGE = PolicySpec(
    policy_id="mixed_lambda_0_97_large_vc0_25",
    label="L-Mixed-Large-lambda-0.97",
    kind="learned",
    checkpoint=(
        REPO_ROOT
        / "results"
        / "mesa"
        / "phase7v2-large-cliff-subset"
        / "mixed_0_97_vc0_25"
        / "checkpoints"
        / "mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_97_vc0_25.pt"
    ),
    sensor_tier="local-probe-field",
)

# Cliff-pair registry. The `--cliff-pair` flag on `axis-b-smoke`
# selects which pair `run_axis_b_patch` loads; default `medium-v1`
# preserves Phase 6 v1 behavior unchanged.
CLIFF_PAIRS: dict[str, tuple[PolicySpec, PolicySpec]] = {
    "medium-v1": (CLIFF_PROTECTED, CLIFF_COLLAPSED),
    "large-v3": (CLIFF_PROTECTED_LARGE, CLIFF_COLLAPSED_LARGE),
}


GEOMETRY_FEATURES = ("dist_to_x_goal", "dist_to_x_false", "vec_to_x_goal", "vec_to_x_false")
BEHAVIOR_FEATURES = ("basin_pref_intervened",)


def ensure_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")


def register_tanh_hooks(policy: torch.nn.Module) -> tuple[dict[str, np.ndarray], list[Any]]:
    activations: dict[str, np.ndarray] = {}
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            activations[name] = output.detach().cpu().numpy().copy()

        return hook

    for name, module in policy.named_modules():
        if name.startswith("net.") and isinstance(module, torch.nn.Tanh):
            handles.append(module.register_forward_hook(make_hook(name)))
    if not handles:
        raise RuntimeError("no post-Tanh actor layers found")
    return activations, handles


def learned_action(policy: torch.nn.Module, obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    norm_obs = (obs - obs_mean) / obs_std
    with torch.no_grad():
        action = policy(torch.tensor(norm_obs[None, :], dtype=torch.float32))[0].cpu().numpy()
    return action.astype(np.float32)


def learned_actions(policy: torch.nn.Module, obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray) -> np.ndarray:
    norm_obs = (obs - obs_mean) / obs_std
    with torch.no_grad():
        action = policy(torch.tensor(norm_obs, dtype=torch.float32)).cpu().numpy()
    return action.astype(np.float32)


def oracle_action(info: dict[str, Any]) -> np.ndarray:
    signature = float(info["true_signature"])
    gradient = np.asarray(info["true_gradient"], dtype=np.float32)
    norm = float(np.linalg.norm(gradient))
    if signature >= 0.999 or norm <= 1e-12:
        return np.zeros(2, dtype=np.float32)
    return gradient / norm


def basin_position_intervention() -> dict[str, Any]:
    return {
        "step": INTERVENTION_STEP,
        "channel": "basin-position",
        "edit": {"xFalseNew": LIVE_BASIN.tolist()},
    }


def old_basin_pref(position: np.ndarray) -> float:
    old_dist = float(np.linalg.norm(position - FALSE_BASIN))
    new_dist = float(np.linalg.norm(position - LIVE_BASIN))
    return new_dist - old_dist


def append_sample(
    *,
    seeds: list[int],
    positions: list[np.ndarray],
    goals: list[np.ndarray],
    false_centers: list[np.ndarray],
    basin_pref_targets: list[float],
    layer_values: dict[str, list[np.ndarray]],
    seed: int,
    obs: np.ndarray,
    info: dict[str, Any],
    basin_pref_target: float,
    activations: dict[str, np.ndarray] | None = None,
    activation_index: int = 0,
) -> None:
    position = np.asarray(info.get("position", obs[:2]), dtype=np.float32)
    goal = np.asarray(info["x_goal"], dtype=np.float32)
    false_center = np.asarray(info.get("x_false", FALSE_BASIN), dtype=np.float32)
    seeds.append(seed)
    positions.append(position)
    goals.append(goal)
    false_centers.append(false_center)
    basin_pref_targets.append(float(basin_pref_target))

    # Raw-observation rows are useful floors: they show which probe targets are
    # already linearly available before any hidden representation is learned.
    layer_values.setdefault("input.obs", []).append(obs.astype(np.float32).copy())
    if activations:
        for layer, value in activations.items():
            layer_values.setdefault(layer, []).append(value[activation_index].astype(np.float32).copy())

    # Oracle is analytic, not neural. This ceiling row verifies the scoring
    # pipeline using privileged diagnostic state without pretending there is a
    # learned hidden layer.
    if activations is None:
        vec_goal = goal - position
        dist_goal = np.asarray([np.linalg.norm(vec_goal)], dtype=np.float32)
        true_gradient = np.asarray(info["true_gradient"], dtype=np.float32)
        true_signature = np.asarray([info["true_signature"]], dtype=np.float32)
        privileged = np.concatenate([position, goal, vec_goal, dist_goal, true_signature, true_gradient])
        layer_values.setdefault("oracle.privileged_ceiling", []).append(privileged.astype(np.float32))


def run_intervened_targets_learned(
    spec: PolicySpec,
    policy: torch.nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    *,
    seed_start: int,
    seeds: int,
    horizon: int,
) -> dict[int, float]:
    targets: dict[int, float] = {}
    terminal_positions: dict[int, np.ndarray] = {}
    with BridgeClient() as client:
        made = client.request(
            {
                "cmd": "make_batch",
                "batch_id": f"phase6-target-{spec.policy_id}",
                "count": seeds,
                "seed_start": seed_start,
                "sensor_tier": spec.sensor_tier,
                "env_config": {"horizon": horizon},
                "interventions": [basin_position_intervention()],
            }
        )
        obs_batch = np.asarray(made["obs"], dtype=np.float32)
        info_batch = made["info"]
        active = np.ones(seeds, dtype=bool)
        for _step in range(horizon + 1):
            actions = learned_actions(policy, obs_batch, obs_mean, obs_std)
            response = client.request(
                {
                    "cmd": "step_batch",
                    "batch_id": f"phase6-target-{spec.policy_id}",
                    "actions": actions.tolist(),
                }
            )
            obs_batch = np.asarray(response["obs"], dtype=np.float32)
            info_batch = response["info"]
            done = np.asarray(response["done"], dtype=bool)
            for index in np.flatnonzero(active & done):
                terminal_positions[seed_start + int(index)] = np.asarray(info_batch[int(index)]["position"], dtype=np.float32)
            active &= ~done
            if not np.any(active):
                break
        for index in np.flatnonzero(active):
            terminal_positions[seed_start + int(index)] = np.asarray(info_batch[int(index)]["position"], dtype=np.float32)
        client.request({"cmd": "close"})
    for seed, position in terminal_positions.items():
        targets[seed] = old_basin_pref(position)
    return targets


def run_intervened_targets_oracle(spec: PolicySpec, *, seed_start: int, seeds: int, horizon: int) -> dict[int, float]:
    targets: dict[int, float] = {}
    terminal_positions: dict[int, np.ndarray] = {}
    with BridgeClient() as client:
        made = client.request(
            {
                "cmd": "make_batch",
                "batch_id": f"phase6-target-{spec.policy_id}",
                "count": seeds,
                "seed_start": seed_start,
                "sensor_tier": spec.sensor_tier,
                "env_config": {"horizon": horizon},
                "interventions": [basin_position_intervention()],
            }
        )
        obs_batch = np.asarray(made["obs"], dtype=np.float32)
        info_batch = made["info"]
        active = np.ones(seeds, dtype=bool)
        for _step in range(horizon + 1):
            actions = []
            for index, info in enumerate(info_batch):
                actions.append(oracle_action(info).tolist() if active[index] else [0.0, 0.0])
            response = client.request(
                {
                    "cmd": "step_batch",
                    "batch_id": f"phase6-target-{spec.policy_id}",
                    "actions": actions,
                }
            )
            obs_batch = np.asarray(response["obs"], dtype=np.float32)
            info_batch = response["info"]
            done = np.asarray(response["done"], dtype=bool)
            for index in np.flatnonzero(active & done):
                terminal_positions[seed_start + int(index)] = np.asarray(info_batch[int(index)]["position"], dtype=np.float32)
            active &= ~done
            if not np.any(active):
                break
        for index in np.flatnonzero(active):
            terminal_positions[seed_start + int(index)] = np.asarray(info_batch[int(index)]["position"], dtype=np.float32)
        client.request({"cmd": "close"})
    for seed, position in terminal_positions.items():
        targets[seed] = old_basin_pref(position)
    return targets


def collect_learned(
    spec: PolicySpec,
    *,
    seed_start: int,
    seeds: int,
    horizon: int,
    include_behavior_target: bool = False,
) -> Collection:
    assert spec.checkpoint is not None
    ensure_checkpoint(spec.checkpoint)
    policy, obs_mean, obs_std = policy_from_checkpoint(load_checkpoint(spec.checkpoint))
    policy.eval()
    targets = (
        run_intervened_targets_learned(
            spec,
            policy,
            obs_mean,
            obs_std,
            seed_start=seed_start,
            seeds=seeds,
            horizon=horizon,
        )
        if include_behavior_target
        else {}
    )
    activations, handles = register_tanh_hooks(policy)

    seed_values: list[int] = []
    positions: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    false_centers: list[np.ndarray] = []
    basin_pref_targets: list[float] = []
    layer_values: dict[str, list[np.ndarray]] = {}

    try:
        with BridgeClient() as client:
            made = client.request(
                {
                    "cmd": "make_batch",
                    "batch_id": f"phase6-{spec.policy_id}",
                    "count": seeds,
                    "seed_start": seed_start,
                    "sensor_tier": spec.sensor_tier,
                    "env_config": {"horizon": horizon},
                }
            )
            obs_batch = np.asarray(made["obs"], dtype=np.float32)
            info_batch = made["info"]
            active = np.ones(seeds, dtype=bool)
            for _step in range(horizon + 1):
                actions = learned_actions(policy, obs_batch, obs_mean, obs_std)
                for index in np.flatnonzero(active):
                    append_sample(
                        seeds=seed_values,
                        positions=positions,
                        goals=goals,
                        false_centers=false_centers,
                        basin_pref_targets=basin_pref_targets,
                        layer_values=layer_values,
                        seed=seed_start + int(index),
                        obs=obs_batch[index],
                        info=info_batch[index],
                        basin_pref_target=targets.get(seed_start + int(index), float("nan")),
                        activations=activations,
                        activation_index=int(index),
                    )
                response = client.request(
                    {
                        "cmd": "step_batch",
                        "batch_id": f"phase6-{spec.policy_id}",
                        "actions": actions.tolist(),
                    }
                )
                obs_batch = np.asarray(response["obs"], dtype=np.float32)
                info_batch = response["info"]
                done = np.asarray(response["done"], dtype=bool)
                active &= ~done
                if not np.any(active):
                    break
            client.request({"cmd": "close"})
    finally:
        for handle in handles:
            handle.remove()

    return make_collection(spec, seed_values, positions, goals, false_centers, basin_pref_targets, layer_values)


def collect_oracle(
    spec: PolicySpec,
    *,
    seed_start: int,
    seeds: int,
    horizon: int,
    include_behavior_target: bool = False,
) -> Collection:
    targets = (
        run_intervened_targets_oracle(spec, seed_start=seed_start, seeds=seeds, horizon=horizon)
        if include_behavior_target
        else {}
    )
    seed_values: list[int] = []
    positions: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    false_centers: list[np.ndarray] = []
    basin_pref_targets: list[float] = []
    layer_values: dict[str, list[np.ndarray]] = {}

    with BridgeClient() as client:
        made = client.request(
            {
                "cmd": "make_batch",
                "batch_id": f"phase6-{spec.policy_id}",
                "count": seeds,
                "seed_start": seed_start,
                "sensor_tier": spec.sensor_tier,
                "env_config": {"horizon": horizon},
            }
        )
        obs_batch = np.asarray(made["obs"], dtype=np.float32)
        info_batch = made["info"]
        active = np.ones(seeds, dtype=bool)
        for _step in range(horizon + 1):
            actions = []
            for index, info in enumerate(info_batch):
                actions.append(oracle_action(info).tolist() if active[index] else [0.0, 0.0])
            for index in np.flatnonzero(active):
                append_sample(
                    seeds=seed_values,
                    positions=positions,
                    goals=goals,
                    false_centers=false_centers,
                    basin_pref_targets=basin_pref_targets,
                    layer_values=layer_values,
                    seed=seed_start + int(index),
                    obs=obs_batch[index],
                    info=info_batch[index],
                    basin_pref_target=targets.get(seed_start + int(index), float("nan")),
                    activations=None,
                )
            response = client.request(
                {
                    "cmd": "step_batch",
                    "batch_id": f"phase6-{spec.policy_id}",
                    "actions": actions,
                }
            )
            obs_batch = np.asarray(response["obs"], dtype=np.float32)
            info_batch = response["info"]
            done = np.asarray(response["done"], dtype=bool)
            active &= ~done
            if not np.any(active):
                break
        client.request({"cmd": "close"})

    return make_collection(spec, seed_values, positions, goals, false_centers, basin_pref_targets, layer_values)


def make_collection(
    spec: PolicySpec,
    seed_values: list[int],
    positions: list[np.ndarray],
    goals: list[np.ndarray],
    false_centers: list[np.ndarray],
    basin_pref_targets: list[float],
    layer_values: dict[str, list[np.ndarray]],
) -> Collection:
    return Collection(
        policy_id=spec.policy_id,
        label=spec.label,
        kind=spec.kind,
        seeds=np.asarray(seed_values, dtype=np.int64),
        positions=np.stack(positions).astype(np.float32),
        goals=np.stack(goals).astype(np.float32),
        false_centers=np.stack(false_centers).astype(np.float32),
        basin_pref_targets=np.asarray(basin_pref_targets, dtype=np.float32),
        layers={name: np.stack(values).astype(np.float32) for name, values in layer_values.items()},
    )


def feature_target(collection: Collection, feature: str) -> np.ndarray:
    if feature == "basin_pref_intervened":
        return collection.basin_pref_targets
    if feature == "dist_to_x_goal":
        return np.linalg.norm(collection.goals - collection.positions, axis=1)
    if feature == "dist_to_x_false":
        return np.linalg.norm(collection.false_centers - collection.positions, axis=1)
    if feature == "vec_to_x_goal":
        return collection.goals - collection.positions
    if feature == "vec_to_x_false":
        return collection.false_centers - collection.positions
    raise KeyError(feature)


def episode_split(seed_values: np.ndarray, *, train_frac: float = 0.8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(seed_values)
    if len(unique) < 2:
        raise ValueError("need at least two episodes for split")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_train = min(max(1, int(math.floor(len(shuffled) * train_frac))), len(shuffled) - 1)
    train_seeds = set(int(x) for x in shuffled[:n_train])
    train_mask = np.asarray([int(s) in train_seeds for s in seed_values])
    return np.flatnonzero(train_mask), np.flatnonzero(~train_mask)


def fit_probe_rows(collection: Collection, *, include_behavior_target: bool = False) -> list[dict[str, Any]]:
    train_idx, test_idx = episode_split(collection.seeds)
    rows: list[dict[str, Any]] = []
    features = (*BEHAVIOR_FEATURES, *GEOMETRY_FEATURES) if include_behavior_target else GEOMETRY_FEATURES
    for layer, values in collection.layers.items():
        X = values.astype(np.float64)
        for feature in features:
            y = feature_target(collection, feature).astype(np.float64)
            probe = Ridge(alpha=1.0)
            probe.fit(X[train_idx], y[train_idx])
            pred_train = probe.predict(X[train_idx])
            pred_test = probe.predict(X[test_idx])

            rng = np.random.default_rng(0)
            y_shuffled = y.copy()
            rng.shuffle(y_shuffled, axis=0)
            shuffled_probe = Ridge(alpha=1.0)
            shuffled_probe.fit(X[train_idx], y_shuffled[train_idx])
            pred_shuffled = shuffled_probe.predict(X[test_idx])

            rows.append(
                {
                    "policy_id": collection.policy_id,
                    "policy_label": collection.label,
                    "policy_kind": collection.kind,
                    "layer": layer,
                    "feature": feature,
                    "target_dim": 1 if y.ndim == 1 else y.shape[1],
                    "r2_train": float(r2_score(y[train_idx], pred_train)),
                    "r2_test": float(r2_score(y[test_idx], pred_test)),
                    "r2_shuffled": float(r2_score(y_shuffled[test_idx], pred_shuffled)),
                    "n_samples": int(len(collection.seeds)),
                    "n_train_samples": int(len(train_idx)),
                    "n_test_samples": int(len(test_idx)),
                    "n_train_episodes": int(len(np.unique(collection.seeds[train_idx]))),
                    "n_test_episodes": int(len(np.unique(collection.seeds[test_idx]))),
                }
            )
    input_baselines = {
        row["feature"]: row["r2_test"]
        for row in rows
        if row["layer"] == "input.obs"
    }
    for row in rows:
        baseline = input_baselines.get(row["feature"])
        row["delta_r2_vs_input"] = "" if baseline is None else float(row["r2_test"] - baseline)
    return rows


def collect_policy(
    spec: PolicySpec,
    *,
    seed_start: int,
    seeds: int,
    horizon: int,
    include_behavior_target: bool = False,
) -> Collection:
    if spec.kind == "learned":
        return collect_learned(
            spec,
            seed_start=seed_start,
            seeds=seeds,
            horizon=horizon,
            include_behavior_target=include_behavior_target,
        )
    if spec.kind == "oracle":
        return collect_oracle(
            spec,
            seed_start=seed_start,
            seeds=seeds,
            horizon=horizon,
            include_behavior_target=include_behavior_target,
        )
    raise ValueError(f"unknown policy kind: {spec.kind}")


def load_learned_policy(spec: PolicySpec) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    if spec.checkpoint is None:
        raise ValueError(f"{spec.label} has no checkpoint")
    ensure_checkpoint(spec.checkpoint)
    policy, obs_mean, obs_std = policy_from_checkpoint(load_checkpoint(spec.checkpoint))
    policy.eval()
    return policy, obs_mean, obs_std


def get_module(policy: torch.nn.Module, layer: str) -> torch.nn.Module:
    modules = dict(policy.named_modules())
    if layer not in modules:
        available = ", ".join(name for name in modules if name)
        raise KeyError(f"unknown layer {layer!r}; available layers: {available}")
    return modules[layer]


def run_patched_rollout(
    client: BridgeClient,
    *,
    policy: torch.nn.Module,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seed: int,
    horizon: int,
    layer: str,
    env_id: str,
    condition: str,
    record: bool = False,
    inject_activations: list[np.ndarray] | None = None,
) -> PatchRollout:
    interventions = [basin_position_intervention()] if condition == "intervened" else []
    made = client.request(
        {
            "cmd": "make",
            "env_id": env_id,
            "seed": seed,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": horizon},
            "interventions": interventions,
        }
    )
    obs = np.asarray(made["obs"], dtype=np.float32)
    info = made["info"]
    terminal_position = np.asarray(info["position"], dtype=np.float32)
    terminal_outcome = "not_done"
    terminal_alignment = float("nan")  # PHASE6B_SPEC v1.1 §5: captured on done.
    captures: list[np.ndarray] = []
    step_index = 0

    module = get_module(policy, layer)

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor | None:
        nonlocal step_index
        if inject_activations is not None:
            if not inject_activations:
                raise RuntimeError("cannot inject empty activation cache")
            source = inject_activations[min(step_index, len(inject_activations) - 1)]
            return torch.as_tensor(source, dtype=output.dtype, device=output.device)
        if record:
            captures.append(output.detach().cpu().numpy().copy())
        return None

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
                # PHASE6B_SPEC v1.1 §5: bridge exposes metrics.terminalAlignment
                # in info on done=true (see scripts/mesa-env-bridge.mjs asInfo()).
                metrics = info.get("metrics") or {}
                ta = metrics.get("terminalAlignment")
                if ta is not None:
                    terminal_alignment = float(ta)
                break
    finally:
        handle.remove()

    return PatchRollout(
        old_basin_pref=old_basin_pref(terminal_position),
        terminal_position=terminal_position,
        terminal_outcome=terminal_outcome,
        steps=step_index,
        activations=captures,
        terminal_alignment=terminal_alignment,
    )


def safe_patch_success(reference_a: float, reference_b: float, patched: float, *, direction: str) -> float:
    if direction == "protected_to_collapsed":
        denom = reference_b - reference_a
        return float("nan") if abs(denom) < 1e-9 else float((reference_b - patched) / denom)
    if direction == "collapsed_to_protected":
        denom = reference_a - reference_b
        return float("nan") if abs(denom) < 1e-9 else float((reference_a - patched) / denom)
    raise ValueError(direction)


def mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float("nan") if not finite else float(np.mean(finite))


def median_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float("nan") if not finite else float(np.median(finite))


def ratio_of_means(values_a: list[float], values_b: list[float], values_patched: list[float], *, direction: str) -> float:
    mean_a = mean_finite(values_a)
    mean_b = mean_finite(values_b)
    mean_patched = mean_finite(values_patched)
    if direction == "protected_to_collapsed":
        denom = mean_b - mean_a
        return float("nan") if abs(denom) < 1e-9 else float((mean_b - mean_patched) / denom)
    if direction == "collapsed_to_protected":
        denom = mean_a - mean_b
        return float("nan") if abs(denom) < 1e-9 else float((mean_a - mean_patched) / denom)
    raise ValueError(direction)


def run_axis_b_patch(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cliff_pair_key = getattr(args, "cliff_pair", "medium-v1")
    if cliff_pair_key not in CLIFF_PAIRS:
        raise ValueError(
            f"unknown cliff pair {cliff_pair_key!r}; "
            f"expected one of {sorted(CLIFF_PAIRS)}"
        )
    protected_spec, collapsed_spec = CLIFF_PAIRS[cliff_pair_key]
    protected_policy, protected_mean, protected_std = load_learned_policy(protected_spec)
    collapsed_policy, collapsed_mean, collapsed_std = load_learned_policy(collapsed_spec)
    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    invalid = [condition for condition in conditions if condition not in {"clean", "intervened"}]
    if invalid:
        raise ValueError(f"unknown conditions: {invalid}")
    layer_arg = args.layers if args.layers else args.layer
    layers = [part.strip() for part in layer_arg.split(",") if part.strip()]
    if not layers:
        raise ValueError("at least one layer is required")

    rows: list[dict[str, Any]] = []
    with BridgeClient() as client:
        for layer in layers:
            print(f"phase6 axis-b patch: layer={layer}", flush=True)
            for condition in conditions:
                for offset in range(args.seeds):
                    seed = args.seed_start + offset
                    prefix = f"phase6-axis-b-{condition}-{layer.replace('.', '_')}-{seed}"
                    clean_protected = run_patched_rollout(
                        client,
                        policy=protected_policy,
                        obs_mean=protected_mean,
                        obs_std=protected_std,
                        seed=seed,
                        horizon=args.horizon,
                        layer=layer,
                        env_id=f"{prefix}-A",
                        condition=condition,
                        record=True,
                    )
                    clean_collapsed = run_patched_rollout(
                        client,
                        policy=collapsed_policy,
                        obs_mean=collapsed_mean,
                        obs_std=collapsed_std,
                        seed=seed,
                        horizon=args.horizon,
                        layer=layer,
                        env_id=f"{prefix}-B",
                        condition=condition,
                        record=True,
                    )
                    patched_protected_to_collapsed = run_patched_rollout(
                        client,
                        policy=collapsed_policy,
                        obs_mean=collapsed_mean,
                        obs_std=collapsed_std,
                        seed=seed,
                        horizon=args.horizon,
                        layer=layer,
                        env_id=f"{prefix}-C",
                        condition=condition,
                        inject_activations=clean_protected.activations,
                    )
                    patched_collapsed_to_protected = run_patched_rollout(
                        client,
                        policy=protected_policy,
                        obs_mean=protected_mean,
                        obs_std=protected_std,
                        seed=seed,
                        horizon=args.horizon,
                        layer=layer,
                        env_id=f"{prefix}-D",
                        condition=condition,
                        inject_activations=clean_collapsed.activations,
                    )
                    success_pc = safe_patch_success(
                        clean_protected.old_basin_pref,
                        clean_collapsed.old_basin_pref,
                        patched_protected_to_collapsed.old_basin_pref,
                        direction="protected_to_collapsed",
                    )
                    success_cp = safe_patch_success(
                        clean_protected.old_basin_pref,
                        clean_collapsed.old_basin_pref,
                        patched_collapsed_to_protected.old_basin_pref,
                        direction="collapsed_to_protected",
                    )
                    # PHASE6B_SPEC v1.1 §6: parallel alignment-normalized metric.
                    # safe_patch_success is metric-agnostic; same formula, different
                    # reference values. NaN-propagates if any rollout truncated
                    # without `done` (terminal_alignment unset).
                    success_pc_align = safe_patch_success(
                        clean_protected.terminal_alignment,
                        clean_collapsed.terminal_alignment,
                        patched_protected_to_collapsed.terminal_alignment,
                        direction="protected_to_collapsed",
                    )
                    success_cp_align = safe_patch_success(
                        clean_protected.terminal_alignment,
                        clean_collapsed.terminal_alignment,
                        patched_collapsed_to_protected.terminal_alignment,
                        direction="collapsed_to_protected",
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "layer": layer,
                            "protected_old_basin_pref": clean_protected.old_basin_pref,
                            "collapsed_old_basin_pref": clean_collapsed.old_basin_pref,
                            "patched_protected_to_collapsed_old_basin_pref": patched_protected_to_collapsed.old_basin_pref,
                            "patched_collapsed_to_protected_old_basin_pref": patched_collapsed_to_protected.old_basin_pref,
                            "patch_success_protected_to_collapsed": success_pc,
                            "patch_success_collapsed_to_protected": success_cp,
                            "baseline_gap_collapsed_minus_protected": clean_collapsed.old_basin_pref - clean_protected.old_basin_pref,
                            # PHASE6B_SPEC v1.1 §5: alignment-based parallel metric.
                            "protected_terminal_alignment": clean_protected.terminal_alignment,
                            "collapsed_terminal_alignment": clean_collapsed.terminal_alignment,
                            "patched_protected_to_collapsed_terminal_alignment": patched_protected_to_collapsed.terminal_alignment,
                            "patched_collapsed_to_protected_terminal_alignment": patched_collapsed_to_protected.terminal_alignment,
                            "patch_success_align_protected_to_collapsed": success_pc_align,
                            "patch_success_align_collapsed_to_protected": success_cp_align,
                            "baseline_gap_align_protected_minus_collapsed": clean_protected.terminal_alignment - clean_collapsed.terminal_alignment,
                            "protected_outcome": clean_protected.terminal_outcome,
                            "collapsed_outcome": clean_collapsed.terminal_outcome,
                            "patched_protected_to_collapsed_outcome": patched_protected_to_collapsed.terminal_outcome,
                            "patched_collapsed_to_protected_outcome": patched_collapsed_to_protected.terminal_outcome,
                            "protected_steps": clean_protected.steps,
                            "collapsed_steps": clean_collapsed.steps,
                            "patched_protected_to_collapsed_steps": patched_protected_to_collapsed.steps,
                            "patched_collapsed_to_protected_steps": patched_collapsed_to_protected.steps,
                        }
                    )
        client.request({"cmd": "close"})

    aggregate_rows: list[dict[str, Any]] = []
    for layer in layers:
        for condition in conditions:
            condition_rows = [row for row in rows if row["condition"] == condition and row["layer"] == layer]
            protected_values = [float(row["protected_old_basin_pref"]) for row in condition_rows]
            collapsed_values = [float(row["collapsed_old_basin_pref"]) for row in condition_rows]
            patched_pc_values = [float(row["patched_protected_to_collapsed_old_basin_pref"]) for row in condition_rows]
            patched_cp_values = [float(row["patched_collapsed_to_protected_old_basin_pref"]) for row in condition_rows]
            success_pc_values = [float(row["patch_success_protected_to_collapsed"]) for row in condition_rows]
            success_cp_values = [float(row["patch_success_collapsed_to_protected"]) for row in condition_rows]
            # PHASE6B_SPEC v1.1 §6: alignment-normalized parallel metric.
            protected_align_values = [float(row["protected_terminal_alignment"]) for row in condition_rows]
            collapsed_align_values = [float(row["collapsed_terminal_alignment"]) for row in condition_rows]
            patched_pc_align_values = [float(row["patched_protected_to_collapsed_terminal_alignment"]) for row in condition_rows]
            patched_cp_align_values = [float(row["patched_collapsed_to_protected_terminal_alignment"]) for row in condition_rows]
            success_pc_align_values = [float(row["patch_success_align_protected_to_collapsed"]) for row in condition_rows]
            success_cp_align_values = [float(row["patch_success_align_collapsed_to_protected"]) for row in condition_rows]
            align_gap_values = [float(row["baseline_gap_align_protected_minus_collapsed"]) for row in condition_rows]
            aggregate_rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "direction": "protected_to_collapsed",
                    "mean_patch_success": mean_finite(success_pc_values),
                    "median_patch_success": median_finite(success_pc_values),
                    "patch_success_ratio_of_means": ratio_of_means(
                        protected_values,
                        collapsed_values,
                        patched_pc_values,
                        direction="protected_to_collapsed",
                    ),
                    "mean_baseline_gap": mean_finite([float(row["baseline_gap_collapsed_minus_protected"]) for row in condition_rows]),
                    "mean_protected_old_basin_pref": mean_finite(protected_values),
                    "mean_collapsed_old_basin_pref": mean_finite(collapsed_values),
                    "mean_patched_old_basin_pref": mean_finite(patched_pc_values),
                    # PHASE6B_SPEC v1.1 §6: alignment-normalized columns. The
                    # canonical v1.1 reading lives here; basin-pref columns
                    # above are retained for transparency only.
                    "mean_patch_success_align": mean_finite(success_pc_align_values),
                    "median_patch_success_align": median_finite(success_pc_align_values),
                    "patch_success_align_ratio_of_means": ratio_of_means(
                        protected_align_values,
                        collapsed_align_values,
                        patched_pc_align_values,
                        direction="protected_to_collapsed",
                    ),
                    "mean_baseline_gap_align": mean_finite(align_gap_values),
                    "mean_protected_alignment": mean_finite(protected_align_values),
                    "mean_collapsed_alignment": mean_finite(collapsed_align_values),
                    "mean_patched_alignment": mean_finite(patched_pc_align_values),
                    "n": len(condition_rows),
                }
            )
            aggregate_rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "direction": "collapsed_to_protected",
                    "mean_patch_success": mean_finite(success_cp_values),
                    "median_patch_success": median_finite(success_cp_values),
                    "patch_success_ratio_of_means": ratio_of_means(
                        protected_values,
                        collapsed_values,
                        patched_cp_values,
                        direction="collapsed_to_protected",
                    ),
                    "mean_baseline_gap": mean_finite([float(row["baseline_gap_collapsed_minus_protected"]) for row in condition_rows]),
                    "mean_protected_old_basin_pref": mean_finite(protected_values),
                    "mean_collapsed_old_basin_pref": mean_finite(collapsed_values),
                    "mean_patched_old_basin_pref": mean_finite(patched_cp_values),
                    "mean_patch_success_align": mean_finite(success_cp_align_values),
                    "median_patch_success_align": median_finite(success_cp_align_values),
                    "patch_success_align_ratio_of_means": ratio_of_means(
                        protected_align_values,
                        collapsed_align_values,
                        patched_cp_align_values,
                        direction="collapsed_to_protected",
                    ),
                    "mean_baseline_gap_align": mean_finite(align_gap_values),
                    "mean_protected_alignment": mean_finite(protected_align_values),
                    "mean_collapsed_alignment": mean_finite(collapsed_align_values),
                    "mean_patched_alignment": mean_finite(patched_cp_align_values),
                    "n": len(condition_rows),
                }
            )

    trial_path = out_dir / "axis-b-patch-smoke.csv"
    aggregate_path = out_dir / "axis-b-patch-smoke-aggregate.csv"
    manifest_path = out_dir / "manifest.json"
    write_csv(trial_path, rows)
    write_csv(aggregate_path, aggregate_rows)
    manifest = {
        "phase": "phase6-axis-b-smoke",
        "cliff_pair": cliff_pair_key,
        "protected": {
            "policy_id": protected_spec.policy_id,
            "label": protected_spec.label,
            "checkpoint": str(protected_spec.checkpoint.relative_to(REPO_ROOT)) if protected_spec.checkpoint else None,
        },
        "collapsed": {
            "policy_id": collapsed_spec.policy_id,
            "label": collapsed_spec.label,
            "checkpoint": str(collapsed_spec.checkpoint.relative_to(REPO_ROOT)) if collapsed_spec.checkpoint else None,
        },
        "layers": layers,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "horizon": args.horizon,
        "conditions": conditions,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"phase6 axis-b smoke: wrote {trial_path.relative_to(REPO_ROOT)}", flush=True)
    for row in aggregate_rows:
        # PHASE6B_SPEC v1.1: print both the v1 basin-pref metric (legacy
        # column, not used for v1.1 verdicts) and the canonical v1.1
        # alignment-normalized metric.
        layer_label = row.get("layer", "")
        print(
            f"  {layer_label} {row['condition']} {row['direction']}: "
            f"ps_align_mean={row['mean_patch_success_align']:.3f} "
            f"median={row['median_patch_success_align']:.3f} "
            f"ratio_of_means={row['patch_success_align_ratio_of_means']:.3f} "
            f"align_gap={row['mean_baseline_gap_align']:.3f}"
            f"   [v1: ps={row['mean_patch_success']:.3f} bp_gap={row['mean_baseline_gap']:.3f}]",
            flush=True,
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    labels = sorted({row["policy_label"] for row in rows})
    for label in labels:
        policy_rows = [row for row in rows if row["policy_label"] == label]
        hidden_rows = [row for row in policy_rows if row["layer"] not in {"input.obs", "oracle.privileged_ceiling"}]
        basin_rows = [row for row in policy_rows if row["feature"] == "basin_pref_intervened"]
        hidden_basin_rows = [row for row in hidden_rows if row["feature"] == "basin_pref_intervened"]
        input_basin = next((row for row in basin_rows if row["layer"] == "input.obs"), None)
        net_rows = [row for row in policy_rows if row["layer"].startswith("net.")]

        def first_net(feature: str) -> dict[str, Any] | None:
            candidates = [row for row in net_rows if row["feature"] == feature]
            return candidates[0] if candidates else None

        def last_net(feature: str) -> dict[str, Any] | None:
            candidates = [row for row in net_rows if row["feature"] == feature]
            return candidates[-1] if candidates else None

        net1_goal = first_net("dist_to_x_goal")
        net_last_goal = last_net("dist_to_x_goal")
        net1_false = first_net("dist_to_x_false")
        net_last_false = last_net("dist_to_x_false")

        summary.append(
            {
                "policy_label": label,
                "max_basin_pref_intervened_r2": max((row["r2_test"] for row in basin_rows), default=None),
                "max_hidden_basin_pref_intervened_r2": max((row["r2_test"] for row in hidden_basin_rows), default=None),
                "input_basin_pref_intervened_r2": None if input_basin is None else input_basin["r2_test"],
                "max_basin_pref_intervened_delta_r2": max(
                    (float(row["delta_r2_vs_input"]) for row in basin_rows if row["delta_r2_vs_input"] != ""),
                    default=None,
                ),
                "max_dist_to_x_goal_r2": max(row["r2_test"] for row in policy_rows if row["feature"] == "dist_to_x_goal"),
                "max_dist_to_x_false_r2": max(row["r2_test"] for row in policy_rows if row["feature"] == "dist_to_x_false"),
                "max_vec_to_x_goal_r2": max(row["r2_test"] for row in policy_rows if row["feature"] == "vec_to_x_goal"),
                "max_vec_to_x_false_r2": max(row["r2_test"] for row in policy_rows if row["feature"] == "vec_to_x_false"),
                "max_dist_to_x_goal_delta_r2": max(float(row["delta_r2_vs_input"]) for row in policy_rows if row["feature"] == "dist_to_x_goal" and row["delta_r2_vs_input"] != ""),
                "net_last_dist_to_x_false_r2": next(
                    (
                        row["r2_test"]
                        for row in reversed(policy_rows)
                        if row["layer"].startswith("net.") and row["feature"] == "dist_to_x_false"
                    ),
                    None,
                ),
                "net1_dist_to_x_goal_delta_r2": None if net1_goal is None else float(net1_goal["delta_r2_vs_input"]),
                "net_last_dist_to_x_goal_delta_r2": None if net_last_goal is None else float(net_last_goal["delta_r2_vs_input"]),
                "net1_dist_to_x_false_delta_r2": None if net1_false is None else float(net1_false["delta_r2_vs_input"]),
                "net_last_dist_to_x_false_delta_r2": None if net_last_false is None else float(net_last_false["delta_r2_vs_input"]),
                "dist_to_x_goal_delta_depth_slope": None if net1_goal is None or net_last_goal is None else float(net_last_goal["delta_r2_vs_input"] - net1_goal["delta_r2_vs_input"]),
                "dist_to_x_false_delta_depth_slope": None if net1_false is None or net_last_false is None else float(net_last_false["delta_r2_vs_input"] - net1_false["delta_r2_vs_input"]),
                "net1_dist_to_x_goal_r2": next(
                    (
                        row["r2_test"]
                        for row in policy_rows
                        if row["layer"] == "net.1" and row["feature"] == "dist_to_x_goal"
                    ),
                    None,
                ),
                "net1_dist_to_x_false_r2": next(
                    (
                        row["r2_test"]
                        for row in policy_rows
                        if row["layer"] == "net.1" and row["feature"] == "dist_to_x_false"
                    ),
                    None,
                ),
                "max_abs_shuffled_r2": max(abs(row["r2_shuffled"]) for row in policy_rows),
                "n_rows": len(policy_rows),
            }
        )
    return summary


def run_axis_a_smoke(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "phase": "phase6-axis-a-smoke",
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "horizon": args.horizon,
        "policies": [],
        "notes": [
            "Oracle-S is an analytic privileged ceiling row, not a fittable neural hidden-layer policy.",
            "input.obs rows are raw-observation floors for geometric confound checks.",
            "v1.5 headline is delta_r2_vs_input depth profile for geometric rider features.",
            "Use --include-behavior-target only to reproduce the failed v1.2 endpoint-shaped basin_pref_intervened smoke.",
        ],
    }

    for spec in SMOKE_POLICIES:
        print(f"phase6 axis-a smoke: collecting {spec.label}", flush=True)
        collection = collect_policy(
            spec,
            seed_start=args.seed_start,
            seeds=args.seeds,
            horizon=args.horizon,
            include_behavior_target=args.include_behavior_target,
        )
        rows = fit_probe_rows(collection, include_behavior_target=args.include_behavior_target)
        all_rows.extend(rows)
        policy_manifest = {
            "policy_id": spec.policy_id,
            "label": spec.label,
            "kind": spec.kind,
            "sensor_tier": spec.sensor_tier,
            "checkpoint": str(spec.checkpoint.relative_to(REPO_ROOT)) if spec.checkpoint else None,
            "n_samples": int(len(collection.seeds)),
            "layers": sorted(collection.layers),
        }
        if args.include_behavior_target:
            policy_manifest.update({
                "basin_pref_target_mean": float(np.mean(collection.basin_pref_targets)),
                "basin_pref_target_std": float(np.std(collection.basin_pref_targets)),
                "basin_pref_target_min": float(np.min(collection.basin_pref_targets)),
                "basin_pref_target_max": float(np.max(collection.basin_pref_targets)),
            })
        manifest["policies"].append(policy_manifest)

    accuracy_path = out_dir / "axis-a-smoke-probe-accuracy.csv"
    summary_path = out_dir / "axis-a-smoke-summary.csv"
    manifest_path = out_dir / "manifest.json"
    summary_rows = summarize(all_rows)
    write_csv(accuracy_path, all_rows)
    write_csv(summary_path, summary_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"phase6 axis-a smoke: wrote {accuracy_path.relative_to(REPO_ROOT)}", flush=True)
    for row in summary_rows:
        print(
            "  {policy_label}: goal_delta_last={goal_last} "
            "false_delta_last={false_last} max_goal={max_dist_to_x_goal_r2:.3f} "
            "max_abs_shuffled={max_abs_shuffled_r2:.3f}".format(
                **row,
                goal_last="-" if row["net_last_dist_to_x_goal_delta_r2"] is None else f"{row['net_last_dist_to_x_goal_delta_r2']:.3f}",
                false_last="-" if row["net_last_dist_to_x_false_delta_r2"] is None else f"{row['net_last_dist_to_x_false_delta_r2']:.3f}",
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 interpretability probe harness")
    sub = parser.add_subparsers(dest="command")
    smoke = sub.add_parser("axis-a-smoke", help="Run the three-policy Axis A smoke slate")
    smoke.add_argument("--out", default=str(PHASE6_OUT / "axis-a-smoke"))
    smoke.add_argument("--seed-start", type=int, default=10000)
    smoke.add_argument("--seeds", type=int, default=64)
    smoke.add_argument("--horizon", type=int, default=200)
    smoke.add_argument(
        "--include-behavior-target",
        action="store_true",
        help="also run paired basin-position interventions for the failed v1.2 basin_pref_intervened target",
    )
    patch = sub.add_parser("axis-b-smoke", help="Run cliff-pair activation patching smoke")
    patch.add_argument("--out", default=str(PHASE6_OUT / "axis-b-smoke"))
    patch.add_argument("--seed-start", type=int, default=10000)
    patch.add_argument("--seeds", type=int, default=8)
    patch.add_argument("--horizon", type=int, default=200)
    patch.add_argument("--layer", default="net.1")
    patch.add_argument("--layers", default="")
    patch.add_argument("--conditions", default="clean,intervened")
    patch.add_argument(
        "--cliff-pair",
        default="medium-v1",
        choices=sorted(CLIFF_PAIRS.keys()),
        help=(
            "which cliff pair to patch. 'medium-v1' (default) is the Phase 6 v1 "
            "Medium pair (mixed_0_95 vs mixed_0_97); 'large-v3' is the Phase 6b "
            "Large pair (mixed_0_99 vs mixed_0_97 at vc=0.25). See PHASE6B_SPEC.md §3."
        ),
    )
    args = parser.parse_args()
    if args.command is None:
        args.command = "axis-a-smoke"
        args.out = str(PHASE6_OUT / "axis-a-smoke")
        args.seed_start = 10000
        args.seeds = 64
        args.horizon = 200
        args.include_behavior_target = False
    return args


def main() -> None:
    args = parse_args()
    if args.command == "axis-a-smoke":
        run_axis_a_smoke(args)
    elif args.command == "axis-b-smoke":
        run_axis_b_patch(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
