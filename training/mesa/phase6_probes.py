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
    layers: dict[str, np.ndarray]


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


FEATURES = ("dist_to_x_goal", "dist_to_x_false", "vec_to_x_goal", "vec_to_x_false")


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


def append_sample(
    *,
    seeds: list[int],
    positions: list[np.ndarray],
    goals: list[np.ndarray],
    false_centers: list[np.ndarray],
    layer_values: dict[str, list[np.ndarray]],
    seed: int,
    obs: np.ndarray,
    info: dict[str, Any],
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


def collect_learned(spec: PolicySpec, *, seed_start: int, seeds: int, horizon: int) -> Collection:
    assert spec.checkpoint is not None
    ensure_checkpoint(spec.checkpoint)
    policy, obs_mean, obs_std = policy_from_checkpoint(load_checkpoint(spec.checkpoint))
    policy.eval()
    activations, handles = register_tanh_hooks(policy)

    seed_values: list[int] = []
    positions: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    false_centers: list[np.ndarray] = []
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
                        layer_values=layer_values,
                        seed=seed_start + int(index),
                        obs=obs_batch[index],
                        info=info_batch[index],
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

    return make_collection(spec, seed_values, positions, goals, false_centers, layer_values)


def collect_oracle(spec: PolicySpec, *, seed_start: int, seeds: int, horizon: int) -> Collection:
    seed_values: list[int] = []
    positions: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    false_centers: list[np.ndarray] = []
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
                    layer_values=layer_values,
                    seed=seed_start + int(index),
                    obs=obs_batch[index],
                    info=info_batch[index],
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

    return make_collection(spec, seed_values, positions, goals, false_centers, layer_values)


def make_collection(
    spec: PolicySpec,
    seed_values: list[int],
    positions: list[np.ndarray],
    goals: list[np.ndarray],
    false_centers: list[np.ndarray],
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
        layers={name: np.stack(values).astype(np.float32) for name, values in layer_values.items()},
    )


def feature_target(collection: Collection, feature: str) -> np.ndarray:
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


def fit_probe_rows(collection: Collection) -> list[dict[str, Any]]:
    train_idx, test_idx = episode_split(collection.seeds)
    rows: list[dict[str, Any]] = []
    for layer, values in collection.layers.items():
        X = values.astype(np.float64)
        for feature in FEATURES:
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
    return rows


def collect_policy(spec: PolicySpec, *, seed_start: int, seeds: int, horizon: int) -> Collection:
    if spec.kind == "learned":
        return collect_learned(spec, seed_start=seed_start, seeds=seeds, horizon=horizon)
    if spec.kind == "oracle":
        return collect_oracle(spec, seed_start=seed_start, seeds=seeds, horizon=horizon)
    raise ValueError(f"unknown policy kind: {spec.kind}")


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
        summary.append(
            {
                "policy_label": label,
                "max_dist_to_x_goal_r2": max(row["r2_test"] for row in policy_rows if row["feature"] == "dist_to_x_goal"),
                "max_dist_to_x_false_r2": max(row["r2_test"] for row in policy_rows if row["feature"] == "dist_to_x_false"),
                "max_vec_to_x_goal_r2": max(row["r2_test"] for row in policy_rows if row["feature"] == "vec_to_x_goal"),
                "max_vec_to_x_false_r2": max(row["r2_test"] for row in policy_rows if row["feature"] == "vec_to_x_false"),
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
        ],
    }

    for spec in SMOKE_POLICIES:
        print(f"phase6 axis-a smoke: collecting {spec.label}", flush=True)
        collection = collect_policy(spec, seed_start=args.seed_start, seeds=args.seeds, horizon=args.horizon)
        rows = fit_probe_rows(collection)
        all_rows.extend(rows)
        manifest["policies"].append(
            {
                "policy_id": spec.policy_id,
                "label": spec.label,
                "kind": spec.kind,
                "sensor_tier": spec.sensor_tier,
                "checkpoint": str(spec.checkpoint.relative_to(REPO_ROOT)) if spec.checkpoint else None,
                "n_samples": int(len(collection.seeds)),
                "layers": sorted(collection.layers),
            }
        )

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
            "  {policy_label}: max_goal={max_dist_to_x_goal_r2:.3f} "
            "max_false={max_dist_to_x_false_r2:.3f} net1_goal={net1} "
            "max_abs_shuffled={max_abs_shuffled_r2:.3f}".format(
                **row,
                net1="-" if row["net1_dist_to_x_goal_r2"] is None else f"{row['net1_dist_to_x_goal_r2']:.3f}",
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
    args = parser.parse_args()
    if args.command is None:
        args.command = "axis-a-smoke"
        args.out = str(PHASE6_OUT / "axis-a-smoke")
        args.seed_start = 10000
        args.seeds = 64
        args.horizon = 200
    return args


def main() -> None:
    args = parse_args()
    if args.command == "axis-a-smoke":
        run_axis_a_smoke(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
