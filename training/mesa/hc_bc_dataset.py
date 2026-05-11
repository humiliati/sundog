"""Behavior-cloning dataset loader for Phase 2 mesa training.

The public class is PyTorch-native when torch/numpy are installed. The parsing
and validation helpers intentionally use only the standard library so Phase 2
artifact smoke tests can run before learning dependencies are present.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:  # Optional until Phase 2 training dependencies are installed.
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by local smoke env.
    np = None  # type: ignore

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by local smoke env.
    torch = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "results" / "mesa" / "phase1-hc-baseline" / "manifest.json"


class _DatasetBase:
    pass


if torch is not None:
    _DatasetBase = torch.utils.data.Dataset  # type: ignore[assignment]


@dataclass(frozen=True)
class Trajectory:
    seed: int
    path: Path
    success: bool
    obs: list[list[float]]
    actions: list[list[float]]
    phase_labels: list[str]

    @property
    def length(self) -> int:
        return len(self.obs)


@dataclass(frozen=True)
class BcSplit:
    manifest_path: Path
    sensor_tier: str
    successful_only: bool
    bc_seeds: list[int]
    train_seeds: list[int]
    val_seeds: list[int]
    excluded_due_to_filter: list[int]
    train_trajectories: list[Trajectory]
    val_trajectories: list[Trajectory]
    obs_mean: list[float]
    obs_std: list[float]
    obs_variance: list[float]
    config_hash: str

    @property
    def n_train_pairs(self) -> int:
        return sum(traj.length for traj in self.train_trajectories)

    @property
    def n_val_pairs(self) -> int:
        return sum(traj.length for traj in self.val_trajectories)

    @property
    def trajectory_count(self) -> int:
        return len(self.train_trajectories) + len(self.val_trajectories)

    @property
    def included_success_rate(self) -> float:
        trajectories = [*self.train_trajectories, *self.val_trajectories]
        if not trajectories:
            return 0.0
        return sum(1 for traj in trajectories if traj.success) / len(trajectories)

    @property
    def avg_trajectory_length(self) -> float:
        trajectories = [*self.train_trajectories, *self.val_trajectories]
        if not trajectories:
            return 0.0
        return sum(traj.length for traj in trajectories) / len(trajectories)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_repo_path(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = (REPO_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return (manifest_path.parent / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
    return entries


def _trial_matches(entries: list[dict[str, Any]], *, sensor_tier: str) -> bool:
    header = next((entry for entry in entries if entry.get("type") == "header"), None)
    if not header:
        return False
    return header.get("controllerFamily") == "hc_signature" and header.get("sensorTier") == sensor_tier


def _parse_trial(path: Path, *, sensor_tier: str) -> Trajectory | None:
    entries = _read_jsonl(path)
    if not _trial_matches(entries, sensor_tier=sensor_tier):
        return None

    header = next(entry for entry in entries if entry.get("type") == "header")
    terminal = next((entry for entry in entries if entry.get("type") == "terminal"), None)
    step_entries = [entry for entry in entries if entry.get("type") == "step"]
    obs = [entry["obs"] for entry in step_entries]
    actions = [entry["a"] for entry in step_entries]
    phase_labels = [entry.get("phaseLabel", "") for entry in step_entries]
    return Trajectory(
        seed=int(header["seed"]),
        path=path,
        success=terminal is not None and terminal.get("outcome") == "success",
        obs=obs,
        actions=actions,
        phase_labels=phase_labels,
    )


def _split_seeds(seeds: list[int], *, seed_base: int) -> tuple[list[int], list[int]]:
    ordered = sorted(seeds)
    if seed_base != 0:
        rng = random.Random(seed_base)
        ordered = ordered[:]
        rng.shuffle(ordered)
    val_count = max(1, math.ceil(len(ordered) / 8))
    train_count = max(1, len(ordered) - val_count)
    return ordered[:train_count], ordered[train_count:]


def _flatten_pairs(trajectories: list[Trajectory]) -> tuple[list[list[float]], list[list[float]]]:
    obs: list[list[float]] = []
    actions: list[list[float]] = []
    for trajectory in trajectories:
        obs.extend(trajectory.obs)
        actions.extend(trajectory.actions)
    return obs, actions


def _column_stats(rows: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    if not rows:
        raise ValueError("dataset contains no rows")
    width = len(rows[0])
    means = [0.0] * width
    for row in rows:
        for index, value in enumerate(row):
            means[index] += value
    means = [value / len(rows) for value in means]

    variances = [0.0] * width
    for row in rows:
        for index, value in enumerate(row):
            delta = value - means[index]
            variances[index] += delta * delta
    variances = [value / len(rows) for value in variances]
    stds = [math.sqrt(value) if value > 0 else 1.0 for value in variances]
    return means, stds, variances


def _is_finite_matrix(rows: list[list[float]]) -> bool:
    return all(math.isfinite(value) for row in rows for value in row)


def _validate_pairs(
    *,
    obs: list[list[float]],
    actions: list[list[float]],
    sensor_tier: str,
    action_max: float,
    obs_variance: list[float],
) -> None:
    if sensor_tier == "local-probe-field" and any(len(row) != 6 for row in obs):
        bad = next(len(row) for row in obs if len(row) != 6)
        raise ValueError(f"obs.shape must be (N, 6) for local-probe-field; saw row width {bad}")
    if any(len(row) != 2 for row in actions):
        bad = next(len(row) for row in actions if len(row) != 2)
        raise ValueError(f"action.shape must be (N, 2); saw row width {bad}")
    if not _is_finite_matrix(obs):
        raise ValueError("obs contains NaN or Inf")
    if not _is_finite_matrix(actions):
        raise ValueError("action contains NaN or Inf")
    limit = action_max + 1e-6
    for row_index, row in enumerate(actions):
        for col_index, value in enumerate(row):
            if abs(value) > limit:
                raise ValueError(f"action[{row_index},{col_index}]={value} exceeds a_max={action_max}")
    for index, variance in enumerate(obs_variance):
        if variance <= 0:
            raise ValueError(f"obs channel {index} has zero variance")


def build_hc_bc_split(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    sensor_tier: str = "local-probe-field",
    successful_only: bool = False,
    seed_base: int = 0,
) -> BcSplit:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    env_config = manifest.get("env", {})
    action_max = float(env_config.get("action_max", env_config.get("actionMax", env_config.get("a_max", 1.0))))

    trajectories: list[Trajectory] = []
    for trial_path_text in manifest.get("trial_paths", []):
        path = _resolve_repo_path(trial_path_text, manifest_path)
        trajectory = _parse_trial(path, sensor_tier=sensor_tier)
        if trajectory is not None:
            trajectories.append(trajectory)

    if not trajectories:
        raise ValueError(f"no hc_signature trajectories found for sensor_tier={sensor_tier!r}")

    trajectories_by_seed = {trajectory.seed: trajectory for trajectory in sorted(trajectories, key=lambda item: item.seed)}
    bc_seeds = sorted(trajectories_by_seed)
    included = [
        trajectory
        for trajectory in trajectories_by_seed.values()
        if (trajectory.success or not successful_only)
    ]
    excluded_due_to_filter = [
        trajectory.seed
        for trajectory in trajectories_by_seed.values()
        if successful_only and not trajectory.success
    ]
    expected_count = len(bc_seeds) - len(excluded_due_to_filter)
    if len(included) != expected_count:
        raise ValueError(
            f"trajectory count mismatch: included={len(included)} expected={expected_count}"
        )

    train_seeds, val_seeds = _split_seeds([trajectory.seed for trajectory in included], seed_base=seed_base)
    train_seed_set = set(train_seeds)
    val_seed_set = set(val_seeds)
    train_trajectories = [trajectory for trajectory in included if trajectory.seed in train_seed_set]
    val_trajectories = [trajectory for trajectory in included if trajectory.seed in val_seed_set]

    train_obs, train_actions = _flatten_pairs(train_trajectories)
    val_obs, val_actions = _flatten_pairs(val_trajectories)
    all_obs = [*train_obs, *val_obs]
    all_actions = [*train_actions, *val_actions]
    obs_mean, obs_std, obs_variance = _column_stats(train_obs)
    _validate_pairs(
        obs=all_obs,
        actions=all_actions,
        sensor_tier=sensor_tier,
        action_max=action_max,
        obs_variance=obs_variance,
    )

    hash_payload = {
        "source_manifest": _repo_relative(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sensor_tier": sensor_tier,
        "bc_seeds": bc_seeds,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "successful_only": successful_only,
        "seed_base": seed_base,
        "n_train_pairs": len(train_obs),
        "n_val_pairs": len(val_obs),
    }
    config_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    return BcSplit(
        manifest_path=manifest_path,
        sensor_tier=sensor_tier,
        successful_only=successful_only,
        bc_seeds=bc_seeds,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        excluded_due_to_filter=excluded_due_to_filter,
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        obs_mean=obs_mean,
        obs_std=obs_std,
        obs_variance=obs_variance,
        config_hash=config_hash,
    )


def print_bc_summary(split: BcSplit) -> None:
    print(
        "bc_dataset: "
        f"{split.n_train_pairs} train pairs, "
        f"{split.n_val_pairs} val pairs, "
        f"{split.avg_trajectory_length:.1f} avg trajectory length, "
        f"{100 * split.included_success_rate:.1f}% successful trajectories included."
    )


def bc_dataset_manifest_block(split: BcSplit, *, cache_path: Path | None = None) -> dict[str, Any]:
    return {
        "source_manifest": _repo_relative(split.manifest_path),
        "sensor_tier": split.sensor_tier,
        "bc_seeds": split.bc_seeds,
        "train_seeds": split.train_seeds,
        "val_seeds": split.val_seeds,
        "successful_only": split.successful_only,
        "excluded_due_to_filter": split.excluded_due_to_filter,
        "n_train_pairs": split.n_train_pairs,
        "n_val_pairs": split.n_val_pairs,
        "obs_mean": split.obs_mean,
        "obs_std": split.obs_std,
        "cache_path": _repo_relative(cache_path) if cache_path else None,
        "config_hash": split.config_hash,
    }


class HCBcDataset(_DatasetBase):
    """Supervised dataset for behavior cloning from HC-Signature rollouts."""

    def __init__(
        self,
        manifest_path: Path,
        split: Literal["train", "val"] = "train",
        sensor_tier: str = "local-probe-field",
        successful_only: bool = False,
        normalize: bool = True,
        seed_base: int = 0,
        cache_dir: Path | None = None,
    ) -> None:
        if np is None or torch is None:
            raise ImportError("HCBcDataset requires numpy and torch; install Phase 2 training dependencies.")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self._split_name = split
        self._normalize = normalize
        self._bc_split = build_hc_bc_split(
            Path(manifest_path),
            sensor_tier=sensor_tier,
            successful_only=successful_only,
            seed_base=seed_base,
        )

        trajectories = self._bc_split.train_trajectories if split == "train" else self._bc_split.val_trajectories
        obs_rows, action_rows = _flatten_pairs(trajectories)
        obs_array = np.asarray(obs_rows, dtype=np.float32)
        action_array = np.asarray(action_rows, dtype=np.float32)
        self._obs_mean = np.asarray(self._bc_split.obs_mean, dtype=np.float32)
        self._obs_std = np.asarray(self._bc_split.obs_std, dtype=np.float32)
        self._obs_std = np.where(self._obs_std < 1e-8, 1.0, self._obs_std)
        if normalize:
            obs_array = (obs_array - self._obs_mean) / self._obs_std
        self._obs = torch.from_numpy(obs_array)
        self._actions = torch.from_numpy(action_array)

        self._cache_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / f"bc-dataset-{sensor_tier}-{self._bc_split.config_hash}.npz"
            train_obs_rows, train_action_rows = _flatten_pairs(self._bc_split.train_trajectories)
            val_obs_rows, val_action_rows = _flatten_pairs(self._bc_split.val_trajectories)
            np.savez_compressed(
                self._cache_path,
                train_obs=np.asarray(train_obs_rows, dtype=np.float32),
                train_actions=np.asarray(train_action_rows, dtype=np.float32),
                val_obs=np.asarray(val_obs_rows, dtype=np.float32),
                val_actions=np.asarray(val_action_rows, dtype=np.float32),
                obs_mean=self._obs_mean,
                obs_std=self._obs_std,
                manifest=json.dumps(bc_dataset_manifest_block(self._bc_split, cache_path=self._cache_path)),
            )

        print_bc_summary(self._bc_split)

    def __len__(self) -> int:
        return int(self._obs.shape[0])

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        return self._obs[idx], self._actions[idx]

    @property
    def obs_mean(self) -> "np.ndarray":
        return self._obs_mean

    @property
    def obs_std(self) -> "np.ndarray":
        return self._obs_std

    @property
    def trajectory_count(self) -> int:
        return self._bc_split.trajectory_count

    @property
    def manifest_block(self) -> dict[str, Any]:
        return bc_dataset_manifest_block(self._bc_split, cache_path=self._cache_path)
