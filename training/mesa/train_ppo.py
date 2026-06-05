"""Small PPO trainer for Phase 2 matched-capacity learned controllers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable

from training.mesa.evaluate_policy import evaluate_checkpoint
from training.mesa.js_bridge_env import BridgeClient, REPO_ROOT
from training.mesa.policy import (
    MesaActorCritic,
    count_parameters,
    policy_config_for_tier,
    policy_to_json_dict,
    seed_everything,
    write_policy_json,
)


DEFAULT_OUT = REPO_ROOT / "results" / "mesa" / "phase2-matched-capacity"

VARIANTS = {
    "signature_ppo_dense": {
        "family": "L-Signature",
        "reward_mode": "signature",
        "lambda": 0.0,
        "signature_shape": "integrated",
    },
    "signature_ppo_terminal": {
        "family": "L-Signature",
        "reward_mode": "signature",
        "lambda": 0.0,
        "signature_shape": "terminal",
    },
    "signature_ppo_threshold": {
        "family": "L-Signature",
        "reward_mode": "signature",
        "lambda": 0.0,
        "signature_shape": "threshold",
    },
    "reward_ppo_dense": {
        "family": "L-Reward",
        "reward_mode": "dense",
        "lambda": 1.0,
        "signature_shape": "integrated",
    },
    "reward_ppo_phase3": {
        "family": "L-Reward",
        "reward_mode": "phase3_dense_action_basin",
        "lambda": 1.0,
        "signature_shape": "integrated",
    },
    "mixed_ppo_lambda_0_5": {
        "family": "L-Mixed",
        "reward_mode": "mixed",
        "lambda": 0.5,
        "signature_shape": "integrated",
    },
    "mixed_ppo_phase3_lambda_0_1": {
        "family": "L-Mixed",
        "reward_mode": "mixed_phase3",
        "lambda": 0.1,
        "signature_shape": "integrated",
    },
    "mixed_ppo_phase3_lambda_0_3": {
        "family": "L-Mixed",
        "reward_mode": "mixed_phase3",
        "lambda": 0.3,
        "signature_shape": "integrated",
    },
    "mixed_ppo_phase3_lambda_0_5": {
        "family": "L-Mixed",
        "reward_mode": "mixed_phase3",
        "lambda": 0.5,
        "signature_shape": "integrated",
    },
    "mixed_ppo_phase3_lambda_0_7": {
        "family": "L-Mixed",
        "reward_mode": "mixed_phase3",
        "lambda": 0.7,
        "signature_shape": "integrated",
    },
    "mixed_ppo_phase3_lambda_0_8": {
        "family": "L-Mixed",
        "reward_mode": "mixed_phase3",
        "lambda": 0.8,
        "signature_shape": "integrated",
    },
    "mixed_ppo_phase3_lambda_0_9": {
        "family": "L-Mixed",
        "reward_mode": "mixed_phase3",
        "lambda": 0.9,
        "signature_shape": "integrated",
    },
    "curriculum_sig_then_reward": {
        "family": "L-Curriculum",
        "reward_mode": "phase3_dense_action_basin",
        "lambda": 1.0,
        "signature_shape": "integrated",
    },
    "curriculum_reward_then_sig": {
        "family": "L-Curriculum",
        "reward_mode": "signature",
        "lambda": 0.0,
        "signature_shape": "integrated",
    },
    "curriculum_reward_then_terminal_sig": {
        "family": "L-Curriculum",
        "reward_mode": "signature",
        "lambda": 0.0,
        "signature_shape": "terminal",
    },
}


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-8))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Small PPO Mesa controller.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--run-label", default="", help="optional slug suffix for budget/tuning variants")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tier", default="Small", choices=["Small", "Medium", "Large"])
    parser.add_argument("--sensor-tier", default="local-probe-field")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--batch-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument("--eval-seed-start", type=int, default=10_000)
    parser.add_argument("--eval-seeds", type=int, default=64)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training; evaluate --load-checkpoint at --eval-seed-start / --eval-seeds and write the eval summary, then exit. Requires --load-checkpoint.",
    )
    parser.add_argument("--success-floor", type=float, default=0.75)
    parser.add_argument("--false-basin-beta", type=float, default=None)
    parser.add_argument("--mixed-lambda", type=float, default=None, help="override lambda for mixed variants")
    parser.add_argument(
        "--reward-compose-form",
        choices=["canonical", "delta"],
        default="canonical",
        help=(
            "mixed-reward composition form; 'delta' uses "
            "S + lambda * (R - S), algebraically equal to canonical at scale 1"
        ),
    )
    parser.add_argument(
        "--reward-channel-scale",
        type=float,
        default=1.0,
        help="multiply the reward channel before mixed-reward composition",
    )
    parser.add_argument(
        "--signature-shape",
        choices=["terminal", "integrated", "threshold"],
        default=None,
        help="override signature objective shape; default comes from variant config",
    )
    parser.add_argument("--signature-threshold", type=float, default=0.5)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument(
        "--basin-channel",
        action="store_true",
        help="v4 Path-A verify-first: train with the basin observable (obs_dim +2; "
        "env basinChannel=true). NOT compatible with the legacy length-6 population.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def signature_reward(
    channels: dict[str, float],
    *,
    signature_shape: str,
    is_terminal_step: bool,
    signature_threshold: float,
) -> float:
    signature = float(channels["signature"])
    if signature_shape == "terminal":
        return signature if is_terminal_step else 0.0
    if signature_shape == "integrated":
        return signature
    if signature_shape == "threshold":
        return 1.0 if signature > signature_threshold else 0.0
    raise ValueError(f"unknown signature_shape: {signature_shape}")


def reward_from_channels(
    channels: dict[str, float],
    *,
    reward_mode: str,
    mixed_lambda: float,
    reward_compose_form: str,
    reward_channel_scale: float,
    signature_shape: str,
    is_terminal_step: bool,
    signature_threshold: float,
) -> float:
    signature = signature_reward(
        channels,
        signature_shape=signature_shape,
        is_terminal_step=is_terminal_step,
        signature_threshold=signature_threshold,
    )
    dense = float(channels["dense"])
    phase3 = float(channels.get("phase3_dense_action_basin", dense))
    if reward_mode == "signature":
        return signature
    if reward_mode == "dense":
        return dense
    if reward_mode == "phase3_dense_action_basin":
        return phase3
    if reward_mode == "mixed":
        scaled_reward = reward_channel_scale * dense
        if reward_compose_form == "canonical":
            return (1 - mixed_lambda) * signature + mixed_lambda * scaled_reward
        if reward_compose_form == "delta":
            return signature + mixed_lambda * (scaled_reward - signature)
        raise ValueError(f"unknown reward_compose_form: {reward_compose_form}")
    if reward_mode == "mixed_phase3":
        scaled_reward = reward_channel_scale * phase3
        if reward_compose_form == "canonical":
            return (1 - mixed_lambda) * signature + mixed_lambda * scaled_reward
        if reward_compose_form == "delta":
            return signature + mixed_lambda * (scaled_reward - signature)
        raise ValueError(f"unknown reward_compose_form: {reward_compose_form}")
    raise ValueError(f"unknown reward_mode: {reward_mode}")


def training_env_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {"horizon": 200}
    if args.false_basin_beta is not None:
        config["falseBasinBeta"] = args.false_basin_beta
    if getattr(args, "basin_channel", False):
        config["basinChannel"] = True
    return config


def status(message: str) -> None:
    print(f"mesa ppo status: {message}", flush=True)


def resolved_variant_config(args: argparse.Namespace) -> dict[str, Any]:
    variant_config = dict(VARIANTS[args.variant])
    reward_mode = str(variant_config["reward_mode"])
    if args.reward_channel_scale <= 0:
        raise ValueError("--reward-channel-scale must be positive")
    if args.mixed_lambda is not None:
        if not 0.0 < args.mixed_lambda < 1.0:
            raise ValueError("--mixed-lambda must satisfy 0 < lambda < 1")
        if reward_mode in {"mixed", "mixed_phase3"}:
            variant_config["lambda"] = float(args.mixed_lambda)
    if args.signature_shape is not None:
        variant_config["signature_shape"] = args.signature_shape
    return variant_config


def slug_for(variant: str, tier: str, seed: int, run_label: str = "") -> str:
    suffix = f"_{run_label}" if run_label else ""
    return f"{variant}_{tier.lower()}_seed_{seed}{suffix}"


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def effective_mixed_lambda(mixed_lambda: float, reward_channel_scale: float) -> float:
    """Return the equivalent unscaled lambda for positive reward rescaling."""
    denominator = (1.0 - mixed_lambda) + mixed_lambda * reward_channel_scale
    return (mixed_lambda * reward_channel_scale) / denominator


def load_pretrain_checkpoint(
    *,
    path: Path,
    model: MesaActorCritic,
    optimizer: torch.optim.Optimizer,
    obs_rms: RunningMeanStd,
    config: Any,
    device: torch.device,
    reset_optimizer: bool,
) -> dict[str, Any]:
    checkpoint_path = path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("policy_config")
    if checkpoint_config != asdict(config):
        raise ValueError(
            f"loaded checkpoint architecture mismatch: {checkpoint_config!r} != {asdict(config)!r}"
        )
    model.actor.load_state_dict(checkpoint["model_state_dict"])
    model.critic.load_state_dict(checkpoint["critic_state_dict"])
    model.log_std.data.copy_(torch.as_tensor(checkpoint["log_std"], dtype=torch.float32, device=device))

    obs_rms.mean = np.asarray(checkpoint["obs_mean"], dtype=np.float64)
    obs_std = np.asarray(checkpoint["obs_std"], dtype=np.float64)
    obs_rms.var = np.maximum(obs_std * obs_std, 1e-8)
    obs_rms.count = float(checkpoint.get("obs_rms_count", checkpoint.get("training", {}).get("env_steps", 1e-4)))
    obs_rms.count = max(obs_rms.count, 1e-4)

    optimizer_state_loaded = False
    if not reset_optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        optimizer_state_loaded = True

    return {
        "path": relative_path(checkpoint_path),
        "family": checkpoint.get("family"),
        "variant": checkpoint.get("variant"),
        "tier": checkpoint.get("tier"),
        "seed": checkpoint.get("seed"),
        "run_label": checkpoint.get("run_label"),
        "env_steps": checkpoint.get("training", {}).get("env_steps"),
        "optimizer_state_loaded": optimizer_state_loaded,
        "reset_optimizer": bool(reset_optimizer),
    }


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    variance = torch.var(y_true)
    if float(variance.item()) < 1e-8:
        return 0.0
    return float((1 - torch.var(y_true - y_pred) / variance).item())


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_training_run(out_dir: Path, row: dict[str, Any]) -> None:
    path = out_dir / "training-runs.csv"
    fieldnames = [
        "family",
        "variant",
        "tier",
        "seed",
        "run_label",
        "sensor_tier",
        "algorithm",
        "parameter_count",
        "env_steps",
        "checkpoint_path",
        "policy_json_path",
        "mixed_lambda",
        "effective_mixed_lambda",
        "reward_compose_form",
        "reward_channel_scale",
        "signature_shape",
        "load_checkpoint_path",
        "reset_optimizer",
        "success_rate",
        "mean_terminal_alignment",
    ]
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [
            old for old in rows
            if not (
                old.get("variant") == row["variant"]
                and old.get("seed") == str(row["seed"])
                and old.get("run_label", "") == str(row.get("run_label", ""))
            )
        ]
        rows = [{field: old.get(field, "") for field in fieldnames} for old in rows]
    rows.append({field: row.get(field, "") for field in fieldnames})
    write_rows(path, rows, fieldnames)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    variant_config = resolved_variant_config(args)
    config = policy_config_for_tier(args.tier, action_scale=1.0)
    if getattr(args, "basin_channel", False):
        # v4 Path-A verify-first: the basin observable adds 2 features to the obs.
        config = replace(config, obs_dim=config.obs_dim + 2)
    model = MesaActorCritic(config, log_std_init=args.log_std_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    obs_rms = RunningMeanStd((config.obs_dim,))
    pretrain_metadata = None
    if args.load_checkpoint is not None:
        pretrain_metadata = load_pretrain_checkpoint(
            path=args.load_checkpoint,
            model=model,
            optimizer=optimizer,
            obs_rms=obs_rms,
            config=config,
            device=device,
            reset_optimizer=args.reset_optimizer,
        )
        status(
            "loaded_checkpoint "
            f"path={pretrain_metadata['path']} "
            f"optimizer_state_loaded={pretrain_metadata['optimizer_state_loaded']}"
        )

    if args.eval_only:
        if args.load_checkpoint is None:
            raise SystemExit("--eval-only requires --load-checkpoint")
        out_dir = args.out.resolve()
        eval_slug = slug_for(args.variant, args.tier, args.seed, args.run_label)
        logs_dir = out_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        eval_rows_path = logs_dir / f"{eval_slug}_evaluation.csv"
        eval_summary_path = logs_dir / f"{eval_slug}_evaluation_summary.json"
        env_config = training_env_config(args)
        status(
            f"eval_only seeds={args.eval_seeds} seed_start={args.eval_seed_start} "
            f"checkpoint={args.load_checkpoint}"
        )
        eval_rows, eval_summary = evaluate_checkpoint(
            Path(args.load_checkpoint),
            sensor_tier=args.sensor_tier,
            seed_start=args.eval_seed_start,
            seeds=args.eval_seeds,
            horizon=200,
            env_config=env_config,
        )
        status(f"writing_evaluation path={eval_summary_path.relative_to(REPO_ROOT)}")
        write_rows(
            eval_rows_path,
            eval_rows,
            [
                "seed",
                "terminalOutcome",
                "steps",
                "terminalAlignment",
                "terminalDistance",
                "pathEfficiency",
                "saturationCount",
            ],
        )
        eval_summary_path.write_text(
            json.dumps(eval_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        success_rate = float(eval_summary.get("success_rate", 0.0))
        if success_rate < args.success_floor:
            status(
                f"{args.variant} below success floor: {success_rate:.3f} < {args.success_floor:.3f}"
            )
        return {
            "eval_only": True,
            "evaluation_summary_path": str(
                eval_summary_path.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "checkpoint": str(args.load_checkpoint),
            "eval_seed_start": int(args.eval_seed_start),
            "eval_seeds": int(args.eval_seeds),
        }

    out_dir = args.out.resolve()
    slug = slug_for(args.variant, args.tier, args.seed, args.run_label)
    checkpoints_dir = out_dir / "checkpoints"
    policies_dir = out_dir / "policies"
    logs_dir = out_dir / "logs"
    manifests_dir = out_dir / "manifests"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / f"{slug}.pt"
    policy_json_path = policies_dir / f"{slug}.policy.json"
    history_path = logs_dir / f"{slug}_history.csv"
    eval_rows_path = logs_dir / f"{slug}_evaluation.csv"
    eval_summary_path = logs_dir / f"{slug}_evaluation_summary.json"
    manifest_path = manifests_dir / f"{slug}.json"

    history: list[dict[str, Any]] = []
    global_step = 0
    env_config = training_env_config(args)

    show_progress = args.progress or sys.stderr.isatty()
    status(
        f"start variant={args.variant} tier={args.tier} updates={args.updates} "
        f"batch_envs={args.batch_envs} rollout_length={args.rollout_length} "
        f"lambda={variant_config['lambda']} signature_shape={variant_config['signature_shape']} "
        f"reward_compose_form={args.reward_compose_form} "
        f"reward_channel_scale={args.reward_channel_scale} "
        f"run_label={args.run_label or '-'}"
    )
    with BridgeClient() as client:
        made = client.request(
            {
                "cmd": "make_batch",
                "batch_id": slug,
                "count": args.batch_envs,
                "seed_start": args.seed_start,
                "sensor_tier": args.sensor_tier,
                "env_config": env_config,
            }
        )
        raw_obs = np.asarray(made["obs"], dtype=np.float32)
        obs_rms.update(raw_obs)

        for update in tqdm(range(1, args.updates + 1), desc=f"ppo {args.variant}", unit="update", disable=not show_progress, leave=False):
            obs_buf = torch.zeros((args.rollout_length, args.batch_envs, config.obs_dim), dtype=torch.float32, device=device)
            action_buf = torch.zeros((args.rollout_length, args.batch_envs, config.act_dim), dtype=torch.float32, device=device)
            logprob_buf = torch.zeros((args.rollout_length, args.batch_envs), dtype=torch.float32, device=device)
            reward_buf = torch.zeros((args.rollout_length, args.batch_envs), dtype=torch.float32, device=device)
            done_buf = torch.zeros((args.rollout_length, args.batch_envs), dtype=torch.float32, device=device)
            value_buf = torch.zeros((args.rollout_length, args.batch_envs), dtype=torch.float32, device=device)

            for step in range(args.rollout_length):
                norm_obs = obs_rms.normalize(raw_obs)
                obs_tensor = torch.from_numpy(norm_obs).to(device)
                with torch.no_grad():
                    env_action, raw_action, log_prob, value, _entropy = model.act(obs_tensor)
                action_list = env_action.cpu().numpy().tolist()
                response = client.request(
                    {
                        "cmd": "step_batch",
                        "batch_id": slug,
                        "actions": action_list,
                        "auto_reset_done": True,
                    }
                )

                rewards = [
                    reward_from_channels(
                        channels,
                        reward_mode=variant_config["reward_mode"],
                        mixed_lambda=float(variant_config["lambda"]),
                        reward_compose_form=args.reward_compose_form,
                        reward_channel_scale=args.reward_channel_scale,
                        signature_shape=str(variant_config["signature_shape"]),
                        is_terminal_step=bool(done),
                        signature_threshold=args.signature_threshold,
                    )
                    for channels, done in zip(response["reward_channels"], response["done"], strict=True)
                ]
                obs_buf[step] = obs_tensor
                action_buf[step] = raw_action
                logprob_buf[step] = log_prob
                reward_buf[step] = torch.tensor(rewards, dtype=torch.float32, device=device)
                done_buf[step] = torch.tensor(response["done"], dtype=torch.float32, device=device)
                value_buf[step] = value

                raw_obs = np.asarray(response["obs"], dtype=np.float32)
                obs_rms.update(raw_obs)
                global_step += args.batch_envs

            with torch.no_grad():
                next_obs = torch.from_numpy(obs_rms.normalize(raw_obs)).to(device)
                next_value = model.critic(next_obs)
                advantages = torch.zeros_like(reward_buf, device=device)
                last_gae = torch.zeros(args.batch_envs, dtype=torch.float32, device=device)
                for step in reversed(range(args.rollout_length)):
                    if step == args.rollout_length - 1:
                        next_nonterminal = 1.0 - done_buf[step]
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - done_buf[step]
                        next_values = value_buf[step + 1]
                    delta = reward_buf[step] + args.gamma * next_values * next_nonterminal - value_buf[step]
                    last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                    advantages[step] = last_gae
                returns = advantages + value_buf

            flat_obs = obs_buf.reshape((-1, config.obs_dim))
            flat_actions = action_buf.reshape((-1, config.act_dim))
            flat_old_logprob = logprob_buf.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = value_buf.reshape(-1)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            batch_size = args.rollout_length * args.batch_envs
            indices = torch.randperm(batch_size, device=device)

            policy_losses = []
            value_losses = []
            entropy_losses = []
            approx_kls = []
            clip_fracs = []

            for _epoch in range(args.epochs):
                for start in range(0, batch_size, args.minibatch_size):
                    mb_idx = indices[start:start + args.minibatch_size]
                    new_logprob, entropy, new_value = model.evaluate_actions(flat_obs[mb_idx], flat_actions[mb_idx])
                    logratio = new_logprob - flat_old_logprob[mb_idx]
                    ratio = logratio.exp()
                    mb_adv = flat_advantages[mb_idx]
                    unclipped = -mb_adv * ratio
                    clipped = -mb_adv * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                    policy_loss = torch.max(unclipped, clipped).mean()
                    value_loss = 0.5 * (new_value - flat_returns[mb_idx]).pow(2).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = ((ratio - 1.0).abs() > args.clip_range).float().mean()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropy_losses.append(float(entropy_loss.item()))
                    approx_kls.append(float(approx_kl.item()))
                    clip_fracs.append(float(clip_frac.item()))

            mean_reward = float(reward_buf.mean().item())
            episode_done_rate = float(done_buf.mean().item())
            ev = explained_variance(flat_values.detach(), flat_returns.detach())
            history.append({
                "update": update,
                "env_steps": global_step,
                "mean_reward": mean_reward,
                "episode_done_rate": episode_done_rate,
                "policy_loss": mean(policy_losses),
                "value_loss": mean(value_losses),
                "entropy": mean(entropy_losses),
                "approx_kl": mean(approx_kls),
                "clip_frac": mean(clip_fracs),
                "explained_variance": ev,
                "log_std": float(model.log_std.mean().detach().cpu().item()),
            })

        status(f"training_loop_complete env_steps={global_step}")

    status("bridge_closed")

    checkpoint = {
        "family": variant_config["family"],
        "variant": args.variant,
        "tier": args.tier,
        "seed": args.seed,
        "run_label": args.run_label,
        "sensor_tier": args.sensor_tier,
        "algorithm": "ppo",
        "policy_config": asdict(config),
        "model_state_dict": model.actor.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "log_std": model.log_std.detach().cpu().numpy().tolist(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_mean": obs_rms.mean.astype(np.float32).tolist(),
        "obs_std": obs_rms.std.astype(np.float32).tolist(),
        "obs_rms_count": obs_rms.count,
        "training": {
            "updates": args.updates,
            "batch_envs": args.batch_envs,
            "rollout_length": args.rollout_length,
            "env_steps": global_step,
            "minibatch_size": args.minibatch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "parameter_count": count_parameters(model.actor),
            "actor_critic_parameter_count": count_parameters(model),
            "device": str(device),
            "load_checkpoint": pretrain_metadata,
        },
        "reward": {
            "mode": variant_config["reward_mode"],
            "lambda": variant_config["lambda"],
            "effective_mixed_lambda": (
                effective_mixed_lambda(float(variant_config["lambda"]), args.reward_channel_scale)
                if variant_config["reward_mode"] in {"mixed", "mixed_phase3"}
                else variant_config["lambda"]
            ),
            "compose_form": args.reward_compose_form,
            "reward_channel_scale": args.reward_channel_scale,
            "signature_shape": variant_config["signature_shape"],
            "signature_threshold": args.signature_threshold,
            "env_config": env_config,
        },
    }
    status(f"writing_checkpoint path={checkpoint_path.relative_to(REPO_ROOT)}")
    torch.save(checkpoint, checkpoint_path)

    policy_payload = policy_to_json_dict(
        model.actor.cpu(),
        family=str(variant_config["family"]),
        variant=args.variant,
        obs_mean=np.asarray(checkpoint["obs_mean"], dtype=np.float32),
        obs_std=np.asarray(checkpoint["obs_std"], dtype=np.float32),
        metadata={
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "algorithm": "ppo",
            "env_steps": global_step,
            "reward_mode": variant_config["reward_mode"],
            "lambda": variant_config["lambda"],
            "effective_mixed_lambda": checkpoint["reward"]["effective_mixed_lambda"],
            "reward_compose_form": args.reward_compose_form,
            "reward_channel_scale": args.reward_channel_scale,
            "signature_shape": variant_config["signature_shape"],
            "signature_threshold": args.signature_threshold,
            "env_config": env_config,
            "load_checkpoint": pretrain_metadata,
        },
    )
    status(f"writing_policy_json path={policy_json_path.relative_to(REPO_ROOT)}")
    write_policy_json(policy_json_path, policy_payload)

    status(f"writing_history path={history_path.relative_to(REPO_ROOT)}")
    write_rows(
        history_path,
        history,
        [
            "update",
            "env_steps",
            "mean_reward",
            "episode_done_rate",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
            "explained_variance",
            "log_std",
        ],
    )

    status(f"evaluating_checkpoint seeds={args.eval_seeds} seed_start={args.eval_seed_start}")
    eval_rows, eval_summary = evaluate_checkpoint(
        checkpoint_path,
        sensor_tier=args.sensor_tier,
        seed_start=args.eval_seed_start,
        seeds=args.eval_seeds,
        horizon=200,
        env_config=env_config,
    )
    status(f"writing_evaluation path={eval_summary_path.relative_to(REPO_ROOT)}")
    write_rows(
        eval_rows_path,
        eval_rows,
        [
            "seed",
            "terminalOutcome",
            "steps",
            "terminalAlignment",
            "terminalDistance",
            "pathEfficiency",
            "saturationCount",
        ],
    )
    eval_summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_manifest = {
        "phase": "phase2-matched-capacity",
        "family": variant_config["family"],
        "variant": args.variant,
        "tier": args.tier,
        "seed": args.seed,
        "run_label": args.run_label,
        "sensor_tier": args.sensor_tier,
        "algorithm": "ppo",
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "policy_json_path": str(policy_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "history_path": str(history_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "evaluation_rows_path": str(eval_rows_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "evaluation_summary_path": str(eval_summary_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "training": checkpoint["training"],
        "reward": checkpoint["reward"],
        "load_checkpoint": pretrain_metadata,
        "evaluation": eval_summary,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    append_training_run(out_dir, {
        "family": variant_config["family"],
        "variant": args.variant,
        "tier": args.tier,
        "seed": str(args.seed),
        "run_label": args.run_label,
        "sensor_tier": args.sensor_tier,
        "algorithm": "ppo",
        "parameter_count": checkpoint["training"]["parameter_count"],
        "env_steps": global_step,
        "checkpoint_path": run_manifest["checkpoint_path"],
        "policy_json_path": run_manifest["policy_json_path"],
        "mixed_lambda": variant_config["lambda"],
        "effective_mixed_lambda": checkpoint["reward"]["effective_mixed_lambda"],
        "reward_compose_form": args.reward_compose_form,
        "reward_channel_scale": args.reward_channel_scale,
        "signature_shape": variant_config["signature_shape"],
        "load_checkpoint_path": pretrain_metadata["path"] if pretrain_metadata else "",
        "reset_optimizer": str(bool(args.reset_optimizer)),
        "success_rate": eval_summary["success_rate"],
        "mean_terminal_alignment": eval_summary["mean_terminal_alignment"],
    })

    summary_path = out_dir / "ppo-small-summary.json"
    existing: dict[str, Any] = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_key = f"{args.variant}:{args.run_label}" if args.run_label else args.variant
    existing[summary_key] = run_manifest
    summary_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if eval_summary["success_rate"] < args.success_floor:
        raise SystemExit(
            f"{args.variant} below success floor: {eval_summary['success_rate']:.3f} < {args.success_floor:.3f}"
        )
    return run_manifest


def main() -> int:
    args = parse_args()
    summary = run_training(args)
    if summary.get("eval_only"):
        eval_summary_path = Path(summary["evaluation_summary_path"])
        if not eval_summary_path.is_absolute():
            eval_summary_path = REPO_ROOT / eval_summary_path
        eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        print(
            "mesa ppo eval-only: "
            f"variant={args.variant} tier={args.tier} "
            f"seed_start={summary['eval_seed_start']} seeds={summary['eval_seeds']} "
            f"success={eval_summary.get('success_count', 0)}/{eval_summary.get('seeds', 0)} "
            f"({100 * eval_summary.get('success_rate', 0):.1f}%) "
            f"mean_S_T={eval_summary.get('mean_terminal_alignment', 0):.4f} "
            f"checkpoint={summary['checkpoint']}"
        )
        return 0
    evaluation = summary["evaluation"]
    print(
        "mesa ppo train: "
        f"variant={summary['variant']} tier={summary['tier']} "
        f"env_steps={summary['training']['env_steps']} "
        f"success={evaluation['success_count']}/{evaluation['seeds']} "
        f"({100 * evaluation['success_rate']:.1f}%) "
        f"mean_S_T={evaluation['mean_terminal_alignment']:.4f} "
        f"checkpoint={summary['checkpoint_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
