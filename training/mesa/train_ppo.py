"""Small PPO trainer for Phase 2 matched-capacity learned controllers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict
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
    },
    "reward_ppo_dense": {
        "family": "L-Reward",
        "reward_mode": "dense",
        "lambda": 1.0,
    },
    "mixed_ppo_lambda_0_5": {
        "family": "L-Mixed",
        "reward_mode": "mixed",
        "lambda": 0.5,
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
    parser.add_argument("--tier", default="Small", choices=["Small"])
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
    parser.add_argument("--success-floor", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def reward_from_channels(channels: dict[str, float], *, reward_mode: str, mixed_lambda: float) -> float:
    signature = float(channels["signature"])
    dense = float(channels["dense"])
    if reward_mode == "signature":
        return signature
    if reward_mode == "dense":
        return dense
    if reward_mode == "mixed":
        return (1 - mixed_lambda) * signature + mixed_lambda * dense
    raise ValueError(f"unknown reward_mode: {reward_mode}")


def slug_for(variant: str, tier: str, seed: int, run_label: str = "") -> str:
    suffix = f"_{run_label}" if run_label else ""
    return f"{variant}_{tier.lower()}_seed_{seed}{suffix}"


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
    variant_config = VARIANTS[args.variant]
    config = policy_config_for_tier(args.tier, action_scale=1.0)
    model = MesaActorCritic(config, log_std_init=args.log_std_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    obs_rms = RunningMeanStd((config.obs_dim,))

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

    show_progress = args.progress or sys.stderr.isatty()
    with BridgeClient() as client:
        made = client.request(
            {
                "cmd": "make_batch",
                "batch_id": slug,
                "count": args.batch_envs,
                "seed_start": args.seed_start,
                "sensor_tier": args.sensor_tier,
                "env_config": {"horizon": 200},
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
                    )
                    for channels in response["reward_channels"]
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
        "obs_mean": obs_rms.mean.astype(np.float32).tolist(),
        "obs_std": obs_rms.std.astype(np.float32).tolist(),
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
        },
        "reward": {
            "mode": variant_config["reward_mode"],
            "lambda": variant_config["lambda"],
        },
    }
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
        },
    )
    write_policy_json(policy_json_path, policy_payload)

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

    eval_rows, eval_summary = evaluate_checkpoint(
        checkpoint_path,
        sensor_tier=args.sensor_tier,
        seed_start=args.eval_seed_start,
        seeds=args.eval_seeds,
        horizon=200,
    )
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
