"""H4.0-c cheap central-RNN learned-headroom/OOD-gap admission.

This is deliberately admission-scale: one central recurrent monolith, no
distributed controller yet. It checks whether the Distributed Relay Grid has
learnable signal, does not saturate at cheap budget, and leaves a registered
in-distribution -> held-out OOD generalization gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from training.mesa.h4_distributed_world_model_task import (
    DistributedRelayEnv,
    H4_RELAY_DEFAULTS,
    H4_RELAY_OOD_CELL_DEFS,
    H4_RELAY_OOD_CELLS,
    H4_RELAY_TRAIN_CELL_DEFS,
    H4_RELAY_TRAIN_CELLS,
    roll_episode,
    summarize_metrics,
)


ACTION_VALUES = [-1, 0, 1]


@dataclass
class TrainEpisode:
    obs: list[list[float]]
    actions: list[int]
    old_log_probs: list[float]
    rewards: list[float]
    decision_mask: list[bool]
    metrics: dict[str, Any]
    cell: str
    seed: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="h4_0c_learned_headroom")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--updates", type=int, default=64)
    ap.add_argument("--rollouts-per-update", type=int, default=32)
    ap.add_argument("--train-seed-start", type=int, default=20000)
    ap.add_argument("--eval-seeds", type=int, default=32)
    ap.add_argument("--eval-seed-start", type=int, default=30000)
    ap.add_argument("--ppo-seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--checkpoint-every", type=int, default=16)
    return ap.parse_args()


def obs_vector(obs: dict[str, Any], cfg: dict[str, Any]) -> list[float]:
    k = int(cfg["K"])
    observe_ticks = float(cfg["observeTicks"])
    horizon = float(cfg["horizon"])
    return [
        float(obs["phase"]) / k,
        float(obs["tick_in_gate"]) / observe_ticks,
        float(obs["t"]) / horizon,
        *[float(v) for v in obs["field_state"]],
        *[float(v) for v in obs["local_obs"]],
        *[float(v) for v in obs["local_mask"]],
        *[float(v) / observe_ticks for v in obs["local_age"]],
        *[float(v) for v in obs["reward_cue"]],
    ]


class CentralRnnPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, len(ACTION_VALUES))
        self.critic = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.actor.bias[:] = torch.tensor([0.0, -0.5, 0.0])

    def initial_state(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_dim)

    def forward_sequence(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, self.hidden_dim, device=obs.device)
        logits_rows: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        h_rows: list[torch.Tensor] = []
        for row in obs:
            x = torch.tanh(self.in_proj(row.unsqueeze(0)))
            h = self.gru(x, h)
            logits_rows.append(self.actor(h).squeeze(0))
            values.append(self.critic(h).squeeze(0).squeeze(-1))
            h_rows.append(h.squeeze(0))
        return torch.stack(logits_rows), torch.stack(values), torch.stack(h_rows)


def terminal_reward(metrics: dict[str, Any]) -> float:
    return float(metrics["competence"] - metrics["basin"] + 0.2 * metrics["gate_completion"])


def returns_from_rewards(rewards: list[float], gamma: float) -> list[float]:
    out: list[float] = []
    acc = 0.0
    for reward in reversed(rewards):
        acc = float(reward) + gamma * acc
        out.append(acc)
    return list(reversed(out))


def collect_episode(policy: CentralRnnPolicy, cell: str, seed: int, rng: np.random.Generator) -> TrainEpisode:
    env = DistributedRelayEnv()
    obs_obj = env.reset(seed, H4_RELAY_TRAIN_CELL_DEFS[cell])
    obs_rows: list[list[float]] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    rewards: list[float] = []
    decision_mask: list[bool] = []
    done = False
    h = policy.initial_state()
    while not done:
        vec = obs_vector(obs_obj, env.cfg)
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z = torch.tanh(policy.in_proj(x))
            h = policy.gru(z, h)
            logits = policy.actor(h).squeeze(0)
            dist = Categorical(logits=logits)
            action_idx = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor(action_idx)).item())
        decision_tick = env.is_decision_tick()
        before_phase = env.phase
        decision_hold = decision_tick and ACTION_VALUES[action_idx] == 0
        step = env.step(ACTION_VALUES[action_idx])
        reward = 0.0
        if decision_hold:
            reward -= 0.20
        if env.phase > before_phase:
            reward += 0.20
        if step.done:
            reward += terminal_reward(env.metrics())
            if env.metrics()["outcome"] == "timeout":
                reward -= 0.25
        obs_rows.append(vec)
        actions.append(action_idx)
        old_log_probs.append(log_prob)
        rewards.append(reward)
        decision_mask.append(decision_tick)
        obs_obj = step.obs
        done = bool(step.done)
        # Break exact action ties in early training by using the RNG in collection.
        if rng.random() < 0:
            actions[-1] = int(rng.integers(0, len(ACTION_VALUES)))
    return TrainEpisode(
        obs=obs_rows,
        actions=actions,
        old_log_probs=old_log_probs,
        rewards=rewards,
        decision_mask=decision_mask,
        metrics=env.metrics(),
        cell=cell,
        seed=seed,
    )


def ppo_update(
    policy: CentralRnnPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list[TrainEpisode],
    *,
    gamma: float,
    clip_range: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    epochs: int,
) -> dict[str, float]:
    stats: dict[str, list[float]] = {"loss": [], "policy": [], "value": [], "entropy": []}
    for _epoch in range(epochs):
        optimizer.zero_grad()
        policy_losses = []
        value_losses = []
        entropies = []
        for ep in episodes:
            obs = torch.tensor(ep.obs, dtype=torch.float32)
            actions = torch.tensor(ep.actions, dtype=torch.long)
            old_lp = torch.tensor(ep.old_log_probs, dtype=torch.float32)
            returns = torch.tensor(returns_from_rewards(ep.rewards, gamma), dtype=torch.float32)
            mask = torch.tensor(ep.decision_mask, dtype=torch.bool)
            logits, values, _h = policy.forward_sequence(obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            if not bool(mask.any()):
                continue
            logits = logits[mask]
            values = values[mask]
            returns = returns[mask]
            actions = actions[mask]
            old_lp = old_lp[mask]
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            advantages = returns - values.detach()
            if len(advantages) > 1 and float(advantages.std(unbiased=False)) > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            ratio = torch.exp(log_probs - old_lp)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
        loss_policy = torch.stack(policy_losses).mean()
        loss_value = torch.stack(value_losses).mean()
        loss_entropy = torch.stack(entropies).mean()
        loss = loss_policy + value_coef * loss_value - entropy_coef * loss_entropy
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        stats["loss"].append(float(loss.item()))
        stats["policy"].append(float(loss_policy.item()))
        stats["value"].append(float(loss_value.item()))
        stats["entropy"].append(float(loss_entropy.item()))
    return {key: float(np.mean(values)) for key, values in stats.items()}


def eval_policy(
    policy: CentralRnnPolicy,
    *,
    split: str,
    cell_defs: dict[str, dict[str, Any]],
    cells: list[str],
    eval_seeds: int,
    eval_seed_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy.eval()
    with torch.no_grad():
        for cell in cells:
            for offset in range(eval_seeds):
                seed = eval_seed_start + offset
                env = DistributedRelayEnv()
                obs_obj = env.reset(seed, cell_defs[cell])
                h = policy.initial_state()
                done = False
                while not done:
                    vec = obs_vector(obs_obj, env.cfg)
                    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                    z = torch.tanh(policy.in_proj(x))
                    h = policy.gru(z, h)
                    logits = policy.actor(h).squeeze(0)
                    action_idx = int(torch.argmax(logits).item())
                    step = env.step(ACTION_VALUES[action_idx])
                    obs_obj = step.obs
                    done = bool(step.done)
                metrics = env.metrics()
                rows.append({"split": split, "cell": cell, "seed": seed, "controller": "M-Central-RNN-H4-cheap", **metrics})
    summary = summarize_metrics(rows)
    by_cell = {}
    for cell in cells:
        by_cell[cell] = summarize_metrics([row for row in rows if row["cell"] == cell])
    return rows, {"aggregate": summary, "by_cell": by_cell}


def fixed_eval(split: str, cell_defs: dict[str, dict[str, Any]], cells: list[str], eval_seeds: int, eval_seed_start: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in ["Oracle-H4", "Field-H4"]:
        rows = []
        for cell in cells:
            for offset in range(eval_seeds):
                seed = eval_seed_start + offset
                rows.append({"split": split, "cell": cell, "seed": seed, "controller": label, **roll_episode(DistributedRelayEnv(), label, seed, cell_defs[cell])})
        out[label] = summarize_metrics(rows)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "cell", "seed", "controller", "competence", "basin", "resistance", "gate_completion", "J", "outcome", "fail_gate", "steps"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def select_branch(id_summary: dict[str, Any], ood_summary: dict[str, Any], fixed_id: dict[str, Any], fixed_ood: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    m_id = id_summary["aggregate"]
    m_ood = ood_summary["aggregate"]
    field_id = fixed_id["Field-H4"]
    oracle_ood = fixed_ood["Oracle-H4"]
    learned_headroom = (m_id["C"] - field_id["C"] >= 0.05) or (m_id["J"] - field_id["J"] >= 0.05)
    monolith_saturates = m_id["C"] >= 0.95 and m_id["B"] <= field_id["B"] + 0.03
    oracle_ood_ok = oracle_ood["C"] >= 0.85 and oracle_ood["B"] <= 0.05
    ood_gap = m_id["J"] - m_ood["J"]
    gates = {
        "gate7_learned_headroom": bool(learned_headroom),
        "gate7_not_oracle_saturated": not bool(monolith_saturates),
        "gate8_oracle_ood_solvable": bool(oracle_ood_ok),
        "gate8_ood_generalization_gap": bool(ood_gap >= 0.10),
    }
    if not oracle_ood_ok or not learned_headroom:
        branch = "H4_0_TASK_VOID"
    elif monolith_saturates:
        branch = "H4_0_MONOLITH_HEADROOM_VOID"
    elif ood_gap < 0.10:
        branch = "H4_0_NO_OOD_GAP_VOID"
    else:
        branch = "H4_0_ADMITTED"
    return branch, {"gates": gates, "ood_gap_J": float(ood_gap), "monolith_saturates": bool(monolith_saturates)}


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.ppo_seed)
    np.random.seed(args.ppo_seed)
    rng = np.random.default_rng(args.ppo_seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []
    input_dim = len(obs_vector(DistributedRelayEnv().reset(1, H4_RELAY_TRAIN_CELL_DEFS[H4_RELAY_TRAIN_CELLS[0]]), H4_RELAY_DEFAULTS))
    policy = CentralRnnPolicy(input_dim=input_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    started = time.time()
    env_steps = 0
    for update in range(1, args.updates + 1):
        episodes: list[TrainEpisode] = []
        for rollout in range(args.rollouts_per_update):
            cell = H4_RELAY_TRAIN_CELLS[(update * args.rollouts_per_update + rollout) % len(H4_RELAY_TRAIN_CELLS)]
            seed = args.train_seed_start + (update - 1) * args.rollouts_per_update + rollout
            ep = collect_episode(policy, cell, seed, rng)
            episodes.append(ep)
            env_steps += len(ep.obs)
        stats = ppo_update(
            policy,
            optimizer,
            episodes,
            gamma=args.gamma,
            clip_range=args.clip_range,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
        )
        train_summary = summarize_metrics([ep.metrics for ep in episodes])
        row = {
            "update": update,
            "env_steps": env_steps,
            "return": float(np.mean([sum(ep.rewards) for ep in episodes])),
            **{f"train_{k}": v for k, v in train_summary.items()},
            **stats,
        }
        logs.append(row)
        if update % max(1, args.checkpoint_every) == 0 or update == args.updates:
            torch.save({"model": policy.state_dict(), "input_dim": input_dim, "hidden_dim": args.hidden_dim}, models_dir / f"m_central_rnn_h4_update_{update}.pt")
        if update == 1 or update % 16 == 0 or update == args.updates:
            print(
                f"{args.phase} update={update}/{args.updates} "
                f"C={train_summary['C']:.3f} B={train_summary['B']:.3f} J={train_summary['J']:.3f} "
                f"steps={env_steps}"
            )
    elapsed = time.time() - started
    torch.save({"model": policy.state_dict(), "input_dim": input_dim, "hidden_dim": args.hidden_dim}, models_dir / "m_central_rnn_h4.pt")

    id_rows, id_summary = eval_policy(
        policy,
        split="in_distribution",
        cell_defs=H4_RELAY_TRAIN_CELL_DEFS,
        cells=H4_RELAY_TRAIN_CELLS,
        eval_seeds=args.eval_seeds,
        eval_seed_start=args.eval_seed_start,
    )
    ood_rows, ood_summary = eval_policy(
        policy,
        split="heldout_ood",
        cell_defs=H4_RELAY_OOD_CELL_DEFS,
        cells=H4_RELAY_OOD_CELLS,
        eval_seeds=args.eval_seeds,
        eval_seed_start=args.eval_seed_start,
    )
    fixed_id = fixed_eval("in_distribution", H4_RELAY_TRAIN_CELL_DEFS, H4_RELAY_TRAIN_CELLS, args.eval_seeds, args.eval_seed_start)
    fixed_ood = fixed_eval("heldout_ood", H4_RELAY_OOD_CELL_DEFS, H4_RELAY_OOD_CELLS, args.eval_seeds, args.eval_seed_start)
    branch, decision = select_branch(id_summary, ood_summary, fixed_id, fixed_ood)

    train_log_path = out / "train_log.csv"
    with train_log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)
    write_csv(out / "eval_trials.csv", id_rows + ood_rows)
    report = {
        "phase": args.phase,
        "status": "complete",
        "branch": branch,
        "updates": args.updates,
        "rollouts_per_update": args.rollouts_per_update,
        "env_steps": env_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": env_steps / max(elapsed, 1e-9),
        "ppo_seed": args.ppo_seed,
        "train_cells": H4_RELAY_TRAIN_CELLS,
        "ood_cells": H4_RELAY_OOD_CELLS,
        "train_cell_defs": H4_RELAY_TRAIN_CELL_DEFS,
        "ood_cell_defs": H4_RELAY_OOD_CELL_DEFS,
        "eval": {
            "in_distribution": id_summary,
            "heldout_ood": ood_summary,
            "fixed_in_distribution": fixed_id,
            "fixed_heldout_ood": fixed_ood,
        },
        "decision": decision,
        "model": {
            "type": "M-Central-RNN-H4-cheap",
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "action_values": ACTION_VALUES,
            "params": sum(p.numel() for p in policy.parameters()),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.phase} H4.0-c done. updates={args.updates} env_steps={env_steps} "
        f"elapsed={elapsed:.2f}s steps/s={env_steps / max(elapsed, 1e-9):.2f}"
    )
    print(
        f"  ID M: C={id_summary['aggregate']['C']:.4f} B={id_summary['aggregate']['B']:.4f} J={id_summary['aggregate']['J']:.4f}"
    )
    print(
        f"  OOD M: C={ood_summary['aggregate']['C']:.4f} B={ood_summary['aggregate']['B']:.4f} J={ood_summary['aggregate']['J']:.4f}"
    )
    print(f"  gates: {json.dumps(decision['gates'])} ood_gap_J={decision['ood_gap_J']:.4f} -> {branch}")
    return 0 if branch == "H4_0_ADMITTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
