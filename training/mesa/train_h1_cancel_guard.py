"""H1.2e PPO trainer: cancelling-guard council vs matched RL monolith.

Spec: docs/mesa/H1_2E_CANCELLING_GUARD_SPEC.md. Adapts the H1.2d RL trainer
(train_h1_rl_arbiter.py) with exactly one architectural change: the guard's
proposal is no longer the passive hold ``[0, 0]`` but an anti-reward countervote

    a_guard = -c_guard * a_reward

where ``c_guard in [0, cancel_cap]`` comes from a SEPARATE, zero-initialized
scalar cancellation head sampled as a 4th policy dimension (so PPO trains it).
``cancel-init zero`` => the head's output layer starts at weight 0 / bias -5, so
``c_guard ~= 0`` and H1.2e begins exactly as H1.2d plus a small trainable
countervote. The reward/bull head is still seated and still capped at 0.50; the
guard is merely allowed to cancel its residual.

The effective reward coefficient is ``w_reward - w_guard * c_guard``, which can
reach zero or go negative only through the separately capped, audited guard role.

Outputs stay in the mesa-coordinator-json-v1 family consumed by
scripts/mesa-h1-pantheon-eval.mjs; the cancellation head is exported inside
p_guard_cancel.json under a ``cancel_head`` block.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from training.mesa.js_bridge_env import BridgeClient, REPO_ROOT

COORD_FORMAT = "mesa-coordinator-json-v1"
SENSOR_TIER = "local-probe-field"
ACTION_MAX = 1.0
PROBE_EPSILON = 0.1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="h1_2e_cancel_probe")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--train-seeds", type=int, default=32)
    ap.add_argument("--val-seeds", type=int, default=16)
    ap.add_argument("--train-seed-start", type=int, default=20000)
    ap.add_argument("--val-seed-start", type=int, default=20300)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--updates", type=int, default=64)
    ap.add_argument("--rollouts-per-update", type=int, default=32)
    ap.add_argument("--ppo-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--log-std-init", type=float, default=-1.25)
    ap.add_argument("--init-guard", required=True)
    ap.add_argument("--init-arbiter", required=True)
    ap.add_argument("--init-monolith-adapter", required=True)
    ap.add_argument("--guard-action-mode", default="cancel-reward", choices=["cancel-reward"])
    ap.add_argument("--cancel-init", default="zero", choices=["zero"])
    ap.add_argument("--cancel-cap", type=float, default=1.0)
    ap.add_argument("--cancel-bias-init", type=float, default=-5.0)
    ap.add_argument("--checkpoint-every", type=int, default=64,
                    help="export current models + checkpoint.json every N updates (interruption insurance)")
    ap.add_argument("--cap-mode", default="reward-asymmetric", choices=["reward-asymmetric"])
    ap.add_argument("--field-cap", type=float, default=1.0)
    ap.add_argument("--reward-cap", type=float, default=0.5)
    ap.add_argument("--guard-cap", type=float, default=0.7)
    ap.add_argument("--field-policy", required=True)
    ap.add_argument("--reward-policy", required=True)
    return ap.parse_args()


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def norm2(v: np.ndarray | list[float]) -> float:
    return float(math.hypot(float(v[0]), float(v[1])))


def clip_action(v: np.ndarray) -> np.ndarray:
    n = norm2(v)
    if n > ACTION_MAX and n > 0:
        return v * (ACTION_MAX / n)
    return v


def softmax_np(v: np.ndarray) -> np.ndarray:
    x = v - np.max(v)
    e = np.exp(x)
    return e / max(float(e.sum()), 1e-12)


def cap_simplex_project(raw: np.ndarray, caps: np.ndarray) -> np.ndarray:
    raw_nn = np.maximum(raw, 0)
    raw_sum_all = float(raw_nn.sum())
    if raw_sum_all > 1e-12:
        w = raw_nn / raw_sum_all
    else:
        w = np.ones_like(raw_nn) / len(raw_nn)
    frozen = np.zeros(len(raw_nn), dtype=bool)
    for _ in range(len(raw_nn)):
        over = [i for i, value in enumerate(w) if not frozen[i] and value > caps[i] + 1e-12]
        if not over:
            break
        for i in over:
            w[i] = caps[i]
            frozen[i] = True
        frozen_mass = float(caps[frozen].sum())
        remaining = max(0.0, 1.0 - frozen_mass)
        unfrozen = [i for i in range(len(raw_nn)) if not frozen[i]]
        if not unfrozen:
            break
        raw_sum = float(raw_nn[unfrozen].sum())
        if raw_sum > 1e-12:
            for i in unfrozen:
                w[i] = remaining * (raw_nn[i] / raw_sum)
        else:
            for i in unfrozen:
                w[i] = remaining / len(unfrozen)
    return w.astype(np.float32)


def cell_seed_hash(cell_id: str, seed: int, channel: int = 0) -> float:
    h = seed & 0xFFFFFFFF
    for ch in cell_id:
        h = ((h ^ ord(ch)) * 0x9E3779B1) & 0xFFFFFFFF
    h = ((h ^ channel) * 0x85EBCA77) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x9E3779B1) & 0xFFFFFFFF
    h ^= h >> 13
    return (h & 0xFFFFFFFF) / 0xFFFFFFFF


def uniform_range(cell_id: str, seed: int, channel: int, lo: float, hi: float) -> float:
    return lo + cell_seed_hash(cell_id, seed, channel) * (hi - lo)


def rotating_channel_index(cell_id: str, seed: int, channel_count: int = 4) -> int:
    return int(math.floor(cell_seed_hash(cell_id, seed, 99) * channel_count))


def normalize_cell_id(cell_id: str) -> str:
    return cell_id[:-4] + "-medium" if cell_id.endswith("-med") else cell_id


def build_probe_for_cell(cell_id: str, seed: int) -> dict[str, Any] | None:
    cid = normalize_cell_id(cell_id)
    if cid == "nominal":
        return None
    if cid == "geometric-light":
        if cell_seed_hash(cid, seed, 0) < 0.5:
            return {"rotate": uniform_range(cid, seed, 1, -math.pi / 8, math.pi / 8)}
        return {"translate": [uniform_range(cid, seed, 2, -0.5, 0.5), uniform_range(cid, seed, 3, -0.5, 0.5)]}
    if cid == "geometric-medium":
        sign = -1 if cell_seed_hash(cid, seed, 0) < 0.5 else 1
        return {"rotate": sign * (math.pi / 4), "translate": [uniform_range(cid, seed, 1, -1, 1), uniform_range(cid, seed, 2, -1, 1)]}
    if cid == "geometric-heavy":
        sign = -1 if cell_seed_hash(cid, seed, 0) < 0.5 else 1
        return {"mirror": "x", "rotate": sign * (math.pi / 2)}
    if cid == "decoy-light":
        return {"decoy": {"strength": 0.3, "decay": "linear", "r": 4.0}}
    if cid == "decoy-medium":
        return {"decoy": {"strength": 0.6, "decay": "linear", "r": 3.0}}
    if cid == "decoy-heavy":
        return {"decoy": {"strength": 1.0, "decay": "inv_sq", "r": 2.0}}
    if cid == "sensor-noise-light":
        return {"perChannelNoise": {str(rotating_channel_index(cid, seed)): 0.05}}
    if cid == "sensor-noise-medium":
        return {"perChannelNoise": {"0": 0.2, "2": 0.2}}
    if cid == "sensor-noise-heavy":
        return {"perChannelNoise": {"0": 0.5, "1": 0.5, "2": 0.5, "3": 0.5}}
    if cid == "sensor-delay-light":
        return {"sensorDelay": 1}
    if cid == "sensor-delay-medium":
        return {"sensorDelay": 3}
    if cid == "sensor-delay-heavy":
        return {"sensorDelay": 5}
    raise ValueError(f"unknown probe cell: {cell_id}")


class JsonPolicy:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        norm = payload["normalization"]
        self.mean = np.asarray(norm["obs_mean"], dtype=np.float32)
        self.std = np.maximum(np.asarray(norm["obs_std"], dtype=np.float32), 1e-8)
        self.layers = [
            (np.asarray(layer["weight"], dtype=np.float32), np.asarray(layer["bias"], dtype=np.float32))
            for layer in payload["layers"]
        ]
        self.action_scale = float(payload.get("action_scale", 1.0))

    def act(self, obs: list[float]) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float32) - self.mean) / self.std
        for weight, bias in self.layers:
            x = np.tanh(weight @ x + bias)
        return clip_action(x * self.action_scale).astype(np.float32)


class CoordActor(nn.Module):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        if payload.get("format") != COORD_FORMAT:
            raise ValueError(f"expected {COORD_FORMAT}, got {payload.get('format')}")
        self.kind = str(payload.get("kind", "coord"))
        self.input_features = list(payload["input_features"])
        norm = payload["normalization"]
        self.register_buffer("mean", torch.tensor(norm["mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(norm["std"], dtype=torch.float32).clamp_min(1e-8))
        self.activations = [str(layer.get("activation", "linear")) for layer in payload["layers"]]
        modules = []
        for layer in payload["layers"]:
            weight = torch.tensor(layer["weight"], dtype=torch.float32)
            bias = torch.tensor(layer["bias"], dtype=torch.float32)
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            modules.append(linear)
        self.layers = nn.ModuleList(modules)

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        x = (raw_features - self.mean) / self.std
        for layer, activation in zip(self.layers, self.activations, strict=True):
            x = layer(x)
            if activation == "tanh":
                x = torch.tanh(x)
            elif activation != "linear":
                raise ValueError(f"unknown activation {activation}")
        return x


class CancelHead(nn.Module):
    """Zero-init scalar cancellation head on the arbiter feature trunk.

    Produces a logit; c_guard = cancel_cap * sigmoid(logit). Weight starts at 0
    and bias at cancel_bias_init (default -5) so c_guard ~= 0 at init.
    """

    def __init__(self, in_dim: int, mean: torch.Tensor, std: torch.Tensor, bias_init: float) -> None:
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.lin = nn.Linear(in_dim, 1)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.fill_(float(bias_init))

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        x = (raw_features - self.mean) / self.std
        return self.lin(x).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, actor: CoordActor, hidden: int = 32) -> None:
        super().__init__()
        self.register_buffer("mean", actor.mean.detach().clone())
        self.register_buffer("std", actor.std.detach().clone())
        dim = len(actor.input_features)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        x = (raw_features - self.mean) / self.std
        return self.net(x).squeeze(-1)


class ActorCritic(nn.Module):
    """Plain 2-output monolith actor-critic (unchanged from H1.2d)."""

    def __init__(self, actor: CoordActor, log_std_init: float) -> None:
        super().__init__()
        self.actor = actor
        out_dim = actor.layers[-1].out_features
        self.value = ValueNet(actor, hidden=max(16, min(64, actor.layers[0].out_features)))
        self.log_std = nn.Parameter(torch.full((out_dim,), float(log_std_init)))

    def policy_mean(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor(features)

    def distribution(self, features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.policy_mean(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, features: torch.Tensor):
        dist = self.distribution(features)
        raw_action = dist.sample()
        return raw_action, dist.log_prob(raw_action).sum(-1), self.value(features), dist.entropy().sum(-1)

    def evaluate_actions(self, features: torch.Tensor, raw_actions: torch.Tensor):
        dist = self.distribution(features)
        return dist.log_prob(raw_actions).sum(-1), dist.entropy().sum(-1), self.value(features)


class CancellingActorCritic(nn.Module):
    """Council actor-critic with a 4th sampled dim = the cancellation logit.

    Policy mean = [arbiter_field, arbiter_reward, arbiter_guard, cancel_logit].
    The first three feed softmax->cap->simplex weights; the fourth feeds
    c_guard = cancel_cap * sigmoid(.). The cancel head is a separate zero-init
    module so H1.2e starts as H1.2d (c_guard ~= 0).
    """

    def __init__(self, arbiter: CoordActor, log_std_init: float, cancel_cap: float, cancel_bias_init: float) -> None:
        super().__init__()
        self.actor = arbiter
        self.cancel_head = CancelHead(len(arbiter.input_features), arbiter.mean, arbiter.std, cancel_bias_init)
        self.value = ValueNet(arbiter, hidden=max(16, min(64, arbiter.layers[0].out_features)))
        self.cancel_cap = float(cancel_cap)
        self.log_std = nn.Parameter(torch.full((4,), float(log_std_init)))

    def policy_mean(self, features: torch.Tensor) -> torch.Tensor:
        arb = self.actor(features)
        cz = self.cancel_head(features).unsqueeze(-1)
        return torch.cat([arb, cz], dim=-1)

    def distribution(self, features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.policy_mean(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, features: torch.Tensor):
        dist = self.distribution(features)
        raw_action = dist.sample()
        return raw_action, dist.log_prob(raw_action).sum(-1), self.value(features), dist.entropy().sum(-1)

    def evaluate_actions(self, features: torch.Tensor, raw_actions: torch.Tensor):
        dist = self.distribution(features)
        return dist.log_prob(raw_actions).sum(-1), dist.entropy().sum(-1), self.value(features)


def param_count(module: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def model_features(model: CoordActor, feature_map: dict[str, float]) -> torch.Tensor:
    values = [float(feature_map[name]) for name in model.input_features]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def coord_forward_np(model: CoordActor, feature_map: dict[str, float]) -> np.ndarray:
    with torch.no_grad():
        return model(model_features(model, feature_map)).squeeze(0).cpu().numpy()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cos2(a: np.ndarray, b: np.ndarray) -> float:
    na = norm2(a)
    nb = norm2(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def local_features(obs: list[float], fa: np.ndarray, ra: np.ndarray, prev_act_norm: float, prev_s_local: float) -> dict[str, float]:
    s0, s1, s2, s3 = obs[2], obs[3], obs[4], obs[5]
    fd = np.asarray([(s0 - s1) / (2 * PROBE_EPSILON), (s2 - s3) / (2 * PROBE_EPSILON)], dtype=np.float32)
    return {
        "obs0": float(obs[0]), "obs1": float(obs[1]), "obs2": float(obs[2]),
        "obs3": float(obs[3]), "obs4": float(obs[4]), "obs5": float(obs[5]),
        "fa_x": float(fa[0]), "fa_y": float(fa[1]), "ra_x": float(ra[0]), "ra_y": float(ra[1]),
        "fa_norm": norm2(fa), "ra_norm": norm2(ra), "disagree_l2": norm2(fa - ra),
        "cos_agree": cos2(fa, ra), "fd_grad_norm": norm2(fd),
        "hist_act_norm_prev": prev_act_norm, "hist_sLocal_prev": prev_s_local,
    }


@dataclass
class Episode:
    features: list[torch.Tensor]
    actions: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    rewards: list[float]
    terminal_alignment: float
    basin_captured: int
    steps: int
    cancel_coeff_mean: float
    cancel_mass_mean: float


def terminal_reward(info: dict[str, Any]) -> tuple[float, int]:
    metrics = info.get("metrics") or {}
    terminal_alignment = float(metrics.get("terminalAlignment", metrics.get("terminal_alignment", 0.0)))
    terminal_distance = float(metrics.get("terminalDistance", metrics.get("terminal_distance", 0.0)))
    position = np.asarray(info.get("position", [0.0, 0.0]), dtype=np.float32)
    x_false = np.asarray(info.get("x_false", [999.0, 999.0]), dtype=np.float32)
    false_dist = norm2(position - x_false)
    basin = 1 if false_dist < terminal_distance and terminal_alignment < 0.5 else 0
    return terminal_alignment - float(basin), basin


def run_episode(
    *,
    client: BridgeClient,
    controller: str,
    agent: nn.Module,
    guard: CoordActor,
    field_policy: JsonPolicy,
    reward_policy: JsonPolicy,
    cell: str,
    seed: int,
    horizon: int,
    caps: np.ndarray,
    cancel_cap: float,
    env_seq: int,
) -> Episode:
    env_id = f"h1_2e_{controller}_{cell}_{seed}_{env_seq}"
    response = client.request(
        {
            "cmd": "make",
            "env_id": env_id,
            "seed": seed,
            "sensor_tier": SENSOR_TIER,
            "env_config": {"horizon": horizon},
            "probe": build_probe_for_cell(cell, seed),
        }
    )
    obs = response["obs"]
    info = response["info"]
    prev_s_local = float(info.get("s_local", sum(obs[2:6]) / 4))
    prev_act_norm = 0.0

    features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    cancel_coeffs: list[float] = []
    cancel_masses: list[float] = []
    done = False
    steps = 0
    terminal_info = info

    while not done:
        fa = field_policy.act(obs)
        ra = reward_policy.act(obs)
        fmap = local_features(obs, fa, ra, prev_act_norm, prev_s_local)
        risk = sigmoid(float(coord_forward_np(guard, fmap)[0]))
        fmap["guard_risk"] = risk
        feat = model_features(agent.actor, fmap)
        raw_action, log_prob, value, _entropy = agent.sample(feat)
        raw_np = raw_action.detach().squeeze(0).cpu().numpy()
        if controller == "council":
            weights = cap_simplex_project(softmax_np(raw_np[:3]), caps)
            c_guard = cancel_cap * sigmoid(float(raw_np[3]))
            a_guard = -c_guard * ra
            action = weights[0] * fa + weights[1] * ra + weights[2] * a_guard
            cancel_coeffs.append(float(c_guard))
            cancel_masses.append(float(weights[2] * c_guard))
        elif controller == "monolith":
            action = raw_np[0] * fa + raw_np[1] * ra
        else:
            raise ValueError(f"unknown controller: {controller}")
        action = clip_action(action)

        step_response = client.request({"cmd": "step", "env_id": env_id, "action": action.tolist()})
        features.append(feat.squeeze(0))
        actions.append(raw_action.squeeze(0))
        log_probs.append(log_prob.squeeze(0).detach())
        values.append(value.squeeze(0).detach())
        rewards.append(0.0)

        obs = step_response["obs"]
        info = step_response["info"]
        done = bool(step_response["done"])
        terminal_info = info
        prev_act_norm = norm2(action)
        prev_s_local = float(info.get("s_local", sum(obs[2:6]) / 4))
        steps += 1

    ep_return, basin = terminal_reward(terminal_info)
    if rewards:
        rewards[-1] = ep_return
    return Episode(
        features=features,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        terminal_alignment=ep_return + basin,
        basin_captured=basin,
        steps=steps,
        cancel_coeff_mean=float(np.mean(cancel_coeffs)) if cancel_coeffs else 0.0,
        cancel_mass_mean=float(np.mean(cancel_masses)) if cancel_masses else 0.0,
    )


def batch_from_episodes(episodes: list[Episode], gamma: float) -> dict[str, torch.Tensor]:
    feats, actions, old_log_probs, values, returns = [], [], [], [], []
    for episode in episodes:
        running = 0.0
        ep_returns: list[float] = []
        for reward in reversed(episode.rewards):
            running = reward + gamma * running
            ep_returns.append(running)
        ep_returns.reverse()
        feats.extend(episode.features)
        actions.extend(episode.actions)
        old_log_probs.extend(episode.log_probs)
        values.extend(episode.values)
        returns.extend(torch.tensor(v, dtype=torch.float32) for v in ep_returns)
    feature_t = torch.stack(feats)
    action_t = torch.stack(actions)
    old_logprob_t = torch.stack(old_log_probs)
    value_t = torch.stack(values)
    return_t = torch.stack(returns)
    adv_t = return_t - value_t
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    return {"features": feature_t, "actions": action_t, "old_log_probs": old_logprob_t, "returns": return_t, "advantages": adv_t}


def ppo_update(agent: nn.Module, optimizer: torch.optim.Optimizer, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, float]:
    n = batch["features"].shape[0]
    indices = torch.randperm(n)
    metrics: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    for _ in range(args.epochs):
        for start in range(0, n, args.minibatch_size):
            mb = indices[start:start + args.minibatch_size]
            new_logprob, entropy, value = agent.evaluate_actions(batch["features"][mb], batch["actions"][mb])
            logratio = new_logprob - batch["old_log_probs"][mb]
            ratio = logratio.exp()
            adv = batch["advantages"][mb]
            policy_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)).mean()
            value_loss = 0.5 * (value - batch["returns"][mb]).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_frac = ((ratio - 1).abs() > args.clip_range).float().mean()
            metrics["policy_loss"].append(float(policy_loss.detach()))
            metrics["value_loss"].append(float(value_loss.detach()))
            metrics["entropy"].append(float(entropy_loss.detach()))
            metrics["approx_kl"].append(float(approx_kl.detach()))
            metrics["clip_frac"].append(float(clip_frac.detach()))
    return {key: float(np.mean(value)) if value else 0.0 for key, value in metrics.items()}


def actor_to_coord_json(actor: CoordActor, *, kind: str, head: str, role_cap: float | None = None,
                        cap_mode: str | None = None, role_caps: dict[str, float] | None = None) -> dict[str, Any]:
    layers = []
    for layer, activation in zip(actor.layers, actor.activations, strict=True):
        layers.append({
            "weight": layer.weight.detach().cpu().numpy().round(6).tolist(),
            "bias": layer.bias.detach().cpu().numpy().round(6).tolist(),
            "activation": activation,
        })
    payload: dict[str, Any] = {
        "format": COORD_FORMAT,
        "kind": kind,
        "input_features": actor.input_features,
        "normalization": {
            "mean": [round(float(v), 6) for v in actor.mean.detach().cpu().numpy().tolist()],
            "std": [round(float(v), 6) for v in actor.std.detach().cpu().numpy().tolist()],
        },
        "layers": layers,
        "head": head,
    }
    if role_cap is not None:
        payload["role_cap"] = role_cap
    if cap_mode is not None:
        payload["cap_mode"] = cap_mode
    if role_caps is not None:
        payload["role_caps"] = role_caps
    return payload


def cancel_head_to_json(council: CancellingActorCritic, input_features: list[str]) -> dict[str, Any]:
    lin = council.cancel_head.lin
    return {
        "input_features": input_features,
        "normalization": {
            "mean": [round(float(v), 6) for v in council.cancel_head.mean.detach().cpu().numpy().tolist()],
            "std": [round(float(v), 6) for v in council.cancel_head.std.detach().cpu().numpy().tolist()],
        },
        "layers": [{
            "weight": lin.weight.detach().cpu().numpy().round(6).tolist(),
            "bias": lin.bias.detach().cpu().numpy().round(6).tolist(),
            "activation": "linear",
        }],
        "head": "sigmoid_scaled",
        "cancel_cap": council.cancel_cap,
    }


def write_outputs(out: Path, guard_payload: dict[str, Any], council: "CancellingActorCritic",
                  monolith: "ActorCritic", cap_mode: str, role_caps: dict[str, float]) -> None:
    """Export the three deployable model JSONs. Used for periodic checkpoints AND
    the final write, so an interrupted run always leaves an eval-able artifact."""
    guard_out = dict(guard_payload)
    guard_out["guard_action_mode"] = "cancel-reward"
    guard_out["cancel_head"] = cancel_head_to_json(council, council.actor.input_features)
    (out / "p_guard_cancel.json").write_text(json.dumps(guard_out) + "\n", encoding="utf-8")
    (out / "p_council_arbiter_cancel.json").write_text(
        json.dumps(actor_to_coord_json(council.actor, kind="arbiter", head="softmax_cap", role_cap=0.70,
                                       cap_mode=cap_mode, role_caps=role_caps)) + "\n", encoding="utf-8")
    (out / "m_adapter_rl_plus.json").write_text(
        json.dumps(actor_to_coord_json(monolith.actor, kind="m_adapter", head="linear_blend")) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.ppo_seed)
    np.random.seed(args.ppo_seed)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    guard_payload = json.loads(repo_path(args.init_guard).read_text(encoding="utf-8"))
    arbiter_payload = json.loads(repo_path(args.init_arbiter).read_text(encoding="utf-8"))
    monolith_payload = json.loads(repo_path(args.init_monolith_adapter).read_text(encoding="utf-8"))
    field_policy = JsonPolicy(json.loads(repo_path(args.field_policy).read_text(encoding="utf-8")))
    reward_policy = JsonPolicy(json.loads(repo_path(args.reward_policy).read_text(encoding="utf-8")))

    guard = CoordActor(guard_payload)
    guard.eval()
    for p in guard.parameters():
        p.requires_grad_(False)
    council = CancellingActorCritic(CoordActor(arbiter_payload), args.log_std_init, args.cancel_cap, args.cancel_bias_init)
    monolith = ActorCritic(CoordActor(monolith_payload), args.log_std_init)
    opt_council = torch.optim.Adam(council.parameters(), lr=args.lr)
    opt_monolith = torch.optim.Adam(monolith.parameters(), lr=args.lr)
    caps = np.asarray([args.field_cap, args.reward_cap, args.guard_cap], dtype=np.float32)
    role_caps = {"field": args.field_cap, "reward": args.reward_cap, "guard": args.guard_cap}

    history: list[dict[str, Any]] = []
    start_time = time.time()
    env_steps = {"council": 0, "monolith": 0}
    episodes_seen = {"council": 0, "monolith": 0}
    env_seq = 0

    with BridgeClient() as client:
        for update in range(1, args.updates + 1):
            cases = []
            for j in range(args.rollouts_per_update):
                seed = args.train_seed_start + ((update - 1) * args.rollouts_per_update + j) % max(args.train_seeds, 1)
                cell = cells[((update - 1) * args.rollouts_per_update + j) % len(cells)]
                cases.append((cell, seed))

            council_eps, monolith_eps = [], []
            for cell, seed in cases:
                env_seq += 1
                council_eps.append(run_episode(client=client, controller="council", agent=council, guard=guard,
                                               field_policy=field_policy, reward_policy=reward_policy, cell=cell, seed=seed,
                                               horizon=args.horizon, caps=caps, cancel_cap=args.cancel_cap, env_seq=env_seq))
                env_seq += 1
                monolith_eps.append(run_episode(client=client, controller="monolith", agent=monolith, guard=guard,
                                                field_policy=field_policy, reward_policy=reward_policy, cell=cell, seed=seed,
                                                horizon=args.horizon, caps=caps, cancel_cap=args.cancel_cap, env_seq=env_seq))

            env_steps["council"] += sum(e.steps for e in council_eps)
            env_steps["monolith"] += sum(e.steps for e in monolith_eps)
            episodes_seen["council"] += len(council_eps)
            episodes_seen["monolith"] += len(monolith_eps)
            c_metrics = ppo_update(council, opt_council, batch_from_episodes(council_eps, args.gamma), args)
            m_metrics = ppo_update(monolith, opt_monolith, batch_from_episodes(monolith_eps, args.gamma), args)
            history.append({
                "update": update,
                "council_return_mean": mean([sum(e.rewards) for e in council_eps]),
                "monolith_return_mean": mean([sum(e.rewards) for e in monolith_eps]),
                "council_alignment_mean": mean([e.terminal_alignment for e in council_eps]),
                "monolith_alignment_mean": mean([e.terminal_alignment for e in monolith_eps]),
                "council_basin_rate": mean([float(e.basin_captured) for e in council_eps]),
                "monolith_basin_rate": mean([float(e.basin_captured) for e in monolith_eps]),
                "cancel_coeff_mean": mean([e.cancel_coeff_mean for e in council_eps]),
                "cancel_mass_mean": mean([e.cancel_mass_mean for e in council_eps]),
                "council_steps": sum(e.steps for e in council_eps),
                "monolith_steps": sum(e.steps for e in monolith_eps),
                "council_clip_frac": c_metrics["clip_frac"],
                "monolith_clip_frac": m_metrics["clip_frac"],
                "council_approx_kl": c_metrics["approx_kl"],
                "council_entropy": c_metrics["entropy"],
                "council_value_loss": c_metrics["value_loss"],
            })
            print(
                f"h1.2e ppo update={update}/{args.updates} "
                f"c_return={history[-1]['council_return_mean']:.3f} m_return={history[-1]['monolith_return_mean']:.3f} "
                f"cancel={history[-1]['cancel_coeff_mean']:.3f} mass={history[-1]['cancel_mass_mean']:.3f} "
                f"steps={history[-1]['council_steps'] + history[-1]['monolith_steps']}",
                flush=True,
            )
            # interruption insurance: periodically export models + progress so a
            # power/sleep event never loses more than --checkpoint-every updates.
            if update % args.checkpoint_every == 0 or update == args.updates:
                write_outputs(out, guard_payload, council, monolith, args.cap_mode, role_caps)
                write_rows(out / "ppo-history.csv", history, list(history[0].keys()))
                (out / "checkpoint.json").write_text(
                    json.dumps({
                        "last_update": update, "updates_total": args.updates,
                        "env_steps": env_steps["council"] + env_steps["monolith"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }) + "\n",
                    encoding="utf-8",
                )

    elapsed = time.time() - start_time
    guard_p = param_count(guard, trainable_only=False)
    arb_p = param_count(council.actor, trainable_only=False)
    cancel_p = param_count(council.cancel_head, trainable_only=False)
    mon_p = param_count(monolith.actor, trainable_only=False)
    council_total = guard_p + arb_p + cancel_p
    budget_ratio = mon_p / max(council_total, 1)

    # final export (also written periodically as checkpoints during the loop)
    write_outputs(out, guard_payload, council, monolith, args.cap_mode, role_caps)
    write_rows(out / "ppo-history.csv", history, list(history[0].keys()) if history else ["update"])

    report = {
        "spec": "docs/mesa/H1_2E_CANCELLING_GUARD_SPEC.md",
        "phase": args.phase,
        "opened_after": ["docs/mesa/H1_2D_RESULTS.md"],
        "algorithm": "ppo",
        "seed": args.ppo_seed,
        "cells": cells,
        "train_seed_start": args.train_seed_start, "train_seeds": args.train_seeds,
        "val_seed_start": args.val_seed_start, "val_seeds": args.val_seeds,
        "updates": args.updates, "rollouts_per_update": args.rollouts_per_update,
        "epochs": args.epochs, "lr": args.lr, "gamma": args.gamma, "clip_range": args.clip_range,
        "guard_action_mode": args.guard_action_mode,
        "cancel": {"init": args.cancel_init, "cancel_cap": args.cancel_cap, "bias_init": args.cancel_bias_init,
                   "head": "separate-zero-init-scalar (4th sampled policy dim)"},
        "cap_mode": args.cap_mode, "role_caps": role_caps,
        "warm_start": {
            "guard": str(repo_path(args.init_guard).relative_to(REPO_ROOT)).replace("\\", "/"),
            "arbiter": str(repo_path(args.init_arbiter).relative_to(REPO_ROOT)).replace("\\", "/"),
            "monolith_adapter": str(repo_path(args.init_monolith_adapter).relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "outputs": {
            "guard": str((out / "p_guard_cancel.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            "arbiter": str((out / "p_council_arbiter_cancel.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            "monolith_adapter": str((out / "m_adapter_rl_plus.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            "history": str((out / "ppo-history.csv").relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "params": {
            "budget_basis": "exported_controller_actor_params; guard frozen for PPO; cancel head is the new H1.2e trainable",
            "guard": guard_p, "arbiter": arb_p, "cancel_head": cancel_p,
            "council_total": council_total, "m_adapter": mon_p,
            "budget_ratio_m_over_council": round(budget_ratio, 4),
            "budget_within_5pct": bool(abs(budget_ratio - 1.0) <= 0.05),
            "ppo_updated_actor_params": {
                "arbiter": param_count(council.actor), "cancel_head": param_count(council.cancel_head),
                "m_adapter": param_count(monolith.actor),
            },
        },
        "rollout_budget": {
            "council_episodes": episodes_seen["council"], "monolith_episodes": episodes_seen["monolith"],
            "council_env_steps": env_steps["council"], "monolith_env_steps": env_steps["monolith"],
            "same_episode_budget": episodes_seen["council"] == episodes_seen["monolith"],
        },
        "timing": {
            "elapsed_sec": round(elapsed, 3),
            "env_steps_per_sec": round((env_steps["council"] + env_steps["monolith"]) / max(elapsed, 1e-9), 2),
        },
        "software": {"python": platform.python_version(), "torch": torch.__version__},
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"H1.2e cancel-guard trainer done. updates={args.updates} "
        f"env_steps={env_steps['council'] + env_steps['monolith']} elapsed={elapsed:.2f}s "
        f"steps/s={report['timing']['env_steps_per_sec']} budget_ratio={budget_ratio:.3f} "
        f"cancel_head_params={cancel_p}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
