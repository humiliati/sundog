"""Policy modules and JSON export helpers for Mesa Phase 2."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


POLICY_JSON_FORMAT = "mesa-policy-json-v1"


@dataclass(frozen=True)
class PolicyConfig:
    tier: str = "Small"
    obs_dim: int = 6
    act_dim: int = 2
    hidden_size: int = 64
    depth: int = 2
    action_scale: float = 1.0
    activation: str = "tanh"


CAPACITY_CONFIGS: dict[str, PolicyConfig] = {
    "small": PolicyConfig(tier="Small", hidden_size=64, depth=2),
    "medium": PolicyConfig(tier="Medium", hidden_size=256, depth=4),
    "large": PolicyConfig(tier="Large", hidden_size=1024, depth=5),
}


class MesaMlpPolicy(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        if config.activation != "tanh":
            raise ValueError(f"unsupported activation: {config.activation}")
        layers: list[nn.Module] = []
        in_dim = config.obs_dim
        for _ in range(config.depth):
            layers.append(nn.Linear(in_dim, config.hidden_size))
            layers.append(nn.Tanh())
            in_dim = config.hidden_size
        layers.append(nn.Linear(in_dim, config.act_dim))
        self.net = nn.Sequential(*layers)
        self.config = config

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.config.action_scale


def policy_config_for_tier(tier: str, *, action_scale: float = 1.0) -> PolicyConfig:
    key = tier.lower()
    if key not in CAPACITY_CONFIGS:
        raise ValueError(f"unknown capacity tier {tier!r}; expected one of {sorted(CAPACITY_CONFIGS)}")
    base = CAPACITY_CONFIGS[key]
    return PolicyConfig(**{**asdict(base), "action_scale": action_scale})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(policy: nn.Module) -> int:
    return sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)


def _round_nested(values: Any) -> Any:
    if isinstance(values, list):
        return [_round_nested(value) for value in values]
    return round(float(values), 10)


def _linear_layers(policy: MesaMlpPolicy) -> list[dict[str, Any]]:
    layers = []
    for module in policy.net:
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy().tolist()
            bias = module.bias.detach().cpu().numpy().tolist()
            layers.append({
                "weight": _round_nested(weight),
                "bias": _round_nested(bias),
            })
    return layers


def policy_to_json_dict(
    policy: MesaMlpPolicy,
    *,
    family: str,
    variant: str,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = policy.config
    return {
        "format": POLICY_JSON_FORMAT,
        "family": family,
        "variant": variant,
        "tier": config.tier,
        "obs_dim": config.obs_dim,
        "act_dim": config.act_dim,
        "activation": config.activation,
        "action_scale": config.action_scale,
        "hidden_size": config.hidden_size,
        "depth": config.depth,
        "parameter_count": count_parameters(policy),
        "layers": _linear_layers(policy),
        "normalization": {
            "obs_mean": _round_nested(obs_mean.tolist()),
            "obs_std": _round_nested(obs_std.tolist()),
        },
        "metadata": metadata or {},
    }


def write_policy_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def policy_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[MesaMlpPolicy, np.ndarray, np.ndarray]:
    config = PolicyConfig(**checkpoint["policy_config"])
    policy = MesaMlpPolicy(config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    obs_mean = np.asarray(checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(checkpoint["obs_std"], dtype=np.float32)
    obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)
    return policy, obs_mean, obs_std

