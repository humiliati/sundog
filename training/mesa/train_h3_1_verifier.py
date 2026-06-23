"""H3.1 PPO trainer for the body-invariant verifier/guard rung."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from training.mesa.h1_trust_features import BASE_H1_FEATURES, H1FeatureState, build_h1_local_features
from training.mesa.h3_body_invariant_task import (
    H3_BODY_ADMITTED_CELLS,
    H3_BODY_CELL_DEFS,
    H3_BODY_PROBE_EPSILON,
    BodyInvariantGateEnv,
)
from training.mesa.train_h1_rl_arbiter import (
    COORD_FORMAT,
    ActorCritic,
    CoordActor,
    Episode,
    actor_to_coord_json,
    batch_from_episodes,
    cap_simplex_project,
    mean,
    model_features,
    param_count,
    ppo_update,
    sigmoid,
    softmax_np,
    write_rows,
)


H3_CERT_FEATURES = [f"certificate_cue_{i}" for i in range(4)]
H3_1_FEATURES = list(BASE_H1_FEATURES) + ["reward_magnitude", "invariant_magnitude", *H3_CERT_FEATURES]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="h3_1_verifier_smoke_seed_0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", default=",".join(H3_BODY_ADMITTED_CELLS))
    ap.add_argument("--train-seeds", type=int, default=32)
    ap.add_argument("--train-seed-start", type=int, default=20000)
    ap.add_argument("--horizon", type=int, default=145)
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
    ap.add_argument("--checkpoint-every", type=int, default=16)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--field-cap", type=float, default=1.0)
    ap.add_argument("--reward-cap", type=float, default=0.5)
    ap.add_argument("--gate-completion-reward", type=float, default=0.25)
    return ap.parse_args()


def norm2(v: np.ndarray | list[float]) -> float:
    return float(math.hypot(float(v[0]), float(v[1])))


def clip_action(v: np.ndarray, action_max: float) -> np.ndarray:
    n = norm2(v)
    if n > action_max and n > 0:
        return v * (action_max / n)
    return v


def h3_terminal_reward(metrics: dict[str, Any], gate_completion_reward: float) -> tuple[float, int, int, float]:
    competence = int(metrics.get("competence", 0))
    basin = int(metrics.get("basin", 0))
    gate_completion = float(metrics.get("gate_completion", 0.0))
    ret = float(competence - basin + gate_completion_reward * gate_completion)
    return ret, competence, basin, gate_completion


def make_coord_payload(
    *,
    kind: str,
    input_features: list[str],
    hidden: int,
    out_dim: int,
    rng: np.random.Generator,
    head: str,
    output_bias: list[float] | None = None,
    random_scale: float = 0.03,
) -> dict[str, Any]:
    dims = [len(input_features), hidden, hidden, out_dim]
    layers = []
    for i in range(len(dims) - 1):
        weight = rng.normal(0.0, random_scale, size=(dims[i + 1], dims[i])).astype(float)
        bias = np.zeros(dims[i + 1], dtype=float)
        if i == len(dims) - 2 and output_bias is not None:
            bias = np.asarray(output_bias, dtype=float)
            weight *= 0.1
        layers.append(
            {
                "weight": weight.round(6).tolist(),
                "bias": bias.round(6).tolist(),
                "activation": "linear" if i == len(dims) - 2 else "tanh",
            }
        )
    return {
        "format": COORD_FORMAT,
        "kind": kind,
        "input_features": input_features,
        "normalization": {"mean": [0.0 for _ in input_features], "std": [1.0 for _ in input_features]},
        "layers": layers,
        "head": head,
    }


class JointVerifierCouncil(nn.Module):
    """Two exported actors trained as one stochastic policy."""

    def __init__(self, arbiter: CoordActor, verifier: CoordActor, log_std_init: float) -> None:
        super().__init__()
        if list(arbiter.input_features) != list(verifier.input_features):
            raise ValueError("H3.1 arbiter and verifier must share feature schema")
        self.arbiter = arbiter
        self.verifier = verifier
        self.input_features = list(arbiter.input_features)
        self.value = nn.Sequential(
            nn.Linear(len(self.input_features), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.full((3,), float(log_std_init)))

    def _mean(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.arbiter(features), self.verifier(features)], dim=-1)

    def distribution(self, features: torch.Tensor) -> torch.distributions.Normal:
        loc = self._mean(features)
        std = torch.exp(self.log_std).expand_as(loc)
        return torch.distributions.Normal(loc, std)

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(features)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(features).squeeze(-1)
        return raw_action, log_prob, value, entropy

    def evaluate_actions(self, features: torch.Tensor, raw_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(features)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(features).squeeze(-1)
        return log_prob, entropy, value


def bootstrap_models(ppo_seed: int) -> tuple[JointVerifierCouncil, ActorCritic, ActorCritic]:
    rng = np.random.default_rng(410000 + int(ppo_seed))
    verifier = CoordActor(
        make_coord_payload(
            kind="verifier_guard",
            input_features=H3_1_FEATURES,
            hidden=32,
            out_dim=1,
            rng=rng,
            head="sigmoid",
            output_bias=[-2.0],
        )
    )
    arbiter = CoordActor(
        make_coord_payload(
            kind="arbiter",
            input_features=H3_1_FEATURES,
            hidden=49,
            out_dim=2,
            rng=rng,
            head="softmax_cap_2way",
            output_bias=[1.0, 0.0],
        )
    )
    m_capped = CoordActor(
        make_coord_payload(
            kind="m_capped_h3_1",
            input_features=H3_1_FEATURES,
            hidden=62,
            out_dim=2,
            rng=rng,
            head="softmax_cap_2way",
            output_bias=[1.0, 0.0],
        )
    )
    m_flat_veto = CoordActor(
        make_coord_payload(
            kind="m_flat_veto_h3_1",
            input_features=H3_1_FEATURES,
            hidden=62,
            out_dim=3,
            rng=rng,
            head="softmax_cap_2way_plus_veto",
            output_bias=[1.0, 0.0, -2.0],
        )
    )
    return (
        JointVerifierCouncil(arbiter, verifier, log_std_init=-1.25),
        ActorCritic(m_capped, log_std_init=-1.25),
        ActorCritic(m_flat_veto, log_std_init=-1.25),
    )


def h3_feature_audit(features: list[str]) -> dict[str, Any]:
    forbidden = [
        f
        for f in features
        if any(
            s in f.lower()
            for s in ("body_coordinate", "invariant_label", "true_", "basin", "cell", "seed", "label", "metric", "outcome")
        )
    ]
    return {
        "feature_schema": "H3.1 local + reward_magnitude + invariant_magnitude + certificate cues",
        "base_feature_count": len(BASE_H1_FEATURES),
        "h3_feature_count": len(H3_1_FEATURES),
        "inference_feature_count": len(features),
        "reward_magnitude_present": "reward_magnitude" in features,
        "invariant_magnitude_present": "invariant_magnitude" in features,
        "certificate_features_present": [f for f in H3_CERT_FEATURES if f in features],
        "missing_features": [f for f in H3_1_FEATURES if f not in features],
        "extra_features": [f for f in features if f not in H3_1_FEATURES],
        "forbidden_feature_scan": forbidden,
        "no_privileged_feature_names": not forbidden,
    }


def build_h3_features(
    *,
    obs_vec: list[float],
    obs_obj: dict[str, Any],
    fa: np.ndarray,
    ra: np.ndarray,
    state: H1FeatureState,
) -> dict[str, float]:
    fmap = build_h1_local_features(obs_vec, fa, ra, eps=H3_BODY_PROBE_EPSILON, state=state, feature_mode="base")
    fmap["reward_magnitude"] = float(obs_obj["reward_magnitude"])
    fmap["invariant_magnitude"] = float(obs_obj["invariant_magnitude"])
    cert = list(obs_obj["certificate"])
    for i, name in enumerate(H3_CERT_FEATURES):
        fmap[name] = float(cert[i]) if i < len(cert) else 0.0
    return fmap


def run_episode(
    *,
    controller: str,
    agent: Any,
    cell: str,
    seed: int,
    horizon: int,
    caps: np.ndarray,
    gate_completion_reward: float,
) -> Episode:
    if cell not in H3_BODY_CELL_DEFS:
        raise ValueError(f"unknown H3.1 cell: {cell}")
    env = BodyInvariantGateEnv({"horizon": horizon})
    obs_obj = env.reset(seed, H3_BODY_CELL_DEFS[cell])
    obs_vec = env.obs_vector(obs_obj)
    feature_state = H1FeatureState()
    feature_state.reset(obs_vec, {"s_local": obs_obj["sLocal"]})

    features: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    done = False
    steps = 0

    while not done:
        fa = np.asarray(env.field_action(), dtype=np.float32)
        ra = np.asarray(env.reward_action(), dtype=np.float32)
        fmap = build_h3_features(obs_vec=obs_vec, obs_obj=obs_obj, fa=fa, ra=ra, state=feature_state)
        feat = model_features(agent.arbiter if controller == "council" else agent.actor, fmap)
        raw_action, log_prob, value, _entropy = agent.sample(feat)
        raw_np = raw_action.detach().squeeze(0).cpu().numpy()

        if controller == "council":
            weights = cap_simplex_project(softmax_np(raw_np[:2]), caps)
            veto = sigmoid(float(raw_np[2]))
            verified_reward = (1.0 - veto) * ra + veto * fa
            action = weights[0] * fa + weights[1] * verified_reward
        elif controller == "m_capped":
            weights = cap_simplex_project(softmax_np(raw_np), caps)
            action = weights[0] * fa + weights[1] * ra
        elif controller == "m_flat_veto":
            weights = cap_simplex_project(softmax_np(raw_np[:2]), caps)
            veto = sigmoid(float(raw_np[2]))
            verified_reward = (1.0 - veto) * ra + veto * fa
            action = weights[0] * fa + weights[1] * verified_reward
        else:
            raise ValueError(f"unknown controller: {controller}")

        action = clip_action(action, float(env.cfg["actionMax"]))
        feature_state.note_action(action, info={"s_local": obs_obj["sLocal"]}, obs=obs_vec)
        step = env.step(action.tolist())
        features.append(feat.squeeze(0))
        actions.append(raw_action.squeeze(0))
        log_probs.append(log_prob.squeeze(0).detach())
        values.append(value.squeeze(0).detach())
        rewards.append(0.0)
        obs_obj = step.obs
        obs_vec = env.obs_vector(obs_obj)
        done = bool(step.done)
        steps += 1

    ep_return, competence, basin, _gate_completion = h3_terminal_reward(env.metrics(), gate_completion_reward)
    if rewards:
        rewards[-1] = ep_return
    return Episode(
        features=features,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        terminal_alignment=float(competence),
        basin_captured=basin,
        steps=steps,
    )


def write_outputs(
    out: Path,
    council: JointVerifierCouncil,
    m_capped: ActorCritic,
    m_flat_veto: ActorCritic,
    role_caps: dict[str, float],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "p_verifier_guard.json").write_text(
        json.dumps(actor_to_coord_json(council.verifier, kind="verifier_guard", head="sigmoid")) + "\n",
        encoding="utf-8",
    )
    (out / "p_council_arbiter_rl.json").write_text(
        json.dumps(
            actor_to_coord_json(
                council.arbiter,
                kind="arbiter",
                head="softmax_cap_2way",
                cap_mode="reward-asymmetric",
                role_caps=role_caps,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "m_capped_rl.json").write_text(
        json.dumps(
            actor_to_coord_json(
                m_capped.actor,
                kind="m_capped_h3_1",
                head="softmax_cap_2way",
                cap_mode="reward-asymmetric",
                role_caps=role_caps,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "m_flat_veto_rl.json").write_text(
        json.dumps(
            actor_to_coord_json(
                m_flat_veto.actor,
                kind="m_flat_veto_h3_1",
                head="softmax_cap_2way_plus_veto",
                cap_mode="reward-asymmetric",
                role_caps=role_caps,
            )
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.ppo_seed)
    np.random.seed(args.ppo_seed)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for cell in cells:
        if cell not in H3_BODY_CELL_DEFS:
            raise ValueError(f"unknown H3.1 cell: {cell}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    council, m_capped, m_flat_veto = bootstrap_models(args.ppo_seed)
    # Override the bootstrap log std to honor CLI values.
    council.log_std.data.fill_(float(args.log_std_init))
    m_capped.log_std.data.fill_(float(args.log_std_init))
    m_flat_veto.log_std.data.fill_(float(args.log_std_init))

    opt_council = torch.optim.Adam(council.parameters(), lr=args.lr)
    opt_capped = torch.optim.Adam(m_capped.parameters(), lr=args.lr)
    opt_flat = torch.optim.Adam(m_flat_veto.parameters(), lr=args.lr)
    caps = np.asarray([args.field_cap, args.reward_cap], dtype=np.float32)
    role_caps = {"field": args.field_cap, "reward": args.reward_cap}

    history: list[dict[str, Any]] = []
    start_time = time.time()
    env_steps = {"council": 0, "m_capped": 0, "m_flat_veto": 0}
    episodes_seen = {"council": 0, "m_capped": 0, "m_flat_veto": 0}
    start_update = 0
    history_fields = [
        "update",
        "council_return_mean",
        "m_capped_return_mean",
        "m_flat_veto_return_mean",
        "council_competence_mean",
        "m_capped_competence_mean",
        "m_flat_veto_competence_mean",
        "council_basin_rate",
        "m_capped_basin_rate",
        "m_flat_veto_basin_rate",
        "council_steps",
        "m_capped_steps",
        "m_flat_veto_steps",
        "council_policy_loss",
        "m_capped_policy_loss",
        "m_flat_veto_policy_loss",
        "council_value_loss",
        "m_capped_value_loss",
        "m_flat_veto_value_loss",
        "council_entropy",
        "m_capped_entropy",
        "m_flat_veto_entropy",
        "council_approx_kl",
        "m_capped_approx_kl",
        "m_flat_veto_approx_kl",
        "council_clip_frac",
        "m_capped_clip_frac",
        "m_flat_veto_clip_frac",
    ]

    state_path = out / "train_state.pt"
    if state_path.exists() and not args.no_resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        council.load_state_dict(st["council"])
        m_capped.load_state_dict(st["m_capped"])
        m_flat_veto.load_state_dict(st["m_flat_veto"])
        opt_council.load_state_dict(st["opt_council"])
        opt_capped.load_state_dict(st["opt_capped"])
        opt_flat.load_state_dict(st["opt_flat"])
        env_steps = st["env_steps"]
        episodes_seen = st["episodes_seen"]
        history = list(st["history"])
        start_update = int(st["update"])
        try:
            torch.set_rng_state(st["torch_rng"])
            np.random.set_state(st["np_rng"])
        except Exception:
            pass
        print(f"{args.phase} RESUME from update {start_update}/{args.updates}", flush=True)

    def save_train_state(update: int) -> None:
        torch.save(
            {
                "update": update,
                "council": council.state_dict(),
                "m_capped": m_capped.state_dict(),
                "m_flat_veto": m_flat_veto.state_dict(),
                "opt_council": opt_council.state_dict(),
                "opt_capped": opt_capped.state_dict(),
                "opt_flat": opt_flat.state_dict(),
                "env_steps": env_steps,
                "episodes_seen": episodes_seen,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "np_rng": np.random.get_state(),
            },
            state_path,
        )

    for update in range(start_update + 1, args.updates + 1):
        cases = []
        for j in range(args.rollouts_per_update):
            seed = args.train_seed_start + ((update - 1) * args.rollouts_per_update + j) % max(args.train_seeds, 1)
            cell = cells[((update - 1) * args.rollouts_per_update + j) % len(cells)]
            cases.append((cell, seed))

        council_eps = []
        capped_eps = []
        flat_eps = []
        for cell, seed in cases:
            council_eps.append(
                run_episode(
                    controller="council",
                    agent=council,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    caps=caps,
                    gate_completion_reward=args.gate_completion_reward,
                )
            )
            capped_eps.append(
                run_episode(
                    controller="m_capped",
                    agent=m_capped,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    caps=caps,
                    gate_completion_reward=args.gate_completion_reward,
                )
            )
            flat_eps.append(
                run_episode(
                    controller="m_flat_veto",
                    agent=m_flat_veto,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    caps=caps,
                    gate_completion_reward=args.gate_completion_reward,
                )
            )

        env_steps["council"] += sum(e.steps for e in council_eps)
        env_steps["m_capped"] += sum(e.steps for e in capped_eps)
        env_steps["m_flat_veto"] += sum(e.steps for e in flat_eps)
        episodes_seen["council"] += len(council_eps)
        episodes_seen["m_capped"] += len(capped_eps)
        episodes_seen["m_flat_veto"] += len(flat_eps)

        c_metrics = ppo_update(council, opt_council, batch_from_episodes(council_eps, args.gamma), args)
        m_metrics = ppo_update(m_capped, opt_capped, batch_from_episodes(capped_eps, args.gamma), args)
        f_metrics = ppo_update(m_flat_veto, opt_flat, batch_from_episodes(flat_eps, args.gamma), args)
        row = {
            "update": update,
            "council_return_mean": mean([sum(e.rewards) for e in council_eps]),
            "m_capped_return_mean": mean([sum(e.rewards) for e in capped_eps]),
            "m_flat_veto_return_mean": mean([sum(e.rewards) for e in flat_eps]),
            "council_competence_mean": mean([e.terminal_alignment for e in council_eps]),
            "m_capped_competence_mean": mean([e.terminal_alignment for e in capped_eps]),
            "m_flat_veto_competence_mean": mean([e.terminal_alignment for e in flat_eps]),
            "council_basin_rate": mean([float(e.basin_captured) for e in council_eps]),
            "m_capped_basin_rate": mean([float(e.basin_captured) for e in capped_eps]),
            "m_flat_veto_basin_rate": mean([float(e.basin_captured) for e in flat_eps]),
            "council_steps": sum(e.steps for e in council_eps),
            "m_capped_steps": sum(e.steps for e in capped_eps),
            "m_flat_veto_steps": sum(e.steps for e in flat_eps),
            "council_policy_loss": c_metrics["policy_loss"],
            "m_capped_policy_loss": m_metrics["policy_loss"],
            "m_flat_veto_policy_loss": f_metrics["policy_loss"],
            "council_value_loss": c_metrics["value_loss"],
            "m_capped_value_loss": m_metrics["value_loss"],
            "m_flat_veto_value_loss": f_metrics["value_loss"],
            "council_entropy": c_metrics["entropy"],
            "m_capped_entropy": m_metrics["entropy"],
            "m_flat_veto_entropy": f_metrics["entropy"],
            "council_approx_kl": c_metrics["approx_kl"],
            "m_capped_approx_kl": m_metrics["approx_kl"],
            "m_flat_veto_approx_kl": f_metrics["approx_kl"],
            "council_clip_frac": c_metrics["clip_frac"],
            "m_capped_clip_frac": m_metrics["clip_frac"],
            "m_flat_veto_clip_frac": f_metrics["clip_frac"],
        }
        history.append(row)
        print(
            f"{args.phase} ppo update={update}/{args.updates} "
            f"c_return={row['council_return_mean']:.3f} "
            f"m_return={row['m_capped_return_mean']:.3f} "
            f"fv_return={row['m_flat_veto_return_mean']:.3f} "
            f"steps={row['council_steps'] + row['m_capped_steps'] + row['m_flat_veto_steps']}",
            flush=True,
        )
        if update % args.checkpoint_every == 0 or update == args.updates:
            write_outputs(out, council, m_capped, m_flat_veto, role_caps)
            save_train_state(update)
            write_rows(out / "ppo-history.csv", history, history_fields)
            (out / "checkpoint.json").write_text(
                json.dumps({"last_update": update, "updates_total": args.updates, "env_steps": sum(env_steps.values())}) + "\n",
                encoding="utf-8",
            )

    elapsed = time.time() - start_time
    verifier_p = param_count(council.verifier, trainable_only=False)
    arbiter_p = param_count(council.arbiter, trainable_only=False)
    m_capped_p = param_count(m_capped.actor, trainable_only=False)
    m_flat_p = param_count(m_flat_veto.actor, trainable_only=False)
    council_total = verifier_p + arbiter_p
    feature_audit = {
        "feature_schema": "H3.1 local + reward_magnitude + invariant_magnitude + certificate cues",
        "verifier": h3_feature_audit(list(council.verifier.input_features)),
        "arbiter": h3_feature_audit(list(council.arbiter.input_features)),
        "m_capped": h3_feature_audit(list(m_capped.actor.input_features)),
        "m_flat_veto": h3_feature_audit(list(m_flat_veto.actor.input_features)),
        "same_controller_features": (
            list(council.verifier.input_features)
            == list(council.arbiter.input_features)
            == list(m_capped.actor.input_features)
            == list(m_flat_veto.actor.input_features)
        ),
        "auxiliary_verifier_labels": False,
    }

    write_outputs(out, council, m_capped, m_flat_veto, role_caps)
    write_rows(out / "ppo-history.csv", history, history_fields)
    report = {
        "spec": "docs/mesa/H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md",
        "phase": args.phase,
        "algorithm": "ppo",
        "objective": f"competence - basin + {args.gate_completion_reward}*gate_completion",
        "seed": args.ppo_seed,
        "feature_audit": feature_audit,
        "cells": cells,
        "train_seed_start": args.train_seed_start,
        "train_seeds": args.train_seeds,
        "horizon": args.horizon,
        "updates": args.updates,
        "rollouts_per_update": args.rollouts_per_update,
        "env_steps": env_steps,
        "episodes_seen": episodes_seen,
        "elapsed_sec": elapsed,
        "steps_per_sec": sum(env_steps.values()) / max(elapsed, 1e-9),
        "role_caps": role_caps,
        "param_counts": {
            "verifier": verifier_p,
            "arbiter": arbiter_p,
            "council_total": council_total,
            "m_capped": m_capped_p,
            "m_flat_veto": m_flat_p,
            "m_capped_budget_ratio": m_capped_p / max(council_total, 1),
            "m_flat_veto_budget_ratio": m_flat_p / max(council_total, 1),
        },
        "last_history": history[-1] if history else {},
        "platform": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.phase} H3.1 trainer done. updates={args.updates} env_steps={sum(env_steps.values())} "
        f"elapsed={elapsed:.2f}s steps/s={sum(env_steps.values()) / max(elapsed, 1e-9):.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
