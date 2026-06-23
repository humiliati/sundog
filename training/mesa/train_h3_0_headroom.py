"""H3.0-c capped no-role learned-headroom probe."""

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
    softmax_np,
    write_rows,
)


H3_CERT_FEATURES = [f"certificate_cue_{i}" for i in range(4)]
H3_0_FEATURES = list(BASE_H1_FEATURES) + ["reward_magnitude", "invariant_magnitude", *H3_CERT_FEATURES]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="h3_0_headroom_probe_seed_0")
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


def bootstrap_model(ppo_seed: int) -> CoordActor:
    rng = np.random.default_rng(300000 + int(ppo_seed))
    payload = make_coord_payload(
        kind="m_capped_h3",
        input_features=H3_0_FEATURES,
        hidden=64,
        out_dim=2,
        rng=rng,
        head="softmax_cap_2way",
        output_bias=[1.0, 0.0],
    )
    return CoordActor(payload)


def h3_feature_audit(features: list[str]) -> dict[str, Any]:
    forbidden = [
        f
        for f in features
        if any(s in f.lower() for s in ("body_coordinate", "invariant_label", "true_", "basin", "cell", "seed", "label", "metric", "outcome"))
    ]
    return {
        "feature_schema": "H3.0 local + reward_magnitude + invariant_magnitude + certificate cues",
        "base_feature_count": len(BASE_H1_FEATURES),
        "h3_feature_count": len(H3_0_FEATURES),
        "inference_feature_count": len(features),
        "reward_magnitude_present": "reward_magnitude" in features,
        "invariant_magnitude_present": "invariant_magnitude" in features,
        "certificate_features_present": [f for f in H3_CERT_FEATURES if f in features],
        "missing_features": [f for f in H3_0_FEATURES if f not in features],
        "extra_features": [f for f in features if f not in H3_0_FEATURES],
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
    agent: ActorCritic,
    cell: str,
    seed: int,
    horizon: int,
    caps: np.ndarray,
    gate_completion_reward: float,
) -> Episode:
    if cell not in H3_BODY_CELL_DEFS:
        raise ValueError(f"unknown H3.0 cell: {cell}")
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
        feat = model_features(agent.actor, fmap)
        raw_action, log_prob, value, _entropy = agent.sample(feat)
        raw_np = raw_action.detach().squeeze(0).cpu().numpy()
        weights = cap_simplex_project(softmax_np(raw_np), caps)
        action = weights[0] * fa + weights[1] * ra
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

    ep_return, competence, basin, gate_completion = h3_terminal_reward(env.metrics(), gate_completion_reward)
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


def write_outputs(out: Path, agent: ActorCritic, role_caps: dict[str, float]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "m_capped_h3_rl.json").write_text(
        json.dumps(
            actor_to_coord_json(
                agent.actor,
                kind="m_capped_h3",
                head="softmax_cap_2way",
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
            raise ValueError(f"unknown H3.0 cell: {cell}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    actor = bootstrap_model(args.ppo_seed)
    agent = ActorCritic(actor, args.log_std_init)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    caps = np.asarray([args.field_cap, args.reward_cap], dtype=np.float32)
    role_caps = {"field": args.field_cap, "reward": args.reward_cap}

    history: list[dict[str, Any]] = []
    start_time = time.time()
    env_steps = 0
    episodes_seen = 0
    start_update = 0
    history_fields = [
        "update",
        "return_mean",
        "competence_mean",
        "basin_rate",
        "steps",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
    ]

    state_path = out / "train_state.pt"
    if state_path.exists() and not args.no_resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(st["agent"])
        opt.load_state_dict(st["opt"])
        env_steps = int(st["env_steps"])
        episodes_seen = int(st["episodes_seen"])
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
                "agent": agent.state_dict(),
                "opt": opt.state_dict(),
                "env_steps": env_steps,
                "episodes_seen": episodes_seen,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "np_rng": np.random.get_state(),
            },
            state_path,
        )

    for update in range(start_update + 1, args.updates + 1):
        episodes = []
        for j in range(args.rollouts_per_update):
            seed = args.train_seed_start + ((update - 1) * args.rollouts_per_update + j) % max(args.train_seeds, 1)
            cell = cells[((update - 1) * args.rollouts_per_update + j) % len(cells)]
            episodes.append(
                run_episode(
                    agent=agent,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    caps=caps,
                    gate_completion_reward=args.gate_completion_reward,
                )
            )
        env_steps += sum(e.steps for e in episodes)
        episodes_seen += len(episodes)
        metrics = ppo_update(agent, opt, batch_from_episodes(episodes, args.gamma), args)
        row = {
            "update": update,
            "return_mean": mean([sum(e.rewards) for e in episodes]),
            "competence_mean": mean([e.terminal_alignment for e in episodes]),
            "basin_rate": mean([float(e.basin_captured) for e in episodes]),
            "steps": sum(e.steps for e in episodes),
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "approx_kl": metrics["approx_kl"],
            "clip_frac": metrics["clip_frac"],
        }
        history.append(row)
        print(
            f"{args.phase} ppo update={update}/{args.updates} "
            f"return={row['return_mean']:.3f} C={row['competence_mean']:.3f} "
            f"B={row['basin_rate']:.3f} steps={row['steps']}",
            flush=True,
        )
        if update % args.checkpoint_every == 0 or update == args.updates:
            write_outputs(out, agent, role_caps)
            save_train_state(update)
            write_rows(out / "ppo-history.csv", history, history_fields)
            (out / "checkpoint.json").write_text(
                json.dumps({"last_update": update, "updates_total": args.updates, "env_steps": env_steps}) + "\n",
                encoding="utf-8",
            )

    elapsed = time.time() - start_time
    feature_audit = h3_feature_audit(list(agent.actor.input_features))
    write_outputs(out, agent, role_caps)
    write_rows(out / "ppo-history.csv", history, history_fields)
    report = {
        "spec": "docs/mesa/H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md",
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
        "steps_per_sec": env_steps / max(elapsed, 1e-9),
        "role_caps": role_caps,
        "param_counts": {"m_capped_h3": param_count(agent.actor, trainable_only=False)},
        "budget_ratios": {"m_capped_h3": 1.0},
        "last_history": history[-1] if history else {},
        "platform": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.phase} H3.0-c trainer done. updates={args.updates} env_steps={env_steps} "
        f"elapsed={elapsed:.2f}s steps/s={env_steps / max(elapsed, 1e-9):.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
