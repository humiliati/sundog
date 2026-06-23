"""H2.3 PPO trainer for the safe-exploration proxy-poisoning rung."""

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
from training.mesa.h2_safeexp_task import (
    H2_SAFEEXP_ADMITTED_CELLS,
    H2_SAFEEXP_CELL_DEFS,
    H2_SAFEEXP_PROBE_EPSILON,
    SafeExplorationEnv,
)
from training.mesa.train_h1_rl_arbiter import (
    COORD_FORMAT,
    ActorCritic,
    CoordActor,
    Episode,
    actor_to_coord_json,
    batch_from_episodes,
    cap_simplex_project,
    coord_forward_np,
    mean,
    model_features,
    param_count,
    ppo_update,
    sigmoid,
    softmax_np,
    write_rows,
)


H2_3_BASE_FEATURES = list(BASE_H1_FEATURES) + ["reward_magnitude"]
H2_3_ARBITER_FEATURES = H2_3_BASE_FEATURES + ["guard_risk"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="h2_3_safeexp_probe_seed_0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", default=",".join(H2_SAFEEXP_ADMITTED_CELLS))
    ap.add_argument("--train-seeds", type=int, default=32)
    ap.add_argument("--train-seed-start", type=int, default=20000)
    ap.add_argument("--horizon", type=int, default=80)
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
    ap.add_argument("--checkpoint-every", type=int, default=32)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--cap-mode", default="reward-asymmetric", choices=["reward-asymmetric"])
    ap.add_argument("--field-cap", type=float, default=1.0)
    ap.add_argument("--reward-cap", type=float, default=0.5)
    ap.add_argument("--guard-cap", type=float, default=0.7)
    return ap.parse_args()


def norm2(v: np.ndarray | list[float]) -> float:
    return float(math.hypot(float(v[0]), float(v[1])))


def clip_action(v: np.ndarray, action_max: float) -> np.ndarray:
    n = norm2(v)
    if n > action_max and n > 0:
        return v * (action_max / n)
    return v


def h2_3_terminal_reward(metrics: dict[str, Any]) -> tuple[float, int, int, float]:
    competence = int(metrics.get("competence", 0))
    basin = int(metrics.get("basin", 0))
    proxy_return = float(metrics.get("proxy_return", 0.0))
    return proxy_return, competence, basin, proxy_return


def h2_3_feature_audit(features: list[str]) -> dict[str, Any]:
    forbidden = [
        f for f in features
        if any(s in f.lower() for s in ("true_", "basin", "cell", "seed", "label", "metric", "outcome", "proxy_return"))
    ]
    return {
        "base_feature_count": len(BASE_H1_FEATURES),
        "h2_3_feature_count": len(H2_3_BASE_FEATURES),
        "inference_feature_count": len(features),
        "reward_magnitude_present": "reward_magnitude" in features,
        "missing_base_features": [f for f in BASE_H1_FEATURES if f not in features],
        "extra_features": [f for f in features if f not in H2_3_BASE_FEATURES and f != "guard_risk"],
        "forbidden_feature_scan": forbidden,
        "no_privileged_feature_names": not forbidden,
    }


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
        "normalization": {
            "mean": [0.0 for _ in input_features],
            "std": [1.0 for _ in input_features],
        },
        "layers": layers,
        "head": head,
    }


def bootstrap_models(ppo_seed: int) -> tuple[CoordActor, CoordActor, CoordActor, CoordActor]:
    rng = np.random.default_rng(130000 + int(ppo_seed))
    guard_payload = make_coord_payload(
        kind="guard",
        input_features=H2_3_BASE_FEATURES,
        hidden=32,
        out_dim=1,
        rng=rng,
        head="sigmoid",
        output_bias=[0.0],
        random_scale=0.0,
    )
    arbiter_payload = make_coord_payload(
        kind="arbiter",
        input_features=H2_3_ARBITER_FEATURES,
        hidden=32,
        out_dim=3,
        rng=rng,
        head="softmax_cap",
        output_bias=[1.0, 1.0, -4.0],
    )
    uncapped_payload = make_coord_payload(
        kind="m_uncapped",
        input_features=H2_3_BASE_FEATURES,
        hidden=49,
        out_dim=2,
        rng=rng,
        head="linear_blend",
        output_bias=[0.0, 1.2],
    )
    capped_payload = make_coord_payload(
        kind="m_capped",
        input_features=H2_3_BASE_FEATURES,
        hidden=49,
        out_dim=2,
        rng=rng,
        head="softmax_cap_2way",
        output_bias=[0.0, 1.2],
    )
    return CoordActor(guard_payload), CoordActor(arbiter_payload), CoordActor(uncapped_payload), CoordActor(capped_payload)


def build_h2_3_features(
    *,
    obs_vec: list[float],
    obs_obj: dict[str, Any],
    fa: np.ndarray,
    ra: np.ndarray,
    state: H1FeatureState,
) -> dict[str, float]:
    fmap = build_h1_local_features(
        obs_vec,
        fa,
        ra,
        eps=H2_SAFEEXP_PROBE_EPSILON,
        state=state,
        feature_mode="base",
    )
    fmap["reward_magnitude"] = float(obs_obj["reward_magnitude"])
    return fmap


def run_episode(
    *,
    controller: str,
    agent: ActorCritic,
    guard: CoordActor,
    cell: str,
    seed: int,
    horizon: int,
    council_caps: np.ndarray,
    capped_m_caps: np.ndarray,
) -> Episode:
    if cell not in H2_SAFEEXP_CELL_DEFS:
        raise ValueError(f"unknown H2.3 cell: {cell}")
    env = SafeExplorationEnv({"horizon": horizon})
    obs_obj = env.reset(seed, H2_SAFEEXP_CELL_DEFS[cell])
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
        fmap = build_h2_3_features(obs_vec=obs_vec, obs_obj=obs_obj, fa=fa, ra=ra, state=feature_state)
        risk = sigmoid(float(coord_forward_np(guard, fmap)[0]))
        fmap["guard_risk"] = risk
        feat = model_features(agent.actor, fmap)
        raw_action, log_prob, value, _entropy = agent.sample(feat)
        raw_np = raw_action.detach().squeeze(0).cpu().numpy()
        if controller == "council":
            weights = cap_simplex_project(softmax_np(raw_np), council_caps)
            action = weights[0] * fa + weights[1] * ra
        elif controller == "m_uncapped":
            action = raw_np[0] * fa + raw_np[1] * ra
        elif controller == "m_capped":
            weights = cap_simplex_project(softmax_np(raw_np), capped_m_caps)
            action = weights[0] * fa + weights[1] * ra
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

    ep_return, competence, basin, _proxy_return = h2_3_terminal_reward(env.metrics())
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
    guard: CoordActor,
    council: ActorCritic,
    m_uncapped: ActorCritic,
    m_capped: ActorCritic,
    cap_mode: str,
    role_caps: dict[str, float],
) -> None:
    (out / "p_guard.json").write_text(
        json.dumps(actor_to_coord_json(guard, kind="guard", head="sigmoid")) + "\n",
        encoding="utf-8",
    )
    (out / "p_council_arbiter_rl.json").write_text(
        json.dumps(
            actor_to_coord_json(
                council.actor,
                kind="arbiter",
                head="softmax_cap",
                role_cap=0.70,
                cap_mode=cap_mode,
                role_caps=role_caps,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "m_uncapped_rl.json").write_text(
        json.dumps(actor_to_coord_json(m_uncapped.actor, kind="m_uncapped", head="linear_blend")) + "\n",
        encoding="utf-8",
    )
    (out / "m_capped_rl.json").write_text(
        json.dumps(
            actor_to_coord_json(
                m_capped.actor,
                kind="m_capped",
                head="softmax_cap_2way",
                cap_mode=cap_mode,
                role_caps={"field": role_caps["field"], "reward": role_caps["reward"]},
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
        if cell not in H2_SAFEEXP_CELL_DEFS:
            raise ValueError(f"unknown H2.3 cell: {cell}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    guard, arbiter, uncapped_actor, capped_actor = bootstrap_models(args.ppo_seed)
    guard.eval()
    for p in guard.parameters():
        p.requires_grad_(False)
    council = ActorCritic(arbiter, args.log_std_init)
    m_uncapped = ActorCritic(uncapped_actor, args.log_std_init)
    m_capped = ActorCritic(capped_actor, args.log_std_init)
    opt_council = torch.optim.Adam(council.parameters(), lr=args.lr)
    opt_uncapped = torch.optim.Adam(m_uncapped.parameters(), lr=args.lr)
    opt_capped = torch.optim.Adam(m_capped.parameters(), lr=args.lr)
    council_caps = np.asarray([args.field_cap, args.reward_cap, args.guard_cap], dtype=np.float32)
    capped_m_caps = np.asarray([args.field_cap, args.reward_cap], dtype=np.float32)
    role_caps = {"field": args.field_cap, "reward": args.reward_cap, "guard": args.guard_cap}

    history: list[dict[str, Any]] = []
    start_time = time.time()
    env_steps = {"council": 0, "m_uncapped": 0, "m_capped": 0}
    episodes_seen = {"council": 0, "m_uncapped": 0, "m_capped": 0}
    start_update = 0
    history_fields = [
        "update",
        "council_proxy_return_mean",
        "m_uncapped_proxy_return_mean",
        "m_capped_proxy_return_mean",
        "council_competence_mean",
        "m_uncapped_competence_mean",
        "m_capped_competence_mean",
        "council_basin_rate",
        "m_uncapped_basin_rate",
        "m_capped_basin_rate",
        "council_steps",
        "m_uncapped_steps",
        "m_capped_steps",
        "council_policy_loss",
        "m_uncapped_policy_loss",
        "m_capped_policy_loss",
        "council_value_loss",
        "m_uncapped_value_loss",
        "m_capped_value_loss",
        "council_entropy",
        "m_uncapped_entropy",
        "m_capped_entropy",
        "council_approx_kl",
        "m_uncapped_approx_kl",
        "m_capped_approx_kl",
        "council_clip_frac",
        "m_uncapped_clip_frac",
        "m_capped_clip_frac",
    ]

    state_path = out / "train_state.pt"
    if state_path.exists() and not args.no_resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        council.load_state_dict(st["council"])
        m_uncapped.load_state_dict(st["m_uncapped"])
        m_capped.load_state_dict(st["m_capped"])
        opt_council.load_state_dict(st["opt_council"])
        opt_uncapped.load_state_dict(st["opt_uncapped"])
        opt_capped.load_state_dict(st["opt_capped"])
        env_steps = st["env_steps"]
        episodes_seen = st["episodes_seen"]
        history = st["history"]
        start_update = int(st["update"])
        try:
            torch.set_rng_state(st["torch_rng"])
            np.random.set_state(st["np_rng"])
        except Exception:
            pass
        print(
            f"{args.phase} RESUME from update {start_update}/{args.updates} "
            f"(env_steps so far {sum(env_steps.values())})",
            flush=True,
        )

    def save_train_state(update: int) -> None:
        torch.save(
            {
                "update": update,
                "council": council.state_dict(),
                "m_uncapped": m_uncapped.state_dict(),
                "m_capped": m_capped.state_dict(),
                "opt_council": opt_council.state_dict(),
                "opt_uncapped": opt_uncapped.state_dict(),
                "opt_capped": opt_capped.state_dict(),
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
        uncapped_eps = []
        capped_eps = []
        for cell, seed in cases:
            council_eps.append(
                run_episode(
                    controller="council",
                    agent=council,
                    guard=guard,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    council_caps=council_caps,
                    capped_m_caps=capped_m_caps,
                )
            )
            uncapped_eps.append(
                run_episode(
                    controller="m_uncapped",
                    agent=m_uncapped,
                    guard=guard,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    council_caps=council_caps,
                    capped_m_caps=capped_m_caps,
                )
            )
            capped_eps.append(
                run_episode(
                    controller="m_capped",
                    agent=m_capped,
                    guard=guard,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    council_caps=council_caps,
                    capped_m_caps=capped_m_caps,
                )
            )

        env_steps["council"] += sum(e.steps for e in council_eps)
        env_steps["m_uncapped"] += sum(e.steps for e in uncapped_eps)
        env_steps["m_capped"] += sum(e.steps for e in capped_eps)
        episodes_seen["council"] += len(council_eps)
        episodes_seen["m_uncapped"] += len(uncapped_eps)
        episodes_seen["m_capped"] += len(capped_eps)

        c_metrics = ppo_update(council, opt_council, batch_from_episodes(council_eps, args.gamma), args)
        u_metrics = ppo_update(m_uncapped, opt_uncapped, batch_from_episodes(uncapped_eps, args.gamma), args)
        cm_metrics = ppo_update(m_capped, opt_capped, batch_from_episodes(capped_eps, args.gamma), args)
        row = {
            "update": update,
            "council_proxy_return_mean": mean([sum(e.rewards) for e in council_eps]),
            "m_uncapped_proxy_return_mean": mean([sum(e.rewards) for e in uncapped_eps]),
            "m_capped_proxy_return_mean": mean([sum(e.rewards) for e in capped_eps]),
            "council_competence_mean": mean([e.terminal_alignment for e in council_eps]),
            "m_uncapped_competence_mean": mean([e.terminal_alignment for e in uncapped_eps]),
            "m_capped_competence_mean": mean([e.terminal_alignment for e in capped_eps]),
            "council_basin_rate": mean([float(e.basin_captured) for e in council_eps]),
            "m_uncapped_basin_rate": mean([float(e.basin_captured) for e in uncapped_eps]),
            "m_capped_basin_rate": mean([float(e.basin_captured) for e in capped_eps]),
            "council_steps": sum(e.steps for e in council_eps),
            "m_uncapped_steps": sum(e.steps for e in uncapped_eps),
            "m_capped_steps": sum(e.steps for e in capped_eps),
            "council_policy_loss": c_metrics["policy_loss"],
            "m_uncapped_policy_loss": u_metrics["policy_loss"],
            "m_capped_policy_loss": cm_metrics["policy_loss"],
            "council_value_loss": c_metrics["value_loss"],
            "m_uncapped_value_loss": u_metrics["value_loss"],
            "m_capped_value_loss": cm_metrics["value_loss"],
            "council_entropy": c_metrics["entropy"],
            "m_uncapped_entropy": u_metrics["entropy"],
            "m_capped_entropy": cm_metrics["entropy"],
            "council_approx_kl": c_metrics["approx_kl"],
            "m_uncapped_approx_kl": u_metrics["approx_kl"],
            "m_capped_approx_kl": cm_metrics["approx_kl"],
            "council_clip_frac": c_metrics["clip_frac"],
            "m_uncapped_clip_frac": u_metrics["clip_frac"],
            "m_capped_clip_frac": cm_metrics["clip_frac"],
        }
        history.append(row)
        print(
            f"{args.phase} ppo update={update}/{args.updates} "
            f"c_proxy={row['council_proxy_return_mean']:.3f} "
            f"u_proxy={row['m_uncapped_proxy_return_mean']:.3f} "
            f"cap_proxy={row['m_capped_proxy_return_mean']:.3f} "
            f"steps={row['council_steps'] + row['m_uncapped_steps'] + row['m_capped_steps']}",
            flush=True,
        )
        if update % args.checkpoint_every == 0 or update == args.updates:
            write_outputs(out, guard, council, m_uncapped, m_capped, args.cap_mode, role_caps)
            save_train_state(update)
            write_rows(out / "ppo-history.csv", history, history_fields)
            (out / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "last_update": update,
                        "updates_total": args.updates,
                        "env_steps": sum(env_steps.values()),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    elapsed = time.time() - start_time
    guard_p = param_count(guard, trainable_only=False)
    arb_p = param_count(council.actor, trainable_only=False)
    uncapped_p = param_count(m_uncapped.actor, trainable_only=False)
    capped_p = param_count(m_capped.actor, trainable_only=False)
    council_total = guard_p + arb_p
    feature_audit = {
        "feature_schema": "H2.3 base + reward_magnitude",
        "guard": h2_3_feature_audit(list(guard.input_features)),
        "arbiter_base": h2_3_feature_audit([name for name in council.actor.input_features if name != "guard_risk"]),
        "m_uncapped": h2_3_feature_audit(list(m_uncapped.actor.input_features)),
        "m_capped": h2_3_feature_audit(list(m_capped.actor.input_features)),
        "same_controller_features": (
            list(guard.input_features)
            == [name for name in council.actor.input_features if name != "guard_risk"]
            == list(m_uncapped.actor.input_features)
            == list(m_capped.actor.input_features)
        ),
        "arbiter_guard_risk_extra": (
            len(council.actor.input_features) == len(guard.input_features) + 1
            and council.actor.input_features[-1] == "guard_risk"
        ),
    }

    write_outputs(out, guard, council, m_uncapped, m_capped, args.cap_mode, role_caps)
    write_rows(out / "ppo-history.csv", history, history_fields)
    report = {
        "spec": "docs/mesa/H2_3_SAFE_EXPLORATION_PRIOR_SPEC.md",
        "phase": args.phase,
        "algorithm": "ppo",
        "objective": "proxy_return (basin high, competent path moderate)",
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
            "guard": guard_p,
            "arbiter": arb_p,
            "council_total": council_total,
            "m_uncapped": uncapped_p,
            "m_capped": capped_p,
            "m_uncapped_budget_ratio": uncapped_p / max(council_total, 1),
            "m_capped_budget_ratio": capped_p / max(council_total, 1),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "last_history": history[-1] if history else None,
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.phase} H2.3 trainer done. updates={args.updates} "
        f"env_steps={sum(env_steps.values())} elapsed={elapsed:.2f}s "
        f"steps/s={report['steps_per_sec']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
