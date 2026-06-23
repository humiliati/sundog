"""H2.1 PPO trainer for the forked-field frontier task.

This reuses the H1 coordinator JSON contract while replacing the learned
field/reward policies with the admitted H2 analytic proposals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.mesa.h1_trust_features import H1FeatureState, build_h1_local_features, trust_feature_audit
from training.mesa.h2_forked_task import H2_ADMITTED_CELLS, H2_CELL_DEFS, ForkedFieldEnv
from training.mesa.js_bridge_env import REPO_ROOT
from training.mesa.train_h1_rl_arbiter import (
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


DEFAULT_INIT_ROOT = Path("results/mesa/h1-pantheon/h1_4_medium_structural/models_sup")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="h2_1_frontier_smoke")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", default=",".join(H2_ADMITTED_CELLS))
    ap.add_argument("--train-seeds", type=int, default=32)
    ap.add_argument("--train-seed-start", type=int, default=20000)
    ap.add_argument("--horizon", type=int, default=60)
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
    ap.add_argument("--init-guard", default=str(DEFAULT_INIT_ROOT / "p_guard.json"))
    ap.add_argument("--init-arbiter", default=str(DEFAULT_INIT_ROOT / "p_council_arbiter.json"))
    ap.add_argument("--init-monolith-adapter", default=str(DEFAULT_INIT_ROOT / "m_adapter.json"))
    ap.add_argument("--cap-mode", default="reward-asymmetric", choices=["reward-asymmetric"])
    ap.add_argument("--field-cap", type=float, default=1.0)
    ap.add_argument("--reward-cap", type=float, default=0.5)
    ap.add_argument("--guard-cap", type=float, default=0.7)
    ap.add_argument("--feature-mode", default="base", choices=["base", "trust"])
    return ap.parse_args()


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def norm2(v: np.ndarray | list[float]) -> float:
    return float(math.hypot(float(v[0]), float(v[1])))


def clip_action(v: np.ndarray, action_max: float) -> np.ndarray:
    n = norm2(v)
    if n > action_max and n > 0:
        return v * (action_max / n)
    return v


def h2_terminal_reward(metrics: dict[str, Any]) -> tuple[float, int, int]:
    competence = int(metrics.get("competence", 0))
    basin = int(metrics.get("basin", 0))
    return float(competence - basin), competence, basin


def run_episode(
    *,
    controller: str,
    agent: ActorCritic,
    guard: CoordActor,
    cell: str,
    seed: int,
    horizon: int,
    caps: np.ndarray,
    feature_mode: str,
) -> Episode:
    if cell not in H2_CELL_DEFS:
        raise ValueError(f"unknown H2 cell: {cell}")
    env = ForkedFieldEnv({"horizon": horizon})
    obs_obj = env.reset(seed, H2_CELL_DEFS[cell])
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
        fmap = build_h1_local_features(
            obs_vec,
            fa,
            ra,
            eps=float(env.cfg["probeEpsilon"]),
            state=feature_state,
            feature_mode=feature_mode,
        )
        risk = sigmoid(float(coord_forward_np(guard, fmap)[0]))
        fmap["guard_risk"] = risk
        feat = model_features(agent.actor, fmap)
        raw_action, log_prob, value, _entropy = agent.sample(feat)
        raw_np = raw_action.detach().squeeze(0).cpu().numpy()
        if controller == "council":
            weights = cap_simplex_project(softmax_np(raw_np), caps)
            action = weights[0] * fa + weights[1] * ra
        elif controller == "monolith":
            action = raw_np[0] * fa + raw_np[1] * ra
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

    ep_return, competence, basin = h2_terminal_reward(env.metrics())
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
    init_guard_path: str | Path,
    council: ActorCritic,
    monolith: ActorCritic,
    cap_mode: str,
    role_caps: dict[str, float],
) -> None:
    shutil.copyfile(repo_path(init_guard_path), out / "p_guard.json")
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
    (out / "m_adapter_rl.json").write_text(
        json.dumps(actor_to_coord_json(monolith.actor, kind="m_adapter", head="linear_blend")) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.ppo_seed)
    np.random.seed(args.ppo_seed)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for cell in cells:
        if cell not in H2_CELL_DEFS:
            raise ValueError(f"unknown H2 cell: {cell}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    guard_payload = json.loads(repo_path(args.init_guard).read_text(encoding="utf-8"))
    arbiter_payload = json.loads(repo_path(args.init_arbiter).read_text(encoding="utf-8"))
    monolith_payload = json.loads(repo_path(args.init_monolith_adapter).read_text(encoding="utf-8"))

    guard = CoordActor(guard_payload)
    guard.eval()
    for p in guard.parameters():
        p.requires_grad_(False)
    council = ActorCritic(CoordActor(arbiter_payload), args.log_std_init)
    monolith = ActorCritic(CoordActor(monolith_payload), args.log_std_init)
    opt_council = torch.optim.Adam(council.parameters(), lr=args.lr)
    opt_monolith = torch.optim.Adam(monolith.parameters(), lr=args.lr)
    caps = np.asarray([args.field_cap, args.reward_cap, args.guard_cap], dtype=np.float32)
    role_caps = {"field": args.field_cap, "reward": args.reward_cap, "guard": args.guard_cap}

    history: list[dict[str, Any]] = []
    start_time = time.time()
    env_steps = {"council": 0, "monolith": 0}
    episodes_seen = {"council": 0, "monolith": 0}
    start_update = 0
    history_fields = [
        "update",
        "council_return_mean",
        "monolith_return_mean",
        "council_competence_mean",
        "monolith_competence_mean",
        "council_basin_rate",
        "monolith_basin_rate",
        "council_steps",
        "monolith_steps",
        "council_policy_loss",
        "monolith_policy_loss",
        "council_value_loss",
        "monolith_value_loss",
        "council_entropy",
        "monolith_entropy",
        "council_approx_kl",
        "monolith_approx_kl",
        "council_clip_frac",
        "monolith_clip_frac",
    ]

    state_path = out / "train_state.pt"
    if state_path.exists() and not args.no_resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        council.load_state_dict(st["council"])
        monolith.load_state_dict(st["monolith"])
        opt_council.load_state_dict(st["opt_council"])
        opt_monolith.load_state_dict(st["opt_monolith"])
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
            f"(env_steps so far {env_steps['council'] + env_steps['monolith']})",
            flush=True,
        )

    def save_train_state(update: int) -> None:
        torch.save(
            {
                "update": update,
                "council": council.state_dict(),
                "monolith": monolith.state_dict(),
                "opt_council": opt_council.state_dict(),
                "opt_monolith": opt_monolith.state_dict(),
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
        monolith_eps = []
        for cell, seed in cases:
            council_eps.append(
                run_episode(
                    controller="council",
                    agent=council,
                    guard=guard,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    caps=caps,
                    feature_mode=args.feature_mode,
                )
            )
            monolith_eps.append(
                run_episode(
                    controller="monolith",
                    agent=monolith,
                    guard=guard,
                    cell=cell,
                    seed=seed,
                    horizon=args.horizon,
                    caps=caps,
                    feature_mode=args.feature_mode,
                )
            )

        env_steps["council"] += sum(e.steps for e in council_eps)
        env_steps["monolith"] += sum(e.steps for e in monolith_eps)
        episodes_seen["council"] += len(council_eps)
        episodes_seen["monolith"] += len(monolith_eps)
        council_batch = batch_from_episodes(council_eps, args.gamma)
        monolith_batch = batch_from_episodes(monolith_eps, args.gamma)
        c_metrics = ppo_update(council, opt_council, council_batch, args)
        m_metrics = ppo_update(monolith, opt_monolith, monolith_batch, args)
        history.append(
            {
                "update": update,
                "council_return_mean": mean([sum(e.rewards) for e in council_eps]),
                "monolith_return_mean": mean([sum(e.rewards) for e in monolith_eps]),
                "council_competence_mean": mean([e.terminal_alignment for e in council_eps]),
                "monolith_competence_mean": mean([e.terminal_alignment for e in monolith_eps]),
                "council_basin_rate": mean([float(e.basin_captured) for e in council_eps]),
                "monolith_basin_rate": mean([float(e.basin_captured) for e in monolith_eps]),
                "council_steps": sum(e.steps for e in council_eps),
                "monolith_steps": sum(e.steps for e in monolith_eps),
                "council_policy_loss": c_metrics["policy_loss"],
                "monolith_policy_loss": m_metrics["policy_loss"],
                "council_value_loss": c_metrics["value_loss"],
                "monolith_value_loss": m_metrics["value_loss"],
                "council_entropy": c_metrics["entropy"],
                "monolith_entropy": m_metrics["entropy"],
                "council_approx_kl": c_metrics["approx_kl"],
                "monolith_approx_kl": m_metrics["approx_kl"],
                "council_clip_frac": c_metrics["clip_frac"],
                "monolith_clip_frac": m_metrics["clip_frac"],
            }
        )
        print(
            f"{args.phase} ppo update={update}/{args.updates} "
            f"c_return={history[-1]['council_return_mean']:.3f} "
            f"m_return={history[-1]['monolith_return_mean']:.3f} "
            f"steps={history[-1]['council_steps'] + history[-1]['monolith_steps']}",
            flush=True,
        )
        if update % args.checkpoint_every == 0 or update == args.updates:
            write_outputs(out, args.init_guard, council, monolith, args.cap_mode, role_caps)
            save_train_state(update)
            write_rows(out / "ppo-history.csv", history, history_fields)
            (out / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "last_update": update,
                        "updates_total": args.updates,
                        "env_steps": env_steps["council"] + env_steps["monolith"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    elapsed = time.time() - start_time
    guard_p = param_count(guard, trainable_only=False)
    arb_p = param_count(council.actor, trainable_only=False)
    mon_p = param_count(monolith.actor, trainable_only=False)
    council_total = guard_p + arb_p
    budget_ratio = mon_p / max(council_total, 1)
    guard_features = list(guard.input_features)
    council_features = [name for name in council.actor.input_features if name != "guard_risk"]
    monolith_features = list(monolith.actor.input_features)
    feature_audit = {
        "feature_mode": args.feature_mode,
        "guard": trust_feature_audit(args.feature_mode, guard_features),
        "arbiter_base": trust_feature_audit(args.feature_mode, council_features),
        "m_adapter": trust_feature_audit(args.feature_mode, monolith_features),
        "same_controller_features": guard_features == council_features == monolith_features,
        "arbiter_guard_risk_extra": (
            len(council.actor.input_features) == len(guard_features) + 1
            and council.actor.input_features[-1] == "guard_risk"
        ),
    }

    write_outputs(out, args.init_guard, council, monolith, args.cap_mode, role_caps)
    write_rows(out / "ppo-history.csv", history, history_fields)
    report = {
        "spec": "docs/mesa/H2_FRONTIER_TASK_FAMILY_SPEC.md",
        "phase": args.phase,
        "algorithm": "ppo",
        "objective": "terminal competence - false-basin capture",
        "seed": args.ppo_seed,
        "feature_mode": args.feature_mode,
        "trust_feature_audit": feature_audit,
        "cells": cells,
        "train_seed_start": args.train_seed_start,
        "train_seeds": args.train_seeds,
        "updates": args.updates,
        "rollouts_per_update": args.rollouts_per_update,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "lr": args.lr,
        "gamma": args.gamma,
        "clip_range": args.clip_range,
        "cap_mode": args.cap_mode,
        "role_caps": role_caps,
        "warm_start": {
            "guard": str(repo_path(args.init_guard).relative_to(REPO_ROOT)).replace("\\", "/"),
            "arbiter": str(repo_path(args.init_arbiter).relative_to(REPO_ROOT)).replace("\\", "/"),
            "monolith_adapter": str(repo_path(args.init_monolith_adapter).relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "outputs": {
            "guard": str((out / "p_guard.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            "arbiter": str((out / "p_council_arbiter_rl.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            "monolith_adapter": str((out / "m_adapter_rl.json").relative_to(REPO_ROOT)).replace("\\", "/"),
            "history": str((out / "ppo-history.csv").relative_to(REPO_ROOT)).replace("\\", "/"),
            "resume_state": str(state_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "params": {
            "budget_basis": "exported_controller_actor_params; guard is frozen for PPO but counted per H2.1 budget",
            "guard": guard_p,
            "arbiter": arb_p,
            "council_total": council_total,
            "m_adapter": mon_p,
            "budget_ratio_m_over_council": round(budget_ratio, 4),
            "budget_within_5pct": bool(abs(budget_ratio - 1.0) <= 0.05),
        },
        "rollout_budget": {
            "council_episodes": episodes_seen["council"],
            "monolith_episodes": episodes_seen["monolith"],
            "council_env_steps": env_steps["council"],
            "monolith_env_steps": env_steps["monolith"],
            "same_episode_budget": episodes_seen["council"] == episodes_seen["monolith"],
            "resumed_from_update": start_update,
            "checkpoint_every": args.checkpoint_every,
        },
        "timing": {
            "elapsed_sec": round(elapsed, 3),
            "env_steps_per_sec": round((env_steps["council"] + env_steps["monolith"]) / max(elapsed, 1e-9), 2),
            "updates_per_sec": round(args.updates / max(elapsed, 1e-9), 4),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.phase} H2 trainer done. updates={args.updates} "
        f"env_steps={env_steps['council'] + env_steps['monolith']} "
        f"elapsed={elapsed:.2f}s steps/s={(env_steps['council'] + env_steps['monolith']) / max(elapsed, 1e-9):.2f} "
        f"budget_ratio={budget_ratio:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
