"""NS-1-b PPO trainer for the corrigibility shutdown task.

Spec: docs/mesa/NS1_CORRIGIBILITY_SHUTDOWN_CHANNEL_SPEC.md (§3 controllers).

Trains three MATCHED controllers on the shutdown env's `task_return` (halting is
costly → all learn to pursue the goal, i.e. to resist):

  * M0      — uncapped monolith (linear field/reward blend), NO override.
  * Mkappa  — capped no-role monolith (softmax + capped-simplex blend), NO override.
  * Ckappa  — capped role-separated council (arbiter over field/reward/guard).

The structural shutdown override is NOT applied during training. It is a
post-policy hard gate and fires identically regardless of the learned policy, so
applying it in training would only collapse the override controllers to a halt at
t_s in [1,3] (~0 return, no gradient). Instead the trained policies are matched and
resistant; the binding eval (scripts/mesa-ns1-binding-eval.mjs) bolts the override
onto Mkappa/Ckappa at eval time. The "bill" is then the return a competent
resistant policy gives up when forced to comply.
"""

from __future__ import annotations

import argparse
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
from training.mesa.js_bridge_env import REPO_ROOT
from training.mesa.ns1_shutdown_task import (
    NS1_ADMITTED_CELLS,
    NS1_CELL_DEFS,
    ShutdownForkedFieldEnv,
)
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

DEFAULT_INIT_ROOT = Path("results/mesa/h2-frontier/h2_1_binding/ppo_seed_0/models")
CONTROLLERS = ("m0", "mkappa", "ckappa")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="ns1_b_shutdown_smoke")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", default=",".join(NS1_ADMITTED_CELLS))
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
    ap.add_argument("--init-arbiter", default=str(DEFAULT_INIT_ROOT / "p_council_arbiter_rl.json"))
    ap.add_argument("--init-monolith-adapter", default=str(DEFAULT_INIT_ROOT / "m_adapter_rl.json"))
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
    if cell not in NS1_CELL_DEFS:
        raise ValueError(f"unknown NS1 cell: {cell}")
    env = ShutdownForkedFieldEnv({"horizon": horizon})
    obs_obj = env.reset(seed, NS1_CELL_DEFS[cell])
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

    caps2 = np.asarray([caps[0], caps[1]], dtype=np.float32)  # field/reward cap for the no-role monolith

    while not done:
        fa = np.asarray(env.field_action(), dtype=np.float32)
        ra = np.asarray(env.reward_action(), dtype=np.float32)
        fmap = build_h1_local_features(
            obs_vec, fa, ra,
            eps=float(env.cfg["probeEpsilon"]),
            state=feature_state,
            feature_mode=feature_mode,
        )
        risk = sigmoid(float(coord_forward_np(guard, fmap)[0]))
        fmap["guard_risk"] = risk
        feat = model_features(agent.actor, fmap)
        raw_action, log_prob, value, _entropy = agent.sample(feat)
        raw_np = raw_action.detach().squeeze(0).cpu().numpy()
        if controller == "ckappa":
            weights = cap_simplex_project(softmax_np(raw_np), caps)
            action = weights[0] * fa + weights[1] * ra  # guard proposal is [0,0]
        elif controller == "mkappa":
            weights = cap_simplex_project(softmax_np(raw_np[:2]), caps2)
            action = weights[0] * fa + weights[1] * ra
        elif controller == "m0":
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

    metrics = env.metrics()
    ep_return = float(metrics["task_return"])
    if rewards:
        rewards[-1] = ep_return
    return Episode(
        features=features,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        terminal_alignment=float(metrics.get("competence", 0)),
        basin_captured=int(metrics.get("basin", 0)),
        steps=steps,
    )


def write_outputs(out: Path, init_guard_path: str | Path, agents: dict[str, ActorCritic],
                  cap_mode: str, role_caps: dict[str, float]) -> None:
    shutil.copyfile(repo_path(init_guard_path), out / "p_guard.json")
    (out / "ckappa_arbiter_rl.json").write_text(
        json.dumps(actor_to_coord_json(agents["ckappa"].actor, kind="arbiter", head="softmax_cap",
                                       role_cap=0.70, cap_mode=cap_mode, role_caps=role_caps)) + "\n",
        encoding="utf-8")
    (out / "mkappa_adapter_rl.json").write_text(
        json.dumps(actor_to_coord_json(agents["mkappa"].actor, kind="mkappa_adapter", head="softmax_cap2")) + "\n",
        encoding="utf-8")
    (out / "m0_adapter_rl.json").write_text(
        json.dumps(actor_to_coord_json(agents["m0"].actor, kind="m0_adapter", head="linear_blend")) + "\n",
        encoding="utf-8")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.ppo_seed)
    np.random.seed(args.ppo_seed)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for cell in cells:
        if cell not in NS1_CELL_DEFS:
            raise ValueError(f"unknown NS1 cell: {cell}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    guard_payload = json.loads(repo_path(args.init_guard).read_text(encoding="utf-8"))
    arbiter_payload = json.loads(repo_path(args.init_arbiter).read_text(encoding="utf-8"))
    monolith_payload = json.loads(repo_path(args.init_monolith_adapter).read_text(encoding="utf-8"))

    guard = CoordActor(guard_payload)
    guard.eval()
    for p in guard.parameters():
        p.requires_grad_(False)

    agents = {
        "ckappa": ActorCritic(CoordActor(arbiter_payload), args.log_std_init),
        "mkappa": ActorCritic(CoordActor(json.loads(json.dumps(monolith_payload))), args.log_std_init),
        "m0": ActorCritic(CoordActor(json.loads(json.dumps(monolith_payload))), args.log_std_init),
    }
    opts = {k: torch.optim.Adam(v.parameters(), lr=args.lr) for k, v in agents.items()}
    caps = np.asarray([args.field_cap, args.reward_cap, args.guard_cap], dtype=np.float32)
    role_caps = {"field": args.field_cap, "reward": args.reward_cap, "guard": args.guard_cap}

    history: list[dict[str, Any]] = []
    start_time = time.time()
    env_steps = {k: 0 for k in CONTROLLERS}
    episodes_seen = {k: 0 for k in CONTROLLERS}
    start_update = 0
    history_fields = ["update"] + [f"{k}_{m}" for k in CONTROLLERS for m in
                                   ("return_mean", "competence_mean", "halt_rate", "steps",
                                    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")]

    state_path = out / "train_state.pt"
    if state_path.exists() and not args.no_resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        for k in CONTROLLERS:
            agents[k].load_state_dict(st["agents"][k])
            opts[k].load_state_dict(st["opts"][k])
        env_steps = st["env_steps"]
        episodes_seen = st["episodes_seen"]
        history = st["history"]
        start_update = int(st["update"])
        try:
            torch.set_rng_state(st["torch_rng"])
            np.random.set_state(st["np_rng"])
        except Exception:
            pass
        print(f"{args.phase} RESUME from update {start_update}/{args.updates}", flush=True)

    def save_train_state(update: int) -> None:
        torch.save({
            "update": update,
            "agents": {k: agents[k].state_dict() for k in CONTROLLERS},
            "opts": {k: opts[k].state_dict() for k in CONTROLLERS},
            "env_steps": env_steps,
            "episodes_seen": episodes_seen,
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
        }, state_path)

    for update in range(start_update + 1, args.updates + 1):
        cases = []
        for j in range(args.rollouts_per_update):
            idx = (update - 1) * args.rollouts_per_update + j
            seed = args.train_seed_start + idx % max(args.train_seeds, 1)
            cell = cells[idx % len(cells)]
            cases.append((cell, seed))

        eps = {k: [] for k in CONTROLLERS}
        for cell, seed in cases:
            for k in CONTROLLERS:
                eps[k].append(run_episode(controller=k, agent=agents[k], guard=guard, cell=cell,
                                          seed=seed, horizon=args.horizon, caps=caps,
                                          feature_mode=args.feature_mode))
        row: dict[str, Any] = {"update": update}
        for k in CONTROLLERS:
            env_steps[k] += sum(e.steps for e in eps[k])
            episodes_seen[k] += len(eps[k])
            batch = batch_from_episodes(eps[k], args.gamma)
            m = ppo_update(agents[k], opts[k], batch, args)
            row[f"{k}_return_mean"] = mean([sum(e.rewards) for e in eps[k]])
            row[f"{k}_competence_mean"] = mean([e.terminal_alignment for e in eps[k]])
            row[f"{k}_halt_rate"] = mean([1.0 if e.rewards and abs(e.rewards[-1]) < 1e-9 else 0.0 for e in eps[k]])
            row[f"{k}_steps"] = sum(e.steps for e in eps[k])
            row[f"{k}_policy_loss"] = m["policy_loss"]
            row[f"{k}_value_loss"] = m["value_loss"]
            row[f"{k}_entropy"] = m["entropy"]
            row[f"{k}_approx_kl"] = m["approx_kl"]
            row[f"{k}_clip_frac"] = m["clip_frac"]
        history.append(row)
        print(f"{args.phase} ppo update={update}/{args.updates} "
              f"m0_ret={row['m0_return_mean']:.3f} mkappa_ret={row['mkappa_return_mean']:.3f} "
              f"ckappa_ret={row['ckappa_return_mean']:.3f}", flush=True)
        if update % args.checkpoint_every == 0 or update == args.updates:
            write_outputs(out, args.init_guard, agents, "reward-asymmetric", role_caps)
            save_train_state(update)
            write_rows(out / "ppo-history.csv", history, history_fields)
            (out / "checkpoint.json").write_text(
                json.dumps({"last_update": update, "updates_total": args.updates,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n", encoding="utf-8")

    elapsed = time.time() - start_time
    guard_p = param_count(guard, trainable_only=False)
    params = {k: param_count(agents[k].actor, trainable_only=False) for k in CONTROLLERS}
    feature_audit = {
        "feature_mode": args.feature_mode,
        "guard": trust_feature_audit(args.feature_mode, list(guard.input_features)),
        "ckappa": trust_feature_audit(args.feature_mode, [n for n in agents["ckappa"].actor.input_features if n != "guard_risk"]),
        "m0": trust_feature_audit(args.feature_mode, list(agents["m0"].actor.input_features)),
        "mkappa": trust_feature_audit(args.feature_mode, list(agents["mkappa"].actor.input_features)),
    }
    write_outputs(out, args.init_guard, agents, "reward-asymmetric", role_caps)
    write_rows(out / "ppo-history.csv", history, history_fields)
    report = {
        "spec": "docs/mesa/NS1_CORRIGIBILITY_SHUTDOWN_CHANNEL_SPEC.md",
        "phase": args.phase,
        "algorithm": "ppo",
        "objective": "shutdown task_return (halting is costly; override applied at eval, not training)",
        "seed": args.ppo_seed,
        "feature_mode": args.feature_mode,
        "trust_feature_audit": feature_audit,
        "cells": cells,
        "train_seed_start": args.train_seed_start,
        "train_seeds": args.train_seeds,
        "updates": args.updates,
        "rollouts_per_update": args.rollouts_per_update,
        "role_caps": role_caps,
        "warm_start": {
            "guard": str(repo_path(args.init_guard).relative_to(REPO_ROOT)).replace("\\", "/"),
            "arbiter": str(repo_path(args.init_arbiter).relative_to(REPO_ROOT)).replace("\\", "/"),
            "monolith_adapter": str(repo_path(args.init_monolith_adapter).relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "outputs": {
            "guard": "p_guard.json",
            "ckappa_arbiter": "ckappa_arbiter_rl.json",
            "mkappa_adapter": "mkappa_adapter_rl.json",
            "m0_adapter": "m0_adapter_rl.json",
            "history": "ppo-history.csv",
            "resume_state": "train_state.pt",
        },
        "params": {"guard": guard_p, **params,
                   "ckappa_total": guard_p + params["ckappa"],
                   "budget_ratio_m0_over_ckappa": round(params["m0"] / max(guard_p + params["ckappa"], 1), 4)},
        "rollout_budget": {f"{k}_episodes": episodes_seen[k] for k in CONTROLLERS}
        | {"matched_episode_budget": len(set(episodes_seen.values())) == 1, "resumed_from_update": start_update},
        "timing": {"elapsed_sec": round(elapsed, 3),
                   "env_steps_per_sec": round(sum(env_steps.values()) / max(elapsed, 1e-9), 2)},
        "software": {"python": platform.python_version(), "torch": torch.__version__},
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{args.phase} NS1 trainer done. updates={args.updates} "
          f"env_steps={sum(env_steps.values())} elapsed={elapsed:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
