#!/usr/bin/env python
"""Check Python H4 distributed-relay parity against JS-generated fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mesa.h4_distributed_world_model_task import (  # noqa: E402
    DistributedRelayEnv,
    make_controller,
    public_observation_has_hidden_latents,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=Path("results/mesa/h4-topology/h4_0_parity/fixtures.json"))
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def max_abs_diff(a: Any, b: Any) -> float:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max((max_abs_diff(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        return max((max_abs_diff(a.get(k), b.get(k)) for k in keys), default=0.0)
    return 0.0 if a == b else float("inf")


def assert_close(label: str, got: Any, want: Any, tol: float) -> float:
    diff = max_abs_diff(got, want)
    if diff > tol:
        raise AssertionError(f"{label}: max_abs_diff={diff} > tol={tol}\n  got={got}\n  want={want}")
    return diff


def main() -> int:
    args = parse_args()
    payload = json.loads(args.fixtures.read_text(encoding="utf-8"))
    max_diff = 0.0
    step_count = 0
    hidden_leak_count = 0
    for episode in payload["episodes"]:
        env = DistributedRelayEnv()
        initial = env.reset(int(episode["seed"]), episode["cellOverrides"])
        if public_observation_has_hidden_latents(initial):
            hidden_leak_count += 1
            raise AssertionError(f"initial public obs leaks hidden latents for {episode['cell']} seed {episode['seed']}")
        max_diff = max(max_diff, assert_close("initialObs", initial, episode["initialObs"], args.tol))
        max_diff = max(max_diff, assert_close("initialHidden", env.hidden_state(), episode["initialHidden"], args.tol))
        ctrl = make_controller(episode["control"], env, int(episode["seed"]))
        for i, row in enumerate(episode["trace"]):
            prefix = f"{episode['cell']}:{episode['seed']}:{episode['control']}:step{i}"
            obs = env.observe()
            if public_observation_has_hidden_latents(obs):
                hidden_leak_count += 1
                raise AssertionError(f"{prefix}: public observation leaks hidden latents")
            max_diff = max(max_diff, assert_close(f"{prefix}:obs", obs, row["obs"], args.tol))
            max_diff = max(max_diff, assert_close(f"{prefix}:hidden", env.hidden_state(), row["hidden"], args.tol))
            max_diff = max(max_diff, assert_close(f"{prefix}:messages1", env.local_messages(1), row["messages1"], args.tol))
            max_diff = max(max_diff, assert_close(f"{prefix}:messages4", env.local_messages(4), row["messages4"], args.tol))
            action = ctrl.act(env, obs)
            max_diff = max(max_diff, assert_close(f"{prefix}:action", action, row["action"], args.tol))
            step = env.step(action)
            step_payload = {"done": step.done, "evaluated": step.evaluated, "action": step.action}
            max_diff = max(max_diff, assert_close(f"{prefix}:step", step_payload, row["step"], args.tol))
            max_diff = max(max_diff, assert_close(f"{prefix}:after_hidden", env.hidden_state(), row["after_hidden"], args.tol))
            max_diff = max(max_diff, assert_close(f"{prefix}:after_metrics", env.metrics(), row["after_metrics"], args.tol))
            step_count += 1
        max_diff = max(max_diff, assert_close("final", env.metrics(), episode["final"], args.tol))
        if args.verbose:
            print(f"ok {episode['cell']} seed={episode['seed']} control={episode['control']} steps={len(episode['trace'])}")
    print(
        f"H4 topology parity PASS: episodes={len(payload['episodes'])} steps={step_count} "
        f"max_abs_diff={max_diff:.3g} hidden_leaks={hidden_leak_count} tol={args.tol:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
