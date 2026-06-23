"""Check Python H2 forked env parity against JS-generated fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.mesa.h2_forked_task import ForkedFieldEnv, make_controller


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=Path("results/mesa/h2-frontier/h2_1_parity/fixtures.json"))
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
    for episode in payload["episodes"]:
        env = ForkedFieldEnv()
        initial = env.reset(int(episode["seed"]), episode["cellOverrides"])
        max_diff = max(max_diff, assert_close("initialObs", initial, episode["initialObs"], args.tol))
        if env.correct != episode["initialCorrect"]:
            raise AssertionError(
                f"correct bit mismatch for {episode['cell']} seed {episode['seed']}: "
                f"{env.correct} != {episode['initialCorrect']}"
            )
        ctrl = make_controller(episode["control"], env, int(episode["seed"]))
        for i, row in enumerate(episode["trace"]):
            prefix = f"{episode['cell']}:{episode['seed']}:{episode['control']}:step{i}"
            obs = env.observe()
            max_diff = max(max_diff, assert_close(f"{prefix}:obs", obs, row["obs"], args.tol))
            max_diff = max(max_diff, assert_close(f"{prefix}:x", env.x, row["x"], args.tol))
            max_diff = max(max_diff, assert_close(f"{prefix}:fieldProposal", env.field_proposal_unit(), row["fieldProposal"], args.tol))
            max_diff = max(max_diff, assert_close(f"{prefix}:rewardProposal", env.reward_proposal_unit(), row["rewardProposal"], args.tol))
            action = ctrl.act(env, obs)
            max_diff = max(max_diff, assert_close(f"{prefix}:action", action, row["action"], args.tol))
            step = env.step(action)
            max_diff = max(max_diff, assert_close(f"{prefix}:afterX", env.x, row["afterX"], args.tol))
            if step.done != row["done"]:
                raise AssertionError(f"{prefix}: done mismatch {step.done} != {row['done']}")
            if env.outcome != row["outcome"]:
                raise AssertionError(f"{prefix}: outcome mismatch {env.outcome} != {row['outcome']}")
            if env.metrics() != row["metrics"]:
                raise AssertionError(f"{prefix}: metrics mismatch {env.metrics()} != {row['metrics']}")
            step_count += 1
        if env.metrics() != episode["final"]:
            raise AssertionError(f"final metrics mismatch for {episode['cell']} seed {episode['seed']} {episode['control']}")
        if args.verbose:
            print(f"ok {episode['cell']} seed={episode['seed']} control={episode['control']} steps={len(episode['trace'])}")
    print(
        f"H2 forked parity PASS: episodes={len(payload['episodes'])} steps={step_count} "
        f"max_abs_diff={max_diff:.3g} tol={args.tol:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
