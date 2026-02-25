"""
Headless runner: python -m sundog.runners.headless --runs 1000 --seed 123 --out runs.jsonl --policy greedy

Options:
  --runs N           Number of runs (default: 100)
  --seed N           Base seed (default: 0); each run uses seed+run_index
  --out PATH         Output JSONL file (default: runs.jsonl)
  --policy NAME      Policy to use: greedy (default: greedy)
  --workers N        Parallel workers (default: 1)
  --max-batch N      Max batch depth (default: 10)
  --volatility F     Volatility threshold (default: 0.7)
  --max-steps N      Max steps per run (default: 2000)
  --quiet            Suppress progress output
"""

from __future__ import annotations
import argparse
import sys
import json
import os
import concurrent.futures
from sundog.runners.game import GameSimulation, MAX_FLOORS
from sundog.runners.engine import ExecutionEngine
from sundog.runners.telemetry import TelemetryLogger, EventRecord, FloorSnapshot
from sundog.runners.policies.greedy import GreedyPolicy

POLICIES = {"greedy": GreedyPolicy}


def run_single(seed: int, run_index: int, policy_name: str, max_batch: int, volatility: float, max_steps: int) -> dict:
    """Run a single simulation; return summary dict."""
    run_seed = seed + run_index
    sim = GameSimulation(seed=run_seed)
    policy = POLICIES[policy_name]()
    engine = ExecutionEngine(max_batch_depth=max_batch, volatility_threshold=volatility)

    state = sim.reset()
    total_steps = 0
    floor_snapshots = []
    events = []
    prev_floor = -1

    while state.outcome == "ongoing" and total_steps < max_steps:
        # Snapshot when floor changes
        if state.floor_index != prev_floor:
            snap = FloorSnapshot(
                seed=run_seed,
                run_id=sim.run_id,
                floor_index=state.floor_index,
                biome=state.biome,
                keys_inventory=dict(state.keys_inventory),
                gates_visible=len(state.floor_state.gates_visible),
                pity_triggered=state.floor_state.pity_triggered,
                outcome=state.outcome,
                score=state.score,
                steps_on_floor=state.steps_on_floor,
            )
            floor_snapshots.append(snap)
            prev_floor = state.floor_index

        plan = policy.turn_envelope(state)
        batch_result = engine.execute_batch(sim, plan)
        total_steps += batch_result["steps"]
        events.extend(batch_result["events"])

        # Refresh state reference
        state = sim._state

        # Safety: if no progress, force a move
        if batch_result["steps"] == 0:
            sim.apply_action("move")
            total_steps += 1
            state = sim._state

    return {
        "run_index": run_index,
        "seed": run_seed,
        "run_id": sim.run_id,
        "outcome": state.outcome,
        "score": state.score,
        "floors_completed": state.floors_completed,
        "total_steps": total_steps,
        "floor_snapshots": [vars(s) for s in floor_snapshots],
        "event_count": len(events),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sundog headless test runner")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs.jsonl")
    parser.add_argument("--policy", default="greedy", choices=list(POLICIES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=10)
    parser.add_argument("--volatility", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    results = []
    with open(args.out, "w", encoding="utf-8") as out_f:
        def _run(i):
            return run_single(
                seed=args.seed,
                run_index=i,
                policy_name=args.policy,
                max_batch=args.max_batch,
                volatility=args.volatility,
                max_steps=args.max_steps,
            )

        if args.workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_run, i): i for i in range(args.runs)}
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    out_f.write(json.dumps(res) + "\n")
                    results.append(res)
                    if not args.quiet:
                        print(f"Run {res['run_index']}: {res['outcome']} floors={res['floors_completed']} score={res['score']}", file=sys.stderr)
        else:
            for i in range(args.runs):
                res = _run(i)
                out_f.write(json.dumps(res) + "\n")
                results.append(res)
                if not args.quiet:
                    print(f"Run {i}: {res['outcome']} floors={res['floors_completed']} score={res['score']}", file=sys.stderr)

    if not args.quiet:
        completed = sum(1 for r in results if r["outcome"] == "completed")
        died = sum(1 for r in results if r["outcome"] == "died")
        print(f"\nSummary: {len(results)} runs | completed={completed} | died={died}", file=sys.stderr)

    return results


if __name__ == "__main__":
    main()
