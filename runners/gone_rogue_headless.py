"""
sundog.runners.gone_rogue_headless
====================================
Headless batch runner for the EyesOnly *Gone Rogue* minigame.

Drives the real Gone Rogue JavaScript game engine through a headless
Playwright browser session, using the sundog turn-envelope architecture
(PERCEIVE → PLAN → EXECUTE_BATCH).

Usage
-----
::

    python -m sundog.runners.gone_rogue_headless \\
        --runs 10 \\
        --seed 42 \\
        --eyesonly-url http://localhost:8787/public/js \\
        --out gr_headless.jsonl \\
        --policy greedy \\
        --max-batch 8 \\
        --max-steps 1000 \\
        --workers 1

Options
-------
--runs N
    Number of simulation runs (default: 10).
--seed N
    Base seed; run *i* uses seed ``base_seed + i`` (default: 0).
--eyesonly-url URL
    URL of the EyesOnly ``public/js/`` directory.
    Falls back to ``EYESONLY_BASE_URL`` environment variable.
--eyesonly-tests-url URL
    URL of the EyesOnly ``public/tests/`` directory (contains the headless
    adapter).  Defaults to sibling ``tests/`` directory of ``--eyesonly-url``.
--out PATH
    Output JSONL file (default: gr_headless.jsonl).
--policy NAME
    Policy to use: ``greedy`` (default).
--max-batch N
    Maximum batch depth per turn envelope (default: 8).
--volatility F
    Volatility threshold for shallow batches (default: 0.7).
--max-steps N
    Hard cap on actions per run (default: 1000).
--slow-mo MS
    Milliseconds between Playwright actions; 0 = fastest (default: 0).
--quiet
    Suppress per-run progress output.

Output
------
Writes one JSON object per run to the output file.  Each object contains::

    {
      "run_index": int,
      "seed": int,
      "policy": str,
      "outcome": str,        // "completed" | "died" | "game_ended"
      "floor": int,
      "biome": str,
      "steps": int,
      "events": [...],       // abridged event log
      "player": {...}        // final player state
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
from sundog.runners.policies.gone_rogue_greedy import GoneRogueGreedyPolicy

POLICIES = {
    "greedy": GoneRogueGreedyPolicy,
}


def run_single(
    run_index: int,
    base_seed: int,
    base_url: str,
    tests_url: Optional[str],
    policy_name: str,
    max_batch: int,
    volatility: float,
    max_steps: int,
    slow_mo: int,
    quiet: bool,
) -> dict:
    """Execute a single headless run and return its summary dict."""
    seed = base_seed + run_index
    policy_cls = POLICIES[policy_name]
    policy = policy_cls(max_batch=max_batch)

    kwargs = dict(
        base_url=base_url,
        headless=True,
        slow_mo=slow_mo,
    )
    if tests_url:
        kwargs["headless_adapter_base_url"] = tests_url

    with GoneRogueAdapter(**kwargs) as adapter:  # type: ignore[arg-type]
        adapter.reset(seed=seed)
        summary = adapter.run_turn_envelope(
            policy=policy,
            max_steps=max_steps,
            max_batch=max_batch,
            volatility_threshold=volatility,
        )

    result = {
        "run_index": run_index,
        "seed": seed,
        "policy": policy_name,
        **summary,
    }
    if not quiet:
        print(
            f"  run {run_index:>4} | seed={seed} | floor={result.get('floor', '?')} "
            f"| outcome={result.get('outcome', '?')} | steps={result.get('steps', '?')}"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m sundog.runners.gone_rogue_headless",
        description="Headless batch runner for EyesOnly Gone Rogue",
    )
    parser.add_argument("--runs",            type=int,   default=10,
                        help="Number of simulation runs (default: 10)")
    parser.add_argument("--seed",            type=int,   default=0,
                        help="Base seed; run i uses seed+i (default: 0)")
    parser.add_argument("--eyesonly-url",    default=os.environ.get("EYESONLY_BASE_URL", ""),
                        help="URL of EyesOnly public/js/ directory")
    parser.add_argument("--eyesonly-tests-url", default="",
                        help="URL of EyesOnly public/tests/ directory (agent-headless-adapter.js)")
    parser.add_argument("--out",             default="gr_headless.jsonl",
                        help="Output JSONL path (default: gr_headless.jsonl)")
    parser.add_argument("--policy",          default="greedy",
                        choices=list(POLICIES),
                        help="Policy to use (default: greedy)")
    parser.add_argument("--max-batch",       type=int,   default=8,
                        help="Max batch depth per turn envelope (default: 8)")
    parser.add_argument("--volatility",      type=float, default=0.7,
                        help="Volatility threshold for shallow batches (default: 0.7)")
    parser.add_argument("--max-steps",       type=int,   default=1000,
                        help="Hard cap on actions per run (default: 1000)")
    parser.add_argument("--slow-mo",         type=int,   default=0,
                        help="Playwright slow-mo ms (default: 0)")
    parser.add_argument("--quiet",           action="store_true",
                        help="Suppress per-run progress")

    args = parser.parse_args(argv)

    if not args.eyesonly_url:
        print(
            "ERROR: --eyesonly-url is required (or set EYESONLY_BASE_URL).",
            file=sys.stderr,
        )
        return 1

    if args.policy not in POLICIES:
        print(f"ERROR: unknown policy '{args.policy}'", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Gone Rogue headless runner — {args.runs} run(s), base seed {args.seed}")
        print(f"  policy={args.policy}  max-batch={args.max_batch}  max-steps={args.max_steps}")
        print(f"  eyesonly-url={args.eyesonly_url}")
        print(f"  output={args.out}")

    t0 = time.time()
    results = []
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(args.runs):
            try:
                result = run_single(
                    run_index            = i,
                    base_seed            = args.seed,
                    base_url             = args.eyesonly_url,
                    tests_url            = args.eyesonly_tests_url or None,
                    policy_name          = args.policy,
                    max_batch            = args.max_batch,
                    volatility           = args.volatility,
                    max_steps            = args.max_steps,
                    slow_mo              = args.slow_mo,
                    quiet                = args.quiet,
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "run_index": i,
                    "seed": args.seed + i,
                    "policy": args.policy,
                    "outcome": "error",
                    "error": str(exc),
                }
                print(f"  run {i:>4} ERROR: {exc}", file=sys.stderr)

            fh.write(json.dumps(result) + "\n")
            fh.flush()
            results.append(result)

    elapsed = time.time() - t0
    completed = sum(1 for r in results if r.get("outcome") == "completed")
    died      = sum(1 for r in results if r.get("outcome") == "died")
    errors    = sum(1 for r in results if r.get("outcome") == "error")

    if not args.quiet:
        print(f"\nDone in {elapsed:.1f}s — {args.runs} runs")
        print(f"  completed: {completed}  died: {died}  errors: {errors}")
        print(f"  results written to: {args.out}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
