"""
sundog.runners.gone_rogue_ui
==============================
Full UI-bound runner for the EyesOnly *Gone Rogue* minigame.

Opens a **visible** Playwright browser window, loads the Gone Rogue game, and
drives it using the sundog turn-envelope architecture.  This is the "human-UI
bound" runner — intended for manual inspection, screenshot capture, and
debugging sessions.

Usage
-----
::

    python -m sundog.runners.gone_rogue_ui \\
        --runs 3 \\
        --seed 42 \\
        --eyesonly-url http://localhost:8787/public/js \\
        --slow-mo 300 \\
        --out gr_ui.jsonl \\
        --screenshot-dir screenshots/

Options
-------
--runs N
    Number of runs (default: 3).
--seed N
    Base seed (default: 0).
--eyesonly-url URL
    URL of the EyesOnly ``public/js/`` directory.
--eyesonly-tests-url URL
    URL of EyesOnly ``public/tests/`` directory.
--out PATH
    Output JSONL (default: gr_ui.jsonl).
--policy NAME
    Policy: ``greedy`` (default).
--max-batch N
    Maximum batch depth (default: 6; shorter for UI mode to stay visible).
--max-steps N
    Hard cap on actions per run (default: 400).
--slow-mo MS
    Milliseconds between Playwright actions; higher = easier to watch
    (default: 200).
--screenshot-dir DIR
    If provided, save a PNG screenshot at the end of each run to this
    directory.  Directory is created if it doesn't exist.
--keep-open
    Leave the browser open after each run so you can inspect state.
    Press Enter in the terminal to continue to the next run.
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


def run_single_ui(
    run_index: int,
    base_seed: int,
    base_url: str,
    tests_url: Optional[str],
    policy_name: str,
    max_batch: int,
    max_steps: int,
    slow_mo: int,
    screenshot_dir: Optional[str],
    keep_open: bool,
) -> dict:
    """Execute a single visible-browser run and return its summary dict."""
    seed = base_seed + run_index
    policy_cls = POLICIES[policy_name]
    policy = policy_cls(max_batch=max_batch)

    kwargs = dict(
        base_url=base_url,
        headless=False,     # ← visible browser
        slow_mo=slow_mo,
    )
    if tests_url:
        kwargs["headless_adapter_base_url"] = tests_url

    print(f"\n  ▶  Run {run_index} | seed={seed} | opening browser…")

    adapter = GoneRogueAdapter(**kwargs)  # type: ignore[arg-type]
    adapter._start_browser()

    try:
        adapter.reset(seed=seed)
        print(f"     Game started on floor 1.")

        summary = adapter.run_turn_envelope(
            policy=policy,
            max_steps=max_steps,
            max_batch=max_batch,
        )

        result = {
            "run_index": run_index,
            "seed": seed,
            "policy": policy_name,
            **summary,
        }

        print(
            f"  ✓  Run {run_index} done | floor={result.get('floor')} "
            f"| outcome={result.get('outcome')} | steps={result.get('steps')}"
        )

        # Screenshot
        if screenshot_dir:
            os.makedirs(screenshot_dir, exist_ok=True)
            shot_path = os.path.join(screenshot_dir, f"run_{run_index:04d}_seed{seed}.png")
            try:
                adapter._page.screenshot(path=shot_path, full_page=True)
                print(f"     Screenshot saved: {shot_path}")
            except Exception as exc:
                print(f"     Screenshot failed: {exc}")

        # Keep open for inspection
        if keep_open:
            input(f"     Browser open for inspection — press Enter to continue…")

    finally:
        adapter.close()

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m sundog.runners.gone_rogue_ui",
        description="UI-bound runner for EyesOnly Gone Rogue (visible browser)",
    )
    parser.add_argument("--runs",                type=int,   default=3)
    parser.add_argument("--seed",                type=int,   default=0)
    parser.add_argument("--eyesonly-url",        default=os.environ.get("EYESONLY_BASE_URL", ""),
                        help="URL of EyesOnly public/js/ directory")
    parser.add_argument("--eyesonly-tests-url",  default="",
                        help="URL of EyesOnly public/tests/ directory")
    parser.add_argument("--out",                 default="gr_ui.jsonl")
    parser.add_argument("--policy",              default="greedy", choices=list(POLICIES))
    parser.add_argument("--max-batch",           type=int,   default=6,
                        help="Max batch depth (default: 6 for UI visibility)")
    parser.add_argument("--max-steps",           type=int,   default=400)
    parser.add_argument("--slow-mo",             type=int,   default=200,
                        help="ms between Playwright actions (default: 200)")
    parser.add_argument("--screenshot-dir",      default="",
                        help="Save end-of-run screenshots to this directory")
    parser.add_argument("--keep-open",           action="store_true",
                        help="Pause before closing browser after each run")

    args = parser.parse_args(argv)

    if not args.eyesonly_url:
        print(
            "ERROR: --eyesonly-url is required (or set EYESONLY_BASE_URL).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Gone Rogue UI runner — {args.runs} run(s), seed {args.seed}")
    print(f"  policy={args.policy}  slow-mo={args.slow_mo}ms")
    print(f"  eyesonly-url={args.eyesonly_url}")
    print(f"  output={args.out}")

    results = []
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(args.runs):
            try:
                result = run_single_ui(
                    run_index     = i,
                    base_seed     = args.seed,
                    base_url      = args.eyesonly_url,
                    tests_url     = args.eyesonly_tests_url or None,
                    policy_name   = args.policy,
                    max_batch     = args.max_batch,
                    max_steps     = args.max_steps,
                    slow_mo       = args.slow_mo,
                    screenshot_dir= args.screenshot_dir or None,
                    keep_open     = args.keep_open,
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "run_index": i,
                    "seed": args.seed + i,
                    "policy": args.policy,
                    "outcome": "error",
                    "error": str(exc),
                }
                print(f"  Run {i} ERROR: {exc}", file=sys.stderr)

            fh.write(json.dumps(result) + "\n")
            fh.flush()
            results.append(result)

    completed = sum(1 for r in results if r.get("outcome") == "completed")
    died      = sum(1 for r in results if r.get("outcome") == "died")
    errors    = sum(1 for r in results if r.get("outcome") == "error")
    print(f"\nDone — {args.runs} runs | completed: {completed}  died: {died}  errors: {errors}")
    print(f"Results written to: {args.out}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
