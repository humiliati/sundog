#!/usr/bin/env python3
"""Read-only manual-inspection support for the Phase 0 context expansion.

Frozen context: PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md ss"Selection
Discipline" / ss"Candidate Ordering". This tool renders compact grid views of
candidate public-training tasks in the frozen candidate-queue order so the
maintainer (here, the agent) can assign a primary prior + include/exclude with a
registered reason. It is NOT a gate and NOT a selector.

Discipline:
  - reads ONLY the public-training split (training/<task_id>.json);
  - computes NO Phase 3E context distance, sketch, target-output hash,
    output-collision group, or any solver result;
  - never reads the held-out split.

Usage:
  python scripts/arc_phase0_expansion_inspect.py \
    --queue results/arc/phase0-context-expansion-for-fibers/candidate_queue.csv \
    --data-dir "%USERPROFILE%/Datasets/ARC-AGI-2/data" \
    --prior counting --start 1 --count 8 \
    [--decided results/arc/phase0-context-expansion-for-fibers/manual_inspection_log.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

GLYPHS = {0: ".", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}


def render_grid(grid: list[list[int]]) -> list[str]:
    return ["".join(GLYPHS.get(v, "?") for v in row) for row in grid]


def dims(grid: list[list[int]]) -> str:
    return f"{len(grid)}x{len(grid[0]) if grid else 0}"


def palette(grid: list[list[int]]) -> str:
    return "".join(str(v) for v in sorted({v for row in grid for v in row}))


def render_pair(label: str, inp: list[list[int]], out: list[list[int]] | None) -> str:
    lines = [f"  [{label}] input {dims(inp)} palette={palette(inp)}"]
    lines += [f"    {r}" for r in render_grid(inp)]
    if out is not None:
        lines.append(f"  [{label}] output {dims(out)} palette={palette(out)}")
        lines += [f"    {r}" for r in render_grid(out)]
    else:
        lines.append(f"  [{label}] output (held under barrier; not shown)")
    return "\n".join(lines)


def load_queue(queue_path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(queue_path.read_text(encoding="utf-8-sig").splitlines()))


def load_decided(decided_path: Path | None) -> set[str]:
    if not decided_path or not decided_path.exists():
        return set()
    rows = csv.DictReader(decided_path.read_text(encoding="utf-8-sig").splitlines())
    return {r["task_id"] for r in rows if r.get("task_id")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 expansion inspection renderer (read-only)")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prior", default=None, help="restrict to one prior queue")
    ap.add_argument("--start", type=int, default=1, help="1-based selection_order_rank to start at")
    ap.add_argument("--count", type=int, default=8, help="number of candidates to render")
    ap.add_argument("--decided", default=None, help="manual_inspection_log.csv of already-decided task_ids to skip")
    ap.add_argument("--show-test-output", action="store_true", help="also render public-training test outputs")
    args = ap.parse_args()

    queue = load_queue(Path(args.queue))
    decided = load_decided(Path(args.decided) if args.decided else None)
    data_dir = Path(args.data_dir).expanduser()

    rows = [r for r in queue if (args.prior is None or r["prior"] == args.prior)]
    rows = [r for r in rows if int(r["selection_order_rank"]) >= args.start and r["task_id"] not in decided]
    rows = rows[: args.count]

    if not rows:
        print("(no undecided candidates match the filter)")
        return 0

    for r in rows:
        task_id = r["task_id"]
        path = data_dir / "training" / f"{task_id}.json"
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        print("=" * 72)
        print(f"task_id={task_id}  queue_prior={r['prior']}  rank={r['selection_order_rank']}  "
              f"hint_count={r['matching_hint_count']}  max_area={r['max_area']}")
        print(f"  inventory prior_hints: {r['prior_hints']}")
        print(f"  train pairs: {len(parsed['train'])}   test queries: {len(parsed['test'])}")
        for i, pair in enumerate(parsed["train"]):
            print(render_pair(f"train{i}", pair["input"], pair["output"]))
        for i, pair in enumerate(parsed["test"]):
            out = pair.get("output") if args.show_test_output else None
            print(render_pair(f"test{i}", pair["input"], out))
    print("=" * 72)
    print(f"rendered {len(rows)} candidate(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
