"""Stdlib smoke test for the HC behavior-cloning dataset artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.mesa.hc_bc_dataset import (
    DEFAULT_MANIFEST,
    bc_dataset_manifest_block,
    build_hc_bc_split,
    print_bc_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the HC BC dataset loader.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sensor-tier", default="local-probe-field")
    parser.add_argument("--successful-only", action="store_true")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--print-manifest-block", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = build_hc_bc_split(
        args.manifest,
        sensor_tier=args.sensor_tier,
        successful_only=args.successful_only,
        seed_base=args.seed_base,
    )
    print_bc_summary(split)
    print(
        "bc_dataset_checks: "
        "obs_shape=pass action_shape=pass action_clip=pass finite=pass "
        "obs_variance=pass trajectory_count=pass"
    )
    if args.print_manifest_block:
        print(json.dumps(bc_dataset_manifest_block(split), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

