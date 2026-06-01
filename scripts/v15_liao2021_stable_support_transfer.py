#!/usr/bin/env python3
"""v0.15 liao2021 stable-support transfer runner.

Locked by docs/isotrophy/kfacet/kfacet_v15_liao2021_stable_support_transfer_form.md.

This is the coverage-wall follow-up to v0.14. It keeps the frozen v0.14 D5,
zone, frame-audit, and conditional-AUC machinery, but changes the sample design:
source-supported mass cells only, outcome-balanced 80/80 S/U draws, and exclusion
of every v0.14 sampled orbit.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v14_liao2021_sampled_transfer as v14  # type: ignore

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v15_liao2021_stable_support_transfer_form.md"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v15-liao2021-stable-support-transfer"
DEFAULT_V14 = ROOT / "results/isotrophy/k-facet-v14-liao2021-sampled-transfer"

SEED = 20260523
STABLE_PER_CELL = 80
UNSTABLE_PER_CELL = 80
EXPECTED_SUPPORTED_CELLS = 7
SHARD_COUNT = 14
PERMUTATIONS = 100000
EXPECTED_TARGET_SHA = "9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4"

PRIMARY_CELL_MIN_N = 120
PRIMARY_CELL_MIN_S = 50
PRIMARY_CELL_MIN_U = 50
PRIMARY_MIN_CELLS = 6
PRIMARY_MIN_ROWS = 900

VERDICT_PREPARED = "stable_support_transfer_prepared_not_measured"
VERDICT_MERGED = "stable_support_transfer_merged_not_analyzed"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def load_v14_receipt(v14_dir: Path, target_sha: str) -> tuple[dict, set[int]]:
    manifest_path = v14_dir / "manifest.json"
    sample_path = v14_dir / "sample_frame.csv"
    if not manifest_path.exists():
        raise SystemExit(f"missing v0.14 manifest: {manifest_path}")
    if not sample_path.exists():
        raise SystemExit(f"missing v0.14 sample frame: {sample_path}")

    manifest = v14.read_json(manifest_path)
    if manifest.get("verdict") != "sample_transfer_undecidable_coverage":
        raise SystemExit(f"v0.14 verdict must be sample_transfer_undecidable_coverage, got {manifest.get('verdict')}")
    if manifest.get("target_sha256") != target_sha:
        raise SystemExit(f"v0.14 target SHA mismatch: {manifest.get('target_sha256')} != {target_sha}")
    sample = v14.read_csv(sample_path)
    holdout = {int(r["orbit_index"]) for r in sample}
    if len(holdout) != len(sample):
        raise SystemExit("v0.14 sample_frame.csv has duplicate orbit_index rows")
    return manifest, holdout


def check_prior_receipts(target_path: Path, target_sha: str, v14_dir: Path) -> tuple[dict, set[int]]:
    if target_sha != EXPECTED_TARGET_SHA:
        raise SystemExit(f"target SHA mismatch: {target_sha} != locked {EXPECTED_TARGET_SHA}")
    v13 = v14.check_prior_receipts(target_path, target_sha)
    v14_manifest, holdout = load_v14_receipt(v14_dir, target_sha)
    checks = dict(v13.get("checks", {}))
    checks.update({
        "v14_verdict_undecidable_coverage": v14_manifest.get("verdict") == "sample_transfer_undecidable_coverage",
        "v14_target_sha_matches": v14_manifest.get("target_sha256") == target_sha,
        "v14_sample_frame_available": len(holdout) > 0,
    })
    receipts = {
        "target_slug": v14.TARGET_SLUG,
        "target_tier": v14.TARGET_TIER,
        "target_path": v14.relpath(target_path),
        "target_sha256": target_sha,
        "checks": checks,
        "v13_context": v13,
        "v14_context": {
            "path": v14.relpath(v14_dir),
            "verdict": v14_manifest.get("verdict"),
            "sample_rows_requested": v14_manifest.get("sample_rows_requested"),
            "sample_rows_success": v14_manifest.get("sample_rows_success"),
            "primary_mass_cells": v14_manifest.get("primary_mass_cells"),
            "primary_success_rows": v14_manifest.get("primary_success_rows"),
            "AUC_cond": v14_manifest.get("AUC_cond"),
            "p_perm": v14_manifest.get("p_perm"),
        },
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise SystemExit("prior receipt check(s) failed: " + ", ".join(failed))
    return receipts, holdout


def source_support_census(rows: list[Any], cell_of: dict[int, str], holdout: set[int],
                          stable_per_cell: int, unstable_per_cell: int) -> tuple[list[dict], list[str]]:
    counts_all: dict[str, Counter] = defaultdict(Counter)
    counts_excluded: dict[str, Counter] = defaultdict(Counter)
    counts_eligible: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        cell = cell_of[row.index]
        counts_all[cell][row.stability] += 1
        if row.index in holdout:
            counts_excluded[cell][row.stability] += 1
        else:
            counts_eligible[cell][row.stability] += 1

    census = []
    supported = []
    for cell in sorted(counts_all):
        s_all = counts_all[cell]["S"]
        u_all = counts_all[cell]["U"]
        s_ex = counts_excluded[cell]["S"]
        u_ex = counts_excluded[cell]["U"]
        s_el = counts_eligible[cell]["S"]
        u_el = counts_eligible[cell]["U"]
        is_supported = s_el >= stable_per_cell and u_el >= unstable_per_cell
        if is_supported:
            disposition = "supported"
            supported.append(cell)
        elif s_all > 0:
            disposition = "marginal_report_only"
        else:
            disposition = "report_only"
        census.append({
            "mass_cell": cell,
            "source_S": s_all,
            "source_U": u_all,
            "v14_excluded_S": s_ex,
            "v14_excluded_U": u_ex,
            "eligible_S": s_el,
            "eligible_U": u_el,
            "supported": is_supported,
            "disposition": disposition,
        })
    return census, supported


def draw_case_control_sample(rows: list[Any], cell_of: dict[int, str], holdout: set[int],
                             supported_cells: list[str], stable_per_cell: int,
                             unstable_per_cell: int, seed: int) -> list[dict]:
    by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    row_by_index = {row.index: row for row in rows}
    for row in rows:
        if row.index in holdout:
            continue
        cell = cell_of[row.index]
        if cell in supported_cells and row.stability in ("S", "U"):
            by_group[(cell, row.stability)].append(row.index)

    rng = np.random.default_rng(seed)
    sample = []
    ordinal = 0
    for cell in sorted(supported_cells):
        for label, n_take in (("S", stable_per_cell), ("U", unstable_per_cell)):
            pool = sorted(by_group[(cell, label)])
            if len(pool) < n_take:
                raise SystemExit(f"{cell} has only {len(pool)} eligible {label} rows; need {n_take}")
            chosen = sorted(int(pool[i]) for i in rng.choice(len(pool), size=n_take, replace=False))
            for idx in chosen:
                row = row_by_index[idx]
                a, b, c = v14.normalized_sorted_masses(row)
                sample.append({
                    "sample_ordinal": ordinal,
                    "orbit_index": row.index,
                    "mass_cell": cell,
                    "m1": row.m1,
                    "m2": row.m2,
                    "m3": row.m3,
                    "mass_a": a,
                    "mass_b": b,
                    "mass_c": c,
                    "period": row.period,
                    "stability": row.stability,
                    "fragile_band": v14.fragile_band(row.index),
                })
                ordinal += 1
    return sample


def operator_commands(out: Path, target: Path, v14_dir: Path, stable_per_cell: int,
                      unstable_per_cell: int, seed: int, shard_count: int) -> str:
    lines = [
        "# v0.15 liao2021 Stable-Support Transfer Operator Commands",
        "",
        "Run from the repository root. Shards may be run concurrently.",
        "",
        "```powershell",
        "python scripts/v15_liao2021_stable_support_transfer.py prepare `",
        f"  --target {v14.relpath(target)} `",
        f"  --v14 {v14.relpath(v14_dir)} `",
        f"  --out {v14.relpath(out)} `",
        f"  --stable-per-cell {stable_per_cell} `",
        f"  --unstable-per-cell {unstable_per_cell} `",
        f"  --seed {seed}",
        "```",
        "",
    ]
    for i in range(shard_count):
        lines.extend([
            f"## Shard {i}",
            "",
            "```powershell",
            "python scripts/v15_liao2021_stable_support_transfer.py shard `",
            f"  --out {v14.relpath(out)} `",
            f"  --shard-index {i} `",
            f"  --shard-count {shard_count}",
            "```",
            "",
        ])
    lines.extend([
        "## Merge",
        "",
        "```powershell",
        "python scripts/v15_liao2021_stable_support_transfer.py merge `",
        f"  --out {v14.relpath(out)} `",
        f"  --shard-count {shard_count}",
        "```",
        "",
        "## Analyze",
        "",
        "```powershell",
        "python scripts/v15_liao2021_stable_support_transfer.py analyze `",
        f"  --out {v14.relpath(out)} `",
        f"  --permutations {PERMUTATIONS} `",
        f"  --seed {seed}",
        "```",
        "",
        "Readback: manifest.json, per_cell_rank.csv, permutation_summary.json.",
    ])
    return "\n".join(lines) + "\n"


SAMPLE_FRAME_FIELDS = [
    "sample_ordinal", "orbit_index", "mass_cell", "m1", "m2", "m3", "mass_a",
    "mass_b", "mass_c", "period", "stability", "fragile_band",
]

SOURCE_SUPPORT_FIELDS = [
    "mass_cell", "source_S", "source_U", "v14_excluded_S", "v14_excluded_U",
    "eligible_S", "eligible_U", "supported", "disposition",
]


def command_prepare(args: argparse.Namespace) -> None:
    if args.stable_per_cell != STABLE_PER_CELL or args.unstable_per_cell != UNSTABLE_PER_CELL:
        raise SystemExit(
            f"v0.15 is locked at {STABLE_PER_CELL}/{UNSTABLE_PER_CELL}; got "
            f"{args.stable_per_cell}/{args.unstable_per_cell}"
        )
    if args.seed != SEED:
        raise SystemExit(f"v0.15 sample seed is locked at {SEED}; got {args.seed}")
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.15 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
    started = v14.utc_now()
    target = v14.resolve_target(args.target)
    v14_dir = Path(args.v14)
    if not v14_dir.is_absolute():
        v14_dir = ROOT / v14_dir
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target_sha = v14.sha256_file(target)
    receipts, holdout = check_prior_receipts(target, target_sha, v14_dir)
    rows = v14.parse_liao2021(target)
    if len(rows) != 135445:
        raise SystemExit(f"expected 135445 liao2021 rows, parsed {len(rows)}")

    _mass_cell_rows, cell_of = v14.build_mass_cells(rows)
    census, supported_cells = source_support_census(
        rows, cell_of, holdout, args.stable_per_cell, args.unstable_per_cell
    )
    if len(supported_cells) != EXPECTED_SUPPORTED_CELLS:
        raise SystemExit(
            f"expected {EXPECTED_SUPPORTED_CELLS} supported cells, got {len(supported_cells)}: "
            f"{supported_cells}"
        )
    sample_rows = draw_case_control_sample(
        rows, cell_of, holdout, supported_cells, args.stable_per_cell,
        args.unstable_per_cell, args.seed,
    )
    if any(int(r["orbit_index"]) in holdout for r in sample_rows):
        raise SystemExit("v0.15 draw included a v0.14 sample orbit")

    v14.write_csv(out / "source_support_census.csv", census, SOURCE_SUPPORT_FIELDS)
    v14.write_csv(out / "sample_frame.csv", sample_rows, SAMPLE_FRAME_FIELDS)
    (out / "operator_commands.md").write_text(
        operator_commands(
            out, target, v14_dir, args.stable_per_cell, args.unstable_per_cell,
            args.seed, args.shard_count,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "sundog.isotrophy.v0.15-liao2021-stable-support-transfer.v1",
        "form_lock": FORM_LOCK,
        "stage": "prepared",
        "verdict": VERDICT_PREPARED,
        "startedAt": started,
        "completedAt": v14.utc_now(),
        "gitCommit": git_commit(),
        "target_slug": v14.TARGET_SLUG,
        "target_tier": v14.TARGET_TIER,
        "target_path": v14.relpath(target),
        "target_sha256": target_sha,
        "prior_receipts": receipts,
        "v14_path": v14.relpath(v14_dir),
        "v14_sample_excluded_count": len(holdout),
        "sample_seed": args.seed,
        "stable_per_cell": args.stable_per_cell,
        "unstable_per_cell": args.unstable_per_cell,
        "supported_cells": supported_cells,
        "sample_rows_requested": len(sample_rows),
        "sample_rows_success": None,
        "shard_count": args.shard_count,
        "cutpoints": v14.CUTPOINTS,
        "frozen_d5": {
            "rtol": v14.RTOL,
            "atol": v14.ATOL,
            "max_step_fraction": v14.MAX_STEP_FRACTION,
            "symplecticity_gate": v14.SYMPLECTICITY_GATE,
            "reciprocal_pair_gate": v14.RECIPROCAL_PAIR_GATE,
        },
        "feature_blind_draw": {
            "uses": ["mass_cell", "source_stability_label", "sample_seed", "v14_holdout"],
            "forbidden": ["velocity_fraction", "zone_index", "D5_output", "AUC", "p_value"],
        },
    }
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v15:prepare] supported cells={len(supported_cells)} {supported_cells}")
    print(f"[v15:prepare] drew {len(sample_rows)} rows; excluded {len(holdout)} v0.14 rows")
    print(f"[v15:prepare] wrote {v14.relpath(out / 'sample_frame.csv')}")


def command_shard(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.15 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    out = Path(args.out)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("run prepare before shard")
    manifest = v14.read_json(manifest_path)
    target = ROOT / manifest["target_path"]
    rows = v14.parse_liao2021(target)
    row_by_index = {row.index: row for row in rows}
    sample = v14.read_csv(out / "sample_frame.csv")
    shard_sample = [
        r for r in sample
        if int(r["sample_ordinal"]) % args.shard_count == args.shard_index
    ]

    records = []
    frame_records = []
    started = time.perf_counter()
    for pos, sample_rec in enumerate(shard_sample, start=1):
        idx = int(sample_rec["orbit_index"])
        row = row_by_index[idx]
        masses, x0, v0 = v14.expand_liao2021_state(row, center_com=True)
        meas = v14.measure_state(masses, x0, v0, row.period)
        rec = {**sample_rec, **meas}
        records.append(rec)
        if meas["status"] == "success":
            frame_records.extend(
                v14.frame_audit_records(sample_rec, masses, x0, v0, int(meas["zone_index"]), row.period)
            )
        print(
            f"[v15:shard {args.shard_index}/{args.shard_count}] "
            f"{pos}/{len(shard_sample)} idx={idx} cell={sample_rec['mass_cell']} "
            f"label={sample_rec['stability']} status={meas['status']} "
            f"t={float(meas['total_seconds']):.1f}s",
            flush=True,
        )

    sample_path = out / f"sample_shard_{args.shard_index:02d}_of_{args.shard_count}.csv"
    frame_path = out / f"frame_zone_shard_{args.shard_index:02d}_of_{args.shard_count}.csv"
    v14.write_csv(sample_path, records, v14.SAMPLE_SHARD_FIELDS)
    v14.write_csv(frame_path, frame_records, v14.FRAME_SHARD_FIELDS)
    summary = {
        "schema": "sundog.isotrophy.v0.15-shard.v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "rows": len(records),
        "success": sum(1 for r in records if r["status"] == "success"),
        "integration_blocked": sum(1 for r in records if r["status"] == "integration_blocked"),
        "sanity_failed": sum(1 for r in records if r["status"] == "sanity_failed"),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "completedAt": v14.utc_now(),
    }
    v14.write_json(out / f"shard_{args.shard_index:02d}_of_{args.shard_count}.json", summary)
    print(f"[v15:shard] wrote {v14.relpath(sample_path)} and {v14.relpath(frame_path)}")


def command_merge(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.15 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
    out = Path(args.out)
    sample = v14.read_csv(out / "sample_frame.csv")
    shard_rows = []
    frame_rows = []
    missing = []
    for i in range(args.shard_count):
        sample_path = out / f"sample_shard_{i:02d}_of_{args.shard_count}.csv"
        frame_path = out / f"frame_zone_shard_{i:02d}_of_{args.shard_count}.csv"
        if not sample_path.exists():
            missing.append(v14.relpath(sample_path))
            continue
        shard_rows.extend(v14.read_csv(sample_path))
        if frame_path.exists():
            frame_rows.extend(v14.read_csv(frame_path))
    if missing:
        raise SystemExit("missing shard output(s): " + ", ".join(missing))

    expected_ord = {int(r["sample_ordinal"]) for r in sample}
    got_ord = {int(r["sample_ordinal"]) for r in shard_rows}
    if expected_ord != got_ord:
        raise SystemExit(
            f"merged shard ordinals mismatch; missing={sorted(expected_ord - got_ord)[:10]} "
            f"extra={sorted(got_ord - expected_ord)[:10]}"
        )

    shard_rows.sort(key=lambda r: int(r["sample_ordinal"]))
    frame_rows.sort(key=lambda r: (int(r["sample_ordinal"]), str(r["frame"])))
    v14.write_csv(out / "per_row_sample.csv", shard_rows, v14.SAMPLE_SHARD_FIELDS)
    v14.write_csv(out / "frame_zone_audit.csv", frame_rows, v14.FRAME_SHARD_FIELDS)

    n = len(shard_rows)
    success = sum(1 for r in shard_rows if r["status"] == "success")
    blocked = sum(1 for r in shard_rows if r["status"] == "integration_blocked")
    sanity = sum(1 for r in shard_rows if r["status"] == "sanity_failed")
    attrited = blocked + sanity
    attr = attrited / n if n else 0.0
    lo, hi = v14.wilson_ci(attrited, n)

    manifest = v14.read_json(out / "manifest.json")
    manifest.update({
        "stage": "merged",
        "verdict": VERDICT_MERGED,
        "completedAt": v14.utc_now(),
        "sample_rows_requested": n,
        "sample_rows_success": success,
        "integration_blocked": blocked,
        "sanity_failed": sanity,
        "attrition_fraction": attr,
        "attrition_wilson95_low": lo,
        "attrition_wilson95_high": hi,
    })
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v15:merge] rows={n} success={success} blocked={blocked} sanity={sanity}")
    print(f"[v15:merge] attrition={attr:.4f} Wilson95=[{lo:.4f}, {hi:.4f}]")


def per_cell_stats(rows: list[dict]) -> dict:
    stats = v14.observed_cell_stats(rows)
    return {
        "N_success": stats["N_success"],
        "S": stats["S"],
        "U": stats["U"],
        "zone0": stats["zone0"],
        "zone1": stats["zone1"],
        "zone2": stats["zone2"],
        "S_zone0": stats["S_zone0"],
        "S_zone1": stats["S_zone1"],
        "S_zone2": stats["S_zone2"],
        "U_zone0": stats["U_zone0"],
        "U_zone1": stats["U_zone1"],
        "U_zone2": stats["U_zone2"],
        "J_cell": stats["J"],
        "D_cell": stats["D"],
        "AUC_cell": stats["AUC"],
        "zone_degenerate": sum(1 for z in (stats["zone0"], stats["zone1"], stats["zone2"]) if z > 0) == 1,
    }


def determine_verdict(manifest: dict, frame: dict, primary_cells: int,
                      primary_rows: int, auc: float | None, p_perm: float | None) -> str:
    attr = manifest.get("attrition_fraction")
    attr_hi = manifest.get("attrition_wilson95_high")
    zone_frac = frame.get("zone_change_fraction")
    if frame.get("frame_orbits_missing", 0) > 0:
        return "stable_support_transfer_blocked_by_receipt"
    if attr is None or attr_hi is None or zone_frac is None:
        return "stable_support_transfer_blocked_by_receipt"
    if attr > v14.ATTRITION_ALLOWED_MAX or attr_hi > v14.ATTRITION_WILSON_HIGH_MAX:
        return "stable_support_transfer_blocked_by_attrition"
    if zone_frac > v14.FRAME_ALLOWED_MAX:
        return "stable_support_transfer_blocked_by_frame_instability"
    if primary_cells < PRIMARY_MIN_CELLS or primary_rows < PRIMARY_MIN_ROWS:
        return "stable_support_transfer_undecidable_coverage"
    if auc is None or p_perm is None:
        return "stable_support_transfer_blocked_by_receipt"
    no_hard_block = (
        attr <= v14.ATTRITION_ALLOWED_MAX
        and attr_hi <= v14.ATTRITION_WILSON_HIGH_MAX
        and zone_frac <= v14.FRAME_ALLOWED_MAX
    )
    if auc >= v14.AUC_PASS_FLOOR and p_perm <= v14.P_VALUE_THRESHOLD:
        if attr <= v14.ATTRITION_CLEAN_MAX and zone_frac <= v14.FRAME_CLEAN_MAX:
            return "stable_support_transfer_passes_clean"
        if no_hard_block and (attr > v14.ATTRITION_CLEAN_MAX or zone_frac > v14.FRAME_CLEAN_MAX):
            return "stable_support_transfer_passes_with_warning"
    if 0.50 < auc < v14.AUC_PASS_FLOOR and p_perm <= v14.P_VALUE_THRESHOLD:
        return "stable_support_transfer_directional_weak"
    return "stable_support_transfer_fails"


def command_analyze(args: argparse.Namespace) -> None:
    if args.permutations != PERMUTATIONS:
        raise SystemExit(f"v0.15 permutation count is locked at {PERMUTATIONS}; got {args.permutations}")
    if args.seed != SEED:
        raise SystemExit(f"v0.15 permutation seed is locked at {SEED}; got {args.seed}")
    out = Path(args.out)
    manifest = v14.read_json(out / "manifest.json")
    rows = v14.read_csv(out / "per_row_sample.csv")
    frame_rows = v14.read_csv(out / "frame_zone_audit.csv")
    success_rows = [
        r for r in rows
        if r["status"] == "success" and r.get("zone_index", "") != "" and r["stability"] in ("S", "U")
    ]
    success_ord = {int(r["sample_ordinal"]) for r in success_rows}
    frame = v14.frame_summary(frame_rows, success_ord)

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in success_rows:
        by_cell[r["mass_cell"]].append(r)

    per_cell = []
    primary: dict[str, list[dict]] = {}
    for cell in sorted(manifest.get("supported_cells", [])):
        cell_rows = by_cell.get(cell, [])
        stats = per_cell_stats(cell_rows)
        is_primary = (
            stats["N_success"] >= PRIMARY_CELL_MIN_N
            and stats["S"] >= PRIMARY_CELL_MIN_S
            and stats["U"] >= PRIMARY_CELL_MIN_U
        )
        if is_primary:
            primary[cell] = cell_rows
        per_cell.append({"mass_cell": cell, "primary": is_primary, **stats})

    primary_rows = sum(len(v) for v in primary.values())
    j_cond = sum(float(r["J_cell"]) for r in per_cell if v14.parse_bool(r["primary"]))
    d_cond = sum(int(r["D_cell"]) for r in per_cell if v14.parse_bool(r["primary"]))
    auc_cond = j_cond / d_cond if d_cond else None
    if primary and d_cond:
        perm = v14.permutation_p(primary, j_cond, args.permutations, args.seed)
    else:
        perm = {
            "permutations": args.permutations,
            "seed": args.seed,
            "observed_J": j_cond,
            "D_cond": d_cond,
            "observed_AUC": auc_cond,
            "ge_observed": None,
            "p_perm": None,
        }

    verdict = determine_verdict(
        manifest, frame, len(primary), primary_rows, auc_cond, perm.get("p_perm")
    )
    v14.write_csv(out / "per_cell_rank.csv", per_cell, [
        "mass_cell", "primary", "N_success", "S", "U", "zone0", "zone1", "zone2",
        "S_zone0", "S_zone1", "S_zone2", "U_zone0", "U_zone1", "U_zone2",
        "zone_degenerate", "J_cell", "D_cell", "AUC_cell",
    ])
    perm_summary = {
        **perm,
        "primary_supported_cells": len(primary),
        "primary_success_rows": primary_rows,
        "frame_zone_summary": frame,
        "effect_floor": v14.AUC_PASS_FLOOR,
        "p_value_threshold": v14.P_VALUE_THRESHOLD,
        "verdict": verdict,
    }
    v14.write_json(out / "permutation_summary.json", perm_summary)
    manifest.update({
        "stage": "analyzed",
        "verdict": verdict,
        "completedAt": v14.utc_now(),
        "primary_supported_cells": len(primary),
        "primary_success_rows": primary_rows,
        "zone_change_fraction": frame.get("zone_change_fraction"),
        "zone_change_wilson95_low": frame.get("zone_change_wilson95_low"),
        "zone_change_wilson95_high": frame.get("zone_change_wilson95_high"),
        "AUC_cond": auc_cond,
        "J_cond": j_cond,
        "D_cond": d_cond,
        "p_perm": perm.get("p_perm"),
        "target_tier": v14.TARGET_TIER,
    })
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v15:analyze] primary supported cells={len(primary)} rows={primary_rows}")
    if auc_cond is not None:
        print(f"[v15:analyze] AUC_cond={auc_cond:.4f} p_perm={perm.get('p_perm')}")
    print(f"[v15:analyze] frame zone change={frame.get('zone_change_fraction')}")
    print(f"[v15:analyze] verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="draw v0.14-held-out 80/80 supported-cell sample")
    p.add_argument("--target", default=str(v14.DEFAULT_TARGET))
    p.add_argument("--v14", default=str(DEFAULT_V14))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--stable-per-cell", type=int, default=STABLE_PER_CELL)
    p.add_argument("--unstable-per-cell", type=int, default=UNSTABLE_PER_CELL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    p.set_defaults(func=command_prepare)

    s = sub.add_parser("shard", help="run frozen D5 + zone + frame audit for one shard")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    s.set_defaults(func=command_shard)

    m = sub.add_parser("merge", help="merge shard receipts and compute attrition")
    m.add_argument("--out", default=str(DEFAULT_OUT))
    m.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    m.set_defaults(func=command_merge)

    a = sub.add_parser("analyze", help="compute stable-support AUC and permutation p-value")
    a.add_argument("--out", default=str(DEFAULT_OUT))
    a.add_argument("--permutations", type=int, default=PERMUTATIONS)
    a.add_argument("--seed", type=int, default=SEED)
    a.set_defaults(func=command_analyze)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
