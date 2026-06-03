#!/usr/bin/env python3
"""v0.18 liao2021 reliability-conditioned per-cell AUC runner.

Locked by docs/isotrophy/kfacet/kfacet_v18_liao2021_reliability_auc_form.md.

This chapter widens the liao2021 mass-cell map and tests whether label-blind
frame reliability predicts per-cell AUC. It reuses the frozen v0.16/v0.17
four-frame ensemble score and adds only the 8x8 grid, quadruple holdout, and
cell-level reliability/reversal analysis.
"""
from __future__ import annotations

import argparse
import math
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
import scripts.v15_liao2021_stable_support_transfer as v15  # type: ignore
import scripts.v16_liao2021_tail_resolved_transfer as v16  # type: ignore

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v18_liao2021_reliability_auc_form.md"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v18-liao2021-reliability-auc"
DEFAULT_V14 = ROOT / "results/isotrophy/k-facet-v14-liao2021-sampled-transfer"
DEFAULT_V15 = ROOT / "results/isotrophy/k-facet-v15-liao2021-stable-support-transfer"
DEFAULT_V16 = ROOT / "results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer"
DEFAULT_V17 = ROOT / "results/isotrophy/k-facet-v17-liao2021-heterogeneity"

SEED = 20260523
GRID_PARTS = 8
EXPECTED_SUPPORTED_CELLS = 18
EXPECTED_GRID_SCAN = {4: 7, 5: 10, 6: 12, 7: 16, 8: 18, 10: 24, 12: 34, 16: 46}
STABLE_PER_CELL = 80
UNSTABLE_PER_CELL = 80
SHARD_COUNT = 16
PERMUTATIONS = 100000
EXPECTED_TARGET_SHA = "9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4"
FRAME_DEGREES = v16.FRAME_DEGREES

PRIMARY_CELL_MIN_S = 50
PRIMARY_CELL_MIN_U = 50
PRIMARY_MIN_CELLS = 16
PRIMARY_MIN_ROWS = 2400

FRAME_WARN_MEDIAN = 0.20
FRAME_WARN_P90 = 0.75
FRAME_FRAGILE_P90 = 0.10
RELIABILITY_RHO_FLOOR = 0.45
RELIABILITY_P_THRESHOLD = 0.05
REVERSAL_AUC_FLOOR = 0.45
REVERSAL_P_THRESHOLD = 0.05

VERDICT_PREPARED = "reliability_auc_prepared_not_measured"
VERDICT_MERGED = "reliability_auc_merged_not_analyzed"

SOURCE_SUPPORT_FIELDS = [
    "mass_cell", "qA", "qB", "row_count", "a_min", "a_max", "b_min", "b_max",
    "orbit_index_min", "orbit_index_max",
    "source_S", "source_U",
    "v14_excluded_S", "v14_excluded_U",
    "v15_excluded_S", "v15_excluded_U",
    "v16_excluded_S", "v16_excluded_U",
    "v17_excluded_S", "v17_excluded_U",
    "eligible_S", "eligible_U", "supported", "disposition",
]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def resolve_dir(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_sample_holdout(result_dir: Path, expected_verdict: str, target_sha: str, label: str) -> tuple[dict, set[int]]:
    manifest_path = result_dir / "manifest.json"
    sample_path = result_dir / "sample_frame.csv"
    if not manifest_path.exists():
        raise SystemExit(f"missing {label} manifest: {manifest_path}")
    if not sample_path.exists():
        raise SystemExit(f"missing {label} sample frame: {sample_path}")
    manifest = v14.read_json(manifest_path)
    if manifest.get("verdict") != expected_verdict:
        raise SystemExit(f"{label} verdict must be {expected_verdict}, got {manifest.get('verdict')}")
    if manifest.get("target_sha256") != target_sha:
        raise SystemExit(f"{label} target SHA mismatch: {manifest.get('target_sha256')} != {target_sha}")
    sample = v14.read_csv(sample_path)
    holdout = {int(r["orbit_index"]) for r in sample}
    if len(holdout) != len(sample):
        raise SystemExit(f"{label} sample_frame.csv has duplicate orbit_index rows")
    return manifest, holdout


def check_prior_receipts(target_path: Path, target_sha: str, v14_dir: Path, v15_dir: Path,
                         v16_dir: Path, v17_dir: Path) -> tuple[dict, dict[str, set[int]]]:
    if target_sha != EXPECTED_TARGET_SHA:
        raise SystemExit(f"target SHA mismatch: {target_sha} != locked {EXPECTED_TARGET_SHA}")
    v13 = v14.check_prior_receipts(target_path, target_sha)
    v14_manifest, holdout14 = load_sample_holdout(
        v14_dir, "sample_transfer_undecidable_coverage", target_sha, "v0.14"
    )
    v15_manifest, holdout15 = load_sample_holdout(
        v15_dir, "stable_support_transfer_directional_weak", target_sha, "v0.15"
    )
    v16_manifest, holdout16 = load_sample_holdout(
        v16_dir, "tail_resolved_transfer_passes_clean", target_sha, "v0.16"
    )
    v17_manifest, holdout17 = load_sample_holdout(
        v17_dir, "heterogeneous_transfer_replicates_clean", target_sha, "v0.17"
    )

    holdouts = {"v14": holdout14, "v15": holdout15, "v16": holdout16, "v17": holdout17}
    overlaps = {}
    labels = sorted(holdouts)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            overlap = holdouts[a] & holdouts[b]
            if overlap:
                overlaps[f"{a}_{b}"] = sorted(overlap)[:10]
    if overlaps:
        raise SystemExit(f"holdout overlap unexpectedly non-empty: {overlaps}")

    checks = dict(v13.get("checks", {}))
    checks.update({
        "v14_verdict_undecidable_coverage": v14_manifest.get("verdict") == "sample_transfer_undecidable_coverage",
        "v15_verdict_directional_weak": v15_manifest.get("verdict") == "stable_support_transfer_directional_weak",
        "v16_verdict_passes_clean": v16_manifest.get("verdict") == "tail_resolved_transfer_passes_clean",
        "v17_verdict_heterogeneous_transfer_replicates_clean": (
            v17_manifest.get("verdict") == "heterogeneous_transfer_replicates_clean"
        ),
        "v14_target_sha_matches": v14_manifest.get("target_sha256") == target_sha,
        "v15_target_sha_matches": v15_manifest.get("target_sha256") == target_sha,
        "v16_target_sha_matches": v16_manifest.get("target_sha256") == target_sha,
        "v17_target_sha_matches": v17_manifest.get("target_sha256") == target_sha,
        "v14_sample_frame_available": len(holdout14) > 0,
        "v15_sample_frame_available": len(holdout15) > 0,
        "v16_sample_frame_available": len(holdout16) > 0,
        "v17_sample_frame_available": len(holdout17) > 0,
        "quadruple_holdout_disjoint": True,
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
        },
        "v15_context": {
            "path": v14.relpath(v15_dir),
            "verdict": v15_manifest.get("verdict"),
            "sample_rows_requested": v15_manifest.get("sample_rows_requested"),
        },
        "v16_context": {
            "path": v14.relpath(v16_dir),
            "verdict": v16_manifest.get("verdict"),
            "sample_rows_requested": v16_manifest.get("sample_rows_requested"),
            "AUC_cond": v16_manifest.get("AUC_cond"),
            "p_perm": v16_manifest.get("p_perm"),
        },
        "v17_context": {
            "path": v14.relpath(v17_dir),
            "verdict": v17_manifest.get("verdict"),
            "sample_rows_requested": v17_manifest.get("sample_rows_requested"),
            "AUC_cond": v17_manifest.get("AUC_cond"),
            "p_perm": v17_manifest.get("p_perm"),
            "heterogeneity_rho": v17_manifest.get("heterogeneity_rho"),
            "heterogeneity_p_rho": v17_manifest.get("heterogeneity_p_rho"),
        },
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise SystemExit("prior receipt check(s) failed: " + ", ".join(failed))
    return receipts, holdouts


def build_mass_cells_parts(rows: list[Any], parts: int) -> tuple[list[dict], dict[int, str]]:
    mass_recs = []
    for row in rows:
        a, b, c = v14.normalized_sorted_masses(row)
        mass_recs.append({"orbit_index": row.index, "a": a, "b": b, "c": c})

    by_index = {r["orbit_index"]: r for r in mass_recs}
    sorted_by_a = sorted(by_index, key=lambda idx: (by_index[idx]["a"], by_index[idx]["b"], idx))
    cell_of: dict[int, str] = {}
    cell_rows = []
    for qa, a_indices in enumerate(v14.split_evenly(sorted_by_a, parts)):
        sorted_by_b = sorted(a_indices, key=lambda idx: (by_index[idx]["b"], by_index[idx]["a"], idx))
        for qb, b_indices in enumerate(v14.split_evenly(sorted_by_b, parts)):
            key = f"mass_qA{qa}_qB{qb}"
            for idx in b_indices:
                cell_of[idx] = key
            vals_a = [by_index[idx]["a"] for idx in b_indices]
            vals_b = [by_index[idx]["b"] for idx in b_indices]
            cell_rows.append({
                "mass_cell": key,
                "qA": qa,
                "qB": qb,
                "row_count": len(b_indices),
                "a_min": min(vals_a) if vals_a else "",
                "a_max": max(vals_a) if vals_a else "",
                "b_min": min(vals_b) if vals_b else "",
                "b_max": max(vals_b) if vals_b else "",
                "orbit_index_min": min(b_indices) if b_indices else "",
                "orbit_index_max": max(b_indices) if b_indices else "",
            })
    return cell_rows, cell_of


def support_count(rows: list[Any], cell_of: dict[int, str], holdout: set[int],
                  stable_per_cell: int, unstable_per_cell: int) -> int:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row.index in holdout:
            continue
        counts[cell_of[row.index]][row.stability] += 1
    return sum(1 for cell_counts in counts.values()
               if cell_counts["S"] >= stable_per_cell and cell_counts["U"] >= unstable_per_cell)


def grid_scan(rows: list[Any], holdout: set[int], stable_per_cell: int,
              unstable_per_cell: int) -> list[dict]:
    out = []
    for parts in sorted(EXPECTED_GRID_SCAN):
        _cell_rows, cell_of = build_mass_cells_parts(rows, parts)
        supported = support_count(rows, cell_of, holdout, stable_per_cell, unstable_per_cell)
        out.append({
            "grid_parts": parts,
            "grid": f"{parts}x{parts}",
            "supported_cells": supported,
            "rows_at_80_80": supported * (stable_per_cell + unstable_per_cell),
            "locked_expected_supported_cells": EXPECTED_GRID_SCAN[parts],
            "matches_lock": supported == EXPECTED_GRID_SCAN[parts],
        })
    return out


def source_support_census(rows: list[Any], cell_rows: list[dict], cell_of: dict[int, str],
                          holdouts: dict[str, set[int]], stable_per_cell: int,
                          unstable_per_cell: int) -> tuple[list[dict], list[str]]:
    counts_all: dict[str, Counter] = defaultdict(Counter)
    counts_excluded: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    counts_eligible: dict[str, Counter] = defaultdict(Counter)
    combined = set().union(*holdouts.values())
    for row in rows:
        cell = cell_of[row.index]
        counts_all[cell][row.stability] += 1
        for label, excluded in holdouts.items():
            if row.index in excluded:
                counts_excluded[cell][label][row.stability] += 1
        if row.index not in combined:
            counts_eligible[cell][row.stability] += 1

    meta = {r["mass_cell"]: r for r in cell_rows}
    census = []
    supported = []
    for cell in sorted(meta):
        s_el = counts_eligible[cell]["S"]
        u_el = counts_eligible[cell]["U"]
        is_supported = s_el >= stable_per_cell and u_el >= unstable_per_cell
        if is_supported:
            disposition = "supported"
            supported.append(cell)
        elif counts_all[cell]["S"] > 0:
            disposition = "marginal_report_only"
        else:
            disposition = "report_only"
        rec = {
            **meta[cell],
            "source_S": counts_all[cell]["S"],
            "source_U": counts_all[cell]["U"],
            "v14_excluded_S": counts_excluded[cell]["v14"]["S"],
            "v14_excluded_U": counts_excluded[cell]["v14"]["U"],
            "v15_excluded_S": counts_excluded[cell]["v15"]["S"],
            "v15_excluded_U": counts_excluded[cell]["v15"]["U"],
            "v16_excluded_S": counts_excluded[cell]["v16"]["S"],
            "v16_excluded_U": counts_excluded[cell]["v16"]["U"],
            "v17_excluded_S": counts_excluded[cell]["v17"]["S"],
            "v17_excluded_U": counts_excluded[cell]["v17"]["U"],
            "eligible_S": s_el,
            "eligible_U": u_el,
            "supported": is_supported,
            "disposition": disposition,
        }
        census.append(rec)
    return census, supported


def operator_commands(out: Path, shard_count: int) -> str:
    lines = [
        "# v0.18 liao2021 Reliability AUC Operator Commands",
        "",
        "Run from the repository root. Shards may be run concurrently. The prepare step",
        "is source/sample only; shard steps perform the long four-frame D5 measurement.",
        "",
        "```powershell",
        "npm run isotrophy:v18:prepare",
        "```",
        "",
    ]
    for i in range(shard_count):
        lines.extend([
            f"## Shard {i}",
            "",
            "```powershell",
            f"npm run isotrophy:v18:shard -- --shard-index {i} --shard-count {shard_count}",
            "```",
            "",
        ])
    lines.extend([
        "## Merge",
        "",
        "```powershell",
        "npm run isotrophy:v18:merge",
        "```",
        "",
        "## Analyze",
        "",
        "```powershell",
        "npm run isotrophy:v18:analyze",
        "```",
        "",
        f"Output directory: `{v14.relpath(out)}`",
        "Readback: manifest.json, per_cell_auc.csv, frame_reliability_map.csv,",
        "reliability_auc_test.json, reversal_guard.json, anatomy_sidecar.csv.",
    ])
    return "\n".join(lines) + "\n"


def command_prepare(args: argparse.Namespace) -> None:
    if args.stable_per_cell != STABLE_PER_CELL or args.unstable_per_cell != UNSTABLE_PER_CELL:
        raise SystemExit(
            f"v0.18 is locked at {STABLE_PER_CELL}/{UNSTABLE_PER_CELL}; got "
            f"{args.stable_per_cell}/{args.unstable_per_cell}"
        )
    if args.seed != SEED:
        raise SystemExit(f"v0.18 sample seed is locked at {SEED}; got {args.seed}")
    if args.grid_parts != GRID_PARTS:
        raise SystemExit(f"v0.18 grid parts is locked at {GRID_PARTS}; got {args.grid_parts}")
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.18 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")

    started = v14.utc_now()
    target = v14.resolve_target(args.target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target_sha = v14.sha256_file(target)
    receipts, holdouts = check_prior_receipts(
        target, target_sha, resolve_dir(args.v14), resolve_dir(args.v15),
        resolve_dir(args.v16), resolve_dir(args.v17),
    )
    holdout = set().union(*holdouts.values())
    rows = v14.parse_liao2021(target)
    if len(rows) != 135445:
        raise SystemExit(f"expected 135445 liao2021 rows, parsed {len(rows)}")

    scan = grid_scan(rows, holdout, args.stable_per_cell, args.unstable_per_cell)
    mismatches = [r for r in scan if not v14.parse_bool(r["matches_lock"])]
    if mismatches:
        raise SystemExit(f"source grid scan differs from lock: {mismatches}")

    # Anchor against the v0.14-v0.17 construction: parts=4 must reproduce 7 cells.
    cell_rows4, cell_of4 = build_mass_cells_parts(rows, 4)
    legacy_cell_rows, legacy_cell_of = v14.build_mass_cells(rows)
    if len(cell_rows4) != len(legacy_cell_rows) or cell_of4 != legacy_cell_of:
        raise SystemExit("generalized split_evenly grid does not reproduce v0.14-v0.17 4x4 cell assignment")

    cell_rows, cell_of = build_mass_cells_parts(rows, args.grid_parts)
    census, supported_cells = source_support_census(
        rows, cell_rows, cell_of, holdouts, args.stable_per_cell, args.unstable_per_cell
    )
    if len(supported_cells) != EXPECTED_SUPPORTED_CELLS:
        raise SystemExit(
            f"expected {EXPECTED_SUPPORTED_CELLS} supported cells, got {len(supported_cells)}: "
            f"{supported_cells}"
        )

    sample_rows = v15.draw_case_control_sample(
        rows, cell_of, holdout, supported_cells, args.stable_per_cell,
        args.unstable_per_cell, args.seed,
    )
    if any(int(r["orbit_index"]) in holdout for r in sample_rows):
        raise SystemExit("v0.18 draw included a v0.14/v0.15/v0.16/v0.17 holdout orbit")

    v14.write_csv(out / "source_grid_scan.csv", scan, [
        "grid_parts", "grid", "supported_cells", "rows_at_80_80",
        "locked_expected_supported_cells", "matches_lock",
    ])
    v14.write_csv(out / "source_support_census.csv", census, SOURCE_SUPPORT_FIELDS)
    v14.write_csv(out / "sample_frame.csv", sample_rows, v16.SAMPLE_FRAME_FIELDS)
    (out / "operator_commands.md").write_text(operator_commands(out, args.shard_count), encoding="utf-8")

    manifest = {
        "schema": "sundog.isotrophy.v0.18-liao2021-reliability-auc.v1",
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
        "v14_path": args.v14,
        "v15_path": args.v15,
        "v16_path": args.v16,
        "v17_path": args.v17,
        "v14_sample_excluded_count": len(holdouts["v14"]),
        "v15_sample_excluded_count": len(holdouts["v15"]),
        "v16_sample_excluded_count": len(holdouts["v16"]),
        "v17_sample_excluded_count": len(holdouts["v17"]),
        "quadruple_holdout_excluded_count": len(holdout),
        "grid_builder": "generalized v0.14 build_mass_cells: sort a, split_evenly(parts), sort b, split_evenly(parts)",
        "grid_parts": args.grid_parts,
        "source_grid_scan": scan,
        "sample_seed": args.seed,
        "stable_per_cell": args.stable_per_cell,
        "unstable_per_cell": args.unstable_per_cell,
        "supported_cells": supported_cells,
        "sample_rows_requested": len(sample_rows),
        "sample_rows_success": None,
        "shard_count": args.shard_count,
        "frame_degrees": FRAME_DEGREES,
        "score": "median(vf_0,vf_37,vf_90,vf_211)",
        "frozen_d5": {
            "rtol": v14.RTOL,
            "atol": v14.ATOL,
            "max_step_fraction": v14.MAX_STEP_FRACTION,
            "symplecticity_gate": v14.SYMPLECTICITY_GATE,
            "reciprocal_pair_gate": v14.RECIPROCAL_PAIR_GATE,
        },
        "feature_blind_draw": {
            "uses": ["mass_cell", "source_stability_label", "sample_seed", "v14_v15_v16_v17_holdout"],
            "forbidden": ["velocity_fraction", "score", "zone_index", "frame_spread", "D5_output", "AUC", "p_value"],
        },
    }
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v18:prepare] grid scan OK; quadruple holdout excluded={len(holdout)}")
    print(f"[v18:prepare] supported cells={len(supported_cells)}")
    print(f"[v18:prepare] drew {len(sample_rows)} rows to {v14.relpath(out / 'sample_frame.csv')}")


def command_shard(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.18 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
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
    started = time.perf_counter()
    for pos, sample_rec in enumerate(shard_sample, start=1):
        idx = int(sample_rec["orbit_index"])
        row = row_by_index[idx]
        masses, x0, v0 = v14.expand_liao2021_state(row, center_com=True)
        meas = v16.measure_ensemble(masses, x0, v0, row.period)
        rec = {**sample_rec, **meas}
        records.append(rec)
        print(
            f"[v18:shard {args.shard_index}/{args.shard_count}] "
            f"{pos}/{len(shard_sample)} idx={idx} cell={sample_rec['mass_cell']} "
            f"label={sample_rec['stability']} status={meas['status']} "
            f"score={meas.get('score')} t={float(meas['total_seconds']):.1f}s",
            flush=True,
        )

    sample_path = out / f"sample_shard_{args.shard_index:02d}_of_{args.shard_count}.csv"
    v14.write_csv(sample_path, records, v16.PER_ROW_FIELDS)
    summary = {
        "schema": "sundog.isotrophy.v0.18-shard.v1",
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
    print(f"[v18:shard] wrote {v14.relpath(sample_path)}")


def command_merge(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.18 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
    out = Path(args.out)
    sample = v14.read_csv(out / "sample_frame.csv")
    shard_rows = []
    missing = []
    for i in range(args.shard_count):
        sample_path = out / f"sample_shard_{i:02d}_of_{args.shard_count}.csv"
        if not sample_path.exists():
            missing.append(v14.relpath(sample_path))
            continue
        shard_rows.extend(v14.read_csv(sample_path))
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
    v14.write_csv(out / "per_row_sample.csv", shard_rows, v16.PER_ROW_FIELDS)

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
    print(f"[v18:merge] rows={n} success={success} blocked={blocked} sanity={sanity}")
    print(f"[v18:merge] attrition={attr:.4f} Wilson95=[{lo:.4f}, {hi:.4f}]")


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xr = v16.midranks(np.asarray(x, dtype=float))
    yr = v16.midranks(np.asarray(y, dtype=float))
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def reliability_auc_test(per_cell: list[dict], permutations: int, seed: int) -> dict:
    primary_rows = [
        r for r in per_cell
        if v14.parse_bool(r["primary"]) and r.get("AUC_cell") is not None and r.get("frame_p90") is not None
    ]
    cells = [r["mass_cell"] for r in primary_rows]
    reliability = [-math.log10(float(r["frame_p90"]) + 1e-9) for r in primary_rows]
    aucs = [float(r["AUC_cell"]) for r in primary_rows]
    rho_obs = spearman_rho(reliability, aucs)
    if rho_obs is None:
        return {
            "cells": cells,
            "k": len(cells),
            "rho": None,
            "p_reliability": None,
            "permutations": permutations,
            "seed": seed,
            "ge_observed": None,
            "pass": False,
            "not_computed_reason": "insufficient variation",
        }
    rng = np.random.default_rng(seed)
    ge = 0
    max_rho = -1.0
    min_rho = 1.0
    auc_arr = np.asarray(aucs, dtype=float)
    for _ in range(permutations):
        perm_auc = auc_arr[rng.permutation(len(auc_arr))]
        rho = spearman_rho(reliability, list(perm_auc))
        if rho is not None:
            ge += int(rho >= rho_obs - 1e-12)
            max_rho = max(max_rho, rho)
            min_rho = min(min_rho, rho)
    p = (ge + 1) / (permutations + 1)
    return {
        "cells": cells,
        "k": len(cells),
        "rho": rho_obs,
        "p_reliability": p,
        "permutations": permutations,
        "seed": seed,
        "ge_observed": ge,
        "null_rho_min": min_rho,
        "null_rho_max": max_rho,
        "rho_floor": RELIABILITY_RHO_FLOOR,
        "p_threshold": RELIABILITY_P_THRESHOLD,
        "pass": bool(rho_obs >= RELIABILITY_RHO_FLOOR and p <= RELIABILITY_P_THRESHOLD),
    }


def reverse_p_value(rows: list[dict], observed_j: float, permutations: int, seed: int) -> dict:
    scores = np.asarray([float(r["score"]) for r in rows], dtype=float)
    ranks = v16.midranks(scores)
    s_count = sum(1 for r in rows if r["stability"] == "S")
    rng = np.random.default_rng(seed)
    le = 0
    min_j = float("inf")
    max_j = -1.0
    for _ in range(permutations):
        idx = rng.choice(len(ranks), size=s_count, replace=False)
        j_perm = float(np.sum(ranks[idx]) - s_count * (s_count + 1) / 2.0)
        le += int(j_perm <= observed_j + 1e-12)
        min_j = min(min_j, j_perm)
        max_j = max(max_j, j_perm)
    return {
        "p_reverse": (le + 1) / (permutations + 1),
        "le_observed": le,
        "null_J_min": min_j,
        "null_J_max": max_j,
    }


def reversal_guard(per_cell: list[dict], primary_by_cell: dict[str, list[dict]],
                   permutations: int, seed: int) -> dict:
    rows = []
    stable_decisive = []
    for order, rec in enumerate(per_cell):
        if not v14.parse_bool(rec["primary"]):
            continue
        cell = rec["mass_cell"]
        auc = rec.get("AUC_cell")
        frame_p90 = rec.get("frame_p90")
        if auc is None or frame_p90 is None:
            continue
        auc_f = float(auc)
        frame_p90_f = float(frame_p90)
        p_rev = None
        le_obs = None
        if auc_f <= REVERSAL_AUC_FLOOR:
            rev = reverse_p_value(primary_by_cell[cell], float(rec["J_cell"]), permutations, seed + order + 1)
            p_rev = rev["p_reverse"]
            le_obs = rev["le_observed"]
        frame_fragile = frame_p90_f >= FRAME_FRAGILE_P90
        decisive_negative = bool(auc_f <= REVERSAL_AUC_FLOOR and p_rev is not None and p_rev <= REVERSAL_P_THRESHOLD)
        stable_decisive_negative = decisive_negative and not frame_fragile
        row = {
            "mass_cell": cell,
            "AUC_cell": auc_f,
            "frame_p90": frame_p90_f,
            "frame_fragile_cell": frame_fragile,
            "p_reverse": p_rev,
            "le_observed": le_obs,
            "decisive_negative_cell": decisive_negative,
            "stable_decisive_negative": stable_decisive_negative,
        }
        rows.append(row)
        if stable_decisive_negative:
            stable_decisive.append(row)
    return {
        "permutations_per_cell": permutations,
        "seed": seed,
        "frame_fragile_threshold": FRAME_FRAGILE_P90,
        "decisive_negative_auc_floor": REVERSAL_AUC_FLOOR,
        "reverse_p_threshold": REVERSAL_P_THRESHOLD,
        "stable_decisive_negative_count": len(stable_decisive),
        "passes": len(stable_decisive) == 0,
        "cells": rows,
    }


def frame_p90_distribution(per_cell: list[dict]) -> dict:
    vals = [float(r["frame_p90"]) for r in per_cell if v14.parse_bool(r["primary"]) and r.get("frame_p90") is not None]
    return {
        "N": len(vals),
        "min": min(vals) if vals else None,
        "median": v16.quantile(vals, 0.50),
        "p75": v16.quantile(vals, 0.75),
        "p90": v16.quantile(vals, 0.90),
        "max": max(vals) if vals else None,
    }


def determine_verdict(manifest: dict, primary_cells: int, primary_rows: int,
                      frame_median: float | None, frame_p90: float | None,
                      auc: float | None, p_perm: float | None,
                      reliability: dict, reversal: dict) -> str:
    attr = manifest.get("attrition_fraction")
    attr_hi = manifest.get("attrition_wilson95_high")
    if attr is None or attr_hi is None or frame_median is None or frame_p90 is None:
        return "blocked_by_receipt"
    if attr > v14.ATTRITION_ALLOWED_MAX or attr_hi > v14.ATTRITION_WILSON_HIGH_MAX:
        return "blocked_by_attrition"
    if frame_median > FRAME_WARN_MEDIAN or frame_p90 > FRAME_WARN_P90:
        return "blocked_by_global_frame_instability"
    if primary_cells < PRIMARY_MIN_CELLS or primary_rows < PRIMARY_MIN_ROWS:
        return "blocked_by_coverage"
    if auc is None or p_perm is None:
        return "blocked_by_receipt"
    if auc <= 0.50 or p_perm > v14.P_VALUE_THRESHOLD:
        base = "wider_transfer_fails"
    elif 0.50 < auc < v14.AUC_PASS_FLOOR and p_perm <= v14.P_VALUE_THRESHOLD:
        base = "wider_transfer_directional_weak"
    else:
        rel_pass = bool(reliability.get("pass"))
        reversal_count = int(reversal.get("stable_decisive_negative_count", 0))
        if rel_pass and reversal_count > 0:
            base = "frame_reliability_supported_but_reversal_guard_fails"
        elif rel_pass:
            base = "reliability_drives_per_cell_auc_supported"
        else:
            base = "wider_transfer_replicates_reliability_unresolved"
    if base.startswith("blocked"):
        return base
    if attr > v14.ATTRITION_CLEAN_MAX:
        return f"{base}_with_attrition_warning"
    return base


def command_analyze(args: argparse.Namespace) -> None:
    if args.permutations != PERMUTATIONS:
        raise SystemExit(f"v0.18 permutation count is locked at {PERMUTATIONS}; got {args.permutations}")
    if args.seed != SEED:
        raise SystemExit(f"v0.18 permutation seed is locked at {SEED}; got {args.seed}")
    out = Path(args.out)
    manifest = v14.read_json(out / "manifest.json")
    rows = v14.read_csv(out / "per_row_sample.csv")
    success_rows = [
        r for r in rows
        if r["status"] == "success" and r.get("score", "") != "" and r["stability"] in ("S", "U")
    ]
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in success_rows:
        by_cell[r["mass_cell"]].append(r)

    dispersion, overall_dispersion = v16.dispersion_rows(success_rows)
    dispersion_by_cell = {
        r["group"]: r for r in dispersion
        if r["group_type"] == "mass_cell"
    }

    per_cell = []
    primary: dict[str, list[dict]] = {}
    for cell in sorted(manifest.get("supported_cells", [])):
        cell_rows = by_cell.get(cell, [])
        stats = v16.continuous_cell_stats(cell_rows, "score")
        disp = dispersion_by_cell.get(cell, {})
        is_primary = stats["S"] >= PRIMARY_CELL_MIN_S and stats["U"] >= PRIMARY_CELL_MIN_U
        zones = Counter(int(r["zone_index_vf0"]) for r in cell_rows if r.get("zone_index_vf0", "") != "")
        if is_primary:
            primary[cell] = cell_rows
        per_cell.append({
            "mass_cell": cell,
            "primary": is_primary,
            **stats,
            "frame_p50": disp.get("median"),
            "frame_p75": disp.get("p75"),
            "frame_p90": disp.get("p90"),
            "frame_p95": disp.get("p95"),
            "frame_max": disp.get("max"),
            "cell_reliability": (
                -math.log10(float(disp["p90"]) + 1e-9)
                if disp.get("p90") is not None else None
            ),
            "S_minus_U_p50": (
                float(stats["S_score_p50"]) - float(stats["U_score_p50"])
                if stats.get("S_score_p50") is not None and stats.get("U_score_p50") is not None else None
            ),
            "zone0": zones[0],
            "zone1": zones[1],
            "zone2": zones[2],
            "zone2_fraction": zones[2] / len(cell_rows) if cell_rows else None,
        })

    primary_rows = sum(len(v) for v in primary.values())
    j_cond = sum(float(r["J_cell"]) for r in per_cell if v14.parse_bool(r["primary"]))
    d_cond = sum(int(r["D_cell"]) for r in per_cell if v14.parse_bool(r["primary"]))
    auc_cond = j_cond / d_cond if d_cond else None
    perm = (
        v16.permutation_p_continuous(primary, "score", j_cond, args.permutations, args.seed)
        if primary and d_cond else
        {"permutations": args.permutations, "seed": args.seed, "observed_J": j_cond,
         "D_cond": d_cond, "observed_AUC": auc_cond, "ge_observed": None, "p_perm": None}
    )
    reliability = reliability_auc_test(per_cell, args.permutations, args.seed)
    reversal = reversal_guard(per_cell, primary, args.permutations, args.seed)
    p90_dist = frame_p90_distribution(per_cell)
    verdict = determine_verdict(
        manifest, len(primary), primary_rows, overall_dispersion.get("median"),
        overall_dispersion.get("p90"), auc_cond, perm.get("p_perm"), reliability, reversal,
    )

    v14.write_csv(out / "per_cell_auc.csv", per_cell, [
        "mass_cell", "primary", "N_success", "S", "U", "J_cell", "D_cell", "AUC_cell",
        "S_score_p25", "S_score_p50", "S_score_p75",
        "U_score_p25", "U_score_p50", "U_score_p75", "S_minus_U_p50",
        "frame_p50", "frame_p75", "frame_p90", "frame_p95", "frame_max", "cell_reliability",
        "zone0", "zone1", "zone2", "zone2_fraction",
    ])
    v14.write_csv(out / "frame_reliability_map.csv", sorted(
        per_cell,
        key=lambda r: (
            r.get("frame_p90") is None,
            float(r["frame_p90"]) if r.get("frame_p90") is not None else float("inf"),
        ),
    ), [
        "mass_cell", "primary", "AUC_cell", "cell_reliability",
        "frame_p50", "frame_p75", "frame_p90", "frame_p95", "frame_max",
        "N_success", "S", "U", "S_minus_U_p50", "zone2_fraction",
    ])
    v14.write_csv(out / "frame_dispersion.csv", dispersion, [
        "group_type", "group", "N", "median", "p75", "p90", "p95", "max",
    ])
    v14.write_csv(out / "anatomy_sidecar.csv", per_cell, [
        "mass_cell", "primary", "AUC_cell", "S_score_p25", "S_score_p50", "S_score_p75",
        "U_score_p25", "U_score_p50", "U_score_p75", "S_minus_U_p50",
        "zone0", "zone1", "zone2", "zone2_fraction",
        "frame_p90", "frame_max",
    ])
    v14.write_json(out / "permutation_summary.json", {
        **perm,
        "primary_supported_cells": len(primary),
        "primary_success_rows": primary_rows,
        "frame_spread_summary": overall_dispersion,
        "effect_floor": v14.AUC_PASS_FLOOR,
        "p_value_threshold": v14.P_VALUE_THRESHOLD,
        "verdict": verdict,
    })
    v14.write_json(out / "reliability_auc_test.json", {
        **reliability,
        "frame_p90_distribution": p90_dist,
        "monte_carlo_not_exact": True,
    })
    v14.write_json(out / "reversal_guard.json", reversal)
    manifest.update({
        "stage": "analyzed",
        "verdict": verdict,
        "completedAt": v14.utc_now(),
        "primary_supported_cells": len(primary),
        "primary_success_rows": primary_rows,
        "frame_spread_median": overall_dispersion.get("median"),
        "frame_spread_p90": overall_dispersion.get("p90"),
        "frame_p90_distribution": p90_dist,
        "AUC_cond": auc_cond,
        "J_cond": j_cond,
        "D_cond": d_cond,
        "p_perm": perm.get("p_perm"),
        "reliability_rho": reliability.get("rho"),
        "reliability_p": reliability.get("p_reliability"),
        "stable_decisive_negative_count": reversal.get("stable_decisive_negative_count"),
        "target_tier": v14.TARGET_TIER,
        "claim_boundary": (
            "Tier-2 / Li-Liao lineage / stable-support sampling / within-cell rank signal / "
            "tail-resolved continuous score / not coarse-zone / not full-catalog prevalence / "
            "not Tier-3 independent / not theorem-facing"
        ),
    })
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v18:analyze] primary cells={len(primary)} rows={primary_rows}")
    if auc_cond is not None:
        print(f"[v18:analyze] AUC_cond={auc_cond:.4f} p_perm={perm.get('p_perm')}")
    print(f"[v18:analyze] reliability rho={reliability.get('rho')} p={reliability.get('p_reliability')}")
    print(f"[v18:analyze] reversal stable_decisive_negative={reversal.get('stable_decisive_negative_count')}")
    print(f"[v18:analyze] frame p90 distribution={p90_dist}")
    print(f"[v18:analyze] verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="draw v0.14-v0.17-held-out 8x8 reliability sample")
    p.add_argument("--target", default=str(v14.DEFAULT_TARGET))
    p.add_argument("--v14", default=str(DEFAULT_V14))
    p.add_argument("--v15", default=str(DEFAULT_V15))
    p.add_argument("--v16", default=str(DEFAULT_V16))
    p.add_argument("--v17", default=str(DEFAULT_V17))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--grid-parts", type=int, default=GRID_PARTS)
    p.add_argument("--stable-per-cell", type=int, default=STABLE_PER_CELL)
    p.add_argument("--unstable-per-cell", type=int, default=UNSTABLE_PER_CELL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    p.set_defaults(func=command_prepare)

    s = sub.add_parser("shard", help="run four-frame D5 + ensemble score for one shard")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    s.set_defaults(func=command_shard)

    m = sub.add_parser("merge", help="merge shard receipts and compute attrition")
    m.add_argument("--out", default=str(DEFAULT_OUT))
    m.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    m.set_defaults(func=command_merge)

    a = sub.add_parser("analyze", help="compute v0.18 pooled transfer and reliability checks")
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
