#!/usr/bin/env python3
"""v0.17 liao2021 tail-resolved heterogeneity replication runner.

Locked by docs/isotrophy/kfacet/kfacet_v17_liao2021_heterogeneity_scope.md.

This chapter is a fresh-row replication of v0.16. It reuses the frozen v0.16
four-frame ensemble score and adds only the triple holdout, the exact Spearman
heterogeneity check, and a non-gating anatomy sidecar over already-measured v0.16
rows.
"""
from __future__ import annotations

import argparse
import itertools
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

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v17_liao2021_heterogeneity_scope.md"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v17-liao2021-heterogeneity"
DEFAULT_V14 = ROOT / "results/isotrophy/k-facet-v14-liao2021-sampled-transfer"
DEFAULT_V15 = ROOT / "results/isotrophy/k-facet-v15-liao2021-stable-support-transfer"
DEFAULT_V16 = ROOT / "results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer"

SEED = 20260523
STABLE_PER_CELL = 80
UNSTABLE_PER_CELL = 80
EXPECTED_SUPPORTED_CELLS = 7
SHARD_COUNT = 14
PERMUTATIONS = 100000
EXPECTED_TARGET_SHA = "9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4"
FRAME_DEGREES = v16.FRAME_DEGREES

PRIMARY_CELL_MIN_N = 120
PRIMARY_CELL_MIN_S = 50
PRIMARY_CELL_MIN_U = 50
PRIMARY_MIN_CELLS = 6
PRIMARY_MIN_ROWS = 900

FRAME_CLEAN_MEDIAN = v16.FRAME_CLEAN_MEDIAN
FRAME_CLEAN_P90 = v16.FRAME_CLEAN_P90
FRAME_WARN_MEDIAN = v16.FRAME_WARN_MEDIAN
FRAME_WARN_P90 = v16.FRAME_WARN_P90

REFERENCE_AUC = {
    "mass_qA0_qB0": 0.97203125,
    "mass_qA1_qB0": 0.57515625,
    "mass_qA2_qB1": 0.43156250,
    "mass_qA3_qB0": 0.47046875,
    "mass_qA3_qB1": 0.69593750,
    "mass_qA3_qB2": 0.63921875,
    "mass_qA3_qB3": 0.74453125,
}

VERDICT_PREPARED = "tail_resolved_heterogeneity_prepared_not_measured"
VERDICT_MERGED = "tail_resolved_heterogeneity_merged_not_analyzed"

SOURCE_SUPPORT_FIELDS = [
    "mass_cell", "source_S", "source_U",
    "v14_excluded_S", "v14_excluded_U",
    "v15_excluded_S", "v15_excluded_U",
    "v16_excluded_S", "v16_excluded_U",
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


def load_v16_reference(v16_dir: Path) -> tuple[list[dict], float]:
    path = v16_dir / "per_cell_rank.csv"
    if not path.exists():
        raise SystemExit(f"missing v0.16 per-cell reference: {path}")
    rows = v14.read_csv(path)
    got = {
        r["mass_cell"]: float(r["AUC_cell"])
        for r in rows
        if v14.parse_bool(r.get("primary", False)) and r.get("AUC_cell", "") != ""
    }
    missing = sorted(set(REFERENCE_AUC) - set(got))
    extra = sorted(set(got) - set(REFERENCE_AUC))
    if missing or extra:
        raise SystemExit(f"v0.16 reference cell mismatch; missing={missing} extra={extra}")
    max_diff = max(abs(got[cell] - ref) for cell, ref in REFERENCE_AUC.items())
    if max_diff > 1e-12:
        raise SystemExit(f"v0.16 reference vector differs from lock; max diff {max_diff}")
    reference_rows = [
        {
            "mass_cell": cell,
            "reference_AUC_cell": REFERENCE_AUC[cell],
            "v16_receipt_AUC_cell": got[cell],
            "abs_diff": abs(got[cell] - REFERENCE_AUC[cell]),
        }
        for cell in sorted(REFERENCE_AUC)
    ]
    return reference_rows, max_diff


def check_prior_receipts(target_path: Path, target_sha: str, v14_dir: Path, v15_dir: Path,
                         v16_dir: Path) -> tuple[dict, set[int], set[int], set[int], list[dict], float]:
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
    overlaps = {
        "v14_v15": holdout14 & holdout15,
        "v14_v16": holdout14 & holdout16,
        "v15_v16": holdout15 & holdout16,
    }
    bad = {name: sorted(vals)[:10] for name, vals in overlaps.items() if vals}
    if bad:
        raise SystemExit(f"holdout overlap unexpectedly non-empty: {bad}")

    reference_rows, reference_max_diff = load_v16_reference(v16_dir)
    checks = dict(v13.get("checks", {}))
    checks.update({
        "v14_verdict_undecidable_coverage": v14_manifest.get("verdict") == "sample_transfer_undecidable_coverage",
        "v15_verdict_directional_weak": v15_manifest.get("verdict") == "stable_support_transfer_directional_weak",
        "v16_verdict_passes_clean": v16_manifest.get("verdict") == "tail_resolved_transfer_passes_clean",
        "v14_target_sha_matches": v14_manifest.get("target_sha256") == target_sha,
        "v15_target_sha_matches": v15_manifest.get("target_sha256") == target_sha,
        "v16_target_sha_matches": v16_manifest.get("target_sha256") == target_sha,
        "v14_sample_frame_available": len(holdout14) > 0,
        "v15_sample_frame_available": len(holdout15) > 0,
        "v16_sample_frame_available": len(holdout16) > 0,
        "v16_reference_vector_matches_lock": reference_max_diff <= 1e-12,
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
            "AUC_cond": v14_manifest.get("AUC_cond"),
            "p_perm": v14_manifest.get("p_perm"),
        },
        "v15_context": {
            "path": v14.relpath(v15_dir),
            "verdict": v15_manifest.get("verdict"),
            "sample_rows_requested": v15_manifest.get("sample_rows_requested"),
            "sample_rows_success": v15_manifest.get("sample_rows_success"),
            "AUC_cond": v15_manifest.get("AUC_cond"),
            "p_perm": v15_manifest.get("p_perm"),
        },
        "v16_context": {
            "path": v14.relpath(v16_dir),
            "verdict": v16_manifest.get("verdict"),
            "sample_rows_requested": v16_manifest.get("sample_rows_requested"),
            "sample_rows_success": v16_manifest.get("sample_rows_success"),
            "AUC_cond": v16_manifest.get("AUC_cond"),
            "p_perm": v16_manifest.get("p_perm"),
            "reference_max_abs_diff": reference_max_diff,
        },
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise SystemExit("prior receipt check(s) failed: " + ", ".join(failed))
    return receipts, holdout14, holdout15, holdout16, reference_rows, reference_max_diff


def source_support_census(rows: list[Any], cell_of: dict[int, str], holdout14: set[int],
                          holdout15: set[int], holdout16: set[int],
                          stable_per_cell: int, unstable_per_cell: int) -> tuple[list[dict], list[str]]:
    counts_all: dict[str, Counter] = defaultdict(Counter)
    counts_v14: dict[str, Counter] = defaultdict(Counter)
    counts_v15: dict[str, Counter] = defaultdict(Counter)
    counts_v16: dict[str, Counter] = defaultdict(Counter)
    counts_eligible: dict[str, Counter] = defaultdict(Counter)
    holdout = holdout14 | holdout15 | holdout16
    for row in rows:
        cell = cell_of[row.index]
        counts_all[cell][row.stability] += 1
        if row.index in holdout14:
            counts_v14[cell][row.stability] += 1
        if row.index in holdout15:
            counts_v15[cell][row.stability] += 1
        if row.index in holdout16:
            counts_v16[cell][row.stability] += 1
        if row.index not in holdout:
            counts_eligible[cell][row.stability] += 1

    census = []
    supported = []
    for cell in sorted(counts_all):
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
        census.append({
            "mass_cell": cell,
            "source_S": counts_all[cell]["S"],
            "source_U": counts_all[cell]["U"],
            "v14_excluded_S": counts_v14[cell]["S"],
            "v14_excluded_U": counts_v14[cell]["U"],
            "v15_excluded_S": counts_v15[cell]["S"],
            "v15_excluded_U": counts_v15[cell]["U"],
            "v16_excluded_S": counts_v16[cell]["S"],
            "v16_excluded_U": counts_v16[cell]["U"],
            "eligible_S": s_el,
            "eligible_U": u_el,
            "supported": is_supported,
            "disposition": disposition,
        })
    return census, supported


def draw_case_control_sample(rows: list[Any], cell_of: dict[int, str], holdout: set[int],
                             supported_cells: list[str], stable_per_cell: int,
                             unstable_per_cell: int, seed: int) -> list[dict]:
    return v15.draw_case_control_sample(
        rows, cell_of, holdout, supported_cells, stable_per_cell, unstable_per_cell, seed
    )


def operator_commands(out: Path, target: Path, v14_dir: Path, v15_dir: Path, v16_dir: Path,
                      stable_per_cell: int, unstable_per_cell: int, seed: int,
                      shard_count: int) -> str:
    lines = [
        "# v0.17 liao2021 Heterogeneity Replication Operator Commands",
        "",
        "Run from the repository root. Shards may be run concurrently.",
        "",
        "```powershell",
        "python scripts/v17_liao2021_heterogeneity_replication.py prepare `",
        f"  --target {v14.relpath(target)} `",
        f"  --v14 {v14.relpath(v14_dir)} `",
        f"  --v15 {v14.relpath(v15_dir)} `",
        f"  --v16 {v14.relpath(v16_dir)} `",
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
            "python scripts/v17_liao2021_heterogeneity_replication.py shard `",
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
        "python scripts/v17_liao2021_heterogeneity_replication.py merge `",
        f"  --out {v14.relpath(out)} `",
        f"  --shard-count {shard_count}",
        "```",
        "",
        "## Analyze",
        "",
        "```powershell",
        "python scripts/v17_liao2021_heterogeneity_replication.py analyze `",
        f"  --out {v14.relpath(out)} `",
        f"  --permutations {PERMUTATIONS} `",
        f"  --seed {seed}",
        "```",
        "",
        "Readback: manifest.json, per_cell_rank.csv, heterogeneity_replication.json, "
        "permutation_summary.json, anatomy_sidecar.csv.",
    ])
    return "\n".join(lines) + "\n"


def command_prepare(args: argparse.Namespace) -> None:
    if args.stable_per_cell != STABLE_PER_CELL or args.unstable_per_cell != UNSTABLE_PER_CELL:
        raise SystemExit(
            f"v0.17 is locked at {STABLE_PER_CELL}/{UNSTABLE_PER_CELL}; got "
            f"{args.stable_per_cell}/{args.unstable_per_cell}"
        )
    if args.seed != SEED:
        raise SystemExit(f"v0.17 sample seed is locked at {SEED}; got {args.seed}")
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.17 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")

    started = v14.utc_now()
    target = v14.resolve_target(args.target)
    v14_dir = resolve_dir(args.v14)
    v15_dir = resolve_dir(args.v15)
    v16_dir = resolve_dir(args.v16)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target_sha = v14.sha256_file(target)
    receipts, holdout14, holdout15, holdout16, reference_rows, reference_max_diff = check_prior_receipts(
        target, target_sha, v14_dir, v15_dir, v16_dir
    )
    holdout = holdout14 | holdout15 | holdout16
    rows = v14.parse_liao2021(target)
    if len(rows) != 135445:
        raise SystemExit(f"expected 135445 liao2021 rows, parsed {len(rows)}")

    _mass_cell_rows, cell_of = v14.build_mass_cells(rows)
    census, supported_cells = source_support_census(
        rows, cell_of, holdout14, holdout15, holdout16,
        args.stable_per_cell, args.unstable_per_cell,
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
        raise SystemExit("v0.17 draw included a v0.14/v0.15/v0.16 holdout orbit")

    v14.write_csv(out / "source_support_census.csv", census, SOURCE_SUPPORT_FIELDS)
    v14.write_csv(out / "v16_reference_vector.csv", reference_rows, [
        "mass_cell", "reference_AUC_cell", "v16_receipt_AUC_cell", "abs_diff",
    ])
    v14.write_csv(out / "sample_frame.csv", sample_rows, v16.SAMPLE_FRAME_FIELDS)
    (out / "operator_commands.md").write_text(
        operator_commands(
            out, target, v14_dir, v15_dir, v16_dir, args.stable_per_cell,
            args.unstable_per_cell, args.seed, args.shard_count,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "sundog.isotrophy.v0.17-liao2021-heterogeneity-replication.v1",
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
        "v15_path": v14.relpath(v15_dir),
        "v16_path": v14.relpath(v16_dir),
        "v14_sample_excluded_count": len(holdout14),
        "v15_sample_excluded_count": len(holdout15),
        "v16_sample_excluded_count": len(holdout16),
        "triple_holdout_excluded_count": len(holdout),
        "v16_reference_max_abs_diff": reference_max_diff,
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
            "uses": ["mass_cell", "source_stability_label", "sample_seed", "v14_v15_v16_holdout"],
            "forbidden": ["velocity_fraction", "score", "zone_index", "D5_output", "AUC", "p_value"],
        },
    }
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v17:prepare] supported cells={len(supported_cells)} {supported_cells}")
    print(f"[v17:prepare] drew {len(sample_rows)} rows; excluded {len(holdout)} v0.14/v0.15/v0.16 rows")
    print(f"[v17:prepare] wrote {v14.relpath(out / 'sample_frame.csv')}")


def command_shard(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.17 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
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
            f"[v17:shard {args.shard_index}/{args.shard_count}] "
            f"{pos}/{len(shard_sample)} idx={idx} cell={sample_rec['mass_cell']} "
            f"label={sample_rec['stability']} status={meas['status']} "
            f"score={meas.get('score')} t={float(meas['total_seconds']):.1f}s",
            flush=True,
        )

    sample_path = out / f"sample_shard_{args.shard_index:02d}_of_{args.shard_count}.csv"
    v14.write_csv(sample_path, records, v16.PER_ROW_FIELDS)
    summary = {
        "schema": "sundog.isotrophy.v0.17-shard.v1",
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
    print(f"[v17:shard] wrote {v14.relpath(sample_path)}")


def command_merge(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.17 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
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
    print(f"[v17:merge] rows={n} success={success} blocked={blocked} sanity={sanity}")
    print(f"[v17:merge] attrition={attr:.4f} Wilson95=[{lo:.4f}, {hi:.4f}]")


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xr = v16.midranks(np.asarray(x, dtype=float))
    yr = v16.midranks(np.asarray(y, dtype=float))
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def exact_spearman_test(reference: dict[str, float], observed: dict[str, float]) -> dict:
    cells = sorted(set(reference) & set(observed))
    ref_vals = [float(reference[c]) for c in cells]
    obs_vals = [float(observed[c]) for c in cells]
    rho_obs = spearman_rho(ref_vals, obs_vals)
    if rho_obs is None:
        return {
            "matched_cells": cells,
            "k": len(cells),
            "rho": None,
            "p_rho": None,
            "permutations_exact": math.factorial(len(cells)) if len(cells) >= 0 else None,
            "ge_observed": None,
        }
    ge = 0
    total = 0
    for perm in itertools.permutations(obs_vals):
        rho = spearman_rho(ref_vals, list(perm))
        total += 1
        if rho is not None and rho >= rho_obs - 1e-12:
            ge += 1
    return {
        "matched_cells": cells,
        "k": len(cells),
        "rho": rho_obs,
        "p_rho": ge / total if total else None,
        "permutations_exact": total,
        "ge_observed": ge,
    }


def auc_se_hanley(auc: float | None, s_count: int, u_count: int) -> float | None:
    if auc is None or s_count <= 0 or u_count <= 0:
        return None
    if auc <= 0.0 or auc >= 1.0:
        return 0.0
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (s_count - 1) * (q1 - auc * auc)
        + (u_count - 1) * (q2 - auc * auc)
    ) / (s_count * u_count)
    return float(math.sqrt(max(var, 0.0)))


def heterogeneity_replication(per_cell: list[dict], pooled_pass: bool) -> dict:
    observed = {
        r["mass_cell"]: float(r["AUC_cell"])
        for r in per_cell
        if v14.parse_bool(r["primary"]) and r.get("AUC_cell") is not None
    }
    sign_panel = []
    for cell in sorted(set(REFERENCE_AUC) & set(observed)):
        row = next(r for r in per_cell if r["mass_cell"] == cell)
        ref = REFERENCE_AUC[cell]
        obs = observed[cell]
        se = auc_se_hanley(obs, int(row["S"]), int(row["U"]))
        sign_panel.append({
            "mass_cell": cell,
            "v16_reference_AUC_cell": ref,
            "v17_AUC_cell": obs,
            "reference_direction": "above_0p5" if ref > 0.5 else "below_0p5" if ref < 0.5 else "tie_0p5",
            "v17_direction": "above_0p5" if obs > 0.5 else "below_0p5" if obs < 0.5 else "tie_0p5",
            "direction_match": (ref > 0.5 and obs > 0.5) or (ref < 0.5 and obs < 0.5) or (ref == 0.5 and obs == 0.5),
            "abs_reference_minus_0p5": abs(ref - 0.5),
            "abs_v17_minus_0p5": abs(obs - 0.5),
            "auc_se_approx": se,
        })
    test = exact_spearman_test(REFERENCE_AUC, observed) if pooled_pass else {
        "matched_cells": sorted(set(REFERENCE_AUC) & set(observed)),
        "k": len(set(REFERENCE_AUC) & set(observed)),
        "rho": None,
        "p_rho": None,
        "permutations_exact": None,
        "ge_observed": None,
        "not_computed_reason": "pooled replication did not pass",
    }
    return {
        **test,
        "p_rho_threshold": 0.05,
        "heterogeneity_replicates": bool(test.get("p_rho") is not None and float(test["p_rho"]) <= 0.05),
        "sign_panel": sign_panel,
    }


def anatomy_sidecar(v16_dir: Path) -> list[dict]:
    path = v16_dir / "per_row_sample.csv"
    if not path.exists():
        return []
    rows = [
        r for r in v14.read_csv(path)
        if r.get("status") == "success" and r.get("score", "") != ""
    ]
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[row["mass_cell"]].append(row)
    out = []
    for cell, cell_rows in sorted(by_cell.items()):
        scores = [float(r["score"]) for r in cell_rows]
        spreads = [float(r["frame_spread"]) for r in cell_rows if r.get("frame_spread", "") != ""]
        periods = [float(r["period"]) for r in cell_rows]
        mass_a = [float(r["mass_a"]) for r in cell_rows]
        mass_b = [float(r["mass_b"]) for r in cell_rows]
        mass_c = [float(r["mass_c"]) for r in cell_rows]
        zones = Counter(int(r["zone_index_vf0"]) for r in cell_rows if r.get("zone_index_vf0", "") != "")
        out.append({
            "mass_cell": cell,
            "source": "v0.16_success_rows",
            "N": len(cell_rows),
            "S": sum(1 for r in cell_rows if r["stability"] == "S"),
            "U": sum(1 for r in cell_rows if r["stability"] == "U"),
            "score_p05": v16.quantile(scores, 0.05),
            "score_p25": v16.quantile(scores, 0.25),
            "score_p50": v16.quantile(scores, 0.50),
            "score_p75": v16.quantile(scores, 0.75),
            "score_p95": v16.quantile(scores, 0.95),
            "frame_spread_p50": v16.quantile(spreads, 0.50),
            "frame_spread_p90": v16.quantile(spreads, 0.90),
            "zone0": zones[0],
            "zone1": zones[1],
            "zone2": zones[2],
            "period_p50": v16.quantile(periods, 0.50),
            "mass_a_p50": v16.quantile(mass_a, 0.50),
            "mass_b_p50": v16.quantile(mass_b, 0.50),
            "mass_c_p50": v16.quantile(mass_c, 0.50),
        })
    return out


def determine_verdict(manifest: dict, primary_cells: int, primary_rows: int,
                      frame_median: float | None, frame_p90: float | None,
                      auc: float | None, p_perm: float | None,
                      heterogeneity_p: float | None) -> str:
    attr = manifest.get("attrition_fraction")
    attr_hi = manifest.get("attrition_wilson95_high")
    if attr is None or attr_hi is None or frame_median is None or frame_p90 is None:
        return "blocked_by_receipt"
    if attr > v14.ATTRITION_ALLOWED_MAX or attr_hi > v14.ATTRITION_WILSON_HIGH_MAX:
        return "blocked_by_attrition"
    if frame_median > FRAME_WARN_MEDIAN or frame_p90 > FRAME_WARN_P90:
        return "blocked_by_frame_instability"
    if primary_cells < PRIMARY_MIN_CELLS or primary_rows < PRIMARY_MIN_ROWS:
        return "blocked_by_coverage"
    if auc is None or p_perm is None:
        return "blocked_by_receipt"
    if auc <= 0.50 or p_perm > v14.P_VALUE_THRESHOLD:
        return "tail_resolved_transfer_not_replicated"
    if 0.50 < auc < v14.AUC_PASS_FLOOR and p_perm <= v14.P_VALUE_THRESHOLD:
        return "tail_resolved_transfer_replication_directional_weak"
    if auc >= v14.AUC_PASS_FLOOR and p_perm <= v14.P_VALUE_THRESHOLD:
        clean = (
            attr <= v14.ATTRITION_CLEAN_MAX
            and frame_median <= FRAME_CLEAN_MEDIAN
            and frame_p90 <= FRAME_CLEAN_P90
        )
        hetero = heterogeneity_p is not None and heterogeneity_p <= 0.05
        if hetero:
            base = "heterogeneous_transfer_replicates_clean"
        else:
            base = "pooled_transfer_replicates_heterogeneity_unresolved"
        return base if clean else f"{base}_with_warning"
    return "tail_resolved_transfer_not_replicated"


def command_analyze(args: argparse.Namespace) -> None:
    if args.permutations != PERMUTATIONS:
        raise SystemExit(f"v0.17 permutation count is locked at {PERMUTATIONS}; got {args.permutations}")
    if args.seed != SEED:
        raise SystemExit(f"v0.17 permutation seed is locked at {SEED}; got {args.seed}")
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

    per_cell = []
    primary: dict[str, list[dict]] = {}
    for cell in sorted(manifest.get("supported_cells", [])):
        cell_rows = by_cell.get(cell, [])
        stats = v16.continuous_cell_stats(cell_rows, "score")
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
    perm = (
        v16.permutation_p_continuous(primary, "score", j_cond, args.permutations, args.seed)
        if primary and d_cond else
        {"permutations": args.permutations, "seed": args.seed, "observed_J": j_cond,
         "D_cond": d_cond, "observed_AUC": auc_cond, "ge_observed": None, "p_perm": None}
    )

    dispersion, overall_dispersion = v16.dispersion_rows(success_rows)
    pooled_pass = bool(
        auc_cond is not None
        and auc_cond >= v14.AUC_PASS_FLOOR
        and perm.get("p_perm") is not None
        and float(perm["p_perm"]) <= v14.P_VALUE_THRESHOLD
    )
    hetero = heterogeneity_replication(per_cell, pooled_pass)
    verdict = determine_verdict(
        manifest, len(primary), primary_rows, overall_dispersion.get("median"),
        overall_dispersion.get("p90"), auc_cond, perm.get("p_perm"), hetero.get("p_rho"),
    )

    anatomy = anatomy_sidecar(resolve_dir(manifest.get("v16_path", str(DEFAULT_V16))))
    v14.write_csv(out / "per_cell_rank.csv", per_cell, [
        "mass_cell", "primary", "N_success", "S", "U", "J_cell", "D_cell", "AUC_cell",
        "S_score_p25", "S_score_p50", "S_score_p75",
        "U_score_p25", "U_score_p50", "U_score_p75",
    ])
    v14.write_csv(out / "frame_dispersion.csv", dispersion, [
        "group_type", "group", "N", "median", "p75", "p90", "p95", "max",
    ])
    v14.write_csv(out / "anatomy_sidecar.csv", anatomy, [
        "mass_cell", "source", "N", "S", "U",
        "score_p05", "score_p25", "score_p50", "score_p75", "score_p95",
        "frame_spread_p50", "frame_spread_p90",
        "zone0", "zone1", "zone2",
        "period_p50", "mass_a_p50", "mass_b_p50", "mass_c_p50",
    ])
    perm_summary = {
        **perm,
        "primary_supported_cells": len(primary),
        "primary_success_rows": primary_rows,
        "frame_spread_summary": overall_dispersion,
        "effect_floor": v14.AUC_PASS_FLOOR,
        "p_value_threshold": v14.P_VALUE_THRESHOLD,
        "verdict": verdict,
    }
    v14.write_json(out / "permutation_summary.json", perm_summary)
    v14.write_json(out / "heterogeneity_replication.json", hetero)
    manifest.update({
        "stage": "analyzed",
        "verdict": verdict,
        "completedAt": v14.utc_now(),
        "primary_supported_cells": len(primary),
        "primary_success_rows": primary_rows,
        "frame_spread_median": overall_dispersion.get("median"),
        "frame_spread_p90": overall_dispersion.get("p90"),
        "AUC_cond": auc_cond,
        "J_cond": j_cond,
        "D_cond": d_cond,
        "p_perm": perm.get("p_perm"),
        "heterogeneity_rho": hetero.get("rho"),
        "heterogeneity_p_rho": hetero.get("p_rho"),
        "target_tier": v14.TARGET_TIER,
    })
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v17:analyze] primary supported cells={len(primary)} rows={primary_rows}")
    if auc_cond is not None:
        print(f"[v17:analyze] AUC_cond={auc_cond:.4f} p_perm={perm.get('p_perm')}")
    print(f"[v17:analyze] heterogeneity rho={hetero.get('rho')} p_rho={hetero.get('p_rho')}")
    print(f"[v17:analyze] frame median={overall_dispersion.get('median')} p90={overall_dispersion.get('p90')}")
    print(f"[v17:analyze] verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="draw v0.14/v0.15/v0.16-held-out 80/80 supported-cell sample")
    p.add_argument("--target", default=str(v14.DEFAULT_TARGET))
    p.add_argument("--v14", default=str(DEFAULT_V14))
    p.add_argument("--v15", default=str(DEFAULT_V15))
    p.add_argument("--v16", default=str(DEFAULT_V16))
    p.add_argument("--out", default=str(DEFAULT_OUT))
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

    a = sub.add_parser("analyze", help="compute v0.17 pooled replication and heterogeneity checks")
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
