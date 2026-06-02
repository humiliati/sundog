#!/usr/bin/env python3
"""v0.16 liao2021 tail-resolved transfer runner.

Locked by docs/isotrophy/kfacet/kfacet_v16_liao2021_tail_resolved_transfer_form.md.

This is a new-feature follow-up to v0.15: a continuous four-frame median
velocity-fraction score on a fresh double-holdout 80/80 supported-cell sample.
"""
from __future__ import annotations

import argparse
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

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v16_liao2021_tail_resolved_transfer_form.md"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v16-liao2021-tail-resolved-transfer"
DEFAULT_V14 = ROOT / "results/isotrophy/k-facet-v14-liao2021-sampled-transfer"
DEFAULT_V15 = ROOT / "results/isotrophy/k-facet-v15-liao2021-stable-support-transfer"

SEED = 20260523
STABLE_PER_CELL = 80
UNSTABLE_PER_CELL = 80
EXPECTED_SUPPORTED_CELLS = 7
SHARD_COUNT = 14
PERMUTATIONS = 100000
EXPECTED_TARGET_SHA = "9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4"
FRAME_DEGREES = (0, 37, 90, 211)

PRIMARY_CELL_MIN_N = 120
PRIMARY_CELL_MIN_S = 50
PRIMARY_CELL_MIN_U = 50
PRIMARY_MIN_CELLS = 6
PRIMARY_MIN_ROWS = 900

FRAME_CLEAN_MEDIAN = 0.10
FRAME_CLEAN_P90 = 0.50
FRAME_WARN_MEDIAN = 0.20
FRAME_WARN_P90 = 0.75

VERDICT_PREPARED = "tail_resolved_transfer_prepared_not_measured"
VERDICT_MERGED = "tail_resolved_transfer_merged_not_analyzed"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


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


def check_prior_receipts(target_path: Path, target_sha: str, v14_dir: Path, v15_dir: Path) -> tuple[dict, set[int], set[int]]:
    if target_sha != EXPECTED_TARGET_SHA:
        raise SystemExit(f"target SHA mismatch: {target_sha} != locked {EXPECTED_TARGET_SHA}")
    v13 = v14.check_prior_receipts(target_path, target_sha)
    v14_manifest, holdout14 = load_sample_holdout(
        v14_dir, "sample_transfer_undecidable_coverage", target_sha, "v0.14"
    )
    v15_manifest, holdout15 = load_sample_holdout(
        v15_dir, "stable_support_transfer_directional_weak", target_sha, "v0.15"
    )
    overlap = holdout14 & holdout15
    if overlap:
        raise SystemExit(f"v0.14 and v0.15 holdouts overlap unexpectedly: {sorted(overlap)[:10]}")

    checks = dict(v13.get("checks", {}))
    checks.update({
        "v14_verdict_undecidable_coverage": True,
        "v15_verdict_directional_weak": True,
        "v14_target_sha_matches": v14_manifest.get("target_sha256") == target_sha,
        "v15_target_sha_matches": v15_manifest.get("target_sha256") == target_sha,
        "v14_sample_frame_available": len(holdout14) > 0,
        "v15_sample_frame_available": len(holdout15) > 0,
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
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise SystemExit("prior receipt check(s) failed: " + ", ".join(failed))
    return receipts, holdout14, holdout15


def source_support_census(rows: list[Any], cell_of: dict[int, str], holdout14: set[int], holdout15: set[int],
                          stable_per_cell: int, unstable_per_cell: int) -> tuple[list[dict], list[str]]:
    counts_all: dict[str, Counter] = defaultdict(Counter)
    counts_v14: dict[str, Counter] = defaultdict(Counter)
    counts_v15: dict[str, Counter] = defaultdict(Counter)
    counts_eligible: dict[str, Counter] = defaultdict(Counter)
    holdout = holdout14 | holdout15
    for row in rows:
        cell = cell_of[row.index]
        counts_all[cell][row.stability] += 1
        if row.index in holdout14:
            counts_v14[cell][row.stability] += 1
        if row.index in holdout15:
            counts_v15[cell][row.stability] += 1
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


def operator_commands(out: Path, target: Path, v14_dir: Path, v15_dir: Path,
                      stable_per_cell: int, unstable_per_cell: int, seed: int,
                      shard_count: int) -> str:
    lines = [
        "# v0.16 liao2021 Tail-Resolved Transfer Operator Commands",
        "",
        "Run from the repository root. Shards may be run concurrently.",
        "",
        "```powershell",
        "python scripts/v16_liao2021_tail_resolved_transfer.py prepare `",
        f"  --target {v14.relpath(target)} `",
        f"  --v14 {v14.relpath(v14_dir)} `",
        f"  --v15 {v14.relpath(v15_dir)} `",
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
            "python scripts/v16_liao2021_tail_resolved_transfer.py shard `",
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
        "python scripts/v16_liao2021_tail_resolved_transfer.py merge `",
        f"  --out {v14.relpath(out)} `",
        f"  --shard-count {shard_count}",
        "```",
        "",
        "## Analyze",
        "",
        "```powershell",
        "python scripts/v16_liao2021_tail_resolved_transfer.py analyze `",
        f"  --out {v14.relpath(out)} `",
        f"  --permutations {PERMUTATIONS} `",
        f"  --seed {seed}",
        "```",
        "",
        "Readback: manifest.json, per_cell_rank.csv, permutation_summary.json.",
    ])
    return "\n".join(lines) + "\n"


SOURCE_SUPPORT_FIELDS = [
    "mass_cell", "source_S", "source_U", "v14_excluded_S", "v14_excluded_U",
    "v15_excluded_S", "v15_excluded_U", "eligible_S", "eligible_U",
    "supported", "disposition",
]

SAMPLE_FRAME_FIELDS = v15.SAMPLE_FRAME_FIELDS

PER_ROW_FIELDS = [
    "sample_ordinal", "orbit_index", "mass_cell", "fragile_band", "m1", "m2", "m3",
    "mass_a", "mass_b", "mass_c", "period", "stability", "status",
    "score", "vf_0", "vf_37", "vf_90", "vf_211", "zone_index_vf0", "frame_spread",
    "secondary_vf0", "total_seconds",
    "status_0", "status_37", "status_90", "status_211",
    "symplecticity_residual_0", "symplecticity_residual_37",
    "symplecticity_residual_90", "symplecticity_residual_211",
    "reciprocal_pair_residual_0", "reciprocal_pair_residual_37",
    "reciprocal_pair_residual_90", "reciprocal_pair_residual_211",
    "integration_error_stage", "integration_error_message",
]


def command_prepare(args: argparse.Namespace) -> None:
    if args.stable_per_cell != STABLE_PER_CELL or args.unstable_per_cell != UNSTABLE_PER_CELL:
        raise SystemExit(
            f"v0.16 is locked at {STABLE_PER_CELL}/{UNSTABLE_PER_CELL}; got "
            f"{args.stable_per_cell}/{args.unstable_per_cell}"
        )
    if args.seed != SEED:
        raise SystemExit(f"v0.16 sample seed is locked at {SEED}; got {args.seed}")
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.16 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")

    started = v14.utc_now()
    target = v14.resolve_target(args.target)
    v14_dir = ROOT / args.v14 if not Path(args.v14).is_absolute() else Path(args.v14)
    v15_dir = ROOT / args.v15 if not Path(args.v15).is_absolute() else Path(args.v15)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target_sha = v14.sha256_file(target)
    receipts, holdout14, holdout15 = check_prior_receipts(target, target_sha, v14_dir, v15_dir)
    holdout = holdout14 | holdout15
    rows = v14.parse_liao2021(target)
    if len(rows) != 135445:
        raise SystemExit(f"expected 135445 liao2021 rows, parsed {len(rows)}")

    _mass_cell_rows, cell_of = v14.build_mass_cells(rows)
    census, supported_cells = source_support_census(
        rows, cell_of, holdout14, holdout15, args.stable_per_cell, args.unstable_per_cell
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
        raise SystemExit("v0.16 draw included a v0.14/v0.15 holdout orbit")

    v14.write_csv(out / "source_support_census.csv", census, SOURCE_SUPPORT_FIELDS)
    v14.write_csv(out / "sample_frame.csv", sample_rows, SAMPLE_FRAME_FIELDS)
    (out / "operator_commands.md").write_text(
        operator_commands(
            out, target, v14_dir, v15_dir, args.stable_per_cell,
            args.unstable_per_cell, args.seed, args.shard_count,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "sundog.isotrophy.v0.16-liao2021-tail-resolved-transfer.v1",
        "form_lock": FORM_LOCK,
        "stage": "prepared",
        "verdict": "tail_resolved_transfer_prepared_not_measured",
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
        "v14_sample_excluded_count": len(holdout14),
        "v15_sample_excluded_count": len(holdout15),
        "double_holdout_excluded_count": len(holdout),
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
            "uses": ["mass_cell", "source_stability_label", "sample_seed", "v14_v15_holdout"],
            "forbidden": ["velocity_fraction", "score", "zone_index", "D5_output", "AUC", "p_value"],
        },
    }
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v16:prepare] supported cells={len(supported_cells)} {supported_cells}")
    print(f"[v16:prepare] drew {len(sample_rows)} rows; excluded {len(holdout)} v0.14/v0.15 rows")
    print(f"[v16:prepare] wrote {v14.relpath(out / 'sample_frame.csv')}")


def state_for_frame(masses: np.ndarray, x0: np.ndarray, v0: np.ndarray, deg: int) -> tuple[np.ndarray, np.ndarray]:
    if deg == 0:
        return x0, v0
    return v14.transform_state(masses, x0, v0, f"rot{deg}")


def measure_ensemble(masses: np.ndarray, x0: np.ndarray, v0: np.ndarray, period: float) -> dict:
    started = time.perf_counter()
    frame_measures: dict[int, dict] = {}
    for deg in FRAME_DEGREES:
        xf, vf_state = state_for_frame(masses, x0, v0, deg)
        frame_measures[deg] = v14.measure_state(masses, xf, vf_state, period)

    statuses = {deg: frame_measures[deg]["status"] for deg in FRAME_DEGREES}
    success = all(statuses[deg] == "success" for deg in FRAME_DEGREES)
    rec: dict[str, Any] = {
        "status": "success" if success else (
            "integration_blocked"
            if any(statuses[deg] == "integration_blocked" for deg in FRAME_DEGREES)
            else "sanity_failed"
        ),
        "total_seconds": time.perf_counter() - started,
        "integration_error_stage": None,
        "integration_error_message": None,
    }
    for deg in FRAME_DEGREES:
        meas = frame_measures[deg]
        rec[f"status_{deg}"] = meas.get("status")
        rec[f"vf_{deg}"] = meas.get("velocity_fraction")
        rec[f"symplecticity_residual_{deg}"] = meas.get("symplecticity_residual")
        rec[f"reciprocal_pair_residual_{deg}"] = meas.get("reciprocal_pair_residual")
        if meas.get("status") != "success" and rec["integration_error_stage"] is None:
            rec["integration_error_stage"] = f"frame_{deg}:{meas.get('integration_error_stage')}"
            rec["integration_error_message"] = meas.get("integration_error_message")

    if success:
        vfs = [float(rec[f"vf_{deg}"]) for deg in FRAME_DEGREES]
        score = float(np.median(np.asarray(vfs, dtype=float)))
        rec.update({
            "score": score,
            "secondary_vf0": vfs[0],
            "zone_index_vf0": v14.zone_index(vfs[0]),
            "frame_spread": max(vfs) - min(vfs),
        })
    else:
        rec.update({
            "score": None,
            "secondary_vf0": rec.get("vf_0"),
            "zone_index_vf0": None,
            "frame_spread": None,
        })
    return rec


def command_shard(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.16 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
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
        meas = measure_ensemble(masses, x0, v0, row.period)
        rec = {**sample_rec, **meas}
        records.append(rec)
        print(
            f"[v16:shard {args.shard_index}/{args.shard_count}] "
            f"{pos}/{len(shard_sample)} idx={idx} cell={sample_rec['mass_cell']} "
            f"label={sample_rec['stability']} status={meas['status']} "
            f"score={meas.get('score')} t={float(meas['total_seconds']):.1f}s",
            flush=True,
        )

    sample_path = out / f"sample_shard_{args.shard_index:02d}_of_{args.shard_count}.csv"
    v14.write_csv(sample_path, records, PER_ROW_FIELDS)
    summary = {
        "schema": "sundog.isotrophy.v0.16-shard.v1",
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
    print(f"[v16:shard] wrote {v14.relpath(sample_path)}")


def command_merge(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.16 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
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
    v14.write_csv(out / "per_row_sample.csv", shard_rows, PER_ROW_FIELDS)

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
        "verdict": "tail_resolved_transfer_merged_not_analyzed",
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
    print(f"[v16:merge] rows={n} success={success} blocked={blocked} sanity={sanity}")
    print(f"[v16:merge] attrition={attr:.4f} Wilson95=[{lo:.4f}, {hi:.4f}]")


def midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def j_from_scores(stable_scores: list[float], unstable_scores: list[float]) -> float:
    scores = np.asarray(stable_scores + unstable_scores, dtype=float)
    ranks = midranks(scores)
    s = len(stable_scores)
    return float(np.sum(ranks[:s]) - s * (s + 1) / 2.0)


def continuous_cell_stats(rows: list[dict], score_field: str) -> dict:
    stable = [float(r[score_field]) for r in rows if r["stability"] == "S"]
    unstable = [float(r[score_field]) for r in rows if r["stability"] == "U"]
    d = len(stable) * len(unstable)
    j = j_from_scores(stable, unstable) if d else 0.0
    return {
        "N_success": len(rows),
        "S": len(stable),
        "U": len(unstable),
        "J_cell": j,
        "D_cell": d,
        "AUC_cell": (j / d if d else None),
        "S_score_p25": quantile(stable, 0.25),
        "S_score_p50": quantile(stable, 0.50),
        "S_score_p75": quantile(stable, 0.75),
        "U_score_p25": quantile(unstable, 0.25),
        "U_score_p50": quantile(unstable, 0.50),
        "U_score_p75": quantile(unstable, 0.75),
    }


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def permutation_p_continuous(primary_rows_by_cell: dict[str, list[dict]], score_field: str,
                             observed_j: float, permutations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    d_total = 0
    for cell, rows in sorted(primary_rows_by_cell.items()):
        scores = np.asarray([float(r[score_field]) for r in rows], dtype=float)
        ranks = midranks(scores)
        s_count = sum(1 for r in rows if r["stability"] == "S")
        d = s_count * (len(rows) - s_count)
        d_total += d
        cells.append((cell, ranks, s_count, d))

    ge = 0
    max_j = -1.0
    min_j = float("inf")
    sum_j = 0.0
    for _ in range(permutations):
        j_perm = 0.0
        for _cell, ranks, s_count, _d in cells:
            idx = rng.choice(len(ranks), size=s_count, replace=False)
            j_perm += float(np.sum(ranks[idx]) - s_count * (s_count + 1) / 2.0)
        ge += int(j_perm >= observed_j - 1e-12)
        max_j = max(max_j, j_perm)
        min_j = min(min_j, j_perm)
        sum_j += j_perm
    return {
        "permutations": permutations,
        "seed": seed,
        "observed_J": observed_j,
        "D_cond": d_total,
        "observed_AUC": observed_j / d_total if d_total else None,
        "ge_observed": ge,
        "p_perm": (ge + 1) / (permutations + 1),
        "null_AUC_mean": (sum_j / permutations / d_total if permutations and d_total else None),
        "null_AUC_min": (min_j / d_total if d_total else None),
        "null_AUC_max": (max_j / d_total if d_total else None),
    }


def dispersion_rows(success_rows: list[dict]) -> tuple[list[dict], dict]:
    groups: list[tuple[str, str, list[dict]]] = [("overall", "all", success_rows)]
    by_cell: dict[str, list[dict]] = defaultdict(list)
    by_band: dict[str, list[dict]] = defaultdict(list)
    for row in success_rows:
        by_cell[row["mass_cell"]].append(row)
        by_band[row["fragile_band"]].append(row)
    groups.extend(("mass_cell", cell, rows) for cell, rows in sorted(by_cell.items()))
    groups.extend(("fragile_band", band, rows) for band, rows in sorted(by_band.items()))

    out = []
    overall = {}
    for group_type, group, rows in groups:
        spreads = [float(r["frame_spread"]) for r in rows if r.get("frame_spread", "") != ""]
        rec = {
            "group_type": group_type,
            "group": group,
            "N": len(spreads),
            "median": quantile(spreads, 0.50),
            "p75": quantile(spreads, 0.75),
            "p90": quantile(spreads, 0.90),
            "p95": quantile(spreads, 0.95),
            "max": max(spreads) if spreads else None,
        }
        if group_type == "overall":
            overall = rec
        out.append(rec)
    return out, overall


def coarse_zone_relationship(success_rows: list[dict], primary: dict[str, list[dict]]) -> tuple[list[dict], float | None]:
    rows_out = []
    zone_primary: dict[str, list[dict]] = {}
    for cell, rows in primary.items():
        zrows = [dict(r, zone_index=r["zone_index_vf0"]) for r in rows]
        zone_primary[cell] = zrows
        stats = v14.observed_cell_stats(zrows)
        rows_out.append({
            "kind": "zone_auc_cell",
            "mass_cell": cell,
            "zone_index_vf0": "",
            "stability": "",
            "N": stats["N_success"],
            "S": stats["S"],
            "U": stats["U"],
            "J": stats["J"],
            "D": stats["D"],
            "AUC": stats["AUC"],
            "score_p5": "",
            "score_p25": "",
            "score_p50": "",
            "score_p75": "",
            "score_p95": "",
        })
    j = sum(float(r["J"]) for r in rows_out if r["kind"] == "zone_auc_cell")
    d = sum(int(r["D"]) for r in rows_out if r["kind"] == "zone_auc_cell")
    zone_auc = j / d if d else None
    rows_out.append({
        "kind": "zone_auc_pooled",
        "mass_cell": "pooled",
        "zone_index_vf0": "",
        "stability": "",
        "N": sum(len(v) for v in primary.values()),
        "S": "",
        "U": "",
        "J": j,
        "D": d,
        "AUC": zone_auc,
        "score_p5": "",
        "score_p25": "",
        "score_p50": "",
        "score_p75": "",
        "score_p95": "",
    })
    for zone in (0, 1, 2):
        for label in ("S", "U", "all"):
            subset = [
                float(r["score"]) for r in success_rows
                if int(r["zone_index_vf0"]) == zone and (label == "all" or r["stability"] == label)
            ]
            rows_out.append({
                "kind": "score_by_zone",
                "mass_cell": "pooled",
                "zone_index_vf0": zone,
                "stability": label,
                "N": len(subset),
                "S": "",
                "U": "",
                "J": "",
                "D": "",
                "AUC": "",
                "score_p5": quantile(subset, 0.05),
                "score_p25": quantile(subset, 0.25),
                "score_p50": quantile(subset, 0.50),
                "score_p75": quantile(subset, 0.75),
                "score_p95": quantile(subset, 0.95),
            })
    return rows_out, zone_auc


def determine_verdict(manifest: dict, primary_cells: int, primary_rows: int,
                      frame_median: float | None, frame_p90: float | None,
                      auc: float | None, p_perm: float | None) -> str:
    attr = manifest.get("attrition_fraction")
    attr_hi = manifest.get("attrition_wilson95_high")
    if attr is None or attr_hi is None or frame_median is None or frame_p90 is None:
        return "tail_resolved_transfer_blocked_by_receipt"
    if attr > v14.ATTRITION_ALLOWED_MAX or attr_hi > v14.ATTRITION_WILSON_HIGH_MAX:
        return "tail_resolved_transfer_blocked_by_attrition"
    if frame_median > FRAME_WARN_MEDIAN or frame_p90 > FRAME_WARN_P90:
        return "tail_resolved_transfer_blocked_by_frame_instability"
    if primary_cells < PRIMARY_MIN_CELLS or primary_rows < PRIMARY_MIN_ROWS:
        return "tail_resolved_transfer_undecidable_coverage"
    if auc is None or p_perm is None:
        return "tail_resolved_transfer_blocked_by_receipt"
    if auc >= v14.AUC_PASS_FLOOR and p_perm <= v14.P_VALUE_THRESHOLD:
        if attr <= v14.ATTRITION_CLEAN_MAX and frame_median <= FRAME_CLEAN_MEDIAN and frame_p90 <= FRAME_CLEAN_P90:
            return "tail_resolved_transfer_passes_clean"
        return "tail_resolved_transfer_passes_with_warning"
    if 0.50 < auc < v14.AUC_PASS_FLOOR and p_perm <= v14.P_VALUE_THRESHOLD:
        return "tail_resolved_transfer_directional_weak"
    return "tail_resolved_transfer_fails"


def command_analyze(args: argparse.Namespace) -> None:
    if args.permutations != PERMUTATIONS:
        raise SystemExit(f"v0.16 permutation count is locked at {PERMUTATIONS}; got {args.permutations}")
    if args.seed != SEED:
        raise SystemExit(f"v0.16 permutation seed is locked at {SEED}; got {args.seed}")
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
        stats = continuous_cell_stats(cell_rows, "score")
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
        permutation_p_continuous(primary, "score", j_cond, args.permutations, args.seed)
        if primary and d_cond else
        {"permutations": args.permutations, "seed": args.seed, "observed_J": j_cond,
         "D_cond": d_cond, "observed_AUC": auc_cond, "ge_observed": None, "p_perm": None}
    )

    vf0_cell = []
    for cell, cell_rows in sorted(primary.items()):
        vf0_cell.append({"mass_cell": cell, **continuous_cell_stats(cell_rows, "vf_0")})
    vf0_j = sum(float(r["J_cell"]) for r in vf0_cell)
    vf0_d = sum(int(r["D_cell"]) for r in vf0_cell)
    vf0_auc = vf0_j / vf0_d if vf0_d else None

    dispersion, overall_dispersion = dispersion_rows(success_rows)
    coarse_rows, zone_auc = coarse_zone_relationship(success_rows, primary)
    verdict = determine_verdict(
        manifest, len(primary), primary_rows, overall_dispersion.get("median"),
        overall_dispersion.get("p90"), auc_cond, perm.get("p_perm"),
    )

    v14.write_csv(out / "per_cell_rank.csv", per_cell, [
        "mass_cell", "primary", "N_success", "S", "U", "J_cell", "D_cell", "AUC_cell",
        "S_score_p25", "S_score_p50", "S_score_p75",
        "U_score_p25", "U_score_p50", "U_score_p75",
    ])
    v14.write_csv(out / "frame_dispersion.csv", dispersion, [
        "group_type", "group", "N", "median", "p75", "p90", "p95", "max",
    ])
    v14.write_csv(out / "coarse_zone_relationship.csv", coarse_rows, [
        "kind", "mass_cell", "zone_index_vf0", "stability", "N", "S", "U", "J", "D",
        "AUC", "score_p5", "score_p25", "score_p50", "score_p75", "score_p95",
    ])
    perm_summary = {
        **perm,
        "primary_supported_cells": len(primary),
        "primary_success_rows": primary_rows,
        "frame_spread_summary": overall_dispersion,
        "secondary_vf0_AUC_cond": vf0_auc,
        "zone_auc_cond_report_only": zone_auc,
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
        "frame_spread_median": overall_dispersion.get("median"),
        "frame_spread_p90": overall_dispersion.get("p90"),
        "AUC_cond": auc_cond,
        "J_cond": j_cond,
        "D_cond": d_cond,
        "p_perm": perm.get("p_perm"),
        "secondary_vf0_AUC_cond": vf0_auc,
        "zone_auc_cond_report_only": zone_auc,
        "target_tier": v14.TARGET_TIER,
    })
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v16:analyze] primary supported cells={len(primary)} rows={primary_rows}")
    if auc_cond is not None:
        print(f"[v16:analyze] AUC_cond={auc_cond:.4f} p_perm={perm.get('p_perm')}")
    print(f"[v16:analyze] frame median={overall_dispersion.get('median')} p90={overall_dispersion.get('p90')}")
    print(f"[v16:analyze] verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="draw v0.14/v0.15-held-out 80/80 supported-cell sample")
    p.add_argument("--target", default=str(v14.DEFAULT_TARGET))
    p.add_argument("--v14", default=str(DEFAULT_V14))
    p.add_argument("--v15", default=str(DEFAULT_V15))
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

    a = sub.add_parser("analyze", help="compute tail-resolved AUC and permutation p-value")
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
