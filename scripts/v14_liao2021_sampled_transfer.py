#!/usr/bin/env python3
"""v0.14 liao2021 sampled zone-transfer runner.

Locked by docs/isotrophy/kfacet/kfacet_v14_liao2021_sampled_transfer_form.md.

Stages and scores the first sampled transfer test from the supp-B v0.11 coarse
velocity-fraction zone rule to the Tier-2 Li/Liao 2021 non-hierarchical catalog.

Commands:
  prepare  Draw the signal-blind 16-cell mass-stratified sample.
  shard    Run frozen D5 + zone + frame-zone audit for one shard.
  merge    Merge shard receipts and compute attrition.
  analyze  Compute within-cell conditional AUC and permutation p-value.

The runner deliberately keeps the long D5 work behind shard commands; prepare,
merge, and analyze are cheap bookkeeping/statistical steps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isotrophy_workbench import canonical_omega_18  # type: ignore
from scripts.v07a_velocity_fraction_audit import (  # type: ignore
    ATOL,
    MAX_STEP_FRACTION,
    RECIPROCAL_PAIR_GATE,
    RTOL,
    SYMPLECTICITY_GATE,
    compute_monodromy_vectorized,
    reciprocal_pair_residual,
    select_gamma_1,
    symplecticity_residual,
    velocity_fraction_and_z_fraction,
)
from scripts.v13a_liao2021_preflight import (  # type: ignore
    PARITY_TRANSLATION,
    STAGED,
    _rotation_z,
    expand_liao2021_state,
    integrate_liao2021_state,
    parse_liao2021,
)

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v14_liao2021_sampled_transfer_form.md"
DEFAULT_TARGET = ROOT / "docs/isotrophy/external_targets/liao2021_nonhierarchical.txt"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v14-liao2021-sampled-transfer"
SOURCE_URL = (
    "https://raw.githubusercontent.com/sjtu-liao/three-body/main/"
    "non-hierarchical-3b-supplementary_data.txt"
)

TARGET_SLUG = "liao2021_nonhierarchical_unequal_mass_135445"
TARGET_TIER = 2
SEED = 20260523
SAMPLE_ROWS_PER_CELL = 80
MASS_CELL_COUNT = 16
SHARD_COUNT = 16
PERMUTATIONS = 100000

CUTPOINTS = (0.25, 0.50)
FRAMES_ROT_DEG = (37.0, 90.0, 211.0)
FRAGILE_BANDS = (
    ("band_A", 74697, 84708),
    ("band_B", 109861, 116560),
)

ATTRITION_CLEAN_MAX = 0.05
ATTRITION_ALLOWED_MAX = 0.10
ATTRITION_WILSON_HIGH_MAX = 0.20
FRAME_CLEAN_MAX = 0.05
FRAME_ALLOWED_MAX = 0.15
AUC_PASS_FLOOR = 0.55
P_VALUE_THRESHOLD = 0.01
PRIMARY_MIN_CELLS = 10
PRIMARY_MIN_ROWS = 800
PRIMARY_CELL_MIN_N = 40
PRIMARY_CELL_MIN_S = 4
PRIMARY_CELL_MIN_U = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", ""):
        return False
    raise ValueError(f"cannot parse boolean {value!r}")


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(float(text))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def zone_index(vf: float) -> int:
    if vf < CUTPOINTS[0]:
        return 0
    if vf < CUTPOINTS[1]:
        return 1
    return 2


def dist_to_cutpoint(vf: float) -> float:
    return min(abs(vf - CUTPOINTS[0]), abs(vf - CUTPOINTS[1]))


def fragile_band(orbit_index: int) -> str:
    for name, lo, hi in FRAGILE_BANDS:
        if lo <= orbit_index <= hi:
            return name
    return "outside"


def resolve_target(path_text: str | None) -> Path:
    if path_text:
        path = (ROOT / path_text).resolve() if not Path(path_text).is_absolute() else Path(path_text)
        if path.exists():
            return path
        if path == DEFAULT_TARGET.resolve() and STAGED.exists():
            return STAGED
        if path == DEFAULT_TARGET.resolve():
            STAGED.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SOURCE_URL, STAGED)  # noqa: S310 - locked public data URL
            raise SystemExit(
                f"promoted target missing; fetched locked source to {STAGED}. "
                "Operator review/promotion is required before v0.14 scoring."
            )
        raise SystemExit(f"target does not exist: {path}")
    if DEFAULT_TARGET.exists():
        return DEFAULT_TARGET
    if STAGED.exists():
        return STAGED
    raise SystemExit(f"no liao2021 target found at {DEFAULT_TARGET} or {STAGED}")


def load_manifest_if_exists(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def check_prior_receipts(target_path: Path, target_sha: str) -> dict:
    """Validate the cheap gates available in the workspace.

    The v13a manifest still records the original raw-vf frame-parity failure in
    some worktrees; v13b superseded that for the coarse zone rule. This check
    therefore enforces the file hash, leakage, v13b zone-stability, and D5
    feasibility receipts, while preserving the observed v13a verdict as context.
    """
    receipts: dict[str, Any] = {
        "target_slug": TARGET_SLUG,
        "target_tier": TARGET_TIER,
        "target_path": relpath(target_path),
        "target_sha256": target_sha,
        "checks": {},
        "warnings": [],
    }

    v13a_path = ROOT / "results/isotrophy/k-facet-v13a-liao2021-preflight/manifest.json"
    v13a = load_manifest_if_exists(v13a_path)
    if not v13a:
        raise SystemExit(f"missing v13a preflight manifest: {v13a_path}")
    expected_sha = v13a.get("download_sha256")
    receipts["v13a_verdict_observed"] = v13a.get("verdict")
    receipts["checks"]["target_sha_matches_v13a"] = target_sha == expected_sha
    if expected_sha and target_sha != expected_sha:
        raise SystemExit(
            f"target SHA mismatch: {target_sha} != v13a download_sha256 {expected_sha}"
        )
    leakage = v13a.get("leakage", {})
    receipts["checks"]["leakage_bounded"] = bool(leakage.get("bounded"))
    receipts["checks"]["leakage_zero"] = float(leakage.get("leakage_fraction", 1.0)) == 0.0
    if not receipts["checks"]["leakage_bounded"] or not receipts["checks"]["leakage_zero"]:
        raise SystemExit("v13a leakage receipt is not bounded at zero")
    if v13a.get("verdict") == "adapter_not_expansion_only":
        receipts["warnings"].append(
            "v13a manifest preserves raw-vf frame-parity failure; v13b zone-stability "
            "receipt is the operative transfer gate for the coarse zone rule."
        )

    n600 = ROOT / "results/isotrophy/k-facet-v13b-frame-zone-stability/liao2021_n600/frame_zone_liao2021.json"
    suppb = ROOT / "results/isotrophy/k-facet-v13b-frame-zone-stability/frame_zone_suppb.json"
    if not n600.exists() or not suppb.exists():
        raise SystemExit("missing v13b frame-zone receipt(s)")
    liao_zone = read_json(n600)
    suppb_zone = read_json(suppb)
    receipts["v13b_liao2021_zone_change_fraction"] = liao_zone.get("zone_change_fraction")
    receipts["v13b_suppb_zone_change_fraction"] = suppb_zone.get("zone_change_fraction")
    receipts["checks"]["v13b_liao2021_zone_stable"] = (
        float(liao_zone.get("zone_change_fraction", 1.0)) <= FRAME_CLEAN_MAX
    )
    receipts["checks"]["v13b_suppb_zone_stable"] = (
        float(suppb_zone.get("zone_change_fraction", 1.0)) <= 0.15
    )
    if not receipts["checks"]["v13b_liao2021_zone_stable"]:
        raise SystemExit("v13b liao2021 frame-zone receipt does not clear the 0.05 bar")
    if not receipts["checks"]["v13b_suppb_zone_stable"]:
        raise SystemExit("v13b supp-B frame-zone receipt does not clear the 0.15 bar")

    rate_path = ROOT / "results/isotrophy/k-facet-v13-liao2021-rate-probe/manifest.json"
    rate = load_manifest_if_exists(rate_path)
    if not rate:
        raise SystemExit(f"missing v13 liao2021 rate-probe manifest: {rate_path}")
    receipts["v13_rate_probe"] = {
        "probe_rows": rate.get("probe_rows"),
        "success": rate.get("success"),
        "integration_blocked": rate.get("integration_blocked"),
        "sanity_failed": rate.get("sanity_failed"),
        "attrition_fraction": rate.get("attrition_fraction"),
        "attrition_wilson95_high": rate.get("attrition_wilson95_high"),
        "seconds_per_row": rate.get("seconds_per_row"),
    }
    receipts["checks"]["v13_rate_attrition_clean"] = (
        float(rate.get("attrition_fraction", 1.0)) <= ATTRITION_ALLOWED_MAX
        and float(rate.get("attrition_wilson95_high", 1.0)) <= ATTRITION_WILSON_HIGH_MAX
    )
    if not receipts["checks"]["v13_rate_attrition_clean"]:
        raise SystemExit("v13 liao2021 rate-probe attrition does not clear the feasibility bar")

    inventory = ROOT / "results/isotrophy/k-facet-v13-external-target-search/target_inventory.csv"
    if inventory.exists():
        row = next((r for r in read_csv(inventory) if r.get("slug") == TARGET_SLUG), None)
        receipts["checks"]["target_inventory_row_present"] = row is not None
        receipts["checks"]["target_inventory_tier2"] = row is not None and int(row["independence_tier"]) == 2
        receipts["checks"]["target_inventory_included"] = row is not None and parse_bool(row["include"])
        if row is None or int(row["independence_tier"]) != 2 or not parse_bool(row["include"]):
            raise SystemExit("v13 target inventory does not contain an included Tier-2 liao2021 row")
    else:
        receipts["warnings"].append("v13 target_inventory.csv missing; downstream hard gates still enforced")

    return receipts


def split_evenly(indices: list[int], parts: int) -> list[list[int]]:
    n = len(indices)
    q, r = divmod(n, parts)
    out = []
    cursor = 0
    for i in range(parts):
        size = q + (1 if i < r else 0)
        out.append(indices[cursor:cursor + size])
        cursor += size
    return out


def normalized_sorted_masses(row: Any) -> tuple[float, float, float]:
    masses = np.asarray([row.m1, row.m2, row.m3], dtype=float)
    mu = np.sort(masses / masses.sum())
    return float(mu[0]), float(mu[1]), float(mu[2])


def build_mass_cells(rows: list[Any]) -> tuple[list[dict], dict[int, str]]:
    mass_recs = []
    for row in rows:
        a, b, c = normalized_sorted_masses(row)
        mass_recs.append({"orbit_index": row.index, "a": a, "b": b, "c": c})

    by_index = {r["orbit_index"]: r for r in mass_recs}
    sorted_by_a = sorted(by_index, key=lambda idx: (by_index[idx]["a"], by_index[idx]["b"], idx))
    cell_of: dict[int, str] = {}
    cell_rows = []
    for qa, a_indices in enumerate(split_evenly(sorted_by_a, 4)):
        sorted_by_b = sorted(a_indices, key=lambda idx: (by_index[idx]["b"], by_index[idx]["a"], idx))
        for qb, b_indices in enumerate(split_evenly(sorted_by_b, 4)):
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


def draw_sample(rows: list[Any], cell_of: dict[int, str], rows_per_cell: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    indices_by_cell: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        indices_by_cell[cell_of[row.index]].append(row.index)

    row_by_index = {row.index: row for row in rows}
    sample = []
    ordinal = 0
    for cell in sorted(indices_by_cell):
        indices = sorted(indices_by_cell[cell])
        take = min(rows_per_cell, len(indices))
        chosen = sorted(int(indices[i]) for i in rng.choice(len(indices), size=take, replace=False))
        for idx in chosen:
            row = row_by_index[idx]
            a, b, c = normalized_sorted_masses(row)
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
                "fragile_band": fragile_band(row.index),
            })
            ordinal += 1
    return sample


def source_profile(rows: list[Any], cell_rows: list[dict], sample_rows: list[dict]) -> list[dict]:
    labels = Counter(row.stability for row in rows)
    sample_labels = Counter(r["stability"] for r in sample_rows)
    sample_bands = Counter(r["fragile_band"] for r in sample_rows)
    return [
        {"metric": "target_slug", "value": TARGET_SLUG},
        {"metric": "target_tier", "value": TARGET_TIER},
        {"metric": "row_count", "value": len(rows)},
        {"metric": "source_stable_S", "value": labels.get("S", 0)},
        {"metric": "source_unstable_U", "value": labels.get("U", 0)},
        {"metric": "mass_cells", "value": len(cell_rows)},
        {"metric": "sample_rows", "value": len(sample_rows)},
        {"metric": "sample_stable_S", "value": sample_labels.get("S", 0)},
        {"metric": "sample_unstable_U", "value": sample_labels.get("U", 0)},
        {"metric": "sample_fragile_band_A", "value": sample_bands.get("band_A", 0)},
        {"metric": "sample_fragile_band_B", "value": sample_bands.get("band_B", 0)},
        {"metric": "sample_fragile_outside", "value": sample_bands.get("outside", 0)},
    ]


def operator_commands(out: Path, target: Path, rows_per_cell: int, seed: int, shard_count: int) -> str:
    lines = [
        "# v0.14 liao2021 Sampled Transfer Operator Commands",
        "",
        "Run from the repository root. Shards may be run concurrently.",
        "",
        "```powershell",
        "python scripts/v14_liao2021_sampled_transfer.py prepare `",
        f"  --target {relpath(target)} `",
        f"  --out {relpath(out)} `",
        f"  --sample-rows-per-cell {rows_per_cell} `",
        f"  --seed {seed}",
        "```",
        "",
    ]
    for i in range(shard_count):
        lines.extend([
            f"## Shard {i}",
            "",
            "```powershell",
            "python scripts/v14_liao2021_sampled_transfer.py shard `",
            f"  --out {relpath(out)} `",
            f"  --shard-index {i} `",
            f"  --shard-count {shard_count}",
            "```",
            "",
        ])
    lines.extend([
        "## Merge",
        "",
        "```powershell",
        "python scripts/v14_liao2021_sampled_transfer.py merge `",
        f"  --out {relpath(out)} `",
        f"  --shard-count {shard_count}",
        "```",
        "",
        "## Analyze",
        "",
        "```powershell",
        "python scripts/v14_liao2021_sampled_transfer.py analyze `",
        f"  --out {relpath(out)} `",
        f"  --permutations {PERMUTATIONS} `",
        f"  --seed {seed}",
        "```",
        "",
        "Readback: manifest.json, per_cell_rank.csv, permutation_summary.json.",
    ])
    return "\n".join(lines) + "\n"


def command_prepare(args: argparse.Namespace) -> None:
    started = utc_now()
    target = resolve_target(args.target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target_sha = sha256_file(target)
    prior = check_prior_receipts(target, target_sha)
    rows = parse_liao2021(target)
    if len(rows) != 135445:
        raise SystemExit(f"expected 135445 liao2021 rows, parsed {len(rows)}")

    cell_rows, cell_of = build_mass_cells(rows)
    sample_rows = draw_sample(rows, cell_of, args.sample_rows_per_cell, args.seed)
    if len(cell_rows) != MASS_CELL_COUNT:
        raise SystemExit(f"expected {MASS_CELL_COUNT} mass cells, got {len(cell_rows)}")

    write_csv(out / "source_profile.csv", source_profile(rows, cell_rows, sample_rows), ["metric", "value"])
    write_csv(out / "mass_cells.csv", cell_rows, [
        "mass_cell", "qA", "qB", "row_count", "a_min", "a_max", "b_min", "b_max",
        "orbit_index_min", "orbit_index_max",
    ])
    write_csv(out / "sample_frame.csv", sample_rows, [
        "sample_ordinal", "orbit_index", "mass_cell", "m1", "m2", "m3", "mass_a",
        "mass_b", "mass_c", "period", "stability", "fragile_band",
    ])
    (out / "operator_commands.md").write_text(
        operator_commands(out, target, args.sample_rows_per_cell, args.seed, args.shard_count),
        encoding="utf-8",
    )
    manifest = {
        "schema": "sundog.isotrophy.v0.14-liao2021-sampled-transfer.v1",
        "form_lock": FORM_LOCK,
        "stage": "prepared",
        "verdict": "prepared_not_measured",
        "startedAt": started,
        "completedAt": utc_now(),
        "gitCommit": git_commit(),
        "target_slug": TARGET_SLUG,
        "target_tier": TARGET_TIER,
        "target_path": relpath(target),
        "target_sha256": target_sha,
        "prior_receipts": prior,
        "sample_seed": args.seed,
        "sample_rows_per_cell": args.sample_rows_per_cell,
        "sample_rows_requested": len(sample_rows),
        "sample_rows_success": None,
        "mass_cells": len(cell_rows),
        "shard_count": args.shard_count,
        "cutpoints": CUTPOINTS,
        "frozen_d5": {
            "rtol": RTOL,
            "atol": ATOL,
            "max_step_fraction": MAX_STEP_FRACTION,
            "symplecticity_gate": SYMPLECTICITY_GATE,
            "reciprocal_pair_gate": RECIPROCAL_PAIR_GATE,
        },
    }
    write_json(out / "manifest.json", manifest)
    print(f"[v14:prepare] parsed {len(rows)} rows; drew {len(sample_rows)} rows across {len(cell_rows)} cells")
    print(f"[v14:prepare] wrote {relpath(out / 'sample_frame.csv')} and operator commands")


def measure_state(masses: np.ndarray, x0: np.ndarray, v0: np.ndarray, period: float) -> dict:
    started = time.perf_counter()
    try:
        t_orbit = time.perf_counter()
        integrated = integrate_liao2021_state(masses, x0, v0, period)
        orbit_seconds = time.perf_counter() - t_orbit
        t_mono = time.perf_counter()
        M_i = compute_monodromy_vectorized(integrated, RTOL, ATOL, MAX_STEP_FRACTION)
        monodromy_seconds = time.perf_counter() - t_mono
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "integration_blocked",
            "integration_error_stage": "d5_integration",
            "integration_error_message": f"{type(exc).__name__}: {exc}",
            "orbit_integration_seconds": None,
            "monodromy_integration_seconds": None,
            "total_seconds": time.perf_counter() - started,
            "symplecticity_residual": None,
            "symplecticity_status": "integration_blocked",
            "reciprocal_pair_residual": None,
            "reciprocal_pair_status": "integration_blocked",
            "velocity_fraction": None,
            "zone_index": None,
            "distance_to_nearest_cutpoint": None,
            "z_fraction": None,
        }

    omega = canonical_omega_18(masses)
    symp = symplecticity_residual(M_i, omega)
    recip = reciprocal_pair_residual(np.linalg.eigvals(M_i))
    symp_status = "pass" if symp <= SYMPLECTICITY_GATE else "fail"
    recip_status = "pass" if recip <= RECIPROCAL_PAIR_GATE else "fail"
    status = "success" if symp_status == "pass" and recip_status == "pass" else "sanity_failed"

    vf = zf = dist = zone = None
    gamma_info: dict[str, Any] = {}
    if status == "success":
        gamma_info = select_gamma_1(M_i, masses)
        feat = velocity_fraction_and_z_fraction(gamma_info["gamma_1"], masses)
        vf = float(feat["velocity_fraction"])
        zf = float(feat["z_fraction"])
        zone = zone_index(vf)
        dist = dist_to_cutpoint(vf)

    return {
        "status": status,
        "integration_error_stage": None if status != "sanity_failed" else "sanity_gate",
        "integration_error_message": None if status != "sanity_failed" else "symplecticity_or_reciprocal_pair",
        "orbit_integration_seconds": orbit_seconds,
        "monodromy_integration_seconds": monodromy_seconds,
        "total_seconds": time.perf_counter() - started,
        "symplecticity_residual": symp,
        "symplecticity_status": symp_status,
        "reciprocal_pair_residual": recip,
        "reciprocal_pair_status": recip_status,
        "velocity_fraction": vf,
        "zone_index": zone,
        "distance_to_nearest_cutpoint": dist,
        "z_fraction": zf,
        "max_eigenvalue_real_part": gamma_info.get("max_real_part"),
        "representative_eigenvalue_real": gamma_info.get("eigenvalue_real"),
        "representative_eigenvalue_imag": gamma_info.get("eigenvalue_imag"),
        "representative_eigenvalue_modulus": gamma_info.get("eigenvalue_modulus"),
        "degenerate_eigenvalue_count": gamma_info.get("degenerate_eigenvalue_count"),
        "eigenspace_dim_used": gamma_info.get("eigenspace_dim_used"),
        "cascade_step_used": gamma_info.get("cascade_step_used"),
    }


def transform_state(masses: np.ndarray, x0: np.ndarray, v0: np.ndarray, frame: str) -> tuple[np.ndarray, np.ndarray]:
    if frame.startswith("rot"):
        deg = float(frame[3:])
        R = _rotation_z(deg)
        return x0 @ R.T, v0 @ R.T
    if frame == "translation":
        xt = x0 + PARITY_TRANSLATION
        xt = xt - np.average(xt, axis=0, weights=masses)
        return xt, v0
    raise ValueError(f"unknown frame {frame}")


def frame_audit_records(sample_rec: dict, masses: np.ndarray, x0: np.ndarray, v0: np.ndarray,
                        base_zone: int, period: float) -> list[dict]:
    recs = []
    frames = [f"rot{int(d)}" for d in FRAMES_ROT_DEG] + ["translation"]
    for frame in frames:
        try:
            xf, vf0 = transform_state(masses, x0, v0, frame)
            meas = measure_state(masses, xf, vf0, period)
            rotated_zone = meas.get("zone_index") if meas["status"] == "success" else None
            recs.append({
                "sample_ordinal": sample_rec["sample_ordinal"],
                "orbit_index": sample_rec["orbit_index"],
                "mass_cell": sample_rec["mass_cell"],
                "fragile_band": sample_rec["fragile_band"],
                "frame": frame,
                "base_zone": base_zone,
                "rotated_zone": rotated_zone,
                "zone_changed": (rotated_zone is not None and int(rotated_zone) != int(base_zone)),
                "rotated_status": meas["status"],
                "rotated_distance_to_nearest_cutpoint": meas.get("distance_to_nearest_cutpoint"),
                "rotated_total_seconds": meas.get("total_seconds"),
            })
        except Exception as exc:  # noqa: BLE001
            recs.append({
                "sample_ordinal": sample_rec["sample_ordinal"],
                "orbit_index": sample_rec["orbit_index"],
                "mass_cell": sample_rec["mass_cell"],
                "fragile_band": sample_rec["fragile_band"],
                "frame": frame,
                "base_zone": base_zone,
                "rotated_zone": None,
                "zone_changed": False,
                "rotated_status": "frame_exception",
                "rotated_distance_to_nearest_cutpoint": None,
                "rotated_total_seconds": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return recs


SAMPLE_SHARD_FIELDS = [
    "sample_ordinal", "orbit_index", "mass_cell", "fragile_band", "m1", "m2", "m3",
    "mass_a", "mass_b", "mass_c", "period", "stability", "status",
    "symplecticity_residual", "symplecticity_status", "reciprocal_pair_residual",
    "reciprocal_pair_status", "velocity_fraction", "zone_index",
    "distance_to_nearest_cutpoint", "z_fraction", "max_eigenvalue_real_part",
    "representative_eigenvalue_real", "representative_eigenvalue_imag",
    "representative_eigenvalue_modulus", "degenerate_eigenvalue_count",
    "eigenspace_dim_used", "cascade_step_used", "orbit_integration_seconds",
    "monodromy_integration_seconds", "total_seconds", "integration_error_stage",
    "integration_error_message",
]

FRAME_SHARD_FIELDS = [
    "sample_ordinal", "orbit_index", "mass_cell", "fragile_band", "frame", "base_zone",
    "rotated_zone", "zone_changed", "rotated_status", "rotated_distance_to_nearest_cutpoint",
    "rotated_total_seconds", "error",
]


def command_shard(args: argparse.Namespace) -> None:
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    out = Path(args.out)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("run prepare before shard")
    manifest = read_json(manifest_path)
    target = ROOT / manifest["target_path"]
    rows = parse_liao2021(target)
    row_by_index = {row.index: row for row in rows}
    sample = read_csv(out / "sample_frame.csv")
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
        masses, x0, v0 = expand_liao2021_state(row, center_com=True)
        meas = measure_state(masses, x0, v0, row.period)
        rec = {**sample_rec, **meas}
        records.append(rec)
        if meas["status"] == "success":
            frame_records.extend(
                frame_audit_records(sample_rec, masses, x0, v0, int(meas["zone_index"]), row.period)
            )
        print(
            f"[v14:shard {args.shard_index}/{args.shard_count}] "
            f"{pos}/{len(shard_sample)} idx={idx} cell={sample_rec['mass_cell']} "
            f"status={meas['status']} t={float(meas['total_seconds']):.1f}s",
            flush=True,
        )

    sample_path = out / f"sample_shard_{args.shard_index:02d}_of_{args.shard_count}.csv"
    frame_path = out / f"frame_zone_shard_{args.shard_index:02d}_of_{args.shard_count}.csv"
    write_csv(sample_path, records, SAMPLE_SHARD_FIELDS)
    write_csv(frame_path, frame_records, FRAME_SHARD_FIELDS)
    shard_summary = {
        "schema": "sundog.isotrophy.v0.14-shard.v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "rows": len(records),
        "success": sum(1 for r in records if r["status"] == "success"),
        "integration_blocked": sum(1 for r in records if r["status"] == "integration_blocked"),
        "sanity_failed": sum(1 for r in records if r["status"] == "sanity_failed"),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "completedAt": utc_now(),
    }
    write_json(out / f"shard_{args.shard_index:02d}_of_{args.shard_count}.json", shard_summary)
    print(f"[v14:shard] wrote {relpath(sample_path)} and {relpath(frame_path)}")


def command_merge(args: argparse.Namespace) -> None:
    out = Path(args.out)
    sample = read_csv(out / "sample_frame.csv")
    shard_rows = []
    frame_rows = []
    missing = []
    for i in range(args.shard_count):
        sample_path = out / f"sample_shard_{i:02d}_of_{args.shard_count}.csv"
        frame_path = out / f"frame_zone_shard_{i:02d}_of_{args.shard_count}.csv"
        if not sample_path.exists():
            missing.append(relpath(sample_path))
            continue
        shard_rows.extend(read_csv(sample_path))
        if frame_path.exists():
            frame_rows.extend(read_csv(frame_path))
    if missing:
        raise SystemExit("missing shard output(s): " + ", ".join(missing))

    expected_ord = {int(r["sample_ordinal"]) for r in sample}
    got_ord = {int(r["sample_ordinal"]) for r in shard_rows}
    if expected_ord != got_ord:
        missing_ord = sorted(expected_ord - got_ord)[:10]
        extra_ord = sorted(got_ord - expected_ord)[:10]
        raise SystemExit(f"merged shard ordinals mismatch; missing={missing_ord} extra={extra_ord}")

    shard_rows.sort(key=lambda r: int(r["sample_ordinal"]))
    frame_rows.sort(key=lambda r: (int(r["sample_ordinal"]), str(r["frame"])))
    write_csv(out / "per_row_sample.csv", shard_rows, SAMPLE_SHARD_FIELDS)
    write_csv(out / "frame_zone_audit.csv", frame_rows, FRAME_SHARD_FIELDS)

    n = len(shard_rows)
    success = sum(1 for r in shard_rows if r["status"] == "success")
    blocked = sum(1 for r in shard_rows if r["status"] == "integration_blocked")
    sanity = sum(1 for r in shard_rows if r["status"] == "sanity_failed")
    attrited = blocked + sanity
    attr = attrited / n if n else 0.0
    lo, hi = wilson_ci(attrited, n)

    manifest = read_json(out / "manifest.json")
    manifest.update({
        "stage": "merged",
        "verdict": "merged_not_analyzed",
        "completedAt": utc_now(),
        "sample_rows_requested": n,
        "sample_rows_success": success,
        "integration_blocked": blocked,
        "sanity_failed": sanity,
        "attrition_fraction": attr,
        "attrition_wilson95_low": lo,
        "attrition_wilson95_high": hi,
    })
    write_json(out / "manifest.json", manifest)
    print(f"[v14:merge] rows={n} success={success} blocked={blocked} sanity={sanity}")
    print(f"[v14:merge] attrition={attr:.4f} Wilson95=[{lo:.4f}, {hi:.4f}]")


def j_from_counts(stable_counts: np.ndarray, unstable_counts: np.ndarray) -> float:
    # Stable rows are expected to rank above unstable rows as zone_index increases.
    j = 0.0
    for zs in range(3):
        for zu in range(3):
            if zs > zu:
                j += stable_counts[zs] * unstable_counts[zu]
            elif zs == zu:
                j += 0.5 * stable_counts[zs] * unstable_counts[zu]
    return float(j)


def observed_cell_stats(rows: list[dict]) -> dict:
    zones_s = [int(r["zone_index"]) for r in rows if r["stability"] == "S"]
    zones_u = [int(r["zone_index"]) for r in rows if r["stability"] == "U"]
    s_counts = np.bincount(zones_s, minlength=3)
    u_counts = np.bincount(zones_u, minlength=3)
    d = int(len(zones_s) * len(zones_u))
    j = j_from_counts(s_counts, u_counts)
    return {
        "N_success": len(rows),
        "S": len(zones_s),
        "U": len(zones_u),
        "zone0": int(s_counts[0] + u_counts[0]),
        "zone1": int(s_counts[1] + u_counts[1]),
        "zone2": int(s_counts[2] + u_counts[2]),
        "S_zone0": int(s_counts[0]),
        "S_zone1": int(s_counts[1]),
        "S_zone2": int(s_counts[2]),
        "U_zone0": int(u_counts[0]),
        "U_zone1": int(u_counts[1]),
        "U_zone2": int(u_counts[2]),
        "J": j,
        "D": d,
        "AUC": (j / d if d else None),
    }


def permutation_p(primary_rows_by_cell: dict[str, list[dict]], observed_j: float,
                  permutations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    d_total = 0
    for cell, rows in sorted(primary_rows_by_cell.items()):
        zones = np.asarray([int(r["zone_index"]) for r in rows], dtype=int)
        s_count = sum(1 for r in rows if r["stability"] == "S")
        total_counts = np.bincount(zones, minlength=3)
        d = s_count * (len(rows) - s_count)
        d_total += d
        cells.append((cell, zones, total_counts, s_count, d))

    ge = 0
    max_j = -1.0
    min_j = float("inf")
    sum_j = 0.0
    for _ in range(permutations):
        j_perm = 0.0
        for _cell, zones, total_counts, s_count, _d in cells:
            idx = rng.choice(len(zones), size=s_count, replace=False)
            s_counts = np.bincount(zones[idx], minlength=3)
            u_counts = total_counts - s_counts
            j_perm += j_from_counts(s_counts, u_counts)
        ge += int(j_perm >= observed_j - 1e-12)
        max_j = max(max_j, j_perm)
        min_j = min(min_j, j_perm)
        sum_j += j_perm
    p = (ge + 1) / (permutations + 1)
    return {
        "permutations": permutations,
        "seed": seed,
        "observed_J": observed_j,
        "D_cond": d_total,
        "observed_AUC": observed_j / d_total if d_total else None,
        "ge_observed": ge,
        "p_perm": p,
        "null_AUC_mean": (sum_j / permutations / d_total if permutations and d_total else None),
        "null_AUC_min": (min_j / d_total if d_total else None),
        "null_AUC_max": (max_j / d_total if d_total else None),
    }


def frame_summary(frame_rows: list[dict], success_ord: set[int]) -> dict:
    by_ord: dict[int, list[dict]] = defaultdict(list)
    for r in frame_rows:
        by_ord[int(r["sample_ordinal"])].append(r)
    checked = changed = missing = 0
    band_counts: dict[str, Counter] = defaultdict(Counter)
    for ord_ in sorted(success_ord):
        rows = by_ord.get(ord_, [])
        ok_rows = [r for r in rows if r.get("rotated_status") == "success"]
        if not ok_rows:
            missing += 1
            continue
        checked += 1
        orbit_changed = any(parse_bool(r.get("zone_changed", "false")) for r in ok_rows)
        changed += int(orbit_changed)
        band = ok_rows[0].get("fragile_band", "outside")
        band_counts[band]["checked"] += 1
        band_counts[band]["changed"] += int(orbit_changed)
    frac = changed / checked if checked else None
    lo, hi = wilson_ci(changed, checked)
    return {
        "frame_orbits_expected": len(success_ord),
        "frame_orbits_checked": checked,
        "frame_orbits_missing": missing,
        "frame_zone_changed": changed,
        "zone_change_fraction": frac,
        "zone_change_wilson95_low": lo,
        "zone_change_wilson95_high": hi,
        "by_fragile_band": {
            band: {"checked": c["checked"], "changed": c["changed"],
                   "fraction": (c["changed"] / c["checked"] if c["checked"] else None)}
            for band, c in sorted(band_counts.items())
        },
    }


def determine_verdict(manifest: dict, frame: dict, primary_cells: int,
                      primary_rows: int, auc: float | None, p_perm: float | None) -> str:
    attr = manifest.get("attrition_fraction")
    attr_hi = manifest.get("attrition_wilson95_high")
    zone_frac = frame.get("zone_change_fraction")
    if frame.get("frame_orbits_missing", 0) > 0:
        return "sample_transfer_blocked_by_receipt"
    if attr is None or attr_hi is None or zone_frac is None:
        return "sample_transfer_blocked_by_receipt"
    if attr > ATTRITION_ALLOWED_MAX or attr_hi > ATTRITION_WILSON_HIGH_MAX:
        return "sample_transfer_blocked_by_attrition"
    if zone_frac > FRAME_ALLOWED_MAX:
        return "sample_transfer_blocked_by_frame_instability"
    if primary_cells < PRIMARY_MIN_CELLS or primary_rows < PRIMARY_MIN_ROWS:
        return "sample_transfer_undecidable_coverage"
    if auc is None or p_perm is None:
        return "sample_transfer_blocked_by_receipt"
    no_hard_block = attr <= ATTRITION_ALLOWED_MAX and attr_hi <= ATTRITION_WILSON_HIGH_MAX and zone_frac <= FRAME_ALLOWED_MAX
    if auc >= AUC_PASS_FLOOR and p_perm <= P_VALUE_THRESHOLD:
        if attr <= ATTRITION_CLEAN_MAX and zone_frac <= FRAME_CLEAN_MAX:
            return "sample_transfer_passes_clean"
        if no_hard_block and (attr > ATTRITION_CLEAN_MAX or zone_frac > FRAME_CLEAN_MAX):
            return "sample_transfer_passes_with_warning"
    if 0.50 < auc < AUC_PASS_FLOOR and p_perm <= P_VALUE_THRESHOLD:
        return "sample_transfer_directional_weak"
    return "sample_transfer_fails"


def command_analyze(args: argparse.Namespace) -> None:
    out = Path(args.out)
    manifest = read_json(out / "manifest.json")
    rows = read_csv(out / "per_row_sample.csv")
    frame_rows = read_csv(out / "frame_zone_audit.csv")
    success_rows = [
        r for r in rows
        if r["status"] == "success" and r.get("zone_index", "") != "" and r["stability"] in ("S", "U")
    ]
    success_ord = {int(r["sample_ordinal"]) for r in success_rows}
    frame = frame_summary(frame_rows, success_ord)

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in success_rows:
        by_cell[r["mass_cell"]].append(r)

    per_cell = []
    primary: dict[str, list[dict]] = {}
    for cell, cell_rows in sorted(by_cell.items()):
        stats = observed_cell_stats(cell_rows)
        is_primary = (
            stats["N_success"] >= PRIMARY_CELL_MIN_N
            and stats["S"] >= PRIMARY_CELL_MIN_S
            and stats["U"] >= PRIMARY_CELL_MIN_U
        )
        if is_primary:
            primary[cell] = cell_rows
        per_cell.append({"mass_cell": cell, "primary": is_primary, **stats})

    primary_rows = sum(len(v) for v in primary.values())
    j_cond = sum(float(r["J"]) for r in per_cell if parse_bool(r["primary"]))
    d_cond = sum(int(r["D"]) for r in per_cell if parse_bool(r["primary"]))
    auc_cond = j_cond / d_cond if d_cond else None
    if primary and d_cond:
        perm = permutation_p(primary, j_cond, args.permutations, args.seed)
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
    write_csv(out / "per_cell_rank.csv", per_cell, [
        "mass_cell", "primary", "N_success", "S", "U", "zone0", "zone1", "zone2",
        "S_zone0", "S_zone1", "S_zone2", "U_zone0", "U_zone1", "U_zone2",
        "J", "D", "AUC",
    ])
    perm_summary = {
        **perm,
        "primary_mass_cells": len(primary),
        "primary_success_rows": primary_rows,
        "frame_zone_summary": frame,
        "effect_floor": AUC_PASS_FLOOR,
        "p_value_threshold": P_VALUE_THRESHOLD,
        "verdict": verdict,
    }
    write_json(out / "permutation_summary.json", perm_summary)
    manifest.update({
        "stage": "analyzed",
        "verdict": verdict,
        "completedAt": utc_now(),
        "primary_mass_cells": len(primary),
        "primary_success_rows": primary_rows,
        "zone_change_fraction": frame.get("zone_change_fraction"),
        "zone_change_wilson95_low": frame.get("zone_change_wilson95_low"),
        "zone_change_wilson95_high": frame.get("zone_change_wilson95_high"),
        "AUC_cond": auc_cond,
        "J_cond": j_cond,
        "D_cond": d_cond,
        "p_perm": perm.get("p_perm"),
        "target_tier": TARGET_TIER,
    })
    write_json(out / "manifest.json", manifest)
    print(f"[v14:analyze] primary cells={len(primary)} rows={primary_rows}")
    if auc_cond is not None:
        print(f"[v14:analyze] AUC_cond={auc_cond:.4f} p_perm={perm.get('p_perm')}")
    print(f"[v14:analyze] frame zone change={frame.get('zone_change_fraction')}")
    print(f"[v14:analyze] verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="draw signal-blind mass-cell sample")
    p.add_argument("--target", default=str(DEFAULT_TARGET))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--sample-rows-per-cell", type=int, default=SAMPLE_ROWS_PER_CELL)
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

    a = sub.add_parser("analyze", help="compute conditional AUC and permutation p-value")
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
