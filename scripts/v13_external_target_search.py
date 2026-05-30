#!/usr/bin/env python3
"""v0.13 external target search harness.

Locked by docs/isotrophy/kfacet/kfacet_v13_external_target_search_form.md.

This is a source-selection harness, not a transfer test. It inventories and
profiles candidate catalogs, checks leakage against prior isotrophy evidence,
and can run D5 feasibility probes that report only measurement tractability:
success/blocked/sanity, runtime, period, mass key, and failure reason.

The probe deliberately drops `velocity_fraction`, `zone_index`, and stability
labels from per-row receipts. v0.13 is not allowed to inspect a feature-vs-S/U
signal before a target is locked for a later chapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isotrophy_workbench import (  # type: ignore
    OrbitRow,
    canonical_omega_18,
    parse_rows,
    read_text,
)
from scripts.v07a_velocity_fraction_audit import (  # type: ignore
    ATOL,
    MAX_STEP_FRACTION,
    RECIPROCAL_PAIR_GATE,
    RTOL,
    SYMPLECTICITY_GATE,
    per_row_pipeline,
)

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v13_external_target_search_form.md"
DEFAULT_OUT = "results/isotrophy/k-facet-v13-external-target-search"
PRIOR_A = "docs/isotrophy/supplementary-A_periodic-3d_mirror.txt"
PRIOR_B = "docs/isotrophy/supplementary-B_piano-init-condit-3d.txt"
SEED = 20260523
OVERLAP_TOL = 1e-9
LEAKAGE_ABORT_FRACTION = 0.05
MIN_ROWS = 200
MIN_STRATA = 5
MIN_BOTH_CLASS_STRATA = 5
MIN_STRATUM_S = 5
MIN_STRATUM_U = 5
MIN_PROBE_ROWS = 30
FULL_PROBE_ROWS = 100
ATTRITION_CLEAN_MAX = 0.10
ATTRITION_WILSON_HIGH_MAX = 0.20
SINGLE_THREAD_HOURS_MAX = 96.0
SHARDED_HOURS_MAX = 24.0
INLINE_BUDGET_SECONDS = 600.0
SUPPORTED_PARSERS = {"liao_mirror_text", "csv_mirror"}

INVENTORY_FIELDS = [
    "slug",
    "include",
    "source_path",
    "source_url",
    "access_date",
    "download_hash",
    "license_note",
    "citation_note",
    "source_family",
    "authors",
    "relationship_to_supp_a",
    "relationship_to_supp_b",
    "orbit_type",
    "dimensionality",
    "body_count",
    "potential",
    "equations",
    "newtonian_substrate",
    "mass_parameters",
    "initial_condition_fields",
    "period_field",
    "stability_label_definition",
    "stability_commensurable",
    "row_count_declared",
    "conditioning_key",
    "parser",
    "parser_source",
    "ansatz",
    "identity_key_strength",
    "independence_tier",
    "proposed_shards",
    "notes",
]


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "pass", "ok"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def tier_value(value: Any) -> int:
    text = clean(value).lower().replace("tier", "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        match = re.search(r"\d+", text)
        return int(match.group(0)) if match else 0


def m3_key(m3: float) -> str:
    return f"{m3:.9f}".rstrip("0").rstrip(".")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def source_ref(row: dict[str, str]) -> str:
    return clean(row.get("source_path")) or clean(row.get("source_url"))


def local_source_path(row: dict[str, str]) -> Path | None:
    raw = clean(row.get("source_path"))
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    return p


def csv_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and clean(row[name]) != "":
            return float(row[name])
    raise KeyError(f"missing numeric field among {names}")


def csv_int(row: dict[str, str], fallback: int, *names: str) -> int:
    for name in names:
        if name in row and clean(row[name]) != "":
            return int(float(row[name]))
    return fallback


def parse_csv_mirror(text: str, slug: str) -> list[OrbitRow]:
    rows: list[OrbitRow] = []
    reader = csv.DictReader(text.splitlines())
    for line_no, raw in enumerate(reader, start=2):
        if not raw:
            continue
        stability = clean(raw.get("stability") or raw.get("label") or raw.get("stable"))
        if stability not in {"S", "U"}:
            raise ValueError(f"row {line_no}: stability must be S/U, got {stability!r}")
        rows.append(
            OrbitRow(
                source=slug,
                line_no=line_no,
                index=csv_int(raw, len(rows) + 1, "index", "orbit_index", "O_index"),
                m3=csv_float(raw, "m3", "m_3", "mass3"),
                z0=csv_float(raw, "z0", "z_0"),
                vx=csv_float(raw, "vx", "v_x"),
                vy=csv_float(raw, "vy", "v_y"),
                vz=csv_float(raw, "vz", "v_z"),
                period=csv_float(raw, "T", "period"),
                stability=stability,
            )
        )
    return rows


def parse_candidate_rows(candidate: dict[str, str]) -> tuple[list[OrbitRow], str | None, str | None]:
    parser = clean(candidate.get("parser"))
    if parser not in SUPPORTED_PARSERS:
        return [], None, f"unsupported parser {parser!r}"
    ref = source_ref(candidate)
    if not ref:
        return [], None, "missing source_path/source_url"
    try:
        text = read_text(ref)
    except Exception as exc:  # noqa: BLE001
        return [], None, f"read failed: {type(exc).__name__}: {exc}"
    try:
        if parser == "liao_mirror_text":
            rows = parse_rows(text, source=clean(candidate.get("parser_source")) or clean(candidate["slug"]))
        elif parser == "csv_mirror":
            rows = parse_csv_mirror(text, clean(candidate["slug"]))
        else:
            rows = []
    except Exception as exc:  # noqa: BLE001
        return [], None, f"parse failed: {type(exc).__name__}: {exc}"
    return rows, hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(), None


def build_overlap_index(rows_by_source: list[tuple[str, list[OrbitRow]]]) -> dict[float, list[tuple[str, OrbitRow]]]:
    idx: dict[float, list[tuple[str, OrbitRow]]] = defaultdict(list)
    for source_name, rows in rows_by_source:
        for row in rows:
            idx[round(row.m3, 9)].append((source_name, row))
    return idx


def load_prior_index() -> dict[float, list[tuple[str, OrbitRow]]]:
    priors = [
        ("supplementary-A", parse_rows(read_text(PRIOR_A), source="A")),
        ("supplementary-B", parse_rows(read_text(PRIOR_B), source="B")),
    ]
    return build_overlap_index(priors)


def overlap_kind(row: OrbitRow, prior_index: dict[float, list[tuple[str, OrbitRow]]]) -> tuple[str | None, str | None]:
    for source_name, prior in prior_index.get(round(row.m3, 9), []):
        if abs(row.m3 - prior.m3) > OVERLAP_TOL or abs(row.period - prior.period) > OVERLAP_TOL:
            continue
        if (
            abs(row.z0 - prior.z0) <= OVERLAP_TOL
            and abs(row.vx - prior.vx) <= OVERLAP_TOL
            and abs(row.vy - prior.vy) <= OVERLAP_TOL
            and abs(row.vz - prior.vz) <= OVERLAP_TOL
        ):
            return "exact", source_name
        if (
            abs(row.z0 + prior.z0) <= OVERLAP_TOL
            and abs(row.vx - prior.vx) <= OVERLAP_TOL
            and abs(row.vy - prior.vy) <= OVERLAP_TOL
            and abs(row.vz + prior.vz) <= OVERLAP_TOL
        ):
            return "reflection", source_name
    return None, None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def profile_rows(rows: list[OrbitRow]) -> dict[str, Any]:
    by_m3: dict[str, list[OrbitRow]] = defaultdict(list)
    for row in rows:
        by_m3[m3_key(row.m3)].append(row)
    both_class = 0
    strata = []
    for key in sorted(by_m3, key=lambda x: float(x)):
        group = by_m3[key]
        s = sum(1 for r in group if r.stability == "S")
        u = sum(1 for r in group if r.stability == "U")
        both_class += int(s >= MIN_STRATUM_S and u >= MIN_STRATUM_U)
        strata.append({"mass_key": key, "N": len(group), "S": s, "U": u})
    return {
        "row_count": len(rows),
        "S_total": sum(1 for r in rows if r.stability == "S"),
        "U_total": sum(1 for r in rows if r.stability == "U"),
        "conditioning_strata_count": len(by_m3),
        "both_class_strata_count": both_class,
        "strata": strata,
    }


def hard_gate_failures(candidate: dict[str, str], rows: list[OrbitRow], profile: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if tier_value(candidate.get("independence_tier")) <= 0:
        failures.append("tier0_or_unset_independence")
    if not truthy(candidate.get("include", "true")):
        failures.append("include_false")
    if not truthy(candidate.get("newtonian_substrate")):
        failures.append("newtonian_substrate_not_confirmed")
    if clean(candidate.get("parser")) not in SUPPORTED_PARSERS:
        failures.append("unsupported_parser")
    if str(clean(candidate.get("body_count"))) not in {"3", "3.0"}:
        failures.append("body_count_not_three")
    if not clean(candidate.get("period_field")):
        failures.append("missing_period_field")
    if not truthy(candidate.get("stability_commensurable")):
        failures.append("stability_not_commensurable")
    if profile["row_count"] < MIN_ROWS:
        failures.append("row_count_below_200")
    if profile["conditioning_strata_count"] < MIN_STRATA:
        failures.append("conditioning_strata_below_5")
    if profile["both_class_strata_count"] < MIN_BOTH_CLASS_STRATA:
        failures.append("both_class_strata_below_5")
    if not source_ref(candidate):
        failures.append("missing_source_ref")
    return failures


def load_inventory(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    normalized = []
    for i, row in enumerate(rows, start=1):
        full = {field: clean(row.get(field)) for field in INVENTORY_FIELDS}
        if not full["slug"]:
            full["slug"] = f"candidate_{i}"
        if full["include"] == "":
            full["include"] = "true"
        normalized.append(full)
    return normalized


def inventory_path(out: Path) -> Path:
    return out / "target_inventory.csv"


def profile_path(out: Path) -> Path:
    return out / "source_profiles.json"


def feasibility_summary_path(out: Path) -> Path:
    return out / "feasibility_summary.json"


def run_inventory(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = inventory_path(out)
    if path.exists() and not args.force:
        print(f"[v13] inventory exists: {repo_rel(path)} (use --force to rewrite)")
    else:
        rows: list[dict[str, Any]] = []
        if args.seed_local_controls:
            rows = [
                {
                    "slug": "liao__supplementary_a_periodic_3d__tier0_control",
                    "include": "true",
                    "source_path": PRIOR_A,
                    "access_date": "2026-05-30",
                    "source_family": "Li-Liao Numerical Tank",
                    "relationship_to_supp_a": "self",
                    "relationship_to_supp_b": "same-paper",
                    "orbit_type": "3D periodic",
                    "dimensionality": "3D",
                    "body_count": "3",
                    "potential": "Newtonian",
                    "newtonian_substrate": "true",
                    "mass_parameters": "m1=m2=1,m3",
                    "initial_condition_fields": "z0,vx,vy,vz",
                    "period_field": "T",
                    "stability_label_definition": "linear monodromy/Floquet S/U",
                    "stability_commensurable": "true",
                    "conditioning_key": "m3",
                    "parser": "liao_mirror_text",
                    "parser_source": "A",
                    "ansatz": "liao_mirror",
                    "identity_key_strength": "preferred",
                    "independence_tier": "0",
                    "proposed_shards": "16",
                    "notes": "Tier 0 control; cannot be selected.",
                },
                {
                    "slug": "liao__supplementary_b_piano_trio__tier0_control",
                    "include": "true",
                    "source_path": PRIOR_B,
                    "access_date": "2026-05-30",
                    "source_family": "Li-Liao Numerical Tank",
                    "relationship_to_supp_a": "same-paper",
                    "relationship_to_supp_b": "self",
                    "orbit_type": "3D piano-trio periodic",
                    "dimensionality": "3D",
                    "body_count": "3",
                    "potential": "Newtonian",
                    "newtonian_substrate": "true",
                    "mass_parameters": "m1=m2=1,m3",
                    "initial_condition_fields": "z0,vx,vy,vz",
                    "period_field": "T",
                    "stability_label_definition": "linear monodromy/Floquet S/U",
                    "stability_commensurable": "true",
                    "conditioning_key": "m3",
                    "parser": "liao_mirror_text",
                    "parser_source": "B",
                    "ansatz": "liao_mirror",
                    "identity_key_strength": "preferred",
                    "independence_tier": "0",
                    "proposed_shards": "16",
                    "notes": "Tier 0 discovery table; cannot be selected.",
                },
            ]
        write_csv(path, rows, INVENTORY_FIELDS)
        print(f"[v13] wrote inventory template: {repo_rel(path)}")

    commands = out / "operator_commands.md"
    commands.write_text(
        "\n".join(
            [
                "# v0.13 Operator Commands",
                "",
                "Populate `target_inventory.csv` with signal-blind candidate sources, then run:",
                "",
                "```powershell",
                "python scripts/v13_external_target_search.py profile `",
                "  --inventory results/isotrophy/k-facet-v13-external-target-search/target_inventory.csv `",
                "  --out results/isotrophy/k-facet-v13-external-target-search",
                "",
                "python scripts/v13_external_target_search.py probe `",
                "  --inventory results/isotrophy/k-facet-v13-external-target-search/target_inventory.csv `",
                "  --probe-rows 100 `",
                "  --seed 20260523 `",
                "  --out results/isotrophy/k-facet-v13-external-target-search",
                "",
                "python scripts/v13_external_target_search.py select `",
                "  --out results/isotrophy/k-facet-v13-external-target-search",
                "```",
                "",
                "The probe command defaults to a 12-row rate probe unless `--authorize-full-probe`",
                "is supplied. This protects the repository's inline-run budget.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        out / "manifest.json",
        {
            "schema": "sundog.isotrophy.v0.13-external-target-search.v1",
            "mode": "v0.13-external-target-search",
            "form_lock": FORM_LOCK,
            "stage": "inventory-template",
            "verdict": None,
            "createdAt": now_iso(),
            "candidate_count": len(read_csv(path)),
        },
    )


def run_profile(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory(Path(args.inventory))
    prior_index = load_prior_index()
    profiles: list[dict[str, Any]] = []
    overlap_summaries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for candidate in inventory:
        slug = candidate["slug"]
        rows, text_hash, parse_error = parse_candidate_rows(candidate)
        local_hash = sha256_file(local_source_path(candidate)) if local_source_path(candidate) else None
        profile = profile_rows(rows) if rows else {
            "row_count": 0,
            "S_total": 0,
            "U_total": 0,
            "conditioning_strata_count": 0,
            "both_class_strata_count": 0,
            "strata": [],
        }

        exact = reflection = 0
        overlap_by_source: dict[str, int] = defaultdict(int)
        overlap_positions: list[int] = []
        identity_strength = clean(candidate.get("identity_key_strength")) or "preferred"
        if rows and identity_strength == "preferred":
            for pos, row in enumerate(rows):
                kind, source_name = overlap_kind(row, prior_index)
                if kind == "exact":
                    exact += 1
                    overlap_positions.append(pos)
                    overlap_by_source[f"{source_name}:exact"] += 1
                elif kind == "reflection":
                    reflection += 1
                    overlap_positions.append(pos)
                    overlap_by_source[f"{source_name}:reflection"] += 1
        overlap_total = exact + reflection
        overlap_fraction = overlap_total / max(profile["row_count"], 1)
        leakage_blocked = overlap_fraction > LEAKAGE_ABORT_FRACTION
        weak_identity = identity_strength != "preferred"
        failures = hard_gate_failures(candidate, rows, profile)
        if parse_error:
            failures.append(parse_error)
        if leakage_blocked:
            failures.append("overlap_fraction_above_0.05")
        if weak_identity:
            failures.append("identity_key_too_weak")
        hard_gate_pass = not failures
        projected_rows = max(profile["row_count"] - overlap_total, 0)

        row = {
            **candidate,
            "source_hash": local_hash or text_hash,
            "parse_error": parse_error,
            "row_count": profile["row_count"],
            "S_total": profile["S_total"],
            "U_total": profile["U_total"],
            "conditioning_strata_count": profile["conditioning_strata_count"],
            "both_class_strata_count": profile["both_class_strata_count"],
            "exact_overlap": exact,
            "reflection_overlap": reflection,
            "overlap_total": overlap_total,
            "overlap_fraction": round(overlap_fraction, 6),
            "leakage_blocked": leakage_blocked,
            "identity_weak_report_only": weak_identity,
            "projected_rows_after_overlap": projected_rows,
            "hard_gate_pass": hard_gate_pass,
            "hard_gate_failures": failures,
            "strata": profile["strata"],
            "overlap_positions": overlap_positions,
        }
        profiles.append(row)
        overlap_summaries.append(
            {
                "slug": slug,
                "identity_key_strength": identity_strength,
                "exact_overlap": exact,
                "reflection_overlap": reflection,
                "overlap_total": overlap_total,
                "overlap_fraction": round(overlap_fraction, 6),
                "overlap_by_source": json.dumps(dict(overlap_by_source), sort_keys=True),
                "leakage_blocked": leakage_blocked,
            }
        )
        if not hard_gate_pass:
            rejected.append(
                {
                    "slug": slug,
                    "reason": ";".join(failures),
                    "row_count": profile["row_count"],
                    "independence_tier": tier_value(candidate.get("independence_tier")),
                }
            )

    write_json(profile_path(out), profiles)
    write_csv(out / "overlap_audits.csv", overlap_summaries, [
        "slug", "identity_key_strength", "exact_overlap", "reflection_overlap",
        "overlap_total", "overlap_fraction", "overlap_by_source", "leakage_blocked",
    ])
    write_csv(out / "rejected_targets.csv", rejected, [
        "slug", "reason", "row_count", "independence_tier",
    ])
    write_json(
        out / "manifest.json",
        {
            "schema": "sundog.isotrophy.v0.13-external-target-search.v1",
            "mode": "v0.13-external-target-search",
            "form_lock": FORM_LOCK,
            "stage": "profile",
            "createdAt": now_iso(),
            "candidate_count": len(inventory),
            "eligible_candidate_count": sum(1 for p in profiles if p["hard_gate_pass"]),
            "viable_candidate_count": None,
            "verdict": None,
        },
    )
    print(f"[v13] profiled {len(inventory)} candidates; eligible={sum(1 for p in profiles if p['hard_gate_pass'])}")


def sample_positions(n: int, k: int, seed: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=min(k, n), replace=False))


def load_profiles_by_slug(out: Path) -> dict[str, dict[str, Any]]:
    profiles = load_json(profile_path(out), [])
    return {p["slug"]: p for p in profiles}


def candidate_rows_after_overlap(candidate: dict[str, str], profile: dict[str, Any]) -> list[OrbitRow]:
    rows, _, parse_error = parse_candidate_rows(candidate)
    if parse_error:
        raise RuntimeError(f"{candidate['slug']}: {parse_error}")
    excluded = set(int(i) for i in profile.get("overlap_positions", []))
    return [row for i, row in enumerate(rows) if i not in excluded]


def probe_one_candidate(
    candidate: dict[str, str],
    profile: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = candidate_rows_after_overlap(candidate, profile)
    requested = min(args.probe_rows, len(rows))
    authorized_rows = requested if args.authorize_full_probe else min(args.rate_probe_rows, requested)
    positions_to_run = sample_positions(len(rows), authorized_rows, args.seed)

    records: list[dict[str, Any]] = []
    blocked = sanity = success = 0
    started = time.perf_counter()
    stopped_for_budget = False

    for sample_i, pos in enumerate(positions_to_run, start=1):
        row = rows[pos]
        rec = per_row_pipeline(row, canonical_omega_18(row.masses))
        is_blocked = bool(rec.get("integration_blocked"))
        is_sanity = (
            not is_blocked
            and (rec.get("symplecticity_status") == "fail" or rec.get("reciprocal_pair_status") == "fail")
        )
        ok = (
            not is_blocked
            and rec.get("symplecticity_status") == "pass"
            and rec.get("reciprocal_pair_status") == "pass"
        )
        blocked += int(is_blocked)
        sanity += int(is_sanity)
        success += int(ok)
        status = "success" if ok else ("integration_blocked" if is_blocked else "sanity_failed")
        records.append(
            {
                "slug": candidate["slug"],
                "sample_index": sample_i,
                "candidate_row_position_after_overlap": pos,
                "orbit_index": row.index,
                "mass_key": m3_key(row.m3),
                "period": round(float(row.period), 8),
                "status": status,
                "runtime_seconds": round(float(rec.get("total_seconds") or 0.0), 6),
                "failure_stage": rec.get("integration_error_stage"),
                "failure_reason": rec.get("integration_error_message"),
                "symplecticity_status": rec.get("symplecticity_status"),
                "reciprocal_pair_status": rec.get("reciprocal_pair_status"),
            }
        )
        elapsed = time.perf_counter() - started
        print(
            f"[v13-probe] {candidate['slug']} {sample_i}/{authorized_rows} "
            f"m={m3_key(row.m3)} T={row.period:.2f} status={status}",
            flush=True,
        )
        if args.respect_inline_budget and elapsed >= args.max_inline_seconds and sample_i < requested:
            stopped_for_budget = True
            break

    rows_run = len(records)
    attrited = blocked + sanity
    attr = (attrited / rows_run) if rows_run else None
    lo, hi = wilson_ci(attrited, rows_run)
    seconds_per_row = (sum(float(r["runtime_seconds"]) for r in records) / rows_run) if rows_run else None
    projected_single = (seconds_per_row * len(rows) / 3600.0) if seconds_per_row is not None else None
    shards = max(1, int(float(clean(candidate.get("proposed_shards")) or "1")))
    projected_sharded = (projected_single / shards) if projected_single is not None else None
    complete_probe = rows_run >= requested and requested >= min(FULL_PROBE_ROWS, len(rows))
    needs_operator = not complete_probe
    d5_viable = (
        complete_probe
        and attr is not None
        and attr <= ATTRITION_CLEAN_MAX
        and hi <= ATTRITION_WILSON_HIGH_MAX
        and projected_single is not None
        and projected_single <= SINGLE_THREAD_HOURS_MAX
        and projected_sharded is not None
        and projected_sharded <= SHARDED_HOURS_MAX
    )
    summary = {
        "slug": candidate["slug"],
        "probe_rows_requested": requested,
        "probe_rows_run": rows_run,
        "authorized_full_probe": args.authorize_full_probe,
        "stopped_for_budget": stopped_for_budget,
        "operator_probe_needed": needs_operator,
        "success": success,
        "blocked": blocked,
        "sanity_fail": sanity,
        "attrited": attrited,
        "attrition_fraction": round(attr, 6) if attr is not None else None,
        "wilson95_low": round(lo, 6),
        "wilson95_high": round(hi, 6),
        "seconds_per_row": round(seconds_per_row, 6) if seconds_per_row is not None else None,
        "candidate_rows_after_overlap": len(rows),
        "projected_full_transfer_hours_single": round(projected_single, 4) if projected_single is not None else None,
        "proposed_shards": shards,
        "projected_full_transfer_hours_sharded": round(projected_sharded, 4) if projected_sharded is not None else None,
        "d5_viable": d5_viable,
    }
    return records, summary


def run_probe(args: argparse.Namespace) -> None:
    if args.rtol != RTOL or args.atol != ATOL:
        raise SystemExit(
            f"v0.13 D5 probe is frozen at rtol=atol={RTOL}; "
            f"got rtol={args.rtol}, atol={args.atol}"
        )
    out = Path(args.out)
    inventory = load_inventory(Path(args.inventory))
    by_slug = {c["slug"]: c for c in inventory}
    profiles = load_profiles_by_slug(out)
    if not profiles:
        raise SystemExit("No source_profiles.json found. Run profile first.")

    slugs = [args.candidate_slug] if args.candidate_slug else sorted(profiles)
    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for slug in slugs:
        profile = profiles.get(slug)
        candidate = by_slug.get(slug)
        if not profile or not candidate:
            print(f"[v13-probe] skip missing {slug}")
            continue
        if not profile.get("hard_gate_pass"):
            print(f"[v13-probe] skip ineligible {slug}: {profile.get('hard_gate_failures')}")
            continue
        records, summary = probe_one_candidate(candidate, profile, args)
        all_records.extend(records)
        summaries.append(summary)

    write_csv(out / "feasibility_probes.csv", all_records, [
        "slug",
        "sample_index",
        "candidate_row_position_after_overlap",
        "orbit_index",
        "mass_key",
        "period",
        "status",
        "runtime_seconds",
        "failure_stage",
        "failure_reason",
        "symplecticity_status",
        "reciprocal_pair_status",
    ])
    write_json(feasibility_summary_path(out), summaries)
    operator_needed = [s for s in summaries if s.get("operator_probe_needed")]
    operator_text = [
        "# v0.13 Operator Probe Commands",
        "",
        "The following commands authorize the full 100-row feasibility probe.",
        "Run only after accepting the wall-clock cost implied by the rate probe.",
        "",
    ]
    for s in operator_needed:
        operator_text.extend(
            [
                "```powershell",
                "python scripts/v13_external_target_search.py probe `",
                f"  --candidate-slug {s['slug']} `",
                f"  --probe-rows {args.probe_rows} `",
                f"  --seed {args.seed} `",
                "  --authorize-full-probe `",
                f"  --out {repo_rel(out)}",
                "```",
                "",
            ]
        )
    (out / "operator_commands.md").write_text("\n".join(operator_text), encoding="utf-8")
    print(f"[v13] wrote feasibility probes for {len(summaries)} candidates")


def run_select(args: argparse.Namespace) -> None:
    out = Path(args.out)
    profiles = load_json(profile_path(out), [])
    summaries = load_json(feasibility_summary_path(out), [])
    summary_by_slug = {s["slug"]: s for s in summaries}

    pending = []
    viable = []
    for profile in profiles:
        if not profile.get("hard_gate_pass"):
            continue
        summary = summary_by_slug.get(profile["slug"])
        if not summary or summary.get("operator_probe_needed"):
            pending.append(profile["slug"])
            continue
        if summary.get("d5_viable"):
            viable.append((profile, summary))

    def rank(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
        profile, summary = item
        source_stability = 0 if profile.get("source_hash") else 1
        return (
            -tier_value(profile.get("independence_tier")),
            float(summary.get("wilson95_high") or 1.0),
            -int(profile.get("projected_rows_after_overlap") or 0),
            source_stability,
            float(summary.get("projected_full_transfer_hours_single") or 1e99),
            profile["slug"],
        )

    viable.sort(key=rank)
    selected = viable[0] if viable else None
    if selected:
        selected_profile, selected_summary = selected
        verdict = "external_target_locked" if tier_value(selected_profile.get("independence_tier")) >= 2 else "near_external_target_locked"
    elif pending:
        selected_profile = selected_summary = None
        verdict = "target_search_operator_probe_needed"
    elif not profiles:
        selected_profile = selected_summary = None
        verdict = "target_search_blocked_by_access"
    else:
        selected_profile = selected_summary = None
        verdict = "no_viable_external_target_found"

    readback = [
        "# v0.13 Selection Readback",
        "",
        f"verdict: `{verdict}`",
        "",
        f"candidate_count: {len(profiles)}",
        f"eligible_candidate_count: {sum(1 for p in profiles if p.get('hard_gate_pass'))}",
        f"viable_candidate_count: {len(viable)}",
    ]
    if pending:
        readback.append(f"operator_probe_pending: {', '.join(pending)}")
    if selected:
        readback.extend(
            [
                "",
                "## Selected Target",
                "",
                f"slug: `{selected_profile['slug']}`",
                f"independence_tier: {tier_value(selected_profile.get('independence_tier'))}",
                f"attrition_fraction: {selected_summary.get('attrition_fraction')}",
                f"wilson95_high: {selected_summary.get('wilson95_high')}",
                f"projected_rows_after_overlap: {selected_profile.get('projected_rows_after_overlap')}",
                f"projected_full_transfer_hours_single: {selected_summary.get('projected_full_transfer_hours_single')}",
            ]
        )
    readback.extend(
        [
            "",
            "This readback is source-selection only. It does not test or confirm the",
            "velocity-fraction stability signal.",
            "",
        ]
    )
    (out / "selection_readback.md").write_text("\n".join(readback), encoding="utf-8")
    write_json(
        out / "manifest.json",
        {
            "schema": "sundog.isotrophy.v0.13-external-target-search.v1",
            "mode": "v0.13-external-target-search",
            "form_lock": FORM_LOCK,
            "stage": "select",
            "verdict": verdict,
            "completedAt": now_iso(),
            "candidate_count": len(profiles),
            "eligible_candidate_count": sum(1 for p in profiles if p.get("hard_gate_pass")),
            "viable_candidate_count": len(viable),
            "selected_slug": selected_profile["slug"] if selected else None,
            "selected_independence_tier": tier_value(selected_profile.get("independence_tier")) if selected else None,
            "selected_attrition_fraction": selected_summary.get("attrition_fraction") if selected else None,
            "selected_wilson95_high": selected_summary.get("wilson95_high") if selected else None,
            "selected_projected_primary_rows": selected_profile.get("projected_rows_after_overlap") if selected else None,
            "selected_projected_runtime_hours": selected_summary.get("projected_full_transfer_hours_single") if selected else None,
        },
    )
    print(f"[v13] verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v0.13 external target search harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inv = sub.add_parser("inventory", help="create a signal-blind inventory template")
    p_inv.add_argument("--out", default=DEFAULT_OUT)
    p_inv.add_argument("--force", action="store_true")
    p_inv.add_argument("--seed-local-controls", action="store_true")
    p_inv.set_defaults(func=run_inventory)

    p_prof = sub.add_parser("profile", help="profile candidates and hard gates")
    p_prof.add_argument("--inventory", default=f"{DEFAULT_OUT}/target_inventory.csv")
    p_prof.add_argument("--out", default=DEFAULT_OUT)
    p_prof.set_defaults(func=run_profile)

    p_probe = sub.add_parser("probe", help="run D5 feasibility probes without feature receipts")
    p_probe.add_argument("--inventory", default=f"{DEFAULT_OUT}/target_inventory.csv")
    p_probe.add_argument("--out", default=DEFAULT_OUT)
    p_probe.add_argument("--candidate-slug")
    p_probe.add_argument("--probe-rows", type=int, default=FULL_PROBE_ROWS)
    p_probe.add_argument("--rate-probe-rows", type=int, default=12)
    p_probe.add_argument("--seed", type=int, default=SEED)
    p_probe.add_argument("--rtol", type=float, default=RTOL)
    p_probe.add_argument("--atol", type=float, default=ATOL)
    p_probe.add_argument("--authorize-full-probe", action="store_true")
    p_probe.add_argument("--respect-inline-budget", action="store_true", default=True)
    p_probe.add_argument("--max-inline-seconds", type=float, default=INLINE_BUDGET_SECONDS)
    p_probe.set_defaults(func=run_probe)

    p_sel = sub.add_parser("select", help="select target from profiles + feasibility probes")
    p_sel.add_argument("--out", default=DEFAULT_OUT)
    p_sel.set_defaults(func=run_select)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
