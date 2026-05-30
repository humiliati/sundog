#!/usr/bin/env python3
"""v0.12 external frozen transfer -- profile + smoke stages (form steps 1-4).

Locked by docs/isotrophy/kfacet/kfacet_v12_external_frozen_transfer_form.md.

Asks whether the FROZEN v0.11 velocity-fraction zone rule transfers from the
supplementary-B piano-trio table (where it was discovered/scored) to the distinct
supplementary-A 3D periodic-orbit table. This file implements ONLY the cheap
profile stage and the bounded 30-row integration smoke; it never runs the full
10,059-row velocity-fraction pass (that is staged for an operator / long-budget
runner per the form's step 6-7).

The velocity-fraction measurement is REUSED verbatim from the v0.7a D5 receipt
(`scripts/v07a_velocity_fraction_audit.per_row_pipeline`) so the feature carries
the identical monodromy / gauge / gate conventions -- no reimplementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isotrophy_workbench import (  # type: ignore
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

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v12_external_frozen_transfer_form.md"
EXPECTED_ROWS = 10059
ZONE_CUTPOINTS = (0.25, 0.50)
OVERLAP_TOL = 1e-9
PERM_SEED = 20260523
PERMUTATIONS = 100000
LEAKAGE_ABORT_FRACTION = 0.05
EFFECT_FLOOR = 0.55
# Primary-stratum gates (full-run; surfaced in the profile candidate count).
PRIMARY_N, PRIMARY_S, PRIMARY_U = 30, 5, 5
DEFAULT_TARGET = "docs/isotrophy/supplementary-A_periodic-3d_mirror.txt"
DEFAULT_DISCOVERY = "docs/isotrophy/supplementary-B_piano-init-condit-3d.txt"
DEFAULT_OUT = "results/isotrophy/k-facet-v12-external-frozen-transfer"


def m3_key(m3: float) -> str:
    return f"{m3:.4f}".rstrip("0").rstrip(".")


def zone_index(vf: float) -> int:
    if vf < ZONE_CUTPOINTS[0]:
        return 0
    if vf < ZONE_CUTPOINTS[1]:
        return 1
    return 2


def is_strict_m3_1(row) -> bool:
    return abs(row.m3 - 1.0) <= OVERLAP_TOL


def build_overlap_index(discovery_rows):
    """m3-binned discovery rows for fast exact + reflection matching."""
    idx = defaultdict(list)
    for d in discovery_rows:
        idx[round(d.m3, 9)].append(d)
    return idx


def overlap_status(row, disc_index, tol=OVERLAP_TOL):
    """Return 'exact', 'reflection', or None.

    Shared ansatz, both supplements canonicalize z0>0, so a shared orbit lands at
    identical ICs (exact). The reflection image (z0,vx,vy,vz)->(-z0,vx,vy,-vz) at
    the same (m3,T) guards against the two supplements picking opposite vz-sign
    representatives of one orbit.
    """
    for d in disc_index.get(round(row.m3, 9), ()):  # same m3 bucket
        if abs(row.m3 - d.m3) > tol or abs(row.period - d.period) > tol:
            continue
        if (abs(row.z0 - d.z0) <= tol and abs(row.vx - d.vx) <= tol
                and abs(row.vy - d.vy) <= tol and abs(row.vz - d.vz) <= tol):
            return "exact"
        if (abs(row.z0 + d.z0) <= tol and abs(row.vx - d.vx) <= tol
                and abs(row.vy - d.vy) <= tol and abs(row.vz + d.vz) <= tol):
            return "reflection"
    return None


def quantile_stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(out: Path) -> dict:
    p = out / "manifest.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_manifest(out: Path, manifest: dict):
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n",
                                       encoding="utf-8")


# --------------------------------------------------------------------------- #
# Shared parse + quarantine                                                    #
# --------------------------------------------------------------------------- #

def parse_and_quarantine(target_path: str, discovery_path: str):
    target = parse_rows(read_text(target_path), source="A")
    discovery = parse_rows(read_text(discovery_path), source="B")
    disc_index = build_overlap_index(discovery)

    exact_n = reflection_n = strict_m3_1_n = 0
    candidate = []          # non-overlap, non-strict-m3=1 -> would enter full vf pass
    overlap_rows = []
    strict_rows = []
    for row in target:
        if is_strict_m3_1(row):
            strict_m3_1_n += 1
            strict_rows.append(row)
            continue
        status = overlap_status(row, disc_index)
        if status == "exact":
            exact_n += 1
            overlap_rows.append((row, status))
        elif status == "reflection":
            reflection_n += 1
            overlap_rows.append((row, status))
        else:
            candidate.append(row)

    return {
        "target": target,
        "discovery": discovery,
        "candidate": candidate,
        "overlap_rows": overlap_rows,
        "strict_rows": strict_rows,
        "exact_overlap": exact_n,
        "reflection_overlap": reflection_n,
        "strict_m3_1": strict_m3_1_n,
    }


# --------------------------------------------------------------------------- #
# Stage: profile                                                              #
# --------------------------------------------------------------------------- #

def run_profile(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target = parse_rows(read_text(args.target), source="A")
    # Source-profile gate.
    abort_reason = None
    if len(target) != EXPECTED_ROWS:
        abort_reason = f"row_count {len(target)} != {EXPECTED_ROWS}"
    bad_labels = sorted({r.stability for r in target if r.stability not in ("S", "U")})
    if bad_labels:
        abort_reason = f"stability labels outside {{S,U}}: {bad_labels}"

    q = parse_and_quarantine(args.target, args.discovery)
    overlap_total = q["exact_overlap"] + q["reflection_overlap"]
    nonoverlap_candidate = len(target) - q["strict_m3_1"] - overlap_total
    overlap_fraction = (overlap_total / max(nonoverlap_candidate + overlap_total, 1))

    # Per-m3 strata on raw catalog counts (no integration needed).
    by_m3 = defaultdict(list)
    for r in target:
        by_m3[round(r.m3, 9)].append(r)
    strata = []
    candidate_primary = 0
    for m3 in sorted(by_m3):
        rows = by_m3[m3]
        s = sum(1 for r in rows if r.stability == "S")
        u = len(rows) - s
        pstats = quantile_stats([r.period for r in rows])
        is_candidate_primary = (len(rows) >= PRIMARY_N and s >= PRIMARY_S and u >= PRIMARY_U
                                and abs(m3 - 1.0) > OVERLAP_TOL)
        if is_candidate_primary:
            candidate_primary += 1
        strata.append({
            "m3": m3_key(m3), "N": len(rows), "S": s, "U": u,
            "S_fraction": round(s / len(rows), 4),
            "period_min": round(pstats["min"], 4),
            "period_median": round(pstats["median"], 4),
            "period_max": round(pstats["max"], 4),
            "candidate_primary_pre_vf": is_candidate_primary,
        })

    write_csv(out / "source_profile.csv", strata,
              ["m3", "N", "S", "U", "S_fraction", "period_min", "period_median",
               "period_max", "candidate_primary_pre_vf"])
    write_csv(out / "overlap_audit.csv",
              [{"m3": m3_key(r.m3), "index": r.index, "z0": r.z0, "vx": r.vx,
                "vy": r.vy, "vz": r.vz, "T": r.period, "stability": r.stability,
                "overlap_kind": kind} for (r, kind) in q["overlap_rows"]],
              ["m3", "index", "z0", "vx", "vy", "vz", "T", "stability", "overlap_kind"])

    manifest = load_manifest(out)
    manifest.update({
        "schema": "sundog.isotrophy.v0.12-external-frozen-transfer.v1",
        "mode": "v0.12-external-frozen-transfer",
        "form_lock": FORM_LOCK,
        "target_path": args.target,
        "discovery_path": args.discovery,
        "zone_cutpoints": list(ZONE_CUTPOINTS),
        "seed": PERM_SEED,
        "permutations": PERMUTATIONS,
        "effect_floor": EFFECT_FLOOR,
        "rtol": RTOL, "atol": ATOL, "max_step_fraction": MAX_STEP_FRACTION,
        "symplecticity_gate": SYMPLECTICITY_GATE, "reciprocal_pair_gate": RECIPROCAL_PAIR_GATE,
        "profile": {
            "row_count": len(target),
            "expected_rows": EXPECTED_ROWS,
            "source_gate_abort_reason": abort_reason,
            "S_total": sum(1 for r in target if r.stability == "S"),
            "U_total": sum(1 for r in target if r.stability == "U"),
            "m3_strata_count": len(strata),
            "m3_min": m3_key(min(by_m3)), "m3_max": m3_key(max(by_m3)),
            "exact_overlap": q["exact_overlap"],
            "reflection_overlap": q["reflection_overlap"],
            "overlap_total": overlap_total,
            "overlap_fraction_of_candidate": round(overlap_fraction, 6),
            "overlap_leakage_would_abort": overlap_fraction > LEAKAGE_ABORT_FRACTION,
            "strict_m3_1_excluded": q["strict_m3_1"],
            "strict_m3_1_exclusion_mode": "all_m3_eq_1_conservative",
            "nonoverlap_candidate_rows": nonoverlap_candidate,
            "candidate_primary_strata_pre_overlap_attrition": candidate_primary,
            "quarantine_sources_note": (
                "Matched against the supp-B catalog file (authoritative ICs for all "
                "supp-B-derived evidence: v05/v07/v09/v11). The listed result-dir "
                "sources are supp-B-derived and subsumed by this match; strict m3=1 "
                "sigma3 rows are excluded conservatively as the whole m3=1 slice."
            ),
        },
    })
    save_manifest(out, manifest)

    print(f"[v12-profile] rows={len(target)} (expect {EXPECTED_ROWS}; "
          f"gate_abort={abort_reason})")
    print(f"[v12-profile] m3 strata={len(strata)} range [{m3_key(min(by_m3))},"
          f"{m3_key(max(by_m3))}]  S={manifest['profile']['S_total']} "
          f"U={manifest['profile']['U_total']}")
    print(f"[v12-profile] overlap: exact={q['exact_overlap']} reflection="
          f"{q['reflection_overlap']} total={overlap_total} "
          f"frac={overlap_fraction:.4%} (abort>{LEAKAGE_ABORT_FRACTION:.0%}: "
          f"{overlap_fraction > LEAKAGE_ABORT_FRACTION})")
    print(f"[v12-profile] strict m3=1 excluded (conservative)={q['strict_m3_1']}")
    print(f"[v12-profile] nonoverlap candidate rows={nonoverlap_candidate}  "
          f"candidate primary strata (pre-vf, N>=30/S>=5/U>=5)={candidate_primary}")
    if abort_reason:
        print(f"[v12-profile] WARNING source gate would ABORT the full run: {abort_reason}")


# --------------------------------------------------------------------------- #
# Stage: smoke                                                                #
# --------------------------------------------------------------------------- #

def select_smoke_rows(candidate, n_bins=6, per_bin=5, cap=30):
    """Deterministic: <=n_bins m3 bins spanning the range, period quantiles each."""
    m3_vals = sorted({round(r.m3, 9) for r in candidate})
    if not m3_vals:
        return []
    positions = np.linspace(0, len(m3_vals) - 1, min(n_bins, len(m3_vals)))
    sel_m3 = sorted({m3_vals[int(round(p))] for p in positions})
    qpos = [0.0, 0.25, 0.5, 0.75, 1.0][:per_bin]
    selected = []
    for m in sel_m3:
        bin_rows = sorted([r for r in candidate if round(r.m3, 9) == m],
                          key=lambda r: r.period)
        n = len(bin_rows)
        idxs = sorted({int(round(qp * (n - 1))) for qp in qpos})
        for j in idxs:
            selected.append(bin_rows[j])
    return selected[:cap]


def run_smoke(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q = parse_and_quarantine(args.target, args.discovery)
    candidate = q["candidate"]
    smoke_rows = select_smoke_rows(candidate, cap=args.smoke_rows)

    records = []
    success = sanity_fail = blocked = 0
    t0 = time.perf_counter()
    for i, row in enumerate(smoke_rows, start=1):
        omega = canonical_omega_18(row.masses)
        rec = per_row_pipeline(row, omega)
        ok = (not rec.get("integration_blocked")
              and rec.get("symplecticity_status") == "pass"
              and rec.get("reciprocal_pair_status") == "pass")
        if rec.get("integration_blocked"):
            blocked += 1
        elif rec.get("symplecticity_status") == "fail" or rec.get("reciprocal_pair_status") == "fail":
            sanity_fail += 1
        if ok:
            success += 1
        vf = rec.get("velocity_fraction")
        records.append({
            "m3": m3_key(row.m3), "index": row.index, "period": round(row.period, 4),
            "stability": row.stability,
            "total_seconds": round(rec.get("total_seconds") or 0.0, 4),
            "integration_blocked": rec.get("integration_blocked"),
            "symplecticity_status": rec.get("symplecticity_status"),
            "reciprocal_pair_status": rec.get("reciprocal_pair_status"),
            "velocity_fraction": vf,
            "zone_index": (zone_index(vf) if vf is not None else None),
        })
        print(f"[v12-smoke] {i:2d}/{len(smoke_rows)} {row.label:16s} "
              f"T={row.period:7.2f} stab={row.stability} "
              f"t={records[-1]['total_seconds']:7.2f}s ok={ok} "
              f"vf={vf if vf is None else round(vf, 4)}", flush=True)

    n = len(smoke_rows)
    wall = time.perf_counter() - t0
    sec_per_row = (sum(r["total_seconds"] for r in records) / n) if n else None
    success_frac = (success / n) if n else None
    sanity_frac = (sanity_fail / n) if n else None
    nonoverlap_candidate = len(candidate)
    proj_wall = (sec_per_row * nonoverlap_candidate) if sec_per_row else None
    proj_analyzable = (round(nonoverlap_candidate * success_frac) if success_frac is not None else None)

    write_csv(out / "integration_smoke.csv", records,
              ["m3", "index", "period", "stability", "total_seconds",
               "integration_blocked", "symplecticity_status", "reciprocal_pair_status",
               "velocity_fraction", "zone_index"])

    smoke = {
        "smoke_rows_requested": args.smoke_rows,
        "smoke_rows_run": n,
        "smoke_wall_seconds": round(wall, 2),
        "seconds_per_row": (round(sec_per_row, 4) if sec_per_row else None),
        "integration_success_count": success,
        "integration_success_fraction": (round(success_frac, 4) if success_frac is not None else None),
        "integration_blocked_count": blocked,
        "sanity_fail_count": sanity_fail,
        "sanity_fail_fraction": (round(sanity_frac, 4) if sanity_frac is not None else None),
        "nonoverlap_candidate_rows": nonoverlap_candidate,
        "projected_full_wall_clock_seconds": (round(proj_wall, 1) if proj_wall else None),
        "projected_full_wall_clock_hours": (round(proj_wall / 3600.0, 2) if proj_wall else None),
        "projected_analyzable_rows": proj_analyzable,
        "full_run_exceeds_inline_budget": (proj_wall is not None and proj_wall > 600),
    }
    manifest = load_manifest(out)
    manifest["smoke"] = smoke
    save_manifest(out, manifest)

    print()
    print(f"[v12-smoke] ran {n} rows in {wall:.1f}s  "
          f"seconds/row={smoke['seconds_per_row']}")
    print(f"[v12-smoke] integration success={success}/{n} "
          f"({smoke['integration_success_fraction']}); blocked={blocked}; "
          f"sanity_fail={sanity_fail}")
    print(f"[v12-smoke] nonoverlap candidate rows={nonoverlap_candidate}")
    print(f"[v12-smoke] PROJECTED full wall-clock="
          f"{smoke['projected_full_wall_clock_hours']} h  "
          f"projected analyzable rows={proj_analyzable}")
    print(f"[v12-smoke] full run exceeds inline budget (>10 min): "
          f"{smoke['full_run_exceeds_inline_budget']}  -> STAGE for operator")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("profile", "smoke"):
        sp = sub.add_parser(name)
        sp.add_argument("--target", default=DEFAULT_TARGET)
        sp.add_argument("--discovery", default=DEFAULT_DISCOVERY)
        sp.add_argument("--out", default=DEFAULT_OUT)
        if name == "smoke":
            sp.add_argument("--smoke-rows", type=int, default=30, dest="smoke_rows")
            sp.add_argument("--rtol", type=float, default=RTOL)  # accepted; locked value used
            sp.add_argument("--atol", type=float, default=ATOL)
    args = ap.parse_args()
    if args.stage == "profile":
        run_profile(args)
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()
