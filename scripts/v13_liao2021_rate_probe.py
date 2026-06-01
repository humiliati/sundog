#!/usr/bin/env python3
"""v0.13 D5 feasibility rate probe on liao2021 (resume of v0.13 proper).

Prerequisites now PASSED: v0.13a leakage bound 0.0 (bounded) and v0.13b frame-zone
stability (coarse_zone_rule_frame_stable_enough_to_test). This runs the v0.13 feasibility
probe -- the SAME firewall as v0.13 -- on liao2021 via the v0.13a expansion-only adapter:
integrate + monodromy + the 1e-4 gates only. It records success / integration_blocked /
sanity_failed + runtime per row. NO velocity_fraction, NO zone, NO stability label.

Default is a 12-row rate probe (gauge + stage); the full 100-row feasibility probe needs
--authorize-full-probe, per the v0.13 long-run discipline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isotrophy_workbench import canonical_omega_18  # type: ignore
from scripts.v07a_velocity_fraction_audit import (  # type: ignore
    compute_monodromy_vectorized,
    reciprocal_pair_residual,
    symplecticity_residual,
)
from scripts.v13a_liao2021_preflight import (  # type: ignore
    ATOL,
    MAX_STEP_FRACTION,
    RECIPROCAL_PAIR_GATE,
    RTOL,
    STAGED,
    SYMPLECTICITY_GATE,
    expand_liao2021_state,
    integrate_liao2021_state,
    parse_liao2021,
)

SEED = 20260523
RATE_PROBE_ROWS = 12
FULL_PROBE_ROWS = 100
ATTRITION_CLEAN_MAX = 0.10
ATTRITION_WILSON_HIGH_MAX = 0.20
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v13-liao2021-rate-probe"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def feasibility_record(masses, x0, v0, period):
    """Integrate + monodromy + gates ONLY -> feasibility status + runtime. No vf."""
    t0 = time.perf_counter()
    try:
        integrated = integrate_liao2021_state(masses, x0, v0, period)
        M_i = compute_monodromy_vectorized(integrated, RTOL, ATOL, MAX_STEP_FRACTION)
    except Exception as exc:  # noqa: BLE001
        return {"status": "integration_blocked", "seconds": round(time.perf_counter() - t0, 4),
                "failure": type(exc).__name__}
    omega = canonical_omega_18(masses)
    symp = symplecticity_residual(M_i, omega)
    recip = reciprocal_pair_residual(np.linalg.eigvals(M_i))
    secs = round(time.perf_counter() - t0, 4)
    ok = symp <= SYMPLECTICITY_GATE and recip <= RECIPROCAL_PAIR_GATE
    return {"status": ("success" if ok else "sanity_failed"), "seconds": secs,
            "symplecticity_residual": symp, "reciprocal_pair_residual": recip,
            "failure": (None if ok else "gate")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=str(STAGED))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--rate-probe-rows", type=int, default=RATE_PROBE_ROWS, dest="rate_probe_rows")
    ap.add_argument("--authorize-full-probe", action="store_true", dest="authorize_full")
    ap.add_argument("--rtol", type=float, default=RTOL)
    ap.add_argument("--atol", type=float, default=ATOL)
    args = ap.parse_args()
    if args.rtol != RTOL or args.atol != ATOL:
        raise SystemExit(f"v0.13 D5 probe is frozen at rtol=atol={RTOL}; got {args.rtol}/{args.atol}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = parse_liao2021(Path(args.table))  # v0.13a leakage 0.0 -> all rows are candidates
    n_target = FULL_PROBE_ROWS if args.authorize_full else args.rate_probe_rows
    n = min(n_target, len(rows))
    rng = np.random.default_rng(SEED)
    pick = sorted(int(i) for i in rng.choice(len(rows), size=n, replace=False))

    records = []
    success = blocked = sanity = 0
    t0 = time.perf_counter()
    for k, idx in enumerate(pick, start=1):
        row = rows[idx]
        masses, x0, v0 = expand_liao2021_state(row, center_com=True)
        rec = feasibility_record(masses, x0, v0, row.period)
        st = rec["status"]
        blocked += int(st == "integration_blocked")
        sanity += int(st == "sanity_failed")
        success += int(st == "success")
        records.append({"orbit_index": row.index, "m1": row.m1, "m2": row.m2, "m3": row.m3,
                        "period": round(row.period, 4), "status": st,
                        "runtime_seconds": rec["seconds"], "failure": rec.get("failure")})
        print(f"[v13-rate] {k}/{n} idx={row.index} m=({row.m1},{row.m2},{row.m3}) "
              f"T={row.period:.2f} status={st} t={rec['seconds']:.1f}s", flush=True)

    wall = time.perf_counter() - t0
    attrited = blocked + sanity
    attr = attrited / n if n else None
    lo, hi = wilson_ci(attrited, n)
    sec_per_row = sum(r["runtime_seconds"] for r in records) / n if n else None
    is_full = n >= FULL_PROBE_ROWS
    summary = {
        "schema": "sundog.isotrophy.v0.13-liao2021-rate-probe.v1",
        "prerequisites": {"v0.13a_leakage_bound": 0.0, "v0.13b_frame_zone_verdict":
                          "coarse_zone_rule_frame_stable_enough_to_test"},
        "seed": SEED, "probe_rows": n, "is_full_probe": is_full,
        "authorized_full_probe": args.authorize_full, "candidate_rows": len(rows),
        "success": success, "integration_blocked": blocked, "sanity_failed": sanity,
        "attrition_fraction": (round(attr, 4) if attr is not None else None),
        "attrition_wilson95_low": round(lo, 4), "attrition_wilson95_high": round(hi, 4),
        "seconds_per_row": (round(sec_per_row, 4) if sec_per_row else None),
        "probe_wall_seconds": round(wall, 2),
        "feasibility_bars": {"attrition_clean_max": ATTRITION_CLEAN_MAX,
                             "attrition_wilson_high_max": ATTRITION_WILSON_HIGH_MAX},
        "projected_full_100row_probe_minutes": (round(sec_per_row * FULL_PROBE_ROWS / 60.0, 1)
                                                if sec_per_row else None),
        "projected_full_catalog_hours_single": (round(sec_per_row * len(rows) / 3600.0, 1)
                                                 if sec_per_row else None),
        "note": ("12-row rate probe is a gauge; it cannot lock the target. Stage the "
                 "100-row feasibility probe (--authorize-full-probe) before any attrition "
                 "verdict; the 135,445-row transfer will need sharding (Phase-15 pattern)."),
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    with (out / "rate_probe.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["orbit_index", "m1", "m2", "m3", "period",
                                           "status", "runtime_seconds", "failure"])
        w.writeheader(); w.writerows(records)

    print()
    print(f"[v13-rate] {'FULL 100-row' if is_full else 'rate'} probe: {n} rows in {wall:.0f}s")
    print(f"[v13-rate] success={success} blocked={blocked} sanity={sanity} -> "
          f"attrition={summary['attrition_fraction']} "
          f"Wilson95=[{summary['attrition_wilson95_low']}, {summary['attrition_wilson95_high']}]")
    print(f"[v13-rate] seconds/row={summary['seconds_per_row']}; projected 100-row probe "
          f"~{summary['projected_full_100row_probe_minutes']} min; full 135k-row catalog "
          f"~{summary['projected_full_catalog_hours_single']} h single-threaded")
    if not is_full:
        print("[v13-rate] RATE PROBE ONLY (cannot lock). Re-run with --authorize-full-probe "
              "for the 100-row feasibility verdict.")


if __name__ == "__main__":
    main()
