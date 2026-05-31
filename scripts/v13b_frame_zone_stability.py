#!/usr/bin/env python3
"""v0.13b frame-zone stability audit.

Locked by docs/isotrophy/kfacet/kfacet_v13b_frame_zone_stability_audit_form.md.

Measures the EXACT frame-zone-change fraction: under the locked v0.13a frame
perturbations, does the frozen v0.11 zone_index(vf) change? Records per row/frame
base_zone, rotated_zone, zone_changed, and the base distance to the nearest cutpoint.
NO S/U conditioning, NO AUC, NO target selection, NO raw vf recorded -- a frame-
stability audit of a frozen projection, not a test of the signal.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isotrophy_workbench import expand_initial_state, parse_rows, read_text  # type: ignore
from scripts.v13a_liao2021_preflight import (  # type: ignore
    PARITY_TRANSLATION,
    SUPP_B,
    STAGED,
    _rotation_z,
    expand_liao2021_state,
    parse_liao2021,
    vf_value,
)

CUTPOINTS = (0.25, 0.50)
FRAMES_ROT_DEG = [37.0, 90.0, 211.0]
DEFAULT_SEED = 20260523
SUPP_B_BAR = 0.15
LIAO2021_BAR = 0.05


def zone_index(vf: float) -> int:
    if vf < CUTPOINTS[0]:
        return 0
    if vf < CUTPOINTS[1]:
        return 1
    return 2


def dist_to_cutpoint(vf: float) -> float:
    return min(abs(vf - CUTPOINTS[0]), abs(vf - CUTPOINTS[1]))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def audit(name, rows, expand_fn, n, seed, bar, out: Path):
    rng = np.random.default_rng(seed)
    pick = sorted(int(i) for i in rng.choice(len(rows), size=min(n, len(rows)), replace=False))
    recs = []
    changed = checked = failed = 0
    pf_changes = pf_total = 0
    for k, i in enumerate(pick, start=1):
        row = rows[i]
        idx = getattr(row, "index", i)
        try:
            masses, x0, v0 = expand_fn(row, center_com=True)
            vf_base = vf_value(masses, x0, v0, row.period)
        except Exception:  # noqa: BLE001
            failed += 1
            recs.append({"orbit_index": idx, "status": "base_failed"})
            print(f"[{name}] {k}/{len(pick)} idx={idx} BASE FAILED", flush=True)
            continue
        bz = zone_index(vf_base)
        dist = dist_to_cutpoint(vf_base)
        frame_zones = {}
        orbit_changed = False
        any_frame = False
        perturbations = [(f"rot{int(d)}", _rotation_z(d), False) for d in FRAMES_ROT_DEG] + [("trans", None, True)]
        for fname, R, is_trans in perturbations:
            try:
                if is_trans:
                    xt = x0 + PARITY_TRANSLATION
                    xt = xt - np.average(xt, axis=0, weights=masses)
                    rz = zone_index(vf_value(masses, xt, v0, row.period))
                else:
                    rz = zone_index(vf_value(masses, x0 @ R.T, v0 @ R.T, row.period))
                frame_zones[fname] = rz
                pf_total += 1
                any_frame = True
                if rz != bz:
                    pf_changes += 1
                    orbit_changed = True
            except Exception:  # noqa: BLE001
                frame_zones[fname] = None
        if not any_frame:
            failed += 1
            recs.append({"orbit_index": idx, "status": "all_frames_failed"})
            print(f"[{name}] {k}/{len(pick)} idx={idx} ALL FRAMES FAILED", flush=True)
            continue
        checked += 1
        changed += int(orbit_changed)
        recs.append({
            "orbit_index": idx, "base_zone": bz, "zone_changed": orbit_changed,
            "distance_to_nearest_cutpoint": round(dist, 6),
            "zone_rot37": frame_zones.get("rot37"), "zone_rot90": frame_zones.get("rot90"),
            "zone_rot211": frame_zones.get("rot211"), "zone_trans": frame_zones.get("trans"),
            "status": "ok",
        })
        print(f"[{name}] {k}/{len(pick)} idx={idx} base_zone={bz} changed={orbit_changed} "
              f"dist={dist:.3f}", flush=True)

    frac = (changed / checked) if checked else None
    lo, hi = wilson_ci(changed, checked)
    decisive_below = hi <= bar
    decisive_above = lo > bar
    straddles = lo <= bar < hi
    dists = sorted(r["distance_to_nearest_cutpoint"] for r in recs if r.get("status") == "ok")
    summary = {
        "catalog": name, "sampled": len(pick), "checked": checked, "integration_failed": failed,
        "orbit_zone_changed_count": changed,
        "zone_change_fraction": (round(frac, 4) if frac is not None else None),
        "wilson95_low": round(lo, 4), "wilson95_high": round(hi, 4),
        "decision_bar": bar, "decisive_below_bar": decisive_below,
        "decisive_above_bar": decisive_above, "straddles_bar": straddles,
        "per_frame_change_rate": (round(pf_changes / pf_total, 4) if pf_total else None),
        "per_frame_total": pf_total, "seed": seed, "frames": FRAMES_ROT_DEG + ["translation"],
        "base_distance_quantiles": {
            "min": (dists[0] if dists else None),
            "median": (float(np.quantile(dists, 0.5)) if dists else None),
            "max": (dists[-1] if dists else None)},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"frame_zone_{name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / f"frame_zone_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, extrasaction="ignore", fieldnames=[
            "orbit_index", "base_zone", "zone_changed", "distance_to_nearest_cutpoint",
            "zone_rot37", "zone_rot90", "zone_rot211", "zone_trans", "status"])
        w.writeheader(); w.writerows(recs)
    verdict = ("decisive_below_bar" if decisive_below else
               "decisive_above_bar" if decisive_above else "straddles_bar_inconclusive")
    print(f"[{name}] DONE: zone_change {changed}/{checked} = {summary['zone_change_fraction']} "
          f"Wilson95 [{summary['wilson95_low']}, {summary['wilson95_high']}] vs bar {bar} -> {verdict} "
          f"(failed {failed})")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", choices=["liao2021", "suppb"], required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=str(ROOT / "results/isotrophy/k-facet-v13b-frame-zone-stability"))
    args = ap.parse_args()
    out = Path(args.out)
    if args.catalog == "liao2021":
        audit("liao2021", parse_liao2021(STAGED), expand_liao2021_state, args.n, args.seed, LIAO2021_BAR, out)
    else:
        audit("suppb", parse_rows(read_text(str(SUPP_B)), source="B"), expand_initial_state,
              args.n, args.seed, SUPP_B_BAR, out)


if __name__ == "__main__":
    main()
