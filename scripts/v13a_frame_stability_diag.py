#!/usr/bin/env python3
"""v0.13a frame-stability diagnostic (escalation follow-up).

Quantifies what FRACTION of orbits have a frame-portable velocity_fraction, on both
catalogs, to decide the v0.13 disposition and whether v0.11 needs a frame caveat:

  liao2021  -- is the cross-ansatz transfer fully dead, or is there a frame-stable
               subset? (expand_liao2021_state, the v0.13a adapter)
  supp-B    -- the v0.11 domain (frozen mirrored expand_initial_state). Does the vf
               frame-fragility touch v0.11's own orbits?

For each sampled orbit it computes the base vf and the vf under the locked rotations,
keeps ONLY the max relative residual |dvf| (never vf, never the stability label), and
classifies the orbit frame-stable iff that residual <= DIAG_STABLE_TOL. This stays
inside the v0.13a R1 firewall (vf is a discarded frame-invariance assertion value).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isotrophy_workbench import expand_initial_state, parse_rows, read_text  # type: ignore
from scripts.v13a_liao2021_preflight import (  # type: ignore
    SUPP_B,
    STAGED,
    TOL_PARITY,
    _rel_resid,
    _rotation_z,
    expand_liao2021_state,
    parse_liao2021,
    vf_value,
)

DIAG_ROTATIONS_DEG = [37.0, 90.0, 211.0]
DIAG_STABLE_TOL = 1e-6   # clean separation: stable orbits ~1e-8, fragile ~1e-3..1e0
DEFAULT_SEED = 20260523


def diag_catalog(name, rows, expand_fn, n, seed, out: Path):
    rng = np.random.default_rng(seed)
    pick = sorted(int(i) for i in rng.choice(len(rows), size=min(n, len(rows)), replace=False))
    recs = []
    n_stable = n_fragile = n_failed = 0
    for k, i in enumerate(pick, start=1):
        row = rows[i]
        idx = getattr(row, "index", i)
        try:
            masses, x0, v0 = expand_fn(row, center_com=True)
            vf_base = vf_value(masses, x0, v0, row.period)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            recs.append({"orbit_index": idx, "max_vf_residual": None, "frame_stable": None,
                         "status": f"base_failed: {type(exc).__name__}"})
            print(f"[{name}] {k}/{len(pick)} idx={idx} BASE FAILED", flush=True)
            continue
        residuals = []
        for deg in DIAG_ROTATIONS_DEG:
            R = _rotation_z(deg)
            try:
                residuals.append(_rel_resid(vf_value(masses, x0 @ R.T, v0 @ R.T, row.period), vf_base))
            except Exception:  # noqa: BLE001
                pass
        if not residuals:
            n_failed += 1
            recs.append({"orbit_index": idx, "max_vf_residual": None, "frame_stable": None,
                         "status": "all_rotations_failed"})
            print(f"[{name}] {k}/{len(pick)} idx={idx} ALL ROTATIONS FAILED", flush=True)
            continue
        mx = max(residuals)
        stable = mx <= DIAG_STABLE_TOL
        n_stable += int(stable)
        n_fragile += int(not stable)
        recs.append({"orbit_index": idx, "max_vf_residual": mx, "frame_stable": stable, "status": "ok"})
        print(f"[{name}] {k}/{len(pick)} idx={idx} max_vf_residual={mx:.2e} stable={stable}", flush=True)

    checked = n_stable + n_fragile
    resid = sorted(r["max_vf_residual"] for r in recs if r["max_vf_residual"] is not None)
    q = (lambda p: float(np.quantile(resid, p)) if resid else None)
    summary = {
        "catalog": name, "sampled": len(pick), "checked": checked,
        "n_frame_stable": n_stable, "n_frame_fragile": n_fragile, "n_integration_failed": n_failed,
        "frame_stable_fraction": (n_stable / checked if checked else None),
        "diag_stable_tol": DIAG_STABLE_TOL, "locked_tol_parity": TOL_PARITY,
        "rotations_deg": DIAG_ROTATIONS_DEG, "seed": seed,
        "residual_quantiles": {"min": (resid[0] if resid else None), "q25": q(0.25),
                               "median": q(0.50), "q75": q(0.75), "max": (resid[-1] if resid else None)},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"frame_stability_{name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / f"frame_stability_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["orbit_index", "max_vf_residual", "frame_stable", "status"])
        w.writeheader(); w.writerows(recs)
    print(f"[{name}] DONE: stable {n_stable}/{checked} "
          f"(frac {summary['frame_stable_fraction']}); failed {n_failed}; "
          f"residual median {summary['residual_quantiles']['median']}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", choices=["liao2021", "suppb"], required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=str(ROOT / "results/isotrophy/k-facet-v13a-frame-stability"))
    args = ap.parse_args()
    out = Path(args.out)
    if args.catalog == "liao2021":
        rows = parse_liao2021(STAGED)
        diag_catalog("liao2021", rows, expand_liao2021_state, args.n, args.seed, out)
    else:
        rows = parse_rows(read_text(str(SUPP_B)), source="B")
        diag_catalog("suppb", rows, expand_initial_state, args.n, args.seed, out)


if __name__ == "__main__":
    main()
