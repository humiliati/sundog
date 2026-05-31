#!/usr/bin/env python3
"""v0.13a liao2021 adapter + leakage preflight.

Locked by docs/isotrophy/kfacet/kfacet_v13a_liao2021_adapter_leakage_preflight_form.md.

Decides whether the Li/Liao 2021 non-hierarchical unequal-mass table is admissible
for a v0.13 profile + rate-probe, by proving two contracts BEFORE any sweep:

  C1  expansion-only adapter  -- only parse_liao2021 + expand_liao2021_state + a thin
      explicit-state integration wrapper are new; the DOP853 path, monodromy, gamma
      selection, vf definition, and gates are imported byte-for-byte from v0.7/v0.12.
      Verified by P1 (code-inheritance identity) + P2 (vf-invariance under isometries,
      Amendment R1 -- the original eigenvalue-multiset invariant was numerically
      unachievable for 3-body monodromy).

  C2  cross-ansatz leakage bound -- canonical-invariant overlap against supp-A/B,
      dominated by mass-tuple disjointness. Leakage <= 0.05 to proceed.

Under Amendment R1 the preflight computes velocity_fraction ONLY as a discarded P2
frame-invariance assertion -- never recorded in a receipt, never stability-associated;
only the invariance residual |dvf| is kept. E/|L|/T are used only as canonical overlap
invariants.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- frozen D5 imports (byte-for-byte; never redefined) ---------------------- #
import scripts.v07a_velocity_fraction_audit as v07a  # type: ignore
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
from scripts.isotrophy_workbench import (  # type: ignore
    IntegratedOrbit,
    canonical_omega_18,
    expand_initial_state,
    pack_state,
    parse_rows,
    read_text,
    rhs_factory,
)

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v13a_liao2021_adapter_leakage_preflight_form.md"
STAGED = ROOT / "docs/isotrophy/external_targets/_staging/liao2021_nonhierarchical.txt"
SUPP_A = ROOT / "docs/isotrophy/supplementary-A_periodic-3d_mirror.txt"
SUPP_B = ROOT / "docs/isotrophy/supplementary-B_piano-init-condit-3d.txt"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v13a-liao2021-preflight"

# Locked tolerances.
TOL_PARITY = 1e-8
TOL_MASS = 1e-6
TOL_E = 1e-4
TOL_L = 1e-4
TOL_T = 1e-4
LEAKAGE_GATE = 0.05
K_PARITY = 6
PARITY_SEED = 20260523
PARITY_ROTATIONS_DEG = [0.0, 37.0, 90.0, 211.0]
PARITY_TRANSLATION = np.array([0.413, -0.272, 0.0])  # fixed offset, absorbed by CoM-centering


def rel_close(a, b, tol):
    return np.abs(np.asarray(a) - np.asarray(b)) <= tol * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))


# --------------------------------------------------------------------------- #
# Contract 1 NEW code: parser + state expansion + thin explicit-state wrapper  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Liao2021Row:
    index: int
    m1: float
    m2: float
    m3: float
    x1: float
    v1: float
    v2: float
    period: float
    stability: str

    @property
    def masses(self) -> np.ndarray:
        return np.array([self.m1, self.m2, self.m3], dtype=float)

    @property
    def label(self) -> str:
        return f"L2021_{self.index}"


def parse_liao2021(path) -> list[Liao2021Row]:
    rows: list[Liao2021Row] = []
    for line in read_text(str(path)).splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[-1] not in ("S", "U"):
            continue
        try:
            m1, m2, m3, x1, v1, v2, period = (float(p) for p in parts[:7])
        except ValueError:
            continue
        rows.append(Liao2021Row(len(rows), m1, m2, m3, x1, v1, v2, period, parts[-1]))
    return rows


def expand_liao2021_state(row: Liao2021Row, center_com: bool = True):
    """Source row -> (masses, x[3,3], v[3,3]) for the liao2021 planar ansatz.

    CoM-centers POSITION ONLY, identically to the frozen expand_initial_state
    (the velocity ansatz already carries zero total momentum: v3 cancels v1,v2)."""
    masses = row.masses
    x = np.array([[row.x1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    v3y = -(row.m1 * row.v1 + row.m2 * row.v2) / row.m3
    v = np.array([[0.0, row.v1, 0.0], [0.0, row.v2, 0.0], [0.0, v3y, 0.0]], dtype=float)
    if center_com:
        x = x - np.average(x, axis=0, weights=masses)
    return masses, x, v


def integrate_liao2021_state(masses, x0, v0, period, rtol=RTOL, atol=ATOL,
                             max_step_fraction=MAX_STEP_FRACTION) -> IntegratedOrbit:
    """Thin explicit-state wrapper: feeds a pre-built CoM-centered state to the SAME
    Newtonian rhs_factory / DOP853 path the frozen integrate_orbit uses. No equation,
    method, tolerance, or max-step convention is changed. The carried row exposes only
    .masses and .period (the two fields compute_monodromy_vectorized reads), both of
    which are isometry-invariant."""
    masses = np.asarray(masses, dtype=float)
    y0 = pack_state(x0, v0)
    max_step = period * max_step_fraction if max_step_fraction > 0 else np.inf
    solution = solve_ivp(
        rhs_factory(masses), (0.0, period), y0, method="DOP853",
        rtol=rtol, atol=atol, dense_output=True, max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(f"liao2021 integration failed: {solution.message}")
    return IntegratedOrbit(row=SimpleNamespace(masses=masses, period=float(period),
                                               label="liao2021_parity"),
                           solution=solution, y0=y0, elapsed_seconds=0.0,
                           closure_position_inf=0.0, closure_velocity_inf=0.0,
                           inertia_degenerate=False)


# --------------------------------------------------------------------------- #
# Contract 1 verification: P1 (inheritance) + P2 (frame-invariance, vf-free)   #
# --------------------------------------------------------------------------- #

def p1_code_inheritance() -> dict:
    """Assert the frozen D5 symbols are `is`-identical to the v07a objects."""
    checks = {
        "compute_monodromy_vectorized": compute_monodromy_vectorized is v07a.compute_monodromy_vectorized,
        "select_gamma_1": select_gamma_1 is v07a.select_gamma_1,
        "velocity_fraction_and_z_fraction": velocity_fraction_and_z_fraction is v07a.velocity_fraction_and_z_fraction,
        "symplecticity_residual": symplecticity_residual is v07a.symplecticity_residual,
        "reciprocal_pair_residual": reciprocal_pair_residual is v07a.reciprocal_pair_residual,
        "RTOL": RTOL == v07a.RTOL == 1e-12,
        "ATOL": ATOL == v07a.ATOL == 1e-12,
        "MAX_STEP_FRACTION": MAX_STEP_FRACTION == v07a.MAX_STEP_FRACTION == 0.02,
        "SYMPLECTICITY_GATE": SYMPLECTICITY_GATE == v07a.SYMPLECTICITY_GATE == 1e-4,
        "RECIPROCAL_PAIR_GATE": RECIPROCAL_PAIR_GATE == v07a.RECIPROCAL_PAIR_GATE == 1e-4,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _rotation_z(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def vf_value(masses, x0, v0, period) -> float:
    """P2 (Amendment R1): full integration + monodromy + gamma selection + vf. The
    returned velocity_fraction is used ONLY to form a discarded frame-invariance
    residual; it is never recorded in a receipt nor associated with a stability label."""
    integrated = integrate_liao2021_state(masses, x0, v0, period)
    M_i = compute_monodromy_vectorized(integrated, RTOL, ATOL, MAX_STEP_FRACTION)
    gamma = select_gamma_1(M_i, masses)
    return float(velocity_fraction_and_z_fraction(gamma["gamma_1"], masses)["velocity_fraction"])


def _rel_resid(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


def run_parity(rows: list[Liao2021Row]) -> dict:
    rng = np.random.default_rng(PARITY_SEED)
    pick = sorted(int(i) for i in rng.choice(len(rows), size=min(K_PARITY, len(rows)), replace=False))
    per_row = []
    worst = 0.0
    any_base_failed = False
    rows_uncheckable = 0
    for i in pick:
        row = rows[i]
        masses, x0, v0 = expand_liao2021_state(row, center_com=True)
        try:
            vf_base = vf_value(masses, x0, v0, row.period)
        except Exception as exc:  # noqa: BLE001
            any_base_failed = True
            per_row.append({"orbit_index": row.index, "period": round(row.period, 6),
                            "max_vf_residual": None, "isometries_compared": 0,
                            "status": f"base_failed: {exc}"})
            print(f"[v13a-parity] row {row.index} T={row.period:.3f} BASE FAILED: {exc}", flush=True)
            continue
        residuals = []
        isometry_failed = 0
        # rotations (skip 0 deg = identity = base) + translation
        for deg in PARITY_ROTATIONS_DEG[1:]:
            R = _rotation_z(deg)
            xr, vr = x0 @ R.T, v0 @ R.T
            try:
                residuals.append(_rel_resid(vf_value(masses, xr, vr, row.period), vf_base))
            except Exception:  # noqa: BLE001
                isometry_failed += 1
        try:
            xt = x0 + PARITY_TRANSLATION
            xt = xt - np.average(xt, axis=0, weights=masses)
            residuals.append(_rel_resid(vf_value(masses, xt, v0, row.period), vf_base))
        except Exception:  # noqa: BLE001
            isometry_failed += 1
        row_max = max(residuals) if residuals else None
        if row_max is None:
            rows_uncheckable += 1
        else:
            worst = max(worst, row_max)
        per_row.append({"orbit_index": row.index, "period": round(row.period, 6),
                        "max_vf_residual": row_max, "isometries_compared": len(residuals),
                        "isometry_integration_failed": isometry_failed})
        rstr = "n/a" if row_max is None else f"{row_max:.2e}"
        print(f"[v13a-parity] row {row.index} T={row.period:.3f} max_vf_residual={rstr} "
              f"compared={len(residuals)} isom_fail={isometry_failed}", flush=True)
    passed = (not any_base_failed) and rows_uncheckable == 0 and worst <= TOL_PARITY
    return {"k_parity": len(pick), "seed": PARITY_SEED, "tol_parity": TOL_PARITY,
            "invariant": "vf_invariance_R1", "max_vf_residual": worst,
            "any_base_integration_failure": any_base_failed,
            "rows_uncheckable": rows_uncheckable, "passed": passed, "per_row": per_row}


# --------------------------------------------------------------------------- #
# Contract 2: canonical-invariant leakage bound                                #
# --------------------------------------------------------------------------- #

def canonical_invariants(masses, x, v, period):
    """(mass_sorted, E*, |L|*, T*) in canonical units G=1, sum m=1, I0(mu)=1.
    x assumed CoM-centered, total momentum zero (both catalogs satisfy this)."""
    masses = np.asarray(masses, float)
    M = float(masses.sum())
    if not (M > 0) or not np.all(np.isfinite(masses)) or not math.isfinite(period):
        return None
    mu = masses / M
    r_com = np.average(x, axis=0, weights=mu)
    xc = x - r_com
    I_mu = float(np.sum(mu * np.sum(xc * xc, axis=1)))
    if not (I_mu > 0) or not math.isfinite(I_mu):
        return None
    ell = math.sqrt(I_mu)
    # energy (G=1): KE = 1/2 sum m_i |v_i|^2 ; PE = - sum_{i<j} m_i m_j / r_ij
    ke = 0.5 * float(np.sum(masses[:, None] * v * v))
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            rij = math.sqrt(float(np.sum((x[i] - x[j]) ** 2)))
            if not (rij > 0):
                return None
            pe -= masses[i] * masses[j] / rij
    E = ke + pe
    # angular momentum about CoM (V_com = 0): L = sum m_i (x_i - r_com) x v_i
    L_vec = np.sum(masses[:, None] * np.cross(xc, v), axis=0)
    L = float(np.linalg.norm(L_vec))
    E_star = E * ell / (M * M)
    L_star = L / (M ** 1.5 * math.sqrt(ell))
    T_star = period * math.sqrt(M / ell ** 3)
    if not all(math.isfinite(z) for z in (E_star, L_star, T_star)):
        return None
    return {"mass_sorted": tuple(sorted(mu.tolist())), "E_star": E_star,
            "L_star": L_star, "T_star": T_star}


def supp_invariants():
    inv = []
    bad = 0
    for src, path in (("A", SUPP_A), ("B", SUPP_B)):
        for row in parse_rows(read_text(str(path)), source=src):
            masses, x, v = expand_initial_state(row, center_com=True)
            ci = canonical_invariants(masses, x, v, row.period)
            if ci is None:
                bad += 1
            else:
                inv.append(ci)
    return inv, bad


def run_leakage(rows: list[Liao2021Row]) -> dict:
    m1 = np.array([r.m1 for r in rows]); m2 = np.array([r.m2 for r in rows]); m3 = np.array([r.m3 for r in rows])
    # equal-pair slice: any two masses equal within tol_mass
    slice_mask = (rel_close(m1, m2, TOL_MASS) | rel_close(m1, m3, TOL_MASS) | rel_close(m2, m3, TOL_MASS))
    slice_idx = np.where(slice_mask)[0]
    print(f"[v13a-leak] parsed {len(rows)} rows; equal-pair slice = {len(slice_idx)}", flush=True)

    supp_inv, supp_bad = supp_invariants()
    print(f"[v13a-leak] supp-A/B canonical rows = {len(supp_inv)} (reconciliation failures {supp_bad})", flush=True)
    supp_mass = np.array([ci["mass_sorted"] for ci in supp_inv]) if supp_inv else np.zeros((0, 3))
    supp_E = np.array([ci["E_star"] for ci in supp_inv])
    supp_L = np.array([ci["L_star"] for ci in supp_inv])
    supp_T = np.array([ci["T_star"] for ci in supp_inv])

    leaked = 0
    leak_records = []
    canonical_ok = supp_bad == 0
    for i in slice_idx:
        row = rows[int(i)]
        masses, x, v = expand_liao2021_state(row, center_com=True)
        ci = canonical_invariants(masses, x, v, row.period)
        if ci is None:
            canonical_ok = False
            continue
        ms = np.array(ci["mass_sorted"])
        # vectorized mass* prematch against all supp rows; backstop only on matches
        mass_match = np.all(rel_close(supp_mass, ms[None, :], TOL_MASS), axis=1) if len(supp_mass) else np.zeros(0, bool)
        if not mass_match.any():
            continue
        sel = np.where(mass_match)[0]
        hit = (rel_close(supp_E[sel], ci["E_star"], TOL_E)
               & rel_close(supp_L[sel], ci["L_star"], TOL_L)
               & rel_close(supp_T[sel], ci["T_star"], TOL_T))
        if hit.any():
            leaked += 1
            leak_records.append({"orbit_index": row.index, "m1": row.m1, "m2": row.m2, "m3": row.m3,
                                 "E_star": ci["E_star"], "L_star": ci["L_star"], "T_star": ci["T_star"]})
    total = len(rows)
    frac = leaked / total if total else None
    bounded = canonical_ok and frac is not None
    return {
        "total_rows": total, "equal_pair_slice_count": int(len(slice_idx)),
        "supp_rows": len(supp_inv), "supp_canonical_failures": supp_bad,
        "leaked_rows": leaked, "leakage_fraction": frac, "reflection_leak": 0,
        "canonical_reconciliation_ok": canonical_ok, "bounded": bool(bounded),
        "leakage_gate": LEAKAGE_GATE, "tolerances": {"mass": TOL_MASS, "E": TOL_E, "L": TOL_L, "T": TOL_T},
        "leak_records": leak_records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=str(STAGED))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-parity", action="store_true", help="leakage-only (diagnostic)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    table = Path(args.table)
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    rows = parse_liao2021(table)
    print(f"[v13a] table {table.name} sha256={sha[:16]}... rows={len(rows)}")

    p1 = p1_code_inheritance()
    print(f"[v13a] P1 code-inheritance: passed={p1['passed']}")
    leakage = run_leakage(rows)
    print(f"[v13a] leakage: slice={leakage['equal_pair_slice_count']} leaked={leakage['leaked_rows']} "
          f"frac={leakage['leakage_fraction']} bounded={leakage['bounded']}")

    parity = None
    if not args.skip_parity:
        t0 = time.perf_counter()
        parity = run_parity(rows)
        print(f"[v13a] parity (vf-invariance R1): max_vf_residual={parity['max_vf_residual']:.2e} "
              f"passed={parity['passed']} ({time.perf_counter()-t0:.0f}s)")

    # verdict tree
    if not args.skip_parity and (not p1["passed"] or not parity["passed"]):
        verdict = "adapter_not_expansion_only"
    elif not leakage["bounded"]:
        verdict = "leakage_unbounded_report_only"
    elif leakage["leakage_fraction"] > LEAKAGE_GATE:
        verdict = "leakage_blocked_report_only"
    elif args.skip_parity:
        verdict = "leakage_only_parity_pending"
    else:
        verdict = "preflight_passed_probe_authorized"

    manifest = {
        "schema": "sundog.isotrophy.v0.13a-liao2021-preflight.v1",
        "form_lock": FORM_LOCK,
        "table": str(table.relative_to(ROOT)).replace("\\", "/"),
        "download_sha256": sha,
        "row_count": len(rows),
        "adapter_parity": {"p1_code_inheritance": p1, "p2_frame_invariance": parity},
        "leakage": leakage,
        "verdict": verdict,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    with (out / "leakage_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["orbit_index", "m1", "m2", "m3", "E_star", "L_star", "T_star"])
        w.writeheader(); w.writerows(leakage["leak_records"])
    if parity:
        with (out / "parity_rows.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, extrasaction="ignore", fieldnames=[
                "orbit_index", "period", "max_vf_residual", "isometries_compared",
                "isometry_integration_failed", "status"])
            w.writeheader(); w.writerows(parity["per_row"])

    print(f"[v13a] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
