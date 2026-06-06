#!/usr/bin/env python3
"""Sundog Certificate Problem v2 — stronger-ISD attacker ladder (Prange/LB/Stern).

Frozen contract: docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V2_SLATE.md

Three attackers recover a light same-syndrome witness e* (He*=z, wt(e*)<=tau) from z
ONLY, on the same code as v1 with a decoupled target manifest. Capacity unit = ops.

  Prange       : info set I (size k); candidate error on the n-k complement.
  Lee-Brickell : allow p=2 errors inside I; enumerate weight-2 info-set patterns.
  Stern (p,l)  : split I into two k/2 halves; meet-in-the-middle on an l-row window.

Valid-iteration convention: only full-rank info-set draws count toward the iteration
budget B; rank-deficient draws are charged to ops and audited (rank_fail), never B.

--smoke runs a THROWAWAY regime: validates each attacker (recovers valid witnesses;
measured curve matches analytic 1-(1-p)^B), CALIBRATES op constants, and emits
prediction_lock.json with the frozen [128,64] w=12 predicted 50% C(ops) per attacker.
Default (no --smoke) is the frozen run (operator-gated; do NOT run without the go).
Deterministic.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---- GF(2) core (op-counted) --------------------------------------------------

def matvec(M, v):
    return ((M & v[None, :]).sum(axis=1) & 1).astype(np.uint8), int(M.shape[0] * M.shape[1])


def gf2_inverse(M):
    """Gauss-Jordan over GF(2); returns (inv|None, ok, bit-ops)."""
    m = M.shape[0]
    A = M.copy().astype(np.uint8)
    I = np.eye(m, dtype=np.uint8)
    ops = 0
    for col in range(m):
        rows = np.nonzero(A[col:, col])[0]
        ops += int(m - col)
        if rows.size == 0:
            return None, False, ops
        piv = col + int(rows[0])
        if piv != col:
            A[[col, piv]] = A[[piv, col]]; I[[col, piv]] = I[[piv, col]]
        mask = A[:, col].astype(bool).copy(); mask[col] = False
        if mask.any():
            A[mask] ^= A[col]; I[mask] ^= I[col]
            ops += int(mask.sum()) * m * 2
    return I, True, ops


def make_code(n, k, seed):
    rng = np.random.default_rng(seed); m = n - k
    P = rng.integers(0, 2, size=(k, m), dtype=np.uint8)
    G = np.concatenate([np.eye(k, dtype=np.uint8), P], axis=1)
    H = np.concatenate([P.T, np.eye(m, dtype=np.uint8)], axis=1)
    assert (((G @ H.T) & 1).sum() == 0)
    return G, H


def sample_targets(G, H, n, k, w, T, target_seed):
    """Decoupled target sampling (independent of any attacker RNG)."""
    rng = np.random.default_rng(target_seed)
    targets = []
    for tid in range(T):
        s = rng.integers(0, 2, size=k, dtype=np.uint8)
        e = np.zeros(n, dtype=np.uint8); e[rng.choice(n, size=w, replace=False)] = 1
        y = ((s @ G) & 1 ^ e).astype(np.uint8)
        z = (H @ y) & 1
        targets.append({"id": tid, "z": z.astype(np.uint8), "label_wt_e": int(w)})
    return targets


def systematic(H, z, J, n):
    """U=H_J^{-1}; return (Hp=U@H, zp=U@z, ops) or (None,None,ops) on rank-fail."""
    Hj = H[:, J]
    inv, ok, ops = gf2_inverse(Hj)
    if not ok:
        return None, None, ops
    Hp = (inv @ H) & 1
    zp = (inv @ z) & 1
    ops += int(inv.shape[0] * inv.shape[1] * H.shape[1])  # U@H
    ops += int(inv.shape[0] * inv.shape[1])               # U@z
    return Hp.astype(np.uint8), zp.astype(np.uint8), ops


def popcount(v):
    return int(v.sum())


# ---- the three attackers (each: first_success over <= max_B valid iters) -------
# Each returns (first_success_B|None, valid_iters, rank_fail, ops).

def attacker_run(kind, H, z, n, k, w, tau, max_B, rng, p=2, l=8):
    m = n - k
    B = 0; rank_fail = 0; ops = 0
    while B < max_B:
        I = rng.choice(n, size=k, replace=False)
        J = np.setdiff1d(np.arange(n), I, assume_unique=False)
        Hp, zp, o = systematic(H, z, J, n); ops += o
        if Hp is None:
            rank_fail += 1; continue
        B += 1
        found = False
        if kind == "prange":
            # candidate error supported on J = zp (since Hp[:,J]=I); weight = wt(zp)
            ops += m
            if popcount(zp) <= tau:
                found = True
        elif kind == "lb":
            for S in itertools.combinations(I.tolist(), p):
                contrib = np.zeros(m, dtype=np.uint8)
                for col in S:
                    contrib ^= Hp[:, col]
                ops += p * m
                eJ = zp ^ contrib; ops += m
                if p + popcount(eJ) <= tau:
                    found = True; break
        elif kind == "stern":
            A = I[: k // 2].tolist(); Bset = I[k // 2:].tolist()
            zwin = zp[:l]
            # build L_A: window-syndrome -> list of S_A
            LA = {}
            for SA in itertools.combinations(A, p):
                wsyn = np.zeros(l, dtype=np.uint8)
                for col in SA:
                    wsyn ^= Hp[:l, col]
                ops += p * l
                LA.setdefault(int(np.packbits(wsyn, axis=0)[0]) if l <= 8 else tuple(wsyn), []).append(SA)
            # probe with L_B
            for SB in itertools.combinations(Bset, p):
                wsyn = np.zeros(l, dtype=np.uint8)
                for col in SB:
                    wsyn ^= Hp[:l, col]
                ops += p * l
                target = wsyn ^ zwin
                key = int(np.packbits(target, axis=0)[0]) if l <= 8 else tuple(target)
                for SA in LA.get(key, []):
                    contrib = np.zeros(m, dtype=np.uint8)
                    for col in (*SA, *SB):
                        contrib ^= Hp[:, col]
                    eJ = zp ^ contrib; ops += 3 * p * m + m
                    if 2 * p + popcount(eJ) <= tau:
                        found = True; break
                if found:
                    break
        if found:
            return B, B, rank_fail, ops
    return None, B, rank_fail, ops


# ---- analytic success-per-valid-iteration -------------------------------------

def p_success(kind, n, k, w, p=2, l=8):
    C = math.comb; m = n - k
    if kind == "prange":
        return C(m, w) / C(n, w)
    if kind == "lb":
        return C(k, p) * C(m, w - p) / C(n, w)
    if kind == "stern":
        return C(k // 2, p) ** 2 * C(m - l, w - 2 * p) / C(n, w)
    raise ValueError(kind)


# ---- analytic enumeration op-counts (exact match to harness counters) ----------
# Per-iter decomposes EXACTLY as base(m) + enum(attacker): Prange does no
# enumeration, so its ops_per_valid_iter IS the shared base probe (gauss + U@H/U@z
# matmul + rank-fail overhead). LB/Stern enumeration is counted analytically below;
# subtracting Prange's per-iter recovers it cleanly (base + rank-fail cancel).

def enum_lb(k, m, p):
    # per weight-p info pattern: p column-XORs of len m (ops += p*m) + 1 eJ XOR (ops += m)
    return math.comb(k, p) * (p + 1) * m


def enum_stern(k, m, l, p):
    half = math.comb(k // 2, p)
    lists = 2 * half * (p * l)                     # build L_A + probe L_B: p*l each
    collide = (half * half / 2 ** l) * (3 * p + 1) * m  # expected collisions * (3p+1)*m
    return lists + collide


# ---- run ----------------------------------------------------------------------

def run_one_size(n, k, w, tau, p, l, T, code_seed, target_seed, seeds, ladders):
    G, H = make_code(n, k, code_seed)
    targets = sample_targets(G, H, n, k, w, T, target_seed)
    out = {"regime": dict(n=n, k=k, w=w, tau=tau, T=T, code_seed=code_seed, target_seed=target_seed), "attackers": {}}
    for kind in ("prange", "lb", "stern"):
        pa = p_success(kind, n, k, w, p, l); Na = 1.0 / pa
        rng = np.random.default_rng(seeds[kind])
        first, rf_tot, ops_tot, viter_tot = [], 0, 0, 0
        max_B = ladders[kind][-1]
        for tg in targets:
            fb, vit, rf, ops = attacker_run(kind, H, tg["z"], n, k, w, tau, max_B, rng, p, l)
            first.append(fb); rf_tot += rf; ops_tot += ops; viter_tot += vit
        rows = []
        for B in ladders[kind]:
            succ = sum(1 for fb in first if fb is not None and fb <= B)
            rows.append({"B": B, "measured": succ / T, "predicted": 1 - (1 - pa) ** B, "delta": abs(succ / T - (1 - (1 - pa) ** B))})
        opvi = ops_tot / max(1, viter_tot)
        out["attackers"][kind] = {
            "p_success": pa, "N_analytic": Na, "ladder": rows,
            "max_delta": max(r["delta"] for r in rows), "valid_iters": viter_tot,
            "rank_fail": rf_tot, "ops_total": ops_tot, "ops_per_valid_iter": opvi,
            "rho_overhead": (viter_tot + rf_tot) / max(1, viter_tot),
            "curve_matches": max(r["delta"] for r in rows) < 0.15,
        }
    return out


def run_smoke(out_dir):
    p, l = 2, 8
    T = 48
    # TWO throwaway sizes (both n=2k, matching the frozen aspect ratio), to fit the
    # base(m) scaling exponent empirically. Neither is the frozen [128,64].
    sizes = [
        dict(n=80,  k=40, w=8, tau=8, code_seed=5151, target_seed=424242,
             seeds={"prange": 901, "lb": 902, "stern": 903},
             ladders={"prange": [10, 50, 200, 800, 2000, 4000], "lb": [1, 3, 10, 30, 100, 300], "stern": [1, 5, 20, 80, 300, 800]}),
        dict(n=120, k=60, w=8, tau=8, code_seed=6262, target_seed=525252,
             seeds={"prange": 911, "lb": 912, "stern": 913},
             ladders={"prange": [10, 50, 200, 800, 2000, 4000], "lb": [1, 3, 10, 30, 100, 300], "stern": [1, 5, 20, 80, 300, 800]}),
    ]
    smokes = []
    for s in sizes:
        smokes.append(run_one_size(s["n"], s["k"], s["w"], s["tau"], p, l, T, s["code_seed"], s["target_seed"], s["seeds"], s["ladders"]))

    # --- decompose base(m) [= Prange per-iter] and validate enum vs analytic ----
    calib_points = []
    for sm in smokes:
        m = sm["regime"]["n"] - sm["regime"]["k"]; k = sm["regime"]["k"]
        base = sm["attackers"]["prange"]["ops_per_valid_iter"]
        enum_lb_meas = sm["attackers"]["lb"]["ops_per_valid_iter"] - base
        enum_st_meas = sm["attackers"]["stern"]["ops_per_valid_iter"] - base
        enum_lb_an = enum_lb(k, m, p); enum_st_an = enum_stern(k, m, l, p)
        calib_points.append({
            "m": m, "k": k, "base": base,
            "enum_lb_measured": enum_lb_meas, "enum_lb_analytic": enum_lb_an, "enum_lb_ratio": enum_lb_meas / enum_lb_an,
            "enum_stern_measured": enum_st_meas, "enum_stern_analytic": enum_st_an, "enum_stern_ratio": enum_st_meas / enum_st_an,
        })

    # fit base(m) = alpha * m^beta from the two sizes
    (m1, b1), (m2, b2) = (calib_points[0]["m"], calib_points[0]["base"]), (calib_points[1]["m"], calib_points[1]["base"])
    beta = math.log(b2 / b1) / math.log(m2 / m1)
    alpha = b1 / m1 ** beta

    # --- predict frozen [128,64] w=12 ------------------------------------------
    fn, fk, fw, ftau = 128, 64, 12, 12
    fm = fn - fk
    base_frozen = alpha * fm ** beta
    enum_frozen = {"prange": 0.0, "lb": enum_lb(fk, fm, p), "stern": enum_stern(fk, fm, l, p)}
    pred = {}
    for kind in ("prange", "lb", "stern"):
        pf = p_success(kind, fn, fk, fw, p, l); Nf = 1.0 / pf
        per_iter = base_frozen + enum_frozen[kind]
        pred[kind] = {"p_success": pf, "N_analytic": Nf, "base_ops": base_frozen, "enum_ops": enum_frozen[kind],
                      "per_iter_ops_frozen": per_iter, "C_ops_50pct": Nf * math.log(2) * per_iter}
    base_C = pred["prange"]["C_ops_50pct"]
    for kind in pred:
        pred[kind]["drop_vs_prange"] = base_C / pred[kind]["C_ops_50pct"]
    C_lb, C_st = pred["lb"]["C_ops_50pct"], pred["stern"]["C_ops_50pct"]

    prediction_lock = {
        "schema": "pvnp-certificate-syndrome-v2-prediction-lock",
        "frozen_regime": dict(n=fn, k=fk, w=fw, tau=ftau, p=p, l=l),
        "formulas": {"prange": "p=C(n-k,w)/C(n,w)", "lb": "p=C(k,p)C(n-k,w-p)/C(n,w)",
                     "stern": "p=C(k/2,p)^2 C(n-k-l,w-2p)/C(n,w)"},
        "cost_model": "per_iter = base(m) + enum; base(m)=alpha*m^beta fit on two throwaway sizes; "
                      "enum analytic & exactly op-counted: enum_lb=C(k,p)(p+1)m, "
                      "enum_stern=2C(k/2,p)pl + (C(k/2,p)^2/2^l)(3p+1)m; C(ops)@50%=N*ln2*per_iter",
        "constants_calibrated_on": "two throwaway smoke sizes [80,40] & [120,60] w=8 (base scaling); enum analytic",
        "base_fit": {"alpha": alpha, "beta": beta, "points": [{"m": m1, "base": b1}, {"m": m2, "base": b2}], "base_frozen_m64": base_frozen},
        "calibration_points": calib_points,
        "predictions": pred,
        "C_best": min(C_lb, C_st), "C_best_source": "stern" if C_st <= C_lb else "lb",
        "lb_stern_ratio": C_lb / C_st,
        "external_crosscheck_v1": {
            "v1_prange_ops_per_valid_iter": 6.24e5,
            "note": "v1 [128,64] Prange measured ~6.24e5 ops/valid-iter, but v1's harness solved "
                    "e_J=H_J^-1 z WITHOUT the full systematic form U@H that LB/Stern require; v2 "
                    "re-baselines Prange WITH U@H, so v2 base is expected ~2x v1. NOT a clean equality check.",
        },
        "tolerance": "measured C(ops) within factor 2 of predicted; LB<->Stern (~%.2fx) may be within T-noise" % (C_lb / C_st),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "pvnp-certificate-syndrome-v2-smoke", "deterministic": True, "p": p, "l": l, "T": T, "sizes": smokes}
    (out_dir / "isd_v2_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "prediction_lock.json").write_text(json.dumps(prediction_lock, indent=2) + "\n", encoding="utf-8")
    return report, prediction_lock, calib_points, (alpha, beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/pvnp/certificate-syndrome-v2")
    args = ap.parse_args()
    if not args.smoke:
        print("Frozen run is operator-gated; pass --smoke for the validation+calibration run.")
        return
    rep, lock, calib, (alpha, beta) = run_smoke(REPO_ROOT / args.out)
    allok = True
    for sm in rep["sizes"]:
        rg = sm["regime"]
        print(f"=== smoke [{rg['n']},{rg['k']}] w={rg['w']} (m={rg['n']-rg['k']}) ===")
        for kind in ("prange", "lb", "stern"):
            a = sm["attackers"][kind]; ok = a["curve_matches"]; allok = allok and ok
            print(f"  {kind:7}: N={a['N_analytic']:8.1f} max|d|={a['max_delta']:.3f} {'MATCH' if ok else 'DEVIATE'}  ops/iter={a['ops_per_valid_iter']:.3e} rho={a['rho_overhead']:.2f}")
    print(f"\nattacker validation: {'ALL VALIDATED (curves track analytic at both sizes)' if allok else 'SOME DEVIATE — debug before freeze'}")
    print("\n=== calibration (enum measured vs analytic; base fit) ===")
    for c in calib:
        print(f"  m={c['m']}: base={c['base']:.3e}  enum_LB {c['enum_lb_measured']:.2e}/{c['enum_lb_analytic']:.2e}={c['enum_lb_ratio']:.2f}x  enum_Stern {c['enum_stern_measured']:.2e}/{c['enum_stern_analytic']:.2e}={c['enum_stern_ratio']:.2f}x")
    print(f"  base(m) = {alpha:.4g} * m^{beta:.3f}   ->  base(64) = {lock['base_fit']['base_frozen_m64']:.3e}")
    print(f"  (v1 [128,64] Prange ~6.24e5 ops/valid-iter; v2 base higher by ~2x because v2 computes U@H — see lock note)")
    print("\n=== prediction_lock (frozen [128,64] w=12) ===")
    for kind in ("prange", "lb", "stern"):
        pr = lock["predictions"][kind]
        print(f"  {kind:7}: per_iter={pr['per_iter_ops_frozen']:.3e} (base {pr['base_ops']:.2e}+enum {pr['enum_ops']:.2e})  C(ops)@50%={pr['C_ops_50pct']:.3e}  drop={pr['drop_vs_prange']:.1f}x  (N={pr['N_analytic']:.1f})")
    print(f"\n  C_best = min(C_LB, C_Stern) = {lock['C_best']:.3e}  [{lock['C_best_source']}]  (LB/Stern = {lock['lb_stern_ratio']:.2f}x)")


if __name__ == "__main__":
    main()
