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
import hashlib
import itertools
import json
import math
import subprocess
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
        wit = None
        if kind == "prange":
            # candidate error supported on J = zp (since Hp[:,J]=I); weight = wt(zp)
            ops += m
            if popcount(zp) <= tau:
                found = True
                wit = np.zeros(n, dtype=np.uint8); wit[J] = zp  # witness (audit only; not counted)
        elif kind == "lb":
            for S in itertools.combinations(I.tolist(), p):
                contrib = np.zeros(m, dtype=np.uint8)
                for col in S:
                    contrib ^= Hp[:, col]
                ops += p * m
                eJ = zp ^ contrib; ops += m
                if p + popcount(eJ) <= tau:
                    found = True
                    wit = np.zeros(n, dtype=np.uint8); wit[J] = eJ
                    for col in S:
                        wit[col] = 1
                    break
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
                        found = True
                        wit = np.zeros(n, dtype=np.uint8); wit[J] = eJ
                        for col in (*SA, *SB):
                            wit[col] = 1
                        break
                if found:
                    break
        if found:
            return B, B, rank_fail, ops, wit
    return None, B, rank_fail, ops, None


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
            fb, vit, rf, ops, _ = attacker_run(kind, H, tg["z"], n, k, w, tau, max_B, rng, p, l)
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


# ---- frozen run: manifest-before-attackers + op-budget ladder + per-file bundle --

def verifier_ops(n, k):
    """Flat witness-verifier op-count, UNCHANGED from v1 (16,576 at [128,64]): two
    GF(2) matvecs (recompute z=Hy + check He*) + wt(e*) popcount + He*==z compare."""
    m = n - k
    return 2 * m * n + n + m


def sample_frozen_manifest(G, H, n, k, w, T, target_seed):
    """Decoupled target sampling (same RNG stream as sample_targets) keeping labels."""
    rng = np.random.default_rng(target_seed)
    targets = []
    for tid in range(T):
        s = rng.integers(0, 2, size=k, dtype=np.uint8)
        e = np.zeros(n, dtype=np.uint8); e[rng.choice(n, size=w, replace=False)] = 1
        y = ((s @ G) & 1 ^ e).astype(np.uint8)
        z = (H @ y) & 1
        targets.append({"id": tid, "z": z.astype(np.uint8), "s": s, "e": e, "wt_e": int(e.sum())})
    return targets


def _median(xs):
    ys = sorted(xs); n = len(ys)
    if n == 0:
        return None
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def _c50_ops(per_target, T):
    """50%-recovery op budget: median of ops_at_success over all T (censored=inf)."""
    vals = sorted((r["ops_at_success"] if r["ops_at_success"] is not None else math.inf) for r in per_target)
    med = vals[T // 2] if T % 2 else 0.5 * (vals[T // 2 - 1] + vals[T // 2])
    return None if med == math.inf else med


def run_frozen(out_dir, regime, lock_pred, seeds, max_B, git_sha, witness_sample=6):
    n, k, w, tau = regime["n"], regime["k"], regime["w"], regime["tau"]
    code_seed, target_seed, T, p, l = regime["code_seed"], regime["target_seed"], regime["T"], regime["p"], regime["l"]
    m = n - k
    out_dir.mkdir(parents=True, exist_ok=True)
    authoritative = lock_pred is not None and lock_pred.get("frozen_regime") == dict(n=n, k=k, w=w, tau=tau, p=p, l=l)

    # --- code + manifest, emitted BEFORE any attacker runs ---
    G, H = make_code(n, k, code_seed)
    code_valid = int(((G @ H.T) & 1).sum()) == 0
    targets = sample_frozen_manifest(G, H, n, k, w, T, target_seed)
    labels_wt_ok = all(t["wt_e"] == w for t in targets)
    manifest = {
        "schema": "pvnp-certificate-syndrome-v2-target-manifest",
        "regime": dict(n=n, k=k, w=w, tau=tau, code_seed=code_seed, target_seed=target_seed, T=T),
        "attacker_visible": [{"id": t["id"], "z": t["z"].tolist()} for t in targets],
        "labels_only_scoring_fields": [{"id": t["id"], "s": t["s"].tolist(), "e": t["e"].tolist(), "wt_e": t["wt_e"]} for t in targets],
        "note": "attackers read ONLY attacker_visible[].z; labels_only_* is never an attacker input.",
    }
    mbytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out_dir / "target_manifest.json").write_bytes(mbytes)
    manifest_sha = hashlib.sha256(mbytes).hexdigest()
    (out_dir / "verifier_access_declaration.json").write_text(json.dumps({
        "schema": "pvnp-certificate-syndrome-v2-access",
        "attacker_input": "z only (from target_manifest.attacker_visible)",
        "scoring_only_never_seen_by_attacker": ["s", "e", "wt_e"],
        "verifier_inputs": "(y, exhibited e*)",
        "target_manifest_sha256": manifest_sha,
        "manifest_emitted_before_attackers": True,
    }, indent=2) + "\n", encoding="utf-8")

    vops = verifier_ops(n, k)

    # --- run each attacker to first success per target (attacker sees ONLY z) ---
    per_attacker = {}
    iter_ladder = sorted({int(x) for kind in ("prange", "lb", "stern")
                          for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]})
    for kind in ("prange", "lb", "stern"):
        rng = np.random.default_rng(seeds[kind])
        pa = p_success(kind, n, k, w, p, l)
        rows = []
        wsamples = []
        for t in targets:
            fb, vit, rf, ops, wit = attacker_run(kind, H, t["z"], n, k, w, tau, max_B[kind], rng, p, l)
            wok = None
            if wit is not None:
                He = (H @ wit) & 1
                wok = bool(np.array_equal(He, t["z"])) and int(wit.sum()) <= tau
                if len(wsamples) < witness_sample:
                    wsamples.append({"id": t["id"], "He_eq_z": bool(np.array_equal(He, t["z"])),
                                     "wt": int(wit.sum()), "tau": tau, "valid": wok})
            rows.append({"id": t["id"], "first_B": fb, "ops": ops,
                         "ops_at_success": (ops if fb is not None else None),
                         "valid_iters": vit, "rank_fail": rf, "witness_ok": wok})
        succ = [r for r in rows if r["first_B"] is not None]
        n_succ = len(succ)
        c50_ops = _c50_ops(rows, T)
        c50_iters = _median([r["first_B"] for r in succ]) if n_succ else None
        # iteration curve vs analytic (cross-check)
        icurve = [{"B": B, "measured": sum(1 for r in rows if r["first_B"] is not None and r["first_B"] <= B) / T,
                   "predicted": 1 - (1 - pa) ** B} for B in iter_ladder]
        # op-budget recovery curve (relative to measured c50_ops if available)
        base_budget = c50_ops if c50_ops else (lock_pred["predictions"][kind]["C_ops_50pct"] if authoritative else 1e6)
        ocurve = [{"frac_of_c50": f, "op_budget": f * base_budget,
                   "recovery": sum(1 for r in rows if r["ops_at_success"] is not None and r["ops_at_success"] <= f * base_budget) / T}
                  for f in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
        per_attacker[kind] = {
            "p_success": pa, "N_analytic": 1.0 / pa, "n_success": n_succ, "n_censored": T - n_succ,
            "C_ops_50pct_measured": c50_ops, "C_iters_50pct_measured": c50_iters,
            "valid_iters_total": sum(r["valid_iters"] for r in rows),
            "rank_fail_total": sum(r["rank_fail"] for r in rows),
            "ops_total": sum(r["ops"] for r in rows),
            "witness_all_valid": all(r["witness_ok"] for r in succ) if n_succ else None,
            "iter_curve": icurve, "op_curve": ocurve, "witness_samples": wsamples, "per_target": rows,
        }

    # --- capacity ladder + prediction-vs-measured ---
    Cm = {kind: per_attacker[kind]["C_ops_50pct_measured"] for kind in ("prange", "lb", "stern")}
    cap = {}
    for kind in ("prange", "lb", "stern"):
        meas = Cm[kind]
        pred = lock_pred["predictions"][kind]["C_ops_50pct"] if authoritative else None
        ratio = (meas / pred) if (meas and pred) else None
        cap[kind] = {
            "C_ops_50pct_measured": meas, "C_ops_50pct_predicted": pred,
            "measured_over_predicted": ratio, "within_factor2": (0.5 <= ratio <= 2.0) if ratio else None,
            "drop_vs_prange_measured": (Cm["prange"] / meas) if (meas and Cm["prange"]) else None,
            "drop_vs_prange_predicted": (lock_pred["predictions"]["prange"]["C_ops_50pct"] / pred) if authoritative else None,
        }
    C_lb, C_st = Cm["lb"], Cm["stern"]
    C_best = min(x for x in (C_lb, C_st) if x) if (C_lb or C_st) else None
    C_best_src = ("stern" if (C_st and (not C_lb or C_st <= C_lb)) else "lb") if C_best else None
    lb_stern_ratio = (C_lb / C_st) if (C_lb and C_st) else None

    # --- suggested verdict (owner makes the final call) ---
    all_witness_ok = all(per_attacker[k]["witness_all_valid"] for k in per_attacker if per_attacker[k]["n_success"])
    verifier_cheap = vops < min(x for x in Cm.values() if x) if any(Cm.values()) else False
    prange_lb_resolves = bool(Cm["prange"] and Cm["lb"] and Cm["prange"] > Cm["lb"])
    within_tol = all(cap[k]["within_factor2"] for k in cap) if authoritative else None
    if not code_valid or not labels_wt_ok:
        verdict = "void_run (code/label integrity)"
    elif not verifier_cheap:
        verdict = "6.1_vacuity_or_6.4_overhead (verifier not below cheapest attacker)"
    elif not prange_lb_resolves:
        verdict = "6.5_boundary (Prange->LB large step did not resolve)"
    elif not all_witness_ok:
        verdict = "void_run (an attacker exhibited an invalid witness)"
    elif authoritative and not within_tol:
        verdict = "model_deviation (a measured C off the locked prediction beyond factor 2)"
    else:
        verdict = "bounded_positive_attacker_hierarchy_one_wayness"

    # --- emit per-file bundle ---
    def w_json(name, obj):
        (out_dir / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    w_json("manifest.json", {
        "schema": "pvnp-certificate-syndrome-v2-run-manifest",
        "regime": dict(n=n, k=k, w=w, tau=tau, code_seed=code_seed, target_seed=target_seed, T=T, p=p, l=l),
        "code_identical_to_v1_fields": ["n=128", "k=64", "w=12", "tau=12", "code_seed=2026128"],
        "attacker_ladder": ["prange", "lee-brickell p=2", "stern p=2 l=8"],
        "git_sha": git_sha, "target_manifest_sha256": manifest_sha,
        "prediction_lock_authoritative": authoritative, "suggested_verdict": verdict,
        "code_valid_GHt0": code_valid, "labels_wt_ok": labels_wt_ok,
    })
    w_json("capacity_ladder.json", {"measured_C_ops": Cm, "per_attacker": cap,
                                    "C_best": C_best, "C_best_source": C_best_src, "lb_stern_ratio": lb_stern_ratio})
    w_json("prediction_vs_measured.json", {
        "authoritative": authoritative, "tolerance": "factor 2",
        "per_attacker": {k: {"measured": cap[k]["C_ops_50pct_measured"], "predicted": cap[k]["C_ops_50pct_predicted"],
                             "ratio": cap[k]["measured_over_predicted"], "within_factor2": cap[k]["within_factor2"]}
                         for k in cap},
        "drops_measured": {k: cap[k]["drop_vs_prange_measured"] for k in cap},
        "drops_predicted": {k: cap[k]["drop_vs_prange_predicted"] for k in cap},
        "C_best": C_best, "C_best_source": C_best_src, "lb_stern_ratio": lb_stern_ratio,
    })
    w_json("valid_iteration_audit.json", {k: {"valid_iters": per_attacker[k]["valid_iters_total"],
                                              "rank_fail_draws": per_attacker[k]["rank_fail_total"],
                                              "ops_total_incl_overhead": per_attacker[k]["ops_total"],
                                              "n_success": per_attacker[k]["n_success"],
                                              "n_censored": per_attacker[k]["n_censored"]} for k in per_attacker})
    w_json("witness_validity_audit.json", {k: {"all_valid": per_attacker[k]["witness_all_valid"],
                                              "samples": per_attacker[k]["witness_samples"]} for k in per_attacker})
    w_json("op_count_report.json", {
        "verifier_ops_flat": vops, "verifier_ops_breakdown": "2*m*n (two matvecs) + n (wt popcount) + m (eq compare)",
        "attacker_C_ops": Cm, "find_vs_check_gap": {k: (Cm[k] / vops if Cm[k] else None) for k in Cm},
        "verifier_below_all_attackers": verifier_cheap,
    })
    # attacker_ladder_curve.csv
    csv_lines = ["attacker,op_budget_frac_of_c50,op_budget,recovery_fraction"]
    for kind in ("prange", "lb", "stern"):
        for r in per_attacker[kind]["op_curve"]:
            csv_lines.append(f"{kind},{r['frac_of_c50']},{r['op_budget']:.1f},{r['recovery']:.4f}")
    (out_dir / "attacker_ladder_curve.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    # iteration cross-check curve (analytic comparison), kept alongside
    w_json("iteration_crosscheck.json", {k: per_attacker[k]["iter_curve"] for k in per_attacker})

    falsifier = [
        "# v2 frozen run — falsifier summary", "",
        f"- suggested verdict: **{verdict}**",
        f"- code valid (G Hᵀ=0): {code_valid}; labels wt=w: {labels_wt_ok}",
        f"- verifier flat op-count: {vops} (< every attacker C: {verifier_cheap})",
        f"- witnesses all valid (He*=z ∧ wt≤τ): {all_witness_ok}",
        f"- measured C(ops)@50%: Prange {Cm['prange']}, LB {Cm['lb']}, Stern {Cm['stern']}",
        f"- C_best = min(C_LB,C_Stern) = {C_best} [{C_best_src}]; LB/Stern = {lb_stern_ratio}",
        f"- Prange→LB resolves: {prange_lb_resolves}; within locked tolerance (factor 2): {within_tol}",
        "",
        "Boundary: C_best is an UPPER BOUND against the tested attackers (BJMM/MMT lower it = new slate).",
        "No crypto one-wayness / no P-vs-NP claim; op-count cost, wall-time diagnostic-only.",
    ]
    (out_dir / "falsifier_summary.md").write_text("\n".join(falsifier) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        f"# v2 frozen run outputs — {regime['n']},{regime['k']} w={w}\n\n"
        f"Suggested verdict: **{verdict}**. See capacity_ladder.json / prediction_vs_measured.json.\n"
        f"Per-file bundle per the slate Required-outputs list. Manifest emitted before attackers "
        f"(sha256 {manifest_sha[:16]}…); attackers consumed only z.\n", encoding="utf-8")

    return {"verdict": verdict, "Cm": Cm, "C_best": C_best, "C_best_source": C_best_src,
            "lb_stern_ratio": lb_stern_ratio, "verifier_ops": vops, "authoritative": authoritative,
            "per_attacker": {k: {kk: per_attacker[k][kk] for kk in ("n_success", "n_censored", "C_ops_50pct_measured",
                              "C_iters_50pct_measured", "N_analytic", "rank_fail_total", "witness_all_valid")} for k in per_attacker},
            "cap": cap, "manifest_sha": manifest_sha}


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--frozen", action="store_true", help="OPERATOR-GATED real frozen [128,64] run")
    ap.add_argument("--frozen-harness-test", action="store_true", help="validate the frozen-run plumbing on a tiny throwaway regime")
    ap.add_argument("--out", default="results/pvnp/certificate-syndrome-v2")
    args = ap.parse_args()

    if args.frozen_harness_test:
        regime = dict(n=48, k=24, w=6, tau=6, code_seed=888, target_seed=999, T=16, p=2, l=8)
        seeds = {"prange": 71, "lb": 72, "stern": 73}
        max_B = {"prange": 3000, "lb": 400, "stern": 600}
        res = run_frozen(REPO_ROOT / "results/pvnp/_certsyn-v2-harness-test", regime, None, seeds, max_B, _git_sha())
        print("=== frozen-harness-test (tiny throwaway [48,24] w=6 — plumbing only) ===")
        print(f"  verdict={res['verdict']}  verifier_ops={res['verifier_ops']}  manifest_sha={res['manifest_sha'][:16]}")
        for k in ("prange", "lb", "stern"):
            pk = res["per_attacker"][k]
            print(f"  {k:7}: n_succ={pk['n_success']}/{pk['n_success']+pk['n_censored']} C(ops)@50%={pk['C_ops_50pct_measured']} witness_all_valid={pk['witness_all_valid']}")
        print(f"  C_best={res['C_best']} [{res['C_best_source']}] LB/Stern={res['lb_stern_ratio']}")
        return

    if args.frozen:
        lock = json.loads((REPO_ROOT / "docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V2_PREDICTION_LOCK.json").read_text(encoding="utf-8"))
        regime = dict(n=128, k=64, w=12, tau=12, code_seed=2026128, target_seed=2026220, T=64, p=2, l=8)
        seeds = {"prange": 12801, "lb": 12802, "stern": 12803}
        max_B = {"prange": 60000, "lb": 1000, "stern": 1000}
        res = run_frozen(REPO_ROOT / args.out, regime, lock, seeds, max_B, _git_sha())
        print("=== v2 FROZEN run [128,64] w=12 ===")
        print(f"  suggested verdict: {res['verdict']}")
        for k in ("prange", "lb", "stern"):
            pk = res["per_attacker"][k]; c = res["cap"][k]
            print(f"  {k:7}: C(ops)@50%={pk['C_ops_50pct_measured']:.3e} pred={c['C_ops_50pct_predicted']:.3e} ratio={c['measured_over_predicted']:.2f} within2x={c['within_factor2']} drop={c['drop_vs_prange_measured']}")
        print(f"  C_best={res['C_best']:.3e} [{res['C_best_source']}]  LB/Stern={res['lb_stern_ratio']:.2f}  verifier={res['verifier_ops']}")
        return

    if not args.smoke:
        print("Pass --smoke (calibration), --frozen-harness-test (plumbing), or --frozen (operator-gated real run).")
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
