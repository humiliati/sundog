#!/usr/bin/env python3
"""Sundog Certificate Problem v3 — scaling-ladder harness (LB + Stern per rung).

Stage-1 frozen contract: docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V3_SLATE.md

Reuses the v2-validated GF(2) core / attackers / bundle emitter (loaded via importlib).
v3 adds: the empirical pre-calibration (direct per-iteration success sampling + Stern `l`
re-optimization), a two-size base(m) cost smoke spanning the v3 m-range, the stage-2
prediction lock, the per-rung frozen scoring (LB+Stern; Prange formula at scale), and the
cross-rung scaling summary.

Operator-facing modes (PowerShell contract; set $env:PYTHONHASHSEED="0"):
  --stage1-smoke   two-size base(m) probe + enum cross-check (inline if <10min)
  --precal         empirical success-rate measurement + l-reopt -> prediction_lock_v3.json (STAGED)
  --frozen --rung N   frozen LB+Stern scoring at rung N (OPERATOR-GATED, needs the lock)
  --summarize      cross-rung scaling_summary.json + SCALING_LADDER.md

Determinism: seed-pinned pure-numpy; PYTHONHASHSEED pinned by the contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("certsyn_v2", REPO / "scripts" / "pvnp-certificate-syndrome-v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

# ---- stage-1 frozen ladder (verbatim from the slate) --------------------------
RUNGS = {
    1: dict(n=128, k=64, w=16, tau=16, code_seed=20263101, target_seed=20263102,
            precal_target_seed=20263103, l_candidates=[5, 6, 7, 8, 9], Cnw="9.334e19"),
    2: dict(n=160, k=80, w=16, tau=16, code_seed=20263201, target_seed=20263202,
            precal_target_seed=20263203, l_candidates=[6, 7, 8, 9, 10], Cnw="4.060e21"),
    3: dict(n=192, k=96, w=18, tau=18, code_seed=20263301, target_seed=20263302,
            precal_target_seed=20263303, l_candidates=[7, 8, 9, 10, 11], Cnw="8.629e24"),
}
P = 2
T = 64           # frozen-scoring targets per rung
T_PRE = 16       # pre-calibration throwaway targets
SUCC_BUDGET = 32  # expected successes targeted by the pre-cal budget
LN2 = math.log(2)
# base(m) cost smoke: two throwaway sizes bracketing the v3 m-range (m=64,96), n=2m, small w
SMOKE_SIZES = [dict(n=128, k=64, w=8, seed=70263164), dict(n=192, k=96, w=8, seed=70263196)]
SMOKE_BASE_VALID = 600   # fixed valid iters per base probe (cost is low-variance)
SMOKE_ENUM_VALID = 120   # light enum cross-check budget


def analytic_l_star(n, k, w, l_candidates, base_m):
    """cost-optimal l over the candidate set using base+enum and analytic p (selection seed)."""
    m = n - k
    best = (math.inf, None)
    for l in l_candidates:
        ps = v2.p_success("stern", n, k, w, P, l)
        if ps <= 0:
            continue
        per = base_m + v2.enum_stern(k, m, l, P)
        c = (1 / ps) * LN2 * per
        if c < best[0]:
            best = (c, l)
    return best[1]


# ---- base(m) cost smoke -------------------------------------------------------

def base_probe(n, k, seed, n_valid):
    """Measured base(m) = systematic-form ops per valid iter (incl rank-fail overhead),
    matching v2's Prange per-iter definition (systematic ops over all draws + m weight-check
    over valid draws) / valid. No success logic — pure cost."""
    m = n - k
    G, H = v2.make_code(n, k, seed)
    rng = np.random.default_rng(seed ^ 0x5151)
    z = (H @ rng.integers(0, 2, size=n, dtype=np.uint8)) & 1  # arbitrary throwaway syndrome
    valid = 0; ops = 0; rankfail = 0; alln = np.arange(n)
    while valid < n_valid:
        I = rng.choice(n, size=k, replace=False)
        J = np.setdiff1d(alln, I)
        Hp, zp, o = v2.systematic(H, z, J, n); ops += o
        if Hp is None:
            rankfail += 1; continue
        valid += 1; ops += m  # the prange weight-check, as in v2's base
    return ops / valid, (valid + rankfail) / valid


def run_stage1_smoke(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = []
    for s in SMOKE_SIZES:
        b, rho = base_probe(s["n"], s["k"], s["seed"], SMOKE_BASE_VALID)
        pts.append(dict(m=s["n"] - s["k"], base=b, rho=rho))
    (m1, b1), (m2, b2) = (pts[0]["m"], pts[0]["base"]), (pts[1]["m"], pts[1]["base"])
    beta = math.log(b2 / b1) / math.log(m2 / m1)
    alpha = b1 / m1 ** beta
    # light enum cross-check at the smaller size (measured LB/Stern per-iter - base vs analytic)
    enum_check = enum_crosscheck(SMOKE_SIZES[0])
    fit = dict(alpha=alpha, beta=beta, points=pts, base_m64=alpha * 64 ** beta,
               base_m80=alpha * 80 ** beta, base_m96=alpha * 96 ** beta,
               v2_reference="base(m)=3.95*m^3.07 (v2 fit at m=40,60)", enum_crosscheck=enum_check)
    (out_dir / "stage1_base_smoke.json").write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
    return fit


def enum_crosscheck(size):
    """measured (attacker per-iter - base) vs analytic enum, at one smoke size; non-stopping."""
    n, k, w, seed = size["n"], size["k"], size["w"], size["seed"]
    m = n - k
    G, H = v2.make_code(n, k, seed)
    rng = np.random.default_rng(seed ^ 0x2727)
    z = (H @ rng.integers(0, 2, size=n, dtype=np.uint8)) & 1
    out = {}
    base_ops, base_valid = 0, 0
    # measure base + lb + stern(l=8) per-iter over SMOKE_ENUM_VALID valid iters each (non-stopping)
    for kind, l in (("prange", 8), ("lb", 8), ("stern", 8)):
        ops, valid = 0, 0
        r = np.random.default_rng(seed ^ (hash_kind(kind)))
        while valid < SMOKE_ENUM_VALID:
            fb, vit, rf, o, wit = v2.attacker_run(kind, H, z, n, k, w, w, 1, r, P, l)
            ops += o; valid += vit
        out[kind] = ops / valid
    enum_lb_meas = out["lb"] - out["prange"]; enum_st_meas = out["stern"] - out["prange"]
    return dict(m=m, base_probe=out["prange"], enum_lb_measured=enum_lb_meas,
                enum_lb_analytic=v2.enum_lb(k, m, P), enum_stern_measured=enum_st_meas,
                enum_stern_analytic=v2.enum_stern(k, m, 8, P))


def hash_kind(kind):  # small fixed offsets (NOT python hash())
    return {"prange": 11, "lb": 22, "stern": 33}[kind]


def fit_base():
    """recompute the base(m) power-law fit (self-contained for --precal)."""
    pts = [(s["n"] - s["k"], base_probe(s["n"], s["k"], s["seed"], SMOKE_BASE_VALID)[0]) for s in SMOKE_SIZES]
    (m1, b1), (m2, b2) = pts
    beta = math.log(b2 / b1) / math.log(m2 / m1)
    alpha = b1 / m1 ** beta
    return alpha, beta, pts


# ---- empirical pre-calibration: direct per-iteration success rate --------------

def wilson95(succ, n):
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    phat = succ / n; denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def precal_success_rate(kind, H, targets_z, n, k, w, tau, valid_total, rng, l):
    """Round-robin per-iteration success sampling over the throwaway targets (non-stopping:
    max_B=1 per call gives exactly one valid iteration with a success/fail verdict)."""
    m = n - k
    n_per = math.ceil(valid_total / len(targets_z))
    per_target, succ, valid, ops = [], 0, 0, 0
    for tz in targets_z:
        s_t = 0
        for _ in range(n_per):
            fb, vit, rf, o, wit = v2.attacker_run(kind, H, tz, n, k, w, tau, 1, rng, P, l)
            valid += vit; ops += o
            if fb is not None:
                # verify the witness from public data only
                if bool(np.array_equal((H @ wit) & 1, tz)) and int(wit.sum()) <= tau:
                    s_t += 1
        per_target.append(s_t); succ += s_t
    p_emp = succ / valid if valid else 0.0
    # median-implied N (robust to the heavy tail): 1/median(per-target p). For heavy-tailed
    # Stern this is < N_empirical=1/mean(p), so the frozen MEDIAN C will land below the
    # mean-based locked prediction — the gap quantifies the tail (logged, not a void).
    per_target_p = sorted(s / n_per for s in per_target)
    mid = len(per_target_p) // 2
    median_p = per_target_p[mid] if len(per_target_p) % 2 else 0.5 * (per_target_p[mid - 1] + per_target_p[mid])
    return dict(successes=succ, valid_iters=valid, ops=ops, per_target=per_target,
                p_empirical=p_emp, wilson95=wilson95(succ, valid),
                N_empirical=(1 / p_emp if p_emp > 0 else None),
                N_median_implied=(1 / median_p if median_p > 0 else None),
                heterogeneity_succ_min_max=[min(per_target), max(per_target)] if per_target else None)


def select_stern_l(rung_cfg, alpha, beta, stern_by_l):
    """deterministic l selection: keep analytic l* if within 10% of best empirical C, else
    lowest empirical C; exact ties -> smaller l."""
    n, k, w = rung_cfg["n"], rung_cfg["k"], rung_cfg["w"]; m = n - k
    l_star = analytic_l_star(n, k, w, rung_cfg["l_candidates"], alpha * m ** beta)
    cand = []
    for l in rung_cfg["l_candidates"]:
        r = stern_by_l[l]
        if r["p_empirical"] <= 0:
            continue
        per = alpha * m ** beta + v2.enum_stern(k, m, l, P)
        c = (1 / r["p_empirical"]) * LN2 * per
        cand.append((c, l, per))
    if not cand:
        return None, l_star, []
    cand.sort(key=lambda x: (x[0], x[1]))  # lowest C, ties -> smaller l
    best_c, best_l, _ = cand[0]
    c_lstar = next((c for c, l, _ in cand if l == l_star), None)
    chosen = l_star if (c_lstar is not None and c_lstar <= 1.10 * best_c) else best_l
    return chosen, l_star, [{"l": l, "C_ops_50pct": c} for c, l, _ in sorted(cand, key=lambda x: x[1])]


def run_precal(out_dir, lock_out):
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha, beta, base_pts = fit_base()
    lock = {"schema": "pvnp-certificate-syndrome-v3-prediction-lock",
            "stage": "stage-2 prediction lock", "base_fit": dict(alpha=alpha, beta=beta, points=base_pts),
            "T_pre": T_PRE, "succ_budget": SUCC_BUDGET, "rungs": {}}
    report = {"schema": "pvnp-certificate-syndrome-v3-precal", "rungs": {}}
    for rid, cfg in RUNGS.items():
        n, k, w, tau = cfg["n"], cfg["k"], cfg["w"], cfg["tau"]; m = n - k
        G, H = v2.make_code(n, k, cfg["code_seed"])
        pre = v2.sample_frozen_manifest(G, H, n, k, w, T_PRE, cfg["precal_target_seed"])
        tz = [t["z"] for t in pre]
        # LB
        p_lb_an = v2.p_success("lb", n, k, w, P)
        lb_budget = math.ceil(SUCC_BUDGET / p_lb_an)
        lb = precal_success_rate("lb", H, tz, n, k, w, tau, lb_budget,
                                 np.random.default_rng(cfg["precal_target_seed"] ^ 0xAA), 8)
        per_lb = alpha * m ** beta + v2.enum_lb(k, m, P)
        C_lb = (1 / lb["p_empirical"]) * LN2 * per_lb if lb["p_empirical"] > 0 else None
        # Stern over l candidates
        stern_by_l = {}
        for l in cfg["l_candidates"]:
            p_st_an = v2.p_success("stern", n, k, w, P, l)
            st_budget = math.ceil(SUCC_BUDGET / p_st_an) if p_st_an > 0 else lb_budget
            stern_by_l[l] = precal_success_rate("stern", H, tz, n, k, w, tau, st_budget,
                                                np.random.default_rng(cfg["precal_target_seed"] ^ (0xB0 + l)), l)
        chosen_l, l_star, l_table = select_stern_l(cfg, alpha, beta, stern_by_l)
        st = stern_by_l[chosen_l]
        per_st = alpha * m ** beta + v2.enum_stern(k, m, chosen_l, P)
        C_st = (1 / st["p_empirical"]) * LN2 * per_st if st["p_empirical"] > 0 else None
        # Prange formula (no enum; its own base probe)
        N_pr = 1 / v2.p_success("prange", n, k, w)
        C_pr = N_pr * LN2 * (alpha * m ** beta)
        # insufficiency
        insufficient = (lb["successes"] < T_PRE) or (st["successes"] < T_PRE)
        # analytic references + optimism factor (analytic/empirical p)
        opt_lb = (lb["p_empirical"] / p_lb_an) if (p_lb_an > 0 and lb["p_empirical"] > 0) else None
        p_st_an_chosen = v2.p_success("stern", n, k, w, P, chosen_l)
        opt_st = (st["p_empirical"] / p_st_an_chosen) if (p_st_an_chosen > 0 and st["p_empirical"] > 0) else None
        lock["rungs"][str(rid)] = {
            "regime": dict(n=n, k=k, w=w, tau=tau, code_seed=cfg["code_seed"],
                           target_seed=cfg["target_seed"], precal_target_seed=cfg["precal_target_seed"]),
            "precal_insufficient": insufficient,
            "lb": {"p_empirical": lb["p_empirical"], "wilson95": lb["wilson95"], "successes": lb["successes"],
                   "valid_iters": lb["valid_iters"], "per_iter_ops": per_lb, "C_ops_50pct": C_lb,
                   "p_analytic": p_lb_an, "C_analytic": (1 / p_lb_an) * LN2 * per_lb,
                   "p_emp_over_p_analytic": opt_lb, "N_empirical_mean": lb["N_empirical"],
                   "N_median_implied": lb["N_median_implied"], "heterogeneity": lb["per_target"]},
            "stern": {"selected_l": chosen_l, "analytic_l_star": l_star, "l_table": l_table,
                      "p_empirical": st["p_empirical"], "wilson95": st["wilson95"], "successes": st["successes"],
                      "valid_iters": st["valid_iters"], "per_iter_ops": per_st, "C_ops_50pct": C_st,
                      "p_analytic": p_st_an_chosen, "C_analytic": (1 / p_st_an_chosen) * LN2 * per_st,
                      "p_emp_over_p_analytic": opt_st, "N_empirical_mean": st["N_empirical"],
                      "N_median_implied": st["N_median_implied"],
                      "C_ops_50pct_median_implied": ((st["N_median_implied"] * LN2 * per_st) if st["N_median_implied"] else None),
                      "heterogeneity": st["per_target"]},
            "prange_formula": {"N_analytic": N_pr, "C_ops_50pct_formula": C_pr,
                               "note": "FORMULA prediction (Prange analytic validated in v2); never a measured C"},
            "C_best_predicted": min(x for x in (C_lb, C_st) if x) if (C_lb or C_st) else None,
            "verifier_ops": v2.verifier_ops(n, k),
            "tolerance": "measured C within factor 2 of empirical C_ops_50pct",
        }
        report["rungs"][str(rid)] = {"lb": lb, "stern_by_l": {str(l): stern_by_l[l] for l in cfg["l_candidates"]},
                                     "selected_l": chosen_l, "insufficient": insufficient}
    (out_dir / "precal_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(lock_out).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock, report


# ---- frozen rung scoring (LB + Stern; Prange formula) -------------------------

def run_frozen_rung(rung, lock, out_dir, git_sha):
    cfg = RUNGS[rung]
    n, k, w, tau = cfg["n"], cfg["k"], cfg["w"], cfg["tau"]; m = n - k
    rlock = lock["rungs"][str(rung)]
    if rlock.get("precal_insufficient"):
        raise SystemExit(f"rung {rung}: precal_insufficient in the lock — frozen scoring blocked.")
    chosen_l = rlock["stern"]["selected_l"]
    out_dir.mkdir(parents=True, exist_ok=True)
    G, H = v2.make_code(n, k, cfg["code_seed"])
    code_valid = int(((G @ H.T) & 1).sum()) == 0
    targets = v2.sample_frozen_manifest(G, H, n, k, w, T, cfg["target_seed"])
    labels_wt_ok = all(t["wt_e"] == w for t in targets)
    manifest = {"schema": "pvnp-certificate-syndrome-v3-target-manifest", "rung": rung,
                "regime": dict(n=n, k=k, w=w, tau=tau, code_seed=cfg["code_seed"], target_seed=cfg["target_seed"], T=T),
                "attacker_visible": [{"id": t["id"], "z": t["z"].tolist()} for t in targets],
                "labels_only_scoring_fields": [{"id": t["id"], "s": t["s"].tolist(), "e": t["e"].tolist(), "wt_e": t["wt_e"]} for t in targets],
                "note": "attackers read ONLY attacker_visible[].z."}
    mbytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out_dir / "target_manifest.json").write_bytes(mbytes)
    msha = hashlib.sha256(mbytes).hexdigest()
    (out_dir / "verifier_access_declaration.json").write_text(json.dumps(
        {"schema": "pvnp-certificate-syndrome-v3-access", "attacker_input": "z only",
         "scoring_only": ["s", "e", "wt_e"], "target_manifest_sha256": msha,
         "manifest_emitted_before_attackers": True}, indent=2) + "\n", encoding="utf-8")
    vops = v2.verifier_ops(n, k)
    # max_B: generous vs empirical N (lock), capped
    def max_B_for(kind):
        Nemp = rlock[kind]["C_ops_50pct"] / rlock[kind]["per_iter_ops"] / LN2
        return int(max(50, min(20 * Nemp, 200000)))
    per_attacker = {}
    for kind, l in (("lb", 8), ("stern", chosen_l)):
        rng = np.random.default_rng(cfg["target_seed"] ^ (0xC0 + (0 if kind == "lb" else chosen_l)))
        rows, wsamples = [], []
        for t in targets:
            fb, vit, rf, ops, wit = v2.attacker_run(kind, H, t["z"], n, k, w, tau, max_B_for(kind), rng, P, l)
            wok = None
            if wit is not None:
                wok = bool(np.array_equal((H @ wit) & 1, t["z"])) and int(wit.sum()) <= tau
                if len(wsamples) < 6:
                    wsamples.append({"id": t["id"], "He_eq_z": bool(np.array_equal((H @ wit) & 1, t["z"])),
                                     "wt": int(wit.sum()), "tau": tau, "valid": wok})
            rows.append({"id": t["id"], "first_B": fb, "ops_at_success": (ops if fb is not None else None),
                         "valid_iters": vit, "rank_fail": rf, "witness_ok": wok})
        succ = [r for r in rows if r["first_B"] is not None]
        c50 = v2._c50_ops(rows, T)
        per_attacker[kind] = {
            "selected_l": (chosen_l if kind == "stern" else None), "n_success": len(succ), "n_censored": T - len(succ),
            "C_ops_50pct_measured": c50, "C_ops_50pct_predicted": rlock[kind]["C_ops_50pct"],
            "ratio": (c50 / rlock[kind]["C_ops_50pct"]) if (c50 and rlock[kind]["C_ops_50pct"]) else None,
            "valid_iters_total": sum(r["valid_iters"] for r in rows), "rank_fail_total": sum(r["rank_fail"] for r in rows),
            "ops_total": sum((r["ops_at_success"] or 0) for r in rows),
            "witness_all_valid": all(r["witness_ok"] for r in succ) if succ else None, "witness_samples": wsamples}
    Cm = {kind: per_attacker[kind]["C_ops_50pct_measured"] for kind in ("lb", "stern")}
    C_pr_formula = rlock["prange_formula"]["C_ops_50pct_formula"]
    C_best = min(x for x in Cm.values() if x) if any(Cm.values()) else None
    C_best_src = ("stern" if (Cm["stern"] and (not Cm["lb"] or Cm["stern"] <= Cm["lb"])) else "lb") if C_best else None
    st_lb = (Cm["stern"] / Cm["lb"]) if (Cm["stern"] and Cm["lb"]) else None
    within = {kind: (0.5 <= per_attacker[kind]["ratio"] <= 2.0) if per_attacker[kind]["ratio"] else None for kind in Cm}
    verifier_cheap = vops < min(x for x in Cm.values() if x) if any(Cm.values()) else False
    allwit = all(per_attacker[kind]["witness_all_valid"] for kind in Cm if per_attacker[kind]["n_success"])
    if not code_valid or not labels_wt_ok:
        verdict = "void_run (integrity)"
    elif not verifier_cheap:
        verdict = "6.1_vacuity_or_6.4_overhead"
    elif not allwit:
        verdict = "void_run (invalid witness)"
    elif not all(within.values()):
        verdict = "model_deviation (a measured C off the empirical lock beyond factor 2)"
    else:
        verdict = "rung_scored_within_empirical_lock"

    def wj(name, obj):
        (out_dir / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    wj("manifest.json", {"schema": "pvnp-certificate-syndrome-v3-run-manifest", "rung": rung, "complete": True,
                         "regime": dict(n=n, k=k, w=w, tau=tau, code_seed=cfg["code_seed"], target_seed=cfg["target_seed"], T=T),
                         "stern_selected_l": chosen_l, "git_sha": git_sha, "target_manifest_sha256": msha,
                         "suggested_verdict": verdict, "code_valid_GHt0": code_valid, "labels_wt_ok": labels_wt_ok})
    wj("capacity_ladder.json", {"measured_C_ops": Cm, "prange_formula_C": C_pr_formula, "C_best": C_best,
                                "C_best_source": C_best_src, "lb_stern_ratio_St_over_LB": st_lb,
                                "drop_vs_prange_formula": {kind: (C_pr_formula / Cm[kind] if Cm[kind] else None) for kind in Cm}})
    wj("prediction_vs_measured.json", {"per_attacker": {kind: {"measured": Cm[kind],
                                       "predicted_empirical": per_attacker[kind]["C_ops_50pct_predicted"],
                                       "ratio": per_attacker[kind]["ratio"], "within_factor2": within[kind]} for kind in Cm},
                                       "C_best": C_best, "C_best_source": C_best_src, "St_over_LB": st_lb})
    wj("valid_iteration_audit.json", {kind: {k2: per_attacker[kind][k2] for k2 in
                                      ("valid_iters_total", "rank_fail_total", "ops_total", "n_success", "n_censored")} for kind in Cm})
    wj("witness_validity_audit.json", {kind: {"all_valid": per_attacker[kind]["witness_all_valid"],
                                      "samples": per_attacker[kind]["witness_samples"]} for kind in Cm})
    wj("op_count_report.json", {"verifier_ops_flat": vops, "attacker_C_ops": Cm, "prange_formula_C": C_pr_formula,
                                "find_vs_check_gap": {kind: (Cm[kind] / vops if Cm[kind] else None) for kind in Cm},
                                "verifier_below_all_attackers": verifier_cheap})
    icross = {"note": "per-attacker first_B distribution summary"}
    for kind in Cm:
        icross[kind] = {"n_success": per_attacker[kind]["n_success"], "n_censored": per_attacker[kind]["n_censored"]}
    wj("iteration_crosscheck.json", icross)
    (out_dir / "attacker_ladder_curve.csv").write_text(
        "attacker,C_ops_50pct_measured,C_ops_50pct_predicted,ratio\n" +
        "".join(f"{kind},{Cm[kind]},{per_attacker[kind]['C_ops_50pct_predicted']},{per_attacker[kind]['ratio']}\n" for kind in Cm),
        encoding="utf-8")
    (out_dir / "falsifier_summary.md").write_text(
        f"# v3 rung {rung} [{n},{k}] w={w}\n\n- verdict: **{verdict}**\n"
        f"- measured C(ops)@50%: LB {Cm['lb']}, Stern(l={chosen_l}) {Cm['stern']}\n"
        f"- C_best={C_best} [{C_best_src}]; St/LB={st_lb}; Prange(formula)={C_pr_formula:.3e}\n"
        f"- verifier flat {vops} ops; gap@C_best {C_best/vops if C_best else None}\n"
        f"- within empirical lock (factor2): {within}\n\n"
        f"Boundary: C_best is an UPPER BOUND vs the tested classes (BJMM/MMT lower it = new slate); "
        f"Prange at scale is a FORMULA prediction, not measured. No crypto/P-vs-NP claim; op-count cost.\n",
        encoding="utf-8")
    (out_dir / "README.md").write_text(f"# v3 rung {rung} outputs\nVerdict: **{verdict}**. LB + Stern(l={chosen_l}) "
                                       f"measured; Prange formula. Manifest before attackers (sha {msha[:16]}).\n", encoding="utf-8")
    return {"rung": rung, "verdict": verdict, "Cm": Cm, "C_best": C_best, "C_best_source": C_best_src,
            "St_over_LB": st_lb, "verifier_ops": vops, "within": within, "per_attacker": per_attacker,
            "prange_formula_C": C_pr_formula}


# ---- cross-rung summary -------------------------------------------------------

def run_summarize(root, lock):
    root = Path(root)
    anchor = {"rung": 0, "regime": "[128,64]w12", "C_lb": 8.314e7, "C_stern": 1.423e8,
              "C_best": 8.314e7, "St_over_LB": 1.71, "verifier": 16576, "gap": 5015, "source": "v2 receipt"}
    rungs = [anchor]
    for rid in (1, 2, 3):
        d = root / f"rung{rid}" / "capacity_ladder.json"
        if not d.exists():
            continue
        cap = json.loads(d.read_text(encoding="utf-8"))
        cfg = RUNGS[rid]; vops = v2.verifier_ops(cfg["n"], cfg["k"])
        Cb = cap["C_best"]
        rungs.append({"rung": rid, "regime": f"[{cfg['n']},{cfg['k']}]w{cfg['w']}",
                      "C_lb": cap["measured_C_ops"]["lb"], "C_stern": cap["measured_C_ops"]["stern"],
                      "C_best": Cb, "St_over_LB": cap["lb_stern_ratio_St_over_LB"], "verifier": vops,
                      "gap": (Cb / vops if Cb else None), "source": "measured"})
    # Claim A: gap scaling; Claim B: crossover
    gaps = [r["gap"] for r in rungs if r["gap"]]
    claimA = None
    if len(rungs) >= 4 and all(r["gap"] for r in rungs):
        top, anchor_gap = rungs[-1]["gap"], rungs[0]["gap"]
        interior_ok = all(r["gap"] >= 5 * anchor_gap for r in rungs[1:])
        claimA = "gap_scales_confirmed" if (top >= 10 * anchor_gap and interior_ok) else "gap_scaling_not_confirmed"
    crossed = [r["rung"] for r in rungs if r["St_over_LB"] and r["St_over_LB"] < 1.0]
    claimB = (f"crossover_located_at_rung_{min(crossed)}" if crossed else "crossover_not_reached_report_optimism_curve")
    summary = {"schema": "pvnp-certificate-syndrome-v3-scaling-summary", "rungs": rungs,
               "claim_A": claimA, "claim_B": claimB,
               "St_over_LB_ladder": [(r["rung"], r["St_over_LB"]) for r in rungs],
               "gap_ladder": [(r["rung"], r["gap"]) for r in rungs]}
    (root).mkdir(parents=True, exist_ok=True)
    (root / "scaling_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# v3 scaling ladder", "", "| rung | regime | C_LB | C_Stern | C_best | St/LB | gap | source |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rungs:
        lines.append(f"| {r['rung']} | {r['regime']} | {r['C_lb']:.3e} | {r['C_stern']:.3e} | "
                     f"{r['C_best']:.3e} | {r['St_over_LB']} | {r['gap']:.0f} | {r['source']} |")
    lines += ["", f"**Claim A (gap scales):** {claimA}", f"**Claim B (crossover):** {claimB}"]
    (root / "SCALING_LADDER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-smoke", action="store_true")
    ap.add_argument("--precal", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--rung", type=int)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out", default="results/pvnp/certificate-syndrome-v3")
    ap.add_argument("--lock-out", default="docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json")
    ap.add_argument("--prediction-lock", default="docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json")
    ap.add_argument("--root", default="results/pvnp/certificate-syndrome-v3")
    ap.add_argument("--harness-test", action="store_true", help="tiny plumbing+determinism test")
    args = ap.parse_args()

    if args.stage1_smoke:
        fit = run_stage1_smoke(REPO / args.out)
        print("=== v3 stage-1 base(m) smoke ===")
        for p in fit["points"]:
            print(f"  m={p['m']}: base={p['base']:.3e}  rho={p['rho']:.2f}")
        print(f"  base(m) = {fit['alpha']:.4g} * m^{fit['beta']:.3f}  ->  base(64)={fit['base_m64']:.3e} "
              f"base(80)={fit['base_m80']:.3e} base(96)={fit['base_m96']:.3e}")
        ec = fit["enum_crosscheck"]
        print(f"  enum check m={ec['m']}: LB meas {ec['enum_lb_measured']:.2e}/an {ec['enum_lb_analytic']:.2e}  "
              f"Stern meas {ec['enum_stern_measured']:.2e}/an {ec['enum_stern_analytic']:.2e}")
        return
    if args.harness_test:
        # tiny throwaway rung to validate frozen-rung plumbing + determinism (NOT a real rung)
        global RUNGS, T, T_PRE
        RUNGS = {9: dict(n=64, k=32, w=6, tau=6, code_seed=9001, target_seed=9002, precal_target_seed=9003,
                         l_candidates=[5, 6, 7, 8], Cnw="test")}
        T, T_PRE = 16, 8
        alpha, beta, _ = fit_base()
        lk, _ = run_precal(REPO / "results/pvnp/_certsyn-v3-harness-test/precal",
                           REPO / "results/pvnp/_certsyn-v3-harness-test/lock.json")
        res = run_frozen_rung(9, lk, REPO / "results/pvnp/_certsyn-v3-harness-test/rung9", v2._git_sha())
        print("=== v3 harness-test (tiny [64,32]w6) ===")
        print(f"  verdict={res['verdict']} verifier={res['verifier_ops']} Cm={ {k: f'{v:.2e}' if v else v for k,v in res['Cm'].items()} }")
        print(f"  C_best={res['C_best']} [{res['C_best_source']}] St/LB={res['St_over_LB']} stern_l={lk['rungs']['9']['stern']['selected_l']}")
        return
    if args.precal:
        lock, report = run_precal(REPO / args.out, REPO / args.lock_out if not Path(args.lock_out).is_absolute() else Path(args.lock_out))
        print("=== v3 empirical pre-calibration -> stage-2 lock ===")
        for rid in (1, 2, 3):
            r = lock["rungs"][str(rid)]
            print(f"  rung {rid} {r['regime']['n']},{r['regime']['k']} w{r['regime']['w']}: "
                  f"LB p_emp={r['lb']['p_empirical']:.2e} C={r['lb']['C_ops_50pct']:.3e}  "
                  f"Stern l={r['stern']['selected_l']} p_emp={r['stern']['p_empirical']:.2e} C={r['stern']['C_ops_50pct']:.3e}  "
                  f"C_best={r['C_best_predicted']:.3e}  insufficient={r['precal_insufficient']}")
        return
    if args.frozen:
        if args.rung not in RUNGS:
            raise SystemExit("--frozen requires --rung in {1,2,3}")
        lock = json.loads(Path(args.prediction_lock).read_text(encoding="utf-8"))
        res = run_frozen_rung(args.rung, lock, REPO / args.out, v2._git_sha())
        print(f"=== v3 FROZEN rung {args.rung} ===  verdict: {res['verdict']}")
        for kind in ("lb", "stern"):
            pa = res["per_attacker"][kind]
            print(f"  {kind:6}: C(ops)@50%={pa['C_ops_50pct_measured']:.3e} pred={pa['C_ops_50pct_predicted']:.3e} "
                  f"ratio={pa['ratio']:.2f} within2x={res['within'][kind]}")
        print(f"  C_best={res['C_best']:.3e} [{res['C_best_source']}] St/LB={res['St_over_LB']:.2f} "
              f"Prange(formula)={res['prange_formula_C']:.3e} verifier={res['verifier_ops']}")
        return
    if args.summarize:
        lock = json.loads(Path(args.prediction_lock).read_text(encoding="utf-8")) if Path(args.prediction_lock).exists() else {}
        s = run_summarize(REPO / args.root, lock)
        print(f"=== v3 scaling summary ===\n  Claim A: {s['claim_A']}\n  Claim B: {s['claim_B']}")
        print(f"  St/LB ladder: {s['St_over_LB_ladder']}")
        return
    print("Pass one of: --stage1-smoke | --precal | --frozen --rung N | --summarize | --harness-test")


if __name__ == "__main__":
    main()
