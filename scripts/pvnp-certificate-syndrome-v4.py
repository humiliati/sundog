#!/usr/bin/env python3
"""Sundog Certificate Problem v4 — median-calibration + rung-2 (R2') resolution.

Stage-1 frozen contract: docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V4_SLATE.md

Closes the two v3 loose ends:
  (1) prediction method: lock the DIRECTLY-MEASURED censored median ops-to-first-success
      (the exact analog of the frozen C@50%), replacing v3's mean-based (1/p)*ln2*per_iter.
  (2) rung-2 resolution: re-measure [160,80]w16 (same code as v3 rung-2, fresh targets) with
      fixed Stern l in {8,9}, C_Stern=min(l8,l9), vs LB -> Stern_wins | LB_wins.

Reuses the v2/v3-validated GF(2) core, attackers, bundle shape (loaded via importlib).
Modes (PowerShell contract; $env:PYTHONHASHSEED="0"):
  --stage1-smoke   base(m) probe spanning the v4 m-range (inline if <10min)
  --median-precal  run-to-first-success censored-median per regime/variant -> prediction_lock_v4.json (STAGED)
  --validate-v3    median lock vs v3's measured rung-1/rung-3 C (zero frozen-scoring)
  --frozen --regime r2prime   R2' frozen scoring: LB + Stern l8 + l9 (OPERATOR-GATED)
  --summarize      method_validated + rung2 outcome -> V4_SUMMARY.md
  --harness-test   tiny plumbing + determinism
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
_s3 = importlib.util.spec_from_file_location("certsyn_v3", REPO / "scripts" / "pvnp-certificate-syndrome-v3.py")
v3 = importlib.util.module_from_spec(_s3); _s3.loader.exec_module(v3)
v2 = v3.v2

LN2 = math.log(2)
P = 2
T = 64           # frozen-scoring targets
T_PRE = 48       # median pre-cal throwaway targets (up from v3's 16)
C_MULT = 12      # precal cap = ceil(C_MULT * N_analytic)
FROZEN_MULT = 15  # frozen max_B = ceil(FROZEN_MULT * locked median_first_B)

# R2' = the rung-2 retest (same code as v3 rung-2, fresh targets). Validation regimes re-run
# the median pre-cal on v3's rung-1/rung-3 regimes to check against v3's measured C.
REGIMES = {
    "r2prime": dict(n=160, k=80, w=16, tau=16, code_seed=20263201, target_seed=20264202,
                    precal_target_seed=20264203, variants=[("lb", None), ("stern", 8), ("stern", 9)],
                    role="rung-2 retest (frozen)"),
    "v3r1": dict(n=128, k=64, w=16, tau=16, code_seed=20263101, target_seed=None,
                 precal_target_seed=20265103, variants=[("lb", None), ("stern", 7)],
                 role="v3 rung-1 method validation (precal only)"),
    "v3r3": dict(n=192, k=96, w=18, tau=18, code_seed=20263301, target_seed=None,
                 precal_target_seed=20265303, variants=[("lb", None), ("stern", 9)],
                 role="v3 rung-3 method validation (precal only)"),
}
# v3's MEASURED frozen C@50% (from receipt 2026-06-06_certificate_syndrome_v3.md) — the
# ground truth the v4 median lock must predict within factor 2.
V3_MEASURED = {
    "v3r1": {"lb": 1.351e8, "stern_l7": 1.016e8},
    "v3r3": {"lb": 1.076e10, "stern_l9": 8.136e9},
}


def variant_key(kind, l):
    return kind if kind == "lb" else f"stern_l{l}"


def code_digest(G, H):
    h = hashlib.sha256()
    h.update(G.tobytes()); h.update(b"|"); h.update(H.tobytes())
    return h.hexdigest()


def per_iter_for(kind, n, k, l, alpha, beta):
    m = n - k; base = alpha * m ** beta
    if kind == "lb":
        return base + v2.enum_lb(k, m, P)
    return base + v2.enum_stern(k, m, l, P)


# ---- the v4 prediction mechanism: censored median ops-to-first-success ----------

def median_precal(kind, H, targets_z, n, k, w, tau, cap, rng, l):
    """Run each throwaway target to first success (capped); the locked C@50% is the censored
    median of ops-to-first-success. Censored (no success within cap) sort as +inf."""
    succ_ops, finite, censored, ops_total = [], [], 0, 0
    for tz in targets_z:
        fb, vit, rf, ops, wit = v2.attacker_run(kind, H, tz, n, k, w, tau, cap, rng, P, l)
        ops_total += ops
        if fb is not None and bool(np.array_equal((H @ wit) & 1, tz)) and int(wit.sum()) <= tau:
            succ_ops.append(ops); finite.append(ops)
        else:
            succ_ops.append(math.inf); censored += 1
    s = sorted(succ_ops); Tn = len(targets_z)
    lo, hi = s[Tn // 2 - 1], s[Tn // 2]  # for T_pre=48: 24th & 25th order statistics
    insufficient = math.isinf(lo) or math.isinf(hi)
    median_C = None if insufficient else 0.5 * (lo + hi)
    return dict(median_C=median_C, insufficient=insufficient, censored=censored,
                finite_count=len(finite), ops_total=ops_total,
                mean_success_ops=(sum(finite) / len(finite)) if finite else None,
                per_target_ops=[(None if math.isinf(o) else int(o)) for o in succ_ops])


def run_one_regime_precal(rid, cfg, alpha, beta):
    n, k, w, tau = cfg["n"], cfg["k"], cfg["w"], cfg["tau"]
    G, H = v2.make_code(n, k, cfg["code_seed"])
    dig = code_digest(G, H)
    pre = v2.sample_frozen_manifest(G, H, n, k, w, T_PRE, cfg["precal_target_seed"])
    tz = [t["z"] for t in pre]
    out = {"regime": dict(n=n, k=k, w=w, tau=tau, code_seed=cfg["code_seed"],
                          precal_target_seed=cfg["precal_target_seed"]), "code_digest": dig, "variants": {}}
    for kind, l in cfg["variants"]:
        pa = v2.p_success(kind, n, k, w, P, l if l else 8)
        Na = 1.0 / pa
        cap = math.ceil(C_MULT * Na)
        rng = np.random.default_rng(cfg["precal_target_seed"] ^ (0xA0 if kind == "lb" else (0xB0 + l)))
        r = median_precal(kind, H, tz, n, k, w, tau, cap, rng, l if l else 8)
        per = per_iter_for(kind, n, k, l, alpha, beta)
        out["variants"][variant_key(kind, l)] = {
            "kind": kind, "l": l, "N_analytic": Na, "cap": cap, "per_iter_ops": per,
            "median_C_ops_50pct": r["median_C"], "mean_success_ops_ref": r["mean_success_ops"],
            "censored": r["censored"], "finite_count": r["finite_count"],
            "precal_insufficient": r["insufficient"], "per_target_ops": r["per_target_ops"],
            "median_first_B": (r["median_C"] / per) if r["median_C"] else None,
        }
    return out


def run_median_precal(out_dir, lock_out):
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha, beta, base_pts = v3.fit_base()
    lock = {"schema": "pvnp-certificate-syndrome-v4-prediction-lock", "stage": "stage-2",
            "method": "censored-median ops-to-first-success (locked); mean is diagnostic only",
            "T_pre": T_PRE, "C_MULT": C_MULT, "base_fit": dict(alpha=alpha, beta=beta, points=base_pts),
            "regimes": {}}
    report = {"schema": "pvnp-certificate-syndrome-v4-precal-report", "regimes": {}}
    for rid, cfg in REGIMES.items():
        res = run_one_regime_precal(rid, cfg, alpha, beta)
        lock["regimes"][rid] = res
        report["regimes"][rid] = {"role": cfg["role"], "code_digest": res["code_digest"],
                                  "variants": {vk: {"median_C": v["median_C_ops_50pct"], "censored": v["censored"],
                                                    "insufficient": v["precal_insufficient"]} for vk, v in res["variants"].items()}}
    out_dir.joinpath("precal_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(lock_out).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock, report


# ---- method validation: median lock vs v3 measured -----------------------------

def run_validate_v3(lock, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, all_ok = [], True
    for rid in ("v3r1", "v3r3"):
        for vk, meas in V3_MEASURED[rid].items():
            pred = lock["regimes"][rid]["variants"][vk]["median_C_ops_50pct"]
            ratio = (pred / meas) if (pred and meas) else None
            ok = (0.5 <= ratio <= 2.0) if ratio else False
            all_ok = all_ok and ok
            rows.append({"regime": rid, "variant": vk, "v4_median_pred": pred, "v3_measured": meas,
                         "pred_over_measured": ratio, "within_factor2": ok})
    # NOTE: this checks ONLY the v3 ground-truth regimes (an in-sample retrofit). The slate's
    # reserved word "method_validated" requires ALL points incl. the fresh R2' retest; that
    # all-points verdict is computed in run_summarize. Emit a scoped label here so the reserved
    # word is not minted from an in-sample-only check.
    res = {"schema": "pvnp-certificate-syndrome-v4-method-validation", "scope": "v3 ground-truth rungs 1/3 only (in-sample retrofit)",
           "all_within_factor2": all_ok,
           "verdict": "v3_ground_truth_validated" if all_ok else "v3_ground_truth_off", "points": rows}
    (out_dir / "method_validation.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


# ---- R2' frozen scoring --------------------------------------------------------

def run_frozen_r2prime(lock, out_dir, git_sha):
    cfg = REGIMES["r2prime"]; rid = "r2prime"
    n, k, w, tau = cfg["n"], cfg["k"], cfg["w"], cfg["tau"]; m = n - k
    rlock = lock["regimes"][rid]
    for vk, v in rlock["variants"].items():
        if v["precal_insufficient"]:
            raise SystemExit(f"R2' {vk}: precal_insufficient in the lock — frozen scoring blocked.")
    out_dir.mkdir(parents=True, exist_ok=True)
    G, H = v2.make_code(n, k, cfg["code_seed"])
    dig = code_digest(G, H)
    code_id_ok = (dig == rlock["code_digest"])
    code_valid = int(((G @ H.T) & 1).sum()) == 0
    targets = v2.sample_frozen_manifest(G, H, n, k, w, T, cfg["target_seed"])
    labels_wt_ok = all(t["wt_e"] == w for t in targets)
    manifest = {"schema": "pvnp-certificate-syndrome-v4-target-manifest", "regime": "r2prime",
                "params": dict(n=n, k=k, w=w, tau=tau, code_seed=cfg["code_seed"], target_seed=cfg["target_seed"], T=T),
                "attacker_visible": [{"id": t["id"], "z": t["z"].tolist()} for t in targets],
                "labels_only_scoring_fields": [{"id": t["id"], "s": t["s"].tolist(), "e": t["e"].tolist(), "wt_e": t["wt_e"]} for t in targets],
                "note": "attackers read ONLY attacker_visible[].z."}
    mbytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out_dir / "target_manifest.json").write_bytes(mbytes)
    msha = hashlib.sha256(mbytes).hexdigest()
    (out_dir / "verifier_access_declaration.json").write_text(json.dumps(
        {"schema": "pvnp-certificate-syndrome-v4-access", "attacker_input": "z only",
         "scoring_only": ["s", "e", "wt_e"], "target_manifest_sha256": msha,
         "manifest_emitted_before_attackers": True}, indent=2) + "\n", encoding="utf-8")
    vops = v2.verifier_ops(n, k)

    per_variant = {}
    for kind, l in cfg["variants"]:
        vk = variant_key(kind, l)
        lv = rlock["variants"][vk]
        max_B = int(max(50, math.ceil(FROZEN_MULT * (lv["median_first_B"] or lv["cap"]))))
        rng = np.random.default_rng(cfg["target_seed"] ^ (0xA0 if kind == "lb" else (0xB0 + l)))
        rows, wsamples = [], []
        for t in targets:
            fb, vit, rf, ops, wit = v2.attacker_run(kind, H, t["z"], n, k, w, tau, max_B, rng, P, l if l else 8)
            wok = None
            if wit is not None:
                wok = bool(np.array_equal((H @ wit) & 1, t["z"])) and int(wit.sum()) <= tau
                if len(wsamples) < 6:
                    wsamples.append({"id": t["id"], "wt": int(wit.sum()), "tau": tau, "valid": wok})
            rows.append({"id": t["id"], "first_B": fb, "ops_at_success": (ops if fb is not None else None),
                         "valid_iters": vit, "rank_fail": rf, "witness_ok": wok})
        succ = [r for r in rows if r["first_B"] is not None]
        c50 = v2._c50_ops(rows, T)
        pred = lv["median_C_ops_50pct"]
        per_variant[vk] = {"kind": kind, "l": l, "max_B": max_B, "n_success": len(succ), "n_censored": T - len(succ),
                           "C_ops_50pct_measured": c50, "C_ops_50pct_predicted": pred,
                           "ratio": (c50 / pred) if (c50 and pred) else None,
                           "within_factor2": ((0.5 <= c50 / pred <= 2.0) if (c50 and pred) else None),
                           "valid_iters_total": sum(r["valid_iters"] for r in rows),
                           "rank_fail_total": sum(r["rank_fail"] for r in rows),
                           "witness_all_valid": all(r["witness_ok"] for r in succ) if succ else None,
                           "witness_samples": wsamples}
    C_lb = per_variant["lb"]["C_ops_50pct_measured"]
    C_st8 = per_variant["stern_l8"]["C_ops_50pct_measured"]
    C_st9 = per_variant["stern_l9"]["C_ops_50pct_measured"]
    C_stern = min(x for x in (C_st8, C_st9) if x) if (C_st8 or C_st9) else None
    best_l = 8 if (C_st8 and (not C_st9 or C_st8 <= C_st9)) else 9
    outcome = None
    if C_stern and C_lb:
        outcome = "Stern_wins" if C_stern < C_lb else "LB_wins"
    st_lb = (C_stern / C_lb) if (C_stern and C_lb) else None
    verifier_cheap = vops < min(x for x in (C_lb, C_st8, C_st9) if x) if any((C_lb, C_st8, C_st9)) else False
    allwit = all(per_variant[vk]["witness_all_valid"] for vk in per_variant if per_variant[vk]["n_success"])
    within_all = all(per_variant[vk]["within_factor2"] for vk in per_variant if per_variant[vk]["within_factor2"] is not None)
    if not code_valid or not labels_wt_ok or not code_id_ok:
        verdict = "void_run (integrity/code-identity)"
    elif not verifier_cheap:
        verdict = "6.1_vacuity_or_6.4_overhead"
    elif not allwit:
        verdict = "void_run (invalid witness)"
    elif not within_all:
        verdict = f"model_deviation (a measured C off the median lock beyond factor 2); rung2 outcome={outcome}"
    else:
        verdict = f"r2prime_resolved: {outcome}"

    def wj(name, obj):
        (out_dir / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    wj("code_identity.json", {"precal_code_digest": rlock["code_digest"], "frozen_code_digest": dig,
                              "match": code_id_ok, "note": "R2' precal + frozen must regenerate the same (G,H)"})
    wj("manifest.json", {"schema": "pvnp-certificate-syndrome-v4-run-manifest", "regime": "r2prime", "complete": True,
                         "params": dict(n=n, k=k, w=w, tau=tau, code_seed=cfg["code_seed"], target_seed=cfg["target_seed"], T=T),
                         "git_sha": git_sha, "target_manifest_sha256": msha, "code_digest": dig,
                         "suggested_verdict": verdict, "code_valid_GHt0": code_valid, "labels_wt_ok": labels_wt_ok})
    wj("capacity_ladder.json", {"measured_C_ops": {"lb": C_lb, "stern_l8": C_st8, "stern_l9": C_st9},
                                "C_Stern_min": C_stern, "best_stern_l": best_l, "C_LB": C_lb,
                                "St_over_LB": st_lb, "outcome": outcome, "verifier_ops": vops})
    wj("prediction_vs_measured.json", {vk: {"measured": per_variant[vk]["C_ops_50pct_measured"],
                                            "predicted_median": per_variant[vk]["C_ops_50pct_predicted"],
                                            "ratio": per_variant[vk]["ratio"], "within_factor2": per_variant[vk]["within_factor2"]}
                                       for vk in per_variant})
    wj("valid_iteration_audit.json", {vk: {kk: per_variant[vk][kk] for kk in
                                      ("valid_iters_total", "rank_fail_total", "n_success", "n_censored", "max_B")} for vk in per_variant})
    wj("witness_validity_audit.json", {vk: {"all_valid": per_variant[vk]["witness_all_valid"],
                                      "samples": per_variant[vk]["witness_samples"]} for vk in per_variant})
    wj("op_count_report.json", {"verifier_ops_flat": vops, "C_best": C_stern if (outcome == "Stern_wins") else C_lb,
                                "find_vs_check_gap": {vk: (per_variant[vk]["C_ops_50pct_measured"] / vops
                                                          if per_variant[vk]["C_ops_50pct_measured"] else None) for vk in per_variant},
                                "verifier_below_all_attackers": verifier_cheap})
    wj("rung2_resolution.json", {"regime": "[160,80]w16 (same code as v3 rung-2, fresh targets)",
                                 "C_LB": C_lb, "C_Stern_l8": C_st8, "C_Stern_l9": C_st9, "C_Stern_min": C_stern,
                                 "best_stern_l": best_l, "St_over_LB": st_lb, "outcome": outcome,
                                 "v3_rung2_was": "LB_wins (St/LB=1.96) with edge l=10 + heavy-tail precal",
                                 "interpretation": ("crossover monotone in w; v3 rung-2 LB-win was the l=10 handicap"
                                                    if outcome == "Stern_wins" else
                                                    "[160,80]w16 genuinely favors LB; LB<->Stern non-monotone in (n,w)")})
    (out_dir / "falsifier_summary.md").write_text(
        f"# v4 R2' [160,80]w16 retest\n\n- verdict: **{verdict}**\n"
        f"- code identity (precal==frozen (G,H)): {code_id_ok}\n"
        f"- C_LB={C_lb} ; C_Stern(l8)={C_st8} ; C_Stern(l9)={C_st9} ; C_Stern=min={C_stern} (best l={best_l})\n"
        f"- St/LB={st_lb} -> **{outcome}**\n- verifier flat {vops}; within median lock (factor2): {within_all}\n\n"
        f"Boundary: C_best upper bound vs LB/Stern (BJMM/MMT=new slate); no crypto/P-vs-NP; op-count cost.\n",
        encoding="utf-8")
    (out_dir / "README.md").write_text(f"# v4 R2' outputs — verdict **{verdict}**\nLB + Stern l8 + l9 on the same "
                                       f"[160,80]w16 code as v3 rung-2, fresh targets. code_identity={code_id_ok}.\n", encoding="utf-8")
    return {"verdict": verdict, "outcome": outcome, "C_lb": C_lb, "C_st8": C_st8, "C_st9": C_st9,
            "C_stern": C_stern, "best_l": best_l, "St_over_LB": st_lb, "code_id_ok": code_id_ok,
            "within_all": within_all, "per_variant": per_variant, "verifier": vops}


def run_summarize(root, lock):
    root = Path(root)
    mv = json.loads((root / "method-validation" / "method_validation.json").read_text(encoding="utf-8")) if (root / "method-validation" / "method_validation.json").exists() else None
    r2 = json.loads((root / "r2prime" / "rung2_resolution.json").read_text(encoding="utf-8")) if (root / "r2prime" / "rung2_resolution.json").exists() else None
    r2pm = json.loads((root / "r2prime" / "prediction_vs_measured.json").read_text(encoding="utf-8")) if (root / "r2prime" / "prediction_vs_measured.json").exists() else None
    r2mf = json.loads((root / "r2prime" / "manifest.json").read_text(encoding="utf-8")) if (root / "r2prime" / "manifest.json").exists() else None
    r2va = json.loads((root / "r2prime" / "valid_iteration_audit.json").read_text(encoding="utf-8")) if (root / "r2prime" / "valid_iteration_audit.json").exists() else None
    # ALL-POINTS method verdict per the slate's binding definition: v3 retrofit AND fresh R2'.
    r2_within = all(v.get("within_factor2") for v in r2pm.values()) if r2pm else None
    v3_ok = (mv and mv.get("all_within_factor2"))
    all_points = "method_validated" if (v3_ok and r2_within) else "method_still_off"
    # near-tie / model_deviation flags
    st_lb = r2["St_over_LB"] if r2 else None
    near_tie = (st_lb is not None and 0.95 <= st_lb <= 1.05)
    r2_verdict = (r2mf["suggested_verdict"] if r2mf else None)
    summary = {"schema": "pvnp-certificate-syndrome-v4-summary",
               "method_verdict_all_points": all_points,
               "method_verdict_note": "slate's binding all-points def (incl. fresh R2'); R2' Stern l8 missed 2.77x => method_still_off",
               "v3_retrofit_verdict": (mv["verdict"] if mv else "not_run"),
               "v3_retrofit_points": (mv["points"] if mv else None),
               "r2prime": {"frozen_verdict": r2_verdict, "outcome_binary": (r2["outcome"] if r2 else None),
                           "St_over_LB": st_lb, "near_tie": near_tie,
                           "prediction_vs_measured": r2pm, "censoring": ({vk: {"n_censored": r2va[vk]["n_censored"], "max_B": r2va[vk]["max_B"]} for vk in r2va} if r2va else None),
                           "honest_reading": "near-tie (St/LB~0.98 within seed-noise + Stern-flattering censoring asymmetry); "
                                             "dissolves v3's l=10 LB-win artifact DOWN to a dead heat, NOT a clean Stern crossover"}}
    (root).mkdir(parents=True, exist_ok=True)
    (root / "scaling_summary_v4.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# v4 summary (honest; see receipt 2026-06-07_certificate_syndrome_v4.md)", "",
             f"**Method verdict (slate all-points, incl. fresh R2'): {all_points}**",
             f"- v3 ground-truth retrofit: {summary['v3_retrofit_verdict']}" + (" (rungs 1/3, all within factor 2)" if v3_ok else "")]
    if mv:
        for p in mv["points"]:
            lines.append(f"  - {p['regime']}/{p['variant']}: pred {p['v4_median_pred']:.3e} vs measured {p['v3_measured']:.3e} "
                         f"(ratio {p['pred_over_measured']:.2f}, within2x {p['within_factor2']})")
    if r2pm:
        lines.append(f"- fresh R2' (prospective) — median lock vs measured:")
        for vk, v in r2pm.items():
            lines.append(f"  - {vk}: measured {v['measured']:.3e} vs predicted_median {v['predicted_median']:.3e} "
                         f"(ratio {v['ratio']:.2f}, within2x {v['within_factor2']})")
        lines.append(f"  => R2' frozen verdict: **{r2_verdict}**")
    lines += ["", f"**R2' rung-2 resolution (HONEST):** near-tie (St/LB={st_lb:.3f}); binary label `{r2['outcome'] if r2 else '?'}` on a ~2% margin"]
    if r2 and r2va:
        lines.append(f"- C_LB={r2['C_LB']:.3e} (cens {r2va['lb']['n_censored']}/64), C_Stern(l8)={r2['C_Stern_l8']:.3e} "
                     f"(cens {r2va['stern_l8']['n_censored']}/64), C_Stern(l9)={r2['C_Stern_l9']:.3e} (cens {r2va['stern_l9']['n_censored']}/64)")
        lines.append(f"- caveats: 2% margin within seed-noise (median wanders 1.43-2.77x on seed flip) + Stern-flattering "
                     f"censoring asymmetry (LB 0 censored vs Stern 10/13 capped below LB's max_B); model_deviation flagged.")
        lines.append(f"- reading: dissolves v3's dramatic l=10 LB-win (St/LB=1.96) DOWN to a dead heat; NOT a clean Stern crossover.")
    (root / "V4_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-smoke", action="store_true")
    ap.add_argument("--median-precal", action="store_true")
    ap.add_argument("--validate-v3", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--regime", default="r2prime")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--harness-test", action="store_true")
    ap.add_argument("--out", default="results/pvnp/certificate-syndrome-v4")
    ap.add_argument("--lock-out", default="docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V4_PREDICTION_LOCK.json")
    ap.add_argument("--prediction-lock", default="docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V4_PREDICTION_LOCK.json")
    ap.add_argument("--root", default="results/pvnp/certificate-syndrome-v4")
    args = ap.parse_args()

    if args.stage1_smoke:
        fit = v3.run_stage1_smoke(REPO / args.out)
        print(f"v4 base(m) = {fit['alpha']:.4g}*m^{fit['beta']:.3f}  base(64)={fit['base_m64']:.3e} "
              f"base(80)={fit['base_m80']:.3e} base(96)={fit['base_m96']:.3e}")
        return
    if args.harness_test:
        global REGIMES, T, T_PRE, V3_MEASURED
        REGIMES = {"r2prime": dict(n=64, k=32, w=6, tau=6, code_seed=9101, target_seed=9102,
                                   precal_target_seed=9103, variants=[("lb", None), ("stern", 8), ("stern", 9)], role="test")}
        T, T_PRE = 16, 12
        lk, _ = run_median_precal(REPO / "results/pvnp/_certsyn-v4-harness-test/precal",
                                  REPO / "results/pvnp/_certsyn-v4-harness-test/lock.json")
        res = run_frozen_r2prime(lk, REPO / "results/pvnp/_certsyn-v4-harness-test/r2prime", v2._git_sha())
        print(f"v4 harness-test: verdict={res['verdict']} code_id={res['code_id_ok']} "
              f"C_LB={res['C_lb']} C_Stern=min={res['C_stern']} outcome={res['outcome']}")
        return
    if args.median_precal:
        lock, _ = run_median_precal(REPO / args.out, Path(args.lock_out) if Path(args.lock_out).is_absolute() else REPO / args.lock_out)
        print("=== v4 median pre-calibration -> stage-2 lock ===")
        for rid, rc in lock["regimes"].items():
            for vk, v in rc["variants"].items():
                mc = v["median_C_ops_50pct"]
                print(f"  {rid}/{vk}: median_C={mc:.3e} censored={v['censored']}/{T_PRE} insufficient={v['precal_insufficient']}" if mc
                      else f"  {rid}/{vk}: INSUFFICIENT censored={v['censored']}/{T_PRE}")
        return
    if args.validate_v3:
        lock = json.loads(Path(args.prediction_lock).read_text(encoding="utf-8"))
        res = run_validate_v3(lock, REPO / args.out / "method-validation" if not Path(args.out).name == "method-validation" else REPO / args.out)
        print(f"=== v4 method validation: {res['verdict']} ===")
        for p in res["points"]:
            print(f"  {p['regime']}/{p['variant']}: pred {p['v4_median_pred']:.3e} vs v3 {p['v3_measured']:.3e} "
                  f"ratio={p['pred_over_measured']:.2f} within2x={p['within_factor2']}")
        return
    if args.frozen:
        if args.regime != "r2prime":
            raise SystemExit("v4 --frozen only scores --regime r2prime")
        lock = json.loads(Path(args.prediction_lock).read_text(encoding="utf-8"))
        res = run_frozen_r2prime(lock, REPO / args.out, v2._git_sha())
        print(f"=== v4 FROZEN R2' [160,80]w16 ===  verdict: {res['verdict']}")
        print(f"  C_LB={res['C_lb']:.3e}  C_Stern(l8)={res['C_st8']:.3e}  C_Stern(l9)={res['C_st9']:.3e}")
        print(f"  C_Stern=min={res['C_stern']:.3e} (best l={res['best_l']})  St/LB={res['St_over_LB']:.2f}  -> {res['outcome']}")
        print(f"  code_identity={res['code_id_ok']}  within median lock={res['within_all']}  verifier={res['verifier']}")
        return
    if args.summarize:
        lock = json.loads(Path(args.prediction_lock).read_text(encoding="utf-8")) if Path(args.prediction_lock).exists() else {}
        s = run_summarize(REPO / args.root, lock)
        print(f"=== v4 summary ===  method(all-points): {s['method_verdict_all_points']}  "
              f"rung2: {s['r2prime']['outcome_binary']} (near_tie={s['r2prime']['near_tie']}, frozen={s['r2prime']['frozen_verdict']})")
        return
    print("Pass: --stage1-smoke | --median-precal | --validate-v3 | --frozen --regime r2prime | --summarize | --harness-test")


if __name__ == "__main__":
    main()
