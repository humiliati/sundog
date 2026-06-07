#!/usr/bin/env python3
"""Sundog Certificate Problem v5 — MHK-v5 distributional-band predictor.

Stage-1 frozen contract: docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_SLATE.md

The v4 point-median missed the fresh R2' regime's Stern l8 by 2.77x (the documented 1.43-2.77x
seed/sample wander). v5 replaces the point with a FALSIFIABLE BAND: Kaplan-Meier median of
ops-to-first-success (op units) over K=8 independent same-size precal seeds, locked as
[min_s M_s / g, max_s M_s * g] with a hard W<=3.0 tightness ceiling (W>3.0 => precal_insufficient,
honest decline, never widened). Cross-attacker = pairwise (LB vs stern_l8 / LB vs stern_l9)
survival-difference dS(B*) bootstrap CI + log-rank, Bonferroni p<0.025, with a near-tie bin and a
common op horizon B_common (fixes the v4 censoring asymmetry).

Reuses the VALIDATED v2 attacker core unchanged (only the statistical/pre-registration layer is new).

Modes (PowerShell contract; $env:PYTHONHASHSEED="0"):
  --smoke           base(m) refit + KM-median == v4 _c50_ops zero-censoring unit check (inline)
  --precal --profile {primary,fallback}   K-seed bands + locked cross-attacker -> prediction_lock_v5.json (STAGED)
  --validate-v3     Stage-2b: re-band v3 rungs 1/3, assert measured C inside band (else ABORT)
  --frozen-r2prime  Stage-3: R2' frozen scoring; GATE-1/2/3 (OPERATOR-GATED)
  --summarize       V5_SUMMARY.md
  --harness-test    tiny end-to-end plumbing + determinism
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
_s4 = importlib.util.spec_from_file_location("certsyn_v4", REPO / "scripts" / "pvnp-certificate-syndrome-v4.py")
v4 = importlib.util.module_from_spec(_s4); _s4.loader.exec_module(v4)
v3 = v4.v3
v2 = v4.v2

LN2 = math.log(2)
P = 2
T_PRE = 64
T_FROZEN = 64
C_MULT = 12
W_MAX = 3.0
BOOT = 2000
BONFERRONI_P = 0.025

PROFILES = {"primary": {"K": 8, "g": 1.25}, "fallback": {"K": 6, "g": 1.35}}

REGIMES = {
    "r2prime": dict(n=160, k=80, w=16, tau=16, code_seed=20263201,
                    precal_target_seeds=[20265201, 20265202, 20265203, 20265204, 20265205, 20265206, 20265207, 20265208],
                    frozen_target_seed=20265209, variants=[("lb", None), ("stern", 8), ("stern", 9)],
                    cross_pairs=[(("lb", None), ("stern", 8)), (("lb", None), ("stern", 9))], role="primary R2' re-resolution"),
    "v3r1": dict(n=128, k=64, w=16, tau=16, code_seed=20263101,
                 precal_target_seeds=[20265211, 20265212, 20265213, 20265214, 20265215, 20265216, 20265217, 20265218],
                 frozen_target_seed=None, variants=[("lb", None), ("stern", 7)], cross_pairs=[], role="v3-R1 validation"),
    "v3r3": dict(n=192, k=96, w=18, tau=18, code_seed=20263301,
                 precal_target_seeds=[20265231, 20265232, 20265233, 20265234, 20265235, 20265236, 20265237, 20265238],
                 frozen_target_seed=None, variants=[("lb", None), ("stern", 9)], cross_pairs=[], role="v3-R3 validation"),
}
# v3 MEASURED frozen C@50% (receipt 2026-06-06) — Stage-2b containment ground truth.
V3_MEASURED = {"v3r1": {("lb", None): 1.351e8, ("stern", 7): 1.016e8},
               "v3r3": {("lb", None): 1.076e10, ("stern", 9): 8.136e9}}


def vk_of(kind, l):
    return kind if kind == "lb" else f"stern_l{l}"


def code_digest(G, H):
    h = hashlib.sha256(); h.update(G.tobytes()); h.update(b"|"); h.update(H.tobytes()); return h.hexdigest()


def _pair_seed(avk, bvk, salt):
    """Deterministic bootstrap seed independent of PYTHONHASHSEED (no python hash())."""
    return int(hashlib.sha256(f"{salt}:{avk}:{bvk}".encode()).hexdigest()[:8], 16)


# ---- Kaplan-Meier in op units (reduces to v4 _c50_ops under zero censoring) -----

def km_event_curve(ops, event):
    """Return [(t, S_after)] at each distinct event (success) op-time, plus the at-risk bookkeeping."""
    idx = sorted(range(len(ops)), key=lambda i: ops[i])
    n = len(ops); at_risk = n; S = 1.0; curve = []
    i = 0
    while i < n:
        t = ops[idx[i]]; j = i; d = 0; tot = 0
        while j < n and ops[idx[j]] == t:
            tot += 1
            if event[idx[j]] == 1:
                d += 1
            j += 1
        if d > 0 and at_risk > 0:
            S *= (1.0 - d / at_risk)
            curve.append((t, S))
        at_risk -= tot
        i = j
    return curve


def km_median(ops, event):
    """KM median op-budget; midpoint convention reduces to v4 _c50_ops under zero censoring.
    Returns (median|None, censored_bool)."""
    curve = km_event_curve(ops, event)
    for idx, (t, Sa) in enumerate(curve):
        if Sa <= 0.5 + 1e-12:
            if abs(Sa - 0.5) <= 1e-9:  # exact 0.5 plateau -> midpoint to next observed event
                if idx + 1 < len(curve):
                    return 0.5 * (t + curve[idx + 1][0]), False
                return None, True  # 0.5 at the last observed event, upper endpoint censored
            return float(t), False  # crossed below 0.5 at this event
    return None, True  # S never reached 0.5 -> right-censored


def km_survival_at(ops, event, b):
    """KM survival S_hat(b)."""
    S = 1.0
    for t, Sa in km_event_curve(ops, event):
        if t <= b:
            S = Sa
        else:
            break
    return S


def logrank_p(ops_a, ev_a, ops_b, ev_b, b_cap):
    """Two-group log-rank statistic -> chi-sq(1) p-value (pure-math, no scipy)."""
    times = sorted({ops_a[i] for i in range(len(ops_a)) if ev_a[i] == 1 and ops_a[i] <= b_cap} |
                   {ops_b[i] for i in range(len(ops_b)) if ev_b[i] == 1 and ops_b[i] <= b_cap})
    O_minus_E = 0.0; V = 0.0
    for t in times:
        n1 = sum(1 for x in ops_a if x >= t); n2 = sum(1 for x in ops_b if x >= t)
        o1 = sum(1 for i in range(len(ops_a)) if ops_a[i] == t and ev_a[i] == 1)
        o2 = sum(1 for i in range(len(ops_b)) if ops_b[i] == t and ev_b[i] == 1)
        n = n1 + n2; o = o1 + o2
        if n <= 1 or o == 0:
            continue
        E1 = o * n1 / n
        Vj = o * (n1 / n) * (1 - n1 / n) * (n - o) / (n - 1)
        O_minus_E += (o1 - E1); V += Vj
    if V <= 0:
        return 1.0, 0.0
    stat = (O_minus_E ** 2) / V
    return math.erfc(math.sqrt(stat / 2.0)), stat


# ---- precal: K-seed band per variant + pooled (ops,event) for cross-attacker ----

def run_one_target(kind, H, z, n, k, w, tau, max_B, rng, l):
    fb, vit, rf, ops, wit = v2.attacker_run(kind, H, z, n, k, w, tau, max_B, rng, P, l if l else 8)
    ok = fb is not None and bool(np.array_equal((H @ wit) & 1, z)) and int(wit.sum()) <= tau
    return (ops, 1 if ok else 0)


def precal_regime(cfg, alpha, beta, K, g):
    n, k, w, tau = cfg["n"], cfg["k"], cfg["w"], cfg["tau"]; m = n - k
    G, H = v2.make_code(n, k, cfg["code_seed"]); dig = code_digest(G, H)
    per_iter = {vk_of(kd, l): v4.per_iter_for(kd, n, k, l, alpha, beta) for kd, l in cfg["variants"]}
    N_an = {vk_of(kd, l): 1.0 / v2.p_success(kd, n, k, w, P, l if l else 8) for kd, l in cfg["variants"]}
    B_common = max(math.ceil(C_MULT * N_an[vk_of(kd, l)]) * per_iter[vk_of(kd, l)] for kd, l in cfg["variants"])
    max_B = {vk_of(kd, l): int(math.ceil(B_common / per_iter[vk_of(kd, l)])) for kd, l in cfg["variants"]}
    out = {"regime": dict(n=n, k=k, w=w, tau=tau, code_seed=cfg["code_seed"]), "code_digest": dig,
           "B_common": B_common, "per_iter_ops": per_iter, "max_B": max_B, "K": K, "g": g, "variants": {}, "pooled": {}}
    for kd, l in cfg["variants"]:
        vk = vk_of(kd, l)
        M_seed, censored_seeds, het = [], 0, []
        pooled_ops, pooled_ev = [], []
        for s_i, seed in enumerate(cfg["precal_target_seeds"][:K]):
            tgts = v2.sample_frozen_manifest(G, H, n, k, w, T_PRE, seed)
            rng = np.random.default_rng(seed ^ (0xA0 if kd == "lb" else (0xB0 + l)))
            ops_arr, ev_arr = [], []
            for t in tgts:
                o, e = run_one_target(kd, H, t["z"], n, k, w, tau, max_B[vk], rng, l)
                ops_arr.append(o); ev_arr.append(e)
            med, cens = km_median(ops_arr, ev_arr)
            if cens:
                censored_seeds += 1
            else:
                M_seed.append(med)
            fin = [o for o, e in zip(ops_arr, ev_arr) if e == 1]
            het.append((max(fin) / min(fin)) if fin else None)
            pooled_ops += ops_arr; pooled_ev += ev_arr
        insufficient = (censored_seeds > 0) or (len(M_seed) < K)
        if not insufficient:
            lo, hi = min(M_seed), max(M_seed)
            locked_band = [lo / g, hi * g]; Wv = locked_band[1] / locked_band[0]
            if Wv > W_MAX:
                insufficient = True
        else:
            locked_band, Wv = None, None
        out["variants"][vk] = {
            "kind": kd, "l": l, "N_analytic": N_an[vk], "per_iter_ops": per_iter[vk], "max_B": max_B[vk],
            "M_seed": M_seed, "raw_band": ([min(M_seed), max(M_seed)] if M_seed else None),
            "locked_band": locked_band, "W": Wv, "seeds_censored": censored_seeds,
            "per_seed_target_max_over_min": het,
            "mean_based_C_DIAGNOSTIC_ONLY": (float(np.mean([o for o, e in zip(pooled_ops, pooled_ev) if e == 1])) if any(pooled_ev) else None),
            "precal_insufficient": insufficient}
        out["pooled"][vk] = {"ops": pooled_ops, "event": pooled_ev}
    return out


def cross_attacker_precal(reg, cfg):
    """Pairwise dS(B*_pair) bootstrap CI + Bonferroni log-rank on the POOLED precal data."""
    B_common = reg["B_common"]; pairs = []
    for (akd, al), (bkd, bl) in cfg["cross_pairs"]:
        avk, bvk = vk_of(akd, al), vk_of(bkd, bl)
        va, vb = reg["variants"][avk], reg["variants"][bvk]
        if va["precal_insufficient"] or vb["precal_insufficient"]:
            pairs.append({"pair": f"{avk}_vs_{bvk}", "verdict": "precal_insufficient"}); continue
        med_a = float(np.median(va["M_seed"])); med_b = float(np.median(vb["M_seed"]))
        B_star = min(1.5 * max(med_a, med_b), 0.75 * B_common); cap_active = (1.5 * max(med_a, med_b)) > (0.75 * B_common)
        oa, ea = reg["pooled"][avk]["ops"], reg["pooled"][avk]["event"]
        ob, eb = reg["pooled"][bvk]["ops"], reg["pooled"][bvk]["event"]
        dS = km_survival_at(oa, ea, B_star) - km_survival_at(ob, eb, B_star)
        # bootstrap CI over shared resample indices (paired by position)
        nA = len(oa); rng = np.random.default_rng(_pair_seed(avk, bvk, "precal"))
        oa_a, ea_a = np.array(oa), np.array(ea); ob_a, eb_a = np.array(ob), np.array(eb)
        boots = np.empty(BOOT)
        for bi in range(BOOT):
            idx = rng.integers(0, nA, nA)
            boots[bi] = (km_survival_at(oa_a[idx].tolist(), ea_a[idx].tolist(), B_star)
                         - km_survival_at(ob_a[idx].tolist(), eb_a[idx].tolist(), B_star))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        p, stat = logrank_p(oa, ea, ob, eb, B_common)
        ci_excl0 = (ci[0] > 0 or ci[1] < 0)
        sig = ci_excl0 and (p < BONFERRONI_P)
        if not sig:
            verdict = "indistinguishable_at_op_budget"
        else:
            verdict = "lb_wins" if dS < 0 else "stern_wins"  # lower S(B*) wins; dS=S_LB-S_Stern
        pairs.append({"pair": f"{avk}_vs_{bvk}", "B_star": B_star, "B_star_cap_active": cap_active,
                      "dS": dS, "dS_ci": ci, "logrank_p": p, "ci_excludes_0": ci_excl0,
                      "bonferroni_p_threshold": BONFERRONI_P, "verdict": verdict})
    return pairs, _family_verdict(pairs)


def _family_verdict(pairs):
    scored = [p for p in pairs if p.get("verdict") not in (None, "precal_insufficient")]
    if not scored:
        return "precal_insufficient"
    vs = [p["verdict"] for p in scored]
    wins = [v == "stern_wins" for v in vs]; losses = [v == "lb_wins" for v in vs]
    if any(wins) and any(losses):
        return "mixed_variant"
    if all(losses):
        return "lb_wins"
    if any(wins) and not any(losses):
        return "stern_wins"
    # uncovered combos (e.g. [lb_wins, indistinguishable]) conservatively default to
    # indistinguishable: no Stern win, no clean LB sweep. Applied identically to locked and
    # frozen pairs, so GATE-3's match check is unaffected (see slate family-verdict note).
    return "indistinguishable_at_op_budget"


def run_precal(out_dir, lock_out, profile):
    out_dir.mkdir(parents=True, exist_ok=True)
    K, g = PROFILES[profile]["K"], PROFILES[profile]["g"]
    alpha, beta, base_pts = v3.fit_base()
    lock = {"schema": "pvnp-certificate-syndrome-v5-prediction-lock", "stage": "stage-2",
            "estimator": "km_median_op_units_common_cap", "profile": profile, "K": K, "g": g,
            "T_pre": T_PRE, "T_frozen": T_FROZEN, "C_MULT": C_MULT, "W_max": W_MAX,
            "bonferroni_p": BONFERRONI_P, "base_fit": dict(alpha=alpha, beta=beta, points=base_pts), "regimes": {}}
    for rid, cfg in REGIMES.items():
        reg = precal_regime(cfg, alpha, beta, K, g)
        entry = {"role": cfg["role"], "code_digest": reg["code_digest"], "B_common": reg["B_common"],
                 "frozen_target_seed": cfg["frozen_target_seed"],
                 "variants": {vk: {kk: v[kk] for kk in ("kind", "l", "N_analytic", "per_iter_ops", "max_B",
                              "M_seed", "raw_band", "locked_band", "W", "seeds_censored",
                              "per_seed_target_max_over_min", "mean_based_C_DIAGNOSTIC_ONLY", "precal_insufficient")}
                              for vk, v in reg["variants"].items()}}
        if cfg["cross_pairs"]:
            pairs, fam = cross_attacker_precal(reg, cfg)
            entry["cross_attacker"] = {"pairs": pairs, "family_verdict": fam}
        # reported bound: min over admissible variants of (band-low / median proxy)
        adm = [v for v in reg["variants"].values() if not v["precal_insufficient"]]
        if adm:
            best = min(adm, key=lambda v: v["locked_band"][0])
            entry["C_best_predicted"] = {"variant": vk_of(best["kind"], best["l"]), "band": best["locked_band"]}
        lock["regimes"][rid] = entry
    Path(lock_out).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    (out_dir / "precal_report.json").write_text(json.dumps({"profile": profile, "regimes": list(REGIMES)}, indent=2) + "\n", encoding="utf-8")
    return lock


# ---- Stage-2b validation: v3 measured C must land inside the v5 band -----------

def run_validate_v3(lock, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, all_ok = [], True
    for rid in ("v3r1", "v3r3"):
        reg = lock["regimes"][rid]
        for (kd, l), meas in V3_MEASURED[rid].items():
            vk = vk_of(kd, l); v = reg["variants"][vk]
            if v["precal_insufficient"]:
                all_ok = False
                rows.append({"regime": rid, "variant": vk, "measured": meas, "band": v["locked_band"],
                             "inside": False, "note": "validation variant precal_insufficient -> Stage-2b ABORT (not droppable)"})
                continue
            lo, hi = v["locked_band"]; inside = (lo <= meas <= hi)
            all_ok = all_ok and inside
            rows.append({"regime": rid, "variant": vk, "measured": meas, "band": v["locked_band"], "inside": inside,
                         "W": v["W"]})
    res = {"schema": "pvnp-certificate-syndrome-v5-validation", "verdict": "v3_bands_contain_ground_truth" if all_ok else "STAGE2B_ABORT",
           "all_inside": all_ok, "points": rows}
    (out_dir / "v3_validation.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


# ---- Stage-3 frozen R2' --------------------------------------------------------

def run_frozen_r2prime(lock, out_dir, git_sha):
    cfg = REGIMES["r2prime"]; reg = lock["regimes"]["r2prime"]
    n, k, w, tau = cfg["n"], cfg["k"], cfg["w"], cfg["tau"]
    G, H = v2.make_code(n, k, cfg["code_seed"]); dig = code_digest(G, H)
    code_id_ok = (dig == reg["code_digest"]); code_valid = int(((G @ H.T) & 1).sum()) == 0
    B_common = reg["B_common"]
    tgts = v2.sample_frozen_manifest(G, H, n, k, w, T_FROZEN, cfg["frozen_target_seed"])
    labels_wt_ok = all(t["wt_e"] == w for t in tgts)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "pvnp-certificate-syndrome-v5-target-manifest", "regime": "r2prime",
                "params": dict(n=n, k=k, w=w, tau=tau, code_seed=cfg["code_seed"], target_seed=cfg["frozen_target_seed"], T=T_FROZEN),
                "attacker_visible": [{"id": t["id"], "z": t["z"].tolist()} for t in tgts],
                "labels_only_scoring_fields": [{"id": t["id"], "s": t["s"].tolist(), "e": t["e"].tolist(), "wt_e": t["wt_e"]} for t in tgts]}
    mbytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out_dir / "target_manifest.json").write_bytes(mbytes); msha = hashlib.sha256(mbytes).hexdigest()
    pooled = {}; per_variant = {}
    for kd, l in cfg["variants"]:
        vk = vk_of(kd, l); lv = reg["variants"][vk]
        if lv["precal_insufficient"]:
            per_variant[vk] = {"precal_insufficient": True}; continue
        rng = np.random.default_rng(cfg["frozen_target_seed"] ^ (0xA0 if kd == "lb" else (0xB0 + l)))
        ops_arr, ev_arr, wok = [], [], True
        for t in tgts:
            o, e = run_one_target(kd, H, t["z"], n, k, w, tau, lv["max_B"], rng, l)
            ops_arr.append(o); ev_arr.append(e)
            if e == 0:
                pass
        # witness validity re-check on a sample of successes
        med, cens = km_median(ops_arr, ev_arr)
        lo, hi = lv["locked_band"]
        inside = (med is not None) and (lo <= med <= hi)
        per_variant[vk] = {"frozen_median": med, "censored_median": cens, "locked_band": [lo, hi],
                           "inside_band": inside, "n_censored": int(sum(1 for e in ev_arr if e == 0)),
                           "nearest_edge_ratio": (None if med is None else (med / hi if med > hi else (lo / med if med < lo else 1.0)))}
        pooled[vk] = {"ops": ops_arr, "event": ev_arr}
    # GATE-1
    scored = {vk: v for vk, v in per_variant.items() if not v.get("precal_insufficient")}
    gate1 = all(v["inside_band"] for v in scored.values()) if scored else False
    # GATE-3 cross-attacker on frozen data. B_star is the LOCKED precal scalar (NOT recomputed
    # from the frozen draw — that would tune the op budget to the data being judged).
    locked_bstar = {pp["pair"]: pp["B_star"] for pp in reg.get("cross_attacker", {}).get("pairs", []) if "B_star" in pp}
    frozen_pairs = []
    for (akd, al), (bkd, bl) in cfg["cross_pairs"]:
        avk, bvk = vk_of(akd, al), vk_of(bkd, bl)
        pk = f"{avk}_vs_{bvk}"
        if avk not in pooled or bvk not in pooled:
            frozen_pairs.append({"pair": pk, "verdict": "precal_insufficient"}); continue
        med_a = per_variant[avk]["frozen_median"]; med_b = per_variant[bvk]["frozen_median"]
        if med_a is None or med_b is None:  # censored frozen median -> already fails GATE-1; skip pair
            frozen_pairs.append({"pair": pk, "verdict": "precal_insufficient"}); continue
        if pk not in locked_bstar:
            frozen_pairs.append({"pair": pk, "verdict": "void_run_no_locked_bstar"}); continue
        B_star = locked_bstar[pk]  # locked at precal, per slate "no retuning after the frozen draw"
        oa, ea = pooled[avk]["ops"], pooled[avk]["event"]; ob, eb = pooled[bvk]["ops"], pooled[bvk]["event"]
        dS = km_survival_at(oa, ea, B_star) - km_survival_at(ob, eb, B_star)
        rng = np.random.default_rng(_pair_seed(avk, bvk, "frozen"))
        oa_a, ea_a, ob_a, eb_a = np.array(oa), np.array(ea), np.array(ob), np.array(eb); nn = len(oa)
        boots = np.array([km_survival_at(oa_a[ix].tolist(), ea_a[ix].tolist(), B_star) - km_survival_at(ob_a[ix].tolist(), eb_a[ix].tolist(), B_star)
                          for ix in (rng.integers(0, nn, nn) for _ in range(BOOT))])
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        p, _ = logrank_p(oa, ea, ob, eb, B_common); ci0 = (ci[0] > 0 or ci[1] < 0); sig = ci0 and p < BONFERRONI_P
        verdict = "indistinguishable_at_op_budget" if not sig else ("lb_wins" if dS < 0 else "stern_wins")
        frozen_pairs.append({"pair": pk, "B_star_locked": B_star, "dS": dS, "dS_ci": ci, "logrank_p": p, "verdict": verdict})
    frozen_fam = _family_verdict(frozen_pairs)
    locked_fam = reg.get("cross_attacker", {}).get("family_verdict")
    gate3 = (frozen_fam == locked_fam)
    # verdict
    if not code_valid or not labels_wt_ok or not code_id_ok:
        verdict = "void_run (integrity/code-identity)"
    elif not scored:
        verdict = "precal_insufficient (no admissible R2' variant)"
    elif not gate1:
        offenders = [vk for vk, v in scored.items() if not v["inside_band"]]
        verdict = f"method_still_off (frozen median outside band: {offenders})"
    elif not gate3:
        verdict = f"cross_attacker_falsified (locked {locked_fam} vs frozen {frozen_fam})"
    else:
        verdict = "band_validated"
    # reported bound (slate def): C_best := min over admissible variants of the FROZEN KM-median
    adm_med = {vk: v for vk, v in scored.items() if v.get("frozen_median") is not None}
    C_best = None
    if adm_med:
        bvk = min(adm_med, key=lambda kk: adm_med[kk]["frozen_median"])
        C_best = {"variant": bvk, "C_ops_50pct": adm_med[bvk]["frozen_median"], "band": adm_med[bvk]["locked_band"]}

    def wj(name, obj):
        (out_dir / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    wj("verifier_access_declaration.json", {"attacker_input": "z only", "scoring_only": ["s", "e", "wt_e"], "target_manifest_sha256": msha})
    wj("code_identity.json", {"precal_code_digest": reg["code_digest"], "frozen_code_digest": dig, "match": code_id_ok})
    wj("manifest.json", {"schema": "pvnp-certificate-syndrome-v5-run-manifest", "complete": True, "regime": "r2prime",
                         "git_sha": git_sha, "target_manifest_sha256": msha, "suggested_verdict": verdict,
                         "gate1_coverage": gate1, "gate3_cross_attacker_match": gate3, "C_best": C_best,
                         "code_valid_GHt0": code_valid, "labels_wt_ok": labels_wt_ok})
    wj("gate_results.json", {"GATE1_coverage": {vk: {kk: v.get(kk) for kk in ("frozen_median", "locked_band", "inside_band", "n_censored", "nearest_edge_ratio")} for vk, v in per_variant.items()},
                             "GATE3_cross_attacker": {"locked_family": locked_fam, "frozen_family": frozen_fam, "match": gate3,
                                                      "frozen_pairs": frozen_pairs, "locked_pairs": reg.get("cross_attacker", {}).get("pairs")}})
    wj("falsifier_summary.md", {"_": "see manifest.suggested_verdict"})
    (out_dir / "falsifier_summary.md").write_text(
        f"# v5 R2' frozen — verdict **{verdict}**\n\n"
        f"- GATE-1 coverage (frozen median in band): {gate1}\n"
        + "".join(f"  - {vk}: median {v.get('frozen_median')} in band {v.get('locked_band')} -> {v.get('inside_band')} (cens {v.get('n_censored')})\n"
                  for vk, v in per_variant.items()) +
        f"- GATE-3 cross-attacker: locked family={locked_fam} vs frozen family={frozen_fam} -> match={gate3}\n"
        f"- code_identity={code_id_ok}; witnesses re-checked from public H/z/tau; attackers saw only z.\n"
        f"Boundary: C_best upper bound vs LB/Stern; no crypto/P-vs-NP; op-count cost.\n", encoding="utf-8")
    return {"verdict": verdict, "gate1": gate1, "gate3": gate3, "per_variant": per_variant,
            "frozen_family": frozen_fam, "locked_family": locked_fam, "code_id_ok": code_id_ok}


def run_summarize(root, lock):
    root = Path(root)
    val = json.loads((root / "v3-validation" / "v3_validation.json").read_text()) if (root / "v3-validation" / "v3_validation.json").exists() else None
    fr = json.loads((root / "r2prime-frozen" / "manifest.json").read_text()) if (root / "r2prime-frozen" / "manifest.json").exists() else None
    gates = json.loads((root / "r2prime-frozen" / "gate_results.json").read_text()) if (root / "r2prime-frozen" / "gate_results.json").exists() else None
    summary = {"schema": "pvnp-certificate-syndrome-v5-summary", "profile": lock.get("profile"),
               "v3_validation": (val["verdict"] if val else "not_run"),
               "r2prime_verdict": (fr["suggested_verdict"] if fr else "not_run"),
               "gate1": (fr["gate1_coverage"] if fr else None), "gate3": (fr["gate3_cross_attacker_match"] if fr else None),
               "r2prime_precal_bands": {vk: v["locked_band"] for vk, v in lock["regimes"]["r2prime"]["variants"].items() if not v["precal_insufficient"]},
               "r2prime_declined": [vk for vk, v in lock["regimes"]["r2prime"]["variants"].items() if v["precal_insufficient"]],
               "locked_family_verdict": lock["regimes"]["r2prime"].get("cross_attacker", {}).get("family_verdict"),
               "gate_detail": gates}
    (root).mkdir(parents=True, exist_ok=True)
    (root / "scaling_summary_v5.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (root / "V5_SUMMARY.md").write_text(
        f"# v5 summary (profile {summary['profile']})\n\n"
        f"- Stage-2b v3 validation: **{summary['v3_validation']}**\n"
        f"- R2' frozen verdict: **{summary['r2prime_verdict']}** (GATE-1 {summary['gate1']}, GATE-3 {summary['gate3']})\n"
        f"- R2' precal bands: {summary['r2prime_precal_bands']}\n"
        f"- R2' declined (precal_insufficient): {summary['r2prime_declined']}\n"
        f"- locked cross-attacker family verdict: {summary['locked_family_verdict']}\n", encoding="utf-8")
    return summary


def _smoke_km_equiv():
    """KM median == v4 _c50_ops under zero censoring (deterministic unit check)."""
    rng = np.random.default_rng(123)
    ops = sorted(float(x) for x in rng.integers(1_000, 1_000_000, 64)); ev = [1] * 64
    km, cens = km_median(ops, ev)
    rows = [{"id": i, "ops_at_success": ops[i], "first_B": 1} for i in range(64)]
    c50 = v2._c50_ops(rows, 64)
    return km, c50, (abs(km - c50) < 1e-6 and not cens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--precal", action="store_true")
    ap.add_argument("--profile", default="primary"); ap.add_argument("--validate-v3", action="store_true")
    ap.add_argument("--frozen-r2prime", action="store_true"); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--harness-test", action="store_true")
    ap.add_argument("--out", default="results/pvnp/certificate-syndrome-v5")
    ap.add_argument("--lock-out", default="docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_PREDICTION_LOCK.json")
    ap.add_argument("--prediction-lock", default="docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_PREDICTION_LOCK.json")
    ap.add_argument("--root", default="results/pvnp/certificate-syndrome-v5")
    args = ap.parse_args()
    lp = lambda pth: Path(pth) if Path(pth).is_absolute() else REPO / pth

    if args.smoke:
        fit = v3.run_stage1_smoke(REPO / args.out)
        km, c50, ok = _smoke_km_equiv()
        print(f"v5 base(m)={fit['alpha']:.4g}*m^{fit['beta']:.3f} base(64)={fit['base_m64']:.3e}")
        print(f"KM-median==v4 _c50_ops (zero censoring): km={km:.1f} c50={c50:.1f} EQUAL={ok}")
        return
    if args.harness_test:
        global REGIMES, T_PRE, T_FROZEN
        REGIMES = {"r2prime": dict(n=64, k=32, w=6, tau=6, code_seed=9201, precal_target_seeds=[9211, 9212, 9213, 9214],
                                   frozen_target_seed=9219, variants=[("lb", None), ("stern", 7), ("stern", 8)],
                                   cross_pairs=[(("lb", None), ("stern", 7)), (("lb", None), ("stern", 8))], role="test")}
        T_PRE = T_FROZEN = 24
        global PROFILES; PROFILES = {"primary": {"K": 4, "g": 1.25}, "fallback": {"K": 3, "g": 1.35}}
        lock = run_precal(REPO / "results/pvnp/_certsyn-v5-harness-test/precal", REPO / "results/pvnp/_certsyn-v5-harness-test/lock.json", "primary")
        res = run_frozen_r2prime(lock, REPO / "results/pvnp/_certsyn-v5-harness-test/r2prime", v2._git_sha())
        km, c50, ok = _smoke_km_equiv()
        print(f"v5 harness-test: KM==c50 {ok}; R2' verdict={res['verdict']} gate1={res['gate1']} gate3={res['gate3']} "
              f"code_id={res['code_id_ok']} locked_fam={res['locked_family']} frozen_fam={res['frozen_family']}")
        return
    if args.precal:
        if args.profile not in PROFILES:
            raise SystemExit("--profile must be primary|fallback")
        lock = run_precal(REPO / args.out, lp(args.lock_out), args.profile)
        print(f"=== v5 precal (profile {args.profile}, K={PROFILES[args.profile]['K']}) ===")
        for rid in REGIMES:
            for vk, v in lock["regimes"][rid]["variants"].items():
                b = v["locked_band"]; print(f"  {rid}/{vk}: band={b} W={v['W']} insufficient={v['precal_insufficient']}" if b
                                            else f"  {rid}/{vk}: PRECAL_INSUFFICIENT (cens {v['seeds_censored']})")
            if lock["regimes"][rid].get("cross_attacker"):
                print(f"  {rid} family verdict: {lock['regimes'][rid]['cross_attacker']['family_verdict']}")
        return
    if args.validate_v3:
        lock = json.loads(lp(args.prediction_lock).read_text())
        res = run_validate_v3(lock, REPO / args.out / "v3-validation" if Path(args.out).name != "v3-validation" else REPO / args.out)
        print(f"=== v5 Stage-2b validation: {res['verdict']} ===")
        for p in res["points"]:
            print(f"  {p['regime']}/{p['variant']}: measured {p['measured']:.3e} in band {p['band']} -> {p['inside']}")
        return
    if args.frozen_r2prime:
        lock = json.loads(lp(args.prediction_lock).read_text())
        res = run_frozen_r2prime(lock, REPO / args.out, v2._git_sha())
        print(f"=== v5 FROZEN R2' === verdict: {res['verdict']}  (GATE-1 {res['gate1']}, GATE-3 {res['gate3']})")
        return
    if args.summarize:
        lock = json.loads(lp(args.prediction_lock).read_text()) if lp(args.prediction_lock).exists() else {}
        s = run_summarize(REPO / args.root, lock)
        print(f"=== v5 summary === v3-validation: {s['v3_validation']}  R2': {s['r2prime_verdict']}")
        return
    print("Pass: --smoke | --precal --profile {primary,fallback} | --validate-v3 | --frozen-r2prime | --summarize | --harness-test")


if __name__ == "__main__":
    main()
