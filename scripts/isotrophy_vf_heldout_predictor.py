#!/usr/bin/env python3
"""Isotrophy v0.10b: monotone-vf held-out predictor.

Locked by docs/isotrophy/kfacet/kfacet_v10b_monotone_vf_heldout_predictor_form.md.
Consumes the frozen v0.9a receipt + the v0.10a manifest. No new variational compute.

Primary gate: pooled leave-one-m3-bin-out held-out AUC of a zone-only weighted-PAVA
monotone risk score, with a within-m3 stratified label-permutation falsifier
(full CV refit), p <= 0.01 AND AUC > 0.5. Hard-label classifier is a NON-GATING
sidecar (every zone is sub-50% stable, so accuracy-vs-always-U is base-rate-trapped).
"""
import argparse
import csv
import json
import math
import os

import numpy as np

ZONE_ORDER = ["positional-dominant", "mixed", "velocity-heavy"]
CUTPOINTS = (0.25, 0.50)
EXPECTED_ZONE = {"positional-dominant": (19, 2, 17), "mixed": (165, 56, 109),
                 "velocity-heavy": (66, 29, 37)}
EXPECTED_TOTAL = (250, 87, 163)
MIN_FOLD_N = 5
SEED = 20260523
N_PERM = 10000
ALPHA = 0.01


def zone_of(vf):
    if vf < CUTPOINTS[0]:
        return 0
    if vf < CUTPOINTS[1]:
        return 1
    return 2


def weighted_pava(values, weights):
    """Weighted Pool-Adjacent-Violators: nondecreasing fit (general k)."""
    lv, lw, lc = [], [], []
    for v, w in zip(values, weights):
        cv, cw, cc = float(v), float(w), 1
        while lv and lv[-1] > cv:
            pv, pw, pc = lv.pop(), lw.pop(), lc.pop()
            cv = (pv * pw + cv * cw) / (pw + cw)
            cw = pw + cw
            cc = pc + cc
        lv.append(cv); lw.append(cw); lc.append(cc)
    out = []
    for v, c in zip(lv, lc):
        out.extend([v] * c)
    return out


def auc(scores, labels):
    """Rank-based AUC = P(score_S > score_U) + 0.5 P(=). labels: 1=S, 0=U."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n_s = int(labels.sum())
    n_u = len(labels) - n_s
    if n_s == 0 or n_u == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[i:j] = (i + j - 1) / 2.0 + 1.0  # average rank, 1-based
        i = j
    rank_of = np.empty(len(scores), float)
    rank_of[order] = ranks
    rank_sum_s = rank_of[labels == 1].sum()
    return (rank_sum_s - n_s * (n_s + 1) / 2.0) / (n_s * n_u)


def cv_heldout_scores(y, fold, zone, n_folds):
    """Pooled leave-one-fold-out scores for label vector y (per-zone weighted PAVA)."""
    gid = fold * 3 + zone  # 0..(n_folds*3-1)
    n_fz = np.bincount(gid, minlength=n_folds * 3).reshape(n_folds, 3).astype(float)
    s_fz = np.bincount(gid, weights=y, minlength=n_folds * 3).reshape(n_folds, 3)
    n_z = n_fz.sum(axis=0)
    s_z = s_fz.sum(axis=0)
    pava = np.zeros((n_folds, 3))
    for f in range(n_folds):
        tn = n_z - n_fz[f]          # train zone counts (this fold held out)
        ts = s_z - s_fz[f]
        raw = np.array([ts[z] / tn[z] if tn[z] > 0 else np.nan for z in range(3)])
        if np.isnan(raw).any():     # empty train zone -> merge with nearest (rare)
            filled = raw.copy()
            for z in range(3):
                if np.isnan(filled[z]):
                    nb = [zz for zz in range(3) if not np.isnan(raw[zz])]
                    filled[z] = raw[min(nb, key=lambda zz: abs(zz - z))]
            raw = filled
            tn = np.where(tn > 0, tn, 1.0)
        pava[f] = weighted_pava(raw, tn)
    return pava[fold, zone]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir",
                    default="results/isotrophy/k-facet-v09a-signed-vf-three-zone")
    ap.add_argument("--v10a", dest="v10a_dir",
                    default="results/isotrophy/k-facet-v10a-jt-trend")
    ap.add_argument("--out", dest="out_dir",
                    default="results/isotrophy/k-facet-v10b-monotone-vf-heldout")
    args = ap.parse_args()

    with open(os.path.join(args.in_dir, "per_row_table.csv"), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # ---- frozen-input cross-check ----
    cc = {"row_count": len(rows)}
    zc = {z: [0, 0, 0] for z in ZONE_ORDER}
    zone_mismatch = 0
    for r in rows:
        zi = zone_of(float(r["velocity_fraction"]))
        if ZONE_ORDER[zi] != r["zone"].strip():
            zone_mismatch += 1
        z = ZONE_ORDER[zi]
        zc[z][0] += 1
        if r["stability"].strip() == "S":
            zc[z][1] += 1
        else:
            zc[z][2] += 1
    counts_ok = all(tuple(zc[z]) == EXPECTED_ZONE[z] for z in ZONE_ORDER)
    tot = (len(rows), sum(zc[z][1] for z in ZONE_ORDER), sum(zc[z][2] for z in ZONE_ORDER))
    v10a = json.load(open(os.path.join(args.v10a_dir, "manifest.json"), encoding="utf-8"))
    v10a_ok = (v10a.get("verdict") == "jt_trend_monotone_registered"
               and v10a.get("exact_enumeration", {}).get("p_value_one_sided", 1) < 0.01)
    cc.update({"zone_counts": {z: zc[z] for z in ZONE_ORDER},
               "matches_v09a_manifest": counts_ok and tot == EXPECTED_TOTAL,
               "zone_recompute_mismatch_count": zone_mismatch,
               "v10a_verdict": v10a.get("verdict"), "v10a_p_under_0p01": v10a_ok})

    os.makedirs(args.out_dir, exist_ok=True)
    if not (cc["matches_v09a_manifest"] and zone_mismatch == 0 and v10a_ok):
        out = {"mode": "v0.10b-monotone-vf-heldout", "verdict": "ABORT_frozen_input_crosscheck_failed",
               "frozen_input_crosscheck": cc}
        json.dump(out, open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8"), indent=2)
        print("[v0.10b] ABORT — frozen-input cross-check failed", cc)
        raise SystemExit(1)

    # ---- partition: leave-one-m3-bin-out, N>=5 ----
    m3 = np.array([float(r["m3"]) for r in rows])
    zone = np.array([zone_of(float(r["velocity_fraction"])) for r in rows])
    y = np.array([1 if r["stability"].strip() == "S" else 0 for r in rows])
    branch = [r.get("branch_label", "").strip() for r in rows]
    bins = sorted(set(m3.tolist()))
    primary_bins = [b for b in bins if (m3 == b).sum() >= MIN_FOLD_N]
    report_bins = [b for b in bins if (m3 == b).sum() < MIN_FOLD_N]
    fold_of_bin = {b: i for i, b in enumerate(primary_bins)}
    is_primary = np.array([m3[i] in fold_of_bin for i in range(len(rows))])
    p_idx = np.where(is_primary)[0]
    fold = np.array([fold_of_bin[m3[i]] for i in p_idx])
    zone_p = zone[p_idx]
    y_p = y[p_idx]
    n_folds = len(primary_bins)

    if not (len(p_idx) == 234 and int(y_p.sum()) == 82):
        out = {"mode": "v0.10b-monotone-vf-heldout", "verdict": "ABORT_partition_mismatch",
               "frozen_input_crosscheck": cc,
               "primary_rows": int(len(p_idx)), "primary_S": int(y_p.sum())}
        json.dump(out, open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8"), indent=2)
        print("[v0.10b] ABORT — partition mismatch", len(p_idx), int(y_p.sum()))
        raise SystemExit(1)

    # ---- observed held-out AUC ----
    obs_scores = cv_heldout_scores(y_p, fold, zone_p, n_folds)
    auc_obs = auc(obs_scores, y_p)

    # ---- within-m3 stratified permutation falsifier (full CV refit) ----
    rng = np.random.default_rng(SEED)
    fold_row_idx = [np.where(fold == f)[0] for f in range(n_folds)]
    ge = 0
    perm_aucs = []
    for _ in range(N_PERM):
        yp = y_p.copy()
        for idx in fold_row_idx:                 # shuffle labels within each m3 bin
            yp[idx] = rng.permutation(yp[idx])
        a = auc(cv_heldout_scores(yp, fold, zone_p, n_folds), yp)
        perm_aucs.append(a)
        if a >= auc_obs:
            ge += 1
    perm_aucs = np.array(perm_aucs, float)
    perm_p = (1 + ge) / (1 + N_PERM)

    # ---- verdict ----
    if auc_obs <= 0.5:
        verdict = "monotone_vf_predictor_fails_heldout"
    elif perm_p <= ALPHA:
        verdict = "monotone_vf_predictor_passes_heldout"
    else:
        verdict = "monotone_vf_predictor_fails_heldout"

    # ---- per-fold diagnostics ----
    per_fold = []
    for f, b in enumerate(primary_bins):
        m = fold == f
        ys, zs = y_p[m], zone_p[m]
        fold_scores = obs_scores[m]
        tn = np.bincount(zone_p[fold != f], minlength=3).astype(float)
        ts = np.bincount(zone_p[fold != f], weights=y_p[fold != f], minlength=3)
        raw = np.array([ts[z] / tn[z] if tn[z] > 0 else np.nan for z in range(3)])
        pav = weighted_pava(np.nan_to_num(raw), np.where(tn > 0, tn, 1.0))
        fa = auc(fold_scores, ys)
        p_glob = y_p[fold != f].mean()
        brier = float(np.mean((fold_scores - ys) ** 2))
        brier_base = float(np.mean((p_glob - ys) ** 2))
        per_fold.append({"m3": b, "N": int(m.sum()), "S": int(ys.sum()), "U": int((1 - ys).sum()),
                         "train_zone_rate": [round(float(x), 4) for x in raw],
                         "train_pava_score": [round(float(x), 4) for x in pav],
                         "fold_auc": (round(fa, 4) if fa is not None else None),
                         "both_classes": bool(ys.sum() > 0 and ys.sum() < m.sum()),
                         "fold_brier": round(brier, 4), "fold_brier_baseline": round(brier_base, 4)})

    # ---- hard-label sidecar (NON-GATING) ----
    def bal_acc(pred, lab):
        s = lab == 1; u = lab == 0
        tpr = pred[s].mean() if s.any() else 0.0
        tnr = (1 - pred[u]).mean() if u.any() else 0.0
        return 0.5 * (tpr + tnr)
    hard_correct = 0
    alwaysU_correct = 0
    disc_pred_right = 0
    disc_pred_wrong = 0
    for f in range(n_folds):
        tr = fold != f; te = fold == f
        ztr, ytr = zone_p[tr], y_p[tr]
        zte, yte = zone_p[te], y_p[te]
        t1 = bal_acc((ztr >= 1).astype(int), ytr)
        t2 = bal_acc((ztr >= 2).astype(int), ytr)
        thr = 2 if t2 >= t1 else 1
        pred = (zte >= thr).astype(int)
        au = np.zeros_like(yte)        # always-U
        hard_correct += int((pred == yte).sum())
        alwaysU_correct += int((au == yte).sum())
        disc = pred != au
        disc_pred_right += int(((pred == yte) & disc).sum())
        disc_pred_wrong += int(((au == yte) & disc).sum())
    n_disc = disc_pred_right + disc_pred_wrong
    # exact one-sided binomial (McNemar) on discordant pairs
    mcnemar_p = None
    if n_disc > 0:
        k = disc_pred_right
        mcnemar_p = sum(math.comb(n_disc, i) for i in range(k, n_disc + 1)) / (2 ** n_disc)
    sidecar = {"gating": False, "hard_accuracy": round(hard_correct / 234, 4),
               "alwaysU_accuracy": round(alwaysU_correct / 234, 4),
               "accuracy_delta": round((hard_correct - alwaysU_correct) / 234, 4),
               "discordant": n_disc, "hard_right_on_discordant": disc_pred_right,
               "mcnemar_one_sided_p": (round(mcnemar_p, 5) if mcnemar_p is not None else None)}

    result = {
        "schema": "sundog.isotrophy.v0.10b-monotone-vf-heldout.v1",
        "mode": "v0.10b-monotone-vf-heldout",
        "form_lock": "docs/isotrophy/kfacet/kfacet_v10b_monotone_vf_heldout_predictor_form.md",
        "input_v09a_per_row_table": os.path.join(args.in_dir, "per_row_table.csv").replace("\\", "/"),
        "input_v10a_manifest": os.path.join(args.v10a_dir, "manifest.json").replace("\\", "/"),
        "frozen_input_crosscheck": cc,
        "primary_gating_folds": [{"m3": b, "N": int((m3 == b).sum())} for b in primary_bins],
        "report_only_folds": [{"m3": b, "N": int((m3 == b).sum())} for b in report_bins],
        "primary_rows": 234, "primary_S": 82, "primary_U": 152,
        "predictor": {"feature": "zone_index", "model": "weighted_pava_by_training_fold"},
        "primary_metric": {"auc_observed": round(float(auc_obs), 4),
                           "auc_delta_vs_constant": round(float(auc_obs) - 0.5, 4),
                           "permutation_p": perm_p, "permutation_auc_mean": round(float(perm_aucs.mean()), 4),
                           "permutation_auc_p95": round(float(np.quantile(perm_aucs, 0.95)), 4),
                           "seed": SEED, "n_permutations": N_PERM, "alpha": ALPHA},
        "per_fold_table": per_fold,
        "hard_label_sidecar": sidecar,
        "verdict": verdict,
    }
    json.dump(result, open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8"), indent=2)
    json.dump(result, open(os.path.join(args.out_dir, "jt_heldout_receipt.json"), "w", encoding="utf-8"), indent=2)

    print(f"[v0.10b] frozen-input cross-check PASSED; primary 234 rows (82 S / 152 U), {n_folds} folds")
    print(f"[v0.10b] held-out AUC = {auc_obs:.4f}  (delta vs constant {auc_obs - 0.5:+.4f})")
    print(f"[v0.10b] within-m3 permutation (n={N_PERM}, seed {SEED}): p = {perm_p:.4e}  "
          f"(perm AUC mean {perm_aucs.mean():.4f}, p95 {np.quantile(perm_aucs, 0.95):.4f})")
    f04 = next(d for d in per_fold if d["m3"] == 0.4)
    print(f"[v0.10b] m3=0.4 fold (load-bearing): N={f04['N']} S={f04['S']} U={f04['U']} "
          f"fold_AUC={f04['fold_auc']} pava={f04['train_pava_score']}")
    print(f"[v0.10b] hard-label sidecar (NON-GATING): acc_delta={sidecar['accuracy_delta']:+.4f} "
          f"mcnemar_p={sidecar['mcnemar_one_sided_p']}")
    print(f"[v0.10b] VERDICT: {verdict}  (alpha={ALPHA})")


if __name__ == "__main__":
    main()
