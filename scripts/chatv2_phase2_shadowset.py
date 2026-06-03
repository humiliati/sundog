#!/usr/bin/env python
"""chatv2 Phase 2 - determining-shadow-set probe (frozen spec + Amendment 1).

Spec: docs/chatv2/PHASE2_DETERMINING_SHADOW_SET_SPEC.md
  Probe 1  = same-seed determining-set *control*, read against a SELECTION-CORRECTED
             null (A4: random-direction best-of-254 max distribution), not an
             absolute threshold. Expected det_shadow_predicted_null.
  Probe 1b = paired-fiber audit (basic).
  Probe 2  = cross-seed transplant *headline*. Tier 1/2 use SOURCE layer l*_a;
             l*_b / b-refit are ceiling. A3 adds subspace_overlap (principal
             angles) + Tier 3b (8-score b-refit) to separate "no shared structure"
             from "shared subspace, fragile per-latent frame".
  A1/A2    = selection spectral gap (residual-stream eigengap) + frame-spread, the
             isotrophy v0.20 label-blind reliability transplant.
Fit on TRAIN, report on HELD. Nothing is fit on held labels.
"""
import argparse, itertools, json, os, time
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression, Ridge

PROBE_SEED, SEEDS, N, NTRAIN = 0, [0, 1, 2], 3000, 1500
THRESH_FUNC, THRESH_STATE, CHANCE = 0.70, 0.60, 0.50   # absolute (descriptive only)
Q_EPS, PAIR_FLOOR, B_G, R_NULL = 0.01, 200, 30, 30

ROOT = "C:/Users/hughe/Dev/sundog"
BASE = f"{ROOT}/results/chatv2/phase1-seedstab"
OUT  = f"{ROOT}/results/chatv2/phase2-determining-shadow-set"


def load(seed, kind):
    d = np.load(f"{BASE}/seed{seed}/bodies/H8_{kind}.npz", allow_pickle=True)
    return d["bodies"], d["z"].astype(int)


def split_idx(n=N):
    perm = default_rng(PROBE_SEED).permutation(n)
    return perm[:NTRAIN], perm[NTRAIN:]


def std_fit(X):
    return X.mean(0), X.std(0) + 1e-8


def view(bodies, l, i):
    return bodies[:, l, i, :]


def layer_select(bodies, z, tr):
    from sklearn.model_selection import cross_val_score
    H, L = z.shape[1], bodies.shape[1]
    best_l, best = 0, -1.0
    for l in range(L):
        accs = []
        for i in range(H):
            X = view(bodies, l, i)[tr]
            mu, sd = std_fit(X)
            accs.append(cross_val_score(LogisticRegression(max_iter=1000),
                                        (X - mu) / sd, z[tr, i], cv=4).mean())
        m = float(np.mean(accs))
        if m > best:
            best, best_l = m, l
    return best_l


def readout_dirs(bodies, z, lstar, tr):
    dirs = []
    for i in range(z.shape[1]):
        X = view(bodies, lstar, i)[tr]
        mu, sd = std_fit(X)
        lr = LogisticRegression(max_iter=1000).fit((X - mu) / sd, z[tr, i])
        w = lr.coef_.ravel()
        dirs.append((mu, sd, w / (np.linalg.norm(w) + 1e-12)))
    return dirs


def score_matrix(bodies, lstar, dirs, idx, calib_stats=None):
    """calib_stats = per-latent (mu, sd) override fit on the TARGET's TRAIN rows
    (Tier-2/3 calibration); None keeps the source dirs' std."""
    H = len(dirs)
    S = np.zeros((len(idx), H))
    for i in range(H):
        mu, sd, w = dirs[i]
        if calib_stats is not None:
            mu, sd = calib_stats[i]
        S[:, i] = ((view(bodies, lstar, i)[idx] - mu) / sd) @ w
    return S


def _fve(y, pred):
    ss = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss


# ---- A1: selection spectral gap + frame spread (label-blind reliability) ---- #
def frame_spread(bodies, z, lstar, i, tr, rng, B=B_G):
    X = view(bodies, lstar, i)[tr]
    mu, sd = std_fit(X)
    Xs, y, n = (X - mu) / sd, z[tr, i], len(tr)
    W = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        lr = LogisticRegression(max_iter=1000).fit(Xs[idx], y[idx])
        w = lr.coef_.ravel()
        W.append(w / (np.linalg.norm(w) + 1e-12))
    W = np.array(W)
    cs = np.abs(W @ W.T)
    iu = np.triu_indices(B, 1)
    return float(1.0 - cs[iu].mean())


def eigengap(bodies, lstar, i, tr):
    X = view(bodies, lstar, i)[tr]
    mu, sd = std_fit(X)
    ev = np.linalg.eigvalsh(np.cov(((X - mu) / sd).T))[::-1]
    return float((ev[0] - ev[1]) / (ev[0] + 1e-12))


def subspace_overlap(Wa, Wb):
    """mean cos^2 of principal angles between span(rows Wa) and span(rows Wb)."""
    Qa = np.linalg.qr(Wa.T)[0]
    Qb = np.linalg.qr(Wb.T)[0]
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(sv, 0.0, 1.0) ** 2))


# ---- Probe 1: same-seed sweep --------------------------------------------- #
def same_seed_sweep(Str, She, ztr, zhe):
    H = Str.shape[1]
    rows = []
    for k in range(1, H):
        for S in itertools.combinations(range(H), k):
            J = [j for j in range(H) if j not in S]
            Xtr, Xhe = Str[:, S], She[:, S]
            ftr, fhe, str_, she_ = [], [], [], []
            for j in J:
                clf = LogisticRegression(max_iter=1000).fit(Xtr, ztr[:, j])
                ftr.append(clf.score(Xtr, ztr[:, j]))
                fhe.append(clf.score(Xhe, zhe[:, j]))
                rg = Ridge().fit(Xtr, Str[:, j])
                str_.append(_fve(Str[:, j], rg.predict(Xtr)))
                she_.append(_fve(She[:, j], rg.predict(Xhe)))
            rows.append(dict(k=k, S=S, func_train=float(np.mean(ftr)),
                             func_held=float(np.mean(fhe)), state_train=float(np.mean(str_)),
                             state_held=float(np.mean(she_))))
    head = {}
    for k in range(1, H):
        kr = [r for r in rows if r["k"] == k]
        sf = max(kr, key=lambda r: r["func_train"])
        ss = max(kr, key=lambda r: r["state_train"])    # train-selected, held-reported
        head[k] = dict(S_func=sf["S"], func_held=sf["func_held"],
                       S_state=ss["S"], state_held=ss["state_held"])
    return rows, head


def null_floor(bodies, z, lstar, tr, he, rng, R=R_NULL):
    """A4: random-direction best-of-254 max distribution -> 95th pct floor."""
    mf, ms = [], []
    for _ in range(R):
        dirs = []
        for i in range(8):
            mu, sd = std_fit(view(bodies, lstar, i)[tr])
            w = rng.standard_normal(192)
            dirs.append((mu, sd, w / np.linalg.norm(w)))
        Str = score_matrix(bodies, lstar, dirs, tr)
        She = score_matrix(bodies, lstar, dirs, he)
        _, head = same_seed_sweep(Str, She, z[tr], z[he])
        mf.append(max(h["func_held"] for h in head.values()))
        ms.append(max(h["state_held"] for h in head.values()))
    return float(np.percentile(mf, 95)), float(np.percentile(ms, 95))


def control_perm(Str, She, ztr, zhe, rng):
    zt = np.column_stack([rng.permutation(ztr[:, j]) for j in range(ztr.shape[1])])
    zh = np.column_stack([rng.permutation(zhe[:, j]) for j in range(zhe.shape[1])])
    _, head = same_seed_sweep(Str, She, zt, zh)
    return float(max(h["func_held"] for h in head.values()))


# ---- Probe 1b: paired fiber ------------------------------------------------ #
def paired_fiber(She, zhe, S, rng):
    H = She.shape[1]
    J = [j for j in range(H) if j not in S]
    n = She.shape[0]
    a, b = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    m = a != b
    a, b = a[m], b[m]
    sc = She[:, S]
    d_shadow = np.linalg.norm(sc[a] - sc[b], axis=1)
    eps = np.quantile(d_shadow, Q_EPS)
    om = She[:, J]
    d_body = np.linalg.norm(om[a] - om[b], axis=1)
    keep = (d_shadow <= eps) & (d_body >= np.median(d_body))
    a, b = a[keep], b[keep]
    if len(a) < PAIR_FLOOR:
        return dict(branch="paired_fiber_deferred_coverage", n_pairs=int(len(a)), D_witness=None)
    disagree = np.any(zhe[a][:, J] != zhe[b][:, J], axis=1)
    margin = np.min(np.abs(She[:, J]), axis=1)
    lo = margin <= np.quantile(margin, 0.1)
    D, D_lo = float(disagree.mean()), (float(disagree[lo[a]].mean()) if lo[a].any() else float("nan"))
    branch = ("paired_fiber_boundary_only" if D_lo > D + 0.1
              else "paired_fiber_closure_positive" if D < 0.1 else "paired_fiber_conflict")
    return dict(branch=branch, n_pairs=int(len(a)), D_witness=round(D, 4))


# ---- Probe 2: cross-seed transplant + A3 ----------------------------------- #
def cross_seed(cache):
    out = []
    for a in SEEDS:
        for b in SEEDS:
            if a == b:
                continue
            ca, cb = cache[a], cache[b]
            la, dirs_a = ca["lstar"], ca["dirs"]
            zb_he = cb["z"][cb["he"]]
            Sa_tr = score_matrix(ca["bodies"], la, dirs_a, ca["tr"])
            decs = [LogisticRegression(max_iter=1000).fit(Sa_tr[:, i:i + 1], ca["z"][ca["tr"], i])
                    for i in range(8)]
            calib_b = [std_fit(view(cb["bodies"], la, i)[cb["tr"]]) for i in range(8)]
            S1 = score_matrix(cb["bodies"], la, dirs_a, cb["he"])
            t1 = np.mean([decs[i].score(S1[:, i:i + 1], zb_he[:, i]) for i in range(8)])
            S2 = score_matrix(cb["bodies"], la, dirs_a, cb["he"], calib_stats=calib_b)
            t2 = np.mean([decs[i].score(S2[:, i:i + 1], zb_he[:, i]) for i in range(8)])
            S3tr = score_matrix(cb["bodies"], la, dirs_a, cb["tr"], calib_stats=calib_b)
            S3he = score_matrix(cb["bodies"], la, dirs_a, cb["he"], calib_stats=calib_b)
            t3 = np.mean([LogisticRegression(max_iter=1000)
                          .fit(S3tr[:, i:i + 1], cb["z"][cb["tr"], i])
                          .score(S3he[:, i:i + 1], zb_he[:, i]) for i in range(8)])
            # A3: Tier 3b (all-8-score b-refit) + principal-angle subspace overlap
            t3b = np.mean([LogisticRegression(max_iter=1000)
                           .fit(S3tr, cb["z"][cb["tr"], i]).score(S3he, zb_he[:, i])
                           for i in range(8)])
            Wa = np.array([d[2] for d in dirs_a])
            Wb = np.array([d[2] for d in cb["dirs"]])
            out.append(dict(a=a, b=b, tier1=float(t1), tier2=float(t2), tier3=float(t3),
                            tier3b=float(t3b), subspace_overlap=round(subspace_overlap(Wa, Wb), 4)))
    return out


def cross_seed_branch(pairs):
    p1 = sum(p["tier1"] >= THRESH_FUNC for p in pairs)
    p2 = sum(p["tier2"] >= THRESH_FUNC for p in pairs)
    p3 = sum(p["tier3"] >= THRESH_FUNC for p in pairs)
    if p1 >= 4:
        return "cross_seed_direct_pass", (p1, p2, p3)
    if p2 >= 4:
        return "cross_seed_calibrated_only", (p1, p2, p3)
    if p3 >= 4:
        return "cross_seed_ceiling_only", (p1, p2, p3)
    return "cross_seed_no_transfer", (p1, p2, p3)


def _rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ar, br = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ar, br)[0, 1])


# ---- driver --------------------------------------------------------------- #
def run(smoke=False):
    os.makedirs(OUT, exist_ok=True)
    rng = default_rng(PROBE_SEED)
    tr, he = split_idx()
    assert len(set(tr.tolist()) & set(he.tolist())) == 0, "split leak"
    seeds = [0] if smoke else SEEDS
    R = 6 if smoke else R_NULL
    t0 = time.time()

    bg0, z0 = load(0, "gen")
    lstar0 = layer_select(bg0, z0, tr)
    nf_func, nf_state = null_floor(bg0, z0, lstar0, tr, he, rng, R=R)
    print(f"[null] selection-corrected 95th-pct floor: func={nf_func:.3f} state={nf_state:.3f} "
          f"(R={R}, pooled seed0)", flush=True)

    cache, summary, sweeps, controls, pf_rows, gap_rows = {}, [], [], [], [], []
    for s in seeds:
        bg, z = (bg0, z0) if s == 0 else load(s, "gen")
        bt, _ = load(s, "twin")
        lstar = lstar0 if s == 0 else layer_select(bg, z, tr)
        dirs = readout_dirs(bg, z, lstar, tr)
        cache[s] = dict(bodies=bg, z=z, tr=tr, he=he, lstar=lstar, dirs=dirs)

        Sg_tr = score_matrix(bg, lstar, dirs, tr)
        Sg_he = score_matrix(bg, lstar, dirs, he)
        rows, head = same_seed_sweep(Sg_tr, Sg_he, z[tr], z[he])
        mxf = max(h["func_held"] for h in head.values())
        mxs = max(h["state_held"] for h in head.values())

        lstar_t = layer_select(bt, z, tr)
        dirs_t = readout_dirs(bt, z, lstar_t, tr)
        _, head_t = same_seed_sweep(score_matrix(bt, lstar_t, dirs_t, tr),
                                    score_matrix(bt, lstar_t, dirs_t, he), z[tr], z[he])
        mxf_t = max(h["func_held"] for h in head_t.values())

        perm_max = control_perm(Sg_tr, Sg_he, z[tr], z[he], rng)
        perm_ok = perm_max <= nf_func
        det_func, det_state = mxf > nf_func, mxs > nf_state
        branch = ("det_shadow_void" if not perm_ok
                  else "det_shadow_predicted_null" if not det_func and not det_state
                  else "det_shadow_functional_closure" if det_func and not det_state
                  else "det_shadow_state_collapse" if det_func and det_state
                  else "det_shadow_partial")

        for i in range(8):
            gap_rows.append(dict(seed=s, latent=i,
                                 eigengap=round(eigengap(bg, lstar, i, tr), 4),
                                 frame_spread=round(frame_spread(bg, z, lstar, i, tr, rng), 4)))
        pf = paired_fiber(Sg_he, z[he], head[4]["S_func"], rng)
        pf_rows.append(dict(seed=s, **pf))
        for r in rows:
            sweeps.append(dict(seed=s, k=r["k"], S="".join(map(str, r["S"])),
                               func_held=round(r["func_held"], 4), state_held=round(r["state_held"], 4)))
        summary.append(dict(seed=s, lstar=lstar, branch=branch,
                            max_func_gen=round(mxf, 4), max_state_gen=round(mxs, 4),
                            max_func_twin=round(mxf_t, 4), nf_func=round(nf_func, 4),
                            nf_state=round(nf_state, 4), det_func=det_func, det_state=det_state))
        controls.append(dict(seed=s, perm_max_func=round(perm_max, 4), perm_ok=perm_ok))
        print(f"[seed {s}] l*={lstar} branch={branch} max_func gen={mxf:.3f} twin={mxf_t:.3f} "
              f"(null {nf_func:.3f}) det_func={det_func} det_state={det_state} pf={pf['branch']}", flush=True)

    xbranch, xcounts, xpairs, A2 = None, None, [], {}
    if len(seeds) >= 2:
        xpairs = cross_seed(cache)
        xbranch, xcounts = cross_seed_branch(xpairs)
        ov = float(np.mean([p["subspace_overlap"] for p in xpairs]))
        t3b = float(np.mean([p["tier3b"] for p in xpairs]))
        for p in xpairs:
            print(f"[cross {p['a']}->{p['b']}] t1={p['tier1']:.3f} t2={p['tier2']:.3f} "
                  f"t3={p['tier3']:.3f} t3b={p['tier3b']:.3f} overlap={p['subspace_overlap']:.3f}", flush=True)
        a3 = ("shared_subspace_fragile_frame" if ov > 0.5 and t3b > 0.70 and all(p["tier1"] < 0.70 for p in xpairs)
              else "different_encoding_subspaces" if ov < 0.2 and t3b < 0.70 else "see_tier_branch")
        eg = [g["eigengap"] for g in gap_rows]
        fsp = [g["frame_spread"] for g in gap_rows]
        A2 = dict(rho_eigengap_framespread=round(_rho(eg, fsp), 3),
                  q10_eigengap=round(float(np.percentile(eg, 10)), 4),
                  q10_framespread=round(float(np.percentile(fsp, 10)), 4),
                  median_framespread=round(float(np.median(fsp)), 4),
                  subspace_overlap=round(ov, 4), tier3b=round(t3b, 4), a3=a3)
        print(f"[cross-seed] branch={xbranch} counts(t1,t2,t3)={xcounts} overlap={ov:.3f} "
              f"tier3b={t3b:.3f} A3={a3}", flush=True)
        print(f"[A1/A2] rho(eigengap,frame_spread)={A2['rho_eigengap_framespread']} "
              f"q10_eigengap={A2['q10_eigengap']} q10_framespread={A2['q10_framespread']} "
              f"median_framespread={A2['median_framespread']}", flush=True)

    dt = time.time() - t0
    agg = [s["branch"] for s in summary]
    print(f"\n=== {'SMOKE' if smoke else 'FULL'} done {dt:.1f}s | probe1={agg} | "
          f"cross-seed={xbranch} | A3={A2.get('a3')} ===", flush=True)

    if not smoke:
        _write_csv(f"{OUT}/same_seed_subset_sweep.csv", sweeps)
        _write_csv(f"{OUT}/same_seed_kfunc_kstate_summary.csv", summary)
        _write_csv(f"{OUT}/same_seed_controls.csv", controls)
        _write_csv(f"{OUT}/paired_fiber_boundary_audit.csv", pf_rows)
        _write_csv(f"{OUT}/cross_seed_transfer.csv", xpairs)
        _write_csv(f"{OUT}/selection_gap.csv", gap_rows)
        with open(f"{OUT}/branch_adjudication.json", "w") as f:
            json.dump(dict(probe1=summary, probe1_aggregate=agg, controls=controls,
                           null_floor=dict(func=nf_func, state=nf_state, R=R),
                           cross_seed_branch=xbranch, cross_seed_counts=xcounts,
                           cross_seed_pairs=xpairs, A1_A2=A2, paired_fiber=pf_rows,
                           wall_clock_s=round(dt, 1)), f, indent=2, default=str)
        print(f"[written] {OUT}", flush=True)
    else:
        print(f"[smoke] extrapolated full ~= {dt * len(SEEDS) / max(1, len(seeds)) * 1.3:.0f}s", flush=True)
    return summary, xbranch


def _write_csv(path, rows):
    if not rows:
        open(path, "w").write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    run(smoke=ap.parse_args().smoke)
