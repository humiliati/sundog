#!/usr/bin/env python
"""chatv2 Phase 2 - determining-shadow-set probe.

Implements the FROZEN spec docs/chatv2/PHASE2_DETERMINING_SHADOW_SET_SPEC.md:
  Probe 1  = same-seed determining-set *prediction-bank control* (expect
             det_shadow_predicted_null; pair-XOR latents are independent).
  Probe 1b = paired-fiber / boundary-layer audit (basic).
  Probe 2  = cross-seed transplant *headline*; Tier 1/2 use SOURCE layer l*_a,
             target-selected l*_b and b-label refit are ceiling-only.

Fit on TRAIN, report on HELD. Nothing is fit on held labels.
"""
import argparse, itertools, json, os, time
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression, Ridge

# ---- frozen parameters (spec sec 14) -------------------------------------- #
PROBE_SEED   = 0
SEEDS        = [0, 1, 2]
N            = 3000
NTRAIN       = 1500
THRESH_FUNC  = 0.70     # mean omitted-latent HELD accuracy
THRESH_STATE = 0.60     # mean omitted-score HELD FVE
NULL_TOL     = 0.12     # control null tolerance above chance
Q_EPS        = 0.01     # paired-fiber radius percentile
PAIR_FLOOR   = 200      # paired-fiber coverage floor
B_BOOT       = 1000     # bootstraps
CHANCE       = 0.50

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
    """l* = argmax_l mean_i CV4(view(l,i)->z_i) on TRAIN."""
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
    """per-latent (mu, sd, w_unit) at lstar, fit on TRAIN."""
    dirs = []
    for i in range(z.shape[1]):
        X = view(bodies, lstar, i)[tr]
        mu, sd = std_fit(X)
        lr = LogisticRegression(max_iter=1000).fit((X - mu) / sd, z[tr, i])
        w = lr.coef_.ravel()
        w = w / (np.linalg.norm(w) + 1e-12)
        dirs.append((mu, sd, w))
    return dirs


def score_matrix(bodies, lstar, dirs, idx, calib_stats=None):
    """(len(idx), H) scores. calib_stats = per-latent (mu, sd) override fit on the
    TARGET's TRAIN rows (Tier-2/3 calibration); None keeps the source dirs' std."""
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


# ---- Probe 1: same-seed sweep --------------------------------------------- #
def same_seed_sweep(Str, She, ztr, zhe):
    """returns rows + per-k headline + k_func/k_state/k_det."""
    H = Str.shape[1]
    rows = []
    for k in range(1, H):
        for S in itertools.combinations(range(H), k):
            J = [j for j in range(H) if j not in S]
            Xtr, Xhe = Str[:, S], She[:, S]
            ftr, fhe, st13, she_ = [], [], [], []
            for j in J:
                clf = LogisticRegression(max_iter=1000).fit(Xtr, ztr[:, j])
                ftr.append(clf.score(Xtr, ztr[:, j]))
                fhe.append(clf.score(Xhe, zhe[:, j]))
                rg = Ridge().fit(Xtr, Str[:, j])
                st13.append(_fve(Str[:, j], rg.predict(Xtr)))
                she_.append(_fve(She[:, j], rg.predict(Xhe)))
            rows.append(dict(k=k, S=S,
                             func_train=float(np.mean(ftr)), func_held=float(np.mean(fhe)),
                             state_train=float(np.mean(st13)), state_held=float(np.mean(she_))))
    head = {}
    for k in range(1, H):
        kr = [r for r in rows if r["k"] == k]
        sf = max(kr, key=lambda r: r["func_train"])
        ss = max(kr, key=lambda r: r["state_train"])
        head[k] = dict(S_func=sf["S"], func_held=sf["func_held"],
                       S_state=ss["S"], state_held=ss["state_held"])
    kf = next((k for k in range(1, H) if head[k]["func_held"] >= THRESH_FUNC), None)
    ks = next((k for k in range(1, H) if head[k]["state_held"] >= THRESH_STATE), None)
    kd = next((k for k in range(1, H)
               if head[k]["func_held"] >= THRESH_FUNC or head[k]["state_held"] >= THRESH_STATE), None)
    return rows, head, kf, ks, kd


def probe1_branch(kf, ks, controls_ok):
    if not controls_ok:
        return "det_shadow_void"
    if kf is None and ks is None:
        return "det_shadow_predicted_null"
    if kf is not None and (ks is None or kf < ks):
        return "det_shadow_functional_closure"   # suspect on this toy
    if kf is not None and ks is not None and abs(kf - ks) <= 1:
        return "det_shadow_state_collapse"
    return "det_shadow_partial"


# ---- Probe 1b: paired-fiber (basic, on a reference subset) ----------------- #
def paired_fiber(She, zhe, S, rng):
    H = She.shape[1]
    J = [j for j in range(H) if j not in S]
    n = She.shape[0]
    a = rng.integers(0, n, 200_000)
    b = rng.integers(0, n, 200_000)
    m = a != b
    a, b = a[m], b[m]
    sc = She[:, S]
    d_shadow = np.linalg.norm(sc[a] - sc[b], axis=1)
    eps = np.quantile(d_shadow, Q_EPS)
    om = She[:, J]
    d_body = np.linalg.norm(om[a] - om[b], axis=1)
    dbody_med = np.median(d_body)
    keep = (d_shadow <= eps) & (d_body >= dbody_med)
    a, b = a[keep], b[keep]
    if len(a) < PAIR_FLOOR:
        return dict(branch="paired_fiber_deferred_coverage", n_pairs=int(len(a)),
                    eps=float(eps), D_witness=None)
    disagree = np.any(zhe[a][:, J] != zhe[b][:, J], axis=1)
    margin = np.min(np.abs(She[:, J]), axis=1)
    lo = margin <= np.quantile(margin, 0.1)
    D = float(disagree.mean())
    D_lo = float(disagree[lo[a]].mean()) if lo[a].any() else float("nan")
    branch = ("paired_fiber_boundary_only" if D_lo > D + 0.1
              else "paired_fiber_closure_positive" if D < 0.1
              else "paired_fiber_conflict")
    return dict(branch=branch, n_pairs=int(len(a)), eps=float(eps),
                D_witness=D, D_witness_lowmargin=D_lo)


# ---- Probe 2: cross-seed transplant --------------------------------------- #
def cross_seed(cache):
    """cache[seed] = dict(bodies_gen, z, tr, he, lstar, dirs)."""
    out = []
    for a in SEEDS:
        for b in SEEDS:
            if a == b:
                continue
            ca, cb = cache[a], cache[b]
            la, dirs_a = ca["lstar"], ca["dirs"]
            zb_he = cb["z"][cb["he"]]
            # a-fit 1D decoders on a's own scores
            Sa_tr = score_matrix(ca["bodies"], la, dirs_a, ca["tr"])
            decs = [LogisticRegression(max_iter=1000).fit(Sa_tr[:, i:i + 1], ca["z"][ca["tr"], i])
                    for i in range(8)]
            # Tier 1: source l*_a, source std, source decoder, on b
            S1 = score_matrix(cb["bodies"], la, dirs_a, cb["he"])
            t1 = np.mean([decs[i].score(S1[:, i:i + 1], zb_he[:, i]) for i in range(8)])
            # target TRAIN std calibration (b's own train mean/std at a's layer l*_a)
            calib_b = [std_fit(view(cb["bodies"], la, i)[cb["tr"]]) for i in range(8)]
            # Tier 2: source w + source decoder, target-train std
            S2 = score_matrix(cb["bodies"], la, dirs_a, cb["he"], calib_stats=calib_b)
            t2 = np.mean([decs[i].score(S2[:, i:i + 1], zb_he[:, i]) for i in range(8)])
            # Tier 3 ceiling: a's directions on b, target-train std, REFIT decoder on b labels
            S3tr = score_matrix(cb["bodies"], la, dirs_a, cb["tr"], calib_stats=calib_b)
            S3he = score_matrix(cb["bodies"], la, dirs_a, cb["he"], calib_stats=calib_b)
            t3 = np.mean([
                LogisticRegression(max_iter=1000)
                .fit(S3tr[:, i:i + 1], cb["z"][cb["tr"], i])
                .score(S3he[:, i:i + 1], zb_he[:, i]) for i in range(8)])
            out.append(dict(a=a, b=b, tier1=float(t1), tier2=float(t2), tier3=float(t3)))
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


# ---- controls -------------------------------------------------------------- #
def control_perm(Str, She, ztr, zhe, rng):
    zt = np.column_stack([rng.permutation(ztr[:, j]) for j in range(ztr.shape[1])])
    zh = np.column_stack([rng.permutation(zhe[:, j]) for j in range(zhe.shape[1])])
    _, head, kf, ks, _ = same_seed_sweep(Str, She, zt, zh)
    worst = max(max(h["func_held"] for h in head.values()),
                CHANCE + max(h["state_held"] for h in head.values()))
    return dict(kf=kf, ks=ks, max_func=max(h["func_held"] for h in head.values()),
                ok=(kf is None and ks is None))


def control_randdir(bodies, z, lstar, tr, he, rng):
    dirs = []
    for i in range(8):
        mu, sd = std_fit(view(bodies, lstar, i)[tr])   # keep train std, randomize direction
        w = rng.standard_normal(192)
        dirs.append((mu, sd, w / np.linalg.norm(w)))
    Str = score_matrix(bodies, lstar, dirs, tr)
    She = score_matrix(bodies, lstar, dirs, he)
    _, head, kf, ks, _ = same_seed_sweep(Str, She, z[tr], z[he])
    return dict(kf=kf, ks=ks, ok=(kf is None and ks is None))


# ---- driver --------------------------------------------------------------- #
def run(smoke=False):
    os.makedirs(OUT, exist_ok=True)
    rng = default_rng(PROBE_SEED)
    tr, he = split_idx()
    assert len(set(tr.tolist()) & set(he.tolist())) == 0, "split leak"
    seeds = [0] if smoke else SEEDS
    t0 = time.time()

    cache, summary, sweeps, controls, pf_rows = {}, [], [], [], []
    for s in seeds:
        bg, z = load(s, "gen")
        bt, _ = load(s, "twin")
        lstar = layer_select(bg, z, tr)
        dirs = readout_dirs(bg, z, lstar, tr)
        cache[s] = dict(bodies=bg, z=z, tr=tr, he=he, lstar=lstar, dirs=dirs)

        Sg_tr = score_matrix(bg, lstar, dirs, tr)
        Sg_he = score_matrix(bg, lstar, dirs, he)
        rows, head, kf, ks, kd = same_seed_sweep(Sg_tr, Sg_he, z[tr], z[he])

        # twin floor: twin layer/dirs fit on twin train
        lstar_t = layer_select(bt, z, tr)
        dirs_t = readout_dirs(bt, z, lstar_t, tr)
        St_tr = score_matrix(bt, lstar_t, dirs_t, tr)
        St_he = score_matrix(bt, lstar_t, dirs_t, he)
        _, head_t, kf_t, ks_t, _ = same_seed_sweep(St_tr, St_he, z[tr], z[he])

        perm = control_perm(Sg_tr, Sg_he, z[tr], z[he], rng)
        rdir = control_randdir(bg, z, lstar, tr, he, rng)
        controls_ok = perm["ok"] and rdir["ok"]
        branch = probe1_branch(kf, ks, controls_ok)

        # paired fiber on the k=4 func headline subset (reference)
        pf = paired_fiber(Sg_he, z[he], head[4]["S_func"], rng)
        pf_rows.append(dict(seed=s, **pf))

        for r in rows:
            sweeps.append(dict(seed=s, body="gen", k=r["k"], S="".join(map(str, r["S"])),
                               func_held=round(r["func_held"], 4), state_held=round(r["state_held"], 4)))
        summary.append(dict(seed=s, lstar=lstar, kf=kf, ks=ks, kd=kd, branch=branch,
                            twin_kf=kf_t, twin_ks=ks_t,
                            max_func_gen=round(max(h["func_held"] for h in head.values()), 4),
                            max_func_twin=round(max(h["func_held"] for h in head_t.values()), 4)))
        controls.append(dict(seed=s, perm_max_func=round(perm["max_func"], 4),
                             perm_ok=perm["ok"], randdir_ok=rdir["ok"], controls_ok=controls_ok))
        print(f"[seed {s}] l*={lstar} branch={branch} k_func={kf} k_state={ks} "
              f"max_func gen={summary[-1]['max_func_gen']:.3f} twin={summary[-1]['max_func_twin']:.3f} "
              f"perm_ok={perm['ok']} randdir_ok={rdir['ok']} pf={pf['branch']}", flush=True)

    # cross-seed (needs >=2 seeds)
    xbranch, xcounts, xpairs = None, None, []
    if len(seeds) >= 2:
        xpairs = cross_seed(cache)
        xbranch, xcounts = cross_seed_branch(xpairs)
        for p in xpairs:
            print(f"[cross {p['a']}->{p['b']}] t1={p['tier1']:.3f} t2={p['tier2']:.3f} t3={p['tier3']:.3f}",
                  flush=True)
        print(f"[cross-seed] branch={xbranch} pass-counts(t1,t2,t3)={xcounts}", flush=True)

    dt = time.time() - t0
    agg = [s["branch"] for s in summary]
    print(f"\n=== {'SMOKE' if smoke else 'FULL'} done {dt:.1f}s | "
          f"probe1 branches={agg} | cross-seed={xbranch} ===", flush=True)

    if not smoke:
        _write_csv(f"{OUT}/same_seed_subset_sweep.csv", sweeps)
        _write_csv(f"{OUT}/same_seed_kfunc_kstate_summary.csv", summary)
        _write_csv(f"{OUT}/same_seed_controls.csv", controls)
        _write_csv(f"{OUT}/paired_fiber_boundary_audit.csv", pf_rows)
        _write_csv(f"{OUT}/cross_seed_transfer.csv", xpairs)
        with open(f"{OUT}/branch_adjudication.json", "w") as f:
            json.dump(dict(probe1=summary, probe1_aggregate=agg, controls=controls,
                           cross_seed_branch=xbranch, cross_seed_counts=xcounts,
                           cross_seed_pairs=xpairs, paired_fiber=pf_rows,
                           wall_clock_s=round(dt, 1)), f, indent=2, default=str)
        print(f"[written] {OUT}", flush=True)
    else:
        print(f"[smoke] extrapolated full ~= {dt * len(SEEDS) / len(seeds) * 1.4:.0f}s "
              f"(3 seeds + cross + controls)", flush=True)
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
    a = ap.parse_args()
    run(smoke=a.smoke)
