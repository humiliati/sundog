#!/usr/bin/env python
"""chatv2 Phase 7 - coupled-latent closure probe (frozen PHASE7 spec).

Reuses the Phase-2 shadow-set machinery. Two registered targets per shadow set S
of z-readouts:
  k_func  = smallest |S| determining the hidden source u (the closure functional),
            mean held label acc >= 0.70 AND above the selection-corrected null.
  k_state = smallest |S| reconstructing the omitted individual z_j, mean held LABEL
            accuracy >= 0.70 (NOT Phase-2 score FVE).
Headline positive = k_func << k_state.

Modes:
  --mode coupled  : load results/chatv2/{--dir}/seed{s}/bodies, target
                    the saved hidden source u.
  --mode control  : load the Phase-2 uncoupled pair-XOR bodies (phase1-seedstab) and
                    target a FROZEN INDEPENDENT u_null ~ Bernoulli(0.5)^(N x 3),
                    default_rng(7007+seed). Expected k_func = none (the runner must
                    not hallucinate hidden-source closure on an independent target).
"""
import argparse, itertools, json, os, sys, time
import numpy as np
from numpy.random import default_rng
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatv2_phase2_shadowset as p2
from sklearn.linear_model import LogisticRegression

ROOT = "C:/Users/hughe/Dev/sundog"
RESULT_BASE = os.path.join(ROOT, "results", "chatv2")
THRESH = 0.70
R_NULL = 30


def _sweep(Str, She, ztr, zhe, Utr, Uhe, H, ku):
    """per-k best-subset held func(u) and state(z_j), train-selected."""
    head = {}
    for k in range(1, H):
        rows = []
        for S in itertools.combinations(range(H), k):
            S = list(S); Xt, Xh = Str[:, S], She[:, S]
            ftr, fhe = [], []
            for l in range(ku):
                c = LogisticRegression(max_iter=500).fit(Xt, Utr[:, l])
                ftr.append(c.score(Xt, Utr[:, l])); fhe.append(c.score(Xh, Uhe[:, l]))
            J = [j for j in range(H) if j not in S]
            st, sh = [], []
            for j in J:
                c = LogisticRegression(max_iter=500).fit(Xt, ztr[:, j])
                st.append(c.score(Xt, ztr[:, j])); sh.append(c.score(Xh, zhe[:, j]))
            rows.append((np.mean(ftr), np.mean(fhe), np.mean(st), np.mean(sh)))
        sf = max(rows, key=lambda r: r[0]); ss = max(rows, key=lambda r: r[2])
        head[k] = (float(sf[1]), float(ss[3]))   # (func_held, state_held)
    return head


def probe_u(bodies, z, U, label, R=R_NULL):
    tr, he = p2.split_idx()
    lstar = p2.layer_select(bodies, z, tr)
    dirs = p2.readout_dirs(bodies, z, lstar, tr)
    Str = p2.score_matrix(bodies, lstar, dirs, tr)
    She = p2.score_matrix(bodies, lstar, dirs, he)
    H, ku, d = z.shape[1], U.shape[1], bodies.shape[-1]
    head = _sweep(Str, She, z[tr], z[he], U[tr], U[he], H, ku)
    # selection-corrected null for the u-target (random directions)
    rng = default_rng(0); nf = []
    for _ in range(R):
        rd = []
        for i in range(H):
            mu, sd = p2.std_fit(p2.view(bodies, lstar, i)[tr])
            w = rng.standard_normal(d); rd.append((mu, sd, w / np.linalg.norm(w)))
        h = _sweep(p2.score_matrix(bodies, lstar, rd, tr),
                   p2.score_matrix(bodies, lstar, rd, he),
                   z[tr], z[he], U[tr], U[he], H, ku)
        nf.append(max(v[0] for v in h.values()))
    nf_func = float(np.percentile(nf, 95))
    kf = next((k for k in range(1, H) if head[k][0] > max(THRESH, nf_func)), None)
    ks = next((k for k in range(1, H) if head[k][1] >= THRESH), None)
    return dict(label=label, lstar=int(lstar), nf_func=round(nf_func, 3),
                k_func=kf, k_state=ks,
                table={k: (round(head[k][0], 3), round(head[k][1], 3)) for k in head})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["coupled", "control"], required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dir", default="phase7-coupled-latent",
                    help="results/chatv2 subdirectory for coupled inputs and probe output")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    result_base = os.path.abspath(RESULT_BASE)
    result_dir = os.path.abspath(os.path.join(result_base, a.dir))
    if result_dir != result_base and not result_dir.startswith(result_base + os.sep):
        raise SystemExit(f"--dir must stay inside {result_base}: {a.dir}")
    out = []
    for s in seeds:
        if a.mode == "coupled":
            d = np.load(os.path.join(result_dir, f"seed{s}", "bodies", "H8_gen.npz"),
                        allow_pickle=True)
            bodies, z, U = d["bodies"], d["z"].astype(int), d["u"].astype(int)
            label = f"coupled-seed{s}"
        else:  # control: Phase-2 uncoupled bodies + frozen independent u_null
            d = np.load(os.path.join(result_base, "phase1-seedstab", f"seed{s}",
                                     "bodies", "H8_gen.npz"), allow_pickle=True)
            bodies, z = d["bodies"], d["z"].astype(int)
            U = default_rng(7007 + s).integers(0, 2, size=(z.shape[0], 3))
            label = f"control-unull-seed{s}"
        r = probe_u(bodies, z, U, label)
        out.append(r)
        print(f"[{label}] l*={r['lstar']} k_func={r['k_func']} k_state={r['k_state']} "
              f"nf_func={r['nf_func']} | func/state by k: {r['table']}", flush=True)
    odir = result_dir
    os.makedirs(odir, exist_ok=True)
    fn = f"{odir}/{a.mode}_probe.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[written] {fn}", flush=True)


if __name__ == "__main__":
    main()
