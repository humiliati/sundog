#!/usr/bin/env python
"""H3-PC-B — the probe-ceiling battery on the real-body MNIST-rotation GAP shadow.

Runs the FROZEN addendum docs/atlas/H3_PROBE_CEILING_MNIST_ADDENDUM.md (second leg of HS4): re-derive
the banked TinyCNN (continuity-gated, NOT byte-asserted), then audit the banked "GAP partially
attenuates theta" claim (post 0.342 vs pre 0.623) with leg 1's calibrated battery on new disjoint
pool/test draws. Delta-injection floors on a nonzero baseline; MI descriptive-only; outcomes V/a/b/c
with the 0.47-adjudication band. R2 UNCLIPPED everywhere. NOT public-eligible.
Attribution: leg 1 (h3_probe_ceiling.py) + Substrate B (shadow_pooled_mnist.py); amnesic probing;
V-information; LEACE; KSG. Run: python scripts/h3_probe_ceiling_mnist.py  (CPU, ~1.5-3 h)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import glob
import hashlib
import json
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, "scripts")
import shadow_pooled_mnist as sb                   # noqa: E402  (substrate: TinyCNN, probes, rotate)
import h3_probe_ceiling as pc1                     # noqa: E402  (leg-1 battery, IDENTICAL by import)
from sklearn.linear_model import Ridge             # noqa: E402
from sklearn.preprocessing import StandardScaler   # noqa: E402
from sklearn.decomposition import PCA              # noqa: E402

SEED = 1234
N_SUB, N_PROBE_BANKED, EPOCHS = 12000, 1800, 20
POOL_LO, POOL_HI, TEST_HI = 12000, 32000, 42000    # banked-permutation index ranges (prereg sec 2)
POOL_TH_SEED, TEST_TH_SEED, VDIR_SEED = SEED + 100001, SEED + 110001, SEED + 120001
DELTAS = [0.05, 0.10]
N_SHUF_MI, MI_SUB = 49, 5000                       # MI trimmed: descriptive-only


def gap_reps(model, imgs, thetas, batch=512):
    """Rotate then extract eval-mode GAP reps, batched (and pre-pool if asked)."""
    rot = sb.rotate_batch(imgs, thetas)
    X = torch.from_numpy(rot[:, None, :, :].astype(np.float32))
    outs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch):
            fmap = model.feature_map(X[s:s + batch])
            outs.append(model.gap(fmap).numpy())
    return np.concatenate(outs)


def prepool_reps(model, imgs, thetas, batch=512):
    rot = sb.rotate_batch(imgs, thetas)
    X = torch.from_numpy(rot[:, None, :, :].astype(np.float32))
    outs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch):
            fmap = model.feature_map(X[s:s + batch])
            outs.append(fmap.reshape(len(fmap), -1).numpy())
    return np.concatenate(outs)


def void(reason, extra=None):
    print(f"\nVERDICT: V (VOID — {reason})")
    out = {"verdict": "V", "void_reason": reason}
    if extra:
        out.update(extra)
    p = Path("results/atlas/h3/probe_ceiling_mnist_result.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return 1


def main():
    print("=" * 96)
    print("H3-PC-B — probe-ceiling battery on the MNIST-rotation GAP shadow (frozen addendum)")
    print("=" * 96)

    # ---- data: verbatim loader kwargs; fallback FORBIDDEN ---- #
    try:
        from sklearn.datasets import fetch_openml
        X_raw, y_raw = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,
                                    parser="liac-arff")
    except Exception as e:
        return void(f"cached openml fetch failed ({type(e).__name__}); 8x8 fallback is forbidden")
    if X_raw.shape != (70000, 784):
        return void(f"data shape gate failed: {X_raw.shape}")
    X_raw = X_raw.astype(np.float32) / 255.0
    y_all = y_raw.astype(np.int64)
    data_sha = hashlib.sha256(X_raw.tobytes()).hexdigest()
    arff = sorted(glob.glob(os.path.expanduser("~/scikit_learn_data/openml/**/mnist_784.arff.gz"),
                            recursive=True))
    arff_sha = hashlib.sha256(Path(arff[0]).read_bytes()).hexdigest() if arff else None
    print(f"  data: X{X_raw.shape}  sha256(X)={data_sha[:16]}…  arff={arff_sha[:16] if arff_sha else None}…")
    imgs_all = X_raw.reshape(-1, 28, 28)

    # ---- banked substrate streams (TWO fresh RandomState(1234) instances; prereg sec 2) ---- #
    rs1 = np.random.RandomState(SEED)              # instance #1 (loader)
    perm_full = rs1.permutation(70000)
    sub_idx = perm_full[:N_SUB]
    imgs, labels = imgs_all[sub_idx], y_all[sub_idx]
    rs2 = np.random.RandomState(SEED)              # instance #2 (main)
    thetas = rs2.uniform(-sb.ROT_RANGE, sb.ROT_RANGE, size=N_SUB).astype(np.float32)

    # split: dedicated torch generator (byte-reproducible)
    perm12 = torch.randperm(N_SUB, generator=torch.Generator().manual_seed(SEED))
    probe_idx = perm12[:N_PROBE_BANKED].numpy()
    train_idx = perm12[N_PROBE_BANKED:].numpy()

    # ---- body: global-torch order pinned = manual_seed -> init -> train randperms ---- #
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = sb.TinyCNN(n_classes=10, ch=16)
    rot_tr = sb.rotate_batch(imgs[train_idx], thetas[train_idx])
    Xtr = torch.from_numpy(rot_tr[:, None, :, :].astype(np.float32))
    ytr = torch.from_numpy(labels[train_idx])
    print(f"  retraining TinyCNN(ch=16): train={len(Xtr)} epochs={EPOCHS} …")
    sb.train_cnn(model, Xtr, ytr, EPOCHS, 128, 1e-3, torch.device("cpu"))

    # ---- continuity gates on the banked probe set ---- #
    z_banked = gap_reps(model, imgs[probe_idx], thetas[probe_idx])
    rot_pr = sb.rotate_batch(imgs[probe_idx], thetas[probe_idx])
    with torch.no_grad():
        logits = model(torch.from_numpy(rot_pr[:, None, :, :].astype(np.float32)))
    cnn_acc = float((logits.argmax(1).numpy() == labels[probe_idx]).mean())
    theta_banked = thetas[probe_idx]
    post_theta = sb.probe_theta_r2(z_banked, theta_banked)
    # banked control consumption mirrored on instance #2: y-shuffle then theta-shuffle
    yperm = labels[probe_idx].copy(); rs2.shuffle(yperm)
    tperm = theta_banked.copy(); rs2.shuffle(tperm)
    perm_theta = sb.probe_theta_r2(z_banked, tperm)
    print(f"  continuity: cnn_acc={cnn_acc:.3f} (0.83±0.05)  post_theta={post_theta:.3f} "
          f"(0.342±0.10)  perm_ctrl={perm_theta:+.3f} (|.|<=0.05)")
    if not (0.78 <= cnn_acc <= 0.88):
        return void(f"CNN-acc continuity failed: {cnn_acc:.3f}", {"data_sha256": data_sha})
    if not (0.242 <= post_theta <= 0.442):
        return void(f"post-GAP theta continuity failed: {post_theta:.3f}", {"data_sha256": data_sha})
    if abs(perm_theta) > 0.05:
        return void(f"permutation control failed: {perm_theta:.3f}", {"data_sha256": data_sha})

    # ---- new disjoint evaluation draws ---- #
    pool_idx, test_idx = perm_full[POOL_LO:POOL_HI], perm_full[POOL_HI:TEST_HI]
    th_pool = np.random.default_rng(POOL_TH_SEED).uniform(-30, 30, len(pool_idx)).astype(np.float32)
    th_test = np.random.default_rng(TEST_TH_SEED).uniform(-30, 30, len(test_idx)).astype(np.float32)
    z_pool = gap_reps(model, imgs_all[pool_idx], th_pool)
    z_test = gap_reps(model, imgs_all[test_idx], th_test)

    # ---- anchors: PRE_pool (substrate dim-fair probe) and BASE (P1 battery convention) ---- #
    pre_feats = prepool_reps(model, imgs_all[pool_idx], th_pool)
    PRE = sb.probe_theta_r2(pre_feats, th_pool)
    del pre_feats
    BASE = pc1.cv_r2(Ridge(alpha=1.0), z_pool, th_pool)
    print(f"  anchors: PRE_pool={PRE:+.4f}  BASE={BASE:+.4f}  (joint-separation gate >= 0.15)")
    if PRE - BASE < 0.15:
        return void(f"joint-separation gate failed: PRE-BASE={PRE - BASE:.3f}",
                    {"data_sha256": data_sha, "PRE_pool": PRE, "BASE": BASE})
    if BASE < 0.20:
        return void(f"positive control failed: BASE={BASE:.3f}", {"data_sha256": data_sha})

    # ---- delta-injection calibration ---- #
    vdir = np.random.default_rng(VDIR_SEED).standard_normal(z_pool.shape[1])
    vdir /= np.linalg.norm(vdir)
    g_pool = (th_pool - th_pool.mean()) / th_pool.std()

    def inject(alpha):
        return z_pool + alpha * g_pool[:, None] * vdir[None, :]

    def bisect(target_delta):
        lo, hi = 0.0, 3.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if pc1.cv_r2(Ridge(alpha=1.0), inject(mid), th_pool) - BASE < target_delta:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)
        return a, pc1.cv_r2(Ridge(alpha=1.0), inject(a), th_pool) - BASE

    alphas, gates = {}, {"cnn_acc": cnn_acc, "post_theta_banked": post_theta,
                         "perm_ctrl": perm_theta, "PRE_pool": PRE, "BASE": BASE}
    for d in DELTAS:
        a, ach = bisect(d)
        alphas[d] = a
        ok = abs(ach - d) <= 0.01
        gates[f"calibration_d{d}"] = {"alpha": a, "achieved_delta": ach, "ok": ok}
        print(f"  calibration delta={d}: alpha={a:.4f} achieved={ach:+.4f} [{'OK' if ok else 'FAIL'}]")
        if not ok:
            return void(f"delta calibration {d} failed", {"data_sha256": data_sha, **gates})
    z_inj = {d: inject(alphas[d]) for d in DELTAS}

    # ---- per-config real CVs (cached: selection + liveness baselines share them) ---- #
    fams = pc1.battery()
    real_cv = {f: [(cfg, mk, pc1.cv_r2(mk(), z_pool, th_pool)) for cfg, mk in fam]
               for f, fam in fams.items()}

    # ---- per-member liveness at each delta (blindness excuses silence, never speech) ---- #
    liveness, floors = {}, {}
    for f, fam in fams.items():
        if f == "P1_ridge":
            floors[f] = 0.05
            liveness[f] = {str(d): True for d in DELTAS}
            continue
        liveness[f] = {}
        for d in DELTAS:
            live = False
            for cfg, mk, rcv in real_cv[f]:
                if pc1.cv_r2(mk(), z_inj[d], th_pool) - rcv >= 0.5 * d:
                    live = True
                    break
            liveness[f][str(d)] = live
        floors[f] = (0.05 if liveness[f]["0.05"] else (0.10 if liveness[f]["0.1"] else None))
        tag = "MEMBER-BLIND" if floors[f] is None else f"floor={floors[f]}"
        print(f"  liveness {f:13s}: d=0.05 {liveness[f]['0.05']}  d=0.10 {liveness[f]['0.1']} -> {tag}")

    # ---- battery readout on the real reps ---- #
    print("\n  battery on the REAL GAP reps (pool-CV -> once-touched frozen split):")
    readout = {}
    for f in fams:
        cfg, mk, cv = max(real_cv[f], key=lambda t: t[2])
        sp = pc1.split_r2(mk(), z_pool, th_pool, z_test, th_test)
        readout[f] = {"config": cfg, "pool_cv": cv, "split": sp, "member_blind": floors[f] is None}
        print(f"    {f:13s} [{cfg:12s}]  pool-CV={cv:+.4f}  split={sp:+.4f}"
              f"{'   (MEMBER-BLIND label)' if floors[f] is None else ''}")

    # ---- descriptive MI (non-gating; real + 0.10-delta only; 49 shuffles) ---- #
    sub = np.random.default_rng(pc1.MI_SUB_SEED).choice(len(z_pool), MI_SUB, replace=False)
    th_sub = th_pool[sub]
    th_std = (th_sub - th_sub.mean()) / th_sub.std()
    shuf_seeds = np.random.default_rng(pc1.SHUF_SEED).integers(0, 2**31, N_SHUF_MI)

    def pca_scores(Z, k):
        Zs = StandardScaler().fit_transform(Z[sub])
        return StandardScaler().fit_transform(PCA(n_components=k, random_state=0).fit_transform(Zs))

    mi = {}
    for k in pc1.PCA_KS:
        S_r, S_i = pca_scores(z_pool, k), pca_scores(z_inj[0.10], k)
        nulls = []
        for s in shuf_seeds:
            perm = np.random.default_rng(int(s)).permutation(MI_SUB)
            nulls.append(pc1.ksg_mi(S_r, th_std[perm]))
        mi[k] = {"mi_real": pc1.ksg_mi(S_r, th_std), "mi_inj010": pc1.ksg_mi(S_i, th_std),
                 "null_max_real": float(np.max(nulls))}
        print(f"  MI (descriptive) PCA-k={k:2d}: real={mi[k]['mi_real']:+.4f}  "
              f"inj0.10={mi[k]['mi_inj010']:+.4f}  null-max={mi[k]['null_max_real']:+.4f}")

    # ---- learning curve (non-gating) ---- #
    best_fam = max(readout, key=lambda f: readout[f]["pool_cv"])
    best_mk = dict((cfg, mk) for cfg, mk, _ in real_cv[best_fam])[readout[best_fam]["config"]]
    lc_rng = np.random.default_rng(pc1.LC_SEED)
    lc = {}
    for n in (2000, 5000, 10000, 20000):
        idx = lc_rng.choice(len(z_pool), n, replace=False) if n < len(z_pool) else np.arange(len(z_pool))
        lc[n] = {"best": pc1.cv_r2(best_mk(), z_pool[idx], th_pool[idx]),
                 "ridge": pc1.cv_r2(Ridge(alpha=1.0), z_pool[idx], th_pool[idx])}
    print("  learning curve: " + "  ".join(f"n={n}: {lc[n]['best']:+.3f}/{lc[n]['ridge']:+.3f}"
                                           for n in lc))

    # ---- verdict (prereg sec 5; precedence V > a > b > c) ---- #
    a_bar = PRE - 0.05
    a_hit = [f for f, r in readout.items() if r["pool_cv"] >= a_bar and r["split"] >= 0.8 * a_bar]
    unrep = [f for f, r in readout.items() if r["pool_cv"] >= a_bar and r["split"] < 0.8 * a_bar]
    b_hit = all(r["pool_cv"] <= BASE + 0.05 and r["split"] <= BASE + 0.05 for r in readout.values())
    sub_outcomes, ceiling, adj47 = [], None, None
    if a_hit:
        verdict = "a"
    elif b_hit and not unrep:
        verdict = "b"
    else:
        verdict = "c"
        counted = {f: r["split"] for f, r in readout.items()
                   if r["pool_cv"] >= BASE + 0.05 and r["split"] >= BASE + 0.05}
        if counted:
            ceiling = max(counted.values())
        else:
            ceiling = BASE
            sub_outcomes.append("CV-ONLY-EXCEEDANCE")
        for f in unrep:
            sub_outcomes.append(f"UNREPLICATED-POSITIVE:{f}")
        adj47 = ("superseded downward" if ceiling < 0.42 else
                 "confirmed by a calibrated battery" if ceiling <= 0.52 else "superseded upward")

    print("\n" + "=" * 96)
    desc = {"a": f"ATTENUATION ILLUSORY (vs the linearly-readable anchor PRE_pool={PRE:.3f}) — "
                 "correction owed to the Substrate-B section.",
            "b": "ATTENUATION BAND CERTIFIED (clean-null SUCCESS) — no probe family recovers "
                 "meaningfully more theta than the banked linear readout; 0.47 superseded downward.",
            "c": f"BOUNDED-PARTIAL (ceiling raised): ceiling={ceiling if ceiling is None else round(ceiling, 4)}, "
                 f"resist band [ceiling, PRE_pool={PRE:.3f}]; the un-receipted 0.47 is {adj47}."}
    print(f"VERDICT: ({verdict}) {desc[verdict]}")
    if sub_outcomes:
        print(f"  sub-outcomes: {sub_outcomes}")
    print("=" * 96)

    out = {"prereg": "docs/atlas/H3_PROBE_CEILING_MNIST_ADDENDUM.md",
           "data_sha256": data_sha, "arff_sha256": arff_sha,
           "params": {"epochs": EPOCHS, "pool_range": [POOL_LO, POOL_HI],
                      "test_range": [POOL_HI, TEST_HI], "pool_theta_seed": POOL_TH_SEED,
                      "test_theta_seed": TEST_TH_SEED, "vdir_seed": VDIR_SEED, "deltas": DELTAS,
                      "mi_shuffles": N_SHUF_MI, "mi_sub": MI_SUB},
           "gates": gates, "alphas": alphas, "liveness": liveness, "floors": floors,
           "readout": readout, "mi_descriptive": mi, "learning_curve": lc,
           "verdict": verdict, "sub_outcomes": sub_outcomes, "ceiling": ceiling,
           "adjudication_047": adj47}
    p = Path("results/atlas/h3/probe_ceiling_mnist_result.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                            else int(o) if isinstance(o, np.integer)
                            else bool(o) if isinstance(o, np.bool_) else o))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
