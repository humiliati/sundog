#!/usr/bin/env python
"""Frozen test for H3-PC-B (scripts/h3_probe_ceiling_mnist.py) — apparatus pins at reduced size.

Per the frozen addendum sec 6: reduced pool/test = the FIRST 4000/2000 rows of the FULL pool/test
index ranges of the banked 70k permutation (subset rule, binding). Retrains the body ONCE per process
and reuses its reps for an in-process battery rerun. Pins: data shape gate, continuity gates
(CNN acc, banked-probe theta, permutation control), delta-calibration convergence, liveness booleans,
byte-identical battery readouts + battery-only verdict letter across the rerun. FULL-run letter
equality is NOT asserted. The MI leg is full-size-only (descriptive) and not exercised here.
Run: python scripts/test_h3_probe_ceiling_mnist.py   (~15-25 min CPU, deterministic)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, "scripts")
import shadow_pooled_mnist as sb                   # noqa: E402
import h3_probe_ceiling as pc1                     # noqa: E402
import h3_probe_ceiling_mnist as pcb               # noqa: E402
from sklearn.linear_model import Ridge             # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H3-PC-B frozen test — apparatus pins at reduced size (pool 4000 / test 2000, subset rows):\n")

# data (verbatim loader; fallback forbidden)
from sklearn.datasets import fetch_openml  # noqa: E402

X_raw, y_raw = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,
                            parser="liac-arff")
check("data shape gate (70000, 784)", X_raw.shape == (70000, 784), str(X_raw.shape))
X_raw = X_raw.astype(np.float32) / 255.0
y_all = y_raw.astype(np.int64)
imgs_all = X_raw.reshape(-1, 28, 28)

# banked streams (two fresh RandomState(1234) instances) + split + body, as the main script
rs1 = np.random.RandomState(1234)
perm_full = rs1.permutation(70000)
sub_idx = perm_full[:pcb.N_SUB]
imgs, labels = imgs_all[sub_idx], y_all[sub_idx]
rs2 = np.random.RandomState(1234)
thetas = rs2.uniform(-sb.ROT_RANGE, sb.ROT_RANGE, size=pcb.N_SUB).astype(np.float32)
perm12 = torch.randperm(pcb.N_SUB, generator=torch.Generator().manual_seed(1234))
probe_idx = perm12[:pcb.N_PROBE_BANKED].numpy()
train_idx = perm12[pcb.N_PROBE_BANKED:].numpy()

np.random.seed(1234)
torch.manual_seed(1234)
model = sb.TinyCNN(n_classes=10, ch=16)
rot_tr = sb.rotate_batch(imgs[train_idx], thetas[train_idx])
sb.train_cnn(model, torch.from_numpy(rot_tr[:, None, :, :].astype(np.float32)),
             torch.from_numpy(labels[train_idx]), pcb.EPOCHS, 128, 1e-3, torch.device("cpu"))

# continuity gates
z_banked = pcb.gap_reps(model, imgs[probe_idx], thetas[probe_idx])
rot_pr = sb.rotate_batch(imgs[probe_idx], thetas[probe_idx])
with torch.no_grad():
    logits = model(torch.from_numpy(rot_pr[:, None, :, :].astype(np.float32)))
cnn_acc = float((logits.argmax(1).numpy() == labels[probe_idx]).mean())
post_theta = sb.probe_theta_r2(z_banked, thetas[probe_idx])
yperm = labels[probe_idx].copy(); rs2.shuffle(yperm)
tperm = thetas[probe_idx].copy(); rs2.shuffle(tperm)
perm_theta = sb.probe_theta_r2(z_banked, tperm)
check("continuity: CNN acc in 0.83±0.05", 0.78 <= cnn_acc <= 0.88, f"acc={cnn_acc:.3f}")
check("continuity: banked-probe post-GAP theta in 0.342±0.10", 0.242 <= post_theta <= 0.442,
      f"theta={post_theta:.3f}")
check("continuity: permutation control |R2| <= 0.05", abs(perm_theta) <= 0.05, f"{perm_theta:+.3f}")

# reduced pool/test = FIRST 4000/2000 rows of the full index ranges (subset rule)
pool_idx = perm_full[pcb.POOL_LO:pcb.POOL_HI][:4000]
test_idx = perm_full[pcb.POOL_HI:pcb.TEST_HI][:2000]
th_pool_f = np.random.default_rng(pcb.POOL_TH_SEED).uniform(-30, 30, pcb.POOL_HI - pcb.POOL_LO)
th_test_f = np.random.default_rng(pcb.TEST_TH_SEED).uniform(-30, 30, pcb.TEST_HI - pcb.POOL_HI)
th_pool = th_pool_f[:4000].astype(np.float32)      # prefix of the FULL theta stream (subset rule)
th_test = th_test_f[:2000].astype(np.float32)
z_pool = pcb.gap_reps(model, imgs_all[pool_idx], th_pool)
z_test = pcb.gap_reps(model, imgs_all[test_idx], th_test)

BASE = pc1.cv_r2(Ridge(alpha=1.0), z_pool, th_pool)
check("positive control at reduced size (BASE >= 0.15)", BASE >= 0.15, f"BASE={BASE:+.4f}")

# delta calibration converges at reduced size
vdir = np.random.default_rng(pcb.VDIR_SEED).standard_normal(z_pool.shape[1])
vdir /= np.linalg.norm(vdir)
g_pool = (th_pool - th_pool.mean()) / th_pool.std()


def inject(alpha):
    return z_pool + alpha * g_pool[:, None] * vdir[None, :]


alphas = {}
for d in pcb.DELTAS:
    lo, hi = 0.0, 3.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if pc1.cv_r2(Ridge(alpha=1.0), inject(mid), th_pool) - BASE < d:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    ach = pc1.cv_r2(Ridge(alpha=1.0), inject(a), th_pool) - BASE
    alphas[d] = a
    check(f"delta calibration {d} converges (|achieved-target| <= 0.01)", abs(ach - d) <= 0.01,
          f"alpha={a:.5f} achieved={ach:+.4f}")

# liveness machinery yields booleans (values size-dependent, not pinned)
z_inj = {d: inject(alphas[d]) for d in pcb.DELTAS}
fams = pc1.battery()
real_cv = {f: [(cfg, mk, pc1.cv_r2(mk(), z_pool, th_pool)) for cfg, mk in fam]
           for f, fam in fams.items()}
live = {}
for f in fams:
    if f == "P1_ridge":
        continue
    live[f] = {}
    for d in pcb.DELTAS:
        hit = False
        for cfg, mk, rcv in real_cv[f]:
            if pc1.cv_r2(mk(), z_inj[d], th_pool) - rcv >= 0.5 * d:
                hit = True
                break
        live[f][d] = hit
check("liveness machinery yields a boolean per member per delta",
      all(isinstance(b, bool) for dd in live.values() for b in dd.values()), str(live))


# battery readouts + battery-only letter, byte-identical across in-process rerun
def run_battery():
    rd = {}
    for f in fams:
        cfg, mk, cv = max(real_cv[f], key=lambda t: t[2])
        sp = pc1.split_r2(mk(), z_pool, th_pool, z_test, th_test)
        rd[f] = (cfg, cv, sp)
    PRE_proxy = 1.0                                 # PRE_pool is full-size-only; (a) not testable here
    a_hit = any(cv >= PRE_proxy - 0.05 and sp >= 0.8 * (PRE_proxy - 0.05) for _, cv, sp in rd.values())
    b_hit = all(cv <= BASE + 0.05 and sp <= BASE + 0.05 for _, cv, sp in rd.values())
    return rd, ("a" if a_hit else "b" if b_hit else "c")


rd1, letter1 = run_battery()
rd2, letter2 = run_battery()
for f in rd1:
    print(f"    {f:13s} [{rd1[f][0]:12s}]  pool-CV={rd1[f][1]:+.4f}  split={rd1[f][2]:+.4f}")
check("battery readouts byte-identical across in-process rerun",
      all(rd1[f][1] == rd2[f][1] and rd1[f][2] == rd2[f][2] for f in rd1))
check("battery-only verdict letter stable across rerun and in {b,c}",
      letter1 == letter2 and letter1 in ("b", "c"), f"letter={letter1}")

print(f"\n{'ALL PASS — H3-PC-B apparatus pinned at reduced size (full-run verdict NOT asserted here).' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
