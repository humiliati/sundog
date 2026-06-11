#!/usr/bin/env python
"""Frozen test for H3-SP (scripts/shadow_stale_projection.py) — pins the APPARATUS at reduced size.

Per the frozen prereg section 6: reduced train/deploy/control = the FIRST 2000/2500/1000 rows of the
FULL-SIZE draws gen_hs2(8000,"train",101235) / gen_hs2(10000,2.0,111235) / gen_hs2(4000,0.0,121235)
(strict subsets — gen_hs2 at a smaller n is a DIFFERENT realization). One init seed (131235), BOTH
heads. Pins: C0 wash + MI gates on the reduced deploy rows, train-fit gate, lam0-control gate, and
byte-identical cell readouts across an in-process rerun. The FULL run's verdict is NOT asserted here.
Run: python scripts/test_shadow_stale_projection.py    (~5-10 min CPU, deterministic)
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

sys.path.insert(0, "scripts")
import shadow_pooled_synthetic_v2 as v2            # noqa: E402
import shadow_stale_projection as sp               # noqa: E402
from sklearn.linear_model import Ridge, LogisticRegression          # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H3-SP frozen test — apparatus pins at reduced size (subset rows; seed 131235, both heads):\n")

# reduced draws = FIRST rows of the FULL-SIZE draws (binding subset rule)
u_tr_f, c_tr_f, d_tr_f, g_tr_f = sp.gen_hs2(sp.N_TRAIN, "train", sp.TRAIN_SEED)
u_dep_f, c_dep_f, d_dep_f, g_dep_f = sp.gen_hs2(sp.N_DEPLOY, 2.0, sp.DEPLOY_SEED)
u_ctl_f, c_ctl_f, d_ctl_f, g_ctl_f = sp.gen_hs2(sp.N_CONTROL, 0.0, sp.CONTROL_SEED)
u_tr, c_tr = u_tr_f[:2000], c_tr_f[:2000]
u_dep, c_dep, d_dep, g_dep = u_dep_f[:2500], c_dep_f[:2500], d_dep_f[:2500], g_dep_f[:2500]
u_ctl, c_ctl = u_ctl_f[:1000], c_ctl_f[:1000]

# C0 + MI gates on the reduced deploy rows
raw = u_dep.mean(axis=1)
mag = np.linalg.norm(raw, axis=1)
c0 = sp.cv_r2(raw, c_dep)
check("C0 wash gate: raw mean washes c on the reduced deploy rows (<= 0.05)", c0 <= 0.05,
      f"raw_c={c0:+.4f}")
mi_g, mi_m = sp.cv_r2(c_dep[:, None], g_dep), sp.cv_r2(c_dep[:, None], mag)
check("MI gate (ridge legs): g and mag carry no c (<= 0.01)", max(mi_g, mi_m) <= 0.01,
      f"g={mi_g:+.4f} mag={mi_m:+.4f}")
skf = StratifiedKFold(5, shuffle=True, random_state=0)
bal = float(cross_val_score(LogisticRegression(max_iter=2000),
                            StandardScaler().fit_transform(c_dep[:, None]),
                            (d_dep > 0).astype(int), cv=skf, scoring="balanced_accuracy").mean())
check("MI gate: d ~ c at chance (<= 0.52)", bal <= 0.52, f"bal-acc={bal:.4f}")


def run_cells():
    out = {}
    for kind in ("lin", "mlp"):
        phi, head, fit = sp.train_reporter(kind, sp.INIT_SEEDS[0], u_tr, c_tr)
        rep_dep = sp.report(phi, head, u_dep)
        rep_ctl = sp.report(phi, head, u_ctl)
        ctl = sp.cv_r2(rep_ctl[:, None], c_ctl)
        coup = sp.cv_r2(rep_dep[:, None], c_dep)         # ridge leg only (reduced pin)
        var_ratio = float(np.var(rep_dep) / np.var(c_dep))
        p_d = sp.partial_r2(rep_dep, d_dep, c_dep)
        p_g = sp.partial_r2(rep_dep, g_dep, c_dep)
        out[kind] = (fit, ctl, coup, var_ratio, p_d, p_g)
    return out


r1 = run_cells()
r2 = run_cells()
for kind in ("lin", "mlp"):
    fit, ctl, coup, vr, p_d, p_g = r1[kind]
    print(f"    {kind}: fit={fit:.3f} ctl={ctl:+.3f} coup(ridge)={coup:+.3f} var={vr:.3f} "
          f"d={p_d:.3f} g={p_g:.3f}")
    check(f"{kind}: train-fit gate (>= 0.7) at reduced size", fit >= 0.7)
    check(f"{kind}: lam0 control gate (>= 0.9) at reduced size", ctl >= 0.9)
check("cell readouts byte-identical across in-process rerun (both heads)",
      all(r1[k] == r2[k] for k in r1))

print(f"\n{'ALL PASS — H3-SP apparatus pinned at reduced size (full-run verdict NOT asserted here).' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
