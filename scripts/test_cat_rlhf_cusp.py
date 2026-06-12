#!/usr/bin/env python
"""Frozen test for H12 (scripts/cat_rlhf_cusp.py).

Runs BEFORE the verdict run: pins the apparatus contract -- exact-gradient correctness
(numpy == torch autograd), the Z2 equivariance of the bandit+trunk, Gibbs closed form, tabular
global convergence, the certification machinery on a known-symmetric point, determinism,
representability, and the fold-walker CALIBRATED on the analytic cusp normal form (the H4
"validate the instrument on a known catastrophe" discipline). Does NOT pin verdict outcomes.
Run: python scripts/test_cat_rlhf_cusp.py
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import cat_rlhf_cusp as cc                      # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H12 frozen test (apparatus contract + instrument calibration):\n")

# ---- exact gradients: numpy manual backprop == torch autograd ---- #
rng = np.random.default_rng(0)
worst = 0.0
for beta, eps in [(0.05, 0.0), (0.3, 0.1), (1.0, -0.2)]:
    th = rng.normal(0, 0.5, cc.NPARAM)
    _, g_np = cc.j_and_grad(th, beta, eps)
    g_t = cc.torch_grad(th, beta, eps)
    worst = max(worst, float(np.max(np.abs(g_np - g_t))))
check("numpy manual gradient == torch autograd (3 random points)", worst < 1e-12,
      f"max|diff| = {worst:.2e}")

# ---- Z2 equivariance: negating W1's y-column permutes the good pair, fixes distractors ---- #
th = rng.normal(0, 0.5, cc.NPARAM)
th2 = th.copy()
W1 = th2[:2 * cc.H1N].reshape(cc.H1N, 2)
W1[:, 1] *= -1.0
L1, _, _ = cc.logits_np(th)
L2, _, _ = cc.logits_np(th2)
perm = np.array([1, 0, 2, 3, 4, 5])
check("Z2: y-flip in parameters permutes logits (a+ <-> a-, distractors fixed)",
      np.allclose(L2, L1[perm], atol=1e-12))
J1, _ = cc.j_and_grad(th, 0.2, 0.0)
J2, _ = cc.j_and_grad(th2, 0.2, 0.0)
check("Z2: J invariant under the flip at eps=0", abs(J1 - J2) < 1e-12)
check("Z2: m flips sign under the flip", abs(cc.m_of(th) + cc.m_of(th2)) < 1e-12)

# ---- Gibbs closed form ---- #
check("m_gibbs(beta, 0) = 0 exactly", cc.m_gibbs(0.3, 0.0) == 0.0)
p = cc.gibbs_pi(0.5, 0.0)
check("Gibbs: good pair share = exp(2)/(exp(2)*2+4) each at beta=0.5",
      abs(p[0] - np.exp(2.0) / (2 * np.exp(2.0) + 4)) < 1e-12)
check("m_gibbs sign follows eps", cc.m_gibbs(0.3, 0.1) > 0 > cc.m_gibbs(0.3, -0.1))

# ---- tabular control (gate iii machinery) ---- #
ok_tab = all(abs(cc.tabular_converge(b, 0.0, s) - cc.m_gibbs(b, 0.0)) <= 1e-6
             for b in (0.1, 0.8) for s in (0, 1))
check("tabular softmax converges to Gibbs (2 beta x 2 inits, |m - m_gibbs| <= 1e-6)", ok_tab)

# ---- convergence + certification machinery at a KL-dominant point ---- #
th_c, J_c, g_c, steps = cc.converge(cc.init_theta(0), 1.0, 0.0)
ok, info = cc.certify(th_c, 1.0, 0.0, full=True)
check("converge reaches ||grad||inf <= 1e-10 at beta=1.0", float(np.max(np.abs(g_c))) <= 1e-10,
      f"grad_inf = {float(np.max(np.abs(g_c))):.2e} ({steps} steps)")
check("certification battery passes at the beta=1.0 attractor (D1: lam_max <= +1e-9, no "
      "unstable direction; perturb-return)", ok, f"info = {info}")
th_d, _, _, _ = cc.converge(cc.init_theta(0), 1.0, 0.0)
check("converge deterministic (byte-identical rerun)", np.array_equal(th_c, th_d))

# ---- representability control ---- #
tv = cc.distill_tv(0.05)
check("representability: trunk distills Gibbs logits at beta=0.05 (TV <= 1e-3)", tv <= 1e-3,
      f"TV = {tv:.2e}")

# ---- fold-walker calibration on the ANALYTIC cusp normal form ---- #
a = -0.25                                       # broken regime; folds at +-(2/(3*sqrt3))|a|^1.5
analytic = (2.0 / (3.0 * np.sqrt(3.0))) * abs(a) ** 1.5


def conv_nf(th, e):
    m = float(th[0])
    for _ in range(200_000):
        g = a * m + m ** 3 - e
        if abs(g) < 1e-14:
            break
        m -= 0.2 * g
    return np.array([m])


e_fold, _ = cc.walk_fold(np.array([np.sqrt(abs(a))]), None, -1, conv_nf,
                         lambda t: float(t[0]), step0=0.01, sign_keep=+1)
check("fold-walker reproduces the analytic cusp fold (|eps_f| = (2/3sqrt3)|a|^1.5)",
      e_fold is not None and abs(abs(e_fold) - analytic) < 2e-5,
      f"measured {abs(e_fold):.6f} vs analytic {analytic:.6f}")

# ---- log-log OLS sanity ---- #
xs = np.array([0.02, 0.05, 0.1, 0.2, 0.3])
slope, r2 = cc.ols_loglog(xs, 3.0 * xs ** 1.5)
check("ols_loglog recovers a 3/2 power law exactly", abs(slope - 1.5) < 1e-10 and r2 > 0.999999)

# ---- BANKED PINS (post-verdict, full run of 2026-06-12; KILL(i) clean informative null) ----
# Asserted against the committed verdict JSON (no recompute in-test). Headline: 128/128 fresh-init
# endpoints symmetric (max |m| 2.8e-5 over ALL points, 6.9e-11 over the 41 certified); certified
# points sit ON the Gibbs optimum (dJ <= 2.2e-16); every certification failure is 'grad'
# (convergence rate), never an unstable Hessian or a failed dynamical certificate.
import json                                     # noqa: E402
from pathlib import Path                        # noqa: E402

_jp = Path("results/atlas/h12/cat_rlhf_cusp_result.json")
if _jp.exists():
    r = json.loads(_jp.read_text())
    c = r["coarse"]
    cert = [x for x in c if x["certified"]]
    check("banked verdict = kill_i (no breaking; tabular gate + representability pass)",
          r["verdict"] == "kill_i" and not r["breaking"] and r["gate_tabular"]
          and r["rep_ok"])
    check("banked coverage: 128 points, 41 certified, all failures 'grad'",
          len(c) == 128 and len(cert) == 41
          and all(x.get("fail") == "grad" for x in c if not x["certified"]))
    check("banked symmetry: max |m| = 2.798e-5 over ALL endpoints (certified: 6.9e-11)",
          abs(max(abs(x["m"]) for x in c) - 2.798167559947551e-05) < 1e-12
          and max(abs(x["m"]) for x in cert) < 1e-10)
    check("banked Gibbs identity: certified points ON the analytic optimum (dJ <= 2.3e-16)",
          max(x["dJ_gibbs"] for x in cert) <= 2.3e-16)
else:
    print("  [SKIP] banked pins (verdict JSON not present)")

print(f"\n{'ALL PASS -- apparatus pinned; instrument calibrated; ready for smoke, then verdict.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
