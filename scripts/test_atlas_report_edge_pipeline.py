#!/usr/bin/env python
"""Frozen test for A1 Leg 0 (scripts/atlas_report_edge_pipeline.py) — locks the registered K0a verdict
(2026-06-12): the wall-value-blind pipeline detects the edge everywhere (specificity 96%, power 100%)
but its worst nuisance cell (gamma=2.0, eps=1%) misses the 1.0-deg recovery gate (median 1.270 deg),
because the midpoint-to-wall offset is gamma-dependent and the blindness-preserving single-constant
correction cannot track it. NO live pull; banked as the lane's resolution-floor null. P2's own leg
PASSED (worst 0.215 vs 1.5) but live is blocked by the both-legs rule.
Run: python scripts/test_atlas_report_edge_pipeline.py   (~2 min; requires the flux cache)
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import atlas_report_edge_pipeline as ap   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


B_HAT = 0.501          # the registered pooled constant correction (blindness rule passed, spread 0.277)
N_REP = 200

print("A1 Leg 0 — frozen test (registered verdict: K0a, resolution-floor null, NO live pull):\n")

# (1) solar-position pins (identities + almanac ranges)
ok_alm, rows = ap.almanac_checks(verbose=False)
check("solar-position identity/almanac pins 5/5", ok_alm,
      "; ".join(f"{d}" for _, _, d in rows))

# (2) the frozen flux cache (the registered curve)
e, f = ap.flux_curve()
check("flux cache: 46 points, max 2740 at e=24.0",
      len(e) == 46 and abs(float(f.max()) - 2740) < 1 and abs(float(e[np.argmax(f)]) - 24.0) < 1e-9,
      f"max={f.max():.0f}@{e[np.argmax(f)]:.1f}")
i28 = int(np.where(e == 28.0)[0][0]); i34 = int(np.where(e == 34.0)[0][0])
check("flux curve: plateau at 28, ~zero at 34 (the wall roll-off; <5 vs plateau ~2618)",
      f[i28] > 2000 and f[i34] < 5.0, f"f(28)={f[i28]:.0f} f(34)={f[i34]:.2f}")

# (3) McDowell-shape calibration (ABORT guard)
rng = ap._rng(0, 0, 0, 0)
mu, day, hloc, w_eff, h_true = ap._candidates(rng, 200000)
w = w_eff * (ap.phi(h_true) / np.max(ap.phi(np.arange(3, 36, 0.1)))) ** 1.0
hs = h_true[ap._draw(rng, w, 20000)]
mode = float(0.5 * (ap.H_BINS[:-1] + ap.H_BINS[1:])[np.argmax(np.histogram(hs, ap.H_BINS)[0])])
tail = float(np.mean(hs > 33.0))
check("McDowell shape: mode=23.5 in [18,29], unsmeared tail<2%", abs(mode - 23.5) < 1e-9 and tail < 0.02,
      f"mode={mode} tail={tail:.2%}")

# (4) the K0a-driving cell, reproduced with the registered seeds (gamma=2.0, eps=1%, w=28.0 -> 1.270)
ci = [(g, e2) for g in (0.5, 1.0, 2.0) for e2 in (0.01, 0.05, 0.10)].index((2.0, 0.01))
ests = np.array([ap.fit_p1(*ap.gen_p1(ap._rng(1, ci, 0, r), 28.0, 2.0, 0.01))[0] for r in range(N_REP)])
med_bad = float(np.median(np.abs(ests - B_HAT - 28.0)))
check("K0a-driving cell (gamma=2, eps=1%, w=28): median |err| = 1.270 (+-0.02) > 1.0 gate",
      abs(med_bad - 1.270) < 0.02 and med_bad > 1.0, f"{med_bad:.3f}")

# (5) a passing cell for contrast (gamma=1, eps=5%, w=32.196 -> 0.232)
ci2 = [(g, e2) for g in (0.5, 1.0, 2.0) for e2 in (0.01, 0.05, 0.10)].index((1.0, 0.05))
ests2 = np.array([ap.fit_p1(*ap.gen_p1(ap._rng(1, ci2, 1, r), 32.196, 1.0, 0.05))[0] for r in range(N_REP)])
med_ok = float(np.median(np.abs(ests2 - B_HAT - 32.196)))
check("contrast cell (gamma=1, eps=5%, w=32.196): median |err| = 0.232 (+-0.02) <= 1.0",
      abs(med_ok - 0.232) < 0.02 and med_ok <= 1.0, f"{med_ok:.3f}")

# (6) wall-free controls: threshold + specificity (registered: 4.56, 96.0%)
lr_a = np.array([ap.fit_p1(*ap.gen_p1(ap._rng(3, 1000 + r, 0, 0), ap.WALL, 1.0, 0.05, wall_free=True))[3]
                 for r in range(N_REP)])
thresh = float(np.percentile(lr_a, 95))
lr_b = np.array([ap.fit_p1(*ap.gen_p1(ap._rng(4, 5000 + r, 0, 0), ap.WALL, 1.0, 0.05, wall_free=True))[3]
                 for r in range(N_REP)])
spec = float(np.mean(lr_b <= thresh))
check("batch-A LR threshold = 4.56 (+-0.05); batch-B specificity = 96.0% (+-0.5%) >= 95%",
      abs(thresh - 4.56) < 0.05 and abs(spec - 0.96) < 0.005 and spec >= 0.95,
      f"thresh={thresh:.2f} spec={spec:.1%}")

# (7) power sanity at N_CZA=1000 (registered 100%; quick 30-replicate subset must stay >= 90%)
det = [ap.fit_p1(*ap.gen_p1(ap._rng(5, 1000, 1, r), 32.196, 1.0, 0.05, n_cza=1000))[3] > thresh
       for r in range(30)]
check("power subset (30 reps, w=32.196) >= 90% (registered: 100%)", np.mean(det) >= 0.90,
      f"{np.mean(det):.0%}")

# (8) P2: the passing midpoint leg (registered worst 0.215; pin the w_t=1.5/m=29.71 cell = 0.158)
ests3 = np.array([ap.fit_p2(*ap.gen_p2(ap._rng(7, 15, 1, r), 29.71, w_t=1.5))[0] for r in range(N_REP)])
med_p2 = float(np.median(np.abs(ests3 - 29.71)))
check("P2 cell (w_t=1.5, m=29.71): median |err| = 0.158 (+-0.02) <= 1.5 gate",
      abs(med_p2 - 0.158) < 0.02 and med_p2 <= 1.5, f"{med_p2:.3f}")

# (9) verdict reproduction: the worst pinned cell breaches the gate => K0a fires => NO live pull
check("VERDICT REPRODUCED: K0a fires (worst cell > 1.0 deg) while specificity/power/P2 pass",
      med_bad > 1.0 and spec >= 0.95 and med_p2 <= 1.5)

print(f"\n{'ALL PASS — the registered A1 Leg-0 verdict is locked: K0a (resolution-floor null). The wall edge is DETECTABLE in citizen-report statistics (specificity 96%, power 100%) but the wall-blind generic-logistic estimator misses the 1.0-deg recovery gate in its worst nuisance cell (gamma-dependent midpoint offset, untrackable by the single-constant blindness-preserving correction). NO live pull. Named follow-up: template-translation estimator (v2 prereg).' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
