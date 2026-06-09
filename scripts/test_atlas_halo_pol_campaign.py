#!/usr/bin/env python
"""Frozen test for the measured-sky LINEAR-pol halo campaign scorecard
(scripts/atlas_halo_pol_campaign.py). Locks the adversarially-sharpened falsification design:
the 9/35/46-deg radii are angularly ISOLATED (the clean differential test), the 22/23/24-deg cluster
is DEGENERATE (discard), the confound-robust DoP ratios are pre-registered, and the predicted ladder
is monotone in radius. Pure-analytic (no RNG, no raytracer). Run: python scripts/test_atlas_halo_pol_campaign.py
"""
import sys
import os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_halo_pol_campaign as cmp  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("Measured-sky LINEAR-pol halo campaign — feasibility + falsification design:\n")

sc = {r["name"].split("-deg")[0] + "deg": r for r in cmp.scorecard()}
iso = {r["name"]: r["isolated"] for r in cmp.scorecard()}

# (1) the clean differential test radii (9/35/46) are angularly ISOLATED
clean_iso = [r for r in cmp.scorecard() if round(r["R"]) in (9, 35, 46)]
check("the clean-test radii 9/35/46-deg are all angularly isolated",
      all(r["isolated"] for r in clean_iso), f"isolated={[round(r['R']) for r in clean_iso if r['isolated']]}")

# (2) the 22/23/24-deg cluster is DEGENERATE (within ~1 deg, DoP gaps below the DoFP floor) -> discard
cluster = [r for r in cmp.scorecard() if 21 < r["R"] < 25]
check("the 22/23/24-deg cluster is flagged degenerate (not a discriminating rung)",
      all(not r["isolated"] and r["tier"] == "degenerate" for r in cluster),
      f"cluster tiers={[r['tier'] for r in cluster]}")

# (3) detectability tiers: 46-deg easy (16% >> floor), 9-deg archival/stack (0.6% < single-shot floor)
t = {round(r["R"]): r["tier"] for r in cmp.scorecard()}
check("46-deg is an EASY target (highest DoP, isolated)", t[46] == "easy")
check("9-deg is archival/stack-only (below the DoFP single-shot floor)", t[9] == "archival/stack")

# (4) the confound-robust differential ratios are the pre-registered targets
dt = cmp.differential_test()
check("DoP(46)/DoP(35) ratio ~ 1.7 (top-end differential)", abs(dt["ratios"]["DoP(46)/DoP(35)"] - 1.72) < 0.1,
      f"={dt['ratios']['DoP(46)/DoP(35)']:.2f}")
check("DoP(35)/DoP(9) ratio ~ 15 (full-span differential)", abs(dt["ratios"]["DoP(35)/DoP(9)"] - 15.4) < 1.0,
      f"={dt['ratios']['DoP(35)/DoP(9)']:.2f}")

# (5) the law predicts a strictly monotone ladder over the isolated radii -> the kill condition
check("the isolated-radius ladder is strictly monotone in R (the falsifiable headline)", dt["monotone"],
      f"DoP={[round(d*100,2) for d in dt['dops']]}%")

# (6) instrument coverage: DoFP frame-stacking reaches all 8; the coarse specMACS swath only the top 2
import s2_optics as so  # noqa: E402
stack_cov = sum(1 for _, R, _ in cmp.HALOS if so.halo_pol_dop(R) >= cmp.INSTRUMENTS["DoFP 100-frame stack"])
swath_cov = sum(1 for _, R, _ in cmp.HALOS if so.halo_pol_dop(R) >= cmp.INSTRUMENTS["specMACS swath (LUCID DoFP)"])
check("DoFP frame-stacking covers all 8 halos; the coarse specMACS swath only the 35/46-deg top",
      stack_cov == 8 and swath_cov == 2, f"stack={stack_cov}/8, swath={swath_cov}/8")

print(f"\n{'ALL PASS -- the campaign reduces to a confound-robust 3-anchor differential (9/35/46-deg isolated radii, monotone DoP ratios immune to common-mode calibration error, U=0 sign-veto), the 22/23/24-deg cluster discarded as degenerate; 46-deg = easiest novel first measurement, the pyramidal ladder = the falsifiable headline. In-house prep; measurement+outreach owner-gated.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
