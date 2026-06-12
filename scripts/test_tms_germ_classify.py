#!/usr/bin/env python
"""Frozen test for S3-A5 (TMS k-gon germ classification). PREREG:
docs/atlas/TMS_KGON_GERM_PREREG.md (frozen 2026-06-12). Pins the registered-run numbers
(2026-06-12): substrate gates, E0 event, u*, theta_M certificates, K2 controls, and the chart
certificate fields. Deterministic. Run: python scripts/test_tms_germ_classify.py
"""
import subprocess
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "scripts")
import tms_potential as tp
import tms_germ_classify as gc

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("S3-A5 frozen test — registered-run pins (2026-06-12):")

print("substrate gates (K1/V0/V1) — full gate script must exit 0:")
r = subprocess.run([sys.executable, "scripts/tms_potential.py"], capture_output=True, text=True)
check("tms_potential.py gates ALL PASS", r.returncode == 0,
      f"exit={r.returncode}" + ("" if r.returncode == 0 else f"  tail: {r.stdout[-300:]}"))

print("E0 event + u* (the registered crossing):")
m5p, m6 = gc.Member("5+", 5, 1), gc.Member("6-gon", 6, 0)
ustar, v5, v6 = gc.find_ustar(m5p, m6)
check("u* = 0.658197813 +/- 1e-6", abs(ustar - 0.658197813) < 1e-6, f"{ustar:.9f}")

print("theta_M certificates (registered values, +/-1e-4):")
for u, name, member, vwarm, rel_pin, L_pin in (
        (0.5, "5+", m5p, v5, 0.22883, 0.037494),
        (0.5, "6-gon", m6, v6, 0.06367, 0.043421),
        (ustar, "5+", m5p, v5, 0.22883, 0.045184),
        (ustar, "6-gon", m6, v6, 0.07712, 0.045184),
        (2.0, "5+", m5p, v5, 0.22883, 0.110410),
        (2.0, "6-gon", m6, v6, 0.08436, 0.055226)):
    v, Hval, ok = gc.polish(member, vwarm.copy(), u)
    cor, rel, _ = gc.essential_corank(member, v, u)
    check(f"{name} @ u={u:.4f}: rel={rel_pin}, L={L_pin}, corank 0",
          ok and cor == 0 and abs(rel - rel_pin) < 1e-4 and abs(Hval / 18 - L_pin) < 1e-5,
          f"rel={rel:.5f} L={Hval/18:.6f} corank={cor}")
check("5+ spectrum RIGID under u (the scaling argument): rel(0.5)==rel(2.0) to 1e-9",
      abs(gc.member_lambda_min(m5p, gc.polish(m5p, v5.copy(), 0.5)[0], 0.5)[0]
          - gc.member_lambda_min(m5p, gc.polish(m5p, v5.copy(), 2.0)[0], 2.0)[0]) < 1e-9)

print("K2 controls (registered readouts):")
import atlas_jet_classify as jc
gen = jc.cusp_c3(*jc.synthetic_swallowtail(-0.40))[0]
dive = jc.cusp_c3(*jc.synthetic_swallowtail(0.0))
check("A4 dive ratio < 0.25", (dive[0] if dive else 0.0) / gen < 0.25)
check("A3 members >= 0.25",
      min((jc.cusp_c3(*jc.synthetic_swallowtail(h)) or [0])[0] / gen for h in (-0.2, -0.1)) >= 0.25)
X, Y, d = jc.synthetic_umbilic(0.0)
check("D4 fires corank-2 (s1_min_rel = 0.0036 +/- 0.001)",
      abs(jc.corank_from_chart(X, Y, d, d)["s1_min_rel"] - 0.0036) < 0.001)
X, Y, d = jc.synthetic_swallowtail_chart(0.0)
check("A4 stays corank-1", jc.corank_from_chart(X, Y, d, d)["corank"] == 1)

print("chart certificate fields (registered run, scripts/tms_liftoff_chart.npz):")
z = np.load("scripts/tms_liftoff_chart.npz")
check("ng = 420 pinned protocol", z["V"].shape == (420, 420))
check("curl REFINED-step = 3.36e-09 +/- 1e-9 (<= 1e-7)",
      abs(float(z["curl_refined"]) - 3.36e-9) < 1e-9 and float(z["curl_refined"]) <= 1e-7,
      f"{float(z['curl_refined']):.2e}")
check("asym (energy audit) <= 1e-6", float(z["asym_worst"]) <= 1e-6,
      f"{float(z['asym_worst']):.2e}")
check("kink cells = 4861 (O4-routed)", int(z["n_kink"]) == 4861, f"{int(z['n_kink'])}")
smooth = z["smooth"]
sgY, sgP = np.sign(z["Y"]), np.sign(z["phi"])
crY = np.zeros_like(smooth); crP = np.zeros_like(smooth)
crY[:-1, :] |= np.diff(sgY, axis=0) != 0; crY[:, :-1] |= np.diff(sgY, axis=1) != 0
crP[:-1, :] |= np.diff(sgP, axis=0) != 0; crP[:, :-1] |= np.diff(sgP, axis=1) != 0
check("Y=0-coincident (O1 candidate) cells = 0 (the O1-decisive pin)",
      int((crY & crP & smooth).sum()) == 0)

print("end-to-end verdict (stage classify must exit 0 = O2):")
r = subprocess.run([sys.executable, "scripts/tms_germ_classify.py", "--stage", "classify"],
                   capture_output=True, encoding="utf-8", errors="replace")
check("classify exit 0 with O2 verdict line",
      r.returncode == 0 and "MAXWELL LEVEL-CROSSING" in r.stdout, f"exit={r.returncode}")

print(f"\n{'ALL PASS' if fail == 0 else str(fail) + ' FAILED'} — S3-A5 frozen test")
sys.exit(1 if fail else 0)
