#!/usr/bin/env python3
"""F-opt (lr=1e-3) seed-stability aggregator — the frozen readout for the criterion
in docs/chatv2/PHASE1_R1_COMPLETION.md ("F-opt falsifier resolution", 2026-06-29).

Reads the three seed manifests (seed-0 = the original cell + seeds 1,2 from the
b1prnpkp3 run), computes objective_excess = body_carry_gen - body_carry_twin per
LEARNED seed, and prints the verdict:
  CLEARED       mean excess >= 0.10 (each learned, gates pass)  -> optimizer axis does NOT falsify R1
  FIRED         mean excess <  0.10 (learned)                   -> R1 re-scopes to lr=3e-4
  UNINFORMATIVE < 2 learned seeds                               -> re-run the UNLEARNED seed(s)
Run (any cwd): python scripts/chatv2_fopt_aggregate.py
"""
import json, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = {
    0: "results/chatv2/phase1-r1/Fopt_lr1e-3",
    1: "results/chatv2/phase1-r1/Fopt_lr1e-3_seed1",
    2: "results/chatv2/phase1-r1/Fopt_lr1e-3_seed2",
}
FLOOR, Z1_MIN, LEAK_MAX, H = 0.10, 0.70, 0.58, 8
DDEC_MIN = H / 2

rows = []
for seed, rel in DIRS.items():
    m = os.path.join(BASE, rel, "manifest.json")
    if not os.path.exists(m):
        rows.append((seed, "MISSING", False, *(float("nan"),) * 5, False)); continue
    rec = next((r for r in json.load(open(m))["records"] if r["H"] == H), None)
    if rec is None:
        rows.append((seed, "NO-REC", False, *(float("nan"),) * 5, False)); continue
    g, t = rec["generative"], (rec.get("twin") or {})
    status = rec.get("status", "?")
    learned = status != "UNLEARNED" and bool(t)   # twin is skipped when gen is UNLEARNED
    excess = (g["body_carry"] - t["body_carry"]) if t else float("nan")
    gates = g["z1_acc"] >= Z1_MIN and g["cross_latent_leak"] <= LEAK_MAX and g["d_dec"] >= DDEC_MIN
    rows.append((seed, status, learned, g["z1_acc"], g["d_dec"], g["cross_latent_leak"],
                 g["body_carry"], t.get("body_carry", float("nan")), excess, gates))

print(f"{'seed':>4} {'status':>9} {'learned':>7} {'z1':>6} {'d_dec':>6} {'leak':>6} {'gen_bc':>7} {'twin_bc':>7} {'excess':>7} {'gates':>6}")
for r in rows:
    if r[1] in ("MISSING", "NO-REC"):
        print(f"{r[0]:>4} {r[1]:>9}  (not present yet)"); continue
    print(f"{r[0]:>4} {r[1]:>9} {str(r[2]):>7} {r[3]:>6.3f} {r[4]:>6.2f} {r[5]:>6.3f} {r[6]:>7.3f} {r[7]:>7.3f} {r[8]:>7.3f} {str(r[9]):>6}")

learned = [r for r in rows if r[2]]
n = len(learned)
mean_excess = sum(r[8] for r in learned) / n if n else float("nan")
all_gates = all(r[9] for r in learned)
print(f"\nlearned seeds: {n}/3   mean objective_excess (learned) = {mean_excess:.3f}   floor = {FLOOR}")
if n < 2:
    v = "UNINFORMATIVE (F3') — fewer than 2 learned seeds; re-run the UNLEARNED seed(s)"
elif mean_excess >= FLOOR and all_gates:
    v = "CLEARED — F-opt holds; the optimizer axis does NOT falsify R1 (seed-0's 0.095 was a low-seed draw)"
elif mean_excess < FLOOR:
    v = "FIRED — objective_excess collapses at lr=1e-3; re-scope R1 to lr=3e-4 (rewrite around what generalized)"
else:
    v = "REVIEW — mean >= floor but a learned seed missed a SHARP gate; read the row"
print(f"VERDICT: {v}")
