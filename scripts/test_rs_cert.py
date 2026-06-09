#!/usr/bin/env python
"""Frozen test for C1 (scripts/rs_cert.py) — the Reed–Solomon evaluation certificate GF(7) sanity,
the numerical witness for the machine-checked Lean pillar sundogcert/Sundogcert/RSCertificate.lean.
Run: python scripts/test_rs_cert.py
"""
import sys
import itertools
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import rs_cert as rc   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("C1 — Reed–Solomon evaluation certificate (GF(7)) sanity:\n")

# rs_unique / DETERMINE: encoding injective on deg<k messages
allpolys = list(itertools.product(range(rc.P), repeat=rc.K))
codewords = [tuple(rc.encode(c)) for c in allpolys]
check("rs_unique (DETERMINE): deg<k encoding is INJECTIVE (k evals determine the poly)",
      len(set(codewords)) == len(allpolys), f"{len(set(codewords))}/{len(allpolys)} distinct")

# corruption_fiber_nontrivial / LOSE: many words decode to the same f
f = (3, 2)
c = rc.encode(f)
fiber = {tuple(c)}
for i in range(rc.N):
    for v in range(rc.P):
        if v != c[i]:
            y = c.copy(); y[i] = v
            if rc.decodes(f, y):
                fiber.add(tuple(y))
check("corruption_fiber_nontrivial (LOSE): ≥2 distinct words decode to the same f",
      len(fiber) >= 2, f"fiber size = {len(fiber)}")

# unique_decoding within the radius: no OTHER deg<k poly decodes a fiber word
d = rc.N - rc.K + 1
check("within the unique-decoding radius (2·τ < d = n−k+1)", 2 * rc.TAU < d, f"2τ={2*rc.TAU} < d={d}")
other = any(g != f and rc.decodes(g, np.array(y)) for y in fiber for g in allpolys)
check("unique_decoding: NO second deg<k poly decodes any fiber word (decoding is unique in radius)",
      not other)

# accept_sound flavor: a genuine decoding passes the cheap check; a non-decoding (deg<k but too far) fails
bad = (0, 0)                                    # the all-zero message: codeword (0,0,0,0,0)
check("accept_sound: the true message decodes its own clean codeword (cheap check accepts)",
      rc.decodes(f, c))
check("the cheap check REJECTS a wrong deg<k message on this word (sound, not vacuous)",
      not rc.decodes(bad, c))

print(f"\n{'ALL PASS -- k evals DETERMINE the unique deg<k message (injective encoding); the cheap O(n·k) check is sound and unique within the radius; yet many corrupted words decode to the SAME message (determine/lose). The evaluation-dual of the syndrome certificate.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
