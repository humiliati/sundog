#!/usr/bin/env python
"""Frozen test for H6 (scripts/sorting_cert.py) — the sorting-network verify≪search certificate. Locks the
0-1-principle SOUNDNESS (the cheap 2^n binary check agrees with the full n! check, and catches a broken
network) and the verify≪search asymmetry. Deterministic. Run: python scripts/test_sorting_cert.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import sorting_cert as sc   # noqa: E402
import numpy as np          # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H6 — sorting-network verify≪search certificate (0-1 principle):\n")

# 0-1 principle SOUNDNESS: the 2^n binary check agrees with the full n! check, for sorting + broken nets
for n in (4, 5, 6):
    for net, tag, want in ((sc.OPT[n], "optimal", True), (sc.bubble_net(n), "bubble", True),
                           (sc.OPT[n][:-1], "broken", False)):
        sb, sp = sc.sorts_binary(net, n), sc.sorts_all_perms(net, n)
        check(f"n={n} {tag}: 2^n binary-check == full n! check (= {want})", sb == sp == want,
              f"binary={sb} full={sp}")

# the broken network (optimal minus one comparator) is genuinely CAUGHT by the cheap check
check("a broken network (optimal − 1 comparator) is CAUGHT by the cheap 2^n check (n=6)",
      sc.sorts_binary(sc.OPT[6][:-1], 6) is False)

# verify≪search asymmetry: verify is cheap; random search fraction collapses with n
rng = np.random.default_rng(0)
def frac(n, trials=20000):
    s = len(sc.OPT[n])
    return sum(sc.sorts_binary(sc.random_net(n, s, rng), n) for _ in range(trials)) / trials
p4, p6 = frac(4), frac(6)
check("verify≪search: P(random size-s net sorts) collapses with n (p(n=4) >> p(n=6))",
      p4 > p6 and p6 <= 1e-3, f"p(4)={p4:.1e}  p(6)={p6:.1e}")
check("verify is cheap: a 56-comparator n=8 bubble net verifies (2^8 binary inputs) — poly in the witness",
      sc.sorts_binary(sc.bubble_net(8), 8) is True)

print(f"\n{'ALL PASS -- the 0-1 principle is a SOUND cheap check (2^n binary == n! full, broken nets caught); verifying is poly in the witness while finding a (minimal) network is a needle-in-a-haystack search. A clean verify≪search certificate.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
