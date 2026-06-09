#!/usr/bin/env python
"""H6 (slate `ww6koomb1`) — the sorting-network verify≪search certificate.

Sorting networks are a clean instance of the lab's referee-free thesis: CHEAP to VERIFY, HARD to FIND.
- VERIFY (cheap, SOUND): the 0-1 PRINCIPLE — a comparator network sorts ALL inputs iff it sorts all 2^n
  BINARY inputs (because comparator networks COMMUTE with monotone maps: a comparator is (min,max), and
  min/max commute with any monotone f, so net(f∘x)=f∘net(x); threshold f's reduce the general case to
  binary). So a network is verifiable in O(2^n · size) — polynomial in the WITNESS (the network) — vs the
  naive n! permutation check.
- FIND (hard, IMPORTED): finding a MINIMAL-size / minimal-depth sorting network is a notoriously hard
  combinatorial search (optimal sizes are open/required massive SAT effort for n≳10). NOT proven hard here
  — the search-hardness is the imported wall, exactly like ISD-hardness in the syndrome certificate.

This file: the empirical demonstration (the Lean 0-1-principle SOUNDNESS pillar is sundogcert/SortingCert.lean).
NOT public-eligible. Attribution: Knuth TAOCP vol 3 (the 0-1 principle); the lab's certificate thesis.
Run: python scripts/sorting_cert.py
"""
import sys
import time
from itertools import permutations, product
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def apply_net(net, x):
    """Apply a comparator network (list of (i,j), i<j) to a vector x -> compare-exchange in place."""
    x = list(x)
    for i, j in net:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x


def is_sorted(x):
    return all(x[k] <= x[k + 1] for k in range(len(x) - 1))


def sorts_binary(net, n):
    """The CHEAP CHECK (0-1 principle): sorts ALL inputs iff sorts all 2^n binary inputs. O(2^n·size)."""
    return all(is_sorted(apply_net(net, b)) for b in product((0, 1), repeat=n))


def sorts_all_perms(net, n):
    """The naive full check: sorts all n! permutations (tiny n only) — to confirm the 0-1 principle agrees."""
    return all(is_sorted(apply_net(net, p)) for p in permutations(range(n)))


def bubble_net(n):
    """A guaranteed-correct sorting network: n passes of adjacent compare-exchanges (bubble sort)."""
    return [(i, i + 1) for _ in range(n) for i in range(n - 1)]


# known small OPTIMAL-size sorting networks (Knuth / Bose-Nelson)
OPT = {
    4: [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)],                                          # size 5
    5: [(0, 1), (3, 4), (2, 4), (2, 3), (0, 3), (0, 2), (1, 4), (1, 3), (1, 2)],          # size 9
    6: [(0, 1), (2, 3), (4, 5), (0, 2), (3, 5), (1, 4), (0, 1), (2, 3), (4, 5),
        (1, 2), (3, 4), (2, 3)],                                                          # size 12
}


def random_net(n, size, rng):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [pairs[k] for k in rng.integers(0, len(pairs), size)]


def main():
    print("=" * 84)
    print("H6 — the sorting-network verify≪search certificate (0-1 principle)")
    print("=" * 84)

    # (1) SOUNDNESS of the cheap check: the 0-1 principle (2^n binary) AGREES with the full n! check
    print("\n(1) 0-1 PRINCIPLE soundness: does the 2^n binary check agree with the full n! permutation check?")
    ok01 = True
    for n in (4, 5, 6):
        for net, tag in ((OPT[n], "optimal"), (bubble_net(n), "bubble"), (OPT[n][:-1], "broken(-1 comp)")):
            sb = sorts_binary(net, n)
            sp = sorts_all_perms(net, n)
            agree = (sb == sp)
            ok01 &= agree
            print(f"    n={n} {tag:16s}: binary-check={sb!s:5}  full-check={sp!s:5}  agree={agree}"
                  + ("" if agree else "  <-- MISMATCH"))
    print(f"    => the cheap 2^n binary check is SOUND (agrees with n!): {ok01}")

    # (2) the ASYMMETRY: verify is cheap (poly in the witness); finding-by-search is a needle in a haystack
    print("\n(2) VERIFY≪SEARCH asymmetry (verify cost vs the fraction of random size-s nets that sort):")
    print(f"    {'n':>3} {'opt size s':>11} {'verify ops 2^n·s':>17} {'#size-s nets':>14} {'P(random sorts)':>16} {'~search 1/P':>14}")
    rng = np.random.default_rng(0)
    for n in (4, 5, 6):
        s = len(OPT[n])
        npairs = n * (n - 1) // 2
        space = npairs ** s
        trials = 200000
        hits = sum(sorts_binary(random_net(n, s, rng), n) for _ in range(trials))
        p = hits / trials
        verify_ops = (2 ** n) * s
        inv = (1.0 / p) if p > 0 else float("inf")
        print(f"    {n:>3} {s:>11} {verify_ops:>17} {space:>14.2e} {p:>16.2e} {inv:>14.2e}")

    # (3) timing: verifying a network is a single fast pass even as n grows (the cheap-check pole)
    print("\n(3) verify timing (the cheap-check pole — a single O(2^n·size) pass):")
    for n in (8, 12, 16):
        net = bubble_net(n)
        t0 = time.time()
        v = sorts_binary(net, n)
        dt = time.time() - t0
        print(f"    n={n:>2}: bubble net ({len(net)} comparators) verified sorts={v} in {dt*1000:.1f} ms "
              f"(2^{n}={2**n} binary inputs)")

    print("\n" + "=" * 84)
    print("RESULT: the 0-1 principle makes sorting-network VERIFICATION sound + cheap (O(2^n·size), poly in")
    print("  the witness), while FINDING a (minimal) network is a needle-in-a-haystack search (P(random")
    print("  sorts)→0, space (n choose 2)^s super-exponential). A clean verify≪search certificate instance —")
    print("  the combinatorics sibling of the syndrome certificate. (Soundness machine-checked: the Lean")
    print("  0-1-principle pillar, sundogcert/SortingCert.lean.)")
    print("=" * 84)
    return 0 if ok01 else 1


if __name__ == "__main__":
    sys.exit(main())
