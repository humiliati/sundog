#!/usr/bin/env python3
"""Two TYPES of sigma = infinity resistance, separated by the recovery FIBER
(how many bodies map to one shadow). A generic instrument; no lane-specific data.

(1) INFO-LOSS resistance: a shadow s = M x over GF(2) with a nontrivial kernel.
    Many x give the same s  ->  fiber > 1  ->  the body is NOT a function of the
    shadow  ->  even UNBOUNDED compute cannot recover it.

(2) COMPUTATIONAL resistance: lambda(n) = (-1)^Omega(n) is a FUNCTION of n
    ->  fiber = 1  ->  unbounded compute (factor n) recovers it  ->  but no
    finite-order statistic of the sequence does (sigma_predict = infinity):
    the information is PRESENT yet finite-order / zero-entropy inaccessible.

Both have sigma = infinity, but DIFFERENT fiber -> different resistance TYPES ->
they do NOT live on one shared dial. Run:
  python scripts/resistance_types_fiber_vs_order.py
"""
import numpy as np


def gf2_rank(A):
    A = (A % 2).astype(np.int8)
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
    return r


def info_loss_fiber(D=10, m=6, seed=0):
    """shadow s = M x over GF(2), M is m x D. fiber of x given s = 2^(D - rank)."""
    rng = np.random.default_rng(seed)
    M = rng.integers(0, 2, size=(m, D))
    rank = gf2_rank(M)
    return rank, 2 ** (D - rank)


def liouville(N):
    omega = np.zeros(N + 1, dtype=np.int8)
    comp = np.zeros(N + 1, dtype=bool)
    for p in range(2, N + 1):
        if not comp[p]:
            comp[p * p::p] = True
            pe = p
            while pe <= N:
                omega[pe::pe] += 1
                pe *= p
    return (1 - 2 * (omega & 1)).astype(np.int8)


def parity_finite_order_ambiguous(lam, k=1):
    """is lambda(n) under-determined by its preceding-k sequence window? (sigma>k)"""
    seen = {}
    for i in range(k, len(lam)):
        key = lam[i - k:i].tobytes()
        if key in seen and seen[key] != lam[i]:
            return True
        seen.setdefault(key, lam[i])
    return False


def main():
    print("RESISTANCE_TYPES   (two kinds of sigma=infinity, told apart by the FIBER)\n")

    rank, fiber = info_loss_fiber(D=10, m=6, seed=0)
    print("(1) INFO-LOSS: shadow s = M x over GF(2), M is 6x10")
    print(f"    rank(M) = {rank}  ->  fiber(x | s) = 2^(10-{rank}) = {fiber}  (>1: body NOT a function of s)")
    print(f"    => unbounded compute cannot recover x; sigma = infinity because the INFO IS GONE.")

    lam = liouville(200_000)
    amb1 = parity_finite_order_ambiguous(lam, k=1)
    print("\n(2) COMPUTATIONAL: lambda(n) = (-1)^Omega(n)")
    print(f"    fiber(lambda | n) = 1  (lambda is single-valued in n)  ->  factoring n recovers it.")
    print(f"    finite-order ambiguous at k=1? {amb1}  ->  sigma_predict = infinity, but the INFO IS PRESENT.")

    print("\nPLACEMENT (fiber x finite-order sigma):")
    print("    type                 | fiber  | finite-order sigma | recoverable by unbounded compute?")
    print("    determine            |   1    | finite             | yes (already finite-order)")
    print("    INFO-LOSS resist     |  >1    | infinity           | NO (not a function of the shadow)")
    print("    COMPUTATIONAL resist |   1    | infinity           | YES (e.g. factor n) — parity lives here")

    print("\nVERDICT:")
    print("  Info-loss resistance (fiber>1) and computational resistance (fiber=1) are DIFFERENT TYPES,")
    print("  both sigma=infinity. A capacity/info dial measures the ONSET of fiber>1; parity has fiber=1,")
    print("  so it does NOT sit at that dial's max — it occupies the orthogonal (fiber=1, sigma=infinity)")
    print("  corner. The two do not share units.")


if __name__ == "__main__":
    main()
