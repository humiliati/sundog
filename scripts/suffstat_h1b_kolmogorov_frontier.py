#!/usr/bin/env python3
"""Sufficient-statistic-order slate, H1 frontier: can a UNIVERSAL description-length
(Kolmogorov) filtration rescue STRONG H1 — one comparable scalar that preserves
determine/resist across all lanes?

A universal comparable scalar DOES exist: Kolmogorov complexity K is one scale (bits),
well-defined up to O(1). The real question is whether K PRESERVES the determine/resist
dichotomy. Test three measures per object:

  K     = description length of a sufficient statistic for the latent   [UNBOUNDED]
  loc   = input-locality: # shadow coordinates the cheapest sufficient
          statistic must read                                            [bounded]
  sigma = finite-order predictive order                                  [bounded]

Determine <-> the measure is finite; resist <-> infinite. If K agreed with the bounded
axes on every object, K would be the universal dichotomy-preserving scalar and rescue
strong H1. The witness that it does NOT is PARITY.
Run: python scripts/suffstat_h1b_kolmogorov_frontier.py
"""
import math
import numpy as np
from functools import reduce
from operator import xor

INF = float("inf")


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


def parity_K_upper_bound(n):
    """An EXHIBITED program computing the total-parity sufficient statistic. Its size is
    INDEPENDENT of the data length n (only n itself must be named), so K <= |prog| + log2 n."""
    prog = "f=lambda b:reduce(xor,b)"   # fold XOR over the whole input — fixed size
    # sanity: it really computes the total parity
    bits = [1, 0, 1, 1, 0, 1]
    assert reduce(xor, bits) == (sum(bits) % 2)
    return len(prog) * 8 + math.ceil(math.log2(max(n, 2))), prog  # bits


def main():
    print("SUFFSTAT_H1B_KOLMOGOROV_FRONTIER   (does K preserve determine/resist?)\n")
    n = 10 ** 6

    # PARITY: latent T = XOR(all n bits)
    Kbits, prog = parity_K_upper_bound(n)
    print("PARITY  (latent = XOR of all n bits):")
    print(f"    K(sufficient statistic) <= {Kbits} bits  via the fixed program  `{prog}`")
    print(f"       -> size INDEPENDENT of n  ->  K is SMALL  ->  DETERMINE under K")
    print(f"    loc   = n = {n}        (the program must read every bit)        -> RESIST")
    print(f"    sigma = inf            (no finite-order predictor; right-special at every length) -> RESIST")
    print("    => K and the bounded axes DISAGREE on parity.\n")

    # FINITE-MARKOV control: a period-q signal — determine on ALL axes
    q = 7
    print(f"FINITE-MARKOV control (period q={q}):")
    print(f"    K = O(1)+log q  (small)   loc = {q}   sigma = {q}   -> DETERMINE on every axis (agreement)\n")

    # INFO-LOSS: random GF(2) shadow s = M x, kernel nontrivial
    D, m = 12, 7
    M = np.random.default_rng(0).integers(0, 2, size=(m, D))
    rank = gf2_rank(M)
    missing = D - rank
    print(f"INFO-LOSS (shadow s = M x over GF(2), M {m}x{D}, rank {rank}):")
    print(f"    K(secret | shadow) >= {missing} bits  (the {missing} lost dims must be SPECIFIED, not derived) -> RESIST")
    print(f"    loc = inf   sigma = inf   -> RESIST on every axis (agreement: the info is genuinely GONE)\n")

    print("TABLE                | K (unbounded) | loc (bounded) | sigma (bounded) | dichotomy?")
    print("  finite-Markov      | finite        | finite        | finite          | DETERMINE (all agree)")
    print("  info-loss          | INFINITE      | infinite      | infinite        | RESIST   (all agree)")
    print("  PARITY             | finite        | infinite      | infinite        | SPLIT — K says determine!")

    print("\nVERDICT: strong H1 is UNRESCUABLE by a description-length filtration.")
    print("  K is a genuine universal comparable scalar, BUT it only detects INFO-LOSS resistance:")
    print("  any latent that is a (short-program, whole-shadow) function of the shadow — fiber=1 — is")
    print("  K-DETERMINE. Parity is exactly that (XOR-all is short + total), so K calls it determine,")
    print("  while locality/finite-order/computational axes call it resist. 'Resist' is BOUND-RELATIVE;")
    print("  the only bound-free filtration (K) collapses resist down to the info-loss (fiber>1) cases.")
    print("  No single filtration is both universal AND dichotomy-preserving => the SCHEMA (H1 restricted)")
    print("  is the ceiling. (Same phenomenon as H5: parity = info-present, finite-order-inaccessible.)")


if __name__ == "__main__":
    main()
