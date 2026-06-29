#!/usr/bin/env python3
"""Sufficient-statistic-order slate, H3: is parity the "check-hard pole" of the
find/check ledger?  The hook's own kill warns that "check" equivocates. This
receipt tests it on the two distinct meanings:

(1) VERIFY-a-witness (the CheckCost / NP sense): given n AND its prime
    factorization as a witness, lambda(n) = (-1)^Omega(n) is POLY to verify
    (check the product = n, check each factor prime, count factors, take parity).
    => parity is CHECK-EASY — the same column as the syndrome cert. FINDING the
    factorization (factoring) is the hard part. So parity is (find-hard, verify-easy).

(2) PREDICT-next (the P-2 / sufficient-statistic sense): lambda(n) from the
    lambda-SEQUENCE history is not determined at ANY finite order (sigma = infinity).

These are DIFFERENT axes. Calling parity "check-hard" conflates verify-cost with
predict-order = the category error. Receipt prints both faces + the placement.
Run: python scripts/suffstat_h3_verify_vs_predict.py
"""
import numpy as np
from math import isqrt


def is_prime(p):
    if p < 2:
        return False
    if p % 2 == 0:
        return p == 2
    for d in range(3, isqrt(p) + 1, 2):
        if p % d == 0:
            return False
    return True


def verify_lambda(n, factors):
    """VERIFY lambda(n) given the factorization witness. Returns (lambda, op_count).
    op_count ~ |witness| (mults) + per-factor primality = polynomial in log n."""
    ops = 0
    prod = 1
    for p in factors:
        assert is_prime(p), f"witness invalid: {p} not prime"
        prod *= p
        ops += 1
    assert prod == n, f"witness invalid: product {prod} != {n}"
    return (-1) ** (len(factors) % 2), ops + len(factors)


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


def predict_order_from_history(lam, K):
    """least k s.t. the preceding-k lambda-window determines lambda(n); else inf.
    For each k, is there a window with both +1 and -1 continuations?"""
    n = len(lam)
    for k in range(1, K + 1):
        seen = {}
        ambiguous = False
        for i in range(k, n):
            key = lam[i - k:i].tobytes()
            nxt = lam[i]
            if key in seen:
                if seen[key] != nxt:
                    ambiguous = True
                    break
            else:
                seen[key] = nxt
        if not ambiguous:
            return k
    return None  # = infinity over the tested range


def main():
    print("SUFFSTAT_H3_VERIFY_VS_PREDICT   (does 'check' equivocate? yes — two axes)\n")

    print("(1) VERIFY-a-witness: lambda(n) given the factorization is POLY (parity is CHECK-EASY)")
    examples = [
        (997, [997]),
        (8, [2, 2, 2]),
        (210, [2, 3, 5, 7]),
        (30030, [2, 3, 5, 7, 11, 13]),
    ]
    for n, factors in examples:
        lam, ops = verify_lambda(n, factors)
        print(f"    n={n:<6} witness={str(factors):<22} -> lambda={lam:+d}  verified in {ops} ops "
              f"(Omega={len(factors)})")
    print("    FIND (factoring n) is the hard direction; VERIFY is linear in the witness. "
          "Same check-easy column as the syndrome cert.")

    print("\n(2) PREDICT-next from the lambda-SEQUENCE history (the P-2 / sufficient-stat axis)")
    N = 200_000
    lam = liouville(N)
    k = predict_order_from_history(lam, 12)
    if k is None:
        print(f"    sigma_predict = infinity (no finite order k<=12 determines lambda(n) from its history)")
    else:
        print(f"    sigma_predict = {k}")
    # show the immediate ambiguity at k=1 (same one-symbol context, both continuations)
    amb = {}
    witness = None
    for i in range(1, N):
        key = int(lam[i - 1])
        if key in amb and amb[key] != lam[i]:
            witness = (key, amb[key], int(lam[i]))
            break
        amb.setdefault(key, lam[i])
    print(f"    ambiguous already at k=1: a context lambda(n-1)={witness[0]:+d} is followed by both "
          f"{witness[1]:+d} and {witness[2]:+d}.")

    print("\nPLACEMENT (three columns, not one):")
    print("    object         | find-cost      | verify-witness | predict-order sigma")
    print("    syndrome cert  | hard (NP)      | EASY (O(mn))   | finite (verdict)")
    print("    parity lambda  | hard (factor)  | EASY (factzn)  | INFINITE")
    print("    halo           | n/a            | n/a            | finite (~1 param)")

    print("\nVERDICT: 'check' equivocates. find/check (verify-a-witness) and predict-order sigma are")
    print("  ORTHOGONAL axes. Parity is VERIFY-EASY + PREDICT-INFINITE. Calling it the 'check-hard pole'")
    print("  of the find/check ledger imports its predict-infinity into the verify column = CATEGORY ERROR.")
    print("  H3-as-stated is KILLED; the salvage (the two axes are distinct) is the deliverable.")


if __name__ == "__main__":
    main()
