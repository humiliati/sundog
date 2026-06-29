#!/usr/bin/env python3
"""Parity-barrier slate P-3: instrumenting where the Maynard-Tao pigeonhole stalls.

A DIDACTIC receipt, not an attack. It makes two things concrete on real integers:

(A) Admissibility -- the obstruction a sieve CAN see. A k-tuple H is admissible iff for
    every prime p it misses some residue class mod p. {0,2,4} covers all classes mod 3
    (always contains a multiple of 3) and is excluded; {0,2,6} is admissible. This is a
    congruence/local obstruction -- exactly the kind sieves handle.

(B) The parity principle -- the obstruction a sieve CANNOT see. A sieve of level D = N^theta
    only constrains prime factors <= D, so among the survivors in [N,2N] (smallest prime
    factor > D) it computes the COUNT (= P1 + P2 + ...) but is blind to the parity of
    Omega(n). For theta > 1/3 survivors are only primes (P1) and semiprimes (P2), and
        P1 = (count - parity_sum) / 2,   parity_sum = sum lambda(n) = P2 - P1.
    The sieve delivers `count`; `parity_sum` is the parity barrier. The "/2" IS the
    factor-of-2 loss: without parity_sum, count alone gives only the P_{<=2} ("almost
    prime") bound -- the Maynard-Tao pigeonhole's exact stall, visualized.

Imported wall (named, NOT proved): the parity barrier is why a sieve cannot lower-bound
primes in a single sparse sequence (Selberg parity principle; Bombieri's asymptotic
sieve), why classical sieves reached only Chen's P2, and why Maynard-Tao (M_k ~ log k,
Maynard 2015) yields BOUNDED GAPS among many shifts, never a specified pair: the twin
tuple k=2 has M_2 far below threshold and is permanently out of reach.

DOES NOT attempt twin primes, bounded gaps, Chowla, Sarnak, or to breach the barrier.
Run: python scripts/parity_p3_pigeonhole_stall.py
"""
import numpy as np

N = 1_000_000          # study the interval [N, 2N]
THETAS = [0.50, 0.40, 0.34, 0.30]   # sieve levels D = N^theta (>1/3 => 2-bucket; 0.30 shows P3)


def is_admissible(H):
    """H admissible iff for every prime p <= len(H) it misses a residue class mod p."""
    H = list(H)
    k = len(H)
    for p in range(2, k + 1):
        if all(spf_small(p)):  # p prime?
            residues = {h % p for h in H}
            if len(residues) == p:
                return False, p  # covers all classes mod p -> always a multiple of p
    return True, None


def spf_small(p):
    """Trivial primality for the tiny p <= k in admissibility (k is small)."""
    return [p % d != 0 for d in range(2, int(p ** 0.5) + 1)] or [True]


def factor_sieve(M):
    """smallest-prime-factor and Omega (with multiplicity) for 0..M."""
    spf = np.zeros(M + 1, dtype=np.int64)
    omega = np.zeros(M + 1, dtype=np.int16)
    for p in range(2, M + 1):
        if spf[p] == 0:                 # p is prime
            sl = spf[p::p]
            sl[sl == 0] = p             # view -> writes back the smallest prime factor
            pe = p
            while pe <= M:
                omega[pe::pe] += 1
                pe *= p
    return spf, omega


def main():
    print(f"PARITY_P3_PIGEONHOLE_STALL   interval [{N}, {2*N}]", flush=True)

    # ---- (A) admissibility: the obstruction the sieve CAN see ----
    print("\n(A) Admissibility (congruence obstruction -- a sieve handles this):", flush=True)
    for H in [(0, 2, 4), (0, 2, 6), (0, 2, 6, 8, 12)]:
        ok, p = is_admissible(H)
        why = "admissible" if ok else f"INADMISSIBLE (covers all residues mod {p})"
        print(f"    H={H!s:<16} -> {why}", flush=True)

    # ---- (B) the parity principle: the obstruction the sieve CANNOT see ----
    spf, omega = factor_sieve(2 * N)
    idx = np.arange(N, 2 * N + 1)
    omg = omega[N:2 * N + 1]
    lam = np.where(omg % 2 == 0, 1, -1)
    spfv = spf[N:2 * N + 1]

    pi_interval = int((omg == 1).sum())   # primes in [N,2N], sanity anchor
    print(f"\n(B) Parity principle on survivors (smallest prime factor > D = N^theta):", flush=True)
    print(f"    sanity: primes in [N,2N] = pi(2N)-pi(N) = {pi_interval}", flush=True)
    print(f"\n    theta     D     survivors   count=P1+P2+..  parity_sum=Sum_lambda    P1    P2    P3   (count-psum)/2", flush=True)
    for th in THETAS:
        D = N ** th
        surv = spfv > D
        cnt = int(surv.sum())
        psum = int(lam[surv].sum())
        p1 = int((surv & (omg == 1)).sum())
        p2 = int((surv & (omg == 2)).sum())
        p3 = int((surv & (omg == 3)).sum())
        recovered = (cnt - psum) // 2
        flag = "= P1 (exact)" if recovered == p1 and p3 == 0 else "!= P1 (P3 present: >2 buckets)"
        print(f"    {th:.2f}  {D:7.0f}   {cnt:9d}   {cnt:12d}   {psum:18d}   {p1:6d} {p2:5d} {p3:4d}   {recovered:7d} {flag}",
              flush=True)

    # ---- the money shot at theta = 1/2 ----
    th = 0.50
    D = N ** th
    surv = spfv > D
    cnt = int(surv.sum()); psum = int(lam[surv].sum())
    p1 = int((surv & (omg == 1)).sum()); p2 = int((surv & (omg == 2)).sum())
    print(f"\n  THE STALL (theta=1/2, D={D:.0f}): the sieve computes count = {cnt} (P1+P2).", flush=True)
    print(f"    To extract primes it needs P1 = (count - parity_sum)/2 = ({cnt} - {psum})/2 = {(cnt-psum)//2}.", flush=True)
    print(f"    P2/P1 = {p2/p1:.3f}: the almost-prime (P2) contamination the sieve canNOT remove.", flush=True)
    print(f"    `count` is a level-D divisor sum (sieve-computable). `parity_sum = Sum lambda` is the", flush=True)
    print(f"    PARITY BARRIER -- not a level-D quantity. Without it, count alone gives only the P_<=2 bound.", flush=True)
    print(f"    That missing factor of 2 is exactly where Maynard-Tao stalls: bounded gaps, never twins.", flush=True)
    print(f"\n  (Imported wall: Selberg parity principle / Bombieri asymptotic sieve; M_k ~ log k.", flush=True)
    print(f"   This INSTRUMENTS the stall; it does NOT attempt twin primes or breach the barrier.)", flush=True)


if __name__ == "__main__":
    main()
