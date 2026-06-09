#!/usr/bin/env python
"""C1 (slate `wm13nclfe`) — the Reed–Solomon evaluation certificate, GF(7) sanity.

A small numerical witness for the machine-checked Lean pillar (sundogcert/Sundogcert/RSCertificate.lean):
the EVALUATION-dual of the syndrome/parity-check certificate. A codeword is the evaluation vector of a
degree-<k message polynomial at n distinct nodes over a field (here GF(7)).
  * rs_unique (DETERMINE):   k evaluations DETERMINE the unique deg-<k polynomial — the encoding is
                             injective on deg-<k messages.
  * accept_sound (cheap CHECK): a claimed decoding f is verified by one forward pass (evaluate + count
                             agreements), O(n*k).
  * corruption_fiber_nontrivial (LOSE): many received words decode to the SAME f — the shadow DETERMINES
                             the message but LOSES which corruptions were applied.
  * unique_decoding: within the unique-decoding radius (2*tau < d = n-k+1) the decoding is THE unique one.
The imported wall (named, NOT here): list-decoding past the radius is NP-hard for general RS (Guruswami–
Vardy 2005). NOT public-eligible. Run: python scripts/rs_cert.py
"""
import sys
import itertools
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

P, N, K, TAU = 7, 5, 2, 1                      # GF(7); n=5 nodes; deg<2 messages; radius tau=1
NODES = np.arange(N)                           # distinct evaluation nodes 0..4


def encode(coeffs):
    """Evaluation vector of a deg-<k polynomial (coeffs low→high) at the nodes, over GF(P)."""
    return np.array([sum(int(c) * pow(int(x), j, P) for j, c in enumerate(coeffs)) % P for x in NODES])


def agree(coeffs, y):
    return int(np.sum(encode(coeffs) == y))    # cheap O(n*k) forward check


def decodes(coeffs, y):
    return agree(coeffs, y) >= N - TAU         # deg<k (by construction) and ≤ tau disagreements


def main():
    print("=" * 78)
    print(f"C1 — Reed–Solomon evaluation certificate, GF({P})  [n={N}, k={K}, τ={TAU}]")
    print("=" * 78)

    # (1) rs_unique / DETERMINE: k evals determine the deg<k poly — encoding injective on deg<k messages
    allpolys = list(itertools.product(range(P), repeat=K))
    codewords = [tuple(encode(c)) for c in allpolys]
    injective = len(set(codewords)) == len(allpolys)
    print(f"\n(1) rs_unique (DETERMINE): {len(allpolys)} deg<{K} polys → {len(set(codewords))} DISTINCT "
          f"codewords  [{'injective ✓' if injective else 'FAIL'}]")

    # (2) corruption_fiber_nontrivial / LOSE: many words decode to the SAME f
    f = (3, 2)                                  # message 3 + 2x
    c = encode(f)
    fiber = {tuple(c)}
    for i in range(N):
        for v in range(P):
            if v != c[i]:
                y = c.copy(); y[i] = v
                if decodes(f, y):
                    fiber.add(tuple(y))
    print(f"(2) corruption_fiber_nontrivial (LOSE): message {f}, codeword {tuple(int(x) for x in c)} → "
          f"{len(fiber)} distinct words all decode to f")
    print(f"    (the clean codeword + {len(fiber)-1} single-symbol corruptions within τ={TAU}; "
          f"DETERMINEs f, LOSEs which corruption)")

    # (3) unique_decoding within the radius: no OTHER deg<k poly decodes any word in the fiber
    d = N - K + 1                               # MDS minimum distance
    within_radius = 2 * TAU < d
    other_fit = False
    for y in fiber:
        ya = np.array(y)
        for g in allpolys:
            if g != f and decodes(g, ya):
                other_fit = True
    print(f"(3) unique_decoding: d=n-k+1={d}, 2τ={2*TAU} < d ⇒ within radius={within_radius}; "
          f"a second deg<{K} poly decoding a fiber word: {other_fit}  [{'unique ✓' if not other_fit else 'FAIL'}]")

    # (4) cheap CHECK vs the message space: verify is one O(n*k) pass
    print(f"(4) cheap CHECK: verifying a claimed f = {N}·{K} = {N*K} field ops (evaluate + count); "
          f"the message space is {P}^{K} = {P**K} (find = search, list-decoding past radius is the wall)")

    ok = injective and len(fiber) >= 2 and (not other_fit) and within_radius
    print("\n" + "=" * 78)
    print("RESULT: the RS evaluation shadow DETERMINES the message (k evals ⇒ unique deg<k poly), the cheap"
          if ok else "RESULT: FAIL — check the sanity")
    if ok:
        print("  check is sound, within the radius the decoding is unique, yet MANY corrupted words decode")
        print("  to the same f (determine/lose). The evaluation-dual of the syndrome certificate, machine-")
        print("  checked in sundogcert/Sundogcert/RSCertificate.lean.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
