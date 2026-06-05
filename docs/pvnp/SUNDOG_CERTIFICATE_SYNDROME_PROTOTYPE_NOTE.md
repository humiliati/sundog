# Sundog Certificate Problem — Syndrome/SIS Instance Prototype Note

Status: **prototype built; mechanism verified on a real regime; not a frozen
receipt.** This is the first empirical step on the §5 experiment of
[`SUNDOG_CERTIFICATE_PROBLEM.md`](SUNDOG_CERTIFICATE_PROBLEM.md) — it converts the
constructed instance (Candidate A) from an on-paper existence proof into a measured
prototype, applying the lane's verify-first discipline *before* freezing a regime.

Date: 2026-06-04 · Code: [`../../scripts/pvnp-certificate-syndrome.py`](../../scripts/pvnp-certificate-syndrome.py)
· Output: `results/pvnp/certificate-syndrome/syndrome_prototype.json` (gitignored).

## What was built

The syndrome/SIS constructed instance over GF(2), exactly as specified in §4:

- **Code:** random systematic binary `[n,k]` code, `G = [I_k | P]`,
  `H = [Pᵀ | I_{n−k}]`, with `G Hᵀ = 0` asserted.
- **Body** `(s, e)`: secret `s ∈ GF(2)^k`, sparse error `e` of weight `w`;
  observation `y = sG + e`.
- **Certificate** `σ = (z = Hy, witness e, tag)`; **safety predicate**
  `Safe := wt(e) ≤ τ`.
- **Verifier** `V` — three-valued, op-counted: `accept` iff a light witness is
  exhibited (`He = Hy`, `wt(e) ≤ τ`); `reject` iff a cheap sound lower bound on the
  coset weight exceeds `τ`; else `quarantine`.

Regime: `[48, 24]`, `w = 4`, `τ = 4` (a small, tractable prototype regime).
Deterministic (fixed seeds); a re-run is **byte-identical**.

## Mechanism check — all three properties hold (measured)

| Property | Result |
| --- | --- |
| **P1 — lossy by algebra** | `z = Hy = He` is independent of `s` (verified across samples; varying `s` with `e` fixed leaves `z` unchanged). **2²⁴ secrets map to each syndrome** → `s` is `qᵏ`-to-one gone — certified, not a measured FVE. |
| **P2 — cheap + sound** | Check op-count **2,376** vs naive decode **≈ 5,113,248** (**~2,150× cheaper**, an op-count gap by structure). **0 false accepts, 0 false rejects**, 200/200 safe bodies accepted with their witness. |
| **P3 — `s` one-way** | `z` invariant under an `s`-flip → the secret is information-theoretically absent from `σ`; recovering `e` (decoding) still yields nothing about `s`. |

## The capacity curve — the find-vs-check gap is real (measured)

A budgeted forger (enumerate light coset members **without** the witness = syndrome
decode) vs the constant cheap check, on 24 fresh targets:

| attacker budget | forge success | check ops |
| ---: | ---: | ---: |
| 200 | 0.00 | 2,376 |
| 1,000 | 0.00 | 2,376 |
| 5,000 | 0.00 | 2,376 |
| 20,000 | 0.04 | 2,376 |
| 80,000 | 0.38 | 2,376 |
| 213,052 (full low-weight space) | 1.00 | 2,376 |

This is the P-vs-NP-shaped axis, measured: **checking is cheap and flat** while
**forging rises with capacity**, with a visible breakpoint (~20k–80k budget on this
toy). That breakpoint is the prototype capacity-relative one-wayness threshold —
the thing every Phase-1 receipt reported as `capacity_threshold = not_estimated`,
now non-trivially present.

**This is the verify-first PASS that v4 never got.** The v4 basin mechanism looked
sound on paper and was falsified by measurement; this instance was *also* checked by
measurement and its three properties and capacity gap all hold.

## Honest limitations (what this is NOT)

1. **Prototype regime, not a frozen receipt.** `[48,24] w=4` is a toy: the threshold
   is small and the full space (213k) is enumerable. The mechanism and curve *shape*
   are demonstrated; an actual one-wayness claim needs a scaled regime where the
   threshold is large, frozen before the attacker runs.
2. **Naive attacker, not ISD.** The forger enumerates low-weight supports; a real
   information-set-decoding attacker is more efficient and would sharpen (and lower)
   the threshold. The frozen run must use a proper ISD attacker across the
   Small/Medium/Large ladder.
3. **Imported hardness.** The asymptotic spoof-hardness rests on the
   syndrome-decoding / SIS assumption; the prototype *exhibits* the gap on a toy but
   does not *prove* asymptotic hardness. This remains an existence proof, not a
   cryptographic one-wayness claim (the §0 / Track A guardrail).
4. **The cheap-reject branch is degenerate (the documented RISK 1 gap).** The cheap
   sound lower bound is left at its trivial value (0), so `V` is effectively
   *accept-with-witness or quarantine*; it never cheaply *rejects*. A non-trivial
   cheap lower bound can itself be decoding-hard — exactly RISK 1 — so the
   false-quarantine band (failure mode 6.2) must be pre-registered in the frozen
   slate, not assumed away. Soundness of *accept* is by construction
   (`accept ⟺ light witness ⟺ within τ of the code = safe`).
5. **No measured `C` at scale.** The prototype threshold is on the toy; the frozen
   §5 run is what supplies a regime-anchored capacity-relative threshold.

## Next step (toward the frozen §5 receipt)

Pre-register (freeze) a scaled regime `(n, k, w, τ)` and the attacker ladder before
running: a proper ISD forger at Small/Medium/Large capacity; `wt(e)` and `s` as
scoring labels only (never verifier inputs); the false-quarantine band of the cheap
bound as the explicit 6.2 boundary; and the verdict branches (one-way below `C`,
broken above `C`, vacuous, sufficiency-quarantine, overhead). That run converts §4
from existence proof to a Sundog receipt with a measured capacity threshold.
