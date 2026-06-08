# Sundog Certificate — Machine-Checked Core (Lean)

> **Deductive core of the [P-vs-NP certificate-syndrome lane](SUNDOG_V_P_V_NP.md).**
> The certificate's **soundness and lossiness** are machine-checked in Lean 4 — `sorry`-free,
> axiom-clean, **referee-free**. The kernel re-checks every theorem in seconds, so the *validity* of
> this core is author-independent. The decoding-hardness assumption (information-set decoding / SIS) is
> **imported, not proven** — Lean certifies the deductive core, never the hardness.

**Public and reproducible:**
[`github.com/humiliati/sundogcert`](https://github.com/humiliati/sundogcert) — `lake build` re-certifies
every theorem; `#print axioms` shows only `[propext, Classical.choice, Quot.sound]` (no `sorry`, no
`native_decide`, no trusted compiler step). Lean `v4.30.0`, mathlib `v4.30.0`.

Working hook:

> Safe policies may be hard to find, but a sound certificate of safety is cheap to check — and now its
> soundness is something a *machine* checks, not a referee.

## What is machine-checked

- **Lossiness by algebra.** The syndrome `H(sG + e) = He` is independent of the secret `s`
  (`syndrome_independent_of_secret`); every message maps to the same syndrome; there are `|F|ᵏ` bodies
  per syndrome (`secret_bits_lost`). The shadow discards `k·log|F|` bits — forced by the algebra, not
  assumed.
- **Soundness.** `accept ⟹ Safe` — the only route to *accept* is an exhibited light witness, which *is*
  the proof (`accept_sound`); no accepted body is unsafe (`no_passing_unsafe`); `reject ⟹ ¬Safe` under a
  sound lower bound (`reject_sound`).

The trust surface is ~30 lines: the scheme definitions and the meaning of *Safe*. Everything above them
is kernel-checked. Peer review shrinks from "trust the proof" to "audit the statement."

## The wall, named

Lean certifies **soundness + lossiness only.** The certificate's security rests on a decoding-hardness
assumption that is **imported, not proven** — hardness is not a mathlib theorem. Every "Lean-verified"
here means the deductive core, never the hardness.

## The reject bound, fully characterized

The load-bearing reject bound `colWeightLb` is pinned down from every direction, each fact kernel-verified:

| regime | behavior | theorem |
|---|---|---|
| any basis | **sound** — never exceeds the true witness weight | `colWeightLb_sound` |
| uniform `H` | **tight** — equals the true distance; reject threshold scales linearly, `τ = n/2 − 1` | `scaling_law` |
| denser `H`, same code | **loose** — collapses to `0`, purely from the basis | `looseness` |
| general | capped by `‖syndrome‖ / density` | `colWeightLb_le_card_div` |

Items (loose) and (general) are **completeness** phenomena, not soundness breaks: a collapsed bound still
never over-claims — it quarantines where it cannot reject. Soundness never depends on the basis; only the
bound's *strength* does.

## The frontier: the looseness is the shadow of the hardness

A *cheap, basis-robust, tight* reject bound — one that doesn't degrade when the parity-check is chosen
adversarially — would return the true minimum coset weight on every basis. That **would be a fast
decoder**: it would solve the very problem (information-set decoding) whose hardness the certificate
imports. So the basis-dependence of `colWeightLb` is not a defect to be patched away — it is the *shadow*
of the hardness assumption. The honest open question is quantitative: how large is the gap between a cheap
bound and the true coset weight, as a function of the decoding margin.

## Relation to the P-vs-NP lane

The certificate-syndrome receipts (v1–v6) measure the **empirical** side — cheaper to check than to find
(op-count cost certificate `0.949 ≤ 1.0`), safety green. This ledger is the **deductive** complement: the
soundness and lossiness those receipts rely on are now machine-checked, axiom-clean. The two are
orthogonal, and neither proves the decoding hardness — which both import.

## Status

**BOUNDED-POSITIVE deductive core.** Soundness + lossiness: machine-checked, axiom-clean, referee-free.
Hardness: imported. Not a cryptographic one-wayness claim; not a claim about P versus NP — a verification
*methodology* whose validity anyone can reproduce, and a clean coding-theory characterization of one bound.
