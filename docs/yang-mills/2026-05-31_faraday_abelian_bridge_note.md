# Faraday → Yang-Mills Bridge Note: the abelian boundary that explains the bounded null

- Note id: `2026-05-31_faraday_abelian_bridge_note`
- Type: cross-substrate **framing note** — not a run, not a receipt, not a new Yang-Mills claim
- Date: 2026-05-31
- Earned empirical basis (verified): Shadow Faraday Phase 7
  ([`../faraday/FARADAY_PHASE7_BOUNDARY.md`](../faraday/FARADAY_PHASE7_BOUNDARY.md), B7-topology)
  and Phase 8 ([`../faraday/FARADAY_PHASE8_BOUNDARY.md`](../faraday/FARADAY_PHASE8_BOUNDARY.md), A8/B8)
- Framing source: [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) §8 (§8.3 names this note)
- Explains: the YM Phase-2 informative powered null
  ([`receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md`](receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md))

## Claim (one line)

The Faraday-clean structural zero and the Yang-Mills small-loop bounded null are the
**abelian and non-abelian ends of one Wilson-loop operator**, and the abelian end
supplies an *algebraic reason* for the non-abelian upper limit: the linearity and
exact-form holonomy that make the abelian loop close as a freebie are exactly what
the non-abelian connection removes. This **explains** the v6a null; it does not
claim a Yang-Mills result.

## One operator, two ends

The shadow the program tests is a Wilson-loop holonomy — `P_shadow = ∮A` abelian,
`(½)Tr ∏ U` non-abelian. Maxwell is the **abelian baby case** of Yang-Mills, so the
*same operator* runs both substrates. The abelian end is now pinned by three exact
statements; the non-abelian end is the registered bounded null.

**Abelian end (exact, earned):**

- **§8.1 [proved].** Faraday closure is `dF = d(dA) = 0`, the Bianchi identity.
  Body-resistance is **zero by theorem** — a correctness check on the operator, with
  no resisting body to separate from.
- **§8.2 [earned, Phase 7].** On a non-contractible patch (`H¹ ≠ 0`) the loop tier
  `∮A = Φ` is state-insufficient yet control-sufficient (Aharonov-Bohm phase
  `= qΦ/ħ`); the gap is the `H¹` period — an **exact** regime-2, but on the
  **topological** axis, and on a **small** body (one integer per `H¹` generator).
- **§8.4 [earned, Phase 8].** Adding sources (`d*F = J`) yields **no new sharp
  regime-2**: the sourced sector is determined by its sources, and the Hodge
  decomposition localizes *all* exact regime-2 content back to the harmonic/AB
  sector. The abelian operator's exact-separation budget is fully accounted —
  zero in the bulk, one topological witness, nothing new from sources.

**Non-abelian end (the registered null):**

- The small-loop signature `{W11,W12,W13,W22}` does not rank-localize any powered
  held-out target on the registered SU(2) 3D cell — including the powered
  (split-half ICC `0.965`), disjoint (leakage CV-R² `−0.332`) finite-temperature
  Polyakov order parameter, where within-β bin-purity@5 `0.3042` sat at/below
  `CTRL_RAND 0.3292` (v6a). Bounded cell-local null.

## The bridge: why the non-abelian end loses the freebie

The abelian loop closes for a structural reason. `dF = 0` is **linear**, and the
holonomy of an exact form around a contractible loop vanishes identically — so the
shadow reconstructs the body by a geometric tautology. The non-abelian Bianchi
identity is

```
DF = dF + A ∧ F = 0,
```

which drags the connection `A` into the covariant derivative. The gauge-invariant
content (Wilson loops, area law) is no longer a linear functional of an exact form;
it carries the non-linear `A ∧ F` coupling. **The precise property that made the
abelian loop a freebie — linearity plus exact-form holonomy — is exactly what the
non-abelian case lacks.** A *compact* gauge-invariant shadow therefore has strictly
less to work with on the non-abelian side, and "harder to separate" is the
**expected**, not surprising, outcome.

## The sharp echo: AB local-blindness ↔ the v6a Polyakov null

Phase 7 states the abelian lesson exactly, in *local* vs *global* terms:

- the **local** tier `P_shadow^point = F` is **control-blind** to topological
  content — `F = 0` on every accessible path, yet the AB phase is nonzero;
- only the **global** loop `∮A` is control-sufficient.

v6a is the non-abelian echo of precisely this split. Its held-out target is the
**Polyakov loop** — a *global* temporal-wrap holonomy — and its signature is
*local* small Wilson loops. The local shadow does not carry the global target: the
same `local-blind / global-sufficient` divergence the abelian case proves exactly.
So the abelian boundary **predicts and explains** the v6a null rather than leaving
it an open surprise — the local-signature program meets, on the non-abelian side,
the wall the abelian side states as a theorem.

## What this does and does not license

- **Does:** give a principled, internally-derived algebraic *reason* the YM
  small-loop signature is bounded-null, grounded in a proved identity (§8.1) and an
  earned exact regime-2 (§8.2 / §8.4); turn the v6a result from "a powered null we
  observed" into "a powered null we can **explain** from an adjacent exact success";
  supply the external-review packet's Q2 with a candidate hypothesis for a lattice
  gauge theorist to validate or refute.
- **Does not:** prove the YM null is *necessary* — this is **framing** (an algebraic
  argument, not a lattice theorem); assert any Yang-Mills existence, mass-gap,
  confinement, continuum, or Clay result; claim a YM *positive* lives on the
  topological axis — the AB witness rides a **small** (one-integer) topological
  body, explicitly **not** the high-dimensional control body the frontier needs, and
  non-abelian topological-charge targets remain P0-deferred; license a new
  experimental probe — the bridge argues the null is **structural** (same outcome
  expected), the opposite of the "fresh motivation to expect a different outcome"
  the reopen clause requires. The lane stays PAUSED at the informative endpoint.

## Public Language Check

- [x] frames an *explanation* of a bounded null, not a Yang-Mills result
- [x] does not say "proves confinement" / "found a mass gap" / "Clay" / continuum
- [x] the abelian↔non-abelian bridge is tagged framing; the only earned empirical
  inputs are the verified Faraday Phase 7 / Phase 8 receipts
- [x] does not license a probe reopen — anti-p-hunting preserved
