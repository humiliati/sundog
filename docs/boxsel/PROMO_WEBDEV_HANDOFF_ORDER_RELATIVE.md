# Promo / Webdev Handoff — the Order-Relative Resolution Law

**From:** research (BoxSEL cross-lane conjecture slate)  
**To:** promo / webdev team  
**Date:** 2026-06-29  
**Status:** candidate asset. Surfacing is **OWNER-GATED**. Frozen-as-portfolio research — this handoff
is informational, **not** a directive to ship. Lead with the boundary, not the headline.

## The one surfaceable asset

The **Order-Relative Resolution Law** — a new axiom-clean Lean core, `Sundogcert/OrderRelative.lean`,
in the **public** `sundogcert` repo. It is the *only* public-eligible output of the recent slate.
Everything else stays internal (see the fence at the bottom).

## Copy-ready one-liner (de-laned)

> A machine-checked law: a budget-bounded process resolves a target exactly when the target's *order*
> is within budget — so "determine vs. resist" is just finite vs. infinite order. Proved once as a
> schema, instantiated on a parity core tied to a machine-checked sufficient-statistic order, with a
> theorem that the order is a **schema, not one universal number**.

## What it actually is (for accurate copy)

- A Lean `structure` + the single law `Resolves k t ↔ ord t ≤ k`; `determine ⟺ finite order`,
  `resist ⟺ infinite order` proved **once** and inherited by every instance.
- Two instances: a parity problem (its order proved equal to the machine-checked `suffStatOrder`) and
  a resist pole (order `= ⊤`); plus `order_is_schema_not_scalar`.
- Axiom-clean `[propext, Classical.choice, Quot.sound]`, in the build-enforced `#guard_msgs` gate,
  full `lake build` green (8530 jobs).

## Claim boundary — DO NOT

- Don't tie it to any named hard problem (P-vs-NP, Riemann, …) or to alignment / safety.
- Don't surface the internal axis names or the conjecture-slate framing (search / pressure / repertoire
  / determination / find-check as "we proved X about Y"). The **public** statement is the abstract law
  + the parity instance, full stop.
- Don't imply a universal cross-lane *number* — the machine-checked theorem says the **opposite**
  (schema, not scalar). Copy must not suggest "one order rules them all."
- Don't imply product / Ask-Sundog / abstention relevance anywhere public.
- It's portfolio research, not a launch.

## Claim boundary — OK to say

- "Machine-checked, axiom-clean, in a public referee-free Lean repo."
- "A clean order-relative resolution law: resolves ⟺ order ≤ budget; determine/resist = finite/∞."
- "Grounded on a parity instance tied to a machine-checked sufficient-statistic order."

## Pre-publish de-laning checklist (REQUIRED before the repo is featured)

1. **Commit + push status.** The module + root import + 6 `AxiomAudit` gate guards are currently
   **uncommitted and unpushed** (owner-gated). It cannot be linked from the site until it's on the
   public repo's `origin`.
2. **Docstring voice review.** `OrderRelative.lean`'s docstring references internal lane labels
   (BoxSEL, the conjecture slate, C1/C2/Phase-7g/C4). No patent / alignment / Phase-5 / product terms
   are present (checked), so this is a **voice/polish** call, not a leak fix — consistent with how
   `ShadowDecay` was de-laned and how `ParityNoSufficientStat` keeps its slate reference. Owner decides
   keep-vs-strip.

## Integration points (only if owner greenlights surfacing)

- `docs/SUNDOG_V_CERTIFICATE_LEAN.md` — the Lean-cores ledger (a row + count bump).
- `index.html` proof-grid + the "N LEAN CORES" header / SVG seal.
- `generality.html` tile, if it gets one.
- **Framing caveat:** this core is a *synthesis / law*, not a "cheaper-to-check-than-to-find worked
  example" like the existing cores. Do **not** force the "machine-check the deductive core, name the
  imported wall" template onto it — the owner should decide whether it's a new facet of the cores
  pillar or framed separately.

## The fence — what stays INTERNAL (must not reach any deployable surface)

The conjecture slate, C1–C4, the trace-based false-closure detector, pressure-abstention, the Phase-7
external-review packet, the BoxSEL lane, and all "flagship / Ask Sundog" framing are **frozen-as-
portfolio** and internal. Only the abstract Lean law + the parity instance are candidates for any
public surface.

---

*Sundog Research Lab — promo/webdev handoff. One candidate public asset (the Order-Relative Resolution
Law Lean core); honest boundary + de-laning checklist + the internal fence. Surfacing owner-gated.*
