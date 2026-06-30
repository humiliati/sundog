# The Order-Relative Resolution Law — named + Lean-anchored (B)

**Date:** 2026-06-29  
**Status:** `LANDED` — machine-checked, axiom-clean, build-enforced. The cross-lane conjecture slate's
([BOXSEL_CONJECTURE_SLATE.md](BOXSEL_CONJECTURE_SLATE.md)) one statement, with the axes as instances.
Lean module in the public repo `Dev/sundogcert`; frozen-as-portfolio.

## The name and the one statement

**Order-Relative Resolution Law.** A bounded process with budget `k` **resolves** a target iff the
target's **order** is `≤ k`; the determine/resist split is finite-order vs infinite-order.

```text
Resolves k t  ↔  ord t ≤ k        (ord : Target → ℕ∞)
determine  ⟺  ord t  finite       resist  ⟺  ord t = ⊤
```

## The Lean anchor

`Sundogcert/OrderRelative.lean` (namespace `Sundog.OrderRelative`), full `lake build` green (8530
jobs), all headline theorems in the build-enforced `AxiomAudit` `#guard_msgs` gate, axiom profile
`[propext, Classical.choice, Quot.sound]`:

- `structure Problem` — the one statement: `(Target, ord : Target → ℕ∞, Resolves : ℕ → Target → Prop)`
  with the single law field `resolves_iff : Resolves k t ↔ ord t ≤ k`.
- `budget_monotone` — more budget keeps resolving (from the law alone).
- `resolvable_iff_finite` — **DETERMINE ⟺ finite order**: some finite budget resolves `t` iff `ord t ≠ ⊤`.
- `resists_iff_infinite` — **RESIST ⟺ infinite order**: no finite budget resolves `t` iff `ord t = ⊤`.

These three are the law's *content*, proved **once** from `resolves_iff` — every instance inherits them.

### Instances (the axes plug in)

- `parityProblem n` — the **determination axis**, grounded: `Resolves k = (∃ sufficient subset-parity of
  card ≤ k)`, and `parityProblem_ord` proves its `ord` equals the machine-checked σ-order
  `ParityNoSufficientStat.suffStatOrder n`. So the schema's `ord` is the *same* order the parity barrier
  already certified.
- `resistPole` — the σ=∞ / `λ` / full-history end (`ord = ⊤`); `resistPole_resists` shows no finite
  budget resolves it, instantiating `resists_iff_infinite`.
- `localityProblem n d` — a **second grounded axis** (added 2026-06-29): coordinate-locality. The target
  `prefixParity n d` depends only on coordinates `< d`; `prefixSufficient_iff` proves the first `k`
  coordinates suffice iff `d ≤ k`, so its `ord = d` (the prefix width — a *different* filtration from
  parity's subset-parity). `localityProblem_ord_ne_top` shows it is determine-only (always finite,
  never `⊤`) — echoing the σ-slate's H2 (locality cannot host the resist pole). `two_axes_and_pole`
  exhibits three orders on one object: locality `d`, parity `n`, pole `⊤`.
- `rationalReachProblem` / `irrationalReachProblem` — a **third grounded axis** (added 2026-06-29):
  search reachability (C1). The order is the **denominator budget**: `DenomReaches x k` = reachable by
  a rational of denominator ≤ k. A rational target `q₀` has `ord = q₀.den` (`rationalReachProblem_ord`);
  an **irrational** target has `ord = ⊤` and genuinely resists (`irrationalReach_resists`) — an
  **earned** resist pole, unlike the fiat `resistPole`. `search_resist_sqrt_two` machine-checks it for
  `√2`; the BoxSEL optimum `(9+√17)/32` is irrational the same way (the C1 search-resist). NB: C4's
  verify-vs-predict *mode-vector* was **not** anchored — on the bit-parity toy verify-cost = determine-
  cost (no divergence); C4's gap is λ-specific (the factorization witness), not Lean-toy-able.

### The honesty guard, machine-checked

`order_is_schema_not_scalar` — two instances assign **incomparable** orders to the same object (a parity
problem has finite order `n`, the resist pole has order `⊤`). This is the C3-census / σ-slate-H1 finding
encoded in Lean: the law anchors a **SCHEMA** (a per-axis `ord` + `Resolves` + the law), **not** the
"one comparable scalar" form that was already falsified. The order is per-instance — the C4 mode-vector
/ H1's ≥6 filtrations.

## What it abstracts

| axis | the "order" | source |
|---|---|---|
| determination (σ-schema) | sufficient-statistic order | `suffStatOrder` (Lean instance) |
| search reach (C1) | grid denominator | `PHASE4L_SEARCH_ORDER_FILTRATION.md` |
| pressure reach (C2) | probe order | `C2_PRESSURE_ABSTENTION_BREAK.md` |
| pressure repertoire (Phase 7g) | failure-shape set | `PHASE7G_PRESSURE_REPERTOIRE_RESULTS.md` |
| find/check (C4) | (verify-order, predict-order) — mode-vector | `C4_FIND_CHECK_ORDER.md` |

## Honest boundary

This Lean-anchors the **schema** and proves its content (determine/resist dichotomy +
budget-monotonicity) once, with the determination axis (parity) as a *grounded* instance tied to the
machine-checked σ-order, plus the resist pole. The other four axes are shown to **fit** the schema by
their sundog probes (C1/C2/7g/C4 above) — they are not each separately re-instantiated in Lean (only the
determination axis is). The closing theorem makes explicit that this is a schema, not a universal scalar
— so naming the "law" does **not** relitigate the falsified one-scalar form. Toy/portfolio; the Lean repo
is public-eligible but pushing is owner-gated.

## Files

- `Dev/sundogcert/Sundogcert/OrderRelative.lean` — the law + theorems + instances (axiom-clean, gated).
- `Dev/sundogcert/Sundogcert/AxiomAudit.lean` — six `#guard_msgs` entries pinning the axiom profile.
- This note + [BOXSEL_CONJECTURE_SLATE.md](BOXSEL_CONJECTURE_SLATE.md).

---

*Sundog Research Lab — the Order-Relative Resolution Law (B). The slate's one statement, machine-checked
and axiom-clean: resolves ⟺ order ≤ budget; determine/resist = finite/∞ order; and the order is a schema
(parity instance grounded to `suffStatOrder`, resist pole, incomparable across instances), not a scalar.
Internal; frozen-as-portfolio.*
