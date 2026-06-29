# BoxSEL → Cross-Lane Conjecture Slate

**Date:** 2026-06-29  
**Status:** `CONJECTURE_SLATE` — proposals only. Nothing here is run, proven, or promoted. Each
entry names a Sundog hook (an instrument that already exists) and an explicit falsifier. Seeded by
the BoxSEL search-gap finding; reaches across the parked portfolio lanes.

## The spine observation

Strip the lanes down and they keep producing one shape: **a bounded process versus a truth it
structurally cannot reach.**

```text
shadow tower   : lossy averaging DETERMINES discrete/topological, RESISTS continuous
parity barrier : the sieve is SIGN-BLIND to λ(n) (needs Type-II; can't see the signed global thing)
algo-approx    : monotone/cancellation-free circuits can't see the SIGNED thing (natural-proofs wall)
BoxSEL         : blind search REACHES generic optima, MISSES the low-measure structured one;
                 restart-variance is blind to STABLE false closure
```

**Honesty constraint (do not re-litigate):** the σ-order slate already **falsified** the strong
"one comparable scalar" form — each lane has its own filtration, σ's are not cross-comparable, only
parity σ=∞ is Lean (`project_sundog_suffstat_order_slate`). So this slate does **not** claim one
universal order. The claim under test is weaker and (we think) truer: **the same mechanism wearing
different filtrations**, with the inroad being the *translation* between them, not their identity.

---

## C1 — The search gap is a determine/resist instance — **LANDED 2026-06-29** (`search_resist_confirmed`)

> Phase 4l ([PHASE4L_SEARCH_ORDER_FILTRATION.md](PHASE4L_SEARCH_ORDER_FILTRATION.md), test 16/16):
> on the nested chain 6│12│24, `grid_min` = 1/2, 4/9, 4/9 (monotone). DETERMINE side ½ is finite-order
> (σ_search = 6); the closed-form optimum `(9+√17)/32` is infinite-order over the chain (never reached,
> not even within +0.02). Measured **resist margin = 4/9 − (9+√17)/32 = (47−9√17)/288 ≈ 0.0343**.
> Non-vacuity guard passed (meter live, plateau measured, monotone). Resist is relative to bounded
> *unstructured* search — Phase-4c's witness-seeded Nelder–Mead reaches it (reachable by analysis, not
> naive search). The one filtration the σ-order schema lacks (reachability, not determination), banked.


**Statement.** Blind search is a *lossy shadow of the feasible set*: it determines the high-measure /
generic optima and resists the low-measure / structured ones — continuous-resist, but over
configurations instead of hidden variables. Schema form: within the *search* filtration (order =
restart count / move-locality), reachable-at-order-`k` ⟺ the optimum's search-order ≤ `k`.

**Hook.** BoxSEL is the rare case where the true optimum is known in closed form,
`inf I_box^n = (9+√17)/32`, so search-order scaling can be measured against ground truth (Phase 4c
already showed every from-scratch optimizer misses it).

**Falsifier.** A low-order search reliably reaches the structured optimum with no order-scaling.

**Promise / marginality.** Promoted to the slate's live inroad by the C3 census: search-reachability
is a **7th filtration** absent from the suffstat-order schema (whose six are all *determination*
filtrations) — so this is the one place the spine genuinely *extends*, not just restates, the
already-banked σ-order work. BoxSEL is the rare closed-form-ground-truth instance. Sharpen here next.
Portfolio.

## C2 — Orthogonal-pressure abstention — **BROKEN-AS-STATED 2026-06-29** (`C2_BROKEN_order_relative`)

> Parity-substrate break ([C2_PRESSURE_ABSTENTION_BREAK.md](C2_PRESSURE_ABSTENTION_BREAK.md), test
> 18/18): any oracle-free pressure family is finite-order. A reasoner closes at order 2; the detector
> escalates pressure to budget M=8. The σ=∞ Liouville false closure is invisible to **every** pressure
> (all `own_r2` ≤ 0.10) → ACCEPTED → **soundness FALSIFIED**. The in-budget LFSR(5) guard *is* caught
> (moves at order 4 → apparatus live); the *finite* above-budget LFSR(12) is **also** missed (σ=12 > 8)
> → it's the budget, not ∞-mystique. **Repair (bounded-positive): orthogonal-pressure abstention is
> sound only for false closures of order ≤ the pressure budget M; the detector's reach IS the pressure
> family's order** — ties C2 to C1 and the σ-order schema. A pressure-based abstention rule must declare
> its order budget and treat σ>M as out of scope, not silently claim soundness on resist-side closures.


**Statement.** Define false closure substrate-independently: *the answer interval moves under a
perturbation the reasoner did not itself apply.* Conjecture that "moves-under-unapplied-pressure" is
a **sound, oracle-free** indicator of false closure across SEL/box, the shadow tower, and
(aspirationally) real retrieval.

**Hook.** BoxSEL Phase 7b/7d already proved the toy version — the pressure trace catches *stable*
false closure that restart-variance is structurally blind to (the 24-pair equivalence certificate).

**Falsifier.** A stable false closure invisible to *every* orthogonal pressure (kills soundness).

**Promise / marginality.** The **clarity** generalization the Phase-7 review hunt wanted — a
definition, not "more corpora" — and the only one that is genuinely flagship-adjacent (it is the
accept/widen/abstain rule). Note neutrally: relevance to Ask Sundog is real but unbuilt; this is a
research conjecture, not a product claim.

## C3 — The one-wall census (the bold unifier)

**Statement.** Every imported wall in the portfolio is the same obstruction: a **local / sign-blind**
process (sieve, monotone circuit, mean-pool, restart-variance) cannot resolve a **global / signed**
invariant (Möbius λ, cancellation, continuous magnitude, the extremal direction). Not one number —
one *mechanism*, with a per-lane translation.

**Hook.** The algo-approx lane already wrote "every imported wall = cancellation" (Slate-2 N-2);
parity, shadow, and BoxSEL are three further exhibits.

**Falsifier.** A lane whose wall is demonstrably **not** sign-blindness / cancellation (e.g. a wall
that is pure locality, or pure information-theoretic capacity, with no signed-invariant in sight).

**Promise / marginality.** Highest ceiling, highest risk — the synthesis trap the Phase-7 review
taught us to avoid. **First move was to break it, not confirm it.**

### Census verification (2026-06-29, against the lane ledgers)

Verdict: **strong C3 is FALSE, and the surviving schema is NOT new — it is the already-banked
sufficient-statistic-order schema** ([[project_sundog_suffstat_order_slate]], H1). The census changed
the conclusion:

| lane | wall, per its ledger | filtration | side |
|---|---|---|---|
| parity | sieve sees `count`, blind to `Σλ`; `partial_not_sufficient` (Lean) | predictive | resist σ=∞ |
| algo-approx | monotone can't compute signed; Jerrum–Snir | find/verify | resist (cancellation) |
| shadow | RESIST⟺‖charFun‖→0, DETERMINE⟺finite mean; H9-strong resist exemplar **is parity** | moment | both poles |
| Yang–Mills | `NEG-A no_rank_local_structure`; local loop blind to global `∮A` (Faraday §8.3) | coordinate-locality | determine-side (topological) |
| BoxSEL | blind search misses the low-measure structured optimum | **search reachability** | — (new) |
| NSE C1 | marginal because **RECOVERABLE** (FVE~0.99); resist is *constructed* (syndrome cert), not scaled | parametric | determine-side |

Three corrections forced by the docs:
- **"Sign-blindness" is not the wall** — it is one filtration (parity's predictive σ=∞). The lanes
  carry ≥6 distinct filtrations (H1's finding). The operator "kernel/range" picture is the same
  content in geometric dress, **not a new unifier.**
- **NSE is not a counterexample wall** — its marginality is *recoverability* (determine-side, finite
  σ), not a capacity ceiling; resistance is *constructible* (syndrome cert), not scaled. (Memory-census
  mislabeled it "capacity.")
- **Yang–Mills is the coordinate-locality instance, determine-side** — the missed invariant is
  topological (`∮A`/Polyakov); local observables are the wrong order, not sign-blind.

**The one genuinely new thing:** BoxSEL's **search-reachability** filtration is *not* among H1's six
(those are all *determination* filtrations; search is about what a bounded process *reaches*). So C3
collapses into the suffstat schema; its only live residue is **C1: search-reachability as a candidate
7th filtration**, with the one lane that has closed-form ground truth to test it. **C3 closed as
redundant; residue promoted to C1.**

## C4 — find/check, a 4th *kind* of axis — **LANDED 2026-06-29** (`EXTENDS_mode_relative`)

> The first three axes each assign ONE order per target. The find/check axis (σ-slate H3) assigns TWO:
> verify-order (witness-check budget) and predict-order (history budget). Probe
> ([C4_FIND_CHECK_ORDER.md](C4_FIND_CHECK_ORDER.md), test 12/12, reuses `suffstat_h3` + `order_meter`):
> parity λ has verify-order **finite** (17 = max Ω over n ≤ 2e5) but predict-order **∞** — the same
> target, two divergent orders ⟹ no single scalar budget; the law holds **per-mode** only. Finite-σ
> LFSR control: both orders finite (5, 6) → modes agree, meter live (non-vacuity guard). **The "order"
> is a mode-vector, not a scalar** — the find/check analog of the σ-schema's "not one comparable
> scalar." **It explains C2:** a pressure detector's "stable under the pressures I applied" is a
> *verify-mode* signal (finite budget); soundness is a *predict-mode* property (∞ for σ=∞ closures).
> C2's break IS that verify-vs-predict mode-confusion.

## M0 — Methodological meta-move (almost free)

**Statement.** *Which other lane can get an exact-oracle closed-form ground truth?* BoxSEL minted
`(9+√17)/32` and then measured every gap against it instead of arguing about the gap. Any lane that
can mint an analogous closed-form "gap value" (a parity-barrier gap, a continuous-resist pole value)
inherits the same leverage.

**Hook.** The exact-rational oracle + `Surd`-field pattern from BoxSEL transplants directly.

---

## Disposition

- **All three resolved (2026-06-29), break-first throughout:**
  - **C1 — LANDED** (`search_resist_confirmed`): search-reachability is a determine/resist filtration,
    the one axis the σ-order schema lacks. ([PHASE4L_SEARCH_ORDER_FILTRATION.md](PHASE4L_SEARCH_ORDER_FILTRATION.md))
  - **C2 — BROKEN-as-stated → repaired** (`C2_BROKEN_order_relative`): orthogonal-pressure abstention
    is sound only for false closures of order ≤ the pressure budget; the detector's reach IS the
    pressure family's order. ([C2_PRESSURE_ABSTENTION_BREAK.md](C2_PRESSURE_ABSTENTION_BREAK.md))
  - **C3 — CLOSED as redundant** with the σ-order schema (census above).
  - **M0** — opportunistic, untouched.
- **The through-line:** every result is *order-relative* — search reach (C1), pressure reach (C2),
  pressure repertoire (Phase 7g), and determination order (C3/σ-schema) are the same finite/∞ dichotomy
  on different axes. **C4 adds the 4th *kind*:** on the find/check axis the order is a *mode-vector*
  (verify-order ⊥ predict-order), not a scalar — and that mode-split *explains* C2 (verify-mode signal
  vs predict-mode soundness). Confirming-first would have laundered C2 past its own falsifier; breaking
  first is what produced the order-relative repair and the C4 diagnosis.
- **Flagship-bound follow-up — DONE (Phase 7g, `INHERITS_order_relative`):** the frozen Phase-7b v2
  pressure detector inherits C2's order-relative law — its reach **is its pressure repertoire**; a
  false closure whose witness is outside the handed repertoire (Helly under a PMP-only family) is
  accepted, and widening the repertoire catches it. Pre-registered (commit `38dc4328`) before the
  runner; test 15/15. ([PHASE7G_PRESSURE_REPERTOIRE_RESULTS.md](PHASE7G_PRESSURE_REPERTOIRE_RESULTS.md))
  So the order-relative law holds on three axes: search reach (C1) / pressure reach (C2) / pressure
  repertoire (Phase 7g).
- **Status:** portfolio research exploration. No lane is promoted; the active build is unchanged
  (`project_sundog_strategy_pivot_20260604`). Links: [[project_sundog_suffstat_order_slate]],
  [[project_sundog_shadow_invertibility_phase5]], [[project_sundog_parity_barrier_lane]],
  [[project_sundog_algo_approx_lane]].

---

*Sundog Research Lab — cross-lane conjecture slate seeded by the BoxSEL search-gap finding.
Internal; conjectures with hooks and falsifiers, nothing run.*
