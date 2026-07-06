# AT-5 — Symbolic Trajectory-Window Scope (the formal shadow, right-sized)

> 2026-07-05. Companion to `NSE_ATTRACTOR_TAIL_SYNTHESIS.md`, per
> `NSE_AT_SYNTHESIS_AT5_SCOPE.md`. **Not a run, not a new theorem hunt.**
> Internal; the slate do-not-say list and the synthesis do-not-reopen list bind.

## 1. The honest size of AT-5

AT-5's formal core **already exists and is gated**:
`Sundog.SurfaceBag.Graded.stackTop_resists_every_window` in
`sundogcert/Sundogcert/SurfaceBagGraded.lean` — for every window order w there exist
valid strings with identical ≤w-gram count vectors and different stack-tops
(σ_surface = ∞ on the window filtration; AxiomAudit-gated,
`[propext, Classical.choice, Quot.sound]`).

The trajectory-window reading is an **interpretation, not a theorem**: a symbolic
trajectory (a finite string over regime symbols with transition constraints) is the
same formal object as a token string; window-count statistics of a trajectory are the
same formal object as w-gram counts. Under that reading, the landed theorem already
says: *there is a trajectory functional (the itinerary/stack-top analogue) that no
finite-window count statistic determines, at any window order.* What is **not** said,
and is never said by any tier below: that Kolmogorov-flow regime dynamics realize such
a functional — AT-4/AT-4b measured that question empirically and it closed negative on
the C1 substrate.

## 2. The two tiers (owner picks one)

| tier | work | verdict token | when it's right |
| --- | --- | --- | --- |
| **Docs-only interpretation** (recommended default) | This document §1 *is* the deliverable: the citation + the interpretation + the claim boundary. Nothing new in `sundogcert`. | `AT5_FORMAL_CORE_ALREADY_LANDED` | Enough for the synthesis; zero overclaim surface; zero build risk. |
| **Thin Lean wrapper** | `sundogcert/Sundogcert/TrajectoryWindow.lean`: a namespace with *aliases/definitions only* (e.g., `TrajSymbol := ...`, `sigmaTrajInfinite := stackTop_resists_every_window` restated under trajectory vocabulary), reusing `SurfaceBagGraded` — **no re-proving**. Gate the alias in `AxiomAudit` (must be the same axiom profile). Docstring names the empirical non-realization (AT-4/4b receipts) as the fence. | `AT5_LEAN_SHIM_LANDED` | Only if the vocabulary `trajectory-window` / `σ_traj` should exist inside `sundogcert` itself. |

## 3. Kill conditions (inherited verbatim from the commissioning scope)

- Wrapper needs nontrivial new combinatorics → stop, take docs-only. No days spent
  rebuilding `SurfaceBagGraded` under new names.
- Vocabulary creates public overclaim risk ("attractor stack-top" reading as a PDE
  theorem) → docs-only.
- AxiomAudit would need a broader axiom profile than the existing theorem → do not land.

## 4. Claim boundary (binds both tiers)

Proves only a symbolic statement about finite strings and window-count statistics.
Nothing about Kolmogorov flow, NSE, attractors, nudging, or empirical ledgers. Does not
reopen AT-4/AT-4b — it supplies the formal shape a surface-blocked trajectory label
*would have had*; the C1 receipts say the substrate did not supply one. σ_traj is not
registered as a schema filtration by either tier (registration on receipts; the
receipts here are the citation of an already-landed theorem, which the σ-slate already
counts via the surface-window axis).

## 5. Status

Docs-only tier: **satisfied by this document** — `AT5_FORMAL_CORE_ALREADY_LANDED` can
be recorded in the slate on owner acknowledgment. Lean-wrapper tier: awaiting owner
selection; if selected, the implementation is aliases + one AxiomAudit block + a full
`lake build`, estimated under an hour, with the §3 kill conditions binding.

Cross-refs: `NSE_AT_SYNTHESIS_AT5_SCOPE.md` (commission), `NSE_ATTRACTOR_TAIL_SYNTHESIS.md`
(§7 menu), `sundogcert/Sundogcert/SurfaceBagGraded.lean` + `SurfaceBag.lean` +
`AveragingDecodability.lean` (the formal family), `AT4_CROSSOVER_TRANSPLANT_RECEIPT.md` /
`AT4B0_ADMISSION_RECEIPT.md` (the empirical non-realization).
