# NSE / AT Internal Synthesis and AT-5 Scope

> 2026-07-05. Internal next-work scope after AT-4b closed at
> `AT4B_UNPOWERED_INPUT`. This is not an external-review packet, not a public
> surface, and not a promotion request.

## Recommendation

Do the synthesis first. Then, if useful, land AT-5 as a thin formal companion.

Reason: the empirical lane has now moved from a slate to a closed receipt set
for all runnable entries. The remaining risk is not lack of another run; it is
ledger drift. A synthesis pins what the lane learned before AT-5 names the
symbolic formal shadow.

The other key update: AT-5 is smaller than originally scoped. Its formal core
already exists in `sundogcert`:

- `Sundogcert/SurfaceBagGraded.lean`
- theorem `Sundog.SurfaceBag.Graded.stackTop_resists_every_window`
- AxiomAudit-gated as `[propext, Classical.choice, Quot.sound]`

So AT-5 should not be treated as a new theorem unless the owner wants a
vocabulary-specific wrapper. The honest AT-5 task is an interpretation shim:
trajectory-window statistics are the same formal shape as token-window
statistics; the PDE realization remains empirical and is not proved in Lean.

## Synthesis Artifact

Target file:

`docs/chatv2/NSE_ATTRACTOR_TAIL_SYNTHESIS.md`

Purpose: close the internal lane state in one readable object, with no new
claims and no new gates.

Required sections:

1. **One-line state.** The maintained-ledger form separated on the finite C1
   proxy; the crossover form closed terminally on this substrate; AT-5 remains
   a symbolic/formal shadow, not an empirical rescue.
2. **Receipt table.** AT-1 through AT-6, plus AT-7 parked, with verdict,
   decisive measurement, and what it forbids future readers from saying.
3. **Two forms, two answers.**
   - Maintained-ledger form: `AT3_LEDGER_SPLIT_CONFIRMED` at G=200,
     `K_dec = 1 < K_sync = 2`, relay-typed but real in the dynamical gauge.
   - Crossover form: `AT4_SURFACE_SUFFICIENT` at G=200 and final
     `AT4B_UNPOWERED_INPUT` at G=300; no surface-blocked natural label was
     established.
4. **Why AT-4b closed.** v1.1 powered the label but failed slice formation;
   v1.2 used the measured 2M scale and leakage-safe blocked split, then failed
   at horizon formation. No surface/crossover numbers were read after either
   input failure.
5. **What survived from the original slate.** Determining-modes absolute pole
   stays closed; growth/crossover natural labels were harder than expected;
   the maintained ledger is the genuine positive object.
6. **Do-not-reopen list.** No more AT-4 variants without a new substrate or new
   objective family; no claim that AT-3 is emergent computation rather than
   relay + temporal pairing; no public or infinite-dimensional NSE claim.
7. **Next-work menu.** AT-5 shim, no-run closure; AT-7 parked; optional website
   or public work remains out of scope unless separately requested.

Done condition:

`NSE_ATTRACTOR_TAIL_SYNTHESIS.md` links every load-bearing receipt and includes
the phrase:

> internal synthesis only; no public or external-review claim is licensed here

## AT-5 Companion

Target file:

`docs/chatv2/AT5_SYMBOLIC_TRAJECTORY_WINDOW_SCOPE.md`

Optional Lean target:

`sundogcert/Sundogcert/TrajectoryWindow.lean`

Recommended first move: write the scope doc before any Lean wrapper.

AT-5 should be scoped as one of two tiers:

| tier | work | verdict |
| --- | --- | --- |
| docs-only interpretation | Cite `SurfaceBagGraded.stackTop_resists_every_window` as the existing formal core and state the trajectory-window reading as an interpretation. | `AT5_FORMAL_CORE_ALREADY_LANDED` |
| thin Lean wrapper | Add a namespace/module with aliases or definitions named for trajectory windows, reusing `SurfaceBagGraded` rather than reproving it. Gate aliases in `AxiomAudit`. | `AT5_LEAN_SHIM_LANDED` |

The docs-only tier is enough for the synthesis. The Lean wrapper is only worth
doing if the owner wants the vocabulary `trajectory-window` / `sigma_traj` to
exist inside `sundogcert`.

AT-5 claim boundary:

- Proves only a symbolic statement about finite strings/lists and window-count
  statistics.
- Does not prove anything about Kolmogorov flow, NSE, attractors, nudging, or
  empirical ledgers.
- Does not reopen AT-4/AT-4b. It supplies the formal shape that a
  surface-blocked trajectory label would have had, not evidence that C1 has one.

AT-5 kill conditions:

- If the wrapper needs nontrivial new combinatorics, stop and use the docs-only
  tier; do not spend days rebuilding `SurfaceBagGraded` under new names.
- If the vocabulary causes public-facing overclaim risk ("attractor stack-top"
  reads like a PDE theorem), keep the shim in docs only.
- If AxiomAudit would require a broader axiom profile than the existing
  `SurfaceBagGraded` theorem, do not land the wrapper.

## Recommended Order

1. Write `NSE_ATTRACTOR_TAIL_SYNTHESIS.md`.
2. Decide whether AT-5 needs only a docs note or a Lean shim.
3. If Lean shim is selected, implement it as aliases over `SurfaceBagGraded`
   and run `lake build Sundogcert.AxiomAudit` in `Dev/sundogcert`.

No more empirical AT runs are recommended from this scope.
