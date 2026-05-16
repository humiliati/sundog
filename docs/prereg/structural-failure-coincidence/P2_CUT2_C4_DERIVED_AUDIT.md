# Structural Failure Coincidence — Cut 2 C4 Computed/Derived Route-Construction Audit

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Controller binding: [`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md)
Nuisance + bridge: [`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md)
Decoy term + temptation: [`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md)
Filed: **2026-05-16 (PT)**. Status: **C4 FILED FOR AUDIT — HOLD FOR
EXECUTION**. Pre-registers the derived-audit predicate set, its
inputs, and its self-test obligation only. Cut-2 execution remains
**HELD** on the C4 audit blockers below, the still-open
C2-A/B/C/D, C3-A/B/C/D, and C5, and a fresh P2-spec admission re-check.
No harness written; nothing run.

## Purpose

C4 closes the meta-condition: the harness must **compute**
`routeConstructionAudit` from the live run objects, not hardcode it.
The current patched harness returns `routeConstructionAudit() =>
{ routeTestVacuous: true, ... }` — a correct *fail-safe* now (it cannot
emit a pass), but an **assertion, not a derivation**. A future genuine
route would still be force-flagged vacuous, and flipping the literal to
`false` would leave the self-seal guard with nothing behind it. C4 makes
the guard real: a predicate set derived from the actual objects, so the
audit that exists to prevent a self-sealing pass cannot itself be
self-sealed.

## 1. The derived predicate set (computed every run)

The run is classified `MACHINERY_LIVE_ROUTE_TEST_VACUOUS`
(`traceabilityClaim:false`, rail `STALLED / UNTESTED`) **unless all
three predicates below hold**, each computed from the live objects with
a frozen threshold and written to `verdict.json` / `manifest.json` as
measured values (never source-asserted booleans):

- **D1 — route is separable from the analytic inverse.** On the
  pre-registered probe set, compare the bound controller's `q̂` to the
  closed-form `arccos(R22/f_par_obs)`. D1 holds iff they **diverge by
  more than a frozen floor on the region where they must differ by
  construction** — the C2 biased low-leverage band and under the
  decoy-edit — *not* a blanket "differ everywhere" (a traceable route
  legitimately agrees with the inverse on the clean eligible set; see
  the load-bearing note §3). Cut-1 had `q̂ ≡ analytic` by construction;
  D1 computed on the live objects returns *false* there.
- **D2 — decoys are reachable (consumes the refined C3-C receipt).**
  Compute the **argmax-sensitivity receipt** from C3-C: finite-
  difference `‖Δ(argmax_q I)/Δd‖` over the pre-registered decoy-edit.
  D2 holds iff it exceeds a frozen floor — the decoy provably moves the
  *converged answer*, not merely the pointwise gradient (which the
  Gaussian ridge zeroes at its own peak / under clipping, per C3-C).
  Cut-1 had `∂J/∂d ≡ 0`; D2 on the live `I` returns *false* there.
- **D3 — boundary behavior is emergent, not a read flag.** Structural
  check on the live adapter object: assert `h ∉ adapterInputs` (A1) and
  that the L2/L3 behavior change coincides with the observable-gated
  `I_route` leverage/curvature collapse, **not** a discrete generator
  bit the controller reads. Cut-1 echoed `f_cza`/`f_tan` generator
  bits; D3 returns *false* there.

Any predicate false ⇒ `MACHINERY_LIVE_ROUTE_TEST_VACUOUS` (or
`…_INCOMPLETE` if the A2 positive control is absent). A pass requires
**all of D1–D3 true *and* the four-quantity score to pass** — the rule
already pinned in the 2026-05-16 `P2_RUN_SPEC.md` companion amendment.

## 2. C4-B — the audit must be regression-tested against the Cut-1 fixture (load-bearing, surfaced adversarially)

The C4 analog of C2-B / C3-C. The derived audit is itself code that
could self-seal (e.g. a D1 floor so small that a near-tautological route
"passes"). The meta-guarantee: **the audit is validated against the very
failure it exists to catch, before it is trusted on any novel route.**
C4-B requires two pre-registered fixtures and a property test:

- **Known-vacuous fixture = the Cut-1 objects** (`routeEstimate` /
  `analyticInverseEstimate`, `∂J/∂d ≡ 0`, generator-bit boundary). The
  derived audit run on this fixture **must** return
  `routeTestVacuous: true` (D1 ∧ D2 ∧ D3 not all true). If it does not,
  C4 is not closed.
- **Synthetic non-vacuous fixture** = a constructed object that is
  separable from the analytic inverse, decoy-argmax-sensitive, and
  emergent-boundary by construction. The derived audit **must** return
  `routeTestVacuous: false` on it (otherwise the audit is a rigged-to-
  vacuous null — the C4 analog of C3-B's rigged-to-fail).

Both fixtures, the D1/D2/D3 floors, and the probe set are pre-registered
(A3 engineering tolerances, never post-results). The audit's own logic
must be inspectable and covered by this property test in the harness
test suite; a hardcoded `routeConstructionAudit()` returning a literal
is **forbidden** once C4 is in force.

## 3. Honest couplings (recorded, not papered over)

- **D1 ↔ C2-A/B.** "The region where route and inverse must differ by
  construction" is the C2 biased low-leverage band, which is not pinned
  until C2-A freezes the nuisance numerics and C2-B fixes `pen(q)`/`q_a`
  (else the `I_route` argmax — hence `q̂` — is ill-posed). D1 cannot be
  audited closed until C2-A/B close.
- **D2 ↔ C3-C/C3-A.** D2 *is* the C3-C argmax-sensitivity receipt; it
  inherits C3-A's frozen `κ, σ_D`, decoy-edit operators. D2's floor must
  equal the C3-C receipt floor (one number, not two).
- **D3 ↔ A1.** D3's `h ∉ adapterInputs` assertion is the A1 invariant
  made into a computed check; no new boundary.

## 4. Cut-2 C4 binding rules

1. `routeConstructionAudit` takes the live run objects (controller,
   adapter, `I`, probe set, decoy-edit) and **returns derived
   predicates**; a literal-returning stub is a **void** Cut-2 run.
2. The Cut-1 known-vacuous fixture test is part of the harness suite and
   must pass (audit flags Cut-1 vacuous) **before** any Cut-2 controller
   instantiation.
3. D1/D2/D3 floors, both fixtures, and the probe set are pre-registered
   and never edited post-results (A3). Immutable geometry/receipt
   boundaries unchanged.
4. A pass requires D1 ∧ D2 ∧ D3 **and** the four-quantity score; any
   predicate false ⇒ `MACHINERY_LIVE_ROUTE_TEST_VACUOUS`.

## Explicit non-bindings (cannot satisfy C4)

- A hardcoded / literal-returning `routeConstructionAudit`.
- A D1 that demands divergence from the analytic inverse **everywhere**
  (forbids the correct answer on the clean eligible set — a rigged-to-
  vacuous null).
- A D2 using pointwise `∂I/∂d` instead of the C3-C argmax-sensitivity
  receipt (falsified by the Gaussian ridge's zero-gradient point).
- Closing C4 without the Cut-1 known-vacuous fixture actually flagging
  vacuous in the test suite.
- Floors/fixtures fit or tuned after seeing any controller result.

## Open items

C4 files the **derived predicate set (D1–D3)**, the **C4-B fixture
self-test**, and the **honest couplings** for audit. Still open before
any Cut-2 run:

- **C4-A:** freeze the C4 numerics named but not instantiated — the D1
  divergence floor + the exact region it is measured on, the D2 floor
  (= the C3-C receipt floor), the D3 structural assertions, the probe
  set, and both C4-B fixtures.
- **C4-C:** make D1 construction-level, not controller-outcome-level.
  D1 should compare the route-only construction (for example `π_route`
  / `argmax I_route`) against the true hidden `h` on the must-differ
  region, not against `arccos(R22/f_par_obs)`. C2-B makes
  `argmax I_route = arccos(R22/f_par_obs)` by design; that equality is
  the fixed route, not a vacuity. D1's anti-Cut-1 target is whether the
  route construction is a clean readout of hidden `h`. If D1 reads the
  bound controller's final `q̂`, then a behavioral outcome can
  masquerade as a machinery-vacuity audit failure, conflating the audit
  with the four-quantity score.
- **C4-D:** make D3 mechanically auditable. `h ∉ adapterInputs` and
  "not a generator-bit echo" need a frozen inspection/taint method:
  e.g. an adapter input-manifest check plus a boundary perturbation test
  showing L2/L3 behavior follows observable intensity/curvature collapse
  rather than discrete `f_cza`/`f_tan` bits passed to the controller.
- **D1 is coupled to C2-A/B; D2 is coupled to C3-C/C3-A** (see §3) —
  C4 closes only once those close.
- Still-open siblings: **C2-A/B/C/D**, **C3-A/B/C/D**,
  **C5** (publication-plumbing freeze).

After C2-A/B/C/D, C3-A/B/C/D, C4 (incl. C4-A and C4-B), and C5 are all
filed, the P2-spec admission check **must be re-run** as one audit of
the whole discriminating cut; only on **ADMIT** may a Cut-2 harness be
built or run. Public-Language Constraint remains fully in force: no
`CONFIRMED` / traceability-success / theorem language anywhere
(including the rail).

## Honest prior (unchanged)

A derived audit that provably flags the Cut-1 self-seal (C4-B), plus the
biased route (C2) and the tempting reachable decoy (C3), keeps the
likely honest outcome at **D / BOUNDARY FOUND** — the Proxy-Collapse
confirmation avenue (`debunked.md`, P1 §C). **B** is earned *only* by a
measured refusal of the tempting decoy at the quantified in-sample cost
**and** emergent failure coincident with L1/L2/L3. Either is a clean
result; the in-between is not.

## Audit Notes

**2026-05-16 (PT) — Codex audit.** Direction accepted; execution
admission withheld. C4 fixes the right meta-failure: the
`routeConstructionAudit` must become a derived predicate set and must be
tested against both the known-vacuous Cut-1 fixture and a synthetic
non-vacuous fixture before it can guard a novel run. Two additional
blockers are now explicit. First, D1 must be computed from the route
construction itself (`π_route` / `argmax I_route`), not from the bound
controller's final behavior, or the audit can reclassify an outcome as
machinery vacuity. Second, D3 needs a mechanical inspection/taint or
perturbation method; a prose claim that boundary behavior is emergent is
not enough for an audit whose purpose is to avoid self-sealing. No
harness has been written and no controller has been instantiated.

**2026-05-16 (PT) — C2-B reconciliation.** C2-B resolves the
construction-level target for D1: because the admitted route construction
is intentionally `argmax I_route = arccos(R22/f_par_obs)`, D1 must not
compare the route against that same closed form. The repaired D1 target
is the P-A form: on the must-differ low-leverage/noisy band, the route
optimum must differ from true hidden `h` by the frozen D1/P-A floor. Cut
1 fails because `g^-1(g(h)) = h`; Cut 2 should pass D1 only if the
route construction is not a clean hidden-cause readout.
