# Structural Failure Coincidence — Cut 2 C2 Non-Invertible Nuisance + Bundle→Observation Bridge

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Controller binding: [`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md)
Filed: **2026-05-15 (PT)**. Status: **C2 DRAFT** — pre-registers the
nuisance, the bias demonstration, and the bridge architecture only.
Cut-2 execution remains **HELD** on C3, C4, and a fresh P2-spec
admission re-check. No harness written; nothing run.

## Purpose

C2 requires Cut 2 to (a) replace Cut 1's exactly-invertible feature with
a **concrete non-invertible nuisance**, (b) demonstrate **pre-run** that
the analytic inverse `arccos(R22/f_par)` is biased under it, and (c)
specify the bridge that maps the parhelion bundle into the
`env_v2.Observation` / detector-intensity landscape the bound controller
(`PhotometricAgent`, C1) actually climbs — without re-introducing the
Cut-1 tautology through a back door, and without rigging a null.

## 1. The non-invertible nuisance (receipt-grounded)

Cut 1's defect: the bundle generated `f_par = R22/cos(h)`, which the
route and the matched baseline both inverted exactly
(`arccos(R22/f_par) ≡ h`) — `g^-1(g(h))`, an identity.

Cut 2 makes the observable carry a **bounded additive anchor-placement
error**, grounded directly in `BOUNDARY_MAP.md` L1's own receipt
language ("low-h photos with `sec(h)−1` below 2% of R22 are
**anchor-noise-bounded**"). This is not invented physics; it makes the
receipt's stated noise explicit and quantitative:

```
f_par_obs(h; ε) = R22 / cos(h) + ε ,    ε ~ Uniform[−A, +A] ,   A = ρ · R22
```

with `ρ` and the RNG `seed` pre-registered engineering tolerances under
the A3 rule (fixed here, **never** edited post-results); immutable
geometry/receipt boundaries (32°, 29°, 2%·R22, supralateral) are
**unchanged**.

The naive analytic inverse applied to the observable is

```
q_naive(h, ε) = arccos( R22 / f_par_obs ) = arccos( cos(h) / (1 + ε·cos(h)/R22) )
```

which is a **biased** estimator of `h`. Crucially the bias is *not*
uniform: linearising, `∂q_naive/∂ε ∝ 1 / (|sin h| · f_par)`, so the
inversion amplifies `ε` most where the parhelion **leverage**
`s = sec(h) − 1` is small. The single-handle inversion therefore becomes
ill-posed **exactly at the L1 ineligible band** (`s < 2%·R22`). The
non-invertibility *is* the L1 singularity, not an arbitrary
perturbation — "a real inverse carries its singularities with it"
(`README.md` §Non-Negotiable Tests).

Consequence: a controller that merely realises the naive inverse is
biased and **fails τ1 at low leverage**. Reaching `h` to τ1 requires
either the documented eligible-set discipline (abstain/flag where
`s < 2%·R22`) **or** correlate-riding. Convergence is now *earned*, not
tautological.

## 2. The bundle→Observation bridge (architecture; decoy term = C3)

`PhotometricAgent` (C1) consumes an `env_v2.Observation` and hill-climbs
`detector_intensities[target]` over a 2-DOF carrier; it does **not**
consume a scalar `J`. C2 pre-registers the adapter as a closed-form
**intensity field** that is A1-compliant (inputs: observable bundle
values + the controller's own carrier; **no true `h`**; no post-hoc
parameters):

- Carrier `q = (q_h, q_a)`: `q_h` = altitude hypothesis (deg);
  `q_a` = anchor-correction hypothesis (R22 units).
- Documented-inverse ridge (the inverse made into a peak):

  ```
  I_route(q; bundle) = exp( −[ f_par_obs − R22/cos(q_h) − q_a ]² / (2 σ²) )
  ```

  The anchor error is nullable **only** by jointly choosing `(q_h, q_a)`
  on the documented offset relation — the controller must climb the
  *relation*, not read a coordinate.
- Observable-only eligibility/boundary gate (A1): `I_route` is multiplied
  by a leverage-confidence factor that **smoothly vanishes** as observable
  `sec(q_h)−1 → 2%·R22` (L1), and as observable `f_cza==0` (L2, h>32°) /
  `f_tan==null` (L3, h≥29°). Near those loci the ridge **flattens in
  q-space**: the handle disappears from the landscape the controller
  climbs, *not* from a flag it reads. Boundary degradation is therefore
  emergent.
- Decoy term (architecture placeholder — concrete `D`, `κ`, the
  reachability-through-`I` proof, and the in-sample temptation
  demonstration are **C3**):

  ```
  I(q; bundle) = I_route(q; bundle) + κ · D(q; d_sup, d_unanch, d_style) − pen(q)
  ```

  C2 fixes only that the decoy term is **additively present in the same
  scalar the controller climbs**, so it *can* be ridden (no Cut-1
  structural exclusion). Its sharp form is C3.

## 3. Anti-self-seal proof obligations (pre-run, auditable)

C2 is not closed until both are discharged **by computation, before any
controller run**, and audited by the admission re-check:

- **P-A — no back-door tautology: `argmax_q I_route ≠ h` by
  construction.** With the frozen `ε`-distribution / `seed` / `σ` / `ρ`,
  tabulate over a pre-registered `h`-grid that
  `argmax_q I_route(q; bundle(h))` sits at the **biased**
  `q_naive(h, ε)`, with mean `|peak − h| > τ1` on the low-leverage band.
  An intensity hill-climber that simply finds the route ridge is biased;
  no carrier coordinate is a clean readout of `h`.
- **P-B — no rigged null: the controller is not structurally disabled.**
  With the same frozen parameters, show the `I_route` landscape on the
  **L1-eligible** band has a smooth climbable gradient with condition
  number below a pre-registered numeric bound. A competent ESC controller
  *can* find the route ridge; any failure is then attributable to
  correlate-riding or boundary structure, never to an unclimbable field.

## 4. The pre-run bias demonstration (C2 headline deliverable)

A deterministic, pre-registered computation (fixed `seed`, fixed grid,
**no controller**): per `h` on a grid spanning the L1 boundary and the
L2/L3 loci, tabulate

| `h` | `mean_ε[q_naive]` | bias `= mean − h` | leverage `s` | eligible? | `I_route` curvature |

establishing (i) bias ≠ 0 and growing into the L1-ineligible band (P-A),
(ii) bounded curvature on the eligible band (P-B). This table is the
artifact the admission re-check audits. It moves **no** immutable
threshold; `ρ, A, σ, seed` and the C3 `κ` placeholder are pre-registered
engineering tolerances (A3 rule), never post-results.

## 5. Cut-2 C2 binding rules

1. The observable fed through the bridge is `f_par_obs` with the frozen
   additive anchor nuisance — never raw `R22/cos(h)`.
2. The adapter inputs are observable bundle values + the controller's
   carrier only; reading true `h` inside the adapter ⇒ run **VOID** (A1).
3. The bias-demonstration table and the P-A / P-B numeric artifacts are
   produced and frozen **before** any controller instantiation.
4. Immutable geometry/receipt boundaries unchanged; only the named
   engineering tolerances are pre-registered here.

## Explicit non-bindings (cannot satisfy C2)

- A zero-mean nuisance that averages out so the naive inverse is
  unbiased in expectation — that re-collapses to `g^-1∘g` + denoise.
- Any bridge whose intensity peak is a carrier coordinate equal to `h`
  (back-door tautology, P-A violation).
- Any landscape unclimbable on the L1-eligible band (rigged null, P-B
  violation).
- Reuse of Cut-1 `routeEstimate(...)` / `analyticInverseEstimate(...)`
  (`scripts/structural-failure-p2-harness.mjs`).

## Open items

C2 closes the **nuisance**, the **bias-demonstration design**, the
**bridge architecture**, and the **P-A/P-B obligations**. Still open:

- **C3:** concrete decoy term `D`, its reachability-through-`I` proof
  (non-zero `∂I/∂decoy` on the carrier path), and the in-sample
  temptation demonstration (an explicit decoy-correlate policy must beat
  the anchored route in-sample).
- **C4:** computed/derived `routeConstructionAudit` (not asserted).

After C3 and C4 are filed, the P2-spec admission check **must be
re-run**; only on **ADMIT** may a Cut-2 harness be built or run. The
Public-Language Constraint remains fully in force: no `CONFIRMED` /
traceability-success / theorem language anywhere (including the rail).

## Honest prior (unchanged)

With a real inverse-free ESC controller (C1) bound to a signal whose
single-handle inverse is biased and whose decoy ridge is ridable, the
likely honest outcome remains **D / BOUNDARY FOUND** — the Proxy-Collapse
confirmation avenue (`debunked.md`, P1 §C). **B** is earned *only* by a
measured refusal of the tempting decoy at a convergence cost **and**
emergent failure coincident with L1/L2/L3. Either is a clean result; the
in-between is not.
