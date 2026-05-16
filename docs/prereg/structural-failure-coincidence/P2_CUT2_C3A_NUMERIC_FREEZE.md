# Structural Failure Coincidence — Cut 2 C3-A Numeric Freeze + Decoy Receipts (folds C3-B/C/D)

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Parent: [`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md) (C3-A/B/C/D)
Inherits: [`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md) (scale/seed/grid/readout) · [`P2_CUT2_C2B_PEN_AND_QA.md`](P2_CUT2_C2B_PEN_AND_QA.md) (`π_route` well-posed)
Controller: [`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md)
Audited by: [`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md) (D2 = the shared argmax-sensitivity floor)
Filed: **2026-05-16 (PT)**. Status: **C3-A FILED FOR AUDIT — HOLD FOR
EXECUTION**. Like C2-A, this freezes **structure, provenance, and receipt
obligations**; concrete `[E]` values and the receipt tables are the
maintainer's pre-run fill (no fabrication), and a failing receipt
**blocks (append-only redesign), never tunes**. Cut-2 execution remains
HELD on the C3 receipts, C4-A/B/C/D, C5, and a fresh P2-spec admission
re-check. No harness written; nothing run.

## Purpose

The C3 numeric freeze in one pass: fix `P_in`, `(w,b)`, `κ`, `σ_D`,
`M`, the edit operators, and the C3-C argmax-sensitivity floor; fold in
the C3-B calibration window and the C3-D `π_route` coupling (now
dischargeable because C2-B made `π_route` well-posed); propagate the
C2-A controller-semantics correction across the whole C3 column.

## 1. Inheritance + propagated controller readout

C3-A **inherits, not redefines**, from C2-A: the bridge scale
(eligible-band route peak ≡ 1.0), the RNG `seed`, the `h`-grid, and the
`q` domains. Every place a C3 receipt reads the **bound controller's**
behavior it uses the **C2-A-1 sustained-TRACK confident-`q̂` criterion**:
a "confident `q̂`" means a sustained TRACK lock under
`PhotometricAgent`'s real SCAN→SEEK→TRACK machine (threshold enforced
only in TRACK after `reacquire_hold_steps`); "abstain / refuse" means
reacquire / never-sustains-TRACK. C3 must not infer lock from peak
intensity alone — the C2-C/D hold #2 correction propagates here intact.

## 2. C3-A keystone load-bearing self-seal (surfaced adversarially)

The C3 analog of C2-A §5. The keystone hazard is the **`P_in` decoy↔`h`
in-sample correlation strength** (with `κ`, `M`). If `P_in`'s
correlation, `κ`, or `M` are chosen *after* seeing controller / `π`
behavior so the temptation "comes out tempting-but-reversible," the
C3-T / C3-B receipts are rigged. Therefore the **`P_in` decoy↔`h`
correlation is frozen by an independent principle before any run** —
pre-registered from each decoy's `BOUNDARY_MAP` characterization
("co-varies with `h` in any finite sample"), **not** fit to outcomes.
With `P_in` independently frozen, C3-T's margin and C3-B's window are
genuine pass/fail with no correlation-strength freedom. A failing
receipt **blocks (append-only redesign), never a tuned
`P_in`/`κ`/`M`** (A3).

## 3. The C3 numeric freeze (provenance-tagged, A3)

`[G]` = immutable / inherited boundary (change = forbidden goalpost
move); `[E]` = pre-registered engineering tolerance (amend-only,
justified, never post-results).

| symbol | role | provenance |
| --- | --- | --- |
| bridge scale, `seed`, `h`-grid, `q` domains | inherited from C2-A | **[G]** (inherited) |
| sustained-TRACK confident-`q̂` readout | controller lock/abstain definition | **[G]** (inherited C2-A-1) |
| `P_in` h-sampling dist | in-sample altitude distribution | **[E]** |
| `P_in` per-decoy decoy↔`h` correlation coeffs | the temptation source | **[E]**, frozen by the §2 independent principle |
| `(w,b)` least-squares decoy→`h` fit | the A2 positive-control policy | procedure **[G]**; fit on frozen `P_in`+`seed`, never refit post-results |
| `κ` decoy-ridge weight | temptation strength | **[E]** (C3-B window) |
| `σ_D` decoy-ridge width | ridge sharpness | **[E]** |
| `M` temptation margin | π_dec must beat π_route in-sample by ≥ M | **[E]** |
| `τ_pc` positive-control move | reused from the frozen A2 spec (`2.0°`) | **[G]** (prior freeze) |
| decoy-edit operator + magnitude | the q2 decoy-edit | **[G]** (spec q2) + magnitude **[E]** |
| counterfactual handle-edit | the q2 handle-edit | **[G]** (spec q2) |
| C3-C argmax-sensitivity floor | reachability `‖Δ(argmax_q I)/Δd‖` | **[E]**, **= the C4 D2 floor (one number, not two)** |
| C3-B(i)/(ii) demo params | calibration window | **[E]** |

Actual `[E]` values are fixed in this freeze and A3-immutable from this
point; the admission re-check audits them, it does not let them move
after results.

## 4. The three C3 receipts (pre-run, computed, frozen before admission)

Mirrors C2-A's receipt discipline; the bound-controller readout is the
inherited sustained-TRACK criterion (§1).

- **C3-A-R — reachability (C3-C, refined).** Tabulate
  `‖Δ(argmax_q I)/Δd‖` over the frozen decoy-edit; it must exceed the
  shared C3-C/D2 floor — decoys move the *converged answer*, not merely
  the pointwise gradient (which the Gaussian ridge zeroes at its peak /
  under clipping). Removes Cut-1's `∂J/∂d ≡ 0` vacuity by computation.
- **C3-A-T — temptation + reversal (C3-T).** Under the §2-frozen
  `P_in`: `mean|π_dec − h|_in ≤ mean|π_route − h|_in − M` (π_route is
  C2-B-well-posed), **and** the advantage reverses — decoy-edit:
  `π_dec` error ≥ `τ_pc`, `π_route` `|Δq̂| ≤ 0.5°`; counterfactual
  handle-edit: `π_route → h′` within `τ2`, `π_dec` stale. (`π_route`,
  `π_dec` are argmax-style reference policies; predictions about the
  *bound controller* refusing/riding use the §1 readout.)
- **C3-A-B — calibration window (C3-B).** The frozen `κ` satisfies both
  (i) temptation real (C3-A-T.1) **and** (ii) on the eligible band the
  `I_route` optimum stays a feasible competing optimum (C2-B made this
  well-posed; the C3-B(ii)↔C2-B coupling is now dischargeable).

## 5. Honest couplings

- **C3-A ↔ C2-A.** Scale, seed, grid, and the sustained-TRACK readout
  are inherited; C3-A may not contradict them.
- **C3-A-R floor = C4 D2 floor.** One shared number; C4-A must use the
  same value (recorded in both).
- **C3-A-T/B ↔ C2-B.** `π_route` is well-posed only because C2-B fixed
  `pen(q)`/`q_a`; the C3-D coupling is design-discharged, numerically
  closed at the joint re-run.
- **q2 edit operators are shared** with the frozen spec quantities; C3
  reuses, never redefines, them.

## Cut-2 C3-A binding rules

1. Every §3 number is frozen here; `[G]`/inherited immutable, `[E]`
   amend-only/justified/never-post-results (A3).
2. `P_in`'s decoy↔`h` correlation, `κ`, `M` are frozen by the §2
   independent principle **before** any run; post-hoc tuning ⇒ **void**.
3. Bound-controller readouts use the inherited sustained-TRACK criterion;
   inferring lock from peak intensity alone ⇒ **void**.
4. `(w,b)` is fit once on frozen `P_in`+`seed`; refit after any result
   ⇒ **void**. Reading true `h` in `D`/`ĥ_dec` ⇒ **VOID** (A1).

## Explicit non-bindings (cannot satisfy C3-A)

- Tuning `P_in`/`κ`/`M`/`σ_D` after a controller or `π` result.
- A coarse peak-intensity lock readout instead of the sustained-TRACK
  criterion.
- Pointwise `∂I/∂d` instead of the argmax-sensitivity receipt.
- A C3-A-R/T/B "pass" computed against a strawman instead of the bound
  `PhotometricAgent` / the frozen reference policies.
- A C3-A-R floor that differs from the C4 D2 floor.

## Open items

C3-A files the freeze structure + the three receipt obligations. As with
C2-A, the concrete `[E]` values and the C3-A-R/T/B receipt tables are
the maintainer's pre-run fill (no fabrication); a negative receipt
blocks Cut-2. Still-open siblings: **C4-A/B/C/D**, **C5**.

After C2-A, C3-A (incl. receipts), C4-A/B/C/D, and C5 are all filed, the
P2-spec admission check **must be re-run** as one audit of the whole
discriminating cut; only on **ADMIT** may a Cut-2 harness be built or
run. Public-Language Constraint remains fully in force: no `CONFIRMED` /
traceability-success / theorem language anywhere (including the rail).

## Honest prior (unchanged)

A correctly-calibrated tempting decoy against a real inverse-free ESC
controller keeps the likely honest outcome at **D / BOUNDARY FOUND** —
the Proxy-Collapse confirmation avenue (`debunked.md`, P1 §C). **B** is
earned *only* by a measured refusal of the tempting decoy at the
quantified in-sample cost **and** emergent failure coincident with
L1/L2/L3. Either is a clean result; the in-between is not.

## Audit Notes

*(reviewer space — append-only below)*

**2026-05-16 (PT) — Codex audit.** Direction accepted; C3-A is **not
yet execution-closing**. The file correctly propagates both C2-A holds:
C3 inherits the sustained-TRACK confident-`qhat` readout, and it treats
numeric values/receipts as pre-run fill rather than fabricated results.
It also correctly upgrades C3-R to argmax-sensitivity and keeps the C3
D2 floor shared with C4. Three admission holds remain. (1) C3-A inherits
C2-A's scale/seed/grid/readout, but C2-A is itself still a scaffold
until its concrete [E] values and receipts land; any C3 receipt computed
before C2-A closure is provisional and cannot admit Cut-2. (2) The text
must be read as freezing **slots/provenance/obligations**, not actual
numeric values, until `P_in`, decoy-correlation coefficients, `kappa`,
`sigma_D`, `M`, edit magnitudes, and the shared C3-C/C4-D2 floor are
filled with concrete values and tables. The sentence "Actual [E] values
are fixed in this freeze" is therefore a future-facing obligation, not a
closure ruling. (3) The independent `P_in` principle needs an
operational receipt: exact finite sample or generator, seed, decoy
coefficient table, and frozen `(w,b)` fit, before C3-A-T/B can be
audited. Using true `h` to fit `(w,b)` is allowed only as this pre-run
reference artifact; the runtime adapter/decoy ridge must read frozen
coefficients and observable `d` only, with C4-D taint checking no `h`
input. No controller/harness run.
