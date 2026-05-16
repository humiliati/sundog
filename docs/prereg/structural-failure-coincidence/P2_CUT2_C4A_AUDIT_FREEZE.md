# Structural Failure Coincidence — Cut 2 C4-A Audit Freeze + Fixture Self-Test (folds C4-B/C/D)

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Parent: [`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md) (D1–D3, C4-A/B/C/D)
Inherits: [`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md) (scale/seed/grid/readout) · [`P2_CUT2_C3A_NUMERIC_FREEZE.md`](P2_CUT2_C3A_NUMERIC_FREEZE.md) (the shared D2 floor)
Cut-1 fixture source: `scripts/structural-failure-p2-harness.mjs` (the real Cut-1 objects)
Filed: **2026-05-16 (PT)**. Status: **C4-A FILED FOR AUDIT — HOLD FOR
EXECUTION**. Like C2-A / C3-A, this freezes **structure, provenance, and
the operational-artifact obligations**; concrete `[E]` values, the
fixtures, the probe set, and the C4-B self-test table are the
maintainer's pre-run fill (no fabrication); a failing receipt **blocks
(append-only redesign), never tunes**. Inherits not-yet-closed C2-A/C3-A
scaffolds, so C4 receipts are provisional until those close. Cut-2
execution remains HELD on C4 + C5 + a fresh P2-spec admission re-check.
No harness written; nothing run.

## Purpose

The last C-condition column. C4-A freezes the derived-audit numerics and
**operationalizes** the fixtures/probe-set/taint-method (the C3-A
`P_in` hold's discipline, propagated: a *principle* is not an artifact —
each must be a concrete, reproducible, seed-deterministic frozen
object). It folds C4-B (the Cut-1 fixture self-test), C4-C (D1
construction-level vs true `h`), and C4-D (the mechanical taint method).

## 1. Inheritance + dependency ordering

C4-A **inherits, not redefines**: C2-A's bridge scale, `seed`, grid,
`q` domains, and the sustained-TRACK confident-`q̂` readout; the **D2
floor = the C3-A-R argmax-sensitivity floor (one shared number)**,
cross-checked in both files. Honest ordering note (propagating the C3-A
audit): C2-A and C3-A are still scaffolds; **any C4 receipt computed
before C2-A/C3-A closure is provisional and cannot admit Cut-2.**

## 2. Operational-frozen-artifact discipline (propagated from the C3-A `P_in` hold)

Every C4 artifact must be a **concrete reproducible pre-registered
object (deterministic given `seed`), not a description**:

- the **D1 probe set** + the must-differ region (the C2 biased
  low-leverage band + the decoy-edit) as an explicit frozen sample;
- the **C4-B Cut-1 known-vacuous fixture** = the *actual* Cut-1 objects
  in `scripts/structural-failure-p2-harness.mjs` (`routeEstimate` /
  `analyticInverseEstimate`, `∂J/∂d≡0`, generator-bit boundary) — an
  immutable real artifact, not a re-description;
- the **synthetic non-vacuous fixture** (see §4);
- the **C4-D taint manifest + perturbation script** (see §5).

A prose statement of any of these is not admission-sufficient — the same
ruling the C3-A `P_in` hold established.

## 3. D1–D3 and the pass rule (recap; C4-C repaired)

`MACHINERY_LIVE_ROUTE_TEST_VACUOUS` unless **all** hold, each computed
from live objects with a frozen floor:

- **D1 (C4-C repaired):** the *route construction* (`argmax I_route` /
  `π_route`, well-posed via C2-B) differs from **true hidden `h`** by
  ≥ the frozen P-A/D1 floor on the must-differ band — *not* a comparison
  to `arccos(R22/f_par_obs)` (C2-B makes those identical by design; that
  equality is the fixed route, not a vacuity). Cut-1 had route ≡ `h`.
- **D2:** `‖Δ(argmax_q I)/Δd‖` over the frozen decoy-edit exceeds the
  **shared C3-A-R floor** (argmax-sensitivity, the C3-C fix).
- **D3:** `h ∉ adapterInputs`, `(w,b)` are frozen constants at runtime
  (not refit), no `h` into `ĥ_dec`/`D`, and the L2/L3 behavior change
  tracks the observable identifiability collapse — verified by §5.

Pass requires D1∧D2∧D3 **and** the four-quantity score.

## 4. C4-A keystone load-bearing self-seal (surfaced adversarially)

The C4 analog of C2-A §5 / C3-A §2. The Cut-1 known-vacuous fixture is
*fixed* (the real harness file — no design freedom). The hazard is the
**synthetic non-vacuous fixture**: if it is hand-built *after* the audit
logic, "the audit passes a non-vacuous case" is rigged. Therefore the
non-vacuous fixture is **the minimal mechanical perturbation of the
Cut-1 fixture that flips exactly the three D1/D2/D3-relevant
properties** (route made biased-but-not-`h` on the must-differ band;
decoy made argmax-sensitive; boundary made observable-emergent), frozen
**before** the audit logic is finalized — no hand-tuning to the audit.
C4-B is then the *same* audit run on both: **must** return vacuous on
Cut-1 and non-vacuous on the minimal-flip — a genuine two-sided test
with no fixture-design freedom. A failing self-test **blocks
(append-only redesign), never tunes the fixture or the floors** (A3).

## 5. C4-D concrete mechanical taint method (no prose-only claim)

Two frozen, reproducible checks:

1. **Input-manifest assertion.** The controller-facing objective's
   realized input set is exactly the A1 set + the gated observables;
   **`h` absent**; no `f_cza`/`f_tan`/`f_par_obs<R22` *branch flags*
   reaching the controller; `(w,b)` are frozen constants at runtime, not
   refit; no `h` flows into `ĥ_dec`/`D`. (`(w,b)` may use true `h`
   **only** as the C3-A pre-run reference fit, never at runtime.)
2. **Boundary-perturbation test.** Perturb the observable the boundary
   rides; confirm the controller's behavior change tracks the landscape
   identifiability/curvature collapse **via the inherited sustained-TRACK
   readout**, not a discrete step that would betray a bit-read. C2-C /
   C2-D define the boundary source; C4-D is its audit (source-vs-audit,
   not circular).

## 6. The C4 numeric/artifact freeze (provenance-tagged, A3)

`[G]` immutable/inherited; `[E]` pre-registered engineering tolerance
(amend-only, justified, never post-results).

| symbol / artifact | role | provenance |
| --- | --- | --- |
| scale, `seed`, grid, `q` domains, sustained-TRACK readout | inherited C2-A | **[G]** (inherited) |
| D2 floor | argmax-sensitivity reachability | **[E]**, **= the C3-A-R floor (one number)** |
| D1 / P-A floor | route-vs-true-`h` separation on the must-differ band | **[E]**; region **[G]** (L1 band, C2) |
| D1 probe set | frozen evaluation sample | **[E]** operational artifact |
| Cut-1 known-vacuous fixture | the real `structural-failure-p2-harness.mjs` Cut-1 objects | **[G]** (immutable real artifact) |
| synthetic non-vacuous fixture | minimal mechanical flip of the Cut-1 fixture | **[E]**, frozen before the audit logic (§4) |
| C4-D input manifest + perturbation script | the taint method | **[E]** operational artifact |

Actual `[E]` values/artifacts are the maintainer's pre-run fill; the
admission re-check audits them, it does not let them move after results.

## 7. Honest couplings

- **D2 floor = C3-A-R floor.** One number; if they ever differ, both are
  void.
- **D1 region ↔ C2-A/B.** The must-differ band is the C2 biased
  low-leverage band; D1 closes only once C2-A/B close.
- **Readout ↔ C2-A-1.** The sustained-TRACK criterion is inherited, not
  re-defined.
- **C4-D ↔ C2-C/C2-D/C3-A.** C4-D audits the boundary sources (C2-C/D)
  and the frozen-`(w,b)`/no-`h` runtime rule (C3-A); it does not define
  them.

## Cut-2 C4-A binding rules

1. Every §6 number/artifact frozen here; `[G]`/inherited immutable,
   `[E]` amend-only/justified/never-post-results (A3).
2. The non-vacuous fixture is the §4 minimal mechanical flip, frozen
   before the audit logic; a hand-tuned fixture ⇒ **void**.
3. The Cut-1 self-test must actually run in the harness suite and return
   vacuous-on-Cut-1 / non-vacuous-on-flip **before** any Cut-2
   instantiation; otherwise the run is **void**.
4. D2 floor must equal the C3-A-R floor; a mismatch ⇒ **void**.
5. Reading true `h` at runtime anywhere (adapter / `D` / audit) ⇒
   **VOID** (A1).

## Explicit non-bindings (cannot satisfy C4-A)

- A hand-built synthetic non-vacuous fixture tuned to the audit logic.
- Prose-only fixtures / probe set / taint method (not operational
  artifacts).
- A D2 floor differing from the C3-A-R floor.
- A peak-intensity lock readout instead of the sustained-TRACK criterion.
- Closing C4-A while C2-A/C3-A are still scaffolds (provisional only).
- Any `[E]` value/fixture tuned after a controller result.

## Open items

C4-A files the audit freeze structure + the operational-artifact and
self-test obligations. Concrete `[E]` values, the fixtures, the probe
set, the taint script, and the C4-B two-sided self-test table are the
maintainer's pre-run fill (no fabrication); a negative receipt blocks
Cut-2. With C4-A filed, **all C-condition columns are filed**; the only
remaining sibling is **C5** (publication-plumbing freeze).

After C2-A, C3-A, C4-A (incl. the self-test), and C5 are all filed *and
their concrete values/artifacts/receipts landed*, the P2-spec admission
check **must be re-run** as one audit of the whole discriminating cut;
only on **ADMIT** may a Cut-2 harness be built or run. Public-Language
Constraint remains fully in force: no `CONFIRMED` /
traceability-success / theorem language anywhere (including the rail).

## Honest prior (unchanged)

A derived audit provably catching the real Cut-1 self-seal (C4-B),
against a biased route (C2) with a tempting reachable decoy (C3), keeps
the likely honest outcome at **D / BOUNDARY FOUND** — the Proxy-Collapse
confirmation avenue (`debunked.md`, P1 §C). **B** is earned *only* by a
measured refusal of the tempting decoy at the quantified in-sample cost
**and** emergent failure coincident with L1/L2/L3. Either is a clean
result; the in-between is not.

## Audit Notes

*(reviewer space — append-only below)*
