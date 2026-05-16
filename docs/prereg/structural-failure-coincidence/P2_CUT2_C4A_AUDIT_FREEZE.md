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

**2026-05-16 (PT) — Codex audit.** Direction accepted; C4-A is **not
yet execution-closing**. The right anti-self-seal is identified: the
non-vacuous fixture must be a minimal mechanical flip of the real Cut-1
fixture, and the same derived audit must return vacuous on Cut-1 and
non-vacuous on the flip. Closure still requires concrete operational
artifacts, not descriptions: a frozen D1 probe-set file/table; a
hashable Cut-1 fixture extraction or fixture manifest from
`scripts/structural-failure-p2-harness.mjs`; a synthetic fixture
generator/diff that mechanically flips exactly D1/D2/D3 and records
which fields changed; a D2 floor matching the C3-A-R value after C3-A
closes; and a C4-D taint/perturbation script with frozen perturbation
magnitudes and pass/fail readout. The "minimal flip" must be auditable
by comparing fixture manifests before the derived-audit logic is trusted;
otherwise fixture design freedom has merely moved later. C4-A remains
provisional while C2-A/C3-A are scaffolds. No controller/harness run.

**2026-05-16 (PT) — maintainer. Cut-1 fixture manifest landed
(Wave-1 partial concrete fill).** The first of the five C4-A operational
artifacts now exists on disk: the hashable Cut-1 known-vacuous fixture
manifest. The remaining four (D1 probe set, minimal-flip generator/diff,
C4-D taint/perturbation script, C4-B two-sided self-test table) are
deferred to later waves because each depends on the C2-A / C3-A numeric
fills that are not yet landed; C4-A remains provisional while those
upstream scaffolds are still in flight. This append records what is
filed and pins its hashes; it does not change the freeze above.

*Why this one first.* The Cut-1 fixture is the only C4-A artifact that
has **zero coupling** to the not-yet-landed C2-A/C3-A `[E]` values:
it is `[G]` immutable (the real, already-frozen `structural-failure-p2-harness.mjs`
objects, §6 of this freeze). Pinning it now lets the minimal-flip
generator be authored against a known-good reference, and lets the
C4-B audit have its vacuous-side fixture independently verifiable.

*Pinned artifacts (paths repo-relative, hashes SHA-256, 2026-05-16
PT).*

| artifact | path | sha256 |
| --- | --- | --- |
| Cut-1 fixture manifest | `results/structural-failure/cut2-prereg/cut1-fixture-manifest.json` | raw `41ce1bd8305c856a2a9b4d7d73d942ba1715d205d61ced7f7d6fd9c5c2942ea1` · canonical `3b69bf3c3e32a9a7807ed7ce3382629e9f5864c42671fc79a2f11c062458c97e` |
| Fixture extractor (regenerator) | `scripts/cut2-cut1-fixture-extract.mjs` | `894e6efbd3e8732e273db8f84ecb84df5fe4995ed6355db82523147375c36fde` |
| Source file (the harness itself) | `scripts/structural-failure-p2-harness.mjs` | `43001506e569f5e646afac4e52b59af94c61fdd2ba70d473f60a501038e56015` |

*Per-fixture-object pinning.* Each of the seven Cut-1 fixture objects
is hashed individually so the C4-B two-sided self-test can detect a
within-function edit even if the harness file's overall hash drifts
elsewhere. Line ranges are 1-indexed inclusive against the source-file
SHA above.

| object | role in Cut-1 vacuity | lines | sha256 |
| --- | --- | --- | --- |
| `makeBundle` | bundle generator (`f_par = R22/cos(h)` — the g(h) tautology source) | 154–159 | `f10eb0ea5917a3d61685bbfa8d65c923a49012a9976e2a3956f37ed4de964a4f` |
| `transparentAdapter` | A1 adapter; decoys structurally excluded from `J` | 175–201 | `dc132106de62d78ddfe9637dd5bd582e476d0b229ea1832ea628ac776c1bc89a` |
| `routeEstimate` | grid-search route inverter (`g⁻¹(g(h))` by construction) | 203–237 | `a93b07f37cda6dde5cd4483abe1479a6a4e86e72a77fe2e24f3cfeb495232e1b` |
| `analyticInverseEstimate` | closed-form matched baseline (identical to route on eligible set — the D1 trap C4-C repairs) | 239–250 | `79d2266a6036f50ed1f1c315b48499f601be754eec58cb1c1fdb660233da8e43` |
| `positiveControlEstimate` | decoy-correlate positive control (the moving part of Cut-1 that stayed valid post-reclassification) | 282–312 | `89466d2d7783859185a735c2fa9226c2fc091ba9796d6fd69d15af8b1f522e6a` |
| `routeConstructionAudit` | hardcoded vacuity assertion (the asserted-not-derived C4 target) | 367–376 | `537ce83141ee2a4c10f9cc3f24fac41fdc18fc63c1a134a82e4291fa42f70349` |
| `classifyRouteOutcome` | verdict classifier (short-circuits on `routeTestVacuous: true`) | 378–427 | `2b644b5fae18d5df45225b54508594479e2c08cc40610d9ab481d083f36330b3` |

*Immutability discipline.* These line ranges are the C4-A §6 [G]
"immutable real artifact" rows. Editing any of them changes the
known-vacuous side of the C4-B two-sided self-test; doing so requires
an append-only redesign of C4-B (not a manifest amendment that
retroactively legitimizes the edit). The fixture-extractor is
deterministic given the source file; re-running it on the same source
yields byte-identical output. To re-pin after an *allowed* harness
edit (i.e. one that does not touch the seven ranges above — e.g.
adding a new helper below `runHarness`):

```
node scripts/cut2-cut1-fixture-extract.mjs
```

The new manifest's canonical-JSON SHA-256 is recorded in the next
audit-notes append.

*Vacuity summary recorded in the manifest itself.* The fixture-objects
manifest captures the five Cut-1 vacuity factors as flags
(`route_equals_analytic_baseline_by_construction`,
`decoys_outside_route_objective`,
`cza_tangent_do_not_affect_q_estimate`,
`supralateral_hardcoded_as_non_handle`,
`route_construction_audit_hardcoded`) plus the
`decoy_correlate_positive_control_moves` flag. These are read by the
derived C4 audit; for this fixture all five route-side flags are true,
which is exactly the condition the derived audit must classify
`MACHINERY_LIVE_ROUTE_TEST_VACUOUS`.

*Still open in C4-A.* D1 probe set; minimal-flip generator/diff; C4-D
input-manifest + boundary-perturbation script; C4-B two-sided self-test
table. Each waits on either C2-A (must-differ band definition + frozen
`[E]` values) or C3-A (shared D2 floor) — work for Waves 4–7 of the
ordered execution plan. C4-A remains provisional while C2-A/C3-A are
scaffolds; this manifest's pinning is `[G]` and stands independently.

Justification: closes the first C4-A operational-artifact obligation
(the Cut-1 known-vacuous fixture, hashable, regenerable) with all hashes
recorded in writing. No frozen body edited; no threshold/boundary
moved; D2 floor still pending C3-A-R closure (one shared number).
Public-Language Constraint remains fully in force. Cut-2-execute
remains HELD on the joint admission re-run with the four remaining
C4-A artifacts + C2-A receipts + C3-A receipts + C5 also landed.
