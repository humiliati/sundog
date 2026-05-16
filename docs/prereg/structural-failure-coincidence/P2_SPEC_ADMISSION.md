# Structural Failure Coincidence — P2 Spec Admission Check

Pre-registration: [`README.md`](README.md) (frozen 2026-05-15 PT)  
P2 run spec under review: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md) (frozen
2026-05-15 PT)  
Gated by: [`BOUNDARY_MAP.md`](BOUNDARY_MAP.md) (P0 PASS) and
[`P1_ADMISSION.md`](P1_ADMISSION.md) (P1 PASS)  
Filed: **2026-05-15 (PT)**. Reviewer: Codex audit.  
Status: append-only below the **Amendments** rule; the body above it is
the frozen P2-spec admission record.

## Purpose

This is the short admission check required by `P2_RUN_SPEC.md` before any
controller evaluation may execute. It mirrors P1, but reviews the run
spec rather than the boundary map:

- every bundle feature must trace to `BOUNDARY_MAP.md` / receipts;
- every threshold must be frozen and have a stated role;
- the transparent adapter must be fully closed-form, with no learned or
  post-hoc parameters;
- mandatory decoys must be present and non-vacuous;
- L5 must be applied as the evidence-admissibility rule inside scoring.

## Admission Checklist

| requirement | ruling | notes |
| --- | --- | --- |
| Frozen run spec exists before controller execution | **PASS** | `P2_RUN_SPEC.md` was filed before any controller run. |
| P0/P1 gates are cited and inherited | **PASS** | `BOUNDARY_MAP.md` and `P1_ADMISSION.md` are explicit gates. |
| Four quantities are scored separately | **PASS** | Convergence, steerability, boundary coincidence, and efficiency are separated; efficiency is non-fatal. |
| L5 is applied inside scoring, not as a fifth quantity | **PASS** | Spec correctly treats rendered ≠ anchored as evidence admissibility. |
| Genuine handles trace to P0/P1 receipts | **PASS** | `f_par`, `f_cza`, and `f_tan` map to L1/L2/L3. |
| Mandatory decoys are present | **PASS** | `d_sup`, `d_unanch`, and `d_style` are named. |
| Outcome mapping preserves the Proxy-Collapse bridge | **PASS** | P2 fail on steerability or boundary coincidence maps to the P1/debunked vocabulary. |
| Transparent adapter is fully specified and cannot leak `h` | **HOLD** | The phrase "per h-regime" could be read as allowing the hidden cause into the adapter, and the CZA/tangent consistency terms are not closed-form enough to implement without interpretation. |
| Decoy-edit test is non-vacuous for this controller cut | **HOLD** | Because the transparent adapter excludes `d_*` from `J`, a decoy perturbation cannot move the existing extremum-seeking controller unless the implementation violates the spec. That is a useful adapter-integrity / void test, but not yet the sharp controller-vs-correlate discriminator the prose claims. |
| Thresholds trace to receipts or declared engineering tolerances | **HOLD** | The thresholds are frozen, but their exact roles/origins are not traced in a table. They may be acceptable as pre-registered operational tolerances, but that must be stated before execution. |

## Findings

### F1 — Adapter gating must not use hidden `h`

The adapter paragraph says `f_cza` / `f_tan` gate consistency terms
"per `h`-regime." That is ambiguous. The scorer may use true `h` to
evaluate convergence after the run, and the bundle generator may use true
`h` to generate `B(h)`, but the controller adapter must not read true
`h` when choosing terms. Otherwise the hidden cause has leaked into the
objective.

**Required pre-run amendment:** publish the adapter as an explicit
algorithm whose allowed inputs are only the bundle values and the
candidate hypothesis `q` (for example: `f_par`, `f_cza`, `f_tan`, `R22`,
and `q`). Gating must be by observable bundle state (`f_cza == 0`,
`f_tan == null`, etc.), not by true hidden `h`.

### F2 — Decoy invariance is currently an adapter-integrity test, not a full correlate test

The decoys are the right design move, but under the hard transparent
adapter `J` excludes `d_sup`, `d_unanch`, and `d_style` by construction.
For the existing extremum-seeking controller, that means decoy-edit
invariance is guaranteed unless the adapter has been implemented
incorrectly. This catches a real hazard — a decoy term entering `J` makes
the run void — but it does not by itself prove the controller rejected a
correlate, because the controller never had access to decoys through the
objective.

**Required pre-run amendment:** choose and record one of these scopes:

1. **Transparent-adapter sanity cut.** P2 first-cut admits only an
   adapter-integrity result: decoy edits are void-guard / leakage checks,
   not a behavioral discriminator. Public interpretation must not call a
   decoy-invariance pass a Proxy-Collapse falsification.
2. **Sensitivity-controlled cut.** Add a pre-registered decoy-correlate
   positive control or raw-bundle controller path that can read decoys. A
   deliberately decoy-driven baseline must move under decoy edits while
   the traceable controller does not. Then the decoy-edit test becomes
   non-vacuous.

Either choice can be honest; the current prose needs one of them before a
controller run.

### F3 — Thresholds are frozen, but their provenance needs a table

The spec freezes `τ1 = 1.5°`, `τ2 = 2.0°`, decoy invariance `≤0.5°`, and
boundary coincidence `±1.5°`. That is good pre-registration discipline.
But the run-admission gate says thresholds must trace to
`BOUNDARY_MAP.md` / receipts, while the exact numeric tolerances are not
receipt values in the same way the 32° and 29° loci are.

**Required pre-run amendment:** add a short threshold-provenance table
stating, for each threshold, whether it is a geometry boundary, a
receipt-derived tolerance, or a pre-registered engineering tolerance. This
does not change any threshold; it prevents a later reader from mistaking
operational tolerance choices for atmospheric-optics facts.

### F4 — Matched-baseline efficiency is under-specified but non-blocking

Quantity (4) is explicitly non-fatal, so this is not an admission
blocker. Still, before execution the matched baseline should be named
concretely: e.g. analytic inverse `q = arccos(R22 / f_par)` on
L1-eligible inputs, with no access to decoys. Otherwise the reported
efficiency ratio will be hard to interpret.

## Admission Verdict

**HOLD — P2 is started, but P2-execute is not admitted yet.**

The frozen run spec is the right artifact-before-agent move, and the main
objects align with the prereg: four quantities, L5 as admissibility, and
mandatory decoys. The admission gate also did its job: it caught three
pre-run ambiguities that would otherwise let the evaluation look sharper
than it is.

Controller execution remains **blocked** until the P2 spec receives
append-only amendments resolving F1–F3 and this admission check is
re-run. The Public-Language Constraint remains in force everywhere,
including the rail: no theorem / universal-proof / traceability-success
language until quantities (1)+(2)+(3) actually pass under an admitted
run.

---

## Amendments

Append-only. Each amendment: timestamp (date + zone), author, one-line
justification. The body above is the frozen P2-spec admission record.

*(no amendments yet)*
