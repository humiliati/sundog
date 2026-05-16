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

**2026-05-15 (PT) — re-review (maintainer).** Re-run of this admission
check after the F1–F4 resolution amendment in
[`P2_RUN_SPEC.md`](P2_RUN_SPEC.md). Each prior HOLD re-checked against
the cited clause, not rubber-stamped:

- **F1 → CLOSED.** Spec amendment A1 publishes the adapter as an explicit
  closed-form algorithm with input set fixed to `{f_par, f_cza, f_tan,
  R22, q}`; `h` excluded; CZA/tangent gating is now on observed
  `f_cza==0` / `f_tan==null`; hard VOID invariant if `h` is read inside
  the adapter. The leak path is removed and the terms are implementable
  without interpretation. **PASS.**
- **F2 → CLOSED.** A2 takes Option 2: a pre-registered decoy-correlate
  positive control (raw-bundle, reads `d_*`) plus `τ_pc = 2.0°`. The
  decoy-edit is now the paired contrast (route invariant **and** positive
  control moves ≥ τ_pc), with an explicit *inconclusive* branch and a
  guard that, absent the positive control, a decoy-invariance pass is
  adapter-integrity only and **never** a Proxy-Collapse falsification.
  Non-vacuous. **PASS.**
- **F3 → CLOSED.** A3 adds the provenance table separating immutable
  geometry/receipt boundaries (32°, 29°, 2%·R22, supralateral) from
  pre-registered engineering tolerances (τ1, τ2, decoy-invariance,
  coincidence window, τ_pc), with the immutability rule. No value
  changed. **PASS.**
- **F4 → RESOLVED** (was non-blocking). A4 names the matched baseline
  (`q = arccos(R22/f_par)`, L1-eligible, no decoy access), distinct from
  the A2 positive control.

**Re-admission verdict: ADMIT — P2-execute is admitted.** Conditions
carried: (i) the controller run obeys the AGENTS.md "~10-minute rule" —
stage as operator PowerShell if the sweep exceeds ~10 min, with the
frozen thresholds/branches; (ii) the Public-Language Constraint stays in
force everywhere (including the rail) until quantities (1)+(2)+(3)
actually pass under this admitted run; (iii) any threshold change remains
append-only, justified, and never post-results, and geometry/receipt
boundaries are immutable. Justification: all three blocking findings are
closed by explicit, closed-form, pre-run amendments; the gate did its job
and is now satisfied.

**2026-05-15 (PT) — Codex execution note.** The admitted P2 first-cut run
has executed; result filed in [`P2_RESULTS.md`](P2_RESULTS.md). This does
not alter the admission ruling. It records that the admitted conditions
were consumed by `npm run p2:structural`, which completed under the
~10-minute rule and produced `TRACEABILITY_HARNESS_PASS` for the route
controller plus `OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED` for the
decoy-correlate positive control.

**2026-05-15 (PT) — correction / reviewer challenge accepted.** The
execution note immediately above is superseded as an interpretation. The
admission ruling remains valid, but the executed harness was not an
adequate admitted route-use test: it reduced the route controller to the
same inverse as the analytic baseline (`g^-1(g(h))`), kept decoys
structurally unreachable by the route objective, did not use CZA/tangent
state to estimate `q`, and hardcoded supralateral as a non-handle.
Corrected execution verdict:
`MACHINERY_LIVE_ROUTE_TEST_VACUOUS`. The positive-control verdict remains
`OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED`. Therefore P2 quantities
(1)–(3) have **not** established traceability under an admitted
discriminating run; the Public-Language Constraint remains in force.

**2026-05-15 (PT) — C1 re-check note.** C1 is closed only by
[`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md):
Cut 2 must bind `sundog.agents.photometric.PhotometricAgent` from
`agents/photometric.py`, preserving its `reset(...)` / `act(obs)`
interface. This closes the "named existing controller" gap that Cut 1
violated. Cut-2-execute is still **NOT admitted**: C2 (non-invertible
nuisance + bias demonstration), C3 (reachable tempting decoys), and C4
(computed vacuity audit) remain open, and the admission check must be
re-run after they are filed.

**2026-05-15 (PT) — Codex audit. Admission re-check of the staged
discriminating-cut pre-registration (Cut 2 + Cut 3).** Reviewing the
`P2_RUN_SPEC.md` staged-cut amendment of the same date. This is a
genuine pass, not a rubber stamp: the design is scrutinized for whether
it actually removes the Cut-1 tautology.

| requirement | ruling | notes |
| --- | --- | --- |
| Cut 2 makes convergence non-tautological | **PARTIAL** | A pre-registered *non-invertible* nuisance is specified, but "monotone unknown-parameter confound" is not yet a concrete closed form. Until written, it is not provable that `arccos(R22/f_par)` is a biased recovery of `h` — otherwise Cut 2 silently re-collapses to `g^-1∘g` + denoise. |
| Decoys genuinely tempt (in-sample-predictive **and** reachable through `J`) | **PARTIAL** | The intent is correct and is the sharp move Cut 1 lacked. But "reachable through the objective" must be demonstrated pre-run (non-zero decoy gradient into `J`) **and** the decoy must measurably *beat* the anchored route in-sample, or invariance is not a costly choice. Not yet shown numerically. |
| Agent under test is the **named existing** controller | **HOLD** | The binding requirement is correctly stated, but the concrete Sundog-controller entrypoint is **not yet named**. Cut 1's core failure was an inlined proxy; Cut 2 cannot be admitted until the real controller module/function is identified and bound by reference. |
| Vacuity audit is **derived**, not asserted | **HOLD** | Spec correctly requires `routeConstructionAudit` be computed from live objects. Current harness returns it hardcoded (correct fail-safe *now*, unacceptable for a discriminating run). Must be re-implemented as a computed predicate before Cut 2. |
| Cut 3 trigger is crisp; px↔° hazard pre-named | **PASS** | Ambiguity trigger is operationalized (three explicit branches); the Phase-15 px↔° centring hazard is named as a Cut-3 admission blocker, not glossed. |
| Thresholds immutable; only σ/seed added | **PASS** | No frozen threshold moved; σ and RNG seed are correctly scoped as pre-registered, never-post-results engineering tolerances under the A3 rule. |
| Outcome framing honest | **PASS** | The amendment states the honest prior (likely **D / BOUNDARY FOUND** = Proxy-Collapse confirmation; **B** only on measured refusal-under-temptation). No theorem posture; the in-between is explicitly disclaimed. |

**Re-admission verdict: HOLD — Cut-2 *design* admitted in principle;
Cut-2-execute NOT admitted.** The staged pre-registration is the right
artifact-before-agent move and the gate did its job: it caught four
pre-run gaps that would otherwise let a "discriminating" run be
self-sealed again. Cut-2 build/run remains **blocked** until append-only
amendments close, and this check is re-run on:

- **C1** name and bind the existing Sundog controller entrypoint (no
  inlined proxy);
- **C2** write the concrete non-invertible nuisance closed form and show
  `arccos(R22/f_par)` is biased under it;
- **C3** demonstrate pre-run that decoys are reachable through `J`
  (non-zero gradient) **and** an explicit decoy-correlate policy beats
  the anchored route in-sample (temptation is real);
- **C4** re-implement `routeConstructionAudit` as a derived predicate.

Cut 3 stays staged-and-blocked behind Cut 2 plus its own px↔° hazard
resolution. Public-Language Constraint remains in force everywhere
(including the rail): no `CONFIRMED` / traceability-success / theorem
language until (1)+(2)+(3) pass under an admitted discriminating run.
Justification: mirrors P1 / P2-spec admission discipline; prevents a
second self-sealing P2 run by gating execution on closed-form closure of
C1–C4.

**2026-05-15 (PT) — filing log (not a ruling).** C1 and C2 have been
filed against the open conditions:
[`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md)
(C1 — controller bound, independently verified pre-existing 2026-04-27,
inverse-free, line-cited) and
[`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md)
(C2 — non-invertible nuisance, bias-demonstration design, bridge
architecture, P-A/P-B obligations). This entry **records filing only and
is not an admission ruling**. The re-check verdict is deliberately
withheld until **C3** (concrete decoy term, reachability-through-`I`,
in-sample temptation) and **C4** (derived `routeConstructionAudit`) are
also filed, so the re-run audits the *whole* discriminating cut at once —
including whether C2's P-A/P-B numeric artifacts actually exist and hold.
Cut-2 build/run remains **blocked**; Public-Language Constraint in force.

**2026-05-16 (PT) — C3 audit.** C3 direction accepted; execution
admission withheld. [`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md)
correctly puts a decoy ridge `κ·D` in the same intensity field the
controller climbs and records the C3-B(ii)↔C2-B coupling. Additional
audit blockers now explicit: (C3-C) reachability must be defined in a
way that survives the Gaussian ridge's zero-gradient point at
`q_h = ĥ_dec(d)` and any clipped `ĥ_dec` regions, either by an
off-ridge carrier band or a finite-difference argmax-sensitivity test;
(C3-D) the C3-T temptation margin against `π_route` is also coupled to
C2-B, not only C3-B(ii), because `π_route` is not well-defined until
`pen(q)` and `q_a` are frozen. C3-A/C3-B remain open; no controller run.

**2026-05-15 (PT) — C2 freeze audit.** C2 direction accepted but
execution freeze **withheld**. [`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md)
now records C2 as **filed for audit — HOLD for execution**, not as an
admitted cut component. Four C2-local blockers must be resolved before
the C3/C4/full admission re-check can pass: (C2-A) freeze the numerical
engineering tolerances/domains (`ρ`, `A`, `σ`, `seed`, grid, sample count,
`q_h`/`q_a` domains, condition-number bound); (C2-B) specify `pen(q)` and
the admissible `q_a` range, because free `q_a` makes `I_route` a
degenerate exact ridge and defeats P-A; (C2-C) specify the
leverage-confidence function without hidden-`h` access; (C2-D) specify
undefined `arccos` handling when noisy `f_par_obs < R22`. No controller
run; Public-Language Constraint remains in force.

**2026-05-16 (PT) — maintainer.** Append-only amendment opening **C5 —
publication-plumbing freeze** as a fifth open condition Cut-2-execute
must clear at the C3/C4 admission re-check, alongside C1 (closed), C2
(filed for audit; blockers C2-A/B/C/D), C3 (open), and C4 (open). Filed
in response to an external read of the program's publication-plumbing
risk surface
([`../../geometry_agent_audit.md`](../../geometry_agent_audit.md)).
**No frozen body edited; no threshold moved; no regime reclassified.**

*Why C5 is needed.* The Public-Language Constraint is currently a
**prose** guard. `scripts/copy-site-docs.mjs` runs as `postbuild` and
copies the entire `docs/` tree to `dist/docs/` with no excludes, so any
verdict file or summary written under `docs/` propagates to public
posture on the next `npm run build` / `npm run deploy`. Today's harness
happens to write only to `results/structural-failure/...`
(`scripts/structural-failure-p2-harness.mjs` lines 710–736), but that is
discipline, not enforcement. C5 makes the constraint mechanical at
admission and at run time.

*C5 — publication-plumbing freeze (open).* Cut-2-execute is blocked
until the re-admission check verifies all three:

1. **Allowed write paths declared and enforced.** The Cut-2 harness
   writes only under `results/structural-failure/cut2-*/`. The
   amendment files under
   `docs/prereg/structural-failure-coincidence/`
   (`P2_CUT2_C3_*.md`, `P2_CUT2_C4_*.md`, any addendum to
   `P2_RESULTS.md`) are filed **by hand** under the existing append-only
   discipline — the harness itself may not write into `docs/`.

2. **Pre/post diff guard.** A `git diff --exit-code` taken *before* any
   Cut-2 build step and again *after* the run, scoped to:

       README.md
       *.html              (repo root)
       docs/               EXCLUDING docs/prereg/structural-failure-coincidence/
       chat/
       public/data/
       dist/

   must return clean on both bookends. The pre-run snapshot is the
   baseline; the post-run snapshot must match it byte-for-byte over
   that scope. The exact `git diff` invocation used as the guard is
   pre-registered alongside the C3/C4 filings.

3. **Violation ⇒ reclassified verdict, never PASS.** Any non-clean diff
   in (2), or any harness write outside the allowed paths in (1),
   reclassifies the verdict as `PUBLICATION_PLUMBING_VIOLATION`. The
   Public-Language Constraint stays in force; the harness MUST NOT emit
   `TRACEABILITY_HARNESS_PASS`, `CONFIRMED`, or any traceability-success
   language; and no public-surface edit produced during the violating
   run may be merged.

*What C5 deliberately does NOT do.* It does not edit
`scripts/copy-site-docs.mjs`, does not introduce a `dist/` build exclude
for the prereg folder, and does not move any frozen threshold or
boundary. The prereg folder may still ship to `dist/` on a normal
build; C5 prevents the Cut-2 *experiment itself* from being the thing
that writes into surfaces that ship. Hardening
`scripts/copy-site-docs.mjs` is a separate, larger decision and is
deliberately out of scope for this amendment.

*Companion amendment.* The matching harness-level write-path policy and
the verdict-file rule on derived-audit failure are filed as the
2026-05-16 amendment to [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md). C5 is the
admission-gate condition; the P2_RUN_SPEC amendment is its operational
form during a Cut-2 run.

Justification: closes the publication-plumbing seam at the admission
gate before any Cut-2 run, by adding a mechanical guard alongside the
prose Public-Language Constraint. C5 is opened, not closed; it must be
satisfied at the re-admission check together with C2-A/B/C/D, C3, and
C4. No body rewrite, no threshold move, no post-hoc edit. The
Public-Language Constraint remains fully in force.

**2026-05-16 (PT) — filing log (not a ruling).** C3 has been filed:
[`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md)
(concrete decoy ridge `κ·D`; obligations C3-R reachability, C3-T
in-sample temptation that reverses under the q2 edits, C3-B calibration
window; C3-A numerics open; C3-B(ii) honestly recorded as coupled to
C2-B). **This records filing only and is not an admission ruling.** The
re-check verdict stays withheld until the *entire* discriminating cut is
filed — **C2-A/B/C/D, C3 (incl. C3-A, C3-B), C4, C5** — so the re-run
audits one coherent object, including whether each condition's pre-run
numeric artifact actually exists and holds, and whether the
cross-couplings (notably C3-B(ii)↔C2-B) are discharged. Cut-2 build/run
remains **blocked**; Public-Language Constraint in force.

**2026-05-16 (PT) — filing log (not a ruling).** C4 has been filed:
[`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md) — the
derived `routeConstructionAudit` predicate set (D1 separability, D2 the
C3-C argmax-sensitivity receipt, D3 emergent boundary), the load-bearing
**C4-B** self-test (the audit must flag the Cut-1 known-vacuous fixture
and pass a synthetic non-vacuous one), and the honest D1↔C2-A/B,
D2↔C3-C/C3-A couplings; C4-A numerics open. **Filing only, not an
admission ruling.** The re-check verdict stays withheld until the entire
discriminating cut is filed — **C2-A/B/C/D, C3-A/B/C/D, C4 (incl. C4-A,
C4-B), C5** — and audits one coherent object, including whether C4-B's
Cut-1 fixture actually flags vacuous in the harness suite and whether
the cross-couplings are discharged. Cut-2 build/run remains **blocked**;
Public-Language Constraint in force.

**2026-05-16 (PT) — C4 audit.** C4 direction accepted; execution
admission withheld. Two C4-local blockers added in
[`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md): (C4-C)
D1 must be construction-level (`π_route` / `argmax I_route` vs
`arccos(R22/f_par_obs)` on the must-differ region), not computed from
the bound controller's final `q̂`, because controller behavior belongs
to the four-quantity score; (C4-D) D3 needs a frozen mechanical
inspection/taint or perturbation method proving boundary behavior comes
from observable intensity/curvature collapse rather than a generator-bit
echo. C4-A/C4-B remain open; no harness run.

**2026-05-16 (PT) — filing log (not a ruling).** C2-B resolution filed:
[`P2_CUT2_C2B_PEN_AND_QA.md`](P2_CUT2_C2B_PEN_AND_QA.md) — convex
anchor-prior `pen(q)=λ(q_a/A)²` with `q_a∈[−A,+A]` gives a unique
route-optimum `(arccos(R22/f_par_obs),0)` = the biased naive inverse;
`π_route`/P-A now well-posed (unblocks C3-T, C3-B(ii), C3-D, C4-C/D1);
load-bearing C2-B(i)/(ii) `λ`-window open. **Filing only, not a ruling.**
**Reviewer action item raised:** C2-B exposes that C4-C/D1's comparison
target is degenerate once `argmax I_route ≡ arccos(R22/f_par_obs)` by
construction — D1 must be the **P-A form** (`argmax I_route = q_naive ≠
true h` on the must-differ band), not "differs from its own closed
form." The C4 reviewer should rule on this at the next audit; the
frozen C4 body was not edited. Re-check verdict stays withheld until the
whole cut is filed — **C2-A/B/C/D, C3-A/B/C/D, C4-A/B/C/D, C5** — and
audited as one object including this C4-C/D1 reformulation and all
cross-couplings. Cut-2 build/run remains **blocked**; Public-Language
Constraint in force.

**2026-05-16 (PT) — C2-B audit.** C2-B direction accepted; execution
admission withheld. The `q_a∈[-A,+A]` plus
`pen(q)=λ(q_a/A)^2` construction is A1-safe and removes the free-`q_a`
degeneracy, making `π_route` well-defined as the biased naive inverse.
Open items: C2-B(i)/(ii) λ-window numerics fold into C2-A, and C4-D1
must use the P-A target (route optimum versus true hidden `h` on the
must-differ band), not route optimum versus `arccos(R22/f_par_obs)`.
No harness run.

**2026-05-16 (PT) — filing log (not a ruling).** C2-C + C2-D filed,
completing the C2 design layer:
[`P2_CUT2_C2CD_LEVERAGE_GATE_AND_INVALID.md`](P2_CUT2_C2CD_LEVERAGE_GATE_AND_INVALID.md).
C2-C = observable-only leverage-confidence gate with a load-bearing
detectable-and-discriminating window and an explicit emergent-vs-flag
coupling to C4-D's D3 taint test; C2-D = `f_par_obs<R22` abstain/invalid
(never clipped), a built-in q3-L1 correlate detector whose abstain must
be emergent from C2-B's degenerate objective. **Filing only, not a
ruling.** Reviewer should note the two new cross-couplings (C2-C↔C4-D,
C2-D↔C2-B/C4-D) for the joint audit. With this, the full open set is
**C2-A, C3-A/B/C/D, C4-A/B/C/D, C5**; the re-check verdict stays
withheld until all are filed and audited as one object. Cut-2 build/run
remains **blocked**; Public-Language Constraint in force.

**2026-05-16 (PT) — C2-C/D audit.** C2-C/D direction accepted;
execution admission withheld. The joint admission re-check must require
C2-A to prove the scalar `C_L1(s_obs)` ramp is behaviorally load-bearing
for the actual `PhotometricAgent` confidence/readout (not only an
argmax-preserving intensity rescale), and to freeze the C2-D
objective-level abstain criterion for `f_par_obs < R22` rows (`O*` floor
/ residual / curvature / condition receipt, no injected branch). C2-A
should also state whether `C_L1` intentionally gates the whole route
package or prove it does not mask L2/L3 consistency-term tests. No
harness run.

**2026-05-16 (PT) — filing log (not a ruling).** C2-A numeric freeze
filed: [`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md).
Resolves the two C2-C/D holds (C2-A-1 `C_L1` proven to drive
`PhotometricAgent`'s own reacquire/lock-fail path, not an argmax-inert
rescale; C2-A-2 frozen objective-level `f_par_obs<R22` abstain, no
branch) and the package-gating clarification (C2-A-3: L1↔L2/L3
`h`-disjoint by geometry, separation receipt). **Filing only, not a
ruling.** Reviewer should specifically scrutinize: (a) the §5
independent bridge-scale freeze (eligible-band peak ≡ 1.0, fixed
*before* the receipts) — this is the keystone anti-self-seal, verify no
residual scale freedom; (b) that a failing C2-A-1/2/3 receipt is treated
as a **block / append-only redesign**, never a tuned pass; (c) the
[G]/[E] provenance tags in §4. The re-check verdict stays withheld until
**C2-A receipts, C3-A/B/C/D, C4-A/B/C/D, C5** are all filed and audited
as one object. Cut-2 build/run remains **blocked**; Public-Language
Constraint in force.

**2026-05-16 (PT) — C2-A audit.** C2-A direction accepted; closure
withheld. The document correctly moves the scalar-ramp proof onto the
bound controller's real reacquire path and freezes the bridge scale by
an independent eligible-band peak convention. But the admission check
must treat the current file as a **numeric-freeze scaffold**, not a
completed freeze, until concrete values and receipt tables are appended
for the [E] rows (`rho`, `sigma`, `lambda`, domains, `C_L1`, `T_*`,
`O_floor`, `r_tol`, `kappa_cond_max`, floors). The C2-A-1 receipt must
respect actual `PhotometricAgent` phase semantics: SCAN/SEEK can still
run below threshold; reacquire is a sustained-TRACK behavior after
`reacquire_hold_steps`, so "lock/confident qhat" needs a frozen
sustained-TRACK readout. C2-A-2 needs the objective scan receipt. No
harness run.

**2026-05-16 (PT) — filing log (not a ruling).** C3-A numeric freeze
filed: [`P2_CUT2_C3A_NUMERIC_FREEZE.md`](P2_CUT2_C3A_NUMERIC_FREEZE.md).
Mirrors the C2-A scaffold discipline (slots/provenance/receipt
obligations now; concrete `[E]` values + the C3-A-R/T/B receipt tables
still owed before closure); inherits C2-A scale/seed/grid; propagates
the sustained-TRACK readout. **Filing only, not a ruling.** Reviewer
should scrutinize at the joint re-run: (a) the §2 independent `P_in`
freeze (the C3 keystone anti-self-seal — verify no correlation-strength
freedom remains and a failing C3-A-T/B blocks rather than tunes); (b)
that the C3-A-R floor equals the C4 D2 floor (one number, cross-checked
in C4-A); (c) the C3-B(ii)↔C2-B coupling is genuinely discharged now
that `π_route` is well-posed. The re-check verdict stays withheld until
**C2-A receipts, C3-A receipts, C4-A/B/C/D, C5** are all filed and
audited as one object. Cut-2 build/run remains **blocked**;
Public-Language Constraint in force.

**2026-05-16 (PT) — C3-A audit.** C3-A direction accepted; closure
withheld. C3-A correctly propagates the C2-A sustained-TRACK readout and
uses argmax-sensitivity for reachability, but it inherits a C2-A scaffold
that is not itself closed. Admission must treat C3-A as
slots/provenance/obligations until concrete values and receipt tables
are appended for `P_in`, decoy correlations, `kappa`, `sigma_D`, `M`,
edit magnitudes, and the shared C3-C/C4-D2 floor. The independent
`P_in` freeze also needs an operational finite sample/generator + seed +
coefficient table + frozen `(w,b)` receipt; a general "co-varies with
`h`" principle is not enough to close C3-A-T/B. No harness run.

**2026-05-16 (PT) — filing log (not a ruling).** C4-A audit freeze
filed: [`P2_CUT2_C4A_AUDIT_FREEZE.md`](P2_CUT2_C4A_AUDIT_FREEZE.md) —
**all C-condition columns are now filed**. Mirrors the C2-A/C3-A
scaffold discipline; propagates the operational-frozen-artifact ruling
to C4's probe set / fixtures / taint method. **Filing only, not a
ruling.** Reviewer should scrutinize at the joint re-run: (a) the
keystone — the synthetic non-vacuous fixture is the *minimal mechanical
flip of the real Cut-1 fixture*, frozen **before** the audit logic, so
C4-B is a genuine two-sided self-test with no fixture-design freedom;
(b) the D2 floor literally equals the C3-A-R floor (one number, void on
mismatch); (c) C4-D's two concrete checks (input-manifest assertion;
boundary-perturbation via the sustained-TRACK readout) are operational,
not prose; (d) C4 receipts are provisional until C2-A/C3-A close. The
re-check verdict stays withheld until **C2-A, C3-A, C4-A
receipts/artifacts, C5** are all filed *with concrete
values/artifacts/receipts landed* and audited as one object. Cut-2
build/run remains **blocked**; Public-Language Constraint in force.

**2026-05-16 (PT) — C4-A audit.** C4-A direction accepted; closure
withheld. The minimal-mechanical-flip fixture is the right C4-B
anti-self-seal, and C4-D is correctly moved from prose to manifest +
perturbation checks. Admission still requires concrete frozen artifacts:
D1 probe-set table/file; Cut-1 fixture extraction/manifest; synthetic
fixture generator/diff showing exactly the D1/D2/D3 flips; D2 floor
matching the eventual C3-A-R floor; and a runnable taint/perturbation
script with frozen magnitudes/readouts. Until those artifacts and the
C2-A/C3-A inherited receipts close, C4-A is a scaffold, not an
execution-closing gate. No harness run.

**2026-05-16 (PT) — filing log (not a ruling).** C5 publication-plumbing
freeze filed:
[`P2_CUT2_C5_PUBLICATION_PLUMBING_FREEZE.md`](P2_CUT2_C5_PUBLICATION_PLUMBING_FREEZE.md)
— **every pre-registration condition is now filed.** Operationalizes the
prose C5 amendment per the C4-A artifact discipline. **Filing only, not
a ruling.** Reviewer should scrutinize at the joint re-run: (a) the
keystone — the guard is genuinely default-deny / allowlist-complement
over the *full tree*, not a blocklist that can be under-scoped, and the
scope is never re-curated post-run; (b) the manifest is hashable and the
pre/post guard script is actually runnable by a third party (not prose);
(c) `PUBLICATION_PLUMBING_VIOLATION` is terminal-dominant over any
D1∧D2∧D3 / four-quantity pass; (d) C5 is independent of the numeric
fills but the joint re-run still gates on C2-A+C3-A+C4-A+C5 together.
The re-check verdict stays withheld until **all four conditions are
filed with concrete values/artifacts/receipts landed** and audited as
one object. Cut-2 build/run remains **blocked**; Public-Language
Constraint in force.

**2026-05-16 (PT) — C5 audit.** C5 direction accepted; closure withheld.
The default-deny allowlist is the right shape, but the runnable guard
must make baseline semantics explicit (clean pre-run tree versus
pre/post snapshot that rejects only new out-of-allowlist changes).
Admission also requires the guard to cover tracked modifications,
untracked files, and ignored-but-public/shipping paths; `git diff
--exit-code` alone is not sufficient. The manifest/guard must normalize
repo-relative paths and reject symlink/junction escapes before applying
the `results/structural-failure/cut2-*/` allowlist. No harness run.

**2026-05-16 (PT) — filing log (not a ruling).** Wave-4 C3-A receipts
and Wave-4.1 amendment receipts are filed. C3-A-R passes, but C3-A-T and
C3-A-B both BLOCK in v1 and remain BLOCK after the Path-Y/Path-Z
amendment. The load-bearing observation is the split: the frozen decoy
policy helps where the route is low-leverage/ineligible-by-obs and hurts
where the route is eligible/strong; the larger eligible subset dominates
the non-degenerate average, so the route wins overall. Path W is the
filed closeout: preserve the BLOCK receipts, do not tune thresholds or
weights inside Wave 4, and defer any Wave-4.2 redesign to a separate
freeze-level discussion. This filing cannot admit Cut-2 execution; the
joint admission re-run must treat C3-A-T/B as blocking unless a later
append-only redesign is itself admitted. Public-Language Constraint
remains in force.
