# Structural Failure Coincidence — P2 Run Specification (pre-registered)

Pre-registration: [`README.md`](README.md) (frozen 2026-05-15 PT)
Gated on: [`BOUNDARY_MAP.md`](BOUNDARY_MAP.md) (P0, frozen) ·
[`P1_ADMISSION.md`](P1_ADMISSION.md) (P1, PASS)
Roadmap: [`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) ▸ Candidate 13
Existing-surface bridge: [`../../debunked.md`](../../debunked.md)
(verdict-vocabulary map in `P1_ADMISSION.md` §C)
Filed & frozen: **2026-05-15 (PT)**. Author: maintainer. Status:
append-only below the **Amendments** rule. The body above it is the
pre-registered P2 protocol and is frozen before any controller is run.

P2 is the first phase that evaluates an agent. This document is the
artifact that must exist and be frozen **before the controller is run**,
exactly as `BOUNDARY_MAP.md` had to exist before P2 was admitted. It
fixes — in advance, with thresholds — what the run measures and what each
result means. No result may be reported against a threshold edited after
the run; threshold changes are append-only amendments with justification.

## Scope of the first P2 cut (user-fixed 2026-05-15)

- **Agent under test:** the project's **existing photometric /
  extremum-seeking controller** (the Sundog controller family). **No
  training.** First cut answers the cheapest honest question: *does the
  method we already built organize its trajectory around the documented
  inverse, or around a correlate?*
- **Indirect signal:** a **closed-form feature bundle** (below). No
  HaloSim renders, no atlas raster. Counterfactual edits are exact.
- **Staging:** a trained learning agent and/or rendered-image signal are
  **out of scope here** and are promoted only if this cut is ambiguous
  (prereg/Candidate-13 staging rule).

## The closed-form feature bundle

`h` (sun altitude) is the withheld hidden cause. The controller receives
only the bundle `B(h)` below. Every genuine handle is a closed-form
function of `h` traced to `BOUNDARY_MAP.md`; **decoys are mandatory** —
without them quantities (2)/(3) cannot discriminate a route from a
correlate.

**Genuine handles (anchored, closed-form):**

| feature | definition | eligibility / singularity (from BOUNDARY_MAP) |
| --- | --- | --- |
| `f_par` parhelion offset | `R22 / cos(h)` (`parhelion-geometry.mjs` `phase3.daggerOffset`) | L1: informative only where `sec(h)−1 ≥ 2%` of R22; tautological / invalid rows flagged ineligible, not silently numeric |
| `f_cza` CZA-present | `1` iff `h ≤ 32°` else `0` (`czaVisibleAtAltitude`) | L2: handle vanishes at the `h = 32°` coded cutoff |
| `f_tan` tangent locus | `tangentArcLocus(h)` curvature, **`null` at `h ≥ 29°`** | L3: no separate handle past the 29° circumscribed merge |

**Mandatory decoys (must be present, must not be usable handles):**

| decoy | definition | why it is a decoy (BOUNDARY_MAP) |
| --- | --- | --- |
| `d_sup` supralateral position | supralateral angular distance from sun | L4: varies only ~0.5° across `h=0–22°` — below measurement noise; a *non-handle at every altitude*. A traceable controller must be invariant to it; a correlate will ride it. |
| `d_unanch` unanchored-primitive flags | presence flags for rendered-optional / named-only primitives | L5 (as the P1-ruled admissibility rule): rendered ≠ anchored. Must never count as inverse evidence. A correlate will latch onto these because they co-vary with `h` in any finite sample. |
| `d_style` distributional nuisance | a synthetic covariate correlated with `h` in-sample only | catches the generic "image style / metadata correlates with `h`" correlate failure named in BOUNDARY_MAP L1's mere-correlate column. |

## Transparent-adapter constraint (hard)

The existing controller is an extremum-seeker; it does not natively read
`B(h)`. The adapter that maps `B(h)` to the controller's scalar
objective **must be closed-form, fixed, and published in this spec — it
may not be learned, tuned on outcomes, or contain free parameters fit
after seeing `h`.** Rationale: a learned/opaque adapter would relocate
the correlate into the adapter and silently pass the traceability test —
the P1 L5 hazard, one level down. The adapter is:

> objective `J(B, q) = − | f_par − R22/cos(q) |` evaluated **only on
> L1-eligible inputs**, with `f_cza` / `f_tan` gating which consistency
> term is active per `h`-regime, and `q` the controller's internal
> altitude hypothesis it extremum-seeks over. Decoys `d_*` are **not**
> arguments of `J`. Any run whose adapter touches a `d_*` term, or adds
> a post-hoc fit parameter, is **void** and reported as such.

The controller is never given `h`; it climbs `J` over its own `q` and
its converged `q̂` is the read-out estimate. What it is *not* told is as
pre-registered as what it is.

## The four quantities — operationalized, with frozen thresholds

Scored **separately** (Admission Rule). (1)–(3) are traceability;
(4) is efficiency-only and non-fatal.

1. **Convergence.** On the L1-eligible regime, `|q̂ − h| ≤ τ1`,
   **τ1 = 1.5°**, on ≥ 90% of eligible samples. Fail ⇒ convergence
   null (D).
2. **Counterfactual steerability.** Two pre-registered edits, scored
   independently:
   - *handle edit:* set the bundle's genuine handles to the values
     implied by a counterfactual `h′`. Pass iff `q̂ → h′` within
     **τ2 = 2.0°**.
   - *decoy edit:* perturb `d_sup`, `d_unanch`, `d_style` across their
     full range with genuine handles **held fixed**. Pass iff
     `|Δq̂| ≤ 0.5°` (controller is **invariant** to decoys).
   Traceable ⇒ pass *both*. Moves with decoys, or fails to move with
   handles ⇒ opaque correlate; (2) failed; **no traceability claim**
   (this is the Proxy-Collapse confirmation surface — `P1_ADMISSION` §C).
3. **Failure-boundary coincidence.** Sweep `h` through every L1–L4
   transition and score whether the controller degrades / abstains /
   switches **at** the analytic locus:
   - L2 CZA cutoff: behavior change within **±1.5°** of `h = 32°`
     (operative coded guard, P1 §B-1).
   - L3 tangent merge: tangent-handle reliance ceases within **±1.5°**
     of `h = 29°` (Pass C7 / `TANGENT_ARC_CIRCUMSCRIBED_H`).
   - L1 sub-regimes: abstains / flags low-leverage on
     `sec(h)−1 < 2%`-R22, tautological, and invalid rows; does **not**
     emit confident `q̂` there.
   - L4: never promotes `d_sup` to a handle at any `h` (permanent-fail
     row, P1 §B-2).
   Pass iff the controller's failure locus coincides with the analytic
   locus on **all** of L1–L4 within the stated tolerances, **with L5
   applied**: only anchored-handle-driven successes count; any success
   attributable to a decoy is scored as a coincidence **failure**, not a
   pass. Sailing smoothly through a boundary ⇒ correlate ⇒ (3) failed.
4. **Matched-baseline efficiency.** Iterations/samples to reach τ1 vs a
   reference controller given the closed-form inverse directly. Report
   the ratio. **No pass/fail threshold** — efficiency loss does not erase
   a traceability result; recorded for the efficiency claim only.

## Outcome mapping (reuse prereg Outcome Branching + P1 §C bridge)

| run result | prereg outcome | rail verdict (`debunked.md`) |
| --- | --- | --- |
| Fails (1) in eligible regime | convergence null | STALLED / BUSTED — D |
| Passes (1), fails (2) handle-edit or moves on decoy-edit | opaque correlate | STALLED / BOUNDARY FOUND — **Proxy-Collapse confirmation** |
| Passes (1)+(2), fails (3) (sails a boundary / decoy-driven success) | route is not the inverse | BOUNDARY FOUND — correlate, no theorem |
| Passes (1)+(2)+(3) | traceability harness passes on this domain | OPERATING ENVELOPE / CONFIRMED (the specific claim) — apparatus / **B**, never "theorem" |

(4) annotates any row; it never overturns (1)–(3).

## Run-admission gate

The controller run may execute **only after this spec is frozen** and a
short P2-spec admission check (mirroring P1: confirm bundle features +
thresholds + adapter all trace to `BOUNDARY_MAP`/receipts, decoys
present, no post-hoc parameters) records PASS. Until then P2-execute is
**blocked**. The Public-Language Constraint remains in force everywhere
(including the rail): no theorem / universal-proof language until
(1)+(2)+(3) have actually passed here.

---

## Amendments

Append-only. Each amendment: timestamp (date + zone), author, one-line
justification. The body above is the frozen pre-registered P2 protocol.

**2026-05-15 (PT) — Codex audit.** P2-spec admission check filed:
[`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md). Verdict **HOLD**:
P2-run-spec is a valid frozen artifact-before-agent record, but
P2-execute is not admitted until three pre-run ambiguities are resolved by
append-only amendment: (F1) adapter gating must be specified without
access to hidden `h`; (F2) decoy invariance must be scoped as
adapter-integrity only or given a sensitivity-controlled correlate
baseline; (F3) thresholds need a provenance table distinguishing geometry
boundaries from pre-registered operational tolerances. No controller has
been run; Public-Language Constraint remains in force. Justification:
the run-admission gate is designed to catch self-sealing or under-specified
P2 protocols before execution.

**2026-05-15 (PT) — maintainer.** Resolves the admission HOLD (F1–F4).
Append-only; the frozen body is unchanged — the clauses below **override
and make precise** the cited frozen passages, per pre-registration
discipline (no body rewrite, no silent post-hoc edit).

*A1 — F1: explicit no-hidden-`h` adapter.* The frozen "per `h`-regime"
wording is made precise. Adapter input set is **exactly
`{f_par, f_cza, f_tan, R22, q}`** (`R22` = scale-lock constant, `q` =
controller's candidate altitude); **`h` is not an input.** Closed-form,
fixed, no free parameters:

```
adapter(f_par, f_cza, f_tan, R22, q):
  eligible = (f_par >= 1.02 * R22)     # observed sec−1 ≥ 2%·R22 (L1); no h
  if not eligible: return ABSTAIN       # no objective; no confident q̂
  J = -abs(f_par - R22 / cos(q))        # data=observed f_par; model=own q
  if f_cza == 0:    drop CZA term       # observed h>32 regime (no h read)
  if f_tan is null: drop tangent term   # observed h≥29 regime (no h read)
  return J
```

All gating is on **observed bundle state** only. **Hard invariant:** any
implementation that reads `h` (or anything `h`-derived beyond the
already-observable bundle values) inside the adapter ⇒ run **VOID**. The
generator/scorer may use true `h`; the adapter may not.

*A2 — F2: decoy-edit made non-vacuous (Option 2).* Adds a pre-registered
**decoy-correlate positive control**: a raw-bundle controller whose
objective is a generic least-squares fit of `q` over the **full** bundle
**including** `d_sup, d_unanch, d_style` (the explicit opaque-correlate
policy). Decoy-edit passes iff **both** (a) route controller
`|Δq̂| ≤ 0.5°` and (b) positive control `|Δq̂| ≥ τ_pc`, **τ_pc = 2.0°**.
If the positive control does not move ≥ τ_pc the decoy battery is too
weak ⇒ result **inconclusive**, not pass. Only under this paired
contrast may a decoy-invariance pass be read as Proxy-Collapse-relevant;
absent the positive control it is **adapter-integrity only**, never a
Proxy-Collapse falsification.

*A3 — F3: threshold provenance (no value changed).*

| threshold | value | provenance | mutable? |
| --- | --- | --- | --- |
| L2 CZA cutoff | `h=32°` | **geometry boundary** — `czaVisibleAtAltitude`; BOUNDARY_MAP L2 / P1 §B-1 | **No** — immutable physics; a change is a geometry re-spec, not a tolerance edit (forbidden goalpost-move) |
| L3 tangent merge | `h=29°` | **geometry boundary** — `TANGENT_ARC_CIRCUMSCRIBED_H`; Pass C7; Tape AH Ch6 p62 | **No** |
| L1 leverage line | `f_par ≥ 1.02·R22` | **receipt-derived boundary** — BOUNDARY_MAP L1 | **No** |
| L4 supralateral | permanent-fail ∀`h` | **geometry boundary** — BOUNDARY_MAP L4 / P1 §B-2 | **No** |
| τ1 convergence | `1.5°` | **pre-registered engineering tolerance** (vs visual-edge noise) | amend-only, justified, **never post-results** |
| τ2 handle-steerability | `2.0°` | pre-registered engineering tolerance | as τ1 |
| decoy-invariance | `≤0.5°` | pre-registered engineering tolerance | as τ1 |
| coincidence window | `±1.5°` about the 32°/29° loci | pre-registered engineering tolerance (window chosen; locus center immutable) | as τ1 |
| τ_pc positive-control | `2.0°` | pre-registered engineering tolerance | as τ1 |

Rule: **geometry/receipt boundaries are immutable** (editing one moves
the falsifier's goalposts — prohibited). Only engineering tolerances may
be amended, with written justification, never after results are seen.

*A4 — F4: matched baseline named (non-blocking).* Quantity-(4) matched
baseline = the **analytic-inverse controller** `q = arccos(R22/f_par)`
on L1-eligible inputs, **no decoy access** (distinct from the A2
decoy-correlate positive control, which deliberately reads decoys).
Efficiency = iteration/sample ratio (under-test : analytic baseline);
non-fatal annotation only.

*Execution discipline.* P2-execute obeys the AGENTS.md "~10-minute
rule": a controller sweep over ~10 min wall-clock is **staged as
operator PowerShell** with the frozen thresholds/branches above, not run
inline; measured rates recorded in the P2 results doc. The negative is
already pre-registered (this spec).

Justification: each required pre-run amendment (F1–F3, plus the
non-blocking F4) is resolved closed-form before any controller run;
re-admission check appended to `P2_SPEC_ADMISSION.md`.

**2026-05-15 (PT) — Codex execution.** P2 first-cut executed under this
admitted spec; result filed in [`P2_RESULTS.md`](P2_RESULTS.md). Command:
`npm run p2:structural`. Harness:
`scripts/structural-failure-p2-harness.mjs`. Output:
`results/structural-failure/p2-execute-first-cut/`. Verdict:
`TRACEABILITY_HARNESS_PASS` for the admitted transparent route controller;
positive-control verdict:
`OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED`. This amendment records
execution only; it changes no threshold, boundary, adapter rule, or
outcome mapping. Public-language guard: apparatus / benchmark result,
not universal theorem proof and not a debunking result.

**2026-05-15 (PT) — correction / reviewer challenge accepted.** The
execution note immediately above is reclassified. The first-cut harness
did **not** instantiate a route that could fail the traceability test:
`makeBundle(h)` generated `f_par = R22/cos(h)`, the route objective
maximized `-|f_par - R22/cos(q)|`, and the matched baseline computed
`arccos(R22/f_par)`. Thus the route and analytic baseline were the same
inverse (`g^-1(g(h))`), with the route doing it by grid search. Decoys
were structurally outside `J`, CZA/tangent terms did not affect `q`, and
supralateral was hardcoded as a non-handle. Corrected disposition:
`MACHINERY_LIVE_ROUTE_TEST_VACUOUS`, not `TRACEABILITY_HARNESS_PASS`.
The positive-control result stands:
`OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED`; the A1 adapter invariant
also stands. This correction changes no frozen thresholds or outcome
mapping; it records that the first execution was inconclusive for
quantities (1)–(3) as a discriminating route-use test. Public-language
guard remains fully in force: no `CONFIRMED`, no traceability-success,
and no theorem language from this first cut.

**2026-05-15 (PT) — Cut 2 C1 controller binding.** C1 is closed only:
the actual existing Sundog extremum-seeking / photometric controller is
named and bound in
[`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md).
Canonical binding: `sundog.agents.photometric.PhotometricAgent`
(`agents/photometric.py`), instantiated and reset as in
`experiments/run_baseline_comparison.py` `_make_photometric`. Any inlined
route inverter, analytic inverse, grid-search proxy, or reimplemented
extremum seeker is not the existing controller and voids a Cut-2 run under
C1. Cut-2 execution remains **HELD** pending C2-C4 and a fresh admission
re-check.

**2026-05-15 (PT) — maintainer. Staged discriminating-cut
pre-registration (Cut 2 + Cut 3).** Append-only; the frozen body is
unchanged. This operationalizes the frozen body's own staging rule
("a trained learning agent and/or rendered-image signal are out of scope
here and are promoted only if this cut is ambiguous") — except the first
cut was not ambiguous, it was **vacuous** (`g^-1(g(h))`). The amendment
pre-registers the next two cuts *before* either is built or run; no
harness has been written for them. No frozen threshold, boundary,
adapter rule, or outcome mapping changes.

*Defect being removed.* Cut 1's feature was `g(h)`, its route was
`g^-1`, its matched baseline was `g^-1` — an identity measured against
itself. A discriminating cut must make each quantity *earned*: the
inverse not algebraically recoverable; the decoy shortcut *tempting*
(in-sample-predictive and reachable through the objective, so a
correlate genuinely wins in-sample and a traceable route must decline at
a convergence cost); boundary degradation *emergent* from the route's
own signal, not a generator flag echoed against its own constant.

**Cut 2 — closed-form discriminating (run after admission).**

- *Agent under test (binding requirement).* The harness must invoke the
  project's **named existing extremum-seeking controller** (the Sundog
  controller family) through the frozen A1 transparent adapter. It
  **may not inline a fresh inverter** as Cut 1 did. The concrete
  controller entrypoint must be named in the admission check and bound by
  reference; an inlined proxy ⇒ run **VOID**.
- *Signal.* Closed-form, seconds-scale. `f_par` carries a pre-registered
  **non-invertible nuisance** (not merely zero-mean noise): a monotone
  unknown-parameter confound such that `arccos(R22/f_par)` is **not** an
  unbiased recovery of `h`. Convergence must be *worked for*.
- *Decoys must tempt.* `d_sup`, `d_unanch`, `d_style` are
  **in-sample-predictive of `h`** and enter the **same observation space
  the route optimizes over** (reachable through the objective, not
  excluded by construction). A decoy-seeking policy must achieve
  *better* in-sample convergence than the anchored route; the traceable
  route must refuse the shortcut. Decoy-edit invariance is now a costly
  behavioral choice, not a structural identity.
- *Boundary emergence.* At L2 (`h→32°`) / L3 (`h→29°`) the anchored
  signal's leverage must vanish **from the signal itself**; the route
  may not read a generator gating bit. q3 passes only if degradation /
  abstention / handle-switch is driven by vanishing signal leverage.
- *Derived vacuity audit (hard gate, not asserted).* The harness must
  **compute** `routeConstructionAudit` from the live objects, not return
  hardcoded `true`/`false`: (i) route `q̂` differs from
  `analyticInverse(bundle)` on a probe set; (ii) the decoy gradient into
  the objective is non-zero (decoys reachable); (iii) boundary behavior
  is not a direct function of a generator flag the route reads. If any
  predicate fails ⇒ `MACHINERY_LIVE_ROUTE_TEST_VACUOUS`, never a pass.
- *Thresholds.* τ1=1.5°, τ2=2.0°, decoy-invariance ≤0.5°,
  coincidence ±1.5°, τ_pc=2.0° — **unchanged, immutable** (A3 rule). The
  only additions are pre-registered engineering tolerances strictly
  required by the nuisance: noise scale `σ` and RNG `seed`, fixed here
  before any run, never edited post-results.
- *Outcome framing (honest prior).* The frozen Outcome Branching table
  governs. The likely honest result is **D / BOUNDARY FOUND** — the
  existing photometric controller riding the in-sample correlate is the
  `debunked.md` Proxy-Collapse confirmation ("most important scientific
  failure" avenue, P1 §C). **B** is earned *only* by a measured refusal
  of the tempting decoy at a convergence cost *and* emergent boundary
  coincidence. Either is a clean result; the in-between is not.

**Cut 3 — rendered-signal escalation (pre-registered, conditional).**
Triggered **only if Cut 2 is ambiguous**, defined crisply as: q1
within `±0.5°` of τ1 (borderline), **or** q2 decoy-edit and the
positive-control contrast both inconclusive, **or** the derived audit
cannot establish decoy-reachability. On trigger, the signal becomes a
HaloSim render / atlas raster (inverse genuinely non-algebraic from
pixels). Known blocking hazard, pre-named: the px↔° centring / scale
problem exhausted in Phase 15 (`SPECULATIVE_HALO_PROOFS.md` follow-ups)
— Cut 3 admission must show that hazard is resolved (e.g. HaloSim-native
Scale) or Cut 3 is itself blocked, not forced.

**Process gate (the lesson of the near-miss).** Neither cut may be
built or run until a P2-spec admission re-check (appended to
`P2_SPEC_ADMISSION.md`) records **ADMIT** for it. Patch-then-run is the
self-seal this program exists to prevent. Public-Language Constraint
stays fully in force until (1)+(2)+(3) pass under an admitted
*discriminating* run; nothing here is a pass.

Justification: stages the next two cuts as frozen artifacts-before-agent
under the spec's own escalation rule; removes the Cut-1 tautology by
construction; no body rewrite, no threshold move, no post-hoc edit.

**2026-05-15 (PT) — maintainer. C2 pre-registration drafted
(append-only).** Filed
[`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md):
the concrete non-invertible nuisance is a receipt-grounded bounded
additive anchor error `f_par_obs = R22/cos(h) + ε`,
`ε ~ U[−ρ·R22, +ρ·R22]`, whose single-handle inverse
`arccos(R22/f_par_obs)` is biased with the bias blowing up *exactly* at
the L1 leverage boundary (`sec(h)−1 < 2%·R22`) — the non-invertibility
*is* the documented singularity. The bundle→Observation bridge is
pre-registered as a closed-form A1-compliant intensity field
`I = I_route + κ·D − pen` that the bound `PhotometricAgent` climbs, with
two pre-run, auditable anti-self-seal obligations: **P-A**
(`argmax I_route ≠ h` by construction — no back-door tautology) and
**P-B** (landscape climbable on the eligible band — no rigged null),
plus a deterministic pre-run bias-demonstration table. C2 closes the
nuisance + bias-demo + bridge architecture **only**; the decoy term `D`,
its reachability/temptation (**C3**) and the derived audit (**C4**)
remain open. No threshold/boundary moved (`ρ, A, σ, seed, κ` are A3
engineering tolerances, never post-results). Cut-2 execution stays
**HELD** pending C3, C4, and the admission re-check; Public-Language
Constraint in force. Justification: drafts the C2 admission condition as
an artifact-before-agent; no harness, nothing run.

**2026-05-15 (PT) — Codex freeze audit.** C2 execution freeze withheld.
The C2 document is now explicitly **filed for audit — HOLD for
execution**. Blockers: freeze the numerical engineering
tolerances/domains; define `pen(q)` and the allowed `q_a` range so
`I_route` is not a degenerate exact ridge; define the leverage-confidence
gate without hidden-`h` access; define invalid naive-inverse handling for
`f_par_obs < R22`. C3 remains the next design condition, but it should
build on a C2 bridge that resolves these blockers. No controller run.
