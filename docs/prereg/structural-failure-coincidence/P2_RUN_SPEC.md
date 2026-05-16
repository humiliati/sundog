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

**2026-05-16 (PT) — maintainer. Cut-2 write-path policy + derived-audit
verdict default.** Append-only; the frozen body is unchanged.
Operationalises the new **C5 — publication-plumbing freeze** condition
filed against [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md) and makes
precise the verdict-file consequence of the C4-required derived
`routeConstructionAudit`. **No frozen threshold, boundary, adapter rule,
decoy obligation, or outcome mapping changes.**

*Cut-2 allowed write paths.* A Cut-2 harness invocation, and any helper
script it invokes, may write only to:

- `results/structural-failure/cut2-*/` (e.g.
  `results/structural-failure/cut2-preflight/`,
  `results/structural-failure/cut2-execute/`);
- transient stdout/stderr and node-module/temp paths outside the
  repository tree.

The amendment files under
`docs/prereg/structural-failure-coincidence/`
(`P2_CUT2_C3_*.md`, `P2_CUT2_C4_*.md`, any addendum to
`P2_RESULTS.md`) are filed **by hand** under the existing append-only
discipline; the harness itself MUST NOT write into `docs/`.

*Cut-2 forbidden write paths.* The Cut-2 harness, and any helper script
it invokes, MUST NOT modify any of:

- `README.md`
- repo-root `*.html`
- `public/data/`
- `chat/`
- `docs/SUNDOG_V_*.md`
- `docs/index.html`
- anywhere under `dist/`
- any deployment / build artifact

Any violation ⇒ run **VOID**, verdict reclassified
`PUBLICATION_PLUMBING_VIOLATION` per C5. The pre/post `git diff
--exit-code` guard specified in C5 of `P2_SPEC_ADMISSION.md` is the
mechanical enforcement bookending the run.

*Verdict-file default-HOLD on derived-audit failure (makes the C4
consequence explicit).* C4 already requires the harness to **compute**
`routeConstructionAudit` from the live objects (not return a hardcoded
boolean), with predicates: (i) route `q̂` differs from
`analyticInverse(bundle)` on a probe set; (ii) decoy gradient into the
objective is non-zero (decoys reachable); (iii) boundary behavior is
not a direct function of a generator flag the route reads. This
amendment fixes what the verdict file is allowed to say:

- If **any** `routeConstructionAudit` predicate returns `false`, the
  harness MUST emit verdict `MACHINERY_LIVE_ROUTE_TEST_VACUOUS` (or
  `PUBLICATION_PLUMBING_VIOLATION` if C5 has also tripped) and MUST NOT
  emit `TRACEABILITY_HARNESS_PASS`, `CONFIRMED`, or any
  traceability-success language anywhere in the verdict, manifest, or
  emitted markdown.
- The verdict file MUST log which predicate(s) failed, with the live
  values used by the computation. A hardcoded or asserted predicate
  result is itself a C4 violation and voids the run.
- A `TRACEABILITY_HARNESS_PASS` verdict is permitted only when **all
  three** `routeConstructionAudit` predicates return `true` *and* the
  four-quantity score (1)+(2)+(3) passes under the admitted Cut-2 spec.
  Neither half alone authorises PASS.

*Scope of this amendment.* This is the verdict-file rule and the
harness write-path policy. It is not a threshold or boundary change.
The τ-thresholds (τ1=1.5°, τ2=2.0°, decoy-invariance ≤0.5°, coincidence
window ±1.5°, τ_pc=2.0°), the 32°/29° loci, the 2%·R22 leverage line,
the L4 supralateral permanent-fail classification, the A1 adapter input
set, the A2 paired-contrast decoy-edit rule, and the A3 immutability /
amend-only-never-post-results rule all remain unchanged.

Justification: closes the publication-plumbing seam at the harness
level and pins the verdict-file consequence of a computed-audit
failure. Both clauses are mechanical re-statements of guards the
prereg already mandates in prose; neither moves a frozen receipt
boundary or an engineering tolerance, and neither edits a frozen body.
Public-Language Constraint remains fully in force everywhere
(including the rail) until quantities (1)+(2)+(3) actually pass under
an admitted discriminating run that also satisfies C5.

**2026-05-16 (PT) — maintainer. C3 pre-registration drafted
(append-only).** Filed
[`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md):
the decoy term is a competing Gaussian ridge `D(q;d) =
exp(−[q_h − ĥ_dec(d)]²/2σ_D²)` centred on an in-sample decoy→altitude
correlate `ĥ_dec` (the A2 positive-control policy instantiated), entering
the *same* scalar `I = I_route + κ·D − pen` the bound controller climbs.
Three pre-run, auditable obligations: **C3-R** (`∂I/∂d` bounded away
from zero on the carrier path — reachability, the explicit removal of
Cut-1's `∂J/∂d ≡ 0` vacuity); **C3-T** (a decoy-correlate policy beats
the anchor-biased route in-sample by margin `M`, and that advantage
**reverses** under the pre-registered decoy-edit and counterfactual
handle-edit — a genuine trap, not a superior route); **C3-B** (the
load-bearing calibration window: `κ` large enough to be reachable, small
enough that the route stays a feasible competing optimum — no
rigged-to-fail null). `D` reads only observable `d_*` and `q_h`; true
`h` anywhere in the adapter ⇒ VOID (A1). No threshold/boundary moved
(`κ, σ_D, M, P_in, (w,b), seed, τ_pc` are A3 engineering tolerances,
never post-results). **Honest coupling recorded:** C3-B(ii)'s
route-side check needs C2-B's `pen(q)`/`q_a` fixed first; C3-R, C3-T,
C3-B(i) are well-posed independently. C3-A (freeze the named C3
numerics) remains open. Cut-2 execution stays **HELD** pending
C2-A/B/C/D, C3 (incl. C3-A, C3-B), C4, C5 and the admission re-run;
Public-Language Constraint in force. Justification: files the C3 design
condition as an artifact-before-agent; no harness, nothing run.

**2026-05-16 (PT) — Codex audit.** C3 execution admission withheld.
Additional C3 blockers recorded in
[`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md):
define reachability so it is not falsified by the Gaussian decoy ridge's
own zero-gradient point at `q_h=ĥ_dec(d)` or by clipped regions, and
couple C3-T's temptation margin against `π_route` to C2-B because the
route policy is not well-defined until `pen(q)` / `q_a` are fixed. No
threshold/boundary moved; no controller run.

**2026-05-16 (PT) — maintainer. C4 pre-registration drafted
(append-only).** Filed
[`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md): the
`routeConstructionAudit` becomes a predicate set **computed from the
live run objects**, not the current hardcoded fail-safe. Three derived
predicates, each with a frozen floor written to `verdict.json`: **D1**
route separable from `arccos(R22/f_par_obs)` on the region they must
differ by construction (not "differ everywhere" — that would forbid the
correct answer); **D2** the C3-C argmax-sensitivity receipt
(`‖Δ(argmax_q I)/Δd‖` > floor — decoys move the *converged answer*, the
correct fix for the Gaussian zero-gradient point); **D3** boundary
behavior emergent (`h ∉ adapterInputs`, no generator-bit echo). Pass
requires D1∧D2∧D3 **and** the four-quantity score. Load-bearing
**C4-B**, surfaced adversarially: the derived audit must be regression-
tested against the **Cut-1 known-vacuous fixture** (must flag vacuous)
*and* a synthetic non-vacuous fixture (must not) — the audit is proven
on the very self-seal it exists to catch before it is trusted. Honest
couplings recorded: D1↔C2-A/B, D2↔C3-C/C3-A, D3↔A1; C4-A (freeze the
floors/fixtures/probe set) open. No threshold/boundary moved; Cut-2
execution stays **HELD** pending C2-A/B/C/D, C3-A/B/C/D, C4
(incl. C4-A, C4-B), C5 and the admission re-run; Public-Language
Constraint in force. Justification: files the C4 meta-condition as an
artifact-before-agent; no harness, nothing run.

**2026-05-16 (PT) — Codex audit.** C4 execution admission withheld.
Additional C4 blockers recorded in
[`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md): D1 must be
computed from route construction (`π_route` / `argmax I_route`), not the
bound controller's final behavior; D3 must specify a mechanical
inspection/taint or boundary-perturbation method for proving emergent
boundary behavior rather than generator-bit echo. No threshold/boundary
moved; no harness written or run.

**2026-05-16 (PT) — maintainer. C2-B resolution drafted
(append-only).** Filed
[`P2_CUT2_C2B_PEN_AND_QA.md`](P2_CUT2_C2B_PEN_AND_QA.md). The free-`q_a`
degeneracy is removed by a receipt-grounded convex anchor-prior:
`q_a ∈ [−A,+A]` (`A = ρ·R22`, no `h`) and `pen(q) = λ(q_a/A)²`. The
controller climbs `O = I_route − pen`, whose **unique** global maximum
is `(q_h*, q_a*) = (arccos(R22/f_par_obs), 0)` — exactly the biased
naive inverse `q_naive ≠ h`. This makes `π_route` well-defined and C2
P-A computable, unblocking C3-T's baseline, C3-B(ii), C3-D and C4-C/D1.
Load-bearing **C2-B(i)/(ii)**, surfaced adversarially: pre-run numeric
proof that the frozen `λ` both breaks the degeneracy (along-manifold
conditioning ≥ floor) and does **not** move the optimum off
`arccos(R22/f_par_obs)` (P-A not a λ-artifact). Honest findings
recorded: C2-B(ii) numerics fold into the C2-A freeze; the
`f_par_obs<R22` geometry is made explicit but its classification is
**deferred to C2-D**; and a genuine **C4-C/D1 tension is flagged** —
once `argmax I_route ≡ arccos(R22/f_par_obs)` by construction, D1's
comparison target must be the **P-A form (vs true `h` on the
must-differ band)**, not "vs its own closed form" (raised for the C4
reviewer; frozen C4 body not edited). `λ`/floor/tolerance are A3
tolerances, never post-results; no immutable boundary moved. Cut-2
execution stays **HELD** pending C2-A/C/D, C3-A/B/C/D, C4-A/B/C/D, C5
and the admission re-run; Public-Language Constraint in force.
Justification: closes the cascade-hub C2-B as an artifact-before-agent;
no harness, nothing run.

**2026-05-16 (PT) — Codex audit.** C2-B direction accepted; execution
admission withheld. The convex anchor prior removes the free-`q_a`
degeneracy and makes `π_route` well-defined without hidden-`h` access.
The λ-window remains an open C2-A numeric freeze item, and C4-D1 must be
repaired to the P-A target (route optimum differs from true hidden `h`
on the must-differ band) because C2-B intentionally makes the route
optimum equal `arccos(R22/f_par_obs)`. No controller run.

**2026-05-16 (PT) — maintainer. C2-C + C2-D drafted (append-only),
completing the C2 design layer.** Filed
[`P2_CUT2_C2CD_LEVERAGE_GATE_AND_INVALID.md`](P2_CUT2_C2CD_LEVERAGE_GATE_AND_INVALID.md).
**C2-C** specifies the observable-only leverage-confidence gate
`I_route_full = C_L1(s_obs)·[P + 1[f_cza_obs]·T_cza + 1[f_tan_obs]·T_tan]`,
`s_obs = f_par_obs/R22 − 1` (no `h`): a smooth L1 ramp (noise-bounded,
hence graded) and genuine CZA/tangent consistency-term *presence* whose
removal at L2/L3 is the documented singularity, not a controller branch
— with the honest "emergent vs flag-read" question explicitly handed to
C4-D's D3 taint test, and a load-bearing C2-C(i)/(ii) window
(boundary must be detectable **and** discriminating, not invisible and
not rigged-to-fail). **C2-D** classifies `f_par_obs < R22`
(`arccos` undefined) rows as **abstain/invalid, never clipped**: not
counted in q1, scored under q3-L1 as a built-in zero-ambiguity
correlate detector (a confident `q̂` where the inverse is undefined ⇒
q3 fail), with the abstain required to be emergent from C2-B's
degenerate objective (coupled to C2-B and C4-D, not an injected branch).
All C2-C/C2-D numerics fold into the C2-A freeze; the C2 design layer is
now complete. No immutable boundary moved; Cut-2 execution stays
**HELD** pending C2-A, C3-A/B/C/D, C4-A/B/C/D, C5 and the admission
re-run; Public-Language Constraint in force. Justification: closes the
last C2 design sub-blockers as artifacts-before-agent; no harness,
nothing run.

**2026-05-16 (PT) — Codex audit.** C2-C/D direction accepted;
execution admission withheld. C2-A must now freeze and demonstrate two
behavioral facts before any Cut-2 harness exists: the scalar
`C_L1(s_obs)` ramp actually produces detectable degradation/abstain
behavior for `PhotometricAgent` rather than merely preserving the same
argmax at lower amplitude, and `f_par_obs < R22` abstention is read from
a frozen objective-level criterion rather than a branch. C2-A should
also rule whether `C_L1` is deliberate whole-route-package gating or
prove it does not mask L2/L3 tests. No controller run.

**2026-05-16 (PT) — maintainer. C2-A numeric freeze drafted
(append-only).** Filed
[`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md),
resolving the two C2-C/D holds + the package-gating clarification.
**C2-A-1:** `C_L1` is not argmax-inert because `PhotometricAgent` has an
absolute-intensity reacquire path (`reacquire_threshold=0.05`,
`reacquire_hold_steps=30`) and a curvature-limited TRACK estimator;
receipt obligation = the frozen ramp drives the full-field target
intensity below the controller's *own* reacquire threshold in the
L1-ineligible band (emergent re-scan, no confident `q̂`) and above it in
the eligible band — proven against the documented controller constants,
not a rescaled argmax, not a branch. **C2-A-2:** `f_par_obs<R22` abstain
is a frozen objective-level criterion (`max_q O<O_floor` / no
`|r|≤r_tol` / cond `>κ_cond_max`), no branch. **C2-A-3:** ruling — not
whole-package masking; L1 (low-`h`) and L2/L3 (high-`h`,`C_L1≈1`) are
`h`-disjoint by geometry, with a separation receipt. **Load-bearing
anti-self-seal:** the bridge `I→detector_intensity` scale is frozen by
an independent convention (eligible-band peak ≡ 1.0) *before* the
receipts, so "prove C_L1 bites" cannot degenerate into "pick a scale
that makes it bite"; a failing receipt **blocks** (append-only
redesign), never a tuned pass. Full §4 freeze is provenance-tagged
([G] immutable geometry/receipt vs [E] pre-registered engineering
tolerance, A3). No immutable boundary moved; Cut-2 execution stays
**HELD** pending the C2-A receipts, C3-A/B/C/D, C4-A/B/C/D, C5 and the
joint admission re-run; Public-Language Constraint in force.
Justification: closes the C2 numeric freeze as an artifact-before-agent;
receipts pre-run, no harness, nothing run.

**2026-05-16 (PT) — Codex audit.** C2-A direction accepted, but the
freeze is not closed yet. The file freezes the correct mechanism and
anti-self-seal convention, but the [E] rows still require concrete
numeric values plus the three receipt tables before admission. C2-A-1
must phrase lock/fail using the real `PhotometricAgent` phase semantics:
the threshold is checked during TRACK after `reacquire_hold_steps`, not
as a pre-SCAN lock gate. Therefore the receipt needs a frozen
sustained-TRACK/confident-`qhat` readout. C2-A-2 still needs the
objective scan showing invalid rows abstain and eligible rows do not. No
controller run.

**2026-05-16 (PT) — maintainer. C3-A numeric freeze drafted
(append-only).** Filed
[`P2_CUT2_C3A_NUMERIC_FREEZE.md`](P2_CUT2_C3A_NUMERIC_FREEZE.md). Mirrors
the accepted C2-A structure: freezes the C3 numeric **slots + provenance
+ receipt obligations** (concrete `[E]` values and receipt tables are
the maintainer's pre-run fill, same as C2-A; a failing receipt blocks,
never tunes). Inherits C2-A's scale/seed/grid and **propagates the
sustained-TRACK confident-`q̂` readout** so every C3 bound-controller
readout respects `PhotometricAgent`'s real SCAN/SEEK/TRACK semantics
(not peak-intensity-alone). Keystone anti-self-seal (the C3 analog of
C2-A §5): the `P_in` decoy↔`h` correlation, `κ`, `M` are frozen by an
independent principle **before** any run, so the temptation cannot be
tuned to "come out right." Three receipts: C3-A-R (argmax-sensitivity
reachability, the C3-C fix), C3-A-T (in-sample temptation + reversal,
`π_route` now C2-B-well-posed), C3-A-B (κ calibration window, C3-B(ii)↔
C2-B now dischargeable). The C3-A-R floor **= the C4 D2 floor (one
shared number)**. No immutable boundary moved; Cut-2 execution stays
**HELD** pending the C3 receipts, C4-A/B/C/D, C5 and the joint admission
re-run; Public-Language Constraint in force. Justification: closes the
C3 numeric-freeze structure as an artifact-before-agent; receipts
pre-run, no harness, nothing run.

**2026-05-16 (PT) — Codex audit.** C3-A direction accepted, but closure
is withheld. The C3 file is a numeric-freeze scaffold until the concrete
[E] values and C3-A-R/T/B receipt tables are appended, and it cannot
close ahead of C2-A because it inherits C2-A's scale/seed/grid/readout.
The `P_in` anti-self-seal needs an operational frozen sample/generator,
seed, decoy coefficient table, and `(w,b)` fit receipt; a prose
independent principle is not sufficient for admission. Runtime `D` must
read only frozen coefficients and observable decoys; C4-D must taint
check no hidden `h` input. No controller run.

**2026-05-16 (PT) — maintainer. C4-A audit freeze drafted
(append-only), completing the C-condition columns.** Filed
[`P2_CUT2_C4A_AUDIT_FREEZE.md`](P2_CUT2_C4A_AUDIT_FREEZE.md). Mirrors
the C2-A/C3-A scaffold discipline and **propagates the C3-A operational-
artifact ruling**: the D1 probe set, the Cut-1 known-vacuous fixture,
the synthetic non-vacuous fixture, and the C4-D taint method must each
be concrete reproducible seed-deterministic objects, not prose. D2
floor **= the shared C3-A-R floor (one number)**; sustained-TRACK
readout inherited; honest ordering recorded (C4 receipts provisional
until C2-A/C3-A close). C4-C repaired D1 = route construction vs **true
`h`** on the must-differ band (not vs `arccos(R22/f_par_obs)`).
**Keystone anti-self-seal:** the synthetic non-vacuous fixture is the
*minimal mechanical flip of the real Cut-1 fixture* (toggling exactly
the D1/D2/D3 properties), frozen before the audit logic — C4-B is the
same audit run both ways (vacuous-on-Cut-1 / non-vacuous-on-flip), no
fixture-design freedom. C4-D made concrete: an input-manifest assertion
(no `h`, no boundary-branch flags, `(w,b)` frozen at runtime) + a
boundary-perturbation test (behavior tracks identifiability collapse via
the sustained-TRACK readout). A failing self-test blocks (append-only
redesign), never tunes. No immutable boundary moved; Cut-2 execution
stays **HELD** pending C4-A artifacts/receipts, C5 and the joint
admission re-run; Public-Language Constraint in force. Justification:
closes the final C-condition column's structure as an
artifact-before-agent; receipts pre-run, no harness, nothing run.

**2026-05-16 (PT) — Codex audit.** C4-A direction accepted, but closure
is withheld. "Minimal mechanical flip" must become a reproducible
fixture generator/diff plus manifests proving only D1/D2/D3-relevant
fields changed; the Cut-1 fixture must be extracted/hashable from the
real harness; the D1 probe set, D2 floor, and C4-D taint/perturbation
script need concrete files/tables and frozen pass/fail readouts. C4-A
also remains provisional until C2-A/C3-A close, because it inherits
their scale/grid/readout and D2 floor. No controller run.
