# Structural Failure Coincidence Pre-Registration

Roadmaps:
[`SUNDOG_V_GEOMETRY.md`](../../SUNDOG_V_GEOMETRY.md),
[`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md)
Pre-registered: **2026-05-15 (PT)**
Author: maintainer
Status: append-only. Edits below the **Amendments** rule require a
timestamp and a written justification. The body above the rule is frozen at
pre-registration time.

This is the Phase 0 falsifier for the traceability claim behind Sundog. It is
not an agent-training spec. It is the artifact that must exist before an agent
is allowed to run, because the central risk is not failure to converge; it is a
controller that converges by an opaque correlate while a post-hoc probe makes
the result look interpretable.

## Claim Under Test

The defensible claim is:

> Sundog is a traceability harness for indirect-inference alignment: a
> benchmark where the hidden cause, the indirect signal, the inverse route, the
> action, and the failure boundary can be checked separately.

The claim being rejected as insufficient is:

> A probe can decode the hidden cause from the agent's hidden state, therefore
> the agent used the closed-form inverse.

Probe decodability is not evidence of route use. A probe can show that
sun-altitude information is present somewhere in the representation; it does
not show that the agent's action is causally organized around the inverse.

## Non-Negotiable Traceability Tests

Traceability requires both tests below.

1. **Causal steerability.** Feed a counterfactual indirect signal generated
   from a different hidden cause. The agent's internal estimate, if one is
   exposed, and its behavior must move to the closed-form-predicted new value.
   Decodable is not enough; the inverse has to be a lever.
2. **Structural failure coincidence.** The agent must fail where the
   closed-form inverse is ill-posed. If the agent succeeds through a regime
   where the documented inverse has no eligible handle, then it is not using
   that inverse. It has found a correlate inside the training distribution.

The second test is the first falsifier because it can be written before
training. A real inverse carries its singularities with it.

## Frozen Structural Predictions

The following table names the geometric loci where a traceable agent must lose
or switch handles. The "mere correlate" column is the predicted failure mode
for an opaque policy that learned dataset regularities rather than the
documented inverse.

| locus | closed-form / identifiability boundary | traceable agent prediction | mere-correlate prediction |
| --- | --- | --- | --- |
| Parhelion offset route | The promoted inverse is `offset = R22 / cos(h)` only on the strict eligible set: p2, p7, p13. Low-h photos with `sec(h) - 1` below 2% of R22 are anchor-noise-bounded; parhelion-derived R22 photos are tautological; p26 right side is geometrically invalid. | Succeeds on strict eligible cases and reports low leverage or ineligibility outside them. It does not count tautological or invalid photos as independent inverse evidence. | Produces smooth altitude estimates across low-leverage, tautological, or invalid rows because image style, crop, metadata, or halo prominence correlates with `h`. |
| CZA visibility cutoff | CZA is usable only while the literature route is in-window; above about 32.2 deg sun altitude the CZA disappears / exits the visible hemisphere. | A CZA-dependent route fails, abstains, or switches at the cutoff. It does not preserve a CZA-apex inverse past disappearance. | Continues to report altitude through the cutoff because other image features carry distributional information. |
| Tangent arc to circumscribed merge | The Pass C7 tangent-locus guard returns no separate upper-tangent handle at about 29 deg and above; the feature has merged into the circumscribed-halo regime. | A tangent-dependent route degrades, abstains, or switches at the merge. It does not claim continuous tangent-curvature recovery through the singularity. | Maintains a continuous tangent-like estimate through the merge because it is using a learned texture/shape correlate rather than the canonical tangent handle. |
| Supralateral route | Across the tested low-altitude span the predicted h-spread is about 0.3 deg, below visual-edge measurement noise. Even with coverage, this is a structural-discrimination failure. | Refuses to promote supralateral position as a useful inverse handle under the documented apparatus. | Treats supralateral brightness, crop position, or co-occurring arcs as a usable altitude channel. |
| Rendered but unanchored primitives | The atlas distinguishes rendered core, optional vocabulary, named-only literature coverage, and not-modeled rows. Rendered does not mean anchored. | Only anchored closed-form rows can count as inverse evidence. Optional or named-only rows can support display, vocabulary, or future hypotheses, not traceability. | Uses the presence of any drawn or named primitive as evidence that the inverse is available. |

## First Falsifier Before Agents

Before training, the experimenter must produce a boundary map with:

- the eligible input regimes for each inverse handle;
- the abstain / switch / fail regimes for each handle;
- the exact source file or receipt that defines each boundary;
- the expected behavior of a traceable agent and of a correlate agent at each
  boundary.

If that map cannot be written crisply from the current geometry receipts, the
agent run is blocked. The fix is more geometry specification, not more
training.

## Agent Run Admission Rule

An agent run may proceed only after the boundary map above exists. The run must
score four quantities separately:

1. convergence to the hidden target;
2. counterfactual steerability under indirect-signal edits;
3. failure-boundary coincidence with the closed-form identifiability map;
4. matched-baseline efficiency.

The first three are the traceability claim. The fourth is an efficiency claim.
Failure on efficiency does not erase a traceability result; failure on
steerability or boundary coincidence does.

## Outcome Branching

| outcome | interpretation | publication stance |
| --- | --- | --- |
| Cannot write the boundary map | The inverse is not understood crisply enough to test an agent. | Halt; publish no agent claim. |
| Agent fails to converge inside eligible regimes | Convergence null survives. | Publishable null / D path. |
| Agent converges but is not steerable | Probe-style decodability trap. | Rebrand as opaque correlate; no traceability claim. |
| Agent converges and is steerable but crosses analytic failure boundaries without degradation, abstention, or handle switch | The policy is using another route. | Rebrand as correlate / benchmark finding; no theorem posture. |
| Agent converges, is steerable, and its failure boundary coincides with the closed-form identifiability boundary | Traceability harness passes on this domain. | Stakeholder-safe B path: benchmark / apparatus claim, not universal theorem. |

## Public-Language Constraint

Until the structural-failure-coincidence test has passed, public copy should
prefer:

- traceability harness;
- indirect-inference alignment benchmark;
- hidden-cause recovery from indirect signals;
- falsifiable apparatus;
- identifiability boundary.

Avoid:

- theorem;
- universal alignment proof;
- "probe decoded it" as route evidence;
- any claim that indirect signals generally beat direct state.

---

## Amendments

Append-only. Each amendment must carry a timestamp (date + zone), author, and
a one-line justification. The body above this rule is frozen at
pre-registration time.

**2026-05-15 (PT) — maintainer.** P0 deliverable created and frozen:
[`BOUNDARY_MAP.md`](BOUNDARY_MAP.md) — the "First Falsifier Before
Agents" boundary map (loci L1–L5, each with eligible / abstain-switch-fail
regime, exact source receipt, and traceable-vs-correlate predictions).
Justification: the prereg required this artifact to exist before any
agent run; it is now written entirely from existing geometry receipts.
P0 completion-gate verdict: **PASSES** — all five loci are receipt-cited,
no row BLOCKED; two coded-vs-stated reconciliations recorded (L2 CZA
coded `h ≤ 32°` operative vs literature ~32.2°; L4 supralateral receipt
~0.5°/h=0–22° vs prereg ~0.3°), neither altering a regime. Program
admitted to **P1 (falsifier admission review)**; P2 (agent run) remains
blocked. Roadmap phase ladder added to
[`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) Candidate 13.

**2026-05-15 (PT) — maintainer.** P1 admission review filed and passed:
[`P1_ADMISSION.md`](P1_ADMISSION.md). Findings: L5 re-scoped to an
evidence-admissibility rule (not a behavioral locus); L2/L4 reconciliation
rulings fixed (operative `h≤32°`; supralateral permanent-fail). The
falsifier is confirmed *informed by* the existing falsifiability surface
[`../../debunked.md`](../../debunked.md): orthogonal to Pushable Occluder,
adjacent to Occluded-Code Score Aliasing, and the rigorous pre-agent
instrument for the `Proxy Collapse` ("most important scientific failure")
avenue. Prioritization + a shared verdict-vocabulary bridge are fixed in
P1_ADMISSION §C so the surfaces communicate. **P2 (agent run) unblocked
in principle but not started** — gated on the agent + matched-baseline
harness; Public-Language Constraint remains in force (no theorem language
anywhere, including the rail) until structural-failure-coincidence passes
in P2. Justification: roadmap P1 gate executed; review carried as
amendments, frozen bodies unedited.

**2026-05-15 (PT) — maintainer.** P2 run specification authored and
frozen: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md). First-cut scope (user-fixed):
the **existing photometric/extremum-seeking controller** (no training)
on a **closed-form feature bundle** (no renders). Spec pre-registers the
decoy-bearing bundle (`d_sup`/`d_unanch`/`d_style`), a hard
**transparent-adapter constraint** (closed-form, fixed, no learned/
post-hoc parameters — closes the P1-L5 hazard one level down), and the
four quantities with frozen thresholds (τ1=1.5°, τ2=2.0°, decoy-
invariance ≤0.5°, boundary-coincidence ±1.5° at h=32°/29°). Justification:
P2's analog of the "artifact before the agent" rule — the run protocol
is frozen *before* the controller is run. **P2-execute remains blocked**
pending a short P2-spec admission check (mirrors P1). Public-Language
Constraint still in force.

**2026-05-15 (PT) — Codex audit.** P2-spec admission check filed:
[`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md). Verdict **HOLD, no
controller execution yet**. The audit confirms the frozen P2 artifact is
the right desk-work gate, but execution remains blocked until append-only
amendments resolve: (1) adapter gating without hidden-`h` leakage, (2)
the non-vacuity / scope of the decoy-edit test under the transparent
adapter, and (3) threshold provenance. Justification: mirrors P1's
admission discipline; prevents a self-sealing P2 run from being reported
as a controller-vs-correlate discriminator.

**2026-05-15 (PT) — maintainer.** P2-spec admission HOLD **resolved and
re-admitted**. F1–F4 closed by an append-only amendment to
[`P2_RUN_SPEC.md`](P2_RUN_SPEC.md): A1 explicit no-hidden-`h` adapter
algorithm (input set `{f_par,f_cza,f_tan,R22,q}`, VOID if `h` read); A2
decoy-edit made non-vacuous via a pre-registered decoy-correlate
positive control (`τ_pc=2.0°`, paired contrast, inconclusive branch); A3
threshold-provenance table separating immutable geometry/receipt
boundaries from pre-registered engineering tolerances; A4 names the
analytic-inverse matched baseline. Re-review appended to
[`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md): **ADMIT — P2-execute
admitted.** Carried conditions: the controller run obeys the AGENTS.md
~10-min staging rule; the Public-Language Constraint stays in force until
quantities (1)+(2)+(3) actually pass under the admitted run; geometry
boundaries immutable, tolerances amend-only/never post-results.
Justification: the gate caught real pre-run ambiguities and is now
satisfied by explicit closed-form fixes.

**2026-05-15 (PT) — Codex execution.** P2 first-cut execution completed:
[`P2_RESULTS.md`](P2_RESULTS.md). Command: `npm run p2:structural`.
Harness: `scripts/structural-failure-p2-harness.mjs`. Output:
`results/structural-failure/p2-execute-first-cut/`. Verdict:
`TRACEABILITY_HARNESS_PASS` for the admitted transparent route controller
on the closed-form feature bundle; decoy-correlate positive control:
`OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED`. Quantities (1)–(3) all
passed for this first cut; (4) recorded a route/analytic sample ratio of
`1601`. Public-language guard remains: this is an apparatus / benchmark
result for this domain, **not** a universal theorem proof and **not** a
debunking result. Justification: P2-execute was admitted, deterministic,
and completed under the ~10-minute rule.

**2026-05-15 (PT) — correction / reviewer challenge accepted.** The P2
execution interpretation immediately above is reclassified. The first-cut
harness was a tautological route test: `f_par` was generated as
`R22/cos(h)`, the route objective inverted it by grid search, and the
analytic baseline inverted it in closed form. It therefore did not
exercise a policy distinct from the analytic inverse, did not make decoy
invariance behavioral, and did not establish q1–q3 traceability. Corrected
verdict:
`MACHINERY_LIVE_ROUTE_TEST_VACUOUS`. The positive-control result remains
valid (`OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED`) and the no-leak
adapter invariant remains valid, so the honest result is machinery-live /
route-test-vacuous. Public-language guard remains in force: no
`CONFIRMED`, no traceability-success, and no theorem language from this
first cut.

**2026-05-15 (PT) — Cut 2 C1 filed.** Controller binding record filed:
[`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md).
The named existing Sundog extremum-seeking controller for Cut 2 is
`sundog.agents.photometric.PhotometricAgent` from `agents/photometric.py`,
with existing runner evidence in `experiments/run_baseline_comparison.py`
and `experiments/stress_tests.py`. This closes **C1 only**. Cut-2-execute
remains **HELD** pending C2-C4 and a fresh admission re-check.

**2026-05-15 (PT) — maintainer.** Staged discriminating-cut
pre-registration filed (append-only) to
[`P2_RUN_SPEC.md`](P2_RUN_SPEC.md): **Cut 2** (closed-form
discriminating — named existing controller, non-invertible nuisance,
*tempting* in-sample decoys reachable through `J`, emergent boundary,
derived vacuity audit) to run after admission; **Cut 3** (rendered
signal) pre-registered as the conditional escalation if Cut 2 is
ambiguous. Admission re-check appended to
[`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md): verdict **HOLD —
Cut-2 design admitted in principle, Cut-2-execute NOT admitted** pending
C1 (bind the named existing controller), C2 (concrete non-invertible
nuisance), C3 (demonstrate decoy temptation + reachability), C4 (derived
audit). No harness written for either cut; no frozen threshold/boundary
moved. Public-Language Constraint remains fully in force. Justification:
the spec's own staging rule, applied after the first cut proved vacuous
rather than ambiguous; artifact-before-agent re-asserted.

**2026-05-15 (PT) — maintainer.** C1 + C2 filed (append-only). **C1**:
[`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md) —
bound the actual pre-existing controller
`sundog.agents.photometric.PhotometricAgent` (verified 2026-04-27
origin, inverse-free extremum-seeker, line-cited). **C2**:
[`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md)
— receipt-grounded non-invertible anchor nuisance (single-handle inverse
biased exactly at the L1 leverage boundary), the A1-compliant
bundle→Observation intensity-field bridge, and two pre-run anti-self-seal
obligations (P-A no back-door tautology, P-B no rigged null) plus a
deterministic bias-demonstration table. **C3** (decoy term +
reachability/temptation) and **C4** (derived audit) remain open; Cut-2
execution stays HELD pending them and the admission re-check. No frozen
threshold/boundary moved; Public-Language Constraint in force.
Justification: closes C1 and files the C2 design condition as an
artifact-before-agent.

**2026-05-15 (PT) — Codex freeze audit.** C2 is **not execution-frozen**.
[`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md)
has been marked **filed for audit — HOLD for execution**. The design is
directionally correct, but the audit found C2-local blockers: missing
numeric tolerance/domain values, unspecified `pen(q)` / `q_a` range
creating an exact-ridge degeneracy, unspecified leverage-confidence
function, and undefined noisy-inverse handling when `f_par_obs < R22`.
Cut-2 execution remains HELD; C3 is still the next design condition, but
must not assume C2 is admitted until those blockers are resolved.

**2026-05-16 (PT) — maintainer.** Publication-plumbing freeze opened as
**C5**, a fifth open condition Cut-2-execute must clear at the C3/C4
admission re-check alongside C1 (closed), C2 (filed for audit; C2-A/B/C/D
open), C3 (open), and C4 (open). Filed in response to an external read
of the program's publication-plumbing risk surface
([`../../geometry_agent_audit.md`](../../geometry_agent_audit.md)).
**C5** condition opened in
[`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md): the Cut-2 harness may
write only under `results/structural-failure/cut2-*/`; a pre/post
`git diff --exit-code` guard scoped to `README.md` / root `*.html` /
`docs/` excluding `docs/prereg/structural-failure-coincidence/` /
`chat/` / `public/data/` / `dist/` must return clean; any violation ⇒
verdict `PUBLICATION_PLUMBING_VIOLATION`, never PASS. Companion
amendment in [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md) declares the matching
allowed/forbidden write paths and pins the verdict-file rule on
derived-audit failure (any `routeConstructionAudit` predicate `false` ⇒
`MACHINERY_LIVE_ROUTE_TEST_VACUOUS`; PASS requires all three predicates
true *and* the four-quantity score to pass). **No frozen body edited;
no threshold, boundary, adapter rule, decoy obligation, or outcome
mapping moved.** Public-Language Constraint remains fully in force.
Justification: closes the prose-only Public-Language seam by adding a
mechanical guard at admission and at run time, while explicitly leaving
`scripts/copy-site-docs.mjs` untouched.

**2026-05-16 (PT) — maintainer.** C3 filed (append-only):
[`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md)
— concrete decoy ridge `κ·D` in the same intensity field the bound
controller climbs, with pre-run obligations C3-R (reachability,
`∂I/∂d ≠ 0` — the explicit fix for Cut-1's structural decoy exclusion),
C3-T (decoy policy beats the anchor-biased route in-sample but the
advantage reverses under the q2 edits — a genuine trap), and C3-B (the
load-bearing `κ` calibration window — no vacuous, no rigged-to-fail),
with the honest C3-B(ii)↔C2-B coupling recorded. C3-A (freeze the named
C3 numerics) remains open. Open-conditions list for Cut-2-execute
re-admission is now **C2-A/B/C/D, C3 (incl. C3-A, C3-B), C4, C5**; the
admission re-check is one audit of the whole cut and remains withheld
until all are filed. No frozen threshold/boundary moved; Public-Language
Constraint in force. Justification: files the C3 design condition as an
artifact-before-agent.

**2026-05-16 (PT) — Codex audit.** C3 execution admission withheld. The
decoy-ridge direction is accepted, but
[`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md)
now records two additional C3-local blockers: reachability must be
defined away from the Gaussian ridge's zero-gradient point / clipped
regions, and C3-T's temptation margin against `π_route` is coupled to
C2-B because the route baseline is not defined until `pen(q)` / `q_a`
are fixed. Cut-2 execution remains HELD; no harness, nothing run.

**2026-05-16 (PT) — maintainer.** C4 filed (append-only):
[`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md) — the
`routeConstructionAudit` is made a predicate set **derived from the live
objects** (D1 route≠analytic on the must-differ region · D2 the C3-C
argmax-sensitivity receipt · D3 emergent boundary), pass requires
D1∧D2∧D3 plus the four-quantity score. Load-bearing **C4-B**: the
derived audit must be regression-tested against the Cut-1 known-vacuous
fixture (must flag vacuous) and a synthetic non-vacuous fixture (must
not) — the guard proven on the self-seal it exists to catch. Honest
couplings recorded (D1↔C2-A/B, D2↔C3-C/C3-A); C4-A numerics open. The
full open-conditions list for Cut-2-execute re-admission is now
**C2-A/B/C/D · C3-A/B/C/D · C4 (incl. C4-A, C4-B) · C5**; the admission
re-check is one audit of the whole cut, withheld until all are filed. No
frozen threshold/boundary moved; Public-Language Constraint in force.
Justification: files the C4 meta-condition as an artifact-before-agent.
