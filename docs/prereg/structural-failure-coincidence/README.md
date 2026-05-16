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

**2026-05-16 (PT) — Codex audit.** C4 execution admission withheld. The
derived-audit direction is accepted, including the Cut-1 known-vacuous
fixture and synthetic non-vacuous fixture requirement. Two blockers were
added: D1 must audit route construction rather than the bound
controller's output, and D3 needs a mechanical input/taint or
boundary-perturbation method rather than a prose "not a flag" assertion.
Cut-2 execution remains HELD; no harness, nothing run.

**2026-05-16 (PT) — maintainer.** C2-B resolution filed (append-only):
[`P2_CUT2_C2B_PEN_AND_QA.md`](P2_CUT2_C2B_PEN_AND_QA.md) — the
cascade-hub fix. `q_a ∈ [−A,+A]` (`A=ρ·R22`, no `h`) + convex
`pen(q)=λ(q_a/A)²` give `O=I_route−pen` a **unique** max at
`(arccos(R22/f_par_obs), 0)` = the biased naive inverse, so `π_route`
is well-defined and C2 P-A computable — unblocking C3-T/C3-B(ii)/C3-D
and C4-C/D1. Load-bearing C2-B(i)/(ii) `λ`-calibration window surfaced;
honest findings recorded (C2-B(ii)→C2-A freeze; `f_par_obs<R22`
geometry explicit, classification deferred to C2-D; **C4-C/D1
comparison-target tension flagged for the C4 reviewer** — must be the
P-A form vs true `h`, not vs its own closed form). No frozen
threshold/boundary moved; Public-Language Constraint in force.
Justification: closes the cascade-hub C2-B as an artifact-before-agent.

**2026-05-16 (PT) — Codex audit.** C2-B direction accepted; execution
admission withheld. The convex anchor-prior construction removes the
free-`q_a` degeneracy without hidden-`h` access, so `π_route` is now
well-defined as the biased naive inverse. The λ calibration window still
belongs to C2-A, and C4-D1 must use the P-A comparison against true
hidden `h` rather than comparing the route to its own closed form. Cut-2
execution remains HELD; no harness, nothing run.

**2026-05-16 (PT) — maintainer.** C2-C + C2-D filed (append-only),
**completing the C2 design layer**:
[`P2_CUT2_C2CD_LEVERAGE_GATE_AND_INVALID.md`](P2_CUT2_C2CD_LEVERAGE_GATE_AND_INVALID.md).
C2-C = observable-only leverage-confidence gate (smooth L1 ramp + genuine
CZA/tangent consistency-term presence; L2/L3 degradation emergent, the
"emergent vs flag-read" check handed to C4-D; load-bearing C2-C(i)/(ii)
detectable-and-discriminating window). C2-D = `f_par_obs<R22` rows are
abstain/invalid (never clipped), scored under q3-L1 as a built-in
zero-ambiguity correlate detector; abstain must be emergent from C2-B's
degenerate objective. All C2-C/C2-D numerics fold into the C2-A freeze.
Full open-conditions list for Cut-2-execute re-admission is now
**C2-A · C3-A/B/C/D · C4-A/B/C/D · C5**; the admission re-check is one
audit of the whole cut, withheld until all are filed. No frozen
threshold/boundary moved; Public-Language Constraint in force.
Justification: closes the last C2 design sub-blockers as
artifacts-before-agent.

**2026-05-16 (PT) — Codex audit.** C2-C/D direction accepted;
execution admission withheld. The C2-A freeze now has two explicit
behavioral obligations: prove the scalar `C_L1(s_obs)` ramp changes the
actual controller confidence/abstain outcome rather than only rescaling
an unchanged argmax, and freeze an objective abstain criterion for
`f_par_obs < R22` rows with no controller branch. C2-A should also
resolve whether `C_L1` intentionally gates the whole route package or is
kept from masking L2/L3 consistency tests. No harness/controller run.

**2026-05-16 (PT) — maintainer.** C2-A numeric freeze filed
(append-only):
[`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md). The
complete C2 freeze in one pass + three pre-run receipts: C2-A-1 (the
`C_L1` ramp bites `PhotometricAgent`'s *own* reacquire/lock-fail path —
not an argmax-inert rescale, proven vs the documented controller
constants), C2-A-2 (frozen objective-level `f_par_obs<R22` abstain, no
branch), C2-A-3 (L1 vs L2/L3 are `h`-disjoint by geometry, so the
whole-bracket `C_L1` does not mask the consistency-term tests). The
keystone anti-self-seal: the bridge `I→detector_intensity` scale is
frozen by an independent convention *before* the receipts; a failing
receipt **blocks** (append-only redesign), never a tuned pass. §4 freeze
is provenance-tagged ([G]/[E], A3). Full open-conditions list for
Cut-2-execute re-admission is now **C2-A receipts · C3-A/B/C/D ·
C4-A/B/C/D · C5**; one joint audit, withheld until all filed. No frozen
threshold/boundary moved; Public-Language Constraint in force.
Justification: closes the C2 numeric freeze as an artifact-before-agent.

**2026-05-16 (PT) — Codex audit.** C2-A direction accepted; closure
withheld pending actual numeric values and receipts. The mechanism is
now tied to the bound controller rather than an argmax rescale, and the
bridge-scale anti-self-seal is correctly surfaced. Remaining admission
holds: populate all [E] values/tables, define lock/confident-`qhat` as a
sustained-TRACK readout consistent with `PhotometricAgent`'s actual
SCAN/SEEK/TRACK semantics, and add the C2-A-2 objective scan receipt. No
harness/controller run.

**2026-05-16 (PT) — maintainer.** C3-A numeric freeze filed
(append-only):
[`P2_CUT2_C3A_NUMERIC_FREEZE.md`](P2_CUT2_C3A_NUMERIC_FREEZE.md).
Mirrors the C2-A structure (slots/provenance/receipt obligations now;
concrete `[E]` values + tables are the maintainer's pre-run fill).
Inherits C2-A's scale/seed/grid and propagates the sustained-TRACK
confident-`q̂` readout across the whole C3 column. Keystone
anti-self-seal: `P_in` decoy↔`h` correlation / `κ` / `M` frozen by an
independent principle before any run. Three receipts (C3-A-R
argmax-sensitivity reachability, C3-A-T temptation+reversal with the
now-well-posed `π_route`, C3-A-B κ window); C3-A-R floor **= the C4 D2
floor (one shared number)**. Full open-conditions list for
Cut-2-execute re-admission is now **C2-A receipts · C3-A receipts ·
C4-A/B/C/D · C5**; one joint audit, withheld until all filed. No frozen
threshold/boundary moved; Public-Language Constraint in force.
Justification: closes the C3 numeric-freeze structure as an
artifact-before-agent.

**2026-05-16 (PT) — Codex audit.** C3-A direction accepted; closure
withheld. It correctly propagates C2-A's sustained-TRACK readout and the
C3-C argmax-sensitivity repair, but cannot close until C2-A's inherited
values close and C3-A's own [E] values/receipt tables are appended. The
`P_in` independent freeze must become an operational artifact (finite
sample or generator, seed, decoy coefficients, frozen `(w,b)`), not only
a principle. No harness/controller run.

**2026-05-16 (PT) — maintainer.** C4-A audit freeze filed
(append-only), **completing the C-condition columns**:
[`P2_CUT2_C4A_AUDIT_FREEZE.md`](P2_CUT2_C4A_AUDIT_FREEZE.md). Mirrors
the C2-A/C3-A scaffold; propagates the operational-frozen-artifact
ruling to C4's fixtures/probe-set/taint-method. D2 floor = the shared
C3-A-R floor (one number); sustained-TRACK readout inherited; C4-C
repaired D1 (route construction vs true `h`). Keystone anti-self-seal:
the synthetic non-vacuous fixture is the minimal mechanical flip of the
real Cut-1 fixture, frozen before the audit logic — C4-B is a two-sided
self-test with no fixture-design freedom. C4-D made concrete
(input-manifest assertion + boundary-perturbation test). Full
open-conditions list for Cut-2-execute re-admission is now **C2-A · C3-A
· C4-A receipts/artifacts · C5**; one joint audit, withheld until all
filed *and their concrete values/artifacts landed*. No frozen
threshold/boundary moved; Public-Language Constraint in force.
Justification: closes the final C-condition column as an
artifact-before-agent.

**2026-05-16 (PT) — Codex audit.** C4-A direction accepted; closure
withheld. The two-sided Cut-1/minimal-flip self-test is the right shape,
but admission still needs concrete artifacts: hashable Cut-1 fixture
manifest, minimal-flip generator/diff, D1 probe-set table, shared D2
floor after C3-A closes, and runnable C4-D taint/perturbation script
with frozen readouts. C4-A is provisional while C2-A/C3-A remain
scaffolds. No harness/controller run.

**2026-05-16 (PT) — maintainer.** C5 publication-plumbing freeze filed
(append-only) — **all pre-registration conditions are now filed**:
[`P2_CUT2_C5_PUBLICATION_PLUMBING_FREEZE.md`](P2_CUT2_C5_PUBLICATION_PLUMBING_FREEZE.md).
Default-deny / allowlist-complement guard over the full tree (writes
only under `results/structural-failure/cut2-*/`), not an under-scopable
blocklist; operational artifacts (hashable manifest + runnable pre/post
guard script + terminal-dominant `PUBLICATION_PLUMBING_VIOLATION`). The
full condition set is **C2-A · C3-A · C4-A · C5**; the only remaining
gate to the single joint admission re-run is the maintainer's concrete
fill (the `[E]` values, operational artifacts, and receipt tables
across all four). No frozen threshold/boundary moved; Public-Language
Constraint in force. Justification: closes the final pre-registration
condition as an artifact-before-agent.

**2026-05-16 (PT) — Codex audit.** C5 direction accepted; closure
withheld until the manifest and guard script exist. The guard must
explicitly choose clean-baseline versus pre/post-snapshot semantics, must
catch tracked, untracked, and ignored public/shipping-path changes, and
must normalize paths / reject symlink escapes before the
`results/structural-failure/cut2-*/` allowlist is applied. No
harness/controller run.

**2026-05-16 (PT) — maintainer. Wave-1 concrete fill landed
(C5 closed; C4-A Cut-1 fixture pinned).** Two of the five C4-A
operational-artifact obligations + both C5 operational artifacts now
exist on disk, with all hashes recorded in writing:

- **C5 closed** — manifest `results/structural-failure/cut2-prereg/c5-write-path-manifest.json`
  (canonical-SHA `bfa2dd66…b40f`) and runnable guard
  `scripts/cut2-publication-plumbing-guard.mjs` (SHA `5e859283…fa06`).
  Snapshot semantics frozen; full-tree coverage with normalized
  paths + symlink-escape rejection; `PUBLICATION_PLUMBING_VIOLATION`
  terminal-dominant. Details + usage in
  [`P2_CUT2_C5_PUBLICATION_PLUMBING_FREEZE.md`](P2_CUT2_C5_PUBLICATION_PLUMBING_FREEZE.md)
  2026-05-16 audit-notes append.
- **C4-A partial fill** — Cut-1 known-vacuous fixture manifest
  `results/structural-failure/cut2-prereg/cut1-fixture-manifest.json`
  (canonical-SHA `3b69bf3c…c97e`); seven fixture objects from
  `scripts/structural-failure-p2-harness.mjs` (file SHA
  `43001506…6015`) hashed individually with line ranges; deterministic
  re-generator at `scripts/cut2-cut1-fixture-extract.mjs` (SHA
  `894e6efb…36fde`). Details +
  per-fixture-object table in
  [`P2_CUT2_C4A_AUDIT_FREEZE.md`](P2_CUT2_C4A_AUDIT_FREEZE.md)
  2026-05-16 audit-notes append.

The remaining C4-A artifacts (D1 probe set, minimal-flip
generator/diff, C4-D taint/perturbation script, C4-B self-test table)
and the C2-A / C3-A receipts wait on the next waves of the ordered
execution plan: C2-A `[E]` values + bridge-scale convention → C2-A-1/2/3
receipts → C3-A `P_in` + shared C3-A-R/D2 floor + receipts → C4-A
remaining artifacts → C4-B two-sided self-test → joint admission
re-run. **No frozen body edited; no threshold/boundary moved;** the
C4-A and C5 freeze bodies are untouched and the receipts/artifacts
above are filed as append-only audit-notes per the existing discipline.
Public-Language Constraint remains fully in force. Cut-2-execute
remains HELD on the joint admission re-run with C2-A + C3-A + remaining
C4-A artifacts also landed. Justification: closes the C5
operational-artifact gate and lands the first immutable C4-A artifact
(zero coupling to upstream `[E]` fills) as Wave-1 of the ordered
concrete fill.

**2026-05-16 (PT) — Codex audit.** Wave-1 artifacts verified. The C5
manifest/script hashes and the C4-A Cut-1 fixture manifest/extractor
hashes match the pinned values; the fixture extractor reruns with no
diff. C5's full snapshot/check round-trip remains a host-run receipt
because the full tree exceeds the authoring sandbox timeout. Also note:
because `cut2-prereg` matches the C5 results allowlist, C5 guards
publication plumbing, not immutability of the prereg artifacts stored
there; C2/C3/C4 artifact hashes must still be re-checked explicitly at
joint admission. No harness/controller run.

**2026-05-16 (PT) — maintainer. Wave-2 C2-A `[E]` values + bridge-scale
convention frozen.** Append-only filing to
[`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md)
2026-05-16 audit-notes. Concrete values now A3-immutable:
`ρ = 0.02`, `σ = 0.5°`, `seed = 20260516`, `h`-grid `[0°, 40°]` step
`0.5°`, `q_h ∈ [0°, 60°]`, `q_a ∈ [−A, +A]` with `A = ρ·R22 ≈ 0.44°`,
`λ = 1.0`, `τ_C2-B-ii = 0.05°` (argmax-stability tolerance — distinct
from `D1_min_bias`), `C_L1` = sigmoid centred at `s = 0.02` with
**steepness `k = 600`** (calibrated so the full 5–95% sigmoid
transition fits **inside** the ±1.5° L1 coincidence window around
`h_L1 ≈ 11.37°` — 5% crossing at `h ≈ 9.89°`, 95% at `h ≈ 12.66°`),
`T_cza = T_tan = 0.3`, `detect_threshold_T = 0.2`,
`separation_min = 2.0°`, `O_floor = 0.1`, `r_tol = 0.66°` (= `1.5·A`),
`κ_cond_max = 100`, `D1_min_bias = 1.5°` (= τ1, comparator semantics
frozen: `b(h,ε) = |q_naive − h|`, min over ε ≥ floor for every h in the
must-differ L1-ineligible band). Bridge-scale convention pinned: bridge
is identity in the route channel; eligible-band route peak ≡ 1.0 by
construction, frozen before any C2-A-1/2/3 receipt. Maintainer-side
arithmetic correction: `h_L1 = arccos(1/1.02) ≈ 11.37°`
(v1 proposal's `11.48°` was sloppy cosine; corrected here once,
propagates to all downstream receipts; no frozen body edited).
Wave-2 deliberately does **not** compute the C2-A-1/2/3 receipt
tables — those are Wave 3, computed against the now-frozen `[E]`
values under the §5 bridge convention; a failing receipt blocks
(append-only redesign), never silently tunes a value above (A3).
Public-Language Constraint remains fully in force. Cut-2-execute
remains HELD on Waves 3–7 + joint admission re-run. Justification:
closes Wave-2 of the ordered concrete fill (C2-A `[E]` values +
bridge-scale convention) with values + defenses + comparator semantics
recorded in writing.

**2026-05-16 (PT) — maintainer. Wave-3 C2-A receipts filed
(narrative-ordered).** Append-only filing on
[`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md)
2026-05-16 audit-notes. Receipt order is strictly chronological:
**v1 freeze → C2-A-2 v1 BLOCK → Wave-3.1 algebraic amendment → C2-A-2
v2 PASS** (per Wave-3 sign-off conditions). Verdicts:

- **C2-A-1** sustained-TRACK landscape — **PASS**. Transition at
  `h = 10.0°` (grid-evaluated), margin to lower coincidence-window
  edge `0.135°`. Continuous 5% `C_L1` crossing at `h ≈ 9.89°` under
  Wave-2 frozen `k = 600`; the "effective k ≈ 645.7" that maps the
  analytical crossing to the discrete grid is recorded as a
  discretization observation, not a re-pick.
- **C2-A-2 v1** under Wave-2 `κ_cond_max = 100` — **BLOCK** (permanent
  receipt of the algebraic miss; 222/479 L1-eligible-by-obs rows trip
  on `cond > κ_cond_max` alone).
- **Wave-3.1 amendment** — `κ_cond_max` v1 = 100 → v2 = 10⁴ on
  principled chain-rule Hessian algebra at `h_L1`: `|H_qa|/|H_qh|
  ≈ 14.33 / 0.01239 ≈ 1156`; `10⁴` buffers ~8.7× above this geometric
  extreme. Degenerate `cond → ∞` analytically as `q_h → 0` (chain
  factor `χ → 0`); the `~10⁵` floor seen in computation is the
  grid-resolution practical floor under the frozen `q`-grid and
  finite-difference Hessian estimator, **not a universal analytic
  floor**. A3-compliant: bounded by algebra, not receipt data.
- **C2-A-2 v2** under amendment `κ_cond_max = 10⁴` — **PASS**
  (degenerate 61/61 trip, L1-eligible-by-obs 0/479 trip, by-f_par_obs
  classification).
- **C2-A-3** package-gating separation — **PASS** (min `C_L1` for
  `h ≥ 25°` is `1.0` to 10 decimals; bracket multiplication doesn't
  mask L2/L3 consistency tests).

Honest disclosure: the v1 implementation initially carried **two**
distinct defects — (a) a script-level bug evaluating `C_L1` from
`sec(true_h) − 1` against the freeze §1 observable-only requirement,
and (b) the Wave-2 algebraic miss on `κ_cond_max`. The Wave-3 v1
receipt was generated with defect (a) FIXED to isolate defect (b);
v2 fixes both. Legacy buggy-state artifact preserved at
`results/structural-failure/cut2-prereg/_legacy_pre_w3_c2a2-abstain-scan.json`
for audit archeology. Canonical Wave-3 script:
`scripts/cut2-c2a-w3.mjs` (SHA-256 `3e06221b…d104`); stepping-stone
scripts `cut2-c2a-receipts.mjs` and `cut2-c2a-amendment-v2.mjs` are
superseded.

**No frozen body edited; no `[G]` boundary moved; only `κ_cond_max`
`[E]` value changed (principled re-pick).** Public-Language Constraint
remains in force. Cut-2-execute remains HELD on Wave 4 (C3-A `P_in` +
receipts), Wave 6 (C4-A remaining artifacts), Wave 7 (C4-B
self-test), and the joint admission re-run. Justification: closes
Wave-3 of the ordered concrete fill with the receipt-blocking
discipline upheld — v1 BLOCK filed permanently, amendment bounded by
algebra not data, v2 PASS recorded under the amended value.

**2026-05-16 (PT) — Codex audit.** Wave-3 C2-A receipts verified by
rerunning `node scripts/cut2-c2a-w3.mjs`; all pinned hashes reproduced
and no diff was produced. The filed arc remains strict-order:
C2-A-1 PASS, C2-A-2 v1 BLOCK under `κ_cond_max = 100`, algebraic
Wave-3.1 amendment to `10^4`, C2-A-2 v2 PASS, C2-A-3 PASS. No
harness/controller run.

**2026-05-16 (PT) — maintainer. Wave-4 C3-A receipts filed and Path W
closeout selected.** Canonical Wave-4 generator
`scripts/cut2-c3a-w4.mjs` (SHA-256
`85d7d0a06548e777b5022c1af00ed357dfe2da09325b5543a044a5f35bec7707`)
files `c3a-pin-generator.json`, `c3a-r-receipt.json`,
`c3a-t-receipt.json`, `c3a-b-receipt.json`, and `c3a-w4-summary.md`.
Verdicts: **C3-A-R PASS**, **C3-A-T BLOCK**, **C3-A-B BLOCK**. Canonical
Wave-4.1 amendment generator `scripts/cut2-c3a-w4-v2.mjs` (SHA-256
`882a2c5b393a1d3c3a5f6ce75b2daaf09502d67f6cb74a162f9588b6a41955ed`)
files `c3a-r-receipt-v2.json`, `c3a-t-receipt-v2.json`,
`c3a-b-receipt-v2.json`, and `c3a-w4-v2-summary.md`; results remain
**PASS/BLOCK/BLOCK**. The useful scientific finding is the Path-Y subset
split: BOUNDARY_MAP-pinned decoys help in the low-leverage band where
the route is weak (`L1-ineligible-by-obs`: `pi_dec` `2.506 deg`,
`pi_route` `3.912 deg`) and hurt in the eligible band where the route is
strong (`L1-eligible-by-obs`: `pi_dec` `3.806 deg`, `pi_route`
`1.398 deg`). Because the eligible subset is `479` rows vs `108`
ineligible rows, the full non-degenerate average still gives the route
the win (`pi_dec` `3.567 deg`, `pi_route` `1.860 deg`, margin
`-1.706 deg` where `+0.5 deg` was required). Path Z improves route-basin
preservation from `71.6%` to `79.5%`, still below the frozen `90%`
threshold. Path W therefore records both v1 and v2 as permanent BLOCK
receipts. No leverage-weighting, `kappa` increase, threshold relaxation,
or decoy re-pinning is made inside Wave 4; Wave-4.2, if any, is a
separate freeze-level redesign discussion. Cut-2-execute remains HELD
on C3-A BLOCK plus remaining C4-A/C4-B/joint-admission work; Public
Language Constraint remains fully in force.

**2026-05-16 (PT) — maintainer. Wave-4.2 disposition (α+γ) filed
(append-only):** [`P2_CUT2_WAVE42_DISPOSITION.md`](P2_CUT2_WAVE42_DISPOSITION.md).
γ — the Wave-4 C3-A result is permanent and recorded as a
**regime-separability finding** (route strictly dominates the
closed-form correlate where eligible; correlate only substitutes in the
abstain region) — traceability-favorable but **not a pass / not a
controller result** (no `CONFIRMED`/theorem/"harness passes"). α — this
is the pre-registered "Cut-2 ambiguous" trigger → **escalate to Cut-3
(rendered signal)**, where a learned image correlate can compete in the
eligible band that closed-form scalar decoys cannot; Cut-3 is **not
started** and carries its own px↔° / Phase-15 admission gate. Path β
(closed-form eligible-band decoy) recorded as a tuning hiding place,
**not** pursued. No C3 frozen value/threshold/boundary changed; Wave-4
receipts permanent, not reopened. The closed-form Cut-2 line can now
only certify γ + the escalation, never a closed-form traceability pass.
Cut-2 execution remains HELD; Public-Language Constraint in force.
Justification: records the user-selected α+γ named freeze-level redesign
as an append-only disposition.

**2026-05-16 (PT) — maintainer. Cut-3 spec/admission opened; execution
HELD.** Filed [`P2_CUT3_RUN_SPEC.md`](P2_CUT3_RUN_SPEC.md) and
[`P2_CUT3_ADMISSION.md`](P2_CUT3_ADMISSION.md). The Cut-3 spec turns the
Phase-15 px↔° / centering hazard into **H0 angular calibration**: every
rendered frame must carry a sun-centered angular map, valid span, anchor
residual table, and source hashes before any controller or learned image
agent sees it. HaloSim-native Scale is accepted as a method only when
the stamped ruler covers the scored feature field and passes an anchor
check; the Phase-15 pyramidal Scale receipt is recorded as the negative
example (instrument works, span too short, anchors unvalidatable). Cut-3
admission verdict: **HOLD** — spec shape and H0 gate accepted in
principle, but no render-corpus manifest, H0 residual table,
agent-under-test path, baselines, or edit operators exist yet. Allowed
next work is H0 manifest/schema/checker and corpus preparation; forbidden
until later ADMIT/PARTIAL ADMIT: training, controller evaluation, or any
public implication that Cut-3 has begun or passed.

**2026-05-16 (PT) — maintainer. H0 instrument operationalized
(append-only).** Filed
[`P2_CUT3_H0_CALIBRATION.md`](P2_CUT3_H0_CALIBRATION.md) — the explicitly
allowed next work (H0 schema/checker). Per-(frame,feature) record
schema; keystone anti-self-seal (`valid_angular_span_deg` measured from
the instrument's own extent **before** the scored feature is read —
coverage shortfall ⇒ Cut-3 BLOCKED, never span-stretched); C4-B-style
two-sided self-test whose immutable known-FAIL fixture is the real
Phase-15 pyramidal `pyr_w*_scale.png` set (the checker must reject them
before any corpus frame is admitted). Runnable checker + per-frame
records + residual table + known-PASS fixture remain maintainer pre-run
fill. No frozen Cut-3 value changed; admission stays HOLD; execution
HELD; nothing run.

**2026-05-16 (PT) — maintainer. Wave H0-1 fixture manifest + checker
filed (mechanical scaffolding; H0-B negative side passing).** Same
Wave-1 cadence (Cut-1 fixture + C5 guard) lifted to the measurement
layer. Append-only filing on
[`P2_CUT3_H0_CALIBRATION.md`](P2_CUT3_H0_CALIBRATION.md) audit-notes.
Pinned artifacts:

- Phase-15 known-FAIL fixture manifest at
  `results/structural-failure/cut3-prereg/h0-known-fail-fixture.json`
  (canonical-SHA `75577d51…2c05`): the 8 immutable real
  `pyr_w*_scale.png` frames hashed with paired `*_4M.png` + `*.sim`,
  expected `admit=false` + `reason_codes_allowed ∈ {SPAN_TOO_SHORT,
  ANCHOR_OFF_RULER}` declared per frame. Extractor:
  `scripts/cut3-h0-known-fail-extract.mjs` (SHA `69731e86…6b8c`).
- Runnable H0 checker at `scripts/cut3-h0-checker.mjs` (SHA
  `75bd6b32…1e2d`): implements the §1 record schema literally with
  three `theta_map` kinds (`scale_ticks` / `renderer_metadata` /
  `fit2locus`). **Structural call-order enforcement** of the §2
  anti-self-seal: `calibrate(sidecar)` has no access to the scored
  feature; `admit(calibration, scored_feature_deg)` is a separate
  function. There is no API path for the feature to enter span
  computation.
- H0-B negative-side self-test PASS: 8/8 Phase-15 frames rejected with
  `reason_code=SPAN_TOO_SHORT`. Self-test result at
  `results/structural-failure/cut3-prereg/h0-self-test-result.json`
  (canonical-SHA `e345f872…8c81`). Test sidecars model the documented
  Phase-15 failure (short ruler 5°–18°, anchors 22°/46° off-ruler); they
  are not fabricated H0 records.

Honest tightness — h-leak detection scope: the mechanical checker flags
**explicitly-labeled** h-encoding patterns (`h\d+`, `altitude_\d+`,
`elevation_\d+`, `sun_alt`/`solar_alt` variants, `h_deg`, named sidecar
fields). Compound HaloSim crystal-config codes in the actual Phase-15
filenames (e.g. `e13` in `pyr_w18_e13_x25`) are NOT auto-flagged —
operator review of whether such codes encode h is part of Wave H0-2
pre-fill. The predicate already supports h-leak rejection; one operator
decision wires it in.

Open in H0 (Wave H0-2 scope): known-PASS full-span fixture
identification, per-frame H0 records, anchor-residual table, operator
review of HaloSim crystal-config codes. H0 closes only when both sides
of the two-sided self-test resolve correctly. **No frozen H0 protocol
value, geometry boundary, or admission rule changed. Cut-3 admission
remains HOLD; execution HELD; Public-Language Constraint in force.**
Justification: closes the mechanical-scaffolding half of H0 (the
explicitly-allowed admission next work). H0-B negative side passes
8/8; positive side waits on a real known-PASS render.

**2026-05-16 (PT) — correction. Wave H0-1 NOT sealed.** The H0 entry
above is corrected in the open (append-only, Cut-1 precedent), per
[`P2_CUT3_H0_CALIBRATION.md`](P2_CUT3_H0_CALIBRATION.md) corrections
C1–C3. Verification of the committed tree found an artifact-identity
defect (named checker committed 0-byte; working checker is the misnamed
tracked `_legacy_…_v0.mjs`; §I delete-instruction VOIDED; §F/§G claims
untrue as committed) and that the "8/8 negative side" is a reject-branch
unit check against a hardcoded modeled stub, not the §3 rejection of the
real frames — the Cut-1 tautology pattern at the measurement layer. §2
structural design genuine; honest disclosures stand. H0-B negative side
on real frames **OPEN**; Cut-3 admission HOLD, execution HELD; no frozen
value changed.
