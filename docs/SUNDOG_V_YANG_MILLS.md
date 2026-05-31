# Sundog vs. Yang-Mills

> **Cross-substrate failure-map entry:** BOUNDED NULL (the gauge-invariant
> shadow carries no separating structure on the SU(2) 3D cell) — see
> [`threebody/CROSS_SUBSTRATE_NOTES.md`](threebody/CROSS_SUBSTRATE_NOTES.md)
> "Cross-Substrate Generality Failure Map".

Working hook:

> Do not reconstruct the gauge field. Certify what survives the gauge.

Short version:

> Yang-Mills is the natural gauge-theory stress test for Sundog's signature
> discipline: if the field itself is too much, can the right invariant shadow
> still certify bounded structure?

Status: **Phase 1 instrumentation closed across the full ladder
2026-05-29; Phase 2 v0 / v1 / v2 / v3 all executed ->
`YM-P2-NEG-A no_rank_local_structure`; bounded-null synthesis filed
2026-05-29; Phase 2 v4 powered-target reopen executed 2026-05-31 ->
`YM-P2-UNDERPOWERED no_powered_target_in_envelope`; Phase 5
external-review packet + email cover-letter draft filed 2026-05-29
(owner-pending send to a lattice gauge theorist).**
Roadmap draft at this file; lit-pass at
[`YANG_MILLS_LITPASS_MEMO.md`](YANG_MILLS_LITPASS_MEMO.md); P0 lock at
[`prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md);
Phase 1 U(1) 2D smoke manifest at
[`prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md`](prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md)
with `P1-A smoke_pass` receipt at
[`yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md);
Phase 1 SU(2) 2D harness manifest at
[`prereg/yang-mills/PHASE1_SU2_2D_gauge_invariance_smoke.md`](prereg/yang-mills/PHASE1_SU2_2D_gauge_invariance_smoke.md)
with `P1-A smoke_pass` receipt at
[`yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md);
Phase 1 SU(2) 3D primary-cell manifest at
[`prereg/yang-mills/PHASE1_SU2_3D_gauge_invariance_smoke.md`](prereg/yang-mills/PHASE1_SU2_3D_gauge_invariance_smoke.md)
with `P1-A smoke_pass` receipt at
[`yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md);
Phase 2 v0 relative-locality spec at
[`prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md`](prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md)
with named-null receipt at
[`yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md);
v1 smearing probe spec at
[`yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md`](yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md);
P0 amendment 1 (APE smearing, vocab v4, frozen Î±=0.5 / N_sm=10) at
[`prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md);
Phase 2 v1 binding spec (reuses v0 ensembles, smeared signature) at
[`prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md`](prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md)
with named-null receipt at
[`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md);
v2 correlator probe spec at
[`yang-mills/specs/2026-05-29_phase2_v2_correlator_probe.md`](yang-mills/specs/2026-05-29_phase2_v2_correlator_probe.md);
Phase 2 v2 binding spec (vocab v5 = 20-dim connected correlators on
v0 ensembles, no smearing, no P0 amendment) at
[`prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md`](prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md)
with named-null receipt at
[`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md);
v3 target-redesign probe spec at
[`yang-mills/specs/2026-05-29_phase2_v3_target_redesign_probe.md`](yang-mills/specs/2026-05-29_phase2_v3_target_redesign_probe.md);
Phase 2 v3 binding spec (vocab v1 signature unchanged + new vocab v2
sigma2_W33 target, no P0 amendment) at
[`prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md`](prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md)
with named-null receipt at
[`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md);
bounded-null synthesis receipt at
[`yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md);
v4 powered-target probe spec at
[`yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md`](yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md);
Phase 2 v4 binding spec (power-gate-first target audit, no P0 amendment)
at
[`prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md`](prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md)
with underpowered-envelope receipt at
[`yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md`](yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md);
Phase 5 external-review packet at
[`yang-mills/EXTERNAL_REVIEW_PACKET.md`](yang-mills/EXTERNAL_REVIEW_PACKET.md)
with email cover-letter draft at
[`yang-mills/EXTERNAL_REVIEW_EMAIL_DRAFT.md`](yang-mills/EXTERNAL_REVIEW_EMAIL_DRAFT.md).
The Phase 1 receipts are instrumentation only. Phase 2 v0 / v1 / v2 / v3 all
landed `YM-P2-NEG-A`; v4 landed `YM-P2-UNDERPOWERED`, a quarantine-class
target-power result, not a fifth null. No continuum, confinement, mass-gap,
or result-bearing Yang-Mills claim exists. The lane is paused unless a dated
v5 P0 amendment selects a powered regime before code.

This is not a claim to progress on the Clay Yang-Mills existence and mass gap
problem. It is a plan for a bounded receipt program that asks whether Sundog's
signature/locality discipline can be applied honestly to gauge-invariant
finite-lattice data without pretending to solve the continuum problem.

Lit-pass disposition:

- The official Jaffe-Witten / Clay problem remains the boundary object, not the
  claim object.
- Recent serious activity is concentrated around finite-lattice loop
  observables, area-law regimes, large-N / t'Hooft regimes, loop equations,
  two-dimensional continuum limits, three-dimensional stochastic
  quantisation, bootstrap methods, and gauge-equivariant ML.
- Recent claimed Clay-solution artifacts exist and are explicitly quarantined
  in the lit-pass memo. They do not support any Sundog claim.
- Sundog's possible niche is narrow: a pre-registered, lossy,
  gauge-invariant signature that preserves a held-out finite-lattice
  observable/regime label in rank space beyond controls.

## 0. Why ARC Changes The Approach

ARC Phase 3 taught the wrong first move:

```text
exact reconstruction first -> sparse fibers, exact-match floors, brittle claims
```

The useful ARC lesson is narrower and better:

```text
certificate/locality first -> ask whether the representation preserves
bounded structure better than controls, even when exact fibers are empty
```

For Yang-Mills, that means the first question is not:

> Can Sundog reconstruct a gauge field, prove a mass gap, or solve the
> continuum theory?

The first question is:

> On a finite lattice with a frozen ensemble and gauge-invariant observables,
> does a compact invariant signature preserve local physical/regime structure
> better than metadata, raw-field, random, and gauge-variant controls?

This places the lane beside ARC's relative-locality certificate and
Navier-Stokes C1's finite-cell discipline. It treats the Clay problem as a
directional stress test, not as a claim target.

## 1. Claim Boundary

This roadmap explicitly does **not** claim:

- a construction of four-dimensional quantum Yang-Mills theory;
- a proof of existence;
- a proof of a positive mass gap;
- a continuum-limit theorem;
- a confinement proof;
- a novel lattice gauge algorithm;
- a replacement for established lattice gauge theory.

What it may eventually claim, if earned by receipts:

- a bounded finite-lattice certificate result;
- a gauge-invariant signature that preserves a pre-registered observable or
  regime label better than controls;
- a named null showing that the proposed shadow is too coarse, too nonlocal,
  or too contaminated by metadata/finite-size artifacts;
- a handoff path for external review by someone who knows lattice gauge theory.

The public sentence, if this is ever surfaced:

> Sundog's Yang-Mills lane is a finite-lattice gauge-invariant certificate
> test, not a Clay-problem claim.

## 2. Why Yang-Mills Fits Sundog

Yang-Mills is a hard stress test for the same themes the repo already cares
about:

- **Gauge redundancy.** Many field descriptions represent the same physical
  state. Sundog already tries to avoid mistaking coordinates for structure.
- **Invariant shadows.** Wilson loops, plaquette traces, correlation functions,
  and related observables are closer to Sundog's "shadow" language than raw
  gauge potentials are.
- **Local-to-global tension.** Confinement and mass-gap proxies are not obvious
  from a single local readout, but finite-lattice observables can still test
  whether local invariant summaries preserve useful structure.
- **Certificate before solver.** The question can be framed as verification of
  bounded properties, not generation of configurations or continuum proof.

The closest Sundog anchors are:

- [`SUNDOG_V_ARC.md`](SUNDOG_V_ARC.md): relative-locality after exact
  reconstruction fails.
- [`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md): finite numerical cell
  before continuum ambition.
- [`faraday/SUNDOG_V_FARADAY.md`](faraday/SUNDOG_V_FARADAY.md): local
  gauge-invariant readout as a bounded apparatus.
- [`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md):
  sufficient-statistic discipline and falsification gates.

### Three-Body Transfer Rules

The internal three-body lane gives this roadmap its clearest operating
discipline. Phase 13 did not establish "the accelerometer proxy works" in the
abstract; it mapped a high-velocity near-escape pocket where the proxy helped
and a low-velocity/equal-mass boundary where it could hurt. Yang-Mills should
inherit that posture:

```text
do not ask whether the invariant shadow works;
ask where, for which cell, against which controls, and where it fails.
```

Concrete transfer rules:

1. **Projection audit before result.** Three-body separated compressed
   full-state diagnostics from sensor-available signals. Yang-Mills must
   separate gauge-invariant signatures from target observables copied into the
   signature. "Gauge invariant" is necessary but not sufficient.
2. **Cell map before mean score.** Report results over
   `group x dimension x lattice size x beta x loop-set`, not as one aggregate
   success. Boundary cells are evidence, not inconvenience.
3. **Decompose the mechanism.** Three-body Phase 14 split warning quality,
   action coupling, and outcome effect. Yang-Mills should split invariance
   quality, held-out-observable coupling, and certificate effect.
4. **Use privileged oracles honestly.** Three-body used oracle and
   precision-lock checks without calling them deployable controllers.
   Yang-Mills may use exact tiny fixtures, abelian/2D sanity cells, known loop
   identities, and bootstrap-style checks as forward oracles, but the primary
   claim remains about the compact invariant signature.
5. **Hard-void gates before long reads.** If a gauge-randomization,
   target-leak, metadata-only, or finite-size split gate fails, the phase is
   void or narrowed before any favorable narrative is interpreted.

The transferable claim shape is:

> A projection can preserve load-bearing structure inside a bounded operating
> envelope; the envelope is part of the result.

## 3. Honest vs. Reach

Honest:

- A finite-lattice, pre-registered certificate lane.
- Gauge-invariant signatures only; no gauge-fixed shortcut is allowed to carry
  the primary claim.
- Relative-locality before fixed-radius fiber claims.
- Controls against metadata, raw-field leakage, random neighbors, and
  gauge-variant encodings.
- Named nulls and design holds as first-class outcomes.

Reach; do not claim:

- "Sundog solves Yang-Mills."
- "Sundog proves confinement."
- "Sundog proves the mass gap."
- "The signature replaces lattice gauge theory."
- "Finite-lattice correlations imply the continuum theorem."
- "A local invariant vector is sufficient for all gauge-theory structure."

## 4. Candidate Finite-Lattice Domain

Phase 0 must choose this exactly. Until then the following is a draft menu, not
a spec.

Candidate primary cell:

```text
theory: pure SU(2) lattice gauge theory
dimension: 2D or 3D first for harness sanity; 4D only after smoke
lattice sizes: small fixed grids, e.g. 8^d / 12^d / 16^d
boundary: periodic
ensemble source: generated locally or imported from a documented baseline
coupling grid: frozen beta/coupling slate
updates: heatbath / overrelaxation / documented library path
```

Candidate abelian/control cell:

```text
U(1) lattice gauge toy or deliberately soluble gauge toy
```

The abelian/toy cell is not a Yang-Mills result. It is only an instrumentation
and leakage-control baseline.

The first handoff should prefer a known, boring finite-lattice setup over a
clever new simulator. If no local library path is available, Phase 0 should
stage the exact external dependency decision before any run.

## 5. Signature Candidates

Primary signatures must be gauge invariant by construction:

- plaquette trace summaries across positions and scales;
- Wilson loop values for a frozen set of rectangular loops;
- Creutz-ratio-style summaries as diagnostics, not proof claims;
- Polyakov-loop-style observables when the finite setup makes them meaningful;
- gauge-invariant two-point or loop-loop correlator summaries;
- smearing/blocking-level summaries, with smearing parameters frozen before
  reading results;
- optional topological-charge proxies only if the finite setup and discretized
  definition are fixed in Phase 0.

Forbidden as primary signatures:

- raw gauge links without quotienting or invariance checks;
- gauge-fixed potentials unless used only as a diagnostic/control;
- metadata-only labels such as lattice size or beta;
- target observables copied into the signature under another name;
- post-result loop-set expansion.

## 6. First Certificate Question

The ARC-informed first certificate should be relative, not absolute:

```text
Do nearest neighbors under the gauge-invariant signature preserve
pre-registered observable/regime labels better than controls?
```

Candidate labels:

- coupling/regime bin, if not leaked trivially through metadata;
- Wilson-loop area-law proxy class inside the finite ensemble;
- correlation-length or exponential-decay proxy class;
- low/high action-density event class;
- same/different ensemble identity under held-out beta/lattice-size splits;
- finite-volume artifact class, as a negative-control label.

Required controls:

1. metadata-only nearest neighbors;
2. raw-link or gauge-fixed representation, diagnostic only;
3. random neighbors;
4. coupling-stratified random neighbors;
5. gauge-randomized copies to prove the primary signature is invariant;
6. target-label permutations over the frozen neighbor graph.

Positive result language, if earned:

> The finite-lattice invariant signature has rank-local structure for the
> registered observable labels beyond controls.

Negative result language:

> The proposed invariant shadow does not preserve usable rank-local structure
> for this finite-lattice target.

## 7. Falsification Surface

Named failures take precedence over narrative success.

1. **Gauge leakage.** The primary result changes under random gauge
   transformations, or a gauge-variant control carries the signal while the
   invariant signature does not.
2. **Metadata shortcut.** Metadata-only neighbors match or beat the invariant
   signature.
3. **Target leakage.** The target observable is directly or nearly directly
   embedded in the signature.
4. **Finite-size artifact.** The signal disappears or reverses under the
   registered lattice-size split.
5. **Coupling-label triviality.** The classifier/neighbor result only recovers
   beta/coupling without preserving the physical observable of interest.
6. **Sparse-fiber non-result.** Absolute fibers are empty or uninformative;
   this is not a failure if the relative-locality lane was the registered
   primary. It is a failure only if the artifact claims fixed-radius sufficiency.
7. **Continuum overreach.** Any finite-lattice result is used to imply the
   continuum Clay statement. This quarantines the public surface.

## 8. Phase Plan

### Phase 0 - Lit-Pass And Domain Lock

Deliverables:

- `docs/YANG_MILLS_LITPASS_MEMO.md` - **filed 2026-05-29**;
- `docs/prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md` - **filed
  2026-05-29**;
- chosen finite-lattice domain, observable labels, source library, and compute
  budget;
- explicit "not a Clay claim" language;
- first external-review target category.

Stop rule:

Do not write runner code until the finite domain and controls are frozen.

Lit-pass recommendation:

```text
default first domain: small pure SU(2) Wilson-action lattice, 2D/3D before 4D
default toy: U(1) or 2D cell for instrumentation only
default primary claim: rank-local structure for held-out gauge-invariant
observable labels beyond controls
```

The first domain should be intentionally boring. The hard part is not novelty
of the simulator; it is avoiding leakage, target-copying, and continuum
overreach.

### Phase 1 - Gauge-Invariance Smoke

Question:

```text
Are the proposed signatures invariant under random gauge transformations?
```

Deliverables:

- `docs/prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md` -
  **filed 2026-05-29**;
- `docs/prereg/yang-mills/PHASE1_SU2_2D_gauge_invariance_smoke.md` -
  **filed 2026-05-29**;
- `docs/prereg/yang-mills/PHASE1_SU2_3D_gauge_invariance_smoke.md` -
  **filed 2026-05-29**;
- tiny deterministic fixture;
- raw-link diagnostic control;
- gauge-randomized copies;
- receipts showing primary signatures stable and raw/control lanes unstable as
  expected - **filed 2026-05-29** at
  [`yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md)
  and
  [`yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md)
  and
  [`yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md).

Failure:

`YM-P1-NEG-A gauge_leakage`.

### Phase 2 - Relative-Locality Certificate

Question:

```text
Does the invariant signature preserve registered observable labels in
nearest-neighbor rank space beyond controls?
```

Deliverables:

- frozen loop/plaquette/correlator signature;
- nearest-neighbor graph;
- metadata/raw/random/permutation controls;
- branch table over positive / metadata-only / negative / inconclusive.

v0 execution:

- `PHASE2_SU2_3D_relative_locality_v0.md` filed 2026-05-29;
- four locked invocations executed 2026-05-29 on `SU2_3D`, 12x12x12,
  beta slate `{2.0, 2.4, 2.8}`;
- receipt filed at
  [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md);
- verdict `YM-P2-NEG-A no_rank_local_structure`: within-beta
  bin-purity@5 = 0.3104, below the registered 0.5 gate, and only 0.0104
  above `CTRL_RAND`.

v1 execution:

- `PHASE2_SU2_3D_relative_locality_v1.md` filed 2026-05-29 after the
  smearing probe spec and P0 amendment 1;
- one locked aggregation invocation executed 2026-05-29 on the same v0
  ensembles, with APE-smeared vocab v4 signature `(alpha, N_sm) = (0.5, 10)`;
- receipt filed at
  [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md);
- verdict `YM-P2-NEG-A no_rank_local_structure`: within-beta
  bin-purity@5 = 0.29375, below the registered 0.5 gate and below
  `CTRL_RAND` by 0.00208, while smearing health and gauge-invariance gates
  passed.

v2 execution:

- `PHASE2_SU2_3D_relative_locality_v2.md` filed 2026-05-29 after the
  connected-correlator probe spec;
- one locked aggregation invocation executed 2026-05-29 on the same v0
  ensembles, with bare-link vocab v5 signature (20 connected 2-point
  correlators over `{W11, W12, W13, W22}` and five frozen cubic
  displacement classes);
- receipt filed at
  [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md);
- verdict `YM-P2-NEG-A no_rank_local_structure`: within-beta
  bin-purity@5 = 0.308333333333, below the registered 0.5 gate, only
  0.020833333333 above `CTRL_RAND`, and below `CTRL_META`; bin-edge
  replay and gauge-randomization integrity passed.

v3 execution:

- `PHASE2_SU2_3D_relative_locality_v3.md` filed 2026-05-29 after the
  target-redesign probe spec;
- one locked aggregation invocation executed 2026-05-29 on the same v0
  ensembles, with unchanged v1 signature re-read from v0 CSVs and new
  held-out target `sigma2_W33`;
- receipt filed at
  [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md);
- verdict `YM-P2-NEG-A no_rank_local_structure`: within-beta
  bin-purity@5 = 0.329166666667, below the registered 0.5 gate and only
  0.027083333333 above `CTRL_RAND`; signature-hash, target-spread,
  edge-timing, and gauge-randomization gates passed.

bounded-null synthesis:

- four consistent Phase 2 named nulls now span three signature vocabs and
  two target classes;
- synthesis receipt filed at
  [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md);
- disposition: PAUSE-and-synthesize, not automatic v4 probe continuation.

v4 execution:

- reopened only after the dated powered-target diagnostic spec and binding
  spec filed 2026-05-31;
- one locked aggregation invocation executed 2026-05-31 on the same v0
  ensembles, with Stage 1 auditing `mean_W14`, `mean_W23`, `sigma2_W14`,
  `sigma2_W23`, and `ratio_W23_W14` for split-half power and signature
  disjointness before any rank-locality score;
- receipt filed at
  [`yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md`](yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md);
- verdict `YM-P2-UNDERPOWERED no_powered_target_in_envelope`: no candidate
  was both powered and disjoint across all three beta values. `mean_W14`
  came closest but missed the beta 2.0 ICC gate and failed the beta 2.8
  leakage gate; `gamma_held` failed the power self-validation as required.
  Stage 2 rank-locality scoring was not run.

Failure / quarantine:

`YM-P2-NEG-A no_rank_local_structure`, `YM-P2-NEG-B metadata_only`, or
`YM-P2-UNDERPOWERED no_powered_target_in_envelope` when no valid target is
admitted for scoring.

### Phase 3 - Observable-Certificate Gate

Question:

```text
Can the signature certify a bounded finite-lattice observable class without
full field reconstruction?
```

Candidate target:

- correlation-length proxy class;
- Wilson-loop area-law proxy class;
- finite ensemble same/different label under held-out splits.

Deliverable:

- verifier-style certificate with source binding and spoof controls, borrowing
  discipline from [`SUNDOG_V_P_V_NP.md`](SUNDOG_V_P_V_NP.md).

Failure:

`YM-P3-NEG-A certificate_spoof`, `YM-P3-NEG-B verifier_cost`, or
`YM-P3-NEG-C target_vacuity`.

### Phase 4 - Scale / Coupling Generality

Question:

```text
Does the certificate survive a pre-registered change in lattice size or
coupling regime?
```

This is the Yang-Mills equivalent of the Navier-Stokes C1 regime-generality
test. It should run only after Phase 2 or Phase 3 earns a non-vacuous result.

Failure:

`YM-P4-NEG-A local_window_only` or `YM-P4-DEFERRED_FINITE_SIZE`.

### Phase 5 - External Review Packet

Deliverables:

- synthesis note;
- minimal reviewer packet;
- exact "what to check / what not to check" questions;
- public-language boundary.

No public page should promote this lane before this review packet exists.

## 9. SVG / Data Handoff

Candidate SVGs if this joins `generality.html`:

- **Gauge-Invariant Shadow Pipeline:** raw links -> invariant loops/plaquettes
  -> signature -> nearest-neighbor certificate -> report.
- **Relative-Locality Matrix:** primary signature vs metadata/raw/random/
  permutation controls.
- **Claim Boundary Ladder:** finite-lattice certificate -> finite-size
  generality -> external review -> public surface; continuum Clay claim is
  explicitly outside the ladder.

Candidate data shape:

```json
{
  "id": "yang_mills",
  "status": "draft-handoff",
  "claimBoundary": "finite-lattice certificate only",
  "primaryQuestion": "rank-local gauge-invariant structure beyond controls",
  "forbiddenClaims": [
    "solves Yang-Mills",
    "proves mass gap",
    "proves confinement"
  ]
}
```

## 10. Handoff Checklist

Next agent should:

1. ~~Promote `docs/prereg/yang-mills/README.md` into a frozen Phase 0 domain
   lock only after the exact finite-lattice cell is chosen.~~
   **Done 2026-05-29** â€” see [`prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md).
2. Use [`YANG_MILLS_LITPASS_MEMO.md`](YANG_MILLS_LITPASS_MEMO.md) as the
   citation spine; re-audit it before any public-facing claim or after six
   months.
3. ~~Choose the first finite-lattice domain and source library.~~
   **Done 2026-05-29** â€” U(1) 2D â†’ SU(2) 2D â†’ SU(2) 3D ladder, in-repo
   Creutz / Kennedy-Pendleton / Brown-Woch generator; see P0 lock.
4. ~~Freeze the observable labels and controls before code.~~
   **Done 2026-05-29** â€” held-out larger-Wilson-loop area-law proxy class
   (Î³_held tertile bin); seven-entry leakage controls battery; see P0 lock.
5. Stage, not run, any compute expected to exceed the repo's ten-minute rule.
6. Keep every output under `results/yang-mills/<phase>/` and every durable
   conclusion as a dated receipt under `docs/yang-mills/receipts/`.
   **Stood up 2026-05-29** â€” see [`yang-mills/receipts/README.md`](yang-mills/receipts/README.md).
7. Add Yang-Mills to the generality gallery only as a draft-handoff card until
   Phase 0 is filed. **Phase 0 now filed**; gallery card admission is the next
   adjacent decision (no public surface yet).
8. ~~File a Phase 1 runner manifest at
   `docs/prereg/yang-mills/PHASE1_<cell>_<short-label>.md` filling every
   Admission Requirement in the P0 lock.~~
   **Done 2026-05-29** - see
   [`prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md`](prereg/yang-mills/PHASE1_U1_2D_gauge_invariance_smoke.md).
9. ~~Implement only the minimal U(1) 2D Phase 1 runner and package script
   required by the manifest.~~
   **Done 2026-05-29** - receipt filed at
   [`yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_U1_2D_phase1_gauge_invariance_smoke_pos.md).
10. ~~File a `SU2_2D` Phase 1 manifest, then implement the minimal SU(2)
    2D Phase 1 runner and package script required by that manifest.~~
    **Done 2026-05-29** - receipt filed at
    [`yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_SU2_2D_phase1_gauge_invariance_smoke_pos.md).
11. ~~File a `SU2_3D` Phase 1 gauge-invariance smoke manifest, then
    implement the minimal SU(2) 3D Phase 1 runner and package script
    required by that manifest.~~
    **Done 2026-05-29** - receipt filed at
    [`yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase1_gauge_invariance_smoke_pos.md).
12. ~~File and execute `PHASE2_SU2_3D_relative_locality_v0.md` on the 12^3
    partner, freeze numeric gamma_held bin edges before scoring, and score all
    seven P0 controls on the same frozen neighbor graph and held-out labels.~~
    **Done 2026-05-29** - named-null receipt filed at
    [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md).
13. ~~File a dated v1 probe spec under `docs/yang-mills/specs/`, then the
    matching P0 amendment and Phase 2 v1 binding spec.~~
    **Done 2026-05-29** - see
    [`yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md`](yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md),
    [`prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md), and
    [`prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md`](prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md).
14. ~~Implement and execute the Phase 2 v1 APE-smearing aggregation runner.~~
    **Done 2026-05-29** - named-null receipt filed at
    [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md).
15. ~~File a dated v2 probe spec, Phase 2 v2 binding spec, and connected-
    correlator aggregation runner; execute the single v2 invocation.~~
    **Done 2026-05-29** - named-null receipt filed at
    [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md).
16. ~~File a dated v3 target-redesign probe spec, Phase 2 v3 binding spec,
    and v3 aggregation runner; execute the single v3 invocation.~~
    **Done 2026-05-29** - named-null receipt filed at
    [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md).
17. ~~Apply the v3 probe spec's default PAUSE-and-synthesize fallback if v3
    also lands `YM-P2-NEG-A`.~~
    **Done 2026-05-29** - bounded-null synthesis receipt filed at
    [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md).
18. ~~Apply the standing no-automatic-v4 guardrail unless fresh external
    scientific motivation appears.~~
    **Done 2026-05-31** - the powered-target diagnostic supplied the dated
    reopen, v4 was filed and executed, and the receipt landed
    `YM-P2-UNDERPOWERED no_powered_target_in_envelope` at
    [`yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md`](yang-mills/receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md).
19. **Standing guardrail (2026-05-31):** no automatic v5 runner. Continuing
    past the underpowered-envelope receipt requires a dated P0 amendment to a
    powered regime, with the beta slate / volume / target-class change stated
    before code.

## 11. Open Decisions

All five originally-open decisions were closed by the P0 lock filed
2026-05-29. See [`prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
Â§ "Open Decisions Closed By This Lock" for the resolution table.

Resolutions, restated here for navigation:

1. ~~Start with SU(2) only, or include an abelian toy control first?~~ â†’
   Staged ladder: `U1_2D` instrumentation, `SU2_2D` harness, `SU2_3D`
   primary; 4D deferred.
2. ~~Which library/generator is acceptable for finite-lattice ensembles?~~ â†’
   In-repo Creutz heatbath + Kennedy-Pendleton + Brown-Woch overrelaxation
   for SU(2); staple-based Metropolis for U(1).
3. ~~Which observable label is least likely to be a metadata shortcut?~~ â†’
   Larger-Wilson-loop area-law proxy class, Î³_held tertile bin; signature
   vocabulary strictly disjoint from target vocabulary.
4. ~~Should Phase 2 use fixed loop sets only, or include smearing/blocking
   levels?~~ â†’ Fixed loop set only at P0; smearing/blocking deferred to a
   later P0 amendment.
5. ~~Who is the right external reviewer category~~ â†’ Lattice gauge theorist.

Decisions still open (not Â§11 originals; surfaced by the P0 lock):

- future target/signature probes after the Phase 2 bounded-null synthesis,
  admitted only with fresh external scientific motivation;
- gallery-card admission timing â€” currently still blocked per Â§10 item 7.

## 12. Ratified Public Boundary

Allowed:

> Sundog is drafting a finite-lattice Yang-Mills certificate lane. It asks
> whether gauge-invariant shadows preserve bounded structure beyond controls.

Forbidden:

> Sundog has a Yang-Mills result.

> Sundog is approaching the Clay problem directly.

> Sundog found a mass gap.
