# Sundog vs. ARC-AGI Abstraction

Working hook:

> ARC-AGI tests whether a system can abstract and reason on novel tasks
> without reward hacking or memorization. Sundog's shadow-coupling
> machinery — gauge-invariant local projections, Blackwell-sufficient
> signatures, 5D mesa subspace — is a geometric alternative to
> high-dimensional pattern matching. This roadmap finds out whether that
> machinery earns its keep on a public benchmark.

This roadmap is filed as Gravity Ledger Candidate 14 and promoted
directly into a committed execution plan targeting the ARC Prize 2026
Paper Track (Theory category). It couples the existing shadow-projection,
signature-sufficiency, and mesa-subspace machinery to the ARC-AGI task
family — the first Sundog roadmap with an external submission deadline
and a public benchmark deliverable alongside the internal receipt chain.

The pairing of theory paper and Kaggle entry is deliberate. A paper
without an entry can be dismissed as arm-waving. An entry without a
paper can be dismissed as architecture search. Run together, the
geometric argument (why shadow coupling should help with abstraction)
and the benchmark artifact (does it actually help) each discipline the
other.

The theory-side differentiator is the "why" — Blackwell sufficiency of
the signature for core-knowledge abstraction tasks, with the 5D mesa
subspace as the dimensionality-reduction mechanism. This is not a new
architecture paper; it is a structural argument about what makes
abstraction work, tested against the benchmark that was designed to
measure it.

**Official ARC Prize 2026 anchors (checked 2026-05-28):** competition
submissions are due November 2, 2026; papers are due November 8, 2026;
Paper Prize submissions must link to a working Kaggle code submission
for ARC-AGI-2 or ARC-AGI-3; the static ARC-AGI-2 track scores exact
task matches with two predictions per test input and no internet during
Kaggle evaluation. The first Sundog lane targets ARC-AGI-2 because its
static grid format is the cleanest bridge to shadow-projected
signatures. ARC-AGI-3 is out of scope unless a later amendment opens an
interactive-agent branch.

> **Gravity Ledger dependency.** This roadmap inherits directly from
> the mesa lane's operating-envelope result (Phases 0–7, especially
> v3.1–v3.8 5D subspace finding) and the geometry lane's forward-rich /
> inverse-narrow field-shape pattern. The shadow-projection operator
> (Phase 2 below) ports the same gauge-cocycle / Procrustes machinery
> used in the three-body workbench and photometric alignment. No new
> physics or simulator infrastructure is required.

## Why ARC Is the Right Next Coupling

The gravity claim names three falsification modes:

1. field manipulation is cheap;
2. the signature is a reward in costume;
3. mesa-optimization re-emerges.

Mesa (`SUNDOG_V_MESA.md`) attacks modes (2) and (3) in a synthetic
environment. The geometry lane attacks the forward/inverse field-shape
pattern in the wild. ARC-AGI attacks a fourth surface the other roadmaps
do not reach:

4. **abstraction failure** — shadow coupling produces competent navigation
   in continuous-field environments but does not generalize to the
   discrete, novel-task abstraction regime that core knowledge priors
   are supposed to enable.

ARC-AGI is purpose-built to test exactly this: can a system solve tasks
it has never seen, using core knowledge (objectness, counting, symmetry,
geometry) rather than pattern-matching over a training distribution? If
the shadow-signature machinery genuinely captures something about
abstraction — if the Blackwell-sufficient statistic carries the right
information — it should show up here. If it does not, the gravity claim
is bounded to continuous-field navigation and the "gravity for agents"
hook requires a named retreat.

ARC is also the most externally legible venue for the Sundog program. The
Paper Track Theory category rewards exactly the kind of structural
argument Sundog already makes — "why does abstraction work?" — and the
receipt-heavy, pre-registered, falsification-first methodology maps
directly onto what the prize committee is looking for.

## What's Honest vs. What's Reach

**Honest:**

- A clean algebraic demonstration that the Sundog signature is Blackwell
  sufficient for a registered, falsifiable subset of ARC-AGI tasks
  emphasizing core knowledge and spatial abstraction.
- Empirical confirmation that the effective representation collapses
  toward a low-dimensional invariant structure (analogous to the 5D mesa
  result) under shadow coupling on those tasks.
- A measured boundary where performance collapses — tasks requiring
  information outside the signature's capacity — with named failure
  modes.
- A working Kaggle entry (even modest-score) demonstrating the operator,
  not as a competitive leaderboard bid, but as an existence proof.

**Reach (do not claim):**

- "Sundog solves ARC-AGI."
- "Shadow coupling achieves human-level abstraction."
- "The signature replaces learned representations for novel tasks."
- "This approach scales to the 85% regime without additional machinery."
- "Low-dimensional structured fields are sufficient for all core
  knowledge priors."

The danger of this work is the same as the mesa roadmap's: a weak
benchmark result could be misread as a falsification of the entire
gravity program. It is not. A structural argument about *why* shadow
coupling helps on a characterized task subset — with an explicit boundary
where it doesn't — is itself a novel, publishable result in the Theory
category. The roadmap is structured so that either outcome ratchets the
program forward.

## Ratified Hook Language

Safe hook:

> Sundog vs. ARC-AGI asks whether shadow-coupled agents, using local
> gauge-invariant projections and Blackwell-sufficient signatures, can
> solve novel abstraction tasks without full state reconstruction — and
> whether the effective representation collapses toward the known 5D mesa
> subspace structure under this coupling.

Short version:

> The field knows enough. The agent doesn't need to reconstruct
> everything — just the signature.

Avoid:

- "Sundog solves ARC."
- "Shadow coupling is all you need for abstraction."
- "Our approach beats transformers / foundation models on ARC."
- "The 5D subspace is universal for core knowledge."
- "Geometric methods are superior to learned representations."

## Claim Boundary

This document does **not** claim that Sundog has demonstrated novel-task
abstraction, competitive ARC-AGI performance, or that shadow coupling
replaces learned representations. It claims that:

1. there is a coherent structural argument — signature as
   Blackwell-sufficient statistic under gauge-invariant shadow
   projection — for why shadow-coupled agents should generalize on
   core-knowledge abstraction tasks without reward hacking or full state
   reconstruction;
2. that argument is currently defended by the mesa lane's 5D subspace
   finding and the geometry lane's forward-rich / inverse-narrow
   field-shape pattern, not by any abstraction benchmark result;
3. the proof targets that would test the argument (ARC-AGI task subset,
   Blackwell sufficiency audit, dimensionality collapse check) are
   concrete enough to live in a roadmap rather than a ledger.

If the Phase 3 Blackwell sufficiency audit fails — if the signature is
not sufficient for the registered task class — the honest response is a
named quarantine on "signature sufficiency for discrete abstraction," not
a program retreat. The mesa and geometry receipts stand independently.

## Pre-Registered Hypothesis (v0.1)

In the registered class of ARC-AGI tasks (core knowledge / abstraction
without heavy reward signal), a shadow-coupled agent using:

- Sundog-style local shadow projections (gauge-invariant,
  Procrustes-aligned),
- signature as Blackwell sufficient statistic, and
- the known 5D mesa subspace structure

achieves non-trivial abstraction performance on held-out tasks with the
signature alone being sufficient (no need for full state reconstruction).
Performance degrades predictably outside the registered domain (e.g.,
when signature capacity is exceeded).

## Falsification Surface

The ARC coupling claim can fail in five named modes:

1. **Representation vacuity** — the shadow projection, applied to ARC
   grid representations, produces features that are either (a) what any
   convolutional encoder would produce, or (b) so abstract that no
   Sundog-specific discipline is doing the work. The gauge invariance
   has no edge over a good embedding layer.

2. **Sufficiency failure** — the signature is not Blackwell sufficient
   for the registered task class. Core-knowledge tasks require mutual
   information that the signature discards. The dimensionality reduction
   is lossy in a load-bearing way.

3. **Subspace non-collapse** — the effective representation under shadow
   coupling does not collapse toward a low-dimensional invariant
   structure. The 5D mesa pattern is environment-specific (continuous
   fields) and does not transfer to discrete grids.

4. **Boundary absence** — the agent's performance does not degrade
   predictably at the registered boundary. Either it fails everywhere
   (no signal) or it succeeds everywhere (no falsifiable boundary), and
   neither outcome sharpens the gravity claim.

5. **Benchmark disconnect** — the Kaggle entry score is so low that the
   paper's structural argument cannot be grounded in any empirical
   artifact, and the "even modest" framing reads as excuse-making.

Modes (1) and (2) are the most informative failures — they would say
something specific about the limits of the shadow-coupling approach.
Mode (5) is the most embarrassing but least informative. Mode (3) would
be the most consequential for the broader Sundog program because it
would bound the mesa subspace finding to continuous domains.

## Starting Assets

These exist and are portable:

- **Shadow projection + Faraday clean zero receipt** — the gauge-
  invariant local-projection operator and its zero-residual baseline
  from the three-body workbench.
- **Gauge cocycle / Procrustes machinery** — the alignment operator that
  makes shadow projections comparable across reference frames. Requires
  adaptation from continuous 2D fields to discrete grids.
- **5D mesa localization proof + signature sufficiency postulates** —
  the v3.1–v3.8 finding that the basin-attractor mechanism lives in an
  entangled 5D subspace at `net.7`. The sufficiency postulates are
  stated but untested outside the mesa environment family.
- **Three-body invariants and isotrophy operators** — the existing
  invariant-subspace machinery ports cleanly as a toy-model comparison
  or warmup.
- **Three-gate failure taxonomy** — residual / coverage / detection
  gates for classifying route failures. Directly applicable to the
  falsification battery (Phase 5).
- **Pre-registration infrastructure** — the `docs/prereg/` ladder
  discipline and append-only methodology.

## Roadmap

### Phase 0 — Registered Task Subset Definition

Goal: curate a falsifiable, pre-registered subset of ARC-AGI tasks that
emphasizes core knowledge and spatial abstraction, and pin the exclusion
criteria before any operator design.

Deliverables:

- A registered task subset drawn from the public ARC-AGI training and
  evaluation splits, with explicit inclusion criteria (core knowledge
  priors exercised: objectness, counting, symmetry, spatial reasoning,
  geometric transformation) and exclusion criteria (tasks requiring
  extensive sequential reasoning, language-like pattern matching, or
  information-theoretic capacity exceeding a stated bound).
- Pre-registration document in `docs/prereg/arc/` following the existing
  append-only ladder discipline.
- Baseline performance numbers: random, brute-force enumeration, and a
  simple CNN/transformer reference on the registered subset.
- Task taxonomy mapping each included task to the core-knowledge prior(s)
  it exercises, so Phase 3 can audit sufficiency per prior.

Execution artifact:
[`docs/prereg/arc/PHASE0_TASK_SUBSET_SPEC.md`](prereg/arc/PHASE0_TASK_SUBSET_SPEC.md).
Phase 0 receipt:
[`docs/prereg/arc/P0_BASELINES.md`](prereg/arc/P0_BASELINES.md) -- **ADMIT**
as of 2026-05-28 after baseline expansion: 36-task public-training subset
registered, leak-control passed, and all preregistered cheap baselines scored
0/36 exact. Phase 1 is admitted with `0/36` exact as the floor any
Sundog-specific result must clear.

Exit criterion: the task subset, exclusion criteria, baseline numbers,
and public-evaluation leak-control policy are filed and pre-registered.
No operator design has begun.

### Phase 1 — ARC Grid Representation as Shadow Domain

Goal: define the formal correspondence between ARC grids and the
shadow-field domain, establishing the gauge-invariant projection operator
for discrete grid representations.

Deliverables:

- Formal definition of the ARC grid as a structured field: cell values as
  field intensities, spatial layout as the manifold, color channels as
  fiber degrees of freedom.
- The local shadow-projection operator adapted from continuous 2D fields
  to discrete grids: probe geometry, neighborhood structure,
  gauge-covariance under grid symmetries (rotation, reflection,
  translation, color permutation).
- Procrustes alignment operator for comparing shadow projections across
  input-output pairs within a single ARC task.
- Validation that the projection operator recovers known grid invariants
  (e.g., translation invariance, rotation equivariance) on synthetic test
  grids before touching ARC tasks.

Exit criterion: the projection operator is defined, implemented, and
validated on synthetic grids. The gauge-covariance properties are
algebraically verified.

Execution artifact:
[`docs/prereg/arc/PHASE1_SHADOW_DOMAIN_SPEC.md`](prereg/arc/PHASE1_SHADOW_DOMAIN_SPEC.md).
Status (2026-05-28): synthetic validation passed on translation, rotation,
reflection, color-role permutation, and a shape-mismatch negative; Phase 2
projection scaffold admitted, with no sufficiency or Kaggle claim.

### Phase 2 — Shadow Projection Operator on ARC Tasks

Goal: apply the shadow-projection operator to the registered ARC task
subset and produce the raw projected representations.

Deliverables:

- Shadow projections computed for every task in the registered subset
  (both training demonstrations and test inputs).
- Per-task closure residuals: how much information the projection retains
  vs. discards, measured against the full grid representation.
- Initial signal-to-noise characterization: on which tasks does the
  shadow projection concentrate signal (low residual, structured
  projection), and on which does it disperse (high residual, noisy
  projection)?
- Comparison against naive baselines (raw pixel features, simple
  convolution features) on the same information-retention metric.

Exit criterion: projections computed, residuals measured, initial signal
characterization filed. No sufficiency claim yet.

Execution artifact:
[`docs/prereg/arc/PHASE2_PROJECTION_SPEC.md`](prereg/arc/PHASE2_PROJECTION_SPEC.md).
Status (2026-05-28): projection measurement passed on the registered
public-training subset (`36` tasks, `266` grids). Mean signature-collision
residual `0.028571`; mean train-pair residual `0.594295`; signal labels
`20` dispersed / `9` mixed / `7` compact. Phase 3 sufficiency-spec writing is
admitted; no decoder, sufficiency, public-evaluation, or Kaggle claim is
admitted.

### Phase 3 — Signature Sufficiency Audit

Goal: formalize and test Blackwell sufficiency of the signature for the
registered task class.

Deliverables:

- Algebraic formulation: under what conditions is the shadow-projected
  signature a Blackwell-sufficient statistic for the input→output mapping
  in the registered task class? State the theorem or conjecture with
  explicit assumptions.
- Empirical test: train a simple decoder (linear or shallow MLP) from the
  signature alone to predict the correct output grid. Compare against a
  decoder from the full grid representation. If the signature decoder
  matches or exceeds the full-grid decoder on the registered subset, the
  sufficiency claim is supported.
- Per-task and per-core-knowledge-prior breakdown: which priors does the
  signature capture (objectness? symmetry? counting?) and which does it
  miss?
- Named quarantines: tasks or priors where sufficiency fails, with
  explicit failure-mode classification (residual / coverage / detection
  gate, per the three-gate taxonomy).

Exit criterion: the sufficiency claim is either (a) supported on the
registered subset with measured residuals and named quarantines outside,
or (b) cleanly falsified with specific failure modes named. Either
outcome is publishable.

Execution artifact:
[`docs/prereg/arc/PHASE3_SUFFICIENCY_SPEC.md`](prereg/arc/PHASE3_SUFFICIENCY_SPEC.md).
Status (2026-05-28): spec filed; decoder-admission roadmap filed; Pass A
representation/decoder contract filed; Pass B split/floor/discrimination
contract filed; Pass C learner/metric contract filed; Pass D receipt/command
contract filed; freeze-marker runner wiring implemented. Three deterministic
low-capacity receipts are filed and all are verdict task-hardness /
decoder-failure, not support or failure of signature sufficiency. Phase 2
baseline comparison sharpened the problem: `P_shadow_grid_v0` preserves
distinctness but makes input-output pairs farther apart than raw-pixel and
coarse-feature baselines. Phase 3 therefore tested learnability of the
`input_rep -> output_rep` mapping from demonstrations, with
`signature_palette` as the primary representation and `signature_only` as a
stricter ablation. A Blackwell sufficiency lane is now filed: algebraic
conditions, held-out-task splits, Branch A/B/C criteria, a frozen Python/PyTorch
decoder, timing probe, and full clean receipt. The `blackwell_task_decoder_v1`
receipt is Branch C bounded failure: `signature_palette` scored zero exact
matches on both held-out lanes. The matched full-grid control also scored zero
exact matches, so the receipt does not support sufficiency and should not be
described as a full-grid superiority result. Subsequent full-grid controls did
not open the comparison arena: V2 (`blackwell_publictrain_rawgrid_gate_v2`,
freeze marker `79C5B060`) used the public-training auxiliary pool and filed
`full_grid_control_floor`, with zero exact matches on both held-out lanes; the
compact-7 diagnostic (`50EAEBBF`) narrowed to the Phase 2 compact-signal tasks
and filed `compact_full_grid_control_floor`, also with zero exact matches.
Compact-7 is qualitatively distinct because shape exact reached `1.000` while
palette exact remained `0.000`; per-instance residuals show dominant-background
predictions with at most two slot-1 colors even when targets use 3--9 colors.
That failure is named **dominant-color mode collapse**. Branch A then filed
`per_task_coord_mlp_v1`, a stochastic per-instance coordinate MLP trained from
scratch on each held-out instance's conditioning demonstrations. Its 20-shard
binding receipt returned **`branch_a_full_grid_floor`**: `raw_grid_per_task`
scored zero exact tasks on both held-out lanes, so no
`signature_palette_per_task` vs. `raw_grid_per_task` sufficiency comparison is
licensed. The failure character is **conditioning starvation +
shape-generalisation failure**, distinct from V1/V2's noise-dominated failure
and compact-7's dominant-color mode collapse. Four full-grid-control receipts
(V1, V2, compact-7, Phase 3A) now agree on the held-out exact-grid floor across
two task distributions and two learner families. Branches A, B, and C are closed
in their filed learner families; Branch D, a different framing such as structured
edit or residual prediction, is the only remaining admissible Phase 3 reopen
path. That path is now started as `structured_edit_residual_v1`: model the
output as an input-derived baseline canvas plus a residual edit mask and edit
colors. Its 20-shard binding receipt returned
**`branch_d_full_grid_edit_floor`**: `raw_grid_edit` scored zero non-baseline
exact tasks on both held-out lanes, so no `signature_palette_edit` vs.
`raw_grid_edit` sufficiency comparison is licensed. The named failure character
is **edit-color-rule failure**: the structured-edit framing found nontrivial
shape/canvas and edit-mask signal, but the per-task edit-color learner did not
recover exact output colors. Five Phase 3 full-grid controls now agree on the
held-out exact-grid floor across two task distributions, two learner families,
and two output framings. The first bottleneck-targeted Branch D variant,
`structured_edit_color_rule_v2`, kept the baseline and edit-mask components
fixed and replaced only the edit-color MLP with a deterministic
conditioning-derived color-rule bank. Its binding receipt returned
**`branch_d_color_rule_full_grid_floor`**: the raw-grid color-rule arena did
not open, so no signature-vs-full-grid sufficiency comparison is licensed. Six
Phase 3 full-grid controls now agree on the held-out exact-grid floor. The
variant did, however, shift the bottleneck into measured slices: 41%
edit-mask failure, 16% color-rule-selection failure, and 9% rule-bank-coverage
failure. Future Phase 3 reopens now require a new pre-registered
mask-targeted Branch D variant, selection-refinement Branch D variant,
rule-bank extension Branch D variant, or Branch E spec. The mask-targeted
variant is now filed as `structured_edit_mask_target_v3`: keep the baseline
picker and deterministic edit-color rule bank fixed, and replace only the
edit-mask predictor with a conditioning-derived mask-candidate bank. Its
binding receipt returned **`branch_d_mask_target_full_grid_floor`**: the mask
repair did not lift the floor, mask-stage labels still dominated, and the
legacy learned mask candidate still won a non-trivial share of selections.
Seven Phase 3 full-grid controls now agree on the floor. With both named
structured-edit bottlenecks probed by deterministic banks and both floored,
Branch E is the live frontier. The first Phase 3E spec is now filed as
`PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md`: before training another solver,
test whether registered ARC contexts contain exact or near
`signature_palette` context-fiber collisions with incompatible required
behavior. No Phase 3E receipt exists yet.

### Phase 4 — 5D / Low-Dimensional Collapse Check

Goal: measure whether shadow coupling induces collapse toward a
low-dimensional invariant structure analogous to the mesa 5D result.

Deliverables:

- Dimensionality analysis of the shadow-projected representation space:
  PCA / SVD spectrum, effective dimensionality measures, comparison
  against the full grid representation's dimensionality.
- If collapse occurs: characterize the invariant subspace. Is it 5D or
  nearby? Does it share structural properties with the mesa `net.7`
  subspace (entangled, resists factorization)?
- If collapse does not occur: characterize why. Is the ARC grid domain
  too heterogeneous? Is the registered subset too broad? Is the
  projection operator under-constrained?
- Cross-reference with Phase 3: do the tasks where sufficiency holds
  coincide with the tasks where dimensionality collapse occurs?

Exit criterion: the dimensionality question is answered with measured
spectra and explicit comparison to the mesa result. The answer feeds the
paper's central structural claim.

### Phase 5 — Falsification Battery

Goal: run explicit falsifiers to characterize the boundary where shadow
coupling fails.

Deliverables:

- **Non-local probes**: tasks requiring global information that local
  shadow projections cannot capture. Measure performance degradation and
  classify the failure mode.
- **Capacity-exceeding tasks**: tasks where the information-theoretic
  content of the correct output exceeds the signature's capacity (as
  measured in Phase 3). Confirm that performance degrades predictably.
- **Gauge-breaking perturbations**: tasks where the grid symmetries the
  projection operator exploits are violated or misleading. Measure
  whether the operator fails gracefully or catastrophically.
- **Comparison with full-state baselines**: on the falsification tasks,
  does a full-state decoder succeed where the signature decoder fails?
  This confirms the failure is signature-specific, not task-impossibility.

Exit criterion: a characterized boundary with named failure modes,
publishable as the falsification receipt in the paper.

### Phase 6 — Kaggle Submission + Paper

Goal: produce the ARC Prize 2026 Paper Track submission and a linked
Kaggle entry.

Deliverables:

- **Paper** (Theory category): structured around the "why" — Blackwell
  sufficiency of the signature under gauge-invariant shadow coupling,
  with the 5D mesa subspace as the dimensionality-reduction mechanism.
  Sections: motivation (Goodhart sidestep applied to abstraction), formal
  setup (shadow projection, signature, gauge covariance), sufficiency
  result (Phase 3 receipt), dimensionality result (Phase 4 receipt),
  falsification boundary (Phase 5 receipt), connection to the broader
  Sundog program (mesa + geometry two-substrate convergence).
  Pre-registered hypothesis stated and adjudicated.
- **Kaggle entry**: a working submission demonstrating the
  shadow-projection operator on the ARC-AGI evaluation set. Score is
  secondary to existence — the entry grounds the paper's structural
  argument in a runnable artifact.
- **Pre-registration adjudication**: the Phase 0 pre-registered
  hypothesis is explicitly adjudicated — confirmed, partially confirmed
  with named quarantines, or falsified with named failure modes.

Exit criterion: paper submitted, Kaggle entry linked, pre-registration
adjudicated.

## Promotion Criteria

This roadmap is self-contained. If Phase 3 produces a clean sufficiency
result and Phase 4 confirms dimensionality collapse, the ARC coupling
earns its own receipt in the Gravity Ledger's two-substrate convergence
narrative (third substrate: discrete abstraction). If Phase 3 fails, the
named quarantine is filed in the Gravity Ledger and the Capset Ledger's
Front-A evaluator machinery is the natural fallback coupling surface.

## Timeline and External Constraints

The ARC Prize 2026 Paper Track has a submission deadline. All phases are
sequenced to deliver a submittable paper and linked Kaggle entry by that
deadline. Phase 0 should be fast (task curation, not operator design).
Phases 1–2 are the implementation core. Phases 3–4 are the measurement
core. Phase 5 is the falsification discipline. Phase 6 is assembly.

If the timeline compresses, Phase 5 can be scoped down to the two most
informative falsifier classes (non-local probes and capacity-exceeding
tasks) without compromising the paper's structural argument. Phase 4 can
report preliminary dimensionality results if the full spectrum analysis
is not complete. Phase 3 is non-negotiable — the sufficiency claim is the
paper's differentiator.
