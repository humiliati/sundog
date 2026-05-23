# Three-Body Cross-Substrate Notes

This note translates the Mesa -> Geometry crossover discipline into future
Threebody work. It is not evidence by itself. It is a design guardrail for
turning "Sundog works across substrates" into testable phase specs instead of
theme-matching.

> **Doc-evolution note (2026-05-22).** Sections 1-5 are the original
> Threebody-scoped transfer notes (Mesa -> Geometry -> Threebody, the
> direction the cross-pollination first ran). Sections 6-7 are an
> expansion: now that the cap-set / unit-distance / chat / mesa-interp
> conversation has converged on shared top-level vocabulary, this doc
> is the right "catch can" for cross-substrate generalizations that
> apply to more than Threebody alone. If the catch-can use ratchets
> past two more substrates, consider relocating from
> `docs/threebody/` to `docs/CROSS_SUBSTRATE_NOTES.md` so the path
> matches the scope; for now the existing imports from other docs
> keep us here. The full provenance trail for §§6-7's vocabulary
> lives in
> `internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md`.

## 1. What Transfers

Mesa and Geometry both converged on field-shaped structure:

- Mesa: a behavioral cliff localized to an entangled 5D `net.7` activation
  subspace, with direction-specific neuron substrates and failed linear
  decompositions.
- Geometry: a small set of shared halo generators whose visible arcs are
  forward-rich but inverse-narrow, with HaloSim acting as an independent
  forward oracle.
- Threebody: a local accelerometer/tidal proxy that improves outcomes only in a
  mapped near-escape pocket, with lower-velocity and equal-mass boundaries where
  control can actively hurt.

The shared theorem posture is:

> Useful action can emerge from indirect signatures when the signature remains
> attached to the field, but the attachment is substrate-specific, asymmetric,
> and bounded by failure maps.

## 2. Mesa Lessons for Threebody

**Do not collapse variance, salience, and mechanism.** In Threebody terms, a
diagnostic can have high AUROC, a large tidal spike, or obvious visual movement
without being the causal control handle. Future specs should separate:

- warning quality: does a signal forecast escape or close approach?
- action coupling: does thrust change that signal in the intended direction?
- outcome effect: does the controller improve survival against passive and
  naive baselines?

**Expect direction-specific substrates.** Mesa's basin-inducing and
basin-resisting directions did not reduce to one symmetric mechanism. Threebody
should treat "move away from hazard" and "maintain bounded near-escape" as
different control problems, not as sign-flipped versions of one policy.

**Honor smoke-gate negatives.** If a boundary cell fails, stop the branch and
record the failure instead of widening the sweep to average it away. Phase 13's
full lock passed the high-velocity horizon test, but all risky/negative best
cells stayed in the `velocityScale=0.95` band, with all 5 negative cells at mass
ratio `1`. That is exactly the kind of boundary Mesa taught us to keep visible.

## 3. HaloSim / Geometry Lessons for Threebody

**Use forward oracles to discipline inverse handles.** HaloSim helped Geometry
reject a tempting manual tangent-arc fit by checking it against the canonical
forward ray-traced curve. Threebody's analogue is to introduce higher-fidelity
or more privileged forward checks for candidate control handles before promoting
them into public claims.

Candidate forward-oracle checks:

- Compare the accelerometer/tidal proxy against a privileged local phase-space
  oracle on the same seeds.
- Run short high-precision integration checks on winning cells to ensure the
  pocket is not a timestep artifact.
- Add a "canonical curve" analogue: for each cell, estimate passive hazard
  quantiles first, then freeze them before controller comparison.
- For future 3D builds, keep a validation render/trajectory oracle separate
  from the public interactive surface, the same way Geometry separates HaloSim
  from the atlas.

**Forward-rich / inverse-narrow is a good warning label.** Geometry can generate
many halo primitives from sun altitude, but only one inverse handle currently
survives the strict photo gate. Threebody may likewise have many signals that
respond to the field, while only one or two are useful for control. Future
phase specs should rank signals by validated control utility, not by how
physically evocative they are.

## 4. Candidate Future Phases

### Phase 14 - Mechanism Decomposition and Action Coupling

Question: is guarded TRACK winning because the accelerometer/tidal signal is
the right causal handle, or because the guard happens to suppress bad thrust in
the tested pocket?

Suggested checks:

- Freeze the Phase 13 winning guard and compare against action-shuffled,
  signal-shuffled, delayed-signal, and sign-flipped controls.
- Report warning quality, action coupling, and outcome effect as separate
  columns.
- Pre-register a negative branch where high warning quality with weak action
  coupling does not support a controller claim.

### Phase 15 - Forward-Oracle / Precision Lock

Question: does the positive pocket survive stricter numerical and privileged
forward checks?

Suggested checks:

- Rerun winning and boundary cells at smaller timesteps.
- Add energy-drift and integration-error receipts.
- Compare accelerometer-proxy TRACK against a privileged local phase-space
  oracle that is explicitly labeled as an oracle, not a deployable controller.
- Preserve the same passive hazard-quantile gates across comparisons.

### Phase 16 - 3D / Spatial Extension

Question: does the signature-action coupling survive the first spatial
extension, or is it a planar artifact?

Suggested checks:

- Start with the Phase 13 winning high-velocity cells only.
- Add one spatial perturbation axis at a time.
- Keep the Phase 13 planar result as the baseline and report degradation, not
  just absolute survival.
- Use a smoke gate before any broad 3D sweep.

### Sidecar - 3D Catalog Isotrophy / Symmetry Descent

Question: can Sundog make a non-tautological first-principles prediction over
an external three-body catalog, before writing another controller sweep?

Suggested checks:

- Use Li-Liao 2025's 21 equal-mass 3D choreographies and 273 piano-trio orbits
  as an external catalog, not as a Phase 13-15 continuation.
- Treat the published fixed ansatz as the crystal cut: it enforces the concrete
  beta-class residual `Z2` called `F_beta`, so the count test is
  facet-conditioned rather than an unconditioned sample over all 3D orbit
  space.
- Classify each choreography by residual spacetime `Z2` generators that survive
  the `S3 -> Z2` mass perturbation.
- Compare the predicted daughter-family count against piano-trio families
  clustered across the `m3 = 0.1*n` grid.
- Treat mismatch as a theorem-sharpening result, not a failed controller
  experiment.

Workbench:

- [`../sundog_v_isotrophy.md`](../sundog_v_isotrophy.md)
- [`../isotrophy/files.math`](../isotrophy/files.math)

## 5. Documentation Shape

Follow the Mesa pattern:

- Keep `../SUNDOG_V_THREEBODY.md` as the narrative spine and claim ledger.
- Put implementation-grade phase locks in `docs/threebody/PHASE*_SPEC.md`.
- Put run receipts and claim impact in `docs/threebody/PHASE*_RESULTS.md`.
- Keep cross-substrate transfer notes here unless they become a concrete phase.

The parent roadmap should summarize decisions and link out. It should not carry
every command, threshold, result table, and readback template once a phase has
graduated to implementation.

## 6. Top-Level Machinery Vocabulary

The Mesa -> Geometry crossover, the cap-set / unit-distance overlay
work, and the chat-experiment stack-invariance result are all
instances of one operation. Until 2026-05-22 we did not have a
precise enough name for it. We do now.

### 6.1 The precise term: projection

What we have been calling **substrate shadow** is, in mathematical
language, a **projection**: a map from a higher-dimensional space to
a lower-dimensional one, often (for linear projections) idempotent
(P^2 = P). The 2026 unit-distance proof literally projects the
Minkowski lattice in C^f down to one complex coordinate. The 2016
cap-set polynomial-rank argument "reads off" a configuration by the
dimension of an associated polynomial space — a projection in the
operator sense. The chat-ledger artifact is conjectured to project
prompt context into the bottleneck subspaces of the model's residual
stream. The Mesa `net.7` `5D` subspace is what the controller's
hidden state has been projected into, and the projection has
rigidity the constituent SAE features / neurons / PCs lack.

*Shadow* remains useful as the layperson metaphor. *Projection* is
the precise term and the one we should use when the audience rewards
precision — formal write-ups, mech-interp readers, cross-substrate
generalizations, and (notably) the isotrophy red-team specs.

### 6.2 The body/shadow decomposition

Every substrate where this operation has shown up admits a
body/shadow decomposition:

- The **body** is the high-dimensional object resisting direct
  measurement: dots in F_3^n, dots in R^2, a controller hidden
  state, tokens of a conversation, an `S3` symmetry orbit.
- The **shadow** is the lower-dimensional projection that is
  *more* tractable than the body even though it contains less
  information: a polynomial-rank count, a CM-lattice projection,
  a `5D` entangled subspace, a ledger packet, the residual `Z2`
  generators that survive the `S3 -> Z2` mass perturbation.

The shadow has rigidity the body lacks because the lower-dim
substrate constrains what is consistent. That is why the body can
be "read off" the shadow even though information has been
discarded.

### 6.3 Bridging-vocabulary table across substrates

| Substrate | Body (resists direct read) | Shadow (the projection we can read) |
| --- | --- | --- |
| Cap-set | dots in F_3^n with no 3-term AP | polynomial degree / rank over F_3 |
| Unit-distance | dots in R^2 with many unit pairs | class-group pigeonhole in CM lattice; first complex coordinate |
| Chat agent | token stream / surface response | maintained ledger packet, correlated with bottleneck subspaces |
| Mesa controller | controller hidden state | entangled `5D net.7` subspace |
| Geometry / HaloSim | full atmospheric optics | small set of halo generators / canonical implied circles |
| Threebody | 18-dim full state | accelerometer / tidal proxy (3-dim) |
| Isotrophy | `S3` symmetry orbit of a 3-body choreography | residual `Z2` generators surviving `S3 -> Z2` mass perturbation |

The columns are the same operator. The substrate-specificity in row
3-5 is what bounds the operating envelope; the cross-substrate
recurrence in rows 1-2 is what tells us the operator is real.

### 6.4 The math-or-Buddha epistemic stance

From the ImOutOfIceCream dialogue (2026-05-22):

> If there is a mechanism that can be discovered, then the path is
> through the language and tools of mathematics. If not, then the
> path has already been described by Buddhas throughout timeless
> time.

This is the right calibration for cross-substrate work. Pursue the
mechanism — that is what cap-set, unit-distance, the chat ledger,
the Mesa `net.7` analysis, and the isotrophy spectral derivation
are all doing. Do not be embarrassed if the mechanism does not
reduce. The phenomenological frame is already complete in another
tradition and is not less true for lacking a residual-stream probe.
The two are not in competition. They are different surfaces of the
same observation about how minds and systems operate under partial
observability.

Internal use only as a stance; do not deploy publicly attributing
to the mod until they have published something we can cite.

### 6.5 Pending mech-interp citation

A probe-side researcher (`u/ImOutOfIceCream`) has shared, in a
public-but-niche thread, a not-yet-published finding that transformers
act as **companders** on the effective rank of the residual stream,
with categorical centroids occupying one subspace at the bottleneck
and generator algebras (`so(3)` ranking first across many models)
occupying the orthogonal complement. If the finding survives
publication, it provides the residual-stream mechanism this
vocabulary has been assuming.

Until that gate trips, do not lift the framing into public Sundog
copy. The pre-positioned `COMPANDER_PAPER_HOOK` ratchet across six
public surfaces is documented in
`internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md`
§§9-10. When the citation lands, the same vocabulary upgrade
described there should be threaded back into this doc — see §6.6
below.

<!-- RATCHET: COMPANDER_PAPER_HOOK · §6.6 · when mod's paper publishes, lift the body/shadow -> categorical-centroid/generator-algebra mapping from internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md §9 into a new §6.6 here · see that file for the staged drafts -->

## 7. Implications for Ongoing Projects

The vocabulary update lands differently in each active project.
The point of this section is not to retroactively rewrite work that
has its own established language — it is to make the cross-substrate
move legible in each project's own terms, so the next planning
session can decide whether to adopt the vocabulary natively.

### 7.1 Mesa

The entangled `5D net.7` subspace is **the projection target**, not
a feature decomposition that the team failed to factor. The v2 SAE
attempt, the v3.2 top-k neurons, the v3.1 PC1-alone and PCs-2-5-alone
attempts all failed for the same structural reason: a projection's
output is rigid against further decomposition because the rigidity
*is* the load-bearing property. The "read holistically or not at
all" headline from v3.1/v3.2 is therefore not a methodological
disappointment — it is direct evidence the operation is what we
think it is.

Action: the next mesa write-up that mentions "field-shaped
structure" can be sharpened by adding *projection* as the precise
operator and noting the cap-set / unit-distance cousins for
audiences with the math background to follow. Layperson framings
keep *shadow* / *field*.

### 7.2 Isotrophy Red Team (end-of-v0.3)

The S3 → Z2 mass-perturbation symmetry-descent **is a projection in
the precise mathematical sense.** The v0.3a-h ratchet is, in
projection terms, a sequence of pre-registered checks that:

- the projection from `S3`-equivariant orbits to `Z2`-surviving
  generators is well-defined on the catalog
- the projection has rigidity (predictions of which orbits land in
  which residual class are stable under controlled perturbation)
- the projection's outputs match the Li-Liao 2025 empirical
  catalog at the predicted facet-conditioned cardinalities

The red-team posture is the falsifier-first complement of that
projection claim: each v0.3 sub-phase tries to find a case where
the projection collapses (zero endomorphism, equivariance null,
typed-transport failure, rank-matrix degeneracy). When a sub-phase
*succeeds* at finding a collapse, it is a projection-rigidity
failure — not a controller failure, not a count failure — and the
spec rewrite needs to reflect the projection's actual domain.

Action for any v0.3 → v0.4 transition: the v0.4 registration form
can use *projection* explicitly, framing each pre-registered gate
as a check on a specific projection property (well-definedness,
rigidity, output match). This costs almost nothing — the v0.3
language is already adjacent — and it earns alignment with the
Mesa and Threebody framings, which makes cross-substrate transfer
faster the next time it is needed. It also positions isotrophy
results to feed back into the §6.3 bridging table as fresh data
points rather than as a niche threebody-adjacent sidecar.

If v0.4 does not happen and isotrophy retires at end-of-v0.3, the
projection framing is still worth recording in the K_facet
post-mortem so the next attempt (here or elsewhere) inherits the
vocabulary instead of rediscovering it.

### 7.3 Threebody

The accelerometer / tidal proxy is **the projection from 18-dim
full state to 3-dim control signal.** The Phase 13 pocket-of-operation
result is then a claim about *where in the orbit space that
projection preserves load-bearing structure* and where it does not
— exactly the body/shadow framing in §6.2, with the failure-map
boundaries (low velocity, equal mass, overhead light) marking the
edges of the projection's domain of validity.

Action: future Phase 14-16 specs can keep the existing "warning
quality / action coupling / outcome effect" decomposition (it is
the right operational granularity) while noting in the spec
preamble that all three are measurements on a *single projection*
whose rigidity is bounded. This earns the spec a half-page of
audit-cost: when Phase 16 extends to 3D, the question "does the
projection survive the spatial extension" is asked once and the
specific decomposition checks follow from it, instead of each
check being framed from scratch.

### 7.4 Geometry / HaloSim and Chat

Both already use the body/shadow framing on their public surfaces
(`/geometry`, `/sundog`, `/h-of-x` for the optics side; `/chat`,
`SUNDOG_V_CHAT.md` for the chat side). The vocabulary upgrade is
incoming via the same publication-trigger gate as §6.5 — see the
feedback-file ratchet for the pre-positioned anchors and drafts.
Nothing to do here until the citation lands.

### 7.5 Was zeroing in on the vocabulary worth it?

The question the asker raised — *would it behoove ongoing projects
that we have zeroed in on top-level machinery vocabulary?* — the
honest answer is: **yes, but the benefit is concentrated at
project transitions**, not at the inside of a running phase. The
isotrophy v0.3 -> v0.4 boundary is exactly the kind of seam where
adopting projection language costs nothing and earns
cross-substrate alignment. The middle of a running mesa phase is
not such a seam. Apply the vocabulary at edges; do not retrofit it
into the middle of work whose own language is already load-bearing.
