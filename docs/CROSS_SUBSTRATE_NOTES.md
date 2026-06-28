# Cross-Substrate Notes

This note translates the Mesa -> Geometry crossover discipline into the rest of
the Sundog portfolio. It started Threebody-scoped (§§1-5); §§6-8 are the
portfolio-wide expansion. It is not evidence by itself. It is a design guardrail
for turning "Sundog works across substrates" into testable phase specs instead of
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

> **Relocation note (2026-05-31).** The catch-can has now ratcheted past the
> threshold flagged above: the Faraday / Maxwell framing synthesis (§8) is the
> substrate that tipped it, so this doc has moved to its predicted top-level home
> `docs/CROSS_SUBSTRATE_NOTES.md`. A compatibility pointer stub remains at the old
> `docs/threebody/CROSS_SUBSTRATE_NOTES.md` (project precedent: the Faraday README
> pointer pattern), so historical inbound links keep resolving. Internal relative
> links were rewritten one directory up; the bidirectional lane-ledger backlinks
> now point at the new path. The `§6.5 -> §6.6` compander ratchet anchor moved
> with the file; its path string in
> `internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md` was updated to
> match.

## Cross-Substrate Generality Failure Map

> The promoted companion to §1 "What Transfers": **where the projection /
> signature-control operator hit a boundary.** Collecting the failures by
> *mode* is the design guardrail this whole doc exists for - "Sundog works
> across substrates" is only as honest as its catalogued failure boundaries.
> Each entry is its own failure mode; the recurring lesson (§6.3) is that a
> sharp control-regime-2 needs a **body that genuinely resists its shadow**,
> and the bullets below are the distinct ways that requirement can fail.
> Bidirectional: each lane ledger links back here.

- **Navier-Stokes C1 - MARGINAL (near-invertible projection).** 2D Kolmogorov
  at moderate Grashof is low-dimensional, so the low-Fourier shadow `Phi_K`
  nearly reconstructs the body (read-off `FVE ~ 0.99` in both physical norms).
  State-rigidity and control-rigidity nearly coincide -> the regime-2
  separation is strictly non-vacuous but **physically marginal**; the only
  under-determined content is dynamically-negligible dissipation-range noise
  (§6.3 row). The non-marginal regime needs high `G` / 3D (numerical wall).
  -> [`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md),
  [`proof/PDE_C1_NONMARGINAL_PROBE.md`](proof/PDE_C1_NONMARGINAL_PROBE.md).
- **Navier-Stokes C2 (Sabra shell) - NUMERICAL WALL (shadow unmeasurable with
  current tooling).** Fixed-dt RK4 blows up integrating through the intermittent
  dissipation bursts the task is about (four-obstruction catalogue); the
  matched-budget comparison never ran. Not a result - a *tooling* boundary.
  Resume needs an adaptive/stiff integrator (the shared C1-high-G / C2 build).
  -> [`proof/PDE_C2_CELLSET_SABRA_v1.md`](proof/PDE_C2_CELLSET_SABRA_v1.md).
- **Yang-Mills (SU(2) 3D, Phase 2) - BOUNDED NULL (shadow carries no separating
  structure on the cell).** Five `YM-P2-NEG-A` reads now span the original
  bare / smeared / correlator signatures against held-out coupling/spatial
  variance plus the amended finite-T Polyakov target. The v4/v5 powered-target
  audits were underpowered and the first v6 pilot was an unbracketed-grid void,
  but v6a supplied the missing powered+disjoint target and still scored at or
  below controls. Bounded cell-local null, lane PAUSED at the informative endpoint;
  external-review packet re-pointed to the powered-null story and now carries the
  §8.3 abelian-boundary explanation (Q2 / Faraday bridge note).
  -> [`SUNDOG_V_YANG_MILLS.md`](SUNDOG_V_YANG_MILLS.md),
  [`yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](yang-mills/receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md).
- **Riemann (Path i) - VACUOUS (rigidity trivially satisfied).** The
  `Z2`-descent is admitted but vacuous on gap-only features: the maximum
  reflection residual is identity-zero, so the projection's rigidity check
  passes *for the wrong reason* and carries no information. Escalation = an
  `S3`-triple bridge (Probe 01b) or external sanity check.
  -> [`SUNDOG_V_RIEMANN.md`](SUNDOG_V_RIEMANN.md).
- **P-vs-NP - BOUNDED-POSITIVE (the cost certificate clears; **Phase-4 adds a
  *measured* capacity threshold on a constructed substrate**; promotion still bounded,
  boundary moved).** Bounded alignment-verification. Phase 1 **v6**
  is the arc's first bounded-positive: the **op-count** cost certificate clears
  (signature 527,297 ops, cheaper to *check* than the policy's 555,876 ops to
  *find*; ratio 0.949 <= a pre-frozen 1.0 threshold), safety green (0 false accepts
  since v2, 0 spoof since v1). The earlier `v0-v5` cost-bounded reading is **retired**:
  **wall-time** superiority was withdrawn as non-reproducible (4 clean runs spanned
  3.5x; the op-count ratio is the only stable cost signal), so "the cost gate never
  clears" was a machine-load artifact, not a verifier-cost fact. The mesa
  verification bridge (Phase 2 v1) is also bounded-positive (op-count ratio 0.7376),
  but the capacity-relative one-wayness transfer (**Phase 3**) is **quarantined** -
  v0 falsified a registered cell, and v1's frozen block-consensus repair closes that
  spoof yet still quarantines on a block-unstable disclosure flag ("consensus-only
  repair"). So the projection is control-useful and now cost-clean, but the lane stays
  **unpromoted** - the boundary moved from the cost envelope to verification-transfer
  fidelity. Body-resistance is marginal (mesa `FVE(net.7|5D) ~ 0.97-0.99`), so this
  is cert-discipline transfer, not a regime-2 separation. NOT the Millennium P-vs-NP
  problem; op-count cost is not a complexity-theoretic claim.
  **Phase-4 update (2026-06-05): a *measured* capacity threshold on a CONSTRUCTED
  substrate - the third resistance flavor.** The Phase-4 synthesis built one constructed
  instance - the **syndrome/SIS certificate** (body = a secret `s` plus a light deviation
  `e`, `y = sG + e`; shadow = the syndrome `z = Hy = He`, which drops `s` by linear
  algebra) - and *measured* the find-vs-check separation the rest of the lane only argued.
  Unlike the marginal *control* substrates above, here the body **genuinely resists** its
  shadow, on a **computational / capacity** axis distinct from both the dimensional
  (cap-set) and topological (Aharonov-Bohm) axes: the secret is lost *algebraically*
  (`2^64` secrets per syndrome, exact - an AB-like unconditional lossiness but on the
  algebraic axis), and the deviation is lost *computationally* (recovering `e` from `z`
  alone is syndrome decoding, NP-hard). Yet the shadow stays **control-sufficient** -
  `Safe(y) := exists e*: He*=Hy and wt(e*)<=tau` is cheap and flat to *check* (16,576 ops).
  That is a regime-2 separation (state-insufficient yet control-sufficient) on a built
  substrate, with a **measured threshold**. **v1** (frozen `[128,64] w=12`, Prange ISD) is
  the lane's first measured capacity-relative one-wayness threshold: `C ~ 5,007` rank-valid
  trials, find-vs-check gap `~ 2.7x10^5x`, the 50%-breakpoint landing *on* the
  pre-registered prediction. **v2** tightened it against a Prange -> Lee-Brickell(p=2) ->
  Stern(p=2,l=8) ladder and produced a **bounded-positive with a NAMED MODEL-DEVIATION**:
  the magnitude held, but the fine ordering reversed at `w=12` - measured
  **Lee-Brickell is the best attacker** (`C_best = 8.31x10^7` ops, gap `~5,015x`), while
  Stern under-performs because its success heuristic is optimistic. **v3** then scaled
  the ladder and resolved the open question: Stern's win is a large-`w` phenomenon,
  overtaking LB at rungs 1 and 3 (`St/LB ~0.75, 0.76`) and locating the crossover between
  `w=12` and `w=16`; the top find-vs-check gap reaches `218,999x` at `[192,96] w18`.
  Two caveats are part of the result, not footnotes: rung 2 is a named pre-calibration
  heavy-tail / mean-vs-median model-deviation, and the frozen Claim-A gate reports
  `gap_scaling_not_confirmed` because the fixed-`n` rung 1 was mis-fit to an interior
  `>=5x` condition. Cross-substrate lesson: a frozen, calibrated, measured ladder can
  confirm the magnitude of a separation and *simultaneously falsify the fine ordering or
  prediction statistic inside it*; only the measurement says so. The resistance is
  **imported** (the SIS/decoding assumption), not proved, and `C_best` is an **upper
  bound** against the *tested* attacker classes (BJMM/MMT = a future slate) - a
  constructed-substrate, certificate-discipline *measurement*, not a complexity-theoretic
  claim. Receipts:
  [`pvnp/receipts/2026-06-04_certificate_syndrome_v1.md`](pvnp/receipts/2026-06-04_certificate_syndrome_v1.md),
  [`pvnp/receipts/2026-06-05_certificate_syndrome_v2.md`](pvnp/receipts/2026-06-05_certificate_syndrome_v2.md),
  [`pvnp/receipts/2026-06-06_certificate_syndrome_v3.md`](pvnp/receipts/2026-06-06_certificate_syndrome_v3.md).
  **Deductive core (2026-06-08):** the instance's *soundness* and *lossiness* (the `z=He` drop of `s`)
  are now **machine-checked in Lean** (axiom-clean, referee-free; public at
  [github.com/humiliati/sundogcert](https://github.com/humiliati/sundogcert),
  [`SUNDOG_V_CERTIFICATE_LEAN.md`](SUNDOG_V_CERTIFICATE_LEAN.md)) — the deductive complement to the
  measured ladders; the imported resistance is unchanged. The same Lean repo now spans five method
  examples (finite-field algebra, real analysis, halo minimum-deviation geometry, vector-calculus
  Aharonov-Bohm gauge-invariance, and the general characteristic-function law); the latter four are method
  context, not P-vs-NP evidence.
  -> [`SUNDOG_V_P_V_NP.md`](SUNDOG_V_P_V_NP.md).
- **ARC-AGI - CONVERGENCE-TO-NULL on the signature *shadow*; CAPABILITY-FLOOR on
  the *body*; body-resistance probe + v2 expansion ran 2026-06-01 -> the
  least-marginal substrate measured, but PR plateaus at ~11 (NOT sample-limited),
  robustly below the high-dim bar.** The earlier "Phase 3
  converged, reopen with a different framing" was
  acted on. (i) The **Phase 3E certificate program** ran the reframing and closed
  *four* negatives on the `signature_palette` shadow's geometry: no exact/near
  fiber collision, sparse fibers at 36 **and** 108 tasks, an oracle leakage
  regression at scale, and a rank-locality negative (palette tied with
  `metadata_only`, *below* `raw_grid`, 28.5% hard-incompatible). (ii) Separately,
  **Branch E** (deterministic program search over the task **body**, not the
  shadow) cleared a modest capability floor - the lane's first non-zero
  exact-match (v1: 2 distinct held-out tasks/lane; v2 replicated, bounded ~3%; the
  learned-ranker E3 is built + byte-verified but ~48 h compute-paused). So the
  shadow does not *organize* the body, yet the body is modestly *solvable* by
  search over it. (iii) The **body-resistance** probe (ARC roadmap **Phase 4**, the
  "5D / low-dim collapse check") **ran 2026-06-01** with the C1/Mesa
  participation-ratio + `FVE(body|shadow)` estimators ported to ARC (verdict
  **`arc_body_inconclusive`**, pinned `1892CBD4`, U_all=491): the raw-grid body has
  **PR 9.15** - the **least-marginal substrate measured** (≈4.6x the control column,
  where NSE-C1 FVE≈0.99, Mesa `net.7` PR≈2.0, and the shell eff-rank≈1.7 are all
  three-for-three marginal), no dominant mode (top-1 energy 0.287), and is
  **reconstruction-resistant** (held-out matched-dim `FVE(body|top-28 PCA)` only
  **0.659**; even top-200 PCA reaches just 0.79 - vs Mesa's 5-D shadow at ≈0.97-0.99).
  So on the reconstruction axis the ARC body *does* resist low-dim collapse - but
  PR 9.15 sits **below** the pre-registered 10x-marginal high-dim bar (`PR_HIGH_MIN=
  20`) at this 491-context register, so the verdict is inconclusive, **not** a
  high-dim claim, and the threshold is **not** retuned. v1 left open whether the
  shortfall was sample-limited (491 contexts). **v2
  (`PHASE4_BODY_RESISTANCE_CONTEXT_EXPANSION_SPEC.md`, pinned `64F8C0DD`)
  FALSIFIED that:** a spectrum-blind 8.8x expansion to all 1000 public-training tasks
  (4308 contexts) shows PR **plateaus at 11.38** (prefix sweep 10.76->11.38 over
  n=491..4308; `PR/bound` 0.0026, far from saturation; both construction lanes agree
  11.40/11.36; matched-dim FVE(top-28) 0.644, top-200 only 0.76) - the energy *ranks*
  grow with sampling (96/159/293 -> 409/719/1428) but the eigenvalue-weighted
  *effective* dimensionality does not. So the ARC body is a **stable intermediate**:
  ~5.7x the marginal controls but a **converged** PR~11, **not** a >=20-PR high-dim
  body even over the whole corpus, and not an under-sample. Net: the *control* column
  stays three-for-three marginal; the ARC *computational* body is the first that is
  materially more dimensional, but its converged value is still short of the bar. (The
  naive `FVE(grid|signature_palette)` 0.43/0.49 is a coarse baseline only.) Next
  admissible escalation = a different (object-centric / pair-relational) body encoding
  under its own spec, or look outside ARC for a >=20-PR computational body; lowering
  the bar stays forbidden. -> [`SUNDOG_V_ARC.md`](SUNDOG_V_ARC.md),
  [`prereg/arc/README.md`](prereg/arc/README.md),
  [`prereg/arc/PHASE4_BODY_RESISTANCE_CONTEXT_EXPANSION_SPEC.md`](prereg/arc/PHASE4_BODY_RESISTANCE_CONTEXT_EXPANSION_SPEC.md)
  (Amendment 2).
- **Isotrophy - CONDITIONAL + TIER-2 EXTERNAL (rigid after conditioning, then
  reliability-explained).** The velocity-fraction shadow ranks stability only
  *within* the right mass strata; it fails as a mass-marginal held-out predictor.
  Read as a continuous tail-resolved score, it transfers to Li/Liao 2021 and its
  per-cell heterogeneity is explained by frame reliability. Already documented
  at §7.2; listed here for the map.
- **Threebody - DEFLATIONARY (the projection is simpler than first claimed).**
  Phase 18 reduced "guarded TRACK / tidal sensing" to a radius-gated inward
  reflex - an over-attribution corrected, not a separation. Already at §7.3.
- **Faraday / classical EM - IDENTITY-SUCCESS (the shadow closes because the
  body<->shadow law is a theorem, so there is no body to resist).** The
  portfolio's only clean structural zero: local plaquette-holonomy data closes
  Faraday induction with no global reconstruction (Branch A). But the closure is
  the Bianchi identity `dF = d(dA) = 0` - body-resistance is *exactly zero by
  theorem* (§8.1), so this is a correctness check on the operator, not a
  regime-2 separation. The exact-zero anchor of the body-resistance axis that the
  *marginal* failures (NSE-C1, Mesa, shell) only approach numerically.
  -> [`SUNDOG_V_FARADAY.md`](SUNDOG_V_FARADAY.md),
  [`faraday/SHADOW_FARADAY.md`](faraday/SHADOW_FARADAY.md).
- **Aharonov-Bohm / EM topology - EXACT-SEPARATION (topological), EARNED 2026-05-31.**
  The same operator's loop tier on a non-contractible patch (`H^1 != 0`):
  state-insufficient (one flux number cannot rebuild the interior field) yet
  control-sufficient (it fixes the AB phase), *exactly*, with the gap an `H^1`
  invariant - while the local `F` tier goes control-blind. The portfolio's first
  *exact* regime-2 witness, on a **topological** (not dimensional) resistance
  axis (§8.2). Landed as Phase 7 case 3 (B7-topology).
  -> [`faraday/FARADAY_PHASE7_BOUNDARY.md`](faraday/FARADAY_PHASE7_BOUNDARY.md).
- **Algorithmic approximation - LIT-NOTE + one machine-checked Lean core (registered 2026-06-26).** A
  reading note + falsifiable hook-slate spun off arXiv:2606.26705 (Kratsios et al.),
  which prices network size by the **algorithmic complexity** (gate-count of a
  circuit over a fixed gate language) of the target rather than its regularity - the
  constructive **upper-bound** mirror of the P-vs-NP lane's **lower-bound** /
  resistance object (same op-count ledger arithmetic, inverted question: cheaper to
  *emulate the program* than to approximate the worst smoothness sibling).
  **Six of six hooks closed (2026-06-26):** H-A0 PDF verification (labels confirmed,
  formal Cor/Prop resource summaries corrected); **H-A4 Atlas short-program →
  `ATLAS_IS_A_SHORT_PROGRAM`** (deductive): the Atlas forward generator is a fixed
  composition of analytic gates (sin/cos/arccos/arctan2/sqrt/abs) over ~1 continuous param
  + fixed ice lattice + finite habit branch, the only iterative element a log-depth
  bisection → bounded depth / polylog-in-1/ε, the √x side of the paper's divide (the
  Atlas's own multi-scattering / turbulent-caustic failure boundary is the Brownian side;
  cost profile O(log 1/ε) rhymes with the APSP headline); **H-A1 cost-certificate isomorphism →
  `UNIFIES_ON_EXACT_FRAGMENT`** (deductive): one shared measure — the faithful op-count of
  a straight-line program over a fixed algebraic dictionary — coincides *exactly* on the
  linear/PL fragment (`CheckCost.verifyCost`'s dominant `m·n` = the gate count of the GF(2)
  matrix–vector map `y↦H·y`, the same object as an affine gate over ℝ; H-A2's ε=0 exactness
  is the witness), bounded by direction (verifier-vs-constructor = the find/check axis),
  ε-dependence (the analytic gates), and lossiness (no cost analog) → a shared yardstick,
  not a merged theorem, now formalized in `Sundogcert/StraightLineCost.lean`
  (`costOf` specializes to both `RProg.gateCount` and `CheckCost.verifyCost`, with
  `compileToDag_cost_le` + `verifier_cost_le` audited); **H-A2 → a real axiom-clean Lean core**
  `Sundogcert/CircuitNet.lean` (`CORE_EARNED`, the falsifier did not fire): the exact
  (ε = 0) tropical→ReLU compilation `compile_eval` + linear-depth bound + min-plus /
  Bellman–Ford gates (the APSP/Cor 5.1 building blocks) proved exact, full `lake build`
  green and in the AxiomAudit gate; o-minimality does zero work, the analytic gates +
  trainability remain imported, and the **linear gate-count wall is now CLOSED** — a
  sharing-aware ReLU DAG (`RProg`) + a recursive `compileToDag` fold the four-gate max
  gadget over a tropical tree, proving an `N`-node tree compiles to a `≤ 4N`-gate DAG
  (`compileToDag_gate_count`) computing it exactly (`compileToDag_eval`), both axiom-clean
  and audited; only the analytic gates remain approximable; and **H-A3, a deductive
  competence-dominance re-read → `SUPPORT_NO_SIZE_SEPARATION`** — the paper's universality
  (Thm 3.1) + compilation (Thm 3.2) re-derive the Competence-Dominance Lemma in
  circuit-complexity vocabulary and add no return-axis size gap a plural topology could
  exploit (Cor 5.1 structured-circuit cheapness is a *target* property available to both
  classes, not a plural-topology win; and as an existence/optimum theory it is blind to
  the lone learnability loophole, the H2.3 cap), its only positive output a *sharpening*
  relocating the live question onto **objective selection** (the §4 non-sovereignty axis);
  and **H-A5 → `ORTHOGONAL_FLOORS` / `NO_SHARED_SEPARATOR`** (deductive): the
  apples-to-apples check rejects a single shared separator. O-minimality is a
  syntax/realizability predicate on target maps and architectures; the charFun law is a
  semantic spectral predicate on averaging measures in lossy shadows. Definable/finite
  does not imply resist (the two-point lattice is short/tame but survives via recurring
  `cos`), and resist does not imply definability (AC/Rajchman measures can decay without
  a short o-minimal description). Strong form: the independence holds even *inside* the
  definable class — Cauchy (rational density) and Gaussian (`ℝ_exp`) both resist while the
  tame two-point lattice survives, so definability does no work on the resist/survive axis.
  The retained bridge is two-stage only: definable generators make shadows scoreable;
  charFun decay decides what washes out.
  The H-A2 core is a **method/deductive example** (like the Halo/Faraday/ShadowDecay Lean
  cores), not generality evidence, so the map entry stays LIT-NOTE.
  -> [`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md).

**Failure-mode taxonomy.** Eight distinct ways a cross-substrate generalization
*fails*: *marginal* (body doesn't resist), *numerical* (shadow unmeasurable),
*bounded-null* (shadow non-separating), *vacuous* (rigidity trivial),
*cost-bounded* (envelope blocks promotion), *convergence-to-null* (needs
reframing), *conditional* (rigid only after conditioning), *deflationary*
(over-attribution). Two non-failure anchors now bound the map: *identity-success*
(Faraday - the shadow closes because the body<->shadow law is an identity, so
there is no body to resist) and *exact-separation* (Aharonov-Bohm - a sharp,
exact, topological regime-2, pre-registered). A third bounded disposition,
*bounded-positive* (P-vs-NP, v6), marks a lane whose core certificate clears (the
op-count cost is cheaper to check than to find) yet stays unpromoted on a moved
boundary - wall-time non-reproducibility plus a quarantined verification bridge. Its
Phase-4 syndrome-certificate instance sharpens this to a *measured* capacity threshold on
a **constructed** substrate (computational/capacity resistance - the third axis), and
adds a sub-pattern: *bounded-positive with a named model-deviation* (v2 - the separation's
magnitude confirmed within the locked tolerance, but the fine attacker-ordering inside it
falsified, Lee-Brickell beating the predicted-best Stern at `w=12`).
Cataloguing the mode - success or
failure - is what turns each substrate into a design constraint for the next.

## 1. What Transfers

Mesa and Geometry both converged on field-shaped structure:

- Mesa: a behavioral cliff localized to an entangled 5D `net.7` activation
  subspace, with direction-specific neuron substrates and failed linear
  decompositions.
- Geometry: a small set of shared halo generators whose visible arcs are
  forward-rich but inverse-narrow, with HaloSim acting as an independent
  forward oracle.
- Threebody: a mapped near-escape survival pocket whose Phase 18 mechanism
  audit is deflationary: at matched duty, radius-gated inward thrust reproduces
  guarded TRACK without tidal sensing or gradient steering, while lower-velocity
  and equal-mass boundaries still expose harms.
- Isotrophy: a two-layer audit-chain result. K_facet v0.3h remains a structural
  null / named-quarantine result on the strict G.2 catalog, while the
  velocity-fraction lane registers a separate conditional catalog signal:
  Floquet velocity-fraction ranks stability within the right mass strata, fails
  as a mass-marginal held-out predictor, then transfers to Li/Liao 2021 as a
  continuous tail-resolved score with frame reliability explaining the per-cell
  heterogeneity.
- De-confound (Attack-B): the determining-shadow **closure** read transfers to **real features**
  when the functional is **linear** and the substrate is **de-confound-clean** (digit features:
  0B confirms the closure dissociation, 0C shows the input-de-confound is load-bearing). It does
  **not** transfer to real **computed-state** tasks (Othello legal moves), whose natural
  functionals are **nonlinear** — the linear instrument cannot bracket them. Closure-measurability
  is gated by *functional linearity + de-confound cleanliness*, not by realness of the substrate
  per se — a substrate-specific, asymmetric transfer in the same spirit as the posture below.

The shared theorem posture is:

> Useful action can emerge from indirect signatures when the signature remains
> attached to the field, but the attachment is substrate-specific, asymmetric,
> sometimes simpler than the first successful controller, and bounded by failure
> maps.

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

Historical note: this section was drafted before Phases 14-18 ran. The
suggestions below remain useful as provenance for the mechanism program, but
the current controller interpretation is the Phase 18 result: the favorable
survival envelope reduces to radius-gated inward thrust, not tidal-gradient
steering.

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

Historical note: the sidecar no longer stops at the original v0.3 symmetry-
descent workbench. The current claim profile is two-layered. First, K_facet v0.3h
resolved 20 strict G.2 rows as structural zeros and quarantined O_617 rather
than converting it into a false 21/21. Second, the v0.10-v0.13 reopening found a
conditional stability signal in the supplementary-B piano-trio catalog:
velocity-fraction ranks stable above unstable rows within fixed `m3` strata
(`AUC_cond = 0.6783`, exact `p = 2.046e-7`) while the mass-marginal held-out
predictor fails (`AUC = 0.4125`). This is a conditional catalog signal, not a
promotion of K_facet and not a controller result. v0.12 then tried to externalize
it: the frozen rule taken to supplementary-A landed
`external_transfer_blocked_by_attrition` (unbiased uniform probe attrition `0.3433`,
Wilson95 `[0.2919, 0.3987]` above the 0.20 gate) -- the frozen D5 measurement is
numerically intractable on ~1/3 of supp-A, a measurement-feasibility wall rather
than a falsification. v0.13's signal-blind target search then returned an
independent-target landscape negative (every Tier-3 external catalog fails a hard gate
-- equal-mass, restricted substrate, or too small), leaving only Tier-2 Li/Liao 2021
(unequal-mass, 135,445 rows) as schema-viable. Pricing that transfer surfaced the
deepest finding of the arc: cross-ansatz overlap is empirically zero (leakage 0.0), but
the velocity-fraction FEATURE is frame-relative -- `select_gamma_1`'s largest-real-part
direction pick flips under a coordinate rotation for most orbits (vf is frame-invariant
for only ~22-44% across the two catalogs). The adapter is provably correct (clean orbits
give vf invariant to 1e-8); it is the MEASUREMENT that is frame-sensitive. v0.13b then
priced this at the registered-zone level: the coarse v0.11 zone (the rule's actual
input) is frame-stable on the v0.11 domain itself (supp-B zone-change `4.35%`, Wilson95
upper `14.5% <= 0.15` bar -- a MODEST caveat), and frame-stable enough cross-ansatz
(liao2021 pooled `~1.3%`, the `n=600` expansion `0.5%`; verdict
`coarse_zone_rule_frame_stable_enough_to_test`, fragility localized to two narrow mass
bands), clearing the transfer test to run.

v0.14 then ran that test: a registered 1280-row sample (16 sorted-mass quantile cells x
80) from liao2021 under the frozen v0.11 within-cell conditional-AUC rule. It landed
`sample_transfer_undecidable_coverage` -- only 7 of 16 cells cleared the locked
`S>=4, U>=4, N>=40` primary bar (560 of a required 800 rows), because liao2021's
non-hierarchical population is overwhelmingly unstable AND overwhelmingly velocity-heavy
(9 cells held zero stable orbits; two evaluable cells were pure zone 2). Inputs were
otherwise clean (attrition `1.33%`, frame zone-change `1.35%`). Per the locked claim
boundary NO transfer reading is licensed: the supp-B conditional signal is neither
confirmed nor refuted on liao2021 -- the obstruction was the target's outcome x feature
geometry, not measurement feasibility (v0.12) or frame fragility (v0.13b).

So the cross-substrate lesson sharpens three times: (1) a real within-stratum signal can
be blocked from external confirmation by the *measurement's* numerical reach, not only by
the data; (2) a derived feature built on eigenvector SELECTION can be frame-relative
even when its registered coarse binning is robust -- so the honest theorem-facing object
is "a coarse, frame-dependent projection whose registered bins are empirically
frame-stable," NOT "a frame-invariant geometric quantity"; and (3) even with a clean
feature and a feasible measurement, external confirmation can still be blocked by the
*target population's outcome geometry* -- liao2021 is too one-sided in both stability
outcome and feature zone to host the within-stratum comparison the rule needs (v0.14
`sample_transfer_undecidable_coverage`). Any substrate whose feature is an
argmax-selected direction (not just isotrophy) inherits the (2) caveat; any conditional
transfer target must clear the (1) and (3) pre-conditions before its statistic is read.

v0.15 and v0.16 then RESOLVED the (3) obstruction and added a fourth sharpening. v0.15
defeated the coverage wall with an outcome-balanced 80/80 case-control draw over the 7
stable-support cells -- but landed `directional_weak` (AUC_cond 0.5125, p 3e-4): the
direction transferred and was significant, yet the coarse {0.25,0.50} zone is 98.4%
saturated in liao2021's stable-support region, so the binning discarded almost all the
information and the effect sat below the 0.55 floor. v0.16 then read the SAME projection
as a continuous 4-frame ensemble-median vf (a new tail-resolved, frame-aggregate feature
on fresh doubly-held-out rows) and PASSED clean: `AUC_cond 0.647, p 1e-5`, while the
coarse zone on those same rows reads 0.510. So (4): even a real, frame-stable,
coverage-satisfied projection can be throttled by FEATURE GRANULARITY -- a binning that
saturates in the target population hides a signal the continuous form recovers. The net
result is the program's first held-out external (Tier-2) confirmation: the
velocity-fraction PROJECTION transfers across the ansatz boundary, even though its coarse
v0.11 FORM does not (pooled and mass-cell-heterogeneous -- 5 of 7 cells in-direction, two
reversed -- so a transfer claim, not a uniform law).

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

- [`sundog_v_isotrophy.md`](sundog_v_isotrophy.md)
- [`isotrophy/files.math`](isotrophy/files.math)
- [`isotrophy/kfacet/kfacet_v11_m3_conditional_vf_rank_form.md`](isotrophy/kfacet/kfacet_v11_m3_conditional_vf_rank_form.md)

## 5. Documentation Shape

Follow the Mesa pattern:

- Keep `SUNDOG_V_THREEBODY.md` as the narrative spine and claim ledger.
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
| Mesa controller | controller hidden state (`net.7`, 256-wide but **effectively ~2-dim** - a fn of 6-dim obs) | entangled `5D net.7` subspace - **sharp on irreducibility, marginal on body-resistance** (`FVE(net.7\|5D)~0.97`); see the body-resistance note below |
| Geometry / HaloSim | full atmospheric optics | small set of halo generators / canonical implied circles |
| Threebody | 18-dim full state | radius-gated inward reflex in the mapped near-escape pocket |
| Isotrophy | `S3` symmetry orbit of a 3-body choreography; supplementary-B / Li-Liao 2021 stability catalogs | residual `Z2` generators surviving `S3 -> Z2` mass perturbation; Floquet velocity-fraction conditioned on mass strata; tail-resolved continuous vf with frame reliability explaining transfer heterogeneity; spectral-gap LOW TAIL (`q10` Floquet Re-gap) as a label-blind a-priori reliability predictor (v0.19 fragility / v0.20 tail-bridge `rho = 0.882`) |
| Navier-Stokes (C1) | 2D Kolmogorov attractor state (finite-Galerkin, 440 real DOF) | low-Fourier signature `Phi_K` (K=3, 18-dim) - **but the body barely resists**: read-off `FVE(body\|shadow) ~ 0.997` (energy) / `0.993` (enstrophy), a near-invertible projection, so the regime-2 separation is **marginal**; genuine under-determination (`R^2 ~ 0.71` per-DOF) sits only in physically-negligible dissipation-range modes |
| Faraday / EM (homogeneous) | EM field history on a contractible patch (smooth `A`, `dF = 0`) | plaquette holonomy `∮A = ∫F` - **zero body-resistance *by identity***: closure is the Bianchi identity `dF = d(dA) = 0`, so the shadow reconstructs the body exactly. The exact-zero anchor of the axis (§8.1) |
| Aharonov-Bohm / EM (topological) [EARNED] | EM field config on a non-contractible patch (`H^1 != 0`) | loop holonomy `∮A = Φ` - **exact *topological* body-resistance**: one flux number is state-insufficient (cannot rebuild interior `B`) yet control-sufficient (fixes the AB phase); local `F` is control-blind. Portfolio's first *exact* regime-2 witness; earned by Phase 7 case 3 (§8.2) |
| Sourced EM / Maxwell (inhomogeneous) [EARNED] | full sourced field on a patch | dual-shadow `∮*F = Q_enc` (Gauss / Ampere-Maxwell) - **determined, not resisting**: by the uniqueness theorem the field is fixed by (sources, boundary, harmonic periods); the low-dim shadow is a lossy Gauss summary with no sharp objective-free regime-2. All exact regime-2 is the harmonic/AB sector above. Marginal by *determinacy* (Phase 8, §8.4) |
| P-vs-NP syndrome/SIS cert [CONSTRUCTED, MEASURED] | secret `s` + light deviation `e` in `y=sG+e`; recovering `e` from `z` alone is syndrome decoding (NP-hard) | syndrome `z = Hy = He` - **computational/capacity resistance** (a third axis beyond dimensional/topological): `s` lost by algebra (`2^64`/syndrome, exact-lossy), the deviation *capacity-hard* to find, yet `Safe(y):=∃e*:He*=Hy∧wt≤τ` cheap+flat to check (16,576-37,152 ops across measured rungs). Measured thresholds: v2 `C_best=8.31×10⁷` ops (Lee-Brickell, gap `≈5,015×`), v3 top rung `C_best=8.136×10⁹` ops (Stern at `[192,96]w18`, gap `≈218,999×`); imported hardness, upper bound vs tested attackers, Claim-A frozen gate caveat disclosed (§ failure-map P-vs-NP Phase-4 update) |

The columns are the same operator. The substrate-specificity in row
3-5 is what bounds the operating envelope; the cross-substrate
recurrence in rows 1-2 is what tells us the operator is real.

**Body-resistance is the load-bearing property (2026-05-29, NSE
addition).** The Navier-Stokes (C1) row is the table's *marginal
boundary*, and it sharpens what "bounds the operating envelope" means
for the **control** reading of the operator (Postulate-1: a
control-rigid shadow of a state-resistant body). Where the body barely
resists the shadow - where the projection is near-invertible, so the
body can be reconstructed from the shadow (NSE-C1: `FVE ~ 0.99` in both
physical norms) - state-rigidity and control-rigidity nearly coincide
and the regime-2 separation collapses toward vacuity (strictly
non-vacuous, but physically marginal). So a *sharp* control-regime-2
needs the body to genuinely resist - and **quantifying this
(2026-05-29) showed two axes were being conflated**, correcting an
earlier over-claim:

- **Body-resistance** (`dim(body) >> dim(shadow)`, shadow *cannot*
  reconstruct the state) - the actual Sundog regime-2 axis. **Cap-set /
  unit-distance** have it (exponential body). **2D NSE (C1) does not**
  (low-dim attractor; `FVE ~ 0.99`). **Mesa does not either** - measured by
  porting the C1 FVE estimator to `net.7`: the 256-wide `net.7` is
  effectively **~2-dimensional** (participation ratio 2.0, robust across
  input distributions), so the 5D cliff shadow reconstructs
  `FVE(net.7|5D) ~ 0.97-0.99`. Structural: `net.7` is a function of the
  **6-dim** observation, so its image is an intrinsically low-dim manifold -
  there is no high-dim body to resist.
- **Shadow-irreducibility** (the shadow *cannot be decomposed* into
  sparser/simpler pieces) - a *different* property. **Mesa has this** (the 5D
  resisted SAE / top-k / PC decomposition; "read holistically"); C1's `Phi_K`
  does not.

So **Mesa is sharp on irreducibility but marginal on body-resistance**, and
2D NSE is marginal on both. Reading Mesa's irreducibility as body-resistance
was the over-claim the FVE measurement falsified. **Honest consequence: no
*control* substrate in the portfolio is demonstrated sharp on the
body-resistance (Sundog regime-2) axis** - both measurable control lanes have
intrinsically low-dimensional states; the body-resisting substrates (cap-set,
unit-distance) are read-off, not control. A sharp *control* regime-2 needs a
genuinely high-dimensional input/state - high-Re turbulence (behind the
numerical wall) or high-dim RL/LLM agents (not in the portfolio). The non-marginal NSE regime (high `G` / 3D,
where physically-relevant content is genuinely under-determined) sits
behind the same numerical wall as the C2 shell model and needs the
adaptive integrator.

**Three-for-three (2026-05-29): every measurable *control* substrate is
marginal on body-resistance.** After C1 (NSE) and Mesa, the Sabra **shell
model** — the best PDE candidate, with a forward cascade and a small-scale
control objective — was probed cheaply (`PDE_C2_SHELL_DIMENSIONALITY_PROBE.md`,
within the fixed-dt stable window): effective rank of the real inertial
cascade ≈ **1.7 of 30** (low-rank, slaved power-law), and the perm-arbiter
shows the shells are *predictable from the low-shell shadow* (`R²_real ≫
R²_perm`) — directionally **marginal** too (window-limited by the same
numerical wall, but eff-rank 1.7 is too low to flip). So the sharp
*control* regime-2 is **not** demonstrated anywhere in the portfolio's
control substrates — cascade-organized PDEs concentrate their relevant content
in slaved/low-rank structure, and the controllers are low-dim by input. The
genuinely body-resisting substrates (cap-set, unit-distance) are read-off, not
control. **Where it would live: a substrate that is high-dimensional *by
construction* — high-dim RL / LLM agents (high-dim observations or model
states) — which is the structural property all three measured substrates
lack.** First data point on that gap (2026-06-01): the **ARC Phase 4
body-resistance probe** (`prereg/arc/PHASE4_BODY_RESISTANCE_SPEC.md` v1 +
`PHASE4_BODY_RESISTANCE_CONTEXT_EXPANSION_SPEC.md` v2) measured the raw-grid
*computational* body's read-off dimensionality. v1 (491 contexts) read **PR 9.15** —
materially above the three marginal controls (≈ 2) and reconstruction-resistant
(top-28 PCA recovers only 0.66) — yet **below** the pre-registered 10×-marginal
high-dim bar (verdict `arc_body_inconclusive`). The disciplined v2 reopen then tested
whether that shortfall was sample-limited: a spectrum-blind 8.8× expansion to all
1000 public-training tasks (**4308 contexts**) shows **PR plateaus at 11.38**
(prefix-stable 10.76→11.38 over n=491..4308; both lanes 11.40/11.36), so the
shortfall is its **converged** value, not an under-sample — `arc_body_inconclusive_expanded`.
The ARC body is thus a **stable intermediate**: the least-marginal substrate measured
(≈ 5.7× the controls), but (a) still a **read-off** body, not the *control*-sufficient
low-dim shadow this gap names, and (b) a converged PR ≈ 11 that does **not** reach the
≥ 20 high-dim threshold — so the control regime-2 frontier stays open, and ARC at
public-training scale does not supply the high-dim body either. Measured detail:
`sundog/docs/proof/PDE_C1_NONMARGINAL_PROBE.md`, `PDE_C1_PROPOSITION.md`,
`PDE_C2_SHELL_DIMENSIONALITY_PROBE.md`.

**The prediction tested (2026-05-30 → 06-01): a de-confounded, high-dim *control*
body that resists and *scales*, with a robust objective-driven (at-the-bar) contrast.** The "where it would live"
claim above was built and probed in the new `chatv2` lane
(`sundog/docs/chatv2/`): a from-scratch transformer trained generatively on a
synthetic `H`-latent channel task, with the C1/Mesa fingerprint ported to its
residual stream. Two methodology catches came first and both matter for reading
the result: (i) the initial run looked *marginal* but was a **variance-masking
artifact** — transformer "massive-activation" outlier features hid the latent
code from the variance metrics (`eff_dim ≈ 1.6` while 16 latents were linearly
decodable); an information-basis re-measurement (`d_dec`, leak, body_carry)
overturned it. (ii) The first computed-latent substrate failed a **mandatory
linear-input-probe pre-check** (the latents were passively input-decodable — the
same confound that makes the result trivial); a pair-XOR latent (provably not
linearly decodable) fixed it. **With both controlled, the generative body is
state-insufficient-yet-control-sufficient with a clean *objective-driven*
contrast: the control-only twin fails to build the non-decision state
(`body_carry` ≈ chance) where generative training builds it (~0.9) — SHARP at
`H = 2, 4`.** This was first seen, SHARP, at `H = 2, 4`, and a standalone `H=8`
probe hit SHARP at `d_dec≈7` (`z1_acc`=0.94). **Scaling, then the honest tempering
(2026-05-31 → 06-01):**

- A capacity + signal bump (`d_model`→192, δ→0.45, de-confound intact at pre-check
  0.50) got the model *learning* at `H=8`, and an `H=8→16` warm-start curriculum
  got it learning at `H=16` too — so the high-`H` wall was **optimization
  (grokking), not capacity**. The **resisting body scales**: `d_dec` 6.8 (`H=8`)
  → 13.7 (`H=16`), `leak`≈chance at both. *This is the robust finding.*
- **But the full SHARP verdict is seed-sensitive.** The curriculum's `H=8`
  (same seed, different init via `--pos-h`) came back MARGINAL (`z1_acc`=0.61),
  as did its `H=16`. A read-position fairness fix (read each latent where its
  channel is freshest) lifts `z1_acc` only ~+0.05 (0.61→0.66) — so the marginal
  is **genuine solution-quality variance, not a read-position artifact.** The
  earlier "first dimensional-axis SHARP" headline was **one favorable draw.**

**Robust across draws:** a high-dimensional control body that resists (`d_dec`
large, `leak`≈chance) and **scales** with `H` — the closest any control substrate
has come. **The 3-seed `H=8` seed-stability test ran (2026-06-01)** and resolved
the scare: the body-resistance is **seed-robust** (high-dim / control-sufficient /
resisting, tight across seeds — the earlier "sensitivity" was a `pos_h` *config*
artifact, not seed variance), and a baseline-ladder decomposition shows the
gen-vs-control contrast is a **robust objective-driven excess of `0.205 ± 0.022`**
(~67% of the body's non-decision content is objective-driven; control training
adds ≈0; ~0.09 is architectural / random-feature, a falsified-"twin incidentally
computes" surprise). Gate E passes 2/3 (`phase1_scaling_sharp_below_bar`) — an
*at-the-bar* SHARP whose underlying contrast is seed-stable. Still `d_dec ≈ 7.6
< 20` (below the high-dim bar) and a synthetic toy; **unpromoted**. Detail:
`chatv2/PHASE1_SEEDSTAB_RECEIPT.md` (+ `PHASE0_2_COMPUTED_LATENTS.md`,
`PHASE0_MINIMUM_FALSIFIABLE.md` Amendment 1).

**Resistance has (at least) two axes (2026-05-31, Faraday addition).** The
body-resistance discussion above is *dimensional* - `dim(body) >> dim(shadow)`,
the cap-set / unit-distance / high-dim-control axis, the one every measured
control substrate falls short on. Faraday / Aharonov-Bohm add a second,
*topological* axis: on a non-contractible domain the *local* gauge-invariant
shadow (`F`) is a complete set of local observables yet fails to determine a
*global* one (the AB holonomy), with the gap living in `H^1`. This resistance is
independent of dimension - AB's body is low-dimensional but resists *exactly*,
because the obstruction is cohomological, not dimensional. So "the body resists
its shadow" can mean *too big to reconstruct* (dimensional) or *globally
under-determined by local data* (topological). The portfolio's only **exact**
regime-2 (AB) lives on the topological axis. On the dimensional axis the measured
*control* substrates are marginal (NSE-C1, Mesa, shell). **chatv2 shows a
de-confounded high-dim body that resists and *scales* (`d_dec` 7→14) — the closest
any control substrate has come — with a **seed-robust** body-resistance and a
robust objective-driven contrast (`0.205 ± 0.022` over 3 seeds), an *at-the-bar*
SHARP (Gate E 2/3). Still `d_dec ≈ 7.6 < 20` and a synthetic toy, so a
robust-but-below-bar dimensional-axis separation, not yet promotable.** See §8 for
the full Faraday/Maxwell reading.

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

### 7.2 Isotrophy Red Team (v0.3-v0.13)

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

Post-v0.11 update: isotrophy now contributes a second projection lesson, separate
from K_facet. The body is the per-row monodromy/Floquet geometry of the
supplementary-B piano-trio catalog; the shadow is a signed velocity-fraction
zone. v0.10a showed the ordered shadow is real in-sample. v0.10b showed it is
not globally calibrated across mass bins. v0.11 showed the projection becomes
rigid again after conditioning on `m3`: within fixed mass-ratio strata, the
frozen velocity-fraction zone order ranks stable rows above unstable rows.

That is almost the cleanest possible cross-substrate boundary result: the
projection works exactly after naming the substrate coordinate that must be held
fixed. It does not become a mass-marginal predictor, and it does not revise the
v0.3h K_facet structural null. For future isotrophy work, the next meaningful
step is therefore not another same-table statistic; it is either an external
catalog transfer, a freshly registered mass-conditional predictor, or a new
mechanism that explains why the velocity-fraction shadow is conditional on
`m3`.

Post-v0.13 update: the external-catalog step was taken, and it produced isotrophy's
THIRD projection lesson. v0.12 took the conditional rule to same-paper supplementary-A
and hit `external_transfer_blocked_by_attrition` (the frozen D5 measurement is
numerically intractable on ~1/3 of the target -- a feasibility wall, not a
falsification). v0.13's signal-blind search found only a Tier-2 (same-lineage Li/Liao
2021) schema-viable target; the v0.13a/b preflight on it surfaced the sharpest lesson of
the program: the projection's FEATURE (dominant-direction velocity-fraction) is itself
frame-relative -- `select_gamma_1`'s largest-real-part argmax flips under a coordinate
rotation for the majority of orbits -- yet its registered coarse zone bins are
frame-stable on the v0.11 domain (supp-B zone-change `4.35% <= 0.15`) and frame-stable
enough cross-ansatz (liao2021 pooled `~1.3%`; verdict
`coarse_zone_rule_frame_stable_enough_to_test`). The honest theorem-facing
object downgrades from "a frame-invariant geometric quantity" to "a coarse,
frame-dependent projection whose registered bins are empirically frame-stable." This is
the most transferable caveat isotrophy has produced: **any projection whose shadow is an
argmax-selected direction** (Mesa `net.7` directions, NSE control shadows, any
eigenvector-keyed feature) **should be frame-audited before its bins are read as
geometry** -- a rotation/translation invariance check on the registered statistic, cheap
relative to the claim it protects.

Post-v0.14 update: the cleared transfer test then ran and produced isotrophy's FOURTH
projection lesson. v0.14 drew a registered 1280-row liao2021 sample (16 sorted-mass
quantile cells) under the frozen v0.11 within-cell conditional-AUC rule and landed
`sample_transfer_undecidable_coverage` -- only 7 of 16 cells could host a stable-vs-
unstable comparison (560 of a required 800 rows), because the non-hierarchical catalog
is overwhelmingly unstable and overwhelmingly velocity-heavy. No transfer reading is
licensed (neither confirmed nor refuted). The lesson generalizes the body/shadow frame:
**a projection rigid on its home substrate can be confirmation-blocked on a new one by
the target's outcome x feature geometry alone** -- before any statistic is read, a
candidate transfer target must be checked for *joint coverage* of both the outcome
classes and the feature bins inside each conditioning cell, or the conditional test is
structurally undecidable. v0.12 (measurement reach), v0.13b (feature frame-stability),
and v0.14 (target coverage) are three distinct pre-conditions a conditional projection
must clear before an external catalog can confirm it.

Post-v0.17 update: the cleared-but-saturated path then resolved into the program's
strongest cross-substrate result -- isotrophy's FIFTH projection lesson. v0.15 defeated
v0.14's coverage wall (outcome-balanced 80/80 case-control over the 7 stable-support
cells) but landed `directional_weak`: the coarse zone is feature-SATURATED in the target
(98.4% one zone), so the binning hid the signal below the floor. v0.16 read the same
projection as a continuous 4-frame ensemble-median vf on fresh doubly-held-out rows and
**passed clean (AUC_cond 0.647, p 1e-5)**, with the coarse zone on those same rows at
0.510. So a fourth transfer pre-condition joins (measurement reach / feature
frame-stability / target coverage): **feature granularity matched to the target
distribution** -- a projection that is frame-stable, coverage-satisfied, and
measurement-feasible can still need the right feature RESOLUTION to express its signal
externally, because a coarse bin that saturates in the target throws the signal away
while the continuous form recovers it. v0.17 then repeated the continuous read on fresh
triple-held-out rows and landed `heterogeneous_transfer_replicates_clean`
(AUC_cond 0.646875, p 1e-5, exact Spearman rho=1.0, p_rho=1/5040 against the v0.16
cell vector). This is the program's first replicated held-out external (Tier-2)
confirmation that the velocity-fraction projection transfers across the ansatz boundary
-- pooled, mass-cell-heterogeneous, Tier-2 and bounded, not theorem-facing. The post-hoc
anatomy reads the heterogeneity as a reliability map: the main sub-0.5 cell is the
localized frame-fragile band already flagged by v0.13b, and the other is near-null. That
"why" is interpretation, not a registered mechanism.

Post-v0.18 update: the reliability-map interpretation is now registered evidence, not
only anatomy. v0.18 widened the liao2021 map to 18 fresh quadruple-held-out 8 x 8 cells
and landed `reliability_drives_per_cell_auc_supported`: pooled AUC_cond 0.620208
(`p = 1e-5`), label-blind frame reliability predicts per-cell AUC (`rho = 0.5975`,
`p = 0.00523`), and the reversal guard found 0 frame-stable decisive-negative cells.
The two decisive negative cells are exactly the frame-fragile cells; the frame-stable
sub-0.5 cell is not statistically decisive (`p_reverse = 0.0968`). So isotrophy's SIXTH
projection lesson is reliability-as-mechanism: for eigenvector/argmax-selected shadows,
the heterogeneity of transfer can itself be governed by the reliability of the selected
direction. This remains Tier-2 / stable-support / within-cell / tail-resolved continuous
score / not theorem-facing, but the external-transfer arc is now confirmed, replicated,
and mechanistically explained as far as the available Tier-2 target permits.
The consolidated claim boundary lives at
[`isotrophy/kfacet/kfacet_external_transfer_capstone.md`](isotrophy/kfacet/kfacet_external_transfer_capstone.md).

Post-v0.19 update: the reliability mechanism now has a first-principles cause, but the
locked verdict remains procedurally `spectral_gap_mechanism_partial`. H1 landed hard:
small Floquet Re-part spectral gap predicts high frame spread on the exact v0.18 rows
(`rho = -0.836`, `p = 1e-5`, reproduce gate exact, large-gap/high-spread falsifier
15/2880). That confirms the exportable **fragility** principle: an argmax-selected
shadow is reparameterization-fragile exactly where the selection spectrum is
near-degenerate. H2 missed because it used a per-cell **median** gap summary
(`rho = 0.063`), while v0.18's AUC relationship runs through the frame-spread tail
(`frame_p90`). So the seventh projection lesson is not "partial means weak"; it is
**well-posedness is spectral, but region-level transfer needs an aggregation matched
to the failure mode**. For chatv2 / Mesa / NSE, compute the eigengap or control-basis
gap first as a label-blind reliability predictor, then decide whether the downstream
claim is median-, tail-, or threshold-driven before collapsing a region to one number.

Post-v0.20 update: that open question is now tested and answered -- TAIL. v0.20 is a
confirmatory re-analysis (no new measurement) on the SAME 18 cells and SAME AUC: it swaps
v0.19's per-cell **median** gap for a pre-registered **low-tail** summary
`tail_gap_reliability = log10(q10(re_gap))` and lands `tail_gap_bridge_supported_confirmatory`.
The direct spectral-gap -> AUC bridge the median missed (`rho = 0.063`) returns at full
strength in the tail (`rho = 0.882`, `p = 1e-5`; leave-one-cell-out 0.863-0.922, so not
one-cell-driven; the coherence guard tail -> frame_reliability `rho = 0.707` confirms the
bridge runs through the reliability mechanism). The locked-primary q10 is the vindicated
sweet spot of tail depth: the deeper q05 (`rho = 0.589`) drifts toward min-outlier noise,
the shallower q20 (`rho = 0.360`) drifts back toward the median's wash-out. So the export
sharpens to its operational form -- for an eigenvector/argmax-selected shadow, compute the
selection spectral gap (Floquet Re-gap / LLM residual-stream eigengap / Mesa control-basis
gap / NSE projection gap), then read its LOW TAIL across the region (the fragile
sub-population, not a central summary) as the label-blind a-priori reliability predictor.
Bounded as registered: confirmatory, same cells, not an evidence upgrade -- what it earns is
that v0.19's "partial" is now legible as a strong, direct, tail-aggregated spectral bridge.

chatv2 cross-check (2026-06-03): the bridge was tested on the **LLM residual-stream**
substrate it names, and chatv2 **contrasts** it. The chatv2 Phase-2
determining-shadow-set receipt (`docs/chatv2/PHASE2_DETERMINING_SHADOW_SET_RECEIPT.md`)
found per-latent readout directions **stable** within-seed (`frame_spread` median
0.12) with **no eigengap -> spread link** (`rho = 0.065`), yet cross-seed
non-transferable because the 8-dim readout subspaces are **near-orthogonal across
seeds** (`subspace_overlap = 0.036 ~= random k/d = 8/192`). So chatv2's non-transfer
is genuine near-random subspace placement, **not** the spectral-gap fragility
mechanism -- a *boundary* of the v0.20 law, not a 4th confirmation. Caveat: the simple
covariance top-eigengap may also just be the wrong discriminative-conditioning measure
on this substrate (a discriminative-gap variant is the registered refinement). Toy;
unpromoted.

### 7.3 Threebody

Phase 18 revises this row. The accelerometer / tidal proxy was a useful scaffold
for finding the high-velocity near-escape pocket, but it is not the load-bearing
projection for survival. The current mechanism reduces to radius crossing an
inherited guard threshold plus an inward reflex at matched duty.

The accelerometer / tidal proxy was previously treated as **the projection from 18-dim
full state to 3-dim control signal.** The Phase 13 pocket-of-operation
result is then a claim about *where in the orbit space that
projection preserves load-bearing structure* and where it does not
— exactly the body/shadow framing in §6.2, with the failure-map
boundaries (low velocity, equal mass, overhead light) marking the
edges of the projection's domain of validity.

Current reading: Phase 18 narrows that old interpretation. The load-bearing
projection is radius-gated geometry, and the failure-map boundaries (low velocity
and equal mass) mark the edges of that projection's domain of validity.

Post-Phase-18 action: future threebody specs can keep the existing "warning
quality / action coupling / outcome effect" decomposition, but they should not
describe the planar near-escape survival mechanism as sophisticated tidal
sensing unless a fresh locked phase earns that claim. For the current pocket,
the mechanism line is closed: radius-gated inward thrust explains survival.

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

## 8. Faraday / Maxwell: the exact anatomy of body-resistance

*(2026-05-31. Framing layer over receipts - same posture as §6: this section reads
the closed Faraday Branch A result and the now-closed Phase 7 boundary cases
through the projection / body-resistance vocabulary. Its one empirical claim - the
Aharonov-Bohm regime-2 separation - was pre-registered and is now earned by the
Phase 7 case 3 receipt; the rest is framing, tagged inline.)*

Faraday is the portfolio's only clean structural-zero substrate, and it earns a
place here because *why* it succeeded states the §6.3 body-resistance lesson as an
algebraic identity instead of a measured near-coincidence. Four readings, then the
reel-in.

### 8.1 Faraday closure = zero body-resistance, by identity [PROVED]

The Faraday law the lane zeroed out is the *homogeneous* Maxwell equation
`dF = 0`, which is the Bianchi identity `d(dA) = 0` (Shadow Faraday Phase 3,
work-order step 2). So the body<->shadow closure is a geometric tautology: the
plaquette-holonomy shadow reconstructs the body because the holonomy of an exact
form around a contractible loop vanishes identically. In §6.3 terms,
**body-resistance is exactly zero, and zero by theorem** - not by low dimension.
Faraday is therefore the *exact-zero anchor* of the body-resistance continuum
whose marginal interior is NSE-C1 (`FVE ~ 0.99`, resistance near-zero by accident
of low dimension). Same lesson; Faraday states it as `0`.

This reframes the "freebie" intuition honestly: Faraday was not classical EM
cooperating with Sundog, it was Sundog landing on the half of Maxwell that is an
identity. A clean zero on an identity is a correctness check on the operator
(`P_shadow` really is gauge-invariant and local, and it really does close the
loop) - it is not a regime-2 separation, because there was never a resisting body
to separate from.

### 8.2 Aharonov-Bohm = exact *topological* body-resistance [EARNED 2026-05-31 - Phase 7 case 3 receipt]

The same operator's *other* tier produces the opposite corner. On a
non-contractible patch (`H^1 != 0`), take the loop-holonomy tier
`P_shadow^stencil = ∮A` as the shadow:

- **State-insufficient.** The enclosed flux `Φ = ∮A` is one number per homology
  class; it cannot rebuild the interior field `B(x)`. Many bodies map to the same
  shadow.
- **Control-sufficient.** For the registered objective "predict the
  Aharonov-Bohm interference shift," `Φ` is *exactly* sufficient (phase
  `= qΦ/ħ`).
- **The local tier goes blind.** `F = 0` along every electron path, yet the phase
  is nonzero - so `P_shadow^point = F` is control-*insufficient* here.

That is a regime-2 separation (state-insufficient yet control-sufficient) that is
**sharp and exact**, and it resists along a **topological / cohomological** axis,
not the dimensional axis of §6.3. It would be the first *exact* (non-marginal,
non-learnability-walled) regime-2 point anywhere in the portfolio - the clean
opposite corner from the three marginal control substrates and the chatv2
dimensional-axis result (a de-confounded high-dim body that resists and scales,
`d_dec` 7→14 — seed-robust, with a robust objective-driven contrast (0.205±0.022,
3 seeds); an at-the-bar SHARP, still `d_dec<20`; toy).

Honest bounds. The resisting body is "small" - one integer per `H^1` generator,
not a high-dimensional state - so AB does **not** close the open frontier of §6.3
(a high-dimensional *control* body on a real substrate); it is a different *kind*
of resistance, not a bigger one. It is now **earned**: the pre-registered Phase 7
case 3 ([`faraday/FARADAY_PHASE7_SPEC.md`](faraday/FARADAY_PHASE7_SPEC.md)) landed
as **B7-topology** in
[`faraday/FARADAY_PHASE7_BOUNDARY.md`](faraday/FARADAY_PHASE7_BOUNDARY.md) on
2026-05-31: `F = 0` on the loop, `oint A = Phi`, the two-tier divergence equal to
the `H^1` invariant `Phi` (numeric mirror `npm run faraday:phase7`).

Corollary it pins down. AB is the *exact boundary* of the Faraday headline.
"Local gauge-invariant shadow suffices for Faraday induction" is true (Branch A)
right up to topology, where it fails and the loop observable becomes mandatory.
Phase 7 case 3 is therefore not a quarantine to file - it is the boundary-definer
for the lane's central claim.

### 8.3 The U(1) -> Yang-Mills bridge: one operator, two portfolio outcomes [FRAMING]

`P_shadow^stencil` is a U(1) Wilson plaquette. Maxwell is the *abelian* baby case
of Yang-Mills, which sits in the failure map above as BOUNDED-NULL. The abelian
loop closes cleanly because `dF = 0` is linear and the holonomy of an exact form
vanishes; the non-abelian Bianchi `DF = dF + A^F = 0` drags the connection into
the covariant derivative and the gauge-invariant content (Wilson loops, area law)
becomes genuinely harder to separate. So **Faraday-clean and YM-bounded-null are
the abelian and non-abelian ends of the same operator** - and the EM side supplies
an *algebraic* reason for the YM upper limit: the very thing that made EM a
freebie (linearity + exact-form holonomy) is exactly what the non-abelian case
lacks. This is now written up as the
[Faraday -> Yang-Mills bridge note](yang-mills/2026-05-31_faraday_abelian_bridge_note.md) -
the first cross-substrate result that *explains a portfolio null from an adjacent
success* - and is folded into the YM external-review packet's Q2. ->
[`SUNDOG_V_YANG_MILLS.md`](SUNDOG_V_YANG_MILLS.md).

### 8.4 Homogeneous / inhomogeneous = topology / metric - where Maxwell-proper lives [PRE-REGISTERED PREDICTION]

Everything above is the *homogeneous* half of Maxwell (`dF = 0`, metric-free,
topological). Maxwell-proper adds the *inhomogeneous* half `d*F = J`, where the
Hodge star `*` brings in the **metric** and with it the **dynamics and sources**.
That is the only side where a *control* regime-2 could be non-trivial - and the
§6.3 three-for-three pattern gives a pre-registered prediction: **likely
marginal**, because classical sourced EM on a patch is nearly fixed by its sources
(the retarded map `J -> F` is essentially invertible modulo homogeneous
solutions). Phase 8 executed and landed
([`faraday/FARADAY_PHASE8_BOUNDARY.md`](faraday/FARADAY_PHASE8_BOUNDARY.md),
2026-05-31), earning the sharpened verdict: the sourced sector is *determined* by
its sources (uniqueness theorem) so it furnishes no new sharp regime-2, and the
Hodge decomposition localizes ALL exact regime-2 content to the harmonic sector -
exactly the Phase 7 Aharonov-Bohm witness. So Maxwell-proper adds **no new sharp
separation** (A8 + B8): a four-for-four-marginal result, disaggregated
(determinacy, not low-dim dynamics) - which is *why* Aharonov-Bohm was the unique
exact witness. The dual-shadow closure `oint *F = Q_enc` recovers Gauss and
Ampere-Maxwell (incl. displacement current); the metric (Hodge `*`) enters here and
only here.
(Phase 7 already corrected the common error here: ordinary electric sources live
on the `d*F = J` side and leave `dF = 0` intact - that correction *is* the
Faraday/Maxwell boundary, stated.)

### 8.5 Why this is the algebraic / cosmological reel-in

The Faraday/AB pair converts the body-resistance story from measured
near-coincidences into *exact statements*: resistance `= 0` by Bianchi (§8.1),
resistance exact-and-topological by `H^1` (§8.2), the abelian/non-abelian boundary
of one Wilson operator (§8.3), and the topology-vs-metric split (§8.4). It is the
most algebraic substrate in the portfolio and the only one giving *exact*
both-ended statements of the axis the rest of this document only measures. The
homogeneous/inhomogeneous decomposition is the template for all gauge theory and
for how fields couple to geometry; the topological side (helicity, flux, linking)
is the same mathematics that organizes solar plasma - a quiet closure back to the
sundog the program is named for. That last is flavor, not a claim.

### 8.6 What this section does and does not license

- **Does:** add Faraday (identity-success) and Aharonov-Bohm (exact-separation)
  to the failure map and the §6.3 table; name the *topological* resistance axis
  as distinct from the *dimensional* one; and carry the AB regime-2 claim, earned
  by Phase 7 case 3 ([`faraday/FARADAY_PHASE7_BOUNDARY.md`](faraday/FARADAY_PHASE7_BOUNDARY.md),
  2026-05-31).
- **Does not:** assert any Maxwell-proper result (that is a future Phase 8); or
  change any public Faraday copy. The Faraday lane's public claim remains the
  registered Branch A clean structural zero on the classical-vacuum,
  magnetically-clean, contractible domain.

## 9. Agentic trace: the override is body-determined, shadow-resisted [EMPIRICAL, BOUNDED]

The determine/resist split (§6.2) showed up, unforced, in a live LLM. In the
sundogcert agentic-trace lane, the H-I falsifier proved (in Lean,
`no_word_function_determines_decisive`) that **which source is *decisive* is not a
function of the observable** — it must be externally bound. A 2026-06 empirical
campaign on hosted and local models then asked whether a deployed model *behaves*
that way (`Dev/sundogcert/AGENTIC_TRACE_H1_CLIFF_TRANSFER_RUN_NOTE.md`).

The task: one authoritative source states a (made-up) fact; a fraction λ of plausible,
**unflagged** field reports contradict it. As λ rises past the majority-crossover the
model can abandon the decisive source for the volume — the live analog of the RS
falsifier (the *decisive minority pruned by the majority*). It does: gpt-3.5-turbo and
even Qwen-0.5B adopt the contradiction as volume grows (graded, not a sharp cliff —
verdict K1).

The determine/resist content is in **what carries the override signal**:

- **The output-shadow RESISTS it.** First-token entropy and self-consistency — the
  cheap *observables* — predict a trial's override at **AUC ≈ 0.50** (chance), on the
  same model, same trials. The model overrides *confidently*; its output gives no tell.
- **The body DETERMINES it.** A cross-validated linear probe on the *internal* state
  (last-prompt-token hidden vector, Qwen-0.5B) predicts the same override at **AUC 0.99**
  (deeper layers crisper, 0.74 → 0.99). No leakage: the probe reads the pre-generation
  state; the override is the post-generation answer that state already fixes.

So the agentic override is **body-determined, shadow-resisted** — the same shape as the
Shadow-Invertibility tower (the lossy shadow *determines* some properties, *resists*
others), now in a transformer residual stream. It is the exact **complement** of the
Lean result: decisiveness is not in the observable (entropy ≈ chance ✓), but it is in
the body (probe ≈ 1.0). The theorem bounds the *shadow*; the body is not the shadow, so
the theorem does not bound it.

**What this licenses.** A precise, falsifiable bound on monitor-side syndrome-gating:
to gate the decisive-source override you must **reach inside (white-box probe) or be
told (external `decisive_indices` binding)** — you cannot read it off the output. This
does **not** claim a general interpretability result: it is a bounded probe (0.5B model,
n = 12/λ; a higher-n and larger/Mamba confirmation is staged but unrun), and it does not
touch the §6.5 compander citation gate. It is recorded here because it is the first
*live-substrate* instance of the determine/resist machinery this section names.
