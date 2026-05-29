# Sundog Navier–Stokes Ledger

Working hook:

> We do not solve Navier–Stokes. We ask whether the shadow of a flow is
> enough to detect the loss of regularity before standard energy
> diagnostics flag it.

Sundog Navier–Stokes is the staging ledger for the program's coupling
between Sundog's sufficient-statistic discipline — Coarse-Graining
Postulate 1, signature-based control, shadow-projection readout — and
the open mathematical front of incompressible Navier–Stokes existence
and smoothness as a Clay Millennium problem.

This document is not a roadmap. It is a holding pattern for ambition.
Candidate 1 has now been commissioned into a proof-track draft; the other
candidates have not run. Each, if executed, would either ratchet the coupling
claim into earned language or push it back to the "out-of-scope analogy" pile in
[`presentation/claims-and-scope.md`](presentation/claims-and-scope.md).

The closest existing anchors are the three-body workbench
([`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md)) — already established
that compressed indirect signatures detect near-singular dynamical
events before they resolve in full phase space — and the
Coarse-Graining Proof Roadmap
([`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md)) —
which states the Postulate-1 measurability predicate generally. NSE is
the natural PDE-substrate leg, parallel to the three-body measured leg
already in Phase 4 of that roadmap.

The lit-pass record that informs the candidate ranking below is
[`NAVIERSTOKES_LITPASS_MEMO.md`](NAVIERSTOKES_LITPASS_MEMO.md) (filed
2026-05-28). The ledger does not re-derive that memo's gap findings; it
cites them by pointer.

### Promotions

No candidates have been promoted out of this ledger yet. The block is
retained for symmetry with [`SUNDOG_V_CAPSET.md`](SUNDOG_V_CAPSET.md)
and will be populated when a candidate graduates into a committed
artifact (proof note, empirical workbench, or public reading).
Promotion criteria are pinned below.

Commissioned but **not promoted**: Candidate 1 is drafted at
[`proof/PDE_DETERMINING_MODES_POSTULATE1.md`](proof/PDE_DETERMINING_MODES_POSTULATE1.md),
with a desk-auditable cell-set v0 instance at
[`proof/PDE_C1_CELLSET_KOLMOGOROV.md`](proof/PDE_C1_CELLSET_KOLMOGOROV.md)
and a tolerance / binning fiber protocol at
[`proof/PDE_C1_FIBER_PROTOCOL.md`](proof/PDE_C1_FIBER_PROTOCOL.md).
The v0 comparator has been tightened: the state-insufficiency side now rests on
finite-Galerkin non-injectivity, while the determining-mode literature supplies
only the state-reconstructive reference ceiling. The fiber protocol closes the
continuous-fiber gate by pinning a tolerance object plus a runnable bin-lattice
adjudication procedure, names the cell-set v0 selector as a proxy with
substitution conditions, bridges the support-level certificate as a separate
follow-up, and introduces the **`PDE-C1-NEG-A` / `PDE-C1-NEG-B`** parallelism
that mirrors the C2 receipts (`PDE-C1-NEG` reads retroactively as
`PDE-C1-NEG-A`). The cell-set v0 patch landed 2026-05-28 (cell-set v0
section 7): `epsilon_K = 0.05 sqrt(2 E_max)`, `h_K = epsilon_K / sqrt(d_K)`,
`n_min = 30`, `delta_action = 0.10`, `S_pos = 0.50`, `N_sample = 50,000`,
sampling interval `0.5` time units, integration step `dt = 0.01`, action
tie-break favouring `no_op`, with a pre-registered single fall-back to
`N_sample = 200,000` if the coverage gate defers. **Lock execution
2026-05-28** via `scripts/pde_c1_kolmogorov_cell.py --preset lock` (24 min,
~1776 steps/sec; receipt at `results/proof/c1-kolmogorov-v0-lock/`)
returned a procedural `STRICTNESS_WITNESS_POSITIVE` that was substantively
non-discriminative: `damp_fraction = 0` (0 / 50,000 samples crossed
`E_max`), `sample_energy_max = 0.09761 < E_max = 0.09765`, 70% of samples
concentrated in a single fiber bin. Diagnosis: 2D Kolmogorov flow at
`G = 100`, `k_f = 4` is linearly stable; the system sits at the laminar
fixed point. **Fiber protocol amended same day** with a vacuity gate
(`delta_proxy_min = 0.01`, new `DEFERRED_VACUITY` non-verdict branch
parallel to `DEFERRED_COVERAGE`); v0 lock receipt re-reads as
`DEFERRED_VACUITY` and is **not** promoted as a positive. The amendment
closes a verdict-rule completeness gap (a constant proxy can never
separate fibers); it is methodology, not a `PDE-C1-NEG-B` retune.
Amendment is wired into the harness (`summarize` / `write_receipt`).
**Cell-set v1** at
[`proof/PDE_C1_CELLSET_KOLMOGOROV_v1.md`](proof/PDE_C1_CELLSET_KOLMOGOROV_v1.md)
re-pinned `k_f = 2` (supercritical Kolmogorov at `G = 100`). **Lock_v1
executed 2026-05-28** in ~18 min (receipt at
`results/proof/c1-kolmogorov-v1-lock/`). Procedural verdict
`DEFERRED_COVERAGE` (`S_eval = 0.4510 < S_pos = 0.50`) but **also
structurally vacuous**: `damp_fraction = 0` exactly because the
post-transient attractor conserves `E_K` to 13 decimal places (1397
occupied bins, non-trivial in signature space, but the safety predicate
is inaccessible). Burn-in had real transient excursions to 2.5× the
mean — v1 escaped v0's laminar trap — but post-burn-in locks onto an
energy-conserved invariant set. **Same-day protocol amendment** adds a
structural-vacuity-precedence rule (when `damp_fraction` is exactly `0`
or `1`, vacuity gate overrides coverage gate, since increasing
`N_sample` cannot resolve a structurally zero proxy). v1 lock receipt
re-reads as `DEFERRED_VACUITY` under the amended rule. Standard
fall-back **not admissible** — `N_sample = 200,000` cannot shake loose
an energy-locked attractor.
**Cell-set v2** at
[`proof/PDE_C1_CELLSET_KOLMOGOROV_v2.md`](proof/PDE_C1_CELLSET_KOLMOGOROV_v2.md)
tripled Grashof to `G = 300` while keeping `k_f = 2`. **Lock_v2 executed
2026-05-28** in ~22 min (receipt at `results/proof/c1-kolmogorov-v2-lock/`).
Procedural verdict `DEFERRED_COVERAGE`. The energy-conservation
symmetry **was successfully broken**: post-burn-in spread is ~95% of
mean, `damp_low_band_count = 430` (not zero), `damp_fraction = 0.0086`
(structurally non-zero, so structural-vacuity precedence does *not*
fire). But the regime is high-dimensional: 45,103 occupied bins from
50,000 samples (~1.1 per bin), `S_eval = 0`. Two intertwined issues —
curse-of-dimensionality coverage collapse + near-vacuity at
`damp_fraction = 0.0086 < delta_proxy_min = 0.01`. Honest receipt
under the pinned methodology; the binning prescription `h_K = ε_K /
sqrt(d_K)` adapts to `E_max` but not to attractor extent.
**Cell-set v3** at
[`proof/PDE_C1_CELLSET_KOLMOGOROV_v3.md`](proof/PDE_C1_CELLSET_KOLMOGOROV_v3.md)
backed Grashof off to `G = 200`. **Lock_v3 executed 2026-05-28** in
~20 min; verdict `DEFERRED_VACUITY` (`proxy_selector_structurally_constant`).
Burn-in had real chaotic excursions (max 3.13) but the post-burn-in
steady-state attractor sits in `[0.7649, 0.7843]`, well below the
transient-contaminated `e_max = 1.07`. **Cross-cell pattern** across
v0 (laminar), v1 (energy-conserved), v3 (steady-state below threshold):
3 of 4 cells have `damp_fraction = 0` *exactly*, and v2's was only
`0.0086`. The shared signal: burn-in-derived `E_max` sits above the
steady-state attractor's typical lookahead excursions.
**Methodology amendment 2026-05-28** (fiber protocol §3): cells may
pin `e_max_burnin_fraction ∈ (0, 1]` and compute `E_max` from the
last fraction of burn-in (excluding transients). Default `1.0`
preserves v0–v3 backwards compatibility.
**Cell-set v4** at
[`proof/PDE_C1_CELLSET_KOLMOGOROV_v4.md`](proof/PDE_C1_CELLSET_KOLMOGOROV_v4.md)
is the first **methodology sibling** (not regime sibling) in the C1
chain. Holds v3's regime constant (`k_f = 2`, `G = 200`) and pins
`e_max_burnin_fraction = 0.25` (last 25% of burn-in). **Lock_v4 and
fallback_v4 both executed 2026-05-28** (~19 min + ~88 min; receipts at
`results/proof/c1-kolmogorov-v4-lock/` and `.../v4-fallback/`). Both
returned `DEFERRED_COVERAGE`, but with a clean two-sided diagnosis:
**Methodology win** — E_max amendment unlocked proxy discrimination
(`damp_fraction = 0.30014` lock / `0.300125` fall-back, stable to 4
dp across 50k and 200k samples; structural-vacuity precedence did not
fire; the cell IS substantively discriminative). **Methodology limit**
— 4× samples produced 3.04× bins (45,827 → 139,361) with
`evaluated_bin_count = 0` in both runs; uniform binning at `K = 4`
empirically infeasible on this attractor regardless of `N_sample`.
**Methodology amendment 4 (added 2026-05-28)** to fiber protocol §2:
**`K` (signature mode count) becomes a cell-set parameter**, with
explicit documentation of `bin_count ∝ (R/h_K)^{d_K}` where
`d_K = 2K^2`. Lowering K reduces dimensional cost exponentially.
**Cell-set v5** at
[`proof/PDE_C1_CELLSET_KOLMOGOROV_v5.md`](proof/PDE_C1_CELLSET_KOLMOGOROV_v5.md)
held the v4 regime and E_max windowing constant and pinned `K = 3`
(signature dim 18 vs v4's 32). **Lock_v5 executed 2026-05-28** (~19 min);
`DEFERRED_COVERAGE`. Two findings: discrimination robust
(`damp_fraction = 0.2977`, matching v4) **and** the K-reduction
coverage hypothesis (amendment 4) **falsified** — occupied bins fell
only 16.5% (45,827 → 38,281) despite halving `d_K`. Root cause:
occupied-bin count tracks the attractor's box dimension, not the
embedding dimension `d_K`. The amendment-4 scaling note in the fiber
protocol is annotated as corrected.

**Step-back synthesis** at
[`proof/PDE_C1_LOCK_EXECUTION_SYNTHESIS.md`](proof/PDE_C1_LOCK_EXECUTION_SYNTHESIS.md)
consolidates v0–v5 (six runs, four amendments, all `DEFERRED_*`).
**Finding A (positive, durable):** a discriminative cell exists at
`(k_f = 2, G = 200)` — `damp_fraction ≈ 0.30`, robust to `K` (3 vs 4)
and `N_sample` (50k vs 200k); a low-dimensional signature is robustly
control-discriminative on a known-smooth 2D NSE attractor. This
supports Reading 2's premise (the safety objective is non-trivially
action-varying), without yet adjudicating fiber-constancy.
**Finding B (negative):** hard-bin fiber adjudication has a
tolerance-fidelity vs. coverage tension governed by attractor measure
concentration; six cells bracket a window hard binning does not fit at
feasible sample budgets. **Forks (no further binning cell):** (A)
pivot to kNN/kernel fiber-locality adjudication (principled fix); (B)
report Finding A as a partial C1 read; (C) bank C1 lessons and pivot
to C2 (structurally insulated from this tension).

**Fork A executed 2026-05-28 — first interpretable C1 read, a
provisional POSITIVE.** The kNN / disintegration adjudicator
(fiber-protocol §5b; design at
[`proof/PDE_C1_KNN_ADJUDICATION_DESIGN.md`](proof/PDE_C1_KNN_ADJUDICATION_DESIGN.md))
was built (`--adjudicator knn`, exact `BallTree`) and run at the v4
regime. It **solved the coverage obstruction** (`fidelity_coverage =
1.0` vs binning's 0) and fired a mechanical `PDE-C1-NEG-A`
(`incompat_fraction = 0.0716`). That NEG-A was **held provisional and
then overturned** by a pre-registered convergence check
([`proof/PDE_C1_KNN_CONVERGENCE_CHECK.md`](proof/PDE_C1_KNN_CONVERGENCE_CHECK.md)):
the first sweep's NEG-A was an artifact of a coverage-failing `k=100`
point contaminating the OLS, and the thresholded statistic carried a
grain confound. The **amended** check (coverage-passing points only;
threshold-free `mean_minority` primary; dense low-`k` sweep) returned
**`STRICTNESS_WITNESS_POSITIVE`**: `mean_minority ≈ 0.70·r_k` cleanly
through the origin (`a_mm = −0.00125 ≤ 0.005`), with three robustness
checks passing (through-origin not plateau; 10–25× below the
random-label level so labels are spatially organized; threshold-free
and thresholded statistics agree). On the v4 cell the low-band-energy
proxy is **control-sufficient on `Phi_K`-fibers** up to a measure-zero
decision surface, with `damp_fraction = 0.30` (non-trivial) — the C1
sidecar's Reading-2 **regime 2** (state-insufficient, control-
sufficient), the non-vacuous Sundog target.

This is **provisional and tightly scoped**: one cell (v4 / K=4 / this
objective); the state-insufficiency-on-attractor half still needs the
deferred twin-state certificate (§4.1 certifies non-injectivity on
`B_abs`, not `supp(mu_SRB)`); `\hat{pi}` is a proxy pending reviewer
acceptance or a derived `J`-optimal selector; the test resolves to
`r_k ≈ 0.02`. It strongly advances C1 but does **not** promote it.

Criterion (b) is closed at the *procedure-and-execution* level (v0–v5
lock cycles + v4 fall-back + the kNN run and its convergence check all
ran cleanly; methodology survived six vacuity / coverage diagnoses, a
contaminated-then-overturned NEG-A, and five pre-registered amendments
— two of them falsified-or-corrected — without any `PDE-C1-NEG-B`); a
*substantive fiber-adjudication* C1 read is now **delivered
provisionally** by the Fork-A POSITIVE. Criterion (a) Front-A vacuity
rebuttal is materially advanced (a concrete regime-2 witness shows the
Postulate-1 reading is non-vacuous on a real substrate), pending
reviewer judgement. Criterion (c) external PDE reviewer remains open,
as do the twin-state certificate and proxy-faithfulness items. Final
(d) close is coupled to (c).

Candidate 2 has also been commissioned at the scoping level:
[`proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md`](proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md)
pins the research object, the Sabra-primary + GOY-cross-check model commitment,
the tiered channel taxonomy (Tier 0 headline; Tiers 1–2 ablations), the
matched-budget baseline rule, the burst-prediction task, and the two-sided
negative as `PDE-C2-NEG-A` (Pareto vacuity) and `PDE-C2-NEG-B` (overfit /
cell-set drift). Cell-set v0 with concrete numerics is deferred to a
follow-up artifact, mirroring the C1 staging.

## Claim Boundary

This document does **not** claim that Sundog has produced original
mathematics on Navier–Stokes existence or smoothness. It claims that:

1. there is a coherent structural argument — that Sundog's
   sufficient-statistic discipline, built to evaluate
   indirect-signal claims under partial observability, is
   well-shaped to *read* and *complement* a specific class of
   existing PDE results (determining-modes / data-assimilation,
   Onsager-type coarse-graining thresholds, shell-model
   intermittency invariants) that already share its epistemic
   structure;
2. that argument is currently defended by analogy and by the
   three-body sibling result, not yet by a Sundog-authored note,
   reading, or evaluator output on any PDE result;
3. the proof targets that would test the argument are non-trivial
   enough that they need to live in a ledger before they live in a
   roadmap.

If a candidate below is promoted, it leaves this ledger.

Avoid broader formulations such as:

- "Sundog solves Navier–Stokes."
- "Signatures replace energy methods."
- "Coarse-graining proves regularity."
- "PINNs miss what signatures see."
- "Determining modes are Sundog's idea."

## The Coupling Claim

The coupling between Sundog's apparatus and the open NSE front is
staged on two fronts, in the
[`SUNDOG_V_CAPSET.md`](SUNDOG_V_CAPSET.md) idiom.

### Front A — Reading / instrument complement (defensible)

The 2024 synchronization-framework paper
([arxiv 2408.01064](https://arxiv.org/abs/2408.01064)) — see
[`NAVIERSTOKES_LITPASS_MEMO.md`](NAVIERSTOKES_LITPASS_MEMO.md) Track B —
formalizes determining modes as *state reconstruction* via observer
synchronization, not as sufficiency for a control objective. The
Coarse-Graining Postulate 1 predicate
([`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md)
§1) names the sharper, smaller object: `𝓕_σ`-measurability of `π*`,
which is Blackwell-sufficiency for the control objective rather than
for state reconstruction.

Front A stages the claim that re-reading the determining-modes program
through Postulate 1 produces a non-vacuous distinction — there are
NSE regimes where state-reconstruction sufficiency holds but
control-sufficiency does not, or vice versa, and the gap is operational.
The product is a reading note plus a pre-registered cell-set where the
distinction is measurable, not a new theorem.

This front is defensible now in the sense that no
Sundog-original PDE mathematics is claimed.

### Front B — Empirical leg on a tractable surrogate (horizon)

A more ambitious coupling: shell models of turbulence (Sabra, GOY) are
infinite-dimensional dynamical systems that retain the NSE energy
cascade and intermittency while remaining laptop-tractable. The 2022
hidden self-similarity result
([arxiv 2201.04005](https://arxiv.org/abs/2201.04005)) gives a
deterministic invariant of cascade intermittency; instanton importance
sampling
([arxiv 2308.00687](https://arxiv.org/abs/2308.00687)) gives a labeled
rare-event dataset.

Front B stages the claim that a Sundog-style signature derived from
shell-model trajectories detects imminent intermittent bursts inside
a pre-registered operating envelope, with regret → 0 against an
oracle baseline and bounded-away-from-zero outside the envelope
(matched-seed, in the
[`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md)
Phase-4 idiom).

This front is *not* defensible now; the candidates below describe
what would need to happen for it to become defensible.

### Bonus precedent

[Onsager's conjecture](https://link.springer.com/article/10.1007/s00222-024-01291-z)
is the existing PDE-side precedent for "coarse-graining sets a
regularity threshold" — energy conserved at C^α for α > 1/3, dissipation
admissible below. Giri–Radu 2024 and Du–Li–Ye 2025
([arxiv 2506.15396](https://arxiv.org/abs/2506.15396)) settle pieces of
the flexible side. This is the existing pure-math precedent that makes
the Postulate-1 measurability-threshold framing not analogy but sibling.
Cited so the framing is not read as imported metaphor.

## Falsification Surface

The coupling claim can fail in five named modes (one more than the
capset ledger, reflecting that NSE has both an algebraic and an
empirical attack surface):

1. **Front-A vacuity.** The Postulate-1 reading of determining modes,
   once written, reduces to what any careful PDE analyst would already
   say. The instrument has no edge over a good review.
2. **Front-A miscalibration.** The Postulate-1 / Blackwell-sufficiency
   distinction lands on confidently wrong readings (e.g., claims a
   control-sufficient signature where the determining-modes literature
   already pins state-reconstruction sufficiency at a smaller mode count
   that suffices for control too).
3. **Front-B vacuity (signature baselines).** On shell models, a
   matched-budget DMD / critical-slowing-down / lacunarity / Rényi
   baseline detects intermittent bursts at least as well as the
   signature-based detector. Signatures add no signal.
4. **Front-B reach (PDE extrapolation).** Even a clean shell-model
   positive result silently smuggles in finite-dimensional truncation
   structure that does not survive the move to full 2D-perturbed or 3D
   small-data NSE. Cross-substrate sameness fails at the substrate
   boundary.
5. **Coupling overreach.** The ledger is published alongside a public
   workbench, reading, or claim that elevates a Front-A reading or
   Front-B shell-model result into a Clay-problem-adjacent posture.
   Detected by [`presentation/claims-and-scope.md`](presentation/claims-and-scope.md)
   audit on any deploy that touches NSE vocabulary.

Each candidate below has to declare which modes it attacks.

## Evaluation Criteria

A candidate is worth pursuing if it:

- attacks at least one failure mode above with a falsifiable artifact;
- can be written without claiming Sundog-original PDE mathematics, *or*
  has a clear external-mathematician sanity-check path if it does;
- produces a deliverable that could plausibly land on the
  application-gallery or `/geometry.html` page alongside the other
  sundog workbenches rather than a standalone diversion;
- is honest about its dependence on prior work — determining-modes
  literature, Onsager construction, shell-model machinery, PINN
  baselines — and references it explicitly.

## Promotion Criteria

A candidate leaves this ledger and becomes a committed artifact
(proof note, workbench, paper section) when **all four** are true:

- (a) the Front-A vacuity check is explicitly rebutted (the artifact
  produces a claim or reading that is *not* what a careful PDE analyst
  would already say without the Sundog apparatus);
- (b) it has a runnable Phase-0 deliverable consistent with the existing
  Sundog harness pattern (pre-registered cell set, matched-seed
  baselines, named pre-registered negative);
- (c) it has an external-mathematician sanity-check path named in the
  promotion note;
- (d) it has a pre-registered failure boundary fixed before any
  numerical reads are interpreted, per
  [AGENTS.md ▸ "Pre-register the negative"](../AGENTS.md).

## Candidate List

Candidate 1 has been commissioned as a proof-track reading draft. No empirical
candidate has run.

### Candidate 1 — Postulate-1 reading of determining modes

- **Attacks failure mode(s):** 1 (Front-A vacuity), 2 (Front-A miscalibration).
- **Cost:** Low (a short note, ≤ ~4 pages).
- **Front:** A (reading).
- **Sketch.** Re-derive the Foias–Temam–Constantin determining-modes
  / nodes / volumes result as a `𝓕_σ`-measurability statement in the
  [`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md)
  Phase-1/2 idiom. Make explicit where the literature's
  state-reconstruction sufficiency *implies* control sufficiency
  (the bulk of the regimes), and identify a regime — if any —
  where control sufficiency is *strictly weaker* than state
  sufficiency, in the sense that a smaller signature suffices for the
  control objective.
- **Pre-registered negative.** If every regime where state
  sufficiency is open also has control sufficiency strictly equivalent
  to state sufficiency under the standard data-assimilation gauge, the
  Postulate-1 reading is vacuous on NSE — record, do not rescue.
- **Draft artifact.** Filed at
  [`proof/PDE_DETERMINING_MODES_POSTULATE1.md`](proof/PDE_DETERMINING_MODES_POSTULATE1.md)
  as a C1 reading note, not yet a promotion. Cell-set v0
  ([`proof/PDE_C1_CELLSET_KOLMOGOROV.md`](proof/PDE_C1_CELLSET_KOLMOGOROV.md))
  pins the pre-registered failure boundary at a Kolmogorov-flow regime.
- **Cross-files.** Would update
  [`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md)
  Phase 2 with a PDE corollary (or a named-negative).
- **External review path.** A practicing PDE analyst familiar with
  determining-modes work (Foias, Titi, Olson collaborators). Named at
  promotion.

### Candidate 2 — Signatures on shell-model intermittency (Sabra / GOY)

- **Attacks failure mode(s):** 3 (Front-B vacuity), 4 (Front-B reach).
- **Cost:** Low-Med (laptop simulation; baselines off-the-shelf).
- **Front:** B (empirical).
- **Sketch.** Generate matched-seed Sabra / GOY trajectories with
  imminent / non-imminent intermittent bursts (labels via instanton
  importance sampling and rare-event statistics). Train a path-signature
  detector on a fixed channel set
  (e.g. shell-wise energy `|u_n|²`, transfer rates,
  hidden self-similarity coordinates per
  [arxiv 2201.04005](https://arxiv.org/abs/2201.04005)).
  Compare matched-budget against DMD / Koopman, critical-slowing-down,
  recurrence lacunarity, and Rényi-entropy detectors. Pre-register cell
  set, signature truncation depth, baseline hyperparameters, and the
  decision threshold.
- **Pre-registered negative.** Two-sided. (a) If the
  signature detector's burst-detection lead-time and false-positive
  rate are not strictly inside the Pareto frontier of the matched-budget
  baselines on the pre-registered cell set, signatures are vacuous on
  shell models — record and stop. (b) If signatures match baselines on
  the registered cell set but the cell set was the entire span, the
  result is over-fit to the data — record as Front-B vacuity.
- **Scoping artifact.** Filed at
  [`proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md`](proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md)
  2026-05-28: drafted scoping, model commitment (Sabra primary + GOY
  cross-check), tiered channel taxonomy, matched-budget baseline rule,
  burst-prediction task definition, and named negatives `PDE-C2-NEG-A`
  (Pareto vacuity) and `PDE-C2-NEG-B` (overfit / cell-set drift). Cell-set
  v0 (concrete `N_shells`, `λ`, `ν`, `τ_burst`, `E_burst`, signature
  truncation `D`, operating point) deferred to a follow-up artifact.
- **Workbench surface.** Would live next to
  [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md) as the
  PDE-substrate empirical sibling. No public deployment until external
  review.
- **External review path.** A practicing shell-model researcher
  (Mailybaev / Biferale / Frisch tradition). Named at promotion.

### Candidate 3 — PINN regularity-confidence diagnostic head

- **Attacks failure mode(s):** 4 (Front-B reach, indirectly), 5
  (coupling overreach guard).
- **Cost:** Med (requires a working PINN-NSE pipeline as input).
- **Front:** B (empirical) with application surface.
- **Sketch.** Take a PINN-predicted velocity field (off-the-shelf
  trained on a controlled NSE setup), compute a signature-based
  regularity-confidence score, pre-register the operating envelope
  inside which the score predicts the PINN's actual error on held-out
  ground truth. Fail cleanly outside.
- **Pre-registered negative.** If the diagnostic score is not Granger-
  or partial-information-gain-significant over the PINN's own residual
  on the pre-registered envelope, the head adds no signal — record and
  stop.
- **Workbench surface.** Optional public diagnostic widget if it lands.
  Not deployable until Candidate 1 or 2 lands first.
- **Deferred until Candidate 1 lands.**

### Candidate 4 — Full PDE signature transit detection on 2D-perturbed NSE

- **Demoted.** Subsumed in spirit by Candidate 2 at a fraction of the
  cost. Reconsider only if Candidate 2 returns a clean positive and an
  external reviewer recommends scaling the test to full PDE.

### Candidate 5 — Vorticity-stretching structural-zero invariant

- **Deferred.** Algebraic ansatz needs clarification before this can be
  named as falsifiable. The Shadow-Faraday Branch-A receipt
  ([`faraday/SHADOW_FARADAY.md`](faraday/SHADOW_FARADAY.md)) is the
  template — but the vorticity-equation analog isn't pinned yet.
  Re-open after Candidate 1 lands.

## Shortlist Recommendation

As of 2026-05-28, the recommended sequencing is:

1. **Candidate 1** (Postulate-1 reading of determining modes) — now
   commissioned as the active Front-A draft. It remains the lowest-cost,
   highest-leverage move; next gate is external PDE sanity check plus a
   runnable cell-set artifact.
2. **Candidate 2** (signatures on shell-model intermittency) — smallest
   empirical leg with the cleanest data path; depends only on standard
   shell-model machinery and off-the-shelf baselines. Scoping is now
   drafted at
   [`proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md`](proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md).
   Cell-set v0 and execution should follow Candidate 1's review outcome
   (`PDE-C1-NEG` → C2 may still run but is re-framed as a turbulence-signature
   detector only, *not* as evidence for a determining-modes coupling).
3. **Candidate 3** (PINN diagnostic head) — only after 1 and 2 land.

Candidates 4 and 5 are deferred pending the outcome of 1 and 2.

## Cross-references

- [`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md)
  — the trunk roadmap. NSE is the natural PDE-substrate leg parallel to
  the three-body measured leg in Phase 4. Candidate 1 is drafted at
  [`proof/PDE_DETERMINING_MODES_POSTULATE1.md`](proof/PDE_DETERMINING_MODES_POSTULATE1.md)
  (Phase-2 PDE corollary, pending review) and Candidate 2 is scoped at
  [`proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md`](proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md)
  (Phase-4 PDE-substrate empirical sibling, pending cell-set v0 and review).
  Neither updates the trunk roadmap before its respective review lands.
- [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md) — sibling empirical
  workbench. Different substrate, same hidden-indirect-signature-output
  pattern. The three-body Phase 13–15 results are the empirical
  precedent for "compressed signatures detect near-singular events
  before they resolve."
- [`SUNDOG_V_CAPSET.md`](SUNDOG_V_CAPSET.md) — sibling ledger and
  pattern source. This document follows its ledger structure
  deliberately so the two are auditable together.
- [`NAVIERSTOKES_LITPASS_MEMO.md`](NAVIERSTOKES_LITPASS_MEMO.md) —
  the 2026-05-28 lit-pass record that informs the candidate ranking
  here. Re-audit point: gap claims are time-stamped, not absolute.
- [`faraday/SHADOW_FARADAY.md`](faraday/SHADOW_FARADAY.md) — the
  structural-zero discipline source for the deferred Candidate 5.
- [`SCIENTIFIC_CRITERIA.md`](SCIENTIFIC_CRITERIA.md) — the research-
  object discipline this ledger inherits. Any promotion must satisfy
  the criteria there.
