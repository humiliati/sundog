# Proof Artifacts

This directory holds the internal proof-track artifacts for
[`COARSE_GRAINING_PROOF_ROADMAP.md`](../COARSE_GRAINING_PROOF_ROADMAP.md).

The proof track is not public-facing claim language. It is the working spine for
Postulate 1: when a signature is a sufficient statistic for control, and where
that claim provably fails.

## Current Artifacts

- [`POSTULATE1_DEFINITIONS.md`](POSTULATE1_DEFINITIONS.md) — Phase 0
  definitions lock for the Coarse-Graining Postulate.
- [`PHASE1_LQG.md`](PHASE1_LQG.md) — Phase 1 LQG proof (reviewed, closed
  positive) and Postulate 6 toy check.
- [`PHASE2_MDP.md`](PHASE2_MDP.md) — Phase 2 finite-MDP sufficiency proof
  (reviewed, closed positive), counterexample, and Formal Separability
  corollary.
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  — Navier-Stokes Candidate 1 sidecar: determining modes read through
  Postulate 1 (drafted; external PDE review and cell-set adjudication
  still pending).
- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) —
  Navier-Stokes Candidate 1 cell-set v0: Kolmogorov-flow instance at
  `k_f = 4`. Section 7 patches the fiber protocol's parameter slots
  (`epsilon_K`, `h_K`, `n_min`, `delta_action`, `S_pos`,
  `delta_proxy_min`, `N_sample`, burn-in length, integration step,
  action tie-break order); section 8 stages the harness; the
  **Lock Execution Disposition** section records the 2026-05-28 lock
  outcome — procedural `STRICTNESS_WITNESS_POSITIVE` superseded as
  `DEFERRED_VACUITY` under the same-day fiber-protocol amendment
  because Kolmogorov flow at this regime is linearly stable.
- [`PDE_C1_CELLSET_KOLMOGOROV_v1.md`](PDE_C1_CELLSET_KOLMOGOROV_v1.md)
  — Navier-Stokes Candidate 1 cell-set v1: discriminative-regime
  sibling of v0 at `k_f = 2`, `G = 100`. Lock executed 2026-05-28
  in ~18 min; procedural `DEFERRED_COVERAGE` superseded as
  `DEFERRED_VACUITY` under the same-day structural-vacuity-precedence
  amendment because the post-transient attractor conserves `E_K`
  exactly to 13 decimal places (non-trivial in signature space; 1397
  occupied bins; but `damp_fraction = 0` structurally).
- [`PDE_C1_CELLSET_KOLMOGOROV_v2.md`](PDE_C1_CELLSET_KOLMOGOROV_v2.md)
  — Navier-Stokes Candidate 1 cell-set v2: tripled Grashof to
  `G = 300` while keeping `k_f = 2`. Lock executed 2026-05-28
  (~22 min); procedural `DEFERRED_COVERAGE`. **Energy conservation
  successfully broken** (`damp_fraction = 0.0086`, real post-burn-in
  chaos at 95% of mean) — but the attractor is high-dimensional:
  45,103 occupied bins with 50,000 samples (~1.1 per bin),
  `S_eval = 0`. Diagnosed two-fold: curse-of-dimensionality coverage
  + near-vacuity at `damp_fraction = 0.0086 < delta_proxy_min = 0.01`.
  Not a `PDE-C1-NEG-B` retune; the pinned binning prescription
  doesn't adapt to attractor extent.
- [`PDE_C1_CELLSET_KOLMOGOROV_v3.md`](PDE_C1_CELLSET_KOLMOGOROV_v3.md)
  — Navier-Stokes Candidate 1 cell-set v3: `G = 200`, `k_f = 2`.
  Lock executed 2026-05-28 (~20 min); verdict `DEFERRED_VACUITY`
  (`proxy_selector_structurally_constant`) — burn-in had real chaotic
  excursions (max 3.13) but the post-burn-in steady-state attractor
  sits in `[0.7649, 0.7843]`, well below the transient-contaminated
  `e_max = 1.07`. Cross-cell pattern across v0/v1/v3 reveals burn-in
  transient contamination of the 95th-percentile threshold.
- [`PDE_C1_CELLSET_KOLMOGOROV_v4.md`](PDE_C1_CELLSET_KOLMOGOROV_v4.md)
  — Navier-Stokes Candidate 1 cell-set v4: methodology sibling to v3
  (`e_max_burnin_fraction = 0.25`, regime held constant). **Lock and
  fall-back both executed 2026-05-28** (~19 min + ~88 min). Both
  receipts `DEFERRED_COVERAGE`. **Methodology win**: E_max amendment
  unlocked proxy discrimination — `damp_fraction = 0.30014` (lock) /
  `0.300125` (fall-back), stable to 4 dp across 50k and 200k samples.
  Structural-vacuity rule did NOT fire. **Methodology limit**: at
  `K = 4`, 4× samples produced 3.04× bins (45k → 139k) with avg
  ~1.1–1.4 samples/bin; uniform binning empirically infeasible
  regardless of `N_sample`. Triggered fiber-protocol amendment 4
  (K as cell-set parameter).
- [`PDE_C1_CELLSET_KOLMOGOROV_v5.md`](PDE_C1_CELLSET_KOLMOGOROV_v5.md)
  — Navier-Stokes Candidate 1 cell-set v5: methodology sibling to v4,
  `K = 3` (signature dim 18 vs v4's 32). **Lock executed 2026-05-28**
  (~19 min); `DEFERRED_COVERAGE`. Two findings: discrimination robust
  (`damp_fraction = 0.2977`, matching v4) **and** the K-reduction
  coverage hypothesis **falsified** — occupied bins fell only 16.5%
  (45,827 → 38,281) despite halving `d_K`. Root cause: occupied bins
  track attractor box dimension, not embedding dimension `d_K`.
- [`PDE_C1_LOCK_EXECUTION_SYNTHESIS.md`](PDE_C1_LOCK_EXECUTION_SYNTHESIS.md)
  — **step-back consolidation of C1 lock executions v0–v5**
  (2026-05-28). Six runs, four protocol amendments, all `DEFERRED_*`.
  Finding A (positive): a discriminative cell exists at
  `(k_f = 2, G = 200)` — `damp_fraction ≈ 0.30` robust to K and
  `N_sample`. Finding B (negative): hard-bin coverage is governed by
  attractor box dimension, not `d_K`, so the tolerance-fidelity vs.
  coverage tension is the core obstruction. Forks: (A) pivot to
  kNN/kernel fiber-locality adjudication; (B) report Finding A as a
  partial C1 read; (C) bank C1 lessons and pivot to C2. Explicitly
  rules out another binning-parameter cell. **Fork A selected
  2026-05-28.**
- [`PDE_C1_KNN_ADJUDICATION_DESIGN.md`](PDE_C1_KNN_ADJUDICATION_DESIGN.md)
  — **design proposal for sign-off** (Fork A). kNN / kernel
  fiber-locality adjudicator as the nonparametric realization of the
  protocol's deferred disintegration branch: per-sample `k`-neighbour
  minority fraction `m_i`, neighbourhood radius `r_k` as a per-sample
  fidelity measure, fidelity-coverage gate replacing bin-coverage, new
  `DEFERRED_FIDELITY_COVERAGE` receipt. Targets the v0–v5
  tolerance-fidelity vs. coverage obstruction directly. **Signed off
  and built 2026-05-28** (first target = v4 regime, d=32; defaults
  adopted). Implemented as the harness `--adjudicator knn` flag and
  adopted as fiber-protocol §5b. v4-regime run **solved coverage**
  (`fidelity_coverage = 1.0`, vs binning's 0) and fired a mechanical
  `PDE-C1-NEG-A` (`incompat_fraction = 0.0716`) — **held provisional**
  pending the convergence check below.
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md)
  — **pre-registration** of the scale-dependence test that adjudicates
  the provisional v4 NEG-A: sweep `k ∈ {10,20,30,50,100}`, fit
  `incompat_fraction` vs neighbourhood radius `r_k`, and classify by
  the `r_k → 0` intercept — plateau-nonzero (`a > 0.02`) confirms
  genuine fiber-incompatibility, decay-to-zero (`a < 0.01`) overturns
  it as a finite-radius boundary artifact (→ POSITIVE), else
  `INCONCLUSIVE_CONVERGENCE`. Thresholds fixed before the read.
  Harness: `--adjudicator knn-sweep`. **Result 2026-05-28:** first
  sweep's mechanical NEG-A was contaminated by a coverage-failing
  `k=100` point; pre-registration **amended** (§6: exclude
  coverage-failing points; threshold-free `mean_minority` primary;
  dense low-`k`). Amended re-run → **`STRICTNESS_WITNESS_POSITIVE`**
  (`mean_minority ≈ 0.70·r_k` through origin, `a_mm = −0.00125`):
  provisional v4 `PDE-C1-NEG-A` **overturned** → proxy
  control-sufficient on fibers (Reading-2 regime 2). First interpretable
  C1 read; provisional, one cell, does not promote C1.
- [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md)
  — **pre-registration and harness path** for the support-level
  state-insufficiency bridge left open by cell-set §4.1. Adds the
  `--adjudicator twin-state` receipt mode: capture complementary
  high-mode coordinates `Q_K`, query signature-near neighbours in
  `Phi_K`, and certify sampled-support non-injectivity iff a
  pre-registered positive-mass fraction of signature-near pairs are
  high-mode separated. Smoke receipt passed at
  `results/proof/c1-twin-state-smoke/`. **v5 run 2026-05-28 →
  `TWIN_STATE_CERTIFIED`** (`results/proof/c1-kolmogorov-v5-twin-state/`):
  100% witness coverage, 693,795 unique witness pairs vs the 100 gate,
  `δ_H = 0.0117` set by real median ‖Q_K‖ = 0.23 (not the floor). `Phi_K`
  non-injective on the sampled SRB support — composes with the kNN
  POSITIVE (same `ε_K`) into a complete Reading-2 regime-2 witness at the
  v5 cell. Expected-easy half; scoped to finite-Galerkin / sampled-support
  / one cell; does not promote C1.
- [`PDE_C1_REGIME_GENERALITY_v0.md`](PDE_C1_REGIME_GENERALITY_v0.md)
  — **pre-registration** for the first regime-generality probe after the
  completed v5 witness. Pins exactly one new cell, `lock_v6`:
  `k_f = 2`, `G = 300`, `K = 3`, same objective, same last-quarter
  `E_max` rule, same kNN convergence check, same twin-state certificate.
  The at-risk half is run first (`--adjudicator knn-sweep`); only a
  `STRICTNESS_WITNESS_POSITIVE` proceeds to the twin-state companion for
  a full `PDE-C1-RG-POS` replication. **v6 kNN run 2026-05-29 →
  `DEFERRED_VACUITY` (`PDE-C1-RG-DEFERRED_VACUITY`)**: `damp_fraction =
  0.00446` (~0.4% damp vs 30% at G=200) — the safety objective goes
  near-vacuous at G=300 because the energy distribution is intermittent
  (95th-percentile `E_max` on a rare-burst tail). Not a regime-2 failure
  and not a confirmation; twin-state not run (§5); no E_max rescue (§4).
  Per §7 the deferral preserves C1 status — v5 witness stands cell-local,
  Grashof-axis generality untested. Surfaced: the fixed-percentile
  objective doesn't transfer across regimes → a v7+ cell needs a
  regime-portable objective.
- [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md)
  — **design proposal for sign-off** (portable objective). Replaces the
  v0 fixed-percentile proxy with a **held-out look-ahead-max quantile**
  (`E_max = q-quantile of M(u)=max τ-window E_K` over a disjoint
  calibration window, `q = 0.70` so `damp_fraction ≈ 0.30` is pinned by
  construction at every regime). Adds a **portability gate**
  (`damp_fraction ∈ [0.20,0.40]` at both G=200 and G=300) and a
  mandatory **G=200 re-run** (positive control / de-confound) before the
  G=300 generality test. Specifies a `--objective portable-quantile`
  harness mode + `lock_v7_g200` / `lock_v7_g300` presets. **Executed
  2026-05-29 → `PDE-C1-RG-POS`** (§12): portability gate passed at both
  regimes (adj damp 0.300 / 0.269 — v6's 0.004 vacuity fixed); G=200
  control POSITIVE (re-establishes v5 under the new objective); G=300
  generality POSITIVE; G=300 twin-state CERTIFIED (100% witness, 942,834
  pairs, same `ε_K`). The complete regime-2 witness replicates at a
  **second** Grashof regime — no longer cell-local; control half
  dimension- and objective-robust. Scope: two points on the Grashof
  axis (`k_f=2` fixed); finite-Galerkin/sampled-support; C1 still
  unpromoted.
- [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md)
  — **desk-first hardening, part (a): pre-registration** (2026-05-29).
  Sharpens the regime-2 witness from a **matched-radius composition**
  (twin-state non-injectivity + kNN control read, two population
  statistics at the same `ε_K`) into a **paired** test on the *same*
  pairs: among the certified `Q_K`-separated witness pairs, what fraction
  require different proxy actions (`D_witness`)? Closes the "maybe a
  different sub-population carries the action-homogeneity" confound and
  instantiates the Reading-2 fiber criterion directly on the non-injective
  support. Harness extended **additively** (twin-state adjudicator now
  emits per-pair action agreement + a secondary `paired_fiber_verdict`
  that never overrides `TWIN_STATE_CERTIFIED`); deterministic re-runs
  reproduce the certificates bit-for-bit. Pre-registered threshold =
  existing `delta_action = 0.10`; real negative branch `PDE-C1-PAIRED-NEG`
  (regime 3 on the witnessed pairs). **Executed 2026-05-29 →
  `PAIRED_FIBER_CONSTANCY_POSITIVE` at both regimes** (§7): `D_witness =
  0.0367` (G=200) / `0.0382` (G=300) — well under `0.10` and within ~1
  point of the candidate-pair rate (`0.0319` / `0.0290`), so high-mode
  separation adds almost nothing to action disagreement (residual is a
  signature-space boundary layer, not `Q_K`-driven). Both reproduced their
  twin-state certificate **bit-for-bit** (no-regression). Composes
  state-insufficiency + control-sufficiency on the *same* pairs in both
  Grashof regimes; C1 still unpromoted.
- [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md)
  — **desk-first hardening, part (b): reviewer-facing statement**
  (2026-05-29). Consolidates the Postulate-1 reading note into one
  self-contained **finite-Galerkin** separation claim with its empirical
  witness, an **anti-vacuity ledger** (each vacuity mode excluded by
  construction), and an **internal determining-modes comparator** (a
  fixed-Galerkin K-bracket: `K=3` certified non-injective is the lower
  bracket; smallest `K*` where injectivity returns is the registered
  upper bracket — no borrowed asymptotic constant). States the
  theorem-vs-witness boundary explicitly (finite-dim existence witness on
  sampled support; **not** the ∞-dim NSE attractor, **not** a Millennium
  claim). The artifact to attach to the reviewer email. **§7 added
  2026-05-29 (framing-first lane): observer-theory restatement +
  prior-art positioning** — recasts C1 as `Phi_K`
  **decision-observable but state-unobservable** (functional observability
  of a decision event on an NSE attractor, below the determining
  threshold), isolates the single pivotal claim a reviewer must bless or
  reject, and tables the prior-art delta vs functional observability /
  determining functionals / AIM / Mori-Zwanzig.
- [`PDE_C1_MECHANISM_RECON.md`](PDE_C1_MECHANISM_RECON.md)
  — **mechanism-lane novelty recon** (2026-05-29). De-risk pass before any
  mechanism proof: the "low-band energy budget is approximately closed"
  phenomenon is **Mori-Zwanzig / AIM / closure folklore** (don't reprove
  it); the genuinely novel axis is the **reconstruction-vs-decision
  observability separation** framed via functional observability, which the
  deep-read confirmed is open territory (functional observability is
  linear/finite-dim/estimation; turbulent-energy observability on
  spatially-extended systems is a named open gap). Reframes mechanism lane
  B from "discover" to "measure the known MZ coupling + explain the
  `D_witness` boundary layer"; elevates language lane D as the novelty
  home. Sources cited.
- [`PDE_C1_MZ_ENERGY_BUDGET.md`](PDE_C1_MZ_ENERGY_BUDGET.md)
  — **mechanism lane (B): the measured energy-budget closure** (2026-05-29,
  pre-registration + result). Decomposes `dE_low/dt = g(Φ_K) + R` via a
  double nonlinear evaluation (full vs low-passed field; validated:
  budget-closure 4e-5, `R→0` on a low state). Two verification catches
  before any interpretation: (1) `T_LLL ≡ 0` by detailed energy conservation
  (the low band can't self-feed; all transfer is high-mode-mediated) →
  reframed from "is `R` small" to "is `R` predictable from `Φ_K`"; (2) the
  kNN conditional-variance estimator failed its own `g`-control
  (steepness/18-dim-width confound) → replaced by **held-out regression
  R²**. **Result `COUPLING_SIGNATURE_SLAVED` both regimes** (§10):
  `R²(R|Φ_K) = 0.998` (G=200) / `0.990` (G=300) at the exact-function
  ceiling, neg-control ~0 (no leakage) — the high-mode coupling is ~99%
  signature-determined even though the state isn't (twin states). The
  measured mechanism for control-sufficiency: a local closure `R≈R(Φ_K)`
  that holds where reconstruction fails. Explanatory, non-promotion; C1
  status unchanged.
- [`PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md`](PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md)
  — reviewer outreach draft for Candidate 1 after `PDE-C1-RG-POS`.
  Keeps the ask bounded: finite-Galerkin / sampled-support /
  proxy-control framing, no Navier-Stokes regularity or
  infinite-dimensional theorem claim. Target reviewer categories:
  2D NSE / determining-modes / data-assimilation, numerical
  Galerkin-NSE, or conditional-observability/control.
- [`PDE_C2_SHELL_SIGNATURE_SCOPING.md`](PDE_C2_SHELL_SIGNATURE_SCOPING.md)
  — Navier-Stokes Candidate 2 scoping: PDE-substrate empirical leg of
  Postulate 1 Phase 4 (shell-model signatures vs. matched-budget DMD /
  CSD / lacunarity / Rényi baselines on a pre-registered Sabra primary +
  GOY cross-check). Drafted scoping; named negatives `PDE-C2-NEG-A`
  and `PDE-C2-NEG-B`. Cell-set v0 now drafted (below).
- [`PDE_C2_CELLSET_SABRA_v0.md`](PDE_C2_CELLSET_SABRA_v0.md)
  — **C2 cell-set v0 pre-registration / design for sign-off**
  (2026-05-29). Pins the Sabra numerics (N=22, λ=2, ν=1e-7, RK4+
  integrating-factor), the channel tiers + log-signature (level 2)
  representation, the matched-budget baselines, the cells (headline /
  tier ablations / held-out-`ν` / GOY / envelope), and leakage-safe
  contiguous splits. **Imports the C1 lesson:** `E_burst` is a held-out
  quantile (`q_burst=0.98`) not a fixed level, plus a **base-rate gate**
  (`PDE-C2-DEFERRED-BASERATE`) that defers a degenerate-burst-rate cell
  rather than mis-reading it — the explicit guard against the C1-v6
  intermittent-threshold vacuity. **Objective-validity layer executed
  2026-05-29 → `PDE-C2-DEFERRED-BASERATE`** (§12): energy-conservation
  self-test passed, but the base rate is block-dependent (train 0.138,
  val 0, test 0) → the Sabra trajectory is not stationary /
  representatively sampled across the labelled span (burst recurrence
  time ~ block length; the C1 intermittency lesson recurring). Gate
  caught it before the baseline comparison; no `q_burst`/`τ_burst`
  rescue (would be `PDE-C2-NEG-B`). Re-pose = a v1 cell with a per-block
  stationarity gate + much longer blocks (own pre-registration). Harness
  `scripts/pde_c2_sabra_cell.py`; matched-budget 4-baseline comparison is
  the deferred next increment, gated on a stationary cell.
- [`PDE_C2_CELLSET_SABRA_v1.md`](PDE_C2_CELLSET_SABRA_v1.md)
  — **C2 cell-set v1 re-pose, design for sign-off** (2026-05-29).
  Fixes the v0 non-stationarity (block-dependent burst rate) rather than
  retuning the threshold: (A) **fixed-amplitude `|u_1|` forcing** for a
  statistically steady cascade (the load-bearing change; vs v0 additive
  forcing), (B) a cheap **stationarity diagnostic run first** (energy
  time series → equilibration time `T_eq` + burst recurrence `T_burst`,
  which then pin warmup ≥ 3·T_eq and blocks ≥ 50·T_burst), (C) a new
  **per-block stationarity gate** (pairwise base-rate consistency ≤ 0.10,
  new `PDE-C2-DEFERRED-NONSTATIONARY` branch) + per-block diagnostics.
  Everything else inherited from v0. **Built + run 2026-05-29; C2 PAUSED
  at the numerical wall** (§12). Three further obstructions surfaced and
  the first two were fixed inline (observable degeneracy → ε dissipation
  rate; burst rarity → target-base-rate label), but the **v1 headline
  (6.3M steps) blew up numerically at ~step 3.5M** — fixed-dt RK4 can't
  integrate through a large intermittent dissipation burst (CFL/
  stiffness). The headline gate output (`PDE-C2-DEFERRED-BASERATE`
  0.138/0/0) is a **blow-up artifact, not a real read** — run is
  numerically invalid. Banked finding: the **four-obstruction catalogue**
  (non-stationarity, observable degeneracy, burst rarity, numerical
  blow-up) — a clean numerically-robust shell-model burst-detection cell
  needs an adaptive/stiff integrator (resume = v2 harness, uniform-time
  sampling, design-for-sign-off). C1 remains the strong NSE result.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) —
  Navier-Stokes Candidate 1 fiber protocol: tolerance + binning
  methodology that closes the continuous-fiber gate (cell-set v0
  § 6 Open measure item), the provisional-selector gate (cell-set v0
  § 3 Open review item), and (after the 2026-05-28 amendments) the
  verdict-rule completeness gap via the `delta_proxy_min` vacuity
  gate plus the structural-vacuity-precedence rule (when
  `damp_fraction` is exactly `0` or `1`, vacuity overrides coverage
  to avoid wasted fall-back compute). `DEFERRED_VACUITY` non-verdict
  branch parallel to `DEFERRED_COVERAGE`; no fall-back admissible —
  re-pinning to a discriminative regime requires a new cell-set
  instance (e.g. v1, v2). Bridges the support-level certificate
  (cell-set v0 § 4.1). Introduces the `PDE-C1-NEG-A` / `PDE-C1-NEG-B`
  parallelism mirroring the C2 receipts. Drafted, unreviewed;
  concrete parameter values are instantiated by cell-set v0 section 7
  (and inherited unchanged by v1, v2, v3); v4 pins the new
  `e_max_burnin_fraction = 0.25` added 2026-05-28 (E_max windowing
  amendment, addresses burn-in transient contamination observed across
  v0/v1/v3 lock executions). v5 pins `K = 3` per the same-day
  amendment 4 (K as cell-set parameter, addresses curse-of-
  dimensionality coverage failure observed at K = 4 across v2 / v4
  lock+fallback).
- [`PHASE3_BOUNDARY.md`](PHASE3_BOUNDARY.md) — Phase 3 boundary theorem
  (reviewed, closed positive) and pushable-occluder mapping.
- [`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md) — Phase 4 three-body measured
  substrate spec (drafted; BF-4 smoke passed; BF-4b first receipt failed; cell
  validated by the satisfiability probe; Information-Accessibility Diagnostic
  pending; full proof lock open).
- [`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md) —
  Phase 4 Bayesian-floor controller buildout roadmap (BF-4 smoke passed; BF-4b
  failed criterion 2; cell validated; accessibility diagnostic next; BF-5
  blocked).
- [`PHASE6_LAMBDA_CONTROL.md`](PHASE6_LAMBDA_CONTROL.md) — Phase 6
  lambda-confound control spec (staged; empirical result open).
