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
  adopted as fiber-protocol §5b. First run launched at the v4 regime.
- [`PDE_C2_SHELL_SIGNATURE_SCOPING.md`](PDE_C2_SHELL_SIGNATURE_SCOPING.md)
  — Navier-Stokes Candidate 2 scoping: PDE-substrate empirical leg of
  Postulate 1 Phase 4 (shell-model signatures vs. matched-budget DMD /
  CSD / lacunarity / Rényi baselines on a pre-registered Sabra primary +
  GOY cross-check). Drafted scoping; cell-set v0 deferred to a
  follow-up artifact; named negatives `PDE-C2-NEG-A` and `PDE-C2-NEG-B`.
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
