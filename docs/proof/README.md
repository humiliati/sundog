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
  — Navier-Stokes Candidate 1 cell-set v2: triples Grashof to
  `G = 300` while keeping `k_f = 2`, intended to break the
  energy-conservation symmetry that made v1 vacuous. All other v0/v1
  §7 values inherited. Drafted, desk-auditable, unreviewed, unrun.
  Harness exposes it as `--preset lock_v2`.
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
  (and inherited unchanged by v1 and v2).
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
