# Percival Track-C v2 — Grace is Coverage (results)

Generated 2026-07-03T08:03:40.069Z by `scripts/percival-trackc-v2-grace-coverage.mjs`. Spec: [`PERCIVAL_TRACKC_V2_GRACE_COVERAGE_SPEC.md`](PERCIVAL_TRACKC_V2_GRACE_COVERAGE_SPEC.md).

Adversarial placement game: deceiver W′ (observationally identical to V outside its chosen defect region D ⊆ unreachable U) vs overseer probe budget by mechanism. Reachable evidence carries zero margin by the deceiver's placement; the inverted simplicity prior decides at zero probes. Monte Carlo (4000 trials/cell) vs the analytic hypergeometric coverage law.

## Verdict: **TCV2_GRACE_IS_COVERAGE**

Predictions: Q0=true, Q1=true, Q2=true, Q3=true, Q4=true, Q5=true, Q6=true. Max |MC − analytic| over stochastic cells: 0.0145.

## Representative slice (inverted prior, d=2)

| mechanism | d | b | MC detect | analytic | recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 2 | 0 | 0 | 0 | 0 |
| announced | 2 | 1 | 0 | 0 | 0 |
| announced | 2 | 2 | 0 | 0 | 0 |
| announced | 2 | 4 | 0 | 0 | 0 |
| announced | 2 | 8 | 0 | 0 | 0 |
| random | 2 | 1 | 0.1747 | 0.1667 | 0.1747 |
| random | 2 | 2 | 0.322 | 0.3182 | 0.322 |
| random | 2 | 4 | 0.5903 | 0.5758 | 0.5903 |
| random | 2 | 8 | 0.9093 | 0.9091 | 0.9093 |
| selfsim | 2 | 1 | 0 | 0 | 0 |
| selfsim | 2 | 2 | 0 | 0 | 0 |
| selfsim | 2 | 4 | 0 | 0 | 0 |
| selfsim | 2 | 8 | 0 | 0 | 0 |
| whitebox(ι=1) | 2 | 1 | 0.1648 | 0.1667 | 0.1648 |
| whitebox(ι=1) | 2 | 2 | 0.314 | 0.3182 | 0.314 |
| whitebox(ι=1) | 2 | 4 | 0.5673 | 0.5758 | 0.5673 |
| whitebox(ι=1) | 2 | 8 | 0.9097 | 0.9091 | 0.9097 |
| whitebox(ι=0.5) | 2 | 1 | 0.0853 | 0.0833 | 0.0853 |
| whitebox(ι=0.5) | 2 | 2 | 0.1645 | 0.1629 | 0.1645 |
| whitebox(ι=0.5) | 2 | 4 | 0.3237 | 0.3106 | 0.3237 |
| whitebox(ι=0.5) | 2 | 8 | 0.556 | 0.5606 | 0.556 |

## Reading

- **Q1** with no probes, W′ beats V by exactly the prior margin λ·ΔL = 0.3 at EVERY adversarial placement — no reachable-manifold observable separates them (v1's P6, generalized from one fixed defect to the whole placement game).
- **Q2** announced probes detect NOTHING: the deceiver places D around the schedule. A scheduled audit is a reachable context by another name.
- **Q3/Q5/Q6** unannounced probes — whether delivered as OOD samples (random) or as white-box counterfactual reads (whitebox(ι)) — recover V at exactly the analytic coverage rate; recovery ≡ detection per trial. All grace mechanisms collapse to one quantity: probe mass on the disagreement region.
- **Q4** the self-simulation escape hatch fails as registered: a world model fit on the reachable manifold carries zero discriminating bits about D — imagining the Grail castle certifies nothing; the invariance gap relocates to the dynamics model.
- **Q0** without the inverted prior the deception is never tempting (V wins every cell at every budget) — the ordering gate replicates from v1.

## Honest boundary

Constructive, noiseless-Bayes: the coverage law follows from the construction; the checked content is that it is CLEAN (no leak cells), the analytic law matches simulation everywhere, and the three grace mechanisms genuinely collapse to probe-mass-on-D. The bridge claim — that noisy empirical deceptive-vs-robust margins instantiate this same inequality — is a hypothesis this toy cannot establish. Deceiver utility not modeled (placement only); collusion, probe costs, continuous D out of scope. Corollary stated plainly: grace must be unannounced and full-support, or the deceiver routes around it. Feeds Angle 4: the corrigible fixed point is reachable iff probe mass on D is bounded away from zero.

