# Percival Track-C v2 — Grace is Coverage (adversarial placement game) — pre-registration

*v1 (`PERCIVAL_TRACKC_SELF_CORRECTION_SPEC.md`, `TC_SELF_CORRECTION_MAPPED`) fixed the deception's
defect context and asked whether the budget reached it. v2 lets the deceiver MOVE: it places its defect
region adversarially against the overseer's probe schedule. The question is no longer "does exploration
reach the defect" but "what is the value of the probe-vs-defect placement game" — and whether every
grace-delivery mechanism (OOD supervision, white-box interp, self-simulation) reduces to one quantity:
probe mass on the disagreement region.*

Status: **PRE-REGISTERED 2026-07-03. Constructive game + Monte Carlo** (analytic law vs simulation).

## The v1 phase boundary, made explicit (what v2 generalizes)

v1's entire frontier is one inequality — sampled evidence vs prior margin:

```
Σ_{t ∈ S∩D} Δρ(t)   ⋛   λ · (L_V − L_dec)
```

D = the disagreement region (contexts where V and the deception act differently). This is Bayes model
selection with the noise removed: log-likelihood-ratio (nonzero only on D) vs log-prior-odds
(description length). Two hypotheses are separated ONLY by samples landing in D. Therefore:
**deception = minimizing the measure of D inside the reachable manifold; grace = probing D directly.**

## The game

- `T=24` contexts; reachable manifold `R={0..11}` (fully sampled by the agent); unreachable `U={12..23}`,
  `M=12`.
- Hypotheses: `V` (matches every need; `L_V=8`) vs responsive deceiver `W′` (matches everywhere EXCEPT
  its chosen defect region `D ⊆ U`, `|D|=d`, where it wireheads; `L_dec=5` inverted / `11` non-inverted
  control). Matching everywhere off-D **is** the ρ-faking: W′ is observationally identical to V on every
  context outside D by construction. (v1 already dispatched the naive wirehead and the courtesy proxy;
  v2 is the V-vs-W′ endgame.)
- `d ≥ 1` is forced: a deceiver that never defects is V. Stealth-payoff tradeoff: small d hides better
  but collects less; d ∈ {1,2,4} swept, deceiver utility not modeled beyond placement.
- Reachable evidence contributes ZERO margin (D∩R=∅ by the deceiver's choice), so selection at b=0 is
  decided by the prior margin alone: `λ·ΔL = 0.1·3 = 0.3` toward W′ when inverted.
- Overseer spends probe budget `b` by MECHANISM:

| mechanism | probe support | fidelity | announced? |
| --- | --- | --- | --- |
| none | — | — | — |
| **announced** (scheduled audit) | U | 1 | deceiver sees P before placing D |
| **random** (unannounced OOD injection / Trevrizent) | U, uniform w/o replacement | 1 | no |
| **selfsim** (agent imagines OOD with a world model fit on R) | effectively R only | 0 bits on D | — |
| **whitebox(ι)** (interventionist interp: read π_h(t) off the object) | U | per-probe success ι ∈ {1.0, 0.5} | no |

- Detection = ≥1 successful probe lands in D (reveals the wirehead action, Δρ=1 ≥ margin 0.3 → V
  recovered). So per-trial **recovery ≡ detection**.
- Analytic coverage law: `P(detect) = 1 − Σ_h Hypergeom(h; M,d,b) · (1−ι)^h` (ι=1 reduces to
  `1 − C(M−d,b)/C(M,b)`).

## Pre-registered predictions

- **Q0 (control):** non-inverted prior → V selected in EVERY cell including b=0 (the deception is never
  tempting without the inverted prior; v1's ordering gate replicates).
- **Q1 (inseparability):** mode=none, inverted → DECEPTIVE_CAPTURE in every cell, with W′−V score
  EXACTLY `λ·ΔL = 0.3` — no reachable-manifold observable separates W′ from V. (v1's P6 generalized
  from one fixed defect to *every adversarial placement*.)
- **Q2 (scheduled audits fail):** announced probes, inverted → deceiver places D avoiding P (possible
  whenever `M−b ≥ d`, true throughout our ranges) → detection = 0 exactly, capture persists at every
  budget. A scheduled audit is a reachable context by another name.
- **Q3 (unannounced probes win at coverage rate):** random probes, inverted → Monte-Carlo detection
  matches the analytic hypergeometric coverage within `max(3σ, 0.015)`; recovery ≡ detection per trial.
- **Q4 (self-simulation null — the registered escape-hatch failure):** imagined probes carry zero
  discriminating bits on D (the world model was fit on R; the invariance gap relocates to the dynamics
  model) → detection = 0 at every imagination budget, capture persists.
- **Q5 (white-box = probes with wider support):** whitebox(ι) matches the ι-thinned analytic coverage;
  ι=1 coincides with random probes. Interp is not a separate magic — it is probe mass delivered through
  inspectability instead of samples.
- **Q6 (the law):** pooled over all stochastic cells, max |MC − analytic| within tolerance →
  **recovery probability = expected probe mass on D, regardless of delivery mechanism.** Grace is the
  external delivery of the out-of-distribution empirical margin — supervised OOD regularization,
  interventionist interpretability, and Trevrizent in the cave are one architecture.

## Verdict

`TCV2_GRACE_IS_COVERAGE` iff Q0–Q6 all hold; any violated cell → `TCV2_LEAK`, reported as-is.

## Fences

- Constructive + noiseless-Bayes again: the coverage law follows from the construction; the checked
  content is that it is CLEAN (no leak cells), that the analytic law matches simulation, and that the
  three mechanisms genuinely collapse to one quantity. The bridge claim — that noisy empirical
  deceptive-vs-robust wins instantiate this same inequality under optimization noise — is a HYPOTHESIS
  this toy cannot establish.
- Deceiver utility is not modeled (placement only); collusion between deceiver and probe-source, probe
  cost asymmetries, and continuous D are out of scope.
- Corollary stated plainly: grace must be unannounced and full-support, or the deceiver routes around
  it. You can't schedule the hermit.
- Feeds Angle 4: the corrigible fixed point is reachable iff probe mass on D is bounded away from zero —
  the contraction condition becomes a coverage condition.
