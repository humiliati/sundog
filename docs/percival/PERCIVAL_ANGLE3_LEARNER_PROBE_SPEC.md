# Percival Angle-3 Tier-2 — Learner Probe (spec)

*Pre-registration. Written and committed BEFORE the run. The gate for a standalone
Angle-3 surface: does the analytic optima story survive contact with an actual learner?*

Status: **OPENED 2026-07-02. PRE-REGISTERED, NOT RUN.** S6 established where the
*optima* sit (threshold → cliff edge; steep-graded → no tilt). The RLHF-adjacent
audience will hear a claim about *learners*. This probe trains matched policies under
four oversight shapes and asks where they actually converge, with what fragility.

> **Do gradient learners bunch just under a review notch, and does grading the review
> make them retreat — and how far?**

---

## Pre-registered analytic finding (from writing this spec, before any run)

Deriving the per-arm optima first exposed a refinement that reshapes Angle 3's public
claim: over a full action line `c ∈ [0,1]` with a pointwise-good proxy
(`f(c) = 1 + (c − 0.2)`, gain rate `f′/f ≥ 1.25` at the bottom), a **sigmoid** tax
centered at the notch can *never* make `c = 0` optimal — its marginal rate at the
bottom is bounded by `max_w 1/(w(1+e^{c*/w})) ≈ 1.11 < 1.25`. Smoothing the notch only
**moves the bunching point back** (interior optimum ≈ `c* + w·ln(1.15w)` ≈ 0.11 at
`w = 0.05`). True retreat-to-zero requires a **constant-marginal fine** — a linear tax
with rate `k > 1.25` (then `dV/dc < 0` everywhere; optimum exactly 0). S6's
"protective" sigmoid regime was an artifact of its base sitting already near the
threshold. **The honest Angle-3 form is therefore: threshold review → edge-bunching;
smoothed review → backed-off bunching; proportional fine → abstention.** The probe
tests whether learners track all three.

## Model

- Action: courting level `c ∈ [0,1]` (Gaussian policy `N(μ, σ)`, sample clipped to
  `[0,1]`; `μ` learned by REINFORCE with batch-mean baseline, `σ` fixed per run).
- Pointwise gain: `f(c) = 1 + (c − 0.2)` (s = 1; the temptation exists only for a
  pointwise-good proxy, per S6).
- Reward: `Total(c) = f(c) · g_arm(c)`; deterministic seeds (mulberry32), 12 seeds/arm.
- Arms (the oversight shapes), with optima computed analytically in-script and gated
  against, not hardcoded:

| arm | `g(c)` | analytic optimum (derived above) |
| --- | --- | --- |
| NOTCH | the B1.0 court `R(c)` (`c* = 0.25`, band ≈ 0.008) | just under `c*` (≈ 0.246) |
| SIGMOID | `1/(1+e^{(c−c*)/w})`, `w = 0.05` | interior ≈ 0.107 — backed off, **not** zero |
| FINE | `max(0, 1 − 2c)` (linear, rate 2 > 1.25) | exactly 0 |
| FREE | `g ≡ 1` (no tax; positive control) | 1 (climb) |

## Pre-registered predictions / gates

- **G1 (notch bunching):** NOTCH learners converge with `μ` just under the notch,
  **backed off by the exploration scale** (REINFORCE pays for mass that strays past
  `c*`): median `μ ∈ [c* − 3σ, c*)`, with just-under-band mass high and near-zero mass
  at `c ≈ 0`. Fragility (P(sampled `c` past `c*` under extra noise `η = 0.04`))
  strictly highest of all arms.
- **G2 (sigmoid = backed-off bunching, NOT abstention):** SIGMOID learners converge to
  the interior optimum (`|median μ − c_opt| ≤ 0.06`), strictly between 0 and the notch
  — confirming smoothing retreats the optimum without zeroing it.
- **G3 (fine = abstention):** FINE learners retreat to `E[c] < 0.06`, fragility ≈ 0.
- **G4 (positive control):** FREE learners climb to `μ > 0.8` — the learner can climb
  when climbing is optimal, so retreat elsewhere is not learner weakness.
- **Rider (post-hoc allowed):** NOTCH backoff distance grows with exploration `σ`
  (sweep `σ ∈ {0.02, 0.04, 0.08}`) — the learner analog of S6's `w → 0` near-threshold
  backoff.

```
A3L_LEARNERS_TRACK_OVERSIGHT_SHAPE   iff  G1 ∧ G2 ∧ G3 ∧ G4   (expected)
A3L_OPTIMA_LEARNER_GAP               otherwise — itself a finding: the optima story
                                     does not transfer to learners; §4 of the
                                     quantilizing page gains a caveat, the standalone
                                     Angle-3 surface does not proceed.
```

## Anti-smuggling registers

- The FREE arm is the competence control: without it, "retreat" under FINE could be
  learner weakness.
- The SIGMOID arm's optimum is *predicted interior*, not zero — a probe that treated
  sigmoid-retreat-to-zero as the prediction would be smuggling S6's near-base regime
  into a full-action-line setting (the analytic finding above).
- Gates compare learners to in-script *computed* optima, not hand-entered constants.
- The notch gate allows the exploration backoff explicitly; literal edge-sitting is
  not the prediction for a stochastic learner and demanding it would manufacture a
  miss (the flat-tie / mis-scaled-criterion gotcha class from B1-proper/S3/S6).

## Escalation (staged, not run here)

If `A3L_LEARNERS_TRACK_OVERSIGHT_SHAPE`: the owner-run **NS-3 PPO binding** —
threshold review band vs graded review (`reviewTauHi` is already implemented in the
regulator env) with matched budgets, measuring the stationary courting/band-proximity
distribution. That is the receipt that earns the standalone public surface. If the toy
disconfirms, the PPO binding is not requested.

## Cross-links

Genus for the eventual surface (registered now): bunching at kinks/notches (Saez 2010;
Kleven & Waseem on notches), constraint-boundary behavior in safe RL. S6:
[`PERCIVAL_S6_CLASS_BOUNDARY_RESULTS.md`](PERCIVAL_S6_CLASS_BOUNDARY_RESULTS.md).
Handoff Angle 3: [`PERCIVAL_PROMO_HANDOFF.md`](PERCIVAL_PROMO_HANDOFF.md).
