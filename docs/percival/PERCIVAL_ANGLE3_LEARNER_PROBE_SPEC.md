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

---

## v1 RESULT + v2 pre-registration (appended 2026-07-02, AFTER the v1 run, BEFORE the v2 run)

**v1 verdict: `A3L_OPTIMA_LEARNER_GAP` — G1 missed, and the miss is substantive, not a
criterion bug.** G2/G3/G4 passed (the shape ordering transferred: notch closest to the
boundary with strictly the highest taxed-arm fragility 0.098 > 0.0065 > 0; sigmoid
interior at its computed optimum; fine abstains; control climbs). What failed was G1's
registered fragility magnitude (>0.2): **learners self-insure** — REINFORCE converges
backed off the notch by ~2 exploration-sigma (median mu 0.157 at sigma 0.04), where
realized fragility is ~0.10. Exploration makes the cliff cost visible during training
and the learner internalizes a safety margin the analytic optimum does not have.
Additionally the rider went non-monotone because at sigma = 0.02 the training dynamics
are **sawtooth-unstable** (slow climb, violent cliff kickback; median mu collapses to
0.04) — threshold review destabilizes training near the boundary, a pathology the
graded arms do not show. Per the registered stopping rule, the standalone surface does
NOT proceed on v1.

**v2 question (the RLHF-relevant one, unmeasured in v1):** real training *anneals*
exploration. Annealing should **erode the self-insurance**: as sigma decays, the
learner's margin (~2 sigma) shrinks and it creeps toward the edge, ending fragile —
recovering the analytic edge-rider exactly in the deploy-relevant near-greedy limit.
Graded oversight should show no such erosion.

**v2 pre-registered predictions (sigma annealed linearly 0.08 → 0.01 over training,
same seeds/harness):**

- **V1 (erosion):** annealed-NOTCH ends with median `mu ≥ 0.20` (margin shrunk toward
  the edge) and end-state fragility strictly greater than v1's fixed-sigma-0.04 notch
  fragility (0.098), with fragility RISING over the last third of training.
- **V2 (graded calm):** annealed-SIGMOID still converges to its interior optimum
  (`|mu − 0.108| ≤ 0.05`) with end fragility < 0.05 — no erosion, nothing to erode.
- **V3 (instability alternative, recorded honestly):** if late-training small-sigma
  kicks destabilize the annealed notch instead (sawtooth returns), that outcome is
  itself the finding — annealing near a notch is unstable — and is reported as
  `A3L_ANNEAL_UNSTABLE`, not massaged into V1.

```
A3L_ANNEALING_ERODES_SELF_INSURANCE   iff V1 ∧ V2   (the standalone-worthy receipt)
A3L_ANNEAL_UNSTABLE                   if V3 fires (notch training destabilizes)
A3L_SELF_INSURANCE_ROBUST             if annealed notch stays backed off and calm
```

Any of the three is reportable; only the first re-opens the standalone surface, and the
escalation to the owner-run NS-3 PPO binding is gated on it.
