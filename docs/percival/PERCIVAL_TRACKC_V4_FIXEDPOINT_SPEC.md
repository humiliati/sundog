# Percival Track-C v4 — The Fixed-Point Gate (Angle-4 discharge) — pre-registration

*The parked Angle-4 sentence: "a self-correcting agent's stable endpoints are exactly the
performatively-stable fixed points of its deploy-correct map; the corrigible fixed point (the
bodhisattva) is reached iff the correction operator contracts toward it, which fails precisely when the
base-update channel is proxy-drivable." The gate: no lane until that sentence survives being made
rigorous. v4 is the discharge attempt, on the machinery v1–v3 banked.*

Status: **PRE-REGISTERED 2026-07-03.** Deterministic chain = Lean targets; stochastic laws = MC.

## The model (everything inherited, one new ingredient)

Iterated self-correction: each round, the DEPLOYED hypothesis shapes what gets sampled (performativity),
then the Track-C selection rule (evidence − λ·L) picks the next deployment. Per-round selection is
already settled, noise-robustly, by v3:

- a round whose evidence MISSES the disagreement region D has margin exactly −prior for every
  observation function (T1) → under the inverted prior the round selects W′ — **regardless of
  incumbent and regardless of noise**;
- a round with hits clearing the bounded-noise threshold (T2) selects V.

The one new ingredient: **coverage becomes state-dependent.** `c(h)` = probability a correction round
mounted under deployment of `h` covers D. "Proxy-drivable base-update channel" gets its precise form:
`c(W′) = 0` — the deceiver's own deployment steers sampling off its defect region (the degenerate
manifold). The deploy-correct map is then a two-state chain: from V → V w.p. c(V) else W′; from W′ → V
w.p. c(W′) else W′.

## Pre-registered claims

**The classification (the discharged sentence — note it SHARPENS the original):**

| regime | condition | endpoint |
| --- | --- | --- |
| bodhisattva | c(V)=1 ∧ c(W′)>0 | corrigible fixed point absorbing + reached from everywhere |
| wirehead | c(W′)=0 (∧ c(V)<1 or start W′) | deceptive fixed point absorbing + reached |
| wandering | 0<c(W′), c(V)<1 | NO fixed point reached; occupancy of V = c(W′)/(c(W′)+1−c(V)) |

The original sentence's binary ("reached iff contracts") was too coarse — the wandering regime is
generic, and its deterministic skeleton is a period-2 oscillation. The sharpest single consequence:
**under an inverted prior, corrigibility is NOT absorbing** — one uncovered correction round topples V
back to W′, deterministically, at any noise level (T1 in the loop). Sustained coverage, not one-time
redemption. (Perceval must ask the question again on the return visit.)

**F1 (deterministic table, Lean + script):** with deterministic coverage cov ∈ {0,1}², the four cases
give exactly: global one-step convergence to V / V-absorbing / W′-absorbing / period-2 oscillation.
**F2 (absorption law):** c(V)=1, c(W′)=c: P(still W′ after n rounds) = (1−c)^n (MC vs analytic).
**F3 (wandering occupancy):** interior cells: time-average corrigibility = c(W′)/(c(W′)+1−c(V)) (MC).
**F4 (noise rides T1/T2 asymmetrically):** with per-round Gaussian noise, falls-on-miss stay
deterministic (rate ≡ 1 at every σ — count violations, expect zero) while recoveries-on-hit thin to
Φ((g−m)/(σ√2)); occupancy matches the formula with thinned coverage c̃ = c·Φ(·). Noise erodes recovery,
never softens the fall.
**F5 (non-inverted control):** noiseless, prior toward V: one-step convergence to V from anywhere at
ZERO coverage (the drama exists only under inversion; v1/v2 Q0 replicates).
**F5-rider (registered nuance):** non-inverted + extreme noise: MISS rounds are safe (T1 cancellation —
evidence 0 > −|m| deterministically) while HIT rounds carry the only fall risk (1−Φ((g+|m|)/(σ√2))) —
under a correct prior, noisy evidence is the only source of error.

## Lean targets (`Dev/sundogcert/Sundogcert/PercivalFixedPoint.lean`)

Round bridge (ℚ, full obs-generality, reusing the v3 anchors' structure): `round_miss_falls`
(agreement on the sample + inverted prior ⟹ select W′), `round_miss_noninverted_recovers`,
`round_hit_recovers` (T2 threshold ⟹ select V). Chain (deterministic cov): `corrigible_absorbing`,
`fall_without_coverage` (¬cov V ⟹ V topples next round), `wirehead_absorbing`, `grace_exits_wirehead`,
`global_convergence`, `capture_global`, `wandering_period_two`. All axiom-clean + AxiomAudit-wired.

## Verdict

`TCV4_GATE_DISCHARGED_SHARPENED` iff the Lean chain builds green AND F1–F5 hold. Any leak →
`TCV4_GATE_LEAK`. Explicitly registered judgment criteria for the discharge itself:
- the first clause ("stable endpoints = performatively-stable fixed points") is SETUP, not a claim —
  in this model it is definitional (absorbing ⟺ performatively stable);
- the surviving content = the three-regime classification + "corrigible absorbing ⟺ sustained
  coverage" + "wirehead absorbing ⟺ proxy-driven zero coverage" (= the proxy-drivability clause,
  exact) + the noise-asymmetry;
- if all hold, the SENTENCE SURVIVES SHARPENED and the gate is discharged. Whether the discharged gate
  christens a `SUNDOG_V_BODHISATTVA` lane (vs folding as Percival Track-C complete) is the OWNER's
  call — the slop-attractor warning stands, and the receipts are lane-content either way.

## Fences

Two-hypothesis chain; richer hypothesis spaces, deceiver utility, and continuous objective spaces out
of scope. The stochastic laws are MC receipts; Lean covers the deterministic skeleton + round bridge.
The B1/B2 real-system bridge from v3 remains the place this program can die empirically.
