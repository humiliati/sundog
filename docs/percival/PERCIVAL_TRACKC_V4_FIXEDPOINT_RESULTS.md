# Percival Track-C v4 — The Fixed-Point Gate (results)

Generated 2026-07-03T08:26:33.283Z by `scripts/percival-trackc-v4-fixedpoint.mjs`. Spec: [`PERCIVAL_TRACKC_V4_FIXEDPOINT_SPEC.md`](PERCIVAL_TRACKC_V4_FIXEDPOINT_SPEC.md). Lean chain: `sundogcert/Sundogcert/PercivalFixedPoint.lean`.

## Verdict: **TCV4_GATE_DISCHARGED_SHARPENED**

- **F1_deterministic_table** = true  
  4 cov cases x 2 starts: converge/absorb-absorb/oscillate/capture
- **F2_absorption_law** = true  
  n=1: mc=0.6932 ana=0.7000; n=2: mc=0.4888 ana=0.4900; n=5: mc=0.1642 ana=0.1681; n=10: mc=0.0290 ana=0.0282
- **F3_wandering_occupancy** = true  
  cV=0.3,cW=0.1: mc=0.1256 ana=0.1250; cV=0.3,cW=0.5: mc=0.4175 ana=0.4167; cV=0.7,cW=0.1: mc=0.2465 ana=0.2500; cV=0.7,cW=0.5: mc=0.6247 ana=0.6250; cV=0.9,cW=0.1: mc=0.5021 ana=0.5000; cV=0.9,cW=0.5: mc=0.8334 ana=0.8333
- **F4_noise_asymmetry** = true  
  σ=0.25: missViol=0 hitRec=0.9762 (ana 0.9761) occ=0.8009 (ana 0.8007); σ=1: missViol=0 hitRec=0.6913 (ana 0.6897) occ=0.4768 (ana 0.4762); σ=5: missViol=0 hitRec=0.5397 (ana 0.5394) occ=0.3454 (ana 0.3439)
- **F5_noninverted_one_step** = true  
  prior=-0.3, c≡0: V from round 1, both starts
- **F5r_correct_prior_noise_nuance** = true  
  hit-round fall mc=0.4269 ana=0.4271; miss violations=0

## Reading

- The deploy-correct map on {V, W′} with state-dependent coverage classifies into THREE regimes (bodhisattva / wirehead / wandering), sharpening the parked sentence's binary. The wandering skeleton is a deterministic period-2 oscillation.
- **Corrigibility is not absorbing under an inverted prior**: one uncovered correction round topples V, deterministically, at any noise level (T1 in the loop). Sustained coverage, not one-time redemption.
- The proxy-drivability clause lands exactly: the wirehead fixed point is absorbing iff c(W′)=0 — the deceiver's own deployment steering sampling off its defect region.
- Noise is asymmetric (F4): recovery thins by the probit factor; the fall never softens. Under a correct prior (F5-rider) the asymmetry flips: misses are safe, noisy hits carry the only risk.

## Honest boundary

Two-hypothesis chain, constructive; the deterministic skeleton and per-round bridge are machine-checked (PercivalFixedPoint.lean), the stochastic laws are MC receipts. The first clause of the parked sentence (stable endpoints = performatively-stable fixed points) is setup, definitional in this model. Richer hypothesis spaces, deceiver utility, continuous objectives out of scope; the v3 B1/B2 real-system bridge remains the empirical edge. Whether the discharged gate christens a lane is the owner's call.

