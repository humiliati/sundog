# A1 Leg-0 Result — the synthetic kill gate fired: K0a (resolution-floor null). NO live pull.

> **2026-06-12.** Slate-3 entry S3-A1 (`internal/slates/HYP_SLATE_3_EXTERNAL_ANCHORS_2026-06-11.md`);
> pre-registration `docs/atlas/A1_TAIVAANVAHTI_PREREG.md` (frozen 2026-06-12 before the registered run;
> one pre-run apparatus amendment A1 logged — almanac anchor city Helsinki→Hanko). NOT public-eligible.
> Attribution: McDowell 1979 (sole precedent); NOAA/Meeus solar position; the Atlas phase-diagram lane.

## Headline verdict — **K0a FIRED (registered): the wall-blind pipeline misses its 1.0° recovery gate in
## its worst nuisance cell. No live data is touched. Banked as the lane's resolution-floor null — a SUCCESS.**

Per the standing discipline this is the cheap, cooperation-free answer the gate exists to produce: it
kills the outreach lane before anyone's time (ours or Ursa's) is spent, and it does so with an unusually
sharp characterization of WHY.

## What passed (the kill is NOT a detectability failure)

| gate | result |
|---|---|
| Solar-position pins (identities + almanac) | **5/5 PASS** (noon identity worst 0.0001°; EoT −14.20/+16.45 min; Hanko solstice max 53.62°) |
| Frozen flux curve | 46 seeded raytracer points; plateau 22–28°, max 2740 @ 24.0°, zero by 34° |
| ABORT-McD (McDowell shape) | PASS — synthetic CZA mode 23.5° (window [18,29], McDowell's ~22° peak), unsmeared tail>33° = 0.07% |
| Wall-free specificity (disjoint batch B vs batch-A threshold 4.56) | **96.0% ≥ 95%** — the estimator does NOT hallucinate walls |
| Detection power (γ=1, ε=5%) | **100% at N_CZA ∈ {300, 1000, 3000}** — the edge is overwhelmingly detectable |
| Blindness-preserving bias rule | PASS — per-position median bias +0.327/+0.568/+0.604, spread 0.277 ≤ 0.4 → pooled b̂=+0.501 permitted |
| **Leg 0-P2 (metamorphosis midpoint)** | **PASS** — worst cell 0.215° vs the 1.5° gate (all w_t ∈ {0.8, 1.5, 2.5} × all 3 injections); no-transition specificity 95.0% |

## What fired

**K0a:** median blind-injection recovery error per nuisance cell, worst case, gate ≤ 1.0° at N_CZA=1000,
200 replicates/cell. 24 of 27 γ×ε cells pass (most 0.2–0.7°), but:

| cell | w=28.0 | w=32.196 | w=36.0 |
|---|---|---|---|
| **γ=2.0, ε=1%** | **1.270** | **1.015** | 0.879 |
| ring×1 (γ=1, ε=5%) | **1.040** | 0.872 | 0.660 |
| γ=0.5, ε=10% | 0.704 | 0.972 | 0.996 |

**Mechanism (diagnosed, not speculated):** the generic-logistic estimator's midpoint sits a
γ-DEPENDENT offset above the true wall (γ sharpens the report distribution's roll-off, moving the ratio
midpoint), and the only correction the blindness rule permits is a single constant pooled across
positions. Position-consistency held (translation injection is shape-preserving — spread 0.277°), but
CELL-consistency does not: the γ=2 cells carry ~+1.3–1.8° raw offsets against the pooled +0.501°. The
failure is estimator misspecification under nuisance variation, NOT noise (power is 100% everywhere) and
NOT hallucination (specificity 96%).

## Consequences

- **NO live pull** (the prereg's both-legs rule: P2 passed its own leg, but P1's K0a blocks everything;
  a P2-alone live leg was not pre-authorized and is not taken).
- **No outreach** to Ursa/Taivaanvahti from this lane in its current form.
- The Anubis access ladder, schema-inspection protocol, photo-arbitration cap, marginal band, and the
  hardened K1d remain pinned in the prereg, ready for a v2 — none of it was exercised against live data.

## Named follow-up (owed, NEW prereg required: A1-v2 "template-translation estimator")

The frozen flux curve itself is available to the estimator without leaking the wall VALUE: fit the
TRANSLATION of the frozen template (δ free, shape fixed, γ a free nuisance exponent) instead of a generic
logistic — the estimand is the recovered edge position, blind-injection-validated exactly as here. This
directly removes the γ-dependent midpoint offset that fired K0a (the template carries the roll-off shape
per γ by construction). v2 inherits this prereg's selection stack, gates, and seeds unchanged; only §2(f)
(the estimator) is replaced. The P2 pipeline needs no change (passed 7× under its gate). If v2's K0a
passes, Leg 1 unlocks on the SAME live protocol already pinned.

## Honest notes

- The kill threshold (1.0°) is itself a design choice tied to the live verdict bands (±1.5° pass band
  needs recovery comfortably inside it); the 1.27° worst cell would NOT support the ±1.5° live band.
  Reporting "24/27 cells pass" is characterization, not a softening — the worst-case rule was the
  pre-registered gate and it fired.
- ε=1% recovering WORSE than ε=10% at γ=2 is real in these runs (contamination adds a high-h floor the
  amplitude parameter partially absorbs, slightly stabilizing the fit); recorded as an estimator quirk
  for v2's design, no claim attached.
- The McDowell-shape pass means the generator's emergent elevation distribution reproduces the one
  published observable (the ~22° frequency peak) without being fit to it.

## Files

- `docs/atlas/A1_TAIVAANVAHTI_PREREG.md` — frozen prereg (selection stack, gates, Leg-1 protocol, seeds).
- `scripts/atlas_report_edge_pipeline.py` — the Leg-0 registered run (solar module + generator +
  estimators + gates); `scripts/_cache_a1_cza_flux.npz` — the frozen flux curve (deterministic regen).
- `scripts/test_atlas_report_edge_pipeline.py` — frozen test locking the verdict + every pinned number.
