# Chat-v2 Phase 7c — Coupled Closure Boundary Receipt

> 2026-06-04. Run + adjudication of the frozen `PHASE7C_COUPLED_LATENT_BOUNDARY_SPEC.md`
> (`p_noise = 0.20`, the bisection cell). **Status: internal receipt — aggregate
> `boundary_transition_zone`.** The noise-survival boundary is **located at p ≈ 0.20**,
> and it is a *zone*, not a knife-edge. Toy; **no real-LLM / NSE / scaling claim;
> R1 not met; unpromoted.**

## Provenance

- Runner: `chatv2_phase0_bodyresist.py --latent coupled --p-noise 0.20`;
  probe `chatv2_phase7_probe.py --mode coupled --dir phase7c-coupled-p20`.
- git `c7f43958`. Training ~1.4 h (3 seeds, bg, GTX-1080, 08:51→10:15);
  probe ~16–22 min (bg). De-confound PASS `0.503`.

## Training (all 3 learned; readout quality is itself split)

| seed | gen `eval_loss` | learned | **z1 (readout)** | body_carry gen/twin |
| --- | --- | --- | --- | --- |
| 0 | 0.4855 | ✓ | **0.88** | 0.91 / 0.59 |
| 1 | 0.4761 | ✓ | 0.75 | 0.83 / 0.73 |
| 2 | 0.4832 | ✓ | **0.66** | 0.79 / 0.59 |

At `p=0.20` the trained readout quality straddles the survival line (0.66–0.88) — this
is what splits the closure outcome.

## Per-seed verdict + aggregate

| seed | k=3 func(u) | **k_func** | k_state | nf_func | branch |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.760 | **3** | None | 0.596 | **`closure_confirmed`** |
| 1 | 0.651 | None (caps 0.657) | 7 | 0.565 | `closure_washed_by_readout` |
| 2 | 0.642 | None (caps 0.642) | None | 0.532 | `closure_washed_by_readout` |

**Aggregate: `boundary_transition_zone`** (1 confirm / 2 wash among 3 learned seeds;
`u_null` control standing clean).

## Reading — the boundary is a zone, and it's readout-quality-gated

The split is **mechanistic, not noise**: at p=0.20 the closure sits right at the bar,
so survival is decided by the readout quality the training happened to reach.
- **seed 0** (z1=0.88, clean) → `func(u)=0.76` clears both the 0.70 bar and the null →
  closure survives.
- **seeds 1/2** (z1=0.66–0.75, noisy) → `func(u)` caps 0.64–0.66 — **above the
  selection-corrected null (0.53–0.57) but below the 0.70 bar.** The closure signal is
  *present* (above null) but *washed* (below threshold), the same partial-wash shape as
  Phase 7.

So p≈0.20 is not a knife-edge `p*`; it is the noise level at which the closure becomes
**recoverable-iff-the-readout-is-clean-enough**, and the trained-readout variance
(z1 0.66→0.88) is exactly what tips individual seeds either way.

## Noise-survival boundary — LOCATED (all three points)

| p_noise | cell | per-seed | aggregate |
| --- | --- | --- | --- |
| **0.10** | 7b | 3/3 confirm | `boundary_high` (survives) |
| **0.20** | 7c | 1 confirm / 2 wash | **`boundary_transition_zone`** (≈ the edge) |
| **0.25** | 7 | 3/3 wash | `boundary_low` (washes) |

The closure survives a trained-body readout for `p ≲ 0.10`, washes for `p ≳ 0.25`, and
**transitions through a readout-quality-gated zone centered at p ≈ 0.20.** Boundary
located. A finer sweep (`p ∈ {0.15, 0.18, 0.22}`) would only sharpen the zone width — a
separate optional cell, not required to call the boundary.

## Tier

Pure boundary-location on the designed coupled toy. The full curve (confirmed →
transition → washed) is the deliverable. Still `H=8`, `d_dec≈7.6`, designed coupling,
deliberately swept noise — **no real-LLM, NSE, or scaling claim; R1 not met.** The
allelopathy lane now holds a fully calibrated instrument: a real negative (Phase 2), a
partial (Phase 7), a confirmed positive (Phase 7b), and a located boundary (Phase 7c),
all against frozen thresholds and one clean independent control.
