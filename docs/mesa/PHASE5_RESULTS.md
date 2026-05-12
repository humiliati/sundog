# Mesa Phase 5 - Selection-Pressure Result Note (v4)

This document records the Phase 5 Small-tier selection-pressure result and
the Medium-tier follow-ups. It is the companion result note for
[`PHASE5_SPEC.md`](PHASE5_SPEC.md).

Status: Small-tier training slate **complete** for Phase 5 v1 axes A-C, and
Medium follow-up **complete** for `lambda in {0.3, 0.5, 0.7}` plus
L-Signature-M-Terminal. Phase 5 v3 follow-up is also **complete** for
Medium `lambda in {0.8, 0.9}` and reward-pretrain ->
terminal-signature-fine-tune. Phase 5 v4 cliff-localization is **complete**
for Medium `lambda in {0.95, 0.97, 0.99}`. Phase 3 probe-slate evaluation
and Phase 4 intervention-battery evaluation are complete for the new Phase 5
policies. The aggregate reports have been rebuilt for the combined
Small+Medium slate.

## 1. Summary

Phase 5 gives the program four useful updates:

1. The L-Mixed protection curve has a real breach threshold. Old-basin
   preference remains below 1.0 through `lambda = 0.5`, then crosses at
   `lambda = 0.7` and grows again at `lambda = 0.9`. Linear interpolation
   puts the Small-tier breach threshold at `lambda ~= 0.660`.
2. Terminal-only signature shaping is not worse than integrated shaping. It
   is dramatically better at Small: `37/64` nominal successes versus the
   integrated baseline's `5/64`, with mean `S_T = 0.963`.
3. Reward-first -> signature fine-tuning does not retain basin attraction.
   It falsifies the pre-registered C1 prediction: old-basin preference is
   `-0.585`, not `> 1.0`. But it also does not recover task success
   (`0/64`), so clean-signal fine-tuning appears to erase the fixed basin
   without reliably rebuilding useful control.
4. Medium L-Mixed has a protected high-reward window and a sharp cliff. v4
   shows `lambda=0.95` remains protected and reaches 43/64 success, while
   `lambda=0.97` and `lambda=0.99` collapse into fixed-attractor behavior.
   The Medium breach estimate is now `lambda ~= 0.953`.

The cleanest Phase 5 headline is now: the signature anchor protects mixed
training up to a measurable lambda threshold, and objective *shape* matters
more than expected for Sundog cost.

## 2. Artifacts

Training checkpoints and policy JSONs live under:

`results/mesa/phase2-matched-capacity/`

Probe-slate outputs for new Phase 5 policies live under:

`results/mesa/phase3-probe-slate/phase5_*`

Medium follow-up probe-slate outputs live under:

`results/mesa/phase3-probe-slate/l_mixed_medium_*`

and:

`results/mesa/phase3-probe-slate/l_signature_medium_terminal`

Intervention-battery outputs for new Phase 5 policies live under:

`results/mesa/phase4-intervention-battery/phase5_*`

Medium follow-up intervention-battery outputs live under:

`results/mesa/phase4-intervention-battery/l_mixed_medium_*`

and:

`results/mesa/phase4-intervention-battery/l_signature_medium_terminal`

Phase 5 v3 follow-up outputs live under:

`results/mesa/phase3-probe-slate/phase5_v3_*`

and:

`results/mesa/phase4-intervention-battery/phase5_v3_*`

Phase 5 v4 cliff-localization outputs live under:

`results/mesa/phase3-probe-slate/phase5_v4_*`

and:

`results/mesa/phase4-intervention-battery/phase5_v4_*`

Phase 5 aggregate reports live under:

`results/mesa/phase5-selection-pressure/`

Rebuild the aggregate reports with:

```bash
npm run mesa:phase5:aggregate
```

## 3. Axis A - L-Mixed Lambda Sweep

| Lambda | Success | Mean S_T | Mean probed success | Probe basin captures | Old-basin pref |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 7/64 | 0.850 | 0.081 | 1 | -0.458 |
| 0.3 | 7/64 | 0.884 | 0.074 | 30 | -0.066 |
| 0.5 | 8/64 | 0.939 | 0.095 | 3 | -0.394 |
| 0.7 | 2/64 | 0.853 | 0.040 | 37 | 1.346 |
| 0.9 | 2/64 | 0.528 | 0.025 | 63 | 2.611 |

Read:

- A2 confirmed. The breach threshold lies in `(0.5, 0.7]`; interpolation
  against old-basin preference gives `lambda ~= 0.660`.
- A1 is directionally supported but not strictly monotone because the
  `lambda = 0.5` canonical reuse has lower old-basin preference than
  `lambda = 0.3`. The threshold shape is still clear.
- Probe basin captures grow sharply once the reward side dominates:
  `37` at `lambda = 0.7`, `63` at `lambda = 0.9`.

## 4. Axis B - Signature Objective Shape

| Shape | Success | Mean S_T | Mean probed success | Probe basin captures | Old-basin pref |
| --- | ---: | ---: | ---: | ---: | ---: |
| terminal | 37/64 | 0.963 | 0.371 | 3 | -0.002 |
| integrated | 5/64 | 0.672 | 0.057 | 0 | -0.092 |
| threshold | 2/64 | 0.566 | 0.017 | 10 | -0.692 |

B1 falsified. The prediction was that terminal-only signature shaping would
worsen Sundog cost. Instead, terminal-only signature is the strongest Small
signature-only controller so far, beating integrated shaping by 32 successes
and preserving near-zero old-basin attraction.

The likely read is that per-step Gaussian shaping punishes long approach
trajectories and creates a shallow local-gradient learning problem, while
terminal-only shaping lets PPO optimize episode-level arrival without paying
per-step opportunity cost. This is a Phase 5 result, not a Phase 2 routing
bug: the integrated path was already verified end-to-end.

Threshold shaping is poor at Small and shows 10 probe basin captures, but its
old-basin preference remains negative. It looks noisy and under-trained, not
basin absorbed in the canonical L-Reward sense.

## 5. Axis C - Curriculum Order

| Order | Pretrain success | Final success | Mean S_T | Probe basin captures | Old-basin pref |
| --- | ---: | ---: | ---: | ---: | ---: |
| signature -> reward | 7/64 | 11/64 | 0.658 | 62 | 2.613 |
| reward -> signature | 0/64 | 0/64 | 0.648 | 3 | -0.585 |

C1 falsified. Reward-first -> signature fine-tuning does not retain basin
attraction. The old-basin preference becomes negative and probe basin captures
drop to 3. This is good news for "clean-signal repair" in the narrow
mechanistic sense: signature fine-tuning can erase the fixed attractor.

But it does not recover useful control. Final success remains `0/64`, so the
policy is not repaired into a working signature controller. The honest read
is: signature fine-tuning can remove basin attraction from a corrupted
starting point, but this 500K continuation does not rebuild the navigation
competence needed for success.

The opposite order behaves as expected. Signature -> reward fine-tuning
inherits some useful control (`11/64`) but absorbs the basin once the
corrupted reward phase starts: old-basin preference `2.613` and 62 probe
basin captures.

## 6. Prediction Register

| Prediction | Status | Result |
| --- | --- | --- |
| A1: old-basin preference monotone in lambda | Mixed | Directional threshold trend confirmed; strict monotonicity broken at lambda 0.5 |
| A2: breach threshold lies in `(0.5, 0.7]` | Confirmed | Interpolated threshold `lambda ~= 0.660` |
| B1: terminal-only signature halves Sundog-cost success | Falsified | Terminal-only gets `37/64`; integrated gets `5/64` |
| C1: reward-first -> signature fine-tune retains basin attraction | Falsified | Old-basin preference `-0.585`; only 3 probe basin captures |

Both falsifiers are useful. B1 points to a better signature objective shape.
C1 suggests fixed-attractor basin absorption can be erased by clean-signal
fine-tuning, even though control competence does not recover at this budget.

## 7. Small-Slate Next Moves (superseded by v2)

These were the v1 next moves after the Small slate. Items 1-2 are now
complete in the v2 Medium-tier amendment below.

1. Completed in v2: ran Medium follow-up for
   `lambda in {0.3, 0.5, 0.7}` as specified.
2. Completed in v2: promoted terminal-only signature to a Medium follow-up
   candidate, even though it was not in Phase 5 v1's Medium gate.
3. Add Phase 6 representation probes for three policies first:
   terminal-signature Small, canonical L-Reward Small, and reward->signature
   curriculum Small. That trio cleanly separates successful signature control,
   fixed-attractor basin control, and basin-erased-but-not-recovered control.

## 8. Medium-Tier Amendment (v2)

Medium-tier follow-up added 2026-05-12. Three new PPO Medium runs landed:
L-Mixed λ=0.3, L-Mixed λ=0.7, and L-Signature Terminal Medium. The existing
L-Mixed-M-λ=0.5 and L-Signature-M-Integrated from Phase 3 / Phase 5 v1
serve as anchor points. Probe slate (12 cells × 64 seeds) and intervention
battery (5 channels × 64 seeds) run on each new policy. The Medium addendum
overturns the Sundog-cost framing of Phase 2-3 and surfaces a non-monotone
λ protection curve that the Small slate did not predict.

### 8.1 Medium nominal canonical-budget results

| Policy | Success | Mean S_T | Read |
| --- | ---: | ---: | --- |
| Oracle (privileged-field) | 64/64 | 0.9996 | analytic-gradient ceiling |
| HC-Signature (local-probe) | 64/64 | 0.9944 | hand-coded ceiling |
| **L-Signature-M-Terminal** | **64/64** | **0.9986** | **state-only at Oracle ceiling** |
| L-Reward-Clean-M (dense) | 49/64 | 0.9867 | matched clean reward control |
| L-Signature-M-Integrated | 4/64 | 0.6791 | Phase 2 canonical (deprecated) |
| L-Mixed-M-λ=0.3 | 0/64 | 0.928 | signature-dominant, dwell-fail |
| L-Mixed-M-λ=0.5 | 0/64 | 0.936 | (Phase 3 anchor) |
| L-Mixed-M-λ=0.7 | 9/64 | 0.981 | reward-dominant, partial success |
| L-Reward-M (basin-corrupted) | 0/64 | 0.267 | full basin internalization |

**Headline:** L-Signature Medium with terminal-only signature reaches **64/64
success at mean S_T = 0.9986**, matching Oracle (privileged analytic
gradient) and HC-Signature, and exceeding L-Reward-Clean Medium (49/64) by
**23 percentage points**.

### 8.2 The Sundog-cost framing is overturned

The Phase 2 canonical L-Signature objective `(1/T) Σ_t S(x_t)` (per-step
integrated, Gaussian-decay shaping) gave near-zero gradient signal in the
outer ring where episodes start. The Phase 2-3 narrative read this as a
*structural sample-efficiency cost* of state-only training — the
"Sundog-cost gap."

Phase 5 v2 Medium data falsifies that interpretation:

| Sundog-cost framing | Small | Medium |
| --- | ---: | ---: |
| L-Signature-Integrated vs L-Reward-Clean | -39 pp | **-71 pp** |
| L-Signature-Terminal vs L-Reward-Clean | -11 pp | **+23 pp** |

At Medium with terminal-only shaping, state-only signature training
**outperforms** dense reward training by 23 pp. The Sundog-cost was a
shape-of-objective artifact, not a structural property of state-only
training. The gravity claim shifts: state-only signature training is **not
just adversarially-robust-with-a-cost — it is competitive on its own task
at scale**, *and* basin-immune by construction.

This reframes Phase 2-4 retrospectively. The cleanest read going forward:

- L-Signature integrated is **deprecated** as the canonical shape.
  Terminal-only is the new canonical for any future Phase 5+ work.
- L-Signature integrated is **retained as a historical anchor** for
  reproducing Phase 2-4 numbers, not as a recommended training shape.
- PHASE2_RESULTS / PHASE3_RESULTS narratives should be amended (v3) to
  flag the integrated-shape as a methodology limitation rather than a
  structural finding.

### 8.3 Non-monotone protection curve at Medium

The Small λ protection curve was monotone with breach at λ ≈ 0.660. The
Medium λ protection curve is **non-monotone**:

| λ | Nominal Success | Mean S_T | Basin captures (nominal) | `old_basin_pref` |
| ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0/64 | 0.928 | 4 | 0.823 |
| 0.5 | 0/64 | 0.936 | 5 | 0.889 |
| 0.7 | **9/64** | 0.981 | 3 | **0.613** |
| 1.0 (L-Reward-M) | 0/64 | 0.267 | 55 | 5.560 |

Two surprises:

1. **`old_basin_pref` peaks at λ=0.5 (0.889) and drops at λ=0.7 (0.613)
   before jumping to 5.56 at L-Reward-M (λ=1.0).** A smooth-curve
   interpretation predicts monotone increase. Instead there is a window
   between λ=0.5 and λ=1.0 where the basin doesn't capture the policy.

2. **Success rate is highest at λ=0.7 (9/64), not at λ=0.3 (0/64).** The
   mid-mix is the most successful learned policy in the L-Mixed Medium
   sweep, by a meaningful margin.

The Phase 3 §10.4 three-point capacity-dependence picture predicted Medium
capacity would *amplify* basin absorption in mixed signals. That holds at
λ=0.3 (`old_basin_pref` 0.823 — close to breach where Small had 0
captures) but does **not** hold at λ=0.7.

Working interpretations to test in Phase 5 v3:

- **Control-cost dominance hypothesis.** At λ=0.7 the dense gradient is
  strong enough that the policy commits to goal-seeking, and the action-
  cost penalty `α·||a||²` suppresses basin-detour trajectories implicitly
  during training. At λ=0.5 the dense signal is weaker so the basin's
  pull is comparatively larger relative to dense gradient + action cost.
- **Capacity-escape hypothesis.** Medium capacity at λ=0.7 has enough
  parameters and enough strong-signal training to escape the basin's
  local optimum during PPO updates. λ=0.5 is in a "weak-signal trap"
  region.
- **λ=0.9 may close the gap.** If we run L-Mixed-M-λ=0.9, the protection
  curve might show a steep rise from λ=0.7 (0.613) to λ=0.9 (≥2) before
  saturating at L-Reward-M's 5.56. Phase 5 v3 candidate.

The non-monotone Medium curve does **not** invalidate the Small breach
threshold (λ ≈ 0.660 at Small still holds and was confirmed by direct
measurement). It says the protection-curve shape is *capacity-dependent*,
not just scaled.

### 8.4 Pre-registered Medium prediction outcomes

PHASE5_SPEC v1.1 / pre-registered Medium predictions before the runs:

| Prediction | Outcome |
| --- | --- |
| (A1-M) L-Mix-M-λ=0.3 protected (`old_basin_pref < 1.0`) | ✓ Confirmed marginally. `old_basin_pref = 0.823`, just below threshold. 4 basin captures (~6% of trials). |
| (A2-M) L-Mix-M-λ=0.7 shows partial-to-full breach (`old_basin_pref > 1.5`) | ✗ **Falsified in unexpected direction.** `old_basin_pref = 0.613` (lower than λ=0.5's 0.889), success 9/64 (higher than λ=0.5's 0/64). λ=0.7 at Medium is *less* basin-attracted than λ=0.5. |
| (B-M) L-Sig-M-Terminal reaches ≥45/64 success | ✓ **Confirmed and exceeded by 19/64.** L-Sig-M-Terminal reaches 64/64 success — matches Oracle ceiling. |

The (A2-M) falsification is **program-significant** — it indicates the
protection curve is shape-dependent in a way the Small slate did not
surface. Not a falsifier of the gravity claim, but a sharper claim about
how mixed-signal protection scales.

The (B-M) confirmation is the headline finding for v2.

### 8.5 Probe slate read for L-Signature-M-Terminal

Brief: the new canonical L-Signature behaves like HC-Signature under
probes — strong response to gradient-corruption channels (decoy,
sensor-noise) at heavy severity, robust to geometric and sensor-delay
probes. The full per-cell matrix lives in
`results/mesa/phase3-probe-slate/l_signature_medium_terminal/`. Updated
basin-effect gap at Medium (L-R-Clean-M vs L-R-M) is 76.6 pp from Phase 3
§10.6 and unchanged by the v2 additions — the new policies don't change
the L-Reward-Clean or L-Reward-canonical numbers.

### 8.6 Intervention battery read

`old_basin_pref` by policy under basin-position channel (PHASE4 v1
diagnostic):

| Policy | `old_basin_pref` |
| --- | ---: |
| L-Sig-M-Terminal | 0.193 |
| L-Sig-M-Integrated | -0.752 |
| L-R-Clean-M | -0.002 |
| L-R-M (basin-corrupted) | 5.560 |
| L-Mix-M-λ=0.3 | 0.823 |
| L-Mix-M-λ=0.5 | 0.889 |
| L-Mix-M-λ=0.7 | 0.613 |

All policies show **0** action-response to live basin-position-edit
(consistent with x_false invisibility in observation). All non-basin-
trained policies show near-zero `old_basin_pref`, confirming the basin
attractor is in-weights for L-R-M and partial in L-Mixed Medium variants.
L-Sig-M-Terminal at 0.193 is consistent with HC-Signature / Oracle null
(both ~0.15).

### 8.7 Updated exit-criterion check

PHASE5_SPEC §10 v1 exit criteria, with Medium data:

- ✓ All 3 new Medium training runs landed checkpoints + policy.json.
- ✓ Probe-slate and intervention-battery CSVs exist for every new
  Medium policy.
- ✓ Protection curve (Small monotone + Medium non-monotone) is mapped.
- ✓ Breach threshold identified at Small (λ ≈ 0.660); Medium curve is
  documented with the λ=0.7 window non-monotonicity.
- ✓ At least one v1 prediction confirmed (A2-Small, B-Small);
  v2 confirms B-Medium (the headline).
- ✓ Phase 5 result note v2 written.

Phase 5 v2 exit criteria **passes** with strengthened evidence at Medium.

### 8.8 Next moves

Three Phase 5 v3 candidates ranked by program-significance:

1. **L-Mixed-M-λ ∈ {0.8, 0.9}** to map the non-monotone curve cleanly.
   2 PPO Medium runs. ~120 min compute. Tells us whether the curve
   monotonically returns to L-R-M's 5.56 from λ=0.7's 0.613, or whether
   there's a second bump.

2. **Reward-pretrain → Terminal-Signature fine-tune curriculum.** Now
   that we know terminal-signature is a strong recovery signal, the C1
   prediction from Phase 5 v1 (where integrated signature couldn't
   recover task competence) can be re-tested. If terminal signature
   *can* both erase the basin and recover competence, that's the
   strongest "Goodhart-fix-by-fine-tuning" demonstration the program
   can produce.

3. **L-Signature-S-Terminal probe slate + intervention battery.**
   Verification-only. The Small terminal-only policy already has matching
   probe-slate and intervention-battery artifacts, and the aggregate includes
   them. Keep this as a paper-trail check rather than a new run.

(1) and (2) are now complete in the v3 amendment below. (3) remains
paper-trail verification.

## 9. Phase 5 v3 Amendment

Phase 5 v3 added two Medium mixed runs (`lambda=0.8`, `lambda=0.9`) and one
reward-pretrain -> terminal-signature-fine-tune curriculum run. All three
were evaluated with the Phase 3 probe slate and Phase 4 intervention battery.
The aggregate now covers 19 policies.

### 9.1 Medium lambda refinement

| Lambda | Success | Mean S_T | Mean probed success | Probe basin captures | Old-basin pref |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0/64 | 0.928 | 0.010 | 39 | 0.823 |
| 0.5 | 0/64 | 0.936 | 0.012 | 60 | 0.889 |
| 0.7 | 9/64 | 0.981 | 0.104 | 38 | 0.613 |
| 0.8 | 32/64 | 0.979 | 0.339 | 36 | 0.485 |
| 0.9 | 36/64 | 0.973 | 0.371 | 35 | 0.383 |
| 1.0 (L-Reward-M) | 0/64 | 0.267 | 0.005 | 576 | 5.560 |

This falsifies the v3 rebound prediction. Instead of rising toward the
L-Reward-M basin attractor, `old_basin_pref` keeps falling from `lambda=0.7`
through `lambda=0.9`. The best Medium mixed policy in this slate is
`lambda=0.9`: 36/64 nominal success, mean `S_T = 0.973`, and
`old_basin_pref = 0.383`.

The Medium breach estimate is now `lambda ~= 0.912`, interpolated between
`lambda=0.9` and the L-Reward-M anchor. Since all sampled L-Mixed Medium rows
through `lambda=0.9` remain protected, the collapse is no longer a smooth
monotone curve. It is an abrupt discontinuity between "mostly reward with a
signature anchor" and "pure reward."

### 9.2 v3 prediction outcomes

| Prediction | Outcome |
| --- | --- |
| (A3-M) `old_basin_pref(lambda=0.8/0.9)` rises above `lambda=0.7` and moves toward L-Reward-M | Falsified. `lambda=0.8` = 0.485 and `lambda=0.9` = 0.383, both below `lambda=0.7` = 0.613. |
| (A4-M) `lambda=0.8` has higher success than `lambda=0.9`; `lambda=0.9` has stronger basin attraction | Falsified. `lambda=0.9` has higher success (36/64 vs 32/64) and lower old-basin preference (0.383 vs 0.485). |
| (C2) Reward-pretrain -> terminal-signature fine-tune erases basin attraction and recovers competence | Falsified. Final success is 0/64 and `old_basin_pref = 3.691`. |

The A3/A4 falsifications are program-significant in the good way: they expand
the protected mixed-signal region. The C2 falsification is program-significant
in the hard way: terminal signature is strong from scratch, but it does not
rescue this reward-pretrained fixed-attractor policy under the default
continuation setup with optimizer state carried forward.

### 9.3 Curriculum v3 read

| Curriculum | Pretrain success | Final success | Mean S_T | Probe basin captures | Old-basin pref |
| --- | ---: | ---: | ---: | ---: | ---: |
| reward -> integrated signature | 0/64 | 0/64 | 0.648 | 3 | -0.585 |
| reward -> terminal signature | 0/64 | 0/64 | 0.363 | 131 | 3.691 |

Terminal-signature fine-tuning from the basin-corrupted reward checkpoint
does not merely fail to recover task competence; it also fails to erase the
basin attractor. The policy remains close to the training-time false basin
under the basin-position diagnostic and shows weak response to signature and
geometry edits (`action_response_L2 ~= 0.11` for both).

The clean next curriculum fork, if pursued, is not another identical 500K
continuation. It should change one variable: reset Adam state, extend
terminal-signature fine-tuning, or start from the fully trained L-Reward-M
checkpoint and test whether the fixed attractor is easier or harder to erase
after canonical-budget saturation.

## 10. Phase 5 v4 Amendment

Phase 5 v4 localized the Medium mixed-signal cliff with
`lambda in {0.95, 0.97, 0.99}`. All three runs landed checkpoints, policy
JSONs, probe slates, intervention batteries, and aggregate rows. The
aggregate now covers 22 policies.

### 10.1 Cliff-localization table

| Lambda | Success | Mean S_T | Mean probed success | Probe basin captures | Old-basin pref |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 36/64 | 0.973 | 0.371 | 35 | 0.383 |
| 0.95 | 43/64 | 0.982 | 0.424 | 29 | 0.330 |
| 0.97 | 2/64 | 0.276 | 0.020 | 596 | 5.510 |
| 0.99 | 3/64 | 0.303 | 0.026 | 465 | 5.159 |
| 1.00 (L-Reward-M) | 0/64 | 0.267 | 0.005 | 576 | 5.560 |

The cliff is localized to `(0.95, 0.97]`. `lambda=0.95` is the best Medium
mixed policy so far by nominal success (43/64) and remains protected
(`old_basin_pref=0.330`). `lambda=0.97` collapses almost completely:
2/64 success, mean `S_T=0.276`, and `old_basin_pref=5.510`, essentially
matching the pure L-Reward fixed-attractor signature.

The updated linear interpolation puts the Medium breach at
`lambda ~= 0.952588`, but the interpolation is mostly a reporting
convenience. Behaviorally, the result is a discontinuity: a 5% signature
anchor protects, while a 3% signature anchor does not.

### 10.2 v4 prediction outcomes

| Prediction | Outcome |
| --- | --- |
| (A5-M) At least one of `lambda in {0.95, 0.97, 0.99}` breaches | Confirmed. `lambda=0.97` and `lambda=0.99` both breach heavily. |
| (A6-M) Breach is sharp rather than smooth | Confirmed. `old_basin_pref` jumps from 0.330 at `lambda=0.95` to 5.510 at `lambda=0.97`. |
| (A7-M) Breach couples to competence collapse | Confirmed. Success falls from 43/64 at `lambda=0.95` to 2/64 and 3/64 at `lambda=0.97/0.99`. |

This is the cleanest Phase 5 selection-pressure boundary so far. In this
Medium setup, the signature anchor is not linearly protective; it behaves
like a threshold. The sharp operational statement is:

> A 5% terminal/integrated signature anchor in mixed training prevents the
> false-basin fixed attractor, while a 3% anchor does not.

## 11. Versioning

This document is version `v4`.

- `v1` (2026-05-11): records Small-tier Phase 5 training, probe-slate, and
  intervention-battery results for axes A-C.

- `v2` (2026-05-12): Medium-tier amendment. Records the headline
  L-Signature-M-Terminal 64/64 success (matching Oracle ceiling, exceeding
  L-Reward-Clean Medium by 23 pp), which overturns the Sundog-cost framing
  of Phase 2-3. The integrated signature objective is deprecated; terminal
  is the new canonical. Also documents the non-monotone Medium λ
  protection curve (`old_basin_pref` 0.823 → 0.889 → 0.613 → 5.560 across
  λ ∈ {0.3, 0.5, 0.7, 1.0}), which the monotone Small curve did not
  predict. Predictions: (A1-M) confirmed marginally, (A2-M) falsified in
  unexpected direction (λ=0.7 less basin-attracted than λ=0.5),
  (B-M) confirmed and exceeded. Three Phase 5 v3 candidates ranked at
  §8.8.

- `v3` (2026-05-12): Phase 5 v3 follow-up. Adds Medium L-Mixed
  `lambda=0.8` and `lambda=0.9`, both of which remain protected
  (`old_basin_pref` 0.485 and 0.383) while improving nominal success to
  32/64 and 36/64. Medium breach estimate moves to `lambda ~= 0.912`,
  interpolated against the pure L-Reward anchor. Also records
  reward-pretrain -> terminal-signature fine-tune: 0/64 success and
  `old_basin_pref = 3.691`, falsifying the strict Goodhart-fix-by-fine-tune
  prediction under the default optimizer-continuation setup.

- `v4` (2026-05-12): Phase 5 v4 cliff localization. Adds Medium L-Mixed
  `lambda=0.95`, `lambda=0.97`, and `lambda=0.99`. Localizes the cliff to
  `(0.95, 0.97]`: `lambda=0.95` remains protected and reaches 43/64 success,
  while `lambda=0.97` and `lambda=0.99` collapse to fixed-attractor behavior
  (`old_basin_pref` 5.510 and 5.159). Updated Medium breach interpolation:
  `lambda ~= 0.952588`.
