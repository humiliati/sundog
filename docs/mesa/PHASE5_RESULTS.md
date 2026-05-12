# Mesa Phase 5 - Selection-Pressure Result Note (v1)

This document records the first Phase 5 Small-tier selection-pressure result.
It is the companion result note for [`PHASE5_SPEC.md`](PHASE5_SPEC.md).

Status: Small-tier training slate **complete** for Phase 5 v1 axes A-C.
Phase 3 probe-slate evaluation and Phase 4 intervention-battery evaluation
are complete for the eight new Phase 5 policies. Medium follow-up is not
started.

## 1. Summary

Phase 5 gives the program three useful updates:

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

The cleanest Phase 5 headline is now: the signature anchor protects mixed
training up to a measurable lambda threshold, and objective *shape* matters
more than expected for Sundog cost.

## 2. Artifacts

Training checkpoints and policy JSONs live under:

`results/mesa/phase2-matched-capacity/`

Probe-slate outputs for new Phase 5 policies live under:

`results/mesa/phase3-probe-slate/phase5_*`

Intervention-battery outputs for new Phase 5 policies live under:

`results/mesa/phase4-intervention-battery/phase5_*`

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

## 7. Next Moves

1. Run Medium follow-up for `lambda in {0.3, 0.5, 0.7}` as specified.
2. Promote terminal-only signature to a Medium follow-up candidate, even
   though it was not in Phase 5 v1's Medium gate.
3. Add Phase 6 representation probes for three policies first:
   terminal-signature Small, canonical L-Reward Small, and reward->signature
   curriculum Small. That trio cleanly separates successful signature control,
   fixed-attractor basin control, and basin-erased-but-not-recovered control.

## 8. Versioning

This document is version `v1`.

- `v1` (2026-05-11): records Small-tier Phase 5 training, probe-slate, and
  intervention-battery results for axes A-C.
