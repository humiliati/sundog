# Caveat: the `identification_accuracy` column is confounded by intensity clipping

`manifest.json` / `level-summary.csv` report a non-monotonic identification accuracy
for the `inferred` condition across detector noise:

| detector_noise σ | 0.00 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|
| id-accuracy | 0.77 | 0.47 | 0.13 | 0.37 | **0.93** |

The recovery to 0.93 at σ=0.20 is **not** a real effect and must not be read as "noise
helps inference." It is an artifact of the environment clipping detector intensities to
`[0, 1]` interacting with the unsupervised inference rule.

Mechanism (causally confirmed): inference records the detector at the running-max SCAN
step under a strict `>` update, so the first detector to reach the highest reading locks
in. The target's true scan peak is ~0.984; under large noise it saturates to exactly 1.0,
and once a detector hits the 1.0 ceiling nothing can strictly exceed it — so the
first-to-saturate sticks, and the target (det 0) wins that race ~93% of the time.

Causal test: removing the upper clip flattens the curve completely —

| σ | id-acc clip `[0,1]` | id-acc no upper clip |
|---|---|---|
| 0.05 | 0.13 | 0.33 |
| 0.10 | 0.37 | 0.33 |
| 0.20 | 0.93 | 0.43 |

The σ=0.20 spike vanishes (0.93 → 0.43). The honest reading: inference is
**geometry-limited, not noise-limited** — even noise-free, the target is the
cleanest-crossed detector only ~60% of the time (its scan peak 0.984 vs the best other
detector's 0.982, a 0.002 margin on the symmetric 8-detector ring).

**This does not affect the H4 headline.** Terminal intensity is scored on the *true*
target (det 0) and is unaffected by the clipping artifact; `known ≫ inferred` holds across
the ladder, so the index remains load-bearing. Only this diagnostic column is confounded.
