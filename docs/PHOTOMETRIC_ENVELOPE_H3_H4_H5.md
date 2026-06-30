# The photometric controller's operating envelope — H3, H4, H5

Three experiments run 2026-06-30 against the photometric mirror-alignment rig, prompted by
a frontier-model deep-research report that scanned the public repos cold and ranked ten
hypotheses (`internal/feedback/Agent/deep-research-report.md`). These three are the
photometric-harness cluster (the report's ranks 3, 4, 5); together they characterize what
the closed-loop photometric controller depends on, where it beats the analytic oracle, and
how its fixed schedule trades speed against accuracy.

## Shared method (so the three are comparable)

- One MuJoCo harness: `env_v2.py` + `agents/photometric.py` + `agents/baselines.py` +
  `agents/bayes_particle.py`, driven through `experiments/stress_tests.py` primitives
  (matched-seed laser/pose, single-axis stressors).
- Every new capability is a **default-off flag** so the canonical agent and the existing
  Phase 6 / baseline anchors stay byte-identical: `SundogEnvV2.set_normal_bias()` (H3),
  `PhotometricAgent(adaptive=True)` (H5), `PhotometricAgent(infer_target=True)` (H4).
- 30 matched seeds/cell, bootstrap CIs + two-sided Mann–Whitney, terminal intensity =
  mean of the last 50 steps, convergence threshold 0.9. Each experiment carries a
  pre-registered falsifier.

## Results at a glance

| # | Claim | Verdict | Headline |
|---|---|---|---|
| H3 | The analytic oracle is only a *local* upper bound under model mismatch | **confirmed** (`flip_confirmed`) | Ordering flips: oracle wins near nominal, photometric overtakes once mismatch grows |
| H4 | The target-detector index is a load-bearing supervision channel | **confirmed** (`index_load_bearing`, 6/7 cells) | Remove the index and you are geometry-capped at ~60% identification |
| H5 | A belief-aware adaptive scheduler *dominates* the fixed schedule | strong claim **falsified**; favorable tradeoff | ~40% faster convergence for ~3% terminal cost, one sharp-spot failure mode |

## H3 — robustness under model mismatch  (`results/mismatch_robustness/`)

Lever: a mirror-calibration bias (`set_normal_bias`) — a fixed tilt of the reflecting
normal the analytic oracle cannot see. Sweep 0°→60°, 5 conditions.

- Nominal: photometric 0.945 vs oracle 0.936 (tied accuracy; oracle ~16× faster).
- 10°: **oracle wins** (0.97 vs 0.99) — near nominal the open-loop oracle is still ahead.
- 30°: **photometric wins** 0.80 vs 0.51 (p=3e-7); 40°: 0.64 vs 0.10 (p=3e-11).
- 50–60°: both fail (joint-limit wall; bias-aware ceiling collapses at 60°).
- The Bayes baseline collapses *harder* than the oracle (0.26 at 30°) — both model-based
  agents fail; only the model-free closed-loop controller degrades gracefully.

Honest scope: the win is **graceful degradation on terminal accuracy**, not convergence —
past 20° photometric never re-reaches the 0.9 threshold. It is "less bad," not "good."

## H4 — is the target index load-bearing supervision?  (`results/index_ablation/`)

Conditions (true target stays det 0): `known` (told the index), `inferred` (locks the
brightest detector seen during SCAN, unsupervised), `wrong` (peaks det 4).

- Nominal: known 0.945 vs inferred 0.707 vs wrong 0.249.
- Detector noise σ=0.05: inferred collapses to 0.133.
- Load-bearing in 6/7 cells; the report's "the layout makes the index obvious" hedge is
  **refuted** — the symmetric 8-detector ring makes the SCAN-time brightest-detector
  contest genuinely ambiguous (target's crossing peak 0.984 vs best other 0.982).
- The `identification_accuracy` column is confounded by intensity clipping — see
  `results/index_ablation/ANOMALY_idacc_clipping.md`. The terminal-at-true-target headline
  is unaffected.

## H5 — adaptive vs fixed scheduler  (`results/adaptive_scheduler/`)

`adaptive=True`: EI-plateau scan-stop + relative-drop reacquire. Fixed vs adaptive across
a 17-cell ladder (nominal + mirror_bias + beam_sigma + detector_noise + laser_height).

- `broadly_favorable_with_failure_modes`: 6 cells adaptive-dominant, 10 tradeoff, 1 regressed.
- Convergence AUC higher in 15/17 cells; time-to-threshold cut ~40% at nominal, up to
  −384 steps under mismatch. Scan exits at ~100 steps vs the fixed 200.
- Cost: a small (~1–5%) terminal drop in most cells; one hard failure mode at sharp beam
  (`beam_sigma=0.05`, ΔterminaI −0.287) where early scan-stop misses a narrow peak.
- No instability: re-acquire count is 0 in every cell (the report's named risk did not fire).
- Next lever if reopened: a coverage-aware scan floor that won't plateau-exit until the
  Lissajous has swept the workspace.

## What the three say together

All three turn on the controller being **closed-loop and model-free**:

- That is exactly why it is robust to model mismatch (H3) — it climbs the *true* signal,
  while the open-loop oracle and the model-based Bayes filter bake in a wrong model.
- It is also why it needs the index (H4): model-free feedback can peak *a* detector, but
  without being told *which*, geometry caps identification at ~60%.
- And it is why scheduling is a real lever (H5): the fixed schedule pays for guaranteed
  scan coverage with speed; adaptive trades coverage for ~40% faster convergence, and the
  one place that backfires is precisely where coverage matters most (a narrow spot).

So the envelope is: model-free feedback buys mismatch-robustness at the cost of (a) a
required "which target" supervision channel and (b) a coverage/speed tradeoff in the scan.

## Reproduce

```
python -m sundog.experiments.mismatch_robustness     # H3
python -m sundog.experiments.index_ablation          # H4
python -m sundog.experiments.adaptive_scheduler       # H5
```

## Status / scope

Portfolio operating-envelope evidence, not proof. Anchors preserved (default-off flags).
A webdev handoff exists for H3 only (`docs/H3_MISMATCH_PROMO_WEBDEV_HANDOFF.md`,
owner-gated); H4 and H5 are internal/methodological and not staged for public surfaces.
The other seven slate hypotheses (#1, #2, #6–#10) are not started.
