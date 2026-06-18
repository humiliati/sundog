# H1.2a Capped-Probe Results — Readback

Status: **PIPELINE GREEN; two pre-registration calibration issues surfaced.**
Ran 2026-06-18. Spec: [`H1_2_SMALL_BAKEOFF_SPEC.md`](H1_2_SMALL_BAKEOFF_SPEC.md)
§9 H1.2a. This is the capped probe (8 eval seeds × 3 cells); the binding branch
is selected by **H1.2b** at full size. Nothing here edits a pre-registered gate.

## Commands (all ran < 1 s each)

```powershell
node scripts/mesa-h1-build-coordinator-dataset.mjs --train-seeds 32 --val-seeds 16
python -m training.mesa.train_h1_arbiter --dataset results/mesa/h1-pantheon/h1_2a/dataset `
  --out results/mesa/h1-pantheon/h1_2a/models --epochs 20 --hidden-size 32 --seed 0
node scripts/mesa-h1-pantheon-eval.mjs --seeds 8
```

## Exit criterion (spec §9 H1.2a): MET

- Dataset builder: 17561 train / 8422 val rows, 2.5 s, **10 294 rows/s**; 13
  basin-captured rollouts for guard-positive signal.
- Trainer: param budget matched (council 3428 vs M-Adapter 3432, ratio 1.001,
  within 5%); guard val AUC 0.81; arbiter CE 0.86; M-Adapter MSE 0.105.
- Eval: 72 trials, 0.80 s, **90 trials/s**; authority cap `≤ 0.70` held on every
  step row; **no leakage** (feature schema partition has zero overlap; trainer X
  drawn only from `inference_features` + `guard_risk`).
- All output files written with stable columns.

## Eval aggregates (3-cell slate, 8 seeds)

| controller | mean S_T | success | basin-capture | sovereignty | breach-trial frac |
| --- | --- | --- | --- | --- | --- |
| Learned-P-Council | **0.969** | 46% | **0.000** | 0.626 | 1.00 |
| M-Adapter (equal-budget monolith) | 0.909 | 83% | 0.083 | — | — |
| Blind-Council (H1.1 arbiter, same heads) | 0.917 | 21% | 0.000 | 0.587 | 0.875 |

Gates: 1 = **false**, 2 = **true**, 3 = **true**, 4 = **false** →
auto-selected indicative branch `H1_2_ARBITER_NULL`. **That label is
misleading** — see diagnosis; the failures are in gates 1 and 4, and both are
calibration issues, not an arbiter that failed to use the signal.

## Diagnosis

1. **Substantive gates pass.** The learned council beats the equal-budget
   monolith on terminal alignment (0.969 vs 0.909) *and* eliminates the
   false-basin captures the monolith suffers (0.000 vs 0.083, in nominal and
   sensor-delay). The cap + label-trained guard discipline **helped** competence
   and proxy-resistance, did not cost them. This is the pro-pantheon direction.

2. **Gate 1 (+0.15 over blind) is mis-calibrated for the strong frozen head.**
   The threshold was set against H1.1's *weak* field head (blind ≈ 0.69). H1.2's
   frozen head is terminal-L-Signature, so even the **blind** council reaches
   0.917 — leaving < 0.15 of headroom below the 1.0 ceiling. The learned council
   *did* improve (0.969 > 0.917, and far above H1.1's 0.69), but a +0.15
   *absolute* gain over a 0.917 baseline is unreachable. Gate 1 must be
   re-expressed (gap-closure toward the ceiling, or a lower absolute step)
   **before H1.2b**.

3. **Gate 4 fails by construction of the chosen target — and exposes a thesis
   question.** The privileged-best-mix target optimally tracks the *true field*,
   so its per-step **max** weight averages **0.669** (84% of steps > 0.60), with
   **field the dominant role on 65%** of steps. The council faithfully
   reproduces this (sov 0.626) and therefore "breaches" the 0.60 audit on every
   trial. But the dominant role is the **field**, not the reward proxy.

   This is the tauroctony tension made numerical: **the 0.60 audit is symmetric
   across roles, while the thesis is not.** Sol / the field is the sky the gods
   live under — it is *allowed* to preside; the figure that must never become
   sovereign is the **bull** (the reward/proxy head). A field-dominant
   controller is the aligned state, not a breach. The sovereignty audit should
   plausibly bind **reward-head authority** (and total-non-field concentration),
   not penalize field primacy. Resolving this is a pre-registration decision for
   H1.2b (see Owed).

## Owed before H1.2b (pre-registration edits, legitimate post-H1.2a)

- **Re-express Gate 1** so it is reachable against a strong-head blind baseline
  (e.g., close ≥ X% of the blind→1.0 alignment gap, no added breaches).
- **Decide the sovereignty audit's shape**: symmetric 0.70 cap on all roles is
  retained (it held), but the *breach audit* should likely key on the **reward
  (bull) head** rather than the field (Sol) head — or on non-field
  concentration. Lock the new definition before H1.2b runs.
- Optionally regularize the arbiter target toward plurality (entropy term / lower
  target cap) if the thesis wants *distributed* authority rather than
  *field-presiding* authority — this is the conceptual fork the probe surfaced.

## Decision

The pipeline is validated and worth running at full H1.2b size. But H1.2b should
**not** run until Gate 1 and the sovereignty-audit shape are re-pinned, because
H1.2a shows the current gates would mislabel a result that is, on the
substance, the pro-pantheon outcome (beats the monolith, kills the basin).

---

## Addendum — v0.1 gate re-pin (same session, owner-decided)

Both owed edits were made (owner decisions, 2026-06-18), re-deriving the gates
from the thesis/headroom rather than reverse-fitting them to the numbers, then
the **same** H1.2a data was re-scored:

- **Gate 1 → gap-closure** (`≥ 40%` of the blind→1.0 alignment gap, no added
  bull breaches).
- **Sovereignty audit → bull-bound** (breach keys on the **reward head** holding
  `> 0.60` authority on `> 20%` of steps; field/Sol primacy is not a breach).
  The 0.70 structural hard cap on all roles is unchanged.

Re-scored aggregates (same run):

| controller | mean S_T | basin-capture | bull-breach frac | reward-authority frac |
| --- | --- | --- | --- | --- |
| Learned-P-Council | 0.969 | 0.000 | **0.125** | **0.037** |
| M-Adapter | 0.909 | 0.083 | — | — |
| Blind-Council | 0.917 | 0.000 | 0.292 | 0.131 |

Gates v0.1: 1 = **true** (gap-closure 0.631), 2 = **true**, 3 = **true**,
4 = **true** → indicative branch **`H1_2_SUPPORT`**.

The bull-bound audit is load-bearing, not cosmetic: the **learned arbiter
suppresses reward-head sovereignty more than the blind blend** (bull-breach
0.125 vs 0.292; reward-authority 3.7% vs 13.1% of steps) while letting the field
preside — the measured form of "Sol presides, the bull is kept from the throne."

**Honesty lock.** H1.2a is INDICATIVE only. Because the v0.1 gates were set in
light of H1.2a, this re-score is a *consistency check that the re-pinned gates
classify the observed pro-pantheon behavior as support* — **not** independent
confirmation. The binding test is **H1.2b** at full size (256/64/64 seeds, 12
cells), run fresh against the now-locked v0.1 gates, with thresholds frozen
before results are seen (admission checklist §10).
