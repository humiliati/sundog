# H3-PC Result — BOUNDED-PARTIAL CONCEALMENT: the banked clf_d c-suppression was probe-relative and sample-starved (verdict c)

> **Run 2026-06-11 against the FROZEN pre-registration** (`H3_PROBE_CEILING_PREREG.md`, frozen
> 2026-06-10 post-adversarial-review). HS4 of slate 2026-06-10 — the slate's TOP PICK. Exact unchanged
> commands: `python scripts/h3_probe_ceiling.py` · `python scripts/test_h3_probe_ceiling.py` (7/7).
> Published as a verification receipt (owner decision 2026-06-11; un-promoted; not peer-reviewed).
> Language rule: this is a **probe-access** result in a ground-truth synthetic
> substrate — no claims about introspection/confabulation as mental phenomena. Attribution: H3 v2;
> amnesic probing (Elazar & Goldberg); V-information (Xu et al. 2020); LEACE (Belrose et al. 2023);
> KSG (Kraskov et al. 2004).

## Headline

**Pre-registered outcome (c) fired: BOUNDED-PARTIAL CONCEALMENT, certified ceiling c-R² = 0.1256.**
The banked H3 claim "clf_d suppresses c to ~0 (probe-robust under any probe)" does NOT survive a
calibrated battery at scale: on the SAME deterministically re-derived clf_d reps at λ=2.0,

- a **Nyström kernel-ridge probe (γ=1.0, α=0.1; certified floor 0.10) recovers c at pool-CV 0.1215 →
  frozen-split 0.1256** (replication near-perfect on a once-touched 10k split);
- **plain ridge itself reads 0.0593 → 0.0673 at n=20k** — the banked 0.0063 was an **N=2000 artifact**:
  the learning curve (ridge 0.007 → 0.023 → 0.037 → 0.059 across 2k→20k) passes through the banked
  value at the banked sample size, exactly. The banked *number* was right; its *reading* was
  sample-starved;
- the **KSG MI leg independently confirms** residual c-information: SIGNAL above the 99-shuffle null
  max at both liveness-passing PCA-k (k=16: 0.0295 vs 0.0108; k=32: 0.0301 vs 0.0145);
- the banked "strong probe" (MLP 128,64) and kNN are **MEMBER-BLIND** — unable to detect even the
  0.20-ridge-equivalent calibrated injection — so their banked/contemporary silence on the real reps
  certifies nothing. The certificate rests on the live members {P1 floor 0.10, P4 floor 0.10,
  P5 floor 0.20} + MI.

**What survives, rescoped:** the H3 objective gap is real and large — reg_c 0.51 vs the clf_d ceiling
0.126 (gap ≈ 0.38 under the strongest probe either side has faced). Training objective remains the
dominant, controlled effect. What changes is the suppression's character: **partial concealment, not
destruction-to-zero** — and no outcome here reaches the (a) counterexample bar (0.30/0.24), so the
suppression is also genuinely strong. The middle band is exactly what fired, as pre-registered.

## Scorecard (all gates from the frozen prereg)

| Gate / leg | Result | Value |
|---|---|---|
| C0 continuity (raw mean washes c, new 20k pool) | PASS | −0.0042 (≤ 0.05) |
| Positive control (reg_c reps carry c) | PASS | ridge 0.4977 (≥ 0.45) |
| Calibration 0.10 / 0.20 (bisection converges) | PASS | α=0.00462 / 0.00823, both on-target ±1e−5 |
| Liveness P2 MLP / P3 kNN | **MEMBER-BLIND** | blind at 0.10 AND 0.20 (pilot-anticipated) |
| Liveness P4 Nyström / P5 GBT | live | floors **0.10** / **0.20** |
| MI leg | **live** | injection detected at PCA-k ∈ {16, 32} (k=32 incl.) |
| Battery on real reps (CV → once-touched split) | — | P1 0.0593→**0.0673** · P4 0.1215→**0.1256** · P5 0.0020→0.0018 · (P2 −0.047 / P3 −0.005, no weight) |
| MI on real reps | **SIGNAL** | above null max at k=16 and k=32 |
| Verdict | **(c)** | counted set {P1, P4}; **ceiling 0.1256**; sub-outcomes: none (no UNREPLICATED-POSITIVE; band non-empty) |

Learning curve (reported, non-gating; best member / ridge): n=2k 0.018/0.007 · 5k 0.059/0.023 ·
10k 0.086/0.037 · 20k 0.121/0.059 — **still rising at n=20k**: the ceiling is a lower bound at the
declared budget, not a plateau.

## Correction issued to the banked H3 result

`H3_POOLED_SHADOW_RESULT.md` carries a dated correction (2026-06-11): the clause "suppresses c to ~0
(probe-robust under any probe)" is rescoped to "suppresses c partially — residual c is recoverable at
R² ≈ 0.13 (kernel probe, n=20k, replicated) and ≈ 0.067 (linear ridge, n=20k), with independent KSG MI
confirmation; the 2024-banked ≈0.006 readout was N=2000 + two-probe-battery relative, and one of those
two probes (the strong MLP) is demonstrably blind to calibrated linear injections at this scale."
The objective-dependence headline and the DETERMINE half are unaffected.

## Honest boundaries (pre-stated in the prereg; all bind)

- The ceiling is **battery-, sample-, and direction-relative**: five probe families + KSG, n=20k, and
  floors calibrated against a fixed random LINEAR injection direction. A nonlinearly-embedded residue
  below the live members' sensitivity remains logically possible; the rising learning curve says the
  true recoverable fraction at larger n is ≥ 0.126.
- Synthetic substrate (the v2 RFF fringe + DeepSets body); MNIST-rotation second leg requires its own
  prereg addendum before running (named, not run).
- Outcome (c) is the pre-registered middle band — neither the (a) counterexample (≥0.30 replicated)
  nor the (b) certified ceiling-at-floor. Both stronger readings were live and neither fired.

## Files
- `scripts/h3_probe_ceiling.py` (battery) + `scripts/test_h3_probe_ceiling.py` (apparatus pins, 7/7).
- `results/atlas/h3/probe_ceiling_result.json` (full readouts, floors, MI, learning curve, seeds).
- `docs/atlas/H3_PROBE_CEILING_PREREG.md` (FROZEN 2026-06-10; reviewer B1–B7 + pilot disclosure).
- Slate: internal hypothesis slate 2026-06-10, HS4 (gitignored internal document).
