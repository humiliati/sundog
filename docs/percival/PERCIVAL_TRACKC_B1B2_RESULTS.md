# Percival Track-C B1/B2 -- real-system bridge (results)

Pre-reg: [`PERCIVAL_TRACKC_B1B2_PREREG.md`](PERCIVAL_TRACKC_B1B2_PREREG.md) (frozen; claim register: CALIBRATION of the classical law -- McNemar/Connor; Miller 2024; Kotawala 2026 -- along a training trajectory + the dial/floor/T-contrast design contributions).

## Verdict: **B1B2_REFUTED**

Floor (self-pair, fp32 fixed-batch): d_s = 0.000000, d_b = 0.000000.
Gates: {"B2a_floor": true, "B1a_calibration": true, "B1b_monotone": false, "B1c_t_arm": false, "B2b_crispness": false}

| pair | d_b | d_s | delta | R_meas | R_formula | rel_err | stab_p | stab_u |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hellaswag:main-142000 | 0.0455 | 0.0210 | -0.005 | 0.0519 | 0.0528 | 0.016 | 0.7305 | 0.565 |
| hellaswag:main-140000 | 0.0485 | 0.0210 | -0.003 | 0.0529 | 0.0529 | 0.0005 | 0.6225 | 0.518 |
| hellaswag:main-130000 | 0.044 | 0.0195 | -0.0035 | 0.0479 | 0.0491 | 0.0247 | 0.6485 | 0.546 |
| hellaswag:main-110000 | 0.0505 | 0.0245 | -0.0055 | 0.0655 | 0.0615 | 0.0639 | 0.723 | 0.565 |
| hellaswag:main-70000 | 0.067 | 0.0365 | -0.0035 | 0.0976 | 0.092 | 0.0607 | 0.621 | 0.531 |
| hellaswag:main-30000 | 0.087 | 0.0435 | -0.0045 | 0.1179 | 0.1095 | 0.0768 | 0.6375 | 0.544 |
| hellaswag:dedup-143000 | 0.087 | 0.0420 | -0.002 | 0.1002 | 0.106 | 0.0553 | 0.563 | 0.524 |
| lambada:main-142000 | -- | 0.0880 | 0.028 | 0.1922 | 0.1942 | 0.0106 | 0.983 | 0.812 |
| lambada:main-140000 | -- | 0.1000 | -0.004 | 0.2175 | 0.2178 | 0.0015 | 0.5775 | 0.5575 |
| lambada:main-130000 | -- | 0.0980 | -0.002 | 0.2225 | 0.2137 | 0.041 | 0.521 | 0.5065 |
| lambada:main-110000 | -- | 0.0910 | -0.029 | 0.1918 | 0.1937 | 0.0099 | 0.9835 | 0.831 |
| lambada:main-70000 | -- | 0.1350 | -0.011 | 0.2877 | 0.2926 | 0.0167 | 0.737 | 0.616 |
| lambada:main-30000 | -- | 0.1550 | 0.053 | 0.3558 | 0.3461 | 0.028 | 0.998 | 0.9665 |
| lambada:dedup-143000 | -- | 0.1370 | -0.007 | 0.3052 | 0.2978 | 0.0248 | 0.63 | 0.5895 |

T-arm (n=300): R(T=0)=0.2131 vs R(T=0.7)=0.0969; agreement-item margin variance 0.0 (T=0) vs 0.020997 (T=0.7).

## Honest boundary

Calibration of a classical law on one substrate family (Pythia-160M) and one primary task; replication-grade statistics, design-grade novelty (checkpoint dial, floor audit, T-contrast). No deception-detection claims; the Track-C link is interpretive. Misses are findings; no post-hoc gate edits.

## Post-hoc reading (labeled; gates untouched — the token stands per the frozen tree)

The verdict fires on B1-b per the frozen adjudication, but the components split sharply, and the
split is the result:

**What lived.**
- **B1-a, the central law: CONFIRMED, tightly.** `R = (d_s − Δ²)/(p_Aq_A + p_Bq_B)` calibrated on
  every pair, both task families — HellaSwag rel. error 0.0005–0.077 (7/7 pairs ≤ 0.077), LAMBADA
  0.0015–0.041. The pairing law the toys derived and the Lean anchors pinned transported to real
  weights at ~3% mean error. This was the actual bridge claim.
- **B2-a, the floor: EXACT ZERO.** Two independent eval passes of the same checkpoint, fp32 + fixed
  batch: `d_s = d_b = 0.000000`. The h=0 zero-variance branch (v3's T1) is real on a physical GPU —
  the harness contributes literally nothing under these controls.
- **Sign-stability direction: uniform.** stab_paired > stab_unpaired on every pair, both tasks
  (typically +8–17 pp). The B2-b thresholds missed (below), but the direction never once inverted.

**What died, and what the deaths say.**
- **B1-b, the monotone dial: REFUTED — behavioral distance is not monotone in training-step distance
  at fine scale.** On BOTH tasks, step-130000 is behaviorally CLOSER to step-143000 than step-140000
  is (HellaSwag d_s 0.0195 vs 0.0210; LAMBADA has its own dip at 110k). Checkpoint churn wiggles;
  disagreement accumulates monotonically only at coarse scale (110k→70k→30k rises cleanly). The dial
  design assumed drift accumulates; it does not, and that is a real observation about training
  dynamics, replicated across two tasks.
- **B1-c: my ratio prediction was ANTI-confirmed** — R(T=0.7)=0.097 < R(T=0)=0.213, opposite the
  registered ≥3× direction. The structural half held exactly as registered (agreement-item margin
  variance 0 at T=0, 0.021 > 0 at T=0.7 — sampling does put unpairable noise on agreement items),
  but the aggregate ratio moved the other way because pairing still cancels item-difficulty variance
  at T>0, and k=4 graded scores make the paired comparison MORE efficient, not less. Miller's
  components decomposition (item variance ↦ pairing; generation variance ↦ resampling; orthogonal
  remedies) is the right frame; our "structurally unpaired" aggregate framing overstated. Conceded.
- **B2-b: thresholds missed for an identifiable reason** — every ladder Δ on HellaSwag is tiny
  (|Δ| ≤ 0.0055), so even paired signs are unstable (there is nearly nothing to be stable ABOUT);
  LAMBADA's 30k rung reached stab_p = 0.998 but its unpaired twin was also stable (0.9665 ≥ 0.90).
  The gate needed a Goldilocks Δ this ladder does not contain.

**Net:** the pairing law and the zero-floor survive contact with reality; the dial and the T-ratio
predictions die informatively. As a Track-C bridge: the toy's noise model is RIGHT about pairing and
cancellation, WRONG (in our extrapolation, not the theorems) about how disagreement accumulates over
training and about sampling's aggregate effect. The theorems' own scope (per-round, fixed D) was
never touched.

