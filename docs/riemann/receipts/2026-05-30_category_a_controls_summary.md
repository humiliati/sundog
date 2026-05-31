# Riemann Category-A Controls — Q1/Q2/Q3 resolved in-house

## Header

- Receipt id: `2026-05-30_category_a_controls_summary`
- Purpose: execute the **ask-for-review matrix** decision for the Riemann lane —
  run every Category-A control (the "why did we hit a wall" questions that are
  controls / robustness / identities) *before* emailing anyone, so the external
  ask shrinks to its one irreducible Category-B sentence.
- Resolves: all three External Review Packet "What We Are Asking" questions.
- Author / runner: Claude (Opus 4.8)
- Artifacts:
  - Q2 proof: [`2026-05-30_Q2_parity_identity_proof.md`](2026-05-30_Q2_parity_identity_proof.md)
  - Q1 control: `scripts/riemann-q1-sinekernel-control.py` →
    `results/riemann/probe05-q1-sinekernel-control/q1_sinekernel_summary.json`
  - Q3 sweep: `scripts/riemann-q3-window-sweep.py` →
    `results/riemann/probe05-q3-window-sweep/q3_window_sweep_summary.json`
- Scope guard: none of these touch or rewrite the registered Probe 01 / Probe 05
  artifacts; they are additive controls. Not structural-zero probes; not evidence
  for or against RH.

## Result

| Packet question | Category | In-house control | Verdict |
| --- | --- | --- | --- |
| **Q1** GUE/sine-kernel reproducibility | A | synthetic CUE + i.i.d. Wigner spectra through the identical D statistic | **CONFIRMS standard null** |
| **Q2** explicit-formula / reflection parity | A (provable) | desk proof | **PROVED identity** |
| **Q3** window / unfolding call | A | re-run to 100k ordinates + alt unfoldings | **bounded null persists & deepens** |

## Q1 — sine-kernel / GUE control

Claim under test: that `D = -0.0064` inside floor `0.0424` is just the standard
GUE / sine-kernel reversibility null (the packet's pre-registered
`R-NL-NEG-A` disposition), not a structural-zero edge.

Method: the **identical** S2 statistic (`D = mean sign(s_i − s_{i+1})`, tie tol
`1e-8`, floor `tau_ind = 3/√m`, same block bootstrap) run on
(1) **CUE eigenphases** — the exact bulk sine-kernel process (Montgomery–Odlyzko
universality), Haar unitary via Mezzadri (2007), unfolded by the theoretical
density `n/2π`; and (2) **i.i.d. GUE Wigner-surmise** spacings — a
correlation-free contrast. `E[D] = 0` holds exactly for both (CUE by conjugation
symmetry `θ ↦ −θ`; i.i.d. by exchangeability), so the only question is whether
the *magnitude* `|D_zeta|` is typical.

| Quantity | Observed zeta | CUE sine-kernel (180×) | i.i.d. Wigner (3000×) |
| --- | --- | --- | --- |
| mean `D` | −0.006403 | +0.000221 (≈ 0) | +0.000100 (≈ 0) |
| std `D` | — | 0.007768 | 0.008235 |
| central 95% of `D` | — | [−0.01571, +0.01531] | [−0.01641, +0.01561] |
| **z-score of zeta `D`** | — | **−0.85** | **−0.79** |
| **percentile of zeta `D`** | — | **19th** | **22nd** |
| frac. of runs with \|D\| ≤ `tau_ind` (0.0424) | — | **100%** | **100%** |
| bootstrap floor `tau_boot` | 0.02041 | 0.02345 (spot-check) | — |

Reading: the observed zeta `D` is a **completely typical draw** from the exact
sine-kernel distribution (0.85σ below its zero mean, well inside the central
95%). Every genuine sine-kernel realization falls within the registered floor,
and the probe's own block-bootstrap floor recomputed on a real sine-kernel
sequence (0.0234) matches the zeta value (0.0204). The correlation-free i.i.d.
model gives the same picture — so the null carries **no pair-correlation
information, let alone structural-zero information.**

**Verdict: CONTROL CONFIRMS STANDARD NULL (R-NL-NEG-A), in-house.** The observed
`D` is indistinguishable from the GUE / sine-kernel baseline.

## Q2 — parity / reflection identity

Proved (see linked receipt): both the explicit-formula odd row (`C1-O`) and the
Probe 01 Z2 reflection residual are **algebraic identities** — an odd function
summed over an origin-symmetric `±γ` multiset is zero; a reflection-invariant
`(gap, |center|)` feature has zero reflection residual. Independent of RH and of
arithmetic. Upgrades `R-C1-NEG-A` from *expected* to *proved*.

## Q3 — window / unfolding sweep

Self-check reproduced the canonical N=5000 result bit-for-bit
(`nPlus=2483, nMinus=2515, D=-0.006403`; source SHA matched). Then:

| N (ordinates) | D | floor `tau_ind` | bounded |
| --- | --- | --- | --- |
| 5,000 (registered) | −0.00640 | 0.04243 | ✓ |
| 10,000 | −0.00060 | 0.03000 | ✓ |
| 20,000 | −0.00630 | 0.02121 | ✓ |
| 50,000 | −0.00504 | 0.01342 | ✓ |
| 100,000 | −0.00100 | 0.00949 | ✓ |

Alt unfoldings at N=100,000: RvM −0.00100 / raw −0.00098 / smoothed −0.00056 —
all bounded, all agree to ~0.001. `|D|` stays inside the shrinking floor at every
window and **deepens toward 0** as N grows — the signature of a genuine
zero-mean reversibility process, not a window-specific tuning artifact. The sign
statistic is also unfolding-insensitive.

**Verdict: bounded null persists across 20× more ordinates and three unfoldings.**

## Collapsed ask — what is left for an external reviewer

All three "What We Are Asking" questions are now answered in-house, and — beyond
mere classification — the controls **affirmatively establish** the packet's
"substrate-level cause, not a tuning artifact" claim: the Probe 05 null *is* the
GUE / sine-kernel reversibility null (a universal substrate feature), with `D` a
finite-window fluctuation strictly inside the floor.

The genuinely irreducible Category-B residue is therefore **narrower** than the
packet's current three-question ask. It is the **representation-sector** judgment
(the packet's "If The Reviewer Has Only One Comment"):

> Is there any legitimate finite-group / representation-sector reading in these
> three Riemann substrates that we are prematurely dismissing — or is the
> "no structural-zero edge here" conclusion the right conservative call?

This is the only part the controls cannot self-grant: they show the
*gap / parity / reversibility* features are standard or vacuous, but not that
some *other* sector (e.g. the paused Path (ii) S3-triple bridge) is absent. It
connects directly to the open S3-triple escalation in `SUNDOG_V_RIEMANN.md`.

Note on the "substrate-level vs finite-window" wording: we can **unilaterally**
adopt the more precise "this is the GUE/sine-kernel reversibility null, a
universal finite-window feature" framing with no sign-off — being *more*
conservative never needs review. Only keeping a stronger structural reading
would.

## Recommended packet edits (not yet applied)

1. Replace "What We Are Asking" Q1–Q3 with a one-line "resolved in-house"
   pointer to this receipt + the three artifacts.
2. Reduce the ask to the single representation-sector sentence above.
3. Sharpen the synthesis wording from "substrate-level cause" to the precise
   "GUE/sine-kernel reversibility null (universal) + identity/linearity parity
   nulls," attaching the Q1 control numbers and the Q2 proof.
4. Keep the identity-zero laundering guard (`R-C1-NEG-B`) intact.
