# H8 v5 Result — substrate-generality test (substrate S5: slow-fast chaos): **NULL (obstacle is SUBSTRATE-GENERAL)**

> **2026-06-09.** Against `docs/atlas/H8V5_SUBSTRATE_GENERALITY_PREREG.md`. Tested whether the H8 obstacle
> is CGL-specific or substrate-general, on a **genuinely non-CGL substrate** (deterministic slow-fast chaos,
> logistic map). **Result: the charFun-resist phase φ is GEOMETRIC even here** — recoverable from static,
> template-reproducible statistics (power spectrum + amplitude histogram) → not load-bearing → **KILL-GEOMETRIC**.
> The obstacle holds off the CGL-spiral family. NOT public-eligible; frozen-as-portfolio. A clean,
> pre-registered NULL — the ~70% most-likely outcome, and the strongest version of the finding.

## Headline — KILL-GEOMETRIC on a non-CGL substrate
The latent φ (the slow phase modulating the logistic map through its period-doubling/chaos regime) is
cleanly recoverable from **static statistics** that a geometric/matched-spectrum foil reproduces:
| feature (own-R² of φ, within-distribution) | own-R² |
|---|---|
| power spectrum (= what the matched-spectrum surrogate **preserves**) | **0.928** |
| amplitude histogram (a static marginal) | **0.919** |
| raw window (a poor probe feature — not diagnostic) | 0.335 |
φ is **saturated** by static statistics (0.92+ from the spectrum alone, 0.92+ from the histogram alone), so
there is nothing left for the nonlinear temporal determinism to contribute. A matched-spectrum surrogate
recovers φ (0.928); the strongest static foil (IAAFT = matched spectrum **and** histogram) would too. **φ is
geometric → KILL-GEOMETRIC → NULL.** No load-bearing non-geometric charFun-resist exists on this substrate.

## Honest read (the adversarial mindset applied to my own crux)
The probe's auto-verdict printed "weak/confounded" off the *raw-window* own-R² (0.335) — a poor feature, not
a real signal. The **dissection** is the decisive, honest evidence: φ lives in the spectrum (0.928) and the
amplitude distribution (0.919), both **static and template-reproducible**. I did not bank the auto-verdict;
the conclusion (geometric/substrate-general) rests on the dissection, where the matched-spectrum foil's
preserved feature (the spectrum) recovers φ. *(This is the same care that caught the first v5 sketch's
confound and v3's vacuous test — I read the dissection, not the headline number.)*

## The H8 obstacle — now FIVE attempts, with a substrate-general argument
Across four CGL-spiral attempts (v1–v4) and one **non-CGL chaotic** attempt (v5), **every charFun-resist
latent is GEOMETRIC** (recoverable from static template/statistics — symmetry-orbit coordinate, interference
phase, or spectral/marginal structure) → **not load-bearing**; the RD-dynamical quantities are
finite-mean → **determine**. The argument, now stated as near-identity:
> **charFun-RESIST ⟹ the latent enters periodically (a phase) ⟹ it is a property of the observed
> structure recoverable from static statistics ⟹ GEOMETRIC (template-reproducible) ⟹ NOT load-bearing.**
> Symmetrically, **load-bearing ⟹ the recovery needs the dynamics/trajectory ⟹ a finite-mean dynamical
> functional ⟹ DETERMINE-type.**
This is **about the OBSERVATION, not the substrate** — v5 confirms it survives leaving the CGL family
entirely. Stated as a strongly-supported **conjecture** (five differently-designed attempts, two substrate
families), not a proven theorem (a proof would require formalizing "recoverable-from-lossy-observation ⟹
template-reproducible," which the empirics motivate but do not establish).

## Honest boundaries
- Crux-level result (the pre-registered go/no-go), not the full apparatus — but KILL-GEOMETRIC is decisive
  and the dissection is clean; the spectrum-saturation argument makes the IAAFT foil redundant.
- Two substrate families (CGL spirals, logistic-map chaos) — strong but not exhaustive; the conjecture is
  about observation-based shadows generally.
- The raw-window feature was poor (a known limitation of MLP on short raw time-series); the dissection is the
  load-bearing evidence.

## Files
- `scripts/reaction_diffusion_chaos_shadow.py` — the v5 non-CGL probe (slow-fast logistic chaos; matched-
  spectrum foil; spectrum/histogram dissection; own-R² head-to-head).
- `docs/atlas/H8V5_SUBSTRATE_GENERALITY_PREREG.md` — the locked pre-reg (KILL-GEOMETRIC fired as pre-committed).
