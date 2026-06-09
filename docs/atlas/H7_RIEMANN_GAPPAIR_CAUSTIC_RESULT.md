# H7 Result — the Riemann gap-pair caustic (empirical reconnaissance)

> **2026-06-09.** Against `docs/atlas/H7_RIEMANN_GAPPAIR_CAUSTIC_PREREG.md`. Pointed the lab's validated
> caustic/spectral machinery at the Riemann zeros (the registered Odlyzko `zeros1.txt`, 100k zeros). NOT
> public-eligible; an **empirical probe, NOT an RH claim**, and the rigorous Riemann lane stays frozen.
> Attribution: Montgomery (pair correlation); Odlyzko (the zeros + the finite-height numerics); Berry &
> Keating (the zeros as a spectrum; spectral form factor; arithmetic corrections).

## Headline — NULL-A (height-resolved), as pre-registered

**The Riemann zeros are GUE-universal, and the probe read the spectrum correctly.** The "gap-pair caustic"
is the universal τ=1 ramp→plateau kink of the spectral form factor. Two apparent "prizes" both **dissolved
under scrutiny** into artifacts/known structure — and the discipline that caught them is the real
deliverable.

## The arc (and the four self-caught bugs)

A first pass flagged a **16σ "deviation from GUE"** and a form-factor "arithmetic" signal. Both were
investigated to destruction:

1. **Form-factor "deviation" → the integer picket-fence (known rigidity).** The unfolded zeros satisfy
   `w_n ≈ n` (the count *is* n), so at integer τ the form factor resonates: `e^{2πi w_n τ} → e^{−2πi S_n}`,
   spiking `K(1)` to ~15–25 vs the GUE plateau ~1. That is the zeros' *known spectral rigidity*, not
   arithmetic; the plateau "deviation" was this spike's reach + a noise-band that counted only the CUE
   error, not the zeros' own fluctuation. The **ramp itself matches GUE** (`K(0.5)`: zeros 0.53, CUE 0.51)
   — the τ=1 caustic is real and universal.
2. **The "16σ" anti-correlation → finite-height arithmetic, converging to GUE.** Static consecutive-gap
   correlation: zeros −0.356, GUE/CUE −0.307. Three artifact hypotheses were tested and **rejected**:
   a *concatenation bug* in the control (block-boundary outliers washed CUE to −0.001 — fixed → −0.307);
   a *finite-size* hypothesis (GUE plateaus at −0.31 across L=500→2000, does not climb to −0.357 — rejected);
   an *unfolding-procedure* mismatch (the zeros give −0.356 under **both** asymptotic ⟨N⟩ and local-poly
   unfolds — rejected). The static excess survived — because it is **real and height-dependent.**

## The resolution — height-dependence + 1/log γ extrapolation

The zeros' consecutive-gap anti-correlation is **height-dependent**: more rigid at low height, weakening
with height (the signature of the finite-height arithmetic correction). Direction rules out an unfolding
artifact (a bad unfold biases *toward* less anti-correlation; ⟨N⟩ is least accurate at low height — yet low
height is *more* anti-correlated).

| γ range | ≈5,000 | … | ≈70,000 |
|---|---|---|---|
| consecutive-gap corr | −0.370 | (monotone) | −0.353 |

Fitting `corr(γ) = C∞ + A/log γ` (the expected arithmetic-correction form) over 20 height bins:

&nbsp;&nbsp;&nbsp;&nbsp; **A = −0.585,  C∞ (γ→∞) = −0.300  ≈  the GUE control −0.307.**

So the static excess is the **known ~1/log γ finite-height arithmetic correction**; extrapolating to
infinite height **recovers GUE**. The low Odlyzko zeros are caught mid-convergence — the probe detected the
convergence and (correctly) did not mistake it for a deviation.

*(The fourth self-caught bug: when porting the height analysis into the main script, a helper that diffs
its input was fed the gaps instead of the levels — computing the correlation of second differences (always
≈−0.6) and printing a nonsensical "C∞=−0.603". The standalone test (correlating gaps directly) had it right;
re-running the banked script rather than trusting the port surfaced it. The number above is post-fix.)*

## Pre-registered scorecard
| Gate | Result |
|---|---|
| P-F1 the τ=1 caustic (GUE ramp→plateau kink) | **PASS** (ramp slope +1.03, plateau flat) |
| zeros are GUE-universal | **PASS** — anti-corr height-extrapolates to GUE (C∞=−0.300 ≈ −0.307) |
| P-F2 arithmetic deviation beyond τ=1 above the floor | **null** (the flag was the picket-fence + noise-band artifact) |
| KILL (no power / all controls identical) | not triggered (CUE −0.307, Poisson 0.00 — clean separation) |
| **Overall** | **NULL-A**, as pre-registered: zeros = GUE + the known finite-height correction |

Power curve (N=10k→100k) computed; the τ=1 caustic and the height trend are stable across N.

## What this is and is NOT
- **IS:** the first time the lab's catastrophe-optics machinery was pointed at the Riemann spectrum, and it
  read GUE universality correctly + detected/extrapolated the finite-height arithmetic correction. A clean,
  bounded, honest reconnaissance.
- **IS NOT:** a new discovery (the finite-height arithmetic approach to GUE is documented — Odlyzko,
  Bogomolny–Keating, Berry); a deviation from GUE (it converges to GUE); an RH claim of any kind.

## Honest boundaries
- Low-height zeros only (γ ≤ 74,920, the first 100k). The arithmetic correction is *largest* here, which is
  why it was visible; it shrinks with height.
- The **jet classifier on the gap-pair density was uninformative** (it reads ρ's curvature, not an optical
  caustic; the 133 "cusps" are KDE noise). The literal "gap-pair density caustic" (Face 1's original image)
  is, as pre-registered, a **smooth blob — no literal caustic.** The caustic lives in the form factor (τ=1).
- The form factor here is a raw periodogram; a clean arithmetic-beyond-τ=1 probe needs the **connected**
  form factor / Montgomery's F(α) with the explicit-formula prime terms — a follow-up, not this run.

## Files
- `scripts/riemann_gappair.py` — the probe (unfold, form factor, controls, height-resolution).
- `results/atlas/h7/formfactor.npz` — the form-factor power curve.
- `docs/atlas/H7_RIEMANN_GAPPAIR_CAUSTIC_PREREG.md` — the locked pre-registration.
