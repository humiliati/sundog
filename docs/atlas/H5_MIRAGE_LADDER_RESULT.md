# H5 Result — the mirage (Δn, s) ladder as a classified caustic / cusp diagram

> **Run against the LOCKED pre-registration** (`H5_MIRAGE_LADDER_PREREG.md`). Hypothesis #5 of slate
> `ww6koomb1`. NOT public-eligible. Attribution: Lehn; Können; Bruton (mirage ray theory); Berry; Nye,
> *Natural Focusing and the Structure of Light* (catastrophe optics); the Atlas jet classifier (H4-validated).

## Headline verdict — BOUNDED-POSITIVE

**Mirages form a classified caustic ladder in the (Δn, s) refraction control plane — the mirage analog of
the halo Atlas — and the superior-mirage 1→3-image onset IS a Whitney A₃ cusp, labelled by the same jet
classifier validated in H4.** A double inversion reaches the 5-image Fata-Morgana rung (partial on the
catastrophe label).

## Stage 1 — the ray-transfer forward model (`scripts/mirage_ladder.py`)

Paraxial flat-Earth ray ODE `dh/dx=ψ, dψ/dx=n'(h)` (rays curve toward higher n), inversion profile
`n(h)=n0 − (Δn/2)tanh((h−h0)/s)` (R=20 km, observer 10 m, inversion 120 m). Back-trace from elevation θ →
`h_target(θ)`; #images = #solutions of `h_target(θ)=H`; image-merge caustics are the folds `dh_target/dθ=0`.
Validated (all PASS):
- **broad gradient → monotonic transfer → 1 image** (no mirage);
- **localized inversion → S-shaped transfer → a 3-image band** (superior mirage); the band *widens* with
  Δn (2.4 → 39 → 112 m) — the cusp's `√(Δn−Δn*)` opening;
- **correct sign**: rays curve **down** toward the dense air below (−116 m bend) = superior mirage.
- (Robust fold-counting on the longest above-ground segment kills the ducting/ground-truncation edge
  artifact.)

## Stage 2 — the (Δn, s) ladder + jet-classifier labels (`scripts/mirage_ladder_sweep.py`)

### P2 — the image-multiplicity ladder (PASS)
```
IMAGE COUNT (single inversion)   1 = simple refraction,  3 = superior mirage
 s\Δn   1e-5  2e-5  3e-5  5e-5  1e-4  2e-4
   3      1     1     3     3   (2)   (2)      (2) = ducting regime
   5      1     1     3     3     3   (2)
   8      1     1     1     3     3   (2)
  20      1     1     1     1     3   (2)
  40      1     1     1     1     1     3
```
A clean **1 → 3** ladder; the **1→3 onset follows a Δn/s ≈ const curve** (the cusp boundary): thinner/
stronger layers onset at lower Δn (onset(s=3)=3e-5 < onset(s=20)=8e-5). The `(2)` cells are the **ducting**
regime (very strong inversions trap/ground-truncate rays — a distinct phenomenon, outside the simple-fold
ladder).

### P3 — the 1→3 onset is a Whitney A₃ cusp (PASS, the headline)
Chart `F(θ, Δn) = (Δn, h_target(θ; Δn, s))`, so `det DF = −∂h_target/∂θ` (caustic = image-merge) and the
cusp `φ=0 ∧ c2=0` is the 1→3 onset, `c3 = −∂³h/∂θ³` bounded ⟹ A₃ (the exact H4 Stage-1 structure, from
real ray physics). Non-dimensionalized chart, jet classifier:

| | caustic | #cusps | \|c3\| | corank |
|---|---|---|---|---|
| single inversion (s=5) | **True** | **1** | **426** (bounded → A₃, not A₄) | **1** (not D₄) |

The superior-mirage onset is a genuine **A₃ cusp**, labelled by the H4-validated tool.

### P4 — the 5-image Fata-Morgana rung (PARTIAL)
A **double** inversion (two stacked layers) reaches **5 images** (4 folds): the births are at Δn≈2e-5
(1→3) and Δn≈3e-5 (3→5), then the folds annihilate (4→3→2) into ducting. The chart shows a caustic with a
cusp (|c3|=844 bounded, corank-1), but the isolated-cusp detector **resolves only 1 of the 2 closely-spaced
births** — an honest limitation of the detector on the tightly-packed Fata-Morgana caustic. The 5-image
rung itself is confirmed by direct fold-counting; the *multi*-cusp catastrophe label is owed.

## Pre-registered scorecard
| Gate | Result |
|---|---|
| P1 model validity (1 image vs S-shaped 3-image; sign) | **PASS** |
| P2 the (Δn,s) 1→3 ladder + Δn/s~const cusp boundary | **PASS** |
| P3 the 1→3 onset reads as an A₃ cusp (corank-1, \|c3\| bounded) | **PASS** |
| P4 Fata-Morgana 5-image rung | **PARTIAL** (5 images confirmed; multi-cusp label incomplete) |
| kill criteria (monotonic-always / fold-only / no-ladder) | none triggered |

## Honest boundaries
- Paraxial **flat-Earth** model (no Earth curvature); 1-D horizontally-stratified atmosphere. Physically
  the mirage-onset Δn (~few×10⁻⁵ across a sharp layer) is in the right order for a strong surface
  inversion. A curved-Earth / wave-optics extension is owed for quantitative sky comparison.
- The **ducting** regime (strong inversion, ground-truncated rays) is excluded from the clean fold ladder
  — a distinct phenomenon, flagged not modelled.
- P4's multi-cusp Fata-Morgana label is partial (detector resolution); the 5-image count is solid.
- Forward-only; no inversion. The catastrophe labels are *computed* (jet classifier), not asserted.

## Files
- `scripts/mirage_ladder.py` — Stage 1 ray-transfer model + phenomenology.
- `scripts/mirage_ladder_sweep.py` — Stage 2 ladder + jet-classifier labels.
- `scripts/test_mirage_ladder.py` — frozen test (6/6 PASS).
- `docs/atlas/H5_MIRAGE_LADDER_PREREG.md` — the locked pre-registration.
