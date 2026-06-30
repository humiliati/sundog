# U-6 — Atlas surrogate: a physical halo curve as a constructed short-program

**Result: physical WITNESS LANDED + honest size verdict; `ATLAS_BRANCH_NOT_CLEAN` did not fire.**
Script: [`scripts/algo_approx_u6_atlas_surrogate.py`](../../scripts/algo_approx_u6_atlas_surrogate.py).
Deterministic, CPU, numpy only — everything is **constructed**, nothing trained.

## The question

The capstone `continuous_relu_approximable` (`Sundogcert/UniversalApprox.lean`) proves universal
approximation for *abstract* continuous `f`. U-6 grounds it once: take **one** real continuous curve
from the portfolio and exhibit the **explicit constructed ReLU short-program** approximant — not a
trained net, the lane's own machine-checked construction — with its measured error.

## The physical curve (grounded in the repo's own `h-of-x` page)

The `h-of-x` math page promotes the relation `offset = R₂₂ / cos(h)`: the parhelion (sundog) angular
offset from the sun as a function of solar altitude `h`, with the 22° halo as the ruler. Over the
page's own altitude-slider domain `h ∈ [0°, 60°]`, normalize altitude to `x = h/60 ∈ [0,1]` and the
offset to its range `[22°, 44°]`:

```
offset(h) = R₂₂ / cos(h),  R₂₂ = 22°,  h ∈ [0°, 60°]
g(x)      = sec(π·x/3) − 1     on [0,1],   g(0)=0,  g(1)=1
```

`g` is C∞, single-valued, monotone on `[0,1]`; its nearest singularity (the `sec` pole at `h = 90°`,
i.e. `x = 1.5`) is **outside** the domain. So it is a fair `C([0,1])` target — and crucially **not a
polynomial in disguise**: the pole limits the polynomial convergence rate, so the degree needed for a
target accuracy is a real measurement, not zero. It passes `ATLAS_BRANCH_NOT_CLEAN`:

```
finite=True   monotone=True   nearest_pole_x=1.50 (outside [0,1])=True   →  branch CLEAN
```

## 1+2 — FIND a polynomial, CONSTRUCT the net, measure both

For degree `d = 1..14`: the near-minimax polynomial `p_d` (Chebyshev-node interpolant of `g`) is the
*find*; the **constructed ReLU net** is built from `p_d` by the lane's machinery — sawtooth-squaring
`R_m` (`SawtoothShared`/`SawtoothDag`), polarization multiply `a·b = 2·R_m(s) − R_m(a)/2 − R_m(b)/2`
with `s=(a+b)/2`, and `clamp01` (`MonomialEval`), in the monomial basis (`PolyEval`). Sawtooth depth
`m = 16` for this faithfulness column.

| d | L∞(g, p_d) — FIND | L∞(g, net) — CONSTRUCT | \|net − poly\| |
|--:|------------------:|-----------------------:|---------------:|
| 1 | 3.646e-1 | 3.646e-1 | 0 |
| 2 | 8.583e-2 | 8.583e-2 | 7.9e-11 |
| 4 | 6.444e-3 | 6.444e-3 | 9.4e-11 |
| 6 | 4.523e-4 | 4.523e-4 | 4.7e-12 |
| 8 | 3.372e-5 | 3.372e-5 | 3.5e-10 |
| 10 | 2.425e-6 | 2.425e-6 | 6.8e-10 |
| 12 | 1.726e-7 | 1.758e-7 | 3.2e-9 |
| 14 | 1.251e-8 | 1.987e-8 | 7.4e-9 |

- **The witness.** The constructed net's measured L∞ **tracks** the near-minimax polynomial across the
  whole range (`|net − poly|` stays `≤ ~1e-9` until the sawtooth floor at `d≈13`) — the lane's
  sawtooth + polarization + clamp construction **faithfully realizes** the polynomial on a real
  physical curve. This is the worked example the capstone lacked.
- **A free analytic cross-check.** The polynomial fit converges **geometrically** at observed
  **3.726×/degree** vs the predicted `ρ = 2 + √3 = 3.732` (the Bernstein-ellipse rate for a pole at
  `x = 1.5`). A clean independent confirmation that the curve and the construction behave exactly as
  the analysis says — so `g` has a `poly(log 1/ε)`-gate certified short program.

## 4 — SIZE vs a baseline (the honest part)

Certified deep-construction gate count (lane bounds: `≤54m` gates/square, `d(d−1)/2` multiplies in the
monomial basis) vs a **naive shallow linear spline** (depth-1, `L∞ ≤ max|g″|/(8N²)`, `~2N` gates), at
matched total accuracy:

| ε | deep (d, m) | deep gates | spline gates | smaller |
|--:|:--:|--:|--:|:--:|
| 1e-2 | (5, 6) | 9,776 | 28 | spline |
| 1e-4 | (8, 10) | 45,509 | 278 | spline |
| 1e-6 | (12, 14) | 150,031 | 2,768 | spline |
| 1e-8 | (14, 17) | 251,084 | 27,666 | spline |
| 1e-10 | (14, 21) | 310,052 | 276,652 | spline |
| 1e-12 | (14, 24) | 354,278 | 2,766,502 | **deep** |

**Honest verdict.** For this smooth 1-D curve the naive shallow spline is the *smaller* net until an
extreme crossover `ε* ~ 1e-12`. The deep construction's `O(deg·log 1/ε)` advantage is **real** (the
deep gate count grows poly-logarithmically while the spline grows as `1/√ε`, so deep *must* win
eventually — and does, at `1e-12`), but it is **constant-heavy**: `54m` gates per square and `O(d²)`
squares in the monomial basis. This is the expected, correct behavior — it does **not** contradict the
depth-separation result for `x²` (where the deep sawtooth beats shallow at moderate ε); it locates
where the *full-pipeline whole-curve* construction pays back its constants. A foregone "construction
wins" would have been the dishonest answer.

## Honest scope (and the falsifier)

- **A single physical witness**, not a claim about halos or about approximation theory. The curve is
  one clean branch; the construction and its bounds are the lane's already-proved cores.
- **`ATLAS_BRANCH_NOT_CLEAN` did not fire** — the parhelion-offset branch is cleanly continuous /
  single-valued / pole-free on `[0,1]`, so no surgery was needed (it is a fair `C([0,1])` target).
- **The size result is honest, not promotional.** The deep construction is not the smaller net for a
  smooth 1-D curve at practical accuracy; the witness is the faithful *realization*, and the analytic
  cross-check (3.726 vs 3.732) is the strongest standalone confirmation.
- **Reopen-gated lane.** U-6 touches the Atlas lane (frozen-as-portfolio); reopened here by owner
  decision for this single worked witness. No Atlas claim is made or revived.

## Provenance

Slate-4 U-6. Curve grounded in the repo's `h-of-x.html` (`offset = R₂₂/cos(h)`). Construction =
`Sundogcert/SawtoothShared.lean` + `SawtoothDag.lean` + `MonomialEval.lean` + `PolyEval.lean`; cost
bounds from U-3 (`ApproxCost.lean`). Cross-links:
[`ALGO_APPROX_CONJECTURE_SLATE_4.md`](ALGO_APPROX_CONJECTURE_SLATE_4.md),
[`ALGO_APPROX_U5_REPRESENTATION_GAP_RESULT.md`](ALGO_APPROX_U5_REPRESENTATION_GAP_RESULT.md).
