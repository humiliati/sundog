# Phase 6.5 — Bifurcation-Set Computation (receipt ledger)

> **The atlas's key structural receipt** (`docs/SUNDOG_V_ATLAS.md` Phase 6.5): compute the bifurcation
> set so the documented transition elevations (22/29/32/46/58) fall OUT of one computation from
> `{refractive index n, crystal-face geometry, sun elevation}` — never hardcoded. This retires the §6
> "armchair catastrophe" gate violation (the live `public/js/parhelion-geometry.mjs` hardcodes
> `czaVisible h<=32` and `TANGENT_ARC_CIRCUMSCRIBED_H=29`). **Two-stage:** 6.5-A (component-B
> admissibility walls) is **LANDED** below; 6.5-B (the 29° component-A cusp) is **STAGED**.
> **NOT public-eligible** — the lit-pass attribution gate (Phase 0.5; Track B, the A₃-cusp identity,
> still open) precedes any outward claim. Scripts: `scripts/atlas_bifurcation_set.py`,
> `scripts/test_atlas_bifurcation_set.py`. Physics fixed by Snell/TIR + Tape; nothing tuned.

## 6.5-A — Component B (ray-admissibility walls) + A₂ fold primitives — LANDED 2026-06-07
Derived from `{n=1.31, wedge geometry}`; pre-registered tolerance `|derived − documented| ≤ 1.0°`.

| feature | kind | component | derived | documented | residual | chromatic band | PASS |
| --- | --- | ---: | ---: | ---: | ---: | --- | :---: |
| 22° halo edge | A₂ fold | A (primitive) | 21.839° | 22° | 0.16° | 21.54–22.37° | ✅ |
| 46° halo edge | A₂ fold | A (primitive) | 45.733° | 46° | 0.27° | 44.88–47.26° | ✅ |
| **CZA disappears** | admissibility wall | **B** | **32.196°** | 32° | 0.20° | 31.02–32.86° | ✅ |
| **CHA appears** | admissibility wall | **B** | **57.804°** | 58° | 0.20° | 57.14–58.98° | ✅ |

**Derivations (the numbers are outputs, not inputs):**
- **A₂ folds (halo edges):** minimum deviation `δ_min = 2·arcsin(n·sin(A/2)) − A` (`∂δ/∂θ = 0`), reusing
  `s2_optics.halo_min_deviation`. The 22°/46° halos are the A₂ fold caustics themselves (not transitions).
- **CZA disappearance (component B):** horizontal plate, light enters the TOP face and exits a SIDE face
  (90° effective wedge). The arc reaches the zenith / the path goes off-sky when `n²−cos²h > 1`, i.e.
  `cos h < √(n²−1)`, i.e. **`h > arccos(√(n²−1)) = 32.196°`** (reusing `cza_formula.py`).
- **CHA appearance (component B):** the SAME 90° plate prism, opposite faces — light enters a SIDE face,
  exits the BOTTOM face. The horizontal↔vertical face swap maps `cos h → sin h`, so the path is
  admissible when **`h > arcsin(√(n²−1)) = 57.804° = 90° − (CZA wall)`** — the exact complement; one
  derivation yields both walls, mutually validating.
- **TIR critical angle** `θ_c = arcsin(1/n) = 49.76°` — the underlying admissibility primitive.

**Discipline checks (in the frozen test):** every wall/fold *recomputes when n changes* (proves
derivation, not a constant); the CHA=90°−CZA complement identity holds to machine precision; the §0.2
chromatic band is reported for each (the transition is a ~1–2° smeared band, not a sharp edge — the
real edge is the smoothed image under the sun's 0.5° disk + tilt-spread + dispersion).

## 6.5-B — Component A (the 29° UTA+LTA→circumscribed cusp) — STAGED, DE-RISKED 2026-06-07
The A₃ cusp does **not** fall out of a scalar `δ(θ)`, but the scoping (web recon) resolved the
dimensionality risk: it is **Walter Tape's published 1980 framework** (*Analytic foundations of halo
theory*, JOSA 70:1175 — the "halo function" `(orientation) → (celestial-sphere point)`; the caustic is
the image of its singular set, located by the **Jacobian kernel**). It is a reimplementation of a
published method, **not** an open-ended 3-D problem. **Minimal sufficient computation (≈1–2 days):**
- **State:** `(γ, α)` ∈ 2-torus — the horizontal column's **two** orientation DOF (γ = c-axis azimuth
  about vertical; α = roll about the c-axis). The "c-axis horizontal" constraint removes the tilt DOF,
  leaving a **square 2-DOF→2-sky-DOF map** (which is why a clean `det J = 0` fold theory exists). Sun
  elevation `h` is the slow control parameter.
- **Forward map:** `sky_dir = F(h; γ, α)` = standard 3-D vector Snell ray trace through the 60° wedge
  (entry refraction → internal → exit refraction), n=1.31. ~20 lines; Stull eq. 22.19 writes it out.
  **No hidden DOF, no full-3-D, no Monte-Carlo** (MC gives brightness/realism only; the cusp is a
  deterministic-caustic property).
- **Caustics:** at each `h`, `{(γ,α): det J = 0}`, `J = ∂(sky)/∂(γ,α)` (finite-difference or analytic) →
  the UTA/LTA **A₂ fold** loci. **Cusp = where the fold locus degenerates** (`∂=∂²=0`) on the
  sun-meridian, where the two arms first touch. Sweep `h`; the elevation of that event = the **derived
  merge** ≈ 29° (pre-registered tol ±1.0°, white-light n).
- **Robustness add-ons (<½ day):** two-wavelength sweep (n_red 1.307 / n_blue 1.317) to quantify the
  λ-shift (this is the documented "29–32°" spread, = dispersion + a definitional choice of "merged");
  cross-check the computed `h_merge` + arc shapes vs an on-disk HaloSim circumscribed-halo render / Tape
  figures.
- **Replaces:** the empirical hardcode in `public/js/parhelion-geometry.mjs` (`TANGENT_ARC_CIRCUMSCRIBED_H
  =29`, single-cell-calibrated `A(h)=A₀(29−18.6)/(29−h)`) — the §6 armchair-gate violation for the cusp.
- **Lit-pass flag (Track B, SYNTHESIS):** Tape frames the merge as "singular points / Jacobian kernel";
  the explicit **"A₃ cusp"** label is the Berry–Upstill/Nye catastrophe classification applied on top —
  correct and standard, but *our* synthesis (no single source says "circumscribed merge = A₃ cusp"
  verbatim). Confirm corank-1 (A₃) vs corank-2 (D₄) numerically. **Honest off-ramp:** if 29° does not
  fall out cleanly, that is a model-insufficiency finding, never a number to tune to.

## Status + downstream
6.5-A makes 32°/58° (and the 22°/46° folds) **derived outputs** — a clean, non-armchair component-B
receipt. 6.5-B (the cusp) is the remaining component-A half. On both landing, Phase 6.5 unblocks Phase 7
(forward sweep — cells between walls), Phase 8 (invisible halos — higher-codim strata), Phase 11
(capstone — the classified bifurcation diagram). Atlas §6 promotion gate ("6.5 lands + 29/32/58 fall
out") is **half-cleared** (B done, A staged).
