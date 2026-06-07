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

## 6.5-B — Component A (the 29° UTA+LTA→circumscribed cusp) — LANDED 2026-06-07
**Derived merge elevation = 29.7° (white-light n=1.31; chromatic band 29.5–30.3° across the visible,
= the documented "29–32°" spread), within ±1.0° of the documented ~29°.** The 29° transition is now a
DERIVED output of the horizontal-column halo-function caustic — replacing the hardcoded
`TANGENT_ARC_CIRCUMSCRIBED_H=29` in `parhelion-geometry.mjs`. **The §6 armchair-catastrophe gate is now
FULLY cleared (22/29/32/46/58 all derived).** Sanity: the caustic touches 21.84° at the top (the 22°
fold); the caustic is two separate arcs below the merge and a connected circumscribed loop above (the
A₃ topology change). Scripts: `scripts/atlas_caustic_map.py` + `scripts/test_atlas_caustic_map.py`
(all pass). **Caustic locator (implemented):** `delta_min(ψ)` — the per-azimuth minimum deviation, which
is the fold-caustic envelope; the **merge = where its mid-ψ GAP closes** (the UTA/LTA wing-tips touch).
The gap is real inadmissibility (robust to the `dmax` cap: identical at dmax 46/70/89° — the column
produces no rays beyond ~48°), not an artifact.

**Method** (the de-risk-confirmed minimal sufficient model — **Walter Tape's 1980 framework**,
*Analytic foundations of halo theory*, JOSA 70:1175 — caustic = the image of the halo-function's
singular set, the **Jacobian kernel**; 2 orientation DOF suffice, **no full-3-D, no Monte-Carlo**):
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
**Phase 6.5 COMPLETE (2026-06-07): both components landed.** Component B (CZA 32.196°, CHA 57.804°) and
component A (the 29.7° A₃ cusp), plus the 22°/46° A₂ fold primitives, are all **derived outputs** of one
computation from `{n, crystal-face geometry, sun elevation}` — the documented 22/29/32/46/58 fall out,
none hardcoded. The atlas §6 promotion gate ("6.5 lands + 29/32/58 fall out vs. the apparatus") is
**cleared on the computation side**; the remaining piece is the optional HaloSim apparatus cross-render
(the on-disk circumscribed/CZA/CHA frames) and the Phase-0.5 lit-pass (Track B, the A₃-cusp identity is
SYNTHESIS) before any outward claim. Phase 6.5 now unblocks **Phase 7** (forward sweep — the walls
partition `(elevation × habit)` into cells), **Phase 8** (invisible halos — higher-codim strata of the
same caustic map: swallowtail A₄ / umbilic D₄), and **Phase 11** (capstone — the classified bifurcation
diagram). **NOT public-eligible.**
