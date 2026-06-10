# Phase 7 — The Classified (elevation × habit) Halo Phase Diagram (receipt ledger)

> **The atlas as a phase diagram.** Phase 7 assembles the derived bifurcation set into the complete
> classified halo phase diagram: the **(sun-elevation × crystal-habit)** control space, the elevation axis
> (0–90°) **partitioned into CELLS** (display regimes) for each habit, every cell's feature catalog
> classified by the mechanism of its bounding transition. **21 partition cells across 6 habits.**
> Built by the 30-agent-class workflow `wsktxxxg9` (6 habit enumerators → adversarial verify → synthesis)
> and **independently reproduced in code** (`scripts/atlas_forward_sweep.py` + `test_atlas_forward_sweep.py`).
> **NOT public-eligible** (frozen-as-portfolio per the 2026-06-04 pivot). n_ice = 1.31 visible centroid.

## Discipline (what "derived" means here)
- **DERIVED** transitions were **computed by running the repo machinery** (`s2_optics.halo_min_deviation`,
  `cza_formula`, `atlas_caustic_map.sky_grid`/`merge_elevation`/`is_merged`, `atlas_strata_map.cusp_field`),
  never asserted — the §6 "armchair catastrophe" gate. The number + the exact call are recorded per wall.
- **CATALOG-NOT-DERIVED** transitions (no forward model — plate reflection features, all Parry-specific
  onsets, the anthelic multi-reflection family, the unexplained-halo trio) use the documented elevation (or
  none) and are **flagged as gaps**, never passed off as derived.
- Each transition is **component-A** (caustic catastrophe: A₂ fold / A₃ cusp / A₃-class merge-metamorphosis)
  or **component-B** (ray-admissibility wall: TIR / grazing / off-sky). The random rings are a **third kind**:
  horizon-clip **occlusion** of an always-present A₂ fold — a viewing-geometry effect, **NOT a bifurcation**.
- **§0.2:** every boundary is a ~1–2° **smeared band** (chromatic + 0.5° sun-disk + tilt), not a sharp edge.

## The master bifurcation set (8 derived phase-boundary elevations) — code-reproduced
`python scripts/atlas_forward_sweep.py` recomputes every row below from the repo machinery.

| Elevation | Boundary | Comp. | Derivation (reproduced) |
| ---: | --- | :---: | --- |
| **29.71°** | column UTA+LTA → circumscribed merge | **A** | `merge_elevation()` = 29.71; `is_merged` False@29.5, True@29.7 (A₃-class metamorphosis) |
| **32.20°** | plate CZA off / column+rosette supralateral off (90°-wedge) | **B** | `arccos(√(n²−1))` = 32.196; `basal90` admissibility bisection = 32.19 (mutually validating) |
| **57.80°** | plate CHA appears | **B** | `arcsin(√(n²−1))` = 57.804 = 90 − CZA wall (exact complement; CZA+CHA = 90.000) |
| **60.74°** | plate parhelia (sun dogs) disappear | **B** | plate forward model (`plate_parhelion_present`, 60° vertical-face wedge) bisection = 60.74 |
| **16.10°** | **Lowitz A₃-lips cusp-pair birth** | **A** | `cusp_field(lowitz60)` deep-interior count 2→4 (the one higher catastrophe in the whole sweep) |
| **~63–65°** | Lowitz family off-sky | **B** | `sky_grid(lowitz60)` admissibility → 0 (63.1 by admissible-fraction; ~65 by ray count — threshold-dependent, both in band) |
| ~31° | Lowitz interior cusp pair reabsorbed at wing-tip wall (4→2) | **B** | `cusp_field` interior inventory (catalog/soft) |
| *(none)* | ~~Wegener arc off-sky~~ | — | **forward-model gap** — `wedge='wegener'` is a prism60-*subset* artifact (mask byte-identical to prism60 at high h); no reliable wall (the synthesis's "52.41°" is not reproduced and is withdrawn as soft) |

Plus the **elevation-independent A₂ fold-ring radii** (all reproduced via `halo_min_deviation`): 22° halo
**21.839°** (60° wedge), 46° halo **45.733°** (90°), and the pyramidal odd radii **8.953/18.272/19.905/
23.818/34.888°** (9/18/20/24/35°). Their only elevation dependence is the horizon-clip occlusion `h ≥ R`.

## Honesty ledger — what was DERIVED, CATALOG, and CORRECTED
- **DERIVED + reproduced** (this pass, independent of the workflow): all 8 walls above + every fold radius.
- **CATALOG-NOT-DERIVED (honest gaps):** plate reflection/multi-path features (parhelic circle, sun pillar,
  120° parhelia, 44/46° parhelia); **all Parry-specific onsets** (`sky_grid('parry')` raises `ValueError` —
  no Parry model); the **anthelion / anthelic-arcs / subhelic / Tricker / diffuse / 120°-parhelia** family
  (P0 catalogue, no elevation cells); **Moilanen / elliptical / Bottlinger** (out of scope — open edges).
- **TWO transitions REFUTED AND REMOVED** (the adversarial verify caught them; reproduced on this pass):
  1. Anthelic **"Wegener reaches anthelion at 22.13°" — FABRICATED.** Re-run: min ray-to-antisolar angle is
     **flat ~44°** at all elevations, never approaches 0. → the two cells it bounded collapse to one 0–52.4°.
  2. Composite **"oriented-pyramidal 23.8° lower-branch appears at 20°" — FABRICATED.** Re-run: the
     caustic min-deviation is **flat 23.82°** at all h; no discrete event. → the 0–20/20–29.7 boundary collapses.
- **THREE catalog corrections:** plate "44/46° parhelia" is **absent** from `HALO_PHENOMENA_ACCOUNTING.md`
  (→ a catalog gap, not "documented"); plate "120° parhelia" is a **rear-sky antisolar** feature
  (column+Parry), not a front-sky plate feature; the AH-CH10 p6 table has the **60°↔62° wedge / 21.8↔22.9°
  radius rows transposed** (confirmed: 60°→21.839, 62°→22.862).

---

## The phase diagram (6 habit columns over 0–90° elevation)

### HABIT 1 — Randomly-oriented crystals (background ring halos)
Random = full SO(3) at every elevation → the minimum-deviation (A₂ fold) ray is ALWAYS generated. No
orientation-driven admissibility wall, no caustic metamorphosis. Each ring is a fixed-radius A₂ fold; the
only elevation dependence is the geometric horizon-clip of the lower arc (occlusion, not a bifurcation).

| Elevation cell | Features present | Upper boundary (mechanism) |
| --- | --- | --- |
| 0 – 8.95° | 22/46° + 9/18/20/24/35° rings (upper arcs; lower arcs clipped) | 8.95° = 9° ring clears horizon [B-occlusion, R=`halo_min_deviation(28.0)`] |
| 8.95 – 21.84° | 9° ring full; 18/20° full once h>18.27/19.90; larger rings clipped | 21.84° = 22° ring clears horizon [B-occlusion] |
| 21.84 – 34.89° | 9/18/20/22/24° rings full; 35/46° clipped (24° clears at 23.82) | 34.89° = 35° ring clears horizon [B-occlusion] |
| 34.89 – 45.73° | 9/18/20/22/24/35° rings full; 46° lower arc clipped | 45.73° = 46° ring clears horizon [B-occlusion] |
| 45.73 – 90° | all rings full; upper rims of larger rings wrap past zenith (cosmetic) | 90° = zenith |

### HABIT 2 — Singly-oriented PLATE (c-axis vertical, 1 roll DOF) — folds only
1 orientation DOF → A₂ folds only (no A₃ cusps). The 29.7° column merge does NOT apply.

| Elevation cell | Features present | Upper boundary (mechanism) |
| --- | --- | --- |
| 0 – 32.20° | parhelia (on/near the 22° halo, splitting outward as h rises); **CZA**; parhelic circle; sun pillar | 32.20° = **CZA disappears** [B, `arccos(√(n²−1))`] |
| 32.20 – 57.80° | parhelia (now well outside the 22° halo, fading); parhelic circle; sun pillar. CZA gone, CHA not yet | 57.80° = **CHA appears** [B, exact CZA complement] |
| 57.80 – 60.74° | parhelia (near limit, very smeared); **CHA** (low band, brightening); parhelic circle | 60.74° = **parhelia disappear** [B, plate forward model] |
| 60.74 – 90° | CHA (brightest ~58–68°); parhelic circle; sun pillar. Parhelia + CZA absent | 90° = zenith |

### HABIT 3 — Singly-oriented horizontal COLUMN (c-axis horizontal, 2 DOF) — the richest, fully modeled
2 DOF → A₂ folds + A₃ apex cusps + the 29.7° A₃-class merge.

| Elevation cell | Features present | Upper boundary (mechanism) |
| --- | --- | --- |
| 0 – 29.71° | upper + lower **tangent arcs** (separate UTA+LTA, 22° fold edge, A₃ point-cusps at apexes); **supralateral/infralateral** arcs (46° family) | 29.71° = **UTA+LTA merge** [A, A₃-class metamorphosis, `merge_elevation()`] |
| 29.71 – 32.20° | **circumscribed halo** (merged loop); supralateral/infralateral (still admissible) | 32.20° = supralateral off-sky [B, `basal90` bisection] |
| 32.20 – 90° | circumscribed halo (persists; `prism60` admissibility rises to h=70, no derived upper wall) | 90° = zenith |

### HABIT 4 — PARRY (1-DOF, folds only) + LOWITZ (2-DOF) — the one higher catastrophe lives here
Parry = 1-DOF (no `parry` wedge — onsets catalog-not-derived except the shared 60°/90° primitives).
Lowitz = the `lowitz60` wedge → the **A₃-lips**.

| Elevation cell | Features present | Upper boundary (mechanism) |
| --- | --- | --- |
| 0 – ~16.1° | Lowitz arcs (22° fold floor); 2 persistent parhelion-side apex A₃ cusps; sunvex Parry [CATALOG]; heliac arc [CATALOG] | ~16.1° = **Lowitz A₃-lips** (interior cusp pair born 2→4) [A, `cusp_field` bisection] |
| ~16.1 – ~31° | Lowitz arcs with the A₃-lips interior cusp pair (4 interior cusps); inner edge climbing above 22° (smooth A₂-fold drift, not a catastrophe); suncave Parry onset [CATALOG] | ~31° = interior cusp pair reabsorbed at the wing-tip wall (4→2) [B]; also ~32.2° CZA-class wall bounds Parry supralateral [B] |
| ~31 – ~63–65° | Lowitz arcs (back to 2 apex cusps), shrinking admissibility; suncave Parry [CATALOG]; Parry CHA-type onset 57.8° [B primitive] | ~63–65° = whole Lowitz family off-sky [B, `sky_grid(lowitz60)` → 0] |

### HABIT 5 — ANTHELIC (antisolar) region — multi-population [cells collapsed by the verify pass]
The only single-crystal-derivable generator is the column Wegener path — and that wedge is a prism60-subset
artifact, so even its numbers are soft. **The "Wegener reaches anthelion at 22.13°" transition is REFUTED**
(min ray-to-antisolar angle flat ~44° at all h).

| Elevation cell | Features present | Upper boundary (mechanism) |
| --- | --- | --- |
| 0 – ~52° | Wegener anthelic arc (rear sky, standing ~44° OFF the anthelion — does NOT thread it; A₂-fold caustic, corank-1, no D₄); anthelion, anthelic arcs, subhelic, Tricker, diffuse arcs, 120° parhelia [all CATALOG] | ~52° = Wegener rear-sky rays vanish [B, **soft/model-gap** — see master set] |
| ~52 – 90° | anthelion + anthelic arcs (non-Wegener routes); subhelic/Tricker/diffuse; 120° parhelia fade [all CATALOG] | 90° = zenith |

The anthelic X-crossing = **two transverse A₂ folds from distinct populations** (Phase-8: NOT a single-map D₄).

### HABIT 6 — COMPOSITE + RARE [pyramidal lower-branch boundary collapsed by the verify pass]
Bullet rosettes ≈ random columns + weak oriented-column arcs (inferred, no rosette wedge). Oriented
pyramidal columns = the `pyrcol` wedge. **The "23.8° lower-branch appears at 20°" transition is REFUTED**
(min-dev flat 23.82° at all h). Moilanen/elliptical/Bottlinger = genuine **open edges** (out of scope).

| Elevation cell | Features present | Upper boundary (mechanism) |
| --- | --- | --- |
| 0 – 29.71° | 22/46° rings (rosettes); UTA+LTA tangent arcs (oriented rosette columns); **23.8° + 9° oriented-pyramidal arcs** (at their folds throughout — NO discrete h=20 event); supralateral/infralateral; Wegener arc; CZA (if oriented-plate members, h<32.2) | 29.71° = UTA+LTA → circumscribed merge [A, `merge_elevation()`] |
| 29.71 – 32° | as above + **circumscribed halo**; supralateral fading | ~32° = supralateral off-sky + CZA disappears [B, 32.196] |
| 32 – 57.80° | 22/46° rings; circumscribed; 23.8°+9° pyramidal arcs (robust); infralateral (broadening) | 57.80° = CHA appears [B, complement] |
| 57.80 – 90° | 22/46° rings; circumscribed; 23.8° pyramidal arc (apex tilts 23.8→25.5° toward zenith); 9° arc; infralateral; CHA (oriented-plate members) | 90° = zenith |

---

## Catalog cross-check (Gate-2)
The cells match the documented displays (`HALO_PHENOMENA_ACCOUNTING.md` + Greenler/Tape/Können/Cowley); **no
uncataloged feature is presented as real-and-derived.** Clean matches: 22/46° halos, odd-radius 9/18/20/24/35°,
parhelia, CZA (Tape AH-CH06 p63 "for elevations >~32° the internal reflection becomes total, the CZA
disappears" = 32.196), CHA (p65, h>~58 = 57.804), tangent arcs, circumscribed halo (p62 "at 29° the two
halos merge … 29° is theoretical" = 29.711), supralateral/infralateral (p60 "forms only h<~32, like the
CZA"), Lowitz arcs, Parry suncave/sunvex, Wegener/Hastings, the antisolar P0 block. The 3 catalog
corrections (above) are the only discrepancies; all flagged, none fabricated-and-passed-as-real.

## Forward-model gaps (the honest edges)
> **Update 2026-06-09 (Phase 12) — several of these are now CLOSED by the polarized halo raytracer**
> (`scripts/s2_halo_raytracer.py`; `ATLAS_PHASE12_POLARIZATION_PHASEDIAGRAM.md`). The raytracer **is** the
> random/plate/column/Parry/pyramidal forward generator these cells lacked, it **reproduces the
> random-ring horizon-clip occlusion** (22° ring below-horizon flux 36%→0% across e=21.84°), and it adds
> a **polarization layer** to every feature (refraction folds → radial `DoP(R)`; TIR features →
> antisymmetric ±V, net→0). The orientation-driven boundary *elevations* (60.74/29.71/32.20/57.80°)
> still belong to the caustic-map/admissibility machinery below (the raytracer generates the features
> but does not re-derive those exact walls by flux-windowing).
- No **random-orientation** sky_grid model (ring cells from `halo_min_deviation` + horizon geometry).
  *(Phase 12: CLOSED — raytracer reproduces the horizon-clip occlusion.)*
- No **plate** model beyond the parhelion (reflection/multi-path features have no derived wall).
  *(Phase 12: PARTIAL — raytracer generates the parhelic circle / 120° parhelia / sundogs + their
  polarization, but the 60.74° parhelia-off elevation stays a Phase-7 derived B-wall.)*
- `wedge='wegener'` is a **prism60-subset artifact** — not a faithful side→basal-TIR→side anthelic model;
  yields no reliable column anthelic window (a faithful model changes the EXIT face after a real basal bounce).
- No **Parry** wedge (`ValueError`); Parry-specific onsets are catalog-not-derived.
  *(Phase 12: PARTIAL — raytracer has a Parry habit generating the above-sun meridian arc; the specific
  onset elevations remain catalog.)*
- No **bullet-rosette** wedge (feature set inferred from random+oriented column generators).
- **Moilanen / elliptical / Bottlinger** — no forward model, no catalog row, mechanism unresolved.

## Status
Phase 7 is the structural payoff of the atlas: the **forward-generated classified phase diagram** — 6 habits,
21 cells, 8 derived phase-boundary elevations (all code-reproduced) + the fold-ring radii, every transition
classified component-A / component-B / occlusion, every catalog-not-derived gap flagged. It reuses the
Phase-8 catastrophe classifications (A₂ folds, A₃ cusps, the 29.7° A₃-class merge, the Lowitz A₃-lips; Berry
confirmed — **no A₄ swallowtail, no D₄ umbilic** anywhere in the 2-DOF sweep). The remaining work is the
forward-model gaps (a faithful Wegener model, a random/plate/Parry generator) — none of which changes the
classified skeleton. **NOT public-eligible** (Phase 0.5 lit-pass + attribution gate any outward claim).
