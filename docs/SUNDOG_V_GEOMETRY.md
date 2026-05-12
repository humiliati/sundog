# Sundog Geometry Workbench

Working hook:

> It draws the parhelion from the math.

Sundog Geometry is the proposed hero application — the page visitors land on
when they reach `sundog.cc`. Unlike Balance and Three-Body, it is not a
control task with a hidden target and an operating envelope. It is a
parametric exercise of the parhelion description: a workbench that renders
the optical phenomenon by exposing the geometry to its physical knobs and
letting the visitor manipulate them. The workbench is itself the proof — that
the description is concrete enough to render, and the rendering is faithful
enough to recognise as the photographed phenomenon.

The public question is small enough to defend:

> Can the parhelion be reduced to a parametric model that renders a
> recognisable halo display from a small set of physically meaningful
> sliders, while staying interactive in the browser?

The workbench should live as `sundog-workbench.html` until the static pose is
locked, then promote to `index.html` as the hero. A short snapshot of the
locked composition with subtle idle scintillation can serve as the public
brand mark; the full interactive workbench remains accessible from the hero
via a "tune the math" link.

*[nice-to-have #14]*

## Related Documents

This roadmap is the workbench's path-to-promotion plan. It is the
human-tunable face of a broader **Sundog Generator** whose programmatic
face and proof methodology are split into companion docs. Authority is
distributed so no document trampolines another's decisions.

- [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) — the site-wide
  visual frame. Owns: theme tokens (`--sd-*`), shared-stylesheet layer
  organisation, the cross-cutting **4a/4b/4c** split that places this
  workbench beside Threebody and Balance, and Migration Steps 1–6 that
  govern how page-local CSS interacts with the shared sheet. **Step 4a in
  that document defers the detailed roadmap here.** Any new theme token
  this workbench needs goes via `sundog-theme.css` and that document, not
  page-local CSS.
- [`SUNDOG_GENERATOR_SPEC.md`](SUNDOG_GENERATOR_SPEC.md) — the
  programmatic generator architecture (Prompt → Parser → Pose JSON →
  Geometry Solver → SVG → optional AI). Defines the three modes (render
  from math / compare to sky / make beautiful) and the external tool
  schema. **The workbench is Mode 1's user-facing surface; that doc is
  the canonical statement of what the generator is and isn't.** It also
  owns the public-framing language we use when describing the project
  outside this repo.
- [`SUNDOG_OVERLAY_PROTOCOL.md`](SUNDOG_OVERLAY_PROTOCOL.md) — the
  photo-overlay test methodology (Mode 2 procedure). Phase 2 calibration
  items below cite the overlay protocol as their gating procedure.
- [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md) and
  [`SUNDOG_V_BALANCE.md`](SUNDOG_V_BALANCE.md) — sibling application
  workbench roadmaps. Different shape (control task with operating
  envelope and pre-registered verdict), same Sundog pattern of *hidden /
  indirect / transformation / output*. The Pre-Committed Cross-Application
  Comparison Row near the bottom of this document is the parallel.
- `public/poses/*.json` — the named-pose library. `canonical.json`
  snapshots the locked default; future poses (`low-altitude`, `cza-heavy`,
  `nine-halo-eye`) ship as Phase 8 lands.
- BoxForge tools library: `C:\Users\hughe\Dev\Dungeon Gleaner Main\tools`,
  agent-accessible. `boxforge.html` and `orb-component.html` are reference
  primitives for the Phase 6 selective 3D handoff. The animation phase
  vocabulary (idle / hover / active / handoff·settle) used in Phases 4–6
  below comes from that library's three-phase animation system, kept
  consistent so phase semantics carry across projects.

## Sundog Expression *(pre-staged for the eventual APPLICATIONS.md row)*

The Halo Atlas geometry model (landed 2026-05-11) brought Geometry into the
canonical Sundog grammar. The earlier framing — that "geometry does not map
onto the indirect-signal pattern the way Balance and Three-Body do" — was
true of the legacy feature-tuned construction; it is no longer true of the
atlas. The atlas treats every visible arc as the upper portion of a
complete circle anchored by environmental state, which is the
field-and-signature pattern in its most universally legible setting.

- **Hidden state:** sun altitude `h`, ice-crystal orientation distribution,
  atmospheric optical depth. None of these is directly visible to an
  observer or to a photograph.
- **Field:** the set of full implied circles — 22° halo, 46° halo, parhelic
  circle, CZA full ring, supralateral arc, upper tangent arc. Each is a
  primitive `(cx, cy, r)` derived deterministically from the hidden state.
  No primitive depends on observer position or rendering choice; the field
  is a property of atmosphere and geometry alone.
- **Signature:** the visible upper arcs the atlas renders — the part of
  each implied circle that lies above the horizon (or inside the camera
  frame). Discrete features — daggers, X-marks, the eyelid touch point —
  fall out as intersections and tangencies of the field circles.
- **Transformation:** `clip(full_circle, visible_region)` per primitive,
  plus closed-form intersection/tangency formulas for the discrete
  features. The pipeline is one-way: hidden state → field → signature.
- **Actionable output:** *inverse inference.* The same atlas math, run
  backwards, recovers sun altitude from the photographed parhelion offset
  (`h = arccos(R_22 / offset)`). The 2026-05-09 calibration of v2 against
  the Troels Nielsen DR photograph recovered `h ≈ 25°` from visible
  geometry alone, with no metadata. The workbench can therefore be used
  for *measurement* of unobserved environmental variables, not only for
  rendering of given ones.

This block is here so the cross-application row in `APPLICATIONS.md` and the
gallery card's identity triplet do not drift between the roadmap and the
eventual writeup. When Phase 7 lands, both surfaces should quote this block
verbatim.

*[/nice-to-have #14]*

## Current State

First executable scaffold landed 2026-05-08. The Sundog tree now includes:

- `sundog-workbench.html`: a single-page workbench at the root of the public
  site (not yet promoted to `index.html` hero). Stage on the left, control
  rail on the right, mirrors the `threebody.html` idiom.
- 16 parameter sliders organised by optical taxonomy: Sun & Pillar, Halos,
  Arcs, Parhelia (daggers), Composition (the 9-halo-eye fiction), Palette &
  Atmosphere.
- SVG geometry primitives:
  - Compass-rose sun (4 rays, rotated 45° from diagonal so they read as
    cardinal directions; bright core scales with `--compass-ray-length` via
    sqrt-softening).
  - Sun pillar (vertical gradient stroke through the sun core).
  - Parhelic arc (quadratic Bezier path, curvature controlled by
    `--parhelic-curvature`; daggers ride the arc rather than a horizontal
    line — calibrated 2026-05-08 against source-photo overlay at
    `c = 0.66`).
  - 22° halo, 46° halo, two secondary halos for the virtual-sun fiction.
  - Two CZA arcs (primary + secondary) with prismatic gradient stroke and
    Gaussian-blurred edge to approximate ice-crystal orientation
    distribution.
  - Parhelia daggers (red-inside / blue-outside dispersion gradient,
    soft-blurred core ellipse + streak line).
- CSS atmosphere overlay (radial gradients, `mix-blend-mode: screen`)
  parametrised by `--dispersion-width`.
- Snapshot button that writes the current slider state to console + clipboard
  as JSON.

Known calibration debt (2026-05-08 visual review):

- `--parhelic-curvature = 0.66` is locked.
- `--cza-bloom` slider was demoted to a baked constant (`stdDeviation = 0.3`,
  edge-soften only) on 2026-05-08 after A/B testing confirmed Gaussian blur
  on each arc independently does not produce the eyelid bell curve at any
  value — the two arcs do not meet within Gaussian-σ of each other at the
  apex, so their tails never sum to a peaked bell. The bell shape is now
  produced geometrically by a paint-fill band between the two arcs
  (`#cza-bell-fill`). A future `--cza-secondary-offset` knob is the right
  lever for tuning bell width if/when a slider is wanted again; tracked as
  a Phase 9/10 composition-fiction question, not a Phase 2 calibration item.
- The compass-rose 45° rotation is in place but may overlap visually with
  pillar + parhelic, reducing the X reading. Phase 2 tune candidate.
- Sun-altitude slider is wired to a CSS variable but is not yet bound to
  derived geometry — parhelion azimuthal offset and CZA visibility cutoff
  are still hard-coded. Phase 3 work.

No animation has been wired. Idle scintillation amplitude is exposed as a
slider, but only the 22° halo opacity oscillates on it.

This section is here so the broadcast surfaces (gallery card,
`APPLICATIONS.md`, `claims-and-scope.md`) do not get an entry until the
runnable-artifact gates below are met. Until then, the hero workbench sits at
**Scaffold** tier — sibling to the Balance Phase 7 smoke read, not yet a
promoted public artifact.

Promotion to **Calibrated Static Pose** tier requires Phase 0–2 gates below
landing. Promotion to **Animated Hero** tier requires Phase 0–5 gates.
Promotion to **Live Hero** (replace `index.html`) requires Phase 0–7 gates.

## Parallel Geometry Models (Legacy vs Halo Scaffold vs Halo Governed)

We are explicitly avoiding sunk-cost lock-in on a feature-tuned geometry
solver. The workbench therefore supports three geometry models that can be
run in parallel long enough to compare under overlay.

### Legacy model *(feature-tuned)*

This is the current baseline implementation: a small set of visible features
are directly controlled and placed, with a few derived relationships.

- Parhelic arc: symmetric quadratic Bezier driven by `--parhelic-curvature`.
- Parhelia daggers: positioned by sampling that Bezier at fixed `x` anchors.
- Sun pillar: a straight vertical line with a gradient stroke.

This model is fast to iterate on, but it bakes in the exact failure mode
described in the issue: emergent structure gets treated as independent,
manually tuned landmarks.

### Halo Scaffold model *(free-ellipse-derived)*

The first attempt at the issue's proposal. Treats a free ellipse — not a true
halo — as a governing primitive, then derives secondary features from
intersections.

- Governing scaffold: a **free ellipse** whose bottom point is the sun (so
  the tangent at the sun is horizontal). Drawn as a sampled lower arc.
- Parhelia daggers: derived as the intersection points between the 22°
  halo circle and that ellipse. Dagger streak direction is aligned to the
  ellipse tangent at the intersection point.
- Compass-rose north/south behavior: rendered as a vesica between two
  equal-radius circles centered horizontally to either side of the sun.
  Their radii are tied to `--sun-pillar-length`, not to any halo.

This is partway to the issue's intent — daggers do come out of an
intersection — but the ellipse axes (`rx`, `ry`) and the pillar circle radii
are still hand-fit constants, not halo angular radii or sun-altitude
derivations. See *Outstanding gaps* below.

### Halo Governed model *(single-circle scaffold; issue-faithful)*

A more committed reading of the issue: insist that the governing primitive
be a **circle** (a halo) rather than a free ellipse, and tie the pillar
geometry to the same halo system that determines the daggers.

- Governing halo: a **circle** whose bottom point is the sun (one radius
  parameter, currently mapped from `--parhelic-curvature`). The parhelic
  arc is its visible lower-arc.
- Parhelia daggers: closed-form intersection of the governing halo and the
  22° halo. Single-step computation — no iterative root-finding.
- Dagger streak direction: tangent to the 22° halo at the dagger point
  (perpendicular to the radial line from the sun). The daggers lie *on*
  the 22° halo, so this is the physically motivated tangent.
- Compass-rose north/south behavior: vesica of two halos **centered at the
  daggers**, each with radius = (sun-to-dagger distance) + small slack
  controlled by `--sun-pillar-length`. The pillar geometry is therefore a
  consequence of the governing halo via the daggers — it has no independent
  pillar-radius constant.

One knob (`--parhelic-curvature`) determines the parhelic arc, both dagger
positions, and the pillar shape. Everything else is shared (CZA, secondary
halos, compass rays, palette).

This model lives in the workbench at the same code path as the others; it
is intentionally *not* the default while it is unevaluated against the
photo corpus.

### Outstanding gaps from issue #15

The issue raised several specific points. Recording the state of each so
the next reviewer doesn't have to rediscover them:

- **`~160°` halo/arc family** (issue's headline proposal): not directly
  modeled. The Halo Governed model's governing halo serves the same
  structural role (a single primitive that explains arc + daggers) but is
  parameterised by pixel radius, not by an angular value tied to
  atmospheric optics. Naming it "160°" in the workbench would be
  speculative without overlay calibration.
- **Ellipses-instead-of-circles for the 22°/46° halos**: not attempted in
  any of the three models. The 22° and 46° halos remain `<circle>` in the
  SVG.
- **Compass-rose center as a halo intersection**: only partially attempted.
  The Halo Governed pillar uses two real halos (centered at the daggers)
  but the compass-rose ray group itself is still a hand-placed `<g>` of
  four `<line>`s plus a core circle. The rays are not yet derived from
  halo tangencies.
- **Overlay fit evidence**: not produced for any of the three models. Phase
  2 calibration has not yet been run on Halo Scaffold or Halo Governed.
  The acceptance-criteria bullet "evidence that it better explains the
  observed relationships between major features" remains open.
- **Halo Scaffold `rx`/`ry` blends**: still placeholder hand-fit constants.
  The "fewer ad hoc placements" framing is not yet earned for that model.

### How to run in parallel

In `sundog-workbench.html`, use the **Geometry Model** selector:

- `Legacy (feature-tuned)` — baseline behavior.
- `Halo Scaffold (ellipse-derived)` — first-attempt scaffold.
- `Halo Governed (single-circle scaffold)` — issue-faithful construction.

The selection persists in `localStorage` and is included in `Snapshot params`
as `geometryModel`. Reference poses for both alternative tracks live as
`public/poses/canonical-halo-scaffold.json` and
`public/poses/canonical-halo-governed.json`. Note that until Phase 8 lands
the `Load params` button, those poses are documentation artifacts only —
they cannot yet be loaded back into the workbench from the UI.

### Canonical Halo Atlas Vocabulary

Atmospheric optics has its own naming convention for halo-display features.
The atlas adopts those names where they apply, so external readers and our
calibration overlays speak the same language. The two annotated reference
photographs in `docs/calibration/1.Photometeor-jeff-mod_marked_red.jpg` and
`docs/calibration/3.DSC_1029m.jpg` were the source of this list.

Every feature below is a geometric primitive — a circle or an upper-arc
clip of a circle — anchored to environmental state (sun altitude, ice-
crystal orientation type). The atlas mode renders the ones marked **✓**;
**deferred** entries are named here so future calibration passes can adopt
the same terminology when comparing photographs that contain them.

| canonical name | construction | atlas |
| --- | --- | --- |
| **22° halo** | concentric on sun, R = 220 (workbench units) | ✓ |
| **46° halo** | concentric on sun, R = 440 (workbench units, 1.8% photo-fit residual; see R_46 note) | ✓ |
| **Sundog / Parhelion** (left / right) | tangencies on the 22° halo at the parhelic-circle altitude; offset from sun = `R_22 / cos(h)` | ✓ (rendered as the "daggers") |
| **Parhelic Circle** | horizontal great circle through the sun on the celestial sphere; visible portion drawn as the unique circle through the parhelia and the sun, smile direction | ✓ |
| **Circumzenithal Arc (CZA)** | tangent to the 46° halo at its top; small upper arc of a much larger full ring | ✓ |
| **Supralateral Arc** | tangent to the 46° halo at its top, curving *away* from the sun above the tangent | ✓ |
| **Upper Tangent Arc** | tangent to the 22° halo at its top, curving up — column-oriented crystal family | ✓ |
| **Lower Tangent Arc** | tangent to the 22° halo at its *bottom*, curving down — mirror of upper tangent | ✓ |
| **Suncave Parry Arc** | tangent to the 22° halo at its top, concave-toward-sun — Parry-orientation crystal family; visually nested *inside* the broader Upper Tangent envelope | deferred (Parry-family) |
| **Parry Supralateral Arc** | Parry-orientation variant of the supralateral arc | deferred (Parry-family) |
| **Sun pillar** | not a halo arc, but rendered in atlas as the vesica of two virtual halos centered at the parhelia | ✓ (atlas pillar lens) |

Two practical notes:

1. The "Parry-family" deferrals (Suncave Parry, Parry Supralateral) are
   ice-crystal-orientation-mode variants of arcs we already render. They
   sit visually adjacent to or nested inside the atlas's Upper Tangent and
   Supralateral primitives. Adding them is a one-primitive-per-mode
   exercise that we hold until calibration evidence on a Parry-rich photo
   asks for the distinction.
2. The Parhelic Circle is named a *circle*, not an arc, throughout this
   doc and the workbench UI (renamed 2026-05-12). The atlas already
   renders the full-circle geometry; the noun was a leftover from the
   legacy Bezier-feature implementation.

### Halo Atlas v2 fixes (2026-05-11)

After overlay inspection three corrections landed:

1. **Parhelic arc direction flipped from frown to smile.** Was drawing the
   upper arc of a circle whose center sat below the daggers — apex above
   the sun. The Troels Nielsen photograph's parhelic feature smiles (apex
   below sun). Now draws the lower arc of a circle whose center sits
   above the daggers. Applies to both `halo_governed` and `halo_atlas`.
2. **CZA anchored to 46° halo top.** Real atmospheric optics: the CZA is
   tangent to the 46° halo at its top. Apex now sits at `(sun.x, sun.y −
   R_46)` by construction; `--cza-curvature` offsets from that anchor.
   Calibration residual against the Troels Nielsen photo: was (2, 15) px,
   now **(2, 1) px**. Default 0.85 produces the canonical tangent pose.
3. **Upper tangent arc** is a new atlas primitive. Tangent to the **22°**
   halo at its top — the "eyelid above the 22° halo" feature distinct
   from the CZA. Slider `--upper-tangent-intensity` (default 0). Without
   this primitive, the only "eyelid"-shaped feature in the atlas was the
   CZA, which sits much higher (tangent to 46° halo) and therefore can't
   serve double duty.

R_46 note: Gemini suggested using the pure 46°/22° angular ratio,
`R_46 = R_22 × 46/22 = 460` workbench units (→ 303 photo px). Calibration
against the Troels Nielsen photo measures `R_46 = 285 photo px = 432
workbench units`. The atmospheric-optics literature reports the same ~5%
gap (refraction in the 46° prism path is more sensitive to crystal
orientation than the 22° path). The workbench's `R_46 = 440` gives a
1.8% photo-fit residual; switching to 460 would *worsen* the fit to 6.3%.
Holding `R_46 = 440`.

Overlay PNG: [`atlas_v2_overlay_troels_nielsen.png`](atlas_v2_overlay_troels_nielsen.png).

### Multi-Photo Calibration Pass (2026-05-12)

Seven photographs from `docs/calibration/` were calibrated against the
atlas. The seven photos span sun altitudes from ~1° (parhelia at the 22°
halo edge) to ~60° (parhelia well outside the halo), giving the
inverse-inference claim a real range to ride on rather than a single
anecdote. Overlays live in `docs/calibration/overlays/`.

| photo | sun (px) | R₂₂ (px) | parhelion offset | implied `h` | residuals (L, R) | fit |
|---|---|---|---|---|---|---|
| p0 Troels Nielsen DR | (400, 356) | 145 | 160 | 25.0° | (-3, 0) | ✓ |
| p2 Polar | (567, 496) | 182 | 192 | 18.6° | (-1, -1) | ✓ |
| p4 Sunrise sundogs | (511, 413) | 240 | 240 | ~1° | (0, +1) | ✓ low-sun edge case |
| p5 Winter rich | (2083, 2333) | 483 | 521 | 22.0° | (-1, 0) | ✓ |
| p7 Tropical | (1033, 946) | 200 | 393 (L only) | 59.4° | (0 L only) | partial — right parhelion not clearly visible |
| p8 Winter polar | (516, 583) | 287 | 317 | 25.1° | (0, -1) | ✓ after sun-position correction |
| p9 Jasper AB | (673, 226) | 290 | 315 | 23.0° | (-5, +5) | ✓ |

Median absolute residual (dagger placement, both sides, across cleanly-measured
photos): **1 px**. Maximum residual: 5 px on p9. The atlas tracks parhelion
positions across the full sun-altitude range with no per-photo tuning beyond
the three anchor measurements (sun position, R₂₂, parhelion offset).

Two lessons came out of running the pass:

1. **Auto sun-detection picks the brightest pixel, which is not always the
   sun.** In winter ice-fog photos (p8) a parhelion can outshine the actual
   sun. Sanity-check: real parhelia should be approximately symmetric in
   *x-offset* from the sun; if a "sun + parhelion + parhelion" triple is
   asymmetric, the auto-detected sun is probably one of the parhelia and the
   true sun is at the symmetric midpoint between the two outer bright spots.
   This is exactly the correction that fixed p8.
2. **Tropical/high-sun photos can show only one prominent parhelion.** p7
   has a clear left parhelion at offset 393 and a much fainter right side.
   The atlas still calibrates against the visible parhelion alone (the
   formula `h = arccos(R₂₂ / offset)` only needs one), and the symmetric
   prediction for the missing-side parhelion is recorded for if/when more
   photos at high sun altitude become available.

### Theorem Anchor: What the Atlas Demonstrates

The atlas is the cleanest natural example we have of the Sundog
field-not-reward distinction set out in `SUNDOG_V_GRAVITY.md`. The
demonstration runs in two directions:

**Forward (field → signature).** Move the `--sun-altitude` slider. The
daggers slide outward along the parhelic circle at exactly `R_22 / cos(h)`;
the CZA shifts with the 46° halo top; the upper tangent arc rides the 22°
halo top. No visible feature can be moved independently of the environmental
state that generates it. This is the Goodhart sidestep made literal: there is
no scalar quantity an observer can corrupt without altering the atmosphere
the sky belongs to. The signatures are deflected by the field; the field is
a property of `h`, ice-crystal orientation, and optical depth, not of any
viewer's policy.

**Inverse (signature → state).** Run the calibration script against a
photograph: measure the sun pixel position, the 22° halo apparent radius,
and the parhelion offset. The atlas math inverts to recover `h` directly:

  `h = arccos(R_22_obs / parhelion_offset_obs)`

This is the workbench acting as a *measurement instrument* for unobserved
state, parallel to how a photometric controller infers an aim direction it
cannot see directly. The 2026-05-12 multi-photo calibration pass exercised
the inverse over seven photographs spanning sun altitudes from ~1° (low-sun
sunrise sundogs with parhelia at the halo edge) through ~25° (the classic
Troels Nielsen pose) up to ~59° (tropical near-zenith). Median absolute
dagger residual across the cohort: **1 photo px**. No EXIF, astronomical
metadata, or per-photo tuning was used — only the three anchor measurements
per photograph (sun, R₂₂, parhelion offset).

The atlas is therefore not merely the brand mark for the project. It is the
audience-conceptualizable entry point for the gravity claim in the same role
the three-body workbench plays in `SUNDOG_V_GRAVITY.md` §The Three-Body
Wedge — the wedge is rhetorical (lead with a literal field-and-signature
phenomenon, then say "this is what we mean by 'gravity for agents' in every
other partially-observed environment we name"), and the wedge is
methodological (the same hidden/field/signature/inference grammar applies).

This subsection is the load-bearing claim the Phase 7 hero promotion has to
carry. Public framing should anchor here.

### Halo Atlas calibration vs Troels Nielsen DR (2026-05-10)

`halo_atlas` extends the Halo Governed v2 reading with three changes the
photo overlay said were needed:

1. **Sun-altitude binding** — daggers at `(SUN.x ± R_22 / cos(h), SUN.y)`.
   At the canonical pose `--sun-altitude = 25°` (the value implied by the
   photo's parhelion offset), daggers move to offset 240, matching the
   measured photographed parhelia within 0–3 photo px.
2. **CZA as a true full-ring upper arc** — replaces the broken
   Bezier-with-buggy-default `applyCza` for atlas mode. The default
   `--cza-curvature = 0.85` now produces apex y = 81 (canonical), not the
   collapsed flat line at y = 240. Bell-fill is the band between primary
   and secondary upper arcs, both as real circles.
3. **Supralateral arc** — new primitive, tangent to the 46° halo at its
   top, curving up. Optional via `--supralateral-intensity` (default 0).

Numerical residuals at the canonical atlas pose (sun-altitude 25°,
parhelic-curvature 0.05, cza-curvature 0.85, supralateral-intensity 0.40):

| feature | atlas prediction | observed | residual |
| --- | --- | --- | --- |
| 22° halo radius | 145 (anchor) | 145 | 0 |
| 46° halo radius | 290 | 285 | 1.8% |
| left dagger x | 240 | 243 | 3 px (was 12 px in v2) |
| right dagger x | 560 | 560 | 0 px (was 15 px in v2) |
| CZA primary apex | (400, 80) | (402, 65) | (2, 15) px |
| parhelic arc | near-horizontal through daggers and sun | photo's parhelic feature is near-horizontal | ✓ |
| supralateral arc | upper-canvas arc tangent to 46° halo top | photo has visible bright structure in this region | qualitative ✓ |

Overlay PNG: [`atlas_overlay_troels_nielsen.png`](atlas_overlay_troels_nielsen.png).
The daggers and parhelic arc residuals close out the two correctable
defects v2 left open. CZA y-residual remains ~15 px — fixable by tying
`--cza-curvature` default to sun altitude instead of holding it at 0.85.

### Halo Governed v2 calibration vs Troels Nielsen DR (2026-05-09)

A first overlay calibration was run on the canonical Troels Nielsen DR
reference. Anchor: the 22° halo (sun at pixel `(400, 356)` in the 800×450
photo, observed halo radius `145 px`, so `1 workbench unit = 0.659 photo px`).
Overlay PNG: [`v2_overlay_troels_nielsen.png`](v2_overlay_troels_nielsen.png).
Numerical residuals:

| feature | v2 prediction | observed | residual |
| --- | --- | --- | --- |
| 22° halo radius | 145 px (anchor) | 145 px | 0 (anchor) |
| 46° halo radius | 290 px | 285 px | 1.8% over |
| left dagger x | 255 | 243 | 12 px (parhelion **outside** the 22° halo) |
| right dagger x | 545 | 560 | 15 px (same) |
| CZA apex | (400, 79) | (402, 65) | (2, 14) px |
| parhelic arc apex (at default `--parhelic-curvature = 0.66`) | (400, 269) | photo's parhelic circle is essentially horizontal through the parhelia | curvature default too high for this photo |

Two calibration findings worth fixing before the workbench can claim a
photo-faithful pose:

1. **Daggers are 8% too close to the sun.** The photo has parhelia at
   distance `≈ R_22 / cos(h)` from the sun, where `h` is the sun's altitude.
   Solving from the observed offset gives an implied sun altitude of `≈ 25°`.
   v2 currently fixes daggers at exactly the 22° halo radius (the
   sun-on-horizon limit). The fix is the Phase 3 binding from
   `--sun-altitude` to dagger placement: `dagger_x = SUN.x ± R_22 / cos(h)`.
2. **Default parhelic-arc curvature is too high.** At sun altitudes where
   parhelia are well-formed, the parhelic circle reads as nearly horizontal
   through the parhelia in the photograph — there is no significant upward
   arc above the sun in this Troels Nielsen scene. The default
   `--parhelic-curvature = 0.66` should drop closer to 0 for this pose.

What v2 *did* get right: 22° halo (by anchor), 46° halo (1.8%), CZA apex
horizontally (2 px), pillar position (centered on sun). Those four are
load-bearing; the two findings above are correctable knob defaults plus a
Phase 3 sun-altitude binding, not a structural failure of the construction.

### Recommendation

No model should become the default until Phase 2 overlay calibration has
been run on the canonical photo corpus for at least Legacy vs Halo Governed,
with the snapshot of each pose archived for review. Specifically:

- The two-vs-three-model decision rests on overlay fit, not on the elegance
  of the construction in prose. Halo Governed reads cleaner on paper, but
  prose elegance is not evidence.
- Halo Scaffold should not be promoted in its current state — its `rx`/`ry`
  blends and pillar circle radii are placeholder constants that have to be
  resolved before any "fewer ad hoc placements" claim can survive review.
- If Halo Governed reaches comparable or better overlay fit and the open
  gaps above are closed (or explicitly accepted), it is the candidate
  default. Until then it sits beside the other two as a comparison track.
- Whichever model ships, the loser(s) should be retained as a regression
  baseline through at least the Phase 7 hero promotion.

## Why This Workbench

The hero of `sundog.cc` carries two loads at once: it is the brand mark, and
it is the entry point that has to communicate what the project is in under a
second. A static splash can do the brand mark. It cannot do the second load
without lying — most static optical hero images either oversell ("we made
the sky") or undersell ("this is decoration, scroll past it").

A parametric workbench resolves that tension. Visitors can manipulate the
optical knobs and see the rendering respond. The brand mark is visible at
rest. The proof is visible on interaction. The two loads collapse into one
artifact.

The choice is also project-coherent: Balance and Three-Body are both
interactive workbenches with public canvases. Routing the hero through the
same idiom keeps the site reading as a single voice. A visitor who lands on
the geometry workbench, scrolls past, and clicks through to the three-body
or balance pages should feel they are using one tool, not three.

## Visual Direction

The page should feel like an observation, not a marketing splash. The first
screen opens directly into the rendering, with a compact stage copy overlay
in the upper-left.

Target composition (the "sundog eye"):

- foreground: compass-rose sun (pupil), 22° halo (iris), CZA arcs
  (eyelids), parhelia daggers along the parhelic arc;
- background: cool dark sky (`#1a3850` → `#0b1720` → `#050a10` radial),
  evoking a clear high-altitude observation;
- right rail: optical parameter sliders grouped by taxonomy (Sun &
  Pillar / Halos / Arcs / Parhelia / Composition / Palette);
- atmosphere overlay: subtle CSS gradients at warm-amber (sun side) and
  cool-blue (zenith side), `mix-blend-mode: screen`, parametrised by
  `--dispersion-width`;
- optional 9-halo-eye fiction: when `--secondary-suns-strength > 0`, two
  virtual suns appear at the upper-left and upper-right with their own halo
  systems; intended for the active animation phase.

Reference photographs (calibration corpus):

- Troels Nielsen DR — single-sun parhelion with sundog daggers and CZA
  eyelid (the canonical pose).
- Triple-sun event with stacked CZA arcs forming the iris-layer reading
  (the 9-halo-eye source).

Both references are pinned in the hero design memory
(`project_sundog_4a_vision.md`); the workbench's static pose should be
recognisable as a stylised composition of these.

Implementation preference:

- SVG for crisp arcs and rays — no canvas, no WebGL.
- CSS gradients + filters for atmospheric softness; `mix-blend-mode: screen`
  for additive blending on the CZA arc overlap.
- Parameters bind via CSS custom properties wherever the SVG attribute
  accepts them (opacity, stroke-width, transforms).
- JS update loop only for derived geometry (parhelic-arc Bezier, dagger
  positions on the arc, sun-core radius scaling).
- BoxForge primitives reserved for the eventual Phase 6 selective 3D handoff
  (compass-rose ray rotation, possibly), not for the static composition.

## Actionability Audit

The workbench must separate parameter tiers from day one.

| Slider | Tier | Use |
| --- | --- | --- |
| `--sun-altitude` | Math-derived (planned) | Drives parhelion azimuthal offset and CZA visibility cutoff in Phase 3 |
| `--halo-22-intensity` | Free | Visual knob; opacity of the 22° halo ring |
| `--halo-46-intensity` | Free | Visual knob; opacity of the 46° halo ring |
| `--cza-intensity` | Free | Visual knob; opacity of both CZA arcs |
| `--cza-curvature` | Free | Visual knob; apex of the primary CZA arc |
| `--sun-pillar-intensity` | Free | Visual knob; opacity of the vertical pillar |
| `--sun-pillar-length` | Free | Visual knob; pillar extent above and below sun |
| `--parhelic-circle-intensity` | Free | Visual knob; opacity of the parhelic arc |
| `--parhelic-curvature` | Math-derived (planned) | Locked at 0.66 by overlay calibration; Phase 3 will derive from sun altitude |
| `--parhelia-intensity` | Free | Visual knob; opacity of the dagger group |
| `--parhelia-dagger-length` | Free | Visual knob; horizontal extent of the dagger streaks |
| `--dispersion-width` | Free | Visual knob; atmosphere overlay intensity |
| `--secondary-suns-strength` | Composition fiction | Controls the 9-halo-eye effect (not standard atmospheric optics) |
| `--ring-overlap-bias` | Composition fiction | Golden-ratio knob for the 9-ring intersection |
| `--compass-ray-length` | Free | Visual knob; ray length and core radius scale together |
| `--rainbow-saturation` | Free | Visual knob; reserved for Phase 2 prismatic gradient tuning |
| `--idle-scintillation-amplitude` | Animation | Phase-4 idle drift amplitude; off when 0 |

The central typing should be explicit:

```text
M       = parhelion description (the math)
P_t     = parameter state at time t (slider values)
G(P_t)  = derived geometry (positions, curvatures, intensities)
R(G)    = rendered SVG composition
```

Phase 3 binds `M`-derived params (sun-altitude → CZA visibility, parhelion
offset, parhelic curvature) so the workbench cannot render a
physically-impossible pose by accident. Free params remain free knobs; the
fiction params (composition family) are deliberately outside `M` and visibly
labelled in the Composition section of the right rail.

The claim is not that the workbench discovers new optics. The claim is that
the parhelion description is concrete enough to render parametrically, and
the rendering is faithful enough that overlaying it against a reference
photograph reads as the same phenomenon.

## Ratified Hook Language

Safe hook:

> Sundog Geometry renders a parhelion display from a small set of physically
> meaningful sliders. The math draws the eye.

Short version:

> It is the proof, watching you watch it.

Long brand-leaning version:

> Three suns. Nine halos. One description. Move the sliders and watch the
> sky bend to fit the equation.

## Roadmap

### Phase 0 - Parameter Taxonomy and Description Lock

Goal: pin down which sliders are free visual knobs, which are math-derived
from the parhelion description, and which are composition fiction outside
the description. Document the binding equations for the math-derived set so
Phase 3 has unambiguous targets.

Deliverables:

- `docs/SUNDOG_V_GEOMETRY.md` (this document) with the Actionability Audit
  table populated and the math-binding TODO list explicit.
- A short notes file with the parhelion description equations the workbench
  will eventually exercise (sun-altitude → parhelion offset, CZA visibility
  window, parhelic-circle curvature derivation).

Gate: the Actionability Audit table identifies every slider's tier and the
math-binding TODOs are enumerated. **Status: in progress with this
document.**

### Phase 1 - Static Composition Scaffold

Goal: a runnable workbench that renders the canonical "sundog eye" pose at
default slider values, with all parameters interactive.

Deliverables:

- `sundog-workbench.html` with full slider rail and SVG stage.
- All 16 parameters wired to CSS custom properties + JS update loop.
- Reset and Snapshot controls.
- Site chrome (header, footer) shared from `sundog-theme.css`.

Gate: open the page, every slider produces a visible response. **Status:
landed 2026-05-08.**

### Phase 2 - Calibration Pass (Static)

Goal: lock visual defaults against the reference photograph corpus using
the translucent-overlay calibration technique. The procedure is canonical
in [`SUNDOG_OVERLAY_PROTOCOL.md`](SUNDOG_OVERLAY_PROTOCOL.md); this phase
runs that protocol manually for each open calibration item until each
threshold passes against `troels-nielsen-dr.jpg`.

Calibration items (live list):

- [x] `--parhelic-curvature = 0.66` (calibrated 2026-05-08).
- [x] `--cza-bloom` promoted to slider (2026-05-08); default lowered from
  the over-aggressive `3.2` to `1.4` pending overlay calibration.
- [x] `--cza-bloom` demoted to baked constant `stdDeviation = 0.3`
  (2026-05-08, ass-dyno value). Mechanism mismatch confirmed: Gaussian
  blur on each arc independently does not sum to a bell curve at the
  eyelid apex regardless of σ. Replaced with `#cza-bell-fill` paint band
  between the two arcs. Slider count drops 17 → 16. The eyelid-bell goal
  is reclassified from a Phase 2 calibration item to a Phase 9/10
  composition-fiction question whose right lever is `--cza-secondary-offset`
  (geometric overlap), not blur.
- [ ] Compass-rose rotation angle — currently 45° from original diagonal;
  test 30°, 22.5°, and back-out-of-rotate to find the pose where the X
  reads as distinct from pillar + parhelic.
- [ ] Default values for `--halo-22-intensity`, `--halo-46-intensity`,
  `--cza-intensity`, `--parhelia-intensity` against the source photo.
- [ ] Default for `--dispersion-width` (atmosphere bloom level).
- [ ] Default for `--compass-ray-length` (X visibility vs core dominance).
- [ ] Decision: promote 9-halo-eye composition (`--secondary-suns-strength`)
  default — `0.0` (off) for the canonical pose, or a small non-zero
  default if the fiction is part of the locked hero.

Gate: snapshot the locked param JSON to `docs/SUNDOG_GEOMETRY_POSE.json`
or similar; future regressions are reviewable against that file.

### Phase 3 - Math Binding

Goal: bind sun-altitude (and any other math-derived sliders) to the
parhelion description equations so the workbench cannot render a
physically-impossible pose.

Deliverables:

- A small JS module (`public/js/parhelion-geometry.mjs`) implementing the
  binding equations: `parhelionOffset(altitude)`, `czaVisible(altitude)`,
  `parhelicCurvature(altitude)`.
- `applyDerivedGeometry()` updates to call the module instead of using
  hard-coded positions.
- A test harness page or unit-test script that pins the equations to known
  reference altitudes.
- Documentation block in this roadmap referencing the parhelion description
  paper / notes the equations are drawn from.

Gate: setting the sun-altitude slider produces visibly correct
parhelion-offset shift; setting altitude > 32° hides the CZA arcs
automatically.

### Phase 4 - Idle Scintillation

Goal: phase-4 of the BoxForge animation discipline — the idle window where
the static pose holds. Subtle motion only.

Deliverables:

- Slow opacity drift on the rainbow gradient stops (saturation pulse,
  3–8 second period, low amplitude).
- Optional micro-rotation of the compass-rose ray group (≤ 1° peak).
- `prefers-reduced-motion` honored — scintillation snaps to off.
- All scintillation gated by `--idle-scintillation-amplitude`; at 0 the
  pose is fully static.

Gate: load the page at default amplitude, the pose looks alive but no
single element distracts the eye for more than ~2 seconds.

### Phase 5 - Active Reveal Animation (Phases 1-3)

Goal: the sequenced reveal that narrates the parhelion construction. This
is the BoxForge phase 1-3 of the animation model.

Phase 1: empty sky, primary sun fades in (pillar first, then core, then X).
Phase 2: 22° halo draws (stroke-dasharray reveal, 1.5s).
Phase 3: CZA arcs draw + parhelia daggers fade in.

Optional sub-phase: when `--secondary-suns-strength > 0`, virtual peripheral
suns and their halos materialize as a Phase 3.5.

Deliverables:

- Animation orchestrator JS (`public/js/parhelion-phases.mjs`).
- Phase trigger UI (replay button, optional auto-trigger on first visit).
- `prefers-reduced-motion` collapses to instant snap to phase-4 idle pose.

Gate: replay button cycles through phases 1-3 → idle phase-4; reduced-motion
visitors see the locked pose only.

### Phase 6 - Selective 3D Handoff (Phase 5+)

Goal: the BoxForge phase 5+ of the animation model — only elements that
benefit from depth get 3D motion. Halos stay planar (they are sky, not
boxes).

Candidates for 3D motion:

- Compass-rose ray group rotation (slow, hover-triggered or auto).
- Possibly the central core treated as a tiny BoxForge orb on hover only.

Explicit non-candidates (stay planar):

- All halos and arcs (no 3D rotation).
- Parhelia daggers (no depth).

Deliverables:

- BoxForge primitives wired as overlays (orb-component or custom).
- Hover and active-state triggers respecting reduced-motion.
- Performance budget: target ≥ 50 fps on a mid-tier laptop with the active
  reveal + idle scintillation + hover motion all running.

Gate: hover the sun, the rays rotate; halos do not. Reduced-motion visitors
see no rotation.

### Phase 7 - Hero Promotion

Goal: replace `index.html` hero with the workbench (or a distilled
non-interactive snapshot of it with a "tune the math" link to the full
workbench, depending on what reads better in the wild).

Deliverables:

- New `index.html` hero that either embeds `sundog-workbench.html` content
  inline OR renders a distilled static pose with a CTA to the workbench.
- Decision recorded in this document and in `claims-and-scope.md`.
- Old hero canvas (`#parhelion-canvas`) decommissioned.
- A short snapshot `.gif` or `.webm` for press / social use, generated
  from the locked phase-4 idle pose with scintillation.

Gate: a first-time visitor lands on the new hero, gets the brand reading +
the parametric proof signal, and can reach the deeper workbench (if not
already on it) in one click.

### Phase 8 - Reproducible Pose Export

Goal: the workbench's `Snapshot params` button currently writes JSON to
console + clipboard. Promote that to a deterministic export pipeline
matching the schema in
[`SUNDOG_GENERATOR_SPEC.md`](SUNDOG_GENERATOR_SPEC.md): identical pose
JSON loaded back into the workbench produces a byte-equivalent SVG.

Deliverables:

- `Load params` button accepting JSON (paste or file picker), reading
  the canonical schema from the Generator Spec.
- Pose pinning script that, given a JSON file, renders the workbench at
  that pose and exports a PNG/SVG snapshot for archival.
- Named-pose library expansion under `public/poses/`:
  - [x] `canonical.json` — snapshot of the current locked defaults
    (landed alongside the Generator Spec).
  - [ ] `low-altitude.json` — sun near the horizon, parhelia close to
    halo; CZA visible.
  - [ ] `cza-heavy.json` — sun altitude well below 32°, CZA at maximum
    intensity, halo and parhelia subdued.
  - [ ] `nine-halo-eye.json` — full composition fiction with
    `--secondary-suns-strength = 0.7` and tuned overlap bias.
  - [ ] `forty-six-halo.json` — 46° halo dominant, 22° subdued.

Gate: round-trip a snapshot → load → render → snapshot and confirm
byte-equal JSON across two browser sessions.

## Post-Verdict / Conditional Roadmap

These phases unlock only after Phase 7 promotion lands and the hero is in
the wild for some adoption period.

### Phase 9 - Educational Mode

Goal: a toggle that overlays annotations on the workbench — angle measures,
crystal-orientation arrows, refraction-angle labels — turning the brand
hero into a teaching artifact.

### Phase 10 - Atmospheric Variations

Goal: alternative sky palettes (dawn, dusk, twilight, polar) with calibrated
photograph references for each. Optional `--sky-tint` becomes a real palette
slider rather than a single-axis warm/cool knob.

### Phase 11 - Linked Description Mode

Goal: clicking any slider opens a small popover with the equation it
represents, the source citation, and a tiny worked example. Couples the
workbench tightly to the eventual parhelion-description paper.

## Claim Boundary

The Geometry workbench claims:

- A parametric rendering of known parhelion physics that is faithful enough
  to overlay against reference photographs and read as the same
  phenomenon.
- The "sundog eye" composition (pupil = sun, iris = halos, eyelids = CZA
  arcs, optional 9-halo virtual-sun extension) is an aesthetic
  identification, deliberately stylised, with no claim that real
  atmospheric optics produces a 9-halo eye on demand.
- Interactive proof that the parhelion description is concrete: every
  slider corresponds to either a math-derived geometric quantity, a free
  visual knob clearly labelled as such, or a composition-fiction knob
  clearly labelled as such.

The Geometry workbench does NOT claim:

- Novel atmospheric optics findings. The physics is well-known; the
  contribution is the parametric-rendering format and the
  composition-as-brand framing.
- That the parhelion is "discovered" by the workbench. The phenomenon has
  been observed and described for centuries; Sundog's contribution is the
  description style and the workbench-as-proof presentation.
- That the workbench renders all halo phenomena. It renders the parhelion +
  CZA + parhelic-arc subset chosen for the hero composition. Other halo
  phenomena (Parry arcs, infralateral arcs, sub-suns, light pillars
  specific to plate-crystal alignments) are out of scope unless added in
  Phase 10.

## Pre-Committed Cross-Application Comparison Row

For the eventual `APPLICATIONS.md` table:

| Application | Hidden | Indirect | Transformation | Output |
| --- | --- | --- | --- | --- |
| Geometry | parhelion description | sky display | parametric optical render | interactive workbench + reproducible pose |
| Three-Body | true 18D state | virial / pairwise / inertia signatures | SCAN/SEEK/TRACK | bounded controller for selected regimes |
| Balance | upright pole angle | shadow projection on floor | shadow-residual control | bounded cart force command |

Geometry occupies a different cell shape than the other two — it is a
description-and-render pair rather than a hidden-state-and-controller pair —
and the table headers should accommodate that. The "Hidden" column for
Geometry reads as the description-being-exercised rather than a withheld
truth.

## Suggested First Build Slice

The first implementation pass (largely already complete) was deliberately
small:

1. SVG stage with the canonical pose hard-coded.
2. Slider rail with all 16 parameters wired to CSS custom properties.
3. Derived-geometry JS for the parhelic-arc Bezier and dagger positioning.
4. Snapshot and reset controls.

That slice is enough to decide whether the workbench-as-hero idiom carries
the brand load. The locked-pose snapshot from Phase 2 calibration becomes
the canonical reference for everything downstream.

*[nice-to-have #15]*

**Exit criterion for the first slice:** the canonical pose, loaded from a
snapshot JSON, reads as the photographed phenomenon when overlaid at 50%
opacity against the Troels Nielsen DR reference photograph (vertical pillar
matches; 22° halo radius matches; parhelia positions match within ±10
SVG units; CZA apex within the eyelid envelope). Below this floor — the
overlay reads as a different phenomenon — re-scope before any Phase 4
animation work.

*[/nice-to-have #15]*
