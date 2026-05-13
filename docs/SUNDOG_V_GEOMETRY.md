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
- [`ICON_ASSETS.md`](ICON_ASSETS.md) — the current favicon/app-icon
  source list and regeneration notes. Phase 11 below expands that into a
  design-team toolkit for a characterized Sundog logo and animation set.
- [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md) and
  [`SUNDOG_V_BALANCE.md`](SUNDOG_V_BALANCE.md) — sibling application
  workbench roadmaps. Different shape (control task with operating
  envelope and pre-registered verdict), same Sundog pattern of *hidden /
  indirect / transformation / output*. The Pre-Committed Cross-Application
  Comparison Row near the bottom of this document is the parallel.
- `public/poses/*.json` — the named-pose library. `canonical.json`
  snapshots the locked default; future poses (`low-altitude`, `cza-heavy`,
  `nine-halo-eye`, `rich-display-vocabulary`) ship as Phase 8 and Phase
  10 land.
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
- Parameter sliders organised by optical taxonomy: Sun & Pillar, Halos,
  Arcs, Parhelia (daggers), Composition (the 9-halo-eye fiction), Palette &
  Atmosphere.
- SVG geometry primitives:
  - Compass-rose sun (4 rays in the calibrated
    `--compass-rotation-deg = 22.5` pose; bright core scales with
    `--compass-ray-length` via sqrt-softening).
  - Sun pillar (vertical gradient stroke through the sun core).
  - Parhelic arc (quadratic Bezier path, curvature controlled by
    `--parhelic-curvature`; daggers ride the arc rather than a horizontal
    line; Halo Atlas canonical default is `c = 0.05` for the near-horizontal
    Troels Nielsen parhelic circle).
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

- `--parhelic-curvature = 0.05` is locked for the Halo Atlas canonical pose.
- `--cza-bloom` slider was demoted to a baked constant (`stdDeviation = 0.3`,
  edge-soften only) on 2026-05-08 after A/B testing confirmed Gaussian blur
  on each arc independently does not produce the eyelid bell curve at any
  value — the two arcs do not meet within Gaussian-σ of each other at the
  apex, so their tails never sum to a peaked bell. The bell shape is now
  produced geometrically by a paint-fill band between the two arcs
  (`#cza-bell-fill`). A future `--cza-secondary-offset` knob is the right
  lever for tuning bell width if/when a slider is wanted again; tracked as
  a Phase 9/12 composition-fiction question, not a Phase 2 calibration item.
- The compass-rose rotation is exposed as `--compass-rotation-deg`; Phase 2
  locks the canonical pose at `22.5` after rejecting the old 45° cardinal
  alignment as too overlapped with the pillar + parhelic belt.
- Halo Atlas already binds `--sun-altitude` to parhelion offset. Phase 3
  still needs to make the binding formal across the generator API and add
  the automatic CZA visibility cutoff.

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
| **Suncave Parry Arc** | Parry-orientation arc above the 22° halo; the shoulders bend back toward the sun, so it reads as a sun-facing cap rather than the upward-opening Upper Tangent envelope | ✓ optional atlas primitive |
| **Parry Supralateral Arc** | Parry-orientation variant of the supralateral family; in the atlas it is represented as paired upper-lateral shoulders outside the 46° halo | ✓ optional atlas primitive |
| **Infralateral Arcs** | paired lower-lateral arcs outside the 46° halo and below the parhelic circle; most visible as peripheral side arcs in rich displays | ✓ optional atlas primitive |
| **Sun pillar** | not a halo arc, but rendered in atlas as the vesica of two virtual halos centered at the parhelia | ✓ (atlas pillar lens) |

Two practical notes:

1. The "Parry-family" and "infralateral" primitives are deliberately
   labelled as vocabulary overlays, not full ray-traced halo physics. They
   let the workbench and overlay script speak the names used by rich halo
   displays while keeping the core parhelion/CZA calibration math separate.
   Use them as candidate labels unless the photograph gives a clean enough
   geometry to support a firmer identification.
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

### Image 1 vocabulary pass (2026-05-12)

`docs/calibration/1.Photometeor-jeff-mod_marked_red.jpg` is useful as a
vocabulary map more than as a residual-calibration target. It is the
annotated version of the clean modern halo photograph at
`docs/calibration/2.Photometeor-jeff_mod_red.jpg`: image 1 names the arc
families; image 2 is the better candidate for numeric overlay tuning.
The right move is to use image 1 as the label key without pretending that
its hand-drawn guide curves should drive the same pixel-error gate as the
photographic p0/p2/p4/p5/p7/p8/p9 cohort.

Image 1 adds three labels the prior atlas vocabulary did not carry:

- **Suncave Parry Arc**: a Parry-orientation arc above the 22° halo whose
  shoulders bend sunward. In the workbench this is separate from the Upper
  Tangent Arc, which opens upward from the 22° halo top.
- **Parry Supralateral Arc**: a Parry-family companion to the supralateral
  structure, best represented in the atlas as paired upper-lateral
  shoulders rather than the central CZA-like cap.
- **Infralateral Arcs**: paired peripheral arcs below the parhelic circle,
  outside the 46° halo, visually useful for the lower-left/lower-right edge
  arcs in rich displays.

Implementation status: optional atlas sliders and overlay flags now exist
for `suncave-parry`, `parry-supralateral`, and `infralateral`. They default
to zero intensity, because the canonical calibrated sundog remains the
parhelion + 22° halo + 46° halo + parhelic circle + CZA/upper-tangent
subset. Turn these on for vocabulary overlays or rich-display annotation,
not for the core inverse sun-altitude residual table.

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

2026-05-12 belt-height correction: across the overlay set, the parhelic
circle and dagger markers consistently sat low by about **5% of R22** even
when the 22° halo itself was well registered. `--parhelic-y-offset-r22`
now defaults to `-0.05`, raising the parhelic belt and both daggers together
while leaving the halo anchors, CZA, and tangent-arc vocabulary untouched.
The x-residual table above is unchanged; the correction is a shared vertical
registration fix for the belt layer.

### Phase 2 → Phase 3 Transition (2026-05-12)

Phase 2 calibration is closed. The Phase 2 gate from the roadmap reads
"snapshot the locked param JSON to `docs/SUNDOG_GEOMETRY_POSE.json` or
similar; future regressions are reviewable against that file." That gate is
satisfied by `public/poses/canonical-halo-atlas.json` with the
`calibration.phase2_default_lock: 2026-05-12` field. Seven photos sit in
`docs/calibration/`, seven overlays in `docs/calibration/overlays/`, and the
residual table above is the regression baseline.

**Phase 3 deliverables landed:**

- `phase3.daggerOffset(altitudeDeg)` — returns `R_22 / cos(h)`. The 7-photo
  pass validated this across altitudes 1°–60°.
- `phase3.czaVisible(altitudeDeg)` — returns true for `h ≤ 32°`. Atlas
  mode now hides the CZA primary/secondary/bell-fill above that altitude.
- `phase3.parhelicCurvature(altitudeDeg)` — empirical-fit smile growth
  that preserves the Phase 2 Troels Nielsen lock at `h=25° → c=0.05`,
  stays nearly flat near the horizon (`c≈0.03`), and rises toward `c≈0.58`
  by `h=60°` for high-sun displays.
- `--parhelic-curvature-derive` toggle CSS variable: when set > 0.5 the
  atlas ignores the manual slider and uses the altitude-derived curvature.
  Default off so the existing slider keeps its current behavior.
- Pose schema extended: `parhelicYOffsetR22`, `compassRotationDeg`, plus
  `upperTangentIntensity`, `lowerTangentIntensity`, `suncaveParryIntensity`,
  `parrySupralateralIntensity`, `infralateralIntensity` are all in
  `canonical-halo-atlas.json`.
- Atlas primitives wired in geometry + HTML + workbench sliders:
  Suncave Parry Arc, Parry Supralateral Arc, Infralateral Arc, Lower
  Tangent Arc. All deferred entries from the Canonical Halo Atlas
  Vocabulary section are now rendered.

**Phase 3 gate as written:** "setting the sun-altitude slider produces
visibly correct parhelion-offset shift; setting altitude > 32° hides the
CZA arcs automatically." First clause: passes by construction across all
seven calibration photos. Second clause: passes — atlas mode emits empty
paths for the CZA family when `h > 32°`.

**Phase 3 polish items landed (2026-05-12):**

- `public/phase3-tests.html` — assertion-style test harness over the
  `phase3.*` namespace. 33 assertions covering daggerOffset across
  altitudes 0°–60°, czaVisible across the 32° boundary, parhelicCurvature
  across the slider range, plus structural invariants (daggerOffset(0)
  === R₂₂; daggerOffset(60°) === 2·R₂₂; R₄₆/R₂₂ === 2; czaVisible
  monotonic at h=32°; parhelicCurvature(25°) === 0.05;
  parhelicCurvature monotonic and bounded in [0,1]).
  All 33 pass at landing.
- "Derive from sun altitude (Phase 3 binding)" checkbox under the
  Parhelic Circle Curvature slider. When checked, the curvature slider
  is disabled, its value display tracks `phase3.parhelicCurvature(h)`
  live as the sun-altitude slider moves, and `--parhelic-curvature-derive`
  is flipped to 1 so the geometry honors the binding. Reset clears the
  toggle. Snapshot exports include `parhelicCurvatureDerive`.
- Atmospheric Optics References section, below.

## Atmospheric Optics References

The atlas's geometric primitives and altitude bindings come from standard
atmospheric-optics literature — they are not invented for this workbench.
This section names the sources so reviewers can verify each formula
independently.

### Primary references

- **Greenler, R. (1980, reissued 1990).** *Rainbows, Halos, and Glories.*
  Cambridge University Press. The canonical reference for refraction
  geometry in hexagonal ice crystals: derives the 22° and 46° angular
  radii from the 60° and 90° prism paths through randomly-oriented and
  oriented columns, the parhelion / sundog formation at the 22° refractive
  cone intersected with the parhelic circle, and the CZA / upper tangent /
  Parry arc families from column- and plate-oriented crystal families.
- **Tape, W. (1994).** *Atmospheric Halos*, American Geophysical Union,
  Antarctic Research Series Vol. 64. Detailed treatment of arc geometry
  including the supralateral / infralateral arc symmetry around the 46°
  halo top and bottom, and the parhelion azimuthal-offset formula as a
  function of sun altitude.
- **Cowley, L. — Atmospheric Optics** (`atoptics.co.uk`). The standard
  public-facing reference with HaloSim ray-trace renderings and short
  geometric explanations of every arc family in our Canonical Halo Atlas
  Vocabulary. Used as a sanity check against our circle-primitive model.

### Formula provenance

| atlas formula | source |
| --- | --- |
| `R_22 / R_sun_sky = 22°` (small-circle angular radius) | Greenler ch. 2 (60° prism path) |
| `R_46 / R_sun_sky = 46°` (small-circle angular radius) | Greenler ch. 2 (90° prism path) |
| `parhelion offset = R_22 / cos(h)` | Tape eq. for parhelic-circle ∩ 22°-cone projection; equivalent to standard projection geometry in Greenler |
| `CZA tangent to 46° halo at top` | Greenler ch. 4 (plate-oriented crystals refracting at 90° produce the CZA's tangent geometry); Cowley atoptics CZA article |
| `CZA visible only for h ≤ 32°` | Greenler ch. 4 — at sun altitudes above ~32° the CZA's tangent point exits the visible hemisphere |
| `Upper / Lower Tangent Arc tangent to 22° halo top/bottom` | Greenler ch. 3 (column-oriented crystals); the "eyelid" geometry |
| `Suncave Parry Arc nested inside Upper Tangent` | Tape, Parry-orientation chapter; Cowley atoptics Parry article |
| `Supralateral Arc tangent to 46° halo top` | Greenler ch. 4 — column-oriented 90° path; the upper sibling of the infralateral |
| `Infralateral Arc tangent to 46° halo bottom` | Greenler ch. 4 — mirror of supralateral |

### What the atlas does NOT yet model

For completeness and honesty about scope:

- **Halo dispersion physics** — the atlas renders solid-colored or
  gradient-stroked arcs, not the wavelength-dependent ray-trace coloration
  HaloSim produces. The prismatic gradient in `sundog-theme.css` is a
  visual analogue, not a refractive calculation.
- **Subhorizon arcs** (subhelic, sub-anthelic, subsun) — these require
  observer-above-horizon geometry that the current 2D atlas projection
  cannot represent.
- **Crystal-orientation mixing** — real sundog displays often show
  partial Parry contribution mixed with column orientation. The atlas
  exposes Parry-family primitives as separate slider intensities; it does
  not model the underlying orientation-distribution that determines their
  relative brightness.
- **Crystal-size effects on arc width** — atlas arcs have fixed stroke
  width; real halos have width that depends on crystal-size dispersion.
  Out of scope.

These omissions are listed so the Phase 7 hero promotion language can
say "geometric atlas faithful to the standard literature" without
overclaiming "ray-traced atmospheric simulation." The atlas is a
geometry-primary description; the atmospheric raytrace is in
`SUNDOG_GENERATOR_SPEC.md` as "optional terminal beautification" and
remains out of scope for the workbench.

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

The Halo Atlas model is now the Phase 2 canonical default. The multi-photo
calibration pass closed the parhelion-offset residuals that Legacy and Halo
Governed left open, and the locked pose is archived in
`docs/SUNDOG_GEOMETRY_POSE.json` plus `public/poses/canonical.json`.

Legacy, Halo Scaffold, and Halo Governed remain useful regression and
comparison tracks through at least the Phase 7 hero promotion. Halo Scaffold
still should not be promoted in its current state: its `rx`/`ry` blends and
pillar circle radii are placeholder constants that have to be resolved before
any "fewer ad hoc placements" claim can survive review.

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
| `--sun-altitude` | Math-derived | Drives Halo Atlas parhelion offset now; Phase 3 formalizes the generator binding and CZA visibility cutoff |
| `--halo-22-intensity` | Free | Visual knob; opacity of the 22° halo ring |
| `--halo-46-intensity` | Free | Visual knob; opacity of the 46° halo ring |
| `--cza-intensity` | Free | Visual knob; opacity of both CZA arcs |
| `--cza-curvature` | Free | Visual knob; apex of the primary CZA arc |
| `--sun-pillar-intensity` | Free | Visual knob; opacity of the vertical pillar |
| `--sun-pillar-length` | Free | Visual knob; pillar extent above and below sun |
| `--parhelic-circle-intensity` | Free | Visual knob; opacity of the parhelic arc |
| `--parhelic-curvature` | Math-derived (planned) | Locked at 0.05 for the Halo Atlas canonical pose; Phase 3 will derive from sun altitude |
| `--parhelic-y-offset-r22` | Calibration-derived | Moves the parhelic belt and daggers together as a fraction of R22; default `-0.05` corrects the common belt-low overlay residual |
| `--parhelia-intensity` | Free | Visual knob; opacity of the dagger group |
| `--parhelia-dagger-length` | Free | Visual knob; horizontal extent of the dagger streaks |
| `--dispersion-width` | Free | Visual knob; atmosphere overlay intensity |
| `--secondary-suns-strength` | Composition fiction | Controls the 9-halo-eye effect (not standard atmospheric optics) |
| `--ring-overlap-bias` | Composition fiction | Golden-ratio knob for the 9-ring intersection |
| `--compass-ray-length` | Free | Visual knob; ray length and core radius scale together |
| `--compass-rotation-deg` | Free | Phase 2 visual knob; locked at 22.5 so the compass X stays distinct from the pillar/parhelic cross |
| `--rainbow-saturation` | Free | Visual knob; reserved for Phase 2 prismatic gradient tuning |
| `--idle-scintillation-amplitude` | Animation | Phase-4 idle drift amplitude; off when 0 |
| `--supralateral-intensity` | Annotation / vocabulary | Optional Halo Atlas layer for rich-display labeling; outside the calibrated core |
| `--upper-tangent-intensity` | Annotation / vocabulary | Optional 22°-halo tangent layer for distinguishing upper tangent from CZA |
| `--lower-tangent-intensity` | Annotation / vocabulary | Optional mirror tangent layer below the 22° halo |
| `--suncave-parry-intensity` | Annotation / vocabulary | Optional Parry-family cap from the image-1 vocabulary pass |
| `--parry-supralateral-intensity` | Annotation / vocabulary | Optional paired Parry supralateral shoulders for rich-display overlays |
| `--infralateral-intensity` | Annotation / vocabulary | Optional lower-lateral peripheral arcs for rich-display overlays |

The central typing should be explicit:

```text
M       = parhelion description (the math)
P_t     = parameter state at time t (slider values)
G(P_t)  = derived geometry (positions, curvatures, intensities)
R(G)    = rendered SVG composition
A(G)    = optional annotation/vocabulary layers over the rendered geometry
```

Phase 3 binds `M`-derived params (sun-altitude → CZA visibility, parhelion
offset, parhelic curvature) so the workbench cannot render a
physically-impossible pose by accident. Free params remain free knobs; the
fiction params (composition family) and annotation params (rich-display
vocabulary) are deliberately outside `M` and visibly labelled in the right
rail.

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
from the parhelion description, which are composition fiction outside the
description, and which are annotation/vocabulary layers. Document the
binding equations for the math-derived set so Phase 3 has unambiguous
targets.

Deliverables:

- `docs/SUNDOG_V_GEOMETRY.md` (this document) with the Actionability Audit
  table populated and the math-binding TODO list explicit.
- A short notes file with the parhelion description equations the workbench
  will eventually exercise (sun-altitude → parhelion offset, CZA visibility
  window, parhelic-circle curvature derivation).

Gate: the Actionability Audit table identifies every slider's tier,
including the annotation/vocabulary tier, and the math-binding TODOs are
enumerated. **Status: in progress with this document.**

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

- [x] `--parhelic-curvature = 0.05` for Halo Atlas canonical
  (locked against the near-horizontal Troels Nielsen parhelic circle).
- [x] `--cza-bloom` promoted to slider (2026-05-08); default lowered from
  the over-aggressive `3.2` to `1.4` pending overlay calibration.
- [x] `--cza-bloom` demoted to baked constant `stdDeviation = 0.3`
  (2026-05-08, ass-dyno value). Mechanism mismatch confirmed: Gaussian
  blur on each arc independently does not sum to a bell curve at the
  eyelid apex regardless of σ. Replaced with `#cza-bell-fill` paint band
  between the two arcs. The eyelid-bell goal
  is reclassified from a Phase 2 calibration item to a Phase 9/12
  composition-fiction question whose right lever is `--cza-secondary-offset`
  (geometric overlap), not blur.
- [x] Compass-rose rotation angle locked at `--compass-rotation-deg = 22.5`.
  The old 45° cardinal alignment made the rays collapse into pillar +
  parhelic; 22.5 keeps the X visible without fully backing out to the
  original diagonals.
- [x] Default values locked against the source photo: `--halo-22-intensity
  = 0.95`, `--halo-46-intensity = 0.45`, `--cza-intensity = 0.95`,
  `--parhelia-intensity = 1.0`.
- [x] Default for `--dispersion-width = 0.70` (atmosphere bloom level).
- [x] Default for `--compass-ray-length = 0.85` (X visibility vs core
  dominance).
- [x] Decision: keep 9-halo-eye composition (`--secondary-suns-strength`)
  default at `0.0` for the canonical pose; the fiction stays opt-in.
- [x] `--parhelic-y-offset-r22 = -0.05` added 2026-05-12. Multi-photo
  review showed the parhelic belt / dagger markers were landing about 5%
  of R22 too low while the 22° halo anchor was correct. This correction
  moves only the parhelic circle and daggers; halo, CZA, and tangent
  anchors remain unchanged.

Gate: snapshot the locked param JSON to `docs/SUNDOG_GEOMETRY_POSE.json`
or similar; future regressions are reviewable against that file. **Status:
landed 2026-05-12 as `docs/SUNDOG_GEOMETRY_POSE.json` and
`public/poses/canonical.json`.**

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

### Phase 4 - Public Explainer Build (`sundog.html`) — 2026-05-12 restructure

**Pivot:** the old Phase 4-7 chained idle scintillation → active reveal →
selective 3D → brand hero. The 2026-05-12 review reframed the goal: the
strongest play for this codebase isn't a brand-mark animation arc, it's an
*educational explainer* that earns SEO, Wikipedia outreach, and the inverse
direction (visitors bringing their own photos to the math). Old Phase 9
"Educational Mode" is absorbed into this Phase 4; the animation goals are
demoted to sub-features of the page rather than load-bearing phases.

Goal: ship `sundog.html` — a long-scroll public page that is *the* sundog
reference on the open web. Replaces `sundog-workbench.html` as the canonical
URL (the workbench embeds inline; a `/sundog-workbench.html` redirect
preserves any deep links). Page does the educational job that legacy
atmospheric-optics pages do (e.g. the WW2010 Illinois reference) while
1-upping them with the interactive atlas math.

Deliverables:

- **`sundog.html`** with the eight-section skeleton:
  1. Hero — live workbench at canonical pose, single-sentence framing.
  2. *What you are looking at* — labeled photograph (annotated reference
     from `docs/calibration/1`).
  3. *Why bright spots at 22°* — two-ice-crystal hexagonal-column refraction
     diagram (static SVG, Greenler-cited).
  4. *Why they slide outward as the sun rises* — interactive sun-altitude
     slider + live `R / cos(h)` callout + inverse-inference demo.
  5. *The full atlas* — every named arc with one paragraph, atlas-rendered
     illustration, "highlight in the live workbench" toggle.
  6. *Try it on your own photo* — Phase 5 upload widget mounts here.
  7. *History and reading* — Greenler, Tape, Cowley; Vädersoltavlan as
     historical illustration.
  8. *For other tools* — JSON pose schema, `phase3` namespace,
     `overlay_calibrate.py`, paper / writeup links.
- "Show advanced controls" toggle that reveals the full 25-slider rail —
  default view shows ~8 hero sliders (sun-altitude, the named-arc
  intensities, the derive-curvature toggle).
- Schema.org `LearningResource` JSON-LD markup with author, citations,
  educational level, learning objectives.
- Sitemap entry and meta tags (canonical, og:image from the calibration
  overlay, Twitter card).
- Animation goals from the old phases land here as polish, not gates:
  - Idle scintillation: subtle saturation drift on the prismatic gradient
    when the workbench has been idle for 5s+, gated by
    `--idle-scintillation-amplitude` and `prefers-reduced-motion`.
  - Active reveal: optional "replay the construction" button on the
    in-page workbench that sequences sun → 22° halo → parhelia → CZA →
    vocabulary layers.

Gate: a visitor unfamiliar with sundogs lands on `sundog.html`, reads
without scrolling away, can locate and label each visible feature in their
own past sky photos by section 5, and can pull the math down through
section 8. Wikipedia editors reviewing the page recognize the formula
provenance (References) and can verify each claim independently.

Status (2026-05-12): Phase 4 is now checkable in-repo. `sundog.html` carries
the eight sections, canonical/meta tags, LearningResource JSON-LD with author
and citations, the Phase 3 harness reference, and the §6 upload mount. The
root crawl artifacts live in `public/sitemap.xml` and `public/robots.txt`.
`npm run sundog:check` verifies the public page contract and the Phase 5
endpoint files before a build.

### Phase 5 - Photo Upload + Inverse Inference

Goal: visitor uploads a sundog photo → marks the sun, the 22° halo edge,
and a parhelion → the atlas overlays the predicted geometry on the photo
in-browser → user sees their own altitude inverse-inferred. With consent,
the photo + JSON pose flow into a training-data pipeline.

**Backend decision:** Cloudflare Workers + R2 (recommended path chosen
2026-05-12). The project already runs on Cloudflare (`wrangler` in
package.json); the upload endpoint extends the existing deployment.

Deliverables:

- **Browser side**
  - Drag-and-drop / file-picker upload widget on `sundog.html` §6.
  - Three-click measurement UX: "Click the sun" → "Click the 22° halo
    edge" (single point gives radius) → "Click a visible parhelion."
  - Live atlas overlay rendered on top of the user's image using the
    same primitive math as `overlay_calibrate.py` (port to JS).
  - Inverse-inferred sun altitude shown as a number: *"Your sun was at
    h ≈ 25.0°."*
  - EXIF stripping happens **client-side** before any POST. Re-encode
    the image through canvas to drop metadata. User sees the EXIF that
    will be removed and confirms.
  - Consent gate: opt-in checkbox *"Share my photo with the Sundog
    project to improve the atlas model."* Default unchecked. Separate
    from the rendering, which always happens locally.
- **Backend side (Cloudflare Workers + R2)**
  - `POST /api/sundog/upload` endpoint accepting `{ image, pose,
    consent }` JSON.
  - R2 bucket `sundog-uploads` with prefixed object keys
    `submissions/{ISO-date}/{opaque-uuid}.{ext}`.
  - Metadata KV / D1 row: `{ submission_id, timestamp, inferred_h_deg,
    pose_json, ip_country, consent_flag, deletion_token }`.
  - Deletion endpoint `POST /api/sundog/delete` accepting the deletion
    token returned to the user at submission time.
  - Rate limit: 5 uploads / IP / hour via Workers KV.
- **Privacy + policy**
  - Page section linking to a `PHOTO_DATA_POLICY.md` doc that states
    what we collect (image, JSON pose, anonymous timestamp, IP-derived
    country), what we use it for (improving the atlas model and
    illustrations), retention period (12 months unless aggregated into
    a derived dataset), and how to request deletion.
  - GDPR-style "right to deletion" reachable via the deletion token
    issued at submission.

Gate: a visitor can upload a sundog photo, see their altitude
inverse-inferred and the atlas overlaid on the photo, and either save
locally or share to the project — and if they share, the data is in R2
within 5 seconds with EXIF stripped, and they receive a deletion token
they can use to remove it.

Status (2026-05-12): Phase 5A and the Phase 5B endpoint scaffold are in
place. `public/js/photo-upload.mjs` provides the three-click in-browser
inverse inference, EXIF-stripped canvas re-encode, overlay download, JSON pose
copy, backend health probe, opt-in upload, and local deletion-url retention.
`functions/api/sundog/` now contains `health`, `policy`, `upload`, `delete`,
and shared Worker helpers wired for the R2 bucket and KV namespace named in
`wrangler.toml`. Remaining gate check is an end-to-end Cloudflare Pages run:
upload a real photo, inspect R2, delete with the returned token, and confirm
the R2 objects and KV deletion index are gone.

Live gate receipt (2026-05-12 local / 2026-05-13 UTC): passed on deployment
`https://60c76dae.sundog-9kv.pages.dev` using calibration p0
(`docs/calibration/0.troelsnielsendr.png`) and the documented
`sun=(400,356), R22=145, parhelion_offset=160` pose. Health reported
R2+KV bindings live and policy `2026-05-12`; upload returned `201` with
`inferred_h_deg=25.01`; R2 listed exactly three objects before deletion
(`*.png`, `*.pose.json`, `*.meta.json`); the KV deletion index pointed at the
same base key; deletion returned `200` with the same submission id; R2 listing
for the base key returned zero objects after deletion; KV lookup returned
`404`.

Verification note: do not verify image deletion by first fetching the image
body through the Cloudflare R2 objects API and then fetching the same URL
again. Image body reads can be served back from Cloudflare's API cache after
deletion. Use the object-list endpoint and/or a fresh metadata lookup for the
gate receipt.

### Phase 6 - Drag-to-Tune Constraint Network

Goal: click any rendered primitive and drag it; the parameter it inverse-
binds to recomputes, and all dependent primitives re-derive automatically.
This is the live demonstration of the field-and-signature claim: every
visible feature has a parameter behind it, and the user can manipulate the
parameter through its signature.

**Approach:** inverse-bind each draggable handle to *one* primary
parameter. No generalized constraint solver. Dragging causes the bound
parameter to update; other primitives that depend on that parameter (via
the Phase 3 binding network) re-derive cleanly.

Inverse-bind table:

| handle | binds parameter | dependent re-derives |
|---|---|---|
| Parhelic-arc apex | `parhelic-curvature` | parhelic-circle stroke only |
| Parhelion / dagger position | `sun-altitude` | both daggers, parhelic-circle endpoints, CZA visibility, supralateral, infralateral |
| CZA apex | `cza-curvature` | CZA primary + secondary, bell-fill |
| Sun center | sun pixel position (translation) | entire atlas translates as a rigid body |
| 22° halo edge | `R₂₂` (anchor scale) | entire atlas rescales via the workbench→photo scale factor |

Deliverables:

- SVG hit-test handles on each primitive (transparent overlay shapes
  sized for comfortable touch / mouse pointer use, not the visible
  stroke pixels).
- Pointer event plumbing (`pointerdown` → `pointermove` → `pointerup`
  with proper capture / release / cancel handling).
- Inverse functions per binding:
  - `sun-altitude` ← `arccos(R₂₂ / |dragged_dagger_x − sun_x|)`
  - `parhelic-curvature` ← solve for apex height given drag y
  - `cza-curvature` ← solve for CZA apex y given drag (clamped to slider
    range)
  - sun-position ← drag delta applied to all primitive centers
  - `R₂₂` ← Euclidean distance from sun to drag position
- Slider sync: as drag updates a bound parameter, the corresponding
  slider's value display updates live so users see the math change.
- `prefers-reduced-motion` doesn't disable drag (it's user-initiated
  motion), but transitions on dependent primitives become instant.

Gate: a user can grab the parhelic arc's apex, drag it up to flatten the
smile or down to deepen it, and the slider value at the right rail
updates in lockstep. The same user can grab a parhelion and drag it
outward; sun-altitude rises, the CZA disappears once h > 32°, and the
opposite parhelion mirrors the motion symmetrically.

Status 2026-05-13: Phase 6 has entered the workbench with two live inverse
bindings:

- Parhelion/dagger handles update `--sun-altitude`, with the handle y-position
  now following the calibrated `--parhelic-y-offset-r22` belt rather than the
  old sun centerline.
- A parhelic-arc apex handle updates `--parhelic-curvature` directly. If the
  Phase 3 "derive from sun altitude" toggle is on, grabbing this handle
  switches curvature back to manual mode so the drag has one unambiguous bound
  parameter.
- The first constraint gate is backed by
  `scripts/check-sundog-phase6-drag.mjs`, which dispatches pointer drags
  against the live module and verifies curvature, altitude, and mirrored
  parhelion positions.

Next Phase 6 handles: CZA apex to `--cza-curvature`, then sun-center
translation and 22° halo edge scaling once the per-primitive bindings feel
stable.

### Phase 7 - Hero Promotion + Wikipedia / SEO Outreach

Goal: `sundog.html` becomes the project's canonical sundog page; the
`index.html` landing page is rebuilt to route visitors there; the
Wikipedia outreach packet ships with the math, references, and claim
license that lets editorial reviewers verify what's original vs. cited.

Deliverables:

- **`index.html` rebuild**
  - Hero section embeds the calibrated atlas snapshot with one CTA into
    `sundog.html`.
  - Other application links (Balance, Three-Body, etc.) routed via a
    secondary navigation row.
  - Old `#parhelion-canvas` decommissioned; any references in the brand
    or icon docs updated.
- **Wikipedia / SEO outreach packet** — `docs/SUNDOG_OUTREACH_PACKET.md`
  - One-page math summary: the seven atlas formulas, each with its
    source citation from the Atmospheric Optics References section.
  - Claim license: which equations are *standard* (cited to Greenler /
    Tape / Cowley) vs. *original to this project* (the integrated
    primitive-atlas presentation; the interactive inverse-inference of
    sun altitude from a photo; the calibration evidence across 7
    photos).
  - Reproducibility statement linking to `phase3-tests.html`,
    `overlay_calibrate.py`, and the calibration overlay set.
  - Suggested Wikipedia edits: which articles (Sun dog, Circumzenithal
    arc, 22° halo) could cite our atlas, and the specific factual claim
    each edit would support.
- **Press / social snapshot assets**
  - Static PNG hero rendered from the canonical pose at multiple aspect
    ratios (16:9, 1:1, 9:16 for stories).
  - 8–15 second WebM clip of the active reveal sequence for press use.
  - OG image referenced from `sundog.html`'s meta tags.
- **Schema.org and search markup**
  - `LearningResource` + `Article` JSON-LD on `sundog.html`.
  - `WebSite` + `Organization` JSON-LD on `index.html`.
  - Updated sitemap.xml; robots.txt rules confirmed.

Gate: a first-time visitor from a Wikipedia citation lands on
`sundog.html`, reaches a verifiable claim within one scroll, and can
trace any formula to a Greenler / Tape / Cowley citation in section 7.
A Wikipedia editor reviewing the page can identify the original
contribution (interactive inverse-inference + integrated atlas) and the
cited content (every individual formula) without ambiguity.

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
  - [ ] `rich-display-vocabulary.json` — Halo Atlas annotation pose with
    upper/lower tangent, suncave Parry, Parry supralateral, and
    infralateral layers exposed for teaching and design review.

Gate: round-trip a snapshot → load → render → snapshot and confirm
byte-equal JSON across two browser sessions.

## Post-Verdict / Conditional Roadmap

These phases unlock only after Phase 7 promotion lands and the hero is in
the wild for some adoption period.

### Phase 9 - Educational Mode *(2026-05-12: absorbed into Phase 4)*

This phase is no longer separate. Its goals — overlay annotations
(angle measures, crystal-orientation arrows, refraction-angle labels;
labeled named arcs) — are now §3 and §5 of `sundog.html`. The "show
annotations" toggle becomes a per-section interaction inside Phase 4,
not a phase of its own.

The label set rule survives: every rare/optional label must say
"candidate" or "annotation" unless the source image has enough geometry
to support a firmer identification. This rule applies to the §5 "full
atlas" walkthrough in `sundog.html`.

### Phase 10 - Rich-Display Overlay Tuning

Goal: tune the optional vocabulary primitives against the richer calibration
photos before they influence logo or animation language. This pass is not a
replacement for the Phase 2 static calibration gate; it is a morphology
pass for rare/side arcs and rich-display vocabulary.

Reference images:

- `docs/calibration/2.Photometeor-jeff_mod_red.jpg` — clean modern halo
  display with CZA, upper tangent, parhelic circle, sundogs, and lower
  peripheral structure. Pair with the annotated image 1 label key when
  deciding which optional vocabulary layers are present.
- `docs/calibration/7.625544777_10241089047341957_4435817776770420050_n.jpg`
  — high-sun / one-sided parhelion case from the multi-photo calibration
  pass; useful for stress-testing asymmetry and partial visibility.
- `docs/calibration/13.480859565_17934474635991868_323320248088719839_n.jpg`
  — square social-photo reference for logo/social-crop tuning.

Deliverables:

- Overlay outputs for p2, p7, and p13 with optional vocabulary layers
  toggled and separately colored. First-pass artifacts live at
  `docs/calibration/overlays/p2.rich-vocabulary.overlay.png`,
  `docs/calibration/overlays/p7.rich-vocabulary.overlay.png`, and
  `docs/calibration/overlays/p13.rich-vocabulary.overlay.png`.
- Per-photo notes that classify each optional primitive as `visible`,
  `candidate`, `not visible`, or `not applicable`; first-pass notes live in
  `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`.
- Updated defaults or pose presets only if the same morphology appears
  consistently across at least two of the three images.
- A short "do not promote" list naming any arc that looked plausible in one
  image but failed the richer pass.

Gate: the optional vocabulary layers are stable enough to give to a design
team as shape language, without letting them rewrite the calibrated core.

### Phase 11 - Characterized Logo & Animation Toolkit

Goal: after Phase 10 overlay tuning, update the logo documentation and hand
the graphic design team a small, bounded toolset for a characterized Sundog
logo/animation family. "Characterized" means the logo can act like a living
mark — idle, blink/reveal, shimmer, label-callout — while preserving the
calibrated parhelion grammar.

Deliverables:

- `docs/ICON_ASSETS.md` expanded into a design brief: source SVG anatomy,
  allowed simplifications, protected geometry, color tokens, small-size
  rules, and animation states.
- A design-team toolset folder or script set that can export at least:
  static SVG, favicon/app-icon PNGs, a transparent logo PNG, and a short
  web animation source (`webm`/`gif`/CSS/SVG animation, final format TBD).
- A canonical "character sheet" for the logo: core sun, 22° iris,
  parhelia glints, CZA/upper eyelid, optional vocabulary arcs, and which
  layers may be omitted at tiny sizes.
- Motion rules mapped to the workbench phases: idle scintillation, active
  reveal, hover shimmer, and reduced-motion fallback.
- Design handoff examples based on the Phase 10 p2/p7/p13 overlay findings,
  so logo expressiveness is borrowed from calibrated sky morphology rather
  than invented decoration.

2026-05-12 first-pass artifacts:

- [x] `scripts/generate-sundog-logo-toolkit.mjs` plus `npm run
  logo:toolkit` export static SVG, transparent SVG, SVG/CSS animation,
  layer manifest JSON, favicon PNG proofs, app-icon PNG proofs, and a
  transparent PNG proof.
- [x] `docs/LOGO_ANIMATION_TOOLKIT.md` now gives the design team a compact
  character sheet, protected geometry, optional vocabulary boundary,
  motion states, and small-size rules.
- [x] `docs/ICON_ASSETS.md` lists the Phase 11 prototype assets and links
  the generator to the richer p2/p7/p13 overlay notes.
- [ ] Production favicon/app-icon replacement remains a later design-review
  step; the current site icon wiring is unchanged.

Gate: a designer can regenerate the Phase 11 prototype icon set and produce one
animated Sundog mark without reading the entire geometry roadmap, while the
assets still trace back to Halo Atlas primitives.

### Phase 12 - Atmospheric Variations

Goal: alternative sky palettes (dawn, dusk, twilight, polar) with calibrated
photograph references for each. Optional `--sky-tint` becomes a real palette
slider rather than a single-axis warm/cool knob.

### Phase 13 - Linked Description Mode

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
  visual knob clearly labelled as such, a composition-fiction knob clearly
  labelled as such, or an annotation/vocabulary layer clearly labelled as
  outside the calibrated core.

The Geometry workbench does NOT claim:

- Novel atmospheric optics findings. The physics is well-known; the
  contribution is the parametric-rendering format and the
  composition-as-brand framing.
- That the parhelion is "discovered" by the workbench. The phenomenon has
  been observed and described for centuries; Sundog's contribution is the
  description style and the workbench-as-proof presentation.
- That the workbench renders all halo phenomena. Its calibrated core is the
  parhelion + CZA + parhelic-arc subset chosen for the hero composition.
  Optional Parry-family and infralateral layers are annotation primitives
  for rich-display vocabulary; sub-suns and full crystal-orientation
  simulation remain out of scope unless added deliberately later.

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
