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

## Document Map

*This roadmap is long; this is the jump table. Line numbers are a
2026-05-14 snapshot — they drift as the document is edited, so treat them
as approximate and fall back to searching the heading text if a number is
stale. The generative HaloSim work (HS-0…HS-7) has moved to its own
document — see Cross-references.*

**Top-level sections**

| line | section |
| ---: | --- |
| 91 | Related Documents |
| 144 | Sundog Expression |
| 184 | Current State |
| 246 | Parallel Geometry Models (Legacy vs Halo Scaffold vs Halo Governed) |
| 561 | Atmospheric Optics References |
| 883 | Why This Workbench |
| 902 | Visual Direction |
| 946 | Actionability Audit |
| 1000 | Ratified Hook Language |
| 1016 | Roadmap |
| 1438 | Post-Verdict / Conditional Roadmap |
| 1890 | HaloSim Cinematic Sidecar — tracked separately |

**Roadmap phase ladder**

| line | phase |
| ---: | --- |
| 1018 | Phase 0 — Parameter Taxonomy and Description Lock |
| 1038 | Phase 1 — Static Composition Scaffold |
| 1053 | Phase 2 — Calibration Pass (Static) |
| 1098 | Phase 3 — Math Binding |
| 1120 | Phase 4 — Public Explainer Build (`sundog.html`) |
| 1182 | Phase 5 — Photo Upload + Inverse Inference |
| 1261 | Phase 6 — Drag-to-Tune Constraint Network |
| 1329 | Phase 7 — Hero Promotion + Wikipedia / SEO Outreach |
| 1376 | Phase 8 — Reproducible Pose Export *(closed)* |
| 1443 | Phase 9 — Educational Mode *(absorbed into Phase 4)* |
| 1456 | Phase 10 — Rich-Display Overlay Tuning |
| 1646 | Phase 11 — Characterized Logo & Animation Toolkit |
| 1675 | Phase 12 — HaloSim-Aligned Vocabulary + Ground-Truth Discipline |
| 1690 | Phase 12A — HaloSim ground-truth validation gate |
| 1731 | Phase 12B — Tilt-dispersion as math-bindable parameter |
| 1810 | Phase 12C — Full-atlas vocabulary completeness |
| 1907 | Phase 13 — Atlas Claim Governance + Ask Sundog Coupling |
| 1955 | Phase 14 — Complete Halo Phenomena Accounting Matrix |
| 1998 | Phase 15 — Speculative / Unphotographed Halo Proof Program |
| 2042 | Phase 12 deferred items |

**Cross-references**

- [`SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md`](SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md)
  — the **generative** HS-0…HS-7 sidecar (split out 2026-05-14; does not
  gate hero promotion; not in the main validation / claim-governance chain).
- [`HALOSIM_VALIDATION_PROTOCOL.md`](calibration/HALOSIM_VALIDATION_PROTOCOL.md)
  — the **validation**-direction HaloSim procedure (Phase 12A/12B). Do
  not conflate with the generative sidecar.
- [`SUNDOG_GENERATOR_SPEC.md`](SUNDOG_GENERATOR_SPEC.md) ·
  [`SUNDOG_OVERLAY_PROTOCOL.md`](SUNDOG_OVERLAY_PROTOCOL.md) ·
  [`SUNDOG_V_PERCEPTION.md`](SUNDOG_V_PERCEPTION.md) ·
  [`SUNDOG_V_CHAT.md`](SUNDOG_V_CHAT.md) ·
  [`UI_UX_THEME_FOUNDATION.md`](site/UI_UX_THEME_FOUNDATION.md) — companion
  docs; full descriptions in **Related Documents** below.

*[nice-to-have #14]*

## Related Documents

This roadmap is the workbench's path-to-promotion plan. It is the
human-tunable face of a broader **Sundog Generator** whose programmatic
face and proof methodology are split into companion docs. Authority is
distributed so no document trampolines another's decisions.

- [`UI_UX_THEME_FOUNDATION.md`](site/UI_UX_THEME_FOUNDATION.md) — the site-wide
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
- [`ICON_ASSETS.md`](site/ICON_ASSETS.md) — the current favicon/app-icon
  source list and regeneration notes. Phase 11 below expands that into a
  design-team toolkit for a characterized Sundog logo and animation set.
- [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md) and
  [`SUNDOG_V_BALANCE.md`](SUNDOG_V_BALANCE.md) — sibling application
  workbench roadmaps. Different shape (control task with operating
  envelope and pre-registered verdict), same Sundog pattern of *hidden /
  indirect / transformation / output*. The Pre-Committed Cross-Application
  Comparison Row near the bottom of this document is the parallel.
- [`SUNDOG_V_PERCEPTION.md`](SUNDOG_V_PERCEPTION.md) — the
  atlas-as-instrument roadmap. Sits beyond Phase 7 hero promotion;
  reframes the §What the atlas does NOT yet model list as a prediction
  surface and instruments three candidate predictions (sub-visible
  parhelic continuation, intersection enhancement, color-order
  inversion at tangencies) ranked by apparatus cost. Inherits mesa
  Phase 6 pre-registration discipline.
- [`SUNDOG_V_CHAT.md`](SUNDOG_V_CHAT.md) — the public assistant /
  claim-boundary roadmap. Phase 13 below borrows its update discipline:
  when `sundog.html` adds or sharpens halo vocabulary, Ask Sundog must
  distinguish rendered primitives, named-only literature coverage,
  not-modeled families, and speculative / proof-program work before the
  next deploy.
- `public/poses/*.json` — the named-pose library. `canonical.json`
  snapshots the locked default; future poses (`low-altitude`, `cza-heavy`,
  `nine-halo-eye`, `rich-display-vocabulary`) ship as Phase 8 and Phase
  10 land.
- BoxForge tools library: sibling checkout `../Dungeon Gleaner Main/tools`,
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
- **Tape, W. (1994) — *Atmospheric Halos*** (AGU Antarctic Research Series
  Vol. 64). Chapters on disk:
  [`docs/calibration/AH-CH06/`](calibration/AH-CH06) (Chapter 6: "The Role
  of Sun Elevation" — display gallery with sun-elevation dependence of arc
  shape; verifies the 29° tangent-arc → circumscribed-halo transition cited
  in wave-2 W1 of the Phase 11 dispatch) and
  [`docs/calibration/AH-CH10/`](calibration/AH-CH10) (Chapter 10:
  "Pyramidal Crystals and Odd Radius Circular Halos"). Tape Ch 10
  establishes the pyramidal-crystal odd-radius family (9°, 18°, 20°,
  22.9°, 23.8°, 24°, 35° halos) — currently outside Sundog's primitive
  set; see "What the atlas does NOT yet model" below.
- **Tape, W. & Moilanen, J. (2006) — *Atmospheric Halos and the Search for
  Angle x*** (AGU Antarctic Research Series Vol. 84). Chapter 11 on disk
  at [`docs/calibration/AH-SAX-CH11/`](calibration/AH-SAX-CH11). Establishes
  the pyramidal-crystal {1 0 -1 1} face inclination history (Bravais 1847's
  x ≈ 54.7°; HaloSim's filename convention `Pyr_*_27.98.xsh` uses the
  complementary 27.98°) and Table 11.1's wedge-angle / halo-radius mapping
  for the Bravais-Clarke pyramidal crystal (9°/18°/22°/28°/35°/46° halo
  radii from specific wedge configurations).
- **HaloSim3** (Cowley, L. & Schroeder, M. 2004). The standard Monte Carlo
  halo simulator. Installed at `%USERPROFILE%\HalSim361.exe` with the full
  asset library (40 orientation `.xng` files, 6 crystal shape `.xsh` files,
  `Water-Ice.xmt` material file, 38 canonical simulation `.sim` files
  including named historical displays such as Parry 1820 and St Petersburg
  1790). **Used as ground-truth oracle for Pass C7** (Phase 11 wave-2 W4
  follow-up); receipt at
  [`docs/calibration/PASS_C7_OUTPUT.txt`](calibration/PASS_C7_OUTPUT.txt).
  Provides the canonical vocabulary for crystal-orientation populations
  (Horiz column, plate, Parry, Lowitz, random) and their tilt-dispersion
  parameterization. See "Crystal orientation vocabulary" + "HaloSim as
  ground-truth oracle" subsections below.

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

### Crystal orientation vocabulary (HaloSim-aligned)

Atmospheric optics distinguishes halo families by the *crystal orientation
distribution* that produces them, not just by the geometric arc shape.
HaloSim canonizes this vocabulary with named orientation files (`.xng`)
parameterized by tilt dispersion (standard deviation in degrees of the
crystal's tilt from the canonical orientation). The Sundog atlas inherits
this vocabulary even where it doesn't yet expose the underlying physical
knob:

| HaloSim orientation | tilt-dispersion exposed | produces (in this atlas) | Sundog status |
| --- | --- | --- | --- |
| `random.xng` | n/a (isotropic) | 22° halo, 46° halo, pyramidal odd-radius halos | implicit in atlas formulas (formulas assume random orientation for halo radii) |
| `Horiz column .{N} deg disp.xng` (N = 0.05° → 5°, 9 levels) | yes | upper / lower tangent arc, circumscribed halo (h > 29°), parhelic circle, Wegener arcs | rendered, but tilt-dispersion is implicit / fixed |
| `plate .{N} deg disp.xng` (N = 0.05° → 34°, 13 levels) | yes | parhelia, CZA, supralateral, infralateral, 120° parhelia | rendered, tilt-dispersion implicit / fixed |
| `Parry .{N} - .{M} deg disp.xng` (8 two-parameter levels) | yes | suncave Parry, sunvex Parry, Parry supralateral, Parry tangent arcs | named in vocabulary but not separately rendered |
| `Lowitz .{N} deg disp.xng` (8 levels including angle-specific 18° / 36°) | yes | Lowitz arcs | not modeled |
| Pyramidal shapes (`Pyr_*_27.98.xsh`) | n/a (shape parameter) | 9°, 18°, 20°, 22.9°, 23.8°, 24°, 35°, 46° odd-radius halos | not modeled — vocabulary recognized via Tape AH Ch 10 + AH-SAX Ch 11 citations |

**Physical parameter, not visual proxy.** Tilt dispersion is a real physical
property of the crystal population at the moment of the halo display
(temperature, crystal-formation conditions, fall-orientation stability). The
atlas's current visual knobs (`--parhelic-curvature`, `--cza-curvature`,
`--cza-secondary-offset`, etc.) implicitly absorb tilt-dispersion effects
without naming them. A future revision (composition-fiction phase beyond
Phase 11) could bind these knobs to explicit tilt-dispersion parameters via
Phase 3-style math binding; this would canonize the physical interpretation
that the visual sliders currently obscure.

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

> **Apology surface, prediction surface.** Beyond Phase 7, this list is
> the staging surface for [`SUNDOG_V_PERCEPTION.md`](SUNDOG_V_PERCEPTION.md):
> the atlas-as-instrument roadmap argues each item is a falsifiable
> prediction the 7-formula field makes when extended along its own
> generators. Halo dispersion physics ↔ Prediction 1 (sub-visible
> parhelic continuation polarization curve) and Prediction 3
> (color-order inversion at tangencies); the multi-circle joint
> structure ↔ Prediction 2 (intersection enhancement), which is novel
> against this list. Subhorizon arcs, crystal-orientation mixing, and
> crystal-size effects on arc width remain on the apology side of the
> ledger with no committed perception phase. The perception roadmap
> inherits the pre-registration / smoke-gate discipline from mesa
> Phase 6 wholesale.

### HaloSim as ground-truth oracle

The Phase 10 attack campaign and Phase 11 outreach passes (especially Pass
C7) used HaloSim ray-tracing output as ground truth for testing the atlas's
forward rendering and inverse-inference under canonical literature
parameterizations. The pattern generalizes:

1. **Forward-rendering claims.** When the atlas asserts a specific arc
   shape or position as a function of `h` (or any other parameter), a
   HaloSim Monte Carlo render at the same `h` provides an independent
   reference. Pass C7 used this to test the upper-tangent-arc opening-
   angle inverse handle against the Pass C5 hand-anchor circle fit and
   found the C5 circle geometrically inconsistent with the canonical
   curve (1.2-5.6° radial over-extension, progressive with azimuth).
2. **Inverse-handle validation.** Where the inverse inference depends on
   the canonical literature parameterization being the one the atlas
   geometric primitives realize, HaloSim becomes the disambiguation
   oracle. Pass C7 demonstrated that "project-original circle-fit
   curvature" and "canonical literature opening-angle" are *not*
   equivalent under the tilt-dispersion conditions HaloSim renders.
3. **Tilt-dispersion sensitivity.** HaloSim renders at different tilt
   dispersions (0.05°, 0.1°, 0.5°, 1°, …) characterize how arc shape
   and brightness profile change with crystal-population parameters that
   the atlas currently hides. This is the natural way to quantify any
   "below atlas measurement precision" hedge the project writes about
   atmospheric effects.

The procedure is documented in
[`docs/calibration/PASS_C7_OUTPUT.txt`](calibration/PASS_C7_OUTPUT.txt):
acquire a HaloSim render at the target `h`, lock the pixel-to-degree
scale via the 22° halo radius (sun-centered Camera View; HaloSim
auto-zooms with crystal-block configuration, so the scale must be
re-locked per render configuration), then run a 2D arc-locus search
(radial scan at each azimuth) to extract the canonical arc curve.

HaloSim does not replace the Sundog atlas: it provides forward-direction
ground truth that the atlas's inverse inference can be validated against.
The two tools are complementary — HaloSim is a desktop Monte Carlo
ray-tracer for atmospheric-optics specialists; the Sundog atlas is an
interactive browser-based parametric workbench with photo-overlay
calibration and inverse inference. Use HaloSim as oracle when a
forward-rendering claim or inverse-handle choice needs independent
verification; use the Sundog atlas as the public-facing interactive
surface and the inverse-inference instrument.

This subsection (and Phase 12A/12B) cover the **validation** direction.
The opposite, **generative** direction — driving HaloSim non-interactively
to render a labelled sun-altitude-swept halo film for hero / press /
animation use — is tracked in its own document,
[`SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md`](SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md).
Same binary, opposite purpose; do not conflate the two when scheduling
work.

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
  - Page section linking to a `site/PHOTO_DATA_POLICY.md` doc that states
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

Status 2026-05-13: Phase 6 has entered the workbench with three live inverse
bindings:

- Parhelion/dagger handles update `--sun-altitude`, with the handle y-position
  now following the calibrated `--parhelic-y-offset-r22` belt rather than the
  old sun centerline.
- A parhelic-arc apex handle updates `--parhelic-curvature` directly. If the
  Phase 3 "derive from sun altitude" toggle is on, grabbing this handle
  switches curvature back to manual mode so the drag has one unambiguous bound
  parameter.
- A CZA apex handle updates `--cza-curvature`; it hides with the rendered CZA
  when `--sun-altitude` exceeds the Phase 3 visibility cutoff at 32°.
- The first constraint gate is backed by
  `scripts/check-sundog-phase6-drag.mjs`, which dispatches pointer drags
  against the live module and verifies curvature, altitude, and mirrored
  parhelion positions, plus CZA apex curvature and hide/show behavior.

Next Phase 6 handles: sun-center translation and 22° halo edge scaling once
the per-primitive bindings feel stable.

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
- **Wikipedia / SEO outreach packet** — [`docs/promo/SUNDOG_OUTREACH_PACKET.md`](promo/SUNDOG_OUTREACH_PACKET.md) ✓ landed 2026-05-12.
  - One-page math summary: the seven atlas formulas with
    Greenler / Tape / Cowley citations.
  - Claim license: every formula is cited; original contributions are
    named (integrated primitive-atlas presentation, interactive
    inverse-inference of sun altitude from a photograph, calibration
    evidence across 7 photos, drag-to-tune constraint network,
    canonical JSON pose schema).
  - Reproducibility statement: three paths (live page, phase3 tests,
    `overlay_calibrate.py` CLI on the 7-photo cohort).
  - Suggested Wikipedia edits per article (Sun dog, CZA, 22° halo, 46°
    halo, Vädersoltavlan, Halo top-level), each tied to a specific
    factual claim already in the article. All suggestions are
    external-link adds — no original-research citation requests.
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

### Phase 8 - Reproducible Pose Export *(closed 2026-05-12)*

Goal: the workbench's `Snapshot params` button writes JSON to console +
clipboard. Promote that to a deterministic export pipeline matching the
schema in [`SUNDOG_GENERATOR_SPEC.md`](SUNDOG_GENERATOR_SPEC.md):
identical pose JSON loaded back into the workbench produces a
byte-equivalent SVG.

Status: **shipped 2026-05-12** across three sub-deliverables.

**Phase 8A — Load Params button.** `parhelion-workbench.mjs` hosts an
`applyPose(params)` function and a file-picker handler bound to the
`Load params` button in `sundog.html` and `sundog-workbench.html`.
Accepts both pose-JSON schemas in the wild — kebab-case (snapshot
output) and camelCase (canonical named poses) — via `findPoseValue()`
with an explicit `CAMEL_OVERRIDES` map for the three names whose
camelCase carries a `Deg` / `R22` suffix. Honors `geometryModel` and
restores the `derive parhelic curvature from altitude` toggle.

**Phase 8B — Named-pose library.** Five atlas poses shipped to
`public/poses/`:

- [x] `canonical.json` / `canonical-halo-atlas.json` — locked defaults
  (calibrated against the Troels Nielsen DR photo, h = 25°).
- [x] `low-altitude.json` — h = 5°. Parhelia ride far from the sun
  because parhelion offset = R₂₂ / cos(h). Long sun pillar; lower
  tangent arc lights up; CZA dark.
- [x] `cza-heavy.json` — h = 22°. CZA + supralateral arc dominant;
  22° halo and parhelia subdued.
- [x] `nine-halo-eye.json` — h = 25°, every named primitive at
  performance intensity. Hero composition for `index.html`.
- [x] `forty-six-halo.json` — h = 30°. 46° halo + supralateral +
  infralateral dominant; 22° halo dimmed.
- [x] `rich-display-vocabulary.json` — pedagogical pose: every named
  primitive visible at non-zero intensity simultaneously. Glossary
  in geometry form.

**Phase 8C — Pose-pinning CLI.** `scripts/render-pose.mjs`:

- Validates pose JSON against the canonical schema (catches typos,
  out-of-range values, unknown `geometryModel`).
- Emits two artifacts per pose into `dist-poses/`:
  - `<name>.html` — self-contained pinned render with the workbench
    skeleton, embedded CSS variables, and a tiny module script that
    calls `applyParhelionGeometry` on load. Pose CSS is declared
    *after* the skeleton stylesheet so pose values win. No sliders,
    no drag — a read-only "pin".
  - `<name>.snapshot.json` — canonicalized pose with `_meta` dropped
    and keys in canonical schema order, suitable as gate input.
- `npm run poses:pin` — emit all poses. `npm run poses:pin:one <file>`
  — single-pose mode.
- Headless PNG rendering is deferred to a follow-up that will add
  puppeteer; the geometry module only touches `querySelector`,
  `getAttribute`, `setAttribute`, `dataset`, so a thin DOM shim is
  also viable.

Gate: **byte-equal round-trip verified**. For every atlas pose,
`pose.json` → `pose.snapshot.json` and re-feeding the snapshot through
the CLI produces a bit-identical output file. Verified 2026-05-12
across `canonical-halo-atlas`, `low-altitude`, `cza-heavy`,
`nine-halo-eye`, `forty-six-halo`, and `rich-display-vocabulary`.

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

> **Read [`MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md) before kickoff.**
> Mesa's v3.1-v3.8 results land five load-bearing constraints on this phase:
> (1) track residuals *per-primitive*, not by total fit (variance ≠
> mechanism); (2) measure each inversion route separately — parhelion
> offset, CZA apex, tangent-arc curvature, supralateral position —
> because forward and inverse are not symmetric; (3) pre-register
> quantitative `do not promote` thresholds before overlays run, and treat
> a clean negative as a deliverable; (4) reject any linear arc-importance
> attribution scheme as a Phase 11 metric — partial delivery of the right
> basis is mesa's documented failure mode for field-shaped objects; (5)
> keep visual salience, route-local residual/sensitivity, and full-overlay
> fit separate, because v3.8 shows variance rank, single-component
> sensitivity, and full mechanism do not collapse into one score.

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
- A per-inversion-route residual table in
  `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`, measured separately for
  parhelion offset, CZA apex, tangent-arc curvature, and supralateral
  position where each route is visible/applicable. This is a Phase 10
  deliverable, not a separate standalone note.
- Pre-registered quantitative `do not promote` thresholds, written into
  `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md` before the next overlay
  run. This is a Phase 10 blocker: if the thresholds are not written first,
  the overlay run can be exploratory only and cannot promote primitives.
- Updated defaults or pose presets only if the same morphology appears
  consistently across at least two of the three images.
- A short "do not promote" list naming any arc that looked plausible in one
  image but failed the richer pass. A candidate primitive is not promoted
  into the default/logo vocabulary if its normalized feature residual is
  `>= 0.06 * R22` (or `>= 12 px` at the current anchor scale) on at least two
  eligible calibration photos, if it has only one eligible supporting photo,
  or if its inversion-route residual is `>= 0.04 * R22` (or `>= 8 px`) on at
  least two eligible photos. Photos where a feature is genuinely hidden,
  cropped, or outside its sun-altitude validity window are excluded from the
  denominator and recorded as `not applicable`.

Gate: the optional vocabulary layers are stable enough to give to a design
team as shape language, without letting them rewrite the calibrated core.

**Status 2026-05-13: Phase 10 gate is *closed*; single-handle verdict
landed.** All anchor-capture work (Tasks #52, #53, #54, #55) shipped; the
detailed verdict and per-route reasoning live in
[`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md)
under "Phase 10 Promotion Verdict" and the new "Single-handle closeout"
subsection.

> **Post-audit state, 2026-05-14:** the synthetic optical audit
> ([`docs/calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md))
> produced three load-bearing findings, and the required technical
> response passes have now landed: A1a/A1b repaired the CZA formula,
> A2/A3 reclassified p27 and moved CZA to a coverage-gate failure, C1
> removed p7 from upper-tangent eligibility, B1/B2 retired the
> "every eligible photo" parhelion framing in favor of the strict
> p2/p7/p13 subset, **Pass C2 (2026-05-14) built and ran a wing-radial
> Lab b\* ridge detector with 22°-halo-radial-profile subtraction
> (not-recovered); Pass C4 (2026-05-14) built and ran the wing-slope
> luminance-gradient curvature detector Persona 1 §5 explicitly named
> (also not-recovered); Pass C5 (2026-05-14) tested the
> manual-sample-selection alternative — 5 hand-anchored points on p2
> fit a clean circle with R\_uta\_obs / R22 = 0.824 and RMS = 1.23 px,
> but p13 and p27 yield no usable anchoring; and Pass C6 (2026-05-14)
> built and ran a matched-filter against a parameterized arc model on
> halo-subtracted b\* and FALSIFIED the C5→matched-filter natural
> extension — correlation is negative across the entire R\_uta scan on
> p2.** The re-audit gate is recorded in
> [`docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md);
> the Post-C2, Post-C4, Post-C5, and Post-C6 addenda to that memo
> carry the detector outcomes. The tangent route is now in **C5↔C6
> substrate tension**: either the gestalt signal C5 picked up lives in
> a different substrate than halo-subtracted b\*, or C5's tight fit is
> hand-anchoring symmetry bias. Recommended verify gate:
> specialist re-anchoring of p2. The remaining tangent-route open
> questions narrow to "matched-filter on alternative substrates
> (absolute b\*, L\* magnitude, chromaticity magnitude — untested),
> polarization filtering, or new calibration photos for coverage
> expansion" — all Phase 10 backlog. Route verdict for tangent-arc
> curvature → h is **partially recovered on p2 under manual sample
> selection; coverage gate fails; C5↔C6 substrate tension flagged**;
> visibility-based upper-tangent promotion as logo / animation
> vocabulary is unchanged.
>
> **Post-C7 update, 2026-05-14:** C7 supersedes the C5-positive half of
> that tangent wording. The canonical literature handle is opening angle /
> arc extent, not the project-original circle-fit curvature handle. Against
> HaloSim ray-tracing at h = 18.6°, C5's circle fit places wing points
> progressively too far radially, so the tangent route remains unpromoted
> under both handles. The useful lesson is now methodological: C5 was a
> tempting manual-anchor positive that failed both natural-extension and
> oracle checks.

**Headline:** of the four candidate `signature → h` inversion routes,
**one is promoted into calibrated core**, and **the other three fail at
three distinct layers of the measurement stack**:

| route | gate outcome | failure layer |
| --- | --- | --- |
| Parhelion offset → h | **promoted (post-audit hedged; Pass B2 2026-05-14)** | passes residual gate at ~0 px on **three photos (p2 h = 18.6°, p7 h = 59.4°, p13 h = 6.83°)** with both unambiguous bilateral peaks and an independently fittable 22° halo — *per audit memo §4.1 / §6, replacing the pre-audit "every eligible photo" framing*. Five additional low-h anchored photos have `sec(h) − 1` below 2 % of R22 and contribute as informational evidence only (anchor-noise-bounded); three of those (p20, p25, p26) have parhelion-derived R22 (tautological); p27 stipulates `offset := R22` explicitly; p26 right side flagged geometrically invalid (`R22 / offset = 1.003 > 1`). See Pass B1 eligibility sub-table and Pass B2 verdict in [`RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md). |
| CZA apex → h | **fails** *(reverdict Pass A3 2026-05-13)* | **coverage gate** *(was: residual gate)* — Pass A1b's literature formula (`scripts/cza_formula.py`) collapsed p2's residual from −19.3 px to **+1.3 px** (well below the 8 px threshold), and Pass A2 re-classified p27's chromatic arc as 46° halo top / supralateral merger (not CZA). The remaining eligibility set is p2 alone: p7 is past the h = 32.2° disappearance threshold, and every low-h photo (p13, p20, p22, p25, p26, p30) has the literature CZA apex predicted off-frame above the top of the photo. **One in-window photo with a sub-px residual; the route fails the pre-registered "fewer than two eligible photos" rule.** Reopening coverage requires new anchors in 5° < h < 32°. |
| Supralateral → h | **fails** *(reverdict Pass A3 2026-05-13)* | **coverage gate + structural-discrimination rider** — Pass A2 added p27 to the candidate set (apex measured at 41.89° above sun); p2 is route-eligible but its supralateral apex is unmeasured. On a strict reading the coverage gate still binds (1 measured photo); on a permissive reading the two-photo threshold is at the edge. **Either reading is dominated by audit memo §2 item 12's structural finding:** supralateral angular distance from sun varies only ~0.5° across h = 0–22°, an order of magnitude less than parhelion-offset. The predicted h-spread between p2 and p27 is ~0.3° = ~2.5 px at p2's R22, **below the typical 5–10 px visual-edge measurement noise on a chromatic-broadened halo arc**. The route would not be a useful inverse handle even with perfect coverage. |
| Tangent-arc curvature → h | **partially recovered on p2 under manual sample selection; coverage gate fails; C5↔C6 substrate tension** *(Pass C1 + C2 + C4 + C5 + C6 landed 2026-05-14)* | **hybrid coverage + detection-tooling failure with verify-gate flagged.** Pass C1 dropped p7. Four automated detector families on the post-C1 sampled set return not-recovered: column-peak intensity (original); wing-radial Lab b\* with halo-radial subtraction (`scripts/tangent_detector.py`, Pass C2); wing-slope luminance-gradient curvature (`scripts/tangent_curvature.py`, Pass C4); **matched-filter on halo-subtracted b\* against a parameterized arc model (`scripts/tangent_matched_filter.py`, Pass C6 — the natural follow-up the C5 receipt named)**. **Pass C5 (manual sample selection)** recovered the route on p2 (R\_uta\_obs / R22 = 0.824, RMS = 1.23 px) with a methodology hedge (possible symmetry bias). **Pass C6 falsified the C5→matched-filter natural extension**: on p2, the matched-filter correlation is negative across the entire R\_uta scan (peak −0.117 at R\_uta = 263, outside tolerance) — even at R\_uta values consistent with C5's manual fit. The route is now in **C5↔C6 substrate tension**: either the gestalt signal C5 found is in a different substrate (not halo-subtracted b\*), or C5's tight fit is hand-anchoring bias. Recommended verify gate before further propagation: specialist re-anchoring of p2. Coverage gate fails (1 / 3 photos). Remaining open paths: matched-filter on alternative substrates (absolute b\*, L\* magnitude, chromaticity magnitude — untested); polarization filtering; new calibration photos. Visibility-based upper-tangent promotion as logo / animation vocabulary is unchanged. |

> **Post-C7 supersession, 2026-05-14:** read the tangent row above as the
> C6 state. Pass C7 reformulated the tangent test under the canonical
> literature handle (opening angle / arc extent; Tape 1994 ch. 6) and
> checked it against HaloSim ray-tracing. That pass found the C5 circle fit
> geometrically inconsistent with the canonical upper-tangent curve at
> h = 18.6°: C5's wing points sit 1.2-5.6° too far radially as azimuth
> increases. The current tangent verdict is therefore **not promoted under
> either the project-original circle-fit handle or the canonical
> opening-angle handle**. C5 is best treated as likely symmetry-bias
> artifact unless an independent specialist re-anchors p2 differently. See
> [`docs/calibration/PASS_C7_OUTPUT.txt`](calibration/PASS_C7_OUTPUT.txt)
> and the Phase 12B / HaloSim-oracle sections below.

Each non-parhelion route fails for a structurally different reason:
**(Pass A3 update 2026-05-13)** CZA fails the coverage gate (only p2 is
in-window with an independent CZA-apex measurement; the rest of the set
is either past the h = 32.2° disappearance threshold or has the
literature-predicted apex off-frame above the top of the photo); the
p2 residual is excellent (+1.3 px under the literature formula).
Supralateral fails the coverage gate too, but with a structural-
discrimination rider: even at the two-photo threshold the angular
spread is ~0.3° (below the visual-edge measurement noise), so the
route would not be a useful inverse handle even with perfect coverage.
Tangent-arc curvature first failed under column-peak detection, then under
C2/C4/C6 automated detector variants, and finally under C7's canonical
opening-angle / HaloSim check. The caveat is no longer "detector not built";
it is that one hand-selected p2 circle fit existed but does not survive the
natural matched-filter or canonical-curve tests.
The taxonomy still matters: a coverage-gate failure on physics grounds
(CZA: route can't be in-frame at low h) is not the same kind of
negative as a coverage-gate failure on discrimination grounds
(supralateral: route can be measured but the signal is too small) is
not the same kind of negative as a detection / anchoring failure with an
oracle check (tangent: manual sample selection found a tempting fit, but
HaloSim plus the canonical opening-angle handle rejected it).

Visibility-based promotions remain final from the earlier 2026-05-13
pass: the upper tangent arc is promoted as logo / animation vocabulary;
the CZA primitive is promoted as conditional core (rendered only when
`h < 32.2°`); lower tangent + infralateral arcs are held as optional
vocabulary compatible with named-pose examples (`low-altitude.json`,
`forty-six-halo.json`). Named poses are presentation surfaces, not
additional calibration evidence.

The concrete do-not-promote list — Suncave Parry, Parry supralateral,
supralateral-as-inversion-route, tangent-curvature-under-column-peak,
CZA-apex-inversion-route, lower-tangent-as-core, and any linear
arc-importance attribution metric — is in the same notes file.

The Phase 10 closeout reads cleanly against the gravity-claim language:
the atlas is **rich in forward generation** (`h → all primitives`) but
supports **one image-recoverable inverse handle** (parhelion offset),
not a redundant four. That is the same forward-rich / inverse-narrow
asymmetric field-shape pattern the mesa side observed in-vitro — now
observed in-the-wild at numerical resolution.

Phase 11 proceeds against the visibility-promoted vocabulary now. Three
open questions are filed but not blocking:

1. **CZA coverage expansion.** *(Reframed Pass A3 2026-05-13; was
   "Lens-optics test for CZA-apex direction inconsistency", which
   assumed the pre-audit residual-gate framing that A1b retracted.)*
   The post-A3 CZA verdict is coverage-gate-bound: only p2 is in-window
   with an independent residual (excellent at +1.3 px under the
   literature formula). Reopening coverage requires new anchored photos
   in 5° < h < 32° where the literature CZA is on-frame. Would a
   controlled-optics calibration set in that altitude band reopen the
   route? Mesa #4-style falsification surface, but the question shape
   has shifted from "is the bias a stable systematic?" to "does the
   route hold up at a second in-window anchor?"
2. **Tangent specialist re-anchoring / new photos.** Pass C7 moves the
   open question away from "which detector should we build?" and toward
   "does any independent specialist anchor p2 in a way that agrees with
   the canonical upper-tangent curve?" On the current evidence, the route
   remains unpromoted under both the project-original circle-fit handle and
   the literature-standard opening-angle handle. Reopening it requires
   either independent specialist re-anchoring, a stronger tangent photo, or
   an alternative substrate that passes a pre-registered detector gate.
3. **Parhelic-belt-y replication.** Does p13's +10.4 px belt-y residual
   replicate across multiple low-h photos, or is it photo-specific?

The expansion triage in
[`docs/calibration/PHASE10_EXPANSION_TRIAGE.md`](calibration/PHASE10_EXPANSION_TRIAGE.md)
records the p19/p20 cropped-CZA demotions and the p27 promotion to
second eligible CZA measurement. The Phase 10 closeout also feeds the
downstream perception roadmap (`SUNDOG_V_PERCEPTION.md`) with
appropriately weakened language: CZA-apex does not supply a stable
systematic bias, only a route-reliability asymmetry.

### Phase 11 - Characterized Logo & Animation Toolkit

Goal: after Phase 10 overlay tuning, update the logo documentation and hand
the graphic design team a small, bounded toolset for a characterized Sundog
logo/animation family. "Characterized" means the logo can act like a living
mark — idle, blink/reveal, shimmer, label-callout — while preserving the
calibrated parhelion grammar.

Deliverables:

- `docs/site/ICON_ASSETS.md` expanded into a design brief: source SVG anatomy,
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
  so logo variants inherit only the visibility-promoted vocabulary and keep
  failed / coverage-blocked primitives out of default marks.

Gate: the design team can reproduce the characterized Sundog mark from a
small documented toolset, and the exported assets preserve the calibrated
22° halo / parhelion / conditional-CZA grammar at both icon and motion sizes.

### Phase 12 - HaloSim-Aligned Vocabulary + Ground-Truth Discipline

*Scheduled 2026-05-14 from the HaloSim gap-check (see "Crystal orientation
vocabulary" and "HaloSim as ground-truth oracle" subsections under
Atmospheric Optics References). Brings the project-internal tool integration
into the roadmap as three bounded sub-phases. Pass C7
([`docs/calibration/PASS_C7_OUTPUT.txt`](calibration/PASS_C7_OUTPUT.txt))
is the worked precedent for the methodological half.*

Goal: formalize HaloSim as a project-internal tool — canonize the
crystal-orientation vocabulary the atlas inherits implicitly, expose
tilt-dispersion as a math-bindable physical parameter alongside the existing
visual sliders, and codify the HaloSim ground-truth check as standard
discipline for any new forward-rendering or inverse-handle claim.

#### Phase 12A — HaloSim ground-truth validation gate *(methodological)*

Goal: codify the discipline Pass C7 established. Any new forward-rendering
claim, new inverse-handle proposal, or atmospheric-optics measurement-
precision hedge should be cross-checked against a HaloSim Monte Carlo
reference render under matched crystal-orientation + tilt-dispersion
assumptions.

Deliverables:

- A short "HaloSim validation procedure" document (or addition to
  [`SUNDOG_OVERLAY_PROTOCOL.md`](SUNDOG_OVERLAY_PROTOCOL.md)) covering: (i)
  acquiring a HaloSim render at the target `h`; (ii) locking the
  pixel-to-degree scale via the visible 22° halo radius (Camera View
  auto-zooms by crystal-block configuration; re-lock per render); (iii) 2D
  arc-locus search (radial scan at each azimuth) to extract the canonical
  curve.
- A standardized naming convention for HaloSim reference renders under
  `docs/calibration/halosim_outputs/` (e.g.
  `halosim_<feature>_p{NN}_h{H.H}_<config>_<rays>mr.bmp`).
- The existing `scripts/tangent_halosim_*.py` measurement scripts elevated
  to reusable project infrastructure (with a short script-level README).

Gate: a developer adding a new arc-shape claim or inverse-handle proposal
can reference the procedure doc and produce a HaloSim-validated cross-check
without re-deriving Pass C7's method from scratch. Pass C7's receipt is
the worked example the procedure doc cites.

Effort estimate: 1–2 hours, writing-only.

**Status: LANDED 2026-05-14.** Procedure document at
[`docs/calibration/HALOSIM_VALIDATION_PROTOCOL.md`](calibration/HALOSIM_VALIDATION_PROTOCOL.md):
when-to-use criteria, 5-step procedure (configure → render → convert +
scale-lock → extract feature locus → compare to claim), the
render-naming convention, the reusable-scripts inventory, six documented
gotchas (auto-zoom per crystal-block config, sun-detection ambiguity,
Monte Carlo noise, concurrent-feature isolation, perspective projection,
AGU paywall), and Pass C7 as the worked example. Gate met: the doc is
self-contained enough that a future inverse-handle test can run without
re-deriving the method.

#### Phase 12B — Tilt-dispersion as math-bindable parameter *(engineering)*

Goal: surface the canonical physical parameter underneath several existing
visual sliders. HaloSim parameterizes orientation populations by tilt
dispersion (stddev in degrees); the atlas's `--parhelic-curvature`,
`--cza-curvature`, `--cza-secondary-offset`, and tangent-arc width are all
implicitly absorbing tilt-dispersion effects. Phase 12B exposes the physical
parameter without removing the existing visual sliders (advanced-controls
tier).

Deliverables:

- **Parameter Taxonomy extension** (Phase 0 retroactive update): add
  `--column-tilt-disp-deg` and `--plate-tilt-disp-deg` as math-derived
  parameters with canonical default values (0.1° per HaloSim's
  `Horiz column .1 deg disp.xng`; matches the project's existing audit
  campaign's tilt assumption).
- **Math bindings** (Phase 3 extension): `tangentArcOpening(h,
  column_tilt_disp_deg)`, `plateOrientationShape(h, plate_tilt_disp_deg)`.
  Bindings reference HaloSim-validated tabulated arc-shape responses at
  canonical tilt-dispersion values (0.05°, 0.1°, 0.5°, 1°, 2°).
- **Workbench UI**: tilt-dispersion controls in the advanced-controls rail.
  Existing visual sliders kept as primary controls; advanced-controls
  expose the physical knobs. A "physics-anchored mode" toggle binds the
  visual sliders to the tilt-dispersion outputs.
- **HaloSim reference render set** under `docs/calibration/halosim_outputs/`
  capturing canonical tangent-arc and CZA shapes at several
  (h, tilt-dispersion) cells for cross-validation.

Gate: the workbench's tangent-arc and CZA shape responds in a physics-
anchored way to a labeled tilt-dispersion control, validated against
HaloSim ground truth at canonical dispersion values. The audit campaign's
"below atlas measurement precision" hedge can be quantified rather than
asserted.

Effort estimate: 0.5–1 day.

**Status: INCREMENT-1 LANDED 2026-05-14.** The circular tangent-arc
primitive (`R_uta = 200` hardcode that Pass C7 falsified) is replaced by
the parametric canonical curl in
[`public/js/parhelion-geometry.mjs`](../public/js/parhelion-geometry.mjs):

- `tangentArcLocus(altitudeDeg, columnTiltDispDeg)` returns the
  `ρ(ψ) = 22 + A(h)·|ψ|^1.5` curve; `tangentArcOpeningCoefficient(h)`
  uses the Tape AH Ch 6 / Pass C7 circumscribed-transition boundary at
  h ≈ 29° (`A(h) = 0.031·(29−18.6)/(29−h)`). Returns `null` for
  h ≥ 29° (circumscribed-halo regime, no separate tangent arc) —
  mirrors the `czaVisibleAtAltitude` guard pattern.
- `applyUpperTangentArc` / `applyLowerTangentArc` rewritten to render
  the parametric locus; `--column-tilt-disp-deg` CSS knob wired
  (canonical default 0.1°, the Pass C7 reference cell).
- Added to the `phase3` test export (`tangentArcLocus`,
  `tangentArcOpeningCoefficient`, `tangentArcCircumscribedAltitude`).
- Verified: reproduces Pass C7 within ~0.2° at the h=18.6° / 0.1°-tilt
  cell; correct h=29° circumscribed boundary; `npm run sundog:check`
  passes (page contract + Phase 6 drag constraints — tangent arcs are
  default-OFF so no canonical-pose regression).

**Single-cell-calibrated.** Only the h=18.6° / 0.1°-tilt cell is
HaloSim-validated. **Increment-2 (spec'd 2026-05-14, pending renders)**
= validate / refine the h-dependence + characterize tilt-dispersion
broadening + add the advanced-controls UI slider for
`--column-tilt-disp-deg`. The render-grid spec is written:
[`HALOSIM_VALIDATION_PROTOCOL.md`](calibration/HALOSIM_VALIDATION_PROTOCOL.md)
§"Tilt-dispersion sensitivity sweep" — an L-shaped ~11-render grid
(7-cell h-sweep at 0.1° tilt + 4-cell tilt-sweep at h=18.6°), with
per-cell HaloSim config, what-gets-measured, and 6 acceptance criteria.
User effort ≈ 1.5–2 h of HaloSim operation; agent-side measurement +
model refinement ≈ half-day once renders are on disk.

**Adjacent fix surfaced 2026-05-14:** running `npm run sundog:check`
during this work caught that the wave-2 W15 patch (schema.org author →
named human "Jeffery Hughes Jr." with Stellar Aqua LLC affiliation, for
WP:SELFPUB compliance) had diverged from the
`scripts/check-sundog-page.mjs` author assertion (still expecting
`author.name === "Stellar Aqua LLC"`). The checker was updated to
validate the W15-intended structure. The W15 patch and its test
contract are now reconciled.

#### Phase 12C — Full-atlas vocabulary completeness *(writing)*

Goal: extend Phase 4's §5 "Full atlas" section in `sundog.html` to name
every halo family in the canonical literature vocabulary, with a one-
paragraph description each, even when the atlas doesn't render the family
as a separate primitive. Treats vocabulary completeness as an educational
deliverable orthogonal to rendering scope.

Deliverables: §5 prose expansion in `sundog.html` covering, at minimum:

- Parry-family arcs (suncave Parry, sunvex Parry, Parry supralateral) —
  citing Tape AH Ch 3 and HaloSim's `Parry arcs.sim` + `Parry 1820
  display.sim`. Already in vocabulary, not yet separately rendered.
- Pyramidal / odd-radius halos (9°, 18°, 20°, 22.9°, 23.8°, 24°, 35°) —
  citing Tape AH Ch 10 + AH-SAX Ch 11 (both on disk under
  `docs/calibration/AH-CH10/` and `docs/calibration/AH-SAX-CH11/`) and
  HaloSim's `Pyramidal *d halo.sim` family. Not modeled.
- Lowitz arcs — citing HaloSim's `Lowitz arcs.sim` + the St Petersburg
  1790 historical display. Not modeled.
- Antisolar features (anthelion, anthelic arcs, paranthelia) — citing
  HaloSim's `Anthelic Point display.sim`. Not modeled.
- Sub-horizon halos and circumhorizon arc — flagged as audience-mismatch
  for the public explainer (aircraft / high-noon-summer specific).

Each entry follows the format: name, formation mechanism (one sentence),
where to learn more (Tape chapter + HaloSim sim file + Cowley atoptics
page), Sundog status (rendered / named-only / not-modeled).

Gate: an attentive reader of `sundog.html` §5 encounters the full halo
vocabulary in one place, with citations, without needing to chase external
references for canonical names. Wikipedia editors reviewing the page can
verify that Sundog's primitive set is a documented subset of the canonical
literature vocabulary.

Effort estimate: 2–4 hours, writing-only (no rendering work).

**Status: LANDED 2026-05-14.** `sundog.html` §5 now carries the rendered
12-primitive grid (lead reframed to state it explicitly as the rendered
subset) followed by a "Named in the literature, beyond the rendered set"
block. Five extended-vocabulary cards land in the required format —
name, one-sentence formation mechanism, where-to-read-more (Tape chapter ·
HaloSim `.sim` file · Cowley atoptics), and a Sundog rendering-status
badge: **Parry-family arcs** (suncave Parry & Parry supralateral rendered
as optional vocabulary overlays, sunvex Parry named-only; Tape AH ch. 3,
`Parry arcs.sim` / `Parry 1820 display.sim`), **Pyramidal / odd-radius
halos** (9°/18°/20°/22.9°/23.8°/24°/35°; Tape AH ch. 10 + AH-SAX ch. 11
on-disk paths, `Pyramidal *d halo.sim`; not modeled), **Lowitz arcs**
(`Lowitz arcs.sim` + St Petersburg 1790; not modeled), **Antisolar
features** (anthelion / anthelic arcs / paranthelia; `Anthelic Point
display.sim`; not modeled), and **Sub-horizon halos & circumhorizon
arc** (not modeled — audience mismatch). Citations are inline on the page
(atoptics linked at root only, consistent with §7; no guessed deep URLs;
on-disk Tape-chapter paths cited as text, not directory links).
`npm run sundog:check` passes (page contract + Phase 6 drag — additive
content reusing the proven `.atlas-card` classes, no canonical-pose or
JSON-LD regression). Gate met: the full canonical halo vocabulary is in
one place with citations, and the rendered set is explicitly labelled as
a documented subset.

### Phase 13 - Atlas Claim Governance + Ask Sundog Coupling

*Scheduled 2026-05-14 from the `sundog.html` vocabulary expansion and the
`SUNDOG_V_CHAT.md` §16.4 integrity gap.*

Goal: make the public atlas and Ask Sundog move together. Whenever
`sundog.html`, the outreach packet, or this roadmap adds a halo name,
sharpens a rendering claim, retracts a route, or promotes a proof tier, the
chat claim map must know the difference between:

- **rendered primitive** — visible in the atlas geometry layer;
- **annotation / optional vocabulary** — drawn or labelled, but not promoted
  as calibrated core;
- **named-only literature coverage** — present in the page as a canonical
  atmospheric-optics name, not rendered or detected by Sundog;
- **not modeled** — acknowledged as outside the current geometry layer;
- **speculative / proof-program candidate** — analytically or
  computationally interesting, but not a public result.

Deliverables:

- A standing "claim phrase inventory" step for any `sundog.html` or
  geometry-roadmap edit that touches halo vocabulary.
- `chat/claim_map.json` route coverage for atlas vocabulary claims. Current
  route: `halo_atlas_vocabulary_status`, with boundaries forbidding Ask
  Sundog from saying the atlas renders, detects, or fully describes every
  named halo family.
- Public data rebuild after claim-map changes:
  `npm run chat:index` / `npm run build` as appropriate.
- Chat gate before deploy for geometry-surface changes:
  `chat:eval:static`, `chat:eval:phase3`, `chat:eval:phase3:adversarial`,
  `chat:eval:phase3:differential`, and `chat:eval:phase4` unless the change
  is provably copy-only outside the indexed chat corpus.
- A "no complete-geometry claim" guardrail in public copy: the atlas can say
  it is a literature-grounded halo vocabulary / workbench; it cannot say it
  has accounted for all sundog geometry until Phase 14/15 evidence supports
  that claim.

Gate: a visitor can ask Ask Sundog "does the atlas model all halos?" or
"what halos are not rendered?" and receive the same boundary that
`sundog.html` uses: rendered subset, named-only literature coverage,
not-modeled families, and future proof-program candidates are distinct.

Status 2026-05-14: **initial coupling landed on the chat side.**
`chat/claim_map.json` has `halo_atlas_vocabulary_status`; `SUNDOG_V_CHAT.md`
§16.4/§16.5 record the update discipline. This phase remains open as a
standing deploy gate for every future geometry-vocabulary change.

### Phase 14 - Complete Halo Phenomena Accounting Matrix

Goal: turn "the full atlas" from prose into an auditable accounting system.
The page already names the first extended families; this phase builds the
project's canonical inventory of atmospheric-optics phenomena and records
what Sundog does, does not, and might later do with each one.

Deliverables:

- New canonical matrix:
  `docs/calibration/HALO_PHENOMENA_ACCOUNTING.md` plus optional machine-readable
  mirror `public/data/halo-phenomena-status.json`.
- Public dictionary surface:
  `legend.html`, a lightweight halo legend that can absorb named-only,
  not-modeled, and speculative vocabulary without burying `sundog.html`.
- One row per named phenomenon / family, with at least:
  canonical name, aliases, literature source (Greenler / Tape / Cowley /
  HaloSim `.sim`), crystal-orientation family, geometric generator,
  observer-geometry requirements, Sundog rendering status, HaloSim status,
  photo-overlay status, inverse-handle status, chat route / public boundary,
  and next proof step.
- Status vocabulary shared with Phase 13:
  `rendered-core`, `rendered-optional`, `named-only`, `not-modeled`,
  `halosim-reproducible`, `analytic-candidate`, `speculative`, `observed`,
  `rejected`.
- Coverage for the families already introduced in Phase 12C:
  Parry-family arcs, pyramidal / odd-radius halos, Lowitz arcs, antisolar
  features, sub-horizon halos, circumhorizon arc, plus the current rendered
  12-primitive subset.
- A "what counts as accounted for?" definition:
  a phenomenon is not accounted for just because its name appears on the
  page. It is accounted for only when the matrix identifies source,
  geometry, render/simulation status, evidence status, and claim boundary.

Gate: any public or chat answer about "all sundog geometry" can point to
the matrix and distinguish complete, partial, named-only, not-modeled, and
speculative entries without improvising.

Status 2026-05-14 (14E addendum 2026-05-15): **14A landed · 14D source pass
landed · 14E receipt pass landed (11/11) · 14F partially
routed.** The ledger at `docs/calibration/HALO_PHENOMENA_ACCOUNTING.md` is
reworked into three lenses matching the Phase 14 directive — **§A Anchor**
(proven core; generator = the project's own validated closed form in
`public/js/parhelion-geometry.mjs` + a receipt), **§B Install** (the
canonical pre-existing generator for every family, read straight from the
on-disk Tape chapter scans `AH-CH06`/`AH-CH10`/`AH-SAX-CH11` and the HaloSim
`.sim` recipe descriptions), and **§C Extrapolate** (named-only / not-modeled
/ speculative rows carry a predicted locus, an inverse-handle candidate, a
falsification line, and a P0–P5 Phase 15 tier). This installs the geometry
the project had wrongly believed it needed to invent: the `parhelion-geometry.mjs`
closed forms (parhelion `R22/cos h`, CZA `h≤32°`, the Pass-C7 `tangentArcLocus`
+ its 29° circumscribed-merge guard) are independently confirmed by Tape
AH-CH06 p62–63, and the odd-radius family's generator is the Tape AH-CH10 p6
wedge→radius table + Galle `α=180−2x` / Rational-Tangents (SAX-CH11), not a
reinvented formula. Honesty ratchet recorded: the supralateral / Parry /
infralateral / sun-pillar / parhelic-circle atlas primitives are hardcoded
placeholders (rendered ≠ anchored) routed to §B for reconciliation. **14B**
`legend.html` is a parallel workstream the owner runs separately — the ledger
is kept row-consistent with it but does not edit it; **14C** machine-readable
mirror stays deferred. **14E** landed 2026-05-15: all 11 un-receipted
`.sim`-backed rows rendered B&W @1M via HS-0 and confirmed reproducing
(`halosim-candidate` → `halosim-reproducible`, 0 not-reproducible) —
receipts at `docs/calibration/halosim_outputs/phase14e/`
(`_PHASE14E_RECEIPTS.md`), source frames byte-safe-pinned by
`scripts/halosim_pin_rays.py`; empirical finding that ~300k b&w is
unreliable (AGENTS.md + HALOSIM_VALIDATION_PROTOCOL.md ray-count guidance
corrected to ~1M b&w). The Gate is **not** closed by this revision
(ledger ↔ legend row-consistency is cross-checked against the in-parallel
`legend.html`, not closed here).

### Phase 15 - Speculative / Unphotographed Halo Proof Program

Goal: investigate unseen, rarely photographed, or speculated halo forms
without turning speculation into public proof. This is the program for
"can the math, a brute-force ray trace, the atlas, and HaloSim together
predict a halo before photographs or specialist consensus catch up?"

Proof tiers:

| tier | meaning | public boundary |
| --- | --- | --- |
| P0 named / rumored | name appears in literature, historical report, or HaloSim catalogue | catalogue only |
| P1 analytic-feasible | ray geometry or projection math says the locus is possible | speculative math candidate |
| P2 brute-force reproducible | independent ray tracer or HaloSim produces the feature under declared crystal / observer conditions | simulated candidate |
| P3 atlas-representable | the locus can be expressed in the atlas parameter grammar without ad hoc placement | model candidate |
| P4 predictive target | pre-registered conditions and photo-search / controlled-observation plan exist | prediction |
| P5 observed / specialist-confirmed | photograph, simulation, and specialist taxonomy agree | promotable evidence |

Deliverables:

- `docs/calibration/SPECULATIVE_HALO_PROOFS.md` with one proof record per
  candidate. Each record names the phenomenon, the proposed generator,
  necessary crystal population, sun-altitude / observer-geometry window,
  expected visual signature, HaloSim configuration, atlas projection,
  falsification criteria, and current proof tier.
- A small brute-force ray-trace or HaloSim-batch workflow for candidates
  where HaloSim's built-in `.sim` files are insufficient.
- Atlas + HaloSim comparison artifacts: rendered locus, ray-trace image,
  overlay in common coordinates, and a residual / disagreement table.
- Pre-registration before any "unseen halo" claim: what would count as a
  positive, what would count as a null, and which proof tier the result can
  enter.
- Chat boundary route updates for any P1-P4 candidate so Ask Sundog says
  "simulated / speculative candidate," never "discovered halo."

Gate: a speculative halo cannot move into public-facing result language
until it reaches at least P4 with pre-registration and independent
simulation receipts; P5 requires real observation or specialist-confirmed
classification. Negative results are kept as useful map entries, not buried.

Status: **seeded 2026-05-15.**
[`docs/calibration/SPECULATIVE_HALO_PROOFS.md`](calibration/SPECULATIVE_HALO_PROOFS.md)
created with the P0–P5 ladder, the gate, and per-candidate proof records.
**Pyramidal / odd-radius halos** and **circumhorizon arc** are at **P2**
(P1 analytic-feasible via the Tape AH-CH10 p6 wedge→Δ_min table + Galle
α=180−2x / Rational-Tangents for pyramidal, and the 90° side→basal-face
generator + Tape AH-CH06 p65 for CHA — **plus** a 14E HaloSim reproduction
receipt under declared crystal/observer conditions:
`docs/calibration/halosim_outputs/phase14e/`). Lowitz / antisolar /
subhorizon are P0 catalogue stubs with promotion paths. P3 is explicitly
**not** claimed (neither locus is in the `parhelion-geometry.mjs` atlas
grammar). The gate is **not** reached — nothing here is public-facing
result language; P4 + pre-registration is required first.

Follow-up #1 (quantitative pyramidal residual) was pursued through
**all three levers, 2026-05-15 → mechanically successful but
quantitatively NEGATIVE; pyramidal stays P2**. (i) 14E 1M receipt:
**0** clean rings. (ii) Dedicated **6M** re-render: **1** marginal ring
(verified not a centring artifact). (iii) **Ray-filter-isolation
campaign**: built `scripts/halosim_filter_frames.py` (byte-safe edit of
the `.sim` ray-filter Entrance/Exit fields — a clean text-edit
extension of the proven HS-0 loop), generated **8 wedge frames**
(entry→exit faces per the Tape AH-CH10 table) and ran 8 HS-0 renders at
4M (`Startup.sim` backed up + restored byte-identical; receipts in
`docs/calibration/halosim_outputs/phase15_pyrfilter/`). Findings: the
face-pair filters do **not** single-isolate one ring (6-fold crystal
symmetry) but **markedly crispened** the family; under the rigorous
detrend+3σ test with a background-agnostic contrast signal and an
edge-excluded ring-SNR centre (the new `--filtered` mode of
`pyramidal_ring_residual.py`) the crispest receipt resolves **1 strong
ring** (4.6σ) — the data-quality progression **1M=0 → 6M=1 marginal →
4M-filtered=1 strong** is real, but still **< the ≥3 azimuthally-
separable rings** a predicted-vs-measured table + 22°/46° linearity
check require (verified across 4 independent centring methods). The
script **refused to fabricate** a table every time — negatives kept as
precisely-characterised map entries per the gate. **(iv) HaloSim-native
angular-Scale method (the fourth and final lever):** HaloSim's own
`Tools → Scale` instrument (degree ruler from the sun, FIX-stamped;
`scripts/pyramidal_scale_read.py`, hard anchor gate) was applied to all
8 wedge frames re-rendered with the scale stamped
(`docs/calibration/halosim_outputs/phase15_pyrfilter/pyr_w*_scale.png`;
`Startup.sim` backed up + restored byte-identical). The scale solves
sun-centring (sun-Y locked at y≈393 by the ruler) but its stamped span
is **shorter than the ring field** (the 22°/46° anchor rings fall beyond
the calibrated ruler tip) and 6-fold symmetry still prevents per-wedge
isolation → the anchor gate **failed and the script refused to
tabulate** (no fabrication). **Disposition: accept P2 as this evidence
chain's ceiling — adopted as the standing position.** Pyramidal is
solidly P2 (P1 analytic Tape AH-CH10 table + P2 qualitative HaloSim
reproduction across 1M / 6M / 8×ray-filter / 8×scale renders); the
no-fabrication gate held at every one of the four levers. P3
(atlas-grammar representability / a clean quantitative residual) needs a
fundamentally different setup (non-symmetric single-ring isolation, or a
2-D ring-template fit on a non-split equidistant render with a full-span
scale) and is **out of scope for this chain** — closed, not a deferred
task. Other tracked follow-ups unchanged: the `chat/claim_map.json` P2
boundary refinement (gated on the Phase 13 chat:eval chain) and a
crystal-block-reduced Lowitz isolation render. This phase still depends on Phase 14's accounting matrix and Phase
12A's HaloSim validation protocol, and may draw on the cinematic sidecar's
batch-render automation when proof work needs sweeps.

#### Phase 12 deferred items (low-priority gaps, not scheduled)

The HaloSim gap-check surfaced additional candidate features that remain
proposed-only and are not part of Phase 12:

- Pyramidal halos as rendered primitives (engineering scope, future
  Phase 14/15 candidate; vocabulary entry lands here in 12C
  regardless).
- Parry-arc family rendered as separate primitives (future Phase 14/15
  candidate).
- Lowitz arcs / Kern arc / Liljequist parhelia / Wegener arcs / antisolar
  features as rendered primitives (specialist scope; defer indefinitely
  unless a specific user case appears).
- Canonical historical displays as named-pose targets (Parry 1820, St
  Petersburg 1790, Saskatoon 1970) — could fit Phase 8 named-pose library
  extension or Phase 11 logo / illustration assets.

These are recorded here so the rationale for the Phase 12 scope is
explicit: 12 covers the *infrastructure and vocabulary* moves with broad
applicability; the *individual exotic features* are downstream candidates
that depend on a clear user-need signal.

## HaloSim Cinematic Sidecar — tracked separately

The generative/cinematic HaloSim work (HS-0 … HS-7) was split out of this
roadmap on 2026-05-14 to keep the main document navigable. It is now
tracked in its own document:
**[`SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md`](SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md)**.

That sidecar is the **generative** use of HaloSim — driving it
non-interactively to render a labelled, sun-altitude-swept halo film for
hero / press / animation use. It is deliberately distinct from Phase
12A/12B above, which use HaloSim as a **validation oracle**. The sidecar
does **not** gate hero promotion and is **not** in the main validation /
claim-governance dependency chain; its outputs feed Phase 7 (press assets), Phase 11
(animation toolkit), and Phase 4 (active-reveal polish). HS-0 is a hard
automation-feasibility gate with a named atlas-only fallback — see that
document for the full HS-0…HS-7 breakdown and ratified scope.
