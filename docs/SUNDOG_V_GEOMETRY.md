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

This roadmap is one of three workbench roadmaps and depends on a fourth
document for the broader visual frame. Authority is split deliberately so
no document trampolines another's decisions.

- [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) — the site-wide
  visual frame. Owns: theme tokens (`--sd-*`), shared-stylesheet layer
  organisation, the cross-cutting **4a/4b/4c** split that places this
  workbench beside Threebody and Balance, and Migration Steps 1–6 that
  govern how page-local CSS interacts with the shared sheet. **Step 4a in
  that document defers the detailed roadmap here.** Any new theme token
  this workbench needs goes via `sundog-theme.css` and that document, not
  page-local CSS.
- [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md) and
  [`SUNDOG_V_BALANCE.md`](SUNDOG_V_BALANCE.md) — sibling application
  workbench roadmaps. Different shape (control task with operating
  envelope and pre-registered verdict), same Sundog pattern of *hidden /
  indirect / transformation / output*. The Pre-Committed Cross-Application
  Comparison Row near the bottom of this document is the parallel.
- BoxForge tools library: `C:\Users\hughe\Dev\Dungeon Gleaner Main\tools`,
  agent-accessible. `boxforge.html` and `orb-component.html` are reference
  primitives for the Phase 6 selective 3D handoff. The animation phase
  vocabulary (idle / hover / active / handoff·settle) used in Phases 4–6
  below comes from that library's three-phase animation system, kept
  consistent so phase semantics carry across projects.

## Sundog Expression *(pre-staged for the eventual APPLICATIONS.md row)*

Geometry does not map onto the indirect-signal pattern the way Balance and
Three-Body do — there is no hidden state and no controller. The Expression
block is therefore reframed as a description-and-rendering pair, kept in the
same row format so cross-application comparison stays legible:

- **Hidden description:** the angular geometry of a parhelion display —
  primary sun position, halo angles (22°, 46°), CZA visibility cutoff,
  parhelion azimuthal offset as a function of sun altitude, parhelic-circle
  curvature, ice-crystal dispersion width.
- **Indirect signal:** the visible halo display on the sky as a 2D
  projection.
- **Transformation:** parametric optical render — ice-crystal physics →
  angular geometry → SVG primitives + CSS gradient atmosphere → screen
  pixels.
- **Actionable output:** an interactive workbench that draws the recognisable
  phenomenon from the slider state, with `Snapshot params` exporting a
  reproducible JSON pose.

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
- `--cza-bell` Gaussian blur at `stdDeviation="3.2"` is **too aggressive** —
  intended bell-curve overlap on the eyelid reads as smudge rather than a
  peaked Gaussian. Phase 2 calibration target.
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
the translucent-overlay calibration technique.

Calibration items (live list):

- [x] `--parhelic-curvature = 0.66` (calibrated 2026-05-08).
- [x] `--cza-bloom` promoted to slider (2026-05-08); default lowered from
  the over-aggressive `3.2` to `1.4` pending overlay calibration.
- [ ] `--cza-bloom` overlay calibration — find the value where the two
  CZA arcs' Gaussian tails sum into a visible bell-curve at intersection
  without smudging the arc identity. Slider range `0 .. 3.2`.
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
console + clipboard. Promote that to a deterministic export pipeline:
identical JSON pose loaded back into the workbench produces a
pixel-equivalent render.

Deliverables:

- `Load params` button accepting JSON (paste or file picker).
- Pose pinning script that, given a JSON file, renders the workbench at
  that pose and exports a PNG/SVG snapshot for archival.
- A small library of named poses (`canonical.json`,
  `nine-halo-eye.json`, `low-altitude.json`) checked in under
  `public/poses/` or similar.

Gate: round-trip a snapshot → load → render → snapshot and confirm
byte-equal JSON.

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
