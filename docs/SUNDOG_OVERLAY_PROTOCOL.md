# Sundog Overlay Protocol

Working purpose:

> Demonstrate that a Sundog parameter pose, rendered through the
> deterministic geometry solver, produces a recognisable parhelion display
> when laid over a real photograph at the calibration thresholds named in
> this document.

This protocol is the **proof amplifier** of the project. The workbench's
strongest defence against prompt-art skepticism is reproducibility under
overlay: a viewer should be able to drag a generated SVG over their own
parhelion photo, align it with two anchor points, and see the geometry
land where the optics already lives.

## When to Use This Protocol

- **Mode 2 of the Sundog Generator**
  ([`SUNDOG_GENERATOR_SPEC.md`](SUNDOG_GENERATOR_SPEC.md)) is the
  programmatic version of this protocol.
- **Phase 2 calibration** of the geometry workbench
  ([`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md)) — overlay against the
  Troels Nielsen DR canonical reference photograph to lock the default
  pose.
- **Public-facing reproduction** — a reader who has a parhelion photo
  should be able to run this protocol from the browser workbench and
  produce a fit report without writing code.
- **AI-output validation** — when AI beautification (Mode 3) is invoked,
  the underlying vector geometry should still pass overlay against the
  reference photo set; AI must not muddy the geometric correctness.

## Related Documents

- [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) — the workbench roadmap.
  Phase 2 calibration targets are tracked there; this doc is the
  procedure those targets cite.
- [`SUNDOG_GENERATOR_SPEC.md`](SUNDOG_GENERATOR_SPEC.md) — the generator
  architecture. Mode 2 ("Compare to Sky") wraps this protocol behind a
  callable interface.
- `public/poses/canonical.json` — the locked canonical pose. Overlay
  fit against the canonical reference photograph is the gate for any
  default-value change.

## Reference Photograph Corpus

The protocol assumes a curated set of reference photographs whose
optical content is unambiguous and well-cited. Initial corpus:

- **`troels-nielsen-dr.jpg`** — single-sun parhelion display with two
  outward-facing daggers, a clean 22° halo, an upper tangent arc, and a
  prominent CZA at frame top. The canonical hero reference. Source:
  Troels Nielsen / DR.
- **`triple-cza-stack.jpg`** — multi-arc event with stacked CZAs and
  visible parhelic curve. Source pending citation in
  `THIRD_PARTY_REUSE.md`.
- **`docs/calibration/1.Photometeor-jeff-mod_marked_red.jpg`** —
  vocabulary reference for rich-display labels: suncave Parry, Parry
  supralateral, upper tangent, and infralateral arcs. Treat as an
  annotation source, not a pixel-residual calibration target.

Each reference photograph should ship with a small JSON sidecar:

```json
{
  "file": "troels-nielsen-dr.jpg",
  "credit": "Troels Nielsen / DR",
  "license": "...",
  "anchors": {
    "sunCenterPx": { "x": 1240, "y": 760 },
    "halo22ApproxRadiusPx": 380,
    "horizonY": 1340
  },
  "expectedFeatures": [
    "parhelion-left", "parhelion-right",
    "halo-22", "sun-pillar", "cza-primary",
    "upper-tangent", "suncave-parry",
    "parry-supralateral", "infralateral"
  ]
}
```

Anchors are reviewer-supplied; they bootstrap the overlay's coordinate
calibration and remove guesswork from the alignment phase.

## Procedure

### Step 1 — Load the photograph

The workbench overlay UI accepts a JPEG/PNG of any size. The reference
sidecar JSON loads alongside if present (drag both, or pick a corpus
entry from the dropdown).

### Step 2 — Place the sun centre

Click the photograph at the sun's centre. The overlay records the click
coordinate as `photoSunCenterPx`. This becomes the registration origin
for the SVG-to-photo transform.

### Step 3 — Calibrate the 22° halo radius

Click a second point on the photograph anywhere along the bright 22°
halo ring. The overlay measures the pixel distance from the sun centre
and uses it as `photoHalo22RadiusPx`. The SVG is then scaled so its
internal `halo-22` circle (radius 220 in SVG units) matches that pixel
radius.

Sanity gate: the calibration distance should be between 100 and 1000
pixels for typical handheld photographs. Outside this range, prompt the
user to re-click — they likely missed the halo and clicked sky.

### Step 4 — Align horizon / vertical

Optional but recommended. Click two points along the horizon (or a
known vertical reference such as a building edge). The overlay computes
the photo's roll angle relative to image axes and rotates the SVG to
match. Without this step, the parhelic-curve "grin" can be misread as
horizon tilt rather than as the optical tilt the workbench actually
encodes.

### Step 5 — Drop the SVG overlay

The system renders the current pose's SVG, scaled per Step 3 and rotated
per Step 4, and lays it over the photograph at `opacity: 0.5`. The user
can adjust opacity from 0.2 to 0.9 with a slider; 0.5 is the default
calibration view.

### Step 6 — Read the fit report

The overlay computes per-feature errors and presents them as a small
table:

| Feature | SVG position (px) | Photo position (px) | Error (px) | Threshold | Verdict |
| --- | --- | --- | --- | --- | --- |
| `halo-22` radius | (registration) | (registration) | — | — | calibrated |
| `parhelion-left` core | x_svg, y_svg | x_photo, y_photo | √Δ² | ±10 SVG-units after scale | PASS / MISS |
| `parhelion-right` core | … | … | … | ±10 SVG-units | PASS / MISS |
| `cza-apex` | … | … | … | inside eyelid envelope | PASS / MISS |
| `sun-pillar` direction | angle | angle | Δ deg | ±2° | PASS / MISS |
| `parhelic-arc` curvature | sampled | sampled | Δ relative | ±15% | PASS / MISS |

`parhelion-*` photo positions can either be reviewer-clicked (manual
mode) or extracted automatically (auto mode, future) by detecting the
brightest non-sun points along the parhelic-arc band.

### Step 7 — Resolve

The protocol returns one of three verdicts:

- **PASS** — every feature is within threshold. The pose JSON tags as
  `overlayCalibrated: true` against this reference photo. If this is the
  canonical reference, the pose can be promoted to a default.
- **PARTIAL** — some features pass, some miss. The fit report lists the
  misses; the reviewer decides whether to adjust pose parameters and
  re-run, or to flag the photograph as an outlier (e.g., a different
  optical regime than the workbench currently models).
- **FAIL** — most features miss. Either the photograph is the wrong
  phenomenon (not a parhelion), the registration was incorrect, or the
  pose is far from the photograph's optical conditions. Re-run the
  registration before adjusting pose.

## Pass / Fail Thresholds

The first-cut thresholds are intentionally permissive — calibration is
not a research-grade fit, it is a "this is recognisable as the same
phenomenon" check.

| Feature | Threshold | Rationale |
| --- | --- | --- |
| `halo-22` radius | registration anchor | calibrated by definition |
| Parhelion core position | ±10 SVG-units (≈ ±2.3% of viewport) | matches the manual-click jitter floor |
| CZA apex | inside the eyelid envelope (vertical band ±15% of CZA arc length) | CZAs vary visibly with sun altitude; envelope is generous |
| Sun pillar direction | ±2° from vertical (post-roll-correction) | pillars are perpendicular to horizon; tighter than this would penalise photo crop |
| Parhelic-arc curvature | ±15% of solver-derived value | curvature reads visually, not numerically; large deltas signal mis-registration not bad geometry |

These thresholds are reviewable. As the corpus grows and the solver
becomes more refined, tightening the thresholds is a way to ratchet the
public claim. The current values are gates, not metrics.

## What This Protocol Does NOT Do

- It does **not** validate atmospheric optics theory. The thresholds
  exist to confirm the workbench's geometry matches *recognisable*
  parhelion appearance, not to publish refraction-angle measurements.
- It does **not** auto-detect features in arbitrary photographs. The
  reviewer clicks the sun centre and the halo radius. Auto-detection is
  a future enhancement and would not change the threshold structure.
- It does **not** rate AI beautification (Mode 3) outputs. AI outputs
  pass overlay if and only if their underlying vector skeleton passes
  overlay; the beautified raster is decoupled from the verdict.
- It does **not** establish the workbench as a measurement instrument.
  The workbench is a deterministic renderer, not a sensor.

## Reproducibility Receipt

A successful overlay produces a small receipt JSON suitable for inclusion
in a writeup or social-media post:

```json
{
  "protocol": "SUNDOG_OVERLAY_PROTOCOL",
  "version": "1.0",
  "timestamp": "2026-05-08T14:32:11Z",
  "reference": "troels-nielsen-dr.jpg",
  "poseSource": "public/poses/canonical.json",
  "verdict": "PASS",
  "fitReport": {
    "parhelionLeftErrorPx": 6.4,
    "parhelionRightErrorPx": 7.1,
    "czaApexInsideEnvelope": true,
    "sunPillarAngleDeltaDeg": 0.8,
    "parhelicCurvatureDeltaPct": 4.2
  }
}
```

The receipt is the artifact a third party can audit. It cites the
reference, the pose, the protocol version, and the per-feature numbers.
A skeptic can rerun the protocol against the same reference and pose
and produce a byte-similar receipt; if they cannot, the protocol or the
pose is the bug, not the geometry claim.

## Build Order

Implementation lives in `public/js/sundog-overlay-fit.mjs` (proposed)
and gets a UI surface in the workbench:

1. Load-photo + drag-to-position registration.
2. Sun-centre click + halo-radius click for scaling.
3. Optional horizon-click pair for rotation correction.
4. SVG overlay at adjustable opacity.
5. Manual-click feature alignment (parhelion cores, CZA apex).
6. Fit report rendered in the right rail or a modal.
7. Receipt-JSON export.

Auto-detection (computer-vision feature finding) is not in v1 — it
would erode the audit story. v1 is reviewer-clicked, transparent, and
boring on purpose. Auto-detection can land in v2 once the manual flow
has set the threshold expectations.
