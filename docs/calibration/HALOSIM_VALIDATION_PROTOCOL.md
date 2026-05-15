# HaloSim validation protocol

**Phase 12A deliverable, codified 2026-05-14 from Pass C7 (W4 follow-up).**

Use HaloSim's Monte Carlo ray-tracing output as ground truth for testing
the Sundog atlas's forward-rendering claims, inverse-handle proposals, and
measurement-precision hedges. This document is the **reusable procedure**;
the worked example is at
[`PASS_C7_OUTPUT.txt`](PASS_C7_OUTPUT.txt).

---

## When to use this protocol

Run a HaloSim validation when any of the following is on the table:

- **New forward-rendering claim.** "The atlas predicts feature X at
  position (a, r) for sun altitude h." Validate against HaloSim's
  ground-truth feature position at the same h under matched
  crystal-orientation assumptions.
- **New inverse-handle proposal.** "The recovery uses circle-fit
  curvature / opening angle / [other geometric measure] as the inverse
  handle." Test whether the proposed handle traces the canonical
  literature locus in HaloSim's ray-tracing output, or whether it lands
  off the canonical curve. Pass C7 used this to falsify Pass C5's
  circle-fit on p2.
- **Atmospheric measurement-precision hedge.** "The route's
  measurement precision is below the atlas's resolution." Quantify by
  comparing HaloSim renders at different tilt dispersions (0.05°, 0.1°,
  0.5°, 1°) to find where arc shape changes exceed the atlas's
  resolution.
- **Ambiguous photo feature classification.** "Is this bright spot at
  (x, y) in photo p_n a parhelion, a circumscribed-halo brightness, a
  46° halo intersection, or something else?" HaloSim at the photo's
  sun altitude under standard crystal populations gives the canonical
  expected feature inventory.

Skip the protocol when the claim is **already supported by an
independent reference** (e.g., a specific Tape chapter, Greenler chapter,
or Cowley atoptics page with a closed-form formula). HaloSim validation
is for cases where the canonical literature parameterization is
computational rather than closed-form.

---

## What HaloSim provides

HaloSim3 (Cowley & Schroeder 2004) is a Monte Carlo ray-tracer for
atmospheric halos. The project's installation lives at
`C:\Users\hughe\HalSim361.exe` with the full asset library at
`C:\Users\hughe\`. Inventory:

| asset class | count | what it gives the protocol |
| --- | ---: | --- |
| Crystal shapes (`.xsh`) | 6 | Hexagonal prism + pyramidal + distorted-hex |
| Orientation distributions (`.xng`) | 40 | Canonical orientation vocabulary parameterized by tilt dispersion (Horiz column / plate / Parry / Lowitz / random) |
| Material (`.xmt`) | 1 | `Water-Ice.xmt` — ice n(λ) sampled 370–800 nm |
| Simulation recipes (`.sim`) | 38 | Bundled named halo displays + the project's own column-only and h=0° calibration sims |

The "Camera View" projection (`Type9` in .sim files) is a perspective
view auto-zoomed to fit the visible features of the active
crystal-block configuration. **The auto-zoom means the pixel-to-degree
scale changes when the crystal-block configuration changes** — even at
the same sun altitude. This is the protocol's first gotcha.

---

## Procedure

### Step 1 — configure the simulation

Decide what crystal populations are relevant to the claim under test.
Start from an existing `.sim` file that's close to the configuration you
want; modify only what you need to change.

**Crystal-block reduction.** If the claim is about a single feature
family (e.g., upper tangent arc), disable crystal blocks that produce
unrelated features. Pass C7 isolated the upper tangent arc by keeping
only Block 1 (Horiz column) and Block 2 (random, for the 22° halo scale
reference) active; the two plate-orientation blocks were disabled.

To disable a crystal block, either:
- Use the HaloSim GUI to clear the block's shape and orientation files,
  OR
- Pre-edit the `.sim` file to replace the block's contents with the
  `no selection` pattern from disabled blocks (project utility:
  `scripts/halosim_pre_edit.py` if one is written; current procedure is
  the manual edit demonstrated in
  [`halosim_p2_h18.6_columnonly.sim`](halosim_outputs/halosim_p2_h18.6_columnonly.sim)).

**Tilt dispersion.** Pick the tilt-dispersion variant that matches the
project's standard atmospheric assumption (`Horiz column .1 deg
disp.xng` is the project default per
[`PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md)). For
sensitivity studies, also render at 0.05°, 0.3°, 1.0°, 2.0° to bracket
the canonical value.

**Sun altitude.** Set to match the claim's target altitude. For
calibration-photo work, use the photo's measured h (e.g. p2 h = 18.6°).

**Ray count.** 4–6 million rays for clean measurements; 25k–100k for
sanity-check renders only (Monte Carlo noise at low ray count produces
asymmetric brightness patterns that confound feature detection).

### Step 2 — render and save

Render in HaloSim. Save the output as BMP into
`docs/calibration/halosim_outputs/` following the naming convention in
the [Naming convention](#naming-convention-for-renders) section below.
Both b&w and colour variants are useful: b&w for clean geometry / 2D
brightness profile work, colour for matching what real-world HaloSim
output normally looks like.

### Step 3 — convert and scale-lock

Convert BMP → PNG (the Read tool can read PNGs but not BMPs; PIL is
already the project's image library):

```python
from PIL import Image
Image.open('halosim_outputs/halosim_<feature>_p<NN>_h<H.H>_<config>_<rays>mr.bmp')\
    .save('halosim_outputs/halosim_<feature>_p<NN>_h<H.H>_<config>_<rays>mr.png',
          'PNG', optimize=True)
```

**Lock the pixel-to-degree scale via the visible 22° halo radius.** The
22° halo is centered on the sun in HaloSim's Camera View; its radius
gives the scale directly. Two approaches:

1. **Visual annotation cross-check.** Draw candidate circles at known
   radii on the image, identify which matches the visible halo by eye.
   Pass C7 used this for the h=0° calibration:
   `halosim_blacknwhite_p2_h0_4.5mr.png` with circles overlaid at
   r = 87, 203, 207, 298 px showed r = 207 matches the halo's inner
   edge.
2. **Per-render automatic identification.** If the render is column-
   orientation-only, the 22° halo from the random block is the cleanest
   ring feature. The script
   [`scripts/tangent_halosim_calibrate.py`](../../scripts/tangent_halosim_calibrate.py)
   identifies candidate ring radii via radial brightness peaks; the
   user picks the right one.

**The scale changes per crystal-block configuration.** Pass C7 found:

| render | crystal blocks | R22 in HaloSim px | 1° in HaloSim px |
| --- | --- | ---: | ---: |
| `halosim_blacknwhite_p2_h0_4.5mr.png` | all 4 (col + rand + 2 plate) | 207 | 9.4 |
| `halosim_tangent_p2_h18.6_25000mr.png` | all 4 | 207 | 9.4 |
| `halosim_tangent_p2_h18.6_columnonly_3mr.png` | column + random only | 464 | 21.1 |

The column-only render auto-zoomed when the plate blocks were disabled.
**Re-lock the scale every time the crystal-block configuration
changes.**

**Sun position.** HaloSim centers the view horizontally on the sun. The
sun's vertical position varies with h. Identify the sun by:
- Saturation density in a 21×21 box around candidate y-positions on the
  central column (`x = W // 2`). The sun core saturates densely (≥80%
  of pixels at value 250+); the upper tangent arc apex saturates only
  sparsely. Pass C7 initially misidentified the apex as the sun until
  the saturation-density check caught it.
- Visual overlay cross-check: draw the predicted halo circle at the
  candidate sun position with R = R22 and confirm it matches the visible
  halo.

### Step 4 — extract the feature locus

For arc-shape claims, sample brightness along the predicted locus to
extract the feature's actual position in HaloSim's rendering. The 2D
arc-locus search Pass C7 used:

```python
# Pseudocode — see scripts/tangent_halosim_measure.py for the working version
for az_deg in range(-50, 51):
    # For each azimuth, scan over radii to find the brightness peak
    # The peak's (radial, azimuth) traces the canonical arc curve
    radii = np.arange(20, 45, 0.3)  # in degrees from sun
    brightness_at_each_r = sample_along_radial(image, sun, az_deg, radii)
    best_r_deg = radii[argmax(brightness_at_each_r)]
    arc_locus.append((az_deg, best_r_deg))
```

The output is the empirical canonical arc curve at the rendered h.
Compare to the claim's predicted curve.

For wing-extent / opening-angle measurements, apply a brightness
threshold (10% / 30% / 50% of peak-to-background range) and find the
azimuth where brightness drops below threshold. Pass C7 reports multiple
thresholds since the C5 hand-anchor matches the 10% (faint-edge)
threshold but not the 30%/50% (bright-body) thresholds.

### Step 5 — compare to the claim

State the verdict in terms of agreement with the claim under test:

- **Within tolerance ⇒ claim confirmed under HaloSim ground truth.**
  Tolerance for arc-position claims: ±1° radial at canonical tilt
  dispersion; ±2° azimuth for opening-angle threshold-dependent
  measurements.
- **Outside tolerance ⇒ claim falsified under HaloSim ground truth.**
  Document the magnitude and direction of the discrepancy; check
  whether the discrepancy correlates with tilt dispersion (in which
  case the claim may hold under different atmospheric assumptions) or
  is structural (in which case the claim's parameterization is wrong).
- **Ambiguous ⇒ note the threshold sensitivity.** Pass C7 noted that
  C5's wings match HaloSim's canonical arc at the 10% (faint) edge but
  not at the bright-body 30%/50% edges. The verdict was falsification
  under standard tolerance; the protocol records the ambiguity for the
  reader.

---

## Naming convention for renders

Save HaloSim renders to `docs/calibration/halosim_outputs/` with names
of the form:

```
halosim_<feature>_p<NN>_h<H.H>_<config>_<rays>mr.bmp
halosim_<feature>_p<NN>_h<H.H>_<config>_<rays>mr.png   (after conversion)
```

Where:
- `<feature>` — short descriptor (e.g., `tangent`, `parhelion`, `cza`,
  `pyramidal`, `parry`). Use `blacknwhite` prefix for b&w renders, e.g.
  `halosim_blacknwhite_p2_h0_4.5mr.bmp`.
- `<NN>` — calibration photo number (e.g., `2`, `13`, `27`) if the
  render targets a specific photo; otherwise `calibration` or `ref`.
- `<H.H>` — sun altitude in degrees with one decimal (e.g., `18.6`,
  `6.83`, `0` for the calibration set).
- `<config>` — crystal-block configuration descriptor: `default` for
  the canonical multi-block sim; `columnonly` for column + random only;
  `plateonly` etc. as needed.
- `<rays>` — ray count: `200k`, `1mr`, `3mr`, `4.5mr`, `6mr`, `100mr`.

Examples already on disk:
- `halosim_blacknwhite_p2_h0_4.5mr.bmp` (h=0° calibration, b&w, 4.5M rays)
- `halosim_tangent_p2_h18.6_25000mr.bmp` (h=18.6°, multi-block, 25k rays)
- `halosim_tangent_p2_h18.6_columnonly_3mr.bmp` (h=18.6°, column-only, 3M rays)

---

## Reusable project infrastructure

The following scripts under `scripts/` are the protocol's reusable
toolkit. Each has an inline docstring describing usage:

| script | role | status |
| --- | --- | --- |
| [`scripts/tangent_opening_angle.py`](../../scripts/tangent_opening_angle.py) | Reduce a photo's `upper_tangent_manual_samples` hand-anchor points to opening-angle metrics (azimuth from zenith, radial from sun). Used to map hand-anchor data into the same coordinate system as HaloSim measurements. | reusable; rename to `<feature>_opening_angle.py` for non-tangent features |
| [`scripts/tangent_halosim_calibrate.py`](../../scripts/tangent_halosim_calibrate.py) | Read a HaloSim render of a clean reference (22° halo only, h=0° or low-h) and report candidate halo radii via radial brightness profile. User picks the right radius from candidates as the scale lock. | reusable as `halosim_calibrate.py` for any reference feature |
| [`scripts/tangent_halosim_measure.py`](../../scripts/tangent_halosim_measure.py) | Initial measurement attempt using fixed-radius sampling. **Superseded** by the 2D arc-locus search inline in Pass C7's analysis. | deprecated; kept for record |
| [`scripts/tangent_halosim_horizon.py`](../../scripts/tangent_halosim_horizon.py) | Horizon-based scale-lock attempt. **Deprecated** because HaloSim's projection isn't linear horizon-to-zenith. | deprecated; kept for record |

The 2D arc-locus search itself is currently inline in the Pass C7
analysis. When a future feature requires it, lift to
`scripts/halosim_arc_locus.py` with arguments for the image path, sun
position, R22, azimuth range, and radial scan range.

Pre-edited `.sim` files for column-only configurations are at
[`docs/calibration/halosim_outputs/halosim_p2_h18.6_columnonly.sim`](halosim_outputs/halosim_p2_h18.6_columnonly.sim)
(project pre-edit) and
[`halosim_tangent_p2_h18.6_columnonly.sim`](halosim_outputs/halosim_tangent_p2_h18.6_columnonly.sim)
(user's GUI edit). Both produce the same column-only configuration; use
either as a template for new column-only sims at different sun
altitudes.

---

## Gotchas

1. **Auto-zoom per crystal-block configuration.** Re-lock the scale
   every time the crystal-block configuration changes, even if the
   `.sim` file's view parameters (FOV `60`, Type9, etc.) appear
   unchanged.
2. **Sun-detection ambiguity.** The brightest pixel on the central
   column can be the sun core OR the upper tangent arc apex (also
   saturated at low h). Use saturation density in a 21×21 box; the sun
   core is densely saturated, the apex is sparsely saturated.
3. **Monte Carlo noise at low ray counts.** Renders below ~1M rays have
   visible asymmetry that confounds feature detection. Use ≥1M rays
   for measurement; higher (4–6M) is better. The 25k-ray render Pass C7
   first attempted gave R/L asymmetry of 110 vs 255 brightness at
   symmetric positions — that asymmetry vanished at 3M rays.
4. **HaloSim renders concurrent features.** Parhelia + 22° halo + 46°
   halo + supralateral arc + circumscribed halo etc. all render
   together in a multi-block .sim. To isolate one feature for clean
   measurement, disable other crystal blocks (Pass C7's column-only
   render).
5. **The "Camera View" projection is perspective.** Near the sun, the
   pixel-to-degree mapping is approximately linear; far from the sun
   (≥30° angular distance), perspective distortion grows. Pass C7's 2D
   arc-locus search handled this implicitly by locking scale per render;
   don't try to apply h=0° scale to h=46° features.
6. **AGU paywall.** The full HaloSim documentation and the books behind
   the canonical orientations (Tape 1994 + Tape & Moilanen 2006) sit
   behind AGU paywalls. The project has on-disk excerpts at
   `AH-CH06/`, `AH-CH10/`, `AH-SAX-CH11/`; for additional chapters,
   acquisition policy is per the user's call.

---

## Worked example: Pass C7

Pass C7 applied this protocol to test the C5 hand-anchor circle fit on
p2 (h = 18.6°) against the canonical literature upper-tangent-arc
opening-angle inverse handle. Full receipt at
[`PASS_C7_OUTPUT.txt`](PASS_C7_OUTPUT.txt).

Summary:

1. **Configure:** column + random crystal blocks active, plate
   blocks disabled (pre-edited `halosim_p2_h18.6_columnonly.sim`),
   `Horiz column .1 deg disp` orientation, h = 18.6°.
2. **Render:** `halosim_tangent_p2_h18.6_columnonly_3mr.bmp` at 3M rays.
3. **Convert + scale-lock:** R22 = 464 px in HaloSim coords → 1° =
   21.09 px/° in this render (different from the h=0° multi-block
   calibration's 9.41 px/° due to auto-zoom).
4. **Extract feature locus:** 2D arc-locus search from az = -50° to
   +50°. Found the canonical upper-tangent-arc curve with apex at
   (az = 0, radial = 22.1° from sun, brightness ≈ 232), curving
   outward to radial ≈ 27° at az ≈ ±32°.
5. **Compare to claim:** C5 hand-anchor on p2 places outer wings at
   az = ±25.5°, radial = 31.6°. Canonical at az = ±25°: radial =
   25.4–26°. **C5 over-extends radially by 5.6° at the outer wings,
   progressively from 1.2° at the inner wings.** Falsification under
   standard tolerance.

Verdict: C5's circle fit is geometrically inconsistent with the
canonical arc curve. C5↔C6 substrate tension resolved in favor of C6;
tangent route remains unpromoted *including* under the canonical
literature handle.

---

## Updates / extensions

When Phase 12B (tilt-dispersion as math-bindable parameter) lands, add
a "Tilt-dispersion sensitivity sweep" subsection here that documents
the canonical (h, tilt_disp) cells to render for cross-validation.

When new feature families enter the atlas (Phase 12C deferred items —
pyramidal halos, Parry arcs as rendered primitives, etc.), add
worked examples for each here.
