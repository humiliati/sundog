# Sundog Outreach Packet

**Phase 7 deliverable · 2026-05-12 · for Wikipedia editorial review, technical reviewers, and SEO-focused publications.**

This packet exists so a Wikipedia editor — or any reviewer who reaches the
Sundog project through a citation — can answer four questions inside one
scroll:

1. What is the project's atlas, in geometric terms?
2. Which equations are standard atmospheric optics, and which are
   original to this project?
3. Can the math be reproduced independently?
4. Which Wikipedia articles could legitimately cite the artifact, and
   for what specific factual claim?

The packet does **not** introduce new geometric claims. Every formula
referenced below predates this project; the project's contribution is
*integration, calibration, and interactive presentation*, not a new
theorem. The "claim license" section below makes that boundary explicit.

---

## Quick links

- Live atlas explainer: <https://sundog.cc/sundog.html>
- Interactive math-binding tests: <https://sundog.cc/phase3-tests.html>
- Geometry source module: [`public/js/parhelion-geometry.mjs`](../public/js/parhelion-geometry.mjs)
- Calibration script: [`scripts/overlay_calibrate.py`](../scripts/overlay_calibrate.py)
- 7-photo calibration overlays: [`docs/calibration/overlays/`](calibration/overlays/)
- Roadmap with full references: [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md)
- Photo-data privacy policy: [`PHOTO_DATA_POLICY.md`](PHOTO_DATA_POLICY.md)

---

## 1. One-page math summary

The Halo Atlas renders a sundog display as a small set of geometric
primitives — each a complete circle whose visible portion in the sky is
one of the named arcs in atmospheric-optics literature. Seven formulas
do the work.

| # | Formula | What it describes | Source |
|---|---|---|---|
| 1 | `R₂₂ / R_sun_sky = 22°` | Angular radius of the 22° halo, from refraction through a 60° prism in column-oriented hexagonal ice. | Greenler ch. 2 |
| 2 | `R₄₆ / R_sun_sky = 46°` | Angular radius of the 46° halo, from the 90° prism path through plate-oriented hexagonal ice. | Greenler ch. 2 |
| 3 | `parhelion_offset = R₂₂ / cos(h)` | Apparent screen-distance from the sun to a parhelion, as a function of sun altitude `h`. Equivalent to the standard great-circle ↔ parhelic-circle projection geometry. | Tape (1994); Greenler ch. 3 |
| 4 | CZA apex at `(sun_x, sun_y − R₄₆)` | The circumzenithal arc is tangent to the 46° halo at its top point — a refractive-geometry constraint, not a fitting choice. | Greenler ch. 4; Cowley, atoptics CZA article |
| 5 | CZA visible only for `h ≤ 32°` | Above ~32° sun altitude, the CZA's tangent point exits the visible hemisphere. | Greenler ch. 4 |
| 6 | Upper / Lower Tangent Arc tangent to 22° halo top/bottom | "Eyelid" geometry from column-oriented crystals. | Greenler ch. 3 |
| 7 | Supralateral / Infralateral Arc tangent to 46° halo top/bottom | Column-orientation 90° path, mirror-paired around the 46° halo. | Greenler ch. 4 |

In the atlas, every visible primitive is a one-line consequence of one of
these formulas. The full reference table — including chapter and page
pointers — lives in
[`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) §"Atmospheric Optics
References".

### Primary references

- **Greenler, R. (1980, reissued 1990).** *Rainbows, Halos, and
  Glories.* Cambridge University Press.
- **Tape, W. (1994).** *Atmospheric Halos.* American Geophysical Union,
  Antarctic Research Series Vol. 64.
- **Cowley, L.** *Atmospheric Optics.* [`atoptics.co.uk`](https://atoptics.co.uk/)

---

## 2. Claim license

The hardest review question for a project like this is: *what's new and
what's borrowed?* The answer is asymmetric: the math is borrowed, the
artifact is original. Specifically:

### Standard (cited to the primary references above)

The following are textbook atmospheric optics and date to at least
Greenler 1980. The Sundog project did not derive them; we implement
them and verify them against photographs.

- The 22° halo's angular radius and its origin in 60° hexagonal-prism
  refraction.
- The 46° halo's angular radius and its origin in 90° hexagonal-prism
  refraction.
- Parhelion azimuthal offset from the sun as a function of sun altitude,
  `R₂₂ / cos(h)`.
- The CZA's tangency to the 46° halo at its top point.
- The CZA's visibility cutoff at sun altitude ≈ 32°.
- Tangent-arc geometry for the 22° halo (upper / lower) and 46° halo
  (supralateral / infralateral).
- The Parry-orientation crystal-family distinction (Suncave Parry, Parry
  Supralateral).

### Original contribution of this project

The Sundog project's contribution is *not* a new equation. It is:

1. **An integrated primitive-atlas presentation.** A single interactive
   workbench (`sundog.html`) renders every named arc as the visible
   upper portion of a complete circle, with each circle anchored to one
   environmental-state variable. We are not aware of an existing public
   resource that surfaces the full vocabulary as one parametric model
   the public can manipulate.
2. **An interactive inverse-inference workflow.** Running the forward
   relation `parhelion_offset = R₂₂ / cos(h)` *backwards* — measuring
   the parhelion offset and the 22° halo radius in a photograph and
   solving for `h` — is a non-obvious use of the formula. It turns the
   workbench into a measurement instrument for an unobserved variable
   (the sun's altitude when a photograph was taken), and the
   `overlay_calibrate.py` runner does this automatically.
3. **Calibration evidence across a seven-photo cohort.** Photographs
   spanning sun altitudes from ~1° to ~60° all calibrate with median
   absolute dagger residual of **1 photo pixel**. The residuals table
   is in [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) §"Multi-Photo
   Calibration Pass (2026-05-12)". This is *demonstration that the
   integrated atlas works at photographic precision*, not a new
   physical result.
4. **A drag-to-tune constraint network.** Five drag handles on the live
   atlas inverse-bind to one parameter each (parhelic-curvature,
   sun-altitude via parhelion offset, cza-curvature, sun pixel,
   R₂₂-anchor) so the visible atlas can be manipulated through any
   named feature. The constraint network is engineering work, not
   physics.
5. **A reproducible JSON pose schema** (`canonical-halo-atlas.json`)
   that round-trips losslessly to and from the rendered SVG. Locks the
   default canonical pose against future regressions.

### What this project does NOT claim

- Novel atmospheric-optics findings.
- That sundogs are "explained" by this project — they have been
  explained for 200+ years; we render the explanation.
- Photometric or radiometric simulation. The atlas does refractive
  geometry, not ray-traced color.
- Sub-horizon arc rendering, crystal-orientation mixing, or
  crystal-size effects on arc width — these are out of scope and named
  in [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) §"What the atlas
  does NOT yet model".

---

## 3. Reproducibility statement

Every claim in this packet can be verified independently without
trusting Sundog-project code. Three reproduction paths:

### Path A — open the live page

1. Open <https://sundog.cc/sundog.html>.
2. §4 "Sun altitude" shows the parhelion offset live as `R₂₂ × sec(h)`
   for any `h` you choose with the slider.
3. §6 photo-upload widget runs the inverse inference on a photograph of
   your own — pick the sun, the 22° halo edge, and a parhelion, and
   the workflow reports the sun's altitude.

### Path B — run the assertions

1. Open <https://sundog.cc/phase3-tests.html>.
2. The page loads `parhelion-geometry.mjs` and runs **32 assertions**
   against the exported `phase3` namespace:
   - `daggerOffset(h)` matches `R₂₂ / cos(h)` to within 0.05 px across
     altitudes 0°-60°.
   - `czaVisible(h)` flips at exactly the 32° boundary.
   - `parhelicCurvature(h)` is monotonic and bounded in [0, 1].
   - Structural invariants: `daggerOffset(0)` = R₂₂, `daggerOffset(60°)`
     = 2·R₂₂, `R₄₆ / R₂₂` = 2 (workbench convention; physics 46°/22° ≈
     2.09, see "R_46 note" below).
3. All 32 assertions pass at landing.

### Path C — run the calibration script

1. `scripts/overlay_calibrate.py` takes a photograph and three anchor
   measurements (sun pixel, R₂₂ in pixels, observed parhelion offset)
   and emits an overlay PNG plus a residuals report. No workbench
   needed.
2. The seven calibration photos under `docs/calibration/` plus their
   overlays under `docs/calibration/overlays/` form the regression
   baseline. Median residual: 1 px. Maximum: 5 px (on p9).
3. Example invocation:
   ```
   python scripts/overlay_calibrate.py docs/calibration/0.troelsnielsendr.png \
     --sun 400,356 --r22 145 --parhelion-offset 160 \
     --parhelion-left 243 --parhelion-right 560 \
     --cza-apex 402,65
   ```
   Expected output: dagger residuals of (-3, 0) px; CZA apex residual
   (2, 1) px; inferred sun altitude ≈ 25.01°.

### Note on R₄₆

The workbench renders the 46° halo at twice the 22° halo's screen
radius (2 × R₂₂). The strict atmospheric-optics ratio is 46°/22° ≈
2.0909. We hold the ratio at 2.0 because the 7-photo calibration
median ratio measured in actual photographs is 1.97, and 2.0 fits the
photographs to 1.8% while 2.0909 fits to 6.3%. The 5–6% gap shows up
routinely in the atmospheric-optics literature because 46°-halo
refraction is more sensitive to ice-crystal orientation than 22°.
Documented in [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) §"R_46
note".

---

## 4. Suggested Wikipedia edits

Each suggestion below names a specific factual claim in an existing
Wikipedia article that the Sundog atlas could legitimately *supplement*
as an external link or "See also" reference. **None** of these
suggestions ask Wikipedia to cite the atlas as a source for the
underlying physics — every formula already cites Greenler / Tape /
Cowley. The atlas's role is as a *runnable demonstration*.

### Sun dog (https://en.wikipedia.org/wiki/Sun_dog)

The article discusses parhelion formation, the 22° refraction, and
notes that parhelia drift from the sun's altitude. The atlas can
supplement:

- **Claim:** *"As the sun rises higher in the sky, the parhelia appear
  farther from the sun, on the parhelic circle."* (currently uncited
  in the article body)
- **Suggested supplement:** External-links entry: "Interactive atlas
  demonstrating the `R₂₂ / cos(h)` relationship across sun altitudes,
  with photograph calibration evidence: sundog.cc/sundog.html"
- **Why it fits:** the article asserts the relationship qualitatively;
  the atlas demonstrates it quantitatively with a slider.

### Circumzenithal arc (https://en.wikipedia.org/wiki/Circumzenithal_arc)

The article describes CZA visibility cutoff at sun altitudes above
~32° and the CZA's tangency to the 46° halo.

- **Claim:** *"The arc is only visible when the sun is below 32.2°
  altitude."*
- **Suggested supplement:** External-links entry: "Atlas with live
  visibility-cutoff demonstration:
  sundog.cc/sundog.html#sun-altitude-binding"
- **Why it fits:** the explanatory threshold is a fact in the article;
  a manipulable slider that disappears the arc at exactly that
  threshold is a useful pedagogical adjunct.

### 22° halo (https://en.wikipedia.org/wiki/22°_halo)

The article shows the 22° halo and discusses related arcs (parhelia,
upper tangent arc, parhelic circle).

- **Claim:** *"The 22° halo is closely associated with parhelia and
  the parhelic circle, which appear at the same angular radius from
  the sun."* (paraphrased)
- **Suggested supplement:** External-links entry: "Interactive atlas
  showing the 22° halo, parhelia, upper/lower tangent arcs, and
  Suncave Parry arc as a single calibrated parametric model:
  sundog.cc/sundog.html#full-atlas"

### 46° halo (https://en.wikipedia.org/wiki/46°_halo)

The article notes the 46° halo's rarity and its tangent companions
(supralateral, infralateral, CZA).

- **Claim:** *"The supralateral arc, infralateral arcs, and
  circumzenithal arc are all tangent to the 46° halo."*
- **Suggested supplement:** External-links entry: "Atlas with all
  four 46°-tangent features toggleable on a single calibrated scene:
  sundog.cc/sundog.html#full-atlas"

### Vädersoltavlan (https://en.wikipedia.org/wiki/Vädersoltavlan)

The article describes the 1535 Stockholm "Sun Dog Painting" — a key
historical depiction of a complex halo display.

- **Claim:** The article identifies the visible features in the
  painting (22° halo, parhelia, upper tangent arc, etc.).
- **Suggested supplement:** Could reference the Halo Atlas as a way
  for modern readers to manipulate the same geometric primitives the
  painting depicts. Lower priority; the article is fine without it.

### Halo (optical phenomenon) (https://en.wikipedia.org/wiki/Halo_(optical_phenomenon))

The top-level halo article references multiple specific halos with
their own pages. The atlas is potentially useful here as an
integrative resource — the page already has a "See also" section.

- **Suggested supplement:** Add to "External links": "Halo Atlas — an
  interactive geometric model of the full named-arc vocabulary,
  rendered as derivable consequences of a small set of circles:
  sundog.cc/sundog.html"

### Editorial caveats

These are *suggestions*, not pull requests. Wikipedia external-links
guidance (`WP:EL`) is conservative; reviewers may reject any of these
if they feel the atlas duplicates `atoptics.co.uk` or HaloSim outputs
that are already cited. The strongest case is **Sun dog** §"Geometry"
— the `R₂₂ / cos(h)` relation is explicit in the article and the
atlas shows it live without requiring HaloSim. The CZA-visibility-cutoff
suggestion is the next strongest.

### Citation form (suggested)

If a Wikipedia editor decides to add the atlas as an external link,
the suggested citation template:

```
* [https://sundog.cc/sundog.html The Sundog Halo Atlas — interactive
  parametric model of the parhelion display, calibrated against
  photographs]. Stellar Aqua LLC, 2026. Source code under MIT license
  at https://github.com/humiliati/sundog.
```

---

## 5. What this packet does NOT request

For clarity: the project is not asking Wikipedia to

- Add the Sundog atlas as a *source* for any underlying physics
  claim (the references in this packet — Greenler, Tape, Cowley —
  are the correct sources).
- Cite the project's `phase3-tests.html` page directly in article
  prose. That page validates the *implementation*, not the physics.
- Reference the photo-upload feature or the Cloudflare backend.
  Those are project mechanics; only the math + the rendered atlas
  are appropriate for editorial citation.

The packet asks only for **external-links additions** where they
genuinely help readers, and only for articles where the atlas
demonstrates a specific claim the article already makes.

---

## 6. Maintenance

This packet is point-in-time as of **2026-05-12**. If the atlas math
ever changes (e.g. if a Phase 10 review promotes a Parry-family
primitive into the canonical pose), the corresponding row in §1 needs
to be updated and the calibration evidence re-verified.

If a Wikipedia edit referencing the atlas is accepted, log it here:

| Date | Article | Edit type | Editor / source |
|---|---|---|---|
| *(no edits accepted yet)* | | | |

---

*The packet's purpose is to make the citation case easy for an
external reviewer. Conservatism beats overreach: every formula is
cited to a primary source, and the project's contribution is named as
implementation and demonstration, not theory.*
