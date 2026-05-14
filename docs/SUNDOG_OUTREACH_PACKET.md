# Sundog Outreach Packet

**Phase 7 deliverable, post-Phase-10-attack-campaign rewrite · 2026-05-14 · for atmospheric-optics specialists, technically literate science-communications editors, and Wikipedia-adjacent reviewers.**

The Halo Atlas is an interactive web page that renders a sundog display
from its underlying geometry. Drag the sun across the sky and every
named arc — the 22° halo, the parhelia, the circumzenithal arc, the
tangent arcs — updates from one parametric model. The math is standard
atmospheric optics from Greenler (1980) and Tape (1994); our contribution
is the *integrated, manipulable surface* and a *calibration workflow
against real photographs*. The packet does not introduce new geometric
claims; the "claim license" in §2 makes the boundary explicit.

## 0. Where this packet fits

This packet supports three review tiers (atmospheric-optics specialists,
technically-literate editors, Wikipedia-adjacent external-link reviewers).
Tier mechanics live in the outreach brief at
[`calibration/PHASE11_OUTREACH_BRIEF.md`](calibration/PHASE11_OUTREACH_BRIEF.md);
the specialist-tier handoff is at
[`calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md`](calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md).
The governing memo for the current claim surface is the re-audit memo at
[`calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md).

A reviewer who reaches the project through a citation can use this
packet to answer:

1. What is the atlas, in geometric terms? (§1)
2. Which equations are standard atmospheric optics, and which are
   original to this project? (§2)
3. Can the math be reproduced independently? (§3)
4. Which Wikipedia articles could legitimately cite the artifact, and
   for what specific factual claim? (§4)

### Current claim surface (audit-survived)

| route | status | failure-mode kind |
| --- | --- | --- |
| Parhelion offset → h | **promoted (post-hedged)** | works on a strict 3-photo subset (p2, p7, p13) |
| CZA apex → h | **fails coverage gate** | dataset / aspect-ratio (only one photo in the right altitude window) |
| Supralateral → h | **fails structural-discrimination** | atmospheric physics (the available signal is below typical measurement noise) |
| Tangent-arc curvature → h | **not promoted** | coverage + detector/anchoring tension: C5 manual samples recover p2, but C6 matched-filter falsifies the natural extension on the same b* substrate. The literature-standard tangent-arc inverse uses opening angle / arc extent (Tape 1994 §6), not circle-fit curvature; "curvature → h" is the project's exploratory framing. |

The audit narrowed three things in particular: it reframed the CZA
verdict from "the route is unreliable" to "we only have one in-window
measurement so we can't yet test it"; it tightened the parhelion-route
eligibility from "every photo in the calibration set" to "the strict
three-photo subset above"; and it sharpened the tangent-route state:
C5 manual sample selection partially recovers p2, C6 matched-filter
does not reproduce that recovery on the same halo-subtracted b*
substrate, and the route remains unpromoted pending specialist
re-anchoring or an alternative-substrate detector.

### Deployment gate

Before any external outreach, this packet, the handoff, and the brief
are bounce-tested by the Phase 11 synthetic-persona dispatch at
[`calibration/PHASE11_OUTREACH_SYNTHETIC_MEMO.md`](calibration/PHASE11_OUTREACH_SYNTHETIC_MEMO.md).
Editors who want to spot-check the audit-survived numbers can run the
regression test (Path D in §3) and the post-audit atlas calibration
(Path C in §3); both reproduce the numbers in the re-audit memo's
verification gate.

### Vocabulary note

Some sections of this packet carry the audit's internal vocabulary.
Plain-English glosses for readers who haven't read the audit memos:

- **The audit** — a 2026-05-13/14 internal review of the project's
  geometry and calibration claims, with eight required correction
  passes (named **Pass A1a / A1b / A2 / A3** for CZA-route work, **Pass
  B1 / B2** for parhelion-route eligibility work, **Pass C1** for the
  initial tangent-route taxonomy fix, followed by detector passes **C2 /
  C4 / C5 / C6**) and a re-audit gate that cleared the campaign.
- **Inverse route / inverse handle** — a way to recover the sun's
  altitude `h` from a feature visible in a photograph (e.g.
  parhelion-offset → `h`).
- **Lever / geometric lever** — how strongly a small change in the
  observed feature corresponds to a measurable change in `h`. Low
  lever = the feature changes barely at all across the relevant
  altitude range, so observed-vs-predicted residuals near zero may
  reflect measurement noise rather than route validity.
- **Eligibility / eligible photo** — a photo where the measurement
  geometry is clean enough to support a particular inverse-route
  test (independent 22° halo to fit, unambiguous parhelion peaks,
  etc.).
- **Audit-survived wording** — the framing the project commits to in
  public after the audit ran; replaces several earlier, looser
  framings.

---

## Quick links

- Live atlas explainer: <https://sundog.cc/sundog.html>
- Interactive math-binding tests (developer-facing implementation validation; not a citation surface): <https://sundog.cc/phase3-tests.html>
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
| 2 | `R₄₆ / R_sun_sky = 46°` | Angular radius of the 46° halo, from the 90° face-pair (basal + prism) refraction in **randomly oriented** hexagonal ice columns (plate-orientation belongs to CZA / supralateral, not the 46° halo itself). | Greenler ch. 2 |
| 3 | `parhelion_offset = R₂₂ / cos(h)` | Apparent screen-distance from the sun to a parhelion, as a function of sun altitude `h`. The standard small-angle screen-projection of the parhelion's azimuthal offset from the sun (Tape 1994 §3 gives the exact great-circle treatment for high-`h` cases). | Tape (1994); Greenler ch. 3 |
| 4 | CZA apex above sun = `arcsin(√(n² − cos²h)) − h` (with `n = 1.31`), scaled by `R₂₂ / 22°`. | The circumzenithal arc's altitude above the sun, as a function of sun altitude `h`. The "CZA tangent to the 46° halo at its top point" framing is geometrically exact only near `h ≈ 22°`; at other altitudes the CZA position varies as the formula. See `scripts/cza_formula.py` and §3 Path D for the regression test. | Cowley, atoptics CZA article; Bravais derivation in Greenler ch. 4 |
| 5 | CZA disappears gradually between ~31° (violet edge drops out first) and ~33° (red edge drops out last); visible-band centroid (n = 1.31) crosses at 32.2° | As `h` rises, the discriminant `n² − cos²h` exceeds 1 at progressively longer wavelengths, so the CZA fades from violet → blue → green → red across this ~2° altitude window rather than vanishing sharply. | Greenler ch. 4; ice n(λ) ranges ~1.306 (red) to ~1.317 (violet); see `scripts/cza_formula.py` for centroid computation |
| 5b | `WB_R₄₆ = round(2.09 · WB_R₂₂)` in workbench coordinates | Pre-audit the workbench held R₄₆ / R₂₂ at 2.0 (encoding 44° in workbench-deg rather than 46°). Pass A1b corrected to ≈ 2.09 (the integer-label 46/22 angular ratio; the underlying minimum-deviation physics gives a comparable value but the packet quotes the integer-label derivation, not a four-figure literature constant). Affects 46° halo radius rendering and supralateral apex base. | Pass A1b in `PHASE10_ATTACK_ROADMAP.md` |
| 6 | Upper / Lower Tangent Arc tangent to 22° halo top/bottom | "Eyelid" geometry from column-oriented crystals. Note: the literature-standard *inverse* handle for tangent arcs is opening angle / arc extent (Tape 1994 §6; Cowley tangent-arcs page), not circle-fit curvature. | Greenler ch. 3 |
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
   solving for `h` — is applied here as a measurement instrument; the
   inverse direction is treated in the standard literature (Tape 1994
   §3 gives the exact great-circle inversion; Cowley's atoptics
   parhelia page covers the small-angle case). The project's
   `overlay_calibrate.py` runner does this automatically, turning the
   workbench into an in-situ measurement tool for an unobserved
   variable (the sun's altitude when a photograph was taken).
3. **Calibration evidence on a post-audited photo cohort.** The
   parhelion-offset route survives audit-driven hedging on a strict
   three-photo subset (p2 h = 18.6°, p7 h = 59.4°, p13 h = 6.83°) —
   photos with unambiguous bilateral peaks, valid geometry, an
   independently fittable 22° halo, and (for p2 and p7) non-trivial
   geometric lever. p13 is included in the strict subset on the
   "unambiguous peaks + ring-fit halo" eligibility criterion per Pass
   B2 of the attack campaign; its 0.71% lever is anchor-noise-bounded,
   so its near-zero residual is informational rather than independently
   route-validating. Five additional anchored photos (p20, p22, p25,
   p26, p27, p30) contribute as low-lever or parhelion-derived
   informational evidence rather than route-validating residuals; the
   Pass B1 schema-level eligibility sub-table in
   [`calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md)
   names the per-photo classifications and the geometric-validity flag
   on p26's right-side parhelion. This is *demonstration that the
   integrated atlas works at photographic precision on a strictly
   bounded subset*, not a new physical result, and not a claim of
   uniform calibration quality across the full photo set.
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
- **A general-purpose photo-to-sun-altitude inverse system across the
  full rich-display vocabulary.** Pass A3 of the Phase 10 attack
  campaign tested four candidate `signature → h` inverse routes;
  parhelion-offset is the only one promoted on the strict 3-photo
  eligibility subset.
- **A validated CZA inverse handle.** CZA-apex fails the coverage
  gate after Pass A1b corrected the atlas formula and Pass A2
  re-classified p27's chromatic feature as 46° halo top / supralateral
  merger; only p2 remains in-window with an independent residual on
  the current anchored set.
- **A validated supralateral inverse handle.** Per Pass A3 + audit
  memo §2 item 12, supralateral's angular distance from sun varies
  only ~0.5° across h = 0–22°, putting the available h-signal below
  visual-edge measurement noise even with perfect coverage. The route
  is structurally weak as an inverse handle on atmospheric-physics
  grounds, not on dataset grounds.
- **A validated tangent-arc-curvature inverse handle, or a class-level
  tangent negative.** The current finding is narrower and more
  interesting: C5 manual sample selection partially recovers p2, but
  C6 matched-filter on the same halo-subtracted b* substrate returns a
  negative result. The route remains unpromoted because coverage fails
  (1 / 3 photos) and the p2 recovery needs specialist re-anchoring or
  an alternative-substrate detector before it can propagate.

---

## 3. Reproducibility statement

Every claim in this packet can be verified independently without
trusting Sundog-project code. Three reproduction paths:

### Path A — open the live page

**When you load the page,** you land on a rendered atlas: a stylized
sky scene centered on a glowing sun, with several named halo features
(22° halo, parhelia / sundogs, parhelic circle, upper tangent arc,
circumzenithal arc) drawn at the geometric positions the formulas in
§1 predict. A control panel lets you drag the sun across the sky,
adjust sun altitude, and toggle individual features on and off.

1. Open <https://sundog.cc/sundog.html>.
2. §4 "Sun altitude" shows the parhelion offset live as `R₂₂ × sec(h)`
   for any `h` you choose with the slider. The parhelia move outward
   from the sun as you raise the slider; below ~32° sun altitude the
   circumzenithal arc remains visible at the top of the scene; above
   that altitude it disappears.
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
2. The post-Pass-A1b atlas uses the literature CZA formula and the
   corrected `WB_R₄₆ = 460` workbench constant (see Path D and "Note
   on R₄₆" below); calibration photos under `docs/calibration/` plus
   their overlays under `docs/calibration/overlays/` form the
   regression baseline. The strict-eligibility parhelion-route
   residuals (p2, p7, p13) are ~0 px under the post-audit atlas;
   per-photo eligibility classifications are in
   [`calibration/RICH_DISPLAY_OVERLAY_NOTES.md`](calibration/RICH_DISPLAY_OVERLAY_NOTES.md)
   Parhelion-Route Per-Photo Eligibility sub-table.
3. Example invocation — parhelion-route demonstration:
   ```
   python scripts/overlay_calibrate.py docs/calibration/0.troelsnielsendr.png \
     --sun 400,356 --r22 145 --parhelion-offset 160 \
     --parhelion-left 243 --parhelion-right 560
   ```
   Expected output: dagger residuals of (-3, 0) px; inferred sun
   altitude ≈ 25.01°. (CZA-apex is intentionally omitted from this
   invocation: under the post-A1b literature formula the CZA prediction
   at h = 25.01° differs from the pre-audit hand-anchor on this photo
   by ~13 px because the hand-anchor was made against the legacy
   `sun_y − R₄₆` approximation. Use Path D to confirm the post-A1b CZA
   formula directly; use the p2 anchor JSON for a re-anchored
   in-window CZA residual measurement.)

### Path D — run the CZA literature-formula regression test

1. `scripts/test_cza_formula.py` runs the literature CZA formula
   against the p2 and p27 anchor JSONs and reports residuals against
   both (a) the post-Pass-A1b literature formula and (b) the
   pre-A1b legacy `sun_y − WB_R₄₆` approximation.
2. Invocation: `python scripts/test_cza_formula.py` from the repo root.
3. Expected output: at h = 22° the literature CZA-above-sun is
   45.734°; the CZA disappearance altitude is 32.196°. On p2
   (h = 18.6°) the residual collapses from −19.0 px (legacy) to
   −1.3 px (literature) — a 17.7 px reduction — and on p27
   (h = 0.5°) the literature formula places the CZA apex at y = −11.5,
   above the top of the photo, confirming the visible chromatic arc
   at p27 y = 142 is the 46° halo top / supralateral merger (not CZA;
   Pass A2 re-classification).

### Note on R₄₆ *(post-audit update 2026-05-14)*

Pre-audit, the workbench had a known-wrong `WB_R₄₆` value (440 px)
matched to a photo-mean ratio of 1.97; this was a confound, not a
measurement, and was retired in the audit. The corrected value is
`WB_R₄₆ = round(2.09 · WB_R₂₂) = 460`, matching the integer-label
angular ratio 46/22 ≈ 2.09. The change affects three things: the 46°
halo is now drawn ~4.5% larger (correctly to literature); the
supralateral apex base shifts outward by the same amount; and the
legacy `sun_y − WB_R₄₆` approximation for the CZA apex is replaced
end-to-end by the literature formula in `scripts/cza_formula.py` (the
approximation is preserved only as a fallback when the literature
formula reports the CZA has disappeared at high `h`).

Documented in [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) §"R_46
note" and in [`PHASE10_ATTACK_ROADMAP.md`](PHASE10_ATTACK_ROADMAP.md).

---

## 4. Suggested Wikipedia edits

> **Tier-3 placement note.** Per the Phase 11 outreach brief §8
> audience-order, Wikipedia-adjacent outreach (tier 3) only activates
> after the specialist tier and the editorial tier clear. The
> suggestions below are not yet ready for an external-links pull
> request; they are scoped to demonstrate the shape of the eventual
> ask. The list was deliberately pruned 2026-05-14 (post-audit
> dispatch) from six articles down to **two**: Sun dog and
> Circumzenithal arc. The dropped four (22° halo, 46° halo,
> Vädersoltavlan, Halo top-level) were dropped because they would
> have pointed multiple articles at the same atlas URL with
> project-framing blurbs, which fails the WP:ELNO #1 "unique
> resource" test against the well-established
> [atoptics.co.uk](https://atoptics.co.uk) reference. See the
> "Editorial caveats" subsection at the end of §4 for the dropped-list
> rationale.

Each surviving suggestion below names a specific factual claim in an
existing Wikipedia article that the Sundog atlas could legitimately
*supplement* as an external link or "See also" reference. **None** of
these suggestions ask Wikipedia to cite the atlas as a source for the
underlying physics — every formula already cites Greenler / Tape /
Cowley. The atlas's role is as a *runnable demonstration*.

### Sun dog (https://en.wikipedia.org/wiki/Sun_dog)

The article discusses parhelion formation, the 22° refraction, and
notes that parhelia drift from the sun's altitude. The atlas can
supplement:

- **Claim:** *"As the sun rises higher in the sky, the parhelia appear
  farther from the sun, on the parhelic circle."* (currently uncited
  in the article body)
- **Suggested supplement:** External-links entry: "Interactive
  demonstration of the `R₂₂ / cos(h)` parhelion-offset relation
  (Greenler 1980, ch. 3): sundog.cc/sundog.html"
- **Why it fits:** the article asserts the relationship qualitatively;
  the atlas demonstrates it quantitatively with a slider, anchored to
  Greenler's derivation rather than to project-original work.

### Circumzenithal arc (https://en.wikipedia.org/wiki/Circumzenithal_arc)

The article describes CZA visibility cutoff at sun altitudes above
~32° and the CZA's relationship to the 46° halo.

- **Claim:** *"The arc is only visible when the sun is below 32.2°
  altitude."* (matches the literature ~32° visibility cutoff;
  Greenler ch. 4)
- **Suggested supplement:** External-links entry: "Atlas with live
  visibility-cutoff demonstration (Greenler 1980, ch. 4):
  sundog.cc/sundog.html#sun-altitude-binding"
- **Why it fits:** the explanatory threshold is a fact in the article;
  a manipulable slider that disappears the arc at exactly that
  threshold is a useful pedagogical adjunct.
- **Wording caveat:** the CZA's "tangency to the 46° halo at its top
  point" framing is geometrically exact only near sun altitude
  ~22° — at other altitudes the literature CZA position varies with
  `h` (see §1 row 4). Any external-link copy that says "the CZA is
  tangent to the 46° halo" should be hedged or scoped to the
  near-disappearance regime.

### Editorial caveats

These are *suggestions*, not pull requests. Wikipedia external-links
guidance (`WP:EL`) is conservative; the post-audit dispatch
(2026-05-14) flagged that any external-link case must satisfy three
constraints before being defensible: (i) on-workbench attribution to
Greenler / Tape / Cowley must be visible above-the-fold on
`sundog.cc/sundog.html` itself (a Wikipedia reviewer clicks the link,
not this packet); (ii) link blurbs must name the literature source by
surname, not the workbench's framing; (iii) the project-internal
"we are not aware of an existing public resource…" line (§2 item 1),
the §2 item 2 inversion-workflow framing, and the §2 item 3
strict-3-photo calibration-evidence framing must not be quoted on
any external-link-defending talk-page reply — these are legitimate
internal claim-license framings on the project's own surfaces but
become WP:NOR / WP:OR risk if quoted in defense of an external-link
addition.

The proposal list was pruned to **two** (Sun dog, Circumzenithal arc)
after the dispatch flagged that 22° halo / 46° halo / Vädersoltavlan /
Halo (optical phenomenon) proposals would point the same atlas URL at
multiple articles with project-framing blurbs ("single calibrated
parametric model," "all four 46°-tangent features toggleable,"
"derivable consequences of a small set of circles") — the WP:ELNO #1
pattern. The dropped proposals are filed for possible reconsideration
*after* the two surviving proposals have actually cleared review at
their respective articles' talk pages.

The strongest case is **Sun dog** §"Geometry" — the `R₂₂ / cos(h)`
relation is explicit in the article and the atlas shows it live
without requiring HaloSim. The CZA-visibility-cutoff suggestion is
the next strongest.

### Citation form (suggested)

If a Wikipedia editor decides to add the atlas as an external link,
the suggested citation template:

```
* [https://sundog.cc/sundog.html The Sundog Halo Atlas — interactive
  parametric model of the parhelion display, calibrated against
  photographs]. Jeffery Hughes Jr. (Stellar Aqua LLC), 2026. Source
  code under MIT license at https://github.com/humiliati/sundog.
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

This packet is point-in-time as of **2026-05-14**, after the Phase 10
attack campaign + re-audit + specialist-handoff-rewrite + public-framing
revision wave. The campaign provenance chain is:

1. **2026-05-12** — original packet drafted (this file's previous
   point-in-time).
2. **2026-05-13** — synthetic three-persona optical audit (memo at
   [`calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md)).
3. **2026-05-13/14** — attack roadmap eight required passes landed
   (§6 hedges, B1 schema, A1a formula spec, A1b atlas patch, A2 p27
   re-classification, A3 CZA + supralateral re-verdict, C1 p7 dropped
   from tangent eligibility, B2 parhelion re-verdict, and follow-on
   tangent detector passes C2 / C4 / C5 / C6).
4. **2026-05-14** — re-audit gate cleared
   ([`calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md));
   specialist handoff rewritten end-to-end;
   `MESA_CROSSOVER_NOTE`, `SUNDOG_V_GRAVITY`, `PROMO_HIGHLIGHTS`,
   and this packet revised to the post-pass failure-mode taxonomy.

If the atlas math ever changes again (e.g. if specialist re-anchoring
confirms C5's p2 tangent recovery, if an alternative-substrate detector
reconciles the C5↔C6 tension, or if new anchored photos in 5° < h < 32° supply a second in-window CZA
residual), the corresponding rows in §1, §2, §3 Path C/D, and the
audit-survived-taxonomy table in §0 need updating and the calibration
evidence re-verified.

If a Wikipedia edit referencing the atlas is accepted, log it here:

| Date | Article | Edit type | Editor / source |
|---|---|---|---|
| *(no edits accepted yet)* | | | |

---

*The packet's purpose is to make the citation case easy for an
external reviewer. Conservatism beats overreach: every formula is
cited to a primary source, and the project's contribution is named as
implementation and demonstration, not theory.*
