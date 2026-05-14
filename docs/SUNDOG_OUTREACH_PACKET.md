# Sundog Outreach Packet

**Phase 7 deliverable, post-Phase-10-attack-campaign rewrite · 2026-05-14 · for Wikipedia editorial review, technically literate editors, and atmospheric-optics specialists.**

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

## 0. Where this packet fits

This packet is the **shared support artifact** across the three Phase
11 outreach tiers documented in
[`calibration/PHASE11_OUTREACH_BRIEF.md`](calibration/PHASE11_OUTREACH_BRIEF.md):

- **Tier 1 — atmospheric-optics specialists.** Served by the rewritten
  specialist handoff in
  [`calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md`](calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md)
  (✅ cleared for external handoff 2026-05-14); this packet provides
  the shared math + claim-license + reproducibility background a
  specialist will skim to anchor the audit-ask in §2 of the handoff.
- **Tier 2 — technically literate science-communications editors.**
  Served primarily by this packet, especially §1 (math summary), §2
  (claim license), and §4 (Wikipedia-adjacent suggestions); the live
  workbench at `sundog.cc/sundog.html` is the principal artifact
  reviewed.
- **Tier 3 — Wikipedia-adjacent editors / external-link reviewers.**
  Served by §4 of this packet plus the live workbench; per the
  brief's §8 audience order, tier 3 only activates after tiers 1 and
  2 clear.

The post-Phase-10 attack campaign + 2026-05-14 re-audit narrowed the
project's public claim surface in three specific directions; this
packet reflects those changes throughout. The load-bearing governing
document is the re-audit memo at
[`calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md).
The campaign provenance chain (synthetic audit → attack roadmap → eight
required passes → re-audit gate) is in the handoff §1.

**Deployment gate.** Before any external outreach happens, this packet
plus the handoff and the brief are bounce-tested by the Phase 11
synthetic-persona dispatch at
[`calibration/PHASE11_OUTREACH_SYNTHETIC_MEMO.md`](calibration/PHASE11_OUTREACH_SYNTHETIC_MEMO.md)
(scaffolded 2026-05-14; persona passes not yet executed). The dispatch
serves the same verify-gate discipline that earned its keep three times
in the Phase 10 attack campaign; only artifacts that survive the
verified-findings bin should ratchet under external use.

The audit-survived post-pass route taxonomy is:

| route | status | failure-mode kind |
| --- | --- | --- |
| Parhelion offset → h | **promoted (post-hedged)** | 3-photo strict eligibility (p2, p7, p13) |
| CZA apex → h | **fails coverage gate** | dataset / aspect-ratio (only p2 in-window) |
| Supralateral → h | **fails structural-discrimination gate** | atmospheric physics (h-spread below noise) |
| Tangent-arc curvature → h | **detection gate; class-level verdict pending C2** | tooling-protocol (filed as Unresolved Open Question) |

Editors who want to spot-check the campaign's findings can run the
regression test (Path D in §3) and the post-A1b atlas calibration
(Path C in §3); both reproduce the audit-survived numbers from the
re-audit memo's verification gate.

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
| 4 | CZA apex above sun = `arcsin(√(n² − cos²h)) − h` (with `n = 1.31`), scaled by `R₂₂ / 22°`. | The circumzenithal arc's altitude above the sun, as a function of sun altitude `h`. The legacy atlas approximated this as `sun_y − R₄₆` (the 46° halo top), which is geometrically correct only at h ≈ 22°; the Phase 10 attack campaign's Pass A1b replaced the approximation with the literature formula (see `scripts/cza_formula.py` and §3 Path D for the regression test). | Cowley, atoptics CZA article; Bravais derivation in Greenler ch. 4 |
| 5 | CZA disappears at h > ~32.196° | Above the disappearance threshold the discriminant `n² − cos²h` exceeds 1 and the CZA passes the zenith. | Greenler ch. 4; computed exactly in `scripts/cza_formula.py` |
| 5b | `WB_R₄₆ = round(2.091 · WB_R₂₂)` in workbench coordinates | Pre-audit the workbench held R₄₆ / R₂₂ at 2.0 (encoding 44° in workbench-deg rather than 46°). Pass A1b corrected to 2.091 per the literature angular ratio. Affects 46° halo radius rendering and supralateral apex base. | Pass A1b in `PHASE10_ATTACK_ROADMAP.md` |
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
- **A class-level tangent-arc-curvature negative.** The current
  detection-gate finding (Pass C1) is protocol-conditional: column-peak
  detection on the post-C1 sampled set (p2, p13, p27) fails with three
  distinct degeneracy modes, but a wing-based or Lab b\* ridge
  detector has not been built and tested. Filed as Unresolved Open
  Question in the specialist handoff.

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

### Note on R₄₆ *(updated 2026-05-14, post-Pass-A1b)*

The pre-audit workbench held `WB_R₄₆ / WB_R₂₂` at 2.0, encoding 44° in
workbench-degree units rather than 46° (because `WB_R₂₂ = 220` at
10 px/deg, so `WB_R₄₆ = 440` mapped to 44°). The pre-audit rationale
held that this matched a 7-photo calibration median ratio of 1.97
better than the literature 2.0909.

The Phase 10 attack campaign's Pass A1b explicitly retired that
rationale and changed `WB_R₄₆ = round(2.091 · WB_R₂₂) = 460`,
matching the literature angular ratio. The change has three
load-bearing effects:

1. The 46° halo is now drawn at the literature angular ratio rather
   than ~4.5% smaller.
2. The supralateral apex base (anchored at `WB_SUN[1] − WB_R₄₆` in
   workbench coords) shifts ~4.5% outward from sun, matching the
   literature 46° rather than the pre-audit 44°.
3. The pre-A1b CZA approximation (`sun_y − WB_R₄₆`) is no longer the
   atlas's CZA-apex predictor; the literature formula in
   `scripts/cza_formula.py` is. The legacy approximation is preserved
   only as a fallback for h > 32.196° where the literature formula
   returns None (CZA disappeared); downstream visibility
   classifications still mark CZA as "not applicable" at high h.

The pre-audit "7-photo median ratio of 1.97" finding does not survive
the post-Pass-B1 eligibility schema: of the seven photos, p20, p25,
p26, and p27 have parhelion-derived or tautological R₂₂ values, so
their R₄₆ / R₂₂ contributions to that median are not measurements of
the literature ratio in the sense the original Note intended. The
post-audit position is: render at the literature ratio (2.091),
acknowledge that photo-by-photo R₄₆ pixel measurement on this set has
its own residual structure (per Pass B1's geometric-validity column
and the parhelion-derived flag), and treat the workbench constant as
literature-conformant rather than photo-mean-conformant. Documented in
[`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) §"R_46 note" *(see also
the Phase 10 closeout table for the post-pass route verdicts)* and in
the
[`PHASE10_ATTACK_ROADMAP.md`](PHASE10_ATTACK_ROADMAP.md) Pass A1b touch
block.

---

## 4. Suggested Wikipedia edits

> **Tier-3 placement note (added 2026-05-14).** Per the Phase 11
> outreach brief §8 audience-order, Wikipedia-adjacent outreach
> (tier 3) only activates after the specialist tier and the editorial
> tier clear. The suggestions below are not yet ready for an
> external-links pull request; they are scoped to demonstrate the
> shape of the eventual ask. Two of the suggestions (CZA visibility
> cutoff; 46° halo tangent companions) have wording updates after
> Pass A1b corrected the legacy CZA approximation; treat the older
> phrasing as superseded.

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
~32° and the CZA's relationship to the 46° halo.

- **Claim:** *"The arc is only visible when the sun is below 32.2°
  altitude."* (computed exactly as 32.196° in `scripts/cza_formula.py`
  per Pass A1a of the Phase 10 attack campaign)
- **Suggested supplement:** External-links entry: "Atlas with live
  visibility-cutoff demonstration:
  sundog.cc/sundog.html#sun-altitude-binding"
- **Why it fits:** the explanatory threshold is a fact in the article;
  a manipulable slider that disappears the arc at exactly that
  threshold is a useful pedagogical adjunct.
- **Wording caveat post-Pass-A1b:** the CZA's "tangency to the 46°
  halo at its top point" framing is geometrically exact only at h ≈ 22°.
  At other sun altitudes the literature CZA position is given by
  `arcsin(√(n² − cos²h)) − h` (see §1 row 4 and §3 Path D); the
  46°-halo-top-tangency is a useful pedagogical approximation, not a
  universal geometric constraint. Any external-link copy that says
  "the CZA is tangent to the 46° halo" should be hedged or scoped to
  the near-disappearance regime.

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

This packet is point-in-time as of **2026-05-14**, after the Phase 10
attack campaign + re-audit + specialist-handoff-rewrite + public-framing-
ratchet wave. The campaign provenance chain is:

1. **2026-05-12** — original packet drafted (this file's previous
   point-in-time).
2. **2026-05-13** — synthetic three-persona optical audit (memo at
   [`calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md)).
3. **2026-05-13/14** — attack roadmap eight required passes landed
   (§6 hedges, B1 schema, A1a formula spec, A1b atlas patch, A2 p27
   re-classification, A3 CZA + supralateral re-verdict, C1 p7 dropped
   from tangent eligibility, B2 parhelion re-verdict).
4. **2026-05-14** — re-audit gate cleared
   ([`calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md));
   specialist handoff rewritten end-to-end;
   `MESA_CROSSOVER_NOTE`, `SUNDOG_V_GRAVITY`, `PROMO_HIGHLIGHTS`,
   and this packet ratcheted to the post-pass failure-mode taxonomy.

If the atlas math ever changes again (e.g. if Pass C2 ships a
wing-based tangent detector that recovers the tangent route, or if new
anchored photos in 5° < h < 32° supply a second in-window CZA
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
