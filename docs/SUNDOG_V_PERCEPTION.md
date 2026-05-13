# Sundog vs. the Limits of Perception

Working hook:

> The atlas describes the visible parhelion. The instrument predicts the
> parhelion features that exist below the human contrast threshold and
> confirms them with hardware.

This roadmap promotes the cross-pollination memo at the top of this file's
prior life into a structured roadmap. It is the geometry-side analogue of
[`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md) Phase 6: where mesa went from
"describing the cliff behaviorally" (Phase 5) to "localizing the
mechanism causally" (Phase 6), geometry goes from "overlaying known arcs"
(Phase 7 hero) to "predicting where unseen features must lie and
detecting them in the real atmosphere."

The transition is structural, not aspirational. The atlas already
parametrises every visible arc as the upper portion of a complete implied
circle derived from sun altitude and crystal orientation. The full
geometry of each circle continues below the contrast floor of unaided
photography. The instrument's job is to integrate signal along the
predicted continuation at the angular positions the atlas already names,
exceeding what the eye and naked-sensor camera resolve.

## Why Perception Is the Right Next Roadmap

The gravity claim in [`SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md) now
carries two independent in-substrate receipts for the field-not-reward
framing: mesa's in-vitro entangled 5D `net.7` subspace, and geometry's
in-the-wild halo atlas. Both converged on the same shape — small
handful of generators, irreducibly entangled, only legible as a whole
— by independent methods.

The two-substrate convergence is currently a structural argument. Either
substrate could be retorted as "you fit a model that happens to look like
that." The retort closes when geometry earns a **predictive receipt**:
the atlas names a feature that no one had measured, and a measurement
confirms it where the math said it would be.

That move turns the workbench into an instrument. The atlas already
inverts photographs to recover sun altitude with median 1 photo-px
residual across seven photos; that is a strong inverse-inference
demonstration. The next ratchet is to predict a feature *the photograph
does not show* and confirm it with instrumentation that integrates
signal along the predicted geometry.

Methodologically, perception inherits mesa's discipline wholesale:
pre-registered predictions, quantitative smoke gates, named null
conditions, and clean negatives as publishable deliverables. The
[`MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md) already lifted the
five transferable findings into Phase 10/11 of geometry; perception
extends the same discipline one substrate further.

## Cross-Pollination Provenance

This roadmap originated as a cross-pollination memo from the mesa work
into the geometry program, written 2026-05-13 alongside
[`MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md). The memo argued that
the atlas's "what it does NOT yet model" list (halo dispersion physics,
subhorizon arcs, crystal-orientation mixing, crystal-size effects) is
not an apology surface but a **prediction surface** — each item is a
falsifiable prediction the 7-formula field makes when extended along its
own generators.

This roadmap commits to the reframe: the atlas predicts; the instrument
confirms or falsifies. Pre-registration discipline is mandatory because
without it, "we extended the atlas and it matched" is indistinguishable
from confirmation bias.

## Claim Boundary

The perception roadmap claims, if Phase 4 lands positive:

- In the tested high-cirrus conditions and apparatus tier, the parhelic
  circle's polarization signature is detectable along the atlas-predicted
  continuation at angular positions and amplitudes consistent with
  standard ice-crystal refraction physics, below the contrast threshold
  of unaided photographic observation.
- The atlas therefore functions as a parametric measurement instrument
  for sub-visible parhelion-display features that exist deterministically
  as consequences of the field.

The perception roadmap does **not** claim:

- "The atlas predicts every atmospheric phenomenon."
- "Sundog instrumentation can replace HaloSim or full ray-trace
  simulators."
- "Sub-visible halo phenomena were previously unknown to atmospheric
  optics." (They are not; standard references treat them as known but
  rarely instrumented in amateur or workbench contexts.)
- "A single positive prediction proves the gravity claim." A predictive
  receipt is one rung on the two-substrate convergence ladder, not the
  ladder itself.

## Avoid

- "Sundog discovered a new atmospheric feature." (Most predicted
  features are textbook physics; the novelty is the predictive-overlay
  workflow, not the phenomena.)
- "The atlas is a spectrometer." (It is a parametric geometry model that
  *aims* a spectrometer.)
- "The gravity claim is now confirmed in the wild." (A positive Phase 4
  ratchets the two-substrate convergence one step; it does not
  retroactively prove the structural argument.)

## Pre-Registration Template

Every candidate prediction in this roadmap, before any apparatus is
pointed at the sky, ships with a written spec naming:

1. **Angular position(s)** the atlas predicts, in workbench coordinates
   and in real-sky coordinates relative to the sun.
2. **Detection regime** — polarization angle, narrowband intensity,
   long-exposure integration, spectroscopic dispersion, or some named
   combination.
3. **Predicted signal curve** — the explicit function (polarization
   angle vs. arc position, intensity vs. wavelength, etc.) the atlas
   math implies, with the source citation for the physics that connects
   geometry to signal.
4. **Smoke gate** — the signal threshold below which the apparatus is
   judged insufficient rather than the prediction falsified, with the
   noise-floor measurement protocol that defines "insufficient."
5. **Null condition** — what observation would actually count *against*
   the prediction, distinct from "we couldn't tell."
6. **Positive condition** — what observation would count *for* the
   prediction, with the discrimination test against confounders
   (e.g., scattered sky polarization at the observation angle).

This template is mandatory. A prediction that cannot fill all six slots
does not get an apparatus run.

## Roadmap

### Phase 0 — Scope Lock, Literature Spine, Actionable/Reach Sort

Goal: pin the apparatus tier, the candidate prediction slate, the
pre-registration template, and the actionable-vs-reach classification
before writing any new code or buying any new hardware.

Deliverables:

- **Apparatus tier locked at Smartphone-Only for Phase 1.** A rotatable
  linear polarizer film, a tripod, a smartphone with manual exposure
  control, and the existing pre-rendered atlas overlay matched to camera
  FOV. Total apparatus cost target: under $50 of polarizer film plus
  whatever tripod and phone are already on hand. Higher-tier apparatus
  (narrowband filters, long-exposure astro cameras, fiber-coupled
  spectrometers) is explicitly deferred to Phase 5 / Phase 6 promotions.
- **Candidate prediction slate** — three predictions, ranked by
  apparatus tier:
  - **Prediction 1 (Phase 1–4): Sub-visible parhelic circle
    continuation.** Polarization angle along the great-circle
    continuation of the parhelic arc beyond the photometric contrast
    drop-off. Smartphone-tier apparatus.
  - **Prediction 2 (Phase 5+, conditional): Intersection enhancement.**
    Integrated brightness at predicted circle-circle crossings should
    exceed the sum of each circle's local brightness by a factor
    predictable from prism-path overlap. Requires narrowband filtering
    and long-exposure capable camera.
  - **Prediction 3 (Phase 6+, conditional, reach): Color-order
    inversion at arc tangencies.** Chromatic dispersion gradient
    cancellation/reinforcement at precise angular positions where two
    arcs' color orderings meet. Requires a small calibrated
    spectrometer.
- **Actionable/reach sort,** written as the table below:

  | Item | Tier | Phase | Status |
  |---|---|---|---|
  | Methodology transfer from mesa (pre-registration, smoke gates) | Doc-only | 0 | Actionable |
  | Computing predicted polarization curve along parhelic continuation | Software | 1 | Actionable |
  | Smartphone + polarizer + tripod + overlay apparatus | Hardware ($30–50) | 1 | Actionable |
  | First observation run, cirrus afternoon | Field | 3 | Actionable, weather-gated |
  | Result note in mesa-style discipline | Doc-only | 4 | Actionable |
  | Narrowband filter + long-exposure camera apparatus | Hardware ($200–400) | 5 | Reach (Phase 4 promotes) |
  | Intersection-enhancement prediction & smoke gate | Software + Field | 5 | Reach |
  | Calibrated spectrometer apparatus | Hardware ($500–2000+) | 6 | Reach (Phase 5 promotes) |
  | Color-order inversion prediction & smoke gate | Software + Field | 6 | Reach |
  | "Atlas is a measurement instrument" public language | Doc + Public | 7 | Earned-language, gated by Phase 4 positive |
  | Polarimetric imager / filter wheel apparatus | Hardware ($2000+) | — | Reach, no committed phase |
  | Subhorizon arc instrumentation (requires observer above horizon) | Field + apparatus | — | Reach, no committed phase |

- **Literature spine** — short, structured, not a survey:
  - Greenler, *Rainbows, Halos, and Glories* (1980): canonical atlas
    geometry and ice-crystal refraction tables.
  - Tape, *Atmospheric Halos* (1994): exhaustive halo taxonomy and
    Parry-orientation chapter.
  - Cowley, *Atmospheric Optics* (online): canonical photographic
    references and modern halo nomenclature.
  - Können, *Polarized Light in Nature* (1985): ice-crystal polarization
    signatures and the standard derivation for parhelic-circle
    polarization geometry.
  - Lynch & Livingston, *Color and Light in Nature* (2nd ed., 2001):
    chromatic dispersion in halos, color-order conventions, and the
    standard treatment of arc tangencies.
  - Standard sky-polarization references (Coulson; Horváth et al.) for
    distinguishing crystal-refraction polarization from Rayleigh
    background.
  - HaloSim documentation (Cowley/Schroeder) as the comparison ceiling
    for any quantitative prediction.

- **Pre-registration template** (the six-slot spec above) ratified and
  committed to the doc.

Exit criterion: the apparatus tier, prediction slate, sort, literature
spine, and pre-registration template are pinned and tracked in this
file. No Phase 1 work begins until exit criterion is met.

### Phase 1 — Apparatus Build and Predicted-Curve Software

Goal: build the smartphone-tier apparatus and the software that computes
the predicted polarization curve along the parhelic-circle continuation
for any (sun-altitude, observer-azimuth, camera-FOV) tuple, so that
Phase 2's pre-registration can name the exact angular positions and
predicted signal values.

Deliverables:

- **Apparatus assembly**:
  - Rotatable linear polarizer film mount that accepts a standard
    smartphone lens. Marked with a fiducial scale so polarizer rotation
    angle is recorded with each exposure.
  - Tripod with phone clamp and altitude/azimuth read-off (a compass
    app plus a clinometer suffices; no theodolite required).
  - Calibration target: a known partial polarizer or a polarization
    reference card photographed under known conditions, used to verify
    the polarizer's measured-angle-vs-rotation curve.

- **Predicted-curve software** (`scripts/perception-predict-parhelic.mjs`
  or equivalent under `public/js/` if browser-renderable):
  - Inputs: sun altitude `h`, observer azimuth (relative to the sun),
    camera FOV, camera optical axis.
  - Outputs: the full great-circle path of the parhelic circle in the
    camera's pixel frame, the predicted polarization angle along that
    path at sample spacing `Δθ` (default 1°), and the predicted
    polarization degree using standard ice-crystal refraction
    coefficients from Können.
  - Sanity test: at the canonical Troels Nielsen pose (`h ≈ 25°`), the
    predicted polarization angle at the visible parhelion locations
    must match the standard literature value to within 1° (this is a
    sanity check against the implementation, not a novel prediction).

- **Atlas overlay extension** (`scripts/atlas-overlay-camera-fov.mjs` or
  inline in the workbench):
  - Render the predicted parhelic continuation as a thin curve on the
    workbench output, distinguishable from the visible parhelic arc
    (e.g., dashed, half-opacity, separately colored).
  - Output an FOV-matched PNG that can be loaded as a reference
    overlay against camera frames.

- **Scripted exposure series** — a short shell script or note in
  `docs/perception/APPARATUS_PROTOCOL.md` describing the exposure
  sequence: bracket the suspected continuation segment with N
  polarizer-rotation steps at fixed angular spacing, fixed shutter,
  fixed ISO, fixed focal length, against the FOV-matched overlay.

Exit criterion: the apparatus is buildable from the protocol doc by
someone who has not read this roadmap; the predicted-curve software
passes its sanity test against the Troels Nielsen pose; the workbench
can emit an FOV-matched continuation overlay.

### Phase 2 — Pre-Registration of Prediction 1

Goal: write the full six-slot pre-registration spec for the sub-visible
parhelic circle continuation, with the predicted curve computed and the
smoke gate quantified, *before* any observation run.

Deliverables:

- `docs/perception/PHASE2_PREDICTION_1.md` containing:
  1. **Angular positions:** the great-circle path in real-sky coordinates
     and the projected pixel path in camera coordinates for a target
     observation window (default: sun altitude 15°–30°, mid-afternoon
     high-cirrus). Continuation segments outside the visible parhelic
     arc are listed explicitly with start/end angular positions and
     the photometric contrast value at which the arc's visible
     portion drops off (estimated from the existing calibration photo
     set under `docs/calibration/`).
  2. **Detection regime:** polarization-angle modulation across rotated
     polarizer exposures; signal is the modulation amplitude at the
     predicted angular positions, integrated across the exposure
     series.
  3. **Predicted signal curve:** polarization-angle-vs-arc-position
     curve from the Phase 1 software, with the predicted modulation
     amplitude in normalized intensity units. The curve is committed
     before any observation.
  4. **Smoke gate:** noise floor measured at sky-background regions
     outside the predicted path; smoke gate is the apparatus-judged-
     insufficient threshold below which a null result is *not* taken
     as falsifying the prediction. Initial proposal: smoke gate at
     3× background modulation amplitude. Calibrated against the
     reference card before each observation run.
  5. **Null condition:** integrated modulation amplitude at the
     predicted continuation positions does not exceed background by
     ≥ 3× while the apparatus passes its smoke gate. Distinct from
     "apparatus failed" or "cirrus was too thin."
  6. **Positive condition:** integrated modulation amplitude at the
     predicted positions exceeds background by ≥ 5× *and* the
     polarization angle's variation along the path tracks the
     predicted curve to within an agreed angular tolerance
     (initial: ±10° along the segment, sharpened in Phase 3 if the
     first run is positive).

- **Discrimination against confounders:** explicit list of confounders
  (scattered-sky Rayleigh polarization at the observation angle; thin
  cloud structure mimicking a curved signal; reflective ground; the
  polarizer's own retardance) and the test that rules each one out.
  Rayleigh-background is the dominant confounder; the discrimination
  test is that the predicted parhelic-continuation polarization angle
  differs from the local Rayleigh polarization angle by a specific
  amount derivable from sky geometry, which the apparatus must
  resolve.

- **Pre-registration commitment:** the doc is committed to the repo
  with no observation data attached. Any later edit that softens the
  null or positive conditions must be filed as a separate amendment
  block with a written justification.

Exit criterion: `PHASE2_PREDICTION_1.md` lands, the predicted curve and
smoke gate are quantitative, and no observation has been run yet.

### Phase 3 — First Observation Run

Goal: run the smartphone-tier apparatus against high-cirrus weather
producing a visible parhelion, capture the scripted exposure series at
the polarizer-rotation steps Phase 2 named, and integrate the
modulation amplitude along the predicted continuation.

Deliverables:

- Raw exposure series committed to `docs/perception/observations/
  <date>_<location>/` along with the EXIF, polarizer rotation angle,
  and apparatus-state log per frame.
- An analysis notebook or script that reproduces the integration of
  modulation amplitude along the Phase 2 predicted path, using the
  Phase 1 software as the geometric reference.
- A run-card (`PHASE3_RUN_<date>.md`) recording the apparatus smoke-gate
  measurement, the cirrus condition assessment, the observation
  geometry, and whether the run is judged valid or rejected at the
  smoke-gate level.

Exit criterion: at least one observation run passes the smoke gate.
Phase 3 may require multiple weather-gated attempts; that is expected
and is not a failure. The exit criterion is one valid run, not one
successful prediction.

### Phase 4 — Result Note (Mesa-Style Discipline)

Goal: write the result note in the same shape as
[`mesa/PHASE6_V32_RESULTS.md`](mesa/PHASE6_V32_RESULTS.md) and adjacent
notes — clean structure, honest verdict, no rhetorical inflation.

Deliverables:

- `docs/perception/PHASE4_RESULTS.md` recording:
  - The Phase 2 pre-registration referenced verbatim.
  - The Phase 3 valid run(s) summarised by their run-cards.
  - The measured modulation amplitude at each predicted position,
    against the smoke gate, null condition, and positive condition
    Phase 2 named.
  - The verdict: **positive**, **partial** (signal but does not track
    the predicted curve closely enough), **null** (no signal above
    smoke gate), or **inconclusive** (apparatus failed smoke gate
    across all attempts).
  - The methodological lessons the run produced, regardless of
    verdict.
  - Cross-reference back to the cross-pollination memo at the top of
    this roadmap and to MESA_CROSSOVER_NOTE.md.

- Verdict-conditional next-step routing:
  - *Positive* → Phase 5 promoted (intersection enhancement).
  - *Partial* → Phase 2 refinement amendment; tighter prediction
    bounds, possibly an apparatus calibration upgrade.
  - *Null with clean smoke gate* → publishable negative; the
    sub-visible parhelic continuation prediction is falsified at the
    smartphone-tier apparatus. Phase 5 conditional on whether
    higher-tier apparatus could still rescue the prediction or whether
    the falsification generalises.
  - *Inconclusive* → apparatus-tier revisit; the smartphone tier may
    be the wrong floor.

Exit criterion: the result note lands with a verdict drawn from the
four above, and the verdict-conditional routing decision is made and
documented.

### Phase 5 — Apparatus Upgrade and Prediction 2 (Intersection Enhancement)

Goal: conditional on a Phase 4 positive or partial verdict, promote the
intersection-enhancement prediction and the narrowband / long-exposure
apparatus.

This phase is **gated**, not scheduled. It runs if and only if Phase 4
verdict promotes it. If Phase 4 returns a clean null, Phase 5 may still
run with a revised prediction that addresses the Phase 4 null, but the
methodological burden shifts: the new prediction must explain why the
sub-visible parhelic continuation was null while the new target is
expected to signal.

Deliverables (when gated open):

- Apparatus upgrade: narrowband filter (initial proposal: an H-α or
  similar narrowband for cleanest signal-vs-Rayleigh discrimination,
  selected against the predicted dispersion curve), long-exposure
  capable camera, optionally a small equatorial mount if the
  observation window exceeds a few minutes.
- Predicted-curve software extension to compute predicted intensity
  enhancement at circle-circle intersections, using prism-path overlap
  factors from Greenler/Tape.
- Pre-registration spec `PHASE5_PREDICTION_2.md` filling all six slots
  of the template, with the same discipline as Phase 2.
- One or more observation runs against the new prediction.
- Result note `PHASE5_RESULTS.md` in the same shape as Phase 4.

Exit criterion: the intersection-enhancement prediction has a verdict
drawn from positive / partial / null / inconclusive, with the result
note committed.

### Phase 6 — Reach Tier: Color-Order Inversion at Arc Tangencies

Goal: conditional on a Phase 5 positive or partial verdict, promote the
color-order inversion prediction and the spectrometer-tier apparatus.

This phase is **explicitly reach** territory by the doc's original
classification. It enters the roadmap only after both lower-tier
predictions have moved.

Deliverables (when gated open):

- Apparatus: a small calibrated spectrometer (fiber-coupled, sufficient
  to resolve the dispersion gradient at the predicted tangency
  position). Apparatus selection note documenting the cost/calibration
  tradeoffs.
- Predicted-curve software extension for chromatic dispersion
  gradient cancellation/reinforcement at named tangencies (22° halo
  red-inside / blue-outside meeting CZA's inverted ordering at the 46°
  halo top tangent, for example). The candidate tangencies are
  enumerated in `PHASE6_TANGENCY_INVENTORY.md`.
- Pre-registration spec `PHASE6_PREDICTION_3.md` filling all six slots
  of the template.
- Observation run(s) and result note `PHASE6_RESULTS.md`.

Exit criterion: the color-order inversion prediction has a verdict and
a result note, *or* the spectrometer apparatus cost is judged
unjustifiable against the cumulative Phase 4 + Phase 5 evidence, in
which case the phase is closed unrun with an explicit "not promoted"
note.

### Phase 7 — Earned-Language Ratchet and Public Artifact

Goal: turn the cumulative result into a defensible public claim, update
the gravity ledger and the public-facing sundog.cc surfaces, and ship a
public artifact only if there is something earned to show.

Deliverables (verdict-conditional):

- Update `docs/SUNDOG_V_GRAVITY.md` §The Gravity Claim §Two-substrate
  convergence: add the predictive-receipt anchor and link to
  `PHASE4_RESULTS.md` (and Phase 5 / Phase 6 results when applicable).
- Update `docs/presentation/claims-and-scope.md` §Unsupported Universal
  Claims if any Phase 4–6 result reframes what is and is not earned.
- Update `docs/PROMO_HIGHLIGHTS.md` if and only if Phase 4 (or later)
  earned-language survives the discipline review.
- Update `docs/SUNDOG_V_GEOMETRY.md` §What the atlas does NOT yet
  model: move any item that has now been instrumented to a separate
  §Predictions confirmed by perception section; leave the remaining
  items as both apology *and* prediction surface, with a forward
  pointer to this roadmap.
- A short interactive page at `perception.html` (sibling to
  `mesa.html`, `threebody.html`, `balance.html`, `mines.html`) showing
  the prediction-overlay workflow with the Phase 4 (and later) result
  as the load-bearing example. Reach: an in-browser playback of the
  modulation-amplitude integration against an interactive sun-altitude
  slider, with the Phase 4 observation run as the canonical example.

Avoid in Phase 7 language:

- "Sundog instrumentation discovered X."
- "The gravity claim is now confirmed in the wild."
- "Halo physics has been rewritten."

Use in Phase 7 language (verdict-conditional):

- "Sundog geometry, in the tested apparatus tier and weather regime,
  predicts a sub-visible parhelion-continuation feature that the
  apparatus confirms / does not confirm."
- "The atlas now ships a predictive overlay alongside its descriptive
  one."
- "The two-substrate convergence on the gravity-frame field-shape
  gains a third receipt: in-vitro (mesa), in-the-wild observational
  (atlas overlays), in-the-wild predictive (perception)." Only after a
  Phase 4 positive.

Exit criterion: the public-facing surfaces, the gravity ledger, and
this roadmap all use the strongest claim Phase 4 (and any later phases)
earned, and no stronger one. If Phase 4 returned a clean null, Phase 7
ships the publishable negative without rhetorical inflation.

---

## Downstream Promotions and Dependencies

This roadmap depends on:

- **`SUNDOG_V_GEOMETRY.md`** — the halo atlas, the inverse-inference
  calibration result, and the `public/poses/canonical-halo-atlas.json`
  pose are load-bearing for Phase 1's predicted-curve software. The
  atlas formulae are the geometric source-of-truth; this roadmap does
  not re-derive them.
- **`SUNDOG_V_MESA.md`** — Phase 6 methodology (pre-registration,
  smoke gates, mesa-style result-note discipline) is the template
  Phase 0–4 of this roadmap inherit. The four stacked methodological
  lessons from mesa v3.x apply here too: feature availability is not
  mechanism, variance is not mechanism, partial delivery of the right
  basis fails, single-component sufficiency fails. The atlas-as-
  instrument framing is the geometry-side answer to those lessons.

This roadmap unlocks:

- **Gravity ledger Candidate 12** (new, to be filed):
  *Predictive Atmospheric Instrumentation.* Promoted into this
  roadmap. The candidate slot in `SUNDOG_V_GRAVITY.md` is reserved
  for archival reference once Phase 0 lands.
- **Geometry Phase 12 (Atmospheric Variations)** and **Phase 13
  (Linked Description Mode)** become richer once the predictive
  overlay is built: alternative sky palettes can carry their own
  predicted-continuation overlays; the linked-description popovers
  can cite Phase 4 results where applicable.

This roadmap does **not** absorb:

- The Spacecraft / Conservation-Law / Fluid-Wake candidates in the
  gravity ledger. Those are independent physical-domain experiments
  that test the gravity claim in domains other than parhelion optics.
- The mesa-trap roadmap. Mesa attacks the in-silico falsification
  mode; perception attacks the in-the-wild predictive-receipt mode.
  Both ratchet the gravity claim by independent paths.

## Implementation Status

**Phase 0:** Not started. This roadmap is the Phase 0 commitment
artifact; the deliverables above (apparatus tier lock, prediction
slate, sort, literature spine, pre-registration template) are now
named here but not yet filed as separate sub-documents. The next move
is to file `docs/perception/APPARATUS_PROTOCOL.md` and lock the
literature spine in writing, which completes Phase 0.

**Phases 1–7:** Not started; gated on Phase 0 completion (Phases 1–4)
and on Phase 4 verdict (Phases 5–6) and on cumulative evidence
(Phase 7).
