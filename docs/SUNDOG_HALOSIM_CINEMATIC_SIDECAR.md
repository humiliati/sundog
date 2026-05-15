# Sundog HaloSim Cinematic Sidecar (HS-0 … HS-7)

*Filed 2026-05-14. Split out of [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md)
the same day to keep the main roadmap navigable. This is a **sidecar**, not
a main-ladder phase: it does not gate hero promotion and is not in the
Phase 0–12 dependency chain.*

This document is the **generative / cinematic** use of HaloSim,
deliberately distinct from Phase 12A/12B in the parent roadmap, which use
HaloSim as a **validation oracle**. Same binary, opposite direction:
Phase 12 asks "is the atlas right?"; this sidecar asks "can HaloSim render
a beautiful, labelled, sun-altitude-swept halo film the atlas can
narrate?" Outputs feed Phase 7 (press / social snapshot assets), Phase 11
(characterized animation toolkit), and Phase 4 (active-reveal polish) — it
does not replace any of them.

**Cross-references.** Phase numbers (Phase 4 / 7 / 10 / 11 / 12A / 12B),
the §5 atlas vocabulary table, and the Actionability Audit referenced
below all live in [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md). The
HaloSim measurement procedure this sidecar reuses is
[`HALOSIM_VALIDATION_PROTOCOL.md`](calibration/HALOSIM_VALIDATION_PROTOCOL.md).

## Why this is realistic — and where it isn't

Grounded in the actual install, not assumption:

- **HaloSim has no CLI.** `C:\Users\hughe\HalSim361.exe` is a 2004 VB6
  GUI app. There is no documented headless/batch flag. Every render in
  `docs/calibration/halosim_outputs/` to date was produced by manual GUI
  operation.
- **But the input is already 100% file-driven.** A `.sim` file fully
  encodes the simulation: crystal blocks, orientation `.xng`, material,
  **sun elevation**, ray count, and view type. Sweeping sun altitude is
  a pure text-templating problem (HS-1).
- **And the output lands at a known path.** HaloSim auto-writes
  `autosave.bmp` (+ `autosav*.sim`) on render — observed in
  `halosim_outputs/`. Automation never has to drive a Save-As dialog;
  it drives *open-sim → Run* and harvests a fixed filename.
- **Assembly + measurement tooling already exists.** `ffmpeg 8.1` and
  `PIL 12.2` are installed; the `tangent_halosim_*` scripts and
  [`HALOSIM_VALIDATION_PROTOCOL.md`](calibration/HALOSIM_VALIDATION_PROTOCOL.md)
  already implement the 22°-halo scale-lock and feature-locus extraction
  that HS-3/HS-4 reuse.
- **AutoHotkey is not installed.** The tier-appropriate GUI driver for a
  native desktop app is the computer-use MCP (or `pywinauto` if a
  dependency is acceptable). The HS-0 spike picks the mechanism.

**The crux risk is not rendering — it is the auto-zoom.** HaloSim's
"Camera View" (`Type9`) auto-zooms to fit the active crystal-block
configuration, so the **pixel-to-degree scale changes frame to frame**
even at the same FOV (documented in the validation protocol's gotcha 1
and scale table). A naive frame stack will "breathe": the 22° halo
pulsing in size as features enter/leave. HS-3 (per-frame
scale-normalization to a common px/°, recenter on sun, crop to a fixed
canvas) is the load-bearing phase. If HS-3 cannot be made stable, the
sidecar still ships via the atlas-only fallback (below).

**What is *not* realistic, stated plainly:**

- **Live in-browser HaloSim raytrace.** Monte Carlo ray-tracing cannot
  run interactively in a page. The hero is an **honest hybrid**: the
  live interactive layer is the existing **atlas SVG** (already
  parametric across sun altitude, already labelled by the §5
  vocabulary); the HaloSim contribution is a **pre-rendered beauty
  video** scrubbed in lockstep by the same sun-altitude slider (HS-6).
- **A literal 0–90° sweep.** Sun-at-zenith is geometrically degenerate
  (halos go concentric, the two parhelia merge into the sun). **Ratified
  scope: 0–60°** — the visually rich band (parhelia clearly separated,
  CZA/tangent arcs strong, no zenith collapse). ≈ 31 frames at 2° steps.
- **"Modded HaloSim."** No source exists (2004 freeware). "Modded"
  means curated `.sim` / material `.xmt` / orientation `.xng` recipes
  and high ray counts (HS-7) — config curation, not a code fork.

## Ratified scope decisions (2026-05-14)

- **Altitude range:** 0–60°, ~2° step (~31 frames). Honest framing; no
  zenith-degeneracy caveat needed in published assets.
- **Commitment posture:** full HS-0…HS-7 committed, but **HS-0 is a hard
  gate** — HS-1 onward are blocked until the automation spike passes.
  The **atlas-only animated/labelled mode** is the named fallback if the
  2004 binary cannot be automated acceptably.

## HS-0 — Automation feasibility spike *(hard gate; blocks HS-1+)*

Goal: prove HaloSim can be driven to render a single `.sim` to disk with
**zero manual clicks**, or conclude it cannot and fall back.

Deliverables:

- Spike, in priority order: (a) test whether
  `HalSim361.exe "<path>.sim"` accepts the sim as a launch argument and
  whether any `Halosim3.ini` field triggers autorun; (b) if not, a
  minimal computer-use (or `pywinauto`) script that does *launch → open
  `.sim` → Run → wait → confirm fresh `autosave.bmp`*; (c) timing /
  reliability characterization (render-complete detection: poll
  `autosave.bmp` mtime + size-stable, with a saturation sanity check
  reusing the protocol's sun-detection box).
- A short feasibility note appended here with the chosen mechanism, the
  per-render wall-clock at the chosen ray count, and the failure modes
  seen.

Gate: one `.sim` rendered to a verified non-blank BMP with no human
interaction, repeatable 5×. **If the gate fails:** record the failure
mode here, mark HS-1…HS-7 *abandoned-as-automated*, and re-scope the
sidecar to the **atlas-only fallback** — an autoplay sun-altitude sweep
of the existing atlas SVG with §5 labels as on-canvas callouts (no
HaloSim beauty layer; HS-4/HS-5/HS-6 still apply to the atlas raster).

## HS-1 — `.sim` frame generator *(blocked on HS-0)*

Goal: deterministically emit one `.sim` per frame for the 0–60° sweep.

Deliverables: a generator (Node or Python) that takes a template `.sim`
(the Pass C7 column-only template,
[`halosim_outputs/halosim_p2_h18.6_columnonly.sim`](calibration/halosim_outputs/halosim_p2_h18.6_columnonly.sim),
or a full multi-block template for the rich display), the altitude list,
ray count, and crystal-block selection, and writes
`hs_frame_<HHH>deg.sim` (e.g. `hs_frame_026deg.sim`). Pins ray count and
view fields; only sun elevation varies per frame.

Gate: every emitted `.sim` opens in HaloSim without a parse error
(spot-check 3 across the range under HS-0's mechanism).

## HS-2 — Automated batch render runner *(blocked on HS-0)*

Goal: render the full frame set unattended.

Deliverables: a runner that, per `.sim`, invokes HS-0's mechanism,
harvests `autosave.bmp` → `hs_frame_<HHH>deg.bmp`, verifies (exists,
size-stable, non-blank, sun present via the saturation box), converts
BMP→PNG via PIL, and logs a per-frame receipt (rays, wall-clock,
verify result). Resumable; retries a failed frame once.

Gate: a 0–60° set rendered with zero manual interaction; receipt table
on disk; ≤1 frame requiring a manual rerun.

## HS-3 — Scale-normalization & canvas lock *(the crux)*

Goal: kill the auto-zoom "breathing" so frames compose into a smooth
clip.

Deliverables: lift the protocol's 22°-halo radial-profile scale-lock
into a reusable `scripts/halosim_normalize.py` — per frame: detect the
22° halo radius, compute that frame's px/°, rescale to a project-wide
constant px/°, recenter on the detected sun, crop/pad to a fixed canvas.
Emits `hs_frame_<HHH>deg.norm.png`.

Gate: across the full set the 22° halo radius is constant within ±1 px
and the sun center is fixed within ±1 px (regression table committed
beside the frames). Failure here ⇒ escalate to fixed-FOV view-type
investigation before HS-5.

## HS-4 — Atlas label overlay per frame *(blocked on HS-0)*

Goal: every frame carries correct atlas labels for that frame's `h`.

Deliverables: reuse `public/js/parhelion-geometry.mjs` (or its Python
twin in the calibration scripts) to compute each named primitive's
position at the frame's altitude, draw labelled callouts onto the
normalized raster, and *cross-check label placement against HaloSim's
own rendered feature* (a label that doesn't sit on its HaloSim feature
is a bug in either the scale-lock or the binding). Vocabulary and
status semantics come from the §5 atlas table; only visibility-promoted
+ conditional-core primitives are labelled by default (Phase 10
verdict), rare vocabulary behind a flag.

Gate: on ≥5 spot-checked frames spanning the range, every default-on
label lands on its HaloSim feature within the HS-3 tolerance.

## HS-5 — Assembly (GIF / MP4 / WebM) *(blocked on HS-0)*

Goal: stitch normalized labelled frames into shippable media.

Deliverables: an `ffmpeg` assembly step producing a looping WebM/MP4
(primary) and a size-bounded GIF (fallback), plus the press aspect
ratios Phase 7 already calls for (16:9, 1:1, 9:16). Frame cadence and
ease documented.

Gate: a smooth 0–60° clip with no visible scale-breathing, labels
legible at 1:1 and 9:16, file sizes within the Phase 7 asset budget.

## HS-6 — Hero hybrid integration *(blocked on HS-0; relates to Phase 4/7)*

Goal: the honest interactive hero — live atlas SVG + optional
pre-rendered HaloSim beauty layer, one slider.

Deliverables: a "cinematic" toggle on the `sundog.html` / `index.html`
hero that scrubs the HS-5 pre-render in lockstep with the existing
sun-altitude slider while the atlas SVG and §5 labels stay live on top;
`prefers-reduced-motion` pauses the video layer but keeps the SVG
interactive; the beauty layer is lazy-loaded and entirely optional
(atlas SVG is fully functional without it).

Gate: moving the sun-altitude slider on the hero advances both the live
atlas SVG and the HaloSim layer in sync; with the toggle off or the
video unloaded, the hero is unchanged from its Phase 7 state.

## HS-7 — "Modded" beauty recipe library *(blocked on HS-0; feeds Phase 11)*

Goal: a curated set of high-quality `.sim` recipes for press-grade
renders.

Deliverables: a documented recipe folder — tuned crystal-block mixes,
`Water-Ice.xmt` / orientation choices, color vs b&w, and high ray
counts (4–6M+) — reproducing several signature displays (canonical
parhelion, rich Parry display, pyramidal odd-radius set). Each recipe
names its provenance (Tape chapter / HaloSim bundled `.sim` /
historical display) consistent with the §5 vocabulary citations.

Gate: a third party can regenerate any library still from its recipe +
the HS-2 runner, and the outputs are press-quality (≥4M rays, clean
geometry, no Monte Carlo asymmetry).

## Fallback ledger

If HS-0 fails: HS-1/HS-2/HS-7 are dropped (no automated HaloSim), HS-3
narrows to "scale-lock any *manually* rendered HaloSim stills" for
Phase 7 hero stills only, and HS-4/HS-5/HS-6 retarget the **atlas SVG
raster** — an autoplay 0–60° sweep of the existing parametric atlas
with §5 labels, assembled by `ffmpeg`, embedded as the hero cinematic.
That fallback is entirely within project control and depends on no
external binary, so the sidecar ships either way; only the
ray-traced-beauty layer is at risk.
