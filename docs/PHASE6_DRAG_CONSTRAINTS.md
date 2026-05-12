# Phase 6 Constraint Network — Drag-to-Tune

Planning doc for the "click and drag a primitive, the whole atlas re-derives"
interaction. Companion to the Phase 6 entry in
[`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md).

This doc is **plan-only**. No JS shipped yet. The architecture below is what
Phase 6 will implement; the questions at the bottom are what to settle
before writing the first handler.

## Approach: one handle, one parameter

The atlas's parameters form a small dependency network rooted at four
inputs: `R₂₂` (anchor scale), sun pixel position, `sun-altitude`, and the
two curvature sliders (`parhelic-curvature`, `cza-curvature`). Every visible
primitive is a function of those four.

Drag interactions inverse-bind **one handle to one parameter at a time**.
No generalized constraint solver. Dragging causes the bound parameter to
update, and the rest of the atlas re-derives via the Phase 3 binding
network already in place.

This is much cheaper than a global solver, and it has a clean pedagogical
story for the explainer page: "grab the bright spot, drag it out, watch the
sun's altitude rise." The user sees the field-and-signature relationship in
the most direct way possible.

## Inverse-bind table

| handle (drag target) | binds primary parameter | inverse formula | dependent primitives that re-derive |
|---|---|---|---|
| **Parhelic-arc apex** | `parhelic-curvature` | solve for apex height `h_apex` given dragged y, then `c = h_apex / 200` clamped to [0, 1] | parhelic-circle stroke only |
| **Parhelion / dagger position** | `sun-altitude` | `h = arccos(R₂₂ / |drag_x − sun_x|)` | both daggers (mirror), parhelic-circle endpoints, CZA visibility (hides above 32°), supralateral, infralateral |
| **CZA apex** | `cza-curvature` | solve for `c` such that `czaApex(c) = drag_y` (linear inverse) | CZA primary + secondary arcs, bell-fill |
| **Sun center** | sun pixel position (translation) | new sun_px = drag position | every primitive translates as a rigid body |
| **22° halo edge** | `R₂₂` (anchor scale) | `R₂₂ = √((drag_x − sun_x)² + (drag_y − sun_y)²)` | every primitive rescales via the workbench→photo scale factor |

### Detail: parhelic-arc apex inverse

The parhelic-arc apex sits at `(sun_x, sun_y + d)` where `d = 200·c` and the
parhelic curvature `c ∈ [0, 1]`. Dragging the apex to a new y:

```
new_d = drag_y − sun_y           (positive = below sun)
new_c = clamp(new_d / 200, 0, 1)
```

Constraint: only the y-component of the drag counts. If the user drags
sideways, the apex stays on the `x = sun_x` axis. The handle's visual
position snaps to the constrained projection during drag.

### Detail: parhelion / dagger inverse

The right parhelion sits at `(sun_x + R₂₂/cos(h), sun_y)` on the parhelic
belt. Dragging it to a new position:

```
offset = |drag_x − sun_x|
new_h_rad = arccos(min(1, R₂₂ / offset))   (clamp avoids NaN at offset < R₂₂)
new_h_deg = degrees(new_h_rad)
```

Symmetric: dragging the *left* dagger updates `sun-altitude` identically;
the right dagger mirrors visually. Only x matters (the belt is constrained
to `y = sun_y + parhelic-y-offset`).

### Detail: CZA apex inverse

The CZA apex is anchored to the 46° halo top:
`czaApexY = (sun_y − R₄₆) + (0.85 − czaCurve)·200`.

Dragging the apex to drag_y:

```
new_curve = 0.85 − (drag_y − (sun_y − R₄₆)) / 200
new_curve = clamp(new_curve, 0.4, 1.4)
```

Snaps to slider range. CZA primary + secondary + bell-fill all re-render
with the new curvature.

### Detail: sun-center drag

Pure translation. New sun pixel = drag position. Atlas scale (`R₂₂`) and
sun altitude (`h`) are unchanged. Every primitive's center recomputes as
`new_center = old_center + (drag_pos − old_sun_pos)`.

Useful for fixing the "auto-detected sun is actually a parhelion" failure
mode from the multi-photo calibration pass (cf. p8).

### Detail: 22° halo edge drag

Drag any point on the visible 22° halo ring → resize the atlas by the
chord from sun_center to drag_point. New `R₂₂ = √((drag_x − sun_x)² +
(drag_y − sun_y)²)`. Scale factor changes; every primitive rescales.

Sun altitude is unchanged. Dagger positions update (their offset is
`R₂₂ / cos(h)`, so the offset in pixels grows as R₂₂ grows).

## SVG hit-test handle architecture

Each draggable primitive needs an invisible hit-test handle in SVG that's
larger than the visible stroke for comfortable touch targets. Standard
pattern:

```html
<g class="layer-parhelic">
  <!-- the visible stroke -->
  <path id="parhelic-path" stroke="..." />
  <!-- the hit-test handle, transparent, fat -->
  <circle class="handle handle-parhelic-apex"
          cx="…" cy="…" r="18"
          fill="transparent" stroke="transparent"
          pointer-events="all" />
</g>
```

The handle's position is the **current rendered apex location**, computed
the same way the visible primitive uses. As the parameter updates during
drag, the handle moves with the primitive.

## Event flow

```
pointerdown on .handle-X
  → save start state (current pose, drag offset)
  → set svg.setPointerCapture
  → svg.dataset.activeDrag = "handle-X"
  → body.classList.add("is-dragging-X")   // cursor feedback

pointermove (only while activeDrag set)
  → compute new parameter value via inverse formula
  → update the bound CSS variable on document.documentElement
  → call applyParhelionGeometry() which re-renders everything
  → handle's own position updates as part of re-render

pointerup / pointercancel
  → snapshot final pose to localStorage (optional)
  → svg.releasePointerCapture
  → clear activeDrag and cursor class
```

`prefers-reduced-motion` doesn't disable drag (user-initiated motion), but
any *transitions* on dependent primitives become instant. The primitives
themselves don't animate during drag — they just re-render at the new
parameter value each frame.

## Slider sync

While a handle is being dragged, the corresponding slider's value display
updates live. The slider's `value` attribute also updates so a subsequent
slider drag picks up the dragged value, not the pre-drag value. The slider
itself doesn't need to fire an `input` event — the geometry already updated
when we set the CSS variable.

## Edge cases and snap behavior

- **Parhelion offset < R₂₂.** Mathematically impossible for `h ≥ 0°`.
  Clamp `new_h` to 0 and snap the dragged dagger to the halo edge with a
  small "you can't go closer than the halo" visual cue (handle wiggles
  back).
- **Sun-altitude > slider max (60°).** Clamp and snap.
- **CZA apex dragged above canvas.** Clamp to `czaCurve ≤ 1.4` (slider
  max).
- **22° halo edge drag → R₂₂ < 30 px.** Clamp to a sane minimum so the
  atlas doesn't collapse to a point.
- **Touch + mouse together.** Only one handle is active at a time; new
  pointerdown cancels any in-flight drag.

## What this plan does NOT cover

- Multi-touch transforms (rotate, pinch-zoom the whole atlas). The five
  handles are sufficient for the pedagogical job; multi-touch is
  decoration.
- Animation during drag (smoothly interpolated transitions on dependent
  primitives). Phase 6 does immediate re-render; smoothing is a Phase 7+
  polish item.
- Undo/redo. A "Reset to pose" button handles single-step recovery; full
  history is overscope.
- Keyboard interaction. Each handle gets `aria-label` and focusability,
  but keyboard-driven parameter editing happens through the sliders, not
  through the handles.

## Open questions before implementation

1. **Which mode(s) get drag-to-tune?** The atlas mode is the obvious
   target. Should `halo_governed` and `halo_scaffold` also be drag-enabled
   for comparison, or do we restrict to atlas?
2. **Photo-overlay mode** (Phase 5 upload): when an uploaded photo is the
   stage, do all five handles still work, or do we restrict to sun-center
   + R₂₂-edge + parhelion (the three measurements the user marked at
   upload time)? Restricting matches the upload UX better.
3. **Mobile tap-targets.** 18px radius is comfortable on desktop, may be
   too small on phones. Consider 24px on touch devices via pointer-type
   detection.
4. **Snap-to-canonical button.** A "snap back to canonical pose" button
   that animates from dragged state to the canonical-halo-atlas pose
   would close the loop pedagogically — show the user where the
   well-calibrated pose lives without making them undo every drag.

## Implementation order when Phase 6 starts

1. Add hit-test handles to the SVG for all five primitives.
2. Pointer event plumbing for the simplest case: sun-center drag (pure
   translation).
3. Add R₂₂ edge drag (uniform scale).
4. Add parhelion drag → sun-altitude binding (the most pedagogically
   important one).
5. Add parhelic-apex drag → curvature binding.
6. Add CZA-apex drag → CZA-curvature binding.
7. Slider sync + value display update during drag.
8. Edge-case clamps, snap behavior, accessibility polish.
9. Mobile / touch tuning.
