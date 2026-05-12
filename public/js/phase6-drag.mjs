/**
 * phase6-drag.mjs — light Phase 6.
 *
 * Adds one drag handle per parhelion. Dragging a dagger updates
 * `--sun-altitude` via the existing slider machinery, so the whole atlas
 * re-derives through the same path a slider input would.
 *
 * Inverse-bind math (mirrors PHASE6_DRAG_CONSTRAINTS.md):
 *   h = arccos( R_22 / |drag_x − sun_x| )
 *
 * Clamps: parhelion-inside-halo → h=0;  offset ≥ 2·R₂₂ → h=60 (slider max).
 *
 * Only meaningful in halo_atlas mode (the only model where sun-altitude
 * drives dagger placement). CSS hides the handle layer outside that mode.
 */

import { phase3 } from "./parhelion-geometry.mjs";

const SVG_NS = "http://www.w3.org/2000/svg";
const SUN_W = { x: 500, y: 500 };       // workbench-space sun
const HALO_22_W = phase3.HALO_22_RADIUS; // = 220

export function enableParhelionDrag(svg, sunAltitudeSlider) {
  if (!svg || !sunAltitudeSlider) return;

  // ---- inject hit-test handle layer once -------------------------------
  let layer = svg.querySelector(".layer-phase6-handles");
  if (!layer) {
    layer = document.createElementNS(SVG_NS, "g");
    layer.setAttribute("class", "layer-phase6-handles");
    // Append last so handles sit above all other primitives in z-order.
    svg.appendChild(layer);
  }

  const handles = [
    { side: "left",  sign: -1, el: null },
    { side: "right", sign: +1, el: null },
  ];
  for (const h of handles) {
    h.el = document.createElementNS(SVG_NS, "circle");
    h.el.setAttribute("class", `phase6-handle phase6-handle-${h.side}`);
    h.el.setAttribute("r", "28");
    h.el.setAttribute("fill", "transparent");
    h.el.setAttribute("stroke", "transparent");
    h.el.setAttribute("pointer-events", "all");
    h.el.style.cursor = "ew-resize";
    h.el.setAttribute("aria-label", `Drag ${h.side} parhelion to set sun altitude`);
    h.el.setAttribute("role", "slider");
    h.el.setAttribute("tabindex", "-1");
    layer.appendChild(h.el);
  }

  // ---- keep handles glued to the rendered daggers ----------------------
  function syncHandles() {
    const sunAlt = Number.parseFloat(sunAltitudeSlider.value) || 0;
    const offset = phase3.daggerOffset(sunAlt);
    for (const h of handles) {
      h.el.setAttribute("cx", String(SUN_W.x + h.sign * offset));
      h.el.setAttribute("cy", String(SUN_W.y));
    }
  }
  syncHandles();
  sunAltitudeSlider.addEventListener("input", syncHandles);

  // ---- pointer drag handlers ------------------------------------------
  for (const h of handles) {
    let dragging = false;
    h.el.addEventListener("pointerdown", (ev) => {
      // Only drag in halo_atlas mode — the only model where the binding is
      // meaningful. In other models the handles are CSS-hidden anyway, but
      // belt-and-suspenders here.
      if (svg.dataset.geometryModel !== "halo_atlas") return;
      dragging = true;
      h.el.setPointerCapture(ev.pointerId);
      svg.dataset.phase6Drag = h.side;
      h.el.classList.add("is-dragging");
      ev.preventDefault();
    });
    h.el.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      // Convert clientX/Y → SVG-local coords so the math runs in workbench units.
      const pt = svg.createSVGPoint();
      pt.x = ev.clientX;
      pt.y = ev.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const local = pt.matrixTransform(ctm.inverse());
      const dx = Math.abs(local.x - SUN_W.x);

      let newH;
      if (dx <= HALO_22_W) {
        newH = 0;                                  // sun on the horizon
      } else if (dx >= 2 * HALO_22_W) {
        newH = 60;                                 // slider max
      } else {
        newH = (Math.acos(HALO_22_W / dx) * 180) / Math.PI;
        newH = Math.max(0, Math.min(60, newH));
      }
      // The slider's step is integer, so round to match its native increments.
      sunAltitudeSlider.value = String(Math.round(newH));
      sunAltitudeSlider.dispatchEvent(new Event("input"));
    });
    const end = (ev) => {
      if (!dragging) return;
      dragging = false;
      try { h.el.releasePointerCapture(ev.pointerId); } catch { /* no-op */ }
      delete svg.dataset.phase6Drag;
      h.el.classList.remove("is-dragging");
    };
    h.el.addEventListener("pointerup", end);
    h.el.addEventListener("pointercancel", end);
  }
}
