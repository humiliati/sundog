/**
 * phase6-drag.mjs - drag-to-tune inverse bindings.
 *
 * Phase 6 keeps the binding model deliberately small: each drag handle writes
 * one existing slider parameter, then the normal workbench apply path redraws
 * every dependent primitive.
 */

import { phase3 } from "./parhelion-geometry.mjs";

const SVG_NS = "http://www.w3.org/2000/svg";
const SUN_W = { x: 500, y: 500 };
const HALO_22_W = phase3.HALO_22_RADIUS;
const PARHELIC_CURVE_PIXELS = 200;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function numberFromSlider(slider, fallback) {
  const value = Number.parseFloat(slider?.value);
  return Number.isFinite(value) ? value : fallback;
}

function numberFromCss(name, fallback) {
  const value = Number.parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue(`--${name}`)
  );
  return Number.isFinite(value) ? value : fallback;
}

function sliderDecimals(slider) {
  const samples = [slider?.step, slider?.min, slider?.max].filter(Boolean);
  return samples.reduce((max, sample) => {
    if (sample === "any") return max;
    const [, decimals = ""] = String(sample).split(".");
    return Math.max(max, decimals.length);
  }, 0);
}

function setSliderValue(slider, rawValue) {
  if (!slider) return;
  const min = Number.parseFloat(slider.min);
  const max = Number.parseFloat(slider.max);
  const step = Number.parseFloat(slider.step);
  const low = Number.isFinite(min) ? min : -Infinity;
  const high = Number.isFinite(max) ? max : Infinity;
  let value = clamp(rawValue, low, high);

  if (Number.isFinite(step) && step > 0) {
    const origin = Number.isFinite(min) ? min : 0;
    value = origin + Math.round((value - origin) / step) * step;
    value = clamp(value, low, high);
  }

  const decimals = sliderDecimals(slider);
  slider.value = decimals > 0 ? value.toFixed(decimals) : String(Math.round(value));
  slider.dispatchEvent(new Event("input", { bubbles: true }));
}

function svgPoint(svg, ev) {
  const point = svg.createSVGPoint();
  point.x = ev.clientX;
  point.y = ev.clientY;
  const ctm = svg.getScreenCTM();
  return ctm ? point.matrixTransform(ctm.inverse()) : null;
}

function parhelicYOffset(controls) {
  return numberFromSlider(
    controls.parhelicYOffsetSlider,
    numberFromCss("parhelic-y-offset-r22", -0.05)
  );
}

function parhelicBeltY(controls) {
  return SUN_W.y + HALO_22_W * parhelicYOffset(controls);
}

function parhelicCurvature(controls) {
  return numberFromSlider(
    controls.parhelicCurvatureSlider,
    numberFromCss("parhelic-curvature", 0.05)
  );
}

function createCircleHandle(layer, { className, radius, label, binding, cursor }) {
  const handle = document.createElementNS(SVG_NS, "circle");
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = label;
  handle.appendChild(title);
  handle.setAttribute("class", `phase6-handle ${className}`);
  handle.setAttribute("r", String(radius));
  handle.setAttribute("pointer-events", "all");
  handle.setAttribute("aria-label", label);
  handle.setAttribute("data-phase6-binding", binding);
  handle.setAttribute("role", "slider");
  handle.setAttribute("tabindex", "-1");
  handle.style.cursor = cursor;
  layer.appendChild(handle);
  return handle;
}

function beginDrag(svg, handle, binding, ev) {
  if (svg.dataset.geometryModel !== "halo_atlas") return false;
  if (typeof handle.setPointerCapture === "function" && ev.pointerId !== undefined) {
    handle.setPointerCapture(ev.pointerId);
  }
  handle.classList.add("is-dragging");
  svg.dataset.phase6Drag = binding;
  ev.preventDefault();
  return true;
}

function endDrag(svg, handle, ev) {
  if (typeof handle.releasePointerCapture === "function" && ev.pointerId !== undefined) {
    try {
      handle.releasePointerCapture(ev.pointerId);
    } catch {
      // The pointer may already be released if the browser cancelled capture.
    }
  }
  handle.classList.remove("is-dragging");
  delete svg.dataset.phase6Drag;
}

function disableDerivedCurvature(controls) {
  const toggle = controls.deriveToggle;
  if (!toggle?.checked) return;
  toggle.checked = false;
  toggle.dispatchEvent(new Event("change", { bubbles: true }));
}

export function enablePhase6Drag(svg, controls = {}) {
  if (!svg) return;
  const {
    sunAltitudeSlider,
    parhelicCurvatureSlider,
    parhelicYOffsetSlider,
    deriveToggle,
  } = controls;

  let layer = svg.querySelector(".layer-phase6-handles");
  if (!layer) {
    layer = document.createElementNS(SVG_NS, "g");
    layer.setAttribute("class", "layer-phase6-handles");
    svg.appendChild(layer);
  }
  layer.replaceChildren();

  const altitudeHandles = sunAltitudeSlider
    ? [
        {
          side: "left",
          sign: -1,
          el: createCircleHandle(layer, {
            className: "phase6-handle-left",
            radius: 28,
            label: "Drag left parhelion to set sun altitude",
            binding: "sun-altitude",
            cursor: "ew-resize",
          }),
        },
        {
          side: "right",
          sign: 1,
          el: createCircleHandle(layer, {
            className: "phase6-handle-right",
            radius: 28,
            label: "Drag right parhelion to set sun altitude",
            binding: "sun-altitude",
            cursor: "ew-resize",
          }),
        },
      ]
    : [];

  const apexHandle = parhelicCurvatureSlider
    ? createCircleHandle(layer, {
        className: "phase6-handle-parhelic-apex",
        radius: 24,
        label: "Drag parhelic arc apex to set curvature",
        binding: "parhelic-curvature",
        cursor: "ns-resize",
      })
    : null;

  function syncHandles() {
    const beltY = parhelicBeltY({ parhelicYOffsetSlider });

    if (sunAltitudeSlider) {
      const sunAlt = numberFromSlider(sunAltitudeSlider, 25);
      const offset = phase3.daggerOffset(sunAlt);
      for (const handle of altitudeHandles) {
        handle.el.setAttribute("cx", String(SUN_W.x + handle.sign * offset));
        handle.el.setAttribute("cy", String(beltY));
        handle.el.setAttribute("aria-valuemin", sunAltitudeSlider.min || "0");
        handle.el.setAttribute("aria-valuemax", sunAltitudeSlider.max || "60");
        handle.el.setAttribute("aria-valuenow", sunAltitudeSlider.value);
      }
    }

    if (apexHandle) {
      const curvature = parhelicCurvature({ parhelicCurvatureSlider });
      const apexY = beltY + PARHELIC_CURVE_PIXELS * curvature;
      apexHandle.setAttribute("cx", String(SUN_W.x));
      apexHandle.setAttribute("cy", String(apexY));
      apexHandle.setAttribute("aria-valuemin", parhelicCurvatureSlider.min || "0");
      apexHandle.setAttribute("aria-valuemax", parhelicCurvatureSlider.max || "1");
      apexHandle.setAttribute("aria-valuenow", parhelicCurvatureSlider.value);
    }
  }

  syncHandles();
  sunAltitudeSlider?.addEventListener("input", syncHandles);
  parhelicCurvatureSlider?.addEventListener("input", syncHandles);
  parhelicYOffsetSlider?.addEventListener("input", syncHandles);
  deriveToggle?.addEventListener("change", syncHandles);

  for (const handle of altitudeHandles) {
    let dragKind = null;
    const applyAltitudeDrag = (ev) => {
      const local = svgPoint(svg, ev);
      if (!local) return;
      const dx = Math.abs(local.x - SUN_W.x);
      let altitude;
      if (dx <= HALO_22_W) {
        altitude = 0;
      } else if (dx >= 2 * HALO_22_W) {
        altitude = 60;
      } else {
        altitude = (Math.acos(HALO_22_W / dx) * 180) / Math.PI;
      }
      setSliderValue(sunAltitudeSlider, altitude);
      ev.preventDefault();
    };
    handle.el.addEventListener("pointerdown", (ev) => {
      dragKind = beginDrag(svg, handle.el, `sun-altitude:${handle.side}`, ev) ? "pointer" : null;
    });
    handle.el.addEventListener("pointermove", (ev) => {
      if (dragKind !== "pointer") return;
      applyAltitudeDrag(ev);
    });
    handle.el.addEventListener("mousedown", (ev) => {
      if (ev.button !== 0 || dragKind) return;
      dragKind = beginDrag(svg, handle.el, `sun-altitude:${handle.side}`, ev) ? "mouse" : null;
    });
    document.addEventListener("mousemove", (ev) => {
      if (dragKind !== "mouse") return;
      applyAltitudeDrag(ev);
    });
    const endPointer = (ev) => {
      if (dragKind !== "pointer") return;
      dragKind = null;
      endDrag(svg, handle.el, ev);
    };
    const endMouse = (ev) => {
      if (dragKind !== "mouse") return;
      dragKind = null;
      endDrag(svg, handle.el, ev);
    };
    handle.el.addEventListener("pointerup", endPointer);
    handle.el.addEventListener("pointercancel", endPointer);
    document.addEventListener("mouseup", endMouse);
  }

  if (apexHandle) {
    let dragKind = null;
    const applyCurvatureDrag = (ev) => {
      const local = svgPoint(svg, ev);
      if (!local) return;
      const beltY = parhelicBeltY({ parhelicYOffsetSlider });
      const curvature = (local.y - beltY) / PARHELIC_CURVE_PIXELS;
      setSliderValue(parhelicCurvatureSlider, curvature);
      ev.preventDefault();
    };
    apexHandle.addEventListener("pointerdown", (ev) => {
      disableDerivedCurvature({ deriveToggle });
      dragKind = beginDrag(svg, apexHandle, "parhelic-curvature", ev) ? "pointer" : null;
    });
    apexHandle.addEventListener("pointermove", (ev) => {
      if (dragKind !== "pointer") return;
      applyCurvatureDrag(ev);
    });
    apexHandle.addEventListener("mousedown", (ev) => {
      if (ev.button !== 0 || dragKind) return;
      disableDerivedCurvature({ deriveToggle });
      dragKind = beginDrag(svg, apexHandle, "parhelic-curvature", ev) ? "mouse" : null;
    });
    document.addEventListener("mousemove", (ev) => {
      if (dragKind !== "mouse") return;
      applyCurvatureDrag(ev);
    });
    const endPointer = (ev) => {
      if (dragKind !== "pointer") return;
      dragKind = null;
      endDrag(svg, apexHandle, ev);
    };
    const endMouse = (ev) => {
      if (dragKind !== "mouse") return;
      dragKind = null;
      endDrag(svg, apexHandle, ev);
    };
    apexHandle.addEventListener("pointerup", endPointer);
    apexHandle.addEventListener("pointercancel", endPointer);
    document.addEventListener("mouseup", endMouse);
  }
}

export function enableParhelionDrag(svg, sunAltitudeSlider) {
  enablePhase6Drag(svg, { sunAltitudeSlider });
}
