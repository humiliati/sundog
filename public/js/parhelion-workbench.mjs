import { applyParhelionGeometry, phase3 } from "./parhelion-geometry.mjs";
import { enablePhase6Drag } from "./phase6-drag.mjs";

const root = document.documentElement;
const rootStyle = () => getComputedStyle(root);
const svg = document.getElementById("parhelion-svg");

const MODEL_STORAGE_KEY = "sundog.geometryModel";

const VALID_MODELS = new Set(["legacy", "halo_scaffold", "halo_governed", "halo_atlas"]);

function getModel() {
  const saved = window.localStorage.getItem(MODEL_STORAGE_KEY);
  return VALID_MODELS.has(saved) ? saved : "halo_atlas";
}

function setModel(model) {
  window.localStorage.setItem(MODEL_STORAGE_KEY, model);
  apply();
}

function setParam(name, value) {
  root.style.setProperty(`--${name}`, value);
  updateParamDisplay(name, value);
  apply();
}

function updateParamDisplay(name, value) {
  const display = document.getElementById(`${name}-value`);
  if (display) {
    const num = Number.parseFloat(value);
    display.textContent = Number.isInteger(num) ? num.toString() : num.toFixed(2);
  }
}

function apply() {
  applyParhelionGeometry({ svg, rootStyle: rootStyle(), model: getModel() });
}

// --- Slider plumbing ------------------------------------------------------

const sliders = document.querySelectorAll('input[type="range"][data-param]');
sliders.forEach((slider) => {
  updateParamDisplay(slider.dataset.param, slider.value);
  slider.addEventListener("input", () => setParam(slider.dataset.param, slider.value));
});

// --- Model toggle ---------------------------------------------------------

const modelSelect = document.getElementById("geometry-model");
if (modelSelect) {
  modelSelect.value = getModel();
  modelSelect.addEventListener("change", () => setModel(modelSelect.value));
}

// --- Phase 3: derive parhelic curvature from altitude --------------------

const deriveToggle = document.getElementById("parhelic-curvature-derive");
const curvatureSlider = document.getElementById("parhelic-curvature");

function refreshDeriveState() {
  if (!deriveToggle || !curvatureSlider) return;
  const on = deriveToggle.checked;
  root.style.setProperty("--parhelic-curvature-derive", on ? "1" : "0");
  curvatureSlider.disabled = on;
  curvatureSlider.parentElement?.classList.toggle("is-derived", on);
  if (on) {
    const sunAlt = Number.parseFloat(rootStyle().getPropertyValue("--sun-altitude")) || 25;
    const derived = phase3.parhelicCurvature(sunAlt);
    curvatureSlider.value = derived.toFixed(2);
    updateParamDisplay("parhelic-curvature", curvatureSlider.value);
  }
  apply();
}

if (deriveToggle) {
  deriveToggle.addEventListener("change", refreshDeriveState);
  const sunAltSlider = document.getElementById("sun-altitude");
  sunAltSlider?.addEventListener("input", () => {
    if (deriveToggle.checked) refreshDeriveState();
  });
  refreshDeriveState();
}

// --- Reset / snapshot -----------------------------------------------------

const reset = document.getElementById("btn-reset");
reset?.addEventListener("click", () => {
  sliders.forEach((slider) => {
    slider.value = slider.defaultValue;
    slider.dispatchEvent(new Event("input"));
  });
  if (deriveToggle) {
    deriveToggle.checked = false;
    refreshDeriveState();
  }
});

const snapshot = document.getElementById("btn-snapshot");
snapshot?.addEventListener("click", () => {
  const params = { geometryModel: getModel() };
  sliders.forEach((slider) => {
    params[slider.dataset.param] = Number.parseFloat(slider.value);
  });
  if (deriveToggle) params.parhelicCurvatureDerive = deriveToggle.checked;
  console.log("Sundog params snapshot:", JSON.stringify(params, null, 2));
  try {
    navigator.clipboard.writeText(JSON.stringify(params, null, 2));
  } catch {
    // ignore
  }
});

// --- Phase 6: drag rendered primitives back into bound parameters ---------
const sunAltSlider = document.getElementById("sun-altitude");
const parhelicYOffsetSlider = document.getElementById("parhelic-y-offset-r22");
if (svg) {
  enablePhase6Drag(svg, {
    sunAltitudeSlider: sunAltSlider,
    parhelicCurvatureSlider: curvatureSlider,
    parhelicYOffsetSlider: parhelicYOffsetSlider,
    deriveToggle: deriveToggle,
  });
}

// Initial layout
apply();
