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

// --- Phase 8A: Load Params (file picker) -> hydrate sliders ---------------
// Accepts both pose-JSON schemas in the wild:
//   - kebab-case (snapshot output): { "sun-altitude": 25, ... }
//   - camelCase (canonical named poses): { "sunAltitudeDeg": 25, ... }

const CAMEL_OVERRIDES = {
  "sun-altitude": "sunAltitudeDeg",
  "parhelic-y-offset-r22": "parhelicYOffsetR22",
  "compass-rotation-deg": "compassRotationDeg",
};

function kebabToCamel(s) {
  return s.replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase());
}

function findPoseValue(params, kebab) {
  if (Object.prototype.hasOwnProperty.call(params, kebab)) return params[kebab];
  const override = CAMEL_OVERRIDES[kebab];
  if (override && Object.prototype.hasOwnProperty.call(params, override)) return params[override];
  const camel = kebabToCamel(kebab);
  if (Object.prototype.hasOwnProperty.call(params, camel)) return params[camel];
  return undefined;
}

function applyPose(params) {
  if (!params || typeof params !== "object") return { hydrated: 0, skipped: 0, errors: ["not an object"] };
  let hydrated = 0;
  let skipped = 0;
  const errors = [];

  if (typeof params.geometryModel === "string" && VALID_MODELS.has(params.geometryModel)) {
    if (modelSelect) modelSelect.value = params.geometryModel;
    setModel(params.geometryModel);
  } else if (params.geometryModel) {
    errors.push(`unknown geometryModel: ${params.geometryModel}`);
  }

  sliders.forEach((slider) => {
    const v = findPoseValue(params, slider.dataset.param);
    if (v === undefined || v === null) { skipped += 1; return; }
    const num = Number(v);
    if (!Number.isFinite(num)) { skipped += 1; errors.push(`${slider.dataset.param}: non-numeric ${JSON.stringify(v)}`); return; }
    slider.value = String(num);
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    hydrated += 1;
  });

  if (deriveToggle && typeof params.parhelicCurvatureDerive === "boolean") {
    deriveToggle.checked = params.parhelicCurvatureDerive;
    refreshDeriveState();
  }

  return { hydrated, skipped, errors };
}

const loadBtn = document.getElementById("btn-load");
const loadFile = document.getElementById("btn-load-file");
loadBtn?.addEventListener("click", () => loadFile?.click());
loadFile?.addEventListener("change", async (ev) => {
  const file = ev.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    const params = JSON.parse(text);
    const result = applyPose(params);
    console.log(`Sundog Load Params: hydrated ${result.hydrated} sliders, skipped ${result.skipped}, errors ${result.errors.length}`, result);
    if (result.errors.length) {
      console.warn("Load Params errors:", result.errors);
    }
  } catch (err) {
    console.error("Load Params failed:", err);
    alert(`Could not load that pose JSON: ${err.message || err}`);
  } finally {
    loadFile.value = "";
  }
});

// --- Phase 6: drag rendered primitives back into bound parameters ---------
const sunAltSlider = document.getElementById("sun-altitude");
const parhelicYOffsetSlider = document.getElementById("parhelic-y-offset-r22");
const czaCurvatureSlider = document.getElementById("cza-curvature");
const czaIntensitySlider = document.getElementById("cza-intensity");
if (svg) {
  enablePhase6Drag(svg, {
    sunAltitudeSlider: sunAltSlider,
    parhelicCurvatureSlider: curvatureSlider,
    parhelicYOffsetSlider: parhelicYOffsetSlider,
    czaCurvatureSlider: czaCurvatureSlider,
    czaIntensitySlider: czaIntensitySlider,
    deriveToggle: deriveToggle,
  });
}

apply();
