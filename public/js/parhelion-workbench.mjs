import { applyParhelionGeometry } from "./parhelion-geometry.mjs";

const root = document.documentElement;
const rootStyle = () => getComputedStyle(root);
const svg = document.getElementById("parhelion-svg");

const MODEL_STORAGE_KEY = "sundog.geometryModel";

const VALID_MODELS = new Set(["legacy", "halo_scaffold", "halo_governed", "halo_atlas"]);

function getModel() {
  const saved = window.localStorage.getItem(MODEL_STORAGE_KEY);
  return VALID_MODELS.has(saved) ? saved : "legacy";
}

function setModel(model) {
  window.localStorage.setItem(MODEL_STORAGE_KEY, model);
  apply();
}

function setParam(name, value) {
  root.style.setProperty(`--${name}`, value);
  const display = document.getElementById(`${name}-value`);
  if (display) {
    const num = Number.parseFloat(value);
    display.textContent = Number.isInteger(num) ? num.toString() : num.toFixed(2);
  }
  apply();
}

function apply() {
  applyParhelionGeometry({ svg, rootStyle: rootStyle(), model: getModel() });
}

// --- Slider plumbing ------------------------------------------------------

const sliders = document.querySelectorAll('input[type="range"][data-param]');
sliders.forEach((slider) => {
  slider.addEventListener("input", () => setParam(slider.dataset.param, slider.value));
});

// --- Model toggle ---------------------------------------------------------

const modelSelect = document.getElementById("geometry-model");
if (modelSelect) {
  modelSelect.value = getModel();
  modelSelect.addEventListener("change", () => setModel(modelSelect.value));
}

// --- Reset / snapshot -----------------------------------------------------

const reset = document.getElementById("btn-reset");
reset?.addEventListener("click", () => {
  sliders.forEach((slider) => {
    slider.value = slider.defaultValue;
    slider.dispatchEvent(new Event("input"));
  });
});

const snapshot = document.getElementById("btn-snapshot");
snapshot?.addEventListener("click", () => {
  const params = { geometryModel: getModel() };
  sliders.forEach((slider) => {
    params[slider.dataset.param] = Number.parseFloat(slider.value);
  });
  console.log("Sundog params snapshot:", JSON.stringify(params, null, 2));
  try {
    navigator.clipboard.writeText(JSON.stringify(params, null, 2));
  } catch {
    // ignore
  }
});

// Initial layout
apply();

