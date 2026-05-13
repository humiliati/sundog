import assert from "node:assert/strict";

class FakeClassList {
  constructor(element) {
    this.element = element;
    this.names = new Set();
  }

  add(...names) {
    for (const name of names) this.names.add(name);
  }

  remove(...names) {
    for (const name of names) this.names.delete(name);
  }

  contains(name) {
    return this.names.has(name);
  }

  fromAttribute(value) {
    this.names = new Set(String(value).split(/\s+/).filter(Boolean));
  }
}

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.parentElement = null;
    this.attributes = new Map();
    this.listeners = new Map();
    this.dataset = {};
    this.style = {};
    this.classList = new FakeClassList(this);
  }

  setAttribute(name, value) {
    const stringValue = String(value);
    this.attributes.set(name, stringValue);
    if (name === "class") this.classList.fromAttribute(stringValue);
    if (name.startsWith("data-")) {
      const key = name
        .slice(5)
        .replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      this.dataset[key] = stringValue;
    }
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  replaceChildren(...children) {
    this.children = [];
    for (const child of children) this.appendChild(child);
  }

  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }

  dispatchEvent(event) {
    if (!event.preventDefault) event.preventDefault = () => {};
    for (const listener of this.listeners.get(event.type) ?? []) listener(event);
    return true;
  }

  setPointerCapture() {}

  releasePointerCapture() {}

  querySelector(selector) {
    for (const child of this.children) {
      if (matches(child, selector)) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
}

class FakeSvg extends FakeElement {
  constructor() {
    super("svg");
    this.dataset.geometryModel = "halo_atlas";
  }

  createSVGPoint() {
    return {
      x: 0,
      y: 0,
      matrixTransform() {
        return { x: this.x, y: this.y };
      },
    };
  }

  getScreenCTM() {
    return { inverse: () => ({}) };
  }
}

function matches(element, selector) {
  if (selector.startsWith(".")) return element.classList.contains(selector.slice(1));
  if (selector.startsWith("#")) return element.getAttribute("id") === selector.slice(1);
  const dataMatch = selector.match(/^\[data-phase6-binding="([^"]+)"\]$/);
  if (dataMatch) return element.dataset.phase6Binding === dataMatch[1];
  return false;
}

function slider({ id, value, min, max, step }) {
  const element = new FakeElement("input");
  element.id = id;
  element.value = value;
  element.defaultValue = value;
  element.min = min;
  element.max = max;
  element.step = step;
  element.setAttribute("id", id);
  return element;
}

function pointer(type, x, y) {
  return {
    type,
    pointerId: 1,
    clientX: x,
    clientY: y,
    preventDefault() {},
  };
}

function nearly(actual, expected, label) {
  assert.ok(Math.abs(actual - expected) < 0.05, `${label}: expected ${expected}, got ${actual}`);
}

const css = new Map([
  ["--parhelic-curvature", "0.05"],
  ["--parhelic-y-offset-r22", "-0.05"],
  ["--cza-curvature", "0.85"],
  ["--cza-intensity", "0.95"],
  ["--sun-altitude", "25"],
]);

const fakeDocument = new FakeElement("document");
fakeDocument.documentElement = new FakeElement("html");
fakeDocument.createElementNS = (_namespace, tagName) => new FakeElement(tagName);

globalThis.document = fakeDocument;
globalThis.getComputedStyle = () => ({
  getPropertyValue: (name) => css.get(name) ?? "",
});

const { enablePhase6Drag } = await import("../public/js/phase6-drag.mjs");

const svg = new FakeSvg();
const sunAltitudeSlider = slider({ id: "sun-altitude", value: "25", min: "0", max: "60", step: "1" });
const parhelicCurvatureSlider = slider({ id: "parhelic-curvature", value: "0.05", min: "0", max: "1", step: "0.01" });
const parhelicYOffsetSlider = slider({ id: "parhelic-y-offset-r22", value: "-0.05", min: "-0.12", max: "0.08", step: "0.01" });
const czaCurvatureSlider = slider({ id: "cza-curvature", value: "0.85", min: "0.4", max: "1.4", step: "0.01" });
const czaIntensitySlider = slider({ id: "cza-intensity", value: "0.95", min: "0", max: "1", step: "0.01" });
const deriveToggle = new FakeElement("input");
deriveToggle.checked = false;

sunAltitudeSlider.addEventListener("input", () => {
  css.set("--sun-altitude", sunAltitudeSlider.value);
});
parhelicCurvatureSlider.addEventListener("input", () => {
  css.set("--parhelic-curvature", parhelicCurvatureSlider.value);
});
parhelicYOffsetSlider.addEventListener("input", () => {
  css.set("--parhelic-y-offset-r22", parhelicYOffsetSlider.value);
});
czaCurvatureSlider.addEventListener("input", () => {
  css.set("--cza-curvature", czaCurvatureSlider.value);
});
czaIntensitySlider.addEventListener("input", () => {
  css.set("--cza-intensity", czaIntensitySlider.value);
});

enablePhase6Drag(svg, {
  sunAltitudeSlider,
  parhelicCurvatureSlider,
  parhelicYOffsetSlider,
  czaCurvatureSlider,
  czaIntensitySlider,
  deriveToggle,
});

const apex = svg.querySelector(".phase6-handle-parhelic-apex");
assert.ok(apex, "missing parhelic apex handle");
assert.equal(apex.getAttribute("cy"), "499");

deriveToggle.checked = true;
apex.dispatchEvent(pointer("pointerdown", 500, 499));
apex.dispatchEvent(pointer("pointermove", 500, 589));
apex.dispatchEvent(pointer("pointerup", 500, 589));

assert.equal(deriveToggle.checked, false, "apex drag should switch derived curvature to manual");
assert.equal(parhelicCurvatureSlider.value, "0.50");
assert.equal(apex.getAttribute("aria-valuenow"), "0.50");
assert.equal(apex.getAttribute("cy"), "589");

const czaApex = svg.querySelector(".phase6-handle-cza-apex");
assert.ok(czaApex, "missing CZA apex handle");
assert.equal(czaApex.getAttribute("cy"), "60");
assert.equal(czaApex.style.display, "");

czaApex.dispatchEvent(pointer("pointerdown", 500, 60));
czaApex.dispatchEvent(pointer("pointermove", 500, 130));
czaApex.dispatchEvent(pointer("pointerup", 500, 130));

assert.equal(czaCurvatureSlider.value, "0.50");
assert.equal(czaApex.getAttribute("aria-valuenow"), "0.50");
assert.equal(czaApex.getAttribute("cy"), "130");

const left = svg.querySelector(".phase6-handle-left");
const right = svg.querySelector(".phase6-handle-right");
assert.ok(left, "missing left parhelion handle");
assert.ok(right, "missing right parhelion handle");

right.dispatchEvent(pointer("pointerdown", 742.74, 489));
right.dispatchEvent(pointer("pointermove", 811.13, 489));
right.dispatchEvent(pointer("pointerup", 811.13, 489));

assert.equal(sunAltitudeSlider.value, "45");
nearly(Number.parseFloat(right.getAttribute("cx")), 811.13, "right parhelion handle x");
nearly(Number.parseFloat(left.getAttribute("cx")), 188.87, "left parhelion handle x");
assert.equal(right.getAttribute("cy"), "489");
assert.equal(left.getAttribute("cy"), "489");
assert.equal(czaApex.style.display, "none", "CZA handle should hide when sun altitude exceeds 32 degrees");
assert.equal(czaApex.getAttribute("aria-hidden"), "true");

console.log("Phase 6 drag constraint check passed");
