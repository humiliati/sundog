/**
 * sundog-page-init.mjs — page-level glue for sundog.html.
 *
 * Loaded after parhelion-workbench.mjs (which wires the hero workbench).
 * This module owns the §6 photo-upload mount, the advanced-controls toggle,
 * the §4 mini live-demo cross-binding, and the smooth-scroll anchor links.
 *
 * Kept as a separate file (rather than an inline <script type="module"> in
 * sundog.html) so Vite's static analysis can resolve the import path through
 * the standard relative-module mechanism.
 */

import { mountPhotoUpload } from "./photo-upload.mjs";

// ---- §6 photo-upload widget --------------------------------------------

const mount = document.getElementById("photo-upload-mount");
if (mount) mountPhotoUpload(mount);

// ---- hero advanced-controls toggle -------------------------------------

const controls = document.getElementById("hero-controls");
const advBtn = document.getElementById("btn-advanced-toggle");
if (advBtn && controls) {
  advBtn.addEventListener("click", () => {
    controls.classList.toggle("is-advanced-hidden");
    const showing = !controls.classList.contains("is-advanced-hidden");
    advBtn.textContent = showing
      ? "Hide advanced controls"
      : "Show advanced controls (25 sliders)";
  });
}

// ---- §4 mini live-demo: drive the same CSS variable as the hero --------

const demoAlt    = document.getElementById("demo-altitude");
const demoAltOut = document.getElementById("demo-altitude-readout");
const demoOff    = document.getElementById("demo-offset-readout");
const demoCza    = document.getElementById("demo-cza-readout");
const realAlt    = document.getElementById("sun-altitude");

function updateDemoReadout() {
  if (!demoAlt || !demoAltOut) return;
  const h = Number.parseFloat(demoAlt.value);
  const sec = 1 / Math.cos((h * Math.PI) / 180);
  demoAltOut.textContent = `h = ${h.toFixed(0)}°`;
  if (demoOff) demoOff.textContent = `R₂₂ × ${sec.toFixed(3)}`;
  if (demoCza) demoCza.textContent = h <= 32 ? "CZA visible: yes" : "CZA visible: no (h > 32°)";
}

if (demoAlt && realAlt) {
  demoAlt.addEventListener("input", () => {
    realAlt.value = demoAlt.value;
    realAlt.dispatchEvent(new Event("input"));
    updateDemoReadout();
  });
  realAlt.addEventListener("input", () => {
    if (Number.parseFloat(realAlt.value) !== Number.parseFloat(demoAlt.value)) {
      demoAlt.value = realAlt.value;
      updateDemoReadout();
    }
  });
  updateDemoReadout();
}

// ---- smooth-scroll for in-page anchor links ----------------------------

document.querySelectorAll("a[href^='#']").forEach((a) => {
  a.addEventListener("click", (e) => {
    const id = a.getAttribute("href").slice(1);
    const target = document.getElementById(id);
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
});
