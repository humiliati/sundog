// Sundog — Mirages interactive ray-tracing explainer.
// A paraxial curved-Earth atmospheric-refraction ray tracer + a side-view ray diagram. Standard
// textbook atmospheric optics (refraction, the effective-Earth-radius k-factor, ducting). Physics
// mirrors scripts/mirage_curved.py: in the height-above-surface frame, dh/dx = ψ, dψ/dx = 1/R_E + n'(h)
// (rays curve toward higher n; the +1/R_E is the surface curving away). Plain ESM, no build step.

const R_E = 6.371e6;              // Earth radius [m]
const INV_RE = 1 / R_E;          // ≈1.57e-7 /m — the ducting-threshold gradient magnitude
const G_STD = -3.92e-8;          // standard-refraction dn/dh [1/m] -> k≈4/3

// --- refractivity gradient profiles n'(h) [1/m] for the presets ----------------------------------- //
export function makeProfile(kind, strength) {
  const s = Math.max(0, Math.min(1, strength));
  if (kind === "standard") {
    return { grad: () => G_STD, label: "Standard air", regimeHint: "ordinary horizon" };
  }
  if (kind === "hot-road") {
    const amp = 6e-7 + 4e-6 * s;                         // strong near-surface positive gradient (hot ground)
    return { grad: (h) => G_STD + amp * Math.exp(-h / 1.2), label: "Hot road / desert",
             regimeHint: "inferior mirage (inverted image low)" };
  }
  if (kind === "cold-sea") {
    const extra = (1.2 + 2.2 * s) * INV_RE;             // ELEVATED inversion -> rays loop back (2nd image)
    return { grad: (h) => G_STD - extra / Math.cosh((h - 20) / 11) ** 2,
             label: "Cold sea (inversion)", regimeHint: "superior mirage / looming" };
  }
  const extra = (2.4 + 3.6 * s) * INV_RE;               // strong elevated inversion -> ducting + stacking
  return { grad: (h) => G_STD - extra / Math.cosh((h - 26) / 12) ** 2,
           label: "Strong inversion (duct)", regimeHint: "ducting — Novaya-Zemlya" };
}

// each preset has a natural viewing geometry (object distance / observer height / object height):
// inferior mirages are short-range with the eye looking down; superior/ducting develop toward the horizon.
const PRESETS = {
  "standard":   { dist: 20, eye: 6, obj: 0.30 },
  "hot-road":   { dist: 6,  eye: 8, obj: 0.05 },
  "cold-sea":   { dist: 32, eye: 6, obj: 0.12 },
  "strong-inv": { dist: 42, eye: 6, obj: 0.10 },
};

// --- RK4 ray trace (height-above-surface frame): returns the path (m) + fate ---------------------- //
export function traceRay(theta, profile, range, hObs, nSteps = 900) {
  const dx = range / nSteps;
  let h = hObs, psi = theta;
  const pts = [{ x: 0, h }];
  let hit = -1;
  const dpsi = (hh) => INV_RE + profile.grad(hh);
  for (let i = 0; i < nSteps; i++) {
    const k1h = psi, k1p = dpsi(h);
    const k2h = psi + 0.5 * dx * k1p, k2p = dpsi(h + 0.5 * dx * k1h);
    const k3h = psi + 0.5 * dx * k2p, k3p = dpsi(h + 0.5 * dx * k2h);
    const k4h = psi + dx * k3p, k4p = dpsi(h + dx * k3h);
    h += (dx / 6) * (k1h + 2 * k2h + 2 * k3h + k4h);
    psi += (dx / 6) * (k1p + 2 * k2p + 2 * k3p + k4p);
    const x = (i + 1) * dx;
    if (h <= 0 && hit < 0) { hit = x; pts.push({ x, h: 0 }); break; }
    pts.push({ x, h });
  }
  return { pts, hit, hEnd: h };
}

export function kFactor(grad0) { return 1 / (1 + R_E * grad0); }

// the object's images: eye-launch angles θ whose ray reaches (objDist, hObj) without hitting the ground.
export function findImages(profile, objDist, hObs, hObj) {
  const out = [];
  const N = 1100;
  let prev = null;
  for (let i = 0; i <= N; i++) {
    const theta = -3.0e-3 + 6.0e-3 * (i / N);
    const r = traceRay(theta, profile, objDist, hObs, 800);
    const reached = (r.hit > 0 && r.hit < objDist * 0.999) ? null : r.hEnd;
    if (reached !== null && prev && prev.reached !== null) {
      const f0 = prev.reached - hObj, f1 = reached - hObj;
      if (f0 === 0 || (f0 < 0) !== (f1 < 0)) {
        const frac = f0 / (f0 - f1);
        out.push({ theta: prev.theta + frac * (theta - prev.theta) });
      }
    }
    prev = { theta, reached };
  }
  return out;
}

// =================================================================================================== //
//  Rendering — Earth-centred side view: the surface curves away (surfaceY), rays drawn at surfaceY+h.
// =================================================================================================== //
const STATE = { kind: "cold-sea", strength: 0.6, hObs: 6, objDistKm: 32, hObjFrac: 0.12 };
let canvas, ctx, statusEl;
const COL = {};

function loadColors() {
  const v = (n, d) => (getComputedStyle(document.documentElement).getPropertyValue(n).trim() || d);
  COL.ink = v("--primary-dark", "#142b3a");
  COL.sky = v("--surface", "#eef4f8");
  COL.ray = "#2e6f95"; COL.img = "#c0532a"; COL.warm = "#e08a2e";
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
}

function draw() {
  if (!canvas || !canvas.width) return;
  const rect = canvas.getBoundingClientRect();
  const W = rect.width, H = rect.height, dpr = canvas.width / W;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const profile = makeProfile(STATE.kind, STATE.strength);
  const range = STATE.objDistKm * 1e3;
  const padL = 58, padR = 16, padT = 16, padB = 26, plotW = W - padL - padR, plotH = H - padT - padB;

  const surfaceY = (xm) => -(xm * xm) / (2 * R_E);          // surface absolute height (drops with distance)
  const surfDrop = -surfaceY(range);
  const hMax = Math.max(60, STATE.hObs * 2.2, 90);
  const yMin = -surfDrop * 1.15 - 5, yMax = hMax;
  const X = (xm) => padL + plotW * (xm / range);
  const Y = (absH) => padT + plotH * (1 - (absH - yMin) / (yMax - yMin));

  ctx.fillStyle = COL.sky; ctx.globalAlpha = 0.45; ctx.fillRect(padL, padT, plotW, plotH); ctx.globalAlpha = 1;

  // refractivity strip (left margin), warm air = low n at bottom in an inversion
  for (let i = 0; i < 44; i++) {
    const hm = Math.max(0, yMax - (yMax - yMin) * (i / 44));
    const g = profile.grad(hm);
    const t = Math.max(0, Math.min(1, (-g - 2e-8) / (3 * INV_RE)));
    ctx.fillStyle = `rgba(${Math.round(70 + 150 * t)},${Math.round(120 - 50 * t)},${Math.round(160 - 110 * t)},0.55)`;
    ctx.fillRect(8, padT + plotH * (i / 44), 16, plotH / 44 + 1);
  }
  ctx.save(); ctx.translate(16, padT + plotH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = COL.ink; ctx.font = "10px var(--sd-font-data, monospace)"; ctx.textAlign = "center";
  ctx.fillText("air refractivity n(h)", 0, 0); ctx.restore();

  // the curving Earth surface
  const surfPath = () => { ctx.beginPath();
    for (let xm = 0; xm <= range; xm += range / 240) { const p = [X(xm), Y(surfaceY(xm))]; xm === 0 ? ctx.moveTo(...p) : ctx.lineTo(...p); } };
  surfPath(); ctx.lineTo(X(range), H); ctx.lineTo(padL, H); ctx.closePath();
  ctx.fillStyle = "#caa46a"; ctx.globalAlpha = 0.4; ctx.fill(); ctx.globalAlpha = 1;
  surfPath(); ctx.strokeStyle = "#9a7b45"; ctx.lineWidth = 1.5; ctx.stroke();

  // ray fan from the eye
  for (let i = 0; i <= 26; i++) {
    const theta = -2.4e-3 + 4.8e-3 * (i / 26);
    const r = traceRay(theta, profile, range, STATE.hObs, 650);
    ctx.beginPath();
    for (const p of r.pts) { const xy = [X(p.x), Y(surfaceY(p.x) + p.h)]; p.x === 0 ? ctx.moveTo(...xy) : ctx.lineTo(...xy); }
    ctx.strokeStyle = (r.hit > 0 && r.hit < range * 0.99) ? "rgba(154,123,69,0.20)" : "rgba(46,111,149,0.18)";
    ctx.lineWidth = 1; ctx.stroke();
  }

  // object (pole) + the images the eye actually sees (highlighted rays). Object sits low (near the
  // surface, where mirages live): 2..42 m above the surface.
  const hObj = 2 + 40 * STATE.hObjFrac;
  const objAbsBase = surfaceY(range);
  ctx.strokeStyle = COL.ink; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(X(range) - 1, Y(objAbsBase)); ctx.lineTo(X(range) - 1, Y(objAbsBase + hObj)); ctx.stroke();
  ctx.fillStyle = COL.ink; ctx.beginPath(); ctx.arc(X(range) - 1, Y(objAbsBase + hObj), 3.2, 0, 7); ctx.fill();

  const images = findImages(profile, range, STATE.hObs, hObj);
  for (const im of images) {
    const r = traceRay(im.theta, profile, range, STATE.hObs, 1100);
    ctx.beginPath();
    for (const p of r.pts) { const xy = [X(p.x), Y(surfaceY(p.x) + p.h)]; p.x === 0 ? ctx.moveTo(...xy) : ctx.lineTo(...xy); }
    ctx.strokeStyle = COL.img; ctx.lineWidth = 2; ctx.globalAlpha = 0.9; ctx.stroke(); ctx.globalAlpha = 1;
  }
  // eye marker
  ctx.fillStyle = COL.warm; ctx.beginPath(); ctx.arc(X(0), Y(STATE.hObs), 4.5, 0, 7); ctx.fill();

  // labels
  ctx.fillStyle = COL.ink; ctx.font = "11px var(--sd-font-data, monospace)";
  ctx.textAlign = "left"; ctx.fillText("eye", X(0) + 7, Y(STATE.hObs) - 6);
  ctx.textAlign = "right"; ctx.fillText("object", X(range) - 6, Y(objAbsBase + hObj) - 6);
  ctx.textAlign = "center"; ctx.fillText(`${STATE.objDistKm} km →`, padL + plotW / 2, H - 7);

  // status
  const k = kFactor(profile.grad(STATE.hObs));
  const n = images.length;
  if (statusEl) statusEl.textContent =
    `${profile.label}  ·  effective Earth radius k≈${k.toFixed(2)}  ·  ${n} image${n === 1 ? "" : "s"} seen  ·  ${profile.regimeHint}`;
}

// --- controls ------------------------------------------------------------------------------------- //
function setSlider(id, value) {
  const el = document.getElementById(id), out = document.getElementById(id + "-val");
  if (el) el.value = value;
  if (out) out.textContent = value;
}

function bind() {
  document.querySelectorAll("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => {
      STATE.kind = b.getAttribute("data-preset");
      document.querySelectorAll("[data-preset]").forEach((x) =>
        x.setAttribute("aria-pressed", x === b ? "true" : "false"));
      // jump to the preset's natural viewing geometry (inferior=short+looking-down; superior/ducting=long)
      const cfg = PRESETS[STATE.kind];
      if (cfg) {
        STATE.objDistKm = cfg.dist; STATE.hObs = cfg.eye; STATE.hObjFrac = cfg.obj;
        setSlider("mir-dist", cfg.dist); setSlider("mir-hobs", cfg.eye); setSlider("mir-hobj", cfg.obj);
      }
      draw();
    }));
  const wire = (id, key) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", () => {
      STATE[key] = Number(el.value);
      const out = document.getElementById(id + "-val");
      if (out) out.textContent = el.value;
      draw();
    });
  };
  wire("mir-strength", "strength");
  wire("mir-hobs", "hObs");
  wire("mir-dist", "objDistKm");
  wire("mir-hobj", "hObjFrac");
}

export function init() {
  canvas = document.getElementById("mirage-canvas");
  statusEl = document.getElementById("mirage-status");
  if (!canvas) return;
  ctx = canvas.getContext("2d");
  loadColors();
  bind();
  const def = document.querySelector(`[data-preset="${STATE.kind}"]`);
  if (def) def.setAttribute("aria-pressed", "true");
  window.addEventListener("resize", () => { resize(); draw(); });
  resize();
  draw();
}

if (typeof document !== "undefined") init();
