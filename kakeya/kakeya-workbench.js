// Kakeya tiny finite-field workbench — browser UI wiring.
// Loads the verified pure core (kakeya-core.js) and renders the internal
// teaching surface defined by
// docs/kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md.
//
// The body (selected points) is shown on the LEFT; the direction shadow on the
// RIGHT. The primary shadow is the q+1-bit coverage bitset only — never the
// point list. Witnesses are one-at-a-time and off by default.

import * as K from "./kakeya-core.js";

const state = {
  q: K.DEFAULT_Q,
  body: new Set(),
  witnessOn: false,
  witnessDir: 0,
  randomSeed: 1,
};

const $ = (id) => document.getElementById(id);
const fmt = (x) => (Math.round(x * 1000) / 1000).toString();

// ---- grid -----------------------------------------------------------------
function buildGrid() {
  const q = state.q;
  const grid = $("grid");
  grid.style.setProperty("--q", q);
  grid.innerHTML = "";
  // render y = q-1 (top) down to 0 (bottom), x = 0..q-1 left to right
  for (let y = q - 1; y >= 0; y--) {
    for (let x = 0; x < q; x++) {
      const idx = K.pointIndex(x, y, q);
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      cell.dataset.index = String(idx);
      cell.title = `(${x}, ${y})`;
      cell.setAttribute("aria-label", `point ${x}, ${y}`);
      cell.addEventListener("click", () => {
        if (state.body.has(idx)) state.body.delete(idx);
        else state.body.add(idx);
        render();
      });
      grid.appendChild(cell);
    }
  }
}

function paintGrid() {
  const q = state.q;
  let witnessSet = new Set();
  if (state.witnessOn) {
    const dirs = K.directions(q);
    const dir = dirs[state.witnessDir];
    const w = dir ? K.witnessLine(dir, q, state.body) : null;
    if (w) witnessSet = new Set(w.points);
  }
  for (const cell of $("grid").children) {
    const idx = Number(cell.dataset.index);
    cell.classList.toggle("on", state.body.has(idx));
    cell.classList.toggle("witness", witnessSet.has(idx));
  }
}

// ---- direction list + summary --------------------------------------------
function renderShadow() {
  const q = state.q;
  const s = K.shadowSummary(q, state.body);
  const dirs = K.directions(q);

  // direction list
  const dl = $("dir-list");
  dl.innerHTML = "";
  dirs.forEach((d, i) => {
    const row = document.createElement("div");
    row.className = "dir-row " + (s.bits[i] ? "covered" : "missing");
    row.innerHTML =
      `<span class="dir-label">${d.label === "inf" ? "∞" : d.label}</span>` +
      `<span class="dir-state">${s.bits[i] ? "covered" : "—"}</span>`;
    dl.appendChild(row);
  });

  // summary (no Euclidean/area/dimension/measure language in metric labels)
  $("summary").innerHTML = `
    <div class="stat"><span>field</span><b>F_${q}², q = ${q}</b></div>
    <div class="stat"><span>selected points |K|</span><b>${s.bodySize} / ${K.pointCount(q)}</b></div>
    <div class="stat"><span>body fraction</span><b>${fmt(s.bodyFraction)}</b></div>
    <div class="stat"><span>directions covered</span><b>${s.directionsCovered} / ${s.directionCount}</b></div>
    <div class="stat"><span>coverage fraction</span><b>${fmt(s.coverageFraction)}</b></div>
    <div class="stat"><span>missing</span><b>${s.missing.length ? s.missing.map((m) => (m === "inf" ? "∞" : m)).join(", ") : "none"}</b></div>
    <div class="verdict v-${s.complete ? "complete" : s.directionsCovered ? "near" : "empty"}">${s.verdict}</div>
  `;

  // Dvir lower-bound card (known finite-field theorem context, with the
  // not-tight-in-the-plane polish so a complete set larger than the floor does
  // not read as a bug).
  $("dvir-card").innerHTML = s.complete
    ? `<p>Every <strong>finite-field</strong> Kakeya set in F_${q}² satisfies Dvir's
         bound <code>|K| ≥ C(q+1, 2) = ${s.dvirFloor}</code>. Here
         <code>|K| = ${s.bodySize}</code> ${s.dvirFloorConsistent ? "✓" : "✗"}.</p>
       <p class="note">Dvir's bound is a valid lower bound, not the exact planar
         minimum — it is not tight in the plane, so minimal Kakeya sets come out
         somewhat larger than ${s.dvirFloor}. This is a known finite-field fact,
         not a measurement.</p>`
    : `<p>Cover every direction to see Dvir's <strong>finite-field</strong>
         lower-bound card (<code>|K| ≥ C(q+1, 2) = ${s.dvirFloor}</code>).</p>`;

  // witness direction options (covered directions only)
  const wsel = $("witness-dir");
  const prev = state.witnessDir;
  wsel.innerHTML = "";
  dirs.forEach((d, i) => {
    if (!s.bits[i]) return;
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = d.label === "inf" ? "∞ (vertical)" : `slope ${d.label}`;
    wsel.appendChild(opt);
  });
  if ([...wsel.options].some((o) => Number(o.value) === prev)) wsel.value = String(prev);
  else if (wsel.options.length) state.witnessDir = Number(wsel.options[0].value);
  $("witness-wrap").classList.toggle("hidden", !state.witnessOn);

  if (state.witnessOn) {
    const dir = dirs[state.witnessDir];
    const w = dir ? K.witnessLine(dir, q, state.body) : null;
    $("witness-out").textContent = w
      ? `one full line in direction ${dir.label === "inf" ? "∞" : dir.label}: intercept b = ${w.intercept}`
      : "selected direction is not covered";
  } else {
    $("witness-out").textContent = "";
  }
}

function render() {
  paintGrid();
  renderShadow();
}

// ---- presets / actions ----------------------------------------------------
function applyPreset(name) {
  const q = state.q;
  const dirs = K.directions(q);
  switch (name) {
    case "empty":
      state.body = K.bEmpty();
      break;
    case "single-line":
      state.body = K.bSingleLine(q, dirs[0], 0);
      break;
    case "whole-plane":
      state.body = K.bWholePlane(q);
      break;
    case "whole-minus-one":
      state.body = K.bWholeMinusOne(q, 0);
      break;
    case "random-line-cover":
      state.body = K.bRandomLineCover(q, state.randomSeed);
      break;
    case "greedy-line-cover":
      state.body = K.bGreedyLineCover(q);
      break;
    default:
      return;
  }
  render();
}

function randomSubset() {
  state.randomSeed += 1;
  const q = state.q;
  state.body = K.bRandomSubset(q, Math.round(K.pointCount(q) / 2), state.randomSeed);
  $("seed-readout").textContent = `seed ${state.randomSeed}`;
  render();
}

function setQ(q) {
  state.q = q;
  state.body = new Set();
  state.witnessDir = 0;
  buildGrid();
  render();
}

// ---- export (primary shadow only) ----------------------------------------
function exportShadow() {
  const obj = K.exportShadow(state.q, state.body);
  $("export-out").textContent = JSON.stringify(obj, null, 2);
}

// ---- wire up --------------------------------------------------------------
function init() {
  // field-size selector
  const qsel = $("q-select");
  K.SUPPORTED_Q.forEach((q) => {
    const opt = document.createElement("option");
    opt.value = String(q);
    opt.textContent = `q = ${q}`;
    if (q === state.q) opt.selected = true;
    qsel.appendChild(opt);
  });
  qsel.addEventListener("change", () => setQ(Number(qsel.value)));

  $("btn-clear").addEventListener("click", () => applyPreset("empty"));
  $("btn-fill").addEventListener("click", () => applyPreset("whole-plane"));
  $("btn-random").addEventListener("click", randomSubset);

  $("preset-select").addEventListener("change", (e) => {
    if (e.target.value) {
      applyPreset(e.target.value);
      e.target.value = "";
    }
  });

  const wt = $("witness-toggle");
  wt.checked = false;
  wt.addEventListener("change", () => {
    state.witnessOn = wt.checked;
    render();
  });
  $("witness-dir").addEventListener("change", (e) => {
    state.witnessDir = Number(e.target.value);
    render();
  });

  $("btn-export").addEventListener("click", exportShadow);

  $("seed-readout").textContent = `seed ${state.randomSeed}`;
  buildGrid();
  render();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
