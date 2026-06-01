// Kakeya reward graphic — body-resistance continuum + finite-field gallery.
// INTERNAL Phase-3 / Front-A asset (not launched). Renders, from the verified
// kakeya-core, a cross-substrate body-resistance axis and a gallery of
// finite-field Kakeya instances (body in F_q^2 + lossy direction-shadow fan),
// with hover tooltips, an animated "directions light up" reveal, and a
// self-contained PNG share card that carries the claim-boundary fences.
// Honest framing only: finite-field register, body resistance — not Euclidean
// evidence, not a regime-2 separation.

import * as K from "./kakeya-core.js";

const C = {
  navy: "#1a3a52",
  navyMid: "#2d5575",
  gold: "#b8831e",
  green: "#2f7d4f",
  faint: "#c9d4dc",
  ink: "#213040",
  line: "#d7e0e7",
  bg: "#f8fbfc",
};

const $ = (id) => document.getElementById(id);
const fmt = (x) => (Math.round(x * 1000) / 1000).toString();
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

let GALLERY = [];
let selectedEntry = null;

// --- body-resistance continuum hero --------------------------------------
function continuumSVG() {
  const W = 920, H = 200, x0 = 70, x1 = W - 70, y = 96;
  const markers = [
    { t: 0.0, label: "Faraday", sub: "exact 0 · Bianchi", tone: C.green, pole: true },
    { t: 0.16, label: "NSE C1", sub: "marginal", tone: C.faint },
    { t: 0.26, label: "Mesa", sub: "marginal", tone: C.faint },
    { t: 0.36, label: "shell", sub: "marginal", tone: C.faint },
    { t: 0.72, label: "Aharonov–Bohm", sub: "exact · topological", tone: C.gold },
    { t: 1.0, label: "Kakeya", sub: "exact maximal", tone: C.navy, pole: true },
  ];
  const xAt = (t) => x0 + (x1 - x0) * t;
  let dots = "";
  for (const m of markers) {
    const mx = xAt(m.t), r = m.pole ? 9 : 5;
    dots += `<circle cx="${mx}" cy="${y}" r="${r}" fill="${m.tone}" stroke="#fff" stroke-width="2"/>`;
    const above = m.pole;
    dots += `<text x="${mx}" y="${above ? y - 22 : y + 28}" text-anchor="middle" font-size="${m.pole ? 15 : 12}" font-weight="${m.pole ? 800 : 600}" fill="${C.ink}">${m.label}</text>`;
    dots += `<text x="${mx}" y="${above ? y - 6 : y + 42}" text-anchor="middle" font-size="10" fill="#6b7b88">${m.sub}</text>`;
  }
  return `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Body-resistance axis from Faraday-zero to Kakeya-maximal" style="width:100%;height:auto">
      <defs><linearGradient id="brgrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color="#e7f4ec"/><stop offset="0.4" stop-color="#eef2f5"/><stop offset="1" stop-color="#1a3a52"/>
      </linearGradient></defs>
      <text x="${x0}" y="34" font-size="12" font-weight="800" letter-spacing="0.08em" fill="${C.gold}">BODY-RESISTANCE AXIS</text>
      <text x="${x0}" y="54" font-size="13" fill="#40505c">how hard a body resists reconstruction from its lossy shadow</text>
      <rect x="${x0}" y="${y - 7}" width="${x1 - x0}" height="14" rx="7" fill="url(#brgrad)" stroke="${C.line}"/>
      <text x="${x0}" y="${H - 18}" font-size="11" fill="#6b7b88">resistance 0 — shadow reconstructs the body</text>
      <text x="${x1}" y="${H - 18}" text-anchor="end" font-size="11" fill="#6b7b88">resistance maximal — shadow reconstructs nothing</text>
      ${dots}
    </svg>`;
}

// --- one Kakeya instance panel -------------------------------------------
// opts: size, animate (covered rays light up staggered), hit (transparent hit
// lines + point data for tooltips).
function panelSVG(q, body, { size = 150, animate = false, hit = false } = {}) {
  const bits = K.shadowBitset(q, body);
  const dirs = K.directions(q);
  const pad = 12, gridSpan = size - pad * 2, step = gridSpan / (q - 1 || 1);
  let dots = "";
  for (let y = 0; y < q; y++) {
    for (let x = 0; x < q; x++) {
      const px = (pad + x * step).toFixed(1);
      const py = (pad + (q - 1 - y) * step).toFixed(1);
      const on = body.has(K.pointIndex(x, y, q));
      const data = hit ? ` data-xy="${x},${y}" data-on="${on ? 1 : 0}"` : "";
      dots += `<circle${data} cx="${px}" cy="${py}" r="${on ? 3.4 : 1.6}" fill="${on ? C.navy : C.faint}"/>`;
    }
  }
  const fanY = size + 30, fanCx = size / 2, R = size / 2 - 6, n = q + 1;
  let vis = "", hits = "", coveredIdx = 0;
  for (let i = 0; i < n; i++) {
    const theta = Math.PI - (Math.PI * (i + 0.5)) / n;
    const ex = (fanCx + R * Math.cos(theta)).toFixed(1);
    const ey = (fanY - R * Math.sin(theta)).toFixed(1);
    const covered = bits[i] === 1;
    const label = dirs[i].label === "inf" ? "∞" : dirs[i].label;
    const cls = "ray" + (animate && covered ? " anim" : "");
    const delay = animate && covered ? ` style="animation-delay:${coveredIdx * 70}ms"` : "";
    if (covered) coveredIdx++;
    vis += `<line class="${cls}"${delay} x1="${fanCx}" y1="${fanY}" x2="${ex}" y2="${ey}" stroke="${covered ? C.green : C.faint}" stroke-width="${covered ? 2.4 : 1.2}" stroke-linecap="round" pointer-events="none"/>`;
    if (hit) {
      let di = "";
      if (covered) { const w = K.witnessLine(dirs[i], q, body); if (w) di = ` data-int="${w.intercept}"`; }
      hits += `<line data-dir="${esc(label)}" data-cov="${covered ? 1 : 0}"${di} x1="${fanCx}" y1="${fanY}" x2="${ex}" y2="${ey}" stroke="transparent" stroke-width="13"/>`;
    }
  }
  return `
    <svg viewBox="0 0 ${size} ${size + 40}" role="img" aria-label="Kakeya body and direction shadow">
      <rect x="1" y="1" width="${size - 2}" height="${size - 2}" rx="8" fill="#fff" stroke="${C.line}"/>
      ${dots}${vis}${hits}
    </svg>`;
}

// --- gallery entries ------------------------------------------------------
function entries() {
  return [
    { id: "single-7", q: 7, title: "Single line", note: "q points, one direction lit. The smallest nonzero shadow.", body: K.bSingleLine(7, K.directions(7)[2], 1) },
    { id: "random-7", q: 7, title: "Random half", note: "Half the plane at random — many points, few full lines. Point count is not direction-completeness.", body: K.bRandomSubset(7, Math.round(K.pointCount(7) / 2), 3) },
    { id: "greedy-7", q: 7, title: "Greedy cover · q=7", note: "Direction-complete by overlapping lines. Every direction present — yet the body cannot drop below ~half (Dvir).", body: K.bGreedyLineCover(7) },
    { id: "wmo-7", q: 7, title: "Whole plane minus one", note: "One point removed, still every direction. The complete shadow cannot reconstruct the body.", body: K.bWholeMinusOne(7, 24) },
    { id: "greedy-5", q: 5, title: "Greedy cover · q=5", note: "The same spectacle in the smallest supported field.", body: K.bGreedyLineCover(5) },
    { id: "greedy-11", q: 11, title: "Greedy cover · q=11", note: "Larger field; the body fraction settles toward the Dvir floor (~1/2).", body: K.bGreedyLineCover(11) },
  ];
}

function factsHTML(entry) {
  const s = K.shadowSummary(entry.q, entry.body);
  return `
    <div class="fact"><span>selected points |K|</span><b>${s.bodySize} / ${K.pointCount(entry.q)}</b></div>
    <div class="fact"><span>body fraction</span><b>${fmt(s.bodyFraction)}</b></div>
    <div class="fact"><span>directions covered</span><b>${s.directionsCovered} / ${s.directionCount}</b></div>
    <div class="fact"><span>coverage fraction</span><b>${fmt(s.coverageFraction)}</b></div>
    <div class="fact"><span>Dvir floor C(q+1,2)</span><b>${s.dvirFloor}${s.complete ? (s.dvirFloorConsistent ? " ✓" : " ✗") : ""}</b></div>`;
}

function renderStage(entry) {
  selectedEntry = entry;
  const s = K.shadowSummary(entry.q, entry.body);
  $("stage-visual").innerHTML = panelSVG(entry.q, entry.body, { size: 230, animate: true, hit: true });
  $("stage-title").textContent = `${entry.title} — F_${entry.q}²`;
  $("stage-note").textContent = entry.note;
  $("stage-facts").innerHTML = factsHTML(entry);
  const v = $("stage-verdict");
  v.textContent = s.verdict;
  v.className = "verdict v-" + (s.complete ? "complete" : s.directionsCovered ? "near" : "empty");
  for (const card of document.querySelectorAll(".gcard")) {
    card.classList.toggle("selected", card.dataset.id === entry.id);
  }
}

function renderGallery() {
  GALLERY = entries();
  const grid = $("gallery-grid");
  grid.innerHTML = "";
  for (const entry of GALLERY) {
    const s = K.shadowSummary(entry.q, entry.body);
    const card = document.createElement("button");
    card.type = "button";
    card.className = "gcard";
    card.dataset.id = entry.id;
    card.dataset.tip = `${entry.title}: ${entry.note}`;
    card.innerHTML =
      `<div class="gcard-svg">${panelSVG(entry.q, entry.body, { size: 132 })}</div>` +
      `<div class="gcard-meta"><strong>${esc(entry.title)}</strong>` +
      `<span>|K| ${s.bodySize}/${K.pointCount(entry.q)} · dirs ${s.directionsCovered}/${s.directionCount}</span></div>`;
    card.addEventListener("click", () => renderStage(entry));
    grid.appendChild(card);
  }
}

// --- hover tooltips -------------------------------------------------------
function tipFor(target) {
  if (!target || !target.dataset) return null;
  const d = target.dataset;
  if (d.dir !== undefined) {
    return d.cov === "1"
      ? `direction ${d.dir}: covered — full line at intercept b=${d.int}`
      : `direction ${d.dir}: missing — no full line in K`;
  }
  if (d.xy !== undefined) return `point (${d.xy}) — ${d.on === "1" ? "selected (in body)" : "empty"}`;
  const card = target.closest && target.closest(".gcard");
  if (card && card.dataset.tip) return card.dataset.tip;
  return null;
}
function setupTooltips() {
  const tip = $("kak-tooltip");
  const move = (e) => {
    const text = tipFor(e.target);
    if (!text) { tip.hidden = true; return; }
    tip.textContent = text;
    tip.hidden = false;
    const pad = 14;
    let x = e.clientX + pad, yy = e.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (yy + r.height > window.innerHeight - 8) yy = e.clientY - r.height - pad;
    tip.style.left = x + "px";
    tip.style.top = yy + "px";
  };
  document.addEventListener("pointermove", move);
  document.addEventListener("pointerleave", () => { tip.hidden = true; }, true);
  window.addEventListener("scroll", () => { tip.hidden = true; }, true);
}

// --- PNG share card (self-contained, fences baked in) --------------------
function buildShareCardSVG(entry) {
  const W = 1200, H = 675;
  const s = K.shadowSummary(entry.q, entry.body);
  const panel = panelSVG(entry.q, entry.body, { size: 300, animate: false, hit: false });
  // mini continuum (kept below the panel; panel occupies y 200..540)
  const cx0 = 70, cx1 = W - 70, cy = 582;
  const facts = [
    ["selected points |K|", `${s.bodySize} / ${K.pointCount(entry.q)}`],
    ["body fraction", fmt(s.bodyFraction)],
    ["directions covered", `${s.directionsCovered} / ${s.directionCount}`],
    ["Dvir floor C(q+1,2)", `${s.dvirFloor}`],
  ];
  let factRows = "";
  facts.forEach(([k, v], i) => {
    const fy = 250 + i * 46;
    factRows += `<text x="560" y="${fy}" font-size="22" fill="#6b7b88">${esc(k)}</text>`;
    factRows += `<text x="${W - 70}" y="${fy}" text-anchor="end" font-size="24" font-weight="800" fill="${C.navy}">${esc(v)}</text>`;
  });
  const verdictTone = s.complete ? C.green : s.directionsCovered ? C.gold : "#6b7b88";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="system-ui, -apple-system, 'Segoe UI', sans-serif">
    <rect width="${W}" height="${H}" fill="${C.bg}"/>
    <rect x="14" y="14" width="${W - 28}" height="${H - 28}" rx="20" fill="#fff" stroke="${C.line}" stroke-width="2"/>
    <text x="70" y="86" font-size="20" font-weight="800" letter-spacing="0.12em" fill="${C.gold}">SUNDOG × KAKEYA · BODY-RESISTANCE</text>
    <text x="70" y="138" font-size="40" font-weight="800" fill="${C.navy}">The exact-maximal body-resistance pole</text>
    <text x="70" y="178" font-size="22" fill="#40505c">Finite field F_${entry.q}²: every direction present, yet the body cannot shrink below a constant fraction.</text>
    <svg x="80" y="200" width="300" height="340">${panel}</svg>
    <text x="560" y="224" font-size="26" font-weight="800" fill="${C.navy}">${esc(entry.title)}</text>
    ${factRows}
    <rect x="560" y="452" width="280" height="44" rx="10" fill="${verdictTone}1f"/>
    <text x="700" y="481" text-anchor="middle" font-size="20" font-weight="800" fill="${verdictTone}">${esc(s.verdict)}</text>
    <rect x="${cx0}" y="${cy}" width="${cx1 - cx0}" height="12" rx="6" fill="#eef2f5" stroke="${C.line}"/>
    <rect x="${cx0}" y="${cy}" width="${cx1 - cx0}" height="12" rx="6" fill="url(#g)"/>
    <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#e7f4ec"/><stop offset="1" stop-color="#1a3a52"/></linearGradient></defs>
    <circle cx="${cx0}" cy="${cy + 6}" r="8" fill="${C.green}" stroke="#fff" stroke-width="2"/>
    <circle cx="${cx1}" cy="${cy + 6}" r="8" fill="${C.navy}" stroke="#fff" stroke-width="2"/>
    <text x="${cx0}" y="${cy - 12}" font-size="16" font-weight="700" fill="${C.ink}">Faraday · exact 0</text>
    <text x="${cx1}" y="${cy - 12}" text-anchor="end" font-size="16" font-weight="700" fill="${C.ink}">Kakeya · exact maximal</text>
    <text x="70" y="${H - 38}" font-size="17" fill="#6b7b88">Finite-field reader graphic · Dvir (2008). Body-resistance reading — not Euclidean evidence, not a proof, not a regime-2 claim.</text>
    <text x="${W - 70}" y="${H - 38}" text-anchor="end" font-size="17" font-weight="700" fill="${C.navyMid}">sundog.cc</text>
  </svg>`;
}

function exportPng() {
  const entry = selectedEntry || GALLERY[0];
  if (!entry) return;
  const svg = buildShareCardSVG(entry);
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = 1200 * scale;
    canvas.height = 675 * scale;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    canvas.toBlob((png) => {
      if (!png) return;
      const a = document.createElement("a");
      const href = URL.createObjectURL(png);
      a.href = href;
      a.download = `kakeya-body-resistance-${entry.id}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(href), 2000);
    }, "image/png");
  };
  img.onerror = () => { URL.revokeObjectURL(url); };
  img.src = url;
}

function init() {
  $("continuum").innerHTML = continuumSVG();
  renderGallery();
  setupTooltips();
  const replay = $("btn-replay");
  if (replay) replay.addEventListener("click", () => { if (selectedEntry) renderStage(selectedEntry); });
  const exportBtn = $("btn-export-png");
  if (exportBtn) exportBtn.addEventListener("click", exportPng);
  const headline = GALLERY.find((e) => e.id === "greedy-7") || GALLERY[0];
  if (headline) renderStage(headline);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
