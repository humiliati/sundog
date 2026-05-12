/**
 * photo-upload.mjs — Phase 5A frontend.
 *
 * Three-click measurement UX over a user-uploaded sundog photograph. Reads the
 * sun pixel, the 22° halo edge, and a parhelion. Inverse-infers sun altitude
 * via `arccos(R₂₂ / parhelion_offset)`, then renders the full atlas onto a
 * canvas overlaid on the photo. Strips EXIF before any future POST by going
 * through canvas.toBlob.
 *
 * Atlas math: reuses `phase3` from parhelion-geometry.mjs for the altitude
 * binding; the drawing primitives are local to this module so we can target a
 * canvas instead of the SVG primitives the workbench uses.
 *
 * No network I/O — Phase 5B will wire the consent checkbox to POST.
 */

import { phase3 } from "./parhelion-geometry.mjs";

const STAGES = Object.freeze({
  IDLE: "idle",
  PICK_SUN: "pick-sun",
  PICK_HALO: "pick-halo",
  PICK_PARHELION: "pick-parhelion",
  DONE: "done",
});

const MAX_LONG_EDGE = 1600; // re-encode cap before storage / overlay

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// ---------- atlas drawing primitives on canvas -----------------------------

function drawCircleOnCanvas(ctx, cx, cy, r, { stroke, width = 2, dashed = false } = {}) {
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  if (dashed) ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawArcOnCanvas(ctx, cx, cy, r, startAngle, endAngle, { stroke, width = 2, dashed = false } = {}) {
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  if (dashed) ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.stroke();
  ctx.restore();
}

function drawCross(ctx, x, y, color, size = 8) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - size, y); ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size); ctx.lineTo(x, y + size);
  ctx.stroke();
  ctx.restore();
}

function drawDot(ctx, x, y, color, r = 6) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// ---------- atlas overlay (the actual primitives) --------------------------

function drawAtlas(ctx, anchors) {
  const { sun, r22, hDeg, showCza, showSupra, showInfra, showTangent } = anchors;
  const r46 = 2 * r22;

  // 22° halo (red)
  drawCircleOnCanvas(ctx, sun.x, sun.y, r22, { stroke: "rgba(255,80,80,0.85)", width: 2 });
  // 46° halo (blue)
  drawCircleOnCanvas(ctx, sun.x, sun.y, r46, { stroke: "rgba(80,180,255,0.78)", width: 2 });

  // Daggers
  const hRad = hDeg * Math.PI / 180;
  const offset = r22 / Math.cos(hRad);
  const left = { x: sun.x - offset, y: sun.y };
  const right = { x: sun.x + offset, y: sun.y };
  drawCross(ctx, left.x,  left.y,  "rgba(255,255,80,1)", 9);
  drawCross(ctx, right.x, right.y, "rgba(255,255,80,1)", 9);

  // Parhelic circle (smile through dagger-sun-dagger). Use a tiny apex below sun
  // — parhelic-curvature 0.05 in workbench coords = 200·0.05·scale_to_photo.
  const apexBelow = 200 * 0.05 * (r22 / 220);
  if (apexBelow < 0.5) {
    ctx.save();
    ctx.strokeStyle = "rgba(255,150,50,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(left.x, left.y);
    ctx.lineTo(right.x, right.y);
    ctx.stroke();
    ctx.restore();
  } else {
    const halfChord = offset;
    const u = (halfChord * halfChord - apexBelow * apexBelow) / (2 * apexBelow);
    const cy = sun.y - u;
    const r  = Math.hypot(halfChord, u);
    const startAngle = Math.atan2(left.y - cy, left.x - sun.x);
    const endAngle   = Math.atan2(right.y - cy, right.x - sun.x);
    drawArcOnCanvas(ctx, sun.x, cy, r, startAngle, endAngle,
                    { stroke: "rgba(255,150,50,0.85)", width: 2 });
  }

  // CZA — tangent to 46° halo at top. Visible only when h ≤ 32°.
  if (showCza && phase3.czaVisible(hDeg)) {
    const apexY = sun.y - r46;
    // CZA circle: anchored apex y at top of 46° halo, endpoint band 240 in workbench
    // → translate via scale. Endpoint pulled to y = sun_y - r46 + 0.85·R22 (approx).
    const endpointY = apexY + 0.85 * r22;
    const halfWidth = 1.35 * r22;
    const num = halfWidth * halfWidth + endpointY * endpointY - apexY * apexY;
    const den = 2 * (endpointY - apexY);
    if (Math.abs(den) > 1e-6) {
      const ccy = num / den;
      const cr = Math.abs(apexY - ccy);
      // Sample upper-arc points across visible x range and stroke a polyline.
      const xMin = Math.max(sun.x - cr, 0);
      const xMax = Math.min(sun.x + cr, ctx.canvas.width);
      ctx.save();
      ctx.strokeStyle = "rgba(170,80,255,0.78)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= 160; i += 1) {
        const x = xMin + (xMax - xMin) * i / 160;
        const u = (x - sun.x) / cr;
        const inside = 1 - u * u;
        if (inside < 0) continue;
        const y = ccy - cr * Math.sqrt(inside);
        if (y > endpointY) continue;
        if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  // Supralateral arc (optional): tangent to 46° halo top, curving up
  if (showSupra) {
    const tangentY = sun.y - r46;
    const R_supra = (400 / 220) * r22;
    const cy = tangentY - R_supra;
    ctx.save();
    ctx.strokeStyle = "rgba(255,200,255,0.78)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    const xMin = Math.max(sun.x - R_supra, 0);
    const xMax = Math.min(sun.x + R_supra, ctx.canvas.width);
    for (let i = 0; i <= 160; i += 1) {
      const x = xMin + (xMax - xMin) * i / 160;
      const u = (x - sun.x) / R_supra;
      const inside = 1 - u * u;
      if (inside < 0) continue;
      const y = cy + R_supra * Math.sqrt(inside);
      if (y < 0 || y > tangentY + 5) continue;
      if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
    }
    ctx.stroke();
    ctx.restore();
  }

  // Infralateral (mirror of supralateral, tangent to 46° halo bottom)
  if (showInfra) {
    const tangentY = sun.y + r46;
    const R_infra = (400 / 220) * r22;
    const cy = tangentY + R_infra;
    ctx.save();
    ctx.strokeStyle = "rgba(255,200,255,0.7)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    const xMin = Math.max(sun.x - R_infra, 0);
    const xMax = Math.min(sun.x + R_infra, ctx.canvas.width);
    for (let i = 0; i <= 160; i += 1) {
      const x = xMin + (xMax - xMin) * i / 160;
      const u = (x - sun.x) / R_infra;
      const inside = 1 - u * u;
      if (inside < 0) continue;
      const y = cy - R_infra * Math.sqrt(inside);
      if (y < tangentY - 5 || y > ctx.canvas.height) continue;
      if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
    }
    ctx.stroke();
    ctx.restore();
  }

  // Upper tangent arc (optional, tangent to 22° halo at top)
  if (showTangent) {
    const tangentY = sun.y - r22;
    const R_uta = (200 / 220) * r22;
    const cy = tangentY - R_uta;
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    const xMin = Math.max(sun.x - R_uta, 0);
    const xMax = Math.min(sun.x + R_uta, ctx.canvas.width);
    for (let i = 0; i <= 160; i += 1) {
      const x = xMin + (xMax - xMin) * i / 160;
      const u = (x - sun.x) / R_uta;
      const inside = 1 - u * u;
      if (inside < 0) continue;
      const y = cy + R_uta * Math.sqrt(inside);
      if (y < 0 || y > tangentY + 5) continue;
      if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
    }
    ctx.stroke();
    ctx.restore();
  }

  // Sun marker (small yellow dot)
  drawDot(ctx, sun.x, sun.y, "rgba(255, 224, 110, 1)", 5);
}

// ---------- the main widget controller -------------------------------------

export function mountPhotoUpload(rootEl) {
  if (!rootEl) return;

  rootEl.innerHTML = `
    <div class="upload-widget" role="region" aria-label="Sundog photo upload">
      <div class="upload-pick" data-stage="idle">
        <h3>Upload a sundog photo</h3>
        <p>The atlas runs entirely in your browser. Nothing is sent anywhere unless you opt in below.</p>
        <label class="upload-button">
          <input type="file" accept="image/jpeg,image/png,image/webp" id="upload-file-input" hidden>
          <span>Choose a photo…</span>
        </label>
        <p class="upload-hint">JPEG, PNG, or WebP. EXIF metadata is stripped before any optional sharing.</p>
      </div>

      <div class="upload-stage" data-stage="measure" hidden>
        <div class="upload-canvas-wrap">
          <canvas id="upload-canvas" aria-label="Sundog photo with atlas overlay"></canvas>
          <div class="upload-instructions" id="upload-instructions"></div>
        </div>
        <aside class="upload-readout">
          <h3>Measurements</h3>
          <dl id="upload-measurements"></dl>
          <h3 class="upload-altitude-heading" hidden>Inverse-inferred altitude</h3>
          <p class="upload-altitude" id="upload-altitude" hidden></p>
          <fieldset class="upload-layer-toggles" hidden id="upload-layer-toggles">
            <legend>Overlay layers</legend>
            <label><input type="checkbox" id="layer-cza" checked> CZA</label>
            <label><input type="checkbox" id="layer-supra"> Supralateral</label>
            <label><input type="checkbox" id="layer-infra"> Infralateral</label>
            <label><input type="checkbox" id="layer-tangent"> Upper tangent</label>
          </fieldset>
          <div class="upload-actions">
            <button type="button" id="upload-restart">Start over</button>
            <button type="button" id="upload-download" hidden>Download overlay PNG</button>
            <button type="button" id="upload-copy-pose" hidden>Copy JSON pose</button>
          </div>
          <div class="upload-share" hidden id="upload-share">
            <label class="upload-share-toggle">
              <input type="checkbox" id="upload-share-consent" disabled>
              <span>Share my photo with the Sundog project for atlas-model training
              <em id="upload-share-note">checking backend…</em></span>
            </label>
            <p class="upload-share-policy">
              EXIF stripped before upload. You receive a deletion token at submission time.
              <a href="/docs/PHOTO_DATA_POLICY.md">Read the policy</a>.
            </p>
            <div class="upload-share-actions" hidden id="upload-share-actions">
              <button type="button" id="upload-share-submit">Submit to project</button>
              <span class="upload-share-status" id="upload-share-status"></span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  `;

  // ---- query references -------------------------------------------------

  const fileInput     = rootEl.querySelector("#upload-file-input");
  const pickPane      = rootEl.querySelector(".upload-pick");
  const stagePane     = rootEl.querySelector(".upload-stage");
  const canvas        = rootEl.querySelector("#upload-canvas");
  const instructions  = rootEl.querySelector("#upload-instructions");
  const measurements  = rootEl.querySelector("#upload-measurements");
  const altitudeHead  = rootEl.querySelector(".upload-altitude-heading");
  const altitudePara  = rootEl.querySelector("#upload-altitude");
  const restartBtn    = rootEl.querySelector("#upload-restart");
  const downloadBtn   = rootEl.querySelector("#upload-download");
  const copyPoseBtn   = rootEl.querySelector("#upload-copy-pose");
  const layerToggles  = rootEl.querySelector("#upload-layer-toggles");
  const shareWrap     = rootEl.querySelector("#upload-share");
  const layerCza      = rootEl.querySelector("#layer-cza");
  const layerSupra    = rootEl.querySelector("#layer-supra");
  const layerInfra    = rootEl.querySelector("#layer-infra");
  const layerTangent  = rootEl.querySelector("#layer-tangent");

  const ctx = canvas.getContext("2d");

  // ---- state -------------------------------------------------------------

  const state = {
    stage: STAGES.IDLE,
    image: null,          // HTMLImageElement (decoded, stripped of EXIF)
    sun: null,            // { x, y }
    haloEdge: null,       // { x, y }  → r22 = distance(sun, haloEdge)
    parhelion: null,      // { x, y }  → offset = |parhelion.x - sun.x|
    cssScale: 1,          // CSS-px ↔ canvas-px multiplier (DPR-aware mapping)
  };

  function updateInstructions(text) { instructions.textContent = text; }

  function setStage(stage) {
    state.stage = stage;
    canvas.classList.remove("is-picking-sun", "is-picking-halo", "is-picking-parhelion");
    switch (stage) {
      case STAGES.PICK_SUN:
        updateInstructions("1 / 3 — Click the sun's center.");
        canvas.classList.add("is-picking-sun");
        break;
      case STAGES.PICK_HALO:
        updateInstructions("2 / 3 — Click anywhere on the 22° halo (the bright ring around the sun).");
        canvas.classList.add("is-picking-halo");
        break;
      case STAGES.PICK_PARHELION:
        updateInstructions("3 / 3 — Click a visible parhelion (bright spot left or right of the sun).");
        canvas.classList.add("is-picking-parhelion");
        break;
      case STAGES.DONE:
        updateInstructions("Done. Drag a marker if you want to refine, or start over.");
        break;
      default:
        updateInstructions("");
    }
  }

  function renderMeasurements() {
    const r22 = state.sun && state.haloEdge ? distance(state.sun, state.haloEdge) : null;
    const offset = state.sun && state.parhelion ? Math.abs(state.parhelion.x - state.sun.x) : null;
    let rows = "";
    rows += `<dt>Sun pixel</dt><dd>${state.sun ? `(${state.sun.x.toFixed(0)}, ${state.sun.y.toFixed(0)})` : "—"}</dd>`;
    rows += `<dt>R₂₂ (halo radius)</dt><dd>${r22 ? `${r22.toFixed(1)} px` : "—"}</dd>`;
    rows += `<dt>Parhelion offset</dt><dd>${offset ? `${offset.toFixed(1)} px` : "—"}</dd>`;
    measurements.innerHTML = rows;
  }

  function computeAltitude() {
    if (!state.sun || !state.haloEdge || !state.parhelion) return null;
    const r22 = distance(state.sun, state.haloEdge);
    const offset = Math.abs(state.parhelion.x - state.sun.x);
    if (offset <= r22) {
      // Parhelion sits inside (or on) the halo — sun is at the horizon
      return 0;
    }
    return Math.acos(r22 / offset) * 180 / Math.PI;
  }

  function renderAll() {
    // base photo
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.image) ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);

    // markers and atlas overlay
    if (state.sun)       drawCross(ctx, state.sun.x,      state.sun.y,      "rgba(0, 255, 255, 1)", 10);
    if (state.haloEdge)  drawCross(ctx, state.haloEdge.x, state.haloEdge.y, "rgba(255, 80, 80, 1)", 10);
    if (state.parhelion) drawCross(ctx, state.parhelion.x, state.parhelion.y, "rgba(255, 255, 80, 1)", 10);

    if (state.sun && state.haloEdge) {
      const r22 = distance(state.sun, state.haloEdge);
      drawCircleOnCanvas(ctx, state.sun.x, state.sun.y, r22, {
        stroke: "rgba(255, 80, 80, 0.45)", width: 1, dashed: true,
      });
    }

    if (state.stage === STAGES.DONE) {
      const hDeg = computeAltitude();
      const r22 = distance(state.sun, state.haloEdge);
      drawAtlas(ctx, {
        sun: state.sun,
        r22,
        hDeg,
        showCza:      layerCza.checked,
        showSupra:    layerSupra.checked,
        showInfra:    layerInfra.checked,
        showTangent:  layerTangent.checked,
      });
    }
  }

  function handleClick(ev) {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * (canvas.width  / rect.width);
    const y = (ev.clientY - rect.top)  * (canvas.height / rect.height);

    if (state.stage === STAGES.PICK_SUN) {
      state.sun = { x, y };
      setStage(STAGES.PICK_HALO);
    } else if (state.stage === STAGES.PICK_HALO) {
      state.haloEdge = { x, y };
      setStage(STAGES.PICK_PARHELION);
    } else if (state.stage === STAGES.PICK_PARHELION) {
      state.parhelion = { x, y };
      setStage(STAGES.DONE);
      const h = computeAltitude();
      altitudeHead.hidden = false;
      altitudePara.hidden = false;
      altitudePara.innerHTML = `Your sun was at <strong>h ≈ ${h.toFixed(1)}°</strong>.`;
      layerToggles.hidden = false;
      downloadBtn.hidden = false;
      copyPoseBtn.hidden = false;
      shareWrap.hidden = false;
    }
    renderMeasurements();
    renderAll();
  }

  canvas.addEventListener("click", handleClick);

  // ---- EXIF strip + load -------------------------------------------------

  async function loadImageStripExif(file) {
    const arrayBuffer = await file.arrayBuffer();
    const blob = new Blob([arrayBuffer], { type: file.type });
    const url = URL.createObjectURL(blob);
    const tempImg = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });

    // Scale down if oversize, write to an off-screen canvas, re-encode through
    // toBlob — that path drops EXIF / GPS / camera-id metadata. The blob we
    // pass forward to Phase 5B's POST will already be metadata-stripped.
    let w = tempImg.naturalWidth, h = tempImg.naturalHeight;
    const longEdge = Math.max(w, h);
    if (longEdge > MAX_LONG_EDGE) {
      const k = MAX_LONG_EDGE / longEdge;
      w = Math.round(w * k); h = Math.round(h * k);
    }
    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    off.getContext("2d").drawImage(tempImg, 0, 0, w, h);
    const cleanBlob = await new Promise((resolve) => off.toBlob(resolve, "image/jpeg", 0.88));
    URL.revokeObjectURL(url);

    const cleanUrl = URL.createObjectURL(cleanBlob);
    const cleanImg = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = cleanUrl;
    });
    return { img: cleanImg, blob: cleanBlob, width: w, height: h };
  }

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    try {
      const { img, blob, width, height } = await loadImageStripExif(file);
      state.image = img;
      canvas.width = width;
      canvas.height = height;
      // Fit canvas into available CSS width while preserving aspect ratio.
      // Actual pixel coords stay in image-space; CSS shrinks for display.
      canvas.style.maxWidth = "100%";
      canvas.style.height = "auto";
      state._cleanBlob = blob;

      pickPane.hidden = true;
      stagePane.hidden = false;
      // Reset markers
      state.sun = state.haloEdge = state.parhelion = null;
      altitudeHead.hidden = altitudePara.hidden = true;
      layerToggles.hidden = downloadBtn.hidden = copyPoseBtn.hidden = shareWrap.hidden = true;
      renderMeasurements();
      setStage(STAGES.PICK_SUN);
      renderAll();
    } catch (err) {
      console.error("photo load failed", err);
      alert("Could not load that image. Try a JPEG, PNG, or WebP under 20 MB.");
    }
  });

  restartBtn.addEventListener("click", () => {
    state.image = null;
    state.sun = state.haloEdge = state.parhelion = null;
    state._cleanBlob = null;
    pickPane.hidden = false;
    stagePane.hidden = true;
    fileInput.value = "";
    setStage(STAGES.IDLE);
  });

  downloadBtn.addEventListener("click", () => {
    canvas.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "sundog-overlay.png";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, "image/png");
  });

  copyPoseBtn.addEventListener("click", async () => {
    const r22 = distance(state.sun, state.haloEdge);
    const offset = Math.abs(state.parhelion.x - state.sun.x);
    const hDeg = computeAltitude();
    const pose = {
      schemaVersion: "1.0",
      source: "photo-upload",
      capturedAt: new Date().toISOString(),
      anchors: {
        sun_px: [Math.round(state.sun.x), Math.round(state.sun.y)],
        halo_edge_px: [Math.round(state.haloEdge.x), Math.round(state.haloEdge.y)],
        parhelion_px: [Math.round(state.parhelion.x), Math.round(state.parhelion.y)],
        r22_px: Number(r22.toFixed(2)),
        parhelion_offset_px: Number(offset.toFixed(2)),
      },
      inferred: { sun_altitude_deg: Number(hDeg.toFixed(2)) },
      atlas: {
        geometryModel: "halo_atlas",
        sunAltitudeDeg: Number(hDeg.toFixed(2)),
        halo22AnchorPx: Number(r22.toFixed(2)),
      },
      layers: {
        cza: layerCza.checked, supralateral: layerSupra.checked,
        infralateral: layerInfra.checked, upperTangent: layerTangent.checked,
      },
    };
    const text = JSON.stringify(pose, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      copyPoseBtn.textContent = "Copied \u2713";
      setTimeout(() => { copyPoseBtn.textContent = "Copy JSON pose"; }, 1600);
    } catch {
      console.log("Sundog pose JSON:", text);
      copyPoseBtn.textContent = "Logged to console";
      setTimeout(() => { copyPoseBtn.textContent = "Copy JSON pose"; }, 1600);
    }
  });

  [layerCza, layerSupra, layerInfra, layerTangent].forEach((cb) => {
    cb.addEventListener("change", renderAll);
  });

  // ---- Phase 5B: share-for-training backend wiring ----------------------

  const shareConsent = rootEl.querySelector("#upload-share-consent");
  const shareNote    = rootEl.querySelector("#upload-share-note");
  const shareActions = rootEl.querySelector("#upload-share-actions");
  const shareSubmit  = rootEl.querySelector("#upload-share-submit");
  const shareStatus  = rootEl.querySelector("#upload-share-status");

  let backendHealth = null;
  let policyVersion = null;

  async function probeBackend() {
    try {
      const res = await fetch("/api/sundog/health", { headers: { accept: "application/json" } });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const j = await res.json();
      backendHealth = !!j?.ok;
      policyVersion = j?.policy_version || null;
      shareConsent.disabled = !backendHealth;
      shareNote.textContent = backendHealth
        ? `(policy version ${policyVersion})`
        : "(offline \u2014 server bindings not configured)";
    } catch (err) {
      backendHealth = false;
      shareConsent.disabled = true;
      shareNote.textContent = "(offline \u2014 try again later)";
    }
  }

  shareConsent.addEventListener("change", () => {
    shareActions.hidden = !shareConsent.checked;
    shareStatus.textContent = "";
  });

  shareSubmit.addEventListener("click", async () => {
    if (!shareConsent.checked || !backendHealth) return;
    shareSubmit.disabled = true;
    shareStatus.textContent = "Uploading\u2026";

    try {
      const blob = state._cleanBlob;
      if (!blob) throw new Error("no cleaned image blob");
      const buf = await blob.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let binary = "";
      for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
      const b64 = btoa(binary);

      const r22 = distance(state.sun, state.haloEdge);
      const offset = Math.abs(state.parhelion.x - state.sun.x);
      const hDeg = computeAltitude();

      const payload = {
        image: { data: b64, mime: blob.type || "image/jpeg", byte_length: bytes.length },
        pose: {
          schemaVersion: "1.0",
          source: "photo-upload",
          capturedAt: new Date().toISOString(),
          anchors: {
            sun_px: [Math.round(state.sun.x), Math.round(state.sun.y)],
            halo_edge_px: [Math.round(state.haloEdge.x), Math.round(state.haloEdge.y)],
            parhelion_px: [Math.round(state.parhelion.x), Math.round(state.parhelion.y)],
            r22_px: Number(r22.toFixed(2)),
            parhelion_offset_px: Number(offset.toFixed(2)),
          },
          inferred: { sun_altitude_deg: Number(hDeg.toFixed(2)) },
          layers: {
            cza: layerCza.checked, supralateral: layerSupra.checked,
            infralateral: layerInfra.checked, upperTangent: layerTangent.checked,
          },
        },
        consent: {
          share_for_training: true,
          agreed_at: new Date().toISOString(),
          agreed_to_policy_version: policyVersion,
        },
        client: { ua: navigator.userAgent, page: location.pathname, exif_stripped: true },
      };

      const res = await fetch("/api/sundog/upload", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j?.error?.message || `status ${res.status}`);

      try {
        const prev = JSON.parse(localStorage.getItem("sundog.submissions") || "[]");
        prev.push({
          submission_id: j.submission_id,
          deletion_url:  j.deletion_url,
          submitted_at:  new Date().toISOString(),
          inferred_h_deg: j.inferred_h_deg,
        });
        localStorage.setItem("sundog.submissions", JSON.stringify(prev));
      } catch { /* localStorage may be disabled */ }

      shareStatus.innerHTML = `Submitted. Submission <code>${j.submission_id.slice(0, 8)}\u2026</code> &middot; ` +
        `<a href="${j.deletion_url}" target="_blank" rel="noopener">save this deletion URL</a>`;
      shareSubmit.textContent = "Submitted \u2713";
    } catch (err) {
      console.error("upload failed:", err);
      shareStatus.textContent = "Upload failed: " + (err?.message || "unknown");
      shareSubmit.disabled = false;
    }
  });

  probeBackend();
  setStage(STAGES.IDLE);
}
