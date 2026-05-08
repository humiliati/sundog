import {
  BALANCE_PRESETS,
  computeBalanceControl,
  computeShadowGeometry,
  createBalanceRuntime,
  initializeBalanceState,
  integrateBalanceStep,
  normalizeBalanceConfig,
  sampleShadowSensor,
  serializeBalanceSample,
} from "./balance-core.mjs";

const canvas = document.getElementById("balance-canvas");
const ctx = canvas.getContext("2d");
const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
const presetSelect = document.getElementById("preset-select");
const playPauseButton = document.getElementById("btn-play-pause");
const resetButton = document.getElementById("btn-reset");
const disturbButton = document.getElementById("btn-disturb");
const lightElevationInput = document.getElementById("light-elevation");
const noiseInput = document.getElementById("sensor-noise");
const delayInput = document.getElementById("sensor-delay");
const forceInput = document.getElementById("force-limit");
const diagnosticsCheckbox = document.getElementById("show-diagnostics");
const statusDisplay = document.getElementById("balance-status");
const metricsDisplay = document.getElementById("balance-metrics");
const phaseDisplay = document.getElementById("phase-display");

const history = [];
const maxHistory = 260;
let isPlaying = true;
let currentMode = "passive";
let state = null;
let runtime = null;
let controllerState = {};
let currentSensor = null;
let currentControl = { force: 0, rawForce: 0, saturated: false, phase: "PASSIVE", reason: "no force" };
let disturbanceTicks = 0;
let lastFrameTime = performance.now();

for (const [key, preset] of Object.entries(BALANCE_PRESETS)) {
  const option = document.createElement("option");
  option.value = key;
  option.textContent = preset.label;
  presetSelect.appendChild(option);
}

function currentConfig(overrides = {}) {
  return normalizeBalanceConfig({
    preset: presetSelect.value,
    controllerMode: currentMode,
    lightElevationDeg: Number.parseFloat(lightElevationInput.value),
    sensorNoiseStd: Number.parseFloat(noiseInput.value),
    sensorDelaySteps: Number.parseInt(delayInput.value, 10),
    forceLimit: Number.parseFloat(forceInput.value),
    seed: 20260508,
    ...overrides,
  });
}

function resetSimulation() {
  const cfg = currentConfig();
  state = initializeBalanceState(cfg);
  runtime = createBalanceRuntime(cfg);
  controllerState = {};
  currentSensor = sampleShadowSensor(state, runtime, cfg);
  currentControl = { force: 0, rawForce: 0, saturated: false, phase: "PASSIVE", reason: "reset" };
  disturbanceTicks = 0;
  history.length = 0;
  isPlaying = true;
  playPauseButton.textContent = "Pause";
  updateModeButtons();
}

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * dpr));
  canvas.height = Math.max(420, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function worldToScreen(x, z = 0) {
  const cssWidth = canvas.width / (window.devicePixelRatio || 1);
  const cssHeight = canvas.height / (window.devicePixelRatio || 1);
  const scale = Math.min(cssWidth / 6.2, cssHeight / 3.7);
  const originX = cssWidth * 0.5;
  const floorY = cssHeight * 0.68;
  return [originX + x * scale, floorY - z * scale, scale, floorY];
}

function drawBackground() {
  const cssWidth = canvas.width / (window.devicePixelRatio || 1);
  const cssHeight = canvas.height / (window.devicePixelRatio || 1);
  const gradient = ctx.createLinearGradient(0, 0, cssWidth, cssHeight);
  gradient.addColorStop(0, "#102a3d");
  gradient.addColorStop(0.55, "#1A3A52");
  gradient.addColorStop(1, "#314956");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  // Robotics balance motif derived with permission from the Lounas portfolio
  // reference. See docs/THIRD_PARTY_REUSE.md for the reuse ledger.
  const cx = cssWidth * 0.18;
  const cy = cssHeight * 0.26;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.strokeStyle = "rgba(244, 196, 48, 0.13)";
  ctx.lineWidth = 1.4;
  for (let i = 0; i < 5; i += 1) {
    ctx.beginPath();
    ctx.ellipse(0, 0, 56 + i * 18, 18 + i * 8, i * 0.45 + state.t * 0.15, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  for (let i = 0; i < 4; i += 1) {
    ctx.beginPath();
    ctx.arc(0, 0, 22 + i * 20, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  for (let x = -80; x < cssWidth + 80; x += 42) {
    ctx.beginPath();
    ctx.moveTo(x, cssHeight * 0.72);
    ctx.lineTo(x + 180, cssHeight);
    ctx.stroke();
  }
}

function drawRail() {
  const cfg = currentConfig();
  const [leftX, floorY] = worldToScreen(-cfg.railLimit, 0);
  const [rightX] = worldToScreen(cfg.railLimit, 0);
  ctx.strokeStyle = "rgba(245, 245, 245, 0.72)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(leftX, floorY);
  ctx.lineTo(rightX, floorY);
  ctx.stroke();

  ctx.strokeStyle = "rgba(244, 196, 48, 0.35)";
  ctx.lineWidth = 2;
  for (const x of [-cfg.railLimit, cfg.railLimit]) {
    const [sx, sy] = worldToScreen(x, 0);
    ctx.beginPath();
    ctx.moveTo(sx, sy - 18);
    ctx.lineTo(sx, sy + 18);
    ctx.stroke();
  }
}

function drawLightAndShadow() {
  const cfg = currentConfig();
  const raw = currentSensor?.raw ?? computeShadowGeometry(state, cfg);
  const [baseX, floorY, scale] = worldToScreen(raw.baseShadowX, 0);
  const [tipX] = worldToScreen(raw.shadowTipX, 0);
  const [uprightX] = worldToScreen(raw.uprightShadowTipX, 0);
  const [poleTipX, poleTipY] = worldToScreen(raw.poleTipX, raw.poleTipZ);

  ctx.strokeStyle = "rgba(0, 0, 0, 0.38)";
  ctx.lineWidth = Math.max(4, 10 * raw.confidence);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(baseX, floorY + 9);
  ctx.lineTo(tipX, floorY + 9);
  ctx.stroke();

  ctx.strokeStyle = "rgba(244, 196, 48, 0.65)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(uprightX, floorY - 18);
  ctx.lineTo(uprightX, floorY + 26);
  ctx.stroke();

  const elevation = Number.parseFloat(lightElevationInput.value) * Math.PI / 180;
  const sign = Math.sign(cfg.lightAzimuthSign || 1);
  const rayDx = -Math.cos(elevation) * sign * scale * 0.8;
  const rayDy = Math.sin(elevation) * scale * 0.8;
  ctx.strokeStyle = "rgba(244, 196, 48, 0.28)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(poleTipX + rayDx, poleTipY - rayDy);
  ctx.lineTo(poleTipX, poleTipY);
  ctx.stroke();
}

function drawCartPole() {
  const cfg = currentConfig();
  const [cartX, floorY, scale] = worldToScreen(state.x, 0);
  const cartW = 0.42 * scale;
  const cartH = 0.22 * scale;
  const poleBaseY = floorY - cartH * 0.8;
  const tipX = cartX + Math.sin(state.theta) * cfg.poleLength * scale;
  const tipY = poleBaseY - Math.cos(state.theta) * cfg.poleLength * scale;

  ctx.fillStyle = "rgba(245, 245, 245, 0.95)";
  ctx.strokeStyle = "rgba(244, 196, 48, 0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(cartX - cartW / 2, floorY - cartH, cartW, cartH, 6);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#1A3A52";
  for (const dx of [-cartW * 0.32, cartW * 0.32]) {
    ctx.beginPath();
    ctx.arc(cartX + dx, floorY + 2, 7, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = "#F4C430";
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(cartX, poleBaseY);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();

  ctx.fillStyle = "#FFFFFF";
  ctx.beginPath();
  ctx.arc(tipX, tipY, 8, 0, Math.PI * 2);
  ctx.fill();

  if (diagnosticsCheckbox.checked) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(cartX, poleBaseY);
    ctx.lineTo(cartX, poleBaseY - cfg.poleLength * scale);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (Math.abs(currentControl.force) > 0.05) {
    const arrowLength = currentControl.force * 8;
    ctx.strokeStyle = currentControl.saturated ? "#ff8b6a" : "rgba(125, 162, 184, 0.95)";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(cartX, floorY - cartH * 1.65);
    ctx.lineTo(cartX + arrowLength, floorY - cartH * 1.65);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cartX + arrowLength, floorY - cartH * 1.65, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawCharts() {
  const cssWidth = canvas.width / (window.devicePixelRatio || 1);
  const cssHeight = canvas.height / (window.devicePixelRatio || 1);
  const chartX = cssWidth * 0.08;
  const chartY = cssHeight * 0.78;
  const chartW = cssWidth * 0.84;
  const chartH = cssHeight * 0.15;
  ctx.fillStyle = "rgba(8, 20, 30, 0.42)";
  ctx.strokeStyle = "rgba(244, 196, 48, 0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(chartX, chartY, chartW, chartH, 8);
  ctx.fill();
  ctx.stroke();

  const series = [
    { key: "theta", color: "#F4C430", scale: 0.65, label: "theta" },
    { key: "shadowResidual", color: "#7DA2B8", scale: 1.4, label: "shadow" },
    { key: "force", color: "#ff8b6a", scale: 12, label: "force" },
  ];

  for (const item of series) {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((sample, index) => {
      const px = chartX + (index / Math.max(1, maxHistory - 1)) * chartW;
      const value = clamp(sample[item.key] / item.scale, -1, 1);
      const py = chartY + chartH / 2 - value * chartH * 0.42;
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(245,245,245,0.75)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Consolas, monospace";
  ctx.fillText("theta", chartX + 12, chartY + 18);
  ctx.fillStyle = "#7DA2B8";
  ctx.fillText("shadow", chartX + 62, chartY + 18);
  ctx.fillStyle = "#ff8b6a";
  ctx.fillText("force", chartX + 126, chartY + 18);
}

function draw() {
  drawBackground();
  drawRail();
  drawLightAndShadow();
  drawCartPole();
  drawCharts();
}

function updateMetrics() {
  const sample = serializeBalanceSample(state, currentSensor, currentControl, currentConfig());
  const terminal = state.fallen
    ? "fallen"
    : state.railHit
      ? "rail hit"
      : state.t >= currentConfig().duration
        ? "timeout"
        : "running";
  statusDisplay.textContent = `${terminal.toUpperCase()} | ${currentControl.phase} | ${currentControl.reason}`;
  phaseDisplay.textContent = currentControl.phase;
  metricsDisplay.innerHTML = `
    <span>t ${sample.t.toFixed(2)}s</span>
    <span>theta ${sample.theta.toFixed(3)} rad</span>
    <span>shadow ${sample.shadowResidual.toFixed(3)}</span>
    <span>confidence ${sample.shadowConfidence.toFixed(2)}</span>
    <span>force ${sample.force.toFixed(2)}</span>
  `;
}

function updateModeButtons() {
  modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === currentMode);
  });
}

function stepSimulation() {
  const cfg = currentConfig();
  currentSensor = sampleShadowSensor(state, runtime, cfg);
  currentControl = computeBalanceControl(state, currentSensor, controllerState, cfg);
  const disturbance = disturbanceTicks > 0 ? -Math.sign(state.theta || 1) * 4.5 : 0;
  if (disturbanceTicks > 0) disturbanceTicks -= 1;
  state = integrateBalanceStep(state, currentControl.force + disturbance, cfg);

  history.push({
    theta: state.theta,
    shadowResidual: currentSensor.residual,
    force: currentControl.force / cfg.forceLimit,
  });
  if (history.length > maxHistory) history.shift();

  if (state.fallen || state.railHit || state.t >= cfg.duration) {
    isPlaying = false;
    playPauseButton.textContent = "Resume";
  }
}

function animate(now) {
  const elapsed = Math.min(0.08, (now - lastFrameTime) / 1000);
  lastFrameTime = now;
  if (isPlaying) {
    const cfg = currentConfig();
    const steps = Math.max(1, Math.floor(elapsed / cfg.dt));
    for (let i = 0; i < Math.min(8, steps); i += 1) {
      stepSimulation();
    }
  }
  draw();
  updateMetrics();
  requestAnimationFrame(animate);
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    currentMode = button.dataset.mode;
    controllerState = {};
    updateModeButtons();
  });
});

playPauseButton.addEventListener("click", () => {
  isPlaying = !isPlaying;
  playPauseButton.textContent = isPlaying ? "Pause" : "Resume";
});

resetButton.addEventListener("click", resetSimulation);
disturbButton.addEventListener("click", () => {
  disturbanceTicks = 18;
});

for (const input of [presetSelect, lightElevationInput, noiseInput, delayInput, forceInput]) {
  input.addEventListener("input", () => {
    document.querySelector(`[data-value-for="${input.id}"]`).textContent = input.value;
    if (input === presetSelect) resetSimulation();
  });
}

window.addEventListener("resize", () => {
  resizeCanvas();
  draw();
});

resizeCanvas();
resetSimulation();
requestAnimationFrame(animate);

