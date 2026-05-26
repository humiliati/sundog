import {
  BALANCE_PRESETS,
  assessBalanceBoundary,
  clamp,
  computeBalanceControl,
  computeShadowGeometry,
  createBalanceRuntime,
  initializeBalanceState,
  integrateBalanceStep,
  normalizeBalanceConfig,
  sampleShadowSensor,
  serializeBalanceSample,
} from "./balance-core.mjs?v=20260517a";

const canvas = document.getElementById("balance-canvas");
const ctx = canvas.getContext("2d");
const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
const presetSelect = document.getElementById("preset-select");
const seedInput = document.getElementById("seed-input");
const playPauseButton = document.getElementById("btn-play-pause");
const resetButton = document.getElementById("btn-reset");
const disturbButton = document.getElementById("btn-disturb");
const recoverySummary = document.getElementById("recovery-summary");
const recoveryCurve = document.getElementById("recovery-curve");
const recoveryStats = document.getElementById("recovery-stats");
const boundaryPanel = document.getElementById("boundary-panel");
const boundaryStatus = document.getElementById("boundary-status");
const boundarySummary = document.getElementById("boundary-summary");
const boundaryList = document.getElementById("boundary-list");
const lightElevationInput = document.getElementById("light-elevation");
const noiseInput = document.getElementById("sensor-noise");
const delayInput = document.getElementById("sensor-delay");
const dropoutInput = document.getElementById("sensor-dropout");
const forceInput = document.getElementById("force-limit");
const railInput = document.getElementById("rail-limit");
const disturbanceForceInput = document.getElementById("disturbance-force");
const diagnosticsCheckbox = document.getElementById("show-diagnostics");
const copyReplayButton = document.getElementById("btn-copy-replay");
const refreshReplayButton = document.getElementById("btn-refresh-replay");
const replayToken = document.getElementById("replay-token");
const statusDisplay = document.getElementById("balance-status");
const metricsDisplay = document.getElementById("balance-metrics");
const phaseDisplay = document.getElementById("phase-display");
const evidencePanel = document.getElementById("balance-evidence-panel");
const evidenceStatus = document.getElementById("balance-evidence-status");
const evidenceSummary = document.getElementById("balance-evidence-summary");
const evidenceMetrics = document.getElementById("balance-evidence-metrics");
const admissionLanes = document.getElementById("balance-admission-lanes");
const evidenceBoundaries = document.getElementById("balance-evidence-boundaries");

const history = [];
const maxHistory = 260;
const recoveryThresholds = Object.freeze({
  theta: 0.09,
  thetaDot: 0.35,
  holdSeconds: 0.25,
  impulseTicks: 18,
  impulseForce: 4.5,
});
let isPlaying = true;
let currentMode = "passive";
let state = null;
let runtime = null;
let controllerState = {};
let currentSensor = null;
let currentControl = { force: 0, rawForce: 0, saturated: false, phase: "PASSIVE", reason: "no force" };
let disturbanceTicks = 0;
let lastFrameTime = performance.now();
let initialStateOverride = null;
let durationOverride = null;
let balanceEvidenceData = null;
let recoveryTracker = createRecoveryTracker();
const evidenceIntegerFormatter = new Intl.NumberFormat("en-US");

for (const [key, preset] of Object.entries(BALANCE_PRESETS)) {
  const option = document.createElement("option");
  option.value = key;
  option.textContent = preset.label;
  presetSelect.appendChild(option);
}

function finiteNumber(value, fallback = null) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function integerNumber(value, fallback = null) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function queryNumber(params, names, fallback = null) {
  for (const name of names) {
    if (params.has(name)) return finiteNumber(params.get(name), fallback);
  }
  return fallback;
}

function readReplaySettings() {
  const params = new URLSearchParams(window.location.search);
  const preset = BALANCE_PRESETS[params.get("preset")] ? params.get("preset") : null;
  const mode = modeButtons.some((button) => button.dataset.mode === params.get("mode"))
    ? params.get("mode")
    : null;
  const seed = queryNumber(params, ["seed"], null);
  const initialState = ["x", "xDot", "theta", "thetaDot"].every((key) => params.has(key))
    ? {
        x: finiteNumber(params.get("x")),
        xDot: finiteNumber(params.get("xDot")),
        theta: finiteNumber(params.get("theta")),
        thetaDot: finiteNumber(params.get("thetaDot")),
      }
    : null;

  return {
    preset,
    mode,
    seed,
    lightElevationDeg: queryNumber(params, ["light", "elevation", "lightElevation"], null),
    forceLimit: queryNumber(params, ["force", "forceLimit"], null),
    railLimit: queryNumber(params, ["rail", "railLimit"], null),
    sensorNoiseStd: queryNumber(params, ["noise", "sensorNoise", "sensorNoiseStd"], null),
    sensorDelaySteps: queryNumber(params, ["delay", "sensorDelay", "sensorDelaySteps"], null),
    sensorDropoutRate: queryNumber(params, ["dropout", "sensorDropout", "sensorDropoutRate"], null),
    disturbanceForce: queryNumber(params, ["disturbanceForce", "impulseForce"], null),
    duration: queryNumber(params, ["duration"], null),
    initialState: initialState && Object.values(initialState).every(Number.isFinite) ? initialState : null,
  };
}

function currentConfig(overrides = {}) {
  return normalizeBalanceConfig({
    preset: presetSelect.value,
    controllerMode: currentMode,
    lightElevationDeg: Number.parseFloat(lightElevationInput.value),
    sensorNoiseStd: Number.parseFloat(noiseInput.value),
    sensorDelaySteps: Number.parseInt(delayInput.value, 10),
    sensorDropoutRate: Number.parseFloat(dropoutInput.value),
    forceLimit: Number.parseFloat(forceInput.value),
    railLimit: Number.parseFloat(railInput.value),
    disturbanceForce: currentImpulseForce(),
    seed: Math.max(0, integerNumber(seedInput.value, 20260508)),
    ...(durationOverride ? { duration: durationOverride } : {}),
    ...(initialStateOverride ? { initialState: initialStateOverride } : {}),
    ...overrides,
  });
}

function createRecoveryTracker() {
  return {
    hasImpulse: false,
    active: false,
    startedAt: null,
    endedAt: null,
    force: 0,
    peakAbsTheta: 0,
    peakTime: null,
    recoveredAt: null,
    recoveryHold: 0,
    terminal: null,
    curve: [],
  };
}

function currentImpulseForce() {
  const parsed = Number.parseFloat(disturbanceForceInput.value);
  return Number.isFinite(parsed) ? parsed : recoveryThresholds.impulseForce;
}

function beginRecoveryTrace() {
  recoveryTracker = createRecoveryTracker();
  recoveryTracker.hasImpulse = true;
  recoveryTracker.active = true;
  recoveryTracker.startedAt = state.t;
  recoveryTracker.force = -Math.sign(state.theta || 1) * currentImpulseForce();
  disturbanceTicks = recoveryThresholds.impulseTicks;
}

function resetSimulation() {
  const cfg = currentConfig();
  state = initializeBalanceState(cfg);
  runtime = createBalanceRuntime(cfg);
  controllerState = {};
  currentSensor = sampleShadowSensor(state, runtime, cfg);
  currentControl = { force: 0, rawForce: 0, saturated: false, phase: "PASSIVE", reason: "reset" };
  disturbanceTicks = 0;
  recoveryTracker = createRecoveryTracker();
  history.length = 0;
  isPlaying = true;
  playPauseButton.textContent = "Pause";
  updateModeButtons();
  updateReplayToken();
}

function syncControlValue(input) {
  const valueNode = document.querySelector(`[data-value-for="${input.id}"]`);
  if (valueNode) valueNode.textContent = input.value;
}

function applyPresetControlDefaults() {
  const preset = BALANCE_PRESETS[presetSelect.value];
  const presetConfig = preset?.config ?? {};
  const presetDefaults = normalizeBalanceConfig({ preset: presetSelect.value });
  noiseInput.value = Object.hasOwn(presetConfig, "sensorNoiseStd")
    ? presetConfig.sensorNoiseStd
    : presetDefaults.sensorNoiseStd;
  delayInput.value = Object.hasOwn(presetConfig, "sensorDelaySteps")
    ? presetConfig.sensorDelaySteps
    : presetDefaults.sensorDelaySteps;
  for (const input of [
    presetSelect,
    seedInput,
    lightElevationInput,
    noiseInput,
    delayInput,
    dropoutInput,
    forceInput,
    railInput,
    disturbanceForceInput,
  ]) {
    syncControlValue(input);
  }
}

function applyReplaySettings(settings) {
  if (settings.preset) presetSelect.value = settings.preset;
  if (settings.mode) currentMode = settings.mode;
  if (Number.isFinite(settings.seed)) seedInput.value = String(Math.max(0, Math.round(settings.seed)));
  if (Number.isFinite(settings.lightElevationDeg)) lightElevationInput.value = String(settings.lightElevationDeg);
  if (Number.isFinite(settings.forceLimit)) forceInput.value = String(settings.forceLimit);
  if (Number.isFinite(settings.railLimit) && settings.railLimit > 0) railInput.value = String(settings.railLimit);
  if (Number.isFinite(settings.sensorNoiseStd)) noiseInput.value = String(settings.sensorNoiseStd);
  if (Number.isFinite(settings.sensorDelaySteps)) delayInput.value = String(Math.round(settings.sensorDelaySteps));
  if (Number.isFinite(settings.sensorDropoutRate) && settings.sensorDropoutRate >= 0) dropoutInput.value = String(settings.sensorDropoutRate);
  if (Number.isFinite(settings.disturbanceForce) && settings.disturbanceForce >= 0) disturbanceForceInput.value = String(settings.disturbanceForce);
  if (Number.isFinite(settings.duration) && settings.duration > 0) durationOverride = settings.duration;
  if (settings.initialState) initialStateOverride = settings.initialState;
  for (const input of [
    presetSelect,
    seedInput,
    lightElevationInput,
    noiseInput,
    delayInput,
    dropoutInput,
    forceInput,
    railInput,
    disturbanceForceInput,
  ]) {
    syncControlValue(input);
  }
}

function getReplayInitialState() {
  if (initialStateOverride) return initialStateOverride;
  return BALANCE_PRESETS[presetSelect.value]?.state ?? BALANCE_PRESETS.easy.state;
}

function roundedParam(value, digits = 9) {
  return String(Number.isFinite(value) ? Number.parseFloat(value.toFixed(digits)) : value);
}

function formatMetric(value, digits = 2, suffix = "") {
  return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "n/a";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[char]);
}

function formatEvidenceInteger(value) {
  return Number.isFinite(value) ? evidenceIntegerFormatter.format(value) : "n/a";
}

function formatEvidenceNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "n/a";
  return String(Number.parseFloat(value.toFixed(digits)));
}

function formatEvidenceSigned(value, digits = 6) {
  if (!Number.isFinite(value)) return "n/a";
  const formatted = formatEvidenceNumber(value, digits);
  return value >= 0 ? `+${formatted}` : formatted;
}

function formatEvidenceDuration(seconds) {
  if (!Number.isFinite(seconds)) return "n/a";
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds - minutes * 60);
  return minutes > 0 ? `${minutes}m ${remaining}s` : `${remaining}s`;
}

function formatAxisName(axis) {
  return String(axis ?? "").replace(/_/g, " ");
}

function formatAxisValue(value) {
  if (!Number.isFinite(value)) return "n/a";
  return Number.parseFloat(value.toFixed(4)).toString();
}

function formatAxes(axes = {}) {
  const entries = Object.entries(axes);
  if (!entries.length) return "none";
  return entries
    .map(([axis, count]) => `${formatAxisName(axis)} ${count}`)
    .join(", ");
}

function laneSummary(lane) {
  if (lane === "hard_gate") return "Canonical operating-envelope cells admitted as claim evidence.";
  if (lane === "observation_parity_gate") return "Moderate sensor degradation admitted when Sundog parity holds.";
  if (lane === "reported_only") return "Severe observation degradation kept visible outside the hard claim.";
  return "Admission lane from the Phase 15 receipt.";
}

function metricCard(label, value, detail) {
  return `
    <div class="balance-evidence-metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <em>${escapeHtml(detail)}</em>
    </div>
  `;
}

function reportedBoundaryText(cells) {
  const groups = new Map();
  for (const cell of cells) {
    if (!groups.has(cell.axis)) groups.set(cell.axis, new Set());
    groups.get(cell.axis).add(formatAxisValue(cell.axisValue));
  }
  return [...groups.entries()]
    .map(([axis, values]) => `${formatAxisName(axis)} ${[...values].sort((a, b) => Number(a) - Number(b)).join(", ")}`)
    .join("; ");
}

function renderBalanceEvidence(data) {
  if (!evidencePanel) return;
  balanceEvidenceData = data;
  const phase15 = data?.phase15 ?? {};
  const claimGate = phase15.claimGate ?? {};
  const hardGateCells = phase15.hardGateCells ?? claimGate.hardGateCells;
  const hardGatePassCells = phase15.hardGatePassCells ?? claimGate.hardGatePassCells;
  const reportedOnlyCells = phase15.reportedOnlyCells ?? claimGate.reportedOnlyCells;
  const pass = claimGate.pass === true;

  evidencePanel.dataset.status = pass ? "pass" : "pending";
  if (evidenceStatus) evidenceStatus.textContent = pass ? "Claim Lock" : "Pending";
  if (evidenceSummary) {
    evidenceSummary.innerHTML = `
      Phase 15 passes the same-information Bayesian-floor claim gate across
      <strong>${escapeHtml(formatEvidenceInteger(hardGatePassCells))}/${escapeHtml(formatEvidenceInteger(hardGateCells))}</strong>
      hard-gate cells. The bundle keeps
      <strong>${escapeHtml(formatEvidenceInteger(reportedOnlyCells))}</strong>
      severe observation-degradation cells visible as reported-only boundary evidence.
    `;
  }

  if (evidenceMetrics) {
    evidenceMetrics.innerHTML = [
      metricCard("Hard gates", `${formatEvidenceInteger(hardGatePassCells)}/${formatEvidenceInteger(hardGateCells)}`, "claim cells passed"),
      metricCard("Reported-only", formatEvidenceInteger(reportedOnlyCells), "boundary cells retained"),
      metricCard("Mean regret", formatEvidenceSigned(phase15.meanRegretVsSundog, 6), "Bayes minus Sundog survival"),
      metricCard("Bayes sanity", `${formatEvidenceInteger(phase15.bayesSanityPassCells)}/${formatEvidenceInteger(phase15.cellCount)}`, "all cells in the receipt"),
      metricCard("Full lock", formatEvidenceInteger(phase15.trialCount), formatEvidenceDuration(phase15.elapsedSeconds)),
    ].join("");
  }

  const lanes = phase15.admissionLanes ?? [];
  if (admissionLanes) {
    admissionLanes.innerHTML = lanes.map((lane) => {
      const claimText = lane.hardGateCells > 0
        ? `${formatEvidenceInteger(lane.hardGatePassCells)}/${formatEvidenceInteger(lane.hardGateCells)}`
        : "reported";
      const sanityText = `${formatEvidenceInteger(lane.bayesSanityPassCells)}/${formatEvidenceInteger(lane.bayesSanityCells)}`;
      return `
        <article class="admission-lane-row" data-lane="${escapeHtml(lane.lane)}">
          <div class="admission-lane-label">
            <strong>${escapeHtml(lane.label)}</strong>
            <em>${escapeHtml(laneSummary(lane.lane))}</em>
          </div>
          <div class="admission-lane-cell">
            <span>Cells</span>
            <strong>${escapeHtml(formatEvidenceInteger(lane.cells))}</strong>
          </div>
          <div class="admission-lane-cell">
            <span>Claim</span>
            <strong>${escapeHtml(claimText)}</strong>
          </div>
          <div class="admission-lane-cell">
            <span>Sanity</span>
            <strong>${escapeHtml(sanityText)}</strong>
          </div>
          <div class="admission-lane-cell">
            <span>Regret</span>
            <strong>${escapeHtml(formatEvidenceSigned(lane.meanRegretVsSundog, 6))}</strong>
          </div>
          <div class="admission-lane-cell">
            <span>Axes</span>
            <strong>${escapeHtml(formatAxes(lane.axes))}</strong>
          </div>
        </article>
      `;
    }).join("");
  }

  if (evidenceBoundaries) {
    const reportedCells = (data.cells ?? []).filter((cell) => cell.phase15?.admissionLane === "reported_only");
    const parityLane = lanes.find((lane) => lane.lane === "observation_parity_gate");
    const parityText = parityLane
      ? `${formatEvidenceInteger(parityLane.hardGatePassCells)}/${formatEvidenceInteger(parityLane.hardGateCells)} observation-parity cells remain hard-gated`
      : "observation-parity cells remain separated from standard hard gates";
    evidenceBoundaries.innerHTML = `
      <strong>Admission boundary:</strong> ${escapeHtml(parityText)}. Reported-only cells cover
      ${escapeHtml(reportedBoundaryText(reportedCells))}; they stay in the public bundle without
      expanding the same-information claim beyond its admitted lane.
    `;
  }
}

async function loadBalanceEvidence() {
  if (!evidencePanel) return;
  try {
    const response = await fetch("/data/balance-phase16-claim-lock.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderBalanceEvidence(await response.json());
  } catch (error) {
    evidencePanel.dataset.status = "unavailable";
    if (evidenceStatus) evidenceStatus.textContent = "Unavailable";
    if (evidenceSummary) evidenceSummary.textContent = `Phase 16 receipt data unavailable: ${error.message}`;
    if (evidenceMetrics) evidenceMetrics.innerHTML = "";
    if (admissionLanes) admissionLanes.innerHTML = "";
    if (evidenceBoundaries) evidenceBoundaries.textContent = "Rebuild the public Balance data bundle to restore this panel.";
  }
}

function buildReplayUrl() {
  const cfg = currentConfig();
  const initial = getReplayInitialState();
  const params = new URLSearchParams();
  params.set("mode", currentMode);
  params.set("preset", cfg.preset);
  params.set("seed", String(cfg.seed));
  params.set("light", roundedParam(cfg.lightElevationDeg, 3));
  params.set("force", roundedParam(cfg.forceLimit, 3));
  params.set("rail", roundedParam(cfg.railLimit, 3));
  params.set("noise", roundedParam(cfg.sensorNoiseStd, 6));
  params.set("delay", String(Math.round(cfg.sensorDelaySteps)));
  params.set("dropout", roundedParam(cfg.sensorDropoutRate, 6));
  params.set("duration", roundedParam(cfg.duration, 3));
  params.set("disturbanceForce", roundedParam(currentImpulseForce(), 3));
  params.set("x", roundedParam(initial.x));
  params.set("xDot", roundedParam(initial.xDot));
  params.set("theta", roundedParam(initial.theta));
  params.set("thetaDot", roundedParam(initial.thetaDot));
  return `${window.location.origin}${window.location.pathname}?${params}`;
}

function buildHarnessReplayCommand() {
  return `npm run balance:phase7:replay -- "${buildReplayUrl()}"`;
}

function updateReplayToken() {
  if (replayToken) replayToken.textContent = buildReplayUrl();
}

async function copyReplayUrl() {
  const url = buildReplayUrl();
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(url);
  } else {
    const textarea = document.createElement("textarea");
    textarea.value = url;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
  replayToken.textContent = url;
}

function updateRecoveryTracker({ disturbanceActive, disturbanceForce, cfg }) {
  if (!recoveryTracker.hasImpulse) return;

  const absTheta = Math.abs(state.theta);
  if (disturbanceActive && !recoveryTracker.active) {
    recoveryTracker.active = true;
  }
  if (!disturbanceActive && recoveryTracker.active) {
    recoveryTracker.active = false;
    recoveryTracker.endedAt = state.t;
  }
  if (recoveryTracker.endedAt !== null && !recoveryTracker.terminal) {
    const relativeTime = state.t - recoveryTracker.endedAt;
    if (absTheta > recoveryTracker.peakAbsTheta) {
      recoveryTracker.peakAbsTheta = absTheta;
      recoveryTracker.peakTime = relativeTime;
    }

    recoveryTracker.curve.push({
      relativeTime,
      theta: state.theta,
      shadowResidual: currentSensor.residual,
      force: currentControl.force,
      confidence: currentSensor.confidence,
      disturbanceForce,
    });
    if (recoveryTracker.curve.length > 220) recoveryTracker.curve.shift();

    const recoveredNow = (
      absTheta <= recoveryThresholds.theta
      && Math.abs(state.thetaDot) <= recoveryThresholds.thetaDot
    );
    if (recoveredNow && recoveryTracker.recoveredAt === null) {
      recoveryTracker.recoveryHold += 1;
      const neededHold = Math.max(1, Math.ceil(recoveryThresholds.holdSeconds / cfg.dt));
      if (recoveryTracker.recoveryHold >= neededHold) {
        const firstHeld = recoveryTracker.curve[Math.max(0, recoveryTracker.curve.length - neededHold)];
        recoveryTracker.recoveredAt = firstHeld?.relativeTime ?? relativeTime;
      }
    } else if (!recoveredNow && recoveryTracker.recoveredAt === null) {
      recoveryTracker.recoveryHold = 0;
    }
  }

  if (state.fallen) recoveryTracker.terminal = "fallen";
  else if (state.railHit) recoveryTracker.terminal = "rail hit";
  else if (state.t >= cfg.duration) recoveryTracker.terminal = "timeout";
}

function renderRecoveryCurve() {
  if (!recoveryCurve) return;
  const width = 240;
  const height = 74;
  const pad = 8;
  const usableW = width - pad * 2;
  const usableH = height - pad * 2;
  const samples = recoveryTracker.curve;
  const thresholdY = height - pad - clamp(recoveryThresholds.theta / 0.65, 0, 1) * usableH;

  if (!samples.length) {
    recoveryCurve.innerHTML = `
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="rgba(245,245,245,0.18)" />
      <line x1="${pad}" y1="${thresholdY}" x2="${width - pad}" y2="${thresholdY}" stroke="rgba(244,196,48,0.28)" stroke-dasharray="4 4" />
    `;
    return;
  }

  const maxT = Math.max(1.6, ...samples.map((sample) => sample.relativeTime));
  const points = samples.map((sample) => {
    const x = pad + clamp(sample.relativeTime / maxT, 0, 1) * usableW;
    const y = height - pad - clamp(Math.abs(sample.theta) / 0.65, 0, 1) * usableH;
    return `${roundForSvg(x)},${roundForSvg(y)}`;
  });
  const forcePoints = samples.map((sample) => {
    const x = pad + clamp(sample.relativeTime / maxT, 0, 1) * usableW;
    const y = height - pad - clamp((sample.force / currentConfig().forceLimit + 1) / 2, 0, 1) * usableH;
    return `${roundForSvg(x)},${roundForSvg(y)}`;
  });

  recoveryCurve.innerHTML = `
    <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="rgba(245,245,245,0.18)" />
    <line x1="${pad}" y1="${thresholdY}" x2="${width - pad}" y2="${thresholdY}" stroke="rgba(244,196,48,0.36)" stroke-dasharray="4 4" />
    <polyline points="${forcePoints.join(" ")}" fill="none" stroke="rgba(255,139,106,0.38)" stroke-width="1.4" />
    <polyline points="${points.join(" ")}" fill="none" stroke="#F4C430" stroke-width="2.2" />
  `;
}

function roundForSvg(value) {
  return Number.isFinite(value) ? Number.parseFloat(value.toFixed(2)) : 0;
}

function updateRecoveryPanel() {
  if (!recoverySummary || !recoveryStats) return;
  const tracker = recoveryTracker;
  let summary = "Apply a disturbance to start a recovery trace.";
  if (tracker.hasImpulse) {
    if (tracker.terminal === "fallen" || tracker.terminal === "rail hit") {
      summary = `${tracker.terminal.toUpperCase()} after impulse; this run delayed but did not recover.`;
    } else if (tracker.recoveredAt !== null) {
      summary = `RECOVERED ${formatMetric(tracker.recoveredAt, 2, "s")} after impulse.`;
    } else if (tracker.endedAt === null) {
      summary = "IMPULSE active; recovery clock starts when the push ends.";
    } else {
      summary = "RECOVERING; curve tracks hidden angle after the impulse.";
    }
  }

  recoverySummary.textContent = summary;
  recoveryStats.innerHTML = `
    <span><strong>Peak theta</strong>${formatMetric(tracker.peakAbsTheta, 3)} rad</span>
    <span><strong>Peak time</strong>${formatMetric(tracker.peakTime, 2, "s")}</span>
    <span><strong>Recovery</strong>${formatMetric(tracker.recoveredAt, 2, "s")}</span>
    <span><strong>Samples</strong>${tracker.curve.length}</span>
  `;
  renderRecoveryCurve();
}

function updateBoundaryPanel() {
  if (!boundaryPanel || !boundaryStatus || !boundarySummary || !boundaryList) return;
  const cfg = currentConfig();
  const assessment = assessBalanceBoundary(cfg, currentSensor, currentControl, state);
  boundaryPanel.dataset.status = assessment.status;
  boundaryPanel.dataset.mechanismCount = String(assessment.mechanisms.length);
  boundaryPanel.dataset.cell = [
    `light=${roundedParam(cfg.lightElevationDeg, 3)}`,
    `delay=${Math.round(cfg.sensorDelaySteps)}`,
    `noise=${roundedParam(cfg.sensorNoiseStd, 4)}`,
    `dropout=${roundedParam(cfg.sensorDropoutRate, 4)}`,
    `force=${roundedParam(cfg.forceLimit, 3)}`,
    `rail=${roundedParam(cfg.railLimit, 3)}`,
    `disturbance=${roundedParam(cfg.disturbanceForce, 3)}`,
  ].join(";");
  boundaryStatus.textContent = assessment.label;
  boundarySummary.textContent = assessment.summary;

  const mechanisms = assessment.mechanisms.slice(0, 4);
  if (!mechanisms.length) {
    boundaryList.innerHTML = "<li>No active light, delay, noise, dropout, force, rail, or disturbance warning.</li>";
    return;
  }

  boundaryList.innerHTML = mechanisms.map((mechanism) => {
    const value = mechanism.value === null || mechanism.value === undefined ? "" : ` <span>${mechanism.value}</span>`;
    return `<li data-severity="${mechanism.severity}"><strong>${mechanism.code}</strong>${value}<em>${mechanism.message}</em></li>`;
  }).join("");
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
  // reference. See docs/site/THIRD_PARTY_REUSE.md for the reuse ledger.
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
  updateRecoveryPanel();
  updateBoundaryPanel();
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
  const disturbance = disturbanceTicks > 0 ? -Math.sign(state.theta || 1) * currentImpulseForce() : 0;
  const disturbanceActive = disturbanceTicks > 0;
  if (disturbanceTicks > 0) disturbanceTicks -= 1;
  state = integrateBalanceStep(state, currentControl.force + disturbance, cfg);
  updateRecoveryTracker({ disturbanceActive, disturbanceForce: disturbance, cfg });

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
  beginRecoveryTrace();
});

for (const input of [
  lightElevationInput,
  noiseInput,
  delayInput,
  dropoutInput,
  forceInput,
  railInput,
  disturbanceForceInput,
]) {
  input.addEventListener("input", () => {
    syncControlValue(input);
    updateReplayToken();
  });
}

seedInput.addEventListener("change", () => {
  syncControlValue(seedInput);
  resetSimulation();
});

seedInput.addEventListener("input", () => {
  syncControlValue(seedInput);
  updateReplayToken();
});

presetSelect.addEventListener("change", () => {
  initialStateOverride = null;
  applyPresetControlDefaults();
  resetSimulation();
});

copyReplayButton.addEventListener("click", () => {
  copyReplayUrl().catch((error) => {
    replayToken.textContent = `copy failed: ${error.message}`;
  });
});

refreshReplayButton.addEventListener("click", updateReplayToken);

window.addEventListener("resize", () => {
  resizeCanvas();
  draw();
});

window.__sundogBalanceReplay = () => ({
  url: buildReplayUrl(),
  harnessCommand: buildHarnessReplayCommand(),
  config: currentConfig(),
  initialState: getReplayInitialState(),
  sample: serializeBalanceSample(state, currentSensor, currentControl, currentConfig()),
  recovery: recoveryTracker,
  boundary: assessBalanceBoundary(currentConfig(), currentSensor, currentControl, state),
  evidence: balanceEvidenceData,
});

const replaySettings = readReplaySettings();
if (replaySettings.preset) presetSelect.value = replaySettings.preset;
resizeCanvas();
applyPresetControlDefaults();
applyReplaySettings(replaySettings);
resetSimulation();
loadBalanceEvidence();
requestAnimationFrame(animate);
