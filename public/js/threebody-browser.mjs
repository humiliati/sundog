import {
  computeAcceleration,
  computeControlThrust,
  computeSignatures,
  computeTidalTensor,
  initializeState,
  integrateStep,
  normalizeConfig,
} from "./threebody-core.mjs";

const canvas = document.getElementById("threebody-canvas");
const ctx = canvas.getContext("2d");
const signatureDisplay = document.getElementById("signature-display");

let isPlaying = true;
let timeSpeed = 1;
let dt = 0.01;
let massRatio = 1;
let state = [];
let trails = [[], [], []];
const maxTrailLength = 200;

let showVirial = true;
let showInertia = true;
let showEnergy = true;
let showTidal = true;
let showTrails = true;
let sensorMode = false;
let showThrust = true;

let controllerMode = "off";
let thrustLimit = 0.5;
let targetTidal = 2;
let currentThrust = [0, 0];
const controllerState = { scanPhase: 0 };

function resetControllerState() {
  controllerState.scanPhase = 0;
  controllerState.step = 0;
  delete controllerState.rng;
  delete controllerState.shuffledGradient;
}

function currentConfig(overrides = {}) {
  return normalizeConfig({
    dt,
    massRatio,
    thrustLimit,
    targetTidal,
    controllerMode,
    ...overrides,
  });
}

function currentInitialParticle() {
  return {
    x: parseFloat(document.getElementById("init-x").value),
    y: parseFloat(document.getElementById("init-y").value),
    vx: parseFloat(document.getElementById("init-vx").value),
    vy: parseFloat(document.getElementById("init-vy").value),
  };
}

function resizeCanvas() {
  const heroSection = document.querySelector(".hero");
  canvas.width = Math.min(heroSection.offsetWidth, 1000);
  canvas.height = Math.min(heroSection.offsetHeight, 800);
}

function initializeSystem() {
  state = initializeState(currentConfig({
    initialParticle: currentInitialParticle(),
  }));
  currentThrust = [0, 0];
  resetControllerState();
  trails = [[], [], []];
}

function computeSensorLimitedSignatures(simState) {
  const positions = simState.slice(0, 6);
  const velocities = simState.slice(6, 12);
  const vx3 = velocities[4];
  const vy3 = velocities[5];
  const cfg = currentConfig();
  const [ax, ay] = computeAcceleration(2, positions, cfg);
  const tidal = computeTidalTensor(simState, cfg);

  return {
    virial: null,
    inertia: null,
    energy: null,
    tidalMagnitude: tidal.magnitude,
    localKineticEnergy: 0.5 * cfg.masses[2] * (vx3 * vx3 + vy3 * vy3),
    accelerationMagnitude: Math.sqrt(ax * ax + ay * ay),
    tidal,
  };
}

function worldToScreen(x, y) {
  const scale = Math.min(canvas.width, canvas.height) / 6;
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  return [centerX + x * scale, centerY - y * scale];
}

function drawTrails() {
  if (!showTrails) return;

  const colors = [
    "rgba(244, 196, 48, 0.3)",
    "rgba(125, 162, 184, 0.3)",
    "rgba(255, 255, 255, 0.2)",
  ];

  for (let i = 0; i < 3; i += 1) {
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let j = 0; j < trails[i].length; j += 1) {
      const [sx, sy] = worldToScreen(trails[i][j][0], trails[i][j][1]);
      if (j === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
  }
}

function drawBodies(positions) {
  const bodyColors = ["#F4C430", "#7DA2B8", "#FFFFFF"];
  const bodyRadii = [12, 12, 6];
  for (let i = 0; i < 3; i += 1) {
    const x = positions[i * 2];
    const y = positions[i * 2 + 1];
    const [sx, sy] = worldToScreen(x, y);
    const gradient = ctx.createRadialGradient(sx, sy, 0, sx, sy, bodyRadii[i] * 2);
    gradient.addColorStop(0, `${bodyColors[i]}AA`);
    gradient.addColorStop(0.5, `${bodyColors[i]}44`);
    gradient.addColorStop(1, `${bodyColors[i]}00`);
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(sx, sy, bodyRadii[i] * 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = bodyColors[i];
    ctx.beginPath();
    ctx.arc(sx, sy, bodyRadii[i], 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawThrust(positions) {
  if (!showThrust || controllerMode === "off") return;

  const [sx3, sy3] = worldToScreen(positions[4], positions[5]);
  const thrustScale = 50;
  const tx = currentThrust[0] * thrustScale;
  const ty = -currentThrust[1] * thrustScale;

  ctx.strokeStyle = "rgba(255, 100, 100, 0.8)";
  ctx.fillStyle = "rgba(255, 100, 100, 0.8)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(sx3, sy3);
  ctx.lineTo(sx3 + tx, sy3 + ty);
  ctx.stroke();

  const arrowSize = 8;
  const angle = Math.atan2(ty, tx);
  ctx.beginPath();
  ctx.moveTo(sx3 + tx, sy3 + ty);
  ctx.lineTo(
    sx3 + tx - arrowSize * Math.cos(angle - Math.PI / 6),
    sy3 + ty - arrowSize * Math.sin(angle - Math.PI / 6),
  );
  ctx.lineTo(
    sx3 + tx - arrowSize * Math.cos(angle + Math.PI / 6),
    sy3 + ty - arrowSize * Math.sin(angle + Math.PI / 6),
  );
  ctx.closePath();
  ctx.fill();
}

function updateSignatureDisplay() {
  const displayText = [];

  if (sensorMode) {
    const sensorSignatures = computeSensorLimitedSignatures(state);
    if (showVirial) displayText.push("Virial: N/A (privileged)");
    if (showInertia) displayText.push("Inertia: N/A (privileged)");
    if (showEnergy) displayText.push("Energy: N/A (privileged)");
    if (showTidal) displayText.push(`Tidal: ${sensorSignatures.tidal.magnitude.toFixed(4)}`);
  } else {
    const signatures = computeSignatures(state, currentConfig());
    const tidal = computeTidalTensor(state, currentConfig());
    if (showVirial) displayText.push(`Virial: ${signatures.virial.toFixed(3)}`);
    if (showInertia) displayText.push(`Inertia: ${signatures.inertia.toFixed(3)}`);
    if (showEnergy) displayText.push(`Energy: ${signatures.energy.toFixed(4)}`);
    if (showTidal) displayText.push(`Tidal: ${tidal.magnitude.toFixed(4)}`);
  }

  signatureDisplay.textContent = displayText.join(" | ") || "No signatures selected";
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const positions = state.slice(0, 6);
  drawTrails();
  drawBodies(positions);
  drawThrust(positions);
  updateSignatureDisplay();
}

function update() {
  if (!isPlaying) return;

  currentThrust = computeControlThrust(state, controllerState, currentConfig());
  const substeps = Math.ceil(timeSpeed);
  const substepDt = dt * timeSpeed / substeps;

  for (let i = 0; i < substeps; i += 1) {
    state = integrateStep(state, substepDt, currentConfig({ dt: substepDt }), currentThrust);
  }

  const positions = state.slice(0, 6);
  for (let i = 0; i < 3; i += 1) {
    trails[i].push([positions[i * 2], positions[i * 2 + 1]]);
    if (trails[i].length > maxTrailLength) trails[i].shift();
  }
}

function animate() {
  update();
  draw();
  requestAnimationFrame(animate);
}

document.getElementById("btn-play-pause").addEventListener("click", function onPlayPause() {
  isPlaying = !isPlaying;
  this.textContent = isPlaying ? "Pause" : "Play";
});

document.getElementById("btn-reset").addEventListener("click", initializeSystem);

document.getElementById("mass-ratio").addEventListener("input", function onMassRatio() {
  massRatio = parseFloat(this.value);
  document.getElementById("mass-ratio-value").textContent = this.value;
  initializeSystem();
});

document.getElementById("time-speed").addEventListener("input", function onTimeSpeed() {
  timeSpeed = parseFloat(this.value);
  document.getElementById("time-speed-value").textContent = `${this.value}x`;
});

document.getElementById("show-virial").addEventListener("change", function onVirial() {
  showVirial = this.checked;
});

document.getElementById("show-inertia").addEventListener("change", function onInertia() {
  showInertia = this.checked;
});

document.getElementById("show-energy").addEventListener("change", function onEnergy() {
  showEnergy = this.checked;
});

document.getElementById("show-tidal").addEventListener("change", function onTidal() {
  showTidal = this.checked;
});

document.getElementById("show-trails").addEventListener("change", function onTrails() {
  showTrails = this.checked;
});

document.getElementById("sensor-mode").addEventListener("change", function onSensorMode() {
  sensorMode = this.checked;
});

document.getElementById("controller-mode").addEventListener("change", function onControllerMode() {
  controllerMode = this.value;
  resetControllerState();
});

document.getElementById("thrust-limit").addEventListener("input", function onThrustLimit() {
  thrustLimit = parseFloat(this.value);
  document.getElementById("thrust-limit-value").textContent = this.value;
});

document.getElementById("target-tidal").addEventListener("input", function onTargetTidal() {
  targetTidal = parseFloat(this.value);
  document.getElementById("target-tidal-value").textContent = this.value;
});

document.getElementById("show-thrust").addEventListener("change", function onShowThrust() {
  showThrust = this.checked;
});

["init-x", "init-y", "init-vx", "init-vy"].forEach((id) => {
  document.getElementById(id).addEventListener("input", function onInitialCondition() {
    document.getElementById(`${id}-value`).textContent = this.value;
  });
});

resizeCanvas();
window.addEventListener("resize", resizeCanvas);
initializeSystem();
animate();
