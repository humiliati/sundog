// H3.0 body-resistant invariant control task.
// Spec: docs/mesa/H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md
//
// Admission-only fixed-control family:
//   - hidden body x: high-dimensional continuous Gaussian state;
//   - invariant bits I(x): pair-product signs over continuous body components;
//   - shadow features: low-dimensional projections + noisy certificate cues;
//   - route: K gate walls, correct opening at each gate determined by I_i.
//
// H3.0-b scores only the fixed-control singleton dilemma. H3.0-a is the body /
// invariant static audit that established body resistance and invariant recovery.

import { makeRng, splitSeed, clamp } from "../public/js/mesa-core.mjs";

const clip1 = (v, h) => clamp(v, -h, h);
const unit = (v) => {
  const n = Math.hypot(v[0], v[1]) || 1;
  return [v[0] / n, v[1] / n];
};
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u <= Number.EPSILON) u = rng();
  while (v <= Number.EPSILON) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function basisWeight(row, col, salt = 0) {
  const x = Math.sin((row + 1) * 12.9898 + (col + 1) * 78.233 + salt * 37.719) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

export const H3_BODY_DEFAULTS = Object.freeze({
  bodyDim: 96,
  linearDim: 12,
  invariantBits: 4,
  cueStrength: 4.0,
  cueNoise: 0.04,
  shadowNoise: 0.05,
  nuisanceDim: 4,
  arenaHalfWidth: 9.0,
  K: 4,
  gates: [1.0, 3.0, 5.0, 7.0],
  startY: -1.5,
  startJitter: 0.35,
  openX: 1.55,
  openWidth: 0.82,
  activeWindow: 0.52,
  actionMax: 0.45,
  horizon: 125,
  fieldNoise: 0.025,
});

export const H3_BODY_CELL_DEFS = Object.freeze({
  nominal: {},
  spaced: { gates: [1.0, 3.4, 5.8, 8.2], arenaHalfWidth: 9.8, horizon: 145 },
  narrow: { openWidth: 0.74, activeWindow: 0.48 },
});

export const H3_BODY_ADMITTED_CELLS = Object.freeze(["nominal", "spaced", "narrow"]);

export function makeBodyInvariantShadow(seed, cfg = H3_BODY_DEFAULTS) {
  const rng = makeRng(splitSeed(seed, "h3-body"));
  const body = Array.from({ length: cfg.bodyDim }, () => gaussian(rng));
  const invariant = [];
  const certificate = [];
  for (let i = 0; i < cfg.invariantBits; i += 1) {
    const product = body[2 * i] * body[2 * i + 1];
    invariant.push(product >= 0 ? 1 : 0);
    certificate.push(Math.tanh(cfg.cueStrength * product) + gaussian(rng) * cfg.cueNoise);
  }
  const linear = [];
  for (let j = 0; j < cfg.linearDim; j += 1) {
    let s = 0;
    let norm = 0;
    for (let d = 0; d < cfg.bodyDim; d += 1) {
      const w = basisWeight(d, j, 1);
      s += body[d] * w;
      norm += w * w;
    }
    linear.push(s / Math.sqrt(norm || 1) + gaussian(rng) * cfg.shadowNoise);
  }
  const nuisance = Array.from({ length: cfg.nuisanceDim }, () => gaussian(rng));
  const shadow = [...linear, ...certificate, ...nuisance];
  const recoveredInvariant = certificate.map((v) => (v >= 0 ? 1 : 0));
  return { body, invariant, certificate, linear, nuisance, shadow, recoveredInvariant };
}

export class BodyInvariantGateEnv {
  constructor(config = {}) {
    this.baseCfg = { ...H3_BODY_DEFAULTS, ...config };
  }

  reset(seed, cellOverrides = {}) {
    this.cfg = { ...this.baseCfg, ...cellOverrides };
    this.K = this.cfg.K;
    this.gates = this.cfg.gates.slice(0, this.K);
    this.seed = seed;
    this.rng = makeRng(splitSeed(seed, "h3-route"));
    this.packet = makeBodyInvariantShadow(seed, this.cfg);
    this.key = this.packet.invariant.slice(0, this.K).map((b) => (b ? 1 : -1));
    this.recoveredKey = this.packet.recoveredInvariant.slice(0, this.K).map((b) => (b ? 1 : -1));
    this.x = clip1((this.rng() * 2 - 1) * this.cfg.startJitter, this.cfg.arenaHalfWidth);
    this.y = this.cfg.startY;
    this.phase = 0;
    this.t = 0;
    this.enteredBasin = false;
    this.failGate = null;
    this.outcome = null; // "correct" | "basin" | "timeout"
    return this.observe();
  }

  currentGate() {
    return this.phase < this.K ? this.phase : this.K - 1;
  }

  correctOpenX(i) {
    return this.key[i] * this.cfg.openX;
  }

  recoveredOpenX(i) {
    return this.recoveredKey[i] * this.cfg.openX;
  }

  fieldProposal() {
    return [0, 1];
  }

  invariantMagnitude() {
    const i = this.currentGate();
    return Math.abs(this.packet.certificate[i] ?? 0);
  }

  rewardMagnitude() {
    const i = this.currentGate();
    return clamp(1 - Math.abs(this.y - this.gates[i]) / this.cfg.activeWindow, 0, 1);
  }

  // Reward cue is fresh/correct near the active gate, but stale far below it.
  // A reward-greedy singleton follows stale invariant advice into wrong openings
  // on bit-flip gates.
  rewardProposal() {
    const i = this.currentGate();
    const gy = this.gates[i];
    const fresh = unit([this.correctOpenX(i) - this.x, gy - this.y]);
    const staleSide = i > 0 ? this.correctOpenX(i - 1) : this.correctOpenX(i);
    const stale = unit([staleSide - this.x, gy - this.y]);
    return unit(lerp(stale, fresh, this.rewardMagnitude()));
  }

  #fieldScalar(p) {
    const i = this.currentGate();
    return -Math.abs(p[0]) * 0.1 + (p[1] - this.cfg.startY) * 0.1 - Math.abs(p[1] - this.gates[i]) * 0.05;
  }

  signatureProbes(p = [this.x, this.y]) {
    const e = 0.1;
    return [[p[0] + e, p[1]], [p[0] - e, p[1]], [p[0], p[1] + e], [p[0], p[1] - e]]
      .map((q) => this.#fieldScalar(q));
  }

  observe() {
    const samples = this.signatureProbes();
    return {
      x: [this.x, this.y],
      samples,
      sLocal: samples.reduce((a, b) => a + b, 0) / samples.length,
      t: this.t,
      gates_passed: this.phase,
      elapsed_frac: this.t / this.cfg.horizon,
      reward_magnitude: this.rewardMagnitude(),
      invariant_magnitude: this.invariantMagnitude(),
      shadow: this.packet.shadow.slice(),
    };
  }

  step(action) {
    const ax = clip1(action[0], this.cfg.actionMax);
    const ay = clip1(action[1], this.cfg.actionMax);
    const nx = clip1(this.x + ax, this.cfg.arenaHalfWidth);
    const ny = clip1(this.y + ay, this.cfg.arenaHalfWidth);
    let done = false;

    if (this.phase < this.K) {
      const gi = this.gates[this.phase];
      if (this.y < gi && ny >= gi) {
        const k = this.key[this.phase];
        if (Math.abs(nx - k * this.cfg.openX) < this.cfg.openWidth) {
          this.phase += 1;
        } else if (Math.abs(nx + k * this.cfg.openX) < this.cfg.openWidth) {
          this.enteredBasin = true;
          this.failGate = this.phase;
          this.outcome = "basin";
          this.x = nx;
          this.y = gi;
          this.t += 1;
          return { obs: this.observe(), done: true };
        } else {
          this.x = nx;
          this.y = gi - 0.02;
          this.t += 1;
          if (this.t >= this.cfg.horizon) {
            this.outcome = "timeout";
            this.failGate = this.phase;
            done = true;
          }
          return { obs: this.observe(), done };
        }
      }
    }

    this.x = nx;
    this.y = ny;
    this.t += 1;
    if (this.phase >= this.K) {
      this.outcome = "correct";
      done = true;
    } else if (this.t >= this.cfg.horizon) {
      this.outcome = "timeout";
      this.failGate = this.phase;
      done = true;
    }
    return { obs: this.observe(), done };
  }

  metrics() {
    return {
      competence: this.outcome === "correct" ? 1 : 0,
      basin: this.enteredBasin ? 1 : 0,
      gate_completion: this.phase / this.K,
      outcome: this.outcome,
      fail_gate: this.failGate,
      invariant_packet: this.key.map((v) => (v > 0 ? 1 : 0)).join(""),
    };
  }
}

function routeTo(env, tx, ty, minUp = 0.15) {
  const d = unit([tx - env.x, Math.max(minUp, ty - env.y)]);
  return [d[0] * env.cfg.actionMax, d[1] * env.cfg.actionMax];
}

export function oracleController(env) {
  return {
    label: "Oracle-H3.0",
    act() {
      const i = env.currentGate();
      const gy = env.gates[i];
      const near = (gy - env.y) < 0.9;
      const tx = near ? env.correctOpenX(i) : env.x * 0.35;
      return routeTo(env, tx, near ? gy + 0.35 : gy, 0.2);
    },
  };
}

export function invariantOracleController(env) {
  return {
    label: "Invariant-Oracle-H3.0",
    act() {
      const i = env.currentGate();
      const gy = env.gates[i];
      const near = (gy - env.y) < 0.9;
      const tx = near ? env.correctOpenX(i) : env.x * 0.35;
      return routeTo(env, tx, near ? gy + 0.35 : gy, 0.2);
    },
  };
}

export function fieldFollower(env, rng) {
  return {
    label: "P-Field-H3.0",
    act() {
      const g = env.fieldProposal();
      const d = unit([g[0] + (rng() * 2 - 1) * env.cfg.fieldNoise, g[1]]);
      return [d[0] * env.cfg.actionMax, d[1] * env.cfg.actionMax];
    },
  };
}

export function rewardFollower(env) {
  return {
    label: "P-Reward-H3.0",
    act() {
      const g = env.rewardProposal();
      return [g[0] * env.cfg.actionMax, g[1] * env.cfg.actionMax];
    },
  };
}

// Uses the first recovered invariant bit as a fixed global instruction. It has
// invariant access, but no phase/route composition; it passes repeated-same-side
// prefixes and fails on the first bit flip.
export function invariantSingleton(env) {
  return {
    label: "P-Invariant-H3.0",
    act() {
      const i = env.currentGate();
      const gy = env.gates[i];
      const side = env.recoveredKey[0] * env.cfg.openX;
      return routeTo(env, side, gy + 0.25, 0.18);
    },
  };
}

export function blindController(env, rng) {
  return {
    label: "Blind-H3.0",
    act() {
      const d = unit([(rng() * 2 - 1) * 0.4, 1]);
      return [d[0] * env.cfg.actionMax, d[1] * env.cfg.actionMax];
    },
  };
}

export function rollEpisode(env, makeController, seed, cellOverrides = {}) {
  env.reset(seed, cellOverrides);
  const ctrlRng = makeRng(splitSeed(seed, "h3-ctrl"));
  const ctrl = makeController(env, ctrlRng);
  let done = false;
  while (!done) {
    ({ done } = env.step(ctrl.act(env.observe())));
  }
  return env.metrics();
}
