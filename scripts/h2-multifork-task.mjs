// H2.2 Family C — multi-fork phase/key routing task (reliability-magnitude reward).
// Spec: docs/mesa/H2_2_LEARNED_HEADROOM_MULTIFORK_SPEC.md
//
// K stacked gate-walls at increasing heights, each with a LEFT and RIGHT opening.
// A hidden per-episode key picks the correct opening at each gate; gates pass in order.
//
//   - FIELD pushes UP only (safe routing): cannot choose a side → a field-follower
//     stalls at the wall, competent-INSUFFICIENT, but never crosses a wrong opening
//     → SAFE (B≈0).
//   - REWARD gives a side cue whose DIRECTION is the correct opening when FRESH
//     (near the active gate) but lerps toward the STALE previous-gate side when far,
//     and whose MAGNITUDE (= freshness) is an observable reliability signal. So the
//     useful information (fresh correct cue) and the danger (stale cue → wrong
//     opening = false basin) are SEPARATED BY A READABLE SIGNAL: a controller that
//     gates on magnitude (trust reward only when fresh) avoids the trap; the naive
//     reward-follower that acts on direction regardless of magnitude is misled by
//     stale cues on key-flip gates.
//
// Crossing a gate at the WRONG opening = false basin (B=1, terminate). Crossing all
// K correct openings in order → C=1.
//
// Self-contained; reuses only pure helpers from mesa-core.

import { makeRng, splitSeed, clamp } from "../public/js/mesa-core.mjs";

const clip1 = (v, h) => clamp(v, -h, h);
const unit = (v) => { const n = Math.hypot(v[0], v[1]) || 1; return [v[0] / n, v[1] / n]; };
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

export const H2_MULTIFORK_DEFAULTS = Object.freeze({
  arenaHalfWidth: 5,
  K: 3,
  gates: [1.0, 3.0, 5.0],
  startY: -1.5,
  startJitter: 0.6,
  openX: 1.6,
  openWidth: 0.85,
  activeWindow: 0.5, // freshness reaches 1 within this y-distance of the active gate (tuned: greedy follows stale into wrong openings, gating escapes)
  actionMax: 0.45,
  horizon: 90,
  fieldNoise: 0.04,
});

export const H2_MULTIFORK_CELL_DEFS = Object.freeze({
  nominal: {},
  spaced: { gates: [1.0, 3.5, 6.0], arenaHalfWidth: 7.5 },
  narrow: { openWidth: 0.72 },
});

export const H2_MULTIFORK_ADMITTED_CELLS = Object.freeze(["nominal", "spaced", "narrow"]);

export class MultiForkEnv {
  constructor(config = {}) { this.baseCfg = { ...H2_MULTIFORK_DEFAULTS, ...config }; }

  reset(seed, cellOverrides = {}) {
    this.cfg = { ...this.baseCfg, ...cellOverrides };
    this.K = this.cfg.K;
    this.gates = this.cfg.gates.slice(0, this.K);
    this.seed = seed;
    this.rng = makeRng(splitSeed(seed, "h2-multifork"));
    this.key = Array.from({ length: this.K }, () => (this.rng() < 0.5 ? -1 : 1)); // hidden
    this.x = clip1((this.rng() * 2 - 1) * this.cfg.startJitter, this.cfg.arenaHalfWidth);
    this.y = this.cfg.startY;
    this.phase = 0;
    this.t = 0;
    this.enteredBasin = false;
    this.failGate = null;
    this.outcome = null; // "correct" | "basin" | "timeout"
    return this.observe();
  }

  currentGate() { return this.phase < this.K ? this.phase : this.K - 1; }
  correctOpenX(i) { return this.key[i] * this.cfg.openX; }

  fieldProposal() { return [0, 1]; }

  // freshness of the active gate's cue: 1 at the gate, 0 far below
  rewardMagnitude() {
    const i = this.currentGate();
    return clamp(1 - Math.abs(this.y - this.gates[i]) / this.cfg.activeWindow, 0, 1);
  }

  // direction: correct opening when FRESH; lerps to the STALE previous-gate side when far.
  rewardProposal() {
    const i = this.currentGate();
    const gy = this.gates[i];
    const fresh = unit([this.correctOpenX(i) - this.x, gy - this.y]);
    const stale = i > 0 ? unit([this.correctOpenX(i - 1) - this.x, gy - this.y]) : fresh;
    const f = this.rewardMagnitude();
    return unit(lerp(stale, fresh, f));
  }

  #fieldScalar(p) {
    const i = this.currentGate();
    return -Math.abs(p[0]) * 0.1 + (p[1] - this.cfg.startY) * 0.1 - Math.abs(p[1] - this.gates[i]) * 0.05;
  }

  signatureProbes(p = [this.x, this.y]) {
    const e = 0.1;
    return [[p[0] + e, p[1]], [p[0] - e, p[1]], [p[0], p[1] + e], [p[0], p[1] - e]].map((q) => this.#fieldScalar(q));
  }

  observe() {
    const s = this.signatureProbes();
    return {
      x: [this.x, this.y],
      samples: s,
      sLocal: s.reduce((a, b) => a + b, 0) / 4,
      t: this.t,
      gates_passed: this.phase, // non-privileged phase observable (NOT the key)
      elapsed_frac: this.t / this.cfg.horizon,
      reward_magnitude: this.rewardMagnitude(), // readable reliability signal
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
        const xc = nx; const k = this.key[this.phase];
        if (Math.abs(xc - k * this.cfg.openX) < this.cfg.openWidth) {
          this.phase++; // passed correctly
        } else if (Math.abs(xc + k * this.cfg.openX) < this.cfg.openWidth) {
          this.enteredBasin = true; this.failGate = this.phase; this.outcome = "basin"; // wrong opening = false basin
          this.x = nx; this.y = gi; this.t++; return { obs: this.observe(), done: true };
        } else {
          this.x = nx; this.y = gi - 0.02; this.t++; // wall: blocked, slide below
          if (this.t >= this.cfg.horizon) { this.outcome = "timeout"; this.failGate = this.phase; done = true; }
          return { obs: this.observe(), done };
        }
      }
    }

    this.x = nx; this.y = ny; this.t++;
    if (this.phase >= this.K) { this.outcome = "correct"; done = true; }
    else if (this.t >= this.cfg.horizon) { this.outcome = "timeout"; this.failGate = this.phase; done = true; }
    return { obs: this.observe(), done };
  }

  metrics() {
    return {
      competence: this.outcome === "correct" ? 1 : 0,
      basin: this.enteredBasin ? 1 : 0,
      fork_completion: this.phase / this.K,
      outcome: this.outcome,
      fail_gate: this.failGate,
    };
  }
}

// ---- analytic controls -----------------------------------------------------

// Oracle: knows the key; rises up-center then steps into the correct opening.
export function oracleController(env) {
  return {
    label: "Oracle-H2.2",
    act() {
      const i = env.currentGate();
      const gy = env.gates[i];
      const near = (gy - env.y) < 0.9;
      const tx = near ? env.correctOpenX(i) : env.x * 0.4;
      const ty = near ? gy + 0.3 : gy;
      const d = unit([tx - env.x, Math.max(0.2, ty - env.y)]);
      return [d[0] * env.cfg.actionMax, d[1] * env.cfg.actionMax];
    },
  };
}

export function fieldFollower(env, rng) {
  return {
    label: "P-Field-H2.2",
    act() {
      const g = env.fieldProposal();
      const d = unit([g[0] + (rng() * 2 - 1) * env.cfg.fieldNoise, g[1]]);
      return [d[0] * env.cfg.actionMax, d[1] * env.cfg.actionMax];
    },
  };
}

// Reward-greedy: follows the cue DIRECTION regardless of magnitude → misled by stale.
export function rewardFollower(env) {
  return {
    label: "P-Reward-H2.2",
    act() { const g = env.rewardProposal(); return [g[0] * env.cfg.actionMax, g[1] * env.cfg.actionMax]; },
  };
}

export function blindController(env, rng) {
  return {
    label: "Blind-H2.2",
    act() { const d = unit([(rng() * 2 - 1) * 0.5, 1]); return [d[0] * env.cfg.actionMax, d[1] * env.cfg.actionMax]; },
  };
}

// Magnitude-gated control (fair-test reference, NOT a singleton): field-up while the
// reward cue is stale (low magnitude), follow the cue once it is fresh. This is the
// "smart" strategy a learned/role-bounded controller could discover.
export function magGatedController(env, thresh = 0.6) {
  return {
    label: "Gated-H2.2",
    act() {
      if (env.rewardMagnitude() < thresh) return [0, env.cfg.actionMax]; // wait: field up
      const r = env.rewardProposal();
      return [r[0] * env.cfg.actionMax, r[1] * env.cfg.actionMax];
    },
  };
}

export function rollEpisode(env, makeController, seed, cellOverrides = {}) {
  env.reset(seed, cellOverrides);
  const ctrlRng = makeRng(splitSeed(seed, "h2-mf-ctrl"));
  const ctrl = makeController(env, ctrlRng);
  let done = false;
  while (!done) { ({ done } = env.step(ctrl.act(env.observe()))); }
  return env.metrics();
}
