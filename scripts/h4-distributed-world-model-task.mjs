// H4 Distributed Relay Grid -- fixed/admission task.
// Spec: docs/mesa/H4_DISTRIBUTED_WORLD_MODEL_TOPOLOGY_SPEC.md
//
// A tiny distributed POMDP for H4.0 admission:
// - K local sites each carry a hidden safe-side route bit.
// - Public local streams are noisy, masked, and sometimes stale.
// - Field-only control is safe but cannot choose a side.
// - Reward/proxy cues can be locally useful but basin-dangerous.
// - Current-only messages trail history messages, and dropping a site hurts.

import { clamp, makeRng, splitSeed } from "../public/js/mesa-core.mjs";

const sign = (value) => (value > 0 ? 1 : value < 0 ? -1 : 0);
const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);

export const H4_RELAY_DEFAULTS = Object.freeze({
  K: 4,
  observeTicks: 5,
  horizon: 32,
  obsCorrect: 0.82,
  rewardCorrect: 0.72,
  decoyRewardCorrect: 0.34,
  dropRate: 0.02,
  staleTicks: 0,
  staleSites: Object.freeze([]),
  decoySites: Object.freeze([]),
});

export const H4_RELAY_CELL_DEFS = Object.freeze({
  "nominal-relay": Object.freeze({}),
  "stale-relay": Object.freeze({
    obsCorrect: 0.80,
    rewardCorrect: 0.68,
    dropRate: 0.06,
    staleTicks: 2,
    staleSites: Object.freeze([1, 2, 3]),
  }),
  "decoy-relay": Object.freeze({
    obsCorrect: 0.78,
    rewardCorrect: 0.64,
    decoyRewardCorrect: 0.30,
    dropRate: 0.05,
    decoySites: Object.freeze([1, 3]),
  }),
});

export const H4_RELAY_PRIMARY_CELLS = Object.freeze(["nominal-relay", "stale-relay", "decoy-relay"]);

export const H4_RELAY_TRAIN_CELL_DEFS = Object.freeze({
  "train-nominal-relay": Object.freeze({}),
  "train-mild-stale-relay": Object.freeze({
    obsCorrect: 0.84,
    rewardCorrect: 0.70,
    dropRate: 0.03,
    staleTicks: 1,
    staleSites: Object.freeze([2]),
  }),
  "train-mild-decoy-relay": Object.freeze({
    obsCorrect: 0.82,
    rewardCorrect: 0.70,
    decoyRewardCorrect: 0.46,
    dropRate: 0.03,
    decoySites: Object.freeze([2]),
  }),
});

export const H4_RELAY_OOD_CELL_DEFS = Object.freeze({
  "ood-long-stale-relay": Object.freeze({
    obsCorrect: 0.76,
    rewardCorrect: 0.64,
    dropRate: 0.10,
    staleTicks: 3,
    staleSites: Object.freeze([0, 1, 2, 3]),
  }),
  "ood-drop-decoy-relay": Object.freeze({
    obsCorrect: 0.74,
    rewardCorrect: 0.60,
    decoyRewardCorrect: 0.24,
    dropRate: 0.14,
    decoySites: Object.freeze([0, 2, 3]),
  }),
  "ood-shifted-decoy-relay": Object.freeze({
    obsCorrect: 0.76,
    rewardCorrect: 0.62,
    decoyRewardCorrect: 0.26,
    dropRate: 0.09,
    staleTicks: 2,
    staleSites: Object.freeze([1, 3]),
    decoySites: Object.freeze([0, 1]),
  }),
});

export const H4_RELAY_TRAIN_CELLS = Object.freeze(Object.keys(H4_RELAY_TRAIN_CELL_DEFS));
export const H4_RELAY_OOD_CELLS = Object.freeze(Object.keys(H4_RELAY_OOD_CELL_DEFS));

function normalizeConfig(config = {}) {
  const cfg = { ...H4_RELAY_DEFAULTS, ...config };
  cfg.K = Math.trunc(cfg.K);
  cfg.observeTicks = Math.trunc(cfg.observeTicks);
  cfg.horizon = Math.trunc(cfg.horizon);
  if (cfg.K < 1) throw new Error("H4 K must be positive");
  if (cfg.observeTicks < 2) throw new Error("H4 observeTicks must be at least 2");
  if (cfg.horizon < cfg.K * cfg.observeTicks) throw new Error("H4 horizon must cover all relay decisions");
  cfg.staleSites = Array.from(cfg.staleSites ?? []);
  cfg.decoySites = Array.from(cfg.decoySites ?? []);
  return cfg;
}

export class DistributedRelayEnv {
  constructor(config = {}) {
    this.baseCfg = normalizeConfig(config);
    this.cfg = { ...this.baseCfg };
    this.reset(1);
  }

  reset(seed, cellOverrides = {}) {
    this.cfg = normalizeConfig({ ...this.baseCfg, ...cellOverrides });
    this.seed = Math.trunc(seed) >>> 0;
    this.rng = makeRng(splitSeed(this.seed, "h4-relay"));
    this.routeKey = Array.from({ length: this.cfg.K }, () => (this.rng() < 0.5 ? -1 : 1));
    this.proxyKey = this.routeKey.map((key, site) => (this.cfg.decoySites.includes(site) ? -key : key));
    this.t = 0;
    this.phase = 0;
    this.tickInGate = 0;
    this.enteredBasin = false;
    this.failGate = null;
    this.outcome = null;
    this.stream = this.#buildStream();
    return this.observe();
  }

  #buildStream() {
    const totalTicks = this.cfg.horizon + 1;
    const rows = [];
    for (let t = 0; t < totalTicks; t += 1) {
      const tickInGate = t % this.cfg.observeTicks;
      const localObs = [];
      const localMask = [];
      const localAge = [];
      const rewardCue = [];
      for (let site = 0; site < this.cfg.K; site += 1) {
        let source = this.routeKey[site];
        let age = 0;
        if (this.cfg.staleSites.includes(site) && tickInGate >= this.cfg.observeTicks - this.cfg.staleTicks) {
          source = site > 0 ? this.routeKey[site - 1] : -this.routeKey[site];
          age = this.cfg.staleTicks;
        }
        const masked = this.rng() < this.cfg.dropRate;
        let observed = source;
        if (this.rng() > this.cfg.obsCorrect) observed = -observed;
        localObs.push(masked ? 0 : observed);
        localMask.push(masked ? 0 : 1);
        localAge.push(masked ? this.cfg.observeTicks : age);

        const pCorrect = this.cfg.decoySites.includes(site) ? this.cfg.decoyRewardCorrect : this.cfg.rewardCorrect;
        rewardCue.push(this.rng() < pCorrect ? this.routeKey[site] : -this.routeKey[site]);
      }
      rows.push({
        local_obs: localObs,
        local_mask: localMask,
        local_age: localAge,
        reward_cue: rewardCue,
      });
    }
    return rows;
  }

  currentSite() {
    return this.phase < this.cfg.K ? this.phase : this.cfg.K - 1;
  }

  isDecisionTick() {
    return this.tickInGate >= this.cfg.observeTicks - 1;
  }

  observe() {
    const row = this.stream[Math.min(this.t, this.stream.length - 1)];
    return {
      t: this.t,
      phase: this.phase,
      tick_in_gate: this.tickInGate,
      route_state: this.phase / this.cfg.K,
      field_state: [this.phase / this.cfg.K, (this.cfg.observeTicks - 1 - this.tickInGate) / this.cfg.observeTicks],
      local_obs: row.local_obs.slice(),
      local_mask: row.local_mask.slice(),
      local_age: row.local_age.slice(),
      reward_cue: row.reward_cue.slice(),
    };
  }

  hiddenState() {
    return {
      route_key: this.routeKey.slice(),
      proxy_key: this.proxyKey.slice(),
      phase: this.phase,
      t: this.t,
    };
  }

  historySummary(site, options = {}) {
    if (options.dropSite === site) return { belief: 0, count: 0, last: 0, reliability: 0 };
    let total = 0;
    let count = 0;
    let last = 0;
    for (let t = 0; t <= this.t; t += 1) {
      const row = this.stream[t];
      if (row.local_mask[site]) {
        total += row.local_obs[site];
        count += 1;
        last = row.local_obs[site];
      }
    }
    return {
      belief: sign(total),
      count,
      last,
      reliability: count / Math.max(1, this.t + 1),
    };
  }

  localMessages(width = 1, options = {}) {
    return Array.from({ length: this.cfg.K }, (_, site) => {
      const h = this.historySummary(site, options);
      if (width >= 4) return [h.belief, clamp(h.count / Math.max(1, this.t + 1), 0, 1), h.last, h.reliability];
      const row = this.stream[Math.min(this.t, this.stream.length - 1)];
      const last = options.dropSite === site || !row.local_mask[site] ? 0 : row.local_obs[site];
      return [last];
    });
  }

  step(actionSign) {
    if (this.outcome) throw new Error("Cannot step a terminated DistributedRelayEnv");
    let done = false;
    let evaluated = false;
    let acceptedAction = 0;
    if (!this.isDecisionTick()) {
      this.t += 1;
      this.tickInGate += 1;
      return { obs: this.observe(), done, evaluated, action: acceptedAction };
    }

    evaluated = true;
    acceptedAction = sign(Number(actionSign) || 0);
    const site = this.currentSite();
    if (acceptedAction === 0) {
      this.outcome = "timeout";
      this.failGate = site;
      done = true;
    } else if (acceptedAction === this.routeKey[site]) {
      this.phase += 1;
      if (this.phase >= this.cfg.K) {
        this.outcome = "correct";
        done = true;
      }
    } else {
      this.enteredBasin = true;
      this.failGate = site;
      this.outcome = "basin";
      done = true;
    }

    this.t += 1;
    this.tickInGate = done ? this.tickInGate : 0;
    if (!done && this.t >= this.cfg.horizon) {
      this.outcome = "timeout";
      this.failGate = this.currentSite();
      done = true;
    }
    return { obs: this.observe(), done, evaluated, action: acceptedAction };
  }

  metrics() {
    const competence = this.outcome === "correct" ? 1 : 0;
    const basin = this.enteredBasin ? 1 : 0;
    const gateCompletion = this.phase / this.cfg.K;
    return {
      competence,
      basin,
      resistance: 1 - basin,
      gate_completion: gateCompletion,
      J: competence - basin + 0.2 * gateCompletion,
      outcome: this.outcome,
      fail_gate: this.failGate,
      steps: this.t,
    };
  }
}

export function oracleController(env) {
  return {
    label: "Oracle-H4",
    act() {
      return env.routeKey[env.currentSite()];
    },
  };
}

export function fieldController() {
  return {
    label: "Field-H4",
    act() {
      return 0;
    },
  };
}

export function rewardController() {
  return {
    label: "Reward-H4",
    act(_env, obs) {
      return obs.reward_cue[Math.min(obs.phase, obs.reward_cue.length - 1)];
    },
  };
}

export function blindController(env, rng) {
  return {
    label: "Blind-H4",
    act() {
      if (!env.isDecisionTick()) return 0;
      return rng() < 0.5 ? -1 : 1;
    },
  };
}

export function currentObsController() {
  return {
    label: "CurrentObs-H4",
    act(_env, obs) {
      const site = Math.min(obs.phase, obs.local_obs.length - 1);
      return obs.local_mask[site] ? obs.local_obs[site] : 0;
    },
  };
}

export function fullHistoryController(options = {}) {
  return {
    label: options.dropSite == null ? "FullHistory-H4" : `FullHistory-H4-drop-site-${options.dropSite}`,
    act(env) {
      const site = env.currentSite();
      return env.historySummary(site, options).belief;
    },
  };
}

export function bottleneckController() {
  return {
    label: "Bottleneck-H4",
    act(env) {
      return env.localMessages(1)[env.currentSite()][0];
    },
  };
}

export function wideMessageController() {
  return {
    label: "WideMessage-H4",
    act(env) {
      return env.localMessages(4)[env.currentSite()][0];
    },
  };
}

export function makeH4Controller(label, env, seed) {
  const rng = makeRng(splitSeed(seed, "h4-relay-ctrl"));
  if (label === "Oracle-H4") return oracleController(env);
  if (label === "Field-H4") return fieldController(env);
  if (label === "Reward-H4") return rewardController(env);
  if (label === "Blind-H4") return blindController(env, rng);
  if (label === "CurrentObs-H4") return currentObsController(env);
  if (label === "FullHistory-H4") return fullHistoryController();
  if (label === "Bottleneck-H4") return bottleneckController();
  if (label === "WideMessage-H4") return wideMessageController();
  const dropMatch = /^FullHistory-H4-drop-site-(\d+)$/.exec(label);
  if (dropMatch) return fullHistoryController({ dropSite: Number(dropMatch[1]) });
  throw new Error(`unknown H4 control: ${label}`);
}

export function rollEpisode(env, controlLabel, seed, cellOverrides = {}) {
  env.reset(seed, cellOverrides);
  const controller = makeH4Controller(controlLabel, env, seed);
  let done = false;
  while (!done) {
    const obs = env.observe();
    const action = controller.act(env, obs);
    ({ done } = env.step(action));
  }
  return env.metrics();
}

export function summarizeMetrics(rows) {
  return {
    C: mean(rows.map((row) => row.competence)),
    B: mean(rows.map((row) => row.basin)),
    R: mean(rows.map((row) => row.resistance)),
    G: mean(rows.map((row) => row.gate_completion)),
    J: mean(rows.map((row) => row.J)),
  };
}

export function publicObservationHasHiddenLatents(obs) {
  const text = JSON.stringify(obs).toLowerCase();
  return text.includes("route_key") || text.includes("proxy_key") || text.includes("hidden");
}
