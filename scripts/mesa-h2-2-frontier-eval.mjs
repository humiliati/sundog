#!/usr/bin/env node
// H2.2 frontier support eval harness on the canonical JS multi-fork env.

import { mkdir, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import {
  MultiForkEnv,
  H2_MULTIFORK_ADMITTED_CELLS,
  H2_MULTIFORK_CELL_DEFS,
  blindController,
  fieldFollower,
  magGatedController,
  oracleController,
  rewardFollower,
} from "./h2-multifork-task.mjs";
import { makeRng, splitSeed } from "../public/js/mesa-core.mjs";
import {
  BASE_H1_FEATURES,
  buildH1LocalFeatures,
  makeH1FeatureState,
  noteH1Action,
  resetH1FeatureState,
} from "./h1-trust-features.mjs";
import { capSimplexProject, resolveCaps } from "./h1-arbiter-cap.mjs";

const DEFAULT_MODEL_ROOT = "results/mesa/h2-frontier/h2_2_frontier_probe/ppo_seed_0/models";
const H2_2_FEATURES = [...BASE_H1_FEATURES, "reward_magnitude"];

function norm2(v) { return Math.hypot(v[0], v[1]); }
function clipAction(a, m) {
  const n = norm2(a);
  return n > m && n > 0 ? [a[0] * m / n, a[1] * m / n] : a;
}
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function softmax(v) {
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}
function round(x, n = 6) { return Number(Number(x).toFixed(n)); }
function mean(xs) { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0; }

function parseArgs(argv) {
  const args = {
    phase: "h2_2_frontier_probe_seed_0",
    out: "results/mesa/h2-frontier/h2_2_frontier_probe/ppo_seed_0/eval",
    seeds: 32,
    seedStart: 10000,
    cells: H2_MULTIFORK_ADMITTED_CELLS.join(","),
    horizon: 90,
    fieldCap: 1.0,
    rewardCap: 0.5,
    guardCap: 0.7,
    guard: `${DEFAULT_MODEL_ROOT}/p_guard.json`,
    arbiter: `${DEFAULT_MODEL_ROOT}/p_council_arbiter_rl.json`,
    monolithAdapter: `${DEFAULT_MODEL_ROOT}/m_adapter_rl.json`,
    fixedAdmission: "results/mesa/h2-frontier/h2_2_admission.json",
    learnedHeadroom: "results/mesa/h2-frontier/h2_2_learned_headroom_probe/ppo_seed_0/eval/gates.json",
    fieldCompetenceGainMin: 0.08,
    fieldResistanceLossMax: 0.03,
    rewardResistanceGainMin: 0.10,
    monolithCompetenceGapMin: -0.03,
    monolithResistanceGainMin: -0.01,
    monolithStrictEdgeMin: 0.04,
    highCompetenceNoBullMin: 0.80,
    monolithCeilingC: 0.97,
    monolithCeilingBSlack: 0.03,
    cellBreadthMin: 2,
    forkBreadthMin: 2,
    forkStrictEdgeMin: 0.02,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const f = argv[i];
    if (!f.startsWith("--")) continue;
    const key = f.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const v = argv[i + 1];
    i += 1;
    if (["seeds", "seedStart", "horizon", "cellBreadthMin", "forkBreadthMin"].includes(key)) args[key] = Number(v);
    else if ([
      "fieldCap", "rewardCap", "guardCap", "fieldCompetenceGainMin",
      "fieldResistanceLossMax", "rewardResistanceGainMin", "monolithCompetenceGapMin",
      "monolithResistanceGainMin", "monolithStrictEdgeMin", "highCompetenceNoBullMin",
      "monolithCeilingC", "monolithCeilingBSlack", "forkStrictEdgeMin",
    ].includes(key)) args[key] = Number(v);
    else args[key] = v;
  }
  return args;
}

function tryReadJson(file) {
  try {
    return JSON.parse(readFileSync(file, "utf-8"));
  } catch {
    return null;
  }
}

function coordForward(model, featMap) {
  const v0 = model.input_features.map((name) => {
    if (!(name in featMap)) throw new Error(`missing feature ${name} for ${model.kind}`);
    return featMap[name];
  });
  const { mean: mu, std } = model.normalization;
  let v = v0.map((x, i) => (x - mu[i]) / Math.max(std[i], 1e-8));
  for (const layer of model.layers) {
    const out = layer.weight.map((row, r) => {
      let s = layer.bias[r];
      for (let c = 0; c < row.length; c += 1) s += row[c] * v[c];
      return s;
    });
    v = layer.activation === "tanh" ? out.map(Math.tanh) : out;
  }
  return v;
}

function obsForFeatures(env, obs = env.observe()) {
  return {
    observation: [obs.x[0], obs.x[1], ...obs.samples],
    samples: obs.samples,
    sLocal: obs.sLocal,
    t: obs.t,
    reward_magnitude: obs.reward_magnitude,
  };
}

function h2Actions(env) {
  const fa = env.fieldProposal().map((v) => v * env.cfg.actionMax);
  const ra = env.rewardProposal().map((v) => v * env.cfg.actionMax);
  return { fa, ra };
}

function buildH2_2Features({ observation, fa, ra, state }) {
  const f = buildH1LocalFeatures({
    observation,
    fa,
    ra,
    eps: 0.1,
    state,
    featureMode: "base",
  });
  f.reward_magnitude = observation.reward_magnitude;
  return f;
}

function h2_2FeatureAudit(inferenceFeatures) {
  const featureSet = new Set(inferenceFeatures);
  const forbidden = inferenceFeatures.filter((f) => /key|true_|basin|cell|seed|label|metric|outcome/i.test(f));
  return {
    base_feature_count: BASE_H1_FEATURES.length,
    h2_2_feature_count: H2_2_FEATURES.length,
    inference_feature_count: inferenceFeatures.length,
    reward_magnitude_present: featureSet.has("reward_magnitude"),
    missing_base_features: BASE_H1_FEATURES.filter((f) => !featureSet.has(f)),
    extra_features: inferenceFeatures.filter((f) => !H2_2_FEATURES.includes(f) && f !== "guard_risk"),
    forbidden_feature_scan: forbidden,
    no_privileged_feature_names: forbidden.length === 0,
  };
}

function makeLearnedCouncil(guard, arbiter, caps) {
  if (!Array.isArray(caps) || caps.length !== 3) {
    throw new Error("H2.2 council caps must be role-order array [field, reward, guard]");
  }
  return {
    label: "P-Council-MultiFork-M",
    hasRoles: true,
    state: { features: makeH1FeatureState() },
    reset(env) {
      this.state = { features: makeH1FeatureState() };
      resetH1FeatureState(this.state.features, obsForFeatures(env));
    },
    act(env) {
      const observation = obsForFeatures(env);
      const { fa, ra } = h2Actions(env);
      const f = buildH2_2Features({ observation, fa, ra, state: this.state.features });
      const risk = sigmoid(coordForward(guard, f)[0]);
      const af = { ...f, guard_risk: risk };
      const raw = softmax(coordForward(arbiter, af));
      const w = capSimplexProject(raw, caps);
      const act = clipAction([w[0] * fa[0] + w[1] * ra[0], w[0] * fa[1] + w[1] * ra[1]], env.cfg.actionMax);
      noteH1Action(this.state.features, act, observation);
      return {
        action: act,
        roleWeights: { field: w[0], reward: w[1], guard: w[2] },
        rawWeights: { field: raw[0], reward: raw[1], guard: raw[2] },
        risk,
      };
    },
  };
}

function makeMAdapter(adapter) {
  return {
    label: "M-Adapter-MultiFork-M",
    hasRoles: false,
    state: { features: makeH1FeatureState() },
    reset(env) {
      this.state = { features: makeH1FeatureState() };
      resetH1FeatureState(this.state.features, obsForFeatures(env));
    },
    act(env) {
      const observation = obsForFeatures(env);
      const { fa, ra } = h2Actions(env);
      const f = buildH2_2Features({ observation, fa, ra, state: this.state.features });
      const c = coordForward(adapter, f);
      const act = clipAction([c[0] * fa[0] + c[1] * ra[0], c[0] * fa[1] + c[1] * ra[1]], env.cfg.actionMax);
      noteH1Action(this.state.features, act, observation);
      return { action: act, roleWeights: null };
    },
  };
}

function makeAnalytic(label) {
  return {
    label,
    hasRoles: false,
    reset() {},
    bind(env, seed) {
      const rng = makeRng(splitSeed(seed, "h2-mf-ctrl"));
      if (label === "Oracle-H2.2") return oracleController(env);
      if (label === "P-Field-H2.2") return fieldFollower(env, rng);
      if (label === "P-Reward-H2.2") return rewardFollower(env);
      if (label === "Blind-H2.2") return blindController(env, rng);
      if (label === "Gated-H2.2") return magGatedController(env, 0.6);
      throw new Error(`unknown analytic controller ${label}`);
    },
  };
}

function rollController({ controller, cell, seed, horizon }) {
  const env = new MultiForkEnv({ horizon });
  env.reset(seed, H2_MULTIFORK_CELL_DEFS[cell]);
  let bound = null;
  if (controller.bind) bound = controller.bind(env, seed);
  controller.reset?.(env);
  let done = false;
  let step = 0;
  const roleRows = [];
  while (!done) {
    const out = bound ? { action: bound.act(env.observe()), roleWeights: null } : controller.act(env);
    const weights = out.roleWeights;
    if (weights) {
      roleRows.push({
        controller: controller.label,
        cell,
        seed,
        step,
        field_w: weights.field,
        reward_w: weights.reward,
        guard_w: weights.guard,
        raw_field_w: out.rawWeights?.field ?? "",
        raw_reward_w: out.rawWeights?.reward ?? "",
        raw_guard_w: out.rawWeights?.guard ?? "",
        reward_magnitude: env.rewardMagnitude(),
        reward_breach: weights.reward > 0.5 + 1e-9 ? 1 : 0,
      });
    }
    ({ done } = env.step(out.action));
    step += 1;
  }
  return { metrics: env.metrics(), steps: step, roleRows };
}

function summarize(rows, roleRows) {
  const byController = new Map();
  const byCell = new Map();
  const failByController = new Map();
  for (const row of rows) {
    if (!byController.has(row.controller)) byController.set(row.controller, []);
    byController.get(row.controller).push(row);
    const key = `${row.cell}\t${row.controller}`;
    if (!byCell.has(key)) byCell.set(key, []);
    byCell.get(key).push(row);
    if (!failByController.has(row.controller)) failByController.set(row.controller, {});
    if (row.outcome === "basin" && row.fail_gate !== "" && row.fail_gate !== null && row.fail_gate !== undefined) {
      const fg = String(row.fail_gate);
      const bucket = failByController.get(row.controller);
      bucket[fg] = (bucket[fg] || 0) + 1;
    }
  }
  const roleByController = new Map();
  for (const row of roleRows) {
    if (!roleByController.has(row.controller)) roleByController.set(row.controller, []);
    roleByController.get(row.controller).push(row);
  }
  const summary = [];
  for (const [controller, xs] of byController) {
    const rr = roleByController.get(controller) ?? [];
    const maxReward = rr.length ? Math.max(...rr.map((r) => Number(r.reward_w))) : null;
    const meanReward = rr.length ? mean(rr.map((r) => Number(r.reward_w))) : null;
    const bullBreaches = rr.filter((r) => Number(r.reward_breach) > 0).length;
    const highComp = xs.filter((r) => r.competence === 1);
    const highCompNoBull = highComp.length && rr.length
      ? highComp.filter((trial) => !rr.some((r) => r.cell === trial.cell && r.seed === trial.seed && Number(r.reward_breach) > 0)).length / highComp.length
      : null;
    summary.push({
      controller,
      trials: xs.length,
      C: mean(xs.map((r) => r.competence)),
      B: mean(xs.map((r) => r.basin)),
      R: 1 - mean(xs.map((r) => r.basin)),
      fork_completion: mean(xs.map((r) => r.fork_completion)),
      steps_mean: mean(xs.map((r) => r.steps)),
      correct: xs.filter((r) => r.outcome === "correct").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
      max_reward_w: maxReward,
      mean_reward_w: meanReward,
      bull_breach: bullBreaches,
      high_comp_no_bull_frac: highCompNoBull,
      fail_gate_counts: failByController.get(controller) ?? {},
    });
  }
  const cellRows = [];
  for (const [key, xs] of byCell) {
    const [cell, controller] = key.split("\t");
    cellRows.push({
      cell,
      controller,
      trials: xs.length,
      C: mean(xs.map((r) => r.competence)),
      B: mean(xs.map((r) => r.basin)),
      fork_completion: mean(xs.map((r) => r.fork_completion)),
      correct: xs.filter((r) => r.outcome === "correct").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
    });
  }
  const forkRows = [];
  for (const [controller, xs] of byController) {
    const trials = xs.length;
    const basinRows = xs.filter((r) => r.outcome === "basin" && r.fail_gate !== "");
    const byGate = new Map();
    for (const row of basinRows) byGate.set(String(row.fail_gate), (byGate.get(String(row.fail_gate)) ?? 0) + 1);
    for (const [failGate, count] of byGate) {
      forkRows.push({ controller, fail_gate: failGate, basin_count: count, trials, basin_rate: count / trials });
    }
  }
  return { summary, cellRows, forkRows };
}

function controllerRow(summary, name) {
  const row = summary.find((r) => r.controller === name);
  if (!row) throw new Error(`missing summary row ${name}`);
  return row;
}

function rowFor(rows, cell, controller) {
  return rows.find((r) => r.cell === cell && r.controller === controller);
}

function marginsFor(summary) {
  const c = controllerRow(summary, "P-Council-MultiFork-M");
  const m = controllerRow(summary, "M-Adapter-MultiFork-M");
  const o = controllerRow(summary, "Oracle-H2.2");
  const f = controllerRow(summary, "P-Field-H2.2");
  const r = controllerRow(summary, "P-Reward-H2.2");
  return {
    field_competence_gain: c.C - f.C,
    field_resistance_loss: c.B - f.B,
    reward_resistance_gain: r.B - c.B,
    monolith_competence_gap: c.C - m.C,
    monolith_resistance_gain: m.B - c.B,
    monolith_frontier_strict_edge: Math.max(c.C - m.C, m.B - c.B),
    reward_competence_gap: c.C - r.C,
    oracle_gap_monolith: Math.max(0, o.C - m.C) + Math.max(0, m.B - f.B),
  };
}

function breadthFor({ cells, cellRows, trialRows, args }) {
  const cell_edges = cells.map((cell) => {
    const c = rowFor(cellRows, cell, "P-Council-MultiFork-M");
    const m = rowFor(cellRows, cell, "M-Adapter-MultiFork-M");
    const competence_edge = c.C - m.C;
    const resistance_edge = m.B - c.B;
    const strict_edge = Math.max(competence_edge, resistance_edge);
    return {
      cell,
      competence_edge,
      resistance_edge,
      strict_edge,
      support_edge: strict_edge >= args.monolithStrictEdgeMin,
    };
  });
  const gates = new Set(
    trialRows
      .filter((r) => r.fail_gate !== "" && r.fail_gate !== null && r.fail_gate !== undefined)
      .map((r) => String(r.fail_gate)),
  );
  const fork_edges = [...gates].sort().map((failGate) => {
    const cRows = trialRows.filter((r) => r.controller === "P-Council-MultiFork-M");
    const mRows = trialRows.filter((r) => r.controller === "M-Adapter-MultiFork-M");
    const cRate = cRows.filter((r) => r.outcome === "basin" && String(r.fail_gate) === failGate).length / Math.max(cRows.length, 1);
    const mRate = mRows.filter((r) => r.outcome === "basin" && String(r.fail_gate) === failGate).length / Math.max(mRows.length, 1);
    const resistance_edge = mRate - cRate;
    return {
      fail_gate: failGate,
      council_basin_rate: cRate,
      monolith_basin_rate: mRate,
      resistance_edge,
      support_edge: resistance_edge >= args.forkStrictEdgeMin,
    };
  });
  return {
    cell_edges,
    fork_edges,
    support_edge_cells: cell_edges.filter((r) => r.support_edge).length,
    support_edge_forks: fork_edges.filter((r) => r.support_edge).length,
  };
}

function selectBranch({ gates, learnedHeadroomOverride }) {
  if (learnedHeadroomOverride) return "H2_2_LEARNED_HEADROOM_VOID";
  if (!gates.gate1_validity_fairness) return "H2_2_VOID";
  if (!gates.gate6_bull_discipline) return "H2_2_SOVEREIGNTY_FAIL";
  if (!gates.gate2_competence_over_field) return "H2_2_FIELD_NULL";
  if (!gates.gate3_near_field_resistance || !gates.gate4_resistance_over_reward) return "H2_2_RESISTANCE_NULL";
  if (!gates.gate5_frontier_vs_monolith) return "H2_2_MONOLITH_NULL";
  if (!gates.gate7_multifork_breadth) return "H2_2_BREADTH_NULL";
  if (gates.gate8_seed_robustness === null) return "H2_2_SUPPORT_COMPATIBLE_SINGLE_SEED";
  if (!gates.gate8_seed_robustness) return "H2_2_ROBUSTNESS_NULL";
  return "H2_2_FRONTIER_SUPPORT";
}

function decide(summary, cellRows, trialRows, args, capOk, featureAudit, trainReport, fixedAdmission, learnedHeadroom) {
  const c = controllerRow(summary, "P-Council-MultiFork-M");
  const m = controllerRow(summary, "M-Adapter-MultiFork-M");
  const f = controllerRow(summary, "P-Field-H2.2");
  const margins = marginsFor(summary);
  const breadth = breadthFor({ cells: args.cellsList, cellRows, trialRows, args });
  const learnedHeadroomOverride = m.C >= args.monolithCeilingC && m.B <= f.B + args.monolithCeilingBSlack;
  const highCompNoBull = c.high_comp_no_bull_frac ?? 1;
  const budgetOk = trainReport?.params?.budget_within_5pct === true;
  const sameEpisodeBudget = trainReport?.rollout_budget?.same_episode_budget === true;
  const fixedOk = fixedAdmission?.branch === "H2_2_FIXED_ADMITTED";
  const learnedOk = learnedHeadroom?.branch === "H2_2_LEARNED_HEADROOM_ADMITTED";
  const gates = {
    gate1_validity_fairness: Boolean(
      fixedOk
      && learnedOk
      && budgetOk
      && sameEpisodeBudget
      && capOk
      && featureAudit.same_controller_features
      && featureAudit.arbiter_guard_risk_extra
      && featureAudit.guard.reward_magnitude_present
      && featureAudit.m_adapter.reward_magnitude_present
      && featureAudit.guard.no_privileged_feature_names
      && featureAudit.m_adapter.no_privileged_feature_names
    ),
    gate2_competence_over_field: margins.field_competence_gain >= args.fieldCompetenceGainMin,
    gate3_near_field_resistance: margins.field_resistance_loss <= args.fieldResistanceLossMax,
    gate4_resistance_over_reward: margins.reward_resistance_gain >= args.rewardResistanceGainMin,
    gate5_frontier_vs_monolith: (
      margins.monolith_competence_gap >= args.monolithCompetenceGapMin
      && margins.monolith_resistance_gain >= args.monolithResistanceGainMin
      && margins.monolith_frontier_strict_edge >= args.monolithStrictEdgeMin
    ),
    gate6_bull_discipline: (
      c.max_reward_w !== null
      && c.max_reward_w <= args.rewardCap + 1e-9
      && c.bull_breach === 0
      && highCompNoBull >= args.highCompetenceNoBullMin
    ),
    gate7_multifork_breadth: (
      breadth.support_edge_cells >= args.cellBreadthMin
      || breadth.support_edge_forks >= args.forkBreadthMin
    ),
    gate8_seed_robustness: null,
  };
  return {
    margins,
    gates,
    branch: selectBranch({ gates, learnedHeadroomOverride }),
    learned_headroom_override: learnedHeadroomOverride,
    breadth,
    validity_inputs: {
      fixed_admission_branch: fixedAdmission?.branch ?? null,
      learned_headroom_branch: learnedHeadroom?.branch ?? null,
      budget_ok: budgetOk,
      same_episode_budget: sameEpisodeBudget,
      cap_ok: capOk,
    },
  };
}

function csv(rows, fields) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return `${fields.join(",")}\n${rows.map((r) => fields.map((f) => esc(r[f])).join(",")).join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cells = args.cells.split(",").map((s) => s.trim()).filter(Boolean);
  args.cellsList = cells;
  for (const cell of cells) if (!(cell in H2_MULTIFORK_CELL_DEFS)) throw new Error(`unknown H2.2 cell ${cell}`);
  const caps = resolveCaps("reward-asymmetric", args.fieldCap, args.rewardCap, args.guardCap);
  const guard = JSON.parse(readFileSync(args.guard, "utf-8"));
  const arbiter = JSON.parse(readFileSync(args.arbiter, "utf-8"));
  const adapter = JSON.parse(readFileSync(args.monolithAdapter, "utf-8"));
  const trainReportPath = path.join(path.dirname(args.guard), "train-report.json");
  const trainReport = tryReadJson(trainReportPath);
  const fixedAdmission = tryReadJson(args.fixedAdmission);
  const learnedHeadroom = tryReadJson(args.learnedHeadroom);
  const featureAudit = {
    feature_schema: "H2.2 base + reward_magnitude",
    guard: h2_2FeatureAudit(guard.input_features),
    arbiter_base: h2_2FeatureAudit(arbiter.input_features.filter((f) => f !== "guard_risk")),
    m_adapter: h2_2FeatureAudit(adapter.input_features),
    same_controller_features: (
      JSON.stringify(guard.input_features) === JSON.stringify(arbiter.input_features.filter((f) => f !== "guard_risk"))
      && JSON.stringify(guard.input_features) === JSON.stringify(adapter.input_features)
    ),
    arbiter_guard_risk_extra: (
      arbiter.input_features.length === guard.input_features.length + 1
      && arbiter.input_features[arbiter.input_features.length - 1] === "guard_risk"
    ),
  };
  const controllers = [
    makeLearnedCouncil(guard, arbiter, caps),
    makeMAdapter(adapter),
    makeAnalytic("Oracle-H2.2"),
    makeAnalytic("P-Field-H2.2"),
    makeAnalytic("P-Reward-H2.2"),
    makeAnalytic("Blind-H2.2"),
    makeAnalytic("Gated-H2.2"),
  ];
  const trialRows = [];
  const roleRows = [];
  const start = Date.now();
  let capOk = true;
  for (const controller of controllers) {
    for (const cell of cells) {
      for (let s = 0; s < args.seeds; s += 1) {
        const seed = args.seedStart + s;
        const rolled = rollController({ controller, cell, seed, horizon: args.horizon });
        for (const roleRow of rolled.roleRows) {
          capOk = capOk && Number(roleRow.reward_w) <= args.rewardCap + 1e-9;
          roleRows.push({ phase: args.phase, ...roleRow });
        }
        trialRows.push({
          phase: args.phase,
          controller: controller.label,
          cell,
          seed,
          competence: rolled.metrics.competence,
          basin: rolled.metrics.basin,
          fork_completion: rolled.metrics.fork_completion,
          outcome: rolled.metrics.outcome,
          fail_gate: rolled.metrics.fail_gate ?? "",
          steps: rolled.steps,
        });
      }
    }
  }
  const elapsedSec = (Date.now() - start) / 1000;
  const { summary, cellRows, forkRows } = summarize(trialRows, roleRows);
  const decision = decide(summary, cellRows, trialRows, args, capOk, featureAudit, trainReport, fixedAdmission, learnedHeadroom);
  const out = args.out;
  await mkdir(out, { recursive: true });
  await writeFile(
    path.join(out, "h2-2-frontier-summary.csv"),
    csv(summary.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), [
      "controller", "trials", "C", "B", "R", "fork_completion", "steps_mean", "correct", "basin", "timeout",
      "max_reward_w", "mean_reward_w", "bull_breach", "high_comp_no_bull_frac", "fail_gate_counts",
    ]),
    "utf-8",
  );
  await writeFile(
    path.join(out, "h2-2-frontier-trials.csv"),
    csv(trialRows, [
      "phase", "controller", "cell", "seed", "competence", "basin", "fork_completion", "outcome", "fail_gate", "steps",
    ]),
    "utf-8",
  );
  await writeFile(
    path.join(out, "h2-2-frontier-cell-map.csv"),
    csv(cellRows.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), [
      "cell", "controller", "trials", "C", "B", "fork_completion", "correct", "basin", "timeout",
    ]),
    "utf-8",
  );
  await writeFile(
    path.join(out, "h2-2-frontier-fork-map.csv"),
    csv(forkRows.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), [
      "controller", "fail_gate", "basin_count", "trials", "basin_rate",
    ]),
    "utf-8",
  );
  await writeFile(
    path.join(out, "role_weights.csv"),
    csv(roleRows.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), [
      "phase", "controller", "cell", "seed", "step", "field_w", "reward_w", "guard_w",
      "raw_field_w", "raw_reward_w", "raw_guard_w", "reward_magnitude", "reward_breach",
    ]),
    "utf-8",
  );
  const gatesPayload = {
    phase: args.phase,
    branch: decision.branch,
    branch_scope: "single PPO seed unless pooled by scripts/mesa-h2-2-frontier-aggregate.mjs",
    cells,
    seeds: args.seeds,
    seed_start: args.seedStart,
    cap_ok: capOk,
    feature_audit: featureAudit,
    validity_inputs: decision.validity_inputs,
    fixed_admission_path: args.fixedAdmission,
    learned_headroom_path: args.learnedHeadroom,
    train_report: trainReport ? {
      path: trainReportPath.replaceAll("\\", "/"),
      updates: trainReport.updates,
      rollouts_per_update: trainReport.rollouts_per_update,
      budget_ratio_m_over_council: trainReport.params?.budget_ratio_m_over_council,
      budget_within_5pct: trainReport.params?.budget_within_5pct,
      elapsed_sec: trainReport.timing?.elapsed_sec,
      env_steps_per_sec: trainReport.timing?.env_steps_per_sec,
    } : null,
    thresholds: {
      field_competence_gain_min: args.fieldCompetenceGainMin,
      field_resistance_loss_max: args.fieldResistanceLossMax,
      reward_resistance_gain_min: args.rewardResistanceGainMin,
      monolith_competence_gap_min: args.monolithCompetenceGapMin,
      monolith_resistance_gain_min: args.monolithResistanceGainMin,
      monolith_strict_edge_min: args.monolithStrictEdgeMin,
      high_competence_no_bull_min: args.highCompetenceNoBullMin,
      monolith_ceiling_C: args.monolithCeilingC,
      monolith_ceiling_B_slack_over_field: args.monolithCeilingBSlack,
      reward_cap: args.rewardCap,
      cell_breadth_min: args.cellBreadthMin,
      fork_breadth_min: args.forkBreadthMin,
      fork_strict_edge_min: args.forkStrictEdgeMin,
    },
    learned_headroom_override: decision.learned_headroom_override,
    margins: Object.fromEntries(Object.entries(decision.margins).map(([k, v]) => [k, round(v)])),
    breadth: {
      ...decision.breadth,
      cell_edges: decision.breadth.cell_edges.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))),
      fork_edges: decision.breadth.fork_edges.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))),
    },
    gates: decision.gates,
    summary: summary.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))),
    timing: {
      elapsed_sec: round(elapsedSec, 3),
      trials_per_sec: round(trialRows.length / Math.max(elapsedSec, 1e-9), 2),
    },
  };
  await writeFile(path.join(out, "gates.json"), `${JSON.stringify(gatesPayload, null, 2)}\n`, "utf-8");
  const readback = [
    `# H2.2 Frontier Eval Readback`,
    ``,
    `Branch: \`${decision.branch}\``,
    ``,
    `Cells: ${cells.join(", ")} x ${args.seeds} seeds. cap_ok=${capOk}.`,
    ``,
    `## Summary`,
    ``,
    `| controller | C | B | fork_completion | max_reward_w |`,
    `| --- | ---: | ---: | ---: | ---: |`,
    ...summary.map((r) => `| ${r.controller} | ${round(r.C, 4)} | ${round(r.B, 4)} | ${round(r.fork_completion, 4)} | ${r.max_reward_w === null ? "" : round(r.max_reward_w, 4)} |`),
    ``,
    `## Margins`,
    ``,
    ...Object.entries(decision.margins).map(([k, v]) => `- \`${k}\`: ${round(v, 4)}`),
    ``,
    `## Breadth`,
    ``,
    `- support-edge cells: ${decision.breadth.support_edge_cells}`,
    `- support-edge forks: ${decision.breadth.support_edge_forks}`,
    ``,
    `## Gates`,
    ``,
    ...Object.entries(decision.gates).map(([k, v]) => `- \`${k}\`: ${v}`),
    ``,
  ].join("\n");
  await writeFile(path.join(out, "branch-readback.md"), readback, "utf-8");

  console.log(
    `H2.2 frontier eval: ${controllers.length} controllers x ${cells.length} cells x ${args.seeds} seeds = ${trialRows.length} trials in ${elapsedSec.toFixed(2)}s cap_ok=${capOk}`,
  );
  for (const row of summary) {
    console.log(
      `  ${row.controller.padEnd(24)} C=${round(row.C, 4)} B=${round(row.B, 4)} FC=${round(row.fork_completion, 4)} max_reward_w=${row.max_reward_w === null ? "" : round(row.max_reward_w, 4)}`,
    );
  }
  console.log(`  gates: ${JSON.stringify(decision.gates)} -> branch ${decision.branch}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
