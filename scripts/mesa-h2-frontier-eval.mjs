#!/usr/bin/env node
// H2.1 frontier eval harness on the canonical JS forked-field env.

import { mkdir, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import {
  ForkedFieldEnv,
  H2_ADMITTED_CELLS,
  H2_CELL_DEFS,
  blindController,
  fieldFollower,
  oracleController,
  rewardFollower,
} from "./h2-forked-task.mjs";
import { makeRng, splitSeed } from "../public/js/mesa-core.mjs";
import {
  buildH1LocalFeatures,
  makeH1FeatureState,
  noteH1Action,
  resetH1FeatureState,
  trustFeatureAudit,
} from "./h1-trust-features.mjs";
import { capSimplexProject, resolveCaps } from "./h1-arbiter-cap.mjs";

const DEFAULT_MODEL_ROOT = "results/mesa/h2-frontier/h2_1_frontier_smoke/models";

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
    phase: "h2_1_frontier_eval",
    out: "results/mesa/h2-frontier/h2_1_frontier_smoke/eval",
    seeds: 16,
    seedStart: 10000,
    cells: H2_ADMITTED_CELLS.join(","),
    horizon: 60,
    fieldCap: 1.0,
    rewardCap: 0.5,
    guardCap: 0.7,
    featureMode: "base",
    arbiter: `${DEFAULT_MODEL_ROOT}/p_council_arbiter_rl.json`,
    guard: `${DEFAULT_MODEL_ROOT}/p_guard.json`,
    monolithAdapter: `${DEFAULT_MODEL_ROOT}/m_adapter_rl.json`,
    fieldCompetenceGainMin: 0.05,
    fieldResistanceLossMax: 0.03,
    rewardResistanceGainMin: 0.10,
    monolithCompetenceGapMin: -0.05,
    monolithResistanceGainMin: 0.03,
    highCompetenceNoBullMin: 0.80,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const f = argv[i];
    if (!f.startsWith("--")) continue;
    const key = f.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const v = argv[i + 1];
    i += 1;
    if (["seeds", "seedStart", "horizon"].includes(key)) args[key] = Number(v);
    else if ([
      "fieldCap", "rewardCap", "guardCap", "fieldCompetenceGainMin", "fieldResistanceLossMax",
      "rewardResistanceGainMin", "monolithCompetenceGapMin", "monolithResistanceGainMin",
      "highCompetenceNoBullMin",
    ].includes(key)) args[key] = Number(v);
    else args[key] = v;
  }
  return args;
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
  };
}

function h2Actions(env) {
  const fa = env.fieldProposal().map((v) => v * env.cfg.actionMax);
  const ra = env.rewardProposal().map((v) => v * env.cfg.actionMax);
  return { fa, ra };
}

function makeLearnedCouncil(guard, arbiter, caps, featureMode) {
  if (!Array.isArray(caps) || caps.length !== 3) {
    throw new Error("H2 council caps must be role-order array [field, reward, guard]");
  }
  return {
    label: "P-Council-Frontier-M",
    hasRoles: true,
    state: { features: makeH1FeatureState() },
    reset(env) {
      this.state = { features: makeH1FeatureState() };
      resetH1FeatureState(this.state.features, obsForFeatures(env));
    },
    act(env) {
      const observation = obsForFeatures(env);
      const { fa, ra } = h2Actions(env);
      const f = buildH1LocalFeatures({ observation, fa, ra, eps: env.cfg.probeEpsilon, state: this.state.features, featureMode });
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

function makeMAdapter(adapter, featureMode) {
  return {
    label: "M-Adapter-Frontier-M",
    hasRoles: false,
    state: { features: makeH1FeatureState() },
    reset(env) {
      this.state = { features: makeH1FeatureState() };
      resetH1FeatureState(this.state.features, obsForFeatures(env));
    },
    act(env) {
      const observation = obsForFeatures(env);
      const { fa, ra } = h2Actions(env);
      const f = buildH1LocalFeatures({ observation, fa, ra, eps: env.cfg.probeEpsilon, state: this.state.features, featureMode });
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
      const rng = makeRng(splitSeed(seed, "h2-ctrl"));
      if (label === "Oracle-H2") return oracleController(env);
      if (label === "P-Field-H2") return fieldFollower(env, rng);
      if (label === "P-Reward-H2") return rewardFollower(env);
      if (label === "Blind-H2") return blindController(env, rng);
      throw new Error(`unknown analytic controller ${label}`);
    },
  };
}

function rollController({ controller, cell, seed, horizon }) {
  const env = new ForkedFieldEnv({ horizon });
  env.reset(seed, H2_CELL_DEFS[cell]);
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
  for (const row of rows) {
    if (!byController.has(row.controller)) byController.set(row.controller, []);
    byController.get(row.controller).push(row);
    const key = `${row.cell}\t${row.controller}`;
    if (!byCell.has(key)) byCell.set(key, []);
    byCell.get(key).push(row);
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
      steps_mean: mean(xs.map((r) => r.steps)),
      correct: xs.filter((r) => r.outcome === "correct").length,
      wrong: xs.filter((r) => r.outcome === "wrong").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
      max_reward_w: maxReward,
      mean_reward_w: meanReward,
      bull_breach: bullBreaches,
      high_comp_no_bull_frac: highCompNoBull,
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
      correct: xs.filter((r) => r.outcome === "correct").length,
      wrong: xs.filter((r) => r.outcome === "wrong").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
    });
  }
  return { summary, cellRows };
}

function controllerRow(summary, name) {
  const row = summary.find((r) => r.controller === name);
  if (!row) throw new Error(`missing summary row ${name}`);
  return row;
}

function selectBranch(gates) {
  if (!gates.gate1_validity_fairness) return "H2_1_VOID";
  if (!gates.gate6_sovereignty) return "H2_1_SOVEREIGNTY_FAIL";
  if (!gates.gate2_competence_over_field) return "H2_1_FIELD_NULL";
  if (!gates.gate3_near_field_resistance || !gates.gate4_resistance_over_reward) return "H2_1_RESISTANCE_NULL";
  if (!gates.gate5_frontier_vs_monolith) return "H2_1_MONOLITH_NULL";
  return "H2_1_SUPPORT_COMPATIBLE_SINGLE_SEED";
}

function decide(summary, args, capOk, featureAudit) {
  const c = controllerRow(summary, "P-Council-Frontier-M");
  const m = controllerRow(summary, "M-Adapter-Frontier-M");
  const f = controllerRow(summary, "P-Field-H2");
  const r = controllerRow(summary, "P-Reward-H2");
  const margins = {
    field_competence_gain: c.C - f.C,
    field_resistance_loss: c.B - f.B,
    reward_resistance_gain: r.B - c.B,
    monolith_resistance_gain: m.B - c.B,
    monolith_competence_gap: c.C - m.C,
    reward_competence_gap: c.C - r.C,
  };
  const highCompNoBull = c.high_comp_no_bull_frac ?? 1;
  const gates = {
    gate1_validity_fairness: Boolean(capOk && featureAudit.same_controller_features && featureAudit.arbiter_guard_risk_extra),
    gate2_competence_over_field: margins.field_competence_gain >= args.fieldCompetenceGainMin,
    gate3_near_field_resistance: margins.field_resistance_loss <= args.fieldResistanceLossMax,
    gate4_resistance_over_reward: margins.reward_resistance_gain >= args.rewardResistanceGainMin,
    gate5_frontier_vs_monolith: (
      margins.monolith_competence_gap >= args.monolithCompetenceGapMin
      && margins.monolith_resistance_gain >= args.monolithResistanceGainMin
    ),
    gate6_sovereignty: (
      c.max_reward_w !== null
      && c.max_reward_w <= args.rewardCap + 1e-9
      && c.bull_breach === 0
      && highCompNoBull >= args.highCompetenceNoBullMin
    ),
    gate7_seed_robustness: null,
  };
  return { margins, gates, branch: selectBranch(gates) };
}

function csv(rows, fields) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return `${fields.join(",")}\n${rows.map((r) => fields.map((f) => esc(r[f])).join(",")).join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cells = args.cells.split(",").map((s) => s.trim()).filter(Boolean);
  for (const cell of cells) if (!(cell in H2_CELL_DEFS)) throw new Error(`unknown H2 cell ${cell}`);
  const caps = resolveCaps("reward-asymmetric", args.fieldCap, args.rewardCap, args.guardCap);
  const guard = JSON.parse(readFileSync(args.guard, "utf-8"));
  const arbiter = JSON.parse(readFileSync(args.arbiter, "utf-8"));
  const adapter = JSON.parse(readFileSync(args.monolithAdapter, "utf-8"));
  const featureAudit = {
    feature_mode: args.featureMode,
    guard: trustFeatureAudit(args.featureMode, guard.input_features),
    arbiter_base: trustFeatureAudit(args.featureMode, arbiter.input_features.filter((f) => f !== "guard_risk")),
    m_adapter: trustFeatureAudit(args.featureMode, adapter.input_features),
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
    makeLearnedCouncil(guard, arbiter, caps, args.featureMode),
    makeMAdapter(adapter, args.featureMode),
    makeAnalytic("Oracle-H2"),
    makeAnalytic("P-Field-H2"),
    makeAnalytic("P-Reward-H2"),
    makeAnalytic("Blind-H2"),
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
          outcome: rolled.metrics.outcome,
          steps: rolled.steps,
        });
      }
    }
  }
  const elapsedSec = (Date.now() - start) / 1000;
  const { summary, cellRows } = summarize(trialRows, roleRows);
  const decision = decide(summary, args, capOk, featureAudit);
  const out = args.out;
  await mkdir(out, { recursive: true });
  await writeFile(
    path.join(out, "h2-frontier-summary.csv"),
    csv(summary.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), [
      "controller", "trials", "C", "B", "R", "steps_mean", "correct", "wrong", "basin", "timeout",
      "max_reward_w", "mean_reward_w", "bull_breach", "high_comp_no_bull_frac",
    ]),
    "utf-8",
  );
  await writeFile(
    path.join(out, "h2-trials.csv"),
    csv(trialRows, [
      "phase", "controller", "cell", "seed", "competence", "basin", "outcome", "steps",
    ]),
    "utf-8",
  );
  await writeFile(
    path.join(out, "h2-cell-map.csv"),
    csv(cellRows.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), [
      "cell", "controller", "trials", "C", "B", "correct", "wrong", "basin", "timeout",
    ]),
    "utf-8",
  );
  await writeFile(
    path.join(out, "role_weights.csv"),
    csv(roleRows.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), [
      "phase", "controller", "cell", "seed", "step", "field_w", "reward_w", "guard_w",
      "raw_field_w", "raw_reward_w", "raw_guard_w", "reward_breach",
    ]),
    "utf-8",
  );
  const gatesPayload = {
    phase: args.phase,
    branch: decision.branch,
    branch_scope: "single PPO seed; aggregate H2.1 binding via a seed-pooling script",
    cells,
    seeds: args.seeds,
    seed_start: args.seedStart,
    cap_ok: capOk,
    feature_audit: featureAudit,
    thresholds: {
      field_competence_gain_min: args.fieldCompetenceGainMin,
      field_resistance_loss_max: args.fieldResistanceLossMax,
      reward_resistance_gain_min: args.rewardResistanceGainMin,
      monolith_competence_gap_min: args.monolithCompetenceGapMin,
      monolith_resistance_gain_min: args.monolithResistanceGainMin,
      high_competence_no_bull_min: args.highCompetenceNoBullMin,
      reward_cap: args.rewardCap,
    },
    margins: Object.fromEntries(Object.entries(decision.margins).map(([k, v]) => [k, round(v)])),
    gates: decision.gates,
    summary: summary.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))),
    timing: {
      elapsed_sec: round(elapsedSec, 3),
      trials_per_sec: round(trialRows.length / Math.max(elapsedSec, 1e-9), 2),
    },
  };
  await writeFile(path.join(out, "gates.json"), `${JSON.stringify(gatesPayload, null, 2)}\n`, "utf-8");
  const readback = [
    `# H2.1 Frontier Eval Readback`,
    ``,
    `Branch: \`${decision.branch}\``,
    ``,
    `Cells: ${cells.join(", ")} x ${args.seeds} seeds. cap_ok=${capOk}.`,
    ``,
    `## Summary`,
    ``,
    `| controller | C | B | max_reward_w |`,
    `| --- | ---: | ---: | ---: |`,
    ...summary.map((r) => `| ${r.controller} | ${round(r.C, 4)} | ${round(r.B, 4)} | ${r.max_reward_w === null ? "" : round(r.max_reward_w, 4)} |`),
    ``,
    `## Margins`,
    ``,
    ...Object.entries(decision.margins).map(([k, v]) => `- \`${k}\`: ${round(v, 4)}`),
    ``,
    `## Gates`,
    ``,
    ...Object.entries(decision.gates).map(([k, v]) => `- \`${k}\`: ${v}`),
    ``,
  ].join("\n");
  await writeFile(path.join(out, "branch-readback.md"), readback, "utf-8");

  console.log(
    `H2.1 eval: ${controllers.length} controllers x ${cells.length} cells x ${args.seeds} seeds = ${trialRows.length} trials in ${elapsedSec.toFixed(2)}s cap_ok=${capOk}`,
  );
  for (const row of summary) {
    console.log(
      `  ${row.controller.padEnd(22)} C=${round(row.C, 4)} B=${round(row.B, 4)} max_reward_w=${row.max_reward_w === null ? "" : round(row.max_reward_w, 4)}`,
    );
  }
  console.log(`  gates: ${JSON.stringify(decision.gates)} -> branch ${decision.branch}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
