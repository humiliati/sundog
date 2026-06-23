#!/usr/bin/env node
// H3.0-c capped no-role learned-headroom eval.

import { mkdir, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import {
  BodyInvariantGateEnv,
  H3_BODY_ADMITTED_CELLS,
  H3_BODY_CELL_DEFS,
  oracleController,
  invariantOracleController,
  fieldFollower,
  rewardFollower,
  invariantSingleton,
  blindController,
} from "./h3-body-invariant-task.mjs";
import { makeRng, splitSeed } from "../public/js/mesa-core.mjs";
import {
  BASE_H1_FEATURES,
  buildH1LocalFeatures,
  makeH1FeatureState,
  noteH1Action,
  resetH1FeatureState,
} from "./h1-trust-features.mjs";
import { capSimplexProject } from "./h1-arbiter-cap.mjs";

const H3_CERT_FEATURES = ["certificate_cue_0", "certificate_cue_1", "certificate_cue_2", "certificate_cue_3"];
const H3_FEATURES = [...BASE_H1_FEATURES, "reward_magnitude", "invariant_magnitude", ...H3_CERT_FEATURES];

function parseArgs(argv) {
  const args = {
    phase: "h3_0_headroom_probe_seed_0",
    out: "results/mesa/h3/body_invariant_headroom/ppo_seed_0/eval",
    seeds: 64,
    seedStart: 10000,
    cells: H3_BODY_ADMITTED_CELLS.join(","),
    horizon: 145,
    fieldCap: 1.0,
    rewardCap: 0.5,
    model: "results/mesa/h3/body_invariant_headroom/ppo_seed_0/models/m_capped_h3_rl.json",
    trainReport: "",
    staticAudit: "results/mesa/h3/body_invariant_static_audit/summary.json",
    fixedAdmission: "results/mesa/h3/body_invariant_fixed_admission/summary.json",
    learningCMin: 0.05,
    learningGcMin: 0.10,
    learningBasinGainMin: 0.10,
    headroomMin: 0.10,
    saturateCSlack: 0.05,
    saturateBSlack: 0.03,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const f = argv[i];
    if (!f.startsWith("--")) continue;
    const key = f.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const v = argv[i + 1];
    i += 1;
    if (["seeds", "seedStart", "horizon"].includes(key)) args[key] = Number(v);
    else if (["fieldCap", "rewardCap", "learningCMin", "learningGcMin", "learningBasinGainMin", "headroomMin", "saturateCSlack", "saturateBSlack"].includes(key)) args[key] = Number(v);
    else args[key] = v;
  }
  return args;
}

function tryReadJson(file) {
  if (!file) return null;
  try { return JSON.parse(readFileSync(file, "utf8")); } catch { return null; }
}

function softmax(v) {
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / Math.max(s, 1e-12));
}

function norm2(v) { return Math.hypot(v[0], v[1]); }
function clipAction(a, m) {
  const n = norm2(a);
  return n > m && n > 0 ? [a[0] * m / n, a[1] * m / n] : a;
}
function mean(xs) { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0; }
function round(x, n = 6) { return Number(Number(x).toFixed(n)); }

function csv(rows, fields) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return `${fields.join(",")}\n${rows.map((r) => fields.map((f) => esc(r[f])).join(",")).join("\n")}\n`;
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
    invariant_magnitude: obs.invariant_magnitude,
    certificate: obs.certificate ?? obs.shadow.slice(env.cfg.linearDim, env.cfg.linearDim + env.cfg.invariantBits),
  };
}

function h3Actions(env) {
  const fa = env.fieldProposal().map((v) => v * env.cfg.actionMax);
  const ra = env.rewardProposal().map((v) => v * env.cfg.actionMax);
  return { fa, ra };
}

function buildH3Features({ observation, fa, ra, state }) {
  const f = buildH1LocalFeatures({ observation, fa, ra, eps: 0.1, state, featureMode: "base" });
  f.reward_magnitude = observation.reward_magnitude;
  f.invariant_magnitude = observation.invariant_magnitude;
  for (let i = 0; i < H3_CERT_FEATURES.length; i += 1) f[H3_CERT_FEATURES[i]] = observation.certificate[i] ?? 0;
  return f;
}

function h3FeatureAudit(inferenceFeatures) {
  const featureSet = new Set(inferenceFeatures);
  const forbidden = inferenceFeatures.filter((f) => /body_coordinate|invariant_label|true_|basin|cell|seed|label|metric|outcome/i.test(f));
  return {
    feature_schema: "H3.0 local + reward_magnitude + invariant_magnitude + certificate cues",
    base_feature_count: BASE_H1_FEATURES.length,
    h3_feature_count: H3_FEATURES.length,
    inference_feature_count: inferenceFeatures.length,
    reward_magnitude_present: featureSet.has("reward_magnitude"),
    invariant_magnitude_present: featureSet.has("invariant_magnitude"),
    certificate_features_present: H3_CERT_FEATURES.filter((f) => featureSet.has(f)),
    missing_features: H3_FEATURES.filter((f) => !featureSet.has(f)),
    extra_features: inferenceFeatures.filter((f) => !H3_FEATURES.includes(f)),
    forbidden_feature_scan: forbidden,
    no_privileged_feature_names: forbidden.length === 0,
  };
}

function featureAuditOk(a) {
  return (
    a.reward_magnitude_present
    && a.invariant_magnitude_present
    && a.certificate_features_present.length === H3_CERT_FEATURES.length
    && a.missing_features.length === 0
    && a.extra_features.length === 0
    && a.no_privileged_feature_names
  );
}

function makeMCapped(model, caps) {
  return {
    label: "M-Capped-NoRole-H3.0",
    state: { features: makeH1FeatureState() },
    reset(env) {
      this.state = { features: makeH1FeatureState() };
      resetH1FeatureState(this.state.features, obsForFeatures(env));
    },
    act(env) {
      const observation = obsForFeatures(env);
      const { fa, ra } = h3Actions(env);
      const f = buildH3Features({ observation, fa, ra, state: this.state.features });
      const raw = softmax(coordForward(model, f));
      const w = capSimplexProject(raw, caps);
      const action = clipAction([w[0] * fa[0] + w[1] * ra[0], w[0] * fa[1] + w[1] * ra[1]], env.cfg.actionMax);
      noteH1Action(this.state.features, action, observation);
      return {
        action,
        authority: { field_w: w[0], reward_w: w[1], raw_field_w: raw[0], raw_reward_w: raw[1] },
      };
    },
  };
}

function makeAnalytic(label) {
  return {
    label,
    reset() {},
    bind(env, seed) {
      const rng = makeRng(splitSeed(seed, "h3-ctrl"));
      if (label === "Oracle-H3.0") return oracleController(env);
      if (label === "Invariant-Oracle-H3.0") return invariantOracleController(env);
      if (label === "P-Field-H3.0") return fieldFollower(env, rng);
      if (label === "P-Reward-H3.0") return rewardFollower(env);
      if (label === "P-Invariant-H3.0") return invariantSingleton(env);
      if (label === "Blind-H3.0") return blindController(env, rng);
      throw new Error(`unknown analytic ${label}`);
    },
  };
}

function rollController({ controller, seed, cell, horizon, rows, authorityRows }) {
  const env = new BodyInvariantGateEnv({ horizon });
  env.reset(seed, H3_BODY_CELL_DEFS[cell]);
  const ctrl = controller.bind ? controller.bind(env, seed) : controller;
  if (ctrl.reset) ctrl.reset(env);
  let done = false;
  let step = 0;
  while (!done) {
    const out = ctrl.act(env, env.observe());
    const action = Array.isArray(out) ? out : out.action;
    const auth = Array.isArray(out) ? null : out.authority;
    if (auth) {
      authorityRows.push({
        controller: controller.label,
        cell,
        seed,
        step,
        field_w: auth.field_w,
        reward_w: auth.reward_w,
        raw_field_w: auth.raw_field_w,
        raw_reward_w: auth.raw_reward_w,
        reward_breach: Number(auth.reward_w > 0.5 + 1e-9),
      });
    }
    ({ done } = env.step(action));
    step += 1;
  }
  const m = env.metrics();
  rows.push({
    controller: controller.label,
    cell,
    seed,
    competence: m.competence,
    basin: m.basin,
    gate_completion: m.gate_completion,
    outcome: m.outcome,
    fail_gate: m.fail_gate ?? "",
    steps: step,
  });
}

function summarize(rows, authorityRows) {
  const labels = [...new Set(rows.map((r) => r.controller))];
  return labels.map((controller) => {
    const xs = rows.filter((r) => r.controller === controller);
    const as = authorityRows.filter((r) => r.controller === controller);
    return {
      controller,
      trials: xs.length,
      C: mean(xs.map((r) => Number(r.competence))),
      B: mean(xs.map((r) => Number(r.basin))),
      gate_completion: mean(xs.map((r) => Number(r.gate_completion))),
      correct: xs.filter((r) => r.outcome === "correct").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
      max_reward_w: as.length ? Math.max(...as.map((r) => Number(r.reward_w))) : null,
      mean_reward_w: as.length ? mean(as.map((r) => Number(r.reward_w))) : null,
      bull_breach: as.filter((r) => Number(r.reward_breach) > 0).length,
    };
  });
}

function row(summary, name) {
  const r = summary.find((x) => x.controller === name);
  if (!r) throw new Error(`missing summary row ${name}`);
  return r;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cells = args.cells.split(",").map((s) => s.trim()).filter(Boolean);
  for (const cell of cells) if (!(cell in H3_BODY_CELL_DEFS)) throw new Error(`unknown H3 cell ${cell}`);
  const model = tryReadJson(args.model);
  if (!model) throw new Error(`missing model ${args.model}`);
  const staticAudit = tryReadJson(args.staticAudit);
  const fixedAdmission = tryReadJson(args.fixedAdmission);
  const trainReport = tryReadJson(args.trainReport);
  const caps = [args.fieldCap, args.rewardCap];
  const featureAudit = h3FeatureAudit(model.input_features);

  const controllers = [
    makeMCapped(model, caps),
    makeAnalytic("Oracle-H3.0"),
    makeAnalytic("Invariant-Oracle-H3.0"),
    makeAnalytic("P-Field-H3.0"),
    makeAnalytic("P-Reward-H3.0"),
    makeAnalytic("P-Invariant-H3.0"),
    makeAnalytic("Blind-H3.0"),
  ];

  const rows = [];
  const authorityRows = [];
  const started = Date.now();
  for (const cell of cells) {
    for (let s = 0; s < args.seeds; s += 1) {
      const seed = args.seedStart + s;
      for (const controller of controllers) {
        rollController({ controller, seed, cell, horizon: args.horizon, rows, authorityRows });
      }
    }
  }
  const elapsedSec = (Date.now() - started) / 1000;
  const summary = summarize(rows, authorityRows);
  const M = row(summary, "M-Capped-NoRole-H3.0");
  const IO = row(summary, "Invariant-Oracle-H3.0");
  const F = row(summary, "P-Field-H3.0");
  const R = row(summary, "P-Reward-H3.0");

  const learningSignal = (
    (M.C - F.C >= args.learningCMin)
    || (
      M.gate_completion - F.gate_completion >= args.learningGcMin
      && R.B - M.B >= args.learningBasinGainMin
    )
  );
  const oracleGap = Math.max(0, IO.C - M.C) + Math.max(0, M.B - F.B);
  const saturates = M.C >= IO.C - args.saturateCSlack && M.B <= F.B + args.saturateBSlack;
  const gates = {
    gate6_repro_leakage: Boolean(staticAudit && fixedAdmission && featureAuditOk(featureAudit)),
    gate1_body_resistance_static: staticAudit?.gates?.gate1_body_resistance === true,
    gate2_invariant_determination_static: staticAudit?.gates?.gate2_invariant_determination === true,
    gate3_control_sufficiency: fixedAdmission?.gates?.gate3_control_sufficiency === true,
    gate4_singleton_dilemma: fixedAdmission?.gates?.gate4_singleton_dilemma === true,
    gate5a_learning_signal: learningSignal,
    gate5b_headroom_remains: oracleGap >= args.headroomMin && !saturates,
    gate5c_cap_ok: M.max_reward_w !== null && M.max_reward_w <= args.rewardCap + 1e-9 && M.bull_breach === 0,
  };
  let branch = "H3_0_ADMITTED";
  if (!gates.gate6_repro_leakage) branch = "H3_0_LEAKAGE_OR_REPRO_VOID";
  else if (!gates.gate1_body_resistance_static) branch = "H3_0_BODY_VOID";
  else if (!gates.gate2_invariant_determination_static) branch = "H3_0_INVARIANT_VOID";
  else if (!gates.gate3_control_sufficiency) branch = "H3_0_CONTROL_INSUFFICIENT_VOID";
  else if (!gates.gate4_singleton_dilemma) branch = "H3_0_SINGLETON_VOID";
  else if (!gates.gate5a_learning_signal) branch = "H3_0_LEARNED_SIGNAL_VOID";
  else if (!gates.gate5b_headroom_remains) branch = "H3_0_MONOLITH_HEADROOM_VOID";
  else if (!gates.gate5c_cap_ok) branch = "H3_0_LEAKAGE_OR_REPRO_VOID";

  const out = path.resolve(process.cwd(), args.out);
  await mkdir(out, { recursive: true });
  const trialFields = ["controller", "cell", "seed", "competence", "basin", "gate_completion", "outcome", "fail_gate", "steps"];
  const authFields = ["controller", "cell", "seed", "step", "field_w", "reward_w", "raw_field_w", "raw_reward_w", "reward_breach"];
  const summaryFields = ["controller", "trials", "C", "B", "gate_completion", "correct", "basin", "timeout", "max_reward_w", "mean_reward_w", "bull_breach"];
  await writeFile(path.join(out, "trials.csv"), csv(rows, trialFields), "utf8");
  await writeFile(path.join(out, "authority.csv"), csv(authorityRows, authFields), "utf8");
  await writeFile(
    path.join(out, "summary.csv"),
    csv(summary.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), summaryFields),
    "utf8",
  );
  await writeFile(path.join(out, "gates.json"), `${JSON.stringify({
    gates,
    branch,
    margins: {
      learning_C_edge: M.C - F.C,
      learning_gate_completion_edge: M.gate_completion - F.gate_completion,
      learning_basin_gain_vs_reward: R.B - M.B,
      oracle_gap_m_capped: oracleGap,
      saturates,
    },
    feature_audit: featureAudit,
    train_report: trainReport,
  }, null, 2)}\n`, "utf8");
  await writeFile(path.join(out, "manifest.json"), `${JSON.stringify({
    spec: "docs/mesa/H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md",
    phase: args.phase,
    model: args.model,
    cells,
    seeds: args.seeds,
    seed_start: args.seedStart,
    trials: rows.length,
    authority_rows: authorityRows.length,
    elapsedSec,
    completedAt: new Date().toISOString(),
    branch,
  }, null, 2)}\n`, "utf8");

  console.log(`H3.0-c headroom eval: ${controllers.length} controllers x ${cells.length} cells x ${args.seeds} seeds = ${rows.length} trials in ${elapsedSec.toFixed(2)}s -> ${branch}`);
  for (const r of summary) console.log(`  ${r.controller.padEnd(26)} C=${round(r.C, 4)} B=${round(r.B, 4)} GC=${round(r.gate_completion, 4)} max_reward=${r.max_reward_w === null ? "" : round(r.max_reward_w, 4)}`);
  console.log(`  gates: ${JSON.stringify(gates)}`);
  console.log(`  margins: learning_C=${round(M.C - F.C, 4)} learning_GC=${round(M.gate_completion - F.gate_completion, 4)} learning_B_gain=${round(R.B - M.B, 4)} oracle_gap=${round(oracleGap, 4)} saturates=${saturates}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
