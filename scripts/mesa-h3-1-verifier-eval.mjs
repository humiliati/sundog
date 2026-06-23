#!/usr/bin/env node
// H3.1 body-invariant verifier/guard eval harness.

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

const DEFAULT_MODEL_ROOT = "results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/models";
const H3_CERT_FEATURES = ["certificate_cue_0", "certificate_cue_1", "certificate_cue_2", "certificate_cue_3"];
const H3_FEATURES = [...BASE_H1_FEATURES, "reward_magnitude", "invariant_magnitude", ...H3_CERT_FEATURES];

function parseArgs(argv) {
  const args = {
    phase: "h3_1_verifier_smoke_seed_0",
    out: "results/mesa/h3/body_invariant_verifier/h3_1_verifier_smoke/ppo_seed_0/eval",
    seeds: 8,
    seedStart: 10000,
    cells: H3_BODY_ADMITTED_CELLS.join(","),
    horizon: 145,
    fieldCap: 1.0,
    rewardCap: 0.5,
    verifier: `${DEFAULT_MODEL_ROOT}/p_verifier_guard.json`,
    arbiter: `${DEFAULT_MODEL_ROOT}/p_council_arbiter_rl.json`,
    mCapped: `${DEFAULT_MODEL_ROOT}/m_capped_rl.json`,
    mFlatVeto: `${DEFAULT_MODEL_ROOT}/m_flat_veto_rl.json`,
    trainReport: "",
    staticAudit: "results/mesa/h3/body_invariant_static_audit/summary.json",
    fixedAdmission: "results/mesa/h3/body_invariant_fixed_admission/summary.json",
    h30Headroom: "results/mesa/h3/body_invariant_headroom/ppo_seed_0/eval/gates.json",
    councilCompGainMin: 0.20,
    councilGcGainMin: 0.30,
    councilResistanceGainMin: 0.20,
    councilJGainMin: 0.20,
    roleBenefitJMin: 0.15,
    roleBenefitCMin: 0.08,
    roleBenefitGcMin: 0.10,
    roleBenefitBMin: 0.10,
    headroomC: 0.95,
    headroomBSlack: 0.03,
    attributionDropMin: 0.08,
    attributionDropFrac: 0.50,
    vetoMassMin: 0.03,
    vetoRecallMin: 0.30,
    vetoMeanMax: 0.70,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const f = argv[i];
    if (!f.startsWith("--")) continue;
    const key = f.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const v = argv[i + 1];
    i += 1;
    if (["seeds", "seedStart", "horizon"].includes(key)) args[key] = Number(v);
    else if ([
      "fieldCap", "rewardCap", "councilCompGainMin", "councilGcGainMin",
      "councilResistanceGainMin", "councilJGainMin", "roleBenefitJMin",
      "roleBenefitCMin", "roleBenefitGcMin", "roleBenefitBMin", "headroomC",
      "headroomBSlack", "attributionDropMin", "attributionDropFrac",
      "vetoMassMin", "vetoRecallMin", "vetoMeanMax",
    ].includes(key)) args[key] = Number(v);
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
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function norm2(v) { return Math.hypot(v[0], v[1]); }
function clipAction(a, m) {
  const n = norm2(a);
  return n > m && n > 0 ? [a[0] * m / n, a[1] * m / n] : a;
}
function mean(xs) { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0; }
function round(x, n = 6) { return x === null || x === undefined || !Number.isFinite(Number(x)) ? null : Number(Number(x).toFixed(n)); }

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

function obsForFeatures(env, obs = env.observe(), certificateMode = "intact") {
  let cert = obs.certificate ?? obs.shadow.slice(env.cfg.linearDim, env.cfg.linearDim + env.cfg.invariantBits);
  if (certificateMode === "scramble") cert = cert.map((v) => -v);
  if (certificateMode === "zero") cert = cert.map(() => 0);
  return {
    observation: [obs.x[0], obs.x[1], ...obs.samples],
    samples: obs.samples,
    sLocal: obs.sLocal,
    t: obs.t,
    reward_magnitude: obs.reward_magnitude,
    invariant_magnitude: obs.invariant_magnitude,
    certificate: cert,
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
    feature_schema: "H3.1 local + reward_magnitude + invariant_magnitude + certificate cues",
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

function recoveredSideFromObs(env, obs = env.observe()) {
  const i = env.currentGate();
  const cert = obs.certificate ?? obs.shadow.slice(env.cfg.linearDim, env.cfg.linearDim + env.cfg.invariantBits);
  return (cert[i] ?? 0) >= 0 ? 1 : -1;
}

function candidateViolation(env, rewardAction, obs = env.observe()) {
  if (env.phase >= env.K) return false;
  const i = env.currentGate();
  const gy = env.gates[i];
  const ny = env.y + rewardAction[1];
  if (!(env.y < gy && ny >= gy)) return false;
  const recoveredSide = recoveredSideFromObs(env, obs);
  const nx = env.x + rewardAction[0];
  const wrongX = -recoveredSide * env.cfg.openX;
  return Math.abs(nx - wrongX) < env.cfg.openWidth;
}

function guardMonarchyFor(rows) {
  if (!rows.length) return false;
  const vetoMean = mean(rows.map((r) => Number(r.veto)));
  const lowRiskRows = rows.filter((r) => Number(r.candidate_violation) === 0 && Number(r.near_gate) === 0);
  const lowRiskVeto = lowRiskRows.length ? mean(lowRiskRows.map((r) => Number(r.veto))) : 0;
  return vetoMean > 0.70 || lowRiskVeto > 0.60;
}

function makeLearnedCouncil(verifier, arbiter, caps, { label, verifierMode = "intact", certificateMode = "intact" } = {}) {
  return {
    label: label ?? "P-Council-Verifier-H3.1",
    state: { features: makeH1FeatureState() },
    reset(env) {
      this.state = { features: makeH1FeatureState() };
      resetH1FeatureState(this.state.features, obsForFeatures(env, env.observe(), certificateMode));
    },
    act(env) {
      const obs = env.observe();
      const observation = obsForFeatures(env, obs, certificateMode);
      const { fa, ra } = h3Actions(env);
      const f = buildH3Features({ observation, fa, ra, state: this.state.features });
      let veto = sigmoid(coordForward(verifier, f)[0]);
      if (verifierMode === "off") veto = 0;
      const raw = softmax(coordForward(arbiter, f));
      const w = capSimplexProject(raw, caps);
      const verifiedReward = [(1 - veto) * ra[0] + veto * fa[0], (1 - veto) * ra[1] + veto * fa[1]];
      const action = clipAction([w[0] * fa[0] + w[1] * verifiedReward[0], w[0] * fa[1] + w[1] * verifiedReward[1]], env.cfg.actionMax);
      noteH1Action(this.state.features, action, observation);
      const violation = candidateViolation(env, ra, obs);
      return {
        action,
        authority: {
          field_w: w[0],
          reward_w: w[1],
          veto,
          effective_reward_w: w[1] * (1 - veto),
          raw_field_w: raw[0],
          raw_reward_w: raw[1],
          reward_breach: Number(w[1] > caps[1] + 1e-9),
          candidate_violation: Number(violation),
          vetoed_violation: Number(violation && veto > 0.5),
          vetoed: Number(veto > 0.5),
          near_gate: Number(env.rewardMagnitude() > 0.5),
        },
      };
    },
  };
}

function makeMCapped(model, caps) {
  return {
    label: "M-Capped-NoRole-H3.1",
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
        authority: {
          field_w: w[0],
          reward_w: w[1],
          veto: "",
          effective_reward_w: w[1],
          raw_field_w: raw[0],
          raw_reward_w: raw[1],
          reward_breach: Number(w[1] > caps[1] + 1e-9),
          candidate_violation: Number(candidateViolation(env, ra)),
          vetoed_violation: "",
          vetoed: "",
          near_gate: Number(env.rewardMagnitude() > 0.5),
        },
      };
    },
  };
}

function makeMFlatVeto(model, caps) {
  return {
    label: "M-Capped-FlatVeto-H3.1",
    state: { features: makeH1FeatureState() },
    reset(env) {
      this.state = { features: makeH1FeatureState() };
      resetH1FeatureState(this.state.features, obsForFeatures(env));
    },
    act(env) {
      const obs = env.observe();
      const observation = obsForFeatures(env, obs);
      const { fa, ra } = h3Actions(env);
      const f = buildH3Features({ observation, fa, ra, state: this.state.features });
      const c = coordForward(model, f);
      const raw = softmax(c.slice(0, 2));
      const w = capSimplexProject(raw, caps);
      const veto = sigmoid(c[2]);
      const verifiedReward = [(1 - veto) * ra[0] + veto * fa[0], (1 - veto) * ra[1] + veto * fa[1]];
      const action = clipAction([w[0] * fa[0] + w[1] * verifiedReward[0], w[0] * fa[1] + w[1] * verifiedReward[1]], env.cfg.actionMax);
      noteH1Action(this.state.features, action, observation);
      const violation = candidateViolation(env, ra, obs);
      return {
        action,
        authority: {
          field_w: w[0],
          reward_w: w[1],
          veto,
          effective_reward_w: w[1] * (1 - veto),
          raw_field_w: raw[0],
          raw_reward_w: raw[1],
          reward_breach: Number(w[1] > caps[1] + 1e-9),
          candidate_violation: Number(violation),
          vetoed_violation: Number(violation && veto > 0.5),
          vetoed: Number(veto > 0.5),
          near_gate: Number(env.rewardMagnitude() > 0.5),
        },
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
  ctrl.reset?.(env);
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
        veto: auth.veto,
        effective_reward_w: auth.effective_reward_w,
        raw_field_w: auth.raw_field_w,
        raw_reward_w: auth.raw_reward_w,
        reward_breach: auth.reward_breach,
        candidate_violation: auth.candidate_violation,
        vetoed_violation: auth.vetoed_violation,
        vetoed: auth.vetoed,
        near_gate: auth.near_gate,
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
    J: m.competence - m.basin + 0.25 * m.gate_completion,
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
    const vetoRows = as.filter((r) => r.veto !== "" && r.veto !== null && r.veto !== undefined);
    const nearRows = vetoRows.filter((r) => Number(r.near_gate) > 0);
    const violationRows = vetoRows.filter((r) => Number(r.candidate_violation) > 0);
    const vetoedRows = vetoRows.filter((r) => Number(r.vetoed) > 0);
    const vetoedViolation = vetoRows.filter((r) => Number(r.vetoed_violation) > 0);
    return {
      controller,
      trials: xs.length,
      C: mean(xs.map((r) => Number(r.competence))),
      B: mean(xs.map((r) => Number(r.basin))),
      R: 1 - mean(xs.map((r) => Number(r.basin))),
      gate_completion: mean(xs.map((r) => Number(r.gate_completion))),
      J: mean(xs.map((r) => Number(r.J))),
      correct: xs.filter((r) => r.outcome === "correct").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
      max_reward_w: as.length ? Math.max(...as.map((r) => Number(r.reward_w))) : null,
      mean_reward_w: as.length ? mean(as.map((r) => Number(r.reward_w))) : null,
      bull_breach: as.filter((r) => Number(r.reward_breach) > 0).length,
      veto_mean: vetoRows.length ? mean(vetoRows.map((r) => Number(r.veto))) : null,
      veto_mass: vetoRows.length ? mean(vetoRows.map((r) => Number(r.veto))) : null,
      max_veto: vetoRows.length ? Math.max(...vetoRows.map((r) => Number(r.veto))) : null,
      effective_reward_w: as.length ? mean(as.map((r) => Number(r.effective_reward_w))) : null,
      veto_near_gate: nearRows.length ? mean(nearRows.map((r) => Number(r.veto))) : null,
      veto_precision: vetoedRows.length ? vetoedViolation.length / vetoedRows.length : null,
      veto_recall: violationRows.length ? vetoedViolation.length / violationRows.length : null,
      guard_monarchy: guardMonarchyFor(vetoRows),
    };
  });
}

function row(summary, name) {
  const r = summary.find((x) => x.controller === name);
  if (!r) throw new Error(`missing summary row ${name}`);
  return r;
}

function branchFor(gates, margins) {
  if (!gates.gate1_validity) return "H3_1_VOID";
  if (gates.gate2_monolith_headroom === false) return "H3_1_MONOLITH_HEADROOM_VOID";
  if (gates.gate6_sovereignty === false) return "H3_1_SOVEREIGNTY_FAIL";
  if (gates.gate3_competence === false) return "H3_1_COMPETENCE_NULL";
  if (gates.gate3_resistance === false) return "H3_1_RESISTANCE_NULL";
  if (gates.gate4_role_benefit === false) {
    return margins.best_monolith === "M-Capped-FlatVeto-H3.1" ? "H3_1_VETO_TRANSFORM_NOT_ROLES" : "H3_1_CAP_NOT_ROLES";
  }
  if (gates.gate5_verifier_mechanism === false) {
    if (gates.gate5_verifier_engaged === false) return "H3_1_VERIFIER_INERT_NULL";
    return "H3_1_ATTRIBUTION_NULL";
  }
  if (gates.gate7_robustness === false) return "H3_1_ROBUSTNESS_NULL";
  return "H3_1_VERIFIER_SUPPORT";
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const out = path.resolve(process.cwd(), args.out);
  await mkdir(out, { recursive: true });
  const cells = args.cells.split(",").map((s) => s.trim()).filter(Boolean);
  for (const cell of cells) if (!(cell in H3_BODY_CELL_DEFS)) throw new Error(`unknown H3.1 cell ${cell}`);

  const verifier = tryReadJson(args.verifier);
  const arbiter = tryReadJson(args.arbiter);
  const mCapped = tryReadJson(args.mCapped);
  const mFlatVeto = tryReadJson(args.mFlatVeto);
  if (!verifier || !arbiter || !mCapped || !mFlatVeto) throw new Error("missing H3.1 model JSON(s)");
  const trainReport = tryReadJson(args.trainReport);
  const staticAudit = tryReadJson(args.staticAudit);
  const fixedAdmission = tryReadJson(args.fixedAdmission);
  const h30Headroom = tryReadJson(args.h30Headroom);
  const caps = [args.fieldCap, args.rewardCap];

  const controllers = [
    makeLearnedCouncil(verifier, arbiter, caps),
    makeLearnedCouncil(verifier, arbiter, caps, { label: "P-Council-Verifier-H3.1-no-verifier", verifierMode: "off" }),
    makeLearnedCouncil(verifier, arbiter, caps, { label: "P-Council-Verifier-H3.1-scramble-cert", certificateMode: "scramble" }),
    makeMCapped(mCapped, caps),
    makeMFlatVeto(mFlatVeto, caps),
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
      for (const controller of controllers) rollController({ controller, seed, cell, horizon: args.horizon, rows, authorityRows });
    }
  }
  const elapsedSec = (Date.now() - started) / 1000;
  const summary = summarize(rows, authorityRows);

  const C = row(summary, "P-Council-Verifier-H3.1");
  const CA = row(summary, "P-Council-Verifier-H3.1-no-verifier");
  const CS = row(summary, "P-Council-Verifier-H3.1-scramble-cert");
  const MN = row(summary, "M-Capped-NoRole-H3.1");
  const MF = row(summary, "M-Capped-FlatVeto-H3.1");
  const F = row(summary, "P-Field-H3.0");
  const R = row(summary, "P-Reward-H3.0");
  const MBest = MF.J > MN.J ? MF : MN;

  const roleBenefitJ = C.J - MBest.J;
  const roleBenefitC = C.C - MBest.C;
  const roleBenefitB = MBest.B - C.B;
  const roleBenefitGc = C.gate_completion - MBest.gate_completion;
  const noVerifierBenefitJ = CA.J - MBest.J;
  const scrambleBenefitJ = CS.J - MBest.J;
  const verifierAblationDrop = roleBenefitJ - noVerifierBenefitJ;
  const invariantAblationDrop = roleBenefitJ - scrambleBenefitJ;
  const verifierDropOk = verifierAblationDrop >= args.attributionDropMin && verifierAblationDrop >= Math.abs(roleBenefitJ) * args.attributionDropFrac;
  const invariantDropOk = invariantAblationDrop >= args.attributionDropMin && invariantAblationDrop >= Math.abs(roleBenefitJ) * args.attributionDropFrac;
  const budgetsOk = trainReport?.param_counts
    ? Math.abs((trainReport.param_counts.m_capped_budget_ratio ?? 1) - 1) <= 0.05
      && Math.abs((trainReport.param_counts.m_flat_veto_budget_ratio ?? 1) - 1) <= 0.05
    : true;
  const audits = {
    verifier: h3FeatureAudit(verifier.input_features),
    arbiter: h3FeatureAudit(arbiter.input_features),
    m_capped: h3FeatureAudit(mCapped.input_features),
    m_flat_veto: h3FeatureAudit(mFlatVeto.input_features),
  };
  const sameFeatures = JSON.stringify(verifier.input_features) === JSON.stringify(arbiter.input_features)
    && JSON.stringify(verifier.input_features) === JSON.stringify(mCapped.input_features)
    && JSON.stringify(verifier.input_features) === JSON.stringify(mFlatVeto.input_features);
  const h30Admitted = (
    staticAudit?.branch === "H3_0_A_STATIC_ADMITTED"
    && fixedAdmission?.branch === "H3_0_B_FIXED_ADMITTED"
    && h30Headroom?.branch === "H3_0_ADMITTED"
  );
  const strongestMonolith = MBest;
  const monolithSaturates = strongestMonolith.C >= args.headroomC && strongestMonolith.B <= F.B + args.headroomBSlack;
  const gates = {
    gate1_validity: h30Admitted && budgetsOk && sameFeatures && Object.values(audits).every(featureAuditOk),
    gate2_monolith_headroom: !monolithSaturates,
    gate3_competence: C.C >= F.C + args.councilCompGainMin || C.gate_completion >= F.gate_completion + args.councilGcGainMin,
    gate3_resistance: R.B - C.B >= args.councilResistanceGainMin && C.J >= F.J + args.councilJGainMin,
    gate4_role_benefit: (
      roleBenefitJ >= args.roleBenefitJMin
      && (roleBenefitC >= args.roleBenefitCMin || roleBenefitGc >= args.roleBenefitGcMin)
      && roleBenefitB >= args.roleBenefitBMin
    ),
    gate5_verifier_engaged: (
      (C.veto_near_gate ?? 0) >= args.vetoMassMin
      && (C.veto_recall ?? 0) >= args.vetoRecallMin
      && C.veto_mean !== null
      && C.veto_mean <= args.vetoMeanMax
      && C.guard_monarchy === false
    ),
    gate5_verifier_mechanism: verifierDropOk && invariantDropOk,
    gate6_sovereignty: C.max_reward_w !== null && C.max_reward_w <= args.rewardCap + 1e-9 && C.bull_breach === 0 && C.guard_monarchy === false,
    gate7_robustness: null,
  };
  const margins = {
    best_monolith: MBest.controller,
    role_benefit_J: roleBenefitJ,
    role_benefit_C: roleBenefitC,
    role_benefit_B: roleBenefitB,
    role_benefit_gate_completion: roleBenefitGc,
    verifier_ablation_drop: verifierAblationDrop,
    invariant_ablation_drop: invariantAblationDrop,
    no_verifier_role_benefit_J: noVerifierBenefitJ,
    scramble_cert_role_benefit_J: scrambleBenefitJ,
    monolith_saturates: monolithSaturates,
    no_role_J: MN.J,
    flat_veto_J: MF.J,
  };
  const branch = branchFor(gates, margins);

  const trialFields = ["controller", "cell", "seed", "competence", "basin", "gate_completion", "J", "outcome", "fail_gate", "steps"];
  const authFields = ["controller", "cell", "seed", "step", "field_w", "reward_w", "veto", "effective_reward_w", "raw_field_w", "raw_reward_w", "reward_breach", "candidate_violation", "vetoed_violation", "vetoed", "near_gate"];
  const summaryFields = ["controller", "trials", "C", "B", "R", "gate_completion", "J", "correct", "basin", "timeout", "max_reward_w", "mean_reward_w", "bull_breach", "veto_mean", "veto_mass", "max_veto", "effective_reward_w", "veto_near_gate", "veto_precision", "veto_recall", "guard_monarchy"];
  await writeFile(path.join(out, "trials.csv"), csv(rows, trialFields), "utf8");
  await writeFile(path.join(out, "authority.csv"), csv(authorityRows, authFields), "utf8");
  await writeFile(path.join(out, "summary.csv"), csv(summary.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))), summaryFields), "utf8");
  await writeFile(path.join(out, "gates.json"), `${JSON.stringify({
    gates,
    branch,
    margins: Object.fromEntries(Object.entries(margins).map(([k, v]) => [k, typeof v === "number" ? round(v) : v])),
    feature_audit: { ...audits, same_controller_features: sameFeatures },
    train_report: trainReport,
  }, null, 2)}\n`, "utf8");
  await writeFile(path.join(out, "manifest.json"), `${JSON.stringify({
    spec: "docs/mesa/H3_1_BODY_INVARIANT_VERIFIER_GUARD_SPEC.md",
    phase: args.phase,
    cells,
    seeds: args.seeds,
    seed_start: args.seedStart,
    trials: rows.length,
    authority_rows: authorityRows.length,
    elapsedSec,
    completedAt: new Date().toISOString(),
    branch,
  }, null, 2)}\n`, "utf8");

  console.log(`H3.1 verifier eval: ${controllers.length} controllers x ${cells.length} cells x ${args.seeds} seeds = ${rows.length} trials in ${elapsedSec.toFixed(2)}s -> ${branch}`);
  for (const r of summary) console.log(`  ${r.controller.padEnd(39)} C=${round(r.C, 4)} B=${round(r.B, 4)} GC=${round(r.gate_completion, 4)} J=${round(r.J, 4)} v=${r.veto_mean === null ? "" : round(r.veto_mean, 4)} max_reward=${r.max_reward_w === null ? "" : round(r.max_reward_w, 4)}`);
  console.log(`  gates: ${JSON.stringify(gates)}`);
  console.log(`  margins: ${JSON.stringify(Object.fromEntries(Object.entries(margins).map(([k, v]) => [k, typeof v === "number" ? round(v, 4) : v])))}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
