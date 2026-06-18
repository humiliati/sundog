// MESA H1.2a -- coordinator dataset builder (Node, canonical env).
//
// Spec: docs/mesa/H1_2_SMALL_BAKEOFF_SPEC.md §4. Rolls the FROZEN field/reward
// heads against the real ShadowFieldEnv across train/val seeds x probe cells
// under a diversified behavior policy, and records, per visited step:
//   * ALLOWED inference features (local obs, proposals, norms, disagreement,
//     finite-diff gradient norm, short local history);
//   * PRIVILEGED labels (true gradient, false-basin geometry, the
//     direction-optimal field weight alpha*, the capped arbiter target weights,
//     the M-Adapter target coeffs, the guard risk scalar, rollout outcomes).
//
// Leakage contract: feature-schema.json partitions every column into
// inference_features vs labels_privileged. The trainer MUST only read
// inference_features (+ the guard's own risk output for the arbiter). True
// gradient and basin geometry are LABELS, never features (spec §4.2/§4.3).

import { mkdir, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  JsonPolicyController,
  SENSOR_TIERS,
  ShadowFieldEnv,
  makeTrialConfig,
  signatureGradient,
  falseBasinField,
  roundNumber,
  clamp,
} from "../public/js/mesa-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const POLICY_DIR = "results/mesa/phase2-matched-capacity/policies";

// --- probe cells (replicated from mesa-probe-slate.mjs; see H1.1 smoke note) --
function cellSeedHash(cellId, seed, channel = 0) {
  let h = (seed >>> 0) ^ (channel * 0x85ebca6b);
  for (let i = 0; i < cellId.length; i += 1) h = Math.imul(h ^ cellId.charCodeAt(i), 0x9e3779b1) >>> 0;
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}
function uniformRange(cellId, seed, channel, lo, hi) {
  return lo + cellSeedHash(cellId, seed, channel) * (hi - lo);
}
function buildProbeForCell(cellId, seed) {
  if (cellId === "nominal") return null;
  if (cellId === "geometric-light") {
    if (cellSeedHash(cellId, seed, 0) < 0.5) return { rotate: uniformRange(cellId, seed, 1, -Math.PI / 8, Math.PI / 8) };
    return { translate: [uniformRange(cellId, seed, 2, -0.5, 0.5), uniformRange(cellId, seed, 3, -0.5, 0.5)] };
  }
  if (cellId === "sensor-delay-light") return { sensorDelay: 1 };
  throw new Error(`unsupported cell: ${cellId}`);
}

function norm2(v) {
  return Math.hypot(v[0], v[1]);
}
function cos2(a, b) {
  const na = norm2(a);
  const nb = norm2(b);
  if (na < 1e-9 || nb < 1e-9) return 0;
  return (a[0] * b[0] + a[1] * b[1]) / (na * nb);
}

// Direction-optimal field weight: alpha in [0,1] maximizing cos(blend, g_true),
// where blend = alpha*fa + (1-alpha)*ra. Pure label-side computation (uses the
// true gradient). Grid search; neutral 0.5 if both proposals are ~0.
function bestAlpha(fa, ra, g) {
  if (norm2(fa) < 1e-9 && norm2(ra) < 1e-9) return 0.5;
  let bestA = 0.5;
  let bestC = -Infinity;
  for (let i = 0; i <= 50; i += 1) {
    const a = i / 50;
    const blend = [a * fa[0] + (1 - a) * ra[0], a * fa[1] + (1 - a) * ra[1]];
    const c = cos2(blend, g);
    if (c > bestC) {
      bestC = c;
      bestA = a;
    }
  }
  return bestA;
}

// --- blind council (H1.1 confidence-gated arbiter), used as one behavior policy
function blindCouncilAction(fa, ra, fdGradNorm, satHist, actionMax, cap) {
  const signalPresence = clamp(fdGradNorm / 0.25, 0, 1.5);
  const cField = (norm2(fa) / actionMax) * (0.5 + signalPresence) + 1e-6;
  const cReward = norm2(ra) / actionMax + 1e-6;
  const recentSat = satHist.length ? satHist.reduce((a, b) => a + b, 0) / satHist.length : 0;
  const cGuard = 0.05 + recentSat;
  let w = [cField, cReward, cGuard];
  const s = w[0] + w[1] + w[2];
  w = w.map((x) => x / s);
  for (let g = 0; g < 3; g += 1) {
    const over = w.findIndex((x) => x > cap + 1e-12);
    if (over === -1) break;
    const excess = w[over] - cap;
    w[over] = cap;
    const rest = w.reduce((a, b, i) => (i === over ? a : a + b), 0);
    if (rest <= 1e-12) break;
    w = w.map((x, i) => (i === over ? x : x + excess * (x / rest)));
  }
  return [w[0] * fa[0] + w[1] * ra[0], w[0] * fa[1] + w[1] * ra[1]];
}

function clipAction(a, actionMax) {
  const n = norm2(a);
  return n > actionMax && n > 0 ? [a[0] * actionMax / n, a[1] * actionMax / n] : a;
}

function parseArgs(argv) {
  const args = {
    out: "results/mesa/h1-pantheon/h1_2a/dataset",
    trainSeeds: 32,
    valSeeds: 16,
    trainSeedStart: 20000,
    valSeedStart: 20300,
    cells: "nominal,geometric-light,sensor-delay-light",
    horizon: 200,
    roleHardCap: 0.7,
    fieldPolicy: `${POLICY_DIR}/signature_ppo_terminal_small_seed_0_phase5.policy.json`,
    rewardPolicy: `${POLICY_DIR}/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json`,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const f = argv[i];
    if (!f.startsWith("--")) continue;
    const v = argv[i + 1];
    i += 1;
    if (f === "--out") args.out = v;
    else if (f === "--train-seeds") args.trainSeeds = Number.parseInt(v, 10);
    else if (f === "--val-seeds") args.valSeeds = Number.parseInt(v, 10);
    else if (f === "--train-seed-start") args.trainSeedStart = Number.parseInt(v, 10);
    else if (f === "--val-seed-start") args.valSeedStart = Number.parseInt(v, 10);
    else if (f === "--cells") args.cells = v;
    else if (f === "--horizon") args.horizon = Number.parseInt(v, 10);
    else if (f === "--role-hard-cap") args.roleHardCap = Number.parseFloat(v);
    else if (f === "--field-policy") args.fieldPolicy = v;
    else if (f === "--reward-policy") args.rewardPolicy = v;
    else if (f === "--phase") { /* label only */ }
    else throw new Error(`Unknown flag: ${f}`);
  }
  return args;
}

const INFERENCE_FEATURES = [
  "obs0", "obs1", "obs2", "obs3", "obs4", "obs5",
  "fa_x", "fa_y", "ra_x", "ra_y", "fa_norm", "ra_norm",
  "disagree_l2", "cos_agree", "fd_grad_norm", "hist_act_norm_prev", "hist_sLocal_prev",
];
const LABELS_PRIVILEGED = [
  "g_x", "g_y", "true_sig", "fb_val", "fb_dist",
  "alpha_star", "tgt_w_field", "tgt_w_reward", "tgt_w_guard",
  "tgt_madapter_field", "tgt_madapter_reward", "risk",
  "roll_basin_captured", "roll_terminal_sig", "behavior",
];
const META_COLUMNS = ["split", "cell", "seed", "t"];
const ALL_COLUMNS = [...META_COLUMNS, ...INFERENCE_FEATURES, ...LABELS_PRIVILEGED];

function rolloutRaw({ field, reward, seed, cellId, horizon, behavior, roleHardCap }) {
  const cfg = makeTrialConfig({ seed, sensorTier: SENSOR_TIERS.LOCAL_PROBE_FIELD, config: { horizon } });
  const env = new ShadowFieldEnv(cfg);
  const probe = buildProbeForCell(cellId, seed);
  if (probe) env.applyProbe(probe);
  const eps = cfg.probeEpsilon;
  const actionMax = cfg.actionMax;

  const steps = [];
  const satHist = [];
  let prevActNorm = 0;
  let prevSLocal = env.lastObservation.sLocal;
  let observation = env.lastObservation;

  while (!env.terminalOutcome) {
    const fa = field.act(observation, cfg).action;
    const ra = reward.act(observation, cfg).action;
    const obs = observation.observation; // 6-dim [x,y,s0,s1,s2,s3]
    const s = observation.samples;
    const fdGrad = [(s[0] - s[1]) / (2 * eps), (s[2] - s[3]) / (2 * eps)];
    const fdGradNorm = norm2(fdGrad);

    // labels (privileged): true gradient & basin geometry at THIS state
    const g = signatureGradient(env.x, env.xGoal, env.config);
    const fbVal = falseBasinField(env.x, env.config);
    const fbDist = Math.hypot(env.x[0] - env.config.falseBasinCenter[0], env.x[1] - env.config.falseBasinCenter[1]);
    const alpha = bestAlpha(fa, ra, g);

    // behavior action
    let act;
    if (behavior === "field") act = clipAction(fa, actionMax);
    else if (behavior === "reward") act = clipAction(ra, actionMax);
    else act = clipAction(blindCouncilAction(fa, ra, fdGradNorm, satHist, actionMax, roleHardCap), actionMax);

    steps.push({
      t: env.stepIndex,
      obs: obs.slice(),
      fa, ra, fdGradNorm,
      disagree_l2: Math.hypot(fa[0] - ra[0], fa[1] - ra[1]),
      cos_agree: cos2(fa, ra),
      hist_act_norm_prev: prevActNorm,
      hist_sLocal_prev: prevSLocal,
      g, fbVal, fbDist, alpha,
      trueSig: env.trueSignature(),
      reward_misaligned: cos2(ra, g) < 0 ? 1 : 0,
    });

    satHist.push(norm2(act) >= 0.99 * actionMax ? 1 : 0);
    while (satHist.length > 10) satHist.shift();
    prevActNorm = norm2(act);
    prevSLocal = observation.sLocal;

    const res = env.step(act);
    observation = res.observation;
  }

  const m = env.metrics();
  const distGoal = m.terminalDistance;
  const distFalse = Math.hypot(env.x[0] - env.config.falseBasinCenter[0], env.x[1] - env.config.falseBasinCenter[1]);
  const basinCaptured = distFalse < distGoal && m.terminalAlignment < 0.5 ? 1 : 0;
  return { steps, terminalSig: m.terminalAlignment, basinCaptured };
}

function buildTargetWeights(alpha, risk, cap) {
  // guard (hold) weight grows with privileged risk; field/reward share the rest
  // by the direction-optimal split alpha. Then hard-cap + renormalize, exactly
  // as the arbiter does at inference, so the target lives in-distribution.
  const wGuard = clamp(risk, 0, 1) * 0.6;
  let w = [alpha * (1 - wGuard), (1 - alpha) * (1 - wGuard), wGuard];
  let sum = w[0] + w[1] + w[2];
  w = w.map((x) => x / (sum || 1));
  for (let g = 0; g < 3; g += 1) {
    const over = w.findIndex((x) => x > cap + 1e-12);
    if (over === -1) break;
    const excess = w[over] - cap;
    w[over] = cap;
    const rest = w.reduce((a, b, i) => (i === over ? a : a + b), 0);
    if (rest <= 1e-12) break;
    w = w.map((x, i) => (i === over ? x : x + excess * (x / rest)));
  }
  return w;
}

function csvRow(r) {
  return ALL_COLUMNS.map((c) => {
    const v = r[c];
    if (v === null || v === undefined) return "";
    if (typeof v === "number") return Number.isFinite(v) ? String(roundNumber(v, 6)) : "";
    const t = String(v);
    return /[",\n]/.test(t) ? `"${t.replaceAll('"', '""')}"` : t;
  }).join(",");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cells = args.cells.split(",").map((c) => c.trim()).filter(Boolean);
  const outDir = path.resolve(repoRoot, args.out);
  const field = new JsonPolicyController(JSON.parse(readFileSync(path.resolve(repoRoot, args.fieldPolicy), "utf8")));
  const reward = new JsonPolicyController(JSON.parse(readFileSync(path.resolve(repoRoot, args.rewardPolicy), "utf8")));

  const behaviors = ["council", "reward", "field"];
  const splits = [
    { name: "train", start: args.trainSeedStart, count: args.trainSeeds },
    { name: "val", start: args.valSeedStart, count: args.valSeeds },
  ];

  const t0 = Date.now();
  const rawBySplit = { train: [], val: [] };
  // pass 1: rollouts -> raw steps + rollout outcomes
  for (const split of splits) {
    for (const cellId of cells) {
      for (let i = 0; i < split.count; i += 1) {
        const seed = split.start + i;
        const behavior = behaviors[seed % behaviors.length];
        const roll = rolloutRaw({ field, reward, seed, cellId, horizon: args.horizon, behavior, roleHardCap: args.roleHardCap });
        rawBySplit[split.name].push({ split: split.name, cell: cellId, seed, behavior, ...roll });
      }
    }
  }

  // per-cell median terminal signature (self-contained "low terminal" criterion)
  const cellMedian = {};
  for (const cellId of cells) {
    const sigs = [];
    for (const k of ["train", "val"]) for (const r of rawBySplit[k]) if (r.cell === cellId) sigs.push(r.terminalSig);
    sigs.sort((a, b) => a - b);
    cellMedian[cellId] = sigs.length ? sigs[Math.floor(sigs.length / 2)] : 0;
  }

  // pass 2: compute risk + targets, emit rows
  const rowsBySplit = { train: [], val: [] };
  let nBasin = 0;
  let nRows = 0;
  for (const split of splits) {
    for (const roll of rawBySplit[split.name]) {
      const lowTerminal = roll.terminalSig < cellMedian[roll.cell] ? 1 : 0;
      const rollFailed = roll.basinCaptured || lowTerminal;
      if (roll.basinCaptured) nBasin += 1;
      for (const st of roll.steps) {
        const risk = clamp(
          0.55 * roll.basinCaptured + 0.30 * lowTerminal + 0.15 * (st.reward_misaligned && rollFailed ? 1 : 0),
          0, 1,
        );
        const tw = buildTargetWeights(st.alpha, risk, args.roleHardCap);
        rowsBySplit[split.name].push({
          split: split.name, cell: roll.cell, seed: roll.seed, t: st.t,
          obs0: st.obs[0], obs1: st.obs[1], obs2: st.obs[2], obs3: st.obs[3], obs4: st.obs[4], obs5: st.obs[5],
          fa_x: st.fa[0], fa_y: st.fa[1], ra_x: st.ra[0], ra_y: st.ra[1],
          fa_norm: norm2(st.fa), ra_norm: norm2(st.ra),
          disagree_l2: st.disagree_l2, cos_agree: st.cos_agree, fd_grad_norm: st.fdGradNorm,
          hist_act_norm_prev: st.hist_act_norm_prev, hist_sLocal_prev: st.hist_sLocal_prev,
          g_x: st.g[0], g_y: st.g[1], true_sig: st.trueSig,
          fb_val: st.fbVal, fb_dist: st.fbDist,
          alpha_star: st.alpha, tgt_w_field: tw[0], tgt_w_reward: tw[1], tgt_w_guard: tw[2],
          tgt_madapter_field: st.alpha, tgt_madapter_reward: 1 - st.alpha,
          risk, roll_basin_captured: roll.basinCaptured, roll_terminal_sig: roll.terminalSig,
          behavior: roll.behavior,
        });
        nRows += 1;
      }
    }
  }

  await mkdir(outDir, { recursive: true });
  for (const split of splits) {
    await writeFile(
      path.join(outDir, `${split.name}.csv`),
      `${ALL_COLUMNS.join(",")}\n${rowsBySplit[split.name].map(csvRow).join("\n")}\n`,
      "utf8",
    );
  }
  const elapsed = (Date.now() - t0) / 1000;
  const schema = {
    format: "mesa-h1-coordinator-dataset-v1",
    meta_columns: META_COLUMNS,
    inference_features: INFERENCE_FEATURES,
    arbiter_extra_input: ["guard_risk"],
    labels_privileged: LABELS_PRIVILEGED,
    targets: {
      guard: "risk",
      arbiter: ["tgt_w_field", "tgt_w_reward", "tgt_w_guard"],
      m_adapter: ["tgt_madapter_field", "tgt_madapter_reward"],
    },
    leakage_rule: "trainer X may only read inference_features (+ guard_risk for arbiter); labels_privileged are targets/diagnostics only.",
  };
  await writeFile(path.join(outDir, "feature-schema.json"), `${JSON.stringify(schema, null, 2)}\n`, "utf8");
  const manifest = {
    spec: "docs/mesa/H1_2_SMALL_BAKEOFF_SPEC.md §4 (H1.2a capped probe)",
    generated: new Date().toISOString(),
    args,
    cells,
    behavior_policy: "round-robin by seed%3: 0=blind-council, 1=reward-only, 2=field-only",
    splits: { train: { start: args.trainSeedStart, count: args.trainSeeds }, val: { start: args.valSeedStart, count: args.valSeeds } },
    rows: { train: rowsBySplit.train.length, val: rowsBySplit.val.length, total: nRows },
    rollouts_basin_captured: nBasin,
    cell_median_terminal_sig: cellMedian,
    elapsed_sec: roundNumber(elapsed, 3),
    rows_per_sec: roundNumber(nRows / Math.max(elapsed, 1e-6), 1),
  };
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  // eslint-disable-next-line no-console
  console.log(`H1.2a dataset: train=${rowsBySplit.train.length} val=${rowsBySplit.val.length} rows  basin_rollouts=${nBasin}  ${elapsed.toFixed(2)}s  ${manifest.rows_per_sec} rows/s`);
  console.log(`  cells=${cells.join(",")}  features=${INFERENCE_FEATURES.length}  cell_median_sig=${JSON.stringify(cellMedian)}`);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
