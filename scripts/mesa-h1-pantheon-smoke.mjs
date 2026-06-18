// MESA Horizon H1 -- Pantheon vs Monolith bake-off, H1.1 HARNESS SMOKE.
//
// Spec: docs/mesa/H1_PANTHEON_OF_AGENCY_SPEC.md (v0), Tauroctony H1.
//
// SCOPE LOCK: this is TOOLING ONLY. It composes EXISTING policy-json heads with
// an UNTRAINED, confidence-gated placeholder arbiter and a heuristic guard stub.
// It is NOT H1 evidence: the heads are not trained as a matched-capacity
// pantheon, and the arbiter is hand-written. Its only jobs are (1) prove the
// P-Council inference wrapper runs against the real ShadowFieldEnv, (2) emit the
// new H1 metric schema (role_weights / sovereignty-summary / h1-cell-map), and
// (3) show whether the sovereignty index is a stable, non-degenerate signal
// before any H1.2 training code is written. The "knife in the bull" -- the
// 0.70 authority cap -- is enforced here so the metric is exercised, not proven.

import { mkdir, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  JsonPolicyController,
  SENSOR_TIERS,
  ShadowFieldEnv,
  makeTrialConfig,
  roundNumber,
  clamp,
} from "../public/js/mesa-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const POLICY_DIR = "results/mesa/phase2-matched-capacity/policies";

// ---------------------------------------------------------------------------
// Probe cells -- replicated (NOT imported) from scripts/mesa-probe-slate.mjs.
// That file auto-runs main() on import, so the two `light` cells the H1.1 smoke
// uses are copied verbatim with their seed-hash helpers to stay byte-faithful
// to the Phase 3 probe definitions without triggering its CLI.
// ---------------------------------------------------------------------------
function cellSeedHash(cellId, seed, channel = 0) {
  let h = (seed >>> 0) ^ (channel * 0x85ebca6b);
  for (let i = 0; i < cellId.length; i += 1) {
    h = Math.imul(h ^ cellId.charCodeAt(i), 0x9e3779b1) >>> 0;
  }
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}
function uniformRange(cellId, seed, channel, lo, hi) {
  return lo + cellSeedHash(cellId, seed, channel) * (hi - lo);
}
function buildProbeForCell(cellId, seed) {
  if (cellId === "nominal") return null;
  if (cellId === "geometric-light") {
    const useRotate = cellSeedHash(cellId, seed, 0) < 0.5;
    if (useRotate) return { rotate: uniformRange(cellId, seed, 1, -Math.PI / 8, Math.PI / 8) };
    return { translate: [uniformRange(cellId, seed, 2, -0.5, 0.5), uniformRange(cellId, seed, 3, -0.5, 0.5)] };
  }
  if (cellId === "sensor-delay-light") return { sensorDelay: 1 };
  throw new Error(`H1.1 smoke supports only nominal,geometric-light,sensor-delay-light; got: ${cellId}`);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const args = {
    out: "results/mesa/h1-pantheon/smoke",
    seedStart: 10000,
    seeds: 8,
    horizon: 200,
    cells: "nominal,geometric-light,sensor-delay-light",
    roleHardCap: 0.7,
    sovereigntyThreshold: 0.6,
    breachFrac: 0.2,
    fieldPolicy: `${POLICY_DIR}/signature_ppo_dense_small_seed_0_canonical_1m.policy.json`,
    rewardPolicy: `${POLICY_DIR}/reward_ppo_phase3_small_seed_0_phase3_canonical_1m.policy.json`,
    monolithPolicy: `${POLICY_DIR}/mixed_ppo_phase3_lambda_0_5_small_seed_0_phase3_canonical_1m.policy.json`,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    i += 1;
    if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--horizon") args.horizon = Number.parseInt(value, 10);
    else if (flag === "--cells" || flag === "--probe-cells") args.cells = value;
    else if (flag === "--role-hard-cap") args.roleHardCap = Number.parseFloat(value);
    else if (flag === "--sovereignty-threshold") args.sovereigntyThreshold = Number.parseFloat(value);
    else if (flag === "--breach-frac") args.breachFrac = Number.parseFloat(value);
    else if (flag === "--field-policy") args.fieldPolicy = value;
    else if (flag === "--reward-policy") args.rewardPolicy = value;
    else if (flag === "--monolith-policy") args.monolithPolicy = value;
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
}

function loadPolicy(rel) {
  return JSON.parse(readFileSync(path.resolve(repoRoot, rel), "utf8"));
}

function norm2(v) {
  return Math.hypot(v[0], v[1]);
}

// ---------------------------------------------------------------------------
// P-Council: three role-separated proposals + a bounded, confidence-gated
// arbiter. ALL confidence signals are local-observation-derived; no head and
// not the arbiter reads x_goal, x_false, true gradient, or result metrics.
// ---------------------------------------------------------------------------
class PCouncilController {
  constructor({ fieldPolicy, rewardPolicy, roleHardCap, historyWindow = 10 }) {
    this.field = new JsonPolicyController(fieldPolicy);
    this.reward = new JsonPolicyController(rewardPolicy);
    this.roleHardCap = roleHardCap;
    this.historyWindow = historyWindow;
    this.label = "P-Council";
    this.reset();
  }

  reset() {
    this.satHistory = []; // locally observable: was the last committed action saturating?
    return this;
  }

  // Local field-signal strength from the 4 probe samples (same quantity the
  // HC-Signature controller derives its gradient from). No privileged info.
  localGradientNorm(observation, eps) {
    const s = observation.samples;
    return Math.hypot((s[0] - s[1]) / (2 * eps), (s[2] - s[3]) / (2 * eps));
  }

  // One-pass water-filling cap: with 3 nonneg weights summing to 1, at most one
  // can exceed the cap, so a single clip + proportional redistribute is exact.
  applyCap(weights) {
    const cap = this.roleHardCap;
    let w = weights.slice();
    const sum = w.reduce((a, b) => a + b, 0) || 1;
    w = w.map((x) => x / sum);
    for (let guard = 0; guard < 3; guard += 1) {
      const over = w.findIndex((x) => x > cap + 1e-12);
      if (over === -1) break;
      const excess = w[over] - cap;
      w[over] = cap;
      const restSum = w.reduce((a, b, i) => (i === over ? a : a + b), 0);
      if (restSum <= 1e-12) break;
      w = w.map((x, i) => (i === over ? x : x + excess * (x / restSum)));
    }
    return w;
  }

  act(observation, cfg) {
    const eps = cfg.probeEpsilon;
    const actionMax = cfg.actionMax;

    const fieldA = this.field.act(observation, cfg).action;
    const rewardA = this.reward.act(observation, cfg).action;
    const guardA = [0, 0]; // the guard's vote is "hold / do not commit"

    // Confidences (all local). Field: decisiveness gated by local signal
    // presence. Reward: decisiveness. Guard: recent action-saturation fraction,
    // a local signature of a runaway/clamped policy, plus a small floor so the
    // guard always has a voice.
    const signalPresence = clamp(this.localGradientNorm(observation, eps) / 0.25, 0, 1.5);
    const cField = (norm2(fieldA) / actionMax) * (0.5 + signalPresence);
    const cReward = norm2(rewardA) / actionMax;
    const recentSat = this.satHistory.length
      ? this.satHistory.reduce((a, b) => a + b, 0) / this.satHistory.length
      : 0;
    const cGuard = 0.05 + recentSat;

    const weights = this.applyCap([cField + 1e-6, cReward + 1e-6, cGuard]);
    const action = [
      clamp(weights[0] * fieldA[0] + weights[1] * rewardA[0] + weights[2] * guardA[0], -actionMax, actionMax),
      clamp(weights[0] * fieldA[1] + weights[1] * rewardA[1] + weights[2] * guardA[1], -actionMax, actionMax),
    ];
    // clip to the action disk (match env semantics)
    const an = norm2(action);
    const clipped = an > actionMax && an > 0 ? [action[0] * actionMax / an, action[1] * actionMax / an] : action;

    // update local saturation history with THIS committed action
    this.satHistory.push(norm2(clipped) >= 0.99 * actionMax ? 1 : 0);
    while (this.satHistory.length > this.historyWindow) this.satHistory.shift();

    return { action: clipped, roleWeights: { field: weights[0], reward: weights[1], guard: weights[2] } };
  }
}

// Monolith baseline wrapper around an existing scalar-mixture policy.
class MonolithController {
  constructor(policy, label) {
    this.policy = new JsonPolicyController(policy);
    this.label = label;
  }
  reset() {
    return this;
  }
  act(observation, cfg) {
    return { action: this.policy.act(observation, cfg).action, roleWeights: null };
  }
}

// ---------------------------------------------------------------------------
// Trial runner
// ---------------------------------------------------------------------------
function runTrial({ controller, seed, cellId, horizon, sovThreshold, breachFrac }) {
  const cfg = makeTrialConfig({
    seed,
    sensorTier: SENSOR_TIERS.LOCAL_PROBE_FIELD,
    config: { horizon },
  });
  const env = new ShadowFieldEnv(cfg);
  const probe = buildProbeForCell(cellId, seed);
  if (probe) env.applyProbe(probe);
  controller.reset();

  const stepRows = [];
  let observation = env.lastObservation;
  let maxWeightSum = 0;
  let breachSteps = 0;
  let nSteps = 0;
  const wAccum = { field: 0, reward: 0, guard: 0 };

  while (!env.terminalOutcome) {
    const decision = controller.act(observation, cfg);
    const rw = decision.roleWeights;
    if (rw) {
      const maxRole = rw.field >= rw.reward && rw.field >= rw.guard ? "field" : rw.reward >= rw.guard ? "reward" : "guard";
      const maxW = Math.max(rw.field, rw.reward, rw.guard);
      maxWeightSum += maxW;
      if (maxW > sovThreshold) breachSteps += 1;
      wAccum.field += rw.field;
      wAccum.reward += rw.reward;
      wAccum.guard += rw.guard;
      stepRows.push({
        controller: controller.label,
        cell: cellId,
        seed,
        t: env.stepIndex,
        w_field: roundNumber(rw.field, 5),
        w_reward: roundNumber(rw.reward, 5),
        w_guard: roundNumber(rw.guard, 5),
        max_role: maxRole,
        max_role_weight: roundNumber(maxW, 5),
      });
      nSteps += 1;
    }
    const result = env.step(decision.action);
    observation = result.observation;
  }

  const m = env.metrics();
  const breachStepsFrac = nSteps ? breachSteps / nSteps : 0;
  return {
    stepRows,
    summary: {
      controller: controller.label,
      cell: cellId,
      seed,
      outcome: m.terminalOutcome,
      steps: m.steps,
      terminal_alignment: roundNumber(m.terminalAlignment, 5),
      terminal_distance: roundNumber(m.terminalDistance, 5),
      saturation_count: m.saturationCount,
      sovereignty_index: nSteps ? roundNumber(maxWeightSum / nSteps, 5) : "",
      breach_steps_frac: nSteps ? roundNumber(breachStepsFrac, 5) : "",
      sovereignty_breach: nSteps ? (breachStepsFrac > breachFrac ? 1 : 0) : "",
      mean_w_field: nSteps ? roundNumber(wAccum.field / nSteps, 5) : "",
      mean_w_reward: nSteps ? roundNumber(wAccum.reward / nSteps, 5) : "",
      mean_w_guard: nSteps ? roundNumber(wAccum.guard / nSteps, 5) : "",
    },
  };
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------
function toCsv(rows, columns) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const t = String(v);
    return /[",\n]/.test(t) ? `"${t.replaceAll('"', '""')}"` : t;
  };
  return `${columns.join(",")}\n${rows.map((r) => columns.map((c) => esc(r[c])).join(",")).join("\n")}\n`;
}

function mean(xs) {
  const f = xs.filter((x) => Number.isFinite(x));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cells = args.cells.split(",").map((c) => c.trim()).filter(Boolean);
  const outDir = path.resolve(repoRoot, args.out);

  const fieldPolicy = loadPolicy(args.fieldPolicy);
  const rewardPolicy = loadPolicy(args.rewardPolicy);
  const monolithPolicy = loadPolicy(args.monolithPolicy);

  const council = new PCouncilController({ fieldPolicy, rewardPolicy, roleHardCap: args.roleHardCap });
  const monolith = new MonolithController(monolithPolicy, "M-Scalar");
  const controllers = [council, monolith];

  const allStepRows = [];
  const allSummaries = [];
  const t0 = Date.now();

  for (const controller of controllers) {
    for (const cellId of cells) {
      for (let i = 0; i < args.seeds; i += 1) {
        const seed = args.seedStart + i;
        const { stepRows, summary } = runTrial({
          controller,
          seed,
          cellId,
          horizon: args.horizon,
          sovThreshold: args.sovereigntyThreshold,
          breachFrac: args.breachFrac,
        });
        allStepRows.push(...stepRows);
        allSummaries.push(summary);
      }
    }
  }

  // cell map: per (controller, cell) aggregate
  const cellMap = [];
  for (const controller of controllers) {
    for (const cellId of cells) {
      const sub = allSummaries.filter((s) => s.controller === controller.label && s.cell === cellId);
      const successes = sub.filter((s) => s.outcome === "success").length;
      const sovVals = sub.map((s) => s.sovereignty_index).filter((x) => Number.isFinite(x));
      const breachTrials = sub.filter((s) => s.sovereignty_breach === 1).length;
      cellMap.push({
        controller: controller.label,
        cell: cellId,
        n_seeds: sub.length,
        success_rate: roundNumber(successes / sub.length, 4),
        mean_terminal_alignment: roundNumber(mean(sub.map((s) => s.terminal_alignment)), 5),
        mean_sovereignty_index: sovVals.length ? roundNumber(mean(sovVals), 5) : "",
        breach_trial_frac: sovVals.length ? roundNumber(breachTrials / sub.length, 4) : "",
      });
    }
  }

  await mkdir(outDir, { recursive: true });
  await writeFile(
    path.join(outDir, "role_weights.csv"),
    toCsv(allStepRows, ["controller", "cell", "seed", "t", "w_field", "w_reward", "w_guard", "max_role", "max_role_weight"]),
    "utf8",
  );
  await writeFile(
    path.join(outDir, "sovereignty-summary.csv"),
    toCsv(allSummaries, [
      "controller", "cell", "seed", "outcome", "steps", "terminal_alignment", "terminal_distance",
      "saturation_count", "sovereignty_index", "breach_steps_frac", "sovereignty_breach",
      "mean_w_field", "mean_w_reward", "mean_w_guard",
    ]),
    "utf8",
  );
  await writeFile(
    path.join(outDir, "h1-cell-map.csv"),
    toCsv(cellMap, ["controller", "cell", "n_seeds", "success_rate", "mean_terminal_alignment", "mean_sovereignty_index", "breach_trial_frac"]),
    "utf8",
  );
  await writeFile(
    path.join(outDir, "h1-smoke-manifest.json"),
    `${JSON.stringify({
      scope: "H1.1 harness smoke -- TOOLING ONLY, not H1 evidence",
      generated: new Date().toISOString(),
      args,
      cells,
      seeds: Array.from({ length: args.seeds }, (_, i) => args.seedStart + i),
      heads: { field: args.fieldPolicy, reward: args.rewardPolicy, guard: "heuristic-brake-stub (untrained)", monolith: args.monolithPolicy },
      arbiter: "confidence-gated, hard-cap=" + args.roleHardCap + " (untrained placeholder)",
      cap_invariant_holds: allStepRows.every((r) => r.max_role_weight <= args.roleHardCap + 1e-9),
    }, null, 2)}\n`,
    "utf8",
  );

  const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
  const capOk = allStepRows.every((r) => r.max_role_weight <= args.roleHardCap + 1e-9);
  // eslint-disable-next-line no-console
  console.log(`H1.1 smoke: ${controllers.length} controllers x ${cells.length} cells x ${args.seeds} seeds`);
  console.log(`  step rows=${allStepRows.length}  trials=${allSummaries.length}  elapsed=${elapsed}s  cap<=${args.roleHardCap} holds=${capOk}`);
  console.log("  --- h1-cell-map ---");
  for (const row of cellMap) {
    console.log(
      `  ${row.controller.padEnd(10)} ${row.cell.padEnd(16)} ` +
      `succ=${(row.success_rate * 100).toFixed(0).padStart(3)}%  ` +
      `S_T=${String(row.mean_terminal_alignment).padEnd(7)}  ` +
      `sov=${String(row.mean_sovereignty_index).padEnd(7)}  ` +
      `breach_trials=${row.breach_trial_frac}`,
    );
  }
  if (!capOk) throw new Error("authority cap invariant violated -- arbiter bug");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
