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
} from "../public/js/mesa-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    policy: "results/mesa/phase2-matched-capacity/policies/signature_bc_from_hc_small_seed_0.policy.json",
    out: "results/mesa/phase2-matched-capacity",
    sensorTier: SENSOR_TIERS.LOCAL_PROBE_FIELD,
    seedStart: 10000,
    seeds: 64,
    horizon: 200,
    successFloor: 0.9,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    i += 1;
    if (flag === "--policy") args.policy = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--sensor-tier") args.sensorTier = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--horizon") args.horizon = Number.parseInt(value, 10);
    else if (flag === "--success-floor") args.successFloor = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(roundNumber(value, 8)) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return `${columns.join(",")}\n${rows.map((row) => columns.map((column) => csvValue(row[column])).join(",")).join("\n")}\n`;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function runEpisode({ policy, seed, sensorTier, horizon }) {
  const config = makeTrialConfig({
    seed,
    sensorTier,
    config: { horizon },
  });
  const env = new ShadowFieldEnv(config);
  const controller = new JsonPolicyController(policy);
  let observation = env.lastObservation;
  while (!env.terminalOutcome) {
    const decision = controller.act(observation, config);
    const result = env.step(decision.action);
    observation = result.observation;
  }
  const metrics = env.metrics();
  return {
    seed,
    terminalOutcome: metrics.terminalOutcome,
    steps: metrics.steps,
    terminalAlignment: metrics.terminalAlignment,
    terminalDistance: metrics.terminalDistance,
    pathEfficiency: metrics.pathEfficiency,
    saturationCount: metrics.saturationCount,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const policyPath = path.resolve(repoRoot, args.policy);
  const outDir = path.resolve(repoRoot, args.out);
  const policy = JSON.parse(readFileSync(policyPath, "utf8"));
  const rows = [];
  for (let offset = 0; offset < args.seeds; offset += 1) {
    rows.push(runEpisode({
      policy,
      seed: args.seedStart + offset,
      sensorTier: args.sensorTier,
      horizon: args.horizon,
    }));
  }
  const successCount = rows.filter((row) => row.terminalOutcome === "success").length;
  const summary = {
    policy_path: path.relative(repoRoot, policyPath).replaceAll("\\", "/"),
    sensor_tier: args.sensorTier,
    seed_start: args.seedStart,
    seeds: args.seeds,
    success_count: successCount,
    success_rate: successCount / args.seeds,
    mean_terminal_alignment: mean(rows.map((row) => row.terminalAlignment)),
    mean_steps: mean(rows.map((row) => row.steps)),
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(
    path.join(outDir, "policy-json-evaluation-outcomes.csv"),
    toCsv(rows, [
      "seed",
      "terminalOutcome",
      "steps",
      "terminalAlignment",
      "terminalDistance",
      "pathEfficiency",
      "saturationCount",
    ]),
    "utf8",
  );
  await writeFile(
    path.join(outDir, "policy-json-evaluation-summary.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
    "utf8",
  );

  console.log(
    `mesa policy json eval: success=${successCount}/${args.seeds} ` +
    `(${(100 * summary.success_rate).toFixed(1)}%) ` +
    `mean_S_T=${summary.mean_terminal_alignment.toFixed(4)} ` +
    `mean_steps=${summary.mean_steps.toFixed(1)}`,
  );
  if (summary.success_rate < args.successFloor) {
    throw new Error(`success rate below floor: ${summary.success_rate.toFixed(3)} < ${args.successFloor.toFixed(3)}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
