import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_HC_SIGNATURE_CONFIG,
  DEFAULT_MESA_CONFIG,
  DEFAULT_ORACLE_CONFIG,
  SENSOR_TIERS,
  defaultControllerConfig,
  defaultTierParams,
  roundNumber,
  runMesaTrial,
  serializeJsonl,
} from "../public/js/mesa-core.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DEFAULT_PHASE1_TIERS = Object.freeze([
  SENSOR_TIERS.PRIVILEGED_FIELD,
  SENSOR_TIERS.LOCAL_PROBE_FIELD,
  SENSOR_TIERS.DELAYED_FIELD,
  SENSOR_TIERS.NOISY_FIELD,
]);

const CSV_COLUMNS = Object.freeze([
  "phase",
  "trialId",
  "controllerFamily",
  "sensorTier",
  "seed",
  "configHash",
  "terminalOutcome",
  "steps",
  "regimeRetention",
  "terminalAlignment",
  "terminalDistance",
  "pathEfficiency",
  "timeToSuccess",
  "saturationCount",
  "clippedCount",
  "pathLength",
  "totalActionMagnitude",
  "trialPath",
]);

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    phase: "phase1-hc-baseline",
    out: "results/mesa/phase1-hc-baseline",
    seedStart: 0,
    seeds: 32,
    sensorTiers: [...DEFAULT_PHASE1_TIERS],
    controllerFamilies: ["hc_signature", "oracle"],
    horizon: DEFAULT_MESA_CONFIG.horizon,
    logEvery: 1,
    delaySteps: undefined,
    noiseStd: undefined,
    verifyReplay: true,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--seed-start") args.seedStart = Number.parseInt(value, 10);
    else if (flag === "--seeds") args.seeds = Number.parseInt(value, 10);
    else if (flag === "--sensor-tiers") args.sensorTiers = parseList(value);
    else if (flag === "--controller-families") args.controllerFamilies = parseList(value);
    else if (flag === "--horizon") args.horizon = Number.parseInt(value, 10);
    else if (flag === "--log-every") args.logEvery = Number.parseInt(value, 10);
    else if (flag === "--delay-steps") args.delaySteps = Number.parseInt(value, 10);
    else if (flag === "--noise-std") args.noiseStd = Number.parseFloat(value);
    else if (flag === "--verify-replay") args.verifyReplay = value !== "false";
    else throw new Error(`Unknown flag: ${flag}`);
  }

  const validTiers = new Set(Object.values(SENSOR_TIERS));
  for (const tier of args.sensorTiers) {
    if (!validTiers.has(tier)) throw new Error(`Unknown sensor tier: ${tier}`);
  }
  const validFamilies = new Set(["hc_signature", "oracle"]);
  for (const family of args.controllerFamilies) {
    if (!validFamilies.has(family)) throw new Error(`Unknown controller family: ${family}`);
  }
  if (!Number.isInteger(args.seedStart) || args.seedStart < 0) {
    throw new Error("--seed-start must be a non-negative integer");
  }
  if (!Number.isInteger(args.seeds) || args.seeds < 1) {
    throw new Error("--seeds must be a positive integer");
  }
  if (!Number.isInteger(args.horizon) || args.horizon < 1) {
    throw new Error("--horizon must be a positive integer");
  }
  if (!Number.isInteger(args.logEvery) || args.logEvery < 1) {
    throw new Error("--log-every must be a positive integer");
  }
  if (args.delaySteps !== undefined && (!Number.isInteger(args.delaySteps) || args.delaySteps < 0)) {
    throw new Error("--delay-steps must be a non-negative integer");
  }
  if (args.noiseStd !== undefined && (!Number.isFinite(args.noiseStd) || args.noiseStd < 0)) {
    throw new Error("--noise-std must be non-negative");
  }
  return args;
}

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: repoRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(roundNumber(value, 8)) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return `${columns.join(",")}\n${rows
    .map((row) => columns.map((column) => csvValue(row[column])).join(","))
    .join("\n")}\n`;
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function summarize(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.controllerFamily}\t${row.sensorTier}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return Array.from(groups.entries()).map(([key, group]) => {
    const [controllerFamily, sensorTier] = key.split("\t");
    const successes = group.filter((row) => row.terminalOutcome === "success").length;
    return {
      controllerFamily,
      sensorTier,
      n: group.length,
      successCount: successes,
      successRate: successes / group.length,
      meanSteps: mean(group.map((row) => row.steps)),
      meanRegimeRetention: mean(group.map((row) => row.regimeRetention)),
      meanTerminalAlignment: mean(group.map((row) => row.terminalAlignment)),
      meanTerminalDistance: mean(group.map((row) => row.terminalDistance)),
      meanSaturationCount: mean(group.map((row) => row.saturationCount)),
    };
  });
}

function tiersForFamily(args, family) {
  if (family === "oracle") {
    return args.sensorTiers.filter((tier) => tier === SENSOR_TIERS.PRIVILEGED_FIELD);
  }
  return args.sensorTiers;
}

function tierOverrides(args, sensorTier) {
  const defaults = defaultTierParams(sensorTier);
  return {
    delaySteps: args.delaySteps ?? defaults.delaySteps,
    noiseStd: args.noiseStd ?? defaults.noiseStd,
  };
}

function trialFileName(trial) {
  return `${trial.trialId}-${trial.configHash}.jsonl`;
}

function verifyReplay(trial, replayTrial) {
  const original = serializeJsonl(trial.entries);
  const replay = serializeJsonl(replayTrial.entries);
  if (original !== replay) {
    throw new Error(`Replay mismatch for ${trial.trialId}`);
  }
}

// PHASE1_PRIME_SPEC v1.2 §6: declarative per-tier observation channel
// metadata. Audit script LT1 cross-checks this against the actual
// observe() implementation in mesa-core.mjs. Channel counts here are
// the documented sensor-tier channel counts; channel_count for the
// learned-policy tier (local-probe-field) must match
// MesaMlpPolicy.config.obs_dim = 6.
const SENSOR_TIER_OBSERVATION_SPEC = Object.freeze({
  "privileged-field": Object.freeze({
    channels: ["pos_x", "pos_y", "xGoal_x", "xGoal_y", "trueSignature", "trueGrad_x", "trueGrad_y"],
    channel_count: 7,
    note: "Oracle tier; intentionally exposes x_goal and trueGradient. LT1 carves this tier out — no learned/HC-non-Oracle policy is configured against it.",
  }),
  "local-probe-field": Object.freeze({
    channels: ["pos_x", "pos_y", "probe_0", "probe_1", "probe_2", "probe_3"],
    channel_count: 6,
    note: "Canonical learned-policy tier; v1 envelope is classified at this tier. matches MesaMlpPolicy.config.obs_dim.",
  }),
  "delayed-field": Object.freeze({
    channels: ["pos_x", "pos_y", "probe_0", "probe_1", "probe_2", "probe_3"],
    channel_count: 6,
    note: "Same channels as local-probe-field, with delayed-state probe samples.",
  }),
  "noisy-field": Object.freeze({
    channels: ["pos_x", "pos_y", "probe_0", "probe_1", "probe_2", "probe_3"],
    channel_count: 6,
    note: "Same channels as local-probe-field, with Gaussian noise applied to probe samples.",
  }),
});

const AGENT_FORBIDDEN_CHANNELS = Object.freeze([
  "x_goal",
  "true_gradient",
  "x_false",
  "privileged_position",
  "reward_edit_log",
  "metrics",
]);

function buildSignatureProvenanceManifest({ args, manifestPath, repoRoot: root }) {
  const v1Rel = path.relative(root, manifestPath).replaceAll("\\", "/");
  const sensorTierBlocks = args.sensorTiers.map((tier) => {
    const overrides = tierOverrides(args, tier);
    const spec = SENSOR_TIER_OBSERVATION_SPEC[tier] ?? {
      channels: ["unknown"],
      channel_count: -1,
      note: `Unrecognized sensor tier ${tier}; manifest carries placeholder metadata — audit LT1 will fail.`,
    };
    return {
      identifier: tier,
      delay_steps: overrides.delaySteps,
      noise_std: overrides.noiseStd,
      agent_observation: { ...spec },
    };
  });

  const rewardFunctions = args.controllerFamilies.map((family) => ({
    controller_family: family,
    identifier: family === "hc_signature" || family === "oracle" ? "none" : family,
    implementation_file: "public/js/mesa-core.mjs",
    implementation_method: "ShadowFieldEnv.rewardChannels",
    depends_on_env_state: ["this.x", "this.xGoal", "this.activeRewardEdit", "this.config.falseBasinCenter", "this.config.falseBasinSigma", "this.config.falseBasinBeta"],
    active_channels: family === "hc_signature" || family === "oracle"
      ? []
      : ["dense", "phase3_dense_action_basin", "sparse", "signature"],
    note: family === "hc_signature" || family === "oracle"
      ? "HC controller; no learned reward path used. rewardChannels() is still computed by the env, but is consumed for nothing in this phase."
      : "Learned policy variant; reward channels are combined into a scalar reward during training. The labeled `signature` channel is consumed as a reward component only, never as observation input (verified by LT3 Path B).",
  }));

  return {
    phase_v2: "phase1-prime-signature-path",
    schema_version: "v1",
    emitted_at: new Date().toISOString(),
    git_sha: gitSha(),
    v1_manifest_path: v1Rel,

    x_goal: {
      privileged: true,
      hidden_from_agent: true,
      sampled_per_seed_from: "ShadowFieldEnv.initializeMesaState; deterministic on seed",
      note: "Per-seed x_goal values are not enumerated here to keep the manifest stable across seed-count changes; they're reproducible from the seed and the env constructor. The audit script reads them from the v1 trial JSONLs if a per-seed cross-check is needed.",
    },

    signature_function: {
      identifier: "gaussian_shadow_field_v1",
      implementation_file: "public/js/mesa-core.mjs",
      implementation_method: "ShadowFieldEnv.trueSignature",
      depends_on_env_state: ["this.x", "this.xGoal", "this.config.sigmaS"],
      depends_on_agent_state: [],
      note: "this.xGoal is a top-level mutable instance field (not under this.config); the geometry intervention writes it. LT2 allowlists this.xGoal as the documented shared geometry baseline.",
    },

    sensor_tiers: sensorTierBlocks,
    reward_functions: rewardFunctions,
    agent_forbidden_channels: [...AGENT_FORBIDDEN_CHANNELS],

    leakage_audit_verdict: {
      audit_script_version: "pending",
      audit_run_at: "pending",
      LT1_no_xgoal_in_obs: "pending",
      LT2_disjoint_accessors: "pending",
      LT3_no_log_feedback: "pending",
      LT4_channel_independence: "pending",
      overall: "pending",
    },
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const trialsDir = path.join(outDir, "trials");
  await mkdir(trialsDir, { recursive: true });

  const manifestPath = path.join(outDir, "manifest.json");
  const trialRows = [];
  const trialPaths = [];

  for (const controllerFamily of args.controllerFamilies) {
    for (const sensorTier of tiersForFamily(args, controllerFamily)) {
      for (let offset = 0; offset < args.seeds; offset += 1) {
        const seed = args.seedStart + offset;
        const overrides = tierOverrides(args, sensorTier);
        const envConfig = {
          horizon: args.horizon,
          logEvery: args.logEvery,
          delaySteps: overrides.delaySteps,
          noiseStd: overrides.noiseStd,
        };
        const trialId = `${controllerFamily}_${sensorTier}_seed_${String(seed).padStart(4, "0")}`;
        const controllerConfig = defaultControllerConfig(controllerFamily);
        const trial = runMesaTrial({
          seed,
          sensorTier,
          controllerFamily,
          envConfig,
          controllerConfig,
          trialId,
          manifestPath: path.relative(repoRoot, manifestPath).replaceAll("\\", "/"),
          logEvery: args.logEvery,
        });
        if (args.verifyReplay) {
          const replay = runMesaTrial({
            seed,
            sensorTier,
            controllerFamily,
            envConfig,
            controllerConfig,
            trialId,
            manifestPath: path.relative(repoRoot, manifestPath).replaceAll("\\", "/"),
            logEvery: args.logEvery,
          });
          verifyReplay(trial, replay);
        }

        const trialPath = path.join(trialsDir, trialFileName(trial));
        await writeFile(trialPath, serializeJsonl(trial.entries), "utf8");
        const relativeTrialPath = path.relative(repoRoot, trialPath).replaceAll("\\", "/");
        trialPaths.push(relativeTrialPath);
        trialRows.push({
          phase: args.phase,
          trialId: trial.trialId,
          controllerFamily,
          sensorTier,
          seed,
          configHash: trial.configHash,
          terminalOutcome: trial.summary.terminalOutcome,
          steps: trial.summary.steps,
          regimeRetention: trial.summary.regimeRetention,
          terminalAlignment: trial.summary.terminalAlignment,
          terminalDistance: trial.summary.terminalDistance,
          pathEfficiency: trial.summary.pathEfficiency,
          timeToSuccess: trial.summary.timeToSuccess,
          saturationCount: trial.summary.saturationCount,
          clippedCount: trial.summary.clippedCount,
          pathLength: trial.summary.pathLength,
          totalActionMagnitude: trial.summary.totalActionMagnitude,
          trialPath: relativeTrialPath,
        });
      }
    }
  }

  const summaryRows = summarize(trialRows);
  await writeFile(path.join(outDir, "trial-outcomes.csv"), toCsv(trialRows, CSV_COLUMNS), "utf8");
  await writeFile(
    path.join(outDir, "summary.csv"),
    toCsv(summaryRows, [
      "controllerFamily",
      "sensorTier",
      "n",
      "successCount",
      "successRate",
      "meanSteps",
      "meanRegimeRetention",
      "meanTerminalAlignment",
      "meanTerminalDistance",
      "meanSaturationCount",
    ]),
    "utf8",
  );

  const manifest = {
    phase: args.phase,
    git_sha: gitSha(),
    created_at: new Date().toISOString(),
    seed_base: args.seedStart,
    env: {
      name: DEFAULT_MESA_CONFIG.name,
      version: DEFAULT_MESA_CONFIG.version,
      L: DEFAULT_MESA_CONFIG.arenaHalfWidth,
      dt: DEFAULT_MESA_CONFIG.dt,
      sigma_S: DEFAULT_MESA_CONFIG.sigmaS,
      sigma_dyn: DEFAULT_MESA_CONFIG.sigmaDyn,
      T_max: args.horizon,
      delta: DEFAULT_MESA_CONFIG.delta,
      delta_regime: DEFAULT_MESA_CONFIG.deltaRegime,
      K_success: DEFAULT_MESA_CONFIG.kSuccess,
    },
    sensor_tiers: args.sensorTiers,
    tier_params: Object.fromEntries(args.sensorTiers.map((tier) => [tier, tierOverrides(args, tier)])),
    controller: {
      families: args.controllerFamilies,
      configs: {
        hc_signature: DEFAULT_HC_SIGNATURE_CONFIG,
        oracle: DEFAULT_ORACLE_CONFIG,
      },
    },
    capacity_tier: null,
    selection_pressure: null,
    probe_slate: null,
    intervention_slate: null,
    trial_count: trialRows.length,
    trial_paths: trialPaths,
    outputs: {
      manifest: path.relative(repoRoot, manifestPath).replaceAll("\\", "/"),
      trial_outcomes: path.relative(repoRoot, path.join(outDir, "trial-outcomes.csv")).replaceAll("\\", "/"),
      summary: path.relative(repoRoot, path.join(outDir, "summary.csv")).replaceAll("\\", "/"),
    },
    replay_verified: args.verifyReplay,
    summary: Object.fromEntries(summaryRows.map((row) => [`${row.controllerFamily}/${row.sensorTier}`, row])),
  };
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  // PHASE1_PRIME_SPEC v1.2 §6: emit a v2 signature-provenance manifest
  // alongside the v1 manifest. Additive; v1 manifest is unchanged. The
  // `leakage_audit_verdict` block starts as `pending` and is populated
  // by `scripts/mesa-signature-provenance-audit.mjs` in a second pass.
  const v2ManifestPath = path.join(outDir, "signature-provenance-manifest.json");
  const v2Manifest = buildSignatureProvenanceManifest({ args, manifestPath, repoRoot });
  await writeFile(v2ManifestPath, `${JSON.stringify(v2Manifest, null, 2)}\n`, "utf8");

  for (const row of summaryRows) {
    console.log(
      `${row.controllerFamily}/${row.sensorTier}: ${row.successCount}/${row.n} success, mean S_T=${roundNumber(row.meanTerminalAlignment, 4)}, mean steps=${roundNumber(row.meanSteps, 2)}`,
    );
  }
  console.log(`Wrote ${path.relative(repoRoot, outDir).replaceAll("\\", "/")}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
