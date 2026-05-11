import readline from "node:readline";
import {
  SENSOR_TIERS,
  ShadowFieldEnv,
  defaultTierParams,
  normalizeMesaConfig,
} from "../public/js/mesa-core.mjs";

const PROTOCOL_VERSION = "mesa-env-bridge-v1";
const envRecords = new Map();
const batchRecords = new Map();

function makeConfig({ seed = 1, sensor_tier: sensorTier = SENSOR_TIERS.LOCAL_PROBE_FIELD, env_config: envConfig = {} }) {
  const tierDefaults = defaultTierParams(sensorTier, {
    delaySteps: envConfig.delaySteps,
    noiseStd: envConfig.noiseStd,
  });
  return normalizeMesaConfig({
    ...envConfig,
    ...tierDefaults,
    seed,
    sensorTier,
  });
}

function buildEnv(record) {
  const env = new ShadowFieldEnv(record.config);
  if (record.probe) env.applyProbe(record.probe);
  for (const intervention of record.interventions) env.scheduleIntervention(intervention);
  return env;
}

function makeRecord(message) {
  const record = {
    seed: message.seed ?? 1,
    sensorTier: message.sensor_tier ?? SENSOR_TIERS.LOCAL_PROBE_FIELD,
    envConfig: message.env_config ?? {},
    probe: message.probe ?? null,
    interventions: Array.isArray(message.interventions) ? message.interventions : [],
    config: makeConfig(message),
    needsAutoReset: false,
  };
  record.env = buildEnv(record);
  return record;
}

function asInfo(env, extra = {}) {
  return {
    seed: env.config.seed,
    sensor_tier: env.config.sensorTier,
    step_index: env.stepIndex,
    terminal_outcome: env.terminalOutcome,
    position: env.x.slice(),
    true_signature: env.lastObservation.trueSignature,
    s_local: env.lastObservation.sLocal,
    metrics: env.terminalOutcome ? env.metrics() : undefined,
    ...extra,
  };
}

function resetRecord(record, extraInfo = {}) {
  record.env = buildEnv(record);
  record.needsAutoReset = false;
  const obs = record.env.lastObservation.observation;
  return {
    obs,
    reward_channels: record.env.rewardChannels(),
    done: false,
    info: asInfo(record.env, extraInfo),
  };
}

function stepRecord(record, action) {
  const result = record.env.step(action);
  if (result.done) record.needsAutoReset = true;
  return {
    obs: result.observation.observation,
    reward_channels: result.rewardChannels,
    done: result.done,
    info: asInfo(record.env, {
      action: result.action,
      intervention_flags: result.interventionFlags,
    }),
  };
}

function stepRecordWithImmediateReset(record, action) {
  const payload = stepRecord(record, action);
  if (!payload.done) return payload;
  const terminalObservation = payload.obs;
  const terminalInfo = payload.info;
  const terminalRewardChannels = payload.reward_channels;
  const resetPayload = resetRecord(record);
  return {
    obs: resetPayload.obs,
    reward_channels: terminalRewardChannels,
    done: true,
    info: {
      ...terminalInfo,
      auto_reset: true,
      terminal_observation: terminalObservation,
      reset_observation: resetPayload.obs,
    },
  };
}

function requireEnv(envId) {
  const record = envRecords.get(envId);
  if (!record) throw new Error(`Unknown env_id: ${envId}`);
  return record;
}

function requireBatch(batchId) {
  const records = batchRecords.get(batchId);
  if (!records) throw new Error(`Unknown batch_id: ${batchId}`);
  return records;
}

function batchPayload(records, singlePayloads) {
  return {
    obs: singlePayloads.map((payload) => payload.obs),
    reward_channels: singlePayloads.map((payload) => payload.reward_channels),
    done: singlePayloads.map((payload) => payload.done),
    info: singlePayloads.map((payload) => payload.info),
    count: records.length,
  };
}

function handle(message) {
  if (message.cmd === "ping") {
    return { protocol_version: PROTOCOL_VERSION };
  }

  if (message.cmd === "make") {
    const envId = message.env_id;
    if (!envId) throw new Error("make requires env_id");
    const record = makeRecord(message);
    envRecords.set(envId, record);
    return { env_id: envId, ...resetRecord(record) };
  }

  if (message.cmd === "reset") {
    return { env_id: message.env_id, ...resetRecord(requireEnv(message.env_id)) };
  }

  if (message.cmd === "step") {
    if (!Array.isArray(message.action)) throw new Error("step requires action array");
    return { env_id: message.env_id, ...stepRecord(requireEnv(message.env_id), message.action) };
  }

  if (message.cmd === "make_batch") {
    const batchId = message.batch_id;
    if (!batchId) throw new Error("make_batch requires batch_id");
    const count = Number.parseInt(message.count, 10);
    if (!Number.isInteger(count) || count < 1) throw new Error("make_batch count must be a positive integer");
    const seedStart = Number.parseInt(message.seed_start ?? 0, 10);
    const records = [];
    for (let index = 0; index < count; index += 1) {
      records.push(makeRecord({
        ...message,
        seed: seedStart + index,
      }));
    }
    batchRecords.set(batchId, records);
    return {
      batch_id: batchId,
      ...batchPayload(records, records.map((record) => resetRecord(record))),
    };
  }

  if (message.cmd === "reset_batch") {
    const records = requireBatch(message.batch_id);
    return {
      batch_id: message.batch_id,
      ...batchPayload(records, records.map((record) => resetRecord(record))),
    };
  }

  if (message.cmd === "step_batch") {
    const records = requireBatch(message.batch_id);
    if (!Array.isArray(message.actions) || message.actions.length !== records.length) {
      throw new Error(`step_batch requires ${records.length} actions`);
    }
    return {
      batch_id: message.batch_id,
      ...batchPayload(
        records,
        records.map((record, index) => {
          if (record.needsAutoReset) {
            return resetRecord(record, {
              auto_reset: true,
              ignored_action: message.actions[index],
            });
          }
          if (message.auto_reset_done) {
            return stepRecordWithImmediateReset(record, message.actions[index]);
          }
          return stepRecord(record, message.actions[index]);
        }),
      ),
    };
  }

  if (message.cmd === "close") {
    envRecords.clear();
    batchRecords.clear();
    return { closed: true };
  }

  throw new Error(`Unknown command: ${message.cmd}`);
}

function writeResponse(response) {
  process.stdout.write(`${JSON.stringify(response)}\n`);
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", (line) => {
  if (!line.trim()) return;
  let id = null;
  try {
    const message = JSON.parse(line);
    id = message.id ?? null;
    const payload = handle(message);
    writeResponse({ ok: true, id, ...payload });
    if (message.cmd === "close") {
      rl.close();
      process.exitCode = 0;
    }
  } catch (error) {
    writeResponse({
      ok: false,
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
