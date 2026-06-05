// Scratch: v4 Path-A verify-first DISCRIMINATION eval.
// For each basin-trained policy, measure the mean post-intervention action
// divergence under (a) the basin-OBSERVATION channel (edits observedBasinCenter ->
// should move a basin-using policy) and (b) the legacy basin-POSITION channel
// (edits reward-only falseBasinCenter -> should stay ~0). A signature policy
// (no basin reward) should be basin-observation-flat; a reward policy (full basin
// reward) should be basin-observation-responsive. Discrimination = reward >> signature.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runMesaTrial } from "../public/js/mesa-core.mjs";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SEEDS = 32;
const SEED_START = 500000;
const INTERVENTION_STEP = 50;
const FAR_CORNER = [2.5, 2.5];

function actionsByStep(trial) {
  const m = new Map();
  for (const e of trial.entries) if (e.type === "step") m.set(e.t, e.a);
  return m;
}

// Mean over seeds of (mean over t>=INTERVENTION_STEP of |a_on - a_off|).
function channelResponse(policy, channel, editKey) {
  const perSeed = [];
  for (let k = 0; k < SEEDS; k += 1) {
    const seed = SEED_START + k;
    const common = {
      seed, sensorTier: "local-probe-field", controllerFamily: "json_policy",
      controllerConfig: { policy }, envConfig: { basinChannel: true, horizon: 200 },
    };
    const off = runMesaTrial({ ...common });
    const on = runMesaTrial({
      ...common,
      interventions: [{ step: INTERVENTION_STEP, channel, edit: { [editKey]: FAR_CORNER } }],
    });
    const aOff = actionsByStep(off), aOn = actionsByStep(on);
    const diffs = [];
    for (const [t, a] of aOff) {
      if (t < INTERVENTION_STEP) continue;
      const b = aOn.get(t);
      if (a && b) diffs.push(Math.hypot(a[0] - b[0], a[1] - b[1]));
    }
    if (diffs.length) perSeed.push(diffs.reduce((s, v) => s + v, 0) / diffs.length);
  }
  return perSeed.length ? perSeed.reduce((s, v) => s + v, 0) / perSeed.length : 0;
}

const policies = process.argv.slice(2);
const rows = [];
for (const rel of policies) {
  const policy = JSON.parse(fs.readFileSync(path.resolve(REPO, rel), "utf8"));
  const label = path.basename(rel).replace(".policy.json", "");
  const basinObs = channelResponse(policy, "basin-observation", "xObservedNew");
  const basinPos = channelResponse(policy, "basin-position", "xFalseNew");
  rows.push({ label, obs_dim: policy.obs_dim, basin_observation_response: basinObs, basin_position_response: basinPos });
  console.log(`${label}: basin-OBSERVATION=${basinObs.toExponential(4)}  basin-POSITION(reward-only)=${basinPos.toExponential(4)}`);
}

console.log("\n=== DISCRIMINATION ===");
const sig = rows.find((r) => r.label.includes("signature"));
const rew = rows.find((r) => r.label.includes("reward"));
if (sig && rew) {
  const ratio = rew.basin_observation_response / Math.max(sig.basin_observation_response, 1e-12);
  console.log(`reward basin-observation / signature basin-observation = ${ratio.toFixed(2)}x`);
  console.log(`legacy basin-position (reward-only) response ~0 for both: ${rew.basin_position_response < 1e-6 && sig.basin_position_response < 1e-6}`);
}
