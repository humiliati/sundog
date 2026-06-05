// Scratch: v4 Path-A verify-first PLUMBING + DECOUPLING check (no training).
// Confirms: (1) obs grows 6->8 with basinChannel; (2) the basin-observation
// channel changes the observed-basin features but NOT reward.false_basin, and the
// legacy reward-only basin-position channel changes reward.false_basin but NOT the
// observed-basin features; (3) an obs-reading policy's action therefore moves under
// basin-observation but is invariant to basin-position (which was structurally inert).
import { ShadowFieldEnv } from "../public/js/mesa-core.mjs";

const len = (basinChannel) => new ShadowFieldEnv({ basinChannel, seed: 7 }).lastObservation.observation.length;
console.log(`obs length: basinChannel=false -> ${len(false)} ; basinChannel=true -> ${len(true)}`);

// Apply one channel edit at step 0 via the real intervention path, read obs + reward.
function applyEdit(channel, edit) {
  const env = new ShadowFieldEnv({ basinChannel: true, seed: 7 });
  const before = { obs: env.observe().observation.slice(), false_basin: env.rewardChannels().false_basin };
  env.scheduleIntervention({ step: env.stepIndex, channel, edit });
  env.applyScheduledInterventions();
  const after = { obs: env.observe().observation.slice(), false_basin: env.rewardChannels().false_basin };
  const basinFeat = (o) => [o.obs[6], o.obs[7]];
  return {
    basin_obs_before: basinFeat(before), basin_obs_after: basinFeat(after),
    basin_obs_changed: basinFeat(before).some((v, i) => v !== basinFeat(after)[i]),
    false_basin_before: before.false_basin, false_basin_after: after.false_basin,
    false_basin_changed: before.false_basin !== after.false_basin,
  };
}

console.log("\n=== basin-observation channel (edits observedBasinCenter) ===");
const bo = applyEdit("basin-observation", { xObservedNew: [2.5, 2.5] });
console.log(JSON.stringify(bo, null, 0));

console.log("\n=== legacy basin-position channel (edits reward-only falseBasinCenter) ===");
const bp = applyEdit("basin-position", { xFalseNew: [2.5, 2.5] });
console.log(JSON.stringify(bp, null, 0));

// Action-visibility for an obs-reading policy: action = 0.1 * basin features.
const act = (basinFeat) => [0.1 * basinFeat[0], 0.1 * basinFeat[1]];
const l2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const boAct = l2(act(bo.basin_obs_before), act(bo.basin_obs_after));
const bpAct = l2(act(bp.basin_obs_before), act(bp.basin_obs_after));
console.log("\n=== action divergence for an obs-reading policy (action = 0.1*basinfeat) ===");
console.log(`basin-observation edit -> |dA| = ${boAct.toExponential(3)}  (action-VISIBLE)`);
console.log(`basin-position  edit -> |dA| = ${bpAct.toExponential(3)}  (inert, as in falsified v4)`);

console.log("\n=== VERDICT ===");
const decoupled = bo.basin_obs_changed && !bo.false_basin_changed && bp.false_basin_changed && !bp.basin_obs_changed;
console.log(`decoupling clean: ${decoupled}`);
console.log(`basin-observation action-visible (>0) AND basin-position inert (==0): ${boAct > 0 && bpAct === 0}`);
