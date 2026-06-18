// Shared role-weight cap geometry for the H1 pantheon arbiter.
//
// Used by BOTH the dataset builder (to project the privileged best-mix TARGET)
// and the eval harness (to project the arbiter's softmax weights at INFERENCE).
// Factoring it here guarantees target-construction and inference use byte-
// identical cap geometry -- a divergence would make the bake-off VOID.
//
// H1.2 used a symmetric cap (0.70 on all roles). H1.2c
// (H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md) makes the cap reward-asymmetric: bind
// the bull, leave Sol uncapped -- field 1.00 / reward 0.50 / guard 0.70.

// Role order is fixed: [field, reward, guard].
export function resolveCaps(capMode, fieldCap, rewardCap, guardCap, symCap = 0.7) {
  if (capMode === "reward-asymmetric") return [fieldCap, rewardCap, guardCap];
  if (capMode === "symmetric" || capMode === undefined || capMode === null) {
    return [symCap, symCap, symCap];
  }
  throw new Error(`Unknown cap-mode: ${capMode}`);
}

// True capped-simplex projection (H1.2c spec §3.2):
//   repeat: freeze any role over its cap at the cap; redistribute the remaining
//   mass among unfrozen roles in proportion to their RAW weights (not the
//   already-clipped current weights), so a capped role can never regain mass.
// Returns weights w with w_i <= caps_i and sum_i w_i = 1 (caps assumed feasible,
// i.e. sum(caps) >= 1).
export function capSimplexProject(raw, caps) {
  const n = raw.length;
  const rawNN = raw.map((v) => Math.max(0, v));
  const rawSumAll = rawNN.reduce((a, b) => a + b, 0);
  let w = rawSumAll > 1e-12 ? rawNN.map((v) => v / rawSumAll) : caps.map(() => 1 / n);
  const frozen = new Array(n).fill(false);
  for (let iter = 0; iter < n; iter += 1) {
    const over = [];
    for (let i = 0; i < n; i += 1) if (!frozen[i] && w[i] > caps[i] + 1e-12) over.push(i);
    if (over.length === 0) break;
    for (const i of over) {
      w[i] = caps[i];
      frozen[i] = true;
    }
    const frozenMass = frozen.reduce((a, f, i) => (f ? a + caps[i] : a), 0);
    const remaining = Math.max(0, 1 - frozenMass);
    const unfrozen = [];
    for (let i = 0; i < n; i += 1) if (!frozen[i]) unfrozen.push(i);
    const rawSum = unfrozen.reduce((a, i) => a + rawNN[i], 0);
    if (unfrozen.length === 0) break;
    if (rawSum > 1e-12) for (const i of unfrozen) w[i] = remaining * (rawNN[i] / rawSum);
    else for (const i of unfrozen) w[i] = remaining / unfrozen.length;
  }
  return w;
}
