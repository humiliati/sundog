// Shared H1 local/trust feature assembly for dataset builder and eval.
//
// H1.2f adds six K=8 temporal trust features to the original 17 instantaneous
// local features. This module intentionally reads only the agent-visible local
// stream: observation samples, frozen head proposals, and previous committed
// actions.

export const BASE_H1_FEATURES = [
  "obs0", "obs1", "obs2", "obs3", "obs4", "obs5",
  "fa_x", "fa_y", "ra_x", "ra_y", "fa_norm", "ra_norm",
  "disagree_l2", "cos_agree", "fd_grad_norm", "hist_act_norm_prev", "hist_sLocal_prev",
];

export const TRUST_K = 8;

export const TRUST_FEATURES = [
  "sample_dispersion",
  "sLocal_var_K",
  "grad_norm_var_K",
  "grad_dir_stability_K",
  "disagree_mean_K",
  "act_dir_consistency_K",
];

export const TRUST_FEATURE_DEFS = {
  sample_dispersion: "std of the four local probe samples at the current step",
  sLocal_var_K: "zero-padded variance of sLocal over the trailing K=8 local observations, including current",
  grad_norm_var_K: "zero-padded variance of finite-difference gradient norm over the trailing K=8 observations, including current",
  grad_dir_stability_K: "mean cosine between consecutive finite-difference gradient directions in the trailing K=8 observation window",
  disagree_mean_K: "zero-padded mean field/reward proposal L2 disagreement over the trailing K=8 observations, including current",
  act_dir_consistency_K: "mean cosine between consecutive previous committed action directions in the trailing K=8 action window",
};

export function h1FeaturesForMode(featureMode = "base") {
  return featureMode === "trust" ? [...BASE_H1_FEATURES, ...TRUST_FEATURES] : BASE_H1_FEATURES.slice();
}

export function norm2(v) {
  return Math.hypot(v[0], v[1]);
}

export function cos2(a, b) {
  const na = norm2(a);
  const nb = norm2(b);
  if (na < 1e-9 || nb < 1e-9) return 0;
  return (a[0] * b[0] + a[1] * b[1]) / (na * nb);
}

function varianceZeroPadded(values, k = TRUST_K) {
  const padded = values.slice(-k);
  while (padded.length < k) padded.unshift(0);
  const m = padded.reduce((a, b) => a + b, 0) / k;
  return padded.reduce((a, b) => a + (b - m) * (b - m), 0) / k;
}

function meanZeroPadded(values, k = TRUST_K) {
  const padded = values.slice(-k);
  while (padded.length < k) padded.unshift(0);
  return padded.reduce((a, b) => a + b, 0) / k;
}

function sampleStd(values) {
  if (!values.length) return 0;
  const m = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((a, b) => a + (b - m) * (b - m), 0) / values.length);
}

function meanConsecutiveCos(vectors) {
  const xs = vectors.slice(-TRUST_K).filter((v) => norm2(v) >= 1e-9);
  if (xs.length < 2) return 0;
  const vals = [];
  for (let i = 1; i < xs.length; i += 1) vals.push(cos2(xs[i - 1], xs[i]));
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

export function makeH1FeatureState() {
  return {
    sLocal: [],
    gradNorm: [],
    gradDir: [],
    disagree: [],
    actions: [],
    prevActNorm: 0,
    prevSLocal: 0,
  };
}

export function resetH1FeatureState(state, observation) {
  state.sLocal = [];
  state.gradNorm = [];
  state.gradDir = [];
  state.disagree = [];
  state.actions = [];
  state.prevActNorm = 0;
  state.prevSLocal = observation?.sLocal ?? 0;
}

export function buildH1LocalFeatures({ observation, fa, ra, eps, state, featureMode = "base" }) {
  const obs = observation.observation ?? observation;
  const samples = observation.samples ?? obs.slice(2, 6);
  const fd = [(samples[0] - samples[1]) / (2 * eps), (samples[2] - samples[3]) / (2 * eps)];
  const fdNorm = norm2(fd);
  const disagree = Math.hypot(fa[0] - ra[0], fa[1] - ra[1]);
  const sLocal = observation.sLocal ?? samples.reduce((a, b) => a + b, 0) / Math.max(samples.length, 1);
  const prevActNorm = state?.prevActNorm ?? 0;
  const prevSLocal = state?.prevSLocal ?? sLocal;

  if (state) {
    state.sLocal.push(sLocal);
    state.gradNorm.push(fdNorm);
    state.gradDir.push(fd);
    state.disagree.push(disagree);
    while (state.sLocal.length > TRUST_K) state.sLocal.shift();
    while (state.gradNorm.length > TRUST_K) state.gradNorm.shift();
    while (state.gradDir.length > TRUST_K) state.gradDir.shift();
    while (state.disagree.length > TRUST_K) state.disagree.shift();
  }

  const f = {
    obs0: obs[0], obs1: obs[1], obs2: obs[2], obs3: obs[3], obs4: obs[4], obs5: obs[5],
    fa_x: fa[0], fa_y: fa[1], ra_x: ra[0], ra_y: ra[1],
    fa_norm: norm2(fa), ra_norm: norm2(ra),
    disagree_l2: disagree, cos_agree: cos2(fa, ra),
    fd_grad_norm: fdNorm, hist_act_norm_prev: prevActNorm, hist_sLocal_prev: prevSLocal,
  };

  if (featureMode === "trust") {
    f.sample_dispersion = sampleStd(samples);
    f.sLocal_var_K = varianceZeroPadded(state?.sLocal ?? [sLocal]);
    f.grad_norm_var_K = varianceZeroPadded(state?.gradNorm ?? [fdNorm]);
    f.grad_dir_stability_K = meanConsecutiveCos(state?.gradDir ?? [fd]);
    f.disagree_mean_K = meanZeroPadded(state?.disagree ?? [disagree]);
    f.act_dir_consistency_K = meanConsecutiveCos(state?.actions ?? []);
  }

  return f;
}

export function noteH1Action(state, action, observation) {
  if (!state) return;
  state.actions.push(action.slice());
  while (state.actions.length > TRUST_K) state.actions.shift();
  state.prevActNorm = norm2(action);
  state.prevSLocal = observation?.sLocal ?? state.prevSLocal;
}

export function trustFeatureAudit(featureMode, inferenceFeatures) {
  const featureSet = new Set(inferenceFeatures);
  const missingBase = BASE_H1_FEATURES.filter((f) => !featureSet.has(f));
  const trustPresent = TRUST_FEATURES.filter((f) => featureSet.has(f));
  const missingTrust = TRUST_FEATURES.filter((f) => !featureSet.has(f));
  return {
    feature_mode: featureMode,
    window_k: TRUST_K,
    base_feature_count: BASE_H1_FEATURES.length,
    trust_feature_count: trustPresent.length,
    inference_feature_count: inferenceFeatures.length,
    trust_features: TRUST_FEATURES,
    missing_base_features: missingBase,
    missing_trust_features: featureMode === "trust" ? missingTrust : [],
    forbidden_feature_scan: inferenceFeatures.filter((f) => /x_false|x_goal|terminal|true_|basin|cell|seed|label|metric/i.test(f)),
    no_privileged_feature_names: inferenceFeatures.every((f) => !/x_false|x_goal|terminal|true_|basin|cell|seed|label|metric/i.test(f)),
    definitions: TRUST_FEATURE_DEFS,
  };
}
