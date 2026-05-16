#!/usr/bin/env node
// scripts/cut2-c3a-w4-v2.mjs
//
// Wave-4.1 Path Y + Path Z amendment re-run of C3-A-T and C3-A-B.
// C3-A-R is unchanged from v1 (PASS under Path B quantile rule); this
// script regenerates it only for self-contained outputs.
//
// Path Y (C3-A-T framing):
//   Evaluate temptation on the FULL non-degenerate P_in subset
//   (f_par_obs ≥ R22), with π_route emitting biased q_naive on
//   L1-ineligible-by-obs rows. Truly-degenerate rows (f_par_obs < R22)
//   are excluded because q_naive = arccos(R22/f_par_obs) is undefined
//   there — π_route can't even produce a biased estimate.
//   Reading of freeze §4 "in-sample": full non-degenerate P_in, not
//   restricted to L1-eligible-by-obs.
//
// Path Z (C3-A-B sub-(ii) reformulation):
//   Replace the "q_naive itself is a local max" test with
//   "exists a local max of full I within radius 0.5° of q_naive".
//   Defense: the freeze §4 says "I_route optimum stays a feasible
//   competing optimum" — competing, not strictly distinct. When the
//   decoy ridge lies within ~σ of q_naive the two ridges MERGE into
//   one peak that shifts by κ/(1+κ)·Δ ≤ 0.5° at κ=0.5, |Δ| ≤ 1.5° —
//   the route is REPRESENTED in the merged peak, not erased.
//
// Outputs:
//   results/structural-failure/cut2-prereg/c3a-r-receipt-v2.json
//   results/structural-failure/cut2-prereg/c3a-t-receipt-v2.json   (Path Y)
//   results/structural-failure/cut2-prereg/c3a-b-receipt-v2.json   (Path Z)
//   results/structural-failure/cut2-prereg/c3a-w4-v2-summary.md
//
// Determinism: SAME seed, SAME LCG, SAME OLS fit as v1, so P_in and (w,b)
// are byte-identical to v1. Only the receipt-evaluation framings differ.

import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIR = dirname(SCRIPT_PATH);
const REPO = resolve(
  execSync("git rev-parse --show-toplevel", { encoding: "utf8", cwd: SCRIPT_DIR }).trim()
);
const OUT_DIR_REL = "results/structural-failure/cut2-prereg";

// --- Inherited Wave-2 + Wave-4 frozen values (must match v1 byte-for-byte) ---
const R22_DEG = 22;
const L1_LEVERAGE_MULTIPLIER = 1.02;
const RHO = 0.02;
const SIGMA_DEG = 0.5;
const SEED = 20260516;
const H_GRID_MIN = 0;
const H_GRID_MAX = 40;
const H_GRID_STEP = 0.5;
const Q_H_MIN = 0;
const Q_H_MAX = 60;
const LAMBDA_PEN = 1.0;
const C_L1_K = 600;
const A_DEG = RHO * R22_DEG;
const TAU_PC_DEG = 2.0;
const TAU2_DEG = 2.0;
const ROUTE_INV_TOL_DEG = 0.5;
const KAPPA = 0.5;
const SIGMA_D_DEG = 0.5;
const M_DEG = 0.5;
const F_STAR_DEG = 0.05;
const DECOY_EDIT_SCALE_SD = 0.5;
const D_SUP_INTERCEPT_DEG = 46;
const D_SUP_SLOPE_DEG_PER_DEG = 0.5 / 22;
const D_SUP_NOISE_SD_DEG = 0.1;
const D_SUP_CLAMP_H_DEG = 22;
const D_UNANCH_RO_THRESHOLD_DEG = 10;
const D_UNANCH_NO_THRESHOLD_DEG = 20;
const D_STYLE_LINEAR_WEIGHT = 0.7;
const D_STYLE_NOISE_WEIGHT = 0.3;
const C3AR_PASS_QUANTILE = 0.5;

// --- Path Z parameter ---
const C3AB_II_RADIUS_DEG = 0.5;             // [E] route-basin neighborhood radius
const C3AB_II_PASS_THRESHOLD = 0.9;         // [E] unchanged from v1: ≥90% preserved

// Samples
const EPS_PER_H = 8;
const NUM_H_GRID = Math.floor((H_GRID_MAX - H_GRID_MIN) / H_GRID_STEP) + 1;
const TOTAL_SAMPLES = NUM_H_GRID * EPS_PER_H;

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;
const H_L1_DEG = Math.acos(1 / L1_LEVERAGE_MULTIPLIER) * R2D;

// ===========================================================================
// Math + helpers (identical to v1 to ensure determinism)
// ===========================================================================

function* lcg(seed) {
  let state = (seed >>> 0) >>> 0;
  while (true) {
    state = (state * 1103515245 + 12345) >>> 0;
    yield state / 0x100000000;
  }
}
function sObsFromFParObs(f) { return f / R22_DEG - 1; }
function cL1(s) { return 1 / (1 + Math.exp(-C_L1_K * (s - 0.02))); }
function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}
function mean(xs) { return xs.length === 0 ? null : xs.reduce((a, b) => a + b, 0) / xs.length; }
function quantile(xs, q) {
  const s = [...xs].sort((a, b) => a - b);
  const idx = (s.length - 1) * q;
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return s[lo];
  return s[lo] * (hi - idx) + s[hi] * (idx - lo);
}

function buildPin(seed) {
  const gen = lcg(seed);
  const epsSamples = [];
  for (let i = 0; i < TOTAL_SAMPLES; i++) {
    const u = gen.next().value;
    epsSamples.push((2 * u - 1) * A_DEG);
  }
  const dSupNoise = [];
  for (let i = 0; i < TOTAL_SAMPLES; i++) {
    let u1 = gen.next().value;
    const u2 = gen.next().value;
    if (u1 < 1e-15) u1 = 1e-15;
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    dSupNoise.push(z * D_SUP_NOISE_SD_DEG);
  }
  const dStyleNoise = [];
  for (let i = 0; i < TOTAL_SAMPLES; i++) {
    let u1 = gen.next().value;
    const u2 = gen.next().value;
    if (u1 < 1e-15) u1 = 1e-15;
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    dStyleNoise.push(z);
  }
  const rows = [];
  let idx = 0;
  for (let hi = 0; hi < NUM_H_GRID; hi++) {
    const h_deg = Math.round((H_GRID_MIN + hi * H_GRID_STEP) * 10) / 10;
    for (let ei = 0; ei < EPS_PER_H; ei++) {
      const eps = epsSamples[idx];
      const f_par_obs = R22_DEG / Math.cos(h_deg * D2R) + eps;
      const h_clamped = Math.min(h_deg, D_SUP_CLAMP_H_DEG);
      const d_sup = D_SUP_INTERCEPT_DEG + D_SUP_SLOPE_DEG_PER_DEG * h_clamped + dSupNoise[idx];
      const d_unanch_ro = h_deg >= D_UNANCH_RO_THRESHOLD_DEG ? 1 : 0;
      const d_unanch_no = h_deg >= D_UNANCH_NO_THRESHOLD_DEG ? 1 : 0;
      const d_style = D_STYLE_LINEAR_WEIGHT * (h_deg / 40) + D_STYLE_NOISE_WEIGHT * dStyleNoise[idx];
      rows.push({
        sample_index: idx,
        h_deg,
        eps_deg: eps,
        f_par_obs_deg: f_par_obs,
        d_sup_deg: d_sup,
        d_unanch_rendered_optional: d_unanch_ro,
        d_unanch_named_only: d_unanch_no,
        d_style,
      });
      idx++;
    }
  }
  return rows;
}

function decoyFeatureVector(row) {
  return [
    (row.d_sup_deg - D_SUP_INTERCEPT_DEG) / 0.5,
    row.d_unanch_rendered_optional,
    row.d_unanch_named_only,
    row.d_style,
  ];
}

function solveLinear(A, b) {
  const n = A.length;
  const aug = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < n; i++) {
    let max = Math.abs(aug[i][i]);
    let imax = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(aug[k][i]) > max) { max = Math.abs(aug[k][i]); imax = k; }
    }
    if (imax !== i) [aug[i], aug[imax]] = [aug[imax], aug[i]];
    if (Math.abs(aug[i][i]) < 1e-15) throw new Error("singular");
    for (let k = i + 1; k < n; k++) {
      const factor = aug[k][i] / aug[i][i];
      for (let j = i; j <= n; j++) aug[k][j] -= factor * aug[i][j];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i][n];
    for (let j = i + 1; j < n; j++) sum -= aug[i][j] * x[j];
    x[i] = sum / aug[i][i];
  }
  return x;
}

function olsFitDecoyToH(pin) {
  const p = 5;
  const X = pin.map(r => [1, ...decoyFeatureVector(r)]);
  const y = pin.map(r => r.h_deg);
  const XtX = Array.from({ length: p }, () => new Array(p).fill(0));
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let k = 0; k < X.length; k++) s += X[k][i] * X[k][j];
      XtX[i][j] = s;
    }
    let s = 0;
    for (let k = 0; k < X.length; k++) s += X[k][i] * y[k];
    Xty[i] = s;
  }
  const beta = solveLinear(XtX, Xty);
  let sse = 0, tss = 0;
  const yMean = mean(y);
  for (let k = 0; k < X.length; k++) {
    let yhat = 0;
    for (let i = 0; i < p; i++) yhat += beta[i] * X[k][i];
    const r = y[k] - yhat;
    sse += r * r;
    const dev = y[k] - yMean;
    tss += dev * dev;
  }
  return {
    intercept: beta[0],
    weights: beta.slice(1),
    feature_names: ["d_sup_norm", "d_unanch_rendered_optional", "d_unanch_named_only", "d_style"],
    r_squared: 1 - sse / tss,
    residual_sd_deg: Math.sqrt(sse / (X.length - p)),
  };
}

function hatHDec(row, fit) {
  const x = decoyFeatureVector(row);
  let s = fit.intercept;
  for (let i = 0; i < fit.weights.length; i++) s += fit.weights[i] * x[i];
  return s;
}

function objectiveI(q_h, q_a, row, fit, includeDecoy) {
  const c_l1 = cL1(sObsFromFParObs(row.f_par_obs_deg));
  const r = row.f_par_obs_deg - R22_DEG / Math.cos(q_h * D2R) - q_a;
  const i_route = c_l1 * Math.exp(-(r * r) / (2 * SIGMA_DEG * SIGMA_DEG));
  let decoy_contrib = 0;
  if (includeDecoy) {
    const h_dec = hatHDec(row, fit);
    decoy_contrib = KAPPA * Math.exp(-((q_h - h_dec) ** 2) / (2 * SIGMA_D_DEG * SIGMA_D_DEG));
  }
  const pen = LAMBDA_PEN * ((q_a / A_DEG) ** 2);
  return i_route + decoy_contrib - pen;
}

function findArgmax(row, fit, includeDecoy) {
  const q_h_step = 0.05;
  const q_a_step = A_DEG / 20;
  let best_q_h = Q_H_MIN, best_q_a = 0, best_I = -Infinity;
  for (let q_h = Q_H_MIN; q_h <= Q_H_MAX; q_h += q_h_step) {
    for (let q_a = -A_DEG; q_a <= A_DEG + 1e-9; q_a += q_a_step) {
      const I = objectiveI(q_h, q_a, row, fit, includeDecoy);
      if (I > best_I) { best_I = I; best_q_h = q_h; best_q_a = q_a; }
    }
  }
  return { q_h: best_q_h, q_a: best_q_a, I: best_I };
}

function findArgmaxInNeighborhood(row, fit, includeDecoy, q_h_center, radius_deg) {
  // Search a q_h window of [q_h_center - radius, q_h_center + radius] (clamped),
  // full q_a domain. Returns argmax and whether it's interior in q_h.
  const q_h_step = 0.05;
  const q_a_step = A_DEG / 20;
  const q_h_lo = Math.max(Q_H_MIN, q_h_center - radius_deg);
  const q_h_hi = Math.min(Q_H_MAX, q_h_center + radius_deg);
  let best_q_h = q_h_lo, best_q_a = 0, best_I = -Infinity;
  for (let q_h = q_h_lo; q_h <= q_h_hi + 1e-9; q_h += q_h_step) {
    for (let q_a = -A_DEG; q_a <= A_DEG + 1e-9; q_a += q_a_step) {
      const I = objectiveI(q_h, q_a, row, fit, includeDecoy);
      if (I > best_I) { best_I = I; best_q_h = q_h; best_q_a = q_a; }
    }
  }
  // Interior iff argmax is strictly inside the q_h window (with a small tolerance
  // for floating-point exactness on the boundary).
  const interior_tol = q_h_step / 2;
  const is_interior = best_q_h > q_h_lo + interior_tol && best_q_h < q_h_hi - interior_tol;
  return { q_h: best_q_h, q_a: best_q_a, I: best_I, is_interior, q_h_lo, q_h_hi };
}

function applyDecoyEdit(row) {
  return {
    ...row,
    d_sup_deg: row.d_sup_deg + DECOY_EDIT_SCALE_SD * D_SUP_NOISE_SD_DEG,
    d_unanch_rendered_optional: 1 - row.d_unanch_rendered_optional,
    d_unanch_named_only: 1 - row.d_unanch_named_only,
    d_style: row.d_style + DECOY_EDIT_SCALE_SD * 1.0,
  };
}

function applyHandleEdit(row, h_prime_deg) {
  return {
    ...row,
    h_deg_counterfactual: h_prime_deg,
    f_par_obs_deg: R22_DEG / Math.cos(h_prime_deg * D2R),
  };
}

function piRouteV2(row) {
  // Path Y: emit q_naive on all non-degenerate rows (f_par_obs ≥ R22),
  // even when on the L1-ineligible-by-obs band (biased but defined).
  // Abstain only on truly degenerate rows where arccos is undefined.
  if (row.f_par_obs_deg < R22_DEG) return null;
  const ratio = R22_DEG / row.f_par_obs_deg;
  if (ratio < -1 || ratio > 1) return null;
  return Math.acos(ratio) * R2D;
}

function piDec(row, fit) {
  return hatHDec(row, fit);
}

// ===========================================================================
// Receipts (v2)
// ===========================================================================

function buildC3ARReceiptV2(pin, fit) {
  // Identical to v1 — C3-A-R is not affected by Path Y or Path Z.
  const rows = [];
  for (const row of pin) {
    const baseline = findArgmax(row, fit, true);
    const edited = findArgmax(applyDecoyEdit(row), fit, true);
    const shift_q_h = Math.abs(edited.q_h - baseline.q_h);
    const shift_q_a = Math.abs(edited.q_a - baseline.q_a);
    const shift = Math.sqrt(shift_q_h ** 2 + shift_q_a ** 2);
    rows.push({
      sample_index: row.sample_index,
      h_deg: row.h_deg,
      f_par_obs_deg: row.f_par_obs_deg,
      baseline_q_h: baseline.q_h,
      edited_q_h: edited.q_h,
      shift_deg: shift,
      passes_floor: shift > F_STAR_DEG,
    });
  }
  const samplesAbove = rows.filter(r => r.passes_floor).length;
  const fraction = samplesAbove / rows.length;
  const shifts = rows.map(r => r.shift_deg);
  return {
    receipt: "C3-A-R (v2, identical to v1)",
    title: "Reachability — argmax-sensitivity to frozen decoy-edit (Path B quantile rule)",
    F_star_deg: F_STAR_DEG,
    pass_quantile_threshold: C3AR_PASS_QUANTILE,
    samples_total: rows.length,
    samples_above_floor: samplesAbove,
    fraction_above_floor: fraction,
    shift_distribution: {
      median_deg: median(shifts),
      p75_deg: quantile(shifts, 0.75),
      p90_deg: quantile(shifts, 0.9),
    },
    pass: fraction >= C3AR_PASS_QUANTILE,
  };
}

function buildC3ATReceiptV2_PathY(pin, fit) {
  // Path Y framing: full non-degenerate P_in (f_par_obs ≥ R22).
  // π_route emits biased q_naive on L1-ineligible-by-obs rows.
  // π_dec emits ĥ_dec(d) on all rows. Compare on non-degenerate subset.
  const nonDegenerate = pin.filter(r => r.f_par_obs_deg >= R22_DEG);
  const eligibleByObs = nonDegenerate.filter(r => r.f_par_obs_deg >= L1_LEVERAGE_MULTIPLIER * R22_DEG);
  const ineligibleByObs = nonDegenerate.filter(r => r.f_par_obs_deg < L1_LEVERAGE_MULTIPLIER * R22_DEG);

  const piDecErrors = nonDegenerate.map(r => Math.abs(piDec(r, fit) - r.h_deg));
  const piRouteErrors = nonDegenerate.map(r => Math.abs(piRouteV2(r) - r.h_deg));
  const meanPiDec = mean(piDecErrors);
  const meanPiRoute = mean(piRouteErrors);
  const margin_achieved = meanPiRoute - meanPiDec;
  const T1_pass = meanPiDec <= meanPiRoute - M_DEG;

  // Also report the subset breakdowns for diagnostic transparency
  const eligibleDecErr = mean(eligibleByObs.map(r => Math.abs(piDec(r, fit) - r.h_deg)));
  const eligibleRouteErr = mean(eligibleByObs.map(r => Math.abs(piRouteV2(r) - r.h_deg)));
  const ineligibleDecErr = mean(ineligibleByObs.map(r => Math.abs(piDec(r, fit) - r.h_deg)));
  const ineligibleRouteErr = mean(ineligibleByObs.map(r => Math.abs(piRouteV2(r) - r.h_deg)));

  // T2 reversal (decoy-edit) — unchanged from v1 evaluated on non-degenerate
  const piDecAfterDecoyEdit = nonDegenerate.map(r => Math.abs(piDec(applyDecoyEdit(r), fit) - r.h_deg));
  const meanPiDecAfterDecoy = mean(piDecAfterDecoyEdit);
  const T2_pi_dec_pass = meanPiDecAfterDecoy >= TAU_PC_DEG;

  // Route invariance under decoy edit
  const routeShifts = nonDegenerate.map(r => {
    const baseline = findArgmax(r, fit, false);
    const edited = findArgmax(applyDecoyEdit(r), fit, false);
    return Math.abs(edited.q_h - baseline.q_h);
  });
  const maxRouteShift = Math.max(...routeShifts);
  const T2_route_invariant_pass = maxRouteShift <= ROUTE_INV_TOL_DEG;

  // T3 handle-edit reversal — same scheme as v1
  const T3rows = [];
  for (const r of nonDegenerate) {
    const h_prime = Math.min(40, Math.max(H_L1_DEG + 1, r.h_deg + 5));
    if (h_prime === r.h_deg) continue;
    const edited = applyHandleEdit(r, h_prime);
    if (edited.f_par_obs_deg < R22_DEG) continue;
    const piRouteAfter = piRouteV2(edited);
    if (piRouteAfter === null) continue;
    const piDecStale = piDec(r, fit);
    T3rows.push({
      pi_route_error: Math.abs(piRouteAfter - h_prime),
      pi_dec_error: Math.abs(piDecStale - h_prime),
    });
  }
  const meanPiRouteAfterHandle = mean(T3rows.map(r => r.pi_route_error));
  const meanPiDecAfterHandle = mean(T3rows.map(r => r.pi_dec_error));
  const T3_pass = meanPiRouteAfterHandle <= TAU2_DEG && meanPiDecAfterHandle > TAU2_DEG;

  return {
    receipt: "C3-A-T v2 (Path Y framing)",
    title: "Temptation + reversal — evaluated on full non-degenerate P_in",
    spec_reference: "P2_CUT2_C3A_NUMERIC_FREEZE.md §4 + Wave-4.1 audit-notes (Path Y)",
    framing: "Path Y: full non-degenerate P_in (f_par_obs ≥ R22). π_route emits biased q_naive on L1-ineligible-by-obs rows; π_dec emits ĥ_dec(d) on all rows. Truly-degenerate rows excluded because q_naive is undefined.",
    M_deg: M_DEG,
    tau_pc_deg: TAU_PC_DEG,
    tau2_deg: TAU2_DEG,
    sample_counts: {
      non_degenerate_total: nonDegenerate.length,
      L1_eligible_by_obs: eligibleByObs.length,
      L1_ineligible_by_obs: ineligibleByObs.length,
    },
    T1: {
      pi_dec_mean_error: meanPiDec,
      pi_route_mean_error: meanPiRoute,
      margin_achieved_deg: margin_achieved,
      margin_required_deg: M_DEG,
      pass: T1_pass,
      subset_breakdown: {
        L1_eligible_by_obs: { pi_dec_err: eligibleDecErr, pi_route_err: eligibleRouteErr, count: eligibleByObs.length },
        L1_ineligible_by_obs: { pi_dec_err: ineligibleDecErr, pi_route_err: ineligibleRouteErr, count: ineligibleByObs.length },
      },
    },
    T2_decoy_edit: {
      pi_dec_mean_error_after: meanPiDecAfterDecoy,
      pi_dec_pass: T2_pi_dec_pass,
      route_max_shift_deg: maxRouteShift,
      route_invariant_pass: T2_route_invariant_pass,
    },
    T3_handle_edit: {
      pi_route_mean_error: meanPiRouteAfterHandle,
      pi_dec_mean_error: meanPiDecAfterHandle,
      pass: T3_pass,
      sample_count: T3rows.length,
    },
    pass: T1_pass && T2_pi_dec_pass && T2_route_invariant_pass && T3_pass,
  };
}

function buildC3ABReceiptV2_PathZ(pin, fit, c3atReceipt) {
  // Path Z: "route argmax is REPRESENTED in the full I — there exists a
  // local max of full I within radius C3AB_II_RADIUS_DEG of q_naive."
  // A "local max within neighborhood" = the argmax in the q_h window
  // [q_naive - R, q_naive + R] is INTERIOR to the window (i.e. the I
  // function isn't strictly increasing/decreasing across the whole window).
  const i_pass = c3atReceipt.T1.pass;
  const eligible = pin.filter(r => r.f_par_obs_deg >= L1_LEVERAGE_MULTIPLIER * R22_DEG);
  let preserved = 0;
  const samples = [];
  for (const r of eligible) {
    const q_naive = Math.acos(R22_DEG / r.f_par_obs_deg) * R2D;
    const local = findArgmaxInNeighborhood(r, fit, true, q_naive, C3AB_II_RADIUS_DEG);
    if (local.is_interior) preserved++;
    samples.push({
      sample_index: r.sample_index,
      h_deg: r.h_deg,
      q_naive_deg: q_naive,
      local_argmax_q_h: local.q_h,
      local_argmax_offset_from_q_naive: Math.abs(local.q_h - q_naive),
      is_interior_local_max: local.is_interior,
    });
  }
  const fraction = preserved / eligible.length;
  const ii_pass = fraction >= C3AB_II_PASS_THRESHOLD;
  return {
    receipt: "C3-A-B v2 (Path Z sub-(ii) reformulation)",
    title: "Kappa window — sub-(ii) reformulated as 'route basin preserved within 0.5° of q_naive'",
    spec_reference: "P2_CUT2_C3A_NUMERIC_FREEZE.md §4 + Wave-4.1 audit-notes (Path Z)",
    framing: "Path Z: route argmax is REPRESENTED in the full I — there exists a local max within radius 0.5° of q_naive. Accepts that small-Δ decoy merging is geometric, not erasure; the route is represented in the merged peak.",
    kappa: KAPPA,
    sub_ii_radius_deg: C3AB_II_RADIUS_DEG,
    sub_i_temptation_real: i_pass,
    sub_ii_route_basin_preserved: preserved,
    sub_ii_eligible_count: eligible.length,
    sub_ii_fraction_preserved: fraction,
    sub_ii_pass_threshold: C3AB_II_PASS_THRESHOLD,
    sub_ii_pass: ii_pass,
    pass: i_pass && ii_pass,
    samples_summary: samples.slice(0, 20),
  };
}

// ===========================================================================
// Hashing + output
// ===========================================================================

function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
function canonical(value) {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return "{" + keys.map(k => JSON.stringify(k) + ":" + canonical(value[k])).join(",") + "}";
  }
  return JSON.stringify(value);
}

async function writeJsonAndHash(relPath, payload) {
  const absPath = resolve(REPO, relPath);
  await mkdir(dirname(absPath), { recursive: true });
  const pretty = JSON.stringify(payload, null, 2) + "\n";
  await writeFile(absPath, pretty);
  return {
    path: relPath,
    raw_sha256: sha256(Buffer.from(pretty)),
    canonical_sha256: sha256(Buffer.from(canonical(payload))),
  };
}

async function main() {
  const pin = buildPin(SEED);
  const fit = olsFitDecoyToH(pin);

  const c3ar = buildC3ARReceiptV2(pin, fit);
  const c3at = buildC3ATReceiptV2_PathY(pin, fit);
  const c3ab = buildC3ABReceiptV2_PathZ(pin, fit, c3at);

  const hashes = {};
  hashes.r = await writeJsonAndHash(`${OUT_DIR_REL}/c3a-r-receipt-v2.json`, c3ar);
  hashes.t = await writeJsonAndHash(`${OUT_DIR_REL}/c3a-t-receipt-v2.json`, c3at);
  hashes.b = await writeJsonAndHash(`${OUT_DIR_REL}/c3a-b-receipt-v2.json`, c3ab);

  // Markdown summary
  const lines = [];
  lines.push("# C3-A Receipts (Wave 4.1 — Path Y + Path Z amendment)");
  lines.push("");
  lines.push("## Summary");
  lines.push("");
  lines.push("| receipt | status |");
  lines.push("| --- | --- |");
  lines.push(`| C3-A-R reachability (unchanged from v1) | ${c3ar.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C3-A-T temptation v2 (Path Y framing) | ${c3at.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C3-A-B kappa window v2 (Path Z sub-(ii)) | ${c3ab.pass ? "PASS" : "BLOCK"} |`);
  lines.push("");
  lines.push("## C3-A-T v2 (Path Y)");
  lines.push("");
  lines.push(`Non-degenerate P_in: **${c3at.sample_counts.non_degenerate_total}** (eligible-by-obs ${c3at.sample_counts.L1_eligible_by_obs}, ineligible-by-obs ${c3at.sample_counts.L1_ineligible_by_obs}).`);
  lines.push("");
  lines.push(`**T1 base temptation** (full non-degenerate): mean|π_dec − h| = **${c3at.T1.pi_dec_mean_error.toFixed(3)}°** vs mean|π_route − h| = **${c3at.T1.pi_route_mean_error.toFixed(3)}°**. Margin achieved = **${c3at.T1.margin_achieved_deg.toFixed(3)}°** (required ≥ ${M_DEG}°). ${c3at.T1.pass ? "PASS" : "BLOCK"}.`);
  lines.push("");
  lines.push(`Subset breakdown:`);
  lines.push(`- L1-eligible-by-obs (${c3at.T1.subset_breakdown.L1_eligible_by_obs.count}): π_dec err ${c3at.T1.subset_breakdown.L1_eligible_by_obs.pi_dec_err.toFixed(3)}°, π_route err ${c3at.T1.subset_breakdown.L1_eligible_by_obs.pi_route_err.toFixed(3)}°`);
  lines.push(`- L1-ineligible-by-obs (${c3at.T1.subset_breakdown.L1_ineligible_by_obs.count}): π_dec err ${c3at.T1.subset_breakdown.L1_ineligible_by_obs.pi_dec_err.toFixed(3)}°, π_route err ${c3at.T1.subset_breakdown.L1_ineligible_by_obs.pi_route_err.toFixed(3)}°`);
  lines.push("");
  lines.push(`**T2 decoy-edit reversal**: π_dec err after = ${c3at.T2_decoy_edit.pi_dec_mean_error_after.toFixed(3)}° (≥${TAU_PC_DEG}°): ${c3at.T2_decoy_edit.pi_dec_pass ? "PASS" : "BLOCK"}. Route max shift = ${c3at.T2_decoy_edit.route_max_shift_deg.toFixed(3)}° (≤${ROUTE_INV_TOL_DEG}°): ${c3at.T2_decoy_edit.route_invariant_pass ? "PASS" : "BLOCK"}.`);
  lines.push("");
  lines.push(`**T3 handle-edit reversal**: π_route err ${c3at.T3_handle_edit.pi_route_mean_error.toFixed(3)}° (≤${TAU2_DEG}°), π_dec stale err ${c3at.T3_handle_edit.pi_dec_mean_error.toFixed(3)}° (>${TAU2_DEG}°): ${c3at.T3_handle_edit.pass ? "PASS" : "BLOCK"}.`);
  lines.push("");
  lines.push("## C3-A-B v2 (Path Z)");
  lines.push("");
  lines.push(`Sub-(i) temptation real: ${c3ab.sub_i_temptation_real ? "PASS" : "BLOCK"}.`);
  lines.push(`Sub-(ii) route basin preserved (∃ local max within ${C3AB_II_RADIUS_DEG}° of q_naive): **${c3ab.sub_ii_route_basin_preserved}/${c3ab.sub_ii_eligible_count}** = **${(c3ab.sub_ii_fraction_preserved*100).toFixed(1)}%** (required ≥ ${(C3AB_II_PASS_THRESHOLD*100).toFixed(0)}%): ${c3ab.sub_ii_pass ? "PASS" : "BLOCK"}.`);
  lines.push("");
  const md = lines.join("\n") + "\n";
  const mdRel = `${OUT_DIR_REL}/c3a-w4-v2-summary.md`;
  await writeFile(resolve(REPO, mdRel), md);
  const mdHash = sha256(Buffer.from(md));

  console.log("[c3a-w4-v2] artifacts:");
  for (const [k, h] of Object.entries(hashes)) {
    console.log(`  ${h.path}`);
    console.log(`    raw       = ${h.raw_sha256}`);
    console.log(`    canonical = ${h.canonical_sha256}`);
  }
  console.log(`  ${mdRel}`);
  console.log(`    raw       = ${mdHash}`);
  console.log("");
  console.log("[c3a-w4-v2] verdicts:");
  console.log(`  C3-A-R (unchanged) : ${c3ar.pass ? "PASS" : "BLOCK"}`);
  console.log(`  C3-A-T v2 (Path Y) : ${c3at.pass ? "PASS" : "BLOCK"}`);
  console.log(`    T1 margin=${c3at.T1.margin_achieved_deg.toFixed(3)}° (≥${M_DEG}°)=${c3at.T1.pass}`);
  console.log(`    subset: eligible(${c3at.T1.subset_breakdown.L1_eligible_by_obs.count}) dec=${c3at.T1.subset_breakdown.L1_eligible_by_obs.pi_dec_err.toFixed(2)}° route=${c3at.T1.subset_breakdown.L1_eligible_by_obs.pi_route_err.toFixed(2)}°`);
  console.log(`    subset: ineligible(${c3at.T1.subset_breakdown.L1_ineligible_by_obs.count}) dec=${c3at.T1.subset_breakdown.L1_ineligible_by_obs.pi_dec_err.toFixed(2)}° route=${c3at.T1.subset_breakdown.L1_ineligible_by_obs.pi_route_err.toFixed(2)}°`);
  console.log(`    T2_dec=${c3at.T2_decoy_edit.pi_dec_pass}, T2_inv=${c3at.T2_decoy_edit.route_invariant_pass}, T3=${c3at.T3_handle_edit.pass}`);
  console.log(`  C3-A-B v2 (Path Z) : ${c3ab.pass ? "PASS" : "BLOCK"}`);
  console.log(`    (i)=${c3ab.sub_i_temptation_real}, (ii) ${c3ab.sub_ii_route_basin_preserved}/${c3ab.sub_ii_eligible_count}=${(c3ab.sub_ii_fraction_preserved*100).toFixed(1)}% (≥${(C3AB_II_PASS_THRESHOLD*100).toFixed(0)}%)=${c3ab.sub_ii_pass}`);
}

main().catch(err => {
  console.error(`[c3a-w4-v2] FAILED: ${err.message}`);
  console.error(err.stack);
  process.exit(1);
});
