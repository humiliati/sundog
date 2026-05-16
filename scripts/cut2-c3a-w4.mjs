#!/usr/bin/env node
// scripts/cut2-c3a-w4.mjs
//
// Wave-4 C3-A receipts generator. Path B (Wave-4 sign-off 2026-05-16):
// C3-A-R uses the median/quantile rule "≥50% of P_in samples have argmax
// shift > F*", NOT every-sample, given the exponential decay of shift at
// large decoy-vs-route offset under κ=0.5 with equal-width Gaussians.
//
// Produces (all under the C5 allowlist):
//   results/structural-failure/cut2-prereg/c3a-pin-generator.json
//   results/structural-failure/cut2-prereg/c3a-r-receipt.json     (reachability)
//   results/structural-failure/cut2-prereg/c3a-t-receipt.json     (temptation + reversal)
//   results/structural-failure/cut2-prereg/c3a-b-receipt.json     (kappa window)
//   results/structural-failure/cut2-prereg/c3a-w4-summary.md
//
// Defect-surface discipline (Wave-3 lesson):
//   - C_L1 reads observable s_obs = f_par_obs/R22 - 1 per C2-A freeze §1.
//   - Constants are named with the freeze section they trace to in
//     adjacent comments.
//   - No Edit churn after authoring; this script is Write-only.

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

// ===========================================================================
// Constants — inherited Wave-2 (P2_CUT2_C2A_NUMERIC_FREEZE.md §4 audit-notes)
// ===========================================================================
const R22_DEG = 22;                         // [G] phase3.HALO_22_RADIUS
const L1_LEVERAGE_MULTIPLIER = 1.02;        // [G] BOUNDARY_MAP L1
const RHO = 0.02;                           // [E] anchor-noise scale
const SIGMA_DEG = 0.5;                      // [E] route Gaussian width in q_h
const SEED = 20260516;                      // [E] determinism seed
const H_GRID_MIN = 0;                       // [E]
const H_GRID_MAX = 40;                      // [E]
const H_GRID_STEP = 0.5;                    // [E]
const Q_H_MIN = 0;                          // [E]
const Q_H_MAX = 60;                         // [E]
const LAMBDA_PEN = 1.0;                     // [E] convex penalty strength
const C_L1_K = 600;                         // [E] sigmoid steepness
const A_DEG = RHO * R22_DEG;                // derived: anchor-noise bound ≈ 0.44°
const BRIDGE_SCALE = 1.0;                   // [E] eligible-band route peak ≡ 1.0
const TAU_PC_DEG = 2.0;                     // [G] reused from A2 positive-control freeze
const TAU2_DEG = 2.0;                       // [G] reused from spec q2 handle-edit
const ROUTE_INV_TOL_DEG = 0.5;              // [G] spec q2 decoy-invariance tolerance

// ===========================================================================
// Constants — Wave-4 [E] (signed off in conversation 2026-05-16, Path B)
// ===========================================================================
const KAPPA = 0.5;                          // [E] decoy-ridge weight
const SIGMA_D_DEG = 0.5;                    // [E] decoy ridge width (= SIGMA_DEG)
const M_DEG = 0.5;                          // [E] temptation margin
const F_STAR_DEG = 0.05;                    // [E] shared C3-A-R / C4 D2 floor (Wave-4 v2 per user pushback)
const DECOY_EDIT_SCALE_SD = 0.5;            // [E] decoy edit magnitude in SDs

// Per-decoy correlation coefficients (BOUNDARY_MAP-pinned + synthetic priors)
const D_SUP_INTERCEPT_DEG = 46;             // BOUNDARY_MAP L4 + Cut-1 harness convention
const D_SUP_SLOPE_DEG_PER_DEG = 0.5 / 22;   // BOUNDARY_MAP L4: 0.5° across h=0-22°
const D_SUP_NOISE_SD_DEG = 0.1;             // BOUNDARY_MAP L4 visual-edge measurement noise
const D_SUP_CLAMP_H_DEG = 22;               // clamp h at 22°, do NOT extrapolate L4
const D_UNANCH_RO_THRESHOLD_DEG = 10;       // synthetic atlas-step nuisance prior (NOT BM-derived)
const D_UNANCH_NO_THRESHOLD_DEG = 20;       // synthetic atlas-step nuisance prior (NOT BM-derived)
const D_STYLE_LINEAR_WEIGHT = 0.7;          // synthetic
const D_STYLE_NOISE_WEIGHT = 0.3;           // synthetic

// Receipt rules
const C3AR_PASS_QUANTILE = 0.5;             // [E] Path B: ≥50% of samples have shift > F*
const C3AB_II_PASS_THRESHOLD = 0.9;         // [E] ≥90% of eligible samples preserve route argmax as local max

// Samples
const EPS_PER_H = 8;                        // [E] noise samples per h-grid point
const NUM_H_GRID = Math.floor((H_GRID_MAX - H_GRID_MIN) / H_GRID_STEP) + 1; // 81
const TOTAL_SAMPLES = NUM_H_GRID * EPS_PER_H; // 648

// Derived
const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;
const H_L1_DEG = Math.acos(1 / L1_LEVERAGE_MULTIPLIER) * R2D;

// ===========================================================================
// Math helpers
// ===========================================================================

function* lcg(seed) {
  let state = (seed >>> 0) >>> 0;
  while (true) {
    state = (state * 1103515245 + 12345) >>> 0;
    yield state / 0x100000000;
  }
}

function sObsFromFParObs(f_par_obs) {
  return f_par_obs / R22_DEG - 1;
}

function cL1(s) {
  return 1 / (1 + Math.exp(-C_L1_K * (s - 0.02)));
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}
function mean(xs) {
  if (xs.length === 0) return null;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
function quantile(xs, q) {
  const s = [...xs].sort((a, b) => a - b);
  const idx = (s.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return s[lo];
  return s[lo] * (hi - idx) + s[hi] * (idx - lo);
}

// ===========================================================================
// P_in generator — deterministic given seed; draw order frozen
// ===========================================================================

function buildPin(seed) {
  // Draw order (frozen per Wave-4 sign-off):
  //   Phase A: 648 uniforms → ε samples
  //   Phase B: 1296 uniforms (Box-Muller pairs) → d_sup_noise
  //   Phase C: 1296 uniforms (Box-Muller pairs) → d_style_noise
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

// ===========================================================================
// OLS (w,b) fit — decoys-only feature vector, h target
// ===========================================================================

function decoyFeatureVector(row) {
  // Decoy sub-vector (4 features). NOTE: genuine handles (f_par, f_cza, f_tan)
  // are deliberately excluded — this represents a PURE DECOY-CORRELATE policy,
  // distinct from Cut-1's positiveControlVector which used all 8 features.
  return [
    (row.d_sup_deg - D_SUP_INTERCEPT_DEG) / 0.5, // d_sup_norm
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
    if (Math.abs(aug[i][i]) < 1e-15) {
      throw new Error(`singular matrix at row ${i}`);
    }
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
  // Add intercept column. p = 5 (intercept + 4 decoys).
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
  // Residuals for diagnostics
  let sse = 0;
  let tss = 0;
  const yMean = mean(y);
  for (let k = 0; k < X.length; k++) {
    let yhat = 0;
    for (let i = 0; i < p; i++) yhat += beta[i] * X[k][i];
    const r = y[k] - yhat;
    sse += r * r;
    const dev = y[k] - yMean;
    tss += dev * dev;
  }
  const r_squared = 1 - sse / tss;
  const residual_sd_deg = Math.sqrt(sse / (X.length - p));
  return {
    intercept: beta[0],
    weights: beta.slice(1),
    feature_names: ["d_sup_norm", "d_unanch_rendered_optional", "d_unanch_named_only", "d_style"],
    r_squared,
    residual_sd_deg,
  };
}

function hatHDec(row, fit) {
  const x = decoyFeatureVector(row);
  let s = fit.intercept;
  for (let i = 0; i < fit.weights.length; i++) s += fit.weights[i] * x[i];
  return s;
}

// ===========================================================================
// Objective + argmax
// ===========================================================================

function objectiveI(q_h, q_a, row, fit, includeDecoy) {
  const c_l1 = cL1(sObsFromFParObs(row.f_par_obs_deg));
  const r = row.f_par_obs_deg - R22_DEG / Math.cos(q_h * D2R) - q_a;
  const i_route = c_l1 * Math.exp(-(r * r) / (2 * SIGMA_DEG * SIGMA_DEG));
  let decoy_contrib = 0;
  if (includeDecoy) {
    const h_dec = hatHDec(row, fit);
    decoy_contrib = KAPPA * Math.exp(-Math.pow(q_h - h_dec, 2) / (2 * SIGMA_D_DEG * SIGMA_D_DEG));
  }
  const pen = LAMBDA_PEN * (q_a / A_DEG) * (q_a / A_DEG);
  return i_route + decoy_contrib - pen;
}

function findArgmax(row, fit, includeDecoy) {
  const q_h_step = 0.05;
  const q_a_step = A_DEG / 20;
  let best_q_h = Q_H_MIN;
  let best_q_a = 0;
  let best_I = -Infinity;
  for (let q_h = Q_H_MIN; q_h <= Q_H_MAX; q_h += q_h_step) {
    for (let q_a = -A_DEG; q_a <= A_DEG + 1e-9; q_a += q_a_step) {
      const I = objectiveI(q_h, q_a, row, fit, includeDecoy);
      if (I > best_I) {
        best_I = I;
        best_q_h = q_h;
        best_q_a = q_a;
      }
    }
  }
  return { q_h: best_q_h, q_a: best_q_a, I: best_I };
}

// ===========================================================================
// Edit operators (q2)
// ===========================================================================

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

// ===========================================================================
// Reference policies (argmax-style, NOT the bound controller)
// ===========================================================================

function piRoute(row) {
  // L1-eligible-by-obs: q_naive = arccos(R22 / f_par_obs); else abstain.
  if (row.f_par_obs_deg < L1_LEVERAGE_MULTIPLIER * R22_DEG) return null;
  return Math.acos(R22_DEG / row.f_par_obs_deg) * R2D;
}

function piDec(row, fit) {
  return hatHDec(row, fit);
}

// ===========================================================================
// Receipt builders
// ===========================================================================

function buildC3ARReceipt(pin, fit) {
  // For each P_in sample, compute argmax(bundle) and argmax(decoy-edited bundle).
  // Quantile rule (Path B): ≥ C3AR_PASS_QUANTILE of samples have shift > F*.
  const rows = [];
  for (const row of pin) {
    const baseline = findArgmax(row, fit, true);
    const edited = findArgmax(applyDecoyEdit(row), fit, true);
    const shift_q_h = Math.abs(edited.q_h - baseline.q_h);
    const shift_q_a = Math.abs(edited.q_a - baseline.q_a);
    const shift = Math.sqrt(shift_q_h * shift_q_h + shift_q_a * shift_q_a);
    rows.push({
      sample_index: row.sample_index,
      h_deg: row.h_deg,
      eps_deg: row.eps_deg,
      f_par_obs_deg: row.f_par_obs_deg,
      hat_h_dec_baseline: hatHDec(row, fit),
      hat_h_dec_edited: hatHDec(applyDecoyEdit(row), fit),
      baseline_q_h: baseline.q_h,
      baseline_q_a: baseline.q_a,
      edited_q_h: edited.q_h,
      edited_q_a: edited.q_a,
      shift_q_h_deg: shift_q_h,
      shift_q_a_deg: shift_q_a,
      shift_deg: shift,
      passes_floor: shift > F_STAR_DEG,
    });
  }
  const samplesAbove = rows.filter(r => r.passes_floor).length;
  const fraction = samplesAbove / rows.length;
  const shifts = rows.map(r => r.shift_deg);
  return {
    receipt: "C3-A-R",
    title: "Reachability — argmax-sensitivity to frozen decoy-edit (Path B quantile rule)",
    spec_reference: "P2_CUT2_C3A_NUMERIC_FREEZE.md §4 + Wave-4 audit-notes",
    F_star_deg: F_STAR_DEG,
    pass_rule: `≥${(C3AR_PASS_QUANTILE * 100).toFixed(0)}% of P_in samples have shift > F* = ${F_STAR_DEG}°`,
    pass_quantile_threshold: C3AR_PASS_QUANTILE,
    samples_total: rows.length,
    samples_above_floor: samplesAbove,
    fraction_above_floor: fraction,
    shift_distribution: {
      min_deg: Math.min(...shifts),
      p10_deg: quantile(shifts, 0.1),
      p25_deg: quantile(shifts, 0.25),
      median_deg: median(shifts),
      p75_deg: quantile(shifts, 0.75),
      p90_deg: quantile(shifts, 0.9),
      max_deg: Math.max(...shifts),
    },
    pass: fraction >= C3AR_PASS_QUANTILE,
    rows,
  };
}

function buildC3ATReceipt(pin, fit) {
  // Restrict comparison to L1-eligible-by-obs samples (where π_route is defined).
  const eligible = pin.filter(r => r.f_par_obs_deg >= L1_LEVERAGE_MULTIPLIER * R22_DEG);

  const piDecErrors = eligible.map(r => Math.abs(piDec(r, fit) - r.h_deg));
  const piRouteErrors = eligible.map(r => Math.abs(piRoute(r) - r.h_deg));
  const meanPiDec = mean(piDecErrors);
  const meanPiRoute = mean(piRouteErrors);
  const margin_achieved = meanPiRoute - meanPiDec;
  const T1_pass = meanPiDec <= meanPiRoute - M_DEG;

  // T2 — decoy-edit reversal:
  //   π_dec on edited bundle uses stale (w,b); error rises ≥ τ_pc
  //   π_route's q̂ stays invariant under decoy-edit (handles fixed): |Δq̂| ≤ 0.5°
  const piDecAfterDecoyEdit = eligible.map(r => Math.abs(piDec(applyDecoyEdit(r), fit) - r.h_deg));
  const meanPiDecAfterDecoy = mean(piDecAfterDecoyEdit);
  const T2_pi_dec_pass = meanPiDecAfterDecoy >= TAU_PC_DEG;

  // Route-only argmax shifts under decoy edit (handles fixed, route landscape unchanged)
  // — q_a still bounded, q_h is q_naive both before and after; should be 0.
  const routeShifts = eligible.map(r => {
    const baseline = findArgmax(r, fit, false);
    const edited = findArgmax(applyDecoyEdit(r), fit, false);
    return Math.abs(edited.q_h - baseline.q_h);
  });
  const maxRouteShift = Math.max(...routeShifts);
  const T2_route_invariant_pass = maxRouteShift <= ROUTE_INV_TOL_DEG;

  // T3 — handle-edit reversal:
  //   π_route adapts to h' within τ2; π_dec stays stale.
  // Choose h' = h + 5° (clamped to h_grid range). For samples where the
  // edit pushes off the L1-eligible-by-obs band of the COUNTERFACTUAL,
  // skip; the test is over samples where both pre and post are eligible.
  const T3rows = [];
  for (const r of eligible) {
    const h_prime = Math.min(40, Math.max(H_L1_DEG + 1, r.h_deg + 5));
    if (h_prime === r.h_deg) continue;
    const edited = applyHandleEdit(r, h_prime);
    if (edited.f_par_obs_deg < L1_LEVERAGE_MULTIPLIER * R22_DEG) continue;
    const piRouteAfter = Math.acos(R22_DEG / edited.f_par_obs_deg) * R2D;
    const piDecStale = piDec(r, fit); // decoys unchanged from baseline
    T3rows.push({
      sample_index: r.sample_index,
      h_deg: r.h_deg,
      h_prime_deg: h_prime,
      pi_route_after_handle_edit: piRouteAfter,
      pi_route_error_vs_h_prime: Math.abs(piRouteAfter - h_prime),
      pi_dec_stale: piDecStale,
      pi_dec_error_vs_h_prime: Math.abs(piDecStale - h_prime),
    });
  }
  const piRouteHandleErrs = T3rows.map(r => r.pi_route_error_vs_h_prime);
  const piDecHandleErrs = T3rows.map(r => r.pi_dec_error_vs_h_prime);
  const meanPiRouteAfterHandle = mean(piRouteHandleErrs);
  const meanPiDecAfterHandle = mean(piDecHandleErrs);
  const T3_pass = meanPiRouteAfterHandle <= TAU2_DEG && meanPiDecAfterHandle > TAU2_DEG;

  return {
    receipt: "C3-A-T",
    title: "Temptation + reversal (in-sample vs argmax-style reference policies)",
    spec_reference: "P2_CUT2_C3A_NUMERIC_FREEZE.md §4 + Wave-4 audit-notes",
    M_deg: M_DEG,
    tau_pc_deg: TAU_PC_DEG,
    tau2_deg: TAU2_DEG,
    route_invariance_tol_deg: ROUTE_INV_TOL_DEG,
    L1_eligible_by_obs_count: eligible.length,
    T1: {
      pi_dec_mean_error: meanPiDec,
      pi_route_mean_error: meanPiRoute,
      margin_achieved_deg: margin_achieved,
      margin_required_deg: M_DEG,
      pass: T1_pass,
    },
    T2_decoy_edit: {
      pi_dec_mean_error_after: meanPiDecAfterDecoy,
      pi_dec_required_threshold: TAU_PC_DEG,
      pi_dec_pass: T2_pi_dec_pass,
      route_max_shift_deg: maxRouteShift,
      route_invariance_tol_deg: ROUTE_INV_TOL_DEG,
      route_invariant_pass: T2_route_invariant_pass,
    },
    T3_handle_edit: {
      pi_route_mean_error: meanPiRouteAfterHandle,
      pi_route_tol: TAU2_DEG,
      pi_dec_mean_error: meanPiDecAfterHandle,
      pi_dec_tol_required_above: TAU2_DEG,
      pass: T3_pass,
      sample_count: T3rows.length,
    },
    pass: T1_pass && T2_pi_dec_pass && T2_route_invariant_pass && T3_pass,
  };
}

function buildC3ABReceipt(pin, fit, c3atReceipt) {
  // (i) C3-A-T.T1 must pass (temptation is real).
  const i_pass = c3atReceipt.T1.pass;

  // (ii) Route argmax (q_naive, q_a=0) remains a local max of the full I
  // on the L1-eligible-by-obs band. Local max iff displacement in either
  // direction by half a grid step does not increase I.
  const eligible = pin.filter(r => r.f_par_obs_deg >= L1_LEVERAGE_MULTIPLIER * R22_DEG);
  const delta_q_h = 0.025; // half of grid step 0.05°
  const delta_q_a = A_DEG / 40; // half of q_a grid step
  let preserved = 0;
  const samples = [];
  for (const r of eligible) {
    const q_naive = Math.acos(R22_DEG / r.f_par_obs_deg) * R2D;
    const I_c = objectiveI(q_naive, 0, r, fit, true);
    const I_qh_p = objectiveI(q_naive + delta_q_h, 0, r, fit, true);
    const I_qh_m = objectiveI(q_naive - delta_q_h, 0, r, fit, true);
    const I_qa_p = objectiveI(q_naive, delta_q_a, r, fit, true);
    const I_qa_m = objectiveI(q_naive, -delta_q_a, r, fit, true);
    const is_local_max =
      I_c >= I_qh_p && I_c >= I_qh_m && I_c >= I_qa_p && I_c >= I_qa_m;
    if (is_local_max) preserved++;
    samples.push({
      sample_index: r.sample_index,
      h_deg: r.h_deg,
      q_naive_deg: q_naive,
      I_center: I_c,
      is_local_max,
    });
  }
  const fraction = preserved / eligible.length;
  const ii_pass = fraction >= C3AB_II_PASS_THRESHOLD;
  return {
    receipt: "C3-A-B",
    title: "Kappa window — temptation real AND route optimum stays a feasible competing optimum",
    spec_reference: "P2_CUT2_C3A_NUMERIC_FREEZE.md §4 + Wave-4 audit-notes",
    kappa: KAPPA,
    sub_i_temptation_real: i_pass,
    sub_ii_route_peak_preserved: preserved,
    sub_ii_eligible_count: eligible.length,
    sub_ii_fraction_preserved: fraction,
    sub_ii_pass_threshold: C3AB_II_PASS_THRESHOLD,
    sub_ii_pass: ii_pass,
    pass: i_pass && ii_pass,
    samples_summary: samples.slice(0, 20), // first 20 only for readability
  };
}

// ===========================================================================
// Output + hashing
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

// ===========================================================================
// Main
// ===========================================================================

async function main() {
  const pin = buildPin(SEED);
  const fit = olsFitDecoyToH(pin);

  const pinPayload = {
    artifact: "C3-A P_in operational artifact (Wave-4)",
    spec_reference: "P2_CUT2_C3A_NUMERIC_FREEZE.md §2 + Wave-4 audit-notes",
    seed: SEED,
    sample_count: pin.length,
    h_grid: { min: H_GRID_MIN, max: H_GRID_MAX, step: H_GRID_STEP, count: NUM_H_GRID },
    eps_per_h: EPS_PER_H,
    draw_order: [
      "Phase A: 648 LCG uniforms → ε samples (1 uniform each)",
      "Phase B: 1296 LCG uniforms → d_sup_noise (Box-Muller, 2 uniforms each)",
      "Phase C: 1296 LCG uniforms → d_style_noise (Box-Muller, 2 uniforms each)",
    ],
    decoy_correlations: {
      d_sup: {
        form: "d_sup(h) = D_SUP_INTERCEPT + D_SUP_SLOPE · min(h, D_SUP_CLAMP_H) + N(0, D_SUP_NOISE_SD)",
        intercept_deg: D_SUP_INTERCEPT_DEG,
        slope_deg_per_deg: D_SUP_SLOPE_DEG_PER_DEG,
        noise_sd_deg: D_SUP_NOISE_SD_DEG,
        clamp_h_deg: D_SUP_CLAMP_H_DEG,
        provenance: "BOUNDARY_MAP L4: '~0.5° across h=0-22°' (linear fit). Clamp at 22° to NOT extrapolate the receipt.",
      },
      d_unanch_rendered_optional: {
        form: "1 iff h ≥ D_UNANCH_RO_THRESHOLD else 0",
        threshold_deg: D_UNANCH_RO_THRESHOLD_DEG,
        provenance: "Synthetic atlas-step nuisance prior. NOT strongly BOUNDARY_MAP-derived. Pre-registered as decoy structure.",
      },
      d_unanch_named_only: {
        form: "1 iff h ≥ D_UNANCH_NO_THRESHOLD else 0",
        threshold_deg: D_UNANCH_NO_THRESHOLD_DEG,
        provenance: "Synthetic atlas-step nuisance prior. NOT strongly BOUNDARY_MAP-derived.",
      },
      d_style: {
        form: "d_style(h) = D_STYLE_LINEAR · (h / 40) + D_STYLE_NOISE · N(0, 1)",
        linear_weight: D_STYLE_LINEAR_WEIGHT,
        noise_weight: D_STYLE_NOISE_WEIGHT,
        provenance: "Synthetic distributional nuisance. Pre-registered linear correlation strength.",
      },
    },
    ols_fit: {
      target: "h_deg",
      feature_names: ["intercept", ...fit.feature_names],
      coefficients: [fit.intercept, ...fit.weights],
      r_squared: fit.r_squared,
      residual_sd_deg: fit.residual_sd_deg,
      runtime_rule: "(w,b) are CONSTANTS at runtime; never refit. True h is read only at this pre-run fit step. The runtime decoy ridge D(q_h; d) reads (w,b) and observable d only.",
    },
    rows: pin,
  };

  const c3ar = buildC3ARReceipt(pin, fit);
  const c3at = buildC3ATReceipt(pin, fit);
  const c3ab = buildC3ABReceipt(pin, fit, c3at);

  const hashes = {};
  hashes.pin = await writeJsonAndHash(`${OUT_DIR_REL}/c3a-pin-generator.json`, pinPayload);
  hashes.r = await writeJsonAndHash(`${OUT_DIR_REL}/c3a-r-receipt.json`, c3ar);
  hashes.t = await writeJsonAndHash(`${OUT_DIR_REL}/c3a-t-receipt.json`, c3at);
  hashes.b = await writeJsonAndHash(`${OUT_DIR_REL}/c3a-b-receipt.json`, c3ab);

  // Markdown summary
  const lines = [];
  lines.push("# C3-A Receipts (Wave 4)");
  lines.push("");
  lines.push(`Generated by \`scripts/cut2-c3a-w4.mjs\`. Deterministic given seed \`${SEED}\`.`);
  lines.push("");
  lines.push("## Summary");
  lines.push("");
  lines.push("| receipt | status |");
  lines.push("| --- | --- |");
  lines.push(`| C3-A-R reachability (Path B quantile) | ${c3ar.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C3-A-T temptation + reversal | ${c3at.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C3-A-B kappa window | ${c3ab.pass ? "PASS" : "BLOCK"} |`);
  lines.push("");
  lines.push(`P_in: **${pin.length}** samples (${NUM_H_GRID} h-grid × ${EPS_PER_H} ε); OLS R² = **${fit.r_squared.toFixed(4)}**, residual σ = **${fit.residual_sd_deg.toFixed(3)}°**.`);
  lines.push("");
  lines.push("## C3-A-R (Reachability, Path B median rule)");
  lines.push("");
  lines.push(`F\\* = ${F_STAR_DEG}°. Pass rule: ≥${(C3AR_PASS_QUANTILE*100).toFixed(0)}% of P_in samples have shift > F\\*.`);
  lines.push("");
  lines.push(`Samples above F\\*: **${c3ar.samples_above_floor}/${c3ar.samples_total}** = **${(c3ar.fraction_above_floor*100).toFixed(1)}%**.`);
  lines.push("");
  lines.push(`Shift distribution: min=${c3ar.shift_distribution.min_deg.toExponential(2)}°, p10=${c3ar.shift_distribution.p10_deg.toExponential(2)}°, p25=${c3ar.shift_distribution.p25_deg.toExponential(2)}°, **median=${c3ar.shift_distribution.median_deg.toExponential(2)}°**, p75=${c3ar.shift_distribution.p75_deg.toExponential(2)}°, p90=${c3ar.shift_distribution.p90_deg.toExponential(2)}°, max=${c3ar.shift_distribution.max_deg.toFixed(3)}°.`);
  lines.push("");
  lines.push("## C3-A-T (Temptation + reversal)");
  lines.push("");
  lines.push(`Restricted to L1-eligible-by-obs (n=${c3at.L1_eligible_by_obs_count}).`);
  lines.push("");
  lines.push(`**T1 base temptation**: mean|π_dec − h| = **${c3at.T1.pi_dec_mean_error.toFixed(3)}°** vs mean|π_route − h| = **${c3at.T1.pi_route_mean_error.toFixed(3)}°**. Margin achieved = **${c3at.T1.margin_achieved_deg.toFixed(3)}°** (required ≥ ${M_DEG}°). ${c3at.T1.pass ? "PASS" : "BLOCK"}.`);
  lines.push("");
  lines.push(`**T2 decoy-edit reversal**: π_dec error after edit = ${c3at.T2_decoy_edit.pi_dec_mean_error_after.toFixed(3)}° (required ≥ τ_pc=${TAU_PC_DEG}°): ${c3at.T2_decoy_edit.pi_dec_pass ? "PASS" : "BLOCK"}. Route max shift = ${c3at.T2_decoy_edit.route_max_shift_deg.toFixed(3)}° (required ≤ ${ROUTE_INV_TOL_DEG}°): ${c3at.T2_decoy_edit.route_invariant_pass ? "PASS" : "BLOCK"}.`);
  lines.push("");
  lines.push(`**T3 handle-edit reversal**: π_route error to h' = ${c3at.T3_handle_edit.pi_route_mean_error.toFixed(3)}° (required ≤ τ2=${TAU2_DEG}°), π_dec stale error to h' = ${c3at.T3_handle_edit.pi_dec_mean_error.toFixed(3)}° (required > τ2): ${c3at.T3_handle_edit.pass ? "PASS" : "BLOCK"}.`);
  lines.push("");
  lines.push("## C3-A-B (Kappa window)");
  lines.push("");
  lines.push(`Sub-(i) temptation real: ${c3ab.sub_i_temptation_real ? "PASS" : "BLOCK"}.`);
  lines.push(`Sub-(ii) route argmax preserved as local max on L1-eligible-by-obs band: **${c3ab.sub_ii_route_peak_preserved}/${c3ab.sub_ii_eligible_count}** = **${(c3ab.sub_ii_fraction_preserved*100).toFixed(1)}%** (required ≥ ${(C3AB_II_PASS_THRESHOLD*100).toFixed(0)}%): ${c3ab.sub_ii_pass ? "PASS" : "BLOCK"}.`);
  lines.push("");
  const md = lines.join("\n") + "\n";
  const mdRel = `${OUT_DIR_REL}/c3a-w4-summary.md`;
  await writeFile(resolve(REPO, mdRel), md);
  const mdHash = sha256(Buffer.from(md));

  // Console output
  console.log("[c3a-w4] artifacts:");
  for (const [k, h] of Object.entries(hashes)) {
    console.log(`  ${h.path}`);
    console.log(`    raw       = ${h.raw_sha256}`);
    console.log(`    canonical = ${h.canonical_sha256}`);
  }
  console.log(`  ${mdRel}`);
  console.log(`    raw       = ${mdHash}`);
  console.log("");
  console.log("[c3a-w4] verdicts:");
  console.log(`  C3-A-R reachability (Path B): ${c3ar.pass ? "PASS" : "BLOCK"}`);
  console.log(`    ${c3ar.samples_above_floor}/${c3ar.samples_total} above F*=${F_STAR_DEG}° (${(c3ar.fraction_above_floor*100).toFixed(1)}%, required ≥ ${(C3AR_PASS_QUANTILE*100).toFixed(0)}%)`);
  console.log(`    shift median=${c3ar.shift_distribution.median_deg.toExponential(2)}° p75=${c3ar.shift_distribution.p75_deg.toExponential(2)}° p90=${c3ar.shift_distribution.p90_deg.toExponential(2)}°`);
  console.log(`  C3-A-T temptation+reversal : ${c3at.pass ? "PASS" : "BLOCK"}`);
  console.log(`    T1 margin=${c3at.T1.margin_achieved_deg.toFixed(3)}° (≥${M_DEG}°)=${c3at.T1.pass}, T2_dec_err=${c3at.T2_decoy_edit.pi_dec_mean_error_after.toFixed(3)}°=${c3at.T2_decoy_edit.pi_dec_pass}, T2_route_inv=${c3at.T2_decoy_edit.route_invariant_pass}, T3=${c3at.T3_handle_edit.pass}`);
  console.log(`  C3-A-B kappa window         : ${c3ab.pass ? "PASS" : "BLOCK"}`);
  console.log(`    (i)=${c3ab.sub_i_temptation_real}, (ii) ${c3ab.sub_ii_route_peak_preserved}/${c3ab.sub_ii_eligible_count}=${(c3ab.sub_ii_fraction_preserved*100).toFixed(1)}% (≥${(C3AB_II_PASS_THRESHOLD*100).toFixed(0)}%)=${c3ab.sub_ii_pass}`);
  console.log("");
  console.log(`[c3a-w4] OLS R²=${fit.r_squared.toFixed(4)} residual σ=${fit.residual_sd_deg.toFixed(3)}°`);
}

main().catch(err => {
  console.error(`[c3a-w4] FAILED: ${err.message}`);
  console.error(err.stack);
  process.exit(1);
});
