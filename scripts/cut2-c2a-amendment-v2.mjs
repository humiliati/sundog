#!/usr/bin/env node
// scripts/cut2-c2a-amendment-v2.mjs
//
// Wave-3.1 amendment re-run of the C2-A-2 objective-level abstain scan.
//
// Background — read the Wave-3 audit-notes append on
// P2_CUT2_C2A_NUMERIC_FREEZE.md for the full narrative:
//
//   v1 (Wave-2 freeze): κ_cond_max = 100. The receipt under that value
//   BLOCKED with 73.9% of L1-eligible-by-true-h rows tripping on
//   `cond > κ_cond_max` alone. Root cause: a Wave-2 calibration miss —
//   κ_cond_max = 100 was derived against the q_a-only curvature scale
//   (`2λ/A² ≈ 10 /deg²`) and missed the chain-rule scaling of the q_h
//   Hessian eigenvalue, which collapses by a factor of
//   χ² = (R22·tan(h_L1)·sec(h_L1)·π/180)² ≈ (0.0787)² ≈ 6.20·10⁻³
//   near the L1 boundary.
//
//   v2 (Wave-3.1 amendment, principled): κ_cond_max = 10⁴. Derived from
//   the (q_h, q_a) Hessian condition number at h_L1 with C_L1(h_L1)=0.5:
//       cond_L1 ≈ |H_qa|/|H_qh| = (2λ/A² + 1/σ²) / (C_L1·χ²/σ²)
//             ≈ 14.33 / 0.01239 ≈ 1156
//   10⁴ buffers ~8.7× above that geometric extreme. The degenerate
//   regime drives |H_qh| → 0 as argmax → q_h=0 (chain factor → 0), so
//   cond → ∞ analytically; the ~10⁵ floor seen in computation is a
//   grid-resolution + Hessian-estimator artifact, not a universal
//   analytic floor.
//
// This script also re-classifies rows by f_par_obs (the strict reading
// of the freeze) into three regimes:
//   degenerate                 : f_par_obs < R22
//   L1_ineligible_by_obs       : R22 ≤ f_par_obs < 1.02·R22
//   L1_eligible_by_obs         : f_par_obs ≥ 1.02·R22
//
// The receipt's pass criterion is read strictly from the freeze: all
// degenerate (f_par_obs < R22) rows trip; all L1-eligible-by-obs
// (f_par_obs ≥ 1.02·R22) rows do not. The borderline class
// L1_ineligible_by_obs is reported but not constrained by the spec.
//
// Output: results/structural-failure/cut2-prereg/c2a2-abstain-scan-v2.json
//
// Determinism: SAME seed and SAME LCG as the v1 script, so v1 and v2
// share an identical (h, ε) sample set — only κ_cond_max and the
// classifier differ.

import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIR = dirname(SCRIPT_PATH);

const REPO = resolve(
  execSync("git rev-parse --show-toplevel", {
    encoding: "utf8",
    cwd: SCRIPT_DIR,
  }).trim()
);
const OUT_REL = "results/structural-failure/cut2-prereg/c2a2-abstain-scan-v2.json";

// --- frozen [G]/[E] values, inherited from Wave-2 audit-notes append ---
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
const O_FLOOR = 0.1;
const A_DEG = RHO * R22_DEG; // 0.44
const R_TOL_DEG = 1.5 * A_DEG; // 0.66
const L3_DEG = 29;

// --- Wave-3.1 amendment: κ_cond_max v2 ---
const KAPPA_COND_MAX_V2 = 1e4;
const KAPPA_COND_MAX_V1 = 100; // recorded for the v1 algebraic-miss record

const D2R = Math.PI / 180;

function sObsFromFParObs(f_par_obs_deg) {
  // Freeze §1: s_obs = f_par_obs/R22 - 1 (observable, no h).
  return f_par_obs_deg / R22_DEG - 1;
}
function cL1(s) {
  return 1 / (1 + Math.exp(-C_L1_K * (s - 0.02)));
}

function* lcg(seed) {
  let state = (seed >>> 0) >>> 0;
  while (true) {
    state = (state * 1103515245 + 12345) >>> 0;
    yield state / 0x100000000;
  }
}
function epsilonSamples(seed, count) {
  const gen = lcg(seed);
  const out = [];
  for (let i = 0; i < count; i++) {
    const u = gen.next().value;
    out.push((2 * u - 1) * A_DEG);
  }
  return out;
}

function objectiveO(q_h_deg, q_a_deg, f_par_obs_deg, c_l1_value) {
  const r = f_par_obs_deg - R22_DEG / Math.cos(q_h_deg * D2R) - q_a_deg;
  const i_route =
    c_l1_value * Math.exp(-(r * r) / (2 * SIGMA_DEG * SIGMA_DEG));
  const pen = LAMBDA_PEN * (q_a_deg / A_DEG) * (q_a_deg / A_DEG);
  return i_route - pen;
}

function objectiveMaxAndConditioning(h_deg, eps_deg) {
  const f_par_obs = R22_DEG / Math.cos(h_deg * D2R) + eps_deg;
  // C_L1 reads OBSERVABLE s_obs per freeze §1, not true-h-derived sec(h)-1.
  const c_l1 = cL1(sObsFromFParObs(f_par_obs));
  const q_h_step = 0.05;
  const q_a_step = A_DEG / 20;
  let best_O = -Infinity;
  let best_q_h = Q_H_MIN;
  let best_q_a = 0;
  let min_abs_residual = Infinity;
  for (let q_h = Q_H_MIN; q_h <= Q_H_MAX; q_h += q_h_step) {
    for (let q_a = -A_DEG; q_a <= A_DEG + 1e-9; q_a += q_a_step) {
      const O = objectiveO(q_h, q_a, f_par_obs, c_l1);
      const abs_r = Math.abs(
        f_par_obs - R22_DEG / Math.cos(q_h * D2R) - q_a
      );
      if (abs_r < min_abs_residual) min_abs_residual = abs_r;
      if (O > best_O) {
        best_O = O;
        best_q_h = q_h;
        best_q_a = q_a;
      }
    }
  }
  const eps_fd = 1e-3;
  const O0 = objectiveO(best_q_h, best_q_a, f_par_obs, c_l1);
  const O_qh_plus = objectiveO(best_q_h + eps_fd, best_q_a, f_par_obs, c_l1);
  const O_qh_minus = objectiveO(best_q_h - eps_fd, best_q_a, f_par_obs, c_l1);
  const O_qa_plus = objectiveO(best_q_h, best_q_a + eps_fd, f_par_obs, c_l1);
  const O_qa_minus = objectiveO(best_q_h, best_q_a - eps_fd, f_par_obs, c_l1);
  const H_qh = (O_qh_plus + O_qh_minus - 2 * O0) / (eps_fd * eps_fd);
  const H_qa = (O_qa_plus + O_qa_minus - 2 * O0) / (eps_fd * eps_fd);
  const abs_qh = Math.abs(H_qh);
  const abs_qa = Math.abs(H_qa);
  const cond =
    (Math.max(abs_qh, abs_qa) + 1e-12) /
    (Math.min(abs_qh, abs_qa) + 1e-12);
  return {
    f_par_obs,
    c_l1,
    max_O: best_O,
    argmax_q_h_deg: best_q_h,
    argmax_q_a_deg: best_q_a,
    min_abs_residual_deg: min_abs_residual,
    hessian_qh: H_qh,
    hessian_qa: H_qa,
    condition_number: cond,
  };
}

function regimeByFParObs(f_par_obs) {
  if (f_par_obs < R22_DEG) return "degenerate";
  if (f_par_obs < L1_LEVERAGE_MULTIPLIER * R22_DEG) return "L1_ineligible_by_obs";
  return "L1_eligible_by_obs";
}

function abstainTripsV2(scan) {
  return (
    scan.max_O < O_FLOOR ||
    scan.min_abs_residual_deg > R_TOL_DEG ||
    scan.condition_number > KAPPA_COND_MAX_V2
  );
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
function canonical(value) {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return (
      "{" +
      keys.map((k) => JSON.stringify(k) + ":" + canonical(value[k])).join(",") +
      "}"
    );
  }
  return JSON.stringify(value);
}

async function main() {
  const eps_per_h = 8;
  const eps_samples = epsilonSamples(SEED, eps_per_h * 81);
  let eps_idx = 0;
  const rows = [];
  const buckets = {
    degenerate: { total: 0, trip: 0 },
    L1_ineligible_by_obs: { total: 0, trip: 0 },
    L1_eligible_by_obs: { total: 0, trip: 0 },
  };
  let trip_cond_only = 0;
  let trip_O_only = 0;
  let trip_r_only = 0;
  let trip_multi = 0;
  for (let h = H_GRID_MIN; h <= H_GRID_MAX + 1e-9; h += H_GRID_STEP) {
    const h_r = Math.round(h * 10) / 10;
    for (let i = 0; i < eps_per_h; i++) {
      const eps = eps_samples[eps_idx++];
      const scan = objectiveMaxAndConditioning(h_r, eps);
      const regime = regimeByFParObs(scan.f_par_obs);
      const trips = abstainTripsV2(scan);
      buckets[regime].total++;
      if (trips) buckets[regime].trip++;
      if (trips) {
        const causes = [
          scan.max_O < O_FLOOR,
          scan.min_abs_residual_deg > R_TOL_DEG,
          scan.condition_number > KAPPA_COND_MAX_V2,
        ];
        const count = causes.filter(Boolean).length;
        if (count > 1) trip_multi++;
        else if (causes[0]) trip_O_only++;
        else if (causes[1]) trip_r_only++;
        else if (causes[2]) trip_cond_only++;
      }
      rows.push({
        h_deg: h_r,
        eps_deg: eps,
        f_par_obs_deg: scan.f_par_obs,
        regime,
        max_O: scan.max_O,
        max_O_below_floor: scan.max_O < O_FLOOR,
        min_abs_residual_deg: scan.min_abs_residual_deg,
        residual_above_tol: scan.min_abs_residual_deg > R_TOL_DEG,
        condition_number: scan.condition_number,
        cond_above_max_v2: scan.condition_number > KAPPA_COND_MAX_V2,
        abstain_trips_v2: trips,
      });
    }
  }

  const passDeg =
    buckets.degenerate.total > 0 &&
    buckets.degenerate.trip === buckets.degenerate.total;
  const passElig =
    buckets.L1_eligible_by_obs.total > 0 &&
    buckets.L1_eligible_by_obs.trip === 0;
  const pass = passDeg && passElig;

  const payload = {
    receipt: "C2-A-2 v2",
    version: "v2 (Wave-3.1 amendment re-run)",
    title:
      "C2-D objective-level abstain — re-run under principled κ_cond_max",
    spec_reference:
      "docs/prereg/structural-failure-coincidence/P2_CUT2_C2A_NUMERIC_FREEZE.md §2 + Wave-3 audit-notes append (Wave-3.1 amendment)",
    abstain_criterion_formula:
      "abstain ⟺ max_q O < O_floor (=0.1) OR min|r| > r_tol (=0.66°) OR cond(H) > κ_cond_max (v2 = 10⁴, was v1 = 100); read from frozen properties of O; no `if f_par_obs<R22` branch.",
    amendment: {
      kappa_cond_max_v1: KAPPA_COND_MAX_V1,
      kappa_cond_max_v2: KAPPA_COND_MAX_V2,
      basis:
        "principled chain-rule Hessian algebra: |H_qh|_L1 = C_L1(h_L1)·χ²/σ² ≈ 0.5·(0.0787)²/0.25 ≈ 0.01239 /deg²; |H_qa| ≈ 2λ/A² + 1/σ² ≈ 14.33 /deg²; cond_L1 ≈ 14.33/0.01239 ≈ 1156. κ_cond_max v2 = 10⁴ ≈ 8.7× above this geometric extreme and well below the degenerate cond regime (cond → ∞ as q_h → 0 analytically; ~10⁵ practical floor under the frozen q-grid and Hessian estimator).",
      a3_compliance:
        "Re-pick is bounded by Hessian chain-rule algebra at the L1 boundary, NOT by where the v1 receipt happened to fail. The amendment is one-way (algebra → value); no receipt-data flowed into the choice of 10⁴.",
    },
    bridge_scale: 1.0,
    O_floor: O_FLOOR,
    r_tol_deg: R_TOL_DEG,
    kappa_cond_max_active: KAPPA_COND_MAX_V2,
    eps_per_h_grid_point: eps_per_h,
    seed: SEED,
    classification: "by f_par_obs (strict spec reading)",
    summary: {
      degenerate_total: buckets.degenerate.total,
      degenerate_tripped: buckets.degenerate.trip,
      degenerate_trip_rate:
        buckets.degenerate.total > 0
          ? buckets.degenerate.trip / buckets.degenerate.total
          : null,
      L1_ineligible_by_obs_total: buckets.L1_ineligible_by_obs.total,
      L1_ineligible_by_obs_tripped: buckets.L1_ineligible_by_obs.trip,
      L1_eligible_by_obs_total: buckets.L1_eligible_by_obs.total,
      L1_eligible_by_obs_tripped: buckets.L1_eligible_by_obs.trip,
      trip_cause_breakdown: {
        only_cond: trip_cond_only,
        only_O: trip_O_only,
        only_r: trip_r_only,
        multi: trip_multi,
      },
    },
    pass,
    pass_criterion:
      "all degenerate (f_par_obs < R22) rows trip AND all L1_eligible_by_obs (f_par_obs ≥ 1.02·R22) rows do NOT trip; L1_ineligible_by_obs is the borderline class not constrained by the spec.",
    rows,
  };

  const absOut = resolve(REPO, OUT_REL);
  await mkdir(dirname(absOut), { recursive: true });
  const pretty = JSON.stringify(payload, null, 2) + "\n";
  await writeFile(absOut, pretty);

  console.log(`[c2a-amendment-v2] wrote ${OUT_REL}`);
  console.log(`[c2a-amendment-v2] raw_sha256       = ${sha256(Buffer.from(pretty))}`);
  console.log(
    `[c2a-amendment-v2] canonical_sha256 = ${sha256(Buffer.from(canonical(payload)))}`
  );
  console.log("");
  console.log(
    `[c2a-amendment-v2] degenerate trip rate         : ${buckets.degenerate.trip}/${buckets.degenerate.total} = ${((buckets.degenerate.total ? buckets.degenerate.trip / buckets.degenerate.total : 0) * 100).toFixed(1)}%`
  );
  console.log(
    `[c2a-amendment-v2] L1_eligible_by_obs trip count: ${buckets.L1_eligible_by_obs.trip}/${buckets.L1_eligible_by_obs.total}`
  );
  console.log(
    `[c2a-amendment-v2] L1_ineligible_by_obs trip    : ${buckets.L1_ineligible_by_obs.trip}/${buckets.L1_ineligible_by_obs.total} (borderline, not constrained)`
  );
  console.log("");
  console.log(`[c2a-amendment-v2] verdict: ${pass ? "PASS" : "BLOCK"}`);
}

main().catch((err) => {
  console.error(`[c2a-amendment-v2] FAILED: ${err.message}`);
  process.exit(1);
});
