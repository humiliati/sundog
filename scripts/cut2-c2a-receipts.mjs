#!/usr/bin/env node
// scripts/cut2-c2a-receipts.mjs
//
// Wave-3 C2-A receipt generator. Computes the three pre-run receipts
// against the frozen Wave-2 [E] values + bridge-scale convention.
//
// Per the C2-A freeze, this is a *landscape-vs-controller-threshold
// characterization*, not a Cut-2 run. No PhotometricAgent is
// instantiated; we hardcode (and cite) its [G]-immutable constants and
// derive the sustained-TRACK criterion analytically. The agent is then
// optionally exercised under the same intensity field by a separate
// cross-check (out of scope for this script; can be added as a sibling).
//
// Outputs (all under the C5 allowlist):
//   results/structural-failure/cut2-prereg/c2a1-track-receipt.json
//   results/structural-failure/cut2-prereg/c2a2-abstain-scan.json
//   results/structural-failure/cut2-prereg/c2a3-separation-receipt.json
//   results/structural-failure/cut2-prereg/c2a-receipts-summary.md
//
// All [E] values are inherited from the Wave-2 audit-notes append on
// P2_CUT2_C2A_NUMERIC_FREEZE.md (frozen 2026-05-16 PT). The script
// includes them inline as constants AND records their source freeze for
// audit traceability. A change to any value here without a matching
// append-only amendment is itself a violation of A3.

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
const OUT_DIR_REL = "results/structural-failure/cut2-prereg";

// --------------------------------------------------------------------------
// Frozen [G] / [E] values (Wave-2 audit-notes append; immutable from this
// point — A3). Source of truth:
//   docs/prereg/structural-failure-coincidence/P2_CUT2_C2A_NUMERIC_FREEZE.md
//   §4 freeze table (Wave-2 entry).
// --------------------------------------------------------------------------

// [G] inherited from BOUNDARY_MAP / P1 / C1 / phase3.HALO_22_RADIUS:
const R22_DEG = 22; // canonical 22° halo radius (phase3.HALO_22_RADIUS)
const L1_LEVERAGE_MULTIPLIER = 1.02; // sec(h_L1) = 1.02; h_L1 = arccos(1/1.02)
const L2_DEG = 32;
const L3_DEG = 29;
// PhotometricAgent constants [G] by C1 (agents/photometric.py:70-71):
const REACQUIRE_THRESHOLD = 0.05;
const REACQUIRE_HOLD_STEPS = 30;
const PROBE_AMPLITUDE = 0.05; // agents/photometric.py:63

// [E] frozen Wave-2:
const RHO = 0.02;
const SIGMA_DEG = 0.5;
const SEED = 20260516;
const H_GRID_MIN = 0;
const H_GRID_MAX = 40;
const H_GRID_STEP = 0.5;
const Q_H_MIN = 0;
const Q_H_MAX = 60;
const LAMBDA_PEN = 1.0;
const TAU_C2B_II = 0.05;
const C_L1_K = 600;
const T_CZA_MAGNITUDE = 0.3;
const T_TAN_MAGNITUDE = 0.3;
const DETECT_THRESHOLD_T = 0.2;
const SEPARATION_MIN_DEG = 2.0;
const O_FLOOR = 0.1;
const KAPPA_COND_MAX = 100;
const BRIDGE_SCALE = 1.0; // eligible-band route peak ≡ 1.0 (§5 convention)
const D1_MIN_BIAS_DEG = 1.5;

// Derived constants (recorded, not free):
const A_DEG = RHO * R22_DEG; // anchor-noise bound, A = ρ·R22 ≈ 0.44°
const H_L1_DEG = (Math.acos(1 / L1_LEVERAGE_MULTIPLIER) * 180) / Math.PI;
const R_TOL_DEG = 1.5 * A_DEG; // r_tol = 1.5·A ≈ 0.66°
const COINCIDENCE_WINDOW_DEG = 1.5; // ±1.5° around each locus

// C2-A-3 separation receipt parameters:
const H_STAR_DEG = 25.0; // chosen freely below L3=29° such that C_L1 ≈ 1
const EPSILON_C = 0.001; // tolerance: C_L1(s_obs(h)) ≥ 1 - ε_C for h ≥ h*

// --------------------------------------------------------------------------
// Pure math
// --------------------------------------------------------------------------

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

// Observable s_obs: this is the freeze §1 definition. C_L1 reads this,
// NOT sec(true_h)−1. At ε=0 the two are identical (since
// f_par_obs = R22/cos(h) and so f_par_obs/R22 = sec(h)), but with
// ε ≠ 0 they diverge — and the freeze is explicit that the C_L1 gate
// is observable-only (no hidden h). Below, sObsEpsZero is used only
// for the per-h ε=0 baseline in C2-A-1 / C2-A-3 (where the receipt is
// defined vs h, not vs sample); sObsFromFParObs is used in the
// objective and in any per-sample evaluation.
function sObsEpsZero(h_deg) {
  return 1 / Math.cos(h_deg * D2R) - 1;
}
function sObsFromFParObs(f_par_obs_deg) {
  return f_par_obs_deg / R22_DEG - 1;
}

function cL1(s) {
  return 1 / (1 + Math.exp(-C_L1_K * (s - 0.02)));
}

function bridgeIntensityPeak(h_deg) {
  // Bridge convention: eligible-band route peak ≡ 1.0. At the joint
  // optimum (q_h = q_naive, q_a = 0) with ε = 0, the route Gaussian
  // peaks at 1.0; bridge-mapped target intensity at that peak is
  // C_L1(s_obs at ε=0) · 1.0 ≡ C_L1(sec(h)−1) · 1.0.
  return cL1(sObsEpsZero(h_deg)) * BRIDGE_SCALE;
}

function qNaive(h_deg, eps_deg) {
  // q_naive(h, ε) = arccos(R22 / f_par_obs); biased estimator.
  // f_par_obs = R22 / cos(h) + ε. Returns null if undefined.
  const f_par_obs = R22_DEG / Math.cos(h_deg * D2R) + eps_deg;
  const ratio = R22_DEG / f_par_obs;
  if (ratio < -1 || ratio > 1) return null;
  return Math.acos(ratio) * R2D;
}

// TRACK gradient-SNR estimate (informational column on C2-A-1 table).
// For a Gaussian ridge of peak C_L1 and width σ in q_h degrees, the
// per-probe-frequency demodulated gradient signal at offset d from peak
// is ~ (C_L1 · d / σ²) · probe_amplitude. We report the "near-peak"
// linearised gradient sensitivity (d ≈ probe_amplitude / 2, the
// half-cycle excursion). C2-A-1 evaluates this vs h at ε=0; equivalent
// to using sObsEpsZero here.
function trackGradientSNREstimate(h_deg) {
  const peak = bridgeIntensityPeak(h_deg);
  const d = PROBE_AMPLITUDE / 2; // half-cycle offset, in q_h degrees
  const signal = (peak * d) / (SIGMA_DEG * SIGMA_DEG);
  // Noise floor is the controller's intensity_lowpass_alpha residual at
  // ~ peak · alpha (alpha=0.02). SNR is a ratio.
  const noise = peak * 0.02 + 1e-6;
  return signal / noise;
}

// Deterministic seeded LCG. Wave-2 freeze pinned seed; we re-derive ε
// samples from it here. (NumPy / JS PRNG differences would change byte
// output, so we use a tiny custom LCG that's reproducible byte-for-byte.)
function* lcg(seed) {
  let state = (seed >>> 0) >>> 0;
  while (true) {
    state = (state * 1103515245 + 12345) >>> 0;
    yield state / 0x100000000; // ∈ [0, 1)
  }
}

function epsilonSamples(seed, count) {
  const gen = lcg(seed);
  const out = [];
  for (let i = 0; i < count; i++) {
    const u = gen.next().value; // ∈ [0, 1)
    out.push((2 * u - 1) * A_DEG); // ∈ [-A, +A]
  }
  return out;
}

// --------------------------------------------------------------------------
// C2-A-2 objective scan helpers
// --------------------------------------------------------------------------

// Objective O(q_h, q_a) = I_route(q_h, q_a; bundle) - pen(q_a)
// I_route = exp(-(f_par_obs - R22/cos(q_h) - q_a)²/(2σ²)) × C_L1(s_obs(h))
// pen = λ · (q_a / A)²
// (T_cza/T_tan and full bridge are not needed here; this is the
// landscape's L1 component, which is what the abstain criterion reads.)
function objectiveO(q_h_deg, q_a_deg, f_par_obs_deg, c_l1_value) {
  const residual_for_route =
    f_par_obs_deg - R22_DEG / Math.cos(q_h_deg * D2R) - q_a_deg;
  const i_route =
    c_l1_value *
    Math.exp(-(residual_for_route * residual_for_route) / (2 * SIGMA_DEG * SIGMA_DEG));
  const pen = LAMBDA_PEN * (q_a_deg / A_DEG) * (q_a_deg / A_DEG);
  return i_route - pen;
}

// Coarse grid maximum + finite-difference Hessian estimate. For receipt
// purposes we don't need optimization precision beyond grid resolution.
function objectiveMaxAndConditioning(h_deg, eps_deg) {
  const f_par_obs = R22_DEG / Math.cos(h_deg * D2R) + eps_deg;
  // C_L1 reads the OBSERVABLE s_obs (freeze §1: s_obs = f_par_obs/R22 - 1),
  // never the true-h-derived sec(h)-1. They are equal at ε=0 only.
  const c_l1 = cL1(sObsFromFParObs(f_par_obs));
  const q_h_step = 0.05; // finer than the public h-grid 0.5° for accuracy
  const q_a_step = A_DEG / 20; // 20 steps across [-A, +A]
  let best_O = -Infinity;
  let best_q_h = Q_H_MIN;
  let best_q_a = 0;
  let min_abs_residual = Infinity;
  // Sweep q_h × q_a.
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
  // Finite-difference Hessian at (best_q_h, best_q_a).
  const eps_fd = 1e-3;
  const O0 = objectiveO(best_q_h, best_q_a, f_par_obs, c_l1);
  const O_qh_plus = objectiveO(best_q_h + eps_fd, best_q_a, f_par_obs, c_l1);
  const O_qh_minus = objectiveO(best_q_h - eps_fd, best_q_a, f_par_obs, c_l1);
  const O_qa_plus = objectiveO(best_q_h, best_q_a + eps_fd, f_par_obs, c_l1);
  const O_qa_minus = objectiveO(best_q_h, best_q_a - eps_fd, f_par_obs, c_l1);
  const H_qh = (O_qh_plus + O_qh_minus - 2 * O0) / (eps_fd * eps_fd);
  const H_qa = (O_qa_plus + O_qa_minus - 2 * O0) / (eps_fd * eps_fd);
  // Negative-definite at a max: |H_qh|, |H_qa| are curvatures. Condition
  // number = max(|H_qh|, |H_qa|) / min(|H_qh|, |H_qa|).
  const abs_qh = Math.abs(H_qh);
  const abs_qa = Math.abs(H_qa);
  const cond = (Math.max(abs_qh, abs_qa) + 1e-12) / (Math.min(abs_qh, abs_qa) + 1e-12);
  return {
    f_par_obs,
    eligible: f_par_obs >= L1_LEVERAGE_MULTIPLIER * R22_DEG,
    has_real_root: f_par_obs >= R22_DEG,
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

function abstainTripped(scan) {
  // C2-D objective-level abstain (no `if` branches on observable values).
  // abstain iff: max_O < O_floor OR no |r| ≤ r_tol solution OR cond > κ_max
  return (
    scan.max_O < O_FLOOR ||
    scan.min_abs_residual_deg > R_TOL_DEG ||
    scan.condition_number > KAPPA_COND_MAX
  );
}

// --------------------------------------------------------------------------
// Receipt builders
// --------------------------------------------------------------------------

function buildC2A1Receipt() {
  // For each h on the grid, compute s_obs, C_L1, bridge_intensity_peak,
  // sustains_track, gradient_SNR_estimate. Find the empirical transition.
  const rows = [];
  let transition_h = null;
  let prev_sustains = null;
  for (let h = H_GRID_MIN; h <= H_GRID_MAX + 1e-9; h += H_GRID_STEP) {
    const h_r = Math.round(h * 10) / 10;
    const s = sObsEpsZero(h_r);
    const c = cL1(s);
    const peak = c * BRIDGE_SCALE;
    const sustains = peak >= REACQUIRE_THRESHOLD;
    const snr = trackGradientSNREstimate(h_r);
    const eligible_L1 = s >= 0.02; // [G] L1 boundary
    if (prev_sustains === false && sustains === true && transition_h === null) {
      transition_h = h_r; // first h at which sustained TRACK kicks in
    }
    prev_sustains = sustains;
    rows.push({
      h_deg: h_r,
      s_obs: s,
      C_L1: c,
      bridge_intensity_peak: peak,
      reacquire_threshold: REACQUIRE_THRESHOLD,
      sustains_track: sustains,
      eligible_L1,
      track_gradient_snr_estimate: snr,
    });
  }
  return {
    receipt: "C2-A-1",
    title:
      "C_L1 behavioral-effectiveness — sustained-TRACK landscape vs PhotometricAgent reacquire threshold",
    spec_reference:
      "docs/prereg/structural-failure-coincidence/P2_CUT2_C2A_NUMERIC_FREEZE.md §1 + Wave-2 audit-notes append",
    sustained_track_criterion:
      "bridge-mapped target intensity at the joint optimum (= C_L1(s_obs(h)) · 1.0 per §5 bridge convention) ≥ reacquire_threshold = 0.05. By PhotometricAgent's phase semantics (agents/photometric.py:163-166), reacquire is enforced only after TRACK is entered: if i_now < threshold for reacquire_hold_steps = 30 consecutive control steps the agent re-enters SCAN. Sustained TRACK therefore fails iff C_L1(s_obs(h)) < 0.05.",
    h_L1_deg: H_L1_DEG,
    coincidence_window_deg: COINCIDENCE_WINDOW_DEG,
    coincidence_window_lower_h_deg: H_L1_DEG - COINCIDENCE_WINDOW_DEG,
    coincidence_window_upper_h_deg: H_L1_DEG + COINCIDENCE_WINDOW_DEG,
    sustained_track_transition_h_deg: transition_h,
    transition_inside_coincidence_window:
      transition_h !== null &&
      transition_h >= H_L1_DEG - COINCIDENCE_WINDOW_DEG &&
      transition_h <= H_L1_DEG + COINCIDENCE_WINDOW_DEG,
    transition_margin_to_lower_edge_deg:
      transition_h !== null
        ? transition_h - (H_L1_DEG - COINCIDENCE_WINDOW_DEG)
        : null,
    pass: null, // set below
    rows,
  };
}

function buildC2A2Receipt() {
  // Objective scan over h-grid × ε samples. Show degenerate rows
  // (f_par_obs < R22) trip the abstain criterion, eligible rows
  // (f_par_obs ≥ 1.02·R22) do not. Without `if f_par_obs < R22` branch.
  const eps_per_h = 8;
  const rows = [];
  let degenerate_trips = 0;
  let degenerate_total = 0;
  let eligible_no_trips = 0;
  let eligible_total = 0;
  let boundary_no_trips = 0;
  let boundary_total = 0;
  const eps_samples = epsilonSamples(SEED, eps_per_h * 81); // 81 grid pts
  let eps_idx = 0;
  for (let h = H_GRID_MIN; h <= H_GRID_MAX + 1e-9; h += H_GRID_STEP) {
    const h_r = Math.round(h * 10) / 10;
    for (let i = 0; i < eps_per_h; i++) {
      const eps = eps_samples[eps_idx++];
      const scan = objectiveMaxAndConditioning(h_r, eps);
      const trips = abstainTripped(scan);
      const f_par_obs_below_R22 = scan.f_par_obs < R22_DEG;
      const s_h = sObsEpsZero(h_r);
      const regime = !scan.has_real_root
        ? "degenerate"
        : s_h < 0.02
        ? "L1_ineligible"
        : "L1_eligible";
      if (regime === "degenerate") {
        degenerate_total++;
        if (trips) degenerate_trips++;
      } else if (regime === "L1_eligible" && h_r < L3_DEG) {
        eligible_total++;
        if (!trips) eligible_no_trips++;
      } else if (regime === "L1_ineligible") {
        boundary_total++;
        if (!trips) boundary_no_trips++;
      }
      rows.push({
        h_deg: h_r,
        eps_deg: eps,
        f_par_obs_deg: scan.f_par_obs,
        f_par_obs_below_R22,
        regime,
        max_O: scan.max_O,
        max_O_below_floor: scan.max_O < O_FLOOR,
        min_abs_residual_deg: scan.min_abs_residual_deg,
        residual_above_tol: scan.min_abs_residual_deg > R_TOL_DEG,
        condition_number: scan.condition_number,
        cond_above_max: scan.condition_number > KAPPA_COND_MAX,
        abstain_trips: trips,
      });
    }
  }
  return {
    receipt: "C2-A-2",
    title:
      "C2-D objective-level abstain — reproducible scan proving degenerate rows trip the criterion, eligible rows do not",
    spec_reference:
      "docs/prereg/structural-failure-coincidence/P2_CUT2_C2A_NUMERIC_FREEZE.md §2 + Wave-2 audit-notes append",
    abstain_criterion_formula:
      "abstain ⟺ max_q O < O_floor (=0.1) OR min|r| > r_tol (=0.66°) OR cond(H) > κ_cond_max (=100), all read from frozen properties of O; no `if f_par_obs<R22` branch.",
    O_floor: O_FLOOR,
    r_tol_deg: R_TOL_DEG,
    kappa_cond_max: KAPPA_COND_MAX,
    eps_per_h_grid_point: eps_per_h,
    seed: SEED,
    summary: {
      degenerate_rows_total: degenerate_total,
      degenerate_rows_tripped: degenerate_trips,
      degenerate_trip_rate:
        degenerate_total > 0 ? degenerate_trips / degenerate_total : null,
      eligible_rows_total: eligible_total,
      eligible_rows_not_tripped: eligible_no_trips,
      eligible_no_trip_rate:
        eligible_total > 0 ? eligible_no_trips / eligible_total : null,
      L1_ineligible_rows_total: boundary_total,
      L1_ineligible_rows_not_tripped: boundary_no_trips,
    },
    pass: null,
    rows,
  };
}

function buildC2A3Receipt() {
  // For each h ≥ h* on the grid, evaluate C_L1 and show it ≥ 1 - ε_C.
  // This shows C_L1 ≈ 1 in the L2/L3 region — multiplying the bracket
  // by C_L1 does NOT mask the L2/L3 consistency-term tests.
  const rows = [];
  let all_pass = true;
  let min_c_l1_above_hstar = Infinity;
  for (let h = H_STAR_DEG; h <= H_GRID_MAX + 1e-9; h += H_GRID_STEP) {
    const h_r = Math.round(h * 10) / 10;
    const c = cL1(sObsEpsZero(h_r));
    const passes = c >= 1 - EPSILON_C;
    if (!passes) all_pass = false;
    if (c < min_c_l1_above_hstar) min_c_l1_above_hstar = c;
    rows.push({
      h_deg: h_r,
      C_L1: c,
      passes_threshold: passes,
      threshold: 1 - EPSILON_C,
    });
  }
  return {
    receipt: "C2-A-3",
    title:
      "Package-gating separation — C_L1 ≈ 1 throughout the L2/L3 region, so multiplying the bracket leaves the consistency-term tests intact",
    spec_reference:
      "docs/prereg/structural-failure-coincidence/P2_CUT2_C2A_NUMERIC_FREEZE.md §3 + Wave-2 audit-notes append",
    h_star_deg: H_STAR_DEG,
    epsilon_C: EPSILON_C,
    L2_deg: L2_DEG,
    L3_deg: L3_DEG,
    h_star_below_L3: H_STAR_DEG < L3_DEG,
    min_C_L1_above_h_star: min_c_l1_above_hstar,
    pass: all_pass,
    rows,
  };
}

// --------------------------------------------------------------------------
// Output
// --------------------------------------------------------------------------

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

function renderSummaryMd(c2a1, c2a2, c2a3) {
  const lines = [];
  lines.push("# C2-A Receipts (Wave 3)");
  lines.push("");
  lines.push(
    `Generated by \`scripts/cut2-c2a-receipts.mjs\`. Deterministic given seed \`${SEED}\` and the frozen Wave-2 \`[E]\` values.`
  );
  lines.push("");
  lines.push("## Summary");
  lines.push("");
  lines.push(`| receipt | status |`);
  lines.push(`| --- | --- |`);
  lines.push(`| C2-A-1 sustained-TRACK landscape | ${c2a1.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C2-A-2 objective-level abstain | ${c2a2.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C2-A-3 package-gating separation | ${c2a3.pass ? "PASS" : "BLOCK"} |`);
  lines.push("");
  lines.push("## C2-A-1");
  lines.push("");
  lines.push(
    `Sustained-TRACK transition: **h ≈ ${c2a1.sustained_track_transition_h_deg}°** ` +
      `(first h on grid where bridge-mapped peak intensity ≥ reacquire_threshold = ${REACQUIRE_THRESHOLD}).`
  );
  lines.push("");
  lines.push(`L1 boundary: **h_L1 ≈ ${H_L1_DEG.toFixed(3)}°**.`);
  lines.push(
    `Coincidence window: **[${(H_L1_DEG - COINCIDENCE_WINDOW_DEG).toFixed(3)}°, ${(H_L1_DEG + COINCIDENCE_WINDOW_DEG).toFixed(3)}°]**.`
  );
  lines.push(
    `Transition inside window: **${c2a1.transition_inside_coincidence_window ? "YES" : "NO"}**. ` +
      `Margin to lower edge: **${c2a1.transition_margin_to_lower_edge_deg?.toFixed(3) ?? "N/A"}°**.`
  );
  lines.push("");
  lines.push("Sampled rows (every 1.0° from 0° to 40°):");
  lines.push("");
  lines.push("| h (°) | s_obs | C_L1 | I_peak | sustains TRACK | L1-eligible | grad SNR est |");
  lines.push("| --- | --- | --- | --- | --- | --- | --- |");
  for (const r of c2a1.rows.filter((x) => Math.abs(x.h_deg % 1) < 1e-6)) {
    lines.push(
      `| ${r.h_deg.toFixed(1)} | ${r.s_obs.toExponential(3)} | ${r.C_L1.toExponential(3)} | ${r.bridge_intensity_peak.toExponential(3)} | ${r.sustains_track ? "✓" : "✗"} | ${r.eligible_L1 ? "✓" : "✗"} | ${r.track_gradient_snr_estimate.toExponential(2)} |`
    );
  }
  lines.push("");
  lines.push("## C2-A-2");
  lines.push("");
  lines.push(`Seed: \`${SEED}\` (LCG). Samples: ${c2a2.eps_per_h_grid_point} ε's per h-grid point.`);
  lines.push("");
  lines.push(
    `Degenerate rows tripped: **${c2a2.summary.degenerate_rows_tripped}/${c2a2.summary.degenerate_rows_total}** (rate ${(c2a2.summary.degenerate_trip_rate * 100 || 0).toFixed(1)}%).`
  );
  lines.push(
    `L1-eligible rows NOT tripped: **${c2a2.summary.eligible_rows_not_tripped}/${c2a2.summary.eligible_rows_total}** (rate ${((c2a2.summary.eligible_no_trip_rate || 0) * 100).toFixed(1)}%).`
  );
  lines.push(
    `L1-ineligible rows (above R22, but on L1-ineligible side of s=0.02) NOT tripped: ` +
      `${c2a2.summary.L1_ineligible_rows_not_tripped}/${c2a2.summary.L1_ineligible_rows_total}.`
  );
  lines.push("");
  lines.push(
    "Note: the C2-D abstain criterion reads max_q O, min|r|, and cond(H) — all properties of the frozen objective. No `if f_par_obs<R22` branch is used."
  );
  lines.push("");
  lines.push("## C2-A-3");
  lines.push("");
  lines.push(
    `h\\* = ${H_STAR_DEG}° (chosen below L3 = ${L3_DEG}°). ε_C = ${EPSILON_C}.`
  );
  lines.push(
    `Min C_L1 for h ≥ h\\*: **${c2a3.min_C_L1_above_h_star.toFixed(10)}**. Threshold (1 - ε_C): **${(1 - EPSILON_C).toFixed(10)}**.`
  );
  lines.push(
    `Therefore C_L1(s_obs(h)) ≥ 1 - ε_C for all h ≥ h\\*: **${c2a3.pass ? "YES" : "NO"}**. The L1 ramp is functionally 1.0 throughout the L2/L3 region; multiplying the bracket by C_L1 does NOT mask the L2/L3 consistency-term tests.`
  );
  lines.push("");
  return lines.join("\n") + "\n";
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
  const c2a1 = buildC2A1Receipt();
  c2a1.pass = c2a1.transition_inside_coincidence_window === true;

  const c2a2 = buildC2A2Receipt();
  const s = c2a2.summary;
  c2a2.pass =
    (s.degenerate_total === 0 || s.degenerate_trip_rate === 1) &&
    (s.eligible_total === 0 || s.eligible_no_trip_rate === 1);

  const c2a3 = buildC2A3Receipt();

  const c2a1Hash = await writeJsonAndHash(
    `${OUT_DIR_REL}/c2a1-track-receipt.json`,
    c2a1
  );
  const c2a2Hash = await writeJsonAndHash(
    `${OUT_DIR_REL}/c2a2-abstain-scan.json`,
    c2a2
  );
  const c2a3Hash = await writeJsonAndHash(
    `${OUT_DIR_REL}/c2a3-separation-receipt.json`,
    c2a3
  );

  // Markdown summary.
  const md = renderSummaryMd(c2a1, c2a2, c2a3);
  const mdRel = `${OUT_DIR_REL}/c2a-receipts-summary.md`;
  const mdAbs = resolve(REPO, mdRel);
  await mkdir(dirname(mdAbs), { recursive: true });
  await writeFile(mdAbs, md);
  const mdHash = sha256(Buffer.from(md));

  console.log("[c2a-receipts] artifacts:");
  console.log(`  ${c2a1Hash.path}`);
  console.log(`    raw      = ${c2a1Hash.raw_sha256}`);
  console.log(`    canonical= ${c2a1Hash.canonical_sha256}`);
  console.log(`  ${c2a2Hash.path}`);
  console.log(`    raw      = ${c2a2Hash.raw_sha256}`);
  console.log(`    canonical= ${c2a2Hash.canonical_sha256}`);
  console.log(`  ${c2a3Hash.path}`);
  console.log(`    raw      = ${c2a3Hash.raw_sha256}`);
  console.log(`    canonical= ${c2a3Hash.canonical_sha256}`);
  console.log(`  ${mdRel}`);
  console.log(`    raw      = ${mdHash}`);
  console.log("");
  console.log("[c2a-receipts] verdicts:");
  console.log(`  C2-A-1 sustained-TRACK landscape: ${c2a1.pass ? "PASS" : "BLOCK"}`);
  console.log(
    `    transition h = ${c2a1.sustained_track_transition_h_deg}°, ` +
      `window [${(H_L1_DEG - COINCIDENCE_WINDOW_DEG).toFixed(3)}°, ${(H_L1_DEG + COINCIDENCE_WINDOW_DEG).toFixed(3)}°], ` +
      `inside=${c2a1.transition_inside_coincidence_window}, ` +
      `margin to lower edge=${c2a1.transition_margin_to_lower_edge_deg?.toFixed(3)}°`
  );
  console.log(`  C2-A-2 objective-level abstain  : ${c2a2.pass ? "PASS" : "BLOCK"}`);
  console.log(
    `    degenerate trip rate=${((c2a2.summary.degenerate_trip_rate || 0) * 100).toFixed(1)}%, ` +
      `eligible no-trip rate=${((c2a2.summary.eligible_no_trip_rate || 0) * 100).toFixed(1)}%`
  );
  console.log(`  C2-A-3 package-gating separation : ${c2a3.pass ? "PASS" : "BLOCK"}`);
  console.log(
    `    min C_L1 above h*=${H_STAR_DEG}° is ${c2a3.min_C_L1_above_h_star.toFixed(10)}, ` +
      `threshold ${(1 - EPSILON_C).toFixed(10)}`
  );
}

main().catch((err) => {
  console.error(`[c2a-receipts] FAILED: ${err.message}`);
  process.exit(1);
});
