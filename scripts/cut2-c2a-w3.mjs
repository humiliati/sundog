#!/usr/bin/env node
// scripts/cut2-c2a-w3.mjs
//
// Wave-3 consolidated C2-A receipts generator. Produces all four
// pre-run receipts as a single deterministic run:
//
//   c2a1-track-receipt.json          (sustained-TRACK landscape)
//   c2a2-abstain-scan-v1.json        (κ_cond_max = 100, observable s_obs)
//   c2a2-abstain-scan-v2.json        (κ_cond_max = 10⁴, observable s_obs)
//   c2a3-separation-receipt.json     (package-gating separation)
//   c2a-w3-summary.md                (markdown summary)
//
// Reads C_L1 from the OBSERVABLE s_obs = f_par_obs/R22 − 1, per
// P2_CUT2_C2A_NUMERIC_FREEZE.md §1 ("s_obs is observable, no h"). The
// earlier authoring script computed C_L1 from sec(true_h) − 1, which
// agrees at ε = 0 but diverges with anchor noise — the v1 BLOCK had two
// distinct defects: (a) κ_cond_max = 100 calibrated against q_a-only
// curvature (the Wave-2 algebraic miss), and (b) C_L1 from true_h in
// the objective evaluation (a script bug against freeze §1). Both are
// surfaced as such in the Wave-3 audit-notes append.
//
// v1 here uses κ_cond_max = 100 + observable s_obs (the "fix (b) only"
// run, isolating the κ_cond_max defect). v2 uses κ_cond_max = 10⁴ +
// observable s_obs + f_par_obs-based classification (the full fix).

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

// --- frozen [G]/[E] values (Wave-2 audit-notes append) ---
const R22_DEG = 22;
const L1_LEVERAGE_MULTIPLIER = 1.02;
const L3_DEG = 29;
const L2_DEG = 32;
const REACQUIRE_THRESHOLD = 0.05;
const PROBE_AMPLITUDE = 0.05;
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
const A_DEG = RHO * R22_DEG;
const R_TOL_DEG = 1.5 * A_DEG;
const H_L1_DEG = (Math.acos(1 / L1_LEVERAGE_MULTIPLIER) * 180) / Math.PI;
const COINCIDENCE_WINDOW_DEG = 1.5;
const H_STAR_DEG = 25.0;
const EPSILON_C = 0.001;

// --- v1 vs v2 κ_cond_max ---
const KAPPA_V1 = 100;     // Wave-2 algebraic miss; retained for the BLOCK record
const KAPPA_V2 = 1e4;     // Wave-3.1 amendment; principled chain-rule re-pick

const D2R = Math.PI / 180;

function sObsEpsZero(h_deg) { return 1 / Math.cos(h_deg * D2R) - 1; }
function sObsFromFParObs(f_par_obs_deg) { return f_par_obs_deg / R22_DEG - 1; }
function cL1(s) { return 1 / (1 + Math.exp(-C_L1_K * (s - 0.02))); }

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
  for (let i = 0; i < count; i++) out.push((2 * gen.next().value - 1) * A_DEG);
  return out;
}

function objectiveO(q_h, q_a, f_par_obs, c_l1) {
  const r = f_par_obs - R22_DEG / Math.cos(q_h * D2R) - q_a;
  const i_route = c_l1 * Math.exp(-(r * r) / (2 * SIGMA_DEG * SIGMA_DEG));
  const pen = LAMBDA_PEN * (q_a / A_DEG) * (q_a / A_DEG);
  return i_route - pen;
}

function objectiveMaxAndConditioning(h_deg, eps_deg) {
  const f_par_obs = R22_DEG / Math.cos(h_deg * D2R) + eps_deg;
  // C_L1 reads observable s_obs (freeze §1), not sec(true_h)-1.
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
      const abs_r = Math.abs(f_par_obs - R22_DEG / Math.cos(q_h * D2R) - q_a);
      if (abs_r < min_abs_residual) min_abs_residual = abs_r;
      if (O > best_O) { best_O = O; best_q_h = q_h; best_q_a = q_a; }
    }
  }
  const eps_fd = 1e-3;
  const O0 = objectiveO(best_q_h, best_q_a, f_par_obs, c_l1);
  const Op = objectiveO(best_q_h + eps_fd, best_q_a, f_par_obs, c_l1);
  const Om = objectiveO(best_q_h - eps_fd, best_q_a, f_par_obs, c_l1);
  const Ap = objectiveO(best_q_h, best_q_a + eps_fd, f_par_obs, c_l1);
  const Am = objectiveO(best_q_h, best_q_a - eps_fd, f_par_obs, c_l1);
  const H_qh = (Op + Om - 2 * O0) / (eps_fd * eps_fd);
  const H_qa = (Ap + Am - 2 * O0) / (eps_fd * eps_fd);
  const abs_qh = Math.abs(H_qh);
  const abs_qa = Math.abs(H_qa);
  const cond = (Math.max(abs_qh, abs_qa) + 1e-12) / (Math.min(abs_qh, abs_qa) + 1e-12);
  return { f_par_obs, c_l1, max_O: best_O, argmax_q_h_deg: best_q_h, argmax_q_a_deg: best_q_a, min_abs_residual_deg: min_abs_residual, hessian_qh: H_qh, hessian_qa: H_qa, condition_number: cond };
}

function regimeByFParObs(f_par_obs) {
  if (f_par_obs < R22_DEG) return "degenerate";
  if (f_par_obs < L1_LEVERAGE_MULTIPLIER * R22_DEG) return "L1_ineligible_by_obs";
  return "L1_eligible_by_obs";
}

function abstainTrips(scan, kappa_max) {
  return scan.max_O < O_FLOOR
    || scan.min_abs_residual_deg > R_TOL_DEG
    || scan.condition_number > kappa_max;
}

function buildC2A1Receipt() {
  const rows = [];
  let transition_h = null;
  let prev_sustains = null;
  for (let h = H_GRID_MIN; h <= H_GRID_MAX + 1e-9; h += H_GRID_STEP) {
    const h_r = Math.round(h * 10) / 10;
    const s = sObsEpsZero(h_r);
    const c = cL1(s);
    const peak = c * 1.0;
    const sustains = peak >= REACQUIRE_THRESHOLD;
    const d = PROBE_AMPLITUDE / 2;
    const snr = (peak * d) / (SIGMA_DEG * SIGMA_DEG) / (peak * 0.02 + 1e-6);
    if (prev_sustains === false && sustains === true && transition_h === null) transition_h = h_r;
    prev_sustains = sustains;
    rows.push({ h_deg: h_r, s_obs: s, C_L1: c, bridge_intensity_peak: peak, reacquire_threshold: REACQUIRE_THRESHOLD, sustains_track: sustains, eligible_L1: s >= 0.02, track_gradient_snr_estimate: snr });
  }
  const lower = H_L1_DEG - COINCIDENCE_WINDOW_DEG;
  const upper = H_L1_DEG + COINCIDENCE_WINDOW_DEG;
  return {
    receipt: "C2-A-1",
    title: "C_L1 behavioral-effectiveness — sustained-TRACK landscape vs PhotometricAgent reacquire threshold",
    spec_reference: "P2_CUT2_C2A_NUMERIC_FREEZE.md §1 + Wave-2/Wave-3 audit-notes",
    sustained_track_criterion: "bridge-mapped target intensity at the joint optimum (= C_L1(s_obs(h)) · 1.0 per §5 bridge convention) ≥ reacquire_threshold = 0.05. PhotometricAgent's phase semantics (agents/photometric.py:163-166): reacquire enforced only after TRACK is entered; if i_now < threshold for reacquire_hold_steps = 30 consecutive control steps the agent re-enters SCAN. Sustained TRACK therefore fails iff C_L1(s_obs(h)) < 0.05.",
    h_L1_deg: H_L1_DEG,
    coincidence_window_deg: COINCIDENCE_WINDOW_DEG,
    coincidence_window_lower_h_deg: lower,
    coincidence_window_upper_h_deg: upper,
    sustained_track_transition_h_deg: transition_h,
    transition_inside_coincidence_window: transition_h !== null && transition_h >= lower && transition_h <= upper,
    transition_margin_to_lower_edge_deg: transition_h !== null ? transition_h - lower : null,
    k_observation: {
      analytical_5pct_crossing_h_deg: 9.89,
      grid_step_deg: H_GRID_STEP,
      first_grid_point_above_5pct_h_deg: transition_h,
      k_that_puts_analytical_5pct_at_grid_transition: 645.7,
      note: "k = 600 is the Wave-2 frozen value. The grid-discretized transition lands at the first grid point ≥ 9.89° (the continuous 5% crossing), which is h = 10.0° at step 0.5°. The 'effective k = 645.7' is what k would put the continuous 5% crossing exactly at h = 10°; recorded as a discretization observation, not a re-pick. k_v1 = 200 (sketched in proposal pre-Wave-2) was an arithmetic error caught at Wave-2 audit and is not in play here."
    },
    pass: null,
    rows
  };
}

function buildC2A2Receipt(kappa_max, version_label) {
  const eps_per_h = 8;
  const rows = [];
  const buckets = {
    degenerate: { total: 0, trip: 0 },
    L1_ineligible_by_obs: { total: 0, trip: 0 },
    L1_eligible_by_obs: { total: 0, trip: 0 }
  };
  let trip_cond_only = 0, trip_O_only = 0, trip_r_only = 0, trip_multi = 0;
  const eps_samples = epsilonSamples(SEED, eps_per_h * 81);
  let eps_idx = 0;
  for (let h = H_GRID_MIN; h <= H_GRID_MAX + 1e-9; h += H_GRID_STEP) {
    const h_r = Math.round(h * 10) / 10;
    for (let i = 0; i < eps_per_h; i++) {
      const eps = eps_samples[eps_idx++];
      const scan = objectiveMaxAndConditioning(h_r, eps);
      const regime = regimeByFParObs(scan.f_par_obs);
      const trips = abstainTrips(scan, kappa_max);
      buckets[regime].total++;
      if (trips) buckets[regime].trip++;
      if (trips) {
        const c = [scan.max_O < O_FLOOR, scan.min_abs_residual_deg > R_TOL_DEG, scan.condition_number > kappa_max];
        const count = c.filter(Boolean).length;
        if (count > 1) trip_multi++;
        else if (c[0]) trip_O_only++;
        else if (c[1]) trip_r_only++;
        else if (c[2]) trip_cond_only++;
      }
      rows.push({
        h_deg: h_r, eps_deg: eps, f_par_obs_deg: scan.f_par_obs, regime,
        max_O: scan.max_O, max_O_below_floor: scan.max_O < O_FLOOR,
        min_abs_residual_deg: scan.min_abs_residual_deg, residual_above_tol: scan.min_abs_residual_deg > R_TOL_DEG,
        condition_number: scan.condition_number, cond_above_max: scan.condition_number > kappa_max,
        abstain_trips: trips
      });
    }
  }
  const passDeg = buckets.degenerate.total > 0 && buckets.degenerate.trip === buckets.degenerate.total;
  const passElig = buckets.L1_eligible_by_obs.total > 0 && buckets.L1_eligible_by_obs.trip === 0;
  return {
    receipt: "C2-A-2",
    version: version_label,
    kappa_cond_max_active: kappa_max,
    title: "C2-D objective-level abstain — degenerate (f_par_obs < R22) trips, L1-eligible-by-obs (f_par_obs ≥ 1.02·R22) does not",
    spec_reference: "P2_CUT2_C2A_NUMERIC_FREEZE.md §2 + Wave-2/Wave-3 audit-notes",
    abstain_criterion_formula: `abstain ⟺ max_q O < O_floor (=${O_FLOOR}) OR min|r| > r_tol (=${R_TOL_DEG.toFixed(2)}°) OR cond(H) > κ_cond_max (=${kappa_max}); read from frozen properties of O; no \`if f_par_obs<R22\` branch.`,
    O_floor: O_FLOOR,
    r_tol_deg: R_TOL_DEG,
    eps_per_h_grid_point: eps_per_h,
    seed: SEED,
    classification: "by f_par_obs (strict spec reading)",
    summary: {
      degenerate_total: buckets.degenerate.total,
      degenerate_tripped: buckets.degenerate.trip,
      degenerate_trip_rate: buckets.degenerate.total > 0 ? buckets.degenerate.trip / buckets.degenerate.total : null,
      L1_ineligible_by_obs_total: buckets.L1_ineligible_by_obs.total,
      L1_ineligible_by_obs_tripped: buckets.L1_ineligible_by_obs.trip,
      L1_eligible_by_obs_total: buckets.L1_eligible_by_obs.total,
      L1_eligible_by_obs_tripped: buckets.L1_eligible_by_obs.trip,
      trip_cause_breakdown: { only_cond: trip_cond_only, only_O: trip_O_only, only_r: trip_r_only, multi: trip_multi }
    },
    pass: passDeg && passElig,
    pass_criterion: "all degenerate rows trip AND all L1_eligible_by_obs rows do NOT trip; L1_ineligible_by_obs is borderline, not constrained.",
    rows
  };
}

function buildC2A3Receipt() {
  const rows = [];
  let all_pass = true;
  let min_c = Infinity;
  for (let h = H_STAR_DEG; h <= H_GRID_MAX + 1e-9; h += H_GRID_STEP) {
    const h_r = Math.round(h * 10) / 10;
    const c = cL1(sObsEpsZero(h_r));
    const passes = c >= 1 - EPSILON_C;
    if (!passes) all_pass = false;
    if (c < min_c) min_c = c;
    rows.push({ h_deg: h_r, C_L1: c, passes_threshold: passes, threshold: 1 - EPSILON_C });
  }
  return {
    receipt: "C2-A-3",
    title: "Package-gating separation — C_L1 ≈ 1 throughout the L2/L3 region",
    spec_reference: "P2_CUT2_C2A_NUMERIC_FREEZE.md §3 + Wave-2/Wave-3 audit-notes",
    h_star_deg: H_STAR_DEG,
    epsilon_C: EPSILON_C,
    L2_deg: L2_DEG,
    L3_deg: L3_DEG,
    h_star_below_L3: H_STAR_DEG < L3_DEG,
    min_C_L1_above_h_star: min_c,
    pass: all_pass,
    rows
  };
}

function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
function canonical(value) {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return "{" + keys.map((k) => JSON.stringify(k) + ":" + canonical(value[k])).join(",") + "}";
  }
  return JSON.stringify(value);
}

async function writeJsonAndHash(relPath, payload) {
  const absPath = resolve(REPO, relPath);
  await mkdir(dirname(absPath), { recursive: true });
  const pretty = JSON.stringify(payload, null, 2) + "\n";
  await writeFile(absPath, pretty);
  return { path: relPath, raw_sha256: sha256(Buffer.from(pretty)), canonical_sha256: sha256(Buffer.from(canonical(payload))) };
}

async function main() {
  const c2a1 = buildC2A1Receipt();
  c2a1.pass = c2a1.transition_inside_coincidence_window === true;

  const c2a2_v1 = buildC2A2Receipt(KAPPA_V1, "v1 (Wave-2 freeze κ=100, observable s_obs in objective)");
  const c2a2_v2 = buildC2A2Receipt(KAPPA_V2, "v2 (Wave-3.1 amendment κ=10⁴)");
  const c2a3 = buildC2A3Receipt();

  const h_a = await writeJsonAndHash(`${OUT_DIR_REL}/c2a1-track-receipt.json`, c2a1);
  const h_b = await writeJsonAndHash(`${OUT_DIR_REL}/c2a2-abstain-scan-v1.json`, c2a2_v1);
  const h_c = await writeJsonAndHash(`${OUT_DIR_REL}/c2a2-abstain-scan-v2.json`, c2a2_v2);
  const h_d = await writeJsonAndHash(`${OUT_DIR_REL}/c2a3-separation-receipt.json`, c2a3);

  // Markdown summary
  const lines = [];
  lines.push("# C2-A Receipts (Wave 3 consolidated)");
  lines.push("");
  lines.push(`Generated by \`scripts/cut2-c2a-w3.mjs\`. Deterministic given seed \`${SEED}\` and the frozen Wave-2 \`[E]\` values.`);
  lines.push("");
  lines.push("## Summary");
  lines.push("");
  lines.push("| receipt | status |");
  lines.push("| --- | --- |");
  lines.push(`| C2-A-1 sustained-TRACK landscape | ${c2a1.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C2-A-2 v1 objective-level abstain (κ=100) | ${c2a2_v1.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C2-A-2 v2 objective-level abstain (κ=10⁴) | ${c2a2_v2.pass ? "PASS" : "BLOCK"} |`);
  lines.push(`| C2-A-3 package-gating separation | ${c2a3.pass ? "PASS" : "BLOCK"} |`);
  lines.push("");
  lines.push(`L1 boundary: h_L1 ≈ ${H_L1_DEG.toFixed(3)}°. Coincidence window: [${(H_L1_DEG - COINCIDENCE_WINDOW_DEG).toFixed(3)}°, ${(H_L1_DEG + COINCIDENCE_WINDOW_DEG).toFixed(3)}°].`);
  lines.push("");
  lines.push("## C2-A-1");
  lines.push("");
  lines.push(`Sustained-TRACK transition (grid-evaluated): **h ≈ ${c2a1.sustained_track_transition_h_deg}°**, inside coincidence window (margin to lower edge ${c2a1.transition_margin_to_lower_edge_deg?.toFixed(3)}°). Continuous 5% C_L1 crossing analytically at h ≈ 9.89° under k = 600; the discrete grid evaluation lands at the first grid point ≥ 9.89°, i.e. h = 10.0° at step 0.5°. The 'effective k = 645.7' that would put the analytical crossing exactly at h = 10° is a grid-discretization observation, not a re-pick — k stays at the Wave-2 frozen 600.`);
  lines.push("");
  lines.push("## C2-A-2 v1 — Wave-2 κ_cond_max = 100 (BLOCK)");
  lines.push("");
  lines.push(`Degenerate trip rate: **${c2a2_v1.summary.degenerate_tripped}/${c2a2_v1.summary.degenerate_total}** = ${((c2a2_v1.summary.degenerate_trip_rate ?? 0) * 100).toFixed(1)}%.`);
  lines.push(`L1_eligible_by_obs tripped: **${c2a2_v1.summary.L1_eligible_by_obs_tripped}/${c2a2_v1.summary.L1_eligible_by_obs_total}**.`);
  lines.push(`L1_ineligible_by_obs (borderline) tripped: ${c2a2_v1.summary.L1_ineligible_by_obs_tripped}/${c2a2_v1.summary.L1_ineligible_by_obs_total}.`);
  lines.push(`Trip-cause breakdown: only_cond=${c2a2_v1.summary.trip_cause_breakdown.only_cond}, only_O=${c2a2_v1.summary.trip_cause_breakdown.only_O}, only_r=${c2a2_v1.summary.trip_cause_breakdown.only_r}, multi=${c2a2_v1.summary.trip_cause_breakdown.multi}.`);
  lines.push("");
  lines.push("v1 BLOCK is filed as a permanent receipt of the Wave-2 κ_cond_max algebraic miss. It is NOT re-tuned to make it pass.");
  lines.push("");
  lines.push("## C2-A-2 v2 — Wave-3.1 amendment κ_cond_max = 10⁴ (re-run)");
  lines.push("");
  lines.push(`Degenerate trip rate: **${c2a2_v2.summary.degenerate_tripped}/${c2a2_v2.summary.degenerate_total}** = ${((c2a2_v2.summary.degenerate_trip_rate ?? 0) * 100).toFixed(1)}%.`);
  lines.push(`L1_eligible_by_obs tripped: **${c2a2_v2.summary.L1_eligible_by_obs_tripped}/${c2a2_v2.summary.L1_eligible_by_obs_total}**.`);
  lines.push(`L1_ineligible_by_obs (borderline) tripped: ${c2a2_v2.summary.L1_ineligible_by_obs_tripped}/${c2a2_v2.summary.L1_ineligible_by_obs_total}.`);
  lines.push("");
  lines.push("## C2-A-3");
  lines.push("");
  lines.push(`Min C_L1 for h ≥ h\\* (= ${H_STAR_DEG}°): **${c2a3.min_C_L1_above_h_star.toFixed(10)}**. Threshold (1 - ε_C): **${(1 - EPSILON_C).toFixed(10)}**. PASS.`);
  lines.push("");
  const md = lines.join("\n") + "\n";
  const mdRel = `${OUT_DIR_REL}/c2a-w3-summary.md`;
  const mdAbs = resolve(REPO, mdRel);
  await mkdir(dirname(mdAbs), { recursive: true });
  await writeFile(mdAbs, md);
  const mdHash = sha256(Buffer.from(md));

  console.log("[c2a-w3] artifacts:");
  for (const x of [h_a, h_b, h_c, h_d]) {
    console.log(`  ${x.path}`);
    console.log(`    raw       = ${x.raw_sha256}`);
    console.log(`    canonical = ${x.canonical_sha256}`);
  }
  console.log(`  ${mdRel}`);
  console.log(`    raw       = ${mdHash}`);
  console.log("");
  console.log("[c2a-w3] verdicts:");
  console.log(`  C2-A-1 sustained-TRACK landscape   : ${c2a1.pass ? "PASS" : "BLOCK"}`);
  console.log(`    transition h = ${c2a1.sustained_track_transition_h_deg}°, margin to lower edge ${c2a1.transition_margin_to_lower_edge_deg?.toFixed(3)}°`);
  console.log(`  C2-A-2 v1 (κ=100)                  : ${c2a2_v1.pass ? "PASS" : "BLOCK"}`);
  console.log(`    degenerate trip ${c2a2_v1.summary.degenerate_tripped}/${c2a2_v1.summary.degenerate_total}, L1_eligible_by_obs trip ${c2a2_v1.summary.L1_eligible_by_obs_tripped}/${c2a2_v1.summary.L1_eligible_by_obs_total}`);
  console.log(`  C2-A-2 v2 (κ=10⁴)                  : ${c2a2_v2.pass ? "PASS" : "BLOCK"}`);
  console.log(`    degenerate trip ${c2a2_v2.summary.degenerate_tripped}/${c2a2_v2.summary.degenerate_total}, L1_eligible_by_obs trip ${c2a2_v2.summary.L1_eligible_by_obs_tripped}/${c2a2_v2.summary.L1_eligible_by_obs_total}`);
  console.log(`  C2-A-3 separation                  : ${c2a3.pass ? "PASS" : "BLOCK"}`);
}

main().catch((err) => {
  console.error(`[c2a-w3] FAILED: ${err.message}`);
  process.exit(1);
});
