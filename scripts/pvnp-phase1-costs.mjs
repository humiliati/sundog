#!/usr/bin/env node
// scripts/pvnp-phase1-costs.mjs
//
// Consolidate the per-stage cost rows from costs.partial.json into the
// canonical costs.csv. Each row is keyed by (component, wall_ms, ops,
// calls, notes) and the file ends with the spec's derived totals:
//
//   C_search          (MLP-policy + inversion-attacker training)
//   C_rollout         (rollout baseline)
//   C_full_state      (full-state baseline)
//   C_signature       (signature compute)
//   C_verify          (verifier compute)
//   C_total_signature = C_signature + C_verify
//
// Also writes derived advantage ratios so the receipt can quote them.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getPhase1RunConfig } from "./lib/pvnp-phase1-run-config.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const ROLLOUT_DENOMINATOR_STABILITY_FLOOR_MS = 5.0;

function parseArgs(argv) {
  const args = { runDir: "results/pvnp/phase1-toy-verifier-v0" };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--run-dir") { args.runDir = argv[i + 1]; i += 1; }
    else throw new Error(`Unknown flag: ${argv[i]}`);
  }
  return args;
}

function csvRow(values) {
  return values.map((v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return s.includes(",") || s.includes("\"") ? `"${s.replace(/"/g, '""')}"` : s;
  }).join(",");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  await mkdir(outDir, { recursive: true });

  const partial = JSON.parse(await readFile(path.join(outDir, "costs.partial.json"), "utf8"));

  const sim = partial.simulation || { wall_ms: 0, ops: 0 };
  const baselines = partial.baselines || {};
  const verifier = partial.verifier || { wall_ms: 0, ops: 0, calls: 0 };
  const signature = partial.signature || { wall_ms: 0, ops: 0, calls: 0 };
  const ablation = partial.ablation || { wall_ms: 0, ops: 0, calls: 0 };
  const attackerInv = partial.attacker_inversion || { wall_ms: 0, ops: 0, calls: 0 };
  const attackerSpoof = partial.attacker_spoof || { wall_ms: 0, ops: 0, calls: 0 };

  const rows = [
    ["component", "wall_ms", "ops", "calls", "notes"].join(","),
    csvRow(["simulation_trajectories", sim.wall_ms.toFixed(2), sim.ops, "n/a",
      "shared cost of producing trajectories used by all baselines + signature"]),
    csvRow(["baseline_rollout", (baselines.rollout?.wall_ms ?? 0).toFixed(2),
      baselines.rollout?.ops ?? 0, baselines.rollout?.calls ?? 0,
      "C_rollout per measurement pair"]),
    csvRow(["baseline_full_state", (baselines.full_state?.wall_ms ?? 0).toFixed(2),
      baselines.full_state?.ops ?? 0, baselines.full_state?.calls ?? 0,
      "C_full_state per measurement pair"]),
    csvRow(["baseline_formal", (baselines.formal?.wall_ms ?? 0).toFixed(2),
      baselines.formal?.ops ?? 0, baselines.formal?.calls ?? 0,
      "C_formal per measurement pair (grid reachability)"]),
    csvRow(["signature_compute", signature.wall_ms.toFixed(2), signature.ops, signature.calls,
      "C_signature (sum over all signatures, including calibration)"]),
    csvRow(["verifier_decision", verifier.wall_ms.toFixed(2), verifier.ops, verifier.calls,
      "C_verify"]),
    csvRow(["ablation_decisions", ablation.wall_ms.toFixed(2), ablation.ops, ablation.calls,
      "vacuity probes (4 drops × measurement pairs)"]),
    csvRow(["attacker_inversion_training", attackerInv.wall_ms.toFixed(2), attackerInv.ops, attackerInv.calls,
      `C_search for A_inv_small (params=${attackerInv.params_count ?? 0})`]),
    csvRow(["attacker_spoof_attempts", attackerSpoof.wall_ms.toFixed(2), attackerSpoof.ops, attackerSpoof.calls,
      "A_spoof_small: ≤64 candidate verifies per unsafe item"]),
  ];

  // Derived totals + ratios.
  const cRollout = baselines.rollout?.wall_ms ?? 0;
  const cFullState = baselines.full_state?.wall_ms ?? 0;
  const cFormal = baselines.formal?.wall_ms ?? 0;
  const cSig = signature.wall_ms;
  const cVer = verifier.wall_ms;
  const cTotalSig = cSig + cVer;

  const cRolloutOps = baselines.rollout?.ops ?? 0;
  const cFullStateOps = baselines.full_state?.ops ?? 0;
  const cFormalOps = baselines.formal?.ops ?? 0;
  const cTotalSigOps = signature.ops + verifier.ops;

  rows.push("");
  rows.push("# derived totals (wall_ms, ops)");
  rows.push(csvRow(["C_rollout_total", cRollout.toFixed(2), cRolloutOps, "", ""]));
  rows.push(csvRow(["C_full_state_total", cFullState.toFixed(2), cFullStateOps, "", ""]));
  rows.push(csvRow(["C_formal_total", cFormal.toFixed(2), cFormalOps, "", ""]));
  rows.push(csvRow(["C_signature_total", cSig.toFixed(2), signature.ops, "", ""]));
  rows.push(csvRow(["C_verify_total", cVer.toFixed(2), verifier.ops, "", ""]));
  rows.push(csvRow(["C_total_signature", cTotalSig.toFixed(2), cTotalSigOps, "", "C_signature + C_verify"]));

  const rolloutRatio = cRollout > 0 ? (cTotalSig / cRollout) : null;
  const fullStateRatio = cFullState > 0 ? (cTotalSig / cFullState) : null;
  const rolloutRatioOps = cRolloutOps > 0 ? (cTotalSigOps / cRolloutOps) : null;
  const fullStateRatioOps = cFullStateOps > 0 ? (cTotalSigOps / cFullStateOps) : null;
  rows.push(csvRow(["rollout_cost_ratio_wall",
    rolloutRatio === null ? "" : rolloutRatio.toFixed(4), "", "", "C_total_signature / C_rollout (wall_ms)"]));
  rows.push(csvRow(["rollout_cost_ratio_ops",
    rolloutRatioOps === null ? "" : rolloutRatioOps.toFixed(4), "", "", "C_total_signature / C_rollout (ops)"]));
  rows.push(csvRow(["full_state_cost_ratio_wall",
    fullStateRatio === null ? "" : fullStateRatio.toFixed(4), "", "", "C_total_signature / C_full_state (wall_ms)"]));
  rows.push(csvRow(["full_state_cost_ratio_ops",
    fullStateRatioOps === null ? "" : fullStateRatioOps.toFixed(4), "", "", "C_total_signature / C_full_state (ops)"]));

  await writeFile(path.join(outDir, "costs.csv"), rows.join("\n") + "\n", "utf8");
  console.log("wrote costs.csv");
  console.log(`  C_total_signature/C_rollout = ${rolloutRatio === null ? "n/a" : rolloutRatio.toFixed(4)} (wall_ms), ${rolloutRatioOps === null ? "n/a" : rolloutRatioOps.toFixed(4)} (ops)`);
  console.log(`  C_total_signature/C_full_state = ${fullStateRatio === null ? "n/a" : fullStateRatio.toFixed(4)} (wall_ms), ${fullStateRatioOps === null ? "n/a" : fullStateRatioOps.toFixed(4)} (ops)`);

  // v4: rollout-denominator audit per PHASE1_V4_SLATE.md §Cost Gate
  // Restatement. Reports whether the rollout wall-time is stable enough
  // to serve as a ratio denominator (>= 5 ms) and names the chosen
  // promotion comparator. For v3 and v4 the comparator is full-state.
  const slate = getPhase1RunConfig(args.runDir);
  const version = slate.schema_suffix;
  if (version === "v4" || version === "v5" || version === "v6") {
    const rolloutStable = cRollout >= ROLLOUT_DENOMINATOR_STABILITY_FLOOR_MS;
    const audit = {
      schema: `pvnp-phase1-cost-denominator-audit-${version}`,
      rollout: {
        wall_ms: cRollout,
        ops: cRolloutOps,
        calls: baselines.rollout?.calls ?? 0,
        wall_ms_below_5ms: cRollout < ROLLOUT_DENOMINATOR_STABILITY_FLOOR_MS,
        stable_enough_for_ratio_denominator: rolloutStable,
        stability_floor_ms: ROLLOUT_DENOMINATOR_STABILITY_FLOOR_MS,
      },
      full_state: {
        wall_ms: cFullState,
        ops: cFullStateOps,
        calls: baselines.full_state?.calls ?? 0,
      },
      formal: {
        wall_ms: cFormal,
        ops: cFormalOps,
        calls: baselines.formal?.calls ?? 0,
      },
      promotion_comparator: version === "v6" ? "op_count_rollout" : (rolloutStable ? "rollout_and_full_state" : "full_state"),
      rollout_ratio_status: version === "v6" ? "wall_time_diagnostic_only" : (rolloutStable ? "promotion_gate" : "diagnostic_only"),
      rollout_ratio_wall_diagnostic: rolloutRatio,
      rollout_ratio_ops: rolloutRatioOps,
      full_state_ratio_wall: fullStateRatio,
      full_state_ratio_ops: fullStateRatioOps,
      c_total_signature_ms: cTotalSig,
      v4_targets: {
        c_total_signature_ms_max: 1010,
        full_state_ratio_wall_max: 105,
        op_count_ratio_max: 1.0,
      },
      v4_passes: {
        c_total_signature_ms: cTotalSig <= 1010,
        full_state_ratio_wall: fullStateRatio !== null && fullStateRatio <= 105,
        op_count_ratio: rolloutRatioOps !== null && rolloutRatioOps <= 1.0,
      },
    };
    await writeFile(
      path.join(outDir, "cost_denominator_audit.json"),
      JSON.stringify(audit, null, 2) + "\n",
      "utf8",
    );
    console.log(`  rollout stable_enough_for_ratio_denominator = ${rolloutStable}`);
    console.log(`  denominator audit: C_total_signature ${audit.v4_passes.c_total_signature_ms ? "PASS" : "FAIL"} | full_state ratio ${audit.v4_passes.full_state_ratio_wall ? "PASS" : "FAIL"} | op ratio ${audit.v4_passes.op_count_ratio ? "PASS" : "FAIL"}`);
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
