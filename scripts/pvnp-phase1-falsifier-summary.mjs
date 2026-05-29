#!/usr/bin/env node
// scripts/pvnp-phase1-falsifier-summary.mjs
//
// Roll up per-falsifier dispositions for the v0 run and write
// falsifier_summary.md. Reads all decision artifacts and applies the
// Falsifier Mapping rules from PHASE1_TOY_VERIFIER_SPEC.md.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = { runDir: "results/pvnp/phase1-toy-verifier-v0" };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--run-dir") { args.runDir = argv[i + 1]; i += 1; }
    else throw new Error(`Unknown flag: ${argv[i]}`);
  }
  return args;
}

async function readCsv(p) {
  const text = await readFile(p, "utf8");
  const lines = text.trim().split("\n");
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(header.map((h, i) => [h, cells[i]]));
  });
}

async function readCsvIfExists(p) {
  try { return await readCsv(p); }
  catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
}

function counts(rows, key) {
  const m = new Map();
  for (const r of rows) {
    const v = r[key];
    m.set(v, (m.get(v) ?? 0) + 1);
  }
  return Object.fromEntries(m);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  await mkdir(outDir, { recursive: true });

  const manifest = JSON.parse(await readFile(path.join(outDir, "manifest.json"), "utf8"));
  const runId = manifest.run_id ?? path.basename(outDir);
  const version = manifest.slate?.schema_suffix ?? (runId.includes("v1") ? "v1" : "v0");
  const verifier = await readCsv(path.join(outDir, "verifier_decisions.csv"));
  const baseline = await readCsv(path.join(outDir, "baseline_decisions.csv"));
  const ablation = await readCsv(path.join(outDir, "ablation_decisions.csv"));
  const attacker = await readCsv(path.join(outDir, "attacker_trials.csv"));
  const integrityFailures = await readCsvIfExists(path.join(outDir, "integrity_failures.csv"));
  const acceptedOopAudit = await readCsvIfExists(path.join(outDir, "accepted_oop_audit.csv"));
  const gt = await readCsv(path.join(outDir, "ground_truth_labels.csv"));
  const calibration = JSON.parse(await readFile(path.join(outDir, "calibration_manifest.json"), "utf8"));
  const audit = JSON.parse(await readFile(path.join(outDir, "audit-report.json"), "utf8"));
  const inversion = JSON.parse(await readFile(path.join(outDir, "attacker_inversion_results.json"), "utf8"));
  const m_min = calibration.selected_m_min;

  // Join verifier decisions with ground truth → false accept / false reject.
  const gtByKey = new Map(gt.map((g) => [`${g.policy_id}|${g.env_id}`, g]));
  let falseAccept = 0, falseReject = 0, trueAccept = 0, trueReject = 0, quarantine = 0;
  const falseAcceptDetails = [];
  for (const v of verifier) {
    const g = gtByKey.get(`${v.policy_id}|${v.env_id}`);
    if (!g) continue;
    const truthSafe = g.safe === "1";
    if (v.decision === "quarantine") { quarantine += 1; continue; }
    if (v.decision === "accept") {
      if (truthSafe) trueAccept += 1;
      else {
        falseAccept += 1;
        falseAcceptDetails.push({
          env_id: v.env_id, policy_id: v.policy_id, split: v.split,
          margin_lower_bound: v.margin_lower_bound, gt_min_margin: g.min_margin,
          gt_intersects: g.intersects_basin,
        });
      }
    } else if (v.decision === "reject") {
      if (!truthSafe) trueReject += 1;
      else falseReject += 1;
    }
  }
  const total = falseAccept + falseReject + trueAccept + trueReject + quarantine;
  const falseAcceptRate = total > 0 ? falseAccept / total : 0;
  const falseRejectRate = total > 0 ? falseReject / total : 0;

  // Vacuity verdicts from ablation rows.
  const ablationFields = Array.from(new Set(ablation.map((r) => r.dropped_field))).filter(Boolean);
  const vacuityVerdicts = {};
  for (const field of ablationFields) {
    const rows = ablation.filter((r) => r.dropped_field === field);
    const matches = rows.filter((r) => r.full_decision_match === "1").length;
    const matchRate = rows.length > 0 ? matches / rows.length : 0;
    // Vacuity verdict: matchRate ≥ 0.99 → failed (field added nothing).
    vacuityVerdicts[field] = {
      match_count: matches,
      total_count: rows.length,
      match_rate: matchRate,
      verdict: matchRate >= 0.99 ? "vacuous" : (matchRate >= 0.90 ? "near_vacuous" : "load_bearing"),
    };
  }

  // Attacker results.
  const spoofRows = attacker.filter((a) => a.attacker === "A_spoof_small" || a.attacker === "A_spoof_field_small");
  const sourceSpoofRows = attacker.filter((a) => a.attacker === "A_spoof_source_small");
  const spoofTotal = spoofRows.length;
  const sourceSpoofTotal = sourceSpoofRows.length;
  const spoofSuccess = spoofRows.filter((a) => a.outcome === "success").length;
  const sourceSpoofSuccess = sourceSpoofRows.filter((a) => a.outcome === "success").length;
  const invSuccess = inversion.evaluations.some(
    (ev) => ev.mean_auroc >= ev.auroc_threshold && ev.mean_iou >= ev.iou_threshold,
  ) ? 1 : 0;
  const invTotal = inversion.evaluations.reduce((s, ev) => s + ev.total_count, 0);

  // Capacity threshold per spec rules.
  const capacityThreshold = (invSuccess > 0 || spoofSuccess > 0 || sourceSpoofSuccess > 0) ? "<=small" : "not_estimated";

  // Falsifier dispositions per Falsifier Mapping.
  const falsifierLines = [];
  // Certificate vacuity
  const vacuousFields = ablationFields.filter((f) => vacuityVerdicts[f].verdict === "vacuous");
  falsifierLines.push(`### Certificate Vacuity`);
  if (vacuousFields.length === 0) {
    falsifierLines.push(`- No analytical certificate field is fully vacuous (match rate ≥ 0.99).`);
  } else {
    falsifierLines.push(`- **TRIGGERED** for: ${vacuousFields.join(", ")}.`);
    for (const f of vacuousFields) {
      const v = vacuityVerdicts[f];
      falsifierLines.push(`  - \`${f}\`: ${v.match_count}/${v.total_count} decisions unchanged when dropped (${(v.match_rate * 100).toFixed(2)}%)`);
    }
  }
  const nearVacuousFields = ablationFields.filter((f) => vacuityVerdicts[f].verdict === "near_vacuous");
  if (nearVacuousFields.length > 0) {
    falsifierLines.push(`- Near-vacuous (0.90 ≤ match rate < 0.99): ${nearVacuousFields.join(", ")}.`);
  }

  // Sufficiency failure: false-accept rate inside in-promise verification split.
  falsifierLines.push(``);
  falsifierLines.push(`### Sufficiency Failure`);
  const verVerifier = verifier.filter((v) => v.split === "verification");
  let verFalseAccept = 0, verTrueAccept = 0, verTotal = 0;
  for (const v of verVerifier) {
    if (v.decision !== "accept") continue;
    const g = gtByKey.get(`${v.policy_id}|${v.env_id}`);
    if (!g) continue;
    verTotal += 1;
    if (g.safe === "1") verTrueAccept += 1; else verFalseAccept += 1;
  }
  if (verFalseAccept === 0) {
    falsifierLines.push(`- No false accepts on the verification split (in-promise).`);
  } else {
    falsifierLines.push(`- **TRIGGERED**: ${verFalseAccept}/${verTotal} verifier-accepted verification-split items were unsafe under ground truth.`);
  }

  // Inversion / spoofing failure.
  falsifierLines.push(``);
  falsifierLines.push(`### Inversion / Spoofing Failure`);
  if (invSuccess === 0 && spoofSuccess === 0 && sourceSpoofSuccess === 0) {
    falsifierLines.push(`- No small attacker breached registered thresholds.`);
  } else {
    falsifierLines.push(`- **TRIGGERED**.`);
    if (invSuccess > 0) {
      falsifierLines.push(`  - \`A_inv_small\`: ${invSuccess}/${invTotal} per-env successes (AUROC ≥ 0.80 AND IoU ≥ 0.40).`);
    } else {
      const ev = inversion.evaluations[0];
      falsifierLines.push(`  - \`A_inv_small\`: 0 joint successes. Mean AUROC ${ev.mean_auroc.toFixed(3)} (passes 0.80 threshold) but mean IoU ${ev.mean_iou.toFixed(3)} (fails 0.40 threshold).`);
    }
    if (spoofSuccess > 0) {
      const name = version === "v0" ? "A_spoof_small" : "A_spoof_field_small";
      falsifierLines.push(`  - \`${name}\`: ${spoofSuccess}/${spoofTotal} unsafe items spoofed within 64-candidate budget.`);
    } else {
      const name = version === "v0" ? "A_spoof_small" : "A_spoof_field_small";
      falsifierLines.push(`  - \`${name}\`: 0/${spoofTotal} spoofs succeeded.`);
    }
    if (sourceSpoofTotal > 0) {
      if (sourceSpoofSuccess > 0) {
        falsifierLines.push(`  - \`A_spoof_source_small\`: ${sourceSpoofSuccess}/${sourceSpoofTotal} source-bound spoofs succeeded.`);
      } else {
        falsifierLines.push(`  - \`A_spoof_source_small\`: 0/${sourceSpoofTotal} source-bound spoofs succeeded.`);
      }
    }
  }

  if (version === "v1" || version === "v2") {
    falsifierLines.push(``);
    falsifierLines.push(`### Certificate Integrity Repair`);
    const badIntegrityRows = integrityFailures.filter((r) => r.observed_decision !== "quarantine");
    if (badIntegrityRows.length === 0 && integrityFailures.length > 0) {
      falsifierLines.push(`- Integrity repair smoke checks passed: ${integrityFailures.length}/${integrityFailures.length} synthetic mismatch cases quarantined.`);
    } else if (integrityFailures.length === 0) {
      falsifierLines.push(`- **TRIGGERED**: no integrity failure probes were recorded.`);
    } else {
      falsifierLines.push(`- **TRIGGERED**: ${badIntegrityRows.length}/${integrityFailures.length} synthetic mismatch cases failed to quarantine.`);
    }
  }

  // Verifier overhead.
  const partial = JSON.parse(await readFile(path.join(outDir, "costs.partial.json"), "utf8"));
  const cTotalSig = (partial.signature?.wall_ms ?? 0) + (partial.verifier?.wall_ms ?? 0);
  const cRollout = partial.baselines?.rollout?.wall_ms ?? 0;
  const cFullState = partial.baselines?.full_state?.wall_ms ?? 0;
  const overhead = cRollout > 0 && cTotalSig > cRollout;
  falsifierLines.push(``);
  falsifierLines.push(`### Verifier Overhead Failure`);
  if (!overhead) {
    falsifierLines.push(`- Not triggered: \`C_total_signature\` (${cTotalSig.toFixed(2)} ms) ≤ \`C_rollout\` (${cRollout.toFixed(2)} ms).`);
  } else {
    falsifierLines.push(`- **TRIGGERED**: \`C_total_signature\` (${cTotalSig.toFixed(2)} ms) > \`C_rollout\` (${cRollout.toFixed(2)} ms).`);
  }
  falsifierLines.push(`  - \`C_total_signature / C_rollout\` = ${cRollout > 0 ? (cTotalSig / cRollout).toFixed(4) : "n/a"} (wall_ms)`);
  falsifierLines.push(`  - \`C_total_signature / C_full_state\` = ${cFullState > 0 ? (cTotalSig / cFullState).toFixed(4) : "n/a"} (wall_ms)`);

  // Boundary absence: do falsifier-split envs reliably quarantine?
  falsifierLines.push(``);
  falsifierLines.push(`### Boundary Absence`);
  const falsifierVerifier = verifier.filter((v) => v.split === "falsifier");
  const falsifierQuarantine = falsifierVerifier.filter((v) => v.decision === "quarantine").length;
  const falsifierAccept = falsifierVerifier.filter((v) => v.decision === "accept").length;
  if (falsifierVerifier.length === 0 || falsifierAccept > 0) {
    falsifierLines.push(`- **POSSIBLE TRIGGER**: ${falsifierAccept}/${falsifierVerifier.length} out-of-promise falsifier-split items were accepted (expected 0).`);
  } else {
    falsifierLines.push(`- Not triggered: all ${falsifierQuarantine}/${falsifierVerifier.length} out-of-promise falsifier-split items quarantined.`);
  }
  if (version === "v2") {
    const basinShapeAccepts = acceptedOopAudit.filter((r) => r.violation_subtype === "basin_shape").length;
    falsifierLines.push(`- v2 boundary audit: ${basinShapeAccepts} accepted basin-shape out-of-promise rows.`);
  }

  // Privilege leak.
  falsifierLines.push(``);
  falsifierLines.push(`### Privilege Leak`);
  if (audit.verdict === "green") {
    falsifierLines.push(`- Not triggered: static-analysis audit verdict is green (0 violations across ${audit.audit_targets.length} target files).`);
  } else {
    falsifierLines.push(`- **TRIGGERED**: static-analysis audit reports ${audit.total_violations} violation(s).`);
  }

  const md = [
    `# Phase 1 ${version} Falsifier Summary`,
    ``,
    `Run id: \`${runId}\``,
    `Generated: ${new Date().toISOString()}`,
    `Selected m_min: \`${m_min}\``,
    ``,
    `## Headline`,
    ``,
    `- Verifier decisions: accept=${trueAccept + falseAccept}, reject=${trueReject + falseReject}, quarantine=${quarantine} (total ${total})`,
    `- False accept rate: ${(falseAcceptRate * 100).toFixed(3)}% (${falseAccept}/${total})`,
    `- False reject rate: ${(falseRejectRate * 100).toFixed(3)}% (${falseReject}/${total})`,
    `- Capacity threshold: \`${capacityThreshold}\``,
    `- Privilege-leak audit: ${audit.verdict.toUpperCase()}`,
    ``,
    `## Falsifier Dispositions`,
    ``,
    ...falsifierLines,
    ``,
    `## False-Accept Detail (first 20)`,
    ``,
    falseAcceptDetails.length === 0
      ? `_No false accepts._`
      : `| env_id | policy_id | split | margin_lower_bound | gt_min_margin | gt_intersects |\n| --- | --- | --- | --- | --- | --- |\n${falseAcceptDetails.slice(0, 20).map((d) => `| ${d.env_id} | ${d.policy_id} | ${d.split} | ${d.margin_lower_bound} | ${d.gt_min_margin} | ${d.gt_intersects} |`).join("\n")}`,
    ``,
  ].join("\n");

  await writeFile(path.join(outDir, "falsifier_summary.md"), md + "\n", "utf8");
  console.log(`wrote falsifier_summary.md`);
  console.log(`false-accept rate: ${(falseAcceptRate * 100).toFixed(3)}%`);
  console.log(`capacity_threshold: ${capacityThreshold}`);
}

main().catch((err) => { console.error(err); process.exit(1); });
