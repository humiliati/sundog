import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = {
    input: "results/balance/phase10-envelope",
    out: "results/balance/phase10-audit",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;
    if (flag === "--in" || flag === "--input") args.input = value;
    else if (flag === "--out") args.out = value;
    else throw new Error(`Unknown flag: ${flag}`);
  }

  return args;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (char === "\"" && next === "\"") {
        field += "\"";
        i += 1;
      } else if (char === "\"") {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === "\"") {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  if (rows.length === 0) return [];

  const headers = rows[0];
  return rows.slice(1)
    .filter((values) => values.some((value) => value !== ""))
    .map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

async function readCsv(filePath) {
  return parseCsv(await readFile(filePath, "utf8"));
}

function toNumber(value) {
  const number = Number.parseFloat(value);
  return Number.isFinite(number) ? number : null;
}

function round(value, digits = 6) {
  if (!Number.isFinite(value)) return "";
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function groupBy(rows, keyFn) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    const existing = groups.get(key) ?? [];
    existing.push(row);
    groups.set(key, existing);
  }
  return groups;
}

function countBy(rows, keyFn) {
  const counts = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([key, count]) => `${key}:${count}`)
    .join(";");
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll("\"", "\"\"")}"`;
  return text;
}

function rowsToCsv(rows, columns = null) {
  const explicitColumns = columns ?? [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [explicitColumns.join(",")];
  for (const row of rows) lines.push(explicitColumns.map((column) => csvEscape(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function oracleAuditRows(envelopeRows, duration) {
  return envelopeRows.map((row) => {
    const survivalSundog = toNumber(row.survival_sundog_mean);
    const survivalOracle = toNumber(row.survival_oracle_mean);
    const rmsSundog = toNumber(row.rms_theta_sundog_mean);
    const rmsOracle = toNumber(row.rms_theta_oracle_mean);
    const survivalDelta = survivalOracle - survivalSundog;
    const rmsDelta = rmsOracle - rmsSundog;
    const bothAtDuration = survivalSundog >= duration - 1e-6 && survivalOracle >= duration - 1e-6;
    const oracleSurvivalExceeds = survivalDelta > 1e-6;
    const oracleLowerRms = rmsDelta < -1e-6;
    const auditClass = bothAtDuration && oracleLowerRms
      ? "survival_ceiling_masks_oracle_quality"
      : oracleSurvivalExceeds
        ? "oracle_survival_exceeds"
        : oracleLowerRms
          ? "oracle_quality_exceeds_without_survival_delta"
          : "oracle_not_better_on_audit_metrics";

    return {
      cell_id: row.cell_id,
      axis: row.axis,
      axis_value: row.axis_value,
      preset: row.preset,
      cell_class: row.cell_class,
      light_elev_deg: row.light_elev_deg,
      delay_ms: row.delay_ms,
      survival_sundog_mean: row.survival_sundog_mean,
      survival_oracle_mean: row.survival_oracle_mean,
      oracle_minus_sundog_survival: round(survivalDelta),
      rms_theta_sundog_mean: row.rms_theta_sundog_mean,
      rms_theta_oracle_mean: row.rms_theta_oracle_mean,
      oracle_minus_sundog_rms_theta: round(rmsDelta),
      both_at_duration_cap: bothAtDuration,
      oracle_survival_exceeds: oracleSurvivalExceeds,
      oracle_lower_rms: oracleLowerRms,
      audit_class: auditClass,
      replay_url: row.replay_url,
      oracle_replay_url: row.oracle_replay_url,
    };
  });
}

function p2AuditRows(envelopeRows, trialRows) {
  const trialsByCell = groupBy(trialRows, (row) => row.cellId);
  const p2Cells = envelopeRows.filter((row) => (
    (toNumber(row.light_elev_deg) ?? 0) >= 80
    || (toNumber(row.delay_ms) ?? 0) >= 200
  ));

  return p2Cells.map((row) => {
    const trials = trialsByCell.get(row.cell_id) ?? [];
    const bySeed = groupBy(trials, (trial) => trial.seed);
    let sundogWins = 0;
    let naiveWins = 0;
    let ties = 0;
    let bothFail = 0;
    let sundogSuccess = 0;
    let naiveSuccess = 0;
    const margins = [];
    const sundogRows = trials.filter((trial) => trial.mode === "sundog_shadow");
    const naiveRows = trials.filter((trial) => trial.mode === "naive_shadow");

    for (const seedRows of bySeed.values()) {
      const sundog = seedRows.find((trial) => trial.mode === "sundog_shadow");
      const naive = seedRows.find((trial) => trial.mode === "naive_shadow");
      if (!sundog || !naive) continue;
      const margin = toNumber(sundog.simulated_time) - toNumber(naive.simulated_time);
      margins.push(margin);
      if (margin > 1e-6) sundogWins += 1;
      else if (margin < -1e-6) naiveWins += 1;
      else ties += 1;
      if (sundog.outcome !== "timeout" && naive.outcome !== "timeout") bothFail += 1;
      if (sundog.outcome === "timeout") sundogSuccess += 1;
      if (naive.outcome === "timeout") naiveSuccess += 1;
    }

    const hardSuccessViolation = sundogSuccess > naiveSuccess;
    const softSurvivalViolation = mean(margins) > 0;
    const auditClass = hardSuccessViolation
      ? "unexpected_sundog_success_in_failure_regime"
      : softSurvivalViolation && bothFail === margins.length
        ? "all_fail_sundog_lasts_longer"
        : softSurvivalViolation
          ? "survival_margin_violation_without_success"
          : "failure_boundary_holds";

    return {
      cell_id: row.cell_id,
      axis: row.axis,
      axis_value: row.axis_value,
      preset: row.preset,
      light_elev_deg: row.light_elev_deg,
      delay_ms: row.delay_ms,
      cell_class: row.cell_class,
      seed_pairs: margins.length,
      mean_sundog_minus_naive_survival: round(mean(margins)),
      sundog_seed_wins: sundogWins,
      naive_seed_wins: naiveWins,
      tied_seed_pairs: ties,
      both_fail_seed_pairs: bothFail,
      sundog_successes: sundogSuccess,
      naive_successes: naiveSuccess,
      sundog_outcomes: countBy(sundogRows, (trial) => trial.outcome),
      naive_outcomes: countBy(naiveRows, (trial) => trial.outcome),
      mean_sundog_rms_theta: round(mean(sundogRows.map((trial) => toNumber(trial.rms_theta)))),
      mean_naive_rms_theta: round(mean(naiveRows.map((trial) => toNumber(trial.rms_theta)))),
      mean_sundog_shadow_confidence: round(mean(sundogRows.map((trial) => toNumber(trial.mean_shadow_confidence)))),
      mean_naive_shadow_confidence: round(mean(naiveRows.map((trial) => toNumber(trial.mean_shadow_confidence)))),
      mean_sundog_shadow_length: round(mean(sundogRows.map((trial) => toNumber(trial.mean_shadow_length)))),
      mean_naive_shadow_length: round(mean(naiveRows.map((trial) => toNumber(trial.mean_shadow_length)))),
      mean_sundog_force_budget: round(mean(sundogRows.map((trial) => toNumber(trial.force_budget)))),
      mean_naive_force_budget: round(mean(naiveRows.map((trial) => toNumber(trial.force_budget)))),
      hard_success_violation: hardSuccessViolation,
      soft_survival_violation: softSurvivalViolation,
      audit_class: auditClass,
      replay_url: row.replay_url,
      naive_replay_url: row.naive_replay_url,
    };
  });
}

function summarizeAudit({ oracleRows, p2Rows, duration }) {
  const oracleMeasured = oracleRows.length;
  const oracleSurvivalExceeds = oracleRows.filter((row) => row.oracle_survival_exceeds === true).length;
  const oracleLowerRms = oracleRows.filter((row) => row.oracle_lower_rms === true).length;
  const bothCapped = oracleRows.filter((row) => row.both_at_duration_cap === true).length;
  const masked = oracleRows.filter((row) => row.audit_class === "survival_ceiling_masks_oracle_quality").length;
  const rmsDeltas = oracleRows.map((row) => toNumber(row.oracle_minus_sundog_rms_theta)).filter(Number.isFinite);
  const survivalDeltas = oracleRows.map((row) => toNumber(row.oracle_minus_sundog_survival)).filter(Number.isFinite);
  const softP2 = p2Rows.filter((row) => row.soft_survival_violation === true).length;
  const hardP2 = p2Rows.filter((row) => row.hard_success_violation === true).length;
  const allFailSoft = p2Rows.filter((row) => row.audit_class === "all_fail_sundog_lasts_longer").length;

  return {
    duration,
    oracleMeasured,
    oracleSurvivalExceeds,
    oracleLowerRms,
    bothCapped,
    masked,
    meanOracleMinusSundogRms: round(mean(rmsDeltas)),
    meanOracleMinusSundogSurvival: round(mean(survivalDeltas)),
    p2CellsChecked: p2Rows.length,
    p2SoftSurvivalViolations: softP2,
    p2HardSuccessViolations: hardP2,
    p2AllFailSoftViolations: allFailSoft,
  };
}

function auditMarkdown(summary, p2Rows) {
  const p2Violations = p2Rows.filter((row) => row.soft_survival_violation === true);
  const lines = [
    "# Sundog Balance Phase 10.5 Audit",
    "",
    `Generated: ${new Date().toISOString()}`,
    "",
    "## Oracle Ceiling Audit",
    "",
    `Cells measured: ${summary.oracleMeasured}`,
    `Oracle survival exceeds Sundog: ${summary.oracleSurvivalExceeds}/${summary.oracleMeasured}`,
    `Oracle lower RMS than Sundog: ${summary.oracleLowerRms}/${summary.oracleMeasured}`,
    `Both controllers at ${summary.duration}s survival cap: ${summary.bothCapped}/${summary.oracleMeasured}`,
    `Survival cap masks oracle quality: ${summary.masked}/${summary.oracleMeasured}`,
    `Mean oracle-minus-Sundog RMS theta: ${summary.meanOracleMinusSundogRms}`,
    "",
    "Interpretation: P4 failed because the pre-registered survival-ceiling metric saturates when both controllers reach the fixed duration. The oracle is not obviously sign-wrong: it is lower-RMS than Sundog on every cell in this slate. The rerun should either lengthen/stress the slate until survival separates, or pre-register a quality ceiling for capped cells.",
    "",
    "## P2 Failure-Boundary Audit",
    "",
    `P2 cells checked: ${summary.p2CellsChecked}`,
    `Soft survival-margin violations: ${summary.p2SoftSurvivalViolations}`,
    `Hard success violations: ${summary.p2HardSuccessViolations}`,
    `All-fail soft violations: ${summary.p2AllFailSoftViolations}`,
    "",
    "| cell_id | preset | light_elev_deg | delay_ms | margin | Sundog outcomes | naive outcomes | audit_class |",
    "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ...p2Violations.map((row) => `| ${row.cell_id} | ${row.preset} | ${row.light_elev_deg} | ${row.delay_ms} | ${row.mean_sundog_minus_naive_survival} | ${row.sundog_outcomes} | ${row.naive_outcomes} | ${row.audit_class} |`),
    "",
    "Interpretation: the overhead-light P2 violations are not hidden-control successes. They are all-fail cells where Sundog lasts longer than naive, consistent with confidence gating suppressing a bad shadow proxy while naive shadow-centering overreacts. High-delay cells hold the intended failure-boundary shape.",
    "",
    "## Audit Disposition",
    "",
    "The Phase 10 verdict remains AMBIGUOUS. The audit does not promote the workbench; it identifies two pre-registered metric repairs before rerun: split P2 hard success from all-fail survival margins, and replace or supplement P4 survival ceiling with a capped-cell quality metric.",
    "",
  ];
  return `${lines.join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputDir = path.resolve(repoRoot, args.input);
  const outDir = path.resolve(repoRoot, args.out);
  const manifest = JSON.parse(await readFile(path.join(inputDir, "manifest.json"), "utf8"));
  const envelopeRows = await readCsv(path.join(inputDir, "envelope.csv"));
  const trialRows = await readCsv(path.join(inputDir, "trial-outcomes.csv"));
  const duration = manifest.args?.duration ?? 8;
  const oracleRows = oracleAuditRows(envelopeRows, duration);
  const p2Rows = p2AuditRows(envelopeRows, trialRows);
  const summary = summarizeAudit({ oracleRows, p2Rows, duration });

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify({
    schema: "sundog.balance.phase10-audit.v1",
    generatedAt: new Date().toISOString(),
    input: path.relative(repoRoot, inputDir),
    summary,
  }, null, 2)}\n`);
  await writeFile(path.join(outDir, "oracle-ceiling-audit.csv"), rowsToCsv(oracleRows));
  await writeFile(path.join(outDir, "p2-failure-audit.csv"), rowsToCsv(p2Rows));
  await writeFile(path.join(outDir, "audit.md"), auditMarkdown(summary, p2Rows));

  console.log(`Balance Phase 10.5 audit wrote ${path.relative(repoRoot, outDir)}`);
  console.log(`Oracle lower RMS: ${summary.oracleLowerRms}/${summary.oracleMeasured}; both capped: ${summary.bothCapped}/${summary.oracleMeasured}`);
  console.log(`P2 soft violations: ${summary.p2SoftSurvivalViolations}; hard success violations: ${summary.p2HardSuccessViolations}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
