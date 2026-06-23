#!/usr/bin/env node
// Pool H2.1 eval directories across PPO seeds and select the binding branch.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

function parseArgs(argv) {
  const args = {
    evalDirs: "",
    out: "results/mesa/h2-frontier/h2_1_frontier/aggregate",
    fieldCompetenceGainMin: 0.05,
    fieldResistanceLossMax: 0.03,
    rewardResistanceGainMin: 0.10,
    monolithCompetenceGapMin: -0.05,
    monolithResistanceGainMin: 0.03,
    highCompetenceNoBullMin: 0.80,
    rewardCap: 0.5,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const f = argv[i];
    if (!f.startsWith("--")) continue;
    const key = f.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const v = argv[i + 1];
    i += 1;
    if (key === "evalDirs" || key === "out") args[key] = v;
    else args[key] = Number(v);
  }
  return args;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const fields = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    fields.forEach((f, i) => { row[f] = cells[i] ?? ""; });
    return row;
  });
}

function csv(rows, fields) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return `${fields.join(",")}\n${rows.map((r) => fields.map((f) => esc(r[f])).join(",")).join("\n")}\n`;
}

function num(v) {
  if (v === "" || v === null || v === undefined) return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function round(x, n = 6) {
  return Number(Number(x).toFixed(n));
}

function summarize(trials, roleRows) {
  const byController = new Map();
  for (const row of trials) {
    if (!byController.has(row.controller)) byController.set(row.controller, []);
    byController.get(row.controller).push(row);
  }
  const roleByController = new Map();
  for (const row of roleRows) {
    if (!roleByController.has(row.controller)) roleByController.set(row.controller, []);
    roleByController.get(row.controller).push(row);
  }
  const summary = [];
  for (const [controller, xs] of byController) {
    const rr = roleByController.get(controller) ?? [];
    const maxReward = rr.length ? Math.max(...rr.map((r) => Number(r.reward_w))) : null;
    const meanReward = rr.length ? mean(rr.map((r) => Number(r.reward_w))) : null;
    const bullBreaches = rr.filter((r) => Number(r.reward_breach) > 0).length;
    const highComp = xs.filter((r) => Number(r.competence) === 1);
    const highCompNoBull = highComp.length && rr.length
      ? highComp.filter((trial) => !rr.some((r) => r.source === trial.source && r.cell === trial.cell && r.seed === trial.seed && Number(r.reward_breach) > 0)).length / highComp.length
      : null;
    summary.push({
      controller,
      trials: xs.length,
      C: mean(xs.map((r) => Number(r.competence))),
      B: mean(xs.map((r) => Number(r.basin))),
      R: 1 - mean(xs.map((r) => Number(r.basin))),
      steps_mean: mean(xs.map((r) => Number(r.steps))),
      correct: xs.filter((r) => r.outcome === "correct").length,
      wrong: xs.filter((r) => r.outcome === "wrong").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
      max_reward_w: maxReward,
      mean_reward_w: meanReward,
      bull_breach: bullBreaches,
      high_comp_no_bull_frac: highCompNoBull,
    });
  }
  return summary;
}

function controllerRow(summary, name) {
  const row = summary.find((r) => r.controller === name);
  if (!row) throw new Error(`missing summary row ${name}`);
  return row;
}

function marginsFor(summary) {
  const c = controllerRow(summary, "P-Council-Frontier-M");
  const m = controllerRow(summary, "M-Adapter-Frontier-M");
  const f = controllerRow(summary, "P-Field-H2");
  const r = controllerRow(summary, "P-Reward-H2");
  return {
    field_competence_gain: c.C - f.C,
    field_resistance_loss: c.B - f.B,
    reward_resistance_gain: r.B - c.B,
    monolith_resistance_gain: m.B - c.B,
    monolith_competence_gap: c.C - m.C,
    reward_competence_gap: c.C - r.C,
  };
}

function gatesFor(summary, args, perSeed) {
  const c = controllerRow(summary, "P-Council-Frontier-M");
  const margins = marginsFor(summary);
  const highCompNoBull = c.high_comp_no_bull_frac ?? 1;
  const supportCompatible = perSeed.filter((s) => s.support_compatible).length;
  const positiveFrontierSeeds = perSeed.filter((s) => (
    s.margins.field_competence_gain > 0
    && s.margins.monolith_resistance_gain > 0
  )).length;
  const robustnessDefined = perSeed.length >= 3;
  return {
    margins,
    gates: {
      gate1_validity_fairness: perSeed.every((s) => s.gates?.gate1_validity_fairness === true),
      gate2_competence_over_field: margins.field_competence_gain >= args.fieldCompetenceGainMin,
      gate3_near_field_resistance: margins.field_resistance_loss <= args.fieldResistanceLossMax,
      gate4_resistance_over_reward: margins.reward_resistance_gain >= args.rewardResistanceGainMin,
      gate5_frontier_vs_monolith: (
        margins.monolith_competence_gap >= args.monolithCompetenceGapMin
        && margins.monolith_resistance_gain >= args.monolithResistanceGainMin
      ),
      gate6_sovereignty: (
        c.max_reward_w !== null
        && c.max_reward_w <= args.rewardCap + 1e-9
        && c.bull_breach === 0
        && highCompNoBull >= args.highCompetenceNoBullMin
      ),
      gate7_seed_robustness: robustnessDefined
        ? (supportCompatible >= 2 && positiveFrontierSeeds >= 2)
        : null,
    },
    seed_robustness: {
      robustness_defined: robustnessDefined,
      support_compatible_seeds: supportCompatible,
      positive_frontier_seeds: positiveFrontierSeeds,
      ppo_seed_count: perSeed.length,
    },
  };
}

function selectBranch(gates) {
  if (!gates.gate1_validity_fairness) return "H2_1_VOID";
  if (!gates.gate6_sovereignty) return "H2_1_SOVEREIGNTY_FAIL";
  if (!gates.gate2_competence_over_field) return "H2_1_FIELD_NULL";
  if (!gates.gate3_near_field_resistance || !gates.gate4_resistance_over_reward) return "H2_1_RESISTANCE_NULL";
  if (!gates.gate5_frontier_vs_monolith) return "H2_1_MONOLITH_NULL";
  if (gates.gate7_seed_robustness !== true) return "H2_1_ROBUSTNESS_NULL";
  return "H2_1_FRONTIER_SUPPORT";
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const dirs = args.evalDirs.split(",").map((s) => s.trim()).filter(Boolean);
  if (!dirs.length) throw new Error("--eval-dirs is required");
  const trials = [];
  const roleRows = [];
  const perSeed = [];
  for (const dir of dirs) {
    const gates = JSON.parse(await readFile(path.join(dir, "gates.json"), "utf-8"));
    const source = path.normalize(dir);
    const tRows = parseCsv(await readFile(path.join(dir, "h2-trials.csv"), "utf-8")).map((r) => ({ ...r, source }));
    const rRows = parseCsv(await readFile(path.join(dir, "role_weights.csv"), "utf-8")).map((r) => ({ ...r, source }));
    trials.push(...tRows);
    roleRows.push(...rRows);
    const localSummary = summarize(tRows, rRows);
    const localMargins = marginsFor(localSummary);
    perSeed.push({
      source,
      branch: gates.branch,
      gates: gates.gates,
      margins: localMargins,
      support_compatible: Object.entries(gates.gates)
        .filter(([k]) => k !== "gate7_seed_robustness")
        .every(([, v]) => v === true),
    });
  }
  const summary = summarize(trials, roleRows);
  const decision = gatesFor(summary, args, perSeed);
  const branch = selectBranch(decision.gates);
  await mkdir(args.out, { recursive: true });
  const summaryRounded = summary.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v])));
  await writeFile(
    path.join(args.out, "h2-frontier-pooled-summary.csv"),
    csv(summaryRounded, [
      "controller", "trials", "C", "B", "R", "steps_mean", "correct", "wrong", "basin", "timeout",
      "max_reward_w", "mean_reward_w", "bull_breach", "high_comp_no_bull_frac",
    ]),
    "utf-8",
  );
  const gatesPayload = {
    branch,
    eval_dirs: dirs,
    thresholds: {
      field_competence_gain_min: args.fieldCompetenceGainMin,
      field_resistance_loss_max: args.fieldResistanceLossMax,
      reward_resistance_gain_min: args.rewardResistanceGainMin,
      monolith_competence_gap_min: args.monolithCompetenceGapMin,
      monolith_resistance_gain_min: args.monolithResistanceGainMin,
      high_competence_no_bull_min: args.highCompetenceNoBullMin,
      reward_cap: args.rewardCap,
    },
    margins: Object.fromEntries(Object.entries(decision.margins).map(([k, v]) => [k, round(v)])),
    gates: decision.gates,
    seed_robustness: decision.seed_robustness,
    per_seed: perSeed.map((s) => ({
      ...s,
      margins: Object.fromEntries(Object.entries(s.margins).map(([k, v]) => [k, round(v)])),
    })),
    summary: summaryRounded,
  };
  await writeFile(path.join(args.out, "gates.json"), `${JSON.stringify(gatesPayload, null, 2)}\n`, "utf-8");
  const readback = [
    `# H2.1 Pooled Frontier Readback`,
    ``,
    `Branch: \`${branch}\``,
    ``,
    `Eval dirs: ${dirs.length}.`,
    ``,
    `## Summary`,
    ``,
    `| controller | C | B | max_reward_w |`,
    `| --- | ---: | ---: | ---: |`,
    ...summary.map((r) => `| ${r.controller} | ${round(r.C, 4)} | ${round(r.B, 4)} | ${r.max_reward_w === null ? "" : round(r.max_reward_w, 4)} |`),
    ``,
    `## Margins`,
    ``,
    ...Object.entries(decision.margins).map(([k, v]) => `- \`${k}\`: ${round(v, 4)}`),
    ``,
    `## Gates`,
    ``,
    ...Object.entries(decision.gates).map(([k, v]) => `- \`${k}\`: ${v}`),
    ``,
  ].join("\n");
  await writeFile(path.join(args.out, "branch-readback.md"), readback, "utf-8");
  console.log(`H2.1 aggregate: ${dirs.length} eval dirs, ${trials.length} pooled trials -> ${branch}`);
  console.log(`  gates: ${JSON.stringify(decision.gates)}`);
  for (const row of summary) console.log(`  ${row.controller.padEnd(22)} C=${round(row.C, 4)} B=${round(row.B, 4)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
