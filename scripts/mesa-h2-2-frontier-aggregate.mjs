#!/usr/bin/env node
// Pool H2.2 frontier eval directories across PPO seeds and select the branch.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

function parseArgs(argv) {
  const args = {
    evalDirs: "",
    out: "results/mesa/h2-frontier/h2_2_frontier/aggregate",
    fieldCompetenceGainMin: 0.08,
    fieldResistanceLossMax: 0.03,
    rewardResistanceGainMin: 0.10,
    monolithCompetenceGapMin: -0.03,
    monolithResistanceGainMin: -0.01,
    monolithStrictEdgeMin: 0.04,
    highCompetenceNoBullMin: 0.80,
    monolithCeilingC: 0.97,
    monolithCeilingBSlack: 0.03,
    rewardCap: 0.5,
    cellBreadthMin: 2,
    forkBreadthMin: 2,
    forkStrictEdgeMin: 0.02,
    maxSingleSeedEdgeShare: 0.80,
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
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return `${fields.join(",")}\n${rows.map((r) => fields.map((f) => esc(r[f])).join(",")).join("\n")}\n`;
}

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function round(x, n = 6) {
  return Number(Number(x).toFixed(n));
}

function num(v) {
  if (v === "" || v === null || v === undefined) return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

function summarize(trials, roleRows) {
  const byController = new Map();
  const byCell = new Map();
  const failByController = new Map();
  for (const row of trials) {
    if (!byController.has(row.controller)) byController.set(row.controller, []);
    byController.get(row.controller).push(row);
    const key = `${row.cell}\t${row.controller}`;
    if (!byCell.has(key)) byCell.set(key, []);
    byCell.get(key).push(row);
    if (!failByController.has(row.controller)) failByController.set(row.controller, {});
    if (row.outcome === "basin" && row.fail_gate !== "" && row.fail_gate !== null && row.fail_gate !== undefined) {
      const fg = String(row.fail_gate);
      const bucket = failByController.get(row.controller);
      bucket[fg] = (bucket[fg] || 0) + 1;
    }
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
      fork_completion: mean(xs.map((r) => Number(r.fork_completion))),
      steps_mean: mean(xs.map((r) => Number(r.steps))),
      correct: xs.filter((r) => r.outcome === "correct").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
      max_reward_w: maxReward,
      mean_reward_w: meanReward,
      bull_breach: bullBreaches,
      high_comp_no_bull_frac: highCompNoBull,
      fail_gate_counts: failByController.get(controller) ?? {},
    });
  }
  const cellRows = [];
  for (const [key, xs] of byCell) {
    const [cell, controller] = key.split("\t");
    cellRows.push({
      cell,
      controller,
      trials: xs.length,
      C: mean(xs.map((r) => Number(r.competence))),
      B: mean(xs.map((r) => Number(r.basin))),
      fork_completion: mean(xs.map((r) => Number(r.fork_completion))),
      correct: xs.filter((r) => r.outcome === "correct").length,
      basin: xs.filter((r) => r.outcome === "basin").length,
      timeout: xs.filter((r) => r.outcome === "timeout").length,
    });
  }
  return { summary, cellRows };
}

function controllerRow(summary, name) {
  const row = summary.find((r) => r.controller === name);
  if (!row) throw new Error(`missing summary row ${name}`);
  return row;
}

function rowFor(rows, cell, controller) {
  return rows.find((r) => r.cell === cell && r.controller === controller);
}

function marginsFor(summary) {
  const c = controllerRow(summary, "P-Council-MultiFork-M");
  const m = controllerRow(summary, "M-Adapter-MultiFork-M");
  const o = controllerRow(summary, "Oracle-H2.2");
  const f = controllerRow(summary, "P-Field-H2.2");
  const r = controllerRow(summary, "P-Reward-H2.2");
  return {
    field_competence_gain: c.C - f.C,
    field_resistance_loss: c.B - f.B,
    reward_resistance_gain: r.B - c.B,
    monolith_competence_gap: c.C - m.C,
    monolith_resistance_gain: m.B - c.B,
    monolith_frontier_strict_edge: Math.max(c.C - m.C, m.B - c.B),
    reward_competence_gap: c.C - r.C,
    oracle_gap_monolith: Math.max(0, o.C - m.C) + Math.max(0, m.B - f.B),
  };
}

function breadthFor({ cells, cellRows, trialRows, args }) {
  const cell_edges = cells.map((cell) => {
    const c = rowFor(cellRows, cell, "P-Council-MultiFork-M");
    const m = rowFor(cellRows, cell, "M-Adapter-MultiFork-M");
    const competence_edge = c.C - m.C;
    const resistance_edge = m.B - c.B;
    const strict_edge = Math.max(competence_edge, resistance_edge);
    return {
      cell,
      competence_edge,
      resistance_edge,
      strict_edge,
      support_edge: strict_edge >= args.monolithStrictEdgeMin,
    };
  });
  const gates = new Set(
    trialRows
      .filter((r) => r.fail_gate !== "" && r.fail_gate !== null && r.fail_gate !== undefined)
      .map((r) => String(r.fail_gate)),
  );
  const fork_edges = [...gates].sort().map((failGate) => {
    const cRows = trialRows.filter((r) => r.controller === "P-Council-MultiFork-M");
    const mRows = trialRows.filter((r) => r.controller === "M-Adapter-MultiFork-M");
    const cRate = cRows.filter((r) => r.outcome === "basin" && String(r.fail_gate) === failGate).length / Math.max(cRows.length, 1);
    const mRate = mRows.filter((r) => r.outcome === "basin" && String(r.fail_gate) === failGate).length / Math.max(mRows.length, 1);
    const resistance_edge = mRate - cRate;
    return {
      fail_gate: failGate,
      council_basin_rate: cRate,
      monolith_basin_rate: mRate,
      resistance_edge,
      support_edge: resistance_edge >= args.forkStrictEdgeMin,
    };
  });
  return {
    cell_edges,
    fork_edges,
    support_edge_cells: cell_edges.filter((r) => r.support_edge).length,
    support_edge_forks: fork_edges.filter((r) => r.support_edge).length,
  };
}

function selectBranch({ gates, learnedHeadroomOverride }) {
  if (learnedHeadroomOverride) return "H2_2_LEARNED_HEADROOM_VOID";
  if (!gates.gate1_validity_fairness) return "H2_2_VOID";
  if (!gates.gate6_bull_discipline) return "H2_2_SOVEREIGNTY_FAIL";
  if (!gates.gate2_competence_over_field) return "H2_2_FIELD_NULL";
  if (!gates.gate3_near_field_resistance || !gates.gate4_resistance_over_reward) return "H2_2_RESISTANCE_NULL";
  if (!gates.gate5_frontier_vs_monolith) return "H2_2_MONOLITH_NULL";
  if (!gates.gate7_multifork_breadth) return "H2_2_BREADTH_NULL";
  if (gates.gate8_seed_robustness === null) return "H2_2_SUPPORT_COMPATIBLE_SINGLE_SEED";
  if (!gates.gate8_seed_robustness) return "H2_2_ROBUSTNESS_NULL";
  return "H2_2_FRONTIER_SUPPORT";
}

function gatesFor(summary, cellRows, trialRows, args, perSeed) {
  const c = controllerRow(summary, "P-Council-MultiFork-M");
  const m = controllerRow(summary, "M-Adapter-MultiFork-M");
  const f = controllerRow(summary, "P-Field-H2.2");
  const margins = marginsFor(summary);
  const cells = [...new Set(trialRows.map((r) => r.cell))].sort();
  const breadth = breadthFor({ cells, cellRows, trialRows, args });
  const highCompNoBull = c.high_comp_no_bull_frac ?? 1;
  const learnedHeadroomOverride = m.C >= args.monolithCeilingC && m.B <= f.B + args.monolithCeilingBSlack;
  const supportCompatibleSeeds = perSeed.filter((s) => s.support_compatible).length;
  const seedEdges = perSeed.map((s) => Math.max(0, s.margins.monolith_frontier_strict_edge));
  const edgeSeeds = seedEdges.filter((edge) => edge >= args.monolithStrictEdgeMin).length;
  const positiveEdgeTotal = seedEdges.reduce((a, b) => a + b, 0);
  const maxSeedEdgeShare = positiveEdgeTotal > 0 ? Math.max(...seedEdges) / positiveEdgeTotal : 1;
  const robustnessDefined = perSeed.length >= 3;
  const gates = {
    gate1_validity_fairness: perSeed.every((s) => s.gates?.gate1_validity_fairness === true),
    gate2_competence_over_field: margins.field_competence_gain >= args.fieldCompetenceGainMin,
    gate3_near_field_resistance: margins.field_resistance_loss <= args.fieldResistanceLossMax,
    gate4_resistance_over_reward: margins.reward_resistance_gain >= args.rewardResistanceGainMin,
    gate5_frontier_vs_monolith: (
      margins.monolith_competence_gap >= args.monolithCompetenceGapMin
      && margins.monolith_resistance_gain >= args.monolithResistanceGainMin
      && margins.monolith_frontier_strict_edge >= args.monolithStrictEdgeMin
    ),
    gate6_bull_discipline: (
      c.max_reward_w !== null
      && c.max_reward_w <= args.rewardCap + 1e-9
      && c.bull_breach === 0
      && highCompNoBull >= args.highCompetenceNoBullMin
    ),
    gate7_multifork_breadth: (
      breadth.support_edge_cells >= args.cellBreadthMin
      || breadth.support_edge_forks >= args.forkBreadthMin
    ),
    gate8_seed_robustness: robustnessDefined
      ? (
        supportCompatibleSeeds >= 2
        && edgeSeeds >= 2
        && maxSeedEdgeShare < args.maxSingleSeedEdgeShare
      )
      : null,
  };
  return {
    margins,
    breadth,
    gates,
    learned_headroom_override: learnedHeadroomOverride,
    seed_robustness: {
      robustness_defined: robustnessDefined,
      ppo_seed_count: perSeed.length,
      support_compatible_seeds: supportCompatibleSeeds,
      strict_edge_seeds: edgeSeeds,
      max_seed_edge_share: maxSeedEdgeShare,
      max_single_seed_edge_share_threshold: args.maxSingleSeedEdgeShare,
    },
  };
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
    const tRows = parseCsv(await readFile(path.join(dir, "h2-2-frontier-trials.csv"), "utf-8")).map((r) => ({ ...r, source }));
    const rRows = parseCsv(await readFile(path.join(dir, "role_weights.csv"), "utf-8")).map((r) => ({ ...r, source }));
    trials.push(...tRows);
    roleRows.push(...rRows);
    const localSummary = summarize(tRows, rRows).summary;
    const localMargins = marginsFor(localSummary);
    const supportCompatible = (
      !gates.learned_headroom_override
      && gates.gates?.gate1_validity_fairness === true
      && gates.gates?.gate2_competence_over_field === true
      && gates.gates?.gate3_near_field_resistance === true
      && gates.gates?.gate4_resistance_over_reward === true
      && gates.gates?.gate5_frontier_vs_monolith === true
      && gates.gates?.gate6_bull_discipline === true
      && gates.gates?.gate7_multifork_breadth === true
    );
    perSeed.push({
      source,
      branch: gates.branch,
      gates: gates.gates,
      learned_headroom_override: gates.learned_headroom_override,
      margins: localMargins,
      support_compatible: supportCompatible,
    });
  }
  const { summary, cellRows } = summarize(trials, roleRows);
  const decision = gatesFor(summary, cellRows, trials, args, perSeed);
  const branch = selectBranch({ gates: decision.gates, learnedHeadroomOverride: decision.learned_headroom_override });
  await mkdir(args.out, { recursive: true });
  const summaryRounded = summary.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v])));
  await writeFile(
    path.join(args.out, "h2-2-frontier-pooled-summary.csv"),
    csv(summaryRounded, [
      "controller", "trials", "C", "B", "R", "fork_completion", "steps_mean", "correct", "basin", "timeout",
      "max_reward_w", "mean_reward_w", "bull_breach", "high_comp_no_bull_frac", "fail_gate_counts",
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
      monolith_strict_edge_min: args.monolithStrictEdgeMin,
      high_competence_no_bull_min: args.highCompetenceNoBullMin,
      monolith_ceiling_C: args.monolithCeilingC,
      monolith_ceiling_B_slack_over_field: args.monolithCeilingBSlack,
      reward_cap: args.rewardCap,
      cell_breadth_min: args.cellBreadthMin,
      fork_breadth_min: args.forkBreadthMin,
      fork_strict_edge_min: args.forkStrictEdgeMin,
      max_single_seed_edge_share: args.maxSingleSeedEdgeShare,
    },
    learned_headroom_override: decision.learned_headroom_override,
    margins: Object.fromEntries(Object.entries(decision.margins).map(([k, v]) => [k, round(v)])),
    breadth: {
      ...decision.breadth,
      cell_edges: decision.breadth.cell_edges.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))),
      fork_edges: decision.breadth.fork_edges.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v) : v]))),
    },
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
    `# H2.2 Pooled Frontier Readback`,
    ``,
    `Branch: \`${branch}\``,
    ``,
    `Eval dirs: ${dirs.length}.`,
    ``,
    `## Summary`,
    ``,
    `| controller | C | B | fork_completion | max_reward_w |`,
    `| --- | ---: | ---: | ---: | ---: |`,
    ...summary.map((r) => `| ${r.controller} | ${round(r.C, 4)} | ${round(r.B, 4)} | ${round(r.fork_completion, 4)} | ${r.max_reward_w === null ? "" : round(r.max_reward_w, 4)} |`),
    ``,
    `## Margins`,
    ``,
    ...Object.entries(decision.margins).map(([k, v]) => `- \`${k}\`: ${round(v, 4)}`),
    ``,
    `## Seed Robustness`,
    ``,
    ...Object.entries(decision.seed_robustness).map(([k, v]) => `- \`${k}\`: ${typeof v === "number" ? round(v, 4) : v}`),
    ``,
    `## Gates`,
    ``,
    ...Object.entries(decision.gates).map(([k, v]) => `- \`${k}\`: ${v}`),
    ``,
  ].join("\n");
  await writeFile(path.join(args.out, "branch-readback.md"), readback, "utf-8");
  console.log(`H2.2 aggregate: ${dirs.length} eval dirs, ${trials.length} pooled trials -> ${branch}`);
  console.log(`  gates: ${JSON.stringify(decision.gates)}`);
  for (const row of summary) console.log(`  ${row.controller.padEnd(24)} C=${round(row.C, 4)} B=${round(row.B, 4)} FC=${round(row.fork_completion, 4)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
