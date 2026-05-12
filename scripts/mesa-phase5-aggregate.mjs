import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const phase2Root = path.join(repoRoot, "results/mesa/phase2-matched-capacity");
const phase3Root = path.join(repoRoot, "results/mesa/phase3-probe-slate");
const phase4Root = path.join(repoRoot, "results/mesa/phase4-intervention-battery");
const outputRoot = path.join(repoRoot, "results/mesa/phase5-selection-pressure");

const POLICIES = Object.freeze([
  {
    axis: "A",
    policy_id: "mixed_lambda_0_1",
    family: "L-Mixed",
    label: "L-Mixed lambda=0.1",
    lambda: 0.1,
    training_slug: "mixed_ppo_phase3_lambda_0_1_small_seed_0_phase5_lambda_0_1",
    phase3_dir: "phase5_l_mixed_lambda_0_1_small",
    phase4_dir: "phase5_l_mixed_lambda_0_1_small",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_3",
    family: "L-Mixed",
    label: "L-Mixed lambda=0.3",
    lambda: 0.3,
    training_slug: "mixed_ppo_phase3_lambda_0_3_small_seed_0_phase5_lambda_0_3",
    phase3_dir: "phase5_l_mixed_lambda_0_3_small",
    phase4_dir: "phase5_l_mixed_lambda_0_3_small",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_5",
    family: "L-Mixed",
    label: "L-Mixed lambda=0.5",
    lambda: 0.5,
    training_slug: "mixed_ppo_phase3_lambda_0_5_small_seed_0_phase3_canonical_1m",
    phase3_dir: "l_mixed_phase3_canonical_1m",
    phase4_dir: "l_mixed_phase3_canonical_1m",
    reused_from: "Phase 3 canonical",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_7",
    family: "L-Mixed",
    label: "L-Mixed lambda=0.7",
    lambda: 0.7,
    training_slug: "mixed_ppo_phase3_lambda_0_7_small_seed_0_phase5_lambda_0_7",
    phase3_dir: "phase5_l_mixed_lambda_0_7_small",
    phase4_dir: "phase5_l_mixed_lambda_0_7_small",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_9",
    family: "L-Mixed",
    label: "L-Mixed lambda=0.9",
    lambda: 0.9,
    training_slug: "mixed_ppo_phase3_lambda_0_9_small_seed_0_phase5_lambda_0_9",
    phase3_dir: "phase5_l_mixed_lambda_0_9_small",
    phase4_dir: "phase5_l_mixed_lambda_0_9_small",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_3_medium",
    family: "L-Mixed",
    label: "L-Mixed-M lambda=0.3",
    tier: "Medium",
    lambda: 0.3,
    training_slug: "mixed_ppo_phase3_lambda_0_3_medium_seed_0_medium_phase5_lambda_0_3_10m",
    phase3_dir: "l_mixed_medium_lambda_0_3",
    phase4_dir: "l_mixed_medium_lambda_0_3",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_5_medium",
    family: "L-Mixed",
    label: "L-Mixed-M lambda=0.5",
    tier: "Medium",
    lambda: 0.5,
    training_slug: "mixed_ppo_phase3_lambda_0_5_medium_seed_0_medium_phase3_canonical_10m",
    phase3_dir: "l_mixed_phase3_medium_canonical_10m",
    phase4_dir: "l_mixed_phase3_medium_10m",
    reused_from: "Phase 3 Medium canonical",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_7_medium",
    family: "L-Mixed",
    label: "L-Mixed-M lambda=0.7",
    tier: "Medium",
    lambda: 0.7,
    training_slug: "mixed_ppo_phase3_lambda_0_7_medium_seed_0_medium_phase5_lambda_0_7_10m",
    phase3_dir: "l_mixed_medium_lambda_0_7",
    phase4_dir: "l_mixed_medium_lambda_0_7",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_8_medium_v3",
    family: "L-Mixed",
    label: "L-Mixed-M lambda=0.8 v3",
    tier: "Medium",
    lambda: 0.8,
    training_slug: "mixed_ppo_phase3_lambda_0_8_medium_seed_0_medium_phase5_v3_lambda_0_8_10m",
    phase3_dir: "phase5_v3_l_mixed_medium_lambda_0_8",
    phase4_dir: "phase5_v3_l_mixed_medium_lambda_0_8",
  },
  {
    axis: "A",
    policy_id: "mixed_lambda_0_9_medium_v3",
    family: "L-Mixed",
    label: "L-Mixed-M lambda=0.9 v3",
    tier: "Medium",
    lambda: 0.9,
    training_slug: "mixed_ppo_phase3_lambda_0_9_medium_seed_0_medium_phase5_v3_lambda_0_9_10m",
    phase3_dir: "phase5_v3_l_mixed_medium_lambda_0_9",
    phase4_dir: "phase5_v3_l_mixed_medium_lambda_0_9",
  },
  {
    axis: "A",
    policy_id: "reward_lambda_1_0_medium_anchor",
    family: "L-Reward",
    label: "L-Reward-M lambda=1.0 anchor",
    tier: "Medium",
    lambda: 1.0,
    training_slug: "reward_ppo_phase3_medium_seed_0_medium_phase3_canonical_10m",
    phase3_dir: "l_reward_phase3_medium_canonical_10m",
    phase4_dir: "l_reward_phase3_medium_10m",
    reused_from: "Phase 3 Medium canonical L-Reward anchor",
  },
  {
    axis: "B",
    policy_id: "signature_terminal",
    family: "L-Signature",
    label: "L-Signature terminal",
    signature_shape: "terminal",
    training_slug: "signature_ppo_terminal_small_seed_0_phase5",
    phase3_dir: "phase5_l_signature_terminal_small",
    phase4_dir: "phase5_l_signature_terminal_small",
  },
  {
    axis: "B",
    policy_id: "signature_integrated",
    family: "L-Signature",
    label: "L-Signature integrated",
    signature_shape: "integrated",
    training_slug: "signature_ppo_dense_small_seed_0_canonical_1m",
    phase3_dir: "l_signature_canonical_1m",
    phase4_dir: "l_signature_canonical_1m",
    reused_from: "Phase 2 canonical / Phase 3 evaluation",
  },
  {
    axis: "B",
    policy_id: "signature_threshold",
    family: "L-Signature",
    label: "L-Signature threshold",
    signature_shape: "threshold",
    training_slug: "signature_ppo_threshold_small_seed_0_phase5",
    phase3_dir: "phase5_l_signature_threshold_small",
    phase4_dir: "phase5_l_signature_threshold_small",
  },
  {
    axis: "B",
    policy_id: "signature_terminal_medium",
    family: "L-Signature",
    label: "L-Signature-M terminal",
    tier: "Medium",
    signature_shape: "terminal",
    training_slug: "signature_ppo_terminal_medium_seed_0_medium_phase5_terminal_10m",
    phase3_dir: "l_signature_medium_terminal",
    phase4_dir: "l_signature_medium_terminal",
  },
  {
    axis: "B",
    policy_id: "signature_integrated_medium",
    family: "L-Signature",
    label: "L-Signature-M integrated",
    tier: "Medium",
    signature_shape: "integrated",
    training_slug: "signature_ppo_dense_medium_seed_0_medium_canonical_10m",
    phase3_dir: "l_signature_medium_canonical_10m",
    phase4_dir: "l_signature_medium_10m",
    reused_from: "Phase 2/3 Medium canonical integrated signature",
  },
  {
    axis: "C",
    policy_id: "curriculum_sig_then_reward",
    family: "L-Curriculum",
    label: "Curriculum sig-then-reward",
    curriculum_order: "signature_then_reward",
    training_slug: "curriculum_sig_then_reward_small_seed_0_phase5",
    pretrain_slug: "signature_ppo_dense_small_seed_0_phase5_sig_pre_500k",
    phase3_dir: "phase5_curriculum_sig_then_reward_small",
    phase4_dir: "phase5_curriculum_sig_then_reward_small",
  },
  {
    axis: "C",
    policy_id: "curriculum_reward_then_sig",
    family: "L-Curriculum",
    label: "Curriculum reward-then-sig",
    curriculum_order: "reward_then_signature",
    training_slug: "curriculum_reward_then_sig_small_seed_0_phase5",
    pretrain_slug: "reward_ppo_phase3_small_seed_0_phase5_reward_pre_500k",
    phase3_dir: "phase5_curriculum_reward_then_sig_small",
    phase4_dir: "phase5_curriculum_reward_then_sig_small",
  },
  {
    axis: "C",
    policy_id: "curriculum_reward_then_terminal_sig_v3",
    family: "L-Curriculum",
    label: "Curriculum reward-then-terminal-sig v3",
    curriculum_order: "reward_then_terminal_signature",
    training_slug: "curriculum_reward_then_terminal_sig_small_seed_0_phase5_v3_reward_pre_terminal_sig_ft_500k",
    pretrain_slug: "reward_ppo_phase3_small_seed_0_phase5_reward_pre_500k",
    phase3_dir: "phase5_v3_curriculum_reward_then_terminal_sig_small",
    phase4_dir: "phase5_v3_curriculum_reward_then_terminal_sig_small",
  },
]);

function csvValue(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return `${columns.join(",")}\n${rows.map((row) => columns.map((col) => csvValue(row[col])).join(",")).join("\n")}\n`;
}

function parseCsv(text) {
  const rows = [];
  let field = "";
  let row = [];
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    if (row.some((value) => value !== "")) rows.push(row);
  }
  if (rows.length === 0) return [];
  const headers = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((header, i) => [header, values[i] ?? ""])));
}

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function round(value, digits = 6) {
  const n = num(value);
  return n === null ? "" : Number(n.toFixed(digits));
}

function tierRank(tier) {
  if (tier === "Small") return 0;
  if (tier === "Medium") return 1;
  return 99;
}

async function readJsonIfExists(file) {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch {
    return null;
  }
}

async function readFirstCsv(dir, suffix) {
  try {
    const entries = await readdir(dir);
    const match = entries.find((entry) => entry.endsWith(suffix));
    if (!match) return [];
    return parseCsv(await readFile(path.join(dir, match), "utf8"));
  } catch {
    return [];
  }
}

async function policyRow(policy) {
  const evalSummary = await readJsonIfExists(path.join(phase2Root, "logs", `${policy.training_slug}_evaluation_summary.json`));
  const pretrainSummary = policy.pretrain_slug
    ? await readJsonIfExists(path.join(phase2Root, "logs", `${policy.pretrain_slug}_evaluation_summary.json`))
    : null;
  const probeRows = await readFirstCsv(path.join(phase3Root, policy.phase3_dir), "_probe-degradation.csv");
  const interventionRows = await readFirstCsv(path.join(phase4Root, policy.phase4_dir), "_intervention-response.csv");
  const nominalProbe = probeRows.find((row) => row.cellId === "nominal") ?? {};
  const basinIntervention = interventionRows.find((row) => row.channel === "basin-position") ?? {};

  const totalProbeBasinCaptures = probeRows
    .filter((row) => row.cellId !== "nominal")
    .reduce((sum, row) => sum + (num(row.failureFalseBasinCapture) ?? 0), 0);
  const maxProbeBasinCaptures = Math.max(0, ...probeRows
    .filter((row) => row.cellId !== "nominal")
    .map((row) => num(row.failureFalseBasinCapture) ?? 0));
  const meanProbedSuccessRate = (() => {
    const values = probeRows
      .filter((row) => row.cellId !== "nominal")
      .map((row) => num(row.successRate))
      .filter((value) => value !== null);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  })();

  return {
    axis: policy.axis,
    policy_id: policy.policy_id,
    family: policy.family,
    policyLabel: policy.label,
    tier: policy.tier ?? "Small",
    lambda: policy.lambda ?? "",
    signature_shape: policy.signature_shape ?? "",
    curriculum_order: policy.curriculum_order ?? "",
    training_slug: policy.training_slug,
    pretrain_slug: policy.pretrain_slug ?? "",
    reused_from: policy.reused_from ?? "",
    success_count: evalSummary?.success_count ?? nominalProbe.successCount ?? "",
    seeds: evalSummary?.seeds ?? nominalProbe.n ?? "",
    success_rate: round(evalSummary?.success_rate ?? nominalProbe.successRate),
    mean_terminal_alignment: round(evalSummary?.mean_terminal_alignment ?? nominalProbe.meanTerminalAlignment),
    pretrain_success_rate: round(pretrainSummary?.success_rate),
    pretrain_mean_terminal_alignment: round(pretrainSummary?.mean_terminal_alignment),
    nominal_probe_success_count: nominalProbe.successCount ?? "",
    nominal_probe_success_rate: round(nominalProbe.successRate),
    mean_probed_success_rate: round(meanProbedSuccessRate),
    total_probe_false_basin_captures: totalProbeBasinCaptures,
    max_probe_false_basin_captures: maxProbeBasinCaptures,
    old_basin_pref: round(basinIntervention.mean_old_basin_preference),
    basin_action_response_L2: round(basinIntervention.mean_action_response_L2),
    signature_sensor_action_response_L2: round((interventionRows.find((row) => row.channel === "signature-sensor") ?? {}).mean_action_response_L2),
    geometry_action_response_L2: round((interventionRows.find((row) => row.channel === "geometry") ?? {}).mean_action_response_L2),
    phase3_dir: policy.phase3_dir,
    phase4_dir: policy.phase4_dir,
  };
}

function interpolateBreach(lambdaRows, threshold = 1.0) {
  const rows = lambdaRows
    .map((row) => ({ lambda: num(row.lambda), old: num(row.old_basin_pref) }))
    .filter((row) => row.lambda !== null && row.old !== null)
    .sort((a, b) => a.lambda - b.lambda);
  for (let i = 1; i < rows.length; i += 1) {
    const lo = rows[i - 1];
    const hi = rows[i];
    if (lo.old < threshold && hi.old >= threshold) {
      const t = (threshold - lo.old) / (hi.old - lo.old);
      return {
        threshold,
        lower_lambda: lo.lambda,
        lower_old_basin_pref: round(lo.old),
        upper_lambda: hi.lambda,
        upper_old_basin_pref: round(hi.old),
        interpolated_lambda: round(lo.lambda + t * (hi.lambda - lo.lambda)),
      };
    }
  }
  return { threshold, lower_lambda: "", lower_old_basin_pref: "", upper_lambda: "", upper_old_basin_pref: "", interpolated_lambda: "" };
}

function groupRowsBy(rows, keyFn) {
  const grouped = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }
  return grouped;
}

async function main() {
  await mkdir(path.join(outputRoot, "reports"), { recursive: true });
  const rows = await Promise.all(POLICIES.map(policyRow));
  const axisA = rows.filter((row) => row.axis === "A").sort((a, b) => tierRank(a.tier) - tierRank(b.tier) || num(a.lambda) - num(b.lambda));
  const axisB = rows.filter((row) => row.axis === "B").sort((a, b) => tierRank(a.tier) - tierRank(b.tier) || String(a.signature_shape).localeCompare(String(b.signature_shape)));
  const axisC = rows.filter((row) => row.axis === "C");

  const columns = [
    "axis", "policy_id", "family", "policyLabel", "tier", "lambda",
    "signature_shape", "curriculum_order", "training_slug", "pretrain_slug",
    "reused_from", "success_count", "seeds", "success_rate",
    "mean_terminal_alignment", "pretrain_success_rate",
    "pretrain_mean_terminal_alignment", "mean_probed_success_rate",
    "total_probe_false_basin_captures", "max_probe_false_basin_captures",
    "old_basin_pref", "basin_action_response_L2",
    "signature_sensor_action_response_L2", "geometry_action_response_L2",
    "phase3_dir", "phase4_dir",
  ];

  await writeFile(path.join(outputRoot, "policies-summary.csv"), toCsv(rows, columns), "utf8");
  await writeFile(path.join(outputRoot, "axis-a-lambda-sweep.csv"), toCsv(axisA, columns), "utf8");
  await writeFile(path.join(outputRoot, "axis-b-signature-shape.csv"), toCsv(axisB, columns), "utf8");
  await writeFile(path.join(outputRoot, "axis-c-curriculum.csv"), toCsv(axisC, columns), "utf8");

  await writeFile(path.join(outputRoot, "reports", "protection-curve.csv"), toCsv(axisA.map((row) => ({
    tier: row.tier,
    lambda: row.lambda,
    policyLabel: row.policyLabel,
    old_basin_pref: row.old_basin_pref,
    success_rate: row.success_rate,
    mean_terminal_alignment: row.mean_terminal_alignment,
    mean_probed_success_rate: row.mean_probed_success_rate,
    total_probe_false_basin_captures: row.total_probe_false_basin_captures,
  })), ["tier", "lambda", "policyLabel", "old_basin_pref", "success_rate", "mean_terminal_alignment", "mean_probed_success_rate", "total_probe_false_basin_captures"]), "utf8");
  const axisAByTier = groupRowsBy(axisA, (row) => row.tier);
  await writeFile(path.join(outputRoot, "reports", "breach-threshold.json"), `${JSON.stringify({
    metric: "old_basin_pref",
    threshold: 1.0,
    by_tier: [...axisAByTier.entries()].map(([tier, tierRows]) => ({
      tier,
      ...interpolateBreach(tierRows, 1.0),
    })),
    note: "Medium interpolation includes the L-Reward lambda=1.0 anchor; L-Mixed Medium rows remain below threshold through lambda=0.9.",
  }, null, 2)}\n`, "utf8");
  await writeFile(path.join(outputRoot, "reports", "sundog-cost-by-shape.csv"), toCsv(axisB.map((row) => ({
    tier: row.tier,
    signature_shape: row.signature_shape,
    policyLabel: row.policyLabel,
    success_rate: row.success_rate,
    success_count: row.success_count,
    mean_terminal_alignment: row.mean_terminal_alignment,
    old_basin_pref: row.old_basin_pref,
    total_probe_false_basin_captures: row.total_probe_false_basin_captures,
  })), ["tier", "signature_shape", "policyLabel", "success_rate", "success_count", "mean_terminal_alignment", "old_basin_pref", "total_probe_false_basin_captures"]), "utf8");
  await writeFile(path.join(outputRoot, "reports", "curriculum-persistence.csv"), toCsv(axisC.map((row) => ({
    curriculum_order: row.curriculum_order,
    pretrain_success_rate: row.pretrain_success_rate,
    pretrain_mean_terminal_alignment: row.pretrain_mean_terminal_alignment,
    final_success_rate: row.success_rate,
    final_mean_terminal_alignment: row.mean_terminal_alignment,
    old_basin_pref: row.old_basin_pref,
    total_probe_false_basin_captures: row.total_probe_false_basin_captures,
  })), ["curriculum_order", "pretrain_success_rate", "pretrain_mean_terminal_alignment", "final_success_rate", "final_mean_terminal_alignment", "old_basin_pref", "total_probe_false_basin_captures"]), "utf8");

  const manifest = {
    phase: "phase5-selection-pressure",
    created_at: new Date().toISOString(),
    tier: "Small+Medium",
    policy_count: rows.length,
    axes: {
      A: "L-Mixed lambda sweep",
      B: "L-Signature objective shape",
      C: "Curriculum order",
    },
    inputs: {
      phase2Root: path.relative(repoRoot, phase2Root).replaceAll("\\", "/"),
      phase3Root: path.relative(repoRoot, phase3Root).replaceAll("\\", "/"),
      phase4Root: path.relative(repoRoot, phase4Root).replaceAll("\\", "/"),
    },
  };
  await writeFile(path.join(outputRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log(`mesa-phase5-aggregate: ${rows.length} policies -> ${path.relative(repoRoot, outputRoot).replaceAll("\\", "/")}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
