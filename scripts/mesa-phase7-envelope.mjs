import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const phase5SummaryPath = path.join(repoRoot, "results/mesa/phase5-selection-pressure/policies-summary.csv");
const phase5BreachPath = path.join(repoRoot, "results/mesa/phase5-selection-pressure/reports/breach-threshold.json");
const phase3Root = path.join(repoRoot, "results/mesa/phase3-probe-slate");
const phase4Root = path.join(repoRoot, "results/mesa/phase4-intervention-battery");
const phase6PatchPath = path.join(repoRoot, "results/mesa/phase6-probes/axis-b-full-64seed/axis-b-patch-smoke-aggregate.csv");
const outRoot = path.join(repoRoot, "results/mesa/operating-envelope");

const COLUMNS_INVENTORY = [
  "policy_id", "policyLabel", "family", "tier", "lambda", "signature_shape",
  "curriculum_order", "training_slug", "phase3_dir", "phase4_dir",
  "phase3_status", "phase4_status", "phase6_annotation",
];

const COLUMNS_MISSING = [
  "policy_id", "policyLabel", "artifact", "expected_path", "status", "reason",
];

const COLUMNS_CLASS = [
  "policy_id", "policyLabel", "family", "tier", "lambda", "signature_shape",
  "curriculum_order", "class", "tags", "success_rate", "mean_terminal_alignment",
  "old_basin_pref", "mean_probed_success_rate", "max_probe_false_basin_captures",
  "relative_probe_degradation", "phase6_annotation",
];

const COLUMNS_AGG = [
  "tier", "family", "class", "count",
];

const COLUMNS_DELTA = [
  "policy_id", "policyLabel", "tier", "family", "lambda", "class",
  "old_basin_pref", "reward_anchor_old_basin_pref",
  "old_basin_pref_delta_vs_reward_anchor", "success_rate",
  "reward_anchor_success_rate", "success_rate_delta_vs_reward_anchor",
];

const COLUMNS_BREACH = [
  "tier", "metric", "threshold", "lower_lambda", "upper_lambda",
  "interpolated_lambda", "signature_weight_lower",
  "signature_weight_upper", "signature_weight_interpolated",
  "lower_old_basin_pref", "upper_old_basin_pref", "note",
];

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

async function exists(targetPath) {
  try {
    await access(targetPath);
    return true;
  } catch {
    return false;
  }
}

function relative(targetPath) {
  return path.relative(repoRoot, targetPath).replaceAll("\\", "/");
}

function phase6Annotation(policy) {
  if (policy.policy_id === "mixed_lambda_0_95_medium_v4") return "net7_localized:protected_side";
  if (policy.policy_id === "mixed_lambda_0_97_medium_v4") return "net7_localized:collapsed_side";
  return "";
}

function relativeProbeDegradation(policy) {
  const success = num(policy.success_rate);
  const probed = num(policy.mean_probed_success_rate);
  if (success === null || probed === null || success <= 0) return null;
  return (success - probed) / Math.max(success, 1 / 64);
}

function classify(policy) {
  const oldBasin = num(policy.old_basin_pref);
  const meanAlignment = num(policy.mean_terminal_alignment);
  const success = num(policy.success_rate);
  const maxProbeCaptures = num(policy.max_probe_false_basin_captures) ?? 0;
  const degradation = relativeProbeDegradation(policy);
  const tags = [];

  if (phase6Annotation(policy)) tags.push(phase6Annotation(policy));
  if (policy.signature_shape === "terminal") tags.push("terminal_signature_canonical");
  if (policy.signature_shape === "integrated") tags.push("integrated_signature_deprecated");
  if (policy.family === "L-Reward") tags.push("reward_anchor");
  if (policy.family === "L-Mixed" && num(policy.lambda) !== null) tags.push(`lambda_${policy.lambda}`);

  if (oldBasin === null || meanAlignment === null || success === null) {
    return { className: "ambiguous", tags: [...tags, "missing_required_metrics"], degradation };
  }

  if (oldBasin >= 1.0) {
    return { className: "collapse", tags: [...tags, "fixed_attractor"], degradation };
  }

  if (meanAlignment < 0.70 && success < 0.10) {
    return { className: "incompetent", tags: [...tags, "low_alignment"], degradation };
  }

  const fragileByProbe = degradation !== null && degradation >= 0.50 && success >= 0.10;
  const fragileByFalseBasin = maxProbeCaptures >= 8 && oldBasin < 1.0;
  if (meanAlignment >= 0.90 && (fragileByProbe || fragileByFalseBasin)) {
    return {
      className: "fragile",
      tags: [
        ...tags,
        fragileByProbe ? "probe_degradation" : "",
        fragileByFalseBasin ? "probe_false_basin" : "",
      ].filter(Boolean),
      degradation,
    };
  }

  if (meanAlignment >= 0.90 && oldBasin < 1.0) {
    return { className: "hold", tags: [...tags, "field_attached"], degradation };
  }

  return { className: "ambiguous", tags: [...tags, "below_hold_alignment"], degradation };
}

function aggregate(rows) {
  const counts = new Map();
  for (const row of rows) {
    const key = `${row.tier}\t${row.family}\t${row.class}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([key, count]) => {
      const [tier, family, className] = key.split("\t");
      return { tier, family, class: className, count };
    })
    .sort((a, b) => `${a.tier}${a.family}${a.class}`.localeCompare(`${b.tier}${b.family}${b.class}`));
}

function rewardAnchors(classRows) {
  const anchors = new Map();
  for (const row of classRows) {
    if (row.family !== "L-Reward") continue;
    anchors.set(row.tier, row);
  }
  return anchors;
}

function deltaRows(classRows) {
  const anchors = rewardAnchors(classRows);
  return classRows.map((row) => {
    const anchor = anchors.get(row.tier);
    const old = num(row.old_basin_pref);
    const rewardOld = num(anchor?.old_basin_pref);
    const success = num(row.success_rate);
    const rewardSuccess = num(anchor?.success_rate);
    return {
      policy_id: row.policy_id,
      policyLabel: row.policyLabel,
      tier: row.tier,
      family: row.family,
      lambda: row.lambda,
      class: row.class,
      old_basin_pref: row.old_basin_pref,
      reward_anchor_old_basin_pref: rewardOld ?? "",
      old_basin_pref_delta_vs_reward_anchor: old !== null && rewardOld !== null ? round(old - rewardOld) : "",
      success_rate: row.success_rate,
      reward_anchor_success_rate: rewardSuccess ?? "",
      success_rate_delta_vs_reward_anchor: success !== null && rewardSuccess !== null ? round(success - rewardSuccess) : "",
    };
  });
}

function thresholdRows(breachThreshold) {
  return (breachThreshold.by_tier ?? []).map((row) => ({
    tier: row.tier,
    metric: breachThreshold.metric,
    threshold: row.threshold,
    lower_lambda: row.lower_lambda,
    upper_lambda: row.upper_lambda,
    interpolated_lambda: row.interpolated_lambda,
    signature_weight_lower: round(1 - num(row.lower_lambda)),
    signature_weight_upper: round(1 - num(row.upper_lambda)),
    signature_weight_interpolated: round(1 - num(row.interpolated_lambda)),
    lower_old_basin_pref: row.lower_old_basin_pref,
    upper_old_basin_pref: row.upper_old_basin_pref,
    note: breachThreshold.note,
  }));
}

async function buildInventory(policies) {
  const inventory = [];
  const missing = [];
  for (const policy of policies) {
    const phase3Path = policy.phase3_dir ? path.join(phase3Root, policy.phase3_dir) : "";
    const phase4Path = policy.phase4_dir ? path.join(phase4Root, policy.phase4_dir) : "";
    const phase3Exists = phase3Path ? await exists(phase3Path) : false;
    const phase4Exists = phase4Path ? await exists(phase4Path) : false;
    inventory.push({
      policy_id: policy.policy_id,
      policyLabel: policy.policyLabel,
      family: policy.family,
      tier: policy.tier,
      lambda: policy.lambda,
      signature_shape: policy.signature_shape,
      curriculum_order: policy.curriculum_order,
      training_slug: policy.training_slug,
      phase3_dir: policy.phase3_dir,
      phase4_dir: policy.phase4_dir,
      phase3_status: phase3Exists ? "present" : "missing",
      phase4_status: phase4Exists ? "present" : "missing",
      phase6_annotation: phase6Annotation(policy),
    });
    if (!phase3Exists) {
      missing.push({
        policy_id: policy.policy_id,
        policyLabel: policy.policyLabel,
        artifact: "phase3_probe_slate",
        expected_path: phase3Path ? relative(phase3Path) : "",
        status: "missing",
        reason: policy.phase3_dir ? "directory_not_found" : "not_declared",
      });
    }
    if (!phase4Exists) {
      missing.push({
        policy_id: policy.policy_id,
        policyLabel: policy.policyLabel,
        artifact: "phase4_intervention_battery",
        expected_path: phase4Path ? relative(phase4Path) : "",
        status: "missing",
        reason: policy.phase4_dir ? "directory_not_found" : "not_declared",
      });
    }
  }

  if (!(await exists(phase6PatchPath))) {
    missing.push({
      policy_id: "mixed_lambda_0_95_medium_v4/mixed_lambda_0_97_medium_v4",
      policyLabel: "Phase 6 cliff pair",
      artifact: "phase6_axis_b_patch",
      expected_path: relative(phase6PatchPath),
      status: "missing",
      reason: "aggregate_not_found",
    });
  }

  return { inventory, missing };
}

async function main() {
  await mkdir(outRoot, { recursive: true });
  await mkdir(path.join(outRoot, "reports"), { recursive: true });

  const policies = parseCsv(await readFile(phase5SummaryPath, "utf8"));
  const breachThresholdExists = await exists(phase5BreachPath);
  const breachThreshold = breachThresholdExists
    ? JSON.parse(await readFile(phase5BreachPath, "utf8"))
    : { by_tier: [] };
  const phase6Rows = await exists(phase6PatchPath)
    ? parseCsv(await readFile(phase6PatchPath, "utf8"))
    : [];
  const { inventory, missing } = await buildInventory(policies);
  if (!breachThresholdExists) {
    missing.push({
      policy_id: "phase5_selection_pressure",
      policyLabel: "Phase 5 breach threshold",
      artifact: "phase5_breach_threshold",
      expected_path: relative(phase5BreachPath),
      status: "missing",
      reason: "threshold_report_not_found",
    });
  }

  const classRows = policies.map((policy) => {
    const verdict = classify(policy);
    return {
      policy_id: policy.policy_id,
      policyLabel: policy.policyLabel,
      family: policy.family,
      tier: policy.tier,
      lambda: policy.lambda,
      signature_shape: policy.signature_shape,
      curriculum_order: policy.curriculum_order,
      class: verdict.className,
      tags: verdict.tags.join(";"),
      success_rate: policy.success_rate,
      mean_terminal_alignment: policy.mean_terminal_alignment,
      old_basin_pref: policy.old_basin_pref,
      mean_probed_success_rate: policy.mean_probed_success_rate,
      max_probe_false_basin_captures: policy.max_probe_false_basin_captures,
      relative_probe_degradation: verdict.degradation === null ? "" : round(verdict.degradation),
      phase6_annotation: phase6Annotation(policy),
    };
  });

  const aggregateRows = aggregate(classRows);
  const deltas = deltaRows(classRows);
  const breachRows = thresholdRows(breachThreshold);
  const candidateRows = classRows.filter((row) => row.class === "hold" || row.class === "fragile");
  const protectedRows = classRows.filter((row) => row.class === "hold");
  const collapsedRows = classRows.filter((row) => row.class === "collapse");
  const fragileRows = classRows.filter((row) => row.class === "fragile");
  const ambiguousRows = classRows.filter((row) => row.class === "ambiguous");

  const summary = {
    phase: "phase7-operating-envelope",
    policy_count: policies.length,
    class_counts: Object.fromEntries(
      ["hold", "collapse", "fragile", "incompetent", "ambiguous"].map((className) => [
        className,
        classRows.filter((row) => row.class === className).length,
      ]),
    ),
    missing_count: missing.length,
    breach_thresholds: breachRows,
    phase6_annotation_count: classRows.filter((row) => row.phase6_annotation).length,
    phase6_net7_best_layer: "net.7",
    phase6_patch_rows: phase6Rows.length,
  };

  const manifest = {
    phase: "phase7-operating-envelope",
    source: {
      phase5_summary: relative(phase5SummaryPath),
      phase5_breach_threshold: relative(phase5BreachPath),
      phase6_patch_aggregate: relative(phase6PatchPath),
    },
    outputs: {
      policies_inventory: "policies-inventory.csv",
      missing_cells: "missing-cells.csv",
      trial_outcomes: "trial-outcomes.csv",
      cell_class_map: "cell-class-map.csv",
      envelope_map: "envelope-map.csv",
      aggregate_envelope: "aggregate-envelope.csv",
      breach_threshold: "reports/breach-threshold.csv",
      cell_delta_map: "cell-delta-map.csv",
      candidate_envelope: "candidate-envelope.csv",
    },
    classification_rules: {
      hold: "old_basin_pref < 1.0 and mean_terminal_alignment >= 0.90 without severe fragility tag",
      collapse: "old_basin_pref >= 1.0",
      fragile: "field-attached nominal cell with severe probe degradation or false-basin probe capture",
      incompetent: "mean_terminal_alignment < 0.70 and success_rate < 0.10 without fixed-attractor collapse",
      ambiguous: "missing metrics or below hold alignment without collapse",
    },
    summary,
  };

  await writeFile(path.join(outRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(path.join(outRoot, "policies-inventory.csv"), toCsv(inventory, COLUMNS_INVENTORY));
  await writeFile(path.join(outRoot, "missing-cells.csv"), toCsv(missing, COLUMNS_MISSING));
  await writeFile(path.join(outRoot, "trial-outcomes.csv"), toCsv(classRows, COLUMNS_CLASS));
  await writeFile(path.join(outRoot, "cell-class-map.csv"), toCsv(classRows, COLUMNS_CLASS));
  await writeFile(path.join(outRoot, "envelope-map.csv"), toCsv(classRows, COLUMNS_CLASS));
  await writeFile(path.join(outRoot, "aggregate-envelope.csv"), toCsv(aggregateRows, COLUMNS_AGG));
  await writeFile(path.join(outRoot, "cell-delta-map.csv"), toCsv(deltas, COLUMNS_DELTA));
  await writeFile(path.join(outRoot, "candidate-envelope.csv"), toCsv(candidateRows, COLUMNS_CLASS));
  await writeFile(path.join(outRoot, "best-by-cell.csv"), toCsv(protectedRows, COLUMNS_CLASS));
  await writeFile(path.join(outRoot, "phase6-mechanistic-annotations.csv"), toCsv(
    classRows.filter((row) => row.phase6_annotation),
    COLUMNS_CLASS,
  ));
  await writeFile(path.join(outRoot, "reports/summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  await writeFile(path.join(outRoot, "reports/breach-threshold.csv"), toCsv(breachRows, COLUMNS_BREACH));
  await writeFile(path.join(outRoot, "reports/protected-pocket.csv"), toCsv(protectedRows, COLUMNS_CLASS));
  await writeFile(path.join(outRoot, "reports/collapsed-pocket.csv"), toCsv(collapsedRows, COLUMNS_CLASS));
  await writeFile(path.join(outRoot, "reports/fragile-pocket.csv"), toCsv(fragileRows, COLUMNS_CLASS));
  await writeFile(path.join(outRoot, "reports/ambiguous-cells.csv"), toCsv(ambiguousRows, COLUMNS_CLASS));

  console.log(`mesa-phase7-envelope: ${policies.length} policies -> ${relative(outRoot)}`);
  console.log(`classes: ${JSON.stringify(summary.class_counts)}`);
  console.log(`missing cells: ${missing.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
