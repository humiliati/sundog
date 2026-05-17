import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();

const sourcePaths = {
  summary: "results/mesa/operating-envelope/reports/summary.json",
  classMap: "results/mesa/operating-envelope/cell-class-map.csv",
  traceability: "results/mesa/operating-envelope/cell-traceability-labels.csv",
  patchAggregate: "results/mesa/phase6-probes/axis-b-full-64seed/axis-b-patch-smoke-aggregate.csv",
  pcaVariance: "results/mesa/phase6-v2-direction/axis-h-pca/axis-h-pca-variance.csv",
};

const outputPath = "public/data/mesa-public-charts.json";
const classOrder = ["hold", "collapse", "fragile", "incompetent", "ambiguous"];
const kSweepValues = [1, 3, 5, 10, 32, 64];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (quoted && next === '"') {
        cell += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  row.push(cell);
  if (row.some((value) => value.length > 0)) rows.push(row);
  if (rows.length === 0) return [];

  const headers = rows[0];
  return rows.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
}

async function readText(relativePath) {
  return readFile(join(root, relativePath), "utf8");
}

function asNumber(value) {
  if (value === undefined || value === null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function cleanPolicy(row, traceabilityByPolicy) {
  return {
    policyId: row.policy_id,
    label: row.policyLabel,
    family: row.family,
    tier: row.tier,
    lambda: asNumber(row.lambda),
    signatureShape: row.signature_shape || null,
    curriculumOrder: row.curriculum_order || null,
    klass: row.class,
    tags: row.tags ? row.tags.split(";").filter(Boolean) : [],
    successRate: asNumber(row.success_rate),
    meanTerminalAlignment: asNumber(row.mean_terminal_alignment),
    oldBasinPref: asNumber(row.old_basin_pref),
    meanProbedSuccessRate: asNumber(row.mean_probed_success_rate),
    maxProbeFalseBasinCaptures: asNumber(row.max_probe_false_basin_captures),
    relativeProbeDegradation: asNumber(row.relative_probe_degradation),
    phase6Annotation: row.phase6_annotation || null,
    traceability: traceabilityByPolicy.get(row.policy_id) ?? null,
  };
}

function buildCliffSeries(policies, tier) {
  return policies
    .filter((policy) => policy.family === "L-Mixed" && policy.tier === tier && policy.lambda !== null)
    .sort((a, b) => a.lambda - b.lambda)
    .map((policy) => ({
      lambda: policy.lambda,
      oldBasinPref: policy.oldBasinPref,
      klass: policy.klass,
      policyId: policy.policyId,
      label: policy.label,
      marker: policy.phase6Annotation?.includes("net7") ? "net7" : null,
    }));
}

function buildPatchRows(rows) {
  return rows
    .filter((row) => row.condition === "clean")
    .map((row) => ({
      layer: row.layer,
      direction: row.direction,
      meanPatchSuccess: asNumber(row.mean_patch_success),
      medianPatchSuccess: asNumber(row.median_patch_success),
      patchSuccessRatioOfMeans: asNumber(row.patch_success_ratio_of_means),
      n: asNumber(row.n),
    }));
}

async function buildKSweepRows(varianceRows) {
  const varianceByK = new Map(
    varianceRows.map((row) => [asNumber(row.rank), asNumber(row.cumulative_variance_fraction)]),
  );
  const rows = [];

  for (const k of kSweepValues) {
    const aggregatePath =
      `results/mesa/phase6-v2-direction/axis-h-pca-k${k}/axis-h-pca-patch-aggregate.csv`;
    const aggregate = parseCsv(await readText(aggregatePath));
    const byDirection = Object.fromEntries(
      aggregate.map((row) => [
        row.direction,
        {
          meanPatchSuccess: asNumber(row.mean_patch_success),
          medianPatchSuccess: asNumber(row.median_patch_success),
          patchSuccessRatioOfMeans: asNumber(row.patch_success_ratio_of_means),
          n: asNumber(row.n),
        },
      ]),
    );

    rows.push({
      k,
      varianceCaptured: varianceByK.get(k) ?? null,
      protectedToCollapsed: byDirection.protected_to_collapsed ?? null,
      collapsedToProtected: byDirection.collapsed_to_protected ?? null,
    });
  }

  return rows;
}

async function main() {
  const summary = JSON.parse(await readText(sourcePaths.summary));
  const classRows = parseCsv(await readText(sourcePaths.classMap));
  const traceabilityRows = parseCsv(await readText(sourcePaths.traceability));
  const patchRows = parseCsv(await readText(sourcePaths.patchAggregate));
  const varianceRows = parseCsv(await readText(sourcePaths.pcaVariance));

  const traceabilityByPolicy = new Map(
    traceabilityRows.map((row) => [
      row.policy_id,
      {
        label: row.traceability_label,
        evidenceAnchor: row.evidence_anchor,
        note: row.note,
      },
    ]),
  );
  const policies = classRows.map((row) => cleanPolicy(row, traceabilityByPolicy));

  const data = {
    schemaVersion: 1,
    purpose:
      "Public chart source for mesa.html and exportable Mesa evidence panels. This file is generated from audited Mesa result artifacts; do not hand-edit.",
    sourcePaths: {
      ...sourcePaths,
      kSweepAggregates: kSweepValues.map(
        (k) => `results/mesa/phase6-v2-direction/axis-h-pca-k${k}/axis-h-pca-patch-aggregate.csv`,
      ),
    },
    claimBoundary:
      "Small/Medium in-vitro operating-envelope map only; not universal mesa immunity, not foundation-model behavior, not deployed-system robustness.",
    summary: {
      phase: summary.phase,
      policyCount: summary.policy_count,
      missingCount: summary.missing_count,
      classCounts: Object.fromEntries(classOrder.map((klass) => [klass, summary.class_counts[klass] ?? 0])),
      breachThresholds: summary.breach_thresholds,
      phase6Net7BestLayer: summary.phase6_net7_best_layer,
    },
    classBalance: classOrder.map((klass) => ({
      klass,
      count: summary.class_counts[klass] ?? 0,
    })),
    policies,
    cliff: {
      medium: buildCliffSeries(policies, "Medium"),
      small: buildCliffSeries(policies, "Small"),
    },
    patchByLayer: buildPatchRows(patchRows),
    kSweep: await buildKSweepRows(varianceRows),
    chartQueue: [
      {
        id: "mesa-evidence-panel",
        target: "homepage Evidence and Metrics panel",
        source: "summary + classBalance + cliff.medium",
        output: "public/media/mesa-evidence-panel.svg",
        status: "generated-by-prebuild",
      },
      {
        id: "mesa-cliff-mini",
        target: "post-rail Working Systems evidence panel",
        source: "cliff.medium + summary.breachThresholds",
        output: "public/media/mesa-cliff-mini.svg",
        status: "generated-by-prebuild",
      },
      {
        id: "mesa-class-balance-strip",
        target: "post-rail Working Systems evidence panel",
        source: "classBalance",
        output: "public/media/mesa-class-balance-strip.svg",
        status: "generated-by-prebuild",
      },
      {
        id: "mesa-ksweep-fingerprint",
        target: "standalone mesa.html chart/image export",
        source: "kSweep",
        output: "public/media/mesa-ksweep-fingerprint.svg",
        status: "generated-by-prebuild",
      },
    ],
  };

  const absoluteOutput = join(root, outputPath);
  await mkdir(dirname(absoluteOutput), { recursive: true });
  await writeFile(absoluteOutput, `${JSON.stringify(data, null, 2)}\n`, "utf8");
  console.log(`mesa public chart data built: ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
