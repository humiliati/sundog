import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const DEFAULT_SOURCE_URL =
  "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1";
const DEFAULT_OUT = "results/riemann/probe01-isotropy-zero-pairs";
const TWO_PI = 2 * Math.PI;

function parseArgs(argv) {
  const args = {
    n: 5000,
    out: DEFAULT_OUT,
    sourceUrl: DEFAULT_SOURCE_URL,
    forceDownload: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--n") {
      args.n = Number.parseInt(argv[++i], 10);
    } else if (arg === "--out") {
      args.out = argv[++i];
    } else if (arg === "--source-url") {
      args.sourceUrl = argv[++i];
    } else if (arg === "--force-download") {
      args.forceDownload = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!Number.isInteger(args.n) || args.n < 2) {
    throw new Error("--n must be an integer >= 2");
  }
  return args;
}

function sha256Buffer(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function sha256File(filePath) {
  return sha256Buffer(fs.readFileSync(filePath));
}

async function downloadSource(url, filePath, forceDownload) {
  if (!forceDownload && fs.existsSync(filePath)) {
    return fs.readFileSync(filePath);
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  fs.writeFileSync(filePath, buffer);
  return buffer;
}

function parseZeros(buffer, n) {
  const text = buffer.toString("utf8");
  const zeros = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const value = Number.parseFloat(trimmed);
    if (Number.isFinite(value)) zeros.push(value);
    if (zeros.length === n) break;
  }
  if (zeros.length < n) {
    throw new Error(`Source supplied ${zeros.length} zeros; need ${n}`);
  }
  return zeros;
}

function gitValue(args) {
  try {
    return execFileSync("git", args, { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function writeCsv(filePath, headers, rows) {
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  }
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function rms(values) {
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
}

function quantile(sortedValues, q) {
  if (sortedValues.length === 0) return null;
  const index = (sortedValues.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sortedValues[lower];
  const weight = index - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

function localZeroDensity(t) {
  return Math.log(t / TWO_PI) / TWO_PI;
}

function buildRows(zeros) {
  const maxHeight = zeros.at(-1);
  const gaps = [];
  for (let i = 0; i < zeros.length - 1; i += 1) {
    gaps.push(zeros[i + 1] - zeros[i]);
  }
  const meanGap = mean(gaps);
  const pairRows = [];
  const spacingRows = [];
  const isotropyRows = [];
  const reflectionResiduals = [];
  const unfoldedSpacings = [];
  const spacingSignComponents = [];
  const heightSignComponents = [];

  for (let i = 0; i < zeros.length - 1; i += 1) {
    const left = zeros[i];
    const right = zeros[i + 1];
    const center = (left + right) / 2;
    const reflectedLeft = -right;
    const reflectedRight = -left;
    const reflectedCenter = (reflectedLeft + reflectedRight) / 2;
    const gap = right - left;
    const reflectedGap = reflectedRight - reflectedLeft;
    const density = localZeroDensity(Math.abs(center));
    const reflectedDensity = localZeroDensity(Math.abs(reflectedCenter));
    const unfoldedSpacing = gap * density;
    const reflectedUnfoldedSpacing = reflectedGap * reflectedDensity;
    const reflectionResidual = Math.abs(unfoldedSpacing - reflectedUnfoldedSpacing);
    const signedHeightNorm = center / maxHeight;
    const reflectedSignedHeightNorm = reflectedCenter / maxHeight;
    const heightEven = (signedHeightNorm + reflectedSignedHeightNorm) / 2;
    const heightOdd = (signedHeightNorm - reflectedSignedHeightNorm) / 2;
    const spacingEven = (unfoldedSpacing + reflectedUnfoldedSpacing) / 2;
    const spacingOdd = (unfoldedSpacing - reflectedUnfoldedSpacing) / 2;
    const rawGapEven = (gap + reflectedGap) / (2 * meanGap);
    const rawGapOdd = (gap - reflectedGap) / (2 * meanGap);

    reflectionResiduals.push(reflectionResidual);
    unfoldedSpacings.push(unfoldedSpacing);
    spacingSignComponents.push(Math.abs(spacingOdd) + Math.abs(rawGapOdd));
    heightSignComponents.push(Math.abs(heightOdd));

    const pairId = i + 1;
    spacingRows.push({
      pair_id: pairId,
      zero_index_left: pairId,
      zero_index_right: pairId + 1,
      gamma_left: left.toFixed(12),
      gamma_right: right.toFixed(12),
      center: center.toFixed(12),
      gap: gap.toFixed(12),
      local_density: density.toFixed(15),
      unfolded_spacing: unfoldedSpacing.toFixed(15),
      reflected_center: reflectedCenter.toFixed(12),
      reflected_gap: reflectedGap.toFixed(12),
      reflected_unfolded_spacing: reflectedUnfoldedSpacing.toFixed(15),
      reflection_residual: reflectionResidual.toExponential(12),
    });
    pairRows.push({
      pair_id: pairId,
      height_even: heightEven.toExponential(12),
      height_odd: heightOdd.toExponential(12),
      density_even: density.toExponential(12),
      density_odd: ((density - reflectedDensity) / 2).toExponential(12),
      raw_gap_even_mean_scaled: rawGapEven.toExponential(12),
      raw_gap_odd_mean_scaled: rawGapOdd.toExponential(12),
      unfolded_spacing_even: spacingEven.toExponential(12),
      unfolded_spacing_odd: spacingOdd.toExponential(12),
    });
    isotropyRows.push({
      pair_id: pairId,
      bridge_path: "path_i_z2_descent",
      action: "functional_equation_reflection_s_to_1_minus_s",
      reflection_residual: reflectionResidual.toExponential(12),
      spacing_sign_component_abs: (Math.abs(spacingOdd) + Math.abs(rawGapOdd)).toExponential(12),
      height_sign_component_abs: Math.abs(heightOdd).toExponential(12),
      sector_read: "spacing_features_even__signed_height_odd",
      structural_zero_reachable: "false",
      flags: "path_i_no_v03h_structural_zero;odd_sector_is_signed_height_carrier",
    });
  }

  return {
    spacingRows,
    pairRows,
    isotropyRows,
    summary: {
      pairCount: zeros.length - 1,
      maxHeight,
      meanGap,
      meanUnfoldedSpacing: mean(unfoldedSpacings),
      rmsUnfoldedSpacing: rms(unfoldedSpacings),
      minUnfoldedSpacing: Math.min(...unfoldedSpacings),
      maxUnfoldedSpacing: Math.max(...unfoldedSpacings),
      q25UnfoldedSpacing: quantile([...unfoldedSpacings].sort((a, b) => a - b), 0.25),
      medianUnfoldedSpacing: quantile([...unfoldedSpacings].sort((a, b) => a - b), 0.5),
      q75UnfoldedSpacing: quantile([...unfoldedSpacings].sort((a, b) => a - b), 0.75),
      maxReflectionResidual: Math.max(...reflectionResiduals),
      rmsSpacingSignComponent: rms(spacingSignComponents),
      maxSpacingSignComponent: Math.max(...spacingSignComponents),
      rmsHeightSignComponent: rms(heightSignComponents),
      maxHeightSignComponent: Math.max(...heightSignComponents),
    },
  };
}

function validateZeros(zeros, maxHeightCeiling) {
  const rows = [];
  let orderOk = true;
  let finiteOk = true;
  let positiveOk = true;
  for (let i = 0; i < zeros.length; i += 1) {
    if (!Number.isFinite(zeros[i])) finiteOk = false;
    if (!(zeros[i] > 0)) positiveOk = false;
    if (i > 0 && !(zeros[i] > zeros[i - 1])) orderOk = false;
    rows.push({
      zero_index: i + 1,
      gamma: zeros[i].toFixed(12),
      source_precision_abs: "3e-9",
      validation_status:
        Number.isFinite(zeros[i]) && zeros[i] > 0 && (i === 0 || zeros[i] > zeros[i - 1])
          ? "ok"
          : "invalid",
    });
  }
  const maxHeightOk = zeros.at(-1) < maxHeightCeiling;
  return {
    rows,
    checks: {
      finiteOk,
      positiveOk,
      orderOk,
      maxHeightOk,
      maxHeightCeiling,
    },
  };
}

function writeReadme(filePath, manifest, summary, disposition) {
  const text = `# Probe 01 Path (i) Run

Status: ${disposition.verdict}

This is a Probe 01 v1 run under Path (i), the Z2 descent along the
functional-equation reflection \`s -> 1 - s\`. It is a parity-decomposition
receipt, not a v0.3h structural-zero receipt and not evidence for or against RH.

## Frozen Domain

- Source: ${manifest.source.url}
- Source SHA256: \`${manifest.source.sha256}\`
- N: ${manifest.domain.n}
- Pair rule: ${manifest.domain.pairConstructionRule}
- Unfolding: ${manifest.domain.unfoldingFormula}
- Bridge: ${manifest.representationBridge.path}
- Max height: ${summary.maxHeight}

## Summary

- Pairs analyzed: ${summary.pairCount}
- Mean unfolded spacing: ${summary.meanUnfoldedSpacing}
- Median unfolded spacing: ${summary.medianUnfoldedSpacing}
- Max reflection residual: ${summary.maxReflectionResidual}
- RMS spacing sign component: ${summary.rmsSpacingSignComponent}
- RMS signed-height component: ${summary.rmsHeightSignComponent}

## Disposition

${disposition.summary}

Branch: ${disposition.branch}
`;
  fs.writeFileSync(filePath, text);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(args.out);
  const sourceDir = path.join(outDir, "source");
  fs.mkdirSync(sourceDir, { recursive: true });
  const sourcePath = path.join(sourceDir, "zeros1.txt");
  const sourceBuffer = await downloadSource(args.sourceUrl, sourcePath, args.forceDownload);
  const sourceHash = sha256Buffer(sourceBuffer);
  const zeros = parseZeros(sourceBuffer, args.n);
  const maxHeightCeiling = 10000;
  const validation = validateZeros(zeros, maxHeightCeiling);
  if (
    !validation.checks.finiteOk ||
    !validation.checks.positiveOk ||
    !validation.checks.orderOk ||
    !validation.checks.maxHeightOk
  ) {
    throw new Error(`Zero validation failed: ${JSON.stringify(validation.checks)}`);
  }

  const { spacingRows, pairRows, isotropyRows, summary } = buildRows(zeros);
  const reflectionResidualThreshold = 1e-12;
  const spacingSignComponentThreshold = 1e-12;
  const disposition =
    summary.maxReflectionResidual <= reflectionResidualThreshold &&
    summary.maxSpacingSignComponent <= spacingSignComponentThreshold
      ? {
          verdict: "bounded Front A parity-decomposition receipt under Path (i)",
          branch: "A - clean bounded catalog under Z2 descent",
          summary:
            "Spacing-derived features are reflection-even inside the registered window. The nonzero sign sector is the signed-height carrier only. Branch B structural-zero language is not reachable under Path (i).",
        }
      : {
          verdict: "falsified in registered cell",
          branch: "D - residual breach",
          summary:
            "A registered reflection residual or spacing sign-sector threshold was crossed.",
        };

  const manifest = {
    probe: "riemann_probe01_isotropy_zero_pairs",
    runId: "probe01-pathi-odlyzko-zeros1-n5000",
    createdAtUtc: new Date().toISOString(),
    command: process.argv.join(" "),
    git: {
      commit: gitValue(["rev-parse", "HEAD"]),
      statusShort: gitValue(["status", "--short"]),
    },
    code: {
      scriptPath: path.relative(process.cwd(), fileURLToPath(import.meta.url)).replaceAll("\\", "/"),
      scriptSha256: sha256File(fileURLToPath(import.meta.url)),
    },
    source: {
      name: "Odlyzko zeta zero table zeros1",
      url: args.sourceUrl,
      localPath: path.relative(process.cwd(), sourcePath).replaceAll("\\", "/"),
      sha256: sourceHash,
      declaredAccuracyAbs: "3e-9",
      declaredContent: "first 100000 positive ordinates of non-trivial Riemann zeta zeros",
    },
    domain: {
      n: args.n,
      maxHeightCeiling,
      observedMaxHeight: summary.maxHeight,
      statistic: "nearest_neighbor_unfolded_spacings",
      pairConstructionRule: "consecutive positive-zero pairs (gamma_i, gamma_{i+1}); reflected pair (-gamma_{i+1}, -gamma_i)",
      unfoldingFormula: "u_i = (gamma_{i+1} - gamma_i) * log(center_i / (2*pi)) / (2*pi)",
      pairWindowDefinition: "nearest-neighbor consecutive pairs only",
      binning: "none for primary receipt; raw per-pair rows emitted",
      randomSeeds: [],
    },
    representationBridge: {
      path: "Path (i) - Z2 descent under functional-equation reflection s -> 1-s",
      structuralZeroReachable: false,
      pathIiS3TripleInvoked: false,
      pathIiiQuarantineInvoked: false,
    },
    thresholds: {
      reflectionResidualAbs: reflectionResidualThreshold,
      spacingSignComponentAbs: spacingSignComponentThreshold,
      quarantinePredicates: [
        "source parse/order/positivity/max-height validation fails",
        "Path (i) bridge degenerates before classification",
        "Path (ii) S3 bridge is invoked inside Probe 01 v1",
      ],
    },
    validation: validation.checks,
    summary,
    disposition,
  };

  writeCsv(path.join(outDir, "zeros.csv"), [
    "zero_index",
    "gamma",
    "source_precision_abs",
    "validation_status",
  ], validation.rows);
  writeCsv(path.join(outDir, "unfolded_spacings.csv"), [
    "pair_id",
    "zero_index_left",
    "zero_index_right",
    "gamma_left",
    "gamma_right",
    "center",
    "gap",
    "local_density",
    "unfolded_spacing",
    "reflected_center",
    "reflected_gap",
    "reflected_unfolded_spacing",
    "reflection_residual",
  ], spacingRows);
  writeCsv(path.join(outDir, "pair_features.csv"), [
    "pair_id",
    "height_even",
    "height_odd",
    "density_even",
    "density_odd",
    "raw_gap_even_mean_scaled",
    "raw_gap_odd_mean_scaled",
    "unfolded_spacing_even",
    "unfolded_spacing_odd",
  ], pairRows);
  writeCsv(path.join(outDir, "isotropy_records.csv"), [
    "pair_id",
    "bridge_path",
    "action",
    "reflection_residual",
    "spacing_sign_component_abs",
    "height_sign_component_abs",
    "sector_read",
    "structural_zero_reachable",
    "flags",
  ], isotropyRows);
  writeCsv(path.join(outDir, "structural_zero_summary.csv"), [
    "metric",
    "value",
  ], [
    { metric: "bridge", value: manifest.representationBridge.path },
    { metric: "structural_zero_reachable", value: "false" },
    { metric: "pairs_analyzed", value: summary.pairCount },
    { metric: "max_reflection_residual", value: summary.maxReflectionResidual },
    { metric: "rms_spacing_sign_component", value: summary.rmsSpacingSignComponent },
    { metric: "max_spacing_sign_component", value: summary.maxSpacingSignComponent },
    { metric: "rms_height_sign_component", value: summary.rmsHeightSignComponent },
    { metric: "max_height_sign_component", value: summary.maxHeightSignComponent },
    { metric: "disposition_branch", value: disposition.branch },
  ]);
  writeCsv(path.join(outDir, "quarantine.csv"), [
    "quarantine_id",
    "reason",
    "disposition",
  ], []);
  writeReadme(path.join(outDir, "README.md"), manifest, summary, disposition);

  const artifactFiles = [
    "manifest.json",
    "zeros.csv",
    "unfolded_spacings.csv",
    "pair_features.csv",
    "isotropy_records.csv",
    "structural_zero_summary.csv",
    "quarantine.csv",
    "README.md",
    "source/zeros1.txt",
  ];
  manifest.artifacts = artifactFiles
    .filter((fileName) => fileName !== "manifest.json")
    .map((fileName) => {
      const artifactPath = path.join(outDir, fileName);
      return {
        path: path.relative(process.cwd(), artifactPath).replaceAll("\\", "/"),
        sha256: sha256File(artifactPath),
      };
    });
  fs.writeFileSync(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

  console.log(JSON.stringify({
    out: path.relative(process.cwd(), outDir).replaceAll("\\", "/"),
    verdict: disposition.verdict,
    pairsAnalyzed: summary.pairCount,
    maxHeight: summary.maxHeight,
    maxReflectionResidual: summary.maxReflectionResidual,
    sourceSha256: sourceHash,
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
