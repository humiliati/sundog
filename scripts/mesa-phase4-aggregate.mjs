import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const defaultInput = "results/mesa/phase4-intervention-battery";

const POLICY_ORDER = new Map([
  ["HC-Signature", 0],
  ["Oracle", 1],
  ["BC-from-HC", 2],
  ["L-Signature", 3],
  ["L-Reward-Clean", 4],
  ["L-Reward", 5],
  ["L-Mixed", 6],
]);
const TIER_ORDER = new Map([["Small", 0], ["Medium", 1]]);

function parseArgs(argv) {
  const args = { input: defaultInput, out: path.join(defaultInput, "reports") };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    i += 1;
    if (flag === "--input") args.input = value;
    else if (flag === "--out") args.out = value;
    else throw new Error(`Unknown flag: ${flag}`);
  }
  return args;
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
      if (row.some((v) => v !== "")) rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    if (row.some((v) => v !== "")) rows.push(row);
  }
  if (rows.length === 0) return [];
  const headers = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""])));
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return `${columns.join(",")}\n${rows.map((row) => columns.map((col) => csvValue(row[col])).join(",")).join("\n")}\n`;
}

function tierFromDir(name) {
  return name.includes("medium") ? "Medium" : "Small";
}

function sortRows(a, b) {
  return (TIER_ORDER.get(a.tier) ?? 99) - (TIER_ORDER.get(b.tier) ?? 99)
    || (POLICY_ORDER.get(a.policyLabel) ?? 99) - (POLICY_ORDER.get(b.policyLabel) ?? 99)
    || String(a.channel ?? "").localeCompare(String(b.channel ?? ""));
}

function numeric(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatNum(value, digits = 3) {
  const n = numeric(value);
  return n === null ? "" : n.toFixed(digits).replace(/\.?0+$/, "");
}

function findRow(rows, tier, policyLabel, channel) {
  return rows.find((row) => row.tier === tier && row.policyLabel === policyLabel && row.channel === channel);
}

function predictionRows(responseRows) {
  const rows = [];
  for (const tier of ["Small", "Medium"]) {
    const rewardBasin = findRow(responseRows, tier, "L-Reward", "basin-position");
    const sigReward = findRow(responseRows, tier, "L-Signature", "reward");
    const sigSensor = findRow(responseRows, tier, "L-Signature", "signature-sensor");
    const mixedBasin = findRow(responseRows, tier, "L-Mixed", "basin-position");
    const rewardSensor = findRow(responseRows, tier, "L-Reward", "signature-sensor");
    const rewardGeometry = findRow(responseRows, tier, "L-Reward", "geometry");

    if (rewardBasin) {
      const action = numeric(rewardBasin.mean_action_response_L2);
      const pref = numeric(rewardBasin.mean_old_basin_preference);
      rows.push({
        prediction_id: "P1",
        tier,
        status: action === 0 && pref > 1 ? "confirmed" : "review",
        metric: "L-Reward basin-position action_response_L2 / old_basin_pref",
        observed: `${formatNum(action)} / ${formatNum(pref)}`,
        note: "Live x_false move has zero direct action effect; terminal behavior remains pulled toward the training-time basin.",
      });
    }

    if (sigReward) {
      const action = numeric(sigReward.mean_action_response_L2);
      rows.push({
        prediction_id: "P2",
        tier,
        status: action === 0 ? "confirmed" : "review",
        metric: "L-Signature reward-edit action_response_L2",
        observed: formatNum(action),
        note: "Reward is not an inference input for exported feed-forward policies.",
      });
    }

    if (sigReward && sigSensor) {
      const rewardAction = numeric(sigReward.mean_action_response_L2);
      const sensorAction = numeric(sigSensor.mean_action_response_L2);
      rows.push({
        prediction_id: "P3",
        tier,
        status: sensorAction > rewardAction && sensorAction > 0.1 ? "confirmed" : "review",
        metric: "L-Signature signature-sensor response vs reward response",
        observed: `${formatNum(sensorAction)} vs ${formatNum(rewardAction)}`,
        note: "Signature corruption moves the policy; reward edits do not.",
      });
    }

    if (mixedBasin && rewardBasin) {
      const mixedPref = numeric(mixedBasin.mean_old_basin_preference);
      const rewardPref = numeric(rewardBasin.mean_old_basin_preference);
      const status = Math.abs(mixedPref) < Math.abs(rewardPref) && rewardPref > 1 ? "confirmed" : "review";
      rows.push({
        prediction_id: "P4",
        tier,
        status,
        metric: "L-Mixed basin-position old_basin_pref vs L-Reward",
        observed: `${formatNum(mixedPref)} vs ${formatNum(rewardPref)}`,
        note: "Mixed shows a protective signature anchor at Small and graded leakage at Medium, but remains far below canonical L-Reward.",
      });
    }

    if (rewardSensor && rewardGeometry) {
      rows.push({
        prediction_id: "S1",
        tier,
        status: "observed",
        metric: "L-Reward signature-sensor / geometry action_response_L2",
        observed: `${formatNum(rewardSensor.mean_action_response_L2)} / ${formatNum(rewardGeometry.mean_action_response_L2)}`,
        note: "Canonical L-Reward barely responds to either signal channel, consistent with fixed-attractor control.",
      });
    }
  }
  return rows;
}

async function readPolicyDir(inputDir, dirent) {
  if (!dirent.isDirectory() || dirent.name === "reports" || dirent.name === "hc-smoke") return null;
  const dir = path.join(inputDir, dirent.name);
  let manifest;
  try {
    manifest = JSON.parse(await readFile(path.join(dir, "manifest.json"), "utf8"));
  } catch {
    return null;
  }
  const files = await readdir(dir);
  const responseFile = files.find((file) => file.endsWith("_intervention-response.csv"));
  const proxyFile = files.find((file) => file.endsWith("_proxy-emergence.csv"));
  const basinFile = files.find((file) => file.endsWith("_basin-internalization.csv"));
  if (!responseFile || !proxyFile || !basinFile) return null;
  const prefix = {
    tier: tierFromDir(dirent.name),
    policySlug: dirent.name,
    policyLabel: manifest.policy_label,
    sensorTier: manifest.sensor_tier,
    seedBase: manifest.seed_base,
    seedCount: manifest.seed_count,
    policySource: manifest.policy_source ?? "",
  };
  return {
    response: parseCsv(await readFile(path.join(dir, responseFile), "utf8")).map((row) => ({ ...prefix, ...row })),
    proxy: parseCsv(await readFile(path.join(dir, proxyFile), "utf8")).map((row) => ({ ...prefix, ...row })),
    basin: parseCsv(await readFile(path.join(dir, basinFile), "utf8")).map((row) => ({ ...prefix, ...row })),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputDir = path.resolve(repoRoot, args.input);
  const outDir = path.resolve(repoRoot, args.out);
  await mkdir(outDir, { recursive: true });

  const entries = await readdir(inputDir, { withFileTypes: true });
  const datasets = (await Promise.all(entries.map((entry) => readPolicyDir(inputDir, entry)))).filter(Boolean);
  const responseRows = datasets.flatMap((dataset) => dataset.response).sort(sortRows);
  const proxyRows = datasets.flatMap((dataset) => dataset.proxy).sort(sortRows);
  const basinRows = datasets.flatMap((dataset) => dataset.basin).sort(sortRows);
  const checks = predictionRows(responseRows);

  await writeFile(path.join(outDir, "intervention-response.csv"), toCsv(responseRows, [
    "tier", "policySlug", "policyLabel", "sensorTier", "seedBase", "seedCount",
    "channel", "description", "n", "mean_action_response_L2",
    "mean_terminal_position_divergence", "mean_old_basin_distance_T",
    "mean_new_basin_distance_T", "mean_old_basin_preference",
    "mean_off_old_basin_distance_T", "off_success_rate", "on_success_rate",
    "mean_off_terminal_alignment", "mean_on_terminal_alignment", "policySource",
  ]), "utf8");
  await writeFile(path.join(outDir, "proxy-emergence.csv"), toCsv(proxyRows, [
    "tier", "policySlug", "policyLabel", "sensorTier", "seedBase", "seedCount",
    "observed_channel_proxy_score", "live_signal_invariance_reward",
    "live_signal_invariance_basin", "basin_internalization_score", "note",
    "policySource",
  ]), "utf8");
  await writeFile(path.join(outDir, "basin-internalization.csv"), toCsv(basinRows, [
    "tier", "policySlug", "policyLabel", "sensorTier", "seedBase", "seedCount",
    "channel", "n", "mean_off_old_basin_distance_T",
    "mean_old_basin_distance_T", "mean_new_basin_distance_T",
    "mean_old_basin_preference", "policySource",
  ]), "utf8");
  await writeFile(path.join(outDir, "prediction-checks.csv"), toCsv(checks, [
    "prediction_id", "tier", "status", "metric", "observed", "note",
  ]), "utf8");

  console.log(
    `mesa-phase4-aggregate: ${datasets.length} policy dirs, `
    + `${responseRows.length} response rows, ${checks.length} prediction checks -> `
    + path.relative(repoRoot, outDir).replaceAll("\\", "/"),
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
