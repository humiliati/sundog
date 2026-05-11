import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs as parseHarnessArgs, runSuite } from "./mines-phase4-baselines.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    source: "results/mines/phase7-smoke/replay-index.json",
    out: "results/mines/phase7-smoke/replay-verification.csv",
    manifest: "results/mines/phase7-smoke/replay-verification.json",
    phase: "phase7-replay-verify",
    limit: null,
    presets: null,
    modes: null,
    cells: null,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (!flag.startsWith("--")) continue;
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${flag} requires a value`);
    }
    i += 1;

    if (flag === "--source") args.source = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--manifest") args.manifest = value;
    else if (flag === "--phase") args.phase = value;
    else if (flag === "--limit") args.limit = value === "all" ? null : Number.parseInt(value, 10);
    else if (flag === "--presets") args.presets = parseList(value);
    else if (flag === "--modes") args.modes = parseList(value);
    else if (flag === "--cells") args.cells = parseList(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (args.limit !== null && (!Number.isInteger(args.limit) || args.limit < 1)) {
    throw new Error("--limit must be a positive integer or all");
  }

  return args;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll("\"", "\"\"")}"`;
  return text;
}

function toCsv(rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function selectRows(rows, args) {
  let selected = rows;
  if (args.presets) selected = selected.filter((row) => args.presets.includes(row.preset));
  if (args.modes) selected = selected.filter((row) => args.modes.includes(row.mode));
  if (args.cells) selected = selected.filter((row) => args.cells.includes(row.sensorCell));
  if (args.limit !== null) selected = selected.slice(0, args.limit);
  return selected;
}

function verifyRow(row, args) {
  const harnessArgs = parseHarnessArgs([
    "--phase",
    args.phase,
    "--out",
    "results/mines/phase7-replay-verify",
    "--replay-url",
    row.browserReplayUrl,
  ]);
  const { trialRows } = runSuite(harnessArgs);
  const replay = trialRows[0];
  const pass = replay.actionTraceHash === row.actionTraceHash
    && replay.terminal === row.terminal
    && replay.turns === row.turns;

  return {
    trialId: row.trialId,
    preset: row.preset,
    sensorCell: row.sensorCell,
    mode: row.mode,
    seed: row.seed,
    expectedTerminal: row.terminal,
    replayTerminal: replay.terminal,
    expectedTurns: row.turns,
    replayTurns: replay.turns,
    expectedActionTraceHash: row.actionTraceHash,
    replayActionTraceHash: replay.actionTraceHash,
    pass,
    browserReplayUrl: row.browserReplayUrl,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sourcePath = path.resolve(repoRoot, args.source);
  const outPath = path.resolve(repoRoot, args.out);
  const manifestPath = path.resolve(repoRoot, args.manifest);
  const sourceRows = JSON.parse(await readFile(sourcePath, "utf8"));
  const selectedRows = selectRows(sourceRows, args);
  const verificationRows = selectedRows.map((row) => verifyRow(row, args));
  const failures = verificationRows.filter((row) => !row.pass);
  const manifest = {
    schema: "sundog.mines.phase7-replay-verification.v1",
    generatedAt: new Date().toISOString(),
    source: path.relative(repoRoot, sourcePath),
    out: path.relative(repoRoot, outPath),
    phase: args.phase,
    checked: verificationRows.length,
    passed: verificationRows.length - failures.length,
    failed: failures.length,
    presets: args.presets,
    modes: args.modes,
    cells: args.cells,
    limit: args.limit,
  };

  await mkdir(path.dirname(outPath), { recursive: true });
  await writeFile(outPath, toCsv(verificationRows, [
    "trialId",
    "preset",
    "sensorCell",
    "mode",
    "seed",
    "expectedTerminal",
    "replayTerminal",
    "expectedTurns",
    "replayTurns",
    "expectedActionTraceHash",
    "replayActionTraceHash",
    "pass",
    "browserReplayUrl",
  ]));
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  console.log(`Mines replay verification: ${manifest.passed}/${manifest.checked} passed`);
  console.log(`Wrote ${path.relative(repoRoot, outPath)}`);
  if (failures.length > 0) {
    console.error(`Replay verification failed for ${failures.length} trial(s)`);
    for (const row of failures.slice(0, 8)) {
      console.error(`${row.trialId}: expected ${row.expectedActionTraceHash}, got ${row.replayActionTraceHash}`);
    }
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
