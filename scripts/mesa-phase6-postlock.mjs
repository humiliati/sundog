// scripts/mesa-phase6-postlock.mjs
//
// Post-lock read-back runner for Phase 6 lambda-control. After
// mesa-phase6-probe-concurrent.mjs has produced all 6 lock policies in
// `results/proof/phase6/training-full/`, this script chains the two
// per-row stages the spec's lock loop runs after training:
//
//   1. mesa-intervention-battery.mjs  — produces the DECISIVE metric
//      (basin-position channel's `mean_old_basin_preference`) used by
//      PHASE6_LAMBDA_CONTROL.md ▸ Read-Back to fire one of 4 branches.
//   2. mesa-probe-slate.mjs           — probe-cell diagnostics referenced
//      by Read-Back ("probe manifest and trial outcomes").
//
// Phase A runs first, the decision table prints as soon as it completes,
// then Phase B runs for completeness. Both phases fan out per-row.
//
// Usage:
//   node scripts/mesa-phase6-postlock.mjs [--fan-out <N>] [--rows <labels>]
//                                          [--out-logs <dir>] [--skip-probe-slate] [--force]
//
// Defaults:
//   --fan-out 4                   (pure JS, no PyTorch BLAS; 4-wide safe on 4-core box)
//   --rows <LOCK_LABELS>          (all 6 spec-canonical lock rows)
//   --out-logs results/proof/phase6/logs-postlock
//
// Resume-safe: each stage skips a row whose `manifest.json` already exists.
// Pass --force to re-run regardless.

import { spawn } from "node:child_process";
import { createWriteStream, mkdirSync, existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  LOCK_LABELS,
  getRow,
  policyPath,
  policyLabel,
  slugifyLabel,
  trackChild,
} from "./lib/mesa-phase6-rows.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// PHASE6_LAMBDA_CONTROL.md ▸ "Source Cliff Being Controlled" — Phase 5 v4
// baselines pinned by the spec. Used as the reference column in the decision
// table so reviewers see how each new reading compares to the registered cliff.
const PHASE5_BASELINE = Object.freeze({
  "0.90": { mean_old_basin_pref: 0.383, read: "protected" },
  "0.95": { mean_old_basin_pref: 0.330, read: "protected" },
  "0.97": { mean_old_basin_pref: 5.510, read: "collapsed" },
  "0.99": { mean_old_basin_pref: 5.159, read: "collapsed" },
  "1.00": { mean_old_basin_pref: 5.560, read: "collapsed" },
});

const BREACH_THRESHOLD = 1.0;     // spec: > 1.0 = collapsed
const AMBIGUOUS_MARGIN = 0.3;     // within ±0.3 of threshold → Branch 4 midpoint

function parseArgs(argv) {
  const args = {
    fanOut: 4,
    rows: LOCK_LABELS.slice(),
    outLogs: "results/proof/phase6/logs-postlock",
    skipProbeSlate: false,
    force: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    if (flag === "--fan-out") {
      const n = Number(value);
      if (!Number.isInteger(n) || n < 1) throw new Error(`--fan-out must be a positive integer (got "${value}")`);
      args.fanOut = n; i += 1;
    } else if (flag === "--rows") {
      args.rows = value.split(",").map((s) => s.trim()).filter(Boolean);
      i += 1;
    } else if (flag === "--out-logs") {
      args.outLogs = value; i += 1;
    } else if (flag === "--skip-probe-slate") {
      args.skipProbeSlate = true;
    } else if (flag === "--force") {
      args.force = true;
    } else if (flag === "--help" || flag === "-h") {
      printHelpAndExit(0);
    } else {
      throw new Error(`Unknown flag: ${flag}`);
    }
  }
  if (args.rows.length === 0) throw new Error("--rows produced an empty set");
  args.rows.forEach((label) => {
    const row = getRow(label);
    if (row.mode !== "lock") {
      throw new Error(`Row ${label} has mode=${row.mode}; postlock requires mode=lock rows.`);
    }
  });
  return args;
}

function printHelpAndExit(code) {
  process.stderr.write(
    `usage: node scripts/mesa-phase6-postlock.mjs [--fan-out <N>] [--rows <labels>] [--out-logs <dir>] [--skip-probe-slate] [--force]\n` +
    `default rows: ${LOCK_LABELS.join(",")}\n`,
  );
  process.exit(code);
}

function ensureDir(rel) {
  const abs = path.resolve(repoRoot, rel);
  mkdirSync(abs, { recursive: true });
  return abs;
}

// Spawn one stage for one row. Resume-safe via `manifest.json` existence check.
function spawnStage({ stage, label, outLogsAbs, force }) {
  return new Promise((resolve) => {
    const row = getRow(label);
    const polLabel = policyLabel(row);
    const policyAbs = path.resolve(repoRoot, policyPath(row));
    const outRel = `results/proof/phase6/${stage}/${label}`;
    const outAbs = ensureDir(outRel);
    const manifestAbs = path.join(outAbs, "manifest.json");

    if (!existsSync(policyAbs)) {
      console.error(`[${stage}:${label}] policy missing: ${policyPath(row)} — skipping (use Phase 6 lock runner first)`);
      resolve({ stage, label, exitCode: 2, wallSeconds: 0, skipped: false, missingPolicy: true });
      return;
    }
    if (!force && existsSync(manifestAbs)) {
      console.log(`[${stage}:${label}] manifest exists, skipping (--force to re-run)`);
      resolve({ stage, label, exitCode: 0, wallSeconds: 0, skipped: true });
      return;
    }

    const scriptRel = stage === "probe-slate"
      ? "scripts/mesa-probe-slate.mjs"
      : "scripts/mesa-intervention-battery.mjs";
    const args = [
      scriptRel,
      "--policy", policyAbs,
      "--policy-label", polLabel,
      "--out", outAbs,
    ];

    const logPath = path.join(outLogsAbs, `${stage}__${label}.log`);
    const logStream = createWriteStream(logPath, { flags: "w" });
    const startedMs = Date.now();

    logStream.write(`# stage ${stage} label ${label}\n`);
    logStream.write(`# started ${new Date(startedMs).toISOString()}\n`);
    logStream.write(`# command: ${process.execPath} ${args.join(" ")}\n`);
    logStream.write(`# policy-label: ${polLabel}\n`);
    logStream.write(`# ─────────────────────────────────────────────────────────────\n`);

    console.log(`[${stage}:${label}] spawn ▸ log=${path.relative(repoRoot, logPath)}`);

    const child = trackChild(spawn(process.execPath, args, {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, NODE_NO_WARNINGS: "1" },
    }));
    child.stdout.pipe(logStream, { end: false });
    child.stderr.pipe(logStream, { end: false });

    child.on("exit", (code, signal) => {
      const wallSeconds = (Date.now() - startedMs) / 1000;
      logStream.write(`# ─────────────────────────────────────────────────────────────\n`);
      logStream.write(`# finished ${new Date().toISOString()} exit=${code} signal=${signal} wall=${wallSeconds.toFixed(2)}s\n`);
      logStream.end();
      console.log(`[${stage}:${label}] exit=${code ?? "null"} wall=${wallSeconds.toFixed(1)} s`);
      resolve({ stage, label, exitCode: code, signal, wallSeconds, skipped: false });
    });
  });
}

// Simple worker pool — at most fanOut shards in flight; pull from queue when free.
async function runPool(items, fanOut, spawnFn) {
  const queue = items.slice();
  const results = [];
  let active = 0;
  return await new Promise((resolveAll, rejectAll) => {
    const launch = () => {
      while (active < fanOut && queue.length > 0) {
        const item = queue.shift();
        active += 1;
        spawnFn(item).then((r) => {
          results.push(r);
          active -= 1;
          if (queue.length === 0 && active === 0) resolveAll(results);
          else launch();
        }).catch(rejectAll);
      }
    };
    launch();
  });
}

// Read the basin-position row's mean_old_basin_preference from the
// intervention-battery CSV. Returns the number or null if anything is off.
function readBasinPref(label) {
  const row = getRow(label);
  const slug = slugifyLabel(policyLabel(row));
  const csvAbs = path.resolve(
    repoRoot,
    `results/proof/phase6/intervention-battery/${label}/${slug}_basin-internalization.csv`,
  );
  if (!existsSync(csvAbs)) return { value: null, reason: `csv missing: ${csvAbs}` };
  const text = readFileSync(csvAbs, "utf8");
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return { value: null, reason: "csv has no rows" };
  const header = lines[0].split(",");
  const idxChannel = header.indexOf("channel");
  const idxPref = header.indexOf("mean_old_basin_preference");
  if (idxChannel < 0 || idxPref < 0) {
    return { value: null, reason: `header missing channel/mean_old_basin_preference (got: ${header.join(",")})` };
  }
  for (const line of lines.slice(1)) {
    const cells = line.split(",");
    if (cells[idxChannel] === "basin-position") {
      const v = Number(cells[idxPref]);
      return Number.isFinite(v) ? { value: v } : { value: null, reason: `non-finite ${cells[idxPref]}` };
    }
  }
  return { value: null, reason: "no basin-position row" };
}

function verdictFor(value) {
  if (value == null || !Number.isFinite(value)) return "unknown";
  if (value < BREACH_THRESHOLD - AMBIGUOUS_MARGIN) return "protected";
  if (value > BREACH_THRESHOLD + AMBIGUOUS_MARGIN) return "collapsed";
  return "ambiguous";
}

function pad(s, w) { return String(s).padEnd(w); }
function padN(s, w) { return String(s).padStart(w); }

function printDecisionTable(readings) {
  console.log("");
  console.log("─── Phase 6 Read-Back: mean_old_basin_preference (basin-position channel) ───");
  console.log(`    gate: > ${BREACH_THRESHOLD} = collapsed; ±${AMBIGUOUS_MARGIN} of gate = ambiguous (Branch 4 midpoint)`);
  console.log("");
  console.log(" " + pad("label", 36) + " " + pad("λ", 5) + " " + pad("expect", 10) + " " + padN("obs", 9) + " " + pad("verdict", 10) + " " + pad("Phase5 ref", 16) + " match");
  console.log(" " + "─".repeat(36) + " " + "─".repeat(5) + " " + "─".repeat(10) + " " + "─".repeat(9) + " " + "─".repeat(10) + " " + "─".repeat(16) + " ─────");

  let noopCleanFlip = false;
  let rescaleMismatch = false;
  let anyAmbiguous = false;
  let allMatch = true;
  let anyUnknown = false;

  for (const r of readings) {
    const row = getRow(r.label);
    const verdict = verdictFor(r.value);
    const expect = row.expect;
    const ref = PHASE5_BASELINE[row.lambda];
    const refStr = ref ? `${ref.mean_old_basin_pref.toFixed(3)} (${ref.read.slice(0, 4)})` : "—";
    const obsStr = r.value == null ? "?" : r.value.toFixed(3);
    let matchSym;
    if (verdict === "unknown") {
      matchSym = "?"; anyUnknown = true; allMatch = false;
    } else if (verdict === "ambiguous") {
      matchSym = "~"; anyAmbiguous = true; allMatch = false;
    } else if (verdict === expect) {
      matchSym = "✓";
    } else {
      matchSym = "✗"; allMatch = false;
      if (row.condition === "noop_delta") noopCleanFlip = true;
      else rescaleMismatch = true;
    }
    console.log(" " + pad(r.label, 36) + " " + pad(row.lambda, 5) + " " + pad(expect, 10) + " " + padN(obsStr, 9) + " " + pad(verdict, 10) + " " + pad(refStr, 16) + " " + matchSym);
  }

  console.log("");
  console.log("─── Branch decision (draft per PHASE6_LAMBDA_CONTROL.md ▸ Outcome Branches) ───");
  if (noopCleanFlip) {
    console.log(" BRANCH 1: No-op moves the cliff (clean side flip on a noop_delta row).");
    console.log(" → File pre-registered negative. Mesa is optimizer-artifact for proof-roadmap purposes;");
    console.log("   pull anniversary cliff promotion; leave Postulates 2/4 speculative.");
  } else if (anyAmbiguous || anyUnknown) {
    console.log(" BRANCH 4: Ambiguous row near threshold (or unread).");
    console.log(" → Add exactly one midpoint in the implicated bracket (0.91 for κ=2, or 0.975 for κ=0.5)");
    console.log("   before changing the public status.");
    if (anyUnknown) console.log("   (Some readings could not be parsed — investigate before midpoint.)");
  } else if (allMatch) {
    console.log(" BRANCH 2: No-op stable AND rescale follows the predicted map.");
    console.log(" → Phase 6 clears. Cliff may be cited as a controlled operating-envelope boundary,");
    console.log("   still with the normal caveat that Phase 4 and Phase 5 remain separate gates.");
  } else if (rescaleMismatch) {
    console.log(" BRANCH 3: No-op stable; rescale does NOT follow the predicted map.");
    console.log(" → Do not kill Mesa, but do not promote the cliff as a clean capacity law.");
    console.log("   Stage an optimizer diagnosis against reward normalization, value-loss scale,");
    console.log("   and gradient norms.");
  } else {
    console.log(" UNCLASSIFIED: logic gap — check the per-row table above.");
  }
  console.log("");
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const outLogsAbs = ensureDir(opts.outLogs);

  // Pre-flight: validate every row's policy file exists before launching anything.
  const missing = opts.rows.filter((label) => !existsSync(path.resolve(repoRoot, policyPath(getRow(label)))));
  if (missing.length > 0) {
    console.error(`[postlock] ${missing.length} policy file(s) missing — run the Phase 6 lock first:`);
    missing.forEach((l) => console.error(`  ${l}: ${policyPath(getRow(l))}`));
    process.exit(2);
  }

  console.log(`[postlock] rows=${opts.rows.length} fan-out=${opts.fanOut} skip-probe-slate=${opts.skipProbeSlate}`);
  console.log(`[postlock] logs dir: ${path.relative(repoRoot, outLogsAbs)}`);

  // ── Phase A: intervention-battery (decisive) ───────────────────────────
  console.log("");
  console.log(`[postlock] ─── Phase A: intervention-battery × ${opts.rows.length} rows, fan-out=${opts.fanOut} ───`);
  const phaseAStart = Date.now();
  const phaseAResults = await runPool(opts.rows, opts.fanOut, (label) =>
    spawnStage({ stage: "intervention-battery", label, outLogsAbs, force: opts.force }),
  );
  const phaseAWall = (Date.now() - phaseAStart) / 1000;
  const phaseAFail = phaseAResults.filter((r) => r.exitCode !== 0 && !r.skipped);
  console.log(`[postlock] Phase A complete: ${(phaseAWall / 60).toFixed(2)} min wall, ${phaseAFail.length} failed of ${phaseAResults.length}`);

  // Read decisive metric from each row's CSV
  const readings = opts.rows.map((label) => {
    const { value, reason } = readBasinPref(label);
    if (value == null) console.warn(`[postlock] ${label}: ${reason}`);
    return { label, value };
  });

  printDecisionTable(readings);

  if (opts.skipProbeSlate) {
    process.exit(phaseAFail.length > 0 ? 1 : 0);
    return;
  }

  // ── Phase B: probe-slate (diagnostics) ─────────────────────────────────
  console.log(`[postlock] ─── Phase B: probe-slate × ${opts.rows.length} rows, fan-out=${opts.fanOut} ───`);
  const phaseBStart = Date.now();
  const phaseBResults = await runPool(opts.rows, opts.fanOut, (label) =>
    spawnStage({ stage: "probe-slate", label, outLogsAbs, force: opts.force }),
  );
  const phaseBWall = (Date.now() - phaseBStart) / 1000;
  const phaseBFail = phaseBResults.filter((r) => r.exitCode !== 0 && !r.skipped);
  console.log(`[postlock] Phase B complete: ${(phaseBWall / 60).toFixed(2)} min wall, ${phaseBFail.length} failed of ${phaseBResults.length}`);
  console.log("");
  console.log("[postlock] Phase A decision table is above. Probe-slate manifests are at:");
  for (const label of opts.rows) {
    console.log(`  results/proof/phase6/probe-slate/${label}/manifest.json`);
  }

  process.exit(phaseAFail.length + phaseBFail.length > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(`[postlock] fatal: ${err.stack || err.message}`);
  process.exit(2);
});
