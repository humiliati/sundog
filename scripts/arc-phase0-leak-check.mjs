// ARC Phase 0 leak-check.
//
// Re-runnable audit that asserts the discipline rules in
// docs/prereg/arc/PHASE0_TASK_SUBSET_SPEC.md and P0_BASELINES.md are held by
// the actual repo state:
//
//   1. The default inventory manifest is non-privileged (no evaluation
//      test outputs leaked into the standard output path).
//   2. The binding task register contains no `evaluation` rows, and any
//      `evaluation-blind` rows are flagged manual_inspection=no.
//   3. Baseline predictions cover only the registered training tasks.
//   4. No Kaggle scaffolding has appeared in the repo (no notebooks with
//      content, no kaggle credentials files).
//   5. ARC scripts other than the inventory and this leak-check do not
//      mention `evaluation` as a path/data literal.
//
// Exit code: 0 if all checks pass (WARN allowed). Nonzero if any FAIL.

import { readFile, readdir, stat } from "node:fs/promises";
import { join, resolve, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REGISTER_PATH = join(ROOT, "docs/prereg/arc/P0_TASK_REGISTER.csv");
const INVENTORY_MANIFEST_PATH = join(ROOT, "results/arc/phase0-inventory/manifest.json");
const PREDICTIONS_PATH = join(ROOT, "results/arc/phase0-baselines/predictions.json");
const SCRIPTS_DIR = join(ROOT, "scripts");
const SCAN_DIRS = [join(ROOT, "scripts"), join(ROOT, "docs"), join(ROOT, "tests"), join(ROOT, "notebooks")];
const SCRIPTS_ALLOWED_EVAL_LITERAL = new Set([
  "arc-phase0-inventory.mjs",
  "arc-phase0-leak-check.mjs",
  "arc-phase3-lodo.mjs",
  "arc-phase3-pttest.mjs"
]);

const results = [];
results.push(await checkInventoryManifest());
results.push(await checkRegister());
results.push(await checkPredictions());
results.push(await checkKaggleScaffolding());
results.push(await checkArcScriptLiterals());

let failCount = 0;
let warnCount = 0;
console.log("ARC Phase 0 leak check:");
console.log("");
const longest = Math.max(...results.map((r) => r.name.length));
for (const r of results) {
  if (r.status === "FAIL") failCount += 1;
  if (r.status === "WARN") warnCount += 1;
  console.log(`  [${r.status.padEnd(4)}]  ${r.name.padEnd(longest)}  ${r.reason ?? ""}`);
}
console.log("");
console.log(`${failCount} fail(s), ${warnCount} warn(s).`);
process.exit(failCount > 0 ? 1 : 0);

async function checkInventoryManifest() {
  try {
    const raw = await readFile(INVENTORY_MANIFEST_PATH, "utf8");
    const manifest = JSON.parse(raw);
    if (manifest.includeEvaluationTestOutput) {
      return {
        name: "inventory non-privileged",
        status: "FAIL",
        reason: "manifest.includeEvaluationTestOutput=true at the default inventory path; privileged audits must write to a _PRIVILEGED_AUDIT path"
      };
    }
    if (manifest.privilegedAudit) {
      return {
        name: "inventory non-privileged",
        status: "FAIL",
        reason: "manifest.privilegedAudit=true at the default inventory path"
      };
    }
    if (typeof manifest.evaluationPolicy === "string" && manifest.evaluationPolicy.includes("PRIVILEGED")) {
      return {
        name: "inventory non-privileged",
        status: "FAIL",
        reason: `manifest.evaluationPolicy advertises PRIVILEGED at default path: "${manifest.evaluationPolicy}"`
      };
    }
    return {
      name: "inventory non-privileged",
      status: "PASS",
      reason: `evaluationPolicy: ${JSON.stringify(manifest.evaluationPolicy ?? "(unset)")}`
    };
  } catch (err) {
    if (err.code === "ENOENT") {
      return {
        name: "inventory non-privileged",
        status: "SKIP",
        reason: `${relative(ROOT, INVENTORY_MANIFEST_PATH)} not present (regenerate with arc:phase0:inventory if needed)`
      };
    }
    return { name: "inventory non-privileged", status: "FAIL", reason: err.message };
  }
}

async function checkRegister() {
  try {
    const raw = await readFile(REGISTER_PATH, "utf8");
    const rows = parseCsv(raw);
    const violations = [];
    let trainingCount = 0;
    let blindCount = 0;
    for (const row of rows) {
      if (row.split === "training") {
        trainingCount += 1;
      } else if (row.split === "evaluation-blind") {
        blindCount += 1;
        if (row.manual_inspection !== "no") {
          violations.push(`${row.task_id}: evaluation-blind row must have manual_inspection=no (has "${row.manual_inspection}")`);
        }
      } else if (row.split === "evaluation") {
        violations.push(`${row.task_id}: split=evaluation is forbidden in the register; use evaluation-blind with manual_inspection=no`);
      } else if (row.split) {
        violations.push(`${row.task_id}: unknown split "${row.split}"`);
      }
    }
    if (violations.length > 0) {
      return { name: "register discipline", status: "FAIL", reason: violations.slice(0, 5).join("; ") };
    }
    return {
      name: "register discipline",
      status: "PASS",
      reason: `${trainingCount} training, ${blindCount} evaluation-blind, 0 evaluation`
    };
  } catch (err) {
    return { name: "register discipline", status: "FAIL", reason: `cannot read register: ${err.message}` };
  }
}

async function checkPredictions() {
  try {
    const regRaw = await readFile(REGISTER_PATH, "utf8");
    const regRows = parseCsv(regRaw);
    const allowedIds = new Set(
      regRows.filter((r) => r.split === "training" && r.status === "include").map((r) => r.task_id)
    );

    const predRaw = await readFile(PREDICTIONS_PATH, "utf8");
    const preds = JSON.parse(predRaw);
    const predIds = new Set(preds.map((p) => p.task_id));

    const leaks = [...predIds].filter((id) => !allowedIds.has(id));
    if (leaks.length > 0) {
      return {
        name: "predictions in register",
        status: "FAIL",
        reason: `${leaks.length} prediction(s) for non-registered tasks: ${leaks.slice(0, 5).join(", ")}`
      };
    }
    return {
      name: "predictions in register",
      status: "PASS",
      reason: `${predIds.size}/${allowedIds.size} registered tasks predicted`
    };
  } catch (err) {
    if (err.code === "ENOENT") {
      return {
        name: "predictions in register",
        status: "SKIP",
        reason: `${relative(ROOT, PREDICTIONS_PATH)} missing; baseline run not yet completed`
      };
    }
    return { name: "predictions in register", status: "FAIL", reason: err.message };
  }
}

async function checkKaggleScaffolding() {
  const offenders = [];
  for (const dir of SCAN_DIRS) {
    for await (const file of walk(dir)) {
      const lower = file.toLowerCase();
      const base = lower.split(/[\\/]/).pop();
      if (base === "kaggle.json") {
        offenders.push(`${relative(ROOT, file)} (kaggle credentials)`);
      } else if (lower.endsWith(".ipynb")) {
        const size = (await stat(file)).size;
        if (size > 0) {
          offenders.push(`${relative(ROOT, file)} (non-empty notebook, ${size} bytes)`);
        }
      }
    }
  }
  if (offenders.length > 0) {
    return { name: "no Kaggle scaffolding", status: "FAIL", reason: offenders.join("; ") };
  }
  return {
    name: "no Kaggle scaffolding",
    status: "PASS",
    reason: "no kaggle.json, no non-empty .ipynb in scripts/docs/tests/notebooks"
  };
}

async function checkArcScriptLiterals() {
  let scripts;
  try {
    scripts = (await readdir(SCRIPTS_DIR)).filter((name) => name.startsWith("arc-") && name.endsWith(".mjs"));
  } catch (err) {
    return { name: "ARC scripts: no eval literals", status: "FAIL", reason: err.message };
  }
  const violations = [];
  for (const name of scripts) {
    if (SCRIPTS_ALLOWED_EVAL_LITERAL.has(name)) {
      continue;
    }
    const text = await readFile(join(SCRIPTS_DIR, name), "utf8");
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i += 1) {
      if (lines[i].includes("evaluation")) {
        violations.push(`${name}:${i + 1}: ${lines[i].trim().slice(0, 100)}`);
      }
    }
  }
  if (violations.length > 0) {
    return {
      name: "ARC scripts: no eval literals",
      status: "FAIL",
      reason: violations.slice(0, 3).join(" | ") + (violations.length > 3 ? ` (+${violations.length - 3} more)` : "")
    };
  }
  return {
    name: "ARC scripts: no eval literals",
    status: "PASS",
    reason: `scanned ${scripts.length - SCRIPTS_ALLOWED_EVAL_LITERAL.size} non-allowlisted ARC script(s)`
  };
}

async function* walk(dir) {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.name === "node_modules" || entry.name === ".git" || entry.name === "dist" || entry.name === "out") {
      continue;
    }
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      yield* walk(full);
    } else {
      yield full;
    }
  }
}

function parseCsv(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n").filter((line) => line.length > 0);
  if (lines.length === 0) return [];
  const header = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    return Object.fromEntries(header.map((column, index) => [column, cells[index] ?? ""]));
  });
}

function parseCsvLine(line) {
  const cells = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === "\"" && line[i + 1] === "\"") {
        cell += "\"";
        i += 1;
      } else if (ch === "\"") {
        inQuotes = false;
      } else {
        cell += ch;
      }
    } else if (ch === "\"") {
      inQuotes = true;
    } else if (ch === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += ch;
    }
  }
  cells.push(cell);
  return cells;
}
