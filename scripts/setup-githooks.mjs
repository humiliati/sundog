// Idempotent setup for repo-local git hooks.
//
// Runs from `npm install` via the `prepare` script. Points
// core.hooksPath at .githooks/ so the committed pre-commit hook is used,
// then ensures the hook script is executable on Unix.

import { execSync } from "node:child_process";
import { chmodSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

try {
  execSync("git rev-parse --is-inside-work-tree", { cwd: ROOT, stdio: "pipe" });
} catch {
  process.exit(0);
}

try {
  execSync("git config core.hooksPath .githooks", { cwd: ROOT, stdio: "pipe" });
  const hook = resolve(ROOT, ".githooks/pre-commit");
  if (existsSync(hook) && process.platform !== "win32") {
    chmodSync(hook, 0o755);
  }
  console.log("Configured .githooks/ as the repo hook path.");
} catch (err) {
  console.warn(`setup-githooks: could not configure hooks path (${err.message}). Hooks will not run.`);
  process.exit(0);
}
