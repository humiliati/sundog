#!/usr/bin/env node
// scripts/cut2-publication-plumbing-guard.mjs
//
// C5 publication-plumbing freeze guard. See
// docs/prereg/structural-failure-coincidence/P2_CUT2_C5_PUBLICATION_PLUMBING_FREEZE.md
//
// Default-deny / allowlist-complement. Anything outside the allowlist tripping
// this guard ⇒ terminal `PUBLICATION_PLUMBING_VIOLATION`, never PASS.
//
// Modes:
//   snapshot --out <file>        Snapshot full-tree SHA-256s outside the
//                                allowlist. Run BEFORE a Cut-2 step.
//   check --snapshot <file>      Re-hash and diff vs snapshot. Exit non-zero
//                                with PUBLICATION_PLUMBING_VIOLATION on any
//                                out-of-allowlist new / removed / modified
//                                path. Run AFTER a Cut-2 step.
//   hash-file <path>             Print SHA-256 of a single file (utility).
//   hash-canonical-json <path>   Print SHA-256 of a JSON file rendered as
//                                canonical (sorted-keys, no-whitespace) bytes.
//                                Used to pin the manifest hash regardless of
//                                pretty-printing.
//
// Semantics chosen and frozen in the C5 audit notes: **snapshot mode**.
// Pre-run snapshots full-tree state (tracked + untracked + ignored, normalized
// paths, symlink-rejected). Post-run rejects new path, removed path, or
// content-hash delta outside the allowlist. Tolerates a normal dirty workflow
// while still catching pre-existing-file overwrites because the snapshot
// records content hashes, not just presence.
//
// Allowlist source of truth: results/structural-failure/cut2-prereg/c5-write-path-manifest.json
//
// CAVEAT: this file MUST NOT be modified during a Cut-2 run. Doing so would
// itself trip the guard.

import { createHash } from "node:crypto";
import { readdir, readFile, lstat, realpath, mkdir, writeFile } from "node:fs/promises";
import { join, relative, resolve, sep, dirname } from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIR = dirname(SCRIPT_PATH);

const SNAPSHOT_EXCLUDES = [".git", "node_modules"];
const DEFAULT_MANIFEST_REPO_REL =
  "results/structural-failure/cut2-prereg/c5-write-path-manifest.json";
const VIOLATION = "PUBLICATION_PLUMBING_VIOLATION";

function repoRoot() {
  try {
    const root = execSync("git rev-parse --show-toplevel", {
      encoding: "utf8",
      cwd: SCRIPT_DIR,
    }).trim();
    return resolve(root);
  } catch {
    throw new Error(
      "git rev-parse failed; the C5 guard requires a git working tree (used for repo-root resolution only, never for the diff itself)."
    );
  }
}

function toPosix(p) {
  return p.split(sep).join("/");
}

// Convert a simple shell-style glob to a regex anchored to repo-relative paths.
// Supported tokens: `*` (matches a single path segment), trailing `/` (matches
// any descendant under the prefix). Sufficient for the allowlist
// `results/structural-failure/cut2-*/`.
function globToRegex(glob) {
  let pattern = glob.replaceAll(".", "\\.").replaceAll("*", "[^/]*");
  if (pattern.endsWith("/")) pattern = pattern + ".+";
  return new RegExp("^" + pattern + "$");
}

function hashBytes(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function hashFile(absPath) {
  const bytes = await readFile(absPath);
  return hashBytes(bytes);
}

// Canonical JSON: sorted keys, no whitespace. Used to pin a manifest's hash
// independently of its on-disk pretty-printing.
function canonicalize(value) {
  if (Array.isArray(value)) {
    return "[" + value.map(canonicalize).join(",") + "]";
  }
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return (
      "{" +
      keys
        .map((k) => JSON.stringify(k) + ":" + canonicalize(value[k]))
        .join(",") +
      "}"
    );
  }
  return JSON.stringify(value);
}

async function loadManifest(absRepo) {
  const manifestAbs = resolve(absRepo, DEFAULT_MANIFEST_REPO_REL);
  const raw = await readFile(manifestAbs, "utf8");
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed.allowlist) || parsed.allowlist.length === 0) {
    throw new Error("manifest missing or empty allowlist");
  }
  if (parsed.default_deny !== true) {
    throw new Error("manifest must declare default_deny: true");
  }
  return {
    manifest: parsed,
    manifestRelPath: DEFAULT_MANIFEST_REPO_REL,
    manifestCanonicalSha256: hashBytes(Buffer.from(canonicalize(parsed))),
    manifestRawSha256: hashBytes(Buffer.from(raw)),
  };
}

async function walkTree(absRepo, allowlistRegexes) {
  const out = []; // { rel: string, abs: string }
  const stack = [absRepo];
  while (stack.length > 0) {
    const dir = stack.pop();
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch (err) {
      if (err.code === "ENOENT") continue;
      throw err;
    }
    for (const entry of entries) {
      const abs = join(dir, entry.name);
      const rel = toPosix(relative(absRepo, abs));
      // Top-level excludes (.git, node_modules) — never relevant to shipping
      // surfaces and would only add hashing cost.
      const topSeg = rel.split("/")[0];
      if (SNAPSHOT_EXCLUDES.includes(topSeg)) continue;
      // Symlink escape rejection — at lstat time, before we follow.
      const lst = await lstat(abs);
      if (lst.isSymbolicLink()) {
        const real = await realpath(abs);
        if (!resolve(real).startsWith(absRepo + sep) && resolve(real) !== absRepo) {
          throw new Error(
            `symlink escape rejected: ${rel} -> ${real} (outside repo root)`
          );
        }
        // Symlinks within the repo are tolerated as files (hash their content
        // via realpath read), not followed as directories.
        if (entry.isDirectory()) continue;
      }
      if (lst.isDirectory()) {
        stack.push(abs);
        continue;
      }
      if (!lst.isFile()) continue;
      // Allowlisted paths are not snapshotted; they're free to change.
      if (allowlistRegexes.some((r) => r.test(rel))) continue;
      out.push({ rel, abs });
    }
  }
  // Deterministic ordering.
  out.sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0));
  return out;
}

async function snapshot(absRepo, outPath) {
  const { manifest, manifestRelPath, manifestCanonicalSha256, manifestRawSha256 } =
    await loadManifest(absRepo);
  const allowlistRegexes = manifest.allowlist.map(globToRegex);
  const files = await walkTree(absRepo, allowlistRegexes);
  const fileRecords = [];
  for (const f of files) {
    fileRecords.push({ path: f.rel, sha256: await hashFile(f.abs) });
  }
  const payload = {
    guard_version: 1,
    mode: "snapshot",
    frozen_at_iso8601: new Date().toISOString(),
    repo_root: absRepo,
    manifest_path: manifestRelPath,
    manifest_canonical_sha256: manifestCanonicalSha256,
    manifest_raw_sha256: manifestRawSha256,
    allowlist: manifest.allowlist,
    snapshot_excludes: SNAPSHOT_EXCLUDES,
    file_count: fileRecords.length,
    files: fileRecords,
  };
  // Snapshot-of-snapshot hash (over the sorted file records) so a downstream
  // reader can verify the snapshot file itself wasn't tampered with.
  payload.files_field_sha256 = hashBytes(Buffer.from(canonicalize(fileRecords)));
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(payload, null, 2) + "\n");
  console.log(
    `[c5-guard] snapshot OK — ${fileRecords.length} files hashed → ${relative(
      absRepo,
      outPath
    )}`
  );
  console.log(`[c5-guard] manifest_canonical_sha256 = ${manifestCanonicalSha256}`);
  console.log(`[c5-guard] files_field_sha256        = ${payload.files_field_sha256}`);
}

async function check(absRepo, snapshotPath) {
  const { manifest, manifestCanonicalSha256, manifestRawSha256 } = await loadManifest(
    absRepo
  );
  const raw = await readFile(snapshotPath, "utf8");
  const snap = JSON.parse(raw);
  if (snap.manifest_canonical_sha256 !== manifestCanonicalSha256) {
    fail(
      `manifest canonical sha256 drift: snapshot=${snap.manifest_canonical_sha256} current=${manifestCanonicalSha256} ` +
        `(the write-path manifest changed between snapshot and check — refuse to validate)`
    );
  }
  const allowlistRegexes = snap.allowlist.map(globToRegex);
  const files = await walkTree(absRepo, allowlistRegexes);

  const oldMap = new Map(snap.files.map((f) => [f.path, f.sha256]));
  const newMap = new Map();
  for (const f of files) newMap.set(f.rel, await hashFile(f.abs));

  const added = [];
  const removed = [];
  const modified = [];
  for (const [p, h] of newMap) {
    if (!oldMap.has(p)) added.push(p);
    else if (oldMap.get(p) !== h)
      modified.push({ path: p, before: oldMap.get(p), after: h });
  }
  for (const [p] of oldMap) {
    if (!newMap.has(p)) removed.push(p);
  }

  const violations = added.length + removed.length + modified.length;
  if (violations === 0) {
    console.log(
      `[c5-guard] check OK — no out-of-allowlist deltas across ${newMap.size} files`
    );
    console.log(`[c5-guard] manifest_canonical_sha256 = ${manifestCanonicalSha256}`);
    console.log(`[c5-guard] manifest_raw_sha256       = ${manifestRawSha256}`);
    process.exit(0);
  }

  console.error(`[c5-guard] ${VIOLATION}: ${violations} out-of-allowlist deltas`);
  if (added.length > 0) {
    console.error(`  NEW (${added.length}):`);
    for (const p of added) console.error(`    + ${p}`);
  }
  if (removed.length > 0) {
    console.error(`  REMOVED (${removed.length}):`);
    for (const p of removed) console.error(`    - ${p}`);
  }
  if (modified.length > 0) {
    console.error(`  MODIFIED (${modified.length}):`);
    for (const m of modified)
      console.error(`    ~ ${m.path}  ${m.before.slice(0, 12)} → ${m.after.slice(0, 12)}`);
  }
  console.error(
    `[c5-guard] terminal-dominant verdict: ${VIOLATION}. ` +
      "Run is VOID per P2_CUT2_C5; no PASS may be reported. " +
      "The guard scope is never re-curated post-hoc."
  );
  process.exit(2);
}

function fail(msg) {
  console.error(`[c5-guard] ${VIOLATION}: ${msg}`);
  process.exit(2);
}

async function main() {
  const [, , cmd, ...rest] = process.argv;
  const absRepo = repoRoot();

  if (cmd === "snapshot") {
    const i = rest.indexOf("--out");
    if (i < 0 || !rest[i + 1]) {
      console.error("usage: snapshot --out <file>");
      process.exit(64);
    }
    const out = resolve(absRepo, rest[i + 1]);
    await snapshot(absRepo, out);
    return;
  }
  if (cmd === "check") {
    const i = rest.indexOf("--snapshot");
    if (i < 0 || !rest[i + 1]) {
      console.error("usage: check --snapshot <file>");
      process.exit(64);
    }
    const snap = resolve(absRepo, rest[i + 1]);
    await check(absRepo, snap);
    return;
  }
  if (cmd === "hash-file") {
    const target = rest[0];
    if (!target) {
      console.error("usage: hash-file <path>");
      process.exit(64);
    }
    console.log(await hashFile(resolve(absRepo, target)));
    return;
  }
  if (cmd === "hash-canonical-json") {
    const target = rest[0];
    if (!target) {
      console.error("usage: hash-canonical-json <path>");
      process.exit(64);
    }
    const raw = await readFile(resolve(absRepo, target), "utf8");
    console.log(hashBytes(Buffer.from(canonicalize(JSON.parse(raw)))));
    return;
  }
  console.error(
    "usage:\n" +
      "  cut2-publication-plumbing-guard.mjs snapshot --out <file>\n" +
      "  cut2-publication-plumbing-guard.mjs check    --snapshot <file>\n" +
      "  cut2-publication-plumbing-guard.mjs hash-file <path>\n" +
      "  cut2-publication-plumbing-guard.mjs hash-canonical-json <path>"
  );
  process.exit(64);
}

main().catch((err) => {
  console.error(`[c5-guard] ${VIOLATION}: ${err.message}`);
  process.exit(2);
});
