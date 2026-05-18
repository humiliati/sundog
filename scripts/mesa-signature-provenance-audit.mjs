#!/usr/bin/env node
// PHASE1_PRIME_SPEC v1.2 §5: static-analysis audit of the signature path.
// Runs four pre-registered leakage tests (LT1–LT4) against the reference
// implementation source tree and emits a green/red/yellow verdict.
//
// Usage:
//   node scripts/mesa-signature-provenance-audit.mjs --out <out-dir>
//   node scripts/mesa-signature-provenance-audit.mjs --out <out-dir> --retrofit-from <dir1>,<dir2>
//
// Outputs:
//   <out-dir>/audit-report.json
//   <out-dir>/audit-report.txt
//   (with --retrofit-from) <run-dir>/signature-provenance-manifest.json updated in-place

import { execFileSync } from "node:child_process";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const AUDIT_SCRIPT_VERSION = "v1.0";
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// ===========================================================================
// Pre-registered allowlists and effect table (PHASE1_PRIME_SPEC v1.2 §5)
// ===========================================================================

// LT2 allowlist: fields that may legitimately be read by both
// trueSignature() and rewardChannels() without triggering LT2.
// Field-level granularity per v1.2; wholesale `this.config` is too
// permissive because of shallow-copy mutation.
const LT2_ALLOWLIST = Object.freeze([
  // Agent + goal position (documented shared geometry baseline)
  "this.x",
  "this.xGoal",
  // Construction-immutable config fields (set once at env init,
  // never written by any intervention)
  "this.config.delta",
  "this.config.rewardControlAlpha",
  "this.config.falseBasinSigma",
  "this.config.falseBasinBeta",
  "this.config.arenaHalfWidth",
  "this.config.action_scale",
  "this.config.actionScale",
  // Functions/methods (not state)
  "this.trueSignature",
]);

// LT2 explicit fail-set: mutable fields known to be written by
// interventions or step(). Read by both methods → leakage.
const LT2_INTERVENTION_WRITTEN_FIELDS = Object.freeze([
  "this.activeRewardEdit",
  "this.activeObservationEdit",
  "this.activeSignatureSensorEdit",
  "this.config.falseBasinCenter",
  "this.config.sigmaS",
  "this.config.textureNoiseStd",
  "this.config.delaySteps",
  "this.config.perChannelNoise",
]);

// LT4 pre-registered effect table.
// For each intervention channel, which methods should be affected (✓)
// and which must not be (✗). "by-geometry" marks intentional couplings
// that propagate via this.xGoal (the geometry baseline).
const LT4_EFFECT_TABLE = Object.freeze({
  "signature-sensor": {
    writes: ["this.activeSignatureSensorEdit"],
    affects_trueSignature: false,
    affects_rewardChannels: false,
    affects_observe: true,
    note: "scales measured signature samples, not true S(x) or reward",
  },
  "reward": {
    writes: ["this.activeRewardEdit"],
    affects_trueSignature: false,
    affects_rewardChannels: true,
    affects_observe: false,
    note: "scales/shifts live reward channel",
  },
  "geometry": {
    writes: ["this.xGoal"],
    affects_trueSignature: "by-geometry",
    affects_rewardChannels: "by-geometry",
    affects_observe: "by-geometry",
    note: "moves goal; legitimately affects all three via shared geometry baseline",
  },
  "basin-position": {
    writes: ["this.config.falseBasinCenter"],
    affects_trueSignature: false,
    affects_rewardChannels: true,
    affects_observe: false,
    note: "moves x_false; only reward surface depends on basin center",
  },
  "observation": {
    writes: ["this.activeObservationEdit"],
    affects_trueSignature: false,
    affects_rewardChannels: false,
    affects_observe: true,
    note: "overwrites observation channels at output time",
  },
});

// LT1 declared sensor-tier observation specs (mirrors
// SENSOR_TIER_OBSERVATION_SPEC in mesa-harness.mjs).
const SENSOR_TIER_CHANNELS = Object.freeze({
  "privileged-field": ["pos_x", "pos_y", "xGoal_x", "xGoal_y", "trueSignature", "trueGrad_x", "trueGrad_y"],
  "local-probe-field": ["pos_x", "pos_y", "probe_0", "probe_1", "probe_2", "probe_3"],
  "delayed-field": ["pos_x", "pos_y", "probe_0", "probe_1", "probe_2", "probe_3"],
  "noisy-field": ["pos_x", "pos_y", "probe_0", "probe_1", "probe_2", "probe_3"],
});

// LT3 allowlist: info fields that the training loop may read into
// the policy observation tensor (none, in the reference impl). All
// other info fields are logging-only.
const LT3_INFO_SENSOR_FIELDS = Object.freeze([
  // empty — the policy reads `obs`, not `info`
]);

// LT3 forbidden: info fields that must not flow into the policy obs.
const LT3_FORBIDDEN_INFO_FIELDS = Object.freeze([
  "x_goal",
  "true_signature",
  "true_gradient",
  "x_false",
  "privileged_position",
  "metrics",
]);

// ===========================================================================
// Source-code parsing helpers
// ===========================================================================

function gitSha() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: REPO_ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

async function loadSource(relativePath) {
  const absolute = path.resolve(REPO_ROOT, relativePath);
  return await readFile(absolute, "utf8");
}

// Extract the body of a method `methodName` inside `className`.
// Uses brace-depth scanning to handle nested braces correctly.
function extractMethodBody(source, className, methodName) {
  const classRe = new RegExp(`(?:export\\s+)?class\\s+${className}\\b`);
  const classMatch = classRe.exec(source);
  if (!classMatch) return null;
  const classStart = classMatch.index;
  // Find the class body start
  let i = source.indexOf("{", classStart);
  if (i < 0) return null;
  // Find the class body end via brace matching
  let depth = 1;
  let classEnd = -1;
  for (let j = i + 1; j < source.length; j += 1) {
    const ch = source[j];
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) { classEnd = j; break; }
    }
  }
  if (classEnd < 0) return null;
  const classBody = source.slice(i + 1, classEnd);

  // Find the method within the class body
  const methodRe = new RegExp(`(?:^|\\n)\\s*(?:async\\s+)?${methodName}\\s*\\(`);
  const mMatch = methodRe.exec(classBody);
  if (!mMatch) return null;
  // Find the opening { after the method's argument list
  let k = classBody.indexOf("{", mMatch.index);
  if (k < 0) return null;
  // Walk braces
  let mDepth = 1;
  let mEnd = -1;
  for (let j = k + 1; j < classBody.length; j += 1) {
    const ch = classBody[j];
    if (ch === "{") mDepth += 1;
    else if (ch === "}") {
      mDepth -= 1;
      if (mDepth === 0) { mEnd = j; break; }
    }
  }
  if (mEnd < 0) return null;
  return classBody.slice(k + 1, mEnd);
}

// Extract `this.X` and `this.X.Y` reads from a method body.
// Returns a deduped sorted array of access-paths up to 2 levels deep.
function extractThisReads(methodBody) {
  if (!methodBody) return [];
  const reads = new Set();
  // Match this.foo or this.foo.bar
  const re = /this\.([A-Za-z_$][\w$]*)(?:\.([A-Za-z_$][\w$]*))?/g;
  let match;
  while ((match = re.exec(methodBody)) !== null) {
    const first = match[1];
    const second = match[2];
    if (second) reads.add(`this.${first}.${second}`);
    else reads.add(`this.${first}`);
  }
  return [...reads].sort();
}

// ===========================================================================
// LT1 — Agent observation tuple does not include x_goal (non-privileged tiers)
// ===========================================================================

async function runLT1() {
  const evidence = { tiers_checked: [], privileged_carved_out: false, failures: [] };
  let mesaCore;
  try {
    mesaCore = await loadSource("public/js/mesa-core.mjs");
  } catch (err) {
    return { verdict: "fail", evidence: { error: `cannot read mesa-core.mjs: ${err.message}` } };
  }

  const observeBody = extractMethodBody(mesaCore, "ShadowFieldEnv", "observe");
  if (!observeBody) {
    return { verdict: "fail", evidence: { error: "cannot locate ShadowFieldEnv.observe()" } };
  }

  // Sniff the observation construction. Look for the privileged-tier branch
  // explicitly and verify it's gated on PRIVILEGED_FIELD.
  const privilegedBranchRe = /sensorTier\s*===\s*SENSOR_TIERS\.PRIVILEGED_FIELD/;
  const privilegedGated = privilegedBranchRe.test(observeBody);
  evidence.privileged_carved_out = privilegedGated;
  if (!privilegedGated) {
    evidence.failures.push("privileged-tier branch in observe() is not gated on SENSOR_TIERS.PRIVILEGED_FIELD; LT1 cannot verify the carve-out");
  }

  // For each non-privileged tier, the observation construction must not
  // spread this.xGoal or trueGrad into the observation array. We do this
  // by scanning the non-privileged branch.
  // Heuristic: the privileged branch is in an `if` block; the else branch
  // is the non-privileged path. Sniff both for `...this.xGoal` and
  // `...trueGrad` references.
  const nonPrivBranchHasXGoal = /else\s*\{[^}]*\.\.\.this\.xGoal/s.test(observeBody);
  const nonPrivBranchHasTrueGrad = /else\s*\{[^}]*\.\.\.trueGrad/s.test(observeBody);

  for (const tier of ["local-probe-field", "delayed-field", "noisy-field"]) {
    evidence.tiers_checked.push(tier);
    if (nonPrivBranchHasXGoal) {
      evidence.failures.push(`${tier}: observation construction in non-privileged branch spreads this.xGoal`);
    }
    if (nonPrivBranchHasTrueGrad) {
      evidence.failures.push(`${tier}: observation construction in non-privileged branch spreads trueGrad`);
    }
  }

  const verdict = evidence.failures.length === 0 ? "pass" : "fail";
  return { verdict, evidence };
}

// ===========================================================================
// LT2 — S(x) and R(s,a) share only documented mutable state
// ===========================================================================

async function runLT2() {
  const evidence = { allowlist: LT2_ALLOWLIST, intersection: [], forbidden_overlap: [] };
  let mesaCore;
  try {
    mesaCore = await loadSource("public/js/mesa-core.mjs");
  } catch (err) {
    return { verdict: "fail", evidence: { error: `cannot read mesa-core.mjs: ${err.message}` } };
  }

  const trueSigBody = extractMethodBody(mesaCore, "ShadowFieldEnv", "trueSignature");
  const rewardBody = extractMethodBody(mesaCore, "ShadowFieldEnv", "rewardChannels");
  if (!trueSigBody) return { verdict: "fail", evidence: { error: "cannot locate trueSignature()" } };
  if (!rewardBody) return { verdict: "fail", evidence: { error: "cannot locate rewardChannels()" } };

  const sigReads = extractThisReads(trueSigBody);
  const rewardReads = extractThisReads(rewardBody);

  evidence.trueSignature_reads = sigReads;
  evidence.rewardChannels_reads = rewardReads;

  // Compute intersection
  const sigSet = new Set(sigReads);
  const intersection = rewardReads.filter((r) => sigSet.has(r)).sort();
  evidence.intersection = intersection;

  // Check intersection against allowlist
  const allowSet = new Set(LT2_ALLOWLIST);
  const outsideAllowlist = intersection.filter((r) => !allowSet.has(r));

  // Cross-check against the intervention-written field list
  const interventionWrittenSet = new Set(LT2_INTERVENTION_WRITTEN_FIELDS);
  const forbiddenOverlap = intersection.filter((r) => interventionWrittenSet.has(r));
  evidence.forbidden_overlap = forbiddenOverlap;

  if (outsideAllowlist.length > 0) {
    evidence.outside_allowlist = outsideAllowlist;
  }

  const verdict = (outsideAllowlist.length === 0 && forbiddenOverlap.length === 0) ? "pass" : "fail";
  return { verdict, evidence };
}

// ===========================================================================
// LT3 — Privileged info + labeled signature channel don't feed back into obs
// ===========================================================================

async function runLT3() {
  const evidence = { path_a: {}, path_b: {} };
  let trainPpo;
  try {
    trainPpo = await loadSource("training/mesa/train_ppo.py");
  } catch (err) {
    return { verdict: "fail", evidence: { error: `cannot read train_ppo.py: ${err.message}` } };
  }

  // Path A: info fields. The training loop reads `obs` from the bridge's
  // make/step response into the policy. `info` fields should not be
  // read into the obs tensor. Heuristic: look for any obs assignment
  // that derives from `info["..."]` references to forbidden fields.
  const forbiddenInfoUses = [];
  for (const field of LT3_FORBIDDEN_INFO_FIELDS) {
    // Look for patterns like obs = ... info["x_goal"] ... or
    // np.array([..., info["x_goal"], ...]) feeding into a policy.forward call.
    // Simple proxy: assert that info["forbidden_field"] never appears in the same
    // line as `obs` or `observation` (modulo logging context).
    const fieldRe = new RegExp(`info\\[["']${field}["']\\]`, "g");
    const matches = [...trainPpo.matchAll(fieldRe)];
    for (const m of matches) {
      // Get the surrounding line
      const lineStart = trainPpo.lastIndexOf("\n", m.index) + 1;
      const lineEnd = trainPpo.indexOf("\n", m.index);
      const line = trainPpo.slice(lineStart, lineEnd > 0 ? lineEnd : trainPpo.length);
      // Heuristic: a use is "into obs" if the same line assigns to obs/observation/policy_input
      if (/\b(obs|observation|policy_input)\s*=/.test(line) || /\bpolicy\.(forward|act)\(.*\binfo\[/.test(line)) {
        forbiddenInfoUses.push({ field, line: line.trim() });
      }
    }
  }
  evidence.path_a.forbidden_info_uses_into_obs = forbiddenInfoUses;

  // Path B: reward_channels.signature feedback. The labeled `signature` reward
  // channel should be consumed only in scalar-reward computation, not as
  // observation input.
  // Heuristic: find references to reward_channels["signature"] or
  // reward_channels.signature; assert they appear in scalar-reward
  // computation paths only.
  const sigChannelRefs = [];
  const sigChannelRe = /reward_channels\s*[\.\[]\s*["']?signature["']?\s*\]?/g;
  let match;
  while ((match = sigChannelRe.exec(trainPpo)) !== null) {
    const lineStart = trainPpo.lastIndexOf("\n", match.index) + 1;
    const lineEnd = trainPpo.indexOf("\n", match.index);
    const line = trainPpo.slice(lineStart, lineEnd > 0 ? lineEnd : trainPpo.length);
    sigChannelRefs.push(line.trim());
  }
  evidence.path_b.signature_channel_references = sigChannelRefs;

  // Look for any reward_channels["signature"] reference that flows into
  // obs/observation/policy_input
  const signatureIntoObs = sigChannelRefs.filter(
    (line) => /\b(obs|observation|policy_input)\b/.test(line),
  );
  evidence.path_b.signature_into_obs = signatureIntoObs;

  const verdict = (forbiddenInfoUses.length === 0 && signatureIntoObs.length === 0) ? "pass" : "fail";
  return { verdict, evidence };
}

// ===========================================================================
// LT4 — Probe/intervention channels match the pre-registered effect table
// ===========================================================================

async function runLT4() {
  const evidence = { effect_table: {}, deviations: [] };
  let mesaCore;
  try {
    mesaCore = await loadSource("public/js/mesa-core.mjs");
  } catch (err) {
    return { verdict: "fail", evidence: { error: `cannot read mesa-core.mjs: ${err.message}` } };
  }

  const interventionBody = extractMethodBody(mesaCore, "ShadowFieldEnv", "applyScheduledInterventions");
  if (!interventionBody) {
    return { verdict: "fail", evidence: { error: "cannot locate applyScheduledInterventions()" } };
  }
  // Transitively-expanded bodies — observe() in particular calls
  // sensorSamples → localProbeSamples → measuredSignature which reads
  // activeSignatureSensorEdit. The expanded body captures these reads
  // that a direct-body scan would miss.
  const trueSigBody = expandMethodBody(mesaCore, "ShadowFieldEnv", "trueSignature");
  const rewardBody = expandMethodBody(mesaCore, "ShadowFieldEnv", "rewardChannels");
  const observeBody = expandMethodBody(mesaCore, "ShadowFieldEnv", "observe");

  // For each intervention channel, scan the intervention body's branch
  // for `this.X = ...` writes, and verify they match the pre-registered
  // write field set.
  for (const [channel, spec] of Object.entries(LT4_EFFECT_TABLE)) {
    // Locate the branch for this channel
    const channelRe = new RegExp(`channel\\s*===\\s*["']${channel}["']`);
    const channelMatch = channelRe.exec(interventionBody);
    if (!channelMatch) {
      evidence.deviations.push(`channel "${channel}": no branch found in applyScheduledInterventions()`);
      continue;
    }
    // Pull the next ~600 chars after the channel match as the branch body
    const branchSlice = interventionBody.slice(channelMatch.index, channelMatch.index + 800);
    // Extract write targets
    const writeRe = /this\.(\w+(?:\.\w+)?)\s*=/g;
    const writes = new Set();
    let wMatch;
    while ((wMatch = writeRe.exec(branchSlice)) !== null) {
      writes.add(`this.${wMatch[1]}`);
    }
    // Stop at the next `else if` boundary
    const elseIfIdx = branchSlice.indexOf("else if");
    const actualWrites = [...writes].filter((w) => {
      if (elseIfIdx < 0) return true;
      const writeIdx = branchSlice.indexOf(`${w} =`);
      return writeIdx >= 0 && writeIdx < elseIfIdx;
    });

    const expectedWrites = new Set(spec.writes);
    const actualWriteSet = new Set(actualWrites);
    const matchesExpected =
      actualWrites.length === spec.writes.length &&
      spec.writes.every((w) => actualWriteSet.has(w));

    // Check method-read effects against the expected affect table
    const sigEffect = methodReadsAnyOf(trueSigBody, spec.writes);
    const rewardEffect = methodReadsAnyOf(rewardBody, spec.writes);
    const observeEffect = methodReadsAnyOf(observeBody, spec.writes);

    const actualBlock = {
      writes: actualWrites,
      affects_trueSignature: sigEffect,
      affects_rewardChannels: rewardEffect,
      affects_observe: observeEffect,
    };
    evidence.effect_table[channel] = { expected: spec, actual: actualBlock };

    if (!matchesExpected) {
      evidence.deviations.push(`channel "${channel}": writes ${JSON.stringify(actualWrites)} but expected ${JSON.stringify(spec.writes)}`);
    }

    // For each affected method, check that observed matches expected.
    // `"by-geometry"` is treated as true (effect should be observed,
    // and it should propagate via this.xGoal).
    const cmp = (expected, actual) => {
      if (expected === "by-geometry") return actual === true;
      return expected === actual;
    };
    if (!cmp(spec.affects_trueSignature, sigEffect)) {
      evidence.deviations.push(`channel "${channel}": trueSignature effect expected=${spec.affects_trueSignature}, actual=${sigEffect}`);
    }
    if (!cmp(spec.affects_rewardChannels, rewardEffect)) {
      evidence.deviations.push(`channel "${channel}": rewardChannels effect expected=${spec.affects_rewardChannels}, actual=${rewardEffect}`);
    }
    if (!cmp(spec.affects_observe, observeEffect)) {
      evidence.deviations.push(`channel "${channel}": observe effect expected=${spec.affects_observe}, actual=${observeEffect}`);
    }
  }

  const verdict = evidence.deviations.length === 0 ? "pass" : "fail";
  return { verdict, evidence };
}

function methodReadsAnyOf(methodBody, fields) {
  if (!methodBody) return false;
  for (const field of fields) {
    // Exact-path match only. The earlier parent-fallback (e.g. matching
    // `this.config` as a proxy for `this.config.falseBasinCenter`) was
    // over-permissive: many methods read `this.config` to pass to helper
    // functions (signatureField, etc.) that use *other* config fields.
    // The audit only counts a method as reading a field when the exact
    // access path appears.
    const fullPathRe = new RegExp(field.replaceAll(".", "\\.") + "\\b");
    if (fullPathRe.test(methodBody)) return true;
  }
  return false;
}

// Build a "transitively expanded" body for a method by concatenating
// the bodies of methods it directly calls via `this.foo(...)`. One-level
// expansion is sufficient for the reference impl's call graph
// (observe() → sensorSamples() → localProbeSamples() → measuredSignature();
// rewardChannels() → trueSignature(); trueSignature() → no helpers).
// LT4's effect detection runs against the expanded body so transitive
// field reads are captured.
function expandMethodBody(source, className, methodName, depth = 2) {
  const visited = new Set();
  const queue = [methodName];
  const bodies = [];
  while (queue.length > 0 && visited.size < depth + 4) {
    const m = queue.shift();
    if (visited.has(m)) continue;
    visited.add(m);
    const body = extractMethodBody(source, className, m);
    if (!body) continue;
    bodies.push(body);
    // Find `this.foo(` calls (excluding property reads) and queue them
    const callRe = /this\.([A-Za-z_$][\w$]*)\s*\(/g;
    let cm;
    while ((cm = callRe.exec(body)) !== null) {
      const calledName = cm[1];
      if (!visited.has(calledName)) queue.push(calledName);
    }
  }
  return bodies.join("\n\n");
}

// ===========================================================================
// Retrofit: emit/update v2 manifest in existing run directories
// ===========================================================================

async function retrofitManifest(runDir, auditResults, overall) {
  const v1ManifestPath = path.join(runDir, "manifest.json");
  let v1Exists = false;
  try {
    await stat(v1ManifestPath);
    v1Exists = true;
  } catch {
    // no v1 manifest at top level; check for sub-runs
  }

  // Walk sub-directories looking for any v1 manifest.json
  const v1Paths = [];
  if (v1Exists) v1Paths.push(v1ManifestPath);
  await walkForManifests(runDir, v1Paths);

  const updated = [];
  for (const v1Path of v1Paths) {
    const subRunDir = path.dirname(v1Path);
    const v2Path = path.join(subRunDir, "signature-provenance-manifest.json");

    let v2Existing = null;
    try {
      v2Existing = JSON.parse(await readFile(v2Path, "utf8"));
    } catch {
      // create from scratch
    }

    let v2 = v2Existing;
    if (!v2) {
      v2 = {
        phase_v2: "phase1-prime-signature-path",
        schema_version: "v1",
        emitted_at: new Date().toISOString(),
        git_sha: gitSha(),
        v1_manifest_path: path.relative(REPO_ROOT, v1Path).replaceAll("\\", "/"),
        synthesized_by_audit: true,
        note: "v2 manifest synthesized post-hoc by mesa-signature-provenance-audit.mjs --retrofit-from; the original v1 run predated the v2 emitter.",
      };
    }

    v2.leakage_audit_verdict = {
      audit_script_version: AUDIT_SCRIPT_VERSION,
      audit_run_at: new Date().toISOString(),
      git_sha: gitSha(),
      LT1_no_xgoal_in_obs: auditResults.LT1.verdict,
      LT2_disjoint_accessors: auditResults.LT2.verdict,
      LT3_no_log_feedback: auditResults.LT3.verdict,
      LT4_channel_independence: auditResults.LT4.verdict,
      overall,
    };

    await writeFile(v2Path, `${JSON.stringify(v2, null, 2)}\n`, "utf8");
    updated.push(path.relative(REPO_ROOT, v2Path).replaceAll("\\", "/"));
  }
  return updated;
}

async function walkForManifests(dir, out, depth = 0) {
  if (depth > 4) return;
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await walkForManifests(fullPath, out, depth + 1);
    } else if (entry.name === "manifest.json") {
      if (!out.includes(fullPath)) out.push(fullPath);
    }
  }
}

// ===========================================================================
// Main
// ===========================================================================

function parseArgs(argv) {
  const args = { out: "results/mesa/phase1-prime/audit", retrofitFrom: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === "--out") {
      args.out = argv[i + 1];
      i += 1;
    } else if (flag === "--retrofit-from") {
      args.retrofitFrom = argv[i + 1].split(",").map((s) => s.trim()).filter(Boolean);
      i += 1;
    }
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.out);
  await mkdir(outDir, { recursive: true });

  const results = {
    LT1: await runLT1(),
    LT2: await runLT2(),
    LT3: await runLT3(),
    LT4: await runLT4(),
  };

  const failed = Object.entries(results).filter(([, r]) => r.verdict === "fail");
  const passed = Object.entries(results).filter(([, r]) => r.verdict === "pass");
  // Decide overall:
  // - All pass → green
  // - LT1 or LT3 fail → red (load-bearing per spec §7)
  // - LT2 or LT4 fail (but LT1 + LT3 pass) → yellow (partial collapse)
  let overall = "green";
  if (results.LT1.verdict === "fail" || results.LT3.verdict === "fail") overall = "red";
  else if (failed.length > 0) overall = "yellow";

  const report = {
    audit_script_version: AUDIT_SCRIPT_VERSION,
    audit_run_at: new Date().toISOString(),
    git_sha: gitSha(),
    repo_root: REPO_ROOT,
    overall,
    pass_count: passed.length,
    fail_count: failed.length,
    results,
  };

  let retrofitUpdates = [];
  if (args.retrofitFrom.length > 0) {
    for (const runDir of args.retrofitFrom) {
      const abs = path.resolve(REPO_ROOT, runDir);
      const updates = await retrofitManifest(abs, results, overall);
      retrofitUpdates.push(...updates);
    }
    report.retrofit_updates = retrofitUpdates;
  }

  await writeFile(path.join(outDir, "audit-report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");

  // Text summary
  const lines = [];
  lines.push(`Phase 1' signature-provenance audit — ${AUDIT_SCRIPT_VERSION}`);
  lines.push(`run_at: ${report.audit_run_at}`);
  lines.push(`git_sha: ${report.git_sha ?? "(no git)"}`);
  lines.push("");
  lines.push(`overall: ${overall.toUpperCase()}`);
  lines.push("");
  for (const [name, r] of Object.entries(results)) {
    lines.push(`${name}: ${r.verdict.toUpperCase()}`);
    if (r.verdict === "fail" && r.evidence) {
      const failureKey = ["failures", "deviations", "outside_allowlist", "forbidden_overlap", "error"]
        .find((k) => r.evidence[k] && (Array.isArray(r.evidence[k]) ? r.evidence[k].length > 0 : true));
      if (failureKey) {
        const detail = r.evidence[failureKey];
        if (Array.isArray(detail)) {
          for (const d of detail) lines.push(`  - ${typeof d === "object" ? JSON.stringify(d) : d}`);
        } else {
          lines.push(`  - ${detail}`);
        }
      }
    }
  }
  if (retrofitUpdates.length > 0) {
    lines.push("");
    lines.push(`retrofit: updated ${retrofitUpdates.length} v2 manifest(s)`);
  }
  const summary = lines.join("\n") + "\n";
  await writeFile(path.join(outDir, "audit-report.txt"), summary, "utf8");

  console.log(summary);
  if (overall === "red") {
    process.exitCode = 2;
  } else if (overall === "yellow") {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error("audit failed:", err);
  process.exitCode = 3;
});

  // Text summary
  const lines = [];
  lines.push(`Phase 1' signature-provenance audit — ${AUDIT_SCRIPT_VERSION}`);
  lines.push(`run_at: ${report.audit_run_at}`);
  lines.push(`git_sha: ${report.git_sha ?? "(no git)"}`);
  lines.push("");
  lines.push(`overall: ${overall.toUpperCase()}`);
  lines.push("");
  for (const [name, r] of Object.entries(results)) {
    lines.push(`${name}: ${r.verdict.toUpperCase()}`);
    if (r.verdict === "fail" && r.evidence) {
      const failureKey = ["failures", "deviations", "outside_allowlist", "forbidden_overlap", "error"]
        .find((k) => r.evidence[k] && (Array.isArray(r.evidence[k]) ? r.evidence[k].length > 0 : true));
      if (failureKey) {
        const detail = r.evidence[failureKey];
        if (Array.isArray(detail)) {
          for (const d of detail) lines.push(`  - ${typeof d === "object" ? JSON.stringify(d) : d}`);
        } else {
          lines.push(`  - ${detail}`);
        }
      }
    }
  }
  if (retrofitUpdates.length > 0) {
    lines.push("");
    lines.push(`retrofit: updated ${retrofitUpdates.length} v2 manifest(s)`);
  }
  const summary = lines.join("\n") + "\n";
  await writeFile(path.join(outDir, "audit-report.txt"), summary, "utf8");

  console.log(summary);
  if (overall === "red") {
    process.exitCode = 2;
  } else if (overall === "yellow") {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error("audit failed:", err);
  process.exitCode = 3;
});
