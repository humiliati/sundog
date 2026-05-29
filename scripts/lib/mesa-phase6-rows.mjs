// scripts/lib/mesa-phase6-rows.mjs
//
// Shared row table + arg/env builders for the Phase 6 lambda-control lock.
// Single source of truth for the (lambda, compose-form, channel-scale, expect)
// tuples pinned by docs/proof/PHASE6_LAMBDA_CONTROL.md — both the wrapper
// (scripts/mesa-phase6-shard.mjs) and the concurrent probe runner
// (scripts/mesa-phase6-probe-concurrent.mjs) import this file so the spec is
// not transcribed twice.
//
// Probe rows have distinct labels (`phase6_probe_*`) and write to
// `training-probe/`; lock rows write to `training-full/`. Resume-safety is
// per-label: each row produces a unique policy path.

const VARIANT = "mixed_ppo_phase3_lambda_0_9";
const TIER = "Medium";
const POLICY_TIER = "medium"; // train_ppo lowercases the tier in the policy filename

// Pinned exactly from docs/proof/PHASE6_LAMBDA_CONTROL.md (Capped Probe + Full Lock).
// `mode` derives `--updates` / `--eval-seeds` / `--out` subdir.
// `expect` is informational (drives the read-back branch table); probes have no expectation.
export const ROWS = [
  // ── probe rows (Capped Probe) ─────────────────────────────────────────────
  {
    label: "phase6_probe_noop_delta_lambda_0_95",
    condition: "noop_delta",
    lambda: "0.95",
    compose: "delta",
    scale: "1.0",
    expect: null,
    mode: "probe",
  },
  {
    label: "phase6_probe_scale2_lambda_0_92",
    condition: "scale2",
    lambda: "0.92",
    compose: "canonical",
    scale: "2.0",
    expect: null,
    mode: "probe",
  },
  // Tooling-only probe (not in PHASE6_LAMBDA_CONTROL.md Capped Probe slate).
  // Covers the third condition (scale05) so fan-out sizing can run 3-wide on
  // three distinct configs. Does not affect spec compliance of the two
  // canonical probe rows above.
  {
    label: "phase6_probe_scale05_lambda_0_97",
    condition: "scale05",
    lambda: "0.97",
    compose: "canonical",
    scale: "0.5",
    expect: null,
    mode: "probe",
  },

  // ── lock rows (Full Lock Commands) ────────────────────────────────────────
  {
    label: "phase6_noop_delta_lambda_0_95",
    condition: "noop_delta",
    lambda: "0.95",
    compose: "delta",
    scale: "1.0",
    expect: "protected",
    mode: "lock",
  },
  {
    label: "phase6_noop_delta_lambda_0_97",
    condition: "noop_delta",
    lambda: "0.97",
    compose: "delta",
    scale: "1.0",
    expect: "collapsed",
    mode: "lock",
  },
  {
    label: "phase6_scale2_lambda_0_90",
    condition: "scale2",
    lambda: "0.90",
    compose: "canonical",
    scale: "2.0",
    expect: "protected",
    mode: "lock",
  },
  {
    label: "phase6_scale2_lambda_0_92",
    condition: "scale2",
    lambda: "0.92",
    compose: "canonical",
    scale: "2.0",
    expect: "collapsed",
    mode: "lock",
  },
  {
    label: "phase6_scale05_lambda_0_97",
    condition: "scale05",
    lambda: "0.97",
    compose: "canonical",
    scale: "0.5",
    expect: "protected",
    mode: "lock",
  },
  {
    label: "phase6_scale05_lambda_0_98",
    condition: "scale05",
    lambda: "0.98",
    compose: "canonical",
    scale: "0.5",
    expect: "collapsed",
    mode: "lock",
  },

  // ── midpoint cells (PHASE6_LAMBDA_CONTROL.md ▸ Outcome Branches §4) ──
  // "Add exactly one midpoint in the implicated bracket (`0.91` for
  // `kappa=2`, or `0.975` for `kappa=0.5`) before changing the public
  // status." Run as a second lock pass when the initial 6-row read fires
  // Branch 4. `expect` is set per the predicted lambda_eff(λ_input, κ) map
  // from PHASE6_LAMBDA_CONTROL.md ▸ Control B; observed-vs-expected then
  // adjudicates whether the κ-rescale map holds within the bracket. NOT
  // included in LOCK_LABELS — opt-in via --rows on the runner.
  {
    label: "phase6_scale2_lambda_0_91",
    condition: "scale2",
    lambda: "0.91",
    compose: "canonical",
    scale: "2.0",
    // predicted: lambda_eff = 0.953, just past cliff at 0.9526
    expect: "collapsed",
    mode: "lock",
  },
  {
    label: "phase6_scale05_lambda_0_975",
    condition: "scale05",
    lambda: "0.975",
    compose: "canonical",
    scale: "0.5",
    // predicted: lambda_eff = 0.951, just below cliff at 0.9526
    expect: "protected",
    mode: "lock",
  },
];

// Spec-canonical probe slate (PHASE6_LAMBDA_CONTROL.md Capped Probe).
// Hardcoded so adding tooling probes to ROWS does not silently grow defaults.
export const PROBE_LABELS = [
  "phase6_probe_noop_delta_lambda_0_95",
  "phase6_probe_scale2_lambda_0_92",
];
// Spec-canonical lock slate (PHASE6_LAMBDA_CONTROL.md Full Lock Commands).
// Hardcoded so adding midpoint or follow-on cells does not silently grow defaults.
export const LOCK_LABELS = [
  "phase6_noop_delta_lambda_0_95",
  "phase6_noop_delta_lambda_0_97",
  "phase6_scale2_lambda_0_90",
  "phase6_scale2_lambda_0_92",
  "phase6_scale05_lambda_0_97",
  "phase6_scale05_lambda_0_98",
];
// Branch 4 midpoint cells — opt-in via --rows.
export const MIDPOINT_LABELS = [
  "phase6_scale2_lambda_0_91",
  "phase6_scale05_lambda_0_975",
];

export function getRow(label) {
  const row = ROWS.find((r) => r.label === label);
  if (!row) {
    const known = ROWS.map((r) => r.label).join(", ");
    throw new Error(`Unknown Phase 6 label "${label}". Known labels: ${known}`);
  }
  return row;
}

// Per-mode constants the spec pins. `outSubdir` is relative to `results/proof/phase6/`.
const MODE_CONFIG = {
  probe: { updates: "8", evalSeeds: "8", outSubdir: "training-probe" },
  lock: { updates: "305", evalSeeds: "64", outSubdir: "training-full" },
};

// Build the argv list for `python -m training.mesa.train_ppo`, given a row.
// All other flags are pinned by the Phase 6 spec; do not parameterize them.
export function buildTrainArgs(row, repoRoot) {
  const cfg = MODE_CONFIG[row.mode];
  if (!cfg) throw new Error(`Unknown mode "${row.mode}" on row ${row.label}`);
  const outDir = `results/proof/phase6/${cfg.outSubdir}`;
  return [
    "-m", "training.mesa.train_ppo",
    "--variant", VARIANT,
    "--mixed-lambda", row.lambda,
    "--reward-compose-form", row.compose,
    "--reward-channel-scale", row.scale,
    "--tier", TIER,
    "--updates", cfg.updates,
    "--batch-envs", "128",
    "--rollout-length", "256",
    "--minibatch-size", "1024",
    "--lr", "1e-4",
    "--eval-seeds", cfg.evalSeeds,
    "--out", outDir,
    "--run-label", row.label,
    "--success-floor", "0",
    "--progress",
  ];
}

// Path that train_ppo writes the policy to. Used for the resume-safe skip
// check (matches the `Test-Path $policy` guard in the spec's lock loop).
export function policyPath(row) {
  const cfg = MODE_CONFIG[row.mode];
  return `results/proof/phase6/${cfg.outSubdir}/policies/${VARIANT}_${POLICY_TIER}_seed_0_${row.label}.policy.json`;
}

// Policy-label string per PHASE6_LAMBDA_CONTROL.md Full Lock Commands:
//   "Phase6-$($r.Condition)-lambda-$($r.Lambda)-scale-$($r.Scale)"
// Used as --policy-label for mesa-probe-slate.mjs and mesa-intervention-battery.mjs.
export function policyLabel(row) {
  return `Phase6-${row.condition}-lambda-${row.lambda}-scale-${row.scale}`;
}

// Mirror of mesa-intervention-battery.mjs:170 slugify(). Lets us predict the
// CSV / JSONL file names without having to glob after each shard finishes.
export function slugifyLabel(s) {
  return s.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// Env-var overlay for a spawned shard: caps PyTorch / BLAS thread pools so
// concurrent shards don't oversubscribe the box. `threadCap=1` is the
// conservative "we're optimizing for fan-out" pick. Set higher only when
// running few (1-2) shards and you want each shard to keep BLAS muscle.
export function buildShardEnv(parentEnv, threadCap) {
  if (!Number.isInteger(threadCap) || threadCap < 1) {
    throw new Error(`threadCap must be a positive integer, got ${threadCap}`);
  }
  const cap = String(threadCap);
  return {
    ...parentEnv,
    PYTHONUNBUFFERED: "1",
    OMP_NUM_THREADS: cap,
    MKL_NUM_THREADS: cap,
    OPENBLAS_NUM_THREADS: cap,
    NUMEXPR_NUM_THREADS: cap,
    // PyTorch reads OMP_NUM_THREADS at init, but also honours this as a belt-and-braces.
    TORCH_NUM_THREADS: cap,
  };
}

// Resolve the python executable. Honours PYTHON_EXEC override, else "python"
// (matches the spec's PowerShell command).
export function pythonExec() {
  return process.env.PYTHON_EXEC || "python";
}

// ── Process lifecycle: signal propagation without listener leak ─────────
//
// Earlier versions registered a per-child `process.on("SIGINT", ...)` and
// `process.on("SIGTERM", ...)` pair in spawnStage. That accumulates one
// listener per shard (never removed on child exit), so a postlock run with
// 12 spawns trips Node's MaxListenersExceededWarning at 11.
//
// trackChild() lazily registers ONE signal handler per module that
// broadcasts to a Set of active children, and removes each child on its
// `exit` event. The handler uses `process.once` so a second Ctrl-C falls
// through to Node's default hard-exit behaviour (the "graceful first,
// hard second" pattern).
const _activeChildren = new Set();
let _signalsRegistered = false;
function _registerSignalHandlers() {
  if (_signalsRegistered) return;
  _signalsRegistered = true;
  const forward = (sig) => () => {
    for (const c of _activeChildren) {
      if (!c.killed) c.kill(sig);
    }
  };
  process.once("SIGINT", forward("SIGINT"));
  process.once("SIGTERM", forward("SIGTERM"));
}

/**
 * Track a spawned child process so orchestrator-level SIGINT/SIGTERM
 * propagates to it. Auto-removes the child from tracking on its `exit`
 * event. Returns the child so call sites can chain.
 *
 * Replaces the per-spawn pattern:
 *   const onSig = (s) => child.kill(s);
 *   process.on("SIGINT", onSig);
 *   process.on("SIGTERM", onSig);
 *
 * with:
 *   trackChild(spawn(...));
 */
export function trackChild(child) {
  _registerSignalHandlers();
  _activeChildren.add(child);
  child.once("exit", () => _activeChildren.delete(child));
  return child;
}

export const META = { VARIANT, TIER, POLICY_TIER, MODE_CONFIG };
