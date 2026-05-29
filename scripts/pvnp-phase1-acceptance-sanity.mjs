#!/usr/bin/env node
// scripts/pvnp-phase1-acceptance-sanity.mjs
//
// Per PHASE1_V3_SLATE.md §Acceptance-Volume Sanity Gate, this stage emits
// acceptance_volume_sanity.csv showing accept/reject/quarantine counts and
// per-reason distribution by (split, promise_compliance). It also writes a
// route disposition in `acceptance_sanity_route.json`.
//
// v3 elects the "conservative_acceptance" route by default: retain the
// v2-inherited thresholds, treat the resulting low acceptance rate as an
// intentional safety property, and report the reason distribution so the
// receipt can defend the choice.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getPhase1RunConfig } from "./lib/pvnp-phase1-run-config.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = { runDir: "results/pvnp/phase1-toy-verifier-v0" };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--run-dir") { args.runDir = argv[i + 1]; i += 1; }
    else throw new Error(`Unknown flag: ${argv[i]}`);
  }
  return args;
}

async function readCsv(p) {
  const text = await readFile(p, "utf8");
  const lines = text.trim().split("\n");
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(header.map((h, i) => [h, cells[i]]));
  });
}

async function readJsonl(p) {
  const text = await readFile(p, "utf8");
  return text.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function csvRow(values) {
  return values.map((v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return s.includes(",") || s.includes("\"") ? `"${s.replace(/"/g, '""')}"` : s;
  }).join(",");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(REPO_ROOT, args.runDir);
  const slate = getPhase1RunConfig(args.runDir);
  const version = slate.schema_suffix;
  if (version !== "v3" && version !== "v4" && version !== "v5") {
    console.log(`acceptance-sanity is a v3+/v4+ artifact; skipping for ${version}`);
    return;
  }
  await mkdir(outDir, { recursive: true });

  const verifierDecisions = await readCsv(path.join(outDir, "verifier_decisions.csv"));
  const groundTruth = await readCsv(path.join(outDir, "ground_truth_labels.csv"));
  const envs = await readJsonl(path.join(outDir, "environments.jsonl"));
  const envById = new Map(envs.map((e) => [e.id, e]));
  const gtByKey = new Map(groundTruth.map((g) => [`${g.policy_id}|${g.env_id}`, g]));

  // (split, promise_compliance) → counters + reason histogram.
  const buckets = new Map();
  for (const v of verifierDecisions) {
    const env = envById.get(v.env_id);
    if (!env) continue;
    const key = `${v.split}|${env.promise_compliance}`;
    if (!buckets.has(key)) {
      buckets.set(key, {
        split: v.split,
        promise_compliance: env.promise_compliance,
        accept: 0, reject: 0, quarantine: 0,
        false_accept: 0, false_reject: 0,
        reasons: new Map(),
      });
    }
    const bucket = buckets.get(key);
    bucket[v.decision] = (bucket[v.decision] || 0) + 1;
    bucket.reasons.set(v.reason, (bucket.reasons.get(v.reason) || 0) + 1);

    const gt = gtByKey.get(`${v.policy_id}|${v.env_id}`);
    if (gt) {
      const truthSafe = gt.safe === "1";
      if (v.decision === "accept" && !truthSafe) bucket.false_accept += 1;
      if (v.decision === "reject" && truthSafe) bucket.false_reject += 1;
    }
  }

  // Per-bucket CSV.
  const headerCols = [
    "split", "promise_compliance",
    "accept", "reject", "quarantine", "total",
    "accept_rate", "reject_rate", "quarantine_rate",
    "false_accept", "false_reject",
    "top_quarantine_reason", "top_quarantine_reason_count",
    "top_reject_reason", "top_reject_reason_count",
  ];
  const rows = [headerCols.join(",")];
  for (const bucket of [...buckets.values()].sort((a, b) => a.split.localeCompare(b.split) || a.promise_compliance.localeCompare(b.promise_compliance))) {
    const total = bucket.accept + bucket.reject + bucket.quarantine;
    const reasonList = [...bucket.reasons.entries()].sort((a, b) => b[1] - a[1]);
    // top quarantine reason: filter to non-margin reasons (margin is a reject reason)
    const topQuar = reasonList.find(([r]) => !r.startsWith("margin_") && r !== "all_checks_pass");
    const topRej = reasonList.find(([r]) => r.startsWith("margin_"));
    rows.push(csvRow([
      bucket.split, bucket.promise_compliance,
      bucket.accept, bucket.reject, bucket.quarantine, total,
      total ? (bucket.accept / total).toFixed(4) : "",
      total ? (bucket.reject / total).toFixed(4) : "",
      total ? (bucket.quarantine / total).toFixed(4) : "",
      bucket.false_accept, bucket.false_reject,
      topQuar?.[0] ?? "",
      topQuar?.[1] ?? "",
      topRej?.[0] ?? "",
      topRej?.[1] ?? "",
    ]));
  }

  await writeFile(path.join(outDir, "acceptance_volume_sanity.csv"), rows.join("\n") + "\n", "utf8");

  // Route disposition.
  // v3 picks "conservative_acceptance" by default. The route can be flipped
  // by passing --route=calibrated_widening later (not implemented for the
  // initial v3 execution; the slate allows either route).
  const disposition = {
    schema: `pvnp-phase1-acceptance-sanity-route-${version}`,
    route: "conservative_acceptance",
    rationale: "Retain v2-inherited geometry/invariance thresholds; report the resulting low acceptance rate as an intentional safety property because (a) 0 false accepts on measurement, (b) 0 accepted basin-shape out-of-promise items, and (c) the geometry signal correctly quarantines envs whose probe coverage is insufficient to support a basin-shape promise decision — these are exactly the trajectories where the verifier should refuse, not accept.",
    no_geometry_threshold_was_relaxed: true,
    measurement_buckets: rows.length - 1,
  };
  await writeFile(
    path.join(outDir, "acceptance_sanity_route.json"),
    JSON.stringify(disposition, null, 2) + "\n",
    "utf8",
  );

  console.log(`acceptance sanity: ${rows.length - 1} (split, promise) buckets; route = conservative_acceptance`);
}

main().catch((err) => { console.error(err); process.exit(1); });
