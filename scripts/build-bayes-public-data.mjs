import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();

const sourcePaths = Object.freeze({
  phase1: "results/bayes/phase1-reference-lock/manifest.json",
  phase2: "results/bayes/phase2-mismatch-lock/manifest.json",
  phase3: "results/bayes/phase3-aliasing-lock/manifest.json",
  phase4b: "results/bayes/phase4b-hybrid-lock/manifest.json",
  phase5: "results/bayes/phase5-envelope-lock/manifest.json",
  phase5b: "results/bayes/phase5b-mismatch-severity-lock/manifest.json",
  phase6: "results/bayes/phase6-photometric-lock/manifest.json",
  phase6b: "results/bayes/phase6b-structure-lock/manifest.json",
});

const outputPath = "public/data/bayes-comparison.json";

const verdict = Object.freeze({
  sundogReadout:
    "Narrow separation only under specific synthetic misspecification: Phase 2 anisotropic, Phase 3 decoy/alias failure boundary, and Phase 5b clean-only model-family collapse. Hybrid did not earn a niche.",
  comparatorState:
    "Correctly specified adaptive or particle Bayes dominates the synthetic envelope and the core photometric task. On the core task it stays faster at comparable terminal accuracy under parameter stress and non-degenerate structural misspecification; response control flips only against degenerate model-less Bayes.",
  safeInterpretation:
    "Real but narrow and substrate-conditional: stronger than a naive-only comparator result, weaker than posterior dominance. The response edge does not transfer to the core task except degenerately.",
  stampSequence: [
    "MODEL KNOWN -> BAYES WINS",
    "SYNTHETIC MISSPECIFICATION -> NARROW RESPONSE EDGE",
    "CORE TASK -> NO TRANSFER",
  ],
});

async function readJson(relativePath) {
  return JSON.parse(await readFile(join(root, relativePath), "utf8"));
}

async function readSources() {
  const missing = [];
  const manifests = {};
  for (const [key, relativePath] of Object.entries(sourcePaths)) {
    try {
      manifests[key] = await readJson(relativePath);
    } catch (error) {
      if (error?.code === "ENOENT") {
        missing.push(relativePath);
      } else {
        throw error;
      }
    }
  }
  if (missing.length) {
    console.warn(
      `bayes public data skipped; missing source manifest(s): ${missing.join(", ")}`,
    );
    return null;
  }
  return manifests;
}

function round(value, digits = 6) {
  return Number.isFinite(value) ? Number.parseFloat(value.toFixed(digits)) : value;
}

function completionIso(manifest) {
  if (typeof manifest.completedAt === "number") {
    return new Date(manifest.completedAt * 1000).toISOString();
  }
  return manifest.completedAt ?? manifest.generatedAt ?? null;
}

function sourceGeneratedAt(manifests) {
  const stamps = Object.values(manifests)
    .map(completionIso)
    .filter(Boolean)
    .sort();
  return stamps.at(-1) ?? "source-manifests";
}

function statusOf(manifest) {
  return manifest.status ?? manifest.exitGate?.status ?? (
    manifest.exitGate?.pass === true ? "pass" : null
  );
}

function anchorPassOf(manifest) {
  if (typeof manifest.anchor?.pass === "boolean") return manifest.anchor.pass;
  if (typeof manifest.exitGate?.selfConsistencyAnchor?.pass === "boolean") {
    return manifest.exitGate.selfConsistencyAnchor.pass;
  }
  if (typeof manifest.phase5Envelope?.selfConsistencyAnchor?.pass === "boolean") {
    return manifest.phase5Envelope.selfConsistencyAnchor.pass;
  }
  return null;
}

function phaseRecord(key, manifest, headline) {
  return {
    phase: manifest.phase,
    status: statusOf(manifest),
    sourcePath: sourcePaths[key],
    completedAt: completionIso(manifest),
    anchorPass: anchorPassOf(manifest),
    headline,
  };
}

function gateScenario(exitGate, scenario) {
  return (exitGate?.scenarioGates ?? []).find((row) => row.scenario === scenario) ?? {};
}

function classCountsObject(rows) {
  if (!rows) return {};
  if (Array.isArray(rows)) {
    return Object.fromEntries(rows.map((row) => [row.classLabel ?? row.class, row.cells]));
  }
  return rows;
}

function phase5bSeverityRows(manifest) {
  const rows = manifest.phase5Envelope?.envelopeRows ?? [];
  return Object.fromEntries(
    rows.map((row) => [
      `${row.severity}_${row.decoyStrength}`,
      {
        severity: row.severity,
        decoyStrength: row.decoyStrength,
        classLabel: row.classLabel,
        bestFamily: row.bestFamily,
        bestMode: row.bestMode,
        bayesMinusResponseScoreDelta: row.bayesMinusResponseScoreDelta,
      },
    ]),
  );
}

function phase6Class(manifest, cellId) {
  return (manifest.cellClasses ?? []).find((row) => row.cell_id === cellId) ?? {};
}

function phase6Summary(manifest, cellId, condition) {
  return manifest.summaries?.[cellId]?.[condition] ?? {};
}

function compactPhase6Cell(manifest, cellId) {
  const row = phase6Class(manifest, cellId);
  const bayes = phase6Summary(manifest, cellId, "bayes_particle");
  return {
    cellId,
    class: row.class,
    bestCondition: row.best_condition,
    runnerUp: row.runner_up,
    leadVsRunnerUp: row.lead_vs_runner_up,
    ci95HalfWidth: row.ci95_half_width,
    bayesMedianTtt: bayes.time_to_threshold?.median,
    bayesFailedSeeds: bayes.time_to_threshold?.n_failed,
  };
}

function buildData(manifests) {
  const p1 = manifests.phase1.exitGate;
  const p2 = manifests.phase2.exitGate;
  const p3 = manifests.phase3.exitGate;
  const p4b = manifests.phase4b.exitGate;
  const p5 = manifests.phase5.exitGate;
  const p5b = manifests.phase5b.exitGate;
  const p6 = manifests.phase6;
  const p6b = manifests.phase6b;

  return {
    schemaVersion: 1,
    generatedAt: sourceGeneratedAt(manifests),
    sourcePaths,
    phases: [
      phaseRecord("phase1", manifests.phase1, {
        kind: p1.kind,
        bayesSuccessRate: p1.bayesSuccessRate,
        sundogSuccessRate: p1.sundogSuccessRate,
        bayesMinusSundogMeanScoreDelta: p1.bayesMinusSundogMeanScoreDelta,
      }),
      phaseRecord("phase2", manifests.phase2, {
        kind: p2.kind,
        separatedScenarios: p2.separatedScenarios,
        separationMargin: p2.separationMargin,
        anisotropicScoreDelta: gateScenario(p2, "anisotropic").responseMinusBayesScoreDelta,
      }),
      phaseRecord("phase3", manifests.phase3, {
        kind: p3.kind,
        dualGateScenarios: p3.dualGateScenarios,
        decoyWrongLockRate: gateScenario(p3, "decoy").hcSundogWrongLockRate,
        aliasWrongLockRate: gateScenario(p3, "alias").hcSundogWrongLockRate,
        decoyBayesRecoveryScoreDelta: gateScenario(p3, "decoy").bayesMinusBestSundogScoreDelta,
        aliasBayesRecoveryScoreDelta: gateScenario(p3, "alias").bayesMinusBestSundogScoreDelta,
      }),
      phaseRecord("phase4b", manifests.phase4b, {
        kind: p4b.kind,
        nicheConfirmed: p4b.nicheConfirmed,
        status: p4b.status,
        claimScenarios: p4b.claimScenarios,
        allArmsClaimScenarios: (p4b.scenarioGates ?? [])
          .filter((row) => row.claimScenario && row.allArms)
          .map((row) => row.scenario),
        posteriorSufficiencyClaimScenarios: (p4b.scenarioGates ?? [])
          .filter((row) => row.claimScenario && row.loadBearingArm)
          .map((row) => row.scenario),
      }),
      phaseRecord("phase5", manifests.phase5, {
        kind: p5.kind,
        cells: p5.cells,
        classifiedCells: p5.classifiedCells,
        classCounts: classCountsObject(p5.classCounts),
        bestMode: "bayes_adaptive",
      }),
      phaseRecord("phase5b", manifests.phase5b, {
        kind: p5b.kind,
        cells: p5b.cells,
        classifiedCells: p5b.classifiedCells,
        classCounts: classCountsObject(p5b.classCounts),
        severityRows: phase5bSeverityRows(manifests.phase5b),
      }),
      phaseRecord("phase6", manifests.phase6, {
        cells: p6.cellClasses.length,
        classCounts: p6.classCounts,
        nominal: compactPhase6Cell(p6, "nominal"),
        maxBeam: compactPhase6Cell(p6, "beam_sigma_0p40"),
        maxDetectorNoise: compactPhase6Cell(p6, "detector_noise_0p20"),
        mixedCell: compactPhase6Cell(p6, "beam_sigma_0p05"),
      }),
      phaseRecord("phase6b", manifests.phase6b, {
        cells: p6b.cellClasses.length,
        classCounts: p6b.classCounts,
        exitDeterminant: p6b.phase6bExitDeterminant,
        s0: compactPhase6Cell(p6b, "nominal_structure_s0"),
        s1: compactPhase6Cell(p6b, "nominal_structure_s1"),
        s2: compactPhase6Cell(p6b, "nominal_structure_s2"),
        s3: compactPhase6Cell(p6b, "nominal_structure_s3"),
      }),
    ],
    verdict,
  };
}

async function main() {
  const manifests = await readSources();
  if (!manifests) return;

  const data = buildData(manifests);
  const absoluteOutput = join(root, outputPath);
  await mkdir(dirname(absoluteOutput), { recursive: true });
  await writeFile(absoluteOutput, `${JSON.stringify(data, null, 2)}\n`, "utf8");
  console.log(`bayes public data built: ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
