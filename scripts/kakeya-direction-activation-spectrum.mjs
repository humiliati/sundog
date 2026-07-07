#!/usr/bin/env node
// H-K3 direction activation spectrum: for each direction and any body K, the
// exact minimal number of added points that lights that direction, plus a
// witness line. activation(d, K) = min over the q intercept lines M in
// direction d of (q - |M intersect K|). Exact: lighting d requires containing
// some full direction-d line, and adding M \ K achieves it.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3E-DIRECTION-ACTIVATION-SPECTRUM";
const DEFAULT_OUT = path.join("results", "kakeya", "direction-activation-spectrum");
const BRUTE_FORCE_Q = 5;

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const raw = argv[i];
    if (raw === "--help" || raw === "-h") {
      args.help = true;
      continue;
    }
    if (!raw.startsWith("--")) continue;
    const body = raw.slice(2);
    const eq = body.indexOf("=");
    if (eq !== -1) {
      args[body.slice(0, eq)] = body.slice(eq + 1);
      continue;
    }
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) {
      args[body] = next;
      i++;
    } else {
      args[body] = true;
    }
  }
  return args;
}

function usage() {
  return `Usage: node scripts/kakeya-direction-activation-spectrum.mjs [options]

Options:
  --out <dir>         Output directory. Default: ${DEFAULT_OUT}
  --help              Show this message.
`;
}

// --- the metric --------------------------------------------------------------

// For one direction: minimal added points over the q intercept lines, with the
// lowest-intercept argmin as witness (same tie-break as Core.witnessLine).
function directionActivation(dir, q, K) {
  let bestCost = Infinity;
  let bestIntercept = -1;
  let bestAdd = null;
  for (let b = 0; b < q; b++) {
    const line = Core.lineMask(dir, b, q);
    const add = [];
    for (const p of line) if (!K.has(p)) add.push(p);
    if (add.length < bestCost) {
      bestCost = add.length;
      bestIntercept = b;
      bestAdd = add.sort((x, y) => x - y);
    }
  }
  return { cost: bestCost, witnessIntercept: bestIntercept, addPoints: bestAdd };
}

function activationSpectrum(q, K) {
  return Core.directions(q).map((dir) => {
    const { cost, witnessIntercept, addPoints } = directionActivation(dir, q, K);
    return { direction: dir.label, lit: cost === 0, cost, witnessIntercept, addPoints };
  });
}

// --- panel bodies --------------------------------------------------------------

function starBody(q, k) {
  // k concurrent lines through the origin, directions 0..k-1, intercept 0.
  const dirs = Core.directions(q);
  const body = new Set();
  for (let i = 0; i < k; i++) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  return body;
}

function linePlusPrefix(q, extra) {
  const dirs = Core.directions(q);
  const line = Core.lineMask(dirs[0], 0, q);
  const body = new Set(line);
  for (let p = 0; p < Core.pointCount(q) && body.size < q + extra; p++) {
    if (!line.has(p)) body.add(p);
  }
  return body;
}

// expected(dirIndex) -> exact cost, "positive" (>= 1, lemma fence), or null.
function panelBodies(q) {
  const dirs = Core.directions(q);
  const n = Core.pointCount(q);
  const line = Core.lineMask(dirs[0], 0, q);
  // Intercept 1 so the crossing pivot differs from the star bodies' origin
  // pivot and the panel carries no duplicate body.
  const crossing = Core.lineMask(dirs[1], 1, q);
  const threshold = new Set(line);
  for (const p of crossing) threshold.add(p);

  const bodies = [
    { id: "empty", law: "all-q", body: Core.bEmpty(), expected: () => q },
    {
      id: "single-line",
      law: "lemma-q-minus-1",
      body: new Set(line),
      expected: (i) => (i === 0 ? 0 : q - 1),
    },
    {
      id: "line-plus-1",
      law: "uniform-q-minus-2",
      body: linePlusPrefix(q, 1),
      expected: (i) => (i === 0 ? 0 : q - 2),
    },
    {
      id: "line-plus-safe",
      law: "lemma-positive",
      body: linePlusPrefix(q, q - 2),
      expected: (i) => (i === 0 ? 0 : "positive"),
    },
    {
      id: "threshold-cross",
      law: "third-directions-q-minus-2",
      body: threshold,
      expected: (i) => (i <= 1 ? 0 : q - 2),
    },
  ];

  for (const k of [...new Set([2, 3, q - 1])]) {
    bodies.push({
      id: `star-k${k}`,
      law: "ladder-q-minus-k",
      body: starBody(q, k),
      expected: (i) => (i < k ? 0 : q - k),
    });
  }

  bodies.push({
    id: "pencil-minus-one",
    law: "star-closure-all-zero",
    body: starBody(q, q),
    expected: () => 0,
  });
  bodies.push({ id: "whole-plane", law: "all-zero", body: Core.bWholePlane(q), expected: () => 0 });
  bodies.push({
    id: "greedy-cover",
    law: "all-zero",
    body: Core.bGreedyLineCover(q),
    expected: () => 0,
  });
  bodies.push({
    id: "random-third",
    law: null,
    body: Core.bRandomSubset(q, Math.floor(n / 3), 11),
    expected: null,
  });
  bodies.push({
    id: "random-half",
    law: null,
    body: Core.bRandomSubset(q, Math.floor(n / 2), 12),
    expected: null,
  });

  return bodies;
}

// --- checks --------------------------------------------------------------------

function* combinations(pool, k, start = 0, acc = []) {
  if (acc.length === k) {
    yield acc;
    return;
  }
  for (let i = start; i <= pool.length - (k - acc.length); i++) {
    yield* combinations(pool, k, i + 1, [...acc, pool[i]]);
  }
}

// Independent brute force (q = 5 only): no (cost - 1)-point addition lights a
// direction whose claimed activation is cost. Uses only Core.shadowBitset.
function bruteForceMinimality(q, K, spectrum) {
  const complement = [];
  for (let p = 0; p < Core.pointCount(q); p++) if (!K.has(p)) complement.push(p);
  const byCost = new Map();
  for (let i = 0; i < spectrum.length; i++) {
    const { cost } = spectrum[i];
    if (cost >= 1) byCost.set(cost, [...(byCost.get(cost) ?? []), i]);
  }
  let subsetsChecked = 0;
  for (const [cost, dirIndices] of byCost) {
    for (const subset of combinations(complement, cost - 1)) {
      subsetsChecked++;
      const enlarged = new Set(K);
      for (const p of subset) enlarged.add(p);
      const bits = Core.shadowBitset(q, enlarged);
      for (const i of dirIndices) {
        if (bits[i] === 1) return { pass: false, subsetsChecked };
      }
    }
  }
  return { pass: true, subsetsChecked };
}

function checkBody(q, entry) {
  const spectrum = activationSpectrum(q, entry.body);
  const bits = Core.shadowBitset(q, entry.body);

  // Coherence: cost 0 iff the registered shadow already has the bit.
  let coherence = true;
  for (let i = 0; i < spectrum.length; i++) {
    coherence = coherence && (spectrum[i].cost === 0) === (bits[i] === 1);
  }

  // Constructive witness: adding the witness points lights the direction.
  let witnessOk = true;
  for (let i = 0; i < spectrum.length; i++) {
    const enlarged = new Set(entry.body);
    for (const p of spectrum[i].addPoints) enlarged.add(p);
    witnessOk =
      witnessOk &&
      spectrum[i].addPoints.length === spectrum[i].cost &&
      Core.shadowBitset(q, enlarged)[i] === 1;
  }

  // Anchor law, where the panel declares one.
  let anchorOk = true;
  if (entry.expected) {
    for (let i = 0; i < spectrum.length; i++) {
      const want = entry.expected(i);
      if (want === "positive") anchorOk = anchorOk && spectrum[i].cost >= 1;
      else anchorOk = anchorOk && spectrum[i].cost === want;
    }
  }

  const brute = q === BRUTE_FORCE_Q ? bruteForceMinimality(q, entry.body, spectrum) : null;

  return {
    id: entry.id,
    law: entry.law,
    bodySize: entry.body.size,
    spectrum,
    checks: {
      coherence,
      witness: witnessOk,
      anchor: anchorOk,
      bruteForce: brute ? brute.pass : null,
      bruteForceSubsets: brute ? brute.subsetsChecked : null,
      pass: coherence && witnessOk && anchorOk && (brute === null || brute.pass),
    },
  };
}

function checkQ(q) {
  const results = panelBodies(q).map((entry) => checkBody(q, entry));
  return {
    q,
    directionCount: Core.directionCount(q),
    pointCount: Core.pointCount(q),
    dvirFloor: Core.dvirFloor(q),
    bruteForce: q === BRUTE_FORCE_Q,
    bodies: results,
    pass: results.every((r) => r.checks.pass),
  };
}

// --- outputs --------------------------------------------------------------------

function csvValue(value) {
  const text = String(value ?? "");
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

function writeCsv(file, perQ) {
  const header = [
    "q",
    "body",
    "body_size",
    "direction",
    "lit",
    "activation_cost",
    "witness_intercept",
    "add_points",
  ];
  const rows = [];
  for (const block of perQ) {
    for (const body of block.bodies) {
      for (const s of body.spectrum) {
        rows.push([
          block.q,
          body.id,
          body.bodySize,
          s.direction,
          s.lit,
          s.cost,
          s.witnessIntercept,
          s.addPoints.join(" "),
        ]);
      }
    }
  }
  fs.writeFileSync(
    file,
    [header, ...rows].map((row) => row.map(csvValue).join(",")).join("\n") + "\n",
  );
}

function writeOperatorCommands(file, command, manifestPath, csvPath) {
  fs.writeFileSync(
    file,
    `# Kakeya Direction Activation Spectrum - Operator Commands

\`\`\`powershell
${command}
\`\`\`

Primary outputs:

- \`${manifestPath}\`
- \`${csvPath}\`

This is a report-only workbench diagnostic. The spectrum and its witness
fields stay in results/; the public shadow export is untouched. No Euclidean
Kakeya claim.
`,
  );
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const outDir = String(args.out ?? DEFAULT_OUT);
  const perQ = Core.SUPPORTED_Q.map(checkQ);
  const pass = perQ.every((block) => block.pass);
  const anchors = perQ.every((block) => block.bodies.every((b) => b.checks.anchor));
  const witness = perQ.every((block) => block.bodies.every((b) => b.checks.witness));
  const bruteQ5 = perQ
    .filter((block) => block.bruteForce)
    .every((block) => block.bodies.every((b) => b.checks.bruteForce === true));

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal workbench diagnostic receipt",
    hook: "H-K3 direction activation spectrum (exact distance-to-information per direction)",
    command: "node scripts/kakeya-direction-activation-spectrum.mjs",
    statement:
      "For any body K in F_q^2 and any direction d, the minimal number of added points that lights d in the registered shadow equals min over the q intercept lines M in direction d of (q - |M intersect K|), witnessed by the argmin line. Lower bound: any enlarged body lighting d contains a full direction-d line M, which costs q - |M intersect K| >= the min. Achievability: add the witness line's missing points.",
    anchorLaws: [
      "empty body: every direction costs q",
      "bare line: own direction 0, every other direction exactly q - 1 (PHASE3D lemma sharpness restated as a metric)",
      "line plus one outside point: every other direction exactly q - 2 (the through-that-point line reuses it plus one line crossing)",
      "threshold cross (L union M): both lit, every third direction exactly q - 2",
      "k concurrent lines, k <= q - 1: every missing direction exactly q - k (marginal-cost ladder rungs achieved)",
      "k = q star (pencil minus one direction): spectrum all-zero - the missing direction's q - 1 off-pivot lines are already inside; a complete Kakeya set of size q^2 - q + 1",
    ],
    boundary:
      "Per-direction marginal costs only; they do not sum to a joint lighting cost (directions share added points). Report-only, workbench-internal; witness fields never enter the public shadow export; no Euclidean or extremal claim.",
    falsifier: {
      name: "ACTIVATION_SPECTRUM_MISMATCH",
      description:
        "Fires if the line-min formula disagrees with the registered shadow (coherence), a witness addition fails to light its direction, an anchor law fails, or the q=5 exhaustive search finds a smaller addition that lights a direction.",
      status:
        pass && anchors && witness && bruteQ5 ? "clear" : "fired",
    },
    perQ,
    pass,
  };

  fs.mkdirSync(outDir, { recursive: true });
  const manifestPath = path.join(outDir, "manifest.json");
  const csvPath = path.join(outDir, "spectrum.csv");
  const commandsPath = path.join(outDir, "operator-commands.md");

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  writeCsv(csvPath, perQ);
  writeOperatorCommands(commandsPath, manifest.command, manifestPath, csvPath);

  console.log(
    [
      "KAK_DIRECTION_ACTIVATION_SPECTRUM",
      `q={${Core.SUPPORTED_Q.join(",")}}`,
      `bodies=${perQ.reduce((sum, block) => sum + block.bodies.length, 0)}`,
      `anchors=${anchors ? "pass" : "fail"}`,
      `witness=${witness ? "pass" : "fail"}`,
      `bruteforce_q5=${bruteQ5 ? "pass" : "fail"}`,
      `falsifier=${manifest.falsifier.status}`,
      `out=${outDir}`,
    ].join(" "),
  );
  process.exit(pass ? 0 : 1);
}

main();
