#!/usr/bin/env node
// H-K3 lemma check: closed-form collision classes for the registered Kakeya
// direction shadow in F_q^2.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-HK3-SHADOW-COLLISION-LEMMA-CHECK";
const DEFAULT_OUT = path.join("results", "kakeya", "shadow-collision-lemma");

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
  return `Usage: node scripts/kakeya-shadow-lemma-check.mjs [options]

Options:
  --out <dir>         Output directory. Default: ${DEFAULT_OUT}
  --help              Show this message.
`;
}

function pointOrder(a, b) {
  return a - b;
}

function sortedPoints(body) {
  return [...body].sort(pointOrder);
}

function bitsKey(bits) {
  return bits.join("");
}

function oneHot(index, count) {
  return Array.from({ length: count }, (_, i) => (i === index ? 1 : 0));
}

function sameBits(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

function binom(n, k) {
  if (k < 0 || k > n) return 0n;
  let kk = BigInt(Math.min(k, n - k));
  let result = 1n;
  for (let i = 1n; i <= kk; i++) {
    result = (result * BigInt(n) - result * (i - 1n)) / i;
  }
  return result;
}

function binomSum(n, maxK) {
  let sum = 0n;
  for (let k = 0; k <= maxK; k++) sum += binom(n, k);
  return sum;
}

function decimal(value) {
  return value.toString();
}

function lineTable(q) {
  return Core.directions(q).map((dir) => ({
    dir,
    label: dir.label,
    lines: Array.from({ length: q }, (_, b) => Core.lineMask(dir, b, q)),
  }));
}

function addOutsidePrefix(line, q, take) {
  const body = new Set(line);
  for (let p = 0; p < Core.pointCount(q) && body.size < line.size + take; p++) {
    if (!line.has(p)) body.add(p);
  }
  return body;
}

function union(a, b) {
  const out = new Set(a);
  for (const p of b) out.add(p);
  return out;
}

function countBits(bits) {
  return bits.reduce((sum, bit) => sum + bit, 0);
}

function checkQ(q) {
  const dirs = Core.directions(q);
  const table = lineTable(q);
  const directionCount = Core.directionCount(q);
  const pointCount = Core.pointCount(q);
  const outsidePointCount = pointCount - q;
  const maxSafeExtra = q - 2;
  const firstBreakExtra = q - 1;
  const perLine = binomSum(outsidePointCount, maxSafeExtra);
  const perDirection = BigInt(q) * perLine;
  const allDirections = BigInt(directionCount) * perDirection;
  const linePlusOnePerDirection = BigInt(q) * binomSum(outsidePointCount, 1);

  let safeRepresentativePass = true;
  let thresholdRepresentativePass = true;
  const safeWitnesses = [];
  const thresholdWitnesses = [];

  for (let dirIndex = 0; dirIndex < table.length; dirIndex++) {
    const expectedSafeBits = oneHot(dirIndex, directionCount);
    const nextDirIndex = (dirIndex + 1) % directionCount;
    const expectedBreakBits = oneHot(dirIndex, directionCount);
    expectedBreakBits[nextDirIndex] = 1;

    for (let intercept = 0; intercept < q; intercept++) {
      const line = table[dirIndex].lines[intercept];
      const safeBody = addOutsidePrefix(line, q, maxSafeExtra);
      const safeBits = Core.shadowBitset(q, safeBody);
      const safeOk = sameBits(safeBits, expectedSafeBits);
      safeRepresentativePass = safeRepresentativePass && safeOk;
      if (!safeOk || safeWitnesses.length < 2) {
        safeWitnesses.push({
          direction: table[dirIndex].label,
          intercept,
          extraPoints: maxSafeExtra,
          bodySize: safeBody.size,
          bitset: bitsKey(safeBits),
          pass: safeOk,
          pointIndices: sortedPoints(safeBody),
        });
      }

      const crossing = table[nextDirIndex].lines[0];
      const thresholdBody = union(line, crossing);
      const thresholdBits = Core.shadowBitset(q, thresholdBody);
      const thresholdOk =
        thresholdBody.size === q + firstBreakExtra &&
        countBits(thresholdBits) >= 2 &&
        thresholdBits[dirIndex] === 1 &&
        thresholdBits[nextDirIndex] === 1;
      thresholdRepresentativePass = thresholdRepresentativePass && thresholdOk;
      if (!thresholdOk || thresholdWitnesses.length < 2) {
        thresholdWitnesses.push({
          baseDirection: table[dirIndex].label,
          baseIntercept: intercept,
          crossingDirection: table[nextDirIndex].label,
          crossingIntercept: 0,
          extraPoints: thresholdBody.size - q,
          bodySize: thresholdBody.size,
          bitset: bitsKey(thresholdBits),
          directionsCovered: countBits(thresholdBits),
          pass: thresholdOk,
          pointIndices: sortedPoints(thresholdBody),
        });
      }
    }
  }

  return {
    q,
    directionCount,
    pointCount,
    lineSize: q,
    outsidePointCount,
    maxSafeExtra,
    firstBreakExtra,
    formula: {
      perLine: `sum_{i=0}^{q-2} C(q^2 - q, i)`,
      perDirection: `q * sum_{i=0}^{q-2} C(q^2 - q, i)`,
      allDirections: `(q + 1) * q * sum_{i=0}^{q-2} C(q^2 - q, i)`,
      firstBreak: "q - 1 outside points can complete a second line",
    },
    counts: {
      perLine: decimal(perLine),
      perDirection: decimal(perDirection),
      allDirections: decimal(allDirections),
      linePlusOnePerDirection: decimal(linePlusOnePerDirection),
    },
    checks: {
      safeRepresentativePass,
      thresholdRepresentativePass,
      pass: safeRepresentativePass && thresholdRepresentativePass,
    },
    witnesses: {
      safe: safeWitnesses,
      firstBreak: thresholdWitnesses,
    },
  };
}

function csvValue(value) {
  const text = String(value ?? "");
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

function writeCsv(file, rows) {
  const header = [
    "q",
    "outside_point_count",
    "max_safe_extra",
    "first_break_extra",
    "per_line",
    "per_direction",
    "all_directions",
    "line_plus_one_per_direction",
    "safe_check",
    "threshold_check",
  ];
  const body = rows.map((row) => [
    row.q,
    row.outsidePointCount,
    row.maxSafeExtra,
    row.firstBreakExtra,
    row.counts.perLine,
    row.counts.perDirection,
    row.counts.allDirections,
    row.counts.linePlusOnePerDirection,
    row.checks.safeRepresentativePass,
    row.checks.thresholdRepresentativePass,
  ]);
  fs.writeFileSync(
    file,
    [header, ...body].map((row) => row.map(csvValue).join(",")).join("\n") + "\n",
  );
}

function writeOperatorCommands(file, command, manifestPath, countsPath) {
  fs.writeFileSync(
    file,
    `# Kakeya Shadow Collision Lemma - Operator Commands

\`\`\`powershell
${command}
\`\`\`

Primary outputs:

- \`${manifestPath}\`
- \`${countsPath}\`

This check records the finite-geometry lemma behind H-K3. It does not enumerate
all bodies and does not make a Euclidean Kakeya claim.
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
  const pass = perQ.every((row) => row.checks.pass);
  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal finite-geometry lemma receipt",
    hook: "H-K3 generalized one-direction shadow collision lemma",
    command: "node scripts/kakeya-shadow-lemma-check.mjs",
    statement:
      "For a line L in F_q^2 and any S subset F_q^2 \\ L with |S| <= q - 2, the body L union S covers exactly the direction of L. The first possible second-direction break occurs at |S| = q - 1.",
    proofSketch: [
      "A line parallel to L is disjoint from L, so it requires q outside points.",
      "A line not parallel to L intersects L in exactly one point, so it requires q - 1 outside points.",
      "Therefore q - 2 outside points cannot complete any second line.",
      "At q - 1 outside points, take a nonparallel line through a point of L; its outside points complete a second direction.",
    ],
    perQ,
    pass,
  };

  fs.mkdirSync(outDir, { recursive: true });
  const manifestPath = path.join(outDir, "manifest.json");
  const countsPath = path.join(outDir, "counts.csv");
  const commandsPath = path.join(outDir, "operator-commands.md");

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  writeCsv(countsPath, perQ);
  writeOperatorCommands(commandsPath, manifest.command, manifestPath, countsPath);

  console.log(
    [
      "KAK_SHADOW_COLLISION_LEMMA",
      `q={${Core.SUPPORTED_Q.join(",")}}`,
      `pass=${pass}`,
      ...perQ.map((row) => `q${row.q}_per_direction=${row.counts.perDirection}`),
      `out=${outDir}`,
    ].join(" "),
  );
  process.exit(pass ? 0 : 1);
}

main();
