#!/usr/bin/env node
// H-K3 audit: measure how lossy the registered finite-field Kakeya direction
// shadow is without exposing point membership as the primary signature.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-HK3-SHADOW-COLLISION-AUDIT";
const DEFAULT_OUT = path.join("results", "kakeya", "shadow-collision-audit");

function defaultOutFor(q) {
  return q === 5 ? DEFAULT_OUT : `${DEFAULT_OUT}-q${q}`;
}

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
  return `Usage: node scripts/kakeya-shadow-collision-audit.mjs [options]

Options:
  --q <n>             Field size to enumerate. Default: 5.
  --max-size <n>      Enumerate bodies with size <= n. Default: 6.
  --max-states <n>    Stop enumeration after this many bodies. Default: 300000.
  --out <dir>         Output directory. Default: ${DEFAULT_OUT} for q=5,
                      otherwise ${DEFAULT_OUT}-q<n>.
  --help              Show this message.
`;
}

function readInt(value, name) {
  const n = Number.parseInt(String(value), 10);
  if (!Number.isFinite(n)) throw new Error(`Invalid integer for ${name}: ${value}`);
  return n;
}

function pointOrder(a, b) {
  return a - b;
}

function sortedPoints(body) {
  return [...body].sort(pointOrder);
}

function bodyKey(body) {
  return sortedPoints(body).join(",");
}

function sameBody(a, b) {
  return a.size === b.size && [...a].every((p) => b.has(p));
}

function bitsKey(bits) {
  return bits.join("");
}

function lineTable(q) {
  return Core.directions(q).map((dir) => ({
    label: dir.label,
    lines: Array.from({ length: q }, (_, b) => sortedPoints(Core.lineMask(dir, b, q))),
  }));
}

function lineContained(line, body) {
  for (const p of line) if (!body.has(p)) return false;
  return true;
}

function shadowBitsFromTable(table, body) {
  return table.map((dir) => (dir.lines.some((line) => lineContained(line, body)) ? 1 : 0));
}

function directionLabelsFromBits(table, bits, keep) {
  return table.filter((_, i) => Boolean(bits[i]) === keep).map((dir) => dir.label);
}

function shadowSummary(q, table, body) {
  const bits = shadowBitsFromTable(table, body);
  const directionsCovered = bits.reduce((sum, bit) => sum + bit, 0);
  const directionCount = table.length;
  const complete = directionsCovered === directionCount;
  return {
    q,
    bodySize: body.size,
    bodyFraction: body.size / Core.pointCount(q),
    directionsCovered,
    directionCount,
    coverageFraction: directionsCovered / directionCount,
    covered: directionLabelsFromBits(table, bits, true),
    missing: directionLabelsFromBits(table, bits, false),
    complete,
    dvirFloor: Core.dvirFloor(q),
    dvirFloorConsistent: complete ? body.size >= Core.dvirFloor(q) : null,
    bitset: bitsKey(bits),
    bits,
  };
}

function bodyDescriptor(q, table, name, body, source) {
  const points = sortedPoints(body);
  return {
    name,
    source,
    size: body.size,
    pointIndices: points,
    coordinates: points.map((idx) => Core.indexToXY(idx, q)),
    shadow: shadowSummary(q, table, body),
  };
}

function preferWitnessPair(group, candidate) {
  if (!group.witnessA) {
    group.witnessA = candidate;
    return;
  }
  if (group.witnessA.bodyKey === candidate.bodyKey) return;
  if (!group.witnessB) {
    group.witnessB = candidate;
    return;
  }
  const currentDifferentSize = group.witnessA.size !== group.witnessB.size;
  const candidateDifferentSize = group.witnessA.size !== candidate.size;
  if (!currentDifferentSize && candidateDifferentSize) group.witnessB = candidate;
}

function addObservedBody(q, table, groups, name, body, source) {
  const summary = shadowSummary(q, table, body);
  const key = summary.bitset;
  let group = groups.get(key);
  if (!group) {
    group = {
      bitset: key,
      bits: summary.bits,
      covered: summary.covered,
      missing: summary.missing,
      directionsCovered: summary.directionsCovered,
      complete: summary.complete,
      count: 0,
      minSize: Infinity,
      maxSize: -Infinity,
      hasDifferentSizes: false,
      witnessA: null,
      witnessB: null,
    };
    groups.set(key, group);
  }
  group.count++;
  if (body.size !== group.minSize && group.minSize !== Infinity) group.hasDifferentSizes = true;
  if (body.size !== group.maxSize && group.maxSize !== -Infinity) group.hasDifferentSizes = true;
  group.minSize = Math.min(group.minSize, body.size);
  group.maxSize = Math.max(group.maxSize, body.size);
  preferWitnessPair(group, {
    name,
    source,
    size: body.size,
    bodyKey: bodyKey(body),
    pointIndices: sortedPoints(body),
  });
}

function enumerateCombinations(n, k, visit, shouldStop) {
  if (k === 0) {
    visit([]);
    return;
  }
  const combo = Array(k);
  function rec(start, depth) {
    if (shouldStop()) return;
    if (depth === k) {
      visit(combo.slice());
      return;
    }
    const remaining = k - depth;
    for (let i = start; i <= n - remaining; i++) {
      combo[depth] = i;
      rec(i + 1, depth + 1);
      if (shouldStop()) return;
    }
  }
  rec(0, 0);
}

function runEnumeration(q, table, maxSize, maxStates) {
  const n = Core.pointCount(q);
  const groups = new Map();
  let stateCount = 0;
  let truncated = false;
  let lastCompletedSize = -1;

  for (let size = 0; size <= maxSize; size++) {
    let stoppedInsideSize = false;
    enumerateCombinations(
      n,
      size,
      (points) => {
        if (stateCount >= maxStates) {
          truncated = true;
          stoppedInsideSize = true;
          return;
        }
        const body = new Set(points);
        addObservedBody(q, table, groups, `enum-size-${size}-state-${stateCount}`, body, {
          kind: "bounded-enumeration",
          bodySize: size,
        });
        stateCount++;
      },
      () => stateCount >= maxStates,
    );
    if (stoppedInsideSize || stateCount >= maxStates) {
      truncated = true;
      break;
    }
    lastCompletedSize = size;
  }

  const signatures = serializeGroups(groups);
  const collisionSignatures = signatures.filter((g) => g.count > 1);
  const differentSizeCollisionSignatures = collisionSignatures.filter((g) => g.hasDifferentSizes);
  const largestCollision = collisionSignatures[0] ?? null;
  const largestNonemptyCollision =
    collisionSignatures.find((g) => g.directionsCovered > 0) ?? null;

  return {
    q,
    pointCount: n,
    maxBodySize: maxSize,
    maxStates,
    stateCount,
    truncated,
    lastCompletedSize,
    signatureCount: signatures.length,
    collisionSignatureCount: collisionSignatures.length,
    differentSizeCollisionSignatureCount: differentSizeCollisionSignatures.length,
    maxCollisionClassCount: largestCollision?.count ?? 0,
    maxNonemptyCollisionClassCount: largestNonemptyCollision?.count ?? 0,
    largestCollision,
    largestNonemptyCollision,
    signatures,
  };
}

function runLineExtensionFamily(q, table) {
  const n = Core.pointCount(q);
  const groups = new Map();
  let stateCount = 0;

  for (let dirIndex = 0; dirIndex < table.length; dirIndex++) {
    const dir = table[dirIndex];
    for (let intercept = 0; intercept < dir.lines.length; intercept++) {
      const linePoints = dir.lines[intercept];
      const lineSet = new Set(linePoints);
      addObservedBody(
        q,
        table,
        groups,
        `line-extension-dir-${dir.label}-b-${intercept}`,
        lineSet,
        {
          kind: "line-extension-family",
          direction: dir.label,
          directionIndex: dirIndex,
          intercept,
          extraPoint: null,
        },
      );
      stateCount++;

      for (let p = 0; p < n; p++) {
        if (lineSet.has(p)) continue;
        const extended = new Set(linePoints);
        extended.add(p);
        addObservedBody(
          q,
          table,
          groups,
          `line-extension-dir-${dir.label}-b-${intercept}-plus-${p}`,
          extended,
          {
            kind: "line-extension-family",
            direction: dir.label,
            directionIndex: dirIndex,
            intercept,
            extraPoint: p,
          },
        );
        stateCount++;
      }
    }
  }

  const signatures = serializeGroups(groups);
  const collisionSignatures = signatures.filter((g) => g.count > 1);
  const differentSizeCollisionSignatures = collisionSignatures.filter((g) => g.hasDifferentSizes);
  const largestCollision = collisionSignatures[0] ?? null;
  const largestNonemptyCollision =
    collisionSignatures.find((g) => g.directionsCovered > 0) ?? null;

  return {
    q,
    family: "line-plus-one-outside-point",
    pointCount: n,
    stateCount,
    signatureCount: signatures.length,
    collisionSignatureCount: collisionSignatures.length,
    differentSizeCollisionSignatureCount: differentSizeCollisionSignatures.length,
    maxCollisionClassCount: largestCollision?.count ?? 0,
    maxNonemptyCollisionClassCount: largestNonemptyCollision?.count ?? 0,
    largestCollision,
    largestNonemptyCollision,
    signatures,
  };
}

function serializeGroups(groups) {
  return [...groups.values()]
    .map((group) => ({
      bitset: group.bitset,
      bits: group.bits,
      covered: group.covered,
      missing: group.missing,
      directionsCovered: group.directionsCovered,
      complete: group.complete,
      count: group.count,
      minSize: group.minSize,
      maxSize: group.maxSize,
      hasDifferentSizes: group.hasDifferentSizes || group.minSize !== group.maxSize,
      witnessA: compactWitness(group.witnessA),
      witnessB: compactWitness(group.witnessB),
    }))
    .sort((a, b) => b.count - a.count || a.directionsCovered - b.directionsCovered);
}

function compactWitness(witness) {
  if (!witness) return null;
  return {
    name: witness.name,
    source: witness.source,
    size: witness.size,
    pointIndices: witness.pointIndices,
  };
}

function guardWitnesses(q, table) {
  const dirs = Core.directions(q);
  const line0 = Core.bSingleLine(q, dirs[0], 0);
  const line1 = Core.bSingleLine(q, dirs[0], 1);
  const whole = Core.bWholePlane(q);
  const minusOne = Core.bWholeMinusOne(q, 0);

  const line0Shadow = shadowSummary(q, table, line0);
  const line1Shadow = shadowSummary(q, table, line1);
  const wholeShadow = shadowSummary(q, table, whole);
  const minusOneShadow = shadowSummary(q, table, minusOne);

  const singleDirection = {
    name: "single-direction parallel-line collision",
    pass: line0Shadow.bitset === line1Shadow.bitset && !sameBody(line0, line1),
    sameSignature: line0Shadow.bitset === line1Shadow.bitset,
    bodiesDiffer: !sameBody(line0, line1),
    bodiesDifferInSize: line0.size !== line1.size,
    signature: line0Shadow.bitset,
    bodies: [
      bodyDescriptor(q, table, "slope-0 intercept-0 line", line0, {
        kind: "registered-guard",
        guard: "single-direction-collision",
      }),
      bodyDescriptor(q, table, "slope-0 intercept-1 line", line1, {
        kind: "registered-guard",
        guard: "single-direction-collision",
      }),
    ],
  };

  const complete = {
    name: "complete-shadow whole-plane collision",
    pass:
      wholeShadow.bitset === minusOneShadow.bitset &&
      wholeShadow.complete &&
      minusOneShadow.complete &&
      !sameBody(whole, minusOne),
    sameSignature: wholeShadow.bitset === minusOneShadow.bitset,
    bodiesDiffer: !sameBody(whole, minusOne),
    bodiesDifferInSize: whole.size !== minusOne.size,
    signature: wholeShadow.bitset,
    bodies: [
      bodyDescriptor(q, table, "whole plane", whole, {
        kind: "registered-guard",
        guard: "complete-shadow-collision",
      }),
      bodyDescriptor(q, table, "whole plane minus point 0", minusOne, {
        kind: "registered-guard",
        guard: "complete-shadow-collision",
      }),
    ],
  };

  return {
    q,
    pass: singleDirection.pass && complete.pass,
    witnesses: [singleDirection, complete],
  };
}

function csvValue(value) {
  const text = String(value ?? "");
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

function witnessPoints(witness) {
  return witness?.pointIndices?.join(" ") ?? "";
}

function writeCsv(file, signatures) {
  const rows = [
    [
      "bitset",
      "directions_covered",
      "covered_labels",
      "count",
      "min_size",
      "max_size",
      "has_different_sizes",
      "witness_a_name",
      "witness_a_size",
      "witness_a_points",
      "witness_b_name",
      "witness_b_size",
      "witness_b_points",
    ],
  ];
  for (const sig of signatures) {
    rows.push([
      sig.bitset,
      sig.directionsCovered,
      sig.covered.join(" "),
      sig.count,
      sig.minSize,
      sig.maxSize,
      sig.hasDifferentSizes,
      sig.witnessA?.name ?? "",
      sig.witnessA?.size ?? "",
      witnessPoints(sig.witnessA),
      sig.witnessB?.name ?? "",
      sig.witnessB?.size ?? "",
      witnessPoints(sig.witnessB),
    ]);
  }
  fs.writeFileSync(file, rows.map((row) => row.map(csvValue).join(",")).join("\n") + "\n");
}

function writeOperatorCommands(
  file,
  command,
  manifestPath,
  enumCsvPath,
  structuredCsvPath,
  witnessesPath,
) {
  fs.writeFileSync(
    file,
    `# Kakeya Shadow Collision Audit - Operator Commands

\`\`\`powershell
${command}
\`\`\`

Primary outputs:

- \`${manifestPath}\`
- \`${enumCsvPath}\`
- \`${structuredCsvPath}\`
- \`${witnessesPath}\`

This audit measures the registered direction-shadow only. It does not expose
point membership as the primary signature, does not search for extremal Kakeya
sets, and does not make a Euclidean Kakeya claim.
`,
  );
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const q = readInt(args.q ?? 5, "q");
  const maxSize = readInt(args["max-size"] ?? 6, "max-size");
  const maxStates = readInt(args["max-states"] ?? 300000, "max-states");
  const outDir = String(args.out ?? defaultOutFor(q));

  if (!Core.isSupportedQ(q)) {
    throw new Error(`Unsupported q=${q}; supported values are ${Core.SUPPORTED_Q.join(", ")}`);
  }
  if (maxSize < 0) throw new Error("--max-size must be nonnegative");
  if (maxStates <= 0) throw new Error("--max-states must be positive");

  const table = lineTable(q);
  const registeredGuard = guardWitnesses(q, table);
  const enumeration = runEnumeration(q, table, maxSize, maxStates);
  const structuredLineExtensions = runLineExtensionFamily(q, table);
  const hasManyToOneMeasurement =
    enumeration.signatureCount < enumeration.stateCount ||
    structuredLineExtensions.signatureCount < structuredLineExtensions.stateCount;
  const hasNonemptyCollisionMeasurement =
    Boolean(enumeration.largestNonemptyCollision) ||
    Boolean(structuredLineExtensions.largestNonemptyCollision);
  const falsifierFired =
    !registeredGuard.pass || !hasManyToOneMeasurement || !hasNonemptyCollisionMeasurement;

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal measurement receipt",
    hook: "H-K3 direction-shadow collision audit for finite-field Kakeya",
    parameters: { q, maxSize, maxStates },
    command: `node scripts/kakeya-shadow-collision-audit.mjs --q ${q} --max-size ${maxSize} --max-states ${maxStates}`,
    core: {
      supportedQ: Core.SUPPORTED_Q,
      directionOrder: Core.directions(q).map((dir) => dir.label),
      pointCount: Core.pointCount(q),
      dvirFloor: Core.dvirFloor(q),
    },
    registeredGuard,
    enumeration: {
      q: enumeration.q,
      pointCount: enumeration.pointCount,
      maxBodySize: enumeration.maxBodySize,
      maxStates: enumeration.maxStates,
      stateCount: enumeration.stateCount,
      truncated: enumeration.truncated,
      lastCompletedSize: enumeration.lastCompletedSize,
      signatureCount: enumeration.signatureCount,
      collisionSignatureCount: enumeration.collisionSignatureCount,
      differentSizeCollisionSignatureCount: enumeration.differentSizeCollisionSignatureCount,
      maxCollisionClassCount: enumeration.maxCollisionClassCount,
      maxNonemptyCollisionClassCount: enumeration.maxNonemptyCollisionClassCount,
      largestCollision: enumeration.largestCollision,
      largestNonemptyCollision: enumeration.largestNonemptyCollision,
    },
    structuredLineExtensions: {
      q: structuredLineExtensions.q,
      family: structuredLineExtensions.family,
      pointCount: structuredLineExtensions.pointCount,
      stateCount: structuredLineExtensions.stateCount,
      signatureCount: structuredLineExtensions.signatureCount,
      collisionSignatureCount: structuredLineExtensions.collisionSignatureCount,
      differentSizeCollisionSignatureCount:
        structuredLineExtensions.differentSizeCollisionSignatureCount,
      maxCollisionClassCount: structuredLineExtensions.maxCollisionClassCount,
      maxNonemptyCollisionClassCount: structuredLineExtensions.maxNonemptyCollisionClassCount,
      largestCollision: structuredLineExtensions.largestCollision,
      largestNonemptyCollision: structuredLineExtensions.largestNonemptyCollision,
    },
    falsifier: {
      name: "KAK_SHADOW_REENCODING_EMPIRICAL",
      fired: falsifierFired,
      reason: falsifierFired
        ? "The registered guard failed or the measured state families did not produce nonempty collisions."
        : `The registered q=${q} shadow is many-to-one on bounded or structured states and on guard witnesses.`,
    },
    interpretation:
      "The direction-coverage bitset is a lossy projection. Collision classes are expected and pedagogically useful; they do not provide new Kakeya mathematics.",
  };

  fs.mkdirSync(outDir, { recursive: true });
  const manifestPath = path.join(outDir, "manifest.json");
  const csvPath = path.join(outDir, "signature-summary.csv");
  const structuredCsvPath = path.join(outDir, "structured-line-extension-summary.csv");
  const witnessesPath = path.join(outDir, "witnesses.json");
  const commandsPath = path.join(outDir, "operator-commands.md");

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  writeCsv(csvPath, enumeration.signatures);
  writeCsv(structuredCsvPath, structuredLineExtensions.signatures);
  fs.writeFileSync(
    witnessesPath,
    JSON.stringify(
      {
        artifactId: `${ARTIFACT_ID}-WITNESSES`,
        registeredGuard,
        largestCollision: enumeration.largestCollision,
        largestNonemptyCollision: enumeration.largestNonemptyCollision,
        structuredLargestCollision: structuredLineExtensions.largestCollision,
        structuredLargestNonemptyCollision: structuredLineExtensions.largestNonemptyCollision,
      },
      null,
      2,
    ) + "\n",
  );
  writeOperatorCommands(
    commandsPath,
    manifest.command,
    manifestPath,
    csvPath,
    structuredCsvPath,
    witnessesPath,
  );

  console.log(
    [
      "KAK_SHADOW_COLLISION_AUDIT",
      `q=${q}`,
      `states=${enumeration.stateCount}`,
      `signatures=${enumeration.signatureCount}`,
      `collisions=${enumeration.collisionSignatureCount}`,
      `max_collision=${enumeration.maxCollisionClassCount}`,
      `max_nonempty_collision=${enumeration.maxNonemptyCollisionClassCount}`,
      `structured_states=${structuredLineExtensions.stateCount}`,
      `structured_signatures=${structuredLineExtensions.signatureCount}`,
      `structured_max_nonempty_collision=${structuredLineExtensions.maxNonemptyCollisionClassCount}`,
      `guard=${registeredGuard.pass ? "pass" : "fail"}`,
      `falsifier=${falsifierFired ? "fired" : "clear"}`,
      `out=${outDir}`,
    ].join(" "),
  );
}

main();
