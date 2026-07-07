#!/usr/bin/env node
// H-K3 4-star cross-ratio orbit sweep: the PHASE3G banked reopener.
//
// Unordered direction quadruples D (4 points of PG(1,q) = the q+1 directions)
// are classified up to PGL(2,q) by the six-set of their cross-ratio
// {l, 1/l, 1-l, 1/(1-l), l/(l-1), (l-1)/l}. Since 4-stars with PGL-equivalent
// direction quadruples are affinely equivalent, the embeddability excess
// ex(4-star) of PHASE3G is an orbit invariant. This sweep computes ex for
// EVERY quadruple at q in {5, 7, 11} (15 + 70 + 495 exact completions), which
// simultaneously (a) maps ex as a function of the orbit, (b) machine-checks
// orbit-invariance as a solver audit, and (c) tests a consequence of the
// pinned Blokhuis-Mazzocca classification (no odd-q minimal set has a mult-4
// point => ex >= 1 for every orbit).
//
// Orbit inventory (field arithmetic): q=5 has ONE orbit (harmonic; F_5\{0,1}
// IS the harmonic six-set). q=7 has TWO (harmonic {2,4,6} + equianharmonic
// {3,5}). q=11 has TWO (harmonic {2,6,10} + one generic orbit; -3 is a
// non-residue so no equianharmonic). PHASE3G's two q=11 probes were both
// generic - the harmonic orbit at q=11 is the sweep's NEW measurement.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3I-CROSSRATIO-ORBIT-SWEEP";
const DEFAULT_OUT = path.join("results", "kakeya", "crossratio-orbit-sweep");
const NODE_BUDGET = 40_000_000;
const INF = -1; // marker for the direction "inf" in cross-ratio arithmetic

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
  return `Usage: node scripts/kakeya-crossratio-orbit-sweep.mjs [options]

Options:
  --out <dir>         Output directory. Default: ${DEFAULT_OUT}
  --help              Show this message.
`;
}

function bmMinimum(q) {
  return (q * (q + 1)) / 2 + (q - 1) / 2;
}

// --- field helpers ------------------------------------------------------------

function mod(x, q) {
  return ((x % q) + q) % q;
}

function inv(x, q) {
  for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i;
  throw new Error(`no inverse of ${x} mod ${q}`);
}

// Cross-ratio (a,b;c,d) over PG(1,q) with INF; all four points distinct, so
// INF appears in at most one numerator and one denominator factor and those
// two factors cancel (replace both by 1).
function crossRatio(a, b, c, d, q) {
  const diff = (x, y) => (x === INF || y === INF ? 1 : mod(x - y, q));
  const num = (diff(a, c) * diff(b, d)) % q;
  const den = (diff(a, d) * diff(b, c)) % q;
  return (num * inv(den, q)) % q;
}

// Six-set of a cross-ratio value (all in F_q \ {0, 1}).
function sixSet(l, q) {
  const set = new Set();
  set.add(l);
  set.add(inv(l, q));
  const oneMinus = mod(1 - l, q);
  set.add(oneMinus);
  set.add(inv(oneMinus, q));
  set.add(mod(l * inv(mod(l - 1, q), q), q)); // l/(l-1)
  set.add(mod(mod(l - 1, q) * inv(l, q), q)); // (l-1)/l
  return [...set].sort((x, y) => x - y);
}

function orbitKey(l, q) {
  return sixSet(l, q).join(",");
}

function orbitLabel(l, q) {
  const six = sixSet(l, q);
  const harmonic = [2, mod((q + 1) / 2, q), q - 1].sort((x, y) => x - y);
  if (six.length === 3 && six.join(",") === harmonic.join(",")) return "harmonic";
  if (six.length === 2 && six.every((v) => mod(v * v - v + 1, q) === 0)) return "equianharmonic";
  return "generic";
}

// j-invariant j = 256 (l^2 - l + 1)^3 / (l^2 (l - 1)^2) mod q (reported only).
function jInvariant(l, q) {
  const t = mod(l * l - l + 1, q);
  const num = mod(256 * t * t * t, q);
  const den = mod(l * l * mod(l - 1, q) * mod(l - 1, q), q);
  return (num * inv(den, q)) % q;
}

// The expected orbit inventory from pure field arithmetic: distinct six-set
// keys over all l in F_q \ {0, 1}.
function expectedOrbits(q) {
  const keys = new Set();
  for (let l = 2; l < q; l++) keys.add(orbitKey(l, q));
  return keys;
}

// --- exact completion solver (PHASE3G machinery) -------------------------------

function wordCount(q) {
  return Math.ceil(Core.pointCount(q) / 32);
}

function popcount32(v) {
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  return (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function solveCompletion(q, K, nodeBudget) {
  const dirs = Core.directions(q);
  const words = wordCount(q);
  const bits = Core.shadowBitset(q, K);
  const targets = [];
  for (let i = 0; i < dirs.length; i++) if (bits[i] === 0) targets.push(i);
  if (targets.length === 0) return { joint: 0, status: "exact", nodes: 0 };

  const addMasks = targets.map((i) => {
    const perB = [];
    for (let b = 0; b < q; b++) {
      const mask = new Uint32Array(words);
      let size = 0;
      for (const p of Core.lineMask(dirs[i], b, q)) {
        if (!K.has(p)) {
          mask[p >> 5] |= 1 << (p & 31);
          size++;
        }
      }
      perB.push({ mask, size });
    }
    return perB;
  });
  const marginals = addMasks.map((perB) => Math.min(...perB.map((x) => x.size)));
  const order = targets.map((_, j) => j).sort((a, b) => marginals[b] - marginals[a]);
  const ordered = order.map((j) => addMasks[j]);

  const seedUnion = new Uint32Array(words);
  let seedCount = 0;
  for (const perB of ordered) {
    let bestB = 0;
    let bestNew = Infinity;
    for (let b = 0; b < q; b++) {
      let added = 0;
      const m = perB[b].mask;
      for (let w = 0; w < words; w++) added += popcount32(m[w] & ~seedUnion[w]);
      if (added < bestNew) {
        bestNew = added;
        bestB = b;
      }
    }
    const m = perB[bestB].mask;
    for (let w = 0; w < words; w++) seedUnion[w] |= m[w];
    seedCount = 0;
    for (let w = 0; w < words; w++) seedCount += popcount32(seedUnion[w]);
  }
  let best = seedCount + 1;

  const k = ordered.length;
  const unions = [];
  for (let level = 0; level <= k; level++) unions.push(new Uint32Array(words));
  let nodes = 0;
  let exhausted = true;

  function dynBound(level, union) {
    const r = k - level;
    if (r === 0) return 0;
    let sum = 0;
    let max = 0;
    for (let j = level; j < k; j++) {
      let mn = Infinity;
      const perB = ordered[j];
      for (let b = 0; b < q; b++) {
        let added = 0;
        const m = perB[b].mask;
        for (let w = 0; w < words; w++) added += popcount32(m[w] & ~union[w]);
        if (added < mn) mn = added;
        if (mn === 0) break;
      }
      sum += mn;
      if (mn > max) max = mn;
    }
    return Math.max(max, sum - (r * (r - 1)) / 2);
  }

  function rec(level, count) {
    if (nodes >= nodeBudget) {
      exhausted = false;
      return;
    }
    if (count + dynBound(level, unions[level]) >= best) return;
    if (level === k) {
      best = count;
      return;
    }
    const prev = unions[level];
    const next = unions[level + 1];
    for (let b = 0; b < q; b++) {
      nodes++;
      const m = ordered[level][b].mask;
      let newCount = 0;
      for (let w = 0; w < words; w++) {
        next[w] = prev[w] | m[w];
        newCount += popcount32(next[w]);
      }
      if (newCount >= best) continue;
      rec(level + 1, newCount);
    }
  }

  rec(0, 0);
  const joint = Math.min(best, seedCount);
  return { joint, status: exhausted ? "exact" : "budget", nodes };
}

// --- sweep -----------------------------------------------------------------------

function starBody(q, dirIndexes) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (const i of dirIndexes) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  return body;
}

function dirValue(index, q) {
  return index === q ? INF : index; // direction index -> PG(1,q) point
}

function sweepQ(q) {
  const dirCount = Core.directionCount(q);
  const bm = bmMinimum(q);
  const expected = expectedOrbits(q);

  const rows = [];
  const orbits = new Map(); // key -> { label, j, exValues:Set, count, sample }
  let solverExact = true;
  let nodesTotal = 0;

  for (let a = 0; a < dirCount; a++)
    for (let b = a + 1; b < dirCount; b++)
      for (let c = b + 1; c < dirCount; c++)
        for (let d = c + 1; d < dirCount; d++) {
          const quad = [a, b, c, d];
          const l = crossRatio(dirValue(a, q), dirValue(b, q), dirValue(c, q), dirValue(d, q), q);
          const key = orbitKey(l, q);
          const label = orbitLabel(l, q);
          const j = jInvariant(l, q);

          const body = starBody(q, quad);
          const solved = solveCompletion(q, body, NODE_BUDGET);
          nodesTotal += solved.nodes;
          solverExact = solverExact && solved.status === "exact";
          const completion = body.size + solved.joint;
          const ex = completion - bm;

          rows.push({
            quad: quad.map((i) => Core.directions(q)[i].label).join(" "),
            lambda: l,
            orbit: key,
            label,
            j,
            bodySize: body.size,
            completion,
            ex,
            nodes: solved.nodes,
          });

          if (!orbits.has(key)) {
            orbits.set(key, {
              label,
              j,
              sixSet: key,
              exValues: new Set(),
              count: 0,
              sample: rows[rows.length - 1].quad,
            });
          }
          const o = orbits.get(key);
          o.exValues.add(ex);
          o.count++;
        }

  const orbitTable = [...orbits.values()].map((o) => ({
    label: o.label,
    sixSet: o.sixSet,
    j: o.j,
    quadruples: o.count,
    ex: [...o.exValues].sort((x, y) => x - y).join("|"),
    invariant: o.exValues.size === 1,
    sample: o.sample,
  }));

  const invariance = orbitTable.every((o) => o.invariant);
  const inventoryMatch =
    orbits.size === expected.size && [...orbits.keys()].every((k) => expected.has(k));
  const allExPositive = rows.every((r) => r.ex >= 1);
  const quadTotal = rows.length;

  return {
    q,
    bmMinimum: bm,
    quadruples: quadTotal,
    orbitCount: orbits.size,
    expectedOrbitCount: expected.size,
    orbitTable,
    checks: { invariance, inventoryMatch, allExPositive, solverExact },
    nodesTotal,
    rows,
    pass: invariance && inventoryMatch && allExPositive && solverExact,
  };
}

// --- outputs ----------------------------------------------------------------------

function csvValue(value) {
  const text = String(value ?? "");
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

function writeCsv(file, perQ) {
  const header = [
    "q",
    "quadruple",
    "lambda",
    "orbit_six_set",
    "orbit_label",
    "j_invariant",
    "body_size",
    "completion",
    "ex",
    "nodes",
  ];
  const out = [];
  for (const block of perQ) {
    for (const r of block.rows) {
      out.push([
        block.q,
        r.quad,
        r.lambda,
        r.orbit,
        r.label,
        r.j,
        r.bodySize,
        r.completion,
        r.ex,
        r.nodes,
      ]);
    }
  }
  fs.writeFileSync(
    file,
    [header, ...out].map((row) => row.map(csvValue).join(",")).join("\n") + "\n",
  );
}

function writeOperatorCommands(file, command, manifestPath, csvPath) {
  fs.writeFileSync(
    file,
    `# Kakeya Cross-Ratio Orbit Sweep - Operator Commands

\`\`\`powershell
${command}
\`\`\`

Primary outputs:

- \`${manifestPath}\`
- \`${csvPath}\`

Report-only workbench diagnostic. ex values are relative to the pinned
Blokhuis-Mazzocca minimum (imported at q = 11). No Euclidean claim.
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
  const perQ = Core.SUPPORTED_Q.map(sweepQ);
  const pass = perQ.every((b) => b.pass);

  // PHASE3G anchor cross-checks: {0,1,2,3} and {0,1,2,4}.
  const anchor = (q, quad) => {
    const block = perQ.find((b) => b.q === q);
    return block.rows.find((r) => r.quad === quad)?.ex;
  };
  const anchors3G =
    anchor(5, "0 1 2 3") === 2 &&
    anchor(5, "0 1 2 4") === 2 &&
    anchor(7, "0 1 2 3") === 2 &&
    anchor(7, "0 1 2 4") === 3 &&
    anchor(11, "0 1 2 3") === 4 &&
    anchor(11, "0 1 2 4") === 4;

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal workbench diagnostic receipt",
    hook: "H-K3 4-star cross-ratio orbit sweep (PHASE3G banked reopener)",
    command: "node scripts/kakeya-crossratio-orbit-sweep.mjs",
    statement:
      "The embeddability excess ex of a 4-star is a PGL(2,q)-orbit invariant of its direction quadruple, classified by the cross-ratio six-set. Full sweep of every quadruple at q in {5, 7, 11} maps ex per orbit, machine-checks orbit invariance (solver audit), and tests the pinned Blokhuis-Mazzocca classification consequence ex >= 1 (no odd-q minimal set has a mult-4 point).",
    preRegistered: [
      "PR1: ex constant on each orbit (affine-equivalence theorem; full-sweep solver audit)",
      "PR2: q=5 harmonic ex=2 (single orbit); q=7 harmonic ex=2, equianharmonic ex=3 (PHASE3G anchors)",
      "PR3: every orbit has ex >= 1 (classification consequence, machine-tested)",
      "PR4: ex(harmonic, q=11) - OPEN, new measurement (both PHASE3G q=11 probes were generic)",
    ],
    falsifier: {
      name: "CROSSRATIO_ORBIT_MISMATCH",
      description:
        "Fires if ex varies within any orbit, the measured orbit inventory disagrees with the field-arithmetic six-set classes, any orbit has ex = 0 (would contradict the pinned classification), any solve exhausts its budget, or a PHASE3G anchor value is not reproduced.",
      status: pass && anchors3G ? "clear" : "fired",
    },
    anchors3G,
    perQ: perQ.map((b) => ({ ...b, rows: undefined })),
    pass: pass && anchors3G,
  };

  fs.mkdirSync(outDir, { recursive: true });
  const manifestPath = path.join(outDir, "manifest.json");
  const csvPath = path.join(outDir, "sweep.csv");
  const commandsPath = path.join(outDir, "operator-commands.md");

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  writeCsv(csvPath, perQ);
  writeOperatorCommands(commandsPath, manifest.command, manifestPath, csvPath);

  console.log(
    [
      "KAK_CROSSRATIO_ORBIT_SWEEP",
      `q={${Core.SUPPORTED_Q.join(",")}}`,
      ...perQ.map(
        (b) =>
          `q${b.q}_orbits=` +
          b.orbitTable.map((o) => `${o.label}:ex=${o.ex}(n=${o.quadruples})`).join("+"),
      ),
      `anchors3G=${anchors3G ? "pass" : "fail"}`,
      `falsifier=${manifest.pass ? "clear" : "fired"}`,
      `out=${outDir}`,
    ].join(" "),
  );
  process.exit(manifest.pass ? 0 : 1);
}

main();
