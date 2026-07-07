#!/usr/bin/env node
// H-K3 equianharmonic structure probe: the PHASE3I reopen condition, run as an
// OUT-OF-REGISTER sidecar at q = 13.
//
// PHASE3I measured: the 4-star excess splits by cross-ratio orbit at q = 7
// (harmonic 2, equianharmonic 3) and collapses at q = 11 (harmonic = generic
// = 4). The visible pattern: the deviating orbit is the equianharmonic one
// (j = 0, roots of l^2 - l + 1), and q = 11 has none (-3 non-residue). q = 13
// is the next field with an equianharmonic orbit (-3 = 10 = 6^2 mod 13).
//
// REGISTER NOTE: q = 13 is NOT added to the workbench. Core geometry functions
// are total in q (directions/lineMask/shadowBitset take q explicitly);
// SUPPORTED_Q, the Phase-2 spec lock, the UI, and the regression suite are
// untouched. This script is a one-hypothesis instrument.
//
// Pre-registered hypotheses (outcomes are measurements, not falsifier
// conditions):
//   EQ-1:  ex(harmonic, 13) = ex(generic, 13)
//   EQ-2:  ex(equianharmonic, 13) differs from that common value
//   EQ-2': the difference is +1 (the q = 7 magnitude)
// Sampling: 8 deterministic representatives per orbit (lexicographic first 4
// + last 4 quadruples, so inf-containing sets are included), exact solves
// only. Controls: q in {7, 11} re-derived through the same code path must
// reproduce PHASE3I (2/3 and 4/4).

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3J-EQUIANHARMONIC-PROBE";
const DEFAULT_OUT = path.join("results", "kakeya", "equianharmonic-probe");
const FIELDS = [7, 11, 13]; // 7/11 = controls, 13 = the probe field
const REPS_PER_ORBIT = 8; // lex-first 4 + lex-last 4
const NODE_BUDGET = 200_000_000;
const INF = -1;

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
  return `Usage: node scripts/kakeya-equianharmonic-probe.mjs [options]

Options:
  --out <dir>         Output directory. Default: ${DEFAULT_OUT}
  --help              Show this message.
`;
}

function bmMinimum(q) {
  return (q * (q + 1)) / 2 + (q - 1) / 2;
}

// --- field / cross-ratio helpers (PHASE3I) --------------------------------------

function mod(x, q) {
  return ((x % q) + q) % q;
}

function inv(x, q) {
  for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i;
  throw new Error(`no inverse of ${x} mod ${q}`);
}

function crossRatio(a, b, c, d, q) {
  const diff = (x, y) => (x === INF || y === INF ? 1 : mod(x - y, q));
  const num = (diff(a, c) * diff(b, d)) % q;
  const den = (diff(a, d) * diff(b, c)) % q;
  return (num * inv(den, q)) % q;
}

function sixSet(l, q) {
  const set = new Set();
  set.add(l);
  set.add(inv(l, q));
  const oneMinus = mod(1 - l, q);
  set.add(oneMinus);
  set.add(inv(oneMinus, q));
  set.add(mod(l * inv(mod(l - 1, q), q), q));
  set.add(mod(mod(l - 1, q) * inv(l, q), q));
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

function expectedOrbitKeys(q) {
  const keys = new Set();
  for (let l = 2; l < q; l++) keys.add(orbitKey(l, q));
  return keys;
}

// --- exact completion solver (PHASE3G/3I machinery) ------------------------------

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

// --- witness + profile (PHASE3G machinery, for the q = 13 anchor) ----------------

function parabolaWitness(q) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (let t = 0; t < q; t++) {
    const slope = (2 * t) % q;
    const intercept = mod(-(t * t), q);
    for (const p of Core.lineMask(dirs[slope], intercept, q)) body.add(p);
  }
  for (const p of Core.lineMask(dirs[q], 0, q)) body.add(p);
  return body;
}

function witnessProfile(q, K) {
  const mult = new Map();
  for (const dir of Core.directions(q)) {
    const w = Core.witnessLine(dir, q, K);
    if (!w) return null;
    for (const p of w.points) mult.set(p, (mult.get(p) ?? 0) + 1);
  }
  let sacrifice = 0;
  let maxMult = 0;
  for (const m of mult.values()) {
    sacrifice += ((m - 1) * (m - 2)) / 2;
    if (m > maxMult) maxMult = m;
  }
  return { sacrifice, maxMult, unionSize: mult.size };
}

// --- probe -----------------------------------------------------------------------

function starBody(q, dirIndexes) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (const i of dirIndexes) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  return body;
}

function dirValue(index, q) {
  return index === q ? INF : index;
}

function probeField(q) {
  const dirCount = q + 1;
  const bm = bmMinimum(q);

  // Bucket all quadruples by orbit (cross-ratio arithmetic only - cheap).
  const buckets = new Map();
  for (let a = 0; a < dirCount; a++)
    for (let b = a + 1; b < dirCount; b++)
      for (let c = b + 1; c < dirCount; c++)
        for (let d = c + 1; d < dirCount; d++) {
          const l = crossRatio(dirValue(a, q), dirValue(b, q), dirValue(c, q), dirValue(d, q), q);
          const key = orbitKey(l, q);
          if (!buckets.has(key)) buckets.set(key, { label: orbitLabel(l, q), quads: [] });
          buckets.get(key).quads.push([a, b, c, d]);
        }

  const inventoryMatch =
    buckets.size === expectedOrbitKeys(q).size &&
    [...buckets.keys()].every((k) => expectedOrbitKeys(q).has(k));

  // Deterministic representatives: lex-first 4 + lex-last 4 per orbit.
  const rows = [];
  const orbitResults = [];
  let solverExact = true;
  let invariance = true;

  for (const [key, bucket] of buckets) {
    const quads = bucket.quads;
    const half = REPS_PER_ORBIT / 2;
    const reps = [...quads.slice(0, half), ...quads.slice(-half)];
    const exValues = new Set();
    for (const quad of reps) {
      const body = starBody(q, quad);
      const solved = solveCompletion(q, body, NODE_BUDGET);
      solverExact = solverExact && solved.status === "exact";
      const completion = body.size + solved.joint;
      const ex = completion - bm;
      exValues.add(ex);
      rows.push({
        q,
        quad: quad.map((i) => Core.directions(q)[i].label).join(" "),
        orbit: bucket.label,
        sixSet: key,
        completion,
        ex,
        nodes: solved.nodes,
        status: solved.status,
      });
    }
    invariance = invariance && exValues.size === 1;
    orbitResults.push({
      label: bucket.label,
      sixSet: key,
      quadrupleCount: quads.length,
      repsSolved: reps.length,
      ex: [...exValues].sort((x, y) => x - y).join("|"),
    });
  }

  return { q, bmMinimum: bm, inventoryMatch, orbitResults, rows, solverExact, invariance };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const outDir = String(args.out ?? DEFAULT_OUT);

  // q = 13 anchor: the parabola witness works at any odd prime.
  const w13 = parabolaWitness(13);
  const p13 = witnessProfile(13, w13);
  const witnessOk =
    w13.size === bmMinimum(13) &&
    Core.shadowSummary(13, w13).complete &&
    p13 !== null &&
    p13.unionSize === w13.size &&
    p13.sacrifice === 6 &&
    p13.maxMult === 3;

  const perQ = FIELDS.map(probeField);

  // Instrument checks (falsifier-gated).
  const exOf = (q, label) =>
    perQ.find((b) => b.q === q)?.orbitResults.find((o) => o.label === label)?.ex;
  const controls =
    exOf(7, "harmonic") === "2" &&
    exOf(7, "equianharmonic") === "3" &&
    exOf(11, "harmonic") === "4" &&
    exOf(11, "generic") === "4";
  const instrumentPass =
    witnessOk &&
    controls &&
    perQ.every((b) => b.inventoryMatch && b.solverExact && b.invariance);

  // Hypothesis outcomes (measured, NOT falsifier-gated).
  const h13 = Number(exOf(13, "harmonic"));
  const g13 = Number(exOf(13, "generic"));
  const e13 = Number(exOf(13, "equianharmonic"));
  const hypotheses = {
    "EQ-1 harmonic = generic at q=13": {
      predicted: true,
      measured: h13 === g13,
      values: `harmonic=${h13} generic=${g13}`,
    },
    "EQ-2 equianharmonic deviates at q=13": {
      predicted: true,
      measured: e13 !== h13 || e13 !== g13,
      values: `equianharmonic=${e13}`,
    },
    "EQ-2' deviation is +1": {
      predicted: true,
      measured: h13 === g13 && e13 === h13 + 1,
      values: `delta=${e13 - h13}`,
    },
  };

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal out-of-register probe receipt (q = 13 sidecar)",
    hook: "H-K3 equianharmonic structure test (PHASE3I reopen condition)",
    command: "node scripts/kakeya-equianharmonic-probe.mjs",
    registerNote:
      "q = 13 is NOT added to the workbench: SUPPORTED_Q, the Phase-2 spec lock, the UI, and the regression suite are untouched. Core geometry functions are total in q; this script passes q = 13 explicitly. ex at q = 13 is relative to the pinned Blokhuis-Mazzocca minimum (odd-q formula; the litpass pin covers general odd q).",
    preRegistered: [
      "EQ-1: ex(harmonic, 13) = ex(generic, 13)",
      "EQ-2: ex(equianharmonic, 13) differs from the common value",
      "EQ-2': the difference is +1 (q = 7 magnitude)",
      `sampling: ${REPS_PER_ORBIT} representatives per orbit (lex-first 4 + lex-last 4), exact solves only`,
      "controls: q in {7, 11} through the same code path must reproduce PHASE3I (2/3 and 4/4)",
    ],
    falsifier: {
      name: "EQUIANHARMONIC_INSTRUMENT_MISMATCH",
      description:
        "Instrument-only: fires on solver budget exhaustion, intra-orbit ex variation among sampled representatives, control mismatch vs PHASE3I, orbit-inventory mismatch vs field arithmetic, or the q=13 parabola witness missing size 97 / completeness / the 6-triple profile. Hypothesis outcomes (EQ-1/2/2') are measurements and cannot fire it.",
      status: instrumentPass ? "clear" : "fired",
    },
    q13Witness: { size: w13.size, sacrifice: p13?.sacrifice, maxMult: p13?.maxMult, pass: witnessOk },
    hypotheses,
    perQ: perQ.map((b) => ({ ...b, rows: undefined })),
    pass: instrumentPass,
  };

  fs.mkdirSync(outDir, { recursive: true });
  const manifestPath = path.join(outDir, "manifest.json");
  const csvPath = path.join(outDir, "probe.csv");
  const commandsPath = path.join(outDir, "operator-commands.md");

  const csvValue = (value) => {
    const text = String(value ?? "");
    if (!/[",\n]/.test(text)) return text;
    return `"${text.replaceAll('"', '""')}"`;
  };
  const header = ["q", "quadruple", "orbit", "six_set", "completion", "ex", "nodes", "status"];
  const csvRows = perQ.flatMap((b) =>
    b.rows.map((r) => [r.q, r.quad, r.orbit, r.sixSet, r.completion, r.ex, r.nodes, r.status]),
  );
  fs.writeFileSync(
    csvPath,
    [header, ...csvRows].map((row) => row.map(csvValue).join(",")).join("\n") + "\n",
  );
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  fs.writeFileSync(
    commandsPath,
    `# Kakeya Equianharmonic Probe - Operator Commands

\`\`\`powershell
${manifest.command}
\`\`\`

Primary outputs:

- \`${manifestPath}\`
- \`${csvPath}\`

Out-of-register q = 13 sidecar; the workbench register (SUPPORTED_Q, spec,
tests) is untouched. No Euclidean claim.
`,
  );

  console.log(
    [
      "KAK_EQUIANHARMONIC_PROBE",
      `fields={${FIELDS.join(",")}}`,
      ...perQ.map(
        (b) =>
          `q${b.q}=` + b.orbitResults.map((o) => `${o.label}:ex=${o.ex}`).join("+"),
      ),
      `witness_q13=${witnessOk ? "pass" : "fail"}`,
      `controls=${controls ? "pass" : "fail"}`,
      `EQ1=${hypotheses["EQ-1 harmonic = generic at q=13"].measured}`,
      `EQ2=${hypotheses["EQ-2 equianharmonic deviates at q=13"].measured}`,
      `EQ2'=${hypotheses["EQ-2' deviation is +1"].measured}`,
      `falsifier=${manifest.falsifier.status}`,
      `out=${outDir}`,
    ].join(" "),
  );
  process.exit(instrumentPass ? 0 : 1);
}

main();
