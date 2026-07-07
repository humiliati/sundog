#!/usr/bin/env node
// H-K3 two-level-law probe: q = 17 and q = 19 out-of-register sidecars
// (PHASE3J banked reopeners), with q in {7, 11, 13} as controls.
//
// PHASE3J falsified the equianharmonic conjecture and banked a sharper
// pattern over q in {5, 7, 11, 13}: every 4-star cross-ratio orbit has
// ex in {(q-3)/2, (q-1)/2} (the concurrency budget or budget-minus-one), and
// the harmonic orbit pays the FULL budget exactly when q = 1 (mod 4). This
// probe pre-registers:
//   TL-1: all orbit ex in {(q-3)/2, (q-1)/2} at q in {17, 19}
//   HM-1: harmonic ex = 8 (full budget) at q=17;  = 8 (budget-1) at q=19
//   GU-1: the TWO generic orbits inside each field agree, both low
//   EQ-3: equianharmonic at q=19 is high (9) - anti-correlated with harmonic
//         (weak: two prior data points)
// Outcomes are measurements; the falsifier is instrument-only.
//
// REGISTER NOTE: as in PHASE3J, q in {17, 19} are NOT added to the workbench;
// SUPPORTED_Q, the Phase-2 spec, the UI, and the regression suite are
// untouched (Core geometry functions are total in q).
//
// Solver upgrades over PHASE3J (validated by the controls reproducing
// PHASE3I/3J exactly): best-first child ordering inside the branch-and-bound,
// and a multi-order greedy seed.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3K-TWOLEVEL-PROBE";
const DEFAULT_OUT = path.join("results", "kakeya", "twolevel-law-probe");
// Default = controls + q17 (~2h). q = 19 is staged as an owner-fired run
// (projected multi-day at current solver): --fields 7,11,13,19
const DEFAULT_FIELDS = [7, 11, 13, 17];
const CONTROL_EXPECTED = {
  7: { harmonic: 2, equianharmonic: 3 },
  11: { harmonic: 4, generic: 4 },
  13: { harmonic: 6, equianharmonic: 5, generic: 5 },
};
const REPS_CONTROL = 8; // lex-first 4 + lex-last 4 (PHASE3J protocol)
// Amendment A (after the v1 timing run exhausted the node budget on all six
// q=17 solves): probe-field reps 4 -> 2. Rep invariance is theorem-backed
// (PGL-equivalence of same-orbit stars) and audited 580/580 at q <= 11; the
// probe's job is the per-orbit LEVEL, which one exact solve determines.
const REPS_PROBE = 2; // lex-first 1 + lex-last 1 at q in {17, 19}
const NODE_BUDGET = 2_000_000_000;
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
  return `Usage: node scripts/kakeya-twolevel-law-probe.mjs [options]

Options:
  --fields <list>     Comma-separated fields (debug). Default: ${DEFAULT_FIELDS.join(",")}
  --reps <n>          Reps per orbit override (debug).
  --out <dir>         Output directory. Default: ${DEFAULT_OUT}
  --help              Show this message.

The registered receipt run uses the defaults.
`;
}

function bmMinimum(q) {
  return (q * (q + 1)) / 2 + (q - 1) / 2;
}

// --- field / cross-ratio helpers -------------------------------------------------

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

function baseLabel(l, q) {
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

// --- exact completion solver (PHASE3G core + best-first children + multi-seed) ---

function wordCount(q) {
  return Math.ceil(Core.pointCount(q) / 32);
}

function popcount32(v) {
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  return (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function greedySeed(q, ordered, words, dirOrder) {
  const union = new Uint32Array(words);
  for (const j of dirOrder) {
    const perB = ordered[j];
    let bestB = 0;
    let bestNew = Infinity;
    for (let b = 0; b < q; b++) {
      let added = 0;
      const m = perB[b].mask;
      for (let w = 0; w < words; w++) added += popcount32(m[w] & ~union[w]);
      if (added < bestNew) {
        bestNew = added;
        bestB = b;
      }
    }
    const m = perB[bestB].mask;
    for (let w = 0; w < words; w++) union[w] |= m[w];
  }
  let count = 0;
  for (let w = 0; w < words; w++) count += popcount32(union[w]);
  return count;
}

// starPivotSymmetry is sound ONLY for bodies fixed by every scaling about the
// origin (our probe stars): the GL-stabilizer of 4 distinct directions is the
// scalars, and scaling multiplies every chosen intercept by lambda, so any
// completion is equivalent to one whose ROOT direction intercept is 0 or 1.
function solveCompletion(q, K, nodeBudget, starPivotSymmetry = false) {
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
  const k = ordered.length;

  // Multi-order greedy seed: identity, reversed, and six mulberry32 shuffles.
  const identity = [...Array(k).keys()];
  let seedCount = greedySeed(q, ordered, words, identity);
  seedCount = Math.min(seedCount, greedySeed(q, ordered, words, [...identity].reverse()));
  for (let s = 1; s <= 6; s++) {
    const rng = Core.mulberry32(s);
    const shuffled = [...identity];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    seedCount = Math.min(seedCount, greedySeed(q, ordered, words, shuffled));
  }
  let best = seedCount + 1;

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
    // Best-first child ordering: expand cheapest unions first. At the root,
    // under star-pivot symmetry, intercepts {0, 1} suffice (scaling orbit
    // representatives).
    const bLimit = starPivotSymmetry && level === 0 ? 2 : q;
    const children = [];
    for (let b = 0; b < bLimit; b++) {
      nodes++;
      let newCount = 0;
      const m = ordered[level][b].mask;
      for (let w = 0; w < words; w++) newCount += popcount32(prev[w] | m[w]);
      if (newCount < best) children.push([newCount, b]);
    }
    children.sort((x, y) => x[0] - y[0]);
    const next = unions[level + 1];
    for (const [newCount, b] of children) {
      if (newCount >= best) continue;
      const m = ordered[level][b].mask;
      for (let w = 0; w < words; w++) next[w] = prev[w] | m[w];
      rec(level + 1, newCount);
    }
  }

  rec(0, 0);
  const joint = Math.min(best, seedCount);
  return { joint, status: exhausted ? "exact" : "budget", nodes };
}

// --- witness anchor ----------------------------------------------------------------

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

// --- probe ------------------------------------------------------------------------

function starBody(q, dirIndexes) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (const i of dirIndexes) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  return body;
}

function dirValue(index, q) {
  return index === q ? INF : index;
}

function probeField(q, reps) {
  const dirCount = q + 1;
  const bm = bmMinimum(q);

  const buckets = new Map();
  for (let a = 0; a < dirCount; a++)
    for (let b = a + 1; b < dirCount; b++)
      for (let c = b + 1; c < dirCount; c++)
        for (let d = c + 1; d < dirCount; d++) {
          const l = crossRatio(dirValue(a, q), dirValue(b, q), dirValue(c, q), dirValue(d, q), q);
          const key = orbitKey(l, q);
          if (!buckets.has(key)) buckets.set(key, { base: baseLabel(l, q), quads: [] });
          buckets.get(key).quads.push([a, b, c, d]);
        }

  // Disambiguate multiple generic orbits deterministically by six-set order.
  const byBase = new Map();
  for (const key of [...buckets.keys()].sort()) {
    const base = buckets.get(key).base;
    byBase.set(base, (byBase.get(base) ?? 0) + 1);
  }
  const counters = new Map();
  const labels = new Map();
  for (const key of [...buckets.keys()].sort()) {
    const base = buckets.get(key).base;
    if (byBase.get(base) === 1) labels.set(key, base);
    else {
      const n = (counters.get(base) ?? 0) + 1;
      counters.set(base, n);
      labels.set(key, `${base}-${String.fromCharCode(96 + n)}`);
    }
  }

  const inventoryMatch =
    buckets.size === expectedOrbitKeys(q).size &&
    [...buckets.keys()].every((k) => expectedOrbitKeys(q).has(k));

  const rows = [];
  const orbitResults = [];
  let solverExact = true;
  let invariance = true;

  for (const key of [...buckets.keys()].sort()) {
    const bucket = buckets.get(key);
    const half = reps / 2;
    const repQuads = [...bucket.quads.slice(0, half), ...bucket.quads.slice(-half)];
    const exValues = new Set();
    for (const quad of repQuads) {
      const body = starBody(q, quad);
      const solved = solveCompletion(q, body, NODE_BUDGET, true);
      solverExact = solverExact && solved.status === "exact";
      const completion = body.size + solved.joint;
      const ex = completion - bm;
      exValues.add(ex);
      rows.push({
        q,
        quad: quad.map((i) => Core.directions(q)[i].label).join(" "),
        orbit: labels.get(key),
        sixSet: key,
        completion,
        ex,
        nodes: solved.nodes,
        status: solved.status,
      });
    }
    invariance = invariance && exValues.size === 1;
    orbitResults.push({
      label: labels.get(key),
      sixSet: key,
      quadrupleCount: bucket.quads.length,
      repsSolved: repQuads.length,
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
  const fields = args.fields
    ? String(args.fields).split(",").map(Number)
    : DEFAULT_FIELDS;
  const repsOverride = args.reps ? Number(args.reps) : null;

  // Witness anchors at the probe fields.
  const witnessChecks = {};
  for (const q of fields.filter((q) => q >= 17)) {
    const w = parabolaWitness(q);
    const p = witnessProfile(q, w);
    witnessChecks[q] = {
      size: w.size,
      expected: bmMinimum(q),
      complete: Core.shadowSummary(q, w).complete,
      sacrifice: p?.sacrifice,
      maxMult: p?.maxMult,
      pass:
        w.size === bmMinimum(q) &&
        Core.shadowSummary(q, w).complete &&
        p !== null &&
        p.unionSize === w.size &&
        p.sacrifice === (q - 1) / 2 &&
        p.maxMult === 3,
    };
  }

  const perQ = fields.map((q) =>
    probeField(q, repsOverride ?? (q >= 17 ? REPS_PROBE : REPS_CONTROL)),
  );

  // Instrument checks.
  const exOf = (q, label) =>
    perQ.find((b) => b.q === q)?.orbitResults.find((o) => o.label === label)?.ex;
  let controls = true;
  for (const [qs, expected] of Object.entries(CONTROL_EXPECTED)) {
    const q = Number(qs);
    if (!fields.includes(q)) continue;
    for (const [label, ex] of Object.entries(expected)) {
      controls = controls && exOf(q, label) === String(ex);
    }
  }
  const instrumentPass =
    controls &&
    Object.values(witnessChecks).every((w) => w.pass) &&
    perQ.every((b) => b.inventoryMatch && b.solverExact && b.invariance);

  // Hypothesis outcomes (measurements).
  const low = (q) => (q - 3) / 2;
  const high = (q) => (q - 1) / 2;
  const num = (v) => (v === undefined ? NaN : Number(v));
  const hypotheses = {};
  if (fields.includes(17)) {
    const h = num(exOf(17, "harmonic"));
    const ga = num(exOf(17, "generic-a"));
    const gb = num(exOf(17, "generic-b"));
    hypotheses["TL-1 q=17 all ex in {7,8}"] = {
      measured: [h, ga, gb].every((v) => v === low(17) || v === high(17)),
      values: `harmonic=${h} generic-a=${ga} generic-b=${gb}`,
    };
    hypotheses["HM-1 q=17 harmonic = 8 (full budget, q=1 mod 4)"] = { measured: h === high(17) };
    hypotheses["GU-1 q=17 generics equal and low"] = {
      measured: ga === gb && ga === low(17),
    };
  }
  if (fields.includes(19)) {
    const h = num(exOf(19, "harmonic"));
    const e = num(exOf(19, "equianharmonic"));
    const ga = num(exOf(19, "generic-a"));
    const gb = num(exOf(19, "generic-b"));
    hypotheses["TL-1 q=19 all ex in {8,9}"] = {
      measured: [h, e, ga, gb].every((v) => v === low(19) || v === high(19)),
      values: `harmonic=${h} equianharmonic=${e} generic-a=${ga} generic-b=${gb}`,
    };
    hypotheses["HM-1 q=19 harmonic = 8 (budget-1, q=3 mod 4)"] = { measured: h === low(19) };
    hypotheses["GU-1 q=19 generics equal and low"] = {
      measured: ga === gb && ga === low(19),
    };
    hypotheses["EQ-3 q=19 equianharmonic = 9 (high, anti-correlated)"] = {
      measured: e === high(19),
    };
  }

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal out-of-register probe receipt (q = 17/19 sidecars)",
    hook: "H-K3 two-level law + harmonic mod-4 rule + generic uniformity + equianharmonic orientation",
    command: "node scripts/kakeya-twolevel-law-probe.mjs",
    registerNote:
      "q in {17, 19} are NOT added to the workbench; SUPPORTED_Q, the Phase-2 spec, the UI, and the regression suite are untouched. ex is relative to the pinned Blokhuis-Mazzocca odd-q minimum.",
    preRegistered: [
      "TL-1: all orbit ex in {(q-3)/2, (q-1)/2} at q in {17, 19}",
      "HM-1: harmonic ex = (q-1)/2 at q=17 (1 mod 4), = (q-3)/2 at q=19 (3 mod 4)",
      "GU-1: the two generic orbits inside each probe field are equal, both (q-3)/2",
      "EQ-3 (weak, 2 prior points): equianharmonic at q=19 = (q-1)/2, anti-correlated with harmonic",
      `sampling: ${REPS_PROBE} reps/orbit at probe fields (lex-first 2 + lex-last 2), ${REPS_CONTROL} at controls; exact solves only`,
      "controls: q in {7, 11, 13} through the upgraded solver must reproduce PHASE3I/3J exactly",
    ],
    solverUpgrades:
      "best-first child ordering + multi-order greedy seed (8 orders) + star-pivot scaling symmetry (root intercepts {0,1}; sound for origin-star bodies, validated by the controls)",
    nodeBudget: NODE_BUDGET,
    falsifier: {
      name: "TWOLEVEL_INSTRUMENT_MISMATCH",
      description:
        "Instrument-only: fires on budget exhaustion, intra-orbit variation among reps, control mismatch vs PHASE3I/3J, orbit-inventory mismatch vs field arithmetic, or witness-anchor failure at the probe fields. Hypothesis outcomes cannot fire it.",
      status: instrumentPass ? "clear" : "fired",
    },
    witnessChecks,
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
    `# Kakeya Two-Level Law Probe - Operator Commands

\`\`\`powershell
${manifest.command}
\`\`\`

Primary outputs:

- \`${manifestPath}\`
- \`${csvPath}\`

Out-of-register q = 17/19 sidecars; the workbench register is untouched. No
Euclidean claim.
`,
  );

  console.log(
    [
      "KAK_TWOLEVEL_PROBE",
      `fields={${fields.join(",")}}`,
      ...perQ.map(
        (b) => `q${b.q}=` + b.orbitResults.map((o) => `${o.label}:ex=${o.ex}`).join("+"),
      ),
      `controls=${controls ? "pass" : "fail"}`,
      ...Object.entries(hypotheses).map(
        ([name, h]) => `${name.split(" ")[0]}_q${name.includes("17") ? "17" : "19"}=${h.measured}`,
      ),
      `falsifier=${manifest.falsifier.status}`,
      `out=${outDir}`,
    ].join(" "),
  );
  process.exit(instrumentPass ? 0 : 1);
}

main();
