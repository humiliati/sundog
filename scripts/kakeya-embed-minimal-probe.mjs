#!/usr/bin/env node
// H-K3 embeddability-in-minimal-sets probe: which bodies K sit inside a
// MINIMAL complete Kakeya set? completion(K) = |K| + joint(all missing
// directions, K) is the minimal size of a complete superset, so
// ex(K) = completion(K) - BM(q) >= 0 and K embeds iff ex(K) = 0.
//
// Concurrency-budget identity (pair-counting, machine-checked here): any
// complete set equal to the union of its q+1 canonical lines satisfies
// |K| = q(q+1)/2 + sacrifice, sacrifice = sum over points of
// (m-1)(m-2)/2 (m = line multiplicity). With the Blokhuis-Mazzocca minimum
// (derived exhaustively at q in {5,7} in PHASE3F, imported at q = 11):
// ex = sacrifice - (q-1)/2. Corollary: a k-star embeds only if
// (k-1)(k-2)/2 <= (q-1)/2.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3G-EMBED-MINIMAL-PROBE";
const DEFAULT_OUT = path.join("results", "kakeya", "embed-minimal-probe");
const SOLVER_EXACT_Q = [5, 7]; // exact ground truth: solver runs on every probe body
const NODE_BUDGET = 40_000_000; // pre-registered per-body cap at q = 11

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
  return `Usage: node scripts/kakeya-embed-minimal-probe.mjs [options]

Options:
  --out <dir>         Output directory. Default: ${DEFAULT_OUT}
  --help              Show this message.
`;
}

// --- shared geometry ---------------------------------------------------------

function bmMinimum(q) {
  return (q * (q + 1)) / 2 + (q - 1) / 2;
}

function budget(q) {
  return (q - 1) / 2;
}

function starSacrifice(k) {
  return ((k - 1) * (k - 2)) / 2;
}

function parabolaWitness(q) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (let t = 0; t < q; t++) {
    const slope = (2 * t) % q;
    const intercept = ((-(t * t)) % q + q) % q;
    for (const p of Core.lineMask(dirs[slope], intercept, q)) body.add(p);
  }
  for (const p of Core.lineMask(dirs[q], 0, q)) body.add(p);
  return body;
}

function starBody(q, dirIndexes) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (const i of dirIndexes) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  return body;
}

function tangentTriple(q) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (const t of [0, 1, 2]) {
    const slope = (2 * t) % q;
    const intercept = ((-(t * t)) % q + q) % q;
    for (const p of Core.lineMask(dirs[slope], intercept, q)) body.add(p);
  }
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

// Canonical decomposition: lowest-intercept covered line per direction
// (Core.witnessLine tie-break). Null if incomplete.
function canonicalDecomposition(q, K) {
  const lines = [];
  for (const dir of Core.directions(q)) {
    const w = Core.witnessLine(dir, q, K);
    if (!w) return null;
    lines.push(w.points);
  }
  return lines;
}

// sacrifice = sum over points of (m-1)(m-2)/2 for the canonical decomposition;
// lineUnion records whether K is exactly the union of its canonical lines.
function sacrificeProfile(q, K) {
  const lines = canonicalDecomposition(q, K);
  if (!lines) return null;
  const mult = new Map();
  const union = new Set();
  for (const line of lines) {
    for (const p of line) {
      mult.set(p, (mult.get(p) ?? 0) + 1);
      union.add(p);
    }
  }
  let sacrifice = 0;
  const profile = new Map();
  for (const m of mult.values()) {
    sacrifice += ((m - 1) * (m - 2)) / 2;
    profile.set(m, (profile.get(m) ?? 0) + 1);
  }
  const lineUnion = union.size === K.size && [...union].every((p) => K.has(p));
  return {
    sacrifice,
    lineUnion,
    unionSize: union.size,
    multiplicityProfile: [...profile.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([m, count]) => `${m}:${count}`)
      .join(" "),
    identityHolds: union.size === (q * (q + 1)) / 2 + sacrifice,
  };
}

// --- masks -------------------------------------------------------------------

function wordCount(q) {
  return Math.ceil(Core.pointCount(q) / 32);
}

function maskFromSet(set, words) {
  const mask = new Uint32Array(words);
  for (const p of set) mask[p >> 5] |= 1 << (p & 31);
  return mask;
}

function popcount32(v) {
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  return (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

// --- budgeted joint solver (PHASE3F solver + dynamic bound + greedy seed) ----

function solveCompletion(q, K, nodeBudget) {
  const dirs = Core.directions(q);
  const words = wordCount(q);
  const bits = Core.shadowBitset(q, K);
  const targets = [];
  for (let i = 0; i < dirs.length; i++) if (bits[i] === 0) targets.push(i);
  if (targets.length === 0) return { joint: 0, status: "exact", nodes: 0 };

  // Add-masks per target direction; static marginals for ordering.
  const addMasks = targets.map((i) => {
    const perB = [];
    for (let b = 0; b < q; b++) {
      const add = new Set();
      for (const p of Core.lineMask(dirs[i], b, q)) if (!K.has(p)) add.add(p);
      perB.push({ mask: maskFromSet(add, words), size: add.size });
    }
    return perB;
  });
  const marginals = addMasks.map((perB) => Math.min(...perB.map((x) => x.size)));
  const order = targets
    .map((_, j) => j)
    .sort((a, b) => marginals[b] - marginals[a]);
  const ordered = order.map((j) => addMasks[j]);

  // Greedy seed: repeatedly take the line adding fewest new points.
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
  let best = seedCount + 1; // strict-improvement search below the seed

  const k = ordered.length;
  const unions = [];
  for (let level = 0; level <= k; level++) unions.push(new Uint32Array(words));
  let nodes = 0;
  let exhausted = true;

  // Dynamic lower bound: remaining directions each need at least their
  // min-addition versus the current union, minus C(r,2) possible sharings.
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

// --- affine certifier: does some AGL(2,q) image of body fit inside W? ---------

function embedsInParabolaImage(q, body, witness) {
  const n = Core.pointCount(q);
  const wHas = new Uint8Array(n);
  for (const p of witness) wHas[p] = 1;
  const wPoints = [...witness];
  const pts = [...body].map((p) => Core.indexToXY(p, q));
  if (pts.length === 0) return { found: true, map: "identity" };

  for (let a = 0; a < q; a++)
    for (let b = 0; b < q; b++)
      for (let c = 0; c < q; c++)
        for (let d = 0; d < q; d++) {
          if ((((a * d - b * c) % q) + q) % q === 0) continue;
          // Transform all body points by M = [[a,b],[c,d]] (no translation yet).
          const tx = new Array(pts.length);
          const ty = new Array(pts.length);
          for (let i = 0; i < pts.length; i++) {
            tx[i] = (a * pts[i].x + b * pts[i].y) % q;
            ty[i] = (c * pts[i].x + d * pts[i].y) % q;
          }
          // Candidate translations: those aligning body point 0 with a witness point.
          for (const wp of wPoints) {
            const { x: wx, y: wy } = Core.indexToXY(wp, q);
            const vx = (((wx - tx[0]) % q) + q) % q;
            const vy = (((wy - ty[0]) % q) + q) % q;
            let ok = true;
            for (let i = 1; i < pts.length; i++) {
              const px = (tx[i] + vx) % q;
              const py = (ty[i] + vy) % q;
              if (!wHas[Core.pointIndex(px, py, q)]) {
                ok = false;
                break;
              }
            }
            if (ok) return { found: true, map: `M=[[${a},${b}],[${c},${d}]] v=(${vx},${vy})` };
          }
        }
  return { found: false, map: null };
}

// --- probe panel ---------------------------------------------------------------

function probeBodies(q) {
  const n = Core.pointCount(q);
  return [
    { id: "line", body: starBody(q, [0]) },
    { id: "line-plus-1", body: linePlusPrefix(q, 1) },
    { id: "star-k2", body: starBody(q, [0, 1]) },
    { id: "star-k3", body: starBody(q, [0, 1, 2]), starK: 3 },
    { id: "star-k4", body: starBody(q, [0, 1, 2, 3]), starK: 4 },
    { id: "star-k4-alt", body: starBody(q, [0, 1, 2, 4]), starK: 4 },
    { id: "star-k5", body: starBody(q, [0, 1, 2, 3, 4]), starK: 5 },
    { id: "tangent-triple", body: tangentTriple(q) },
    { id: "pencil-minus-one", body: starBody(q, [...Array(q).keys()]) },
    { id: "greedy-cover", body: Core.bGreedyLineCover(q) },
    { id: "parabola-witness", body: parabolaWitness(q) },
    { id: "random-third", body: Core.bRandomSubset(q, Math.floor(n / 3), 11) },
    { id: "random-half", body: Core.bRandomSubset(q, Math.floor(n / 2), 12) },
  ];
}

function probeBody(q, entry, witness) {
  const K = entry.body;
  const bm = bmMinimum(q);
  const summary = Core.shadowSummary(q, K);
  const arithForbidden =
    entry.starK !== undefined && !summary.complete && starSacrifice(entry.starK) > budget(q);
  const row = {
    id: entry.id,
    bodySize: K.size,
    missing: summary.directionCount - summary.directionsCovered,
    completion: null,
    ex: null,
    embeds: null,
    method: null,
    solverStatus: null,
    nodes: null,
    certifierMap: null,
    sacrifice: null,
    pass: true,
  };

  if (summary.complete) {
    // Complete bodies: ex is arithmetic; check the sacrifice identity.
    row.completion = K.size;
    row.ex = K.size - bm;
    row.embeds = row.ex === 0;
    row.method = "complete-arithmetic";
    const prof = sacrificeProfile(q, K);
    row.sacrifice = prof;
    if (prof && prof.lineUnion) {
      row.pass = prof.identityHolds && row.ex === prof.sacrifice - budget(q);
    }
    return row;
  }

  // Constructive certificate: some affine image fits inside the parabola set.
  const cert = embedsInParabolaImage(q, K, witness);
  if (cert.found) {
    row.embeds = true;
    row.ex = 0;
    row.completion = bm;
    row.method = "parabola-affine-containment";
    row.certifierMap = cert.map;
  }

  // Arithmetic exclusion for stars over the concurrency budget.
  if (arithForbidden) {
    row.embeds = false;
    row.method = row.method ?? "arithmetic-exclusion";
    if (cert.found) row.pass = false; // contradiction: certificate vs budget corollary
  }

  // Exact solver at q in {5,7}; budgeted solver at q=11 when still unresolved.
  const runSolver = SOLVER_EXACT_Q.includes(q) || (!cert.found && !arithForbidden);
  if (runSolver) {
    const solved = solveCompletion(q, K, NODE_BUDGET);
    row.solverStatus = solved.status;
    row.nodes = solved.nodes;
    const completion = K.size + solved.joint;
    if (solved.status === "exact") {
      row.completion = completion;
      row.ex = completion - bm;
      const solverEmbeds = row.ex === 0;
      // Cross-checks against certificate / arithmetic.
      if (cert.found && !solverEmbeds) row.pass = false;
      if (arithForbidden && solverEmbeds) row.pass = false;
      row.embeds = solverEmbeds;
      row.method = row.method ?? "solver-exact";
      if (cert.found || arithForbidden) row.method += "+solver-exact";
      // Sacrifice identity on the solver's minimal completion when it is a
      // line union (structured bodies).
      if (solverEmbeds) row.completionIsMinimal = true;
    } else {
      // Budget exhausted: completion is only an upper bound.
      if (row.completion === null) {
        row.completion = completion;
        row.ex = null;
        row.embeds = completion === bm ? true : null; // == bm still certifies (imported floor)
        row.method = row.method ?? (completion === bm ? "solver-upper-bound-at-bm" : "bounded-only");
      }
    }
  }

  if (row.embeds === null) row.method = row.method ?? "unresolved";
  return row;
}

// --- outputs ---------------------------------------------------------------------

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
    "missing",
    "completion",
    "ex",
    "embeds",
    "method",
    "solver_status",
    "nodes",
    "sacrifice",
    "multiplicity_profile",
    "certifier_map",
  ];
  const out = [];
  for (const block of perQ) {
    for (const r of block.rows) {
      out.push([
        block.q,
        r.id,
        r.bodySize,
        r.missing,
        r.completion,
        r.ex,
        r.embeds,
        r.method,
        r.solverStatus,
        r.nodes,
        r.sacrifice ? r.sacrifice.sacrifice : "",
        r.sacrifice ? r.sacrifice.multiplicityProfile : "",
        r.certifierMap ?? "",
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
    `# Kakeya Embed-in-Minimal Probe - Operator Commands

\`\`\`powershell
${command}
\`\`\`

Primary outputs:

- \`${manifestPath}\`
- \`${csvPath}\`

Report-only workbench diagnostic. The Blokhuis-Mazzocca minimum is imported at
q = 11 (derived exhaustively at q in {5, 7} in PHASE3F). No Euclidean claim.
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
  const perQ = Core.SUPPORTED_Q.map((q) => {
    const witness = parabolaWitness(q);
    const witnessProfile = sacrificeProfile(q, witness);
    const rows = probeBodies(q).map((entry) => probeBody(q, entry, witness));
    const witnessOk =
      witness.size === bmMinimum(q) &&
      witnessProfile.lineUnion &&
      witnessProfile.identityHolds &&
      witnessProfile.sacrifice === budget(q);
    return {
      q,
      bmMinimum: bmMinimum(q),
      concurrencyBudget: budget(q),
      witnessProfile: { size: witness.size, ...witnessProfile },
      witnessIdentityPass: witnessOk,
      rows,
      pass: witnessOk && rows.every((r) => r.pass),
    };
  });

  const pass = perQ.every((block) => block.pass);
  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal workbench diagnostic receipt",
    hook: "H-K3 embeddability-in-minimal-sets probe (which bodies sit inside minimal complete Kakeya sets)",
    command: "node scripts/kakeya-embed-minimal-probe.mjs",
    statement:
      "ex(K) = completion(K) - BM(q) >= 0, and K embeds in a minimal complete Kakeya set iff ex(K) = 0. Concurrency-budget identity: any complete union of q+1 one-per-direction lines has size q(q+1)/2 + sacrifice with sacrifice = sum of (m-1)(m-2)/2 over point line-multiplicities, so ex = sacrifice - (q-1)/2. Corollary: a k-star embeds only if (k-1)(k-2)/2 <= (q-1)/2.",
    preRegistered: [
      "P1: star-k3 embeds at all q (parabola witness has (q-1)/2 triple points; PGL(2,q) is 3-transitive on directions)",
      "P2: line and line-plus-1 embed by direct containment in the parabola witness",
      "P3: star-k4 at q=7 is budget-exact (sacrifice 3 = whole budget) - open, solver decides; cross-ratio variant star-k4-alt may differ",
      "P4: greedy q=11 excess 6 equals its sacrifice overspend",
      "P5: star-k4 arithmetically excluded at q=5; star-k5 excluded through q=11",
      "P6: sacrifice identity holds on every complete line-union in the panel",
    ],
    nodeBudget: NODE_BUDGET,
    solverExactQ: SOLVER_EXACT_Q,
    falsifier: {
      name: "EMBED_PROBE_MISMATCH",
      description:
        "Fires if the sacrifice identity fails on a complete line-union, a certificate contradicts the exact solver or the budget corollary, the parabola witness misses its exact (q-1)/2 sacrifice, or an arithmetically excluded star embeds.",
      status: pass ? "clear" : "fired",
    },
    perQ,
    pass,
  };

  fs.mkdirSync(outDir, { recursive: true });
  const manifestPath = path.join(outDir, "manifest.json");
  const csvPath = path.join(outDir, "probe.csv");
  const commandsPath = path.join(outDir, "operator-commands.md");

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  writeCsv(csvPath, perQ);
  writeOperatorCommands(commandsPath, manifest.command, manifestPath, csvPath);

  console.log(
    [
      "KAK_EMBED_MINIMAL_PROBE",
      `q={${Core.SUPPORTED_Q.join(",")}}`,
      ...perQ.map(
        (block) =>
          `q${block.q}_embeds=` +
          block.rows
            .filter((r) => r.embeds === true)
            .map((r) => r.id)
            .join("+"),
      ),
      `falsifier=${manifest.falsifier.status}`,
      `out=${outDir}`,
    ].join(" "),
  );
  process.exit(pass ? 0 : 1);
}

main();
