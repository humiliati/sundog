#!/usr/bin/env node
// H-K3 joint-vs-marginal activation gap: for a set D of unlit directions, the
// exact minimal number of added points lighting ALL of D, versus the sum of
// the per-direction marginals (PHASE3E). Any solution contains a full line per
// target direction, so joint(D, K) = min over line choices (b_d) of
// |union(M_d \ K)|, enumerated exactly (with branch-and-bound) inside the
// registered per-q caps.
//
// Proven sandwich (machine-checked here): 0 <= gap <= C(|D|, 2), where
// gap = sum of marginals - joint. Deficit = C(|D|,2) - gap is the forced
// concurrence tax.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3F-JOINT-ACTIVATION-GAP";
const DEFAULT_OUT = path.join("results", "kakeya", "joint-activation-gap");
const BRUTE_FORCE_Q = 5;
const BRUTE_FORCE_MAX_JOINT = 3;
// Pre-registered enumeration caps: full subset lattice where (q+1 choose k)*q^k
// stays enumerable, |D| <= 4 at q = 11 (the full pencil there is carried by the
// constructed parabola witness, not enumeration).
const SUBSET_CAP = { 5: 6, 7: 8, 11: 4 };

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
  return `Usage: node scripts/kakeya-joint-activation-gap.mjs [options]

Options:
  --out <dir>         Output directory. Default: ${DEFAULT_OUT}
  --help              Show this message.
`;
}

// --- word-mask helpers -------------------------------------------------------

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

function popcountWords(mask) {
  let sum = 0;
  for (let w = 0; w < mask.length; w++) sum += popcount32(mask[w]);
  return sum;
}

function binomSmall(n, k) {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 0; i < k; i++) result = (result * (n - i)) / (i + 1);
  return result;
}

// --- exact joint solver --------------------------------------------------------

// addMasks[j][b] = mask of lineMask(dirs[targets[j]], b) \ K. DFS over one
// intercept choice per target direction; prune on the running union count.
function solveJoint(q, targets, addMasks, words) {
  let best = Infinity;
  let bestChoice = null;
  const choice = new Array(targets.length).fill(-1);
  const unions = [];
  for (let level = 0; level <= targets.length; level++) unions.push(new Uint32Array(words));
  let nodes = 0;

  function rec(level, count) {
    if (count >= best) return;
    if (level === targets.length) {
      best = count;
      bestChoice = [...choice];
      return;
    }
    const prev = unions[level];
    const next = unions[level + 1];
    for (let b = 0; b < q; b++) {
      nodes++;
      const add = addMasks[level][b];
      let newCount = 0;
      for (let w = 0; w < words; w++) {
        next[w] = prev[w] | add[w];
        newCount += popcount32(next[w]);
      }
      if (newCount >= best) continue;
      choice[level] = b;
      rec(level + 1, newCount);
    }
  }

  rec(0, 0);
  return { joint: best, intercepts: bestChoice, nodes };
}

// --- bodies (PHASE3E panel continuity: same seeds) -------------------------------

function starBody(q, k) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (let i = 0; i < k; i++) for (const p of Core.lineMask(dirs[i], 0, q)) body.add(p);
  return body;
}

function panelBodies(q) {
  const n = Core.pointCount(q);
  const dirs = Core.directions(q);
  return [
    { id: "empty", body: Core.bEmpty() },
    { id: "single-line", body: new Set(Core.lineMask(dirs[0], 0, q)) },
    { id: "star-k2", body: starBody(q, 2) },
    { id: `star-k${q - 1}`, body: starBody(q, q - 1) },
    { id: "random-third", body: Core.bRandomSubset(q, Math.floor(n / 3), 11) },
    { id: "random-half", body: Core.bRandomSubset(q, Math.floor(n / 2), 12) },
  ];
}

// --- parabola witness (odd q) ----------------------------------------------------
// Tangent to y = x^2 at t: y = 2t x - t^2 (slope 2t, intercept -t^2); for odd q
// the slopes 2t are a bijection onto the finite slopes and no three tangents
// are concurrent. Adding the vertical x = 0 completes the pencil at total size
// q(q+1)/2 + (q-1)/2 = the Blokhuis-Mazzocca minimum (imported, odd q).
function parabolaWitness(q) {
  const dirs = Core.directions(q);
  const body = new Set();
  for (let t = 0; t < q; t++) {
    const slope = (2 * t) % q;
    const intercept = ((-(t * t)) % q + q) % q;
    for (const p of Core.lineMask(dirs[slope], intercept, q)) body.add(p);
  }
  for (const p of Core.lineMask(dirs[q], 0, q)) body.add(p); // vertical x = 0
  return body;
}

function bmMinimum(q) {
  return (q * (q + 1)) / 2 + (q - 1) / 2;
}

// --- per-body analysis ------------------------------------------------------------

function activationOf(dir, q, K) {
  let best = Infinity;
  for (let b = 0; b < q; b++) {
    let missing = 0;
    for (const p of Core.lineMask(dir, b, q)) if (!K.has(p)) missing++;
    best = Math.min(best, missing);
  }
  return best;
}

function* subsetsUpTo(items, cap) {
  const k = items.length;
  const total = 1 << k;
  for (let bits = 1; bits < total; bits++) {
    const subset = [];
    for (let i = 0; i < k; i++) if (bits & (1 << i)) subset.push(items[i]);
    if (subset.length <= cap) yield subset;
  }
}

function* combinations(pool, k, start = 0, acc = []) {
  if (acc.length === k) {
    yield acc;
    return;
  }
  for (let i = start; i <= pool.length - (k - acc.length); i++) {
    yield* combinations(pool, k, i + 1, [...acc, pool[i]]);
  }
}

function analyzeBody(q, entry, cap) {
  const dirs = Core.directions(q);
  const words = wordCount(q);
  const K = entry.body;
  const bits = Core.shadowBitset(q, K);
  const activations = dirs.map((dir) => activationOf(dir, q, K));
  const missing = [];
  for (let i = 0; i < dirs.length; i++) if (bits[i] === 0) missing.push(i);

  // Per-direction add-masks, ordered by descending activation for pruning.
  const addMaskFor = (dirIndex) => {
    const masks = [];
    for (let b = 0; b < q; b++) {
      const add = new Set();
      for (const p of Core.lineMask(dirs[dirIndex], b, q)) if (!K.has(p)) add.add(p);
      masks.push(maskFromSet(add, words));
    }
    return masks;
  };
  const addMasksByDir = new Map(missing.map((i) => [i, addMaskFor(i)]));

  const rows = [];
  let checksPass = true;
  let nodesTotal = 0;
  let bruteSubsets = 0;
  let brutePass = true;

  for (const D of subsetsUpTo(missing, cap)) {
    const ordered = [...D].sort((a, b2) => activations[b2] - activations[a]);
    const addMasks = ordered.map((i) => addMasksByDir.get(i));
    const { joint, intercepts, nodes } = solveJoint(q, ordered, addMasks, words);
    nodesTotal += nodes;

    const k = D.length;
    const marginalSum = D.reduce((sum, i) => sum + activations[i], 0);
    const maxMarginal = Math.max(...D.map((i) => activations[i]));
    const gap = marginalSum - joint;
    const pairBound = binomSmall(k, 2);
    const deficit = pairBound - gap;

    // Sandwich + gap bound (both proven; machine-checked anyway).
    let ok = joint >= maxMarginal && joint <= marginalSum && gap >= 0 && deficit >= 0;
    // Singleton cross-check against the PHASE3E metric.
    if (k === 1) ok = ok && joint === activations[D[0]];

    // Constructive witness: applying the argmin intercepts lights all of D.
    const enlarged = new Set(K);
    for (let j = 0; j < ordered.length; j++) {
      for (const p of Core.lineMask(dirs[ordered[j]], intercepts[j], q)) enlarged.add(p);
    }
    const litBits = Core.shadowBitset(q, enlarged);
    ok = ok && D.every((i) => litBits[i] === 1) && enlarged.size === K.size + joint;

    // Independent q=5 brute force for small joints: no (joint-1)-point
    // addition lights all of D.
    if (q === BRUTE_FORCE_Q && k >= 2 && joint <= BRUTE_FORCE_MAX_JOINT) {
      const complement = [];
      for (let p = 0; p < Core.pointCount(q); p++) if (!K.has(p)) complement.push(p);
      for (const subset of combinations(complement, joint - 1)) {
        bruteSubsets++;
        const trial = new Set(K);
        for (const p of subset) trial.add(p);
        const trialBits = Core.shadowBitset(q, trial);
        if (D.every((i) => trialBits[i] === 1)) {
          brutePass = false;
          break;
        }
      }
    }

    checksPass = checksPass && ok;
    rows.push({
      directions: D.map((i) => dirs[i].label),
      k,
      marginalSum,
      maxMarginal,
      joint,
      gap,
      pairBound,
      deficit,
      witnessIntercepts: ordered.map((i, j) => `${dirs[i].label}:${intercepts[j]}`),
      pass: ok,
    });
  }

  const multi = rows.filter((r) => r.k >= 2);
  return {
    id: entry.id,
    bodySize: K.size,
    missingCount: missing.length,
    activations: missing.map((i) => ({ direction: dirs[i].label, cost: activations[i] })),
    rowCount: rows.length,
    maxGap: multi.length ? Math.max(...multi.map((r) => r.gap)) : 0,
    maxDeficit: multi.length ? Math.max(...multi.map((r) => r.deficit)) : 0,
    tightRows: multi.filter((r) => r.deficit === 0).length,
    multiRows: multi.length,
    nodesTotal,
    bruteSubsets,
    checks: { pass: checksPass && brutePass, bruteForce: q === BRUTE_FORCE_Q ? brutePass : null },
    rows,
  };
}

// --- anchors ---------------------------------------------------------------------

function checkAnchors(q, cap, bodies) {
  const dirs = Core.directions(q);
  const empty = bodies.find((b) => b.id === "empty");

  // (i) Finite-slope tightness from empty (proven via parabola tangents,
  // odd q): every D avoiding "inf" with k >= 2 has deficit exactly 0.
  const finiteSlopeTight = empty.rows
    .filter((r) => r.k >= 2 && !r.directions.includes("inf"))
    .every((r) => r.deficit === 0);

  // (ii) Parabola witness: complete Kakeya set at the imported BM minimum.
  const witness = parabolaWitness(q);
  const witnessSummary = Core.shadowSummary(q, witness);
  const witnessOk = witness.size === bmMinimum(q) && witnessSummary.complete;

  // (iii) Full pencil: exact where enumerated (cap covers q + 1), otherwise
  // witnessed upper bound at q = 11.
  const fullRow = empty.rows.find((r) => r.k === q + 1);
  const fullPencil = {
    enumerated: Boolean(fullRow),
    joint: fullRow ? fullRow.joint : null,
    bmMinimum: bmMinimum(q),
    witnessedUpperBound: witness.size,
    deficit: fullRow ? fullRow.deficit : null,
    expectedDeficit: (q - 1) / 2,
  };
  const fullOk = fullRow
    ? fullRow.joint === bmMinimum(q) && fullRow.deficit === (q - 1) / 2
    : witnessOk;

  // (iv) Greedy cover comparison (measured, not an anchor gate).
  const greedySize = Core.bGreedyLineCover(q).size;

  return {
    finiteSlopeTight,
    parabolaWitness: { size: witness.size, complete: witnessSummary.complete, pass: witnessOk },
    fullPencil,
    fullPencilPass: fullOk,
    greedy: { size: greedySize, bmMinimum: bmMinimum(q), beatsGreedy: bmMinimum(q) < greedySize },
    pass: finiteSlopeTight && witnessOk && fullOk,
  };
}

// --- outputs -----------------------------------------------------------------------

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
    "directions",
    "k",
    "marginal_sum",
    "max_marginal",
    "joint",
    "gap",
    "pair_bound",
    "deficit",
    "witness_intercepts",
  ];
  const out = [];
  for (const block of perQ) {
    for (const body of block.bodies) {
      for (const r of body.rows) {
        out.push([
          block.q,
          body.id,
          body.bodySize,
          r.directions.join(" "),
          r.k,
          r.marginalSum,
          r.maxMarginal,
          r.joint,
          r.gap,
          r.pairBound,
          r.deficit,
          r.witnessIntercepts.join(" "),
        ]);
      }
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
    `# Kakeya Joint Activation Gap - Operator Commands

\`\`\`powershell
${command}
\`\`\`

Primary outputs:

- \`${manifestPath}\`
- \`${csvPath}\`

Report-only workbench diagnostic. The Blokhuis-Mazzocca planar minimum is an
imported literature anchor (machine-verified as exact enumeration at q in
{5, 7}, witnessed only at q = 11). No Euclidean Kakeya claim.
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
    const cap = Math.min(SUBSET_CAP[q], Core.directionCount(q));
    const bodies = panelBodies(q).map((entry) => analyzeBody(q, entry, cap));
    const anchors = checkAnchors(q, cap, bodies);
    return {
      q,
      subsetCap: cap,
      bodies,
      anchors,
      pass: bodies.every((b) => b.checks.pass) && anchors.pass,
    };
  });

  const pass = perQ.every((block) => block.pass);
  const sandwich = perQ.every((block) => block.bodies.every((b) => b.checks.pass));
  const anchors = perQ.every((block) => block.anchors.pass);
  const bruteQ5 = perQ
    .filter((block) => block.q === BRUTE_FORCE_Q)
    .every((block) => block.bodies.every((b) => b.checks.bruteForce !== false));

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal workbench diagnostic receipt",
    hook: "H-K3 joint-vs-marginal activation gap (interaction structure of shared added points)",
    command: "node scripts/kakeya-joint-activation-gap.mjs",
    statement:
      "For a set D of unlit directions of a body K, joint(D, K) = min over per-direction line choices of |union(M_d \\ K)| is the exact minimal number of added points lighting all of D. Gap = sum of marginals - joint satisfies 0 <= gap <= C(|D|, 2) for any body (pairwise inclusion-exclusion: distinct-direction lines share at most one point). Deficit = C(|D|,2) - gap is the forced concurrence tax.",
    provenAnchors: [
      "sandwich: max marginal <= joint <= sum of marginals",
      "gap bound: 0 <= gap <= C(k, 2) for any body and any D",
      "finite-slope tightness from empty (odd q): parabola tangents (slope 2t, no three concurrent) achieve gap = C(k, 2) for every all-finite-slope D",
      "parabola tangents + one vertical = complete Kakeya set of size q(q+1)/2 + (q-1)/2",
    ],
    importedAnchor:
      "Blokhuis-Mazzocca: q(q+1)/2 + (q-1)/2 is the exact planar minimum for odd q. Machine-verified as exhaustive enumeration at q in {5, 7}; at q = 11 the workbench only witnesses the upper bound - equality is cited, not derived. Bibliographic pin owed to KAKEYA_LITPASS_MEMO.md.",
    subsetCaps: SUBSET_CAP,
    falsifier: {
      name: "JOINT_GAP_MISMATCH",
      description:
        "Fires if the sandwich or gap bound fails on any enumerated row, a singleton disagrees with the PHASE3E marginal, a joint witness fails to light its direction set at the claimed cost, the finite-slope tightness or full-pencil/parabola anchors fail, or the q=5 exhaustive search finds a smaller joint addition.",
      status: pass ? "clear" : "fired",
    },
    perQ: perQ.map((block) => ({
      ...block,
      bodies: block.bodies.map((b) => ({ ...b, rows: undefined })), // full rows live in the CSV
    })),
    pass,
  };

  fs.mkdirSync(outDir, { recursive: true });
  const manifestPath = path.join(outDir, "manifest.json");
  const csvPath = path.join(outDir, "gap.csv");
  const commandsPath = path.join(outDir, "operator-commands.md");

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  writeCsv(csvPath, perQ);
  writeOperatorCommands(commandsPath, manifest.command, manifestPath, csvPath);

  const rowTotal = perQ.reduce(
    (sum, block) => sum + block.bodies.reduce((s, b) => s + b.rowCount, 0),
    0,
  );
  console.log(
    [
      "KAK_JOINT_ACTIVATION_GAP",
      `q={${Core.SUPPORTED_Q.join(",")}}`,
      `rows=${rowTotal}`,
      `sandwich=${sandwich ? "pass" : "fail"}`,
      `anchors=${anchors ? "pass" : "fail"}`,
      `bruteforce_q5=${bruteQ5 ? "pass" : "fail"}`,
      ...perQ.map(
        (block) =>
          `q${block.q}_full_pencil=${
            block.anchors.fullPencil.enumerated
              ? block.anchors.fullPencil.joint
              : "<=" + block.anchors.fullPencil.witnessedUpperBound
          }/greedy=${block.anchors.greedy.size}`,
      ),
      `falsifier=${manifest.falsifier.status}`,
      `out=${outDir}`,
    ].join(" "),
  );
  process.exit(pass ? 0 : 1);
}

main();
