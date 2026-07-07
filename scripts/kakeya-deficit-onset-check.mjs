#!/usr/bin/env node
// H-K3 deficit-onset check: the PHASE3F open question for inf-containing
// direction sets, closed by axis avoidance.
//
// A parabola is a conic tangent to the line at infinity at its AXIS direction;
// its tangent family covers every direction except the axis. The standard
// y = x^2 has axis = inf (hence PHASE3F's finite-slope-only proof). The affine
// map (x, y) -> (y, x + s*y) sends inf to slope s, so a rotated parabola's
// tangents cover all directions except s. Therefore, for any direction set D
// with |D| = k <= q, pick an axis a not in D and take the k tangents with
// directions in D: no three concurrent, union size = kq - C(k,2), deficit 0.
// Self-certifying: joint >= kq - C(k,2) from empty is the proven PHASE3F
// bound, so the construction is optimal without any search.
//
// Onset: at k = q + 1 no axis remains (dually, a (q+2)-arc through the
// infinity point, impossible for odd q); the tax is (q-1)/2 - measured
// exhaustively at q in {5, 7} (PHASE3F), imported (Blokhuis-Mazzocca, litpass
// addendum 2026-07-06) at q = 11.

import fs from "node:fs";
import path from "node:path";
import * as Core from "../kakeya/kakeya-core.js";

const ARTIFACT_ID = "KAK-PHASE3H-DEFICIT-ONSET";
const DEFAULT_OUT = path.join("results", "kakeya", "deficit-onset");
const EXHAUSTIVE_Q = [5, 7]; // independent brute-force cross-check fields

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
  return `Usage: node scripts/kakeya-deficit-onset-check.mjs [options]

Options:
  --out <dir>         Output directory. Default: ${DEFAULT_OUT}
  --help              Show this message.
`;
}

function bmMinimum(q) {
  return (q * (q + 1)) / 2 + (q - 1) / 2;
}

function binom2(k) {
  return (k * (k - 1)) / 2;
}

// --- axis-avoiding tangent family ---------------------------------------------

// Direction index of the line through two distinct points.
function directionIndexOf(pointA, pointB, q) {
  const a = Core.indexToXY(pointA, q);
  const b = Core.indexToXY(pointB, q);
  if (a.x === b.x) return q; // inf
  const dx = (((b.x - a.x) % q) + q) % q;
  const dy = (((b.y - a.y) % q) + q) % q;
  // slope = dy / dx mod q
  let inv = 1;
  for (let i = 1; i < q; i++) {
    if ((dx * i) % q === 1) {
      inv = i;
      break;
    }
  }
  return (dy * inv) % q;
}

// Tangent family of a parabola with axis direction dirs[axisIndex]:
// Map from direction index (every index except axisIndex) to the tangent line
// (Set of point indices). Standard family (axis = inf): tangent at t is
// y = 2tx - t^2. For axis = slope s, apply (x, y) -> (y, x + s*y).
function axisFamily(q, axisIndex) {
  const family = new Map();
  for (let t = 0; t < q; t++) {
    const points = [];
    for (let x = 0; x < q; x++) {
      const y = ((2 * t * x - t * t) % q + q) % q;
      if (axisIndex === q) {
        points.push(Core.pointIndex(x, y, q));
      } else {
        const s = axisIndex;
        const nx = y;
        const ny = (x + s * y) % q;
        points.push(Core.pointIndex(nx, ny, q));
      }
    }
    const dirIndex = directionIndexOf(points[0], points[1], q);
    family.set(dirIndex, new Set(points));
  }
  return family;
}

// Family sanity: q distinct lines of q points each, directions bijective onto
// all indices except the axis.
function familyValid(q, axisIndex, family) {
  if (family.size !== q) return false;
  for (const [dirIndex, line] of family) {
    if (dirIndex === axisIndex || line.size !== q) return false;
  }
  return true;
}

// --- subset sweep ---------------------------------------------------------------

function checkQ(q) {
  const dirCount = Core.directionCount(q);
  const families = [];
  let familiesValid = true;
  for (let a = 0; a < dirCount; a++) {
    const fam = axisFamily(q, a);
    familiesValid = familiesValid && familyValid(q, a, fam);
    families.push(fam);
  }

  // Every nonempty D with |D| <= q: canonical axis = lowest index not in D.
  let subsetsChecked = 0;
  let constructionPass = true;
  const perK = new Map();
  const total = 1 << dirCount;
  for (let bits = 1; bits < total; bits++) {
    const D = [];
    for (let i = 0; i < dirCount; i++) if (bits & (1 << i)) D.push(i);
    const k = D.length;
    if (k > q) continue;
    let axis = -1;
    for (let i = 0; i < dirCount; i++) {
      if (!(bits & (1 << i))) {
        axis = i;
        break;
      }
    }
    const fam = families[axis];
    const union = new Set();
    for (const i of D) for (const p of fam.get(i)) union.add(p);
    const ok = union.size === k * q - binom2(k);
    constructionPass = constructionPass && ok;
    subsetsChecked++;
    perK.set(k, (perK.get(k) ?? 0) + (ok ? 1 : 0));
  }

  // Full-pencil boundary: for EVERY axis, q tangents + the cheapest axis-line
  // land exactly on the Blokhuis-Mazzocca minimum, complete.
  const dirs = Core.directions(q);
  let axisCompletions = true;
  const completions = [];
  for (let a = 0; a < dirCount; a++) {
    const union = new Set();
    for (const line of families[a].values()) for (const p of line) union.add(p);
    let bestAdd = Infinity;
    let bestB = -1;
    for (let b = 0; b < q; b++) {
      let add = 0;
      for (const p of Core.lineMask(dirs[a], b, q)) if (!union.has(p)) add++;
      if (add < bestAdd) {
        bestAdd = add;
        bestB = b;
      }
    }
    const complete = new Set(union);
    for (const p of Core.lineMask(dirs[a], bestB, q)) complete.add(p);
    const summary = Core.shadowSummary(q, complete);
    const ok =
      union.size === (q * (q + 1)) / 2 &&
      bestAdd === (q - 1) / 2 &&
      complete.size === bmMinimum(q) &&
      summary.complete;
    axisCompletions = axisCompletions && ok;
    completions.push({
      axis: dirs[a].label,
      tangentUnion: union.size,
      lastLineCost: bestAdd,
      total: complete.size,
      complete: summary.complete,
      pass: ok,
    });
  }

  return {
    q,
    familiesValid,
    subsetsChecked,
    perK: [...perK.entries()].sort((x, y) => x[0] - y[0]).map(([k, n]) => `${k}:${n}`).join(" "),
    constructionPass,
    axisCompletions,
    completions,
    pass: familiesValid && constructionPass && axisCompletions,
  };
}

// --- independent exhaustive cross-check (q in {5, 7}) ----------------------------

function exhaustiveCrossCheck(q) {
  const dirCount = Core.directionCount(q);
  const dirs = Core.directions(q);
  const n = Core.pointCount(q);
  const words = Math.ceil(n / 32);
  const lineMasks = dirs.map((d) => {
    const perB = [];
    for (let b = 0; b < q; b++) {
      const mask = new Uint32Array(words);
      for (const p of Core.lineMask(d, b, q)) mask[p >> 5] |= 1 << (p & 31);
      perB.push(mask);
    }
    return perB;
  });
  const pop32 = (v) => {
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
  };

  let checked = 0;
  let pass = true;
  const total = 1 << dirCount;
  for (let bits = 1; bits < total; bits++) {
    const D = [];
    for (let i = 0; i < dirCount; i++) if (bits & (1 << i)) D.push(i);
    const k = D.length;
    if (k > q) continue;
    // Exact joint from empty by full enumeration with pruning.
    let best = Infinity;
    const unions = [];
    for (let level = 0; level <= k; level++) unions.push(new Uint32Array(words));
    const rec = (level, count) => {
      if (count >= best) return;
      if (level === k) {
        best = count;
        return;
      }
      const prev = unions[level];
      const next = unions[level + 1];
      for (let b = 0; b < q; b++) {
        const m = lineMasks[D[level]][b];
        let c = 0;
        for (let w = 0; w < words; w++) {
          next[w] = prev[w] | m[w];
          c += pop32(next[w]);
        }
        if (c < best) rec(level + 1, c);
      }
    };
    rec(0, 0);
    pass = pass && best === k * q - binom2(k);
    checked++;
  }
  return { q, checked, pass };
}

// --- outputs ----------------------------------------------------------------------

function writeOperatorCommands(file, command, manifestPath) {
  fs.writeFileSync(
    file,
    `# Kakeya Deficit Onset - Operator Commands

\`\`\`powershell
${command}
\`\`\`

Primary output:

- \`${manifestPath}\`

Report-only workbench receipt. The k = q + 1 onset value at q = 11 imports the
Blokhuis-Mazzocca minimum (litpass addendum 2026-07-06). No Euclidean claim.
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
  const exhaustive = EXHAUSTIVE_Q.map(exhaustiveCrossCheck);
  const pass = perQ.every((b) => b.pass) && exhaustive.every((e) => e.pass);

  const manifest = {
    artifactId: ARTIFACT_ID,
    generatedAt: new Date().toISOString(),
    status: "internal workbench theorem receipt",
    hook: "H-K3 deficit onset: axis avoidance dissolves the inf-dichotomy; onset exactly at k = q + 1",
    command: "node scripts/kakeya-deficit-onset-check.mjs",
    statement:
      "For ANY direction set D with |D| = k <= q (odd q), the deficit is 0: the k tangents (directions in D) of a parabola whose axis avoids D form a family with no three concurrent, so the union has exactly kq - C(k,2) points, matching the proven PHASE3F lower bound - self-certifying optimality, no search. The onset is exactly k = q + 1, where no axis remains (dually: a (q+2)-arc through the infinity point, impossible for odd q), with tax (q-1)/2.",
    construction:
      "Standard tangents y = 2tx - t^2 (axis = inf); for axis = slope s apply the affine map (x, y) -> (y, x + s*y), which sends inf to slope s and is a bijection on the remaining directions.",
    importedAnchor:
      "The k = q + 1 tax value at q = 11 rests on the Blokhuis-Mazzocca minimum (pinned: Building Bridges, Bolyai Soc. Math. Studies 19, Springer 2008, pp. 205-218; arXiv:0911.4370). At q in {5, 7} it is derived exhaustively (PHASE3F).",
    falsifier: {
      name: "DEFICIT_ONSET_MISMATCH",
      description:
        "Fires if any axis family is not a direction-bijection of q full lines, any D with k <= q misses the exact kq - C(k,2) union size, the exhaustive q in {5,7} cross-check disagrees with the closed form, or any axis completion misses the Blokhuis-Mazzocca size or completeness.",
      status: pass ? "clear" : "fired",
    },
    perQ,
    exhaustiveCrossCheck: exhaustive,
    pass,
  };

  fs.mkdirSync(outDir, { recursive: true });
  const manifestPath = path.join(outDir, "manifest.json");
  const commandsPath = path.join(outDir, "operator-commands.md");
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  writeOperatorCommands(commandsPath, manifest.command, manifestPath);

  console.log(
    [
      "KAK_DEFICIT_ONSET",
      `q={${Core.SUPPORTED_Q.join(",")}}`,
      ...perQ.map((b) => `q${b.q}_subsets=${b.subsetsChecked}`),
      `construction=${perQ.every((b) => b.constructionPass) ? "pass" : "fail"}`,
      ...exhaustive.map((e) => `exhaustive_q${e.q}=${e.pass ? "pass" : "fail"}`),
      `axis_completions=${perQ.every((b) => b.axisCompletions) ? "pass" : "fail"}`,
      `falsifier=${manifest.falsifier.status}`,
      `out=${outDir}`,
    ].join(" "),
  );
  process.exit(pass ? 0 : 1);
}

main();
