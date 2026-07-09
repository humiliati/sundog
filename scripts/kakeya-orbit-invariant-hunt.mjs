#!/usr/bin/env node
// PHASE3M step 1 - orbit-invariant hunt for the 4-star excess level.
//
// Tabulates candidate PGL-invariant quadratic-character data per cross-ratio
// orbit across q in {5..37} and tests which invariants separate the KNOWN
// levels. Epistemics of the level data (from PHASE3I/3J/3K/3L):
//   EXACT   - solver-certified (q <= 17)
//   DEF-LOW - construction achieved (q-3)/2: definitive refutation of high
//   UB-HIGH - construction stalled at (q-1)/2: unreliable within +1
// Rule mining uses only EXACT + DEF-LOW rows; surviving rules then emit
// predictions for every UB-HIGH row (tested by the PHASE3M descent
// extension and the in-flight q=19 exact solves).
//
// Candidate invariants per orbit (canonical lambda = min of the six-set):
//   S  = multiset {chi(l), chi(1-l), chi(l/(l-1))}  (class-invariant, checked)
//   cJ = chi(j)        (j-invariant; skipped when j = 0)
//   cJ2 = chi(j-1728)  (skipped when 0)
//   cM1 = chi(-1), c2 = chi(2)  (field-level bits)
// plus the orbit type (harmonic / equianharmonic / generic).

import fs from "node:fs";
import path from "node:path";

const DEFAULT_OUT = path.join("results", "kakeya", "orbit-invariant-hunt");
const FIELDS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
const INF = -1;

// Known levels: label -> [level, epistemics]
const KNOWN = {
  5: { harmonic: ["HIGH", "EXACT"] },
  7: { harmonic: ["LOW", "EXACT"], equianharmonic: ["HIGH", "EXACT"] },
  11: { harmonic: ["LOW", "EXACT"], generic: ["LOW", "EXACT"] },
  13: {
    harmonic: ["HIGH", "EXACT"],
    equianharmonic: ["LOW", "EXACT"],
    generic: ["LOW", "EXACT"],
  },
  17: {
    harmonic: ["LOW", "EXACT"],
    "generic-a": ["LOW", "EXACT"],
    "generic-b": ["LOW", "EXACT"],
  },
  19: {
    // 19-harmonic promoted UB-HIGH -> EXACT by the PHASE3M q=19 B&B solve
    // (ex=9, solverExact, falsifier clear; 2026-07-08).
    harmonic: ["HIGH", "EXACT"],
    equianharmonic: ["HIGH", "UB-HIGH"],
    "generic-a": ["LOW", "DEF-LOW"],
    "generic-b": ["LOW", "DEF-LOW"],
  },
  23: {
    harmonic: ["LOW", "DEF-LOW"],
    "generic-a": ["LOW", "DEF-LOW"],
    "generic-b": ["LOW", "DEF-LOW"],
    "generic-c": ["HIGH", "UB-HIGH"],
  },
  29: {
    harmonic: ["HIGH", "UB-HIGH"],
    "generic-a": ["LOW", "DEF-LOW"],
    "generic-b": ["LOW", "DEF-LOW"],
    "generic-c": ["LOW", "DEF-LOW"],
    "generic-d": ["LOW", "DEF-LOW"],
  },
  31: {
    harmonic: ["LOW", "DEF-LOW"],
    equianharmonic: ["HIGH", "UB-HIGH"],
    "generic-a": ["LOW", "DEF-LOW"],
    "generic-b": ["HIGH", "UB-HIGH"],
    "generic-c": ["LOW", "DEF-LOW"],
    "generic-d": ["LOW", "DEF-LOW"],
  },
  37: {
    harmonic: ["HIGH", "UB-HIGH"],
    equianharmonic: ["LOW", "DEF-LOW"],
    "generic-a": ["LOW", "DEF-LOW"],
    "generic-b": ["LOW", "DEF-LOW"],
    "generic-c": ["LOW", "DEF-LOW"],
    "generic-d": ["LOW", "DEF-LOW"],
    "generic-e": ["LOW", "DEF-LOW"],
  },
};

function mod(x, q) {
  return ((x % q) + q) % q;
}
function inv(x, q) {
  for (let i = 1; i < q; i++) if ((x * i) % q === 1) return i;
  throw new Error(`no inverse of ${x} mod ${q}`);
}
function chi(x, q) {
  // Legendre symbol via Euler's criterion (q odd prime).
  const e = (q - 1) / 2;
  let base = mod(x, q);
  if (base === 0) return 0;
  let result = 1;
  let exp = e;
  while (exp > 0) {
    if (exp & 1) result = (result * base) % q;
    base = (base * base) % q;
    exp >>= 1;
  }
  return result === 1 ? 1 : -1;
}

function sixSet(l, q) {
  const s = new Set();
  s.add(l);
  s.add(inv(l, q));
  const om = mod(1 - l, q);
  s.add(om);
  s.add(inv(om, q));
  s.add(mod(l * inv(mod(l - 1, q), q), q));
  s.add(mod(mod(l - 1, q) * inv(l, q), q));
  return [...s].sort((x, y) => x - y);
}
function signature(l, q) {
  return [chi(l, q), chi(mod(1 - l, q), q), chi(mod(l * inv(mod(l - 1, q), q), q), q)]
    .sort((a, b) => b - a)
    .join("");
}
function jInvariant(l, q) {
  const t = mod(l * l - l + 1, q);
  const num = mod(256 * t * t * t, q);
  const den = mod(l * l * mod(l - 1, q) * mod(l - 1, q), q);
  return (num * inv(den, q)) % q;
}
function labelOf(l, q) {
  const six = sixSet(l, q);
  const harmonic = [2, mod((q + 1) / 2, q), q - 1].sort((x, y) => x - y);
  if (six.length === 3 && six.join(",") === harmonic.join(",")) return "harmonic";
  if (six.length === 2 && six.every((v) => mod(v * v - v + 1, q) === 0)) return "equianharmonic";
  return "generic";
}

function main() {
  const outDir = process.argv.includes("--out")
    ? process.argv[process.argv.indexOf("--out") + 1]
    : DEFAULT_OUT;

  const rows = [];
  let signatureClassInvariant = true;

  for (const q of FIELDS) {
    // Classes over F_q \ {0,1}, canonical lambda = min of six-set.
    const seen = new Set();
    const classes = [];
    for (let l = 2; l < q; l++) {
      const six = sixSet(l, q);
      const key = six.join(",");
      if (seen.has(key)) continue;
      seen.add(key);
      // Signature must be constant across the class (falsifier leg).
      const sigs = new Set(six.map((v) => signature(v, q)));
      if (sigs.size !== 1) signatureClassInvariant = false;
      classes.push({ lambda: six[0], six, key, label: labelOf(l, q), sig: signature(six[0], q) });
    }
    // Disambiguate generic labels by six-set key order (matches 3K/3L).
    const byBase = new Map();
    for (const c of [...classes].sort((a, b) => (a.key < b.key ? -1 : 1)))
      byBase.set(c.label, (byBase.get(c.label) ?? 0) + 1);
    const counters = new Map();
    for (const c of [...classes].sort((a, b) => (a.key < b.key ? -1 : 1))) {
      if (byBase.get(c.label) > 1) {
        const n = (counters.get(c.label) ?? 0) + 1;
        counters.set(c.label, n);
        c.label = `${c.label}-${String.fromCharCode(96 + n)}`;
      }
    }

    for (const c of classes) {
      const j = jInvariant(c.lambda, q);
      const known = KNOWN[q]?.[c.label] ?? [null, null];
      rows.push({
        q,
        orbit: c.label,
        lambda: c.lambda,
        sixSet: c.key,
        sig: c.sig,
        j,
        cJ: j === 0 ? "0" : String(chi(j, q)),
        cJ2: mod(j - 1728, q) === 0 ? "0" : String(chi(mod(j - 1728, q), q)),
        cM1: chi(-1, q),
        c2: chi(2, q),
        level: known[0],
        epistemics: known[1],
      });
    }
  }

  // Rule mining on EXACT + DEF-LOW rows.
  const train = rows.filter((r) => r.epistemics === "EXACT" || r.epistemics === "DEF-LOW");
  const candidates = {
    sig: (r) => r.sig,
    "sig+type": (r) => `${r.sig}|${r.orbit.replace(/-.*/, "")}`,
    cJ: (r) => r.cJ,
    cJ2: (r) => r.cJ2,
    "cJ+cJ2": (r) => `${r.cJ}|${r.cJ2}`,
    "sig+cJ": (r) => `${r.sig}|${r.cJ}`,
    "sig+cJ2": (r) => `${r.sig}|${r.cJ2}`,
    "cM1+c2+cJ": (r) => `${r.cM1}|${r.c2}|${r.cJ}`,
  };
  const mining = {};
  for (const [name, f] of Object.entries(candidates)) {
    const map = new Map(); // value -> first row with that value
    let consistent = true;
    const collisions = [];
    for (const r of train) {
      const v = f(r);
      if (!map.has(v)) map.set(v, r);
      else if (map.get(v).level !== r.level) {
        consistent = false;
        collisions.push([
          `${map.get(v).q}/${map.get(v).orbit}(${map.get(v).level})`,
          `${r.q}/${r.orbit}(${r.level})`,
          `@${name}=${v}`,
        ]);
      }
    }
    mining[name] = { consistent, collisions };
    if (consistent) {
      mining[name].predictions = rows
        .filter((r) => r.epistemics === "UB-HIGH")
        .map((r) => `${r.q}/${r.orbit}: ${map.get(f(r))?.level ?? "?(unseen)"}`);
    }
  }

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(
    path.join(outDir, "hunt.json"),
    JSON.stringify({ generatedAt: new Date().toISOString(), signatureClassInvariant, rows, mining }, null, 2) + "\n",
  );

  console.log(`signature class-invariance: ${signatureClassInvariant}`);
  console.log("q  orbit           lam  sig  j     cJ  cJ2  level  epistemics");
  for (const r of rows) {
    console.log(
      `${String(r.q).padEnd(2)} ${r.orbit.padEnd(15)} ${String(r.lambda).padEnd(4)} ${r.sig.padEnd(4)} ${String(r.j).padEnd(5)} ${String(r.cJ).padEnd(3)} ${String(r.cJ2).padEnd(4)} ${String(r.level ?? "-").padEnd(6)} ${r.epistemics ?? "-"}`,
    );
  }
  console.log("\nRule mining (train = EXACT + DEF-LOW):");
  for (const [name, m] of Object.entries(mining)) {
    console.log(
      `  ${name.padEnd(12)} consistent=${m.consistent}` +
        (m.consistent
          ? "  predictions: " + m.predictions.join("  ")
          : "  collisions: " + m.collisions.map((c) => c.join(" vs ")).join(" ; ")),
    );
  }
}

main();
