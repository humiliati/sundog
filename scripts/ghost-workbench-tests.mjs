// Acceptance tests for the Ghost Phase 2 toy closure workbench.
// Implements docs/ghost/PHASE2_TOY_WORKBENCH_SPEC.md section 6.
// Run: `npm run ghost:test`.

import * as G from "../ghost/ghost-core.js";

let pass = 0;
const failures = [];

function check(name, cond) {
  if (cond) pass++;
  else failures.push(name);
}

function hasForbiddenKey(obj, forbidden) {
  if (!obj || typeof obj !== "object") return false;
  for (const key of Object.keys(obj)) {
    if (forbidden.some((f) => key.toLowerCase().includes(f))) return true;
    if (hasForbiddenKey(obj[key], forbidden)) return true;
  }
  return false;
}

const periodic = G.makePeriodicSystem();
const pWindow = G.windowBounds(periodic, 64, 12);
const pSymbols = periodic.symbols.slice(pWindow.start, pWindow.end);
const pCandidates = G.periodCandidates(pSymbols, 12);

check("T1 periodic system global period is 4", periodic.globalPeriod === 4);
check("T2 periodic large window admits period 4", pCandidates.includes(4));
check("T3 periodic large window has no smaller period", pCandidates.every((p) => p >= 4));
check("T4 periodic analysis closes repeat cell", G.analyzeWindow(periodic, 64, 12).periodicClosed === true);

const fib = G.makeFibonacciSystem(11);
const lenA = (level) => G.fibonacciSymbolLength("A", level);
const lenB = (level) => G.fibonacciSymbolLength("B", level);

let recurrenceOk = true;
for (let level = 1; level <= 11; level++) {
  if (lenA(level) !== lenA(level - 1) + lenB(level - 1)) recurrenceOk = false;
  if (lenB(level) !== lenA(level - 1)) recurrenceOk = false;
}
check("T5 Fibonacci length recurrence", recurrenceOk);

const fibCounts = G.symbolCounts(fib.symbols);
check("T6 Fibonacci generated word has both symbols", fibCounts.A > 0 && fibCounts.B > 0);

const fPrefix = fib.symbols.slice(0, 89);
check("T7 Fibonacci prefix has no period <= 16", G.periodCandidates(fPrefix, 16).length === 0);

let partitionsOk = true;
for (let level = 0; level <= fib.maxAncestryLevel; level++) {
  if (!G.intervalsPartition(fib, level)) partitionsOk = false;
}
check("T8 Fibonacci intervals partition every level", partitionsOk);

const fAnalysis = G.analyzeWindow(fib, 110, 14, 4);
check("T9 Fibonacci window reports ancestry blocks", fAnalysis.ancestry.intersectingBlocks > 0);
check("T10 Fibonacci ancestry partition flag is true", fAnalysis.ancestry.partitionOk === true);
check("T11 Fibonacci does not close periodic repeat cell", fAnalysis.periodicClosed === false);

const exported = G.exportReaderAnalysis(fib, 110, 14, 4);
const forbidden = ["theorem", "proof", "invariant", "conjecture", "claim"];
check("T12 export avoids theorem-shaped keys", !hasForbiddenKey(exported, forbidden));
check("T13 export keeps cliff inactive", exported.cliff.active === false);

console.log(`GHOST_WORKBENCH_TESTS pass=${pass} fail=${failures.length}`);
for (const failure of failures) console.log("  FAIL " + failure);
process.exit(failures.length === 0 ? 0 : 1);

