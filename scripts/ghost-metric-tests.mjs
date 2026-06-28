// Acceptance tests for the Ghost Phase 4 metric probe (rigorous falsification
// battery). Implements docs/ghost/PHASE4_METRIC_PROBE_SPEC.md section 7.
// Stage S1: 1D primitive substitutions + periodic control.
// Run: `npm run ghost:metric:test`.

import * as M from "../ghost/metric-probe-core.js";

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

// T1 length recurrences
const fibLen = (d) => M.generateWord("fibonacci", d).length;
const fibOk = [2, 3, 4, 5, 6, 7].every((d) => fibLen(d) === fibLen(d - 1) + fibLen(d - 2));
check("T1 Fibonacci length recurrence", fibOk);
check("T1 period-doubling length 2^d", [1, 2, 3, 4, 5, 6].every((d) => M.generateWord("period-doubling", d).length === 2 ** d));
check("T1 Thue-Morse length 2^d", [1, 2, 3, 4, 5, 6].every((d) => M.generateWord("thue-morse", d).length === 2 ** d));

// T2 cut points partition the word, roles defined
for (const name of ["fibonacci", "period-doubling", "thue-morse"]) {
  const part = M.cutPointsPartition(name, 6);
  const anc = M.generateWithAncestry(name, 6);
  check(`T2 ${name} cut points partition`, part.ok && anc.role.length === anc.letters.length);
}

// T3 recognizability radius finite
const depths = { fibonacci: [12, 13], "period-doubling": [8, 9], "thue-morse": [8, 9] };
const radii = {};
for (const name of Object.keys(depths)) {
  const [da] = depths[name];
  const r = M.recognizabilityRadius1D(name, da);
  radii[name] = r;
  check(`T3 ${name} recognizability radius finite (= ${r})`, Number.isFinite(r));
}

// T4 depth stability (privileged-truth check: Mosse fixed constant)
for (const name of Object.keys(depths)) {
  const [da, db] = depths[name];
  const ra = M.recognizabilityRadius1D(name, da);
  const rb = M.recognizabilityRadius1D(name, db);
  check(`T4 ${name} recognizability radius depth-stable (${ra} == ${rb})`, Number.isFinite(ra) && ra === rb);
}

// T5 periodic repeat-cell capture radius
const cap = M.repeatCellCaptureRadius(["A", "B", "C", "D"]);
check(`T5 periodic ABCD capture radius == 5 (= ${cap})`, cap === 5);

// T7 falsification harness (1D portion)
const report = M.falsificationReport1D(12, 13);
check("T7 unbounded Ghost Boundary Heuristic falsified (1D)", report.unboundedHeuristicFalsified === true);
check(
  "T7 every aperiodic substrate reports finite + depth-stable radius",
  Object.values(report.aperiodic).every((s) => s.finite && s.depthStable),
);
check("T7 periodic control has a repeat cell and finite capture radius", report.periodicControl.hasRepeatCell && Number.isFinite(report.periodicControl.captureRadius));

// T8 export discipline (no NEW-invariant keys)
check("T8 report avoids theorem-shaped keys", !hasForbiddenKey(report, ["theorem", "proof", "invariant", "conjecture", "claim"]));

// --- Stage S2: 2D Penrose recognizability ---
// T6a role derivation is correct: own color recomputed from path matches color
check("T6a Penrose role/color derivation matches generation", M.penroseColorConsistency(4) === true);

// Convergence across depths 5,6,7 (edge-normalized). The radius converges from
// below as the interior core grows; it is depth-stable once finite-size effects
// die out (see spec 4.4). Measure stability at d6 vs d7.
const conv = M.penroseConvergence([5, 6, 7]);
const [, e6, e7] = conv.radiiEdges;

// T6b recognizability radius finite on the interior core
check(`T6b Penrose recognizability radius finite (~${e7 == null ? "n/a" : e7.toFixed(3)} edges)`, conv.finite === true);

// T6c depth-stable at sufficient depth (d6 vs d7 within tolerance)
check(
  `T6c Penrose recognizability depth-stable at d6 vs d7 (${e6 == null ? "n/a" : e6.toFixed(4)} vs ${e7 == null ? "n/a" : e7.toFixed(4)})`,
  conv.finite && Math.abs(e7 - e6) <= 1e-2,
);

// T6d converges from below + bounded
check("T6d Penrose recognizability bounded and converging", conv.bounded === true && conv.converging === true);

// T7 (2D) falsification + export discipline
check("T7 unbounded Ghost Boundary Heuristic falsified (2D Penrose)", conv.unboundedHeuristicFalsified === true);
check("T8 Penrose report avoids theorem-shaped keys", !hasForbiddenKey(conv, ["theorem", "proof", "invariant", "conjecture", "claim"]));

console.log(`GHOST_METRIC_TESTS pass=${pass} fail=${failures.length}`);
console.log("  measured recognizability radii (1D, depth A):", JSON.stringify(radii));
console.log("  Penrose recognizability radius (edges) by depth [5,6,7]:", conv.radiiEdges.map((x) => (x == null ? "n/a" : x.toFixed(4))).join(", "));
for (const failure of failures) console.log("  FAIL " + failure);
process.exit(failures.length === 0 ? 0 : 1);
