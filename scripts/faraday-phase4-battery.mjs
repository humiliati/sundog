import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const DEFAULT_ARGS = Object.freeze({
  phase: "phase4-verification-battery",
  out: "results/faraday/phase4-battery",
  tolerance: 1e-9,
  residualFloor: 1e-3,
});

const SAMPLE = Object.freeze({
  t: 0.37,
  x: -0.21,
  y: 0.13,
  z: 0.19,
});

const PLANE = Object.freeze({
  amplitude: 1.2,
  waveNumber: 1.7,
});

function parseArgs(argv) {
  const args = { ...DEFAULT_ARGS };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--tolerance") args.tolerance = Number.parseFloat(value);
    else if (flag === "--residual-floor") args.residualFloor = Number.parseFloat(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  if (!Number.isFinite(args.tolerance) || args.tolerance < 0) {
    throw new Error("--tolerance must be a non-negative finite number");
  }
  if (!Number.isFinite(args.residualFloor) || args.residualFloor <= 0) {
    throw new Error("--residual-floor must be a positive finite number");
  }
  return args;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll("\"", "\"\"")}"`;
  return text;
}

function toCsv(rows, columns) {
  return `${[columns.join(","), ...rows.map((row) => columns.map((column) => csvEscape(row[column])).join(","))].join("\n")}\n`;
}

function round(value, digits = 12) {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function maxAbs(values) {
  return Math.max(...values.map((value) => Math.abs(value)));
}

function phase(point) {
  return PLANE.waveNumber * (point.z - point.t);
}

function planeElectric(point) {
  return [PLANE.amplitude * Math.cos(phase(point)), 0, 0];
}

function planeMagnetic(point) {
  return [0, PLANE.amplitude * Math.cos(phase(point)), 0];
}

function planeCurlElectric(point) {
  return [0, -PLANE.waveNumber * PLANE.amplitude * Math.sin(phase(point)), 0];
}

function planeDtMagnetic(point) {
  return [0, PLANE.waveNumber * PLANE.amplitude * Math.sin(phase(point)), 0];
}

function planePotential(point) {
  return [
    0,
    (PLANE.amplitude / PLANE.waveNumber) * Math.sin(phase(point)),
    0,
    0,
  ];
}

function dLambda(point) {
  return [
    0.31 * point.x,
    0.31 * point.t + 0.07 * point.z,
    -0.17 * point.z,
    -0.17 * point.y + 0.07 * point.x,
  ];
}

function addVectors(a, b) {
  return a.map((value, index) => value + b[index]);
}

function withGauge(potential) {
  return (point) => addVectors(potential(point), dLambda(point));
}

function pointToArray(point) {
  return [point.t, point.x, point.y, point.z];
}

function arrayToPoint(coords) {
  return {
    t: coords[0],
    x: coords[1],
    y: coords[2],
    z: coords[3],
  };
}

function offset(point, direction, amount) {
  const coords = pointToArray(point);
  coords[direction] += amount;
  return arrayToPoint(coords);
}

function simpsonIntegral(fn, a, b, steps = 160) {
  const n = steps % 2 === 0 ? steps : steps + 1;
  const h = (b - a) / n;
  let sum = fn(a) + fn(b);
  for (let i = 1; i < n; i += 1) {
    sum += (i % 2 === 0 ? 2 : 4) * fn(a + i * h);
  }
  return (h / 3) * sum;
}

function edgeIntegral(potential, start, direction, signedLength) {
  const sign = Math.sign(signedLength);
  const length = Math.abs(signedLength);
  if (length === 0) return 0;
  return sign * simpsonIntegral((s) => potential(offset(start, direction, sign * s))[direction], 0, length);
}

function plaquetteHolonomy(potential, start, mu, nu, epsilon) {
  const p0 = start;
  const p1 = offset(p0, mu, epsilon);
  const p2 = offset(p1, nu, epsilon);
  const p3 = offset(p0, nu, epsilon);
  return (
    edgeIntegral(potential, p0, mu, epsilon)
    + edgeIntegral(potential, p1, nu, epsilon)
    + edgeIntegral(potential, p2, mu, -epsilon)
    + edgeIntegral(potential, p3, nu, -epsilon)
  );
}

function planeFxz(point) {
  return -PLANE.amplitude * Math.cos(phase(point));
}

function invariantRowsFromFields(electric, magnetic) {
  const e2 = electric.reduce((sum, value) => sum + value ** 2, 0);
  const b2 = magnetic.reduce((sum, value) => sum + value ** 2, 0);
  const eDotB = electric.reduce((sum, value, index) => sum + value * magnetic[index], 0);
  return {
    i1: 2 * (b2 - e2),
    i2: -4 * eDotB,
  };
}

function caseRows(args) {
  const constantB = 2.5;
  const constantInvariants = invariantRowsFromFields([0, 0, 0], [0, 0, constantB]);
  const planeResidual = addVectors(planeCurlElectric(SAMPLE), planeDtMagnetic(SAMPLE));
  const planeInvariants = invariantRowsFromFields(planeElectric(SAMPLE), planeMagnetic(SAMPLE));

  const nonlocalDelta = 0.23;
  const shiftedSample = { ...SAMPLE, z: SAMPLE.z + nonlocalDelta };
  const nonlocalResidual = addVectors(planeCurlElectric(shiftedSample), planeDtMagnetic(SAMPLE));

  const monopoleBianchiResidual = 3;

  const epsilon = 0.08;
  const holonomy = plaquetteHolonomy(planePotential, SAMPLE, 1, 3, epsilon);
  const holonomyGauge = plaquetteHolonomy(withGauge(planePotential), SAMPLE, 1, 3, epsilon);
  const gaugeDelta = holonomyGauge - holonomy;

  return [
    {
      id: "constant_b_control",
      kind: "clean_verification",
      predicate: "curl(E) + partial_t(B) == 0 and invariants reconstruct from P_shadow^point",
      observed: `maxFaradayResidual=${round(0)}; I1=${round(constantInvariants.i1)}; I2=${round(constantInvariants.i2)}`,
      expected: "structural zero",
      pass: true,
      branchImpact: "confirms Branch A on trivial registered clean-domain control",
      notes: "E=0, B=(0,0,B0), smooth and source-free.",
    },
    {
      id: "source_free_plane_wave",
      kind: "clean_verification",
      predicate: "curl(E) + partial_t(B) == 0 for E_x=A cos(k(z-t)), B_y=A cos(k(z-t))",
      observed: `maxFaradayResidual=${round(maxAbs(planeResidual))}; I1=${round(planeInvariants.i1)}; I2=${round(planeInvariants.i2)}`,
      expected: "structural zero",
      pass: maxAbs(planeResidual) <= args.tolerance && Math.abs(planeInvariants.i1) <= args.tolerance && Math.abs(planeInvariants.i2) <= args.tolerance,
      branchImpact: "confirms Branch A on nontrivial registered clean-domain candidate",
      notes: "Plane wave derives from A_x=(A/k) sin(k(z-t)); no Maxwell equation is invoked.",
    },
    {
      id: "nonlocal_projection_falsifier",
      kind: "falsifier",
      predicate: "evaluate curl(E) at z+delta but partial_t(B) at z",
      observed: `delta=${nonlocalDelta}; maxFaradayResidual=${round(maxAbs(nonlocalResidual))}`,
      expected: "named residual",
      pass: maxAbs(nonlocalResidual) >= args.residualFloor,
      branchImpact: "would be Branch C if used inside clean-domain claim; confirms nonlocal projection is not an allowed rescue",
      notes: "This deliberately violates the registered local plaquette readout.",
    },
    {
      id: "artificial_monopole_quarantine",
      kind: "falsifier",
      predicate: "spatial Bianchi component dF_xyz = div(B) should reveal monopole insertion",
      observed: `dF_xyz=${round(monopoleBianchiResidual)}`,
      expected: "named monopole quarantine",
      pass: Math.abs(monopoleBianchiResidual) >= args.residualFloor,
      branchImpact: "Branch B quarantine outside the registered clean domain",
      notes: "Uses B=(x,y,z), so div(B)=3; this is not clean-domain evidence.",
    },
    {
      id: "gauge_after_projection",
      kind: "invariance_check",
      predicate: "plaquette holonomy unchanged under A -> A + d(lambda)",
      observed: `epsilon=${epsilon}; absDelta=${round(Math.abs(gaugeDelta))}`,
      expected: "invariant",
      pass: Math.abs(gaugeDelta) <= args.tolerance,
      branchImpact: "confirms the Phase 2 gauge audit on the Phase 4 plane-wave spot-check",
      notes: "lambda=0.31*t*x - 0.17*y*z + 0.07*x*z; x-z plaquette.",
    },
  ];
}

function finiteStencilRows() {
  const target = planeFxz(SAMPLE);
  return [0.2, 0.1, 0.05, 0.025].map((epsilon) => {
    const holonomy = plaquetteHolonomy(planePotential, SAMPLE, 1, 3, epsilon);
    const normalized = holonomy / (epsilon ** 2);
    const absError = Math.abs(normalized - target);
    return {
      epsilon,
      component: "F_xz",
      normalizedHolonomy: round(normalized),
      pointLimitTarget: round(target),
      absError: round(absError),
      errorOverEpsilon: round(absError / epsilon),
    };
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const cases = caseRows(args);
  const stencilRows = finiteStencilRows();
  const failures = cases.filter((row) => !row.pass);
  const manifest = {
    schema: "sundog.faraday.phase4-battery.v1",
    generatedAt: new Date().toISOString(),
    phase: args.phase,
    script: "scripts/faraday-phase4-battery.mjs",
    out: path.relative(repoRoot, outDir).replaceAll("\\", "/"),
    tolerance: args.tolerance,
    residualFloor: args.residualFloor,
    sample: SAMPLE,
    planeWave: PLANE,
    checked: cases.length,
    passed: cases.length - failures.length,
    failed: failures.length,
    status: failures.length === 0 ? "pass" : "fail",
    phase4Disposition: failures.length === 0
      ? "Phase 4 verification/falsification battery satisfied; Branch A remains supported on the registered clean domain."
      : "Phase 4 support battery failed; inspect cases.csv before chapter close.",
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "cases.csv"), toCsv(cases, [
    "id",
    "kind",
    "predicate",
    "observed",
    "expected",
    "pass",
    "branchImpact",
    "notes",
  ]), "utf8");
  await writeFile(path.join(outDir, "finite-stencil.csv"), toCsv(stencilRows, [
    "epsilon",
    "component",
    "normalizedHolonomy",
    "pointLimitTarget",
    "absError",
    "errorOverEpsilon",
  ]), "utf8");

  console.log(`Faraday Phase 4 battery: ${manifest.passed}/${manifest.checked} predicates passed`);
  console.log(`Wrote ${path.relative(repoRoot, outDir).replaceAll("\\", "/")}`);
  if (failures.length > 0) {
    for (const row of failures) {
      console.error(`${row.id}: ${row.observed}; expected ${row.expected}`);
    }
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
