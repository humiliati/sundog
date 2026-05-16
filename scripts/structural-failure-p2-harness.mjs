import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { phase3 } from "../public/js/parhelion-geometry.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const R22 = phase3.HALO_22_RADIUS;
const Q_MIN = 0;
const Q_MAX = 80;
const L1_LEVERAGE_MULTIPLIER = 1.02;
const L1_LEVERAGE_ALTITUDE_DEG = radToDeg(Math.acos(1 / L1_LEVERAGE_MULTIPLIER));

const THRESHOLDS = Object.freeze({
  convergenceDeg: 1.5,
  handleSteerDeg: 2.0,
  decoyInvariantDeg: 0.5,
  boundaryWindowDeg: 1.5,
  positiveControlMoveDeg: 2.0,
});

const GEOMETRY_BOUNDARIES = Object.freeze({
  czaCutoffDeg: 32,
  tangentMergeDeg: phase3.tangentArcCircumscribedAltitude,
  l1LeverageAltitudeDeg: L1_LEVERAGE_ALTITUDE_DEG,
});

function parseNumberList(value) {
  return String(value).split(",").map((item) => Number.parseFloat(item.trim())).filter(Number.isFinite);
}

function parseArgs(argv) {
  const args = {
    phase: "p2-execute-first-cut",
    out: "results/structural-failure/p2-execute-first-cut",
    hMin: 0,
    hMax: 70,
    hStep: 1,
    boundaryMin: 0,
    boundaryMax: 40,
    boundaryStep: 0.25,
    qStep: 0.05,
    counterfactualDelta: 20,
    decoyEditAltitudes: [0, 80],
  };

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (!flag.startsWith("--")) continue;
    i += 1;

    if (flag === "--phase") args.phase = value;
    else if (flag === "--out") args.out = value;
    else if (flag === "--h-min") args.hMin = Number.parseFloat(value);
    else if (flag === "--h-max") args.hMax = Number.parseFloat(value);
    else if (flag === "--h-step") args.hStep = Number.parseFloat(value);
    else if (flag === "--boundary-min") args.boundaryMin = Number.parseFloat(value);
    else if (flag === "--boundary-max") args.boundaryMax = Number.parseFloat(value);
    else if (flag === "--boundary-step") args.boundaryStep = Number.parseFloat(value);
    else if (flag === "--q-step") args.qStep = Number.parseFloat(value);
    else if (flag === "--counterfactual-delta") args.counterfactualDelta = Number.parseFloat(value);
    else if (flag === "--decoy-edit-altitudes") args.decoyEditAltitudes = parseNumberList(value);
    else throw new Error(`Unknown flag: ${flag}`);
  }

  for (const [name, value] of [
    ["--h-min", args.hMin],
    ["--h-max", args.hMax],
    ["--h-step", args.hStep],
    ["--boundary-min", args.boundaryMin],
    ["--boundary-max", args.boundaryMax],
    ["--boundary-step", args.boundaryStep],
    ["--q-step", args.qStep],
    ["--counterfactual-delta", args.counterfactualDelta],
  ]) {
    if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
  }
  if (args.hStep <= 0 || args.boundaryStep <= 0 || args.qStep <= 0) {
    throw new Error("--h-step, --boundary-step, and --q-step must be positive");
  }
  if (args.hMin < Q_MIN || args.hMax > Q_MAX || args.hMin >= args.hMax) {
    throw new Error(`--h-min/--h-max must satisfy ${Q_MIN} <= min < max <= ${Q_MAX}`);
  }
  if (args.boundaryMin < Q_MIN || args.boundaryMax > Q_MAX || args.boundaryMin >= args.boundaryMax) {
    throw new Error(`--boundary-min/--boundary-max must satisfy ${Q_MIN} <= min < max <= ${Q_MAX}`);
  }
  if (!Array.isArray(args.decoyEditAltitudes) || args.decoyEditAltitudes.length < 2) {
    throw new Error("--decoy-edit-altitudes must contain at least two altitudes");
  }
  return args;
}

function radToDeg(rad) {
  return (rad * 180) / Math.PI;
}

function degToRad(deg) {
  return (deg * Math.PI) / 180;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round(value, digits = 6) {
  if (!Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function altitudeRange(min, max, step) {
  const values = [];
  for (let h = min; h <= max + step * 0.5; h += step) {
    values.push(round(Math.min(h, max), 10));
  }
  return [...new Set(values)];
}

function isL1EligibleFPar(fPar) {
  return fPar >= L1_LEVERAGE_MULTIPLIER * R22;
}

function tangentFeature(altitudeDeg) {
  const locus = phase3.tangentArcLocus(altitudeDeg);
  if (locus === null) return null;
  return {
    openingCoefficient: phase3.tangentArcOpeningCoefficient(altitudeDeg),
    sampleCount: locus.length,
  };
}

function naturalDecoys(altitudeDeg) {
  const h = clamp(altitudeDeg, Q_MIN, Q_MAX);
  const dSupDeg = 46 + 0.5 * (Math.min(h, 22) / 22);
  const renderedOptionalFlag = h >= 12 && h <= 60 ? 1 : 0;
  const namedOnlyPrimitiveFlag = h >= 30 ? 1 : 0;
  return {
    d_sup_deg: dSupDeg,
    d_unanch_rendered_optional: renderedOptionalFlag,
    d_unanch_named_only: namedOnlyPrimitiveFlag,
    d_style: h / Q_MAX,
  };
}

function genuineHandlesFromAltitude(altitudeDeg) {
  return {
    f_par: phase3.daggerOffset(altitudeDeg),
    f_cza: phase3.czaVisible(altitudeDeg) ? 1 : 0,
    f_tan: tangentFeature(altitudeDeg),
  };
}

function makeBundle(altitudeDeg) {
  return {
    ...genuineHandlesFromAltitude(altitudeDeg),
    ...naturalDecoys(altitudeDeg),
  };
}

function editGenuineHandles(bundle, counterfactualAltitudeDeg) {
  return {
    ...bundle,
    ...genuineHandlesFromAltitude(counterfactualAltitudeDeg),
  };
}

function editDecoys(bundle, decoyAltitudeDeg) {
  return {
    ...bundle,
    ...naturalDecoys(decoyAltitudeDeg),
  };
}

function transparentAdapter(bundle, qDeg) {
  const eligible = isL1EligibleFPar(bundle.f_par);
  if (!eligible) {
    return {
      status: "abstain",
      objective: null,
      activeHandles: {
        parhelion: false,
        cza: bundle.f_cza === 1,
        tangent: bundle.f_tan !== null,
        supralateral: false,
      },
    };
  }

  const modelFPar = R22 / Math.cos(degToRad(qDeg));
  return {
    status: "ok",
    objective: -Math.abs(bundle.f_par - modelFPar),
    activeHandles: {
      parhelion: true,
      cza: bundle.f_cza === 1,
      tangent: bundle.f_tan !== null,
      supralateral: false,
    },
  };
}

function routeEstimate(bundle, args) {
  const first = transparentAdapter(bundle, Q_MIN);
  if (first.status === "abstain") {
    return {
      controller: "transparent_route_extremum_seek",
      status: "abstain",
      qHatDeg: null,
      samples: 0,
      activeHandles: first.activeHandles,
    };
  }

  let bestQ = Q_MIN;
  let bestObjective = -Infinity;
  let samples = 0;
  for (let q = Q_MIN; q <= Q_MAX + args.qStep * 0.5; q += args.qStep) {
    const candidateQ = Math.min(q, Q_MAX);
    const candidate = transparentAdapter(bundle, candidateQ);
    samples += 1;
    if (candidate.objective > bestObjective) {
      bestQ = candidateQ;
      bestObjective = candidate.objective;
    }
  }

  const final = transparentAdapter(bundle, bestQ);
  return {
    controller: "transparent_route_extremum_seek",
    status: "ok",
    qHatDeg: bestQ,
    samples,
    objective: bestObjective,
    activeHandles: final.activeHandles,
  };
}

function analyticInverseEstimate(bundle) {
  if (!isL1EligibleFPar(bundle.f_par)) {
    return { controller: "analytic_inverse", status: "abstain", qHatDeg: null, samples: 0 };
  }
  const ratio = clamp(R22 / bundle.f_par, -1, 1);
  return {
    controller: "analytic_inverse",
    status: "ok",
    qHatDeg: radToDeg(Math.acos(ratio)),
    samples: 1,
  };
}

const PAR_NORM_SPAN = phase3.daggerOffset(Q_MAX) / R22 - 1;
const TAN_NORM_SPAN = phase3.tangentArcOpeningCoefficient(GEOMETRY_BOUNDARIES.tangentMergeDeg - 0.5);

function positiveControlVector(bundle) {
  const fParNorm = ((bundle.f_par / R22) - 1) / PAR_NORM_SPAN;
  const tanPresent = bundle.f_tan === null ? 0 : 1;
  const tanNorm = bundle.f_tan === null
    ? 0
    : clamp(bundle.f_tan.openingCoefficient / TAN_NORM_SPAN, 0, 1);
  return [
    fParNorm,
    bundle.f_cza,
    tanPresent,
    tanNorm,
    (bundle.d_sup_deg - 46) / 0.5,
    bundle.d_unanch_rendered_optional,
    bundle.d_unanch_named_only,
    bundle.d_style,
  ];
}

function squaredDistance(left, right) {
  let total = 0;
  for (let i = 0; i < left.length; i += 1) {
    const diff = left[i] - right[i];
    total += diff * diff;
  }
  return total;
}

function positiveControlEstimate(bundle, args) {
  const target = positiveControlVector(bundle);
  let bestQ = Q_MIN;
  let bestLoss = Infinity;
  let samples = 0;
  for (let q = Q_MIN; q <= Q_MAX + args.qStep * 0.5; q += args.qStep) {
    const candidateQ = Math.min(q, Q_MAX);
    const model = positiveControlVector(makeBundle(candidateQ));
    const loss = squaredDistance(target, model);
    samples += 1;
    if (loss < bestLoss) {
      bestQ = candidateQ;
      bestLoss = loss;
    }
  }
  return {
    controller: "decoy_correlate_positive_control",
    status: "ok",
    qHatDeg: bestQ,
    samples,
    loss: bestLoss,
    activeHandles: {
      parhelion: true,
      cza: true,
      tangent: true,
      supralateral: true,
      unanchoredPrimitiveFlags: true,
      style: true,
    },
  };
}

function counterfactualAltitude(altitudeDeg, args) {
  const upward = altitudeDeg + args.counterfactualDelta;
  const downward = altitudeDeg - args.counterfactualDelta;
  let candidate = upward <= args.hMax ? upward : downward;
  if (candidate < L1_LEVERAGE_ALTITUDE_DEG + 0.5) candidate = Math.min(args.hMax, L1_LEVERAGE_ALTITUDE_DEG + args.counterfactualDelta);
  return clamp(candidate, args.hMin, args.hMax);
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(round(value, 9)) : "";
  if (typeof value === "boolean") return value ? "true" : "false";
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll("\"", "\"\"")}"` : text;
}

function rowsToCsv(rows, explicitColumns = null) {
  const columns = explicitColumns ?? [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const lines = [columns.join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  return `${lines.join("\n")}\n`;
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function max(values) {
  const finite = values.filter(Number.isFinite);
  return finite.length === 0 ? null : Math.max(...finite);
}

function min(values) {
  const finite = values.filter(Number.isFinite);
  return finite.length === 0 ? null : Math.min(...finite);
}

function rate(flags) {
  if (flags.length === 0) return null;
  return flags.filter(Boolean).length / flags.length;
}

function estimateTransition(rows, key) {
  for (let i = 1; i < rows.length; i += 1) {
    if (rows[i - 1][key] === true && rows[i][key] === false) {
      return (rows[i - 1].hDeg + rows[i].hDeg) / 2;
    }
  }
  return null;
}

function routeConstructionAudit() {
  return {
    routeTestVacuous: true,
    reason: "The bundle generator sets f_par = R22/cos(h), while the route objective maximizes -abs(f_par - R22/cos(q)); the matched analytic baseline is the same inverse arccos(R22/f_par). The route is g^-1(g(h)) by grid search, not an independent policy test.",
    routeAndAnalyticBaselineSameInverse: true,
    decoysReachableThroughRouteObjective: false,
    czaTangentAffectQEstimate: false,
    supralateralHardcodedNonHandle: true,
  };
}

function classifyRouteOutcome({ q1, q2, q3, audit, positiveControlMoved }) {
  if (audit.routeTestVacuous) {
    return {
      verdict: positiveControlMoved
        ? "MACHINERY_LIVE_ROUTE_TEST_VACUOUS"
        : "MACHINERY_INCOMPLETE_ROUTE_TEST_VACUOUS",
      preregOutcome: "instrument does not exercise discriminating route behavior",
      railVerdict: "STALLED / UNTESTED",
      traceabilityClaim: false,
    };
  }
  if (!q1.pass) {
    return {
      verdict: "CONVERGENCE_NULL",
      preregOutcome: "convergence null",
      railVerdict: "STALLED / BUSTED",
      traceabilityClaim: false,
    };
  }
  if (q2.inconclusive) {
    return {
      verdict: "INCONCLUSIVE_DECOY_BATTERY",
      preregOutcome: "decoy battery too weak",
      railVerdict: "STALLED",
      traceabilityClaim: false,
    };
  }
  if (!q2.pass) {
    return {
      verdict: "OPAQUE_CORRELATE",
      preregOutcome: "opaque correlate",
      railVerdict: "STALLED / BOUNDARY FOUND",
      traceabilityClaim: false,
    };
  }
  if (!q3.pass) {
    return {
      verdict: "ROUTE_NOT_THE_INVERSE",
      preregOutcome: "route is not the inverse",
      railVerdict: "BOUNDARY FOUND",
      traceabilityClaim: false,
    };
  }
  return {
    verdict: "TRACEABILITY_HARNESS_PASS",
    preregOutcome: "traceability harness passes on this domain",
    railVerdict: "OPERATING ENVELOPE / CONFIRMED",
    traceabilityClaim: true,
  };
}

function verdictMarkdown(verdict, args) {
  const q = verdict.quantities;
  const lines = [
    "# Structural Failure Coincidence P2 Execution Verdict",
    "",
    `Generated: ${verdict.generatedAt}`,
    `Phase: ${args.phase}`,
    `Verdict: ${verdict.routeOutcome.verdict}`,
    `Prereg outcome: ${verdict.routeOutcome.preregOutcome}`,
    `Rail vocabulary: ${verdict.routeOutcome.railVerdict}`,
    "",
    "## Quantities",
    "",
    `1. Convergence mechanical check: ${q.convergence.pass ? "PASS" : "FAIL"} (${q.convergence.eligiblePassCount}/${q.convergence.eligibleCount}, rate ${round(q.convergence.passRate, 4)})`,
    `2. Counterfactual mechanical check: ${q.steerability.pass ? "PASS" : q.steerability.inconclusive ? "INCONCLUSIVE" : "FAIL"}`,
    `3. Boundary-state mechanical check: ${q.boundaries.pass ? "PASS" : "FAIL"}`,
    `4. Matched-baseline efficiency: route/analytic sample ratio ${round(q.efficiency.routeToAnalyticSampleRatio, 4)}`,
    "",
    "## Route Construction Audit",
    "",
    `Route test vacuous: ${verdict.routeConstructionAudit.routeTestVacuous}`,
    `Route and analytic baseline are the same inverse: ${verdict.routeConstructionAudit.routeAndAnalyticBaselineSameInverse}`,
    `Decoys reachable through route objective: ${verdict.routeConstructionAudit.decoysReachableThroughRouteObjective}`,
    `CZA/tangent affect q estimate: ${verdict.routeConstructionAudit.czaTangentAffectQEstimate}`,
    `Reason: ${verdict.routeConstructionAudit.reason}`,
    "",
    "## Positive Control",
    "",
    `Decoy-correlate positive control verdict: ${verdict.positiveControl.outcome}`,
    `Max positive-control decoy movement: ${round(q.steerability.maxPositiveControlDecoyDeltaDeg, 4)} deg`,
    `Min per-sample positive-control battery movement: ${round(q.steerability.minPositiveControlBatteryDeltaDeg, 4)} deg`,
    `Max route decoy movement: ${round(q.steerability.maxRouteDecoyDeltaDeg, 4)} deg`,
    "",
    "## Boundary Events",
    "",
    `L1 low-leverage abstention: ${q.boundaries.l1.pass ? "PASS" : "FAIL"} (threshold h = ${round(GEOMETRY_BOUNDARIES.l1LeverageAltitudeDeg, 4)} deg)`,
    `L2 CZA drop: ${q.boundaries.l2.pass ? "PASS" : "FAIL"} (observed ${round(q.boundaries.l2.observedDeg, 4)} deg, expected ${GEOMETRY_BOUNDARIES.czaCutoffDeg} deg)`,
    `L3 tangent drop: ${q.boundaries.l3.pass ? "PASS" : "FAIL"} (observed ${round(q.boundaries.l3.observedDeg, 4)} deg, expected ${GEOMETRY_BOUNDARIES.tangentMergeDeg} deg)`,
    `L4 supralateral non-handle: ${q.boundaries.l4.pass ? "PASS" : "FAIL"}`,
    "",
    "## Public-Language Guard",
    "",
    verdict.routeOutcome.traceabilityClaim
      ? "This is an apparatus/benchmark result for the closed-form feature-bundle first cut. It is not a universal theorem proof."
      : "This is a machinery-live / route-test-vacuous result. It is not a traceability pass; do not use CONFIRMED or theorem language.",
    "",
  ];
  return `${lines.join("\n")}\n`;
}

function runHarness(args) {
  const sampleAltitudes = altitudeRange(args.hMin, args.hMax, args.hStep);
  const trialRows = [];
  const counterfactualRows = [];
  const decoyRows = [];
  const efficiencyRatios = [];

  for (const hDeg of sampleAltitudes) {
    const bundle = makeBundle(hDeg);
    const route = routeEstimate(bundle, args);
    const analytic = analyticInverseEstimate(bundle);
    const positiveControl = positiveControlEstimate(bundle, args);
    const eligible = isL1EligibleFPar(bundle.f_par);
    const routeAbsError = route.status === "ok" ? Math.abs(route.qHatDeg - hDeg) : null;
    const analyticAbsError = analytic.status === "ok" ? Math.abs(analytic.qHatDeg - hDeg) : null;
    if (route.status === "ok" && analytic.samples > 0) efficiencyRatios.push(route.samples / analytic.samples);

    trialRows.push({
      hDeg,
      eligible,
      routeStatus: route.status,
      routeQHatDeg: route.qHatDeg,
      routeAbsErrorDeg: routeAbsError,
      routeSamples: route.samples,
      analyticStatus: analytic.status,
      analyticQHatDeg: analytic.qHatDeg,
      analyticAbsErrorDeg: analyticAbsError,
      analyticSamples: analytic.samples,
      positiveControlQHatDeg: positiveControl.qHatDeg,
      fPar: bundle.f_par,
      fCza: bundle.f_cza,
      fTanPresent: bundle.f_tan !== null,
      dSupDeg: bundle.d_sup_deg,
      dUnanchRenderedOptional: bundle.d_unanch_rendered_optional,
      dUnanchNamedOnly: bundle.d_unanch_named_only,
      dStyle: bundle.d_style,
      routeActiveCza: route.activeHandles.cza,
      routeActiveTangent: route.activeHandles.tangent,
      routeActiveSupralateral: route.activeHandles.supralateral,
    });

    if (eligible) {
      const hPrime = counterfactualAltitude(hDeg, args);
      const handleEdited = editGenuineHandles(bundle, hPrime);
      const routeEdited = routeEstimate(handleEdited, args);
      counterfactualRows.push({
        hDeg,
        hPrimeDeg: hPrime,
        routeBaseQHatDeg: route.qHatDeg,
        routeHandleEditQHatDeg: routeEdited.qHatDeg,
        routeHandleEditAbsErrorToHPrimeDeg: routeEdited.status === "ok" ? Math.abs(routeEdited.qHatDeg - hPrime) : null,
        pass: routeEdited.status === "ok" && Math.abs(routeEdited.qHatDeg - hPrime) <= THRESHOLDS.handleSteerDeg,
      });

      const routeDeltas = [];
      const pcDeltas = [];
      for (const decoyAltitude of args.decoyEditAltitudes) {
        const decoyEdited = editDecoys(bundle, decoyAltitude);
        const routeDecoy = routeEstimate(decoyEdited, args);
        const pcDecoy = positiveControlEstimate(decoyEdited, args);
        const routeDelta = route.status === "ok" && routeDecoy.status === "ok"
          ? Math.abs(routeDecoy.qHatDeg - route.qHatDeg)
          : null;
        const pcDelta = Math.abs(pcDecoy.qHatDeg - positiveControl.qHatDeg);
        routeDeltas.push(routeDelta);
        pcDeltas.push(pcDelta);
        decoyRows.push({
          hDeg,
          decoyAltitudeDeg: decoyAltitude,
          routeBaseQHatDeg: route.qHatDeg,
          routeDecoyQHatDeg: routeDecoy.qHatDeg,
          routeDeltaDeg: routeDelta,
          positiveControlBaseQHatDeg: positiveControl.qHatDeg,
          positiveControlDecoyQHatDeg: pcDecoy.qHatDeg,
          positiveControlDeltaDeg: pcDelta,
        });
      }
    }
  }

  const boundaryRows = altitudeRange(args.boundaryMin, args.boundaryMax, args.boundaryStep).map((hDeg) => {
    const bundle = makeBundle(hDeg);
    const route = routeEstimate(bundle, args);
    return {
      hDeg,
      l1Eligible: isL1EligibleFPar(bundle.f_par),
      routeStatus: route.status,
      activeCza: route.activeHandles.cza,
      activeTangent: route.activeHandles.tangent,
      activeSupralateral: route.activeHandles.supralateral,
    };
  });

  const eligibleTrialRows = trialRows.filter((row) => row.eligible);
  const ineligibleTrialRows = trialRows.filter((row) => !row.eligible);
  const convergenceFlags = eligibleTrialRows.map((row) => (
    row.routeStatus === "ok" && row.routeAbsErrorDeg <= THRESHOLDS.convergenceDeg
  ));
  const handleFlags = counterfactualRows.map((row) => row.pass);
  const maxRouteDecoyDelta = max(decoyRows.map((row) => row.routeDeltaDeg));
  const maxPositiveControlDecoyDelta = max(decoyRows.map((row) => row.positiveControlDeltaDeg));
  const decoyRowsByAltitude = new Map();
  for (const row of decoyRows) {
    const rows = decoyRowsByAltitude.get(row.hDeg) ?? [];
    rows.push(row);
    decoyRowsByAltitude.set(row.hDeg, rows);
  }
  const positiveControlBatteryDeltas = Array.from(decoyRowsByAltitude.values())
    .map((rows) => max(rows.map((row) => row.positiveControlDeltaDeg)));
  const routeDecoyInvariant = decoyRows.every((row) => (
    Number.isFinite(row.routeDeltaDeg) && row.routeDeltaDeg <= THRESHOLDS.decoyInvariantDeg
  ));
  const positiveControlMoved = positiveControlBatteryDeltas.every((delta) => (
    Number.isFinite(delta) && delta >= THRESHOLDS.positiveControlMoveDeg
  ));

  const observedCzaDrop = estimateTransition(boundaryRows, "activeCza");
  const observedTangentDrop = estimateTransition(boundaryRows, "activeTangent");
  const l1Pass = ineligibleTrialRows.every((row) => row.routeStatus === "abstain")
    && eligibleTrialRows.every((row) => row.routeStatus === "ok");
  const l2Pass = Number.isFinite(observedCzaDrop)
    && Math.abs(observedCzaDrop - GEOMETRY_BOUNDARIES.czaCutoffDeg) <= THRESHOLDS.boundaryWindowDeg;
  const l3Pass = Number.isFinite(observedTangentDrop)
    && Math.abs(observedTangentDrop - GEOMETRY_BOUNDARIES.tangentMergeDeg) <= THRESHOLDS.boundaryWindowDeg;
  const l4Pass = boundaryRows.every((row) => row.activeSupralateral === false)
    && trialRows.every((row) => row.routeActiveSupralateral === false)
    && routeDecoyInvariant;

  const q1 = {
    pass: rate(convergenceFlags) >= 0.9,
    eligibleCount: eligibleTrialRows.length,
    eligiblePassCount: convergenceFlags.filter(Boolean).length,
    passRate: rate(convergenceFlags),
    maxEligibleAbsErrorDeg: max(eligibleTrialRows.map((row) => row.routeAbsErrorDeg)),
  };
  const q2 = {
    pass: handleFlags.every(Boolean) && routeDecoyInvariant && positiveControlMoved,
    inconclusive: handleFlags.every(Boolean) && routeDecoyInvariant && !positiveControlMoved,
    handleEditPassCount: handleFlags.filter(Boolean).length,
    handleEditCount: handleFlags.length,
    maxHandleEditAbsErrorDeg: max(counterfactualRows.map((row) => row.routeHandleEditAbsErrorToHPrimeDeg)),
    routeDecoyInvariant,
    positiveControlMoved,
    maxRouteDecoyDeltaDeg: maxRouteDecoyDelta,
    maxPositiveControlDecoyDeltaDeg: maxPositiveControlDecoyDelta,
    minPositiveControlBatteryDeltaDeg: min(positiveControlBatteryDeltas),
  };
  const q3 = {
    pass: l1Pass && l2Pass && l3Pass && l4Pass,
    l1: {
      pass: l1Pass,
      lowLeverageAbstainCount: ineligibleTrialRows.filter((row) => row.routeStatus === "abstain").length,
      lowLeverageCount: ineligibleTrialRows.length,
      thresholdAltitudeDeg: GEOMETRY_BOUNDARIES.l1LeverageAltitudeDeg,
    },
    l2: {
      pass: l2Pass,
      observedDeg: observedCzaDrop,
      expectedDeg: GEOMETRY_BOUNDARIES.czaCutoffDeg,
      toleranceDeg: THRESHOLDS.boundaryWindowDeg,
    },
    l3: {
      pass: l3Pass,
      observedDeg: observedTangentDrop,
      expectedDeg: GEOMETRY_BOUNDARIES.tangentMergeDeg,
      toleranceDeg: THRESHOLDS.boundaryWindowDeg,
    },
    l4: {
      pass: l4Pass,
      promotedAsHandle: false,
    },
  };
  const q4 = {
    routeToAnalyticSampleRatio: mean(efficiencyRatios),
    meanRouteSamples: mean(eligibleTrialRows.map((row) => row.routeSamples)),
    analyticSamples: 1,
  };

  const positiveControlOutcome = positiveControlMoved
    ? "OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED"
    : "POSITIVE_CONTROL_TOO_WEAK_INCONCLUSIVE";

  const quantities = {
    convergence: q1,
    steerability: q2,
    boundaries: q3,
    efficiency: q4,
  };
  const constructionAudit = routeConstructionAudit();
  const routeOutcome = classifyRouteOutcome({
    q1,
    q2,
    q3,
    audit: constructionAudit,
    positiveControlMoved,
  });

  return {
    generatedAt: new Date().toISOString(),
    schema: "sundog.structural-failure.p2-execute.v1",
    args,
    thresholds: THRESHOLDS,
    geometryBoundaries: GEOMETRY_BOUNDARIES,
    adapterInvariant: {
      transparentAdapterInputs: ["f_par", "f_cza", "f_tan", "R22", "q"],
      hiddenAltitudeReadInsideAdapter: false,
      decoysReadInsideAdapter: false,
    },
    routeConstructionAudit: constructionAudit,
    quantities,
    routeOutcome,
    positiveControl: {
      outcome: positiveControlOutcome,
      readsDecoys: true,
      policy: "generic least-squares fit over normalized full bundle, including d_sup, d_unanch, and d_style",
    },
    rows: {
      trials: trialRows,
      counterfactuals: counterfactualRows,
      decoys: decoyRows,
      boundaries: boundaryRows,
    },
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(repoRoot, args.out);
  const verdict = runHarness(args);

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify({
    schema: verdict.schema,
    generatedAt: verdict.generatedAt,
    args,
    thresholds: verdict.thresholds,
    geometryBoundaries: verdict.geometryBoundaries,
    adapterInvariant: verdict.adapterInvariant,
    routeConstructionAudit: verdict.routeConstructionAudit,
    verdict: verdict.routeOutcome.verdict,
    positiveControlOutcome: verdict.positiveControl.outcome,
  }, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "trial-outcomes.csv"), rowsToCsv(verdict.rows.trials), "utf8");
  await writeFile(path.join(outDir, "counterfactual-edits.csv"), rowsToCsv(verdict.rows.counterfactuals), "utf8");
  await writeFile(path.join(outDir, "decoy-edits.csv"), rowsToCsv(verdict.rows.decoys), "utf8");
  await writeFile(path.join(outDir, "boundary-events.csv"), rowsToCsv(verdict.rows.boundaries), "utf8");
  await writeFile(path.join(outDir, "verdict.json"), `${JSON.stringify({
    generatedAt: verdict.generatedAt,
    schema: verdict.schema,
    thresholds: verdict.thresholds,
    geometryBoundaries: verdict.geometryBoundaries,
    adapterInvariant: verdict.adapterInvariant,
    routeConstructionAudit: verdict.routeConstructionAudit,
    quantities: verdict.quantities,
    routeOutcome: verdict.routeOutcome,
    positiveControl: verdict.positiveControl,
  }, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "verdict.md"), verdictMarkdown(verdict, args), "utf8");

  console.log(`[p2] wrote ${path.relative(repoRoot, outDir)}`);
  console.log(`[p2] verdict ${verdict.routeOutcome.verdict}`);
  console.log(`[p2] q1 convergence ${verdict.quantities.convergence.eligiblePassCount}/${verdict.quantities.convergence.eligibleCount}`);
  console.log(`[p2] q2 steerability ${verdict.quantities.steerability.pass ? "PASS" : verdict.quantities.steerability.inconclusive ? "INCONCLUSIVE" : "FAIL"}`);
  console.log(`[p2] q3 boundaries ${verdict.quantities.boundaries.pass ? "PASS" : "FAIL"}`);
  console.log(`[p2] positive control ${verdict.positiveControl.outcome}`);
}

main().catch((error) => {
  console.error(`[p2] ${error.stack || error.message}`);
  process.exitCode = 1;
});
