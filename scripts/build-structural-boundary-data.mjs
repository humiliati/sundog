import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const outputPath = "public/data/structural-failure-boundary-map.json";

const sourcePaths = {
  prereg: "docs/prereg/structural-failure-coincidence/README.md",
  boundaryMap: "docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md",
  wave42Disposition:
    "docs/prereg/structural-failure-coincidence/P2_CUT2_WAVE42_DISPOSITION.md",
  cut3RunSpec: "docs/prereg/structural-failure-coincidence/P2_CUT3_RUN_SPEC.md",
  cut3Admission: "docs/prereg/structural-failure-coincidence/P2_CUT3_ADMISSION.md",
  h0Calibration:
    "docs/prereg/structural-failure-coincidence/P2_CUT3_H0_CALIBRATION.md",
};

const evidenceColumns = [
  {
    id: "handle",
    label: "Handle",
    meaning: "The closed-form route or evidence rule under test.",
  },
  {
    id: "eligible",
    label: "Eligible window",
    meaning: "Where the inverse is allowed to carry evidence.",
  },
  {
    id: "mustBreak",
    label: "Must break",
    meaning: "Where a traceable system must fail, abstain, or switch handles.",
  },
  {
    id: "correlateTell",
    label: "Correlate tell",
    meaning: "What an opaque shortcut would keep doing past the boundary.",
  },
];

const loci = [
  {
    id: "L1",
    title: "Parhelion offset route",
    displayStatus: "eligible with abstain band",
    visualClass: "eligible-abstain",
    handle: "offset = R22 / cos(h)",
    eligible:
      "Strict eligible photo set p2, p7, p13; Cut-2 leverage fill uses sec(h)-1 >= 0.02, h about 11.37 deg, as the low-leverage transition.",
    mustBreak:
      "Low-h low leverage, parhelion-derived-R22 tautology, and p26 invalid rows must be reported as low leverage or ineligible, not independent inverse evidence.",
    correlateTell:
      "Smooth confident altitude estimates continue across low-leverage, tautological, or invalid rows.",
    traceablePrediction:
      "Succeeds on strict eligible cases and abstains or reports low leverage outside them.",
    sourceReceipts: [
      "docs/calibration/HALO_PHENOMENA_ACCOUNTING.md",
      "docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md",
      "public/js/parhelion-geometry.mjs phase3.daggerOffset",
      "docs/prereg/structural-failure-coincidence/P2_CUT2_C2A_NUMERIC_FREEZE.md",
    ],
  },
  {
    id: "L2",
    title: "CZA visibility cutoff",
    displayStatus: "cutoff",
    visualClass: "cutoff",
    handle: "circumzenithal-arc visibility route",
    eligible: "Sun altitude h <= 32 deg, the coded operative guard.",
    mustBreak:
      "Above 32 deg a CZA-dependent route must fail, abstain, or switch handles.",
    correlateTell:
      "Altitude estimates continue through the cutoff because other image features still correlate with h.",
    traceablePrediction:
      "Does not preserve a CZA-apex inverse past disappearance.",
    sourceReceipts: [
      "public/js/parhelion-geometry.mjs czaVisibleAtAltitude",
      "docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md L2",
    ],
  },
  {
    id: "L3",
    title: "Tangent arc merge",
    displayStatus: "merge",
    visualClass: "merge",
    handle: "upper-tangent locus route",
    eligible: "h < 29 deg, where the separate upper-tangent handle exists.",
    mustBreak:
      "At h >= 29 deg tangentArcLocus returns null because the tangent arcs have merged into the circumscribed-halo regime.",
    correlateTell:
      "A tangent-like estimate remains continuous through the singularity.",
    traceablePrediction:
      "Degrades, abstains, or switches at the merge.",
    sourceReceipts: [
      "public/js/parhelion-geometry.mjs TANGENT_ARC_CIRCUMSCRIBED_H",
      "docs/calibration/PASS_C7_OUTPUT.txt",
      "docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md L3",
    ],
  },
  {
    id: "L4",
    title: "Supralateral route",
    displayStatus: "permanent fail",
    visualClass: "permanent-fail",
    handle: "supralateral angular-position candidate",
    eligible: "None under the documented apparatus.",
    mustBreak:
      "All altitudes: the handle fails the structural-discrimination gate, with only about 0.5 deg spread over h = 0-22 deg.",
    correlateTell:
      "Brightness, crop position, or co-occurring arcs are treated as if they were a usable altitude channel.",
    traceablePrediction:
      "Refuses to promote supralateral position as inverse evidence.",
    sourceReceipts: [
      "docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md",
      "docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md L4",
    ],
  },
  {
    id: "L5",
    title: "Rendered is not anchored",
    displayStatus: "admissibility rule",
    visualClass: "admissibility",
    handle: "evidence-admissibility boundary",
    eligible:
      "Only anchored closed-form rows with project receipts may count as inverse evidence.",
    mustBreak:
      "Rendered-optional, named-only, not-modeled, and atlas-placeholder primitives cannot carry traceability evidence by themselves.",
    correlateTell:
      "The mere presence of a drawn or named primitive is treated as evidence that an inverse is available.",
    traceablePrediction:
      "Uses unanchored rendered primitives only as display, vocabulary, nuisance, or future-hypothesis material.",
    sourceReceipts: [
      "docs/calibration/HALO_PHENOMENA_ACCOUNTING.md",
      "docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md L5",
      "docs/prereg/structural-failure-coincidence/P1_ADMISSION.md",
    ],
  },
];

const statusLadder = [
  {
    id: "p0",
    label: "P0 boundary map",
    status: "pass",
    publicLabel: "Frozen falsifier written",
    source: sourcePaths.boundaryMap,
  },
  {
    id: "p1",
    label: "P1 admission",
    status: "pass",
    publicLabel: "Boundary map admitted as testable apparatus",
    source: "docs/prereg/structural-failure-coincidence/P1_ADMISSION.md",
  },
  {
    id: "p2-first-cut",
    label: "P2 first cut",
    status: "reclassified",
    publicLabel: "Machinery live; route test vacuous",
    source: "docs/prereg/structural-failure-coincidence/P2_RESULTS.md",
  },
  {
    id: "cut2",
    label: "Cut 2 closed-form line",
    status: "regime-separability",
    publicLabel: "Closed-form correlate cannot compete where the route is eligible",
    source: sourcePaths.wave42Disposition,
  },
  {
    id: "cut3",
    label: "Cut 3 rendered escalation",
    status: "hold",
    publicLabel: "Spec opened; execution held on H0/corpus/admission artifacts",
    source: sourcePaths.cut3Admission,
  },
  {
    id: "h0",
    label: "H0 angular calibration",
    status: "open",
    publicLabel: "Checker design exists; real-frame negative side remains open",
    source: sourcePaths.h0Calibration,
  },
];

const data = {
  schemaVersion: 1,
  purpose:
    "Public chart source for the Structural Failure Boundary Map. This file is generated from frozen preregistration claims and current append-only status; do not hand-edit.",
  sourcePaths,
  publicationTier: "pre_registered_falsifier",
  claimBoundary:
    "Publish as a falsifier and apparatus map: a traceable system should fail where the closed-form inverse loses identifiability. Do not present this as a universal theorem, agent traceability pass, or rendered-signal result.",
  currentPublicStatus:
    "P0/P1 passed; P2 first cut was reclassified as machinery-live route-test-vacuous; the closed-form Cut-2 line produced a regime-separability finding and escalated to Cut-3; Cut-3 execution is held on H0/corpus/admission artifacts.",
  copyBlocks: {
    eyebrow: "Pre-registered falsifier",
    headline: "The falsifier before the agent",
    dek:
      "A traceable system should fail where the closed-form inverse loses identifiability.",
    safeSummary:
      "Sundog froze five geometric loci before agent execution: where a route is eligible, where it must abstain or switch, and how a mere correlate would keep reporting confidence past the boundary.",
    statusLine:
      "Current status: frozen map and closed-form regime-separability finding; rendered Cut-3 remains held.",
  },
  allowedLanguage: [
    "traceability harness",
    "indirect-inference alignment benchmark",
    "frozen falsifier",
    "identifiability boundary",
    "failure-boundary coincidence",
    "closed-form regime-separability finding",
  ],
  forbiddenLanguage: [
    "theorem proved",
    "universal alignment proof",
    "agent traceability confirmed",
    "rendered Cut-3 passed",
    "probe decoded it, therefore the route was used",
  ],
  evidenceColumns,
  loci,
  statusLadder,
  chartQueue: [
    {
      id: "structural-boundary-five-locus-map",
      target: "post-rail Working Systems evidence panel",
      source: "loci",
      output: "public/media/structural-boundary-five-locus-map.svg",
      status: "ready-for-design",
    },
    {
      id: "structural-boundary-status-ladder",
      target: "structural-failure page or expanded evidence panel",
      source: "statusLadder",
      output: "public/media/structural-boundary-status-ladder.svg",
      status: "ready-for-design",
    },
  ],
};

const absoluteOutput = join(root, outputPath);
await mkdir(dirname(absoluteOutput), { recursive: true });
await writeFile(absoluteOutput, `${JSON.stringify(data, null, 2)}\n`, "utf8");
console.log(`structural boundary public data built: ${outputPath}`);
