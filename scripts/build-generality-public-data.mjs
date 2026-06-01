import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const outputPath = "public/data/high-stakes-generality-gallery.json";

const statusPalette = {
  "review-gated": {
    label: "Review gated",
    tone: "amber",
    fill: "#FFF4D6",
    stroke: "#B8831E",
    text: "#684811",
  },
  "design-hold": {
    label: "Design hold",
    tone: "slate",
    fill: "#EDF2F7",
    stroke: "#64748B",
    text: "#37465A",
  },
  "draft-handoff": {
    label: "Draft handoff",
    tone: "green",
    fill: "#E7F4EC",
    stroke: "#2F7D4F",
    text: "#214B36",
  },
  "cost-hold": {
    label: "Cost hold",
    tone: "red",
    fill: "#F8E7E3",
    stroke: "#A85043",
    text: "#6B3029",
  },
  "execution-hold": {
    label: "Execution hold",
    tone: "slate",
    fill: "#EDF2F7",
    stroke: "#64748B",
    text: "#37465A",
  },
  running: {
    label: "Running",
    tone: "blue",
    fill: "#EAF1FA",
    stroke: "#1A3A52",
    text: "#1A3A52",
  },
};

const projects = [
  {
    id: "riemann",
    slug: "riemann",
    shortName: "Riemann",
    fullName: "Riemann bounded-null ledger",
    category: "math-ledger",
    status: "review-gated",
    transferQuestion:
      "Can Sundog-style projection and symmetry receipts find a useful structural edge in RH-adjacent zero data?",
    currentRead:
      "Three lanes returned bounded nulls by distinct causes; no structural-zero edge was found.",
    blocker:
      "External sanity review before any public-facing claim beyond bounded-null discipline.",
    nextAction:
      "Send the external-review email, record the response, and require a new preregistration for any reopened lane.",
    publicBoundary:
      "No RH proof, disproof, Hilbert-Polya operator, or new zero formula is claimed.",
    chartHeadline: "3 lanes, 3 bounded nulls",
    metrics: [
      { label: "lanes", value: 3, unit: "lanes" },
      { label: "bounded nulls", value: 3, unit: "receipts" },
      { label: "claimed RH progress", value: 0, unit: "claims" },
    ],
    sources: [
      "docs/SUNDOG_V_RIEMANN.md",
      "docs/riemann/RIEMANN_BOUNDED_NULL_SYNTHESIS.md",
      "docs/riemann/EXTERNAL_REVIEW_PACKET.md",
      "docs/riemann/EXTERNAL_REVIEW_EMAIL_DRAFT.md",
    ],
    svgFocus: ["status-card", "review-gate", "null-lane-count"],
  },
  {
    id: "navier_stokes_c1",
    slug: "navier-stokes-c1",
    shortName: "Navier-Stokes C1",
    fullName: "Kolmogorov-flow C1 regime-generality lane",
    category: "proof-track",
    status: "review-gated",
    transferQuestion:
      "Does a one-cell finite-Galerkin C1 witness survive a Grashof-axis regime change?",
    currentRead:
      "The complete Reading-2 witness now holds at G=200 and G=300 under a portable objective.",
    blocker:
      "External PDE review and proxy-faithfulness work remain open before promotion.",
    nextAction:
      "Surface reviewer candidates or firm up proxy faithfulness with a derived objective selector.",
    publicBoundary:
      "No Navier-Stokes solution or infinite-dimensional theorem is claimed; C1 remains finite-Galerkin, sampled-support, and unpromoted.",
    chartHeadline: "two-regime witness; review gated",
    metrics: [
      { label: "regimes", value: 2, unit: "G points" },
      { label: "G300 damp fraction", value: 0.2688, unit: "fraction" },
      { label: "G300 witness pairs", value: 942834, unit: "pairs" },
    ],
    sources: [
      "docs/SUNDOG_V_NAVIERSTOKES.md",
      "docs/proof/PDE_C1_KNN_CONVERGENCE_CHECK.md",
      "docs/proof/PDE_C1_TWIN_STATE_CERTIFICATE.md",
      "docs/proof/PDE_C1_REGIME_GENERALITY_v0.md",
      "docs/proof/PDE_C1_REGIME_GENERALITY_v1.md",
    ],
    svgFocus: ["status-card", "transfer-lane", "review-gate"],
  },
  {
    id: "pvnp",
    slug: "p-vs-np",
    shortName: "P-vs-NP",
    fullName: "Bounded alignment-verification verifier chain",
    category: "math-ledger",
    status: "draft-handoff",
    transferQuestion:
      "Can a compact signature behave like a cheaper certificate for a bounded alignment-verification problem?",
    currentRead:
      "The v0-v6 chain is safety-complete and v6 op-count positive; Phase 2 v1 earned a bounded-positive mesa bridge receipt; Phase 3 v0 falsified at small capacity.",
    blocker:
      "A source-bound small-tier spoof crossed the bridge accept rule; v1 repair is only a draft.",
    nextAction:
      "Review and freeze Phase 3 v1 block-stability repair slate.",
    publicBoundary:
      "No complexity-class result, polynomial certificate, or P-vs-NP progress is claimed.",
    chartHeadline: "op-count positive; wall-time diagnostic",
    metrics: [
      { label: "filed receipts", value: 10, unit: "incl p3v0" },
      { label: "v6 false accepts", value: 0, unit: "items" },
      { label: "p3 threshold", value: 1, unit: "<= small" },
    ],
    sources: [
      "docs/SUNDOG_V_P_V_NP.md",
      "docs/pvnp/README.md",
      "docs/pvnp/PHASE1_V6_SLATE.md",
      "docs/pvnp/PHASE2_MESA_BRIDGE_V0_SLATE.md",
      "docs/pvnp/PHASE2_MESA_BRIDGE_V1_SLATE.md",
      "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md",
      "docs/pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md",
      "docs/pvnp/receipts/2026-05-31_phase2_mesa_bridge_v0.md",
      "docs/pvnp/receipts/2026-05-31_phase2_mesa_bridge_v1.md",
      "docs/pvnp/receipts/2026-05-31_phase3_capacity_one_wayness_v0.md",
      "docs/pvnp/receipts/README.md",
    ],
    svgFocus: ["status-card", "receipt-chain", "op-count-positive"],
  },
  {
    id: "yang_mills",
    slug: "yang-mills",
    shortName: "Yang-Mills",
    fullName: "Finite-lattice gauge-invariant certificate lane",
    category: "math-ledger",
    status: "review-gated",
    transferQuestion:
      "Can a compact gauge-invariant signature preserve finite-lattice observable structure beyond controls?",
    currentRead:
      "Escalated from four uninformative Wilson-target nulls to a powered, disjoint finite-T Polyakov order-parameter test - the compact signature still showed no rank-locality. An informative bounded null.",
    blocker:
      "External lattice-gauge-theorist sanity check: is the powered finite-T test sound, and is the null expected on gauge-theory grounds?",
    nextAction:
      "Send the re-pointed external-review packet to a lattice gauge theorist; any reopened probe requires fresh external scientific motivation and a new dated preregistration.",
    publicBoundary:
      "No Yang-Mills existence, mass-gap, confinement, continuum-limit, or Clay-problem claim is licensed.",
    chartHeadline: "powered test, informative null",
    pageHeadline:
      "An audit-first escalation to a powered, disjoint finite-temperature Polyakov order-parameter test on SU(2) 2+1D - the small-loop signature still showed no rank-locality beyond controls.",
    metrics: [
      { label: "target power (ICC)", value: 0.965, unit: "fraction" },
      { label: "informative null", value: 1, unit: "synthesis" },
      { label: "claimed Clay progress", value: 0, unit: "claims" },
    ],
    sources: [
      "docs/SUNDOG_V_YANG_MILLS.md",
      "docs/YANG_MILLS_LITPASS_MEMO.md",
      "docs/prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md",
      "docs/prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md",
      "docs/yang-mills/receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md",
      "docs/yang-mills/receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md",
      "docs/yang-mills/EXTERNAL_REVIEW_PACKET.md",
      "docs/yang-mills/EXTERNAL_REVIEW_EMAIL_DRAFT.md",
    ],
    svgFocus: ["status-card", "review-gate", "powered-informative-null"],
  },
  {
    id: "arc_agi",
    slug: "arc-agi",
    shortName: "ARC-AGI",
    fullName: "ARC Phase 3E relative-locality hold",
    category: "benchmark-workbench",
    status: "execution-hold",
    transferQuestion:
      "After exact-grid decoder floors, does the signature still preserve rank-local program-sketch geometry?",
    currentRead:
      "Seven exact-grid floors stand; expanded absolute fibers stayed sparse and surfaced oracle leakage on the 108-task register.",
    blocker:
      "Relative-locality tooling must be admitted with wiring, result path, leak checks, smoke fingerprint, and freeze marker before receipt runs.",
    nextAction:
      "Admit the runner as tooling, run smoke only if under the ten-minute rule, and stage the full command if needed.",
    publicBoundary:
      "No ARC solver, benchmark-performance claim, or Blackwell-sufficiency proof is claimed.",
    chartHeadline: "absolute fibers sparse; relative test held",
    metrics: [
      { label: "exact-grid floors", value: 7, unit: "receipts" },
      { label: "expanded register", value: 108, unit: "tasks" },
      { label: "near pairs", value: 0, unit: "pairs" },
      { label: "lookup fraction", value: 0.351, unit: "fraction" },
    ],
    sources: [
      "docs/SUNDOG_V_ARC.md",
      "docs/prereg/arc/README.md",
      "docs/prereg/arc/PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md",
      "docs/prereg/arc/PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md",
    ],
    svgFocus: ["status-card", "benchmark-floor", "execution-hold"],
  },
  {
    id: "threebody_15c",
    slug: "three-body-15c",
    shortName: "Three-Body 15C",
    fullName: "Multi-step counterfactual horizon audit",
    category: "workbench-diagnostic",
    status: "running",
    transferQuestion:
      "Does the Phase 15B normalizer-collapse diagnosis change when counterfactuals run over longer horizons?",
    currentRead:
      "The audit is implemented and smoke-passed; 6 of 12 lock shards are logged, but no candidate-split branch read exists yet.",
    blocker:
      "All shards and the candidate-envelope horizon table must land before interpretation.",
    nextAction:
      "Finish the remaining shards, aggregate the candidate-split readback, and keep pooled observations diagnostic only.",
    publicBoundary:
      "No Phase 15 verdict revision, controller retune, or claim upgrade is licensed by interim shard logs.",
    chartHeadline: "6 / 12 shards logged",
    metrics: [
      { label: "shards logged", value: 6, unit: "of 12" },
      { label: "horizons", value: 4, unit: "N values" },
      { label: "branch reads", value: 0, unit: "reads" },
    ],
    sources: [
      "docs/threebody/PHASE15C_SPEC.md",
      "docs/threebody/PHASE15C_RESULTS.md",
      "docs/threebody/PHASE15B_RESULTS.md",
    ],
    svgFocus: ["status-card", "running-lock", "horizon-audit"],
  },
];

const data = {
  schemaVersion: 1,
  generatedBy: "scripts/build-generality-public-data.mjs",
  purpose:
    "Curated chart source for high-stakes generality SVGs. This is not a claim document; source ledgers remain authoritative.",
  pageCandidate: "generality.html",
  sourceOfTruth: [
    "docs/TODO.md",
    "docs/README.md",
    "docs/index.html",
    "project ledgers listed per project",
  ],
  brandFrame:
    "Sundog tests whether signatures that work in one domain survive substrate changes, and publishes the walls as receipts.",
  brandUse:
    "Use this data for SVG cards, a generality gallery, Ask Sundog claim-boundary answers, and future standalone project pages.",
  allowedLanguage: [
    "generality trial",
    "transfer test",
    "bounded null",
    "one-cell witness",
    "execution hold",
    "draft handoff",
    "finite-lattice certificate lane",
    "review gate",
    "cost-unadjudicated receipt",
    "rank-local sketch coherence",
  ],
  forbiddenLanguage: [
    "solves Riemann",
    "solves Navier-Stokes",
    "solves Yang-Mills",
    "proves Yang-Mills mass gap",
    "proves confinement",
    "solves P-vs-NP",
    "solves ARC",
    "proves general intelligence",
    "proves universal alignment",
    "turns nulls into wins without review",
  ],
  statusPalette,
  projects,
  charts: [
    {
      id: "generality-status-matrix",
      title: "High-stakes generality status matrix",
      output: "public/media/generality-status-matrix.svg",
      dataSource: "projects",
      intendedUse: "generality gallery overview",
    },
    {
      id: "generality-transfer-lanes",
      title: "Transfer lanes and current blockers",
      output: "public/media/generality-transfer-lanes.svg",
      dataSource: "projects",
      intendedUse: "page explainer or docs embed",
    },
    ...projects.map((project) => ({
      id: `generality-${project.slug}`,
      title: `${project.shortName} generality card`,
      output: `public/media/generality-${project.slug}.svg`,
      dataSource: `projects.${project.id}`,
      intendedUse: "individual project card",
    })),
  ],
};

const absoluteOutput = join(root, outputPath);
await mkdir(dirname(absoluteOutput), { recursive: true });
await writeFile(absoluteOutput, `${JSON.stringify(data, null, 2)}\n`, "utf8");
console.log(`generality public data built: ${outputPath}`);
