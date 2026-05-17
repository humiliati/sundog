import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const outputPath = "public/media/coarse-graining-proof-ladder.svg";

const colors = {
  ink: "#173348",
  body: "#40505C",
  muted: "#6B7B88",
  line: "#DCE6EC",
  paper: "#F8FBFC",
  panel: "#FFFFFF",
  navy: "#1A3A52",
  gold: "#C9972F",
  green: "#3D7A5A",
  greenSoft: "#E8F2ED",
  amber: "#B8831E",
  amberSoft: "#FFF4D6",
  red: "#A85043",
  redSoft: "#F8E7E3",
  slate: "#64748B",
  slateSoft: "#EDF2F7",
  blue: "#2C5F7F",
  blueSoft: "#EAF1FA",
};

const statusPalette = {
  positive: { fill: colors.greenSoft, stroke: colors.green, text: "#214B36" },
  open: { fill: colors.amberSoft, stroke: colors.amber, text: "#684811" },
  locked: { fill: colors.slateSoft, stroke: colors.slate, text: "#37465A" },
  caution: { fill: colors.redSoft, stroke: colors.red, text: "#6B3029" },
};

const phases = [
  {
    id: "0",
    label: "Definitions",
    status: "positive",
    badge: "closed",
    source: "docs/proof/POSTULATE1_DEFINITIONS.md",
    check: /Status|definitions lock|Sundog-solvable/i,
    note: "Predicate and symbols pinned once.",
  },
  {
    id: "1",
    label: "LQG",
    status: "positive",
    badge: "positive",
    source: "docs/proof/PHASE1_LQG.md",
    check: /closed positive|Bayes-optimal|LQG/i,
    note: "Computable existence case closed.",
  },
  {
    id: "2",
    label: "Finite MDP",
    status: "positive",
    badge: "positive",
    source: "docs/proof/PHASE2_MDP.md",
    check: /closed positive|finite-MDP|separability/i,
    note: "Sufficiency and corollary closed.",
  },
  {
    id: "3",
    label: "Boundary",
    status: "positive",
    badge: "positive",
    source: "docs/proof/PHASE3_BOUNDARY.md",
    check: /closed positive|boundary theorem|pushable/i,
    note: "Failure direction named.",
  },
  {
    id: "4",
    label: "Three-Body",
    status: "open",
    badge: "blocked/open",
    source: "docs/proof/PHASE4_THREEBODY.md",
    check: /blocked \/ open|Bayesian-floor|floor repair/i,
    note: "Bayesian-floor gate not yet closed.",
  },
  {
    id: "5",
    label: "Cross-Substrate",
    status: "locked",
    badge: "locked",
    source: "docs/COARSE_GRAINING_PROOF_ROADMAP.md",
    check: /Phase 5|Cross-substrate|operator-identity/i,
    note: "Waits for Phase 4 positive exit.",
  },
  {
    id: "6",
    label: "Lambda Control",
    status: "open",
    badge: "staged/open",
    source: "docs/proof/PHASE6_LAMBDA_CONTROL.md",
    check: /staged|empirical result open|lambda-confound/i,
    note: "Public-cite gate still open.",
  },
];

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function wrapText(text, maxChars) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";

  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (next.length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }

  if (line) lines.push(line);
  return lines;
}

function textBlock({ text, x, y, maxChars, size = 15, lineHeight = 20, fill = colors.body, weight = 500 }) {
  const tspans = wrapText(text, maxChars)
    .map((line, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${escapeXml(line)}</tspan>`)
    .join("");
  return `<text x="${x}" y="${y}" fill="${fill}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="${size}" font-weight="${weight}" letter-spacing="0">${tspans}</text>`;
}

function pill({ x, y, label, palette, width = null }) {
  const pillWidth = width ?? Math.max(78, label.length * 7 + 28);
  return `<g>
    <rect x="${x}" y="${y}" width="${pillWidth}" height="28" rx="8" fill="${palette.fill}" stroke="${palette.stroke}" stroke-width="1"/>
    <text x="${x + 14}" y="${y + 18}" fill="${palette.text}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="800" letter-spacing="0">${escapeXml(label.toUpperCase())}</text>
  </g>`;
}

async function exists(relativePath) {
  try {
    await access(join(root, relativePath));
    return true;
  } catch {
    return false;
  }
}

async function readMaybe(relativePath) {
  try {
    return await readFile(join(root, relativePath), "utf8");
  } catch {
    return "";
  }
}

async function collectPhaseStatuses() {
  const measured = [];
  const regretManifestPath =
    "results/proof/phase4/bf4-probe-20260516-173223/regret/phase4-regret-manifest.json";
  const regretManifestText = await readMaybe(regretManifestPath);
  let floorStatus = null;

  if (regretManifestText) {
    const manifest = JSON.parse(regretManifestText);
    floorStatus = manifest.floorSanity?.status ?? null;
  }

  for (const phase of phases) {
    const fileExists = await exists(phase.source);
    const text = await readMaybe(phase.source);
    const sourceCheck = fileExists && phase.check.test(text);
    const note =
      phase.id === "4" && floorStatus === "non_decisive_floor_repair_required"
        ? "BF-4 probe is non-decisive; floor repair required."
        : phase.note;

    measured.push({
      ...phase,
      sourceCheck,
      note,
    });
  }

  return { phases: measured, floorStatus };
}

function phaseCard(phase, index) {
  const x = 76 + (index % 4) * 278;
  const y = 254 + Math.floor(index / 4) * 170;
  const palette = phase.sourceCheck ? statusPalette[phase.status] : statusPalette.caution;
  const statusLabel = phase.sourceCheck ? phase.badge : "check source";

  return `<g>
    <rect x="${x}" y="${y}" width="248" height="136" rx="10" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1.2"/>
    <rect x="${x}" y="${y}" width="8" height="136" rx="4" fill="${palette.stroke}"/>
    <circle cx="${x + 36}" cy="${y + 36}" r="19" fill="${palette.fill}" stroke="${palette.stroke}" stroke-width="1.4"/>
    <text x="${x + 30}" y="${y + 42}" fill="${palette.text}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="17" font-weight="850" letter-spacing="0">${escapeXml(phase.id)}</text>
    <text x="${x + 66}" y="${y + 34}" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="18" font-weight="850" letter-spacing="0">${escapeXml(phase.label)}</text>
    ${pill({ x: x + 66, y: y + 49, label: statusLabel, palette, width: Math.min(156, Math.max(98, statusLabel.length * 7 + 26)) })}
    ${textBlock({ text: phase.note, x: x + 28, y: y + 102, maxChars: 29, size: 13, lineHeight: 17, fill: colors.body, weight: 600 })}
  </g>`;
}

function buildSvg(data) {
  const width = 1280;
  const height = 760;
  const closedCount = data.phases.filter((phase) => phase.status === "positive" && phase.sourceCheck).length;
  const openCount = data.phases.filter((phase) => phase.status === "open" && phase.sourceCheck).length;
  const lockedCount = data.phases.filter((phase) => phase.status === "locked" && phase.sourceCheck).length;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">Coarse-Graining Proof Trunk Status Ladder</title>
  <desc id="desc">Generated status ladder for the Sundog coarse-graining proof track.</desc>
  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M32 0H0V32" fill="none" stroke="${colors.line}" stroke-width="1" opacity="0.38"/>
    </pattern>
  </defs>
  <rect width="${width}" height="${height}" fill="${colors.paper}"/>
  <rect width="${width}" height="${height}" fill="url(#grid)" opacity="0.72"/>
  <rect x="48" y="40" width="1184" height="680" rx="14" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1.5"/>
  <text x="78" y="88" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="38" font-weight="850" letter-spacing="0">Coarse-Graining Proof Trunk</text>
  ${pill({ x: 78, y: 112, label: "status ladder", palette: statusPalette.locked, width: 142 })}
  ${pill({ x: 232, y: 112, label: `${closedCount} closed positive`, palette: statusPalette.positive, width: 162 })}
  ${pill({ x: 406, y: 112, label: `${openCount} open gates`, palette: statusPalette.open, width: 128 })}
  ${pill({ x: 546, y: 112, label: `${lockedCount} locked`, palette: statusPalette.locked, width: 108 })}
  ${textBlock({
    text: "This is a proof-track posture panel, not public theorem language. The public rule remains: no claim upgrade until the cross-substrate leg lands and the lambda-confound control clears.",
    x: 78,
    y: 174,
    maxChars: 108,
    size: 17,
    lineHeight: 23,
    fill: colors.body,
    weight: 500,
  })}
  ${data.phases.map((phase, index) => phaseCard(phase, index)).join("\n")}
  <rect x="910" y="424" width="304" height="136" rx="10" fill="${colors.paper}" stroke="${colors.line}" stroke-width="1"/>
  <text x="938" y="462" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="20" font-weight="850" letter-spacing="0">Public boundary</text>
  ${textBlock({
    text: "The trunk studies when a signature is sufficient for control. It does not claim the nonlinear learned general case, and it does not rescue Phase 4 if the Bayes floor fails.",
    x: 938,
    y: 497,
    maxChars: 37,
    size: 14,
    lineHeight: 18,
    fill: colors.body,
    weight: 600,
  })}
  <line x1="78" y1="626" x2="1176" y2="626" stroke="${colors.line}" stroke-width="1"/>
  <text x="78" y="666" fill="${colors.muted}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="14" font-weight="700" letter-spacing="0">Generated from docs/COARSE_GRAINING_PROOF_ROADMAP.md and docs/proof/*. Phase 4 probe readback: ${escapeXml(data.floorStatus ?? "no floor readback")}.</text>
</svg>
`;
}

async function main() {
  const data = await collectPhaseStatuses();
  const absoluteOutput = join(root, outputPath);
  await mkdir(dirname(absoluteOutput), { recursive: true });
  await writeFile(absoluteOutput, buildSvg(data), "utf8");
  console.log(`coarse-graining public media built: ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
