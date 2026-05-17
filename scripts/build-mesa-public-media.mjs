import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const inputPath = "public/data/mesa-public-charts.json";

const outputs = {
  evidencePanel: "public/media/mesa-evidence-panel.svg",
  cliffMini: "public/media/mesa-cliff-mini.svg",
  classBalanceStrip: "public/media/mesa-class-balance-strip.svg",
  kSweepFingerprint: "public/media/mesa-ksweep-fingerprint.svg",
};

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

const classPalette = {
  hold: { fill: colors.green, soft: colors.greenSoft, text: "#214B36", label: "Hold" },
  collapse: { fill: colors.red, soft: colors.redSoft, text: "#6B3029", label: "Collapse" },
  fragile: { fill: colors.amber, soft: colors.amberSoft, text: "#684811", label: "Fragile" },
  incompetent: { fill: colors.slate, soft: colors.slateSoft, text: "#37465A", label: "Incompetent" },
  ambiguous: { fill: colors.blue, soft: colors.blueSoft, text: colors.navy, label: "Ambiguous" },
};

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fixed(value, digits = 2) {
  if (!Number.isFinite(value)) return "n/a";
  return Number(value).toFixed(digits).replace(/\.?0+$/, "");
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
    } else if (word.length > maxChars) {
      if (line) {
        lines.push(line);
        line = "";
      }
      for (let index = 0; index < word.length; index += maxChars) {
        lines.push(word.slice(index, index + maxChars));
      }
    } else {
      line = next;
    }
  }

  if (line) lines.push(line);
  return lines;
}

function textBlock({
  text,
  x,
  y,
  maxChars,
  size = 16,
  lineHeight = 21,
  fill = colors.body,
  weight = 400,
  family = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
}) {
  const tspans = wrapText(text, maxChars)
    .map((line, index) => {
      const dy = index === 0 ? 0 : lineHeight;
      return `<tspan x="${x}" dy="${dy}">${escapeXml(line)}</tspan>`;
    })
    .join("");
  return `<text x="${x}" y="${y}" fill="${fill}" font-family="${family}" font-size="${size}" font-weight="${weight}" letter-spacing="0">${tspans}</text>`;
}

function pill({ x, y, label, fill, stroke, text, width = null }) {
  const pillWidth = width ?? Math.max(82, label.length * 7 + 28);
  return `<g>
    <rect x="${x}" y="${y}" width="${pillWidth}" height="28" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1"/>
    <text x="${x + 14}" y="${y + 18}" fill="${text}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="800" letter-spacing="0">${escapeXml(label.toUpperCase())}</text>
  </g>`;
}

function svgShell({ width, height, title, body }) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(title)}</title>
  <desc id="desc">Generated Mesa operating-envelope chart from public Sundog data.</desc>
  <rect width="${width}" height="${height}" fill="${colors.paper}"/>
  <path d="M0 0H${width}V${height}H0Z" fill="url(#grid)" opacity="0.72"/>
  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M32 0H0V32" fill="none" stroke="${colors.line}" stroke-width="1" opacity="0.38"/>
    </pattern>
  </defs>
  ${body}
</svg>
`;
}

function buildClassBalance(data, { x, y, width, includeLegend = true }) {
  const rows = data.classBalance ?? [];
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  let cursorX = x;
  const segments = [];

  for (const row of rows) {
    const palette = classPalette[row.klass] ?? classPalette.ambiguous;
    const segmentWidth = total > 0 ? (row.count / total) * width : 0;
    segments.push(`<g>
      <rect x="${cursorX}" y="${y}" width="${segmentWidth}" height="42" fill="${palette.fill}"/>
      ${
        segmentWidth >= 78
          ? `<text x="${cursorX + 14}" y="${y + 26}" fill="#FFFFFF" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="14" font-weight="800" letter-spacing="0">${escapeXml(`${palette.label} ${row.count}`)}</text>`
          : ""
      }
    </g>`);
    cursorX += segmentWidth;
  }

  const legend = includeLegend
    ? rows
        .map((row, index) => {
          const palette = classPalette[row.klass] ?? classPalette.ambiguous;
          const column = index % 3;
          const legendX = x + column * 205;
          const legendY = y + 78 + Math.floor(index / 3) * 34;
          return `<g>
            <rect x="${legendX}" y="${legendY - 15}" width="16" height="16" rx="4" fill="${palette.fill}"/>
            <text x="${legendX + 24}" y="${legendY - 2}" fill="${colors.body}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="14" font-weight="700" letter-spacing="0">${escapeXml(`${palette.label}: ${row.count}`)}</text>
          </g>`;
        })
        .join("\n")
    : "";

  return `<g>
    <rect x="${x}" y="${y}" width="${width}" height="42" rx="8" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1"/>
    <clipPath id="classBalanceClip"><rect x="${x}" y="${y}" width="${width}" height="42" rx="8"/></clipPath>
    <g clip-path="url(#classBalanceClip)">${segments.join("\n")}</g>
    <rect x="${x}" y="${y}" width="${width}" height="42" rx="8" fill="none" stroke="${colors.line}" stroke-width="1.5"/>
    ${legend}
  </g>`;
}

function buildCliffPlot(data, { x, y, width, height, tier = "Medium" }) {
  const key = tier.toLowerCase();
  const points = data.cliff?.[key] ?? [];
  const threshold = data.summary.breachThresholds.find((item) => item.tier === tier);
  const yValues = points.map((point) => point.oldBasinPref).filter(Number.isFinite);
  const minY = Math.min(-0.5, ...yValues);
  const maxY = Math.max(6, ...yValues);
  const pad = { left: 54, right: 22, top: 24, bottom: 44 };
  const chartX = x + pad.left;
  const chartY = y + pad.top;
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const toX = (lambda) => chartX + Math.max(0, Math.min(1, lambda)) * chartW;
  const toY = (value) => chartY + chartH - ((value - minY) / (maxY - minY)) * chartH;
  const pathPoints = points.map((point) => `${toX(point.lambda)},${toY(point.oldBasinPref)}`).join(" ");
  const thresholdY = toY(1);
  const thresholdX = threshold ? toX(threshold.interpolated_lambda) : null;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const yTicks = [0, 1, 3, 5];

  return `<g>
    <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="10" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1"/>
    <rect x="${chartX}" y="${chartY}" width="${chartW}" height="${thresholdY - chartY}" fill="${colors.redSoft}" opacity="0.55"/>
    <rect x="${chartX}" y="${thresholdY}" width="${chartW}" height="${chartY + chartH - thresholdY}" fill="${colors.greenSoft}" opacity="0.62"/>
    ${ticks
      .map((tick) => {
        const tx = toX(tick);
        return `<line x1="${tx}" y1="${chartY}" x2="${tx}" y2="${chartY + chartH}" stroke="${colors.line}" stroke-width="1"/><text x="${tx}" y="${chartY + chartH + 25}" text-anchor="middle" fill="${colors.muted}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="12" letter-spacing="0">${fixed(tick, 2)}</text>`;
      })
      .join("\n")}
    ${yTicks
      .map((tick) => {
        const ty = toY(tick);
        return `<line x1="${chartX - 6}" y1="${ty}" x2="${chartX + chartW}" y2="${ty}" stroke="${tick === 1 ? colors.gold : colors.line}" stroke-width="${tick === 1 ? 2 : 1}" opacity="${tick === 1 ? 0.95 : 0.8}"/><text x="${chartX - 14}" y="${ty + 4}" text-anchor="end" fill="${colors.muted}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="12" letter-spacing="0">${fixed(tick, 1)}</text>`;
      })
      .join("\n")}
    <polyline points="${pathPoints}" fill="none" stroke="${colors.navy}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    ${
      thresholdX
        ? `<line x1="${thresholdX}" y1="${chartY}" x2="${thresholdX}" y2="${chartY + chartH}" stroke="${colors.gold}" stroke-width="2.5" stroke-dasharray="7 6"/>
           <text x="${Math.min(thresholdX + 14, chartX + chartW - 205)}" y="${chartY + 22}" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="13" font-weight="800" letter-spacing="0">${escapeXml(`breach lambda ~= ${fixed(threshold.interpolated_lambda, 6)}`)}</text>`
        : ""
    }
    ${points
      .map((point) => {
        const palette = classPalette[point.klass] ?? classPalette.ambiguous;
        const radius = point.marker ? 7 : 5.5;
        return `<circle cx="${toX(point.lambda)}" cy="${toY(point.oldBasinPref)}" r="${radius}" fill="${palette.fill}" stroke="#FFFFFF" stroke-width="2">
          <title>${escapeXml(`${point.label}: old_basin_pref=${fixed(point.oldBasinPref, 3)} (${point.klass})`)}</title>
        </circle>`;
      })
      .join("\n")}
    <text x="${chartX + chartW / 2}" y="${chartY + chartH + 42}" text-anchor="middle" fill="${colors.body}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="13" font-weight="700" letter-spacing="0">reward weight lambda</text>
    <text x="${x + 18}" y="${y + 24}" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="16" font-weight="850" letter-spacing="0">${escapeXml(`${tier} L-Mixed lambda cliff`)}</text>
    <text x="${x + 18}" y="${y + height - 14}" fill="${colors.muted}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="700" letter-spacing="0">old_basin_pref threshold = 1</text>
  </g>`;
}

function buildKSweepPlot(data, { x, y, width, height }) {
  const rows = data.kSweep ?? [];
  const pad = { left: 54, right: 22, top: 36, bottom: 42 };
  const chartX = x + pad.left;
  const chartY = y + pad.top;
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const maxK = Math.max(...rows.map((row) => row.k), 64);
  const toX = (k) => chartX + (Math.log2(k) / Math.log2(maxK)) * chartW;
  const toY = (value) => chartY + chartH - Math.max(0, Math.min(1, value)) * chartH;
  const pToC = rows.map((row) => `${toX(row.k)},${toY(row.protectedToCollapsed?.meanPatchSuccess ?? 0)}`).join(" ");
  const cToP = rows.map((row) => `${toX(row.k)},${toY(row.collapsedToProtected?.meanPatchSuccess ?? 0)}`).join(" ");

  return `<g>
    <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="10" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1"/>
    <text x="${x + 18}" y="${y + 25}" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="16" font-weight="850" letter-spacing="0">PCA k-sweep patch fingerprint</text>
    ${[0, 0.25, 0.5, 0.75, 1]
      .map((tick) => `<line x1="${chartX - 6}" y1="${toY(tick)}" x2="${chartX + chartW}" y2="${toY(tick)}" stroke="${colors.line}" stroke-width="1"/><text x="${chartX - 14}" y="${toY(tick) + 4}" text-anchor="end" fill="${colors.muted}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="12" letter-spacing="0">${fixed(tick, 2)}</text>`)
      .join("\n")}
    ${[1, 3, 5, 10, 32, 64]
      .map((tick) => `<line x1="${toX(tick)}" y1="${chartY}" x2="${toX(tick)}" y2="${chartY + chartH}" stroke="${colors.line}" stroke-width="1"/><text x="${toX(tick)}" y="${chartY + chartH + 25}" text-anchor="middle" fill="${colors.muted}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="12" letter-spacing="0">k${tick}</text>`)
      .join("\n")}
    <polyline points="${pToC}" fill="none" stroke="${colors.red}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    <polyline points="${cToP}" fill="none" stroke="${colors.green}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    ${rows
      .map((row) => `<circle cx="${toX(row.k)}" cy="${toY(row.protectedToCollapsed?.meanPatchSuccess ?? 0)}" r="5" fill="${colors.red}" stroke="#FFFFFF" stroke-width="2"/><circle cx="${toX(row.k)}" cy="${toY(row.collapsedToProtected?.meanPatchSuccess ?? 0)}" r="5" fill="${colors.green}" stroke="#FFFFFF" stroke-width="2"/>`)
      .join("\n")}
    <rect x="${chartX + 14}" y="${chartY + 14}" width="210" height="48" rx="8" fill="#FFFFFF" stroke="${colors.line}" stroke-width="1"/>
    <line x1="${chartX + 28}" y1="${chartY + 32}" x2="${chartX + 62}" y2="${chartY + 32}" stroke="${colors.red}" stroke-width="3"/>
    <text x="${chartX + 72}" y="${chartY + 36}" fill="${colors.body}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="700" letter-spacing="0">protected -> collapsed</text>
    <line x1="${chartX + 28}" y1="${chartY + 52}" x2="${chartX + 62}" y2="${chartY + 52}" stroke="${colors.green}" stroke-width="3"/>
    <text x="${chartX + 72}" y="${chartY + 56}" fill="${colors.body}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="700" letter-spacing="0">collapsed -> protected</text>
  </g>`;
}

function buildEvidencePanel(data) {
  const width = 1280;
  const height = 820;
  const summary = data.summary;
  const mediumThreshold = summary.breachThresholds.find((item) => item.tier === "Medium");
  const smallThreshold = summary.breachThresholds.find((item) => item.tier === "Small");

  const body = `
    <rect x="48" y="40" width="1184" height="740" rx="14" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1.5"/>
    <text x="78" y="88" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="38" font-weight="850" letter-spacing="0">Mesa Optimization Envelope</text>
    ${pill({ x: 78, y: 112, label: "Small/Medium only", fill: colors.blueSoft, stroke: colors.blue, text: colors.navy, width: 170 })}
    ${pill({ x: 260, y: 112, label: `${summary.policyCount} policies`, fill: colors.slateSoft, stroke: colors.slate, text: "#37465A", width: 126 })}
    ${pill({ x: 398, y: 112, label: `${summary.missingCount} missing cells`, fill: colors.greenSoft, stroke: colors.green, text: "#214B36", width: 146 })}
    ${textBlock({
      text: data.claimBoundary,
      x: 78,
      y: 174,
      maxChars: 94,
      size: 17,
      lineHeight: 23,
      fill: colors.body,
      weight: 500,
    })}
    <text x="78" y="268" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="18" font-weight="850" letter-spacing="0">Class balance across audited cells</text>
    ${buildClassBalance(data, { x: 78, y: 294, width: 510, includeLegend: true })}
    <rect x="78" y="454" width="510" height="188" rx="10" fill="${colors.paper}" stroke="${colors.line}" stroke-width="1"/>
    <text x="104" y="491" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="20" font-weight="850" letter-spacing="0">Breach thresholds</text>
    <text x="104" y="529" fill="${colors.body}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="15" font-weight="700" letter-spacing="0">Small:  lambda ~= ${fixed(smallThreshold?.interpolated_lambda, 6)} | signature weight ~= ${fixed(smallThreshold?.signature_weight_interpolated, 6)}</text>
    <text x="104" y="562" fill="${colors.body}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="15" font-weight="700" letter-spacing="0">Medium: lambda ~= ${fixed(mediumThreshold?.interpolated_lambda, 6)} | signature weight ~= ${fixed(mediumThreshold?.signature_weight_interpolated, 6)}</text>
    ${textBlock({
      text: "Read as an operating-envelope map: where reward-coupled behavior takes over, where field attachment holds, and which cells remain fragile or ambiguous.",
      x: 104,
      y: 604,
      maxChars: 62,
      size: 15,
      lineHeight: 20,
      fill: colors.body,
      weight: 500,
    })}
    ${buildCliffPlot(data, { x: 626, y: 254, width: 550, height: 388, tier: "Medium" })}
    <line x1="78" y1="690" x2="1176" y2="690" stroke="${colors.line}" stroke-width="1"/>
    <text x="78" y="730" fill="${colors.muted}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="14" font-weight="700" letter-spacing="0">Generated from public/data/mesa-public-charts.json. This is an in-vitro Small/Medium result, not a universal mesa-optimization claim.</text>
  `;

  return svgShell({ width, height, title: "Mesa Optimization Envelope", body });
}

function buildCliffMini(data) {
  return svgShell({
    width: 900,
    height: 480,
    title: "Mesa Medium Lambda Cliff",
    body: `
      <text x="38" y="50" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="30" font-weight="850" letter-spacing="0">Mesa Medium Lambda Cliff</text>
      <text x="38" y="82" fill="${colors.body}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="15" font-weight="600" letter-spacing="0">old_basin_pref crosses the pre-set threshold between lambda=0.95 and lambda=0.97.</text>
      ${buildCliffPlot(data, { x: 38, y: 112, width: 824, height: 318, tier: "Medium" })}
    `,
  });
}

function buildClassBalanceStrip(data) {
  return svgShell({
    width: 900,
    height: 250,
    title: "Mesa Class Balance Strip",
    body: `
      <text x="38" y="50" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="30" font-weight="850" letter-spacing="0">Mesa Class Balance</text>
      <text x="38" y="82" fill="${colors.body}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="15" font-weight="600" letter-spacing="0">22 audited cells; missing cells = 0.</text>
      ${buildClassBalance(data, { x: 38, y: 116, width: 824, includeLegend: true })}
    `,
  });
}

function buildKSweepFingerprint(data) {
  return svgShell({
    width: 900,
    height: 430,
    title: "Mesa k-sweep Patch Fingerprint",
    body: `
      ${buildKSweepPlot(data, { x: 38, y: 36, width: 824, height: 340 })}
      <text x="38" y="404" fill="${colors.muted}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="13" font-weight="700" letter-spacing="0">Patch success rises after k=1; exported for mesa.html interpretation, not homepage overclaim.</text>
    `,
  });
}

async function writeSvg(relativePath, svg) {
  const absolutePath = join(root, relativePath);
  await mkdir(dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, svg, "utf8");
  console.log(`mesa public media built: ${relativePath}`);
}

async function main() {
  const data = JSON.parse(await readFile(join(root, inputPath), "utf8"));
  await writeSvg(outputs.evidencePanel, buildEvidencePanel(data));
  await writeSvg(outputs.cliffMini, buildCliffMini(data));
  await writeSvg(outputs.classBalanceStrip, buildClassBalanceStrip(data));
  await writeSvg(outputs.kSweepFingerprint, buildKSweepFingerprint(data));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
