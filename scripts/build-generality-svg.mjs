import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const inputPath = "public/data/high-stakes-generality-gallery.json";
const mediaDir = "public/media";

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
  blueSoft: "#EAF1FA",
  slate: "#64748B",
  slateSoft: "#EDF2F7",
};

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

function textBlock({
  text,
  x,
  y,
  maxChars,
  size = 16,
  lineHeight = 21,
  fill = colors.body,
  weight = 400,
  anchor = "start",
}) {
  const lines = wrapText(text, maxChars);
  const tspans = lines
    .map((line, index) => {
      const dy = index === 0 ? 0 : lineHeight;
      return `<tspan x="${x}" dy="${dy}">${escapeXml(line)}</tspan>`;
    })
    .join("");
  return `<text x="${x}" y="${y}" fill="${fill}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" letter-spacing="0">${tspans}</text>`;
}

function metricLabel(metric) {
  if (metric.unit === "fraction") return `${Number(metric.value).toFixed(3)}`;
  return String(metric.value);
}

function statusChip(project, data, x, y, width = 154) {
  const palette = data.statusPalette[project.status] ?? {
    label: project.status,
    fill: colors.slateSoft,
    stroke: colors.slate,
    text: "#37465A",
  };
  return `<g>
    <rect x="${x}" y="${y}" width="${width}" height="30" rx="8" fill="${palette.fill}" stroke="${palette.stroke}" stroke-width="1"/>
    <text x="${x + 14}" y="${y + 20}" fill="${palette.text}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="800" letter-spacing="0">${escapeXml(palette.label.toUpperCase())}</text>
  </g>`;
}

function svgShell({ width, height, title, body }) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(title)}</title>
  <desc id="desc">Generated Sundog generality chart.</desc>
  <rect width="${width}" height="${height}" fill="${colors.paper}"/>
  ${body}
</svg>
`;
}

function projectCard(project, data, x, y, width, height) {
  const palette = data.statusPalette[project.status];
  const metricGap = 102;
  const metrics = project.metrics
    .slice(0, 3)
    .map((metric, index) => {
      const mx = x + 26 + index * metricGap;
      return `<g>
        <text x="${mx}" y="${y + height - 58}" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="22" font-weight="850" letter-spacing="0">${escapeXml(metricLabel(metric))}</text>
        ${textBlock({
          text: `${metric.label}${metric.unit && metric.unit !== "fraction" ? ` (${metric.unit})` : ""}`,
          x: mx,
          y: y + height - 36,
          maxChars: 14,
          size: 10,
          lineHeight: 13,
          fill: colors.muted,
          weight: 700,
        })}
      </g>`;
    })
    .join("");

  return `<g class="generality-card" data-project="${escapeXml(project.id)}">
    <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="8" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1"/>
    <rect x="${x}" y="${y}" width="8" height="${height}" rx="4" fill="${palette.stroke}"/>
    ${statusChip(project, data, x + width - 176, y + 20)}
    <text x="${x + 26}" y="${y + 40}" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="24" font-weight="850" letter-spacing="0">${escapeXml(project.shortName)}</text>
    ${textBlock({ text: project.chartHeadline, x: x + 26, y: y + 68, maxChars: 32, size: 13, lineHeight: 17, fill: colors.gold, weight: 800 })}
    ${textBlock({ text: project.currentRead, x: x + 26, y: y + 116, maxChars: 47, size: 14, lineHeight: 19, fill: colors.body })}
    <line x1="${x + 26}" y1="${y + height - 88}" x2="${x + width - 26}" y2="${y + height - 88}" stroke="${colors.line}" stroke-width="1"/>
    ${metrics}
  </g>`;
}

function buildStatusMatrix(data) {
  const width = 1280;
  const cardWidth = 570;
  const cardHeight = 250;
  const gap = 26;
  const left = 56;
  const top = 228;
  const rows = Math.ceil(data.projects.length / 2);
  const height = top + rows * cardHeight + (rows - 1) * gap + 70;
  const cards = data.projects
    .map((project, index) => {
      const col = index % 2;
      const row = Math.floor(index / 2);
      return projectCard(
        project,
        data,
        left + col * (cardWidth + gap),
        top + row * (cardHeight + gap),
        cardWidth,
        cardHeight,
      );
    })
    .join("\n");

  const body = `
    <text x="${left}" y="72" fill="${colors.gold}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="14" font-weight="850" letter-spacing="0.08em">HIGH-STAKES GENERALITY</text>
    ${textBlock({ text: "Transfer tests, walls, and review gates", x: left, y: 116, maxChars: 42, size: 38, lineHeight: 43, fill: colors.ink, weight: 850 })}
    ${textBlock({ text: data.brandFrame, x: left, y: 164, maxChars: 104, size: 18, lineHeight: 25, fill: colors.body })}
    ${cards}
  `;
  return svgShell({ width, height, title: "High-stakes generality status matrix", body });
}

function laneRow(project, data, x, y, width) {
  const palette = data.statusPalette[project.status];
  const stepW = 245;
  const stepGap = 22;
  const steps = [
    { label: "Question", value: project.transferQuestion },
    { label: "Current read", value: project.currentRead },
    { label: "Blocker", value: project.blocker },
    { label: "Next", value: project.nextAction },
  ];
  const boxes = steps
    .map((step, index) => {
      const bx = x + 210 + index * (stepW + stepGap);
      return `<g>
        <rect x="${bx}" y="${y}" width="${stepW}" height="126" rx="8" fill="${colors.panel}" stroke="${index === 2 ? palette.stroke : colors.line}" stroke-width="${index === 2 ? 1.5 : 1}"/>
        <text x="${bx + 16}" y="${y + 25}" fill="${index === 2 ? palette.text : colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="11" font-weight="850" letter-spacing="0.04em">${escapeXml(step.label.toUpperCase())}</text>
        ${textBlock({ text: step.value, x: bx + 16, y: y + 50, maxChars: 29, size: 12, lineHeight: 16, fill: colors.body })}
      </g>`;
    })
    .join("");
  const connectors = [0, 1, 2]
    .map((index) => {
      const cx = x + 210 + (index + 1) * stepW + index * stepGap;
      return `<line x1="${cx}" y1="${y + 63}" x2="${cx + stepGap}" y2="${y + 63}" stroke="${colors.line}" stroke-width="2" marker-end="url(#arrow)"/>`;
    })
    .join("");

  return `<g data-project="${escapeXml(project.id)}">
    <text x="${x}" y="${y + 32}" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="20" font-weight="850" letter-spacing="0">${escapeXml(project.shortName)}</text>
    ${statusChip(project, data, x, y + 48, 166)}
    ${boxes}
    ${connectors}
  </g>`;
}

function buildTransferLanes(data) {
  const width = 1480;
  const left = 52;
  const top = 190;
  const laneH = 150;
  const height = top + data.projects.length * laneH + 54;
  const lanes = data.projects
    .map((project, index) => laneRow(project, data, left, top + index * laneH, width - left * 2))
    .join("\n");
  const body = `
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="${colors.line}" stroke="none"/>
      </marker>
    </defs>
    <text x="${left}" y="72" fill="${colors.gold}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="14" font-weight="850" letter-spacing="0.08em">TRANSFER LANES</text>
    ${textBlock({ text: "What changed, what happened, what waits", x: left, y: 116, maxChars: 56, size: 36, lineHeight: 42, fill: colors.ink, weight: 850 })}
    ${textBlock({ text: "These are not victory cards. They are the current transfer questions and the gates that keep each result honest.", x: left, y: 158, maxChars: 110, size: 17, lineHeight: 23, fill: colors.body })}
    ${lanes}
  `;
  return svgShell({ width, height, title: "High-stakes generality transfer lanes", body });
}

function buildSingleProject(project, data) {
  const width = 900;
  const height = 660;
  const palette = data.statusPalette[project.status];
  const metricY = 474;
  const metrics = project.metrics
    .map((metric, index) => {
      const x = 72 + index * 198;
      return `<g>
        <rect x="${x}" y="${metricY}" width="164" height="86" rx="8" fill="${palette.fill}" stroke="${palette.stroke}" stroke-width="1"/>
        <text x="${x + 18}" y="${metricY + 36}" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="26" font-weight="850" letter-spacing="0">${escapeXml(metricLabel(metric))}</text>
        ${textBlock({ text: `${metric.label}${metric.unit && metric.unit !== "fraction" ? ` (${metric.unit})` : ""}`, x: x + 18, y: metricY + 60, maxChars: 18, size: 11, lineHeight: 14, fill: palette.text, weight: 800 })}
      </g>`;
    })
    .join("");
  const body = `
    <rect x="40" y="40" width="820" height="580" rx="8" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1"/>
    <rect x="40" y="40" width="10" height="580" rx="5" fill="${palette.stroke}"/>
    ${statusChip(project, data, 686, 70)}
    <text x="72" y="92" fill="${colors.gold}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="850" letter-spacing="0.08em">${escapeXml(project.category.toUpperCase())}</text>
    ${textBlock({ text: project.fullName, x: 72, y: 132, maxChars: 36, size: 32, lineHeight: 38, fill: colors.ink, weight: 850 })}
    ${textBlock({ text: project.currentRead, x: 72, y: 224, maxChars: 74, size: 16, lineHeight: 23, fill: colors.body })}
    <text x="72" y="324" fill="${palette.text}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="850" letter-spacing="0.06em">NEXT GATE</text>
    ${textBlock({ text: project.nextAction, x: 72, y: 350, maxChars: 84, size: 14, lineHeight: 20, fill: colors.body })}
    ${metrics}
  `;
  return svgShell({ width, height, title: `${project.shortName} generality card`, body });
}

const data = JSON.parse(await readFile(join(root, inputPath), "utf8"));
await mkdir(join(root, mediaDir), { recursive: true });

const outputs = [
  ["generality-status-matrix.svg", buildStatusMatrix(data)],
  ["generality-transfer-lanes.svg", buildTransferLanes(data)],
  ...data.projects.map((project) => [`generality-${project.slug}.svg`, buildSingleProject(project, data)]),
];

for (const [name, svg] of outputs) {
  const outputPath = join(root, mediaDir, name);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, svg, "utf8");
  console.log(`generality svg built: ${join(mediaDir, name)}`);
}
