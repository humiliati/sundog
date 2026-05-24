import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const inputPath = "public/data/structural-failure-boundary-map.json";
const outputPath = "public/media/structural-boundary-five-locus-map.svg";

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
};

const classPalette = {
  "eligible-abstain": {
    stroke: colors.green,
    fill: colors.greenSoft,
    pillFill: "#D9EBE1",
    text: "#214B36",
  },
  cutoff: {
    stroke: colors.amber,
    fill: colors.amberSoft,
    pillFill: "#F8E5AF",
    text: "#684811",
  },
  merge: {
    stroke: colors.amber,
    fill: colors.amberSoft,
    pillFill: "#F8E5AF",
    text: "#684811",
  },
  "permanent-fail": {
    stroke: colors.red,
    fill: colors.redSoft,
    pillFill: "#F1D4CF",
    text: "#6B3029",
  },
  admissibility: {
    stroke: colors.slate,
    fill: colors.slateSoft,
    pillFill: "#DCE4EC",
    text: "#37465A",
  },
};

const statusPalette = {
  pass: { fill: colors.greenSoft, stroke: colors.green, text: "#214B36" },
  reclassified: { fill: colors.amberSoft, stroke: colors.amber, text: "#684811" },
  "regime-separability": { fill: "#EAF1FA", stroke: colors.navy, text: colors.navy },
  hold: { fill: colors.redSoft, stroke: colors.red, text: "#6B3029" },
  open: { fill: colors.slateSoft, stroke: colors.slate, text: "#37465A" },
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
    if (word.length > maxChars) {
      if (line) {
        lines.push(line);
        line = "";
      }
      for (let index = 0; index < word.length; index += maxChars) {
        lines.push(word.slice(index, index + maxChars));
      }
      continue;
    }

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
  family = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  className = "",
}) {
  const lines = wrapText(text, maxChars);
  const classAttribute = className ? ` class="${escapeXml(className)}"` : "";
  const tspans = lines
    .map((line, index) =>
      `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${escapeXml(line)}</tspan>`,
    )
    .join("");
  return `<text${classAttribute} x="${x}" y="${y}" fill="${fill}" font-family="${family}" font-size="${size}" font-weight="${weight}" letter-spacing="0">${tspans}</text>`;
}

function textLines(text, maxChars) {
  return wrapText(text, maxChars).length;
}

function dataAttributes(attributes = {}) {
  return Object.entries(attributes)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => {
      const attributeName = key.replace(/[A-Z]/g, (match) => `-${match.toLowerCase()}`);
      return ` data-${attributeName}="${escapeXml(value)}"`;
    })
    .join("");
}

function pill({
  x,
  y,
  label,
  fill,
  stroke,
  text,
  width = null,
  title = "",
  className = "",
  data = {},
}) {
  const labelWidth = width ?? Math.max(80, label.length * 7.2 + 28);
  const classAttribute = className ? ` class="${escapeXml(className)}"` : "";
  return `<g${classAttribute}${dataAttributes(data)}>
    ${title ? `<title>${escapeXml(title)}</title>` : ""}
    <rect x="${x}" y="${y}" width="${labelWidth}" height="28" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1"/>
    <text x="${x + 14}" y="${y + 18}" fill="${text}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="12" font-weight="800" letter-spacing="0">${escapeXml(label.toUpperCase())}</text>
  </g>`;
}

function statusChipLabel(item) {
  const shortLabels = {
    p0: "P0 map: pass",
    p1: "P1: pass",
    "p2-first-cut": "P2 first cut: reclassified",
    cut2: "Cut 2: separability",
    cut3: "Cut 3: hold",
    h0: "H0 calibration: open",
  };
  return shortLabels[item.id] ?? `${item.label}: ${item.status}`;
}

function statusChips(statusLadder, x, y, maxWidth) {
  let cursorX = x;
  let cursorY = y;
  const chips = [];

  for (const item of statusLadder) {
    const palette = statusPalette[item.status] ?? statusPalette.open;
    const fullLabel = `${item.label}: ${item.status}`;
    const label = statusChipLabel(item);
    const width = Math.min(260, Math.max(112, label.length * 6.8 + 26));
    if (cursorX + width > x + maxWidth) {
      cursorX = x;
      cursorY += 36;
    }
    chips.push(
      pill({
        x: cursorX,
        y: cursorY,
        label,
        width,
        title: item.publicLabel ? `${fullLabel}. ${item.publicLabel}` : fullLabel,
        className: "status-chip",
        data: {
          id: item.id,
          label: item.label,
          status: item.status,
          publicLabel: item.publicLabel,
          source: item.source,
        },
        ...palette,
      }),
    );
    cursorX += width + 10;
  }

  return {
    markup: chips.join("\n"),
    bottom: cursorY + 28,
  };
}

function buildSvg(data) {
  const width = 1280;
  const margin = 48;
  const contentWidth = width - margin * 2;
  const tableTop = 252;
  const headerHeight = 42;
  const gap = 14;
  const columns = [
    { id: "handle", label: "Handle", x: margin + 18, width: 236, maxChars: 27 },
    { id: "eligible", label: "Eligible window", x: margin + 272, width: 270, maxChars: 35 },
    { id: "mustBreak", label: "Must break", x: margin + 560, width: 330, maxChars: 44 },
    { id: "correlateTell", label: "Correlate tell", x: margin + 908, width: 288, maxChars: 38 },
  ];
  const colRects = [
    { x: margin, width: 254 },
    { x: margin + 254 + gap, width: 288 },
    { x: margin + 254 + gap + 288 + gap, width: 348 },
    { x: margin + 254 + gap + 288 + gap + 348 + gap, width: 280 },
  ];

  const chips = statusChips(data.statusLadder, margin, 174, contentWidth);
  let rowY = tableTop + headerHeight;
  const rows = [];

  for (const locus of data.loci) {
    const maxLines = Math.max(
      textLines(`${locus.title}. ${locus.handle}`, columns[0].maxChars),
      textLines(locus.eligible, columns[1].maxChars),
      textLines(locus.mustBreak, columns[2].maxChars),
      textLines(locus.correlateTell, columns[3].maxChars),
    );
    const rowHeight = Math.max(134, maxLines * 18 + 68);
    const palette = classPalette[locus.visualClass] ?? classPalette.admissibility;
    const y = rowY;

    rows.push(`<g class="boundary-row"${dataAttributes({
      id: locus.id,
      title: locus.title,
      status: locus.displayStatus,
      visualClass: locus.visualClass,
      handle: locus.handle,
    })}>
      <title>${escapeXml(`${locus.id}: ${locus.title} - ${locus.displayStatus}`)}</title>
      <rect x="${margin}" y="${y}" width="${contentWidth}" height="${rowHeight}" rx="8" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1"/>
      <rect x="${margin}" y="${y}" width="8" height="${rowHeight}" rx="4" fill="${palette.stroke}"/>
      ${colRects
        .slice(1)
        .map(
          (col) =>
            `<line x1="${col.x - gap / 2}" y1="${y + 18}" x2="${col.x - gap / 2}" y2="${y + rowHeight - 18}" stroke="${colors.line}" stroke-width="1"/>`,
        )
        .join("\n")}
      <circle cx="${margin + 34}" cy="${y + 34}" r="17" fill="${palette.fill}" stroke="${palette.stroke}" stroke-width="1.5"/>
      <text x="${margin + 24}" y="${y + 40}" fill="${palette.text}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="15" font-weight="800" letter-spacing="0">${escapeXml(locus.id)}</text>
      ${pill({
        x: columns[0].x + 45,
        y: y + 18,
        label: locus.displayStatus,
        fill: palette.pillFill,
        stroke: palette.stroke,
        text: palette.text,
        title: `${locus.id}: ${locus.title} - ${locus.displayStatus}`,
        className: "locus-status-chip",
        data: {
          id: locus.id,
          status: locus.displayStatus,
        },
      })}
      ${textBlock({
        text: locus.title,
        x: columns[0].x,
        y: y + 73,
        maxChars: columns[0].maxChars,
        size: 16,
        lineHeight: 20,
        fill: colors.ink,
        weight: 800,
      })}
      ${textBlock({
        text: locus.handle,
        x: columns[0].x,
        y: y + 99,
        maxChars: columns[0].maxChars,
        size: 13,
        lineHeight: 17,
        fill: colors.muted,
        family: "ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace",
      })}
      ${textBlock({
        text: locus.eligible,
        x: columns[1].x,
        y: y + 36,
        maxChars: columns[1].maxChars,
        size: 14,
        lineHeight: 18,
      })}
      ${textBlock({
        text: locus.mustBreak,
        x: columns[2].x,
        y: y + 36,
        maxChars: columns[2].maxChars,
        size: 14,
        lineHeight: 18,
      })}
      ${textBlock({
        text: locus.correlateTell,
        x: columns[3].x,
        y: y + 36,
        maxChars: columns[3].maxChars,
        size: 14,
        lineHeight: 18,
      })}
    </g>`);

    rowY += rowHeight + 10;
  }

  const height = rowY + 78;
  const sourceText = `Generated from ${inputPath}; source map: ${data.sourcePaths.boundaryMap}`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">Structural failure boundary map</title>
  <desc id="desc">Five-locus map showing where a traceable system must fail, abstain, or switch when the closed-form inverse loses identifiability.</desc>
  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#E8EEF2" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="${width}" height="${height}" fill="${colors.paper}"/>
  <rect width="${width}" height="${height}" fill="url(#grid)" opacity="0.55"/>
  <rect x="${margin}" y="34" width="${contentWidth}" height="${height - 68}" rx="8" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1"/>
  <text x="${margin + 24}" y="78" fill="${colors.gold}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="13" font-weight="800" letter-spacing="0">${escapeXml(data.copyBlocks.eyebrow.toUpperCase())}</text>
  <text x="${margin + 24}" y="121" fill="${colors.ink}" font-family="Georgia, 'Times New Roman', serif" font-size="34" font-weight="700" letter-spacing="0">${escapeXml(data.copyBlocks.headline)}</text>
  ${textBlock({
    text: data.copyBlocks.dek,
    x: margin + 24,
    y: 153,
    maxChars: 82,
    size: 17,
    lineHeight: 23,
    fill: colors.body,
  })}
  ${chips.markup}
  <rect x="${margin}" y="${tableTop}" width="${contentWidth}" height="${headerHeight}" rx="8" fill="${colors.navy}"/>
  ${columns
    .map(
      (column) =>
        `<text x="${column.x}" y="${tableTop + 27}" fill="#FFFFFF" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="13" font-weight="800" letter-spacing="0">${escapeXml(column.label.toUpperCase())}</text>`,
    )
    .join("\n")}
  ${rows.join("\n")}
  <line x1="${margin + 24}" y1="${height - 49}" x2="${width - margin - 24}" y2="${height - 49}" stroke="${colors.line}" stroke-width="1"/>
  ${textBlock({
    text: data.copyBlocks.statusLine,
    x: margin + 24,
    y: height - 27,
    maxChars: 94,
    size: 13,
    lineHeight: 17,
    fill: colors.ink,
    weight: 700,
  })}
  <text x="${width - margin - 24}" y="${height - 27}" text-anchor="end" fill="${colors.muted}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="11" letter-spacing="0">${escapeXml(sourceText)}</text>
</svg>
`;
}

const data = JSON.parse(await readFile(join(root, inputPath), "utf8"));
const svg = buildSvg(data);
const absoluteOutput = join(root, outputPath);
await mkdir(dirname(absoluteOutput), { recursive: true });
await writeFile(absoluteOutput, svg, "utf8");
console.log(`structural boundary SVG built: ${outputPath}`);
