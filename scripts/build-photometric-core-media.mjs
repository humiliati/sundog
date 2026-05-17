import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const inputPath = "results/analysis/analysis_summary.json";

const outputs = {
  data: "public/data/photometric-core-metrics.json",
  terminal: "public/media/photometric-terminal-intensity.svg",
  convergence: "public/media/photometric-convergence-time.svg",
};

const colors = {
  ink: "#173348",
  body: "#40505C",
  muted: "#6B7B88",
  line: "#DCE6EC",
  paper: "#F8FBFC",
  panel: "#FFFFFF",
  navy: "#1A3A52",
  blue: "#2C5F7F",
  blueSoft: "#EAF1FA",
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

const conditionOrder = ["photometric", "doa_direct", "doa_noisy", "random"];
const conditionMeta = {
  photometric: {
    label: "Photometric",
    shortLabel: "Photometric",
    fill: colors.green,
    soft: colors.greenSoft,
  },
  doa_direct: {
    label: "Target-aware oracle",
    shortLabel: "Oracle",
    fill: colors.navy,
    soft: colors.blueSoft,
  },
  doa_noisy: {
    label: "Noisy oracle",
    shortLabel: "Noisy",
    fill: colors.amber,
    soft: colors.amberSoft,
  },
  random: {
    label: "Random",
    shortLabel: "Random",
    fill: colors.slate,
    soft: colors.slateSoft,
  },
};

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fixed(value, digits = 3) {
  if (!Number.isFinite(value)) return "n/a";
  const text = Number(value).toFixed(digits);
  return text.includes(".") ? text.replace(/\.?0+$/, "") : text;
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

function svgShell({ width, height, title, body }) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(title)}</title>
  <desc id="desc">Generated Sundog photometric core metric chart from analysis summary.</desc>
  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M32 0H0V32" fill="none" stroke="${colors.line}" stroke-width="1" opacity="0.38"/>
    </pattern>
  </defs>
  <rect width="${width}" height="${height}" fill="${colors.paper}"/>
  <rect width="${width}" height="${height}" fill="url(#grid)" opacity="0.72"/>
  <rect x="30" y="28" width="${width - 60}" height="${height - 56}" rx="14" fill="${colors.panel}" stroke="${colors.line}" stroke-width="1.5"/>
  ${body}
</svg>
`;
}

function buildPublicData(summary) {
  const conditions = conditionOrder.map((id) => {
    const condition = summary.conditions[id];
    const meta = conditionMeta[id];
    return {
      id,
      label: meta.label,
      n: condition.n_seeds,
      terminalIntensity: {
        mean: condition.terminal_intensity.mean,
        median: condition.terminal_intensity.median,
        ci95: condition.terminal_intensity.ci95,
      },
      timeToThreshold: {
        medianSteps: condition.time_to_threshold.median,
        meanSteps: condition.time_to_threshold.mean,
        failedSeeds: condition.time_to_threshold.n_failed,
      },
    };
  });

  return {
    schemaVersion: 1,
    purpose:
      "Public chart source for the homepage core photometric metric panels. Generated from results/analysis/analysis_summary.json; do not hand-edit.",
    sourcePath: inputPath,
    claimBoundary:
      "Core MuJoCo mirror-alignment task only; no detected terminal-intensity difference at n=30, with slower acquisition for indirect photometric feedback.",
    conditions,
    tests: summary.tests,
    chartQueue: [
      {
        id: "photometric-terminal-intensity",
        target: "homepage core result metric panel",
        source: "conditions[*].terminalIntensity + photometric_vs_doa_direct_terminal_intensity",
        output: outputs.terminal,
        status: "generated-by-prebuild",
      },
      {
        id: "photometric-convergence-time",
        target: "homepage core result metric panel",
        source: "conditions[*].timeToThreshold",
        output: outputs.convergence,
        status: "generated-by-prebuild",
      },
    ],
  };
}

function buildTerminalSvg(data) {
  const width = 900;
  const height = 520;
  const chartX = 92;
  const chartY = 150;
  const chartW = 720;
  const chartH = 245;
  const yMin = 0;
  const yMax = 1;
  const toY = (value) => chartY + chartH - ((value - yMin) / (yMax - yMin)) * chartH;
  const test = data.tests.photometric_vs_doa_direct_terminal_intensity;
  const bars = data.conditions.map((condition, index) => {
    const meta = conditionMeta[condition.id];
    const groupW = chartW / data.conditions.length;
    const barW = 72;
    const x = chartX + index * groupW + groupW / 2 - barW / 2;
    const mean = condition.terminalIntensity.mean;
    const [ciLow, ciHigh] = condition.terminalIntensity.ci95;
    const y = toY(mean);
    const h = chartY + chartH - y;
    const ciX = x + barW / 2;
    const ciLowY = toY(ciLow);
    const ciHighY = toY(ciHigh);
    const valueLabel = mean < 0.001 ? "~0" : fixed(mean, 3);

    return `<g>
      <rect x="${x}" y="${y}" width="${barW}" height="${h}" rx="8" fill="${meta.fill}"/>
      <line x1="${ciX}" y1="${ciHighY}" x2="${ciX}" y2="${ciLowY}" stroke="${colors.ink}" stroke-width="2"/>
      <line x1="${ciX - 14}" y1="${ciHighY}" x2="${ciX + 14}" y2="${ciHighY}" stroke="${colors.ink}" stroke-width="2"/>
      <line x1="${ciX - 14}" y1="${ciLowY}" x2="${ciX + 14}" y2="${ciLowY}" stroke="${colors.ink}" stroke-width="2"/>
      <text x="${ciX}" y="${Math.min(y - 10, ciHighY - 10)}" text-anchor="middle" fill="${colors.ink}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="14" font-weight="800" letter-spacing="0">${escapeXml(valueLabel)}</text>
      <text x="${ciX}" y="${chartY + chartH + 34}" text-anchor="middle" fill="${colors.body}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="14" font-weight="800" letter-spacing="0">${escapeXml(meta.shortLabel)}</text>
      <text x="${ciX}" y="${chartY + chartH + 54}" text-anchor="middle" fill="${colors.muted}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="11" font-weight="700" letter-spacing="0">n=${condition.n}</text>
    </g>`;
  });

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((tick) => {
    const y = toY(tick);
    return `<line x1="${chartX - 8}" y1="${y}" x2="${chartX + chartW}" y2="${y}" stroke="${colors.line}" stroke-width="1"/>
    <text x="${chartX - 18}" y="${y + 4}" text-anchor="end" fill="${colors.muted}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="12" letter-spacing="0">${fixed(tick, 2)}</text>`;
  });

  const body = `
    <text x="64" y="76" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="34" font-weight="850" letter-spacing="0">Terminal Target Intensity</text>
    ${textBlock({
      text: `30 matched MuJoCo scenes. Photometric vs target-aware oracle: U=${fixed(test.U, 0)}, p=${fixed(test.p, 3)}; no detected terminal-intensity difference.`,
      x: 64,
      y: 110,
      maxChars: 96,
      size: 15,
      lineHeight: 20,
      fill: colors.body,
      weight: 600,
    })}
    <rect x="${chartX}" y="${chartY}" width="${chartW}" height="${chartH}" rx="10" fill="${colors.paper}" stroke="${colors.line}" stroke-width="1"/>
    ${ticks.join("\n")}
    ${bars.join("\n")}
    <text x="${chartX + chartW / 2}" y="${height - 46}" text-anchor="middle" fill="${colors.muted}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="13" font-weight="700" letter-spacing="0">Bars show mean terminal intensity; whiskers show 95% CI from results/analysis/analysis_summary.json.</text>
  `;

  return svgShell({ width, height, title: "Terminal Target Intensity", body });
}

function buildConvergenceSvg(data) {
  const width = 900;
  const height = 520;
  const chartX = 230;
  const chartY = 150;
  const chartW = 580;
  const rowH = 58;
  const maxDisplayed = 200;
  const testCondition = data.conditions.find((condition) => condition.id === "photometric");
  const oracleCondition = data.conditions.find((condition) => condition.id === "doa_direct");
  const ratio = testCondition.timeToThreshold.medianSteps / oracleCondition.timeToThreshold.medianSteps;

  const rows = data.conditions.map((condition, index) => {
    const meta = conditionMeta[condition.id];
    const y = chartY + index * rowH;
    const median = condition.timeToThreshold.medianSteps;
    const failed = condition.timeToThreshold.failedSeeds;
    const capped = failed === condition.n;
    const barW = Math.max(median > 0 ? 10 : 0, Math.min(median, maxDisplayed) / maxDisplayed * chartW);
    const label = capped ? `${fixed(median, 0)} censored` : `${fixed(median, 1)} steps`;
    const endMark = median > maxDisplayed ? `<path d="M${chartX + chartW + 4} ${y + 20}l12 10-12 10" fill="none" stroke="${meta.fill}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>` : "";

    return `<g>
      <text x="${chartX - 26}" y="${y + 34}" text-anchor="end" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="15" font-weight="850" letter-spacing="0">${escapeXml(meta.label)}</text>
      <rect x="${chartX}" y="${y + 10}" width="${chartW}" height="32" rx="8" fill="${meta.soft}" stroke="${colors.line}" stroke-width="1"/>
      <rect x="${chartX}" y="${y + 10}" width="${barW}" height="32" rx="8" fill="${meta.fill}"/>
      ${endMark}
      <text x="${Math.min(chartX + barW + 12, chartX + chartW - 112)}" y="${y + 31}" fill="${barW > chartW * 0.72 ? "#FFFFFF" : colors.ink}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="14" font-weight="850" letter-spacing="0">${escapeXml(label)}</text>
    </g>`;
  });

  const ticks = [0, 50, 100, 150, 200].map((tick) => {
    const x = chartX + (tick / maxDisplayed) * chartW;
    return `<line x1="${x}" y1="${chartY - 14}" x2="${x}" y2="${chartY + rowH * 4 - 10}" stroke="${colors.line}" stroke-width="1"/>
    <text x="${x}" y="${chartY + rowH * 4 + 22}" text-anchor="middle" fill="${colors.muted}" font-family="ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace" font-size="12" letter-spacing="0">${tick}</text>`;
  });

  const body = `
    <text x="64" y="76" fill="${colors.ink}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="34" font-weight="850" letter-spacing="0">Time-to-Convergence</text>
    ${textBlock({
      text: `Median steps to intensity 0.9. Indirect photometric feedback pays an acquisition cost: ${fixed(testCondition.timeToThreshold.medianSteps, 0)} vs ${fixed(oracleCondition.timeToThreshold.medianSteps, 1)} steps, about ${fixed(ratio, 1)}x slower.`,
      x: 64,
      y: 110,
      maxChars: 96,
      size: 15,
      lineHeight: 20,
      fill: colors.body,
      weight: 600,
    })}
    <rect x="${chartX}" y="${chartY - 14}" width="${chartW}" height="${rowH * 4 + 4}" rx="10" fill="${colors.paper}" stroke="${colors.line}" stroke-width="1"/>
    ${ticks.join("\n")}
    ${rows.join("\n")}
    <text x="${chartX + chartW / 2}" y="${height - 46}" text-anchor="middle" fill="${colors.muted}" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="13" font-weight="700" letter-spacing="0">Horizontal scale is 0-200 steps; random is fully censored at 500 and marked at the edge.</text>
  `;

  return svgShell({ width, height, title: "Time-to-Convergence", body });
}

async function writeText(relativePath, text) {
  const absolutePath = join(root, relativePath);
  await mkdir(dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, text, "utf8");
  console.log(`photometric core media built: ${relativePath}`);
}

async function main() {
  const summary = JSON.parse(await readFile(join(root, inputPath), "utf8"));
  const publicData = buildPublicData(summary);

  await writeText(outputs.data, `${JSON.stringify(publicData, null, 2)}\n`);
  await writeText(outputs.terminal, buildTerminalSvg(publicData));
  await writeText(outputs.convergence, buildConvergenceSvg(publicData));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
