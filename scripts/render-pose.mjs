#!/usr/bin/env node
// scripts/render-pose.mjs — Phase 8C pose-pinning CLI
//
// Usage:
//   node scripts/render-pose.mjs <pose.json>           # pin single pose
//   node scripts/render-pose.mjs --all                 # pin every pose in public/poses/
//   node scripts/render-pose.mjs <pose.json> --out X   # custom output dir
//
// Emits per pose:
//   dist-poses/<name>.html              — self-contained pinned render (open in browser)
//   dist-poses/<name>.snapshot.json     — canonicalized pose (sorted keys, _meta dropped)
//
// The .html file embeds the SVG skeleton + workbench CSS + applies the pose's
// CSS variables and calls applyParhelionGeometry on load. No sliders, no drag.
// It's a read-only "pin" — anyone with the URL sees the same display.
//
// Headless PNG rendering is a follow-up (will use puppeteer). The CLI's
// primary gate-test value is canonicalization + standalone pin pages.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const POSES_DIR = path.join(REPO_ROOT, "public", "poses");
const TPL_SVG = path.join(__dirname, "templates", "svg-skeleton.html");
const TPL_CSS = path.join(__dirname, "templates", "svg-skeleton.css.html");

// --- Canonical schema -----------------------------------------------------
// Keys in canonical order; each maps to the CSS variable name (no `--` prefix)
// and the slider data-param used by the workbench. Order matches the canonical
// pose JSON shipped in public/poses/canonical-halo-atlas.json so that round-trip
// produces stable byte output.
const SCHEMA = [
  ["sunAltitudeDeg",            "sun-altitude",              "sun-altitude"],
  ["halo22Intensity",           "halo-22-intensity",         "halo-22-intensity"],
  ["halo46Intensity",           "halo-46-intensity",         "halo-46-intensity"],
  ["czaIntensity",              "cza-intensity",             "cza-intensity"],
  ["czaCurvature",              "cza-curvature",             "cza-curvature"],
  ["parhelicCircleIntensity",   "parhelic-circle-intensity", "parhelic-circle-intensity"],
  ["parhelicCurvature",         "parhelic-curvature",        "parhelic-curvature"],
  ["parhelicYOffsetR22",        "parhelic-y-offset-r22",     "parhelic-y-offset-r22"],
  ["parheliaIntensity",         "parhelia-intensity",        "parhelia-intensity"],
  ["parheliaDaggerLength",      "parhelia-dagger-length",    "parhelia-dagger-length"],
  ["sunPillarIntensity",        "sun-pillar-intensity",      "sun-pillar-intensity"],
  ["sunPillarLength",           "sun-pillar-length",         "sun-pillar-length"],
  ["compassRayLength",          "compass-ray-length",        "compass-ray-length"],
  ["compassRotationDeg",        "compass-rotation-deg",      "compass-rotation-deg"],
  ["dispersionWidth",           "dispersion-width",          "dispersion-width"],
  ["rainbowSaturation",         "rainbow-saturation",        "rainbow-saturation"],
  ["secondarySunsStrength",     "secondary-suns-strength",   "secondary-suns-strength"],
  ["ringOverlapBias",           "ring-overlap-bias",         "ring-overlap-bias"],
  ["supralateralIntensity",     "supralateral-intensity",    "supralateral-intensity"],
  ["upperTangentIntensity",     "upper-tangent-intensity",   "upper-tangent-intensity"],
  ["lowerTangentIntensity",     "lower-tangent-intensity",   "lower-tangent-intensity"],
  ["suncaveParryIntensity",     "suncave-parry-intensity",   "suncave-parry-intensity"],
  ["parrySupralateralIntensity","parry-supralateral-intensity","parry-supralateral-intensity"],
  ["infralateralIntensity",     "infralateral-intensity",    "infralateral-intensity"],
  ["idleScintillationAmplitude","idle-scintillation-amplitude","idle-scintillation-amplitude"],
];
const VALID_MODELS = new Set(["legacy", "halo_scaffold", "halo_governed", "halo_atlas"]);

// --- Pose hydration -------------------------------------------------------
// Mirrors public/js/parhelion-workbench.mjs::findPoseValue — accepts both
// camelCase (canonical) and kebab-case (snapshot) keys.
const CAMEL_OVERRIDES = {
  "sun-altitude": "sunAltitudeDeg",
  "parhelic-y-offset-r22": "parhelicYOffsetR22",
  "compass-rotation-deg": "compassRotationDeg",
};
function kebabToCamel(s) {
  return s.replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase());
}
function findValue(pose, camelKey, kebabKey) {
  if (Object.prototype.hasOwnProperty.call(pose, camelKey)) return pose[camelKey];
  if (Object.prototype.hasOwnProperty.call(pose, kebabKey)) return pose[kebabKey];
  const override = CAMEL_OVERRIDES[kebabKey];
  if (override && Object.prototype.hasOwnProperty.call(pose, override)) return pose[override];
  const altCamel = kebabToCamel(kebabKey);
  if (altCamel !== camelKey && Object.prototype.hasOwnProperty.call(pose, altCamel)) return pose[altCamel];
  return undefined;
}

// --- Validate -------------------------------------------------------------
function validatePose(pose, source) {
  const errors = [];
  const warnings = [];
  if (!pose || typeof pose !== "object") {
    errors.push(`${source}: not a JSON object`);
    return { errors, warnings, resolved: {} };
  }
  const model = pose.geometryModel;
  if (!model) warnings.push("geometryModel missing — will default to halo_atlas at render time");
  else if (!VALID_MODELS.has(model)) errors.push(`unknown geometryModel: ${JSON.stringify(model)}`);

  const resolved = { geometryModel: VALID_MODELS.has(model) ? model : "halo_atlas" };
  for (const [camel, css, kebab] of SCHEMA) {
    const v = findValue(pose, camel, kebab);
    if (v === undefined || v === null) {
      warnings.push(`field missing: ${camel} (no slider hydration)`);
      continue;
    }
    const num = Number(v);
    if (!Number.isFinite(num)) {
      errors.push(`${camel}: non-numeric ${JSON.stringify(v)}`);
      continue;
    }
    resolved[camel] = num;
  }
  return { errors, warnings, resolved };
}

// --- Emit canonical snapshot ---------------------------------------------
function emitCanonical(resolved) {
  const out = { geometryModel: resolved.geometryModel };
  for (const [camel] of SCHEMA) {
    if (Object.prototype.hasOwnProperty.call(resolved, camel)) out[camel] = resolved[camel];
  }
  return JSON.stringify(out, null, 2) + "\n";
}

// --- Emit standalone pinned HTML -----------------------------------------
function emitPinnedHtml(name, resolved, poseMeta) {
  const svgSkeleton = fs.readFileSync(TPL_SVG, "utf8");
  const styleBlock = fs.readFileSync(TPL_CSS, "utf8");

  const cssLines = [`  --geometry-model: ${resolved.geometryModel};`];
  for (const [camel, css] of SCHEMA) {
    if (resolved[camel] === undefined) continue;
    cssLines.push(`  --${css}: ${resolved[camel]};`);
  }
  const cssVars = `:root {\n${cssLines.join("\n")}\n}`;

  // Inline the resolved pose JSON for any tooling that wants to re-extract it
  const poseJson = JSON.stringify({ geometryModel: resolved.geometryModel,
    ...Object.fromEntries(SCHEMA.map(([k]) => [k, resolved[k]]).filter(([,v]) => v !== undefined))
  }, null, 2);

  const title = (poseMeta && poseMeta.name) ? `Sundog pose: ${poseMeta.name}` : `Sundog pose: ${name}`;
  const desc = (poseMeta && poseMeta.description) ? poseMeta.description : "";

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${escapeHtml(title)}</title>
  <meta name="description" content="${escapeHtml(desc)}" />
  <link rel="canonical" href="https://sundog.cc/poses/${escapeHtml(name)}.html" />
${styleBlock}
  <style>${cssVars}</style>
  <style>
    body { margin: 0; background: #0b1220; color: #c8d0e0; font-family: ui-sans-serif,system-ui,sans-serif; }
    .pose-stage { display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 1rem; }
    .pose-stage svg { width: min(95vw, 1100px); height: auto; }
    .pose-caption { position: fixed; top: 1rem; left: 1rem; max-width: 28rem; font-size: 0.85rem; line-height: 1.4; opacity: 0.75; }
    .pose-caption h1 { font-size: 1rem; margin: 0 0 0.4rem; font-weight: 600; }
    .pose-caption .meta { font-family: ui-monospace,monospace; font-size: 0.72rem; opacity: 0.7; margin-top: 0.5rem; }
    .pose-caption a { color: #8fb8ff; }
  </style>
</head>
<body>
  <div class="pose-caption">
    <h1>${escapeHtml(title)}</h1>
    <div>${escapeHtml(desc)}</div>
    <div class="meta">model: ${resolved.geometryModel} · h=${resolved.sunAltitudeDeg ?? "?"}°</div>
    <div class="meta"><a href="/sundog-workbench.html">Open in workbench →</a></div>
  </div>
  <div class="pose-stage">
${indent(svgSkeleton, 4)}
  </div>
  <script type="application/json" id="pose-data">${escapeJsonForScript(poseJson)}</script>
  <script type="module">
    import { applyParhelionGeometry } from "/js/parhelion-geometry.mjs";
    const svg = document.getElementById("parhelion-svg");
    const rootStyle = getComputedStyle(document.documentElement);
    applyParhelionGeometry({ svg, rootStyle, model: "${resolved.geometryModel}" });
  </script>
</body>
</html>
`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function escapeJsonForScript(s) {
  // application/json script tags interpret </script> as a tag close. Escape just that.
  return s.replace(/<\/(script)/gi, "<\\/$1");
}
function indent(s, n) {
  const pad = " ".repeat(n);
  return s.split("\n").map(l => l ? pad + l : l).join("\n");
}

// --- Main -----------------------------------------------------------------
function main() {
  const args = process.argv.slice(2);
  if (!args.length || args.includes("--help") || args.includes("-h")) {
    console.error("usage: render-pose.mjs <pose.json> [--out <dir>]");
    console.error("       render-pose.mjs --all [--out <dir>]");
    process.exit(args.length ? 0 : 1);
  }
  let outDir = path.join(REPO_ROOT, "dist-poses");
  const outIdx = args.indexOf("--out");
  if (outIdx >= 0) {
    outDir = path.resolve(args[outIdx + 1]);
    args.splice(outIdx, 2);
  }
  fs.mkdirSync(outDir, { recursive: true });

  let inputs = [];
  if (args.includes("--all")) {
    inputs = fs.readdirSync(POSES_DIR)
      .filter(f => f.endsWith(".json") && !f.startsWith("_"))
      .map(f => path.join(POSES_DIR, f));
  } else {
    inputs = args.filter(a => !a.startsWith("--")).map(a => path.resolve(a));
  }
  if (!inputs.length) {
    console.error("no inputs");
    process.exit(1);
  }

  let totalErrors = 0;
  for (const input of inputs) {
    const raw = JSON.parse(fs.readFileSync(input, "utf8"));
    const { errors, warnings, resolved } = validatePose(raw, input);
    const name = path.basename(input, ".json");
    const meta = raw._meta || {};
    console.log(`\n📌 ${name}`);
    console.log(`   model: ${resolved.geometryModel}  sunAlt: ${resolved.sunAltitudeDeg ?? "?"}°`);
    if (warnings.length) console.log(`   ⚠ ${warnings.length} warnings`);
    for (const w of warnings) console.log(`     - ${w}`);
    if (errors.length) {
      totalErrors += errors.length;
      console.log(`   ✗ ${errors.length} errors`);
      for (const e of errors) console.log(`     - ${e}`);
      continue;
    }
    const html = emitPinnedHtml(name, resolved, meta);
    const snapshot = emitCanonical(resolved);
    fs.writeFileSync(path.join(outDir, `${name}.html`), html);
    fs.writeFileSync(path.join(outDir, `${name}.snapshot.json`), snapshot);
    console.log(`   → ${path.relative(REPO_ROOT, path.join(outDir, name + ".html"))}`);
    console.log(`   → ${path.relative(REPO_ROOT, path.join(outDir, name + ".snapshot.json"))}`);
  }

  if (totalErrors) {
    console.error(`\n${totalErrors} errors across ${inputs.length} pose(s)`);
    process.exit(2);
  }
  console.log(`\nDone — ${inputs.length} pose(s) pinned to ${path.relative(REPO_ROOT, outDir)}/`);
}

main();
