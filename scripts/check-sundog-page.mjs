import { access, readFile } from "node:fs/promises";
import { join } from "node:path";

const root = process.cwd();
const failures = [];

function ok(condition, message) {
  if (!condition) failures.push(message);
}

async function exists(path) {
  try {
    await access(join(root, path));
    return true;
  } catch {
    return false;
  }
}

function extractJsonLd(html) {
  const scripts = [...html.matchAll(/<script\s+type=["']application\/ld\+json["']>([\s\S]*?)<\/script>/gi)];
  return scripts.map((match) => JSON.parse(match[1]));
}

const html = await readFile(join(root, "sundog.html"), "utf8");
const upload = await readFile(join(root, "public/js/photo-upload.mjs"), "utf8");
const workbench = await readFile(join(root, "public/js/parhelion-workbench.mjs"), "utf8");
const phase6Drag = await readFile(join(root, "public/js/phase6-drag.mjs"), "utf8");
const sitemap = await readFile(join(root, "public/sitemap.xml"), "utf8");
const robots = await readFile(join(root, "public/robots.txt"), "utf8");

for (const id of [
  "hero",
  "what-it-is",
  "why-22",
  "sun-altitude-binding",
  "full-atlas",
  "try-your-photo",
  "history",
  "for-tools",
]) {
  ok(html.includes(`id="${id}"`), `sundog.html missing #${id}`);
}

ok(html.includes('<link rel="canonical" href="https://sundog.cc/sundog">'), "missing canonical sundog URL");
ok(html.includes('property="og:image"'), "missing og:image");
ok(html.includes('name="twitter:card"'), "missing Twitter card");
ok(html.includes('id="photo-upload-mount"'), "missing Phase 5 upload mount");
ok(html.includes("Show advanced controls"), "missing advanced-controls reveal");
ok(html.includes("33 assertions"), "Phase 3 assertion count is stale");
ok(html.includes("phase6-handle-parhelic-apex"), "missing Phase 6 parhelic apex handle styling");
ok(html.includes("phase6-handle-cza-apex"), "missing Phase 6 CZA apex handle styling");

const jsonLd = extractJsonLd(html);
const learningResource = jsonLd.find((item) => item["@type"] === "LearningResource");
ok(!!learningResource, "missing LearningResource JSON-LD");
// Wave-2 W15 (2026-05-14): author is a named human (WP:SELFPUB compliance —
// a corporate-only author is a self-published-source yellow flag on
// Wikipedia external-link review) with Stellar Aqua LLC as the affiliation.
ok(
  learningResource?.author?.name === "Jeffery Hughes Jr." &&
    learningResource?.author?.affiliation?.name === "Stellar Aqua LLC",
  "LearningResource author must be the named human contributor with Stellar Aqua LLC affiliation",
);
ok(Array.isArray(learningResource?.citation) && learningResource.citation.length >= 3, "LearningResource citations incomplete");
ok(!!learningResource?.educationalLevel, "LearningResource missing educationalLevel");
ok(Array.isArray(learningResource?.teaches) && learningResource.teaches.length >= 5, "LearningResource teaches list incomplete");

for (const endpoint of [
  "functions/api/sundog/_lib.js",
  "functions/api/sundog/health.js",
  "functions/api/sundog/policy.js",
  "functions/api/sundog/upload.js",
  "functions/api/sundog/delete.js",
]) {
  ok(await exists(endpoint), `missing ${endpoint}`);
}

ok(upload.includes('fetch("/api/sundog/health"'), "photo upload missing health probe");
ok(upload.includes('fetch("/api/sundog/upload"'), "photo upload missing upload POST");
ok(upload.includes("canvas.toBlob"), "photo upload missing canvas EXIF-strip path");
ok(upload.includes("sundog.submissions"), "photo upload missing deletion-url localStorage retention");
ok(workbench.includes("enablePhase6Drag"), "workbench missing Phase 6 drag wiring");
ok(phase6Drag.includes("data-phase6-binding") && phase6Drag.includes('binding: "parhelic-curvature"'), "Phase 6 drag missing parhelic curvature binding");
ok(phase6Drag.includes('binding: "cza-curvature"'), "Phase 6 drag missing CZA curvature binding");
ok(phase6Drag.includes("phase3.daggerOffset"), "Phase 6 drag missing sun-altitude dagger binding");
ok(sitemap.includes("https://sundog.cc/sundog"), "sitemap missing /sundog");
ok(robots.includes("Sitemap: https://sundog.cc/sitemap.xml"), "robots.txt missing sitemap pointer");

if (failures.length > 0) {
  console.error("Sundog page check failed:");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log("sundog page check passed");
