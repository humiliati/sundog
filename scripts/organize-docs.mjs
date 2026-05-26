#!/usr/bin/env node

import { mkdir, readdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";

const root = process.cwd();
const apply = process.argv.includes("--apply");
const pendingPublicStubs = [];

const publicStubNotice = (oldPath, newPath) => {
  const oldDir = dirname(oldPath);
  const relTarget = relative(oldDir, newPath).replaceAll("\\", "/");
  return `# Document Moved

This compatibility pointer is kept so older public links to \`${oldPath}\` do
not break.

Canonical location: [${newPath}](${relTarget})
`;
};

const moves = [
  // Shadow Faraday public receipt set.
  ["docs/SHADOW_FARADAY.md", "docs/faraday/SHADOW_FARADAY.md", "public-stub"],
  ["docs/SUNDOG_V_FARADAY.md", "docs/faraday/SUNDOG_V_FARADAY.md", "public-stub"],
  ["docs/FARADAY_PHASE3_DERIVATIONS.md", "docs/faraday/FARADAY_PHASE3_DERIVATIONS.md", "public-stub"],
  ["docs/FARADAY_PHASE4_VERIFICATION.md", "docs/faraday/FARADAY_PHASE4_VERIFICATION.md", "public-stub"],
  ["docs/FARADAY_PHASE7_SPEC.md", "docs/faraday/FARADAY_PHASE7_SPEC.md", "public-stub"],
  ["docs/SHADOW_FARADAY_PHASE3_DERIVATION.md", "docs/faraday/SHADOW_FARADAY_PHASE3_DERIVATION.md", "public-stub"],

  // Isotrophy public receipt set.
  ["docs/sundog_v_isotrophy.md", "docs/isotrophy/sundog_v_isotrophy.md", "public-stub"],
  ["docs/SUNDOG_V_ISOTROPHY_KFACET.md", "docs/isotrophy/SUNDOG_V_ISOTROPHY_KFACET.md", "public-stub"],
  ["docs/ISOTROPHY_PROMO_HANDOFF_2026-05-24.md", "docs/isotrophy/ISOTROPHY_PROMO_HANDOFF_2026-05-24.md", "public-stub"],

  // K_facet appendices promoted out of the anniversary catch-all.
  ["internal/anniversary/kfacet-runner-spec.md", "docs/isotrophy/kfacet/kfacet-runner-spec.md"],
  ["internal/anniversary/kfacet_isotrophy_program_pause.md", "docs/isotrophy/kfacet/kfacet_isotrophy_program_pause.md"],
  ["internal/anniversary/kfacet_v03_freeze_b_comparison.md", "docs/isotrophy/kfacet/kfacet_v03_freeze_b_comparison.md"],
  ["internal/anniversary/kfacet_v03_gamma_crossm3_preregistration.md", "docs/isotrophy/kfacet/kfacet_v03_gamma_crossm3_preregistration.md"],
  ["internal/anniversary/kfacet_v03h_o617_deep_dive.md", "docs/isotrophy/kfacet/kfacet_v03h_o617_deep_dive.md"],
  ["internal/anniversary/kfacet_v03h_writeup.md", "docs/isotrophy/kfacet/kfacet_v03h_writeup.md"],
  ["internal/anniversary/kfacet_v04_writeup.md", "docs/isotrophy/kfacet/kfacet_v04_writeup.md"],
  ["internal/anniversary/kfacet_v04a_domain_map_preregistration.md", "docs/isotrophy/kfacet/kfacet_v04a_domain_map_preregistration.md"],
  ["internal/anniversary/kfacet_v04b_gamma3_form.md", "docs/isotrophy/kfacet/kfacet_v04b_gamma3_form.md"],
  ["internal/anniversary/kfacet_v04b_gamma3prime_form.md", "docs/isotrophy/kfacet/kfacet_v04b_gamma3prime_form.md"],
  ["internal/anniversary/kfacet_v04b_mechanism_preregistration.md", "docs/isotrophy/kfacet/kfacet_v04b_mechanism_preregistration.md"],
  ["internal/anniversary/kfacet_v05_writeup.md", "docs/isotrophy/kfacet/kfacet_v05_writeup.md"],
  ["internal/anniversary/kfacet_v05a_branch_map_form.md", "docs/isotrophy/kfacet/kfacet_v05a_branch_map_form.md"],
  ["internal/anniversary/kfacet_v05b_branch_predictor_form.md", "docs/isotrophy/kfacet/kfacet_v05b_branch_predictor_form.md"],
  ["internal/anniversary/kfacet_v06_mechanism_preregistration.md", "docs/isotrophy/kfacet/kfacet_v06_mechanism_preregistration.md"],
  ["internal/anniversary/kfacet_v06_writeup.md", "docs/isotrophy/kfacet/kfacet_v06_writeup.md"],
  ["internal/anniversary/kfacet_v06a_energy_quartile_audit_form.md", "docs/isotrophy/kfacet/kfacet_v06a_energy_quartile_audit_form.md"],
  ["internal/anniversary/kfacet_v06b_within_branch_energy_audit_form.md", "docs/isotrophy/kfacet/kfacet_v06b_within_branch_energy_audit_form.md"],
  ["internal/anniversary/kfacet_v07_mechanism_preregistration.md", "docs/isotrophy/kfacet/kfacet_v07_mechanism_preregistration.md"],
  ["internal/anniversary/kfacet_v07_writeup.md", "docs/isotrophy/kfacet/kfacet_v07_writeup.md"],
  ["internal/anniversary/kfacet_v07a_prime_restricted_scope_form.md", "docs/isotrophy/kfacet/kfacet_v07a_prime_restricted_scope_form.md"],
  ["internal/anniversary/kfacet_v07a_velocity_fraction_audit_form.md", "docs/isotrophy/kfacet/kfacet_v07a_velocity_fraction_audit_form.md"],
  ["internal/anniversary/kfacet_v08_mechanism_preregistration.md", "docs/isotrophy/kfacet/kfacet_v08_mechanism_preregistration.md"],
  ["internal/anniversary/kfacet_v08_writeup.md", "docs/isotrophy/kfacet/kfacet_v08_writeup.md"],
  ["internal/anniversary/kfacet_v08a_purity_quartile_audit_form.md", "docs/isotrophy/kfacet/kfacet_v08a_purity_quartile_audit_form.md"],
  ["internal/anniversary/kfacet_v09_mechanism_preregistration.md", "docs/isotrophy/kfacet/kfacet_v09_mechanism_preregistration.md"],
  ["internal/anniversary/kfacet_v09_writeup.md", "docs/isotrophy/kfacet/kfacet_v09_writeup.md"],
  ["internal/anniversary/kfacet_v09a_signed_vf_three_zone_form.md", "docs/isotrophy/kfacet/kfacet_v09a_signed_vf_three_zone_form.md"],

  // Historical isotrophy appendix notes. These are public because docs/** ships.
  ["internal/anniversary/isotrophy_handoff_note.md", "docs/isotrophy/archive/isotrophy_handoff_note.md"],
  ["internal/anniversary/isotrophy_handoff_note2.md", "docs/isotrophy/archive/isotrophy_handoff_note2.md"],
  ["internal/anniversary/supercalifragilisticexpialidocious.md", "docs/isotrophy/archive/supercalifragilisticexpialidocious.md"],

  // Public brand, site-operations, and promo docs moved out of the root docs shelf.
  ["docs/BRAND_POSITIONING.md", "docs/brand/BRAND_POSITIONING.md", "public-stub"],
  ["docs/BRAND_ROADMAP.md", "docs/brand/BRAND_ROADMAP.md", "public-stub"],
  ["docs/LEGAL_STANDING.md", "docs/brand/LEGAL_STANDING.md", "public-stub"],
  ["docs/Mythos-Benchmark.md", "docs/brand/Mythos-Benchmark.md", "public-stub"],
  ["docs/gemini-benchmark.md", "docs/brand/gemini-benchmark.md", "public-stub"],

  ["docs/WEBSITE_DEVELOPMENT.md", "docs/site/WEBSITE_DEVELOPMENT.md", "public-stub"],
  ["docs/SEO_AND_SOCIAL_READINESS_ROADMAP.md", "docs/site/SEO_AND_SOCIAL_READINESS_ROADMAP.md", "public-stub"],
  ["docs/ICON_ASSETS.md", "docs/site/ICON_ASSETS.md", "public-stub"],
  ["docs/LOGO_ANIMATION_TOOLKIT.md", "docs/site/LOGO_ANIMATION_TOOLKIT.md", "public-stub"],
  ["docs/PHOTO_DATA_POLICY.md", "docs/site/PHOTO_DATA_POLICY.md", "public-stub"],
  ["docs/THIRD_PARTY_REUSE.md", "docs/site/THIRD_PARTY_REUSE.md", "public-stub"],
  ["docs/UI_UX_THEME_FOUNDATION.md", "docs/site/UI_UX_THEME_FOUNDATION.md", "public-stub"],
  ["docs/HIGHLIGHTS_RAIL_ROADMAP.md", "docs/site/HIGHLIGHTS_RAIL_ROADMAP.md", "public-stub"],

  ["docs/PROMO_HIGHLIGHTS.md", "docs/promo/PROMO_HIGHLIGHTS.md", "public-stub"],
  ["docs/SUNDOG_OUTREACH_PACKET.md", "docs/promo/SUNDOG_OUTREACH_PACKET.md", "public-stub"],
];

const replacements = new Map();
for (const [oldPath, newPath] of moves) {
  replacements.set(oldPath, newPath);
  replacements.set(oldPath.replaceAll("/", "\\"), newPath.replaceAll("/", "\\"));

  if (oldPath.startsWith("docs/")) {
    replacements.set(`/docs/${oldPath.slice("docs/".length)}`, `/docs/${newPath.slice("docs/".length)}`);
  }

  if (oldPath.startsWith("internal/anniversary/")) {
    const oldName = oldPath.slice("internal/anniversary/".length);
    const newDocsPath = newPath;
    replacements.set(`../${oldPath}`, newDocsPath.startsWith("docs/") ? newDocsPath.slice("docs/".length) : newDocsPath);
    replacements.set(`../../${oldPath}`, newDocsPath.startsWith("docs/") ? `../${newDocsPath.slice("docs/".length)}` : newDocsPath);
    replacements.set(`https://github.com/humiliati/sundog/blob/main/${oldPath}`, newDocsPath);
    replacements.set(oldName, oldName);
  }
}

// More readable local links for files now living inside docs/isotrophy.
for (const [oldPath, newPath] of moves.filter(([oldPath]) => oldPath.startsWith("internal/anniversary/"))) {
  replacements.set(`../${oldPath}`, newPath.replace("docs/isotrophy/", ""));
}

const textExtensions = new Set([
  ".css",
  ".html",
  ".js",
  ".json",
  ".jsonl",
  ".md",
  ".mjs",
  ".py",
  ".svg",
  ".toml",
  ".txt",
  ".xml",
]);

const excludedDirs = new Set([
  ".git",
  ".wrangler",
  "__pycache__",
  "AppData",
  "dist",
  "dist-poses",
  "node_modules",
  "results",
  "tmp",
]);

function extname(path) {
  const name = path.split(/[\\/]/).pop() ?? "";
  const idx = name.lastIndexOf(".");
  return idx === -1 ? "" : name.slice(idx);
}

async function pathExists(path) {
  try {
    await stat(join(root, path));
    return true;
  } catch {
    return false;
  }
}

async function isExpectedPublicStub(oldPath, newPath) {
  try {
    const text = await readFile(join(root, oldPath), "utf8");
    return text === publicStubNotice(oldPath, newPath);
  } catch {
    return false;
  }
}

async function walk(dir = root) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(dir, entry.name);
    const rel = relative(root, path);
    if (entry.isDirectory()) {
      if (!excludedDirs.has(entry.name)) {
        files.push(...await walk(path));
      }
    } else if (entry.isFile() && textExtensions.has(extname(entry.name))) {
      files.push(rel);
    }
  }
  return files;
}

async function moveFiles() {
  for (const [oldPath, newPath, mode] of moves) {
    const oldExists = await pathExists(oldPath);
    const newExists = await pathExists(newPath);

    if (!oldExists && newExists) {
      console.log(`already moved: ${oldPath} -> ${newPath}`);
      continue;
    }
    if (!oldExists) {
      console.log(`missing source: ${oldPath}`);
      continue;
    }
    if (newExists) {
      if (mode === "public-stub" && await isExpectedPublicStub(oldPath, newPath)) {
        console.log(`already moved with public stub: ${oldPath} -> ${newPath}`);
        continue;
      }
      throw new Error(`Refusing to overwrite existing destination: ${newPath}`);
    }

    console.log(`move: ${oldPath} -> ${newPath}`);
    if (apply) {
      await mkdir(join(root, dirname(newPath)), { recursive: true });
      await rename(join(root, oldPath), join(root, newPath));
      if (mode === "public-stub") {
        pendingPublicStubs.push([oldPath, newPath]);
      }
    }
  }
}

async function replaceReferences() {
  const files = await walk();
  let changed = 0;
  for (const rel of files) {
    const normalizedRel = rel.split(sep).join("/");
    if (
      normalizedRel === "scripts/organize-docs.mjs" ||
      normalizedRel === "internal/CONTENT_ORGANIZATION_MANIFEST.md"
    ) {
      continue;
    }
    const publicStubMove = moves.find(([oldPath, , mode]) => (
      mode === "public-stub" && oldPath === normalizedRel
    ));
    if (publicStubMove && await isExpectedPublicStub(publicStubMove[0], publicStubMove[1])) {
      continue;
    }
    if (pendingPublicStubs.some(([oldPath]) => oldPath === normalizedRel)) {
      continue;
    }
    if (moves.some(([, newPath]) => rel.split(sep).join("/") === newPath) && !apply) {
      // During dry-run these files are still at their old paths.
    }
    const abs = join(root, rel);
    let text = await readFile(abs, "utf8");
    let next = text;
    for (const [from, to] of replacements) {
      if (from === to) continue;
      next = next.split(from).join(to);
    }
    if (next !== text) {
      changed += 1;
      console.log(`update refs: ${rel}`);
      if (apply) {
        await writeFile(abs, next, "utf8");
      }
    }
  }
  console.log(`reference files ${apply ? "updated" : "to update"}: ${changed}`);
}

console.log(apply ? "Applying document organization..." : "Dry-run document organization...");
await moveFiles();
await replaceReferences();
for (const [oldPath, newPath] of pendingPublicStubs) {
  await writeFile(join(root, oldPath), publicStubNotice(oldPath, newPath), "utf8");
}
console.log(apply ? "Done." : "Dry-run only. Re-run with --apply to make changes.");
