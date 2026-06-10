import { cp, mkdir, readdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const dist = join(root, "dist");
const sourceDocs = join(root, "docs");
const targetDocs = join(dist, "docs");
const sourceChat = join(root, "chat");
const targetChat = join(dist, "chat");
const rootPublicArtifacts = [
  "README.md",
  "LICENSE",
  "COPYRIGHT.md",
  "CITATION.cff"
];
const publicChatArtifacts = [
  "claim_map.json",
  "contents.json",
  "prompts/gold-normal.jsonl",
  "prompts/gold-boundary.jsonl",
  "prompts/gold-adversarial.jsonl",
  "prompts/gold-wild.jsonl",
  "prompts/gold-differential.jsonl"
];

// Docs withheld from the public site (dist/docs/). A path that names a
// directory withholds the whole subtree. IMPORTANT: this guard operates on the
// working tree (readdir), so it withholds files even when they are gitignored
// but still present on disk — which is exactly the case for the confidential
// material below after the 2026-06-03 leak remediation. Keep this set in sync
// with the /docs/* entries in .gitignore.
//   501c3 = confidential patent / counsel material + 501(c)(3) governance
//           drafts. Public accessibility can itself count as a patent
//           disclosure, so this MUST never reach dist/.
const DOCS_NO_PUBLISH = new Set([
  "501c3",
  // White-box "representational" instrument R&D (kill-gated; each carries an
  // explicit "No public surface" stamp). These describe the same
  // determining-shadow-set / body-shadow method that is the subject of the 501c3
  // invention disclosure, so a public (even unlinked) copy bears on patent
  // enabling-disclosure timing. Held until BOTH the research freeze lifts
  // (LATTICE + threebody v0.19) AND counsel clears. A bare filename withholds that
  // file; "chatv2" (no slash) withholds the whole docs/chatv2/ subtree.
  "chatv2",
  "deconfound",
  "SUNDOG_V_DECONFOUND.md",
  "DECONFOUND_REAL_DATA_MEMO.md",
  "SUNDOG_V_JEPA.md",
  "JEPA_LIT_PASS_MEMO.md",
  "SUNDOG_V_ALLELOPATHY.md",
  // CROSS_SUBSTRATE_NOTES.md was removed from this set 2026-06-09 (owner
  // decision): 14 Ask Sundog routes cite it as visible trace support, the
  // file is git-tracked in the public repo (already disclosed), and
  // withholding it only made the cited evidence unreachable on the site.
]);

async function copyPublicDocs(sourceDir, targetDir, noPublish = new Set(), relBase = "") {
  const entries = await readdir(sourceDir, { withFileTypes: true });

  await mkdir(targetDir, { recursive: true });

  for (const entry of entries) {
    const rel = relBase ? `${relBase}/${entry.name}` : entry.name;
    if (noPublish.has(rel)) {
      console.log(`[copy-site-docs] withheld (no-publish): docs/${rel}`);
      continue;
    }
    const source = join(sourceDir, entry.name);
    const target = join(targetDir, entry.name);

    if (entry.isDirectory()) {
      await copyPublicDocs(source, target, noPublish, rel);
    } else if (entry.isFile()) {
      await cp(source, target);
    }
  }
}

await mkdir(dist, { recursive: true });
for (const artifact of rootPublicArtifacts) {
  await cp(join(root, artifact), join(dist, artifact));
}
await rm(targetDocs, { recursive: true, force: true });
await copyPublicDocs(sourceDocs, targetDocs, DOCS_NO_PUBLISH);
await rm(targetChat, { recursive: true, force: true });
await mkdir(targetChat, { recursive: true });

for (const artifact of publicChatArtifacts) {
  const target = join(targetChat, artifact);
  await mkdir(dirname(target), { recursive: true });
  await cp(join(sourceChat, artifact), target);
}

// Unlinked Kakeya reviewer surface (review-only; not in nav or sitemap). Deploy
// the interactive workbench/gallery + their modules so the /kakeya landing's
// links resolve in dist. No public inbound path until external sign-off.
const sourceKakeya = join(root, "kakeya");
const targetKakeya = join(dist, "kakeya");
await rm(targetKakeya, { recursive: true, force: true });
await copyPublicDocs(sourceKakeya, targetKakeya);
