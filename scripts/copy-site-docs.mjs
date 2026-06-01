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

async function copyPublicDocs(sourceDir, targetDir) {
  const entries = await readdir(sourceDir, { withFileTypes: true });

  await mkdir(targetDir, { recursive: true });

  for (const entry of entries) {
    const source = join(sourceDir, entry.name);
    const target = join(targetDir, entry.name);

    if (entry.isDirectory()) {
      await copyPublicDocs(source, target);
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
await copyPublicDocs(sourceDocs, targetDocs);
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
