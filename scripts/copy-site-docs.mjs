import { cp, mkdir, readdir, rm } from "node:fs/promises";
import { join } from "node:path";

const root = process.cwd();
const dist = join(root, "dist");
const sourceDocs = join(root, "docs");
const targetDocs = join(dist, "docs");

async function copyPublicDocs(sourceDir, targetDir) {
  const entries = await readdir(sourceDir, { withFileTypes: true });

  await mkdir(targetDir, { recursive: true });

  for (const entry of entries) {
    const source = join(sourceDir, entry.name);
    const target = join(targetDir, entry.name);

    if (entry.isDirectory()) {
      await copyPublicDocs(source, target);
    } else if (entry.isFile() && (entry.name.endsWith(".md") || entry.name === "Public-notes.tex")) {
      await cp(source, target);
    }
  }
}

await mkdir(dist, { recursive: true });
await cp(join(root, "README.md"), join(dist, "README.md"));
await rm(targetDocs, { recursive: true, force: true });
await copyPublicDocs(sourceDocs, targetDocs);
