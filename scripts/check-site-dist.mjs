import { access, readdir, readFile } from "node:fs/promises";
import { dirname, join, normalize } from "node:path";

const dist = join(process.cwd(), "dist");
const missing = [];

async function htmlFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await htmlFiles(path));
    } else if (entry.isFile() && entry.name.endsWith(".html")) {
      files.push(path);
    }
  }

  return files;
}

for (const htmlPath of await htmlFiles(dist)) {
  const html = await readFile(htmlPath, "utf8");
  const hrefs = [...html.matchAll(/\bhref=(["'])(.*?)\1/g)].map((match) => match[2]);

  for (const href of hrefs) {
    if (
      href.startsWith("#") ||
      href.startsWith("http://") ||
      href.startsWith("https://") ||
      href.startsWith("mailto:") ||
      href.startsWith("tel:")
    ) {
      continue;
    }

    const cleanHref = href.split("#")[0].split("?")[0];
    if (!cleanHref) {
      continue;
    }

    const target = normalize(join(dirname(htmlPath), cleanHref));
    if (!target.startsWith(dist)) {
      missing.push(`${htmlPath}: ${href}`);
      continue;
    }

    try {
      await access(target);
    } catch {
      missing.push(`${htmlPath}: ${href}`);
    }
  }
}

if (missing.length > 0) {
  console.error("Missing local files referenced by built HTML:");
  for (const href of missing) {
    console.error(`- ${href}`);
  }
  process.exit(1);
}

console.log("dist link check passed");
