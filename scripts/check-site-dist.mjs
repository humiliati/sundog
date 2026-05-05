import { access, readFile } from "node:fs/promises";
import { dirname, join, normalize } from "node:path";

const dist = join(process.cwd(), "dist");
const indexPath = join(dist, "index.html");
const html = await readFile(indexPath, "utf8");
const hrefs = [...html.matchAll(/\bhref=(["'])(.*?)\1/g)].map((match) => match[2]);
const missing = [];

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

  const target = normalize(join(dirname(indexPath), cleanHref));
  if (!target.startsWith(dist)) {
    missing.push(href);
    continue;
  }

  try {
    await access(target);
  } catch {
    missing.push(href);
  }
}

if (missing.length > 0) {
  console.error("Missing local files referenced by dist/index.html:");
  for (const href of missing) {
    console.error(`- ${href}`);
  }
  process.exit(1);
}

console.log("dist link check passed");
