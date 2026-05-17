import { readFileSync, readdirSync } from "node:fs";
import { basename, extname, resolve } from "node:path";
import { defineConfig } from "vite";

const root = process.cwd();
const pageManifest = JSON.parse(readFileSync(resolve(root, "site-pages.json"), "utf8"));
const manifestEntries = pageManifest.pages ?? [];
const manifestEntryNames = new Set(manifestEntries.map((page) => page.entry));
const rootHtmlFiles = readdirSync(root, { withFileTypes: true })
  .filter((entry) => entry.isFile() && extname(entry.name) === ".html")
  .map((entry) => entry.name)
  .sort();
const unfiledRootHtml = rootHtmlFiles.filter((name) => !manifestEntryNames.has(name));
const missingManifestFiles = manifestEntries
  .map((page) => page.entry)
  .filter((name) => !rootHtmlFiles.includes(name));
const duplicateManifestEntries = manifestEntries
  .map((page) => page.entry)
  .filter((entry, index, entries) => entries.indexOf(entry) !== index);
const missingIntent = manifestEntries
  .filter((page) => typeof page.publicLaunchIntent !== "string" || page.publicLaunchIntent.trim() === "")
  .map((page) => page.entry);

if (unfiledRootHtml.length > 0) {
  throw new Error(
    `Root HTML page(s) lack site-pages.json publicLaunchIntent: ${unfiledRootHtml.join(", ")}`,
  );
}

if (missingManifestFiles.length > 0) {
  throw new Error(
    `site-pages.json references missing root HTML page(s): ${missingManifestFiles.join(", ")}`,
  );
}

if (duplicateManifestEntries.length > 0) {
  throw new Error(
    `site-pages.json contains duplicate root HTML page(s): ${[...new Set(duplicateManifestEntries)].join(", ")}`,
  );
}

if (missingIntent.length > 0) {
  throw new Error(
    `site-pages.json page(s) need publicLaunchIntent: ${missingIntent.join(", ")}`,
  );
}

const htmlEntries = Object.fromEntries(
  manifestEntries.map((page) => [basename(page.entry, ".html"), resolve(root, page.entry)]),
);

export default defineConfig({
  build: {
    rollupOptions: {
      input: htmlEntries,
    },
  },
});
