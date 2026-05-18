import { existsSync, readFileSync, readdirSync } from "node:fs";
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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function bayesVerdictFallbackPlugin() {
  const dataPath = resolve(root, "public/data/bayes-comparison.json");
  const keys = ["sundogReadout", "comparatorState", "safeInterpretation"];

  return {
    name: "sundog-bayes-verdict-fallback",
    transformIndexHtml(html, context) {
      if (!context.filename.endsWith("alignment.html") || !existsSync(dataPath)) {
        return html;
      }

      const data = JSON.parse(readFileSync(dataPath, "utf8"));
      const verdict = data.verdict ?? {};
      return keys.reduce((updatedHtml, key) => {
        if (typeof verdict[key] !== "string") return updatedHtml;
        const pattern = new RegExp(
          `(<td\\b(?=[^>]*\\bdata-bayes-verdict="${escapeRegExp(key)}")[^>]*>)[\\s\\S]*?(<\\/td>)`,
        );
        return updatedHtml.replace(pattern, `$1${escapeHtml(verdict[key])}$2`);
      }, html);
    },
  };
}

export default defineConfig({
  plugins: [bayesVerdictFallbackPlugin()],
  build: {
    rollupOptions: {
      input: htmlEntries,
    },
  },
});
