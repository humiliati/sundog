import { readdirSync } from "node:fs";
import { basename, extname, join, resolve } from "node:path";
import { defineConfig } from "vite";

const root = process.cwd();
const htmlEntries = Object.fromEntries(
  readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isFile() && extname(entry.name) === ".html")
    .map((entry) => [basename(entry.name, ".html"), resolve(root, entry.name)]),
);

export default defineConfig({
  build: {
    rollupOptions: {
      input: htmlEntries,
    },
  },
});
