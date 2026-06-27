// Durable static file server for Ghost workbench QA.
//
// The Ghost toy workbench (ghost/workbench.html) is plain ESM with no npm
// dependencies, so it does not need the full-site Vite dev server (whose
// dependency optimizer scans the entire multi-page site and is slow/overkill
// for this one internal page). This serves the repo root directly.
//
// Run: `npm run ghost:serve`  ->  http://127.0.0.1:5188/ghost/workbench.html
// Override the port with `npm run ghost:serve -- 5200` or GHOST_PORT=5200.
import http from "node:http";
import { readFile } from "node:fs/promises";
import { join, normalize, extname, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = normalize(join(dirname(fileURLToPath(import.meta.url)), ".."));
const PORT = Number(process.argv[2] || process.env.GHOST_PORT || 5188);
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".webp": "image/webp",
};

http
  .createServer(async (req, res) => {
    try {
      let pathname = decodeURIComponent(new URL(req.url, `http://127.0.0.1:${PORT}`).pathname);
      if (pathname.endsWith("/")) pathname += "index.html";
      const filePath = normalize(join(ROOT, pathname));
      if (!filePath.startsWith(ROOT)) {
        res.writeHead(403, { "Content-Type": "text/plain" }).end("forbidden");
        return;
      }
      const body = await readFile(filePath);
      res.writeHead(200, { "Content-Type": TYPES[extname(filePath)] || "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404, { "Content-Type": "text/plain" }).end("not found");
    }
  })
  .listen(PORT, "127.0.0.1", () =>
    console.log(`ghost:serve  ${ROOT}  ->  http://127.0.0.1:${PORT}/ghost/workbench.html`),
  );
