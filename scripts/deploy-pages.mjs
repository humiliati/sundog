import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { localCredentialPath } from "./cloudflare-auth.mjs";

const tokenKey = "CLOUDFLARE_TOKEN_SUNDOG_PAGES_DEPLOY";

async function readLocalEnv(path) {
  const text = await readFile(path, "utf8").catch(() => "");
  const env = {};

  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)=(.+?)\s*$/);
    if (match) {
      env[match[1]] = match[2].trim();
    }
  }

  return env;
}

const localEnv = await readLocalEnv(localCredentialPath);
const token = (process.env.CLOUDFLARE_API_TOKEN || localEnv[tokenKey] || "").trim();

if (!token) {
  console.error(`No Pages deploy token found. Set CLOUDFLARE_API_TOKEN or add ${tokenKey} to the configured credential file.`);
  process.exit(1);
}

const wrangler = "node_modules/wrangler/bin/wrangler.js";

const child = spawn("node", [wrangler, "pages", "deploy", "dist", "--project-name", "sundog", "--commit-dirty=true"], {
  cwd: process.cwd(),
  env: {
    ...process.env,
    CLOUDFLARE_API_TOKEN: token,
  },
  stdio: "inherit",
  shell: false,
});

child.on("exit", (code) => {
  process.exit(code ?? 1);
});
