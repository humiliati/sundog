import { appendFile, readFile, writeFile } from "node:fs/promises";
import { cloudflareRequest, localCredentialPath } from "./cloudflare-auth.mjs";

const projectName = "sundog";
const zoneName = "sundog.cc";
const repoOwner = "humiliati";
const repoName = "sundog";
const productionBranch = "main";

function ok(result) {
  return result.response.ok && result.body.success !== false;
}

function firstResult(result) {
  return Array.isArray(result.body.result) ? result.body.result[0] : undefined;
}

function cloudflareErrors(result) {
  return Array.isArray(result.body.errors)
    ? result.body.errors.map((error) => error.message ?? JSON.stringify(error)).join("; ")
    : `HTTP ${result.response.status}`;
}

async function githubRepoMetadata() {
  const response = await fetch(`https://api.github.com/repos/${repoOwner}/${repoName}`, {
    headers: {
      "Accept": "application/vnd.github+json",
      "User-Agent": "sundog-cloudflare-pages-bootstrap",
    },
  });

  if (!response.ok) {
    return {};
  }

  const body = await response.json();
  return {
    repo_id: String(body.id),
    owner_id: String(body.owner?.id ?? ""),
  };
}

async function ensureLocalKeyFile(entries) {
  let existing = "";
  try {
    existing = await readFile(localCredentialPath, "utf8");
  } catch {
    await writeFile(localCredentialPath, [
      "# Local Sundog Cloudflare material.",
      "# Keep this file outside the repo. Do not commit or print token values.",
      "",
    ].join("\n"));
  }

  const nextLines = [];
  for (const [key, value] of Object.entries(entries)) {
    if (!value || existing.includes(`${key}=`)) {
      continue;
    }
    nextLines.push(`${key}=${value}`);
  }

  if (nextLines.length > 0) {
    await appendFile(localCredentialPath, `${nextLines.join("\n")}\n`);
  }
}

const accounts = await cloudflareRequest("/accounts");
if (!ok(accounts) || accounts.body.result.length === 0) {
  console.error(`Unable to find a Cloudflare account: ${cloudflareErrors(accounts)}`);
  process.exit(1);
}

const account = accounts.body.result[0];
const zones = await cloudflareRequest(`/zones?name=${encodeURIComponent(zoneName)}`);
const zone = ok(zones) ? firstResult(zones) : undefined;

const existingProject = await cloudflareRequest(`/accounts/${account.id}/pages/projects/${projectName}`);
let project;
let created = false;

if (ok(existingProject)) {
  project = existingProject.body.result;
} else if (existingProject.response.status === 404) {
  const repo = await githubRepoMetadata();
  const createProject = await cloudflareRequest(`/accounts/${account.id}/pages/projects`, {
    method: "POST",
    body: JSON.stringify({
      name: projectName,
      production_branch: productionBranch,
      build_config: {
        build_command: "npm run build",
        destination_dir: "dist",
        root_dir: "/",
        build_caching: true,
      },
      source: {
        type: "github",
        config: {
          owner: repoOwner,
          owner_id: repo.owner_id,
          repo_name: repoName,
          repo_id: repo.repo_id,
          production_branch: productionBranch,
          production_deployments_enabled: true,
          preview_deployment_setting: "all",
          pr_comments_enabled: true,
          deployments_enabled: true,
        },
      },
    }),
  });

  if (!ok(createProject)) {
    console.error(`Unable to create Pages project from GitHub repo: ${cloudflareErrors(createProject)}`);
    console.error("The likely fix is to connect/install the Cloudflare GitHub app for humiliati/sundog in the Cloudflare dashboard, then rerun this command.");
    process.exit(1);
  }

  project = createProject.body.result;
  created = true;
} else {
  console.error(`Unable to inspect Pages project: ${cloudflareErrors(existingProject)}`);
  process.exit(1);
}

await ensureLocalKeyFile({
  CLOUDFLARE_ACCOUNT_ID: account.id,
  CLOUDFLARE_ZONE_ID_SUNDOG_CC: zone?.id,
  CLOUDFLARE_PAGES_PROJECT_NAME: project.name,
  CLOUDFLARE_PAGES_PROJECT_ID: project.id,
  CLOUDFLARE_PAGES_SUBDOMAIN: project.subdomain,
  CLOUDFLARE_PAGES_PRODUCTION_BRANCH: project.production_branch,
  SUNDOG_GITHUB_REPO: `${repoOwner}/${repoName}`,
});

console.log(`${created ? "Created" : "Found"} Cloudflare Pages project: ${project.name}`);
console.log(`Production branch: ${project.production_branch}`);
console.log(`Build command: ${project.build_config?.build_command ?? "(unset)"}`);
console.log(`Output directory: ${project.build_config?.destination_dir ?? "(unset)"}`);
console.log(`Source: ${project.source?.type ?? "(none)"}`);
console.log(`Configured credential file updated: ${localCredentialPath}`);
