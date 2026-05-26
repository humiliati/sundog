import { appendFile, readFile } from "node:fs/promises";
import { cloudflareRequest, localCredentialPath } from "./cloudflare-auth.mjs";

const zoneName = "sundog.cc";

function ok(result) {
  return result.response.ok && result.body.success !== false;
}

function errors(result) {
  return Array.isArray(result.body.errors)
    ? result.body.errors.map((error) => error.message ?? JSON.stringify(error)).join("; ")
    : `HTTP ${result.response.status}`;
}

function requirePermission(groups, name, scope) {
  const group = groups.find((candidate) => (
    candidate.name === name &&
    (!scope || candidate.scopes?.includes(scope))
  ));

  if (!group) {
    throw new Error(`Missing Cloudflare permission group: ${name}`);
  }

  return { id: group.id };
}

async function appendSecretBlock(lines) {
  const existing = await readFile(localCredentialPath, "utf8").catch(() => "");
  const next = lines.filter(([key]) => !existing.includes(`${key}=`));

  if (next.length === 0) {
    return;
  }

  await appendFile(localCredentialPath, [
    "",
    "# Scoped Cloudflare API tokens. Token values are shown once by Cloudflare.",
    ...next.map(([key, value]) => `${key}=${value}`),
    "",
  ].join("\n"));
}

function readLocalEnv(text) {
  const env = {};
  for (const line of text.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)=(.+?)\s*$/);
    if (match) {
      env[match[1]] = match[2].trim();
    }
  }
  return env;
}

async function createToken(spec) {
  const result = await cloudflareRequest("/user/tokens", {
    method: "POST",
    body: JSON.stringify({
      name: spec.name,
      policies: spec.policies,
    }),
  });

  if (!ok(result)) {
    throw new Error(errors(result));
  }

  const token = result.body.result;
  const value = token.value ?? token.token ?? token.secret;
  if (!value) {
    throw new Error("Cloudflare created a token but did not return a token value.");
  }

  return { id: token.id, value };
}

async function updateToken(spec, tokenId) {
  const result = await cloudflareRequest(`/user/tokens/${tokenId}`, {
    method: "PUT",
    body: JSON.stringify({
      name: spec.name,
      policies: spec.policies,
    }),
  });

  if (!ok(result)) {
    throw new Error(errors(result));
  }
}

const accounts = await cloudflareRequest("/accounts");
const zones = await cloudflareRequest(`/zones?name=${encodeURIComponent(zoneName)}`);
const permissionGroups = await cloudflareRequest("/user/tokens/permission_groups");

if (!ok(accounts) || accounts.body.result.length === 0) {
  console.error(`Unable to find Cloudflare account: ${errors(accounts)}`);
  process.exit(1);
}

if (!ok(zones) || zones.body.result.length === 0) {
  console.error(`Unable to find ${zoneName} zone: ${errors(zones)}`);
  process.exit(1);
}

if (!ok(permissionGroups)) {
  console.error(`Unable to read token permission groups: ${errors(permissionGroups)}`);
  process.exit(1);
}

const account = accounts.body.result[0];
const zone = zones.body.result[0];
const groups = permissionGroups.body.result;
const accountResource = `com.cloudflare.api.account.${account.id}`;
const zoneResource = `com.cloudflare.api.account.zone.${zone.id}`;
const userResource = `com.cloudflare.api.user.${(await cloudflareRequest("/user")).body.result.id}`;

const accountPolicy = (permissionNames) => ({
  effect: "allow",
  resources: { [accountResource]: "*" },
  permission_groups: permissionNames.map((name) => requirePermission(groups, name, "com.cloudflare.api.account")),
});

const zonePolicy = (permissionNames) => ({
  effect: "allow",
  resources: { [zoneResource]: "*" },
  permission_groups: permissionNames.map((name) => requirePermission(groups, name, "com.cloudflare.api.account.zone")),
});

const userPolicy = (permissionNames) => ({
  effect: "allow",
  resources: { [userResource]: "*" },
  permission_groups: permissionNames.map((name) => requirePermission(groups, name, "com.cloudflare.api.user")),
});

const specs = [
  {
    name: "sundog-pages-deploy",
    env: "CLOUDFLARE_TOKEN_SUNDOG_PAGES_DEPLOY",
    idEnv: "CLOUDFLARE_TOKEN_ID_SUNDOG_PAGES_DEPLOY",
    policies: [
      accountPolicy(["Pages Write", "Account Settings Read"]),
      userPolicy(["Memberships Read", "User Details Read"]),
    ],
  },
  {
    name: "sundog-workers-edit",
    env: "CLOUDFLARE_TOKEN_SUNDOG_WORKERS_EDIT",
    idEnv: "CLOUDFLARE_TOKEN_ID_SUNDOG_WORKERS_EDIT",
    policies: [
      accountPolicy(["Workers Scripts Write", "Account Settings Read"]),
      zonePolicy(["Workers Routes Write", "Zone Read"]),
    ],
  },
  {
    name: "sundog-dns-edit",
    env: "CLOUDFLARE_TOKEN_SUNDOG_DNS_EDIT",
    idEnv: "CLOUDFLARE_TOKEN_ID_SUNDOG_DNS_EDIT",
    policies: [
      zonePolicy(["DNS Write", "Zone Read"]),
    ],
  },
  {
    name: "sundog-session-readonly",
    env: "CLOUDFLARE_TOKEN_SUNDOG_SESSION_READONLY",
    idEnv: "CLOUDFLARE_TOKEN_ID_SUNDOG_SESSION_READONLY",
    policies: [
      accountPolicy(["Pages Read", "Account Settings Read", "Workers Scripts Read"]),
      zonePolicy(["DNS Read", "Zone Read", "Workers Routes Read"]),
      userPolicy(["Memberships Read", "User Details Read"]),
    ],
  },
];

const existingSecrets = await readFile(localCredentialPath, "utf8").catch(() => "");
const localEnv = readLocalEnv(existingSecrets);
const created = [];
const updated = [];

for (const spec of specs) {
  if (existingSecrets.includes(`${spec.env}=`)) {
    if (localEnv[spec.idEnv]) {
      await updateToken(spec, localEnv[spec.idEnv]);
      updated.push(spec.name);
      console.log(`Updated scoped token policy: ${spec.name}`);
    } else {
      console.log(`Skipping ${spec.name}; token value already recorded locally but token ID is missing.`);
    }
    continue;
  }

  try {
    const token = await createToken(spec);
    await appendSecretBlock([
      [spec.idEnv, token.id],
      [spec.env, token.value],
    ]);
    created.push(spec.name);
    console.log(`Created scoped token: ${spec.name}`);
  } catch (error) {
    console.error(`Unable to create ${spec.name}: ${error.message}`);
    console.error("If this says API token auth is required, create a short-lived Cloudflare token from the Create additional tokens template and rerun with CLOUDFLARE_API_TOKEN set.");
    process.exit(1);
  }
}

console.log(`Scoped token bootstrap complete. Created ${created.length} token(s), updated ${updated.length} token policy set(s). Values were written only to the configured credential file.`);
