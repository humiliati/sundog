import { appendFile, readFile } from "node:fs/promises";
import { cloudflareRequest, localCredentialPath } from "./cloudflare-auth.mjs";

const projectName = "sundog";
const domainName = "sundog.cc";

function ok(result) {
  return result.response.ok && result.body.success !== false;
}

function errors(result) {
  return Array.isArray(result.body.errors)
    ? result.body.errors.map((error) => error.message ?? JSON.stringify(error)).join("; ")
    : `HTTP ${result.response.status}`;
}

function summarizeDomain(domain) {
  return {
    name: domain.name,
    status: domain.status,
    validation: domain.validation_data?.status,
    verification: domain.verification_data?.status,
    validation_error: domain.validation_data?.error_message,
    verification_error: domain.verification_data?.error_message,
  };
}

async function appendLocalEntries(entries) {
  const existing = await readFile(localCredentialPath, "utf8").catch(() => "");
  const lines = [];

  for (const [key, value] of Object.entries(entries)) {
    if (value && !existing.includes(`${key}=`)) {
      lines.push(`${key}=${value}`);
    }
  }

  if (lines.length > 0) {
    await appendFile(localCredentialPath, [
      "",
      "# Cloudflare Pages custom domain metadata.",
      ...lines,
      "",
    ].join("\n"));
  }
}

const accounts = await cloudflareRequest("/accounts");
if (!ok(accounts) || accounts.body.result.length === 0) {
  console.error(`Unable to find Cloudflare account: ${errors(accounts)}`);
  process.exit(1);
}

const account = accounts.body.result[0];
const project = await cloudflareRequest(`/accounts/${account.id}/pages/projects/${projectName}`);
if (!ok(project)) {
  console.error(`Unable to read Pages project ${projectName}: ${errors(project)}`);
  process.exit(1);
}

const projectSubdomain = project.body.result.subdomain;
const zones = await cloudflareRequest(`/zones?name=${encodeURIComponent(domainName)}`);
if (!ok(zones) || zones.body.result.length === 0) {
  console.error(`Unable to find zone ${domainName}: ${errors(zones)}`);
  process.exit(1);
}

const zone = zones.body.result[0];
let domain = await cloudflareRequest(`/accounts/${account.id}/pages/projects/${projectName}/domains/${domainName}`);
let created = false;

if (domain.response.status === 404) {
  domain = await cloudflareRequest(`/accounts/${account.id}/pages/projects/${projectName}/domains`, {
    method: "POST",
    body: JSON.stringify({ name: domainName }),
  });
  created = true;
}

if (!ok(domain)) {
  console.error(`Unable to attach Pages domain ${domainName}: ${errors(domain)}`);
  process.exit(1);
}

const dnsRecords = await cloudflareRequest(`/zones/${zone.id}/dns_records?name=${encodeURIComponent(domainName)}&per_page=100`);
if (!ok(dnsRecords)) {
  console.error(`Unable to inspect DNS records for ${domainName}: ${errors(dnsRecords)}`);
  process.exit(1);
}

const records = dnsRecords.body.result;
const pagesRecord = records.find((record) => record.type === "CNAME" && record.content === projectSubdomain);
const conflictingRecords = records.filter((record) => (
  ["A", "AAAA", "CNAME"].includes(record.type) &&
  !(record.type === "CNAME" && record.content === projectSubdomain)
));

let dnsAction = "already-present";
if (!pagesRecord) {
  if (conflictingRecords.length > 0) {
    console.error(`Refusing to overwrite existing apex DNS records for ${domainName}.`);
    for (const record of conflictingRecords) {
      console.error(`- ${record.type} ${record.name} -> ${record.content}`);
    }
    process.exit(1);
  }

  const createDns = await cloudflareRequest(`/zones/${zone.id}/dns_records`, {
    method: "POST",
    body: JSON.stringify({
      type: "CNAME",
      name: domainName,
      content: projectSubdomain,
      proxied: true,
      ttl: 1,
      comment: "Sundog Cloudflare Pages apex domain",
    }),
  });

  if (!ok(createDns)) {
    console.error(`Unable to create Pages CNAME for ${domainName}: ${errors(createDns)}`);
    process.exit(1);
  }
  dnsAction = "created";
}

await appendLocalEntries({
  CLOUDFLARE_PAGES_CUSTOM_DOMAIN: domainName,
  CLOUDFLARE_PAGES_CUSTOM_DOMAIN_STATUS: domain.body.result.status,
});

console.log(`${created ? "Attached" : "Found"} Pages custom domain: ${domainName}`);
console.log(`Project subdomain: ${projectSubdomain}`);
console.log(`DNS CNAME: ${dnsAction}`);
console.log(JSON.stringify(summarizeDomain(domain.body.result), null, 2));
