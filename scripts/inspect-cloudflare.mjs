import { cloudflareRequest } from "./cloudflare-auth.mjs";

function ok(result) {
  return result.response.ok && result.body.success !== false;
}

function names(items) {
  return items.map((item) => item.name).filter(Boolean);
}

const user = await cloudflareRequest("/user");
if (!ok(user)) {
  console.error("Unable to read Cloudflare user.");
  process.exit(1);
}

const accounts = await cloudflareRequest("/accounts");
const zones = await cloudflareRequest("/zones?name=sundog.cc");
const permissionGroups = await cloudflareRequest("/user/tokens/permission_groups");

console.log(`Auth mode: ${user.mode}`);
console.log(`Accounts visible: ${ok(accounts) ? names(accounts.body.result).join(", ") || "(none)" : "unavailable"}`);
console.log(`sundog.cc zones visible: ${ok(zones) ? zones.body.result.length : "unavailable"}`);
console.log(`Can read token permission groups: ${ok(permissionGroups) ? "yes" : "no"}`);

if (ok(accounts)) {
  for (const account of accounts.body.result) {
    const pages = await cloudflareRequest(`/accounts/${account.id}/pages/projects`);
    console.log(`Pages projects in ${account.name}: ${ok(pages) ? names(pages.body.result).join(", ") || "(none)" : "unavailable"}`);
  }
}
