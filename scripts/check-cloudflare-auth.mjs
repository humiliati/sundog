import { cloudflareHeaders, cloudflareRequest } from "./cloudflare-auth.mjs";

const auth = await cloudflareHeaders();

if (!auth) {
  console.error(
    "No usable Cloudflare auth found. Set CLOUDFLARE_API_TOKEN, or keep the legacy global key and account email in the local credential file.",
  );
  process.exit(1);
}

const verificationPath = auth.mode === "api-token" ? "/user/tokens/verify" : "/user";
const { response, body } = await cloudflareRequest(verificationPath);

if (!response.ok || body.success === false) {
  console.error(`Cloudflare auth failed in ${auth.mode} mode.`);
  if (Array.isArray(body.errors)) {
    for (const error of body.errors) {
      console.error(`- ${error.message ?? JSON.stringify(error)}`);
    }
  }
  process.exit(1);
}

console.log(`Cloudflare auth verified with ${auth.mode}.`);

if (auth.mode === "api-token") {
  const permissionGroups = await cloudflareRequest("/user/tokens/permission_groups");
  if (permissionGroups.response.ok && permissionGroups.body.success !== false) {
    console.log("Token can read API token permission groups.");
  } else {
    console.log("Token cannot read API token permission groups.");
  }
}
