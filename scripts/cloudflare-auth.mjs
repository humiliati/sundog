import { readFile } from "node:fs/promises";

export const globalKeyPath = "C:\\Users\\hughe\\yek eralfduolc.txt";
export const apiBase = "https://api.cloudflare.com/client/v4";

export async function readLegacyCredentialFile() {
  try {
    const text = await readFile(globalKeyPath, "utf8");
    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    const email = lines.find((line) => line.includes("@"));
    const key = lines
      .filter((line) => !line.includes("@") && !line.includes("="))
      .flatMap((line) => [...line.matchAll(/[A-Za-z0-9_]{30,}/g)].map((match) => match[0]))
      .find((line) => line.length >= 30);

    return { email, key };
  } catch {
    return {};
  }
}

export function tokenHeaders() {
  const token = process.env.CLOUDFLARE_API_TOKEN?.trim();
  if (!token) {
    return null;
  }

  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

export async function globalKeyHeaders() {
  const fileCredentials = await readLegacyCredentialFile();
  const email = process.env.CLOUDFLARE_EMAIL?.trim() || fileCredentials.email;
  const key = process.env.CLOUDFLARE_API_KEY?.trim() || fileCredentials.key;

  if (!email || !key) {
    return null;
  }

  return {
    "X-Auth-Email": email,
    "X-Auth-Key": key,
    "Content-Type": "application/json",
  };
}

export async function cloudflareHeaders() {
  const headers = tokenHeaders();
  if (headers) {
    return { mode: "api-token", headers };
  }

  const legacyHeaders = await globalKeyHeaders();
  if (legacyHeaders) {
    return { mode: "legacy-global-key", headers: legacyHeaders };
  }

  return null;
}

export async function cloudflareRequest(path, options = {}) {
  const auth = await cloudflareHeaders();
  if (!auth) {
    throw new Error(
      "No usable Cloudflare auth found. Set CLOUDFLARE_API_TOKEN, or keep the legacy global key and account email in the local credential file.",
    );
  }

  const response = await fetch(`${apiBase}${path}`, {
    ...options,
    headers: {
      ...auth.headers,
      ...options.headers,
    },
  });
  const body = await response.json().catch(() => ({}));

  return { ...auth, response, body };
}
