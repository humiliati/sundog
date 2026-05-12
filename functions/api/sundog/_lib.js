/**
 * Shared helpers for /api/sundog/* Pages Functions.
 *
 * No external deps — uses the Web Crypto + R2 APIs available in the Workers
 * runtime. Keep this file pure (no side effects on import) so each endpoint
 * can import only what it needs.
 */

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };

// ---------- CORS / response helpers ----------------------------------------

export function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { ...JSON_HEADERS, ...(init.headers ?? {}) },
  });
}

export function errorResponse(status, code, message) {
  return jsonResponse({ error: { code, message } }, { status });
}

export function withCors(response, origin = "*") {
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", origin);
  headers.set("access-control-allow-methods", "GET, POST, OPTIONS");
  headers.set("access-control-allow-headers", "content-type");
  headers.set("access-control-max-age", "600");
  return new Response(response.body, { status: response.status, headers });
}

export function handleOptions(request) {
  return withCors(new Response(null, { status: 204 }), request.headers.get("origin") || "*");
}

// ---------- hashing + ID generation ---------------------------------------

const ENCODER = new TextEncoder();

export async function sha256Hex(input) {
  const bytes = typeof input === "string" ? ENCODER.encode(input) : input;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Cryptographically random 32-byte token, hex-encoded. The user receives
 * this exactly once at submission time; only the SHA-256 hash is persisted.
 */
export function makeDeletionToken() {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * ULID-style sortable identifier — 10 chars of base32 timestamp + 16 chars
 * of base32 randomness. Sorts lexicographically by submission time.
 */
const ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
function toBase32(int, len) {
  let s = "";
  let n = BigInt(int);
  const base = 32n;
  for (let i = 0; i < len; i += 1) {
    s = ULID_CHARS[Number(n % base)] + s;
    n = n / base;
  }
  return s;
}
export function makeSubmissionId() {
  const timeMs = BigInt(Date.now());
  const timePart = toBase32(timeMs, 10);
  const rand = new Uint8Array(10);
  crypto.getRandomValues(rand);
  // Pack 10 random bytes into 16 base32 chars.
  let randPart = "";
  for (let i = 0; i < 16; i += 1) {
    randPart += ULID_CHARS[rand[i % 10] % 32];
  }
  return timePart + randPart;
}

// ---------- request parsing -----------------------------------------------

export async function readJsonBody(request, maxBytes = 16 * 1024 * 1024) {
  const ct = request.headers.get("content-type") || "";
  if (!ct.includes("application/json")) {
    throw httpError(415, "unsupported_media_type", "Expected application/json body.");
  }
  const cl = Number(request.headers.get("content-length") || "0");
  if (cl > maxBytes) {
    throw httpError(413, "payload_too_large", `Body exceeds ${maxBytes} bytes.`);
  }
  try {
    return await request.json();
  } catch {
    throw httpError(400, "invalid_json", "Body did not parse as JSON.");
  }
}

export function httpError(status, code, message) {
  const err = new Error(message);
  err.status = status;
  err.code = code;
  return err;
}

/** Decode an `image.data` field that is either a data: URL or raw base64. */
export function decodeImageBase64(input) {
  if (typeof input !== "string" || !input.length) {
    throw httpError(400, "invalid_image", "image.data must be a non-empty string.");
  }
  let b64 = input;
  if (b64.startsWith("data:")) {
    const idx = b64.indexOf(",");
    if (idx < 0) throw httpError(400, "invalid_image", "image.data data: URL malformed.");
    b64 = b64.slice(idx + 1);
  }
  // atob is available in the Workers runtime.
  let binary;
  try { binary = atob(b64); } catch {
    throw httpError(400, "invalid_image", "image.data did not decode as base64.");
  }
  const buf = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) buf[i] = binary.charCodeAt(i);
  return buf;
}

// ---------- rate limiting --------------------------------------------------

/**
 * Token-bucket-lite rate limiter keyed by SHA-256(IP). Counts uploads in
 * the current rolling hour; returns { allowed, remaining, resetSeconds }.
 *
 * The raw IP is never persisted — only its hash, as a KV key. The hash is
 * truncated to 16 hex chars (64 bits) which is plenty for the 5-uploads/hour
 * counter without preserving meaningful identity.
 */
export async function checkRateLimit(env, ip, limit) {
  const ipHash = (await sha256Hex(ip || "unknown")).slice(0, 16);
  const hour = Math.floor(Date.now() / 3_600_000); // hour bucket
  const key = `rl:${ipHash}:${hour}`;
  const raw = await env.SUNDOG_KV.get(key);
  const used = raw ? Number.parseInt(raw, 10) : 0;
  if (used >= limit) {
    return { allowed: false, remaining: 0, resetSeconds: 3600 - (Math.floor(Date.now() / 1000) % 3600), key };
  }
  // KV TTL = remainder of the hour, plus 60s grace
  const ttl = 3600 - (Math.floor(Date.now() / 1000) % 3600) + 60;
  await env.SUNDOG_KV.put(key, String(used + 1), { expirationTtl: ttl });
  return { allowed: true, remaining: limit - (used + 1), resetSeconds: ttl, key };
}
