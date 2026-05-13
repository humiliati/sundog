import {
  jsonResponse, errorResponse, withCors, handleOptions,
  readJsonBody, sha256Hex, httpError,
} from "./_lib.js";

/**
 * POST /api/sundog/delete
 * GET  /api/sundog/delete?token=...     (convenience — emails / shared URLs)
 *
 * Caller provides the deletion token they received at submission time.
 * We SHA-256 it, look up the R2 key prefix in KV, and delete all three
 * objects (image, pose.json, meta.json). The KV index entry is also
 * removed so the token can't be replayed.
 *
 * Response: { status: "deleted", submission_id } on success;
 *           { status: "not_found" } if the token doesn't map to anything.
 */

export const onRequestOptions = ({ request }) => handleOptions(request);

async function handleDelete(env, token, origin) {
  if (!env.sundog_uploads) return withCors(errorResponse(503, "r2_not_bound", "R2 not configured."), origin);
  if (!env.sundog_rate_limit)      return withCors(errorResponse(503, "kv_not_bound", "KV not configured."), origin);
  if (typeof token !== "string" || token.length < 16) {
    return withCors(errorResponse(400, "invalid_token", "Token missing or too short."), origin);
  }
  const tokenHash = await sha256Hex(token);
  const baseKey = await env.sundog_rate_limit.get(`del:${tokenHash}`);
  if (!baseKey) {
    return withCors(jsonResponse({ status: "not_found" }, { status: 404 }), origin);
  }

  // Read meta before deleting the index so we can return the submission_id
  // (best-effort; if meta is already gone, we still return success).
  let submissionId = null;
  try {
    const metaObj = await env.sundog_uploads.get(`${baseKey}.meta.json`);
    if (metaObj) submissionId = JSON.parse(await metaObj.text()).submission_id;
  } catch { /* ignore */ }

  // Try every plausible extension; only the one that exists will succeed.
  const candidates = [`${baseKey}.jpg`, `${baseKey}.png`, `${baseKey}.webp`,
                      `${baseKey}.pose.json`, `${baseKey}.meta.json`];
  const results = await Promise.all(candidates.map(async (k) => {
    try { await env.sundog_uploads.delete(k); return { key: k, ok: true }; }
    catch (e) { return { key: k, ok: false, error: String(e) }; }
  }));

  await env.sundog_rate_limit.delete(`del:${tokenHash}`);

  return withCors(jsonResponse({
    status: "deleted",
    submission_id: submissionId,
    deleted_keys: results.filter((r) => r.ok).map((r) => r.key),
  }), origin);
}

export async function onRequestPost(ctx) {
  const origin = ctx.request.headers.get("origin") || "*";
  try {
    const body = await readJsonBody(ctx.request, 4096);
    return await handleDelete(ctx.env, body?.token, origin);
  } catch (err) {
    if (err?.status) return withCors(errorResponse(err.status, err.code || "error", err.message), origin);
    return withCors(errorResponse(500, "internal_error", "Unexpected error."), origin);
  }
}

export async function onRequestGet(ctx) {
  const origin = ctx.request.headers.get("origin") || "*";
  const url = new URL(ctx.request.url);
  const token = url.searchParams.get("token");
  return await handleDelete(ctx.env, token, origin);
}

export const onRequest = (ctx) => {
  const m = ctx.request.method;
  if (m === "POST") return onRequestPost(ctx);
  if (m === "GET")  return onRequestGet(ctx);
  if (m === "OPTIONS") return handleOptions(ctx.request);
  return withCors(errorResponse(405, "method_not_allowed", "Use GET or POST."));
};
