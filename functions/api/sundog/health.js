import { jsonResponse, errorResponse, withCors, handleOptions } from "./_lib.js";

/**
 * GET /api/sundog/health
 *
 * Returns { ok, ts, bindings } so monitoring + the frontend can verify the
 * Phase 5B Worker is reachable AND that R2 + KV bindings are wired before
 * showing the share-for-training opt-in as live.
 */

export const onRequestOptions = ({ request }) => handleOptions(request);

export async function onRequestGet({ env }) {
  const r2Ready = !!env.sundog_uploads;
  const kvReady = !!env.sundog_rate_limit;
  const policyVersion = env.POLICY_VERSION || "unknown";
  return withCors(
    jsonResponse({
      ok: r2Ready && kvReady,
      ts: new Date().toISOString(),
      bindings: { r2: r2Ready, kv: kvReady },
      policy_version: policyVersion,
    }),
  );
}

export const onRequest = ({ request, env }) => {
  if (request.method === "GET") return onRequestGet({ env });
  if (request.method === "OPTIONS") return handleOptions(request);
  return withCors(errorResponse(405, "method_not_allowed", "Use GET."));
};
