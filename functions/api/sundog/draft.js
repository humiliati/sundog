/**
 * POST /api/sundog/draft  —  cross-vendor hosted-model proxy for the chat.html demo (ticket #8).
 *
 * The DETERMINISTIC backend runs entirely client-side (no key, no network) — it never hits this
 * endpoint. This Function exists only for the HOSTED backends (gpt-4o-mini / claude-haiku-4-5 /
 * Groq llama-3.3 / qwen), whose API keys cannot live in client code.
 *
 * Flow: the browser builds the TRACE from the claim map (same router the widget uses), POSTs
 * { prompt, backend, trace } here; this Function calls the chosen hosted model with the SAME
 * heavy-trace prompt the eval harness uses (it imports the eval adapters directly), and returns the
 * raw draft. The CLIENT then runs the SAME claim gate on the draft — so the demo's discipline is
 * byte-identical to the deployed widget's, and the gate decision is shown live.
 *
 * Safety: per-IP KV rate limit (reuses `checkRateLimit`); prompt length cap; backend whitelist; and
 * a graceful 503 when a backend's key isn't configured (so the UI shows "not enabled" instead of
 * erroring). Keys are read from `env.*` (Cloudflare secrets / local .dev.vars) — never committed.
 */

import { createOpenAIAdapter } from "../../../chat/eval/lib/adapters/openai-adapter.mjs";
import { createAnthropicAdapter } from "../../../chat/eval/lib/adapters/anthropic-adapter.mjs";
import { createGroqAdapter } from "../../../chat/eval/lib/adapters/groq-adapter.mjs";
import { readJsonBody, jsonResponse, errorResponse, withCors, handleOptions, checkRateLimit } from "./_lib.js";

const HOSTED_LIMIT_PER_HOUR = 10;
const MAX_PROMPT_CHARS = 2000;

// The four hosted backends named in ship-ticket #8. The deterministic backend is NOT here — it
// runs client-side. Each entry maps to an eval adapter factory + the model id + the env key it needs.
const BACKENDS = {
  "gpt-4o-mini": { factory: createOpenAIAdapter, model: "gpt-4o-mini", keyVar: "OPENAI_API_KEY", label: "OpenAI gpt-4o-mini" },
  "claude-haiku-4-5": { factory: createAnthropicAdapter, model: "claude-haiku-4-5", keyVar: "ANTHROPIC_API_KEY", label: "Anthropic claude-haiku-4-5" },
  "llama-3.3": { factory: createGroqAdapter, model: "llama-3.3-70b-versatile", keyVar: "GROQ_API_KEY", label: "Groq llama-3.3-70b" },
  "qwen": { factory: createGroqAdapter, model: "qwen/qwen3-32b", keyVar: "GROQ_API_KEY", label: "Groq qwen3-32b" },
};

export function onRequestOptions({ request }) {
  return handleOptions(request);
}

export async function onRequestPost({ request, env }) {
  const origin = request.headers.get("origin") || "*";
  try {
    const body = await readJsonBody(request, 64 * 1024);
    const backendId = String(body?.backend || "");
    const prompt = String(body?.prompt || "").trim();
    const trace = body?.trace;

    const backend = BACKENDS[backendId];
    if (!backend) {
      return withCors(errorResponse(400, "unknown_backend", `backend must be one of: ${Object.keys(BACKENDS).join(", ")}.`), origin);
    }
    if (!prompt) {
      return withCors(errorResponse(400, "empty_prompt", "A non-empty prompt is required."), origin);
    }
    if (prompt.length > MAX_PROMPT_CHARS) {
      return withCors(errorResponse(413, "prompt_too_long", `Prompt exceeds ${MAX_PROMPT_CHARS} characters.`), origin);
    }
    if (!trace || typeof trace !== "object") {
      return withCors(errorResponse(400, "missing_trace", "A client-built trace is required (the demo builds it from the claim map)."), origin);
    }

    // Graceful when a key isn't configured on this deployment — the UI shows "not enabled".
    if (!env || !env[backend.keyVar]) {
      return withCors(errorResponse(503, "backend_unavailable", `The ${backend.label} demo is not enabled on this deployment (no API key configured).`), origin);
    }

    // Per-IP rate limit (hosted calls cost money / hit free-tier quotas). The deterministic backend
    // never reaches here, so it stays unlimited. Skipped if the KV binding is absent (e.g. local dev).
    let remaining = HOSTED_LIMIT_PER_HOUR;
    if (env.sundog_rate_limit) {
      const ip = request.headers.get("cf-connecting-ip") || request.headers.get("x-forwarded-for") || "unknown";
      const rl = await checkRateLimit(env, ip, HOSTED_LIMIT_PER_HOUR);
      if (!rl.allowed) {
        return withCors(errorResponse(429, "rate_limited", `Hosted-demo limit reached (${HOSTED_LIMIT_PER_HOUR}/hour). Try the deterministic backend, or come back later.`), origin);
      }
      remaining = rl.remaining;
    }

    const adapter = backend.factory({ env, model: backend.model });
    const draftText = await adapter.draft({ prompt, trace, context: {} });

    return withCors(jsonResponse({ draft: draftText, backend: backendId, model: backend.model, remaining }), origin);
  } catch (error) {
    const status = error?.status || 502;
    const code = error?.code || "draft_failed";
    // Don't surface provider internals — a short, safe message for unexpected failures.
    const message = error?.code ? error.message : "The hosted model call failed. Try again, or use the deterministic backend.";
    return withCors(errorResponse(status, code, message), origin);
  }
}
