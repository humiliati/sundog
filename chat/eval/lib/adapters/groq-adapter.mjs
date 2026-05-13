// Groq hosted-model adapter for Phase 12 cross-architecture pass.
//
// Groq's API is OpenAI-compatible (Chat Completions shape), so this
// adapter reuses the OpenAI adapter's `buildUserMessage` and
// `__SYSTEM_PROMPT` to keep the trace handoff identical across vendors.
// The only differences are the base URL, the auth header (Bearer), and
// the model id.
//
// Configuration via env:
//   GROQ_API_KEY     — required.
//   GROQ_MODEL       — defaults to "llama-3.3-70b-versatile".
//   GROQ_BASE_URL    — defaults to "https://api.groq.com/openai/v1".
//   GROQ_TEMPERATURE — defaults to "0".

import { buildUserMessage, __SYSTEM_PROMPT } from "./openai-adapter.mjs";

const DEFAULT_MODEL = "llama-3.3-70b-versatile";
const DEFAULT_BASE_URL = "https://api.groq.com/openai/v1";
const DEFAULT_TEMPERATURE = 0;

export function createGroqAdapter(options = {}) {
  const env = options.env || (typeof process !== "undefined" ? process.env : {});
  const apiKey = options.apiKey ?? env.GROQ_API_KEY;
  const model = options.model ?? env.GROQ_MODEL ?? DEFAULT_MODEL;
  const baseUrl = (options.baseUrl ?? env.GROQ_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/$/, "");
  const temperature = Number(options.temperature ?? env.GROQ_TEMPERATURE ?? DEFAULT_TEMPERATURE);

  if (!apiKey) {
    throw new Error("GroqAdapter: GROQ_API_KEY is not set. Export the key before running the hosted runner.");
  }

  const info = { backend: "groq", model, temperature, baseUrl };

  async function draft({ prompt, trace, context = {} }) {
    const userMessage = buildUserMessage({ prompt, trace, context });
    const body = {
      model,
      temperature,
      max_tokens: 320,
      messages: [
        { role: "system", content: __SYSTEM_PROMPT },
        { role: "user", content: userMessage }
      ]
    };

    // Retry on 429 with exponential backoff. Groq's free tier has tight
    // tokens-per-minute limits; the heavy-trace payload (~500-1000 input
    // tokens per call) burns through quickly when running a full slate.
    const MAX_ATTEMPTS = 5;
    let lastError = null;
    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
      const response = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${apiKey}`
        },
        body: JSON.stringify(body)
      });

      if (response.ok) {
        const payload = await response.json();
        let draftText = payload?.choices?.[0]?.message?.content?.trim();
        if (!draftText) {
          throw new Error(`GroqAdapter: empty draft text in response (usage=${JSON.stringify(payload?.usage)}).`);
        }
        // Strip reasoning-model <think>...</think> blocks before returning.
        // Qwen3-32B (and other reasoning models on Groq) emit their internal
        // chain-of-thought before the final answer. The gate's content
        // rules would catch forbidden phrases inside the thinking, even
        // when the actual answer is disciplined. The reasoning trace is
        // not the answer the user sees; stripping it matches what the
        // real widget would do.
        draftText = draftText.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
        // Handle unterminated reasoning: if the response opens with <think>
        // but never closes the tag (max_tokens cut it off mid-thought),
        // the whole response is reasoning. Mark the draft as truncated
        // rather than feed reasoning text to the gate.
        if (/^<think>/i.test(draftText) && !/<\/think>/i.test(draftText)) {
          draftText = "[reasoning trace truncated before final answer]";
        }
        if (!draftText) {
          throw new Error(`GroqAdapter: draft was only reasoning trace (usage=${JSON.stringify(payload?.usage)}).`);
        }
        return draftText;
      }

      // 429 → wait and retry. Use the `Retry-After` header if present;
      // otherwise exponential backoff starting at 4s.
      if (response.status === 429 && attempt < MAX_ATTEMPTS - 1) {
        const retryAfterHeader = response.headers.get("retry-after");
        const retryAfterSec = retryAfterHeader ? Math.ceil(Number(retryAfterHeader)) : null;
        const backoffMs = retryAfterSec
          ? Math.min(retryAfterSec * 1000, 30000)
          : Math.min(4000 * Math.pow(2, attempt), 30000);
        await new Promise((r) => setTimeout(r, backoffMs));
        continue;
      }

      const text = await response.text().catch(() => "");
      lastError = new Error(`GroqAdapter: ${response.status} ${response.statusText} — ${text.slice(0, 300)}`);
      break;
    }
    throw lastError || new Error("GroqAdapter: exhausted retry attempts.");
  }

  return { info, draft };
}
