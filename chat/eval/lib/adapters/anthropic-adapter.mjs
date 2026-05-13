// Anthropic Claude hosted-model adapter for Phase 8d cross-vendor pass.
//
// Contract: matches the existing `draftWithAdapter` shape in
// `public/js/sundog-claim-gate.mjs`. Adapter.draft({prompt, trace,
// context}) returns a single string — the model's draft answer.
//
// Trace handoff: HEAVY, identical to the OpenAI adapter so cross-vendor
// comparison is fair. Same system prompt, same user-message JSON
// payload (full trace + referenceAnswer). Only difference: the
// Messages API takes `system` as a top-level parameter, not as a
// `messages` entry.
//
// Configuration via env:
//   ANTHROPIC_API_KEY     — required.
//   ANTHROPIC_MODEL       — defaults to "claude-haiku-4-5".
//   ANTHROPIC_BASE_URL    — defaults to "https://api.anthropic.com/v1".
//   ANTHROPIC_TEMPERATURE — defaults to "0".
//
// On HTTP/network/parse error the adapter throws with a clear message
// so the runner can decide whether to retry or skip.

import { buildUserMessage, __SYSTEM_PROMPT } from "./openai-adapter.mjs";

const DEFAULT_MODEL = "claude-haiku-4-5";
const DEFAULT_BASE_URL = "https://api.anthropic.com/v1";
const DEFAULT_TEMPERATURE = 0;
const ANTHROPIC_VERSION = "2023-06-01";

export function createAnthropicAdapter(options = {}) {
  const env = options.env || (typeof process !== "undefined" ? process.env : {});
  const apiKey = options.apiKey ?? env.ANTHROPIC_API_KEY;
  const model = options.model ?? env.ANTHROPIC_MODEL ?? DEFAULT_MODEL;
  const baseUrl = (options.baseUrl ?? env.ANTHROPIC_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/$/, "");
  const temperature = Number(options.temperature ?? env.ANTHROPIC_TEMPERATURE ?? DEFAULT_TEMPERATURE);

  if (!apiKey) {
    throw new Error("AnthropicAdapter: ANTHROPIC_API_KEY is not set. Export the key before running the hosted runner.");
  }

  const info = { backend: "anthropic", model, temperature, baseUrl };

  async function draft({ prompt, trace, context = {} }) {
    const userMessage = buildUserMessage({ prompt, trace, context });
    const body = {
      model,
      temperature,
      max_tokens: 320,
      system: __SYSTEM_PROMPT,
      messages: [
        { role: "user", content: userMessage }
      ]
    };

    const response = await fetch(`${baseUrl}/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": ANTHROPIC_VERSION
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`AnthropicAdapter: ${response.status} ${response.statusText} — ${text.slice(0, 200)}`);
    }

    const payload = await response.json();
    // Messages API: content is an array of blocks; we want the text from
    // the first text-typed block.
    const textBlock = Array.isArray(payload?.content)
      ? payload.content.find((b) => b?.type === "text")
      : null;
    const draftText = textBlock?.text?.trim();
    if (!draftText) {
      throw new Error(`AnthropicAdapter: empty draft text in response (stop_reason=${payload?.stop_reason}, usage=${JSON.stringify(payload?.usage)}).`);
    }

    return draftText;
  }

  return { info, draft };
}
