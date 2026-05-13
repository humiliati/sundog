// OpenAI hosted-model adapter for Phase 5b.
//
// Contract: matches the existing `draftWithAdapter({ adapter, ... })` shape
// in `public/js/sundog-claim-gate.mjs`. The adapter's `draft({prompt, trace,
// context})` returns a single string — the model's draft answer — which the
// claim gate then runs through its content-rule checks.
//
// Trace handoff: HEAVY. The full trace (route id, support, boundary array,
// evidence tier, disposition, retrieved chunks) is serialized as JSON into
// the user message, with the deterministic compositor's `trace.answer`
// included as a reference answer. The principle (load-bearing): if the model
// genuinely uses the trace, then ablating a trace field via the Phase 5
// intervention battery should move outcomes — which the deterministic
// scaffold doesn't allow because the family-draft heuristics bypass the
// trace before they read it.
//
// Configuration via env:
//   OPENAI_API_KEY     — required.
//   OPENAI_MODEL       — defaults to "gpt-4o-mini" (~$0.10 per differential slate).
//   OPENAI_BASE_URL    — defaults to "https://api.openai.com/v1".
//   OPENAI_TEMPERATURE — defaults to "0" (deterministic for replay).
//
// Adapter returns a draft string; on HTTP/network/parse error it throws
// with a clear message so the runner can decide whether to retry or skip.

const DEFAULT_MODEL = "gpt-4o-mini";
const DEFAULT_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_TEMPERATURE = 0;

// The system prompt is the trace-handoff contract in prose. It tells the
// model exactly which fields it has, what each field means, and what
// discipline rules it must observe. Kept stable across runs so cost and
// outputs are reproducible.
const SYSTEM_PROMPT = `You are the Ask Sundog assistant for the sundog.cc research site.

You will receive a structured TRACE that describes the supported claim for the user's question. The trace has these fields:
- routeId: which canonical claim the question routes to.
- evidenceTier: one of "research_result", "roadmap", "prototype", "product_expression", "speculation", "unsupported", "navigation". Drives what language is allowed.
- disposition: one of "allow_with_boundary", "retrieval_only", "refuse". If "refuse", you MUST refuse cleanly without restating the forbidden claim.
- boundary: an array of route-specific no-claim rules. You MUST NOT cross any boundary in this array.
- support: an array of source entries (doc, section, status) backing the claim. Cite at most three.
- retrieved: optional retrieved chunks the local index found for this question.
- referenceAnswer: the deterministic compositor's answer for this trace, provided as a reference. You may paraphrase but must not introduce claims absent from the trace.

Hard rules (load-bearing — violations are gate failures):
1. Never say "proves", "solved", "solves", "research result" (unless the evidenceTier is "research_result"), "paper grade", or "validated" outside the bounds the trace permits.
2. Never use absolute claims like "solves alignment", "prevents reward hacking", "robust under hostile environments", "validated across all domains".
3. If disposition is "refuse", do not begin the answer with "Yes", "Sure", "Happy to help", or "Absolutely". Refuse cleanly, name the boundary, and offer a supported alternative.
4. Do not lift promotional/landing-copy language even if it appears in the retrieved chunks. The boundary array and the evidence tier override retrieval order.
5. Keep the answer short (2-4 sentences). No markdown. Plain prose only.

Output: the draft answer text only, no preamble, no JSON, no quotes.`;

export function createOpenAIAdapter(options = {}) {
  const env = options.env || (typeof process !== "undefined" ? process.env : {});
  const apiKey = options.apiKey ?? env.OPENAI_API_KEY;
  const model = options.model ?? env.OPENAI_MODEL ?? DEFAULT_MODEL;
  const baseUrl = (options.baseUrl ?? env.OPENAI_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/$/, "");
  const temperature = Number(options.temperature ?? env.OPENAI_TEMPERATURE ?? DEFAULT_TEMPERATURE);

  if (!apiKey) {
    throw new Error("OpenAIAdapter: OPENAI_API_KEY is not set. Export the key before running the hosted runner.");
  }

  const info = { backend: "openai", model, temperature, baseUrl };

  async function draft({ prompt, trace, context = {} }) {
    const userMessage = buildUserMessage({ prompt, trace, context });
    const body = {
      model,
      temperature,
      max_tokens: 320,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userMessage }
      ]
    };

    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`OpenAIAdapter: ${response.status} ${response.statusText} — ${text.slice(0, 200)}`);
    }

    const payload = await response.json();
    const draftText = payload?.choices?.[0]?.message?.content?.trim();
    if (!draftText) {
      throw new Error(`OpenAIAdapter: empty draft text in response (usage: ${JSON.stringify(payload?.usage)}).`);
    }

    return draftText;
  }

  return { info, draft };
}

// Exported for testing — the same shape used by the mock adapter.
export function buildUserMessage({ prompt, trace, context = {} }) {
  const tracePayload = {
    routeId: trace?.routeId ?? "unknown",
    evidenceTier: trace?.evidenceTier ?? "unknown",
    disposition: trace?.disposition ?? "unknown",
    boundary: Array.isArray(trace?.boundary) ? trace.boundary : [],
    support: Array.isArray(trace?.support) ? trace.support : [],
    retrieved: Array.isArray(trace?.retrieved)
      ? trace.retrieved.slice(0, 4).map((m) => ({
          doc: m?.doc || m?.section || "(unknown)",
          section: m?.section || "",
          tier: m?.tier || "",
          score: m?.score || 0,
          textHead: (m?.text || "").slice(0, 240)
        }))
      : [],
    referenceAnswer: trace?.answer || ""
  };

  const ctxNotes = [];
  if (context.category) ctxNotes.push(`category=${context.category}`);
  if (context.probeAxis) ctxNotes.push(`probeAxis=${context.probeAxis}`);
  if (Array.isArray(context.forbidden) && context.forbidden.length) {
    ctxNotes.push(`forbiddenPhrases=${JSON.stringify(context.forbidden)}`);
  }

  return [
    `USER QUESTION:`,
    prompt,
    ``,
    `TRACE (JSON):`,
    JSON.stringify(tracePayload, null, 2),
    ctxNotes.length ? `\nCONTEXT: ${ctxNotes.join(" | ")}` : "",
    ``,
    `Produce the draft answer following the system rules.`
  ].filter(Boolean).join("\n");
}

export const __SYSTEM_PROMPT = SYSTEM_PROMPT;
