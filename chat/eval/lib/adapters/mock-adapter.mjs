// Mock hosted-adapter — for harness smoke-testing without API cost.
//
// Same contract as the OpenAI adapter: `adapter.draft({prompt, trace,
// context})` returns a string. The output is deterministic and varies
// with the trace fields it sees, so:
//
//  - the runner integration can be verified end-to-end in CI/sandbox
//  - Phase 5 interventions on the mock backend register flips (so we
//    can prove the harness detects causal authority when the model
//    *does* use the trace)
//  - a real OpenAI run can be compared against the mock to see how
//    much variation comes from the model versus the framing
//
// Design — the mock "model" produces drafts that try to follow the
// system prompt but with characteristic LLM-shaped failure modes:
//
//   - If disposition === "refuse", it issues a refusal that names the
//     boundary. (It would lift the forbidden phrase into the refusal
//     if the boundary array is empty — that's the only field-conditional
//     failure mode the mock simulates, so removing the boundary should
//     produce a flip under intervention.)
//   - Otherwise it paraphrases the referenceAnswer and tacks on a brief
//     source citation built from `support` if present.
//   - It uses upgrade language only if `evidenceTier === "research_result"`.
//   - It lifts promo content from retrieved[0] if `support` is empty
//     (simulating retrieval-order capture when support is removed).

export function createMockAdapter(options = {}) {
  const info = {
    backend: "mock",
    model: "mock-llm-shaped-v0",
    temperature: 0,
    deterministic: true,
    seed: options.seed ?? 0
  };

  async function draft({ prompt, trace, context = {} }) {
    if (!trace || typeof trace !== "object") {
      return "I do not have a supported answer for that question. Ask Sundog answers only from the supported claim map.";
    }

    const disposition = trace.disposition || "unknown";
    const tier = trace.evidenceTier || "unknown";
    const support = Array.isArray(trace.support) ? trace.support : [];
    const boundary = Array.isArray(trace.boundary) ? trace.boundary : [];
    const retrieved = Array.isArray(trace.retrieved) ? trace.retrieved : [];
    const ref = String(trace.answer || "").trim();

    // 1. Refusal route handling. If disposition is "refuse" AND boundary
    //    is populated, refuse cleanly. If boundary is empty (intervention
    //    removed it), the mock fails — it doesn't have the rule text to
    //    cite, so it lifts the prompt's forbidden phrase. This makes
    //    boundary_removed a load-bearing intervention on the mock.
    if (disposition === "refuse") {
      if (boundary.length > 0) {
        const rule = boundary[0];
        return `No - I cannot make that claim. The boundary "${rule}" controls this route. The supported alternative is to point at the current controlled result on sundog.cc and the evidence tier the page carries.`;
      }
      // No boundary rule available → mock fails. It tries to be helpful
      // and ends up echoing the forbidden phrase from context.
      const forbidden = (context.forbidden || [])[0] || "the stronger claim";
      return `Yes - based on the available evidence, the answer is that ${forbidden}. (Note: the trace did not include a specific boundary rule, so this draft falls back to the user-facing framing.)`;
    }

    // 2. Out-of-scope route — terse, no claim, no citation.
    if (trace.routeId === "unsupported_static_route") {
      const category = context.category || "";
      if (category === "comparison" || category === "off_topic" || category === "prompt_injection") {
        return "I do not have a supported Sundog-corpus answer for that. The safe route is to ask about the current controlled result, the framework pattern, or the evidence-tiered application map.";
      }
      return ref || "I do not have a supported answer in the Sundog corpus for that question.";
    }

    // 3. Retrieval-only — the LLM-style mock paraphrases retrieved[0].
    if (disposition === "retrieval_only") {
      const top = retrieved[0];
      if (top?.text) {
        return `Based on the retrieved evidence: ${top.text.slice(0, 220).trim()} Source: ${top.doc || "(unknown)"}.`;
      }
      return ref || "I have retrieval matches but no supported claim for that question.";
    }

    // 4. Normal allow_with_boundary path. Paraphrase the reference answer
    //    and add a citation. If support is empty (intervention removed
    //    it) but retrieved has a promo entry up front, lift the promo
    //    content — that's the retrieval-order-capture failure mode the
    //    deterministic compositor doesn't exhibit.
    let body = ref || "The supported framing is described in the Sundog claim map for this route.";
    if (support.length === 0 && retrieved.length > 0 && /landing|promo|hero/i.test(retrieved[0].doc || "")) {
      body = retrieved[0].text || body;
    }

    // 5. Upgrade-language discipline tied to the tier.
    //    Without research_result tier, soften language. With it, allow.
    if (tier !== "research_result") {
      body = body
        .replace(/\bproves\b/gi, "supports")
        .replace(/\bproved\b/gi, "supported")
        .replace(/\bsolves\b/gi, "addresses")
        .replace(/\bsolved\b/gi, "addressed")
        .replace(/\bvalidated\b/gi, "treated as established")
        .replace(/\bpaper grade\b/gi, "documented result")
        .replace(/\bpaper-grade\b/gi, "documented result");
    }

    // 6. Citation tail.
    const citation = support.length
      ? ` Source: ${support.slice(0, 2).map((s) => `${s.doc}${s.section ? ` (${s.section})` : ""}`).join("; ")}.`
      : "";

    return `${body}${citation}`;
  }

  return { info, draft };
}
