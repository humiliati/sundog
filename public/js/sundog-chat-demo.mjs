/**
 * sundog-chat-demo.mjs — the cross-vendor demo on chat.html (ship-ticket #8).
 *
 * Pick a backend, ask a question, watch the SAME claim gate hold the line:
 *   - "deterministic" runs entirely in the browser (router builds the trace; the answer IS the
 *     supported template). No network, no key.
 *   - hosted backends POST { prompt, backend, trace } to /api/sundog/draft; the returned model draft
 *     is then run through gateModelDraft() RIGHT HERE — identical to the deployed widget — so the
 *     demo shows whether the model's draft PASSED the gate or was rejected and replaced with the
 *     supported answer. That accept/reject decision is the whole point of the page.
 *
 * Untrusted model text is rendered with textContent only (never innerHTML).
 */

import { buildTraceAnswer, loadClaimMap } from "./sundog-chat-router.mjs";
import { gateModelDraft } from "./sundog-claim-gate.mjs";

const HOSTED_LABEL = {
  "gpt-4o-mini": "gpt-4o-mini",
  "claude-haiku-4-5": "claude-haiku-4-5",
  "llama-3.3": "llama-3.3-70b",
  "qwen": "qwen3-32b",
};

const root = document.getElementById("sd-demo");
if (root) init(root);

function init(root) {
  const form = root.querySelector("#sd-demo-form");
  const backendSel = root.querySelector("#sd-demo-backend");
  const input = root.querySelector("#sd-demo-input");
  const ask = root.querySelector("#sd-demo-ask");
  const output = root.querySelector("#sd-demo-output");
  let claimMap = null;

  for (const btn of root.querySelectorAll(".sd-demo-preset")) {
    btn.addEventListener("click", () => { input.value = btn.textContent; input.focus(); });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const prompt = input.value.trim();
    if (prompt) run(prompt, backendSel.value);
  });

  async function run(prompt, backend) {
    setBusy(true);
    showLoading(backend);
    try {
      claimMap ||= await loadClaimMap();
      const trace = buildTraceAnswer(claimMap, prompt);
      trace.traceVisible = true; // the demo always shows the trace drawer

      if (backend === "deterministic") {
        renderResult({ backend, answer: trace.answer, trace, gate: null });
        return;
      }

      const res = await fetch("/api/sundog/draft", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt, backend, trace }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        renderError(backend, data?.error?.message || `Request failed (${res.status}).`);
        return;
      }
      // Run the SAME gate the deployed widget runs, on the model's draft.
      const gated = gateModelDraft({ prompt, trace, draft: data.draft, context: {} });
      renderResult({ backend, answer: gated.answer, trace, gate: gated.draft, model: data.model, proposed: data.draft });
    } catch (error) {
      renderError(backend, "Could not reach the demo. Check your connection, or try the deterministic backend.");
    } finally {
      setBusy(false);
    }
  }

  function setBusy(busy) { ask.disabled = busy; ask.textContent = busy ? "…" : "Ask"; }

  function showLoading(backend) {
    output.hidden = false;
    output.replaceChildren(el("p", "sd-demo-loading", backend === "deterministic"
      ? "Routing through the claim map…"
      : `Drafting with ${HOSTED_LABEL[backend] || backend}, then checking the gate…`));
  }

  function renderError(backend, message) {
    output.hidden = false;
    const box = el("div", "sd-demo-card sd-demo-card--error");
    box.append(el("p", "sd-demo-badge sd-demo-badge--blocked", "Demo unavailable"), el("p", "sd-demo-answer", message));
    output.replaceChildren(box);
  }

  function renderResult({ backend, answer, trace, gate, model }) {
    output.hidden = false;
    const card = el("div", "sd-demo-card");

    // Gate verdict — the discipline, made visible.
    let badge;
    if (backend === "deterministic") {
      badge = el("p", "sd-demo-badge sd-demo-badge--det", "Deterministic — answered from the supported template (no model).");
    } else if (gate && gate.status === "accepted") {
      badge = el("p", "sd-demo-badge sd-demo-badge--pass", `✓ ${model || backend}'s draft passed the gate.`);
    } else {
      badge = el("p", "sd-demo-badge sd-demo-badge--reject", `✗ ${model || backend}'s draft was rejected by the gate — showing the supported answer instead.`);
    }
    card.append(badge);

    card.append(el("p", "sd-demo-answer", answer || "(no answer)"));

    // The trace: evidence tier, disposition, boundary rules, sources.
    const rail = el("div", "sd-demo-rail");
    rail.append(chip("tier", "evidence: " + (trace.evidenceTier || "unknown")));
    if (trace.disposition) rail.append(chip("disp", trace.disposition.replaceAll("_", " ")));
    card.append(rail);

    if (gate && gate.status === "rejected" && Array.isArray(gate.failures) && gate.failures.length) {
      card.append(el("p", "sd-demo-failures-label", "Gate caught:"));
      const ul = el("ul", "sd-demo-failures");
      for (const f of gate.failures.slice(0, 6)) ul.append(el("li", null, f.replaceAll("_", " ")));
      card.append(ul);
    }

    if (Array.isArray(trace.boundary) && trace.boundary.length) {
      card.append(el("p", "sd-demo-sub", "Boundary held:"));
      const ul = el("ul", "sd-demo-boundary");
      for (const b of trace.boundary.slice(0, 4)) ul.append(el("li", null, b));
      card.append(ul);
    }

    if (Array.isArray(trace.support) && trace.support.length) {
      const src = trace.support.map((s) => `${s.doc}${s.section ? " — " + s.section : ""}`).join(" · ");
      card.append(el("p", "sd-demo-source", "Source: " + src));
    }

    output.replaceChildren(card);
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text; // textContent — safe for untrusted model output
    return node;
  }
  function chip(kind, text) { return el("span", "sd-demo-chip sd-demo-chip--" + kind, text); }
}
