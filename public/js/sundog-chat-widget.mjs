import { DEFAULT_FAQS, buildTraceAnswer, loadClaimMap } from "./sundog-chat-router.mjs";
import { attachRetrievedMatches, buildRetrievalTrace, loadChatIndex } from "./sundog-retrieval.mjs";
import { applyMascotState, applyPanelMascotState, bubbleClassFor, deriveMascotState, getMascotLabel, stampLabelFor, __PUBLIC_LABELS } from "./sundog-chat-mascot.mjs";

const ROOT_ID = "sd-chat-widget-root";
const MASCOT_CSS_HREF = "/css/sundog-chat-mascot.css";

if (!document.getElementById(ROOT_ID)) {
  ensureMascotStylesheet();
  initAskSundog();
}

// Ensure the mascot stylesheet loads without requiring every host HTML page
// to add a <link>. Idempotent — checks the document for an existing tag.
function ensureMascotStylesheet() {
  const existing = document.querySelector(`link[rel="stylesheet"][href="${MASCOT_CSS_HREF}"]`);
  if (existing) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = MASCOT_CSS_HREF;
  document.head.appendChild(link);
}

async function initAskSundog() {
  const root = document.createElement("div");
  root.id = ROOT_ID;
  root.className = "sd-chat-widget";
  root.innerHTML = `
    <button class="sd-chat-launch" type="button" aria-label="Ask Sundog help" aria-expanded="false" aria-controls="sd-chat-panel">
      <span class="sd-chat-launch__label">Help</span>
    </button>
    <section class="sd-chat-panel" id="sd-chat-panel" hidden aria-label="Ask Sundog">
      <header class="sd-chat-header">
        <div>
          <div class="sd-chat-title">Ask Sundog</div>
          <div class="sd-chat-subtitle">Trace-conditioned site helper</div>
        </div>
        <button class="sd-chat-close" type="button" aria-label="Close Ask Sundog">x</button>
      </header>
      <div class="sd-chat-mascot-strip" role="status" aria-live="polite" aria-atomic="true">
        <span class="sd-chat-mascot-strip__face" aria-hidden="true"></span>
        <span class="sd-chat-mascot-strip__label">Ready</span>
      </div>
      <div class="sd-chat-body" role="log" aria-live="polite">
        <div class="sd-chat-message sd-chat-message-assistant">
          <p>Ready.</p>
        </div>
        <div class="sd-chat-faqs" role="group" aria-label="Suggested questions"></div>
        <div class="sd-chat-answer-stack"></div>
      </div>
      <form class="sd-chat-form">
        <input class="sd-chat-input" type="search" autocomplete="off" placeholder="What does Sundog claim?" aria-label="Ask Sundog question">
        <button class="sd-chat-submit" type="submit">Ask</button>
      </form>
    </section>
  `;

  document.body.appendChild(root);

  const launch = root.querySelector(".sd-chat-launch");
  const panel = root.querySelector(".sd-chat-panel");
  const close = root.querySelector(".sd-chat-close");
  const form = root.querySelector(".sd-chat-form");
  const input = root.querySelector(".sd-chat-input");
  const body = root.querySelector(".sd-chat-body");
  const answerStack = root.querySelector(".sd-chat-answer-stack");
  const faqs = root.querySelector(".sd-chat-faqs");

  // Idle mascot state on mount. Trace-driven updates fire in answerQuestion.
  applyMascotState(launch, null);
  applyPanelMascotState(panel, null);

  // Conversation history surface — drives the `held_refusal` state when
  // the gate fires twice in a row on related routes.
  let lastTrace = null;

  // First-visit onboarding nudge — set by maybeShowOnboarding(), cleared on dismiss.
  let dismissOnboard = null;

  let claimMap = null;
  let chatIndex = null;

  renderFaqs(faqs, async (question) => {
    openPanel();
    input.value = question;
    await answerQuestion(question);
  });

  launch.addEventListener("click", () => {
    if (panel.hidden) {
      openPanel();
    } else {
      closePanel();
    }
  });

  close.addEventListener("click", closePanel);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    await answerQuestion(question);
    input.value = "";
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) {
      closePanel();
    }
  });

  maybeShowOnboarding();

  async function answerQuestion(question) {
    const trace = await traceFor(question);
    const mascotState = deriveMascotState(trace, lastTrace);
    collapsePreviousTraces(answerStack);
    answerStack.append(renderExchange(question, trace, mascotState));
    body.scrollTo({ top: body.scrollHeight, behavior: "smooth" });
    applyMascotState(launch, trace, { previousTrace: lastTrace });
    applyPanelMascotState(panel, trace, { previousTrace: lastTrace });
    lastTrace = trace;
  }

  async function traceFor(question) {
    try {
      claimMap ||= await loadClaimMap();
      const staticTrace = buildTraceAnswer(claimMap, question);

      try {
        chatIndex ||= await loadChatIndex();
        if (staticTrace.routeId === "unsupported_static_route") {
          return buildRetrievalTrace(chatIndex, question) || staticTrace;
        }
        return attachRetrievedMatches(chatIndex, question, staticTrace);
      } catch (retrievalError) {
        if (staticTrace.routeId !== "unsupported_static_route") {
          return {
            ...staticTrace,
            boundary: [
              ...(staticTrace.boundary || []),
              `Local retrieval index unavailable; answered from claim map only (${retrievalError.message}).`
            ]
          };
        }
      }

      return staticTrace;
    } catch (error) {
      return {
        answer: "The local claim map could not be loaded, so the widget is staying quiet instead of guessing.",
        intent: "claim_map_load_error",
        routeId: "claim_map_load_error",
        disposition: "blocked",
        evidenceTier: "unsupported",
        support: [],
        boundary: [
          "Do not answer without the local claim map.",
          error.message
        ],
        confidence: "high_for_load_failure",
        nextAction: {
          label: "Open chat roadmap",
          href: "/docs/SUNDOG_V_CHAT.md"
        },
        traceVisible: true
      };
    }
  }

  function openPanel() {
    if (dismissOnboard) dismissOnboard();
    panel.hidden = false;
    launch.setAttribute("aria-expanded", "true");
    window.setTimeout(() => input.focus({ preventScroll: true }), 0);
  }

  function closePanel() {
    panel.hidden = true;
    launch.setAttribute("aria-expanded", "false");
    launch.focus({ preventScroll: true });
  }

  // First-visit onboarding nudge: a one-time hint above the launcher telling a
  // new visitor what the assistant does. Shown once (localStorage-gated), and
  // dismissed on panel-open / explicit "Got it" / a 12s timeout.
  function maybeShowOnboarding() {
    let seen;
    try {
      seen = localStorage.getItem("sd-chat-onboarded");
    } catch (error) {
      return; // storage blocked (private mode) — don't nag every visit
    }
    if (seen === "1") return;

    const tip = document.createElement("div");
    tip.className = "sd-chat-onboard";
    tip.setAttribute("role", "status");

    const text = document.createElement("span");
    text.className = "sd-chat-onboard__text";
    text.textContent = "Ask Sundog anything — I show my source and won't overclaim.";

    const dismissBtn = document.createElement("button");
    dismissBtn.className = "sd-chat-onboard__dismiss";
    dismissBtn.type = "button";
    dismissBtn.textContent = "Got it";

    tip.append(text, dismissBtn);
    root.appendChild(tip);
    window.requestAnimationFrame(() => tip.classList.add("sd-chat-onboard--in"));

    dismissOnboard = () => {
      dismissOnboard = null;
      tip.remove();
      try {
        localStorage.setItem("sd-chat-onboarded", "1");
      } catch (error) {
        /* storage blocked — already handled by the read guard above */
      }
    };

    dismissBtn.addEventListener("click", dismissOnboard);
    window.setTimeout(() => {
      if (dismissOnboard) dismissOnboard();
    }, 12000);
  }
}

function collapsePreviousTraces(container) {
  for (const trace of container.querySelectorAll(".sd-chat-trace[open]")) {
    trace.open = false;
  }
}

function renderFaqs(container, onAsk) {
  container.replaceChildren(...DEFAULT_FAQS.map((question) => {
    const button = document.createElement("button");
    button.className = "sd-chat-faq";
    button.type = "button";
    button.textContent = question;
    button.addEventListener("click", () => onAsk(question));
    return button;
  }));
}

function renderExchange(question, trace, mascotState = null) {
  const fragment = document.createDocumentFragment();
  const userMessage = document.createElement("div");
  userMessage.className = "sd-chat-message sd-chat-message-user";
  userMessage.textContent = question;

  const assistantMessage = document.createElement("div");
  assistantMessage.className = "sd-chat-message sd-chat-message-assistant";
  // Bubble morphology — second-strongest visual signal after the mascot
  // face. The bubble class is the epistemic-class carrier; reduced-motion
  // users still see it because the shape itself is information, not chrome.
  const bubbleClass = bubbleClassFor(mascotState);
  if (bubbleClass) {
    assistantMessage.classList.add(`sd-chat-bubble--${bubbleClass}`);
    assistantMessage.dataset.bubble = bubbleClass;
  }
  if (mascotState) assistantMessage.dataset.mascotState = mascotState;
  // Resolve the public label from the already-derived mascot state so
  // the tier rail's state-label chip shares the same source of truth
  // as the stamp and the panel strip (and `held_refusal` keeps its
  // "Boundary Held" label even though renderTierRail can't re-derive
  // without conversation history).
  const labelForRail = mascotState ? (__PUBLIC_LABELS[mascotState] || null) : null;
  assistantMessage.append(renderTierRail(trace, labelForRail), paragraph(trace.answer), renderNextAction(trace), renderTrace(trace));
  assistantMessage.dataset.trace = JSON.stringify(trace);

  // Stamp overlay — third reinforcing visual channel. Rendered after
  // the content so it sits visually on top via absolute positioning.
  // user-select: none and pointer-events: none are enforced in CSS so
  // the stamp doesn't interfere with copy-paste of the answer text.
  // aria-hidden because the public label on the panel strip and the
  // tier-rail chips already carry the same information for AT users.
  const stampLabel = stampLabelFor(mascotState);
  if (stampLabel) {
    const stamp = document.createElement("span");
    stamp.className = "sd-chat-stamp";
    stamp.textContent = stampLabel;
    stamp.setAttribute("aria-hidden", "true");
    stamp.dataset.stampState = mascotState;
    assistantMessage.append(stamp);
    assistantMessage.classList.add("sd-chat-bubble--has-stamp");
  }

  fragment.append(userMessage, assistantMessage);
  return fragment;
}

function renderTierRail(trace, mascotLabel = null) {
  const rail = document.createElement("div");
  rail.className = "sd-chat-tier-rail";
  rail.append(chip(trace.evidenceTier || "unknown", "tier"));
  if (trace.earned) {
    rail.append(chip(earnedChipLabel(trace.earnedConfidence), "earned"));
  }
  if (trace.boundary?.length) rail.append(chip("Boundary Active", "state"));
  if (trace.disposition === "refuse") {
    rail.append(chip("Refused", "state"));
  } else if (trace.disposition === "retrieval_only") {
    rail.append(chip("Retrieval Only", "state"));
  }
  // Public mascot-state label as a distinct chip type. Redundant with
  // the panel strip header but inline with the trace summary, so a
  // visitor scanning a single message can read the discipline without
  // looking up at the strip. Skipped for "Ready" (idle) — the chip
  // would just add noise on the welcome message.
  const labelForChip = mascotLabel || getMascotLabel(trace);
  if (labelForChip && labelForChip !== "Ready") {
    rail.append(chip(labelForChip, "state-label"));
  }
  return rail;
}

function renderNextAction(trace) {
  if (!trace.nextAction) return document.createTextNode("");

  const link = document.createElement("a");
  link.className = "sd-chat-next";
  link.href = trace.nextAction.href;
  link.textContent = trace.nextAction.label;
  return link;
}

function renderTrace(trace) {
  const details = document.createElement("details");
  details.className = "sd-chat-trace";
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "Trace";
  details.append(summary);

  const meta = document.createElement("dl");
  meta.className = "sd-chat-trace-meta";
  addDefinition(meta, "Confidence", trace.confidence || "unknown");
  addDefinition(meta, "Intent", trace.intent || "static_claim_route");
  details.append(meta);

  if (trace.support?.length) {
    details.append(traceList("Sources", trace.support.map((support) => `${support.doc} - ${support.section} (${support.status})`)));
  }

  if (trace.earned || trace.earnedDetail) {
    details.append(traceList("Earned", [trace.earnedDetail || trace.earned]));
  }

  if (trace.boundary?.length) {
    details.append(traceList("Boundary", trace.boundary));
  }

  if (trace.retrieved?.length) {
    details.append(traceList("Retrieved", trace.retrieved.map((match) => `${match.doc} - ${match.section} [${match.tier}; score ${match.score}]`)));
  }

  return details;
}

function traceList(label, items) {
  const section = document.createElement("section");
  const heading = document.createElement("h4");
  heading.textContent = label;
  const list = document.createElement("ul");
  for (const item of items) {
    const entry = document.createElement("li");
    entry.textContent = item;
    list.append(entry);
  }
  section.append(heading, list);
  return section;
}

function addDefinition(list, term, value) {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  list.append(dt, dd);
}

function chip(value, kind = "tier") {
  const span = document.createElement("span");
  span.className = `sd-chat-chip sd-chat-chip--${kind}`;
  span.textContent = String(value || "unknown").replaceAll("_", " ");
  return span;
}

function earnedChipLabel(confidence) {
  switch (confidence) {
    case "under_review":
      return "Earned - under review";
    case "promising":
      return "Earned - partial";
    case "null":
      return "Earned - method";
    case "education":
      return "Explainer";
    case "inventory":
      return "Earned inventory";
    case "win":
    default:
      return "Earned";
  }
}

function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}
