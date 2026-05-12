export const DEFAULT_FAQS = [
  "What does Sundog claim?",
  "What does Sundog not claim?",
  "Show me the strongest safe claim.",
  "What is the current controlled result?",
  "What is Alignment Without Sight?",
  "What evidence tiers does Sundog use?",
  "Which applications are research results?",
  "What is an operating-envelope study?",
  "What is the Three-Body Sundog approach?",
  "Does Sundog solve the three-body problem?",
  "What is the Sundog Balance workbench?",
  "Does Balance prove robotics control?",
  "What is Pressure Mines?",
  "Does Sundog solve Minesweeper?",
  "What is EyesOnly / Gone Rogue?",
  "What is Dungeon Gleaner?",
  "What is Money Bags?",
  "What is the mesa experiment testing?",
  "Does Sundog prevent reward hacking?",
  "Does Sundog solve inner alignment?",
  "What is Ask Sundog?",
  "Is Sundog Chat an LLM-alignment result?",
  "Where can I inspect the data?",
  "Is Sundog a crypto project?",
  "Is this project a SunDog: Frozen Legacy port?"
];

const CLAIM_MAP_URLS = [
  "/data/sundog-claim-map.json",
  "/chat/claim_map.json"
];

let claimMapPromise;

export async function loadClaimMap() {
  if (!claimMapPromise) {
    claimMapPromise = loadFirstJson(CLAIM_MAP_URLS);
  }

  return claimMapPromise;
}

async function loadFirstJson(urls) {
  const failures = [];
  for (const url of urls) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        failures.push(`${url}: ${response.status}`);
        continue;
      }
      return response.json();
    } catch (error) {
      failures.push(`${url}: ${error.message}`);
    }
  }

  throw new Error(`Unable to load claim map (${failures.join("; ")})`);
}

export function routePrompt(claimMap, prompt) {
  const normalizedPrompt = normalizeText(prompt);
  const routes = [
    ...(claimMap.claims || []),
    ...(claimMap.nonClaimRoutes || [])
  ];

  let fallbackRoute = null;

  for (const route of routes) {
    const patterns = route.questionPatterns || [];
    const exactPattern = patterns.find((pattern) => normalizeText(pattern) === normalizedPrompt);
    if (exactPattern) {
      return route;
    }

    const partialPattern = patterns.find((pattern) => {
      const normalizedPattern = normalizeText(pattern);
      return normalizedPattern.length >= 8 && normalizedPrompt.includes(normalizedPattern);
    });
    if (partialPattern && !fallbackRoute) {
      fallbackRoute = route;
    }
  }

  return fallbackRoute;
}

export function buildTraceAnswer(claimMap, prompt) {
  const route = routePrompt(claimMap, prompt);

  if (!route) {
    return {
      answer: "I do not have a supported static answer for that yet. The safe route is to ask about the current controlled result, evidence tiers, application boundaries, mesa roadmap, or source confusion.",
      intent: "unsupported_static_route",
      routeId: "unsupported_static_route",
      disposition: "unsupported",
      evidenceTier: "unsupported",
      support: [],
      boundary: [
        "Do not improvise beyond the claim map.",
        "Route unsupported questions to the nearest source-boundary document."
      ],
      confidence: "high_for_static_router_limit",
      nextAction: {
        label: "Open chat roadmap",
        href: "/docs/SUNDOG_V_CHAT.md"
      },
      traceVisible: true
    };
  }

  return {
    answer: route.answerTemplate,
    intent: "static_claim_route",
    routeId: route.id,
    disposition: route.disposition || "allow",
    evidenceTier: route.evidenceTier || "unknown",
    support: route.support || [],
    boundary: route.boundaries || [],
    confidence: confidenceFor(route),
    nextAction: route.nextAction || null,
    traceVisible: true
  };
}

function confidenceFor(route) {
  if (route.disposition === "refuse") {
    return "high_for_boundary_refusal";
  }
  if ((route.support || []).some((support) => support.status === "primary")) {
    return "high_for_static_claim_route";
  }
  return "medium_for_static_navigation";
}

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}
