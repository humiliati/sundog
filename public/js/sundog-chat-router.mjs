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
  "Is the code open source?",
  "How do I cite Sundog?",
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
  const correctedPrompt = correctPromptTokens(normalizedPrompt, routeVocabulary(routes));

  let fallbackRoute = null;
  let fuzzyRoute = null;
  let fuzzyScore = 0;
  let fuzzyMargin = 0;

  for (const route of routes) {
    const patterns = route.questionPatterns || [];
    const exactPattern = patterns.find((pattern) => normalizeText(pattern) === correctedPrompt);
    if (exactPattern) {
      return route;
    }

    const partialPattern = patterns.find((pattern) => {
      const normalizedPattern = normalizeText(pattern);
      return normalizedPattern.length >= 8 && correctedPrompt.includes(normalizedPattern);
    });
    if (partialPattern && !fallbackRoute) {
      fallbackRoute = route;
    }

    if (!fallbackRoute) {
      const routeScore = Math.max(0, ...patterns.map((pattern) => fuzzyPatternScore(correctedPrompt, pattern)));
      if (routeScore > fuzzyScore) {
        fuzzyMargin = routeScore - fuzzyScore;
        fuzzyScore = routeScore;
        fuzzyRoute = route;
      } else if (routeScore > 0) {
        fuzzyMargin = Math.min(fuzzyMargin, fuzzyScore - routeScore);
      }
    }
  }

  if (fallbackRoute) return fallbackRoute;

  // Fuzzy routing is a typo-tolerance fallback, not a semantic router. It
  // requires a strong score plus a margin so misspellings like "Sundgo cliam"
  // can land, while vague cross-route prompts still fall through safely.
  if (fuzzyRoute && fuzzyScore >= 2.35 && fuzzyMargin >= 0.35) {
    return fuzzyRoute;
  }

  return null;
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

const ROUTER_STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "can", "current", "does", "for", "how",
  "i", "in", "is", "it", "me", "of", "on", "or", "show", "the", "this", "to",
  "what", "where", "which", "with"
]);

function fuzzyPatternScore(normalizedPrompt, pattern) {
  const promptTokens = significantTokens(normalizedPrompt);
  const patternTokens = significantTokens(normalizeText(pattern));
  if (!promptTokens.length || !patternTokens.length) return 0;

  let score = 0;
  let matched = 0;

  for (const token of patternTokens) {
    const best = bestTokenMatch(token, promptTokens);
    if (best >= 0.72) matched += 1;
    score += best;
  }

  const coverage = matched / patternTokens.length;
  const average = score / patternTokens.length;
  const compactPattern = patternTokens.join(" ");
  const compactPrompt = promptTokens.join(" ");
  const phraseBonus = compactPrompt.includes(compactPattern) ? 0.75 : 0;

  if (coverage < 0.62) return 0;
  return average + coverage + phraseBonus + Math.min(patternTokens.length, 5) * 0.18;
}

function significantTokens(text) {
  return String(text || "")
    .split(/\s+/)
    .filter((token) => token.length >= 3 && !ROUTER_STOPWORDS.has(token));
}

function routeVocabulary(routes) {
  const vocabulary = new Set();
  for (const route of routes) {
    for (const pattern of route.questionPatterns || []) {
      for (const token of significantTokens(normalizeText(pattern))) {
        vocabulary.add(token);
      }
    }
  }
  return vocabulary;
}

function correctPromptTokens(normalizedPrompt, vocabulary) {
  return normalizedPrompt
    .split(/\s+/)
    .map((token) => correctedToken(token, vocabulary))
    .join(" ")
    .trim();
}

function correctedToken(token, vocabulary) {
  if (token.length < 5 || vocabulary.has(token)) return token;

  let bestToken = token;
  let bestDistance = Infinity;

  for (const candidate of vocabulary) {
    if (Math.abs(candidate.length - token.length) > 2) continue;

    const distance = isAdjacentTransposition(candidate, token)
      ? 1
      : levenshtein(candidate, token);
    const limit = Math.max(candidate.length, token.length) >= 8 ? 2 : 1;
    if (distance <= limit && distance < bestDistance) {
      bestDistance = distance;
      bestToken = candidate;
    }
  }

  return bestToken;
}

function bestTokenMatch(patternToken, promptTokens) {
  let best = 0;
  for (const promptToken of promptTokens) {
    if (patternToken === promptToken) return 1;
    if (promptToken.includes(patternToken) || patternToken.includes(promptToken)) {
      best = Math.max(best, 0.88);
      continue;
    }
    if (isAdjacentTransposition(patternToken, promptToken)) {
      best = Math.max(best, 0.86);
      continue;
    }
    if (patternToken.length >= 5 && promptToken.length >= 5) {
      const distance = levenshtein(patternToken, promptToken);
      const limit = Math.max(patternToken.length, promptToken.length) >= 8 ? 2 : 1;
      if (distance <= limit) {
        best = Math.max(best, 1 - distance / Math.max(patternToken.length, promptToken.length));
      }
    }
  }
  return best;
}

function isAdjacentTransposition(a, b) {
  if (a.length !== b.length || a.length < 4) return false;

  const mismatchIndexes = [];
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) mismatchIndexes.push(i);
    if (mismatchIndexes.length > 2) return false;
  }

  if (mismatchIndexes.length !== 2) return false;
  const [first, second] = mismatchIndexes;
  return second === first + 1 && a[first] === b[second] && a[second] === b[first];
}

function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;

  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  const current = Array.from({ length: b.length + 1 }, () => 0);

  for (let i = 1; i <= a.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const substitutionCost = a[i - 1] === b[j - 1] ? 0 : 1;
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + substitutionCost
      );
    }
    for (let j = 0; j <= b.length; j += 1) {
      previous[j] = current[j];
    }
  }

  return previous[b.length];
}
