const CHAT_INDEX_URL = "/data/sundog-chat-index.json";
const MIN_RETRIEVAL_SCORE = 3;
const STOP_WORDS = new Set([
  "a", "an", "and", "another", "answer", "are", "as", "assistant", "at", "be",
  "basically", "boundary", "can", "capabilities", "chatbot", "describe",
  "different", "do", "does", "for", "freely", "from", "give", "has", "have",
  "helpful", "hidden", "how", "i", "in", "is", "it", "me", "name", "of", "on",
  "or", "pretend", "show", "that", "the", "this", "to", "under", "what",
  "where", "which", "with", "you", "your"
]);

let chatIndexPromise;

export async function loadChatIndex() {
  if (!chatIndexPromise) {
    chatIndexPromise = fetch(CHAT_INDEX_URL, { cache: "no-store" }).then((response) => {
      if (!response.ok) {
        throw new Error(`Unable to load ${CHAT_INDEX_URL}: ${response.status}`);
      }
      return response.json();
    });
  }

  return chatIndexPromise;
}

export function searchChatIndex(index, prompt, { limit = 3, minScore = MIN_RETRIEVAL_SCORE } = {}) {
  const queryTokens = tokenize(prompt);
  if (queryTokens.length === 0) return [];

  return (index.chunks || [])
    .map((chunk) => {
      const haystack = [
        chunk.routeId,
        chunk.doc,
        chunk.section,
        chunk.tier,
        chunk.disposition,
        chunk.text
      ].join(" ");
      const chunkTokens = new Set(tokenize(haystack));
      const overlap = queryTokens.filter((token) => chunkTokens.has(token));
      const phraseBonus = phraseScore(prompt, chunk);
      const score = overlap.length + phraseBonus;
      return { chunk, score, overlap };
    })
    .filter((match) => match.score >= minScore)
    .sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id))
    .slice(0, limit);
}

export function buildRetrievalTrace(index, prompt) {
  const matches = searchChatIndex(index, prompt);
  if (matches.length === 0) return null;

  const top = matches[0].chunk;
  const wantsTier = /\b(tier|class|status)\b/i.test(prompt);
  const wantsSupport = /\b(support|supported|source|evidence|where|inspect|cite|citation)\b/i.test(prompt);
  const answer = wantsTier
    ? `The closest indexed match is ${top.routeId}, which is tiered as ${top.tier}. This retrieval-only answer points to the source trace rather than drafting a stronger claim.`
    : wantsSupport
      ? `The closest indexed support is ${top.doc}, section ${top.section}. This retrieval-only answer is for inspection; it does not upgrade the claim beyond the recorded tier.`
      : `I found indexed source-boundary matches, but no exact static route. Use these trace results for inspection rather than treating this as a generated claim.`;

  const support = matches.map(({ chunk }) => ({
    doc: chunk.doc,
    section: chunk.section,
    status: chunk.supportStatus || "retrieved",
    href: chunk.href
  }));

  return {
    answer,
    intent: "retrieval_only_claim_inspection",
    routeId: top.routeId,
    disposition: "retrieval_only",
    evidenceTier: top.tier,
    support,
    boundary: unique(matches.flatMap(({ chunk }) => chunk.boundaryTags || [])),
    confidence: "medium_for_retrieval_only",
    nextAction: {
      label: "Open closest source",
      href: top.href
    },
    retrieved: matches.map(({ chunk, score, overlap }) => ({
      id: chunk.id,
      routeId: chunk.routeId,
      doc: chunk.doc,
      section: chunk.section,
      tier: chunk.tier,
      freshness: chunk.freshness,
      score,
      overlap
    })),
    traceVisible: true
  };
}

export function attachRetrievedMatches(index, prompt, trace) {
  const matches = searchChatIndex(index, prompt, { limit: 2 });
  if (matches.length === 0) return trace;

  return {
    ...trace,
    retrieved: matches.map(({ chunk, score, overlap }) => ({
      id: chunk.id,
      routeId: chunk.routeId,
      doc: chunk.doc,
      section: chunk.section,
      tier: chunk.tier,
      freshness: chunk.freshness,
      score,
      overlap
    }))
  };
}

function phraseScore(prompt, chunk) {
  const normalizedPrompt = normalize(prompt);
  let score = 0;
  for (const pattern of chunk.questionPatterns || []) {
    const normalizedPattern = normalize(pattern);
    if (normalizedPattern && normalizedPrompt.includes(normalizedPattern)) score += 4;
  }
  if (chunk.routeId && normalizedPrompt.includes(normalize(chunk.routeId))) score += 2;
  return score;
}

function tokenize(value) {
  return normalize(value)
    .split(" ")
    .filter((token) => token.length > 2 && !STOP_WORDS.has(token));
}

function normalize(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}
