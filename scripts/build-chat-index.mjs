import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const claimMapPath = join(root, "chat", "claim_map.json");
const publicDataDir = join(root, "public", "data");

const claimMap = JSON.parse(await readFile(claimMapPath, "utf8"));
const routes = [
  ...(claimMap.claims || []),
  ...(claimMap.nonClaimRoutes || [])
];

const chunks = routes.flatMap((route) => {
  const support = route.support?.length
    ? route.support
    : [{ doc: claimMap.roadmap?.doc || "docs/SUNDOG_V_CHAT.md", section: route.id, status: "generated" }];

  return support.map((entry, index) => ({
    id: `${route.id}:${index + 1}`,
    routeId: route.id,
    doc: entry.doc,
    href: toHref(entry.doc),
    section: entry.section || route.id,
    tier: route.evidenceTier || "unknown",
    disposition: route.disposition || "allow",
    freshness: freshnessFor(route, entry),
    supportStatus: entry.status || "supporting",
    boundaryTags: route.boundaries || [],
    questionPatterns: route.questionPatterns || [],
    text: [
      route.answerTemplate,
      route.earned,
      route.earnedDetail,
      ...(route.questionPatterns || []),
      ...(route.boundaries || [])
    ].filter(Boolean).join(" ")
  }));
});

const chatIndex = {
  version: "0.1.0",
  status: "phase2_seeded",
  source: "chat/claim_map.json",
  purpose: "Browser-local retrieval index for Ask Sundog claim-boundary inspection. This is not a generative answer source.",
  chunks
};

const evidenceTiers = {
  version: claimMap.version,
  source: "chat/claim_map.json",
  tiers: claimMap.evidenceTiers || []
};

const boundaryRules = {
  version: claimMap.version,
  source: "chat/claim_map.json",
  rules: routes.map((route) => ({
    routeId: route.id,
    disposition: route.disposition || "allow",
    evidenceTier: route.evidenceTier || "unknown",
    boundaries: route.boundaries || [],
    supportDocs: (route.support || []).map((entry) => entry.doc)
  }))
};

await writeJson(join(publicDataDir, "sundog-claim-map.json"), claimMap);
await writeJson(join(publicDataDir, "sundog-chat-index.json"), chatIndex);
await writeJson(join(publicDataDir, "sundog-evidence-tiers.json"), evidenceTiers);
await writeJson(join(publicDataDir, "sundog-boundary-rules.json"), boundaryRules);

console.log(`chat index built: ${chunks.length} chunks`);

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

function toHref(doc) {
  if (!doc) return "/docs/SUNDOG_V_CHAT.md";
  if (doc.startsWith("/")) return doc;
  const normalized = doc.replaceAll("\\", "/");
  if (/^[^/]+\.html$/.test(normalized)) return `/${normalized.slice(0, -5)}`;
  return `/${normalized}`;
}

function freshnessFor(route, entry) {
  if ((entry.status || "").includes("boundary")) return "current_boundary";
  if (route.evidenceTier === "roadmap") return "living_roadmap";
  if (route.evidenceTier === "unsupported") return "current_no_claim";
  return "current_claim_map";
}
