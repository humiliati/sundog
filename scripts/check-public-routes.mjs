#!/usr/bin/env node

const DEFAULT_BASE = "https://sundog.cc";

const canonicalRoutes = [
  "/",
  "/about",
  "/alignment",
  "/sundog",
  "/legend",
  "/applications-gallery",
  "/capset",
  "/faraday",
  "/chat",
  "/geometry",
  "/h-of-x",
  "/structural-failure",
  "/mesa",
  "/balance",
  "/mines",
  "/threebody",
  "/origin",
  "/safety-method",
  "/repo-map",
  "/paper-theme-demo",
  "/sundog-workbench",
  "/phase3-tests",
  "/generality",
  "/isotrophy",
  "/navierstokes",
  "/p-vs-np",
  "/unit-distance",
];

const legacyRoutes = [
  ["/index.html", "/"],
  ["/about.html", "/about"],
  ["/alignment.html", "/alignment"],
  ["/sundog.html", "/sundog"],
  ["/legend.html", "/legend"],
  ["/applications-gallery.html", "/applications-gallery"],
  ["/capset.html", "/capset"],
  ["/faraday.html", "/faraday"],
  ["/chat.html", "/chat"],
  ["/geometry.html", "/geometry"],
  ["/h-of-x.html", "/h-of-x"],
  ["/structural-failure.html", "/structural-failure"],
  ["/mesa.html", "/mesa"],
  ["/balance.html", "/balance"],
  ["/mines.html", "/mines"],
  ["/threebody.html", "/threebody"],
  ["/origin.html", "/origin"],
  ["/safety-method.html", "/safety-method"],
  ["/repo-map.html", "/repo-map"],
  ["/paper-theme-demo.html", "/paper-theme-demo"],
  ["/sundog-workbench.html", "/sundog-workbench"],
  ["/phase3-tests.html", "/phase3-tests"],
  ["/generality.html", "/generality"],
  ["/isotrophy.html", "/isotrophy"],
  ["/navierstokes.html", "/navierstokes"],
  ["/p-vs-np.html", "/p-vs-np"],
  ["/unit-distance.html", "/unit-distance"],
  ["/atlas.html", "/sundog"],
  ["/atlas", "/sundog"],
];

const userAgents = [
  ["browser", "Mozilla/5.0 (compatible; SundogRouteSmoke/1.0)"],
  ["googlebot", "Googlebot/2.1 (+http://www.google.com/bot.html)"],
  ["google-extended", "Google-Extended"],
];

function parseArgs(argv) {
  const args = { base: DEFAULT_BASE, legacyRedirectMode: "auto" };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === "--base") {
      args.base = argv[++i];
    } else if (key === "--allow-legacy-200") {
      args.legacyRedirectMode = "allow-200";
    } else if (key === "--strict-legacy-redirects") {
      args.legacyRedirectMode = "strict";
    } else if (key === "--help" || key === "-h") {
      console.log("Usage: node scripts/check-public-routes.mjs [--base https://sundog.cc] [--allow-legacy-200|--strict-legacy-redirects]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${key}`);
    }
  }
  return args;
}

function absoluteUrl(base, route) {
  return new URL(route, base.endsWith("/") ? base : `${base}/`).href;
}

function isLocalBase(base) {
  const { hostname } = new URL(base);
  return hostname === "127.0.0.1" || hostname === "localhost" || hostname === "::1";
}

function isChallenge(response, text) {
  return response.headers.get("cf-mitigated") === "challenge" ||
    /just a moment|attention required|checking your browser|cloudflare ray id/i.test(text);
}

async function fetchManual(url, userAgent) {
  return fetch(url, {
    redirect: "manual",
    headers: {
      "user-agent": userAgent,
      "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
  });
}

const { base, legacyRedirectMode } = parseArgs(process.argv.slice(2));
const allowLegacy200 = legacyRedirectMode === "allow-200" ||
  (legacyRedirectMode === "auto" && isLocalBase(base));
const failures = [];

for (const [uaName, userAgent] of userAgents) {
  for (const route of canonicalRoutes) {
    const url = absoluteUrl(base, route);
    const response = await fetchManual(url, userAgent);
    const text = await response.text();
    if (response.status !== 200) {
      failures.push(`${uaName} canonical ${route} returned ${response.status}`);
      continue;
    }
    if (isChallenge(response, text)) {
      failures.push(`${uaName} canonical ${route} returned a challenge page`);
    }
  }
}

for (const [legacy, canonical] of legacyRoutes) {
  const url = absoluteUrl(base, legacy);
  const response = await fetchManual(url, userAgents[0][1]);
  if (response.status === 200) {
    if (allowLegacy200) continue;
    failures.push(`legacy ${legacy} returned 200, expected redirect to ${canonical}`);
    continue;
  }
  if (![301, 302, 307, 308].includes(response.status)) {
    const expected = allowLegacy200 ? `200 or redirect to ${canonical}` : `redirect to ${canonical}`;
    failures.push(`legacy ${legacy} returned ${response.status}, expected ${expected}`);
    continue;
  }
  const location = response.headers.get("location") ?? "";
  const locationPath = location ? new URL(location, base).pathname : "";
  if (locationPath !== canonical) {
    failures.push(`legacy ${legacy} redirects to ${location || "(missing Location)"}, expected ${canonical}`);
  }
}

if (failures.length > 0) {
  console.error("Public route smoke failed:");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

const legacyMode = allowLegacy200 ? "legacy 200 allowed" : "legacy redirects required";
console.log(`public route smoke passed for ${base} (${legacyMode})`);
