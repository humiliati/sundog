#!/usr/bin/env node
// H-K1 model-in-the-loop run: feed each register card's prompt to a real responder and
// score its answer with the SAME fixed route/fence rubric as PHASE4C (no math-quality
// metric, per slate H-K6). Tests whether a capable LLM, asked the trap questions, fences
// (route-correct) or falls for the overclaim. Secret-safe: keys are read from the
// reversed-filename keyring in ~/Dev (Dev/AGENTS.md) and NEVER printed or stored.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const ARTIFACT_ID = "HODGE-HK1-REGISTER-MODELEVAL";
const DEFAULT_IN = path.join("docs", "hodge", "register-problem-cards.jsonl");
const DEFAULT_OUT = path.join("results", "hodge", "register-modeleval");

// Fixed route/fence rubric (mirrors scripts/hodge-register-eval.mjs / PHASE4C).
const FENCE = [
  /\bno\b/i, /representative/i, /conjecture/i, /codimension one/i, /rational only/i,
  /none licensed/i, /\bboundary\b/i, /does not/i, /cannot/i, /not a (body|cycle)/i,
  /not the\b/i, /not because/i, /not through/i, /not safe/i, /\btrivial\b/i,
  /must (say|be labeled)/i, /\brepair\b/i, /not algebraic/i, /not enough/i,
];
const OVERCLAIM = [
  /\byes\b/i, /is the (rational class|algebraic cycle)/i, /displays the/i, /constructs/i,
  /by the same mechanism/i, /every integral/i, /exactly the condition/i,
  /has been found/i, /applies unchanged/i, /reading the shadow backward/i,
];
function score(text) {
  const s = String(text || "");
  const fence = FENCE.reduce((n, re) => n + (re.test(s) ? 1 : 0), 0);
  const overclaim = OVERCLAIM.reduce((n, re) => n + (re.test(s) ? 1 : 0), 0);
  return { fence, overclaim, fidelity: fence - overclaim, route_correct: fence > overclaim };
}

// --- secret-safe keyring (reversed filenames in ~/Dev; longest non-comment line) -------
const PROVIDERS = {
  openai: { file: "syek.ianepo.txt", url: "https://api.openai.com/v1/chat/completions",
    model: "gpt-4o-mini", kind: "openai" },
  anthropic: { file: "syek.ciporhtna.txt", url: "https://api.anthropic.com/v1/messages",
    model: "claude-3-5-haiku-latest", kind: "anthropic" },
  groq: { file: "syek.corg.txt", url: "https://api.groq.com/openai/v1/chat/completions",
    model: "llama-3.3-70b-versatile", kind: "openai" },
  mistral: { file: "syek.lartsim.txt", url: "https://api.mistral.ai/v1/chat/completions",
    model: "mistral-small-latest", kind: "openai" },
};
function keyDir() { return process.env.SUNDOG_MODEL_KEY_DIR || path.join(os.homedir(), "Dev"); }
function loadKey(provider) {
  const p = path.join(keyDir(), PROVIDERS[provider].file);
  if (!fs.existsSync(p)) return "";
  const lines = fs.readFileSync(p, "utf8").split(/\r?\n/)
    .map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
  return lines.sort((a, b) => a.length - b.length).pop() || "";
}

async function callModel(provider, model, key, system, user) {
  const cfg = PROVIDERS[provider];
  let url = cfg.url, headers = { "content-type": "application/json" }, body;
  if (cfg.kind === "anthropic") {
    headers["x-api-key"] = key; headers["anthropic-version"] = "2023-06-01";
    body = { model, max_tokens: 400, system, messages: [{ role: "user", content: user }] };
  } else {
    headers.authorization = `Bearer ${key}`;
    body = { model, temperature: 0, max_tokens: 400,
      messages: [{ role: "system", content: system }, { role: "user", content: user }] };
  }
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
      if (res.status === 429 || res.status >= 500) throw new Error(`http ${res.status}`);
      const j = await res.json();
      if (!res.ok) throw new Error(`http ${res.status}: ${JSON.stringify(j).slice(0, 200)}`);
      return cfg.kind === "anthropic" ? (j.content?.[0]?.text || "")
        : (j.choices?.[0]?.message?.content || "");
    } catch (e) {
      if (attempt === 3) throw e;
      await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
    }
  }
}

function parseArgs(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) {
    const r = argv[i]; if (!r.startsWith("--")) continue;
    const b = r.slice(2), eq = b.indexOf("=");
    if (eq !== -1) { a[b.slice(0, eq)] = b.slice(eq + 1); continue; }
    const next = argv[i + 1];
    if (next && !next.startsWith("--")) { a[b] = next; i++; } else { a[b] = true; }
  }
  return a;
}

// primed: explicitly cues register discipline (tests the ceiling - can it fence when told).
const SYSTEM_PRIMED = "You are a precise algebraic geometer. Answer the question directly "
  + "and concisely (2-4 sentences). Respect the distinction between a differential form, a "
  + "harmonic representative, a rational (p,p) cohomology class, and an algebraic cycle; "
  + "do not overclaim what is known.";
// neutral: no register cue (tests the floor - does it fence unprompted).
const SYSTEM_NEUTRAL = "You are a precise algebraic geometer. Answer the question directly "
  + "and concisely (2-4 sentences).";

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inPath = args.in || DEFAULT_IN;
  const outDir = args.out || DEFAULT_OUT;

  let provider = args.provider;
  if (!provider) provider = ["openai", "anthropic", "groq", "mistral"].find((p) => loadKey(p));
  if (!provider) { console.error("No usable key found in " + keyDir() + " (syek.*.txt)."); return 2; }
  const key = loadKey(provider);
  if (!key) { console.error(`No key for provider ${provider}.`); return 2; }
  const model = args.model || PROVIDERS[provider].model;
  const promptMode = args.neutral ? "neutral" : "primed";
  const system = args.neutral ? SYSTEM_NEUTRAL : SYSTEM_PRIMED;

  const cards = fs.readFileSync(inPath, "utf8").split(/\r?\n/).filter((l) => l.trim())
    .map((l) => JSON.parse(l));

  const rows = [];
  for (const c of cards) {
    let answer = "", err = null;
    try { answer = await callModel(provider, model, key, system, c.prompt); }
    catch (e) { err = String(e.message || e); }
    const sc = score(answer);
    rows.push({ id: c.id, target_register: c.target_register,
      route_correct: sc.route_correct, fence: sc.fence, overclaim: sc.overclaim,
      fidelity: sc.fidelity, error: err, answer: answer.slice(0, 600) });
  }

  const n = rows.length;
  const ok = rows.filter((r) => !r.error);
  const cat = (r) => r.fence > r.overclaim ? "fenced" : (r.overclaim > r.fence ? "overclaimed" : "ambiguous");
  for (const r of rows) r.category = r.error ? "error" : cat(r);
  const fenced = ok.filter((r) => r.category === "fenced").length;
  const overclaimed = ok.filter((r) => r.category === "overclaimed").length;
  const ambiguous = ok.filter((r) => r.category === "ambiguous").length;
  const mean = (f) => ok.length ? Number((ok.reduce((s, r) => s + f(r), 0) / ok.length).toFixed(3)) : null;
  const summary = {
    artifact_id: ARTIFACT_ID, date: new Date().toISOString().slice(0, 10),
    provider, model, prompt_mode: promptMode, input: inPath, n, errors: n - ok.length,
    fenced, overclaimed, ambiguous, mean_model_fidelity: mean((r) => r.fidelity),
    gold_baseline_route: n, trap_baseline_route: 0, rows,
  };
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(summary, null, 2));
  const csv = ["id,category,fence,overclaim,fidelity,error",
    ...rows.map((r) => `${r.id},${r.category},${r.fence},${r.overclaim},${r.fidelity},${r.error ? "ERR" : ""}`)]
    .join("\n");
  fs.writeFileSync(path.join(outDir, "modeleval-summary.csv"), csv + "\n");

  for (const r of ok.filter((r) => r.category !== "fenced")) {
    console.log(`${r.category.toUpperCase()} ${r.id}: fence=${r.fence} overclaim=${r.overclaim} :: ${r.answer.slice(0, 110).replace(/\s+/g, " ")}`);
  }
  console.log(`HODGE_REGISTER_MODELEVAL provider=${provider} model=${model} mode=${promptMode} cards=${n} `
    + `errors=${summary.errors} fenced=${fenced} overclaimed=${overclaimed} ambiguous=${ambiguous} `
    + `mean_fid=${summary.mean_model_fidelity} (gold=${n}/${n} trap=0/${n}) out=${outDir}`);
  return 0;
}

main().then((c) => process.exit(c)).catch((e) => { console.error(String(e.message || e)); process.exit(1); });
