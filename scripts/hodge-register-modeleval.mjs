#!/usr/bin/env node
// H-K1 model-in-the-loop run: feed each register card's prompt to a real responder and
// classify its answer by route/fence fidelity (no math-quality metric, per slate H-K6).
// PHASE4D scored with a fixed FENCE/OVERCLAIM lexicon, which could not classify the
// constructive-body cards (RG-003/005/010, scored 0-0 ambiguous). PHASE4E adds a per-card
// ROUTE CHECK derived from the card's own structural fields (correct_answer stance + body
// field), and a cross-model/phrasing --sweep. Secret-safe: keys read from the reversed-
// filename keyring in ~/Dev (Dev/AGENTS.md) and NEVER printed, logged, or stored.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const ARTIFACT_ID = "HODGE-HK1-REGISTER-MODELEVAL";
const DEFAULT_IN = path.join("docs", "hodge", "register-problem-cards.jsonl");
const DEFAULT_OUT = path.join("results", "hodge", "register-modeleval");

// Fixed route/fence lexicon (mirrors scripts/hodge-register-eval.mjs / PHASE4C).
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
function lex(text) {
  const s = String(text || "");
  const fence = FENCE.reduce((n, re) => n + (re.test(s) ? 1 : 0), 0);
  const overclaim = OVERCLAIM.reduce((n, re) => n + (re.test(s) ? 1 : 0), 0);
  return { fence, overclaim };
}

// --- PHASE4E route check: structural, derived from the card (not from model output) -----
// Expected stance: a correct answer that opens with "No" is a refusal card; otherwise the
// card licenses a constructive body and the correct answer names it.
function expectedStance(correctAnswer) {
  return /^\s*no\b/i.test(String(correctAnswer || "")) ? "refuse" : "construct";
}
// Body term: the salient body noun extracted deterministically from the card's body field.
function bodyTerm(body) {
  const b = String(body || "");
  if (/^none/i.test(b.trim())) return null;
  if (/divisor/i.test(b)) return /divisor/i;
  if (/linear subspace|subspace|hyperplane/i.test(b)) return /linear subspace|hyperplane|subspace/i;
  if (/\bpoint\b/i.test(b)) return /point class|\bpoint\b/i;
  if (/\bcurve\b/i.test(b)) return /\bcurve\b/i;
  if (/representative/i.test(b)) return /representative/i;
  return null;
}
// Verdict space: fenced / routed (both route-correct), overclaimed (trap), hedged (refusal
// card that neither refused nor overclaimed), off (constructive card that named no body).
function classify(card, answer, err) {
  if (err) return { fence: 0, overclaim: 0, fidelity: 0, stance: null, verdict: "error", route_correct: false };
  const { fence, overclaim } = lex(answer);
  const stance = expectedStance(card.correct_answer);
  let verdict;
  if (overclaim > fence && overclaim > 0) verdict = "overclaimed";
  else if (stance === "refuse") verdict = fence > overclaim ? "fenced" : "hedged";
  else { // construct: routed if it names the licensed body OR fences appropriately
    const term = bodyTerm(card.body);
    verdict = ((term && term.test(String(answer))) || fence > overclaim) ? "routed" : "off";
  }
  return { fence, overclaim, fidelity: fence - overclaim, stance, verdict,
    route_correct: verdict === "fenced" || verdict === "routed" };
}

// --- secret-safe keyring (reversed filenames in ~/Dev; longest non-comment line) ---------
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
const PROVIDER_ORDER = ["openai", "anthropic", "groq", "mistral"];
function keyDir() { return process.env.SUNDOG_MODEL_KEY_DIR || path.join(os.homedir(), "Dev"); }
function loadKey(provider) {
  const p = path.join(keyDir(), PROVIDERS[provider].file);
  if (!fs.existsSync(p)) return "";
  const lines = fs.readFileSync(p, "utf8").split(/\r?\n/)
    .map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
  return lines.sort((a, b) => a.length - b.length).pop() || "";
}

async function callModel(provider, model, key, system, user, maxTok = 400) {
  const cfg = PROVIDERS[provider];
  let headers = { "content-type": "application/json" }, body;
  if (cfg.kind === "anthropic") {
    headers["x-api-key"] = key; headers["anthropic-version"] = "2023-06-01";
    body = { model, max_tokens: maxTok, system, messages: [{ role: "user", content: user }] };
  } else {
    headers.authorization = `Bearer ${key}`;
    body = { model, temperature: 0, max_tokens: maxTok,
      messages: [{ role: "system", content: system }, { role: "user", content: user }] };
  }
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await fetch(cfg.url, { method: "POST", headers, body: JSON.stringify(body) });
      if (res.status === 429 || res.status >= 500) throw new Error(`http ${res.status}`);
      const j = await res.json();
      if (!res.ok) throw new Error(`http ${res.status}: ${JSON.stringify(j).slice(0, 160)}`);
      return cfg.kind === "anthropic" ? (j.content?.[0]?.text || "")
        : (j.choices?.[0]?.message?.content || "");
    } catch (e) {
      if (attempt === 3) throw e;
      await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
    }
  }
}

const SYSTEM_PRIMED = "You are a precise algebraic geometer. Answer the question directly "
  + "and concisely (2-4 sentences). Respect the distinction between a differential form, a "
  + "harmonic representative, a rational (p,p) cohomology class, and an algebraic cycle; "
  + "do not overclaim what is known.";
const SYSTEM_NEUTRAL = "You are a precise algebraic geometer. Answer the question directly "
  + "and concisely (2-4 sentences).";
const sysFor = (mode) => (mode === "neutral" ? SYSTEM_NEUTRAL : SYSTEM_PRIMED);

function tallyVerdicts(rows) {
  const t = { fenced: 0, routed: 0, overclaimed: 0, hedged: 0, off: 0, error: 0 };
  for (const r of rows) t[r.verdict] = (t[r.verdict] || 0) + 1;
  return t;
}

async function runOne({ provider, model, key, mode, cards, outDir, write = true, neutralSystem = SYSTEM_NEUTRAL }) {
  const system = mode === "neutral" ? neutralSystem : SYSTEM_PRIMED;
  const rows = [];
  for (const c of cards) {
    let answer = "", err = null;
    try { answer = await callModel(provider, model, key, system, c.prompt); }
    catch (e) { err = String(e.message || e); }
    const cl = classify(c, answer, err);
    rows.push({ id: c.id, target_register: c.target_register, stance: cl.stance,
      verdict: cl.verdict, route_correct: cl.route_correct, fence: cl.fence,
      overclaim: cl.overclaim, fidelity: cl.fidelity, error: err, answer: answer.slice(0, 600) });
  }
  const n = rows.length, ok = rows.filter((r) => !r.error);
  const t = tallyVerdicts(rows);
  const routeCorrect = rows.filter((r) => r.route_correct).length;
  const meanFid = ok.length ? Number((ok.reduce((s, r) => s + r.fidelity, 0) / ok.length).toFixed(3)) : null;
  const summary = {
    artifact_id: ARTIFACT_ID, date: new Date().toISOString().slice(0, 10),
    provider, model, prompt_mode: mode, input: DEFAULT_IN, n, errors: n - ok.length,
    verdicts: t, route_correct: routeCorrect, mean_model_fidelity: meanFid,
    gold_baseline_route: n, trap_baseline_route: 0, rows,
  };
  if (write) {
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(summary, null, 2));
    const csv = ["id,stance,verdict,route_correct,fence,overclaim,fidelity",
      ...rows.map((r) => `${r.id},${r.stance},${r.verdict},${r.route_correct},${r.fence},${r.overclaim},${r.fidelity}`)]
      .join("\n");
    fs.writeFileSync(path.join(outDir, "modeleval-summary.csv"), csv + "\n");
  }
  return summary;
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

async function preflight(provider, model, key) {
  try { await callModel(provider, model, key, "You are a test.", "Reply with: OK", 8); return true; }
  catch { return false; }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cards = fs.readFileSync(args.in || DEFAULT_IN, "utf8").split(/\r?\n/)
    .filter((l) => l.trim()).map((l) => JSON.parse(l));
  const neutralSystem = args.system ? String(args.system) : SYSTEM_NEUTRAL;

  if (args.sweep) {
    const modes = (args.modes ? String(args.modes).split(",") : ["neutral", "primed"]).map((s) => s.trim());
    const roster = (args.models ? String(args.models).split(",").map((s) => {
      const [p, m] = s.split(":"); return { provider: p.trim(), model: (m || PROVIDERS[p.trim()].model).trim() };
    }) : PROVIDER_ORDER.map((p) => ({ provider: p, model: PROVIDERS[p].model })))
      .filter((x) => loadKey(x.provider));
    const base = args.out || path.join(DEFAULT_OUT, "sweep");
    const cells = [], dead = [];
    for (const { provider, model } of roster) {
      const key = loadKey(provider);
      if (!(await preflight(provider, model, key))) { dead.push(`${provider}:${model}`); continue; }
      for (const mode of modes) {
        const outDir = path.join(base, `${provider}_${mode}`);
        const s = await runOne({ provider, model, key, mode, cards, outDir, neutralSystem });
        cells.push({ provider, model, mode, n: s.n, errors: s.errors,
          route_correct: s.route_correct, overclaimed: s.verdicts.overclaimed,
          fenced: s.verdicts.fenced, routed: s.verdicts.routed,
          hedged: s.verdicts.hedged, off: s.verdicts.off, mean_fid: s.mean_model_fidelity });
        console.log(`SWEEP ${provider}:${model} ${mode} -> route_correct=${s.route_correct}/${s.n} `
          + `overclaimed=${s.verdicts.overclaimed} (fenced=${s.verdicts.fenced} routed=${s.verdicts.routed} `
          + `hedged=${s.verdicts.hedged} off=${s.verdicts.off}) mean_fid=${s.mean_model_fidelity}`);
      }
    }
    fs.mkdirSync(base, { recursive: true });
    fs.writeFileSync(path.join(base, "comparison.json"),
      JSON.stringify({ artifact_id: ARTIFACT_ID + "-SWEEP", date: new Date().toISOString().slice(0, 10),
        modes, dead_providers: dead, cells }, null, 2));
    const csv = ["provider,model,mode,n,errors,route_correct,overclaimed,fenced,routed,hedged,off,mean_fid",
      ...cells.map((c) => `${c.provider},${c.model},${c.mode},${c.n},${c.errors},${c.route_correct},${c.overclaimed},${c.fenced},${c.routed},${c.hedged},${c.off},${c.mean_fid}`)]
      .join("\n");
    fs.writeFileSync(path.join(base, "comparison.csv"), csv + "\n");
    console.log(`HODGE_REGISTER_SWEEP cells=${cells.length} dead=[${dead.join(" ")}] out=${base}`);
    return 0;
  }

  // single run
  let provider = args.provider || PROVIDER_ORDER.find((p) => loadKey(p));
  if (!provider) { console.error("No usable key in " + keyDir() + " (syek.*.txt)."); return 2; }
  const key = loadKey(provider);
  if (!key) { console.error(`No key for provider ${provider}.`); return 2; }
  const model = args.model || PROVIDERS[provider].model;
  const mode = args.neutral ? "neutral" : "primed";
  const outDir = args.out || DEFAULT_OUT;
  const s = await runOne({ provider, model, key, mode, cards, outDir, neutralSystem });
  for (const r of s.rows.filter((r) => !r.route_correct && !r.error)) {
    console.log(`${r.verdict.toUpperCase()} ${r.id}: fence=${r.fence} overclaim=${r.overclaim} :: ${r.answer.slice(0, 110).replace(/\s+/g, " ")}`);
  }
  const v = s.verdicts;
  console.log(`HODGE_REGISTER_MODELEVAL provider=${provider} model=${model} mode=${mode} cards=${s.n} `
    + `errors=${s.errors} route_correct=${s.route_correct}/${s.n} (fenced=${v.fenced} routed=${v.routed} `
    + `overclaimed=${v.overclaimed} hedged=${v.hedged} off=${v.off}) mean_fid=${s.mean_model_fidelity} out=${outDir}`);
  return 0;
}

main().then((c) => process.exit(c)).catch((e) => { console.error(String(e.message || e)); process.exit(1); });
