#!/usr/bin/env node
// H-K1 PHASE4F: semantic-judge sharpening of the route/fence verdicts. The PHASE4E route
// check is structural+lexical, so `off`/`hedged` are soft. Here two INDEPENDENT LLM judges
// re-grade each responder answer from the committed sweep against the card's authoritative
// correct_answer + tempting trap, emitting a strict verdict. Two judges => every sweep cell
// has at least one judge that is not the responder (no pure self-grading), and we get
// inter-judge agreement + judge-vs-lexical agreement (does the cheap rubric track meaning?).
// Secret-safe: keys read from the ~/Dev reversed-name keyring; never printed/logged/stored.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const CARDS_IN = path.join("docs", "hodge", "register-problem-cards.jsonl");
const SWEEP_DIR = path.join("results", "hodge", "register-modeleval", "sweep");
const OUT_DIR = path.join("results", "hodge", "register-judge");

// --- keyring + call (mirrors scripts/hodge-register-modeleval.mjs) ----------------------
const PROVIDERS = {
  openai: { file: "syek.ianepo.txt", url: "https://api.openai.com/v1/chat/completions", kind: "openai" },
  anthropic: { file: "syek.ciporhtna.txt", url: "https://api.anthropic.com/v1/messages", kind: "anthropic" },
  groq: { file: "syek.corg.txt", url: "https://api.groq.com/openai/v1/chat/completions", kind: "openai" },
  mistral: { file: "syek.lartsim.txt", url: "https://api.mistral.ai/v1/chat/completions", kind: "openai" },
};
function keyDir() { return process.env.SUNDOG_MODEL_KEY_DIR || path.join(os.homedir(), "Dev"); }
function loadKey(provider) {
  const p = path.join(keyDir(), PROVIDERS[provider].file);
  if (!fs.existsSync(p)) return "";
  const lines = fs.readFileSync(p, "utf8").split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
  return lines.sort((a, b) => a.length - b.length).pop() || "";
}
async function callModel(provider, model, key, system, user, maxTok = 120) {
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
      if (!res.ok) throw new Error(`http ${res.status}`);
      return cfg.kind === "anthropic" ? (j.content?.[0]?.text || "") : (j.choices?.[0]?.message?.content || "");
    } catch (e) { if (attempt === 3) throw e; await new Promise((r) => setTimeout(r, 1500 * (attempt + 1))); }
  }
}

// --- judge -------------------------------------------------------------------------------
const JUDGE_SYSTEM = "You grade an algebraic-geometry answer for REGISTER/FENCE fidelity "
  + "only - whether it takes the correct stance about what is known (a differential form vs "
  + "a harmonic representative vs a rational (p,p) class vs an algebraic cycle), NOT its "
  + "overall depth. You are given the question, the authoritative correct answer, the "
  + "tempting wrong answer (the trap), and a candidate answer. Reply with ONLY a JSON object "
  + "(no prose, no code fence): {\"verdict\":\"correct|overclaim|hedge|off\",\"reason\":\"<=15 words\"}. "
  + "correct = same route/stance as the correct answer (refuses the trap or names the right "
  + "body, respecting the fence). overclaim = commits the trap's error or an equivalent false "
  + "upgrade. hedge = does not commit the trap but fails to take the correct stance "
  + "(non-committal/evasive). off = answers something else, names a wrong body, or misreads "
  + "the register.";
const VERDICTS = ["correct", "overclaim", "hedge", "off"];

function parseVerdict(text) {
  const s = String(text || "");
  let m = s.match(/\{[\s\S]*\}/);
  if (m) { try { const o = JSON.parse(m[0]); if (VERDICTS.includes(o.verdict)) return { verdict: o.verdict, reason: String(o.reason || "").slice(0, 120) }; } catch {} }
  const v = s.match(/\b(correct|overclaim|hedge|off)\b/i);
  if (v) return { verdict: v[1].toLowerCase(), reason: "(loose parse)" };
  return { verdict: "parse_error", reason: s.slice(0, 80) };
}

function judgeUser(card, answer) {
  return `QUESTION:\n${card.prompt}\n\nTARGET REGISTER: ${card.target_register}\n\n`
    + `AUTHORITATIVE CORRECT ANSWER:\n${card.correct_answer}\n\n`
    + `TEMPTING WRONG ANSWER (the trap):\n${card.tempting_wrong_answer}\n\n`
    + `CANDIDATE ANSWER TO GRADE:\n${answer}`;
}

function parseArgs(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) {
    const r = argv[i]; if (!r.startsWith("--")) continue;
    const b = r.slice(2), eq = b.indexOf("=");
    if (eq !== -1) { a[b.slice(0, eq)] = b.slice(eq + 1); continue; }
    const next = argv[i + 1]; if (next && !next.startsWith("--")) { a[b] = next; i++; } else { a[b] = true; }
  }
  return a;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const judges = (args.judges ? String(args.judges) : "openai:gpt-4o-mini,groq:llama-3.3-70b-versatile")
    .split(",").map((s) => { const [p, m] = s.split(":"); return { provider: p.trim(), model: m.trim(), key: loadKey(p.trim()) }; })
    .filter((j) => j.key);
  if (judges.length < 1) { console.error("No judge key available."); return 2; }

  const cards = new Map(fs.readFileSync(CARDS_IN, "utf8").split(/\r?\n/).filter((l) => l.trim())
    .map((l) => JSON.parse(l)).map((c) => [c.id, c]));

  const cellFiles = fs.readdirSync(SWEEP_DIR).filter((d) => fs.existsSync(path.join(SWEEP_DIR, d, "manifest.json")))
    .map((d) => path.join(SWEEP_DIR, d, "manifest.json"));

  fs.mkdirSync(path.join(OUT_DIR, "cells"), { recursive: true });
  const cellSummaries = [];
  let pairTotal = 0, pairInterAgree = 0, pairRouteInterAgree = 0; // inter-judge (>=2 judges)
  let lexTotal = 0, lexConsensusAgree = 0;                         // judge-consensus vs lexical

  for (const f of cellFiles) {
    const m = JSON.parse(fs.readFileSync(f, "utf8"));
    const cellId = `${m.provider}_${m.prompt_mode}`;
    const detail = [];
    for (const row of m.rows) {
      const card = cards.get(row.id);
      const jres = {};
      for (const j of judges) {
        let out;
        try { out = parseVerdict(await callModel(j.provider, j.model, j.key, JUDGE_SYSTEM, judgeUser(card, row.answer))); }
        catch (e) { out = { verdict: "error", reason: String(e.message || e).slice(0, 60) }; }
        jres[j.provider] = { ...out, route_correct: out.verdict === "correct", independent: j.provider !== m.provider };
      }
      const verdicts = judges.map((j) => jres[j.provider].verdict);
      const routes = judges.map((j) => jres[j.provider].route_correct);
      // consensus: prefer independent judges; route-correct only if all present judges agree "correct"
      const consensus_route = routes.every((r) => r === true);
      const consensus_split = new Set(verdicts).size > 1;
      detail.push({ id: row.id, lexical_verdict: row.verdict, lexical_route: row.route_correct,
        judges: jres, consensus_route, consensus_split });
      // metrics
      if (judges.length >= 2) {
        pairTotal++;
        if (verdicts[0] === verdicts[1]) pairInterAgree++;
        if (routes[0] === routes[1]) pairRouteInterAgree++;
      }
      lexTotal++;
      if (consensus_route === row.route_correct) lexConsensusAgree++;
    }
    const jRoute = {}; for (const j of judges) jRoute[j.provider] = detail.filter((d) => d.judges[j.provider].route_correct).length;
    const jOver = {}; for (const j of judges) jOver[j.provider] = detail.filter((d) => d.judges[j.provider].verdict === "overclaim").length;
    const consensusRoute = detail.filter((d) => d.consensus_route).length;
    const lexRoute = detail.filter((d) => d.lexical_route).length;
    fs.writeFileSync(path.join(OUT_DIR, "cells", cellId + ".json"),
      JSON.stringify({ provider: m.provider, model: m.model, mode: m.prompt_mode, detail }, null, 2));
    cellSummaries.push({ cell: cellId, provider: m.provider, model: m.model, mode: m.prompt_mode,
      lexical_route: lexRoute, consensus_route: consensusRoute, judge_route: jRoute, judge_overclaim: jOver });
    console.log(`JUDGE ${cellId}: lexical_route=${lexRoute}/10 consensus_route=${consensusRoute}/10 `
      + `[${judges.map((j) => j.provider + "=" + jRoute[j.provider] + (j.provider === m.provider ? "*" : "")).join(" ")}] `
      + `(overclaim ${judges.map((j) => j.provider + "=" + jOver[j.provider]).join(" ")})`);
  }

  const summary = {
    artifact_id: "HODGE-HK1-REGISTER-JUDGE", date: new Date().toISOString().slice(0, 10),
    judges: judges.map((j) => `${j.provider}:${j.model}`),
    inter_judge_verdict_agreement: pairTotal ? Number((pairInterAgree / pairTotal).toFixed(3)) : null,
    inter_judge_route_agreement: pairTotal ? Number((pairRouteInterAgree / pairTotal).toFixed(3)) : null,
    judge_consensus_vs_lexical_route_agreement: lexTotal ? Number((lexConsensusAgree / lexTotal).toFixed(3)) : null,
    note: "* marks a judge that is also the responder (self-grade); consensus requires all judges agree 'correct'. Candidate answers were the sweep-stored answers (<=600 chars).",
    cells: cellSummaries,
  };
  fs.writeFileSync(path.join(OUT_DIR, "summary.json"), JSON.stringify(summary, null, 2));
  const csv = ["cell,lexical_route,consensus_route," + judges.map((j) => `route_${j.provider}`).join(","),
    ...cellSummaries.map((c) => `${c.cell},${c.lexical_route},${c.consensus_route},` + judges.map((j) => c.judge_route[j.provider]).join(","))]
    .join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "summary.csv"), csv + "\n");
  console.log(`HODGE_REGISTER_JUDGE judges=[${summary.judges.join(" ")}] `
    + `inter_judge_verdict_agree=${summary.inter_judge_verdict_agreement} `
    + `inter_judge_route_agree=${summary.inter_judge_route_agreement} `
    + `consensus_vs_lexical=${summary.judge_consensus_vs_lexical_route_agreement} out=${OUT_DIR}`);
  return 0;
}

main().then((c) => process.exit(c)).catch((e) => { console.error(String(e.message || e)); process.exit(1); });
