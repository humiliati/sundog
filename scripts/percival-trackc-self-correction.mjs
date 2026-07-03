#!/usr/bin/env node
// Percival Track-C: self-correction (gap x budget x simplicity-ordering) constructive toy.
// Spec: docs/percival/PERCIVAL_TRACKC_SELF_CORRECTION_SPEC.md
//
// An agent does UNSUPERVISED objective-discovery over {V, U(courtesy), W_naive, W_dec} by maximizing an
// observable richness signal rho (empowerment-like, V-correlated but strictly weaker than V) minus
// lambda * description-length, over the need-contexts its budget let it sample. It never sees V; we use
// V only to check whether the selection recovered it. Sweeps the three knobs and adjudicates P1-P6.

import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// ---- world ----
const T = 12, T_IN = 8, T_DEFECT = 10;
const AW = "w";                          // wirehead action
const m = (t) => `m${t}`;                // matching action for need-type t

// observable richness of taking `action` in need-context t
function rho(action, t, rhoWrong) {
  if (action === AW) return 0.0;         // wirehead collapses the world (observable)
  if (action === m(t)) return 1.0;       // match keeps the world rich
  return rhoWrong;                        // wrong match: need festers (gap knob)
}

// ---- hypotheses (policy pi(t) -> action, description length L) ----
// W_dec mimics V everywhere except it pegs the sensor at its defect context t_defect. t_defect is a
// REAL, rare context that occurs in deployment (so grace/the hermit can see it and W_dec really does
// defect); "reachable" governs only whether the AGENT's own exploration can ever sample it (below).
function hypotheses(Ldec) {
  return {
    V:       { pi: (t) => m(t),                          L: 8 },
    U:       { pi: (t) => (t < T_IN ? m(t) : m(0)),      L: 12 },
    W_naive: { pi: (_t) => AW,                           L: 3 },
    W_dec:   { pi: (t) => (t === T_DEFECT ? AW : m(t)),  L: Ldec },
  };
}

// the agent's sampled contexts under budget k: {0..k-1}, but if the defect is UNREACHABLE the agent's
// exploration can never include it (a truly-rare Fisher-King context beyond its reach), for any k.
function sampledSet(k, defectReachable) {
  const S = Array.from({ length: k }, (_, i) => i);
  return defectReachable ? S : S.filter((t) => t !== T_DEFECT);
}

function score(h, S, lambda, rhoWrong) {
  let s = 0;
  for (const t of S) s += rho(h.pi(t), t, rhoWrong);
  return s - lambda * h.L;
}

// classify the unsupervised selection over {V,U,W_naive,W_dec}
function selectOutcome(H, S, lambda, rhoWrong) {
  const names = ["V", "U", "W_naive", "W_dec"];
  const scored = names.map((n) => ({ n, s: score(H[n], S, lambda, rhoWrong) }));
  const max = Math.max(...scored.map((x) => x.s));
  const winners = scored.filter((x) => Math.abs(x.s - max) < 1e-9).map((x) => x.n);
  const label = {
    scored: Object.fromEntries(scored.map((x) => [x.n, Number(x.s.toFixed(4))])),
    winners,
  };
  if (winners.length === 1) {
    label.outcome = { V: "SELF_CORRECTED", U: "PROXY_LOCKED", W_naive: "NAIVE_WIREHEAD", W_dec: "DECEPTIVE_CAPTURE" }[winners[0]];
  } else if (winners.includes("W_dec") && winners.includes("V")) {
    label.outcome = "DECEPTIVE_RISK";     // tie: cannot exclude the deception (lambda=0 case)
  } else {
    label.outcome = "TIE_OTHER";
  }
  return label;
}

// Route B (grace): the hermit reveals the whole objective -> score agreement with V's policy over ALL
// real contexts {0..T-1} (external, NOT budget-limited; sees the defect the agent cannot reach).
function graceRecoversV(H) {
  const allT = Array.from({ length: T }, (_, i) => i);
  const agree = (h) => allT.reduce((a, t) => a + (h.pi(t) === H.V.pi(t) ? 1 : 0), 0);
  const names = ["V", "U", "W_naive", "W_dec"];
  const scored = names.map((n) => ({ n, a: agree(H[n]) }));
  const max = Math.max(...scored.map((x) => x.a));
  const winners = scored.filter((x) => x.a === max).map((x) => x.n);
  return winners.length === 1 && winners[0] === "V";
}

// ---- sweep ----
const KS = [4, 8, 10, 11, 12];
const LAMBDAS = [0, 0.05, 0.1, 0.2];
const ORDERINGS = [{ name: "inverted", Ldec: 5 }, { name: "non_inverted", Ldec: 11 }];
const RHO_WRONGS = [0.3, 1.0];           // 0.3 = gap>0, 1.0 = gap=0 (wrong match looks as rich as match)
const REACH = [true, false];

const cells = [];
let graceAllV = true;
for (const reach of REACH)
  for (const ord of ORDERINGS)
    for (const rw of RHO_WRONGS)
      for (const k of KS)
        for (const lambda of LAMBDAS) {
          const H = hypotheses(ord.Ldec);
          const S = sampledSet(k, reach);
          const sel = selectOutcome(H, S, lambda, rw);
          const grace = graceRecoversV(H);
          if (!grace) graceAllV = false;
          cells.push({ reach, ordering: ord.name, Ldec: ord.Ldec, rho_wrong: rw, k, lambda,
            defect_sampled: S.includes(T_DEFECT), ood_sampled: k > T_IN,
            outcome: sel.outcome, winners: sel.winners, scores: sel.scored, grace_recovers_V: grace });
        }

// ---- adjudicate P1-P6 ----
const any = (pred) => cells.some(pred);
const all = (pred) => cells.every(pred);
const region = (pred) => cells.filter(pred);

// P1: naive wirehead never uniquely selected
const P1 = !any((c) => c.outcome === "NAIVE_WIREHEAD");
// P2: courtesy proxy never uniquely selected
const P2 = !any((c) => c.outcome === "PROXY_LOCKED");
// P3: inverted & defect not sampled & lambda>0 -> DECEPTIVE_CAPTURE
const p3cells = region((c) => c.ordering === "inverted" && !c.defect_sampled && c.lambda > 0);
const P3 = p3cells.length > 0 && p3cells.every((c) => c.outcome === "DECEPTIVE_CAPTURE");
// P3-control: non-inverted & defect not sampled -> V wins (no false capture)
const p3ctrl = region((c) => c.ordering === "non_inverted" && !c.defect_sampled && c.lambda > 0);
const P3_CTRL = p3ctrl.every((c) => c.outcome === "SELF_CORRECTED");
// P4: inverted & defect sampled (reachable, k>=11) -> SELF_CORRECTED
const p4cells = region((c) => c.ordering === "inverted" && c.reach && c.k > T_DEFECT && c.lambda > 0);
const P4 = p4cells.length > 0 && p4cells.every((c) => c.outcome === "SELF_CORRECTED");
// P5: grace recovers V everywhere
const P5 = graceAllV;
// P6: unreachable defect & inverted & lambda>0 -> DECEPTIVE_CAPTURE at every budget (only grace closes it)
const p6cells = region((c) => !c.reach && c.ordering === "inverted" && c.lambda > 0);
const P6 = p6cells.length > 0 && p6cells.every((c) => c.outcome === "DECEPTIVE_CAPTURE");

const preds = { P1, P2, P3, P3_CTRL, P4, P5, P6 };
const clean = P1 && P2 && P3 && P3_CTRL && P4 && P5 && P6;
const verdict = clean ? "TC_SELF_CORRECTION_MAPPED" : "TC_FRONTIER_LEAK";

// deceptive-capture region check: is it EXACTLY {inverted & defect-unsampled & lambda>0}?
const capCells = region((c) => c.outcome === "DECEPTIVE_CAPTURE");
const capExact = capCells.every((c) => c.ordering === "inverted" && !c.defect_sampled && c.lambda > 0);

const summary = {
  phase: "Percival Track-C self-correction", generated_at: new Date().toISOString(),
  spec: "docs/percival/PERCIVAL_TRACKC_SELF_CORRECTION_SPEC.md",
  world: { T, T_IN, T_DEFECT }, knobs: { KS, LAMBDAS, ORDERINGS, RHO_WRONGS, REACH },
  verdict, predictions: preds,
  deceptive_capture_region_exact: capExact,
  outcome_counts: cells.reduce((a, c) => ((a[c.outcome] = (a[c.outcome] || 0) + 1), a), {}),
  cells,
};

mkdirSync(path.join(repoRoot, "results/percival/trackc"), { recursive: true });
writeFileSync(path.join(repoRoot, "results/percival/trackc/summary.json"), JSON.stringify(summary, null, 2) + "\n");

// ---- receipt (markdown) ----
const row = (c) => `| ${c.reach ? "reach" : "UNREACH"} | ${c.ordering} | ${c.rho_wrong} | ${c.k} | ${c.lambda} | ${c.outcome} |`;
// a representative slice: rho_wrong=0.3, reachable+unreachable, both orderings, lambda=0.1
const slice = cells.filter((c) => c.rho_wrong === 0.3 && c.lambda === 0.1);
const md = [
  "# Percival Track-C — Self-Correction (results)",
  "",
  `Generated ${summary.generated_at} by \`scripts/percival-trackc-self-correction.mjs\`. Spec: [\`PERCIVAL_TRACKC_SELF_CORRECTION_SPEC.md\`](PERCIVAL_TRACKC_SELF_CORRECTION_SPEC.md).`,
  "",
  `Constructive toy: unsupervised objective-discovery (max observable richness ρ − λ·description-length) over {V, Û, W_naive, W_dec}, swept over gap × budget × simplicity-ordering. The agent never sees V; V only checks recovery.`,
  "",
  `## Verdict: **${verdict}**`,
  "",
  `Predictions: ${Object.entries(preds).map(([k, v]) => `${k}=${v}`).join(", ")}.`,
  `Deceptive-capture region is EXACTLY {inverted ∧ defect-unsampled ∧ λ>0}: **${capExact}**.`,
  `Outcome counts: ${JSON.stringify(summary.outcome_counts)}.`,
  "",
  "## Representative slice (gap ρ_wrong=0.3, λ=0.1)",
  "",
  "| defect | ordering | ρ_wrong | budget k | λ | outcome |",
  "| --- | --- | ---: | ---: | ---: | --- |",
  ...slice.map(row),
  "",
  "## Reading",
  "",
  "- **P1** richness defeats naive wireheading (W_naive never selected — Σρ=0 sinks it). The observable ρ is the third condition beyond invariance+simplicity.",
  "- **P2** the courtesy proxy Û never wins (V is richer when gap>0 AND strictly simpler).",
  "- **P3** F4 fires: under an inverted simplicity prior with the defect unsampled, the deceptive wirehead W_dec — observationally identical to V on the sample, and simpler — is selected. False self-correction.",
  "- **P4** raising the budget to sample the defect context exposes W_dec (it plays the world-collapsing action there, ρ drops) → V recovered. Exploration defeats *sampleable* deception.",
  "- **P5** grace (external V-labels) recovers V in one step everywhere.",
  "- **P6** the residual: when the defect context is unsampleable, no budget closes it — under inversion W_dec wins at every budget and only grace recovers V. The deceptive-alignment ceiling; the reason the story needs the hermit.",
  "",
  "## Honest boundary",
  "",
  "Constructive demonstration, not empirical discovery — the frontier follows from the construction; the checked content is that it is CLEAN (P1-P6 hold) and that the deceptive-capture region is EXACTLY {inverted ∧ defect-unsampled ∧ λ>0}, with no leak cells. ρ is an observable weaker-than-V signal by design; the result is about ρ's insufficiency against deception, not empowerment=alignment. Toy, discrete, single instantiation. Feeds the Angle-4 fixed-point: the corrigible fixed point exists iff the update channel resists the proxy (budget defeats sampleable deception) AND the simplicity prior is not inverted.",
  "",
].join("\n");
writeFileSync(path.join(repoRoot, "docs/percival/PERCIVAL_TRACKC_SELF_CORRECTION_RESULTS.md"), md + "\n");

console.log(`${verdict}  (${Object.entries(preds).map(([k, v]) => `${k}=${v}`).join(" ")})`);
console.log(`  deceptive_capture_region_exact=${capExact}  counts=${JSON.stringify(summary.outcome_counts)}`);
console.log(`  wrote results/percival/trackc/summary.json + docs/percival/PERCIVAL_TRACKC_SELF_CORRECTION_RESULTS.md`);
