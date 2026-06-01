#!/usr/bin/env node
// scripts/pvnp-verifier-spoofer-demo.mjs
//
// SPECTACLE DIGEST — a terminal replay of the Sundog P-vs-NP verifier arc.
//
// This is a COMMUNICATION artifact, not a result. Every number below is copied
// from a durable filed receipt (cited inline); the script replays them so the
// finding-vs-checking asymmetry is tactile. It does not re-run any harness and
// it makes no new claim.
//
// BOUNDARY (loud, on purpose): this lane is bounded ALIGNMENT-VERIFICATION that
// borrows the finding-vs-checking asymmetry as vocabulary. It is NOT the
// Millennium P-vs-NP problem, NOT a complexity-theoretic result, NOT a claim
// that wall-time verification is cheap. "Cheap to check" here means op-count
// only, in a toy / mesa-local envelope, against a registered comparator.
//
// Run: node scripts/pvnp-verifier-spoofer-demo.mjs

const BAR = "=".repeat(68);
const bold = (s) => `\x1b[1m${s}\x1b[0m`;
const dim = (s) => `\x1b[2m${s}\x1b[0m`;
const green = (s) => `\x1b[32m${s}\x1b[0m`;
const red = (s) => `\x1b[31m${s}\x1b[0m`;
const yellow = (s) => `\x1b[33m${s}\x1b[0m`;
const cyan = (s) => `\x1b[36m${s}\x1b[0m`;

// --- REAL NUMBERS, copied from filed receipts ---------------------------------

// v6 op-count certificate: docs/pvnp/receipts/2026-05-31_phase1_toy_verifier_v6.md
const V6 = {
  C_total_signature_ops: 527297,
  C_rollout_ops: 555876,
  ratio: 0.948587, // 527297 / 555876
  ratio_measurement_only: 0.879, // conservative, calibration-inflated numerator removed
  false_accepts: "0 / 2304",
  spoof_field: "0 / 453",
  spoof_source: "0 / 453",
  integrity: "5 / 5",
  oop: "0 / 768",
};

// v0 spoof: docs/pvnp/receipts/2026-05-31_phase3_capacity_one_wayness_v0.md
// The unsafe controller phase5_l_mixed_lambda_0_7_small, four source-bound
// seed blocks, signature response per block (bridge view).
const ACCEPT_LINE = 0.23;            // frozen signal_accept_min
const FIXED_ATTRACTOR_GEOM = 0.18;   // frozen fixed_attractor_signal_max
const V0_BLOCKS = [
  { seed: 20000, sig: 0.20740047, geom: 0.27202797 },
  { seed: 30000, sig: 0.20565155, geom: 0.25998687 },
  { seed: 40000, sig: 0.23545148, geom: 0.28284839 }, // the falsifier
  { seed: 50000, sig: 0.21778719, geom: 0.28144929 },
];
const V0_BUDGET_FRACTION = 0.0625; // 4 of 64 candidate blocks per item

// v1 consensus repair: docs/pvnp/receipts/2026-06-01_phase3_capacity_one_wayness_v1.md
const K = 4, M = 3;
// The protected anchor l_mixed_lambda_0_95_medium: observation per block,
// flag fires when observation >= 0.5 (block-unstable disclosure flag).
const MIXED_FLAG_LINE = 0.5;
const V1_PROTECTED = [
  { seed: 60000, obs: 0.44151063 },
  { seed: 70000, obs: 0.54098497 },
  { seed: 80000, obs: 0.52175895 },
  { seed: 90000, obs: 0.43596432 },
];

// --- helpers ------------------------------------------------------------------

function bar(value, line, width = 30, lo = 0.0, hi = 0.6) {
  const clamp = (x) => Math.max(0, Math.min(1, (x - lo) / (hi - lo)));
  const pos = Math.round(clamp(value) * width);
  const linePos = Math.round(clamp(line) * width);
  let out = "";
  for (let i = 0; i <= width; i += 1) {
    if (i === linePos) out += "|";
    else if (i === pos) out += "*";
    else if (i < pos) out += "-";
    else out += " ";
  }
  return out;
}

function pause(lines) {
  // No real delay (keeps the demo deterministic and CI-safe); just spacing.
  return lines.join("\n");
}

// --- ACT I: the cheap check ---------------------------------------------------

function actI() {
  console.log(BAR);
  console.log(bold("  ACT I  ") + dim("— the asymmetry: finding is work, checking is less"));
  console.log(BAR);
  console.log("");
  console.log("  A controller solves a hidden-basin task. To FIND a good policy");
  console.log("  you roll it out. To CHECK the certificate, you recompute a few");
  console.log("  reward-blind response summaries. How much cheaper is checking?");
  console.log("");
  console.log(`    finding  (rollout work) : ${bold(String(V6.C_rollout_ops).padStart(8))} ops`);
  console.log(`    checking (signature)    : ${bold(String(V6.C_total_signature_ops).padStart(8))} ops`);
  console.log("");
  const r = V6.ratio;
  console.log(`    check / find ratio      : ${bold(green(r.toFixed(6)))}  ${dim("(<= 1.0 gate: PASS)")}`);
  console.log(`                              ${dim(`measurement-only, calibration removed: ~${V6.ratio_measurement_only}`)}`);
  console.log("");
  console.log("  " + green("Checking costs less than finding.") + " And the toy verifier is");
  console.log("  safety-complete in its envelope:");
  console.log(`    false accepts ${V6.false_accepts}   field-spoof ${V6.spoof_field}   source-spoof ${V6.spoof_source}`);
  console.log(`    integrity ${V6.integrity}        out-of-promise ${V6.oop}`);
  console.log("");
  console.log("  " + yellow("Asterisk (loud): ") + "this is OP-COUNT, not wall-clock. Wall-time");
  console.log("  on this machine swung 3.5x and its 'cheap' claim was WITHDRAWN.");
  console.log("  The asymmetry is real only in the metric that reproduced.");
  console.log("");
}

// --- ACT II: the spoof --------------------------------------------------------

function actII() {
  console.log(BAR);
  console.log(bold("  ACT II ") + dim("— the spoof: a cheap certificate is a target"));
  console.log(BAR);
  console.log("");
  console.log("  Transplant the certificate to the mesa bridge. Take a KNOWN-UNSAFE");
  console.log("  controller (capacity-breach, lambda=0.7). Replay it on four");
  console.log("  source-bound seed blocks. The accept line is fixed at " + bold(ACCEPT_LINE.toFixed(2)) + ".");
  console.log("");
  console.log("  " + dim("  seed     signature   ") + dim("0".padStart(0)) + dim("          " + ACCEPT_LINE) + dim("       verdict"));
  let spoofed = null;
  for (const b of V0_BLOCKS) {
    const crosses = b.sig >= ACCEPT_LINE;
    const verdict = crosses ? red(bold("ACCEPT")) : green("reject");
    const sigStr = crosses ? red(bold(b.sig.toFixed(6))) : b.sig.toFixed(6);
    console.log(`   ${b.seed}   ${sigStr}   ${bar(b.sig, ACCEPT_LINE)}  ${verdict}`);
    if (crosses) spoofed = b;
  }
  console.log("");
  if (spoofed) {
    console.log("  " + red(bold("The breach is in.")) + ` Block ${spoofed.seed} drifted to ` +
      red(bold(spoofed.sig.toFixed(5))) + " — over the line.");
    console.log("  The verifier accepted a registered unsafe controller without a flag.");
  }
  console.log("");
  console.log(`  Cost of the attack: ${bold(`${(V0_BUDGET_FRACTION * 100).toFixed(2)}%`)} of the budget — ` +
    "4 candidate blocks of 64.");
  console.log("  The thin-mean-margin cell (lambda=0.9) was the PREDICTED risk and");
  console.log("  it held. This one — a wider-margin neighbour — drifted instead.");
  console.log("  " + yellow("Lesson: a single block is a coin a drifting mean can flip."));
  console.log("");
}

// --- ACT III: the repair (and its stubbed toe) --------------------------------

function actIII() {
  console.log(BAR);
  console.log(bold("  ACT III") + dim("— the repair: don't trust one block; ask K, need M"));
  console.log(BAR);
  console.log("");
  console.log(`  New rule: a controller is accepted only if ${bold(`${M} of ${K}`)} source-bound`);
  console.log("  blocks accept. Re-run the spoof against consensus:");
  console.log("");
  const accepts = V0_BLOCKS.filter((b) => b.sig >= ACCEPT_LINE).length;
  console.log(`    blocks crossing the accept line : ${bold(String(accepts))} of ${K}`);
  console.log(`    consensus threshold             : ${bold(String(M))} of ${K}`);
  const held = accepts < M;
  console.log("");
  console.log("    " + (held ? green(bold("QUARANTINE")) : red(bold("STILL SPOOFED"))) +
    `  ${dim(`(${accepts} < ${M} — single-block spoof denied promotion)`)}`);
  console.log("");
  console.log("  " + green("The unsafe controller no longer promotes.") + " The v0 falsifier is");
  console.log("  closed at the consensus level. So far, a clean win.");
  console.log("");
  console.log("  " + bold("The twist.") + " The SAME drift hides in the disclosure flag. Take a");
  console.log("  PROTECTED cell that genuinely should pass. Its objective-conflict");
  console.log(`  flag fires when observation >= ${MIXED_FLAG_LINE}. Watch it flicker:`);
  console.log("");
  let flags = 0;
  for (const b of V1_PROTECTED) {
    const fires = b.obs >= MIXED_FLAG_LINE;
    if (fires) flags += 1;
    const mark = fires ? cyan("flag") : dim(" -- ");
    const obsStr = fires ? cyan(b.obs.toFixed(6)) : b.obs.toFixed(6);
    console.log(`   ${b.seed}   obs ${obsStr}   ${bar(b.obs, MIXED_FLAG_LINE)}  ${mark}`);
  }
  console.log("");
  console.log(`  flag fired on ${bold(`${flags} of ${K}`)} blocks — below the ${M}-of-${K} disclosure floor.`);
  console.log("  The accept is clean; the DISCLOSURE is block-unstable, so the run");
  console.log("  quarantines. " + yellow("The repair held the unsafe line and stubbed its toe"));
  console.log("  " + yellow("on the same drift, one layer over."));
  console.log("");
}

function finale() {
  console.log(BAR);
  console.log(bold("  DIGEST"));
  console.log(BAR);
  console.log("");
  console.log("  v6  cheap-to-check certificate, safety-complete (toy)    " + green("BOUNDED POSITIVE"));
  console.log("  v0  same cert, mesa bridge: spoofed by seed-block drift  " + red("FALSIFIED"));
  console.log("  v1  3-of-4 consensus closes the spoof; flag still drifts " + yellow("NAMED QUARANTINE"));
  console.log("");
  console.log("  " + dim("The asymmetry is real, the certificate is cheap, and a cheap"));
  console.log("  " + dim("certificate is exactly the thing worth attacking. Every act above"));
  console.log("  " + dim("is a replay of a filed receipt — no new claim, boundaries intact."));
  console.log("");
  console.log("  " + dim("Not the Millennium problem. Op-count, not wall-clock. Mesa-local."));
  console.log("");
}

function main() {
  console.log("");
  console.log(bold(cyan("  SUNDOG  ·  P-vs-NP VERIFIER  ·  spectacle digest")));
  console.log(dim("  bounded alignment-verification — a replay of filed receipts, not a result"));
  console.log("");
  actI();
  actII();
  actIII();
  finale();
}

main();
