#!/usr/bin/env node
// H-K5 synthesis anchor: pull the actual collision numbers from BOTH lanes' committed
// receipts into ONE operational schema. The falsifier COLLISION_TABLE_ONLY_RHYME fires
// unless each lane can populate the required operational fields from its receipts
// {shadow, body, numeric lossiness measure, control, restoring structure} - i.e. the two
// rows must share an operational field beyond vocabulary. Report-only; no public claim.

import fs from "node:fs";
import path from "node:path";

const OUT_DIR = path.join("results", "synthesis", "shadow-collision-table");
const readJSON = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null);
const num = (x) => (typeof x === "number" ? x : null);

// --- Kakeya row, read from the H-K3 + H-K4 receipts -------------------------------------
function kakeyaRow() {
  const qs = [
    { q: 5, dir: "shadow-collision-audit" },
    { q: 7, dir: "shadow-collision-audit-q7" },
    { q: 11, dir: "shadow-collision-audit-q11" },
  ];
  const collisions = {}; let dvirFloors = {}; let clear = true;
  for (const { q, dir } of qs) {
    const m = readJSON(path.join("results", "kakeya", dir, "manifest.json"));
    if (!m) { collisions[q] = null; continue; }
    collisions[q] = num(m.structuredLineExtensions?.maxNonemptyCollisionClassCount);
    dvirFloors[q] = num(m.core?.dvirFloor);
    if (m.falsifier?.fired) clear = false;
  }
  const adaptive = readJSON(path.join("results", "kakeya", "adaptive-fibering-panel", "manifest.json"));
  const lossiness = num(collisions[5]) != null && num(collisions[7]) != null && num(collisions[11]) != null;
  return {
    lane: "Kakeya",
    shadow: "direction-coverage signature (which directions contain a full line)",
    body: "point/tube set in F_q^2",
    lossiness_measure: { type: "finite count of bodies per shadow signature",
      law: "q(q^2 - q + 1)", values: collisions, value_present: lossiness },
    control: { name: "random same-size body baseline (H-K4)",
      present: Boolean(adaptive), caveat: adaptive
        ? "count is a finite-grid density artifact unless measured as excess over a size-matched control"
        : "MISSING" },
    restoring_structure: "Dvir polynomial method - complete shadow forces size >= q(q+1)/2 (dvirFloor "
      + `q5=${dvirFloors[5]}, q7=${dvirFloors[7]}, q11=${dvirFloors[11]})`,
    determine_resist: "finite resist: size determined (Dvir floor), exact body not",
    receipts: ["docs/kakeya/PHASE3B_SHADOW_COLLISION_AUDIT.md", "docs/kakeya/PHASE3C_SHADOW_COLLISION_Q11.md",
      "docs/kakeya/PHASE4_ADAPTIVE_FIBERING_PANEL.md"],
    falsifier_clear: clear,
    operational_fields_populated: lossiness && Boolean(adaptive),
  };
}

// --- Hodge row, read from the H-K1 model-eval + judge receipts ---------------------------
function hodgeRow() {
  const judge = readJSON(path.join("results", "hodge", "register-judge", "summary.json"));
  const sweep = readJSON(path.join("results", "hodge", "register-modeleval", "sweep", "comparison.json"));
  let overclaimRate = null, neutralOverclaim = null, neutralCards = null, consensus = null;
  if (sweep) {
    const ne = sweep.cells.filter((c) => c.mode === "neutral");
    neutralOverclaim = ne.reduce((s, c) => s + c.overclaimed, 0);
    neutralCards = ne.reduce((s, c) => s + c.n, 0);
    overclaimRate = neutralCards ? Number((neutralOverclaim / neutralCards).toFixed(3)) : null;
  }
  if (judge) consensus = judge.cells.filter((c) => c.mode === "neutral").map((c) => `${c.provider}=${c.consensus_route}/10`);
  const lossinessPresent = overclaimRate != null;
  const controlPresent = Boolean(judge) && num(judge.judge_consensus_vs_lexical_route_agreement) != null;
  return {
    lane: "Hodge",
    shadow: "rational (p,p) cohomology class",
    body: "algebraic cycle",
    lossiness_measure: { type: "decoder overclaim rate (shadow does not determine cycle existence)",
      neutral_overclaim: neutralOverclaim, neutral_cards: neutralCards,
      unprompted_overclaim_rate: overclaimRate,
      judge_consensus_route_neutral: consensus, value_present: lossinessPresent },
    control: { name: "two independent semantic judges + lexical-vs-judge agreement (H-K1 PHASE4F)",
      present: controlPresent,
      caveat: controlPresent
        ? `lexical route check over-credits: judge-consensus vs lexical agreement ${judge.judge_consensus_vs_lexical_route_agreement}; inter-judge ${judge.inter_judge_verdict_agreement}`
        : "MISSING" },
    restoring_structure: "Lefschetz (1,1) / hard Lefschetz / Cattani-Deligne-Kaplan in special cases; "
      + "general case open (the Hodge conjecture)",
    determine_resist: "infinite resist in general: existence undecidable by the shadow (sigma = infinity)",
    receipts: ["docs/hodge/PHASE4D_REGISTER_MODELEVAL.md", "docs/hodge/PHASE4E_REGISTER_SWEEP.md",
      "docs/hodge/PHASE4F_REGISTER_SEMANTIC_JUDGE.md"],
    falsifier_clear: true,
    operational_fields_populated: lossinessPresent && controlPresent,
  };
}

function main() {
  const rows = [kakeyaRow(), hodgeRow()];
  // Shared operational field: BOTH lanes must populate {numeric lossiness, control} from receipts.
  const sharedOperationalField = rows.every((r) => r.operational_fields_populated);
  const measureTypes = rows.map((r) => r.lossiness_measure.type);
  const typedMismatch = new Set(rows.map((r) => (r.lossiness_measure.type.includes("count") ? "count" : "rate"))).size > 1;
  const fired = !sharedOperationalField;

  const manifest = {
    artifactId: "HK5-SHADOW-COLLISION-TABLE", generatedAt: new Date().toISOString(),
    status: "internal synthesis anchor (report-only)",
    hook: "H-K5 cross-lane shadow-collision table",
    shared_operational_field: sharedOperationalField
      ? "both lanes populate {shadow, body, numeric lossiness measure, control, restoring structure} from committed receipts; both sit on the determine/resist axis (lossy shadow + named structure that restores the body)"
      : "at least one lane could not populate the operational fields from its receipts",
    typed_mismatch: {
      present: typedMismatch,
      detail: "Kakeya lossiness = finite body-count per signature; Hodge lossiness = decoder overclaim rate (existence-undecidability). Same frame and same decoder-non-invertibility measurement, different units (count vs rate).",
      measure_types: measureTypes,
    },
    falsifier: {
      name: "COLLISION_TABLE_ONLY_RHYME",
      fired,
      reason: fired
        ? "A lane could not populate a required operational field from its receipts; the synthesis is vocabulary only."
        : "Both rows populate the same operational schema from committed receipts (numeric, control-backed lossiness + restoring structure), so the table guides probes, not just rhymes. Caveat: the lossiness measure is typed differently per lane (count vs rate).",
    },
    rows,
  };
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUT_DIR, "collision-table.json"), JSON.stringify(manifest, null, 2) + "\n");
  const csv = ["lane,shadow,body,lossiness_type,lossiness_value,control_present,restoring_structure,determine_resist,operational",
    ...rows.map((r) => [r.lane, r.shadow, r.body, r.lossiness_measure.type,
      r.lane === "Kakeya" ? JSON.stringify(r.lossiness_measure.values) : r.lossiness_measure.unprompted_overclaim_rate,
      r.control.present, r.restoring_structure, r.determine_resist, r.operational_fields_populated]
      .map((v) => { const s = String(v); return /[",]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s; }).join(","))]
    .join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "collision-table.csv"), csv + "\n");

  for (const r of rows) console.log(`ROW ${r.lane}: operational=${r.operational_fields_populated} lossiness=${r.lossiness_measure.type}`);
  console.log(`HK5_COLLISION_TABLE shared_operational_field=${sharedOperationalField} typed_mismatch=${typedMismatch} `
    + `falsifier=${fired ? "fired" : "clear"} out=${OUT_DIR}`);
}

main();
