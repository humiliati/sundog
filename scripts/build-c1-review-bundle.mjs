#!/usr/bin/env node
// scripts/build-c1-review-bundle.mjs
//
// Assemble the PDE C1 external-review packet into ONE self-contained markdown
// bundle so the email's `[link]` blocker collapses to "attach this one file."
//
// It concatenates, in reviewer read-order: a cover header (scope + the four
// locked questions + the answer menu), the separation statement, the current
// result doc, the two adjudicator halves, and the three verdict-bearing run
// receipts inlined verbatim. Nothing is paraphrased — every section is the
// on-disk file, so the bundle cannot drift from the artifacts.
//
// Output: internal/outreach/PDE_C1_REVIEW_BUNDLE.md
// Render to PDF separately (e.g. the repo pdf skill) if a single PDF is wanted.

import { readFileSync, writeFileSync, mkdirSync, statSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = "internal/outreach/PDE_C1_REVIEW_BUNDLE.md";

// Reviewer read-order. Each entry is inlined verbatim under a divider.
const SECTIONS = [
  { tag: "PRIMARY · separation statement", file: "docs/proof/PDE_C1_SEPARATION_STATEMENT.md" },
  { tag: "RESULT · two-regime portable objective", file: "docs/proof/PDE_C1_REGIME_GENERALITY_v1.md" },
  { tag: "ADJUDICATOR · control-sufficiency (kNN convergence)", file: "docs/proof/PDE_C1_KNN_CONVERGENCE_CHECK.md" },
  { tag: "ADJUDICATOR · state-insufficiency (twin-state)", file: "docs/proof/PDE_C1_TWIN_STATE_CERTIFICATE.md" },
];

const RECEIPTS = [
  { tag: "RUN RECEIPT · G=200 kNN sweep (positive control)", file: "results/proof/c1-rg-v1-g200-knn-sweep/PDE_C1_KOLMOGOROV_RESULTS.md" },
  { tag: "RUN RECEIPT · G=300 kNN sweep (generality test)", file: "results/proof/c1-rg-v1-g300-knn-sweep/PDE_C1_KOLMOGOROV_RESULTS.md" },
  { tag: "RUN RECEIPT · G=300 twin-state (support companion)", file: "results/proof/c1-rg-v1-g300-twin-state/PDE_C1_KOLMOGOROV_RESULTS.md" },
];

function read(rel) {
  return readFileSync(path.resolve(REPO, rel), "utf8");
}

function sha12(text) {
  return createHash("sha256").update(text).digest("hex").slice(0, 12);
}

const COVER = `# PDE C1 Finite-Galerkin Separation — Self-Contained Review Bundle

> **This is one file containing the whole reviewer packet**, assembled by
> \`scripts/build-c1-review-bundle.mjs\` from the on-disk artifacts. It is the
> attachment the outreach email refers to as the packet. Send this (or its PDF
> rendering) plus the email from
> \`docs/proof/PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md\`.

**Status: PROVISIONAL / UNPROMOTED, gated on external review.** This is a
structural separation in a finite-dimensional Galerkin truncation of 2D
Navier–Stokes. It is **not** a Navier–Stokes existence/smoothness claim, **not**
a new determining-modes theorem, **not** a statement about the
infinite-dimensional NSE attractor, and **not** a Clay-problem claim.

## The one-sentence ask

On the sampled invariant measure of a 2D Kolmogorov-flow Galerkin model
(\`k_f = 2\`, \`K = 3\`, \`G ∈ {200, 300}\`), the low-band signature \`Φ_K\` is
**state-insufficient** (certified non-injective) yet **control-sufficient** for a
registered decision (\`Φ_K\`-measurable up to a measure-\`δ ≈ 0.037\` boundary
layer). Is this a genuine finite-Galerkin separation between state-reconstruction
and action-sufficiency, or ordinary observer / data-assimilation / LES-closure
behavior that should not be made load-bearing?

## The four locked questions

1. **State-insufficiency language.** Given the twin-state certificate is
   sampled-SRB / finite-Galerkin (not an exact attractor theorem), is
   "\`Φ_K\` non-injective / state-insufficient on \`supp μ\`" reasonable, or should
   it be weakened to a finite-sample claim?
2. **Control-sufficiency language.** Is the kNN/disintegration reading of
   "\`π*\` factors through \`Φ_K\` up to \`μ\`-measure zero" (local action-mixing
   \`mean_minority → 0\`, plus paired action-constancy on the certified
   non-injective pairs) mathematically fair, or overstated?
3. **Objective legitimacy.** Is a held-out look-ahead-max quantile a defensible
   regime-portable proxy action, given the older burn-in-percentile trigger went
   vacuous at \`G = 300\` and was replaced (not retuned)?
4. **Real separation vs renaming.** A genuine "a sub-determining mode set can be
   control-sufficient without being state-reconstructive" separation, or ordinary
   functional-observability / data-assimilation / LES-closure behavior?

## The answer menu (a paragraph is plenty — a negative reply is the most valuable)

\`\`\`text
The framing seems conservative / basically right.
Weaken or rename X (e.g. "state-insufficient" → finite-sample only).
The control-sufficiency / fiber language is too strong because Z.
This is standard observer / data-assimilation, cite B.
The objective is implicitly a function of Φ_K; the separation is vacuous because W.
\`\`\`

## Anti-folklore guard (worth pre-empting)

The natural reviewer reflex — "this is just LES / closure: of course coarse
energy is predictable from coarse state" — is exactly what the non-injectivity
certificate is meant to foreclose: LES / AIM / closure assume or derive
**state-sufficiency** via slaving; C1 *certifies state-insufficiency* (twin
states) and shows the **decision** survives the genuinely unresolved fine state.
Please stress-test that distinction specifically.

---

*Read order below: separation statement → result → two adjudicators → three run
receipts (verbatim). Each section is the unmodified on-disk artifact.*
`;

function main() {
  const parts = [COVER];
  const provenance = [];

  for (const s of [...SECTIONS, ...RECEIPTS]) {
    const body = read(s.file);
    provenance.push({ file: s.file.replaceAll("\\", "/"), bytes: Buffer.byteLength(body), sha256_12: sha12(body) });
    parts.push(
      `\n\n${"=".repeat(78)}\n` +
      `## [ ${s.tag} ]\n` +
      `*source: \`${s.file.replaceAll("\\", "/")}\` · sha256:${sha12(body)}*\n` +
      `${"=".repeat(78)}\n\n` +
      body.trim() + "\n",
    );
  }

  // Provenance footer so a reviewer (or we) can confirm the bundle matches disk.
  const footer = [
    `\n\n${"=".repeat(78)}\n## [ BUNDLE PROVENANCE ]\n${"=".repeat(78)}\n`,
    "Each section above is a verbatim copy of an on-disk artifact. Hashes:",
    "",
    "| File | Bytes | sha256 (first 12) |",
    "| --- | ---: | --- |",
    ...provenance.map((p) => `| \`${p.file}\` | ${p.bytes} | \`${p.sha256_12}\` |`),
    "",
    "Regenerate with `node scripts/build-c1-review-bundle.mjs`. If a hash here",
    "does not match a fresh read of the file, the bundle is stale — rebuild before",
    "sending.",
    "",
  ].join("\n");
  parts.push(footer);

  // Closing page. The PDF renderer replaces the sentinel below with an on-brand
  // halo graphic; in plain markdown the text conclusion stands on its own.
  const closing = [
    `\n\n${"=".repeat(78)}\n## [ CLOSING ]\n${"=".repeat(78)}\n`,
    "<!-- PDF_CLOSING_GRAPHIC -->",
    "",
    "### Thank you for reading to the end.",
    "",
    "That is the whole packet. The ask was narrow on purpose: not endorsement,",
    "not a Navier–Stokes claim — just a framing check from someone who knows where",
    "the honest boundary of this problem sits. If the language is too strong, the",
    "most useful thing you can tell us is exactly which phrase to weaken and why.",
    "A negative answer is the most valuable outcome we could get.",
    "",
    "> **One last note, for a cover-to-cover read.** A sundog — the bright parhelion",
    "> beside the sun that gives this lab its name — is the sky's own `Φ_K`: a",
    "> low-information projection of an enormous hidden state (every ice crystal,",
    "> every ray) that is nonetheless *enough to read the sun's altitude*, yet never",
    "> enough to reconstruct the whole sky. State-insufficient, decision-sufficient.",
    "> That is the separation this packet is asking you to check — and the reason we",
    "> went looking for it in a 2D Navier–Stokes attractor in the first place.",
    "",
    "*— Jeffery Hughes Jr., Sundog*",
    "",
  ].join("\n");
  parts.push(closing);

  const assembled = parts.join("");
  mkdirSync(path.resolve(REPO, "internal/outreach"), { recursive: true });
  writeFileSync(path.resolve(REPO, OUT), assembled, "utf8");

  const bytes = Buffer.byteLength(assembled);
  console.log(`C1 review bundle written: ${OUT}`);
  console.log(`  sections: ${SECTIONS.length} primary + ${RECEIPTS.length} receipts`);
  console.log(`  size: ${bytes} bytes (${Math.round(bytes / 1024)} KB)`);
  console.log(`  provenance hashes: ${provenance.length}`);
}

main();
