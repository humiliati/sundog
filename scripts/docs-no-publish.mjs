// Docs withheld from the public site (dist/docs/). A path entry that names a
// directory withholds the whole subtree; a bare filename withholds that file.
//
// This is the SINGLE source of truth, imported by:
//   - copy-site-docs.mjs            (build-time withhold from dist/)
//   - check-public-copy-integrity.mjs ([D] route-citation WARN)
//   - build-chat-index.mjs          (emits trace hrefs only for published docs)
//
// IMPORTANT: the withhold operates on the working tree (readdir), so it holds
// back files even when they are gitignored but still present on disk — exactly
// the case for the confidential material below after the 2026-06-03 leak
// remediation. Keep the /docs/* entries in .gitignore in sync with this set.
//
//   501c3 = confidential patent / counsel material + 501(c)(3) governance
//           drafts. Public accessibility can itself count as a patent
//           disclosure, so this MUST never reach dist/.
//   chatv2 / deconfound / JEPA / allelopathy = white-box "representational"
//           instrument R&D (kill-gated; each carries an explicit "No public
//           surface" stamp). Held until BOTH the research freeze lifts
//           (LATTICE + threebody v0.19) AND counsel clears.
//
// CROSS_SUBSTRATE_NOTES.md was removed from this set 2026-06-09 (owner
// decision): 14 Ask Sundog routes cite it as visible trace support, the file
// is git-tracked in the public repo (already disclosed), and withholding it
// only made the cited evidence unreachable on the site.
export const DOCS_NO_PUBLISH = new Set([
  "501c3",
  "chatv2",
  "deconfound",
  "SUNDOG_V_DECONFOUND.md",
  "DECONFOUND_REAL_DATA_MEMO.md",
  "SUNDOG_V_JEPA.md",
  "JEPA_LIT_PASS_MEMO.md",
  "SUNDOG_V_ALLELOPATHY.md",
]);

// True when a doc path (with or without a leading "docs/" or "/docs/")
// is withheld from dist/.
export function isNoPublish(doc) {
  const rel = String(doc || "").replaceAll("\\", "/").replace(/^\/?docs\//, "");
  return DOCS_NO_PUBLISH.has(rel.split("/")[0]) || DOCS_NO_PUBLISH.has(rel);
}
