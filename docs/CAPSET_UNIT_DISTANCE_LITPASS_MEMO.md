# Capset / Unit-Distance Lit-Pass Memo

**Status**: public citation refresh, 2026-05-30  
**Surfaces**: `/capset`, `/unit-distance`, `docs/SUNDOG_V_CAPSET.md`  
**Purpose**: Move the public pages from "nice analogy with links" to
citation-first framing. The pages should make the outside mathematics primary
and Sundog's role secondary: reader overlay, workbench, evidence-tier boundary.

## Claim Boundary

Sundog did not prove, verify, or extend the cap-set result or the 2026
unit-distance result.

The public claim is narrower:

1. `/capset` is an interactive primer for the 2016 polynomial-method
   breakthrough.
2. `/unit-distance` is a plain-English overlay of the OpenAI result and the
   external mathematicians' companion account.
3. The cap-set / unit-distance "substrate rhyme" is Sundog editorial framing,
   not a theorem.

## Primary Anchors

| Source | Public URL | Supports |
| --- | --- | --- |
| OpenAI announcement, "An OpenAI model has disproved a central conjecture in discrete geometry" (2026-05-20) | https://openai.com/index/model-disproves-discrete-geometry-conjecture/ | Public announcement; problem statement; model-origin claim; checked-by-external-mathematicians claim; `n^(1+delta)` result; note that Sawin obtains `delta = 0.014`. |
| OpenAI proof PDF, "Planar Point Sets with Many Unit Distances" | https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf | Load-bearing proof artifact for the unit-distance construction. |
| Alon, Bloom, Gowers, Litt, Sawin, Shankar, Tsimerman, Wang, Wood, "Remarks on the disproof of the unit distance conjecture", arXiv:2605.20695 (2026-05-20) | https://arxiv.org/abs/2605.20695 | Human-digested, human-verified account; credits Ellenberg-Venkatesh, Golod-Shafarevich, and Hajir-Maire-Ramakrishna ideas. |
| Croot, Lev, Pach, "Progression-free sets in Z_4^n are exponentially small", Annals of Mathematics 185(1), 2017 | https://annals.math.princeton.edu/2017/185-1/p07 | Polynomial-method precursor; official Annals version and DOI. |
| Ellenberg, Gijswijt, "On large subsets of F_q^n with no three-term arithmetic progression", Annals of Mathematics 185(1), 2017 | https://annals.math.princeton.edu/2017/185-1/p08 | Cap-set bound; official Annals version and DOI. |
| Hajir, Maire, Ramakrishna, "Cutting towers of number fields", arXiv:1901.04354 | https://arxiv.org/abs/1901.04354 | Background for tower-cutting language referenced by the unit-distance companion remarks. |

## Local Corrections From This Pass

- Treat Wikipedia/overview pages as optional background only. The load-bearing
  rails should point to OpenAI, arXiv, Annals, and the local Sundog claim ledger.
- The unit-distance page previously linked `https://arxiv.org/abs/math/0006148`
  as a Hajir-Maire source; that URL resolves to an unrelated paper. Replace it
  with Hajir-Maire-Ramakrishna `https://arxiv.org/abs/1901.04354`.
- The page copy should say "OpenAI announced" and "the companion paper says"
  unless the proof artifact itself is the source. Avoid laundering the
  announcement into Sundog's own verification claim.
- The public source order should be: lit-pass memo, primary result, companion
  remarks, cap-set precedents, local Sundog ledger/workbench.

## Page-Level Framing

### `/capset`

Lead with the fact that the workbench is a citation-backed primer:
Croot-Lev-Pach introduced the polynomial-method precursor; Ellenberg-Gijswijt
adapted it to cap-set; Sundog supplies only an interactive toy model and the
analogy to the 2026 unit-distance result.

### `/unit-distance`

Lead with the fact that the page is a reader overlay:
OpenAI's model produced the result; external mathematicians checked/digested it;
the page explains the mechanism and evidence tiers without claiming authorship
or independent verification.

## Follow-Up Ratchets

- If the OpenAI proof or companion note receives a journal version, add it above
  the announcement/PDF links and update both inspection trails.
- If Sawin's explicit `delta = 0.014` refinement appears as a separate citable
  paper or note, cite that directly instead of using the OpenAI announcement as
  the current public source for the numerical refinement.
- If Sundog writes the planned three-gate reading note, promote it as a local
  interpretation artifact, not as a mathematical source.
