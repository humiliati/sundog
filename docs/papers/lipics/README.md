# LIPIcs build — `syndrome-decoding.tex`

LaTeX (LIPIcs / ITP) conversion of `../SYNDROME_DECODING_FORMALIZATION.md`.
**Compile-debug pass DONE 2026-06-10** (TeX Live 2026, pdflatex): builds clean —
11 pages, zero errors, zero undefined references/citations, zero overfull boxes;
all pages + the TikZ wheel figure visually verified.

## To build

1. The required **Dagstuhl LIPIcs author-kit files** are vendored here:
   `lipics-v2021.cls` (v3.1.3, 2023/05/12 — verified byte-identical to Dagstuhl's
   current `master`), `cc-by.pdf`, `lipics-logo-bw.pdf`.
2. Engine: **pdflatex** (LIPIcs requirement).
   ```sh
   pdflatex syndrome-decoding
   pdflatex syndrome-decoding      # second pass resolves \cref / \maketitle
   ```
   The bibliography is an inline `thebibliography` (no BibTeX run needed).

## Build gotchas (already handled in the .tex — do not regress)

- **`thm-restate` class option is OMITTED.** Under TeX Live 2026 the option's
  thmtools/cleveref counter-aliasing breaks `lipics-v2021.cls` v3.1.3
  (`! Command \c@lemma already defined`). The paper declares no theorem
  environments, so nothing is lost. If restatable theorems are ever added,
  re-add the option and re-test (a `latexrelease` kernel rollback alone does
  NOT fix it; a newer Dagstuhl class release might).
- **`\doi` is `\providecommand`-defined in the preamble** — the inline
  bibliography doesn't get it from the BibTeX `plainurl` flow.
- **Inline code breaks at spaces** (`breaklines=true, breakatwhitespace=true`
  in the `lean` listings style). Displayed Lean blocks are wrapped *by hand*
  so every shown line break is faithful Lean style, never a soft wrap.
- **`\lstinline` arguments containing `{}` must use non-brace delimiters**,
  e.g. `\lstinline|{t_j, t̄_j}|`.

## How unicode is handled (so it compiles under pdflatex)

- **Prose** unicode (≤, ⊕, σ, …) → mapped by `newunicodechar` in the preamble.
- **Lean code** unicode (∀, ↔, ↦, …) → mapped by the `listings` `literate` table in the preamble.
- If pdflatex stops with *"Unicode character X not set up for use with LaTeX"*
  (prose) or *"Invalid UTF-8 byte sequence"* (inside listings), add one
  `\newunicodechar{X}{…}` line (prose) or one `{X}{{…}}1` literate pair (code).
  **Every non-ASCII char that appears inside any `\lstinline`/`lstlisting` must
  be in the literate table** — a missing one is a fatal error, not a warning.

## Known polish items

- Run the LIPIcs class's own checks; it may want `\ccsdesc`/`\category` tweaks.
- `\category{}` / `\relatedversion{}` are empty placeholders.

## Before submission (from the MD drafting notes)

- **Anonymise** for ITP double-blind: the `\author{}`/`\Copyright{}`/`\authorrunning{}`
  block and the `\supplement{}` artifact link are placeholders (currently set to
  "Anonymous Author(s)" — fine for review; fill for camera-ready).
- **Anonymity caveat:** the venue-neutral MD source of this paper lives in the
  PUBLIC sundog repo, and the Lean artifact repo name is identifying. Reviewers
  searching distinctive phrases can de-anonymize; check the venue's prior-public
  / preprint policy, and use an anonymized artifact mirror
  (e.g. anonymous.4open.science) for the `\supplement` link.
- **Re-run** the negative prior-art searches (the "to our knowledge, first" hedges).
- **Pin** the related-version / artifact DOI once the repo is archived.
- `poly-reductions` is repo-cited; re-check for a canonical paper near submission.
- Check the **ITP CFP page limit** (recent ITPs: ~16 pp + references, LIPIcs
  style); the draft is 11 pp incl. bibliography — comfortably inside.
