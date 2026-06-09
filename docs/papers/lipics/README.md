# LIPIcs build — `syndrome-decoding.tex`

LaTeX (LIPIcs / ITP) conversion of `../SYNDROME_DECODING_FORMALIZATION.md`. **First-pass draft — needs one
compile-debug pass** (no LaTeX engine is available in this checkout's Windows or WSL environments).

## To build

1. The required **Dagstuhl LIPIcs author-kit files** are vendored here from Dagstuhl's official
   `dagstuhl-publishing/styles` release `v2021.2.3`: `lipics-v2021.cls`, `cc-by.pdf`,
   `lipics-logo-bw.pdf`.
2. Engine: **pdflatex** (LIPIcs requirement).
   ```sh
   pdflatex syndrome-decoding
   pdflatex syndrome-decoding      # second pass resolves \cref / \maketitle
   ```
   The bibliography is an inline `thebibliography` (no BibTeX run needed).

## How unicode is handled (so it compiles under pdflatex)

- **Prose** unicode (≤, ⊕, σ, …) → mapped by `newunicodechar` in the preamble.
- **Lean code** unicode (∀, ↔, ↦, …) → mapped by the `listings` `literate` table in the preamble.
- If pdflatex stops with *"Unicode character X not set up for use with LaTeX"*, add one `\newunicodechar{X}{…}`
  line (prose) or one `{X}{{…}}1` literate pair (code).

## Known polish items (expected for a first conversion)

- **`\textsc{...}` inside `$…$`** (problem names in the chain) compiles in modern LaTeX but check spacing; if you
  prefer, define `\newcommand{\prob}[1]{\text{\textsc{#1}}}` and use `\prob{3sat}` in math.
- Run the LIPIcs class's own checks; it may want `\ccsdesc`/`\category` tweaks.

## Before submission (from the MD drafting notes)

- **Anonymise** for ITP double-blind: the `\author{}`/`\Copyright{}`/`\authorrunning{}` block and the
  `\supplement{}` artifact link are placeholders.
- **Re-run** the negative prior-art searches (the "to our knowledge, first" hedges).
- **Pin** the related-version / artifact DOI once the repo is archived.
- `poly-reductions` is repo-cited; re-check for a canonical paper near submission.
