# CPP build — `syndrome-decoding-cpp.tex`

ACM acmart (sigplan) conversion of the paper for **CPP 2027** (co-located with
POPL 2027, Mexico City). Converted 2026-06-10 from the LIPIcs build
([`../lipics/syndrome-decoding.tex`](../lipics/syndrome-decoding.tex)); the MD master
is [`../SYNDROME_DECODING_FORMALIZATION.md`](../SYNDROME_DECODING_FORMALIZATION.md).
**Builds clean** (TeX Live 2026): 7 pp two-column incl. bibliography, 0 errors,
0 undefined refs/cites, 0 overfull boxes; all pages visually verified.
CPP limit: 12 pp **excluding** bibliography — ample headroom.

## To build

```sh
pdflatex syndrome-decoding-cpp
bibtex   syndrome-decoding-cpp
pdflatex syndrome-decoding-cpp
pdflatex syndrome-decoding-cpp
```

acmart and ACM-Reference-Format ship with TeX Live (nothing vendored).

## Submission-mode notes (do not regress)

- Class options are `[sigplan,review,anonymous]` per the CPP CFP (lightweight
  double-blind). Camera-ready: drop `review,anonymous`, fill the real author
  block, `\acmConference`/`\acmYear`/`\acmDOI`, and `\setcopyright`.
- `\acmConference` currently holds **CPP '27 placeholders** — update from the
  real CFP once POPL 2027 posts it (expected ~July 2026; abstracts ~Sep 9–10,
  papers ~Sep 16–17, 2026, by the CPP 2025/2026 pattern).
- **No `\lstinline` inside the abstract** — acmart stores the abstract in a
  macro, so a raw `#` (e.g. `#guard_msgs`) is a fatal error. Use
  `\texttt{\#guard\_msgs}` there.
- Displayed Lean listings use `style=leandisplay` (`\footnotesize`) and are
  **hand-wrapped to ≤48 chars** in faithful Lean style for the two-column
  measure. Never let them soft-wrap; token content must match the artifact
  (verbatim-checked 2026-06-10 against sundogcert rev `d5d1223`).
- `\emergencystretch=2em` absorbs the long-code-identifier line-breaking
  pressure of the narrow columns; without it ~30 overfulls appear.
- The wheel figure is a `figure*` (two-column span).
- Unicode handling is identical to the LIPIcs build (newunicodechar + listings
  literate table); every non-ASCII char inside listings must be in the literate
  table.

## Content status

Content is a faithful port of the MD master as of 2026-06-10 (incl. the
prior-art-recheck corrections and the verbatim-listing fixes). **Still owed for
CPP specifically:** sharpen the §5 proof-engineering / experience-report angle
and pitch the artifact for CPP's artifact evaluation (owner sign-off on content
changes). Pre-submission gates tracked in
[`../lipics/README.md`](../lipics/README.md) "Before submission" +
[`../PRIOR_ART_RECHECK_2026-06-10.md`](../PRIOR_ART_RECHECK_2026-06-10.md).
