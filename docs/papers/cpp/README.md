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
- `\acmConference` currently holds **CPP '27 placeholders**. Real CFP is now
  live ([popl27.sigplan.org/home/CPP-2027](https://popl27.sigplan.org/home/CPP-2027)):
  CPP 2027, Jan 11–12 2027, Hilton Mexico City Reforma. Fill these at
  camera-ready.
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

## CPP 2027 CFP — the real facts (fetched 2026-06-16)

- **Dates (AoE, strictly enforced, no extensions):** abstract **Sep 3, 2026**;
  paper **Sep 10, 2026**; reviews round 1 due Oct 10; round 2 due Nov 3;
  notification **Nov 10, 2026**; camera-ready **Nov 25, 2026**; conference
  **Jan 11–12, 2027** (Mexico City).
- **Format:** acmart `sigplan`, two-column, 10pt; **12 pp excl. bibliography &
  appendices.** Our build is ~7 pp excl. bib → big headroom. ✓
- **Double-blind:** lightweight (omit names/affiliations; third-person
  self-citation). Draft verified compliant 2026-06-16 (no identifying strings,
  no first-person self-cites, anon author block, github URL absent from the CPP
  build). ✓
- **Submission site:** [cpp2027.hotcrp.com](https://cpp2027.hotcrp.com).
- **Supplementary material is submitted AS AN ARCHIVE at submission time, NOT a
  URL.** The **anonymized `decodecert` zip** (`Dev/decodecert-artifact-review.zip`)
  IS this archive — no anonymous URL host needed. Anonymous supplementary is seen
  by reviewers before first reviews.
- **No separate Artifact Evaluation track** is mentioned in the CFP (so the
  earlier "pitch for AE" item is moot; the reproducibility paragraph in §5 stays
  as the artifact story). Confirm on the page in case a separate AE call posts later.
- **Concurrent submission banned** → confirms the sequential plan: CPP first,
  ITP 2027 (~Feb) as fallback *after* the Nov 10 notification. arXiv preprint is
  explicitly allowed ("authors may freely disseminate drafts").
- **In-person:** ≥1 author must attend in Mexico City.
- **TO VERIFY on the page:** whether the two-round review includes an
  author-response/rebuttal window the author must be available for.

## Content status

Content is a faithful port of the MD master (incl. the prior-art-recheck
corrections, the verbatim-listing fixes, and the **P1–P4 §5/§6 sharpening**,
applied 2026-06-10). No further content owed for CPP; remaining work is
submission mechanics (HotCRP registration, abstract/topics/conflicts, upload),
the deadline-week prior-art re-run, and camera-ready de-anonymization. Gates in
[`../lipics/README.md`](../lipics/README.md) +
[`../PRIOR_ART_RECHECK_2026-06-10.md`](../PRIOR_ART_RECHECK_2026-06-10.md).
