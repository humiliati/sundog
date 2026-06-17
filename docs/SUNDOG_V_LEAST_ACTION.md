# Sundog Least Action Ledger

Public name (locked 2026-06-17): **Sundog Principle of Least Reader-Action**
(short: **Least Reader-Action**).

Naming decisions (Phase 0):

- **Least Reader-Action** is the public name for the discipline.
- **The Sundog Lagrangian** is promoted to public — but strictly as the name of
  the *legibility functional* `J(doc, reader, task)` and its `B/Phi/T/A/I/F/R`
  coordinate chart. It is a reader-action object, never a physics claim: it does
  not assert that nature optimizes anything or that the lanes share one
  mathematical Lagrangian (see Claim Boundary). **Lagrangia** is its allowed
  short nickname for the same object.
- Canonical filename stays `SUNDOG_V_LEAST_ACTION.md` (already linked from
  `docs/README.md`; the file path is not the public title).

Working hook:

> A result is not mature until the shortest honest path through it is visible.

Short version:

> Euler should not be easier to read than our own work from last week.

Status: **DRAFT SCAFFOLD.** Opened 2026-06-17 as a legibility and
meta-generality roadmap. This is not a new physics claim. It is a discipline
for making Sundog claims, proofs, ledgers, and public pages recoverable by a
future reader who does not remember the context.

## Historical Spark

This roadmap starts from the sting of reading Euler raw and finding him
embarrassingly humane: clear, patient, and careful about carrying the reader
from intuition to mechanism. The immediate prompt was Euler's 1762 note to
Lagrange, paraphrased here until the source is locked: Maupertuis, were he
alive, would be glad to see least action raised to its highest dignity.

Phase 0.5 must verify the exact quotation before any public epigraph. Candidate
source: Philip E. B. Jourdain, "Maupertuis and the Principle of Least Action,"
*The Monist* 22(3), 1912, p. 459, citing Euler to Lagrange, 1762-11-09.
Modern secondary check: Marius Stan's "Euler, Newton, and Foundations for
Mechanics" (Oxford Handbook of Newton chapter draft, 2017) dates the same
letter to 1762-11-09 and gives a close variant. Phase 0.5 should reconcile
the Jourdain wording, the Stan wording, and the original correspondence before
this becomes an epigraph.

The scientific inspiration is the same lineage that makes Snell's law feel
uncannily modern: a local law of refraction can be recovered from a global
stationary-time principle. The lesson for Sundog is not "everything optimizes."
The lesson is that a hard domain can become legible when the right functional
names what is being varied, what is held fixed, and what condition survives all
allowed perturbations.

Terminology note: use **principle** in public copy. Use **principal** only when
quoting an older source verbatim.

## Claim Boundary

This ledger does **not** claim:

- that nature literally minimizes a Sundog quantity;
- that all Sundog lanes share one mathematical Lagrangian;
- that legibility is the same as simplicity;
- that a shorter document is automatically better;
- that historical analogy is evidence.

It claims only that Sundog now needs a cross-lane legibility discipline strong
enough to test its own generality. A domain is not "generalized" merely because
the same words appear in two places. The generality becomes live only when a
reader can recover the same role structure, the same imports, and the same
falsifier without private memory.

## The Principle

Working statement:

> Minimize the reader-action required to recover the claim, subject to the
> constraint that no assumption, import, boundary, or falsifier is hidden.

Reader-action is the work a competent future reader must do before they can
answer five questions:

1. What is the hidden body or object?
2. What is the shadow, trace, certificate, or indirect signal?
3. What transformation makes the shadow useful?
4. What action, proof step, or decision is justified?
5. What would make the claim fail?

The right edit is not always the shortest edit. A diagram, table, toy example,
or duplicated summary can reduce reader-action even if it adds words. A dense
proof can be legible if every coordinate is named before it is used. A short
paper can be illegible if it saves length by exporting work to the reader.

## The Sundog Lagrangian, v0

For each lane, write a local coordinate chart:

| symbol | role | reader-facing test |
| --- | --- | --- |
| `B` | body / hidden world / object of interest | Can the reader name what is not directly observed? |
| `Phi` | shadow / signature / trace / certificate | Can the reader say what is actually measured or checked? |
| `T` | transform from `Phi` to useful structure | Can the reader follow the mechanism without guessing private context? |
| `A` | action / accepted proof step / policy decision | Can the reader say what the system is allowed to do with `T(Phi)`? |
| `I` | imports / assumptions / external facts | Are non-proved dependencies named where the claim uses them? |
| `F` | falsifier / boundary / null route | Is there a named result that would demote or void the claim? |
| `R` | receipt / replay / checked artifact | Can the reader rerun, inspect, or audit the evidence path? |

The draft "action" functional is qualitative at first:

```text
J(doc, reader, task)
  = missing-inference cost
  + notation-hop cost
  + cross-reference cost
  + hidden-import cost
  + status-ambiguity cost
  + replay-friction cost
```

A least-action pass tries to reduce `J` without reducing truthfulness. If an
edit reduces `J` by hiding an import or weakening a falsifier, it is not a
least-action edit. It is compression by debt.

Stationary condition, stated non-numerically:

> After the pass, no local rewrite should make the document easier to recover
> without either adding real redundancy or weakening the claim boundary.

This deliberately uses the mature "stationary action" reading rather than the
folk "always shortest path" reading. The goal is not minimum word count. The
goal is a stable explanatory path.

## Meta-Generality Rule

Sundog generality must be earned at the role level, not at the slogan level.

Two lanes instantiate the same general pattern only if a reader can map:

```text
B -> Phi -> T(Phi) -> A
```

and also see the lane-specific:

```text
I, F, R
```

without erasing what makes the lanes different.

The dangerous failure mode is metaphorical overreach: "halo," "shadow,"
"least action," "certificate," and "field" become theme words that make the
portfolio sound unified while making each claim harder to audit. The good
failure mode is a typed map: "this lane shares the `Phi -> T -> A` shape, but
its import is physical optics rather than hardness; its falsifier is detector
tooling rather than a capacity threshold."

## Why Snell Belongs Here

Snell's law is the legibility mascot for this roadmap because it is a
translation success:

- ray optics gives a local angle law;
- Fermat gives a global stationary-time rule;
- the two describe the same path from different coordinates.

The public lesson is not that light has foresight. It is that a well-chosen
functional can make a scattered set of local phenomena read as one rule.

Sundog needs the same kind of translation discipline. The portfolio already
contains several coordinate systems:

- photometric control: probe, response, track;
- halo geometry: body, shadow, forward generator, inverse boundary;
- Lean certificates: deductive core, imported wall, axiom surface;
- chat claim discipline: prompt, retrieval trace, answer boundary, refusal;
- hard-math lanes: substrate, projection, receipt, null or promotion.

This ledger asks whether those can be made mutually legible without pretending
they are one theorem already.

## Current Pain Point

The immediate pain point is the author's inability to grok recent papers or
certificate work after a short delay, including the public `sundogcert` Lean
surface. That failure is not merely aesthetic. It is an epistemic risk:

- if the author cannot recover the argument, reviewers will not recover it;
- if reviewers cannot recover it, external critique will hit the notation
  layer instead of the load-bearing claim;
- if the notation layer absorbs the critique, the science cannot learn.

This roadmap treats self-legibility after one week as a real test. A document
that only works while the author's working memory is warm has not passed.

## Phase Ladder

### Phase 0 - Scope Lock

Deliverables:

- choose the canonical filename and title (`SUNDOG_V_LEAST_ACTION.md` for now);
- decide whether "Sundog Lagrangian" is public language, internal language, or
  retired language;
- name the first three target artifacts for a least-action pass.

Exit criterion:

- the roadmap names a finite first pass and does not attempt to rewrite the
  whole repo.

Decided 2026-06-17: filename `SUNDOG_V_LEAST_ACTION.md`; public name
**Least Reader-Action**; "Sundog Lagrangian" promoted to public as the
functional/chart name (scoped per Safe Language). First-pass targets: the
`sundogcert` README and `SUNDOG_V_CERTIFICATE_LEAN.md` (both audited — see
Phase 1), with one recent proof note still to be chosen as the third.

### Phase 0.5 - Source and Attribution Lock

Deliverables:

- verify the Euler-to-Lagrange quotation against a primary or stable secondary
  source;
- file a short note on Fermat, Snell, Maupertuis, Euler, Lagrange, and
  Hamilton sufficient for public attribution;
- add a warning against saying "least" when "stationary" is the honest term.

Exit criterion:

- public epigraph and historical paragraph are citation-ready, or the quote is
  kept internal.

### Phase 1 - Reader-Action Audit

Target artifacts, proposed:

- [`SUNDOG_V_CERTIFICATE_LEAN.md`](SUNDOG_V_CERTIFICATE_LEAN.md);
- the public `sundogcert` README / theorem map;
- one recent paper or proof note that the author finds hard to reload.

Audit questions:

- Can a cold reader name `B`, `Phi`, `T`, `A`, `I`, `F`, and `R`?
- Where do they first need private context?
- Which symbol or file path causes the most rereading?
- Which import is load-bearing but visually quiet?

Exit criterion:

- each target artifact has a one-page reader-action map and a ranked edit list.

First pass filed 2026-06-17:
[`least_action/PHASE1_SUNDOGCERT_READER_ACTION_AUDIT.md`](least_action/PHASE1_SUNDOGCERT_READER_ACTION_AUDIT.md)
covers two of the three targets (README + certificate ledger). It already earned
one catch: the public docs disagree on their own headline count — the README and
`METHOD.md` say seven worked examples / six kinds of math, while
`SUNDOG_V_CERTIFICATE_LEAN.md` still says six / five and omits `AuditCost`
(the seventh, landed 2026-06-10/11). The `B` and `F` coordinates are the weak
ones at the front door; `I` and `R` already pass. Ranked edit list lives in the
audit file; the Euler-pass rewrites are Phase 2.

### Phase 2 - Euler Pass

An Euler pass is a rewrite whose purpose is not polish but recovery. It adds
the missing rungs between intuition and formal mechanism.

Required blocks for each target artifact:

- "What is varied?"
- "What is held fixed?"
- "What condition survives?"
- "What is imported?"
- "What would falsify or demote this?"
- "What can be checked without trusting the author?"

Exit criterion:

- the revised artifact can be summarized by the author after one week without
  reopening private scratch notes.

### Phase 3 - Coordinate Chart Template

Deliverables:

- a reusable `B/Phi/T/A/I/F/R` template for new `SUNDOG_V_*` ledgers;
- a small example filled out for at least three different substrates;
- guidance for when the chart must be omitted because it creates false unity.

Exit criterion:

- the template improves at least two documents and is rejected explicitly for
  at least one document where it would be misleading.

### Phase 4 - Meta-Generality Test

Deliverables:

- select three claims currently described as cross-substrate;
- map them through the coordinate chart;
- classify each as one of:
  - same mechanism;
  - same role structure, different mechanism;
  - metaphor only;
  - false friend / misleading analogy.

Exit criterion:

- at least one tempting analogy is demoted if the map does not hold.

### Phase 5 - Snell/Fermat Public Worked Example

Deliverables:

- a short, diagram-first explainer showing how Snell's law and Fermat's
  principle describe the same refraction path;
- a companion note translating the example into the Sundog reader-action
  discipline;
- an explicit "not evidence" banner so the optics analogy does not inflate the
  agent-control claim.

Exit criterion:

- the worked example makes the legibility method clearer without becoming a
  new public theorem claim.

### Phase 6 - Public Surface Gate

No public page or `site-pages.json` entry should be added until the SEO/social
readiness rules are satisfied.

Deliverables:

- title, description, OG/Twitter metadata plan;
- designed 1200x630 `og:image`;
- JSON-LD plan if promoted;
- internal link path;
- post-deploy validator checklist.

Exit criterion:

- Bucket 1 of `site/SEO_AND_SOCIAL_READINESS_ROADMAP.md` is cleared for any
  public-share page, or the work remains docs-only.

## Falsification Surface

This roadmap fails if:

- the least-action vocabulary adds beauty but not recoverability;
- a cold reader still cannot reconstruct the target artifact's claim after the
  pass;
- the coordinate chart hides imports or flattens domain-specific differences;
- "generality" is asserted by shared adjectives rather than role-preserving
  maps;
- the public Snell/Fermat analogy causes readers to think Sundog has a new
  physics result.

Pre-registered negative:

> If two cold readers can read a least-action-passed artifact and still cannot
> name its import, falsifier, and receipt without help, the pass fails. The next
> action is not more branding; it is a more literal document map.

## Safe Language

Use:

- "least reader-action";
- "the Sundog Lagrangian" / "Lagrangia" — naming the reader-action functional
  `J` and the `B/Phi/T/A/I/F/R` chart, never a physics object;
- "stationary explanatory path";
- "role-preserving generality";
- "coordinate chart for claims";
- "deduction/import/falsifier made visible."

Avoid:

- "Sundog proves a principle of least action";
- "nature optimizes Sundog";
- "all lanes share one *mathematical or physical* Lagrangian" (the term names
  the legibility functional only — promoting it to public copy does not promote
  the physics claim it still disavows);
- "Euler would have approved";
- "Snell proves the agent claim."

## First Concrete Edits

1. Add this ledger to `docs/README.md`. **[done — README.md lines 89-92.]**
2. Add a least-action map to `SUNDOG_V_CERTIFICATE_LEAN.md`.
3. Draft a `sundogcert` theorem map that starts from the reader's question,
   not from the file hierarchy.
4. Write the Snell/Fermat source note before any public-facing essay.
5. Run a one-week self-recovery check on the revised certificate artifact.

## Open Questions

- ~~Should the public name be **Least Action**, **Least Reader-Action**, or
  **Lagrangian Legibility**?~~ **Resolved 2026-06-17: Least Reader-Action;
  "the Sundog Lagrangian" promoted to public as the functional/chart name.**
- Does `J(doc, reader, task)` need real scoring, or is a qualitative audit
  enough for the first pass?
- Which existing document is the best negative control: a file that should not
  accept the coordinate chart?
- Should the template become a reusable doc snippet, or stay prose-only until
  it survives three edits?
