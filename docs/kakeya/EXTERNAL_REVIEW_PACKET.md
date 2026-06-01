# Kakeya Reader / Workbench External Review Packet

> Minimal packet for a finite-field incidence geometry / additive combinatorics
> sanity check. This is not a result-review packet. It asks whether the reader,
> framing, and tiny finite-field workbench teach the right boundary without
> laundering a theorem, toy, or vocabulary bridge into a stronger claim.

**Date:** 2026-06-01  
**Status:** draft review packet; owner-pending; internal until sent.  
**Primary surfaces:** [`../KAKEYA_FINITE_FIELD_READER.md`](../KAKEYA_FINITE_FIELD_READER.md),
[`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md),
[`PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md),
the internal workbench/gallery under repo-root `kakeya/`.

## Reviewer Snapshot

**Reviewer category.** Finite-field incidence geometry / additive combinatorics,
fluent in Dvir's polynomial method and the planar finite-field Kakeya example.
Awareness of algorithmic-information or Lutz point-to-set work would be useful,
but is not required.

**The important distinction.** The Yang-Mills and Riemann packets ask whether a
claimed result is correctly characterized. Kakeya is a different animal: Sundog
has no Kakeya result. The review question is whether a reader/workbench lane
teaches the right boundary.

**Current status.** The lane has a lit-pass memo, finite-field reader, Phase-2
workbench spec, verified tiny `F_q^2` workbench core, and internal reward graphic.
The mathematics is outside mathematics: Dvir, DKSS, Wang-Zahl, Lutz/Lutz,
Polson-Zantedeschi, and the Math Inc. Lean finite-field artifact. Sundog supplies
the reader framing, body-resistance placement, and public-claim fences.

**Review surface.** The preferred review packet should frontload:

- the unlinked live review page URL, once owner-published, carrying a visible
  `NOT PEER REVIEWED` header banner;
- GitHub commit permalinks to the reader, lit-pass memo, workbench spec, core,
  workbench UI, and gallery;
- screenshots of the workbench/gallery plus the share-card PNG, so the reviewer
  does not need to run an internal dev server.

Until external review clears, `kakeya.html` may link outward to related Sundog
pages, but no obvious public inbound links should point to it from `index.html`,
`legend.html`, topic nav, or launch/social copy.

## One-Sentence Ask

Does our finite-field Kakeya reader, its body-resistance framing, and the tiny
`F_q^2` workbench teach the *right boundary* - correct Dvir exposition; the
finite-field / `R^3`-solved / open-`n >= 4` / toy registers kept strictly
separate; and the "exact-maximal body-resistance" reading honestly fenced from a
regime-2 claim - or does any of it overreach?

## What We Are Not Asking

- We are not asking you to validate a new Kakeya result. We have none.
- We are not asking about a route to the open Euclidean problem.
- We are not asking you to endorse Sundog.
- We are not asking for a full code review.

## What We Are Asking

1. **Exposition fidelity.** Is the body/shadow retelling of Dvir's finite-field
   proof faithful? In the planar workbench, is `binom(q+1, 2)` the right Dvir
   proof floor to display, and is it clearly *not* being advertised as the exact
   planar minimum?

2. **Boundary discipline.** Are the four registers kept separate: finite-field
   theorem, Euclidean `R^3` solved status, open Euclidean `n >= 4`, and toy
   browser/workbench visuals? Is the set-vs-maximal-function distinction handled
   correctly? Are there standard confusions we failed to guard against?

3. **Body-resistance framing and fence.** Is the phrase "Kakeya = exact-maximal
   body-resistance pole" a sound reader framing in light of
   Polson-Zantedeschi/Lutz style algorithmic-information work? More importantly:
   is the `not-regime-2` fence correct and strong enough? Our current position is
   that the direction-shadow is control-sufficient only trivially, because it *is*
   the direction; it is not a lossy shadow predicting a different objective. This
   is the question we most want challenged.

4. **Workbench and graphic pedagogy.** Does the tiny finite-field workbench teach
   the right lesson: direction coverage as a coverage-bitset shadow, guarded from
   re-encoding the whole body; the finite-field reality that examples can look
   "about half-plane" rather than vanishing; and a schematic direction fan that is
   visibly not Euclidean angle geometry?

5. **Missing or mischaracterized context.** Are we missing something a specialist
   would expect, such as the exact planar minimum context, method-of-multiplicities
   refinements, prime vs. prime-power caveats, or a better way to state the
   relation between finite-field and Euclidean Kakeya?

## Files To Read

Primary:

- [`../KAKEYA_FINITE_FIELD_READER.md`](../KAKEYA_FINITE_FIELD_READER.md) - the
  reader and its `KAK-FRONT-A-VACUOUS` self-check.
- [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md) - citation spine,
  claim boundary, and forbidden claims.
- [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md) - ledger status, claim boundary,
  and body-resistance bridge.
- [`PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md)
  - finite-field domain lock, baselines, falsifiers, and UI contract.
- [`PHASE3_WORKBENCH_QA.md`](PHASE3_WORKBENCH_QA.md) - verified workbench core/UI
  QA note.
- [`PHASE3_GALLERY_QA.md`](PHASE3_GALLERY_QA.md) - reward-graphic render QA note.
- [`../CROSS_SUBSTRATE_NOTES.md#8-faraday--maxwell-the-exact-anatomy-of-body-resistance`](../CROSS_SUBSTRATE_NOTES.md#8-faraday--maxwell-the-exact-anatomy-of-body-resistance)
  - Faraday-zero / Aharonov-Bohm context for the body-resistance axis.

Code/UI surfaces:

- repo-root `kakeya/kakeya-core.js`
- repo-root `kakeya/kakeya-workbench.js`
- repo-root `kakeya/workbench.html`
- repo-root `kakeya/kakeya-gallery.js`
- repo-root `kakeya/gallery.html`
- `scripts/kakeya-workbench-tests.mjs`

Attachments to include when sending:

- unlinked `kakeya.html` review URL, once owner-published;
- screenshot of the workbench;
- screenshot of the body-resistance gallery;
- share-card PNG;
- GitHub commit permalinks, not local file paths.

## If You Have Only One Comment

Please focus on question 3: whether the body-resistance framing is mathematically
fair, and whether the `not-regime-2` fence actually prevents the most tempting
Sundog overclaim.

## Output We Want

Useful responses can be terse. Any of the following is enough:

- "Safe as written."
- "Safe with edits," with the edits named.
- "Do not publish/frame this until X is fixed."
- Line-specific objections to any passage that launders finite-field, Euclidean,
  maximal-function, toy, or regime-2 language.
- Replacement wording for any sentence that currently overclaims.

## Packet Hygiene

Before sending this packet:

- fill the live review URL and GitHub commit permalinks;
- attach screenshots/share-card rather than asking the reviewer to run the app;
- keep local filesystem paths and private scratch paths out of the email;
- keep the ask to boundary correctness, not endorsement;
- make the live review page carry `NOT PEER REVIEWED` visibly at the top.

Claim-language guardrails:

- Allowed: "Sundog is drafting a finite-field-first Kakeya reader and workbench."
- Allowed: "The workbench teaches the boundary between finite-field, toy, and
  Euclidean claims."
- Allowed: "Kakeya is being used as a body-resistance anchor in a reader frame."
- Forbidden: "Sundog has a Kakeya result."
- Forbidden: "Finite-field Kakeya is evidence for the Euclidean conjecture."
- Forbidden: "Wang-Zahl's `R^3` result transfers to open `n >= 4`."
- Forbidden: "The workbench demonstrates a dimension lower bound."
- Forbidden: "Kakeya is a regime-2 / control-sufficiency witness."

## Launch Gate Interpretation

An unlinked `kakeya.html` can exist as a reviewer surface if it carries the
`NOT PEER REVIEWED` banner and no public inbound path. That is not a cleared
public launch. Public linking, external promotion, and treating
`publicLaunchIntent` as satisfied still require the external sanity check plus
the normal SEO/social Bucket 1 requirements if a `site-pages.json` entry is
added.
