# Hodge Reader-Lane External Review Packet

> Minimal packet for an external sanity check by an algebraic geometer or
> Hodge-theory reader. This is **not** a public page and **not** a Clay-problem
> claim. It exists to make it easy for a reviewer to say "yes, this reader
> preserves the mathematics and its boundaries," "no, this commits category
> error X / mis-states known case Y," or "this is only a standard exposition —
> it adds nothing, file the null."

**Date:** 2026-06-01
**Status:** draft reviewer packet for a reading-only lane (no empirical result,
no code, no executable probe).
**Primary artifacts under review:**
[`../HODGE_READER_NOTE.md`](../HODGE_READER_NOTE.md) (Phase 1),
[`../HODGE_PHASE2_BRIDGE_NOTE.md`](../HODGE_PHASE2_BRIDGE_NOTE.md) (Phase 2),
[`PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md`](PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md)
(Phase 3).

## Reviewer Snapshot

Repository map:

- Primary working packet: `docs/hodge/`.
- Main ledger: [`../SUNDOG_V_HODGE.md`](../SUNDOG_V_HODGE.md).
- Lit-pass (citation spine + claim boundary): [`../HODGE_LITPASS_MEMO.md`](../HODGE_LITPASS_MEMO.md).
- Phase 1 reader (the body/shadow reading): [`../HODGE_READER_NOTE.md`](../HODGE_READER_NOTE.md).
- Phase 2 bridge (Faraday/AB vocabulary + non-transfer guard): [`../HODGE_PHASE2_BRIDGE_NOTE.md`](../HODGE_PHASE2_BRIDGE_NOTE.md).
- Phase 3 gallery spec (known-example pre-registration): [`PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md`](PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md).

Why we looked at Hodge:

> The Hodge conjecture is the cleanest high-mathematics anchor for a distinction
> the rest of the Sundog portfolio runs on: a **body** (here, an algebraic cycle)
> versus a **shadow** (here, a rational `(p,p)` cohomology class). The lane is a
> *reader* — it explains established mathematics and one open conjecture in that
> vocabulary, and fences the boundary so the vocabulary is never mistaken for a
> method. It makes no mathematical claim of its own. The review question is
> whether the reading is faithful, commits no category error, and is more than a
> restatement of standard exposition.

Current status:

- Phase 0 lit-pass filed and citation-verified (eight primary anchors resolved by
  title/author/arXiv-ID).
- Phase 1 reader drafted; it *provisionally* clears its own internal red-team /
  vacuity self-check (see reader §7) — an internal judgment only, which is
  exactly what this review is meant to confirm or overturn.
- Phase 2 bridge drafted; it claims the Faraday / Aharonov-Bohm connection
  transfers **caution and vocabulary only**, never method, evidence,
  algebraic-cycle construction, or any "body-resistance / regime-2" framing.
- Phase 3 gallery spec pre-registered (roster, labels, falsifiers); no gallery is
  built and no example is rendered.
- **No Hodge result, algebraicity criterion, cycle construction, or evidence for
  any open case is claimed.** No public page, `site-pages.json` entry, or
  executable probe is live.
- The public surface is blocked until an external sanity check confirms or
  corrects the reading and its boundaries.

Core mapping (the reading under review):

| Sundog concept | Hodge instantiation | Boundary |
| --- | --- | --- |
| Body | algebraic cycle `Z` of codimension `p` | the conjectural object; may or may not exist for a given shadow |
| Body to shadow map | cycle class map `cl(Z) ∈ H^{2p}(X,ℚ)` | always defined; always lands in rational type `(p,p)` |
| Shadow | rational `(p,p)` class, `α ∈ H^{2p}(X,ℚ) ∩ H^{p,p}(X)` | the "right kind of shadow"; rationality + type are necessary, not sufficient |
| The conjecture | is `cl` *rationally onto* the admissible shadows (`Alg^p = Hdg^p`)? | known for `p=1` (Lefschetz `(1,1)`) and dually `p=n-1`; first dimension where the general statement is not known is four |
| Existence pole | Hodge as an *existence / algebraicity* question | explicitly **not** a reconstruction/compression, body-resistance, or regime-2 statement |

Ways to weaken or correct this packet (all welcome):

- Show the reader is **vacuous** — that §3 (register ladder), §5 (existence-pole
  placement), and §6 (fences) say nothing a careful standard Hodge exposition
  does not. Then it should be filed as a null (`HODGE-FRONT-A-VACUOUS`) and kept
  internal.
- Show a surviving **category error** — e.g. the register ladder mislabels an
  arrow, or the existence-pole framing quietly conflates the conjecture with a
  property of the map `cl`.
- Show a **mis-stated known case** — e.g. the "general conjecture settled in
  dimension `≤ 3`, first generally open dimension is four" summary, or the
  codimension `n-1` duality, is not as clean as stated.
- Show a **mislabeled example** in the Phase 3 roster, or a wrong "known
  because" attribution.
- Show the **Faraday/AB bridge over-transfers** despite its disclaimers — that
  invoking electromagnetic Hodge-star / harmonic vocabulary beside the Hodge
  conjecture reads, to a specialist, as implying a method. Then the bridge should
  be demoted to an internal warning paragraph.
- Show the **existence-pole framing is itself misleading** — that "existence /
  algebraicity vs reconstruction" imposes a Sundog dichotomy that distorts how
  Hodge theory is actually organized.

## One-Sentence Ask

Please sanity-check whether our **reader** preserves the rational Hodge
conjecture and its boundaries faithfully — that the body/shadow translation does
not distort Deligne's statement, the register ladder names the right category
errors, the existence-pole placement is fair rather than misleading, the
Faraday/AB bridge is vocabulary-and-caution only, and the known-example roster is
correct and safely labeled — or, if the reader only restates standard exposition,
tell us so we file it as a null and keep it internal.

## What We Are Not Asking

- Not asking for any progress on, or position on, the Clay Hodge conjecture
  itself.
- Not asking whether the lane is evidence for or against any open Hodge case. (It
  is not, by construction.)
- Not asking for endorsement of Sundog, the broader portfolio, or any public
  presentation.
- Not asking for a full referee report, or for adjudication of recent claimed
  "proofs of Hodge" circulating online (quarantined by default in the lit-pass).
- Not asking about integral Hodge, compact-Kähler, motives, Tate, generalized
  Hodge, or periods except where they appear as explicit fences.

## What We Are Asking

Six calls. Q1–Q3 are the centerpiece (does the *reading* preserve the
mathematics); Q4–Q6 are accuracy checks on the bridge, the examples, and the
fences. A short reply is enough.

1. **Translation fidelity.** Does the body/shadow translation — body = algebraic
   cycle; shadow = rational `(p,p)` class; `cl` = body-to-shadow map; the
   conjecture = "is `cl` rationally onto the admissible shadows, `Alg^p = Hdg^p`"
   — faithfully render Deligne's statement, or does the shadow/body framing
   distort it? (Reader §2, §4.)

2. **Register ladder and category errors.** Is the R1–R4 ladder mathematically
   correct: closed form → harmonic representative → cohomology class, with
   rational `(p,p)` selecting the Hodge-class shadow, while algebraic cycles map
   **body → shadow** by `cl` and the reverse direction is exactly the conjectural
   one? Are the three flagged confusions the right ones to warn about — CE1 (a
   harmonic representative is *not* the rational class), CE2 (type `(p,p)` is
   necessary but *not* sufficient for algebraicity; asserting sufficiency *is*
   the conjecture), CE3 (a picture of a form does not exhibit a cycle)? In
   particular: is the attribution right that the `(p,q)` decomposition and
   pure-type question are set by the **complex structure**, with only the
   harmonic representative depending on the Kähler metric? (Reader §3.)

3. **Existence-pole placement** (our most lane-specific claim). We frame Hodge as
   an **existence / algebraicity** question — whether the body exists at all
   behind an admissible shadow (surjectivity of `cl`) — and we state explicitly
   that this is **not** a reconstruction/compression statement and **not** a
   "body-resistance" or "regime-2" separation. Is that a fair, non-misleading
   characterization for a non-specialist, or does it impose a frame that distorts
   the mathematics? (Reader §5.)

4. **Faraday / Aharonov-Bohm bridge.** The bridge note claims the physics
   transfers **caution and vocabulary only — never method, evidence,
   algebraic-cycle construction, or regime-2 / body-resistance framing**. Is even
   the analogy safe, or does invoking the electromagnetic Hodge star / harmonic
   survivor beside the Hodge conjecture risk implying a transfer, to an algebraic
   geometer's eye? If it does, we will demote the bridge to an internal warning.
   (Phase 2 bridge note.)

5. **Known-example roster.** Are the roster entries G1–G5 correct, correctly
   attributed in their "known because" field, and safely labeled — in particular
   G3 (divisors via Lefschetz `(1,1)`, the flagship) and G5 (codimension-two
   classes on a threefold, *rational* known via hard-Lefschetz duality, with the
   integral statement fenced as can-fail)? And are the calls to **defer**
   (abelian varieties, K3, products, cubic fourfolds) and to **hard-exclude the
   general/open fourfold codimension-two case** from the known roster the right
   ones? (Phase 3 gallery spec §4–§5.)

6. **Known-case wording and fences.** Is "Lefschetz `(1,1)` for `p=1`, dually
   `p=n-1` by hard Lefschetz, hence the general conjecture is settled for all
   smooth projective varieties of dimension `≤ 3`, while the first dimension
   where the general statement is not known is four (codimension-two on a
   fourfold)" stated correctly? And are the two hard fences accurate as stated —
   *rational ≠ integral* (Atiyah-Hirzebruch obstruction; Kollár integral
   failures) and *projective ≠ compact Kähler* (Voisin counterexample)? (Reader
   §2, §6; lit-pass Track B/C.)

The most useful possible answers look like:

```text
Yes, the reading is faithful and the boundaries are right.
No, this is essentially standard exposition — it adds nothing (file the null).
Category error at: ___ (e.g. the ladder's R2->R3 arrow / the existence-pole framing).
Known case mis-stated at: ___ (e.g. dimension <= 3 is subtler because Z).
Example G_ is mislabeled / wrongly attributed because ___.
The Faraday/AB bridge over-reaches; demote it because ___.
The existence-pole framing is misleading because ___; soften or drop it.
```

## Reviewer Time Budget

- **10 minutes:** read the reader note §1–§5 (what it is/isn't, the statement,
  the register ladder, the body/shadow translation, the existence-pole placement)
  — enough for Q1–Q3.
- **25 minutes:** additionally read the reader §6 fences + §7 self-check, and the
  Phase 2 bridge note (one page) — enough for Q4 and Q6.
- **45 minutes:** additionally skim the Phase 3 gallery roster (G1–G5 + the
  deferred / hard-excluded lists) and the lit-pass Track A–C — enough for Q5 and
  a vacuity judgment.

## Core Claims To Audit

The two load-bearing sentences:

> **Spine.** Hodge is the visible-shadow / hidden-body problem: a rational
> `(p,p)` class is the right *kind* of shadow, but the conjecture asks whether it
> comes from an algebraic cycle — i.e. whether the body-to-shadow map `cl` is
> rationally onto the admissible shadows.

> **Placement.** Hodge is the *existence / algebraicity* pole of the portfolio's
> body/shadow axis — distinct from reconstruction/compression poles, and
> explicitly **not** a body-resistance or regime-2 separation.

If the spine distorts Deligne's statement, or the placement is a misleading
frame, the lane should be revised (or filed as a null) before any public surface
exists.

## Files To Read

Primary (the reading under review):

- [`../HODGE_READER_NOTE.md`](../HODGE_READER_NOTE.md) — Phase 1 reader; the
  register ladder (§3), body/shadow translation (§4), existence-pole placement
  (§5), fences (§6), vacuity self-check (§7).
- [`../HODGE_PHASE2_BRIDGE_NOTE.md`](../HODGE_PHASE2_BRIDGE_NOTE.md) — Phase 2
  bridge; the register-alignment table and the six forbidden transfers.
- [`PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md`](PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md)
  — Phase 3 roster G1–G5, deferred and hard-excluded lists, falsifiers.

Context (not strictly needed for the calls):

- [`../HODGE_LITPASS_MEMO.md`](../HODGE_LITPASS_MEMO.md) — citation spine, the
  canonical Deligne statement, Track B (known cases) and Track C (integral /
  Kähler fences), and the pre-registered negatives.
- [`../SUNDOG_V_HODGE.md`](../SUNDOG_V_HODGE.md) — the lane charter and
  claim-language guardrails.

There is no code, ensemble, or result directory to inspect: this is a reading,
not an experiment.

## If The Reviewer Has Only One Comment

Ask them to answer this:

> A Hodge specialist reading our note would object if either (a) it commits a
> category error or mis-states a known case — confusing forms / harmonic
> representatives / rational `(p,p)` classes / algebraic cycles, or claiming as
> known something that is open — or (b) it is *only* a standard exposition, so
> the Sundog body/shadow framing adds nothing and should be filed as a null. If
> either holds, please name it. If neither, the reader is a faithful,
> appropriately fenced reading that we may cautiously develop toward a public
> educational page.

## Output We Want From Review

Any of:

- "Faithful reading; boundaries are right — OK to develop cautiously toward a
  public educational artifact."
- "Essentially standard exposition; it adds nothing beyond a careful intro —
  file `HODGE-FRONT-A-VACUOUS` and keep it internal."
- "Category error at X; fix or remove before any public surface."
- "Known case mis-stated at Y; correct it (cite Z)."
- "Example G_ is wrong / mislabeled; fix or drop it."
- "The Faraday/AB bridge over-reaches; demote it to an internal warning."
- "The existence-pole framing is a misleading frame; soften or drop it."

## Packet Hygiene

Do not send a polished public page first. Send this packet (or a PDF rendering of
it) and the three artifacts it points to. Any future `hodge.html` page can be
made later only if it carries the same fenced reading and a clear "external
review pending" or "external review returned: <verdict>" banner.

Public-language boundary (from the charter's Claim-Language Guardrails) remains
binding on every output of this review process:

- Allowed: "Sundog is drafting a Hodge reader lane. It asks how the body/shadow
  discipline clarifies the gap between a visible cohomology class and an
  algebraic cycle."
- Allowed: "The Faraday / Aharonov-Bohm bridge supplies vocabulary and caution,
  not a Hodge argument."
- Forbidden: "Sundog is working toward a proof of Hodge."
- Forbidden: "Hodge decomposition in physics transfers to algebraic cycles."
- Forbidden: "A known-example gallery is evidence for open Hodge cases."
- Forbidden: "A visual form/cycle diagram demonstrates algebraicity."

A faithful-reading verdict is **not** "a Hodge result"; it licenses, at most, a
fenced educational artifact. The forbidden phrasing remains forbidden even after
a positive review.
