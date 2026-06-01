# External Review Email Draft (Kakeya Reader / Workbench)

> Send to a finite-field incidence geometry / additive combinatorics
> mathematician. Owner fills `[Name]`, `[reviewer specialty signal]`,
> `[link or attachment to packet]`, `[review-page URL]`, and `[Your name]`
> before sending. Three versions below; pick the one that matches your
> relationship with the reviewer.

> **This is a boundary-correctness ask, not a result review.** Sundog has no
> Kakeya result. The whole point of the email is to check that a finite-field
> *reader/workbench* teaches the right boundary without laundering a theorem, a
> toy, or a vocabulary bridge into a stronger claim. Keep the claim-language
> guardrails in `EXTERNAL_REVIEW_PACKET.md` § "Packet Hygiene" intact in whatever
> you actually send.

## Send Prep (staged 2026-06-01)

**Status: not yet sendable — three owner-gated steps remain (below).** The sender
signature is pre-filled as **Jeffery Hughes Jr.**; the review page (`kakeya.html`)
is drafted with the `NOT PEER REVIEWED` banner + `noindex`; and the packet now
carries a concrete "Review Surface — Links" block (review URL + GitHub permalinks).

**Which version to send:** the **Short Version** for a cold reviewer; the
**Slightly Warmer Version** if you have a relationship. Both ask the same three
load-bearing questions; the packet carries all five. **Q3 (the not-regime-2 fence)
is the one to flag** as the call you most want.

**Owner-gated before this can go** (the email links a packet that links these):

1. **Deploy `kakeya.html` unlinked** → `https://sundog.cc/kakeya` (it already
   carries the banner + `noindex`; ensure `kakeya/workbench.html` +
   `kakeya/gallery.html` deploy alongside, and that no public page links to it).
2. **`git push` HEAD** (`ba3d547`) so the packet's GitHub commit permalinks
   resolve — it is not yet on `origin`.
3. **Attach screenshots** of the workbench + gallery (+ a share-card PNG;
   `public/og/kakeya.png` is not yet generated), so the reviewer needn't run a dev
   server.

**Then fill:** `[Name]` (a finite-field incidence / additive-combinatorics
mathematician), `[link or attachment to docs/kakeya/EXTERNAL_REVIEW_PACKET.md]`,
and `[reviewer specialty signal]` in the decline follow-up. The full pre-send
hygiene list is the Owner Fill-In Checklist at the bottom.

---

## Short Version

**Subject:** Quick sanity check: does our finite-field Kakeya *explainer* teach
the right boundary?

Hi [Name],

Could I ask a small sanity check on a finite-field Kakeya reader? This is **not**
a research claim — the opposite. We built an explainer, and I want to be sure it
draws the boundary honestly before anything goes public.

The artifact is three things:

- a reader that walks Dvir's 2008 finite-field theorem (every Kakeya set in
  `F_q^n` has `≥ C_n q^n` points) in a plain "body / shadow" vocabulary — the
  complete direction-shadow forces a low-degree polynomial certificate to vanish
  everywhere, contradiction;
- a tiny `F_q^2` workbench (`q ∈ {5, 7, 11}`) that lets you toggle points and
  watch direction coverage — the lesson being that direction-completeness forces
  a constant fraction of the plane, not a vanishing set;
- a framing that places Kakeya as the **exact-maximal** end of a cross-substrate
  "body-resistance" axis (the shadow reconstructs nothing, yet the body can't
  shrink).

We claim no Kakeya result, no finite-field→Euclidean transfer, and nothing about
the open Euclidean problem. The question is just: **does the explainer teach the
right boundary, or does any of it overreach?** Three things I'd most value your
eye on:

1. Is the body/shadow retelling of Dvir's proof faithful, and is the displayed
   floor `C(q+1,2)` the right *proof* bound to show (and clearly **not** the exact
   planar minimum)?
2. Do we keep the registers honestly separate — finite-field (Dvir) vs Euclidean
   `R^3` (Wang–Zahl 2025, different methods) vs open `n ≥ 4` vs the toy — and is
   the *set*-vs-*maximal-function* distinction handled right?
3. The one I most want challenged: we read Kakeya as **body-resistance** and
   explicitly fence it as **not** a "regime-2 / control-sufficiency" separation,
   because the direction-shadow is control-sufficient only *trivially* (it *is*
   the direction). Is that fence correct and strong enough?

Packet: [link or attachment to docs/kakeya/EXTERNAL_REVIEW_PACKET.md]. It points
to an unlinked `NOT PEER REVIEWED` page, screenshots, and GitHub commit
permalinks, so there's nothing to install or run.

A one-paragraph "safe as written" / "safe with edits, namely X" / "don't publish
until Y" is genuinely enough — explicitly not homework.

Thanks,
Jeffery Hughes Jr.

## Slightly Warmer Version

**Subject:** Small finite-field Kakeya explainer — sanity check to keep us honest

Hi [Name],

I have a small, bounded request if you have the bandwidth. We run an independent
research/education project (Sundog) and we've written a finite-field Kakeya
explainer: a plain-language reader of Dvir's theorem in a "body/shadow" framing,
a little `F_q^2` workbench, and a graphic that places Kakeya at the
maximal-resistance end of a cross-substrate axis we use across the project.

It is deliberately **not** a research claim. What I want is the opposite of an
endorsement: a specialist telling us where the explainer is drawing a boundary
*wrong* — where it risks laundering the finite-field theorem into the Euclidean
problem, reading the 2025 `R^3` result as bearing on open `n ≥ 4`, blurring the
set conjecture with the maximal-function one, or overselling our "body-resistance"
vocabulary as something it isn't.

That last point is the one I'd most like you to push on. We frame Kakeya as a
body-resistance statement and *explicitly* deny it's a "control-sufficiency"
separation — our argument is that the direction-shadow is sufficient for the
direction only trivially, because it *is* the direction. If that fence is wrong,
or too weak, I'd rather hear it from you now than ship it.

Everything is in the packet:
[link or attachment to docs/kakeya/EXTERNAL_REVIEW_PACKET.md]. A 10-minute skim of
the reader plus the workbench screenshots should be enough for the main call; the
citation spine and the claim-boundary fences are one click deeper. No endorsement
wanted — even "this is standard, cite X, and drop the word Y" would be very
helpful.

Specifically, if you have time, three things:

1. Is the body/shadow retelling of Dvir faithful, and is `C(q+1,2)` the right
   proof floor to display (not the exact planar minimum)?
2. Are the finite-field / `R^3` / `n ≥ 4` / toy registers kept properly separate?
3. Is the "body-resistance, not regime-2" fence mathematically fair?

Thanks,
Jeffery Hughes Jr.

## Follow-Up If They Say Yes

Thank you. The most useful path:

1. Read the reader
   ([`KAKEYA_FINITE_FIELD_READER.md`](../KAKEYA_FINITE_FIELD_READER.md)) — §4 is
   the Dvir walkthrough (flagged as standard), §5 the body-resistance placement,
   §6 the binding fences.
2. Skim the lit-pass memo
   ([`KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md)) for the citation spine
   and the forbidden-claims list, and the ledger's body-resistance bridge
   ([`SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)).
3. Glance at the workbench screenshots / share-card PNG and, if curious, the
   `F_q^2` workbench spec
   ([`PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md)).
4. The five questions are in `EXTERNAL_REVIEW_PACKET.md` § "What We Are Asking";
   question 3 (the not-regime-2 fence) is the one I'd most value.
5. Reply with any of:
   - "boundary is fair as written";
   - "safe with edits: …";
   - "don't publish until you fix X";
   - line-specific objections to any sentence that launders finite-field /
     Euclidean / maximal-function / toy / regime-2 language;
   - "cite this standard reference; the framing is in family with X."

I am not asking for a full referee report.

## Follow-Up If They Decline

No worries at all, and thank you for considering it. If there's someone who'd be
a better fit for a quick finite-field Kakeya / additive-combinatorics boundary
check — ideally someone with [reviewer specialty signal, e.g. "polynomial-method
incidence geometry" or "finite-field Kakeya / Nikodym work"] — I'd be grateful
for a pointer.

## Owner Fill-In Checklist Before Sending

- [ ] `[Name]` filled in the salutation and `[reviewer specialty signal]` in the
      Decline follow-up.
- [ ] Packet link points to a stable URL or is attached; **GitHub commit
      permalinks resolve** (the Kakeya lane is pushed) and point at the reader,
      lit-pass memo, workbench spec, core, workbench UI, and gallery — not local
      file paths.
- [ ] The unlinked review page (if used) carries a visible `NOT PEER REVIEWED`
      banner, `noindex`, and no public inbound path; `[review-page URL]` filled.
- [ ] Workbench + gallery screenshots and the share-card PNG attached, so the
      reviewer need not run a dev server.
- [ ] `[Your name]` filled in the sign-off.
- [ ] Subject line matches one of the two registered options above (do not
      freelance it into launch/marketing language).
- [ ] No file or message at this path includes the forbidden phrases per the
      packet's claim-language guardrails: "Sundog has a Kakeya result,"
      "finite-field Kakeya is evidence for the Euclidean conjecture," "Wang–Zahl's
      `R^3` result transfers to open `n ≥ 4`," "the workbench demonstrates a
      dimension lower bound," "Kakeya is a regime-2 / control-sufficiency witness."
- [ ] The ask stays boundary-correctness, not endorsement, and reviewer-only
      sharing of the unlinked page does not get treated as a public launch.
