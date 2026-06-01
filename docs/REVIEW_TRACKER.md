# Sundog External-Review Tracker

**Purpose.** One surface for every lane waiting on (or cleared by) an external
sanity check. Each lane has a staged reviewer packet; this doc tracks *what's
open, who can answer it, what the owner must do to send, and what clears when it
lands*. The source of truth for the questions is each lane's packet (linked
below); this doc is the dispatch / decision surface, not a re-statement.

**Last updated:** 2026-06-01.

**Status legend:** `DRAFTED` (packet ready, not sent) · `SENT` (awaiting reply) ·
`RETURNED` (verdict in) · `RESOLVED-IN-HOUSE` (no external review needed) ·
`CLOSED`.

## At a glance

| Lane | Status | Reviewer community | Open Qs | Owner next action |
| --- | --- | --- | --- | --- |
| Navier–Stokes C1 | DRAFTED | PDE / determining-modes / data-assimilation | 4 | Pick reviewer + fill `[Name]`/`[link]`, send (**email staged**) |
| Kakeya | DRAFTED | finite-field incidence / additive combinatorics | 5 | Publish unlinked review URL + permalinks + screenshots, pick reviewer, send |
| Hodge | DRAFTED | algebraic geometer / Hodge theorist | 6 | Fill `[Name]`/`[link]`/`[Your name]`, send |
| Yang–Mills | **SENT** (2026-06-01) | lattice gauge theorist | 7 | Awaiting reply — sent to Dr. Biagio Lucini via LinkedIn DM |
| Riemann | RESOLVED-IN-HOUSE | — | 0 (3 resolved) | None for framing; publish decision is separate |

**The bottleneck is uniform.** The open lanes are *written and waiting*, each
blocked only on "identify a specialist + fill placeholders + send" — to **four
different communities** (no reviewer covers two). The cost is reviewer
identification and dispatch, not drafting. *(Yang-Mills is now out the door — sent
to Dr. Biagio Lucini via LinkedIn DM, 2026-06-01; awaiting reply. The team reports
good peer connections on LinkedIn but no DM replies yet.)*

## The one question underneath all of them

Across reader lanes (Kakeya, Hodge) and empirical lanes (NSE, YM), the
load-bearing call is the same: **is the Sundog framing (regime-2 / body-resistance
/ separation) fenced from overclaim, or is it just ordinary / standard X?** A
"fenced and non-vacuous" in any lane de-risks the portfolio's framing; a "this is
standard / overclaimed" anywhere is the most informative possible negative.

| Review concern | NSE-C1 | Yang-Mills | Hodge | Kakeya |
| --- | --- | --- | --- | --- |
| Overclaim fence (regime-2 / body-resistance bounded?) | Q1–Q2, Q4 | Q2 | Q3 | **Q3 ★** |
| Non-vacuity (more than ordinary / standard / textbook?) | Q4 | Q2 | one-comment (b) | Q1, Q5 |
| Faithful rendering of the established math | Q1–Q2 | Q1, Q3–Q5 | Q1–Q2 | Q1 |
| Methodology soundness (empirical machinery) | Q1–Q3 | Q1, Q3, Q6 | — | Q2, Q4 |
| Pre-reg / anti-p-hunting discipline | implicit | Q7 | — | — |

★ = the lane's own "if you have only one comment" focus.

**Cross-references now live.** The Yang-Mills packet's centerpiece (Q2) rests on
the Faraday Phase 7/8 abelian-bridge note; the Kakeya packet cites
[`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md) §8. The Faraday / Maxwell
work is load-bearing in two pending reviews.

## Per-lane detail

### Navier–Stokes C1 — DRAFTED (active hold)

- **Under review:** the finite-Galerkin **state-insufficient yet
  control-sufficient** `Phi_K` separation — the strong, provisional regime-2
  witness (currently UNPROMOTED).
- **Load-bearing call (Q4):** a genuine state-reconstruction ↔ control-sufficiency
  separation, or **ordinary data-assimilation / observer behavior** that should
  not be made load-bearing? (Q1: is "state-insufficient on the support" fair for
  the twin-state certificate? Q2: is "control-sufficient on `Phi_K`-fibers up to a
  measure-zero decision surface" fair? Q3: is the held-out look-ahead-max quantile
  a defensible regime-portable objective?)
- **Pre-empted objection:** "isn't this just near-determination?" — the
  objective-overlap discriminator (`proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md`)
  shows control-sufficiency does *not* track predictability (Spearman −0.75 / −1.0).
  One honest open puzzle: palinstrophy (predictable but not control-sufficient).
- **Owner-fill:** `[Name]` (reviewer), `[link]` (packet), tailor the
  specialty phrase, timeline; signature pre-filled. **Email staged →**
  [`proof/PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md`](proof/PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md)
  ▸ "Send Prep".
- **Packet:** `proof/PDE_C1_REGIME_GENERALITY_v1.md`,
  `proof/PDE_C1_KNN_CONVERGENCE_CHECK.md`,
  `proof/PDE_C1_TWIN_STATE_CERTIFICATE.md`,
  [`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md).
- **Gate:** paper-facing use + promotion of C1; public surface.

### Kakeya — DRAFTED

- **Under review:** finite-field reader + body-resistance framing + tiny `F_q^2`
  workbench. **No Kakeya result.**
- **Load-bearing call (Q3 ★):** is the **not-regime-2 fence** correct and strong
  enough — is "exact-maximal body-resistance pole" a sound reader framing, or an
  overclaim? (Their position: the direction-shadow is control-sufficient only
  trivially, because it *is* the direction.)
- **Owner-fill:** review page `kakeya.html` drafted (NOT PEER REVIEWED banner +
  `noindex`); the review URL + GitHub permalinks are staged into the packet's
  "Review Surface — Links" block, and the email is staged. **Still owner-gated:**
  deploy `kakeya.html` unlinked → `sundog.cc/kakeya`; push HEAD so the commit
  permalinks resolve; attach workbench + gallery screenshots + share-card PNG; pick
  reviewer.
- **Packet:** [`kakeya/EXTERNAL_REVIEW_PACKET.md`](kakeya/EXTERNAL_REVIEW_PACKET.md),
  [`SUNDOG_V_KAKEYA.md`](SUNDOG_V_KAKEYA.md).
- **Gate:** public `kakeya.html` launch + any inbound links.

### Hodge — DRAFTED

- **Under review:** reading-only — body = algebraic cycle / shadow = rational
  `(p,p)` class; reader + Phase-2 bridge + Phase-3 gallery spec. **No Hodge
  result.**
- **Load-bearing call:** **(a)** a category error / mis-stated known case (forms
  vs harmonic reps vs `(p,p)` classes vs cycles; or claiming-known something
  open), or **(b)** vacuous — *only* standard exposition → file
  `HODGE-FRONT-A-VACUOUS`? (6 Qs: translation fidelity, register ladder / CE1–3,
  existence-pole placement, Faraday/AB bridge, roster G1–G5, known-case wording +
  fences.)
- **Owner-fill:** `[Name]` / `[link]` / `[Your name]` in
  [`hodge/EXTERNAL_REVIEW_EMAIL_DRAFT.md`](hodge/EXTERNAL_REVIEW_EMAIL_DRAFT.md).
- **Packet:** [`hodge/EXTERNAL_REVIEW_PACKET.md`](hodge/EXTERNAL_REVIEW_PACKET.md),
  [`SUNDOG_V_HODGE.md`](SUNDOG_V_HODGE.md).
- **Gate:** any public Hodge educational artifact.

### Yang–Mills — SENT 2026-06-01 (Dr. Biagio Lucini)

- **Status:** **sent to Dr. Biagio Lucini** (lattice gauge theory, Swansea — SU(N)
  deconfinement / Polyakov-loop work, an ideal fit for the finite-T Polyakov
  target) via **LinkedIn DM**, 2026-06-01; awaiting reply. Packet re-pointed to the
  informative-null story 2026-05-31.
- **Under review:** the **informative powered null** — on a powered (split-half
  ICC 0.965), disjoint (CV-R² −0.332) finite-T Polyakov order-parameter target,
  the small-loop gauge-invariant signature carries no rank-locality
  (primary@5 0.3042 vs CTRL_RAND 0.3292). Lane PAUSED.
- **Load-bearing call:** **(a)** is "powered" overstated / the `12²×4`, `N_t=4`
  cell mis-conditioned, or **(b)** is the null the **textbook** outcome (small
  local loops can't resolve the non-local Polyakov order parameter)? (7 Qs:
  powered-test soundness, abelian-boundary explanation, powered-target audit,
  β-slate, finite-T artifacts, probe-ladder completeness, anti-p-hunting.)
- **Note:** Q2 (the new centerpiece) rests on the Faraday Phase 7/8 abelian-bridge
  — `yang-mills/2026-05-31_faraday_abelian_bridge_note.md`.
- **Owner-fill:** pick reviewer + fill; send packet + informative-null synthesis +
  bridge note.
- **Packet:** [`yang-mills/EXTERNAL_REVIEW_PACKET.md`](yang-mills/EXTERNAL_REVIEW_PACKET.md),
  [`SUNDOG_V_YANG_MILLS.md`](SUNDOG_V_YANG_MILLS.md).
- **Gate:** public Yang-Mills generality-gallery card.

### Riemann — RESOLVED-IN-HOUSE (2026-05-30)

- **Was under review:** the bounded null across three lanes (Z2 reflection,
  explicit-formula parity, nonlinear S2). **All three framing questions answered
  in-house:** Q1 by a sine-kernel / GUE control (observed D at z = −0.85), Q2 by a
  parity-identity proof, Q3 by a 100k-ordinate window / unfolding sweep; the
  representation-sector question closed at desk.
- **Remaining:** external review no longer gates the framing. Only optional
  pointer value (is the prime-conjugate-selector reopen a known device? a
  citation?) plus the publish decision, which is separate and owner-owned.
- **Packet:** [`riemann/EXTERNAL_REVIEW_PACKET.md`](riemann/EXTERNAL_REVIEW_PACKET.md)
  (retained as a finished record), [`SUNDOG_V_RIEMANN.md`](SUNDOG_V_RIEMANN.md).
- **Action:** **none for framing** — off the "waiting" list.

## Send log

Append one row per dispatch so the portfolio's review state stays visible.

| Date | Lane | Reviewer (role, not always name) | Channel | Outcome / next check |
| --- | --- | --- | --- | --- |
| 2026-06-01 | Yang–Mills | Dr. Biagio Lucini (lattice gauge theory, Swansea) | LinkedIn DM | Awaiting reply (no reply yet) |
