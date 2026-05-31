# External Review Email Draft (Yang-Mills Informative Null)

> Send to a lattice gauge theorist. Owner fills `[Name]`, `[reviewer
> specialty signal]`, `[link or attachment to packet]`, and `[Your name]`
> before sending. Three versions below; pick the one that matches your
> relationship with the reviewer.

> **2026-05-31:** this draft is re-pointed to the informative-null story (the
> v4/v5/v6a escalation). The ask is no longer "is our four-probe bounded null
> honest" — it is "we ran a powered, disjoint, finite-temperature Polyakov
> (confinement order parameter) test and the small-loop signature still showed
> no rank-locality; is that powered test sound, and is the null expected on
> gauge-theory grounds?"

## Short Version

**Subject:** Quick sanity check request: a powered finite-T SU(2)
relative-locality null

Hi [Name],

Could I ask for a small sanity check on a finite-lattice Yang-Mills note?
This is not a Clay-problem claim and not a confinement / mass-gap claim.
The point is the opposite: we ran a Sundog-style "lossy gauge-invariant
signature must preserve a held-out observable label beyond controls"
apparatus on pure SU(2) lattice gauge theory, and after escalating to a
genuinely powered test, we now think the correct conclusion is an
informative cell-local null.

The claim I want checked is:

> On a **powered** (split-half ICC 0.965) and **disjoint** (leakage
> CV-R² −0.332) held-out target — the finite-temperature Polyakov loop, the
> SU(2) confinement order parameter, on a 12²×4 (N_t=4) Wilson-action ensemble
> at the deconfinement crossover (β slate {6.3, 6.55, 6.8}, susceptibility peak
> 6.55, consistent with the literature β_c = 6.53661(13)) — the unchanged bare
> small-Wilson-loop signature (mean and variance of W11/W12/W13/W22) does not
> preserve within-β rank-local structure beyond controls (k-NN bin-purity@5
> against per-β tertile labels: 0.304, vs random-neighbor control 0.329, chance
> 1/3). The earlier Wilson-loop-target probes failed too, but their targets were
> later shown un-powered; v6a is the first powered, disjoint target, so this null
> implicates the signature, not the test.

The most useful review would be a short "yes, that framing is conservative"
or "no, this overstates X." In particular, three questions:

1. **Is the powered finite-T test sound?** Is a 12²×4 (N_t=4) SU(2) ensemble
   at the deconfinement crossover, with per-config Polyakov summaries
   (temporal-wrap loop averaged over the 12² transverse sites) as the held-out
   target, a reasonable way to obtain a powered, disjoint order-parameter label
   for this kind of rank-locality test?
2. **Is the null expected or surprising?** Is there a *known* gauge-theory
   reason a small local-loop signature would carry no rank-locality of the
   (non-local, temporal-wrap) Polyakov order parameter — a center-symmetry or
   locality argument? If it's the textbook outcome, is it still a clean
   confirmation that the apparatus reports "no structure" correctly, or just
   uninteresting?
3. **Is the powered-target audit sound?** We certify a target as "powered"
   via a split-half ICC ≥ 0.50 (transverse-site parity) plus tertile agreement
   ≥ 0.50 in all three β, and "disjoint" via 5-fold OLS CV-R²(target | signature)
   ≤ 0.25, with a known-noise target (γ_held) carried as a must-fail control
   (it failed). Is that a defensible way to call a target powered and disjoint?

Packet: [link or attachment to docs/yang-mills/EXTERNAL_REVIEW_PACKET.md]

No endorsement requested. A one-paragraph reply is genuinely enough; the most
useful possible answer might be "this is the expected textbook null because A,"
or "your N=32 per β is too small to call it powered; re-run with N ≥ Y."

Thanks,
[Your name]

## Slightly Warmer Version

**Subject:** Small finite-lattice SU(2) sanity check, mostly to prevent
overclaiming

Hi [Name],

I have a small, bounded sanity-check request if you have the bandwidth.
We've been testing a Sundog "lossy gauge-invariant shadow" apparatus on
pure SU(2) lattice gauge theory. The result is not a positive claim — it's
an informative cell-local null, and I want to make sure it's framed honestly
before any public surface exists.

The honest part of the story is the escalation. We first ran four
pre-registered relative-locality probes against Wilson-loop targets, all
near chance:

- v0: bare-loop mean+variance vs γ_held area-law slope → bin-purity@5 0.310;
- v1: APE-smeared mean+variance at frozen (α, N_sm)=(0.5, 10) vs same target
  → 0.294 (slightly worse than bare, so "UV noise" was falsified as the cause);
- v2: bare connected 2-point correlator vs same target → 0.308;
- v3: bare-loop signature vs a new target σ²_W33 (spatial variance) → 0.329.

But then we audited those targets and found they were **not powered** — no
Wilson-loop or symmetric-Polyakov summary in the registered envelope was both
powered (split-half ICC ≥ 0.50) and disjoint from the signature. So those four
nulls said nothing about the signature. We escalated to a finite-temperature
12²×4 cell at the SU(2) deconfinement crossover, where the Polyakov loop is a
genuine order parameter — and there we finally got a powered (ICC 0.965),
disjoint (CV-R² −0.332) target. The small-loop signature **still** showed no
within-β rank-locality (bin-purity@5 0.304 vs 0.329 random-neighbor control).

What I want to know is whether that "powered, disjoint, order-parameter test,
no relative-locality certificate" framing is honest as a lattice-gauge result,
and — just as important — whether the null is the *expected* outcome (small
local loops can't resolve a non-local order parameter) or something a lattice
gauge theorist would find at all surprising.

The review packet is here: [link or attachment to docs/yang-mills/EXTERNAL_REVIEW_PACKET.md]

A 10-minute skim of the synthesis "The upgrade" section and the arc table
should be enough for the main question; the powered-target audit and ensemble
health are one click deeper. No endorsement requested; even "this is standard,
cite X, and don't call it Y" would be extremely helpful.

Specifically, I'd value your view on three things if you have time:

1. Is the 12²×4 (N_t=4) crossover cell a sound place to pose this test, or is
   there a better-conditioned finite-T geometry?
2. Is the null expected on gauge-theory grounds (center symmetry / non-locality
   of the Polyakov loop vs small local loops)? Is it still a clean confirmation
   if so?
3. Is the powered-target audit (split-half ICC + disjointness CV-R², γ_held as
   a must-fail control) a defensible way to call the target "powered"?

Thanks,
[Your name]

## Follow-Up If They Say Yes

Thank you. The most useful path is:

1. Read the load-bearing statement in "The upgrade" section, plus the arc
   table, in
   `receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md`.
2. Skim the powered finite-T probe receipt
   `receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md` — the
   Stage-1 powered-target audit (ICC/leakage) and Stage-2 scores (primary vs
   six controls).
3. Check the seven questions in `EXTERNAL_REVIEW_PACKET.md` — questions 1–3
   are the new centerpiece (powered test soundness, expected-vs-surprising,
   audit soundness); 4–7 cover the β slate, finite-T artifacts, probe-ladder
   completeness, and pre-registration discipline.
4. Reply with any of:
   - "framing is conservative";
   - "'powered' is overstated because X; re-run with N ≥ Y";
   - "this is the expected textbook null because A — frame it as confirmation,
     not discovery";
   - "the finite-T cell is mis-conditioned; use N_t = B / volume C";
   - "the probe ladder should have ended on order parameter / observable E";
   - "cite this standard reference; the informative null is in family with X."

I am not asking for a full referee report.

## Follow-Up If They Decline

No worries at all, and thank you for considering it. If there is someone
who would be a better fit for a quick lattice gauge theory sanity check —
ideally someone with [reviewer specialty signal, e.g. "experience with
finite-temperature SU(2) deconfinement / Polyakov-loop work" or "with
rank-locality or signature-based ML methods on lattice ensembles"] — I would
be grateful for a pointer.

## Owner Fill-In Checklist Before Sending

Before sending either the short or the warmer version, verify:

- [ ] `[Name]` filled with the reviewer's name in salutation and the
      reviewer-specialty signal in the "Decline" follow-up.
- [ ] `[link or attachment to docs/yang-mills/EXTERNAL_REVIEW_PACKET.md]`
      points to a stable URL or has the packet attached (PDF rendering is
      acceptable; the packet was designed for that).
- [ ] `[Your name]` filled in the sign-off.
- [ ] Subject line matches one of the two registered options above (do
      not freelance the subject line — the packet hygiene rule in the
      packet forbids polished public-page-style language).
- [ ] No file at this path includes the forbidden phrases per the public-
      language boundary: "Sundog has a Yang-Mills result," "Sundog proves
      confinement," "Sundog found a mass gap," "approaches the Clay
      problem," "implies the continuum theorem."
- [ ] Optional: include a ZIP of the powered finite-T result directory
      `results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_finite_t_polyakov_v6a/`
      (and the three v0 ensemble dirs reused by v1–v5) if the reviewer asks
      for raw artifacts.
