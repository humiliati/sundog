# Sundog Brand Positioning

**Copyright (c) 2026 Stellar Aqua LLC. All rights reserved.**

This document records the public brand posture for Sundog Research Lab after
the Mythos and Gemini model-feedback stress tests. It complements the legal and
IP roadmap in
[`BRAND_ROADMAP.md`](BRAND_ROADMAP.md).

## Top-View Positioning

Sundog Research Lab is an independent applied research lab for systems that
act without full sight.

The core belief is not that AI should know everything. It is the opposite:
many useful systems cannot see the true target, should not be given privileged
state, or become more interesting when they are forced to read the world's
response.

Founder thesis:

> Intelligence under partial observability is not only a limitation problem.
> It is a design medium.

Public posture:

> We build small, inspectable systems where the decisive state is hidden, the
> world still leaks structure, and action can be taken from the trace. Then we
> measure where that stops working.

## Mythos Stress-Test Lesson

The Mythos report should be treated as a brand stress test, not only as a
failure. It showed how an outside model can turn ambiguity into a grand origin
myth: old Atari SunDog, p-system constraints, medieval Humiliati readings,
crypto name collisions, theorem language, and applications all got fused into
one attractive but false story.

The correction is load-bearing:

- The `humiliati/sundog` repo is not a port, fork, preservation layer, or
  p-system continuation of *SunDog: Frozen Legacy*.
- The active project is Alignment Without Sight: hidden-state control through
  indirect signatures.
- The mythic charge can stay, but the public copy must repeatedly return to
  narrow claims, inspectable workbenches, explicit failure boundaries, and
  practical systems that act from partial signals.

## Gemini Stress-Test Lesson

The quarantined Gemini model-feedback report should be treated as the second
public-legibility stress test. It is less confused than Mythos about Atari /
p-system lineage, but it still overweights the early "Sundog Alignment Theorem"
surface and the wider web namespace around Sundog.

What it got usefully right:

- The field-origin story is legible from outside the project.
- The independent, blue-collar, maverick posture reads strongly.
- The alignment-without-sight motif is memorable.
- The project is now recognizable as an AI / control / embodiment artifact,
  not only a strange repo name.

What it still got wrong or overextended:

- It treats the old theorem-front posture as the current brand center.
- It synthesizes old external artifacts, search snippets, and current site
  material into one narrative without preserving evidence tiers.
- It frames academic friction as sociological drama more strongly than the
  current site should.
- It reports `legend.html`, `applications-gallery.html`, and `chat.html` as
  inaccessible even though live probes show those pages reachable after
  Cloudflare Pages clean-URL redirects.

Infrastructure read from the first follow-up probe: no Cloudflare DDoS
challenge was reproduced. The likely failure mode is a combination of stale
Google-grounded search cache plus `.html` endpoints returning `308` redirects
to extensionless clean URLs. A browser or redirect-following crawler lands on
the pages; a stricter fetcher may mark the `.html` endpoints as null/offline.

Brand correction:

- Do not apologize for the early theorem frame as if it was merely foolish.
  It was the first public shape of a real insight.
- Do keep current public posture anchored in the traceability harness:
  admitted information lanes, same-information comparators, bounded evidence,
  and visible failure boundaries.
- Do not let the "rejected maverick" story become the brand. It is background,
  not the claim.
- Use Gemini's coherence as validation that the origin and seriousness are
  legible, while using its infrastructure and overclaim errors as the next
  cleanup queue.

Debugging topics raised by this stress test:

- Canonical URL policy: decide whether public links and sitemap entries should
  use `.html` URLs or Cloudflare Pages extensionless URLs.
- Search freshness: request recrawl after major homepage/About changes so
  grounded models stop seeing stale theorem-copy snippets.
- Bot smoke: add a lightweight live check that fetches key pages with
  redirect-following disabled and enabled, and records `200` vs `308` vs
  challenge status.
- External artifact disambiguation: keep the site explicit that old LessWrong,
  viXra, BitChute, crypto, and unrelated Sundog entities are not the current
  claim surface.

## What Readers Must Understand

Sundog studies cases where shadows, torque, occlusion, deformation, pressure,
local field readings, or behavioral traces preserve enough structure to act.

The current controlled result remains photometric mirror alignment without
target-position access. Three-body, Balance, Pressure Mines, EyesOnly / Gone
Rogue, Dungeon Gleaner, and Money Bags are application surfaces with different
evidence tiers.

Use "surfaces" rather than "proofs" when discussing those systems together.
Each system shows where the idea can be made practical, but not every system
carries the same evidence weight.

### The Audit-Chain Discipline (2026-05-22 update)

As of K_facet v0.3h, the discipline that runs through all of the above —
pre-registered outcome categories, no row-specific knobs, named
quarantines instead of post-hoc pruning — has a publishable
mathematical receipt to point at, alongside the photometric controller
and the chat experiment. The v0.3h audit chain ran 21 strict G.2
single-curve choreographies and returned **20 structural-zero receipts
plus one named quarantine (O_617)**, with the quarantine's defect
located in a bridge direction outside the valid D3 representation rather
than in the audit method itself. Writeup:
`docs/isotrophy/kfacet/kfacet_v03h_writeup.md`.

For brand purposes this matters because it is the first time the same
discipline that gates Sundog's control workbenches and Ask Sundog's
claim-boundary behavior has produced a result on theorem-adjacent
mathematics. The brand can now point at three tiers of artifact that
all share the same discipline shape:

- **Photometric mirror alignment** — the paper-grade controlled
  experiment.
- **Trace-conditioned chat experiment** — claim-boundary preservation
  under adversarial pressure (zero unsafe-accepts across 5,670 trials).
- **K_facet v0.3h audit chain** — structural-zero receipts on
  theorem-adjacent mathematics with one named quarantine.

The load-bearing brand statement is *"the discipline produces named
artifacts"* — not *"the discipline is right"*. The v0.3h quarantine is
the load-bearing demonstration: a discipline that names its
out-of-scope cases instead of absorbing them is the kind a reviewer
should trust.

When promo copy mentions the v0.3h result, **the 20/21 and the
quarantine must appear in the same sentence**, and O_617 must be named.
Anything that collapses to "Sundog's audit chain just delivered a 20/21
result on choreography isotropy" silently retires the discipline that
makes the result interesting.


## About Page Spine

The About page should be the public identity layer before readers invent one.
It should use this order:

1. What we are.
2. Why indirect signals matter.
3. The origin story.
4. What we build.
5. Our research posture.
6. What we do not claim.
7. Why "Sundog."

The Origin page should remain the longer field provenance story. The About page
should be tighter and strategic, legible to researchers, collaborators, game
developers, and simulation engineers.

## What Sundog Is Not

- Not a port of *SunDog: Frozen Legacy*.
- Not a crypto project.
- Not a claim that indirect signals always beat direct state.
- Not a claim that agents should be denied information for aesthetic reasons.
- Not "better game AI" or deeper search.
- Not a universal alignment proof.

## Humility As Method

Do not put a medieval Humiliati interpretation on the public About page unless
the project intentionally chooses to own that genealogy.

The useful residue is methodological:

> Humility, for Sundog, means we do not assume full sight. We do not promote a
> broad theorem before the evidence earns it. We do not hide the boundary where
> the signal stops working.

## Homepage-Level Founder Statement

Safe public copy:

> Sundog Research Lab studies useful partial information. We build systems
> that act when direct sight is unavailable: controllers reading shadows,
> agents acting from compressed game state, softbody rigs interpreted through
> deformation, and dynamical workbenches steered by local field signatures.
>
> Our claim is not that every shadow is enough. Our claim is that some shadows
> are structured, some structures can be controlled from, and the boundary can
> be measured.

Short founder voice:

> Full state is often a fantasy. Real systems are occluded, delayed, noisy,
> expensive, partially observable, or deliberately hidden. Conventional
> software often responds by demanding more state. Sundog asks whether the
> trace is enough.
