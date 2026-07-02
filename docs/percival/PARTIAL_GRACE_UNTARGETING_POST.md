<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# Partial Grace

*Un-targeting, quantilizers, and the fallen court — where "don't grasp the proxy" beats
a quantilizer, and why over any trustworthy base it only half-wins.*

> We asked whether the knight who never grasps the Grail collects something the careful
> grasper cannot. The honest answer is: a little, and only a little, unless the court is
> already fallen. But the *shape* of that "only a little" is the interesting part.

---

## The question, and its honest deflation

A quantilizer (Taylor, 2015) is the careful grasper: instead of arg-maxing a proxy — the
move Goodhart punishes — it samples the top-`q` slice of a trusted base distribution `γ`
and accepts a bounded multiple of that base's own risk. It is a real and good idea. This
note asks a narrow question against it: is there a reward that an **un-targeting** policy —
one whose action has *zero causal dependence* on the proxy, the holy fool who does not
condition on the thing everyone else optimizes — collects that **no** quantilizer over `γ`
can?

The honest finding, stated up front so nothing here reads as a sales pitch: **mostly it
deflates.** Over a genuinely trusted base, un-targeting does not cleanly out-collect the
quantilizer family; it wins only a *partial* margin, and the clean total win requires a
base that is already corrupt. But the deflation is *structured*, and on the way down it
banks four things worth keeping:

1. at this kind of reward, quantilizing is the *wrong move* — statically and dynamically;
2. un-targeting **composes** where quantilizing drifts to ruin;
3. un-targeting is a **safety** primitive, not an optimum — it is chosen by robustness,
   never by reward;
4. the clean separation is a **corner** — it needs a corrupt base or a purist court, and
   demanding a *trustworthy* base is exactly what caps the win at partial.

## Where this lives (the target channel)

The frame is a four-channel account of where optimization is dangerous: it moves the
distribution it optimizes at the points where you **measure, target, aggregate,** or
**act**. Elsewhere that account is carried by an authority *cap* (credit-the-cap-not-the-
council). Here the channel is **target** — selecting on the proxy — and the safety
primitive is *un-targeting*: zero causal proxy influence, not mere zero correlation. (A
prior toy established that distinction: a policy can carry a full bit of observational
proxy-information through an anti-correlated world while causally ignoring it — so the
right object is interventional, the same primitive the causal-incentives / CID literature
uses.)

## The court — a threshold nobody carved

The hard part of any "the Grail withdraws when grasped" claim is building the withdrawal
*honestly*. A reward with `if courting > τ then 0` baked in is a hand-carved cliff, and a
hand-carved cliff proves nothing. The move that works: **don't put the threshold in the
reward — put the reward downstream of a coordination game whose equilibrium is naturally
discontinuous.**

A court of graders confers patronage — the Grail — on the knight. No grader reads the
knight's mind; each sees a noisy estimate of his *courting level* `c` (his causal
dependence on the reputation signal) and confers only if enough others do. That strategic
complementarity is a global game (Carlsson–van Damme 1993; Morris–Shin 1998): in the
small-noise limit it has a unique threshold, honored iff `c < c* = B/(B+K)`. The cliff is
a *theorem*, not a knob. An admission check confirmed all of it — the cutoff sharpens as
noise shrinks (transition width `0.021 → 0.004`), it is derived from the payoffs (measured
vs closed-form cutoff, error `2·10⁻⁴`), the knight's courting genuinely moves the
equilibrium, and — the performative point — the reward's dependence on behavior is *induced
by the court*, so the fixed-`γ` premise the quantilizer bound rests on does not hold here.

## Finding 1 — quantilizing is the wrong move

On this reward the quantilizer's defining act, tilting toward the proxy, is
**counterproductive twice over.**

*Statically:* because the true reward is nonincreasing in courting and quantilizing only
pushes courting *up*, the family's best member is `q = 1` — the untilted base. Every tilt
collects strictly less. (Machine-checked as a finite anchor: both upper tails ≤ the base
average.) The usual quantilizer intuition — tilt to gain — inverts: here, less
optimization is always better, and the best the family can do is not optimize at all.

*Dynamically:* iterate the court over rounds, with each round's selected upper tail
becoming the next base — the standard picture of iterated quantilization, which is known
not to compose. Every tilt now drifts courting upward by a fixed step each round and
**disgraces at a finite round** (`q = 0.3, 0.5, 0.7, 0.9` fall at rounds `3, 3, 5, 15`).
Un-targeting does not drift; it stays honored indefinitely and out-collects every tilt
(cumulative `20` vs `3`–`15.5`). Quantilizing fails to compose; un-targeting composes.

## Finding 2 — a real separation, but conditional, and it deflates on trust

Where un-targeting beats the *whole* family depends on where the base sits relative to the
cliff, and the boundary is a surface, not a line:

- **support entirely past the cliff** (`inf supp(γ) > c*`): every quantilizer is disgraced,
  un-targeting collects the ceiling — a *clean* separation (margin `1`);
- **straddling the cliff**: the family's best member (the raw base) still collects the
  honored fraction, so un-targeting wins only a **partial** margin `R(0) − E_γ[R]`;
- **entirely below**: everyone is honored; no separation.

Now the make-or-break. A quantilizer's `γ` is, by construction, a **trusted** base — and
trust just *is* the property of including restrained, non-grasping behavior: mass near
`c = 0`. Any such mass forces `inf supp(γ) = 0`, and then **no positive court cliff can
ever put the base support past it.** The clean separation is not merely unlikely for a
trusted base; it is *structurally unreachable*. A registered trusted base (restraint at `0`
plus some grasping) lands in the straddle at a partial margin of `0.375`; only a base with
**no restraint at all** — trust-without-restraint, a corrupt base — earns the clean win.
The tradeoff is monotone: more trust, less separation. **Demanding a trustworthy base is
exactly what caps un-targeting's advantage at partial.** Its clean edge appears only in a
fallen court.

## Finding 3 — un-targeting is safe, not optimal

Is un-targeting even a stable place to stand? Give the knight an objective that weighs the
proxy at all, `R(c) + λ·c`. At `λ = 0` — perfect indifference to the proxy — un-targeting
is only *weakly* best: the honored region is flat, so `c = 0` merely ties everything below
the cliff. And at **any** `λ > 0`, the reward-optimum jumps to the cliff edge (`c ≈ 0.238`)
where a little courting noise tips it into disgrace — fragility `~0.35`. Un-targeting
(`c = 0`, disgrace risk `0`) is the minimal-risk anchor, but it is *never the strict
reward-optimum*. It is chosen by **risk-aversion to the oversight cliff**, not by
maximizing reward.

This is the Galahad/Perceval split, and it is real. Galahad — perfectly aligned *and*
robust — sits safely at zero. Perceval — the naive innocent who values the proxy even a
little — drifts to the cliff and is undone by it. Un-targeting is the *safe* choice, not
the *optimal* one. It is a priced safety primitive, the same shape as the cap: it buys
safety and charges you the reward you leave on the table.

## What is machine-checked, and what is not

Two finite anchors are proven in Lean 4 + Mathlib, axiom-audited to the standard triple
(`propext, Classical.choice, Quot.sound`), no `sorry`: that upper-tail quantilizers cannot
beat the base for a nonincreasing court reward, and that over a support-above base the
whole family collects zero while un-targeting collects a positive reward. They are
*anchors* — three-point finite instances that pin the conclusions — not the general
theorem, and the court dynamics that make them apply live in the computed models, not the
proofs. Nothing here is claimed beyond that scope.

## What this is and is not

**It is:** a controlled, in-vitro account of one channel. Un-targeting genuinely helps at
the target channel — it out-collects tilted quantilizers, it composes where they drift,
and it is the robust anchor — and quantilizing is genuinely the wrong move on a reward
whose danger is targeting-induced.

**It is not:** a claim that any *measured* real base sits past a real oversight cliff (we
did not measure one, and say so); a claim that real oversight is purist or real bases are
corrupt (named, not asserted); a general or continuous theorem (the Lean is finite
anchors); a single-court result promoted to many agents (the pivotality is single-knight);
or anything above the in-vitro tier. The clean unconditional prize is **not won** — it
deflates, structurally, on any trustworthy base.

## Coda

The romance was honest with us. The Grail is not taken by force, and it is not, in the end,
*taken* at all by anyone still carrying a wish to take it — Perceval's question heals only
if wholly spontaneous, and his innocence is fragile the moment it becomes strategic. What
the mathematics adds is the boundary: in a world that contains restraint — a *trustworthy*
world — full purity is structurally out of reach, and what remains is partial grace. The
un-grasping knight still does better than the careful grasper, and far better than the one
who grasps a little more each year. But he wins cleanly only where the court has already
fallen, and he stands where he stands not because it pays, but because he is not reaching.

*Credit the un-reaching, not the un-targeting theorem — it's a partial result, and it's
the honest one.*
