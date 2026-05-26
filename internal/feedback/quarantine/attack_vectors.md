# Attack vectors

> Anniversary brainstorm staging — Sundog Year 1. The project's posture is
> falsificationist; this file keeps the adversary inside the tent. Each attack is
> stated in its **sharpest** form (steelman), then **what it costs** if true,
> then **the available defense** — and whether that defense is real or only a
> hedge. An attack with no honest defense is a finding, not a debate.
>
> **Status (2026-05-16):** staging, internal. Two attacks (#3 framing, #4 cliff
> confound) are publication-gating for the anniversary push and route to
> [`../../roadmaps/fix_roadmap.md`](../../roadmaps/fix_roadmap.md) (the unsexy fixes, deferred by owner until
> the brainstorm docs are massaged). Voice de-chatted; sharpness retained.

## The deepest one first

### 1. "You rediscovered that inverse problems are ill-posed"

**Steelman.** The cross-substrate "convergence" is an equivocation. "Signature"
means a shadow (core), a halo geometry (atlas), and a hidden-layer subspace
(mesa) — one word, three unrelated objects. *Every* ill-posed inverse problem is
forward-rich / inverse-narrow. Pattern-matching that phrase across three systems
and calling it a discovered law is the regularization-theory equivalent of
noticing that many things fall and announcing gravity.

- **Cost if true:** the gravity frame loses its only empirical support (the
  two-substrate field-shape). The narrow results survive untouched; the *grand*
  claim is reduced to a slogan.
- **Defense (real, but unbuilt):** a *shared mechanism*, not a shared adjective.
  Postulate 1 (coarse-graining / sufficient-statistic-for-control) is precisely
  the claim that the same operator appears in all three. The defense exists only
  once the [proof roadmap](../../../docs/COARSE_GRAINING_PROOF_ROADMAP.md) lands a substrate
  with the *measured* sufficient statistic, not the eyeballed asymmetry. Until
  then this attack is correct and must be conceded in those words.

## Where we already lost a round

### 2. Our strongest experiment is a Goodhart *success* story

**Steelman.** Goodhart is a statement about behavior *under optimization
pressure toward the proxy*. The one place real selection pressure was applied —
Mesa — the controller broke, sharply, at λ≈0.952588. Everywhere else "immunity"
means "we did not push and nothing broke." So the flagship evidence for
Goodhart-resistance is a measured Goodhart *failure*.

- **Cost if unaddressed:** an alignment reviewer weaponizes the project's best
  result against its headline. Fatal for the gravity essay if framed as
  immunity.
- **Defense (real, requires reframing):** stop selling immunity; sell the
  **located cliff**. The contribution is that the failure is sharp, reproducible,
  and predicted by Postulate 2 (immunity is capacity-relative; the cliff is where
  the agent's capacity can invert the channel). A predicted failure boundary is a
  stronger result than an unfalsified immunity claim. This is a *framing* fix,
  free to make, and it must be made before May 19.

## Publication-gating (3 days out)

### 3. The 16× is the result, not a footnote

**Steelman.** "Matches the oracle's terminal accuracy" with a non-significant
Mann–Whitney (U=526, p=0.26, n=30) is *failure to detect a difference*, not
evidence of equivalence — and it costs ~16× slower acquisition. We have not
shown signature control is competitive; we have shown it is asymptotically
not-worse while an order of magnitude slower, and acquisition speed is the only
thing that matters in deployment.

- **Cost if unaddressed:** a reviewer kills the headline sentence on contact;
  it is the single most public claim.
- **Defense (two moves, both owed to [`../../roadmaps/fix_roadmap.md`](../../roadmaps/fix_roadmap.md)):**
  (a) replace "statistically indistinguishable" with a TOST / equivalence test
  against a *pre-registered* margin, or reword to "not detectably different at
  n=30," which is honest and harder to dismiss;
  (b) reframe the 16× via Postulate 3 — if the slowdown is a *predicted
  conserved quantity* (information withheld × excess time ≥ const), the attack
  becomes the headline. (a) is gating and cheap; (b) is research.

### 4. The cliff is suspiciously clean — λ may be a confound

**Steelman.** A phase transition pinned to six digits (λ≈0.952588) is either a
real critical point or a tell that λ is collinear with an effective learning
rate, a gradient-norm ratio, or a reward-scale normalization. If rescaling the
reward gradient moves the cliff predictably, the "cliff" is a relabeled training
instability and the mesa story deflates.

- **Cost if true:** Mesa — the empirical spine of the gravity frame — collapses
  to an optimizer artifact.
- **Defense (a pre-registered experiment, not an argument):** show the cliff is
  **invariant** under a transformation that should not move it and **moves
  predictably** under one that should. This is a concrete, cheap, falsifiable
  test and it is owed *before* the cliff is cited in any anniversary surface.
  Routed to the proof roadmap as a gating control.

**2026-05-17 update.** Large Phase 7 v2 no longer supports a simple monotone
cliff story: the `lambda=0.99` checkpoint recovers by terminal-alignment eval
while `lambda=0.95` and `lambda=0.97` remain weak. This does not erase the
Medium cliff, but it changes the public interpretation. The safer claim is
coherent-signal protection, with signature-pure and possibly reward-pure as
coherent classes, and mixed-signal control as the unstable region when the
mixture creates inference noise. Because the Large eval summary lacks
`old_basin_pref`, the recovery is not yet a mesa-trap escape receipt.

## The long-pole structural ones

### 5. HaloSim grades its own homework

**Steelman.** A 3-photo eligibility set (p2/p7/p13) reached *after* an 8-pass
correction campaign that found a hardcoded CZA bug, validated against renders
from the same simulator, is not "in the wild" — it is a heavily curated
near-fit. And the atlas still draws the parhelic-belt rule that the project's own
work falsified (Spearman ρ≈0.086). The most public surface (sundog.cc) is the
most internally wounded.

- **Cost if unaddressed:** reputational mismatch on launch day; a halo specialist
  finds the drawn-but-falsified rule in minutes.
- **Defense (mostly already done, one gap):** the post-audit hedged language is
  in place across coupled surfaces (`PHASE10_ATTACK_ROADMAP.md`,
  `PHASE10_OPTICAL_REAUDIT_MEMO.md`). The remaining gap is cosmetic-but-loud:
  the atlas should not *render* a rule it has falsified. Routes to
  [`../../roadmaps/fix_roadmap.md`](../../roadmaps/fix_roadmap.md) (deferred), flagged here so it is not lost.

### 6. Garden of forking paths

**Steelman.** 387 commits, ~15 adaptively-chosen phases per workstream, constant
re-spec ("amend Phase 15 smoke to full ladder," "v2 amendments"). Even with
per-phase preregistration, *which phase ran next* was chosen after seeing the
last one. Family-wise error across the whole program is unbounded; the surviving
claims are the survivors of an adaptively pruned search.

- **Cost if true:** the strongest *meta*-attack — it does not refute any single
  result, it discounts the entire evidentiary stack.
- **Defense (partial, honest):** per-phase pre-registered negatives are real and
  documented (AGENTS.md ▸ "Pre-register the negative"). What is missing is a
  program-level accounting of the decision tree. The honest concession: each
  result stands on its own pre-registered gate; the *program* is hypothesis-
  generating, not a single confirmatory study — which is exactly how
  `../../docs/SCIENTIFIC_CRITERIA.md` already frames it. Do not overclaim past that.

## Standing defense: the Bayesian floor

Per `../../quarantine/scratchpad_brainstorm_notes.md`: **do not** open a `bayes_v_sundog.md`
track. Instead, every Sundog workbench (core, threebody, balance, mines, and any
future substrate such as the wishing-well/vortex candidate) should carry a
**Bayesian-optimal control baseline** alongside the existing oracle and random
baselines.

Why this is the right answer to #3 and #6 at once:

- It replaces "indistinguishable from the oracle" (an absence-of-evidence claim)
  with "fraction of the *information-theoretic optimum* recovered from the
  signature" — a positive, bounded, comparable number that does not depend on the
  adaptively-chosen phase sequence.
- It makes the 16× interpretable: the Bayes-optimal estimator *also* pays a
  variance/time cost for the withheld information; the gap between Sundog and
  Bayes is the real result, and it is the empirical handle for Postulate 3's
  conservation law.
- It is a measurement commitment, not a new doc — it slots into each existing
  roadmap's baseline slate. Tracked as a cross-cutting control in
  [`../../../docs/COARSE_GRAINING_PROOF_ROADMAP.md`](../../../docs/COARSE_GRAINING_PROOF_ROADMAP.md).

---

**Triage**

| # | Attack | Class | Routes to |
|---|---|---|---|
| 3 | 16× / equivalence wording | Publication-gating, cheap | `../../roadmaps/fix_roadmap.md` (deferred) |
| 4 | λ confound | Publication-gating, one experiment | proof roadmap (gating control) |
| 2 | Goodhart success story | Framing, free, do before 5/19 | `../../theory/postulations.md` ▸ Post. 2 |
| 5 | Atlas draws falsified rule | Cosmetic-loud | `../../roadmaps/fix_roadmap.md` (deferred) |
| 1 | "Inverse problems are hard" | Deep; defended only by Post. 1 | proof roadmap |
| 6 | Forking paths | Meta; conceded honestly | `../../docs/SCIENTIFIC_CRITERIA.md` |

**Cross-references**

- The "breaks when" lines in [`analogies.md`](../../theory/analogies.md) are the seeds of
  attacks 1, 2, and 5.
- Defenses 1–4 are not arguments; they are work items in
  [`../../theory/postulations.md`](../../theory/postulations.md) and
  [`../../../docs/COARSE_GRAINING_PROOF_ROADMAP.md`](../../../docs/COARSE_GRAINING_PROOF_ROADMAP.md).
- Scope guardrails: `../../docs/SCIENTIFIC_CRITERIA.md`,
  `../../docs/presentation/claims-and-scope.md`.
