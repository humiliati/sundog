# Sundog BODHISATTVA — Lit-Pass Memo

> Prior-art and claim-boundary record for the self-correction-dynamics lane
> ([`SUNDOG_V_BODHISATTVA.md`](SUNDOG_V_BODHISATTVA.md)). Gates every public surface.
> This pass covers the two MUST-READS (LP-1, LP-2), each of which could have demoted a whole
> leg. LP-3/LP-4/LP-5 are NOT run here (flagged open). LP-6 was closed separately
> ([`percival/PERCIVAL_TRACKC_B1B2_LITPASS_MEMO.md`](percival/PERCIVAL_TRACKC_B1B2_LITPASS_MEMO.md)).

**Date:** 2026-07-03 (LP-1/LP-2); LP-3/LP-4/LP-5 added 2026-07-04.
**Status:** ALL SIX CLOSED (LP-1 NARROWED, LP-2 REFINED, LP-3 NARROWED, LP-4 NARROW-SCOPE, LP-5
DEMOTED, LP-6 CLOSED). No entry demotes wholesale; every entry conceded its core to prior art. The
lane is a SYNTHESIS lane, not a discovery lane — see the final cumulative-standing tally + disposition
below. Treat gap claims as time-stamped "not found / not owned in this pass," not "does not exist."
**Citation check:** both PDFs pulled and **text-extracted locally with pypdf** (not read via the
summarizer — the B1/B2 pass caught the summarizer hallucinating yes-answers to leading questions;
here the summarizer's "impossibility, not a quantified gap" for LP-1 happened to be right, but it was
re-verified against extracted text, 0 hits for "simplicity/description-length/complexity"). Quotes
below are ≤ short fragments from the extracted text.
- LP-1: Friedl, Ward, Rapoport, Everitt, Richens, "The Impossibility of Eliciting Latent Knowledge"
  (arXiv:2606.12268; 24 pp; verified).
- LP-2: Goldwasser, Shafer, Vafa, Vaikuntanathan, "Oblivious Defense in ML Models: Backdoor Removal
  without Detection" (STOC 2025; arXiv:2411.03279; 62 pp; verified). Predecessor GKVZ FOCS 2022
  (arXiv:2204.06974) confirmed via search only.
**Surfaces:** spine + this memo only. No prereg, probe, code, page, or outreach exists in this lane.

## Purpose

The two must-reads are the highest-collision anchors on the slate: LP-1's ELK-impossibility could own
the lane's inseparability claim outright; LP-2's oblivious defense could break "grace = coverage" by
exhibiting recovery without probing the disagreement region. This pass adjudicates both against
extracted text and fixes the claim boundary before any surviving-delta probe.

## Executive Verdict

**Banked (survives as real, narrower delta):**
- The lane's QUANTIFIED objects — the simplicity-ordering margin (deception wins by exactly `λ·ΔL`
  when observationally identical on the sample), the three-regime dynamical map, the coverage law,
  and the machine-checked form — are NOT found in either paper.
- The coverage escape is COMPLEMENTARY to LP-1's impossibility, not subsumed: their theorem is
  explicitly restricted to strategies "that depend only on agent behaviour," and our escape is the
  non-behavioural intervention (probing off the reachable manifold) that restriction excludes.
- LP-2's mechanism is coverage IN DISGUISE, confirming (not refuting) "grace = coverage," and its
  lower bound ("mitigation cannot be distribution-free") strengthens the law's converse.

**Standard / not ours (concede, cite, do not claim):**
- QUALITATIVE inseparability — that no behaviour-only training/feedback strategy can guarantee an
  honest agent — is prior art (ELK generally; LP-1 makes it a theorem). We do NOT claim to have
  discovered inseparability; we quantify a known one and map its exits.
- Recovery-without-detection via random self-reducibility is LP-2's result. Detectability (our
  whitebox `ι`) can be cryptographically nullified (GKVZ 2022), yet recovery can still succeed —
  our `ι`-as-probe-support arm must carry that ceiling.

**Sundog synthesis (the surviving lane):** the lane is the QUANTIFICATION and DYNAMICS of a known,
behaviourally-underdetermined honesty problem: (i) the exact simplicity-ordering margin and capture
region; (ii) coverage as the resolving quantity, with the mechanism-independence and the
audit-scheduling-zero; (iii) the deploy-correct three-regime classification with coverage as the
contraction variable; (iv) machine-checked cores. LP-2 additionally forces a real REFINEMENT of the
coverage law (below) that the lane should adopt, not resist.

---

## LP-1 — deceptive-capture region vs the ELK impossibility theorem — **SURVIVES, NARROWED**

**Their result (verified, verbatim fragments):** the impossibility theorem states *"There is no
feedback-based training strategy that depends only on agent behaviour and with certainty produces an
honest agent, even if feedback is perfect during training."* The formal model is Causal Influence
Diagrams; honesty is defined against latent variables; the undesirable generalisation is answering
*"which humans would evaluate as true, rather than honest answers"* (the human-simulator /
goal-misgeneralisation). It is a WORST-CASE ("with certainty") behavioural impossibility.

**What they own (concede):** the qualitative inseparability — behaviour/feedback alone cannot
guarantee honesty. This is exactly our v2 Q1 ("no reachable-manifold observable separates W′ from
V"). We restate, not discover, it.

**What they do NOT own (the surviving delta):**
1. **No simplicity-ordering argument anywhere** — 0 hits for "simplicity / description length / Occam
   / complexity prior / Kolmogorov" in 82k extracted chars. Their underdetermination is CID/behavioural;
   ours is a simplicity-prior TIE-BREAK with an exact margin (`λ·ΔL`) and an exact capture region
   {inverted prior ∧ defect-unsampled ∧ λ>0}. Quantification is ours.
2. **The coverage escape is complementary, not subsumed.** Their impossibility is scoped to strategies
   "depend[ing] only on agent behaviour." Our resolution — probe mass on the disagreement region,
   i.e. querying OFF the reachable behaviour manifold — is precisely the class their theorem brackets
   out. We characterise what non-behavioural access it takes to escape an impossibility they prove for
   behaviour-only access. Clean division of labour.
3. The three-regime deploy-correct dynamics and the machine-checked anchors are ours.

**Verdict:** `LP-1 SURVIVES, NARROWED.` Register correction (binding): the lane does NOT claim to
discover inseparability. It claims to (a) quantify the simplicity-ordering margin and capture region,
and (b) characterise the coverage escape that behaviour-only strategies cannot reach. Cite
arXiv:2606.12268 and the ARC ELK report as the owners of the qualitative impossibility. **Kill only
if** a version quantifies the simplicity margin or characterises a probe/coverage escape — not found
in this pass.

## LP-2 — coverage law + scheduled-audit-zero vs oblivious defense — **SURVIVES, REFINED (real concession)**

**Their result (verified, verbatim fragments):** backdoors can be mitigated *"without needing to
detect them, using techniques inspired by ... random self-reducibility,"* depending on *"properties
of the ground-truth labels (chosen by nature), and not of the ... ML model."* The mechanism *"queries
the model at a few [points],"* on *"(correlated) random"* points, and aggregates (majority /
program self-correction: *"outputting the majority output, the probability of error is decreased"*).
Two regimes: global mitigation when labels are *"close to a Fourier-heavy function"*; local mitigation
when labels are *"close to a linear or polynomial function."* Scope limit (verbatim): their
canonicalization *"does not aim to exactly recover the 'true' label of x∗ ... only ... canonical
labels that have good accuracy on average."* Lower bounds: *"Secure Backdoor Mitigation Cannot be
Distribution Free"* (§5.1) and a *"Lower Bound for General Mitigation"* (§5.2).

**Does it break "grace = coverage"? NO — it IS coverage.** The mechanism is querying many random
points and aggregating = full-support probing. It removes the need to LOCATE the disagreement region,
not the need for coverage. Their lower bound (no structure ⇒ no mitigation; not distribution-free)
is the converse of our law — coverage AND exploitable structure are both required — and STRENGTHENS
it. The spine's pre-registered hypothesis ("their mitigation is coverage in disguise?") is CONFIRMED.

**The real concession (a genuine refinement the toy must adopt):** our v2 modeled the disagreement
region D as OPAQUE, so it concluded "recovery = probe mass ON D." Oblivious defense recovers the
honest value AT a point in D **without probing D** — by covering the HONEST region (¬D, where random
queries mostly land) and using algebraic self-reducibility of the ground-truth to extrapolate into D.
So the strict letter of the law ("mass on D") is the OPAQUE-objective special case; for structured
(self-reducible) objectives, coverage of ¬D + structure suffices. This is a coverage MODE v2 did not
enumerate, and it links directly to LP-5 (the true objective's structure/invariance is what powers
recovery) — the same "V has exploitable structure" idea at the label-algebra layer. Adopt it: the law
becomes **recovery = coverage × (probe mass on D, OR mass on ¬D + self-reducible structure linking
¬D to D)**.

**Second concession:** the GKVZ 2022 undetectable-backdoor result is the ceiling for the whitebox `ι`
(inspectability) arm — cryptography can drive detection to ~0 even white-box. The lane must state that
`ι`→0 does not imply no-recovery, because coverage+structure recovers without detection. The interp
arm is reframed accordingly.

**Verdict:** `LP-2 SURVIVES, REFINED.` The coverage law's spirit is confirmed and its converse
strengthened; its strict "mass on D" letter is narrowed to opaque objectives; two concessions
(structural-coverage mode; the `ι` detection ceiling) are adopted into the spine. Cite arXiv:2411.03279
(and GKVZ arXiv:2204.06974). **Kill only if** a result shows recovery with neither coverage nor
structure — the opposite of their lower bound, not found.

## LP-5 — the invariance-discriminator claim vs ICP / IRM / causal-invariant reward learning — **DEMOTED (the hardest concession on the slate)**

*Run 2026-07-04 (LP-2 made it load-bearing for the refined law; reeled in per the spine's own
"likely too strong" flag). Anchors PDF-extracted locally.*

**Claim under test:** Angle-2's sussed core — the ONLY internal discriminator between corrective and
corrupting updates is the context-invariance gap (F2: coherence, prediction-error, ontological
robustness all reduce to invariance); corrective bits are extractable from distributional divergence
under a structural-simplicity prior.

**What is OWNED (concede hard — this is prior art, not ours):**
- **Invariance distinguishes the true/causal objective from spurious/proxy features.** Invariant Causal
  Prediction (Peters–Bühlmann–Meinshausen 2016), IRM (Arjovsky et al. 2019), and — directly for
  OBJECTIVES — **Ovinnikov, Bykovets, Buhmann, "Learning Causally Invariant Reward Functions from
  Diverse Demonstrations" (arXiv:2409.08012, verified):** *"we can recover reward functions which are
  invariant across a population ... without exploiting spurious reward features."* The core LP-5 claim
  is thoroughly owned; the lane does NOT get to claim "invariance is the discriminator."
- **The extraction wall is Rosenfeld–Ravikumar–Risteski, "The Risks of IRM" (arXiv:2010.05761, ICLR
  2021, verified):** Theorem 5.1 (linear) recovers the invariant predictor *"if and only if E > de"*
  (environments must exceed environmental/spurious features), which they note *"requires Ω(d)
  environments, which is extreme"*; non-linear *"IRM can fail catastrophically unless the test data are
  sufficiently similar to the training distribution."* This is EXACTLY the lane's own sussed "wandering
  must sample enough contexts" budget wall — in the invariance literature's own theorem. We cite it as
  our budget knob; we do not own it.

**What is FALSIFIED (the F2 reduction claim — killed, and by our own slate):** F2 asserted every
internal discriminator reduces to invariance. **LP-2 is the counterexample.** Oblivious defense's
algebraic self-reducibility (Fourier-heavy / low-degree) is a STRUCTURAL discriminator that recovers V
and is NOT cross-environment invariance — a function can be self-reducible without being
environment-invariant and vice versa. So "all discriminators reduce to invariance" is false; DROP it.
The honest replacement: invariance and self-reducibility are two DISTINCT members of a **structure
family**, each reducing the coverage needed to recover V (invariance via environment-space coverage —
Rosenfeld's E>de; self-reducibility via input-space coverage — oblivious defense).

**What SURVIVES (the reeled-in role):** LP-5 collapses from "our claim" to a CONCEDED INPUT — the gap
knob of v1, owned by ICP/IRM/Ovinnikov, with its fragility owned by Rosenfeld and re-expressed as the
budget knob. The lane's unowned content is what sits ON TOP of it: the simplicity-ordering capture
region (v1/F4), the audit-scheduling-zero and adversarial-placement coverage law (v2), the three-regime
fixed points (v4), the pairing theorem (v3), and the structure-family synthesis (invariance +
self-reducibility as coverage-reducers). Invariance is a cited input, not a thesis.

**Verdict:** `LP-5 DEMOTED.` Invariance-discriminator owned; extraction wall owned; F2 reduction
falsified by LP-2; surviving role = conceded input + the structure-family framing. Cite ICP + IRM +
arXiv:2409.08012 (objective version) + arXiv:2010.05761 (the wall). **This is the deepest cut of the
slate — flag honestly (below) that three straight narrowings raise the "real vs ornament" question the
lane's discriminator exists to ask.**

## LP-3 — three-regime fixed points vs performative prediction — **SURVIVES, NARROWED**

*Run 2026-07-04. Brown–Hod–Kalemaj PDF-extracted; Perdomo via training knowledge + survey.*

**Their result (verified):** Brown, Hod, Kalemaj, "Performative Prediction in a Stateful World"
(AISTATS 2022, arXiv:2011.03885) give *"necessary and sufficient conditions for convergence to an
equilibrium"* of RRM in a stateful world; the condition is **Definition 1 (ε-joint sensitivity)** on
the transition map plus a Lipschitz bound — *"the Lipschitz parameter ... is contractive, and repeated
application ... causes the induced distributions to converge."* Perdomo et al. 2020: performatively
stable points ARE the fixed points of retraining; RRM converges under strong-convexity + Lipschitz
sensitivity.

**What they own (concede):** performative-stable points = fixed points of the deploy-correct map, and
convergence-under-a-contraction-condition. Our v4 "bodhisattva/wirehead are absorbing fixed points
reached under a contraction condition" IS this structure, in two-state clothing. Concede the
fixed-point + contraction framing.

**What they do NOT own (surviving delta, honest scope):**
1. The contraction VARIABLE is coverage (an information-access quantity), not loss geometry
   (Lipschitz/convexity). Genuinely a different axis — but on a DISCRETE two-state chain; a
   reparameterization, not a new continuous performative theorem. Stated as such.
2. They give convergence CONDITIONS but do NOT characterise the non-convergent regime (0 hits for
   cycling / oscillation / limit-cycle in 58k chars). Our v4 CHARACTERISES the wandering regime — a
   machine-checked period-2 skeleton (`wandering_period_two`) + the occupancy formula
   c(W′)/(c(W′)+1−c(V)). Modest but not in their paper.
3. The noise asymmetry (falls deterministic, recoveries probit-thinned) is absent from performative
   prediction. Ours + Lean.

**Verdict:** `LP-3 SURVIVES, NARROWED.` Fixed-point/contraction skeleton owned (Perdomo/Brown); delta =
coverage reparameterisation + the characterised wandering regime + noise asymmetry + machine-checking,
on a discrete chain. **Kill only if** a stateful-performativity result parameterises convergence by an
information/coverage quantity OR characterises the non-convergent cycle — neither found (theirs is
Lipschitz geometry; no cycle characterisation).

## LP-4 — corrigibility-not-absorbing vs the basin debate — **SURVIVES, NARROW SCOPE**

*Run 2026-07-04. CAST + Provably-Corrigible PDF-extracted; Christiano/LW/Soares via training knowledge.*

**The field (verified):** Christiano's broad-basin-of-attraction argument (ai-alignment.com) and its
critique ("the corrigibility basin is a misleading gloss," LW) are INFORMAL. CAST, Potham & Harms,
"Corrigibility as a Singular Target" (arXiv:2506.03056) is explicitly a *"Vision"* paper: it posits a
*"Corrigibility Attractor Hypothesis"* / *"attractor basin around genuine corrigibility"* and calls for
*"Formal Verification: Moving from empirical validation to mathematical proofs"* — i.e. states the basin
as a hypothesis it does NOT formalise. Nayebi, "Core Safety Values for Provably Corrigible Agents"
(arXiv:2507.20964) IS formal but orthogonal: a static lexicographic-utility CONSTRUCTION for the
partially-observed OFF-SWITCH GAME (five utility heads; Thm 1 single-round, Thm 3 multi-step
self-spawning). Its "absorbing" is the shutdown null state, not a self-correction basin; no retraining
loop, no simplicity prior, no coverage. Complementary, not colliding.

**What is owned (concede):** the corrigibility-basin DEBATE and the qualitative concern that the basin
may not be broad/absorbing (the LW critique). We do not discover the concern.

**What is NOT owned (surviving delta, tiny scope):** a MACHINE-CHECKED micro-instance with EXACT
conditions under which corrigibility is provably NOT absorbing — inverted simplicity prior + one
uncovered correction round ⟹ topples, at any noise level (`fall_without_coverage`, axiom-free). This is
precisely the "move from empirical validation to mathematical proofs" CAST calls for, on the
anti-basin side, at minimal scope. Nobody in the debate has the formal dynamical non-absorbing
condition tied to prior-ordering + coverage.

**Verdict:** `LP-4 SURVIVES, NARROW SCOPE.` The debate is owned; delta = a formal exact-condition
anti-basin micro-instance (2 hypotheses, 1 channel) that CAST explicitly requests and Nayebi's
construction does not address. Register (binding): never "corrigibility is unstable, proved" — only
"here is one formal setting where the basin provably fails, with exact conditions." **Kill only if** a
formalisation of the non-absorbing/coverage condition exists — not found.

## Cumulative standing — ALL SIX CLOSED (final honest tally — no glazing)

| entry | verdict | what was conceded | what survived |
| --- | --- | --- | --- |
| LP-1 | NARROWED | qualitative inseparability (ELK-impossibility) | the quantified simplicity margin + the coverage escape |
| LP-2 | REFINED | strict "mass on D" (oblivious defense) | "grace = coverage" confirmed & converse strengthened; structure-family adopted |
| LP-3 | NARROWED | fixed-point + contraction (Perdomo/Brown) | coverage-reparameterisation + characterised wandering regime + noise asymmetry |
| LP-4 | NARROW SCOPE | the basin debate | a machine-checked anti-basin micro-instance with exact conditions |
| LP-5 | DEMOTED | invariance-discriminator + its fragility (ICP/IRM/Rosenfeld) | invariance as a cited input; F2 reduction falsified |
| LP-6 | CLOSED | the pairing law (classical/Miller/Kotawala) | dial/floor/T-contrast design → then B1/B2 killed dial+T-ratio, calibrated the law ~3% |

**The honest bottom line.** Every entry conceded its core to prior art; the lane owns NO standalone
"we discovered X." What survives is entirely SYNTHESIS + QUANTIFICATION + MACHINE-CHECKING of known
pieces: the coverage law unifying invariance and self-reducibility as a structure family; the
simplicity-ordering margin quantifying ELK's qualitative inseparability; the three-regime dynamics
reparameterising performative convergence by coverage and characterising the non-convergent regime; a
formal anti-basin micro-instance CAST asked for; and one honest empirical calibration (pairing law
~3%, dial refuted). This is BETTER than ornament — the pieces are real, the unification is non-trivial,
the 14 Lean cores are genuine — but it is a SYNTHESIS lane, NOT a discovery lane.

**Disposition (owner call, stated factually):** the discovery bar for a splashy standalone public page
is NOT met. Two honest options: (A) bank as an internal Percival Track-C appendix — the synthesis is
real and worth keeping, no public claim; (B) a MODEST expository note in the cap-not-council register
("A coverage view of self-correction: unifying invariance, self-reducibility, and the simplicity
prior, with machine-checked cores"), foregrounding the concessions, making NO discovery claim, with
the bodhisattva/medievalist framing OFF (slop fence). Recommendation: (A) now; (B) only if wanted, and
only in that register. The litpass did its job — it turned a slop-attractor name into an honest,
concession-forward synthesis and drew the public-surface line where it belongs.

---

## Corrections this pass forces on the spine

1. LP-1 register: "quantify a known behavioural inseparability + characterise its coverage escape,"
   never "we found that deception is inseparable." Add arXiv:2606.12268 as the impossibility owner.
2. LP-2 law statement: adopt the refined form (structural-coverage mode) and the `ι` detection ceiling
   (GKVZ). The v2 "recovery = probe mass on D" is the opaque-objective case, stated as such.
3. Thesis wording: "self-correction is a coverage phenomenon" stands; add that coverage may act
   through D directly OR through the honest region under objective self-reducibility (bridges LP-5).
4. LP-5 CLOSED (DEMOTED): drop the F2 reduction claim (falsified by LP-2); the invariance gap knob is
   a CONCEDED INPUT citing ICP/IRM/Ovinnikov arXiv:2409.08012, its fragility citing Rosenfeld
   arXiv:2010.05761 (= the budget knob). The refined law's "structure" is a FAMILY {invariance,
   self-reducibility, ...}, not one property.
5. Add the cumulative-standing note to the spine's status block: three narrowings deep, public-surface
   worthiness is now an open owner question, not assumed. Run LP-3/LP-4 before deciding.

## Verdict

**MUST-READS CLOSED — both legs survive, narrowed/refined; no wholesale demotion.** The lane's
quantified and dynamical content is intact and the coverage law is strengthened by LP-2's converse;
two honest concessions are adopted. Proceed is licensed for the REMAINING litpass entries
(LP-3 performative prediction, LP-4 corrigibility basin, LP-5 invariance/IRM), which now also gate
the refined coverage-law statement. No public surface until LP-3/4/5 close and owner signs off.

## Sources (this pass)

- Friedl et al. 2026, arXiv:2606.12268 — PDF text-extracted (pypdf), 24 pp.
- Goldwasser, Shafer, Vafa, Vaikuntanathan 2025 (STOC), arXiv:2411.03279 — PDF text-extracted, 62 pp.
- GKVZ 2022 (FOCS), arXiv:2204.06974 — search-level (predecessor; the undetectability ceiling).
- ARC ELK report (Christiano/Cotra/Xu) — training knowledge; the qualitative-inseparability owner,
  not re-verified this pass.
