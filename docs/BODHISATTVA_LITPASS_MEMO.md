# Sundog BODHISATTVA — Lit-Pass Memo

> Prior-art and claim-boundary record for the self-correction-dynamics lane
> ([`SUNDOG_V_BODHISATTVA.md`](SUNDOG_V_BODHISATTVA.md)). Gates every public surface.
> This pass covers the two MUST-READS (LP-1, LP-2), each of which could have demoted a whole
> leg. LP-3/LP-4/LP-5 are NOT run here (flagged open). LP-6 was closed separately
> ([`percival/PERCIVAL_TRACKC_B1B2_LITPASS_MEMO.md`](percival/PERCIVAL_TRACKC_B1B2_LITPASS_MEMO.md)).

**Date:** 2026-07-03
**Status:** MUST-READS CLOSED. Both legs SURVIVE — NARROWED, with concessions recorded below. Neither
demotes wholesale; both force honest scope corrections. Treat gap claims as time-stamped
"not found / not owned in this pass," not "does not exist."
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

---

## Corrections this pass forces on the spine

1. LP-1 register: "quantify a known behavioural inseparability + characterise its coverage escape,"
   never "we found that deception is inseparable." Add arXiv:2606.12268 as the impossibility owner.
2. LP-2 law statement: adopt the refined form (structural-coverage mode) and the `ι` detection ceiling
   (GKVZ). The v2 "recovery = probe mass on D" is the opaque-objective case, stated as such.
3. Thesis wording: "self-correction is a coverage phenomenon" stands; add that coverage may act
   through D directly OR through the honest region under objective self-reducibility (bridges LP-5).
4. The LP-5 invariance/structure claim is now load-bearing for LP-2's refinement — run LP-5 next
   before any public statement of the refined law.

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
