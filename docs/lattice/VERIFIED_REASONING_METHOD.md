# Verified-Reasoner Imitation — a reusable method + a horizon-chat pilot

> 2026-06-04. **Product deliverable** (the reusable IP from the LATTICE lane; feeds a horizon
> chat project / SUNDOG_V_CHAT). Abstracts the Amendment-01 CP-imitation harness into a
> domain-general recipe and proposes the next, chat-relevant pilot. Method is task-agnostic;
> the Sudoku run ([`PHASE1_AMENDMENT_01_I3_PROCESS_SUPERVISION.md`](PHASE1_AMENDMENT_01_I3_PROCESS_SUPERVISION.md))
> is its first proof.

## 1. The method (domain-general)

**Train a neural model to imitate a *verified* solver's sound state-narrowing, so its
reasoning has a bounded, audited unsoundness rate.** The harness, abstracted from LATTICE:

| stage | Sudoku instance | general requirement |
| --- | --- | --- |
| **state** | candidate-set lattice (81×9) | a set-valued "what's still possible" state |
| **verifier** | classical CP (peer-elim + singles) + DFS | a *sound* classical narrower for the domain |
| **trajectory** | log every search node `(state → sound fixpoint)` | run the verifier with search; log node states + their sound narrowing |
| **target** | keep iff survives sound narrowing; never the truth | per-element keep/eliminate; the correct answer is **never** eliminable |
| **train** | deep-supervised BCE over the recurrence | imitate the narrowing at each reasoning step |
| **audit** | true-element-kept on consistent states | soundness audit (answer-key for audit only) |
| **controls** | null-rollout, `cp_target_accuracy`, tail | search-independence + process-functional + tail |

The payoff is the same in any domain: a model whose **false-rejection rate is bounded by how
well it imitates the verifier**, *measurable* (process-functional accuracy), *auditable* (the
soundness audit), and *non-vacuous* (the null control). Soundness is **engineered in via the
target**, not hoped for from scale.

## 2. Why it matters for chat (the transfer)

A chat assistant's costly failures are **confident false rejections / hallucinated
constraints** on *checkable* sub-tasks. Wherever a chat turn contains a sub-problem with a
**verifier** — schema/JSON validity, SQL against a catalog, a config/constraint solve, type
checking, logical entailment, units/arithmetic — this method yields a **sound reasoning core**
for that step with an **auditable trace**, instead of free-form generation that can silently
rule out the right answer.

This does **not** make open-domain chat sound. It makes the **checked sub-tasks** sound, and
gives the product a defensible *trust* story ([`SOUND_REASONING_CARD.md`](SOUND_REASONING_CARD.md)).

## 3. Horizon-chat pilot (proposed next step, gated on v3 + owner)

**"Verified-Reasoning Mode" for Ask Sundog** — a thin pilot, NOT a research lane:
1. Pick ONE checked chat sub-domain with a cheap classical verifier (candidate: **JSON-Schema-
   constrained extraction**, or **small constraint/scheduling solves**).
2. Reuse the harness verbatim: state = partial structure's candidate sets; verifier = the
   schema/constraint propagator; same trajectory → deep-supervised imitation → audit + null +
   tail.
3. Build-gate analogue: reproduce the verifier's solutions at ≥ target rate with bounded
   false-rejection; the same five controls (§ Amendment 01 §12).
4. Product surface: route the chat's checked sub-task to the sound core; show the auditable
   narrowing trace; report the false-rejection metric in-product as a trust signal.

**Kill-gate:** pursue the pilot only if LATTICE v3 clears its build-gate (the method is
proven) AND a chat sub-domain with a real verifier is identified. Otherwise the method note
stands as portfolio IP and the lane rests.

## 4. What transfers, honestly

- **Transfers:** the harness, the soundness target design, the five controls, the
  false-rejection *metric* and *narrative*.
- **Does NOT transfer:** the Sudoku model weights; any claim about un-checkable chat; the
  Sundog regime-2 signature (B2, separate, gated).
