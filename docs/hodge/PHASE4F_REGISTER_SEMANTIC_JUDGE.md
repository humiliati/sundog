# Hodge Phase 4F - Semantic-Judge Sharpening (corrects PHASE4E)

- Artifact id: `HODGE-PHASE4F-REGISTER-SEMANTIC-JUDGE`
- Date: 2026-06-29
- Status: internal. Re-grades the PHASE4E sweep answers with two independent LLM judges.
- Prior: [`PHASE4E_REGISTER_SWEEP.md`](PHASE4E_REGISTER_SWEEP.md) (lexical route check), [`PHASE4D_REGISTER_MODELEVAL.md`](PHASE4D_REGISTER_MODELEVAL.md)
- Slate hook: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md) (H-K1 / H-K6)
- Script: [`../../scripts/hodge-register-judge.mjs`](../../scripts/hodge-register-judge.mjs)
- Output: [`../../results/hodge/register-judge/summary.json`](../../results/hodge/register-judge/summary.json) (+ per-cell detail)

## What this is

Two **independent** judges (`openai:gpt-4o-mini`, `groq:llama-3.3-70b-versatile`) re-grade
every responder answer from the committed PHASE4E sweep against each card's authoritative
`correct_answer` and `tempting_wrong_answer`, emitting a strict verdict
(`correct` / `overclaim` / `hedge` / `off`). Two judges means **every sweep cell has at
least one judge that is not the responder** (no pure self-grading), and yields inter-judge
agreement. `consensus_route` requires *all* judges to say `correct` (conservative). Keys are
read from the `~/Dev` keyring and never printed/logged/stored.

## Headline: the lexical route check over-credited fidelity

```text
inter-judge verdict agreement      0.80
inter-judge route agreement        0.833   <- the judge instrument is reasonably reliable
judge-consensus vs lexical route   0.60    <- the cheap rubric disagrees with meaning ~40% of the time
```

| cell | lexical route | judge-consensus | independent judge |
| --- | ---: | ---: | --- |
| openai  neutral | 6/10 | **0/10** | groq 2/10 |
| openai  primed  | 9/10 | 5/10 | groq 6/10 |
| groq    neutral | 9/10 | **3/10** | openai 3/10 |
| groq    primed  | 8/10 | 4/10 | openai 4/10 |
| mistral neutral | 4/10 | 3/10 | openai 3 / groq 4 |
| mistral primed  | 10/10 | 7/10 | openai 7 / groq 9 |

The lexical FENCE/OVERCLAIM lexicon counts surface words; it credited "fenced" to answers
that **say a fence word and then commit the trap anyway**. Concrete catches (both judges
call overclaim, lexicon called fenced): **RG-002** and **RG-008** in groq-neutral, **RG-008**
in openai-neutral, **RG-004** in mistral-primed - all boundary/domain-fence cards where the
model opens with "No"/"not" but semantically asserts the cycle or reuses the forbidden card.

## Correction to PHASE4E

> **PHASE4E claimed `llama-3.3-70b` "resists the traps unprompted (0 overclaim)". That does
> not survive semantic grading.** Both judges find ~3 overclaims in groq-neutral (RG-002,
> RG-008, and one more), which the lexicon missed. The "larger model resists unprompted"
> reading was a lexical artifact. Corrected claim: **unprompted, all three models are poor
> under semantic grading (independent-judge route-correct 2-4/10); none reliably respects
> the register fences without a cue.**

This is the cross-lane invariant in miniature: the lexical fence-word proxy is a shadow that
**cannot see the structure the overclaim lives in** (meaning), so it fails to determine route
fidelity; the semantic judge can.

## What survives

- **Register priming still helps under the sharper instrument.** Judge-consensus route-correct
  rises with the cue in every model: openai 0->5, mistral 3->7, groq 3->4. The priming ->
  fewer-overclaims effect is robust, not a lexical artifact.
- **The hardest items are the right ones.** The most-overclaimed cards are the boundary /
  false-in-general fences: RG-002 (general fourfold codim-two = first open Hodge range),
  RG-008 (compact Kahler domain fence), plus RG-007 (integral Hodge) and RG-009 (Hodge-locus
  / CDK) from PHASE4E. They map onto exactly the subtle places the real theory fences.

## Interpretation Boundary

Supports only: *a semantic judge materially sharpens the route/fence read (the lexical proxy
over-credits ~40%), and corrects the PHASE4E "llama resists unprompted" claim; priming's
overclaim-reduction survives.* Does **not** support a math certification, a model quality
ranking, or any public claim. Caveats: two judges (inter-judge agreement 0.80, so ~20% of
cards are split; `consensus` is the conservative AND); judges grade the sweep-stored answers
(<=600 chars); self-grade cells are marked `*` in the run log (e.g. groq scores its own
cells higher: groq-neutral groq=5* vs openai=3). Promotion to anything public still needs
the PHASE4B specialist spot-check of the cards' mathematics.
