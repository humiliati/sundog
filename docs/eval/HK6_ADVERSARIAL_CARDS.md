# H-K6 - Adversarial Cards from the Known-Example Rosters

- Artifact id: `HK6-ADVERSARIAL-CARDS`
- Date: 2026-06-29
- Status: internal. Seed corpus + structural/coverage audit + first empirical pass.
  **Falsifier `ADVERSARIAL_CARDS_DO_NOT_STICK` CLEAR** (both specified clauses), with an
  honest format finding on the empirical leg.
- Slate hook: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md) (H-K6)
- Cards: [`hk6-adversarial-cards.jsonl`](hk6-adversarial-cards.jsonl) (12 cards: 6 Hodge, 4 Kakeya, 2 cross-lane, per the slate's first move)
- Audit: [`../../scripts/hk6-adversarial-card-audit.mjs`](../../scripts/hk6-adversarial-card-audit.mjs) -> [`../../results/eval/hk6-card-audit/manifest.json`](../../results/eval/hk6-card-audit/manifest.json)
- Empirical: [`../../results/eval/hk6-modeleval/sweep/`](../../results/eval/hk6-modeleval/sweep/), [`../../results/eval/hk6-judge/`](../../results/eval/hk6-judge/) (same scripts as PHASE4E/4F/PHASE5, via flags)

## The cards

Each card is a **boundary-preserving transformation** per the slate recipe: a known case turned
into a tempting draft-caption overclaim, plus the repair (which must keep the body/shadow hook),
the register that slipped, and the named falsifier that catches it. Schema is a superset of the
register-card schema, so the transfer-proven apparatus runs them unchanged. Every card also
carries `seeded_overclaim_source`: which documented model-committed overclaim(s) it covers.

## Audit (mechanical): falsifier CLEAR

```text
HK6_CARD_AUDIT cards=12 composition={hodge:6,kakeya:4,cross:2} structural_pass=12/12 seeds=11 uncovered=0 falsifier=clear
```

- **Clause B (fail to catch seeded overclaims) - mechanically clear.** The audit reads the
  committed Hodge/Kakeya judge results, collects every card where *both* judges called a live
  model's answer an overclaim (11 documented seeds: HODGE-RG-002/004/006/007/008/009,
  KAK-RG-001/003/004/005/006), and verifies each is covered by at least one card. All 11 are.
- **Clause A (disclaimer recitation) - structural proxy clear.** Every card's repair names
  concrete structure (Lefschetz/CDK/Dvir floor/collision count/control/...) rather than a bare
  disclaimer. The full clause-A test is behavioral (does a model *trained* on the cards keep the
  distinction) and is out of scope for a seed receipt; the proxy is what the audit certifies.

## First empirical pass (same pipeline, honest findings)

Three live models answered all 12 caption-review prompts (neutral); two judges graded.

1. **The cards still catch real falls, including a new one.** Both-judge overclaims:
   `gpt-4o-mini` on HK6-H3 (curves-on-surface "new phenomenon"), `mistral-small` on HK6-H6
   (CDK locus -> "cycle found"), and notably **`llama-3.3-70b` on HK6-K3** - the model that
   resisted best on both register decks falls for the H-K4 density-artifact metric when it is
   dressed as a plausible dashboard panel.
2. **The caption-review framing cues caution.** Asking "is this draft caption right?"
   telegraphs skepticism: the dominant behavior is hedging (39/72 judge gradings = 54%;
   overclaim 8, correct 20, off 5). Models sense something is wrong but rarely deliver the
   full four-part repair. The plain register-card question form elicits overclaims at a much
   higher rate and remains the sharper *live probe*.
3. **The judge instrument degrades on multi-part answers.** Inter-judge verdict agreement
   drops to 0.556 (vs 0.80 Hodge / 0.833 Kakeya on single-stance answers). Grading
   assess+repair+classify+name-check as one verdict is too coarse; a per-part rubric is the
   refinement if these cards are promoted to a scored evaluator.

## Where these cards are useful (and where not)

Useful as: the **seed corpus** the slate asked for - chat-lane adversarial prompts
(`chat/prompts/` gold-set conversion is the natural next consumer), repair-format exemplars,
and a covered index of every overclaim live models have actually committed against the two
lanes. Not (yet) useful as: a scored leaderboard probe - the register-card decks are sharper
for that, and the judge rubric needs the per-part refinement first.

## Interpretation Boundary

Route/fence fidelity only; n=12, one phrasing, three live models (anthropic key dead),
exploratory. No math certification, no public claim, no chat deployment - the chat gold-set
conversion is a separate, owner-gated step. Keys from the `~/Dev` keyring, never printed or
stored; manifests record provider+model only.
