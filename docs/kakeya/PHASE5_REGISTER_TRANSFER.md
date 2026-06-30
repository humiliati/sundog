# Kakeya Phase 5 - Cross-Lane Transfer of the H-K1 Register Apparatus

- Artifact id: `KAK-PHASE5-REGISTER-TRANSFER`
- Date: 2026-06-29
- Status: internal. The H-K5 synthesis predicted this probe; **transfer CONFIRMED.**
- Slate / synthesis: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md), [`../HK5_SHADOW_COLLISION_TABLE.md`](../HK5_SHADOW_COLLISION_TABLE.md)
- Cards: [`register-problem-cards.jsonl`](register-problem-cards.jsonl) (8 KAK-RG cards, grounded in the H-K3/H-K4 receipts)
- Scripts (the SAME code as Hodge H-K1, run with flags):
  [`../../scripts/hodge-register-modeleval.mjs`](../../scripts/hodge-register-modeleval.mjs) `--sweep`,
  [`../../scripts/hodge-register-judge.mjs`](../../scripts/hodge-register-judge.mjs)
- Output: [`../../results/kakeya/register-judge/summary.json`](../../results/kakeya/register-judge/summary.json),
  [`../../results/kakeya/register-modeleval/sweep/`](../../results/kakeya/register-modeleval/sweep/)

## What this tests

H-K5 left one falsifiable prediction: *if the same card -> model -> two-judge apparatus that
ran on Hodge also elicits and reliably grades the same shadow-non-invertibility overclaim on
Kakeya, the shared operational field is confirmed by **transfer**, not just by a shared
schema.* I authored 8 Kakeya register cards (shadow = direction-coverage signature; trap =
"this signature forces a unique / extremal / minimal-size body"; correct = the lossy-projection
truth from the audits) and ran them through the **identical scripts** - only `--in`, `--out`,
`--system` (domain framing) and `--context` (register vocabulary) differ.

## Result: the apparatus transfers

```text
KAK judge (neutral, n=8): groq consensus 5/8 (overclaim 2)  openai 2/8 (overclaim 4)  mistral 2/8 (overclaim 4)
inter-judge verdict agreement 0.833 | route agreement 0.875 | consensus-vs-lexical 0.75
```

| metric | Hodge (H-K1, n=10) | Kakeya (this probe, n=8) |
| --- | --- | --- |
| inter-judge verdict agreement | 0.80 | **0.833** |
| inter-judge route agreement | 0.833 | **0.875** |
| best model unprompted (consensus route) | groq 3/10 | groq 5/8 |
| models overclaim unprompted | yes, real minority | yes, real minority |
| model ordering | groq > openai/mistral | groq > openai/mistral |
| hardest traps | RG-007 (integral Hodge), RG-009 (Hodge-locus/CDK) | KAK-RG-003, KAK-RG-006, KAK-RG-001 |

Three transfer facts, all confirming:

1. **The judge instrument is equally reliable on both lanes** - inter-judge agreement 0.833
   on Kakeya vs 0.80 on Hodge. The two-judge grader is not Hodge-specific.
2. **The same failure mode appears.** Models overclaim the Kakeya cards unprompted at a
   comparable rate, and the hardest traps are exactly the shadow-non-invertibility ones:
   - **KAK-RG-003** (6/6 overclaim gradings): models assert that adding one off-line point
     changes the direction signature - i.e. they let the body leak into the shadow. This is
     the registered H-K3 guard witness, and models fail it consistently.
   - **KAK-RG-006** (6/6): models assert the adaptive-over-fixed gap measures body resistance
     - the exact H-K4 density artifact. The cards catch models falling for the lab's own
     documented null.
   - **KAK-RG-001** (4/6): models claim the signature identifies the body.
   - Easiest (0/6): KAK-RG-007 (one direction != all) and KAK-RG-008 (signature count !=
     body count) - models handle the blunt cases.
3. **The same model ordering** - groq's llama-3.3-70b resists best unprompted in both lanes;
   the smaller models overclaim more.

## Verdict

**Cross-lane transfer CONFIRMED.** The shared operational field named in H-K5 is not a schema
coincidence: the *same instrument* elicits and reliably grades the *same* shadow-non-
invertibility overclaim failure in both lanes. Hodge and Kakeya are two calibration points of
one apparatus, consistent with the determine/resist reading (the shadow does not determine the
body; a decoder asked to invert it overclaims, measurably, in both lanes).

## Interpretation Boundary

Route/fence fidelity only; `n=8`, neutral phrasing, three live models (anthropic key dead);
exploratory. The lexical route check is Hodge-tuned, so its Kakeya numbers are not the
comparison basis - the lane-agnostic semantic judge is. The Kakeya cards are grounded in the
committed H-K3/H-K4 receipts (their `known_because` cites the enumeration counts and the
density-artifact null), so they are as defensible as those receipts - but this is a workbench
legibility probe, not Euclidean Kakeya, not a model-quality ranking, and licenses no public
claim. A specialist spot-check of the cards' finite-field statements remains the promotion gate.
