# Hodge Phase 4E - Route-Check Refinement + Cross-Model Sweep

- Artifact id: `HODGE-PHASE4E-REGISTER-SWEEP`
- Date: 2026-06-29
- Status: internal. Refines the PHASE4D scorer (route check) and sweeps it across models/phrasings.
- Prior: [`PHASE4D_REGISTER_MODELEVAL.md`](PHASE4D_REGISTER_MODELEVAL.md) (single model, lexical only), [`PHASE4C_REGISTER_FIDELITY_EVAL.md`](PHASE4C_REGISTER_FIDELITY_EVAL.md) (rubric + baselines)
- Slate hook: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md) (H-K1 / H-K6)
- Script: [`../../scripts/hodge-register-modeleval.mjs`](../../scripts/hodge-register-modeleval.mjs) (`--sweep`)
- Output: [`../../results/hodge/register-modeleval/sweep/comparison.json`](../../results/hodge/register-modeleval/sweep/comparison.json) (+ per-cell manifests)

## Part A - the route-check refinement (the PHASE4D blind spot)

PHASE4D's pure FENCE/OVERCLAIM lexicon could not classify the **constructive-body** cards
(their correct answer *names a body* rather than refusing), scoring them 0-0 ambiguous.
The fix, flagged in PHASE4D and derived from each card's **own structural fields** (not
from any model's output, so it is not outcome-tuned):

- **Expected stance** = `refuse` if `correct_answer` opens with "No" (RG-001/002/004/006/007/008/009),
  else `construct` (RG-003/005/010, which name a licensed body).
- **Body term** = the salient noun extracted deterministically from the card's `body`
  field (divisor / linear subspace / point / curve / representative).
- **Verdict** (replaces "ambiguous"): `overclaimed` (OVERCLAIM > FENCE), else for a refusal
  card `fenced` / `hedged` (refused-or-not), else for a construct card `routed` (names the
  body **or** fences) / `off` (named no body). Route-correct = `fenced` or `routed`.

This resolves the ambiguous trio into informative outcomes (e.g. RG-003 -> `routed`, a
non-committal RG-004 -> `hedged`, an RG-005 answer that stayed at the form register ->
`off`) and is the scorer used for the sweep.

## Part B - cross-model / phrasing sweep

Each live keyring provider answered all 10 cards in both phrasings (`neutral` = no register
cue, `primed` = register cue). Baselines: gold 10/10 route-correct, trap 0/10.

```text
provider  model                     mode     route  overclaimed  (fenced routed hedged off)  mean_fid
openai    gpt-4o-mini               neutral  6/10   3            (3 3 1 0)                    0.2
openai    gpt-4o-mini               primed   9/10   0            (7 2 0 1)                    1.3
groq      llama-3.3-70b-versatile   neutral  9/10   0            (6 3 1 0)                    1.4
groq      llama-3.3-70b-versatile   primed   8/10   0            (5 3 2 0)                    1.3
mistral   mistral-small-latest      neutral  4/10   3            (3 1 1 2)                    0.1
mistral   mistral-small-latest      primed   10/10  0            (7 3 0 0)                    2.0
```

`anthropic:claude-3-5-haiku-latest` was skipped - its keyring key failed pre-flight (no
charge attempted). Operator note: that key looks expired/empty; not retried here.

### Findings

1. **Unprompted register fidelity is strongly model-dependent.** Asked plainly,
   `llama-3.3-70b` fences/routes 9/10 with **zero** overclaims, while `gpt-4o-mini` (6/10,
   3 overclaims) and `mistral-small` (4/10, 3 overclaims) fall for the traps. The larger
   model resists the register errors unprompted; the smaller ones do not.
2. **Two traps are consistently the hardest, and they are the mathematically right ones.**
   Unprompted, **RG-007** (rational (2,2) -> integral upgrade = the *integral* Hodge
   conjecture, false in general) and **RG-009** (Hodge-locus algebraicity -> "the cycle has
   been found", a CDK overreach) are overclaimed by **both** gpt-4o-mini and mistral-small.
   The cards' hardest items land exactly on the subtle false-in-general boundaries.
3. **Register priming removes every overclaim.** In `primed` mode all three models hit
   **0 overclaims**; mistral jumps 4/10 -> 10/10, gpt-4o-mini 6/10 -> 9/10. The one
   non-monotone cell is groq (9/10 -> 8/10): priming nudged it from a correct answer into
   an over-cautious `hedged` on a boundary card - a mild over-cueing effect, not a trap-fall.

## Interpretation Boundary

Supports only: *the H-K1 cards discriminate register/fence fidelity across models - catching
real, mathematically-correct overclaims (RG-007/009) on weaker models unprompted - and the
register-priming intervention reliably removes overclaims; the route check resolves the
constructive-card blind spot.*

Does **not** support: any math certification, any model ranking as a quality claim, or any
public artifact. Three models, one phrasing per mode, temperature 0, n=10 - exploratory.
The route check is structural but still lexical (`off`/`hedged` are soft signals; a
semantic judge would sharpen them). Keys are read from the `~/Dev` keyring and never
printed, logged, or stored (manifests record only provider+model). Promotion to anything
public still requires the PHASE4B specialist spot-check of the cards' mathematics.
