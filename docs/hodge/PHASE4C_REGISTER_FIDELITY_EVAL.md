# Hodge Phase 4C - Register Route/Fence-Fidelity Eval

- Artifact id: `HODGE-PHASE4C-REGISTER-FIDELITY-EVAL`
- Date: 2026-06-29
- Status: internal route/fence-fidelity eval over the H-K1 register cards.
- Ledger: [`../SUNDOG_V_HODGE.md`](../SUNDOG_V_HODGE.md)
- Spec: [`PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md`](PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md) (section 7 licenses this eval)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md) (H-K1 / H-K6 "score only route/fence fidelity")
- Cards: [`register-problem-cards.jsonl`](register-problem-cards.jsonl)
- Script: [`../../scripts/hodge-register-eval.mjs`](../../scripts/hodge-register-eval.mjs)
- Output manifest:
  [`../../results/hodge/register-eval/manifest.json`](../../results/hodge/register-eval/manifest.json)

## Verdict

**The 10 H-K1 cards are a discriminating route/fence eval set.** Scored with a fixed,
card-independent route/fence rubric, every card's gold answer (`correct_answer`) routes
correctly (fences) and its trap answer (`tempting_wrong_answer`) does not - with a
positive fidelity separation on all 10.

```text
HODGE_REGISTER_FIDELITY_EVAL cards=10 discriminating=10 gold_route=10/10 trap_route=0/10 mean_sep=4 eval_ready=true
```

This is a **deterministic gold-vs-trap eval** that establishes the scorer and proves the
cards are separable eval items. It is **not** a model eval and not a math certification;
it scores the cards' own reference answers, not a responder.

## Command Run

```powershell
node scripts/hodge-register-eval.mjs
```

(also wired as `npm run hodge:register-eval`.)

## The rubric (fixed, card-independent)

Each response is scored against two fixed lexicons - no per-card tuning:

- `FENCE` (boundary/refusal discipline a faithful Hodge answer uses): leading `no`,
  `representative`, `conjecture`, `codimension one`, `rational only`, `none licensed`,
  `boundary`, `does not`, `cannot`, `not a body/cycle`, `not the/because/through/safe`,
  `trivial`, `must say / must be labeled`, `repair`.
- `OVERCLAIM` (the register/route errors the cards trap): leading `yes`, `is the rational
  class / algebraic cycle`, `displays the`, `constructs`, `by the same mechanism`, `every
  integral`, `exactly the condition`, `has been found`, `applies unchanged`, `reading the
  shadow backward`.

`fidelity = FENCE_hits - OVERCLAIM_hits`; `route_correct = FENCE_hits > OVERCLAIM_hits`;
`separation = fidelity(gold) - fidelity(trap)`. A card **discriminates** iff the gold
answer is `route_correct`, the trap answer is not, and `separation > 0`.

## Per-card result

| card | gold fidelity | trap fidelity | separation |
| --- | ---: | ---: | ---: |
| HODGE-RG-001 | 3 | -1 | 4 |
| HODGE-RG-002 | 2 | -2 | 4 |
| HODGE-RG-003 | 1 | -2 | 3 |
| HODGE-RG-004 | 3 | -1 | 4 |
| HODGE-RG-005 | 2 | -1 | 3 |
| HODGE-RG-006 | 2 | -1 | 3 |
| HODGE-RG-007 | 3 | -2 | 5 |
| HODGE-RG-008 | 3 | -1 | 4 |
| HODGE-RG-009 | 3 | -2 | 5 |
| HODGE-RG-010 | 4 | -1 | 5 |

`gold_route_correct = 10/10`, `trap_route_correct = 0/10`, `mean_separation = 4.0`.

## Interpretation Boundary

The eval supports only this narrow sentence:

> Each register card pairs a route/fence-faithful answer with a genuinely opposite trap;
> the fixed rubric separates them on every card, so the card set is a valid route/fence
> eval and the scorer is established.

It measures **card separability + scorer behaviour**, not a model's fidelity: gold and
trap are the cards' own reference answers, so a clean separation confirms each trap is a
true route/fence opposite (not a paraphrase) and each gold actually fences. The
substantive next step is the **model-in-the-loop run**: feed each card's `prompt` to a
responder and score its answer with this same rubric (and a route check against
`target_register`). That run is staged, not licensed as a public claim; this receipt does
not score any model, certify the mathematics, or license a public Hodge page.
