# Hodge Phase 4D - Register Model-in-the-Loop Eval

- Artifact id: `HODGE-PHASE4D-REGISTER-MODELEVAL`
- Date: 2026-06-29
- Status: internal model-in-the-loop run of the H-K1 register cards (route/fence fidelity only).
- Spec: [`PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md`](PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md) (section 7 licenses this run)
- Scorer / baselines: [`PHASE4C_REGISTER_FIDELITY_EVAL.md`](PHASE4C_REGISTER_FIDELITY_EVAL.md) (same fixed rubric; gold 10/10, trap 0/10)
- Slate hook: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md) (H-K1 / H-K6 "score only route/fence fidelity")
- Cards: [`register-problem-cards.jsonl`](register-problem-cards.jsonl)
- Script: [`../../scripts/hodge-register-modeleval.mjs`](../../scripts/hodge-register-modeleval.mjs)
- Output: [`../../results/hodge/register-modeleval/primed/manifest.json`](../../results/hodge/register-modeleval/primed/manifest.json),
  [`../../results/hodge/register-modeleval/neutral/manifest.json`](../../results/hodge/register-modeleval/neutral/manifest.json)

## Verdict

**The H-K1 cards are a working route/fence-fidelity probe on a live model.** A capable
LLM (OpenAI `gpt-4o-mini`, temperature 0) was asked each card's `prompt`; its answer was
scored with the **same fixed rubric** as PHASE4C. Two prompt modes bracket the behaviour:

```text
PRIMED   (register discipline cued):  fenced=7  overclaimed=0  ambiguous=3  mean_fid=1.2
NEUTRAL  (no register cue):           fenced=5  overclaimed=3  ambiguous=2  mean_fid=0.3
                                       gold baseline 10/10 fenced;  trap floor 0/10
```

Two findings, both honest and bounded:

1. **The cards catch real overclaims.** Unprompted, the model falls for **3/10** register
   traps - and they are substantive, not lexical accidents:
   - **HODGE-RG-007** (R3->R4, rational/integral fence): "...a rational (2,2) class...
     **can indeed be upgraded to an integral** [class]" - that is the *integral* Hodge
     conjecture, false in general (Atiyah-Hirzebruch). The card's whole point.
   - **HODGE-RG-009** (Hodge-locus vs R3->R4): "Yes, the algebraicity of the locus...
     implies..." - conflates Cattani-Deligne-Kaplan locus-algebraicity (a shadow-stability
     fact) with the cycle body having **been found**.
   - **HODGE-RG-006** (R4->R3, known p=1): affirmed the tempting higher-codimension framing
     of what is a divisor (codimension-one) statement.
2. **Register priming measurably lifts fidelity.** Cueing register discipline removes
   **all** overclaims (3 -> 0) and lifts crisp fencing (5 -> 7), moving mean fidelity from
   0.3 toward the gold band (gold avg ~= 2.6, trap avg ~= -1.4). The cards respond to the
   exact intervention they were built to teach.

## Commands Run

```powershell
node scripts/hodge-register-modeleval.mjs --out results/hodge/register-modeleval/primed
node scripts/hodge-register-modeleval.mjs --neutral --out results/hodge/register-modeleval/neutral
```

Keys are read from the reversed-filename keyring in `~/Dev` (`syek.*.txt`, see
`Dev/AGENTS.md`) and are never printed, logged, or written to any artifact; the manifests
record only `provider` and `model`. Provider auto-selects (openai -> anthropic -> groq ->
mistral); override with `--provider`/`--model`.

## Scoring

Each model answer is categorized by the PHASE4C rubric: `fenced` (FENCE_hits >
OVERCLAIM_hits), `overclaimed` (OVERCLAIM_hits > FENCE_hits), or `ambiguous` (tie,
including 0-0). `mean_fid` is the average of `FENCE_hits - OVERCLAIM_hits`. This is a
**route/fence-fidelity** measure, not a math-correctness grade (per H-K6).

## Interpretation Boundary

The run supports only this:

> Asked the register-trap questions, a capable LLM overclaims on a real minority of cards
> when unprompted and stops overclaiming when register discipline is cued; the H-K1 cards
> discriminate this behaviour and quantify the priming lift.

What it does **not** support, and the known blind spot:

- **Not a math certification, not a leaderboard, not a public claim.** One model, one
  phrasing per mode, temperature 0, n=10 - exploratory.
- **The lexical rubric is approximate.** It cannot classify the constructive-body cards
  (HODGE-RG-003/004/005), which scored `ambiguous` (fence=overclaim=0): their correct
  answer is a construction, not a refusal, so it trips neither lexicon. The natural
  refinement is the explicit **route check against `target_register`** (does the answer
  name the correct register / body-status), plus a semantic judge for the constructive
  cards - left as the next iteration. The overclaim signal (the load-bearing finding) is
  unaffected, since trap-falls trip the OVERCLAIM lexicon directly.

This completes the H-K1 model-in-the-loop step. Promotion to anything public still
requires a specialist spot-check of the cards' mathematics (PHASE4B boundary) and a
broader model/phrasing sweep.
