# ARC-AGI Abstraction Pre-Registration

Roadmaps:
[`SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md),
[`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) Candidate 14

Filed: **2026-05-28 (PT)**

Status: **Phase 0 ADMIT -- Phase 1 (ARC grid representation as shadow domain)
unblocked**. The shadow-projection operator, signature decoder, and Sundog
feature scoring are admitted against the registered 36-task subset. The
preregistered cheap-baseline floor is `0/36` exact across `random_valid`,
`identity_copy`, `dsl_lite_v0`, `dsl_lite_v1`, `dsl_lite_v2`, and
`tiny_learned_v0`; any Phase 1+ Sundog result must clear that floor. Kaggle
notebook work and public-evaluation grid inspection remain blocked until
Phase 6.

## Official Anchors

Checked **2026-05-28** against:

- [ARC Prize 2026 overview](https://arcprize.org/competitions/2026): competition
  start March 25, 2026; submissions due November 2, 2026; papers due
  November 8, 2026; results announced December 4, 2026.
- [Paper Prize](https://arcprize.org/competitions/2026/paper): paper
  submissions must link to a Kaggle code submission for ARC-AGI-2 or ARC-AGI-3;
  a high score is not required for eligibility, but the score feeds the
  Accuracy rubric.
- [ARC-AGI-2 track](https://arcprize.org/competitions/2026/arc-agi-2): static
  grid reasoning track; Kaggle notebook submission; no internet during
  evaluation; two predicted outputs per test input; exact-match task score.
- [Official ARC-AGI-2 repo](https://github.com/arcprize/ARC-AGI-2): 1,000
  public training tasks and 120 public evaluation tasks, with private sets held
  out for the competition.

## Current Phase Artifacts

- [`PHASE0_TASK_SUBSET_SPEC.md`](PHASE0_TASK_SUBSET_SPEC.md) -- frozen Phase 0
  work order for task inventory, subset registration, baselines, and evaluation
  leak control.
- [`P0_BASELINES.md`](P0_BASELINES.md) -- Phase 0 inventory/register/baseline
  receipt. Verdict: **PARTIAL ADMIT** because the registered subset is clean,
  but all cheap baselines solve 0/36 exact, triggering the zero-floor caveat.

## Append-Only Rule

Once a phase spec is used to admit or block a run, the body above its
amendments line is frozen. Corrections or refinements must be appended with:

- date and timezone;
- author;
- one-line justification;
- explicit statement of whether the prior verdict changes.

## Public-Language Constraint

Until Phase 3 adjudicates signature sufficiency, public copy may say:

- ARC-AGI abstraction coupling roadmap;
- registered task-subset audit;
- shadow-projection hypothesis for static grid reasoning;
- falsifiable sufficiency test.

Avoid:

- "Sundog solves ARC";
- "human-level abstraction";
- "the 5D subspace is universal";
- any claim that a Kaggle entry validates the theory without the Phase 3
  sufficiency audit.
