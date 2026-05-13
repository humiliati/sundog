# Pushable Occluder Pre-Registration

Roadmap: [`SUNDOG_V_PATH.md`](../SUNDOG_V_PATH.md)
Pre-registered: **2026-05-13 (PT)**
Author: maintainer
Status: append-only. Edits below the **Amendments** rule require a timestamp and a written justification. The body above the rule is frozen at pre-registration time.

This is the Phase 0 acceptance artifact for the Pushable Occluder
falsification slate. The roadmap names the canonical path as
`docs/_prereg/pushable_occluder_2026-05-12.md`. This file is the same
artifact under the project's existing `docs/prereg/` directory naming;
the roadmap's path reference should be reconciled at next edit. The
prereg note itself is authoritative for what was committed before code
ran.

## Hypotheses (verbatim from roadmap §Hypothesis)

**H1.** A flat scan/seek/track photometric controller — the same
architecture that succeeds on the Phase-1 mirror-alignment task and the
Phase-2 Option-A static-occluder task — fails to discover a two-stage
plan when alignment requires first pushing an occluder out of the beam
path, because the indirect photometric signal does not expose the
preparatory action as a usable gradient.

**H2 (oracle reachability).** An oracle controller with access to block
geometry can solve the task: push the block out of the beam cone, then
run the standard photometric alignment loop.

**H3 (continuity with prior limit).** The failure mode of the flat
controller on the pushable-occluder task is structurally similar to its
failure mode at tight joint limits — both involve an optimum that lies
beyond a non-photometric prerequisite, not beyond the controller's
expressivity in a vacuum.

If H1 fails (the flat controller solves the task) the verdict is not
`BOUNDARY FOUND`. It is `CONFIRMED` for a stronger claim than the
program currently makes, and the rail card is rewritten. The roadmap's
§Outcome Branching table — superseded by the Pre-Registered Verdict
Template in the restructured roadmap — is the disposition rule, not
this prereg note.

## Expected Failure Mode (frozen prediction, 2026-05-13 PT)

Recorded here so a future-us cannot retro-fit it. This paragraph is
lifted from §Expected Failure Mode of the roadmap at pre-registration
time and timestamped:

- *Flat photometric controller:* fails. Push utilisation low or zero.
  Terminal photometric error high. Time-to-acquisition undefined for
  most seeds. The agent spends its budget optimising mirror angle
  around a local maximum produced by the occluded beam reaching the
  detector through scattering or edge effects.
- *Oracle:* succeeds. Terminal photometric error matches Phase 1.
- *Hierarchical photometric:* plausibly succeeds if the high-level mode
  switch fires. Included for discussion only.
- *Random:* fails noisily.

If the flat photometric controller succeeds on a non-trivial fraction
of seeds, the prediction is wrong and the verdict changes. The
roadmap's verdict template, not this paragraph, decides how the result
is reported. This paragraph records what was expected at
pre-registration time, nothing more.

## Honest Non-Confounds — Enforcement Pointers

Each non-confound from §Honest non-confounds is mapped to the file
where the constraint will be enforced. The audit trail at Phase 2 must
confirm each pointer was honored.

| # | Non-confound | Enforcement site | Verification |
| --- | --- | --- | --- |
| 1 | Beam cone geometry is not adversarial. Block must be pushable out of the cone in fewer push steps than the controller's episode budget. | `env_v2.py` (scene/state assembly: occluder_block initialization, workspace bound, push step accounting). | Phase 1 oracle run on N=64 seeds: ≥ 90% must succeed within episode budget. Recorded in `docs/_results/pushable_occluder_2026-05-12.md` (Phase 2 artifact). |
| 2 | Push effector is not photometric. The pusher must not change the photometric signal by occluding the detector itself or reflecting light. | `optics.py` (the pusher geometry must not appear in `compute_detector_intensities` ray paths or reflectance terms; pusher mesh excluded from optical bodies). | Unit test in Phase 1: with the mirror frozen and the block frozen, sweeping the pusher across its workspace must produce detector readings constant to within sensor noise floor. |
| 3 | Probe schedule is comparable. The flat controller's probe frequency is re-tuned for the 4D action space using the same procedure as Phase 1, not pessimised to guarantee failure. | `agents/photometric.py` (probe-frequency hyperparameter and re-tune procedure) plus tuning notebook entry alongside the prereg. | Phase 2 acceptance: tuning procedure documented step-for-step against Phase-1 procedure; deviations called out by line. |
| 4 | Seeds are matched across controllers. Block initial position, mirror initial pose, and any stochastic optics noise are shared per-seed across all four controllers. | `experiments/run_baseline_comparison.py` seed-handling code (existing matched-seed harness; verify same seed seeds block init and not just mirror init). | Phase 2 acceptance: per-seed seed-trace audit — for at least 4 sampled seeds, confirm flat / oracle / hierarchical / random observe identical block_xy at t=0 and identical optics noise stream. |

If any pointer above is violated at Phase 2 audit time, the result is
unreportable until it is fixed. The fix lands as an amendment to this
file below, with the corresponding code commit referenced.

## Open Decisions Deferred to Phase 1

The roadmap §Open Questions flags four decisions. The two that need
landing during Phase 1 (and recorded as amendments to this file) are:

1. **Push effector embodiment.** Second joint chain vs. shared
   kinematic chain. Decision required before Phase 1 implementation
   begins. The cleaner-implementation argument and the joint-limit
   confound concern are both Phase 1 calls.
2. **Block dynamics realism.** First-order overdamped chosen for clean
   failure attribution. Phase 1 ratifies the parameter values
   (`alpha`, workspace bound, friction surrogate if any) and notes any
   deviation from PHASE2_BLOCKS_DESIGN.md Option B.

The two remaining open questions (multi-block stretch; naming) are not
Phase 0/1 decisions and do not need amendments here.

## Tier Bookkeeping Pending Decision

Phase 4 of the roadmap notes that the Evidence Tier for a falsification
result is undecided ("`Falsification result` row vs. footnote vs. cell
`n/a — falsification slate` per HIGHLIGHTS_RAIL_ROADMAP). The PATH
restructure carries the open question forward; the prereg note records
that no tier-table commitment was made at pre-registration. The card
ships under whichever tier convention is ratified by Phase 4 land time.

## Smoke-Gate Vocabulary Note

Mesa Phase 3 and Perception Phase 0/2 both use "smoke gate" for the
quantitative threshold below which the apparatus is judged
insufficient. PATH's two halt rows in the verdict template — "Oracle
fails → Halt" and "Random performs comparably to flat → re-tune" — are
the smoke-gate equivalent. They are program-wide-equivalent for the
purpose of cross-doc reading, even if the word does not appear in PATH.
This note is recorded so a reviewer cross-walking from MESA or
PERCEPTION can confirm the semantics line up.

---

## Amendments

Append-only. Each amendment must carry a timestamp (date + zone),
author, and a one-line justification. The body above this rule is
frozen at pre-registration time.

*(no amendments yet)*
