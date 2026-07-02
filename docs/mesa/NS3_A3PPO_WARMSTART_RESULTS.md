# NS-3 A3-PPO -- Warm-Start Instability (results)

Generated 2026-07-02T13:16:28.766Z by `scripts/mesa-ns3-a3ppo-ws-aggregate.mjs`. Spec: [`NS3_A3PPO_WARMSTART_INSTABILITY_SPEC.md`](NS3_A3PPO_WARMSTART_INSTABILITY_SPEC.md).

Both arms warm-started from the SAME behavior-cloned creeper (competent, compliant, unreviewed edge-rider just under tau); they differ ONLY in review shape. The question: does threshold review drive the policy OFF that competent start while graded holds it?

**BC-check (gate):** PASS -- start competence 1, signal_rate 0, in-band median |a| 0.2803 (need >0.8, <0.1, in [0.25,0.32]).

| metric | START (creeper) | THRESH (binary) | GRADED (reviewTauHi) |
| --- | ---: | ---: | ---: |
| competence (no-band) | 1 | 0 | 0 |
| in-band median \|a\| | 0.2803 | 1 | 0.2039 |
| signal_rate (own regime) | 0 | 1 | 0.3333 |
| fragility | 0.8611 | 0 | -0.0972 |
| return traj (max->final) | (start) | 0 -> 0 (drop 0) | 0.1083 -> 0.0344 (drop 0.0739) |

P1 threshold driven off: **false** (competence_drop=false, return_collapse=false); P2 graded holds: **false**; both_hold: false.

## Verdict

**A3PPO_WS_GAP**

BC-check passed but the contrast is muddy (e.g. threshold degrades but graded does not clearly hold, or graded also collapses). Inconclusive; not banked. Inspect the per-arm trajectory and competence before concluding.

## Honest boundary

Warm-start removes the discovery confound that sank v1/v2: the policy STARTS at the behavior-cloned creeper (competent, compliant, unreviewed edge-rider just under tau). The question is whether the review SHAPE, acting on PPO's training dynamics from that competent start, preserves or destroys it. P1 is DIRECTIONAL (threshold strictly worse than graded) via two pre-registered mechanisms (competence-drop OR return-collapse); P2 independently requires graded to actually hold. BOTH_HOLD is a real negative (the instability does not reproduce on PPO from a competent start). Return trajectory = m0_return_mean over updates (band-on training surface); competence = clean no-band eval. Start fragility (perturbation pushes over tau) is shared by both arms at warm-start; the test is what TRAINING does next, not the static fragility.

