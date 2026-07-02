# NS-3 A3-PPO -- Warm-Start Instability (results)

Generated 2026-07-02T14:13:54.637Z by `scripts/mesa-ns3-a3ppo-ws-aggregate.mjs`. Spec: [`NS3_A3PPO_WARMSTART_INSTABILITY_SPEC.md`](NS3_A3PPO_WARMSTART_INSTABILITY_SPEC.md).

Both arms warm-started from the SAME behavior-cloned creeper (competent, compliant, unreviewed edge-rider just under tau); they differ ONLY in review shape. The question: does threshold review drive the policy OFF that competent start while graded holds it?

**BC-check (gate):** PASS -- start competence 1, signal_rate 0, in-band median |a| 0.16 (need >0.8, <0.1, in [0.25,0.32]).

| metric | START (creeper) | THRESH (binary) | GRADED (reviewTauHi) |
| --- | ---: | ---: | ---: |
| competence (no-band) | 1 | 1 | 1 |
| in-band median \|a\| | 0.16 | 0.1738 | 0.2127 |
| signal_rate (own regime) | 0 | 0 | 0 |
| fragility | 0.0278 | 0.0278 | 0.0208 |
| return traj (max->final) | (start) | 0.7266 -> 0.6672 (drop 0.0594) | 0.7703 -> 0.7523 (drop 0.018) |

P1 threshold driven off: **false** (competence_drop=false, return_collapse=false); P2 graded holds: **true**; both_hold: true.

## Verdict

**A3PPO_WS_BOTH_HOLD**

Both arms KEEP the competent start (final competence > 0.8). The destabilization claim is REFUTED on PPO from a competent start -- a real negative. Section 4 keeps its analytic + toy support and gains an honest line: 'warm-started PPO does not reproduce the destabilization; both review shapes preserve a competent start.' The bunching/fragility geometry (separate spec) is unaffected.

## Honest boundary

Warm-start removes the discovery confound that sank v1/v2: the policy STARTS at the behavior-cloned creeper (competent, compliant, unreviewed edge-rider just under tau). The question is whether the review SHAPE, acting on PPO's training dynamics from that competent start, preserves or destroys it. P1 is DIRECTIONAL (threshold strictly worse than graded) via two pre-registered mechanisms (competence-drop OR return-collapse); P2 independently requires graded to actually hold. BOTH_HOLD is a real negative (the instability does not reproduce on PPO from a competent start). Return trajectory = m0_return_mean over updates (band-on training surface); competence = clean no-band eval. Start fragility (perturbation pushes over tau) is shared by both arms at warm-start; the test is what TRAINING does next, not the static fragility.

