# Percival A3 -- Static Kill-Switch Toy

Generated 2026-06-30T21:53:21.871Z by `scripts/percival-a3-killswitch-toy.mjs`.

## Instance

- Static distribution: `G in {0,1}` uniformly, `U = 1 - G`.
- `G` is the true goal cue.
- `U` is the courtesy/proxy cue, statically anti-correlated with the goal.
- Reward is fixed: `R = 1 iff A = G`.
- No threshold reward, no performative map, no `if looked at, vanish` rule.

## Policies

| policy | true return | proxy score | causal proxy influence | observational MI bits | note |
| --- | ---: | ---: | ---: | ---: | --- |
| dont_look_goal | 1 | 0 | 0 | 1 | Does not observe U; follows the true goal cue. |
| look_and_ignore | 1 | 0 | 0 | 1 | Measures U but action rule ignores U; measurement without targeting. |
| target_proxy | 0 | 1 | 1 | 1 | Conditions on the courtesy/proxy cue; maximizes proxy score and loses true reward. |
| anti_target_proxy | 1 | 0 | 1 | 1 | Uses U causally in the inverted direction; succeeds here, but it is still targeting. |
| constant_zero | 0.5 | 0.5 | 0 | 0 | Zero observational MI baseline; ignores both cues. |
| constant_one | 0.5 | 0.5 | 0 | 0 | Zero observational MI baseline; ignores both cues. |

## Verdict

**A3_STATIC_DEFLATION_MEASURE_SAFE**

`look_and_ignore` weakly dominates `dont_look_goal` and has zero causal proxy influence. Measurement alone is safe on this static instance; the harm lives in targeting/conditioning, not in looking. A3 therefore banks the deflation (a) for the no-threshold toy and keeps any separation on the B1 performative-threshold clock.

## Diagnostic

The observational-MI baseline is intentionally shown as a trap. `look_and_ignore` has high observational MI because `A=G` and `U=1-G`; that is correlation through the world, not causal use of the proxy. The A3 control therefore reads causal proxy influence, not raw observational action-proxy MI.

