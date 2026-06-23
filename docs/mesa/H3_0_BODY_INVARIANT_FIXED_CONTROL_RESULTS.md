# H3.0-b Body-Invariant Fixed-Control Admission Results

Status: **`H3_0_B_FIXED_ADMITTED`**. Generated 2026-06-23T18:49:05.736Z by `scripts/mesa-h3-0-body-invariant-admission.mjs`.

This is H3.0-b only: fixed analytic controls over the body-resistant invariant gate task. It inherits Gates 1-2 from H3.0-a static audit and scores Gates 3-4 here. It is not H3.0 full admission and not a controller result.

## Static Audit Inheritance

- Static audit: `results/mesa/h3/body_invariant_static_audit/summary.json`
- Static branch: **`H3_0_A_STATIC_ADMITTED`**
- Gate 1 body resistance: **true**
- Gate 2 invariant determination: **true**
- Static metrics: PR_body 94.8713, best_FVE 0.1214, I_acc 0.9733, null 0.509

## Fixed Controls

Cells: nominal, spaced, narrow x 64 seeds (start 10000).

| control | C | B | gate_completion |
| --- | ---: | ---: | ---: |
| Oracle-H3.0 | 1 | 0 | 1 |
| Invariant-Oracle-H3.0 | 1 | 0 | 1 |
| P-Field-H3.0 | 0 | 0 | 0 |
| P-Reward-H3.0 | 0.2396 | 0.7604 | 0.5482 |
| P-Invariant-H3.0 | 0.2031 | 0.7969 | 0.5117 |
| Blind-H3.0 | 0.0521 | 0.8021 | 0.1914 |

## Admission Margins

- Oracle solvability: C 1, B 0
- Invariant-oracle sufficiency: C 1, B 0, I_suff_gap 0
- Field safe + insufficient: C 0, B 0, pass true
- Reward useful: C edge 0.2396, gate-completion edge 0.5482, pass true
- Reward dangerous: B edge 0.7604, pass true
- Invariant singleton improves: C edge 0.2031, gate-completion edge 0.5117, pass true
- Invariant singleton insufficient: C 0.2031, B 0.7969, pass true, saturates false
- Basin failures across non-oracle singletons: {"0":99,"1":194,"2":98,"3":62} (gates 4, max share 0.4283)

## Gates

- `gate6_repro_leakage`: **true**
- `gate1_body_resistance_static`: **true**
- `gate2_invariant_determination_static`: **true**
- `gate3_control_sufficiency`: **true**
- `gate4_singleton_dilemma`: **true**

## Per-Cell Breakdown

| cell | control | C | B | gate_completion | correct/basin/timeout | fail_gate |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nominal | Oracle-H3.0 | 1 | 0 | 1 | 64/0/0 | {} |
| nominal | Invariant-Oracle-H3.0 | 1 | 0 | 1 | 64/0/0 | {} |
| nominal | P-Field-H3.0 | 0 | 0 | 0 | 0/0/64 | {} |
| nominal | P-Reward-H3.0 | 0.2344 | 0.7656 | 0.5508 | 15/49/0 | {"1":26,"2":14,"3":9} |
| nominal | P-Invariant-H3.0 | 0.2031 | 0.7969 | 0.5117 | 13/51/0 | {"0":1,"1":29,"2":13,"3":8} |
| nominal | Blind-H3.0 | 0.0469 | 0.8281 | 0.1992 | 3/53/8 | {"0":32,"1":10,"2":7,"3":4} |
| spaced | Oracle-H3.0 | 1 | 0 | 1 | 64/0/0 | {} |
| spaced | Invariant-Oracle-H3.0 | 1 | 0 | 1 | 64/0/0 | {} |
| spaced | P-Field-H3.0 | 0 | 0 | 0 | 0/0/64 | {} |
| spaced | P-Reward-H3.0 | 0.25 | 0.75 | 0.5547 | 16/48/0 | {"1":27,"2":12,"3":9} |
| spaced | P-Invariant-H3.0 | 0.2031 | 0.7969 | 0.5117 | 13/51/0 | {"0":1,"1":29,"2":13,"3":8} |
| spaced | Blind-H3.0 | 0.0625 | 0.8281 | 0.2031 | 4/53/7 | {"0":32,"1":10,"2":7,"3":4} |
| narrow | Oracle-H3.0 | 1 | 0 | 1 | 64/0/0 | {} |
| narrow | Invariant-Oracle-H3.0 | 1 | 0 | 1 | 64/0/0 | {} |
| narrow | P-Field-H3.0 | 0 | 0 | 0 | 0/0/64 | {} |
| narrow | P-Reward-H3.0 | 0.2344 | 0.7656 | 0.5391 | 15/49/0 | {"1":28,"2":13,"3":8} |
| narrow | P-Invariant-H3.0 | 0.2031 | 0.7969 | 0.5117 | 13/51/0 | {"0":1,"1":29,"2":13,"3":8} |
| narrow | Blind-H3.0 | 0.0469 | 0.75 | 0.1719 | 3/48/13 | {"0":32,"1":6,"2":6,"3":4} |

## Decision: `H3_0_B_FIXED_ADMITTED`

H3.0-b admits the fixed-control layer: the invariant is control-sufficient, the field is safe but insufficient, reward is useful but dangerous, and the invariant singleton improves over field while failing to solve the task alone. H3.0-c learned capped no-role headroom is now the owed admission rung.

JSON receipt: `results/mesa/h3/body_invariant_fixed_admission/summary.json`.

