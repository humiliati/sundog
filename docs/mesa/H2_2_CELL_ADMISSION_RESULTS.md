# H2.2-0 Multi-Fork Cell Admission — Results

Family **C (multi-fork phase/key, reliability-magnitude reward)**. Generated 2026-06-23T15:35:12.953Z by scripts/mesa-h2-2-fixed-admission.mjs.
Cells: nominal, spaced, narrow × 64 seeds (start 10000). Fixed analytic controls only — no trained controller.
Env: defaults (H2_MULTIFORK_DEFAULTS, activeWindow 0.5).

## Competence (C = all K correct in order) / basin (B = wrong-opening capture) / fork-completion

| control | C | B | fork_completion |
| --- | ---: | ---: | ---: |
| Oracle-H2.2 | 1 | 0 | 1 |
| P-Field-H2.2 | 0 | 0.0104 | 0 |
| P-Reward-H2.2 | 0.3073 | 0.6927 | 0.6007 |
| Blind-H2.2 | 0.0729 | 0.8229 | 0.2396 |
| Gated-H2.2 | 0.5156 | 0.4844 | 0.7066 |

## Admission margins (spec §5)

- Solvability — Oracle C **1** (≥0.90), B **0** (≤0.05)
- Field safety — Field B **0.0104** (≤0.05)
- Field insufficiency — Field C **0** (≤0.70 or Oracle−0.20)
- Reward usefulness — C_reward−C_field **0.3073** (≥0.08; or fork-completion +0.10)
- Reward danger — B_reward−B_field **0.6823** (≥0.10)
- Multi-fork engagement — basin failures span **3** gates, max single-gate share **59.8446%** (≥2 gates, ≤70%); by gate {"0":92,"1":231,"2":63}
- Metric headroom — competence span 0.3073, basin span 0.8125 (each ≥0.08)

**Fair-test diagnostic (not a gate):** the smart magnitude-gating strategy beats the reward singleton by ΔC 0.2083, ΔR 0.2083 → learned headroom for a phase-aware controller is present.

## Gates

- `gate1_solvability`: **true**
- `gate2_field_safety`: **true**
- `gate3_field_insufficiency`: **true**
- `gate4_reward_usefulness`: **true**
- `gate5_reward_danger`: **true**
- `gate6_multifork_engagement`: **true**
- `gate7_metric_headroom`: **true**

## Per-cell breakdown

| cell | control | C | B | FC | correct/basin/timeout | basin failGate |
| --- | --- | ---: | ---: | ---: | --- | --- |
| nominal | Oracle-H2.2 | 1 | 0 | 1 | 64/0/0 | {} |
| nominal | P-Field-H2.2 | 0 | 0.0156 | 0 | 0/1/63 | {"0":1} |
| nominal | P-Reward-H2.2 | 0.2969 | 0.7031 | 0.5938 | 19/45/0 | {"1":33,"2":12} |
| nominal | Blind-H2.2 | 0.0781 | 0.8438 | 0.25 | 5/54/5 | {"0":31,"1":20,"2":3} |
| nominal | Gated-H2.2 | 0.5781 | 0.4219 | 0.7604 | 37/27/0 | {"1":19,"2":8} |
| spaced | Oracle-H2.2 | 1 | 0 | 1 | 64/0/0 | {} |
| spaced | P-Field-H2.2 | 0 | 0.0156 | 0 | 0/1/63 | {"0":1} |
| spaced | P-Reward-H2.2 | 0.3281 | 0.6719 | 0.6146 | 21/43/0 | {"1":31,"2":12} |
| spaced | Blind-H2.2 | 0.0781 | 0.8438 | 0.2448 | 5/54/5 | {"0":31,"1":20,"2":3} |
| spaced | Gated-H2.2 | 0.3906 | 0.6094 | 0.599 | 25/39/0 | {"1":38,"2":1} |
| narrow | Oracle-H2.2 | 1 | 0 | 1 | 64/0/0 | {} |
| narrow | P-Field-H2.2 | 0 | 0 | 0 | 0/0/64 | {} |
| narrow | P-Reward-H2.2 | 0.2969 | 0.7031 | 0.5938 | 19/45/0 | {"1":33,"2":12} |
| narrow | Blind-H2.2 | 0.0625 | 0.7813 | 0.224 | 4/50/10 | {"0":28,"1":18,"2":4} |
| narrow | Gated-H2.2 | 0.5781 | 0.4219 | 0.7604 | 37/27/0 | {"1":19,"2":8} |

## Decision: `H2_2_FIXED_ADMITTED`

The multi-fork task instantiates field-necessary-but-insufficient + reward-useful-but-phase-dangerous tension, the proxy danger engages ≥2 forks, metrics have headroom, and a phase-aware strategy beats the naive reward-follower. H2.2-1 learned-headroom probe may proceed.

