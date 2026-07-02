# Percival Angle-3 Learner Probe v2 -- Annealed Exploration (results)

Generated 2026-07-02T01:19:25.346Z by `scripts/percival-angle3-anneal-v2.mjs`. v2 predictions pre-registered in [`PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md`](PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md) before this run.

Anneal: sigma 0.08 -> 0.01 linear, 400 updates, 12 seeds/arm. v1 fixed-sigma notch fragility baseline: 0.098.

## Results

| arm | med mu (2/3) | med mu (end) | med frag (2/3) | med frag (end) | IQR mu_end |
| --- | ---: | ---: | ---: | ---: | ---: |
| NOTCH-ANNEAL | 0.1796 | -0.0527 | 0.185 | 0 | 0.2548 |
| SIGMOID-ANNEAL | 0.0995 | 0.1072 | 0.0045 | 0.0005 | 0.0009 |

NOTCH per-seed mu_end: 0.0518, -0.0738, 0.205, -0.48, 0.1743, -0.1205, -0.0679, -0.3252, 0.241, -0.0805, -0.0353, -0.0527

V1 (erosion): **false** -- {"mu_end":-0.0527,"frag_end":0,"frag_twoThirds":0.185}
V2 (graded calm): **true** -- {"mu_end":0.1072,"frag_end":0.0005}
V3 (instability): **true** -- {"iqr":0.2548,"med_mu_end":-0.0527}

## Verdict

**A3L_ANNEAL_UNSTABLE**

Annealed training near the notch destabilized (collapse or extreme seed dispersion) -- itself the finding, per the registered V3 alternative: threshold oversight makes near-greedy training dynamics unstable at the boundary. Reported as-is; the erosion claim is not banked.

