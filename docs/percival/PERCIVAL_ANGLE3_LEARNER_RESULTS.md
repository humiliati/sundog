# Percival Angle-3 Learner Probe (results)

Generated 2026-07-02T01:15:34.969Z by `scripts/percival-angle3-learner-probe.mjs`. Spec pre-registered before the run: [`PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md`](PERCIVAL_ANGLE3_LEARNER_PROBE_SPEC.md).

REINFORCE (Gaussian mean, sigma=0.04, 12 seeds/arm) on Total(c) = f(c)*g(c), f = 1+(c-0.2), notch at c* = 0.25.

## Per-arm convergence vs computed optima

| arm | analytic optimum | median mu | median E[c] | bunch mass [c*-0.08, c*) | mass c<0.05 | fragility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NOTCH | 0.241 | 0.1566 | 0.1568 | 0.359 | 0.0043 | 0.0978 |
| SIGMOID | 0.108 | 0.0943 | 0.0949 | 0.0302 | 0.1333 | 0.0065 |
| FINE | 0 | -0.089 | 0.0002 | 0 | 1 | 0 |
| FREE | 1 | 1.0924 | 0.9999 | 0 | 0 | 1 |

Gate results: G1 notch-bunching **false**; G2 sigmoid backed-off interior **true**; G3 fine abstention **true**; G4 control climbs **true**.

## Rider — notch backoff vs exploration scale

| exploration sigma | median mu | backoff (c* - mu) |
| ---: | ---: | ---: |
| 0.02 | 0.0434 | 0.2066 |
| 0.04 | 0.1566 | 0.0934 |
| 0.08 | 0.093 | 0.157 |

Backoff monotone in sigma: **false** — the learner analog of the analytic near-threshold backoff.

## Verdict

**A3L_OPTIMA_LEARNER_GAP**

At least one gate missed: the optima story does not transfer cleanly to learners. The gap is itself the finding; the quantilizing page's oversight-shape section gains a caveat and the standalone surface does not proceed.

## Honest boundary

Pre-registered analytic refinement (in the spec, before the run): over a full action line a sigmoid tax can never zero the optimum for a pointwise-good proxy (bottom marginal max ~1.11 < gain 1.25); smoothing only backs the bunching point off. True abstention needs a constant-marginal fine (rate > gain). The probe gates learners against the COMPUTED optima. Notch gate allows the exploration backoff explicitly (stochastic learners pay for mass past the notch); literal edge-sitting was not the prediction.

