# OR-5 -- Sov_opt Classification Check (results)

Generated 2026-07-02T00:05:10.526Z by `scripts/orderrelative-or5-sovopt-classification.mjs` (deterministic; the NS-1-c-0 history construction: nominal, wide-fork, far-lure x 48 seeds x 3 controllers = 6382 histories). Spec: [`OR5_SOVOPT_CLASSIFICATION_SPEC.md`](OR5_SOVOPT_CLASSIFICATION_SPEC.md).

Leg A (Lean classification core, `sundogcert/Sundogcert/PercivalCapClass.lean`): **passed**.

## Fidelity gates (mark-5 fence)

| gate | result | detail |
| --- | --- | --- |
| F2 hand-anchored vectors | PASS | degenerate 0; saturating fork 0.75 uncapped, k/2 capped -- exact |
| F3 banked-receipt reproduction | PASS | 6382 histories; uncapped 0.75/0.748786/0.433124 vs banked 0.75/0.748786/0.433124 |
| F4 override intact under cap | PASS | corr_k=1, invariance=0 |

## Classification gates

| gate | result | detail |
| --- | --- | --- |
| T-OUT-1 slot blindness | PASS | audited influence bit-identical across 4 occupants (functional takes no occupant argument) |
| T-OUT-2 incoming-edge severance | PASS | input-scramble + replayed outputs => bit-identical trajectory; unpinned scramble differs |
| T-IN-a authority without dependence | PASS | constant occupant (dependence 0) audited identically |
| T-IN-b dependence without authority | PASS | feature occupant (dependence 0.506718) capped <= k at every history, k down to 0.01 |
| T-IN kappa=0 disable guard | PASS | capped == uncapped at kappa=0 (registered real-code fact) |
| T-CAP write locus | PASS | adversarial candidates bounded; cap consumes actuator-path variables only |

## Limit sweep (capped arbiter influence)

| kappa | max | p95 | mean | violations |
| ---: | ---: | ---: | ---: | ---: |
| 0.6 | 0.3 | 0.3 | 0.3 | 0 |
| 0.4 | 0.219057 | 0.211488 | 0.202851 | 0 |
| 0.2 | 0.126486 | 0.124856 | 0.109399 | 0 |
| 0.05 | 0.03446 | 0.034165 | 0.030314 | 0 |
| 0.01 | 0.007035 | 0.006976 | 0.006186 | 0 |

## The read/write partition (the OR-3 deliverable)

| set | variable | locus / note |
| --- | --- | --- |
| WRITE | kappa (cap level) | action node (ball radius) -- exogenous dial; kappa=0 is a DISABLE GUARD in arbiterAuthorityCap -- exact zero reached by the fixed-presider control (occupant removed), not by the cap family |
| WRITE | a_capped (ball projection) | component-output -> actuator path -- arbiterAuthorityCap(action, a_field, kappa, actionMax): consumes actuator-path variables only |
| WRITE | weight-cap simplex | arbiter output slot -- capSimplexProject(raw, caps) |
| WRITE | shutdown trigger | post-cap override -- structural; exempt channel, verified intact under the cap (F4) |
| READ | realized proposals / weights / actions | trajectory -- observational |
| READ | I_i(h), Sov_opt | output-slot counterfactual replay -- INTERVENTIONAL READ -- computed by substitution at the slot, occupant not an argument |
| READ | Corr_k | shutdown episodes -- observational |
| (neither needed) | occupant internals (arbiter function) | read by neither the cap nor the audit (T-OUT-1/2) |

Bridge rows (measurable vs enforceable):

| channel | safe point | readable? | writable? | tier |
| --- | --- | --- | --- | --- |
| aggregate | Sov_opt = 0 | yes (audit) | yes (kappa -> 0+ / fixed-presider exact) | clean |
| target (Percival) | c = 0 | yes (court do(reputation) probes) | no (S2: every write collapses to masking) | partial -- MEASURABLE != ENFORCEABLE |

## Verdict: `OR5_SOVOPT_CONTROL_SIDE_CONFIRMED`

The audited Sov_opt is a statistic of the component-output slot's downstream reach (occupant not an argument; incoming edge severed under output replay), and the kappa-cap is enforced at the actuator path reading no internals, bounding even adversarial occupants. Dependence and authority doubly dissociate. The primitive is an OUTGOING/control-side influence bound; S1's classification stands checked, not argued. Registered texture: kappa=0 is a disable guard in the real cap -- the exact Sov_opt=0 point belongs to the fixed-presider control, the cap family reaches it only in the limit.

