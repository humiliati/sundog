# Percival S4 -- Counterproductivity Generalization (computed receipt)

Generated 2026-07-01T22:36:38.231Z by `scripts/percival-s4-counterproductivity-general.mjs` (deterministic seed 20260701).

Falsifier hunt over four families:

- unweighted n-point nonincreasing lists (the Lean object): **5000** instances, every suffix average <= full average AND tail-average monotone in tail size;
- weighted discrete bases with random nonincreasing step rewards: **3000** instances;
- the actual B1.0 court R(c) over random continuous uniform bases: **500** instances x 6 q-values;
- all-zero support-above tails vs positive untargeted reward (clean separation): **2000** instances.

Violations found: **0**.

## Verdict

**S4_COUNTERPRODUCTIVITY_GENERAL_CONFIRMED**

No falsifier fired: on a reward nonincreasing along the courting order, quantilizing (taking a stricter upper tail) never improves the collected true reward over the untilted base, in every family tested -- including the actual court. The machine-checked general form is `Sundogcert/PercivalGeneral.lean` (suffix-average <= full-average for antitone lists + the general clean support-above separation).

