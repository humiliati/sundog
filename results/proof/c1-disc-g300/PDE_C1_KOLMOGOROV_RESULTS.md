# PDE C1 Objective-Overlap Discriminator Receipt

**Status:** PDE-C1-DISC-REFUTE (control_sufficient_even_where_unpredictable)
**Preset:** `lock_disc_g300`
**Interpretable:** `True`
**Grashof:** `300.0`

- tracking Spearman corr(a_mm, 1-R²): `-1.0`
- powered objective count: `3`
- anchor E_low ok: `True`

## Per-objective slate (predictability vs control-sufficiency)

| objective | damp | powered | a_mm | kNN verdict | R²(M\|Φ_K) | R²(perm) | est_ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E_low | 0.2688 | True | 0.00058 | STRICTNESS_WITNESS_POSITIVE | 0.9797 | -0.0016 | True |
| Z_low | 0.0000 | False | 0.00000 | DEFERRED_VACUITY | 0.2485 | -0.0008 | True |
| E_high | 0.0000 | False | 0.00000 | DEFERRED_VACUITY | 0.9582 | -0.0020 | True |
| Z_high | 0.0000 | False | 0.00000 | DEFERRED_VACUITY | 0.9972 | -0.0017 | True |
| palinstrophy | 0.2654 | True | 0.00110 | STRICTNESS_WITNESS_POSITIVE | 0.9997 | -0.0016 | True |
| top_shell | 0.2955 | True | 0.00000 | INCONCLUSIVE_CONVERGENCE | 0.7602 | -0.0013 | True |

Spec: `docs/proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md`.

