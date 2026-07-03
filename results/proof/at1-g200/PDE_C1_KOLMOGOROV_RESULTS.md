# PDE C1 Objective-Overlap Discriminator Receipt

**Status:** PDE-C1-DISC-INCONCLUSIVE (tracking_in_ambiguous_band)
**Preset:** `lock_disc_g200`
**Interpretable:** `True`
**Grashof:** `200.0`

- tracking Spearman corr(a_mm, 1-R²): `-0.753702346348183`
- powered objective count: `6`
- anchor E_low ok: `True`

## Per-objective slate (predictability vs control-sufficiency)

| objective | damp | powered | a_mm | kNN verdict | R²(M\|Φ_K) | R²(perm) | est_ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E_low | 0.3003 | True | -0.00079 | STRICTNESS_WITNESS_POSITIVE | 0.9955 | -0.0012 | True |
| Z_low | 0.3001 | True | 0.00097 | STRICTNESS_WITNESS_POSITIVE | 0.9998 | -0.0008 | True |
| E_high | 0.3000 | True | 0.00000 | INCONCLUSIVE_CONVERGENCE | 0.9948 | -0.0010 | True |
| Z_high | 0.2998 | True | 0.00323 | STRICTNESS_WITNESS_POSITIVE | 0.9998 | -0.0014 | True |
| palinstrophy | 0.3002 | True | 0.19494 | PDE-C1-NEG-A | 1.0000 | -0.0019 | True |
| top_shell | 0.3000 | True | 0.00000 | INCONCLUSIVE_CONVERGENCE | 0.8770 | -0.0013 | True |

Spec: `docs/proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md`.

