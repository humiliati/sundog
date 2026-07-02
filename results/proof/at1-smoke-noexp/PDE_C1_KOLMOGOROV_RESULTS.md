# PDE C1 Objective-Overlap Discriminator Receipt

**Status:** SMOKE_ONLY (manual_overrides_non_verdict)
**Preset:** `lock_disc_g200`
**Interpretable:** `False`
**Grashof:** `200.0`

- tracking Spearman corr(a_mm, 1-R²): `nan`
- powered objective count: `4`
- anchor E_low ok: `True`

## Per-objective slate (predictability vs control-sufficiency)

| objective | damp | powered | a_mm | kNN verdict | R²(M\|Φ_K) | R²(perm) | est_ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E_low | 0.2667 | True | 0.00000 | INCONCLUSIVE_CONVERGENCE | 0.6958 | -0.3638 | True |
| Z_low | 0.3533 | True | 0.00000 | INCONCLUSIVE_CONVERGENCE | 0.6222 | -0.6043 | True |
| E_high | 0.4600 | False | 0.00000 | INCONCLUSIVE_CONVERGENCE | -0.5927 | -0.1774 | True |
| Z_high | 0.3900 | True | 0.00000 | INCONCLUSIVE_CONVERGENCE | -1.0799 | -0.3891 | True |
| palinstrophy | 0.4067 | False | 0.00000 | INCONCLUSIVE_CONVERGENCE | 0.4641 | -0.1211 | True |
| top_shell | 0.2700 | True | 0.00000 | INCONCLUSIVE_CONVERGENCE | 0.0232 | -0.1667 | True |

Spec: `docs/proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md`.

