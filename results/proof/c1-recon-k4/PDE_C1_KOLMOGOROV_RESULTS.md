# PDE C1 State-Reconstruction / m_det Bracket Receipt

**Status:** STATE_RECON_MEASURED  (m_det bracket measurement, not C1 promotion)
**Preset:** `lock_v5_k4`  **K:** `4`  (signature dim `32`, high-mode dim `408`)

## Does Phi_K determine the unresolved high modes Q_K?

- **`R²(E_high | Phi_K)`** (high-band energy predictability): `0.9183`
- `R²(E_high permuted)` control (want < 0.10): `-0.001145`
- `FVE(top-16 high modes | Phi_K)` (variance explained): `0.9986`
- train / test: `35000` / `15000`

## Reading

At K=4 the signature **determines** the unresolved energy (`R² = 0.918 >= 0.90`): at/above the inertial-manifold / determining bracket — state-reconstruction is effectively recovered here.

## Files

- `manifest.json`
