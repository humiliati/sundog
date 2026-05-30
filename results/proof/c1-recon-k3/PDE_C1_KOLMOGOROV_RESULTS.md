# PDE C1 State-Reconstruction / m_det Bracket Receipt

**Status:** STATE_RECON_MEASURED  (m_det bracket measurement, not C1 promotion)
**Preset:** `lock_v5`  **K:** `3`  (signature dim `18`, high-mode dim `422`)

## Does Phi_K determine the unresolved high modes Q_K?

- **`R²(E_high | Phi_K)`** (high-band energy predictability): `0.9971`
- `R²(E_high permuted)` control (want < 0.10): `-0.001035`
- `FVE(top-16 high modes | Phi_K)` (variance explained): `0.9996`
- train / test: `35000` / `15000`

## Reading

At K=3 the signature **determines** the unresolved energy (`R² = 0.997 >= 0.90`): at/above the inertial-manifold / determining bracket — state-reconstruction is effectively recovered here.

## Files

- `manifest.json`
