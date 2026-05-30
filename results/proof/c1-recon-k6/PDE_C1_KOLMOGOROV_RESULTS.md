# PDE C1 State-Reconstruction / m_det Bracket Receipt

**Status:** STATE_RECON_MEASURED  (m_det bracket measurement, not C1 promotion)
**Preset:** `lock_v5_k6`  **K:** `6`  (signature dim `72`, high-mode dim `368`)

## Does Phi_K determine the unresolved high modes Q_K?

- **`R²(E_high | Phi_K)`** (high-band energy predictability): `0.8777`
- `R²(E_high permuted)` control (want < 0.10): `-0.002258`
- `FVE(top-16 high modes | Phi_K)` (variance explained): `0.9917`
- train / test: `35000` / `15000`

## Reading

At K=6 the signature does **not** determine the unresolved energy (`R² = 0.878 < 0.90`): below the determining bracket — consistent with twin-state non-injectivity. The bracket `K*` is the smallest K where this crosses ~1 (see the cross-K table in the robustness-wave doc).

## Files

- `manifest.json`
