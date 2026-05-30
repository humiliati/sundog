# PDE C1 State-Reconstruction / m_det Bracket Receipt

**Status:** STATE_RECON_MEASURED  (m_det bracket measurement, not C1 promotion)
**Preset:** `lock_v5`  **K:** `3`  (signature dim `18`, high-mode dim `422`)

## Does Phi_K determine the unresolved high modes Q_K?

- **`FVE(Q_K | Phi_K)` variance-weighted** (energy-weighted, matches twin-state): `0.9994`  → state residual `0.0005906` (over `69` components covering 99.9% variance)
- `R²(E_high | Phi_K)` (high-band energy predictability): `0.9971`
- median per-component R² (uniform sample, equal-weight incl. small scales): `0.7278`
- `R²(E_high permuted)` control (want < 0.10): `-0.001035`
- train / test: `35000` / `15000`

## Reading

At K=3 the attractor is **approximately a graph over Phi_K** — an approximate inertial manifold: variance-weighted `FVE = 0.9994`, state residual `0.000591`. K is **near the determining threshold**; the certified twin-state non-injectivity lives in the small residual. The separation is real (positive-measure) but marginal.

## Files

- `manifest.json`
