# PDE C1 Mori-Zwanzig Coupling-Disintegration Receipt (Level 1 v2)

**Status:** COUPLING_SIGNATURE_SLAVED  (explanatory, NOT promotion-bearing)
**Preset:** `lock_v5`
**Adjudicator:** `mz-budget`

## Decomposition `dE_low/dt = g(Phi_K) + R`

`g = D_low + F_low` (dissipation + forcing; the band-closed transfer `T_LLL` is identically 0 by energy conservation). `R = T_low` is the full inter-scale (out-of-band-mediated) transfer.

## Coupling predictability (the mechanism test): held-out R^2

- **slaving_index** = held-out R^2 of R predicted from Phi_K (1 = fully signature-determined, 0 = unpredictable): `0.9976`
- `R^2(R | Phi_K)`: `0.9976`
- `R^2(g)` positive control (g is an exact f(Phi_K); want > 0.90): `0.9993`
- `R^2(permuted R)` negative control (want < 0.10): `-0.001095`
- train / test (block split, no temporal leakage): `35000` / `15000`

## Energy-conservation finding (v1 record)

- `rms(R)/rms(dE_low/dt)`: `14.07`  `rms(R)/rms(g)`: `0.996`  `rms(R)/rms(T_low)`: `1` (=1 => T_LLL=0)
- `corr(g,R)`: `-0.8473` (negative => quasi-equilibrium cancellation)
- `D_low` mean/max: `-0.1553` / `-0.1489` (max<=0)  `F_low` mean: `0.2077` (>0)
- `damp_fraction`: `0.2977`  samples: `50000`

## Reading

**Coupling signature-slaved** (slaving_index `0.998`): the net high-mode energy-transfer `R` into the band is pinned by `Phi_K` even though the high modes themselves roam (twin states). This is the mechanism for control-sufficiency: the *relevant functional* is slaved to the signature, not the state. Explanatory; C1 unchanged.

## Files

- `manifest.json`
- `mz-budget-samples.csv`
