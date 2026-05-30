# PDE C1 Mori-Zwanzig Coupling-Disintegration Receipt (Level 1 v2)

**Status:** COUPLING_SIGNATURE_SLAVED  (explanatory, NOT promotion-bearing)
**Preset:** `lock_v7_g300`
**Adjudicator:** `mz-budget`

## Decomposition `dE_low/dt = g(Phi_K) + R`

`g = D_low + F_low` (dissipation + forcing; the band-closed transfer `T_LLL` is identically 0 by energy conservation). `R = T_low` is the full inter-scale (out-of-band-mediated) transfer.

## Coupling predictability (the mechanism test): held-out R^2

- **slaving_index** = held-out R^2 of R predicted from Phi_K (1 = fully signature-determined, 0 = unpredictable): `0.9896`
- `R^2(R | Phi_K)`: `0.9896`
- `R^2(g)` positive control (g is an exact f(Phi_K); want > 0.90): `0.9798`
- `R^2(permuted R)` negative control (want < 0.10): `-0.000455`
- train / test (block split, no temporal leakage): `35000` / `15000`

## Energy-conservation finding (v1 record)

- `rms(R)/rms(dE_low/dt)`: `2.922`  `rms(R)/rms(g)`: `0.9226`  `rms(R)/rms(T_low)`: `1` (=1 => T_LLL=0)
- `corr(g,R)`: `-0.9259` (negative => quasi-equilibrium cancellation)
- `D_low` mean/max: `-0.129` / `-0.1177` (max<=0)  `F_low` mean: `0.1713` (>0)
- `damp_fraction`: `0.2688`  samples: `50000`

## Reading

**Coupling signature-slaved** (slaving_index `0.99`): the net high-mode energy-transfer `R` into the band is pinned by `Phi_K` even though the high modes themselves roam (twin states). This is the mechanism for control-sufficiency: the *relevant functional* is slaved to the signature, not the state. Explanatory; C1 unchanged.

## Files

- `manifest.json`
- `mz-budget-samples.csv`
