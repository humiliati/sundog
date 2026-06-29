# The Resistant-Body Test — pre-registration (deconfound recoverability from dimension)

- **Status: PRE-REGISTERED 2026-06-29; prediction + falsifier frozen below BEFORE the run.**
- **What it decides:** the `navierstokes.html` "re-aimed" paragraph currently says a sharp
  regime-2 split needs a body that is *high-dimensional by construction*. The page's own
  data hints otherwise (the 256-wide `net.7` is effectively ~2-D and is rebuilt ~99% → it
  came back **marginal**). This test asks, directly: does regime-2 sharpness track **low
  recoverability** (the shadow can't rebuild the body) or **high dimension**?

> **FENCES (binding).** Constructed / toy (flat-Gaussian bodies, linear shadows). It is
> **not** a new NSE result, **not** a change to C1, and does **not** by itself authorize any
> public copy. Its job is to decide *which axis* the re-aim sentence should name. The
> resistance it exercises is **information-theoretic recoverability** (can the shadow rebuild
> the body), i.e. the `recon-FVE` the page already uses — consistent with the H5 fiber axis,
> not a computational-hardness claim. Internal research note.

## Why this test (the confound)

- The page (para 1–2) frames everything on **reconstruction**: "does the low-dimensional
  shadow rebuild the full state?", `net.7` "reconstructs ~99%". Para 3 then pivots the
  *lesson* to **dimension**. That is the inconsistency under test.
- The internal resist-construction Pass 2 raised body dimension `D` *at fixed shadow rank*,
  so dimension and recoverability moved **together** (more `D` ⇒ shadow captures a smaller
  fraction ⇒ lower recon-FVE). It could not separate them. This test separates them.

## Design

- **Body** `x ∈ ℝ^D`, `x ~ N(0, I)` (flat spectrum — every coordinate equally important).
- **Shadow** `s = ` the first `r` coordinates (a rank-`r` linear shadow).
- **Recoverability dial** `ρ_rec = r / D` = the fraction of body variance the shadow can
  capture. With a flat spectrum the trained reconstructor's `recon-FVE → r/D` exactly
  (the dropped `D−r` coordinates are independent ⇒ information-theoretically gone).
- **Control task** target `y = x₀ + x₁ + 0.1·noise` — a functional living **inside** the
  shadow (coords 0,1 are kept whenever `r ≥ 2`), so control is *possible* in every cell;
  what varies is only how much of the *rest* of the body the shadow drops.
- **Fixed compute:** identical probe capacity in every cell; a 2× larger MLP reconstructor
  is also run to check compute cannot beat the information floor.
- **Measurables** (all trained, cross-validated — not asserted):
  - `control_suff` = CV R²(`s → y`).
  - `recon_FVE` = CV R²(`s → x`) (mean over coordinates) = recoverability.
  - **`sharpness = control_suff · (1 − recon_FVE)`** = regime-2 strength (control-sufficient
    AND state-insufficient).
- **Grid:** `D ∈ {8, 32, 128}` × `ρ_rec ∈ {0.25, 0.5, 0.75, 0.9}` (`r = round(ρ_rec·D)`, `r ≥ 2`).

## Frozen prediction (recoverability is the axis, dimension is not)

1. `control_suff ≈ 1` in **every** cell (the control-relevant functional is always in the shadow).
2. `recon_FVE ≈ r/D` empirically, and the 2× reconstructor does **not** beat it (information floor; compute cannot cross).
3. `sharpness ≈ 1 − ρ_rec`: **flat in `D`** at matched `ρ_rec`, and **rising as `ρ_rec` falls** at fixed `D`.
4. **Decisive cells:** the high-`D` / high-recoverability cell (`D=128, ρ_rec=0.9`, the `net.7` analogue) is **marginal** (sharpness ≈ 0.1) while a low-`D` / low-recoverability cell (`D=8, ρ_rec≈0.25`) is **sharp** (≈ 0.75). High dimension does not buy sharpness; low recoverability does.

## Falsifier (any one ⇒ the recoverability framing is wrong / dimension matters)

- `sharpness` **rises with `D`** at fixed `ρ_rec` (dimension is the axis after all); or
- `control_suff` **collapses at high `D`** under fixed compute (dimension matters via a control-compute cost — sharpness would then need dimension); or
- `recon_FVE` is **beaten** by the larger reconstructor (the resistance is computational, not an information floor — different claim).

## Receipt

`scripts/resistant_body_test.py` → RESULT block appended below after the run.
