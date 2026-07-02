# AT-6 — charFun Typing Receipt

> 2026-07-02. Run of `AT6_CHARFUN_TYPING_SPEC.md` (frozen before first read).
> Script `scripts/at6_charfun_typing.py` (read-only import of the frozen C1 harness;
> harness file untouched). Artifact `results/chatv2/at6/at6_typing.json`.
> **Non-promotional. Types shadows, not resistance; no control claim; no PDE theorem.**

## Verdict: `AT6_TYPING_BROKEN(R3, R4, R5)` — at BOTH regimes, and informatively

Frozen gate: some T with all phase rows ≤ 0.55 and all component rows ≥ 0.65. No such T
exists at either G. Liveness held (all rows ≥ 0.55 at T = 1; components ≈ 0.98–1.00
as the intended anchor). Per spec, the broken rows are recorded, not rescued.

| G=200 | T=1 | T=50 | T=1000 | T=5000 | | G=300 | T=1 | T=50 | T=1000 | T=5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 comp | .998 | .996 | .991 | .991 | | R1 | .979 | .981 | .957 | .968 |
| R2 comp | .996 | .996 | .998 | 1.00 | | R2 | .989 | .991 | .961 | .831 |
| R3 phase | .985 | .985 | .983 | .981 | | R3 | .987 | .985 | .983 | .991 |
| R4 phase | .998 | .998 | .994 | .994 | | R4 | .921 | .927 | .739 | .803 |
| R5 phase | .925 | .927 | .916 | .914 | | R5 | .938 | .942 | .711 | .625 |

Mechanism |⟨e^{iθ}⟩_T|: 1.00 → 0.96 (G=200) / 0.93 (G=300) across the whole grid.

## Diagnosis (post-read diagnostics, non-gate-bearing; verdict unchanged)

1. **R3 was mis-registered in *scale*, not kind.** The mode-(1,0) phase is balanced (0.50 /
   0.54) and does rotate — ~19–22 rad over the full 1,000-t.u. stream, i.e. period
   ≈ 3×10⁴ steps ≫ T_max = 5,000. Within the registered grid it is locally quasi-static
   (the mechanism curve ≈ 1.0 confirms), so averaging cannot wash it *in-grid*. The charFun
   dichotomy is untested by this row, not violated: washing needs window ≫ period.
2. **The real finding (R4/R5 at G=200): amplitude-washing ≠ decodability-washing on a
   noiseless deterministic cell.** E_low's burst cycle has autocorr first-zero ≈ 261 steps;
   T = 5,000 averages over many cycles, yet cycle-timing (R4) stays decodable at 0.994.
   Averaging attenuates the oscillatory component toward zero — as the in-tree theorems
   say — but a z-scored linear readout **rescales** the attenuated ripple; at float64 with
   no observation noise, decodability tracks separability, not amplitude. This is the
   portfolio's ceiling-vs-crossover lesson appearing a third time, now on the PDE side:
   absolute "washed to chance" is a *noise-floor* statement, and the cell as simulated has
   no noise floor.
3. **Where the dynamics supply effective noise, the predicted pattern appears.** At G=300
   (aperiodic: no autocorr zero-crossing in 20k steps) R4/R5 decay with T (0.92→0.80,
   0.94→0.63) while R1 holds — the charFun typing is directionally visible exactly where
   trajectory complexity plays the role of noise. Also noted: R2 slides late at G=300
   (0.83 at T=5000) — component rows are not immortal either once T approaches residence
   scales.

**Bridge honesty (the named import earned its fence):** the in-tree `ShadowDecay*` theorems
average over a *measure*; the spec's named import — "physical time-average = this averaging
map" — is exactly where the run strained: single-trajectory deterministic averages preserve
linear decodability even as amplitudes damp. The Lean results are untouched (they are
amplitude/measure statements); the *bridge* now has a measured boundary: it holds in the
presence of an observation-noise floor, and this cell has none.

Spec-hygiene note (owned): spec §2 said all labels are "balanced by construction"; R3's
balance was in fact empirical (it landed 0.50/0.54, so no harm — recorded for accuracy).

## Consequence for AT-4 (the slate's designed sequencing: AT-6 re-registers AT-4's inputs)

1. **The surface taxonomy must be SNR-aware.** Order-blind window statistics on a
   deterministic cell do not lose information by averaging alone. AT-4's registered surface
   family should carry a declared observation-noise model (or quantized/binned readouts) so
   that "surface-washed" is a well-defined, achievable condition — otherwise AT-4's
   crossover would be fighting the same no-noise-floor artifact, not the phenomenon.
2. **Label frequency must sit inside the window grid** (R3's lesson): no slow-phase labels
   whose period exceeds T_max.
3. G=300 is the friendlier regime for any washing-based construction (its aperiodicity
   supplies the effective noise); G=200's near-regularity makes everything decodable —
   consistent with the C1 discriminator's own G=200 anomaly being the *predictable*-side
   surprise.

## Status

AT-6 filed: `AT6_TYPING_BROKEN` at both regimes — the entry's own falsifier branch, doing
its stated job (feeding AT-4's re-registration). The optional Lean shim (rotation-vs-two-
point interpretation lemma on `ShadowDecayLattice`) remains FORMALIZABLE and is now *more*
motivated: it would pin the amplitude half formally while the receipt pins the
decodability half empirically. Not built here.

Cross-refs: `NSE_ATTRACTOR_TAIL_HYPOTHESES.md` AT-6/AT-4, `AT6_CHARFUN_TYPING_SPEC.md`,
`results/chatv2/at6/at6_typing.json`, `sundogcert` `ShadowDecay{General,Cauchy,Lattice}.lean`.
