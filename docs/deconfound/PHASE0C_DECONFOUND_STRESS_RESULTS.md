# Deconfound Phase 0C - De-confound-Stress Boundary Results

**Verdict: `deconfound_load_bearing_confirmed`** (2026-06-04).

As the input-deconfound leaks, the state-keeper body exposes the constructed functional `u`
**monotonically more** — the Phase-0B double-dissociation degrades exactly in step with the
de-confound. The input-deconfound is the **necessary precondition** for the closure read, shown
not asserted.

Binding spec: `docs/deconfound/PHASE0C_DECONFOUND_STRESS_SPEC.md` (re-locked).
Runner: `scripts/deconfound_attack_b_phase0c_stress.py`.
Summary: `results/deconfound/attack-b-phase0c-stress/summary.json`.

## Provenance (readback)

- git SHA: `71c96f62`; runner sha256 (16): `1A0923BFFF77B763`; imported (unedited) 0B runner
  sha256 (16): `B3A49A8B89EAB53B` (matches the Phase-0B receipt — no edit to the locked 0B runner).
- substrate: frozen 0B digit features, z-scored + shared-factor injection
  `b_j = 1[zscore(feat_j) + alpha*g > median]`, `g = default_rng(20260604).standard_normal`,
  `u = XOR(b_0,b_1,b_2)`. `alpha=0` asserts-equal the 0B substrate.
- calibration: `results/deconfound/attack-b-alpha-calibration-2026-06-04.txt` (rungs locked
  pre-run; not tuned during the run).
- run: 3 seeds `{0,1,2}`, `n_perm = 1000`, wall 317.6s (inline). `u_null` clean
  (`k_null = none`) in every interpreted body/seed -> no `closure_void_control`.

## Headline — `state_det_u` (state-keeper max selection-corrected-significant `det(u)`) vs the leak

| alpha | input `det(u\|b)` | label | `state_det_u` (median) | interpreted seeds |
| ---: | ---: | --- | ---: | ---: |
| 0.00 | +0.077 | HOLD | **0.192** (`s0`) | 3 |
| 0.75 | +0.190 | MARG | 0.423 | 3 |
| 1.00 | +0.231 | LEAK | 0.522 | 2 |
| 2.50 | +0.459 | LEAK | **0.654** (`sdeep`) | 3 |

`rise = sdeep - s0 = 0.461` (>> the 0.15 load-bearing bar). `state_det_u` is **monotone** in the
input leak and tracks it amplified ~1.4-2.5x (the nonlinear state body makes `u` more linearly
accessible than the raw input). `alpha=1.0` carried 2 interpreted seeds (one body missed a
learned-body gate); the headline uses the 3-seed endpoints `s0`/`sdeep`, so this does not affect
the verdict, and the MARG/LEAK rungs corroborate the monotone trend.

## Not a retro-flag of Phase 0B

`HOLD state_kfunc_hits = 0` — at the clean substrate the state body does **not** expose `u` at
the inherited 0B determination bar (`k_func` at det>=0.70). So 0B's binary read **replicates**;
`closure_confounded_throughout` does not fire.

**Finer-grained nuance (recorded, not a refutation):** the continuous read shows the state body
weakly exposes `u` even at HOLD (`s0 = 0.192`), below the 0.70 bar 0B used and ~5x below the
functional-keeper's ~1.0. 0B's dissociation holds at its bar; the continuous instrument simply
resolves a weak nonlinear baseline the binary read could not see. (A 1-seed smoke read
`s0 = 0.148`; the 3-seed median is 0.192.)

## Honest tier and boundaries

**R1.5 methods-validation.** Does **not** lift the ceiling — the functional is still constructed.
It converts "the input-deconfound is the matched guard" from a claim into a **measured
dependency**: the closure read degrades smoothly as the guard fails. This makes the lane's core
validity gate demonstrably load-bearing and pre-empts the sharpest methods question ("is the
de-confound criterion adequate?") with a degradation curve.

**Allowed (per spec §6):**
> On real digit features with an injected correlation knob, the closure double-dissociation
> degraded as the input-deconfound leaked: the state-reconstruction body's exposure of the
> constructed functional `u` rose monotonically with the linear presence of `u` in the input.
> This locates the input-deconfound as the necessary precondition for the determining-shadow
> closure read.

**Forbidden:** claiming the injected correlation is "natural" (it is injected; the
native-correlation variant is under-pooled pixels, a separate cell); model-discovered functional;
Othello rescue; real-JEPA behavior; R2 / "more than we know". External review before promotion.
