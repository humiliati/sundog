# Deconfound Phase 0B - Attack-B Real-Feature Closure Results

**Verdict: `attack_b_closure_confirmed`** (2026-06-04).

The determining-shadow-set closure read ports to a de-confound-clean **real-feature**
substrate as a clean **double dissociation**: a supervised functional objective produced a
closure bracket (`k_func << k_state`) that the state-reconstruction objective did not.

Binding spec: `docs/deconfound/PHASE0B_ATTACK_B_CLOSURE_SPEC.md` (locked).
Runner: `scripts/deconfound_attack_b_phase0_closure.py`.
Summary: `results/deconfound/attack-b-phase0-closure/summary.json`.

## Provenance (readback §9)

- git SHA: `1f0a60a4`; runner sha256 (16): `B3A49A8B89EAB53B`.
- substrate: `sklearn.datasets.load_digits`, 8x8 -> 8 pooled (4x2) median bits; `u = XOR(b_0,b_1,b_2)`, outside `b_j, j in {3..7}`.
- run: 5 seeds `{0..4}`, `n_perm = 1000`, wall time 100.4s (inline; under the ~10-min rule).
- **de-confound replay:** linear input-probe `det(u|b) = 0.0771 <= 0.20` (HOLD) — matches the 0-pre row; `u` is not linearly in the input, so the functional objective genuinely *computes* the parity.
- split base rates `[0.530, 0.531, 0.531]` (within 0.08, OK).

## Learned-body gates

| body | per-seed result |
| --- | --- |
| functional-keeper (rd8) | learned 5/5 (`u` acc 1.00, det 1.00) |
| state-keeper (rd8) | learned 5/5 (mean-bit acc 0.98-0.99, det 0.96-0.98) |
| functional-keeper-compressed (rd3, diagnostic) | learned **4/5** (seed 2 `u` det 0.692, just under the 0.70 gate) |

## Determining-shadow read (representative seed 0; pattern identical across all 5 seeds)

| body | `k_func(u)` | `k_state(b_j∉S)` | `k_null` | margin | bracket |
| --- | ---: | ---: | ---: | ---: | :--: |
| functional-keeper | 2 (det 0.870, p 0.001) | **none** | none | 7 | **yes** |
| state-keeper | **none** | 4 (det 0.813, p 0.001) | none | 0 | no |
| compressed (rd3) | 2 (det 0.719, p 0.001) | none | none | 2 | yes (where learned) |

All selection-corrected p-values = 0.001 (1/1001), well under the 0.01 bar. `k_null = none`
in every interpreted body/seed (no `closure_void_control`).

## Headline (paired by seed)

| quantity | value |
| --- | --- |
| functional-keeper closure bracket | **5/5** seeds (require >=4/5) |
| state-keeper closure bracket | **0/5** seeds (require <=1/5) |
| median paired `keeper_gap` | **7.0** (require >=2) |
| `u_null` negative | all interpreted seeds |

**Primary pass clears on all four conditions.** The compressed diagnostic is *not consulted*
(the functional-keeper is never stateful); where it learned it also bracketed, corroborating.

## Why this is real, not a probe artifact

The **state-keeper determines `b_j∉S` at k=4** (det 0.81) — so the probe *can* find the
outside features when a body carries them. The functional-keeper's `k_state = none` is
therefore **genuine discard**, not probe weakness. Conversely the state-keeper's
`k_func(u) = none` shows the parity is not linearly exposed by a state-carrying body. The two
objectives produce **opposite** observability structure on the *same* real features.

## De-confound boundary — demonstrated, not asserted

A nonlinear self-test (`scripts/deconfound_nonlinear_selftest.py`; receipt
`results/deconfound/attack-b-nonlinear-deconfound-2026-06-04.txt`) makes the R1.5 ceiling
explicit:

| probe `b -> u` | acc | det |
| --- | ---: | ---: |
| linear (LogReg) | 0.567 | +0.077 (de-confound HOLDS) |
| nonlinear (MLP) | 1.000 | +1.000 |
| nonlinear control `b -> random` | 0.500 | -0.036 |

`u` is **linearly-hidden but nonlinearly-trivial** — an explicit function of visible inputs.
The random-label control (det ~ 0) confirms the MLP reads real structure, not overfitting. So
the linear de-confound is the *matched* guard for a *linear* determining-shadow read, and is
inadequate for any "u is hidden / more than we know" claim **by construction** — which is
exactly the R1.5 boundary. This pre-empts the sharpest methods question (is linear `b -> u`
too weak a leakage test?) with a concrete answer: yes, linear-only, and that *is* the ceiling.

## Honest tier and boundaries

**R1.5** — real feature distribution + a real (MLP) architecture, but the functional is
**constructed (ours)**, so this does **not** advance "more than we know." It is the
substrate-port of the calibrated closure read onto a de-confound-clean *real* substrate, plus
the keep-functional-vs-keep-state (JEPA-principle) contrast measured on real data.

Caveats: the de-confound holds partly because digit-pooled correlation is mild (~0.11, far
below the ~0.5 leak boundary located by the 0-pre sweep); a richer real substrate would need
its own 0-pre. "Functional objective" here = supervised-on-`u`, a clean instance of *targets
the functional*, **not** a JEPA implementation.

**Allowed (per spec §7):**
> On median-binarized sklearn digit features, a supervised functional objective produced a
> closure bracket (`k_func << k_state`) that the state-reconstruction objective did not, under
> a passed input-deconfound precheck and an independent `u_null` control.

**Forbidden:** model-discovered functional (it is constructed); Othello closure rescued;
real-JEPA behavior (this is a JEPA-*principle* contrast); R2 / "more than we know". External
review before any promotion.
