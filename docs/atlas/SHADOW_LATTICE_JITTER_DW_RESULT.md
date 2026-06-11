# HS10 RESULT — LATTICE-FRAG: CONFIRMED (with one documented instrument deviation)

**Run:** 2026-06-11 (prereg frozen same day at `77e38e7c`, before any code).
**Prereg:** `docs/atlas/SHADOW_LATTICE_JITTER_DW_PREREG.md`.
**Outcome: CONFIRMED — clean confirmatory success.** Both pre-registered kills passed with wide
margin; the law's SURVIVE clause now carries a quantitative atomicity tolerance. One deviation
(D1, instrument repair after a run-1 gate abort) is documented below; kill thresholds, θ, grids,
seeds, and n were never touched.

---

## Headline

On the frozen S0 v2 band-pass apparatus with averaging population ξ = ±1 + ε·N(0,1):

| | result | threshold |
|---|---|---|
| **K1 bridge RMS** (R_emp vs analytic-through-estimator, masked, pooled main ε) | **0.0048** | kill if > 0.10 |
| **K2 horizon exponent** (OLS log λ* vs log ε) | **−1.088** (R² = 0.980) | kill if outside [−1.25, −0.75] |
| horizon constant | C = λ*·ε = **0.327** | (convention-dependent, guidance was 0.36–0.40) |

λ*(ε): **2.516 / 1.818 / 1.367 / 0.654** at ε = 0.15 / 0.2 / 0.3 / 0.5 — every crossing inside its
Nyquist-valid window, per-ε bridge RMS 0.0041–0.0057.

**What this banks:** the jittered-lattice recovery horizon *reduces to the Debye-Waller factor*
through the frozen apparatus — the readout-to-envelope bridge holds at 20× inside tolerance, and
the horizon obeys the 1/ε law. The lattice SURVIVE clause is no longer exact-atomicity-only
(measure-zero physically): survival degrades with analog jitter exactly as the charFun product law
prescribes, with the recovery horizon at λ* ≈ 0.33/ε on this apparatus. As pre-stated, the
qualitative wash was already a theorem; what was at stake (and passed) is the quantitative bridge
every future "this wash means resist" interpretation leans on.

## Gates (all PASS in run 2)

- **G1** ε=0 denominator control: RMS(â − pred) over 110 λ = **0.0045** ≤ 0.10 — the apparatus
  reproduces its own band-dephasing structure (the banked cont(λ=2)=0.655-vs-envelope-1.0 gap).
- **G2** regression: existing pops byte-unchanged (lattice [0.7289, 0.4391, 0.5373, 0.566],
  gaussian [0.6939, 0.5059, 0.0, 0.0] — pinned in the frozen test).
- **G3** pairing determinism byte-identical; R(λ=0.05) = 0.9997/0.9995/0.9988/0.9968 (|R−1| ≤ 0.02).
- **G4** split-half SE at the crossings: max **0.0247** ≤ 0.03.
- **Sanity** (no early collapse): min masked R_emp = 0.9842 (ε=0.01), 0.8696 (ε=0.03) ≥ 0.8.

## Deviation D1 — instrument v2 after the run-1 gate abort (the gates working as designed)

**Run 1** (band-aggregate amplitude ratio â(λ,ε)/â(λ,0), exactly as the prereg §2 pinned it)
**GATE-ABORTED on G4** (split-half SE 0.0419 > 0.03 at the ε=0.5 crossing). Diagnosis from run 1's
own table: λ* = 1.552/1.509/1.471 for ε = 0.15/0.2/0.3 — *flat*, parked at the λ≈1.5 recurrence
null (ε=0.5's 0.530 at the 0.5 null), while K1's per-ε RMS was 0.006–0.019, i.e. R_emp ≈ R_pred
everywhere. So the dips were in the **prediction too**: near band-straddling zeros of cos(2πλt),
the DW reweighting displaces the numerator's zero from the denominator's, the band-aggregate
ratio excurses **structurally**, and the θ-crossing detector measured null positions, not DW
horizons. The prereg's "n is a power knob" remedy could not have fixed this — the honest record is
that the v1 observable's cancellation claim was only approximate, and G4 caught it.

**The repair (v2):** take the ratio **per-t**, where the cancellation is exact
(ρ_t = f_t(ε)/f_t(0) → DW(2πλtε) pointwise), and aggregate with pred-derived weights
W(λ,t) = (env_f(t)²·f_pred_t(λ,0))², which kill the per-t nulls quadratically. The v2 observable
is smooth in λ; the prereg's DENOM_FLOOR mask is superseded (Nyquist window retained). Frozen-test
additions verify v2 at machine precision (λ=0 noiseless anchor, 1.8e-15) and verify the structural
fix directly (R_pred at λ ∈ [1.4, 1.6], ε=0.15 stays on the DW plateau 0.82→0.73 where v1 dove
below θ). **Unchanged:** K1/K2 thresholds, θ = 0.5, ε sets, λ grids, seeds, n, the paired-draw
design, G1–G4, the analytic prediction discipline (charFun pushed through the same estimator on
the same draws — forward-generated, no fitting). Run 1 is preserved at
`results/shadow/hs10_lattice_jitter_dw_run1_gateabort.json`; run 2 (the verdict run) at
`results/shadow/hs10_lattice_jitter_dw.json`.

## Secondary (no kill): the frozen recovery readout corroborates

`sweep_pop('lattice_jitter')`, frozen LAMBDAS, n = 600:

| ε | cont(λ=2) | recovery half-life λ*_c | min disc |
|---|---|---|---|
| 0.15 | 0.56 | censored (> 2.0) | 1.00 |
| 0.2 | 0.49 | censored (> 2.0) | 1.00 |
| 0.3 | 0.30 | 1.5 | 1.00 |
| 0.5 | 0.00 | 0.75 | 1.00 |

The probe-level horizon closes with ε in the same direction and magnitude regime
(λ*_c·ε ≈ 0.38–0.45 where observable), and **disc = 1.00 throughout** — jitter keeps a finite
centered mean, so determination is untouched (the theorem, reported not claimed).

## Frozen test

`scripts/test_shadow_lattice_jitter_dw.py` — **20/20**: G2 regression pins, paired-draw contract,
charFun product law, v1+v2 machine-precision anchors, null-robustness, mask logic, crossing
interpolation, and three banked byte-stable pins (R_emp(1.0, 0.3) = 0.639625…,
R_emp(2.5, 0.15) = 0.505140…, R_emp(0.65, 0.5) = 0.503855…).

## Follow-on (owner-gated, not this run)

- **Lean:** charFun-of-convolution = product (`ShadowDecay` tower), making the jittered-lattice
  envelope cos(s)·e^(−ε²s²/2) machine-checked — the slate-named next theorem.
- HS11 (detector sensitivity floor) runs on this same frozen apparatus and is the complementary
  parked leg.

*Attribution: Debye 1913 / Waller 1923; Lukacs, "Characteristic Functions". Nearest banked priors
and the delta: see prereg §5.*
