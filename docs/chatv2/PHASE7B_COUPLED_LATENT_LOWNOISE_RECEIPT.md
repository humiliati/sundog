# Chat-v2 Phase 7b — Low-Noise Coupled Closure Receipt

> 2026-06-04. Run + adjudication of the frozen `PHASE7B_COUPLED_LATENT_LOWNOISE_SPEC.md`
> (single-knob change off Phase 7: `p_noise 0.25 → 0.10`). **Status: internal receipt
> — the lane's first `closure_confirmed`.** Toy closure on a *designed* coupled
> substrate at *deliberately low* noise. **Not** a real-LLM or NSE claim; promote-gate
> **R1 not met**. Its value is a **positive control** for the instrument + a mapped
> noise-survival boundary.

## Provenance

- Runner: `chatv2_phase0_bodyresist.py --latent coupled --p-noise 0.10` (train);
  `chatv2_phase7_probe.py --mode coupled --dir phase7b-coupled-lownoise` (probe).
- git `c59e6a98` (1 dirty file). Training ~2.15 h (3 seeds, bg, GTX-1080);
  probe ~16–22 min (bg). De-confound PASS `0.500`.

## Training (all 3 gen learned)

| seed | gen `eval_loss` | learned | z1 (readout) | body_carry gen/twin | Phase-0 verdict |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.4791 | ✓ | 0.94 | 0.90 / 0.60 | SHARP |
| 1 | 0.4798 | ✓ | 0.94 | 0.87 / 0.68 | MARGINAL |
| 2 | 0.4743 | ✓ | 0.90 | 0.88 / 0.58 | SHARP |

`z1 ≈ 0.90–0.94` — the low-noise readout is **far cleaner** than Phase 7's ~0.69; that
clean readout is what lets the closure survive.

## Verdict: `closure_confirmed` (all 3 seeds)

| seed | k=2 func(u) | **k_func** | state(z)@k6/k7 | **k_state** | nf_func |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.754 | **2** | 0.732 / 0.859 | 6 | 0.662 |
| 1 | 0.704 | **2** | 0.689 / 0.827 | 7 | 0.647 |
| 2 | 0.753 | **2** | 0.726 / 0.889 | 6 | 0.595 |

**`k_func = 2 ≪ k_state ∈ {6,7}` (gap 4–5) on every seed.** A 2-latent shadow set
determines the hidden coupling source `u` (closure) while the individual states resist
until k=6–7 — the NSE-like functional-closure-before-state-reconstruction bracket,
measured in a trained body. `func(u)` at k=2 (0.70–0.75) clears **both** the frozen
0.70 bar **and** the selection-corrected null.

## The closure survived the readout

Data `func(u) @ k=3 = 0.905` → trained body **0.77–0.86** (wash ~0.04–0.13, vs Phase 7's
~0.14 at the same k but lower data value). The clean low-noise readout preserves the
closure signal; Phase 7's `p=0.25` washed it below the bar, Phase 7b's `p=0.10` does not.

## Honest nuances (disclosed)

- **The null floor rose to 0.60–0.66** (vs Phase 7's 0.53). At low noise the body
  encodes the `z`'s so cleanly that even *random* directions partially predict `u`
  (selection optimism). `func(u)` at k=2 still clears it — the *learned* readout beats
  random — but the margin is modest. **seed1's k=2 is thin** (0.704 vs bar 0.70 / null
  0.647); it is solid by k=3 (0.773). The load-bearing signal is the **gap**
  (`k_func ≪ k_state`), robust on all three.
- **Standing `u_null` control clean** (Phase 7 §3: `k_func = none`, func ≈ 0.50 on an
  independent target) — certifies the runner does not manufacture closure. The Phase-7b
  positive is credible against it.

## Noise-survival boundary (the scientific object)

| p_noise | data `func(u)` | trained-body `func(u)` | verdict |
| --- | --- | --- | --- |
| 0.25 (Phase 7) | 0.757 | ~0.63 (< bar) | `closure_washed_by_readout` |
| 0.10 (Phase 7b) | 0.905 | ~0.84 (k=3) | **`closure_confirmed`** |

The closure crosses into trained-body recoverability **somewhere in `(0.10, 0.25)`**.
A finer sweep (`p ∈ {0.15, 0.20}`) would locate the threshold — a separate future cell.

## Tier + what this establishes

A **positive control** for the determining-shadow-set instrument: it proves the
instrument **can measure a surviving `k_func ≪ k_state` bracket** when one exists and
survives the readout — not only nulls (Phase 2) and washes (Phase 7). That makes the
lane's negatives **trustworthy**: the instrument would have found a closure if one were
there. The whole allelopathy arc is now a calibrated set — a real negative (cross-seed),
a partial (Phase 7 wash), and a real positive (Phase 7b) — measured by one instrument
against frozen thresholds and a clean independent control.

**Still a toy.** Designed coupling, deliberately low noise, `H=8`, `d_dec≈7.6`. No
real-LLM, NSE, or scaling claim; R1 not met; unpromoted.
