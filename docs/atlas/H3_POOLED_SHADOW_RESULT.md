# H3 Result — pooled activations as a determine/resist shadow (the imported-wall test, ANSWERED)

> **Run against the LOCKED pre-registration** (`H3_POOLED_SHADOW_PREREG.md`). Hypothesis #3 of slate
> `ww6koomb1`. The Shadow-Invertibility Law's mechanism was proved synthetically (S0/S1), on halo physics
> (S2 partial), and in Lean (the charFun law) — but whether a **real trained body** instantiates the
> lossy-averaging split was the standing *imported wall* ("named, not proved"). H3 answers it. **NOT
> public-eligible.** Attribution: the Shadow-Invertibility Law (`ATLAS_PHASE5_CROSS_SUBSTRATE.md`);
> Debye–Waller; DeepSets (Zaheer 2017); random Fourier features (Rahimi & Recht 2007).

> **CORRECTION (2026-06-11, H3-PC / slate-2026-06-10 HS4 — `H3_PROBE_CEILING_RESULT.md`):** the
> clause "clf_d suppresses c to ~0 (probe-robust under any probe)" is RESCOPED. A calibrated battery
> at n=20k recovers residual c from the same clf_d reps at R² = 0.1256 (Nyström kernel ridge,
> replicated on a once-touched split) and 0.067 (plain ridge), with independent KSG-MI confirmation;
> the ≈0.006 readout below is an N=2000 artifact (the learning curve passes through it at n=2000),
> and the "strong MLP" probe underwriting "probe-robust" is demonstrably blind to 0.20-ridge-equivalent
> calibrated injections (MEMBER-BLIND — its silence certifies nothing). **Correct reading: partial
> concealment, not destruction-to-zero.** The objective gap survives rescoped (reg_c 0.51 vs clf_d
> ceiling ≈0.126); the DETERMINE half and the objective-dependence headline are unaffected.

## Headline (BOUNDED-POSITIVE, reproducible)

**A trained body's mean-pooled activation does NOT generically inherit the Shadow-Invertibility
continuous-resist. The resist assumed a *linear* averaged shadow; a *nonlinear* per-unit encoder
defeats it, and the degree of defeat is OBJECTIVE-DEPENDENT.** The discrete latent is **determined**
through the pooling in every body. Precisely, once raw (linear) averaging has fully washed the
continuous `c`:

- a body trained to **keep** `c` (`reg_c`) **recovers** it post-pool by learning a nonlinear
  *demodulate-then-pool* code (c-R² up to 0.84, holds to λ=2.0; **probe-robust** ≈0.5 under both a
  linear and a strong nonlinear probe);
- the **same architecture + training** that only classifies `d` (`clf_d`) **suppresses** `c` to ~0
  (probe-robust under any probe) — the pure **objective gap reg_c − clf_d ≈ 0.50** at λ=2.0;
- the discrete `d` is **determined** (balanced-acc ≥0.99) at every λ in every body — structurally
  stable through genuine lossy averaging.

This *sharpens* the imported wall and the §4 founding-theorem correction: the continuous-resist side is
not merely fragile in principle — an incentivized trained encoder actively **defeats** it; the
discrete-determine side is **robust**.

---

## The arc: v1 was confounded → v2 is the corrected test

**v1** (`scripts/shadow_pooled_synthetic.py`, run via a 7-agent build→verify→synthesize workflow) was
**confounded**, and the adversarial verification panel caught it: it made the continuous `c` a *shared
mean* across units, so mean-pooling trivially **concentrated** onto `c` (a sufficient statistic) — raw
`mean_i u_i` with **no encoder and no training** recovered `c` at R²≈0.94. The discrete `d` was a
*noiseless broadcast constant* (zero lossiness → trivial preservation). So v1 tested nothing about the
trained body. (Reproducibility defect too: hash-salted seed + unpinned BLAS.) **The confound is itself
the lesson:** mean-pooling resists a continuous latent *only when that latent is orthogonal to the
pooled axis* — i.e., carried in the per-unit fluctuation being averaged out (as in S0/Debye–Waller),
**not** when it is a shared latent the units jointly estimate.

**v2** (`scripts/shadow_pooled_synthetic_v2.py`) fixes the construction so raw averaging genuinely
**washes** `c` (a *hard anti-confound gate*), making any post-pool recovery attributable to the trained
encoder:
- `c` lives in a per-unit **random-Fourier fringe** `cos(ω_m c_i + ψ_m)`, `c_i = c + λ ξ_i`, with high
  frequencies `ω_m∈[3,6.5]` so each component Debye–Waller **washes** under the spread, yet `c` is
  **identifiable from a single unit** (RFF is injective);
- the key subtlety: a *linear* readout of the fringe still washes with the raw mean (averaging commutes
  with linear maps), so **only a nonlinear demodulator** makes `mean_i φ(u_i) ≈ mean_i c_i = c` — the
  trained nonlinear encoder is genuinely load-bearing;
- `d` is carried *through* lossy averaging (per-unit noisy channel `d·a + η_i`, **not** a broadcast
  constant), surviving by structural stability;
- reproducible: `torch.set_num_threads(1)` + fixed integer seeds (byte-identical across reruns).

---

## v2 results (seed 1234, K=64, N_train=8000, train-λ=1.0; c = Ridge 5-fold R², d = balanced-acc)

c-RECOVERY (R²) vs λ:

| feature \ λ | 0.0 | 0.3 | 0.5 | 0.75 | **1.0** | **1.5** | **2.0** | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|
| raw mean (no φ) | 1.00 | 0.98 | 0.94 | 0.58 | **0.01** | 0.00 | 0.00 | 0.00 | 0.00 |
| single unit (C1) | 1.00 | 0.48 | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| random-φ pool | 1.00 | 0.96 | 0.90 | 0.80 | 0.67 | 0.43 | 0.23 | 0.12 | 0.08 |
| **clf_d pool** | 0.97 | 0.79 | 0.36 | 0.16 | **0.06** | 0.02 | **0.01** | 0.00 | 0.00 |
| recon pool | 1.00 | 0.98 | 0.94 | 0.59 | 0.04 | 0.08 | 0.09 | 0.06 | 0.02 |
| **reg_c pool** | 1.00 | 0.98 | 0.95 | 0.91 | **0.84** | 0.66 | **0.51** | 0.31 | 0.20 |

d-RECOVERY (balanced-acc): **≥0.99 at every λ for every body and for the raw mean** (chance 0.5).

**Washed regime** = λ≥1.0 (raw c-R² ≤ 0.013). There, `reg_c` recovers `c` while raw/clf_d wash.

### Gate scorecard
| Gate | Result | Value |
|---|---|---|
| C0 anti-confound (raw washes `c`) | **PASS** | onset λ=1.0, max raw-c in regime 0.013 |
| C1 (`c` in a single un-pooled unit) | **PASS** | unit c-R²(λ=0) = 0.999 |
| DETERMINE (`d` through lossy avg) | **PASS** | min d-acc = 0.995 |
| **DEFEAT** (reg_c recovers `c` where raw washed) | **PASS** | peak 0.84, holds (≥0.4) to λ=2.0 |
| Objective gap (reg_c − clf_d, λ=2.0) | — | **0.50** |
| reg_c train-fit (demodulation learned) | — | 0.850 |

### Adversarial self-checks (probe-robustness + reproducibility)
- **reg_c recovery is real:** λ=2.0 ridge 0.51 / strong-MLP 0.46; λ=1.5 ridge 0.66 / MLP 0.63.
- **clf_d suppression is real (not a linear-probe artifact):** c-R² ≈ 0.00 under *both* linear and
  strong probes. So the objective gap is **probe-robust**.
- **Nonlinearity floor (honest caveat):** a random untrained φ-pool retains a *weak, linear-probe-
  dependent* `c`-signal (ridge 0.23–0.67, but ~0 under the strong MLP) — ReLU rectifies the
  `c`-distribution into a pooled signal. This is real but secondary; the **dominant, probe-robust**
  effect is the training objective.
- **Defeat weakens at very high spread:** reg_c c-R² decays 0.51→0.31→0.20 (λ=2.0→2.5→3.0) as `c_i`
  extrapolates beyond the training range. The defeat is strong but not unlimited.
- **Reproducible:** byte-identical c-R² across reruns and retrains (single-thread + fixed seeds).

Frozen test `scripts/test_shadow_pooled_synthetic_v2.py` locks: C1, C0, DEFEAT, clf_d-suppression,
objective gap ≥0.30, probe-robustness (both directions), DETERMINE, and reproducibility.

---

## Substrate B — real MNIST CNN (external validity; from the v1 workflow, honest read)

> **CORRECTION (2026-06-11, H3-PC-B / slate-2026-06-10 HS4 second leg —
> `H3_PROBE_CEILING_MNIST_RESULT.md`):** P-B1's "partial attenuation" is RESCOPED to the LINEAR
> readout. On a continuity-gated re-derivation of this body, the calibrated battery recovers θ from
> the GAP shadow at **R² = 0.5651 (pool-CV) / 0.5606 (once-touched split, Nyström)** vs a pre-GAP
> linearly-readable anchor of 0.6239 — residual attenuation ≈ 0.06, and the pre-registered
> "attenuation illusory" bar was missed by 0.009. The "~0.47 strong-probe" figure below (never
> receipted in the banked JSON) is adjudicated **superseded upward**. The linear statement
> (0.62 → 0.34) stands; the determine half is untouched. The ceiling is a lower bound (learning curve
> unconverged at n=20k).

A tiny CNN (2 conv → **global average pool = shadow** → FC), trained to classify digit `y` on MNIST
with an applied rotation nuisance `θ ~ U[−30°,30°]`; held-out CNN acc 0.83.

- **Determine (P-B2): PASS** — post-GAP `y`-acc 0.87 (8.7× chance); permutation control → chance.
  *Partly trivial* (y is the training target).
- **Resist (P-B1): PARTIAL** *(rescoped — see the 2026-06-11 correction above: a linear-probe
  statement only)* — `θ`-R² 0.62 (pre-GAP) → 0.34 (post-GAP, linear) but **~0.47 under a strong
  nonlinear probe**, so GAP only *partially* attenuates θ (true drop ~0.15, not a wash).
- **Sweep (P-B3): endpoint-positive, non-monotone** — ensemble-spread over augmentations washes θ-R²
  0.34→0.15 (λ=0→4) while `y` holds, but the curve rises through λ≤1 and the λ=4 point uses rotations
  beyond the trained domain (partly distribution-shift). KILL-B1 not fired (θ not fully recoverable).

So a real CNN's GAP shadow **determines** the class and — per the 2026-06-11 correction — only
**thinly attenuates** the continuous nuisance against its linearly-readable content (≈0.06 residual;
the apparent 0.62→0.34 drop was mostly probe-relative). Consistent with v2's sharpened reading: a
trained nonlinear body does not inherit the linear-averaging resist; what the shadow "hides" depends
on the probe class brought to it.

---

## The sharpened imported wall

| | Shadow-Invertibility Law (as imported) | H3 finding (trained body) |
|---|---|---|
| continuous-resist | continuous washes under lossy averaging | holds only for **linear** averaging; a **nonlinear** encoder defeats it, **objective-dependently** (reg_c demodulates → recovers; clf_d → suppresses) |
| discrete-determine | discrete survives (structural stability) | **holds** — `d` determined through lossy averaging in every body |
| lossiness essential | injective shadow loses nothing | holds (λ=0 recovers `c`) |

**§4 tie-in:** the founding theorem bet on the continuous (`x_c`) resisting. H3 shows that side is not
just fragile but **actively defeatable** by an incentivized trained encoder, while the discrete
(`x_d`, certificate/parity/coset) **survives** — strengthening the §4 correction that the
alignment-relevant property lives on the discrete/determinable side.

## Honest boundaries
- Synthetic substrate (S0-style fringe + a DeepSets body); the toy `d` is parity/class, not a
  certificate/coset — the alignment leap remains a separate assumption. Forward-only, no inversion.
- The defeat is partly **architectural** (nonlinearity) and partly **objective** (training); the
  controlled, probe-robust claim is the **reg_c vs clf_d** contrast.
- The defeat weakens at very high spread (extrapolation); the nonlinearity floor is probe-dependent.
- Substrate B's resist is *partial* and its sweep non-monotone (see above) — a real-body positive with
  documented caveats, not a clean wash.

## Files
- `scripts/shadow_pooled_synthetic_v2.py` (+ `test_shadow_pooled_synthetic_v2.py`) — the corrected leg.
- `results/atlas/h3/synthetic_v2_result.json` — the v2 sweep + gates.
- `scripts/shadow_pooled_synthetic.py` / `shadow_pooled_mnist.py` + `results/atlas/h3/*.json` — v1
  (confounded synthetic; partial-positive MNIST) and its adversarial verification.
- `docs/atlas/H3_POOLED_SHADOW_PREREG.md` — the locked pre-registration.
