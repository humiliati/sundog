# H3 pre-registration — pooled activations as a determine/resist shadow (the imported-wall test)

> **LOCKED 2026-06-08, before running.** Hypothesis #3 of slate `ww6koomb1`. Tests the
> Shadow-Invertibility Law's **imported wall**: the law's mechanism is proved synthetically (S0/S1),
> on halo physics (S2 partial), and in Lean (the charFun law) — but memory flags over and over that
> *"a real trained body instantiates this averaging"* is **named, not proved** (the registered-but-
> untested synthetic→real step). A trained net's **pooled (mean-over-units) activation IS a lossy
> averaged shadow.** Does it **determine** a discrete latent and **resist** a continuous one, and does
> the split **sharpen with lossiness λ** and **depend on the training objective**? Published as a
> verification receipt (owner decision 2026-06-11; un-promoted; not peer-reviewed).
> Attribution: the Shadow-Invertibility Law (`ATLAS_PHASE5_CROSS_SUBSTRATE.md`); Debye–Waller; DeepSets
> (Zaheer et al. 2017); the charFun-spectrum law (`SHADOW_CHARFUN_DETERMINE_RESIST_LAW.md`).

## The core claim under test

A trained body `g = head ∘ pool ∘ φ` has a **mean-pool bottleneck** `z = mean_i φ(u_i)` over `K` units.
The Law predicts: the **discrete latent `d`** (shared by all units, structurally stable) is **recovered
from `z`** (determined); the **continuous latent `c`** (carried per-unit with ensemble spread `λ`) is
**lost from `z`** (resists) — *to the extent the trained encoder `φ` did not learn an averaging-robust
`c`-code*. The crucial nuance: through a **trained** φ the split is **objective-dependent** — a body
with no incentive to keep `c` washes it (law holds); a body trained to keep `c` may learn a
pooling-robust code and **defeat** the resist (the law's limit). Both outcomes are pre-registered and
falsifiable.

---

## Substrate A — synthetic-latent mean-pool net (the controlled gold standard)

### Data (parallels S0; `K = 64` units/sample, `N ≈ 4000` train + held-out)
- Latents per sample: continuous `c ~ U[c_lo, c_hi]`; discrete `d ∈ {−1, +1}` (balanced).
- Per-unit ensemble spread (THE lossiness knob, exactly S0): `c_i = c + λ · ξ_i`, `ξ_i ~ N(0,1)`.
- Raw unit feature `u_i ∈ R^F`: a FIXED random nonlinear mixing `u_i = W₂·tanh(W₁·[e(c_i); d·b]) + noise`
  that **entangles** `c_i` and `d` (so a *trained* φ is genuinely needed to disentangle), where `e(c_i)`
  is a sinusoidal embedding of `c_i`. (W₁,W₂,b fixed random; same across the experiment.)

### Bodies (the objective-dependence leg — train all three, same architecture)
`φ` = 2-layer MLP `R^F→R^H`; `z = mean_i φ(u_i)`; head `R^H→·`. Objectives:
- **`clf_d`** — classify `d` from `z` (cross-entropy). No incentive to keep `c`.
- **`reg_c`** — regress `c` from `z` (MSE). Incentivized to keep `c` pooling-robustly.
- **`recon`** — reconstruct the *pooled raw input* `mean_i u_i` from `z` (autoencoder; told NEITHER latent).

### Probes (frozen sklearn on FROZEN reps; `c`: R² (KFold), `d`: balanced-acc over majority)
- **PRE-pool**: probe the per-unit reps `{φ(u_i)}` (mean-of-probe-over-units, or a single unit) for `c`, `d`.
- **POST-pool**: probe `z = mean_i φ(u_i)` for `c`, `d`.
- **Lossiness sweep** `λ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0}` (re-draw data per λ; the body is
  trained ONCE at a fixed train-λ, then probed across eval-λ — AND a variant trained per-λ; pre-register
  the train-once-probe-across-λ as primary).
- **CharFun tie-in (H2)**: measure the empirical per-dim distribution of `φ(u_i)` and its characteristic-
  function decay; check it predicts the `c`-resist rate (stretch, report if clean).

### Controls (non-triviality)
- **Majority baseline** for `d` (balanced-acc over chance).
- **Label-permutation**: shuffle `c` and `d` independently → all recoveries → chance (no leakage).
- **λ=0 lossless point**: post-pool `c`-recovery must be HIGH (a lossless ensemble loses nothing).
- **Pre/post gap**: the load-bearing control — at high λ, PRE-pool `c` ≫ POST-pool `c` (the *pooling*,
  not the encoder, kills `c`).

### Pre-registered predictions (A)
- **P-A1 (lossiness-essential):** λ=0 → post-pool `c`-R² high (≥0.6) for all bodies.
- **P-A2 (resist, clf_d/recon):** λ→2 → post-pool `c`-R² low (≤0.15) for `clf_d` (and largely `recon`).
- **P-A3 (determine):** post-pool `d`-acc stays high (≥0.85 over majority) across λ, all bodies.
- **P-A4 (pre/post gap):** at λ=2, PRE-pool `c`-R² − POST-pool `c`-R² ≥ 0.4 for `clf_d`.
- **P-A5 (objective-dependence — the headline):** at matched λ=2, post-pool `c`-R²(`reg_c`) >
  post-pool `c`-R²(`clf_d`) by ≥0.2 — a body trained to keep `c` learns a (more) averaging-robust code.

### Kill criteria (A)
- **KILL-A1:** post-pool `c`-R² stays HIGH (>0.5) at λ=2 for **`clf_d`** → trained bodies keep the
  continuous through pooling regardless of objective → the imported wall does NOT hold (law fails for
  trained bodies). *(A clean, important null.)*
- **KILL-A2:** post-pool `d`-acc DROPS to chance → pooling destroys the discrete → no determine half.
- **KILL-A3:** `c`-resist is λ-INDEPENDENT (flat across the sweep) → not a lossiness effect.

---

## Substrate B — MNIST CNN (real data, external validity)

### Data
MNIST with an APPLIED continuous nuisance: rotation `θ ~ U[−30°, +30°]` per image. Latents: discrete =
digit class `y ∈ {0..9}`; continuous = `θ`.

### Body
Small CNN → conv feature map → **global average pool (GAP) = the shadow** → FC head. Trained to classify
`y` (standard supervised). `θ` is a nuisance the model is NOT told to keep.

### Probes (held-out; `y`: acc over 0.1 chance; `θ`: R²)
- **PRE-pool**: probe the flattened pre-GAP feature map for `y` and `θ`.
- **POST-pool (GAP)**: probe the GAP vector for `y` and `θ`.
- **Lossiness sweep**: ensemble-average the GAP vector over `K` augmented copies of the SAME image with
  rotation spread `λ` (θ + extra N(0,λ·σ) rotations) → `θ`-recovery washes while `y` holds.

### Pre-registered predictions (B)
- **P-B1 (resist):** `θ`-R² DROPS post-GAP vs pre-GAP (the continuous nuisance is washed by spatial avg).
- **P-B2 (determine):** `y`-acc stays high post-GAP (the class survives — partly trivial as it is the
  training target; weak evidence, flagged honestly).
- **P-B3 (sweep):** ensemble-spread λ over augmentations → `θ`-R² washes further toward chance.

### Kill criteria (B)
- **KILL-B1:** `θ` fully recoverable post-GAP (`θ`-R² post ≈ pre) → real CNNs do NOT resist a continuous
  nuisance via GAP → the synthetic resist is construction-specific (an informative null sharpening the
  imported-wall claim).

---

## Honest boundaries (pre-committed)
- `y`-determine in B is partly trivial (it IS the training target); the load-bearing B test is `θ`-resist.
- A null on any kill criterion is a **bounded result**, not a failure: it would sharpen the imported wall
  to "real trained bodies instantiate the split CONDITIONALLY (on objective / architecture), not
  generically." That conditional is itself the contribution.
- This is the synthetic→real STEP, not the alignment claim: the toy `d` is parity/class, not a
  certificate/coset; whether a real *alignment* invariant is P-structurally-stable remains a separate
  assumption (per the §6 boundary of `ATLAS_PHASE5_CROSS_SUBSTRATE.md`).
- Forward-only / no inversion; falsifiable gates above are the exact pass/fail; bands not points (report
  seeds + error bars).

## Files (to be produced)
- `scripts/shadow_pooled_synthetic.py` (+ `test_*`) — substrate A.
- `scripts/shadow_pooled_mnist.py` (+ `test_*`) — substrate B.
- `docs/atlas/H3_POOLED_SHADOW_RESULT.md` — the receipt (written after running, against THIS pre-reg).
