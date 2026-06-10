# H9 result — a LOAD-BEARING determine latent on a trajectory shadow (the arrow of time)

> **2026-06-09. POSITIVE (weak notion), banked after a modest adversarial red-team (agent `a597b6f6`).**
> The FIRST genuine load-bearing instance in the H8/H9 arc — and it is on the **determine** side, exactly
> where the H8 capstone theorem (`H8_SHADOW_GEOMETRICITY_THEOREM.md`, R2) said the only escape lived. NOT
> public-eligible. Pre-reg: `H9_LOADBEARING_DETERMINE_PREREG.md` (design-locked before the frozen run).
> Probe: `scripts/trajectory_determine_shadow.py`; frozen test `scripts/test_trajectory_determine_shadow.py`
> (6/6). Read the **dissection**, not the recovery number — see §1.

## The claim (honest, weak/order-k)
A **determine-type** latent — the **signed rotational current / arrow of time** `φ` of a rotational
Ornstein–Uhlenbeck process `dv = −(kI + φJ)v dt + σ dW` — is **load-bearing on the lossy ensemble shadow
against the time-symmetric (equilibrium / order-2 / IAAFT) surrogate class**: recoverable from the directed
worldline (and its jitter-averaged shadow) but from **no** time-symmetric statistic. This **inverts** the H8
no-go: H8 proved no load-bearing *resist* on snapshot shadows (resist ⟹ phase ⟹ geometric); H9 exhibits a
load-bearing *determine* on a trajectory shadow (finite-mean ⟹ concentrates ⟹ survives the lossy shadow).

## 1 — Lead with the DISSECTION (the recovery number is tautological)
The arrow feature `⟨v_t × v_{t+1}⟩` **is** the estimator of `φ` (the current), so its own-R² is "a line fits a
line" — `corr(arrow, φ)=0.987`, own-R²=0.973. **That number carries no information about load-bearingness.**
The entire content is that the **time-symmetric foils are blind** (frozen, n=400 W=4000 seed=20260609):

| feature on the worldline | own-R²(signed φ) | reading |
|---|---|---|
| **symmetric-order-2** (var + symmetric autocov + **symmetrized** cross-cov + histogram) | **0.000** | blind **by time-reversal symmetry** (Risken: the rotational-OU stationary covariance is φ-independent) |
| **IAAFT** matched-spectrum surrogate | **0.000** | load-bearing vs the **standard** time-series null |
| arrow feature (trivial-PASS) | 0.973 | the φ-estimator; tautological |
| shuffled-φ label (trivial-FAIL) | 0.000 | anti-vacuity |

## 2 — The ACTUAL H8 object: load-bearing on the LOSSY ENSEMBLE SHADOW (not just per-trajectory)
The red-team's sharpest point: per-trajectory recovery is *not* the H8 shadow (= mean feature over a latent
jitter population). So the gate was **moved onto the real object** — recover `φ` from the **jitter-averaged**
arrow feature, `shadow = mean over {φ + λξ}`. A determine latent has **finite mean**, so the average
**survives** the jitter (graceful decay), the opposite of a resist (which washes to 0):

| jitter λ | 0.0 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| shadow own-R²(φ) | 0.991 | 0.962 | 0.879 | 0.655* |

(*λ=2.0 is high-variance run-to-run with J=10 jitters — the edge where the determine signal finally degrades;
the gate is the λ≤1.0 region.) Determine concentration `std ∝ 1/√W`: 0.0120 / 0.0067 / 0.0023 at W=250/1000/4000
(log-log slope ≈ −0.57 ≈ −0.5) — the determine signature, the **opposite** of a resist.

## 3 — The apparatus is NOT rigged (it can say "geometric", and grade)
- **all-spectral extreme — fGn Hurst** `H`: real own-R²=1.000, matched-spectrum-surrogate own-R²=1.000 → the
  surrogate **also** recovers it → **GEOMETRIC, not load-bearing** (`H` literally is the spectral slope —
  a trivially-easy endpoint, stated as such).
- **graded — a latent in the (clean) arrow AND a (nuisance-confounded) variance channel**: real=0.735,
  surrogate=0.464 → the surrogate recovers `ψ` **partially** (the variance half survives IAAFT, the arrow half
  does not) → a **discriminating INTERMEDIATE**, strictly below real. So the apparatus outputs 0 / intermediate
  / 1, not a rigged "always load-bearing."

## 4 — Honest scope & boundaries (pre-committed; the red-team's required caveats, all stated)
- **The content is LOW-ORDER.** `φ` is recoverable from a single **raw order-2 two-time cross-covariance**; the
  foil is blind by **time-reversal projection** (symmetrized order-2 = 0.000 while the *raw* order-2 cross-cov
  recovers φ at ~0.97), **not** by deep dynamical inaccessibility. The statement is "an antisymmetric latent is
  invisible to symmetric statistics" — true, robust, and the standard surrogate-data fact, but **one notch above
  a tautology**. Do not oversell it.
- **What is genuinely NEW vs textbook.** *Textbook:* the arrow-of-time / IAAFT detectability itself (the whole
  surrogate-data literature). *New here:* (a) the **framework placement** — a determine latent that is
  load-bearing on the lossy ensemble shadow, the clean **inversion of the H8 resist no-go** (closing the arc's
  R2 escape-route note with a positive); (b) the **symmetry-GUARANTEED** fair foil (φ ⟂ symmetric order-2 by
  Risken, not by empirical luck — robust across a full dt∈[0.02,0.40] / σ / k / W sweep, **no** discretization
  O(dt) leak, the red-team confirmed); (c) the negative-control pair fixing "not rigged" at two points + a grade.
- **Weak vs strong.** This is load-bearing **only vs the time-symmetric class**. A surrogate that also matched
  the antisymmetric 2-time structure (e.g. a VAR(1) fit — the red-team showed it recovers φ at 0.984) is **not**
  in the time-symmetric class and is not claimed against. **Strong notion** (load-bearing vs **all finite-order**
  surrogates) needs a **causal-state / ε-machine** latent with no finite-order sufficient statistic, tested
  against an order-k surrogate ladder — the **follow-on**, not this receipt.

## 5 — Red-team disposition (agent `a597b6f6`, modest single-skeptic per the lightweight protocol)
Verdict: **NOT holed — survives every kill attempt**, conditional on four caveats, all now addressed:
1. "lead with the dissection, not 0.973" → §1 (dissection table first; recovery labelled tautological).
2. "the content is low-order" → §4 (stated explicitly).
3. "the gate tests per-trajectory recovery, not the λ-ensemble shadow" → §2 (gate **moved** onto the shadow
   object; the red-team's own λ-shadow numbers reproduced: 0.99/0.96/0.88 at λ=0/0.5/1.0).
4. "the negative control is the trivial all-spectral extreme" → §3 (added the **graded** discriminating control).
The red-team's robustness sweep (sym-order-2 = 0.000 across the full dt/σ/k/W grid; KILL-GEOMETRIC never fires)
is the single most bankable part and is folded into §4(b). No kill; bankable with the caveats stated above.

## Files
- `scripts/trajectory_determine_shadow.py` — probe (rotational OU; dissection; λ-ensemble shadow; determine
  concentration; fGn + graded negative controls). Reproduce: `python scripts/trajectory_determine_shadow.py [--frozen]`.
- `scripts/test_trajectory_determine_shadow.py` — frozen test (6/6: arrow-load-bearing, trivial-FAIL,
  determine-concentration, ensemble-shadow-determine, Hurst-geometric, graded-discriminating).
- `docs/atlas/H9_LOADBEARING_DETERMINE_PREREG.md` — the design-locked pre-reg.

## Status
**Banked POSITIVE (weak notion).** Frozen-as-portfolio; NOT public-eligible. The first load-bearing instance in
the arc, on the determine side, as the theorem predicted. **Owner-gated next step:** the strong-notion
ε-machine / causal-state follow-on (load-bearing vs all finite-order surrogates).
