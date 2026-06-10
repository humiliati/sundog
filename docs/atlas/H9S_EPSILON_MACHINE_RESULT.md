# H9-strong result — a determine latent load-bearing vs ALL finite-order surrogates (a causal-state / ε-machine)

> **2026-06-09. POSITIVE (strong notion), banked after a modest adversarial red-team (agent `a717868d`).**
> The decisive strengthening of the H9 weak positive (`H9_LOADBEARING_DETERMINE_RESULT.md`, load-bearing only
> vs the time-symmetric class). Here a determine latent is load-bearing vs the **entire finite-order surrogate
> class** — it has **no finite-order sufficient statistic**. NOT public-eligible. Pre-reg
> `H9S_EPSILON_MACHINE_PREREG.md` (design-locked before the frozen run). Probe `scripts/epsilon_machine_shadow.py`;
> frozen test `scripts/test_epsilon_machine_shadow.py` (4/4). Read the **dissection**, not the recovery number.

## The claim (strong notion)
A **determine-type** latent `φ` riding a **causal state with no finite-order sufficient statistic** — the
running **parity** `P_t = b_1⊕…⊕b_t` of a fair-coin driver (a 2-state but **infinite-Markov-order**,
strictly-sofic ε-machine) — is **load-bearing against the order-k surrogate class for every k**: recoverable
from the real sequence via the causal state, but from **no** order-k matched-statistics surrogate. This realizes
the strong notion the H8 theorem's R2 escape-route pointed at (a determine latent that no finite-order static
template reproduces), going beyond H9-weak's time-symmetric-only foil.

Substrate: observable `(b_t, c_t)`, `b_t~Bernoulli(½)`, `c_t = P_t` w.p. `(1+φ)/2` else `1−P_t`; latent
`φ∈[0,1]` = parity-readout fidelity (`corr(c_t,P_t)=φ`).

## 1 — Lead with the DISSECTION (the recovery number is tautological)
The feature includes `corr(c_t, full running parity)`, which **is** the φ estimator, so real own-R²=0.999 is
"a line fits a line." **All content is in the surrogate ladder being blind** (frozen, n=250 L=6000 seed=20260609):

| order-k Markov-resample surrogate | own-R²(φ) |
|---|---|
| k=1 | 0.000 |
| k=2 | 0.000 |
| k=3 | 0.000 |
| k=4 | 0.000 |
| real (causal state) — trivial-PASS | 0.999 |
| shuffled-φ — trivial-FAIL | 0.000 |

φ is invisible at **every** finite order. (Red-team: the surrogate's cross-val R² is genuinely **negative**
pre-`max(0,·)`-clamp — honest failure, not a small positive floored away; no feature entry leaks φ on any
surrogate.)

## 2 — The order-k ladder is a CALIBRATED order-meter (the negative control)
A finite **order-d** latent (`c` reads the parity of the last `d` consecutive bits) spans a d-block, so the
order-k surrogate preserves it **iff k ≥ d−1** — recovery rises and **crosses at k=d−1, the crossing tracking
the latent's order**:

| latent | k=1 | k=2 | k=3 | k=4 | crossing |
|---|---|---|---|---|---|
| order-3 parity | 0.00 | **1.00** | 1.00 | 1.00 | k=2 ✓ |
| order-4 parity | 0.00 | 0.00 | **1.00** | 1.00 | k=3 ✓ |
| **full parity (positive)** | 0.00 | 0.00 | 0.00 | 0.00 | **never** |

So the ladder is **not broken** — it detects finite order exactly where it sits; the full-parity latent is
reported as beyond *every* rung. Determine concentration `std∝1/√L` (slope −0.50, red-team) — a determine
latent (the H9 lineage), the opposite of a resist.

## 3 — Red-team hardening (agent `a717868d`, modest single-skeptic)
Verdict: **REAL and bankable — survives every kill attempt.** Added robustness beyond the frozen gates:
- **Surrogate is a faithful order-k match, NOT a strawman:** real-vs-surrogate (k+1)-block total-variation
  distance is within the two-independent-draws **estimation floor** for every k (it destroys only blocks > k+1,
  exactly correct). The negative-control crossing is a real mechanism (`corr(c, order-d parity)` preserved iff
  k≥d−1, verified by direct correlation).
- **Ladder pressure-tested to k=8** (262144 contexts, 43× the L=6000 data): still 0.000 — a local Markov over
  `(b,c)` **cannot** hold the global parity constraint (resampling `b` regenerates a fresh running parity
  decoupled from `c`). Even an **augmented surrogate handed the hidden causal state as an explicit channel**
  still recovers 0.000. The k=4→0.060 leak seen at the smaller calibrate L=5000 is a **data-limit artifact**;
  frozen L=6000 clears it (0.000) and the k=8 sweep shows the margin is far larger than feared.
- **Analytic claim confirmed broadly** (L=200000): a battery of finite-order functions `g` — single bits at many
  lags, pair parities, block parities up to length 12, AND/OR of recent bits, the c-channel's own history —
  **all** have `corr(c_t, g)≈0` (≤0.003 vs a 0.0022 floor); only the full running parity recovers φ.

## 4 — Honest scope & boundaries (pre-committed; the red-team's required caveats, all stated)
- **All-k is ANALYTIC; the empirics test k=1..4.** The "load-bearing vs every finite order" rests on the
  **complementary-parity independence** argument (for any finite index set S, `P_t = (⊕_S b)⊕(⊕_{not S} b)` and
  the complementary parity is an independent fair coin that randomizes the relation — holds for all k). The
  ladder `k≤4` (+ the k=8 robustness sweep) is the empirical confirmation; the negative control shows the ladder
  CAN detect finite order. Do not let the empirics imply all-k on their own.
- **What is TEXTBOOK vs NEW.** *Textbook (do not oversell):* the fact that parity is a strictly-sofic,
  infinite-Markov-order process with **no finite-order sufficient statistic** is canonical computational
  mechanics (Crutchfield & Young ε-machines; parity is *the* standard example). *New here, narrowly:* (a) the
  **framework placement** — exhibiting the **strong load-bearing notion** (vs all finite order) that H9-weak and
  the H8 R2 escape-route only pointed at; (b) the **order-k surrogate ladder as a calibrated order-meter** (the
  negative controls crossing at k=d−1, the crossing tracking the latent's order).
- **The "shadow" framing is THINNER than H9 proper.** There is **no explicit lossy-jitter-ensemble-shadow
  object** here (as in `H9_..._RESULT.md` §2); the lossiness is the readout noise `(1−φ)/2`, and load-bearingness
  is **per-sequence** recovery, not a jitter-averaged shadow. The determine *signature* (std∝1/√L) holds, but
  state this so it is not conflated with H9's §2 ensemble-shadow gate.
- **Designed/synthetic substrate** (a constructed ε-machine), like every Shadow-lane probe — it tests whether
  the framework's strong notion is *instantiable and detectable*, not a claim about natural systems.

## Files
- `scripts/epsilon_machine_shadow.py` — probe (parity ε-machine; order-k Markov-resample surrogate ladder;
  parity-correlation feature; order-d negative controls; determine concentration). Reproduce:
  `python scripts/epsilon_machine_shadow.py [--frozen]`.
- `scripts/test_epsilon_machine_shadow.py` — frozen test (4/4: full-parity load-bearing vs all finite order,
  trivial-FAIL, ladder-detects-finite-order-control, determine-concentration).
- `docs/atlas/H9S_EPSILON_MACHINE_PREREG.md` — the design-locked pre-reg.

## Status
**Banked POSITIVE (strong notion).** Frozen-as-portfolio; NOT public-eligible. Together with the H9 weak result
and the H8 capstone no-go, the arc is closed on both sides: **no load-bearing charFun-RESIST on snapshot/window
shadows (H8), but a load-bearing DETERMINE on trajectory/causal-state shadows — vs the time-symmetric class
(H9-weak) and, for a no-finite-order-sufficient-statistic latent, vs ALL finite order (H9-strong).**
