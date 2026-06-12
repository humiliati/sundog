# H3-PC Pre-registration — PROBE-CEILING: is the banked clf_d c-suppression destruction or probe-relative concealment?

**STATUS: FROZEN 2026-06-10** (post-review B1–B7 + N1–N7 applied; liveness pilot disclosed in §4; no
edits below this line after this stamp — addenda only).

> **HS4 of slate 2026-06-10** (internal hypothesis slate, gitignored; workflow `wf_f50ff2d1-983`).
> Red-team of the banked H3 v2 claim (`H3_POOLED_SHADOW_RESULT.md`). **This document is BINDING once
> frozen: the battery, gates, seeds, and verdict precedence below may not change after the first run.**
> Adversarially reviewed pre-freeze (agent `acd0d501b922b4828`, 2026-06-10): blocking findings B1–B7
> applied (per-member floors, MEMBER-BLIND semantics, outcome-(c) repair, threshold-anchored
> replication, full seed enumeration, binding reduced-test wording, MI-leg hardening) + non-blocking
> N1–N7. A pre-freeze **liveness pilot** (injected reps ONLY — the battery never touched the real-rep
> readouts) gauged member detectability; its numbers are disclosed in §4.
> Published as a verification receipt (owner decision 2026-06-11; un-promoted; a research artifact,
> not a peer-reviewed publication). A clean null is a SUCCESS; forward-generate only.
>
> Attribution / prior-art lineage (named per slate discipline): amnesic probing (Elazar & Goldberg
> 2018→2021); V-information / probe-quality framing (Xu et al. 2020; Pimentel et al. 2020); concept
> erasure counterexamples (LEACE, Belrose et al. 2023); KSG mutual-information estimator (Kraskov,
> Stögbauer & Grassberger 2004). The concealment-vs-destruction distinction is established field-wide;
> the claim here is the lab-specific one: the first **injection-calibrated probe-ceiling certificate**
> on the H3 v2 representation. Language rule (slate-wide): this is a **probe-access** question in a
> ground-truth substrate — no claims about introspection/confabulation as mental phenomena.

## 1. The question

H3 v2 banked (seed-1234 apparatus, λ=2.0, eval seed 1234+7+2000, N=2000): the clf_d body's pooled rep
reads c-R² = **0.0063** under Ridge(α=1.0) and ≈0.00 under one strong MLP probe (≤0.15 test gate), and
this was called "probe-robust suppression." That certificate covers exactly **two probes at N=2000**.
The open question: is the c-information **destroyed** in the pooled rep, or merely **invisible to that
weak battery**? "Classification destroys c" and "classification hides c from ridge+MLP" are different
laws; only the weak one is banked.

## 2. Fixed substrate (unchanged, byte-reproducible)

- Apparatus: `scripts/shadow_pooled_synthetic_v2.py` constants verbatim (SEED=1234, K=64, M=64, D=8,
  H=32, F=72, TRAIN_LAM=1.0, W_RFF∈[3.0,6.5], SIGMA_D=1.5, OBS_NOISE=0.05, C∈[1,2]).
- Body: `train_body("clf_d", …)` retrained deterministically (torch single-thread, OBJ_SEED=11) on the
  banked train draw `gen(8000, 1.0, SEED+1)`. Positive-control body: `train_body("reg_c", …)`
  (OBJ_SEED=22), same draw.
- Eval λ = **2.0** (the banked anchor; raw fully washed).
- **New disjoint draws** (the banked eval seed 3241 is not reused):
  - PROBE POOL: `gen(20000, 2.0, SEED+50001)` — all probe training, hyperparameter selection, CV,
    calibration happen here.
  - FROZEN TEST SPLIT: `gen(10000, 2.0, SEED+60001)` — touched ONCE, after all selection is frozen;
    every replication number comes from here.
- Reps: `phi_pool(body, units)` → (n, 32). Raw-mean continuity check on the pool: ridge c-R² ≤ 0.05
  (C0 still holds on the new draw; if it fails the run is VOID — apparatus, not result).

## 3. The frozen probe battery (the only probes; no additions after freeze)

All on standardized reps (StandardScaler refit per dataset: real pool, each injected pool); selection
= 5-fold KFold(shuffle, rs=0) CV R² on the pool; replication = fit on the full pool, R² scored once on
the frozen test split. **R² is UNCLIPPED everywhere** (the substrate's max(0,·) convention is NOT
imported; negative values recorded as-is — a negative reads as "no signal" under every gate).
Hyperparameter selection rule: argmax pool-CV per family; ties broken by first-in-grid order (grids
enumerated left-to-right, row-major, in the table below).

| ID | Estimator | Hyperparameter grid (CV-selected on the pool) |
|----|-----------|----------------------------------------------|
| P1 | Ridge(α=1.0) | none (the banked weak baseline, for continuity) |
| P2 | MLPRegressor(hidden=(128,64), max_iter=600, rs=0) | none (the banked strong probe) |
| P3 | kNN regression | k ∈ {5, 10, 20, 50, 100} |
| P4 | Nyström(m=2000, RBF) + Ridge | γ ∈ {0.001, 0.01, 0.1, 1.0} × α ∈ {0.1, 1.0, 10.0} |
| P5 | HistGradientBoostingRegressor(rs=0) | learning_rate ∈ {0.05, 0.1} × max_iter ∈ {200, 500} |
| MI | KSG estimator (k=5, Kraskov 2004), chunked cKDTree | PCA-k ∈ {2, 4, 8, 16, 32} per §4 protocol |

Probe-data scaling leg (REPORTED, non-gating): learning curve of the best battery member + P1 at
n ∈ {2000, 5000, 10000, 20000} from the pool (subsamples = seeded draws without replacement,
`default_rng(SEED+90001=91235)`) — distinguishes sample-starvation from absence.

## 4. Injection calibration (defines the certified detection floor, PER MEMBER)

Synthetic c-injection into the clf_d pool reps: `z′ = z + α · g(c) · v`, with `g(c)` = c standardized
by POOL mean/std, `v` = a fixed random unit vector in R³² (`default_rng(SEED+555=1789)`). The
injection is a **fixed random LINEAR direction**; every floor below is a linear-direction floor (§7).
Injected datasets get their own StandardScaler refit (per §3); the test split is never injected.

- α set by deterministic bisection (bracket α∈[0, 3], 28 iterations) so P1 (ridge, pool CV) reads
  R² ≈ {0.10, 0.20}, each within ±0.01. (This DEFINES "ridge-equivalent strength"; ridge detecting its
  own calibration is expected, not evidence.) **Calibration gate** (apparatus): bisection converges —
  |ridge pool-CV − target| ≤ 0.01 at both levels within the bracket/iteration budget. Failure ⇒ VOID.
- **Per-member liveness + floors** (replaces run-level VOID — one blind probe cannot deadlock the
  run): each member P2–P5 is evaluated on BOTH injection levels; a member is **live at level L** iff
  ANY config in its grid reads pool-CV R² ≥ 0.05 on the L-injection. A member live at neither level is
  named **MEMBER-BLIND** (to linear injections): it is excluded from outcome (b)'s certificate list
  and carries NO floor; its silence on the real reps is uninformative and is reported as such. Each
  live member's **certified floor = the weakest level at which it is live** (0.10 or 0.20). P1's floor
  is 0.10 by construction.

  **Pilot disclosure** (pre-freeze, INJECTED reps only — the real-rep battery readouts were never
  computed; n=8000 prefix-style draw of the pool seed; pilot grid = a SUBSET of the frozen grids;
  α(0.10)=0.0080, α(0.20)=0.0126):

  | member (pilot config) | CV @ 0.20 inj | CV @ 0.10 inj | expectation (NOT a gate) |
  |---|---|---|---|
  | kNN k∈{5,20,50,100} | best +0.026 | best −0.001 | likely MEMBER-BLIND (linear spike invisible to 32-d Euclidean neighborhoods — reviewer B2's geometry confirmed) |
  | HistGBT (0.1, 200) | **+0.152** | +0.041 | live at 0.20; 0.10 undecided at full n |
  | MLP (128,64) | −0.051 | −0.057 | likely MEMBER-BLIND — **the banked "strong probe" cannot see a 0.20-ridge-equivalent linear signal at this n**; its banked silence carried less weight than the result doc implied (to be scoped in the result doc regardless of verdict) |
  | Nyström γ∈{0.01,0.1}, α=1.0 | +0.021 / +0.003 | +0.009 / +0.001 | undecided — the frozen grid is wider (γ∈{0.001…1.0} × α∈{0.1,1,10}) and full-n may differ |

  The full run at n=20,000 with the full grids DECIDES liveness; the pilot binds nothing.
- **MI leg** (pool-only; never touches the test split): KSG (k=5) on a **seeded subsample of 5000
  pool points** (`default_rng(SEED+85001=86235)`), inputs = standardized PCA scores + standardized c.
  PCA fit on the (subsampled) pool reps. Injection-liveness per PCA-k ∈ {2, 4, 8, 16, 32}: MI on the
  **0.10** injection > **max** of 99 shuffled-null MIs (shuffle seeds = `default_rng(SEED+80001=81235)
  .integers(0, 2**31, 99)`; rank rule, no percentile interpolation). If NO PCA-k passes, the MI leg is
  **VOID** (named: KSG blindness must not masquerade as information absence) and outcome (b)'s
  certificate carries the explicit qualifier **"probe-battery-only"**; the result JSON records
  `mi_leg: live|void`. If live, outcome (b)'s MI clause is evaluated at **EVERY PCA-k that passed
  injection-liveness, with k=32 mandatorily among them** (k=32 failing liveness ⇒ MI leg VOID).
- Positive control: the reg_c body's pool reps must read CV c-R² ≥ 0.45 under at least one battery
  member (banked 0.51 ridge at the old seed; tolerance for the new draw). Failure ⇒ VOID.

## 5. Pre-registered outcomes (exhaustive; precedence top-down)

| # | Outcome | Condition (all on the clf_d reps at λ=2.0) | Reading |
|---|---------|--------------------------------------------|---------|
| V | **VOID** | C0 continuity fails, calibration (bisection convergence) fails, or the positive control fails. (Member liveness is per-member, NOT a VOID condition.) | fix apparatus, re-run; not a result |
| a | **CONCEALMENT COUNTEREXAMPLE** | some battery member: pool-CV R² ≥ **0.30** AND frozen-split R² ≥ **0.24** (= 0.8 × the pre-registered threshold — anchored to the THRESHOLD, not the winner's-curse-inflated CV value) | the banked "probe-robust suppression" clause BREAKS — H3 result doc gets a correction; the suppression was battery-relative |
| b | **CERTIFIED PROBE-CEILING** (the clean-null SUCCESS = the kill branch) | EVERY non-MEMBER-BLIND member: pool-CV R² ≤ 0.05 AND frozen-split R² ≤ 0.05; AND (MI leg live ⇒ KSG MI ≤ max shuffled-null at EVERY liveness-passing PCA-k, incl. k=32) | suppression certified informational **down to each live member's per-member floor (§4: 0.10 for P1 + the MI leg when live; 0.10 or 0.20 per nonlinear member as measured), relative to the declared battery and to a fixed random linear direction** — no finite battery certifies more, stated as such. If `mi_leg: void`, the certificate is explicitly **probe-battery-only**. MEMBER-BLIND members are listed as contributing nothing |
| c | **BOUNDED-PARTIAL CONCEALMENT** | anything else | named outcome. Counted set = members with frozen-split R² ≥ 0.05; **ceiling = max frozen-split R² over the counted set if non-empty, else 0 with sub-outcome SPLIT-ONLY-FLUCTUATION** (a split reading ≥0.05 whose member pool-CV was ≤0.05 is noise on a once-touched split unless ≥0.05 on BOTH). Any member with pool-CV ≥ 0.30 but split < 0.24 is named **UNREPLICATED-POSITIVE** and MANDATES a re-run addendum (fresh split seed, pre-registered before running) — it may not be silently banded |

Multiplicity control: thresholds frozen here; the split is touched once; (a) requires both the CV
threshold and the threshold-anchored split bar. No cherry-picking the split; no anchoring to
selection-inflated CV values (reviewer finding B4).

## 6. Determinism, files, commands

- **Thread pinning (binding):** single-thread torch AND `OMP_NUM_THREADS=1` + `MKL_NUM_THREADS=1` set
  by the script itself before numeric imports (the substrate's v1 lesson: unpinned BLAS broke
  determinism). MLPRegressor(rs=0), HistGBT(rs=0), cKDTree, kNN are deterministic in this environment.
- **Complete seed ledger** (all disjoint from the banked set {1235, 1241, 1341, 1441, 1541, 1741,
  1991, 2241, 2741, 3241, 3741, 4241, 1245, 1256, 1267, 2233}): pool draw **51235**; test draw
  **61235**; injection direction **1789**; Nyström components **71235**; KSG shuffle seeds from
  **81235**; MI subsample **86235**; learning-curve subsamples **91235**.
- Script: `scripts/h3_probe_ceiling.py` (writes `results/atlas/h3/probe_ceiling_result.json`, incl.
  `mi_leg`, per-member floors, unclipped R² everywhere).
- Frozen test: `scripts/test_h3_probe_ceiling.py` — reduced-size but real path: **the reduced pool /
  test are the FIRST 4000 / 2000 rows of the FULL-SIZE draws** `gen(20000, 2.0, 51235)` /
  `gen(10000, 2.0, 61235)` (a strict subset — NOT a fresh `gen(4000, …)` call, which would be a
  different realization). It pins: C0 continuity, bisection convergence, per-member liveness verdicts,
  and **byte-identical headline numbers + verdict letter across reruns AT THE REDUCED SIZE; equality
  with the full run's verdict letter is NOT asserted or required** (pinning it would force apparatus
  tuning — the regen-drift failure mode). The FULL run's numbers are the result; the test pins the
  apparatus.
- Exact unchanged commands: `python scripts/h3_probe_ceiling.py` · `python scripts/test_h3_probe_ceiling.py`.
- Existing suite must stay green: `python scripts/test_shadow_pooled_synthetic_v2.py`.

## 7. Honest boundaries (pre-stated)

- The certificate is **battery-relative, floor-relative, and direction-relative** by construction:
  outcome (b) claims "no information recoverable at ≥ each live member's measured floor
  (0.10-ridge-equivalent for P1/MI; 0.10 or 0.20 per nonlinear member) by these probe families + KSG,
  against a fixed random LINEAR injection direction," never "no information exists." (Fan 1991-style
  deconvolution lower bounds are out of scope.)
- Synthetic substrate; the linear injection defines the floor in a linear direction — a nonlinear
  member's sensitivity to a linear spike does not bound its sensitivity to a nonlinearly embedded
  residue; such a residue weaker than every member's sensitivity remains logically possible (that is
  what "floor" means). MEMBER-BLIND members (blind to linear injections) are retained for nonlinear
  coverage but contribute no floor and no certificate weight.
- Second leg (named, NOT run under this prereg): the same frozen battery on the banked MNIST-rotation
  rep (`shadow_pooled_mnist.py` artifacts) — requires its own prereg addendum before running.
- Verdict (a) does NOT overturn H3's objective-gap finding (reg_c ≫ clf_d under ridge stands); it
  corrects only the "probe-robust" clause's scope.
