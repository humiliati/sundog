# H3-SP Pre-registration — STALE-STRATEGY PROJECTION: does a stale reporter's output land on the determinable subspace?

**STATUS: FROZEN 2026-06-11** (post-review rewrite B1–B6 + pilot disclosed in §6; no edits below this
line after this stamp — addenda only).

> **HS2 of slate 2026-06-10** (internal hypothesis slate, gitignored). Third leg of the
> pooled-shadow triptych (HS4 → verdict (c); HS3/H10 → clean kill). **BINDING once frozen.**
> Published as a verification receipt (owner decision 2026-06-11; un-promoted; not peer-reviewed).
> A clean null is a SUCCESS; forward-generate only. **Language rule:** this measures
> the projection structure of a STALE DEMODULATOR's output in a ground-truth synthetic substrate — no
> claims about confabulation/introspection as mental phenomena (hence `stale_projection` filenames);
> the probe-vs-introspection gap is the standing limitation. Adversarially reviewed pre-freeze (agent
> `a7ac50f0bdedd717a`): blocking B1–B6 applied (PP run-verdict added; gen rng-order pinned; SG
> re-banded at 0.3 with projection computed for ALL structured cells; λ=0 control de-fanged to its
> sound clause; C0 wash gate restored on the modified generator; {g,mag} grouped + PS-d/PS-g split +
> g-counterfactual added). Attribution: H3 v2; Nisbett & Wilson 1977; Turpin et al. 2023; HS4 (probe
> conventions; the MEMBER-BLIND finding).

## 1. The question

Train a reporter to estimate the continuous latent `c` where it is accessible (per-sample
λ ~ U[0, 0.5]; raw averaging not yet washed — banked raw_c 0.94 at λ=0.5). FREEZE it. Deploy at
λ = 2.0 (banked raw_c = 0.0). The reporter still outputs numbers. **What structure do they have?**
The open question, post-review, rides on the **d-partial**: the discrete channel is in-distribution at
deploy and the c-objective gave the reporter no incentive to couple to it. A gain-channel lock is
pre-named as partially foreordained (off-manifold scale-sensitivity) and carries a weaker reading.

## 2. Substrate (additive; `shadow_pooled_synthetic_v2.py` is import-only, NOT modified)

`gen_hs2(n, lam_spec, seed)` reuses the v2 constants verbatim (SEED=1234, K=64, M=64, D=8, F=72,
W_RFF, PSI, A_DISC, SIGMA_D=1.5, OBS_NOISE=0.05, c ~ U[1,2], d ∈ {±1}) and adds the determinable gain
nuisance `g ~ U[0.5, 1.5]`.

**Binding rng order (single `default_rng(seed)` stream, v2 order with insertions):** `c` (n), `d` (n),
`g` (n), `lam` (length-n block drawn ONLY when `lam_spec == "train"`; a scalar `lam_spec` consumes NO
draw), `xi` (n×K), `eta` (n×K×D), then `units = g[:,None,None] * concat(fringe, disc)`, THEN
`units += obs_noise` (n×K×F draw). The gain multiplies the signal, not the observation noise.

Proxy families at deploy (the declared determinable set): **family d** = the true `d`; **family
g/mag** = the true `g` AND `mag` = ‖mean_i u_i‖₂, grouped as ONE family (at λ=2, mag ≈ g·const —
corr ≈ 1 by construction; the family takes the max of its two partials but counts once).

**Apparatus gates on the deploy draw (any failure ⇒ VOID):**
- **C0 wash gate** (the generator changed, so the banked wash is re-gated, HS4 convention): ridge
  c-R²(raw `mean_i u_i`) ≤ 0.05.
- **MI gate** (proxy-locking must not be rational inference E[c|proxies]): R²(g ~ c) ≤ 0.01 and
  R²(mag ~ c) ≤ 0.01 under BOTH Ridge(α=1.0) AND MLPRegressor((128,64), max_iter=600, rs=0);
  balanced-acc(d ~ c, logistic) ≤ 0.52. CV convention everywhere: KFold(5, shuffle=True,
  random_state=0), unclipped R²; 1-D inputs standardized before the MLP probe.

## 3. Reporters (pinned pipeline)

Two reporters per seed, identical except the head: **R-lin** = v2.Phi + Linear(32,1); **R-mlp** =
v2.Phi + (Linear(32,64) → ReLU → Linear(64,1)). Init rule (binding): `torch.manual_seed` (and
`np.random.seed`) set to the init seed immediately before constructing EACH reporter; Phi is
constructed first and consumes a fixed draw count, so **Phi init is byte-identical across the two head
architectures**; the streams diverge at head construction; no cross-head training identity is claimed.

Training (pinned, the banked v2 recipe): MSE on true `c`; data `gen_hs2(8000, "train", 101235)`;
Adam(lr=1e-3), batch 256, 120 epochs; torch single-thread. Init seeds SEED+130001+k, k ∈ {0..4} =
**131235–131239**. **Train-fit gate (apparatus):** train R² ≥ 0.7 for every (head, seed) ⇒ else VOID.
After training: FROZEN (eval mode; no updates anywhere downstream).

## 4. Deploy readouts (λ=2.0 draw; report = the frozen reporter's scalar)

Per (head, seed) cell:

1. **Coupling band:** R²(c ~ report) under Ridge(α=1.0), MLPRegressor((128,64), rs=0) — defensible on
   a 1-D input despite HS4's 32-D MEMBER-BLIND finding (univariate fit is a different regime; stated
   citing HS4) — and Nyström(γ=1.0, m=2000, rs=71235)+Ridge(α=0.1) (HS4's live member, continuity).
   Band by the MAX of the three: **decoupled** ≤ 0.05 < **weakly-coupled** ≤ 0.3 < **SG** (> 0.3).
   Worst-case leak budget, pre-stated: coupling ≤ 0.3 × proxy~c ≤ 0.01 ⇒ induced proxy partial-R²
   ≤ ~0.003 ≪ the 0.1 threshold, so weakly-coupled cells partition normally.
2. **Structure floor:** Var(report)/Var(c) ≥ 0.25 ⇒ structured; else **CO** (collapsed). Rationale:
   0.25 = a quarter of the latent's variance rendered as output spread; the reporter is deterministic,
   so propagated input noise counts as structure — K2's reading is softened accordingly. The λ=0
   variance ratio is banked per cell as the reference point.
3. **Projection readout (computed for ALL structured cells, any coupling band):** partial-R² of report
   vs each proxy, controlling for c — residualize report and proxy on c (linear; point-biserial form
   for d), R² between residuals. Per-proxy partials ALL banked in the JSON; family scores = d-partial
   and max(g-partial, mag-partial).
4. **g-counterfactual (banked, non-gating; separates lock-on-g from generic scale-sensitivity):**
   re-evaluate the frozen reporter on `units/g` (same draw, gain divided out); bank
   R²(report ~ report_cf) and the increment R²(report ~ [report_cf, g]) − R²(report ~ report_cf) —
   the scale-channel share of the report's variance.

**Within-design control (λ=0 control draw):** R²(c ~ report) ≥ 0.9 (Ridge convention) for every
(head, seed) ⇒ else VOID — the sound clause only. λ=0 proxy partials are NOT a gate (at λ=0 mag is
genuinely c-dependent by physics and the residualization is linear); they are banked as per-cell
**baselines**, and PS verdicts additionally require deploy partial ≥ 5× the same cell's λ=0 baseline.

## 5. Pre-registered outcomes (cell verdicts → run verdict; precedence top-down)

Cell verdict (precedence: SG > CO > then by family partials, PS-d > PS-g > PP > IS):
- **SG** coupling > 0.3 · **CO** structure floor fails ·
- **PS-d** d-partial ≥ 0.5 AND ≥ 5× its λ=0 baseline · **PS-g** g/mag-family ≥ 0.5 AND ≥ 5× baseline ·
- **PP** max family partial ∈ [0.1, 0.5) · **IS** all family partials < 0.1.

Run verdict:
| # | Outcome | Condition | Reading (all "in this substrate") |
|---|---------|-----------|-----------------------------------|
| V | **VOID** | C0, MI, train-fit, or the λ=0 R²≥0.9 control fails | apparatus; fix and re-run |
| A-d | **SIGNATURE (determinable-subspace)** | ≥ 4/5 seeds PS-d, BOTH heads | the stale report locks onto the in-distribution discrete survivor — the strong finding: a proxy-correlation audit could flag stale reports; the determine/resist axis structures report behavior |
| A-g | **SCALE-CHANNEL PROJECTION** | ≥ 4/5 seeds PS-g, both heads | a gain-family lock — pre-named as partially foreordained (multiplicative gain off-manifold); weaker reading; g-counterfactual share banked alongside |
| P | **PARTIAL-PROJECTION** | ≥ 4/5 seeds PP, both heads | graded determinable-subspace lock; audits get a weak flag; magnitudes banked |
| K1 | **GRACEFUL-COLLAPSE** (clean null = SUCCESS) | ≥ 4/5 seeds CO, both heads | stale strategies degrade toward mean-reporting; report-audits cannot flag what collapses to the prior |
| K2 | **ISOTROPIC** (clean null = SUCCESS) | ≥ 4/5 seeds IS, both heads | variance above floor but latent-decoupled (consistent with propagated input noise); no determinable-subspace signature for audits |
| G | **STALE-GENERALIZES** | ≥ 4/5 seeds SG, both heads | at this deploy λ and training mixture, the accessible-regime demodulator transfers (banked coupling values); the deploy-shift cannot manufacture decoupling for this strategy class — projection readouts still banked (computed for all structured cells) |
| D | **ARCHITECTURE-DEPENDENT** | the heads' modal cell verdicts differ (each ≥ 4/5 internally) | the projection target is head-dependent — itself the pre-registered finding |
| M | **MIXED** | any combination not matching the rows above | banked with the full per-cell table; no headline claim |

No replication split is needed: n_deploy = 10k gives R² SE ≈ 0.006 at the 0.1/0.5 thresholds, and no
selection-over-grid occurs (all pipelines fixed a priori).

## 6. Determinism, seeds, files, commands

- Thread pinning as HS4 (OMP/MKL=1 in-script before numeric imports; torch single-thread).
- **Seed ledger** — fully enumerated exclusions: v2 banked {module 1234; train 1235; objective 1245,
  1256, 1267; rand-phi 2233; eval grid 1241, 1341, 1441, 1541, 1741, 1991, 2241, 2741, 3241, 3741,
  4241}; HS4 {51235, 61235, 1789, 71235, 81235, 86235, 91235}; H10's 5150-family (max derived ≈
  15,373). HS2: train **101235**, deploy **111235**, control **121235**, inits **131235–131239**
  (131235 = the frozen test's single seed); n_train=8000, n_deploy=10000, n_control=4000. All
  verified non-colliding.
- Script: `scripts/shadow_stale_projection.py` → `results/atlas/h3/stale_projection_result.json`
  (per-cell table with all per-proxy partials, coupling bands, variance ratios, λ=0 baselines,
  g-counterfactual shares, gates, run verdict; unclipped throughout).
- Frozen test: `scripts/test_shadow_stale_projection.py` — reduced real path (train 2000 / deploy
  2500 / control 1000 = FIRST rows of the full-size draws; seed k=0 both heads), pinning: C0 + MI +
  train-fit + control gates, byte-identical readouts across an in-process rerun. Full-run verdict NOT
  asserted at reduced size.
- **Disclosed pre-freeze apparatus pilot** (HS4 precedent; gates only, NO deploy projection readouts;
  1 seed 131235, reduced n = 2000/2500/1000, fresh reduced-n draws): train-fit lin **0.984** / mlp
  **0.984** (gate 0.7); λ=0 control lin **0.996** / mlp **0.996** (gate 0.9); C0 wash on the modified
  generator **−0.019** (gate ≤ 0.05); MI gate g **−0.003** / mag **−0.003** (≤ 0.01), d bal-acc
  **0.489** (≤ 0.52). All gates passable with margin; no tuning pressure. Expectations are NOT gates;
  the full run decides.
- Exact unchanged commands: `python scripts/shadow_stale_projection.py` ·
  `python scripts/test_shadow_stale_projection.py`. Existing suite stays green
  (`python scripts/test_shadow_pooled_synthetic_v2.py`).

## 7. Honest boundaries (pre-stated)

- Synthetic substrate; the "report" is a regression head, not a language-model self-report. The
  declared proxy set is what an auditor would have; an undeclared survivor reads as PP/IS (accepted).
- A-d does not claim audits work on real systems; it establishes the mechanism in the one substrate
  with exact ground truth. A-g is explicitly discounted as partially construct-driven. K1/K2/G bound
  the design honestly. HS3/H10's lesson binds: existence of structure ≠ trainability of detection.
- The within-design control's proxy clause was REMOVED as a gate post-review (unsound: λ=0 mag–c
  physics + linear residualization + 30 conjunctive chances to void); its sound replacement is the
  per-cell baseline + the 5× PS requirement.
