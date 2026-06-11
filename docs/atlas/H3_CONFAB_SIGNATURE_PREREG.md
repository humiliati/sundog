# H3-CS Pre-registration — STALE-STRATEGY PROJECTION: does a decoupled report land on the determinable subspace?

**STATUS: DRAFT — not yet frozen** (adversarial review pending; this stamp flips to FROZEN after).

> **HS2 of slate 2026-06-10** (`internal/slates/HYP_SLATE_2026-06-10.md`; workflow `wf_f50ff2d1-983`).
> Third leg of the pooled-shadow triptych (HS4 = probe-ceiling, landed verdict (c); HS3/H10 = adversarial
> hide-d, landed clean kill). **BINDING once frozen.** NOT public-eligible. A clean null is a SUCCESS;
> forward-generate only. **Language rule (slate-wide):** this measures the projection structure of a
> STALE DEMODULATOR's output in a ground-truth synthetic substrate — no claims about
> confabulation/introspection as mental phenomena; the probe-vs-introspection gap is a limitation
> stated up front. Attribution: H3 v2 substrate; Nisbett & Wilson 1977 (behavioral analog, no ground
> truth); Turpin et al. 2023 (unfaithful explanations); HS4's H3-PC (the probe conventions + the
> correction that scoped "probe-robust").

## 1. The question

Train a reporter to estimate the continuous latent `c` in the regime where `c` is accessible
(λ ∈ [0, 0.5] — raw averaging has NOT washed; the strategy is cheap to learn). FREEZE it. Deploy it at
λ = 2.0, where the fringe is washed and the report is (to be certified, not assumed) decoupled from
true `c`. The reporter still outputs numbers. **What is the STRUCTURE of that output?** Two live
hypotheses, neither decidable a priori:

- **Projection signature:** the stale output locks onto the *determinable* survivors of pooling —
  the discrete `d`, the gain `g`, the pooled magnitude — i.e., the report projects onto the
  determinable subspace (the lab's determine/resist axis showing up inside a report head). If so, a
  proxy-correlation audit can flag stale reports about inaccessible state without trusting the system.
- **No signature:** the output collapses toward mean-reporting (graceful "I don't know" rendered as a
  number) or stays structured but isotropic. Either way that clean null BOUNDS what report-audits can
  ever catch — equally bankable.

## 2. Substrate (additive extension; `shadow_pooled_synthetic_v2.py` is import-only, NOT modified)

`gen_hs2(n, lam_spec, seed)` reuses the v2 constants verbatim (SEED=1234, K=64, M=64, D=8, F=72,
W_RFF, PSI, A_DISC, SIGMA_D=1.5, OBS_NOISE=0.05, c ~ U[1,2], d ∈ {±1}) and adds ONE determinable
nuisance:

- **gain `g ~ U[0.5, 1.5]`**, per-sample, multiplying the whole unit vector (`units *= g`). The pooled
  mean scales by `g` ⇒ `g` is determinable through pooling; the fringe mean at λ=2 is
  Debye–Waller-washed (×g ≈ 0 still) ⇒ the c-wash is untouched.
- `c, d, g` are **independent draws** (the generator's independence is the design's backbone).
- `lam_spec`: a scalar λ (deploy/control draws) OR the string `"train"` = per-sample λ_i ~ U[0, 0.5]
  (seeded, same rng stream) — the accessible-regime training mixture.

Proxies measured at deploy (the determinable set): `d` (true), `g` (true), `mag` = ‖mean_i u_i‖₂
(computed from data, no ground truth needed).

**MI gate (apparatus; on the deploy draw):** ridge R²(g ~ c) ≤ 0.01, R²(mag ~ c) ≤ 0.01, balanced-acc
(d ~ c, logistic) ≤ 0.52 — so any proxy-locking CANNOT be rational inference E[c | proxies]. Failure ⇒
VOID (generator broken).

## 3. Reporters (the only trained objects; pinned pipeline)

Two reporters per seed, identical except the head (the slate's dual-head rule):

- **R-lin:** v2.Phi (F→128→128→H=32, mean-pool) + Linear(32, 1).
- **R-mlp:** v2.Phi + MLP head (32→64→1, ReLU).

Training (pinned): objective = MSE on true `c`; data = `gen_hs2(8000, "train", TRAIN_SEED)`; optimizer
Adam(lr=1e-3), batch 256, 120 epochs (the banked v2 recipe); torch single-thread; init seeds
SEED+130001+k for k ∈ {0..4} (5 seeds; both heads share the seed's init stream order: Phi then head).
**Train-fit gate (apparatus):** train R² ≥ 0.7 for every (head, seed) — the reporter genuinely learned
to estimate c in the accessible regime. Failure ⇒ VOID.

After training: **FROZEN** (eval mode, no further updates anywhere).

## 4. Deploy readouts (all on the λ=2.0 deploy draw; report = frozen reporter's scalar output)

For each (head, seed):

1. **Coupling certificate:** R²(c ~ report) under BOTH Ridge(α=1.0) and the banked strong probe
   MLPRegressor((128,64), max_iter=600, rs=0), 5-fold KFold(rs=0), unclipped. Both ≤ **0.05** ⇒
   decoupled. (0.01 was the slate draft; 0.05 is the pre-registered bound actually used — the 1-D
   noise floor at n=10k makes 0.01 needlessly brittle; the partial-R² readouts control for residual c
   regardless.) If coupling > 0.05: that (head, seed) is named **STALE-GENERALIZES** (the accessible-
   regime demodulator partially transfers to λ=2 — informative in its own right, NOT a void).
2. **Structure floor:** Var(report)/Var(c) ≥ **0.25** ⇒ structured; else **COLLAPSED** (graceful
   degradation to mean-reporting).
3. **Projection readout** (only meaningful for decoupled+structured cells): partial-R² of report vs
   each proxy CONTROLLING for c — residualize both report and proxy on c (linear), then R² between
   residuals (point-biserial form for binary d). Record max over {d, g, mag}.

**Within-design control (apparatus):** the same frozen reporter on the λ=0 control draw must read
R²(c ~ report) ≥ 0.9 AND every proxy partial-R² ≤ 0.1 — it really is a c-reporter where access exists.
Failure for any (head, seed) ⇒ VOID.

## 5. Pre-registered outcomes (per-cell verdicts → run verdict; precedence top-down; no dead zone)

Cell verdict (one per head × seed):
- **SG** STALE-GENERALIZES (coupling > 0.05) · **CO** COLLAPSED (structure floor fails) ·
- **PS** PROJECTION SIGNATURE: decoupled + structured + max proxy partial-R² ≥ **0.5** ·
- **IS** ISOTROPIC: decoupled + structured + ALL proxy partial-R² < **0.1** ·
- **PP** PARTIAL-PROJECTION: decoupled + structured + max partial-R² ∈ [0.1, 0.5).

Run verdict:
| # | Outcome | Condition | Reading |
|---|---------|-----------|---------|
| V | **VOID** | MI gate, train-fit gate, or within-design control fails | apparatus; fix and re-run |
| A | **SIGNATURE** | ≥ 4/5 seeds PS, BOTH heads | the stale report projects onto the determinable subspace — a proxy-correlation audit can flag reports about inaccessible state; the determine/resist axis structures report behavior |
| K1 | **GRACEFUL-COLLAPSE** (clean null = SUCCESS) | ≥ 4/5 seeds CO, both heads | stale strategies degrade to mean-reporting; report-audits cannot flag what collapses to the prior — banked as an audit bound |
| K2 | **ISOTROPIC** (clean null = SUCCESS) | ≥ 4/5 seeds IS, both heads | structured residue with no determinable-subspace lock — no signature for audits; banked |
| G | **STALE-GENERALIZES** | ≥ 4/5 seeds SG, both heads | the deploy-shift cannot manufacture decoupling for this strategy class — the accessible-regime demodulator transfers; banked as a named boundary of the design |
| D | **ARCHITECTURE-DEPENDENT** | the two heads' modal cell verdicts differ (each ≥ 4/5 internally) | the projection target is head-dependent — itself the pre-registered banked finding (slate fix #4) |
| M | **MIXED** | anything else (seed splits ≤ 3/2 in either head) | banked with full per-cell table; no headline claim |

## 6. Determinism, seeds, files, commands

- Thread pinning as HS4 (OMP/MKL=1 in-script before numeric imports; torch single-thread).
- **Seed ledger** (disjoint from banked {1235, 1241–4241, 1245, 1256, 1267, 2233} and HS4's {51235,
  61235, 1789, 71235, 81235, 86235, 91235}): train draw **101235** (SEED+100001); deploy λ=2 draw
  **111235** (SEED+110001); λ=0 control draw **121235** (SEED+120001); reporter init seeds
  **131235–131239** (SEED+130001+k, k ∈ {0..4}, matching §3); n_train=8000, n_deploy=10000,
  n_control=4000.
- Script: `scripts/shadow_confab_signature.py` → `results/atlas/h3/confab_signature_result.json`
  (per-cell table, gates, run verdict, unclipped values).
- Frozen test: `scripts/test_shadow_confab_signature.py` — reduced-size real path (train 2000 / deploy
  2500 / control 1000 = the FIRST rows of the full-size draws, subset rule as HS4), 1 seed both heads,
  pinning: MI gate, train-fit gate, within-design control, byte-identical readouts across an
  in-process rerun. Full-run verdict NOT asserted at reduced size.
- Exact unchanged commands: `python scripts/shadow_confab_signature.py` ·
  `python scripts/test_shadow_confab_signature.py`. Existing suite stays green
  (`python scripts/test_shadow_pooled_synthetic_v2.py`).

## 7. Honest boundaries (pre-stated)

- Synthetic substrate; the "report" is a regression head, not a language-model self-report — this is
  the projection structure of a stale demodulator, full stop. The probe-vs-introspection gap is the
  standing limitation.
- The proxy set {d, g, mag} is the DECLARED determinable set; a signature on an undeclared survivor
  would read as IS/PP here (stated, accepted — the declared set is what an auditor would have).
- Outcome A does not claim audits work on real systems; it establishes the mechanism exists in the
  one substrate where ground truth is exact. K1/K2/G all bound the design honestly.
- HS3's landed lesson binds interpretation: existence of structure ≠ trainability of detection; no
  trained-detector claims are made here.
