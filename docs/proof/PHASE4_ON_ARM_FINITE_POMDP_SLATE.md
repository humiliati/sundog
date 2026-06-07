# Phase 4 ON-arm — Finite-POMDP Sufficiency Receipt (FROZEN → EXECUTED)

> 2026-06-06. Companion to [`../COARSE_GRAINING_PROOF_ROADMAP.md`](../COARSE_GRAINING_PROOF_ROADMAP.md)
> ▸ Phase 4 and [`PHASE2_MDP.md`](PHASE2_MDP.md). **FROZEN before execution** (anti-p-hack: the
> construction, the `Φ` map, the seeds, and the pass condition were fixed before any number was
> read). Operator approved B-i (finite POMDP) on 2026-06-06. **EXECUTED 2026-06-06 → verdict
> `on_arm_pass_two_sided_gate_fired`** (§8). Harness `scripts/proof_phase4_on_arm_finite_pomdp.py`;
> receipt `results/proof/phase4-on-arm/`. The frozen pass condition (§1, §6) was met as written; no
> knob was changed after reading numbers.
>
> **Why this exists.** The committed three-body Phase-4 receipt
> (`results/proof/phase4/phase4-regret-summary.csv`) has `on=0` rows: the two-sided gate's ON arm —
> *signature-only regret → 0 on the `𝓕_σ`-measurable cell set* — has **never been populated** on any
> substrate (three-body produced only `off`, non-decisively; Balance measured the OFF-direction
> "Φ-accessible" positive). This run populates the ON arm for the first time, on the substrate where
> measurability is **provable** (Phase-2's reviewer-closed criterion), and makes the two-sided gate
> fire.

## 1. Frozen gate (verbatim from the roadmap, NOT retuned)

> *signature-only regret vs Bayes must → 0 (within bootstrap CI) **on** the `𝓕_σ`-measurable cell set
> and stay bounded away from 0 (CI excludes 0) **off** it. If regret is bounded-away **on** the
> measurable set, sufficiency-for-control is empirically false → halt and falsify. If regret → 0
> **off** the measurable set, the [classifier] is wrong.* (COARSE_GRAINING_PROOF_ROADMAP.md §Phase 4)

## 2. Substrate (frozen) — two-mode memory corridor POMDP

A finite, episodic POMDP that realizes Phase-2 Theorem 1's positive toy (ON) and its required
counterexample (OFF) in one substrate, in the multi-step belief regime the Phase-4 Bayes-floor
instrument lives in.

- **Hidden mode** `m ∈ {0,1}`, drawn uniformly per episode.
- **Horizon** `T = 6` steps. Step `t=0` = **signpost**; `t=1..T-2` = **corridor**; `t=T-1` = **fork**.
- **Signature observation `Φ`** (the same-information budget both controllers receive):
  - signpost cell is mode-revealing but **noisy**: emits `S{m}` with prob `1-ε`, the wrong `S{1-m}`
    with prob `ε` (`ε = 0.15`, frozen);
  - corridor cells are mode-independent (`C0..C{T-3}`);
  - **fork cell `FORK` is identical for both modes** (aliased — the OFF construction).
- **Actions** `A`: on signpost/corridor, `advance` is uniquely optimal (any other action = penalty);
  at the fork, `go0`/`go1`, **safe iff action == m**.
- **Return / safety** = fraction of decisions that are safe-optimal over the episode (the `T_safe`
  analog), in `[0,1]`.

`Φ`-fibers and their ON/OFF labels are assigned by **Phase-2 Theorem 1** (a cell is ON iff the
optimal-action correspondence has a non-empty intersection over the states/beliefs mapping to it):
- signpost + corridor cells → optimal action is `advance` for **both** modes → `⋂A* ≠ ∅` → **ON**;
- `FORK` → `A*` is `{go0}` for `m=0`, `{go1}` for `m=1`, **disjoint** → `⋂A* = ∅` → **OFF**
  (Phase-2's flip-counterexample, exactly).
Labels are **by construction + the proven criterion**, not fit to the result.

## 3. Controllers (frozen)

- **Signature controller** = the best **memoryless `Φ`-measurable** policy `g: Σ → A` (the strongest
  policy that factors through the instantaneous signature; computed by exact enumeration/DP over the
  finite `Σ→A` space). On ON cells `g = advance` (optimal). On `FORK` (aliased 50/50) the best `g`
  is either `go0`/`go1` → expected safe rate `0.5`.
- **Bayes-optimal same-information floor** = the optimal policy on the **belief** over `m` formed from
  the **same `Φ` history** (exact belief-MDP value iteration; finite). It learns `m` from the noisy
  signpost → at the fork it acts on the posterior → expected safe rate `≈ 1-ε`.
  No privileged true `m`; same observation budget as the signature controller, used with history.

## 4. Regret instrument (canonical schema)

`regret = (return_bayes − return_signature)` per paired decision, normalized to `[0,1]`; per-cell
classified `on/off/undecidable` (`undecidable` = fiber with `< fiberMinSamples=20` samples); paired
**2000-iteration bootstrap** 95% CI (2.5/97.5 percentile), matching
`phase4-regret-manifest.json`'s reducer. Emits `phase4-regret-summary.csv` in the committed schema
(`on/off/undecidable` rows) plus a per-case CSV and a manifest, under
`results/proof/phase4-on-arm/` (the committed `results/proof/phase4/` three-body receipt is **not**
touched).

## 5. Frozen knobs

`T=6`, `ε=0.15`, `n_episodes=2000`, `mode~Uniform{0,1}`, `fiberMinSamples=20`, `bootstrap_iters=2000`,
`bootstrap_seed=40604` (matched to the three-body receipt for cross-comparability),
`classSeedRule = bootstrap_seed + classIndex*1009 over [on,off,undecidable]`.

## 6. Pre-registered verdict branches

- **ON-arm PASS (the target):** `on` regret CI **includes 0 / → 0 within CI** AND `off` regret CI
  **excludes 0** (bounded away `> 0`). → the two-sided gate fires; first banked ON-arm sufficiency
  receipt.
- **Sufficiency-false:** `on` regret CI bounded-away `> 0` → halt and falsify (would mean the
  signature is insufficient even on the provably-measurable set — a construction/implementation bug,
  since Phase-2 proves it sufficient there).
- **Classifier-wrong:** `off` regret CI includes 0 → the OFF construction did not bite (re-examine
  `ε`/aliasing), report-only.

## 7. Honest scope (load-bearing — carried into the receipt)

This closes the **empirical ON-arm of the *sufficiency* gate on a constructed finite POMDP** — the
first populated `on` rows, the two-sided gate firing for the first time. It is the re-pegged
**Blackwell control-sufficiency** claim's missing positive, where measurability is provable. It does
**NOT** establish: the founding **body-resistance / regime-2** claim, a real or high-dimensional
substrate, or a **trained** body. The non-tautological content is the **discrimination** — the same
instrument fires `→0` ON and `bounded-away` OFF against a known ground truth — not the ON value
alone. Per `CROSS_SUBSTRATE_NOTES.md`, the founding resistance receipt remains the standing frontier
and is out of scope here.

## 8. Receipt (EXECUTED 2026-06-06)

**Verdict: `on_arm_pass_two_sided_gate_fired`** — the two-sided gate fired as pre-registered; the
`on` class is populated for the first time across the program (the committed three-body receipt had
`on=0`). Deterministic given `bootstrap_seed=40604`; no wall-time in the receipt.

| cell_class | row_count | mean_regret | 95% CI | neg_regret_rate |
| --- | ---: | ---: | --- | ---: |
| **on** | 2000 | **0.0** | **[0.0, 0.0]** | 0.0 |
| **off** | 2000 | **0.3525** | **[0.32598, 0.37950]** | 0.076 |
| undecidable | 0 | — | — | — |

- **ON arm → 0 within CI:** PASS. On the `𝓕_σ`-measurable cells the best memoryless `Φ`-policy is
  *exactly* the same-information Bayes-optimal action (Phase-2 Theorem 1, sufficiency direction), so
  per-decision regret is structurally 0. The strongest possible ON result; the `[0,0]` CI is honest,
  not engineered.
- **OFF arm bounded away from 0:** PASS. CI excludes 0; mean `0.3525` matches the closed-form
  `(1−ε) − 0.5 = 0.35` (ε=0.15) — a confirmation of the frozen prediction, not a fit.
- **The non-tautological content is the discrimination:** the *same* instrument fires `→0` ON and
  `bounded-away` OFF against a known ground truth — i.e. the Phase-4 gate is validated/calibrated.

Receipt files: `results/proof/phase4-on-arm/{phase4-regret-summary.csv, phase4-regret.csv,
phase4-regret-manifest.json}`.

**What this does and does not establish (re-stated, load-bearing).** Establishes: the first populated
ON-arm of the sufficiency gate, on a constructed finite POMDP where measurability is provable — the
`row_count=0` gap is closed and the two-sided gate fires. Does **not** establish: the founding
body-resistance / regime-2 claim, a real / high-dim substrate, or a trained body (see `§7`,
`../CROSS_SUBSTRATE_NOTES.md`). The standing frontier is unchanged.

---

*Sundog Research Lab — Phase-4 ON-arm finite-POMDP slate. FROZEN → EXECUTED; two-sided gate fired.
Sufficiency, not resistance; constructed substrate, not real; provable measurability, not fit.*
