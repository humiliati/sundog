# Phase-4 ON-arm — Three-Body Staging + GPU/1080 Feasibility

> 2026-06-06. **Staging handoff, NOT executed.** Operator-staged long-budget job to attempt the
> **real-substrate** ON cell (the thing three-body's committed run could not produce: `on=0`). Sibling
> to the *constructed-substrate* close [`PHASE4_ON_ARM_FINITE_POMDP_SLATE.md`](PHASE4_ON_ARM_FINITE_POMDP_SLATE.md)
> (executed, gate fired) and the substrate-empirical leg in
> [`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md). The roadmap forbids running
> the three-body IAD inline (~30 h / 8 seeds / 1 cell; AGENTS.md ~10-min rule). This doc stages it.

## 0. What this attempts (and the honest prior)

Populate the canonical `on` class on a **real** substrate: a three-body regime + signature where the
Bayes-floor action is **fiber-constant** (`𝓕_σ`-measurable, common `action_key` over ≥`fiberMinSamples`)
and the signature controller **matches** the Bayes floor (regret → 0), with a paired `off` regime
bounded away. **Honest prior: likely `on=0` / `undecidable` again** — three-body's `near_escape` pocket
had no measurable fibers, and the only banked positive (Balance) is OFF-direction. A clean real `on`
would be the genuine scale-up of the constructed-POMDP win; a null is an informative real-substrate
boundary. Either way the regime + partition are **pre-registered before the run** (anti-p-hack).

## 1. Interpreter / GPU finding (load-bearing)

- The **only** CUDA-capable interpreter on this host is the **`sundog-gpu` venv, Python 3.12.10**
  (`torch 2.5.1+cu121`, GTX 1080, Pascal cc 6.1). The default `python`/`py` is **3.14** with a
  **CPU-only** torch — using it silently runs on CPU (the 2026-06-05 footgun). Any GPU runner **must**
  invoke `C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe`.
- The 1080 (8 GB, cc 6.1) is ample for batched particle rollouts (512 particles × a few state dims).

## 2. The compute + why CPU-Node is slow

`threebody-phase4-bayes-floor.mjs` (Node) runs a particle-filter MPC: per control decision it rolls
out **`--particle-count 512` × `--planning-horizon-steps 800`** of planar-restricted-3-body dynamics,
with `--candidate-hold-steps` action persistence. Serial in Node → ~3.75 h/seed. The rollouts are
embarrassingly parallel over a trivial ODE → the GPU win is structural.

## 3. Two staging paths

### Path A — CPU-Node, sharded + checkpointed (exists today; safe, slow)
The infrastructure is already there and resume-safe:
- `scripts/threebody-phase4-iad-shard.mjs --seed <S> --particles 512` — single-seed shard; **skips if
  `manifest.json` exists** in the shard dir (`--force` to redo). Checkpoint = per-seed shard dir.
- `scripts/threebody-phase4-iad-concurrent.mjs` — worker-pool orchestrator over seeds (offload, not
  speedup, on a many-core host — the dynamics are serial per shard).
- `scripts/threebody-phase4-iad-merge.mjs` → `scripts/threebody-phase4-regret.mjs` (the `on/off/
  undecidable` fiber classifier + 2000-iter paired bootstrap → `phase4-regret-summary.csv`).
- **Run as a long-budget job** (`workflow_dispatch` / overnight), not an agent session. Cost for the
  ON-arm: `(ON-candidate cells + OFF anchor) × 8 seeds × ~3.75 h` ≈ tens of hours. No new code.

### Path B — GPU-torch port on the 1080 (real build; fast; needs a faithfulness gate)
Reimplement only the **particle-MPC rollout** in torch on the 3.12 venv:
- State the planar R3BP step as a batched torch op; integrate `[512 particles × K candidate actions]`
  × 800 steps as one `(B, ...)` tensor on the 1080. Keep the **exact** Node contract: same particle
  count, horizon, candidate-hold, resampling, action_key set, and the shard/`manifest.json` resume
  protocol (so the merge/regret reducer is unchanged and the on/off classifier is identical).
- **FAITHFULNESS GATE (mandatory, before any banked number):** on a small case (1 seed, 64 particles,
  short horizon, fixed RNG) the torch port must reproduce the Node reference's per-decision
  `action_key` sequence and `T_safe`/regret **within tolerance** (ideally byte-identical action keys;
  any divergence = the port is not the proven harness → fix before use). This is the same discipline
  that caught the v5 "FINAL" fabrication — *the GPU run does not exist until it matches CPU on a
  controlled case.*
- Expected speedup: 10–50× (512-particle batch on a trivial ODE) → the full ON-arm becomes
  single-digit hours, and future three-body Phase-4/5 work becomes tractable. **Effort: ~half a day**
  (port + the faithfulness gate). Recommended if three-body is going to be run more than once.

## 4. The ON-arm experiment design (pre-register before the run)

1. **Regime (the load-bearing pre-registration).** Do NOT reuse `near_escape` (produced `on=0`).
   Pre-register an **ON-candidate regime** where the optimal thrust is plausibly coarsely
   signature-determined (e.g. a deep-captured / low-`|tidal|` stable pocket where "hold toward
   periapsis" is fiber-constant), **plus** the `near_escape` cell as the paired **OFF anchor** →
   two-sided. The regime choice is the p-hack surface; freeze it with a stated rationale, not tuned to
   the result.
2. **Signature + Bayes floor:** unchanged — `track_sensor_accel_guarded` signature vs
   `bayes_floor_particle_mpc` (same observation budget), per the committed manifest.
3. **Classifier (unchanged, canonical):** `threebody-phase4-regret.mjs` partition keys
   (`guard_t, log_binned_abs_tidal_magnitude, gradient_angle_sector, log_binned_gradient_magnitude,
   sensor_noise_std`), `fiberMinSamples=20`, exact `action_key` common-action rule, `bootstrap_seed=40604`.
4. **Frozen gate (roadmap verbatim, NOT retuned):** `on` regret → 0 within CI AND `off` CI excludes 0.
   Named nulls: ON-candidate regime → `undecidable` (fibers under-sampled — raise seeds/trials) or
   `off` (Bayes action not fiber-constant even in the stable pocket → "real substrate not cleanly
   `𝓕_σ`-measurable under this signature" — report, do not retune).
5. **Scope (carried into the receipt):** a real-substrate **sufficiency** result, still NOT
   body-resistance/regime-2 and NOT a trained body. The constructed-POMDP `on` is banked; this tests
   whether it survives on a real continuous substrate.

## 5. Recommended sequence

1. **Freeze §4.1 regime + partition** (operator) — the only genuine decision; everything else is pinned.
2. **Build Path B** (GPU torch port + faithfulness gate vs Node) — *or* skip to Path A if a one-shot
   overnight CPU run is acceptable and the port is not worth the half-day.
3. **Stage the run** (`workflow_dispatch` / overnight) with the venv-3.12 interpreter for Path B, or the
   existing Node shard fleet for Path A; resume-safe via the per-seed `manifest.json` checkpoint.
4. **Reduce + classify** with the unchanged `threebody-phase4-regret.mjs`; file the receipt against the
   frozen gate; update `COARSE_GRAINING_PROOF_ROADMAP.md` §Phase-4.

---

*Sundog Research Lab — Phase-4 ON-arm three-body staging. Operator-staged; GPU path = the 3.12
sundog-gpu venv on the 1080, gated by a CPU-faithfulness check. Real-substrate sufficiency attempt;
honest prior is a null. Regime pre-registration is the one decision that must precede the run.*
