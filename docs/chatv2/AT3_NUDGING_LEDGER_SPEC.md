# AT-3 — Nudging-Ledger Spec (frozen; the maintained-ledger form, first build)

> 2026-07-03, **v1.1** (pre-lock amendments from the smoke, before any verdict-bearing
> run: (1) calibration block PRECEDES the ledger window — the registered C1 ordering,
> made explicit after the smoke caught a calibration-after implementation infidelity;
> (2) ledger window 200,000 → **500,000 steps** — the smoke measured E_low envelope
> wander on multi-10k-step scales (realized 20k-window damp = 0.000 against a
> calibration threshold from the adjacent epoch); short windows are unpowered for the
> registered quantile action. Realized in-window damp is reported per regime; smoke uses
> a loudly-labeled fallback threshold for readout-path validation only.)
> Lift of slate entry AT-3 into its frozen pre-registration. **New script,
> read-only import of the frozen harness (no harness change — no sign-off surface).**
> Analysis and grids frozen here before any curve exists. Non-promotional; no new
> synchronization threshold claimed for NSE (K_sync is measured on the 32×32 truncation);
> no "the ledger understands/models the flow" — licensed grammar only. AT-2b context:
> the static read collapsed decision to state budgets (K=1/2); AT-3 asks whether the
> DYNAMICAL sync threshold opens a gap the static read cannot see.

## 1. Objects (frozen)

- **Truth u:** the registered cell (at2 preset numerics: grid 32, dt 0.01, k_f 2, ν=√(1/G),
  seed 0), burn-in 100,000 steps, then a **500,000-step ledger window** per config (v1.1).
- **Ledger v (AOT nudging):** same integrator; explicit stage gains
  **−μ·P_Kobs(v − u)** on the observed modes (select(K_obs) + conjugates), then the
  standard implicit viscous solve. **v(0) = a fresh random field (seed 1)** — the ledger
  starts wrong and is maintained only by the observation stream.
- **Grids (frozen):** K_obs ∈ {1,…,6} (nested, as AT-2); **μ ∈ {1, 10, 50}**; regimes
  G ∈ {300 (primary), 200 (replication)}. No μ tuning outside the grid (`AT3_NEG_B`).
- **State-sync error:** err(t) = ‖ω_v − ω_u‖₂/‖ω_u‖₂ on the full grid.
  **Synchronized** = median err over the last 25% of the window < 0.01;
  **state-insufficient** = that median ≥ 0.1; between = transient (reported, no verdict
  role). **K_sync(μ, G)** = least K_obs synchronized.
- **Decision readout:** logistic regression on **Φ_K3(v)** (the registered K=3 signature
  computed from the *ledger* state) at sample instants (every 50 steps, first 25% of the
  window discarded as ledger transient; 70/30 contiguous split, 2,500-step gap, seed 0),
  against the **true frozen actions** (J_q(τ=500) of the *truth*: lookahead-max E_low_K3
  > q=0.70 calibration threshold, computed exactly as registered).
- **Controls (frozen):**
  - **Scrambled ledger** (floor): identical, but each step feeds an observation drawn
    uniformly from the already-elapsed window (seed 2) — same marginal statistics,
    broken dynamics. Run at μ = 10, all K_obs.
  - **Decision-only twin:** a 9-mode Galerkin truncation (select(3) band only, all other
    modes zeroed each step; closure-free, registered as crude) with the same nudging —
    same observation budget, no capacity to reconstruct the high state. If the twin
    matches the full ledger's decision read, the carry is decision-typed.
  - **Liveness:** if no (K_obs, μ) in the grids synchronizes, `AT3_DEAD_APPARATUS` —
    void, not a negative (binding lesson 4).

## 2. Branch table (frozen; δ = 0.10, δ_twin = 0.05)

Let acc(K, μ) = ledger decision accuracy, maj = majority-class rate, scr(K) = scrambled
accuracy. A (K_obs, μ) cell is **split-positive** iff: state-insufficient (median err ≥
0.1, non-transient) AND acc ≥ max(maj, scr(K)) + 0.10 AND |acc_full − acc_twin| ≤ 0.05.

| branch | fires iff |
| --- | --- |
| `AT3_LEDGER_SPLIT_CONFIRMED` | ∃ registered cell split-positive with K_obs < K_sync(μ) at the same μ |
| `AT3_VACUOUS_GAUGE_COLLAPSE` | at every μ, decision success and state sync cross at the same K_obs (within grid resolution) — the §3.6 negative at the ledger level, completing AT-2b's static read dynamically |
| `AT3_JOINT_INSUFFICIENT` | below K_sync the decision also fails everywhere (acc < max(maj, scr) + 0.10) — the ledger form does not separate here |
| `AT3_SHARP` / `AT3_GRADED` | reported: the failure shape of acc across K_obs (cliff vs smooth) |
| `AT3_DEAD_APPARATUS` | liveness fails (nothing synchronizes) — void |
| `AT3_NEG_B` | post-read change to grids, thresholds, window, or readout — voids |

Verdict per regime; primary = G=300, replication = G=200 (reported; no gate arithmetic
across regimes).

## 3. Commands & deliverables

- Script: `scripts/pde_c1_nudging_ledger.py` (new; read-only imports). Per-config JSON
  checkpointing — the sweep is resumable; re-running skips finished configs.
- **Smoke (agent-run, pre-batch):** `--smoke` (5k-step window, K_obs ∈ {2, 6}, μ = 10,
  G=300) — validates sync machinery, readout, controls. Non-verdict.
- **Owner batch (overnight, one command per regime):**
  `python scripts/pde_c1_nudging_ledger.py --grashof 300 --out results/proof/at3-g300`
  `python scripts/pde_c1_nudging_ledger.py --grashof 200 --out results/proof/at3-g200`
  (≈ 18 full + 6 scrambled + 18 twin configs per regime; ~4–5 h each, resumable.)
- Post + receipt: `AT3_NUDGING_LEDGER_RECEIPT.md` — K_sync table (μ × G), the
  acc/err/twin table per cell, branch verdicts, comparison of K_sync against the static
  K_state=1/2 and the K\* > 4 twin-certificate wall.

## 4. Does not claim (inherited + entry-specific)

No NSE synchronization theorem (K_sync is cell-specific, µ-grid-specific); the twin is a
declared crude truncation, not "the" decision-only estimator; μ-performance claims only
relative to the registered controls; a confirmed split licenses only "the maintained
ledger carries the registered decision at a budget below its own synchronization
threshold, on this cell" — nothing about understanding, world-models, or ∞-dim NSE;
the slate do-not-say list binds in full.
