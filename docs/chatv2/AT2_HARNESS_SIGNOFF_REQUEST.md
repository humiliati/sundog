# AT-2 Harness Sign-Off REQUEST — STAMPED

> 2026-07-03. AT-2 (growth law) needs a harness surface beyond `AT1_HARNESS_SIGNOFF.md`'s
> AT-1-only scope. **Status: `AT2_HARNESS_SIGNED_OFF` — stamped by owner instruction
> ("let's implement", 2026-07-03, alongside the spec's v1.1 amendment). Implemented same
> day; regression gate PASSED** (self-test ✓; capped smoke with/without `--at2-export`
> identical on pre-existing manifest fields ✓; K3-consistency check: per-step E_low_K3
> equals ‖Φ_K3-slice‖² at sample instants to float32 precision ✓; smoke receipts at
> `results/proof/at2-smoke-{noexp,exp}/`). Same discipline as AT-1: additive only, frozen
> presets keep semantics.

## Why a new surface (the coupling trap, found at freeze time)

In the harness, `k_signature` defines **both** the shadow Φ_K **and** the E_low band —
so sweeping K via the existing presets would change the *objective* along with the
observation budget, breaking the growth law's semantics (K_min must be measured against
a **fixed** target). Verified: `select_low_modes` is exactly nested
(select(1) ⊂ … ⊂ select(6), sizes 1/4/9/16/25/36, forced-mode swap included), so the
clean design is **one wide emission, sliced in post**: the objective stays the frozen
K=3 E_low; the shadows Φ_K, K ∈ {1..6}, are column slices of a K=6 signature.
This also collapses the slate's ~8–16 owner runs to **two** (one per G).

## Requested surface (all additive)

1. **New presets `at2_growth_g200` / `at2_growth_g300`:** lock_v7 numerics unchanged
   (grid 32, dt 0.01, k_f 2, seed 0, burn-in 100k, 50k samples @ interval 50,
   calibration 50k / gap 5k) except `k_signature = 6` (wider *emitted* shadow only) and
   `lookahead_steps = 2000` (so every τ ≤ 2000 is computable in post). Not added to
   `VERDICT_BEARING_PRESETS` — their own manifests are non-interpretable by the existing
   machinery; only the AT-2 export is read.
2. **`--at2-export PATH` flag** writing a schema-v1-at2 npz (~70 MB): per-step
   **E_low_K3 series** (float32; computed over the frozen K=3 mode set via a dedicated
   mask — the registered objective's observable, independent of the run's K=6 band);
   Φ_K6 at sample instants (50k × 72) + the (kx, ky) list per signature column;
   adj/calib starts; per-sample state proxies (E_high, high-mode norm — declared, for
   the §3.6 vacuity read); config echo. No change to existing scoring or manifest fields.
3. **Post-processor `scripts/at2_growth_law.py`** reading the npz only.
4. **Regression gate (AT-1 pattern):** `--self-test` passes; capped smoke with/without
   `--at2-export` agrees on pre-existing manifest fields; old presets bit-unchanged.
   Drift ⇒ `AT2_HARNESS_VOID`.

## Also owner-run (existing presets, no harness change)

The K\* bracket side-deliverable (`PDE_C1_SEPARATION_STATEMENT.md` §5, pre-registered,
never run): `lock_v5_k5` and `lock_v5_k6` twin runs (~35 min each). Banked K=2/3/4 are
all `TWIN_STATE_CERTIFIED` with abundant witnesses, so the bracket is genuinely open;
if K=5/6 still certify, the honest report is "K\* > 6 at this cell."

## Owner runs, total (batchable overnight)

- `python scripts/pde_c1_kolmogorov_cell.py --preset at2_growth_g200 --at2-export results/proof/at2-g200/at2-samples.npz --out results/proof/at2-g200` (~50 min)
- same for `at2_growth_g300` (~50 min)
- `--preset lock_v5_k5 --out results/proof/c1-k5-twin` and `--preset lock_v5_k6 --out results/proof/c1-k6-twin` (~35 min each)

The frozen AT-2 spec (τ-grid, K_min definition, gates incl. the AT-1-inherited
threshold-atom clearance check, identical thresholds at every cell) is drafted alongside
and freezes before any read. Nothing here reopens banked receipts or edits public surfaces.
