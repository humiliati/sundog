# NSE-H3 Mid-Energy Sub-Regime Restriction — Spec (frozen)

> 2026-07-07. Registered follow-up from `NSE_H3_G675_STRUCTURE_DIGIN.md`: the
> matched-Re G=675 witness has both regime-2 halves on a dense nucleus but is
> coverage-limited overall (0.469 < s_pos 0.50), and coverage *peaks in the
> mid-energy band* (0.690). This tests whether the frozen certificate CLEARS on
> the mid-energy sub-attractor. **Frozen before the restricted read.**
> Non-promotional; a positive is scoped to the mid-energy sub-regime of one
> matched-Re cell, proxy-relative, no C1 promotion, `docs/chatv2/` no-publish.

## 0. The gate-shopping risk, and how this spec binds it

The dig-in *identified* the mid-E band by observing its coverage post-hoc. To keep
this a genuine test and not a cherry-pick, three things are fixed here **before**
the read:

1. **The band is fixed by rule, not by a coverage target:** the **central energy
   tercile** `[q_{1/3}, q_{2/3}]` of the eval-block `E_low` — the principled
   "sub-attractor near the energy mode," not "the cut that gives coverage 0.690."
   No coverage-derived boundary is used.
2. **Every verdict gate is the frozen protocol, unchanged:** ε_K (global frozen),
   `s_pos = 0.50`, twin witness-mass gates, `delta_action = 0.10`. The *only* new
   element is the sample restriction.
3. **One read, typed outcome, no iteration.** No second band, no widening. The
   `UNDERCOVERED` null (below) is a first-class result — restricting to a
   self-contained sub-sample may *reduce* coverage below the cross-band 0.690.

## 1. Restriction (frozen)

- **Cell:** the banked `fallback_v7_g675_kf3` samples
  (`results/proof/c1-h3-kf3-g675-adaptive/samples.npz`, 200k). Post-processing;
  no re-integration.
- **Band:** `E_low(u) = ‖Φ_K(u)‖²`; `q1, q2 = quantile(E_low, [1/3, 2/3])`;
  mid-E set = `{u : q1 < E_low(u) ≤ q2}` (~66k samples).
- **Self-contained sub-attractor:** the twin certificate is rebuilt entirely
  **within** the mid-E set (BallTree on mid-E signatures/high-modes/actions only) —
  both query points and their neighbours are mid-E. This is the honest
  "sub-attractor" object; its coverage is whatever the restriction yields (the
  cross-band 0.690 is *not* assumed to survive).
- **ε_K:** the banked frozen ε_K for the cell (0.0589), unchanged — the
  restriction is on the sample set, not the radius.

## 2. Adjudication (frozen twin-state)

Run the **unchanged** `aggregate_twin_state` on the mid-E sub-sample: it yields
candidate coverage, the state-insufficiency witness certificate, and the
paired-fiber (control-sufficiency) read in one pass.

## 3. Regression gate (agent-run; precondition)

Apply the identical restriction machinery to the banked **anchor** samples
(`c1-relative-reg-g200`, `-g300`; 50k each) and confirm the restricted read still
`TWIN_STATE_CERTIFIED` — the anchors are near mono-energy (range 1.03/1.15:1) so the
central tercile is representative and must reproduce the banked certification. A
failure means the restriction machinery is broken; fix before reading G=675.

## 4. Verdict branches (frozen)

| mid-E restricted read | verdict |
| --- | --- |
| coverage ≥ 0.50 ∧ `TWIN_STATE_CERTIFIED` ∧ paired `POSITIVE` | **`NSE-H3-FORCING-GENERAL-MIDE`** (scoped to the mid-energy sub-attractor) |
| coverage ≥ 0.50 ∧ `CERTIFIED` ∧ paired `NEG` | `NSE-H3-GRASHOF-LOCAL-MIDE` |
| coverage < 0.50 (`DEFERRED_COVERAGE`) | `NSE-H3-MIDE-UNDERCOVERED` (the restriction does not clear either — the mode-band's within-band coverage is below the cross-band figure) |
| high-mode floor / vacuity | typed as the frozen aggregator reports |

No gate widening, no retune after the read. `FORCING-GENERAL-MIDE` is the first
genuinely gate-clearing forcing-axis positive, if it fires — but **scoped to the
mid-energy sub-regime**, not the full G=675 attractor, and inheriting all the
lane's fences (finite, proxy-relative, sampled-support, no promotion).

## 5. Claim boundary

A `FORCING-GENERAL-MIDE` positive says: on the mid-energy sub-attractor of the
matched-Re (k_f=3, G=675) cell, the frozen regime-2 certificate clears its
registered gates (coverage ≥ 0.50, state-insufficient, control-sufficient). It
does **not** claim the full G=675 attractor (which is coverage-limited — A SLIVER,
frozen DEFER), overturn any prior verdict, reveal scale-invariant structure,
generalize across forcing wavenumbers, or promote C1. It is one scoped sub-regime
point on one forcing-axis move.

## 6. Deliverable

`scripts/nse_h3_mide_subregime.py`: `--self-test` (synthetic restriction check),
`--regress G200_DIR G300_DIR` (§3 anchor regression), `--run CELL_DIR` (the mid-E
read + branch). Post-processing on banked `samples.npz`; agent-runnable (the 66k
BallTree is lighter than the 200k reprocess, but the box is contended — expect
minutes, not seconds). Receipt: `NSE_H3_MIDE_SUBREGIME_RECEIPT.md`.

Cross-refs: `NSE_H3_G675_STRUCTURE_DIGIN.md` (the lead),
`NSE_H3_ADMISSION_RECEIPT.md`, `PDE_C1_TWIN_STATE_CERTIFICATE.md` (the frozen
certificate), `results/proof/c1-h3-kf3-g675-adaptive/samples.npz` (the cell),
`results/proof/c1-relative-reg-g{200,300}/samples.npz` (regression comparators).

---

> **Post-run status (2026-07-07): `NSE-H3-FORCING-GENERAL-MIDE`**
> (`NSE_H3_MIDE_SUBREGIME_RECEIPT.md`). Regression PASS (anchors reproduce
> CERTIFIED under restriction). Mid-E read: within-band coverage 0.6863 (> 0.50;
> ≈ cross-band 0.690 ⇒ mode band self-coherent, UNDERCOVERED null was live and did
> not fire), `TWIN_STATE_CERTIFIED` (82,547 witnesses), paired POSITIVE (disagree
> 0.0218). **First clean gate-clearing forcing-axis positive — scoped to the
> mid-energy sub-attractor**, not the full G=675 attractor (tails stay
> coverage-limited). Regime-2 generalizes to matched-Re forcing on the mode band.
