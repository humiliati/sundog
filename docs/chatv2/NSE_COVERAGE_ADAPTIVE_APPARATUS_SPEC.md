# Coverage-Adaptive Fiber Apparatus — Spec (Approach A, frozen)

> 2026-07-07. Lift of `NSE_COVERAGE_ADAPTIVE_APPARATUS_SCOPE.md` into its frozen
> pre-registration. Owner signed off **Approach A** (fixed-radius,
> density-stratified; `ε_K` unchanged) plus the additive `--adjudicator
> twin-state-adaptive` harness mode. **Every constant, gate, and branch is frozen
> here before any adaptive number exists.** Non-promotional; apparatus-generality
> question, not an H3 rescue — H3 stays closed at `NSE-H3-INCONCLUSIVE_COVERAGE`
> unless this apparatus passes §3 regression *and then* reads G=675.

## 1. Approach A, exactly

`ε_K = 0.05·√(2·E_max)` is **unchanged** (same global rule, same value per cell).
No pair whose signature distance exceeds `ε_K` is ever compared — fidelity is
preserved by construction. The only change is coverage bookkeeping:

- Query the `twin_k_neighbors = 50` nearest neighbours (same as frozen); let
  `within_i` = number of a sample's `k−1 = 49` neighbours with signature distance
  `≤ ε_K`.
- **Admitted (dense-fiber center):** `within_i ≥ ADAPTIVE_K_MIN`.
- **Covered fraction:** `f = mean(admitted)` — this **replaces** the frozen
  `candidate_sample_fraction ≥ s_pos` coverage gate. `f` is a reported scope
  fence, never a pass/fail on its own except the sliver floor below.
- The twin certificate (state-insufficiency witnesses + paired action-disagree)
  is read on the **union of admitted fibers**: only pairs with both endpoints
  admitted and signature distance `≤ ε_K`.

**Frozen constants:**
- `ADAPTIVE_K_MIN = 10` (a fifth of the frozen `k = 50`; a dense fiber must have
  ≥ 10 genuinely-near neighbours, not a chance singleton).
- `ADAPTIVE_COVERED_FLOOR = 0.10` (`f < 0.10` ⇒ sliver — a resolvable core too
  small to carry a scoped claim).
- Witness-mass thresholds **unchanged** from frozen: `twin_min_witness_fraction
  = 0.01`, `twin_min_unique_pairs = 100`, `delta_h = max(1e-6, 0.05·median‖Q_K‖)`,
  paired-constancy threshold `delta_action = 0.10`. **Not loosened** — the
  certificate is at least as strict as frozen; only coverage is re-posed.

**Reduction to frozen on compact cells (why regression must pass):** where the
attractor is compact (`r_50 ≤ ε_K` for ~all samples, as at G=200/G=300), every
sample has `within_i = 49 ≥ 10`, so `admitted` = all, `f = 1.0`, the admitted
pair set equals the frozen candidate set, and the adaptive read is
*bit-identical* to frozen. This is a theorem about the construction; §3 confirms
it numerically.

## 2. Verdicts (adaptive namespace; never collide with frozen)

- `TWIN_STATE_ADAPTIVE_CERTIFIED` — `f ≥ floor`, witness mass met on admitted
  fibers (state-insufficient); pair with the paired read for the composed claim.
- `TWIN_STATE_ADAPTIVE_NO_CERTIFICATE` — `f ≥ floor`, witness mass not met.
- `TWIN_STATE_ADAPTIVE_SLIVER` — `f < floor`.
- `TWIN_STATE_ADAPTIVE_HIGH_MODE_FLOOR` — high-mode norms numerically flat.
- Paired read (control-sufficiency on admitted fibers):
  `PAIRED_FIBER_CONSTANCY_POSITIVE` iff witness action-disagree ≤ `delta_action`,
  else `..._NEG`, else `PAIRED_FIBER_UNDEFINED`.

The frozen verdict is computed in the **same pass** (via the untouched
aggregator) and emitted under `frozen_*` keys.

## 3. Regression gate (PRECONDITION — agent-run; R0)

Re-run G=200 and G=300 with `--adjudicator twin-state-adaptive` at the banked
N = 50k. **All must hold or the apparatus is rejected:**

1. `frozen_verdict` = `TWIN_STATE_CERTIFIED` on each cell (reproduces the banked
   certification; the adaptive adjudicator computes it in the same pass, so this
   is the frozen read on identical samples).
2. Adaptive verdict maps to frozen: `TWIN_STATE_ADAPTIVE_CERTIFIED` where frozen
   is `TWIN_STATE_CERTIFIED`.
3. `f ≥ 0.999` (compact ⇒ full admission — the reduction theorem numerically).
4. **Binding:** `|adaptive_disagree − frozen_disagree| ≤ 0.005` on the same run
   (on a compact cell the admitted pair set equals the frozen candidate set, so
   this should be ~0 exactly). **Informational (preset-confounded, not gated):**
   `|adaptive_disagree − banked_disagree|` against the banked 0.0367 / 0.0382,
   which came from `lock_v5` / `lock_v7_g300` — a different-objective reference,
   reported for context only.

Any failure ⇒ **`NSE-H3-APPARATUS-REJECTED`, final** — the apparatus is not the
same test on the cells where the test already worked; no G=675 read is taken.

## 4. Application (R1; owner-run, only if R0 passes)

Re-run the G=675 fallback cell (`fallback_v7_g675_kf3`, N = 200k) with
`--adjudicator twin-state-adaptive`. Branches (from
`NSE_COVERAGE_ADAPTIVE_APPARATUS_SCOPE.md` §6):

| R1 outcome | verdict |
| --- | --- |
| `f < 0.10` (`ADAPTIVE_SLIVER`) | `NSE-H3-COVERAGE-SLIVER` |
| `f ≥ 0.10`, `ADAPTIVE_CERTIFIED` + `PAIRED_..._POSITIVE` | `NSE-H3-FORCING-GENERAL` (scoped to `f`) |
| `f ≥ 0.10`, `ADAPTIVE_CERTIFIED` + `PAIRED_..._NEG` | `NSE-H3-GRASHOF-LOCAL` |
| `f ≥ 0.10`, `ADAPTIVE_NO_CERTIFICATE` | `NSE-H3-COVERAGE-WALL-CONFIRMED` |

No gate widening, no constant retune after any read; failures typed and final.

## 5. Harness change (additive; validated pre-R0)

New `--adjudicator twin-state-adaptive`: captures high modes (like `twin-state`),
computes the frozen read via the untouched `aggregate_twin_state` plus the
adaptive read in one pass, and persists the raw sample arrays
(`samples.npz`: signatures, high modes, actions, `ε_K`) so any *third* apparatus
iteration is pure post-processing. No existing adjudicator path changes. Gated by
the harness self-test + a non-verdict smoke before R0.

## 6. Does not claim

No promotion, no infinite-dimensional statement, no change to H1's proxy-relative
typing or any banked verdict. A reopened `FORCING-GENERAL` is scoped to a covered
fraction `f` of one matched-Re cell under a new apparatus — one point, one axis,
fenced. `docs/chatv2/` stays no-publish.

Cross-refs: `NSE_COVERAGE_ADAPTIVE_APPARATUS_SCOPE.md`,
`NSE_H3_ADMISSION_RECEIPT.md`, `PDE_C1_TWIN_STATE_CERTIFICATE.md`,
`PDE_C1_REGIME_GENERALITY_v1.md` (ε_K rule), `results/proof/c1-paired-fiber-g{200,300}/`
(banked regression comparators), `results/proof/c1-h3-kf3-g675-fb-twin/`
(the R1 target cell).

---

> **Post-run status (2026-07-07): R0 regression PASS** (`NSE_COVERAGE_ADAPTIVE_
> RECEIPT.md`). Both compact cells reduced to frozen **bit-identically**
> (f=1.0000, |adaptive−frozen| disagree 0.00000, pair sets 693,774 / 942,834
> identical to frozen; banked xref within 0.0001). The apparatus is the same
> test where the test already worked ⇒ **G=675 read unblocked** (not
> APPARATUS-REJECTED). R1 staged (owner-run): `fallback_v7_g675_kf3` with
> `--adjudicator twin-state-adaptive`, ~2 h.

> **FINAL (2026-07-07): R1 `TWIN_STATE_ADAPTIVE_SLIVER` ⇒ `NSE-H3-COVERAGE-SLIVER`.**
> Covered fraction f=0.036 (< 0.10 floor), dense-count median **0**, p95 8 (< k_min).
> The fidelity-preserving apparatus confirms the G=675 attractor lacks ε_K-dense
> fiber structure — the wall is a **geometric fact, not bookkeeping**. Forcing
> axis closed across four concordant probes. Anchor witness + closed slate
> untouched.
