# Regime-Conditioned ε_K (Approach B) — Scope

> 2026-07-07. The shelved escalation from
> `NSE_COVERAGE_ADAPTIVE_APPARATUS_SCOPE.md`, now scoped on its own. **Approach A
> answered its question with a null: at *absolute* resolution the G=675 attractor
> has no ε_K-dense fibers (`NSE-H3-COVERAGE-SLIVER`, dense-count median 0).**
> Approach B asks a *different* question — is the proxy control-sufficient at
> **relative (scale-invariant) resolution**? A wide attractor (energy range
> 3.7:1 at G=675) can be self-similar: fiber structure at every scale even when
> no single absolute radius resolves it. B is that test, not a rescue of A.
> Non-promotional; a B-positive is scoped to relative resolution and does **not**
> overturn A's absolute-resolution null.

## 0. The honesty cost, stated up front

B **changes the fidelity criterion** — the one move in this lane that can
manufacture a false positive. Two consequences, both flagged before any number:

1. **B cannot claim A's bit-identical reduction.** A left `ε_K` untouched, so it
   was provably frozen on compact cells. B's per-sample `ε_K(u)` already varies
   on the G=300 anchor (energy range 1.15:1 ⇒ radius 0.0620–0.0664 vs frozen
   0.0664). So B reduces to frozen only *approximately*; its regression gate is
   **verdict-preservation within tolerance**, not bit-identity. This is a real
   weakening versus A, and it is why B needs a second guard.
2. **Inflation risk.** At G=675 `ε_K(u)` swings 0.043–0.083 (nearly 2× frozen in
   high-energy regions). A bigger radius there could sweep in genuinely-different
   states as "fiber neighbours" and manufacture a low disagree. §4's
   scale-consistency guard is the anti-inflation test that makes B admissible.

## 1. The design

Replace the global `ε_K = 0.05·√(2·e_max)` (e_max = objective threshold, a single
scalar) with a **per-sample** radius keyed to the local signature energy:

```text
ε_K(u) = 0.05·√(2·E_low(u))          E_low(u) = ‖Φ_K(u)‖²  (the signature energy)
pair (i,j) is a fiber pair iff  ‖Φ_i − Φ_j‖ ≤ 0.05·√(2·E_pair),  E_pair = ½(E_i+E_j)
```

Same "5% of the local RMS signature amplitude" the frozen rule uses, localized
per-sample instead of pinned to the global objective scale. Pair-mean energy is
the symmetric scale (not max — max inflates; not min — min over-shrinks at
boundaries). Measured radii (banked manifests):

| cell | E_low range | frozen ε_K | relative ε_K(u) range |
| --- | --- | --- | --- |
| G=200 anchor | [0.715, 0.735] (1.03:1) | 0.0606 | 0.0598–0.0606 (≈ frozen) |
| G=300 anchor | [0.768, 0.882] (1.15:1) | 0.0664 | 0.0620–0.0664 (mild) |
| G=675 matched-Re | [0.369, 1.370] (3.72:1) | 0.0589 | **0.0429–0.0828** (2×) |

Neighbour query: frozen k = 50 NN (raw Φ space); admit pairs passing the
per-sample radius test. If a high-energy sample's radius exceeds its 50th-NN
distance the radius is query-clipped — reported as a diagnostic; if pervasive, a
larger-k rerun is a registered follow-up, not a silent change.

## 2. Certificate + covered fraction

Read the twin-state certificate (state-insufficiency witnesses + paired
action-disagree) on the relative-fiber pairs, with the same **unchanged**
witness-mass thresholds as frozen/A (`twin_min_witness_fraction = 0.01`,
`twin_min_unique_pairs = 100`, `delta_h` by the same rule, paired threshold
`delta_action = 0.10`). Report the covered fraction `f_rel` (samples with ≥ 1
relative-fiber pair) as a scope fence, floor 0.10 as in A.

## 3. Guard 1 — regression (agent-run; R0)

Re-run G=200 and G=300 with the relative adjudicator (N = 50k). Required:

1. verdict reproduces banked `TWIN_STATE_CERTIFIED` on both cells;
2. `f_rel ≥ 0.98` (near-full admission — approximate reduction);
3. `|relative_disagree − banked_disagree| ≤ 0.005` (banked 0.0367 / 0.0382);
4. **scale-consistency (Guard 2) does not spuriously fire** on the compact cells
   (energy near-constant ⇒ one effective tercile ⇒ trivially consistent).

Any failure ⇒ `NSE-H3-APPARATUS-REJECTED-B`, final — the relative criterion does
not preserve the certified reading where the test already worked.

## 4. Guard 2 — scale-consistency (the anti-inflation test; NEW, load-bearing)

The guard A never needed. Stratify admitted witness pairs into **energy terciles**
by `E_pair`; compute the paired action-disagree fraction per tercile. Require:

```text
max_tercile(disagree) − min_tercile(disagree) ≤ CONSISTENCY_TOL = 0.03
```

Fiber-constancy must hold **uniformly across scales**. If disagree inflates in the
high-energy tercile (where ε_K(u) is largest), the relative resolution is smearing
different states together there — the coverage is false fiber-membership, and any
aggregate "positive" is an artifact ⇒ `NSE-H3-APPARATUS-REJECTED-B` (inflation).
A B-positive is admissible **only** if it is scale-consistent.

## 5. R1 branches (owner-run, N = 200k, only if R0 passes)

| R1 outcome | verdict |
| --- | --- |
| `f_rel < 0.10` | `NSE-H3-COVERAGE-SLIVER-RELATIVE` (even relative resolution finds no fibers) |
| covered, CERTIFIED, paired POSITIVE, **scale-consistent** | `NSE-H3-FORCING-GENERAL-RELATIVE` (scoped to relative resolution + `f_rel`) |
| covered, CERTIFIED, paired POSITIVE, **scale-inconsistent** | `NSE-H3-APPARATUS-REJECTED-B` (relative resolution smears; positive is an artifact) |
| covered, CERTIFIED, paired NEG, scale-consistent | `NSE-H3-GRASHOF-LOCAL` |
| covered, `NO_CERTIFICATE` | `NSE-H3-COVERAGE-WALL-CONFIRMED-RELATIVE` |

## 6. Honest prior + claim boundary

A already showed the fibers aren't there at absolute scale; B bets that the 3.7:1
energy spread hides scale-invariant structure. Plausible but not favoured — the
likely outcomes are a scale-consistent relative positive (genuinely interesting:
the witness is scale-invariant, not absolute), an inflation rejection (the relative
radius was smearing), or a relative sliver (structureless at every scale). **A
B-positive claims only: control-sufficient at relative/scale-invariant resolution
on a fraction `f_rel` of one matched-Re cell.** It is strictly weaker than an
absolute-resolution positive, does not overturn A's null, does not touch H1/H2 or
the closed slate, and is not promotion or any infinite-dimensional statement.

## 7. The one owner decision

Additive harness sign-off: a new `--adjudicator twin-state-relative` (or a
`--relative-eps` mode on the adaptive family) that computes the frozen read (via
the untouched aggregator) + the relative read + the tercile scale-consistency
table in one pass, and persists `samples.npz` (gitignored). Additive-only; no
existing adjudicator path changes; validated by the harness self-test + a
synthetic apparatus self-test (reduction-on-compact, inflation-detection) + a
non-verdict smoke before R0. On sign-off I freeze the spec (the `ε_K(u)` rule,
`CONSISTENCY_TOL`, floors, regression tolerances), build it, run the synthetic
self-test and R0 agent-side — B earns its G=675 read or is rejected without owner
compute.

Cross-refs: `NSE_COVERAGE_ADAPTIVE_APPARATUS_SCOPE.md` / `..._SPEC.md` /
`..._RECEIPT.md` (Approach A, the absolute-resolution null),
`NSE_H3_ADMISSION_RECEIPT.md`, `PDE_C1_REGIME_GENERALITY_v1.md` (the ε_K rule B
localizes), `results/proof/c1-adaptive-reg-g{200,300}/` (regression comparators),
`results/proof/c1-h3-kf3-g675-adaptive/` (the A null this re-reads at relative scale).

---

## 8. Build status (2026-07-07): built + synthetic-validated; R0 ready

Additive `--adjudicator twin-state-relative` built (frozen read via the untouched
aggregator + relative read + tercile scale-consistency table in one pass;
`samples.npz` gitignored). **Frozen constants** (in `pde_c1_kolmogorov_cell.py`):
`RELATIVE_CONSISTENCY_TOL = 0.03`, `RELATIVE_COVERED_FLOOR = 0.10`,
`RELATIVE_MIN_TERCILE_PAIRS = 30`; per-sample `ε_K(u) = 0.05·√(2·E_low(u))`, pair
radius on mean energy. Harness self-test passed; smoke passed.

**Synthetic self-test 3/3** (`scripts/nse_coverage_adaptive_regression.py
--self-test-b`), validating both load-bearing properties:
- **B-T1 reduction:** compact cell at E≈0.72 ⇒ relative bit-matches frozen
  (f_rel=1.000, disagree 0.3984 == 0.3984).
- **B-T2 inflation caught (the guard works):** low-E true fibers (disagree 0.0) +
  high-E false fibers (disagree 0.49) ⇒ tercile spread 0.49 > 0.03 ⇒
  `scale_consistent=False`. The anti-inflation test fires exactly as designed.
- **B-T3 self-similar positive:** constant action per cluster ⇒ all terciles
  disagree 0.0 ⇒ `scale_consistent=True`.

Next: **R0 regression** (agent-run, ~80 min) — `lock_v7_g200` + `lock_v7_g300`
with `--adjudicator twin-state-relative`, then `--evaluate` per §3. R0 PASS
unblocks the owner-run R1 (G=675, N=200k); FAIL ⇒ `NSE-H3-APPARATUS-REJECTED-B`.

> **Post-run status (2026-07-07): R0 FAIL ⇒ `NSE-H3-APPARATUS-REJECTED-B`**
> (`NSE_REGIME_CONDITIONED_EPS_RECEIPT.md`). Reduction flawless (relative
> bit-identical to frozen on both cells, f_rel=1.0), but the scale-consistency
> guard **fires on the certified anchor** (spread 0.063/0.052 > 0.03): a true
> positive's witness has natural U-shaped energy structure (~0.06 spread) above
> the tolerance, and raw max−min can't distinguish it from monotonic inflation.
> The gate worked — caught the defect on the control, no G=675 read taken, no
> re-tune. Owner fork (receipt): accept final, or commission a v2 guard
> (monotonic/anchor-calibrated, new registration).
