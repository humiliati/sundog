# v0.3 Cross-m_3 Sentinel Pre-Registration (II)

Status: **VERDICT LANDED, 2026-05-22**.
The sentinel sweep executed against this pre-registration on 2026-05-22
(`17:17:40 - 20:40:58 UTC-7`, ~3h 23min). Joint verdict: **(Q1.D, Q2.D) =
gate pathology on both axes**, resolved by the sigma_3 symmetry probe
into a **domain-of-applicability finding**: all tested supplementary-B
piano-trios fail the `sigma_3` cycle; six of seven carry `F_beta = (12)`-swap
symmetry, while `O_434(0.4)` is smaller-symmetry. The tested set is therefore
`Z_2`-or-smaller, not `D_3`. The v0.3 `Gamma_i` mechanism is therefore
structurally inapplicable to this catalog. See the **Verdict (Landed)**
section below.
Audience: operator who executes the staged sentinel runs; future paper-side writer.
Companion: `kfacet_v03_freeze_b_comparison.md` (the alpha structural-null verdict
that motivates II). Methodology hand-off: `kfacet_v03h_writeup.md`.

## One-Line Read

`v0.3-alpha-freeze-b` closed the equal-mass strict-G.2 cell of a **2 x 2
design** as structural null: zero daughters predicted from 20 resolved
strict-G.2 rows against 273 observed piano-trios. II tests two of the
remaining three cells:

| | **strict G.2 family** | **piano-trio family** |
| :--- | :--- | :--- |
| **m_3 = 1**   | `alpha` verdict: structural null | II.Q2 control: 3 supp-B sentinels |
| **m_3 != 1**  | (empty; strict G.2 is m_3 = 1 only) | II.Q1 primary: 4 supp-B sentinels at m_3 = 0.4 |

The two slices answer two structurally separate questions:

- **Q1 (m_3 axis):** does `Gamma` wake up where supplementary-B is densest?
  m_3 = 0.4 carries 55 rows; the equal-mass null may be an artifact of
  the m_3 = 1 slice.
- **Q2 (family axis):** do piano-trio orbits themselves carry the
  standard-sector signal even though strict-G.2 seeds did not? Supp-B's
  38 rows at m_3 = 1 are **entirely disjoint** from the strict-G.2 21
  indices, so this is a fresh equal-mass test, not a redundant
  duplicate.

**No full-slice run is authorized inline**; the sentinel sweep is staged
as an operator command. Verdict is read as a 2 x 2 joint outcome over
(Q1, Q2), not as a single bundled pass/fail.

## Verdict (Landed, 2026-05-22)

The seven sentinels ran against this pre-registration on 2026-05-22. All
seven halted at the runner-stage `D3` gate with kernel-projected residuals
6 -- 8 orders above the `1e-3` relation floor:

```text
Row          period   kernel-projected D3 residual    omega gate
O_50(0.4)   105.33    5.05e+04                        FAIL
O_62(0.4)   110.39    1.22e+05                        FAIL
O_67(0.4)   111.25    1.27e+05                        FAIL
O_434(0.4)  200.66    5.92e+04                        FAIL
O_242(1.0)   84.10    1.74e+04                        FAIL
O_282(1.0)   92.06    6.15e+03                        FAIL
O_284(1.0)   92.20    1.79e+04                        FAIL

D3 relation floor:        1e-3
sweep wall time:          ~3h 23min  (17:17:40 - 20:40:58)
```

Per-axis outcomes:

```text
Q1 (m_3 = 0.4):       Q1.D  (gate pathology, 4 of 4)
Q2 (m_3 = 1 supp-B):  Q2.D  (gate pathology, 3 of 3)
Joint verdict:        (Q1.D, Q2.D)
```

The pre-registered D-outcome action was to **diagnose at the runner
level**. A targeted `sigma_3-scan` (`isotrophy:sigma3-scan` against
supp-B) on the same seven rows produced the structural diagnosis at the
catalog-closure level, separate from the variational-kernel `D3` gate:

```text
sigma_3 closure residuals (after SO(3) gauge minimization):

  Strict G.2 21 (m_3 = 1):     ~1e-9 to 3e-8    catalog admission residuals
  Piano-trio sentinels:         0.60 to 0.79     orbit-scale; ~7-9 orders above

  sigma_any_strict_single_curve_candidate_count: 0 / 7 piano-trio rows

F_beta closure residuals:
  Strict G.2 21:                ~1e-9
  Piano-trio sentinels (6/7):   8e-9 to 4e-8    clean, comparable to G.2
  Outlier O_434(0.4):           0.25            also broken
```

**Resolved verdict: domain-of-applicability mismatch.**
Supplementary-B piano-trio orbits carry the `F_beta = (12)`-swap symmetry
at integration precision (6 of 7 sentinels), but **do NOT carry the
`sigma_3 = (123)`-cycle symmetry at any phase or any spatial rotation**.
They sit in `Z_2`, not `D_3`.

The v0.3 `Gamma_i` mechanism is defined on `D_3`-symmetric orbits and is
therefore **structurally inapplicable to the piano-trio catalog**. The
sentinel gate failures are not subtle drifts -- they are the runner
detecting that its precondition (orbit is `D_3`-symmetric to integration
precision) is violated by 7 -- 9 orders of magnitude.

`O_434(0.4)` is flagged as a separate sub-investigation: it breaks
`F_beta` closure too (residual 0.25), suggesting an even smaller
symmetry class than the other six piano-trio sentinels.

### Implication for the v0.3 alpha verdict

The `alpha` verdict (`K_facet_v0.3h = 0` on the resolved m_3 = 1 strict
G.2 rows) **stands and is sharpened**:

```text
v0.3 Gamma mechanism:
  - Defined: rank gate on F_beta-even standard D_3 isotypic of ker(M_i - I)
  - Domain of applicability: D_3-symmetric orbits (the strict G.2 21)
  - Empirical result on domain: 20 of 21 structural zero; 1 quarantined (O_617)
  - Predicted daughter count: 0

Supplementary-B catalog:
  - 273 piano-trio orbits at varied m_3
  - Sentinel symmetry class verified: Z_2 for 6 of 7, smaller for O_434;
    NOT D_3
  - Lies outside the v0.3 prediction's domain of applicability

Mechanism vs catalog:
  Predicted-vs-observed (0 vs 273) is a domain-of-applicability mismatch,
  not a falsification of v0.3 within its domain. Any v0.4 prediction of
  piano-trios must start from the Z_2 symmetry class, not D_3, and explicitly
  track smaller-symmetry outliers.
```

### Subsequent action

Per the pre-registration's `(Q1.D, Q2.D)` action: HALT the cross-m_3
extension; do not stage the full 55-row `m_3 = 0.4` sweep. Open the v0.3
epilogue chapter recording the domain-of-applicability finding. Open a
v0.4 question on `Z_2`-symmetric mechanisms; this is a fresh chapter and
should be designed paper-side before any new audit chain is built.

### Receipts

```text
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4/O{50,62,67,434}/gate_receipt.json
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB/O{242,282,284}/gate_receipt.json
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4-adaptive-floor/manifest.json
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB-adaptive-floor/manifest.json
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4-bridge-audit/manifest.json
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB-bridge-audit/manifest.json
results/isotrophy/k-facet-v03-piano-symmetry-probe/m3eq0.4/{manifest.json, residuals.csv}
results/isotrophy/k-facet-v03-piano-symmetry-probe/m3eq1.0-suppB/{manifest.json, residuals.csv}
```

## What's Being Tested

For each sentinel row at `m_3 in {0.4, 1.0 supp-B}`:

- Run the full v0.3h audit chain (sentinel gates + adaptive-floor reprocessor
  + bridge audit) on the supplementary-B initial conditions, not on the
  strict G.2 seeds.
- Record `c_i` (standard-E multiplicity) and `d_i` (Gamma_i rank) per row.
- Pre-registered outcome: each row is either a structural zero
  (`E = 0`, `c_i = d_i = 0`), a positive-signal row (`c_i >= 1`,
  `d_i = rank`), or a quarantined row (bridge / leakage / gate failure).

### Two structurally separate questions

The seven sentinel rows split into two slices that test two different
questions. The verdict is **read per-slice**, then composed into a joint
outcome over the 2 x 2 design:

- **Q1 (m_3 axis):** Aggregate the 4 m_3 = 0.4 sentinels. Does any of
  them surface a standard-E sector or a non-trivial `Gamma_i` rank?
- **Q2 (family axis):** Aggregate the 3 m_3 = 1 supp-B sentinels. Does
  any of them surface a standard-E sector? This is a fresh equal-mass
  test on piano-trio orbits (disjoint from strict G.2).

Decision rule: stage the full 55-row m_3 = 0.4 sweep only if **Q1 is
null AND no surprise from Q2**. Any positive signal on either axis
halts the scaling and triggers a structural review tailored to which
axis produced the signal.

## Pre-Registered Slices

### Primary slice -- m_3 = 0.4

```text
55 rows total
Stability:    35 S / 20 U
Period range: T in [105.33, 200.66]
Index range:  [50, 434]
```

Sentinel subset (4 rows -- shortest-period focus, stability mix):

| Index | m_3 | Period   | Stability | z_0       | Rationale                         |
| ----: | ---:| --------:| :-------- | --------- | --------------------------------- |
| 50    | 0.4 | 105.3306 | S         | 2.128e-01 | Shortest period, stable           |
| 62    | 0.4 | 110.3876 | U         | 2.411e-01 | Short period, unstable            |
| 67    | 0.4 | 111.2539 | U         | 2.092e-01 | Short period, unstable, different z_0 |
| 434   | 0.4 | 200.6615 | S         | 1.356e-01 | Longest period, stable, sanity-check upper bound |

### Control slice -- m_3 = 1.0 supp-B

```text
38 rows total
Stability:    7 S / 31 U
Period range: T in [84.10, 202.30]
Index range:  [242, 1492]
```

**Critical: m_3=1 supp-B indices are entirely disjoint from the strict G.2
21 indices.** Verified by no-integration probe:

```text
m_3=1 strict G.2:    {62, 64, 231, 264, ..., 1497}   (21 rows)
m_3=1 supp-B:        {242, 282, 284, 337, ...}        (38 rows)
intersection:        empty
```

So the control slice is **not** a redundant test of the alpha verdict;
it is a fresh equal-mass test on a different family (piano-trio
initial conditions, not single-curve choreographies).

Sentinel subset (3 rows -- shortest-period focus, all unstable since
the slice skews 31U/7S):

| Index | m_3 | Period  | Stability | z_0       | Rationale                  |
| ----: | ---:| -------:| :-------- | --------- | -------------------------- |
| 242   | 1.0 | 84.1038 | U         | 1.387e-01 | Shortest period            |
| 282   | 1.0 | 92.0571 | U         | 1.861e-01 | Short, different z_0       |
| 284   | 1.0 | 92.2039 | U         | 3.089e-01 | Short, larger z_0          |

## Compute Discipline

- **Inline rule:** anything expected over ~10 minutes wall time is staged,
  not run by the inline agent session. The sentinel-runner takes
  20-45 min per row at rtol=1e-12; therefore none of the runs below
  execute inline.
- Per-row wall-time estimate (from v0.3h calibrated sweep):
  - T ~ 84-110:  ~22-28 min
  - T ~ 150:     ~30-35 min
  - T ~ 200:     ~40-45 min
- Per-slice budgets:
  - m_3=0.4 sentinels (4 rows, T ~ 105-200): **~120-150 min**
  - m_3=1.0 sentinels (3 rows, T ~ 84-92):   **~70-90 min**
- Total sentinel budget: **~3-4 hours** of operator-driven compute.

## Verification Step (cheap, inline-safe)

Before staging the sentinel runs, the operator should confirm the parse
is consistent:

```powershell
npm run isotrophy:parse:b
```

Expected: 273 rows parsed. Then spot-check the sentinel indices exist
with the expected `(m_3, T)` shown in the tables above. If any sentinel
row is missing or has a different period, halt and reconcile before
running.

## PowerShell Run Commands (Staged)

The runs below MUST be executed by the operator outside the inline agent
session. Each command is the existing v0.3h sentinel pipeline with
`--source B` and the supp-B path swapped in.

### m_3 = 0.4 sentinels

```powershell
# O_50(0.4) -- shortest, stable
python scripts/isotrophy_workbench.py kfacet-sentinel `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 0.4 --sentinel-index 50 `
  --rtol 1e-12 --atol 1e-12 --closure-floor 1e-7 `
  --verify-partial-eps --fd-h 1e-6 --fd-floor 1e-4 --joint-baseline-floor 1e-8 `
  --authorize-sentinel-run --k-gamma 3 --k-int 10 --gamma-projector-floor 1e-3 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4/O50

# O_62(0.4) -- short, unstable
python scripts/isotrophy_workbench.py kfacet-sentinel `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 0.4 --sentinel-index 62 `
  --rtol 1e-12 --atol 1e-12 --closure-floor 1e-7 `
  --verify-partial-eps --fd-h 1e-6 --fd-floor 1e-4 --joint-baseline-floor 1e-8 `
  --authorize-sentinel-run --k-gamma 3 --k-int 10 --gamma-projector-floor 1e-3 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4/O62

# O_67(0.4) -- short, unstable, different z_0
python scripts/isotrophy_workbench.py kfacet-sentinel `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 0.4 --sentinel-index 67 `
  --rtol 1e-12 --atol 1e-12 --closure-floor 1e-7 `
  --verify-partial-eps --fd-h 1e-6 --fd-floor 1e-4 --joint-baseline-floor 1e-8 `
  --authorize-sentinel-run --k-gamma 3 --k-int 10 --gamma-projector-floor 1e-3 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4/O67

# O_434(0.4) -- longest, stable, upper-bound sanity check
python scripts/isotrophy_workbench.py kfacet-sentinel `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 0.4 --sentinel-index 434 `
  --rtol 1e-12 --atol 1e-12 --closure-floor 1e-7 `
  --verify-partial-eps --fd-h 1e-6 --fd-floor 1e-4 --joint-baseline-floor 1e-8 `
  --authorize-sentinel-run --k-gamma 3 --k-int 10 --gamma-projector-floor 1e-3 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4/O434
```

### m_3 = 1.0 supp-B sentinels

```powershell
# O_242(1.0) -- shortest period in supp-B m_3=1
python scripts/isotrophy_workbench.py kfacet-sentinel `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 1.0 --sentinel-index 242 `
  --rtol 1e-12 --atol 1e-12 --closure-floor 1e-7 `
  --verify-partial-eps --fd-h 1e-6 --fd-floor 1e-4 --joint-baseline-floor 1e-8 `
  --authorize-sentinel-run --k-gamma 3 --k-int 10 --gamma-projector-floor 1e-3 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB/O242

# O_282(1.0)
python scripts/isotrophy_workbench.py kfacet-sentinel `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 1.0 --sentinel-index 282 `
  --rtol 1e-12 --atol 1e-12 --closure-floor 1e-7 `
  --verify-partial-eps --fd-h 1e-6 --fd-floor 1e-4 --joint-baseline-floor 1e-8 `
  --authorize-sentinel-run --k-gamma 3 --k-int 10 --gamma-projector-floor 1e-3 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB/O282

# O_284(1.0)
python scripts/isotrophy_workbench.py kfacet-sentinel `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 1.0 --sentinel-index 284 `
  --rtol 1e-12 --atol 1e-12 --closure-floor 1e-7 `
  --verify-partial-eps --fd-h 1e-6 --fd-floor 1e-4 --joint-baseline-floor 1e-8 `
  --authorize-sentinel-run --k-gamma 3 --k-int 10 --gamma-projector-floor 1e-3 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB/O284
```

### After-sentinels reprocessor + bridge audit

```powershell
# Adaptive-floor reprocessor over the 7 sentinel rows
python scripts/isotrophy_workbench.py kfacet-reprocess-floor `
  --input-dir results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4-adaptive-floor `
  --allow-failed-rows
python scripts/isotrophy_workbench.py kfacet-reprocess-floor `
  --input-dir results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB-adaptive-floor `
  --allow-failed-rows

# Bridge audit (auto-targets failed_rows from manifest)
python scripts/isotrophy_workbench.py kfacet-bridge-audit `
  --input-dir results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4-adaptive-floor `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 0.4 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq0.4-bridge-audit
python scripts/isotrophy_workbench.py kfacet-bridge-audit `
  --input-dir results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB-adaptive-floor `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 1.0 `
  --out results/isotrophy/k-facet-v03-gamma-crossm3/m3eq1.0-suppB-bridge-audit
```

## Pre-Registered Decision Gates

After the sentinel runs complete, read the seven per-row gate receipts
and the two adaptive-floor manifests. Verdict is read in two passes:
**per-axis outcome first, joint verdict second**.

### Per-axis outcome alphabet

Each slice produces a single per-axis outcome over `{Q.A, Q.B, Q.C, Q.D}`:

- **Q.A (null):** every sentinel in the slice has gates pass,
  `c_i = 0`, `d_i = 0`, reprocessor outcome `adaptive_floor_resolved`,
  `E = 0` in the D3 isotypic readout.
- **Q.B (signal):** at least one sentinel surfaces a standard-E
  sector (`c_i >= 1`) or a nonzero `Gamma_i` rank (`d_i >= 1`) after
  the adaptive floor stabilizes.
- **Q.C (quarantine):** at least one sentinel enters the bridge band
  or fails the F_beta leakage gate, and no positive `Gamma` signal is
  recorded on the remaining sentinels. Per-row WHY-dive required before
  the slice promotes to A or B.
- **Q.D (gate pathology):** partial-epsilon FD residual fails relative
  tolerance, or D3 leakage exceeds `1e-3` for reasons unrelated to a
  bridge SV (e.g. typed sigma_3 construction breaks at this m_3
  because the supp-B orbit satisfies sigma_3 closure only loosely).
  Per-row diagnosis at the runner level; may indicate the v0.3h
  discipline needs tightening before the audit chain applies to
  supp-B initial conditions.

Apply this alphabet to both slices independently:

- `Q1` outcome = per-axis outcome over the 4 m_3 = 0.4 sentinels.
- `Q2` outcome = per-axis outcome over the 3 m_3 = 1 supp-B sentinels.

### Joint verdict table

The four main scientific cases of `(Q1, Q2)`. C and D outcomes resolve
into A or B per-row via the WHY-dive / runner diagnosis before joint
reading; the joint table is the verdict over the **resolved** (Q1, Q2)
pair.

| Q1 (m_3 axis) | Q2 (family axis) | Joint interpretation                                     | Action                                                |
| :------------ | :--------------- | :------------------------------------------------------- | :---------------------------------------------------- |
| `Q1.A` (null) | `Q2.A` (null)    | Mechanism is null everywhere across the tested cells.    | Stage full 55-row m_3 = 0.4 sweep as a follow-on. If also null, move to (I) v0.3 epilogue. |
| `Q1.A` (null) | `Q2.B` (signal)  | Family-dependent: piano-trios carry the signal that      | HALT scaling. Open a family-axis structural review:    |
|               |                  | G.2 seeds did not. m_3 is not the relevant variable.     | what distinguishes piano-trio orbits from G.2 strict? |
| `Q1.B` (signal) | `Q2.A` (null)  | m_3-dependent: standard-E sector wakes up off equal      | HALT scaling. Open an m_3-axis structural review:     |
|               |                  | mass; family identity does not matter.                   | does the F_beta cocycle change off equal mass?         |
| `Q1.B` (signal) | `Q2.B` (signal)| Mechanism wakes up everywhere except the G.2-strict-at-  | HALT scaling. The alpha structural null may be an    |
|               |                  | m_3 = 1 corner specifically. The alpha null is a corner. | artifact of the strict G.2 m_3 = 1 cell's rigidity.   |

Mixed cases (one axis A or B, the other C or D) reduce to one of the
four after per-row diagnosis. If diagnosis cannot resolve a slice
within the existing v0.3h discipline (e.g. a new structural class
that does not fit `bridge_approx_sign_isotypic`,
`bridge_approx_trivial_isotypic`, or related categories), open a
methodology review before reading the joint verdict.

### Subsequent action by joint outcome

- **(A, A)** -> stage full m_3 = 0.4 sweep. If also null, write the
  v0.3 epilogue chapter and close the trunk. Each of the three
  positive cases below opens its own v0.3-extension or v0.4 design.
- **(A, B)** -> family-dependent mechanism review. The G.2 strict 21
  may be a particularly rigid sub-family; piano-trios may carry the
  signal through a different `D3` representation, a `D_2` subgroup,
  or a non-dihedral mechanism.
- **(B, A)** -> m_3-dependent mechanism review. The standard-E sector
  may be a mass-perturbation effect. The F_beta cocycle off equal
  mass may not be the same as the equal-mass one used in v0.3.
- **(B, B)** -> joint review. The alpha verdict is then properly
  understood as "the strict G.2 m_3 = 1 cell is uniquely rigid", and
  the broader catalog is the structural object. Significant redesign
  before scaling.

## Out of Scope for II

- Full 55-row m_3=0.4 sweep (staged only after sentinel verdict).
- Other m_3 sub-catalogs (m_3 in {0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2,
  ...}). These are subsequent II-extensions, prioritized if the
  sentinel verdict warrants.
- The typed transport lemma derivation (III). That's paper-side
  rigor, not blocked by II.
- O_617 m_3-perturbation (delta). Independent sub-investigation.

## What This Pre-Registration Does NOT Prejudice

- The alpha verdict (`K_facet_v0.3h = 0` on the resolved m_3=1
  strict-G.2 rows) stands. II is testing whether the verdict
  generalizes, not retracting it.
- If sentinels show signal (Outcome B), it does not retroactively
  invalidate alpha; it indicates the mechanism is m_3-dependent. The
  alpha-frozen prediction is correct for m_3=1 G.2 strict rows.
- If sentinels are null (Outcome A) and the full m_3=0.4 sweep is
  also null, II strengthens alpha by extending its scope, but does
  not change the v0.3h methodology surface.

## Receipt Schema Expected from Sentinel Runs

Per row (mirrors v0.3h sentinel runs):

```text
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq{m}/O{idx}/
  gate_receipt.json
  M_i.npy
  K_fib_basis.npy
  omega.npy
  D3_*.npy
  partial_eps_M_i.npy
  partial_eps_M_i_fd.npy
  Gamma_i.npy
  Gamma_xi_basis.npy
```

Aggregate manifest from reprocessor:

```text
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq{m}-adaptive-floor/manifest.json
  summary.outcome_counts -> resolved / failed / suspicious
  rows[].selected_floor -> per-row selected floor
  rows[].selected_D3_isotypic_summary -> T(_)+S(_)+E(_)+c_i
```

Aggregate manifest from bridge audit:

```text
results/isotrophy/k-facet-v03-gamma-crossm3/m3eq{m}-bridge-audit/manifest.json
  summary.outcome_counts -> no_bridge_present / defective_E_block / ...
  rows[].outcome -> per-row bridge disposition
```

## Sequencing After II

The roadmap path with both `alpha` and II completed, parameterized by
the joint `(Q1, Q2)` verdict:

```
alpha (DONE):  K_facet_v0.3h structural-null verdict (strict G.2, m_3=1 cell)
II (THIS):     cross-m_3 + cross-family sentinel + joint verdict
  --> (A, A) "everywhere null":     stage full m_3=0.4 sweep
                                    --> if also null, move to (I) v0.3 epilogue
  --> (A, B) "family-dependent":    halt; open family-axis review
                                    --> piano-trios may use a different mechanism
  --> (B, A) "m_3-dependent":       halt; open m_3-axis review
                                    --> F_beta cocycle off equal mass differs
  --> (B, B) "G.2 corner is unique": halt; open joint review
                                    --> alpha may be a rigidity artifact
III (deferred): typed transport lemma paper-side rigor
delta (low):    O_617 m_3-perturbation follow-up
```

The eventual write-up shape (paper-side) depends on II's joint verdict:

- **(A, A)** -> v0.3 epilogue is a clean structural-negative result
  across the 2 x 2 design: methodology + audit chain + null verdict
  across both axes + named quarantine (O_617). Cleanest publication
  path; closes v0.3 as a falsifiable negative.
- **(A, B)** -> v0.3 epilogue + open v0.4 chapter on the
  family-dependent mechanism. The piano-trio family carries
  standard-sector content that strict G.2 does not, even at equal
  mass; the next mechanism candidate must explain that asymmetry.
- **(B, A)** -> v0.3 epilogue + open v0.4 chapter on the
  m_3-dependent mechanism. The F_beta cocycle or the relevant
  isotypic decomposition changes off equal mass; v0.4 must include
  the mass dependence.
- **(B, B)** -> v0.3 needs a substantial methodology revision. The
  alpha null is then understood as a corner phenomenon, not a generic
  fact about the catalog. Significant redesign before any paper-side
  claim.

## How to Verify This Pre-Registration is Still Current

Re-run the verification step:

```powershell
npm run isotrophy:parse:b
```

Expected: still 273 rows; sentinel indices still match the tables
above. If supp-B has been updated upstream and the row counts or
periods drift, this pre-registration should be re-issued before any
sentinel run.
