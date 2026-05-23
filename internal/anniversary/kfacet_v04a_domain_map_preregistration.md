# v0.4a Domain Map Pre-Registration

Status: pre-registered, not yet run. 2026-05-22.
Audience: operator who executes the staged sweep; v0.4b paper-side writer;
future coding agents.
Companions:
- `kfacet_v04a0_o434_anatomy.md` (methodological lesson that motivates the
  two-pass design; the run receipt lives at
  `results/isotrophy/k-facet-v04a0-o434-anatomy/anatomy_receipt.json`).
- `kfacet_v03h_writeup.md` (v0.3 closure + projection-language framing).
- `kfacet_v03_gamma_crossm3_preregistration.md` (II sentinel + symmetry
  probe that produced the original O_434 misclassification).
- `docs/threebody/CROSS_SUBSTRATE_NOTES.md` §6-§7 (projection vocabulary).

## One-Line Read

v0.4a builds the **domain map** of the supplementary-B piano-trio body
under the `Z_2 = (12)`-swap projection. Each of the 273 catalog rows
gets classified into one of four bands (`Z2_clean / marginal_Z2 /
smaller_symmetry / undefined`) by a **two-pass gauge classifier with
no per-row knobs**: a coarse Pass 1 over the full catalog, then a
tight Pass 2 on every row that does not land cleanly in `Z2_clean` at
Pass 1. The deliverable is a single table
`table[m_3][stability][class]` that serves as v0.4b's audit ground
truth. v0.4a is **not a mechanism test**; it audits the projection's
well-definedness so that v0.4b's mechanism prediction (m_3 / stability
stratification, registered separately) can be compared against an
honest classifier rather than against a self-referential audit.

## What's Being Tested

In projection language (`CROSS_SUBSTRATE_NOTES.md` §6.3):

```text
Body:        supplementary-B piano-trio orbit as a primary Z_2 object.
             (not a daughter of strict S_3; not a v0.3 shadow.)

Projection:  orbit -> Z_2 symmetry-class shadow.
             The shadow is the orbit's gauge-minimized F_beta closure
             residual after SO(3) phase + rotation optimization, plus
             the band assignment in {Z2_clean, marginal_Z2,
             smaller_symmetry, undefined}.

Well-definedness gate: the projection's image is a deterministic
             function of (orbit IC, period). The two-pass classifier
             below pre-registers that determinism by removing per-row
             tolerance choices.

Output observable: table[m_3][stability][class] over the 273 rows.

Rigidity check: the classification at Pass 2 (tight tolerances) is
             stable across the sentinel-to-full transition. Sentinel
             rows whose classification changes between the II
             7-sentinel probe and the v0.4a full sweep get flagged.
```

The v0.4a deliverable is **the table**, not a mechanism. A separate v0.4b
pre-registration will register an independent predictor (m_3/stability
stratification per Codex anchor `gamma`) and compare it against the
table.

## Pre-Registered Scope

```text
Total rows:         273
m_3 values (15):    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                    1.4, 1.5, 1.6, 1.7, 1.9
Per-m_3 row counts:
  0.4:  55    0.5:  31    0.6:  22    0.7:  18    0.8:  18
  0.9:  25    1.0:  38    1.1:  23    1.2:   7    1.3:   4
  1.4:   4    1.5:   8    1.6:   8    1.7:  10    1.9:   2
Stability:          97 S / 176 U
```

## The Two-Pass Gauge Classifier (Pre-Registered)

The methodological lesson from the O_434 anatomy probe was that the
default sigma_3-scan tolerances (`identity_rotation_tolerance = 1e-6`,
`phase_grid = 73`) misclassified one row by SIX orders of magnitude.
The two-pass design encodes that lesson as a deterministic procedure.

### Pass 1 (coarse, full catalog)

```text
For each of the 273 rows:
  sigma3-scan with:
    --identity-rotation-tolerance  1e-6
    --phase-grid                   73
    --n-samples                    1009
    --rtol  1e-12
    --atol  1e-12
    --sigma-tolerance              1e-5
    --sigma-closure-multiple       3
```

Pass 1 emits a per-m_3 manifest + `residuals.csv` covering every row in
that m_3 slice. The Pass 1 classification is the four-band assignment
read from each row's `F_beta_residual_inf`.

### Pass 2 (tight, conditional)

```text
For every row whose Pass 1 classification is NOT Z2_clean
(i.e., F_beta_residual_inf > 1e-4 OR undefined / integration failure):
  sigma3-scan with:
    --identity-rotation-tolerance  1e-9
    --phase-grid                   361
    (all other settings identical to Pass 1)
```

The final classification for each row is:
- If Pass 1 already classifies it as `Z2_clean`, that is the final
  classification. Pass 2 is not invoked.
- Otherwise, Pass 2's `F_beta_residual_inf` produces the final band
  assignment.

### Four-band classifier

```text
Z2_clean:          F_beta_residual_inf <= 1e-4
marginal_Z2:       1e-4 < F_beta_residual_inf <= 1e-2
smaller_symmetry:  1e-2 < F_beta_residual_inf <= 1
undefined:         F_beta_residual_inf > 1, integration failure,
                   or sigma_3-scan precondition violated
```

**Important wording:** *No `smaller_symmetry` rows are currently confirmed;
the band exists to prevent silent coercion if the full sweep finds one.*
The O_434 anatomy probe was the only row tested at tight tolerances and
it landed at `1.06e-7` (well inside `Z2_clean`); the band is therefore
empty-allowed under the pre-registration. If Pass 2 populates it,
v0.4b must revise to track those rows explicitly as a separate
theoretical concern.

No per-row knobs. No tolerance dialing. The only two thresholds are
the pre-registered Pass 1 / Pass 2 tolerances and the four band edges
above.

## Compute Discipline

```text
Pass 1 estimate:    ~17 sec / row * 273 = ~77 min staged
Pass 2 estimate:    ~85 sec / row * N_flagged
                     N_flagged worst case: bounded above by 273 - rows
                     that Pass 1 already places in Z2_clean. From the
                     7-row sentinel evidence we expect most rows pass
                     at default tolerances; a typical N_flagged budget
                     is 5-30, giving Pass 2 ~7-42 min.
Aggregator + classification: seconds.

Total expected:     ~85-120 min staged, all operator-driven.
```

Above the inline ~10-minute rule. Stage as operator-driven commands.

## PowerShell Run Commands (Staged)

### Pass 1: 15 commands, one per m_3

```powershell
$m3_values = @("0.4","0.5","0.6","0.7","0.8","0.9","1.0","1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.9")
foreach ($m3 in $m3_values) {
  python scripts/isotrophy_workbench.py sigma3-scan `
    --source B `
    --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
    --m3 $m3 `
    --limit 0 `
    --n-samples 1009 `
    --phase-grid 73 `
    --rtol 1e-12 `
    --atol 1e-12 `
    --sigma-tolerance 1e-5 `
    --sigma-closure-multiple 3 `
    --identity-rotation-tolerance 1e-6 `
    --out "results/isotrophy/k-facet-v04a-domain-map/pass1/m3eq$($m3)"
}
```

### Aggregator (paper-side): identify Pass 2 candidates

After Pass 1 finishes, run a small classifier script (write at v0.4a
implementation time; spec is in **Aggregator schema** below) to:

1. Load every Pass 1 `residuals.csv` across all 15 m_3 values.
2. Apply the four-band classifier on `F_beta_residual_inf`.
3. Emit a list of `(m_3, index)` rows that landed outside `Z2_clean`.
4. Write a draft `table[m_3][stability][class]` reflecting Pass 1
   provisional bands.

The aggregator output is a single JSON file at
`results/isotrophy/k-facet-v04a-domain-map/pass1/aggregator_manifest.json`.

### Pass 2: conditional re-runs

For each `(m_3, index)` in the aggregator's flagged set:

```powershell
python scripts/isotrophy_workbench.py sigma3-scan `
  --source B `
  --path docs/isotrophy/supplementary-B_piano-init-condit-3d.txt `
  --m3 $m3 `
  --indices $idx `
  --n-samples 1009 `
  --phase-grid 361 `
  --rtol 1e-12 `
  --atol 1e-12 `
  --sigma-tolerance 1e-5 `
  --sigma-closure-multiple 3 `
  --identity-rotation-tolerance 1e-9 `
  --out "results/isotrophy/k-facet-v04a-domain-map/pass2/m3eq$($m3)/O$($idx)"
```

### Final aggregator: emit the table

After Pass 2 finishes, re-run the aggregator with the Pass 2 receipts
overriding the Pass 1 bands for flagged rows. The aggregator emits:

```text
results/isotrophy/k-facet-v04a-domain-map/manifest.json
  - mode: kfacet_v04a_domain_map
  - version: v0.4a-domain-map
  - thresholds: {Z2_clean, marginal_Z2, smaller_symmetry, undefined}
  - pass1_tolerances + pass2_tolerances
  - per_row_table: 273 entries with final_class + provenance (pass1 / pass2)
  - table_m3_by_stability_by_class: nested counts
  - summary:
      total_rows
      class_counts: {Z2_clean: ?, marginal_Z2: ?, smaller_symmetry: ?, undefined: ?}
      pass2_invoked_count
      pass2_reclassification_count: rows where Pass 2 changed the band
```

## Pre-Registered Decision Gates

The v0.4a verdict is the final `manifest.json` table. Three structurally
distinct outcomes:

### Outcome A: `Z2_clean` is the only populated band

All 273 rows land in `Z2_clean` (possibly through Pass 2 rescue).
Verdict: the projection is well-defined and uniform across the catalog;
the v0.3 II symmetry probe's near-universal Z_2 finding scales to 273.
v0.4b is then the mechanism test on a clean Z_2-only catalog.

### Outcome B: marginal_Z2 band is populated; smaller_symmetry / undefined are empty

Some rows land in `marginal_Z2` after Pass 2, but none populate the
smaller or undefined bands. Verdict: the projection is well-defined but
not uniform; v0.4b's predictor must explain the marginal/clean split.

### Outcome C: smaller_symmetry and/or undefined are populated after Pass 2

Verdict: the projection has structural exceptions. v0.4b must revise to
treat those rows as a separate theoretical category (the "outlier lane
as receipt category" stays as a placeholder until v0.4b decides whether
they need a second mechanism or whether they are catalog edge cases).

A failure to converge in Pass 2 (`undefined` rows) is also a methodology
finding: the catalog admission tolerance for supplementary-B may be looser
than the sigma_3-scan can resolve. This would push back into a
catalog-reconstruction conversation rather than a v0.4b mechanism
conversation.

## Falsification Surface

```text
Pre-registered claim: with two-pass gauge classification, the supplementary-B
piano-trio body decomposes cleanly into the four-band shadow defined above,
and the decomposition is a deterministic function of the orbit (no per-row
tolerance choices).

Falsifier:
  - Any row whose Pass 2 classification is inconsistent across re-runs at
    the registered tolerances (i.e., a determinism failure at fixed
    tolerances), OR
  - Any persistent `smaller_symmetry` or `undefined` row after Pass 2,
    populating bands the pre-registration declared empty-allowed.
```

Either falsifier is a real result. The first invalidates the
"deterministic projection" framing and pushes back into a methodology
review of the gauge minimizer's stability. The second populates a band
that v0.4b must then treat as a structural concern, not as a discard
category.

## What v0.4a Does NOT Do

- **No mechanism prediction.** v0.4a's role is to build the table; the
  predictor (m_3 / stability stratification) is registered separately in
  v0.4b.
- **No variational / `kfacet-sentinel` runs.** The full v0.3h audit chain
  is not invoked for v0.4a. The sigma_3-scan alone produces the
  classification.
- **No tightening of the catalog admission criterion.** The four-band
  classifier reads off `F_beta_residual_inf` at the registered
  tolerances; catalog admission tightening (e.g., requiring all 273
  rows to satisfy `sigma_3` closure tighter than the published catalog
  did) is out of scope.
- **No retraction of any v0.3 verdict.** v0.3 closed cleanly as a
  domain-of-applicability finding; v0.4a is a fresh chapter on the supp-B
  body, not a revisit.

## Where This Builds On The O_434 Anatomy Probe

The two-pass design is the encoded form of the O_434 anatomy probe's
finding (`results/isotrophy/k-facet-v04a0-o434-anatomy/anatomy_receipt.json`):

```text
O_434 anatomy verdict:           gauge_artifact
Original F_beta residual:        2.525e-01   (Pass 1 tolerances)
Tight-rerun F_beta residual:     1.060e-07   (Pass 2 tolerances)
Reduction factor:                4.20e-07    (six orders of magnitude)
```

The methodological lesson: a coarse SO(3) gauge minimizer can miss the
correct rotation axis by enough that a structurally Z2_clean row reads
as smaller_symmetry. The two-pass classifier rechecks every non-clean
row at tight tolerances exactly to prevent that silent coercion. The
O_434 receipt is the worked example that motivates the design.

## Sequencing After v0.4a

```text
v0.4a (THIS):  build the table[m_3][stability][class] from the two-pass
               classifier. Pre-registered.
v0.4b:         register an independent mechanism predictor (m_3 /
               stability stratification per Codex anchor gamma).
               Compare predicted table vs v0.4a's observed table.
v0.4c:         read the verdict. If predictor matches, v0.4 closes as a
               positive mechanism on the supp-B body. If not, v0.4
               closes as a structural-negative on the chosen predictor;
               propose v0.4b' / v0.5 alternatives.
```

v0.4a's output is required input for v0.4b. v0.4b should not be staged
until v0.4a's manifest has landed.

## How To Verify This Pre-Registration Is Still Current

Re-parse supp-B before any sweep launch:

```powershell
npm run isotrophy:parse:b
```

Confirm `total_rows = 273` and the per-m_3 counts match the scope table
above. If the local mirror has been updated and the counts drift, this
pre-registration should be re-issued.

Also confirm the O_434 anatomy receipt is in place:

```text
results/isotrophy/k-facet-v04a0-o434-anatomy/anatomy_receipt.json
results/isotrophy/k-facet-v04a0-o434-anatomy/q5_tighter_gauge/residuals.csv
```

The methodological motivation for the two-pass design lives there.
