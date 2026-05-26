# K_facet v0.3h Methodology + Result Handoff

Status: current, 2026-05-22.
Audience: collaborators, paper-side writers, future coding agents.
Canonical sources: `docs/isotrophy/kfacet/kfacet-runner-spec.md`,
`docs/isotrophy/sundog_v_isotrophy.md`, and `docs/SUNDOG_V_THREEBODY.md`.

## What v0.3h Tested

v0.3h tested the catalog-level structural-zero hypothesis for `K_facet`.
After the v0.2 containment count collapsed to the equivariance-null, the
test was rebuilt around the rank-matrix `Gamma_i`: look inside
`ker(M_i - I)` for an `F_beta`-even standard `D3` sector. If that sector is
present, `Gamma_i` can carry a branch count. If it is absent, then
`c_i = d_i = 0` by structure, not by tolerance or post-hoc pruning.

The question for the 21 strict G.2 single-curve choreographies at `m_3 = 1`
was therefore sharp: does any row contain a valid standard `D3` isotypic
block, or is the v0.3h prediction structurally zero on the catalog?

## Audit Chain Methodology

The audit chain was deliberately receipt-first and closure-relative, matching
the discipline that resolved G.2. It proceeded in three stages.

First, the sentinel/Gamma runner built `M_i`, typed `D3` operators, reduced
`omega`, and the closed-form `partial_eps M_i`, then gated D3 relations,
leakage, and finite-difference consistency. Second, the adaptive-floor
reprocessor reused the saved `.npy` artifacts, with no integration, and chose
the smallest pre-registered floor satisfying all three guards: D3 leakage
`<= 1e-3`, gap ratio `<= 1e-3`, and first-rejected singular value `>= 1e-3`.
Third, the bridge audit examined any row not resolved by that floor rule,
testing bridge vectors for neutral overlap, Jordan behavior, and defective
D3 signatures.

No row-specific knobs were introduced. Constants and outcome categories were
registered in code/spec before interpretation, so each row either became a
structural-zero receipt, a named refinement case, or an excluded defective
case.

## Result And Quarantine

Final result:

```text
20 rows  no_bridge_present              T(2) + S(5 or 6) + E(0)
 1 row   defective_E_block_confirmed    O_617
```

The load-bearing v0.3h catalog statement is: **20 of 21 strict rows are
confirmed structural zeros** under the adaptive-floor discipline. For those
rows the standard `D3` sector is empty, so `c_i = d_i = 0`.

`O_617` is quarantined, not counted for or against the prediction. Its bridge
singular value is `7.8359e-4`; neutral overlap is `1.81e-4`; the Jordan-chain
test amplifies rather than drops (`90.04`). At the bridge-admitted kernel the
representation reads `T(2) + S(6) + E(1)`, which is not a valid real standard
`D3` block, and `||sigma3^3 - I||_inf = 3.96e-2` shows the order-3 relation
fails at about 4%.

Correction from the companion deep dive: `O_617` is a clean opposite-strict
catalog row. Its admission residual is `1.01e-8`; the earlier `1.62e-1` value
is the canonical residual and is not the admitting orientation. The quarantine
therefore reflects a bridge direction outside the valid `D3` representation,
not an admission weakness and not a failure of the `Gamma_i` audit
chain.

## Freeze + Supplementary-B Comparison

The follow-on freeze is now explicit. The resolved v0.3h `Gamma_i` mechanism
predicts zero piano-trio daughters from the strict G.2 seeds:

```text
20 resolved structural-zero rows: sum d_i = 0
O_617: quarantined, excluded from evidence
K_facet_v0.3h(resolved Gamma mechanism) = 0
```

Supplementary-B parses as a nonempty piano-trio catalog:

```text
supplementary-B rows: 273
m3 = 1 rows:          38
m3 != 1 rows:        235
stability:            97 S, 176 U
```

This closes v0.3 as a **structural-null mechanism** against supplementary-B:
the audit chain measured a real prediction, and that prediction does not
explain the published piano-trio rows. This does not mean "no piano-trios";
it means the v0.3h `Gamma_i` standard-sector mechanism contributes zero
resolved daughters.

## Reproducibility Surface

Primary scripts:

```bash
npm run isotrophy:kfacet:sentinel:gamma
npm run isotrophy:kfacet:reprocess-floor
npm run isotrophy:kfacet:bridge-audit
```

Key receipt directories:

```text
results/isotrophy/k-facet-v03-sentinel-calibration-O62-gamma/
results/isotrophy/k-facet-v03-sentinel-sweep-calibrated/
results/isotrophy/k-facet-v03-sentinel-sweep-adaptive-floor-21/
results/isotrophy/k-facet-v03-bridge-audit-21/
results/isotrophy/k-facet-v03-freeze-supplementary-b-comparison/
```

Load-bearing constants include the adaptive floor ladder, projector/leakage
floor `1e-3`, gap ratio `1e-3`, `k_gamma = 3`, and `k_int = 10`. The final
claim should be cited as **20/21 structural zeros plus one quarantined O_617
defective-D3 bridge**, not as a closed 21/21 theorem-facing result.

## v0.3 Epilogue (closed, 2026-05-22)

The `alpha` freeze + supplementary-B comparison ran on 2026-05-22 and the
II cross-m_3 sentinel sweep + symmetry probe followed on the same day.
The combined verdict closes v0.3 as a **domain-of-applicability finding**,
not a within-domain falsification:

```text
v0.3 Gamma_i mechanism:
  - Defined: rank gate on F_beta-even standard D_3 isotypic of
    ker(M_i - I).
  - Domain of applicability: D_3-symmetric orbits.
  - Verified domain: the 21 strict G.2 single-curve choreographies at
    m_3 = 1 (catalog A).
  - Empirical result on domain: 20 of 21 structural zero; 1 quarantined
    row (O_617) with the `bridge_approx_sign_isotypic` diagnosis.
  - Predicted daughter count from G.2 strict 21: 0.

Supplementary-B catalog:
  - 273 piano-trio orbits at varied m_3.
  - Symmetry class verified by sigma_3-scan + cross-m_3 sentinel:
    all seven tested rows are NOT sigma_3-symmetric (residual ~0.7,
    7--9 orders above catalog-strict); six of seven are F_beta =
    (12)-swap symmetric (residual ~1e-8, comparable to strict G.2);
    O_434(0.4) also breaks F_beta. Symmetry class is Z_2-or-smaller,
    not D_3.
  - Lies outside the v0.3 prediction's domain of applicability.

Comparison verdict:
  Predicted 0 daughters from G.2 strict (verified). Observed 273
  piano-trios (verified). The mismatch is a **domain-of-applicability**
  finding: v0.3 cannot predict piano-trios by construction because they
  do not carry the assumed symmetry. The audit chain is intact; the
  prediction is intact; the catalog is in a different symmetry class.
```

The publishable v0.3 result is therefore:

> v0.3's Gamma_i rank gate predicts zero daughter piano-trios from the
> 21 strict G.2 D_3-symmetric choreographies at m_3 = 1, with 20 of 21
> structural-zero on the domain and the 21st (`O_617`) quarantined with
> a precise causal diagnosis (`bridge_approx_sign_isotypic`).
> Supplementary-B's piano-trio sentinels are empirically Z_2-or-smaller,
> not D_3; the predicted-vs-observed mismatch reflects
> domain-of-applicability, not within-domain falsification.

Sub-investigation candidates carried forward, not blocking the
epilogue:

- `O_617` (m_3 = 1 strict G.2): defective-D3 boundary, see
  `kfacet_v03h_o617_deep_dive.md`.
- `O_434(0.4)` (supp-B): breaks both `sigma_3` and `F_beta` closure;
  possibly a sub-Z_2 symmetry class. Cheap row-anatomy probe candidate.
- Typed transport lemma (III): paper-side rigor; deferred since v0.3c,
  no longer blocking the empirical close.

The next chapter is **v0.4**: design a candidate Z_2-equivariant
mechanism (or an a-symmetric one) for piano-trio prediction. This is
fresh paper-side work; do not scaffold runner code until the
representation theory closes on Z_2.

Companion docs:

- `kfacet_v03_freeze_b_comparison.md` -- alpha freeze + domain
  addendum.
- `kfacet_v03_gamma_crossm3_preregistration.md` -- II pre-registration
  + (Q1.D, Q2.D) verdict.
- `kfacet_v03h_o617_deep_dive.md` -- quarantined-row companion.

Follow-on: v0.4a treats supplementary-B piano-trios as primary `Z_2`
objects and pre-registers a two-pass gauge domain map over all 273 rows;
see `docs/isotrophy/kfacet/kfacet_v04a_domain_map_preregistration.md`.

## Open Polish Item (Run-Friendly)

A ~10-hour tooling-polish initiation is registered in
[`../../TODO.md`](../../TODO.md) under
**Onboarding / Polish (Run-Friendly) -> V0.3h K_facet Tooling Polish**.
It promotes the three one-shot diagnostic scripts
(`scripts/o617_deep_dive.py`, `scripts/o617_why_dive.py`,
`scripts/catalog_near_t_separator.py`) to discoverable workbench
subcommands with `--row` parameters and npm entries, fixes the
projector double-counting artifact in the catalog separator via
simultaneous (sigma_3, F_beta) eigendecomposition, and bakes the
"always signed inner products" lesson from the (II) round into a shared
helper. The polish does **not** alter the load-bearing v0.3h evidence.
The TODO entry includes the full paper trail (this writeup, the
deep dive, the spec, the public mirrors, and the workbench script) as
its table of contents.
