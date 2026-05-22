# K_facet v0.3h Methodology + Result Handoff

Status: current, 2026-05-22.
Audience: collaborators, paper-side writers, future coding agents.
Canonical sources: `internal/anniversary/kfacet-runner-spec.md`,
`docs/sundog_v_isotrophy.md`, and `docs/SUNDOG_V_THREEBODY.md`.

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
```

Load-bearing constants include the adaptive floor ladder, projector/leakage
floor `1e-3`, gap ratio `1e-3`, `k_gamma = 3`, and `k_int = 10`. The final
claim should be cited as **20/21 structural zeros plus one quarantined O_617
defective-D3 bridge**, not as a closed 21/21 theorem-facing result.
