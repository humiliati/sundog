# K_facet v0.3h Audit Chain — Hand-Off Writeup

Status: current, 2026-05-22.
Audience: collaborators, paper-side writers, future coding agents.
Source docs: `docs/sundog_v_isotrophy.md`, `docs/SUNDOG_V_THREEBODY.md`,
`internal/anniversary/kfacet-runner-spec.md`.
Companion deep dive: `internal/anniversary/kfacet_v03h_o617_deep_dive.md`
(structural anatomy of the one quarantined row).

## One-Line Read

The v0.3h K_facet `Gamma_i` rank-matrix prediction collapses to a structural
zero on 20 of the 21 strict G.2 single-curve choreographies at `m_3=1` in
Li-Liao supplementary-A, with the 21st row (`O_617`) sharply quarantined as
a defective-D3 bridge case rather than buried as numerical noise. The result
rests on a three-stage no-tunable audit chain whose receipts are reproducible
from three npm scripts.

## What v0.3h Tested

v0.3 frames the K_facet branch-validity object as
`Gamma_i = <xi, omega . d_eps M_i . xi'>` on the `F_beta`-even standard
isotypic of `ker(M_i - I)`, with `d_i = rank_floor(Gamma_i)` as the
structural count. v0.3h asked, for the 21 strict catalog rows: is `d_i = 0`
by structure (no standard `D3` sector ever appears), by numerical accident
(small but nonzero), or by ansatz failure (the row does not carry the
representation)?

## Audit Chain Methodology

The chain ratchets one row at a time through three pre-registered gates,
each with no per-row knobs and a closure-relative discipline matching G.2:

1. **Sentinel gates** (`npm run isotrophy:kfacet:sentinel:gamma`). For each
   row: integrate, build `M_i` and the typed `D3` operators, gate by
   D3 relations + reduced `omega` + closed-form vs. finite-difference
   `d_eps M_i` + leakage of D3 outside `ker(M_i - I)`. All four gates
   closure-relative; the F_beta leakage gate is the load-bearing check
   that prevents a non-D3-invariant kernel from counting as evidence.
2. **Adaptive-floor reprocessor** (`npm run isotrophy:kfacet:reprocess-floor`).
   No-integration receipt sweep over the sentinel's `.npy` artifacts. Picks
   the smallest floor per row in a pre-registered ladder satisfying
   simultaneously: D3 leakage `<= 1e-3`, gap ratio `<= 1e-3`, first-rejected
   singular value `>= 1e-3`. The triple guard prevents both noise-band cuts
   and bridge-band cuts.
3. **Bridge audit** (`npm run isotrophy:kfacet:bridge-audit`). No-integration
   triage for any row that the reprocessor cannot place in a clean spectral
   gap. Tests each bridge singular vector for neutral-basis overlap, Jordan
   chain alignment via `||(M-I)^2 v|| / ||(M-I) v||`, and defective-D3
   signatures at the bridge-admitted kernel.

The five preregistered outcome categories — `no_bridge_present`,
`all_neutral_overlap_explained`, `jordan_block_explained`,
`defective_E_block_confirmed`, `jordan_suspected`,
`unexplained_escalate_to_rtol` — discharge a row to one of three
follow-ups (quotient refinement, Jordan-block accommodation, defective
exclusion) or escalate to a tighter-rtol rerun.

## Result and Quarantine

Final manifest:

```text
20 rows  no_bridge_present              ker(M-I) = T(2) + S(5 or 6) + E(0)
 1 row   defective_E_block_confirmed    O_617
```

Twenty rows read `c_i = d_i = 0` by structure: no standard D3 sector
appears in any kernel. The single quarantined row `O_617` has its bridge
singular value at `7.836e-4` cleanly diagnosed: neutral overlap `1.81e-4`
(not neutral), Jordan chain drop `90.04` (norm *amplifies* under second
application, opposite of a Jordan root), and at the bridge-admitted
`k_dim = 8` the D3 representation itself is defective with isotypic
`T(2) + S(6) + E(1)`, `P_E` marginal singular value `0.01475`, and
`||sigma_3^3 - I||_inf = 3.96e-2`. The orbit does not carry `D3` at the
bridge boundary, so it sits outside v0.3h `Gamma_i` evidence. It is
excluded, not refuted: the defective-D3 fact is itself a structural
finding, and the deep-dive companion note investigates whether it points
to a near-symmetry-broken orbit or a catalog selection edge case.

## Reproducibility Surface

Three subcommands of `scripts/isotrophy_workbench.py`, exposed as:

```bash
npm run isotrophy:kfacet:sentinel:gamma          # per-row sentinel + Gamma_i
npm run isotrophy:kfacet:reprocess-floor          # adaptive-floor sweep
npm run isotrophy:kfacet:bridge-audit             # bridge triage
```

Constants are pre-registered at module level
(`ADAPTIVE_FLOOR_LADDER`, `ADAPTIVE_FLOOR_PROJECTOR_FLOOR`,
`BRIDGE_AUDIT_LADDER`, `BRIDGE_NEUTRAL_OVERLAP_THRESHOLD`,
`BRIDGE_JORDAN_CHAIN_THRESHOLD`, ...) with inline rationale comments.
Receipt directories under `results/isotrophy/k-facet-v03-*` carry per-row
JSON plus the `.npy` matrices (`M_i`, `D3_*`, `omega`, `Gamma_i`, etc.).
Each subcommand's manifest aggregates outcome counts, suspicious rows,
and (for the reprocessor) selected-floor histograms so a single grep
suffices to confirm the catalog claim against a fresh receipt set.

## Out of Scope

- `O_617` structural deep dive (see companion note).
- Paper-side derivation of the `Gamma_i` rank gate or the closure-relative
  discipline; this writeup describes the *audit chain*, not the underlying
  representation theory.
- Cross-`m_3` or supplementary-B verification. The discipline is portable;
  the result is `m_3 = 1` only.
