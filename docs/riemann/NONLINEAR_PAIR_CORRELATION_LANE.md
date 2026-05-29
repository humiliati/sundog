# Nonlinear Pair-Correlation Lane - Scoping Note

> Follow-on scoping note from the Riemann C1 linearity audit. Filed
> 2026-05-28. This is not a probe spec and not a claim about RH.

## Why This Lane Opens

The C1 explicit-formula cell set made the Front-A reflection scaffold concrete
and exposed its main defect: the explicit formula is linear in the test
function. Parity decomposition in that substrate is therefore mostly free.
Forced cancellations are real, but they are not a structural-zero edge.

The next plausible place to look is the nonlinear zero-statistics substrate:
pair correlations, nearest-neighbor distributions, gap ratios, and n-point
counts. These are not linear sums of a single test function over individual
zeros. They involve products, pair counts, local spacings, and windowed
statistics. That does not make the lane promising by default, but it gives the
Sundog apparatus an object where structural constraints may have bite rather
than being forced by linearity.

## Claim Boundary

This lane does **not** claim:

- a replacement for Montgomery-Odlyzko / GUE statistics;
- a new pair-correlation theorem;
- evidence for or against RH;
- a D3 bridge for Riemann zeros;
- a structural-zero receipt already exists.

It stages a question:

> Can a pre-registered finite-catalog / representation-style audit detect a
> nontrivial structural boundary in nonlinear zero statistics, beyond the
> standard GUE comparison?

The expected answer may be no. A clean no is publishable inside the Sundog
ledger because it names where the apparatus has no edge.

## Candidate Objects

Start with objects already native to Track A of
[`RIEMANN_LITPASS_MEMO.md`](../RIEMANN_LITPASS_MEMO.md):

- unfolded nearest-neighbor gaps `s_n`;
- consecutive gap pairs `(s_n, s_{n+1})`;
- gap ratios `r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})`;
- two-point window counts
  `C_I(u) = # {(i, j): i != j, gamma_i, gamma_j in I, unfolded distance in u-window}`;
- centered deviations from the registered GUE / sine-kernel baseline.

The important shift from C1 is that the candidate object is nonlinear in the
zero list: pair counts and gap ratios are not forced to vanish by odd/even
test-function parity.

## Candidate Symmetry / Representation Hooks

Three low-risk hooks are worth desk-auditing before any run:

1. **S2 gap-pair reflection.** Act on consecutive gap pairs by
   `(s_n, s_{n+1}) -> (s_{n+1}, s_n)`. This is a real finite symmetry of the
   local two-gap feature, not the functional equation itself. It can test
   asymmetry of local spacing dynamics without importing D3.
2. **Cyclic triple gap hook.** Use triples `(s_n, s_{n+1}, s_{n+2})` with a C3
   cyclic action. This is weaker than S3 and may still be mostly bookkeeping,
   but it avoids pretending ordered zeros have a canonical full S3 action.
3. **Baseline-residual sign sectors.** Decompose pair-correlation residuals
   against the sine-kernel / GUE baseline into registered local feature bins.
   The representation object is the finite bin catalog, not a spectral
   operator.

None of these is admitted. Each needs a bridge note like
[`REPRESENTATION_BRIDGE_NOTES.md`](REPRESENTATION_BRIDGE_NOTES.md) before a
probe can run.

## Candidate Probe 05 Shape

Working name:

```text
PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md
```

Minimum v0 spec:

- zero source: Odlyzko `zeros1` or a higher table, frozen before execution;
- window: one registered `N` and one registered height interval;
- unfolding: one density rule;
- statistic: choose exactly one of gap pairs, gap ratios, or two-point counts;
- baseline: registered GUE / sine-kernel comparator where applicable;
- symmetry hook: S2 gap-pair reflection or C3 gap-triple rotation;
- residual floor: bootstrap or analytic sampling floor fixed before inspection;
- disposition: structural asymmetry, baseline agreement, null, or bridge
  quarantine.

## Named Negatives

- **R-NL-NEG-A: GUE dominance.** The registered residual is completely explained
  by standard GUE / sine-kernel statistics; Sundog adds no distinguishable
  audit edge.
- **R-NL-NEG-B: representation triviality.** The proposed S2/C3 action is
  trivial on the registered invariants, reproducing the C1 vacuity problem in
  nonlinear clothing.
- **R-NL-NEG-C: sampling-floor failure.** The finite zero window cannot support
  the claimed residual or asymmetry below the registered sampling floor.
- **R-NL-NEG-D: bridge overreach.** The note silently upgrades S2/C3 local gap
  symmetries into D3 / S3 claims.

## Recommended Next Artifact

Do not run this lane yet. The next useful file is a bridge note:

```text
docs/riemann/NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md
```

It should answer one question before any code:

> Which finite symmetry is actually present in the nonlinear statistic, and
> what would count as a nontrivial sector rather than a relabeling?

If the answer is "none," file `R-NL-NEG-B` and stop. If an S2 gap-pair bridge
survives, then write Probe 05.

**Filed 2026-05-28:**
[`NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md`](NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md).
Adjudication: only the **S₂ gap-pair swap survives** — and it survives as a
*time-reversibility test* of the gap sequence, whose expectation is pinned to
zero by the GUE / sine-kernel baseline, so its predicted landing is
`R-NL-NEG-A` (GUE dominance), not a structural-zero edge. The **C3 triple is a
relabeling** (`R-NL-NEG-B`: cyclic action on linearly-ordered gaps is not a
symmetry the data has). The **residual sign-bins are downgraded** (standard
GUE residual analysis, no structural-zero content). Probe 05 v0 is now filed at
[`PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md`](PROBE_05_NONLINEAR_ZERO_STATISTICS_SPEC.md)
on the S₂ hook only, with `R-NL-NEG-A` pre-registered as the expected outcome.

## Status

- Drafted: 2026-05-28.
- Trigger: C1 explicit-formula linearity audit.
- Execution: Probe 05 executed 2026-05-28 (S₂ hook only).
- Admission: S₂ gap-pair swap admitted for a Probe 05 v0 (NEG-A expected);
  C3 quarantined; residual bins downgraded. See bridge notes.
- Probe 05 v0 spec filed: 2026-05-28.
- Probe 05 result: `R-NL-NEG-A` bounded reversibility null as predicted
  (`D=-0.0064`, inside floor `0.0424`, `0.45σ`; `tau_boot<tau_ind` confirms GUE
  anti-persistence). Receipt:
  [`receipts/2026-05-28_probe05_reversibility_null.md`](receipts/2026-05-28_probe05_reversibility_null.md).
- Next gate: cross-lane synthesis note (three lanes, three named vacuity causes,
  no structural-zero edge) OR external review.

## Cross-References

- [`RIEMANN_C1_CELLSET_V0.md`](RIEMANN_C1_CELLSET_V0.md)
  - linear explicit-formula cell set whose audit opened this lane.
- [`FRONT_A_FUNCTIONAL_EQUATION_READING.md`](FRONT_A_FUNCTIONAL_EQUATION_READING.md)
  - Front-A reading note.
- [`../RIEMANN_LITPASS_MEMO.md`](../RIEMANN_LITPASS_MEMO.md)
  - Track A prior art and standard baselines.
- [`../SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md)
  - main ledger.
