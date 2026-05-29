# Riemann C1 Cell Set v0 - Explicit-Formula Parity Scaffold

> Concrete v0 cell set for
> [`FRONT_A_FUNCTIONAL_EQUATION_READING.md`](FRONT_A_FUNCTIONAL_EQUATION_READING.md).
> Status: drafted, desk-auditable, unreviewed, unrun. Filed 2026-05-28.

This file pins the first reviewable cell set for the Riemann Front-A reading.
It does **not** run a smoothed explicit-formula computation, does **not** claim
a new explicit formula, and does **not** test RH. Its job is to make the
functional-equation reading rejectable before any numerical residual is
interpreted.

## Internal Audit Addendum - Linearity Verdict

Post-filing desk audit sharpened the status of this cell set: it is soundly
constructed, but it appears to operationalize the vacuity rather than escape it.

The reason is structural. The explicit formula is linear in the test function.
Therefore:

- `C1-O` is forced-vacuous on the zero side under the symmetric `+-gamma`
  window: an odd test function summed over a symmetric multiset is zero before
  any arithmetic data is loaded.
- `C1-E` is a classical explicit-formula consistency check. A clean pass would
  verify the normalization, data, truncation floor, and implementation; it
  would not by itself be a Sundog Front-A edge.
- `C1-M` is determined by linearity: `h_M = h_E + epsilon * h_O`, so the mixed
  row is an implementation linearity check, not an independent receipt.

This addendum does not delete the cell set. It records the likely disposition:
without reviewer rescue, the cell files `R-C1-NEG-A` (Front-A vacuity) or
`R-C1-NEG-D` (competitor / standard-treatment dominance), with `R-C1-NEG-B`
kept as the useful hygiene guard against identity-zero laundering.

Secondary scale note: with `A = 10`, only the first handful of zeros and very
small primes carry meaningful numerical mass under the Gaussian. The stated
`N_zero = 5000` and prime cutoff `p <= 1,000,000` are conservative inert
ceilings for this smoke-scale cell, not evidence that the full window is
actively tested. Any future executable version must either right-size the
cutoffs to `A` or explicitly label the run as a small-term pipeline smoke.

## 1. Regime And Formula Contract

The cell set works with a standard smoothed explicit formula for the completed
Riemann zeta function on the critical line. The formula is treated as an
imported object, not a Sundog-derived object.

**Formula contract.**

- Use one Weil / Riemann explicit-formula normalization cited from the lit-pass
  source list before execution.
- Express the formula as a comparison between:
  - a zero-side sum over ordinates `gamma`;
  - a prime-side sum over prime powers;
  - archimedean / gamma-factor terms;
  - pole / main terms required by the chosen normalization.
- The implementation must include a `formula_normalization` manifest field with
  source, equation number or section, Fourier-transform convention, and all
  constants.

**Pre-execution hold.** This v0 file pins the cell set but does not pin a final
equation-number-level normalization. A reviewer or maintainer must fill that
normalization before any run. If no such normalization can be fixed without
changing the cell after seeing output, file `R-C1-NEG-C` (floor / normalization
failure).

## 2. Symmetry Object

The admitted symmetry is the functional-equation reflection, rendered on the
critical-line ordinate as:

```text
R(t) = -t
Z2 = {id, R}
```

For any registered test function `h(t)`, define:

```text
h_even(t) = (h(t) + h(-t)) / 2
h_odd(t)  = (h(t) - h(-t)) / 2
```

The scaffold does not ask whether the Z2 symmetry exists. It asks whether the
receipt separates two categories:

```text
forced parity cancellation
informative arithmetic residual
```

Forced cancellation may be identity-zero. It must be labeled as such and may not
be counted as an empirical residual pass.

## 3. Kernel Family

The v0 kernel family is a Gaussian pair with one odd perturbation.

Let `A = 10`.

```text
h_E(t) = exp(-(t / A)^2)
h_O(t) = (t / A) * exp(-(t / A)^2)
h_M(t) = h_E(t) + epsilon * h_O(t),    epsilon = 0.1
```

Roles:

- `h_E` is the residual-bearing even test. It is the only row allowed to
  support a bounded operational residual receipt.
- `h_O` is the identity-zero negative control. Under a symmetric zero window and
  a correctly parity-aware formula read, this row is expected to cancel by
  parity. It is not evidence of signal.
- `h_M` is the laundering guard. Its odd component must not improve or rescue
  the even residual; the mixed row should reduce to the `h_E` residual plus the
  separately labeled forced cancellation of `epsilon * h_O`.

The value `A = 10` is a v0 review default, not an optimized scale. If a reviewer
rejects it before execution, replace this file with a new v0.1 cell set. If it
is changed after inspecting output, file `R-C1-NEG-C`.

## 4. Cutoffs And Source Data

Default v0 cutoffs:

| Object | v0 value | Role |
| --- | --- | --- |
| Zero source | Odlyzko `zeros1` | zero-side ordinates |
| Zero window | first `N_zero = 5000` positive ordinates, mirrored to `+-gamma` | symmetric zero-side truncation |
| Observed zero ceiling | `T_zero = 5447.861998301` from Probe 01 v1 | registered window read-back |
| Prime cutoff | prime powers with base prime `p <= 1,000,000` | prime-side truncation |
| Prime-power depth | all `k` with `p^k <= 1,000,000` | finite prime-power slate |
| Arithmetic precision | at least double precision for v0 smoke; higher precision if reviewer requests | numerical floor input |

The zero window is symmetric by construction. The run must never use only
positive ordinates for an odd-sector verdict.

## 5. Registered Rows

| Row | Test function | Expected parity read | Can support a bounded positive? |
| --- | --- | --- | --- |
| `C1-E` | `h_E` | even sector, residual-bearing | pipeline consistency only unless reviewer finds non-vacuous audit edge |
| `C1-O` | `h_O` | odd sector, identity-zero negative control | no |
| `C1-M` | `h_M` | even residual plus labeled odd cancellation | no independent row; linearity check only |

Per Probe 01 v1, an identity-zero row is a null / guardrail receipt. It is not a
discovery.

## 6. Error Floor And Threshold Rule

The residual threshold must be derived before execution from a floor
independent of the observed residual.

The v0 floor is:

```text
tau_floor =
  10 * (zero_tail_bound + prime_tail_bound + archimedean_quadrature_bound
       + source_precision_bound + floating_point_bound)
```

Required components:

- **Zero tail bound.** Bound the contribution of omitted zeros `|gamma| >
  T_zero` using the chosen zero-counting density and the Gaussian decay of the
  registered `h`.
- **Prime tail bound.** Bound omitted prime powers above the cutoff using the
  chosen Fourier-transform convention and a monotone majorant for the Gaussian
  transform.
- **Archimedean quadrature bound.** Pin a deterministic quadrature tolerance
  for the gamma-factor integral or term in the chosen normalization.
- **Source precision bound.** Propagate Odlyzko `zeros1` source precision
  through the zero-side sum.
- **Floating-point bound.** Pin a conservative roundoff bound or require a
  higher-precision implementation.

If any component cannot be bounded before looking at the residual, file
`R-C1-NEG-C`.

## 7. Disposition Branches

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `R-C1-A` | `C1-E` residual <= `tau_floor`, `C1-O` labeled identity-zero, `C1-M` decomposes cleanly, and reviewer accepts the forced-cancellation / residual split as a non-vacuous audit distinction | bounded operational receipt |
| `R-C1-NEG-A` | reviewer says the cell is only standard even-test-function bookkeeping | Front-A vacuity |
| `R-C1-NEG-B` | `C1-O` or the odd part of `C1-M` is presented as empirical evidence | identity-zero laundering |
| `R-C1-NEG-C` | formula normalization, cutoffs, or floor components cannot be fixed pre-run | floor / normalization failure |
| `R-C1-NEG-D` | Connes / standard explicit-formula treatments already supply the exact operational distinction claimed | competitor dominance |
| `R-C1-FAIL` | `C1-E` residual > `tau_floor` after an admitted run | residual breach in registered cell |

No branch may be rescued by changing `A`, `N_zero`, prime cutoff, formula
normalization, or floor components after output inspection.

## 8. What Would Count As Non-Vacuous

After the linearity audit, the cell is presumed vacuity-boundary unless all
three are true:

1. the reviewer accepts that the forced-cancellation / residual split is a real
   audit distinction, not just decorative language for standard parity;
2. the run can compute `C1-E` against an independently fixed floor;
3. the report keeps `C1-O` and the odd part of `C1-M` in the negative-control
   bucket rather than counting them as evidence.

If these fail, the correct result is not a better cell. It is the named
negative.

## 9. Review Questions

Ask the external reviewer:

1. Is the selected explicit-formula normalization appropriate for the three
   registered rows?
2. Is `h_O` an admissible odd perturbation under that normalization, or should
   the odd negative control be expressed differently?
3. Is `A = 10` a reasonable desk-audit scale, or should a new pre-execution v0.1
   cell be filed with a different scale?
4. Are the proposed tail-bound components sufficient to keep the residual floor
   independent of the observed result?
5. Does this cell add any audit value beyond ordinary explicit-formula
   bookkeeping?

## 10. Status

- Drafted: 2026-05-28.
- Internal audit: linearity verdict added 2026-05-28; expected disposition is
  `R-C1-NEG-A` / `R-C1-NEG-D` unless a reviewer treats the labeling-hygiene
  distinction as a real audit contribution.
- Execution: none.
- Review: none.
- Formula normalization: not yet equation-number-pinned.
- Floor calculator: not written.
- Promotion: blocked until review and a pre-run formula/floor fill.

## Cross-References

- [`FRONT_A_FUNCTIONAL_EQUATION_READING.md`](FRONT_A_FUNCTIONAL_EQUATION_READING.md)
  - Front-A reading note this cell set instantiates.
- [`receipts/2026-05-28_probe01_pathi_parity_decomposition.md`](receipts/2026-05-28_probe01_pathi_parity_decomposition.md)
  - negative-control precedent for identity-zero labeling.
- [`../RIEMANN_LITPASS_MEMO.md`](../RIEMANN_LITPASS_MEMO.md)
  - prior-art and competitor boundaries.
- [`../SUNDOG_V_RIEMANN.md`](../SUNDOG_V_RIEMANN.md)
  - main ledger and promotion criteria.
