# Kakeya Phase 3H - Deficit Onset by Axis Avoidance

- Artifact id: `KAK-PHASE3H-DEFICIT-ONSET`
- Date: 2026-07-06
- Status: internal workbench theorem receipt; closes PHASE3F next-rung option
  (i), the deficit-onset theory for inf-containing direction sets.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Prior receipts:
  [`PHASE3F_JOINT_ACTIVATION_GAP.md`](PHASE3F_JOINT_ACTIVATION_GAP.md),
  [`PHASE3G_EMBED_MINIMAL_PROBE.md`](PHASE3G_EMBED_MINIMAL_PROBE.md)
- Litpass anchor:
  [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md) (addendum
  2026-07-06: Blokhuis-Mazzocca pin)
- Script:
  [`../../scripts/kakeya-deficit-onset-check.mjs`](../../scripts/kakeya-deficit-onset-check.mjs)
- Manifest:
  [`../../results/kakeya/deficit-onset/manifest.json`](../../results/kakeya/deficit-onset/manifest.json)

## Verdict

**The inf-dichotomy dissolves; the deficit onset is exactly `k = q + 1`.**
PHASE3F could only *prove* zero deficit for all-finite-slope direction sets
(via the standard parabola) and *measure* it for inf-containing ones. The
correct frame is **axis avoidance**: a parabola is a conic tangent to the line
at infinity at its **axis** direction, and its tangent family covers every
direction *except* the axis. The standard `y = x^2` has axis = `inf` - the
"special" role of `inf` in PHASE3F was an artifact of that one choice.

> **Theorem (workbench-checked, odd q).** For any direction set `D` with
> `|D| = k <= q`, the deficit is `0`: pick an axis `a` not in `D` and take the
> `k` tangents with directions in `D` of a parabola with axis `a`. No three
> are concurrent, so the union has exactly `kq - C(k,2)` points - which is the
> proven PHASE3F lower bound, so the construction is optimal with no search.
> The onset is exactly `k = q + 1`, where no axis remains, with tax
> `(q - 1)/2`.

The construction is one affine map: `(x, y) -> (y, x + s*y)` sends the
standard family's axis `inf` to slope `s` and permutes the remaining
directions bijectively.

## Why the onset sits at q + 1 (and nowhere earlier)

- **`k <= q`:** at least one direction is unused; make it the axis. Dually,
  the `k` tangents plus the line at infinity form a `(k+2)`-arc contained in
  the dual conic - always available while `k + 2 <= q + 2` with the tangency
  point unclaimed.
- **`k = q + 1`:** a deficit-0 configuration would dualize to a `(q+2)`-arc
  through the infinity point; for odd `q` arcs stop at `q + 1` (classical,
  conic-classified). The workbench states this as measurement + import, not
  derivation: the tax is `(q-1)/2` exhaustively at `q in {5, 7}` (PHASE3F
  enumeration) and via the pinned Blokhuis-Mazzocca minimum at `q = 11`.
- **Axis symmetry of the boundary:** for *every* choice of axis `a` (all
  `q + 1` of them, `inf` included), the `q` tangents form a `q(q+1)/2`-point
  deficit-0 family (exactly the Dvir floor, not complete) and the cheapest
  `a`-line costs exactly `(q-1)/2` more, landing on the Blokhuis-Mazzocca
  minimum, complete. Machine-checked for all `q + 1` axes at each field -
  `inf` is not special.

## Coverage

| `q` | direction sets checked (`1 <= k <= q`) | construction | independent cross-check |
| ---: | ---: | --- | --- |
| `5` | `62` (full lattice below the pencil) | all exactly `kq - C(k,2)` | exhaustive joint enumeration, all agree |
| `7` | `254` | all exact | exhaustive, all agree |
| `11` | `4094` | all exact | not needed: bound + construction self-certify |

This retires PHASE3F's `q = 11` cap: the joint-gap receipt certified the
subset lattice only to `k <= 4` (enumeration budget); the construction now
certifies **all 4094 sets to `k = 11`** with zero search, because achieving
the proven lower bound is its own optimality certificate.

## Executable Receipt

Command:

```powershell
npm run kakeya:deficit-onset
```

Output:

```text
KAK_DEFICIT_ONSET q={5,7,11} q5_subsets=62 q7_subsets=254 q11_subsets=4094 construction=pass exhaustive_q5=pass exhaustive_q7=pass axis_completions=pass falsifier=clear out=results\kakeya\deficit-onset
```

Checks: family sanity (each axis family = `q` full lines, direction-bijective
onto the non-axis directions, all `q + 1` axes per field); the `kq - C(k,2)`
size law on every direction set with `k <= q`; independent exhaustive joint
enumeration at `q in {5, 7}` (brute-force over all intercept choices, no reuse
of the construction); and all `q + 1` axis completions (tangent union
`= q(q+1)/2`, last line `= (q-1)/2`, total `= BM`, complete). Deterministic,
no API.

Falsifier `DEFICIT_ONSET_MISMATCH`: fires on any family/bijection failure,
size-law miss, exhaustive disagreement, or axis-completion miss. **Clear.**

## Regression Pin

`scripts/kakeya-workbench-tests.mjs` gains two pins per supported `q` (suite
now 72/72):

1. `T8m axis-avoiding-family-deficit-zero`: the slope-0-axis tangent family
   (an inf-containing `k = q` set) has exactly `q(q+1)/2` points with slope 0
   dark and `inf` lit.
2. `T8n axis-symmetric-pencil-completion`: for axes `inf` and slope 0 alike,
   the tangents-plus-cheapest-axis-line completion costs exactly `(q-1)/2`
   and lands on the minimum, complete.

## Interpretation Boundary

Supports only:

> In the finite-field workbench (odd prime `q`), joint direction activation
> from empty has zero concurrence deficit for every direction set of size up
> to `q`, by an explicit axis-avoiding parabola construction whose optimality
> is certified by the already-proven pairwise bound; the deficit onset is
> exactly at the full pencil, with tax `(q-1)/2`.

The onset *value* at `q = 11` imports the pinned Blokhuis-Mazzocca minimum;
at `q in {5, 7}` it is exhaustively derived. The dual-arc framing at
`k = q + 1` cites classical arc theory as explanation, not as a workbench
derivation. Even `q` is out of scope (the pinned source gives sharp minimum
`q(q+1)/2` there - zero tax, hyperoval territory - noted as contrast only).
Report-only; public export untouched; no Euclidean claim.

With this receipt the PHASE3E-3H arc is closed: marginal metric (3E), joint
gap + sandwich (3F), embeddability frontier (3G), and deficit onset (3H).
Remaining banked reopeners: the 4-star cross-ratio orbit sweep, and the
(now literature-answered) minimal-set census as an optional machine
confirmation at `q = 5`.
