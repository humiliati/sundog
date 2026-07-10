# Kakeya Phase 3S - The Value-Set Reduction (M3 Working Receipt)

- Artifact id: `KAK-PHASE3S-VALUESET-REDUCTION`
- Date: 2026-07-09
- Status: internal M3 working receipt. **Delivered: a proved exact reduction of
  the floor conjecture to a value-set-sum statement; an exhaustive proof of the
  floor at `q = 7` over ALL configurations; a rigorous `+1` theorem with an
  elementary core; a regime split isolating the hard case as a permutation-
  polynomial statement. The general floor (`q >= 11`) remains OPEN.**
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt:
  [`PHASE3R_FLOOR_PROOF_PROGRAM.md`](PHASE3R_FLOOR_PROOF_PROGRAM.md)
- Script:
  [`../../scripts/kakeya-valueset-reduction.mjs`](../../scripts/kakeya-valueset-reduction.mjs)
  (`npm run kakeya:valueset`)
- Results: `results/kakeya/valueset-reduction/`

## The reduction (PROVED; machine-verified incl. an independent geometric path)

Send the pencil vertex `W` to the vertical direction: a one-per-direction
completion becomes a graph `{(a, f(a)) : a in F_q}` plus one infinite point
`(d)`, `d in F_q`. For `s in F_q` let `m_{s,c} = #{a : f(a) = sa + c}` (fibers
of the translate `f - s*id`), `N_s` = number of nonempty fibers = **value-set
size of `f - s*id`**, `pi_s = sum_c C(m,2)`, `sigma_s = sum_c (m-1)(m-2)/2`.

- **I1** (per slope): `pi_s - sigma_s = q - N_s`.
  *Proof:* `C(m,2) - (m-1)(m-2)/2 = m-1`; sum over the `N_s` nonempty fibers,
  `sum (m-1) = q - N_s`. ∎
- **I2**: `sum_s N_s = q(q+1)/2 + sum_s sigma_s`.
  *Proof:* sum I1 over all `q` slopes, using `sum_s pi_s = C(q,2)` (every pair
  of graph points determines exactly one slope). ∎
- **I3**: `sacrifice(f,d) = sum_{s != d} N_s - q(q-1)/2 = (q - N_d) + sum_s sigma_s`.
  *Proof:* slope-`d` lines all pass through `(d)`, so their contribution is
  `sum_c w(m+1) = sum_c C(m,2) = pi_d`; apply I1 to the other slopes and I2. ∎

Machine verification: 18 cases (quadratic, linear, random; `q in {5,7,11}`),
all three identities plus a **fully independent projective line-enumeration**
of the dual point set - all agree. Sanity anchor: `f = a^2` gives
`N_s = (q+1)/2`, `sigma_s = 0`, `sacrifice = (q-1)/2` - the BM minimum, exactly.

**Corollary (floor, value-set form).** The floor conjecture
`sacrifice >= q-2` for 4-star completions is equivalent to: *a 4-point fiber
in one translate forces `sum_{s != d} N_s >= (q^2+q-4)/2`* - i.e. forces the
translate value-set sum `(q-3)/2` above the BM minimum `(q^2-1)/2`. The
PHASE3R pencil factor `X^q | R(X, d_i)` is the 4-fiber
`(X-c_0)^4 | H(X, s_0)` of the classical Redei polynomial of `f`.

## Theorem: the floor holds at q = 7, exhaustively (PROVED)

Enumerating **all** `7^7 = 823,543` functions and all `d`: the minimum
sacrifice among configurations containing any 4-secant (finite 4-fiber with
`s != d`, or a 3-fiber at slope `d`) is exactly **`5 = q-2`**. This decides
the floor at `q = 7` over every configuration - strictly stronger than the
per-orbit B&B receipts. At `q = 5` the exhaustive minimum with a 4-secant is
`4 = q-1 > q-2`: the floor holds loosely (that field has no LOW orbit),
matching all prior data. The BM minimum `(q-1)/2` is also re-derived
exhaustively at both fields.

## Theorem (+1): a 4-star forces sacrifice >= (q-1)/2 + 1 (PROVED, BM import)

If `sacrifice(U) = (q-1)/2`, then `U` is a minimum Kakeya set, hence (BM
classification, pinned) the conic construction. The conic construction has
maximum point multiplicity 3 by an **elementary** argument: three concurrent
tangents of a conic would be three collinear points of the dual conic
(impossible), and the single extra line raises any multiplicity by at most 1.
A 4-star gives its pivot multiplicity 4 - contradiction. ∎
(Exhaustively confirmed at `q in {5,7}`: no global minimizer contains a
4-secant.) This upgrades the PHASE3G/3N "no 4-star embeds" explanation to a
proof; it yields `ex >= 1`, far below the floor's `(q-3)/2` for large `q`.

## The regime split (where the remaining hardness lives)

From I3, `sacrifice = (q - N_d) + sum_s sigma_s >= (q - N_d) + 3` (the 4-fiber
alone gives `sigma_{s_0} >= 3`), which meets the floor only when `N_d <= 5` -
a nearly empty regime, since `max_s N_s >= (q+1)/2` always (I2). Reindexing
translates (replace `f` by `f - d*id`), the entire difficulty concentrates in:

> **Sub-conjecture (permutation form).** If `f - d*id` is near-bijective
> (`N_d` large) and some translate `f - s_0*id` has a 4-point fiber, then
> `sum_s sigma_s >= N_d - 2`. Extreme case: `f` a **permutation** of `F_q`
> with a 4-fiber in some translate implies `sum_s sigma_s >= q - 2`.

This is where Redei/Hermite permutation-polynomial machinery would bite, and
it is exhaustible at `q = 7` (5040 permutations) as a next instrument. The
"shared `X^q` factor -> `+(q-3)/2`" question of PHASE3R is now exactly this
statement.

## Adversarial hunt (measurement, honest weakness noted)

Hillclimb over `f` with a frozen 4-fiber, `q in {11..37}`, two cross-ratio
quadruples each: **no counterexample** (nothing below `q-2` anywhere; the
floor value is reached at `q = 11`). Caveat: the hillclimb is weak at large
`q` (its minima sit far above the known construction optima there), so its
evidence is "nothing approached the floor from below," not tightness.

## Literature status (searched this pass)

The spectrum-gap / small-example classification of Blokhuis-De Boeck-
Mazzocca-Storme (Des. Codes Cryptogr. 72 (2014) 21-31) covers **q even**
(second/third smallest, hyperoval constructions). For **q odd** only the
minimum classification (BM 2008) is known - there is no published spectrum
theorem to import past the `+1` step. The remaining `(q-5)/2` of the floor is
genuinely open.

## Falsifier

`VALUESET_REDUCTION_MISMATCH` (instrument): identity or independent-geometry
disagreement; exhaustive minima off the BM value; a global minimizer with a
4-secant. **Clear.** A below-floor hunt find would be a counterexample
(measurement channel): none found.

## Honest verdict

M3 is **not closed**. Delivered: the exact value-set reduction (new,
proved, verified), the floor decided exhaustively at `q <= 7`, a rigorous
`+1`, the regime split with a sharp permutation-form target, and a
literature confirmation that no import covers the gap. Next instruments if
pursued: (i) exhaust the permutation sub-conjecture at `q = 7` and sample it
at `q in {11,13}`; (ii) attack the permutation form with Hermite/power-sum
identities - the first genuinely new analytic step, now with the smallest
possible surface.

## Interpretation Boundary

Workbench-internal. Identities and the `q <= 7` exhaustive results are proved;
the general floor is open; BM minimum/classification and Segre are imported,
not reproved. Register untouched, no pins. No Euclidean claim.
