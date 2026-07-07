# Kakeya Phase 3I - 4-Star Cross-Ratio Orbit Sweep

- Artifact id: `KAK-PHASE3I-CROSSRATIO-ORBIT-SWEEP`
- Date: 2026-07-06
- Status: internal workbench diagnostic receipt; executes the PHASE3G banked
  reopener (cross-ratio orbit sweep of 4-star embeddability excess).
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Prior receipts:
  [`PHASE3G_EMBED_MINIMAL_PROBE.md`](PHASE3G_EMBED_MINIMAL_PROBE.md),
  [`PHASE3H_DEFICIT_ONSET.md`](PHASE3H_DEFICIT_ONSET.md)
- Litpass anchor:
  [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md) (Blokhuis-Mazzocca
  pin, addendum 2026-07-06)
- Script:
  [`../../scripts/kakeya-crossratio-orbit-sweep.mjs`](../../scripts/kakeya-crossratio-orbit-sweep.mjs)
- Manifest:
  [`../../results/kakeya/crossratio-orbit-sweep/manifest.json`](../../results/kakeya/crossratio-orbit-sweep/manifest.json)

## Verdict

**The 4-star excess is exactly an orbit function, and the orbit split is real
but collapsible.** Every one of the 580 direction quadruples at
`q in {5, 7, 11}` was solved exactly; `ex` is constant on each PGL(2,q) orbit
(the affine-equivalence theorem, now a 580-row solver audit), and the map from
orbits to excess is:

| `q` | orbit | six-set | quadruples | `ex` |
| ---: | --- | --- | ---: | ---: |
| `5` | harmonic (the only orbit) | `{2,3,4}` | `15` | `2` |
| `7` | harmonic | `{2,4,6}` | `42` | `2` |
| `7` | equianharmonic | `{3,5}` | `28` | `3` |
| `11` | harmonic | `{2,6,10}` | `165` | `4` |
| `11` | generic | `{3,4,5,7,8,9}` | `330` | `4` |

Two headline readings:

- **PHASE3G's split explained.** At `q = 7` the probed quadruples `{0,1,2,3}`
  (harmonic) and `{0,1,2,4}` (equianharmonic) sit in the field's only two
  orbits - the measured `ex = 2` vs `3` *is* the orbit function. At `q = 5`
  everything is harmonic (F_5 \ {0,1} is exactly the harmonic six-set), which
  is why 3G's two probes agreed there.
- **PR4 answered: `ex(harmonic, q = 11) = 4`, identical to generic.** Both 3G
  probes at `q = 11` were generic (cross-ratios 5 and 7); the unprobed
  harmonic orbit turns out to give the same excess. So cross-ratio dependence
  is not monotone in any obvious invariant: the orbit structure separates
  excess at `q = 7` and collapses at `q = 11`.

The visible pattern is that the *deviating* orbit at `q = 7` is the
equianharmonic one (`lambda^2 - lambda + 1 = 0`, j = 0), and `q = 11` has no
equianharmonic orbit (`-3` is a non-residue mod 11). Whether the deviation is
an equianharmonic phenomenon is untestable in the current workbench: the next
field with an equianharmonic orbit is `q = 13` (`-3` is a QR mod 13), outside
the locked `SUPPORTED_Q`. Banked as a reopen condition, owner-gated (extending
`kakeya-core` touches the Phase-2 spec lock).

## Pre-Registered Predictions vs Outcomes

| # | prediction | outcome |
| --- | --- | --- |
| PR1 | `ex` constant on each orbit (theorem; full-sweep solver audit) | **CONFIRMED** - 580/580, no intra-orbit variation |
| PR2 | `q=5` harmonic `2`; `q=7` harmonic `2`, equianharmonic `3` | **CONFIRMED** |
| PR3 | every orbit has `ex >= 1` (pinned classification consequence: no odd-`q` minimal set has a mult-4 point) | **CONFIRMED** - all 580 |
| PR4 | `ex(harmonic, q=11)` - open, new measurement | **ANSWERED: `4`** (= generic; the split collapses at `q = 11`) |

## Orbit Inventory (field arithmetic, machine-matched)

The six-set classes over `F_q \ {0, 1}` partition as: `q = 5`: one class
(harmonic = everything); `q = 7`: harmonic `{2,4,6}` + equianharmonic `{3,5}`;
`q = 11`: harmonic `{2,6,10}` + one generic class `{3,4,5,7,8,9}`. The sweep's
measured orbit keys match this inventory exactly, and the orbit sizes
(`15`; `42 + 28`; `165 + 330`) sum to `C(q+1, 4)` at each field. Quadruples
containing `inf` are handled uniformly (`inf` is just a point of `PG(1,q)` in
the cross-ratio arithmetic) - PHASE3G had probed only finite-slope quadruples.

## Executable Receipt

Command:

```powershell
npm run kakeya:crossratio-sweep
```

Output:

```text
KAK_CROSSRATIO_ORBIT_SWEEP q={5,7,11} q5_orbits=harmonic:ex=2(n=15) q7_orbits=harmonic:ex=2(n=42)+equianharmonic:ex=3(n=28) q11_orbits=generic:ex=4(n=330)+harmonic:ex=4(n=165) anchors3G=pass falsifier=clear out=results\kakeya\crossratio-orbit-sweep
```

Machinery: the PHASE3G exact completion solver (branch-and-bound + dynamic
remaining-marginal bound + greedy seed) on the 4-star of each quadruple; every
solve exact within the 40M-node budget. Checks: intra-orbit `ex` constancy,
measured-vs-arithmetic orbit inventory, `ex >= 1` everywhere, and reproduction
of all six PHASE3G anchor values. Full per-quadruple detail in `sweep.csv`
(580 rows). Deterministic, no API.

Falsifier `CROSSRATIO_ORBIT_MISMATCH`: fires on intra-orbit variation,
inventory mismatch, any `ex = 0` (would contradict the pinned classification),
a budget exhaustion, or a 3G anchor miss. **Clear.**

## Regression Pin

`scripts/kakeya-workbench-tests.mjs` gains two pins per supported `q` (suite
now 78/78):

1. `T8o crossratio-orbit-classes`: the six-set classes partition
   `F_q \ {0,1}`, contain the harmonic class, and number `1/2/2` at
   `q = 5/7/11`.
2. `T8p anchor-quadruple-orbits`: `{0,1,2,3}` is harmonic/harmonic/generic and
   `{0,1,2,4}` is harmonic/equianharmonic/generic across the three fields.

## Interpretation Boundary

Supports only:

> In the finite-field workbench (odd prime `q in {5, 7, 11}`), the
> embeddability excess of a 4-star is a PGL(2,q)-orbit invariant of its
> direction quadruple; the orbit-to-excess map is measured exhaustively, with
> the excess split present at `q = 7` and collapsed at `q = 11`.

`ex` values are relative to the pinned Blokhuis-Mazzocca minimum (derived
exhaustively at `q in {5, 7}`, imported at `q = 11`). The equianharmonic
conjecture (deviation tied to the j = 0 orbit) is a pattern over three data
points, not a claim; testing it requires `q = 13`, outside the locked
workbench. Report-only diagnostic in `results/`; public export untouched; no
Euclidean claim, no incidence-geometry novelty claim.

With this receipt the PHASE3E-3I arc has no open banked reopeners inside the
current field range. Reopen conditions on record: (i) `q = 13` extension for
the equianharmonic question (owner-gated spec change); (ii) optional `q = 5`
machine confirmation of the literature-answered minimal-set census.
