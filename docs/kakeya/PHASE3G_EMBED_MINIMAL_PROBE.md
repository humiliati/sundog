# Kakeya Phase 3G - Embeddability-in-Minimal-Sets Probe

- Artifact id: `KAK-PHASE3G-EMBED-MINIMAL-PROBE`
- Date: 2026-07-06
- Status: internal workbench diagnostic receipt; executes PHASE3F next-rung
  option (ii). Companion open thread: deficit-onset theory for inf-containing
  sets (PHASE3F option (i), not started).
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Prior receipt:
  [`PHASE3F_JOINT_ACTIVATION_GAP.md`](PHASE3F_JOINT_ACTIVATION_GAP.md)
- Script:
  [`../../scripts/kakeya-embed-minimal-probe.mjs`](../../scripts/kakeya-embed-minimal-probe.mjs)
- Manifest:
  [`../../results/kakeya/embed-minimal-probe/manifest.json`](../../results/kakeya/embed-minimal-probe/manifest.json)

## Verdict

**Embeddability in a minimal complete Kakeya set is governed by a concurrency
budget, and the budget is necessary but not sufficient.** Define

```text
completion(K) = |K| + joint(all missing directions, K)   (minimal complete superset)
ex(K)         = completion(K) - BM(q) >= 0
K embeds in a minimal complete set  <=>  ex(K) = 0
```

Every probe body resolved at all three fields (no budget exhaustion, no
unknowns). The frontier: **lines, line-plus-a-point, 2-stars, 3-stars, and
tangent triples embed at every q; 4-stars embed nowhere** - including at
`q in {7, 11}` where the budget arithmetic would allow them - **and the
excess of a 4-star depends on the cross-ratio of its directions** (`ex = 2`
vs `3` at `q = 7` for slopes `{0,1,2,3}` vs `{0,1,2,4}`).

## The Concurrency-Budget Identity (proven, machine-checked)

Any complete set equal to the union of its `q + 1` canonical one-per-direction
lines satisfies (pair-counting: distinct-direction lines meet exactly once, so
`sum_p C(m_p, 2) = C(q+1, 2)`):

```text
|K| = q(q+1)/2 + sacrifice,   sacrifice = sum_p (m_p - 1)(m_p - 2)/2
```

where `m_p` is the number of decomposition lines through `p`. With the
Blokhuis-Mazzocca minimum (PHASE3F: derived exhaustively at `q in {5, 7}`,
imported at `q = 11`):

```text
ex(K) = sacrifice(K) - (q-1)/2
```

- The Dvir floor `q(q+1)/2` is the zero-sacrifice fiction; completeness costs
  a concurrency budget of exactly `(q-1)/2`.
- **Budget corollary:** a `k`-star inside a minimal set is a point of line-
  multiplicity `k`, costing `(k-1)(k-2)/2`; so a `k`-star embeds only if
  `(k-1)(k-2)/2 <= (q-1)/2`. This *proves* star-k4 out at `q = 5` (3 > 2) and
  star-k5 out through `q = 11` (6 > 5).
- The parabola witness spends its budget as exactly `(q-1)/2` triple points
  (the vertical meets tangent pairs `t <-> 2c - t`); multiplicity profiles
  `[3:2]`, `[3:3]`, `[3:5]` measured at `q = 5, 7, 11`.

## Pre-Registered Predictions vs Outcomes

| # | prediction | outcome |
| --- | --- | --- |
| P1 | star-k3 embeds at all `q` (witness triples + PGL 3-transitivity) | **CONFIRMED**, constructive at all `q` |
| P2 | line, line-plus-1 embed by containment | **CONFIRMED** |
| P3 | star-k4 at `q = 7` budget-exact - open; cross-ratio may matter | **RESOLVED NEGATIVE**: `ex = 2`; alt cross-ratio `ex = 3` - **cross-ratio dependence is real** |
| P4 | greedy `q = 11` excess 6 = sacrifice overspend | **CONFIRMED**: sacrifice 11, budget 5, profile `[1:32 2:36 3:8 4:1]` (a quadruple point) |
| P5 | arithmetic exclusions (star-k4 at `q=5`, star-k5 everywhere) | **CONFIRMED** (solver-exact cross-checks: `ex = 2`, `4`) |
| P6 | sacrifice identity on every complete line-union | **CONFIRMED** (incl. pencil-minus-one: `ex = 4/12/40 = sacrifice - budget`) |

## Measured Frontier

| body | q=5 | q=7 | q=11 |
| --- | --- | --- | --- |
| line / line+1 / star-k2 / star-k3 / tangent-triple | ex `0` | ex `0` | ex `0` |
| star-k4 `{0,1,2,3}` | ex `2` (arith) | ex `2` | ex `4` |
| star-k4-alt `{0,1,2,4}` | ex `2` (arith) | **ex `3`** | ex `4` |
| star-k5 | complete, ex `4` | ex `4` (arith) | excluded (arith) |
| pencil-minus-one | ex `4` | ex `12` | ex `40` |
| greedy cover | ex `0` | ex `0` | ex `6` |
| random-third / random-half | ex `0` / `0` | ex `0` / `2` | ex `6` / `12` |

Readings:

- **The budget is not sufficient.** At `q = 7` a mult-4 point costs exactly
  the whole budget (3 = 3) and at `q = 11` it fits with room (3 <= 5), yet no
  minimal set accommodates either probed 4-star: realizability fails
  geometrically, not arithmetically.
- **Cross-ratio sensitivity.** PGL(2,q) is 3-transitive on directions (all
  3-stars are equivalent) but not 4-transitive; direction quadruples carry a
  cross-ratio invariant, and at `q = 7` it separates `ex = 2` from `ex = 3`.
  Verdicts for 4-stars are per-direction-set, not universal.
- **Greedy anatomy.** At `q in {5, 7}` the greedy cover's multiplicity profile
  is *identical* to the parabola witness (`[1:6 2:9 3:2]`, `[1:9 2:19 3:3]`) -
  greedy lands on parabola-like minimal sets. At `q = 11` it overspends
  (one mult-4 point among its sacrifice of 11) and pays `ex = 6`.
- **Organic embeddability decays.** Random bodies: both embed at `q = 5`, only
  the sparser at `q = 7`, neither at `q = 11` (`ex = 6, 12`).

## Certificates and the q = 11 Asymmetry

Verdict machinery, in order:

1. **Complete bodies**: `ex = |K| - BM` arithmetically; sacrifice identity
   checked on the canonical decomposition.
2. **Affine certifier** (embeds): brute over `AGL(2,q)` images of `K` tested
   for containment in the parabola witness - a constructive minimal superset.
3. **Arithmetic exclusion** (does not embed): the budget corollary.
4. **Exact solver**: the PHASE3F branch-and-bound plus a dynamic remaining-
   marginal bound (`sum of min-adds vs current union - C(r,2)`) and a greedy
   seed. Exact on *every* probe body: all of `q in {5, 7}` (ground truth
   cross-checking the certificates) and every unresolved `q = 11` body within
   the pre-registered 40M-node budget (worst observed: 13.2M nodes,
   random-third full 12-direction completion).

At `q = 11` the two verdict directions have different epistemic weight:
**"does not embed" is fully workbench-derived** (solver-exact completion
`> 71`, and `BM <= 71` from the explicit witness), while **"embeds" leans on
the imported floor** (`BM >= 71`, Blokhuis-Mazzocca; bibliographic pin landed
2026-07-06 in [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md)).

## Executable Receipt

Command:

```powershell
npm run kakeya:embed-probe
```

Output:

```text
KAK_EMBED_MINIMAL_PROBE q={5,7,11} q5_embeds=line+line-plus-1+star-k2+star-k3+tangent-triple+greedy-cover+parabola-witness+random-third+random-half q7_embeds=line+line-plus-1+star-k2+star-k3+tangent-triple+greedy-cover+parabola-witness+random-third q11_embeds=line+line-plus-1+star-k2+star-k3+tangent-triple+parabola-witness falsifier=clear out=results\kakeya\embed-minimal-probe
```

Falsifier `EMBED_PROBE_MISMATCH`: fires if the sacrifice identity fails on a
complete line-union, a certificate contradicts the exact solver or the budget
corollary, the witness misses its exact `(q-1)/2` sacrifice, or an
arithmetically excluded star embeds. **Clear.** 13 probe bodies per field,
deterministic (PHASE3E/3F seeds), no API.

## Regression Pin

`scripts/kakeya-workbench-tests.mjs` gains two pins per supported `q` (suite
now 66/66):

1. `T8k parabola-concurrency-budget`: the witness's canonical decomposition
   sacrifices exactly `(q-1)/2`, as `(q-1)/2` triple points and nothing
   heavier.
2. `T8l pencil-budget-corollary`: the pencil-minus-one's excess over the
   minimum equals its sacrifice minus the budget.

## Interpretation Boundary

Supports only:

> In the finite-field workbench, embeddability of a body in a minimal complete
> Kakeya set is decided by an exact completion solver plus a concurrency-budget
> identity; the budget is necessary but not sufficient, and 4-star verdicts
> depend on the direction cross-ratio.

Star verdicts are for the probed direction sets (3-stars are all equivalent
by 3-transitivity; 4-stars are not). No classification of minimal sets is
claimed - "no 4-star embeds" here means "not these two cross-ratio classes at
these fields". The `q = 11` embed verdicts import the Blokhuis-Mazzocca floor.
Report-only diagnostic in `results/`; public export untouched; no Euclidean
claim.

Next-rung options if pursued: (i) the deferred **deficit-onset theory for
inf-containing sets** (PHASE3F option (i) - the agreed circle-back); (ii)
**cross-ratio orbit sweep**: all `PGL`-orbits of direction quadruples at
`q in {7, 11}` to map `ex` as a function of cross-ratio; (iii) multiplicity-
profile census of *all* minimal sets at `q = 5` (exhaustible: are they all
parabola-profile `[3:2]`?).
