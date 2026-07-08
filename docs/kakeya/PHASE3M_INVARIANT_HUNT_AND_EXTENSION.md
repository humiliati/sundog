# Kakeya Phase 3M - Invariant Hunt + Descent Extension + q=19 Solves

- Artifact id: `KAK-PHASE3M-INVARIANT-HUNT-AND-EXTENSION`
- Date: 2026-07-08
- Status: internal receipt; executes the three PHASE3L banked moves. Hunt and
  extension COMPLETE; the two q=19 exact solves IN FLIGHT at filing time
  (results to be appended as an addendum).
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Prior receipt:
  [`PHASE3L_STAR_PARABOLA_CONSTRUCTION.md`](PHASE3L_STAR_PARABOLA_CONSTRUCTION.md)
- Scripts:
  [`../../scripts/kakeya-orbit-invariant-hunt.mjs`](../../scripts/kakeya-orbit-invariant-hunt.mjs)
  (`npm run kakeya:invariant-hunt`),
  [`../../scripts/kakeya-star-parabola-construction.mjs`](../../scripts/kakeya-star-parabola-construction.mjs)
  (`--refine`, `npm run kakeya:star-construction -- --refine`),
  [`../../scripts/kakeya-twolevel-law-probe.mjs`](../../scripts/kakeya-twolevel-law-probe.mjs)
  (Amendment D: `--orbits`, `--reps 1`, `--budget`)
- Results: `results/kakeya/orbit-invariant-hunt/`,
  `results/kakeya/star-parabola-construction-refined/`,
  `results/kakeya/twolevel-q19-{harmonic,equianharmonic}/` (in flight)

## Step 1 - Invariant Hunt

Battery of PGL-invariant quadratic-character data per cross-ratio orbit,
`q in {5..37}` (37 orbits), trained ONLY on solver-certified exacts and
construction-definitive lows; construction highs held out as unreliable.

- **The bare chi-signature multiset FAILS on exact data**: harmonic at
  `q = 11` and equianharmonic at `q = 7` share `{-,-,-}` with opposite
  levels. (Signature class-invariance itself machine-checked: pass.)
- **Two rules survive all training rows**:
  1. `sig + orbit type` - abstains on generic-`{-,-,-}` (no training data;
     the two suspect generics at 23/31 are exactly the only such orbits);
  2. `sig + chi(j - 1728)` - total. Structural fact: **harmonic orbits have
     j = 1728 identically** (cJ2 = 0 at every field), so this rule extends
     the signature cleanly across types. It subsumes the mod-8 rule
     (harmonic signature `{chi(-1), chi(2), chi(2)}` is `{+,-,-}` exactly at
     `q = 5 (mod 8)`).
- **Pre-registered prediction sheet** (both rules, before the descent test):
  19-harmonic **LOW**; 19-eq, 29-harmonic, 31-eq, 37-harmonic **HIGH**;
  23-generic-c, 31-generic-b **HIGH** (per `sig+cJ2`; the type rule
  abstains).

## Step 2 - Descent Extension (family gap closed)

`--refine` adds coordinate descent over single-direction line swaps
(incremental sacrifice deltas), started from the top 200 pure-parabola
candidates plus 50 seeded random restarts per orbit; every refined winner is
independently rebuilt and verified.

**Validation: 11/11 MATCH-ALL.** The descent closes the one PHASE3L gap -
q=11 harmonic, pure 5 -> refined 4 = exact. The instrument now reproduces
every solver-certified value. Falsifier
`CONSTRUCTION_INSTRUMENT_MISMATCH`: **clear** (37/37 verified).

## Step 3 - Descent Test of the Prediction Sheet

With the validated instrument, **every evaluation-field HIGH reading is
descent-resistant** (250 starts each, none lowered):

| row | descent | rules predict | verdict |
| --- | :---: | :---: | --- |
| 19-eq | 9 | HIGH | consistent |
| 23-generic-c | 11 | HIGH (`sig+cJ2`) | consistent |
| 29-harmonic | 14 | HIGH | consistent |
| 31-generic-b | 15 | HIGH (`sig+cJ2`) | consistent |
| 31-eq | 15 | HIGH | consistent |
| 37-harmonic | 18 | HIGH | consistent |
| **19-harmonic** | **9** | **LOW** | **DIVERGENCE** |

The 19-harmonic row is the showdown: both surviving rules predict LOW
(trained on the exact `q = 11` harmonic, which has the identical signature,
type, and cJ2), but the gap-closing descent could not find an 8. Either the
descent missed it (the rules are right) or 19-harmonic is genuinely high -
in which case **both rules die, and no invariant in the battery can separate
11-h from 19-h** (they agree on every column), forcing a strictly deeper
invariant. Maximally informative either way.

## The Two q=19 Exact Solves (in flight)

Amendment D to the two-level probe (documented in-script): `--orbits` label
filter (control expectations restricted accordingly), `--reps 1` (lex-first
representative), `--budget` override. Smoke-tested at `q = 13` (harmonic
`ex = 6` reproduced, falsifier clear). Launched as two parallel runs, 40B-node
budgets:

```powershell
node scripts/kakeya-twolevel-law-probe.mjs --fields 19 --orbits harmonic --reps 1 --budget 40000000000 --out results/kakeya/twolevel-q19-harmonic
node scripts/kakeya-twolevel-law-probe.mjs --fields 19 --orbits equianharmonic --reps 1 --budget 40000000000 --out results/kakeya/twolevel-q19-equianharmonic
```

- harmonic solve: adjudicates the divergence above (8 = rules survive,
  9 = third consecutive pattern-generation death + battery exhaustion);
- equianharmonic solve: settles EQ-3 (both rules and descent say HIGH = 9).

Results will be appended as an addendum when the runs land (projected
5-24 h each; budget exhaustion reports bounded status honestly).

## Interpretation Boundary

Supports only:

> In the finite-field workbench plus sidecars (odd primes 5-37), the
> descent-augmented construction reproduces all eleven solver-certified
> 4-star excess values; two character-type invariant rules fit all exact and
> definitive-low data; every construction high is descent-resistant; the one
> rule-vs-instrument divergence (19-harmonic) is under exact adjudication.

Rule mining is exploratory (trained on 26 rows); the surviving rules are
banked conjectures, not claims. Construction values remain upper bounds.
`ex` imports the pinned Blokhuis-Mazzocca minimum. Register untouched, no
pins (sidecar fields). No Euclidean claim, no incidence-geometry novelty
claim.
