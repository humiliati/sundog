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

## Addendum 2026-07-08 - q=19 harmonic solve: HIGH, rules dead, battery exhausted

**19-harmonic = ex 9 = HIGH, exact** (`solverExact`, falsifier clear,
12,240,259,365 nodes, ~18.4 h). The showdown resolves against the invariant
rules:

- **The descent was right, the rules were wrong.** The descent-augmented
  construction returned 9 and could not find 8 - because 8 does not exist.
  Both surviving rules predicted LOW(8). The construction now reproduces the
  exact optimum at **12/12** known field-orbits, upgrading it from
  "upper bound with occasional gaps" toward a reliable optimum-finder (its
  other evaluation highs gain credibility, though they remain formally UB
  until solved).
- **Re-mining with 19-harmonic promoted to EXACT: all eight battery rules go
  inconsistent** (`npm run kakeya:invariant-hunt`). The character battery is
  exhausted.

**Correction to the pre-registration wording.** The Step-3 note said "11-h and
19-h agree on every column"; that is not literally true. They agree on
signature `{-,-,-}`, type (harmonic), `chi(j-1728) = 0`, `chi(-1)`, and
`chi(2)` - but **differ on `cJ = chi(j)`** (`chi(1) = +1` at `q=11` vs
`chi(18) = chi(-1) = -1` at `q=19`). Exhaustion holds for a subtler reason:
`cJ` is the *only* character coarsening in the battery that separates the
critical pair, and every rule employing `cJ` is inconsistent on other
training rows (it was `consistent=false` before the promotion). So no
conjunction of battery invariants is a consistent classifier - the level is
not a function of the orbit's quadratic-character data.

**What this forces.** The complete PGL invariant `j` does separate the pair
(1 vs 18) - but that is the orbit label, not a predictive rule. A genuine
level rule needs arithmetic of `j` finer than its quadratic character:
candidates for a PHASE3N reopener are higher-power residue symbols of `j`
(cubic/quartic), or the level being genuinely not orbit-determined (a
field-global term). Banked, not claimed.

**Two-level law survives**: `9 in {8, 9} = {(q-3)/2, (q-1)/2}` - the excess is
still two-level; only the *which-level* character rule died.

**EQ-3 (equianharmonic) solve: ex 9 = HIGH, exact** (`solverExact`, falsifier
clear, 12,560,647,473 nodes, ~18.7 h). EQ-3 confirmed: the equianharmonic
roots `{8, 12}` are both non-QR mod 19, and the level is HIGH, matching the
within-equianharmonic sub-pattern (roots-QR -> low, roots-non-QR -> high),
which now reads high at `q in {7, 19}` (exact) + `31` (UB), low at
`q in {13}` (exact) + `37` (UB-def-low). Two-level law holds (`9 in {8, 9}`).

**Scope of the battery death vs the equianharmonic sub-pattern.** The
exhaustion above concerns the *global* classifier over all orbit types; its
fatal collision is 11-harmonic (LOW) vs 19-harmonic (HIGH), both harmonic.
The *within-equianharmonic* roots-character sub-pattern is untouched by that
collision and remains consistent on its five data points - a smaller,
still-live conjecture. (The harmonic mod-8 rule, by contrast, is dead:
`19 = 3 (mod 8)` predicted low, actual high.)

**Both q=19 solves complete; the PHASE3M empirical leg is closed.** Net: the
descent-augmented construction is exact at 13/13 solver-known field-orbits;
the two-level law is exact-confirmed at all of `q <= 19` and holds in
verified upper bounds through `q = 37`; the global character classifier is
falsified; one within-type sub-pattern survives. The open thread is the
PHASE3N deeper-invariant question (owner-gated).
