# Kakeya Phase 3J - Equianharmonic Structure Probe (q = 13 Sidecar)

- Artifact id: `KAK-PHASE3J-EQUIANHARMONIC-PROBE`
- Date: 2026-07-06
- Status: internal out-of-register probe receipt; executes the PHASE3I reopen
  condition. **Falsifier-fenced null: the equianharmonic conjecture is
  FALSIFIED**, replaced by a sharper observed structure.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Prior receipt:
  [`PHASE3I_CROSSRATIO_ORBIT_SWEEP.md`](PHASE3I_CROSSRATIO_ORBIT_SWEEP.md)
- Litpass anchor:
  [`../KAKEYA_LITPASS_MEMO.md`](../KAKEYA_LITPASS_MEMO.md) (Blokhuis-Mazzocca
  pin covers general odd `q`, so `BM(13) = 97` is the same import)
- Script:
  [`../../scripts/kakeya-equianharmonic-probe.mjs`](../../scripts/kakeya-equianharmonic-probe.mjs)
- Manifest:
  [`../../results/kakeya/equianharmonic-probe/manifest.json`](../../results/kakeya/equianharmonic-probe/manifest.json)

## Register Note (read first)

`q = 13` is **not** added to the workbench. `SUPPORTED_Q`, the Phase-2 spec
lock, the UI, and the regression suite (still 78/78, no new pins) are
untouched. The core geometry functions are total in `q`, so this sidecar
passes `q = 13` explicitly as a one-hypothesis instrument. That is why this
receipt adds no workbench pins - deliberately.

## Verdict

**The equianharmonic conjecture is falsified.** PHASE3I's pattern - "the
deviating orbit is the equianharmonic one" - predicted that at `q = 13` (the
next field where `-3` is a QR, so the orbit exists) the equianharmonic orbit
would deviate from a common harmonic/generic value. Measured:

| `q` | harmonic | equianharmonic | generic |
| ---: | ---: | ---: | ---: |
| `7` (control) | `2` | `3` | - |
| `11` (control) | `4` | - | `4` |
| `13` (probe) | **`6`** | `5` | `5` |

- **EQ-1 (harmonic = generic at 13): FALSE** - harmonic `6`, generic `5`.
- **EQ-2 (equianharmonic deviates): FALSE IN INTENT** - the equianharmonic
  orbit sits *with* generic at `5`; the deviator at `q = 13` is the
  **harmonic** orbit. (The pre-registered boolean, worded as "differs from
  the common value", technically fires only because there is no common
  value; the conjecture it encoded is dead.)
- **EQ-2' (+1 magnitude): FALSE** as posed.

## The Sharper Observed Structure (pattern, not claim)

Folding in PHASE3I's `q = 5` row, all **8 of 8** measured orbit-field pairs
satisfy:

```text
ex(4-star orbit)  in  { (q-3)/2 , (q-1)/2 }   =   { budget - 1 , budget }
```

- the 4-star excess never exceeds the full-pencil concurrency tax `(q-1)/2`
  (the PHASE3F/3H budget), and never drops more than one below it;
- **the harmonic orbit pays the full budget exactly when `q = 1 (mod 4)`** -
  q=5 high, q=7 low, q=11 low, q=13 high (4/4) - i.e. exactly when its own
  cross-ratio `-1` is a quadratic residue;
- the equianharmonic orientation is *undetermined*: high at `q = 7`, low at
  `q = 13` (two data points with opposite signs - the naive
  "equianharmonic-high-iff-q=3-mod-4" reading has exactly as much data as the
  conjecture this probe just killed);
- generic is low at both fields where it exists (`11`, `13`).

Discriminating next fields, banked (each would be a new pre-registered
out-of-register sidecar): **`q = 17`** (`1 mod 4`, no equianharmonic, and -
new regime - **two** distinct generic orbits: tests harmonic-high plus whether
"generic" is uniform within a field) and **`q = 19`** (`3 mod 4`, with an
equianharmonic orbit: tests both remaining orientations).

## Instrument

- Orbit inventory at `q = 13` from field arithmetic: harmonic `{2,7,12}`,
  equianharmonic `{4,10}` (`-3 = 6^2 mod 13`), generic `{3,5,6,8,9,11}`;
  quadruple counts `273/182/546` summing to `C(14,4) = 1001`. Measured keys
  match.
- 8 deterministic representatives per orbit (lexicographic first 4 + last 4;
  inf-containing quadruples included at every field), 56 solves total, **all
  exact** (worst case 8.96M nodes vs the 200M budget) via the PHASE3G solver.
- `ex` constant across representatives within every orbit (invariance spot
  check; the theorem was audited 580/580 in PHASE3I).
- `q = 13` anchor: the parabola witness works at any odd prime - size `97 =
  BM(13)`, complete, sacrifice exactly `6 = (q-1)/2` as triple points.
- Controls: `q in {7, 11}` re-derived through the same code path reproduce
  PHASE3I exactly (`2/3` and `4/4`).

Falsifier `EQUIANHARMONIC_INSTRUMENT_MISMATCH` (instrument-only; hypothesis
outcomes cannot fire it): budget exhaustion, intra-orbit variation, control
mismatch, inventory mismatch, or witness failure. **Clear.**

## Executable Receipt

Command:

```powershell
npm run kakeya:equianharmonic
```

Output:

```text
KAK_EQUIANHARMONIC_PROBE fields={7,11,13} q7=harmonic:ex=2+equianharmonic:ex=3 q11=generic:ex=4+harmonic:ex=4 q13=equianharmonic:ex=5+generic:ex=5+harmonic:ex=6 witness_q13=pass controls=pass EQ1=false EQ2=true EQ2'=false falsifier=clear out=results\kakeya\equianharmonic-probe
```

(Build note: the first CSV emission lacked field quoting, so the comma-bearing
six-set column shifted on parse; fixed to the lane's `csvValue` quoting before
this receipt. Manifest values were computed in-memory and unaffected.)

## Interpretation Boundary

Supports only:

> In the finite-field workbench plus a `q = 13` out-of-register sidecar (odd
> primes 5-13), the 4-star embeddability excess per cross-ratio orbit always
> equals the concurrency budget or budget-minus-one; the equianharmonic-
> deviator conjecture is falsified; the harmonic orbit's level tracks
> `q mod 4` over the four measured fields.

The two-level law and the `q mod 4` pattern are observations over 8 and 4
data points respectively - banked conjectures, not claims. `ex` at `q = 13`
is relative to the pinned Blokhuis-Mazzocca minimum (odd-`q` formula,
imported). Report-only; the workbench register is untouched; no Euclidean
claim, no incidence-geometry novelty claim.

Reopen conditions on record: `q = 17` and `q = 19` sidecars (new
pre-registrations, owner-gated) for the two-level law, the harmonic
`q mod 4` rule, the equianharmonic orientation, and generic uniformity.
