# Kakeya Phase 3K - Two-Level Law Probe (q = 17 Sidecar; q = 19 Staged)

- Artifact id: `KAK-PHASE3K-TWOLEVEL-PROBE`
- Date: 2026-07-07 (v1 timing run 2026-07-06)
- Status: internal out-of-register probe receipt. **q = 17 certified exact:
  the two-level law and generic uniformity hold; the harmonic mod-4 rule is
  FALSIFIED.** q = 19 staged as an owner-fired run.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Prior receipt:
  [`PHASE3J_EQUIANHARMONIC_PROBE.md`](PHASE3J_EQUIANHARMONIC_PROBE.md)
- Script:
  [`../../scripts/kakeya-twolevel-law-probe.mjs`](../../scripts/kakeya-twolevel-law-probe.mjs)
- Runs: controls
  [`../../results/kakeya/twolevel-law-probe-controls/manifest.json`](../../results/kakeya/twolevel-law-probe-controls/manifest.json),
  q17 [`../../results/kakeya/twolevel-law-probe-q17/manifest.json`](../../results/kakeya/twolevel-law-probe-q17/manifest.json),
  v1 bounded timing run kept at `results/kakeya/twolevel-law-probe-timing/`

## Register Note

As in PHASE3J: `q in {17, 19}` are **not** added to the workbench.
`SUPPORTED_Q`, the Phase-2 spec, the UI, and the regression suite (78/78,
no new pins) are untouched. `ex` is relative to the pinned Blokhuis-Mazzocca
odd-`q` minimum (`BM(17) = 161`, witness anchor verified: size 161, complete,
sacrifice exactly 8 as triple points).

## Verdict

**q = 17 (certified exact, all six solves): every orbit sits at the LOW
level.**

| `q` | harmonic | equianharmonic | generic |
| ---: | ---: | ---: | ---: |
| `5` | 2 (high) | - | - |
| `7` | 2 (low) | 3 (high) | - |
| `11` | 4 (low) | - | 4 (low) |
| `13` | 6 (high) | 5 (low) | 5 (low) |
| `17` | **7 (low)** | - | **7 / 7 (low, both orbits)** |

- **TL-1 CONFIRMED and strengthened**: `ex in {(q-3)/2, (q-1)/2}` now holds
  **11/11** measured orbit-field pairs.
- **HM-1 FALSIFIED**: the harmonic orbit was predicted at the full budget
  (`8`) for `q = 1 (mod 4)`; it is exactly `7`. The harmonic mod-4 rule dies
  at its first out-of-sample test - the second consecutive pattern to do so
  (after PHASE3J's equianharmonic conjecture).
- **GU-1 CONFIRMED**: the first field with *two* generic orbits, and they
  agree (both `7`) - "generic" is uniform at `q = 17`.
- **EQ-3 PENDING**: the equianharmonic orientation test lives at `q = 19`
  (staged below).

## Pattern Status After This Kill (read with suspicion)

The surviving post-hoc observation is now: harmonic is high exactly at
`q in {5, 13}` = the measured fields with `q = 5 (mod 8)` (equivalently:
`-1` a QR and `2` a non-QR), low at `q in {7, 11, 17}` - 5/5. For the
equianharmonic orbit, both roots of `l^2 - l + 1` are non-QRs at `q = 7`
(high) and QRs at `q = 13` (low) - 2/2. **This is the third pattern
iteration on this data; each predecessor died on its first new field. Treat
as a forking-paths artifact until it survives a pre-registered test.**
Discriminators: `q = 19` (mod 8 class 3 -> harmonic low; equianharmonic
roots `{8, 12}` both non-QR -> high, which coincides with EQ-3's
anti-correlation prediction) and `q = 29` (the next `5 (mod 8)` prime ->
harmonic high is the sharp bet).

## Instrument (amendments from the v1 run, all documented in-script)

The v1 timing run (4.0h, six q=17 solves) **exhausted its 500M-node budget on
every solve** - falsifier correctly fired; artifacts kept. Its constructive
completions already certified `ex <= 7` (falsifying HM-1 outright); exactness
needed:

- **Amendment A**: probe-field reps 4 -> 2 (lex-first 1 + lex-last 1). Rep
  invariance is theorem-backed (PGL-equivalence of same-orbit stars) and was
  audited 580/580 at `q <= 11`; one exact solve per orbit determines the
  level.
- **Amendment B**: star-pivot scaling symmetry. The GL-stabilizer of four
  distinct directions is the scalars, and scaling multiplies every chosen
  intercept by `lambda`, so any completion is equivalent to one whose root
  intercept is `0` or `1` - root branching `q -> 2`, an ~8x tree cut, sound
  precisely for origin-star bodies. **Controls revalidated exact through the
  symmetric solver in 22.6 s** (`2/3`, `4/4`, `6/5/5`).
- **Amendment C**: node budget 500M -> 2B.

Result: all six q=17 solves exact at 192M-526M nodes (~2.0h total). The
falsifier `TWOLEVEL_INSTRUMENT_MISMATCH` (instrument-only: budget, intra-orbit
variation, control mismatch, inventory mismatch, witness anchor) is **clear**
on both receipt runs.

## Executable Receipt

Two runs (identical code path; the default command is now
controls + q17 in one invocation, ~2h):

```powershell
node scripts/kakeya-twolevel-law-probe.mjs --fields 7,11,13 --out results/kakeya/twolevel-law-probe-controls
node scripts/kakeya-twolevel-law-probe.mjs --fields 17 --out results/kakeya/twolevel-law-probe-q17
```

Outputs:

```text
KAK_TWOLEVEL_PROBE fields={7,11,13} q7=harmonic:ex=2+equianharmonic:ex=3 q11=harmonic:ex=4+generic:ex=4 q13=harmonic:ex=6+generic:ex=5+equianharmonic:ex=5 controls=pass falsifier=clear out=results/kakeya/twolevel-law-probe-controls
KAK_TWOLEVEL_PROBE fields={17} q17=harmonic:ex=7+generic-a:ex=7+generic-b:ex=7 controls=pass TL-1_q17=true HM-1_q17=false GU-1_q17=true falsifier=clear out=results/kakeya/twolevel-law-probe-q17
```

(`controls=pass` is vacuous in a fields-17-only run; the controls run above is
the non-vacuous control certificate. `npm run kakeya:twolevel` executes the
combined default.)

## Staged: q = 19 (owner-fired, Augury-G3 pattern)

Projected multi-day at the current solver (depth 16, ~20x per field step from
the measured 192M-526M nodes at q=17). Command, unchanged code:

```powershell
node scripts/kakeya-twolevel-law-probe.mjs --fields 7,11,13,19 --out results/kakeya/twolevel-law-probe-q19
```

Pre-registered for that run: TL-1 (`ex in {8, 9}`), HM-1-successor (harmonic
`8`, low - both the dead mod-4 rule and the surviving mod-8 observation agree
here, so q=19 does NOT discriminate them for harmonic; it discriminates the
equianharmonic orientation), GU-1 (generics equal, low), EQ-3 (equianharmonic
`9`, high). If a solve exhausts 2B nodes, the run reports bounded status
honestly; a budget raise is a documented amendment, not a silent change.

## Interpretation Boundary

Supports only:

> In the finite-field workbench plus out-of-register sidecars (odd primes
> 5-17), every measured 4-star cross-ratio orbit has excess `(q-3)/2` or
> `(q-1)/2` (11/11); the harmonic mod-4 rule is falsified at `q = 17`;
> generic orbits are uniform at the one field measured with two of them.

The mod-8 / quadratic-character observations are third-iteration post-hoc
patterns over 11 points with two dead predecessors - banked, explicitly
suspect, testable at the staged fields. `ex` values import the pinned BM
minimum. Report-only; register untouched; no Euclidean claim, no
incidence-geometry novelty claim.
