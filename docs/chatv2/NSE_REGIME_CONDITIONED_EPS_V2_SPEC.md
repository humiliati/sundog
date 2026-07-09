# Regime-Conditioned ε_K v2 — Inflation-Shell Guard (frozen)

> 2026-07-07. New registration after `NSE-H3-APPARATUS-REJECTED-B`
> (`NSE_REGIME_CONDITIONED_EPS_RECEIPT.md`): the v1 tercile-spread guard fired on
> the certified anchor because it conflated a true positive's natural U-shaped
> energy-disagree structure (~0.06 spread) with inflation. This v2 **redesigns the
> guard** — everything else (per-sample `ε_K(u)`, pair radius, certificate) is
> unchanged from v1. Non-promotional; a v2 positive is scoped to relative
> resolution and does not overturn Approach A's absolute-resolution null.

## 1. The defect v1 had, and the v2 fix

v1 terciled *all* witness pairs by energy and required a small disagree spread.
But a certified positive's disagree is naturally non-monotonic in energy
(anchor: [0.045, 0.0, 0.063]), so the raw spread rejected the control. The tercile
test cannot separate "benign energy-structure of a true witness" from "inflation."

**v2 isolates the inflation itself.** Inflation = the relative radius, where it
*exceeds* frozen `ε_K`, admitting genuinely-different states. So partition the
admitted relative-fiber witness pairs by signature distance `d`:

```text
CORE  pairs:  d ≤ ε_K_frozen            (also absolute-resolution fibers; trusted)
SHELL pairs:  ε_K_frozen < d ≤ relative_pair_radius   (added ONLY by the relative radius)
```

The SHELL is exactly the set of pairs the relative radius adds. If those pairs are
fiber-constant, the relative resolution is faithful; if they smear, the SHELL
disagree is elevated. **On a compact cell the relative radius equals frozen ⇒ the
SHELL is empty ⇒ the guard is inert** (this is what fixes the v1 false rejection,
by correct construction, not by loosening a tolerance).

## 2. The v2 guard (binding)

```text
disagree_core  = unique action-disagree fraction among CORE witness pairs
disagree_shell = unique action-disagree fraction among SHELL witness pairs
inflation_clean =
   None   if  shell_unique < MIN_SHELL_PAIRS   (no radius-added pairs to test -> no inflation)
   True   if  disagree_shell ≤ ABS_BAR  AND  disagree_shell ≤ disagree_core + MARGIN
   False  otherwise
```

**Frozen constants:**
- `ABS_BAR = delta_action = 0.10` — the SHELL pairs must be fiber-constant by the
  *same* standard the certificate's paired read uses (no new bar).
- `MARGIN = 0.05` — the SHELL may exceed the CORE disagree by at most this.
  **Calibrated from the control:** R0-v1 measured a certified anchor's natural
  disagree variation across energy sub-regions at ~0.05–0.06; the SHELL is allowed
  to differ from the trusted CORE by up to that measured natural variation before
  it counts as inflation. This is calibrate-from-control, not tune-to-pass — it is
  set from the anchor and no G=675 number has been read.
- `MIN_SHELL_PAIRS = 100` — power floor (mirrors `twin_min_unique_pairs`).

Gating: the guard PASSES iff `inflation_clean ∈ {True, None}`; it FAILS
(`= False`) only when the SHELL is powered and either sub-condition is violated.
The v1 tercile spread is still computed and **reported as a diagnostic only** (not
gating).

## 3. Regression gate (agent-run; R0-v2)

Re-run G=200 + G=300 with the v2 adjudicator (N = 50k). Required:

1. frozen `TWIN_STATE_CERTIFIED` on both;
2. relative `TWIN_STATE_RELATIVE_CERTIFIED` on both;
3. `f_rel ≥ 0.98`;
4. `|relative_disagree − frozen_disagree| ≤ 0.005`;
5. **`inflation_clean ≠ False`** (on compact cells the SHELL is empty ⇒ `None` ⇒
   pass; the guard must be inert here — the v1-failure case must now pass **by
   correct construction**).

Any failure ⇒ `NSE-H3-APPARATUS-REJECTED-B2`, final. (If condition 5 still
fails — i.e. the SHELL is non-empty on a supposedly compact cell and smears — that
is a genuine finding about the anchor, not a tolerance to move.)

## 4. R1 branches (owner-run, N = 200k, only if R0-v2 passes)

| R1 outcome | verdict |
| --- | --- |
| `f_rel < 0.10` | `NSE-H3-COVERAGE-SLIVER-RELATIVE` |
| covered, CERTIFIED, paired POSITIVE, **inflation_clean ∈ {True, None}** | `NSE-H3-FORCING-GENERAL-RELATIVE` (scoped to relative resolution + `f_rel`) |
| covered, CERTIFIED, paired POSITIVE, **inflation_clean = False** | `NSE-H3-APPARATUS-REJECTED-B2` (relative radius smears; positive is an artifact) |
| covered, CERTIFIED, paired NEG, inflation_clean ≠ False | `NSE-H3-GRASHOF-LOCAL` |
| covered, `NO_CERTIFICATE` | `NSE-H3-COVERAGE-WALL-CONFIRMED-RELATIVE` |

No gate widening, no constant retune after any read.

## 5. Harness + validation

The v2 guard is added to the existing `--adjudicator twin-state-relative`
(CORE/SHELL split in the same chunked pass; the tercile fields remain as
diagnostics). Validated before R0-v2 by: harness self-test, a non-verdict smoke,
and the synthetic self-test `--self-test-b` — which must now include the
**anchor-like compact case passing** (SHELL empty ⇒ `inflation_clean=None`), the
case that broke v1, plus a SHELL-inflation catch and a SHELL-clean positive.

## 6. Does not claim

Same as v1 §6: a v2 positive claims only control-sufficiency at relative
resolution on a fraction `f_rel` of one matched-Re cell, scoped and fenced; it
does not overturn A's null, touch H1/H2 or the closed slate, or make any
infinite-dimensional statement. `docs/chatv2/` stays no-publish.

Cross-refs: `NSE_REGIME_CONDITIONED_EPS_SCOPE.md` + `..._RECEIPT.md` (v1, the
rejected tercile guard), `NSE_COVERAGE_ADAPTIVE_APPARATUS_RECEIPT.md` (Approach A
null), `results/proof/c1-relative-reg-g{200,300}/` (the v1 R0 that measured the
anchor's natural disagree structure).

---

> **Post-run status (2026-07-07): R0-v2 PASS** (`NSE_REGIME_CONDITIONED_EPS_V2_
> RECEIPT.md`), via post-processing on the banked v1 samples (no re-integration).
> Both anchors: SHELL = 0 pairs ⇒ `inflation_clean = None` ⇒ pass — the guard is
> inert by construction where relative = frozen (v1 wrongly rejected the same data
> at spread 0.063/0.052). Synthetic self-test 3/3 incl. the v1-failure case now
> passing. **G=675 relative read unblocked**; R1-v2 is post-processable on
> `c1-h3-kf3-g675-adaptive/samples.npz` (no owner 2 h run).

> **FINAL (2026-07-07): R1-v2 ⇒ `NSE-H3-FORCING-GENERAL-RELATIVE` (fenced)**
> (`..._V2_RECEIPT.md` ▸ R1-v2). Branch fired positive per the frozen §4 table,
> but the SHELL is 0.099% of witness pairs (relative radius mostly *smaller* than
> frozen) ⇒ CORE-dominated; the operative change vs frozen is dropping the s_pos
> 0.50 gate, not the radius. Honest content: the ~45% of Φ_K-near pairs that exist
> at G=675 are control-sufficient (disagree 0.033), on a sparse support minority.
> Does NOT reveal scale-invariant structure, overturn A's null, or promote C1.
