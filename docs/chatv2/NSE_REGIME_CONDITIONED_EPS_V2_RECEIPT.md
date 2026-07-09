# Regime-Conditioned ε_K v2 (Inflation-Shell Guard) — Receipt

> 2026-07-07. Build + R0 regression of `NSE_REGIME_CONDITIONED_EPS_V2_SPEC.md`,
> the redesigned guard after v1's `NSE-H3-APPARATUS-REJECTED-B`. Non-promotional.

## The redesign (why v2 is not a tolerance loosening)

v1 terciled *all* witness pairs by energy and rejected on raw disagree spread —
which fired on the certified anchor because a true positive's disagree is
naturally non-monotonic in energy. v2 tests only the **SHELL** (witness pairs the
relative radius adds beyond frozen ε_K); on a compact cell relative = frozen ⇒ the
SHELL is *empty* ⇒ the guard is inert **by construction**. The v1 tolerance
(0.03) is unchanged and now non-gating (retired to a reported diagnostic).

## Build + synthetic validation (3/3)

`scripts/nse_coverage_adaptive_regression.py --self-test-b`:
- **B-T1 (the v1-failure case, now fixed):** compact anchor-like ⇒ SHELL = 0 ⇒
  `inflation_clean = None` ⇒ passes. v1's tercile guard wrongly rejected this.
- **B-T2 (inflation caught):** high-E cluster spread so pairwise sits between
  frozen ε_K and its relative radius (SHELL, 3,297 pairs) with random actions ⇒
  shell disagree 0.498 vs core 0.227 ⇒ `inflation_clean = False`.
- **B-T3 (shell-clean positive):** same geometry, constant action per cluster ⇒
  shell disagree 0.000 ⇒ `inflation_clean = True`.

## R0-v2 regression: PASS (post-processing on the banked v1 samples)

No re-integration — the v1 R0 exported `samples.npz`, and the v2 guard is pure
post-processing on the witness pairs (`--reprocess-b`, the "third apparatus
iteration = post-processing" path). Both cells:

| cell | frozen | relative | f_rel | \|rel−froz\| | SHELL pairs | inflation_clean | v1 would say |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G=200 | `TWIN_STATE_CERTIFIED` | `..._RELATIVE_CERTIFIED` | 1.0000 | 0.00000 | **0** | **None → pass** | False (spread 0.063) |
| G=300 | `TWIN_STATE_CERTIFIED` | `..._RELATIVE_CERTIFIED` | 1.0000 | 0.00000 | **0** | **None → pass** | False (spread 0.052) |

**`REGRESSION-B v2: PASS → G=675 relative read unblocked.`** The SHELL is exactly
empty on both anchors (all 693,774 / 942,830 witness pairs are CORE) — the guard
correctly abstains where there is no inflation to detect. This passes **by correct
construction, not by tuning**: the relative radius has added zero pairs on the
compact cells, so there is nothing for the anti-inflation test to flag.

## R1-v2 — application (the reopened G=675 relative read)

Also post-processable: the banked G=675 samples
(`results/proof/c1-h3-kf3-g675-adaptive/samples.npz`, same
`fallback_v7_g675_kf3` cell / seed / 200k samples) feed the v2 relative aggregator
directly — no owner-run re-integration needed. Branches (spec §4), read off `f_rel`
+ the paired verdict + `inflation_clean`:

| R1 outcome | verdict |
| --- | --- |
| `f_rel < 0.10` | `NSE-H3-COVERAGE-SLIVER-RELATIVE` |
| covered, CERTIFIED, paired POSITIVE, `inflation_clean ∈ {True, None}` | `NSE-H3-FORCING-GENERAL-RELATIVE` (scoped to relative resolution + `f_rel`) |
| covered, CERTIFIED, paired POSITIVE, `inflation_clean = False` | `NSE-H3-APPARATUS-REJECTED-B2` (smears) |
| covered, CERTIFIED, paired NEG | `NSE-H3-GRASHOF-LOCAL` |
| covered, `NO_CERTIFICATE` | `NSE-H3-COVERAGE-WALL-CONFIRMED-RELATIVE` |

## R1-v2 result (2026-07-07): `NSE-H3-FORCING-GENERAL-RELATIVE` — heavily fenced

Reprocessed on the banked G=675 samples (`--reprocess-b1`; ~30 min — the relative
pair set is large and the box was sharing cores, not the "seconds" the anchor took).
The branch fired **positive per the frozen §4 table** (no-rescue: recorded as
registered), but the mechanistic read makes it modest:

| field | value |
| --- | --- |
| verdict / paired / inflation_clean | `RELATIVE_CERTIFIED` / `PAIRED_...POSITIVE` / `True` |
| f_rel | **0.4527** (frozen candidate cov 0.4692 — frozen DEFERRED only because 0.469 < s_pos 0.50) |
| relative ε_K p05/p50/p95 | 0.0519 / **0.0563** / 0.0618 (frozen 0.0589) |
| CORE (within frozen ε_K) | 164,850 pairs, disagree **0.0333** |
| SHELL (relative-radius-added) | **163 pairs** (0.099% of witnesses), disagree 0.0184 |

**Four load-bearing fences (the honest reading):**

1. **The relative mechanism contributed almost nothing.** The relative radius is
   *mostly smaller* than frozen (p50 0.0563 < 0.0589, because mean E_low 0.64 <
   e_max 0.69), so the SHELL is a negligible **0.099%** of witness pairs. The
   positive is **CORE-dominated** — i.e. it rests on *frozen-radius* fibers, not
   on anything distinctively "relative." B is nearly degenerate with frozen here.
2. **The operative change vs frozen is dropping the s_pos gate, not the radius.**
   Frozen DEFERRED at coverage 0.469 *only* because 0.469 < s_pos = 0.50. B (like
   A) replaces that gate with the f_rel ≥ 0.10 scope fence and reads the witness on
   the covered ~45%. So the real content is: *the Φ_K-near pairs that exist are
   control-sufficient, read below the absolute coverage gate.*
3. **Support minority, sparse.** The certificate is on 45% of the support (< the
   frozen 0.50 majority bar), and Approach A already showed that support is sparse
   (dense-count median 0). These are ≥1-near (often 1–9-neighbour) pairs, not dense
   fibers — B's `NO`-s_pos read is exactly the minority case the frozen gate guards.
4. **The v2 guard is barely powered here** (SHELL 163 pairs, just over the 100
   floor) — but moot, since the SHELL is negligible to the positive regardless.

**What it does and does not say.** It says: on the ~45% of the matched-Re G=675
attractor where Φ_K-near pairs exist, the proxy is control-sufficient (disagree
0.033, anchor-like) and state-insufficient — the forcing-axis witness *exists on
the resolvable minority*. It does **not** say the relative resolution revealed
hidden scale-invariant structure (it didn't — SHELL 0.099%); it does **not**
overturn A's dense-fiber null (A asked a stricter question); it does **not** clear
the absolute coverage gate; and it is scoped to relative resolution + f_rel = 0.45,
proxy-relative, finite, `docs/chatv2/` no-publish. No C1 promotion, no
infinite-dimensional claim.

## H3 forcing axis — final three-apparatus picture

- **Frozen (≥1-near, s_pos 0.50):** DEFER at coverage 0.469 (a hair under the gate).
- **A (dense, k_min 10):** `COVERAGE-SLIVER` — no substantial dense fibers (median 0).
- **B v2 (relative, s_pos dropped):** `FORCING-GENERAL-RELATIVE`, fenced — the
  ~45% of near-pairs that exist are control-sufficient; relative radius immaterial.

Net: the matched-Re witness **exists on the covered minority but does not cover
enough of the support to clear the absolute gate, and has no dense-fiber core.**
Three apparatus, three consistent readings of one thin-but-constant structure.

Cross-refs: `NSE_REGIME_CONDITIONED_EPS_V2_SPEC.md`,
`NSE_REGIME_CONDITIONED_EPS_RECEIPT.md` (v1 rejection),
`NSE_COVERAGE_ADAPTIVE_APPARATUS_RECEIPT.md` (A null),
`results/proof/c1-relative-reg-g{200,300}/v2_reprocess_manifest.json`,
`results/proof/c1-h3-kf3-g675-adaptive/v2_reprocess_manifest.json` (this read).
