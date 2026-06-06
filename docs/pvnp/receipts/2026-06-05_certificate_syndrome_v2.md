# Sundog Certificate Problem — Syndrome Certificate v2 Receipt (Stronger-ISD Ladder)

- Receipt id: `pvnp-certificate-syndrome-v2-2026-06-05`
- Phase / probe: Phase 4 / §5 capacity experiment — the syndrome/SIS certificate's
  measured capacity-relative one-wayness against a **Prange → Lee-Brickell(p=2) →
  Stern(p=2, l=8)** attacker ladder (tightening v1's Prange-only upper bound)
- Date run: 2026-06-05 (local); wall-time ≈ 2.5 h (Prange-dominated; **op-count is the
  cost signal, wall-time diagnostic-only**)
- Runner: `python scripts/pvnp-certificate-syndrome-v2.py --frozen` (frozen regime)
- Result dir: `results/pvnp/certificate-syndrome-v2/` (transient, gitignored)
- Frozen slate: [`SUNDOG_CERTIFICATE_SYNDROME_V2_SLATE.md`](../SUNDOG_CERTIFICATE_SYNDROME_V2_SLATE.md)
  (FROZEN 2026-06-05)
- Prediction lock: [`SUNDOG_CERTIFICATE_SYNDROME_V2_PREDICTION_LOCK.json`](../SUNDOG_CERTIFICATE_SYNDROME_V2_PREDICTION_LOCK.json)
  (sha256 `4e46be88…4374`, two-size-smoke-calibrated, frozen before the run)
- v1 receipt (Prange baseline): [`2026-06-04_certificate_syndrome_v1.md`](2026-06-04_certificate_syndrome_v1.md)
- `git_sha` at run: `544113518a37b78b29cba0ad2491f6bf43b15c1c`;
  `target_manifest.json` sha256 `77e97bdc…` (emitted before any attacker)

## Verdict

**Bounded-positive (measured attacker-hierarchy one-wayness) WITH a named LB↔Stern
model-deviation.** Two findings, both pre-registered and both honest:

1. **Bounded-positive headline (confirmed).** The find-vs-check separation holds for
   every attacker: each stronger attacker's witness-recovery threshold `C(ops)@50%`
   sits far above the flat 16,576-op verifier, and **all three measured `C` land within
   the locked factor-2 tolerance** (Prange 1.03×, LB 0.87×, Stern 1.64× of prediction).
   The big Prange→LB step resolves cleanly (≈86×). Witnesses are all valid; the
   privilege, label, and rank-fail audits pass. The absolute work-factor calibration
   was a success — Prange's measured per-iter (1.38×10⁶) matches the locked base
   (1.39×10⁶) almost exactly.

2. **Named model-deviation (the scientific finding).** The locked prediction's
   fine-grained claim that **Stern is the strongest attacker** is **falsified**.
   Measured **Lee-Brickell is the best** (drop **86×**, `C_best = 8.31×10⁷`), while
   **Stern underperforms** (drop **50×**, `C = 1.42×10⁸`, **1.64×** its locked `C`).
   The predicted **1.11× Stern-edge** became a measured **1.71× LB-edge**. Root cause is
   visible in the iteration cross-check (below): Stern's measured success-per-iteration
   deviates from its analytic work-factor heuristic at this regime (max |Δ| = **0.228**,
   ~2× slower than `1−(1−p)^B`), whereas Prange and LB track analytic (max |Δ| 0.055,
   0.093) and Stern itself *matched* analytic in the two smaller-`w` smoke regimes.

The reported one-wayness upper bound is therefore **`C_best = 8.31×10⁷` ops, set by
Lee-Brickell, not Stern.** This still tightens v1's Prange-only bound by ~86×.

## Frozen regime (as pre-registered)

`[n=128, k=64], w=12, τ=12`, GF(2), `code_seed=2026128` (the **same code as v1**),
`T=64` targets from the decoupled **`target_seed=2026220`** (sampled and persisted to
`target_manifest.json` *before* any attacker; attackers consumed only the syndrome `z`).
Full low-weight space `C(128,12)=2.37×10¹⁶` (non-enumerable). Frozen and lock-pinned
before the run; nothing was retuned.

## The measured capacity ladder (op-count = the cross-attacker unit)

| Attacker | measured `C(ops)@50%` | locked predicted | meas/pred | within 2× | drop vs Prange (meas) | drop (predicted) |
| --- | ---: | ---: | ---: | :---: | ---: | ---: |
| Prange (re-baselined) | **7.157×10⁹** | 6.982×10⁹ | 1.03 | ✓ | 1.0× | 1.0× |
| Lee-Brickell p=2 | **8.314×10⁷** | 9.594×10⁷ | 0.87 | ✓ | **86.1×** | 72.8× |
| Stern p=2, l=8 | **1.423×10⁸** | 8.663×10⁷ | 1.64 | ✓ | **50.3×** | 80.6× |

`C_best = min(C_LB, C_Stern) = 8.314×10⁷` ops **[Lee-Brickell]**; `C_Stern / C_LB =
1.71×` (Stern is the *more expensive* of the two — the reversal). Every measured `C` is
within the locked factor-2 band; the deviation is in the **LB↔Stern ordering**, not in
any single absolute magnitude.

## The LB↔Stern reversal — iteration cross-check (why Stern lost)

Each attacker's witness-recovery curve vs valid iterations `B`, measured vs the
pre-registered analytic `1−(1−p)^B`:

| Attacker | curve max \|Δ\| vs analytic | reads as |
| --- | ---: | --- |
| Prange | **0.055** | tracks analytic (as v1) |
| Lee-Brickell | **0.093** | tracks analytic; saturates to 1.0 by `B≈256` |
| Stern | **0.228** | **deviates** — rises ~2× slower than analytic |

Stern's analytic work factor (`p = C(k/2,p)²·C(n−k−l,w−2p)/C(n,w)`) is the standard ISD
heuristic, which assumes the redundancy-weight distribution is uniform. At the smaller
smoke regimes (`[80,40]`, `[120,60]`, both `w=8`) it matched measurement (max |Δ| 0.099,
0.118). At the frozen `[128,64] w=12` it is **optimistic by ~1.6× in iterations** — Stern's
real success-per-iteration is lower than the heuristic predicts, so its `C` lands 1.64×
high and it loses to plain Lee-Brickell. **The deviation is purely in the iteration count,
not the cost model:** all three measured per-iteration op-counts matched the locked
base+enum to within 2% (Prange 0.990×, LB 0.982×, **Stern 0.990×** — measured 1.82×10⁶ vs
locked 1.84×10⁶), so the two-size-smoke cost calibration was accurate for every attacker;
only Stern's *success-probability* heuristic was optimistic at this regime. (5/64 Stern targets did not reach first success
within `max_B=1000`, a heavier-than-geometric tail consistent with the heuristic
optimism; the `C@50%` median (32nd/33rd of 64) is finite and unaffected by the 5
censored at the top.) The slate already hedged that Stern's win is a *large-`w`*
phenomenon and that the `w=12` margin is compressed/within-noise — the measurement shows
the compression goes past parity into reversal. Because the slate deliberately did
**not** make LB→Stern a hard monotonicity tripwire, this is a **named model-deviation,
not a 6.5 quarantine**.

## The find-vs-check gap (the P-vs-NP-shaped result)

| Quantity | Measured |
| --- | ---: |
| Verifier check (flat, op-counted, unchanged from v1) | **16,576 ops** (`2·m·n` two matvecs + `n` wt-popcount + `m` equality) |
| `C_best` (best tested attacker = Lee-Brickell) | **8.314×10⁷ ops** |
| Find-vs-check gap at `C_best` | **≈ 5,015×** |
| Prange gap / LB gap / Stern gap | 431,773× / 5,015× / 8,584× |

`verifier_below_all_attackers = true` — no 6.1 vacuity / 6.4 overhead. A capacity-bounded
adversary below `C_best` cannot recover the deviation from `z`, while the verifier checks
a witness ~5,000× more cheaply even against the *strongest* tested attacker.

## Valid-iteration / rank-fail audit

| Attacker | valid iters | rank-fail draws | ops incl. overhead | success |
| --- | ---: | ---: | ---: | ---: |
| Prange | 529,316 | 1,304,317 | 7.305×10¹¹ | 64/64 |
| Lee-Brickell | 4,340 | 10,429 | 7.592×10⁹ | 64/64 |
| Stern | 13,139 | 32,239 | 2.395×10¹⁰ | **59/64** (5 censored at `max_B=1000`) |

Rank-fail overhead `ρ ≈ 3.46` per attacker (matches the GF(2) random-singularity rate),
folded into the measured ops as pre-registered.

## Witness validity audit

Every attacker, on success, exhibits a valid witness `e*` re-checked **`He*=z ∧
wt(e*)≤τ` using only public `H, z, τ`** (never the planted labels). All sampled
witnesses valid; all at `wt = 12 = τ` (at `[128,64] w=12` the planted weight-12 error is
effectively the unique `≤τ` witness, so the attackers recover the deviation itself).

## Privilege / label audit

`verifier_access_declaration.json`: every attacker received **only** `z` (from the
manifest's `attacker_visible` section); the scoring labels `s, e, wt(e)` live in the
manifest's `labels_only_scoring_fields` and are **never** passed into `attacker_run`
(audited at every call site by the pre-freeze audit). Manifest emitted and hashed before
any attacker ran. `code_valid (G Hᵀ=0) = true`, `labels_wt_ok = true`.

## v1 Prange cross-check (trials, not ops)

v2's Prange median is **≈5,186 valid trials** at 50% (`C_ops 7.157×10⁹ / per-iter
1.380×10⁶`), statistically consistent with v1's **≈5,007** trials (`N≈7224`) — the
instrumentation-independent cross-check **passes**. As pre-registered, the v2 Prange
*op-count* (7.16×10⁹) is **2.29× v1's** (~3.1×10⁹) **because v2 computes the full
systematic form `U·H`** (needed by LB/Stern) that v1 omitted — essentially the predicted
~2× factor; the cross-check is on trials, not ops.

## Gate results

| Gate | Result |
| --- | --- |
| Code identical to v1 + non-enumerable | pass (`n,k,w,τ,code_seed` = v1; `C(128,12)=2.37×10¹⁶`) |
| Target manifest before attackers | pass (emitted + hashed `77e97bdc…`; attackers read only `z`) |
| Code validity | pass (`G Hᵀ = 0`) |
| Verifier cheap + unchanged | pass (16,576 ops, flat, `≪ C_best`) |
| Witness validity | pass (every success exhibits `He*=z ∧ wt≤τ`) |
| Privilege / label audit | pass (attackers saw only `z`; labels scoring-only) |
| Valid-iteration audit | pass (per attacker; Stern 5/64 censored, disclosed) |
| Ladder ordering | **Prange→LB resolves cleanly (≈86×)**; **LB↔Stern reversed vs prediction (named model-deviation)** |
| Prediction match (absolute) | pass — all three `C` within the locked factor-2 tolerance |
| Determinism | by construction (seed-pinned pure-numpy) + the identical frozen-run code path was byte-identical on the throwaway harness-test re-run; a literal full-frozen re-run was not run (≈2.5 h, not load-bearing) |
| Cost reported | pass (per-attacker ops, the `C` ladder, find-vs-check gaps, rank audit) |

## Prediction-vs-measured (anti-p-hack)

The regime, seeds, attacker classes/parameters, op-count formulas, and the full
`prediction_lock.json` were frozen (sha256 `4e46be88…`) **before** the run; the harness
opens the lock read-only and computes the measured `C` before any lock reference. The
**absolute** predictions were confirmed (all within factor-2 — a calibration success),
and the **fine LB↔Stern ordering was falsified** (Stern predicted-best, LB measured-best).
That reversal is the finding, not a fit: a deviation far from the locked prediction is
itself the result, and one occurred — in the ordering, exactly where the slate flagged
the margin as compressed/within-noise.

## Claim boundary

`C_best = 8.31×10⁷` ops is an **upper bound** on the certificate's one-wayness against
the **tested attacker classes** (Prange/LB/Stern); a stronger class (BJMM/MMT/
representation) would lower it — a **new slate**, not a v2 edit. The measured best is now
**Lee-Brickell**, not Stern, at this regime. No cryptographic one-wayness claim; hardness
is **imported** (the SIS/decoding assumption), not proved. No claim that verification is
"in P"/"polynomial", no general alignment-verification claim, no body-resistance /
Sundog-regime-2 claim, no wall-time cheapness, and **no progress on P vs NP**. The result
is a control-substrate, certificate-discipline measurement on a non-enumerable but
bounded-scale regime. `invert-s` remains unconditionally vacuous (`2⁶⁴`/syndrome); spoof
remains structurally impossible (sound existence verifier).

## What this earns

On one constructed instance, the lane now has a **two-attacker-deep measured capacity
ladder** with a pre-registered prediction: checking is flat-cheap (16,576 ops), the
shadow is control-sufficient but loses the secret by algebra, and recovering the
deviation is capacity-hard with `C_best = 8.31×10⁷` ops — a find-vs-check gap of ≈5,015×
even against the strongest tested attacker. It also produced a genuine, pre-registered
**model-deviation**: the Stern ISD work-factor heuristic is optimistic at `[128,64] w=12`,
so plain Lee-Brickell is the better attacker there — a result that only a frozen,
calibrated, measured experiment could surface, and which a future slate could confirm
(re-run Stern with a corrected/measured success model, or test the large-`w` regime where
Stern's advantage is predicted to reappear).

## Pre-freeze audit

A 5-dimension adversarial pre-freeze audit (frozen-path-vs-spec, privilege/witness,
anti-p-hack/determinism, numerical-consistency, boundary-claims) returned **GREEN, zero
confirmed blockers**, with load-bearing claims re-verified at source lines, before this
run executed.

## Required-outputs bundle (under `results/pvnp/certificate-syndrome-v2/`)

`target_manifest.json`, `verifier_access_declaration.json`, `manifest.json`,
`capacity_ladder.json`, `prediction_vs_measured.json`, `valid_iteration_audit.json`,
`witness_validity_audit.json`, `op_count_report.json`, `attacker_ladder_curve.csv`,
`iteration_crosscheck.json`, `falsifier_summary.md`, `README.md`, plus the durable
`prediction_lock.json`. (Results dir is transient/gitignored; the load-bearing numbers
are reconciled in this receipt per the v1 artifact-packaging note.)
