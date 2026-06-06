# Sundog Certificate Problem — Syndrome Certificate v2 Stronger-ISD Slate

Status: **DRAFT — opened for review, NOT frozen.** No v2 attacker run may execute
against the frozen regime until this slate is frozen and each attacker has been
smoke-validated on a throwaway regime. (Freeze-before-execute, per the lane.)

Date opened: 2026-06-04

This is the **stronger-attacker** successor to
[`SUNDOG_CERTIFICATE_SYNDROME_V1_SLATE.md`](SUNDOG_CERTIFICATE_SYNDROME_V1_SLATE.md)
(Prange ISD → bounded positive, measured `C`, receipt
[`receipts/2026-06-04_certificate_syndrome_v1.md`](receipts/2026-06-04_certificate_syndrome_v1.md)).
v1's measured `C` was a threshold **against Prange ISD** — an explicit upper bound.
v2 tightens that bound: it measures the witness-recovery threshold against a ladder
of stronger attackers on the same frozen code/regime with a new frozen decoupled
target manifest, and tests whether `C` drops as the better-attack work factors
predict.

## What v2 measures

On the **same frozen `[128,64] w=12` code/regime as v1**, with a fresh decoupled
`T=64` target manifest shared by every attacker, v2 measures the witness-recovery
(invert-`e`) capacity threshold `C` for a **ladder of attacker classes** — Prange
(the baseline algorithm, re-run on the v2 target manifest), Lee-Brickell (p=2), and
Stern (p=2, l=8). The single question:

> Does `C` drop monotonically as attacker strength increases, by the predicted
> work factors? (Stronger finder → lower `C` → tighter upper bound on the
> certificate's one-wayness.)

**Target manifest + decoupled sampling (load-bearing).** v1 did not emit a target
manifest, and its harness **interleaved target sampling with Prange attack draws on a
single RNG stream**, so the v1 target set cannot be reconstructed without re-running
the full ~82-min v1 stream with instrumentation (the per-target draw count depends on
the attack). v2 therefore **decouples target sampling from attacks**: a dedicated,
frozen `target_seed` (below) samples the `T=64` targets *before any attacker runs*,
and the harness persists `target_manifest.json` — per target: id, the syndrome `z`,
and the scoring labels (`s`, `e`, `wt(e)`) in a clearly-marked **labels-only** section
that no attacker reads. All three attackers consume the manifest's `z` values only, so
"same targets for all attackers" is **auditable by construction** and the v1
interleaving flaw is fixed going forward.

Because these are a fresh, cleanly-sampled target set (not v1's interleaved one), v2
**re-baselines Prange in-ladder** — Prange is re-run on the v2 targets as the ladder's
own baseline. v1's measured Prange `C` (≈5007 trials / ~3×10⁹ ops at 50%) is then a
**statistical cross-check** the v2 Prange must be consistent with (same regime, fresh
targets → statistically equal `C`), not the baseline itself. This stays within v2 (a
clean target set, no new slate id needed).

The certificate, the existence safety predicate `Safe(y) := ∃ e* : He*=Hy ∧
wt(e*)≤τ`, the witness-verifier, and the `invert-s`-is-vacuous / spoof-is-structural
facts are **unchanged from v1**. Only the attacker class is strengthened.

## Capacity unit: operations

The attackers do different work per iteration (Stern does more per iteration but
far fewer iterations), so the only fair cross-attacker capacity unit is **operations**.
`C_attacker := the measured op-count budget at which that attacker reaches 50%
witness recovery` over the `T` targets. Each attacker also reports its natural
iteration budget and its valid-iteration audit. The robust ladder claim is that
`C_Prange > C_LB` resolves cleanly in ops; the LB→Stern step is small (~1.11× at
`w=12`, locked) and is within `T=64` noise, so it is reported as separately resolved or
within-noise rather than used as a hard monotonicity tripwire.

## Frozen regime (code identical to v1; targets decoupled)

`[n=128, k=64], w=12, τ=12`, GF(2), `code_seed=2026128` (the **same code as v1**),
`T=64` targets sampled from a dedicated, decoupled **`target_seed = 2026220`** (a
fresh mechanical seed, independent of any attacker RNG), persisted to
`target_manifest.json` before any attacker runs. Full enumeration space
`C(128,12)=2.37×10¹⁶` (non-enumerable). The same code + the one frozen target set
shared by all three attackers is what makes the cross-attacker `C` comparison
apples-to-apples; changing the code, `(n,k,w,τ)`, or `target_seed` voids the
comparison. (v1's body seed base `7000000` is *not* reused for targets — its target
stream was interleaved with attacks; see the decoupling note above.)

## The attacker ladder (frozen, named classes)

All three find a light same-syndrome witness `e*` (`He*=z`, `wt(e*)≤τ`) given only
`z`. **Valid-iteration convention** (generalizing v1's rank-valid rule): each
attacker counts only iterations whose required Gaussian-elimination step is
full-rank; rank-deficient draws are charged to measured ops and audited as
`rank_fail_draws`, **not** counted as iterations. The analytic curves are read
against valid iterations; the cross-attacker comparison is in measured ops. For
Stern, a full-rank draw with zero useful list collisions is still a valid iteration
and still charged to ops; collision/list work is never free.

1. **Prange** (baseline algorithm, re-run on the v2 target manifest): random size-`k`
   information set; candidate error supported on the `n−k` complement; success iff
   its weight `≤ τ`.
2. **Lee-Brickell, p=2** (frozen `p=2`): Prange, but allow exactly `p=2` error
   positions inside the information set — enumerate the `C(k,2)` weight-2 info-set
   patterns, add each to the candidate, accept if total weight `≤ τ`.
3. **Stern, p=2, l=8** (frozen `p=2, l=8` — the cost-model-optimal window; the draft's
   `l=4` was suboptimal because it left ~16× more list collisions to score per
   iteration): split the information set into two halves of size `k/2`; on an `l=8`-bit
   window of the redundancy, build the two lists of weight-`p` half-patterns and their
   window-syndromes; for each collision on the window, form the full candidate and
   accept if weight `≤ τ`.

No MMT/BJMM/representation attacks in v2 (a future slate). Implementations are frozen
to these algorithms with these parameters; each must reproduce a valid witness
(`He*=z`, `wt≤τ`) before its curve is trusted.

## Pre-registered prediction (the comparator)

**Locked** work factors on `[128,64] w=12`, from the two-size throwaway-smoke
calibration (`prediction_lock.json`; see Locked formulas). Per-iteration cost
decomposes exactly as `per_iter = base(m) + enum`: `base(m)` is the shared
systematic-form work (Gaussian elimination + the `U·H`/`U·z` matmul) **with the
measured rank-fail overhead `ρ≈3.46` folded in**, fit empirically as
`base(m)=3.95·m^3.07` on two throwaway sizes (`[80,40]`, `[120,60]`; `m=40,60`) →
`base(64)=1.39×10⁶` ops; `enum` is the attacker-specific enumeration, analytically
op-counted (`0` for Prange, `C(k,p)(p+1)m` for LB, `2C(k/2,p)pl +
(C(k/2,p)²/2^l)(3p+1)m` for Stern) and validated against measurement at both smoke
sizes (within 2–11%). The draft's gauss-only `(n−k)³=2.62×10⁵` per-iter estimate
omitted the `U·H` matmul (`≈2m³`, so the base scales `~m³` — fit β=3.07) and the
rank-fail overhead, making the calibrated base ~5× higher; the **drops** (ratios,
instrumentation-invariant) are what the experiment tests.

| Attacker | iters `N` | per-iter ops | work `C`(ops)@50% | `C` drop vs Prange |
| --- | ---: | ---: | ---: | ---: |
| Prange (re-baselined) | 7,224 | 1.39×10⁶ | **6.98×10⁹** | 1× |
| Lee-Brickell p=2 | 78 | 1.78×10⁶ | **9.59×10⁷** | **≈ 72.8×** |
| Stern p=2, l=8 | 68 | 1.84×10⁶ | **8.66×10⁷** | **≈ 80.6×** |

**Locked expected ladder:** `C_Stern (~81×) < C_LB (~73×) < C_Prange (1×)` — monotone,
but **compressed at this small `w`: Stern beats LB by only ~1.11×** (`C_best = min(C_LB,
C_Stern) = 8.66×10⁷` ops, Stern-set), because Stern's collision overhead nearly cancels
its iteration advantage when `w=12`. (Stern's dramatic win is a *large-`w`* phenomenon —
e.g. `[256,128] w=24`; that is the scaled-regime lever, a separate slate.) The
compressed LB↔Stern margin means the `T=64` sampling noise cannot be expected to resolve
a ~1.11× gap, so the LB↔Stern ordering is reported as **within-noise (not separately
resolved)** rather than as a violated monotone gate. The run measures each attacker's
witness-recovery curve vs an op budget, locates each 50%-breakpoint `C(ops)`, and reports
whether the measured drops match the locked prediction; a measured drop far from it is
itself the finding.

**Freeze requirement (status: prediction produced):** the table above is the **locked**
prediction, recorded machine-readably in `prediction_lock.json` (produced by the two-size
throwaway smoke — per-attacker success-probability formula, valid-iteration definition,
op-count formula, calibrated `base(m)` fit + analytic `enum`, predicted 50% `C(ops)`,
`C_best`, and the frozen tolerance). Before the frozen run, copy `prediction_lock.json`
from the transient `results/` dir to a durable committed location (alongside this slate or
in the receipt). Updating constants after seeing frozen-target results voids the run.

### Locked formulas (the `prediction_lock.json` contract)

For each attacker: a **success-probability per valid iteration** `p` (pure
combinatorics, no free constants), an **iteration count** `N = 1/p`, a **per-iteration
op-count** `per_iter = base(m) + enum`, and the predicted **50% threshold**
`C(ops) = N·ln2·per_iter`. `base(m)` is the shared gauss+matmul work **with the measured
rank-fail overhead `ρ≈3.46` folded in** (Prange's per-iter is the base probe, since Prange
does no enumeration); `enum` runs once per valid iteration and is the analytic op-count
(not ρ-inflated, so ρ is not applied a second time). The `base(m)=α·m^β` fit and the
`enum` op-counts are calibrated on the **two-size throwaway smoke only** and then locked;
they may not move after a frozen target is scored.

- **Prange:** `p = C(n−k,w)/C(n,w)`; `per_iter = c_gauss`.
- **Lee-Brickell (p):** `p = C(k,p)·C(n−k,w−p)/C(n,w)`; `per_iter = c_gauss +
  C(k,p)·c_score`.
- **Stern (p,l):** `p = C(k/2,p)²·C(n−k−l,w−2p)/C(n,w)`; `per_iter = c_gauss +
  2·C(k/2,p)·c_list + (C(k/2,p)²/2^l)·c_collide`. The collision term
  `(C(k/2,p)²/2^l)·c_collide` is mandatory and is what makes `l=8` optimal at `w=12`.

Frozen parameters: `LB p=2`; `Stern p=2, l=8`. The **locked** (two-size-smoke-calibrated)
50% `C(ops)` are Prange `6.98×10⁹`, LB `9.59×10⁷` (≈72.8×), Stern `8.66×10⁷` (≈80.6×),
with `C_best = 8.66×10⁷` (Stern-set, LB/Stern ≈1.11×) and the frozen tolerance band, all in
`prediction_lock.json`. (The earlier gauss-only draft — `c_gauss≈(n−k)³`,
`c_score=(p+1)m`, `c_collide=(3p+1)m`, `c_list=l` — gave `1.9×10⁹ / 3.0×10⁷ / 2.3×10⁷`; it
omitted the `U·H` matmul and rank-fail overhead, hence the ~5× higher calibrated base.)

## Primary gates

| Gate | Required |
| --- | --- |
| Code identical to v1 | `(n,k,w,τ,code_seed)` byte-equal to v1; `target_seed=2026220` frozen |
| Target manifest | the `T=64` targets sampled from `target_seed` and persisted to `target_manifest.json` **before** any attacker runs; every attacker reads only `z` from it; the labels-only section is never an attacker input |
| Code validity | `G Hᵀ = 0` |
| Verifier cheap + unchanged | check op-count flat (`16,576` ops, as v1) `≪ C` for every attacker |
| Witness validity | every attacker, on success, exhibits a valid `e*` (`He*=z`, `wt≤τ`) — verified |
| Privilege / label audit | every attacker receives **only** `z`; planted `e`/`wt(e)`/`s` scoring-only |
| Valid-iteration audit | per attacker: `rank_fail_draws`, valid iterations, measured ops (incl. overhead) |
| Ladder ordering | `C_Prange > C_LB` resolved cleanly (the ~73× step is large); `C_LB ≥ C_Stern` consistent with the locked prediction, with LB↔Stern reported as separately-resolved **or within-noise** (the ~1.11× step is expected to be unresolvable at `T=64`) — within-noise is a pass, not a violation; each curve a clean `0→1` transition |
| Prediction match | each measured `C(ops)` within the frozen `prediction_lock.json` tolerance |
| v1 Prange cross-check | the v2 Prange's **50% valid-iteration count** is consistent with v1's (`≈5007` trials, `N≈7224`) — this is instrumentation-independent. The v2 Prange **op-count** is *not* expected to equal v1's: v1 solved `e_J=H_J⁻¹z` without the full systematic form `U·H`, whereas v2 re-baselines Prange *with* `U·H` (needed by LB/Stern), so v2's Prange `C(ops)` (~6.98×10⁹) is ~2× v1's (~3×10⁹). Cross-check on trials, not ops |
| Determinism | byte-identical re-run under the frozen seeds (or established by construction + smoke, per v1) |
| Cost reported | per-attacker ops, the `C` ladder, the find-vs-check gap at each `C` |

## Verdict branches

- **bounded-positive measured attacker-hierarchy one-wayness** — the three curves
  rise `0→1`; the large Prange→LB step resolves cleanly; the LB→Stern step is either
  separately resolved in the predicted direction or explicitly reported as
  within-noise under the frozen tolerance; the verifier stays cheap; and the audits
  pass. Report the tightened upper bound against the **best tested attacker class**
  (`C_best = min(C_LB, C_Stern)` in measured ops, with `C_Stern` also reported) and
  the full Prange/LB/Stern `C`-ladder.
- **model-deviation (named, not a failure)** — the ladder is monotone but a measured
  `C` deviates from its predicted work factor beyond tolerance: report the deviation
  honestly (the better-attack model is mis-calibrated for this regime); the tightened
  bound still stands at the measured `C_best`.
- **6.1 vacuity / 6.4 overhead** — the verifier op-count is not below the cheapest
  attacker's effort: falsified.
- **non-monotone / no clean threshold (6.5 boundary)** — on a **large** predicted
  step (Prange→LB, ~73×) a stronger attacker fails to beat the weaker one, or a curve
  has no locatable breakpoint: named quarantine; this is almost certainly an
  implementation bug the smoke should have caught. (The **small** LB→Stern step, ~1.11×,
  being unresolved at `T=64` is *not* this branch — it is reported as within-noise per
  the Ladder-ordering gate.)
- **void_run** — code/regime not matching the frozen spec, the target manifest not
  materialized before attacker runs, labels leaked, an attacker given more than `z`, a
  non-validated attacker (no exhibited witness), or non-deterministic.

## Required outputs (under `results/pvnp/certificate-syndrome-v2/`)

- `manifest.json` (code = v1, `target_seed`, attacker ladder, git sha, verdict);
- `target_manifest.json` (the decoupled `target_seed` target set; attacker-visible `z`
  and labels-only scoring fields separated; emitted before any attacker runs);
- `prediction_lock.json` (formulas, constants, predicted `C(ops)`, tolerance,
  smoke calibration provenance);
- `verifier_access_declaration.json` (every attacker sees only `z`; labels scoring-only);
- `attacker_ladder_curve.csv` (per attacker, per op-budget: witness-recovery success,
  iterations, measured ops);
- `capacity_ladder.json` (per attacker: measured `C(ops)`, predicted work factor,
  measured/predicted ratio, drop vs Prange, and `C_best = min(C_LB, C_Stern)`);
- `valid_iteration_audit.json` (per attacker: valid iterations, `rank_fail_draws`,
  measured ops incl. overhead);
- `witness_validity_audit.json` (per attacker: a sample of recovered `e*` re-checked
  `He*=z ∧ wt≤τ`);
- `op_count_report.json` (verifier ops flat; attacker ops at each `C`; gaps);
- `prediction_vs_measured.json`; `falsifier_summary.md`; `README.md`.

(Per the v1 receipt's artifact-packaging note: this slate **pre-registers** the
per-file output bundle above; the harness must emit it, not a single consolidated
JSON.)

## Anti-P-Hack rule

- Reuse v1's exact code (`n,k,w,τ,code_seed`); sample the targets from the frozen
  decoupled `target_seed` and materialize `target_manifest.json` before any attacker
  runs; freeze the attacker parameters (`LB p=2`, `Stern p=2, l=8`) and
  `prediction_lock.json` **before** the frozen run; do not retune parameters or
  constants to make the ladder land on the prediction.
- Every attacker receives **only** `z`; planted `e`/`wt(e)`/`s` are scoring-only.
- Each attacker is smoke-validated on a **throwaway** regime first (it must recover
  valid witnesses and match its analytic curve) before the frozen run. Smoke may
  calibrate op constants only because it is disjoint from the frozen regime; the
  calibrated constants must be locked before frozen targets are scored.
- Deterministic seeds; report determinism.
- The measured `C_best` among tested stronger attackers is still an **upper bound**
  (BJMM/MMT would lower it) — a new slate, not a v2 edit.

## Freeze checklist

- [ ] Sample and hash the decoupled `target_seed=2026220` `T=64` target manifest;
      confirm the code (`n,k,w,τ,code_seed`) is byte-equal to v1 and all attackers
      consume only manifest `z`.
- [ ] Freeze `LB p=2` and `Stern p=2, l=8`, plus `prediction_lock.json` (the Locked
      formulas above + smoke-calibrated constants + predicted 50% `C(ops)` + tolerance).
- [ ] Implement the three attackers (Prange reused) sharing the v1 verifier + GF(2)
      core; smoke each on a throwaway regime (recover valid witnesses; curve matches
      analytic) before the frozen run.
- [ ] Re-affirm: every attacker sees only `z`; labels scoring-only; per-file outputs.

## Freeze rule

Edits allowed without a new slate id: typo/path/output-naming corrections preserving
semantics. Edits requiring a new slate id: changing the code (`n,k,w,τ,code_seed`) or
`target_seed` (would break the within-v2 cross-attacker comparison); changing the
attacker set or the frozen `p`/`l`; adding a representation attack (MMT/BJMM); giving
an attacker more than `z`; using `wt(e)`/`s` as an input; retuning parameters or
locked constants after seeing the frozen ladder.
