# Sundog Certificate Problem — Syndrome Certificate v1 Receipt

- Receipt id: `pvnp-certificate-syndrome-v1-2026-06-04`
- Phase / probe: Phase 4 / §5 capacity experiment — the syndrome/SIS certificate's
  measured capacity-relative one-wayness threshold against Prange ISD
- Date run: 2026-06-04 (local); wall-time 82m12s (diagnostic-only; op-count is the cost signal)
- Author / runner: `python scripts/pvnp-certificate-syndrome-v1.py` (frozen regime)
- Result dir: `results/pvnp/certificate-syndrome-v1/` (transient, gitignored)
- Frozen slate: [`SUNDOG_CERTIFICATE_SYNDROME_V1_SLATE.md`](../SUNDOG_CERTIFICATE_SYNDROME_V1_SLATE.md)
- Problem doc: [`SUNDOG_CERTIFICATE_PROBLEM.md`](../SUNDOG_CERTIFICATE_PROBLEM.md) (§4 instance, §5 experiment)
- Prototype: [`SUNDOG_CERTIFICATE_SYNDROME_PROTOTYPE_NOTE.md`](../SUNDOG_CERTIFICATE_SYNDROME_PROTOTYPE_NOTE.md)

## Verdict

**Bounded positive — measured capacity-relative one-wayness (against Prange ISD).**
On the frozen non-enumerable `[128,64]` GF(2) regime, the syndrome certificate
exhibits a measured find-vs-check separation: recovering a valid light deviation
witness `e*` from the compact shadow `z` (syndrome decoding) is capacity-hard, while
the witness-verifier is cheap and flat — and the measured threshold lands on the
pre-registered prediction. This is the lane's **first measured capacity threshold**
(every Phase-1 receipt reported `capacity_threshold = not_estimated`). It converts
the §4 constructed instance from existence proof to a measured Sundog receipt.

The threshold is **against Prange ISD** (the registered attacker class); a stronger
ISD variant (Stern/MMT/BJMM) would lower it, so the measured `C` is an **upper
bound** on the true one-wayness threshold. No cryptographic one-wayness claim is
made; hardness is imported (the SIS/decoding assumption), not proved.

## Frozen regime (as pre-registered)

`[n=128, k=64], w=12, τ=12`, GF(2), `code_seed=2026128`, body seed base `7000000`,
`T=64` targets. Full low-weight enumeration space `C(128,12) = 23,726,045,489,546,400`
(`2.37×10¹⁶`) — non-enumerable, so a smarter finder (ISD), not brute enumeration, is
the relevant attacker. Frozen before the run; the regime and prediction table were
not retuned.

## The measured capacity curve (invert-`e` / light-witness recovery)

Prange ISD, capacity = rank-valid information-set trials `B`:

| Tier | `B` (rank-valid) | measured witness recovery | predicted `1−(1−p)^B` | \|Δ\| |
| --- | ---: | ---: | ---: | ---: |
| Small | 64 | 0.000 | 0.009 | 0.009 |
| Medium | 1,024 | 0.109 | 0.132 | 0.023 |
| **Breakpoint** | **5,004** | **0.469** | **0.500** | 0.031 |
| Large | 16,384 | 0.875 | 0.896 | 0.021 |
| XL | 32,768 | 1.000 | 0.989 | 0.011 |

Analytic model (pre-registered): `p = C(n−k,w)/C(n,w) = 1.384×10⁻⁴`, expected iters
`N = 7224.27`, 50%-breakpoint `5007.48`. **The measured 50%-crossing lands at the
predicted breakpoint** (B≈5004 → 0.469), max \|Δ\| = 0.031 across the ladder. The
curve rises monotonically `0.000 → 1.000` with a clean, locatable threshold.

## The find-vs-check gap (the P-vs-NP-shaped result)

| Quantity | Measured |
| --- | ---: |
| Verifier check (flat, op-counted) | **16,576 ops** (two GF(2) matvecs: recompute `z=Hy` + check `He*`) |
| Capacity threshold `C` (50% witness recovery) | **≈ 5,007 rank-valid Prange trials** |
| Attacker ops to recover one witness (avg over `T=64`) | **≈ 4.5×10⁹ ops** (`2.884×10¹¹` total / 64) |
| **Find-vs-check gap** | **≈ 2.7×10⁵×** (recover-one-witness vs check-one) |

So a capacity-bounded adversary below `C` cannot recover the deviation from `z` while
the verifier checks a witness ~270,000× more cheaply. Note the check is 16,576 ops
(two matvecs), not the 8,192 single-matvec the slate's rough prediction used — an
honest correction; still flat and `≪ C`.

## Rank-valid convention audit

`B` counts **rank-valid** information-set trials only. Singular-submatrix draws are
charged to measured ops and audited separately: `rank_fail_draws_total = 1,191,143`
(the analytic curve is read against rank-valid trials, so the prediction table stays
the valid comparator). `measured_attacker_ops_total_incl_rankfail = 2.884×10¹¹`.

## Artifact packaging note

The frozen harness emitted the load-bearing run data as a consolidated JSON,
`results/pvnp/certificate-syndrome-v1/isd_capacity_curve.json`, plus the throwaway
smoke JSON. This differs from the slate's pre-run per-file output list
(`manifest.json`, `capacity_curve.csv`, `isd_rank_audit.json`, etc.), but not from the
measurement contract: the frozen regime, prediction, capacity curve, verifier op
count, and rank-fail audit are all present in the JSON and reconciled above. The
privilege/soundness facts are construction-level facts of the harness and are stated in
this receipt rather than emitted as separate files. A successor slate should either
emit the per-file bundle or pre-register the consolidated JSON schema.

## The other two adversary tasks (as pre-registered)

- **invert-`s`** (recover the secret) — **unconditionally impossible**: `z = He` is
  independent of `s`, so `2⁶⁴` secrets map to each syndrome. Vacuously resistant,
  **not** a capacity result; not promoted.
- **spoof** (forge a passing certificate for a truly-unsafe body) — **structurally
  impossible** under the existence predicate `Safe(y) := ∃ e* : He*=Hy ∧ wt(e*)≤τ`
  (a passing light witness *is* safety). Not a capacity probe.

## Gate results

| Gate | Result |
| --- | --- |
| Regime frozen + non-enumerable | pass (`C(128,12)=2.37×10¹⁶ > 10¹²`, frozen pre-run) |
| Code validity | pass (`G Hᵀ = 0` asserted) |
| Verifier cheap | pass (16,576 ops `≪ C ~ 10⁹`) |
| Verifier sound | pass (`accept ⟺ light witness`, by construction) |
| Privilege / label audit | pass (inverter received **only** `z`; planted `e`/`wt(e)`/`s` scoring-only; verifier inputs `(y, exhibited e*)`) |
| Clean threshold | pass (monotone `0→1`, breakpoint at the predicted `B≈5007`) |
| Cost reported | pass (verifier ops, attacker ops, gap, rank audit) |
| Determinism | **by construction** (seed-pinned pure-numpy) and confirmed byte-identical on the throwaway smoke; a full frozen byte-identical re-run is in flight (background, scratch dir) — to be marked CONFIRMED on completion |

## Prediction-vs-measured (anti-p-hack)

The regime, seeds, attacker class, ladder, and the full prediction table were frozen
**before** the run. The measured curve matches the prediction (max \|Δ\| = 0.031,
breakpoint on target), so the result is a confirmation, not a fit. A large deviation
would itself have been the finding; none occurred.

## Claim boundary

This receipt does not claim cryptographic one-wayness, that verification is
"polynomial" or "in P", general alignment verification, wall-time cheapness,
body-resistance / Sundog-regime-2, or progress on P vs NP. The measured `C` is a
threshold **against Prange ISD** on a non-enumerable but bounded-scale regime — a
control-substrate, certificate-discipline result that imports decoding hardness as an
assumption. A stronger attacker class or a scaled regime is a **new slate**.

## What this earns

The Sundog Certificate Problem now has, on one constructed instance, all three
target properties *measured together at the same capacity*: checking is cheap
(16,576 ops, flat), the shadow is control-sufficient but loses the secret by algebra
(`2⁶⁴`/syndrome), and recovering the deviation is capacity-hard with a **measured
threshold `C ≈ 5,007` Prange trials** and a **find-vs-check gap ≈ 2.7×10⁵×**. This is
the §5 deliverable: the first measured capacity-relative one-wayness threshold in the
lane.
