# H3-PL v3 Pre-registration — privilege-at-pooling + seed-nulled back-action (re-registration)

**Status: DRAFT v3 (2026-07-02) — PRE-DRY-RUN.** Re-registration after the v2 run voided both clauses
on achievability grounds (receipt `H3_PRIVILEGE_LOCALIZATION_RESULT.md`, run commit `2ace4df8`; the
frozen v2 spec `815ea529` stays immutable as the record — this file supersedes, it does not edit).
Freeze happens only after the **achievability dry-run** (§7 — the new pre-freeze step this cycle
exists to prove out) passes and its values are pinned here. Standing discipline header, verbatim:
**clean null = success; forward-generate only; frozen test before first run.**

## 0. Defect disposition (v2-run → v3) + one scoping catch

- **D8 (back-action gate unreachable):** v2 gated on `d-acc(joint) ≥ 0.90` while §7 de-saturated the
  body to 0.75–0.85 by design. **v3 gate = the BAND FLOOR: `d-acc(joint) ≥ 0.75`** — "still a
  competent d-body by the same standard the substrate was tuned to." δ-free; the band predates run-1.
- **D9a (arms read where the substrate washes per-unit c):** v2 read all arms at λ = 2.0 with a
  linear/kernel battery that cannot demodulate the per-unit RFF code → all-blind arms. **The void
  receipt's own sketch ("read at λ ≈ 0") is REJECTED here as degenerate:** at λ = 0 all units are
  identical, so pooled-trained ≡ per-unit-trained and `m_pool ≡ 0` structurally — a third unfireable
  clause caught at scoping, for once before the run. **v3 keeps λ = 2.0 and fixes the READOUT CLASS
  instead:** new battery member **P6, the demodulating probe** (per-unit MLP → mean → linear, the
  substrate's own head class as a probe; precedent: the hide-d study's analytic existence arm).
  `reg_c ≈ 0.51` banked proves demodulate-then-pool reads c at λ = 2.0 — the signal is there; v2's
  battery just couldn't read it.
- **D9b (injection too weak at high dim):** fixed strengths {0.10, 0.20} contribute R² ≈ s²/128 in a
  128-column arm → members read blind even where c is present. **v3 injection = fixed-R² TARGET:**
  per arm, calibrate the injected strength (bisection) until ridge-CV on the injected features hits
  R² ≈ 0.10; a member is live if it detects THAT calibrated injection. P6's liveness is certified by
  its native positive control instead (reads c from RAW per-unit features at λ = 2.0).

## 1. Contamination disclosure (run-1 was seen; this block is the firewall)

Run-1 (`2ace4df8`) revealed: joint d-acc ≈ 0.789 (all β), rec_joint ≈ 0.41, d-drift ≈ 0.012–0.017 vs
null ≈ 0.0067, CKA ≈ 0.86 vs null floor ≈ 0.73, pooled ceiling ≈ 0.049 (ridge-only, marginal battery).

- **Informed by run-1 (design only):** the P6 member, the fixed-R² injection policy, the band-floor
  gate semantics, fresh seeds, this disclosure block.
- **NOT changed and NOT tuned (verdict constants, all pre-run-1):** ACCESS thresholds
  (`m_enc ≥ −0.10` / `< −0.25`; `m_pool ≥ 0.15` — run-1's blind arms gave ZERO information about
  trained-per-unit ceilings, so these are untainted); back-action recovery margin
  (`rec ≥ ceil_pooled + 0.20`); null metrics {linear CKA, d-acc drift}, k = 20, 95th pct.
- **Known in advance and stated plainly:** under the fixed gate, run-1's numbers would land the
  back-action clause at (d) RESTRUCTURING-PRICED (rec 0.41 ≫ threshold; drift outside null). v3 does
  not claim surprise there; its back-action value is the verdict becoming *well-formed* on fresh
  seeds. The ACCESS clause (`m_enc` — did `clf_d` preserve per-unit c?) is **genuinely open**: no one
  has ever measured it.
- **All verdict randomness is fresh:** every seed in §6 differs from every v2 seed; the k = 20 null
  is fully re-run against the new base body.

## 2. Claim (two clauses; same structure as v2, corrected gates)

On ONE FROZEN de-saturated `clf_d` body (`SIGMA_D = 9.0`, carried from the still-valid tuning pass
`db1c52b2`) of the v2 pooled-shadow substrate at λ = 2.0, all ceilings measured in the same run by
the same battery (now incl. P6) at matched column budget:

**(Access / localization).** `ceil_trained_perunit ≥ ceil_raw_perunit − 0.10` and
`ceil_trained_perunit ≥ ceil_pooled + 0.15` — the encoder preserved the per-unit `c` that the raw
input carries; the pool is where it dies.

**(Back-action / observer effect).** `rec_joint ≥ ceil_pooled + 0.20` with `d-acc(joint) ≥ 0.75`,
only with ≥ 1 of {linear CKA, d-acc drift} outside the fresh k = 20 same-objective seed-retrain null
(95th pct). Δ pooled c-decodability = the recovery readout, reported, never a gate metric.

## 3. Outcomes (unchanged six-region partition; gate constants per §0)

ACCESS: **(a) LOCALIZED-AT-POOL** (`m_enc ≥ −0.10 ∧ m_pool ≥ 0.15`) / **(b) ENCODER-DEEP** kill
(`m_enc < −0.25`) / **(c) PARTIAL-MIXED** (else). BACK-ACTION (evaluated iff `d-acc(joint) ≥ 0.75`,
else OBJECTIVE-ABANDONED non-verdict): **(d) RESTRUCTURING-PRICED** / **(e) RECOVERY WITHOUT
DISTURBANCE** kill (both metrics inside null) / **(f) NO RECOVERY**. Verdict β = smallest
β ∈ {0.3, 1.0, 3.0} with `d-acc(joint) ≥ 0.75`, all reported.

## 4. Instrument (v2 §4 + P6 + fixed-R² injection)

Counted set {P1 ridge, P4 Nyström (γ=1.0, α=0.1, n_components=100), P5 GBT, **P6 demodulator**} +
KSG-MI confirmatory. Matched column budget `k_col = 128` (PCA, CV-side fit) for P1/P4/P5; **P6 reads
the arm's UN-REDUCED per-unit tensor** (its per-unit → mean architecture is the dimensionality
control; on the pooled arm P6 degrades to MLP-on-pooled and its liveness decides). P6 spec: per-unit
MLP (in→128→128→32) → mean over units → linear, MSE on c, trained CV-side only, same budget class as
the substrate recipe, seeded. Liveness: P1/P4/P5 per arm via the fixed-R²-target injection (§0 D9b);
P6 via the native raw-arm positive control (§7). Ceiling per arm = max over live members, best-of-3
on CV, once-touched split scored exactly once per arm (4 arms incl. joint-pooled = 4 touches, stated).
Full-training discipline throughout; smoke = non-verdict.

## 5. Substrate, gates, back-action machinery

As v2 §3/§5/§6 verbatim except: fresh seeds (§6); the §0 gate constants; P6 added; injection policy
per §0. Hard aborts unchanged: C0 (raw-mean c-R² ≤ 0.10 at λ=2), C1 (single raw unit ≥ 0.50 at λ=0),
run-body pooled d-acc ∈ [0.75, 0.85]. k = 20 retrain null re-run in full against the fresh base body
(checkpointed). De-saturated d white-box-advantage control carried unchanged.

## 6. Seeds ledger (ALL fresh vs v2)

body 52235 · CV/split 62235 · battery 2789 · joint 72235 · retrain null 82235+i (i = 0..19) ·
dry-run 87235 · reserve 92235.

## 7. Achievability dry-run (pre-freeze; the process fix, applied)

`scripts/hs1_v3_dryrun.py`, n = 8000, minutes. **Asserts, all required to freeze:**
1. fresh-seed body reproduces the de-sat band (pooled d-acc ∈ [0.75, 0.85]) + C0/C1 pass;
2. **P6 positive control:** P6 trained on RAW per-unit features at λ = 2.0 reads c at CV-R² ≥ 0.30
   (the reference arm is alive; `reg_c ≈ 0.51` banked says it should be);
3. **injection calibration converges per arm:** for {raw-perunit, trained-perunit, pooled} the
   bisection finds a strength with ridge-CV on injected ∈ [0.05, 0.20] (the linear members are
   live-able everywhere);
4. **β = 0 gate control:** continued pure-d training holds d-acc ≥ 0.75 (the gate has a reachable
   region; β > 0 arms are NOT dry-run — the recovery question stays unharvested);
5. gate arithmetic in code: `BACKACTION_GATE ≤ DESAT_BAND[1]`, `BACKACTION_GATE = DESAT_BAND[0]`.
Values pinned into this file at freeze; the frozen test gains these as regression asserts.
**Boundary:** the dry-run never trains P6 on trained-per-unit features and never runs β > 0 — the
two open questions stay untouched.

## 8. Kill fireability (post-fix audit)

(b) fires on measured `m_enc < −0.25` with a LIVE P6 on both per-unit arms — live. (e) fires on
recovery with both disturbance metrics inside the fresh null — live (unchanged from v2's B2 fix).
(f) fires on a threshold miss — live, and no longer masked by an unreachable gate. OBJECTIVE-
ABANDONED requires d-acc(joint) < 0.75 = genuine competence collapse, not the band's own design.

## 9. Deliverables

Prereg freeze (this file + pinned §7 values) → config update (`hs1_privilege_config.py` v3 constants:
gate 0.75, P6, fresh seeds, injection policy; v2 values preserved in git at `9bdfb255`) → frozen test
update (achievability asserts added) → run script delta (P6 member + calibrated injection) → run →
receipt `H3_PRIVILEGE_LOCALIZATION_V3_RESULT.md` + JSON.
