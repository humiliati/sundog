# H3-PL Pre-registration — privilege located AT the pooling stage + seed-nulled back-action (HS1, amended)

**Status: DRAFT v1 (2026-07-02) — pre-adversarial-review. NOT frozen; no experiment code has run.**
Freeze happens only after the structured adversarial review applies its blocking-defect list and the
pre-freeze tuning pass pins the de-saturated control (§7). Standing discipline header, verbatim:
**clean null = success; forward-generate only; frozen test before first run.**

Amendments baked from the T1 wave (this is the slate's HS1 with three corrections applied):
ceiling-relative paired thresholds (the written "pooled ≤ 0.1" is dead — the pooled ceiling on this
body is a measured 0.1256 lower bound, still rising at n=20k); the calibrated battery as the only
instrument (naive ridge under-read this channel 20×); full-training discipline (smoke = non-verdict).

---

## 1. Claim (amended, two clauses)

On the FROZEN `clf_d` body of the v2 pooled-shadow substrate at λ = 2.0:

**(Access / localization).** The continuous latent `c` survives in the trained PER-UNIT features
`φ_d(uᵢ)` strictly beyond what the pooled mean of the same features carries: with both ceilings
measured **in the same run, on the same regenerated body, by the same calibrated battery**,

> `ceil_perunit ≥ ceil_pooled + 0.15` **and** `ceil_perunit ≥ 0.30`.

Suppression is then localized AT the pooling stage, not encoder-deep.

**(Back-action / observer effect).** Granting pooled access and jointly training encoder+report to
recover `c` (from the frozen-body initialization, d-objective retained) succeeds —
`rec_joint ≥ ceil_pooled + 0.20` with `d-acc(joint) ≥ 0.90` — only with encoder restructuring
exceeding the 95th percentile of a k = 10 same-objective seed-retrain null on at least one of
{linear CKA to baseline, d-acc drift, Δ pooled c-decodability}.

## 2. Pre-registered outcomes (no dead zone; every region named)

**Access clause** (margin `m = ceil_perunit − ceil_pooled`):
- **(a) LOCALIZED** — `m ≥ 0.15 ∧ ceil_perunit ≥ 0.30`: privilege sits at the pool; the body kept
  what its objective didn't spend.
- **(b) ENCODER-DEEP** (the kill; an informative null) — `m < 0.05`: pooling adds essentially
  nothing; an unincentivized d-body carries no extra probe-recoverable `c` anywhere upstream.
- **(c) BOUNDED-PARTIAL LOCALIZATION** — everything else (`0.05 ≤ m < 0.15`, or `m ≥ 0.15` with
  `ceil_perunit < 0.30`): attenuation partially localized at pooling, quantified by `m`. Given the
  probe-ceiling study landed in its middle band twice, (c) is the expected-likely outcome and is a
  full verdict, not a consolation.

**Back-action clause** (evaluated only if `d-acc(joint) ≥ 0.90`; below that the cell is
**OBJECTIVE-ABANDONED**, a named non-verdict — the claim concerns recovery while remaining a d-body):
- **(d) RESTRUCTURING-PRICED RECOVERY** — `rec_joint ≥ ceil_pooled + 0.20` with ≥ 1 restructuring
  metric outside the seed null: recovery paid a measurable observer-effect price.
- **(e) RECOVERY WITHOUT DISTURBANCE** (the kill; genuinely interesting) — recovery at threshold
  with ALL metrics inside the seed null: pooled access is recoverable for free; the
  restructuring-price intuition dies.
- **(f) NO RECOVERY** — `rec_joint < ceil_pooled + 0.20`: joint incentive cannot reopen the pooled
  channel from this initialization at this budget; report directly (bounds the reg_c-from-scratch
  0.51 as an initialization effect, not an objective effect).

## 3. Substrate (regenerated, deterministic)

`scripts/shadow_pooled_synthetic_v2.py` conventions, untouched: `gen(n, λ=2.0, seed)`; `clf_d`
encoder trained to convergence per the banked recipe (OBJ_SEED offsets fixed); n = 20,000 train,
10,000 **once-touched** verdict split (opened exactly once, after all model/member selection on the
train/CV side — threshold-anchored, per the probe-ceiling winner's-curse fix). Per-unit features:
`φ_d(u) ∈ (n, K, H)` read pre-pool from the frozen encoder; probe input = flattened `(K·H)` in the
substrate's fixed unit order.

**C0 anti-confound gate (inherited, HARD):** raw mean `c`-R² low at λ = 2.0 (raw averaging washes
`c`); if C0 fails the substrate regen is wrong — abort, not a verdict.

## 4. Instrument — the calibrated battery, with a per-unit calibration leg

Counted set = {P1 ridge, P4 Nyström (γ=1.0, α=0.1), P5 GBT} + KSG-MI confirmation (MI is
confirmatory, never the ceiling); MLP and kNN run and are reported but carry no verdict weight
unless they SPEAK (blindness excuses silence, never speech — the BD-1 rule).

- **Per-member injection calibration on EACH input class** (pooled `(H)` and per-unit `(K·H)`
  separately): inject `c` at strengths {0.10, 0.20}; a member is LIVE on that class at the smallest
  strength it detects (its floor). Members blind at 0.20 on a class are MEMBER-BLIND there and
  leave the counted set for that class.
- **Ceiling** per class = max over live counted members of once-touched-split R², replicated
  pool-CV → split (threshold-anchored).
- **Per-unit dimensionality policy (pinned, not improvised):** probes run on the flattened `(K·H)`
  features AND on PCA-k ∈ {64, 256} columns; the VERDICT column is flattened-raw if any counted
  member is live there, else the largest live PCA column; all columns reported.
- **Full-training discipline:** all probes at full budget, best-of-3 fits per member per column;
  smoke runs are non-verdicts by design.

## 5. Back-action arm

Joint training: encoder initialized at the frozen `clf_d` weights; loss = d-loss + β·c-report-loss,
**β = 1.0 primary** (β ∈ {0.3, 3.0} reported, non-verdict); same optimizer/budget class as the
substrate recipe; report head = the substrate's standard head class on pooled features.
`rec_joint` = the battery ceiling (pooled class) on the JOINT body's pooled features — same
instrument, no bespoke readout.

**Seed-retrain null (k = 10):** retrain `clf_d` from scratch, same objective/hyper, seeds
81235+i (i = 0..9); for each, compute the three metrics against the frozen baseline:
1. **linear CKA** between pooled features on a fixed 5,000-sample probe set (pinned seed);
2. **d-acc drift** = |d-acc − d-acc_frozen| on the verdict split;
3. **Δ pooled c-decodability** = Nyström-P4 (fixed hyper) pooled R² minus the frozen baseline's —
   the forced quantity, reported directly.
Null threshold per metric = 95th percentile of the k = 10 values. (k = 10 is the slate's floor; the
review may raise it — cost permitting — since a 95th percentile on 10 draws is coarse; if k stays
10, the operative rule is "exceeds the MAX of the null draws", stated plainly.)

## 6. Controls

- **C0** raw-wash gate (§3, hard abort).
- **De-saturated d control:** on the retuned variant (§7), white-box (per-unit) d-probe advantage
  over pooled d-acc must be ≤ 0.05 — fireable only because pooled d-acc sits in 0.75–0.85 there.
  Fires ⇒ the "determine survives pooling" background assumption wobbles on this substrate; report
  as a scope limitation on (a)–(f), not a verdict change.
- **Battery liveness** per §4 (member-blind semantics).
- **Determinism:** numpy-only, fixed seeds, byte-stable result JSON; per-row checkpointing + the
  detached-watchdog pattern for the long arms.

## 7. Pre-freeze tuning pass (control-only; the one thing that runs before freeze)

Retune the substrate's d-channel (raise η noise / shrink a) until pooled d-acc lands in
**0.75–0.85** at λ = 2.0; pin the found (η, a) INTO this prereg at freeze (TO-PIN-AT-FREEZE). This
pass touches only the de-saturated CONTROL variant — never the primary body, never `c`-probes.

## 8. Seeds ledger (pinned)

substrate regen 51235 · CV/split 61235 · battery 1789 · joint arm 71235 · retrain null 81235+i
(i = 0..9) · tuning pass 86235 · reserve 91235.

## 9. Kill fireability audit

(b) fires on a measured margin < 0.05 — live (the probe-ceiling study proves the battery reads
this substrate). (e) fires on recovery-with-all-metrics-inside-null — live (metrics are continuous,
null is empirical). (f) fires on a threshold miss — live. C0/OBJECTIVE-ABANDONED are gates, not
verdicts. No conjunction requires two unlikely events simultaneously; each region of outcome space
has exactly one name.

## 10. Priors named (nearest first)

Banked: the probe-ceiling receipts (pooled ceiling 0.1256, LOWER BOUND, curve unconverged —
the reason every threshold here is in-run relative); reg_c 0.51 (white-box objective scale); the
adversarial hide-d frontier (existence ≠ trainability); v2's C1 (raw units carry `c` — which is
why (Access) is about TRAINED features, not a relabeling). External: Binder et al. 2024
(behavioral self-prediction advantage, uncontrolled substrate); amnesic probing (Elazar &
Goldberg); LEACE; V-information. **Limitation, up front:** everything here is probe-access
asymmetry and report-head training side-effects in a ground-truth substrate — no claim about
introspection as a mental phenomenon.

## 11. Cost & runtime

Battery arms: minutes–hours (two classes × columns × best-of-3). Joint arm + k = 10 retrain null:
the driver — ~1–1.5 days CPU with checkpointing. Total inside the slate's 1.5-day envelope.

## 12. Deliverables

`scripts/shadow_introspect_privilege.py` + `scripts/test_shadow_introspect_privilege.py` (frozen
test BEFORE first run, pinning thresholds, seeds, and battery membership), receipt
`docs/atlas/H3_PRIVILEGE_LOCALIZATION_RESULT.md`, result JSON under `results/atlas/h3/`.
