# H3-PL Pre-registration — privilege located AT the pooling stage + seed-nulled back-action (HS1, amended)

**Status: FROZEN v2 (2026-07-02).** v1's structured review found 7 blocking/medium defects (§0); all
applied. The de-saturated-body tuning pass (§7) landed and its values are pinned below. **No
experiment code has run; no threshold in this file may change from here.** Next actions are
mechanical: the frozen test, then the run. Standing discipline header, verbatim: **clean null =
success; forward-generate only; frozen test before first run.**

## 0. Review disposition (v1 → v2)

- **B1 (blocking) — ACCESS margin was a data-processing tautology.** Per-unit `φ_d(u)` is `(n,64,32)`
  = 2048 dims and pooling is a fixed function of it, so per-unit c ≥ pooled c by construction; with
  C1 (raw units carry c) and no incentive for `clf_d` to destroy it, per-unit ≈ white-box scale ≫
  pooled 0.126 → old outcome (a) foreordained, kill (b) unfireable (the HS1-v1 / HS5 refutation
  class), plus a 2048-vs-32-dim probe confound. **Fixed:** THREE ceilings at MATCHED probe
  dimensionality; the informative contrast is trained-per-unit vs RAW-per-unit (the encoder's effect,
  C1 is the info-theoretic reference), not per-unit vs pooled (≈ DPI). §1–§2, §4 rewritten.
- **B2 (blocking) — back-action disturbance set contained its own outcome.** "Δ pooled
  c-decodability" IS the recovery, tautologically outside the same-objective null → (d) foreordained,
  (e) unfireable. **Fixed:** Δ-pooled-c-decodability is the recovery READOUT, not a disturbance
  metric; the null gate = {linear CKA, d-acc drift} only. §5.
- **B3 (blocking, self-consistency) — two contradictory bodies** (banked-recipe 0.99 vs de-saturated
  0.75–0.85). **Fixed:** ONE de-saturated body for all arms; C0/C1 re-verified on it. §3, §7.
- **B4 (blocking) — per-unit injection site + battery hypers undefined** (`inject` targets 32-dim
  pooled; `Nystroem n_components=2000` interpolates 2048). **Fixed:** matched column budget across
  ceilings + injection into per-unit features pre-reduction; `n_components < k`. §4.
- **B5 (medium) — k=10 too small for a 95th-pct tail.** **Fixed:** k=20 seed-retrain null. §5.
- **B6 (medium) — verdict-β cherry-pick risk.** **Fixed:** verdict β = smallest β with d-acc ≥ 0.90;
  "∃ β" claim, all reported. §5.
- **B7 (medium) — cross-arm multiplicity on the once-touched split.** **Fixed:** all selection on CV;
  split scored exactly once per ceiling (3 touches, stated). §4.

## 1. Claim (v2, two clauses)

On ONE FROZEN de-saturated `clf_d` body (§7) of the v2 pooled-shadow substrate at λ = 2.0, all
ceilings measured in the same run by the same calibrated battery at MATCHED probe dimensionality:

**(Access / localization).** The `clf_d` encoder PRESERVES the per-unit continuous latent `c` that
the raw input carries (C1), while pooling then suppresses it — i.e. `c` is lost AT the pool, not
encoder-deep:

> `ceil_trained_perunit ≥ ceil_raw_perunit − 0.10` (encoder preserved it) **and**
> `ceil_trained_perunit ≥ ceil_pooled + 0.15` (the pool, not the encoder, is where it goes).

**(Back-action / observer effect).** Granting pooled access and jointly training encoder+report to
recover `c` from the frozen-body init (d-objective retained) succeeds —
`rec_joint ≥ ceil_pooled + 0.20`, `d-acc(joint) ≥ 0.90` — only with encoder restructuring beyond the
95th percentile of a **k = 20** same-objective seed-retrain null on ≥ 1 of {linear CKA to baseline,
d-acc drift}. (Δ pooled c-decodability is the recovery readout, reported directly, NOT a null metric.)

## 2. Pre-registered outcomes (no dead zone; every region named)

Let `m_enc = ceil_trained_perunit − ceil_raw_perunit` (encoder effect; ≤ 0 expected) and
`m_pool = ceil_trained_perunit − ceil_pooled` (pool effect).

- **(a) LOCALIZED-AT-POOL** — `m_enc ≥ −0.10 ∧ m_pool ≥ 0.15`: the encoder kept the per-unit `c`; the
  pool is where it dies. Privilege sits at the pooling stage.
- **(b) ENCODER-DEEP** (the kill; informative null) — `m_enc < −0.25`: `clf_d` training itself
  attenuated `c` per-unit before any pooling — an unincentivized d-body scrambles `c` upstream. This
  is now GENUINELY fireable (raw-per-unit is the reference, not pooled).
- **(c) PARTIAL / MIXED** — everything else (e.g. `−0.25 ≤ m_enc < −0.10`, or `m_pool < 0.15`):
  attenuation split between encoder and pool, quantified by `(m_enc, m_pool)`. Given the probe-ceiling
  study landed in its middle band twice, (c) is expected-likely and is a full verdict.

Back-action (evaluated only if `d-acc(joint) ≥ 0.90`; else **OBJECTIVE-ABANDONED**, a named
non-verdict): **(d) RESTRUCTURING-PRICED** (`rec_joint` clears, ≥ 1 disturbance metric outside null) /
**(e) RECOVERY WITHOUT DISTURBANCE** — the kill (`rec_joint` clears, BOTH {CKA, d-acc drift} inside
null): pooled access is recoverable for free; the observer-effect intuition dies / **(f) NO RECOVERY**
(`rec_joint` below threshold): joint incentive cannot reopen the pooled channel from this init.

## 3. Substrate (regenerated, deterministic; ONE de-saturated body)

`scripts/shadow_pooled_synthetic_v2.py` conventions (K = 64 units, H = 32 rep dim, F feature dim,
λ = 2.0), with the **PINNED de-saturation `SIGMA_D = 9.0`** (§7). ONE `clf_d` encoder trained on that
de-saturated d-channel (pooled d-acc 0.80 at the tuning body — no second, saturated body exists in
this experiment). Three feature representations read from that frozen encoder: **raw-per-unit** `u`
`(n,K,F)`; **trained-per-unit** `φ_d(u)` `(n,K,H)` pre-pool; **pooled** `φ_d(u).mean(units)` `(n,H)`.
n = 20,000 train, 10,000 once-touched verdict split (opened once per ceiling; §4).

**Gates re-verified on THIS body (hard aborts, not verdicts) — pinned at freeze from the tuning body,
re-checked on the run body at run:** C0 raw-mean-of-`u` c-R² ≤ 0.10 at λ = 2.0 (raw averaging washes
`c`; tuning value **0.0**); C1 single raw unit recovers `c`, `unit_c ≥ 0.5` (tuning value **0.999**) —
C1 is load-bearing as the ACCESS reference. Run aborts (not a verdict) if either fails on the run
body, or if the run body's pooled d-acc falls outside [0.75, 0.85] (substrate-regen failure).

## 4. Instrument — calibrated battery at MATCHED dimensionality (B1/B4/B7)

Counted set {P1 ridge, P4 Nyström, P5 GBT} + KSG-MI (confirmatory). MLP/kNN reported, verdict-weight
only if they SPEAK (BD-1: blindness excuses silence, never speech).

- **Common column budget `k_col = 128`** (PCA) applied to EVERY ceiling arm — raw-per-unit (2048→128),
  trained-per-unit (2048→128), pooled (32; kept whole, ≤ 128). All R² comparisons are at ≤ 128
  columns so no arm wins on dimensionality. Nyström `n_components = 100 < k_col` (re-picked from the
  2000 that interpolated 2048 dims). PCA fit on the CV/train side only.
- **Injection calibration per arm, matched:** inject a `c`-proportional signal into each arm's
  features BEFORE PCA reduction, at strengths {0.10, 0.20}; a member is LIVE on an arm at the smallest
  strength it detects (its floor); blind at 0.20 → MEMBER-BLIND on that arm, leaves the counted set
  there. The three arms therefore carry independently-calibrated floors — a floor difference across
  arms is itself reportable.
- **Ceiling** per arm = max over live counted members of once-touched-split R², threshold-anchored
  (config — member, PCA basis, hypers — frozen on CV; the verdict split is scored EXACTLY ONCE per
  arm = 3 touches total; the 3-arm family is acknowledged and no per-arm re-peeking is permitted).
- **Full-training discipline:** all probes at full budget, best-of-3 fits per member per arm, chosen
  on CV; smoke = non-verdict.

## 5. Back-action arm (B2/B5/B6)

Joint training: encoder init at the frozen de-saturated `clf_d`; loss = d-loss + β·c-report-loss;
report head = the substrate's standard head on pooled features; same optimizer/budget class as §3.
`rec_joint` = the §4 battery ceiling on the JOINT body's POOLED features (same instrument).
**Verdict β = the smallest β ∈ {0.3, 1.0, 3.0} with `d-acc(joint) ≥ 0.90`** (pre-registered selection
rule; all three reported); the back-action claim is existential over that set.

**Seed-retrain null (k = 20):** retrain the de-saturated `clf_d` from scratch, same objective/hyper,
seeds 81235+i (i = 0..19); disturbance metrics vs the frozen baseline: (1) linear CKA between pooled
features on a fixed 5,000-sample probe set; (2) d-acc drift on the verdict split. Null threshold =
95th percentile over 20 draws. **Δ pooled c-decodability (P4, fixed hyper) is reported as the recovery
readout, NOT gated against the null** (it IS the recovery — B2).

## 6. Controls

- **C0 / C1** (§3, hard aborts; C1 also the ACCESS reference).
- **De-saturated d control:** on this body, white-box (per-unit) d-probe advantage over pooled d-acc
  ≤ 0.05, fireable because pooled d-acc sits in 0.75–0.85. Fires ⇒ "determine survives pooling"
  wobbles here; reported as a scope limitation on (a)–(f), not a verdict change.
- **Battery liveness** per §4. **Determinism:** numpy-only, fixed seeds, byte-stable JSON; per-row
  checkpoint + detached-watchdog for the long arms.

## 7. De-saturation tuning pass — DONE + PINNED (control-only; ran before freeze)

`scripts/hs1_desaturate_tuning.py` (receipt `results/atlas/hs1/desaturate_tuning.json`, committed
`db1c52b2`) swept the per-unit d-noise `SIGMA_D` (the η knob; `a` = A_DISC left frozen). The trained
`clf_d` encoder denoises via pooling far better than a linear model, so pooled d-acc is stubborn —
0.996 (3.0) → 0.901 (6.0) → 0.800 (9.0). **PINNED: `SIGMA_D = 9.0`** → pooled d-acc = 0.8004 (centre of
[0.75, 0.85]); all other substrate constants unchanged (K = 64, H = 32, A_DISC frozen unit vector,
`OBS_NOISE = 0.05`). C-channel gates on that body: **C0 raw-mean c-R² = 0.0** (≤ 0.10 ✓),
**C1 single-unit c-R² = 0.999** (≥ 0.50 ✓) — the d-noise knob is orthogonal to the continuous channel,
so this pass touched no `c`-probe and fixed no threshold. Tuning seed 86235.

## 8. Seeds ledger (pinned)

substrate/body 51235 · CV/split 61235 · battery 1789 · joint arm 71235 · retrain null 81235+i
(i = 0..19) · tuning pass 86235 · reserve 91235.

## 9. Kill fireability audit (post-fix)

(b) ENCODER-DEEP fires on `m_enc < −0.25` — now LIVE: raw-per-unit is a genuine, higher reference than
pooled, so "encoder attenuated c per-unit" is measurable and NOT foreordained (the v1 defect). (e)
RECOVERY-WITHOUT-DISTURBANCE fires on recovery with {CKA, d-acc drift} inside null — now LIVE: the
recovery readout was removed from the gate (B2). (f) fires on a threshold miss. C0/C1/OBJECTIVE-
ABANDONED are gates. Every outcome region has exactly one name; no conjunction needs two unlikely
events at once.

## 10. Priors named (nearest first)

Banked: probe-ceiling receipts (pooled ceiling 0.1256, LOWER BOUND — the reason thresholds are in-run
relative); reg_c 0.51 (white-box scale); v2 C1 (raw units carry `c` — now the ACCESS reference, which
is why the informative contrast is trained-vs-raw, not the DPI-foreordained per-unit-vs-pooled);
adversarial hide-d (existence ≠ trainability). External: Binder et al. 2024 (behavioral
self-prediction, uncontrolled substrate); amnesic probing (Elazar & Goldberg); LEACE; V-information.
**Limitation up front:** probe-access asymmetry + report-head training side-effects in a ground-truth
substrate — no claim about introspection as a mental phenomenon.

## 11. Cost & runtime

Battery: 3 arms × ≤128 cols × best-of-3 — minutes–hours. Joint arm + k = 20 retrain null: the driver,
~1–1.5 days CPU with checkpointing (k = 20 vs 10 ≈ doubles the null leg; still inside the slate's
envelope). Total ≈ 1.5 days.

## 12. Deliverables

`scripts/shadow_introspect_privilege.py` + `scripts/test_shadow_introspect_privilege.py` (frozen test
BEFORE first run — thresholds, seeds, battery membership, the three-ceiling matched-dim protocol),
receipt `docs/atlas/H3_PRIVILEGE_LOCALIZATION_RESULT.md`, JSON under `results/atlas/h3/`.
