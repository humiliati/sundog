# Sundog Certificate Problem — Syndrome Certificate v1 Capacity Slate

Status: **FROZEN for implementation (2026-06-04 local).** The three pre-freeze
review repairs are reconciled across all artifacts: the existence safety predicate
`Safe(y) := ∃ e* : He* = Hy ∧ wt(e*) ≤ τ` (planted `e` a label only), the parent-doc
framing aligned to invert-`e` / light-witness recovery (spoof structurally
impossible), and the Prange rank-valid `B` convention. The regime, seeds, attacker
class, ladder, and prediction table are now locked. No v1 attacker run may execute
against the frozen regime until the post-freeze Prange ISD attacker is implemented
and smoked on a **throwaway** regime; the frozen-regime run is the next
operator-gated step. (Freeze-before-execute discipline, per the lane: the Phase-3 v2
pre-freeze battery was generated before its slate froze and had to be quarantined.)

Date opened: 2026-06-04

This is the §5 experiment of
[`SUNDOG_CERTIFICATE_PROBLEM.md`](SUNDOG_CERTIFICATE_PROBLEM.md): the frozen, scaled,
ISD-attacker capacity ladder that converts the constructed instance (Candidate A,
the syndrome/SIS certificate) from an existence proof into a **Sundog receipt with a
measured capacity-relative one-wayness threshold**. The mechanism is already
verified on a toy regime (P1/P2/P3 + the find-vs-check curve), see
[`SUNDOG_CERTIFICATE_SYNDROME_PROTOTYPE_NOTE.md`](SUNDOG_CERTIFICATE_SYNDROME_PROTOTYPE_NOTE.md)
and `scripts/pvnp-certificate-syndrome.py`. v1 scales the regime so it is
non-enumerable, replaces the naive forger with a real attacker, and freezes the
contract.

## What v1 measures

The capacity-relative one-wayness threshold **`C`** of the syndrome certificate:
the effort at which a named attacker class, given only the compact shadow `z = Hy`
(the syndrome, with the witness withheld), can **recover a valid light deviation
witness `e*` with `He* = z`** — which is exactly syndrome decoding — while the
friendly verifier (with the witness) stays cheap. Below `C` the shadow is
sufficient for verification but insufficient for cheap witness recovery; above
`C`, witness recovery succeeds. This is the P-vs-NP-shaped checker-vs-finder
axis, measured. Exact recovery of the planted `e` is reported separately as a
diagnostic; verifier success requires any valid light witness, not necessarily
the planted one.

The three adversary tasks (§1.3 of the problem doc), and which one is the capacity
probe:

1. **verify** safety from `(y, witness e*)` — cheap, op-counted, the flat baseline.
2. **invert-`e`** (recover a valid light deviation witness `e*` from `z` alone)
   — the **capacity probe**; hardness is syndrome decoding. *This is the
   measured curve.*
3. **invert-`s`** (recover the secret) — **unconditionally** impossible (`z = He` is
   independent of `s`; `qᵏ` secrets per syndrome). Reported as vacuously resistant,
   **not** a capacity threshold.

Spoof (forge a passing certificate for a truly-unsafe body) is **structurally
impossible** with the sound witness-verifier because safety is the
distance-to-code predicate: `accept ⟺ a light witness exists ⟺ within τ of the
code = safe`. Spoof resistance is reported as structural, not as the capacity
probe.

## Frozen regime

| Parameter | Value | Rationale (pre-registered) |
| --- | --- | --- |
| field | GF(2) | simplest; decoding hardness well-studied |
| `n` | **128** | non-enumerable at `w=12` |
| `k` | **64** | rate 1/2; `s ∈ GF(2)^64` is the secret |
| `n−k` | **64** | syndrome length |
| `w` | **12** | error weight = body deviation |
| `τ` | **12** | light-witness threshold in `Safe(y)` |
| code | random systematic `G=[I_k|P]`, `H=[Pᵀ|I]`, **code_seed = frozen below** | `G Hᵀ = 0` asserted |
| body seed base | **frozen below** | deterministic targets |

**Why this regime (principled, pre-chosen, not fitted):** the full low-weight
enumeration space is `C(128,12) ≈ 2.37×10¹⁶` — definitively non-enumerable (the
prototype's `2.13×10⁵` toy critique is answered), so a smarter finder, not brute
enumeration, is the relevant attacker. The Prange-ISD work factor is in a runnable
range (below), so a **measurable** breakpoint exists. The verifier check is
`n(n−k) = 8192` bit-ops, flat.

Frozen seeds (chosen before any run, mechanical): `code_seed = 2026128`; body seed
base `= 7000000`; target count `T = 64`.

## The instance (frozen)

- **Body** `(s, e)`: `s ∈ GF(2)^64`, planted `e` of weight `w=12`; observation
  `y = sG + e`. The planted `e` is a generation/scoring label; verifier
  semantics are defined by the existence of a light same-syndrome witness.
- **Certificate / shadow** `σ = (z = Hy, witness e*, tag t = PRF_K(y))`. The compact
  shadow handed to the inverter is `z` only (length 64); the witness is withheld.
- **Safety predicate** `Safe(y) := ∃ e* : He* = Hy ∧ wt(e*) ≤ τ` (equivalently,
  `y` is within Hamming distance `τ` of the code). The planted `wt(e)` is not a
  verifier input and is not the semantic safety definition.
- **Verifier** `V(y, e*?, τ)` — three-valued, op-counted:
  `accept` iff a witness `e*` is exhibited with `He* = Hy` and `wt(e*) ≤ τ` (SOUND);
  `reject` iff a cheap sound lower bound on the coset weight exceeds `τ`;
  else `quarantine`.
  The cheap reject branch is **diagnostic** (a tight bound can itself be
  decoding-hard — RISK 1); the **false-quarantine band** of the cheap bound is the
  named **6.2** boundary and must be reported, not assumed away. Accept soundness is
  by construction.

## The attacker (frozen, named class)

**Attacker class = Prange information-set decoding (ISD).** Capacity = **ISD
iteration budget `B`** (each iteration: random size-`k` information set, GF(2)
Gaussian elimination to systematic form, read off the redundancy-supported error
candidate, accept if its weight `≤ τ`). The implementation is frozen as Prange
(no Lee–Brickell/Stern/MMT/BJMM in v1).

**Iteration convention:** `B` counts rank-valid Prange information-set trials.
Rank-deficient random draws are implementation overhead: they must be recorded
as `rank_fail_draws` and charged in measured attacker ops, but they do **not**
increment `B`. The analytic success curve below is read against rank-valid
trials; the receipt must report both predicted core ops and measured ops
including rank-fail overhead. If the implementation counts raw draws as `B`
instead, the prediction table is not the frozen comparator and the run voids.

**Honest scope of `C`:** the measured threshold is the one-wayness threshold
**against Prange ISD specifically** — a named, reproducible attacker class (the
measurement contract's "adversary class" parameter). A stronger ISD variant would
lower `C`, so the measured `C` is an **upper bound** on the true threshold under
best-known attacks. Extending the attacker class is a future slate, not a v1 edit.

## Capacity ladder (frozen) + pre-registered prediction

Predicted from the Prange model (expected iterations `N = C(n,w)/C(n−k,w) ≈ 7220`,
per-iteration ops `≈ 5.24×10⁵`, success after `B` iters `≈ 1 − e^{−B/N}`):

| Tier | budget `B` (ISD iters) | predicted invert-`e` witness success | predicted attacker ops |
| --- | ---: | ---: | ---: |
| Small | 64 | ≈ 0.9% | ≈ 3.4×10⁷ |
| Medium | 1,024 | ≈ 13% | ≈ 5.4×10⁸ |
| Large | 16,384 | ≈ 90% | ≈ 8.6×10⁹ |
| XL | 32,768 | ≈ 99% | ≈ 1.7×10¹⁰ |

**Pre-registered breakpoint:** the 50%-crossing at `B ≈ N·ln2 ≈ 5004` ISD
iterations `≈ 2.6×10⁹ ops`. **Check op-count = 8192 (flat)** → the find-vs-check gap
is `≈ 3.2×10⁵×` at the breakpoint (`2.6×10⁹ / 8192`) and `≈ 4.6×10⁵×` at the full
expected work factor `W ≈ 3.8×10⁹`. The
run reports the **measured** curve and breakpoint and whether they match this
prediction; a large deviation is itself a finding (the regime is weaker or stronger
than the Prange model — e.g., an early break would signal a structural weakness).

## Primary gates

| Gate | Required |
| --- | --- |
| Regime frozen + non-enumerable | `(n,k,w,τ,seeds)` frozen before any ISD run; `C(n,w) > 10¹² ` |
| Code validity | `G Hᵀ = 0` over GF(2) |
| Verifier cheap | check op-count `≪ C` (target `≤ 10⁴` vs `C ~ 10⁹` ops) |
| Verifier sound | 0 false accepts over a safe+unsafe battery (`accept ⟺ light witness`) |
| Privilege / label audit | `wt(e)`, `s`, and `e` are **scoring labels only**, never verifier or attacker-given inputs; the inverter receives **only** `z` |
| Determinism | byte-identical re-run under the frozen seeds |
| Inversion-`s` vacuity disclosed | report that `s` is unconditionally hidden (not a capacity result) |
| Clean threshold | the invert-`e` curve has a monotone transition with a locatable breakpoint `C` |
| Cost reported | attacker ops and verifier ops both reported; the gap stated at `C` |

## Verdict branches

- **bounded-positive measured one-wayness** — the invert-`e` curve rises from ~0 to
  ~1 with a clean breakpoint `C`, the verifier op-count stays `≪ C`, and the
  soundness/label gates pass: report the **measured capacity-relative one-wayness
  threshold `C` (against Prange ISD)** and the find-vs-check gap. This is the §4→
  receipt conversion.
- **6.1 certificate vacuity / 6.4 overhead** — the verifier op-count is not below
  the attacker effort (no gap): falsified; the certificate buys nothing.
- **6.3 broken-early** — the inverter succeeds at trivial capacity (`B` far below the
  predicted floor, e.g. the Small tier): a *weak* measured `C`; report it honestly
  (still a threshold, but the regime gives little one-wayness — a structural
  weakness, not a win).
- **6.5 boundary absence** — no clean transition / no locatable breakpoint:
  named quarantine.
- **void_run** — regime not frozen before the run, labels leaked to the verifier or
  attacker, the inverter given more than `z`, or non-deterministic.

## Required outputs (under `results/pvnp/certificate-syndrome-v1/`)

- `manifest.json` (frozen regime, seeds, attacker class, git sha, verdict);
- `verifier_access_declaration.json` (inverter receives only `z`; planted
  `wt(e)/s/e` are scoring-only; verifier inputs = `(y, exhibited witness e*)`);
- `capacity_curve.csv` (per tier: budget, invert-`e` success, attacker ops,
  verifier ops);
- `soundness_battery.csv` (safe+unsafe bodies, decisions, false accepts = 0);
- `op_count_report.json` (verifier ops; attacker ops at the breakpoint; the gap);
- `isd_rank_audit.json` (`rank_valid_trials`, `rank_fail_draws`, and whether `B`
  used the frozen rank-valid convention);
- `prediction_vs_measured.json` (the pre-registered curve vs the measured curve,
  the predicted vs measured breakpoint `C`);
- `false_quarantine_band.json` (the 6.2 boundary of the cheap reject bound);
- `falsifier_summary.md`, `README.md`.

A durable reviewed receipt belongs under `docs/pvnp/receipts/`.

## Anti-P-Hack rule

- Freeze `(n,k,w,τ,seeds)` **and the prediction table** before the ISD run. The
  measurement confirms or falsifies the prediction; the regime is **not** retuned to
  make the breakpoint land nicely.
- The inverter receives **only** `z` — never `y`, planted `e`, `wt(e)`, or `s`.
  The verifier receives `(y, exhibited witness e*)` — never planted `wt(e)` or `s`
  as a decision input.
- Deterministic seeds; report determinism.
- A smarter attacker class (Stern/MMT/BJMM) is a **new slate id**, not a v1 edit —
  the measured `C` is explicitly "against Prange ISD."

## Freeze checklist

- [x] Confirm `C(128,12) = 23,726,045,489,546,400 > 10¹²` (non-enumerable) and the
      Prange prediction (exact `N = 7224.27`, 50% breakpoint `5007.48` rank-valid
      trials; the rounded `~7220 / ~5004` table stands). Done 2026-06-04.
- [x] Freeze `code_seed = 2026128`, body seed base `= 7000000`, `T = 64`. Done.
- [x] Freeze the Prange ISD attacker spec and the ladder tiers (Small 64 / Medium
      1024 / Large 16384 / XL 32768). Done.
- [x] Freeze the Prange rank convention: `B` counts rank-valid information-set
      trials; rank-fail draws are charged to measured ops and audited separately. Done.
- [ ] Implement the ISD attacker (a separate, post-freeze build) and smoke it on a
      **throwaway** small regime (not the frozen one) to confirm it matches the
      analytic prediction before the frozen run.
- [x] Re-affirm: the inverter sees only `z`; planted `e`/`wt(e)`/`s` are scoring-only.
      Affirmed.

## Freeze rule

Edits allowed without a new slate id: typo fixes; path/command corrections; output
naming that preserves semantics. Edits requiring a new slate id: changing
`(n,k,w,τ)` or the seeds; changing the attacker class or the ladder tiers; giving
the inverter anything beyond `z`; using `wt(e)`/`s` as a verifier or attacker input;
retuning the regime after seeing the curve.
