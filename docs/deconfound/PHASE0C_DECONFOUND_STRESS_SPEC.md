# Deconfound Phase 0C - De-confound-Stress Boundary Spec

> 2026-06-04, LOCKED + EXECUTED. Verdict:
> `deconfound_load_bearing_confirmed` (see
> `PHASE0C_DECONFOUND_STRESS_RESULTS.md`). This cell deliberately degrades the input-deconfound and asks
> whether the Phase-0B closure double-dissociation degrades with it — i.e. whether the
> de-confound is **load-bearing or decorative**. It is a boundary-locate (Phase-7c structure
> on a new knob), not a new rung.
>
> **Amendment (2026-06-04, post-smoke; locked before execution):** the headline changes from a binary
> state-keeper `k_func` flip to a **continuous `state_det_u`** (§4/§5). The timing smoke showed
> the de-confound leak tops out at det 0.459 (alpha=2.5), **below** the inherited `k_func` bar
> (det >= 0.70), so a binary flip is structurally pinned at `none` and would false-read as
> `closure_robust_to_leak`; reaching det 0.70 needs near-degenerate correlation (bits ->
> `sign(g)`). The continuous measure reads the dissociation degradation at non-degenerate leak.
>
> **Post-state-det smoke repair (2026-06-04; locked before full run):** the re-smoke validated the
> load-bearing trend but found a weak HOLD baseline (`state_det_u = 0.148` for seed 0 at
> `alpha=0`). This is a refinement of Phase 0B, not a retro-flag: Phase 0B's inherited claim was
> that the state body does **not determine** `u` at the 0.70 bar, not that every continuous
> lower-signal read is exactly zero. Phase 0C therefore gates on the **baseline-subtracted rise**
> (`sdeep - s0`) and retro-flags 0B only if the inherited HOLD state-keeper `k_func(u)` read is
> finite in a majority of interpreted seeds. The runner was patched for this baseline-relative
> branch logic before the full run.

## 1. Decision Lock

Phase-0B confirmed a closure double dissociation on a de-confound-**clean** substrate
(functional-keeper keeps `u`, discards outside state; state-keeper keeps state, does not expose
`u`). Phase 0C sweeps the substrate from clean to **leaking** and locates where the
dissociation collapses.

**Frozen:** everything in `PHASE0B_ATTACK_B_CLOSURE_SPEC.md` (locked) — `u = XOR(b_0,b_1,b_2)`,
the state-keeper / functional-keeper bodies, the MLP architecture, optimizer, the
determining-shadow read (§4), the selection-corrected null, the `u_null` control, the bracket
definition.

**Swept (the only new variable):** input-feature correlation, via an injected shared factor,
which controls how linearly `u` leaks into the input — i.e. the de-confound strength.

**Repurposed gate:** the Phase-0B **de-confound-replay abort is suspended.** `det(u|b)` is no
longer a pass/fail gate; it is the **recorded independent variable** of the sweep.

**Dropped:** the read-dim-3 capacity diagnostic (irrelevant here — the headline is the
state-keeper's continuous `state_det_u`, and the functional-keeper is expected to bracket at
every rung).

## 2. Substrate + correlation knob

Start from the frozen Phase-0B real digit features (8 pooled blocks). **Standardize** each
feature (z-score over the table, so the knob is scale-free), inject a per-sample shared
Gaussian factor, then median-binarize:

```text
b_j = 1[ zscore(feat_j) + alpha * g  >  median_over_samples ],   g ~ N(0,1) shared across j
u   = XOR(b_0, b_1, b_2)
```

- `alpha = 0` reproduces the Phase-0B substrate **exactly** (median of z = median of feat) —
  the de-confound-clean control rung.
- `g` is a single fixed per-sample vector generated once by
  `default_rng(20260604).standard_normal((n,1))` and reused across all alpha rungs. This
  makes alpha a pure strength knob rather than a new random direction per rung.
- rising `alpha` makes all 8 bits share `g` -> correlation rises -> `parity(b_S)` becomes
  linearly predictable -> the de-confound leaks (validated direction: the 0-pre `s=3` copula
  curve crosses `det = 0.10` near bit-corr ~0.4 and `det > 0.20` near bit-corr ~0.5).

**Model-free alpha calibration (completed before lock):**
`python scripts/deconfound_attack_b_alpha_calibration.py`, receipt
`results/deconfound/attack-b-alpha-calibration-2026-06-04.txt`.

**Pre-registered rungs (post-calibration lock):** `alpha in {0.0, 0.75, 1.0, 2.5}`.

| alpha | bitcorr | det(u\|b) | label | role |
| ---: | ---: | ---: | --- | --- |
| 0.00 | 0.116 | +0.077 | HOLD | exact Phase-0B clean control |
| 0.75 | 0.234 | +0.190 | MARG | near-boundary stress |
| 1.00 | 0.334 | +0.231 | LEAK | first measured leak rung |
| 2.50 | 0.650 | +0.459 | LEAK | deep-leak rung |

The full calibration curve is reported in the receipt. The realized de-confound
`det(u|b)` and bit-correlation are still measured and reported at each rung during the
stress run; these rungs are locked from the calibration and are not tuned during the model
run.

**Seeds:** `{0,1,2}` per rung (boundary-locate, as Phase-7c). `n_perm = 1000`.

## 3. The read (per rung, per seed)

Identical to Phase-0B §3-§4, run at each locked `alpha`:

1. **de-confound det** `det(u|b)` (linear LogReg, 5-fold) — the independent variable.
2. **train** state-keeper + functional-keeper (matched init per seed, frozen 0B architecture).
3. **learned-body gates** (0B §3): functional-keeper `u` det>=0.70 & acc>=0.80; state-keeper
   mean-bit det>=0.70 & acc>=0.80; split sanity. (The de-confound-replay gate is suspended.)
4. **determining-shadow read** (0B §4): `k_state(b_j∉S)` and `k_null` for both bodies (the
   `u_null` control), plus **`state_det_u`** for the state-keeper (§4) — the max
   selection-corrected-significant `det(u)` over k, with the held-out-selection guard. The
   functional-keeper's `k_func(u)` at the 0.70 bar is recorded as context only.

## 4. Headline statistic — the state-keeper's continuous exposure of `u`

The de-confound leak cannot reach the determination bar (det >= 0.70) without driving the
substrate to degeneracy (all bits -> `sign(g)`), and the state-keeper is not trained on `u`, so
a **binary** `k_func` flip is structurally pinned at `none` — the wrong instrument for a
*degradation* question. The discriminating signal is therefore **continuous**:

```text
state_det_u = max over k in {1..8} of the best-subset held-out det(u) from the STATE-KEEPER
              body, restricted to subsets whose selection-corrected permutation p <= 0.01
              (0 if no k is significant).
```

This is the Phase-0B §4 determining read on the state body, reporting the **max significant
determination** of `u` rather than the first crossing of the 0.70 bar. Per rung, `state_det_u`
is the **median over interpreted seeds** (a rung needs >= 2 interpreted seeds after
learned-body gates, else it is void).

The headline is the **trend of `state_det_u` vs the realized de-confound det** across rungs:

| de-confound | expected `state_det_u` | meaning |
| --- | --- | --- |
| HOLD (det ~ 0.08) | low but measured, not assumed zero | state body may weakly expose `u` below the 0.70 determination bar; Phase 0B remains clean if inherited `k_func(u)` stays `none` |
| LEAK (det up to ~0.46) | rises toward the input-leak level | `u` leaks into the input the state body carries -> dissociation **degrades** |

**Pre-registered effect size.** Let `s0 = state_det_u` at the HOLD control (`alpha=0`) and
`sdeep = state_det_u` at the deep-LEAK rung (`alpha=2.5`). The state body **exposes-more** if
`sdeep - s0 >= 0.15`; it **stays-flat** if `sdeep - s0 < 0.15`. The intermediate rungs
(`alpha=0.75, 1.0`) are reported to show whether `state_det_u` rises monotonically with det
(corroborating context, not a gate). `s0` is reported as the HOLD baseline; by itself it is not
a retro-flag unless the inherited state-keeper `k_func(u)` read at the 0.70 bar is finite in
at least 2 of the 3 interpreted HOLD seeds.

(Report-only context: the functional-keeper still reaches `k_func` at the 0.70 bar at every
rung — trivially in the LEAK zone, since `u` was in the input — so it is *not* evidence of
objective-driven keeping there. The state-keeper's `k_state(b_j∉S)`, both bodies' `k_state`,
`keeper_gap`, and the `u_null` control are recorded per rung; only `state_det_u` vs de-confound
det is the headline.)

## 5. Branches

Precedence (first match wins): voids, then the inherited-HOLD retro-flag, then the
baseline-subtracted `sdeep - s0` classification (§4).

| branch | condition | reading |
| --- | --- | --- |
| `closure_void_unlearned` | split gate fails, or either required rung (`alpha=0.0` HOLD control, `alpha=2.5` deep LEAK) has fewer than 2 interpreted seeds after learned-body gates | no phase result |
| `closure_void_control` | `u_null` clears (`k_null` finite, `p<=0.01`) in any interpreted body/seed | instrument hallucinated; no result |
| `rungs_missed_boundary` | the deep-LEAK rung's realized de-confound `det <= 0.20` | sweep under-stressed; refine `alpha` upward, re-pose |
| `closure_confounded_throughout` | at `alpha=0`, the inherited state-keeper `k_func(u)` read is finite in >=2 interpreted HOLD seeds | would retro-flag the Phase-0B read at its own bar; investigate before any further use |
| `deconfound_load_bearing_confirmed` | `sdeep - s0 >= 0.15` | **the de-confound is the necessary precondition** — input leak drives state-body exposure above the measured HOLD baseline, and the dissociation degrades with it; shown, not asserted |
| `closure_robust_to_leak` | `sdeep - s0 < 0.15` | the body's nonlinearity blocks the input leak: the body-level closure read is **robust to input-deconfound leakage** — a genuine, stronger-than-expected finding; flag for follow-up |

## 6. Integrity boundaries

**Tier: R1.5 methods-validation.** This does **not** lift the ceiling — the functional is still
constructed. It demonstrates that Phase-0B's core validity gate does real work.

**Allowed (on `deconfound_load_bearing_confirmed`):**
> On real digit features with an injected correlation knob, the closure double-dissociation
> degraded as the input-deconfound leaked: the state-reconstruction body began exposing the
> constructed functional `u` once `u` became linearly present in the input. This locates the
> input-deconfound as the necessary precondition for the determining-shadow closure read.

**Forbidden:**
- claiming the injected correlation is "natural" (it is injected; the native-correlation
  variant is under-pooled pixels, a separate cell);
- model-discovered functional; Othello rescue; real-JEPA behavior; R2 / "more than we know".

## 7. Run order

1. Operator lock review.
2. Calibration is already complete:
   `python scripts/deconfound_attack_b_alpha_calibration.py`.
3. The runner `scripts/deconfound_attack_b_phase0c_stress.py` is written (imports the frozen
   Phase-0B `Model` / `train_model` / `read_body` / `bracket` / `determine_k`; no edit to the
   locked 0B runner; substrate replicates the calibration exactly, with an `alpha=0 == 0B`
   assertion). It now reports **`state_det_u`** (max selection-corrected-significant `det(u)`
   over k, §4) instead of the binary `k_func` flip, and its verdict is baseline-relative:
   `closure_confounded_throughout` uses the inherited HOLD `k_func` read, while
   `deconfound_load_bearing_confirmed` uses `sdeep - s0`.
4. Timing smoke complete. The pre-amendment smoke validated machinery + substrate
   (dets match calibration, ~4-5 min full estimate). The post-`state_det_u` re-smoke wrote
   JSON in about 19.5 seconds and triggered the baseline-relative repair in §4/§5.
5. Run all rungs; record verdict + the realized det / `state_det_u` (`s0`, `sdeep`) table in
   `docs/deconfound/PHASE0C_DECONFOUND_STRESS_RESULTS.md`.

## 8. Readback checklist

- git SHA + script SHA; reuse-hash of the imported 0B runner;
- alpha-calibration receipt path and the four locked rung rows;
- per-rung realized `det(u|b)`, bit-correlation, HOLD/MARG/LEAK label;
- per-rung per-seed learned-body gates;
- per-rung per-seed `k_func`/`k_state`/`k_null` for both bodies + `keeper_gap`;
- the state-keeper `state_det_u` (with `s0`, `sdeep`, `sdeep - s0`) vs realized-det table (the headline);
- the HOLD inherited state-keeper `k_func(u)` majority check used only for the retro-flag branch;
- selection-corrected null summary; branch verdict; allowed/forbidden language from §6.

---

*Sundog Research Lab - Deconfound Phase 0C de-confound-stress boundary spec. Locked + executed
2026-06-04; verdict `deconfound_load_bearing_confirmed`.*
