# Attack-B — Internal Review Packet (staged, unsent)

> 2026-06-04. **Staged** for a small, bounded external *methods* review — **not sent**.
> Internal, withheld (`docs/deconfound/` is in `DOCS_NO_PUBLISH`). Tier **R1.5**. Deploy this
> packet only on the review trigger (an R2 emergent-functional result, or folding into the
> NSE-C1 packet). The ask is **"is this measurement valid under its stated boundary?"** — not
> "is it important?".

## 1. The reviewable claim (one sentence)

> On median-binarized sklearn digit features, a supervised functional objective produced a
> closure bracket (`k_func ≪ k_state`) that a state-reconstruction objective did not, under a
> passed input-deconfound precheck and a `u_null` control — and that bracket **degrades
> monotonically** as the de-confound is deliberately leaked.

## 2. Claim boundary

- **R1.5** — real feature *distribution* + real (MLP) *architecture*, but the functional is
  **constructed**.
- Substrate: `sklearn.datasets.load_digits` (8×8 → 8 pooled, median-binarized bits). Real
  correlation, but **mild** (~0.11).
- Functional: `u = XOR(b_0,b_1,b_2)` — a constructed parity over *visible* features.
  **Linearly-hidden but nonlinearly-trivial** (self-test: linear probe ≈ chance, MLP = 1.000).
- **Does NOT claim:** JEPA behavior (the functional-keeper is *supervised-on-u*, a clean
  instance of "targets the functional," not a JEPA implementation); Othello / computed-state
  (that slate failed — its natural functionals are nonlinear); R2 / "more than we know" (the
  functional is ours).
- The de-confound is **linear-only by construction** — and that *is* the ceiling.

## 3. Experiment diagram

```
real sklearn digits (8×8)
      │  pool 4×2, median-binarize
      ▼
  b ∈ {0,1}^8   ── real features, mild correlation, de-confound-clean (0-pre)
      │
      ├───────────────► u = XOR(b0,b1,b2)   (constructed; linearly-hidden, nonlinearly-trivial)
      │
  ┌───┴────────────────────────────┐
  ▼                                ▼
STATE-KEEPER (reconstruct all b)   FUNCTIONAL-KEEPER (predict u)
  keeps state                      keeps functional, discards b_j∉S
  └────────────────┬───────────────┘
                   ▼
   determining-shadow read on each body:  k_func(u)  vs  k_state(b_j∉S),  + u_null control
```

## 4. The arc

| cell | question | result |
| --- | --- | --- |
| **0-pre** (model-free) | does the constructed parity stay input-undecodable on real correlated features? | de-confound boundary located: clean to bit-corr ~0.33, leaks ≥0.5; **real digits anchor det 0.077 → clean** |
| **0B** | does a functional objective keep `u` and discard the state, where a generative objective doesn't? | **`attack_b_closure_confirmed`** — double dissociation: func-keeper `k_func=2`/`k_state=none` (bracket 5/5); state-keeper `k_func=none`/`k_state=4` (0/5); median gap 7; `u_null` clean. Control: state-keeper *does* determine `b_j` at k=4 → func's `k_state=none` is genuine discard |
| **0C** | is the de-confound load-bearing, or decorative? | **`deconfound_load_bearing_confirmed`** — `state_det_u` rises monotonically `0.192→0.423→0.522→0.654` with the input leak (`0.077→0.459`); rise 0.461; not confounded (HOLD `k_func` hits=0 → 0B replicates) |
| **self-test** | is the linear de-confound too weak a leakage test? | linear `b→u` det 0.077 (holds) vs nonlinear MLP det **1.000** vs random-label control −0.036 → `u` is an **explicit input-function**, linear-only by construction = the R1.5 ceiling |

## 5. Exact reviewer questions (adjudicate validity, not importance)

1. **Is the de-confound sufficient?** It is a *linear* input-probe ≈ chance (det 0.077); the
   self-test shows `u` is nonlinearly trivial. Is the **linear bar** the right validity gate for
   a *linear* determining-shadow read — or must it be nonlinear/causal (amnesic / LEACE)? Does
   0C's degradation curve adequately *demonstrate* the linear de-confound is load-bearing?
2. **Is the determining-shadow read valid?** Closed-form ridge probe; best-size-k subset
   selected on probe-train, scored on held-out; 1000-perm **label-permutation
   selection-corrected null** (p≤0.01); "invalid if subset chosen on held-out rows" guard. Are
   the selection correction, split design, small dataset (n=1797), and linear-probe choice
   sound? Is the state-keeper a fair "state-is-present, probe-can-find-it" control (it
   determines `b_j` at k=4)?
3. **Does 0C address the main confound?** The main worry: "func-keeper keeps `u`, state-keeper
   keeps `b`" is partly *pre-ordained* by the objectives. Does 0C — the state-keeper exposing
   `u` **monotonically more** as the de-confound leaks — actually show the de-confound is the
   load-bearing precondition, or is there a residual confound (e.g. the nonlinear-amplification
   baseline `s0=0.192`)?
4. **Is the R1.5 boundary honest, or should it be lower?** And what would be required before any
   R2 language?

**Reviewer types.** (a) *mechanistic/probing* — attacks the probe, null, leakage, objective
contrast; (b) *ML/stat methods* — attacks split design, selection correction, small dataset,
ridge/logistic probe, seed handling; (c) optional *friendly outsider* — reads the claim cold and
says what they think it claims (catches language drift).
**What not to ask:** "is this important?" — ask "is this measurement valid under its boundary?"

## 6. Receipts + runnable commands

All on `sklearn` digits (no download); numpy / sklearn / torch (1080-class CPU fine). Full arc
≈ 10 min.

| cell | command | receipt |
| --- | --- | --- |
| 0-pre | `python scripts/deconfound_synth_b_datacheck.py` | `results/deconfound/attack-b-0pre-digits-2026-06-04.txt` |
| 0B | `python scripts/deconfound_attack_b_phase0_closure.py --out results/deconfound/attack-b-phase0-closure` | `docs/deconfound/PHASE0B_ATTACK_B_CLOSURE_RESULTS.md` ; `.../attack-b-phase0-closure/summary.json` |
| 0C calib | `python scripts/deconfound_attack_b_alpha_calibration.py` | `results/deconfound/attack-b-alpha-calibration-2026-06-04.txt` |
| 0C | `python scripts/deconfound_attack_b_phase0c_stress.py --out results/deconfound/attack-b-phase0c-stress` | `docs/deconfound/PHASE0C_DECONFOUND_STRESS_RESULTS.md` ; `.../attack-b-phase0c-stress/summary.json` |
| self-test | `python scripts/deconfound_nonlinear_selftest.py` | `results/deconfound/attack-b-nonlinear-deconfound-2026-06-04.txt` |

Specs: `PHASE0B_ATTACK_B_CLOSURE_SPEC.md`, `PHASE0C_DECONFOUND_STRESS_SPEC.md` (both
pre-registered + locked). Instrument shared with `SUNDOG_V_DECONFOUND.md`.

## 7. Honest limitations (stated up front)

- digit correlation is **mild** (~0.11) → the de-confound holds partly for that reason (0-pre
  leak boundary at ~0.5).
- functional is **constructed** (parity over visible features) → linearly-hidden,
  nonlinearly-trivial → does **not** advance "more than we know."
- **supervised-on-u ≠ JEPA**; it is the clean instance of a functional-targeting objective.
- small dataset (1797), MLP bodies, linear probe.
- the closure contrast is *partly* objective-pre-ordained; the non-trivial **measured** content
  is (a) the func-keeper *discards* irrelevant state (0B `k_state=none`, with the state-keeper
  control proving the probe finds it), and (b) the state-keeper's exposure *tracks the
  de-confound leak* (0C monotone curve).

---

*Sundog Research Lab — Attack-B internal review packet. Staged, unsent. Withheld. R1.5.*
