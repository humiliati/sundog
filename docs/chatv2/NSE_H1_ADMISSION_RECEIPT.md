# NSE-H1 Rung 0 — Admission Receipt

> 2026-07-06. Lock runs of `NSE_H1_JSELECTOR_SPEC.md` rung 0 (truth-only + paired-action
> rollouts; artifacts `results/proof/nse-h1-g{200,300}/`). Self-test 6/6 (bitwise
> apparatus checks) and a G=200 smoke preceded the locks. **Non-promotional. No fiber,
> kNN, or adjudication number was computed or read.** All gates and constants were
> frozen in the spec before any `Delta_J` existed.

## Rung-0 branch table

| cell | branch | damp_eval (gate [0.20,0.40]) | blockwise (8 blocks, gate [0.10,0.60]) | atom | liveness |
| --- | --- | --- | --- | --- | --- |
| G=200 | **`H1_CELL_ADMITTED`** | **0.307** (cal 0.302) | 0.290–0.315, flat | 0.000 | zfrac 0.000, IQR 1.37e-2 |
| G=300 | **`NSE-H1-UNPOWERED`** (G1, G2) | 0.074 (cal 0.302) | 0.070–0.075, flat | 0.000 | zfrac 0.000, IQR 2.86e-2 |

Reported, not gated: overlap with the same-stream `pi_hat` anchor = **0.515** at G=200
(independence baseline ≈ 0.578 — mildly anti-correlated) and 0.648 at G=300 (baseline
≈ 0.689). `a_J` lag-1 autocorrelation −0.184 / −0.080. n_eval = 1,600 per cell,
n_cal = 398, margins m = −0.6016 / −0.6182, `e_max_cal` = 0.7344 / 0.8823.

## Finding 1 (binding on any downstream read): the registered actuator has uniformly negative action value

`Delta_J = J_0 − J_1` is **negative almost everywhere on both cells**: G=200
frac(`Delta_J` > 0) = **0.0000** (range [−0.630, −0.594] — a 0.036-wide band);
G=300 frac = 0.0010 (q99 = −0.145). Median `J_1` ≈ 1.34 / 1.52 vs median `J_0` ≈ 0.73 /
0.87: the frozen actuator (μ=10 for 1.0 time unit ⇒ band attenuation ~e⁻¹⁰) crushes the
low band so deep off-attractor that the free-run rebound **overshoots the natural
excursion scale** — damping now makes the τ-horizon max worse at essentially every
sampled instant. Bitwise apparatus checks (T1/T2/T4) rule out an implementation
artifact; this is the dynamics.

**Consequence, typed:** under this actuator the "paired-action value selector"
degenerates to a **least-harm ranking** on a thin relative sliver. The G=200 admission
is formally valid (every gate measures label form, not sign), but any rung-1 positive
would carry the fence *"value = relative least-harm; no positive-benefit instants
exist under the registered action"* — it would not test the slate's intended "damp
because damping improves the registered future cost." μ_act and τ_act are frozen and
were not retuned after this read.

## Finding 2: the G=300 failure is a new, sharper localization of the known wander

The G=300 collapse (eval damp 0.074) is **not** the AT-4b envelope wander shape: all 8
eval blocks are flat (0.070–0.075) — the eval segment is internally stationary. The
shift is **calibration-vs-eval and `Delta_J`-specific**: the excursion-magnitude
functional is segment-stationary (`J_0` median 0.8745 → 0.8734; `y_pi` transfers
0.302 → 0.278) while the action-value functional shifts (`Delta_J` q70 −0.618 → −0.656).
Fifth appearance of the G=300 slow-phase structure, now typed: **the phase structure
lives in the controllability variable, not the magnitude variable, at these scales.**
G=200 transfers perfectly on both functionals (q70 shift 4e-4).

## Verdict (per the frozen table)

**`NSE-H1-UNPOWERED` (cell-typed: g300, stages G1+G2).** The registered diagnostic
remains live: rung 1 may run on the admitted G=200 cell — a transfer **failure** there
escalates to `NSE-H1-PROXY-ONLY`; a success does **not** upgrade the verdict. No gate
was widened; no constant retuned; no fiber number read.

## Owner fork (staged, not spent)

1. **Commission a bounded H1-v1.1 formation amendment** (recommended; the AT-4b v1→v1.1
   idiom — label-formation amendment with zero downstream reads): one registered change,
   **μ_act 10 → 1.0** (attenuation ~e⁻¹, a mild drag pulse instead of a band crush),
   everything else identical, pre-committed stop if `Delta_J` still has no positive
   mass or formation fails. Rationale: Finding 1 says the v1 actuator is the wrong
   probe for action *value*; a benefit-bearing actuator is what H1's hypothesis is
   about. New registration section in the spec before any run.
2. **Fire the registered G=200 diagnostic rung 1** on the v1 export (least-harm fence
   attached; escalate-on-failure only). Compatible with 1 — the export is banked.
3. **Close H1 at `NSE-H1-UNPOWERED`** as final.

Cross-refs: `NSE_H1_JSELECTOR_SPEC.md`, `NSE_POST_AT_HYPOTHESES_SLATE.md` §3,
`NSE_STATIONARITY_GATE_CHECKLIST.md` (imported; its blockwise table fired first, as
doctrine requires), `AT4B0_ADMISSION_RECEIPT.md` (the wander's prior form),
`results/proof/nse-h1-g{200,300}/h1_{summary.json,export.npz}`.

---

# v1.1 Formation Amendment — Rung 0 Receipt (2026-07-06)

> Lock runs of spec §8 (`mu_act = 1.0`; one constant changed, gate G6 added; frozen
> pre-run; truth checkpoints reused bitwise as registered). Artifacts
> `results/proof/nse-h1-g{200,300}-v11/`. A v1.1 smoke preceded the locks.
> **No fiber or adjudication number was read.**

## v1.1 branch table — both cells admitted

| cell | branch | damp_eval | blockwise (8) | atom | benefit_mass (G6 ≥ 0.05) | m | overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G=200 | **`H1_CELL_ADMITTED`** | 0.302 | 0.280–0.325 | 0.000 | **1.0000** | **+0.0251** | 0.634 |
| G=300 | **`H1_CELL_ADMITTED`** | 0.262 | 0.260–0.270 | 0.000 | **1.0000** | **+0.0733** | 0.624 |

Liveness clean both cells (zfrac 0.000; IQR 1.18e-2 / 5.79e-2 — the wander regime
carries ~5x the benefit spread). Overlap independence baselines ≈ 0.580 / 0.606: the
benefit selector is mildly positively correlated with `pi_hat`, far below the 0.95
collapsed-proxy ceiling.

## Finding 3: the benefit selector exists — and the v1 degeneracy was a dial boundary

Under the mild drag pulse, `Delta_J > 0` at **every** eval instant on both cells
(benefit_mass 1.0000) and both margins are positive — the **unfenced benefit
selector** (`m_sign_fence = benefit-positive-at-margin`): the selected class is
"damping helps most", the slate's intended object. Between attenuation ~e⁻¹⁰ (v1:
frac positive 0.0000) and ~e⁻¹ (v1.1: 1.0000) the actuator crosses a
rebound-overshoot boundary; the sign of action value on this cell is a property of
actuation strength, not of the flow alone.

## Finding 4 (sharpens Finding 2): the G=300 level shift belonged to the rebound regime

Under the benefit-bearing actuator the calibration-vs-eval shift is **gone**: cal
0.302 → eval 0.262, blockwise dead-flat 0.260–0.270. The v1 nonstationarity was a
property of the *rebound-overshoot* `Delta_J` (overshoot amplitude rides the slow
phase), not of the controllability variable per se. Finding 2's typing narrows to:
the G=300 phase structure couples to off-attractor relaxation, not to mild-drag
action value at these scales.

## Disposition (per the frozen table)

Both cells admitted ⇒ **rung 1 unblocked, owner-gated** (spec §4). The v1.1
pre-committed stop did not fire. Rung-1 tooling `scripts/nse_h1_jselector_fiber.py`
is built and self-tested on synthetic data only (4/4: fiber-constancy, random-label
calibration ≈ 0.42, kNN mirror, pair gate) — **no real export has been read.**
Staged commands (owner go):

```
python scripts/nse_h1_jselector_fiber.py --grashof 200 --export results/proof/nse-h1-g200-v11/h1_export.npz --out results/proof/nse-h1-g200-v11
python scripts/nse_h1_jselector_fiber.py --grashof 300 --export results/proof/nse-h1-g300-v11/h1_export.npz --out results/proof/nse-h1-g300-v11
```

Verdict assembly after both reads: both transfer ∧ mean overlap 0.629 < 0.95 ⇒
`NSE-H1-JSELECTOR-POS`; any transfer failure ⇒ `NSE-H1-PROXY-ONLY`; pairs < 100
after option A ⇒ `NSE-H1-UNPOWERED(rung1-pairs)`.
