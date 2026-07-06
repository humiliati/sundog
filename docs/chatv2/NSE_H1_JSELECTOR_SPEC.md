# NSE-H1 — Paired-Action J-Selector Spec (frozen; both rungs)

> 2026-07-06. Lift of H1 in `NSE_POST_AT_HYPOTHESES_SLATE.md` §3 into its frozen
> pre-registration — **every constant, gate, and branch is frozen here before any
> Delta_J exists.** Non-promotional: finite-Galerkin 32x32 proxy, no NSE claim, no
> C1 promotion; a positive strengthens proxy-faithfulness of the existing two-regime
> witness and nothing else. Imports `NSE_STATIONARITY_GATE_CHECKLIST.md` (H4)
> verbatim — this spec is its first importer.

## 1. Cells, stream, segments

- **Cells:** the C1 pair `(k_f=2, G=200)` and `(k_f=2, G=300)`; 32x32
  pseudo-spectral vorticity, `n_modes=16`, `K=3` (18-dim `Phi_K3`), `dt=0.01`,
  `nu=sqrt(1/G)`, semi-implicit Euler — the AT-3 config idiom, READ-ONLY import
  of the frozen harness. **Fresh truth stream, seed 0** (the banked witnesses on
  seed 20260528 are untouched; the banked `epsilon_K` radii are imported as fixed
  numbers, not re-derived).
- **Segments per cell:** burn-in `100,000`; **calibration segment `100,000`**
  steps; guard gap `5,000` (> tau; H4 item 1); **eval segment `400,000`** steps;
  lookahead tail `tau+1`. Decision instants every `250` steps on both segments
  (`n_cal = 400`, `n_eval = 1,600`). Calibration precedes eval (C1 convention).

## 2. The frozen cost and actuator

- **Cost (the slate's "natural first cost", inherited, not invented):**
  `J_a(s) = max over t in (s, s+tau] of E_low_K3` under action `a`, with
  `tau = 500` steps (5.0 time units — the v7 horizon). `E_low_K3` is the portable
  low-band velocity-energy observable (`low_energy`, the signature dot).
  **The window excludes `s`** (one registered deviation from the house inclusive
  idiom, with rationale: the state at `s` is identical under both actions by
  definition, so an inclusive window injects exact-zero `Delta_J` atoms whenever
  the excursion peak sits at `s` — a manufactured liveness confound). The
  `pi_hat` anchor below keeps the house **inclusive** window `[s, s+tau]`,
  because it must match the banked object it stands in for.
- **no_op:** the truth continuation (deterministic cell) — `J_0(s)` is read off
  the truth series directly.
- **damp_low_band (the one new registered operator):** from `v_0 = u(s)`, steps
  `1..tau_act` integrate with the damping term **`-mu_act * P_K3(v)` at the
  explicit stage** (AT-3's receipted nudging form with target 0), then steps
  `tau_act+1..tau` free-run. **`tau_act = 100`** steps (1.0 time unit),
  **`mu_act = 10`** (AT-3's receipted primary gain), `P_K3` = the K=3 signature
  band + conjugates. `J_1(s)` = the cost max over the same inclusive window.
- **Rejected alternatives (pre-registered, no retune after any read):**
  full-horizon damping (suppresses the maxed quantity nearly state-independently
  — `J_1` degenerates toward a constant and the selector collapses to a `J_0`
  ranking by construction); impulse damping (an unanchored amplitude constant
  with no receipted precedent). `tau_act` and `mu_act` are frozen; they are not
  dials.
- **Selector:** `Delta_J(s) = J_0(s) - J_1(s)`;
  `a_J(s) = damp_low_band iff Delta_J(s) >= m`.

## 3. Rung 0 — calibration + admission (per cell; agent-runnable, ~12 min)

- **Margin:** `m` = empirical `q_0.70` of `Delta_J` over the 400 calibration
  instants (target damp fraction 0.30 on calibration — the slate's number).
  `m` is **never** tuned on overlap with `pi_hat` or on any eval-side read.
- **Gates (all frozen; any failure = cell not admitted, stage-typed):**
  - **G1 held-out damp:** `mean(a_J)` over eval instants in `[0.20, 0.40]`.
  - **G2 stationarity (H4 items 2–3):** eval segment cut into 8 contiguous
    50,000-step blocks; every blockwise damp in `[0.10, 0.60]`; table reported
    before any other read.
  - **G3 atom clearance (H4 item 4):** mass of `Delta_J` within `±1e-6` of `m`
    at most `0.05`; ball-straddle reported.
  - **G4 paired-action liveness (slate mandate):** fraction of eval instants
    with `|Delta_J| <= 1e-12` at most `0.10`; `IQR(Delta_J) >= 1e-9`.
  - **G5 power:** usable eval instants `>= 800`; every block `>= 100`.
- **Reported, not gated:** overlap = `mean(a_J == y_pi)` on eval instants, where
  `y_pi(s) = J_0(s) > e_max_cal` and `e_max_cal` = `q_0.70` of `J_0` over the same
  calibration instants (the anchor `pi_hat` on this stream); medians of
  `J_0, J_1, Delta_J`; lag-1 autocorrelation of the `a_J` sequence.
- **Branch per cell:** `H1_CELL_ADMITTED`, else `NSE-H1-UNPOWERED` (cell- and
  stage-typed). **Apparatus checks (self-test, must pass before any run):**
  `mu=0` damped step bitwise-equals the plain step; the damping step equals
  AT-3's `nudged_step` with zero target; series-derived `J_0` bitwise-matches an
  explicit no_op rollout from the captured state (state-capture fidelity);
  quantile targeting and the atom detector behave on synthetic data.

## 4. Rung 1 — fiber transfer (owner-gated; runs only per §5)

- **Input:** the rung-0 export (per instant: `s`, `Phi_K3(u(s))`, `J_0`, `J_1`,
  `Delta_J`, `a_J`, `y_pi`). No re-integration.
- **Reads per admitted cell:** (i) **paired-fiber constancy** on `a_J` at the
  banked matched radii `epsilon_K = 0.060598` (G=200) / `0.066422` (G=300),
  `delta_action = 0.10`, unique-pair gate `>= 100` (all imported from the banked
  manifests / `PDE_C1_FIBER_PROTOCOL.md`); (ii) **kNN action-mismatch** `a_mm`
  (k=30) on `a_J`. **Comparators (frozen):** the same two reads on `y_pi` from the
  same stream; shuffled-`a_J` floor (20 shuffles, seed 3, mean reported); the
  banked manifest values (0.0367 / 0.0382 disagree fractions) cited for scale.
- **Power option A (pre-registered, one use, not a rescue):** if unique pairs
  `< 100` at stride 250, extend the export to stride 50 over the same segments
  (owner-run, ~35 min/cell), everything else unchanged.
- **Transfer criterion per cell (all three):** fiber disagree fraction on `a_J`
  `<= 0.05`; `<= 2x` the same-stream `y_pi` disagree fraction; shuffle floor
  `>= 0.25` (read liveness).
- **State-insufficiency half:** inherited, not rerun — `TWIN_STATE_CERTIFIED`
  is label-free geometry, banked at both regimes at the same `epsilon_K`.

## 5. Verdict table (frozen; no widening, no retune, failures final)

| condition | verdict |
| --- | --- |
| both cells admitted, both transfer, mean overlap `< 0.95` | `NSE-H1-JSELECTOR-POS` |
| both cells admitted, both transfer, mean overlap `>= 0.95` | `NSE-H1-COLLAPSED-PROXY` |
| any admitted cell fails transfer (gates passed, read powered) | `NSE-H1-PROXY-ONLY` |
| any cell fails admission | `NSE-H1-UNPOWERED` (cell-typed) |
| rung-1 pairs `< 100` after option A | `NSE-H1-UNPOWERED` (rung1-pairs) |

Registered diagnostic on a one-cell admission failure: rung 1 may still run on
the surviving admitted cell; a transfer **failure** there escalates the verdict
to `NSE-H1-PROXY-ONLY` (one cell suffices for a negative); a transfer success
does **not** upgrade the verdict (no positive from half the pair).

## 6. Deliverables and staged commands

- Rung 0: `scripts/nse_h1_jselector_admission.py` →
  `results/proof/nse-h1-g{200,300}/` + `NSE_H1_ADMISSION_RECEIPT.md`.
  `python scripts/nse_h1_jselector_admission.py --grashof 200 --out results/proof/nse-h1-g200`
  (then `--grashof 300`). Resumable (truth + rollout checkpoints); `--smoke` and
  `--self-test` are non-verdict.
- Rung 1 (built only after both admission receipts; owner-gated):
  `scripts/nse_h1_jselector_fiber.py` → `NSE_H1_FIBER_RECEIPT.md`.

## 7. Does not claim

No infinite-dimensional NSE statement; no C1 promotion; no new physics — the
actuator is a registered numerical operator on a truncation, not a control-theory
claim about Navier–Stokes. Overlap with `pi_hat` is a reported diagnostic and a
branch discriminator, never a tuning signal. A positive says exactly: *the
regime-2 fiber structure of the C1 witness is visible under an action-value
selector, not only under the threshold proxy.* `docs/chatv2/` stays no-publish.

Cross-refs: `NSE_POST_AT_HYPOTHESES_SLATE.md` §3,
`NSE_STATIONARITY_GATE_CHECKLIST.md`, `AT3_NUDGING_LEDGER_SPEC.md` (the nudging
form and gain), `PDE_C1_FIBER_PROTOCOL.md`,
`results/proof/c1-paired-fiber-g{200,300}/manifest.json` (the banked radii),
`PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md` (comparator slate rows),
`NSE_ATTRACTOR_TAIL_SYNTHESIS.md` (closed context).

---

> **Post-run status (2026-07-06): Rung 0 filed — `NSE-H1-UNPOWERED` (cell-typed:
> g300, stages G1+G2)** in `NSE_H1_ADMISSION_RECEIPT.md`. G=200 `H1_CELL_ADMITTED`
> (damp 0.307, all gates clean, overlap 0.515). Two receipted findings bind any
> continuation: (1) the v1 actuator has **uniformly negative action value** on both
> cells (G=200 frac(Delta_J>0) = 0.0000; rebound overshoot) — the selector formed as a
> least-harm ranking, not a benefit selector; (2) the G=300 collapse is a
> **calibration-vs-eval, Delta_J-specific** level shift with a flat blockwise table —
> the phase structure lives in the controllability variable, not the magnitude
> variable. Live owner fork (receipt §"Owner fork"): bounded v1.1 actuator amendment
> (μ_act 10 → 1.0, recommended), G=200 diagnostic rung 1 (least-harm fence), or close.

## 8. v1.1 Formation Amendment (commissioned 2026-07-06; pre-committed stop)

> Owner selected the bounded v1.1 actuator amendment after the rung-0 receipt
> (Finding 1: the v1 actuator has uniformly negative action value — a least-harm
> ranking is not the slate's benefit selector). This amendment changes exactly one
> frozen constant and adds one gate. **It is frozen here before any v1.1 number
> exists; no fiber or adjudication number has been read at any point.**

- **Override to §2:** `mu_act: 10 -> 1.0` (band attenuation over the actuation
  window ~e^-1 instead of ~e^-10 — a mild drag pulse; the measured
  rebound-overshoot mechanism is the reason). `tau_act = 100`, the horizon, the
  exclusive window convention, segments, stride, seed, and gates G1–G5 are all
  unchanged.
- **New gate (v1.1 only) — G6 benefit-mass:** frac(`Delta_J` > 0) over eval
  instants `>= 0.05` per cell — the action-value premise requires measured
  positive-benefit instants to exist. Reported beside it (fence-bearing, not
  gated): the sign of `m`. If `m > 0`, the selected class is benefit-positive at
  margin (unfenced benefit selector); if `m <= 0` with G6 passing, the selector
  carries the fence "mixed most-benefit/least-harm ranking".
- **Pre-committed stop:** if any cell fails v1.1 admission (G1–G6), H1 closes at
  `NSE-H1-UNPOWERED` **final — no v1.2.** The v1 G=200 diagnostic rung-1 option
  survives independently (receipt fork item 2).
- **Truth reuse (registered):** the truth stream is identical by construction
  (same seed and segments; only the rollout action changes), so v1.1 reuses the
  v1 truth checkpoints bitwise (`truth_ckpt.npz` copied into the v1.1 dirs).
- Commands:
  `python scripts/nse_h1_jselector_admission.py --grashof 200 --out results/proof/nse-h1-g200-v11 --formation-version v1.1`
  (then `--grashof 300 --out results/proof/nse-h1-g300-v11`).

> **Post-run status (2026-07-06): v1.1 rung 0 filed — BOTH cells
> `H1_CELL_ADMITTED`** (receipt, v1.1 section). G=200 damp 0.302 / G=300 damp
> 0.262, blockwise flat, atom 0, **benefit_mass 1.0000 both cells, m > 0 both**
> — the unfenced benefit selector formed; the v1 degeneracy was an
> actuation-strength dial boundary, and the G=300 controllability level shift
> vanished with the rebound regime (receipt Findings 3–4). Pre-committed stop
> did not fire. **Rung 1 unblocked, owner-gated** — tooling
> `scripts/nse_h1_jselector_fiber.py` self-tested 4/4 on synthetic data only;
> staged commands in the receipt; no real export read.
