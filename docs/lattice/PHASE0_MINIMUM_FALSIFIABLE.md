# Lattice-Deduction Phase 0 — The Minimum Falsifiable Cell (pre-registration)

> 2026-06-02. **Frozen pre-registration; execution DEFERRED** to model access
> (no checkpoint released — see [`LANE_CHARTER.md`](LANE_CHARTER.md) §Access). The
> smallest pre-registered experiment that could **falsify** the lane claim: *the
> LDT realizes a certified state-insufficient, control-sufficient regime-2 with a
> genuinely high-dimensional neural body.* Phase 0 answers one gate — the **B2
> twin-state decision-vs-reconstruction** test — before any B1 fingerprint or B3
> soundness-frontier work is built.
>
> **Headline probe chosen (owner, 2026-06-02): B2.** It is the only probe that can
> yield a **certified** (not measured-marginal) witness, because the
> state-insufficiency leg is certified by the abstraction's construction rather
> than estimated.

## 0. The mechanism this cell is built around

Sound abstract interpretation gives, *for the ideal operator*, a control-sufficient
decision over a state-insufficient abstraction — **Target A, folklore, not
claimed** (the same posture as C1's Mori–Zwanzig closure: the mechanism is known,
don't reprove it). The novel, falsifiable content is entirely about the **learned**
model:

1. The conflict head and the eliminations are **learned**, so the model's
   "soundness" is *empirical* — it can make a **false elimination** (remove a
   candidate that survives in a real solution). Whether control-sufficiency holds
   *soundly across the certified fiber* is a measurement, not a guarantee.
2. The within-pass activation body is high-dimensional *by capacity* (`d=128` over
   tokens × layers × `L=16`). Whether the learned reasoner *uses* that capacity
   (a genuinely high-dim body) or **compresses to what the elimination needs** (the
   marginal, control-trained signature seen on Mesa) is a measurement.

C1 measured this separation **statistically** (`R²(R | Φ_K) = 0.99` — the
high-mode coupling is ~a function of the low shadow though the state isn't). LDT
could give the analog **exactly**: the fiber is *exactly* `γ(a)`, so a clean read
here would be the first **exact decision-observable / state-unobservable** witness
on a *computational* substrate — the computational analog of the exact Aharonov–Bohm
topological witness. The gate on that prize is the body-dimensionality leg.

## 1. The decisive structure (B2 twin-state, on the learned model)

Objects, at each visited lattice state `a` along the model's solving trajectories
(and at constructed under-constrained lattices):

| object | definition |
| --- | --- |
| **shadow** `Φ` | the candidate-set lattice `a` (729-dim multi-hot for 9×9; the architecturally-given control shadow) |
| **body** `h(a)` | the within-pass activation tensor (all layers × `L` iterations × cell tokens) — the only high-dim object |
| **decision** | the model's per-cell elimination set + the conflict/backtrack flag (what it *outputs*) |
| **hidden truth** | the concrete solution grid(s) `g*` consistent with `a`, i.e. `γ(a) ∩ solutions` |

**The certified twin (C1 adjudicator port).** A twin pair is two members
`g*₁ ≠ g*₂` of the same fiber `γ(a) ∩ solutions` at a **shared intermediate
lattice `a`** (constructible whenever the fiber has ≥ 2 valid completions). The
model's input each pass is `a` alone (clues are baked into the initial lattice), so
**both twins receive an identical forward pass** — the decision and body are shared
while the truth differs. This certifies, *by construction*:

- **state-insufficiency** — `a` (and therefore `h(a)`, which depends only on `a`)
  does **not** reconstruct `g*`; the fiber is genuinely non-trivial. *Certified,
  not estimated.*

What remains **measured on the learned model** (the non-vacuous legs):

- **control-sufficiency WITH soundness** — does the learned elimination at `a` keep
  the **whole** fiber (every `g* ∈ γ(a) ∩ solutions` survives)? A learned false
  elimination = a soundness break. The decision must also be **non-trivial** (real
  deductive progress, not no-ops).
- **body resistance** — is `h(a)` genuinely high-dimensional (information-basis
  `d_dec`) and **wider than the decision needs** (`k_control(elimination) ≪
  d_dec`), or does it collapse to the decision (marginal)?

**Decision-observable but state-unobservable** is then exact here: the decision is
determinate and (if sound) correct from `a`, while the solution is provably not a
function of `a`/`h(a)`.

## 2. The minimum cell — Sudoku-Extreme on the reimplemented LDT

- **Substrate (frozen):** the 800K LDT (`d=128`, 4 layers, `L=16`, learned CLS
  conflict head) reimplemented and trained on public **Sudoku-Extreme** (the
  paper's headline cell). **Build-gate (entry condition): reproduce 100%
  Sudoku-Extreme accuracy** before any B2 number is interpreted (else F4).
- **Fiber sampling (spectrum-blind w.r.t. the body):** collect intermediate
  lattices `a` from the model's own solving trajectories on held-out puzzles, plus
  a constructed set of deliberately under-constrained lattices, **selected only by
  fiber size** (`|γ(a) ∩ solutions| ≥ 2`, away from singletons), never by anything
  read off the body. For each, enumerate the fiber (exact for Sudoku) → the twins.
- **What is logged:** for each `a` — the fiber, the model's elimination + conflict
  output, and the full within-pass activations (the reimplementation gives total
  instrumentation).

## 3. Measures (with verify-before-file guards + the masked-variance correction)

Computed over the sampled lattices:

1. **Fiber non-triviality (state-insufficiency, certified).** Distribution of
   `|γ(a) ∩ solutions|`; median and fraction with fiber ≥ 2. *Guard (F5):* drop
   near-singleton lattices — the fiber must be genuinely open or the separation is
   the trivial "not solved yet."
2. **Fiber-soundness (control-sufficiency, learned).** Across all twins, the rate
   at which the model's elimination removes a candidate present in *some* fiber
   member (a false elimination = soundness break). *Guard:* count only eliminations
   the model actually commits (above its own threshold); report on-trajectory and
   on constructed lattices separately.
3. **Decision non-triviality.** Fraction of lattices where the model makes a
   *correct, progress-making* elimination (so "sound" is not achieved by doing
   nothing).
4. **Body dimensionality — information-basis, NOT variance PR.** chatv2 Amendment 1
   proved variance PR is *masked* by outlier residual directions on LLM-like
   bodies. So:
   - `d_dec(body)` — effective rank (singular-value participation ratio) of the
     stacked per-decision linear-readout directions `W = [w₁ … w_m]` decoding the
     `m` committed cell-eliminations from `h(a)`. Faithful high-dim reasoning →
     `d_dec` large; collapsed → `d_dec` small.
   - `k_control` — smallest body subspace that reads the elimination decision at
     ≈ full accuracy (accuracy-vs-`k` saturation on information-basis directions).
   - cross-decode guard: confirm `g*` is **not** decodable from `h(a)` beyond the
     fiber-uncertainty floor (it must not be — `h` depends only on `a`); a positive
     here would flag a leak in the activation logging, not a result.

## 4. Pre-registered failure modes (named before the run)

- **F1 — Marginal body (the expected null given the portfolio).** Fiber is
  non-trivial and the model is sound, but `d_dec` is small / `k_control ≈ d_dec` —
  the learned reasoner compresses to the decision. *Disposition:* a clean,
  honest **CERTIFIED-MARGINAL-BODY** result — regime-2 is certified at the
  abstraction level but the *neural body* is marginal. This **resolves the
  confound** every prior substrate carried (it isolates "high-dim body" from
  "control-sufficient shadow"); it is **not** re-skinned as sharp.
- **F2 — Vacuity (reduces to Target A).** The "separation" is just the definitional
  α/γ folklore with nothing added by the learned model. *Guarded by:* every leg
  except the certified one is measured on the **trained network** (false-elim rate,
  `d_dec`), and the fiber-soundness leg can genuinely fail (F3).
- **F3 — Soundness break (a real learned-model finding).** Non-negligible
  false-elimination rate → the learned model violates control-sufficiency-with-
  soundness; the paper's "empirical soundness" does not hold on the certified
  fiber. *Disposition:* report as an **UNSOUND** verdict (a substantive negative
  about the learned model), not a lane failure.
- **F4 — Substrate fidelity / access.** The reimplementation does not reach 100%
  Sudoku-Extreme. *Disposition:* fix or re-register; **no B2 number is read off a
  model that fails the build-gate.** Not a result.
- **F5 — Trivial fiber.** The sampled lattices are near-singletons, so the
  state-insufficiency is the trivial "puzzle not finished." *Guarded by* the
  fiber-size floor in §3.1 and constructed under-constrained lattices.

## 5. Pre-registered verdict (frozen thresholds; deferred execution)

Per the measured population (thresholds set from the portfolio priors, **not**
tuned to any LDT output, which does not exist yet):

- **`certified_state_insufficient`** iff median `|γ(a) ∩ solutions| ≥ 2` **and**
  ≥ 50% of sampled lattices have fiber ≥ 2 (§3.1). *(Expected to pass — this is the
  certified leg.)*
- **`sound`** iff false-elimination rate `≤ 0.1%` of committed eliminations (§3.2)
  **and** decision-non-triviality `≥ 0.5` (§3.3); else **`UNSOUND`** (F3).
- **`body_high_dim`** iff `d_dec(body) ≥ 16` (≫ the marginals' ~2; ≈ the chatv2
  scale) **and** `k_control / d_dec ≤ 0.25` (the body is wider than the decision).

**Headline verdicts:**

| verdict | condition |
| --- | --- |
| **`CERTIFIED_SHARP`** | `certified_state_insufficient` ∧ `sound` ∧ `body_high_dim` — the regime-2 split is certified *and* the learned neural body is genuinely high-dim and resists. The lane's bet wins; the first exact computational decision-vs-reconstruction witness. |
| **`CERTIFIED_MARGINAL_BODY`** | `certified_state_insufficient` ∧ `sound` ∧ ¬`body_high_dim` — regime-2 certified at the abstraction; the learned reasoner implements it with a low-dim body (the expected null, F1). Honest, confound-resolving, **not** sharp. |
| **`UNSOUND`** | ¬`sound` — the learned conflict head fails across the certified fiber (F3); a substantive negative about empirical soundness. |
| **`VACUOUS` / `INVALID`** | F2 / F4 / F5 fire. Re-register; no result. |

Thresholds are frozen here and **not** retuned after seeing the spectrum (the
anti-p-hack rule the portfolio enforces). `d_dec ≥ 16` and `ρ ≤ 0.25` are the
load-bearing numbers; the certified-state-insufficiency leg is expected to pass and
is *not* the discriminator — the **body** leg is.

## 6. Design decisions — status

- **D1 — substrate. SIGNED OFF:** Sudoku-Extreme on the reimplemented 800K LDT
  (the paper's headline cell; exact fibers; cheap to train).
- **D2 — access. RESOLVED (deferred):** reimplement + train ourselves (full
  instrumentation); execution gated on the build-gate (reproduce 100%). Snowflake /
  Maze are later-phase generalizations, not Phase 0.
- **D3 — body grain. RESOLVED:** log activations at every layer × iteration ×
  token; compute the fingerprint at each and report the grain with max `d_dec` as
  the body, plus the depth/iteration profile.
- **D4 — fiber construction. RESOLVED:** on-trajectory intermediate lattices +
  constructed under-constrained lattices, selected by fiber size only (body-blind).

## 7. Cost / build / reuse

- **Compute (when executed):** train one LDT (paper: ~minutes on a B200) + the B2
  analysis (fiber enumeration is exact and cheap for Sudoku; the body fingerprint
  is a handful of decodes/SVDs). Dominated by the reimplementation effort, not the
  measurement.
- **Reuse (do not rebuild):** the C1 **twin-state adjudicator** + the
  decision-observable/state-unobservable measure (`scripts/pde_c1_kolmogorov_cell.py`);
  the chatv2 **information-basis fingerprint** (`d_dec`, cross-decode;
  `scripts/chatv2_phase0_bodyresist.py`). New code is the **LDT reimplementation +
  trainer** and the fiber sampler.
- **Reserved implementation names (deferred):**
  - LDT model + trainer: `scripts/lattice_ldt_model.py`
  - Phase-0 B2 runner: `scripts/lattice_phase0_twinstate.py`
  - npm: `lattice:phase0:twinstate`
  - results: `results/lattice/phase0-twinstate/`
  - build-gate receipt: `results/lattice/build-gate-sudoku-extreme/` (must show
    100% before the B2 runner is admitted).

## 8. What Phase 0 explicitly does NOT do

- It does **not** claim the definitional α/γ separation (Target A) as a result.
- It does **not** prove or refute the paper's soundness theorem — it **measures**
  empirical soundness of *one reimplemented model* across a certified fiber.
- It does **not** make a capability / SOTA claim about the LDT or about Sundog, and
  uses no public-evaluation surface.
- It returns one of `CERTIFIED_SHARP` / `CERTIFIED_MARGINAL_BODY` / `UNSOUND` /
  `VACUOUS|INVALID` + the measured legs. **No public surface; no promotion**;
  external/sanity gate required before any promotion, as for every lane.

## 9. Cross-references

- [`LANE_CHARTER.md`](LANE_CHARTER.md) — the reframe (α/γ = projection/fiber), the
  verify-catch (lattice = inter-step state → body relocated), the two layers.
- [`../chatv2/PHASE0_MINIMUM_FALSIFIABLE.md`](../chatv2/PHASE0_MINIMUM_FALSIFIABLE.md)
  — the sibling fingerprint + the masked-variance → information-basis lesson (§4.4).
- [`../proof/PDE_C1_PROPOSITION.md`](../proof/PDE_C1_PROPOSITION.md) — the
  twin-state adjudicator + the `R²(R|Φ_K)=0.99` statistical separation B2 ports to
  an *exact* computational one.
- [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) — the marginal
  control-substrate column and the body-resistance axis (§6.3) this cell tests with
  a *certified* state-insufficiency.
