# Lattice-Deduction Research Lane — Charter

> Opened 2026-06-02. **Paper-design-only frozen lane** (execution deferred — see
> §Access). Tests whether the **Lattice Deduction Transformer** (LDT, arXiv
> 2605.08605v1) is the substrate where the Sundog **body-resistance regime-2**
> turns from *measured-marginal* to **architecturally certified** — because the
> LDT bakes the abstraction map (α/γ) that the Sundog framing borrows directly
> into its forward pass.

> **Naming note.** This `docs/lattice/` folder is a **research** lane and a
> sibling of [`../chatv2/`](../chatv2/LANE_CHARTER.md) (the LLM-body-resistance
> lane). It tests a *borrowed external substrate* (a published model), not a
> Sundog product surface. Nothing here is promoted, and nothing claims a result
> about the LDT paper's authors' system until the deferred execution runs under
> the named build-gate.

## The reframe (why this substrate, and why now)

The LDT forces its state, between every forward pass, through an **interpretable
abstract domain** — a per-cell candidate-set lattice — so that each pass is a
*sound abstract-interpretation deduction step*. Its α/γ Galois pair is, term for
term, the Sundog projection/fiber geometry:

| LDT object | Sundog object |
| --- | --- |
| `α` (abstraction): concrete grids → candidate-set lattice `a` | the projection `π: state → shadow` |
| `γ` (concretization): `γ(a) = {grids consistent with a}` | the fiber `π⁻¹` over a shadow value |
| `α` is many-to-one (`|γ(a) ∩ solutions| > 1` for non-singleton `a`) | **state-insufficiency** — the shadow does not pin the state |
| `ded_p(a) = α(γ(a) ∩ ‖p‖)` is a function of `a` (given puzzle `p`) | the decision/dynamics **factor through the shadow** = control-sufficiency |
| abstract-interpretation **soundness** (never eliminates a surviving candidate) | control-sufficiency **with a guarantee** |

So the LDT does not merely *admit* a regime-2 reading — it is **engineered** so
that a control-sufficient, state-insufficient shadow sits in the architecture. On
the three measured control substrates (NSE-C1, Mesa, Sabra) state-insufficiency
was *measured and came back marginal*; here it is **certified by construction**
(the fiber `γ(a)` is genuinely non-trivial for any intermediate lattice). That is
the one thing every prior substrate lacked.

### The verify-before-design catch (architectural, load-bearing)

The paper (verified 2026-06-02 from the HTML) states: **"the lattice itself is
the inter-step state — no separate hidden state between steps."** For 9×9 Sudoku
the lattice is 729-dim (81 cells × 9 digits); the model is `d_model=128`, 4 layers
unrolled `L=16` per pass, with a *learned* conflict head (CLS sigmoid, `λ_cls=0.1`,
`θ=0.6`).

This **relocates the body.** If the lattice is the *entire* Markov state between
passes, then `FVE(inter-step-state | lattice) ≡ 1` trivially — at that grain the
shadow *is* the state, the maximally **marginal** case, not regime-2. The only
high-dimensional "body" lives **inside a single forward pass** (the 128-dim
activation tensor over tokens × layers × iterations) and is then collapsed back to
the lattice. **The signature test must be run on the within-pass activations,
against the lattice as the architecturally-given shadow — not on the inter-step
state.** Missing this would have produced a vacuous `FVE ≈ 1` "marginal" non-result.

## Two layers — only one is worth claiming

- **Target A — definitional (acknowledged, NOT claimed).** The *ideal* α/γ
  operator gives regime-2 for free: sound abstract interpretation is
  control-sufficient over a state-insufficient abstraction. This is
  textbook/folklore (the same "mechanism is known — don't reprove it" posture as
  the C1 Mori–Zwanzig closure). The lane **names it and walks past it.** No credit
  is taken for the definitional separation.
- **Target B — the *learned* neural model (the novel, measurable content).** Three
  probes, each reusing portfolio machinery:
  - **B2 — twin-state decision-vs-reconstruction (the Phase-0 headline).** On
    *certified* same-lattice-different-solution twins, does the **learned** model
    keep its decision (i) a function of the lattice and (ii) **sound**, while
    (iii) neither the lattice nor the activations reconstruct the specific
    solution? = C1's twin-state adjudicator + "decision-observable but
    state-unobservable," on a real neural reasoner. The only probe that could
    yield a **certified** (not marginal) witness.
  - **B1 — internal-body fingerprint.** The three-number `eff_dim / k_control /
    FVE` fingerprint on the within-pass activations (the chatv2 sibling). **Uses
    the information-basis estimators, not the variance PR** — chatv2 Amendment 1
    proved variance PR is *masked* by outlier residual directions on LLM-like
    bodies. Honest prior: likely marginal (small model, hard bottleneck).
  - **B3 — the learned-soundness frontier.** The paper's loss already trades
    **false-eliminations (soundness breaks)** against **missed-conflicts
    (abstention failures)** via `λ_cls`. That tradeoff *is* the Sundog
    control-sufficiency-vs-state-fidelity axis, written as a loss by authors not
    thinking about it. The probe characterizes where the learned model sits on it.
    The "empirical soundness" caveat (a learned conflict head *can* miss) is the
    measurable boundary.

## The claim this lane makes falsifiable

> On the LDT substrate there is a **certified** state-insufficient,
> control-sufficient shadow (the candidate-set lattice), and the **learned** model
> realizes it **soundly** across the certified fiber while its within-pass body is
> genuinely high-dimensional and not reducible to the shadow — i.e. the regime-2
> split is *architecturally certified* on a real neural reasoner, not merely
> measured-marginal as on NSE-C1 / Mesa / Sabra.

The honest prior remains **marginal-leaning**: every measurable substrate so far
collapsed (NSE-C1 `FVE~0.99`, Mesa `net.7` PR~2, Sabra eff-rank~1.7, ARC Phase 4
PR≈11 below the bar), and `d_model=128` caps the achievable body dimensionality.
What makes LDT worth the lane anyway: the state-insufficiency leg is **certified**
(not measured-low), and the control-sufficient shadow is **architecturally given
with a soundness guarantee** — so a null cleanly *isolates* "does the reasoner
build a high-dim body" from "is there a control-sufficient shadow," which were
confounded on every prior substrate.

## Discipline (inherited from the portfolio)

- **Pre-registration before any verdict-bearing run.** Named failure modes up
  front; verify-before-claim. The vacuity guard is binding: every B-layer measure
  must read the **learned** model, never restate the definitional (Target A)
  separation.
- **No public surface; nothing promoted without the named external/sanity gate.**
- **Reuse, don't rebuild.** The C1 twin-state adjudicator
  (`scripts/pde_c1_kolmogorov_cell.py`) and the chatv2 information-basis
  fingerprint (`scripts/chatv2_phase0_bodyresist.py`, the `d_dec` / cross-decode
  measures) both exist and are validated. New code is the LDT itself.

## Access — paper-design-only, execution deferred

No checkpoint or code is released with the paper (verified 2026-06-02). The lane
is therefore **frozen as a pre-registration**, with execution gated on obtaining a
model. The chosen path (owner, 2026-06-02): **reimplement + train the 800K LDT**
(public Sudoku-Extreme data, ~minutes on a B200-class GPU), which also yields full
instrumentation (every lattice state + every activation logged). **Build-gate
(entry condition for any signature measurement):** the reimplementation must
**reproduce the paper's 100% Sudoku-Extreme accuracy** before a single B-layer
number is interpreted (guards against measuring a broken model — failure mode F4).

## Structure

- `LANE_CHARTER.md` — this file (mission + reframe + claim + discipline + access).
- `PHASE0_MINIMUM_FALSIFIABLE.md` — the frozen B2 twin-state cell: the smallest
  pre-registered experiment that could falsify the lane claim.
- Later phases (B1 fingerprint, B3 soundness-frontier) + sidecars accumulate here.

## Cross-references

- [`../chatv2/LANE_CHARTER.md`](../chatv2/LANE_CHARTER.md) — the sibling
  LLM-body-resistance lane (first non-flatly-marginal read; the masked-variance →
  information-basis lesson this lane inherits).
- [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) — the body-resistance
  axis (§6.3) and the marginal control-substrate column this lane tries to beat
  with a *certified* (not measured) state-insufficiency.
- [`../proof/PDE_C1_PROPOSITION.md`](../proof/PDE_C1_PROPOSITION.md) — the
  twin-state adjudicator + "decision-observable but state-unobservable" measure
  that B2 ports to a computational substrate.
