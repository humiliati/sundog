# NSE Post-AT Hypotheses Slate

> 2026-07-06. Fresh internal hypotheses slate after
> `NSE_ATTRACTOR_TAIL_SYNTHESIS.md` and
> `AT5_SYMBOLIC_TRAJECTORY_WINDOW_SCOPE.md`. This is not an external-review
> packet, not a public surface, not a promotion request, and not authorization
> for any long empirical run. It starts a new board while treating the AT lane
> as closed evidence, not live queue.

## 0. Starting State

The live positive is still C1:

- Reading-2 regime-2 is complete at two Grashof regimes,
  `(k_f=2, G=200)` and `(k_f=2, G=300)`, under the portable objective:
  state-insufficient on sampled SRB support and control-sufficient on
  `Phi_K`-fibers.
- The positive remains finite-Galerkin, sampled-support, numerical, and
  proxy-relative. It is not an infinite-dimensional NSE theorem.
- The open C1 gaps are now narrow and named: proxy faithfulness, resolution /
  discretization stability, broader regime geometry, and external mathematical
  review if promotion is ever sought.

The AT lane is closed context:

- Maintained-ledger form: `AT3_LEDGER_SPLIT_CONFIRMED`, with
  `K_dec = 1 < K_sync = 2`, relay-typed.
- Crossover form: `AT4_SURFACE_SUFFICIENT` at G=200 and final
  `AT4B_UNPOWERED_INPUT` at G=300. No surface-blocked natural label was
  established.
- AT-5: formal core already landed as the symbolic trajectory-window shadow;
  docs-only interpretation is sufficient unless vocabulary inside
  `sundogcert` is specifically desired.

## 1. Rules This Slate Inherits

1. No public, promotional, or external-review work is implied by this slate.
2. No AT-4/AT-4b variant reopens on the same substrate/objective family.
3. Every new empirical entry needs an input/admission gate before any
   adjudicator is run.
4. Every decision functional with a quantile threshold gets atom-clearance,
   balance, stationarity, and leakage checks before interpretation.
5. Surface comparators remain first-class. A positive must say what matched
   surface statistic, persistence statistic, or random floor it beat.
6. Long runs stay owner-run; this document can stage specs and commands, but
   it does not spend compute by itself.

## 2. Candidate Ranking

| rank | entry | kind | recommendation |
| --- | --- | --- | --- |
| A1 | H1: C1 proxy-faithfulness | finite-proxy hardening | board first |
| A2 | H2: C1 resolution stability | numerical robustness | board after H1 or in parallel if cheap |
| B1 | H3: C1 forcing-axis generality | regime geometry | park until H1/H2 are scoped |
| B2 | H4: stationarity gate as reusable doctrine | synthesis/spec | write only if the next empirical spec needs it |
| C1 | H5: AT symbolic closeout | docs/formal bookkeeping | already effectively satisfied |

## 2.1 Why H1/H2 Are the Sharp Pair

The next useful question is not "can we find another positive?" It is "what
does the existing positive really mean?"

H1 and H2 hit the two load-bearing ambiguities left after
`PDE-C1-RG-POS`:

- **H1 asks whether the action is faithful to a control objective.** The
  objective-overlap discriminator already showed that `Phi_K` predicts many
  physical observables and that control-sufficiency does not simply track
  objective predictability. That was valuable, but it still used threshold
  labels. H1 should instead test a paired-action value selector: damp because
  damping improves the registered future cost, not because a no-op rollout
  crosses an absolute or quantile threshold.
- **H2 asks whether the strongest current package survives the known
  resolution lift.** The old robustness wave already demonstrated N-refinement
  at G=200 for the v5-style witness (`lock_v5_n48`, grid 32 -> 48). H2 is not
  generic "try a finer grid"; it is the update: lift the portable-objective
  two-regime witness, or the H1 selector if H1 lands, through the same
  refinement discipline.

Why these before H3: another forcing or Grashof point would multiply the same
kind of evidence while leaving the two sharp objections intact. If H1 fails,
the witness remains a proxy-relative fact. If H2 fails, the witness remains a
finite-resolution fact. H3 is only worth the spend once those meanings are
settled.

## 3. H1 - C1 Proxy-Faithfulness

**Hypothesis.** The two-regime C1 witness is not merely an artifact of the
current low-band-energy safety proxy `pi_hat`; the same regime-2 structure is
visible under a registered decision selector tied more directly to action
value.

**Why this is first.** Regime-generality is no longer cell-local, but the
proxy gap remains the cleanest objection to the claim ladder. Closing it
strengthens the existing witness without broadening scope or changing the
substrate.

**Sharper v0: paired-action value selector.** For each sampled state `u`,
clone the state and evaluate the two registered actions on the same horizon:

```text
J_0(u) = future cost under no_op
J_1(u) = future cost under damp_low_band
Delta_J(u) = J_0(u) - J_1(u)
```

The recommended default is to inherit the v7 horizon (`tau = 500` steps,
5.0 time units) and the same cell pair `(G=200, G=300)`. The first H1 spec
must freeze the cost before any read. The natural first cost is the portable
lookahead-max observable already used by C1, but applied as an action-value
difference rather than as a no-op threshold. If a more policy-native cost is
chosen, that is fine only if it is frozen in the H1 spec before labels exist.

Define the selector on a calibration block:

```text
a_J(u) = damp_low_band iff Delta_J(u) >= m
```

with one margin `m` calibrated to target damp fraction 0.30 on calibration.
Held-out adjudication must then land in `[0.20, 0.40]`; otherwise the selector
is unpowered, not rescued. Report overlap with `pi_hat`, but do not use
overlap to tune `m`.

**Admission gate.**

- Define the selector before seeing fiber-adjudication numbers.
- Show the selector is non-vacuous on both G=200 and G=300 cells:
  damp in `[0.20, 0.40]`, no threshold atom, and stable calibration on held-out
  blocks.
- Show paired-action liveness: `Delta_J` has spread above numerical tolerance
  and the two action rollouts do not collapse to the same cost.
- Freeze the comparator set: prior `pi_hat`, matched persistence/surface
  statistics available from the cell, random-label floor, and the
  `PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md` objective-slate rows.

**Decision gate.**

- `NSE-H1-JSELECTOR-POS`: selector agrees with the existing C1 regime-2 story
  on both cells: control-sufficient on `Phi_K`-fibers while twin-state
  state-insufficiency remains certified at the same `epsilon_K`.
- `NSE-H1-COLLAPSED-PROXY`: selector is powered and positive but is nearly the
  same label as `pi_hat` (pre-register the overlap ceiling, e.g. `>= 0.95`);
  useful sanity check, weak new evidence.
- `NSE-H1-PROXY-ONLY`: the existing `pi_hat` witness does not transfer to the
  paired-action selector.
- `NSE-H1-UNPOWERED`: selector cannot form a balanced, atom-clear, stationary,
  action-live decision on one or both cells.

**Claim boundary.** A positive would strengthen proxy faithfulness only. It
would not promote C1, prove NSE, or discharge resolution stability.

## 4. H2 - C1 Resolution Stability

**Hypothesis.** The strongest current C1 package survives the known
N-refinement discipline rather than depending on the current 32x32 Galerkin
proxy.

**Why this is not generic.** The old robustness wave already passed the first
N-refinement at G=200 for the v5-style witness: `lock_v5_n48`, grid 32 -> 48,
`n_modes` 16 -> 24, same `K=3` signature, nearly identical clause numbers. H2
should not rerun that. H2 should lift the **current** result: the v7 portable
objective, and then the H1 selector if H1 lands.

**Sharper v0: v7 N-refinement regression.**

1. Add or stage a `lock_v7_g200_n48` preset mirroring `lock_v5_n48`:
   grid 48, `n_modes=24`, same `G=200`, `k_f=2`, `K=3`, `dt=0.01`, v7
   held-out calibration/adjudication split, and portable objective.
2. Run G=200 first as the regression cell. It must reproduce the v7 positive
   within the pre-registered tolerances before any G=300 refinement is
   interpreted.
3. Only if G=200 passes, stage `lock_v7_g300_n48`. G=300 may hit the known
   intermittency/runtime wall; that is `NUMERIC-WALL`, not a rescue prompt.
4. If H1 lands before H2 runs, H2 inherits the H1 selector; otherwise H2 uses
   the v7 portable selector.

**Admission gate.**

- Pre-register the resolution perturbation exactly: grid 32 -> 48 and
  `n_modes` 16 -> 24, with no dt/objective/K retune after seeing a read.
- Re-run only the minimal pair needed for comparability: objective formation,
  kNN control-sufficiency, and twin-state state-insufficiency at matched
  `epsilon_K`.
- Require solver diagnostics and objective portability to pass before any
  resistance interpretation.

**Decision gate.**

- `NSE-H2-V7-N48-STABLE`: G=200 v7 portable witness survives the registered
  N-refinement.
- `NSE-H2-TWO-REGIME-N48-STABLE`: both G=200 and G=300 survive the registered
  N-refinement.
- `NSE-H2-RES-SENSITIVE`: one half breaks while the objective remains powered,
  diagnostics pass, and the selector is action-live.
- `NSE-H2-NUMERIC-WALL`: diagnostics or runtime prevent a fair read.

**Claim boundary.** A positive is still numerical, but it materially raises the
finite-proxy claim from "two-regime witness" to "two-regime,
current-selector, N-refinement-stable witness."

## 5. H3 - C1 Forcing-Axis Generality

**Hypothesis.** The regime-2 witness is not only a Grashof-axis phenomenon at
fixed `k_f=2`; it persists under at least one registered forcing-geometry move.

**Why this is parked.** It is the natural generality question, but it expands
the cell geometry and risks re-learning objective formation from scratch. H1
and H2 buy more claim-strength per unit of confusion.

**Admission gate.**

- Pick one forcing-axis move before running: e.g. new `k_f`, forcing shape, or
  objective family. Do not combine axes in v0.
- Require the portable objective to form without rescue:
  damp in `[0.20, 0.80]`, stationarity across held-out blocks, and no
  threshold atom.
- If objective formation fails, stop at input. No adjudicator read.

**Decision gate.**

- `NSE-H3-FORCING-GENERAL`: full regime-2 witness forms at the new forcing
  cell.
- `NSE-H3-GRASHOF-LOCAL`: the witness fails with a powered and diagnosed
  objective.
- `NSE-H3-INPUT-UNPOWERED`: the cell cannot form the registered objective.

**Claim boundary.** A positive would broaden regime geometry by one axis only.
It would not claim substrate-generality, universality, or theorem status.

## 6. H4 - Stationarity Gate as Reusable Doctrine

**Hypothesis.** AT-4b's real lesson is portable: before any future
surface-crossover experiment, stationarity of the decision process must be an
input gate, not a post-hoc explanation.

**Deliverable.** A short reusable spec section or checklist that any future
natural-label crossover attempt imports verbatim:

- blocked split with guard gaps at least `max(W, tau)`;
- blockwise damp table before probes;
- input stop if held-out damp collapses or drifts outside `[0.20, 0.80]`;
- no surface/crossover read after input failure.

**Verdict.**

- `NSE-H4-STATIONARITY-GATE-LANDED`: checklist written and cross-linked.
- No empirical positive or negative is available from H4 by itself.

**Claim boundary.** H4 is process hygiene. It is not a new AT run and not a
route around `AT4B_UNPOWERED_INPUT`.

## 7. H5 - AT Symbolic Closeout

**Hypothesis.** No further empirical AT work is needed for the symbolic
trajectory-window pole; the docs-only AT-5 tier is enough unless the owner
explicitly wants vocabulary inside Lean.

**Current status.** `AT5_SYMBOLIC_TRAJECTORY_WINDOW_SCOPE.md` already supplies
the interpretation and records `AT5_FORMAL_CORE_ALREADY_LANDED` as the
docs-tier outcome.

**Optional action.** Add a thin `TrajectoryWindow.lean` alias module over
`SurfaceBagGraded` only if the vocabulary is worth carrying in `sundogcert`.
Kill immediately if it needs new combinatorics or a broader axiom profile.

**Claim boundary.** Symbolic only. No Kolmogorov-flow, attractor, nudging, or
empirical-ledger theorem.

## 8. Recommended Next Cut

Board H1 first as a **scope/spec**, not a run: freeze the paired-action
selector and decide whether its cost is the inherited portable lookahead-max
or a more policy-native cost. H2 is the next hardening move, but its sharper
form is now clear: v7/H1 N-refinement using the already-proven `lock_v5_n48`
pattern. H3 is the first true expansion move and should wait until H1/H2 say
what the current witness means.

Receipt base: `NSE_ATTRACTOR_TAIL_SYNTHESIS.md`,
`NSE_AT_SYNTHESIS_AT5_SCOPE.md`, `AT5_SYMBOLIC_TRAJECTORY_WINDOW_SCOPE.md`,
`docs/proof/PDE_C1_ROBUSTNESS_WAVE.md`,
`docs/proof/PDE_C1_REGIME_GENERALITY_v0.md`,
`docs/proof/PDE_DETERMINING_MODES_POSTULATE1.md`,
`docs/proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md`, and the C1
paired-fiber result manifests under
`results/proof/c1-paired-fiber-g{200,300}/`.
