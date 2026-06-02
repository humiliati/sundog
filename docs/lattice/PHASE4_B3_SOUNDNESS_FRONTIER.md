# Lattice-Deduction Phase 4 — B3 Learned-Soundness Frontier (companion design memo)

> 2026-06-02. **Companion design memo; execution gated** downstream of the Phase-1
> build-gate and the Phase-2 B2 result (see [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md)
> §Phase 4). B3 is the lane's most novel probe: it operationalizes the Sundog
> **control-sufficiency-vs-state-fidelity** axis that the LDT's loss *already
> encodes* — false-eliminations vs missed-conflicts — and measures where the
> learned model sits on it. Paper-design only; no model, no receipt, no public
> surface.

## 0. The observation B3 is built on

The LDT loss weights a candidate-elimination objective against a learned conflict
head (`λ_cls = 0.1`; litpass focal anchor). That weighting trades two errors:

- a **false elimination** = removing a candidate that survives in a real solution =
  the control-shadow becoming **wrongly state-restrictive** (a *soundness* break,
  the B2 leg);
- a **missed conflict / over-abstention** = failing to detect `⊥` or refusing to
  commit = the control-shadow failing to **act** (a *progress* failure).

That is the Sundog control-vs-state tradeoff, written as a loss by authors not
thinking about Sundog. B3 measures the frontier the learned model actually realizes:

> How hard can the decision be pushed (progress / decisiveness) before the
> control-shadow's **soundness over the certified fiber breaks**?

This is the "empirical soundness" caveat made into a measured Pareto frontier — the
B2 separation (decision-observable, state-unobservable) *under pressure*.

## 1. The two frontier axes

Measured over the certified-fiber population (B2 sampler) + on-trajectory states:

| axis | metric | direction |
| --- | --- | --- |
| **soundness** | false-elimination rate = fraction of committed eliminations that remove a candidate present in some `g* ∈ γ(a) ∩ solutions` | lower = sounder |
| **progress / decisiveness** | committed-elimination rate **and** solve rate; complement = abstention/backtrack/no-op rate | higher = more decisive |

A point on the frontier = `(soundness, progress)` at one operating configuration.

## 2. Knobs — inference-time primary, retraining gated

**Primary sweep (cheap; single build-gated model, no retraining):**

1. **conflict threshold `θ_CLS`** (paper default 0.6) — lower → more backtracking /
   abstention (safer, less progress); higher → more decisive (riskier).
2. **elimination threshold** (sigmoid confidence to commit a candidate removal) —
   higher → fewer, more-confident eliminations (safer); lower → more eliminations
   (riskier).
3. **lattice population** — on-trajectory states vs constructed under-constrained
   fibers, swept as a population factor (the frontier must hold on genuinely-open
   fibers, not just near-singletons).

**Admitted-only extension (expensive; gated):**

4. **loss-weight `λ_cls` analogue** — requires **retraining** N models at swept `λ`.
   This is *not* in the primary B3 frontier (a paper-design memo must not assume a
   retraining budget). It is admitted only if (a) the build-gate path is open and
   (b) an explicit retraining budget + seed-control plan is signed off; otherwise
   the frontier is the inference-time `(θ_CLS, elim-θ)` sweep on the single model.
   **Default for this memo: knob 4 OUT; knobs 1–3 IN.**

## 3. The frontier read (branches)

Inheriting the roadmap §Phase 4 branch names, adjudicated on the swept frontier:

| branch | condition | Sundog reading |
| --- | --- | --- |
| `frontier_stable_sound` | soundness stays below the false-elim threshold across a **useful** progress range (progress well above the abstention floor) | control-sufficiency-with-soundness is **robust** — the model is decisive *and* sound. The pro-regime-2 read. |
| `frontier_tradeoff_sharp` | a real Pareto edge: progress beyond a point can be bought **only** by unsoundness | the control-shadow's soundness has a **measured price** — the Sundog control-vs-state axis made visible. Substantive, lane-relevant. |
| `frontier_abstention_marginal` | the model is "sound" mostly by **doing little** (low progress, high abstention across the sound region) | soundness is **vacuous** (achieved by no-ops). The key guard against a fake-positive soundness read. |
| `frontier_sampler_invalid` | the constructed lattices do not represent the intended open-fiber class | methods failure; repair the sampler before reading the frontier. |

## 4. The abstention-vacuity guard (the load-bearing one)

The dominant way B3 fakes a positive is **soundness by inaction**: a model that
eliminates almost nothing is trivially sound. So `frontier_stable_sound` requires a
**frozen progress floor** — soundness must hold at a progress level materially above
the all-abstain baseline (e.g. ≥ a frozen fraction of the build-gated model's
default-threshold elimination rate, AND a non-trivial solve rate). Reporting the
**area under the sound-and-decisive region**, not a single point, is the frozen
summary; a model that is sound only at near-zero progress reads
`frontier_abstention_marginal`, never `stable_sound`.

## 5. Relationship to B2 / B1

- **B2** is one point on this frontier (the build-gated model at its default
  thresholds). B3 asks whether that point is a knife-edge or a plateau.
- A `frontier_tradeoff_sharp` with B2 `CERTIFIED_SHARP` is the strongest combined
  read: a certified, exact, sound separation **with a measured soundness price** —
  the computational analogue of "the C1 shadow is control-sufficient up to a
  measure-δ boundary," now with the boundary *swept*.
- B3 does **not** rescue an `UNSOUND` B2: if the model is unsound at its default
  operating point, B3 characterizes the frontier of that failure but earns no
  Sundog-positive.

## 6. Pre-registered failure modes

- **B3-F1 — abstention-marginal (key vacuity).** Soundness only at near-zero
  progress. Guarded by the §4 progress floor.
- **B3-F2 — no-frontier.** Soundness and progress do not trade across the swept
  thresholds (always sound or always unsound) — the thresholds are not the operative
  knob; the frontier is uninformative (consider the gated `λ` extension, with budget
  sign-off).
- **B3-F3 — sampler-invalid.** Constructed fibers unrepresentative of genuinely-open
  states (→ `frontier_sampler_invalid`).
- **B3-F4 — retraining confound (only if knob 4 admitted).** Seed/training variance
  across the `λ` sweep masquerades as a frontier. Guarded by seed-control + a
  fixed-seed fixed-data retraining protocol; out of scope under the default
  (inference-only) plan.

## 7. Frozen-where-it-matters / reserved names

Frozen before any run: the `(θ_CLS, elim-θ)` sweep grid; the false-elimination /
progress / abstention / solve-rate metrics; the **progress floor** (§4); the
Pareto-region (area-under-sound-decisive) summary; the four branch conditions; the
knob-4 admission gate (retraining requires explicit budget + seed plan).

- runner: `scripts/lattice_phase4_soundness_frontier.py`
- npm: `lattice:phase4:soundness-frontier`
- results: `results/lattice/phase4-soundness-frontier/` (reserved in the roadmap)
- reuse: the B2 fiber sampler + false-elimination check; the build-gated LDT (no
  retraining under the default plan).

## 8. What B3 does NOT do

- It does **not** prove or refute the LDT paper's soundness theorem — it measures
  *one reimplemented model's* empirical frontier.
- It does **not** retrain under the default plan (knob 4 is gated, budget-signed).
- It does **not** claim the definitional separation, a capability/SOTA result, or
  any public-evaluation / Kaggle claim.
- An `UNSOUND` or `abstention_marginal` read is a **substantive negative about the
  learned model**, not a lane failure, and never a Sundog-positive.

## 9. Cross-references

- [`PHASE0_MINIMUM_FALSIFIABLE.md`](PHASE0_MINIMUM_FALSIFIABLE.md) — B2: the single
  operating point this frontier sweeps around; the false-elimination / fiber
  machinery reused.
- [`PHASE3_B1_INTERNAL_BODY_FINGERPRINT.md`](PHASE3_B1_INTERNAL_BODY_FINGERPRINT.md)
  — the body map; B3 is the decision-side complement (B1 = where the body is, B3 =
  how hard the decision can be pushed).
- [`LITPASS_MEMO.md`](LITPASS_MEMO.md) — the focal-paper `λ_cls` / conflict-head
  anchor and the Target-A fence (soundness is Cousot's; the *learned* frontier is
  the novel object).
- [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md) — §Phase 4 in the ladder.
