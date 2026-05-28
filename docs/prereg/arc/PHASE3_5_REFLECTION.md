# Phase 3.5 -- Deterministic Low-Capacity Learner Family Reflection

Roadmap: [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)

Phase 3 spec: [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)

Filed: **2026-05-28 (PT)**

Status: **REFLECTION ONLY -- NO EXECUTION OR VERDICT ADMISSION**. This
document names the three-receipt convergence under the Phase 3
deterministic-low-capacity-learner family as a methodological finding,
scopes what kind of learner Phase 3 would need to produce a non-trivial
sufficiency verdict, and surveys Phase 4 branch options. It does not
admit any new learner, change any binding receipt's verdict, or relax any
discipline rule. Public-language constraints from Pass D and the Phase
0-2 receipts remain in force.

## Why This Document

Phase 3 has filed three binding receipts under three distinct
deterministic low-capacity learner families:

| receipt | learner version | candidate construction |
| --- | --- | --- |
| first | `nn_output_transfer_v1` | nearest-input output-transfer (conditioning outputs only) |
| second | `nn_delta_transfer_v1` | + same-shape `delta_overlay` synthesis |
| third | `candidate_combinator_v1` | + per-pair bijective colormap fit, seven non-identity D4 variants, cross-pair cell unions (delta_overlay retained) |

All three receipts produced verdict **task hardness / decoder failure**
under the frozen Pass C verdict gates. The third learner is empirically
a strict superset of the first two on slot-1 and slot-2 metrics, and its
verdict was the same as the first two. The convergence is itself a
finding, distinct from any single receipt's verdict, and it shapes what
Phase 3 would need to do next to produce evidence about signature
sufficiency rather than evidence about the learner family.

The reflection here is preregistered to keep that distinction frozen:
the convergence is **not** a sufficiency-failure conclusion, **not** a
support-on-registered-subset conclusion, and **not** a public claim
about ARC. It is a learner-class observation, and the next admitted
work must follow from that observation, not from a re-interpretation of
the existing receipts.

## What The Three Receipts Showed

The numerically stable observations across all three LODO `k_ge_3`
receipts (105 instances, 102 active after Pass B trivial-task suppression):

| metric | nn_output_transfer_v1 | nn_delta_transfer_v1 | candidate_combinator_v1 |
| --- | ---: | ---: | ---: |
| `signature_only.rep_exact_slot1` | `0.000` | `0.010` | `0.000` |
| `signature_palette.grid_exact_any_slot` | `0.000` | `0.000` | `0.000` |
| `signature_palette.rep_exact_slot1` | `0.000` | `0.010` | `0.000` |
| `signature_palette.rep_sim_best` | `0.784` | `0.803` | `0.789` |
| `metadata_only.rep_exact_slot1` | `0.039` | `0.010` | `0.039` |
| `metadata_only.rep_sim_best` | `0.939` | `0.942` | `0.942` |
| `raw_grid_lowcap.grid_exact_any_slot` | `0.000` | `0.010` | `0.010` |
| `raw_grid_lowcap.rep_sim_best` | `0.912` | `0.920` | `0.922` |
| `metadata_only.coverage_failure_rate` | `0.951` | `0.941` | `0.941` |
| `signature_palette.coverage_failure_rate` | `1.000` | `0.990` | `0.990` |
| `raw_grid_lowcap.coverage_failure_rate` | `1.000` | `0.990` | `0.990` |
| `signature_only.coverage_failure_rate` | `1.000` | `0.990` | `0.990` |
| `metadata_only - signature_palette` gap on `rep_sim_best` | `0.155` | `0.139` | `0.153` |

(See the binding receipts under `results/arc/phase3-sufficiency/`,
`results/arc/phase3-sufficiency-nn_delta_transfer_v1/`, and
`results/arc/phase3-sufficiency-candidate_combinator_v1/`, hashed in the
respective verdict amendments in
[`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md).)

What is stable across the three receipts:

1. **Exact-rate floor.** Every learner-arm exact metric is below the
   Pass C decisive thresholds for "support". `grid_exact_rate_any_slot`
   never exceeds `0.010`. `rep_exact_rate_slot1` never exceeds `0.039`.
2. **Coverage floor.** Every grid-bearing arm reports coverage failure
   on at least `0.99` of LODO `k_ge_3` instances under
   `nn_delta_transfer_v1` and `candidate_combinator_v1`; under
   `nn_output_transfer_v1` coverage is exactly `1.0` for all arms except
   `metadata_only`. The held-out outputs are nearly always outside the
   learner's candidate pool, regardless of which deterministic primitive
   set we extend the pool with.
3. **The signature-vs-metadata `rep_sim_best` gap.** `metadata_only`
   leads `signature_palette` on `mean_output_rep_similarity_best` by
   `0.139` to `0.155` across the three receipts. This is the decisive
   metric in the Pass C ladder for every receipt, and it has the same
   sign and roughly the same magnitude every time.
4. **Verdict.** All three receipts trigger the Pass C task-hardness
   condition (every grid-bearing arm has `grid_exact_rate_any_slot <
   0.03`; every representation arm has `rep_exact_rate_slot1 < 0.05`)
   and the Pass C sufficiency-failure condition (signature_palette
   materially trails metadata_only on the first decisive metric). Task
   hardness preempts sufficiency failure under the same learner-class
   reasoning each time.

## What The Receipts Did Not Show

The three receipts are silent on the following questions:

1. **Is the signature representation sufficient for ARC-style
   abstraction?** The decisive metric in the verdict ladder
   (`rep_sim_best`) is computed inside a learner class whose candidate
   pool overwhelmingly does not contain the held-out output. The metric
   gap therefore reflects how the candidate-identity rules for each arm
   trade off against chance-collision probability, not how well each
   representation captures the underlying rule.
2. **Is `metadata_only` actually sufficient?** Its `rep_exact_slot1`
   hits are coincidence: two grids with the same shape, palette,
   density, and color histogram receive the same 28-dim identity
   vector. Three to five such collisions in 102 LODO `k_ge_3` instances
   does not establish that the metadata representation captures the
   rule, only that its identity rule is coarse enough to be chance-hit
   more often than the signature's.
3. **Would a more capable learner reveal a different result?** The
   three filed receipts cannot answer this. They have all been pulled
   from the same "rank conditioning-derived candidates by arm distance,
   emit at most two" family, with progressively richer deterministic
   synthesis primitives. None of them learns anything per-task in the
   gradient-descent sense; none of them can synthesise an output grid
   that is not a deterministic function of the conditioning data alone.

## Why A Fourth Deterministic-Low-Capacity Learner Would Likely
Converge Again

The next-learner reservation in PHASE3_SUFFICIENCY_SPEC.md named a third
candidate, `finite_program_selector_v1`, that was not filed. Under the
same deterministic-low-capacity discipline (no fitted parameters, no
optimizer, no cross-task weights, candidates emitted from frozen
primitives applied to conditioning data), program selection would have
the same structural property:

- the candidate pool is bounded by the number of frozen DSL programs
  times the number of source conditioning configurations;
- ARC tasks are explicitly designed so that the registered rule is novel
  per task, not a member of any frozen pre-listed DSL;
- the Phase 0 `oracle_copy_floor_lodo_rerun` rows in every receipt
  already report `0/115` exact for `dsl_lite_v0`, `dsl_lite_v1`,
  `dsl_lite_v2`, `tiny_learned_v0`, `identity_copy`, and `random_valid`
  -- the most natural seed DSL is already shown to be exhausted by the
  registered subset.

Either the program search would re-discover a primitive already in
`candidate_combinator_v1` (D4, colormap, cell union, delta-overlay,
output-copy) and add nothing, or it would need a DSL strictly richer
than Phase 0's set, which is the substantive design change. A program
selector inside the current discipline is therefore not expected to
escape the convergence.

The signature-vs-metadata gap on `rep_sim_best` is even more directly
attributable to the learner class. Every receipt's gap is driven by
`metadata_only`'s coarser candidate-identity rule producing systematically
larger best-similarity values than `signature_palette`'s finer identity
rule, regardless of what the underlying primitive set is. This gap is
not evidence about representation sufficiency; it is evidence about how
arm-distance ranking interacts with identity-rule coarseness inside this
learner class.

## What Kind Of Learner Would Produce A Non-Trivial Verdict

To break the convergence pattern, the next learner family would need at
least one of the following structural changes, all of which are outside
the current Pass C "deliberately small first learner" constraint:

1. **Per-task gradient-trained mapper.** A small MLP that fits an
   `input_rep -> output_rep` mapping on `k - 1` conditioning pairs and
   applies it to the query input. This learner can synthesize an output
   representation that is not a deterministic function of the
   conditioning outputs alone, and so its candidate pool is unbounded
   by the conditioning-data primitive set. Requires admitting fitted
   parameters, a seed slate, a training budget, and an early-stop rule.
2. **Program induction over an unfrozen DSL.** A search that learns the
   DSL primitives from the conditioning data instead of receiving them
   frozen, in the spirit of Dreamcoder-style approaches. Inside the
   "no pretrained model" rule this is feasible but would require a
   primitive-induction operator definition and a substantial Pass A
   extension.
3. **Iterative refinement learner.** Start from a deterministic
   candidate (e.g. `output_copy` or `delta_overlay`) and edit until the
   resulting input-output pair under the conditioning representation
   matches a metric computed against the conditioning pairs. Requires
   admitting an edit operator with a search budget.
4. **Representation-conditioned generative model.** A model that emits
   output candidates conditioned on the representation arm. Outside the
   "no pretrained model" rule unless the model is trained per task from
   scratch.

The first option is the smallest step inside the spirit of the existing
Pass C admission. Option 3 is the next-smallest, because the edit
operator can be a frozen primitive set and the search can be bounded.
Options 2 and 4 are larger lifts.

Any of these would require:

- a new Pass C-style admission amendment in PHASE3_SUFFICIENCY_SPEC.md
  pinning learner architecture, hyperparameters, seed slate, fitting
  budget, and (for stochastic learners) the reproducibility envelope;
- a new runner script set following the established naming pattern;
- a new receipt directory under `results/arc/`;
- the freeze-marker commit discipline currently in force.

## Phase 4 Branch Options

This reflection names four candidate next moves. Each is consistent
with the discipline already in force; the choice is the operator's.

### Branch A -- Admit A Stochastic Per-Task Mapper

Smallest step beyond the current learner family. A new Pass C-style
admission amendment files a `per_task_mlp_v1` or
`per_task_linear_v1` learner with:

- a fixed shallow architecture;
- a seed slate;
- a per-task training budget capped under the repo's ten-minute rule
  (or staged as an external command if it exceeds that);
- the same Pass A representation arms and Pass B instance handling.

The candidate pool would no longer be conditioning-derived. If a fitted
mapper produces materially higher exact rates than the cheap baselines
on LODO `k_ge_3`, Phase 3 finally has signal about representation
sufficiency rather than about candidate-pool structure. If it does not,
the convergence extends to the gradient-trained class and the
representation question survives more learner families.

### Branch B -- Pass C Threshold Revision

Across all three receipts, `shape_exact_rate_slot1` ranges over `0.51` to
`0.61` and `palette_exact_rate_slot1` over `0.43` to `0.48`, both well
above chance. The current Pass C verdict table treats these as
tie-breakers below the exact-rate tier; they never reach decisiveness
in practice. A Pass C amendment could operationalise the consistent
shape/palette recovery as its own admissible verdict tier
("structural support"), reflecting that representation arms do recover
non-trivial structural information about held-out outputs even when
they cannot reconstruct them. This is a threshold-only change; it does
not admit any new learner.

The risk is that "structural support" becomes a moving goalpost. The
amendment must therefore freeze the threshold band based on these three
receipts before any later receipt is generated, and explicitly tie the
verdict to the existing three receipts' numbers, not to any future
learner's.

### Branch C -- Pause Phase 3, Return To Gravity Ledger

Three task-hardness verdicts is itself a defensible stopping point on
the Phase 3 deterministic-low-capacity branch. The Sundog program
continues elsewhere; this reflection becomes the canonical
characterisation of what Phase 3 did and did not show. The Phase 3
receipts and PHASE3_SUFFICIENCY_SPEC.md remain binding; no new learner
work is admitted; the program returns to Gravity Ledger items outside
ARC.

The risk is that the Phase 3 lane is read by an outside reader as a
sufficiency-failure conclusion. The reflection above and the Pass D
public-language drafts already cover that risk in writing, but pausing
without filing Branch A or Branch B leaves the question hanging on the
lane's public face.

### Branch D -- Different Sufficiency Framing Entirely

The current Phase 3 framing scores `input_rep -> output_rep` mapping
under a fixed candidate-emission learner. An alternative framing could
score the representation arms directly on properties of the
representation itself (e.g., does the representation predict the
correct shape and palette of the held-out output without producing a
candidate grid?). The three receipts' consistent shape/palette recovery
above chance is the empirical motivation. Implementing this would
require a substantial Pass A or new "Phase 3 alt" spec.

This is the largest design change of the four branches and is the
least mechanically continuous with the existing receipts.

## Discipline Carries

Nothing in this document admits any of the following. The previous
binding constraints remain in force:

- no edits to Pass A arm schemas, Pass B instance handling, Pass C
  verdict ladder thresholds, or Pass D receipt schema without a new
  append-only amendment in `PHASE3_SUFFICIENCY_SPEC.md`;
- no public-evaluation grid inspection (Phase 6 only);
- no Kaggle notebook prep, Kaggle private or semi-private splits;
- no sufficiency, dimensionality, palette-dependent, or partial-support
  claim from any Phase 3 receipt;
- no description of any Phase 3 receipt as evidence on the
  public-evaluation split or as an ARC solve.

This reflection cites the three binding receipts by their filed
artifact hashes; it does not re-interpret them and does not propose any
re-run that would replace those receipts. Each branch above, if filed,
must produce its own receipt under a learner-version-suffixed path and
must not overwrite the existing receipts.

The public-language drafts in Pass D and the verdict amendments remain
the canonical phrasing for any external citation of Phase 3 results.

## Pre-Registered Stance

If the operator chooses Branch A: a new Pass C-style admission amendment
for a stochastic per-task learner is filed under PHASE3_SUFFICIENCY_SPEC.md,
following the same template as the `nn_delta_transfer_v1` and
`candidate_combinator_v1` admissions.

If the operator chooses Branch B: a new Pass C threshold-revision
amendment is filed defining the structural-support verdict tier,
explicitly bound to the three filed receipts and not to future
unfiled receipts.

If the operator chooses Branch C: no further amendment is filed; this
reflection becomes the canonical characterisation and the lane status
is "Phase 3 paused after deterministic-low-capacity convergence".

If the operator chooses Branch D: a new spec (likely
`PHASE3_ALT_FRAMING_SPEC.md` or similar) is filed alongside
PHASE3_SUFFICIENCY_SPEC.md, with its own Pass A--D structure.

This document does not pre-commit to any branch. The operator's
choice of branch becomes the next admitted Phase 3 work after this
reflection is committed.

## Post-Blackwell / Compact-7 Addendum

Filed: **2026-05-28 (PT)**.

Subsequent Phase 3 amendments used this reflection's branch menu to test
whether a stronger or narrower full-grid control could open a meaningful
signature-vs-full-grid comparison. Those later receipts do not replace the
three deterministic-low-capacity receipts above; they add a separate
full-grid-control calibration layer:

- V1 Blackwell runner: `signature_palette` and the matched full-grid control
  both floored at zero exact matches, so no sufficiency comparison was
  interpretable.
- V2 public-training raw-grid control: `full_grid_control_floor` under freeze
  marker `79C5B060`, with zero exact matches on both held-out lanes.
- Compact-7 diagnostic: `compact_full_grid_control_floor` under freeze marker
  `50EAEBBF`, with zero exact matches despite `shape_exact_slot1 = 1.000`.

The compact-7 residual audit names a qualitatively new failure mode:
**dominant-color mode collapse**. Predictions often match output shape and
background mass while omitting the target object palette/content; 13 of 13
held-out compact-7 predictions use at most two slot-1 colors, while targets use
3--9 colors.

After compact-7, the narrowed deterministic full-grid-control path is closed.
No further raw-grid capacity bump, extra seed, or additional task-distribution
narrowing in this learner family is admitted as a Phase 3 reopen path. The
remaining reopen paths are:

- **Branch A**: admit a stochastic per-task learner with a new Pass C-style
  amendment and receipt path.
- **Branch D**: admit a different framing, such as structured edit prediction
  or residual learning, with its own Pass A--D contract.

Branch C remains available as a pause state. Branch B is closed by the V2 and
compact-7 receipts.
