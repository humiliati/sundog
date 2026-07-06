# GEN-1 Object-DSL v0 — Perception/Grammar Freeze + Ceiling-Probe Protocol

Parent / boundary documents:

- [`../findcheck/GENERATOR_CLASS_SLATE.md`](../../findcheck/GENERATOR_CLASS_SLATE.md) (GEN-1, gate ladder)
- [`PHASE3_BRANCH_E3_LEARNED_RANKER_SPEC.md`](PHASE3_BRANCH_E3_LEARNED_RANKER_SPEC.md) (Amendment D — the
  generator-ceiling closure this class answers; the validation fingerprint file this probe's baseline reads)
- [`PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md`](PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md) (the frozen v2 bank)
- `../findcheck/FIND_CHECK_SUFFICIENCY_SLATE.md` (FC-4 RESULT — GEN-1 gate-0, the σ split, PASSED 2026-07-01)

Filed: **2026-07-01 (PT)**

Status: **V0 FREEZE + PROBE PROTOCOL; PROBE EXECUTION NOT ADMITTED (tooling not built).** This is NOT a
solver-branch spec and adjudicates NO capability branch. It freezes the GEN-1 perception vocabulary and rule
grammar and pre-registers the ceiling probe whose outcome decides whether a Branch-F-style binding spec may be
*written*. A later tooling freeze-marker amendment (runner, leak-check, smoke, timing, staged command) must be
filed before any probe run. ARC public-language constraints inherited in full (no solve / sufficiency /
public-eval / Kaggle claims; Phase 6 gates public-eval).

## Why this exists (boundary citation)

E3 Amendment D closed the deterministic program-search arc at its generator ceiling: on gated lanes the oracle
ceiling equals the already-solved set and **89% of instances admit no train-consistent program**; on validation
lanes the same picture holds (**135/155 = 87% zero-candidate rate** under the frozen v2 bank, from the E3
validation fingerprint file). The binding wall is generator-CLASS expressivity. GEN-1 bets that ARC-style rules
are **order-1 statements in an object filtration** — a bet whose gate-0 (FC-4: object-order-1 R² 1.000 vs best
bounded-pixel 0.535, controls clean) has passed. This document is gate-1's freeze; the probe is gate-2.

## §1 — Frozen perception vocabulary (v0)

**The kill condition this section answers:** "object segmentation is itself underdetermined
(connectivity/multicolor/background choices) and leaks per-task tuning." The v0 answer is structural: a FIXED
small set of **views** is enumerated as part of every candidate program (the program *declares* its view; the
CHECK verifies it against train pairs). View choice is searched-and-verified, never tuned post hoc.

**Views (exactly 3):**

```text
V1  cc4   — 4-connected components of same-colored nonzero cells
V2  cc8   — 8-connected components of same-colored nonzero cells
V3  blob4 — 4-connected components of nonzero cells (multicolor blobs)
```

Background is **fixed at color 0** in v0 (the ARC convention). Non-zero-background inference is a NAMED v0
limitation (a pre-authorized v1 extension, §4 — not tunable inside v0).

**Per-object attributes (frozen list):**

```text
area          cell count
color         cell color (blob4: dominant color; n_colors also recorded)
bbox          (r0, c0, h, w)
centroid      (mean row, mean col)
shape         translation-normalized cell set (offsets from bbox origin), hashed
shapeD4       D4-canonical shape (min hash over the 8 transforms)
touches_border  bool
n_holes       count of enclosed background regions within the object
```

**Relations (v0, used only inside selectors/anchors):** `same_color`, `same_shape`, `same_shapeD4`,
`bbox_contains`. Full spatial relations (adjacency graphs, alignment groups) are pre-named v1 material.

## §2 — Frozen rule grammar (v0)

A candidate program is one tuple; **holes (κ, δ, k, dims) are CEGIS-solved** from the conditioning train pairs
(constraints intersected across pairs; a program is ADMITTED only if it then reproduces EVERY conditioning
train output exactly — the unchanged FC-1 CHECK). Skeletons are enumerated; holes are solved, not enumerated.

```text
program  := (view V, canvas C, select S, transform T, others O)

canvas C := input_copy            # render onto the input
          | blank_like_input     # same dims, background
          | blank_solved_dims    # dims = hole, CEGIS from train outputs (constant dims, or constant
                                 #   ratio to input dims; else the skeleton is inadmissible)
          | selection_bbox_crop  # output = the selection's bbox crop (after transform)

select S := all | argmax(area) | argmin(area)
          | unique(color) | unique(shape) | unique(shapeD4) | unique(area)
          | modal(shape) | nonmodal(shape)
          | touches_border | not_touches_border
          | where(color = κ)                       # κ a CEGIS hole

transform T := identity | delete | recolor(κ) | recolor_to(color_of(argmax(area)))
          | move(δ)                                # δ a CEGIS hole, constant across pairs
          | gravity(dir ∈ {up,down,left,right})    # translate until touching border/another object
          | reflect(d ∈ D4) | scale(k ∈ {2,3})     # in place (bbox-anchored)
          | copy_move(δ)                           # keep original + translated copy

others O := keep | delete                          # the non-selected objects' fate
```

Composition depth = **1 object rule** (v0), plus an optional global post ∈ `{identity, crop_nonzero_bbox}`.
Skeleton count ≈ 3·4·12·~12·2·2 ≈ **~14k**, under a frozen per-instance admission budget of **20,000**
(comparable to v2's). Deterministic enumeration order (view, canvas, select, transform, others, post);
budget exhaustion is logged, never silently dropped.

**Frozen with this file's hash:** §1 + §2 in their entirety. No attribute, view, selector, transform, budget,
or ordering change after any probe target is read.

## §3 — The ceiling probe (gate-2; validation lanes ONLY)

**Question:** does the v0 object-DSL bank contain the correct output for materially more validation tasks than
the frozen v2 bank — BEFORE any selector/solver investment? (The E3 lesson operationalized: measure the oracle
ceiling first.)

```text
register   = docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv   (sha256_expansion)
lanes      = validation_lodo ∪ validation_pttest  (155 instances; U_primary is NOT touched)
barrier    = generate all admitted candidates from conditioning pairs only →
             write candidate fingerprints (instance id, program id, output grid_hash) + sha256 →
             only then read validation targets to compute ceilings
baseline   = the v2-bank validation ceiling, computed OFFLINE from the E3 receipt's
             candidate_fingerprints_no_targets_validation.jsonl (per-candidate grid_hash vs the
             identically-hashed target; the E3 barrier file's sha256 is recorded) — no v2 rerun
metrics    = per lane + pooled: oracle ceiling (distinct tasks with target-in-bank; instances),
             no_admitted rate, budget-exhaust rate, per-prior breakdown (diagnostic)
```

**Pre-registered gates (pooled validation, distinct tasks; precedence in table order):**

| gate | condition | consequence |
| --- | --- | --- |
| `GEN1_CEILING_LIFT` | GEN-1 ceiling ≥ max(2 × v2 ceiling, v2 ceiling + 3) | a Branch-F-style binding spec may be WRITTEN (own arena gate, verdict table, selector plan — the E3 ranker spec re-attaches there) |
| `GEN1_CEILING_MARGINAL` | above EMPTY, below LIFT | ONE v1 grammar-extension round (§4, pre-named families only), then re-probe; terminal after that |
| `GEN1_CEILING_EMPTY` | GEN-1 ceiling ≤ v2 ceiling + 1 | GEN-1 dies; after a MARGINAL→v1 round has also failed (or immediately if v0 lands here), the honest terminal state stands: ARC closed at the deterministic baseline with the generator-class wall characterized |

The absolute anchor in LIFT guards against the near-zero baseline (87% zero-candidate) making a trivial +1
look like a lift. **No gate, view, selector, or budget may be retuned after reading validation targets.**

## §4 — The ONE pre-authorized v1 extension round (pre-named NOW, so extension ≠ tuning)

If (and only if) the v0 probe lands `GEN1_CEILING_MARGINAL`, exactly one grammar extension round is admitted,
drawing ONLY from these pre-named families (append-only amendment, then one re-probe, then terminal):

1. **Relational placement:** `move_onto(S2)` / `align_with(S2)` (anchor = a second selector).
2. **Drawing:** `draw_line_between(S2a, S2b)` (straight lines/paths between object anchors).
3. **Count-driven canvas:** output dims or content repetition driven by an object COUNT (CEGIS-checked).
4. **Attribute-map recolor:** per-object recolor by a solved attribute→color map (e.g., size-rank → color).
5. **Background inference:** background = most-frequent color, as a 4th enumerated view.

Anything outside these five families requires killing GEN-1 first and opening a new slate candidate.

## §5 — Execution gating & required artifacts (for the later tooling amendment)

Execution requires a freeze-marker amendment recording: runner (`docs/prereg/arc/gen1_object_dsl_probe.py`
reserved), wrapper + npm wiring (`arc:gen1:ceiling-probe` reserved), leak-check receipt, a solver-correctness
smoke on known-solvable synthetics (an object-rule analog of the Branch-E sanity check), measured s/instance +
ten-minute-rule decision, and the exact staged command. Binding output path reserved:
`results/arc/gen1-object-dsl-ceiling-probe/` (manifest, split, barrier + sha256, candidates_by_instance,
ceiling_summary, v2_baseline_ceiling, per_prior_ceiling, receipt, adjudication, commands, hashes).

## §6 — Public language

Before any probe receipt:

> "A perception/grammar freeze and ceiling-probe protocol for an object-centric generator class is filed.
> No tooling, no probe run, no receipt exists. It adjudicates no capability branch."

After a probe receipt, the ONLY permitted statements are the gate outcomes of §3 phrased as ceiling
measurements ("the v0 object-DSL bank's validation oracle ceiling was X distinct tasks vs the v2 bank's Y"),
never capability, sufficiency, solve, public-eval, or Kaggle claims. A `LIFT` outcome permits saying a binding
spec may now be written; it is not itself a capability result.
