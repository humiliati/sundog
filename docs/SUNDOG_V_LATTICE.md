# Sundog vs. Lattice Deduction

Working hook:

> The shadow is not discovered after the fact. In a lattice-deduction reasoner,
> the shadow is the state the system is forced to pass through.

Short version:

> Sundog vs. Lattice asks whether a neural reasoner whose state is an explicit
> abstract lattice can give an exact computational witness: decision-observable,
> state-unobservable, and sound across the fiber.

Status: **paper-design-only lane opened 2026-06-02**. The phase specs live under
[`lattice/`](lattice/). The current frozen pre-registration is the B2 twin-state
cell, committed as a sibling research lane to [`chatv2/`](chatv2/), not as
`chatv3` and not as a product surface. Execution is deferred until a usable LDT
model exists or the reimplementation/build-gate path is explicitly opened.

This roadmap is the parent scaffold. It names the coupling, the phase ladder,
the public-workbench gate, and the promotion / peer-review exit paths. It does
not add `lattice.html`, does not add a `site-pages.json` launch entry, and does
not promote anything into [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md)
until a receipt earns it.

Related docs:

- [`lattice/LANE_CHARTER.md`](lattice/LANE_CHARTER.md) - lane charter,
  alpha/gamma reframe, access gate, and Target A/B split.
- [`lattice/PHASE0_MINIMUM_FALSIFIABLE.md`](lattice/PHASE0_MINIMUM_FALSIFIABLE.md)
  - frozen B2 twin-state decision-vs-reconstruction cell.
- [`lattice/LITPASS_MEMO.md`](lattice/LITPASS_MEMO.md) - Phase -1 verified citation
  spine (Cousot 1977; HRM / TRM; SATNet / RRN / NeuroSAT; D3PM; Massive
  Activations), peer-review shortlist, and lane-specific pre-registered negatives.
- [`lattice/PHASE3_B1_INTERNAL_BODY_FINGERPRINT.md`](lattice/PHASE3_B1_INTERNAL_BODY_FINGERPRINT.md)
  - Phase 3 B1 companion design memo: where the body lives across layers/iterations
  and whether it is wider than the elimination decision.
- [`lattice/PHASE4_B3_SOUNDNESS_FRONTIER.md`](lattice/PHASE4_B3_SOUNDNESS_FRONTIER.md)
  - Phase 4 B3 companion design memo: the false-elimination vs progress/abstention
  frontier the LDT loss already encodes.
- [`chatv2/LANE_CHARTER.md`](chatv2/LANE_CHARTER.md) - sibling residual-body
  lane and the variance-PR masking lesson this lane inherits.
- [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md) - body-resistance axis
  and cross-substrate failure map.
- [`proof/PDE_C1_PROPOSITION.md`](proof/PDE_C1_PROPOSITION.md) - twin-state /
  decision-observable-state-unobservable precedent.

## 0. Why This Lane Exists

Most Sundog lanes start by trying to discover a useful projection:

- ARC searched for a signature that could organize task fibers and found no
  certifiable local geometry.
- Mesa found an irreducible control subspace but a marginal body-resistance
  profile.
- NSE and shell probes found low-dimensional or slaved bodies.
- Chatv2 is the first deconfounded residual-stream lane where the body resists
  and scales, but the SHARP verdict is still seed-stability gated.

The Lattice Deduction Transformer proposal changes the order of operations. The
abstract lattice is not an after-the-fact compression. It is the object the model
is architecturally forced to act through. The paper's alpha/gamma Galois pair is
the Sundog projection/fiber pair in direct form:

| lattice-deduction object | Sundog object |
| --- | --- |
| alpha: concrete states to abstract lattice | projection / shadow map |
| gamma: abstract lattice to consistent concrete states | fiber over the shadow |
| non-singleton gamma fiber | state-insufficiency |
| sound deduction over the lattice | control-sufficiency with a guarantee |
| learned within-pass activations | candidate high-dimensional body |

That makes the substrate attractive for a precise reason: the shadow leg is
architecturally given. The remaining question is not "can we find a
control-sufficient shadow?" It is:

```text
Does the learned reasoner build a genuinely high-dimensional body around that
shadow, while preserving sound decisions across the certified fiber?
```

That isolates the confound that has kept prior control substrates marginal.

## 1. What Is Honest vs. What Is Reach

**Honest:**

- A paper-design-only pre-registration can state the analogy and the exact
  measurement target before a model is available.
- A reimplementation can be measured only after a build-gate reproduces the
  paper's Sudoku-Extreme headline cell.
- A clean positive would be an exact computational analogue of the C1
  decision-vs-reconstruction story: same lattice, different concrete solutions,
  shared learned decision.
- A clean null is still valuable: it would show that even with a certified
  shadow, the learned neural body collapses to a low-dimensional decision body.
- An unsound result is substantive: it would say the learned model does not
  preserve the certified fiber under the measured eliminations.

**Reach; do not claim:**

- "Sundog proves the LDT paper."
- "Sundog solves Sudoku / reasoning / abstract interpretation."
- "The alpha/gamma analogy is a Sundog discovery."
- "A definitional abstraction/fiber pair is itself a new result."
- "A reimplemented model result is a claim about the authors' unreleased model."
- "A public workbench is warranted before build-gate and review."

The strongest permissible positive would be narrow:

> In a reproduced LDT-style Sudoku reasoner, Sundog measured a certified
> state-insufficient lattice fiber, sound learned eliminations across that fiber,
> and a high-dimensional within-pass body wider than the decision shadow.

Everything stronger waits for external review.

## 2. Claim Boundary

This document claims only that:

1. the LDT abstraction/concretization setup is structurally coupled to Sundog's
   projection/fiber vocabulary;
2. the current `docs/lattice/` pre-registration correctly avoids the vacuous
   inter-step test by relocating the body measurement to within-pass activations;
3. the phase path is concrete enough to stage a future workbench and review
   packet if the model/substrate couples well.

It does not claim that the lane has run, that the LDT model has been reproduced,
or that the analogy has earned public promotion.

## 3. Ratified Hook Language

Safe hook:

> Sundog vs. Lattice asks whether a neural deduction system whose state is an
> explicit abstract lattice can make sound decisions across a concrete fiber
> without reconstructing which concrete solution is inside it.

Short public-safe version, if a review-only page ever exists:

> Same lattice, many solutions. Does the learned reasoner act soundly from the
> lattice without secretly needing the full solution?

Avoid:

- "Sundog discovers the LDT mechanism."
- "The model proves regime-2 by construction."
- "The lattice is high-dimensional body-resistance."
- "The workbench validates the paper."
- "This is chatv3."

## 4. Phase Ladder

### Phase -1 - Paper-Design Freeze

Status: **done** - analogy + vacuity guard in
[`lattice/LANE_CHARTER.md`](lattice/LANE_CHARTER.md); verified citation spine +
peer-review shortlist in [`lattice/LITPASS_MEMO.md`](lattice/LITPASS_MEMO.md).

Goal: register the analogy and the vacuity guard before any implementation.

Exit criteria:

- alpha/gamma mapped to projection/fiber;
- Target A definitional separation explicitly not claimed;
- Target B learned-model content named;
- access/build-gate deferred;
- no public page or cross-substrate row promoted;
- citation spine verified (abstract-interpretation foundation + HRM/TRM lineage +
  neural-symbolic precedents + the residual-stream body-measurement basis) and a
  peer-review shortlist named (owner-to-select at Phase 7), in `LITPASS_MEMO.md`.

Primary failure mode:

- **analogy inflation** - the roadmap claims credit for abstract-interpretation
  folklore rather than using it as the substrate.

### Phase 0 - B2 Twin-State Minimum Cell

Status: **frozen; execution deferred** in
[`lattice/PHASE0_MINIMUM_FALSIFIABLE.md`](lattice/PHASE0_MINIMUM_FALSIFIABLE.md).

Core question:

> Given a non-singleton lattice fiber, does the learned model make sound,
> non-trivial decisions that are shared across concrete solutions, while its
> within-pass body remains high-dimensional relative to the decision?

Frozen headline verdicts:

- `CERTIFIED_SHARP`
- `CERTIFIED_MARGINAL_BODY`
- `UNSOUND`
- `VACUOUS` / `INVALID`

Exit criterion:

- no execution until Phase 1 build-gate produces a faithful model.

### Phase 1 - Access and Build Gate

Goal: obtain a measured substrate without confusing "we built a model" with
"the lane has a result."

Preferred path:

- reimplement the 800K LDT-style Sudoku model;
- train on the public Sudoku-Extreme setup;
- instrument every lattice state and within-pass activation;
- reproduce the paper's 100% Sudoku-Extreme headline cell before interpreting
  any body/fiber number.

Reserved receipt:

```text
results/lattice/build-gate-sudoku-extreme/
```

Branch criteria:

| branch | condition | disposition |
| --- | --- | --- |
| `build_gate_pass` | reproduced Sudoku-Extreme target, clean manifest, instrumentation available | admit Phase 0 execution |
| `build_gate_partial` | model learns but misses target | no B-layer verdict; diagnose implementation |
| `build_gate_fail` | cannot reproduce substrate | close or re-register |
| `access_checkpoint_available` | authors or public release provide usable model | run an access/fidelity audit before Phase 0 |

The build-gate is not optional. A broken model would make every downstream
measurement uninterpretable.

### Phase 2 - B2 Twin-State Execution

Goal: run the frozen Phase 0 minimum cell against the admitted model.

Receipt:

```text
results/lattice/phase0-twinstate/
```

Outputs:

- fiber-size distribution;
- false-elimination rate;
- decision non-triviality;
- `d_dec` information-basis body dimensionality;
- `k_control / d_dec`;
- cross-decode guard for concrete solution leakage;
- branch adjudication.

Promotion implication:

- `CERTIFIED_SHARP`: earns a cross-substrate row candidate and a review packet.
- `CERTIFIED_MARGINAL_BODY`: earns a clean null note, not a public claim.
- `UNSOUND`: earns a learned-soundness critique, not a Sundog-positive.
- `VACUOUS` / `INVALID`: repair the substrate or sampler.

### Phase 3 - B1 Internal-Body Fingerprint

Companion design memo: [`lattice/PHASE3_B1_INTERNAL_BODY_FINGERPRINT.md`](lattice/PHASE3_B1_INTERNAL_BODY_FINGERPRINT.md)
(filed 2026-06-02; explanatory not headline; execution gated on Phase 1 + Phase 2).

Goal: separate the twin-state headline from a broader body-resistance profile.

Question:

> Across layers, unrolled steps, tokens, and lattice states, where does the
> learned reasoner carry information, and is that body wider than the decision
> needs?

Measures:

- layer/iteration profile of `d_dec`;
- raw PR and robust PR for continuity, but not as the sole gate;
- `k_control` for eliminations and conflict decisions;
- FVE-style reconstruction from the lattice shadow where meaningful;
- outlier/medium characterization inherited from chatv2.

Disposition:

- If Phase 2 is positive, Phase 3 explains where the body lives.
- If Phase 2 is marginal, Phase 3 explains whether the marginality is global or
  localized.
- If Phase 2 is unsound, Phase 3 can still diagnose the learned failure, but it
  does not rescue promotion.

### Phase 4 - B3 Learned-Soundness Frontier

Companion design memo: [`lattice/PHASE4_B3_SOUNDNESS_FRONTIER.md`](lattice/PHASE4_B3_SOUNDNESS_FRONTIER.md)
(filed 2026-06-02; inference-time sweep primary, lambda_cls retraining gated;
execution gated on Phase 1 + Phase 2).

Goal: characterize the tradeoff the paper-facing loss already encodes:
false-eliminations vs abstention/missed-conflicts.

Possible knobs:

- conflict threshold;
- elimination threshold;
- loss-weight analogue if retraining is admitted;
- on-trajectory vs constructed under-constrained lattice populations.

Receipts:

```text
results/lattice/phase4-soundness-frontier/
```

Branch criteria:

| branch | condition |
| --- | --- |
| `frontier_stable_sound` | soundness remains below false-elim threshold while progress stays non-trivial |
| `frontier_tradeoff_sharp` | a real Pareto edge appears: progress can be bought only by unsoundness |
| `frontier_abstention_marginal` | the model is sound mostly by doing little |
| `frontier_sampler_invalid` | constructed lattices do not represent the intended fiber class |

### Phase 5 - Coupling Adjudication

Goal: decide what the lane has earned before any website work begins.

Adjudication table:

| evidence state | earned language | next move |
| --- | --- | --- |
| Phase 2 `CERTIFIED_SHARP` + Phase 4 stable | exact computational regime-2 candidate | external review packet, then review-only workbench |
| Phase 2 `CERTIFIED_MARGINAL_BODY` | certified abstraction, marginal neural body | write null note; no public workbench unless educational |
| Phase 2 `UNSOUND` | empirical soundness boundary | ask for peer review / author sanity check; no promo |
| build gate fails | no substrate | close deferred or re-register |
| analogy challenged by reviewer | analogy is reader overlay only | revise or retire lane |

Phase 5 is where this roadmap either couples to the larger body-resistance
program or stays a bounded literature-inspired sidecar.

### Phase 6 - Review-Only Workbench Design

Goal: design `lattice.html` only after the empirical lane earns a surface.

The first page should be **review-only**, likely noindex/unlinked, mirroring the
Kakeya public-boundary discipline. It should be an inspectable workbench, not a
promotional page.

Candidate workbench panes:

1. **Lattice board.** Sudoku cells with candidate sets, eliminated candidates,
   and committed decisions.
2. **Fiber twins.** Two concrete solutions consistent with the same lattice,
   shown only when the fiber is certified non-singleton.
3. **Soundness rail.** False-elimination / missed-conflict counters and the
   exact candidate whose removal would break the fiber.
4. **Body trace.** Layer/iteration heatmap of `d_dec`, `k_control`, and
   decision-readout strength.
5. **Claim boundary.** Visible badge: result / null / unsound / review-only.

Reserved page:

```text
lattice.html
```

Do not add it to `site-pages.json` until the SEO/social roadmap requirements are
intentionally handled and the page's evidence tier is known.

### Phase 7 - External Review Packet

Goal: get the analogy and measurement checked by people outside the lane.

Reviewer profiles:

- abstract interpretation / Galois connection specialist;
- neural reasoning or Sudoku/LDT-style model builder;
- constraint programming / exact Sudoku enumeration specialist;
- interpretability person familiar with residual-stream outlier features.

Packet contents:

- roadmap claim boundary;
- `docs/lattice/` phase specs;
- build-gate receipt;
- Phase 2 and Phase 4 receipts;
- workbench screenshots if Phase 6 exists;
- direct questions where the lane could be wrong.

Ask reviewers specifically:

- Is alpha/gamma -> projection/fiber a legitimate coupling or just vocabulary?
- Does the twin-state construction certify state-insufficiency in the claimed
  sense?
- Is false-elimination rate the right learned-soundness boundary?
- Is `d_dec` a defensible body-dimensionality proxy for this substrate?
- Would the workbench teach the right thing?

### Phase 8 - Public Launch, Promo, or Peer-Review Ask

Goal: choose the public posture by result, not by enthusiasm.

Outcome ladder:

| result | public posture |
| --- | --- |
| `CERTIFIED_SHARP` + favorable review | promote `lattice.html` as a review-backed research workbench; add cross-substrate row |
| `CERTIFIED_SHARP` + unresolved review concerns | publish as review-only ask, not promo |
| `CERTIFIED_MARGINAL_BODY` | publish a null/methodology note only if educational value is high |
| `UNSOUND` | ask for peer review / author sanity check; no positive promo |
| build/access failure | no page |

Any public launch must keep the sentence:

> This is a bounded measurement of a reimplemented lattice-deduction substrate,
> not a validation of the original paper and not a claim that Sundog solves
> symbolic reasoning.

## 5. Workbench Boundary

`lattice.html` is late-path because the public surface can easily overteach the
wrong lesson. The board will be visually compelling: cells, candidates, twins,
and eliminations. That makes the boundary more important, not less.

The workbench may ship only if one of these is true:

1. a positive result earns review-only inspection; or
2. a clean null/unsound result is valuable enough as a methodology exhibit and
   is explicitly labeled as such.

It must not ship as:

- a Sudoku solver demo;
- a proof of the LDT paper;
- a generic "neural reasoning" showcase;
- a replacement for the frozen receipts.

## 6. Relationship to Chatv2

Do not rename this lane `chatv3`.

Chatv2 asks whether a transformer residual stream can build a high-dimensional
computed body whose narrow decision shadow is state-insufficient. Lattice asks a
different question: if the control shadow is architecturally supplied as an
abstract lattice, does the learned within-pass body become high-dimensional and
sound across the exact fiber?

The two lanes cross-pollinate:

- chatv2 contributes the `d_dec` / information-basis lesson;
- lattice contributes the exact fiber / certified-shadow substrate;
- both inherit the C1 twin-state decision-vs-reconstruction frame.

But they should stay separate until both have receipts. If chatv2 stabilizes,
lattice is the exact-abstract-interpretation analogue. If chatv2 remains
seed-sensitive, lattice is the cleanest next substrate because the shadow leg is
no longer learned or discovered.

## 7. Promotion Criteria

The rung ladder (**R0** receipt -> **R1** reimplemented-substrate -> **R2**
substrate-general / real-model -> **R3** theory) with its per-rung pre-registered
gates, the do-not-claim ledger, and the adversarial pre-mortem are the **canonical
promote-gate**: [`lattice/PROMOTE_GATE.md`](lattice/PROMOTE_GATE.md) (ported from
the chatv2 gate; binding once ratified). **No rung's language may be used until
that rung's gate is cleared** - a result is a *receipt* until promoted, and
external review **plus the chatv2 audit team** gate any claim past R1. The R1 gate
folds in the build-gate pass, the cross-decode / vacuity guards, generality across
>=2 tasks, and honest reporting of `CERTIFIED_MARGINAL_BODY` / `UNSOUND` as results.

Where a promoted result then goes (mechanics):

- a narrow row to [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md);
- an entry in the review tracker if public-facing;
- a `site-pages.json` entry only after SEO/social readiness is deliberately
  satisfied;
- a short note in the relevant promo document only after review.

## 8. Immediate Next Steps

Recommended now:

1. Keep [`lattice/`](lattice/) frozen and paper-design-only.
2. Let the active chatv2 seed-stability cell finish.
3. Draft B1/B3 specs as companion design memos only if they clarify the lane.
4. Do not build `lattice.html` yet.
5. If a model/reimplementation window opens, start with Phase 1 build-gate, not
   with the twin-state measurement.

This is enough scaffolding for the lane to be real without cutting into the
current chatv2 adjudication. The workbench comes near the end, when it can
illuminate a receipt rather than audition for one.
