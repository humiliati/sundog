# Sundog vs. Allelopathy

Working hook:

> The shadow is partial. When does it stop being partial?

Short version:

> Allelopathy, EM side-channels, and "field" language all collapse to one
> falsifiable question: when can a leaky, low-dimensional observation constrain
> the rest of a system? Sundog's in-scope version is now split in two: use the
> existing chatv2 residual bodies as a cheap prediction-bank control for the
> uncoupled same-seed null, and put the discovery budget on cross-seed transplant
> plus a coupled-substrate closure bracket. An animated SVG is allowed only
> after the spec and receipts decide what it is allowed to show.

Status: Scaffold opened 2026-06-03. No public page, `site-pages.json` entry,
SVG, promotion copy, or external-review packet exists. The first target is a
research-internal roadmap and probe spec using the saved seed-stability bodies:
`results/chatv2/phase1-seedstab/seed{0,1,2}/bodies/H8_gen.npz`. Those bodies are
`H=8`, computed pair-XOR, fair-readout residual tensors with shape
`(3000, 4, 8, 192)`, and the receipt already records stable same-seed resistance:
`d_dec = 7.5-7.6`, `z1_acc = 0.80-0.86`, `leak = 0.51-0.52`, and decomposed
`objective_excess = 0.205 +/- 0.022`. Updated posture: because the pair-XOR
latents are independent by construction, same-seed determining-set closure is a
predicted null, not the headline discovery. The saved-body sweep is still useful
as a verify-before-file control: bank the prediction that independent latent
shadows do not reconstruct other independent latent bodies. The genuine open
frontier is (i) cross-seed coordinate transplant on the saved bodies and (ii)
`k_func << k_state` closure on a coupled substrate.

This is not a claim about real LLMs, consciousness fields, plant allelopathy,
biological signaling, or the electromagnetic emanations of hardware. It is a
partial-observability probe on a synthetic toy residual stream.

**Pre-registration freeze (2026-06-03).** This roadmap is frozen as the
plan-of-record: the probe split (same-seed prediction-bank control vs cross-seed
transplant headline), the load-bearing `closure is a coupling phenomenon`
reservation of `k_func << k_state` to a coupled substrate, the Core Objects
definitions, the branch tables, the attack list, and the allowed/forbidden
language are committed. Per the pre-registration discipline, thresholds and branch
definitions may be **tightened or replaced before any result is read, never
loosened after a sweep is seen**. Phase 0 translates this frozen plan into the
runnable spec (`docs/chatv2/PHASE2_DETERMINING_SHADOW_SET_SPEC.md`); no
verdict-bearing run is admitted until that spec freezes against this roadmap.

## Claim Boundary

This document explicitly does **not** claim:

- that a GPU's EM leakage is the model's representation;
- that hardware side-channels are useful for reading modern model activations;
- that separately trained models can in general read or steer each other;
- that the chatv2 toy result explains intelligence, world models, or scaling;
- that a small animated graphic is evidence for any real AI-safety claim;
- that a positive cross-seed result would license metaphysical field language.

What this roadmap may stage:

- a same-seed determining-shadow-set **control**: sweep shadow subsets `S` of
  the eight independent latent readout directions and verify the predicted null;
- a cross-seed "allelopathy" **headline** probe: test whether a source seed's
  shadow set constrains a target seed's body without target-label refitting;
- a coupled-substrate closure bracket: test `k_func << k_state` only where
  there is real latent/state coupling for a closure relation to exploit;
- negative controls that make label leakage, same-seed overfit, and variance
  artifacts visible;
- a data-driven, self-contained animated SVG that visualizes the actual branch:
  collapse, partial collapse, or no collapse.

## Why This Fits Sundog

The useful kernel is not the costume. It is the observability question:

```text
body X  ->  shadow pi_S(X)  ->  what can be reconstructed or constrained?
```

The existing chatv2 lane already measured the one-shadow case: a `z_1` readout
is control-sufficient for `z_1` but state-insufficient for the other latents
(`cross_latent_leak` sits near chance). The next structural question is not
"does one shadow fail?" but:

```text
How many shadows are needed before the rest of the body stops resisting?
```

That number, if it exists below `H`, is the rigorous form of "what constrains the
body through the shadow." If no subset below all eight latents collapses the
body, that is also a result: the toy body remains additively resistant under the
registered information basis.

But the saved chatv2 toy has an important built-in answer: its latents are
independent Bernoulli variables in independent parity channels. That independence
is what made the de-confound clean, and it also removes the coupling a
determining set would need to exploit. So the same-seed sweep should be read as
a prediction-bank control, not a fishing expedition. The expected branch is:

```text
independent latents + leak ~= chance + d_dec ~= H  ->  no within-seed closure below H
```

The discovery question moves to representations that may share geometry across
training seeds, or to substrates where the state variables are actually coupled.

## The NSE Lesson - Functional Closure Before State Reconstruction

The Navier-Stokes C1 lane gives the best clue for making this probe sharp. Its
important distinction is not "can the shadow reconstruct the whole body?" but:

```text
Can the shadow determine the relevant unresolved functional before it determines
the unresolved state?
```

In C1 terms:

- `Phi_K` is the low-mode shadow.
- `Q_K` is the unresolved body.
- state reconstruction was physically marginal at moderate 2D NSE:
  `FVE(Q_K | Phi_K) ~= 0.99` in energy/enstrophy norms.
- the useful closure read was instead the unresolved transfer functional `R`:
  `R2(R | Phi_K) ~= 0.99`, while state pairs still varied inside `Phi_K` fibers.

The direct transplant into this lane is a two-threshold bracket:

```text
k_func  = smallest |S| where the relevant omitted functional is determined
k_state = smallest |S| where the registered omitted state proxy is reconstructed
```

The strongest result would be `k_func << k_state`: a small shadow set closes the
decision-relevant or coupling-relevant quantity while the body still resists.
If `k_func ~= k_state`, the probe is closer to determining modes / state
reconstruction. If neither threshold is reached, the body stays resistant under
the available shadow family.

This reframes the animated graphic too: the satisfying visual is not "shadows
rebuild the body." It is "a shadow set pins the closure handle while the body
cloud still has thickness."

Load-bearing transplant from C1: **closure is a coupling phenomenon.** C1 worked
because high and low modes are dynamically coupled (`R2(R | Phi_K) ~= 0.998` is
a coupling statement). The saved chatv2 pair-XOR toy was built to remove
cross-latent coupling. Therefore its within-seed `k_func << k_state` result is
not expected to exist; a null would confirm the design, not disappoint the lane.

## The EMF Bridge

EM and power side-channels are real: circuits leak information through physical
channels, and cryptographic attacks sometimes reconstruct hidden computation
from those leaks. Sundog may use that as a lens because it is the same abstract
shape:

```text
hidden computation -> leaky channel -> partial reconstruction
```

The hard line:

- **Allowed:** "EM side-channels are a concrete example of leaky observation and
  reconstruction."
- **Forbidden:** "the physical EM foam around a GPU is the model's body."

For this lane, the body lives in activation geometry. The side-channel analogy
belongs in the animation as a small caution strip, not as the substrate claim.

## The Allelopathy Bridge

The non-mystical allelopathy question is:

> Do independently trained systems carry enough shared representational
> structure that a shadow read from one can constrain the body of another?

That is testable now because the seed-stability run saved three independent
`H=8` bodies under the same task family. The hard version is direct transplant:
fit the source seed's layer, shadow readout directions, standardization, and
decoder; apply them to the target seed's body; and measure target-latent recovery
without target-label layer selection or refit. Any label-aware target alignment
is an upper-bound diagnostic, not the verdict.

## Core Objects

| object | in this lane |
| --- | --- |
| body | saved chatv2 generative residual body, preferably the frozen best layer per seed |
| atomic shadow | a latent readout score or direction for one `z_i` |
| shadow set `S` | a subset of latent readout scores, `|S| = 1..7` |
| resistance | omitted latents/body coordinates remain unreconstructable from `S` |
| collapse | `S` predicts the omitted information above the pre-registered threshold |
| determining number `k_det` | smallest `|S|` whose best subset collapses resistance |
| functional closure number `k_func` | smallest `|S|` that determines a registered omitted functional |
| state reconstruction number `k_state` | smallest `|S|` that reconstructs the registered omitted state proxy; for saved bodies, omitted readout-score coordinates `s_j` |
| cross-seed pass | source-seed `S` constrains target-seed body without target-label refit |

The probe should preserve the chatv2 information-basis lesson: raw variance PR
alone is not enough. Readout-direction geometry, omitted-latent leakage, and
body reconstruction should all be reported separately.

## Probe 1 - Same-Seed Determining-Set Control

Question:

> Does the saved independent-latent toy behave exactly as it should: no shadow
> subset `S < H` reconstructs or closes the omitted independent latents?

This probe is deliberately **not** the discovery headline. It is a cheap
verify-before-file control. The same-seed result is already strongly predicted
by the construction and receipt:

```text
z_i independent by construction
cross_latent_leak ~= 0.51
d_dec ~= 7.6 ~= H
=> omitted-latent closure should stay at chance for every S < H
```

Frozen design shape for the later spec:

- Use only saved `H=8` generative bodies from `phase1-seedstab`.
- Fit per-latent readout directions on a training split inside each seed.
- For every subset `S` with `|S| = 1..7`, compute shadow scores on held-out rows.
- Predict omitted latent labels `z_j, j not in S` from the scores in `S`.
- Predict omitted readout scores `s_j` from the scores in `S`; this is the
  registered `k_state` body proxy for the saved-body spec, not a full
  residual-minus-`S` reconstruction.
- Separately classify `k_func` and `k_state`:
  - `k_func`: closure of registered omitted functionals.
  - `k_state`: reconstruction of omitted readout-score coordinates.
- Compare against permutation controls, same-size random projections, and the
  existing one-shadow `z_1` leak baseline.

Candidate functionals for the spec to freeze, in priority order:

1. **Omitted-latent functional.** Predict the omitted latent vector or a frozen
   aggregate of it from `S`. This is cheap and available from saved bodies. In
   the saved pair-XOR toy, it is predicted to stay at chance.
2. **Readout-closure functional.** Predict the omitted latent readout scores,
   not the labels. This asks whether the source shadow closes the information
   basis itself. In the saved pair-XOR toy, any pass here would be surprising
   and must first be attacked as leakage or same-seed overfit.
3. **Model-output functional.** Scoped out of this saved-body spec. If
   logits/checkpoints are admitted later, measure the next-token loss or decision
   margin under a separate coupled-substrate/Phase 7 spec. This is the closest
   chatv2 analogue of NSE's unresolved-transfer `R`, but the saved `.npz` bodies
   do not contain the runner surface needed to measure it, and the uncoupled toy
   is the wrong substrate for a headline `k_func << k_state` closure.

Proposed, non-binding gates for the spec to freeze:

```text
k_func:  rest_functional_acc_or_R2 >= 0.70
k_state: omitted_score_FVE >= 0.60
permutation controls near chance / <= 0.12 where applicable
```

The spec may tighten or replace those thresholds before any result is read. It
must not loosen them after seeing the sweep.

Expected branches:

| branch | condition | interpretation |
| --- | --- | --- |
| `det_shadow_predicted_null` | no subset `|S| <= 7` reaches either threshold and controls pass | predicted null confirmed: independent latents do not determine each other |
| `det_shadow_functional_closure` | `k_func < k_state` or `k_func` exists while `k_state` does not | surprising in this toy; attack as leakage before treating as NSE-like closure |
| `det_shadow_state_collapse` | `k_func ~= k_state < 8` | the shadow set reconstructs because it has effectively become state-determining |
| `det_shadow_partial` | one metric passes but controls or companion reads fail | constrained in one basis, not enough for the headline |
| `det_shadow_void` | split, leakage, shape, or control contract fails | no result; repair spec/tooling |

The SVG is not allowed to draw a clean collapse if the branch is partial or null.
If `det_shadow_functional_closure` lands, the body should remain visibly thick
while the functional gauge locks.

## Probe 1b - Paired Fiber / Boundary-Layer Audit

The NSE hardening wave turned a global regression into a stronger local claim by
finding signature-near state pairs and asking whether the action stayed
constant across them. This lane should port that audit.

For each subset `S`:

- find held-out pairs whose shadow scores are within a frozen radius;
- require the omitted body targets to differ enough to count as non-injective;
- measure whether the registered functional stays constant across those pairs;
- report the disagreement rate `D_witness`;
- stratify disagreements by decision/readout margin.

Proposed branches:

| branch | condition | interpretation |
| --- | --- | --- |
| `paired_fiber_closure_positive` | same-`S` pairs differ in body but agree on functional except a thin margin set | local functional closure, NSE-style |
| `paired_fiber_boundary_only` | disagreements concentrate near a frozen decision margin | the bad set is a boundary layer, not global insufficiency |
| `paired_fiber_conflict` | same-`S` pairs differ in body and functional away from the margin | `S` is not functionally sufficient |
| `paired_fiber_deferred_coverage` | too few same-`S` pairs under the frozen radius | no verdict; adjust only in a new spec |

This audit also guards the animation: if all failure lives in a boundary layer,
the SVG should show a thin switching surface, not a failed or magical transfer.

## Probe 2 - Cross-Seed Transplant

Question:

> Does a shadow set learned on seed `a` constrain the body of independently
> trained seed `b`?

This is the real "allelopathy" frontier for the saved bodies. It is not answered
by within-seed independence, because it asks whether independently trained
models converge to a shared coordinate geometry for the same task. The analogy
to NSE is not "low modes determine high modes"; it is "two systems share a
common attractor/task geometry strongly enough that a shadow basis learned in
one system lands in another."

Use all six directed pairs among seeds `{0, 1, 2}`. The verdict tier order should
be hard:

1. **Direct transplant:** source layer, source readout directions, source
   standardization, and source decoder applied to the target body. No target
   labels, no target layer selection, no target refit. This is the only strong
   allelopathy-positive tier.
2. **Unlabeled calibration:** same source layer/directions/decoder with target
   mean/scale calibration only. This is a weaker coordinate-shape pass.
3. **Label-aware or Procrustes ceiling:** uses target labels or paired latent
   information. This is an upper bound and must be labeled as such.

Proposed branch table:

| branch | condition | interpretation |
| --- | --- | --- |
| `cross_seed_direct_pass` | direct transplant passes on at least 4/6 directed pairs | independently trained bodies share enough coordinate structure for the source shadow to constrain target |
| `cross_seed_calibrated_only` | direct fails, unlabeled calibration passes | shared structure exists only after distribution calibration; weaker allelopathy |
| `cross_seed_ceiling_only` | only label-aware alignment passes | abstract isomorphism, not a transplant result |
| `cross_seed_no_transfer` | all non-leaky tiers fail | same-seed structure is not cross-seed readable |
| `cross_seed_void` | pair/run contract fails | no result |

The animation must show the actual tier. A `ceiling_only` result should look like
"needs an oracle alignment," not like successful cross-seed signaling.

## First-Principles Attack List

Before any SVG or public language, attack the probes in these ways:

1. **Label leakage.** Ensure no omitted latent labels enter the shadow scores,
   alignment, subset selection, or target-body reconstruction.
2. **Subset cherry-pick.** Choose subset-selection policy on a training split and
   report held-out performance. Exhaustive all-subset reporting is allowed, but
   the headline `k_det` must obey the frozen selection rule.
3. **Same-seed overfit.** Use train/held-out splits inside every seed and
   bootstrap the seed rows.
4. **Cross-seed coordinate accident.** Report direct, calibrated, and ceiling
   tiers separately. Do not rescue a failed direct transplant with a leaky
   alignment and call it allelopathy.
5. **Variance masking.** Keep the information-basis reads primary; raw variance
   FVE is a diagnostic only.
6. **Twin/untrained floor.** Run the determining-set sweep on twin and untrained
   bodies where available or cheap. A positive that appears equally in untrained
   bodies is architectural floor, not learned structure.
7. **Permutation controls.** Shuffle shadow scores and labels independently; the
   reconstruction must collapse to the registered null.
8. **Graphic overclaim.** The SVG may visualize constraints, not causation, mind
   reading, or real-world model collusion.
9. **Base-rate vacuity.** Any functional/action target must have a pinned,
   non-degenerate base rate, preferably around `20-30%`. A target that almost
   never fires is deferred, not positive.
10. **State-vs-functional conflation.** Report `k_func` and `k_state`
    separately. A result that only reconstructs the body is not the NSE-like
    closure story.
11. **Uncoupled-substrate overread.** Do not demand a closure bracket from a
    substrate with no coupling. The saved pair-XOR body is designed to be
    independent; its same-seed null is a feature.

## Roadmap

### Phase 0 - Spec Freeze

Filed as `docs/chatv2/PHASE2_DETERMINING_SHADOW_SET_SPEC.md` (draft for
sign-off, not run).

Required contents:

- exact input paths and body tensor shape;
- layer-selection rule;
- train/held-out split rule;
- subset-selection rule;
- collapse thresholds;
- `k_func` / `k_state` definitions and branch table;
- paired-fiber radius and boundary-layer metric;
- cross-seed tier definitions;
- negative controls;
- receipt schema;
- timing smoke and the under-10-minute inline rule.

Reserved implementation names:

```text
scripts/chatv2_phase2_shadowset.py
results/chatv2/phase2-determining-shadow-set/
```

No verdict-bearing run is admitted until Phase 0 freezes.

### Phase 1 - Same-Seed Prediction-Bank Control

Run the subset sweep on seeds `0, 1, 2`.

Expected output:

```text
same_seed_subset_sweep.csv
same_seed_kdet_summary.csv
same_seed_kfunc_kstate_summary.csv
paired_fiber_boundary_audit.csv
same_seed_controls.csv
same_seed_branch_adjudication.md
```

Expected branch: `det_shadow_predicted_null`. The receipt should report both the
best subset at each `k` and the full distribution over subsets, so a later
reader can see whether the predicted null is broad or whether one suspicious
subset needs a leakage audit.

### Phase 2 - Cross-Seed Transplant Headline

Run the six directed source-target pairs.

Expected output:

```text
cross_seed_transfer.csv
cross_seed_tiers.csv
cross_seed_controls.csv
cross_seed_branch_adjudication.md
```

The headline must name the tier: direct, calibrated, ceiling-only, no-transfer,
or void.

### Phase 3 - Internal Receipt

Write `docs/chatv2/PHASE2_DETERMINING_SHADOW_SET_RECEIPT.md`.

The receipt should include:

- the frozen spec hash or runner hash;
- git commit and dirty-state note;
- measured wall-clock;
- `k_det` per seed and aggregate branch;
- `k_func` / `k_state` per seed;
- paired-fiber `D_witness` and boundary-layer disposition;
- cross-seed tier branch;
- controls table;
- allowed and forbidden language;
- the exact data contract for the animation generator.

### Phase 4 - Animated SVG Gate

Only after Phase 3, generate a self-contained animated SVG plus static fallback:

```text
public/media/allelopathy-determining-shadow-set.animated.svg
public/media/allelopathy-determining-shadow-set.static.svg
public/media/allelopathy-determining-shadow-set.layers.json
```

The SVG should be generated from the receipt CSV/JSON, not hand-tuned to the
desired story. If the branch is null, the SVG shows the null: one or seven
shadows still leave the omitted body dark.

### Phase 5 - Optional Internal Page

If a review surface is useful, add an unlinked, noindex internal page. Do not add
a `publicLaunchIntent` page or public inbound path unless the normal website SEO
and social-readiness gate is opened and satisfied.

No external review is expected for this internal animated graphic. External
review becomes relevant only if the language escalates toward real LLMs,
steganographic collusion, hardware side-channel claims, or public AI-safety
claims.

### Phase 6 - High-Fidelity NSE Sidecar (Optional, Separate Spec)

If the user wants the literal PDE angle, stage it as a separate sidecar rather
than blending it into the toy SVG:

```text
docs/proof/PDE_C1_HIGH_FIDELITY_FUNCTIONAL_SHADOWSET_SPEC.md
```

Target shape:

```text
body        = high-fidelity vorticity state
shadow S    = selected low Fourier modes or shell summaries
rest        = unresolved modes Q_S
functional  = unresolved transfer R, lookahead energy/enstrophy hazard, or closure term
headline    = minimal |S| where functional is determined while state is not
```

The sidecar should use the C1 lesson directly:

```text
k_func = smallest S with high held-out R2(functional | S)
k_state = smallest S with high FVE(state | S)
NSE-like pass = k_func << k_state
```

Assumption for pursuing it: high-fidelity numerics or an adaptive/stiff
integrator removes the high-G numerical wall. Without that, it repeats the C1/C2
fixed-dt trap and should remain staged, not run.

### Phase 7 - Coupled-Latent Toy (Optional, Separate Spec)

If the goal is to test the closure bracket cheaply without the NSE numerical
wall, build a new toy where latent variables are coupled by construction. This
is a new substrate, not a reinterpretation of the saved pair-XOR bodies.

Candidate design shape:

```text
primitive latents u_1...u_m       independent sources
state latents z_1...z_H           coupled functions of u, e.g. Markov/tree/parity constraints
shadow set S                      readouts for a subset of z
functional R_toy                  a coupled closure target, e.g. parity/checksum/transition hazard
state target                      omitted z or residual body coordinates
headline                         k_func << k_state, if it exists
```

Good coupled-toy properties:

- de-confound still passes: no target is linearly input-decodable by accident;
- marginal/additive baselines are included;
- the coupling graph is known, so the expected `k_func` is pre-registerable;
- the uncoupled pair-XOR toy is run as a negative control with the same runner.

This is the toy analogue of high-fidelity NSE: closure has something real to
close over.

## SVG Storyboard

The graphic should be a receipt visual, not a marketing flourish.

1. **Body cloud.** A compact residual-stream cloud appears with eight possible
   latent shadows around it. The body is not drawn as a literal brain or field.
2. **One shadow.** `z_1` projects out cleanly; the rest of the body remains
   unlit. This anchors the existing seed-stability receipt.
3. **Set sweep.** Shadows are added one by one. Two meters move independently:
   `functional closure` and `state reconstruction`.
4. **Closure bracket.** If `k_func < k_state`, the functional meter locks while
   the body cloud remains thick. If only `k_state` lands, the graphic shows
   state reconstruction rather than NSE-like closure. If neither lands, the
   badge says `no closure below H`.
5. **Boundary layer.** Paired-fiber disagreements, if present, appear as a thin
   switching surface rather than as scattered noise.
6. **Cross-seed transplant.** A source seed's shadow fan is carried across to a
   target seed. The fan either lands directly, lands only after calibration, or
   misses. The branch is visible.
7. **Side-channel caution strip.** A small leaky-channel trace appears below the
   main panel: "same math, different substrate." It never becomes the main body.

Motion rules:

- self-contained SVG/CSS only, no external JS dependency;
- `prefers-reduced-motion` freezes to the static end-state;
- all text must remain legible at social-card and in-page sizes;
- do not animate ambiguity away; partial and null outcomes need their own visual
  states.

## Promotion Criteria

The animation may exist internally when all are true:

- Phase 0 spec filed and linked here;
- Phase 1 and Phase 2 receipts filed;
- controls pass or the failure branch is shown honestly;
- SVG data contract generated from receipts;
- reduced-motion fallback exists;
- no public site entry or inbound link is added by accident.

Public promotion requires a separate site/page gate, including the normal
`site-pages.json` and `docs/site/SEO_AND_SOCIAL_READINESS_ROADMAP.md` work. The
default posture is internal only.

## Allowed Language

Before probes:

> We are opening an internal same-seed control plus cross-seed transplant probe:
> the saved pair-XOR bodies should confirm the independent-latent null, while
> the live allelopathy question is whether a shadow basis learned in one seed
> lands in another.

If a same-seed closure appears on the saved pair-XOR bodies (unexpected):

> A subset of `k_func` shadows appeared to close the omitted functional on the
> saved pair-XOR bodies. Because those latents are independent by construction,
> this contradicts the predicted null: it is **suspect** and is **not announced**
> until it survives the full leakage, same-seed-overfit, and permutation attack.
> A genuine `k_func << k_state` closure is expected only on a coupled substrate
> (Phase 7), not on these bodies.

If same-seed state reconstruction lands first (also unexpected here):

> `k_func` and `k_state` collapsed together on the saved bodies. On independent
> pair-XOR latents this is also unexpected; it is held to the same leakage and
> overfit attack before any claim. If it survives, it is a determining-set result,
> not the stronger NSE-like functional-closure story.

If a closure lands on a coupled substrate (Phase 7):

> On a substrate with constructed latent coupling, a subset of `k_func` shadows
> determines the registered closure functional before the omitted state is
> reconstructed (`k_func << k_state`). This is a toy observability result on a
> coupled toy, not a real-LLM, NSE, or side-channel claim.

If no collapse:

> The toy body remains resistant below the full latent set: adding shadows does
> not close the registered functional or reconstruct the omitted content under
> the frozen controls.

If the predicted same-seed null lands:

> The saved pair-XOR body confirmed the independence prediction: below the full
> latent set, same-seed shadows do not close omitted independent latents. The
> result is a control receipt; the live allelopathy question is cross-seed
> transplant.

If cross-seed direct pass:

> A source seed's registered shadow directions constrain target seed bodies
> without target-label refitting on the toy substrate.

If direct fails:

> The same-seed shadow geometry did not directly transplant across independently
> trained seeds; any calibrated or label-aware pass is reported as a weaker tier.

## Forbidden Language

- "AI models communicate through a field."
- "The EM foam is the body."
- "This proves hidden collusion between model instances."
- "The toy result explains why LLMs work."
- "The animation demonstrates consciousness-field behavior."
- "A label-aware cross-seed alignment is allelopathy."

The test can be beautiful. The claim stays small.
