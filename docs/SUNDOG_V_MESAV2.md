# Sundog vs. Mesa-Optimization — v2 Traceability & Separability Spine

> _Filed 2026-05-13 as a ratified future-integration spine. Sister doc to
> [`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md), not a replacement. v1 keeps its
> empirical phase ledger and Phase 8 v1 public artifact; v2 adds the
> traceability spine that converts v1's earned envelope into a defensible
> traceability claim and protects it from "Sundog solves interpretability"
> drift._

Working hook:

> Sundog does not make traceability disappear. It moves the most important
> part of the objective outside the learned policy, into an auditable
> signature path. The mesa experiment then asks whether learned agents stay
> coupled to that path or reconstruct proxy objectives anyway.

## Why a v2 and not just an addendum

v1 earned three load-bearing anchors: a behavioral cliff at
`1 − λ ≈ 0.048` (Phase 5 v4), a mechanistic locus at an entangled 5D
subspace at `net.7` (Phase 6 v3.x), and an operating envelope with 22
classified cells (Phase 7 v1). Two pressures now push for a spine that
v1 does not carry on its own.

1. **Institutional readability.** Outside reviewers will ask whether this
   is "just another interpretability problem" or whether "Bayes nets
   already solved this." v1's phase ledger does not answer either
   framing directly; v2 ships the answer as Phase 0.5 + Phase 0.6.
2. **Crossover ratchet.** The
   [Mesa↔Geometry crossover note](MESA_CROSSOVER_NOTE.md) (2026-05-13)
   now carries an in-vitro + in-the-wild receipt pair into the gravity
   frame. The traceability claim becomes the public-facing through-line
   that links the two substrates, and it needs a doc to live in.

v2 is the spine. It does not rewrite v1.

## Relation to v1

| Surface | v1 owns | v2 owns |
| --- | --- | --- |
| Empirical phase ledger (Phases 1–7) | yes (authoritative) | retroactive traceability labels only |
| Phase 8 public artifact (`mesa.html`) | yes (authoritative) | post-v2 ratchet candidate |
| Formal Separability Appendix (theorem prose) | yes | counterexample slate (Phase 6.5) |
| Traceability taxonomy | — | Phase 0.5 |
| Bayes-net / NN / Sundog comparison | — | Phase 0.6 |
| Signature-provenance manifest + leakage audit | — | Phase 1' retrofit |
| Traceability-labeled envelope cells | — | Phase 7 relabel pass |
| Claim ratchet writeup (post-v2) | — | Phase 8' |

v1 phase numbering is preserved. v2 phases are named with primes (`1'`,
`7'`, `8'`) when they retrofit, and with fractional indices (`0.5`,
`0.6`, `6.5`) when they are net-new.

## Ratification decision, 2026-05-13

Status: **ratified as a sister/spine doc for future integration or fork**.
It is not a successor to v1 and does not modify the authority of the v1
phase ledger, Phase 8 v1 `mesa.html`, or the currently running Phase 6
v3.8 work.

Ratification locks these calls:

1. **Counterexample namespace:** Phase 6.5 counterexample scripts live
   in a sibling `scripts/mesa-counterexample-*` namespace so they cannot
   be confused with v1 canonical runs.
2. **Phase 7' scope:** relabel only the existing v1 22-cell envelope.
   Large-tier extension is not a blocker; if it happens later, it gets
   labels after the v1 map is already ratified.
3. **Phase 0.6 home:** the Bayes-net / NN / Sundog comparison lives
   here as the canonical v2 prose. It may be mirrored into
   `claims-and-scope.md` during Phase 8' or presentation cleanup, but
   this doc owns the source language.
4. **Formal Separability Appendix:** keep the theorem prose in
   `SUNDOG_V_MESA.md` for v1 continuity. v2 owns the runnable
   counterexample slate and cross-links back to the appendix when
   Phase 6.5 lands.

Cheap-first sequencing is ratified:

1. Phase 0.5 and Phase 0.6 writing.
2. Phase 7' relabeling on existing envelope artifacts.
3. Phase 1' signature-provenance audit retrofit.
4. Phase 6.5 counterexample slate.
5. Phase 8' claim-ratchet writeup.

## What v2 is honest about, and what it isn't

**Honest:**

- A traceability taxonomy that distinguishes signature / controller /
  policy / harness layers and names what fails at each.
- A short Bayes-net / transformer / Sundog comparison that answers the
  two most common institutional dismissals directly.
- A signature-path audit that produces a per-run provenance manifest and
  a static leakage test suite, retrofitted onto v1's Phase 1 artifacts.
- A counterexample environment slate that ships the Formal Separability
  appendix's failure cases as data, not as prose.
- A relabeling pass on v1's Phase 2–7 deliverables that surfaces the
  traceability reading already implicit in those artifacts.

**Reach (do not claim, even on a strong v2 result):**

- "Sundog solves interpretability."
- "Sundog is immune to mesa-optimization at any scale."
- "The signature path is uncorruptible."
- "Inner alignment is solved for signature-trained agents."
- v2 extends to LLM-scale or foundation-model behavior. It does not.

The v2 ratchet, like v1's, is bounded: it earns a traceability claim
inside the shadow-field navigation family across the mapped
(capacity, selection-pressure) pocket, and explicitly does not earn
anything outside it.

## Ratified hook language (v2)

Safe hook:

> Sundog's traceability advantage is strongest at the signature and
> harness layer, weakest inside learned policies. v2 measures where on
> that gradient the gravity claim survives.

Short version:

> The field is auditable. The agent might still imagine its own.

Avoid:

- "Sundog is traceable end-to-end."
- "Learned signature-trained agents are interpretable by construction."
- "External signature implies external objective."

## v2 Spine Status (2026-05-18)

A single-glance status surface for the v2 spine. Per-phase status
lines below are authoritative; this table mirrors them and is
also mirrored in
[`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md) §Implementation Status →
MESAv2 Spine — Status Roll-Up.

| v2 Phase | Status | Last touched | Notes |
| --- | --- | --- | --- |
| 0.5 — Traceability Claim Boundary | ✅ complete (2026-05-13) | 2026-05-13 | 4-row layer taxonomy; mirrored to `docs/presentation/claims-and-scope.md` |
| 0.6 — Bayes-Net / NN / Sundog Comparison | ✅ complete (2026-05-13, stamped 2026-05-18) | 2026-05-18 | Comparison table in-doc; exit-criterion self-check landed |
| 1' — Reference Signature Path Audit | 📝 spec filed (2026-05-18); implementation queued | 2026-05-18 | Spec at [`mesa/PHASE1_PRIME_SPEC.md`](mesa/PHASE1_PRIME_SPEC.md); 4 leakage tests pre-registered; 3 outcome branches; manifest + audit + result note unbuilt |
| 6.5 — Formal Separability Counterexample Pass | ⏳ not started | 2026-05-13 | ~1–2 weeks Small-tier; sequenced after 1'; reserves `sensor-hacked` / `geometry-hacked` labels |
| 7' — Envelope Cell Relabeling | ✅ complete + 2× extended | 2026-05-18 | v1 retrofit 2026-05-13; **v2 envelope extension (Phase 7 v2)** and **v3 vocabulary extension (Phase 7 v3, new labels `field-coupled, under-budget` and `bootstrap-collapse`)** both 2026-05-18 |
| 8' — Traceability Claim Ratchet | ⏳ not started | 2026-05-13 | Post-v2 writeup; depends on 0.5/0.6/1'/6.5/7' |

Next-step decision-leverage (per the 2026-05-18 v2-spine survey):
**Phase 1' is the highest-leverage substantive forward motion** —
unblocked, cheap (~1–2 days, mechanical), and its outcome forks
Phase 8' between the three pre-drafted claim-language branches
(clean / partial-collapse / leakage-found). Phase 6.5 is the larger
v2 lift but is sequenced after 1' per this doc's "Sequencing &
sizing" section.

## v2 Phases

### Phase 0.5 — Traceability Claim Boundary (net-new)

**Goal:** name what "traceability" means before the rest of v2 uses the
word.

**Deliverable:** a four-row taxonomy, anchored in this doc and mirrored
into `docs/presentation/claims-and-scope.md`.

| Layer | Traceability claim | What can fail |
| --- | --- | --- |
| External signature `S(x)` | Auditable if derived from environment geometry | Sensor spoofing, bad geometry, cheap environment edits |
| Hand-coded controller | Auditable because `SCAN`/`SEEK`/`TRACK` is explicit | Bad thresholds, brittle coupling, hidden implementation leakage |
| Learned policy | Not automatically auditable | Internal proxy formation, shortcut learning, reward-like representation |
| Evaluation harness | Auditable if logs, seeds, probes, and interventions are reproducible | Metric overfitting, probe incompleteness, logging leakage |

**Exit criterion:** the roadmap states, in writing, that Sundog's
traceability advantage is strongest at the signature/harness layer and
weakest inside learned policies. Phase 0.6 and Phase 1' both depend on
this taxonomy.

**Status:** **complete (2026-05-13).** Source taxonomy lives in this
doc; mirror landed in
[`docs/presentation/claims-and-scope.md`](presentation/claims-and-scope.md)
§Traceability claim taxonomy (Mesa v2), inserted between the §Earned
envelope language and §Required boundary text subsections of §The
Gravity Frame. Exit-criterion sentence appears verbatim in both
locations.

### Phase 0.6 — Bayes-Net / Neural-Net / Sundog Comparison (net-new)

**Goal:** answer the two most common outside-academic dismissals in one
short appendix-style section.

**Claim to test in prose:**

> Bayesian graphical models are structurally inspectable but can still
> be computationally hard, causally wrong, or unscalable. Neural
> policies are scalable but internally opaque. Sundog attempts a third
> move: keep the objective-defining signal external and geometrically
> auditable, then test whether learned policies preserve or corrupt
> that coupling.

**Deliverable:** comparison table inline in this doc.

| Family | What is traceable | What remains hard |
| --- | --- | --- |
| Bayes nets | Variables, edges, conditional dependencies | Inference scale, learned structure, hidden confounders |
| Transformers / deep RL | Training objective and behavior under probes | Internal objective, circuits, feature semantics |
| Hand-coded Sundog | Signal path, scan/seek/track logic, intervention response | Operating envelope, sensor validity |
| Learned Sundog | External signature path and training regime | Whether the policy internalizes a proxy |

**Exit criterion:** the roadmap can answer a reviewer who says "this is
just another interpretability problem" and a reviewer who says "Bayes
nets already solved traceability," using only the table above plus the
v1 envelope receipts. Not a literature survey.

**Status:** **complete (2026-05-13, stamped 2026-05-18).** The
comparison table above is the deliverable and is in-doc. Exit-criterion
self-check: the table separates Bayes nets (auditable structure,
inference-scale hard) from transformer/deep-RL (auditable training
signal, internal objective opaque) from hand-coded Sundog (auditable
SCAN/SEEK/TRACK logic, envelope-bounded) from learned Sundog
(auditable external signature path, internal proxy formation under
test by Phases 3/4/6/7). A reviewer raising "just another
interpretability problem" can be pointed at row 4's "What remains
hard" column; a reviewer raising "Bayes nets already solved
traceability" can be pointed at row 1's same column. The table plus
the v1 envelope receipts (PHASE3/4/5/6/7 result notes) satisfy the
exit criterion without a literature survey. No amendments since
initial filing.

### Phase 1' — Reference Signature Path Audit (retrofit on v1 Phase 1)

**Goal:** prove `S(x)` is externally defined, reproducible, separable
from reward, and inspectable before any learning enters.

**Deliverables:**

1. **Signature provenance manifest** emitted per run, alongside the
   existing v1 manifest:

   ```json
   {
     "x_goal": "... privileged, hidden from agent ...",
     "signature_function": "gaussian_shadow_field_v1",
     "sensor_tier": "local_probe_field",
     "probe_offsets": "...",
     "reward_function": "none / matched_reward_v1",
     "agent_observation": "...",
     "agent_forbidden_channels": ["x_goal", "privileged_gradient", "reward_edit_log"]
   }
   ```

2. **Static leakage test suite** at
   `scripts/mesa-signature-provenance-audit.mjs`. Each test passes or
   fails the build:

   - No learned or hand-coded non-Oracle policy receives `x_goal`.
   - `S(x)` and `R(s, a)` are implemented through separate accessors
     with no shared mutable state.
   - Logging does not feed back into policy input.
   - Proxy probes can alter the reward or observation channel without
     altering true environment geometry, and vice versa.

3. **Result note:** `docs/mesa/V2_SIGNATURE_PATH_AUDIT.md` summarizing
   manifest coverage across v1's existing Small and Medium runs and
   listing any leakage findings.

**Exit criterion:** the signature path can be inspected independently
of agent performance. Any unexpected coupling found by the audit is
filed as a v2 finding and routed back to v1's PHASE1_HC_BASELINE before
the audit is declared green.

**Status:** **spec filed (2026-05-18), amended to v1.1 (2026-05-18)**
at [`mesa/PHASE1_PRIME_SPEC.md`](mesa/PHASE1_PRIME_SPEC.md). Four
pre-registered leakage tests (LT1–LT4) with pass/fail verdicts,
manifest schema with `leakage_audit_verdict` per-test field, three
pre-registered outcomes (clean / partial-collapse / leakage-found)
that route Phase 8' between its three drafted claim-language
branches. Compute envelope: ~1–2 days, mechanical, no env runs.
**v1.1 amendment (2026-05-18)** reconciled the four tests against
the reference implementation's intentional benign coupling:
goal-centered env design has `S(x)` peak at `this.xGoal` and reward
as distance-to-`this.xGoal`, so the geometry baseline is a
*documented* shared field, not leakage. LT1 scoped to non-privileged
tiers; LT2 allowlist extended with `this.xGoal`; LT3 extended to
check the labeled-signature reward channel doesn't feed back into
observation; LT4 rewritten with an explicit pre-registered effect
table that encodes the geometry-affects-both case as expected;
§6 manifest dependency corrected. Manifest emitter, audit script,
and result note are all unbuilt; spec ratification is the v1 gate.

### Phase 6.5 — Formal Separability Counterexample Pass (net-new)

**Goal:** construct the exact cases where Sundog's traceability
advantage fails, as runnable environments instead of prose. The
Appendix A theorem currently lists four collapse conditions; v2 ships
them as data.

**Counterexample environments:**

1. **Policy-dependent signature.** `S` includes action history.
   Predicted result: collapses into reward-like feedback. Learned
   policies should look behaviorally indistinguishable from L-Reward
   under proxy splits.
2. **Cheap sensor spoof.** Agent can alter measured `S_obs` without
   changing true `S(x)`. Predicted result: signature tracking becomes
   reward hacking. Phase 4-style sensor-edit interventions become
   "free" for the learned policy.
3. **Cheap geometry edit.** Agent can cheaply alter the environment
   state that generates the signature. Predicted result: field
   manipulation becomes trivial; the gravity claim's mode (1)
   surfaces inside the mesa harness instead of being deferred to
   spacecraft / conservation-law candidates.
4. **Decompilable scalar signature.** `S(x)` reducible to an obvious
   scalar shortcut. Predicted result: learned policy internalizes the
   shortcut; Phase 6-style representation probes find it explicitly.

**Counterexample table (delivered as part of this phase):**

| Counterexample | Separation should hold? | Expected failure mode |
| --- | --- | --- |
| Clean geometric `S(x)` (v1 canonical) | Yes, bounded | Normal operating envelope |
| Policy-dependent `S` | No | Reward-in-costume |
| Cheap sensor | No | Sensor hacking |
| Cheap geometry | No | Field manipulation |
| Decompilable scalar | Weak / no | Internal proxy |

**Deliverables:**

- `docs/mesa/V2_COUNTEREXAMPLE_SLATE.md` (spec + results template).
- Four environment extensions to the shadow-field harness, each
  flagged so they cannot be confused with the v1 canonical at run time.
- Per-counterexample matched runs of HC-Signature + L-Signature +
  L-Reward at Small tier, reported alongside v1 canonical for
  side-by-side legibility.

**Exit criterion:** the theorem appendix is no longer prose-only.
Each of the four collapse conditions has at least one runnable
environment and at least one matched-family result row. The appendix
in `SUNDOG_V_MESA.md` is updated to point at the slate.

### Phase 7' — Envelope Cell Relabeling (retrofit on v1 Phase 7 v1)

**Goal:** add traceability labels to the 22 already-classified envelope
cells so the operating-envelope map shows not just where the system
succeeds but what it was coupled to when it succeeded.

**Label set:**

- `field-coupled`
- `reward-coupled`
- `observation-coupled`
- `sensor-hacked`
- `geometry-hacked`
- `ambiguous`
- `undertrained`
- `probe-insufficient`

**Deliverable:** new column in `cell-class-map.csv` and a
mirror table in `PHASE7_RESULTS.md`. Labels are derived from existing
Phase 3 probe responses and Phase 4 intervention-response matrices;
v2 does not retrain.

**Exit criterion:** every v1 envelope cell carries one traceability
label or an explicit `probe-insufficient` flag. The Phase 8 v1
`mesa.html` chip-grid tooltips can be extended to surface the label
in a v2 follow-up.

**Status:** **complete (2026-05-13).** Labels live in a sibling file
[`results/mesa/operating-envelope/cell-traceability-labels.csv`](../results/mesa/operating-envelope/cell-traceability-labels.csv)
keyed by `policy_id`, not as a column edit on `cell-class-map.csv`,
because the v1 Phase 7 harness regenerates that file. The v1 pipeline
is untouched. Mirror table landed in
[`mesa/PHASE7_RESULTS.md`](mesa/PHASE7_RESULTS.md) §11 (Phase 7'
retrofit). Earned reading: 9 `field-coupled` (8 hold + 1 fragile
probe-marginal), 7 `reward-coupled`, 4 `undertrained`, 2 `ambiguous`.
The remaining four labels (`observation-coupled`, `sensor-hacked`,
`geometry-hacked`, `probe-insufficient`) have count 0 in the v1
envelope — `sensor-hacked` and `geometry-hacked` are reserved for
Phase 6.5 counterexample cells; the other two are probe-resolution
slots.

**v2 envelope extension (Phase 7 v2, 2026-05-18).** A separate
v2 result note at
[`mesa/PHASE7_V2_RESULTS.md`](mesa/PHASE7_V2_RESULTS.md) labels the
six Large-tier cliff-subset cells under the same traceability vocab
introduced here. Profile-based labels only — v2 did not run the Phase 3
probe-slate or Phase 4 intervention battery on Large, so
`reward-coupled (profile)` is used for the L-Mixed trough cells. The
`λ=1.00` Large cell adds the first probe-confirmed `undertrained` row
caused by a *bootstrap failure* rather than a deprecated objective —
suggesting the `undertrained` label spans at least two distinct
mechanisms (deprecated-objective vs critic-target-collapse) that v3
labels should disambiguate.

**v3 vocabulary extension (Phase 7 v3, 2026-05-18).** The Phase 7 v3
intervention battery at [`mesa/PHASE7_V3_RESULTS.md`](mesa/PHASE7_V3_RESULTS.md)
introduces two new traceability labels:

- **`field-coupled, under-budget`** — the policy reads the external
  signature (probe-confirmed via healthy signature-sensor response)
  and is *not* basin-internalized (`old_basin_pref` well below the
  Medium collapse floor), but does not navigate to the goal
  effectively at the tested mixture weight. Distinct from
  `field-coupled` (effective navigation) and `reward-coupled`
  (lost signature response, basin-attracted). The Large U-trough
  cells (`λ ∈ {0.95, 0.97}`) carry this label probe-confirmed; v2's
  `reward-coupled (profile)` qualifier was falsified by v3.
- **`bootstrap-collapse`** — the policy is degenerate (zero
  action_response_L2 across every channel) but consistently
  terminates at the old `x_false` direction (`old_basin_pref ≈ 7`).
  Distinct from `undertrained` (which would have shown zero
  `old_basin_pref` because the trained policy never reached the
  basin) and from `reward-coupled` (which retains intervention
  responsiveness while heading to the basin). The `λ=1.00` Large
  cell carries this label probe-confirmed; v2's `undertrained` was
  partially falsified by v3.

These two labels extend the v2 traceability vocabulary (§11 above);
they are not retroactive on the v1 22-cell envelope, but the v2
sibling at PHASE7_V2_RESULTS.md will be amended to use the v3 labels
where they apply.

### Phase 8' — Traceability Claim Ratchet (post-v2 writeup)

**Goal:** replace "Sundog solves traceability" or any broad equivalent
language with earned claim language whose strength is bounded by the
v2 phases above and the v1 envelope.

**Candidate claim language by v2 outcome:**

> *If Phase 1' is clean, Phase 6.5 shows the predicted collapses, and
> Phase 7' relabels cleanly:* In the tested shadow-field navigation
> family, the external geometric signature remained more causally
> traceable than matched reward channels, and learned signature-trained
> controllers retained field-coupling across a bounded capacity and
> selection-pressure pocket. The four documented collapse conditions
> from the Formal Separability appendix were observed to produce the
> predicted failure modes when constructed.
>
> *If Phase 6.5 shows partial collapse outside the prediction:* Sundog's
> external signature path was traceable, but learned controllers
> retained that advantage only under imitation or low-capacity regimes.
> Specific collapse conditions matched the appendix prediction; others
> diverged and are filed as open boundaries.
>
> *If Phase 1' surfaces real leakage in v1 artifacts:* The geometric
> signature was externally auditable in principle, but the v1
> implementation had specific coupling points that the audit recovered.
> The audit becomes the program's primary traceability deliverable; the
> behavioral envelope is held pending re-derivation against the
> patched implementation.

**Do not claim, under any v2 outcome:**

- "Sundog is immune to mesa-optimization."
- "Reward hacking does not occur in signature-trained agents."
- "Inner alignment is solved."
- "Geometry solves interpretability."

**Exit criterion:** `PROMO_HIGHLIGHTS.md` §The Gravity Claim,
`claims-and-scope.md` §The Gravity Frame, and `SUNDOG_V_GRAVITY.md`
Candidates 1 & 2 are updated to use the strongest v2 claim earned and
no stronger one. `mesa.html` v2 surfaces the traceability label per
cell.

## Retroactive traceability labels for v1 phases

These do not need new runs; they are framing deliverables that surface
the traceability reading already implicit in v1 artifacts. Each lands as
a short note on the corresponding `PHASEn_RESULTS.md`, not as a v2
sub-phase.

| v1 phase | v2 retroactive deliverable | Already in artifacts? |
| --- | --- | --- |
| Phase 2 | Trace-hook logging contract (`obs`, `S(x)`, `S_obs`, `R`, `dist_to_goal_eval_only`, `action`, `logits`, `hidden`, `phase_label`) | Mostly — needs one-page contract doc |
| Phase 3 | `proxy_collapse_index = reward_coupling + observation_coupling − field_coupling` reported per family / tier | Inputs exist; scalar metric is new |
| Phase 4 | Per-family causal-authority graph (HC-Signature / L-Signature-clean / L-Signature-collapsed / L-Reward) | Reports exist; graph rendering is new |
| Phase 5 | Traceability degradation curve: capacity × selection-pressure cells where L-Signature stops behaving like HC-Signature and starts behaving like L-Reward | Cliff localization already supports this; relabel pass only |
| Phase 6 | Probe targets named in the decompilable-signature register: hidden-goal estimate, distance-to-goal estimate, reward-correlated scalar, `S(x)` estimate, `∇S(x)` estimate, decoy-field attraction, dwell/success classifier, action-history proxy | Some probes exist; register is new |
| Phase 7 | Envelope cell labels (Phase 7' above) | Inputs exist; label column is new |

## Sequencing & sizing

v2 sequencing is intentionally cheap-to-expensive:

1. **Phase 0.5 + 0.6** — writing only, in this doc. Ratification-gated,
   not implementation-gated. ~half a day of editing.
2. **Phase 7'** — relabeling pass against existing CSVs. ~half a day,
   no new training, no new probes.
3. **Phase 1'** — manifest emission is mechanical; static leakage audit
   is small. ~1–2 days. May surface findings that route back to v1.
4. **Phase 6.5** — counterexample slate. ~1–2 weeks at Small tier with
   four new environments and matched-family runs. The biggest v2 lift.
5. **Phase 8'** — writeup ratchet, post-v2. Gated on the four phases
   above.

Retroactive labels for Phase 2/3/4/5/6 can be filed in parallel with
the above, in any order, by any contributor.

## What v2 earns the gravity ledger

- A traceability spine that protects the gravity claim from
  "Sundog solves interpretability" drift before the public framing
  picks it up.
- A direct institutional answer to "isn't this just interpretability?"
  and "didn't Bayes nets already solve traceability?", grounded in v1's
  earned receipts rather than in survey prose.
- A theorem appendix that ships its failure cases as data, raising
  Appendix A from a sanitized condition statement to an experimentally
  bounded one.
- A relabeling of v1's operating envelope that surfaces *what the
  system was coupled to when it succeeded*, not just whether it
  succeeded.
- A v2-versioned writeup target for Phase 8 that ratchets the public
  claim language with the spine in place.

## Ratification resolutions

1. Phase 6.5 counterexample environments live in the sibling
   `scripts/mesa-counterexample-*` family.
2. Phase 7' relabels the v1 22-cell map first and does not block on a
   v2 Large-tier extension.
3. Phase 0.6 belongs here as source prose. `claims-and-scope.md` can
   mirror it later if Phase 8' needs a public-facing excerpt.
4. Formal Separability theorem prose stays in `SUNDOG_V_MESA.md` for v1
   continuity. v2 owns the runnable counterexample slate and links back
   to the theorem.

## References

- [`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md) — v1 roadmap (authoritative
  for empirical artifacts; Appendix A holds the theorem)
- [`mesa/MESAv2.md`](mesa/MESAv2.md) — brainstorm of record that
  seeded this doc
- [`MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md) — mesa↔geometry
  crossover, including the two-substrate framing referenced above
- [`SUNDOG_V_GRAVITY.md`](SUNDOG_V_GRAVITY.md) — gravity staging
  ledger (Candidates 1 & 2 are v2's downstream surface)
- [`SUNDOG_V_PERCEPTION.md`](SUNDOG_V_PERCEPTION.md) — sister roadmap
  using the same Phase 5→6 structural-transition pattern
