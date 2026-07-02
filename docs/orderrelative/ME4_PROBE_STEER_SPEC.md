# ME-4 — the Probe-Steer Gap on GPT-2's Stack-Top (spec)

*Pre-registration. Parent: [`ME_QUADRANT_HYPOTHESES_SLATE.md`](ME_QUADRANT_HYPOTHESES_SLATE.md)
entry ME-4, upgraded by ME-1's census to the **admission test for the E∧¬M model-state
cell**. Chat-v2 lane discipline inherited in full (no R2 / promotion / world-model
language; existence-tier claims only). Status: **SPEC PINNED 2026-07-02, NOT RUN.
Owner-cadence.***

## The question

GPT-2's bracket stack-top is linearly **readable** in the residual stream (banked:
0.931 at count-ambiguous positions, `chatv2_h2_stacktop_probe.py`, 2026-07-01). Is it
**writable at the read address**? Patch the residual along the read direction (or
transplant a donor activation) and test whether the model's *closing-bracket behavior*
follows the patched top. M≠E on a real substrate: the read is banked; this is the write.

**Why the verbs now match (the admission-test duty).** ME-1's census rejected the raw
regime-2 mapping because "steer an objective through the shadow" is not the quadrant's
write-verb. Here the verbs are literal: the residual vector at (position, layer) is a
**node the experimenter owns** in the compute graph; patching is `do(node = value)` — a
node-write in ME-2's exact sense. The condition is "the model's effective stack-top is
τ′," read by the probe, enforced (or not) through the owned node. If the write fails
while the read succeeds, the model's state is **edge/function-typed relative to the
residual node** — the probe reads a computed shadow, not a register — and the census
places model-states in M∧¬E.

**The pretty symmetry, named:** at count-ambiguous positions the counterfactual is the
**one-pop swap** — behave as if the top two unclosed openers were exchanged (`([` vs
`[(`). ME-4 is the empirical `do()` of `SurfaceBagGraded`'s witness pair.

## Grounding (frozen inputs)

- Model: GPT-2-small, CPU, cached. Corpus, lexer, position extraction, count-ambiguity
  definition (≥ 2 distinct types unclosed), and probe training **reused unchanged** from
  `scripts/chatv2_h2_stacktop_probe.py` (frozen; the new work goes in a new script,
  `scripts/chatv2_me4_probe_steer.py`).
- Banked read-side numbers (the context): residual probe 0.931 on the count-ambiguous
  slice vs order-blind counts 0.770; probe layer L11; GPT-2 is a weak tracker
  (all-position residual 0.926 loses to counts 0.965).

## Design

**Query set.** Positions where the *next real token is a closing bracket* and depth ≥ 1
(the closer-preference is behaviorally live there). Primary slice: count-ambiguous
positions (target ≥ 500); secondary: all qualifying positions (≥ 1000). Deterministic
seed; per-cell JSON checkpoints (the chat-v2 teardown gotcha — resumable).

**Counterfactual target τ′.** The **alternative unclosed type** — the type at stack
position 2 (the one-pop swap). Both closers are syntactically legal given counts at
count-ambiguous positions, so the counterfactual stays on the data manifold by
construction.

**Interventions (patch at the final context position, layer L, downstream layers
recompute):**
- (a) **probe-direction:** `h ← h + α·(w_τ′ − w_τ)` (probe class-weight difference);
- (b) **difference-in-means:** `h ← h + α·(μ_τ′ − μ_τ)` (held-out class-conditional
  activation means);
- (c) **donor transplant:** replace `h` with an actual activation from a donor position
  whose true top is τ′ and whose next real token is also a closer (type-matched primary;
  type+depth-matched as diagnostic) — on-manifold by construction, the decisive family.

**Grid (pre-registered, no post-hoc widening):** layers {8, 11}; α ∈ {2, 4} for
(a)/(b); (c) has no α. Controls at every treatment cell: **random-direction** patch
(same norm) and **shuffled-donor** transplant (donor with top = τ, i.e. a null swap).
Estimated runtime 1–2 h CPU.

**Readout.** Restricted 3-way closer preference: the model's next-token logits over the
three bare single-char closer token ids `)`, `]`, `}` only (measurement caveat
registered: GPT-2 BPE merges closers into multi-char tokens like `)):` — the restricted
readout undercounts closer mass; it compares *relative* preference, which is what the
condition needs).

## Gates (all pre-registered; the decision cell is the best treatment cell, with its own
controls)

- **G0 — behavioral floor (admission):** unpatched closer-preference agrees with the
  TRUE top on ≥ 0.60 of the primary slice. Fail → `ME4_BEHAVIORAL_FLOOR` (GPT-2's
  behavior doesn't track the top well enough to read a write off it; escalation =
  code-trained model, scope-and-hold — NOT a resistance finding, the UNLEARNED-guard
  analog).
- **G1 — write validity (the read must take the write):** the probe, applied to the
  patched residual at the patch layer, reports τ′ on ≥ 0.90 of patched cases. Fail at
  every cell → `ME4_CONFOUNDED` (we never wrote the shadow; nothing is scored).
- **G2 — on-manifold:** patched continuation NLL ≤ 3× the unpatched median at the
  decision cell; random-direction control moves closer-preference by < half the
  treatment effect. Fail → `ME4_CONFOUNDED`.
- **G3 — the verdict metric:** `follow(τ′)` = fraction of patched cases whose closer
  preference switches to τ′'s closer, normalized by the unpatched true-agreement
  (`follow_rel = follow(τ′) / agree(τ)`), on the primary slice.

## Branch table

| branch | condition | census placement / consequence |
| --- | --- | --- |
| `ME4_STEERS` | some (a)/(b)/(c) cell passes G1+G2 with `follow_rel ≥ 0.75` | model-state → **M∧E** (the linear address is causally load-bearing — an owned-node condition in the compute graph); a genuine positive for the interp-orthodox reading |
| `ME4_RESISTS` | ALL families at all cells pass G1 (the write takes at the read address) + G2, yet `follow_rel ≤ 0.40` everywhere | model-state → **M∧¬E** — readable, unwritable at the read address; the model's own states join the target channel's cell; ME-2 reading: the probe reads a computed shadow, not a register |
| `ME4_PARTIAL` | between the bands, controls clean | bounded-partial recorded (HS4 discipline: the middle band is a named outcome); census row marked **graded** — the steering shortfall is a price, feeds ME-5 |
| `ME4_BEHAVIORAL_FLOOR` | G0 fails | substrate inadequate; escalation registered (code model), no census placement |
| `ME4_CONFOUNDED` | G1 or G2 fail everywhere | methodological null; fix or shelve, no census placement |

**The sharpest signature, called in advance:** transplant (c) passing G1 (probe reads
τ′ off the transplanted activation) while behavior ignores it — *the read address
accepts the write; the computation doesn't consult it.*

## Honest prior (stated before any run)

Genuinely open — the slate's highest-information entry. The interp literature pulls both
ways: activation-patching successes (ROME-line, function vectors, ITI) argue `STEERS`;
concept-erasure failures and distributed-representation results argue `RESISTS`/`PARTIAL`.
GPT-2-small's weak tracking makes `BEHAVIORAL_FLOOR` a live risk (the banked receipt's
own caveat). Rough ordering: FLOOR ≳ PARTIAL > STEERS ≈ RESISTS. Every branch is banked;
none is a failure.

## Fences

Existence-tier: one model (small, weak), one quantity (3-class stack-top), one corpus
(Python), light lexer — all inherited from the H2 receipt. No claim about
interpretability methods in general; no R2 language; `PROMOTE_GATE.md` untouched. The
census placement applies to *this* state on *this* substrate; generalization is future
rows, not this spec. Owner-cadence run; the script must be resumable (per-cell
checkpoints) and deterministic (fixed seeds; frozen probe).

## Outputs

- Script: `scripts/chatv2_me4_probe_steer.py` (new; probe script untouched).
- Receipt: `docs/orderrelative/ME4_PROBE_STEER_RESULTS.md` +
  `results/orderrelative/me4-probe-steer/summary.json`.
- RESULT block in the ME slate; census row per the branch table; memory.
