# Sibling Handoff — the ∞-Dimensional Attractor Tail (fresh NSE × Chat-v2 slate)

> 2026-07-02, drafted by the webdev/licensing session for a **fresh sibling agent**.
> You are being spun up raw: you have the persistent memory directory (read
> `project_sundog_navierstokes_ledger`, `project_sundog_chatv2_bodyresist_lane`,
> `project_sundog_suffstat_order_slate`, `project_sundog_lean_formalization`,
> `project_sundog_orderrelative_lane` before anything else) but **none** of the
> session context that produced this document. Everything else you need, you grep.
> Repos: `Dev\sundog` (site + research docs + receipts) and `Dev\sundogcert`
> (public Lean). This handoff lives in `docs/chatv2/` — no-publish, internal.

**Your assignment:** assemble a fresh, ranked slate of conjectures/hypotheses that
could crack the **∞-dimensional NSE-attractor analogue** of the body-resistance
question — the one thread of the NSE re-aim paragraph that remains *"separately
hypothesized, cited but not applied."* Slate only. You run nothing, commit nothing,
deploy nothing, and touch no public page. The deliverable is one document (spec in
§6) that the owner reviews.

---

## 1. The target, stated precisely

`navierstokes.html` (re-aim paragraph, ~line 857) now ends:

> "…so the substrate claim stays with the toy result at its licensed scope, and
> the ∞-dimensional NSE-attractor analogue likewise remains separately
> *hypothesized*, cited but not applied."

The question behind that clause: does a dissipative PDE system — the NSE global
attractor, or an honest finite proxy of it — carry a **decision-relevant functional
that no compact shadow can determine**, the way GPT-2's stack-top defeats every
count statistic? I.e., does the *decision-observable-but-state-unobservable* split
(body-resistance regime-2) survive the trip from a 768-dim residual stream to an
∞-dimensional phase space whose attractor is finite-dimensional?

Two distinct forms, both in scope for your slate:

- **The crossover / input-undecodability form** (what actually ran on the LLM side):
  an impossibility theorem about *labels and statistics* (some functional is not a
  function of any window statistic) paired with a measured carrier that reads it
  exactly where the statistic collapses. This form is now fully instantiated once —
  machine-checked pair + measured crossover + fully-receipted negative above it.
- **The maintained-ledger / shadow form** (the NSE page's original design, never
  built): the shadow is not a passive projection but an *actively maintained
  ledger* of the trajectory, and the question is what the ledger provably cannot
  carry while control stays sufficient. Nothing on this form has ever run.

"Cited but not applied" is literal: the one existing document is
`docs/proof/PDE_DETERMINING_MODES_POSTULATE1.md` (2026-05-28, **drafted,
unreviewed**, promotion explicitly blocked) — a *translation* of the
determining-modes / synchronization literature through Postulate 1's
control-sufficiency predicate. Its key move: state-reconstruction sufficiency ⟹
control sufficiency, but control sufficiency only needs common optimal actions on
signature fibers — the two can separate for objectives constant on state-distinct
fibers. Its pre-registered vacuity negative is quoted in the doc. Its claim
boundary (no Millennium claim, no new determining-modes theorem, no smaller mode
count) **binds your slate too**. Citations live in
`docs/NAVIERSTOKES_LITPASS_MEMO.md`; the parent roadmap is
`docs/COARSE_GRAINING_PROOF_ROADMAP.md`.

## 2. What is already banked (orientation — verify via receipts, not prose)

**NSE side.** C1 (Kolmogorov flow): a certified Reading-2 witness — state-insufficient
AND control-sufficient — at two Grashof regimes (G=200 d=32, G=300 d=18, ~942k witness
pairs at eps_K≈0.0664), but **marginal in every physical norm**; C2 (Sabra shells)
paused at a numerical wall. The re-aim paragraph was recoverability-corrected
2026-06-29: the pivot's measured reason is that all three physics substrates were
*recoverable from compact shadows* (NSE C1 FVE~0.99; Mesa net.7 256-wide but
eff-dim~2, rebuilt ~99% by its shadow; Sabra eff-rank~1.7). Docs:
`docs/SUNDOG_V_NAVIERSTOKES.md` (spine), `docs/proof/PDE_C1_*.md` (the full C1
chain — separation statement, twin-state certificate, fiber protocol, MZ energy
budget), `docs/proof/PDE_C2_*.md`, receipts under `results/proof/`.

**Chat-v2 side.** R1 licensed at honest toy scope; R2 gate NOT STARTED and it stays
that way — your slate does not reopen it by itself. The intersection arc closed
end-to-end 2026-07-02: the intersection {input-undecodable ∧ model-computed ∧
high-dim} is **long-range order-dependent state**; positive = the state-crossover
(counts 0.965→0.770 across the ambiguity slice while GPT-2's residual holds
0.926→0.931); negative = six label families / two gate designs / two model scales
all failed the matched-baseline bank gate, apparatus validating on known-answer
controls at every rung. Docs: `docs/chatv2/` (LANE_CHARTER, PROMOTE_GATE,
R2_INTERSECTION_HYPOTHESES.md, H1_V3_* series, R2_* series,
PROMO_WEBDEV_HANDOFF_STATE_CROSSOVER.md); receipts + manifests under
`results/chatv2/` — **check mtimes; chatv2 prose goes stale within days**.

**The bridge that makes this one problem.** The Order-Relative Resolution Law
(`sundogcert`, public): `Resolves k t ↔ ord t ≤ k`, determine/resist =
finite/infinite order, **eight grounded filtrations**, schema-not-scalar guard,
composition law with both walls, approximation dimension. The 8th filtration is the
surface-window pair (`SurfaceBag.lean`, `SurfaceBagGraded.lean`: bag determines
depth, never the stack-top, at every window order — σ_surface = ∞, axiom-lean
`[propext, Quot.sound]`). The σ-order schema (`docs/SUFFICIENT_STAT_ORDER_SLATE.md`)
and the σ_read/σ_write split (`docs/SUNDOG_V_ORDERRELATIVE.md`) are the live
formal vocabulary. The measure-theoretic determine/resist axis already in-tree —
`ShadowDecayGeneral`/`ShadowDecayCauchy`/`ShadowDecayLattice` (resist ⟺ charFun→0,
determine ⟺ finite mean, Cauchy the proven separator) — is the closest existing
Lean to a PDE-shaped shadow and is probably underused.

## 3. Binding lessons — design around these or the slate is dead on arrival

1. **Recoverability is the axis, not dimension** (Phase-0 Amendment 1 + the net.7
   trap). Any gate keyed on raw d_dec/rank will be matched by a control twin. Gate
   on leak/recoverability and objective-driven carry.
2. **Absolute input-undecodability at bank scale on natural data looks
   unattainable** — it failed at SIX families; even H2's hard-slice counts sat at
   0.770, not chance. The lane's real positives were **relative margins**
   (crossover form: model ≥ registered surface statistic + δ, ≥ random-init floor
   + δ, minus order-shuffle). Prefer crossover-form gates; if you propose an
   absolute gate, say why yours survives where six died.
3. **The ambiguity-slice diagnosis:** state is surface-undecodable only on the
   count-ambiguous slice, and natural distributions make that slice thin and
   *itself skewed* (V3-0b: balance killed 366→29 axes). Slices must be part of the
   design, with liveness controls (a bag-determined axis must read ≈1.0 on the
   slice) and the caution that ambiguity ≠ unpredictability.
4. **UNLEARNED / numerically-walled ≠ resistance negative.** C2 Sabra sits at a
   numerical wall; the arity-3 grok wall was a learnability ceiling. Type the
   verdicts so "couldn't run/train" never files as "no resistance."
5. **Pre-registration discipline:** typed verdict tokens, kill conditions, gates =
   exact commands, controls with known answers, no verdict-shopping (no third
   same-day re-registration), falsifier-fenced throughout. Reconcile every derived
   number across sections before presenting (house rule).
6. **The determining-modes vacuity negative** (§1) is already registered — your
   slate entries must each say whether they survive it: if control sufficiency
   collapses to state sufficiency under the standard data-assimilation gauge in
   your regime, record, do not rescue.

## 4. Grep map

`Dev\sundog`: `docs/SUNDOG_V_NAVIERSTOKES.md`, `docs/proof/PDE_*` (esp.
`PDE_DETERMINING_MODES_POSTULATE1.md`, `PDE_C1_SEPARATION_STATEMENT.md`,
`PDE_C1_TWIN_STATE_CERTIFICATE.md`, `PDE_C2_SHELL_*`),
`docs/NAVIERSTOKES_LITPASS_MEMO.md`, `docs/COARSE_GRAINING_PROOF_ROADMAP.md`,
`docs/CROSS_SUBSTRATE_NOTES.md`, `docs/chatv2/*`, `docs/SUFFICIENT_STAT_ORDER_SLATE.md`,
`docs/SUNDOG_V_ORDERRELATIVE.md`, `results/proof/`, `results/chatv2/`,
`scripts/chatv2_*.py` (the harness family you'd inherit), `navierstokes.html`.
Seed patterns: `rg -i "determining|attractor|inertial manifold|nudging|data.assimilation|Grashof|witness pair|recoverab|eps_K|signature fiber"`.

`Dev\sundogcert`: `Sundogcert/OrderRelative*.lean`, `SurfaceBag*.lean`,
`ShadowDecay*.lean`, `AxiomAudit.lean` (the guard style any Lean candidate must
meet), `README.md` (the new Order-Relative section = the public register your
wording must stay consistent with). **Do not touch owner WIP**: uncommitted
working-tree changes (root/AxiomAudit wiring, `PercivalAuditPay.lean`) and
`AnalyticGate`/`AbstractionCert`.

## 5. Known hooks — optional, non-binding; fresh eyes are the point

Take or leave; if you use one, sharpen it past what's written here.

- **A determining-modes filtration for the σ schema:** ord(functional) = least
  number of low modes (Galerkin window) whose asymptotic knowledge determines it;
  determining-modes theorems put energy-type functionals at finite order; the
  conjecture slot is a decision-relevant functional with **no finite determining
  window** (σ_modes = ∞) — the attractor's "stack-top." A finite toy instance
  (shells, or the C1 discretization itself) might be Lean-anchorable in the
  SurfaceBagGraded pattern. Tier any Lean candidate honestly
  (FORMALIZABLE vs DAYDREAM — the U-4 mis-tier is the cautionary tale).
- **AOT nudging as the maintained ledger:** data-assimilation/nudging *is* an
  actively maintained shadow, and its literature is the determine side. The
  resistance question is which functionals survive nudging recovery at a fixed
  observation budget, and whether the failure is sharp (regime-2) or graded.
- **The crossover transplant:** run the state-crossover *form* on the PDE side —
  registered surface statistic = mode-window/observable history; carrier = the C1
  controller (or a small learned surrogate); evaluate on the ambiguous slice
  (near regime transitions / heteroclinic passages, where truncations provably
  collapse). Relative margins, matched floors, liveness controls.
- **The charFun axis as the PDE-shadow separator:** ShadowDecay* already proves a
  determine/resist dichotomy for averaged shadows; nobody has asked what it says
  about time-averaged attractor observables.

## 6. Deliverable

One document: `docs/chatv2/NSE_ATTRACTOR_TAIL_HYPOTHESES.md` (stays no-publish).
4–8 hypotheses, **ranked**, cheapest-decisive-first within rank. Per entry:

- one-sentence claim (the falsifiable form, condition included);
- mechanism + which bridge it uses (σ-order / determining-modes / nudging /
  charFun / crossover-form / other);
- the cheapest decisive probe (CPU-tier preferred; GTX-1080 fp16 is available;
  the H200 is **not** to be boarded on this lane's account; agent-launched
  background jobs die at session teardown — anything multi-hour is owner-run);
- pre-registered gates with typed verdict tokens (suggest `AT1_…`, `AT2_…`) and
  an explicit kill condition;
- the standing falsifier it answers to, including §3.6's vacuity negative;
- Lean-anchor candidate (if any) with an honest FORMALIZABLE/DAYDREAM tier;
- what it does **not** claim.

Close the slate with: (a) a do-not-say list inheriting the state-crossover
keep-true box (no world-model/"understanding" language, model claims only relative
to matched baselines, negatives said plainly) plus the determining-modes claim
boundary (no Millennium, no new mode-count theorem); (b) the open owner decisions
you are explicitly leaving to the owner. PROMOTE_GATE and the NSE page are not
yours to edit. When the slate is drafted: stop and present. Nothing runs without
the owner's word.

---

*Handoff v1. You have the memory; trust the receipts over the prose, check
`results/*/` mtimes first, and write the slate you would want to inherit.*
