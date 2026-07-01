# Sundog Documentation

**Copyright © 2026 Stellar Aqua LLC. All rights reserved.**

Sundog-authored source, documentation, visual materials, and data compilations
are controlled by Stellar Aqua LLC unless a file clearly states otherwise. The
root [`LICENSE`](../LICENSE) is a rights notice, not a public open-source grant.
Citation metadata lives in [`CITATION.cff`](../CITATION.cff).

---

This folder is organized as a research-facing landing layer over the Sundog
code, experiments, prior theorem materials, and application bridges.

## Primary Entry Points

- [Table of contents](index.html): browsable HTML index of all documents in
  this folder.
- [Active TODO](TODO.md): consolidated operator queue for outstanding
  experiments, blockers, public-surface gates, and launch follow-ups.
- [Researcher guide](RESEARCHER_GUIDE.md): shortest path for reviewers and
  collaborators.
- [Scientific criteria](SCIENTIFIC_CRITERIA.md): what is testable, what is
  only partially supported, and what remains future work.
- [Coarse-graining proof roadmap](COARSE_GRAINING_PROOF_ROADMAP.md): staged
  analytical + empirical proof path for the Coarse-Graining Postulate (the
  sufficient-statistic-for-control trunk; the Formal Separability appendix is
  its corollary). Theory-track, research-internal until its Phase 5 lands.
- [Postulate 1 definitions lock](proof/POSTULATE1_DEFINITIONS.md): Phase 0
  proof-track symbol ledger and Sundog-solvable predicate.
- [Proof artifacts index](proof/README.md): current proof-track artifacts,
  including the Phase 1 LQG draft, Phase 2 finite-MDP proof, Phase 3 boundary
  theorem, Phase 4 three-body spec, Bayesian-floor buildout, Phase 6
  lambda-control spec, and the Navier-Stokes C1 lock synthesis, kNN
  convergence check, twin-state certificate, and regime-generality design
  notes.
- [Bayesian floor profile template](BAYESIAN_FLOOR_PROFILE_TEMPLATE.md):
  reusable profile contract for installing same-observation Bayesian floors
  across workbenches without reinventing the audit shape.
- [Application map](APPLICATIONS.md): how the operating-envelope workbenches
  and product systems relate to the theorem.
- [Near-anniversary roadmap triage](ANNIVERSARY_ROADMAP_TRIAGE.md):
  *Historical (anniversary target 2026-05-19 is past).* Low-risk and
  high-value roadmap lanes that were scoped for the anniversary window.
  Retained for the record; not a primary entry point for new readers.
- [Promo folder](promo/README.md): promo highlights, outreach packet, and
  public-copy handoff material.
- [Brand folder](brand/README.md): public posture, brand/IP, legal standing,
  and summary lessons from quarantined model-feedback reports.
- [Site folder](site/README.md): website development, SEO/social readiness,
  UI/UX theme, assets, and policy ledgers.
- [Promo highlights](promo/PROMO_HIGHLIGHTS.md): hooks, headlines, provocative
  statements, and future-facing language.
- [Brand positioning](brand/BRAND_POSITIONING.md): public lab posture after the
  Mythos stress test, including the About-page spine and what-not-to-claim
  boundaries.
- [Standalone app roadmap](STANDALONE_APP_ROADMAP.md): plan for a no/low
  dependency `.html`, `.exe`, and eventual `.apk` observer app.
- [Three-Body roadmap](SUNDOG_V_THREEBODY.md): Phase 11 operating-envelope
  workbench for guarded local control in a near-escape dynamical pocket.
- [Three-Body Phase 11 summary](THREEBODY_PHASE11_SUMMARY.md): compact result
  readout for guard quantiles, outside-pocket expansion, and comparison slate.
- [Isotrophy folder](isotrophy/README.md): public K_facet ledger, full
  isotrophy roadmap, supplementary mirrors, and permanent K_facet appendix
  set after the v0.3-v0.9 pause.
- [Isotrophy K_facet v0.3h verdict](SUNDOG_V_ISOTROPHY_KFACET.md): public
  staging ledger for the v0.3h audit-chain result — 20 structural-zero
  receipts plus one named quarantine (O_617). Standing Rules 5 + 6 bind
  public copy; audit chain intact, theorem-facing result not closed.
- [Shadow Faraday folder](faraday/README.md): closed Phase 1-6 Faraday receipt
  chain, Phase 3 derivations, Phase 4 verification battery, and Phase 7
  source/topology boundary-audit spec.
- [Shadow Faraday ledger](SUNDOG_V_FARADAY.md): top-level Faraday roadmap
  with closed Phase 1-6 receipt chain and Phase 7 source/topology
  boundary-audit spec.
- [Sundog Balance roadmap](sundog_v_balance.md): confirmed cart-pole
  operating-envelope workbench for balancing from a shadow-derived signal.
- [Sundog Pressure Mines roadmap](sundog_v_minesweeper.md): Phase 11
  operating-envelope workbench for hidden mines and noisy pressure fields.
- [Sundog Chat roadmap](SUNDOG_V_CHAT.md): browser site-helper roadmap and
  substantiation for evidence-tier and claim-boundary preservation under prompt
  pressure, including the completed Phase 12 open-weight sweep.
- [Sundog Chat v2 ledger](SUNDOG_V_CHAT_V2.md): next-generation chat-library
  roadmap; staging surface set, audit-chain claim integration, and the
  Phase 13+ refresh plan. **Research-lane status: R1 MET 2026-06-29** per the
  disciplined [promote-gate](chatv2/PROMOTE_GATE.md) and the
  [R1 completion battery](chatv2/PHASE1_R1_COMPLETION.md) — a de-confounded,
  seed-robust, objective-driven body-resistance on *toy from-scratch
  transformers* with *parity-family latents* at `d_dec < 20`
  (`d_dec ≈ 7.6`, high-dim bar UNMET and honestly reported). Two architectures
  (A1 `d=192`, A2 `d=128`) and two latent computations (pair-XOR SHARP to
  `H=8`, 3-parity SHARP at `H≤4`) both cleared; F-readout / F-δ / F-opt
  falsifiers all passed. **R2 (real pretrained LLM + external mech-interp
  review) is NOT STARTED; R3 (theory of AI) is FAR.** The gate's §3
  do-not-claim ledger binds: no "world model," no "generative training is
  special," no "regime-2 confirmed for AI," no "explains
  generalization/scaling." A framework that *fits* a toy is not a theory that
  *predicts*.
- [Bayesian comparator ledger](SUNDOG_V_BAYES.md): cross-cutting Bayesian
  floor/comparator design — same-observation floor profiles, route-fidelity
  comparators, and the audit-chain integration across workbenches.
- [Sundog Least Action ledger](SUNDOG_V_LEAST_ACTION.md): legibility and
  meta-generality roadmap inspired by Euler/Lagrange, Snell/Fermat, and the
  principle of least reader-action; stages the `B/Phi/T/A/I/F/R` coordinate
  chart for making claims recoverable without hiding imports or falsifiers.
- [Sundog Algorithmic-Approximation lane](algo-approx/SUNDOG_V_ALGO_APPROX.md):
  lane ledger off arXiv:2606.26705 (Kratsios et al.); machine-checked find/check
  ledger (seven cheap-CHECK instances under one `Certifies` interface: max-flow,
  König, 2-SAT, Pratt, shortest-path, syndrome, ReLU gate-count) plus the
  tropical / PL core with linear gate count `≤ 4N` via a sharing-aware DAG and
  the N-1 monotone depth-as-computation theorem. Public surface at
  [/algorithmic-approximation](../algorithmic-approximation.html); slate at
  [`ALGO_APPROX_CONJECTURE_SLATE_2.md`](algo-approx/ALGO_APPROX_CONJECTURE_SLATE_2.md);
  receipts at
  [`ALGO_APPROX_N2_CANCELLATION_SPINE.md`](algo-approx/ALGO_APPROX_N2_CANCELLATION_SPINE.md)
  (synthesis lens, typed conjecture),
  [`ALGO_APPROX_N4_EPS_REGIONS_RESULT.md`](algo-approx/ALGO_APPROX_N4_EPS_REGIONS_RESULT.md)
  (empirical with named SGD residual), and
  [`ALGO_APPROX_CD2_GROKKING_RESULT.md`](algo-approx/ALGO_APPROX_CD2_GROKKING_RESULT.md)
  (the prior null this redeems).
- [Sundog Certificate (Lean) ledger](SUNDOG_V_CERTIFICATE_LEAN.md) **— now
  fifteen worked examples + one synthesis law + one constructive
  universal-approximation capstone**: the worked-example ledger (finite-field,
  real analysis ×2, geometric optics, gauge topology, Karp 3SAT→decoding,
  finite audit game, tropical/PL with linear gate count `≤ 4N`, N-1 monotone
  depth-as-computation, find/check ledger, plus five slate-3 closures:
  `cpl_iff_reluNet`, `AnalyticGate` + `SawtoothApprox`, `QueryGap`
  `check_lt_find` (a proved find/check gap in a restricted query model —
  **not** a P-vs-NP claim), `GradedCancellation`, and the empirical
  ε-essential sample-complexity) sits alongside **the Order-Relative
  Resolution Law** synthesis core (`Sundogcert/OrderRelative.lean` +
  `OrderRelativeMoment` / `OrderRelativeAlgDegree` / `OrderRelativeCohomology`):
  a schema law `Resolves k t ↔ ord t ≤ k` grounded on **seven instance
  families** (determination, coordinate-locality, search-reachability,
  radical-reach, spectral/moment, algebraic-degree, topological/cohomological),
  with `order_is_schema_not_scalar` as the honesty guard and the
  **composition law** (`compose_order_eq_lcm`, `compose_lcm_not_max`,
  `converse_fails`) showing the scalar order is the lattice join (lcm) of a
  latent component vector — **axis-internal**, on the group-order axes, with
  both walls of the characterization proved (three positive axes, two
  negative, plus a proved converse failure). **Above it all**, the
  constructive universal-approximation capstone
  (`UniversalApprox.continuous_relu_approximable`) — every continuous function
  on `[0,1]` is uniformly ε-approximable by an *explicit* ReLU net, proved
  end-to-end via the analytic-gate chain, axiom-clean, with **only
  Stone-Weierstrass density imported**. The capstone is the **constructive
  direction** of a classical fact; not a new approximation theorem, not a
  rate improvement. The synthesis law is a *schema*, not a universal scalar;
  the composition law is **axis-internal**, not a cross-axis identity. Not a
  worked example, not a P-vs-NP claim, not a learnability claim.
- [Tauroctony brand surface](../tauroctony.html) **— DELIVERED 2026-06-30**:
  the cosmogram ledger's deflation receipt as a Class A public page. The
  pantheon scaffold collapsed, on purpose, into a single inequality
  (`Sov_opt ≤ κ`); four safety axes (competence parity, corrigibility via
  override, non-sovereignty `p95 0.30 vs 0.71`, safe-interruptibility
  `band_avoidance ≤ 0.13` at every `κ ∈ {0.4 … 1.0}`); replicated under a
  learned presider (NS-3); competent sandbagger elicited then geometrically
  deterred (NS-4 / `CAP_DETERS_COMPETENT_SANDBAG`); three machine-checked
  Lean lemmas (`optimum_mono`, `signature_noninterference` axiom-free,
  `ruin_break_even`); released task family at `released/non-sovereignty/`,
  Apache-2.0. Standing claim boundary: Mithraic framing is **ornament
  throughout, not evidence**; operating-envelope evidence not proof; does
  not transfer to foundation models; field-grounding **relocates** the
  attack surface (not deletes). Companions: brand essay at
  [`mesa/CREDIT_THE_CAP_NOT_THE_COUNCIL_POST.md`](mesa/CREDIT_THE_CAP_NOT_THE_COUNCIL_POST.md);
  LW/AF technical companion (brand-stripped) at
  [`mesa/CAP_NOT_COUNCIL_LW_POST.md`](mesa/CAP_NOT_COUNCIL_LW_POST.md).
- [Sundog Tauroctony ledger](SUNDOG_V_TAUROCTONY.md): the portfolio's
  mythopoetic cosmogram — the tauroctony as one image holding the empirical
  ledgers (Sol/field, the bull/maximand, the two torchbearers, the bestiary as
  lanes). Braids a pantheon theory of agency, a hedged precession keystone, and
  the figure-by-figure map; governed by the Ornament Rule (`[TYPED]`/`[ORNAMENT]`).
  Supersedes the Least Action ledger in spirit (graceful-retire todo 3.5).
- [H1 Pantheon of Agency spec](mesa/H1_PANTHEON_OF_AGENCY_SPEC.md): first
  typed test of the Tauroctony pantheon thesis; pins the pantheon-vs-monolith
  bake-off, controller families, sovereignty metric, falsifier, and staged
  smoke shape.
- [H1.2 Small Pantheon bake-off spec](mesa/H1_2_SMALL_BAKEOFF_SPEC.md): records
  the learned-arbiter / label-trained-guard Small-tier bake-off after the H1.1
  smoke; freezes heads, matches coordinator budget against a monolithic
  adapter, and closes with the H1.2b binding `NULL`.
- [H1.2c reward-asymmetric cap spec](mesa/H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md):
  registered follow-up to the H1.2b negative; closes `H1_2C_NULL`, showing that
  bounding the reward/bull head while leaving the field/Sol head uncapped did
  not repair the diagnosed pantheon tax.
- [H1.2d RL arbiter spec](mesa/H1_2D_RL_ARBITER_SPEC.md): registered test of
  the named H1.2c bottleneck; trains the arbiter by direct rollout return and
  compares against a same-run equal-budget RL monolithic adapter.
- [H1.2d build smoke](mesa/H1_2D_SMOKE_RESULTS.md): confirms the PPO trainer
  and H1.2d eval branch run end-to-end; estimates the full H1.2d-a probe above
  the inline-run threshold.
- [H1.2d-a PPO probe results](mesa/H1_2D_A_RESULTS.md): records the three-cell
  PPO probe `H1_2D_SUPPORT`; superseded by the binding result below.
- [H1.2d-b RL arbiter binding results](mesa/H1_2D_RESULTS.md): closes the
  frozen-head Small-tier line as `H1_2D_PROXY_NULL`; RL fixes the arbiter and
  competence tax, but the monolith still wins proxy capture.
- [H1.2e cancelling-guard spec](mesa/H1_2E_CANCELLING_GUARD_SPEC.md):
  registered reopening that changes the guard from passive hold to an
  anti-reward countervote, without changing tier, heads, caps, or features.
- [Cautes and Cautopates essay](CAUTES_CAUTOPATES.md): companion narrative for
  the two torchbearers — reward vs punishment training economies and the third
  "field" path; the accessible on-ramp the Tauroctony ledger cites.
- [Sundog Perception roadmap](SUNDOG_V_PERCEPTION.md): atlas-as-instrument
  program — smartphone-tier Phase 1, deep observation ladder, and the path
  from sensing-grade to evidence-grade calibration.
- [Sundog Path ledger](SUNDOG_V_PATH.md): cross-workbench path-integral and
  trajectory-coupling notes; staging for cross-application connective tissue.
- [Mesa v2 spine](SUNDOG_V_MESAV2.md): future-integration spine sister to the
  mesa-v1 roadmap — four actionable phases (0.5, 0.6, 1', 6.5) plus the
  Formal Separability appendix retention.
- [Sundog Geometry roadmap](SUNDOG_V_GEOMETRY.md): parhelion-derived
  geometry workbench shelf — parametric halo render, parhelion-offset
  inverse, calibration boundaries, and the path-to-promotion phases.
- [Sundog Atlas roadmap](SUNDOG_V_ATLAS.md): the halo possibility-space as a
  **classified bifurcation diagram** — forward-generate halos from the crystal
  geometry, classify the transition walls (A: caustic catastrophes; B:
  ray-admissibility). Structural arc complete (6.5 bifurcation set → 8 strata,
  Berry-confirmed → 7 phase diagram → 11 capstone small-parameter model);
  frozen-as-portfolio, NOT public-eligible.
- [Sundog Atlas project folder](atlas/README.md): the atlas's banked receipts
  (`ATLAS_PHASE65/7/8/11`), the determining-shadow (S2) tower, and the lit-pass
  memo. Atlas phases carry the `ATLAS_PHASE…` prefix (distinct from the geometry
  workbench's `calibration/PHASE…` scheme).
- [SEO and social-readiness roadmap](site/SEO_AND_SOCIAL_READINESS_ROADMAP.md):
  per-page OG/Twitter/JSON-LD matrix, Phase 1 cleared 2026-05-21 across
  thirteen Class A pages; Phase 2 staged.
- [Sundog Gimmicks Ledger](SUNDOG_V_GIMMICKS.md): candidate game-native
  workbenches under evaluation before promotion to full roadmap documents.
- [Third-party reuse ledger](site/THIRD_PARTY_REUSE.md): permissions and attribution
  notes for borrowed design, code, and asset references.
- [Brand and IP roadmap](brand/BRAND_ROADMAP.md): legal hygiene, registration, and
  future entity formation plan.
- [Legal standing](brand/LEGAL_STANDING.md): summary of the legal standing pass and
  current ownership structure.
- [Paper draft](PAPER_v1_draft.md): current academic paper draft.
- [Paper outline](PAPER_OUTLINE_v0.md): venue framing, reviewer risks, and
  stress-test interpretation.
- [Oracle leakage audit](ORACLE_LEAKAGE_AUDIT.md): P0 receipt for the
  photometric "without target-position access" claim and its API boundary.

## High-Stakes Trial Ledgers

These ledgers couple Sundog machinery to hard external targets. They are
receipt-generation and boundary-mapping trials, not claims to have solved the
underlying public problems.

- [Riemann saga](SUNDOG_V_RIEMANN.md): RH-adjacent zero statistics, explicit
  formulae, and nonlinear gap-pair probes. Current result is a bounded-null
  synthesis: three lanes, three identified substrate causes, no structural-zero
  edge; public surface blocked on external review.
- [Riemann project folder](riemann/README.md): synthesis, Probe 01 and Probe 05
  receipts, C1 cell set, bridge notes, and external-review packet.
- [Riemann bounded-null synthesis](riemann/RIEMANN_BOUNDED_NULL_SYNTHESIS.md):
  capstone readout for the three-lane null; shortest artifact for external
  sanity review.
- [Riemann external-review packet](riemann/EXTERNAL_REVIEW_PACKET.md): minimal
  sanity-check packet and questions for the current review-gated bounded-null
  result.
- [Navier-Stokes ledger](SUNDOG_V_NAVIERSTOKES.md): Clay Millennium PDE coupling
  held to determining-mode / Kolmogorov-flow cell sets, vacuity gates, and
  proof-track receipts; no Navier-Stokes solution claimed.
- [Navier-Stokes lit-pass memo](NAVIERSTOKES_LITPASS_MEMO.md): prior-art spine
  and candidate ranking for the NSE lane.
- [Navier-Stokes C1 portable-objective design](proof/PDE_C1_REGIME_GENERALITY_v1.md):
  sign-off proposal after the `G = 300` regime-generality attempt hit objective
  vacuity rather than a fiber-locality verdict.
- [Yang-Mills handoff roadmap](SUNDOG_V_YANG_MILLS.md): draft finite-lattice
  gauge-invariant certificate lane. No lit-pass, preregistration, runner,
  receipt, mass-gap claim, or Clay-problem claim is live.
- [P-vs-NP verification ledger](SUNDOG_V_P_V_NP.md): bounded
  alignment-verification bridge asking whether compact signatures can certify
  named operating envelopes more cheaply than safe policies can be found.
- [P-vs-NP lit-pass memo](P_V_NP_LITPASS_MEMO.md): prior-art spine for
  promise-bounded verifier scaffolding.
- [P-vs-NP project folder](pvnp/README.md): Phase 1 toy-verifier slates,
  receipts, quarantines, and result conventions.
- [P-vs-NP receipt index](pvnp/receipts/README.md): v0-v5 receipt trail,
  including the current v5 provisional cost-adjudication hold.
- [BoxSEL false-closure ledger](SUNDOG_V_BOXSEL.md): bounded false-closure
  detection over Statistical EL box embeddings (BoxSEL, arXiv:2407.11821) —
  separates logical concentration from search/representation/loss-induced
  narrowing against an exact small-fragment oracle, then tests whether embedding
  traces can drive an accept/widen/abstain rule. Scaffold; lit-pass filled, no
  oracle/sampler/run yet.
- [BoxSEL lit-pass memo](boxsel/BOXSEL_LITPASS_MEMO.md): prior-art spine, claim
  ledger, the Phase-1 PMP replication gate (Proposition 2 vs Appendix Algorithm 2
  discrepancy), and the false-closure falsifier set.
- [Capset ledger](SUNDOG_V_CAPSET.md): hard-math coupling precedent around the
  OpenAI unit-distance / cap-set disproof context; evaluator front plus
  substrate-analogue horizon.
- [ARC-AGI roadmap](SUNDOG_V_ARC.md): abstraction-coupling trial promoted from
  the Gravity ledger, with a Phase 0 preregistration lane under `prereg/arc/`.
- [ARC-AGI preregistration folder](prereg/arc/README.md): task registers,
  decoder floors, Branch D variants, Phase 3E fiber certificates, and the
  current relative-locality execution hold.
- [ARC Phase 3E relative-locality spec](prereg/arc/PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md):
  rank-neighbor certificate proposal after absolute fibers stayed sparse in the
  expanded 108-task register.
- [Gravity ledger](SUNDOG_V_GRAVITY.md): highest-ambition staging pattern and
  candidate source ledger for Goodhart / partial-observability trials.
- [Mesa-optimization ledger](SUNDOG_V_MESA.md): high-stakes alignment substrate
  with localized cliff behavior and traceable operating-envelope boundaries.

## Supporting Docs

- [Phase-2 blocks design](PHASE2_BLOCKS_DESIGN.md): occlusion and block-based
  follow-up experiment design.
- [Icon assets](site/ICON_ASSETS.md): favicon, app icon, manifest, and HTML tags for
  `sundog.cc`.
- [Logo animation toolkit](site/LOGO_ANIMATION_TOOLKIT.md): Phase 11 characterized
  Sundog mark assets, protected layers, and motion rules.
- [Website development](site/WEBSITE_DEVELOPMENT.md): how to edit root HTML pages,
  link docs, build `dist/`, and deploy through Cloudflare Pages.
- [Chat claim map](../chat/claim_map.json): Phase 0 claim classes, source
  boundaries, evidence tiers, answer templates, and refusal rules for the
  `/chat` roadmap artifacts.
- [Chat contents index](../chat/contents.json): roadmap artifact index for
  planned prompts, public data, widget modules, evaluation scripts, and result
  manifests.
- [Chat prompt gold slate](../chat/prompts/gold-normal.jsonl): Phase 0 prompt
  slate split across normal, boundary-sensitive, and adversarial JSONL files
  under `../chat/prompts/`.
- [Chat browser search index](../public/data/sundog-chat-index.json): Phase 2
  generated static retrieval index for the no-LLM claim inspector.

## Reading Order

1. Start with the repository root `README.md`.
2. Read `RESEARCHER_GUIDE.md`.
3. Read `SCIENTIFIC_CRITERIA.md`.
4. Inspect `PAPER_v1_draft.md` and the recorded results under `../results/`.
5. Use `APPLICATIONS.md` to branch into the related product repositories.
