# Sundog Documentation

**Copyright © 2026 Stellar Aqua LLC. Licensed under the MIT License.**

Sundog-authored source and associated documentation are published under the root
[`LICENSE`](../LICENSE). Citation metadata lives in [`CITATION.cff`](../CITATION.cff).

---

This folder is organized as a research-facing landing layer over the Sundog
code, experiments, prior theorem materials, and application bridges.

## Primary Entry Points

- [Table of contents](index.html): browsable HTML index of all documents in
  this folder.
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
  including the Phase 1 LQG draft.
- [Application map](APPLICATIONS.md): how the operating-envelope workbenches
  and product systems relate to the theorem.
- [Promo highlights](PROMO_HIGHLIGHTS.md): hooks, headlines, provocative
  statements, and future-facing language.
- [Brand positioning](BRAND_POSITIONING.md): public lab posture after the
  Mythos stress test, including the About-page spine and what-not-to-claim
  boundaries.
- [Standalone app roadmap](STANDALONE_APP_ROADMAP.md): plan for a no/low
  dependency `.html`, `.exe`, and eventual `.apk` observer app.
- [Three-Body roadmap](SUNDOG_V_THREEBODY.md): Phase 11 operating-envelope
  workbench for guarded local control in a near-escape dynamical pocket.
- [Three-Body Phase 11 summary](THREEBODY_PHASE11_SUMMARY.md): compact result
  readout for guard quantiles, outside-pocket expansion, and comparison slate.
- [Sundog Balance roadmap](sundog_v_balance.md): confirmed cart-pole
  operating-envelope workbench for balancing from a shadow-derived signal.
- [Sundog Pressure Mines roadmap](sundog_v_minesweeper.md): Phase 11
  operating-envelope workbench for hidden mines and noisy pressure fields.
- [Sundog Chat roadmap](SUNDOG_V_CHAT.md): browser site-helper roadmap and
  substantiation for evidence-tier and claim-boundary preservation under prompt
  pressure, including the completed Phase 12 open-weight sweep.
- [Sundog Gimmicks Ledger](SUNDOG_V_GIMMICKS.md): candidate game-native
  workbenches under evaluation before promotion to full roadmap documents.
- [Third-party reuse ledger](THIRD_PARTY_REUSE.md): permissions and attribution
  notes for borrowed design, code, and asset references.
- [Brand and IP roadmap](BRAND_ROADMAP.md): legal hygiene, registration, and
  future entity formation plan.
- [Legal standing](LEGAL_STANDING.md): summary of the legal standing pass and
  current ownership structure.
- [Paper draft](PAPER_v1_draft.md): current academic paper draft.
- [Paper outline](PAPER_OUTLINE_v0.md): venue framing, reviewer risks, and
  stress-test interpretation.

## Supporting Docs

- [Runners](runners.md): Gone Rogue / EyesOnly runner integration.
- [Phase-2 blocks design](PHASE2_BLOCKS_DESIGN.md): occlusion and block-based
  follow-up experiment design.
- [Icon assets](ICON_ASSETS.md): favicon, app icon, manifest, and HTML tags for
  `sundog.cc`.
- [Logo animation toolkit](LOGO_ANIMATION_TOOLKIT.md): Phase 11 characterized
  Sundog mark assets, protected layers, and motion rules.
- [Website development](WEBSITE_DEVELOPMENT.md): how to edit root HTML pages,
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
