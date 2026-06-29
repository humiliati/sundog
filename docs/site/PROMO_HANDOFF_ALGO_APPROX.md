# Hand-off — Algorithmic-Approximation / find-check lane → promo + webdev

**Audience:** the promo / webdev team building public surfaces on `sundog.cc`.
**Date:** 2026-06-29. **Author:** research lane (algorithmic-approximation, off arXiv:2606.26705).
**Status:** the research lane is **complete and machine-checked**; the **public surfaces that
describe it are stale**. This document is what to check, what is safe to say, and what to build.

> **Read Part 2 (the claim boundary) before writing any copy.** This material is adjacent to
> P-vs-NP, primality, and "neural nets generalize" — the three things most likely to be
> overclaimed. The existing pages (`p-vs-np.html`, `generality.html`) are scrupulous about this;
> match that discipline exactly. When in doubt, under-claim and ask the owner.

---

## Part 1 — Lane progress (what you can verify)

Everything below lives in the **public** `sundogcert` Lean repo (`Dev/sundogcert`, Lean 4.30 +
mathlib) and is **axiom-clean and build-gated**: `lake build` re-certifies every theorem, and
`Sundogcert/AxiomAudit.lean` pins each headline to Lean's three foundational axioms (`propext`,
`Classical.choice`, `Quot.sound`) — a `sorry` or extra axiom **fails the build**. "Referee-free":
the kernel re-checks in seconds, so validity is author-independent. The narrative docs are in
`Dev/sundog/docs/` and are copied into `dist/` by `npm run build`, so they are publicly linkable.

| Theme | Modules (`Sundogcert/…`) | What is proven (the CHECK side) |
|---|---|---|
| Exact compilation | `CircuitNet` | tropical/piecewise-linear circuits compile **exactly** (ε=0) to ReLU nets; **linear gate count** `≤ 4N` via a sharing-aware DAG (`compileToDag`) — *this resolves the old "DAG follow-up" caveat still printed in the pillar doc* |
| Regions intrinsic | `RegionCount` | linear-region structure is realization-independent (exactness) |
| Monotone wall | `CancellationFree` | the cancellation-free (`IsMono`) fragment computes exactly the monotone functions; `abs` needs the negative scale |
| Depth = computation | `DepthSeparation` | the tent map's `d`-fold has `2^d` regions from depth `O(d)` |
| **N-1: monotone depth is region-polynomial** | `FoldCancellation`, `PieceCover`, `RegionPoly` | folding needs cancellation (`isMono_not_iterTent`); a cancellation-free circuit's region count is **linear in its size** (`isMono_hasPieceCover`) — closed on both the depth and circuit-structure axes |
| **N-3: the find/check ledger** | `Certifies`, `MaxFlowMinCut`, `MatchingCover`, `TwoSat`, `PrattCert`, `ShortestPathCert`, `StraightLineCost` | one `Certifies` interface, **7 instances** with cheap-CHECK theorems: syndrome verifier · shortest-path · ReLU gate-count · max-flow/min-cut · König matching/cover · 2-SAT · Pratt primality |
| **N-2: the cancellation spine** | `CancellationSpine` | `isMono_tame` (cancellation-free ⇒ monotone ∧ convex ∧ region-polynomial) — the machine-checked half of "cancellation is the single coordinate"; the full claim stays a typed conjecture |
| **N-4: ε-essential regions (empirical)** | `scripts/algo_approx_n4_eps_regions.py` | the ε-essential region count predicts a net's generalization threshold (0.89 vs 0.50 for exact `k`), **modulo SGD trainability** |

Narrative receipts (the lane docs now live in `Dev/sundog/docs/algo-approx/`):
`ALGO_APPROX_CONJECTURE_SLATE_2.md` (the slate, with every hook's status),
`ALGO_APPROX_N2_CANCELLATION_SPINE.md`, `ALGO_APPROX_N4_EPS_REGIONS_RESULT.md`,
`ALGO_APPROX_CD2_GROKKING_RESULT.md`. The cross-lane pillar stays at
`Dev/sundog/docs/SUNDOG_V_CERTIFICATE_LEAN.md` (it is not algo-approx-specific).

---

## Part 2 — The claim boundary (do / do-not-say)

**Safe to say:**
- "A public, referee-free Lean repo: machine-checked, axiom-clean cores spanning many kinds of math."
- "Cheaper to **check** than to **find** — the verifier side is proved cheap; finding the witness stays the hard part." (This is the lane's whole frame.)
- "Primality is **in NP**: a short certificate exists that a checker verifies fast" (Pratt).
- "Max-flow/min-cut, König, shortest-path: a dual witness **certifies** an optimum, cheaply."
- "Cancellation-free (monotone) circuits stay simple; the exponential blow-up needs cancellation."

**Do NOT say (these are the overclaim traps):**
- ❌ Anything implying **P vs NP** is solved/advanced, or a **complexity-theoretic separation**. This lane borrows the find-vs-check *asymmetry as vocabulary*. NOT the Millennium problem.
- ❌ That **hardness** is proven. Every "find" / hardness fact is **imported, not proven** (decoding hardness, depth-vs-width lower bounds, factoring). Lean certifies the deductive CHECK side only.
- ❌ That Pratt is a **fast primality test** or relates to **breaking encryption / factoring**. It is NP-*membership* (a cert exists); finding it still needs factoring `p−1`.
- ❌ That the find/check modules are **solvers**. They CHECK a supplied optimum; they do not find it.
- ❌ That N-1 is a **lower bound for general neural nets**. It bounds the *cancellation-free* fragment; the general depth-vs-width lower bound (Telgarsky) is imported.
- ❌ That N-2 "every wall is cancellation" is **proved**. It is a typed conjecture, machine-checked only on the fold/monotone/region axes.
- ❌ That N-4 **proves** anything about generalization. It is an **empirical** finding (region geometry predicts the threshold *modulo trainability*) — no theorem, no guarantee.
- ❌ "Lean-verified" applied to the *whole story*. It means the **deductive core**, never the imported hardness.

---

## Part 3 — Public-surface work (in priority order)

### 3a. Pillar bump — `docs/SUNDOG_V_CERTIFICATE_LEAN.md` (low effort, high value)

This is the "Lean ledger" the other pages cite, and it is **stale**: it reads
*"eight worked examples across seven kinds of math"* and lists only the original `CircuitNet`
example (with a *"linear gate-count … still needs the DAG follow-up"* caveat that is now
**resolved** by `compileToDag`). Since then the repo gained the whole slate-1/slate-2 build.

Update it to reflect: **two new categories** — *combinatorial optimization* (max-flow/min-cut,
König, shortest-path), *decision & number theory* (2-SAT, Pratt) — plus the N-1 region results and
the unified `Certifies` find/check ledger (7 instances). Refresh the example count and drop the
resolved DAG caveat. Keep the existing per-example "what stays imported" column — that column is
the claim hygiene; carry it for every new row (see Part 2 for the per-module imports).

### 3b. Copy bump — `p-vs-np.html` + `generality.html` (low effort)

Both pages state the Lean method repo *"now spans three examples (finite-field algebra, real
analysis, geometric optics)."* That count is well out of date. Update the breadth line to include
**circuit complexity / approximation theory, combinatorial optimization, decision, and number
theory** — framed (as those pages already do) as **method context, not extra P-vs-NP evidence**.
Do not alter those pages' standing claim boundaries; only the breadth sentence.

### 3c. Optional new page — the find/check ledger / cancellation spine (bigger, owner-gated)

If the team wants a dedicated surface, the cleanest is an **exhibit-led page** in the
`ghost.html` mold: lead with the one-line payload ("cheaper to check than to find — machine-checked
across seven problems"), a small table of the 7 ledger instances, and links into the public docs +
the `sundogcert` repo. Natural names: `find-check.html` or `algorithmic-approximation.html`;
sibling to `p-vs-np.html`, inbound from the `generality.html` umbrella.

Follow the established page-launch pattern (see `ghost.html` / `p-vs-np.html` as templates and
`docs/site/SEO_AND_SOCIAL_READINESS_ROADMAP.md`): a `site-pages.json` `entry` + a claim-disciplined
`publicLaunchIntent`, then **Bucket 1** before public push — bespoke 1200×630 `public/og/<page>.png`,
OG/Twitter metadata, JSON-LD, tuned title/description, a clean-url `_redirects` row, a sitemap
entry, an inbound internal link, and a **post-deploy LinkedIn/Twitter validator pass**.

---

## Part 4 — Build / deploy mechanics

- Source is the repo **root** (`index.html`, Vite); publish **only `dist/`** (`npm run build` →
  `npm run deploy`). `npm run build` copies `docs/**` into `dist/`, so all the
  `docs/algo-approx/ALGO_APPROX_*.md` receipts are already publicly linkable once deployed.
- New `site-pages.json` entry ⇒ matching row in `docs/site/SEO_AND_SOCIAL_READINESS_ROADMAP.md`,
  Bucket 1 cleared before `publicLaunchIntent` is treated as satisfied (per `AGENTS.md`).
- `npm run sundog:check` only gates `sundog.html` + the geometry module — not relevant unless you
  touch those.
- **Owner-gated:** `npm run deploy` (Cloudflare credentials are local operator secrets);
  **any new public claim**; and **asset/licensing sign-off** for new OG art. Do not deploy or push
  externally without owner approval — this lane's promo risk is claim hygiene, not mechanics.

---

## Part 5 — Open decisions for the team

1. **Pillar-bump only, or also a new page?** 3a+3b alone fully refresh the existing surfaces with
   low risk. 3c is a new public commitment (and a new claim surface) — owner call.
2. **If a new page: name + placement** (`find-check` vs `algorithmic-approximation`; which umbrella
   row in `generality.html`).
3. **Which results to feature.** Recommendation: lead with the *find/check ledger* (concrete,
   relatable: max-flow, primality) and the *referee-free* angle; keep N-2 (conjecture) and N-4
   (empirical) as honest secondary cards, never headlines.
4. **External push timing.** Indexable launch is fine once Bucket 1 is clean; hold
   LinkedIn/Twitter sharing for the post-deploy validator pass + owner sign-off.

## Appendix — source pointers

- Lean: `Dev/sundogcert/Sundogcert/{CircuitNet,RegionCount,CancellationFree,DepthSeparation,FoldCancellation,PieceCover,RegionPoly,CancellationSpine,Certifies,MaxFlowMinCut,MatchingCover,TwoSat,PrattCert,ShortestPathCert,StraightLineCost}.lean`; gate `Sundogcert/AxiomAudit.lean`; `README.md` (module table).
- Docs: `Dev/sundog/docs/algo-approx/ALGO_APPROX_CONJECTURE_SLATE_2.md`, `…/ALGO_APPROX_N2_CANCELLATION_SPINE.md`, `…/ALGO_APPROX_N4_EPS_REGIONS_RESULT.md`, `…/ALGO_APPROX_CD2_GROKKING_RESULT.md`; cross-lane pillar `Dev/sundog/docs/SUNDOG_V_CERTIFICATE_LEAN.md`.
- Experiment: `Dev/sundog/scripts/algo_approx_n4_eps_regions.py`.
- Site mechanics: `AGENTS.md`, `docs/site/SEO_AND_SOCIAL_READINESS_ROADMAP.md`, `docs/site/WEBSITE_DEVELOPMENT.md`; page templates `p-vs-np.html`, `ghost.html`.
