# SUNDOG_V_P_V_NP — Tracks D & C Handoff (post-v3 syndrome ladder)

> 2026-06-06. **Scaffold/roadmap, NOT execution.** This hands the P-vs-NP team two on-pivot next
> moves — **Track D** (wire the certificate framework to the flagship + compander thread) and
> **Track C** (assemble the portfolio paper). Direction B is closed (`DIRECTIONB_GATE0_NOTE.md`);
> the syndrome/ISD v1→v3 capacity ladder is now filed. House template:
> `docs/chatv2/JEPA_0D_HANDOFF.md`. Parent: `../SUNDOG_V_P_V_NP.md` +
> `SUNDOG_CERTIFICATE_PROBLEM.md`.

## 0. Current state (read first)

The **v2 stronger-ISD run and v3 scaling ladder have both landed**. Treat their receipts
as the source of truth; do not re-run or reinterpret the frozen artifacts unless a new
slate explicitly asks for it. This handoff now starts from the filed ladder, not a pending
operator run.

## 1. What is already banked (the material D & C draw on)

All figures are read from dated receipts; cite the receipt, never a remembered number.

| asset | status | figure | source |
| --- | --- | --- | --- |
| Formal promise problem (3-valued V, 7 cert components, named-failure contract) | banked | — | `SUNDOG_CERTIFICATE_PROBLEM.md` §1, §1.4 |
| **Claim 1 — cheap verification (op-count)** | banked | v6 ≈**0.879** honest / **0.948587** conservative; Phase-2 v1 **0.7376** (weaker, non-comparable comparator) | receipts `2026-05-31_phase1_toy_verifier_v6.md`, `..._phase2_mesa_bridge_v1.md` |
| **Claim 2 — capacity-relative spoof resistance** | bounded + **seed-fragile** | v0 falsified → v1 consensus-only → v2b bounded-positive (frozen seeds); `source_block_safety_claim_allowed=false` | receipts `..._phase3_..._v1/v2b.md` |
| **Claim 3 — disclosure robustness** | pre-registered **NULL** | anchor `clean_consensus` on all 3 fresh batteries (v3) | receipt `2026-06-04_phase3_capacity_one_wayness_v3.md` |
| **Measured boundary 1 — syndrome/ISD certificate** | banked measured ladder | v1 Prange threshold C≈**5007** trials; v2 best tested attacker **Lee-Brickell** (`C_best=8.31×10⁷`, gap ≈5,015×); v3 locates the LB↔Stern crossover and reaches top gap **218,999×** at `[192,96]w18` with Claim-A gate caveat + rung-2 model-deviation disclosed | receipts `2026-06-04/05/06_certificate_syndrome_v{1,2,3}.md` |
| **Measured boundary 2 — Direction-B Gate-0** | banked negative | emergent leg-(d) one-wayness from a trained body **NOT available**: σ lossy by algebra (secret-from-σ ≈ **−0.002**); trained body **exposes** z (z-det −0.018 → **+0.31**, Δ wrong sign) | `DIRECTIONB_GATE0_NOTE.md`, `results/pvnp/directionb-gate0/` |
| Substrate migration off mesa (three-for-three marginal → AB/topological) | banked reasoning | mesa FVE 0.97–0.99 / PR≈2; NSE-C1 0.99; Sabra 1.7/30 | `SUNDOG_CERTIFICATE_PROBLEM.md` §3, `../CROSS_SUBSTRATE_NOTES.md` |
| **Syndrome methodology finding** | banked | empirical pre-calibration is necessary; analytic ISD heuristics are wrong in both directions, and heavy-tailed Stern needs a median-implied prediction lock | receipt `2026-06-06_certificate_syndrome_v3.md` |

**The one-line framing both tracks must preserve:** *the one-wayness leg is imported (by H_pub
algebra); the emergent, non-imported, trained-body prize is the state-insufficiency + control-
sufficiency + de-confound bundle (chatv2 pair-XOR), which never needed the one-wayness leg to beat
the constructed instance.* No P-vs-NP claim, no cryptographic one-wayness, no "verification in P."

---

## 2. Track C — the portfolio paper (the honest worked example)

**Goal.** Assemble the Phase-8 paper-shaped note around the three claims + two measured boundaries.
The negatives (Claim-3 null, the mesa migration, the Direction-B import boundary) are **features** —
they are the honesty that makes the bounded positive credible.

### C1. Deliverable
One internal draft, e.g. `docs/pvnp/CERTIFICATE_PAPER_NOTE.md` (NOT public; evidence-tier per
`../SCIENTIFIC_CRITERIA.md`; owner sign-off before any outward surface). Lead with the boundary, not
the provocation (roadmap §8 exit criterion).

### C2. Required sections (map to roadmap §8 + `SUNDOG_CERTIFICATE_PROBLEM.md`)
1. Motivation — P-vs-NP as **vocabulary, not target** (Cook–Levin–Karp as language; the
   relativization / natural-proofs / algebrization guardrails stand).
2. The formal promise problem (§1): 3-valued verifier, 7 certificate components, out-of-envelope →
   quarantine, the named-failure contract (§1.4).
3. Certificate definition (locality / invariance / sufficiency) + capacity-relative one-wayness as a
   **measured** verify/invert/spoof battery only.
4. **The three claims, separated** (§2) — cost (op-count, earned), spoof-resistance (bounded +
   seed-fragile), disclosure-robustness (the measured null). **Do not conflate; do not average the
   two non-comparable op-count comparators.**
5. The constructed instance + **measured boundary 1** (syndrome/ISD): the v1 Prange curve,
   v2 stronger-ISD ladder, and v3 scaling/crossover ladder; state every `C_best` as an
   upper bound against the named tested attackers.
6. **Measured boundary 2** (Direction-B Gate-0): the import-vs-demonstrate line, resolved by
   measurement — one-wayness is imported, the (a)+(b)+(c) bundle is the emergent prize.
7. Faraday receipt grammar (structural zero / named quarantine) as the discipline model.
8. What remains unsolved (asymptotic / best-attacker threshold; real-data de-confound; the
   demonstrate-not-import frontier the gradient barrier closed on smooth bodies).

### C3. Title options (roadmap §8)
"Finding Is Hard, Checking the Shadow Is Cheaper" · "Capacity-Bounded Verification from Indirect
Signatures" · "Signature Certificates for Alignment Verification."

### C4. Discipline (inherited, load-bearing)
op-count not wall-time; measured-not-asymptotic; "imports hardness" stated plainly; seed-fragility of
Claim 2 carried with `source_block_safety_claim_allowed=false`; read every number off the emitted
artifact at write time.

---

## 3. Track D — certificate framework as the flagship's theory-spine

**Goal.** Reframe P-vs-NP from an orphaned generality lane into the **theory-spine of the Ask Sundog
widget** (the 0/5670 flagship) and connect it to the compander / stack-invariance thread — serving
the product north star, not open-ended generality.

### D1. The mapping (certificate framework ↔ widget operating discipline)
The widget already *is* a capacity-bounded signature verifier; the framework gives it vocabulary.
Verify each row against the canonical files before asserting it (don't fabricate widget internals):
`public/js/sundog-chat-widget.mjs`, `chat/claim_map.json`, `chat/prompts/gold-*.jsonl`,
`docs/SUNDOG_V_CHAT.md`.

| certificate framework | Ask Sundog widget |
| --- | --- |
| 3-valued `accept / reject / quarantine` (never binary) | answer / refuse / route-to-quarantine on out-of-scope claims |
| promise envelope `D`; out-of-envelope → quarantine, never silent accept | the widget's registered claim-boundary scope; refuses outside it |
| false-accept-first metric ordering | the **0 unsafe-accepts / 5,670** discipline |
| cheap to check vs hard to find | the widget checks claim-boundary cheaply, does not re-solve alignment |
| imported hardness, bounded claim | the widget makes a **bounded** claim, not a universal one |

### D2. The compander bridge (gated)
Capacity-relative one-wayness vocabulary gives precise language to the stack-invariance line ("the
artifact keeps the relevant low-dim subspaces aligned"). The **Direction-B four-leg decomposition +
Gate-0 measured boundary** is also a candidate thinking-partner contribution to the mod's "look at
JEPA next" step (offer per `../../internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md`
§9e) — **internal only**.
**HARD GATE:** any public lift of the compander/JEPA framing is blocked behind the §10
publication-trigger gate of that file (the `COMPANDER_PAPER_HOOK` rail; 9 anchors across
`unit-distance.html` ×2, `chat.html`, `capset.html`, `geometry.html`, `safety-method.html`,
`docs/promo/PROMO_HIGHLIGHTS.md`, `docs/CROSS_SUBSTRATE_NOTES.md`, `docs/SUNDOG_V_CHAT.md`). Do
**not** deploy ahead of the mod's publication; reuse `scripts/rollout-compander-citation.mjs` when
the gate trips. (Confirm the live anchor set with the §10b deployment grep before any rollout.)

### D3. Deliverables
1. **Internal spine note** (`docs/pvnp/CERTIFICATE_AS_WIDGET_SPINE.md`): the D1 mapping table +
   exactly which banked claim licenses which widget statement (and which it does **not**). Internal.
2. **Staged, DO-NOT-DEPLOY public-copy upgrade**: a draft positioning P-vs-NP as the spine, behind
   owner sign-off + evidence tiers; the compander half behind the citation rail.
3. **Portfolio-surface placement proposal** (`applications-gallery.html`,
   `public/data/high-stakes-generality-gallery.json`, `site-pages.json`): the bounded positive +
   the two measured boundaries, with the seed-fragility and measured-not-asymptotic caveats inline.
   Proposal only — owner approves before any deploy.

### D4. Discipline
Public copy = owner sign-off + `../SCIENTIFIC_CRITERIA.md` evidence tiers. Never imply "alignment
verification is in P" or cryptographic one-wayness. Carry the Claim-2 seed-fragility and the
measured-not-asymptotic boundary. Compander framing stays internal until the publication gate.

---

## 4. Sequencing & dependencies

```text
[v1-v3 syndrome ladder filed]   <-- complete; use receipts, no rerun
        |
   Step 1: Track C  — assemble CERTIFICATE_PAPER_NOTE.md (uses the now-complete claim set)
        |
   Step 2: Track D  — CERTIFICATE_AS_WIDGET_SPINE.md (internal; reuses C's framing)
        |
   Step 3: Track D  — staged public-copy upgrade + portfolio placement (DO NOT DEPLOY)
                      gated on: owner sign-off  AND  (for the compander half) the COMPANDER_PAPER_HOOK
                      publication trigger — an INDEPENDENT external gate that may never trip.
```

C is the prerequisite framing for D; both are internal-doc work until the owner opens the public
surface. None of it touches the live v2 artifacts.

## 5. Definition of done

The P-vs-NP team can, without re-deriving any claim or touching the live run:
1. fold the v2 C-ladder into the figures;
2. assemble the portfolio paper note (Track C) leading with the boundary;
3. write the widget-spine note and a staged (un-deployed) public upgrade + portfolio placement
   proposal (Track D);
4. leave every public deploy behind owner sign-off and (for the compander half) the citation gate.

## 6. Footguns (inherited lane discipline)

- **Do not touch the live v2 run** (`results/pvnp/certificate-syndrome-v2/`, the frozen slate,
  scorer, prediction-lock) until its receipt is filed.
- Read every receipt number off the emitted artifact **at write time** (a fabricated v5 "FINAL" + two
  fabricated freeze-blockers were caught exactly this way).
- Wall-time is **diagnostic-only**; op-count is the cost signal. Never reinstate a withdrawn
  wall-time claim.
- Never retune a frozen threshold as a "repair"; a new rule = a new slate id.
- Do not conflate the three claims or average the two non-comparable op-count comparators.
- Public copy needs owner sign-off + evidence tiers; the compander/JEPA framing stays internal until
  the §10 publication gate.
- Lane status is **frozen-as-portfolio** (2026-06-04 pivot); D & C are the portfolio-ization, not a
  re-opening of active research. (Reconcile: the freeze currently lives in memory; these docs still
  read as in-progress — the team may put the freeze on-record while assembling C.)

---

*Sundog Research Lab — P-vs-NP Tracks D & C handoff. Direction B closed by measurement; A (v2) in
flight; D & C scaffolded for post-v2 execution. Bounded, boundary-first, R1/control-substrate. No
complexity-theoretic result claimed.*
