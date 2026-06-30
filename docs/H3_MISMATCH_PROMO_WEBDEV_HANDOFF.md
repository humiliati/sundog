# Promo / Webdev Handoff — Robustness under model mismatch (H3): the oracle is only a *local* upper bound

**Status: ACTIONABLE, owner-gated deploy.** Ready-to-paste copy below; nothing here auto-deploys.
This is an **operating-envelope extension** of the existing photometric claim, **not a new core
result**. It strengthens the page we already ship — it does not replace it.

---

## Where this came from (slate provenance — read this first)

A **zero-context frontier-model agent scanned the public `sundog` + `sundogcert` repos** (black-box,
from GitHub source only) and produced a deep-research report ranking ten hypotheses
(`internal/feedback/Agent/deep-research-report.md`). We triaged it against the actual code: it was
factually grounded where it quoted source, but — because it could only see the public surface — it
also invented one bug that doesn't exist (a license "mismatch": `package.json` is correctly
`UNLICENSED`) and mistook unauthenticated-view blind spots for gaps (e.g. "sundogcert CI looks
unconfigured" — the axiom-audit build gate runs fine). **Treat the report as an outside reviewer's
cold read, not an internal roadmap.** Of its ten hypotheses, we built and ran exactly one so far —
its rank-3, below. The rest are untouched.

So the framing for any public copy: *"an external review prompted us to map a new robustness axis"* —
honest, and it makes the result look like diligence rather than self-promotion. Do **not** imply the
agent validated Sundog or that the whole slate is confirmed.

---

## TL;DR — what's new and shippable

The public page already says: under nominal geometry the photometric controller **ties** the
target-aware analytic oracle on *terminal accuracy* while the oracle **wins hard on speed**
(~11.5 vs ~188 steps). That sentence has an implicit assumption — that the oracle's geometry is
correct. **We tested what happens when it isn't.**

We injected a **mirror calibration bias** (a warped/miscalibrated mirror — a model mismatch the
analytic oracle cannot see, but the closed-loop photometric controller measures through) and swept it
0°→60° at 30 seeds/level across five conditions. Result: **the ordering flips.** The oracle is the
better agent *near nominal*, but as mismatch grows it collapses while the photometric controller
degrades gracefully — so **"oracle" is only an upper bound in the nominal geometry, not universally.**

This is operating-envelope evidence (like the existing Phase 6 beam-sigma / detector-noise sweeps),
sitting next to the photometric ledger row. It is **not** proof, and it does **not** claim the
photometric agent is "better" — see the keep-true box.

---

## The finding in one picture (a crossover, not a victory)

| mirror bias | photometric (terminal) | oracle (terminal) | who wins | regime |
|---|---|---|---|---|
| 0° | 0.95 | 0.94 | tie | nominal — oracle faster, equal accuracy |
| 10° | 0.97 | 0.99 | **oracle** (p=0.03) | small mismatch — open-loop oracle still ahead |
| 20° | 0.89 | 0.90 | tie | crossover |
| 30° | 0.79 | 0.51 | **photometric** (p=3e-7) | mismatch — feedback degrades gracefully |
| 40° | 0.63 | 0.15 | **photometric** (p=3e-11) | oracle nearly dead; feedback holds |
| 50° | 0.20 | 0.00 | photometric, but both low | feedback's own reach giving out |
| 60° | 0.00 | 0.00 | neither | joint-limit wall (peak unreachable) |

The Bayes baseline collapses *even harder* than the oracle (0.26 at 30°), consistent with its
documented "stress = misspecification" design. Both model-based agents fail; only the model-free
closed-loop controller degrades gracefully.

Data: `results/mismatch_robustness/` (`manifest.json` status `flip_confirmed`, `boundary-map.csv`,
`level-summary.csv`, `trial-outcomes.csv` — 900 episodes + per-seed bias-aware ceiling).

---

## Paste-ready copy (pick a length)

**Tagline (≤ 12 words):**
> The analytic oracle is only an upper bound in the *nominal* geometry.

**One sentence (sidebar / meta description):**
> When the mirror is miscalibrated — a model error the target-aware oracle cannot see — the
> closed-loop photometric controller degrades gracefully while the oracle collapses, so the oracle's
> advantage is local to the nominal geometry, not universal.

**Short paragraph (card / operating-envelope section body):**
> The photometric controller ties the target-aware analytic oracle on terminal accuracy under nominal
> conditions and trails it badly on speed. But "oracle" assumes the geometry is exactly known. We swept
> a mirror-calibration bias — a model mismatch invisible to the oracle's analytic solve — and the
> ordering **flips**: the oracle stays ahead through small mismatch, then collapses as the bias grows,
> while the closed-loop photometric controller holds far more of its terminal intensity. A same-model
> Bayesian baseline collapses harder still. The result is a robustness boundary map (30 seeds/level),
> not a proof: it shows *where* indirect photometric feedback is slower-but-equally-good,
> slower-but-more-robust, or simply inferior.

---

## Facts you can cite (each is true and in the run artifacts)

| Plain-language claim | One-liner you can use | Backed by |
|---|---|---|
| Nominal tie reproduced | "At zero bias, terminal accuracy ties (0.95 vs 0.94) and the oracle is ~16× faster — matching the headline result." | bias-0 anchor, `manifest.json` `anchor_bias0.pass = true` |
| The flip is real | "Under a mirror-calibration mismatch the photometric controller significantly beats the oracle on terminal accuracy (p < 1e-6 at 30°)." | `boundary-map.csv`, Mann–Whitney |
| It's a crossover, not supremacy | "Near nominal the oracle wins; the photometric controller only overtakes once mismatch is large." | 10° = oracle_dominant; 30–40° = photometric_dominant |
| Mechanism named | "Model-based agents (oracle, Bayes) collapse under unmodelled mismatch; the model-free closed-loop one degrades gracefully." | Bayes 0.26 vs oracle 0.51 at 30° |
| Honest wall | "Past ~50° both fail — when the bias pushes the true peak beyond the joint limit, no agent can win." | 60° both_fail, ceiling → 0 |
| Reproducible | "Five conditions × seven bias levels × 30 matched seeds; one command." | `python -m sundog.experiments.mismatch_robustness` |

---

## Keep-true box (the whole guardrail — do not cross these)

1. **Not a new core claim, not proof.** This is operating-envelope evidence on a synthetic injected
   mismatch (a deliberately miscalibrated mirror), in the same MuJoCo setting. Shelve it beside the
   Phase 6 sweeps, not above the headline result.
2. **The win is *graceful degradation on terminal accuracy* only.** Under mismatch the photometric
   controller never re-reaches the 0.9 convergence threshold and stays slow — it is "less bad," not
   "good." Never say it "solves" or "beats" the task under mismatch.
3. **The oracle still wins near nominal.** Lead with the crossover. Do not imply the photometric
   controller is universally more robust.
4. **Don't credit the scanning agent.** It prompted the question; it did not validate anything.

---

## Suggested surface edits (concrete)

- **`alignment.html`** — `#photometric-ledger-row`: append one clause to the "what it shows" cell, e.g.
  *"…and, under an injected mirror-calibration mismatch, retains terminal accuracy as the oracle
  collapses (operating-envelope crossover, 30 seeds/level)."* Keep the existing speed caveat verbatim.
- **`alignment.html`** — operating-envelope section (near the Phase 6 links): add a third axis card
  "Model mismatch (mirror calibration)" next to beam-sigma / detector-noise, linking the boundary map.
- **`docs/SUNDOG_V_BAYES.md`** — add a short "Phase 6c — model-mismatch crossover" subsection so the
  page's doc cross-refs stay complete; point it at `results/mismatch_robustness/manifest.json`.
- **Figure**: the boundary-map line chart (terminal intensity vs bias, five series + ceiling). The run
  did not emit a PNG (matplotlib is an optional dep and isn't installed in the experiment env); webdev
  can render from `level-summary.csv`, or ask and we'll produce an asset.

---

## Source pointers

- Lever: `env_v2.py` `set_normal_bias()` / `tilt_normal()`; stressor in `experiments/stress_tests.py`.
- Experiment: `experiments/mismatch_robustness.py`.
- Slate: `internal/feedback/Agent/deep-research-report.md` (hypothesis #3 of 10).
