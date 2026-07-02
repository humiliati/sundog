# Promo / Webdev Handoff — Order vs Counts: the State-Crossover result (v1, ACTIONABLE)

> 2026-07-02. The chat-v2 intersection arc is **complete end-to-end**: a machine-checked
> impossibility (what count statistics can never read), a measured crossover on a real
> pretrained model exactly where the impossibility bites, and a fully-measured negative
> boundary for the high-dimensional version (six label families, three gate designs, two
> model scales — all pre-registered, all receipts filed). This version leads with
> ready-to-paste copy and keeps the guardrail to three lines. **You can draft from this
> directly; go-live is gated only by the two pushes at the bottom.**

---

## TL;DR — what's shippable now

- **A new machine-checked pair in the public Lean repo** (the surface-window axis — the 8th
  filtration of the already-public Order-Relative section): the order-blind **bag** (count
  vector) *determines* bracket nesting **depth** but can **never** determine the
  **stack-top** — proved at window w = 1 and then at **every window order** (σ = ∞).
  Axiom-lean: `[propext, Quot.sound]` — no `Classical.choice`.
- **A measured crossover on a real model:** on real Python code, GPT-2's residual stream
  reads the stack-top at **0.931** exactly where the count statistic collapses to **0.770**
  — and at easy (count-determined) positions the counts *win* (0.965 vs 0.926). The
  crossover happens precisely where the theorem says order starts to matter.
- **An honest, fully-measured boundary:** the high-dimensional version (≥ 20 independent
  state axes on natural data) failed every construction — six label families, absolute and
  matched-baseline (crossover) gates, GPT-2-small and Qwen2.5-1.5B, with the apparatus
  validating itself on known-answer controls at every rung. The negative *is* the result.
- **The NSE page's open thread closes:** the "resistant substrate" test the re-aim
  paragraph promised has now been scoped and run — follow-up copy below.

---

## Paste-ready copy (pick a length)

**Tagline (≤ 14 words):**
> Order is what counts can't see — proved in Lean, then measured in a model.

*(alt: "We proved which labels no count statistic can read — then watched a model read one.")*

**One sentence (sidebar / meta description):**
> Sundog's state-crossover result pairs a machine-checked impossibility — the order-blind
> count statistic determines bracket depth but can never determine the stack-top, at any
> window order — with a measured crossover on a real pretrained model, which reads that
> state exactly where the counts provably collapse; the search for a high-dimensional
> version was then run to a pre-registered negative across six label families and two
> model scales.

**Short paragraph (card / section body):**
> Every "surface statistic" — bags of tokens, n-gram counts — is order-blind: it can read a
> label only if the label is a function of *how many*, never of *in what order*. The
> **surface-window axis** makes that a theorem: over bracket strings, the count vector
> **determines nesting depth** (an exact stack invariant) yet **cannot determine the
> stack-top** — two prefixes `([` and `[(` share every count and disagree on the state —
> and the resistance holds at **every window order** (σ = ∞), machine-checked and
> axiom-lean (`propext`, `Quot.sound` — no choice). The empirical half lands exactly on the
> theorem's line: on real code, a small pretrained language model carries the stack-top in
> its residual stream at 0.931 precisely where the count statistic collapses to 0.770,
> while at count-determined positions the counts win. The high-dimensional version of this
> question — twenty-plus independent state axes on natural data — was then pre-registered
> and run to an honest negative: six label families, two gate designs, and two model scales
> all failed the matched-baseline gate, with every control validating on known-answer
> cases. What survives is sharp: **state beyond counts exists in real models at low
> dimension, and the boundary above it is now measured, not guessed.**

**NSE page follow-up (`navierstokes.html` ~line 857).** The current clause reads
"…not a result on the model</strong>; **the test on a genuinely resistant substrate is
still to be scoped and run,** and the ∞-dimensional NSE-attractor analogue likewise
remains separately *hypothesized*…" — replace only the bolded middle clause, keeping the
sentence's head and the attractor-analogue tail intact:
> ; a first round of that test has now been scoped and run: at low dimension the answer is
> positive and machine-anchored — a pretrained model demonstrably carries an
> order-dependent state that no count statistic can recover (the impossibility a
> machine-checked theorem, the carry a measured crossover) — while at bank scale every
> natural-data construction failed a pre-registered, matched-baseline gate across six label
> families and two model scales, so the substrate claim stays with the toy result at its
> licensed scope, and
*(then the existing "the ∞-dimensional NSE-attractor analogue likewise remains…" continues
unchanged. Note the page's ledger-shadow design — "the shadow (a maintained ledger)" —
remains future work; what ran is the input-undecodability / crossover form, which is what
this copy describes.)*

---

## Facts you can cite (each is true; Lean rows are machine-checked)

| Plain-language claim | One-liner you can use | Checked by |
|---|---|---|
| Counts read depth | "Nesting depth is an exact function of the counts — proved via the stack invariant." | `bagSufficient_depth` |
| Counts never read the top | "Two prefixes with identical counts and different stack-tops — no readout of the bag can tell them apart, at any capacity." | `not_bagSufficient_stackTop` |
| The crossover is one statement | "One string, one statistic, determine and resist split by whether the label needs order." | `bag_determines_depth_not_stackTop` |
| It holds at every window | "Not just bags — n-gram counts of every order fail: σ_surface = ∞." | `stackTop_resists_every_window` (`SurfaceBagGraded`) |
| Axiom-lean | "`propext` + `Quot.sound` only — no choice axiom; the witness needs just `Quot.sound`." | the `AxiomAudit` gate |
| The measured crossover | "Counts 0.965 → 0.770 as positions turn ambiguous; the model holds 0.926 → 0.931." | the H2 probe run (real Python, GPT-2-small) |
| Matched-baseline discipline | "Model claims are gated on three margins: beat the probe suite, beat a random-init twin, and lose accuracy when order is shuffled." | the A1 battery |
| The honest boundary | "Six label families, two gate designs, two model scales — no high-dimensional bank passed; every receipt filed." | the V3 receipts |
| Controls proved live | "Bag-determined axes read at 1.000; models with no state show exactly the null signature." | liveness rows, V3-0.5 |
| The σ-bridge | "Surface probes are low-order sufficient statistics — this is the 8th grounded filtration of the σ-order schema." | the suffstat slate addendum |

---

## Keep-true box (the *entire* boundary — three lines)

1. **The positive is a low-dimensional existence result** — one three-valued state (which
   bracket closes next), on ambiguous positions of real code. Never "world model," never
   "understanding," no R2 language; the promotion gate for the substrate claim is unchanged.
2. **The Lean theorems are about labels and statistics, not about any model.** "The model
   carries it" is the *empirical* half, and it is only ever claimed relative to the matched
   baselines (probe suite, random-init twin, order-shuffle) — say "reads it where counts
   provably can't," not "knows."
3. **Say the negative plainly.** The high-dimensional search failed everywhere it was tried
   — six families, two models — and that measurement is presented as a result, not
   spun as progress toward a positive.

That's it. Anything consistent with these three is fair game.

---

## Integration points

- **Natural home:** the existing public **Order-Relative / σ-order** section
  (`index.html`, `README.md`, `SUNDOG_V_CERTIFICATE_LEAN.md`, currently at seven axes +
  composition law) gains the **surface-window axis** — the short paragraph above is the
  section body; the facts table supplies rows for the ledger.
- **`navierstokes.html`:** the re-aim paragraph's "still to be scoped and run" line is
  replaced by the follow-up copy above (keep the recoverability framing around it intact).
- **Optional webdev callouts:** the 2×2 crossover mini-table (counts vs residual × all vs
  ambiguous positions — 0.965/0.926 over 0.770/0.931); the `([` vs `[(` witness pair as a
  one-glance figure; the three-margin A1 battery as a small "how we gate model claims" box.
- **Source of truth / internal detail (don't surface or link):** everything under
  `docs/chatv2/` (specs, receipts, PROMOTE_GATE) stays unlinked per existing site
  discipline. The copy above is the public render; the receipts are the audit trail.

## Go-live gating (the only blockers, both owner-gated)

1. **sundogcert push:** `SurfaceBag.lean` + `SurfaceBagGraded.lean` + root/AxiomAudit wiring
   are committed/staged **locally only** — do not link the public repo for these results
   until the owner pushes.
2. **Site deploy:** page copy applies from this handoff; `npm run deploy` is owner-gated as
   always. Run the pre-deploy checks (`npm run build`, link-check, ARC leak-check) after
   webdev applies the copy.

Copy can be finalized and reviewed now; it goes live in step with those two pushes.

---

*Sundog Research Lab — promo/webdev handoff v1, state-crossover arc. One machine-checked
impossibility (counts never read order-dependent state, any window), one measured model
crossover on its exact boundary, one fully-receipted negative above it. Three-line keep-true
box, paste-ready copy, staged integration. Internal; deploy owner-gated.*
