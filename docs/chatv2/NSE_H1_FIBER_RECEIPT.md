# NSE-H1 Rung 1 — Fiber Transfer Receipt (final H1 verdict)

> 2026-07-06. Owner-cleared fire of spec §4 on the v1.1 exports (both cells
> `H1_CELL_ADMITTED`, unfenced benefit selector). Pure post-processing; artifacts
> `results/proof/nse-h1-g{200,300}-v11/h1_fiber.json`. **Non-promotional.** All
> criteria were frozen in `NSE_H1_JSELECTOR_SPEC.md` §4–5 before any fiber number
> existed; no criterion was touched after the reads.

## Reads (paired-fiber constancy at the banked matched radii)

| cell | ε_K | unique pairs (≥100) | a_J disagree | y_pi disagree (same stream) | banked `pi_hat` | shuffle floor (≥0.25) | criterion (≤0.05 ∧ ≤2×y_pi) | transfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G=200 | 0.060598 | 2,028 | **0.1534** | 0.0429 | 0.0367 | 0.419 | 0.05 ∧ 0.0858 | **FAIL** |
| G=300 | 0.066422 | 2,578 | **0.0586** | 0.0043 | 0.0382 | 0.388 | 0.05 ∧ 0.0086 | **FAIL** |

- The reads are powered and live: pair gates cleared without option A; shuffle
  floors well above 0.25; and the same-stream `y_pi` comparator **reproduces the
  banked fiber constancy on a fresh seed-0 stream** (0.0429 vs banked 0.0367 at
  G=200) — the apparatus finds the known structure where it exists.
- kNN mirror: fidelity coverage 0.000 on both cells — at n = 1,600 the r₃₀ radius
  exceeds ε_K everywhere (the sample is 31× sparser than the banked 50k). Reported
  NaN, not gated; the paired-fiber read is the registered primary.
- **Criterion-robustness:** the verdict does not depend on the frozen 0.05 bound.
  G=200 (0.1534) fails even the banked protocol's own δ_action = 0.10 constancy
  bound outright; G=300 (0.0586) would sit inside that absolute bound but fails
  the matched comparator by ~7× (a_J is 13.6× less fiber-constant than `y_pi` on
  the identical pairs). Under any registered reading, transfer fails.

## Verdict (per the frozen table): `NSE-H1-PROXY-ONLY`

Any admitted cell failing a powered transfer read ⇒ `NSE-H1-PROXY-ONLY`. Both did.

## What this says

- **The two-regime C1 witness is a proxy-relative fact** — the slate's own
  pre-registered framing for this branch. `Phi_K3` fiber-determines the
  threshold-forecast label (`y_pi` constancy 0.043 / 0.004, matching the banked
  certificates) but does **not** fiber-determine action value: the benefit
  selector varies within fibers 3.6× / 13.6× more than the matched comparator at
  the same radii on the same instants.
- Interpretive lead (typed, not claimed): `Delta_J` is a difference of two
  rollouts through the actuation window — the high-mode configuration that seeds
  the post-actuation relaxation is decision-relevant for *value* even where it is
  irrelevant for *threshold forecast*. The modes the signature discards matter
  for how much damping helps, not for whether the excursion comes.

## What this does not say

No infinite-dimensional NSE statement. No demotion of the banked witness — the
`pi_hat` certificates stand exactly as receipted, now with sharper typing of what
they are about. No claim beyond the one registered actuator family and horizon
(μ_act = 1.0, τ_act = 100, τ = 500); other action families are unregistered
territory, not implied negatives.

## Ledger effects

1. **H1 closes at `NSE-H1-PROXY-ONLY`** — an informative negative, first
   adjudicated entry of the post-AT slate. No rescue rounds exist: the reads were
   powered, live, and criterion-robust.
2. The v1 G=200 diagnostic option (least-harm fence) **expires moot** — it could
   only have escalated to the verdict now reached directly.
3. **H2 consequence (slate §4, pre-registered):** H1 did not land ⇒ H2 uses the
   v7 portable selector, not an H1 selector.
4. Carried mandate for any future H1-style registration: fiber-transfer claims
   need the matched same-stream comparator (`y_pi` here) — absolute constancy
   bounds alone would have mistyped G=300.

Cross-refs: `NSE_H1_JSELECTOR_SPEC.md` (§4–5, §8), `NSE_H1_ADMISSION_RECEIPT.md`
(v1 + v1.1 rungs, Findings 1–4), `NSE_POST_AT_HYPOTHESES_SLATE.md` §3 (branch
language) + §2.1 ("if H1 fails, the witness remains a proxy-relative fact"),
`results/proof/c1-paired-fiber-g{200,300}/manifest.json` (banked radii and
constancy values), `NSE_STATIONARITY_GATE_CHECKLIST.md` (imported gates).
