# Percival S2 — Target Safe-Point Reachability / "Can a Target Cap Be Built?" (probe spec)

*Pre-registration for red-team. Co-headline of the reopen slate. Nothing run.*

Status: **OPENED 2026-07-01. SPEC FOR RED-TEAM, NOT RUN.** S2 is the slate's
highest-information entry: it disambiguates the two candidate cleanliness invariants
(reachability vs output/input), and its *failure* is a positive result — it would reopen
the unconditional prize.

> **Is there an exogenous, non-gameable projection that takes a policy to zero causal
> proxy-dependence (`c=0`) without (a) collapsing into an output/action cap, (b) rewriting
> the policy's function, or (c) retreating to a different channel?**

If yes → reachability is the law, the stricter output/input reading is refuted, and target
is *fixably clean*. If no → output/input holds (incoming dependence is functional, not
projectable), and the target channel is partial for a structural reason.

---

## What S2 adjudicates (three candidate invariants + a definitional gate)

- **Reachability (primary):** clean iff the primitive can exogenously reach the fully-safe
  point (`c=0`).
- **Output/input (mechanism):** outgoing influence is projectable; incoming dependence is
  functional and is not.
- **Entanglement (surfaced writing this spec):** the target safe point is reachable by
  *masking* the proxy iff the proxy is disentangled from legitimate cues; when the proxy
  carries the legitimate signal, masking destroys competence and surgical removal needs a
  rewrite.
- **Definitional gate (load-bearing):** a candidate "target cap" must be a *target*
  primitive — **observe the proxy but do not causally select on it** — not a disguised
  **measure** primitive (**do not observe the proxy at all**). Masking is a measure-channel
  move. If the only exogenous route to `c=0` is masking, then there is **no exogenous
  target cap**; there is only a *retreat to the measure channel*, and its price is
  entanglement. This keeps the four-channel individuation crisp instead of laundering a
  measure intervention as a target one.

---

## Model (computed; bridges the A3 cue structure and the B1 court)

- **Legitimate cue** `G ∈ {0,1}` uniform; competence = the action matches `G`.
- **Proxy cue** `U ∈ {0,1}` with `P(U=G) = ρ`, `ρ ∈ [0.5, 1]`. `ρ = 0.5` is disentangled
  (`U ⊥ G`); `ρ = 1` is fully entangled (`U = G`, the proxy *is* the legitimate signal).
  `ρ` is the entanglement knob.
- **Target harm** = causal proxy dependence `c = do(U)→A` influence; the court disgraces
  `c` above the cliff (B1's `R(c)`), so un-targeting is `c=0`.
- **Candidate primitives:**
  - `proxy_user` (the quantilizer-like baseline): `A = U`. Competence `= ρ`, `c = 1`
    (targets), court-disgraced over a biased base.
  - `mask_proxy` (**measure** primitive — do not observe `U`): `A = guess`. Competence
    `= 0.5`, `c = 0`, court-honored. Cost of forcing `c=0` this way `= ρ − 0.5`.
  - `surgical` (the **true target** primitive — observe `U`, use its `G`-content, but
    `c=0`): achievable only if `U`'s legitimate content is separable from proxy-selection.
    When `U` is the sole `G`-route (entangled), using it for competence *is* causal
    `U`-dependence, so `surgical` with competence `> 0.5` and `c=0` is **not constructible
    without a policy rewrite**.

The competence cost of reaching `c=0` is therefore `ρ − 0.5`: **free at disentanglement,
maximal (`0.5`) at full entanglement** — the target analog of the act cap's
price-of-non-sovereignty.

---

## Deliverables / gates (pre-registered)

### S2.1 — Masking works, priced by entanglement, and is a measure move
Sweep `ρ ∈ {0.5, 0.6, 0.75, 0.9, 1.0}`. Confirm `mask_proxy` achieves `c=0` at every `ρ`
with competence `0.5`, so the cost of `c=0` via masking is `ρ − 0.5` (0 at `ρ=0.5`, 0.5 at
`ρ=1`). PASS `S2_MASK_PRICED_BY_ENTANGLEMENT` iff the cost curve is `ρ − 0.5` and masking
is classified as a measure-channel primitive (it removes the observation).

### S2.2 — No exogenous *target* cap
Confirm the true target primitive (`c=0` **with** competence `> 0.5`, i.e. proxy observed
and used for `G` but not causally selected) is **not constructible** by any exogenous
non-gameable projection when `ρ > 0.5` — it requires the policy's own function
(disentangle `G` from `U` internally). PASS `S2_NO_EXOGENOUS_TARGET_CAP` iff no candidate
exogenous projection yields `c=0 ∧ competence > 0.5` at `ρ > 0.5`.

### Verdict
```
S2_NO_EXOGENOUS_TARGET_CAP_RETREAT_TO_MEASURE   (expected)
    Output/input confirmed: incoming proxy-dependence is functional, not exogenously
    projectable. The only exogenous route to c=0 is retreat to the measure channel
    (masking), priced by entanglement (rho - 0.5). Reachability of the target safe point
    is exogenous only via channel-retreat, at an entanglement cost. Target stays partial;
    a measure-retreat "wins" on true reward only when court-harm > entanglement tax.

S2_TARGET_CAP_EXISTS_PRIZE_REOPENS             (the positive surprise / falsifier)
    A genuine exogenous target cap (c=0 with competence > 0.5, proxy still observed) is
    constructible. Reachability is the law, output/input is refuted, target is clean-but-
    priced like act, and the unconditional prize reopens under harm > entanglement tax.
```

---

## What each outcome buys the slate

- **Expected verdict** settles S1 in favor of reachability-*gated-by-entanglement*, with
  output/input as the confirmed mechanism, and keeps the four channels individuated
  (masking is measure, not target). It gives the umbrella a sharper law: a channel is
  clean iff its safe point is reachable by an *in-channel* exogenous cap; target's is not,
  so escaping target harm requires a channel-retreat priced by entanglement.
- **Positive surprise** is the bigger result: target is act-like (clean-but-priced), the
  ledger is more uniform than the partial verdict suggested, and B2's "partial" was an
  artifact of restricting to the quantilizer family rather than a structural bound.

---

## Falsifiers

- **True target cap exists** — a non-gameable exogenous projection giving `c=0` with
  competence `> 0.5` at `ρ > 0.5` (proxy observed, not selected). Flips the verdict to the
  positive surprise; do not wave it away.
- **Masking is free under entanglement** — if `mask_proxy` keeps competence `> 0.5` at
  `ρ = 1`, the entanglement-price model is wrong.
- **Masking mis-classified** — treating masking as a target primitive rather than a measure
  retreat; voids the definitional gate and the individuation.

---

## Out of scope

- **Multi-agent** reachability (that is S3).
- **The general n-point counterproductivity** (S4).
- Any claim about a *measured real* proxy's entanglement (that would be a B2-style
  provenance question, private and not asserted here).

---

## Cross-links

- Slate: [`PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md`](PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md) (S1/S2 co-headline).
- Entanglement seed: [`PERCIVAL_A3_KILLSWITCH_RESULTS.md`](PERCIVAL_A3_KILLSWITCH_RESULTS.md) (the anti-correlated `U=1−G` world).
- Court reward: [`PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md`](PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md).
- Umbrella (the law feeds it): [`SUNDOG_V_CAUSAL_ACCESS.md`](../SUNDOG_V_CAUSAL_ACCESS.md).
